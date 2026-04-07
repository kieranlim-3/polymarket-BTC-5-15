"""
src/polymarket_client.py
────────────────────────
Async wrapper around the Polymarket CLOB REST API.

Responsibilities
────────────────
- Discover active BTC/ETH price-prediction markets
- Filter by duration (5-min, 15-min) and liquidity (≥ $50 k)
- Fetch order book + compute mid/best prices
- Place YES/NO limit/market orders (live) or return mock receipt (paper)
- Maintain portfolio USDC balance from on-chain or API source

Authentication
──────────────
Polymarket uses EIP-712 signed headers (L1) or API-key + HMAC (L2).
py-clob-client handles both; we use it for auth and fall back to raw
aiohttp for async market-data calls to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType, BUY, SELL
    from py_clob_client.constants import POLYGON
    HAS_CLOB = True
except ImportError:
    HAS_CLOB = False

from src.config import Settings
from src.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MarketInfo:
    condition_id: str
    question: str          # full question text
    asset: str             # "BTC" or "ETH"
    threshold: float       # price threshold in the question
    direction: str         # "above" or "below"
    expiry_ts: float       # unix timestamp of resolution
    duration_minutes: int  # 5 or 15
    yes_token_id: str
    no_token_id: str
    yes_price: float       # current best YES ask (0-1)
    no_price: float        # current best NO ask (0-1)
    total_volume_usd: float
    liquidity_usd: float


@dataclass
class OrderReceipt:
    order_id: str
    market_id: str
    side: str              # "YES" or "NO"
    size_usdc: float
    price: float
    filled: bool
    timestamp: float
    paper: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Market question parser
# ─────────────────────────────────────────────────────────────────────────────

_PRICE_PATTERN = re.compile(
    r"(BTC|ETH|Bitcoin|Ethereum)[^\d]*"  # asset
    r"(above|below|over|under|higher than|lower than)\s+"  # direction
    r"\$?([\d,]+(?:\.\d+)?)",  # threshold
    re.IGNORECASE,
)

_DURATION_PATTERNS = [
    (re.compile(r"\b5[\s-]?min", re.IGNORECASE), 5),
    (re.compile(r"\b15[\s-]?min", re.IGNORECASE), 15),
]


def parse_market_question(
    question: str,
    end_date_iso: str,
    start_date_iso: str = "",
) -> Optional[tuple[str, float, str, int]]:
    """
    Returns (asset, threshold, direction, duration_minutes) or None if unrecognised.
    """
    m = _PRICE_PATTERN.search(question)
    if not m:
        return None

    raw_asset = m.group(1).upper()
    asset = "BTC" if raw_asset in ("BTC", "BITCOIN") else "ETH"
    direction_raw = m.group(2).lower()
    direction = "above" if any(w in direction_raw for w in ("above", "over", "higher")) else "below"
    threshold = float(m.group(3).replace(",", ""))

    # Infer duration from question text first, then from start/end delta
    duration = None
    for pattern, mins in _DURATION_PATTERNS:
        if pattern.search(question):
            duration = mins
            break

    if duration is None and end_date_iso and start_date_iso:
        try:
            from dateutil.parser import parse as dp
            delta = (dp(end_date_iso) - dp(start_date_iso)).total_seconds() / 60
            if 3 <= delta <= 7:
                duration = 5
            elif 12 <= delta <= 18:
                duration = 15
        except Exception:
            pass

    if duration is None:
        return None

    return asset, threshold, direction, duration


# ─────────────────────────────────────────────────────────────────────────────
# PolymarketClient
# ─────────────────────────────────────────────────────────────────────────────

class PolymarketClient:

    def __init__(self, settings: Settings, paper_mode: bool = True) -> None:
        self._s = settings
        self._paper = paper_mode
        self._session: Optional[aiohttp.ClientSession] = None
        self._clob: Optional[object] = None  # ClobClient (sync, wrapped in executor)
        self._portfolio_usdc: float = 10_000.0  # bootstrap; updated from chain

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"User-Agent": "polymarket-arb-bot/1.0"},
        )
        if HAS_CLOB and not self._paper:
            await self._init_clob_client()
        log.info("polymarket_client.started", paper=self._paper)

    async def stop(self) -> None:
        if self._session:
            await self._session.close()

    async def _init_clob_client(self) -> None:
        """Initialise the synchronous py-clob-client in a thread pool."""
        loop = asyncio.get_event_loop()
        def _build():
            client = ClobClient(
                host=self._s.poly_host,
                chain_id=POLYGON,
                key=self._s.poly_private_key,
                signature_type=2,  # POLY_GNOSIS_SAFE
                funder=None,
            )
            # Set L2 credentials if provided
            if self._s.poly_api_key and self._s.poly_api_secret:
                client.set_api_creds(client.create_or_derive_api_creds())
            return client
        self._clob = await loop.run_in_executor(None, _build)
        log.info("polymarket_client.clob_initialised")

    # ── Portfolio ──────────────────────────────────────────────────────────

    async def get_portfolio_usdc(self) -> float:
        """
        Return current USDC balance.
        Live: query Polygon via web3.
        Paper: return cached simulation balance.
        """
        if self._paper:
            return self._portfolio_usdc

        try:
            from web3 import Web3
            w3 = Web3(Web3.HTTPProvider(self._s.alchemy_rpc_url))
            # USDC on Polygon: 0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174
            USDC_ADDRESS = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
            USDC_ABI = [
                {"inputs": [{"name": "account", "type": "address"}],
                 "name": "balanceOf",
                 "outputs": [{"name": "", "type": "uint256"}],
                 "type": "function"}
            ]
            contract = w3.eth.contract(address=USDC_ADDRESS, abi=USDC_ABI)
            wallet = Web3.to_checksum_address(
                # Derive address from private key
                w3.eth.account.from_key(self._s.poly_private_key).address
            )
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(None, contract.functions.balanceOf(wallet).call)
            balance = raw / 1e6  # USDC has 6 decimals
            self._portfolio_usdc = balance
            return balance
        except Exception as exc:
            log.error("polymarket_client.balance_error", exc=str(exc))
            return self._portfolio_usdc  # fall back to cached

    def update_paper_balance(self, delta: float) -> None:
        """Adjust simulated balance by delta (positive = profit)."""
        self._portfolio_usdc = max(0.0, self._portfolio_usdc + delta)

    # ── Market Discovery ───────────────────────────────────────────────────

    async def fetch_active_markets(
        self,
        target_assets: tuple[str, ...],
        target_durations: tuple[int, ...],
        min_liquidity: float,
        max_markets: int,
    ) -> list[MarketInfo]:
        """
        Fetch and filter open BTC/ETH price markets from Polymarket CLOB.
        Returns up to max_markets MarketInfo objects.
        """
        markets: list[MarketInfo] = []

        # Paginate through active markets
        cursor = ""
        while len(markets) < max_markets:
            params: dict = {"active": "true", "closed": "false", "limit": 100}
            if cursor:
                params["next_cursor"] = cursor

            try:
                async with self._session.get(
                    f"{self._s.poly_host}/markets",
                    params=params,
                ) as resp:
                    resp.raise_for_status()
                    payload = await resp.json(content_type=None)
            except Exception as exc:
                log.error("polymarket_client.fetch_markets_error", exc=str(exc))
                break

            raw_markets = payload.get("data", [])
            cursor = payload.get("next_cursor", "")

            for raw in raw_markets:
                if len(markets) >= max_markets:
                    break
                mi = await self._parse_raw_market(
                    raw, target_assets, target_durations, min_liquidity
                )
                if mi:
                    markets.append(mi)

            if not cursor or not raw_markets:
                break

        log.info("polymarket_client.markets_loaded", count=len(markets))
        return markets

    async def _parse_raw_market(
        self,
        raw: dict,
        target_assets: tuple[str, ...],
        target_durations: tuple[int, ...],
        min_liquidity: float,
    ) -> Optional[MarketInfo]:
        try:
            question = raw.get("question", "")
            condition_id = raw.get("condition_id", "")
            end_date = raw.get("end_date_iso", "")
            start_date = raw.get("start_date_iso", "")

            parsed = parse_market_question(question, end_date, start_date)
            if not parsed:
                return None

            asset, threshold, direction, duration = parsed
            if asset not in target_assets or duration not in target_durations:
                return None

            # Liquidity check
            volume = float(raw.get("volume", 0) or 0)
            liquidity = float(raw.get("liquidity", volume) or 0)
            if liquidity < min_liquidity:
                return None

            # Parse expiry
            try:
                from dateutil.parser import parse as dp
                expiry_ts = dp(end_date).timestamp()
            except Exception:
                expiry_ts = time.time() + duration * 60

            # Tokens
            tokens = raw.get("tokens", [])
            yes_token = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), {})
            no_token = next((t for t in tokens if t.get("outcome", "").upper() == "NO"), {})

            # Fetch current prices from order book
            yes_price, no_price = await self._fetch_mid_prices(
                yes_token.get("token_id", ""),
                no_token.get("token_id", ""),
            )

            return MarketInfo(
                condition_id=condition_id,
                question=question,
                asset=asset,
                threshold=threshold,
                direction=direction,
                expiry_ts=expiry_ts,
                duration_minutes=duration,
                yes_token_id=yes_token.get("token_id", ""),
                no_token_id=no_token.get("token_id", ""),
                yes_price=yes_price,
                no_price=no_price,
                total_volume_usd=volume,
                liquidity_usd=liquidity,
            )
        except Exception as exc:
            log.warning("polymarket_client.parse_market_error", exc=str(exc))
            return None

    async def _fetch_mid_prices(
        self, yes_token_id: str, no_token_id: str
    ) -> tuple[float, float]:
        """Return (yes_mid, no_mid) from order book. Returns (0.5, 0.5) on failure."""
        if not yes_token_id:
            return 0.5, 0.5
        try:
            async with self._session.get(
                f"{self._s.poly_host}/book",
                params={"token_id": yes_token_id},
            ) as resp:
                resp.raise_for_status()
                book = await resp.json(content_type=None)

            bids = book.get("bids", [])
            asks = book.get("asks", [])
            best_bid = float(bids[0]["price"]) if bids else 0.0
            best_ask = float(asks[0]["price"]) if asks else 1.0
            yes_mid = (best_bid + best_ask) / 2.0
            no_mid = 1.0 - yes_mid
            return round(yes_mid, 4), round(no_mid, 4)
        except Exception as exc:
            log.warning("polymarket_client.book_error", exc=str(exc))
            return 0.5, 0.5

    # ── Order Placement ────────────────────────────────────────────────────

    async def place_order(
        self,
        market: MarketInfo,
        side: str,        # "YES" or "NO"
        size_usdc: float,
        price: float,
    ) -> OrderReceipt:
        """
        Place a market/limit order.
        Paper mode: returns a simulated receipt instantly.
        Live mode: submits via py-clob-client in a thread executor.
        """
        if self._paper:
            return self._paper_order(market, side, size_usdc, price)

        return await self._live_order(market, side, size_usdc, price)

    def _paper_order(
        self, market: MarketInfo, side: str, size_usdc: float, price: float
    ) -> OrderReceipt:
        import uuid
        receipt = OrderReceipt(
            order_id=f"PAPER-{uuid.uuid4().hex[:8]}",
            market_id=market.condition_id,
            side=side,
            size_usdc=size_usdc,
            price=price,
            filled=True,
            timestamp=time.time(),
            paper=True,
        )
        log.info(
            "paper_order.placed",
            order_id=receipt.order_id,
            market=market.question[:60],
            side=side,
            size=size_usdc,
            price=price,
        )
        return receipt

    async def _live_order(
        self, market: MarketInfo, side: str, size_usdc: float, price: float
    ) -> OrderReceipt:
        if not HAS_CLOB or self._clob is None:
            raise RuntimeError("ClobClient not initialised for live trading")

        token_id = market.yes_token_id if side == "YES" else market.no_token_id
        loop = asyncio.get_event_loop()

        def _submit():
            args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size_usdc / price,  # convert USDC → shares
                side=BUY,
            )
            return self._clob.create_and_post_order(args)

        try:
            resp = await asyncio.wait_for(
                loop.run_in_executor(None, _submit),
                timeout=5.0,
            )
            order_id = resp.get("orderID", "unknown")
            log.info("live_order.placed", order_id=order_id, side=side, size=size_usdc)
            return OrderReceipt(
                order_id=order_id,
                market_id=market.condition_id,
                side=side,
                size_usdc=size_usdc,
                price=price,
                filled=resp.get("status") == "FILLED",
                timestamp=time.time(),
                paper=False,
            )
        except asyncio.TimeoutError:
            raise RuntimeError("Order placement timed out (>5 s)")
