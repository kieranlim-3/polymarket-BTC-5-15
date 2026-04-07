"""
src/price_feed.py
─────────────────
Connects to Binance combined WebSocket stream for BTC/USDT and ETH/USDT.

Features
────────
- Auto-reconnect with exponential backoff (max 10 retries)
- Marks prices as stale if last update > 10 seconds ago
- Thread-safe price cache (asyncio.Lock)
- Publishes price events to all registered subscriber callbacks
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Optional

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from src.logger import get_logger

log = get_logger(__name__)

PriceCallback = Callable[["PriceTick"], Awaitable[None]]


@dataclass
class PriceTick:
    symbol: str        # "BTC" or "ETH"
    price: float
    timestamp: float   # unix seconds (epoch)
    volume_24h: float  # base asset volume


@dataclass
class PriceFeed:
    ws_url: str
    streams: tuple[str, ...]
    stale_seconds: int = 10
    max_retries: int = 10
    backoff_base: float = 1.5

    # internal state
    _prices: dict[str, PriceTick] = field(default_factory=dict, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _callbacks: list[PriceCallback] = field(default_factory=list, init=False)
    _running: bool = field(default=False, init=False)
    _task: Optional[asyncio.Task] = field(default=None, init=False)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run_forever(), name="price-feed")
        log.info("price_feed.started", streams=self.streams)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("price_feed.stopped")

    # ── Subscription ───────────────────────────────────────────────────────

    def subscribe(self, callback: PriceCallback) -> None:
        """Register an async callback; called on every price update."""
        self._callbacks.append(callback)

    # ── Public getters ─────────────────────────────────────────────────────

    async def get_price(self, symbol: str) -> Optional[float]:
        """Return latest price for symbol, or None if stale/missing."""
        async with self._lock:
            tick = self._prices.get(symbol)
            if tick is None:
                return None
            if time.time() - tick.timestamp > self.stale_seconds:
                return None
            return tick.price

    async def is_stale(self) -> bool:
        """True if ANY tracked symbol has stale data."""
        now = time.time()
        async with self._lock:
            for tick in self._prices.values():
                if now - tick.timestamp > self.stale_seconds:
                    return True
            # Also stale if we haven't received data yet
            if not self._prices:
                return True
        return False

    async def get_all_prices(self) -> dict[str, float]:
        """Return {symbol: price} for all non-stale symbols."""
        now = time.time()
        result: dict[str, float] = {}
        async with self._lock:
            for sym, tick in self._prices.items():
                if now - tick.timestamp <= self.stale_seconds:
                    result[sym] = tick.price
        return result

    # ── WebSocket internals ────────────────────────────────────────────────

    def _build_url(self) -> str:
        combined = "/".join(self.streams)
        return f"{self.ws_url}?streams={combined}"

    async def _run_forever(self) -> None:
        retries = 0
        while self._running:
            try:
                await self._connect_and_consume()
                retries = 0  # successful connection resets backoff
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                retries += 1
                if retries > self.max_retries:
                    log.error(
                        "price_feed.max_retries_exceeded",
                        retries=retries,
                        exc=str(exc),
                    )
                    break
                wait = self.backoff_base ** retries
                log.warning(
                    "price_feed.reconnecting",
                    attempt=retries,
                    wait_seconds=round(wait, 1),
                    exc=str(exc),
                )
                await asyncio.sleep(wait)

    async def _connect_and_consume(self) -> None:
        url = self._build_url()
        log.info("price_feed.connecting", url=url)
        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
            open_timeout=10,
        ) as ws:
            log.info("price_feed.connected")
            async for raw in ws:
                if not self._running:
                    break
                await self._handle_message(raw)

    async def _handle_message(self, raw: str) -> None:
        try:
            envelope = json.loads(raw)
            data = envelope.get("data", envelope)  # combined stream wraps in {"data": ...}
            stream = envelope.get("stream", "")

            # Binance 24hr ticker payload key reference:
            # "s" = symbol, "c" = last price, "E" = event time (ms), "v" = volume
            symbol_raw = data.get("s", "")  # e.g. "BTCUSDT"
            price_str = data.get("c", "")   # last price
            event_time_ms = data.get("E", 0)
            volume_str = data.get("v", "0")

            if not symbol_raw or not price_str:
                return

            symbol = _normalise_symbol(symbol_raw)  # "BTC" or "ETH"
            if symbol not in ("BTC", "ETH"):
                return

            tick = PriceTick(
                symbol=symbol,
                price=float(price_str),
                timestamp=event_time_ms / 1000.0 if event_time_ms else time.time(),
                volume_24h=float(volume_str),
            )

            async with self._lock:
                self._prices[symbol] = tick

            log.debug("price_feed.tick", symbol=symbol, price=tick.price)

            # Notify subscribers concurrently (errors isolated)
            if self._callbacks:
                await asyncio.gather(
                    *[cb(tick) for cb in self._callbacks],
                    return_exceptions=True,
                )

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            log.warning("price_feed.parse_error", exc=str(exc), raw=raw[:200])


# ── Helpers ────────────────────────────────────────────────────────────────

def _normalise_symbol(raw: str) -> str:
    """'BTCUSDT' → 'BTC', 'ETHUSDT' → 'ETH'."""
    for quote in ("USDT", "BUSD", "USD"):
        if raw.upper().endswith(quote):
            return raw.upper()[: -len(quote)]
    return raw.upper()
