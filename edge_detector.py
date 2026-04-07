"""
src/edge_detector.py
────────────────────
Detects latency-arbitrage opportunities between Binance spot prices
and Polymarket binary prediction markets.

Probability Model
─────────────────
For a binary contract "Will ASSET be ABOVE/BELOW $T at time τ?":

  • Assume log-normal returns: ln(P_τ / P_0) ~ N(μ·Δt, σ²·Δt)
  • Annualised vol σ is estimated from a rolling window of price ticks
  • μ is set to 0 (risk-neutral / short horizon)
  • P(P_τ > T) = Φ((ln(P_0/T)) / (σ·√Δt))   for "above"
  • P(P_τ < T) = 1 − above                    for "below"

Edge Definition
───────────────
  edge = |our_prob − market_implied_prob|

  • edge < min_detectable_edge  → ignore (noise)
  • edge ≥ min_exec_edge        → trade in our-favoured direction

The module is fully stateless except for the rolling vol estimator.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import norm

from src.polymarket_client import MarketInfo
from src.logger import get_logger

log = get_logger(__name__)

# Default annualised volatility if we have insufficient data
_FALLBACK_VOL_ANNUAL = 0.80   # 80 % — conservative for BTC/ETH


@dataclass
class Opportunity:
    market: MarketInfo
    our_prob: float            # our estimated probability of YES outcome
    market_prob: float         # Polymarket's implied YES probability
    edge: float                # our_prob − market_prob (signed)
    abs_edge: float            # |edge|
    recommended_side: str      # "YES" or "NO"
    recommended_price: float   # best available price for that side
    spot_price: float          # Binance price at detection time
    detected_at: float         # unix timestamp


class EdgeDetector:

    def __init__(
        self,
        min_detectable_edge: float,   # e.g. 0.05 = 5 %
        min_exec_edge: float,         # e.g. 0.08 = 8 %
        vol_window_seconds: int = 300,
    ) -> None:
        self._min_detect = min_detectable_edge
        self._min_exec = min_exec_edge
        self._vol_window = vol_window_seconds

        # Rolling price history for vol estimation: {symbol: deque[(ts, price)]}
        self._price_history: dict[str, deque] = {
            "BTC": deque(maxlen=500),
            "ETH": deque(maxlen=500),
        }

    # ── Public API ─────────────────────────────────────────────────────────

    def update_price(self, symbol: str, price: float, timestamp: float) -> None:
        """Feed each new price tick here to maintain the vol estimator."""
        if symbol in self._price_history:
            self._price_history[symbol].append((timestamp, price))

    def scan(
        self,
        markets: list[MarketInfo],
        spot_prices: dict[str, float],  # {"BTC": 65000.0, "ETH": 3200.0}
    ) -> list[Opportunity]:
        """
        Evaluate all markets and return sorted list of Opportunity objects
        with abs_edge ≥ min_detectable_edge.
        """
        now = time.time()
        opportunities: list[Opportunity] = []

        for market in markets:
            spot = spot_prices.get(market.asset)
            if spot is None:
                continue

            # Skip expired markets
            time_to_expiry = market.expiry_ts - now
            if time_to_expiry <= 0:
                continue

            opp = self._evaluate(market, spot, time_to_expiry, now)
            if opp and opp.abs_edge >= self._min_detect:
                opportunities.append(opp)

        # Sort by edge descending
        opportunities.sort(key=lambda o: o.abs_edge, reverse=True)
        return opportunities

    def find_executable(self, opportunities: list[Opportunity]) -> list[Opportunity]:
        """Filter opportunities that meet the minimum execution edge."""
        return [o for o in opportunities if o.abs_edge >= self._min_exec]

    # ── Core Evaluation ────────────────────────────────────────────────────

    def _evaluate(
        self,
        market: MarketInfo,
        spot: float,
        time_to_expiry_seconds: float,
        now: float,
    ) -> Optional[Opportunity]:
        try:
            our_prob = self._estimate_prob(
                asset=market.asset,
                spot=spot,
                threshold=market.threshold,
                direction=market.direction,
                tte_seconds=time_to_expiry_seconds,
            )
            if our_prob is None:
                return None

            # Market's implied YES probability = YES ask price (binary market)
            market_yes_prob = market.yes_price  # 0–1

            edge = our_prob - market_yes_prob   # +ve → we think YES is underpriced
            abs_edge = abs(edge)

            if abs_edge < self._min_detect:
                return None

            if edge > 0:
                # Market underprices YES → buy YES
                side = "YES"
                price = market.yes_price
            else:
                # Market underprices NO → buy NO
                side = "NO"
                price = market.no_price

            return Opportunity(
                market=market,
                our_prob=round(our_prob, 4),
                market_prob=round(market_yes_prob, 4),
                edge=round(edge, 4),
                abs_edge=round(abs_edge, 4),
                recommended_side=side,
                recommended_price=round(price, 4),
                spot_price=spot,
                detected_at=now,
            )

        except Exception as exc:
            log.warning("edge_detector.eval_error", market=market.condition_id, exc=str(exc))
            return None

    def _estimate_prob(
        self,
        asset: str,
        spot: float,
        threshold: float,
        direction: str,
        tte_seconds: float,
    ) -> Optional[float]:
        """
        Log-normal probability estimate.
        Returns P(outcome = YES).
        """
        if spot <= 0 or threshold <= 0 or tte_seconds <= 0:
            return None

        vol_annual = self._estimate_vol(asset)
        tte_years = tte_seconds / (365 * 24 * 3600)
        sigma_sqrt_t = vol_annual * math.sqrt(tte_years)

        if sigma_sqrt_t == 0:
            # Deterministic: already above/below threshold
            if direction == "above":
                return 1.0 if spot > threshold else 0.0
            else:
                return 1.0 if spot < threshold else 0.0

        # d = (ln(S/K)) / (σ√t)  [risk-neutral, μ=0]
        d = math.log(spot / threshold) / sigma_sqrt_t

        if direction == "above":
            prob = float(norm.cdf(d))
        else:
            prob = float(norm.cdf(-d))

        # Clip to avoid extreme probabilities that inflate Kelly size
        return max(0.02, min(0.98, prob))

    def _estimate_vol(self, asset: str) -> float:
        """
        Estimate annualised vol from recent log-returns.
        Falls back to _FALLBACK_VOL_ANNUAL if insufficient data.
        """
        history = self._price_history.get(asset, deque())
        now = time.time()

        # Keep only recent ticks within vol_window
        relevant = [(ts, p) for ts, p in history if now - ts <= self._vol_window]
        if len(relevant) < 10:
            return _FALLBACK_VOL_ANNUAL

        prices = np.array([p for _, p in relevant])
        log_returns = np.diff(np.log(prices))

        if len(log_returns) == 0:
            return _FALLBACK_VOL_ANNUAL

        # Annualise: assume ticks arrive ~every second
        tick_interval = (relevant[-1][0] - relevant[0][0]) / max(1, len(relevant) - 1)
        ticks_per_year = (365 * 24 * 3600) / max(tick_interval, 1)

        vol_annual = float(np.std(log_returns) * math.sqrt(ticks_per_year))
        # Sanity bounds: 10 % – 500 %
        return max(0.10, min(5.0, vol_annual))
