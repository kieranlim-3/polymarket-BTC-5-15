"""
src/position_sizer.py
─────────────────────
Half-Kelly Criterion position sizing for binary Polymarket outcomes.

Kelly Formula (binary)
──────────────────────
Given:
  p  = our estimated probability of winning
  b  = net odds = (1 − market_price) / market_price   [for a $1 stake]

Full Kelly fraction of bankroll:
  f* = (p·b − (1−p)) / b   =   (p − (1−p)/b)

Half-Kelly (standard risk management):
  f  = f* / 2

Applied constraints:
  • f is clamped to [0, max_pct_of_portfolio]
  • Dollar size = f · portfolio_balance
  • Minimum trade size: $10 (Polymarket minimum)
  • Negative Kelly (negative edge) → size = 0 (no trade)
"""

from __future__ import annotations

from dataclasses import dataclass

from src.edge_detector import Opportunity
from src.logger import get_logger

log = get_logger(__name__)

MIN_TRADE_USDC = 10.0   # Polymarket minimum


@dataclass
class SizingResult:
    size_usdc: float
    kelly_fraction: float   # full Kelly (before halving)
    half_kelly_fraction: float
    capped: bool            # True if hard-cap was applied
    reject_reason: str = "" # non-empty → do not trade


class PositionSizer:

    def __init__(
        self,
        kelly_fraction: float = 0.5,      # 0.5 = half-Kelly
        max_portfolio_pct: float = 0.08,  # 8 % hard cap
    ) -> None:
        self._kelly_fraction = kelly_fraction
        self._max_pct = max_portfolio_pct

    def size(
        self,
        opportunity: Opportunity,
        portfolio_usdc: float,
    ) -> SizingResult:
        """
        Calculate the position size for a given opportunity.
        Returns SizingResult; check .reject_reason before placing order.
        """
        p = opportunity.our_prob
        price = opportunity.recommended_price  # market price (0-1)

        if price <= 0 or price >= 1:
            return SizingResult(
                size_usdc=0.0,
                kelly_fraction=0.0,
                half_kelly_fraction=0.0,
                capped=False,
                reject_reason=f"Invalid market price: {price}",
            )

        # Net odds: if we pay `price` for a $1 outcome
        b = (1.0 - price) / price

        # Full Kelly
        full_kelly = (p * b - (1.0 - p)) / b

        if full_kelly <= 0:
            return SizingResult(
                size_usdc=0.0,
                kelly_fraction=full_kelly,
                half_kelly_fraction=0.0,
                capped=False,
                reject_reason=f"Negative Kelly ({full_kelly:.4f}); no edge",
            )

        half_kelly = full_kelly * self._kelly_fraction
        max_fraction = self._max_pct

        capped = half_kelly > max_fraction
        fraction = min(half_kelly, max_fraction)

        size_usdc = fraction * portfolio_usdc

        if size_usdc < MIN_TRADE_USDC:
            return SizingResult(
                size_usdc=0.0,
                kelly_fraction=full_kelly,
                half_kelly_fraction=half_kelly,
                capped=capped,
                reject_reason=(
                    f"Size ${size_usdc:.2f} below minimum ${MIN_TRADE_USDC:.2f}. "
                    f"Portfolio too small or edge too thin."
                ),
            )

        log.debug(
            "position_sizer",
            p=p,
            b=round(b, 3),
            full_kelly=round(full_kelly, 4),
            half_kelly=round(half_kelly, 4),
            fraction=round(fraction, 4),
            size_usdc=round(size_usdc, 2),
            capped=capped,
        )

        return SizingResult(
            size_usdc=round(size_usdc, 2),
            kelly_fraction=round(full_kelly, 4),
            half_kelly_fraction=round(half_kelly, 4),
            capped=capped,
        )
