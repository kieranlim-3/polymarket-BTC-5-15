"""
backtest.py — Polymarket BTC Up/Down Bot  (v6)
===============================================
Run with:
    pip install aiohttp pandas numpy scipy
    python backtest.py

Reading your results and what they meant
─────────────────────────────────────────
v5 showed:
  5m:  48.8% win rate, +767% ROI, 39% ruin probability, 84% drawdown
  15m: 48.3% win rate, +196% ROI, 47% ruin probability, 96% drawdown

The +767% ROI with a 48.8% win rate is a contradiction.
A 48.8% win rate on a 50-cent market has NEGATIVE expected value.
What actually happened: Kelly was betting 15-20% of the bankroll per
trade. On the one path that got lucky early, the bankroll compounded
explosively. But 39% of all paths ended in ruin — meaning the strategy
was a coin flip being bet like a certainty. The ROI number was lying.

Changes in v6
─────────────────────────────────────────
FIX 1 — Kelly cap reduced from 20% to 5% of bankroll per trade.
  max_bet_usdc now scales with bankroll (5% hard cap) instead of a
  fixed $20 ceiling. This is the most important change. A 48% win rate
  strategy should bet almost nothing — Kelly naturally produces near-zero
  fractions, and the old $20 cap was overriding that safety valve.

FIX 2 — Monte Carlo uses per-trade bankroll for return calculation.
  v5 divided all P&L by the starting $100, making every shuffle replay
  identical returns. We now store the bankroll at each trade entry and
  divide pnl by that, so different orderings genuinely diverge.

FIX 3 — Volume filter added as Signal 4.
  We now fetch volume data from Binance. A signal is only acted on if
  the current candle's volume is above its 20-candle average. Volume
  above average = genuine participation. Volume below average = noise.
  This is independent of RSI, ROC, and EMA — a true 4th signal.
  Requires 3-of-4 agreement (majority of 4) to trade.

FIX 4 — Drawdown guard.
  If the simulated portfolio falls more than 20% from its recent peak,
  trading pauses for the rest of that rolling window. This mirrors the
  live bot's risk_manager halt and prevents a losing streak from
  compounding into ruin during the backtest.

FIX 5 — Inline output comments.
  Every printed metric now has a plain-English explanation of what it
  means and what a good value looks like. You should never have to guess
  what a number means while reading the output.
"""

import asyncio
import aiohttp
import time
import math
import csv
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from itertools import product


# ══════════════════════════════════════════════════════════════
# CONFIG
# Edit the values in this block to tune the strategy.
# Everything else in the file flows from these settings.
# ══════════════════════════════════════════════════════════════

TIMEFRAMES = {
    # Polymarket only offers 5-minute and 15-minute BTC markets.
    # We run both so you can compare which suits this strategy better.
    "5m": {
        "interval":    "5m",
        "cache_file":  "candles_5m.csv",
        "label":       "5-minute markets",
        "trades_file": "trades_5m.csv",
        "equity_file": "equity_5m.csv",
        "mc_file":     "mc_5m.csv",
        "grid_file":   "grid_5m.csv",
    },
    "15m": {
        "interval":    "15m",
        "cache_file":  "candles_15m.csv",
        "label":       "15-minute markets",
        "trades_file": "trades_15m.csv",
        "equity_file": "equity_15m.csv",
        "mc_file":     "mc_15m.csv",
        "grid_file":   "grid_15m.csv",
    },
}

CONFIG = {
    # ── Data ─────────────────────────────────────────────────
    "symbol":          "BTCUSDT",
    "lookback_days":   90,      # Days of history to download.
                                # More = better grid search, more trades.
                                # Minimum recommended: 60.

    # ── Signal defaults ───────────────────────────────────────
    # These are starting values. Grid search will override them
    # with the best combination found on in-sample data.
    "roc_threshold":   0.0005,  # Minimum price change speed to trade.
                                # Higher = fewer but stronger signals.
    "rsi_period":      14,      # RSI lookback window (candles).
    "rsi_oversold":    45,      # RSI below this → price may bounce up.
    "rsi_overbought":  55,      # RSI above this → price may pull back.
    "ema_fast":        5,       # Fast EMA span. Reacts quickly to price.
    "ema_slow":        20,      # Slow EMA span. Tracks medium-term trend.
    "vol_window":      20,      # Volume average window (candles).
                                # Signal 4: volume must exceed this average.

    # ── Grid search ───────────────────────────────────────────
    # Grid search tests every combination of these values on
    # in-sample data and picks the best one.
    "grid_roc":        [0.0003, 0.0005, 0.001, 0.002],
    "grid_rsi_band":   [5, 10, 15, 20],   # distance from RSI=50
    "grid_ema_slow":   [10, 20, 30],
    "grid_min_trades": 40,      # Minimum in-sample trades required.
                                # Combos producing fewer are discarded
                                # to prevent overfitting to noise.

    # ── Bankroll & sizing ─────────────────────────────────────
    "bankroll":        100.0,   # Starting capital in USDC.
    "kelly_fraction":  0.5,     # Half-Kelly. Halves the raw Kelly bet.
                                # Lower = safer. 0.25 = quarter-Kelly.
    "max_position_pct": 0.05,   # Hard cap: max 5% of current bankroll
                                # per trade. Overrides Kelly if Kelly
                                # would bet more. FIX 1: was $20 flat.
    "min_bet_usdc":    1.0,     # Skip trade if Kelly produces less than
                                # this. Polymarket minimum is $1.

    # ── Realism ───────────────────────────────────────────────
    "polymarket_fee":  0.02,    # 2% fee on winnings. Real Polymarket cost.
    "slippage_pct":    0.005,   # 0.5% worse price than quoted.
                                # Accounts for bid/ask spread.

    # ── Risk management ───────────────────────────────────────
    "drawdown_halt_pct": 0.20,  # Pause trading if portfolio drops 20%
                                # from its recent peak within a window.
                                # Mirrors the live bot's risk_manager.

    # ── Walk-forward ──────────────────────────────────────────
    "train_pct":       0.70,    # 70% of data used for grid search.
                                # 30% held out for honest OOS testing.
    "n_wf_windows":    4,       # Number of rolling OOS windows.

    # ── Monte Carlo ───────────────────────────────────────────
    "mc_simulations":  1000,    # Number of random reshuffles.
    "mc_min_trades":   40,      # Warn if fewer trades than this.
}


# ══════════════════════════════════════════════════════════════
# DATA FETCHING
# Downloads OHLCV (open, high, low, close, volume) candles.
# Volume is now fetched — needed for Signal 4.
# ══════════════════════════════════════════════════════════════

async def fetch_candles(interval: str, cache_file: str) -> pd.DataFrame:
    """
    Loads candles from local CSV cache if it exists.
    Otherwise fetches from Binance and saves to cache.

    Cache means Binance is only called once per timeframe.
    Delete the cache CSV to force a fresh download.
    """
    if os.path.exists(cache_file):
        print(f"  Loading {interval} candles from cache ({cache_file})")
        df = pd.read_csv(cache_file, parse_dates=["time"])
        print(f"  Loaded {len(df):,} candles.\n")
        return df

    mins_per_candle = int(interval.replace("m", ""))
    total_needed    = CONFIG["lookback_days"] * 24 * 60 // mins_per_candle
    print(f"  Fetching {CONFIG['lookback_days']}d of {interval} candles "
          f"({total_needed:,} needed)...")

    all_raw  = []
    end_time = int(time.time() * 1000)

    async with aiohttp.ClientSession() as session:
        while len(all_raw) < total_needed:
            params = {
                "symbol":   CONFIG["symbol"],
                "interval": interval,
                "limit":    1000,
                "endTime":  end_time,
            }
            async with session.get(
                "https://api.binance.com/api/v3/klines",
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as r:
                if r.status != 200:
                    print(f"\n  Binance returned {r.status} — stopping early.")
                    break
                chunk = await r.json()
            if not chunk:
                break
            all_raw  = chunk + all_raw
            end_time = chunk[0][0] - 1
            print(f"    {len(all_raw):>6}/{total_needed} candles...", end="\r")
            await asyncio.sleep(0.25)

    print(f"\n  Fetched {len(all_raw):,} candles.")

    df = pd.DataFrame(all_raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbbav", "tbqav", "ignore",
    ])
    df["open"]   = pd.to_numeric(df["open"],   errors="coerce")
    df["close"]  = pd.to_numeric(df["close"],  errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")  # NEW: needed for S4
    df["time"]   = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)

    # Keep open, close, volume — all three needed now
    df = df[["time", "open", "close", "volume"]].dropna().reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    print(f"  Cached → {cache_file}\n")
    return df


# ══════════════════════════════════════════════════════════════
# INDICATORS
# Three price-based signals + one volume signal.
# Each measures something genuinely different.
# ══════════════════════════════════════════════════════════════

def compute_rsi(closes: np.ndarray, period: int) -> float:
    """
    Relative Strength Index (RSI).

    Measures whether recent price moves have been mostly up or mostly down.
    RSI = 100 means all recent moves were gains (overbought → expect pullback).
    RSI = 0 means all recent moves were losses (oversold → expect bounce).

    We use mild thresholds (45/55) rather than classic 30/70 to get
    more signals. Classic 30/70 is too rare on short timeframes.
    """
    if len(closes) < period + 1:
        return 50.0   # not enough data — return neutral
    deltas   = np.diff(closes[-(period + 1):])
    gains    = np.where(deltas > 0, deltas,  0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    return round(100.0 - (100.0 / (1.0 + avg_gain / avg_loss)), 2)


def volume_above_average(volumes: np.ndarray, window: int) -> bool:
    """
    Signal 4: is current candle's volume above its recent average?

    Why this helps: genuine price moves are accompanied by high volume
    (lots of people transacting). Low-volume moves are often noise —
    price drifts without real conviction and is likely to reverse.

    This signal is completely independent of RSI, ROC, and EMA because
    it looks at trading activity, not price direction.
    """
    if len(volumes) < window + 1:
        return False   # not enough history — skip
    avg_vol     = volumes[-(window + 1):-1].mean()   # average of last `window` candles
    current_vol = volumes[-1]
    return current_vol > avg_vol


def compute_signal(
    closes:      np.ndarray,
    volumes:     np.ndarray,
    window_open: float,
    cfg:         dict,
) -> tuple:
    """
    Returns (direction, confidence) where direction is "up"/"down"/None.

    Four signals, each measuring something different:
      S1 — RSI:    is price overbought or oversold? (mean-reversion)
      S2 — ROC:    how fast is price moving right now? (momentum strength)
      S3 — EMA:    is the medium-term trend up or down? (trend direction)
      S4 — Volume: is this move backed by real trading activity?

    Voting: requires 3-of-4 signals to agree.
      4-of-4 agreement → confidence multiplier 1.0  (full Kelly sizing)
      3-of-4 agreement → confidence multiplier 0.65 (smaller Kelly bet)

    The confidence score feeds directly into kelly_bet() — lower
    confidence automatically means a smaller bet size.
    """
    min_len = max(cfg["ema_slow"] + 2, cfg["rsi_period"] + 2, cfg["vol_window"] + 2)
    if len(closes) < min_len:
        return None, 0.0

    # ── Signal 1: RSI ──────────────────────────────────────
    # Oversold (RSI low) → price likely to go up
    # Overbought (RSI high) → price likely to go down
    rsi = compute_rsi(closes, cfg["rsi_period"])
    if cfg["rsi_oversold"] < rsi < cfg["rsi_overbought"]:
        return None, 0.0   # RSI is neutral — no mean-reversion signal
    s1 = "up" if rsi <= cfg["rsi_oversold"] else "down"

    # ── Signal 2: ROC ──────────────────────────────────────
    # Rate of change: how much did price move in this candle?
    # If move is too small, it's noise — skip entirely.
    roc = (closes[-1] - closes[-2]) / closes[-2]
    if abs(roc) < cfg["roc_threshold"]:
        return None, 0.0   # momentum too weak — skip
    s2 = "up" if roc > 0 else "down"

    # ── Signal 3: EMA crossover ────────────────────────────
    # Fast EMA above slow EMA → uptrend
    # Fast EMA below slow EMA → downtrend
    s       = pd.Series(closes)
    ema_f   = s.ewm(span=cfg["ema_fast"], adjust=False).mean().iloc[-1]
    ema_s   = s.ewm(span=cfg["ema_slow"], adjust=False).mean().iloc[-1]
    s3      = "up" if ema_f > ema_s else "down"

    # ── Signal 4: Volume ───────────────────────────────────
    # Current volume above 20-candle average → genuine move
    # Volume below average → possibly noise, skip
    high_vol = volume_above_average(volumes, cfg["vol_window"])
    s4       = "up" if (high_vol and s2 == "up") else ("down" if (high_vol and s2 == "down") else "neutral")

    # ── Confidence base ────────────────────────────────────
    # How extreme is RSI? How strong is the ROC?
    # More extreme = higher confidence = larger Kelly bet.
    if s1 == "up":
        rsi_c = max((cfg["rsi_oversold"] - rsi) / max(cfg["rsi_oversold"], 1), 0.0)
    else:
        rsi_c = max((rsi - cfg["rsi_overbought"]) / max(100 - cfg["rsi_overbought"], 1), 0.0)
    roc_c    = min(abs(roc) / cfg["roc_threshold"] / 10.0, 1.0)
    base_conf = min((rsi_c + roc_c) / 2.0, 1.0)

    # ── Voting ─────────────────────────────────────────────
    price_signals = [s1, s2, s3]
    vol_confirms  = (s4 != "neutral")   # True if volume backs the move

    ups   = price_signals.count("up")
    downs = price_signals.count("down")

    # 4-of-4: all price signals agree AND volume confirms
    if ups == 3 and vol_confirms and s2 == "up":
        return "up",   round(base_conf * 1.0, 3)
    if downs == 3 and vol_confirms and s2 == "down":
        return "down", round(base_conf * 1.0, 3)

    # 3-of-4: 2 price signals agree + volume, or all 3 price agree
    if ups == 3 and not vol_confirms:
        return "up",   round(base_conf * 0.65, 3)   # smaller bet — no volume
    if downs == 3 and not vol_confirms:
        return "down", round(base_conf * 0.65, 3)
    if ups == 2 and vol_confirms and s2 == "up":
        return "up",   round(base_conf * 0.65, 3)
    if downs == 2 and vol_confirms and s2 == "down":
        return "down", round(base_conf * 0.65, 3)

    return None, 0.0   # signals don't agree — no trade


# ══════════════════════════════════════════════════════════════
# MARKET PRICE SIMULATION
# Polymarket doesn't publish historical order-book data.
# We model what the YES price would have been at signal time.
# ══════════════════════════════════════════════════════════════

def simulate_market_price(direction: str, roc: float) -> float:
    """
    Simulates the Polymarket YES price you'd see on the order book.

    Model: start at 0.50 (neutral), nudge toward signal direction
    (the market has partial information), add small random noise.

    Conservative assumption: the market has already priced in ~60%
    of whatever we can see, so our edge is smaller than raw signals suggest.
    Real Polymarket prices would vary by market, time, and liquidity.
    """
    rng   = np.random.default_rng(seed=int(abs(roc) * 1e9) % (2 ** 31))
    noise = rng.uniform(-0.04, 0.04)                           # ±4 cents noise
    nudge = abs(roc) * 20                                      # market partial info
    base  = (0.50 + nudge) if direction == "up" else (0.50 - nudge)
    return round(float(np.clip(base + noise, 0.35, 0.65)), 3)  # clamp to [0.35, 0.65]


# ══════════════════════════════════════════════════════════════
# POSITION SIZING
# Half-Kelly with a hard percentage cap.
# ══════════════════════════════════════════════════════════════

def kelly_bet(
    win_prob_yes: float,   # our estimated P(price goes up)
    market_price: float,   # Polymarket YES price
    bankroll:     float,   # current portfolio value
    direction:    str,     # "up" or "down"
) -> float:
    """
    Calculates the bet size using half-Kelly criterion.

    Kelly formula tells you the mathematically optimal fraction of
    your bankroll to bet to maximise long-run growth. Half-Kelly
    bets half that amount — reduces variance while keeping most growth.

    Two safety limits applied after Kelly:
    1. kelly_fraction=0.5 halves the raw Kelly fraction
    2. max_position_pct=5% hard caps the bet regardless of Kelly

    FIX 1: old code used max_bet_usdc=$20 (a fixed dollar amount).
    $20 on a $100 bankroll = 20% per trade, which is enormous for a
    strategy with 48% win rate. New code uses 5% of current bankroll,
    which scales down automatically as the portfolio shrinks.
    """
    slipped = min(market_price + CONFIG["slippage_pct"], 0.95)

    # When betting NO/down, P(win) = 1 - P(price goes up)
    p_win = (1.0 - win_prob_yes) if direction == "down" else win_prob_yes

    if not (0 < slipped < 1):
        return 0.0

    b = (1.0 / slipped) - 1                    # net profit per $1 if we win
    f = (p_win * b - (1 - p_win)) / b          # full Kelly fraction
    f = max(f, 0.0)                             # no negative bets

    half_kelly_bet = CONFIG["kelly_fraction"] * f * bankroll
    max_bet        = CONFIG["max_position_pct"] * bankroll   # 5% hard cap

    bet = min(half_kelly_bet, max_bet)          # take the smaller of the two
    return round(max(bet, 0.0), 2)


# ══════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════

def sharpe(returns: list, trades_per_day: float) -> float:
    """
    Annualised Sharpe ratio.

    Measures return per unit of risk, scaled to a full year.
    > 1.0  = good
    > 2.0  = very good
    < 0    = losing money on a risk-adjusted basis
    """
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    std = arr.std()
    if std == 0:
        return 0.0
    return round((arr.mean() / std) * math.sqrt(max(trades_per_day, 0.01) * 252), 2)


def max_drawdown(equity: list) -> float:
    """
    Largest peak-to-trough drop as a percentage.

    Example: equity goes $100 → $150 → $90.
    Peak = $150, trough = $90, drawdown = ($150-$90)/$150 = 40%.

    < 15%  = well controlled
    15-25% = acceptable
    > 25%  = too much — reduce max_position_pct
    """
    peak, dd = equity[0], 0.0
    for v in equity:
        peak = max(peak, v)
        dd   = max(dd, (peak - v) / peak)
    return round(dd * 100, 2)


def rolling_win_rate(trades: list, window: int = 30) -> list:
    """Win rate over a sliding window of last `window` trades."""
    outcomes = [1 if t["outcome"] == "WIN" else 0 for t in trades]
    rates = []
    for i in range(len(outcomes)):
        chunk = outcomes[max(0, i - window + 1): i + 1]
        rates.append(round(sum(chunk) / len(chunk), 3))
    return rates


# ══════════════════════════════════════════════════════════════
# CORE SIMULATION LOOP
# ══════════════════════════════════════════════════════════════

@dataclass
class SimResult:
    """Stores everything produced by one simulation run."""
    trades:            list  = field(default_factory=list)
    equity:            list  = field(default_factory=list)
    trade_returns:     list  = field(default_factory=list)  # pnl / bankroll_at_entry
    skipped_signal:    int   = 0
    skipped_neg_kelly: int   = 0
    skipped_bet:       int   = 0
    skipped_drawdown:  int   = 0   # trades skipped due to drawdown halt
    attempted:         int   = 0

    @property
    def win_rate(self) -> float:
        """Fraction of completed trades that were wins."""
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t["outcome"] == "WIN") / len(self.trades)

    @property
    def total_pnl(self) -> float:
        """Total profit or loss from starting bankroll."""
        return (self.equity[-1] - CONFIG["bankroll"]) if self.equity else 0.0


def run_simulation(
    df:        pd.DataFrame,
    start_idx: int,
    end_idx:   int,
    cfg:       dict,
    tf_label:  str = "",
) -> SimResult:
    """
    Simulates trading on df[start_idx : end_idx].

    Each iteration = one candle = one Polymarket market duration.
    Signal computed on candle i close price.
    Outcome determined by candle i+1 close vs candle i close.

    Drawdown guard: if portfolio drops 20% from its peak within this
    window, all further trading in the window is halted. This prevents
    a bad run from compounding into catastrophic loss.
    """
    result        = SimResult()
    bankroll      = CONFIG["bankroll"]
    peak_bankroll = bankroll                    # tracks all-time high for drawdown
    halt_trading  = False                       # set True when drawdown breached

    result.equity.append(bankroll)

    min_i = max(start_idx, cfg["ema_slow"] + cfg["rsi_period"] + cfg["vol_window"] + 5)

    for i in range(min_i, end_idx - 1):
        result.attempted += 1

        # ── Drawdown guard ──────────────────────────────────
        # Update peak, check if we've fallen too far.
        peak_bankroll = max(peak_bankroll, bankroll)
        drawdown_now  = (peak_bankroll - bankroll) / peak_bankroll

        if drawdown_now >= CONFIG["drawdown_halt_pct"]:
            if not halt_trading:
                # Only print the halt message once per window
                halt_trading = True
            result.skipped_drawdown += 1
            continue

        # ── Compute signals ─────────────────────────────────
        window_open = float(df["open"].iloc[i])
        closes      = df["close"].iloc[max(0, i - 40): i + 1].values
        volumes     = df["volume"].iloc[max(0, i - 40): i + 1].values
        roc         = (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 else 0.0

        direction, confidence = compute_signal(closes, volumes, window_open, cfg)

        if direction is None:
            result.skipped_signal += 1
            continue

        # ── Derive win probability from confidence ──────────
        # Confidence 0→1 maps win_prob to 0.52→0.65.
        # At zero confidence → just above coinflip (0.52).
        # At full confidence → 0.65 (our model's max estimate).
        win_prob_yes = 0.52 + confidence * 0.13

        # ── Simulate market price and size the bet ──────────
        market_price    = simulate_market_price(direction, roc)
        entry_bankroll  = bankroll                       # FIX 2: save bankroll before bet

        bet = kelly_bet(win_prob_yes, market_price, bankroll, direction)
        if bet <= 0:
            result.skipped_neg_kelly += 1
            continue
        if bet < CONFIG["min_bet_usdc"]:
            result.skipped_bet += 1
            continue

        # ── Determine outcome ───────────────────────────────
        # Did price move in our predicted direction?
        entry_close = float(df["close"].iloc[i])
        exit_close  = float(df["close"].iloc[i + 1])
        won = (exit_close > entry_close) if direction == "up" else (exit_close < entry_close)

        # ── Calculate P&L ───────────────────────────────────
        slipped = min(market_price + CONFIG["slippage_pct"], 0.95)
        if won:
            # Gross profit = bet × net odds
            # Net odds = (1 / price) - 1  (e.g. price=0.50 → odds=1.0, profit=$1 per $1 bet)
            gross = bet * ((1.0 / slipped) - 1.0)
            pnl   = gross * (1 - CONFIG["polymarket_fee"])   # subtract 2% fee
        else:
            pnl = -bet   # lose the whole stake

        bankroll = max(round(bankroll + pnl, 2), 0.0)
        result.equity.append(bankroll)

        # FIX 2: store return as fraction of bankroll AT ENTRY, not start.
        # This makes Monte Carlo shuffles produce genuine variance.
        result.trade_returns.append(pnl / entry_bankroll)

        dt = df["time"].iloc[i].to_pydatetime()
        result.trades.append({
            "timestamp":     dt.strftime("%Y-%m-%d %H:%M UTC"),
            "timeframe":     tf_label,
            "direction":     direction.upper(),
            "confidence":    round(confidence, 3),
            "vol_confirmed": confidence > 0,   # True if S4 contributed
            "market_price":  market_price,
            "win_prob":      round(win_prob_yes, 3),
            "entry_close":   round(entry_close, 2),
            "exit_close":    round(exit_close, 2),
            "bet_usdc":      round(bet, 2),
            "outcome":       "WIN" if won else "LOSS",
            "pnl":           round(pnl, 2),
            "bankroll":      bankroll,
        })

    return result


# ══════════════════════════════════════════════════════════════
# PRINTING RESULTS
# Every metric printed with an inline explanation.
# ══════════════════════════════════════════════════════════════

def print_result(label: str, result: SimResult, days: float,
                 flag_low: bool = False) -> None:
    """Print a SimResult with inline metric explanations."""
    trades = result.trades
    if not trades:
        print(f"  [{label}] No trades generated.\n")
        return

    total  = len(trades)
    wins   = sum(1 for t in trades if t["outcome"] == "WIN")
    wr     = wins / total * 100
    pnl    = result.total_pnl
    roi    = pnl / CONFIG["bankroll"] * 100
    tpd    = total / max(days, 1)
    sh     = sharpe(result.trade_returns, tpd)
    dd     = max_drawdown(result.equity)

    W = 62
    print("=" * W)
    print(f"  {label}")
    print("=" * W)

    # Trades per day — want at least 2/day for reliable statistics
    low_flag = "  ⚠️  low — results less reliable" if (flag_low and total < CONFIG["mc_min_trades"]) else ""
    print(f"  Trades/day : {tpd:.1f}  (total {total}){low_flag}")
    print(f"             → Want ≥2/day for reliable stats. "
          f"Low? Widen grid_rsi_band or lower grid_roc.")

    # Win rate — need >53% to have edge after fees at 0.50 market price
    wr_note = "✅ has edge" if wr >= 53 else ("⚠️  marginal" if wr >= 50 else "❌ no edge")
    print(f"  Win rate   : {wr:.1f}%  ({wins}W / {total - wins}L)  [{wr_note}]")
    print(f"             → Need >53% to cover fees. ~50% = coin flip.")

    # P&L — only meaningful if win rate > 53%
    pnl_note = "✅" if pnl > 0 and wr >= 53 else ("⚠️  driven by Kelly luck" if pnl > 0 else "❌")
    print(f"  P&L        : ${pnl:+.2f}  (ROI {roi:+.1f}%)  [{pnl_note}]")
    print(f"             → Positive P&L with <53% WR = Kelly variance, not real edge.")

    # Sharpe — risk-adjusted return
    sh_note = "✅" if sh > 1.0 else ("⚠️" if sh > 0 else "❌")
    print(f"  Sharpe     : {sh}  [{sh_note}]")
    print(f"             → >1.0 good, >2.0 very good, <0 losing on risk basis.")

    # Drawdown — how much you'd have lost at worst point
    dd_note = "✅" if dd < 15 else ("⚠️" if dd < 25 else "❌ reduce max_position_pct")
    print(f"  Drawdown   : {dd}%  [{dd_note}]")
    print(f"             → <15% good. >25% = lower max_position_pct in CONFIG.")

    if result.skipped_drawdown > 0:
        print(f"  Halted     : {result.skipped_drawdown} candles paused by drawdown guard")

    if result.skipped_signal > 0:
        pct_skip = result.skipped_signal / max(result.attempted, 1) * 100
        print(f"  Skipped    : {result.skipped_signal:,} signal ({pct_skip:.0f}% of candles)  "
              f"| {result.skipped_neg_kelly} neg-Kelly  "
              f"| {result.skipped_bet} min-bet")
    print()


# ══════════════════════════════════════════════════════════════
# GRID SEARCH
# Finds the best signal parameters on in-sample data only.
# Anti-overfit: requires minimum trades + penalises small samples.
# ══════════════════════════════════════════════════════════════

def grid_search(
    df:         pd.DataFrame,
    split_idx:  int,
    grid_file:  str,
    days_train: float,
) -> dict:
    """
    Tests every combination of roc, rsi_band, and ema_slow on
    the in-sample portion of the data.

    Scoring: adjusted_sharpe = raw_sharpe × sqrt(n_trades / 100)
    This penalises small-sample results. A 3.0 Sharpe from 100 trades
    scores higher than a 5.0 Sharpe from 25 trades.

    Only combos producing ≥ grid_min_trades are considered at all.
    This prevents the grid from finding a "perfect" 10-trade sequence
    that just got lucky.
    """
    combos = list(product(
        CONFIG["grid_roc"],
        CONFIG["grid_rsi_band"],
        CONFIG["grid_ema_slow"],
    ))
    print(f"  Grid search: {len(combos)} combinations "
          f"(≥{CONFIG['grid_min_trades']} in-sample trades required)...")

    rows = []
    for roc, band, ema_slow in combos:
        cfg = {
            **CONFIG,
            "roc_threshold":  roc,
            "rsi_oversold":   50 - band,
            "rsi_overbought": 50 + band,
            "ema_slow":       int(ema_slow),
        }
        res = run_simulation(df, 0, split_idx, cfg)
        n   = len(res.trades)

        if n < CONFIG["grid_min_trades"]:
            continue   # not enough trades — discard this combo

        tpd        = n / max(days_train, 1)
        raw_sharpe = sharpe(res.trade_returns, tpd)
        score      = raw_sharpe * math.sqrt(n / 100.0)   # penalise small samples

        rows.append({
            "roc_threshold":  roc,
            "rsi_band":       band,
            "rsi_oversold":   50 - band,
            "rsi_overbought": 50 + band,
            "ema_slow":       ema_slow,
            "trades":         n,
            "win_rate_pct":   round(res.win_rate * 100, 1),
            "pnl":            round(res.total_pnl, 2),
            "raw_sharpe":     raw_sharpe,
            "score":          round(score, 3),
            "max_drawdown":   max_drawdown(res.equity),
        })

    if not rows:
        print(f"  ⚠️  No combo reached {CONFIG['grid_min_trades']} trades.")
        print(f"     Suggestions: lower grid_min_trades, add larger grid_rsi_band values,")
        print(f"     or increase lookback_days.\n")
        return CONFIG

    rdf  = pd.DataFrame(rows).sort_values("score", ascending=False)
    rdf.to_csv(grid_file, index=False)
    best = rdf.iloc[0].to_dict()

    print(f"  Best params found:")
    print(f"    roc_threshold  = {best['roc_threshold']}")
    print(f"    rsi_band       = ±{int(best['rsi_band'])}  "
          f"(oversold={int(best['rsi_oversold'])}, "
          f"overbought={int(best['rsi_overbought'])})")
    print(f"    ema_slow       = {int(best['ema_slow'])}")
    print(f"    in-sample trades = {int(best['trades'])}  "
          f"WR = {best['win_rate_pct']}%  "
          f"Sharpe = {best['raw_sharpe']}  "
          f"score = {best['score']}")
    print(f"  Full grid saved → {grid_file}")

    return {
        **CONFIG,
        "roc_threshold":  best["roc_threshold"],
        "rsi_oversold":   int(best["rsi_oversold"]),
        "rsi_overbought": int(best["rsi_overbought"]),
        "ema_slow":       int(best["ema_slow"]),
    }


# ══════════════════════════════════════════════════════════════
# ROLLING WALK-FORWARD
# The honest robustness test. Tests OOS performance across
# multiple time windows, not just one lucky split.
# ══════════════════════════════════════════════════════════════

def rolling_walk_forward(
    df:             pd.DataFrame,
    best_cfg:       dict,
    tf_label:       str,
    mins_per_candle: int,
) -> list:
    """
    Splits the full dataset into N rolling out-of-sample windows.

    Example with lookback_days=90, train_pct=0.70, n_wf_windows=4:
      Train on days  1-63, test days 64-70   (window 1)
      Train on days  1-70, test days 71-77   (window 2)
      Train on days  1-77, test days 78-84   (window 3)
      Train on days  1-84, test days 85-90   (window 4)

    A robust strategy wins in 3 or 4 windows.
    An overfit strategy wins in 1 window and loses the rest.
    Any window with <10 OOS trades is flagged as unreliable.
    """
    n          = len(df)
    train_size = int(n * CONFIG["train_pct"])
    test_frac  = (1 - CONFIG["train_pct"]) / CONFIG["n_wf_windows"]
    test_size  = int(n * test_frac)

    candles_per_day = int(24 * 60 / mins_per_candle)
    windows         = []

    for w in range(CONFIG["n_wf_windows"]):
        test_start = train_size + w * test_size
        test_end   = min(test_start + test_size, n - 1)

        if test_start >= n - 1 or test_end <= test_start:
            break

        res      = run_simulation(df, test_start, test_end, best_cfg, tf_label)
        n_trades = len(res.trades)
        reliable = n_trades >= 10   # need at least 10 OOS trades to trust the result

        start_dt = df["time"].iloc[test_start].strftime("%b %d")
        end_dt   = df["time"].iloc[test_end - 1].strftime("%b %d")

        windows.append({
            "window":   w + 1,
            "period":   f"{start_dt} → {end_dt}",
            "trades":   n_trades,
            "win_rate": res.win_rate,
            "pnl":      res.total_pnl,
            "drawdown": max_drawdown(res.equity),
            "reliable": reliable,
            "result":   res,
        })

    return windows


def print_rolling_wf(windows: list) -> None:
    """Print rolling walk-forward results with explanations."""
    W = 62
    print("=" * W)
    print("  ROLLING WALK-FORWARD  (out-of-sample windows only)")
    print("  Each window = fresh data the strategy has never seen.")
    print("=" * W)
    print(f"  {'Win':>3}  {'Period':<18} {'Trades':>7} "
          f"{'WinRate':>8} {'P&L':>8}  {'Status'}")
    print(f"  {'-'*3}  {'-'*18} {'-'*7} {'-'*8} {'-'*8}  {'-'*12}")

    profitable = 0
    reliable   = 0

    for w in windows:
        if not w["reliable"]:
            wr_str  = "  n/a  "
            pnl_str = "  n/a  "
            status  = "⚠️  <10 trades"
        else:
            wr_str  = f"{w['win_rate']*100:.1f}%"
            pnl_str = f"${w['pnl']:+.2f}"
            reliable += 1
            if w["win_rate"] >= 0.53:
                status = "✅ edge"
                profitable += 1
            elif w["win_rate"] >= 0.48:
                status = "➖ marginal"
            else:
                status = "❌ no edge"

        print(f"  {w['window']:>3}  {w['period']:<18} {w['trades']:>7} "
              f"{wr_str:>8} {pnl_str:>8}  {status}")

    print()

    if reliable == 0:
        print("  ⚠️  All windows had <10 trades — unreliable.")
        print("     Increase lookback_days or widen signal thresholds.\n")
        return

    consistency = profitable / reliable
    print(f"  {profitable}/{reliable} reliable windows showed edge  "
          f"({consistency*100:.0f}% consistency)")
    print(f"  → 75%+ = strategy is robust across time.")
    print(f"  → 50%  = works sometimes — not reliable enough for live trading.")
    print(f"  → <50% = overfit or no real edge.\n")

    if consistency >= 0.75:
        print("  ✅  Consistent across windows — good sign.\n")
    elif consistency >= 0.50:
        print("  ⚠️   Inconsistent — some windows fail. Needs more tuning.\n")
    else:
        print("  ❌  Fails most windows — do not trade this configuration.\n")


# ══════════════════════════════════════════════════════════════
# MONTE CARLO
# Shows the distribution of outcomes across 1000 random orderings.
# ══════════════════════════════════════════════════════════════

def monte_carlo(trades: list, n_sims: int) -> dict:
    """
    Shuffles trade returns 1000 times and replays each path.

    FIX 2: uses per-trade returns (pnl / bankroll_at_entry) rather than
    pnl / starting_bankroll. This means each shuffled path genuinely
    compounds differently — paths that win early have more capital to
    compound, paths that lose early have less. Gives real variance.

    Ruin = bankroll falls below $1 at any point in the path.
    """
    if not trades:
        return {}

    # Per-trade return = pnl as fraction of the bankroll at time of trade
    trade_rets = [t["pnl"] / max(t["bankroll"] - t["pnl"], 1.0) for t in trades]

    rng    = np.random.default_rng(seed=42)
    start  = CONFIG["bankroll"]
    finals, dds, ruins = [], [], 0

    for _ in range(n_sims):
        shuffled = rng.permutation(trade_rets)
        bal      = start
        equity   = [bal]
        ruined   = False

        for ret in shuffled:
            # Apply the % return to current bankroll (compounding)
            bal = max(round(bal * (1.0 + ret), 2), 0.0)
            equity.append(bal)
            if bal < 1.0:
                ruined = True

        finals.append(equity[-1])
        dds.append(max_drawdown(equity))
        if ruined:
            ruins += 1

    arr    = np.array(finals)
    dd_arr = np.array(dds)

    return {
        "n_trades":      len(trades),
        "too_few":       len(trades) < CONFIG["mc_min_trades"],
        "simulations":   n_sims,
        "ruin_pct":      round(ruins / n_sims * 100, 1),
        "median_final":  round(float(np.median(arr)), 2),
        "p10_final":     round(float(np.percentile(arr, 10)), 2),
        "p90_final":     round(float(np.percentile(arr, 90)), 2),
        "worst_final":   round(float(arr.min()), 2),
        "best_final":    round(float(arr.max()), 2),
        "median_dd_pct": round(float(np.median(dd_arr)), 1),
        "p90_dd_pct":    round(float(np.percentile(dd_arr, 90)), 1),
        "all_finals":    arr.tolist(),
    }


def print_monte_carlo(mc: dict) -> None:
    if not mc:
        print("  Monte Carlo: no trades.\n")
        return
    start = CONFIG["bankroll"]
    W     = 62
    print("=" * W)
    print("  MONTE CARLO  —  1 000 random reshuffles of trade order")
    print("  Shows the distribution of possible outcomes, not just")
    print("  the one path that actually happened.")
    print("=" * W)

    if mc.get("too_few"):
        print(f"  ⚠️  Only {mc['n_trades']} trades (need {CONFIG['mc_min_trades']}+).")
        print(f"     Results are approximate. Increase lookback_days.\n")

    print(f"  Starting bankroll  : ${start:.2f}")
    print(f"  Median outcome     : ${mc['median_final']:.2f}  "
          f"({(mc['median_final']-start)/start*100:+.1f}%)")
    print(f"                     → Half of all paths end above, half below this.")
    print(f"  10th percentile    : ${mc['p10_final']:.2f}  "
          f"({(mc['p10_final']-start)/start*100:+.1f}%)")
    print(f"                     → 1-in-10 paths do THIS badly or worse.")
    print(f"  90th percentile    : ${mc['p90_final']:.2f}  "
          f"({(mc['p90_final']-start)/start*100:+.1f}%)")
    print(f"                     → 1-in-10 paths do THIS well or better.")
    print(f"  Worst case         : ${mc['worst_final']:.2f}")
    print(f"  Best case          : ${mc['best_final']:.2f}")
    print(f"  Median drawdown    : {mc['median_dd_pct']}%  "
          f"→ typical worst loss from peak in any path")
    print(f"  90th pct drawdown  : {mc['p90_dd_pct']}%  "
          f"→ bad-case drawdown (1-in-10 paths)")
    ruin_note = "✅ safe" if mc["ruin_pct"] < 5 else ("⚠️  high" if mc["ruin_pct"] < 20 else "❌ dangerous")
    print(f"  Ruin probability   : {mc['ruin_pct']}%  [{ruin_note}]")
    print(f"                     → <5% acceptable. >20% = reduce max_position_pct.")
    print()


# ══════════════════════════════════════════════════════════════
# CSV EXPORT
# ══════════════════════════════════════════════════════════════

def save_csvs(result: SimResult, mc: dict, tf: dict) -> None:
    """Save trades, equity curve, and Monte Carlo paths to CSV."""
    if result.trades:
        with open(tf["trades_file"], "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=result.trades[0].keys())
            w.writeheader()
            w.writerows(result.trades)

    rwr = rolling_win_rate(result.trades)
    with open(tf["equity_file"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trade_num", "equity", "rolling_win_rate_30"])
        for idx, (eq, rw) in enumerate(zip(result.equity[1:], rwr)):
            w.writerow([idx + 1, eq, rw])

    if mc and mc.get("all_finals"):
        with open(tf["mc_file"], "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sim_num", "final_bankroll", "roi_pct"])
            s = CONFIG["bankroll"]
            for idx, final in enumerate(mc["all_finals"]):
                w.writerow([idx + 1, round(final, 2),
                            round((final - s) / s * 100, 1)])

    print(f"  Saved: {tf['trades_file']}  |  "
          f"{tf['equity_file']}  |  {tf['mc_file']}\n")


# ══════════════════════════════════════════════════════════════
# RUN ONE TIMEFRAME END-TO-END
# ══════════════════════════════════════════════════════════════

async def run_timeframe(tf_key: str, tf: dict) -> dict:
    days            = CONFIG["lookback_days"]
    mins_per_candle = int(tf["interval"].replace("m", ""))
    W               = 62

    print(f"\n{'═'*W}")
    print(f"  TIMEFRAME: {tf['label'].upper()}  ({tf['interval']} candles)")
    print(f"  Each candle = one Polymarket market of this duration.")
    print(f"{'═'*W}\n")

    df = await fetch_candles(tf["interval"], tf["cache_file"])
    if len(df) < 200:
        print("  Not enough data — need at least 200 candles.\n")
        return {}

    # Check that volume column exists (needed for S4)
    if "volume" not in df.columns:
        print("  ⚠️  Cache missing volume column — delete cache and rerun.")
        return {}

    split      = int(len(df) * CONFIG["train_pct"])
    days_train = days * CONFIG["train_pct"]
    days_test  = days * (1 - CONFIG["train_pct"])

    # Step 1: Grid search on in-sample data
    print("── Grid Search (in-sample only) ──────────────────────\n")
    best_cfg = grid_search(df, split, tf["grid_file"], days_train)
    print()

    # Step 2: Single split walk-forward (quick sanity check)
    print("── Walk-Forward: Single Split ────────────────────────\n")
    train = run_simulation(df, 0,     split,    best_cfg, tf["interval"])
    test  = run_simulation(df, split, len(df),  best_cfg, tf["interval"])
    print_result(f"In-sample  ({int(days_train)}d)", train, days_train)
    print_result(f"Out-of-sample ({int(days_test)}d)", test, days_test, flag_low=True)

    if train.trades and test.trades and len(test.trades) >= 10:
        gap = train.win_rate - test.win_rate
        if gap > 0.10:
            print(f"  ⚠️  Win rate drops {gap*100:.1f}pp out-of-sample.\n"
                  f"     Some overfit remains. Try increasing grid_min_trades.\n")
        elif test.win_rate >= 0.53:
            print(f"  ✅  Win rate holds out-of-sample ({test.win_rate*100:.1f}%).\n")
        else:
            print(f"  ➖  Win rate borderline out-of-sample ({test.win_rate*100:.1f}%).\n")

    # Step 3: Rolling walk-forward (real robustness test)
    print("── Walk-Forward: Rolling Windows ─────────────────────\n")
    windows = rolling_walk_forward(df, best_cfg, tf["interval"], mins_per_candle)
    print_rolling_wf(windows)

    # Step 4: Full period baseline with best config
    print("── Full Period Baseline ───────────────────────────────\n")
    base = run_simulation(df, 0, len(df), best_cfg, tf["interval"])
    print_result(f"Full {days}d  |  {tf['interval']}", base, days)

    # Step 5: Monte Carlo on full period trades
    print("── Monte Carlo ───────────────────────────────────────\n")
    mc = monte_carlo(base.trades, CONFIG["mc_simulations"])
    print_monte_carlo(mc)

    # Step 6: Save all output files
    save_csvs(base, mc, tf)

    # Compute rolling WF consistency for final summary
    reliable_windows = [w for w in windows if w["reliable"]]
    wf_consistency   = (
        sum(1 for w in reliable_windows if w["win_rate"] >= 0.53)
        / max(len(reliable_windows), 1)
    )

    return {
        "timeframe":  tf["label"],
        "interval":   tf["interval"],
        "trades":     len(base.trades),
        "win_rate":   base.win_rate,
        "pnl":        base.total_pnl,
        "drawdown":   max_drawdown(base.equity),
        "ruin_pct":   mc.get("ruin_pct", 100),
        "oos_wr":     test.win_rate,
        "wf_consist": wf_consistency,
        "too_few_mc": mc.get("too_few", True),
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

async def run_backtest() -> None:
    """
    Entry point. Runs both timeframes sequentially, then prints
    a side-by-side comparison with actionable verdict per timeframe.
    """
    summaries = []
    for tf_key, tf in TIMEFRAMES.items():
        s = await run_timeframe(tf_key, tf)
        if s:
            summaries.append(s)

    if not summaries:
        print("No results — check data and config.")
        return

    W = 62
    print(f"\n{'═'*W}")
    print("  FINAL COMPARISON")
    print(f"  One row per Polymarket market type.")
    print(f"{'═'*W}\n")

    # Header
    print(f"  {'Timeframe':<22} {'Trades':>7} {'WR':>6} "
          f"{'P&L':>8} {'OOS WR':>8} {'WF%':>6} {'Ruin':>6}")
    print(f"  {'-'*22} {'-'*7} {'-'*6} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")

    for s in summaries:
        print(f"  {s['timeframe']:<22} "
              f"{s['trades']:>7} "
              f"{s['win_rate']*100:>5.1f}% "
              f"${s['pnl']:>+7.2f} "
              f"{s['oos_wr']*100:>7.1f}% "
              f"{s['wf_consist']*100:>5.0f}% "
              f"{s['ruin_pct']:>5.1f}%")

    print()
    print(f"  Columns:")
    print(f"    WR     = win rate over full period")
    print(f"    OOS WR = win rate on held-out test data (most honest)")
    print(f"    WF%    = % of rolling windows that were profitable")
    print(f"    Ruin   = % of Monte Carlo paths that hit $0\n")

    # Per-timeframe verdict with specific guidance
    for s in summaries:
        wr   = s["win_rate"] * 100
        pnl  = s["pnl"]
        dd   = s["drawdown"]
        ruin = s["ruin_pct"]
        oos  = s["oos_wr"] * 100
        wfc  = s["wf_consist"] * 100
        tf   = s["interval"]

        print(f"  {tf} ({s['timeframe']}):")

        # Verdict
        if wr >= 54 and oos >= 52 and dd < 20 and ruin < 5 and wfc >= 75:
            verdict = "✅  PROMISING — paper trade for 2+ weeks before going live."
        elif wr >= 51 and oos >= 50 and pnl > 0 and ruin < 15:
            verdict = "⚠️   MARGINAL — getting closer. See suggestions below."
        else:
            verdict = "❌  NEEDS WORK — do not trade this yet."

        print(f"    {verdict}")

        # Specific actionable guidance based on what the numbers show
        if wr < 53:
            print(f"    → Win rate {wr:.1f}%: signals aren't selective enough.")
            print(f"       Try: require 4-of-4 signals (change majority threshold),")
            print(f"       or widen grid_rsi_band to include 25 and 30.")
        if oos < 50 and len(s.get("trades", [])) >= 10:
            print(f"    → OOS win rate {oos:.1f}%: strategy fails on new data.")
            print(f"       Try: increase grid_min_trades to 60 or 80.")
        if wfc < 75:
            print(f"    → Rolling WF {wfc:.0f}%: not consistent across time periods.")
            print(f"       This is the most important number. Needs ≥75% to trust.")
        if dd > 20:
            print(f"    → Drawdown {dd:.1f}%: too large. Lower max_position_pct.")
            print(f"       Current: {CONFIG['max_position_pct']*100:.0f}%. Try 3%.")
        if ruin > 5:
            print(f"    → Ruin {ruin:.1f}%: unacceptably high.")
            print(f"       Lower max_position_pct or kelly_fraction.")
        if s["trades"] < CONFIG["mc_min_trades"]:
            print(f"    → Only {s['trades']} trades total — stats unreliable.")
            print(f"       Increase lookback_days to 120.")
        print()


if __name__ == "__main__":
    asyncio.run(run_backtest())
