"""
backtest.py — Polymarket BTC Up/Down Bot
========================================
Run with:
    pip install aiohttp pandas numpy scipy
    python backtest.py

What this file does
───────────────────
1. Downloads 30 days of 1-minute BTC candles from Binance
2. Simulates trading Polymarket YES/NO binary markets using 3 signals
3. Sizes each bet with half-Kelly criterion
4. Reports full stats: ROI, Sharpe, drawdown, signal accuracy
5. Walk-forward validation  — splits data in-sample / out-of-sample
6. Vary step sensitivity    — runs step=3,5,10 to check robustness
7. Monte Carlo simulation   — 1000 random reshuffles of trade P&L
8. Saves trades + equity + MC CSVs for external charting

Bugs fixed from original
─────────────────────────
BUG 1 — Dead code: early return in compute_signal() killed EMA + voting
BUG 2 — win_prob and market_price were hardcoded constants (0.55, 0.48)
BUG 3 — Kelly ignored bet direction (NO bets used wrong probability)
BUG 4 — Sharpe annualisation factor was wrong (72576 regardless of frequency)
"""

import asyncio
import aiohttp
import time
import math
import csv
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

CONFIG = {
    "symbol":         "BTCUSDT",
    "interval":       "1m",
    "lookback_days":  30,

    # Signal params
    "roc_threshold":  0.0003,
    "ema_fast":       3,
    "ema_slow":       8,

    # Bankroll
    "bankroll":       100.0,
    "kelly_fraction": 0.5,
    "min_bet_usdc":   1.0,
    "max_bet_usdc":   20.0,

    # Realism
    "polymarket_fee": 0.02,    # 2% fee on winnings
    "slippage_pct":   0.005,   # 0.5% adverse slippage on entry price

    # Walk-forward split
    "train_pct":      0.70,    # 70% in-sample, 30% out-of-sample

    # Step sensitivity
    "step_variants":  [3, 5, 10],

    # Monte Carlo
    "mc_simulations": 1000,

    # Output files
    "trades_file":    "backtest_trades.csv",
    "equity_file":    "backtest_equity.csv",
    "mc_file":        "backtest_montecarlo.csv",
}


# ══════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════

async def fetch_all_candles() -> pd.DataFrame:
    """Download 1-minute OHLCV candles from Binance public API."""
    total_needed = CONFIG["lookback_days"] * 24 * 60
    print(f"Fetching {CONFIG['lookback_days']} days of 1m candles from Binance...")
    all_raw = []
    end_time = int(time.time() * 1000)

    async with aiohttp.ClientSession() as session:
        while len(all_raw) < total_needed:
            params = {
                "symbol":   CONFIG["symbol"],
                "interval": CONFIG["interval"],
                "limit":    1000,
                "endTime":  end_time,
            }
            async with session.get(
                "https://api.binance.com/api/v3/klines",
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as r:
                if r.status != 200:
                    break
                chunk = await r.json()
            if not chunk:
                break
            all_raw = chunk + all_raw
            end_time = chunk[0][0] - 1
            print(f"  {len(all_raw):>6}/{total_needed} candles...", end="\r")
            await asyncio.sleep(0.2)

    print(f"\nFetched {len(all_raw)} candles.\n")
    df = pd.DataFrame(all_raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbbav", "tbqav", "ignore",
    ])
    df["open"]  = pd.to_numeric(df["open"],  errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["time"]  = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    return df[["time", "open", "close"]].dropna().reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════

def compute_signal(closes: np.ndarray, window_open: float):
    """
    Returns (direction, confidence) where direction is "up" / "down" / None.

    Three signals ALL must agree:
      S1 — current price vs candle open
      S2 — rate of change (ROC) momentum
      S3 — EMA fast vs EMA slow crossover

    ── BUG 1 FIXED ──────────────────────────────────────────────
    The original code had `return s1` halfway through the function.
    Python stops executing a function the moment it hits `return`.
    So S3 (EMA crossover) and the voting block that followed it
    were completely unreachable — they literally never ran, ever.
    The fix: remove the early return so all three signals are computed
    before any decision is made.
    ──────────────────────────────────────────────────────────────
    """
    if len(closes) < 3:
        return None, 0.0

    current = closes[-1]

    # Signal 1: is price above the candle's opening price?
    s1 = "up" if current > window_open else "down"

    # Signal 2: recent momentum (rate of change)
    roc = (closes[-1] - closes[-2]) / closes[-2]
    if abs(roc) < CONFIG["roc_threshold"]:
        return None, 0.0          # not enough momentum — skip this bar
    s2 = "up" if roc > 0 else "down"

    # Signal 3: EMA fast vs EMA slow  ← NOW ACTUALLY RUNS
    s = pd.Series(closes)
    ema_fast = s.ewm(span=CONFIG["ema_fast"], adjust=False).mean().iloc[-1]
    ema_slow = s.ewm(span=CONFIG["ema_slow"], adjust=False).mean().iloc[-1]
    s3 = "up" if ema_fast > ema_slow else "down"

    # All three must agree — no majority, unanimous only
    signals = [s1, s2, s3]
    if signals.count("up") == 3:
        confidence = min(abs(roc) / CONFIG["roc_threshold"] / 10.0, 1.0)
        return "up", confidence
    elif signals.count("down") == 3:
        confidence = min(abs(roc) / CONFIG["roc_threshold"] / 10.0, 1.0)
        return "down", confidence

    return None, 0.0


# ══════════════════════════════════════════════════════════════
# MARKET PRICE SIMULATION
# ══════════════════════════════════════════════════════════════

def simulate_market_price(direction: str, roc: float) -> float:
    """
    Simulate the Polymarket YES order-book price.

    ── BUG 2 FIXED ──────────────────────────────────────────────
    The original hardcoded market_price = 0.48 for every single trade.
    That's like a casino that always pays out 2.08x on every bet
    regardless of what the odds actually are.  Every trade had identical
    economics so the backtest told you nothing real about the strategy.
    The fix: model a price that varies with signal strength and noise,
    assuming the market has partially priced in what we can see.
    ──────────────────────────────────────────────────────────────
    """
    rng   = np.random.default_rng(seed=int(abs(roc) * 1e9) % (2 ** 31))
    noise = rng.uniform(-0.04, 0.04)
    nudge = abs(roc) * 20   # market partially prices in the signal already

    base = (0.50 + nudge) if direction == "up" else (0.50 - nudge)
    return round(float(np.clip(base + noise, 0.35, 0.65)), 3)


# ══════════════════════════════════════════════════════════════
# POSITION SIZING (KELLY)
# ══════════════════════════════════════════════════════════════

def kelly_bet(
    win_prob_yes: float,
    market_price: float,
    bankroll:     float,
    direction:    str,
) -> float:
    """
    Half-Kelly bet sizing for a binary Polymarket market.

    ── BUG 3 FIXED ──────────────────────────────────────────────
    The original always fed win_prob = P(price goes UP) into Kelly,
    even when the bet was on NO (price goes DOWN).

    Simple analogy: imagine you think there's a 60% chance of rain.
    If you bet "YES rain", your chance of winning is 60%.
    If you bet "NO rain", your chance of winning is 40% — not 60%.
    The original code always used 60% regardless of which side you bet,
    so NO bets were systematically oversized and miscalculated.
    The fix: flip the probability when direction is "down".
    ──────────────────────────────────────────────────────────────
    """
    slipped_price = min(market_price + CONFIG["slippage_pct"], 0.95)

    # P(this bet wins) — flipped for NO/down bets
    p_win = (1.0 - win_prob_yes) if direction == "down" else win_prob_yes

    if not (0 < slipped_price < 1):
        return 0.0

    b = (1.0 / slipped_price) - 1          # net profit per $1 staked if we win
    f = (p_win * b - (1 - p_win)) / b      # full Kelly fraction
    f = max(f, 0.0)                          # clamp — no negative bets

    bet = CONFIG["kelly_fraction"] * f * bankroll
    return round(min(max(bet, 0.0), CONFIG["max_bet_usdc"]), 2)


# ══════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════

def sharpe(returns: list, trades_per_day: float) -> float:
    """
    Annualised Sharpe ratio.

    ── BUG 4 FIXED ──────────────────────────────────────────────
    The original used math.sqrt(12 * 24 * 252) = sqrt(72576) as a fixed
    scaling factor regardless of how often trades actually happened.

    Here is what that means in plain English:
    The Sharpe ratio measures how much return you get per unit of risk,
    scaled up to a full year. To scale up correctly you need to know how
    many trades happen per year. If you trade once a day, you multiply by
    sqrt(252). If you trade 10 times a day, you multiply by sqrt(2520).
    The original always used sqrt(72576) — equivalent to ~288 trades/day —
    even if the strategy only placed 5 trades that day. This wildly
    inflated the Sharpe number, making it look much better than it was.
    The fix: compute actual trades per day and use that to scale.
    ──────────────────────────────────────────────────────────────
    """
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    std = arr.std()
    if std == 0:
        return 0.0
    return round((arr.mean() / std) * math.sqrt(trades_per_day * 252), 2)


def max_drawdown(equity: list) -> float:
    """Largest peak-to-trough percentage drop in the equity curve."""
    peak, dd = equity[0], 0.0
    for v in equity:
        peak = max(peak, v)
        dd   = max(dd, (peak - v) / peak)
    return round(dd * 100, 2)


def rolling_win_rate(trades: list, window: int = 50) -> list:
    """Win rate over a sliding window of `window` trades."""
    outcomes = [1 if t["outcome"] == "WIN" else 0 for t in trades]
    rates = []
    for i in range(len(outcomes)):
        chunk = outcomes[max(0, i - window + 1) : i + 1]
        rates.append(round(sum(chunk) / len(chunk), 3))
    return rates


# ══════════════════════════════════════════════════════════════
# CORE SIMULATION LOOP
# ══════════════════════════════════════════════════════════════

@dataclass
class SimResult:
    trades:            list = field(default_factory=list)
    equity:            list = field(default_factory=list)
    returns:           list = field(default_factory=list)
    skipped_signal:    int  = 0
    skipped_neg_kelly: int  = 0
    skipped_bet:       int  = 0
    attempted:         int  = 0


def run_simulation(
    df:        pd.DataFrame,
    step:      int,
    start_idx: int,
    end_idx:   int,
) -> SimResult:
    """
    Simulate trading on df[start_idx:end_idx] with a given candle step.
    Returns a SimResult with full trade log and equity curve.
    """
    result   = SimResult()
    bankroll = CONFIG["bankroll"]
    result.equity.append(bankroll)

    min_i = max(start_idx, CONFIG["ema_slow"] + 5)

    for i in range(min_i, end_idx - step, step):
        result.attempted += 1

        window_open = float(df["open"].iloc[i])
        closes      = df["close"].iloc[max(0, i - 14) : i + 1].values
        roc         = (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 else 0.0

        direction, confidence = compute_signal(closes, window_open)
        if direction is None:
            result.skipped_signal += 1
            continue

        win_prob_yes = 0.52 + confidence * 0.13   # scales 0.52 → 0.65
        market_price = simulate_market_price(direction, roc)

        bet = kelly_bet(win_prob_yes, market_price, bankroll, direction)
        if bet <= 0:
            result.skipped_neg_kelly += 1
            continue
        if bet < CONFIG["min_bet_usdc"]:
            result.skipped_bet += 1
            continue

        exit_i      = min(i + step, end_idx - 1)
        entry_close = float(df["close"].iloc[i])
        exit_close  = float(df["close"].iloc[exit_i])

        won = (exit_close > entry_close) if direction == "up" else (exit_close < entry_close)

        slipped = min(market_price + CONFIG["slippage_pct"], 0.95)
        if won:
            gross = bet * ((1.0 / slipped) - 1.0)
            pnl   = gross * (1 - CONFIG["polymarket_fee"])
        else:
            pnl = -bet

        bankroll = max(round(bankroll + pnl, 2), 0.0)
        result.equity.append(bankroll)
        result.returns.append(pnl / CONFIG["bankroll"])

        dt = df["time"].iloc[i].to_pydatetime()
        result.trades.append({
            "timestamp":    dt.strftime("%Y-%m-%d %H:%M UTC"),
            "step":         step,
            "direction":    direction.upper(),
            "confidence":   round(confidence, 3),
            "market_price": market_price,
            "win_prob":     round(win_prob_yes, 3),
            "entry_close":  round(entry_close, 2),
            "exit_close":   round(exit_close, 2),
            "bet_usdc":     bet,
            "outcome":      "WIN" if won else "LOSS",
            "pnl":          round(pnl, 2),
            "bankroll":     bankroll,
        })

    return result


def print_result(label: str, result: SimResult, days: float) -> None:
    """Pretty-print one SimResult to the console."""
    trades = result.trades
    if not trades:
        print(f"  [{label}] No trades generated.\n")
        return

    total = len(trades)
    wins  = sum(1 for t in trades if t["outcome"] == "WIN")
    wr    = wins / total * 100
    final = result.equity[-1]
    pnl   = final - CONFIG["bankroll"]
    roi   = pnl / CONFIG["bankroll"] * 100
    tpd   = total / max(days, 1)
    sh    = sharpe(result.returns, tpd)
    dd    = max_drawdown(result.equity)

    W = 58
    print("=" * W)
    print(f"  {label}")
    print("=" * W)
    print(f"  Trades     : {total}  ({tpd:.1f}/day)")
    print(f"  Win rate   : {wr:.1f}%  ({wins}W / {total - wins}L)")
    print(f"  P&L        : ${pnl:+.2f}  (ROI {roi:+.1f}%)")
    print(f"  Sharpe     : {sh}")
    print(f"  Drawdown   : {dd}%")
    if result.skipped_signal or result.skipped_neg_kelly or result.skipped_bet:
        print(f"  Skipped    : {result.skipped_signal} signal | "
              f"{result.skipped_neg_kelly} neg-Kelly | "
              f"{result.skipped_bet} min-bet")
    print()


# ══════════════════════════════════════════════════════════════
# IMPROVEMENT 1 — WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════

def walk_forward(df: pd.DataFrame, step: int) -> tuple:
    """
    Split the dataset into in-sample (train) and out-of-sample (test).

    Why this matters:
    If you tune your signal thresholds on all 30 days and then also
    report performance on those same 30 days, you are marking your own
    exam — the result will look better than it really is.

    Walk-forward forces you to test on data the model has never seen:
      70% → in-sample  (you can tune parameters here)
      30% → out-of-sample (honest test — do NOT tune on this)

    A strategy that works in-sample but falls apart out-of-sample
    is overfit and not ready for live trading.
    """
    split = int(len(df) * CONFIG["train_pct"])
    train = run_simulation(df, step, 0,     split)
    test  = run_simulation(df, step, split, len(df))
    return train, test


# ══════════════════════════════════════════════════════════════
# IMPROVEMENT 2 — STEP SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════

def step_sensitivity(df: pd.DataFrame) -> dict:
    """
    Run the full simulation for each step value in step_variants.

    Why this matters:
    If the strategy only looks good at step=5 but falls apart at step=3
    or step=10, that's a red flag — it means you found a lucky number,
    not a real edge. A genuine edge should be roughly consistent
    regardless of whether you check the price every 3, 5, or 10 minutes.
    This test exposes strategies that only work by luck of timing.
    """
    results = {}
    for step in CONFIG["step_variants"]:
        results[step] = run_simulation(df, step, 0, len(df))
    return results


# ══════════════════════════════════════════════════════════════
# IMPROVEMENT 3 — MONTE CARLO SIMULATION
# ══════════════════════════════════════════════════════════════

def monte_carlo(trades: list, n_sims: int = 1000) -> dict:
    """
    Randomly reshuffle the trade P&L sequence 1000 times and replay equity.

    Why this matters:
    Your actual trade sequence is just ONE of trillions of possible orderings.
    Maybe you got lucky — your early trades happened to be winners, which
    padded your bankroll before the losses came. Or maybe you got unlucky.

    Monte Carlo answers: across all possible orderings of these exact trades,
    what is the distribution of outcomes?
      - What happens in the median case?
      - What happens in the worst 10% of cases?
      - How often does the bankroll hit zero (ruin)?

    A strategy where 30% of paths lead to ruin is very different from
    one where ruin only happens in 1% of paths — even if both have the
    same total P&L in the actual observed sequence.
    """
    if not trades:
        return {}

    pnl_list = [t["pnl"] for t in trades]
    bankroll  = CONFIG["bankroll"]
    rng       = np.random.default_rng(seed=42)

    final_bankrolls: list[float] = []
    max_drawdowns:   list[float] = []
    ruin_count = 0

    for _ in range(n_sims):
        shuffled = rng.permutation(pnl_list)
        equity   = [bankroll]
        bal      = bankroll
        ruined   = False

        for pnl in shuffled:
            bal = max(round(bal + pnl, 2), 0.0)
            equity.append(bal)
            if bal == 0.0:
                ruined = True

        final_bankrolls.append(equity[-1])
        max_drawdowns.append(max_drawdown(equity))
        if ruined:
            ruin_count += 1

    arr    = np.array(final_bankrolls)
    dd_arr = np.array(max_drawdowns)

    return {
        "simulations":   n_sims,
        "ruin_pct":      round(ruin_count / n_sims * 100, 1),
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
        print("  Monte Carlo: no trades to simulate.\n")
        return
    start = CONFIG["bankroll"]
    W = 58
    print("=" * W)
    print("  MONTE CARLO  —  1 000 random reshuffles of trade order")
    print("=" * W)
    print(f"  Starting bankroll  : ${start:.2f}")
    print(f"  Median outcome     : ${mc['median_final']:.2f}  "
          f"({(mc['median_final'] - start) / start * 100:+.1f}%)")
    print(f"  10th percentile    : ${mc['p10_final']:.2f}  "
          f"({(mc['p10_final'] - start) / start * 100:+.1f}%)")
    print(f"  90th percentile    : ${mc['p90_final']:.2f}  "
          f"({(mc['p90_final'] - start) / start * 100:+.1f}%)")
    print(f"  Worst case         : ${mc['worst_final']:.2f}")
    print(f"  Best case          : ${mc['best_final']:.2f}")
    print(f"  Median drawdown    : {mc['median_dd_pct']}%")
    print(f"  90th pct drawdown  : {mc['p90_dd_pct']}%")
    print(f"  Ruin probability   : {mc['ruin_pct']}%  "
          f"(hit $0 in {mc['ruin_pct']}% of paths)")
    print()


# ══════════════════════════════════════════════════════════════
# CSV EXPORT
# ══════════════════════════════════════════════════════════════

def save_csvs(base_result: SimResult, mc: dict) -> None:
    """Save trades, equity curve (with rolling win rate), and MC outcomes."""

    if base_result.trades:
        with open(CONFIG["trades_file"], "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=base_result.trades[0].keys())
            w.writeheader()
            w.writerows(base_result.trades)

    rwr = rolling_win_rate(base_result.trades)
    with open(CONFIG["equity_file"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trade_num", "equity", "rolling_win_rate_50"])
        for idx, (eq, rw) in enumerate(zip(base_result.equity[1:], rwr)):
            w.writerow([idx + 1, eq, rw])

    if mc and mc.get("all_finals"):
        with open(CONFIG["mc_file"], "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sim_num", "final_bankroll", "roi_pct"])
            start = CONFIG["bankroll"]
            for idx, final in enumerate(mc["all_finals"]):
                w.writerow([
                    idx + 1,
                    round(final, 2),
                    round((final - start) / start * 100, 1),
                ])

    print(f"  Trades saved  → {CONFIG['trades_file']}")
    print(f"  Equity saved  → {CONFIG['equity_file']}")
    print(f"  MC saved      → {CONFIG['mc_file']}\n")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

async def run_backtest() -> None:
    df = await fetch_all_candles()
    if len(df) < 50:
        print("Not enough data — aborting.")
        return

    default_step = 5
    days         = CONFIG["lookback_days"]

    # ── 1. Baseline run ───────────────────────────────────
    print("── BASELINE (step=5, full 30 days) ──────────────────\n")
    base = run_simulation(df, default_step, 0, len(df))
    print_result(f"Full period  |  step={default_step}", base, days)

    # ── 2. Walk-forward validation ────────────────────────
    print("── WALK-FORWARD VALIDATION ───────────────────────────\n")
    train, test = walk_forward(df, default_step)
    train_days  = days * CONFIG["train_pct"]
    test_days   = days * (1 - CONFIG["train_pct"])
    print_result(f"In-sample  (first {int(train_days)}d)  |  step={default_step}", train, train_days)
    print_result(f"Out-of-sample (last {int(test_days)}d) |  step={default_step}", test,  test_days)

    if base.trades and test.trades:
        base_wr = sum(1 for t in base.trades if t["outcome"] == "WIN") / len(base.trades)
        test_wr = sum(1 for t in test.trades if t["outcome"] == "WIN") / len(test.trades)
        if test_wr < base_wr - 0.10:
            print("  ⚠️  Win rate drops >10pp out-of-sample — likely overfit.\n")

    # ── 3. Step sensitivity ───────────────────────────────
    print("── STEP SENSITIVITY ANALYSIS ─────────────────────────\n")
    step_results = step_sensitivity(df)
    for step, res in step_results.items():
        print_result(f"step={step}", res, days)

    pnls = [
        res.equity[-1] - CONFIG["bankroll"]
        for res in step_results.values()
        if res.equity
    ]
    all_positive = all(p > 0 for p in pnls)
    if all_positive:
        print("  ✅  P&L positive across all step sizes — robust.\n")
    else:
        print("  ❌  P&L flips sign across step sizes — fragile.\n")

    # ── 4. Monte Carlo ────────────────────────────────────
    print("── MONTE CARLO SIMULATION ────────────────────────────\n")
    mc = monte_carlo(base.trades, CONFIG["mc_simulations"])
    print_monte_carlo(mc)

    # ── 5. Verdict ────────────────────────────────────────
    W = 58
    print("=" * W)
    print("  OVERALL VERDICT")
    print("=" * W)

    total = len(base.trades)
    if total == 0:
        print("  ❌  No trades — lower roc_threshold or min_bet_usdc.")
    else:
        wins  = sum(1 for t in base.trades if t["outcome"] == "WIN")
        wr    = wins / total * 100
        final = base.equity[-1]
        pnl   = final - CONFIG["bankroll"]
        dd    = max_drawdown(base.equity)
        ruin  = mc.get("ruin_pct", 100)

        if wr >= 55 and pnl > 0 and dd < 20 and ruin < 5 and all_positive:
            print("  ✅  PROMISING — paper trade ≥2 weeks before going live.")
        elif wr >= 50 and pnl > 0 and ruin < 15:
            print("  ⚠️   MARGINAL — needs more tuning.")
        else:
            print("  ❌  NEEDS WORK — do not trade live yet.")
    print()

    # ── 6. Save files ─────────────────────────────────────
    save_csvs(base, mc)


if __name__ == "__main__":
    asyncio.run(run_backtest())
