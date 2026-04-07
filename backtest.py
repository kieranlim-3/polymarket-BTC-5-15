"""
backtest.py — Polymarket BTC Up/Down Bot (simplified)
======================================================
Run with:
    pip install aiohttp pandas numpy
    python backtest.py
"""

import asyncio
import aiohttp
import time
import math
import csv
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# ── CONFIG ────────────────────────────────────────────────
CONFIG = {
    "symbol":         "BTCUSDT",
    "interval":       "1m",
    "lookback_days":  30,

    # Signal params
    "roc_threshold":  0.0003,   # lowered from 0.001
    "ema_fast":       3,
    "ema_slow":       8,

    # Bankroll
    "bankroll":       100.0,
    "kelly_fraction": 0.5,
    "min_bet_usdc":   1.0,
    "max_bet_usdc":   20.0,

    # Output
    "trades_file":    "backtest_trades.csv",
    "equity_file":    "backtest_equity.csv",
}

# ── FETCH CANDLES ─────────────────────────────────────────
async def fetch_all_candles():
    total_needed = CONFIG["lookback_days"] * 24 * 60
    print(f"Fetching {CONFIG['lookback_days']} days of 1m candles from Binance...")
    all_raw = []
    end_time = int(time.time() * 1000)
    async with aiohttp.ClientSession() as session:
        while len(all_raw) < total_needed:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol":   CONFIG["symbol"],
                "interval": CONFIG["interval"],
                "limit":    1000,
                "endTime":  end_time,
            }
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as r:
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
        "open_time","open","high","low","close","volume",
        "close_time","qav","trades","tbbav","tbqav","ignore"
    ])
    df["open"]  = pd.to_numeric(df["open"],  errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["time"]  = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    return df[["time","open","close"]].dropna().reset_index(drop=True)

# ── SIGNALS ───────────────────────────────────────────────
def compute_signal(closes, window_open):
    if len(closes) < 3:
        return None

    current = closes[-1]

    # Signal 1: price vs window open
    s1 = "up" if current > window_open else "down"

    # Signal 2: ROC
    roc = (closes[-1] - closes[-2]) / closes[-2]
    if abs(roc) < CONFIG["roc_threshold"]:
        return None  # momentum not strong enough
    s2 = "up" if roc > 0 else "down"

    # Both must agree
    if s1 == s2:
        return s1
    return None

    current = closes[-1]

    # Signal 1: price vs window open
    s1 = "up" if current > window_open else "down"

    # Signal 2: ROC
    roc = (closes[-1] - closes[-2]) / closes[-2]
    s2 = None
    if abs(roc) >= CONFIG["roc_threshold"]:
        s2 = "up" if roc > 0 else "down"

    # Signal 3: EMA crossover
    s = pd.Series(closes)
    ema_f = s.ewm(span=CONFIG["ema_fast"], adjust=False).mean().iloc[-1]
    ema_s = s.ewm(span=CONFIG["ema_slow"], adjust=False).mean().iloc[-1]
    s3 = "up" if ema_f > ema_s else "down"

    # Only count non-None signals
    valid = [x for x in [s1, s2, s3] if x is not None]
    if not valid:
        return None

    ups   = valid.count("up")
    downs = valid.count("down")

    if ups == len(valid):
        return "up"
    elif downs == len(valid):
        return "down"
    return None

# ── KELLY ─────────────────────────────────────────────────
def kelly_bet(win_prob, market_price, bankroll):
    if market_price <= 0 or market_price >= 1:
        return 0.0
    b = (1.0 / market_price) - 1
    f = (win_prob * b - (1 - win_prob)) / b
    f = max(f, 0.0)
    bet = CONFIG["kelly_fraction"] * f * bankroll
    return round(min(max(bet, 0.0), CONFIG["max_bet_usdc"]), 2)

# ── STATS ─────────────────────────────────────────────────
def sharpe(returns):
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    std = arr.std()
    if std == 0:
        return 0.0
    return round((arr.mean() / std) * math.sqrt(12 * 24 * 252), 2)

def max_drawdown(equity):
    peak = equity[0]
    dd = 0.0
    for v in equity:
        peak = max(peak, v)
        dd = max(dd, (peak - v) / peak)
    return round(dd * 100, 2)

# ── MAIN ──────────────────────────────────────────────────
async def run_backtest():
    df = await fetch_all_candles()
    if len(df) < 50:
        print("Not enough data.")
        return

    bankroll = CONFIG["bankroll"]
    trades   = []
    equity   = [bankroll]
    returns  = []

    step  = 5
    min_i = CONFIG["ema_slow"] + 5

    skipped_signal = 0
    skipped_bet    = 0
    attempted      = 0

    print(f"Running backtest on {len(df)} candles...\n")

    for i in range(min_i, len(df) - step, step):
        attempted += 1

        window_open = float(df["open"].iloc[i])
        closes = df["close"].iloc[max(0, i - 14):i + 1].values

        direction = compute_signal(closes, window_open)

        if direction is None:
            skipped_signal += 1
            continue

        market_price = 0.48
        win_prob     = 0.55

        bet = kelly_bet(win_prob, market_price, bankroll)
        if bet < CONFIG["min_bet_usdc"]:
            skipped_bet += 1
            continue

        exit_i      = min(i + step, len(df) - 1)
        entry_close = df["close"].iloc[i]
        exit_close  = df["close"].iloc[exit_i]

        if direction == "up":
            won = exit_close > entry_close
        else:
            won = exit_close < entry_close

        pnl      = bet * ((1.0 / market_price) - 1.0) if won else -bet
        bankroll = round(bankroll + pnl, 2)

        equity.append(bankroll)
        returns.append(pnl / CONFIG["bankroll"])

        dt = df["time"].iloc[i].to_pydatetime()
        trades.append({
            "timestamp":   dt.strftime("%Y-%m-%d %H:%M UTC"),
            "direction":   direction.upper(),
            "entry_close": round(float(entry_close), 2),
            "exit_close":  round(float(exit_close), 2),
            "bet_usdc":    bet,
            "outcome":     "WIN" if won else "LOSS",
            "pnl":         round(pnl, 2),
            "bankroll":    bankroll,
        })

    print(f"  Windows attempted : {attempted}")
    print(f"  Skipped (signal)  : {skipped_signal}")
    print(f"  Skipped (bet size): {skipped_bet}")
    print(f"  Trades placed     : {len(trades)}\n")

    if not trades:
        print("Still no trades — printing debug info:")
        print("Sample closes:", df["close"].iloc[min_i:min_i+5].tolist())
        print("Sample open:  ", df["open"].iloc[min_i])
        return

    with open(CONFIG["trades_file"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=trades[0].keys())
        w.writeheader()
        w.writerows(trades)

    with open(CONFIG["equity_file"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trade_num", "equity"])
        for idx, eq in enumerate(equity):
            w.writerow([idx, eq])

    total  = len(trades)
    wins   = sum(1 for t in trades if t["outcome"] == "WIN")
    losses = total - wins
    wr     = wins / total * 100
    pnl    = bankroll - CONFIG["bankroll"]
    roi    = pnl / CONFIG["bankroll"] * 100
    sh     = sharpe(returns)
    dd     = max_drawdown(equity)

    W = 54
    print("=" * W)
    print("      BACKTEST RESULTS — Polymarket BTC Bot")
    print("=" * W)
    print(f"  Period       : {CONFIG['lookback_days']} days")
    print(f"  Total trades : {total}")
    print(f"  Wins         : {wins} ({wr:.1f}%)")
    print(f"  Losses       : {losses} ({100-wr:.1f}%)")
    print(f"  Start        : ${CONFIG['bankroll']:.2f}")
    print(f"  Final        : ${bankroll:.2f}")
    print(f"  P&L          : ${pnl:+.2f}")
    print(f"  ROI          : {roi:+.1f}%")
    print(f"  Sharpe       : {sh}")
    print(f"  Max drawdown : {dd}%")
    print("=" * W)

    if wr >= 55 and pnl > 0 and dd < 20:
        print("\n  ✅  PROMISING — consider paper trading first")
    elif wr >= 50 and pnl > 0:
        print("\n  ⚠️   MARGINAL — needs tuning")
    else:
        print("\n  ❌  NEEDS WORK — don't go live yet")

    print(f"\n  Trades → {CONFIG['trades_file']}")
    print(f"  Equity → {CONFIG['equity_file']}\n")

if __name__ == "__main__":
    asyncio.run(run_backtest())
