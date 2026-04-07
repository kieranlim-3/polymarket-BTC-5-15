# Polymarket Latency Arbitrage Bot

A production-ready async Python bot that detects and exploits latency
arbitrage opportunities between Binance spot prices and Polymarket
binary prediction markets.

> **⚠️ PAPER MODE IS THE DEFAULT. Live trading requires three explicit CLI flags
> and a typed confirmation prompt. Read this README before enabling live mode.**

---

## Architecture

```
bot.py                 ← Async orchestrator, main event loop
src/
  config.py            ← Env var loading + validation (dotenv)
  logger.py            ← Rotating structured log (50 MB, JSON)
  database.py          ← SQLite async (paper trade persistence)
  telegram_notifier.py ← Async alert queue (fire-and-forget)
  price_feed.py        ← Binance WebSocket + stale-data guard
  polymarket_client.py ← REST market discovery + order placement
  edge_detector.py     ← Log-normal probability model + edge scan
  position_sizer.py    ← Half-Kelly with 8% hard cap
  risk_manager.py      ← Daily halt, drawdown halt, consec-loss pause
```

### Data Flow

```
Binance WS ──► price_feed ──► edge_detector ──► executable opps
                                                      │
Polymarket REST ──► market list ──────────────────────┘
                                                      │
                                               risk_manager.check()
                                                      │
                                              position_sizer.size()
                                                      │
                                           polymarket_client.place_order()
                                                      │
                                    ┌─────────────────┴──────────────────┐
                               telegram alert                     DB log (paper)
```

---

## Setup

### 1. Python version

Requires Python **3.11+**.

```bash
python3 --version
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure credentials

```bash
cp .env.example .env
$EDITOR .env        # fill in all required values
```

Required env vars:

| Variable            | Description |
|---------------------|-------------|
| `POLY_API_KEY`      | Polymarket CLOB L2 API key |
| `POLY_PRIVATE_KEY`  | Ethereum private key (0x-prefixed) |
| `TELEGRAM_TOKEN`    | Telegram bot token from @BotFather |
| `TELEGRAM_CHAT_ID`  | Your chat/group ID |
| `ALCHEMY_RPC_URL`   | Polygon Mainnet Alchemy endpoint |

### 4. Run in paper mode (recommended first)

```bash
python bot.py
```

Paper mode:
- No real orders are placed
- All trades logged to `paper_trades.db` (SQLite)
- Full Telegram alerts sent (labelled `📄 PAPER`)
- Full risk management applied to simulated balance

### 5. Enable live trading

Only after validating paper mode performance:

```bash
python bot.py --live --confirm --i-understand-risks
```

You will be prompted to type `YES I ACCEPT` at the terminal.

---

## Risk Controls

| Control | Trigger | Behaviour |
|---------|---------|-----------|
| Daily loss halt | Day P&L < −20% of day-start balance | Permanent halt; Telegram alert; manual restart |
| Drawdown halt | Portfolio < 60% of all-time high | Permanent halt; Telegram alert; manual restart |
| Consecutive loss pause | 5 losses in a row | 30-minute pause; auto-resume; Telegram alert |
| Stale price pause | No Binance update for >10 s | Trading suspended; auto-resumes when feed recovers |

**Permanent halts require you to restart the process.** This is by design.

---

## Position Sizing

Each trade uses **half-Kelly Criterion**:

```
Kelly fraction  f* = (p·b − q) / b
Half-Kelly      f  = f* / 2
Position size   $  = f · portfolio_balance
Hard cap        min(f, 8%) of portfolio
```

Where:
- `p` = our model's estimated win probability
- `b` = net odds implied by the market price
- `q = 1 − p`

---

## Edge Detection

The bot estimates win probability using a **log-normal price model**:

```
P(S_τ > K) = Φ( ln(S₀/K) / (σ√Δt) )   for "above" markets
```

Where:
- `S₀` = current Binance spot price
- `K`  = market threshold (from question parsing)
- `σ`  = 5-minute rolling annualised volatility
- `Δt` = time to market expiry in years

Edge thresholds:
- `< 5%`  → ignored (noise)
- `≥ 5%`  → monitored
- `≥ 8%`  → executed

---

## Monitoring

**Telegram alerts sent on:**
- Every trade entry (market, side, size, price, edge)
- Every trade exit/settlement (market, P&L, hold time)
- Every error
- Every halt trigger
- Hourly heartbeat (balance, trades today, win rate)

**Rotating log file:**
- Location: `logs/bot.log`
- Format: JSON (structured fields for easy grep/parsing)
- Max: 50 MB per file, 5 rotating backups

**SQLite database (paper mode):**
- Location: `paper_trades.db`
- Tables: `paper_trades`, `daily_summary`
- Query example:
  ```sql
  SELECT asset, side, edge, pnl_usdc
  FROM paper_trades
  WHERE DATE(created_at, 'unixepoch') = DATE('now')
  ORDER BY created_at DESC;
  ```

---

## Execution Timing Budget

| Stage | Target |
|-------|--------|
| Edge detection | < 200 ms |
| Risk + sizing | < 50 ms |
| Order placement | < 500 ms |
| **Total** | **< 800 ms** |

Orders that exceed the 800 ms budget are cancelled and logged.

---

## WebSocket Reconnection

| Retry | Wait |
|-------|------|
| 1 | 1.5 s |
| 2 | 2.25 s |
| 3 | 3.4 s |
| … | … |
| 10 | ~57 s |

After 10 failed retries the feed stops and all trading is suspended
until the process is restarted.

---

## Adjustable Thresholds (env vars)

Override any default without editing source code:

```bash
BOT_MIN_EDGE_PCT=5.0          # minimum detectable edge %
BOT_EXEC_EDGE_PCT=8.0         # minimum execution edge %
BOT_MAX_PORTFOLIO_PCT=8.0     # Kelly hard cap %
BOT_MIN_LIQUIDITY_USD=50000   # minimum market liquidity
BOT_DAILY_HALT_PCT=20.0       # daily loss halt %
BOT_DRAWDOWN_HALT_PCT=40.0    # portfolio drawdown halt %
```

---

## Disclaimer

This software is for educational purposes.
Prediction markets carry significant financial risk.
Past performance in paper mode does not guarantee live results.
The log-normal probability model is a simplification; real markets
exhibit fat tails, stochastic volatility, and liquidity risk.
**Use at your own risk.**
