#!/usr/bin/env python3
"""
bot.py  ─  Polymarket Latency Arbitrage Bot
════════════════════════════════════════════
Entry point.  All trading logic is coordinated here.

Usage
─────
Paper mode (safe default):
    python bot.py

Live mode (requires all three flags):
    python bot.py --live --confirm --i-understand-risks

Flags
─────
  --live                Enable live order submission to Polymarket
  --confirm             Second acknowledgement required for live mode
  --i-understand-risks  Third acknowledgement; confirms real-money risk

Without all three flags the bot runs in paper mode regardless of env vars.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import signal
import sys
import time
from typing import Optional

# ── Internal modules ─────────────────────────────────────────────────────────
from src.config import load_settings, Settings
from src.logger import setup_logging, get_logger
from src.telegram_notifier import TelegramNotifier, AlertLevel
from src.price_feed import PriceFeed, PriceTick
from src.polymarket_client import PolymarketClient, MarketInfo, OrderReceipt
from src.edge_detector import EdgeDetector, Opportunity
from src.position_sizer import PositionSizer
from src.risk_manager import RiskManager, TradeResult, HaltReason
from src.database import Database

# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket Latency Arbitrage Bot")
    parser.add_argument("--live",              action="store_true", help="Enable live trading")
    parser.add_argument("--confirm",           action="store_true", help="Confirm live trading")
    parser.add_argument("--i-understand-risks",action="store_true",
                        dest="understand_risks", help="Acknowledge real-money risk")
    return parser.parse_args()


def resolve_paper_mode(args: argparse.Namespace) -> bool:
    """Return True (paper mode) unless ALL three live flags are set."""
    if args.live and args.confirm and args.understand_risks:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Active-position tracker (for paper P&L simulation)
# ─────────────────────────────────────────────────────────────────────────────

class ActivePosition:
    """Tracks an open paper/live position until market resolves."""

    def __init__(
        self,
        receipt: OrderReceipt,
        opportunity: Opportunity,
        expiry_ts: float,
    ) -> None:
        self.receipt = receipt
        self.opportunity = opportunity
        self.expiry_ts = expiry_ts
        self.entered_at = time.time()

    def is_expired(self) -> bool:
        return time.time() >= self.expiry_ts

    def simulate_pnl(self, final_spot_price: float) -> float:
        """
        Approximate P&L for paper mode based on whether the outcome resolved
        in our favour.  Returns USDC profit or loss.
        """
        opp = self.opportunity
        market = opp.market
        side = opp.recommended_side
        size = self.receipt.size_usdc
        entry_price = self.receipt.price

        # Determine if outcome resolved YES or NO
        if market.direction == "above":
            yes_won = final_spot_price > market.threshold
        else:
            yes_won = final_spot_price < market.threshold

        if (side == "YES" and yes_won) or (side == "NO" and not yes_won):
            # We won: receive $1 per share, paid entry_price per share
            shares = size / entry_price
            return round(shares * (1.0 - entry_price), 2)  # profit per share * shares
        else:
            # We lost: forfeit the entire stake
            return round(-size, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Main Bot class
# ─────────────────────────────────────────────────────────────────────────────

class ArbitrageBot:

    def __init__(self, settings: Settings, paper_mode: bool) -> None:
        self._s = settings
        self._paper = paper_mode
        self._running = False
        self._log = get_logger(__name__)

        # Component instances
        self._telegram = TelegramNotifier(
            settings.telegram_token,
            settings.telegram_chat_id,
            paper_mode=paper_mode,
        )
        self._price_feed = PriceFeed(
            ws_url=settings.binance_ws_url,
            streams=settings.binance_streams,
            stale_seconds=settings.price_stale_seconds,
            max_retries=settings.ws_max_retries,
            backoff_base=settings.ws_backoff_base,
        )
        self._poly = PolymarketClient(settings, paper_mode=paper_mode)
        self._edge = EdgeDetector(
            min_detectable_edge=settings.min_detectable_edge_pct / 100.0,
            min_exec_edge=settings.min_exec_edge_pct / 100.0,
        )
        self._sizer = PositionSizer(
            kelly_fraction=settings.kelly_fraction,
            max_portfolio_pct=settings.max_position_pct / 100.0,
        )
        self._risk = RiskManager(
            daily_halt_pct=settings.daily_halt_pct,
            drawdown_halt_pct=settings.drawdown_halt_pct,
            consec_loss_count=settings.consec_loss_pause_count,
            consec_loss_pause_minutes=settings.consec_loss_pause_minutes,
        )
        self._db = Database(settings.db_path)

        # State
        self._markets: list[MarketInfo] = []
        self._active_positions: list[ActivePosition] = []
        self._last_market_refresh: float = 0.0
        self._market_refresh_interval: float = 60.0   # refresh market list every 60 s
        self._last_heartbeat: float = 0.0
        self._last_midnight_reset: str = ""             # date string "YYYY-MM-DD"

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._log.info("bot.starting", paper=self._paper)

        await self._db.start()
        await self._telegram.start()
        await self._poly.start()

        # Initial balance
        balance = await self._poly.get_portfolio_usdc()
        self._risk.initialise(balance)

        # Register price callback
        self._price_feed.subscribe(self._on_price_tick)
        await self._price_feed.start()

        mode = "PAPER" if self._paper else "LIVE"
        self._telegram.send(
            AlertLevel.INFO,
            f"Bot started in {mode} mode.\n"
            f"Balance: ${balance:,.2f}\n"
            f"Min exec edge: {self._s.min_exec_edge_pct}%",
        )
        self._log.info("bot.started", balance=balance, mode=mode)
        self._running = True

    async def stop(self, reason: str = "Manual shutdown") -> None:
        self._log.info("bot.stopping", reason=reason)
        self._running = False
        await self._price_feed.stop()
        await self._poly.stop()
        await self._telegram.stop()
        await self._db.stop()
        self._log.info("bot.stopped")

    # ── Main Loop ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main event loop — runs until shutdown signal received."""
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.exception("bot.tick_error", exc=str(exc))
                self._telegram.error("Main loop error", exc)
                await asyncio.sleep(5)

            await asyncio.sleep(0.1)  # 100 ms main loop cadence

    async def _tick(self) -> None:
        now = time.time()

        # ── Midnight reset ─────────────────────────────────────────────────
        today = datetime.date.today().isoformat()
        if today != self._last_midnight_reset:
            await self._midnight_reset(today)

        # ── Heartbeat ─────────────────────────────────────────────────────
        if now - self._last_heartbeat >= self._s.heartbeat_interval_seconds:
            await self._send_heartbeat()
            self._last_heartbeat = now

        # ── Check risk manager (permanent halts) ───────────────────────────
        if self._risk.is_permanently_halted:
            # Already halted — just wait; manual restart required
            await asyncio.sleep(60)
            return

        # ── Stale data guard ───────────────────────────────────────────────
        if await self._price_feed.is_stale():
            self._log.warning("bot.price_data_stale_pausing_trading")
            await asyncio.sleep(2)
            return

        # ── Refresh market list ────────────────────────────────────────────
        if now - self._last_market_refresh >= self._market_refresh_interval:
            await self._refresh_markets()
            self._last_market_refresh = now

        if not self._markets:
            await asyncio.sleep(5)
            return

        # ── Resolve expired paper positions ────────────────────────────────
        await self._settle_expired_positions()

        # ── Scan for opportunities ─────────────────────────────────────────
        spot_prices = await self._price_feed.get_all_prices()
        if not spot_prices:
            return

        opportunities = self._edge.scan(self._markets, spot_prices)
        executable = self._edge.find_executable(opportunities)

        if executable:
            self._log.info("bot.opportunities_found", count=len(executable))

        for opp in executable:
            await self._attempt_trade(opp, now)

    # ── Price tick callback ────────────────────────────────────────────────

    async def _on_price_tick(self, tick: PriceTick) -> None:
        """Called on every Binance price update."""
        self._edge.update_price(tick.symbol, tick.price, tick.timestamp)

    # ── Trade Execution ────────────────────────────────────────────────────

    async def _attempt_trade(self, opp: Opportunity, now: float) -> None:
        detection_ts = time.time()

        # ── Risk check ─────────────────────────────────────────────────────
        balance = await self._poly.get_portfolio_usdc()
        self._risk.update_balance(balance)

        halt = self._risk.check(balance)
        if halt:
            if halt.permanent:
                self._telegram.halt(halt.message)
                self._log.error("bot.trading_halted", reason=halt.reason.name)
            else:
                # Temporary pause — just skip this cycle
                pass
            return

        # ── Position sizing ────────────────────────────────────────────────
        sizing = self._sizer.size(opp, balance)
        if sizing.reject_reason:
            self._log.info(
                "bot.trade_rejected_sizing",
                reason=sizing.reject_reason,
                market=opp.market.question[:60],
            )
            return

        # ── Execution timing budget ────────────────────────────────────────
        elapsed_ms = (time.time() - detection_ts) * 1000
        budget_remaining_ms = self._s.execution_timeout_ms - elapsed_ms
        if budget_remaining_ms < 50:
            self._log.warning(
                "bot.timing_budget_exceeded",
                elapsed_ms=round(elapsed_ms, 1),
            )
            return

        # ── Place order ────────────────────────────────────────────────────
        try:
            receipt = await asyncio.wait_for(
                self._poly.place_order(
                    market=opp.market,
                    side=opp.recommended_side,
                    size_usdc=sizing.size_usdc,
                    price=opp.recommended_price,
                ),
                timeout=budget_remaining_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            self._log.error("bot.order_timeout")
            self._telegram.error("Order timeout", RuntimeError("Execution budget exceeded"))
            return
        except Exception as exc:
            self._log.exception("bot.order_error", exc=str(exc))
            self._telegram.error("Order placement failed", exc)
            return

        total_ms = (time.time() - detection_ts) * 1000
        self._log.info(
            "bot.order_placed",
            order_id=receipt.order_id,
            side=receipt.side,
            size=sizing.size_usdc,
            price=opp.recommended_price,
            edge_pct=round(opp.abs_edge * 100, 2),
            total_ms=round(total_ms, 1),
        )

        # ── Telegram alert ─────────────────────────────────────────────────
        self._telegram.trade_entry(
            market=opp.market.question[:80],
            side=opp.recommended_side,
            size=sizing.size_usdc,
            price=opp.recommended_price,
            edge=opp.abs_edge * 100,
        )

        # ── Paper DB log ───────────────────────────────────────────────────
        if self._paper:
            await self._db.log_paper_trade(
                order_id=receipt.order_id,
                market_id=opp.market.condition_id,
                market_question=opp.market.question,
                asset=opp.market.asset,
                side=opp.recommended_side,
                size_usdc=sizing.size_usdc,
                price=opp.recommended_price,
                our_prob=opp.our_prob,
                market_prob=opp.market_prob,
                edge=opp.edge,
                kelly_fraction=sizing.kelly_fraction,
                spot_price=opp.spot_price,
            )
            # Update simulated balance (stake is immediately committed)
            self._poly.update_paper_balance(-sizing.size_usdc)

        # ── Track as active position ───────────────────────────────────────
        self._active_positions.append(
            ActivePosition(receipt, opp, opp.market.expiry_ts)
        )

    # ── Position Settlement ────────────────────────────────────────────────

    async def _settle_expired_positions(self) -> None:
        """Settle any positions whose market has expired."""
        if not self._active_positions:
            return

        now = time.time()
        still_open: list[ActivePosition] = []

        for pos in self._active_positions:
            if not pos.is_expired():
                still_open.append(pos)
                continue

            # Attempt to get final price for the asset
            final_price = await self._price_feed.get_price(pos.opportunity.market.asset)
            if final_price is None:
                # Use last known spot at detection if feed is stale
                final_price = pos.opportunity.spot_price

            pnl = pos.simulate_pnl(final_price) if self._paper else 0.0
            hold_secs = now - pos.entered_at

            self._log.info(
                "bot.position_settled",
                order_id=pos.receipt.order_id,
                pnl=pnl,
                hold_secs=round(hold_secs, 1),
                paper=self._paper,
            )

            if self._paper:
                balance_now = await self._poly.get_portfolio_usdc()
                new_balance = balance_now + pos.receipt.size_usdc + pnl  # return stake ± pnl
                self._poly.update_paper_balance(pos.receipt.size_usdc + pnl)

                await self._db.resolve_paper_trade(
                    order_id=pos.receipt.order_id,
                    pnl_usdc=pnl,
                    portfolio_after=new_balance,
                )

                self._risk.record_trade(
                    TradeResult(pnl_usdc=pnl, portfolio_after=new_balance)
                )

            self._telegram.trade_exit(
                market=pos.opportunity.market.question[:80],
                pnl=pnl,
                hold_secs=hold_secs,
            )

        self._active_positions = still_open

    # ── Market Refresh ─────────────────────────────────────────────────────

    async def _refresh_markets(self) -> None:
        self._log.info("bot.refreshing_markets")
        try:
            self._markets = await self._poly.fetch_active_markets(
                target_assets=self._s.target_assets,
                target_durations=self._s.target_durations_minutes,
                min_liquidity=self._s.min_market_liquidity_usd,
                max_markets=self._s.max_monitored_markets,
            )
            self._log.info("bot.markets_refreshed", count=len(self._markets))
        except Exception as exc:
            self._log.error("bot.market_refresh_error", exc=str(exc))
            self._telegram.error("Market refresh failed", exc)

    # ── Heartbeat ──────────────────────────────────────────────────────────

    async def _send_heartbeat(self) -> None:
        balance = await self._poly.get_portfolio_usdc()
        stats = await self._db.get_today_stats() if self._paper else {}
        trades_today = stats.get("total", 0) or self._risk.total_trades
        self._telegram.heartbeat(balance, trades_today, self._risk.win_rate)
        self._log.info(
            "bot.heartbeat",
            balance=balance,
            day_pnl=self._risk.day_pnl,
            trades_today=trades_today,
            win_rate=self._risk.win_rate,
            active_positions=len(self._active_positions),
        )

    # ── Midnight reset ─────────────────────────────────────────────────────

    async def _midnight_reset(self, today: str) -> None:
        balance = await self._poly.get_portfolio_usdc()
        # Persist yesterday's summary
        yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
        if self._last_midnight_reset:
            stats = await self._db.get_today_stats() if self._paper else {}
            await self._db.upsert_daily_summary(
                date=yesterday,
                start_balance=self._risk._day_start_balance or balance,
                end_balance=balance,
                total_trades=stats.get("total", 0) or self._risk.total_trades,
                winning_trades=stats.get("wins", 0) or self._risk._winning_trades,
                total_pnl=stats.get("total_pnl", 0) or self._risk.day_pnl,
            )
        self._risk.reset_daily_stats(balance)
        self._last_midnight_reset = today
        self._log.info("bot.midnight_reset", today=today, balance=balance)


# ─────────────────────────────────────────────────────────────────────────────
# Startup banner and safety prompts
# ─────────────────────────────────────────────────────────────────────────────

def print_banner(paper_mode: bool) -> None:
    mode = "📄 PAPER MODE" if paper_mode else "🔴 LIVE MODE"
    print("=" * 60)
    print(f"  Polymarket Latency Arbitrage Bot")
    print(f"  Mode: {mode}")
    if not paper_mode:
        print("  ⚠️  REAL MONEY WILL BE TRADED")
    print("=" * 60)


def live_mode_safety_check() -> None:
    """Interactive confirmation before live trading starts."""
    print("\n⚠️  LIVE MODE ACTIVATED")
    print("This bot will place real orders using your Polymarket account.")
    print("You can lose real money. Ensure you have tested in paper mode first.\n")
    ans = input("Type 'YES I ACCEPT' to continue: ").strip()
    if ans != "YES I ACCEPT":
        sys.exit("Live mode aborted. Run without --live to use paper mode.")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    args = parse_args()
    paper_mode = resolve_paper_mode(args)

    # Load settings (exits with error message if env vars missing)
    settings = load_settings()

    # Setup logging before anything else
    setup_logging(settings.log_dir, settings.log_max_bytes, settings.log_backup_count)
    log = get_logger("main")

    print_banner(paper_mode)

    if not paper_mode:
        live_mode_safety_check()

    bot = ArbitrageBot(settings, paper_mode=paper_mode)

    # Graceful shutdown on SIGINT / SIGTERM
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _handle_signal():
        log.info("signal.received_shutting_down")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    await bot.start()

    # Run bot loop and shutdown-waiter concurrently
    bot_task = asyncio.create_task(bot.run(), name="bot-main")
    await shutdown_event.wait()

    log.info("main.shutting_down")
    bot_task.cancel()
    try:
        await bot_task
    except asyncio.CancelledError:
        pass

    await bot.stop()
    log.info("main.exited_cleanly")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
