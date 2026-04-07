"""
src/database.py
───────────────
Async SQLite persistence layer for paper-mode trade logs.

Tables
──────
- paper_trades   : every simulated order with full context
- daily_summary  : one row per day with aggregate stats
"""

from __future__ import annotations

import time
from typing import Optional

import aiosqlite

from src.logger import get_logger

log = get_logger(__name__)

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS paper_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id        TEXT    NOT NULL UNIQUE,
    created_at      REAL    NOT NULL,           -- unix timestamp
    market_id       TEXT    NOT NULL,
    market_question TEXT    NOT NULL,
    asset           TEXT    NOT NULL,           -- BTC / ETH
    side            TEXT    NOT NULL,           -- YES / NO
    size_usdc       REAL    NOT NULL,
    price           REAL    NOT NULL,
    our_prob        REAL    NOT NULL,
    market_prob     REAL    NOT NULL,
    edge            REAL    NOT NULL,
    kelly_fraction  REAL    NOT NULL,
    spot_price      REAL    NOT NULL,
    resolved        INTEGER NOT NULL DEFAULT 0, -- 0 = open, 1 = resolved
    pnl_usdc        REAL,                       -- NULL until resolved
    portfolio_after REAL,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS daily_summary (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT    NOT NULL UNIQUE,    -- YYYY-MM-DD
    start_balance   REAL    NOT NULL,
    end_balance     REAL,
    total_trades    INTEGER NOT NULL DEFAULT 0,
    winning_trades  INTEGER NOT NULL DEFAULT 0,
    total_pnl       REAL    NOT NULL DEFAULT 0,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_paper_trades_created ON paper_trades (created_at);
CREATE INDEX IF NOT EXISTS idx_paper_trades_asset   ON paper_trades (asset);
"""


class Database:

    def __init__(self, db_path: str) -> None:
        self._path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_DDL)
        await self._conn.commit()
        log.info("database.started", path=self._path)

    async def stop(self) -> None:
        if self._conn:
            await self._conn.close()
        log.info("database.stopped")

    # ── Paper trade writes ─────────────────────────────────────────────────

    async def log_paper_trade(
        self,
        order_id: str,
        market_id: str,
        market_question: str,
        asset: str,
        side: str,
        size_usdc: float,
        price: float,
        our_prob: float,
        market_prob: float,
        edge: float,
        kelly_fraction: float,
        spot_price: float,
        notes: str = "",
    ) -> None:
        sql = """
        INSERT OR IGNORE INTO paper_trades
          (order_id, created_at, market_id, market_question, asset, side,
           size_usdc, price, our_prob, market_prob, edge, kelly_fraction, spot_price, notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        await self._conn.execute(
            sql,
            (
                order_id, time.time(), market_id, market_question, asset, side,
                size_usdc, price, our_prob, market_prob, edge, kelly_fraction, spot_price, notes,
            ),
        )
        await self._conn.commit()
        log.debug("database.paper_trade_logged", order_id=order_id)

    async def resolve_paper_trade(
        self,
        order_id: str,
        pnl_usdc: float,
        portfolio_after: float,
    ) -> None:
        sql = """
        UPDATE paper_trades
        SET resolved = 1, pnl_usdc = ?, portfolio_after = ?
        WHERE order_id = ?
        """
        await self._conn.execute(sql, (pnl_usdc, portfolio_after, order_id))
        await self._conn.commit()

    async def upsert_daily_summary(
        self,
        date: str,               # "YYYY-MM-DD"
        start_balance: float,
        end_balance: float,
        total_trades: int,
        winning_trades: int,
        total_pnl: float,
    ) -> None:
        sql = """
        INSERT INTO daily_summary
            (date, start_balance, end_balance, total_trades, winning_trades, total_pnl)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            end_balance    = excluded.end_balance,
            total_trades   = excluded.total_trades,
            winning_trades = excluded.winning_trades,
            total_pnl      = excluded.total_pnl
        """
        await self._conn.execute(
            sql, (date, start_balance, end_balance, total_trades, winning_trades, total_pnl)
        )
        await self._conn.commit()

    # ── Reads ──────────────────────────────────────────────────────────────

    async def get_open_trades(self) -> list[dict]:
        async with self._conn.execute(
            "SELECT * FROM paper_trades WHERE resolved = 0 ORDER BY created_at"
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_today_stats(self) -> dict:
        """Return aggregate stats for today's paper trades."""
        import datetime
        today = datetime.date.today().isoformat()
        async with self._conn.execute(
            """
            SELECT
                COUNT(*)                                    AS total,
                SUM(CASE WHEN pnl_usdc > 0 THEN 1 ELSE 0 END)  AS wins,
                COALESCE(SUM(pnl_usdc), 0)                  AS total_pnl,
                COALESCE(AVG(pnl_usdc), 0)                  AS avg_pnl
            FROM paper_trades
            WHERE DATE(created_at, 'unixepoch') = ?
            """,
            (today,),
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else {}

    async def get_recent_trades(self, limit: int = 20) -> list[dict]:
        async with self._conn.execute(
            "SELECT * FROM paper_trades ORDER BY created_at DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]
