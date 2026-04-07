"""
src/config.py
─────────────
Loads, validates and exposes all configuration from environment variables.
Raises a descriptive error at startup if required vars are missing.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load .env before anything else
load_dotenv()


def _require(key: str) -> str:
    """Return env var or abort with a clear message."""
    val = os.getenv(key, "").strip()
    if not val:
        sys.exit(
            f"[CONFIG ERROR] Required environment variable '{key}' is missing or empty.\n"
            f"Copy .env.example → .env and fill in your credentials."
        )
    return val


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key, "")
    try:
        return float(raw) if raw.strip() else default
    except ValueError:
        sys.exit(f"[CONFIG ERROR] '{key}' must be a number, got: {raw!r}")


def _int_env(key: str, default: int) -> int:
    raw = os.getenv(key, "")
    try:
        return int(raw) if raw.strip() else default
    except ValueError:
        sys.exit(f"[CONFIG ERROR] '{key}' must be an integer, got: {raw!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Settings dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Settings:
    # ── Credentials ────────────────────────────────────────────────────────
    poly_api_key: str
    poly_private_key: str
    poly_api_secret: str          # may be empty if using L1 only
    poly_api_passphrase: str      # may be empty if using L1 only
    telegram_token: str
    telegram_chat_id: str
    alchemy_rpc_url: str

    # ── Binance WebSocket ───────────────────────────────────────────────────
    binance_ws_url: str = "wss://stream.binance.com:9443/stream"
    binance_streams: tuple = ("btcusdt@ticker", "ethusdt@ticker")
    price_stale_seconds: int = 10          # reject prices older than this
    ws_max_retries: int = 10
    ws_backoff_base: float = 1.5          # seconds, doubles each retry

    # ── Edge Detection ──────────────────────────────────────────────────────
    min_detectable_edge_pct: float = 5.0   # below this: ignored entirely
    min_exec_edge_pct: float = 8.0         # below this: no trade
    target_assets: tuple = ("BTC", "ETH")
    target_durations_minutes: tuple = (5, 15)
    min_market_liquidity_usd: float = 50_000.0
    max_monitored_markets: int = 20

    # ── Position Sizing ─────────────────────────────────────────────────────
    kelly_fraction: float = 0.5            # half-Kelly
    max_position_pct: float = 8.0          # hard cap: 8 % of portfolio

    # ── Risk Management ─────────────────────────────────────────────────────
    daily_halt_pct: float = 20.0           # halt if day P&L < -20 %
    drawdown_halt_pct: float = 40.0        # halt if port < 60 % of ATH → 40 % drawdown
    consec_loss_pause_count: int = 5
    consec_loss_pause_minutes: int = 30

    # ── Execution ───────────────────────────────────────────────────────────
    execution_timeout_ms: int = 800        # total detection→order budget

    # ── Monitoring ─────────────────────────────────────────────────────────
    heartbeat_interval_seconds: int = 3600  # 1 hour

    # ── Logging ─────────────────────────────────────────────────────────────
    log_dir: str = "logs"
    log_max_bytes: int = 50 * 1024 * 1024  # 50 MB
    log_backup_count: int = 5
    db_path: str = "paper_trades.db"

    # ── Polymarket API ──────────────────────────────────────────────────────
    poly_host: str = "https://clob.polymarket.com"
    poly_chain_id: int = 137               # Polygon mainnet


def load_settings() -> Settings:
    """Factory: read env vars and return a validated Settings instance."""
    return Settings(
        # Credentials
        poly_api_key=_require("POLY_API_KEY"),
        poly_private_key=_require("POLY_PRIVATE_KEY"),
        poly_api_secret=_optional("POLY_API_SECRET"),
        poly_api_passphrase=_optional("POLY_API_PASSPHRASE"),
        telegram_token=_require("TELEGRAM_TOKEN"),
        telegram_chat_id=_require("TELEGRAM_CHAT_ID"),
        alchemy_rpc_url=_require("ALCHEMY_RPC_URL"),

        # Thresholds (overridable via env)
        min_detectable_edge_pct=_float_env("BOT_MIN_EDGE_PCT", 5.0),
        min_exec_edge_pct=_float_env("BOT_EXEC_EDGE_PCT", 8.0),
        max_position_pct=_float_env("BOT_MAX_PORTFOLIO_PCT", 8.0),
        min_market_liquidity_usd=_float_env("BOT_MIN_LIQUIDITY_USD", 50_000.0),
        daily_halt_pct=_float_env("BOT_DAILY_HALT_PCT", 20.0),
        drawdown_halt_pct=_float_env("BOT_DRAWDOWN_HALT_PCT", 40.0),
    )
