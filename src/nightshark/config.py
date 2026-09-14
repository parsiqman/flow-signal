"""Every tunable in one place, overridable by environment variable.

Config is frozen: the runner reads it once at startup and the loop cannot
mutate its own risk limits. That is deliberate -- a bot that can raise its own
drawdown ceiling does not have a drawdown ceiling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields


def _env(name: str, default, cast):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return cast(raw)


@dataclass(frozen=True)
class Config:
    # ---- market -----------------------------------------------------------
    # Kalshi groups markets into series -> events -> markets. The 15-minute BTC
    # series ticker MUST be confirmed against the live API before real trading;
    # `python -m nightshark.runner --list-markets` prints what the account can
    # actually see. Series tickers change and are not guessable.
    series_ticker: str = "KXBTC"
    window_minutes: int = 15

    # ---- data -------------------------------------------------------------
    # "synthetic" needs no network and is the default so the bot is runnable
    # and testable offline. "rest" pulls real candles from `feed_url`.
    feed: str = "synthetic"
    feed_url: str = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    feed_symbol: str = "BTC-USD"
    history_minutes: int = 360          # 1m candles pulled per decision
    synthetic_seed: int = 7

    # ---- model ------------------------------------------------------------
    model: str = "claude-opus-5"
    effort: str = "medium"              # low | medium | high | xhigh | max
    max_tokens: int = 2048
    llm_timeout_s: float = 60.0

    # ---- execution --------------------------------------------------------
    contracts_per_trade: int = 10       # fixed size; no scaling on conviction
    max_price_cents: int = 70           # never pay above this for a contract
    min_price_cents: int = 15           # nor below -- too cheap means too far OTM
    min_confidence: float = 0.0         # 0 disables the confidence gate
    dry_run: bool = True                # paper by default. Opt in to real money.
    kalshi_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    kalshi_key_id: str = ""
    kalshi_private_key_path: str = ""

    # ---- risk -------------------------------------------------------------
    starting_bankroll: float = 1000.0
    max_drawdown: float = 200.0         # halt when equity is this far below peak
    daily_loss_limit: float = 100.0
    max_consecutive_losses: int = 6
    max_trades_per_day: int = 40
    state_path: str = "state/nightshark_state.json"

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from NIGHTSHARK_*-prefixed environment variables.

        Two exceptions read conventional names instead, because they are
        secrets that other tooling already sets: KALSHI_KEY_ID and
        KALSHI_PRIVATE_KEY_PATH.
        """
        values = {}
        for f in fields(cls):
            values[f.name] = _env(f"NIGHTSHARK_{f.name.upper()}", f.default, f.type if not isinstance(f.type, str) else _CASTS[f.name])
        values["kalshi_key_id"] = os.environ.get("KALSHI_KEY_ID", values["kalshi_key_id"])
        values["kalshi_private_key_path"] = os.environ.get(
            "KALSHI_PRIVATE_KEY_PATH", values["kalshi_private_key_path"]
        )
        return cls(**values)

    def describe_risk(self) -> str:
        return (
            f"bankroll ${self.starting_bankroll:.2f} | "
            f"max drawdown ${self.max_drawdown:.2f} | "
            f"daily loss limit ${self.daily_loss_limit:.2f} | "
            f"{self.contracts_per_trade} contracts/trade | "
            f"{'PAPER' if self.dry_run else 'LIVE MONEY'}"
        )


# Field name -> cast function, used by from_env (dataclass .type is a string
# under `from __future__ import annotations`).
_CASTS = {
    "series_ticker": str, "window_minutes": int,
    "feed": str, "feed_url": str, "feed_symbol": str,
    "history_minutes": int, "synthetic_seed": int,
    "model": str, "effort": str, "max_tokens": int, "llm_timeout_s": float,
    "contracts_per_trade": int, "max_price_cents": int, "min_price_cents": int,
    "min_confidence": float, "dry_run": bool, "kalshi_base_url": str,
    "kalshi_key_id": str, "kalshi_private_key_path": str,
    "starting_bankroll": float, "max_drawdown": float, "daily_loss_limit": float,
    "max_consecutive_losses": int, "max_trades_per_day": int, "state_path": str,
}
