"""The stop. Everything else in this bot tries to make money; this tries to
bound how much it can lose.

Three properties matter more than the specific limits:

1. **State survives restart.** Drawdown is measured against the all-time equity
   peak stored on disk. A bot that forgets its losses when the process restarts
   has no drawdown limit -- it has a drawdown limit per process lifetime, which
   is not the same thing and is discovered the hard way.
2. **The halt is sticky.** Once tripped it stays tripped until a human clears it
   with `--reset-halt`. Auto-resuming after a drawdown breach is how a limit
   becomes a speed bump.
3. **The gate runs before the order, not after.** `allow_trade()` is checked
   ahead of every entry, and it is the only thing standing between a bad run and
   the account.

Limits are read from frozen config, so the loop cannot widen them at runtime.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone


@dataclass
class RiskState:
    bankroll: float = 0.0
    equity: float = 0.0
    peak_equity: float = 0.0
    realized_pnl: float = 0.0
    day: str = ""
    day_pnl: float = 0.0
    day_trades: int = 0
    consecutive_losses: int = 0
    wins: int = 0
    losses: int = 0
    halted: bool = False
    halt_reason: str = ""
    halted_at: str = ""

    @property
    def drawdown(self) -> float:
        return max(0.0, self.peak_equity - self.equity)

    @property
    def trades(self) -> int:
        return self.wins + self.losses

    @property
    def hit_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0


class RiskManager:
    def __init__(self, cfg, state_path: str | None = None):
        self.cfg = cfg
        self.path = state_path or cfg.state_path
        self.state = self._load()

    # -- persistence --------------------------------------------------------
    def _load(self) -> RiskState:
        if self.path and os.path.exists(self.path):
            with open(self.path) as f:
                return RiskState(**json.load(f))
        return RiskState(bankroll=self.cfg.starting_bankroll,
                         equity=self.cfg.starting_bankroll,
                         peak_equity=self.cfg.starting_bankroll,
                         day=self._today())

    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as f:
            json.dump(asdict(self.state), f, indent=2)
        os.replace(tmp, self.path)     # atomic: a crash mid-write cannot corrupt state

    @staticmethod
    def _today(now: datetime | None = None) -> str:
        return (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")

    def _roll_day(self, now: datetime | None = None) -> None:
        """Roll the daily counters. `now` is injectable so the simulator ages
        its day with SIMULATED time -- otherwise replaying a few hundred
        windows trips the live daily caps partway through and reports a hit
        rate on a truncated sample."""
        today = self._today(now)
        if self.state.day != today:
            self.state.day = today
            self.state.day_pnl = 0.0
            self.state.day_trades = 0

    # -- the gate -----------------------------------------------------------
    def allow_trade(self, cost: float, now: datetime | None = None) -> tuple[bool, str]:
        """Check every limit. Returns (allowed, reason-if-not)."""
        self._roll_day(now)
        s, c = self.state, self.cfg

        if s.halted:
            return False, f"HALTED: {s.halt_reason}"
        if s.drawdown >= c.max_drawdown:
            return False, self._halt(f"max drawdown hit: ${s.drawdown:.2f} >= ${c.max_drawdown:.2f}")
        if -s.day_pnl >= c.daily_loss_limit:
            return False, f"daily loss limit: ${-s.day_pnl:.2f} >= ${c.daily_loss_limit:.2f}"
        if s.consecutive_losses >= c.max_consecutive_losses:
            return False, f"{s.consecutive_losses} consecutive losses"
        if s.day_trades >= c.max_trades_per_day:
            return False, f"daily trade cap: {s.day_trades} >= {c.max_trades_per_day}"
        if cost > s.equity:
            return False, f"insufficient equity: need ${cost:.2f}, have ${s.equity:.2f}"
        return True, ""

    def _halt(self, reason: str) -> str:
        self.state.halted = True
        self.state.halt_reason = reason
        self.state.halted_at = datetime.now(timezone.utc).isoformat()
        self.save()
        return f"HALTED: {reason}"

    def reset_halt(self) -> None:
        """Deliberate human action only.

        Re-baselines the equity peak to current equity. Without that, clearing
        the flag is theatre: the breached drawdown is still on the books and the
        gate re-halts on the very next check. Re-baselining is the human saying
        "I accept this loss; the next $max_drawdown starts from here" -- which
        is what resuming after a breach actually means. The loss stays in
        realized_pnl; only the drawdown yardstick moves.
        """
        self.state.halted = False
        self.state.halt_reason = ""
        self.state.halted_at = ""
        self.state.consecutive_losses = 0
        self.state.peak_equity = self.state.equity
        self.save()

    # -- bookkeeping --------------------------------------------------------
    def record_entry(self, cost: float, now: datetime | None = None) -> None:
        self._roll_day(now)
        self.state.day_trades += 1
        self.save()

    def record_settlement(self, pnl: float, now: datetime | None = None) -> None:
        """Book a settled trade and re-check the drawdown limit immediately."""
        self._roll_day(now)
        s = self.state
        s.realized_pnl += pnl
        s.equity += pnl
        s.day_pnl += pnl
        s.peak_equity = max(s.peak_equity, s.equity)
        if pnl > 0:
            s.wins += 1
            s.consecutive_losses = 0
        else:
            s.losses += 1
            s.consecutive_losses += 1
        if s.drawdown >= self.cfg.max_drawdown:
            self._halt(f"max drawdown hit: ${s.drawdown:.2f} >= ${self.cfg.max_drawdown:.2f}")
        self.save()

    def summary(self) -> str:
        s = self.state
        return (f"equity ${s.equity:.2f} | pnl ${s.realized_pnl:+.2f} | "
                f"dd ${s.drawdown:.2f}/{self.cfg.max_drawdown:.0f} | "
                f"{s.wins}W-{s.losses}L ({s.hit_rate:.0%}) | "
                f"today ${s.day_pnl:+.2f} in {s.day_trades} "
                f"{'| HALTED: ' + s.halt_reason if s.halted else ''}")
