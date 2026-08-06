"""
The promotion pipeline: idea -> screened -> tested -> validated -> paper -> live.

Stages exist so that expensive, irreversible steps come last and cheap
disqualifying ones come first. Each gate is a question that can kill the idea,
and the cost of answering rises at every stage. Screening costs an hour of
thinking; live trading costs money and cannot be undone.

The rule the pipeline enforces is that **stages cannot be skipped**. A promising
backtest does not get to jump to live. Not because the process is sacred, but
because every skipped stage is a check that would have been cheaper to fail
than the one that follows it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd

from .registry import Hypothesis, HypothesisRegistry, TrialLedger
from .validation import GauntletResult, minimum_backtest_length


class Stage(IntEnum):
    IDEA = 0
    SCREENED = 1        # economic rationale survives scrutiny; data obtainable
    TESTED = 2          # a backtest exists and is not obviously broken
    VALIDATED = 3       # cleared the full gauntlet on out-of-sample data
    PAPER = 4           # traded live-but-unfunded for a full quarter
    LIVE = 5            # real money, sized far below the backtest
    KILLED = -1


STAGE_NAMES = {s.value: s.name.lower() for s in Stage}


@dataclass
class PromotionCheck:
    allowed: bool
    reason: str


def can_promote(current: Stage, target: Stage,
                gauntlet: GauntletResult | None = None,
                paper_days: int = 0,
                data_years: float = 0.0,
                n_trials: int = 0,
                observed_sharpe: float = 0.0) -> PromotionCheck:
    """
    May this candidate advance? Answers with a reason either way.

    The interesting gates:

    IDEA -> SCREENED requires a named counterparty. Enforced by Hypothesis
    validation rather than here.

    TESTED -> VALIDATED requires the full gauntlet AND enough data for the
    trial count. Minimum backtest length is the gate people skip: searching 500
    configurations for a Sharpe-1 strategy needs roughly 12 years of data before
    the winner can be told apart from the search itself. Backtesting five years
    and declaring victory is not a smaller version of that -- it is a different,
    invalid activity.

    VALIDATED -> PAPER is free and should always be taken.

    PAPER -> LIVE requires a full quarter, because the gap between a backtest
    and a fill is where this class of strategy actually dies, and a quarter is
    the shortest window that contains a month-end, an expiry cycle and at least
    one surprise.
    """
    if target == Stage.KILLED:
        return PromotionCheck(True, "killing a candidate is always allowed")
    if target <= current:
        return PromotionCheck(False, f"already at or beyond {STAGE_NAMES[target]}")
    if target - current > 1:
        return PromotionCheck(
            False,
            f"cannot skip from {STAGE_NAMES[current]} to {STAGE_NAMES[target]}; "
            f"each stage is a cheaper check than the one after it")

    if target == Stage.VALIDATED:
        if gauntlet is None:
            return PromotionCheck(False, "no gauntlet result supplied")
        if not gauntlet.passed:
            return PromotionCheck(False, gauntlet.summary())
        need = minimum_backtest_length(n_trials, observed_sharpe)
        if data_years < need:
            return PromotionCheck(
                False,
                f"{n_trials} trials at Sharpe {observed_sharpe:.2f} needs "
                f"~{need:.1f}y of data to be distinguishable from the search; "
                f"only {data_years:.1f}y available")
        return PromotionCheck(True, "gauntlet clean and sample long enough")

    if target == Stage.LIVE:
        if paper_days < 63:
            return PromotionCheck(
                False, f"paper traded {paper_days} days; a full quarter (63) is "
                       "the minimum that contains an expiry cycle and a surprise")
        return PromotionCheck(True, "paper period complete")

    return PromotionCheck(True, f"promoted to {STAGE_NAMES[target]}")


class Pipeline:
    """Thin coordinator over the registry, the ledger and the gates."""

    def __init__(self, registry: HypothesisRegistry, ledger: TrialLedger):
        self.registry = registry
        self.ledger = ledger

    def promote(self, hid: str, target: Stage, **kwargs) -> PromotionCheck:
        h = self.registry.get(hid)
        current = Stage[h.status.upper()] if h.status.upper() in Stage.__members__ \
            else Stage.IDEA
        kwargs.setdefault("n_trials", self.ledger.count(hid))
        check = can_promote(current, target, **kwargs)
        if check.allowed:
            self.registry.set_status(hid, STAGE_NAMES[target], check.reason)
        else:
            self.registry.get(hid).notes.append(
                f"BLOCKED -> {STAGE_NAMES[target]}: {check.reason}")
            self.registry._save()
        return check

    def status(self) -> pd.DataFrame:
        """Everything in flight, with its trial count attached."""
        rows = []
        for h in self.registry.all():
            rows.append({
                "id": h.id, "title": h.title[:44], "source": h.source,
                "stage": h.status,
                "runs": self.ledger.count(h.id),
                "distinct_configs": self.ledger.distinct_configs(h.id),
            })
        if not rows:
            return pd.DataFrame(columns=["id", "title", "source", "stage", "runs"])
        return pd.DataFrame(rows).sort_values("stage").reset_index(drop=True)

    def research_debt(self) -> pd.DataFrame:
        """
        How much data each candidate now needs, given how much it has been
        searched.

        The number rises every time you run another backtest. Watching it climb
        is the most effective discipline in this whole package: it makes the
        cost of one more parameter tweak visible at the moment you are tempted
        to make it.
        """
        rows = []
        for h in self.registry.all():
            n = self.ledger.count(h.id)
            sharpes = self.ledger.sharpes(h.id)
            best = float(np.nanmax(sharpes)) if len(sharpes) else 0.0
            rows.append({
                "id": h.id,
                "runs": n,
                "best_sharpe_seen": round(best, 2) if best else None,
                "years_of_data_required": round(
                    minimum_backtest_length(n, best), 1) if best > 0 else None,
            })
        return pd.DataFrame(rows)
