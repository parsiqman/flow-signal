"""
The contract every candidate strategy implements.

One interface, so the harness never knows or cares what a strategy does. This
matters more than it looks: if each idea gets its own bespoke backtest, then
each idea gets its own bespoke bugs, and results are not comparable across
candidates. A shared harness means a shared, tested, once-audited execution
path -- and it means adding a new idea costs a signal function, not a project.

Naming note: this package is `lab`, not `platform`, because `platform` is a
standard-library module and shadowing it breaks imports in confusing ways.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Bar:
    """
    One observation the harness hands to a strategy.

    Deliberately minimal and market-agnostic: a panel of prices plus whatever
    extra columns a given dataset carries (implied vol, days to earnings,
    funding rate). A strategy declares what it needs via `required_columns`
    and the harness refuses to run it against data that lacks them, rather
    than silently producing NaNs that look like results.
    """
    day: int
    frame: pd.DataFrame          # one row per instrument, indexed by name


@dataclass
class Order:
    """An intended position. Sizing is the harness's job, not the signal's."""
    name: str
    direction: int               # +1 long, -1 short
    conviction: float            # 0..1, scales size within the risk budget
    horizon_days: int
    meta: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """
    A candidate. Implement `generate` and declare your data needs.

    Deliberately NOT responsible for: position sizing, risk limits, cost
    modelling, or exits driven by risk. Those belong to the harness so that
    every candidate is subject to identical discipline and no strategy can
    flatter itself by quietly assuming better execution than its rivals.
    """

    name: str = "unnamed"
    required_columns: tuple[str, ...] = ("spot",)
    # Minimum history before the strategy may trade. The harness enforces it.
    warmup_days: int = 60

    @abstractmethod
    def generate(self, bar: Bar, history: pd.DataFrame) -> list[Order]:
        """
        Return intended positions for this bar.

        `history` contains data strictly up to and including `bar.day`. The
        harness constructs it by truncation, so lookahead is impossible by
        construction rather than by discipline.
        """

    def parameters(self) -> dict[str, Any]:
        """
        Every tunable knob, for the trial ledger.

        Used to detect when a "new idea" is actually the same idea with a
        different constant -- which is the most common way a trial count gets
        understated, and the trial count is what the deflation math runs on.
        """
        return {}

    def fingerprint(self) -> str:
        """Stable identity for this exact configuration."""
        import hashlib
        import json
        payload = json.dumps({"name": self.name, "params": self.parameters()},
                             sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def validate_data(strategy: Strategy, panel: pd.DataFrame) -> None:
    """
    Refuse to run a strategy against data that cannot support it.

    A missing column that silently becomes NaN produces a backtest that runs
    cleanly and means nothing. Fail loudly instead.
    """
    missing = [c for c in strategy.required_columns if c not in panel.columns]
    if missing:
        raise ValueError(
            f"{strategy.name} requires columns {missing} which are not in this "
            f"dataset (has: {sorted(panel.columns)}). Refusing to run rather "
            f"than producing NaN-shaped results.")
    for c in strategy.required_columns:
        if panel[c].isna().all():
            raise ValueError(f"{strategy.name}: column '{c}' is entirely NaN.")


def returns_from_curve(curve: pd.Series) -> np.ndarray:
    """Daily returns from an equity curve, with degenerate cases handled."""
    v = np.asarray(curve, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.array([])
    r = np.diff(v) / v[:-1]
    return r[np.isfinite(r)]
