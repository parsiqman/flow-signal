"""
Pre-registration of hypotheses, and an automatic ledger of every trial run.

Two jobs, both about honesty over time rather than cleverness in the moment.

**Pre-registration.** An idea is written down BEFORE it is tested, including
what would count as success and -- the field that does the most work -- who is
on the other side of the trade and why they keep taking it. Writing the success
criterion afterwards is how a disappointing result becomes a "promising signal
worth exploring further". Recording it first makes that move visible.

**The trial ledger.** Every backtest is counted, automatically, and persisted.
This is not bookkeeping: the deflated Sharpe ratio is a function of how many
things you tried, so an undercounted trial log silently inflates every
significance number downstream. Humans reliably undercount, because the twenty
runs spent "just getting it working" do not feel like trials. They are. The
ledger counts them whether or not they felt like science.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "research"


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """
    One pre-registered idea.

    `who_pays` and `why_they_persist` are required and validated. If you cannot
    name the counterparty and say why they keep showing up, you have found a
    pattern in a dataset, not an edge -- and patterns in datasets are free,
    plentiful, and worth nothing.
    """
    id: str
    title: str
    source: str                  # 'risk_premium' | 'constraint' | 'behavioural' | ...
    thesis: str
    who_pays: str                # who is on the other side
    why_they_persist: str        # why they keep doing it
    prediction: str              # what should be observable if true
    success_criteria: dict[str, Any]   # declared BEFORE testing
    kill_criteria: str           # what would make you abandon it
    data_required: str
    capacity_note: str = ""
    decay_risk: str = ""
    status: str = "registered"   # registered -> screened -> tested -> validated -> paper -> live -> killed
    created_at: float = field(default_factory=time.time)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        thin = [f for f in ("thesis", "who_pays", "why_they_persist",
                            "prediction", "kill_criteria")
                if len(str(getattr(self, f)).strip()) < 25]
        if thin:
            raise ValueError(
                f"Hypothesis '{self.id}' has placeholder-length fields: {thin}. "
                "These are the fields that separate an edge from a pattern; "
                "a one-word answer means the idea is not ready to test.")
        if not self.success_criteria:
            raise ValueError(
                f"Hypothesis '{self.id}' declares no success criteria. Declaring "
                "them after seeing results is how a failure becomes a 'promising "
                "direction'.")


class HypothesisRegistry:
    """Append-mostly store of ideas. Edits to a tested hypothesis are logged."""

    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "hypotheses.json"
        self._items: dict[str, Hypothesis] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self._items = {k: Hypothesis(**v) for k, v in raw.items()}

    def _save(self) -> None:
        self.path.write_text(json.dumps(
            {k: asdict(v) for k, v in self._items.items()}, indent=2))

    def register(self, h: Hypothesis) -> Hypothesis:
        h.validate()
        if h.id in self._items:
            raise ValueError(f"Hypothesis '{h.id}' is already registered. "
                             "Use amend() so the change is recorded.")
        self._items[h.id] = h
        self._save()
        return h

    def amend(self, hid: str, note: str, **changes) -> Hypothesis:
        """
        Change a registered hypothesis, leaving a trace.

        Amending success criteria after testing is not forbidden -- sometimes
        the first attempt was genuinely ill-posed -- but it is recorded, so the
        record shows what was decided before results and what after.
        """
        h = self._items[hid]
        stamp = time.strftime("%Y-%m-%d")
        h.notes.append(f"[{stamp}] AMENDED ({', '.join(changes)}): {note}")
        for k, v in changes.items():
            setattr(h, k, v)
        self._save()
        return h

    def set_status(self, hid: str, status: str, note: str = "") -> None:
        h = self._items[hid]
        h.notes.append(f"[{time.strftime('%Y-%m-%d')}] {h.status} -> {status}"
                       + (f": {note}" if note else ""))
        h.status = status
        self._save()

    def get(self, hid: str) -> Hypothesis:
        return self._items[hid]

    def all(self) -> list[Hypothesis]:
        return list(self._items.values())

    def to_frame(self) -> pd.DataFrame:
        if not self._items:
            return pd.DataFrame(columns=["id", "title", "source", "status"])
        return pd.DataFrame([{
            "id": h.id, "title": h.title, "source": h.source,
            "status": h.status, "who_pays": h.who_pays[:60],
        } for h in self._items.values()])


# ---------------------------------------------------------------------------
# Trial ledger
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    hypothesis_id: str
    fingerprint: str
    params: dict[str, Any]
    dataset: str
    sharpe: float | None
    cagr: float | None
    stage: str          # 'exploration' | 'tuning' | 'validation'
    at: float = field(default_factory=time.time)


class TrialLedger:
    """
    Counts every backtest ever run, per hypothesis and per dataset.

    The count feeds `deflated_sharpe_ratio`. Undercounting it is the single
    easiest way to produce a result that looks significant and is not, which is
    why recording is a side effect of running rather than a thing to remember.
    """

    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "trials.jsonl"

    def record(self, trial: Trial) -> None:
        with self.path.open("a") as fh:
            fh.write(json.dumps(asdict(trial)) + "\n")

    def all(self) -> list[Trial]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                out.append(Trial(**json.loads(line)))
        return out

    def count(self, hypothesis_id: str | None = None,
              dataset: str | None = None) -> int:
        """
        Trials run. This is the N that goes into the deflation math.

        Counts exploration runs too. A run made while debugging still saw the
        data, and still contributed to which configuration ended up looking
        good -- the fact that it did not feel like a hypothesis test at the
        time does not exempt it.
        """
        return sum(1 for t in self.all()
                   if (hypothesis_id is None or t.hypothesis_id == hypothesis_id)
                   and (dataset is None or t.dataset == dataset))

    def sharpes(self, hypothesis_id: str | None = None,
                dataset: str | None = None):
        import numpy as np
        vals = [t.sharpe for t in self.all()
                if (hypothesis_id is None or t.hypothesis_id == hypothesis_id)
                and (dataset is None or t.dataset == dataset)
                and t.sharpe is not None]
        return np.array(vals, dtype=float)

    def distinct_configs(self, hypothesis_id: str | None = None) -> int:
        """
        Distinct parameter sets tried, as opposed to total runs.

        Re-running an identical configuration is not a new trial; running a
        slightly different one is. Reporting both makes it obvious when a
        "new idea" was the old idea with a changed constant.
        """
        return len({t.fingerprint for t in self.all()
                    if hypothesis_id is None or t.hypothesis_id == hypothesis_id})

    def summary(self) -> pd.DataFrame:
        trials = self.all()
        if not trials:
            return pd.DataFrame(columns=["hypothesis_id", "runs", "distinct_configs"])
        df = pd.DataFrame([asdict(t) for t in trials])
        g = df.groupby("hypothesis_id").agg(
            runs=("fingerprint", "size"),
            distinct_configs=("fingerprint", "nunique"),
            best_sharpe=("sharpe", "max"),
            median_sharpe=("sharpe", "median"),
        ).reset_index()
        return g
