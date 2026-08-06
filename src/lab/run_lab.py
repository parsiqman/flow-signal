"""
The research platform, demonstrated end to end on the strategy we already have.

    python src/lab/run_lab.py

Runs the VRP harvest through the whole machine: scouting queue, hypothesis
pre-registration, an honestly-counted parameter search, then the full
validation gauntlet, then a promotion decision.

The point of the exercise is the last step. The VRP strategy's headline Sharpe
of ~2.5 was produced by searching 36 configurations, and this pipeline reports
what a search of that size produces from pure noise -- which is the comparison
that decides whether the number means anything.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lab import scout, validation
    from lab.pipeline import Pipeline, Stage
    from lab.protocol import returns_from_curve
    from lab.registry import HypothesisRegistry, Trial, TrialLedger
else:
    from . import scout, validation
    from .pipeline import Pipeline, Stage
    from .protocol import returns_from_curve
    from .registry import HypothesisRegistry, Trial, TrialLedger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from options_alpha.backtest import run_backtest              # noqa: E402
from options_alpha.research import DEFAULT_GRID, split_panel  # noqa: E402
from options_alpha.strategy import StrategyConfig            # noqa: E402
from options_alpha.synthetic import MarketConfig, generate_market  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def section(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main(root: Path | None = None) -> None:
    tmp = None
    if root is None:
        tmp = tempfile.mkdtemp(prefix="lab-demo-")
        root = Path(tmp)

    registry = HypothesisRegistry(root)
    ledger = TrialLedger(root)
    pipe = Pipeline(registry, ledger)

    section("1. SCOUTING QUEUE -- ideas ordered by why someone loses to you")
    print(scout.queue().to_string(index=False))
    print("\nNote the ordering criterion: durability of the counterparty's")
    print("motive, not expected return. Expected return at the idea stage is")
    print("imagination; a named counterparty is a fact you can check.")

    section("2. PRE-REGISTRATION -- criteria fixed BEFORE any test")
    vrp = next(i for i in scout.CATALOGUE if i.source == "risk_premium")
    h = scout.to_hypothesis(
        vrp, hid="vrp-001",
        prediction="Implied volatility exceeds subsequently realised volatility "
                   "on average, so systematically selling defined-risk option "
                   "structures earns a positive risk-adjusted return net of "
                   "costs, with losses concentrated in volatility shocks.",
        success_criteria={
            "deflated_sharpe": ">= 0.95",
            "probability_of_overfit": "<= 0.25",
            "profitable_sample_share": ">= 0.70",
            "survives_null_data": "p <= 0.05",
        },
        kill_criteria="Frictionless straddle capture measured on real chains is "
                      "below 3%, or the strategy earns as much on null data as "
                      "on real data, or drawdown exceeds 30% in any sample.")
    registry.register(h)
    pipe.promote("vrp-001", Stage.SCREENED)
    print(f"registered: {h.id} -- {h.title}")
    print(f"  who pays:  {h.who_pays}")
    print(f"  criteria:  {h.success_criteria}")
    print(f"  kill if:   {h.kill_criteria}")

    section("3. THE SEARCH -- every run counted, including the ones that failed")
    panel = generate_market(MarketConfig(n_names=40, n_days=252 * 10, seed=17))
    tune, test = split_panel(panel)

    trial_returns, trial_sharpes = [], []
    from itertools import product
    keys = list(DEFAULT_GRID)
    for combo in product(*(DEFAULT_GRID[k] for k in keys)):
        cfg = replace(StrategyConfig(), **dict(zip(keys, combo)))
        res = run_backtest(tune, cfg)
        r = returns_from_curve(res["curve"]["marked_equity"])
        sr = (float(r.mean() / r.std(ddof=1) * np.sqrt(252))
              if len(r) > 20 and r.std(ddof=1) > 0 else np.nan)
        ledger.record(Trial(hypothesis_id="vrp-001", fingerprint=str(hash(combo)),
                            params=dict(zip(keys, combo)), dataset="synthetic-seed17",
                            sharpe=sr, cagr=res["stats"]["cagr"], stage="tuning"))
        trial_returns.append(r)
        trial_sharpes.append(sr)

    n_trials = ledger.count("vrp-001")
    print(f"configurations run: {n_trials}")
    print(f"best in-sample Sharpe:   {np.nanmax(trial_sharpes):.2f}")
    print(f"median in-sample Sharpe: {np.nanmedian(trial_sharpes):.2f}")

    disp = float(np.nanstd(np.array(trial_sharpes) / np.sqrt(252), ddof=1))
    luck = validation.expected_max_sharpe(n_trials, disp) * np.sqrt(252)
    print(f"\n  >> Expected best Sharpe from {n_trials} ZERO-EDGE strategies: {luck:.2f}")
    print("     Any candidate below this line is indistinguishable from the search.")

    section("4. OUT-OF-SAMPLE -- the frozen winner, run once")
    best_i = int(np.nanargmax(trial_sharpes))
    best_combo = list(product(*(DEFAULT_GRID[k] for k in keys)))[best_i]
    frozen = replace(StrategyConfig(), **dict(zip(keys, best_combo)))
    oos = run_backtest(test, frozen)
    oos_r = returns_from_curve(oos["curve"]["marked_equity"])
    print(f"frozen config: {dict(zip(keys, best_combo))}")
    for k in ("cagr", "sharpe", "max_drawdown", "worst_month"):
        print(f"  {k:16} {oos['stats'][k]}")

    section("5. NULL TEST -- same strategy, premium removed from the market")
    print("If it earns as much with no variance risk premium in the data, then")
    print("whatever it is capturing, it is not the premium it claims.\n")
    null_runs = []
    for seed in (91, 92, 93, 94, 95):
        null_mkt = MarketConfig(n_names=40, n_days=252 * 6, seed=seed,
                                vrp_mean=0.0, vrp_noise=0.02)
        null_panel = generate_market(null_mkt)
        nres = run_backtest(null_panel, frozen, null_mkt)
        null_runs.append(returns_from_curve(nres["curve"]["marked_equity"]))
        print(f"  seed {seed}: CAGR {nres['stats']['cagr']:+.4f} "
              f"sharpe {nres['stats']['sharpe']}")

    section("6. STABILITY -- fresh markets never tuned on")
    stability = []
    for seed in range(60, 68):
        mc = MarketConfig(n_names=40, n_days=252 * 8, seed=seed)
        s = run_backtest(generate_market(mc), frozen, mc)["stats"]
        stability.append(s["cagr"])
    stability = np.array(stability)
    print(f"  profitable in {100 * (stability > 0).mean():.0f}% of 8 markets; "
          f"mean CAGR {stability.mean():+.3f}")

    section("7. THE GAUNTLET")
    min_len = min(len(r) for r in trial_returns if len(r) > 0)
    matrix = np.column_stack([r[:min_len] for r in trial_returns if len(r) > 0])
    g = validation.run_gauntlet(
        candidate="vrp-001", oos_returns=oos_r, n_trials=n_trials,
        trial_returns_matrix=matrix, trial_sharpes=np.array(trial_sharpes),
        null_runs=null_runs, stability_cagrs=stability)
    print(g.to_frame().to_string(index=False))
    print("\n" + g.summary())

    section("8. PROMOTION DECISION")
    data_years = len(test["day"].unique()) / 252
    check = pipe.promote("vrp-001", Stage.TESTED)
    print(f"-> tested:    {check.allowed}  ({check.reason})")
    check = pipe.promote("vrp-001", Stage.VALIDATED, gauntlet=g,
                         data_years=data_years,
                         observed_sharpe=float(oos["stats"]["sharpe"] or 0))
    print(f"-> validated: {check.allowed}  ({check.reason})")
    check = pipe.promote("vrp-001", Stage.LIVE, paper_days=0)
    print(f"-> live:      {check.allowed}  ({check.reason})")

    section("9. RESEARCH DEBT -- what the search has cost in required data")
    print(pipe.research_debt().to_string(index=False))
    print("\nThis number rises with every backtest. That is the point: it puts")
    print("the price of one more parameter tweak in front of you at the moment")
    print("you are tempted to make it.")

    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
