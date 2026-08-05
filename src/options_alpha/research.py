"""
Walk-forward parameter search, with the split fixed before anything is fitted.

The single discipline that matters here: the grid is searched on the FIRST half
of the sample only, one configuration is frozen, and it is then run once on the
second half. The test-half number is the only one that means anything. Every
other figure in this module is a tuning artefact and is labelled as such.

Two guards against fooling ourselves, beyond the split itself:

  - the selection criterion is declared up front (below), not chosen after
    looking at which configuration won
  - the spread between the best and the median configuration is reported. If
    the best config is wildly better than typical, the search found noise, and
    the out-of-sample result should be expected to collapse toward the median
"""

from __future__ import annotations
from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .strategy import StrategyConfig
from .synthetic import MarketConfig


# Declared BEFORE running anything: risk-adjusted growth, with a hard floor on
# survivability. A configuration that returns 12% a year through a 60% drawdown
# is not better than one returning 6% through 15% -- the first one gets
# abandoned in month three of the drawdown, in real life, by a real person.
MAX_ACCEPTABLE_DD = 0.30


def objective(stats: dict) -> float:
    """Higher is better. Non-survivors score negative, regardless of return."""
    if stats["cagr"] <= 0:
        return stats["cagr"]
    if abs(stats["max_drawdown"]) > MAX_ACCEPTABLE_DD:
        return -abs(stats["max_drawdown"])
    return stats["cagr"] / max(abs(stats["max_drawdown"]), 0.01)


def split_panel(panel: pd.DataFrame, frac: float = 0.5):
    """Chronological split. Never random -- that would leak the future."""
    cutoff = int(panel["day"].max() * frac)
    tune = panel[panel["day"] <= cutoff].copy()
    test = panel[panel["day"] > cutoff].copy()
    test["day"] = test["day"] - test["day"].min()
    for part, lo, hi in ((tune, 0, cutoff), (test, cutoff, panel["day"].max())):
        part.attrs["shock_days"] = [s - (lo if part is test else 0)
                                    for s in panel.attrs.get("shock_days", [])
                                    if lo < s <= hi]
        part.attrs["config"] = panel.attrs.get("config")
    return tune, test


DEFAULT_GRID = {
    "short_delta": [0.10, 0.16, 0.25],
    "wing_width_frac": [0.4, 0.8, 1.5],
    "profit_target": [0.5, 1.0],
    "both_sides": [True, False],
}


def sweep(panel: pd.DataFrame, grid: dict | None = None,
          base: StrategyConfig | None = None,
          mkt: MarketConfig | None = None) -> pd.DataFrame:
    """Run every configuration in the grid over one panel."""
    grid = grid or DEFAULT_GRID
    base = base or StrategyConfig()
    keys = list(grid)
    rows = []
    for combo in product(*(grid[k] for k in keys)):
        cfg = replace(base, **dict(zip(keys, combo)))
        stats = run_backtest(panel, cfg, mkt)["stats"]
        rows.append({**dict(zip(keys, combo)),
                     "cagr": stats["cagr"],
                     "max_dd": stats["max_drawdown"],
                     "sharpe": stats["sharpe"],
                     "worst_month": stats["worst_month"],
                     "n_trades": stats["n_trades"],
                     "score": objective(stats)})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def walk_forward(panel: pd.DataFrame, grid: dict | None = None,
                 base: StrategyConfig | None = None,
                 mkt: MarketConfig | None = None) -> dict:
    """
    Tune on the first half, freeze, evaluate once on the second half.

    Returns the frozen config, the in-sample table, and the single out-of-sample
    result. Also runs the *median* in-sample configuration out-of-sample, as a
    reference point: if the tuned config does no better than the median one,
    the tuning added nothing and the parameters are noise.
    """
    grid = grid or DEFAULT_GRID
    tune, test = split_panel(panel)

    table = sweep(tune, grid, base, mkt)
    keys = [k for k in grid]
    best_row = table.iloc[0]
    frozen = replace(base or StrategyConfig(),
                     **{k: _cast(best_row[k]) for k in keys})

    median_row = table.iloc[len(table) // 2]
    median_cfg = replace(base or StrategyConfig(),
                         **{k: _cast(median_row[k]) for k in keys})

    oos_best = run_backtest(test, frozen, mkt)
    oos_median = run_backtest(test, median_cfg, mkt)

    return {
        "frozen_config": frozen,
        "in_sample_table": table,
        "in_sample_best": dict(best_row),
        "oos_stats": oos_best["stats"],
        "oos_curve": oos_best["curve"],
        "oos_trades": oos_best["trades"],
        "oos_median_config_stats": oos_median["stats"],
        "overfit_gap": round(float(best_row["score"] - table["score"].median()), 3),
        "is_vs_oos_cagr": (round(float(best_row["cagr"]), 4),
                           round(float(oos_best["stats"]["cagr"]), 4)),
    }


def _cast(v):
    """Grid values round-trip through a DataFrame, so bools come back as objects."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    f = float(v)
    return f


def ablation(panel: pd.DataFrame, cfg: StrategyConfig,
             mkt: MarketConfig | None = None) -> pd.DataFrame:
    """
    Turn each design decision off, one at a time, and see what it was worth.

    This is how you find out which parts of a strategy are load-bearing and
    which are decoration. A component that changes nothing when removed should
    be removed -- it is unfitted complexity waiting to break.
    """
    variants = {
        "full strategy": cfg,
        "regime filter ON": replace(cfg, use_regime_filter=True),
        "no earnings exclusion": replace(cfg, exclude_earnings=False),
        # Drops the "only sell vol that is rich" floor. Still takes the 20
        # richest names, so this is not "sell everything" -- it is "sell the
        # richest 20 even when none of them are actually rich".
        "no richness floor": replace(cfg, min_richness=-9.0),
        "no profit target (hold to expiry)": replace(cfg, profit_target=1.0),
        # NB: risk-budget sizing normalises by max loss, so widening the wings
        # also shrinks position size. This row conflates the two and is a
        # leverage comparison, not a wings-vs-naked comparison.
        "4x wider wings (also 4x smaller)": replace(
            cfg, wing_width_frac=cfg.wing_width_frac * 4),
        "half the position size": replace(cfg, risk_per_position=cfg.risk_per_position / 2),
    }
    rows = []
    for label, v in variants.items():
        s = run_backtest(panel, v, mkt)["stats"]
        rows.append({"variant": label, "cagr": s["cagr"], "max_dd": s["max_drawdown"],
                     "sharpe": s["sharpe"], "worst_month": s["worst_month"],
                     "worst_shock_dd": s["worst_shock_drawdown"],
                     "n_trades": s["n_trades"]})
    return pd.DataFrame(rows)


def across_seeds(cfg: StrategyConfig, seeds=range(20, 32),
                 n_names: int = 60, n_days: int = 252 * 12) -> pd.DataFrame:
    """
    Re-run the frozen config on freshly generated markets.

    A single synthetic history is one draw. If the strategy only works on the
    market it was tuned against, that shows up here as a wide, mostly negative
    spread -- and no amount of in-sample polish will fix it.
    """
    from .synthetic import generate_market
    rows = []
    for sd in seeds:
        panel = generate_market(MarketConfig(n_names=n_names, n_days=n_days, seed=sd))
        s = run_backtest(panel, cfg)["stats"]
        rows.append({"seed": sd, "cagr": s["cagr"], "max_dd": s["max_drawdown"],
                     "sharpe": s["sharpe"], "n_shocks": s["n_shocks_in_sample"],
                     "worst_shock_dd": s["worst_shock_drawdown"]})
    return pd.DataFrame(rows)
