"""
Cost economics: how much edge each venue demands before it pays anything.

The output that matters is not "which venue has the lowest fees" -- it is
"which venue's fees are smallest relative to the size of the move you are
trying to forecast". Those two questions have different answers, and getting
them confused is the most common way a retail algo dies.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from .venues import Venue, ALL_VENUES, CAMP


# Holding periods to evaluate, in calendar days. Spans intraday scalping to a
# monthly rebalance, because the answer flips across this range.
HORIZONS = [0.25, 1, 3, 5, 10, 21, 63]


def cost_table(venues: list[Venue] | None = None,
               horizons: list[float] | None = None) -> pd.DataFrame:
    """Round-trip friction as a fraction of the move's own volatility."""
    venues = venues or ALL_VENUES
    horizons = horizons or HORIZONS
    rows = []
    for v in venues:
        row = {"venue": v.name, "camp": CAMP[v.name],
               "round_trip_bps": round(v.round_trip_cost_bps(), 1)}
        for h in horizons:
            row[f"c/n @{_fmt_h(h)}"] = round(v.cost_to_noise(h), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def min_viable_horizon(venue: Venue, target_ic: float,
                       max_days: float = 252.0) -> float | None:
    """
    Shortest holding period at which a forecast of skill `target_ic` still
    clears costs.

    `target_ic` is the per-bet information coefficient -- the correlation
    between forecast and realised return, which for a single bet equals the
    gross Sharpe of that bet. Published cross-sectional equity signals live
    around 0.02-0.05; a genuinely good one is 0.06+. Anything above 0.10
    sustained is a claim that should be disbelieved by default.

    Returns None if no horizon up to `max_days` works.
    """
    grid = np.concatenate([np.arange(0.05, 2, 0.05), np.arange(2, max_days, 0.5)])
    for h in grid:
        if venue.cost_to_noise(float(h)) < target_ic:
            return float(h)
    return None


def breakeven_ic(venue: Venue, holding_days: float) -> float:
    """The IC at which the strategy exactly pays its own costs and no more."""
    return venue.cost_to_noise(holding_days)


def viability_table(target_ics=(0.02, 0.05, 0.10),
                    venues: list[Venue] | None = None) -> pd.DataFrame:
    """
    For each venue and each level of assumed skill, the shortest holding period
    that is not automatically a loser.

    Reading this table is most of the decision. A venue whose minimum viable
    horizon is 20+ days at realistic skill is not a "trading algo" venue at
    small size; it is a slow rebalancing scheme.
    """
    venues = venues or ALL_VENUES
    rows = []
    for v in venues:
        row = {"venue": v.name, "camp": CAMP[v.name]}
        for ic in target_ics:
            h = min_viable_horizon(v, ic)
            row[f"min horizon @ IC={ic:.2f}"] = "never" if h is None else _fmt_h(h)
        rows.append(row)
    return pd.DataFrame(rows)


def capital_efficiency(venue: Venue, capital_usd: float,
                       holding_days: float) -> dict:
    """
    Annual cost drag as a percentage of capital, at full deployment.

    Turnover is the multiplier everyone underestimates: a 34bp fee looks small
    until you pay it 100 times a year.
    """
    turns_per_year = venue.sessions_per_year / max(holding_days, 1e-9)
    drag_bps = venue.round_trip_cost_bps() * turns_per_year
    carry_bps = venue.carry_bps_per_day * venue.sessions_per_year
    total_bps = drag_bps + carry_bps
    return {
        "venue": venue.name,
        "turns_per_year": round(turns_per_year, 1),
        "cost_drag_pct_per_year": round(total_bps / 100, 1),
        "data_cost_pct_per_year": round(
            venue.data_cost_usd_per_month * 12 / capital_usd * 100, 2),
        "total_hurdle_pct_per_year": round(
            total_bps / 100 + venue.data_cost_usd_per_month * 12 / capital_usd * 100, 1),
    }


def _fmt_h(h: float) -> str:
    if h < 1:
        return f"{h*24:.0f}h"
    return f"{h:.0f}d" if h == int(h) else f"{h:.1f}d"
