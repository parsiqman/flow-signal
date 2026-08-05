"""
Statistical power: how long until you *know* whether the thing works.

This is the axis that actually binds a solo researcher, and it is the one
market-selection debates almost always skip. CLAUDE.md already names the
constraint -- priority #1 is walk-forward validation, tune on A, evaluate
frozen on B. That is only possible in a venue that hands you enough
independent bets to make period B mean something.

The engine is Grinold's fundamental law, run net of costs:

    per-bet net Sharpe  =  IC  -  cost/sigma
    annual IR           =  per-bet net Sharpe  x  sqrt(bets per year)
    years to t = 2      =  (2 / IR)^2

The middle term is why a venue can have wonderful cost economics and still be
useless: if it only offers eight instruments, sqrt(breadth) never rescues it.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from .venues import Venue, ALL_VENUES, CAMP, TRANSFER_COEFFICIENT

T_TARGET = 2.0  # t-stat required to call an edge real


def net_bet_sharpe(venue: Venue, ic: float, holding_days: float) -> float:
    """Per-bet Sharpe after friction. Negative means the venue eats the edge."""
    return ic - venue.cost_to_noise(holding_days)


def annual_ir(venue: Venue, ic: float, holding_days: float) -> float:
    """Annualised information ratio, net of costs and implementation loss."""
    s = net_bet_sharpe(venue, ic, holding_days)
    breadth = venue.effective_breadth(holding_days)
    return TRANSFER_COEFFICIENT * s * np.sqrt(breadth)


def years_to_validate(venue: Venue, ic: float, holding_days: float,
                      t_target: float = T_TARGET) -> float:
    """
    Years of out-of-sample data needed for the edge to clear `t_target`.

    Returns np.inf when the net edge is non-positive -- no amount of waiting
    proves a losing strategy is a winner.
    """
    ir = annual_ir(venue, ic, holding_days)
    if ir <= 0:
        return np.inf
    return (t_target / ir) ** 2


def best_horizon(venue: Venue, ic: float,
                 horizons=(0.25, 1, 2, 3, 5, 10, 21, 63)) -> tuple[float, float]:
    """
    The holding period that maximises net IR for a given level of skill.

    Assumes IC is horizon-invariant, which is generous to fast venues: real
    signals usually decay, so short horizons in practice do worse than this
    says. Treat the fast-venue results as an upper bound.
    """
    scored = [(h, annual_ir(venue, ic, h)) for h in horizons]
    h, ir = max(scored, key=lambda x: x[1])
    return h, ir


def power_table(ic: float = 0.05, venues: list[Venue] | None = None) -> pd.DataFrame:
    """One row per venue at its own best horizon, for a fixed level of skill."""
    venues = venues or ALL_VENUES
    rows = []
    for v in venues:
        h, ir = best_horizon(v, ic)
        yrs = years_to_validate(v, ic, h)
        rows.append({
            "venue": v.name,
            "camp": CAMP[v.name],
            "gated": "YES" if v.is_gated() else "",
            "best_horizon_d": h,
            "cost/noise": round(v.cost_to_noise(h), 3),
            "net_bet_sharpe": round(net_bet_sharpe(v, ic, h), 4),
            "breadth/yr": round(v.effective_breadth(h)),
            "net_IR": round(ir, 2),
            "yrs_to_t2": ("never" if np.isinf(yrs) else round(yrs, 2)),
            "free_history_yrs": v.history_years_free,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("net_IR", ascending=False).reset_index(drop=True)


def power_table_plausible(venues: list[Venue] | None = None) -> pd.DataFrame:
    """
    The fairer comparison: each venue evaluated at the level of skill that is
    actually attainable *there*.

    Holding IC fixed across venues quietly assumes every market is equally hard
    to forecast, which is exactly the assumption under dispute -- crypto's whole
    pitch is that it is less efficient, and the event-driven options thesis
    claims a much larger per-bet edge than any cross-sectional signal. Both
    claims get granted here, at face value, and the venues are compared anyway.
    """
    venues = venues or ALL_VENUES
    rows = []
    for v in venues:
        ic = v.plausible_ic
        h, ir = best_horizon(v, ic)
        yrs = years_to_validate(v, ic, h)
        ok, why = can_be_validated(v, ic)
        rows.append({
            "venue": v.name,
            "camp": CAMP[v.name],
            "assumed_IC": ic,
            "best_horizon_d": h,
            "cost/noise": round(v.cost_to_noise(h), 3),
            "breadth/yr": round(v.effective_breadth(h)),
            "net_IR": round(ir, 2),
            "yrs_to_t2": "never" if np.isinf(yrs) else round(yrs, 1),
            "history_cover": round(history_cover(v, ic), 3),
            "testable?": "yes" if ok else "no",
        })
    df = pd.DataFrame(rows)
    return df.sort_values("net_IR", ascending=False).reset_index(drop=True)


def can_be_validated(venue: Venue, ic: float) -> tuple[bool, str]:
    """
    The decisive feasibility test: does the free history available in this
    venue cover the out-of-sample period the edge needs to prove itself?

    Walk-forward needs roughly 2x the validation period -- half to tune on,
    half to evaluate frozen. A venue that needs 6 years of clean out-of-sample
    data but only offers 4 years of free history cannot be honestly tested,
    regardless of how attractive its economics look.
    """
    h, _ = best_horizon(venue, ic)
    yrs = years_to_validate(venue, ic, h)
    if np.isinf(yrs):
        return False, "net edge <= 0 at every horizon; nothing to validate"
    needed = yrs * 2
    if venue.history_years_free <= 0:
        return False, f"needs ~{needed:.1f}y; NO free history exists at all"
    ratio = venue.history_years_free / needed
    if ratio >= 1.0:
        return True, (f"needs ~{needed:.1f}y, has {venue.history_years_free:.0f}y "
                      f"({ratio:.1f}x cover)")
    if ratio >= 0.35:
        return False, (f"needs ~{needed:.1f}y, has {venue.history_years_free:.0f}y "
                       f"({ratio:.2f}x) - MARGINAL, closes with a slightly "
                       f"better signal")
    return False, (f"needs ~{needed:.0f}y, has {venue.history_years_free:.0f}y "
                   f"({ratio:.3f}x) - not close")


def history_cover(venue: Venue, ic: float) -> float:
    """Free history as a multiple of what a tune+test split would require."""
    h, _ = best_horizon(venue, ic)
    yrs = years_to_validate(venue, ic, h)
    if np.isinf(yrs) or yrs <= 0:
        return 0.0
    return venue.history_years_free / (2 * yrs)


def monte_carlo_check(venue: Venue, ic: float, holding_days: float,
                      years: float = 2.0, n_paths: int = 4000,
                      seed: int = 11) -> dict:
    """
    Simulation cross-check on the analytic IR, and -- more usefully -- the
    probability of being *fooled* after a given amount of data.

    Two failure modes get quantified here, and both are real:
      - p_false_negative: a genuine edge that fails to clear t=2 and gets
        discarded.
      - p_false_positive: a zero-edge strategy in the same venue that clears
        t=2 anyway. This is the number that should govern how much you trust
        a good-looking backtest.
    """
    rng = np.random.default_rng(seed)
    breadth = venue.effective_breadth(holding_days)
    n_bets = max(1, int(round(breadth * years)))
    sigma = venue.bet_sigma_bps(holding_days)
    cost = venue.holding_cost_bps(holding_days)
    # TC scales the realised edge, matching annual_ir(); the null keeps paying
    # full costs because implementation loss does not refund fees.
    mu_net = TRANSFER_COEFFICIENT * (ic * sigma - cost)

    # Real-edge paths.
    draws = rng.normal(mu_net, sigma, size=(n_paths, n_bets))
    t_real = draws.mean(axis=1) / (draws.std(axis=1, ddof=1) / np.sqrt(n_bets))

    # Zero-edge paths in the same venue, paying the same costs.
    null = rng.normal(-cost, sigma, size=(n_paths, n_bets))
    t_null = null.mean(axis=1) / (null.std(axis=1, ddof=1) / np.sqrt(n_bets))

    realised_ir = float(np.median(t_real) / np.sqrt(years)) if years > 0 else np.nan
    return {
        "venue": venue.name,
        "bets_in_window": n_bets,
        "analytic_IR": round(annual_ir(venue, ic, holding_days), 3),
        "simulated_IR": round(realised_ir, 3),
        "p_detect_real_edge": round(float((t_real > T_TARGET).mean()), 3),
        "p_false_positive": round(float((t_null > T_TARGET).mean()), 3),
    }


def skill_sweep(venue: Venue, ics=(0.02, 0.03, 0.05, 0.08, 0.12)) -> pd.DataFrame:
    """How the answer changes with how good you actually are."""
    rows = []
    for ic in ics:
        h, ir = best_horizon(venue, ic)
        yrs = years_to_validate(venue, ic, h)
        rows.append({
            "IC": ic,
            "best_horizon_d": h,
            "net_IR": round(ir, 2),
            "yrs_to_t2": "never" if np.isinf(yrs) else round(yrs, 2),
        })
    return pd.DataFrame(rows)
