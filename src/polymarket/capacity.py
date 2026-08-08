"""
What the longshot rule actually earns on capital, and what breaks it.

The per-trade numbers look like a money printer: 7.5c/share on an 85c token is
8.8% per trade, and the rule fired in 2,588 markets in a single out-of-sample
year. Multiplying those two together gives a number nobody should believe, and
this module exists to find out which of the steps between them is load-bearing.

The four things that turn a per-trade edge into an annual return, in the order
they bite:

  1. **Depth caps the stake.** Median depth at the touch in the rule's bands is
     $21-$245. You cannot deploy more than the book holds, so the per-trade
     percentage is applied to a very small base.
  2. **Capital is locked until resolution.** 8.8% per trade is not 8.8% per
     month. The same dollar can only be recycled as fast as markets settle,
     and that is the difference between a great return and a mediocre one.
  3. **Losses arrive together.** Short-longshot / long-favourite is ONE bet
     wearing many hats. When favourites lose, they lose across the whole book
     at once. This is the same structure that made XIV lose 96% in a night,
     and the reason the options project ruled leverage out rather than
     deferring it.
  4. **Adverse selection.** The historical edge is the average over all fills.
     A systematic taker does not get the average: they get filled most easily
     precisely when the resting side knows something. The haircut is unknown
     and unknowable from tape alone, so it is a parameter here, and the
     break-even value is reported.

Nothing in this module is measured. It is arithmetic on measured inputs, with
every assumption named, and its output is a sensitivity table rather than a
number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Band:
    """One price band the rule trades, with what the book actually showed."""
    name: str
    price: float                 # typical fill price for the claim taken
    edge_cents: float            # gross edge per share, before spread
    half_spread_cents: float     # measured
    depth_usd: float             # median at the touch
    opportunities_per_year: int


# Measured on 2026-08-08: walk-forward edges, live-book spread and depth.
# The 0.05-0.10 band is excluded -- its 4.44c edge lost to a 4.92c half-spread.
MEASURED_BANDS = [
    Band("0.10-0.20", price=0.15, edge_cents=8.81, half_spread_cents=1.00,
         depth_usd=46.2, opportunities_per_year=430),
    Band("0.80-0.90", price=0.85, edge_cents=8.54, half_spread_cents=1.00,
         depth_usd=244.6, opportunities_per_year=1_290),
    Band("0.90-0.95", price=0.925, edge_cents=4.30, half_spread_cents=0.85,
         depth_usd=54.1, opportunities_per_year=868),
]


@dataclass
class Assumptions:
    """
    Every judgement call, in one place, so the result can be argued with.

    `depth_fraction` is the share of displayed size taken. Taking 100% of the
    touch is the theoretical maximum and a practical fiction: the book refreshes
    against you, and a taker who always lifts the whole offer is the most
    predictable participant in the market.

    `adverse_selection` scales the realised edge down. It is the parameter that
    cannot be estimated from the tape, because the tape does not record which
    resting orders were informed.

    `fill_rate` is the share of intended trades that actually get done. The
    opportunity count comes from historical fills, which are trades that DID
    happen; an automated taker arriving later gets some fraction of them.
    """
    depth_fraction: float = 0.33
    adverse_selection: float = 0.30      # keep 70% of the historical edge
    fill_rate: float = 0.50
    hold_days: float = 21.0
    fee_bps_on_winnings: float = 0.0
    loss_correlation: float = 0.25       # within-year clustering of upsets
    capital_utilisation: float = 0.80    # never fully deployed


def per_trade_economics(b: Band, a: Assumptions) -> dict:
    """Stake, expected profit and per-trade risk for one band."""
    stake = b.depth_usd * a.depth_fraction
    shares = stake / b.price
    net_cents = (b.edge_cents * (1.0 - a.adverse_selection)) - b.half_spread_cents
    profit = shares * net_cents / 100.0
    # A binary claim bought at p pays 1 or 0. Per-share sd is sqrt(p(1-p)).
    sd_per_share = float(np.sqrt(b.price * (1.0 - b.price)))
    sd = shares * sd_per_share
    return {
        "band": b.name, "stake_usd": round(stake, 2),
        "net_edge_cents": round(net_cents, 3),
        "profit_usd": round(profit, 3),
        "roi_per_trade": round(profit / stake, 4) if stake else 0.0,
        "sd_usd": round(sd, 2),
        "sharpe_per_trade": round(profit / sd, 4) if sd else 0.0,
    }


def annual_model(bands=None, a: Assumptions | None = None) -> dict:
    """
    Roll per-trade economics up to a year, on the capital actually required.

    The capital number is the one that decides whether this is a business.
    Positions are open for `hold_days`, so the average number of simultaneously
    open positions is (trades per year) x (hold_days / 365), and the capital
    tied up is that many positions times the stake.
    """
    bands = bands or MEASURED_BANDS
    a = a or Assumptions()
    rows, total_profit, total_capital, var_sum, n_total = [], 0.0, 0.0, 0.0, 0
    for b in bands:
        e = per_trade_economics(b, a)
        n = int(b.opportunities_per_year * a.fill_rate)
        concurrent = n * (a.hold_days / 365.0)
        cap = concurrent * e["stake_usd"]
        rows.append({**e, "trades_per_year": n,
                     "concurrent_positions": round(concurrent, 1),
                     "capital_usd": round(cap, 0),
                     "annual_profit_usd": round(e["profit_usd"] * n, 0)})
        total_profit += e["profit_usd"] * n
        total_capital += cap
        var_sum += n * (e["sd_usd"] ** 2)
        n_total += n

    # Correlated losses: with pairwise correlation rho among n bets, the
    # variance of the sum is n*var*(1 + (n-1)*rho) rather than n*var. At any
    # realistic rho this term, not the bet count, sets the risk -- which is the
    # whole reason breadth arguments mislead in a one-directional book.
    rho = a.loss_correlation
    indep_var = var_sum
    corr_var = indep_var * (1.0 + max(n_total - 1, 0) * rho)
    sd_indep = float(np.sqrt(indep_var))
    sd_corr = float(np.sqrt(corr_var))

    capital = total_capital / max(a.capital_utilisation, 1e-9)
    return {
        "by_band": pd.DataFrame(rows),
        "trades_per_year": n_total,
        "annual_profit_usd": round(total_profit, 0),
        "capital_required_usd": round(capital, 0),
        "return_on_capital": round(total_profit / capital, 4) if capital else 0.0,
        # The closed form for correlated variance is reported for intuition
        # only and is NOT a loss distribution: it is unbounded, while a book of
        # long binary claims cannot lose more than it staked. Use
        # `drawdown_simulation` for anything that matters -- it is bounded by
        # construction because it simulates the actual payoffs.
        "annual_sd_if_independent_usd": round(sd_indep, 0),
        "annual_sd_with_correlation_usd": round(sd_corr, 0),
        "sharpe_if_independent": round(total_profit / sd_indep, 2) if sd_indep else 0.0,
        "sharpe_with_correlation": round(total_profit / sd_corr, 2) if sd_corr else 0.0,
        "sd_caveat": ("closed-form correlated sd is unbounded and overstates "
                      "the tail; use drawdown_simulation"),
    }


def breakeven_adverse_selection(bands=None, a: Assumptions | None = None) -> float:
    """
    How much of the historical edge can be lost before the rule stops paying.

    Reported because adverse selection is the one input that cannot be measured
    from the tape. If the rule only works when a taker keeps nearly all of the
    average edge, it is not a rule, it is a hope.
    """
    bands = bands or MEASURED_BANDS
    a = a or Assumptions()
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        trial = Assumptions(**{**a.__dict__, "adverse_selection": mid})
        if annual_model(bands, trial)["annual_profit_usd"] > 0:
            lo = mid
        else:
            hi = mid
    return round(lo, 4)


def sensitivity(bands=None, a: Assumptions | None = None) -> pd.DataFrame:
    """Return on capital across the assumptions that are genuinely uncertain."""
    bands = bands or MEASURED_BANDS
    base = a or Assumptions()
    rows = []
    for adv in (0.0, 0.15, 0.30, 0.50, 0.70):
        for fill in (0.25, 0.50, 0.80):
            trial = Assumptions(**{**base.__dict__, "adverse_selection": adv,
                                   "fill_rate": fill})
            m = annual_model(bands, trial)
            rows.append({
                "adverse_selection": adv, "fill_rate": fill,
                "annual_profit_usd": m["annual_profit_usd"],
                "capital_usd": m["capital_required_usd"],
                "return_on_capital": m["return_on_capital"],
                "sharpe_corr": m["sharpe_with_correlation"],
            })
    return pd.DataFrame(rows)


def drawdown_simulation(bands=None, a: Assumptions | None = None,
                        n_paths: int = 4000, seed: int = 0) -> dict:
    """
    Simulate a year of trading with CORRELATED outcomes, and report the tail.

    A one-directional short-longshot book does not experience its losses
    independently. A single common factor is imposed here -- a year in which
    upsets are more frequent than usual hits every favourite at once -- because
    the average return is not what ends accounts.
    """
    bands = bands or MEASURED_BANDS
    a = a or Assumptions()
    rng = np.random.default_rng(seed)
    rho = a.loss_correlation

    profits = np.zeros(n_paths)
    for b in bands:
        e = per_trade_economics(b, a)
        n = int(b.opportunities_per_year * a.fill_rate)
        if n <= 0:
            continue
        shares = e["stake_usd"] / b.price
        cost_per_share = b.price + b.half_spread_cents / 100.0
        # True win probability implied by the measured edge.
        p_true = min(max(b.price + b.edge_cents / 100.0
                         * (1.0 - a.adverse_selection), 0.01), 0.99)
        # Gaussian copula: one common factor per path, one idiosyncratic draw
        # per bet. rho is the shared loading.
        common = rng.standard_normal((n_paths, 1))
        idio = rng.standard_normal((n_paths, n))
        z = np.sqrt(rho) * common + np.sqrt(1 - rho) * idio
        from math import erf, sqrt as _sqrt
        thresh = np.vectorize(lambda q: _inv_norm(q))(np.array([p_true]))[0]
        wins = (z < thresh)
        payoff = wins * 1.0
        profits += (payoff.sum(axis=1) * shares
                    - n * shares * cost_per_share)

    cap = annual_model(bands, a)["capital_required_usd"]
    return {
        "median_annual_profit_usd": round(float(np.median(profits)), 0),
        "p05_annual_profit_usd": round(float(np.quantile(profits, 0.05)), 0),
        "p01_annual_profit_usd": round(float(np.quantile(profits, 0.01)), 0),
        "prob_losing_year": round(float((profits < 0).mean()), 3),
        "worst_1pct_return_on_capital": round(
            float(np.quantile(profits, 0.01)) / cap, 3) if cap else 0.0,
        "capital_usd": cap,
    }


def _inv_norm(p: float) -> float:
    """Inverse standard normal CDF (Acklam), for the copula threshold."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
