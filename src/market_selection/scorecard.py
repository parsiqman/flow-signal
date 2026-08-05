"""
The decision itself: hard gates first, then a weighted score, then a check on
whether the weights were doing the work.

A weighted scorecard is easy to rig -- pick the weights that produce the answer
you already wanted. `weight_sensitivity()` exists to catch exactly that. If a
venue only wins under one particular weighting, the scorecard has told you
nothing and you should say so.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from .venues import Venue, ALL_VENUES, CAMP
from .power import best_horizon, annual_ir, years_to_validate, can_be_validated

# Criteria, each scored 0-10 (higher is better).
CRITERIA = [
    "cost_economics",     # friction relative to the size of the move
    "statistical_power",  # bets per year -> how fast an edge proves itself
    "data_access",        # can you get free, deep, clean research data
    "legal_access",       # can a US retail person actually trade it
    "infra_fit",          # Chromebook + Colab + Render, no local machine
    "crowding_headroom",  # is the obvious edge already arbitraged
    "repo_carryover",     # how much of flow_backtest.py survives the choice
]

# Default weights. Deliberately close to uniform, with a thumb on the scale for
# the two things that kill projects before they start: not being able to trade
# it, and not being able to test it.
DEFAULT_WEIGHTS = {
    "cost_economics": 1.0,
    "statistical_power": 1.5,
    "data_access": 1.5,
    "legal_access": 1.5,
    "infra_fit": 0.75,
    "crowding_headroom": 1.0,
    "repo_carryover": 0.5,
}

# Judgment scores for criteria the cost model cannot compute. Each carries its
# reasoning so a future session can argue with the number rather than guess at
# where it came from.
QUALITATIVE = {
    "US cash equities (liquid universe)": {
        "infra_fit": 9,
        "crowding_headroom": 5,
        "repo_carryover": 6,
        "why": {
            "infra_fit": "4pm close is a natural batch boundary: one Render cron "
                         "a day, no always-on worker, no overnight babysitting.",
            "crowding_headroom": "Intraday is owned by HFT, but daily "
                                 "cross-sectional signals remain workable at "
                                 "retail size. Crowded, not closed.",
            "repo_carryover": "Detector/evaluate/sweep scaffolding transfers "
                              "wholesale; only the flow-specific filters go.",
        },
    },
    "US equity options (short-dated, event-driven)": {
        "infra_fit": 7,
        "crowding_headroom": 6,
        "repo_carryover": 10,
        "why": {
            "infra_fit": "Same batch window, but needs an intraday leg to act on "
                         "flow before the close.",
            "crowding_headroom": "Every flow-scanner subscriber sees the same "
                                 "prints simultaneously; the edge is in the "
                                 "filtering, which is where the crowd is worst.",
            "repo_carryover": "This IS the existing repo. Nothing is thrown away.",
        },
    },
    "Crypto spot on US CEX (Kraken Pro / Coinbase Advanced)": {
        "infra_fit": 6,
        "crowding_headroom": 6,
        "repo_carryover": 4,
        "why": {
            "infra_fit": "24/7 means no natural batch boundary and an always-on "
                         "Render worker; more moving parts for a cloud-only setup.",
            "crowding_headroom": "Long-tail alts are genuinely less efficient, but "
                                 "reported volume is polluted by wash trading, "
                                 "which corrupts the backtest before the strategy "
                                 "ever runs.",
            "repo_carryover": "Event/precision/recall harness transfers; the "
                              "options-specific machinery does not.",
        },
    },
    "Crypto perps, CFTC-regulated US (Kraken/Bitnomial, Coinbase FM)": {
        "infra_fit": 6,
        "crowding_headroom": 4,
        "repo_carryover": 3,
        "why": {
            "infra_fit": "Always-on, plus leverage means liquidation risk needs "
                         "monitoring a cron job cannot provide.",
            "crowding_headroom": "BTC/ETH perps are the most heavily quantified "
                                 "instruments in crypto. No retail edge hides "
                                 "in four majors.",
            "repo_carryover": "Little. A four-instrument directional book "
                              "shares no structure with the flow detector.",
        },
    },
    "On-chain DeFi (Uniswap/Aerodrome on Base/Arbitrum)": {
        "infra_fit": 4,
        "crowding_headroom": 7,
        "repo_carryover": 5,
        "why": {
            "infra_fit": "Requires hot private keys on a cloud host, RPC "
                         "management, nonce/revert handling, and per-swap tax "
                         "accounting. The heaviest ops burden of the six by far.",
            "crowding_headroom": "The one venue where the informed-flow analogue "
                                 "is FREE and public: wallet-level accumulation is "
                                 "visible on chain. But searchers see it first and "
                                 "act in the same block.",
            "repo_carryover": "The 'unusual accumulation ahead of an event' thesis "
                              "maps cleanly onto wallets and token unlocks; the "
                              "detector logic survives, the data layer does not.",
        },
    },
    "Offshore DeFi perps (Hyperliquid)": {
        "infra_fit": 5,
        "crowding_headroom": 6,
        "repo_carryover": 3,
        "why": {
            "infra_fit": "Cleanest API of the six and trivially scriptable, "
                         "which is irrelevant while access itself is barred.",
            "crowding_headroom": "Deep perp universe with real long-tail "
                                 "inefficiency; the best crypto venue on merit.",
            "repo_carryover": "Little. Perp funding/basis signals share no "
                              "structure with the options flow detector.",
        },
    },
}


def _scale(values: dict[str, float], higher_is_better: bool) -> dict[str, float]:
    """Map raw metric values onto 0-10, linearly between best and worst."""
    finite = [v for v in values.values() if np.isfinite(v)]
    if not finite:
        return {k: 0.0 for k in values}
    lo, hi = min(finite), max(finite)
    out = {}
    for k, v in values.items():
        if not np.isfinite(v):
            out[k] = 0.0
            continue
        if hi == lo:
            out[k] = 5.0
        else:
            frac = (v - lo) / (hi - lo)
            out[k] = 10 * (frac if higher_is_better else 1 - frac)
    return out


def score_venues(ic: float | None = None,
                 venues: list[Venue] | None = None) -> pd.DataFrame:
    """
    Score every venue on every criterion, computed where possible.

    `ic=None` uses each venue's own `plausible_ic`, which grants crypto its
    inefficiency claim and the options thesis its large per-bet edge rather
    than assuming all markets are equally forecastable. Pass a float to hold
    skill fixed across venues instead.
    """
    venues = venues or ALL_VENUES

    cost_raw, power_raw, data_raw = {}, {}, {}
    for v in venues:
        v_ic = v.plausible_ic if ic is None else ic
        h, _ = best_horizon(v, v_ic)
        cost_raw[v.name] = v.cost_to_noise(h)
        # Cap at a large finite number so an unvalidatable venue scores 0
        # rather than poisoning the whole scale with an inf.
        yrs = years_to_validate(v, v_ic, h)
        power_raw[v.name] = min(yrs, 100.0) if np.isfinite(yrs) else 100.0
        # Free history depth, penalised by any subscription cost.
        data_raw[v.name] = v.history_years_free - (v.data_cost_usd_per_month / 100.0)

    cost_s = _scale(cost_raw, higher_is_better=False)
    power_s = _scale(power_raw, higher_is_better=False)
    data_s = _scale(data_raw, higher_is_better=True)

    rows = []
    for v in venues:
        q = QUALITATIVE[v.name]
        rows.append({
            "venue": v.name,
            "camp": CAMP[v.name],
            "cost_economics": round(cost_s[v.name], 1),
            "statistical_power": round(power_s[v.name], 1),
            "data_access": round(data_s[v.name], 1),
            "legal_access": 0.0 if v.gate_legal_us_retail else (
                6.0 if v.gate_reasons else 10.0),
            "infra_fit": float(q["infra_fit"]),
            "crowding_headroom": float(q["crowding_headroom"]),
            "repo_carryover": float(q["repo_carryover"]),
            "hard_gate": "BLOCKED" if v.is_gated() else "",
        })
    return pd.DataFrame(rows)


def weighted_ranking(scores: pd.DataFrame,
                     weights: dict[str, float] | None = None,
                     apply_gates: bool = True) -> pd.DataFrame:
    """Collapse the scorecard to a ranking under one set of weights."""
    weights = weights or DEFAULT_WEIGHTS
    total_w = sum(weights.values())
    s = scores.copy()
    s["score"] = sum(s[c] * w for c, w in weights.items()) / total_w
    if apply_gates:
        s.loc[s["hard_gate"] == "BLOCKED", "score"] = np.nan
    return s.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)


def weight_sensitivity(scores: pd.DataFrame, n_draws: int = 20_000,
                       seed: int = 3, apply_gates: bool = True) -> pd.DataFrame:
    """
    Re-rank under 20k random weightings drawn from a flat Dirichlet.

    This is the honesty check on the whole exercise. A venue that wins under
    almost every plausible weighting is a real conclusion. A venue that wins
    only under the default weights means the default weights, not the evidence,
    picked the answer.
    """
    rng = np.random.default_rng(seed)
    mat = scores[CRITERIA].to_numpy()
    names = scores["venue"].to_list()
    blocked = (scores["hard_gate"] == "BLOCKED").to_numpy() if apply_gates \
        else np.zeros(len(names), dtype=bool)

    w = rng.dirichlet(np.ones(len(CRITERIA)), size=n_draws)   # (draws, criteria)
    totals = mat @ w.T                                        # (venues, draws)
    totals[blocked, :] = -np.inf

    winners = np.argmax(totals, axis=0)
    counts = np.bincount(winners, minlength=len(names))
    top2 = np.argsort(-totals, axis=0)[:2, :]
    top2_counts = np.bincount(top2.ravel(), minlength=len(names))

    return pd.DataFrame({
        "venue": names,
        "camp": [CAMP[n] for n in names],
        "pct_rank_1": (counts / n_draws * 100).round(1),
        "pct_top_2": (top2_counts / n_draws * 100).round(1),
    }).sort_values("pct_rank_1", ascending=False).reset_index(drop=True)


def gate_report(venues: list[Venue] | None = None,
                ic: float | None = None) -> pd.DataFrame:
    """Every hard blocker and testability verdict, stated plainly."""
    venues = venues or ALL_VENUES
    rows = []
    for v in venues:
        ok, why = can_be_validated(v, v.plausible_ic if ic is None else ic)
        rows.append({
            "venue": v.name,
            "legal_block": "YES" if v.gate_legal_us_retail else "",
            "testable": "yes" if ok else "NO",
            "verdict": why,
            "caveats": "; ".join(v.gate_reasons) if v.gate_reasons else "",
        })
    return pd.DataFrame(rows)
