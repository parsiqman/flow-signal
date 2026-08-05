"""
Market selection: crypto/DeFi vs traditional equities, decided on evidence.

    python src/market_selection/run_analysis.py

Prints the full chain of reasoning: cost economics, statistical power, hard
gates, weighted score, and the sensitivity checks that say whether any of it
is robust. DECISION.md is the prose write-up of this output.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow running as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from market_selection import economics, power, scorecard, venues as V
else:
    from . import economics, power, scorecard, venues as V

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

BASE_IC = 0.05        # assumed per-bet skill; swept later
CAPITAL = 25_000.0    # assumed starting capital


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    section("1. COST ECONOMICS - friction vs the size of the move (cost/noise)")
    print("Values are round-trip cost as a fraction of the position's own return sd.")
    print("A signal must have an information coefficient ABOVE this to earn anything.\n")
    print(economics.cost_table().to_string(index=False))

    section("2. MINIMUM VIABLE HOLDING PERIOD, by assumed skill")
    print(economics.viability_table().to_string(index=False))

    section(f"3. ANNUAL COST DRAG on ${CAPITAL:,.0f}, at each venue's best horizon")
    rows = []
    for v in V.ALL_VENUES:
        h, _ = power.best_horizon(v, v.plausible_ic)
        rows.append(economics.capital_efficiency(v, CAPITAL, h))
    print(pd.DataFrame(rows).to_string(index=False))

    section(f"4a. STATISTICAL POWER at a uniform IC={BASE_IC}")
    print("Same assumed skill everywhere, so only market structure differs.\n")
    print(power.power_table(ic=BASE_IC).to_string(index=False))

    section("4b. STATISTICAL POWER at each venue's own plausible IC")
    print("Grants crypto its inefficiency premium and the options thesis its")
    print("large per-bet edge. This is the comparison the decision rests on.\n")
    print(power.power_table_plausible().to_string(index=False))

    section("5. HARD GATES and testability")
    print(scorecard.gate_report().to_string(index=False))

    section("6. SCORECARD")
    scores = scorecard.score_venues()
    print(scores.to_string(index=False))

    section("7. RANKING under default weights")
    print(f"weights: {scorecard.DEFAULT_WEIGHTS}\n")
    ranked = scorecard.weighted_ranking(scores)
    print(ranked[["venue", "camp", "score", "hard_gate"]].to_string(index=False))

    section("8. WEIGHT SENSITIVITY - 20,000 random weightings")
    print("If the winner only wins under the default weights, the weights chose")
    print("the answer and this whole exercise proves nothing.\n")
    print(scorecard.weight_sensitivity(scores).to_string(index=False))

    section("9. SKILL SENSITIVITY - does the answer survive being less good?")
    for v in V.ALL_VENUES:
        if v.is_gated():
            continue
        print(f"\n-- {v.name}")
        print(power.skill_sweep(v).to_string(index=False))

    section("10. STRESS TEST - the least certain inputs")
    stress_report()

    section("11. FALSE-POSITIVE RISK - how much to trust a good backtest")
    print("Probability a ZERO-edge strategy clears t=2 anyway, after 2 years,")
    print("and the probability a REAL edge is missed in the same window.\n")
    mc = []
    for v in V.ALL_VENUES:
        h, _ = power.best_horizon(v, v.plausible_ic)
        mc.append(power.monte_carlo_check(v, v.plausible_ic, h, years=2.0))
    print(pd.DataFrame(mc).to_string(index=False))

    section("12. BOTTOM LINE")
    verdict(scores, ranked)


def stress_report() -> None:
    """
    Re-run the ranking after breaking the assumptions least supported by data.

    A conclusion that survives all four of these is worth acting on. One that
    flips under any of them needs the underlying number nailed down first.
    """
    import copy

    scenarios = {
        "baseline": lambda vs: vs,
        "US perp fees 3x worse (6bp -> 18bp)": _mutate(
            "Crypto perps, CFTC-regulated US (Kraken/Bitnomial, Coinbase FM)",
            commission_bps=18.0),
        "crypto vol 50% higher (costs hurt less)": _mutate_many(
            {"Crypto spot on US CEX (Kraken Pro / Coinbase Advanced)": dict(daily_vol_bps=600.0),
             "On-chain DeFi (Uniswap/Aerodrome on Base/Arbitrum)": dict(daily_vol_bps=900.0),
             "Crypto perps, CFTC-regulated US (Kraken/Bitnomial, Coinbase FM)": dict(daily_vol_bps=450.0)}),
        "equity spreads 2x worse (2bp -> 4bp)": _mutate(
            "US cash equities (liquid universe)", half_spread_bps=4.0),
        "crypto all-maker fills (Kraken 25bp maker only)": _mutate(
            "Crypto spot on US CEX (Kraken Pro / Coinbase Advanced)",
            commission_bps=25.0),
        "equity breadth halved (1500 -> 750 names)": _mutate(
            "US cash equities (liquid universe)", n_assets=750),
        "DeFi pool fees halved (25bp -> 12bp)": _mutate(
            "On-chain DeFi (Uniswap/Aerodrome on Base/Arbitrum)", commission_bps=12.0),
    }

    rows = []
    for label, mutate in scenarios.items():
        vs = mutate([copy.deepcopy(v) for v in V.ALL_VENUES])
        scores = scorecard.score_venues(venues=vs)
        ranked = scorecard.weighted_ranking(scores)
        winner = ranked.iloc[0]
        runner = ranked.iloc[1]
        rows.append({
            "scenario": label,
            "winner": winner["venue"],
            "score": round(winner["score"], 2),
            "runner_up": runner["venue"],
            "margin": round(winner["score"] - runner["score"], 2),
        })
    print(pd.DataFrame(rows).to_string(index=False))


def _mutate(venue_name: str, **changes):
    def f(vs):
        for v in vs:
            if v.name == venue_name:
                for k, val in changes.items():
                    setattr(v, k, val)
        return vs
    return f


def _mutate_many(spec: dict):
    def f(vs):
        for v in vs:
            for k, val in spec.get(v.name, {}).items():
                setattr(v, k, val)
        return vs
    return f


def verdict(scores: pd.DataFrame, ranked: pd.DataFrame) -> None:
    sens = scorecard.weight_sensitivity(scores)
    top = ranked.iloc[0]
    robust = sens.iloc[0]

    print(f"Highest score under default weights : {top['venue']}")
    print(f"Wins most random weightings         : {robust['venue']} "
          f"({robust['pct_rank_1']:.0f}% of 20k draws)")

    agree = top["venue"] == robust["venue"]
    print(f"\nDefault weights and random weights {'AGREE' if agree else 'DISAGREE'}.")
    if not agree:
        print("-> The default weights are doing the work. Treat the ranking as "
              "unresolved and nail down the disputed criteria first.")
        return

    camp_share = (sens.groupby("camp")["pct_rank_1"].sum()
                  .sort_values(ascending=False))
    print("\nShare of random weightings won, by camp:")
    for camp, pct in camp_share.items():
        print(f"  {camp:<14} {pct:5.1f}%")

    blocked = [v.name for v in V.ALL_VENUES if v.is_gated()]
    if blocked:
        print("\nRuled out by hard gates before scoring:")
        for b in blocked:
            v = next(x for x in V.ALL_VENUES if x.name == b)
            print(f"  - {b}\n      {' '.join(v.gate_reasons)}")


if __name__ == "__main__":
    main()
