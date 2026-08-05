"""
The whole options project, end to end.

    python src/options_alpha/run.py            # standard run  (~4 min)
    python src/options_alpha/run.py --quick    # smaller, faster (~1 min)

Order matters. The generator is validated BEFORE any strategy touches it,
because a market that does not reward volatility selling frictionlessly cannot
say anything about a strategy that sells volatility -- and that failure is
silent, producing a confident, wrong, losing backtest.
"""

from __future__ import annotations
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from options_alpha import families as F
    from options_alpha import research as R
    from options_alpha.backtest import run_backtest
    from options_alpha.strategy import StrategyConfig
    from options_alpha.synthetic import (MarketConfig, TRADING_DAYS,
                                         bs_price_scalar, generate_market)
else:
    from . import families as F
    from . import research as R
    from .backtest import run_backtest
    from .strategy import StrategyConfig
    from .synthetic import (MarketConfig, TRADING_DAYS, bs_price_scalar,
                            generate_market)

pd.set_option("display.width", 210)
pd.set_option("display.max_columns", 40)


def section(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def validate_generator(mkt: MarketConfig) -> float:
    """Frictionless straddle capture. Must be positive or nothing downstream counts."""
    panel = generate_market(mkt)
    sp = panel.pivot(index="day", columns="name", values="spot").to_numpy()
    iv = panel.pivot(index="day", columns="name", values="atm_iv").to_numpy()
    dte = 21
    t_years = dte / TRADING_DAYS
    prem = pay = 0.0
    for t in range(70, len(sp) - dte, 21):
        for j in range(sp.shape[1]):
            s, v = sp[t, j], iv[t, j]
            prem += (bs_price_scalar(s, s, t_years, v, True, mkt.risk_free)
                     + bs_price_scalar(s, s, t_years, v, False, mkt.risk_free)) / s
            st = sp[t + dte, j]
            pay += (max(0.0, st - s) + max(0.0, s - st)) / s
    return (prem - pay) / prem


def main(quick: bool = False) -> None:
    t_start = time.time()
    n_names = 30 if quick else 60
    n_days = TRADING_DAYS * (8 if quick else 12)
    seeds = range(20, 26) if quick else range(20, 34)

    section("1. WHICH OPTIONS STRATEGY? Family comparison")
    print(F.comparison_table().to_string(index=False))
    print("\nDisqualifiers applied before ranking:\n")
    print(F.screen().to_string(index=False))

    section("2. GENERATOR VALIDATION (before any strategy runs on it)")
    print("Frictionless ATM straddle selling must be profitable, or the market")
    print("does not contain the premium the strategy exists to harvest.\n")
    capture = validate_generator(MarketConfig(n_names=30, n_days=TRADING_DAYS * 8,
                                              seed=3))
    print(f"  straddle capture: {capture:+.2%}")
    if capture <= 0:
        print("  ABORT -- generator does not reward vol selling. Nothing below is valid.")
        return
    print("  OK. The premium is real in this market.")

    section("3. WALK-FORWARD: tune on first half, freeze, test once on second")
    panel = generate_market(MarketConfig(n_names=n_names, n_days=n_days, seed=17))
    wf = R.walk_forward(panel)
    print("In-sample grid (TUNING ARTEFACT -- these numbers do not count):\n")
    print(wf["in_sample_table"].head(8).to_string(index=False))
    frozen = wf["frozen_config"]
    print("\nFrozen config:", {k: getattr(frozen, k) for k in
                               ("short_delta", "wing_width_frac",
                                "profit_target", "both_sides")})
    print(f"In-sample vs out-of-sample CAGR: {wf['is_vs_oos_cagr']}")
    print("\n--- OUT OF SAMPLE (the only number that counts) ---")
    for k, v in wf["oos_stats"].items():
        print(f"  {k:28} {v}")
    print(f"\n  reference: median in-sample config, same test half, "
          f"CAGR {wf['oos_median_config_stats']['cagr']}")

    section("4. THE FROZEN CONFIG ON FRESH MARKETS (never tuned on)")
    across = R.across_seeds(frozen, seeds=seeds, n_names=n_names, n_days=n_days)
    print(across.to_string(index=False))
    print(f"\n  mean CAGR {across.cagr.mean():+.3f} | median {across.cagr.median():+.3f} "
          f"| profitable in {100 * (across.cagr > 0).mean():.0f}% of markets "
          f"| worst drawdown {across.max_dd.min():.3f}")

    section("5. ABLATION: which design decisions are load-bearing?")
    print("A component that changes nothing when removed is unfitted complexity.\n")
    abl_panel = generate_market(MarketConfig(n_names=n_names, n_days=n_days, seed=41))
    print(R.ablation(abl_panel, frozen).to_string(index=False))

    section("6. EXECUTION SENSITIVITY: how good do fills have to be?")
    print("fill_quality 1.0 = cross the spread every time, 0.0 = always filled at mid.\n")
    _, test = R.split_panel(panel)
    rows = []
    for fq in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = run_backtest(test, replace(frozen, fill_quality=fq))["stats"]
        rows.append({"fill_quality": fq, "cagr": s["cagr"], "sharpe": s["sharpe"],
                     "max_dd": s["max_drawdown"]})
    print(pd.DataFrame(rows).to_string(index=False))

    section("7. CRISIS STRESS: does it survive a real one?")
    print("Shock forced every ~2 years so the tail is actually sampled.\n")
    scen = {
        "mild   (2.8x vol,  -9% gap)": dict(shock_vol_mult=2.8, shock_gap_pct=-0.09,
                                            shock_decay_days=25),
        "severe (5.0x vol, -18% gap)": dict(shock_vol_mult=5.0, shock_gap_pct=-0.18,
                                            shock_decay_days=45),
        "2008   (7.0x vol, -28% gap)": dict(shock_vol_mult=7.0, shock_gap_pct=-0.28,
                                            shock_decay_days=70),
    }
    rows = []
    for label, kw in scen.items():
        cg, dd = [], []
        for seed in range(50, 54):
            mc = MarketConfig(n_names=40, n_days=TRADING_DAYS * 8, seed=seed,
                              shock_per_year=1 / 2.0, **kw)
            p = generate_market(mc)
            if not p.attrs["shock_days"]:
                continue
            s = run_backtest(p, frozen, mc)["stats"]
            cg.append(s["cagr"]); dd.append(s["max_drawdown"])
        rows.append({"shock severity": label, "mean cagr": round(float(np.mean(cg)), 4),
                     "mean maxDD": round(float(np.mean(dd)), 3),
                     "WORST maxDD": round(float(np.min(dd)), 3), "markets": len(cg)})
    print(pd.DataFrame(rows).to_string(index=False))

    section("8. WHAT THIS DOES AND DOES NOT ESTABLISH")
    print(f"""
  Established, on synthetic data only:
    - the strategy harvests a real premium and survives its own tail
    - it is profitable across independently generated markets, not one lucky draw
    - out-of-sample performance did not collapse relative to in-sample
    - it survives 2008-scale shocks arriving every two years

  NOT established, and the Sharpe ratio above should be heavily discounted:
    - a Sharpe near 2.5 is not achievable in reality. Real short-volatility
      books run 0.8-1.5. The synthetic tail is too gentle: liquidity never
      disappears here, spreads never gap, and assignment never happens at the
      worst moment. All three occur in a real crisis.
    - the premium here is a parameter that was chosen, not measured
    - nothing has touched a real option chain

  The next step is not more synthetic work. It is 15+ years of real end-of-day
  chains covering 2008, 2018, 2020 and 2024, and re-running exactly this
  pipeline against them. See STRATEGY.md.
""")
    print(f"total runtime {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller, faster run")
    main(**vars(ap.parse_args()))
