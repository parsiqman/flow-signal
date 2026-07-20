"""
Walk-forward validation for the flow -> M&A signal (Priority #1 in CLAUDE.md).

The threshold sweep in flow_backtest.py tunes and reports on the same period,
which overfits by construction. Here:

  1. Generate a 2-year synthetic market.
  2. Split chronologically: year 1 = tuning period, year 2 = held-out test.
  3. Grid-search SignalConfig thresholds on the tuning period only.
  4. Freeze the winning config and evaluate it once on the test period.
  5. Report in-sample vs out-of-sample precision / recall / ROI.

Boundary handling: events announced within `embargo_days` business days after
the split are dropped from BOTH evaluations, because their pre-announcement
leak flow straddles the boundary (the test period would see the announcement
but not the leak). Leak flow from a dropped event still sits in the tape as
noise, which is conservative for precision on both sides.

Selection criterion: highest tuning-period ROI among configs with at least
`min_alerts` tuning alerts. The alert floor stops the grid from "winning" with
a config that fired twice and got lucky -- exactly the overfit this exercise
exists to expose.
"""

from __future__ import annotations
import itertools

import pandas as pd

from flow_backtest import (SignalConfig, detect_signals, evaluate,
                           simulate_pnl, generate_synthetic_market)

LOOKAHEAD_DAYS = 5


def split_market(flow: pd.DataFrame, events: pd.DataFrame, split_date,
                 embargo_days: int = 5):
    """Chronological split. Returns (flow_a, events_a, flow_b, events_b).

    Catalysts are NOT split: earnings/FDA dates are scheduled and known in
    advance, so using the full calendar in both periods is not lookahead.
    """
    split_date = pd.Timestamp(split_date)
    embargo_end = split_date + pd.tseries.offsets.BDay(embargo_days)

    flow_a = flow[flow["date"] < split_date]
    flow_b = flow[flow["date"] >= split_date]

    ann = pd.to_datetime(events["announce_date"])
    events_a = events[ann < split_date]
    events_b = events[ann >= embargo_end]
    n_dropped = len(events) - len(events_a) - len(events_b)
    return flow_a, events_a, flow_b, events_b, n_dropped


def build_grid() -> list[SignalConfig]:
    """Config grid searched on the tuning period."""
    grid = []
    for prem, voloi, askp, catwin, accum in itertools.product(
            [100_000, 150_000, 250_000],
            [2.0, 3.0],
            [0.75, 0.85],
            [0, 5, 7],
            [1, 2]):
        grid.append(SignalConfig(min_daily_premium=prem,
                                 min_vol_oi_ratio=voloi,
                                 min_ask_side_pct=askp,
                                 exclude_catalyst_window=catwin,
                                 accumulation_days=accum))
    return grid


def score_config(cfg: SignalConfig, flow, events, catalysts) -> dict:
    alerts = detect_signals(flow, cfg, catalysts)
    ev = evaluate(alerts, events, lookahead_days=LOOKAHEAD_DAYS)
    pnl = simulate_pnl(ev, events)
    return {"config": cfg,
            "n_alerts": ev["n_alerts"],
            "hits": ev["hits"],
            "precision": ev["precision"],
            "events_caught": ev["events_caught"],
            "n_events": ev["n_events"],
            "recall": ev["recall"],
            "roi": pnl["roi"]}


def tune(flow, events, catalysts, grid: list[SignalConfig],
         min_alerts: int = 10) -> tuple[SignalConfig, pd.DataFrame]:
    """Grid-search on the tuning period; return best config + full table."""
    rows = [score_config(cfg, flow, events, catalysts) for cfg in grid]
    table = pd.DataFrame(rows).sort_values("roi", ascending=False)
    eligible = table[table["n_alerts"] >= min_alerts]
    if eligible.empty:
        raise RuntimeError(f"No config produced >= {min_alerts} tuning alerts")
    return eligible.iloc[0]["config"], table


def fmt(r: dict) -> str:
    prec = f'{r["precision"]:.3f}' if r["n_alerts"] else "n/a"
    roi = f'{r["roi"]:+.0%}' if r["n_alerts"] else "n/a"
    return (f'alerts={r["n_alerts"]:>4}  hits={r["hits"]:>3}  '
            f'precision={prec}  '
            f'recall={r["events_caught"]}/{r["n_events"]} ({r["recall"]:.2f})  '
            f'roi={roi}')


def run_walk_forward(seed: int, verbose: bool = False):
    """One tune-on-A / test-on-B pass. Returns (in_sample, out_sample, cfg)."""
    flow, events, catalysts = generate_synthetic_market(
        n_days=504, n_ma_events=24, seed=seed)
    split_date = sorted(flow["date"].unique())[252]
    flow_a, events_a, flow_b, events_b, n_dropped = split_market(
        flow, events, split_date, embargo_days=LOOKAHEAD_DAYS)

    grid = build_grid()
    best_cfg, table = tune(flow_a, events_a, catalysts, grid)
    in_sample = score_config(best_cfg, flow_a, events_a, catalysts)
    out_sample = score_config(best_cfg, flow_b, events_b, catalysts)

    if verbose:
        print(f"  flow records: {len(flow):,} | "
              f"split at {pd.Timestamp(split_date).date()}")
        print(f"  tuning period: {len(flow_a):,} records, {len(events_a)} events"
              f" | test period: {len(flow_b):,} records, {len(events_b)} events"
              f" | {n_dropped} boundary events embargoed\n")
        print(f"Tuned {len(grid)} configs on period A only. Selected "
              f"(best tuning ROI, >=10 alerts):\n  {best_cfg.label()}\n")
        top = table.head(5).copy()
        top["config"] = top["config"].apply(lambda c: c.label())
        print("Top 5 tuning-period configs:")
        print(top.to_string(index=False,
                            formatters={"precision": "{:.3f}".format,
                                        "recall": "{:.2f}".format,
                                        "roi": "{:+.0%}".format}))
        print("\nFrozen config, in-sample (period A, tuned on it):")
        print(f"  {fmt(in_sample)}")
        print("Frozen config, OUT-OF-SAMPLE (period B, never seen in tuning):")
        print(f"  {fmt(out_sample)}")
    return in_sample, out_sample, best_cfg


def main(n_seeds: int = 10):
    print("Walk-forward validation: 2-year synthetic market (400 tickers, "
          "24 M&A events),\nyear 1 = tuning, year 2 = frozen out-of-sample "
          "test.\n")
    print(f"--- Seed 42 (detailed) {'-' * 40}")
    results = [run_walk_forward(42, verbose=True)]

    print(f"\n--- Repeating across {n_seeds - 1} more market seeds {'-' * 24}")
    for seed in range(100, 100 + n_seeds - 1):
        res = run_walk_forward(seed)
        print(f"  seed {seed}: IS  {fmt(res[0])}")
        print(f"            OOS {fmt(res[1])}")
        results.append(res)

    agg = pd.DataFrame(
        [{"metric": k,
          "in_sample_mean": pd.Series([r[0][k] for r in results]).mean(),
          "out_of_sample_mean": pd.Series([r[1][k] for r in results]).mean(),
          "degradation_mean": pd.Series([r[1][k] - r[0][k]
                                         for r in results]).mean(),
          "degradation_std": pd.Series([r[1][k] - r[0][k]
                                        for r in results]).std()}
         for k in ("precision", "recall", "roi")])
    print(f"\n=== Aggregate over {len(results)} seeds "
          "(degradation = OOS - IS for the per-seed selected config) ===")
    print(agg.to_string(index=False, float_format="{:+.3f}".format))
    print("\nIn-sample numbers benefit from picking the best of "
          f"{len(build_grid())} configs on the same period they're scored on; "
          "the out-of-sample\ncolumn is the honest estimate of live "
          "performance. Per-seed swings show how much single-period results "
          "are luck.")
    return results, agg


if __name__ == "__main__":
    import sys
    main(n_seeds=int(sys.argv[1]) if len(sys.argv) > 1 else 10)
