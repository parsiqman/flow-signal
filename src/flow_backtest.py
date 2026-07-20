"""
Unusual Options Flow -> M&A Signal Backtest Framework
======================================================
Tests whether a filter on options flow (like the PYPL/Stripe screenshot pattern)
would have flagged real acquisition targets BEFORE announcement -- and at what
false-positive cost.

Core idea:
  1. Take a tape of options flow records (real or synthetic).
  2. Apply a configurable "informed trading" filter (OTM calls, short DTE,
     ask-side sweeps, volume >> open interest, no scheduled catalyst).
  3. An alert is a HIT if that ticker announces an acquisition within
     `lookahead_days`. Otherwise it's a false positive.
  4. Report precision / recall and a lottery-ticket P&L simulation.

Runs on synthetic data by default so the mechanics are testable without a
paid data feed. Swap in real data via CSV (see README / notebook cells).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ----------------------------------------------------------------------------
# 1. Signal configuration
# ----------------------------------------------------------------------------

@dataclass
class SignalConfig:
    """Thresholds defining what counts as 'suspicious' flow."""
    min_premium: float = 25_000        # min $ premium per record
    min_daily_premium: float = 150_000 # min total $ premium per ticker-day
    min_vol_oi_ratio: float = 2.0      # volume must exceed OI by this factor (new positioning)
    moneyness_band: tuple = (1.01, 1.10)  # strike/spot: 1-10% OTM calls
    max_dte: int = 14                  # short-dated only
    min_ask_side_pct: float = 0.75     # fraction of premium executed at/above ask
    exclude_catalyst_window: int = 7   # drop alerts within N days of scheduled catalyst
    accumulation_days: int = 1         # require signal on >= N days in trailing window
    accumulation_window: int = 3

    def label(self) -> str:
        return (f"prem>={self.min_daily_premium/1000:.0f}k, vol/OI>={self.min_vol_oi_ratio}, "
                f"OTM {self.moneyness_band}, DTE<={self.max_dte}, "
                f"ask%>={self.min_ask_side_pct}, no-catalyst={self.exclude_catalyst_window}d")


# ----------------------------------------------------------------------------
# 2. Detection
# ----------------------------------------------------------------------------

def detect_signals(flow: pd.DataFrame, config: SignalConfig,
                   catalysts: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    flow columns required:
      date, ticker, opt_type ('C'/'P'), strike, spot, dte, volume,
      open_interest, premium, ask_side_pct, exec_type ('sweep'/'block'/'normal')
    catalysts columns (optional): ticker, date  (earnings, FDA dates, etc.)

    Returns one row per (ticker, date) alert.
    """
    f = flow.copy()
    f["moneyness"] = f["strike"] / f["spot"]

    lo, hi = config.moneyness_band
    mask = (
        (f["opt_type"] == "C")
        & (f["premium"] >= config.min_premium)
        & (f["moneyness"].between(lo, hi))
        & (f["dte"] <= config.max_dte)
        & (f["volume"] >= config.min_vol_oi_ratio * f["open_interest"].clip(lower=1))
        & (f["ask_side_pct"] >= config.min_ask_side_pct)
    )
    candidates = f[mask]
    if candidates.empty:
        return pd.DataFrame(columns=["ticker", "date", "total_premium", "n_records"])

    daily = (candidates.groupby(["ticker", "date"])
             .agg(total_premium=("premium", "sum"),
                  n_records=("premium", "size"),
                  avg_vol_oi=("volume", "sum"))
             .reset_index())
    daily = daily[daily["total_premium"] >= config.min_daily_premium]

    # Catalyst exclusion: unexplained urgency is the tell. Flow right before
    # earnings is almost always just earnings speculation.
    if catalysts is not None and not catalysts.empty and config.exclude_catalyst_window > 0:
        cat = catalysts.copy()
        cat["date"] = pd.to_datetime(cat["date"])
        daily["date"] = pd.to_datetime(daily["date"])
        merged = daily.merge(cat.rename(columns={"date": "cat_date"}), on="ticker", how="left")
        merged["days_to_cat"] = (merged["cat_date"] - merged["date"]).dt.days
        near = merged[(merged["days_to_cat"] >= 0)
                      & (merged["days_to_cat"] <= config.exclude_catalyst_window)]
        drop_keys = set(zip(near["ticker"], near["date"]))
        daily = daily[~daily.apply(lambda r: (r["ticker"], r["date"]) in drop_keys, axis=1)]

    # Multi-day accumulation requirement
    if config.accumulation_days > 1:
        daily = daily.sort_values(["ticker", "date"])
        keep = []
        for tkr, grp in daily.groupby("ticker"):
            dates = pd.to_datetime(grp["date"]).reset_index(drop=True)
            for i in range(len(dates)):
                window_start = dates[i] - pd.Timedelta(days=config.accumulation_window)
                n_in_window = ((dates >= window_start) & (dates <= dates[i])).sum()
                if n_in_window >= config.accumulation_days:
                    keep.append((tkr, grp["date"].iloc[i]))
        keys = set(keep)
        daily = daily[daily.apply(lambda r: (r["ticker"], r["date"]) in keys, axis=1)]

    return daily.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3. Evaluation against announcement dates
# ----------------------------------------------------------------------------

def evaluate(alerts: pd.DataFrame, events: pd.DataFrame,
             lookahead_days: int = 5) -> dict:
    """
    events columns: ticker, announce_date, jump_pct (stock move on announcement)
    An alert is a hit if its ticker announces within `lookahead_days` after the alert.
    Recall = fraction of events that had >=1 alert in the window before them.
    """
    a = alerts.copy()
    e = events.copy()
    if a.empty:
        return {"n_alerts": 0, "hits": 0, "precision": np.nan,
                "events_caught": 0, "n_events": len(e), "recall": 0.0,
                "alerts": a.assign(hit=[]), "caught_events": []}
    a["date"] = pd.to_datetime(a["date"])
    e["announce_date"] = pd.to_datetime(e["announce_date"])

    ev_map = e.groupby("ticker")["announce_date"].apply(list).to_dict()

    def is_hit(row):
        for ad in ev_map.get(row["ticker"], []):
            delta = (ad - row["date"]).days
            if 0 < delta <= lookahead_days:
                return True
        return False

    a["hit"] = a.apply(is_hit, axis=1)

    caught = []
    for _, ev in e.iterrows():
        pre = a[(a["ticker"] == ev["ticker"])
                & (a["date"] < ev["announce_date"])
                & (a["date"] >= ev["announce_date"] - pd.Timedelta(days=lookahead_days))]
        if len(pre) > 0:
            caught.append(ev["ticker"])

    n_alerts = len(a)
    hits = int(a["hit"].sum())
    return {
        "n_alerts": n_alerts,
        "hits": hits,
        "precision": hits / n_alerts if n_alerts else np.nan,
        "events_caught": len(caught),
        "n_events": len(e),
        "recall": len(caught) / len(e) if len(e) else np.nan,
        "alerts": a,
        "caught_events": caught,
    }


def simulate_pnl(eval_result: dict, events: pd.DataFrame,
                 stake_per_alert: float = 1_000,
                 loss_on_miss: float = -0.85,
                 rng: np.random.Generator | None = None) -> dict:
    """
    Lottery-ticket P&L: fixed stake per alert.
    Misses: OTM short-dated calls decay; assume avg -85% (parameterizable).
    Hits: payoff scales with the announcement jump. A 5% OTM weekly call on a
    stock that jumps 20-60% overnight returns roughly 5x-40x; we approximate
    payoff_multiple = max(0, (jump - otm_gap)) / est_option_cost_frac.
    Deliberately rough -- the point is order-of-magnitude economics.
    """
    rng = rng or np.random.default_rng(7)
    a = eval_result["alerts"]
    if a.empty:
        return {"total_pnl": 0.0, "spent": 0.0, "roi": np.nan}
    e = events.copy()
    e["announce_date"] = pd.to_datetime(e["announce_date"])
    jump_map = e.set_index("ticker")["jump_pct"].to_dict()

    pnl = []
    for _, row in a.iterrows():
        if row["hit"]:
            jump = jump_map.get(row["ticker"], 0.25)
            otm_gap = 0.05
            option_cost_frac = 0.012  # ~1.2% of spot for a 5% OTM weekly
            multiple = max(0.0, (jump - otm_gap) / option_cost_frac)
            multiple *= rng.uniform(0.6, 1.0)  # slippage / imperfect strike
            pnl.append(stake_per_alert * (multiple - 1))
        else:
            pnl.append(stake_per_alert * loss_on_miss * rng.uniform(0.8, 1.15))
    total = float(np.sum(pnl))
    spent = stake_per_alert * len(a)
    return {"total_pnl": total, "spent": spent, "roi": total / spent,
            "pnl_series": pnl}


# ----------------------------------------------------------------------------
# 4. Synthetic market generator (demo mode)
# ----------------------------------------------------------------------------

def generate_synthetic_market(n_tickers: int = 400, n_days: int = 252,
                              n_ma_events: int = 12,
                              informed_leak_prob: float = 0.55,
                              seed: int = 42):
    """
    Builds one year of synthetic options flow for a 400-name universe.
    - Background noise flow every day (speculation, hedging, spreads).
    - Earnings dates that attract PYPL-screenshot-lookalike call buying
      (the false positives that kill naive filters).
    - n_ma_events acquisitions; with prob `informed_leak_prob`, informed
      buyers front-run 1-4 days early with exactly the suspicious pattern.
      (Academic estimates of pre-M&A informed options activity are in the
      25-60% range depending on methodology -- this knob lets you test both.)
    Returns (flow_df, events_df, catalysts_df).
    """
    rng = np.random.default_rng(seed)
    tickers = [f"TK{i:03d}" for i in range(n_tickers)]
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    spot = {t: rng.uniform(15, 300) for t in tickers}

    rows = []

    def add_record(t, d, informed=False, earnings_spec=False):
        s = spot[t]
        if informed:
            money = rng.uniform(1.03, 1.08)
            dte = int(rng.integers(2, 10))
            vol = int(rng.integers(800, 6000))
            oi = int(vol / rng.uniform(3, 15))
            prem = float(rng.uniform(40_000, 300_000))
            askp = rng.uniform(0.85, 1.0)
            etype = "sweep"
            ot = "C"
        elif earnings_spec:
            money = rng.uniform(1.02, 1.09)
            dte = int(rng.integers(2, 12))
            vol = int(rng.integers(500, 5000))
            oi = int(vol / rng.uniform(2, 10))
            prem = float(rng.uniform(30_000, 250_000))
            askp = rng.uniform(0.75, 1.0)
            etype = "sweep"
            ot = "C"
        else:
            money = rng.uniform(0.85, 1.20)
            dte = int(rng.integers(1, 60))
            vol = int(rng.lognormal(5, 1.2))
            oi = int(vol * rng.uniform(0.3, 8))
            prem = float(rng.lognormal(9.5, 1.3))
            askp = rng.uniform(0.2, 1.0)
            etype = rng.choice(["sweep", "block", "normal"], p=[0.3, 0.2, 0.5])
            ot = rng.choice(["C", "P"], p=[0.55, 0.45])
        rows.append(dict(date=d, ticker=t, opt_type=ot,
                         strike=round(s * money, 2), spot=round(s, 2),
                         dte=dte, volume=vol, open_interest=oi,
                         premium=round(prem, 0), ask_side_pct=round(askp, 2),
                         exec_type=etype))

    # background noise: ~30 random tickers get flow each day, 1-6 records each
    for d in dates:
        for t in rng.choice(tickers, size=30, replace=False):
            for _ in range(int(rng.integers(1, 7))):
                add_record(t, d)

    # earnings catalysts: each ticker gets ~2 earnings dates; some attract
    # aggressive call speculation in the 1-5 days prior (lookalike noise)
    cat_rows = []
    for t in tickers:
        for _ in range(2):
            ed = dates[int(rng.integers(20, n_days - 1))]
            cat_rows.append(dict(ticker=t, date=ed))
            if rng.random() < 0.35:  # 35% of earnings attract heavy spec flow
                for back in range(1, int(rng.integers(2, 6))):
                    d = ed - pd.Timedelta(days=back)
                    if d in dates:
                        for _ in range(int(rng.integers(2, 6))):
                            add_record(t, d, earnings_spec=True)
    catalysts = pd.DataFrame(cat_rows)

    # Rumor / lookalike flow: tickers that get exactly the "informed" pattern
    # with NO catalyst and NO subsequent deal. This is the tape's dirty secret:
    # takeover chatter, momentum chasing, and unseeable spread legs generate
    # suspicious-looking sweeps constantly. This is what caps real precision.
    n_rumor_bursts = int(n_tickers * 0.35)
    for _ in range(n_rumor_bursts):
        t = tickers[int(rng.integers(0, n_tickers))]
        d0 = dates[int(rng.integers(5, n_days - 5))]
        for back in range(int(rng.integers(1, 4))):
            d = d0 + pd.Timedelta(days=back)
            if d in dates:
                for _ in range(int(rng.integers(3, 8))):
                    add_record(t, d, informed=True)

    # M&A events
    ev_rows = []
    ma_tickers = rng.choice(tickers, size=n_ma_events, replace=False)
    for t in ma_tickers:
        ad = dates[int(rng.integers(30, n_days - 1))]
        jump = float(rng.uniform(0.15, 0.60))
        ev_rows.append(dict(ticker=t, announce_date=ad, jump_pct=round(jump, 3)))
        if rng.random() < informed_leak_prob:
            lead = int(rng.integers(1, 5))
            for back in range(1, lead + 1):
                d = ad - pd.Timedelta(days=back)
                if d in dates:
                    for _ in range(int(rng.integers(3, 9))):
                        add_record(t, d, informed=True)
    events = pd.DataFrame(ev_rows)

    flow = pd.DataFrame(rows)
    flow["date"] = pd.to_datetime(flow["date"])
    return flow, events, catalysts


# ----------------------------------------------------------------------------
# 5. Threshold sweep
# ----------------------------------------------------------------------------

def sweep_thresholds(flow, events, catalysts) -> pd.DataFrame:
    """Run a grid of configs and report the precision/recall tradeoff."""
    results = []
    grid = [
        SignalConfig(min_daily_premium=50_000,  min_vol_oi_ratio=1.0,
                     min_ask_side_pct=0.5, exclude_catalyst_window=0),
        SignalConfig(min_daily_premium=150_000, min_vol_oi_ratio=2.0,
                     min_ask_side_pct=0.75, exclude_catalyst_window=0),
        SignalConfig(min_daily_premium=150_000, min_vol_oi_ratio=2.0,
                     min_ask_side_pct=0.75, exclude_catalyst_window=7),
        SignalConfig(min_daily_premium=250_000, min_vol_oi_ratio=3.0,
                     min_ask_side_pct=0.85, exclude_catalyst_window=7),
        SignalConfig(min_daily_premium=250_000, min_vol_oi_ratio=3.0,
                     min_ask_side_pct=0.85, exclude_catalyst_window=7,
                     accumulation_days=2),
    ]
    names = ["loose (naive screener)",
             "moderate",
             "moderate + catalyst filter",
             "strict + catalyst filter",
             "strict + catalyst + 2-day accumulation"]
    for name, cfg in zip(names, grid):
        alerts = detect_signals(flow, cfg, catalysts)
        ev = evaluate(alerts, events, lookahead_days=5)
        pnl = simulate_pnl(ev, events)
        results.append(dict(
            config=name,
            alerts_per_year=ev["n_alerts"],
            true_hits=ev["hits"],
            precision=round(ev["precision"], 3) if ev["n_alerts"] else 0,
            events_caught=f'{ev["events_caught"]}/{ev["n_events"]}',
            recall=round(ev["recall"], 2),
            roi_on_stakes=f'{pnl["roi"]:+.0%}' if ev["n_alerts"] else "n/a",
        ))
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Generating synthetic market (400 tickers, 1 year, 12 M&A events)...")
    flow, events, catalysts = generate_synthetic_market()
    print(f"  flow records: {len(flow):,} | M&A events: {len(events)} | "
          f"catalyst dates: {len(catalysts)}\n")

    table = sweep_thresholds(flow, events, catalysts)
    print(table.to_string(index=False))
    print("\nInterpretation: precision = of all alerts, how many preceded a real "
          "announcement.\nRecall = of all announcements, how many were flagged in "
          "advance.\nROI uses $1k lottery-ticket stakes, -85% avg on misses.")
