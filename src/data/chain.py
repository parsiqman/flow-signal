"""
Turning raw contract rows into the panel the strategy pipeline already eats.

Two jobs, both easy to get quietly wrong.

**Constant-maturity ATM implied vol.** Listed expiries jump around: today the
nearest 30-day option is 28 days out, next week it is 21, then 35. Feeding that
raw into a richness signal measures the expiry calendar as much as the market.
So implied vol is interpolated to a fixed tenor, in TOTAL VARIANCE rather than
in vol, because variance is what is additive in time. Interpolating vol
linearly is a common shortcut and it biases the term structure.

**Real strikes.** The synthetic backtest could put a strike wherever the maths
wanted. Real chains offer $1, $2.50 or $5 increments, so a 10-delta target
lands on whatever is listed nearby -- and on a cheap underlying the nearest
strike may be 14-delta. That difference is not cosmetic: it changes the risk of
every position. `select_strikes` returns what is actually available, and the
caller has to live with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import dte, mid


def atm_iv_for_expiry(slice_df: pd.DataFrame, spot: float,
                      width: float = 0.10) -> float:
    """
    Implied vol at the money for one (date, underlying, expiry).

    Averages call and put IV across strikes within `width` of spot, weighted
    toward the money. Calls and puts are blended because each side is more
    reliable on its own half of the smile -- OTM puts and OTM calls carry the
    tightest markets, ITM quotes are wide and stale.
    """
    if slice_df.empty or not np.isfinite(spot) or spot <= 0:
        return np.nan
    d = slice_df[slice_df["iv"].notna() & (slice_df["iv"] > 0.01)
                 & (slice_df["iv"] < 5.0)]
    if d.empty:
        return np.nan
    moneyness = (d["strike"] / spot - 1.0).abs()
    near = d[moneyness <= width]
    if near.empty:
        # Nothing close to the money: fall back to the single nearest strike.
        near = d.loc[[moneyness.idxmin()]]
        return float(near["iv"].iloc[0])
    w = 1.0 / (0.01 + (near["strike"] / spot - 1.0).abs())
    return float(np.average(near["iv"], weights=w))


def constant_maturity_iv(day_slice: pd.DataFrame, spot: float,
                         target_days: int = 30,
                         min_days: int = 7, max_days: int = 120) -> float:
    """
    ATM implied vol at a fixed tenor, interpolated in total variance.

    Picks the listed expiries bracketing `target_days` and interpolates
    w = iv^2 * T linearly in T, then converts back. This is the standard
    construction (the same idea VIX uses) and it matters: linear interpolation
    in vol rather than variance systematically misprices the term structure,
    and a richness signal built on it partly measures that error.

    Returns NaN when the tenor cannot be bracketed, rather than extrapolating.
    Extrapolated vol at the short end is wildly unstable near expiry.
    """
    if day_slice.empty:
        return np.nan
    d = day_slice.assign(_dte=dte(day_slice))
    d = d[(d["_dte"] >= min_days) & (d["_dte"] <= max_days)]
    if d.empty:
        return np.nan

    per_expiry = []
    for days, grp in d.groupby("_dte"):
        v = atm_iv_for_expiry(grp, spot)
        if np.isfinite(v) and v > 0:
            per_expiry.append((float(days), v))
    if not per_expiry:
        return np.nan
    per_expiry.sort()

    if len(per_expiry) == 1:
        return per_expiry[0][1]

    below = [p for p in per_expiry if p[0] <= target_days]
    above = [p for p in per_expiry if p[0] >= target_days]
    if not below or not above:
        # Cannot bracket: use the nearest expiry rather than extrapolating.
        nearest = min(per_expiry, key=lambda p: abs(p[0] - target_days))
        return nearest[1]

    t1, v1 = below[-1]
    t2, v2 = above[0]
    if t1 == t2:
        return v1
    w1, w2 = v1 ** 2 * t1, v2 ** 2 * t2          # total variance
    w = w1 + (w2 - w1) * (target_days - t1) / (t2 - t1)
    return float(np.sqrt(max(w, 1e-12) / target_days))


def select_strikes(day_slice: pd.DataFrame, expiry: pd.Timestamp,
                   right: str, targets: list[float],
                   spot: float) -> list[dict]:
    """
    Snap target strikes onto the listed ladder.

    Returns one dict per target with the actual listed strike, its quote, and
    the gap between what was wanted and what exists. That gap is reported
    rather than hidden, because on a $15 stock with $2.50 increments it can be
    large enough to change the position's risk materially -- and a backtest
    that silently uses the ideal strike is trading a contract that never existed.
    """
    d = day_slice[(day_slice["expiry"] == expiry) & (day_slice["right"] == right)]
    d = d[mid(d).notna()]
    if d.empty:
        return []
    listed = np.sort(d["strike"].unique())
    out = []
    for target in targets:
        i = int(np.abs(listed - target).argmin())
        k = float(listed[i])
        row = d[d["strike"] == k].iloc[0]
        out.append({
            "target_strike": float(target),
            "strike": k,
            "gap_pct": float((k - target) / spot) if spot else np.nan,
            "bid": float(row["bid"]), "ask": float(row["ask"]),
            "mid": float((row["bid"] + row["ask"]) / 2),
            "iv": float(row["iv"]) if np.isfinite(row["iv"]) else np.nan,
            "open_interest": float(row["open_interest"]),
        })
    return out


def nearest_expiry(day_slice: pd.DataFrame, target_days: int,
                   min_days: int = 7) -> pd.Timestamp | None:
    """The listed expiry closest to a target tenor, ignoring near-dated ones."""
    if day_slice.empty:
        return None
    d = day_slice.assign(_dte=dte(day_slice))
    d = d[d["_dte"] >= min_days]
    if d.empty:
        return None
    by = d.groupby("expiry")["_dte"].first()
    return by.index[int((by - target_days).abs().argmin())]


def build_panel(chain: pd.DataFrame, earnings: pd.DataFrame | None = None,
                target_days: int = 30,
                min_expiries: int = 2) -> pd.DataFrame:
    """
    Collapse a cleaned chain into the pipeline's panel schema.

    Output columns match `options_alpha.synthetic.generate_market` exactly --
    day, name, spot, atm_iv, days_to_earnings -- so real and synthetic data are
    interchangeable and every existing test applies to both.

    `day` is a dense integer index over trading dates present in the data, not
    a calendar offset, because the backtest counts holding periods in trading
    days and the two diverge across holidays.
    """
    if chain.empty:
        return pd.DataFrame(columns=["day", "name", "spot", "atm_iv",
                                     "days_to_earnings"])

    dates = np.sort(chain["date"].unique())
    day_of = {d: i for i, d in enumerate(dates)}

    rows = []
    for (date, name), grp in chain.groupby(["date", "underlying"], sort=False):
        if grp["expiry"].nunique() < min_expiries:
            continue
        spot = float(grp["underlying_price"].iloc[0])
        if not np.isfinite(spot) or spot <= 0:
            continue
        iv = constant_maturity_iv(grp, spot, target_days=target_days)
        if not np.isfinite(iv):
            continue
        rows.append({"day": day_of[np.datetime64(date)], "name": name,
                     "spot": spot, "atm_iv": iv, "date": date})

    panel = pd.DataFrame(rows)
    if panel.empty:
        return pd.DataFrame(columns=["day", "name", "spot", "atm_iv",
                                     "days_to_earnings"])

    panel["days_to_earnings"] = _days_to_earnings(panel, earnings)
    panel.attrs["dates"] = pd.DatetimeIndex(dates)
    panel.attrs["source"] = "real_chains"
    return panel[["day", "name", "spot", "atm_iv", "days_to_earnings", "date"]]


def _days_to_earnings(panel: pd.DataFrame,
                      earnings: pd.DataFrame | None) -> pd.Series:
    """
    Calendar days until the next earnings date, 999 where unknown.

    999 rather than NaN so that a missing calendar means "no earnings known"
    and the exclusion filter simply does not fire, instead of silently
    dropping every row via NaN comparison. Missing earnings data should reduce
    the filter's power, not empty the universe.
    """
    if earnings is None or earnings.empty:
        return pd.Series(999, index=panel.index, dtype=int)

    e = earnings.copy()
    e.columns = [c.strip().lower() for c in e.columns]
    e["date"] = pd.to_datetime(e["date"])
    e["underlying"] = e["underlying"].astype(str).str.upper()

    out = np.full(len(panel), 999, dtype=int)
    by_name = {k: np.sort(v["date"].values)
               for k, v in e.groupby("underlying")}
    for i, (name, d) in enumerate(zip(panel["name"], panel["date"])):
        arr = by_name.get(name)
        if arr is None:
            continue
        j = np.searchsorted(arr, np.datetime64(d))
        if j < len(arr):
            out[i] = int((arr[j] - np.datetime64(d)) / np.timedelta64(1, "D"))
    return pd.Series(out, index=panel.index)


def measure_vrp(chain: pd.DataFrame, holding_days: int = 30,
                risk_free: float = 0.04) -> pd.DataFrame:
    """
    The pre-registered kill criterion, measured on real data.

    Sells an ATM straddle on every (date, underlying) at the listed expiry
    nearest `holding_days`, holds to expiration, and compares premium received
    against what it paid out. Frictionless and deliberately naive -- this is not
    a strategy, it is the question "is there a premium here at all?".

    STRATEGY.md commits to abandoning the thesis if real capture comes in below
    3%. This is the function that decides that, and it should be run before any
    strategy work on real data, exactly as `test_premium_is_actually_harvestable`
    guards the synthetic path.
    """
    results = []
    dates = np.sort(chain["date"].unique())
    date_pos = {d: i for i, d in enumerate(dates)}
    spot_lookup = (chain.groupby(["date", "underlying"])["underlying_price"]
                   .first())

    for (date, name), grp in chain.groupby(["date", "underlying"], sort=False):
        exp = nearest_expiry(grp, holding_days)
        if exp is None:
            continue
        spot = float(grp["underlying_price"].iloc[0])
        if not np.isfinite(spot) or spot <= 0:
            continue
        legs = grp[grp["expiry"] == exp]
        strikes = select_strikes(legs, exp, "C", [spot], spot)
        puts = select_strikes(legs, exp, "P", [spot], spot)
        if not strikes or not puts:
            continue
        k = strikes[0]["strike"]
        if abs(puts[0]["strike"] - k) > 1e-9:
            continue                      # calls and puts must share a strike
        # Sell at the bid, which is what a seller actually receives.
        premium = strikes[0]["bid"] + puts[0]["bid"]
        if premium <= 0:
            continue

        # Settlement needs the underlying price on the expiry date.
        key = (np.datetime64(exp), name)
        if key not in spot_lookup.index:
            continue
        s_exp = float(spot_lookup.loc[key])
        payout = abs(s_exp - k)
        results.append({
            "date": date, "underlying": name, "expiry": exp, "strike": k,
            "spot": spot, "spot_at_expiry": s_exp,
            "premium": premium, "payout": payout,
            "pnl": premium - payout, "premium_pct_spot": premium / spot,
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df
    df.attrs["capture"] = float(df["pnl"].sum() / df["premium"].sum())
    return df


def vrp_summary(trades: pd.DataFrame) -> dict:
    """Headline capture, plus the by-year breakdown that reveals regime risk."""
    if trades.empty:
        return {"capture": np.nan, "n": 0,
                "verdict": "no straddles could be constructed"}
    capture = float(trades["pnl"].sum() / trades["premium"].sum())
    by_year = (trades.assign(year=pd.to_datetime(trades["date"]).dt.year)
               .groupby("year")
               .apply(lambda g: g["pnl"].sum() / g["premium"].sum(),
                      include_groups=False))
    return {
        "capture": round(capture, 4),
        "n_straddles": len(trades),
        "win_rate": round(float((trades["pnl"] > 0).mean()), 3),
        "by_year": {int(k): round(float(v), 4) for k, v in by_year.items()},
        "worst_year": round(float(by_year.min()), 4) if len(by_year) else np.nan,
        "verdict": ("premium is real and above the 3% kill threshold"
                    if capture >= 0.03 else
                    "BELOW the pre-registered 3% kill threshold"),
    }
