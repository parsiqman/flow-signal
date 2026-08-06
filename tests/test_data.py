"""
Tests for real option-chain ingestion.

The quality gate is the part that has to be right, and it is tested the only way
a gate can honestly be tested: by generating data with each defect deliberately
present and checking it is caught. A gate validated only against clean data is
an assertion, not a test.

Two properties get equal weight:
  - it CATCHES the defects (dirty fixture blocks)
  - it does NOT fire on good data (clean fixture passes)

The second matters as much as the first. A gate that blocks everything is
useless in a different way, and would be quietly disabled within a week.

    python tests/test_data.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import chain, loaders, quality, schema      # noqa: E402


def _clean(**kw):
    return loaders.synthetic_chain(n_underlyings=6, n_days=120, seed=0, **kw)


# --- schema ------------------------------------------------------------------

def test_essential_columns_cannot_be_defaulted():
    df = _clean().drop(columns=["bid"])
    try:
        schema.conform(df)
        raise AssertionError("must refuse a chain with no bid column")
    except ValueError as e:
        assert "bid" in str(e) and "essential" in str(e)


def test_optional_columns_default_only_when_permitted():
    df = _clean().drop(columns=["open_interest"])
    try:
        schema.conform(df, strict=True)
        raise AssertionError("strict mode must refuse to invent open_interest")
    except ValueError as e:
        assert "open_interest" in str(e)
    out = schema.conform(df, strict=False)
    assert (out["open_interest"] == 0).all()


def test_mid_is_nan_where_the_contract_cannot_be_sold():
    """
    The single most damaging silent error: a zero bid has no buyer at any
    price, but (0 + ask)/2 looks like a perfectly tradeable quote.
    """
    df = pd.DataFrame({"bid": [1.0, 0.0, 2.0, 3.0], "ask": [1.2, 0.5, 0.0, 2.0]})
    m = schema.mid(df)
    assert np.isfinite(m.iloc[0])
    assert np.isnan(m.iloc[1])      # zero bid
    assert np.isnan(m.iloc[2])      # zero ask
    assert np.isnan(m.iloc[3])      # crossed


def test_right_is_normalised_across_vendor_spellings():
    raw = pd.DataFrame({
        "date": ["2021-01-04"] * 2, "act_symbol": ["AAA", "AAA"],
        "expiration": ["2021-02-19"] * 2, "strike": [100.0, 100.0],
        "call_put": ["Call", "put"], "bid": [1.0, 1.0], "ask": [1.1, 1.1],
        "underlying_price": [100.0, 100.0],
    })
    out = loaders.from_frame(raw, loaders.DOLTHUB_OPTIONS, strict=False)
    assert set(out["right"]) == {"C", "P"}


# --- the gate catches what it claims -----------------------------------------

def test_clean_data_passes_the_gate():
    """A gate that fires on good data gets disabled within a week."""
    rep = quality.check_chain(_clean())
    assert not rep.blocked, rep.summary()


def test_dirty_data_is_blocked():
    rep = quality.check_chain(_clean(dirty=True))
    assert rep.blocked
    checks = {f.check for f in rep.findings if f.severity in ("block", "warn")}
    assert "crossed_market" in checks
    assert "below_intrinsic" in checks or "put_call_parity" in checks


def test_survivorship_filtering_is_detected():
    """
    The bias that most flatters a short-vol strategy, and the hardest to see:
    the names that blew up are simply absent, leaving no gap to notice.
    """
    honest = loaders.synthetic_chain(n_underlyings=30, n_days=150, seed=1)
    filtered = loaders.synthetic_chain(n_underlyings=30, n_days=150, seed=1,
                                       survivorship_filtered=True)
    assert not quality.check_chain(honest).blocked
    rep = quality.check_chain(filtered)
    assert rep.blocked
    assert any(f.check == "survivorship" and f.severity == "block"
               for f in rep.findings)


def test_mostly_unusable_quotes_are_blocked():
    df = _clean()
    df.loc[df.index[: int(len(df) * 0.8)], "bid"] = 0.0
    rep = quality.check_chain(df)
    assert rep.blocked
    assert any(f.check == "mostly_unusable" for f in rep.findings)


def test_short_history_blocks_when_a_minimum_is_demanded():
    rep = quality.check_chain(_clean(), min_years=5.0)
    assert rep.blocked
    assert any(f.check == "insufficient_history" for f in rep.findings)


def test_assert_usable_raises_rather_than_warns():
    quality.assert_usable(_clean())          # must not raise
    try:
        quality.assert_usable(_clean(dirty=True))
        raise AssertionError("must raise on dirty data, not merely warn")
    except ValueError as e:
        assert "quality gate" in str(e)


# --- cleaning ----------------------------------------------------------------

def test_clean_drops_unsellable_and_adjusted_contracts():
    dirty = _clean(dirty=True)
    out = quality.clean(dirty, require_bid=True)
    assert (out["bid"] > 0).all()
    assert (out["ask"] > 0).all()
    assert (out["bid"] <= out["ask"]).all()
    assert not out["is_adjusted"].any()
    assert len(out) < len(dirty)


def test_clean_can_keep_zero_bids_when_buying():
    dirty = _clean(dirty=True)
    selling = quality.clean(dirty, require_bid=True)
    buying = quality.clean(dirty, require_bid=False)
    assert len(buying) > len(selling)


def test_wide_markets_are_excluded():
    df = _clean()
    tight = quality.clean(df, max_relative_spread=0.10)
    loose = quality.clean(df, max_relative_spread=None)
    assert len(tight) < len(loose)


# --- chain mathematics -------------------------------------------------------

def test_atm_iv_tracks_the_money():
    df = quality.clean(_clean())
    one = df[(df["underlying"] == df["underlying"].iloc[0])]
    d0 = one[one["date"] == one["date"].iloc[0]]
    spot = float(d0["underlying_price"].iloc[0])
    exp = chain.nearest_expiry(d0, 30)
    iv = chain.atm_iv_for_expiry(d0[d0["expiry"] == exp], spot)
    assert 0.05 < iv < 2.0


def test_constant_maturity_interpolates_the_term_structure():
    """With a real term slope, constant-maturity IV must differ from nearest."""
    df = quality.clean(loaders.synthetic_chain(n_underlyings=3, n_days=60,
                                               seed=5, term_slope=0.5))
    one = df[df["underlying"] == df["underlying"].iloc[0]]
    d0 = one[one["date"] == one["date"].iloc[0]]
    spot = float(d0["underlying_price"].iloc[0])
    exp = chain.nearest_expiry(d0, 30)
    nearest = chain.atm_iv_for_expiry(d0[d0["expiry"] == exp], spot)
    cm = chain.constant_maturity_iv(d0, spot, target_days=30)
    assert np.isfinite(cm) and abs(cm - nearest) > 1e-4


def test_variance_interpolation_lands_between_the_bracketing_expiries():
    df = quality.clean(loaders.synthetic_chain(n_underlyings=2, n_days=40,
                                               seed=6, term_slope=0.6))
    one = df[df["underlying"] == df["underlying"].iloc[0]]
    d0 = one[one["date"] == one["date"].iloc[0]]
    spot = float(d0["underlying_price"].iloc[0])
    per = []
    for days, g in d0.assign(_d=schema.dte(d0)).groupby("_d"):
        per.append((days, chain.atm_iv_for_expiry(g, spot)))
    per.sort()
    cm = chain.constant_maturity_iv(d0, spot, target_days=30)
    lo, hi = min(p[1] for p in per), max(p[1] for p in per)
    assert lo - 1e-9 <= cm <= hi + 1e-9, (lo, cm, hi)


def test_strikes_snap_to_the_listed_ladder():
    """
    Real chains do not offer whatever strike the maths wants, and the gap has
    to be reported rather than hidden.
    """
    df = quality.clean(_clean())
    d0 = df[(df["date"] == df["date"].iloc[0])
            & (df["underlying"] == df["underlying"].iloc[0])]
    spot = float(d0["underlying_price"].iloc[0])
    exp = chain.nearest_expiry(d0, 30)
    picks = chain.select_strikes(d0, exp, "P", [spot * 0.913], spot)
    assert picks
    p = picks[0]
    listed = set(d0[(d0["expiry"] == exp) & (d0["right"] == "P")]["strike"])
    assert p["strike"] in listed
    assert p["strike"] != p["target_strike"]      # it snapped
    assert abs(p["gap_pct"]) < 0.10
    assert p["bid"] > 0 and p["ask"] >= p["bid"]


def test_panel_matches_the_synthetic_schema():
    """Real and synthetic panels must be interchangeable downstream."""
    df = quality.clean(_clean())
    panel = chain.build_panel(df, target_days=30)
    for c in schema.PANEL_COLUMNS:
        assert c in panel.columns, c
    assert panel["day"].min() == 0
    assert panel["atm_iv"].between(0.01, 5.0).all()
    assert (panel["spot"] > 0).all()


def test_missing_earnings_calendar_does_not_empty_the_universe():
    """
    999 rather than NaN, so an absent calendar weakens the filter instead of
    silently dropping every row through NaN comparison.
    """
    df = quality.clean(_clean())
    panel = chain.build_panel(df, earnings=None)
    assert (panel["days_to_earnings"] == 999).all()
    assert len(panel) > 0


def test_earnings_calendar_is_applied_when_present():
    df = quality.clean(_clean())
    name = df["underlying"].iloc[0]
    d0 = pd.Timestamp(df["date"].min())
    earnings = pd.DataFrame({"underlying": [name],
                             "date": [d0 + pd.Timedelta(days=10)]})
    panel = chain.build_panel(df, earnings=earnings)
    row = panel[(panel["name"] == name) & (panel["date"] == d0)]
    assert len(row) == 1
    assert int(row["days_to_earnings"].iloc[0]) == 10


# --- the kill criterion ------------------------------------------------------

def test_vrp_measurement_works_in_both_directions():
    """
    The pre-registered kill criterion has to detect a premium that IS there and
    reject one that is not. Testing only the positive case would let a broken
    measurement pass forever.
    """
    none = quality.clean(loaders.synthetic_chain(n_underlyings=6, n_days=180,
                                                 seed=3, vrp=0.0))
    rich = quality.clean(loaders.synthetic_chain(n_underlyings=6, n_days=180,
                                                 seed=3, vrp=0.25))
    s_none = chain.vrp_summary(chain.measure_vrp(none))
    s_rich = chain.vrp_summary(chain.measure_vrp(rich))
    assert s_none["capture"] < 0.03, s_none
    assert "BELOW" in s_none["verdict"]
    assert s_rich["capture"] > 0.05, s_rich
    assert s_rich["capture"] > s_none["capture"] + 0.08


def test_vrp_sells_at_the_bid_not_the_mid():
    """A seller receives the bid. Using mid inflates measured capture."""
    df = quality.clean(loaders.synthetic_chain(n_underlyings=4, n_days=120,
                                               seed=8, vrp=0.2))
    trades = chain.measure_vrp(df)
    assert len(trades) > 20
    key = ["date", "underlying", "expiry", "strike"]
    calls = df[df["right"] == "C"][key + ["bid", "ask"]]
    puts = df[df["right"] == "P"][key + ["bid", "ask"]]
    j = (trades.merge(calls, on=key, suffixes=("", "_c"))
               .merge(puts, on=key, suffixes=("_c", "_p")))
    assert len(j) > 0
    bid_premium = j["bid_c"] + j["bid_p"]
    mid_premium = (j["bid_c"] + j["ask_c"]) / 2 + (j["bid_p"] + j["ask_p"]) / 2
    np.testing.assert_allclose(j["premium"], bid_premium, rtol=1e-9)
    # Selling at the mid would overstate the premium collected on every trade.
    assert (bid_premium < mid_premium).all()


def test_vrp_summary_handles_an_empty_result():
    out = chain.vrp_summary(pd.DataFrame())
    assert out["n"] == 0 or out["n_straddles"] == 0 if "n_straddles" in out else True
    assert "no straddles" in out["verdict"]


# --- loaders -----------------------------------------------------------------

def test_csv_round_trip_through_a_vendor_map():
    with tempfile.TemporaryDirectory() as d:
        src = _clean()
        vendor = src.rename(columns={
            "underlying": "act_symbol", "expiry": "expiration",
            "right": "call_put", "iv": "vol"})
        p = Path(d) / "chain.csv"
        vendor.to_csv(p, index=False)
        out = loaders.load_csv(p, vendor="dolthub", strict=False)
        assert len(out) == len(src)
        assert set(out.columns) == set(schema.REQUIRED)


def test_orats_paired_layout_produces_both_rights():
    """
    Mapping only the call columns yields a dataset that looks complete and
    contains no puts at all -- a silent, total failure.
    """
    raw = pd.DataFrame({
        "trade_date": ["2021-01-04"] * 3, "ticker": ["AAA"] * 3,
        "expir_date": ["2021-02-19"] * 3, "strike": [90.0, 100.0, 110.0],
        "stkpx": [100.0] * 3,
        "cbid": [11.0, 3.0, 0.5], "cask": [11.4, 3.2, 0.6],
        "pbid": [0.4, 2.8, 10.2], "pask": [0.5, 3.0, 10.6],
    })
    out = loaders.load_orats_pair(raw)
    assert set(out["right"]) == {"C", "P"}
    assert len(out) == 6


def test_unknown_vendor_is_refused():
    try:
        loaders.load_csv("nowhere.csv", vendor="not-a-vendor")
        raise AssertionError("must refuse an unknown vendor")
    except ValueError as e:
        assert "unknown vendor" in str(e)


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for f in fns:
        try:
            f()
            print(f"  PASS  {f.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {f.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
