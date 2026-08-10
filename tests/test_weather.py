"""
Tests for the weather-vs-forecast strategy.

Weighted, as always, toward the generator and toward the ways this specific
measurement fabricates an edge. Three exist here that do not exist elsewhere in
the repo: a forecast that saw the future, an ensemble that is overconfident
rather than skilful, and a market question parsed into the wrong band.

    python tests/test_weather.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polymarket import weather as W  # noqa: E402


# --- parsing: a wrong band scores the wrong outcome silently ---------------

def test_bands_are_parsed_or_refused_never_guessed():
    ok = W.parse_temperature_market("m1", "Highest temperature in NYC on Jan 5: 40-44 degrees?",
                                    end_date="2026-01-05T00:00:00Z")
    assert ok and ok.city == "nyc" and (ok.lo_f, ok.hi_f) == (40.0, 45.0)

    above = W.parse_temperature_market("m2", "Will the temperature in Chicago be above 90 degrees?",
                                       end_date="2026-07-01T00:00:00Z")
    assert above and (above.lo_f, above.hi_f) == (90.0, np.inf)

    below = W.parse_temperature_market("m3", "Temperature in Denver below 20 degrees?",
                                       end_date="2026-01-01T00:00:00Z")
    assert below and (below.lo_f, below.hi_f) == (-np.inf, 20.0)

    for junk in ("Will Trump win the primary?",
                 "Highest temperature somewhere nice?",
                 "Temperature in NYC?"):
        assert W.parse_temperature_market("x", junk, end_date="2026-01-01T00:00:00Z") is None


def test_a_market_with_no_date_is_refused():
    assert W.parse_temperature_market("m", "Temperature in NYC 40-44 degrees?") is None


# --- the lookahead guard ---------------------------------------------------

def test_a_forecast_issued_after_the_trade_is_refused():
    """
    The single most dangerous failure available here. A forecast that saw the
    outcome produces a huge, confident, fictional edge.
    """
    df = pd.DataFrame({
        "forecast_issued_at": pd.to_datetime(["2026-01-05", "2026-01-09"], utc=True),
        "traded_at": pd.to_datetime(["2026-01-06", "2026-01-06"], utc=True)})
    try:
        W.assert_no_lookahead(df)
    except ValueError as e:
        assert "lookahead" in str(e)
    else:
        raise AssertionError("lookahead was not caught")


def test_clean_data_passes_the_guard():
    df = pd.DataFrame({
        "forecast_issued_at": pd.to_datetime(["2026-01-05"], utc=True),
        "traded_at": pd.to_datetime(["2026-01-06"], utc=True)})
    W.assert_no_lookahead(df)


# --- the ensemble must be honest before it can be skilful ------------------

def test_an_underdispersed_ensemble_is_widened_toward_truth():
    """
    Raw ensembles cluster tighter than reality. Fitting the inflation on a
    proper scoring rule must recover a widening factor, not a narrowing one,
    when members are too confident.
    """
    rng = np.random.default_rng(0)
    truth, members = [], []
    for _ in range(300):
        mu = rng.normal(50, 10)
        obs = mu + rng.normal(0, 6)         # reality is +/-6
        members.append(mu + rng.normal(0, 2, 30))   # ensemble thinks +/-2
        truth.append(obs)
    k = W.fit_spread_inflation(members, np.array(truth))
    assert k > 1.5, k


def test_a_well_dispersed_ensemble_is_left_alone():
    rng = np.random.default_rng(1)
    truth, members = [], []
    for _ in range(300):
        mu = rng.normal(50, 10)
        truth.append(mu + rng.normal(0, 5))
        members.append(mu + rng.normal(0, 5, 30))
    k = W.fit_spread_inflation(members, np.array(truth))
    assert 0.75 < k < 1.35, k


def test_band_probability_reflects_the_widening():
    mem = np.full(50, 50.0) + np.linspace(-1, 1, 50)
    tight = W.ensemble_band_probability(mem, 45, 55, spread_inflation=1.0)
    wide = W.ensemble_band_probability(mem, 45, 55, spread_inflation=20.0)
    assert tight == 1.0 and wide < 1.0


# --- detection, both directions -------------------------------------------

def _signals(n, forecast_skill, seed=2):
    """
    forecast_skill 1.0 -> forecast knows the true probability.
    forecast_skill 0.0 -> forecast is the market price, i.e. no information.
    """
    rng = np.random.default_rng(seed)
    p_true = rng.uniform(0.05, 0.95, n)
    p_mkt = np.clip(p_true + rng.normal(0, 0.12, n), 0.02, 0.98)
    p_fc = np.clip(forecast_skill * p_true + (1 - forecast_skill) * p_mkt
                   + rng.normal(0, 0.03, n), 0.01, 0.99)
    return pd.DataFrame({
        "market_id": np.arange(n), "p_forecast": p_fc, "p_market": p_mkt,
        "outcome": (rng.random(n) < p_true).astype(float)})


def test_a_genuinely_better_forecast_shows_an_edge():
    r = W.edge_vs_market(_signals(4000, forecast_skill=1.0), half_spread_cents=1.0)
    assert r["net_edge_cents"] > 2.0, r
    assert r["t_stat"] > 3.0, r


def test_a_forecast_with_no_information_shows_none():
    """The one that matters. A forecast that only echoes the price must not pay."""
    r = W.edge_vs_market(_signals(4000, forecast_skill=0.0), half_spread_cents=1.0)
    assert r["t_stat"] < 2.0, r


def test_trading_only_on_disagreement_is_what_makes_it_a_strategy():
    sig = _signals(4000, forecast_skill=1.0)
    loose = W.edge_vs_market(sig, min_disagreement=0.0)
    tight = W.edge_vs_market(sig, min_disagreement=0.20)
    assert tight["n"] < loose["n"]
    assert tight["gross_edge_cents"] > loose["gross_edge_cents"]


def test_calibration_is_reported_before_any_edge_is_believed():
    rep = W.calibration_report(_signals(3000, forecast_skill=1.0))
    assert len(rep) >= 5
    assert rep["gap"].abs().max() < 0.20, rep


# --- the live shape: band in the OUTCOME, not the question -----------------

def test_the_band_is_read_from_outcomes_on_live_markets():
    """
    The live questions are "Highest temperature in Hong Kong on August 10?"
    with legs like "84-85F". A parser reading only the question can never
    match one, which is why three collection runs reported "0 parsed" and it
    said nothing at all about wording.
    """
    m = {"conditionId": "0xabc", "endDate": "2026-08-10T00:00:00Z",
         "question": "Highest temperature in Hong Kong on August 10?",
         "outcomes": '["83F or below", "84-85F", "86-87F", "88F or above"]'}
    got = W.parse_multi_outcome_market(m)
    assert [(t.lo_f, t.hi_f) for t in got] == [
        (-np.inf, 83.0), (84.0, 86.0), (86.0, 88.0), (88.0, np.inf)]
    assert all(t.city == "hong kong" for t in got)
    assert len({t.market_id for t in got}) == 4      # one per leg, distinct


def test_the_bands_partition_without_gaps_or_overlap():
    """A gap would silently drop trades; an overlap would double-count them."""
    m = {"conditionId": "0x1", "endDate": "2026-08-10T00:00:00Z",
         "question": "Highest temperature in Seoul on August 10?",
         "outcomes": '["70-74", "75-79", "80-84"]'}
    got = W.parse_multi_outcome_market(m)
    for a, b in zip(got, got[1:]):
        assert a.hi_f == b.lo_f, (a.hi_f, b.lo_f)


def test_non_band_legs_are_skipped_not_guessed():
    m = {"conditionId": "0x2", "endDate": "2026-08-10T00:00:00Z",
         "question": "Will the temperature in Denver be a record?",
         "outcomes": '["Yes", "No"]'}
    assert W.parse_multi_outcome_market(m) == []


def test_global_cities_from_the_live_search_are_known():
    """Seoul, Hong Kong and Incheon appear live and were all missing before."""
    for c in ("seoul", "hong kong", "incheon", "singapore", "sydney"):
        assert c in W.CITIES, c


def test_a_question_carrying_its_own_band_still_works():
    """The closed-market crawl uses that shape; both must parse."""
    m = {"conditionId": "0x3", "endDate": "2026-01-05T00:00:00Z",
         "question": "Highest temperature in NYC on Jan 5: 40-44 degrees?",
         "outcomes": '["Yes", "No"]'}
    got = W.parse_multi_outcome_market(m)
    assert len(got) == 1 and (got[0].lo_f, got[0].hi_f) == (40.0, 45.0)


def test_search_uses_public_search_because_the_filters_do_not_filter():
    import inspect
    src = inspect.getsource(W.search_temperature_markets)
    assert "public-search" in src
    assert "events" in src        # results nest, they do not arrive as a list


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
