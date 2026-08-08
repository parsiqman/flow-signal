"""
Tests for the research platform.

The validation module is the part that has to be right. Everything else in this
repo produces numbers; `validation.py` decides whether numbers mean anything, so
a subtle error there would not produce a wrong backtest -- it would produce a
wrong *belief about every backtest*, silently and permanently.

So the statistical functions are checked against Monte Carlo ground truth rather
than against themselves: the expected-maximum-Sharpe formula is compared with
the actual maximum Sharpe of simulated zero-edge strategies, and PBO is checked
against both identical-noise configurations and a genuinely dominant one, over
many datasets rather than one, because PBO turns out to be highly variable on a
single draw.

    python tests/test_lab.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab import scout, validation                          # noqa: E402
from lab.pipeline import Pipeline, Stage, can_promote      # noqa: E402
from lab.protocol import (Bar, Order, Strategy,            # noqa: E402
                          returns_from_curve, validate_data)
from lab.registry import (Hypothesis, HypothesisRegistry,  # noqa: E402
                          Trial, TrialLedger)

import pandas as pd                                        # noqa: E402

TRADING_DAYS = 252


# --- the luck baseline, against Monte Carlo ---------------------------------

def test_expected_max_sharpe_matches_simulation():
    """
    The headline claim -- 'N zero-edge strategies produce a best Sharpe of about
    X' -- checked by actually simulating N zero-edge strategies.
    """
    rng = np.random.default_rng(0)
    t, n = 1000, 50
    maxima = []
    for _ in range(400):
        draws = rng.normal(0, 1, size=(t, n))
        sr = draws.mean(axis=0) / draws.std(axis=0, ddof=1)
        maxima.append(sr.max())
    empirical = float(np.mean(maxima))
    predicted = validation.expected_max_sharpe(n, sharpe_dispersion=1 / np.sqrt(t))
    assert abs(empirical - predicted) / empirical < 0.12, (empirical, predicted)


def test_expected_max_sharpe_grows_with_trials():
    d = 1 / np.sqrt(1000)
    vals = [validation.expected_max_sharpe(n, d) for n in (2, 10, 100, 1000)]
    assert all(a < b for a, b in zip(vals, vals[1:])), vals
    assert validation.expected_max_sharpe(1, d) == 0.0


def test_more_trials_means_more_data_required():
    a = validation.minimum_backtest_length(10, 1.0)
    b = validation.minimum_backtest_length(1000, 1.0)
    assert b > a > 0
    # Higher target Sharpe needs less data, quadratically.
    assert (validation.minimum_backtest_length(100, 2.0)
            < validation.minimum_backtest_length(100, 1.0) / 3)


# --- deflated Sharpe --------------------------------------------------------

def test_deflation_punishes_a_large_search():
    """Identical returns, different trial counts: more searching, less belief."""
    rng = np.random.default_rng(2)
    r = rng.normal(0.0007, 0.01, 1500)           # ~1.1 annual Sharpe
    few = validation.deflated_sharpe_ratio(r, n_trials=2)
    many = validation.deflated_sharpe_ratio(r, n_trials=5000)
    assert few["sharpe_annual"] == many["sharpe_annual"]
    assert few["dsr"] > many["dsr"]
    assert many["benchmark_sharpe"] > few["benchmark_sharpe"]


def test_deflation_punishes_negative_skew():
    """
    A short-vol return profile -- many small gains, rare large losses -- must be
    trusted less than a symmetric one at the same Sharpe. This is the whole
    reason a raw Sharpe flatters option selling.
    """
    rng = np.random.default_rng(3)
    n = 1500
    symmetric = rng.normal(0.0007, 0.01, n)
    skewed = np.where(rng.random(n) < 0.97, 0.0022, -0.06)
    sym = validation.deflated_sharpe_ratio(symmetric, n_trials=50)
    skw = validation.deflated_sharpe_ratio(skewed, n_trials=50)
    assert skw["skew"] < -1.0
    # Comparable Sharpe, but the skewed one must not be trusted more.
    if abs(skw["sharpe_annual"] - sym["sharpe_annual"]) < 0.6:
        assert skw["dsr"] <= sym["dsr"] + 1e-9


def test_zero_edge_strategy_fails_deflation():
    rng = np.random.default_rng(4)
    r = rng.normal(0.0, 0.01, 1200)
    out = validation.deflated_sharpe_ratio(r, n_trials=100)
    assert out["dsr"] < 0.95
    assert "NOT" in out["verdict"]


def test_strong_genuine_edge_survives_a_modest_search():
    rng = np.random.default_rng(5)
    r = rng.normal(0.0016, 0.008, 2500)          # ~3.2 annual Sharpe
    out = validation.deflated_sharpe_ratio(r, n_trials=30)
    assert out["dsr"] > 0.95, out


def test_deflation_handles_degenerate_input():
    out = validation.deflated_sharpe_ratio(np.array([0.01, 0.02]), n_trials=5)
    assert out["verdict"] == "insufficient data"
    flat = validation.deflated_sharpe_ratio(np.zeros(100), n_trials=5)
    assert not np.isfinite(flat["dsr"]) or flat["verdict"] == "insufficient data"


# --- probability of backtest overfitting ------------------------------------

def test_pbo_is_high_when_configs_are_identical_noise():
    """
    With no real differences between configurations, selection cannot
    generalise and PBO must say so.

    Measured behaviour is ~0.85, not the ~0.5 a naive independence argument
    predicts: train and test are exact complements of one fixed sample, so a
    winner selected partly on split luck gives that luck back out-of-sample.
    Verified below by the train/test Sharpe correlation being clearly negative.
    """
    rng = np.random.default_rng(6)
    m = rng.normal(0, 0.01, size=(1200, 20))
    out = validation.probability_of_overfit(m, n_splits=8)
    assert out["pbo"] > 0.6, out
    assert "NOISE" in out["verdict"]


def test_pbo_calibration_across_many_datasets():
    """
    PBO is noisy on any single dataset, so it is calibrated in aggregate.

    Measured over 12 independent datasets: a genuine edge gives PBO 0.0 every
    time, while identical-noise configurations scatter widely (~0.1 to ~0.86).
    The separation is what the gate relies on -- and the width of the noise
    range is why a single low PBO is necessary but not sufficient evidence.
    """
    noise_pbo, real_pbo = [], []
    for sd in range(30, 42):
        rng = np.random.default_rng(sd)
        m = rng.normal(0, 0.01, size=(1200, 20))
        noise_pbo.append(validation.probability_of_overfit(m, n_splits=8)["pbo"])
        r = m.copy()
        r[:, 7] += 0.0025
        real_pbo.append(validation.probability_of_overfit(r, n_splits=8)["pbo"])
    noise_pbo, real_pbo = np.array(noise_pbo), np.array(real_pbo)
    assert real_pbo.max() < 0.25, real_pbo
    assert noise_pbo.mean() > 0.40, noise_pbo
    assert noise_pbo.mean() - real_pbo.mean() > 0.35
    # The documented caveat: noise sometimes sneaks under the gate.
    assert noise_pbo.min() < 0.25, "docstring claims noise can pass; it must"


def test_pbo_is_low_when_one_config_genuinely_dominates():
    rng = np.random.default_rng(7)
    m = rng.normal(0, 0.01, size=(1200, 20))
    m[:, 3] += 0.0025                            # a real, persistent edge
    out = validation.probability_of_overfit(m, n_splits=8)
    assert out["pbo"] < 0.20, out
    assert "generalises" in out["verdict"]


def test_pbo_refuses_insufficient_data():
    rng = np.random.default_rng(8)
    out = validation.probability_of_overfit(rng.normal(0, 1, (20, 5)), n_splits=10)
    assert not np.isfinite(out["pbo"])


# --- null tests --------------------------------------------------------------

def test_null_test_flags_a_strategy_that_works_without_the_effect():
    rng = np.random.default_rng(9)
    null_runs = [rng.normal(0.0012, 0.01, 800) for _ in range(20)]
    out = validation.null_test(null_runs, observed_sharpe_annual=1.5)
    assert out["p_value"] > 0.05
    assert "WITHOUT" in out["verdict"]


def test_null_test_passes_when_the_edge_is_specific():
    rng = np.random.default_rng(10)
    null_runs = [rng.normal(-0.0002, 0.01, 800) for _ in range(20)]
    out = validation.null_test(null_runs, observed_sharpe_annual=2.0)
    assert out["p_value"] <= 0.05


# --- the gauntlet ------------------------------------------------------------

def test_gauntlet_requires_every_gate():
    """No weighted score: a strong return must not outvote a failed overfit test."""
    rng = np.random.default_rng(11)
    good = rng.normal(0.0016, 0.008, 2000)
    m = rng.normal(0, 0.01, size=(1000, 12))
    m[:, 0] += 0.003
    g = validation.run_gauntlet(
        "x", good, n_trials=12, trial_returns_matrix=m,
        null_runs=[rng.normal(0.004, 0.01, 500)],     # works on null -> must fail
        stability_cagrs=np.array([0.1] * 10))
    assert not g.passed
    assert any(not gate.passed and gate.name == "null_data_test" for gate in g.gates)


def test_gauntlet_can_pass():
    rng = np.random.default_rng(12)
    good = rng.normal(0.0018, 0.008, 2500)
    m = rng.normal(0, 0.01, size=(1200, 10))
    m[:, 0] += 0.003
    g = validation.run_gauntlet(
        "x", good, n_trials=10, trial_returns_matrix=m,
        null_runs=[rng.normal(-0.001, 0.01, 600) for _ in range(10)],
        stability_cagrs=np.array([0.08] * 9 + [-0.01]))
    assert g.passed, g.to_frame()


# --- registry ----------------------------------------------------------------

def _hypothesis(hid="h1") -> Hypothesis:
    return Hypothesis(
        id=hid, title="Test idea", source="risk_premium",
        thesis="A sufficiently long thesis describing the mechanism at work here.",
        who_pays="Hedgers who must buy protection regardless of its price level.",
        why_they_persist="Their mandates require the hedge every single quarter.",
        prediction="Selling the hedge earns a positive risk-adjusted return.",
        success_criteria={"dsr": ">= 0.95"},
        kill_criteria="Measured premium below three percent on real market data.",
        data_required="option chains")


def test_registry_rejects_placeholder_reasoning():
    h = _hypothesis()
    h.who_pays = "people"
    try:
        h.validate()
        raise AssertionError("should have rejected a one-word counterparty")
    except ValueError as e:
        assert "placeholder" in str(e)


def test_registry_rejects_missing_success_criteria():
    h = _hypothesis()
    h.success_criteria = {}
    try:
        h.validate()
        raise AssertionError("should have rejected empty success criteria")
    except ValueError as e:
        assert "success criteria" in str(e)


def test_registry_persists_and_blocks_silent_overwrite():
    with tempfile.TemporaryDirectory() as d:
        reg = HypothesisRegistry(d)
        reg.register(_hypothesis())
        try:
            reg.register(_hypothesis())
            raise AssertionError("duplicate registration should fail")
        except ValueError as e:
            assert "already registered" in str(e)
        assert len(HypothesisRegistry(d).all()) == 1   # survived a reload


def test_amendments_are_recorded():
    with tempfile.TemporaryDirectory() as d:
        reg = HypothesisRegistry(d)
        reg.register(_hypothesis())
        reg.amend("h1", "loosened after seeing results",
                  success_criteria={"dsr": ">= 0.5"})
        h = HypothesisRegistry(d).get("h1")
        assert any("AMENDED" in n for n in h.notes)
        assert h.success_criteria == {"dsr": ">= 0.5"}


# --- trial ledger ------------------------------------------------------------

def test_ledger_counts_every_run_including_exploration():
    with tempfile.TemporaryDirectory() as d:
        led = TrialLedger(d)
        for i in range(7):
            led.record(Trial("h1", f"fp{i % 3}", {"a": i}, "ds", 1.0, 0.1,
                             stage="exploration" if i < 4 else "tuning"))
        assert led.count("h1") == 7           # runs, not ideas
        assert led.distinct_configs("h1") == 3
        assert TrialLedger(d).count("h1") == 7   # persisted


def test_ledger_separates_datasets():
    with tempfile.TemporaryDirectory() as d:
        led = TrialLedger(d)
        led.record(Trial("h1", "a", {}, "synthetic", 1.0, 0.1, "tuning"))
        led.record(Trial("h1", "b", {}, "real", 1.0, 0.1, "tuning"))
        assert led.count("h1", dataset="real") == 1
        assert led.count("h1") == 2


# --- pipeline ----------------------------------------------------------------

def test_stages_cannot_be_skipped():
    c = can_promote(Stage.TESTED, Stage.LIVE)
    assert not c.allowed and "skip" in c.reason


def test_validation_requires_a_clean_gauntlet():
    failing = validation.GauntletResult("x", [
        validation.Gate("g", "d", False, 0.1, 0.9)])
    c = can_promote(Stage.TESTED, Stage.VALIDATED, gauntlet=failing,
                    data_years=20, n_trials=5, observed_sharpe=1.0)
    assert not c.allowed


def test_validation_requires_enough_data_for_the_search():
    """A clean gauntlet is not enough if the search was bigger than the sample."""
    clean = validation.GauntletResult("x", [
        validation.Gate("g", "d", True, 1.0, 0.9)])
    c = can_promote(Stage.TESTED, Stage.VALIDATED, gauntlet=clean,
                    data_years=2.0, n_trials=5000, observed_sharpe=1.0)
    assert not c.allowed and "needs" in c.reason
    ok = can_promote(Stage.TESTED, Stage.VALIDATED, gauntlet=clean,
                     data_years=40.0, n_trials=5000, observed_sharpe=1.0)
    assert ok.allowed


def test_live_requires_a_full_paper_quarter():
    assert not can_promote(Stage.PAPER, Stage.LIVE, paper_days=20).allowed
    assert can_promote(Stage.PAPER, Stage.LIVE, paper_days=70).allowed


def test_pipeline_records_blocked_promotions():
    with tempfile.TemporaryDirectory() as d:
        reg, led = HypothesisRegistry(d), TrialLedger(d)
        reg.register(_hypothesis())
        pipe = Pipeline(reg, led)
        pipe.promote("h1", Stage.LIVE)
        assert any("BLOCKED" in n for n in reg.get("h1").notes)


def test_research_debt_rises_with_the_search():
    with tempfile.TemporaryDirectory() as d:
        reg, led = HypothesisRegistry(d), TrialLedger(d)
        reg.register(_hypothesis())
        pipe = Pipeline(reg, led)
        for i in range(5):
            led.record(Trial("h1", f"f{i}", {}, "ds", 1.5, 0.1, "tuning"))
        small = pipe.research_debt().iloc[0]["years_of_data_required"]
        for i in range(200):
            led.record(Trial("h1", f"g{i}", {}, "ds", 1.5, 0.1, "tuning"))
        big = pipe.research_debt().iloc[0]["years_of_data_required"]
        assert big > small


# --- protocol ----------------------------------------------------------------

class _Dummy(Strategy):
    name = "dummy"
    required_columns = ("spot", "atm_iv")

    def generate(self, bar: Bar, history: pd.DataFrame) -> list[Order]:
        return [Order(name="A", direction=-1, conviction=0.5, horizon_days=30)]

    def parameters(self):
        return {"k": 1}


def test_missing_columns_are_refused_not_silently_nan():
    s = _Dummy()
    ok = pd.DataFrame({"spot": [1.0, 2.0], "atm_iv": [0.2, 0.3]})
    validate_data(s, ok)
    try:
        validate_data(s, pd.DataFrame({"spot": [1.0]}))
        raise AssertionError("should refuse data missing a required column")
    except ValueError as e:
        assert "atm_iv" in str(e)


def test_fingerprint_is_stable_and_parameter_sensitive():
    a, b = _Dummy(), _Dummy()
    assert a.fingerprint() == b.fingerprint()
    b.parameters = lambda: {"k": 2}
    assert a.fingerprint() != b.fingerprint()


def test_returns_from_curve_handles_short_and_dirty_input():
    assert len(returns_from_curve(pd.Series([1.0]))) == 0
    r = returns_from_curve(pd.Series([100.0, 110.0, 121.0]))
    np.testing.assert_allclose(r, [0.1, 0.1], rtol=1e-9)


# --- scouting ----------------------------------------------------------------

def test_scout_ranks_durable_accessible_sources_first():
    q = scout.queue()
    assert q.iloc[0]["source"] in ("risk_premium", "constraint")
    info = q[q["source"] == "information"]
    assert info.iloc[0]["score"] < q.iloc[0]["score"]


def test_every_catalogued_idea_names_a_counterparty():
    for idea in scout.CATALOGUE:
        assert len(idea.who_pays) > 25, idea.title
        assert len(idea.why_they_persist) > 25, idea.title
        assert idea.source in scout.SOURCES


def test_idea_promotes_to_a_valid_hypothesis():
    idea = scout.CATALOGUE[0]
    h = scout.to_hypothesis(
        idea, "x1",
        prediction="A prediction long enough to be a real commitment to test.",
        success_criteria={"dsr": ">= 0.95"},
        kill_criteria="Abandoned if measured premium is below three percent.")
    h.validate()


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
