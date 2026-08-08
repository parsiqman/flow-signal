"""
Tests for the favourite-longshot rule.

Weighted deliberately toward the GENERATOR and toward the ways this
measurement produces a confident positive from nothing. The rule itself is
twenty lines of arithmetic; what needs proving is that a tape with no bias in
it yields no rule, that a tape with a known bias yields the right one, and that
market clustering and the spread are handled rather than assumed away.

    python tests/test_longshot.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polymarket import longshot  # noqa: E402


# --- generators ----------------------------------------------------------

def unbiased_tape(n_markets: int = 1500, seed: int = 0,
                  fills_per_market: int = 3) -> pd.DataFrame:
    """
    Prices are honest: a claim priced p pays off with probability exactly p.

    This is the tape that must produce NO rule. Any edge found here is
    manufactured by the statistics.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_markets):
        p = float(rng.uniform(0.02, 0.98))
        won = float(rng.random() < p)
        for _ in range(fills_per_market):
            rows.append({"wallet": f"0x{rng.integers(0, 50)}", "market_id": m,
                         "timestamp": float(m), "price": p,
                         "size": float(rng.integers(10, 500)), "side": "BUY",
                         "outcome": won, "resolved_at": float(m)})
    return pd.DataFrame(rows)


def biased_tape(n_markets: int = 1500, seed: int = 0, strength: float = 0.06,
                fills_per_market: int = 3) -> pd.DataFrame:
    """
    Favourite-longshot bias, injected with a known sign and size.

    Longshots are overpriced and favourites underpriced: the true probability
    is pulled toward 0.5 relative to the quoted price, so cheap claims pay off
    LESS often than they cost and dear claims pay off MORE often.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_markets):
        p = float(rng.uniform(0.02, 0.98))
        # sign(p - 0.5) is -1 for cheap claims, so this subtracts at the cheap
        # end and adds at the dear end: exactly the documented signature.
        true_p = float(np.clip(p + strength * np.sign(p - 0.5), 0.01, 0.99))
        won = float(rng.random() < true_p)
        for _ in range(fills_per_market):
            rows.append({"wallet": f"0x{rng.integers(0, 50)}", "market_id": m,
                         "timestamp": float(m), "price": p,
                         "size": float(rng.integers(10, 500)), "side": "BUY",
                         "outcome": won, "resolved_at": float(m)})
    return pd.DataFrame(rows)


# --- the generator itself, first ------------------------------------------

def test_the_biased_tape_actually_contains_the_bias_it_advertises():
    """
    The lesson this repo keeps relearning: a harness that does not contain the
    effect it claims to measure fails SILENTLY and looks like a clean negative.
    Check the fixture before trusting anything measured in it.
    """
    cal = longshot.calibrate(biased_tape())
    cheap = cal[cal["avg_price"] < 0.2]
    dear = cal[cal["avg_price"] > 0.8]
    assert (cheap["edge"] < 0).all(), cheap      # longshots overpriced
    assert (dear["edge"] > 0).all(), dear        # favourites underpriced


def test_the_unbiased_tape_really_is_unbiased():
    cal = longshot.calibrate(unbiased_tape())
    assert cal["edge"].abs().max() < 0.05, cal


# --- the thing that must not happen ---------------------------------------

def test_no_rule_is_fitted_to_an_honest_tape():
    """The whole point. Honest prices must yield nothing to trade."""
    rule = longshot.fit(unbiased_tape())
    assert rule.is_empty(), rule.describe()


def test_a_rule_fitted_on_noise_makes_no_money_out_of_sample():
    """
    Even when a band scrapes through the bar by luck, it must not pay in a
    period it was not fitted on. This is the difference between a backtest and
    a strategy.
    """
    wf = longshot.walk_forward(unbiased_tape(n_markets=2000, seed=7))
    oos = wf["out_of_sample"]
    if oos.get("n_markets", 0) == 0:
        return                                    # no rule fitted: also correct
    # A band DOES sometimes scrape through the in-sample bar -- that is the
    # residual false-positive rate working as designed, not a defect. What must
    # not happen is that it looks real out of sample.
    assert abs(oos["t_stat_net"]) < 2.0, oos


def test_the_known_bias_is_recovered_and_pays_out_of_sample():
    wf = longshot.walk_forward(biased_tape(n_markets=6000, seed=3),
                               half_spread_cents=0.0)
    assert not wf["rule"].startswith("no band"), wf["rule"]
    oos = wf["out_of_sample"]
    assert oos["net_edge_cents"] > 1.0, oos
    assert oos["t_stat_net"] > 2.0, oos


# --- clustering, cost, and the multiple-testing bar ------------------------

def test_many_fills_in_one_market_do_not_inflate_the_sample():
    """
    A market resolves once. 200 fills in it is ONE observation, and treating it
    as 200 is the bug that produced a t-statistic of 124 earlier in this repo.
    """
    few = longshot.calibrate(unbiased_tape(n_markets=300, fills_per_market=1))
    many = longshot.calibrate(unbiased_tape(n_markets=300, fills_per_market=40))
    # Same markets, 40x the fills: the effective sample must not grow with fills.
    assert many["n_eff"].sum() < 3 * few["n_eff"].sum(), (few, many)


def test_the_spread_is_charged_and_can_erase_the_edge():
    tape = biased_tape(n_markets=2400, seed=3)
    free = longshot.walk_forward(tape, half_spread_cents=0.0)["out_of_sample"]
    dear = longshot.walk_forward(tape, half_spread_cents=99.0)["out_of_sample"]
    assert free["net_edge_cents"] > dear["net_edge_cents"]
    assert dear["net_edge_cents"] < 0
    assert "spread eats" in dear["verdict"]


def test_breakeven_spread_is_reported_so_the_edge_can_be_checked_against_a_book():
    oos = longshot.walk_forward(biased_tape(n_markets=2400, seed=3),
                                half_spread_cents=1.0)["out_of_sample"]
    assert oos["breakeven_half_spread_cents"] > 0
    # net = gross - cost, so break-even must be exactly the gross edge.
    assert abs(oos["breakeven_half_spread_cents"]
               - (oos["net_edge_cents"] + oos["cost_cents"])) < 0.01


def test_the_bar_accounts_for_every_band_examined():
    """Fitting ten bands and keeping the good ones is ten tests, not one."""
    from polymarket import wallets
    rule = longshot.fit(biased_tape(n_markets=600))
    assert rule.n_bands_tested >= 8
    assert rule.min_t == wallets.luck_threshold_t(rule.n_bands_tested)
    assert rule.min_t > 1.64


def test_bands_are_fixed_in_code_before_any_result_is_seen():
    """
    Free choice of cut points can manufacture an edge in a null tape. The band
    edges are a constant, not an argument tuned per dataset.
    """
    assert longshot.DEFAULT_BANDS[0] == 0.0 and longshot.DEFAULT_BANDS[-1] == 1.0
    assert list(longshot.DEFAULT_BANDS) == sorted(longshot.DEFAULT_BANDS)


# --- the null test --------------------------------------------------------

def test_a_real_bias_beats_the_honest_price_null():
    tape = biased_tape(n_markets=1200, seed=5)
    rule = longshot.fit(tape)
    res = longshot.null_check(rule, tape, n_draws=60, half_spread_cents=0.0)
    assert res["p_value"] < 0.05, res
    assert "exceeds" in res["verdict"], res


def test_the_null_keeps_the_two_tokens_of_a_market_opposite():
    """
    A binary market has two tokens whose fates are opposite. The first null
    drew one Bernoulli per market and broadcast it to every fill, giving both
    tokens the same fate -- impossible -- and centring the null at -38 cents.
    Fills on both sides must stay complementary under the draw.
    """
    rows = []
    for m in range(200):
        won = float(m % 2)
        rows.append({"wallet": "a", "market_id": m, "timestamp": 0.0,
                     "price": 0.9, "size": 100.0, "side": "BUY",
                     "outcome": won, "resolved_at": float(m)})
        rows.append({"wallet": "b", "market_id": m, "timestamp": 1.0,
                     "price": 0.1, "size": 100.0, "side": "BUY",
                     "outcome": 1.0 - won, "resolved_at": float(m)})
    tape = pd.DataFrame(rows)
    rule = longshot.LongshotRule(bands=longshot.DEFAULT_BANDS,
                                 side={"0.80-0.90": 1}, fitted_edge={"0.80-0.90": 0.0})
    res = longshot.null_check(rule, tape, n_draws=40, half_spread_cents=0.0)
    # Honest prices, so a rule buying the 0.8-0.9 band should break even.
    assert abs(res["null_mean_cents"]) < 6.0, res


def test_the_null_is_centred_near_zero_so_the_rule_can_actually_fail_it():
    """
    The null must be "prices are honest", not "price and outcome are
    unrelated". The first version permuted outcomes across markets, which put
    the null mean at -30 cents: a rule buying 93c favourites then wins half the
    time, so ANY real edge clears it. A test the strategy cannot fail measures
    nothing, and this pins the null where it belongs.
    """
    tape = biased_tape(n_markets=1200, seed=5)
    rule = longshot.fit(tape)
    res = longshot.null_check(rule, tape, n_draws=60, half_spread_cents=0.0)
    assert abs(res["null_mean_cents"]) < 1.0, res


def test_the_null_redraws_whole_markets_not_individual_fills():
    """
    Drawing per fill would let one market both win and lose, which cannot
    happen, and would shrink the null variance until everything looks
    significant.
    """
    import inspect
    src = inspect.getsource(longshot.null_check)
    assert 'groupby(["market_id", "outcome"])' in src
    assert "1.0 - fake[\"outcome\"]" in src


# --- walk-forward hygiene --------------------------------------------------

def test_the_split_is_on_resolution_time_not_trade_time():
    import inspect
    src = inspect.getsource(longshot.walk_forward)
    assert 'split_on: str = "resolved_at"' in src


def test_train_and_test_share_no_markets():
    tape = biased_tape(n_markets=800, seed=11)
    wf = longshot.walk_forward(tape)
    assert wf["n_markets_train"] + wf["n_markets_test"] == 800


def test_too_little_history_refuses_rather_than_guessing():
    wf = longshot.walk_forward(biased_tape(n_markets=20))
    assert "too few" in wf["verdict"]


# --- power: is a null informative, or just small? -------------------------

def test_a_thin_sample_is_reported_as_underpowered_not_as_negative():
    """
    The live run returned "no band cleared the bar" from 333 markets, where the
    typical band could not resolve anything under ~20 cents against a 2-8 cent
    effect. That is not a negative result; it is an untested question, and
    reporting it as a finding is the failure this whole repo exists to stop.
    """
    cal = longshot.calibrate(unbiased_tape(n_markets=120, seed=1))
    pw = longshot.power_verdict(cal, bar=2.57)
    assert pw["underpowered"] is True, pw
    assert "UNDERPOWERED" in pw["verdict"]


def test_a_large_sample_is_reported_as_adequately_powered():
    cal = longshot.calibrate(unbiased_tape(n_markets=20000, seed=1))
    pw = longshot.power_verdict(cal, bar=2.57)
    assert pw["underpowered"] is False, pw
    assert pw["median_mde_cents"] <= longshot.DOCUMENTED_EFFECT_CENTS[1]


def test_minimum_detectable_edge_falls_as_the_sample_grows():
    small = longshot.power_verdict(
        longshot.calibrate(unbiased_tape(n_markets=400, seed=2)), 2.57)
    big = longshot.power_verdict(
        longshot.calibrate(unbiased_tape(n_markets=8000, seed=2)), 2.57)
    assert big["median_mde_cents"] < small["median_mde_cents"]


# --- correlation: the parameter the capital model turns on -----------------

def _clustered_tape(n_days=200, per_day=12, rho_like=0.9, seed=1):
    """
    Markets resolving on the same day share a common shock.

    `rho_like` is the probability that a day's markets all follow that day's
    common draw rather than an independent one -- a crude but transparent way
    to dial in known clustering.
    """
    rng = np.random.default_rng(seed)
    rows = []
    mid = 0
    for d in range(n_days):
        common = float(rng.random() < 0.85)
        for _ in range(per_day):
            won = common if rng.random() < rho_like else float(rng.random() < 0.85)
            rows.append({"wallet": "w", "market_id": mid, "timestamp": 0.0,
                         "price": 0.85, "size": 100.0, "side": "BUY",
                         "outcome": won, "resolved_at": float(d) * 86400.0})
            mid += 1
    return pd.DataFrame(rows)


def _rule_for(band="0.80-0.90"):
    return longshot.LongshotRule(bands=longshot.DEFAULT_BANDS, side={band: 1},
                                 fitted_edge={band: 0.0})


def test_correlation_is_near_zero_when_outcomes_are_independent():
    tape = _clustered_tape(rho_like=0.0, seed=4)
    r = longshot.loss_correlation(_rule_for(), tape)
    assert r["icc"] < 0.05, r


def test_correlation_is_detected_when_markets_share_a_resolution_day():
    tape = _clustered_tape(rho_like=1.0, seed=4)
    r = longshot.loss_correlation(_rule_for(), tape)
    assert r["icc"] > 0.5, r


def test_correlation_collapses_the_effective_number_of_bets():
    """
    The breadth argument -- 1,294 bets, therefore diversified -- is exactly
    what this number falsifies, and it fails in the flattering direction.
    """
    tape = _clustered_tape(rho_like=1.0, seed=4)
    r = longshot.loss_correlation(_rule_for(), tape)
    assert r["effective_independent_bets"] < r["n_markets"] / 5
    assert r["breadth_lost_pct"] > 80


def test_the_ruin_threshold_is_carried_alongside_the_measurement():
    """A measured number with no bar to compare it to decides nothing."""
    tape = _clustered_tape(rho_like=0.5, seed=4)
    r = longshot.loss_correlation(_rule_for(), tape)
    assert r["ruin_threshold_icc"] == 0.07
    assert ("ABOVE" in r["verdict"]) == (r["icc"] > 0.07)


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
