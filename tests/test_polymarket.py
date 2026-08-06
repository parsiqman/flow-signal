"""
Tests for the Polymarket copy-trading evaluator.

The evaluator's entire job is separating skill from luck, and that claim cannot
be checked against real data -- nobody knows which real wallets are skilled.
So it is checked against populations where the answer was recorded in advance,
and BOTH error directions are measured:

  - it must find real edges when they are large enough to be findable
  - it must find nothing in a population where nobody has an edge

A detector that only ever says "no" passes the second test and is useless. Both
are asserted here.

Two bugs were caught by these tests during development, and both are the same
species as the synthetic-options-market bug: a fixture that did not contain the
effect it advertised, and a statistic that looked right and was not.

    python tests/test_polymarket.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polymarket import client, execution, fixtures, wallets   # noqa: E402


# --- the fixture must contain what it claims --------------------------------

def test_fixture_actually_contains_the_edge_it_advertises():
    """
    The bug this guards against: the first version of the generator used
    `skill_edge` only to decide WHEN a skilled trader acted, while the
    mispricing available to act on was a penny of noise. "Skilled" wallets had
    no edge, so the evaluator looked broken when it was working correctly.
    """
    for target in (0.05, 0.15):
        trades, truth = fixtures.generate_wallets(
            n_wallets=150, skilled_frac=0.3, skill_edge=target,
            trades_per_wallet=(150, 300), seed=11)
        scored = wallets.score_wallets(trades, 20).merge(truth, on="wallet")
        skilled = scored[scored["is_skilled"]]["edge_per_share"].mean()
        plain = scored[~scored["is_skilled"]]["edge_per_share"].mean()
        assert abs(skilled - target) < 0.03, (target, skilled)
        assert abs(plain) < 0.02, plain


def test_zero_skill_population_has_no_edge_on_average():
    trades = fixtures.generate_no_skill_population(n_wallets=400, seed=5)
    scored = wallets.score_wallets(trades, 20)
    assert abs(scored["edge_per_share"].mean()) < 0.02


# --- edge measurement --------------------------------------------------------

def test_sells_are_counted_not_ignored():
    """
    A SELL of YES at p is a BUY of NO at (1-p). Counting only BUYs silently
    discards half a wallet's activity, and the omission is invisible downstream.
    """
    df = pd.DataFrame({
        "wallet": ["a", "a"], "market_id": [1, 2], "timestamp": [0.0, 1.0],
        "price": [0.30, 0.70], "size": [100.0, 100.0],
        "side": ["BUY", "SELL"], "outcome": [1.0, 0.0], "resolved_at": [2.0, 2.0]})
    t = wallets.normalise_trades(df)
    # Both trades won: bought at 30c and it resolved yes; sold at 70c and it
    # resolved no, which is buying NO at 30c.
    assert t["profit"].iloc[0] > 0 and t["profit"].iloc[1] > 0
    np.testing.assert_allclose(t["eff_price"].iloc[1], 0.30)
    np.testing.assert_allclose(t["eff_outcome"].iloc[1], 1.0)


def test_win_rate_is_not_edge():
    """
    A wallet buying only 90c favourites wins 90% of the time with zero edge.
    The metric must report ~0, not 90%.
    """
    rng = np.random.default_rng(0)
    n = 500
    outcomes = (rng.random(n) < 0.90).astype(float)
    df = pd.DataFrame({
        "wallet": ["fav"] * n, "market_id": np.arange(n),
        "timestamp": np.arange(n, dtype=float), "price": [0.90] * n,
        "size": [100.0] * n, "side": ["BUY"] * n, "outcome": outcomes,
        "resolved_at": [1e9] * n})
    scored = wallets.score_wallets(df, min_trades=20)
    assert (outcomes == 1).mean() > 0.85          # high win rate
    assert abs(scored["edge_per_share"].iloc[0]) < 0.04   # ...and no edge


def test_effective_sample_size_accounts_for_size_concentration():
    """
    The standard error of a size-WEIGHTED mean is not sd/sqrt(n). Trade sizes
    are heavily skewed, so effective n can be a small fraction of trade count.
    Using sd/sqrt(n) inflated every t-statistic and let noise past the gate.
    """
    n = 200
    rng = np.random.default_rng(1)
    sizes = np.ones(n)
    sizes[0] = 1000.0                    # one dominant trade
    df = pd.DataFrame({
        "wallet": ["w"] * n, "market_id": np.arange(n),
        "timestamp": np.arange(n, dtype=float),
        "price": rng.uniform(0.3, 0.7, n), "size": sizes,
        "side": ["BUY"] * n, "outcome": (rng.random(n) < 0.5).astype(float),
        "resolved_at": [1e9] * n})
    scored = wallets.score_wallets(df, min_trades=20)
    assert scored["n_eff"].iloc[0] < 20, "one huge trade must collapse effective n"
    assert scored["n_trades"].iloc[0] == n


# --- the luck baseline -------------------------------------------------------

def test_luck_threshold_matches_simulation():
    """The 95th percentile of the max, checked against actual maxima."""
    rng = np.random.default_rng(3)
    for n in (500, 5000):
        maxima = [rng.normal(0, 1, n).max() for _ in range(800)]
        empirical = float(np.quantile(maxima, 0.95))
        predicted = wallets.luck_threshold_t(n, quantile=0.95)
        assert abs(empirical - predicted) < 0.20, (n, empirical, predicted)


def test_expected_max_is_a_weaker_bar_than_the_quantile():
    """
    Gating on the expected maximum is an error: the observed max exceeds its
    own mean roughly half the time.
    """
    assert wallets.luck_threshold_t(5000, 0.95) > np.sqrt(2 * np.log(5000)) * 0.9
    for n in (100, 1000, 10000):
        se = np.sqrt(0.25 / 100)
        assert wallets.luck_threshold(n, 100) > wallets.expected_best_edge(n, 100) * 0.99


def test_scanning_more_wallets_raises_the_bar():
    """More searching finds more luck, not more skill."""
    bars = [wallets.identifiable_edge_cents(n, 200)
            for n in (100, 1_000, 10_000, 100_000)]
    assert all(a < b for a, b in zip(bars, bars[1:])), bars


def test_longer_histories_lower_the_bar():
    bars = [wallets.identifiable_edge_cents(10_000, n) for n in (50, 200, 1000)]
    assert all(a > b for a, b in zip(bars, bars[1:])), bars


# --- detection, both directions ---------------------------------------------

def test_finds_skill_when_the_edge_is_large_enough():
    trades, truth = fixtures.generate_wallets(
        n_wallets=700, skilled_frac=0.05, skill_edge=0.25,
        trades_per_wallet=(150, 400), seed=7)
    ranked = wallets.luck_adjusted_ranking(
        wallets.score_wallets(trades, 20), n_wallets_scanned=700)
    m = ranked.merge(truth, on="wallet")
    flagged = m[m["clears_luck"]]
    assert len(flagged) >= 3, "must find a 25c edge"
    assert flagged["is_skilled"].mean() > 0.8, "and must not flag noise as skill"


def test_rarely_flags_anyone_in_a_zero_skill_population():
    """
    Measured false-positive rate is ~20% of populations, not the nominal 5%:
    per-share P&L is bimodal, so wallet t-statistics have fatter tails than
    normal. Documented rather than papered over -- clearing this gate is
    necessary, not sufficient, exactly like a low PBO in `lab.validation`.
    """
    hits = 0
    for seed in range(40, 48):
        trades = fixtures.generate_no_skill_population(n_wallets=600, seed=seed)
        ranked = wallets.luck_adjusted_ranking(
            wallets.score_wallets(trades, 20), n_wallets_scanned=1200)
        hits += int(ranked["clears_luck"].any())
    assert hits <= 4, f"{hits}/8 zero-skill populations produced a false positive"


# --- persistence -------------------------------------------------------------

def test_persistence_detects_a_large_real_edge():
    """A test that can only ever say 'no' is not a test."""
    trades, _ = fixtures.generate_wallets(
        n_wallets=600, skilled_frac=0.20, skill_edge=0.15,
        trades_per_wallet=(60, 240), seed=2)
    r = wallets.persistence_test(trades, top_frac=0.10)
    assert r["gap"] > 0
    assert r["gap_t_stat"] > 2.0
    assert "NO EVIDENCE" not in r["verdict"]


def test_persistence_reports_nothing_on_a_null_population():
    trades = fixtures.generate_no_skill_population(
        n_wallets=600, seed=3, trades_per_wallet=(60, 240))
    r = wallets.persistence_test(trades, top_frac=0.10)
    assert abs(r["gap"]) < 0.02
    assert "NO EVIDENCE" in r["verdict"]


def test_persistence_refuses_when_too_few_wallets_overlap():
    trades, _ = fixtures.generate_wallets(n_wallets=15, seed=4)
    r = wallets.persistence_test(trades)
    assert r["n_selected"] < 20 or "too few" in r["verdict"]


# --- bias attribution --------------------------------------------------------

def test_bias_harvesting_is_separated_from_skill():
    """
    A wallet fading longshots earns money with no skill of its own. Copying it
    is a strictly worse wrapper around a rule you could run directly.
    """
    trades, _ = fixtures.generate_wallets(
        n_wallets=60, skilled_frac=0.0, longshot_bias=0.6,
        trades_per_wallet=(200, 400), seed=9)
    scored = wallets.score_wallets(trades, 20)
    best = scored.sort_values("edge_per_share", ascending=False).iloc[0]["wallet"]
    out = wallets.bias_attribution(trades, best)
    assert "by_price_band" in out
    assert 0.0 <= out["stake_in_extreme_bands"] <= 1.0


# --- execution ---------------------------------------------------------------

def test_cross_venue_costs_real_money():
    """
    US persons may trade Polymarket US but not Polymarket Global, where the
    wallet histories live. Signal and fill are on different venues, and that
    is a regulatory fact rather than an engineering choice.
    """
    from dataclasses import replace
    same = replace(execution.ExecutionModel(), cross_venue=False)
    cross = execution.ExecutionModel()
    assert cross.total_slippage_cents() > same.total_slippage_cents()
    a = execution.copy_economics(8.0, model=same)
    b = execution.copy_economics(8.0, model=cross)
    assert a["est_annual_return_pct"] > b["est_annual_return_pct"]


def test_slippage_can_exceed_the_entire_leader_edge():
    tiny = execution.copy_economics(1.0)
    assert tiny["net_edge_cents"] < 0
    assert "EXCEEDS" in tiny["verdict"]


def test_thin_books_destroy_the_economics():
    s = execution.sensitivity(leader_edge_cents=8.0)
    thin = s[s["scenario"].str.contains("thin")].iloc[0]
    thick = s[s["scenario"].str.contains("thick")].iloc[0]
    assert thin["net_edge_c"] < thick["net_edge_c"]
    assert thin["capacity_$"] < thick["capacity_$"]


def test_identification_bar_exceeds_the_profitability_bar():
    """
    THE finding. There is a wide band of traders who would make money for you
    and whom you cannot tell apart from lucky ones. That gap, not execution
    cost, is what makes copy trading hard.
    """
    needed_to_profit = execution.required_leader_edge(
        target_annual_pct=20.0)["leader_edge_needed_cents"]
    needed_to_identify = wallets.identifiable_edge_cents(10_000, 200)
    assert needed_to_identify > 2 * needed_to_profit, (needed_to_identify,
                                                       needed_to_profit)


# --- API client and discovery -----------------------------------------------

def test_leaderboard_seeding_is_refused():
    """
    Seeding from the public leaderboard selects on the outcome variable: you
    would rank traders by past profit inside a set already filtered for past
    profit. The result looks excellent and means nothing, and nothing in the
    pipeline appears to fail. Refused in code, not just in a comment.
    """
    try:
        client.discover_by_leaderboard()
        raise AssertionError("leaderboard seeding must be refused")
    except NotImplementedError as e:
        assert "outcome variable" in str(e)


def test_resolved_markets_parse_to_a_winning_index():
    mk = client._markets_to_frame(client.sample_response_shapes()["markets"])
    assert len(mk) == 1
    assert mk["winning_index"].iloc[0] == 0
    assert np.isfinite(mk["resolved_at"].iloc[0])


def test_unresolved_markets_carry_no_winner():
    """A market still trading at 0.6/0.4 has not resolved and must be excluded."""
    rows = [{"conditionId": "0xopen", "question": "q", "closed": False,
             "endDate": "2030-01-01T00:00:00Z",
             "outcomePrices": '["0.6", "0.4"]', "volumeNum": 1000.0}]
    mk = client._markets_to_frame(rows)
    assert mk["winning_index"].isna().all()


def test_trade_normalisation_scores_both_sides_correctly():
    shapes = client.sample_response_shapes()
    raw = pd.DataFrame(shapes["trades"])
    mk = client._markets_to_frame(shapes["markets"])
    norm = client.normalise_trades(raw, mk)
    scored = wallets.normalise_trades(norm)
    # Wallet 1 bought the winning side at 42c: profit. Wallet 2 sold it: loss.
    assert scored["profit"].iloc[0] > 0
    assert scored["profit"].iloc[1] < 0


def test_api_shape_drift_fails_loudly():
    """
    A silently missing field becomes a NaN column and then a plausible result
    computed from nothing -- the failure mode that has already produced three
    confident wrong answers in this project.
    """
    shapes = client.sample_response_shapes()
    raw = pd.DataFrame(shapes["trades"]).drop(columns=["price"])
    mk = client._markets_to_frame(shapes["markets"])
    try:
        client.normalise_trades(raw, mk)
        raise AssertionError("must refuse a response missing a required field")
    except ValueError as e:
        msg = str(e)
        assert "could not find" in msg and "price" in msg
        # The message must name the columns that WERE present, or an unattended
        # failure gives no route to a fix.
        assert "Response columns" in msg


def test_field_names_are_resolved_not_hardcoded():
    """
    A single hardcoded field name that turns out wrong crashes an unattended
    run. Alternative spellings must resolve; genuinely absent fields must not.
    """
    shapes = client.sample_response_shapes()
    raw = pd.DataFrame(shapes["trades"]).rename(
        columns={"proxyWallet": "user", "conditionId": "market", "size": "shares"})
    resolved = client.validate_trade_fields(raw)
    assert resolved["wallet"] == "user"
    assert resolved["market_id"] == "market"
    assert resolved["size"] == "shares"


def test_response_description_reports_what_matched():
    """Unattended runs are read from logs, so the mapping must be printed."""
    raw = pd.DataFrame(client.sample_response_shapes()["trades"])
    desc = client.describe_response(raw, "trades")
    assert "proxyWallet" in desc and "wallet" in desc


def test_normalisation_drops_trades_with_no_resolution():
    shapes = client.sample_response_shapes()
    raw = pd.DataFrame(shapes["trades"])
    raw["conditionId"] = "0xunknown"          # not in the markets frame
    mk = client._markets_to_frame(shapes["markets"])
    assert len(client.normalise_trades(raw, mk)) == 0


def test_persistence_split_defaults_to_resolution_time():
    """
    Splitting on trade time is lookahead: a trade placed in period A on a market
    resolving in period B has an outcome nobody knew when ranking at the end of
    A. The default must be resolution time, and a bad column must be refused.
    """
    trades, _ = fixtures.generate_wallets(n_wallets=500, skilled_frac=0.2,
                                          skill_edge=0.15, seed=2,
                                          trades_per_wallet=(60, 240))
    r = wallets.persistence_test(trades)          # default = resolved_at
    assert r["gap_t_stat"] > 2.0
    try:
        wallets.persistence_test(trades, split_on="not_a_column")
        raise AssertionError("must refuse an unknown split column")
    except ValueError as e:
        assert "lookahead" in str(e)


# --- market clustering (found by the first live run) -------------------------

def test_many_fills_in_one_market_is_one_bet_not_many():
    """
    A market resolves ONCE, so every fill a wallet makes in it shares a single
    outcome. Counting 42 fills as 42 observations inflates the t-statistic by
    sqrt(42).

    This is not hypothetical. The first live scan reported a t-statistic of 124
    and a 54c/share edge, and 7 of the 11 wallets that "cleared" the luck bar
    had two or fewer distinct markets between them. Synthetic fixtures never
    caught it because they spread each wallet's trades evenly over hundreds of
    markets; real traders pile into a handful.
    """
    n = 42
    df = pd.DataFrame({
        "wallet": ["w"] * n, "market_id": [1] * n,
        "timestamp": np.arange(n, dtype=float), "price": [0.4] * n,
        "size": [100.0] * n, "side": ["BUY"] * n, "outcome": [1.0] * n,
        "resolved_at": [9e8] * n})
    assert len(wallets.score_wallets(df, min_trades=20, min_markets=10)) == 0


def test_the_same_wallet_across_many_markets_is_evaluable():
    """The companion check: a filter that rejects everything is not a filter."""
    m = 20
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "wallet": ["w"] * m, "market_id": range(m),
        "timestamp": np.arange(m, dtype=float), "price": [0.4] * m,
        "size": [100.0] * m, "side": ["BUY"] * m,
        "outcome": rng.integers(0, 2, m).astype(float), "resolved_at": [9e8] * m})
    scored = wallets.score_wallets(df, min_trades=10, min_markets=10)
    assert len(scored) == 1
    assert scored["n_markets"].iloc[0] == m


def test_variance_floor_stops_absurd_t_statistics():
    """
    A wallet whose market-level edges happen to be near-identical produces a
    vanishing standard error and an astronomical t from no real information.
    The live run's worst case was t=124.5 on an edge of 0.001.
    """
    m = 15
    df = pd.DataFrame({
        "wallet": ["w"] * m, "market_id": range(m),
        "timestamp": np.arange(m, dtype=float), "price": [0.5] * m,
        "size": [100.0] * m, "side": ["BUY"] * m,
        "outcome": [1.0] * m,                    # identical every time
        "resolved_at": [9e8] * m})
    scored = wallets.score_wallets(df, min_trades=10, min_markets=10)
    assert len(scored) == 1
    assert abs(float(scored["t_stat"].iloc[0])) < 30, scored["t_stat"].iloc[0]


def test_full_history_selection_is_by_activity_not_profit():
    """
    Narrowing candidates before fetching full histories must use an
    outcome-INDEPENDENT criterion, or it becomes leaderboard seeding wearing a
    different hat.
    """
    import inspect
    src = inspect.getsource(client.fetch_full_histories)
    assert "activity" in src
    for banned in ("profit", "edge_per_share", "roi", "pnl"):
        assert f"-{banned}" not in src and f'"{banned}"' not in src, banned


# --- specialists (the failure the live runs exposed) -------------------------

def test_category_labelling_buckets_by_subject():
    mk = pd.DataFrame({"market_id": list("abcd"), "question": [
        "Highest temperature in NYC on Jan 5?", "Will Trump win the primary?",
        "Bitcoin above 100k in March?", "Will the Fed cut rates in June?"]})
    cats = client.label_categories(mk)["category"].tolist()
    assert cats == ["weather", "politics", "crypto", "econ"], cats


def test_category_scoring_lowers_the_bar_for_a_specialist():
    """
    A specialist trades one domain. Scoring inside that domain means the
    multiple-testing penalty reflects the few hundred wallets active there,
    not the tens of thousands active anywhere -- so a real edge needs to be
    smaller to be provable.
    """
    rng = np.random.default_rng(0)
    trades, _ = fixtures.generate_wallets(n_wallets=400, skilled_frac=0.0,
                                          seed=4, trades_per_wallet=(30, 90),
                                          n_markets=400)
    trades["category"] = rng.choice(["sports", "politics", "crypto"], len(trades))
    rows = []
    for m in range(40):
        p = float(rng.uniform(0.35, 0.65))
        o = 1.0 if rng.random() < p + 0.25 else 0.0
        for _ in range(3):
            rows.append({"wallet": "0xSPEC", "market_id": 10_000 + m,
                         "timestamp": float(m), "price": p, "size": 100.0,
                         "side": "BUY", "outcome": o, "resolved_at": float(m + 1),
                         "category": "weather"})
    allt = pd.concat([trades, pd.DataFrame(rows)], ignore_index=True)

    by_cat = wallets.score_by_category(allt, min_trades=15, min_markets=8)
    spec = by_cat[by_cat["wallet"] == "0xSPEC"]
    assert len(spec) == 1, "specialist must be scored inside its category"
    assert bool(spec["clears_luck"].iloc[0])
    # focus is measured from the wallet's OWN trades, so a claimed specialist
    # can be checked rather than believed.
    assert spec["focus"].iloc[0] > 0.9


def test_score_by_category_requires_a_category_column():
    trades, _ = fixtures.generate_wallets(n_wallets=40, seed=1)
    try:
        wallets.score_by_category(trades)
        raise AssertionError("must refuse trades with no category column")
    except ValueError as e:
        assert "category" in str(e)


def test_concentrated_edge_surfaces_what_the_market_filter_excludes():
    """
    Informed trading is a large edge over few markets -- exactly what the
    10-market minimum removes. These must be surfaced for inspection, and the
    function must not pretend to score them.
    """
    rows = []
    for m in range(5):
        rows.append({"wallet": "0xINSIDER", "market_id": m, "timestamp": float(m),
                     "price": 0.30, "size": 500.0, "side": "BUY",
                     "outcome": 1.0, "resolved_at": float(m + 1)})
    for m in range(30):
        rows.append({"wallet": "0xNORMAL", "market_id": 100 + m,
                     "timestamp": float(m), "price": 0.5, "size": 100.0,
                     "side": "BUY", "outcome": float(m % 2),
                     "resolved_at": float(m + 1)})
    df = pd.DataFrame(rows)
    assert len(wallets.score_wallets(df, min_trades=3, min_markets=10)) == 1
    hits = wallets.concentrated_edge_candidates(df, min_markets=3, min_edge=0.15)
    assert "0xINSIDER" in set(hits["wallet"])
    assert "0xNORMAL" not in set(hits["wallet"])


# --- the test a single named wallet can actually support ---------------------

def _one_wallet(n_markets, edge_early, edge_late, seed=3):
    """One wallet, edge specified separately for each half of its history."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_markets):
        edge = edge_early if m < n_markets // 2 else edge_late
        price = 0.5
        won = float(rng.random() < price + edge)
        rows.append({"wallet": "0xW", "market_id": m, "timestamp": float(m),
                     "price": price, "size": 100.0, "side": "BUY",
                     "outcome": won, "resolved_at": float(m)})
    return pd.DataFrame(rows)


def test_within_wallet_split_catches_an_edge_that_only_existed_early():
    df = _one_wallet(400, edge_early=0.20, edge_late=0.0)
    r = wallets.wallet_out_of_sample(df, "0xW")
    assert r["early"]["edge_per_share"] > 0.1, r
    assert r["late"]["t_stat"] < 1.64, r
    assert "streak" in r["verdict"], r


def test_within_wallet_split_confirms_an_edge_present_throughout():
    df = _one_wallet(400, edge_early=0.20, edge_late=0.20)
    r = wallets.wallet_out_of_sample(df, "0xW")
    assert r["early"]["t_stat"] > 1.64 and r["late"]["t_stat"] > 1.64, r
    assert "BOTH" in r["verdict"], r


def test_within_wallet_split_refuses_when_there_is_too_little_history():
    df = _one_wallet(12, edge_early=0.2, edge_late=0.2)
    r = wallets.wallet_out_of_sample(df, "0xW", min_markets_each=10)
    assert "needs" in r["verdict"], r
    assert "early" not in r


def test_a_test_that_could_not_run_is_not_a_failed_test():
    """
    persistence_test returns 'too few wallets' for a single named account. The
    verdict logic must not read that as evidence against the wallet -- doing so
    filed a t-statistic of 10.19 under 'consistent with having been lucky'.
    """
    import inspect
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "scripts" / "polymarket_scan.py"
    text = src.read_text()
    assert "could NOT be run" in text
    assert "ran = gap_t is not None" in text
    del inspect


# --- lookup by id (the bug that faked a null result) -------------------------

class _IgnoresTheFilter:
    """
    A server that answers 200 with its default page whatever you ask for.

    This is not a hypothetical. Gamma does exactly this with an unrecognised
    query parameter, and it turned a named-wallet run into "no wallet had
    enough resolved trades to score" -- the most dangerous kind of wrong
    answer, because it is shaped like a finding.
    """

    def __init__(self):
        self.calls = []

    def _get(self, url, params):
        self.calls.append((url, params))
        return [{"conditionId": f"0xUNRELATED{i}", "question": "Some election",
                 "outcomePrices": '["1", "0"]', "endDate": "2024-01-01T00:00:00Z"}
                for i in range(20)]


class _HonoursRepeatedParam:
    """Returns only what was asked for, and only via the repeated-param form."""

    def __init__(self, known):
        self.known = known
        self.forms_tried = []

    def _get(self, url, params):
        cid = params.get("condition_ids")
        self.forms_tried.append(type(cid).__name__)
        if not isinstance(cid, list):
            return [{"conditionId": "0xNOISE", "question": "Some election",
                     "outcomePrices": '["1", "0"]'}]
        return [{"conditionId": c, "question": self.known[c],
                 "outcomePrices": '["1", "0"]',
                 "endDate": "2024-01-01T00:00:00Z"}
                for c in cid if c in self.known]


def test_lookup_rejects_markets_it_did_not_ask_for():
    """A 200 with the wrong markets must not count as a successful lookup."""
    fake = _IgnoresTheFilter()
    out = client.markets_by_condition_ids(fake, ["0xWEATHER1", "0xWEATHER2"])
    assert out.empty, out
    assert out.attrs["lookup_form"] == "none verified"
    # ...and it must have fallen through to the per-id path rather than
    # accepting the junk.
    assert any("/markets/0xWEATHER1" in u for u, _ in fake.calls)


def test_lookup_finds_the_form_that_actually_filters():
    known = {"0xW1": "Highest temperature in NYC on Jan 5?",
             "0xW2": "Highest temperature in Chicago on Jan 6?"}
    fake = _HonoursRepeatedParam(known)
    out = client.markets_by_condition_ids(fake, list(known), batch=2)
    assert set(out["market_id"]) == set(known)
    assert out.attrs["lookup_form"] == "gamma condition_ids[]"
    assert out["winning_index"].notna().all()


def test_repeated_params_are_sent_as_repeated_not_as_a_list_repr():
    """
    doseq=True is what makes the array form work at all. Without it urlencode
    ships `condition_ids=%5B%27a%27...`, which Gamma ignores -- and an ignored
    filter is the whole bug this guards.
    """
    import inspect
    src = inspect.getsource(client.PolymarketClient._get)
    assert "doseq=True" in src


def test_a_partial_batch_answer_is_topped_up_per_id():
    """
    Gamma answered for 13 of 1,831 requested ids -- all from the last two days
    -- and returned nothing for the rest. Silently scoring the 13 gave a
    confident verdict off an effective sample of 4.7. A partial answer has to
    be completed, not accepted.
    """
    recent = {f"0xR{i}": "Highest temperature in NYC?" for i in range(3)}
    old = [f"0xOLD{i}" for i in range(5)]

    class _AnswersOnlyForRecent:
        def _get(self, url, params):
            if "/markets/" in url:                       # clob, id in the path
                cid = url.rsplit("/", 1)[-1]
                return {"condition_id": cid, "question": "Older weather market",
                        "end_date_iso": "2024-01-01T00:00:00Z",
                        "tokens": [{"outcome": "Yes", "winner": True},
                                   {"outcome": "No", "winner": False}]}
            asked = params.get("condition_ids") or []
            return [{"conditionId": c, "question": recent[c],
                     "outcomePrices": '["1", "0"]',
                     "endDate": "2024-01-01T00:00:00Z"}
                    for c in asked if c in recent]

    out = client.markets_by_condition_ids(
        _AnswersOnlyForRecent(), list(recent) + old, batch=8)
    assert set(out["market_id"]) == set(recent) | set(old)
    assert out.attrs["n_from_gamma"] == 3
    assert out.attrs["n_from_clob"] == 5
    assert out["winning_index"].notna().all()


def test_clob_fallback_maps_the_winner_flag_to_the_gamma_outcome_order():
    """
    CLOB reports the winner per token; trades index against Yes-first order.
    Getting this backwards would score every fill as its own opposite, which
    reads as a large negative edge rather than as a bug.
    """
    class _Clob:
        def _get(self, url, params):
            return {"condition_id": "0xW1", "question": "NYC above 90F?",
                    "market_slug": "nyc-above-90f",
                    "end_date_iso": "2024-07-01T00:00:00Z",
                    # deliberately No-first, as CLOB is free to return
                    "tokens": [{"outcome": "No", "winner": False},
                               {"outcome": "Yes", "winner": True}]}

    out = client.markets_by_condition_ids(_Clob(), ["0xW1"])
    assert len(out) == 1
    assert int(out["winning_index"].iat[0]) == 0      # Yes won, Yes is index 0


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
