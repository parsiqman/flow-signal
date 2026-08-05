"""
Tests for the market-selection model.

The point is not that the numbers are right -- they rest on assumptions stated
in venues.py and cannot be tested. The point is that the machinery is not lying:
the analytic shortcuts agree with simulation, the monotonicities run the right
way, and the gates actually gate. A scorecard that silently mis-ranks is worse
than no scorecard, because it launders a guess as an analysis.

    python -m pytest tests/ -q          (or: python tests/test_market_selection.py)
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market_selection import economics, power, scorecard  # noqa: E402
from market_selection import venues as V                  # noqa: E402


# --- cost mechanics ------------------------------------------------------

def test_round_trip_counts_both_sides():
    v = V.Venue(name="t", style="cross_sectional", commission_bps=10,
                half_spread_bps=5, slippage_bps=0)
    assert v.round_trip_cost_bps() == 30.0


def test_fixed_cost_amortises_over_clip_size():
    """Gas is a fixed dollar cost, so it must hurt small clips more."""
    v = V.DEFI_ONCHAIN_L2
    small = v.round_trip_cost_bps(clip_usd=100)
    large = v.round_trip_cost_bps(clip_usd=100_000)
    assert small > large
    # ...but on an L2 the effect is tiny even at $100. This is the finding that
    # kills the "gas fees make DeFi unviable" argument: pool fees do, gas does not.
    assert (small - large) < 7.0


def test_cost_to_noise_falls_with_horizon_when_carry_is_zero():
    v = V.CRYPTO_SPOT_CEX
    assert v.carry_bps_per_day == 0
    ratios = [v.cost_to_noise(h) for h in (1, 5, 21, 63)]
    assert all(a > b for a, b in zip(ratios, ratios[1:]))


def test_carry_eventually_reverses_the_horizon_benefit():
    """With funding to pay, holding longer stops being free."""
    v = V.CRYPTO_PERPS_US
    assert v.carry_bps_per_day > 0
    assert v.cost_to_noise(252) > v.cost_to_noise(21)


def test_high_vol_offsets_high_fees():
    """
    The counterintuitive core of the cost model: crypto spot pays 18x the
    round-trip fee of equities but only ~6x the cost-to-noise ratio, because
    it is ~3x more volatile. Fee comparisons alone overstate crypto's penalty.
    """
    eq, cx = V.US_EQUITIES, V.CRYPTO_SPOT_CEX
    fee_ratio = cx.round_trip_cost_bps() / eq.round_trip_cost_bps()
    noise_ratio = cx.cost_to_noise(5) / eq.cost_to_noise(5)
    assert fee_ratio > 15
    assert noise_ratio < fee_ratio / 2


# --- breadth -------------------------------------------------------------

def test_breadth_never_exceeds_nominal():
    for v in V.ALL_VENUES:
        for h in (1, 5, 21):
            nominal = v.n_assets * (v.sessions_per_year / h)
            assert v.effective_breadth(h) <= nominal + 1e-6


def test_correlation_caps_breadth_at_one_over_rho():
    """A large universe of highly correlated names is not a large universe."""
    v = V.Venue(name="t", style="directional", n_assets=10_000,
                avg_pairwise_corr=0.8, sessions_per_year=252)
    n_eff = v.effective_breadth(252) * 252 / 252   # one rebalance per year
    assert abs(n_eff - 1 / 0.8) < 0.05


def test_hedging_the_common_factor_buys_breadth():
    """Same universe, cross-sectional vs directional: hedging must help."""
    import dataclasses
    base = V.CRYPTO_SPOT_CEX
    directional = dataclasses.replace(base, style="directional")
    assert base.effective_breadth(5) > directional.effective_breadth(5)


def test_event_cap_binds():
    v = V.US_EQUITY_OPTIONS
    assert v.max_bets_per_year is not None
    assert v.effective_breadth(0.25) == v.max_bets_per_year


# --- power ---------------------------------------------------------------

def test_analytic_ir_matches_simulation():
    """
    The whole ranking rests on annual_ir(). If the closed form disagrees with a
    brute-force simulation of the same bets, the ranking is fiction.
    """
    for v in (V.US_EQUITIES, V.CRYPTO_SPOT_CEX, V.DEFI_ONCHAIN_L2):
        h, _ = power.best_horizon(v, v.plausible_ic)
        mc = power.monte_carlo_check(v, v.plausible_ic, h, years=8.0,
                                     n_paths=6000)
        analytic, simulated = mc["analytic_IR"], mc["simulated_IR"]
        assert abs(analytic - simulated) < 0.1 * max(abs(analytic), 0.1), mc


def test_no_edge_means_never_validated():
    v = V.CRYPTO_SPOT_CEX
    tiny_ic = v.cost_to_noise(63) * 0.5   # below the cost floor at every horizon
    assert np.isinf(power.years_to_validate(v, tiny_ic, 63))


def test_more_skill_is_never_worse():
    for v in V.ALL_VENUES:
        irs = [power.best_horizon(v, ic)[1] for ic in (0.02, 0.05, 0.10, 0.15)]
        assert all(a <= b + 1e-9 for a, b in zip(irs, irs[1:])), v.name


def test_false_positive_rate_is_higher_where_bets_are_scarce():
    """
    Fewer independent bets means a lucky zero-edge backtest clears t=2 more
    often. This is the quantitative form of 'crypto backtests lie more'.
    """
    eq = power.monte_carlo_check(V.US_EQUITIES, 0.04, 3, years=2.0)
    cx = power.monte_carlo_check(V.CRYPTO_SPOT_CEX, 0.05, 63, years=2.0)
    assert cx["bets_in_window"] < eq["bets_in_window"]
    assert cx["p_false_positive"] > eq["p_false_positive"]


# --- economics helpers ---------------------------------------------------

def test_min_viable_horizon_respects_the_cost_floor():
    for v in V.ALL_VENUES:
        for ic in (0.03, 0.05, 0.10):
            h = economics.min_viable_horizon(v, ic)
            if h is not None:
                assert v.cost_to_noise(h) < ic


def test_cheap_venues_can_trade_faster():
    """A venue with lower cost/noise must permit a shorter viable horizon."""
    fast = economics.min_viable_horizon(V.US_EQUITIES, 0.05)
    slow = economics.min_viable_horizon(V.CRYPTO_SPOT_CEX, 0.05)
    assert fast is not None and slow is not None
    assert fast < slow


# --- scorecard -----------------------------------------------------------

def test_scores_stay_in_range():
    s = scorecard.score_venues()
    for c in scorecard.CRITERIA:
        assert s[c].between(0, 10).all(), c


def test_gated_venue_never_wins_any_weighting():
    """A legal blocker must be a veto, not a criterion that can be outweighed."""
    s = scorecard.score_venues()
    sens = scorecard.weight_sensitivity(s, n_draws=3000)
    blocked = [v.name for v in V.ALL_VENUES if v.is_gated()]
    assert blocked
    for name in blocked:
        assert sens.loc[sens["venue"] == name, "pct_rank_1"].iat[0] == 0.0


def test_gates_can_be_disabled_for_counterfactuals():
    """
    Turning gates off must actually change something, or the flag is a lie.

    Checked on the ranking rather than the win-rate, because Hyperliquid loses
    on merit too -- it never reaches the top two even ungated. That is itself
    worth knowing: the legal block is not what decides against it.
    """
    s = scorecard.score_venues()
    hl = "Offshore DeFi perps (Hyperliquid)"
    gated = scorecard.weighted_ranking(s, apply_gates=True)
    ungated = scorecard.weighted_ranking(s, apply_gates=False)
    assert np.isnan(gated.loc[gated["venue"] == hl, "score"].iat[0])
    assert np.isfinite(ungated.loc[ungated["venue"] == hl, "score"].iat[0])
    # ...and it still does not win.
    assert ungated.iloc[0]["venue"] == V.US_EQUITIES.name


def test_conclusion_is_not_an_artifact_of_the_default_weights():
    """The headline claim, asserted: equities wins on evidence, not weighting."""
    s = scorecard.score_venues()
    ranked = scorecard.weighted_ranking(s)
    sens = scorecard.weight_sensitivity(s)
    assert ranked.iloc[0]["venue"] == V.US_EQUITIES.name
    assert sens.iloc[0]["venue"] == V.US_EQUITIES.name
    assert sens.iloc[0]["pct_rank_1"] > 75.0


def test_every_venue_has_qualitative_reasoning_recorded():
    """No judgment score without a stated reason for it."""
    for v in V.ALL_VENUES:
        q = scorecard.QUALITATIVE[v.name]
        assert set(q["why"]) == {"infra_fit", "crowding_headroom", "repo_carryover"}
        assert all(len(text) > 20 for text in q["why"].values())


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
