"""
Tests for the options strategy.

Weighted deliberately toward the *generator* rather than the strategy. Three
bugs during development each produced a confident, plausible, completely wrong
backtest, and none of them would have been caught by testing the strategy:

  1. The log-vol process folded its own common factor back into the per-name
     state each step, compounding into 107% average volatility.
  2. Implied vol was priced off the latent vol process, which knows nothing
     about earnings jumps or crash gaps. The "+16% premium" was nominal, real
     premium was negative, and every strategy result was an artefact.
  3. Time to expiry was computed as dte/365 while volatility was annualised
     with sqrt(252) -- and dte counts trading days. Every option was priced
     for 83% of its true time exposure.

Bug 2 is the dangerous one: the backtest ran fine and produced a confident
losing answer. The guard is test_premium_is_actually_harvestable, which checks
the market rewards vol selling frictionlessly BEFORE any strategy runs on it.

    python tests/test_options_alpha.py
"""

from __future__ import annotations
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from options_alpha import families as F                      # noqa: E402
from options_alpha import research as R                      # noqa: E402
from options_alpha.backtest import run_backtest              # noqa: E402
from options_alpha.strategy import (StrategyConfig, build_condor,  # noqa: E402
                                    condor_payoff, forecast_realised_vol,
                                    regime_multiplier)
from options_alpha.synthetic import (MarketConfig, TRADING_DAYS,  # noqa: E402
                                     bs_price, bs_price_scalar,
                                     generate_market, smile_iv,
                                     strike_for_delta)

SMALL = MarketConfig(n_names=25, n_days=252 * 6, seed=5)


# --- the generator must be fair --------------------------------------------

def test_premium_is_actually_harvestable():
    """
    THE critical test. Frictionless ATM straddle selling must make money in
    this market, or the market does not contain the premium the strategy is
    built to harvest, and every backtest result is measuring the generator's
    parameters rather than the strategy.
    """
    mkt = MarketConfig(n_names=30, n_days=252 * 8, seed=3)
    panel = generate_market(mkt)
    sp = panel.pivot(index="day", columns="name", values="spot").to_numpy()
    iv = panel.pivot(index="day", columns="name", values="atm_iv").to_numpy()
    dte = 21
    t_years = dte / TRADING_DAYS
    premium = payout = 0.0
    for t in range(70, len(sp) - dte, 21):
        for j in range(sp.shape[1]):
            s, v = sp[t, j], iv[t, j]
            premium += (bs_price_scalar(s, s, t_years, v, True, mkt.risk_free)
                        + bs_price_scalar(s, s, t_years, v, False, mkt.risk_free)) / s
            st = sp[t + dte, j]
            payout += (max(0.0, st - s) + max(0.0, s - st)) / s
    capture = (premium - payout) / premium
    assert capture > 0.03, f"market does not reward vol selling: capture={capture:.3%}"
    assert capture < 0.30, f"premium implausibly generous: capture={capture:.3%}"


def test_volatility_process_is_stationary():
    """Guards the compounding bug: vol must mean-revert, not explode."""
    panel = generate_market(SMALL)
    rv = panel["rv_true"]
    assert 0.15 < rv.mean() < 0.60, f"mean vol {rv.mean():.3f} is not plausible"
    assert rv.quantile(0.99) < 2.0
    # second half must not be systematically higher than the first
    first = panel[panel.day < SMALL.n_days // 2]["rv_true"].mean()
    second = panel[panel.day >= SMALL.n_days // 2]["rv_true"].mean()
    assert abs(second - first) / first < 0.5, "vol drifts -- process is not stationary"


def test_year_fraction_uses_trading_days():
    """
    Volatility is annualised with sqrt(252), so a 21-trading-day option must be
    priced with t = 21/252. Using 21/365 underprices it by ~17%, which silently
    swamps any premium being measured.
    """
    s = 100.0
    correct = bs_price_scalar(s, s, 21 / TRADING_DAYS, 0.30, True)
    calendar = bs_price_scalar(s, s, 21 / 365.0, 0.30, True)
    assert correct > calendar
    assert 0.14 < (correct - calendar) / calendar < 0.22

    cfg = StrategyConfig(dte=21)
    built = build_condor(100.0, 0.30, cfg, MarketConfig())
    assert built is not None
    _, credit, _ = built
    ref = bs_price_scalar(100.0, 100.0, cfg.dte / TRADING_DAYS, 0.30, False)
    assert credit < ref, "credit should be less than a single ATM put"


def test_shocks_are_generated_and_hurt():
    mkt = MarketConfig(n_names=20, n_days=252 * 10, seed=9,
                       shock_per_year=1 / 2, shock_vol_mult=4.0)
    panel = generate_market(mkt)
    assert panel.attrs["shock_days"], "no shocks generated despite high frequency"
    s0 = panel.attrs["shock_days"][0]
    during = panel[(panel.day >= s0) & (panel.day < s0 + 10)]["rv_true"].mean()
    before = panel[(panel.day >= s0 - 40) & (panel.day < s0 - 10)]["rv_true"].mean()
    assert during > 1.5 * before, "shock did not raise volatility"


def test_smile_makes_downside_puts_richer():
    mkt = MarketConfig()
    assert smile_iv(0.30, 0.90, mkt) > smile_iv(0.30, 1.00, mkt)
    assert smile_iv(0.30, 1.10, mkt) < smile_iv(0.30, 1.00, mkt)


# --- pricing -----------------------------------------------------------------

def test_scalar_and_vector_pricing_agree():
    for args in [(100, 95, 0.1, 0.3, True), (50, 60, 0.25, 0.8, False),
                 (200, 200, 0.02, 0.15, True)]:
        a = bs_price_scalar(*args)
        b = float(bs_price(*args))
        assert abs(a - b) < 1e-6, args


def test_put_call_parity():
    s, k, t, v, r = 100.0, 105.0, 0.15, 0.35, 0.04
    c = bs_price_scalar(s, k, t, v, True, r)
    p = bs_price_scalar(s, k, t, v, False, r)
    assert abs((c - p) - (s - k * np.exp(-r * t))) < 1e-8


def test_strike_for_delta_scales_with_vol():
    """Higher vol must push a fixed-delta strike further from spot."""
    lo = strike_for_delta(100, 0.1, 0.20, -0.16, is_call=False)
    hi = strike_for_delta(100, 0.1, 0.60, -0.16, is_call=False)
    assert hi < lo < 100


# --- position structure ------------------------------------------------------

def test_condor_payoff_is_bounded_by_width():
    k = {"short_put": 90.0, "long_put": 85.0,
         "short_call": 110.0, "long_call": 115.0}
    for s in (10, 60, 85, 90, 100, 110, 115, 400):
        assert 0.0 <= condor_payoff(s, k) <= 5.0 + 1e-9, s
    assert condor_payoff(100, k) == 0.0
    assert condor_payoff(50, k) == 5.0


def test_put_only_structure_ignores_the_call_side():
    k = {"short_put": 90.0, "long_put": 85.0}
    assert condor_payoff(200.0, k) == 0.0      # no call leg to lose on
    assert condor_payoff(50.0, k) == 5.0


def test_refuses_structures_with_impossible_strikes():
    """At extreme vol the wing lands below zero; that must be refused."""
    cfg = StrategyConfig(wing_width_frac=4.0, dte=35)
    assert build_condor(100.0, 2.5, cfg, MarketConfig()) is None


def test_refuses_trades_the_spread_would_eat():
    """
    Credit must fall monotonically as fills worsen, and a bad enough market must
    be refused outright rather than traded at any price.

    Also records something the family model got conservatively wrong: at a full
    spread cross (fill_quality=1.0) the net credit is only ~4% below the
    mid-price credit, because the bought and sold legs partly offset. The
    four-leg structure is considerably cheaper to trade than a leg-by-leg cost
    estimate suggests.
    """
    mkt = MarketConfig()
    credits = []
    refused_at = None
    for fq in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0, 20.0):
        built = build_condor(100.0, 0.30, StrategyConfig(fill_quality=fq), mkt)
        if built is None:
            refused_at = fq
            break
        credits.append(built[1])
    assert refused_at is not None, "no spread is ever bad enough to refuse"
    assert all(a > b for a, b in zip(credits, credits[1:])), credits
    assert credits[2] / credits[0] > 0.90   # full cross costs <10% of the credit


def test_max_loss_and_credit_are_consistent():
    built = build_condor(100.0, 0.35, StrategyConfig(), MarketConfig())
    assert built is not None
    _, credit, max_loss = built
    assert credit > 0 and max_loss > 0


# --- signal and defence ------------------------------------------------------

def test_vol_forecast_tracks_actual_vol():
    rng = np.random.default_rng(0)
    for true_vol in (0.15, 0.45):
        daily = true_vol / np.sqrt(TRADING_DAYS)
        path = 100 * np.exp(np.cumsum(rng.normal(0, daily, (300, 1)), axis=0))
        est = forecast_realised_vol(path)[0]
        assert 0.6 * true_vol < est < 1.6 * true_vol, (true_vol, est)


def test_regime_filter_cuts_size_when_vol_accelerates():
    # Enabled explicitly: the default is now OFF, because the ablation showed
    # the filter costs return without improving any risk measure.
    cfg = StrategyConfig(use_regime_filter=True)
    calm = 100 * np.exp(np.cumsum(np.full(200, 0.0005)))
    assert regime_multiplier(calm, cfg) == 1.0
    rng = np.random.default_rng(1)
    spiking = np.concatenate([
        100 * np.exp(np.cumsum(rng.normal(0, 0.004, 180))),
        100 * np.exp(np.cumsum(rng.normal(0, 0.045, 20)))])
    assert regime_multiplier(spiking, cfg) < 1.0
    assert regime_multiplier(spiking, replace(cfg, use_regime_filter=False)) == 1.0


# --- backtest integrity ------------------------------------------------------

def test_no_lookahead():
    """
    Truncating the panel must not change any decision made before the cut.

    If the engine peeked at future data, the equity path over the shared period
    would differ between the truncated and full runs.
    """
    panel = generate_market(MarketConfig(n_names=15, n_days=700, seed=4))
    cut = 500
    full = run_backtest(panel, StrategyConfig(n_positions=5))["curve"]
    part = run_backtest(panel[panel.day < cut].copy(),
                        StrategyConfig(n_positions=5))["curve"]
    shared = part.index.intersection(full.index)
    shared = shared[shared < cut - StrategyConfig().dte - 5]
    assert len(shared) > 100
    np.testing.assert_allclose(full.loc[shared, "equity"],
                               part.loc[shared, "equity"], rtol=1e-9)


def test_capital_at_risk_respects_the_portfolio_limit():
    cfg = StrategyConfig(max_portfolio_risk=0.15, n_positions=10)
    res = run_backtest(generate_market(SMALL), cfg)
    curve = res["curve"]
    # Allow headroom for equity marked down after positions were sized.
    assert (curve["at_risk"] / curve["equity"]).max() < cfg.max_portfolio_risk * 1.6


def test_losses_are_bounded_by_defined_risk():
    """No trade may lose more than the capital committed to it."""
    res = run_backtest(generate_market(SMALL), StrategyConfig())
    tr = res["trades"]
    assert len(tr) > 50
    assert tr["return_on_risk"].min() >= -1.0001, tr["return_on_risk"].min()


def test_costs_reduce_returns():
    panel = generate_market(SMALL)
    free = run_backtest(panel, StrategyConfig(fill_quality=0.0))["stats"]
    paid = run_backtest(panel, StrategyConfig(fill_quality=1.0))["stats"]
    assert free["cagr"] > paid["cagr"]


# --- research discipline -----------------------------------------------------

def test_split_is_chronological_and_disjoint():
    panel = generate_market(SMALL)
    tune, test = R.split_panel(panel)
    assert tune["day"].max() < panel["day"].max()
    assert len(tune) > 0 and len(test) > 0
    assert len(tune) + len(test) == len(panel)


def test_objective_rejects_unsurvivable_configs():
    good = {"cagr": 0.06, "max_drawdown": -0.15}
    reckless = {"cagr": 0.25, "max_drawdown": -0.65}
    assert R.objective(good) > R.objective(reckless)
    assert R.objective(reckless) < 0


# --- family screen -----------------------------------------------------------

def test_screen_passes_the_synthesis_and_fails_naked_index_vol():
    v = R
    verdicts = dict(zip(F.screen()["family"], F.screen()["verdict"]))
    assert verdicts[F.DEFINED_RISK_VRP.name] == "PASS"
    assert verdicts[F.INDEX_SHORT_VOL.name] == "FAIL"
    assert verdicts[F.FLOW_EVENT_DRIVEN.name] == "FAIL"


def test_risk_premium_and_mispricing_carry_different_proof_burdens():
    assert F.DEFINED_RISK_VRP.proof_burden() == "implement"
    assert F.FLOW_EVENT_DRIVEN.proof_burden() == "discover"
    assert F.EARNINGS_VOL_CRUSH.proof_burden() == "discover"


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
