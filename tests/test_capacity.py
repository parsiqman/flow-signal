"""
Tests for the capital model behind the longshot rule.

The model's job is to stop a per-trade percentage being mistaken for a return.
These check the three steps where that mistake happens.

    python tests/test_capacity.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polymarket import capacity as C  # noqa: E402


def test_depth_caps_the_stake_so_roi_per_trade_is_not_a_return():
    """8.8% per trade on $80 of depth is $7, not a business by itself."""
    b = C.MEASURED_BANDS[1]                       # 0.80-0.90
    e = C.per_trade_economics(b, C.Assumptions())
    assert e["stake_usd"] < b.depth_usd
    assert e["profit_usd"] < 10.0


def test_capital_recycles_so_turnover_far_exceeds_the_capital_base():
    """
    The capital base is small because positions close and the money comes
    back. That same recycling means a bad year loses a MULTIPLE of capital,
    which is leverage by another name.
    """
    m = C.annual_model()
    bb = m["by_band"]
    turnover = float((bb["stake_usd"] * bb["trades_per_year"]).sum())
    assert turnover > 10 * m["capital_required_usd"]


def test_correlated_losses_destroy_the_breadth_argument():
    """
    Independent bets give a Sharpe that scales with sqrt(n). A one-directional
    book does not get that, and the difference is not marginal.
    """
    indep = C.drawdown_simulation(a=C.Assumptions(loss_correlation=0.0))
    corr = C.drawdown_simulation(a=C.Assumptions(loss_correlation=0.25))
    assert indep["prob_losing_year"] < 0.01
    assert corr["prob_losing_year"] > 0.10
    assert corr["p01_annual_profit_usd"] < indep["p01_annual_profit_usd"]


def test_a_small_correlation_is_enough_to_wipe_out_the_capital_base():
    """
    The threshold is the finding: ruin does not need a large correlation, and
    a long-favourite book cannot plausibly be below it.
    """
    cap = C.annual_model()["capital_required_usd"]
    safe = C.drawdown_simulation(a=C.Assumptions(loss_correlation=0.02))
    ruin = C.drawdown_simulation(a=C.Assumptions(loss_correlation=0.10))
    assert safe["p01_annual_profit_usd"] > -cap
    assert ruin["p01_annual_profit_usd"] < -cap


def test_the_return_survives_heavy_adverse_selection_but_the_risk_does_not():
    """
    Return is robust: most of the measured edge can be lost and the rule still
    pays. That is exactly why return is the wrong thing to look at here.
    """
    assert C.breakeven_adverse_selection() > 0.5
    m = C.annual_model(a=C.Assumptions(adverse_selection=0.70))
    assert m["return_on_capital"] > 0.15
    assert m["sharpe_with_correlation"] < 0.5


def test_every_assumption_is_named_rather_than_buried():
    a = C.Assumptions()
    for f in ("depth_fraction", "adverse_selection", "fill_rate", "hold_days",
              "loss_correlation", "capital_utilisation"):
        assert hasattr(a, f), f


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
