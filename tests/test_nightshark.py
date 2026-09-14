"""
Tests for the Kalshi 15-minute BTC bot.

The interesting failures in a trading loop are silent ones: a feature that peeks
one candle into the future, a drawdown limit that resets when the process
restarts, a direction that maps to the wrong side of the contract. None of those
raise an exception -- they just quietly change what the backtest claims and what
the account does. So each gets an explicit test here.

The money-losing paths are asserted as carefully as the money-making ones: an
API error must produce NO trade, never a coin flip.

    python tests/test_nightshark.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nightshark.config import Config
from nightshark.data import CandleSeries, SyntheticFeed
from nightshark.execution import (NO, YES, Fill, PaperBroker, binary_fair_value,
                                  fee_cents, select_contract, side_for)
from nightshark.features import atr, extract, rsi
from nightshark.llm import ABSTAIN, UP, ClaudeDirectionModel, ScriptedModel
from nightshark.risk import RiskManager
from nightshark.runner import decide, run_window, window_bounds

NOW = datetime(2026, 9, 14, 16, 4, tzinfo=timezone.utc)


def _cfg(**kw):
    base = dict(starting_bankroll=1000.0, state_path="", dry_run=True)
    base.update(kw)
    return Config(**base)


# -- data -------------------------------------------------------------------
def test_feed_never_returns_an_unclosed_candle():
    """The whole no-lookahead guarantee rests on this one inequality."""
    feed = SyntheticFeed(seed=3)
    for offset in range(0, 240, 7):
        as_of = NOW - timedelta(minutes=offset)
        h = feed.history(as_of, 120)
        assert (h.ts + 60 <= as_of.timestamp()).all(), f"lookahead at {as_of}"


def test_history_is_a_function_of_as_of_only():
    """Same timestamp, same data -- regardless of what was asked for before."""
    feed = SyntheticFeed(seed=3)
    a = feed.history(NOW, 120)
    feed.history(NOW + timedelta(hours=5), 300)      # peek into the future
    b = feed.history(NOW, 120)
    assert np.array_equal(a.close, b.close)


def test_resample_aligns_to_the_clock_and_drops_partial_buckets():
    feed = SyntheticFeed(seed=3)
    h = feed.history(NOW, 307)                        # deliberately not a multiple of 15
    r = h.resample(15)
    assert set((r.ts % 900).tolist()) == {0}, "15m buckets must land on :00/:15/:30/:45"
    assert (r.high >= r.low).all()
    # every bucket kept must be complete
    assert r.ts[-1] + 900 <= NOW.timestamp()


# -- features ---------------------------------------------------------------
def test_features_are_finite_and_small():
    feed = SyntheticFeed(seed=5)
    f = extract(feed.history(NOW, 360), NOW, NOW + timedelta(minutes=11), 84000.0)
    for k, v in f.values.items():
        if isinstance(v, float):
            assert np.isfinite(v), f"{k} is {v}"
    text = f.to_prompt_text()
    assert len(text) < 1200, f"summary bloated to {len(text)} chars"
    for tag in ("VOL", "TREND", "MOM", "RANGE", "CANDLES"):
        assert tag in text


def test_rsi_and_atr_on_known_shapes():
    rising = np.arange(100, 140, dtype=float)
    assert rsi(rising, 14) > 99, "a monotonic rise must pin RSI at ~100"
    falling = rising[::-1].copy()
    assert rsi(falling, 14) < 1
    flat = np.full(40, 100.0)
    assert 49 <= rsi(flat, 14) <= 51

    # ATR is returned in bp of price, so a 1% true range reads ~100bp
    n = 30
    s = CandleSeries(ts=np.arange(n) * 60.0, open=np.full(n, 100.0),
                     high=np.full(n, 100.5), low=np.full(n, 99.5),
                     close=np.full(n, 100.0), volume=np.ones(n), interval_s=60)
    assert abs(atr(s, 14) - 100.0) < 1.0, atr(s, 14)


def test_features_reject_too_little_history():
    feed = SyntheticFeed(seed=5)
    try:
        extract(feed.history(NOW, 30), NOW)
        raise AssertionError("should have refused 30 candles")
    except ValueError:
        pass


# -- execution --------------------------------------------------------------
def test_direction_maps_to_the_right_side():
    assert side_for("UP") == YES
    assert side_for("DOWN") == NO
    for bad in (ABSTAIN, "up", ""):
        try:
            side_for(bad)
            raise AssertionError(f"{bad!r} should not be tradeable")
        except ValueError:
            pass


def test_select_contract_picks_the_nearest_strike_in_the_right_window():
    end = NOW + timedelta(minutes=11)
    b = PaperBroker()
    cs = b.list_contracts(window_end=end, spot=84326.0, minutes_left=11, annual_vol_pct=60)
    picked = select_contract(cs, 84326.0, end)
    assert abs(picked.strike - 84326.0) <= 125.0, "must pick the nearest strike"
    # a contract from a different window must never be selected
    other = b.list_contracts(window_end=end + timedelta(minutes=30), spot=84326.0,
                             minutes_left=41, annual_vol_pct=60)
    assert select_contract(other, 84326.0, end) is None


def test_settlement_pnl_both_directions():
    f = Fill(ticker="T", side=YES, count=10, price_cents=50, fee_cents=18,
             order_id="x", strike=84000.0)
    assert abs(f.cost - 5.18) < 1e-9
    assert abs(f.settle(84100.0) - (10 - 5.18)) < 1e-9      # YES wins
    assert abs(f.settle(83900.0) + 5.18) < 1e-9             # YES loses: lose the stake
    n = Fill(ticker="T", side=NO, count=10, price_cents=50, fee_cents=18,
             order_id="x", strike=84000.0)
    assert abs(n.settle(83900.0) - (10 - 5.18)) < 1e-9      # NO wins when price falls
    assert abs(n.settle(84100.0) + 5.18) < 1e-9


def test_fees_peak_at_a_coin_flip():
    """Kalshi's fee is maximal at 50c -- exactly where direction bets live."""
    assert fee_cents(50, 100) >= fee_cents(20, 100)
    assert fee_cents(50, 100) >= fee_cents(80, 100)
    assert fee_cents(50, 1) >= 1                            # always rounds up


def test_fair_value_is_monotonic_and_bounded():
    fv = lambda strike: binary_fair_value(84000.0, strike, 10.0, 60.0)
    assert fv(83000) > fv(84000) > fv(85000)
    assert 0.0 <= fv(90000) <= 1.0 and 0.0 <= fv(70000) <= 1.0
    assert abs(fv(84000) - 0.5) < 0.02, "at-the-money must price near 50c"
    assert binary_fair_value(84000, 83000, 0, 60) == 1.0     # expired, in the money


# -- risk -------------------------------------------------------------------
def test_drawdown_halts_and_survives_restart():
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "s.json")
        cfg = _cfg(max_drawdown=20.0, daily_loss_limit=1e9, state_path=path)
        r = RiskManager(cfg)
        for _ in range(4):
            r.record_entry(6.0)
            r.record_settlement(-6.0)
        assert r.state.halted
        assert not RiskManager(cfg).allow_trade(1.0)[0], "halt must survive restart"


def test_reset_rebaselines_so_trading_can_actually_resume():
    with tempfile.TemporaryDirectory() as d:
        cfg = _cfg(max_drawdown=20.0, daily_loss_limit=1e9,
                   state_path=str(Path(d) / "s.json"))
        r = RiskManager(cfg)
        for _ in range(4):
            r.record_entry(6.0)
            r.record_settlement(-6.0)
        r.reset_halt()
        ok, why = r.allow_trade(6.0)
        assert ok, f"reset must let trading resume, got {why}"
        assert r.state.realized_pnl == -24.0, "the loss stays on the books"


def test_daily_counters_follow_simulated_time_not_the_wall_clock():
    """Replaying hours of windows must not trip the LIVE daily caps, or the
    measurement stops early and reports a hit rate on a truncated sample."""
    with tempfile.TemporaryDirectory() as d:
        cfg = _cfg(max_trades_per_day=3, state_path=str(Path(d) / "s.json"))
        r = RiskManager(cfg)
        day1 = datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc)
        for i in range(3):
            r.record_entry(1.0, now=day1 + timedelta(minutes=15 * i))
        assert not r.allow_trade(1.0, now=day1 + timedelta(hours=1))[0], "cap applies within a day"
        ok, why = r.allow_trade(1.0, now=day1 + timedelta(days=1))
        assert ok, f"a new simulated day must reset the cap, got {why}"


def test_other_limits_block_before_the_drawdown_does():
    with tempfile.TemporaryDirectory() as d:
        cfg = _cfg(max_drawdown=1e9, daily_loss_limit=10.0,
                   state_path=str(Path(d) / "s.json"))
        r = RiskManager(cfg)
        r.record_entry(6.0); r.record_settlement(-11.0)
        assert not r.allow_trade(6.0)[0]

        cfg2 = _cfg(max_drawdown=1e9, daily_loss_limit=1e9, max_consecutive_losses=3,
                    state_path=str(Path(d) / "s2.json"))
        r2 = RiskManager(cfg2)
        for _ in range(3):
            r2.record_entry(1.0); r2.record_settlement(-1.0)
        assert not r2.allow_trade(1.0)[0]

    with tempfile.TemporaryDirectory() as d:
        cfg3 = _cfg(state_path=str(Path(d) / "s3.json"))
        r3 = RiskManager(cfg3)
        assert not r3.allow_trade(10_000.0)[0], "cannot spend more than equity"


# -- model plumbing ---------------------------------------------------------
class _Block:
    def __init__(self, text): self.type, self.text = "text", text


class _Resp:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_Block(text)]
        self.stop_reason = stop_reason
        self.stop_details = None
        self.model = "claude-opus-5"
        self.usage = type("U", (), {"input_tokens": 500, "output_tokens": 40})()


class _Client:
    """Minimal stand-in for anthropic.Anthropic."""
    def __init__(self, resp=None, raises=None):
        self._resp, self._raises = resp, raises
        self.messages = type("M", (), {"create": self._create})()
        self.last_kwargs = None

    def with_options(self, **_):
        return self

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._raises:
            raise self._raises
        return self._resp


def test_good_response_parses():
    c = _Client(_Resp(json.dumps({"direction": "UP", "confidence": 0.62,
                                  "reason": "clean uptrend"})))
    s = ClaudeDirectionModel(client=c).predict("VOL ...")
    assert s.direction == UP and s.tradeable
    assert abs(s.confidence - 0.62) < 1e-9
    assert s.input_tokens == 500
    # the request must carry the schema and the model we configured
    assert c.last_kwargs["output_config"]["format"]["type"] == "json_schema"
    assert c.last_kwargs["model"] == "claude-opus-5"


def test_every_failure_mode_abstains_rather_than_guessing():
    cases = {
        "malformed json": _Client(_Resp("not json at all")),
        "wrong direction": _Client(_Resp(json.dumps({"direction": "SIDEWAYS",
                                                     "confidence": 0.9, "reason": ""}))),
        "refusal": _Client(_Resp("{}", stop_reason="refusal")),
        "truncated": _Client(_Resp("{", stop_reason="max_tokens")),
        "transport blew up": _Client(raises=RuntimeError("connection reset")),
    }
    for name, client in cases.items():
        s = ClaudeDirectionModel(client=client).predict("VOL ...")
        assert s.direction == ABSTAIN, f"{name} produced a trade: {s.direction}"
        assert not s.tradeable
        assert s.error, f"{name} abstained without recording why"


# -- end to end -------------------------------------------------------------
def test_window_bounds_align_to_quarter_hours():
    for minute, expect in ((0, 0), (7, 0), (14, 0), (15, 15), (44, 30), (59, 45)):
        start, end = window_bounds(NOW.replace(minute=minute), 15)
        assert start.minute == expect
        assert (end - start) == timedelta(minutes=15)


def test_full_window_produces_a_priced_settled_trade():
    with tempfile.TemporaryDirectory() as d:
        cfg = _cfg(state_path=str(Path(d) / "s.json"), contracts_per_trade=10)
        feed, broker = SyntheticFeed(seed=11), PaperBroker()
        risk = RiskManager(cfg)
        res = run_window(cfg, feed, broker, ScriptedModel([UP]), risk, NOW)
        assert res.fill is not None, f"no trade: {res.skipped}"
        assert res.fill["side"] == YES
        assert 1 <= res.fill["price_cents"] <= 99
        assert res.pnl is not None
        # PnL must be one of the only two outcomes a binary can have
        cost = res.fill["count"] * res.fill["price_cents"] / 100 + res.fill["fee_cents"] / 100
        assert abs(res.pnl - (10 - cost)) < 1e-6 or abs(res.pnl + cost) < 1e-6
        assert risk.state.trades == 1


def test_abstain_and_price_caps_block_the_order():
    with tempfile.TemporaryDirectory() as d:
        cfg = _cfg(state_path=str(Path(d) / "s.json"))
        feed, broker, risk = SyntheticFeed(seed=11), PaperBroker(), RiskManager(cfg)

        class _Abstainer:
            def predict(self, *a, **k):
                from nightshark.llm import Signal
                return Signal(direction=ABSTAIN, error="boom")

        res = run_window(cfg, feed, broker, _Abstainer(), risk, NOW)
        assert res.fill is None and "no signal" in res.skipped
        assert risk.state.trades == 0, "an abstain must not touch the account"

        tight = _cfg(state_path=str(Path(d) / "s2.json"), max_price_cents=2)
        res2 = run_window(tight, feed, broker, ScriptedModel([UP]),
                          RiskManager(tight), NOW)
        assert res2.fill is None and "outside" in res2.skipped


def test_halted_risk_blocks_trading_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        cfg = _cfg(state_path=str(Path(d) / "s.json"), max_drawdown=5.0,
                   daily_loss_limit=1e9)
        risk = RiskManager(cfg)
        risk.record_entry(6.0); risk.record_settlement(-6.0)
        assert risk.state.halted
        res = run_window(cfg, SyntheticFeed(seed=11), PaperBroker(),
                         ScriptedModel([UP]), risk, NOW)
        assert res.fill is None and "risk" in res.skipped


def test_model_is_not_told_the_contract_price():
    """If the market's own probability leaks into the prompt, the model just
    echoes it and the bot has no independent signal."""
    captured = {}

    class _Spy(ScriptedModel):
        def predict(self, feature_text, extra_context=""):
            captured["prompt"] = f"{feature_text}\n{extra_context}"
            return super().predict(feature_text, extra_context)

    cfg = _cfg()
    end = NOW + timedelta(minutes=11)
    start, _ = window_bounds(NOW, 15)
    feats, contract, sig, why = decide(cfg, SyntheticFeed(seed=11), PaperBroker(),
                                       _Spy([UP]), NOW, start, end)
    prompt = captured["prompt"]
    assert str(contract.yes_ask) + "c" not in prompt
    assert "ask" not in prompt.lower() and "bid" not in prompt.lower()
    assert f"{contract.strike:,.0f}" in prompt, "but the strike itself must be there"


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
        except Exception as e:
            failed += 1
            print(f"  ERROR {f.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
