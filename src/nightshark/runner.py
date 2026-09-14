"""The 15-minute loop: data -> features -> Claude -> risk gate -> order -> settle.

One deliberate omission: the model is NOT told the contract's market price.

It would be easy to pass "YES is trading at 59c" into the prompt, and the model
would use it -- that price is the market's own probability estimate, built from
far more information than this feature summary. The model would learn to echo it,
the bot would look calibrated, and it would have no independent signal at all.
So the model gets market STATE and answers direction; the runner separately
decides, mechanically, whether the offered price is worth paying. Keeping those
two judgements apart is the only way to find out whether the model knows anything.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from .config import Config
from .data import build_feed
from .execution import Fill, PaperBroker, build_broker, select_contract, side_for
from .features import extract
from .llm import ClaudeDirectionModel, ScriptedModel

TRADE_LOG = "results/nightshark/trades.jsonl"


def window_bounds(now: datetime, minutes: int = 15) -> tuple[datetime, datetime]:
    """The settlement window containing `now`, aligned to the clock."""
    epoch_min = int(now.timestamp()) // 60
    start_min = (epoch_min // minutes) * minutes
    start = datetime.fromtimestamp(start_min * 60, timezone.utc)
    return start, start + timedelta(minutes=minutes)


def log_event(row: dict, path: str = TRADE_LOG) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


@dataclass
class WindowResult:
    decided_at: datetime
    window_end: datetime
    skipped: str = ""
    signal: dict | None = None
    fill: dict | None = None
    settlement_price: float = 0.0
    pnl: float | None = None

    def to_dict(self) -> dict:
        return {
            "decided_at": self.decided_at.isoformat(),
            "window_end": self.window_end.isoformat(),
            "skipped": self.skipped, "signal": self.signal, "fill": self.fill,
            "settlement_price": round(self.settlement_price, 2),
            "pnl": None if self.pnl is None else round(self.pnl, 2),
        }


def decide(cfg, feed, broker, model, decided_at, window_start, window_end):
    """Everything up to (not including) placing the order. Pure enough to test."""
    hist = feed.history(decided_at, cfg.history_minutes)
    spot = float(hist.close[-1])
    minutes_left = (window_end - decided_at).total_seconds() / 60

    open_hist = feed.history(window_start + timedelta(minutes=1), 2)
    window_open_price = float(open_hist.close[-1]) if len(open_hist) else spot

    feats = extract(hist, decided_at, window_end, window_open_price)

    contracts = broker.list_contracts(
        series_ticker=cfg.series_ticker, window_end=window_end, spot=spot,
        minutes_left=minutes_left, annual_vol_pct=feats.values["rv_60m"],
    )

    contract = select_contract(contracts, spot, window_end)
    if contract is None:
        return feats, None, None, "no contract for this window"

    # The threshold IS the question. Distance to it in bp is decision-relevant;
    # the contract's price deliberately is not (see module docstring).
    dist_bp = (spot / contract.strike - 1) * 10_000
    ctx = (f"REFERENCE strike {contract.strike:,.0f}; spot is {dist_bp:+.0f}bp "
           f"{'above' if dist_bp >= 0 else 'below'} it. "
           f"Call UP if price at settlement will be ABOVE {contract.strike:,.0f}.")

    signal = model.predict(feats.to_prompt_text(), ctx)
    return feats, contract, signal, ""


def run_window(cfg, feed, broker, model, risk, decided_at, settle_now=True,
               precomputed=None) -> WindowResult:
    """`precomputed` is a decide() result computed elsewhere -- it lets the
    simulator fan the model calls out across threads and then apply them to the
    ledger in strict chronological order, which the running equity depends on."""
    window_start, window_end = window_bounds(decided_at, cfg.window_minutes)
    res = WindowResult(decided_at=decided_at, window_end=window_end)

    feats, contract, signal, why = precomputed if precomputed is not None else decide(
        cfg, feed, broker, model, decided_at, window_start, window_end)
    if why:
        res.skipped = why
        return res
    res.signal = signal.to_dict()

    if not signal.tradeable:
        res.skipped = f"no signal ({signal.error or 'abstain'})"
        return res
    if signal.confidence < cfg.min_confidence:
        res.skipped = f"confidence {signal.confidence:.2f} < {cfg.min_confidence:.2f}"
        return res

    side = side_for(signal.direction)
    price = contract.ask_for(side)
    if price < cfg.min_price_cents or price > cfg.max_price_cents:
        res.skipped = f"{side} ask {price}c outside [{cfg.min_price_cents},{cfg.max_price_cents}]"
        return res

    est_cost = (cfg.contracts_per_trade * price) / 100.0
    ok, reason = risk.allow_trade(est_cost, now=decided_at)
    if not ok:
        res.skipped = f"risk: {reason}"
        return res

    fill, err = broker.buy(contract, side, cfg.contracts_per_trade, cfg.max_price_cents)
    if fill is None:
        res.skipped = f"not filled: {err}"
        return res
    res.fill = fill.to_dict()
    risk.record_entry(fill.cost, now=decided_at)

    if settle_now:
        settlement = feed.spot(window_end)
        res.settlement_price = settlement
        res.pnl = fill.settle(settlement)
        risk.record_settlement(res.pnl, now=window_end)
    return res


def _emit(res: WindowResult, risk) -> None:
    log_event(res.to_dict())
    sig = res.signal or {}
    head = f"{res.decided_at:%m-%d %H:%M}Z -> {res.window_end:%H:%M}Z"
    if res.skipped and not res.fill:
        print(f"{head}  SKIP  {res.skipped}"
              + (f"  [{sig.get('direction')} {sig.get('confidence')}]" if sig else ""))
        return
    print(f"{head}  {sig.get('direction')} conf {sig.get('confidence'):.2f} "
          f"@ {res.fill['price_cents']}c x{res.fill['count']} "
          f"strike {res.fill['strike']:,.0f} -> settle {res.settlement_price:,.0f} "
          f"pnl ${res.pnl:+.2f}   | {risk.summary()}")
    if sig.get("reason"):
        print(f"           why: {sig['reason']}")


def relax_limits(cfg):
    """Config with the risk limits lifted, for measurement runs only.

    The kill switch and the measurement want opposite things. Live, you want to
    stop after six losses. Measuring, that stop truncates the sample at the
    worst possible moment -- a losing streak -- and the hit rate you read back
    is conditioned on having stopped, which is worse than no number at all.
    So `--measure` observes the whole sample. It is never a live mode: the
    ledger still records every trade, nothing gates it.
    """
    return replace(cfg, max_drawdown=1e12, daily_loss_limit=1e12,
                   max_consecutive_losses=10 ** 9, max_trades_per_day=10 ** 9)


def simulate(cfg, n_windows: int, model, risk, measuring: bool = False,
             workers: int = 1) -> None:
    """Replay N consecutive past windows end-to-end. No waiting, no real money.

    This is the measurement tool. Run it before the live loop and after any
    change to features or prompt -- and read the hit rate against the breakeven
    it prints, not against 50%.
    """
    if measuring:
        print("MEASURE MODE: risk limits lifted to observe the full sample. "
              "Not a live configuration.\n")
    feed = build_feed(cfg)
    broker = build_broker(cfg)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    lead = timedelta(minutes=cfg.window_minutes - 4)      # decide 4m before close

    stamps = []
    for i in range(n_windows, 0, -1):
        start, _ = window_bounds(now - timedelta(minutes=cfg.window_minutes * i),
                                 cfg.window_minutes)
        stamps.append(start + lead)

    # The model calls are independent, so fan them out; the ledger is not, so
    # apply the results strictly in order afterwards.
    plans = [None] * len(stamps)
    if workers > 1:
        def _plan(ts):
            ws, we = window_bounds(ts, cfg.window_minutes)
            return decide(cfg, feed, broker, model, ts, ws, we)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            plans = list(ex.map(_plan, stamps))

    prices = []
    for idx, ts in enumerate(stamps):
        res = run_window(cfg, feed, broker, model, risk, ts, settle_now=True,
                         precomputed=plans[idx])
        _emit(res, risk)
        if res.fill:
            prices.append(res.fill["price_cents"])
        if risk.state.halted:
            print(f"\nStopped early: {risk.state.halt_reason}")
            break

    s = risk.state
    print(f"\n{'='*72}\n{risk.summary()}")
    if prices and s.trades:
        avg = sum(prices) / len(prices)
        be = avg / 100 + 0.07 * (avg / 100) * (1 - avg / 100)
        print(f"avg entry {avg:.0f}c -> breakeven hit rate ~{be:.0%}; "
              f"actual {s.hit_rate:.0%} over {s.trades} settled trades")
        print("A hit rate below breakeven means this configuration loses money.")


def live(cfg, model, risk) -> None:
    """The real loop. Decides once per window, sleeps between."""
    feed = build_feed(cfg)
    broker = build_broker(cfg)
    lead = timedelta(minutes=4)
    print(f"live loop | {cfg.describe_risk()} | feed={cfg.feed} model={cfg.model}")

    pending = []
    while not risk.state.halted:
        now = datetime.now(timezone.utc)
        _, window_end = window_bounds(now, cfg.window_minutes)
        decide_at = window_end - lead
        if decide_at <= now:                       # too late for this one
            _, window_end = window_bounds(window_end + timedelta(seconds=1), cfg.window_minutes)
            decide_at = window_end - lead

        time.sleep(max(0.0, (decide_at - datetime.now(timezone.utc)).total_seconds()))
        res = run_window(cfg, feed, broker, model, risk,
                         datetime.now(timezone.utc), settle_now=False)
        if res.fill:
            pending.append((res, window_end))
        else:
            _emit(res, risk)

        # Settle anything whose window has closed (+30s for the feed to catch up).
        time.sleep(max(0.0, (window_end - datetime.now(timezone.utc)).total_seconds() + 30))
        still = []
        for r, end in pending:
            if datetime.now(timezone.utc) < end + timedelta(seconds=30):
                still.append((r, end))
                continue
            r.settlement_price = feed.spot(end)
            fill = Fill(ticker=r.fill["ticker"], side=r.fill["side"], count=r.fill["count"],
                        price_cents=r.fill["price_cents"], fee_cents=r.fill["fee_cents"],
                        order_id=r.fill["order_id"], strike=r.fill["strike"])
            r.pnl = fill.settle(r.settlement_price)
            risk.record_settlement(r.pnl, now=end)
            _emit(r, risk)
        pending = still

    print(f"\nHALTED: {risk.state.halt_reason}\n{risk.summary()}")
    print("Clear it deliberately with --reset-halt once you know why.")


def main(argv=None) -> int:
    from .risk import RiskManager

    p = argparse.ArgumentParser(prog="nightshark", description=__doc__.split("\n")[0])
    p.add_argument("--simulate", type=int, metavar="N",
                   help="replay N past windows end-to-end and report (no waiting)")
    p.add_argument("--live", action="store_true", help="run the real 15-minute loop")
    p.add_argument("--list-markets", action="store_true",
                   help="print the Kalshi markets this key can see (verifies the series ticker)")
    p.add_argument("--reset-halt", action="store_true", help="clear a tripped kill switch")
    p.add_argument("--status", action="store_true", help="print risk state and exit")
    p.add_argument("--measure", action="store_true",
                   help="with --simulate: lift risk limits so the whole sample is "
                        "observed instead of stopping at the first losing streak")
    p.add_argument("--workers", type=int, default=1, metavar="N",
                   help="with --simulate: run N model calls concurrently "
                        "(the ledger is still applied in order)")
    p.add_argument("--scripted", action="store_true",
                   help="use the offline stand-in model instead of the API")
    p.add_argument("--state", default=None, help="override the risk state file")
    args = p.parse_args(argv)

    cfg = Config.from_env()
    if args.measure:
        if not args.simulate:
            p.error("--measure only applies to --simulate")
        cfg = relax_limits(cfg)
    risk = RiskManager(cfg, state_path=args.state)

    if args.status:
        print(cfg.describe_risk()); print(risk.summary()); return 0
    if args.reset_halt:
        risk.reset_halt(); print("halt cleared."); print(risk.summary()); return 0
    if args.list_markets:
        broker = build_broker(cfg)
        if isinstance(broker, PaperBroker):
            print("dry_run is on -- set NIGHTSHARK_DRY_RUN=false to query the real exchange.")
            return 1
        for c in broker.list_contracts(series_ticker=cfg.series_ticker):
            print(f"{c.ticker:<40} strike {c.strike:>10,.0f}  closes {c.close_time:%Y-%m-%d %H:%M}Z"
                  f"  yes {c.yes_bid}/{c.yes_ask}")
        return 0

    model = ScriptedModel() if args.scripted else ClaudeDirectionModel(
        model=cfg.model, effort=cfg.effort, max_tokens=cfg.max_tokens,
        timeout_s=cfg.llm_timeout_s)

    if args.simulate:
        simulate(cfg, args.simulate, model, risk, measuring=args.measure,
                 workers=max(1, args.workers)); return 0
    if args.live:
        if not cfg.dry_run:
            print("!! LIVE MONEY. Ctrl-C within 10s to abort."); time.sleep(10)
        live(cfg, model, risk); return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
