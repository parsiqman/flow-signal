#!/usr/bin/env python3
"""
Polymarket wallet-skill scan. Runs unattended in CI and writes a report.

    python scripts/polymarket_scan.py --markets 200 --out results/
    python scripts/polymarket_scan.py --offline        # fixtures, no network

Answers one question: do wallets exist whose edge is large enough to be told
apart from the luckiest trader in a population of thousands?

Written to be read by someone who was asleep when it ran. Every stage prints
what it did, every failure is caught and reported rather than crashing the job
half-way, and the final verdict is stated in one line at the top of the report.

`--offline` runs the whole pipeline on synthetic fixtures with known ground
truth. That path is exercised in CI on every push, so a break in the analysis
code is caught without touching the network -- and when the live run fails, it
distinguishes "the API changed" from "our code is broken".
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polymarket import client, execution, fixtures, wallets   # noqa: E402


def log(msg: str = "") -> None:
    print(msg, flush=True)


def section(title: str) -> None:
    log(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def collect(args) -> tuple[pd.DataFrame, dict]:
    """Fetch and normalise, or generate fixtures when offline."""
    if args.offline:
        section("OFFLINE MODE - synthetic fixtures with known ground truth")
        trades, truth = fixtures.generate_wallets(
            n_wallets=args.offline_wallets, skilled_frac=args.offline_skilled,
            skill_edge=args.offline_edge, trades_per_wallet=(60, 300), seed=0,
            n_markets=400)
        log(f"generated {len(trades):,} trades across "
            f"{trades.wallet.nunique():,} wallets")
        log(f"ground truth: {int(truth.is_skilled.sum())} skilled wallets at "
            f"{args.offline_edge * 100:.0f}c/share")
        return trades, {"n_wallets_discovered": int(trades.wallet.nunique()),
                        "mode": "offline",
                        "n_truly_skilled": int(truth.is_skilled.sum())}

    section("1. FETCHING RESOLVED MARKETS")
    cfg = client.ClientConfig(cache_dir=args.cache, rate_limit_s=args.rate_limit)
    api = client.PolymarketClient(cfg)

    markets = api.resolved_markets(limit=args.market_pool,
                                   min_volume=args.min_volume)
    log(f"{len(markets):,} resolved markets above ${args.min_volume:,.0f} volume")
    if markets.empty:
        raise RuntimeError("no resolved markets returned; check the Gamma API")
    log(f"field mapping: {client.MARKET_FIELDS}")
    resolved = markets[markets["winning_index"].notna()]
    log(f"{len(resolved):,} have a determinable winning outcome")
    if resolved.empty:
        raise RuntimeError(
            "no market has a winning_index. outcomePrices is probably not in "
            "the expected shape -- see MARKET_FIELD_CANDIDATES in client.py")

    section("2. DISCOVERING WALLETS BY MARKET PARTICIPATION")
    log("NOT from the leaderboard: that selects on the outcome variable and")
    log("would make every number downstream meaningless.\n")
    raw, meta = client.discover_population(api, resolved,
                                           n_markets=args.markets,
                                           seed=args.seed)
    for k, v in meta.items():
        log(f"  {k:24} {v:,}" if isinstance(v, int) else f"  {k:24} {v}")
    if raw.empty:
        raise RuntimeError("no trades returned from any sampled market")

    log("\n" + client.describe_response(raw, "trades"))

    section("3. FETCHING FULL WALLET HISTORIES")
    log("Judging a wallet on whichever trades fell inside our market sample is")
    log("both a tiny sample and the wrong one. Candidates are ranked by ACTIVITY")
    log("(outcome-independent), never by profit.\n")
    seed_trades = client.normalise_trades(raw, resolved)
    activity = seed_trades.groupby("wallet").size().to_dict()
    candidates = list(activity)
    log(f"{len(candidates):,} candidate wallets from the sample")

    full_raw, fmeta = client.fetch_full_histories(
        api, candidates, max_wallets=args.max_wallets, activity=activity)
    for k, v in fmeta.items():
        log(f"  {k:22} {v:,}")
    if full_raw.empty:
        log("full-history fetch returned nothing; falling back to the sample")
        trades, meta["n_wallets_discovered"] = seed_trades, len(candidates)
    else:
        trades = client.normalise_trades(full_raw, resolved)
        # N for the luck correction is the number of wallets whose PERFORMANCE
        # was examined, not the number discovered.
        meta["n_wallets_discovered"] = fmeta["n_wallets_examined"]
        meta.update(fmeta)

    log(f"\n{len(trades):,} resolved trades | {trades.wallet.nunique():,} wallets")
    log(f"median distinct markets per wallet: "
        f"{trades.groupby('wallet')['market_id'].nunique().median():.0f}")
    meta["mode"] = "live"
    return trades, meta


def analyse(trades: pd.DataFrame, meta: dict, args) -> dict:
    """Score, gate, and test persistence. Returns the report payload."""
    n_scanned = int(meta.get("n_wallets_discovered", trades.wallet.nunique()))

    section("4. SCORING WALLETS")
    scored = wallets.score_wallets(trades, min_trades=args.min_trades,
                                   min_markets=args.min_markets)
    log(f"{len(scored):,} wallets with >= {args.min_trades} fills across "
        f">= {args.min_markets} distinct markets (of {n_scanned:,} examined)")
    log("Trades are aggregated to the MARKET first: a market resolves once, so "
        "42 fills in one market is ONE bet, not 42.")
    if scored.empty:
        return {"verdict": "no wallet had enough resolved trades to score",
                "n_wallets_scanned": n_scanned, "n_scored": 0}

    ranked = wallets.luck_adjusted_ranking(scored, n_wallets_scanned=n_scanned)
    t_needed = float(ranked["t_needed"].iloc[0])
    log(f"t-statistic needed to clear luck at N={n_scanned:,}: {t_needed:.2f}")
    log(f"best t observed: {ranked['t_stat'].max():.2f}")
    log(f"best edge observed: {ranked['edge_per_share'].max() * 100:.1f}c/share")

    cols = ["wallet", "n_trades", "n_eff", "edge_per_share", "roi", "t_stat",
            "clears_luck"]
    log("\ntop 15 by edge:")
    log(ranked.head(15)[cols].to_string(index=False))

    n_clear = int(ranked["clears_luck"].sum())
    log(f"\n>>> wallets clearing the luck bar: {n_clear}")

    section("5. PERSISTENCE (split on RESOLUTION time, not trade time)")
    try:
        pers = wallets.persistence_test(trades, top_frac=args.top_frac)
    except Exception as e:                                   # noqa: BLE001
        pers = {"verdict": f"persistence test failed: {e}"}
    for k, v in pers.items():
        log(f"  {k:26} {v}")

    section("6. BIAS ATTRIBUTION for the top wallets")
    bias = []
    for w in ranked.head(5)["wallet"]:
        try:
            b = wallets.bias_attribution(trades, w)
            bias.append({"wallet": w, "overall_edge": b.get("overall_edge"),
                         "extreme_band_stake": b.get("stake_in_extreme_bands"),
                         "verdict": b.get("verdict")})
            log(f"  {w[:14]}... edge {b.get('overall_edge')} "
                f"extreme {b.get('stake_in_extreme_bands')}")
        except Exception as e:                               # noqa: BLE001
            log(f"  {w[:14]}... bias attribution failed: {e}")

    section("7. COPY ECONOMICS")
    econ = None
    if n_clear:
        edge_c = float(ranked[ranked["clears_luck"]]["edge_per_share"].iloc[0] * 100)
        econ = execution.copy_economics(edge_c)
        for k, v in econ.items():
            log(f"  {k:30} {v}")
    else:
        log("  nothing cleared the luck bar; execution economics are moot")

    gap_t = pers.get("gap_t_stat")
    persists = bool(gap_t is not None and np.isfinite(gap_t) and gap_t > 2.0
                    and pers.get("gap", 0) > 0)
    if n_clear == 0 and not persists:
        verdict = ("NO. No wallet is distinguishable from the luckiest of the "
                   "population, and past performance does not predict future "
                   "performance. There is nothing here safe to copy.")
    elif n_clear and persists:
        verdict = (f"MAYBE. {n_clear} wallet(s) clear the luck bar AND past "
                   f"performance predicts future performance. Check bias "
                   f"attribution before building anything.")
    elif persists:
        verdict = ("WEAK SIGNAL. No individual wallet clears the luck bar, but "
                   "past performance does predict future performance in "
                   "aggregate. Worth a larger scan.")
    else:
        verdict = (f"NO. {n_clear} wallet(s) cleared the luck bar but past "
                   f"performance does NOT predict future performance -- "
                   f"consistent with those wallets having been lucky.")

    return {
        "verdict": verdict,
        "n_wallets_scanned": n_scanned,
        "n_scored": int(len(scored)),
        "t_needed": round(t_needed, 2),
        "best_t": round(float(ranked["t_stat"].max()), 2),
        "best_edge_cents": round(float(ranked["edge_per_share"].max() * 100), 2),
        "n_clearing_luck": n_clear,
        "persistence": pers,
        "bias": bias,
        "economics": econ,
        "ranked": ranked,
    }


def write_report(report: dict, meta: dict, args, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    ranked = report.pop("ranked", None)
    if ranked is not None and len(ranked):
        ranked.head(500).to_csv(out / "wallet_scores.csv", index=False)

    payload = {"generated_at": stamp, "args": vars(args), "meta": meta,
               **{k: v for k, v in report.items()}}
    (out / "scan_result.json").write_text(
        json.dumps(payload, indent=2, default=str))

    lines = [
        f"# Polymarket wallet scan — {stamp}",
        "",
        f"## {report.get('verdict', 'no verdict')}",
        "",
        "| | |",
        "|---|---|",
        f"| mode | {meta.get('mode')} |",
        f"| wallets discovered | {report.get('n_wallets_scanned', 0):,} |",
        f"| wallets scored (>={args.min_trades} fills, >={args.min_markets} markets) | {report.get('n_scored', 0):,} |",
        f"| t-stat needed to clear luck | {report.get('t_needed')} |",
        f"| best t observed | {report.get('best_t')} |",
        f"| best edge observed | {report.get('best_edge_cents')} c/share |",
        f"| **wallets clearing the bar** | **{report.get('n_clearing_luck', 0)}** |",
        "",
        "## Persistence (the decisive test)",
        "",
    ]
    for k, v in (report.get("persistence") or {}).items():
        lines.append(f"- `{k}`: {v}")
    if report.get("economics"):
        lines += ["", "## Copy economics", ""]
        for k, v in report["economics"].items():
            lines.append(f"- `{k}`: {v}")
    lines += ["", "## How to read this", "",
              "A low wallet count clearing the bar is the expected result. The",
              "measured false-positive rate of that gate is ~20% of populations,",
              "so clearing it is necessary, not sufficient — persistence is the",
              "test that matters. See POLYMARKET.md.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    log(f"\nwrote {out}/REPORT.md, scan_result.json, wallet_scores.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", type=int, default=200,
                    help="markets to sample for wallet discovery")
    ap.add_argument("--market-pool", type=int, default=3000,
                    help="resolved markets to fetch before sampling")
    ap.add_argument("--min-volume", type=float, default=5000.0)
    ap.add_argument("--min-trades", type=int, default=20)
    ap.add_argument("--min-markets", type=int, default=10,
                    help="distinct resolved markets required to judge a wallet")
    ap.add_argument("--max-wallets", type=int, default=400,
                    help="wallets to pull full history for; the honest N")
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate-limit", type=float, default=0.25)
    ap.add_argument("--cache", default="/tmp/pm_cache")
    ap.add_argument("--out", default="results/polymarket")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--offline-wallets", type=int, default=1200)
    ap.add_argument("--offline-skilled", type=float, default=0.05)
    ap.add_argument("--offline-edge", type=float, default=0.15)
    args = ap.parse_args()

    out = Path(args.out)
    try:
        trades, meta = collect(args)
        report = analyse(trades, meta, args)
        write_report(report, meta, args, out)
        section("VERDICT")
        log(report["verdict"])
        return 0
    except Exception as e:                                   # noqa: BLE001
        # Never fail silently in CI: write what went wrong where it can be read.
        out.mkdir(parents=True, exist_ok=True)
        tb = traceback.format_exc()
        (out / "REPORT.md").write_text(
            f"# Polymarket scan FAILED — "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"```\n{tb}\n```\n\n"
            f"If this is a field-resolution error, the API shape has drifted: "
            f"add the correct spelling to the candidate lists in "
            f"`src/polymarket/client.py`. The `--offline` run in the same "
            f"workflow tells you whether the analysis code itself still works.\n")
        log(tb)
        return 1


if __name__ == "__main__":
    sys.exit(main())
