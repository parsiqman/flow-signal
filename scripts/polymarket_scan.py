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


# Stage-by-stage market-pool counts. Logs are not retrievable from a finished
# CI run, so anything needed to diagnose a thin pool has to reach the report.
POOL_DIAG: dict = {}


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

    if args.wallets:
        return collect_named_wallets(args)

    section("1. FETCHING RESOLVED MARKETS")
    cfg = client.ClientConfig(cache_dir=args.cache, rate_limit_s=args.rate_limit)
    api = client.PolymarketClient(cfg)

    markets = api.resolved_markets(limit=args.market_pool,
                                   min_volume=args.min_volume)
    log(f"{len(markets):,} resolved markets above ${args.min_volume:,.0f} volume")
    if markets.empty:
        raise RuntimeError("no resolved markets returned; check the Gamma API")
    log(f"field mapping: {client.MARKET_FIELDS}")
    a = markets.attrs
    log(f"  raw markets fetched      : {a.get('n_raw', '?'):,}"
        if isinstance(a.get('n_raw'), int) else f"  raw markets fetched      : ?")
    log(f"  with a winning outcome   : {a.get('n_with_winner', '?')}")
    log(f"  after volume filter      : {a.get('n_after_volume', '?')}")
    log(f"  sample outcomePrices     : {a.get('outcome_price_samples', [])}")
    resolved = markets[markets["winning_index"].notna()]
    log(f"{len(resolved):,} have a determinable winning outcome")
    POOL_DIAG.update({
        "raw_markets_fetched": a.get("n_raw"),
        "with_winning_outcome": a.get("n_with_winner"),
        "after_volume_filter": a.get("n_after_volume"),
        "resolved_pool": len(resolved),
        "outcome_price_samples": a.get("outcome_price_samples", []),
    })
    if len(resolved) < 50:
        log("\n  WARNING: the resolved pool is very thin. Every downstream")
        log("  number is starved by this, not by the wallets. Check the")
        log("  outcomePrices sample above against _markets_to_frame.")
    if resolved.empty:
        raise RuntimeError(
            "no market has a winning_index. outcomePrices is probably not in "
            "the expected shape -- see MARKET_FIELD_CANDIDATES in client.py")

    section("2. DISCOVERING WALLETS BY MARKET PARTICIPATION")
    log("NOT from the leaderboard: that selects on the outcome variable and")
    log("would make every number downstream meaningless.\n")
    if args.stratified:
        log("STRATIFIED by category. A specialist -- someone with genuine")
        log("forecasting skill in one domain -- trades almost only in that")
        log("domain, so a uniform sample catches them once and the market-count")
        log("filter then discards them. Concentrating the sample fixes that,")
        log("and shrinks the multiple-testing penalty at the same time.\n")
        resolved = client.label_categories(resolved)
        counts = resolved["category"].value_counts()
        log(counts.to_string())
        POOL_DIAG["category_pool"] = counts.to_dict()
        log("")
        raw, meta = client.discover_stratified(
            api, resolved, per_category=args.per_category, seed=args.seed,
            categories=args.categories.split(",") if args.categories else None)
    else:
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
        if "category" in resolved.columns:
            trades = trades.merge(
                resolved[["market_id", "category"]].drop_duplicates(),
                on="market_id", how="left")
            trades["category"] = trades["category"].fillna("other")
    else:
        trades = client.normalise_trades(full_raw, resolved)
        if "category" in resolved.columns:
            trades = trades.merge(
                resolved[["market_id", "category"]].drop_duplicates(),
                on="market_id", how="left")
            trades["category"] = trades["category"].fillna("other")
        # N for the luck correction is the number of wallets whose PERFORMANCE
        # was examined, not the number discovered.
        meta["n_wallets_discovered"] = fmeta["n_wallets_examined"]
        meta.update(fmeta)

    log(f"\n{len(trades):,} resolved trades | {trades.wallet.nunique():,} wallets")
    log(f"median distinct markets per wallet: "
        f"{trades.groupby('wallet')['market_id'].nunique().median():.0f}")
    meta["mode"] = "live"
    return trades, meta


def probe_endpoints(args) -> int:
    """
    Hit a list of candidate endpoints and report exactly what each returns.

    Written after three runs were spent guessing which URL carries name-to-
    address pairs. Guessing costs a five-minute CI round trip per attempt and
    tells you nothing when it fails; one probe reports every status code and
    response shape at once. The same principle as the market-pool diagnostics:
    when the remote shape is unknown, measure it rather than assume it.
    """
    import json as _json
    import urllib.error
    import urllib.parse
    import urllib.request

    name = args.probe
    candidates = [
        ("https://lb-api.polymarket.com/leaderboard",
         {"window": "all", "limit": 50, "orderBy": "profit"}),
        ("https://lb-api.polymarket.com/rank", {"window": "all", "limit": 50}),
        ("https://polymarket.com/api/leaderboard", {"window": "all"}),
        ("https://polymarket.com/api/profile/search", {"q": name}),
        ("https://gamma-api.polymarket.com/public-profile", {"name": name}),
        ("https://gamma-api.polymarket.com/public-search", {"q": name}),
        ("https://gamma-api.polymarket.com/search", {"q": name}),
        ("https://data-api.polymarket.com/leaderboard", {"window": "all"}),
        ("https://data-api.polymarket.com/profile", {"name": name}),
        ("https://data-api.polymarket.com/traders", {"limit": 25}),
        ("https://gamma-api.polymarket.com/events", {"limit": 1}),
    ]
    section(f"ENDPOINT PROBE for {name!r}")
    for url, params in candidates:
        full = f"{url}?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(
                full, headers={"User-Agent": "flow-signal-research/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                body = r.read().decode()[:400_000]
            payload = _json.loads(body)
            if isinstance(payload, list):
                shape = f"list[{len(payload)}]"
                sample = payload[0] if payload else {}
            else:
                shape = f"dict keys={sorted(payload)[:10]}"
                sample = payload
            keys = sorted(sample)[:18] if isinstance(sample, dict) else str(sample)[:120]
            log(f"\n  200 {url}")
            log(f"      params {params}")
            log(f"      {shape}")
            log(f"      sample keys: {keys}")
            if name.lower() in body.lower():
                log(f"      >>> CONTAINS {name!r} <<<")
                idx = body.lower().index(name.lower())
                log(f"      context: ...{body[max(0, idx-260):idx+260]}...")
        except urllib.error.HTTPError as e:
            log(f"  {e.code} {url}  params={params}")
        except Exception as e:                               # noqa: BLE001
            log(f"  ERR {url}: {type(e).__name__} {e}")
    return 0


def collect_named_wallets(args) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate specific addresses someone has pointed at, rather than searching.

    The multiple-testing penalty is different here and it matters. When you scan
    10,000 wallets and keep the best, the bar is what the luckiest of 10,000
    produces by chance. When someone names an address for reasons outside this
    dataset -- a tracker, a news story, a domain reputation -- you are testing a
    pre-specified hypothesis and the bar is far lower.

    But not N=1 either, and pretending otherwise is the trap. Whoever surfaced
    that wallet found it by searching too; you have inherited their selection
    without inheriting its size. So the report gives BOTH bars: N=1 for a
    genuinely pre-specified test, and N=1000 as a stand-in for the search that
    plausibly produced the recommendation. An edge that clears only the first is
    consistent with having been handed somebody else's lucky draw.
    """
    section("1. NAMED-WALLET MODE")
    raw_ids = [w.strip() for w in args.wallets.split(",") if w.strip()]
    cfg0 = client.ClientConfig(cache_dir=args.cache, rate_limit_s=args.rate_limit)
    api0 = client.PolymarketClient(cfg0)
    addrs = []
    for ident in raw_ids:
        if ident.startswith("0x") and len(ident) >= 40:
            addrs.append(ident.lower())
            continue
        log(f"resolving username {ident!r}...")
        found = client.resolve_username(api0, ident)
        if found:
            log(f"  -> {found}")
            addrs.extend(found)
        else:
            log(f"  -> NOT RESOLVED. Supply the address from the profile URL "
                f"(polymarket.com/profile/0x...) instead of the display name.")
    if not addrs:
        raise RuntimeError(
            "no wallet addresses to evaluate. Username lookup failed; pass the "
            "0x address from the profile URL with --wallets.")
    log(f"evaluating {len(addrs)} named wallet(s):")
    for a in addrs:
        log(f"  {a}")

    cfg = client.ClientConfig(cache_dir=args.cache, rate_limit_s=args.rate_limit)
    api = client.PolymarketClient(cfg)

    markets = api.resolved_markets(limit=args.market_pool,
                                   min_volume=0.0)
    resolved = markets[markets["winning_index"].notna()]
    log(f"\n{len(resolved):,} resolved markets available to match trades against")
    if resolved.empty:
        raise RuntimeError("no resolved markets; cannot score anything")
    resolved = client.label_categories(resolved)

    frames = []
    for a in addrs:
        raw = api.user_trades(a, limit=20_000)
        log(f"  {a[:12]}... {len(raw):,} raw fills")
        if raw:
            frames.append(pd.DataFrame(raw))
    if not frames:
        raise RuntimeError("no trades returned for any named wallet")

    trades = client.normalise_trades(pd.concat(frames, ignore_index=True),
                                     resolved)
    trades = trades.merge(resolved[["market_id", "category"]].drop_duplicates(),
                          on="market_id", how="left")
    trades["category"] = trades["category"].fillna("other")
    log(f"\n{len(trades):,} fills matched to resolved markets")
    if len(trades):
        log(f"distinct markets: {trades['market_id'].nunique():,}")
        log("\nUnmatched fills are dropped silently by the join, so a low match")
        log("rate here means the market pool is too shallow to judge this wallet,")
        log("NOT that the wallet traded little.")
    return trades, {"n_wallets_discovered": 1, "mode": "named",
                    "named_wallets": addrs,
                    "match_note": "scored only on fills matched to resolved markets"}


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
    if meta.get("mode") == "named":
        bar_pre = wallets.luck_threshold_t(1, 0.95)
        bar_inherited = wallets.luck_threshold_t(1000, 0.95)
        log(f"\n  bar if genuinely pre-specified (N=1)      : {bar_pre:.2f}")
        log(f"  bar if inheriting someone's search (N=1000): {bar_inherited:.2f}")
        log("  An edge clearing only the first may just be somebody else's")
        log("  lucky draw handed to you.")
        ranked["t_needed_pre_specified"] = bar_pre
        ranked["t_needed_inherited_search"] = bar_inherited
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

    section("5b. PER-CATEGORY SCORING (finds specialists a uniform scan misses)")
    by_cat = pd.DataFrame()
    if "category" in trades.columns:
        try:
            by_cat = wallets.score_by_category(
                trades, min_trades=max(10, args.min_trades // 2),
                min_markets=max(5, args.min_markets // 2))
        except Exception as e:                               # noqa: BLE001
            log(f"  per-category scoring failed: {e}")
    if len(by_cat):
        log(f"{len(by_cat)} wallet-category pairs scored")
        cc = ["wallet", "category", "n_trades", "n_markets", "edge_per_share",
              "t_stat", "t_needed", "focus", "clears_luck"]
        log(by_cat.head(15)[cc].to_string(index=False))
        n_cat_clear = int(by_cat["clears_luck"].sum())
        log(f"\n>>> wallet-category pairs clearing their category bar: {n_cat_clear}")
        if n_cat_clear:
            log("\nSPECIALISTS (focus >= 0.6 means most of their activity is here):")
            log(by_cat[by_cat["clears_luck"]][cc].to_string(index=False))
    else:
        log("  no category data or too few wallets per category")

    section("5c. CONCENTRATED EDGE (the shape informed trading takes)")
    log("Large edge over FEW markets. The 10-market minimum elsewhere excludes")
    log("this by construction, and from P&L alone it is indistinguishable from")
    log("luck -- so these are SURFACED for inspection, not scored.\n")
    conc = pd.DataFrame()
    try:
        conc = wallets.concentrated_edge_candidates(trades)
    except Exception as e:                                   # noqa: BLE001
        log(f"  failed: {e}")
    if len(conc):
        log(f"{len(conc)} wallets with >=15c edge over 3-9 markets")
        log(conc.head(20).to_string(index=False))
    else:
        log("  none found")

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
        "n_category_clearing": int(by_cat["clears_luck"].sum()) if len(by_cat) else 0,
        "n_concentrated": int(len(conc)),
        "ranked": ranked,
        "by_category": by_cat,
        "concentrated": conc,
    }


def write_report(report: dict, meta: dict, args, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    ranked = report.pop("ranked", None)
    if ranked is not None and len(ranked):
        ranked.head(500).to_csv(out / "wallet_scores.csv", index=False)
    for key, fname in (("by_category", "category_scores.csv"),
                       ("concentrated", "concentrated_edge.csv")):
        tbl = report.pop(key, None)
        if tbl is not None and len(tbl):
            tbl.head(500).to_csv(out / fname, index=False)

    payload = {"generated_at": stamp, "args": vars(args), "meta": meta,
               "market_pool": POOL_DIAG,
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
        f"| **wallets clearing the bar (uniform)** | **{report.get('n_clearing_luck', 0)}** |",
        f"| **wallet-category pairs clearing (specialists)** | **{report.get('n_category_clearing', 0)}** |",
        f"| concentrated-edge wallets surfaced | {report.get('n_concentrated', 0)} |",
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
    ap.add_argument("--stratified", action="store_true",
                    help="sample markets within categories, to find specialists")
    ap.add_argument("--per-category", type=int, default=120)
    ap.add_argument("--categories", default="",
                    help="comma-separated subset, e.g. weather,econ")
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate-limit", type=float, default=0.25)
    ap.add_argument("--cache", default="/tmp/pm_cache")
    ap.add_argument("--out", default="results/polymarket")
    ap.add_argument("--wallets", default="",
                    help="comma-separated addresses to evaluate directly, "
                         "instead of searching for candidates")
    ap.add_argument("--probe", default="",
                    help="probe candidate endpoints for this username and "
                         "report what each returns; makes no other calls")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--offline-wallets", type=int, default=1200)
    ap.add_argument("--offline-skilled", type=float, default=0.05)
    ap.add_argument("--offline-edge", type=float, default=0.15)
    args = ap.parse_args()

    out = Path(args.out)
    if args.probe:
        return probe_endpoints(args)
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
