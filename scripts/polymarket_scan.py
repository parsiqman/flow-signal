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

from polymarket import book, client, execution, fixtures, longshot, wallets   # noqa: E402


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

    # Date windows, not offsets. The pagination probe measured that Gamma
    # ignores `offset` outright -- five pages at offsets 0..2000 returned the
    # same 500 markets -- so the previous loop re-read page one and the pool
    # arrived at 333 when 3,000 were asked for.
    markets = client.markets_by_windows(api, days_back=args.days_back,
                                        window_days=args.window_days)
    log(f"  crawl                    : {markets.attrs.get('crawl', '?')}")
    log(f"  windows still truncated  : "
        f"{markets.attrs.get('n_windows_truncated', '?')}")
    if args.min_volume:
        markets = markets[markets["volume"].fillna(0) >= args.min_volume]
    markets = markets.reset_index(drop=True)
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
            categories=args.categories.split(",") if args.categories else None,
            max_trades_per_market=args.max_trades_per_market)
    else:
        raw, meta = client.discover_population(api, resolved,
                                               n_markets=args.markets,
                                               seed=args.seed)
    for k, v in meta.items():
        log(f"  {k:24} {v:,}" if isinstance(v, int) else f"  {k:24} {v}")
    if raw.empty:
        raise RuntimeError("no trades returned from any sampled market")

    log("\n" + client.describe_response(raw, "trades"))

    if args.longshot:
        # Stop here. The rule is a claim about MARKET structure, so the right
        # tape is every fill in a sampled set of markets. Fetching full wallet
        # histories would re-weight it toward the markets that active wallets
        # happen to trade, which is a selection on the traders rather than a
        # sample of the venue -- the same mistake as calibrating on the weather
        # specialist's own markets, one step removed.
        seed = client.normalise_trades(raw, resolved)
        if "category" in resolved.columns:
            seed = seed.merge(resolved[["market_id", "category"]].drop_duplicates(),
                              on="market_id", how="left")
            seed["category"] = seed["category"].fillna("other")
        log(f"\n{len(seed):,} fills across {seed['market_id'].nunique():,} "
            f"markets (market sample, no wallet selection)")
        meta["n_wallets_discovered"] = int(seed["wallet"].nunique())
        return seed, meta

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


def measure_books(args) -> int:
    """
    Measure what the book actually costs in the bands the rule trades.

    The walk-forward says 5.27c/share net, but that net was computed against an
    ASSUMED 1c half-spread while break-even sits at 6.27c. The whole result
    therefore rests on a number that was typed rather than measured, in the
    place where that is most dangerous.
    """
    section("ORDER BOOK COST, BY PRICE BAND")
    cfg = client.ClientConfig(cache_dir=None, rate_limit_s=args.rate_limit)
    api = client.PolymarketClient(cfg)

    mkts = book.open_markets(api, min_volume=args.min_volume)
    log(f"{len(mkts):,} open markets above ${args.min_volume:,.0f} volume")
    if not mkts:
        raise RuntimeError("no open markets returned; cannot measure a book")

    books = book.sample_books(api, mkts, max_books=args.max_books)
    log(f"{len(books):,} token books sampled")
    log("Sampled across end-date windows, NOT in volume order: ordering by")
    log("volume measures the tightest books in the venue and calls it the cost")
    log("of trading, which biases the spread down in the flattering direction.")
    if books.empty:
        raise RuntimeError("no books returned; check the CLOB /book shape")
    log(f"one-sided (no bid or no ask): {int(books['one_sided'].sum()):,}")

    costs = book.cost_by_band(books)
    log("\nmeasured cost and depth by band:")
    log(costs.to_string(index=False))

    # A median over a handful of books is not a measurement. The first run put
    # 170 of 194 books in the two near-resolution buckets and left four to six
    # in each band the rule trades.
    thin = costs[costs["n_tokens"] < 30]
    if len(thin):
        log("\n  WARNING: these bands are too thinly sampled to trust:")
        log(thin[["band", "n_tokens", "median_half_spread_cents"]]
            .to_string(index=False))

    # The rule's own bands, with each charged its OWN measured spread rather
    # than one average across bands.
    fitted = {"0.05-0.10": -4.44, "0.10-0.20": -8.81,
              "0.80-0.90": 8.54, "0.90-0.95": 4.30}
    gross = {b: abs(v) for b, v in fitted.items()}
    net = book.edge_after_measured_cost(gross, costs)
    log("\nfitted edge against MEASURED spread, per band:")
    log(net.to_string(index=False) if len(net) else "  no overlap with rule bands")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    costs.to_csv(out / "book_costs.csv", index=False)
    if len(net):
        net.to_csv(out / "edge_after_book.csv", index=False)
    survive = int(net["survives"].sum()) if len(net) else 0
    rule_bands = set(net["band"]) if len(net) else set()
    thin_rule = costs[costs["band"].astype(str).isin(rule_bands)
                      & (costs["n_tokens"] < 30)]
    trustworthy = len(thin_rule) == 0
    lines = ["# Order book cost by band", "",
             f"Sampled {len(books):,} token books across {len(mkts):,} open "
             f"markets.", "",
             f"**{survive} of {len(net)} rule bands keep a positive edge against "
             f"their own measured half-spread.**", "",
             (f"NOT YET TRUSTWORTHY: {len(thin_rule)} of the rule's bands are "
              f"sampled by fewer than 30 books. A median over a handful of "
              f"books is not a measurement."
              if not trustworthy else
              "Every rule band is sampled by at least 30 books."), "",
             costs.to_markdown(index=False), ""]
    if len(net):
        lines += ["## Rule bands, net of measured cost", "",
                  net.to_markdown(index=False), ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    section("VERDICT")
    if not trustworthy:
        log(f"{survive} of {len(net)} rule bands survive their measured spread, "
            f"BUT {len(thin_rule)} of those bands rest on fewer than 30 books. "
            f"Treat as preliminary.")
    else:
        log(f"{survive} of {len(net)} rule bands survive their measured spread")
    return 0


def probe_pagination(args) -> int:
    """
    Measure how deep the Gamma market listing actually paginates.

    The longshot calibration reached 333 markets of a requested 3,000 because
    `resolved_markets` fetched exactly one page and stopped. Which of the three
    plausible causes it is -- an offset ceiling, a page-size cap, or a filter
    that empties out -- changes the fix completely, and guessing between them
    costs a five-minute CI round trip per attempt while telling you nothing
    when it fails. So: measure it.

    Reports, for each strategy, how many UNIQUE markets it can actually reach.
    Uniqueness is the number that matters: a strategy that happily returns
    10,000 rows which are the same 500 markets over and over is worse than one
    that stops honestly at 500.
    """
    section("PAGINATION PROBE")
    cfg = client.ClientConfig(cache_dir=None, rate_limit_s=args.rate_limit)
    api = client.PolymarketClient(cfg)
    base = {"closed": "true", "order": "endDate", "ascending": "false"}

    log("A. offset ceiling, at page size 500 and 100")
    for size in (500, 100):
        for off in (0, size, size * 4, size * 10, size * 20, 5000, 10000):
            try:
                b = api._get(f"{client.GAMMA}/markets",
                             {**base, "limit": size, "offset": off})
                n = len(b) if isinstance(b, list) else -1
                ids = {str(r.get("conditionId", "")) for r in b} if n > 0 else set()
                log(f"  limit={size:<4} offset={off:<6} -> {n:>4} rows, "
                    f"{len(ids):>4} distinct")
            except Exception as e:                           # noqa: BLE001
                log(f"  limit={size:<4} offset={off:<6} -> {type(e).__name__}: "
                    f"{str(e)[:90]}")

    log("\nB. does paging actually advance, or repeat the same markets?")
    seen, pages = set(), 0
    for off in range(0, 3000, 500):
        try:
            b = api._get(f"{client.GAMMA}/markets",
                         {**base, "limit": 500, "offset": off})
        except Exception as e:                               # noqa: BLE001
            log(f"  stopped at offset {off}: {type(e).__name__} {str(e)[:80]}")
            break
        if not b:
            log(f"  empty page at offset {off}")
            break
        pages += 1
        seen |= {str(r.get("conditionId", "")) for r in b}
        log(f"  after offset {off:<5}: {len(seen):,} distinct markets so far")

    log("\nC. time-windowed slices (sidesteps any offset ceiling entirely)")
    import datetime as _dt
    for months in (1, 3, 6):
        lo = (_dt.datetime.now(timezone.utc)
              - _dt.timedelta(days=30 * months)).strftime("%Y-%m-%d")
        hi = (_dt.datetime.now(timezone.utc)
              - _dt.timedelta(days=30 * (months - 1))).strftime("%Y-%m-%d")
        for lo_key, hi_key in (("end_date_min", "end_date_max"),
                               ("start_date_min", "start_date_max")):
            try:
                b = api._get(f"{client.GAMMA}/markets",
                             {**base, "limit": 500, lo_key: lo, hi_key: hi})
                n = len(b) if isinstance(b, list) else -1
                log(f"  {lo_key}={lo} {hi_key}={hi} -> {n} rows")
            except Exception as e:                           # noqa: BLE001
                log(f"  {lo_key}={lo} -> {type(e).__name__}: {str(e)[:70]}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "REPORT.md").write_text(
        "# Pagination probe\n\nSee the job log for the full table. Reachable "
        f"distinct markets by naive offset paging: **{len(seen):,}** over "
        f"{pages} pages.\n")
    return 0


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
        # public-search answered 200 on the first probe but returned only a
        # pagination key, so the profile results sit behind a parameter we did
        # not send. lb-api/rank answered 400, meaning the path exists and the
        # arguments are wrong. Both are worth pushing on rather than replacing.
        ("https://gamma-api.polymarket.com/public-search",
         {"q": name, "limit_per_type": 10}),
        ("https://gamma-api.polymarket.com/public-search",
         {"q": name, "type": "profile"}),
        ("https://gamma-api.polymarket.com/public-search",
         {"q": name, "search_profiles": "true"}),
        ("https://gamma-api.polymarket.com/public-search",
         {"q": name, "events_status": "all", "limit_per_type": 20,
          "search_profiles": "true"}),
        ("https://lb-api.polymarket.com/rank",
         {"window": "all", "limit": 50, "rankType": "pnl"}),
        ("https://lb-api.polymarket.com/rank",
         {"window": "all", "limit": 50, "orderBy": "pnl", "category": "weather"}),
        ("https://lb-api.polymarket.com/rank",
         {"p": 1, "window": "all", "type": "pnl"}),
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
            # Keys alone were not enough last time: an endpoint answered 200
            # with a single pagination key and the useful part was elsewhere.
            log(f"      body[:900]: {body[:900]}")
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

    # Fetch the trades FIRST, then resolve exactly the markets they touched.
    # Matching a specialist's fills against a generic pool of recent markets
    # fails completely -- the first named run scored zero trades that way.
    frames = []
    for a in addrs:
        raw = api.user_trades(a, limit=20_000)
        log(f"  {a[:12]}... {len(raw):,} raw fills")
        if raw:
            frames.append(pd.DataFrame(raw))
    if not frames:
        raise RuntimeError("no trades returned for any named wallet")
    raw_all = pd.concat(frames, ignore_index=True)

    fields = client.resolve_fields(raw_all, client.TRADE_FIELD_CANDIDATES,
                                   ("market_id",), "named-wallet trades")
    cond_ids = raw_all[fields["market_id"]].astype(str).unique().tolist()

    # What the wallet actually trades, straight from the fills. This is the
    # cross-check on the market lookup: if these say "highest temperature in
    # NYC" and the resolved markets come back filed under politics, the join is
    # wrong, and without this line that shows up only as an empty result.
    title_col = next((c for c in ("title", "slug", "eventSlug", "question")
                      if c in raw_all.columns), None)
    if title_col:
        log("\nwhat this wallet trades (from the fills themselves):")
        for t in raw_all[title_col].dropna().astype(str).unique()[:6]:
            log(f"  - {t[:90]}")

    log(f"\n{len(cond_ids):,} distinct markets touched; resolving them by id")
    resolved = client.markets_by_condition_ids(api, cond_ids)
    log(f"  lookup form used         : {resolved.attrs.get('lookup_form', '?')}")
    log(f"  ids requested            : {resolved.attrs.get('n_requested', 0):,}")
    log(f"  answered by gamma batch  : {resolved.attrs.get('n_from_gamma', 0):,}")
    log(f"  answered per-id by clob  : {resolved.attrs.get('n_from_clob', 0):,}")
    log(f"  markets matched by id    : {len(resolved):,}")
    resolved = resolved[resolved["winning_index"].notna()]
    log(f"  with a determinable outcome: {len(resolved):,}")
    if resolved.empty:
        raise RuntimeError(
            "none of this wallet's markets could be resolved by condition id; "
            "check the markets-by-id lookup in client.markets_by_condition_ids")
    resolved = client.label_categories(resolved)
    log("\ncategories:")
    log(resolved["category"].value_counts().to_string())

    trades = client.normalise_trades(raw_all, resolved)
    trades = trades.merge(resolved[["market_id", "category"]].drop_duplicates(),
                          on="market_id", how="left")
    trades["category"] = trades["category"].fillna("other")
    log(f"\n{len(trades):,} fills matched to resolved markets "
        f"({len(trades) / max(len(raw_all), 1):.0%} of raw)")
    if trades.empty:
        raise RuntimeError(
            f"{len(raw_all):,} fills, {len(resolved):,} resolved markets, and "
            f"zero of them joined. The condition-id lookup returned markets "
            f"this wallet never traded. Do NOT read this as 'no edge'.")
    if len(trades):
        log(f"distinct markets: {trades['market_id'].nunique():,}")
        log("\nUnmatched fills are dropped silently by the join, so a low match")
        log("rate here means the market pool is too shallow to judge this wallet,")
        log("NOT that the wallet traded little.")
    coverage = len(trades) / max(len(raw_all), 1)
    if coverage < 0.5:
        log(f"\n  WARNING: only {coverage:.0%} of fills were matched. Anything")
        log("  computed below describes that subset, not this wallet.")
    return trades, {"n_wallets_discovered": 1, "mode": "named",
                    "named_wallets": addrs,
                    "n_raw_fills": int(len(raw_all)),
                    "n_markets_touched": int(len(cond_ids)),
                    "n_markets_resolved": int(len(resolved)),
                    "fill_coverage": float(coverage),
                    "match_note": "scored only on fills matched to resolved markets"}


def analyse_longshot(trades: pd.DataFrame, meta: dict, args) -> dict:
    """
    Fit and walk-forward the favourite-longshot rule on the whole market tape.

    Deliberately NOT run on one wallet's markets. That trader picked which
    markets to be in, so his tape is a selected sample of exactly the places he
    thought the bias was worth taking -- calibrating on it would measure his
    selection as if it were the market's structure. The population tape is the
    honest sample, and it is also the one a rule would actually trade.
    """
    section("LONGSHOT RULE: calibration on the market tape")
    log("The counterparty is named: people overpay for lottery-shaped payoffs.")
    log("That is a preference, not a mispricing arbitrage will close -- which")
    log("is why it persists, and why capacity is small and recreational.\n")

    cal = longshot.calibrate(trades)
    log("full-sample calibration (NOT the result -- shown for shape only):")
    log(cal.to_string(index=False))

    section("WALK-FORWARD (fit on early markets, evaluate on later ones)")
    wf = longshot.walk_forward(trades, half_spread_cents=args.half_spread_cents)
    if "out_of_sample" not in wf:
        return {"verdict": f"INCONCLUSIVE. {wf.get('verdict', 'no split')}",
                "longshot": wf, "n_wallets_scanned": 0, "n_scored": 0}

    log(f"  split at                 : {wf['split_ts']}")
    log(f"  markets train / test     : {wf['n_markets_train']:,} / "
        f"{wf['n_markets_test']:,}")
    log(f"  bands examined           : {wf['n_bands_tested']}")
    log(f"  bar after correction     : {wf['bar_used']}")
    log(f"  rule fitted              : {wf['rule']}")
    log("\ntrain calibration:")
    log(wf["calibration_train"].to_string(index=False))
    log("\ntest calibration (the rule never saw this):")
    log(wf["calibration_test"].to_string(index=False))

    oos = wf["out_of_sample"]
    log("\nOUT OF SAMPLE, net of the spread:")
    for k, v in oos.items():
        log(f"  {k:30} {v}")

    section("NULL CHECK (would honest prices produce this by chance?)")
    rule = longshot.fit(trades[trades["market_id"].isin(
        trades.groupby("market_id")["resolved_at"].min()
        .sort_values().index[:wf["n_markets_train"]])])
    nul = longshot.null_check(rule, trades, n_draws=120,
                              half_spread_cents=args.half_spread_cents)
    for k, v in nul.items():
        log(f"  {k:30} {v}")

    section("POWER: could this sample have SEEN the effect?")
    log("Asked before reading the result, not after. A band with an effective")
    log("sample of 30 cannot resolve anything under ~20 cents, and the bias")
    log("being hunted is 2-8 cents. Such a band reporting 'no edge' has")
    log("reported nothing.\n")
    pw = longshot.power_verdict(wf["calibration_train"], wf["bar_used"])
    for k, v in pw.items():
        log(f"  {k:32} {v}")
    log("\nminimum detectable edge by band (test period):")
    log(longshot.minimum_detectable_edge(
        wf["calibration_test"], wf["bar_used"]).to_string(index=False))

    section("LOSS CORRELATION (the parameter the capital model turns on)")
    log("A book that is long favourites everywhere is ONE bet wearing many")
    log("hats. Ruin arrives near an intraclass correlation of 0.07, so the")
    log("breadth argument -- thousands of bets, therefore diversified -- lives")
    log("or dies on this number.\n")
    corr = longshot.loss_correlation(rule, trades)
    for k, v in corr.items():
        log(f"  {k:32} {v}")

    log("\nAcross time scales, as a robustness check. Day-level ICC is NOT")
    log("blind to a slow regime factor -- a monthly swing raises between-day")
    log("variance too -- so the scales should agree. Disagreement means the")
    log("grouping is wrong, not that the risk is hidden.")
    multi = longshot.loss_correlation_multiscale(rule, trades)
    for name, r in multi.items():
        if name == "trend":
            continue
        log(f"  {name:9} icc={r.get('icc')} buckets={r.get('n_buckets')} "
            f"avg_size={r.get('avg_bucket_size')}")
    for k, v in (multi.get("trend") or {}).items():
        log(f"  trend.{k:26} {v}")

    net = oos.get("net_edge_cents", 0.0)
    t = oos.get("t_stat_net", 0.0)
    if rule.is_empty() and pw.get("underpowered"):
        verdict = (f"UNDERPOWERED, NOT NEGATIVE. No band cleared the bar, but "
                   f"the typical band could not have resolved an edge below "
                   f"{pw['median_mde_cents']}c and the effect being hunted is "
                   f"2-8c. This sample "
                   f"({wf['n_markets_train']:,}+{wf['n_markets_test']:,} "
                   f"markets) did not test the question. Deepen the market "
                   f"pool before drawing any conclusion.")
    elif rule.is_empty():
        verdict = ("NO. No price band shows a bias that survives the "
                   "multiple-testing bar, on a sample large enough to have "
                   "seen one. There is no rule here to trade.")
    elif net > 0 and t and t > 2.0 and nul.get("p_value", 1.0) < 0.05:
        verdict = (f"YES, MEASURABLY. The rule pays {net:.2f}c/share net of a "
                   f"{oos['cost_cents']:.2f}c spread out of sample (t={t}), and "
                   f"exceeds what honest prices produce by chance. Break-even "
                   f"half-spread is {oos['breakeven_half_spread_cents']:.2f}c "
                   f"-- check that against a real book before sizing anything.")
    elif net > 0:
        verdict = (f"WEAK. The rule pays {net:.2f}c/share out of sample but at "
                   f"t={t}, which is not distinguishable from luck at this "
                   f"sample size.")
    else:
        verdict = (f"NO. The bias is visible but the spread eats it: "
                   f"{oos.get('gross_edge_cents', 0):.2f}c gross against "
                   f"{oos.get('cost_cents', 0):.2f}c of cost.")

    if corr.get("icc") is not None and corr["icc"] > corr["ruin_threshold_icc"]:
        verdict = (f"{verdict} RISK: measured loss correlation "
                   f"{corr['icc']:.3f} exceeds the ~0.07 that wipes out the "
                   f"capital base; {corr['n_markets']:,} markets behave like "
                   f"{corr['effective_independent_bets']:.0f} independent bets.")

    tr = multi.get("trend") or {}
    if tr.get("scales_disagree"):
        verdict = (f"{verdict} CAUTION: correlation scales disagree "
                   f"({tr['max_icc']:.3f} at {tr['max_at_scale']} scale); check "
                   f"the resolution grouping before trusting either number.")

    return {"verdict": verdict, "longshot": wf, "null": nul, "power": pw,
            "loss_correlation": corr, "correlation_by_scale": multi,
            "n_wallets_scanned": int(meta.get("n_wallets_discovered", 0)),
            "n_scored": 0}


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

    # The cross-sectional test needs a population to rank. With one named
    # wallet it has nothing to work with and says so -- and the verdict logic
    # below used to read that non-answer as a failed test, which is how a
    # t-statistic of 10.19 got filed under "consistent with having been lucky".
    # A single account still has an out-of-sample test: its own two halves.
    oos = {}
    if len(scored) <= 3:
        section("5a. WITHIN-WALLET SPLIT (the test one account can support)")
        for w in scored["wallet"]:
            try:
                r = wallets.wallet_out_of_sample(trades, w)
            except Exception as e:                           # noqa: BLE001
                r = {"verdict": f"within-wallet split failed: {e}"}
            oos[w] = r
            log(f"  {w[:14]}...")
            for k, v in r.items():
                log(f"    {k:22} {v}")

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

    section("6b. TRADE STYLE (can this edge be copied AT ALL?)")
    log("Edge size says whether the edge is real. It says nothing about")
    log("whether any of it is available to a copier, and the two are")
    log("independent -- a latency bot and a forecaster can post the same edge")
    log("per share with opposite answers.\n")
    style = []
    for w in ranked.head(5)["wallet"]:
        try:
            s = wallets.style_attribution(trades, w)
            style.append({"wallet": w, **s})
            log(f"  {w[:14]}... {s['style']}")
            log(f"    both sides {s['both_sides_frac']} | "
                f"fills/market {s['median_fills_per_market']} | "
                f"span {s['median_span_hours']}h")
            log(f"    {s['copyable']}")
        except Exception as e:                               # noqa: BLE001
            log(f"  {w[:14]}... style attribution failed: {e}")

    section("7. COPY ECONOMICS")
    econ = None
    if n_clear:
        edge_c = float(ranked[ranked["clears_luck"]]["edge_per_share"].iloc[0] * 100)
        econ = execution.copy_economics(edge_c)
        for k, v in econ.items():
            log(f"  {k:30} {v}")
    else:
        log("  nothing cleared the luck bar; execution economics are moot")

    # The luck gate assumes DIRECTIONAL, hold-to-resolution betting: it asks
    # whether a wallet's per-market outcome edge could have come from chance,
    # and floors the variance at the binomial bound for a bettor taking a view.
    # A market maker is delta-neutral -- its per-market P&L is genuinely
    # low-variance because it is not betting on the outcome at all -- so the
    # floor imposes a directional bettor's uncertainty on a book that has none,
    # and the t-statistic comes out small no matter how much money was made.
    # Reporting that as "does not clear the luck bar" is a category error, and
    # the run that prompted this said exactly that about a wallet the same
    # report showed making $31,910.
    mm = [st for st in style if st.get("style") == "market maker / latency"]
    top_is_mm = bool(style and style[0].get("style") == "market maker / latency")

    gap_t = pers.get("gap_t_stat")
    persists = bool(gap_t is not None and np.isfinite(gap_t) and gap_t > 2.0
                    and pers.get("gap", 0) > 0)
    # Distinguish "the test ran and failed" from "the test could not run".
    # Only the first is evidence, and conflating them manufactures a negative.
    ran = gap_t is not None and np.isfinite(gap_t)
    if not ran:
        halves_ok = [r for r in oos.values()
                     if r.get("early", {}).get("edge_per_share", 0) > 0
                     and r.get("late", {}).get("edge_per_share", 0) > 0]
        if n_clear and halves_ok:
            verdict = (f"MAYBE. {n_clear} wallet(s) clear the luck bar, and the "
                       f"edge is positive in both halves of their own history. "
                       f"The cross-sectional persistence test could NOT be run "
                       f"(too few wallets) -- this is weaker evidence than it "
                       f"looks. Check bias attribution before building anything.")
        elif n_clear:
            verdict = (f"INCONCLUSIVE. {n_clear} wallet(s) clear the luck bar, "
                       f"but no out-of-sample test could be run on this sample. "
                       f"Nothing here has been shown to repeat.")
        else:
            verdict = ("NO. Nothing clears the luck bar, and no out-of-sample "
                       "test could be run on this sample.")
    elif n_clear == 0 and not persists:
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

    if top_is_mm:
        prof = float(ranked["total_profit"].iloc[0]) if "total_profit" in ranked else 0.0
        verdict = (f"UNCOPYABLE BY STYLE. The top wallet is a market maker "
                   f"({style[0]['both_sides_frac']:.0%} of its markets traded "
                   f"on both sides, median {style[0]['median_fills_per_market']:.0f} "
                   f"fills per market, {style[0]['median_span_hours']:.1f}h "
                   f"median span). It made ${prof:,.0f} in this sample and that "
                   f"is real -- but the edge is in being first to the book, so "
                   f"a copier is the slower side of it, not a participant in "
                   f"it. The luck gate below assumes directional "
                   f"hold-to-resolution betting and does NOT apply to a "
                   f"delta-neutral book; do not read its t-statistic as a "
                   f"judgement on this wallet. (Gate said: {verdict})")

    # A verdict is only as good as the share of the record it saw. Scoring this
    # account on 13 of its 1,831 markets produced a confident "NO" off n_eff
    # 4.7 -- an answer about the data pipeline wearing the clothes of an answer
    # about the trader. Below half coverage, say so instead of concluding.
    cov = meta.get("fill_coverage")
    if cov is not None and cov < 0.5:
        verdict = (f"INCONCLUSIVE. Only {cov:.0%} of this wallet's fills could "
                   f"be matched to a resolved market, so the sample scored is "
                   f"not its record. Fix coverage before reading any verdict. "
                   f"(Underlying result on the matched subset: {verdict})")

    return {
        "verdict": verdict,
        "fill_coverage": cov,
        "n_wallets_scanned": n_scanned,
        "n_scored": int(len(scored)),
        "t_needed": round(t_needed, 2),
        "best_t": round(float(ranked["t_stat"].max()), 2),
        "best_edge_cents": round(float(ranked["edge_per_share"].max() * 100), 2),
        "n_clearing_luck": n_clear,
        "persistence": pers,
        "within_wallet_split": oos,
        "bias": bias,
        "style": style,
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
    for w, r in (report.get("within_wallet_split") or {}).items():
        lines += ["", f"### Within-wallet split — `{w}`", ""]
        for k, v in r.items():
            lines.append(f"- `{k}`: {v}")
    for b in (report.get("bias") or []):
        lines += ["", f"### Bias attribution — `{b.get('wallet', '')}`", ""]
        for k, v in b.items():
            if k != "wallet":
                lines.append(f"- `{k}`: {v}")
    for st in (report.get("style") or []):
        lines += ["", f"### Trade style — `{st.get('wallet', '')}`", ""]
        for k, v in st.items():
            if k != "wallet":
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
    ap.add_argument("--max-trades-per-market", type=int, default=5000,
                    help="cap fills fetched per market. For the longshot "
                         "calibration the sample size is MARKETS, not fills -- "
                         "each market contributes one observation per band "
                         "however many times it traded -- so a low cap buys "
                         "several times the market coverage for the same "
                         "number of requests, which is what the power gate "
                         "actually needs.")
    ap.add_argument("--days-back", type=int, default=730,
                    help="how far back the date-windowed market crawl reaches")
    ap.add_argument("--window-days", type=int, default=14,
                    help="width of each crawl window before adaptive splitting")
    ap.add_argument("--books", action="store_true",
                    help="measure live order-book spread and depth by band")
    ap.add_argument("--max-books", type=int, default=400)
    ap.add_argument("--probe-pagination", action="store_true",
                    help="measure how deep the market listing paginates")
    ap.add_argument("--longshot", action="store_true",
                    help="fit and walk-forward the favourite-longshot rule on "
                         "the market tape instead of scoring wallets")
    ap.add_argument("--half-spread-cents", type=float, default=1.0,
                    help="cents per share given up crossing to get filled")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--offline-wallets", type=int, default=1200)
    ap.add_argument("--offline-skilled", type=float, default=0.05)
    ap.add_argument("--offline-edge", type=float, default=0.15)
    args = ap.parse_args()

    out = Path(args.out)
    if args.books:
        return measure_books(args)
    if args.probe_pagination:
        return probe_pagination(args)
    if args.probe:
        return probe_endpoints(args)
    try:
        trades, meta = collect(args)
        if args.longshot:
            report = analyse_longshot(trades, meta, args)
            write_report(report, meta, args, out)
            section("VERDICT")
            log(report["verdict"])
            return 0
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
