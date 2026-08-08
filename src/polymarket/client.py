"""
Polymarket Global API client, and an unbiased way to enumerate candidate wallets.

Two public, keyless APIs are used:
  - Gamma  (gamma-api.polymarket.com)  : markets, including resolution outcomes
  - Data   (data-api.polymarket.com)   : trades, by market or by user

Neither is reachable from the Claude Code sandbox -- every Polymarket host
answers 403 to CONNECT under this environment's egress policy. The client is
therefore written and tested here against recorded response shapes, and the
crawl runs in Colab.

---

**The single most important design decision in this file is how wallets are
discovered, and it is easy to get catastrophically wrong.**

Polymarket publishes a public leaderboard. Seeding candidates from it would be
selecting on the outcome variable: you would rank traders by past profit having
already filtered to traders with past profit, and then "discover" that past
profit predicts past profit. Every number downstream would be meaningless, and
nothing about the analysis would look broken.

So wallets are enumerated by **market participation**: sample resolved markets,
take every wallet that traded in them, and keep the lot regardless of how they
did. That is an unbiased population, and its size is the honest `n_wallets`
for the luck correction in `wallets.luck_threshold_t`.

`discover_by_leaderboard` exists but raises. It is there so the trap is
documented in code rather than in someone's memory.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

class PaginationLimit(RuntimeError):
    """The API refused a page: end of available data, not a failure."""


GAMMA = "https://gamma-api.polymarket.com"
DATA = "https://data-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

# Public API field names, as CANDIDATE LISTS rather than single guesses.
#
# The shapes below are documented, but they drift, and this code could not be
# tested against the live API from the sandbox it was written in -- every
# Polymarket host is blocked there. A single hardcoded name that turns out to be
# wrong produces a KeyError at best and a NaN column at worst. Accepting several
# plausible spellings and reporting which one matched turns an unattended crash
# into a line in the log.
TRADE_FIELD_CANDIDATES = {
    "wallet": ["proxyWallet", "proxy_wallet", "user", "userAddress", "maker",
               "owner", "account", "wallet"],
    "market_id": ["conditionId", "condition_id", "market", "marketId",
                  "market_id"],
    "timestamp": ["timestamp", "matchTime", "match_time", "time", "createdAt",
                  "created_at"],
    "price": ["price", "matchedPrice", "avgPrice"],
    "size": ["size", "shares", "amount", "quantity", "matchedAmount"],
    "side": ["side", "direction", "takerSide"],
    "outcome_index": ["outcomeIndex", "outcome_index", "outcomeIdx",
                      "outcome_idx", "asset_index"],
}
MARKET_FIELD_CANDIDATES = {
    "market_id": ["conditionId", "condition_id", "market_id", "id"],
    "question": ["question", "title", "slug"],
    "category": ["category", "categories", "tags", "eventCategory", "groupSlug"],
    "slug": ["slug", "marketSlug", "ticker"],
    "end_date": ["endDate", "end_date", "endDateIso", "closedTime",
                 "resolutionTime"],
    "outcome_prices": ["outcomePrices", "outcome_prices", "prices"],
    "volume": ["volumeNum", "volume", "volumeClob", "volume_num"],
}

# Filled in by `resolve_fields` on first successful fetch, so downstream code
# and the logs both show what was actually matched.
TRADE_FIELDS: dict[str, str] = {}
MARKET_FIELDS: dict[str, str] = {}


def resolve_fields(df: pd.DataFrame, candidates: dict[str, list[str]],
                   required: tuple[str, ...], label: str) -> dict[str, str]:
    """
    Match canonical names onto whichever spelling this response actually uses.

    Comparison is case-insensitive because vendors are inconsistent about it.
    Raises with the full column list when a REQUIRED field cannot be found --
    the one situation where guessing would be worse than stopping.
    """
    lower = {str(c).lower(): str(c) for c in df.columns}
    found: dict[str, str] = {}
    for canonical, options in candidates.items():
        for opt in options:
            if opt.lower() in lower:
                found[canonical] = lower[opt.lower()]
                break
    missing = [f for f in required if f not in found]
    if missing:
        raise ValueError(
            f"{label}: could not find {missing} in the response.\n"
            f"Tried: {({m: candidates[m] for m in missing})}\n"
            f"Response columns: {sorted(str(c) for c in df.columns)}\n"
            f"Add the correct spelling to the candidate list in client.py.")
    return found


@dataclass
class ClientConfig:
    rate_limit_s: float = 0.25        # be a good citizen; the API is free
    max_retries: int = 4
    timeout_s: float = 30.0
    page_size: int = 500
    cache_dir: Path | None = None     # disk cache makes the crawl resumable


class PolymarketClient:
    """
    Thin, cached, rate-limited client.

    Caching is not an optimisation here, it is a correctness feature: a crawl
    over thousands of markets will be interrupted, and re-fetching from scratch
    each time both hammers a free API and risks a different slice of data on
    every run, which would make results irreproducible.
    """

    def __init__(self, cfg: ClientConfig | None = None):
        self.cfg = cfg or ClientConfig()
        self._last_call = 0.0
        if self.cfg.cache_dir:
            Path(self.cfg.cache_dir).mkdir(parents=True, exist_ok=True)

    # --- plumbing --------------------------------------------------------
    def _cache_path(self, key: str) -> Path | None:
        if not self.cfg.cache_dir:
            return None
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)[:180]
        return Path(self.cfg.cache_dir) / f"{safe}.json"

    def _get(self, url: str, params: dict) -> Any:
        key = url.replace("https://", "") + "?" + "&".join(
            f"{k}={v}" for k, v in sorted(params.items()))
        cp = self._cache_path(key)
        if cp and cp.exists():
            return json.loads(cp.read_text())

        import urllib.error
        import urllib.parse
        import urllib.request

        wait = self.cfg.rate_limit_s - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)

        # doseq=True so a list value becomes a repeated parameter
        # (?id=a&id=b) rather than the urlencoded repr of a Python list.
        # Gamma's array filters only accept the repeated form.
        full = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
        last_err = None
        for attempt in range(self.cfg.max_retries):
            try:
                req = urllib.request.Request(
                    full, headers={"User-Agent": "flow-signal-research/1.0"})
                with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as r:
                    payload = json.loads(r.read().decode())
                self._last_call = time.time()
                if cp:
                    cp.write_text(json.dumps(payload))
                return payload
            except urllib.error.HTTPError as e:
                # A 4xx is the server saying the request itself is wrong, so
                # retrying it four times only wastes the rate limit and then
                # crashes anyway. Gamma answers 422 past its pagination ceiling,
                # which is a normal end-of-data signal rather than a fault --
                # it took down a whole run before this distinction existed.
                if 400 <= e.code < 500 and e.code != 429:
                    raise PaginationLimit(
                        f"{e.code} on {full}; treating as end of data") from e
                last_err = e
                time.sleep(2 ** attempt)
            except Exception as e:                      # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"GET failed after {self.cfg.max_retries} tries: "
                           f"{full}\n{last_err}")

    # --- endpoints -------------------------------------------------------
    def resolved_markets(self, limit: int = 2000,
                         min_volume: float = 5_000.0) -> pd.DataFrame:
        """
        Closed markets with a known outcome.

        `min_volume` filters out markets too thin for any trade to be
        meaningful. Note this is a filter on the MARKET, not on traders, so it
        does not bias the wallet population -- every wallet active in a
        surviving market is still kept regardless of performance.
        """
        rows, offset, empty_pages = [], 0, 0
        # Page well past `limit`: most closed markets carry no determinable
        # outcome, so the resolved pool is a small fraction of what is fetched.
        hard_cap = max(limit * 8, 8000)
        while len(rows) < hard_cap:
            try:
                batch = self._get(f"{GAMMA}/markets", {
                    "closed": "true", "limit": self.cfg.page_size,
                    "offset": offset, "order": "endDate", "ascending": "false"})
            except PaginationLimit:
                break            # hit the API's offset ceiling; use what we have
            if not batch:
                empty_pages += 1
                if empty_pages >= 2:
                    break
            else:
                empty_pages = 0
                rows.extend(batch)
            offset += self.cfg.page_size
        df = _markets_to_frame(rows)
        attrs = dict(df.attrs)
        if min_volume:
            df = df[df["volume"].fillna(0) >= min_volume]
        out = df.head(limit).reset_index(drop=True)
        out.attrs.update(attrs)
        out.attrs["n_after_volume"] = len(df)
        return out

    def _paged_trades(self, params: dict, limit: int) -> list[dict]:
        """Page a /trades query, stopping cleanly at the API's own ceiling."""
        out, offset = [], 0
        while len(out) < limit:
            try:
                batch = self._get(f"{DATA}/trades",
                                  {**params, "limit": self.cfg.page_size,
                                   "offset": offset, "takerOnly": "false"})
            except PaginationLimit:
                break
            if not batch:
                break
            out.extend(batch)
            offset += self.cfg.page_size
            if len(batch) < self.cfg.page_size:
                break
        return out[:limit]

    def market_trades(self, market_id: str, limit: int = 5000) -> list[dict]:
        """Every trade in one market. This is the unbiased discovery unit."""
        return self._paged_trades({"market": market_id}, limit)

    def user_trades(self, wallet: str, limit: int = 5000) -> list[dict]:
        """Full history for one wallet, across all markets."""
        return self._paged_trades({"user": wallet}, limit)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_by_leaderboard(*args, **kwargs):
    """
    Do not do this. Kept as a named, raising function so the trap is recorded.

    Seeding candidate wallets from Polymarket's public leaderboard selects on
    the outcome variable. You would rank traders by past profit within a set
    already filtered for past profit, then conclude that past profit predicts
    past profit. The resulting numbers would look excellent and mean nothing,
    and no part of the pipeline would appear to fail.
    """
    raise NotImplementedError(
        "Leaderboard seeding selects on the outcome variable and invalidates "
        "every downstream statistic. Use discover_population(), which "
        "enumerates wallets by market participation instead.")


# Keyword rules for bucketing markets when the API gives no usable category.
# Deliberately crude: this only has to be good enough to concentrate a sample,
# and a wallet's measured specialisation is verified against its actual trades
# afterwards rather than trusted from the label.
CATEGORY_RULES = {
    "weather": ["temperature", "weather", "rain", "snow", "hurricane", "storm",
                "degrees", "celsius", "fahrenheit", "climate", "tornado",
                "high temp", "low temp", "heat", "wildfire", "drought", "flood",
                "el nino", "la nina", "noaa", "cyclone", "typhoon", "blizzard"],
    "sports": ["nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball",
               "tennis", "ufc", "boxing", "premier league", "champions league",
               "world cup", "olympics", "golf", "cricket", "baseball", "f1",
               "formula 1", "super bowl", "playoffs", "vs.", " vs ", "beat the",
               "win the game", "match", "series", "grand slam", "heavyweight"],
    "politics": ["election", "president", "senate", "congress", "governor",
                 "parliament", "prime minister", "nominee", "primary", "vote",
                 "cabinet", "impeach", "supreme court"],
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "solana", "crypto", "token",
               "coinbase", "binance", "stablecoin", "etf"],
    "econ": ["fed", "inflation", "cpi", "gdp", "rate cut", "rate hike",
             "unemployment", "recession", "jobs report", "fomc"],
    "entertainment": ["oscar", "grammy", "emmy", "box office", "album",
                      "rotten tomatoes", "netflix", "movie", "billboard"],
}


def label_categories(markets: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket markets by subject, from the API's own category when present and from
    the question text otherwise.

    This exists because specialists are the whole point. A trader with genuine
    forecasting skill in one domain -- weather being the clearest case, where
    better models than the market consensus are a real, legal and repeatable
    edge -- trades almost exclusively in that domain. Sampling uniformly across
    all of Polymarket gives such a trader a vanishing chance of appearing, and
    the first live scan demonstrated it: 44 markets drawn at random, nobody found.
    """
    out = markets.copy()
    parts = [out["question"].fillna("").astype(str)]
    for col in ("slug", "api_category"):
        if col in out.columns:
            parts.append(out[col].fillna("").astype(str))
    text = parts[0]
    for extra in parts[1:]:
        text = text + " " + extra
    text = text.str.lower()

    cat = pd.Series("other", index=out.index, dtype=object)
    for name, words in CATEGORY_RULES.items():
        hit = text.apply(lambda s, w=words: any(k in s for k in w))
        cat = cat.where(~(hit & (cat == "other")), name)
    out["category"] = cat
    return out


def discover_stratified(client: PolymarketClient, markets: pd.DataFrame,
                        per_category: int = 120, seed: int = 0,
                        categories: list[str] | None = None,
                        max_trades_per_market: int = 5000
                        ) -> tuple[pd.DataFrame, dict]:
    """
    Sample markets WITHIN each category rather than uniformly across all of them.

    Two things this buys, and both matter:

    1. **Specialists become findable.** Concentrating the sample inside a domain
       means a trader who only touches that domain appears in enough markets to
       be judged, instead of surfacing once and being filtered out.
    2. **The luck bar falls.** The multiple-testing correction scales with how
       many wallets were searched. Scanning 400 weather traders carries a far
       weaker penalty than scanning 40,000 of everybody, so a real edge inside a
       category needs to be much smaller to be provable there.

    Point 2 is easy to miss: narrowing the search BEFORE looking at performance
    makes edges easier to prove, not harder. What must never happen is narrowing
    on performance itself -- that is the leaderboard trap.
    """
    rng = np.random.default_rng(seed)
    labelled = label_categories(markets)
    # "other" is sampled like any other bucket. Excluding it discarded most of
    # the pool -- the keyword rules are crude by design, so whatever they fail to
    # label is not noise, it is the majority of the market universe.
    cats = categories or list(labelled["category"].unique())

    frames, meta_by_cat = [], {}
    for cat in cats:
        pool = labelled[labelled["category"] == cat]
        if pool.empty:
            continue
        take = min(per_category, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        sample = pool.iloc[np.sort(idx)]
        got = 0
        for mid in sample["market_id"]:
            try:
                raw = client.market_trades(mid, limit=max_trades_per_market)
            except RuntimeError:
                continue
            if raw:
                df = pd.DataFrame(raw)
                df["_category"] = cat
                frames.append(df)
                got += 1
        meta_by_cat[cat] = {"pool": len(pool), "sampled": take, "fetched": got}

    if not frames:
        return pd.DataFrame(), {"by_category": meta_by_cat,
                                "n_wallets_discovered": 0}
    raw_trades = pd.concat(frames, ignore_index=True)
    try:
        wcol = resolve_fields(raw_trades, TRADE_FIELD_CANDIDATES,
                              ("wallet",), "trades")["wallet"]
        n_wallets = int(raw_trades[wcol].nunique())
    except ValueError:
        n_wallets = 0
    return raw_trades, {"by_category": meta_by_cat,
                        "n_raw_trades": len(raw_trades),
                        "n_wallets_discovered": n_wallets}


def discover_population(client: PolymarketClient, markets: pd.DataFrame,
                        n_markets: int = 200, seed: int = 0,
                        max_trades_per_market: int = 5000
                        ) -> tuple[pd.DataFrame, dict]:
    """
    Enumerate wallets by participation in a random sample of resolved markets.

    Returns (raw_trades, meta). `meta['n_wallets_discovered']` is the population
    size to pass to the luck correction -- it must be the number SEARCHED, not
    the number surviving later filters. Filtering first and counting after is
    the most common way that correction gets quietly understated.

    Markets are sampled at random rather than taken by volume rank, so the
    population is not tilted toward whichever traders specialise in the largest
    markets.
    """
    rng = np.random.default_rng(seed)
    if len(markets) > n_markets:
        idx = rng.choice(len(markets), size=n_markets, replace=False)
        sample = markets.iloc[np.sort(idx)]
    else:
        sample = markets

    frames, failures = [], 0
    for mid in sample["market_id"]:
        try:
            raw = client.market_trades(mid, limit=max_trades_per_market)
        except RuntimeError:
            failures += 1
            continue
        if raw:
            frames.append(pd.DataFrame(raw))

    if not frames:
        return pd.DataFrame(), {"n_wallets_discovered": 0, "n_markets": 0,
                                "failures": failures}
    raw_trades = pd.concat(frames, ignore_index=True)
    try:
        wallet_col = resolve_fields(raw_trades, TRADE_FIELD_CANDIDATES,
                                    ("wallet",), "trades")["wallet"]
    except ValueError:
        wallet_col = ""
    meta = {
        "n_markets_sampled": len(sample),
        "n_markets_fetched": len(frames),
        "n_raw_trades": len(raw_trades),
        "n_wallets_discovered": int(raw_trades[wallet_col].nunique())
        if wallet_col in raw_trades else 0,
        "failures": failures,
    }
    return raw_trades, meta


def markets_by_condition_ids(client: "PolymarketClient",
                             condition_ids: list[str],
                             batch: int = 20) -> pd.DataFrame:
    """
    Look up resolution data for SPECIFIC markets, by condition id.

    Named-wallet mode needs this. Matching one wallet's fills against a generic
    pool of recent markets fails whenever that wallet trades a niche corner --
    and a weather specialist trading daily temperature markets is exactly that
    case. The first named run fetched the account successfully and scored zero
    trades, because none of its markets were in a pool built from the most
    recent few thousand closed markets.

    Fetching by id inverts the dependency: take the markets the wallet actually
    traded, then resolve those. No pool depth required.

    The verification below is the point of this function, not decoration.
    Gamma ignores query parameters it does not recognise and answers 200 with
    its DEFAULT listing, so a wrong parameter name looks exactly like a
    successful lookup. The first attempt at this shipped `condition_ids` as a
    comma-joined string, and the run came back with 92 batches x the default
    page of 20 = 1,840 "markets", none of them the wallet's, every one filed
    under politics for an account that trades weather. Zero fills matched and
    the report said "not enough resolved trades to score" -- a silent wrong
    answer dressed as a negative result.

    So: a candidate request form is accepted only if what comes back actually
    contains ids that were asked for. Nothing is trusted for returning 200.
    """
    ids = [c for c in dict.fromkeys(condition_ids) if isinstance(c, str) and c]
    if not ids:
        return _markets_to_frame([])
    batch = max(1, min(batch, 50))

    def _asked_for(payload: Any, chunk: list[str]) -> list[dict]:
        """Rows whose condition id is one we requested. Others are noise."""
        if not isinstance(payload, list):
            return []
        want = {c.lower() for c in chunk}
        keep = []
        for r in payload:
            if not isinstance(r, dict):
                continue
            cid = r.get("conditionId") or r.get("condition_id") or ""
            if str(cid).lower() in want:
                keep.append(r)
        return keep

    # Candidate request forms, most-likely first. `doseq` in _get turns a list
    # into the repeated form (?condition_ids=a&condition_ids=b).
    forms = [
        ("gamma condition_ids[]", lambda c: {"condition_ids": list(c),
                                             "limit": len(c)}),
        ("gamma condition_ids=csv", lambda c: {"condition_ids": ",".join(c),
                                               "limit": len(c)}),
        ("gamma conditionIds[]", lambda c: {"conditionIds": list(c),
                                            "limit": len(c)}),
    ]

    rows: list[dict] = []
    chosen = None
    probe = ids[:batch]
    for label, build in forms:
        try:
            payload = client._get(f"{GAMMA}/markets", build(probe))
        except Exception:                                    # noqa: BLE001
            continue
        hits = _asked_for(payload, probe)
        if hits:
            chosen = (label, build)
            rows.extend(hits)
            break

    if chosen is not None:
        label, build = chosen
        for i in range(batch, len(ids), batch):
            chunk = ids[i:i + batch]
            try:
                payload = client._get(f"{GAMMA}/markets", build(chunk))
            except Exception:                                # noqa: BLE001
                continue
            rows.extend(_asked_for(payload, chunk))
    else:
        label = "none verified"

    n_gamma = len(rows)
    # Gamma's batch filter answers for recent markets and quietly returns
    # nothing for the rest -- asked for this wallet's 1,831 condition ids it
    # returned 13, all of them from the last two days. A partial answer is the
    # same hazard as a wrong one: 13 markets scored the account at n_eff 4.7,
    # which is a coin-flip dressed as a verdict. So anything Gamma did not
    # answer for is re-asked one id at a time, where the id is in the URL path
    # and cannot be silently dropped by a filter.
    have = {str(r.get("conditionId") or r.get("condition_id") or "").lower()
            for r in rows}
    missing = [c for c in ids if c.lower() not in have]
    if missing:
        rows.extend(_clob_markets(client, missing))

    df = _markets_to_frame(rows)
    df.attrs["lookup_form"] = label
    df.attrs["n_requested"] = len(ids)
    df.attrs["n_from_gamma"] = n_gamma
    df.attrs["n_from_clob"] = len(rows) - n_gamma
    df.attrs["n_matched"] = len(df)
    return df


def _clob_markets(client: "PolymarketClient", ids: list[str]) -> list[dict]:
    """
    Fallback: one CLOB request per condition id, reshaped to the Gamma schema.

    Slower than a batch filter -- one round trip per market -- but it cannot
    return somebody else's market, because the id is in the path rather than in
    a query parameter the server is free to ignore. Given how the batch path
    failed, that property is worth the extra requests.

    CLOB reports resolution as a `winner` flag per token rather than as prices,
    which is also strictly better: no 0.995-rounding judgement call.
    """
    out = []
    for cid in ids:
        try:
            m = client._get(f"{CLOB}/markets/{cid}", {})
        except Exception:                                    # noqa: BLE001
            continue                    # 404 for an unknown id is not an error
        if not isinstance(m, dict) or not m.get("condition_id"):
            continue
        toks = [t for t in (m.get("tokens") or []) if isinstance(t, dict)]
        # Trades carry outcomeIndex against Gamma's outcome order, which is
        # Yes-first on binary markets. Sort to match, and only when the labels
        # say it is safe to -- reordering a multi-outcome market on a guess
        # would map every fill to the wrong leg.
        labels = [str(t.get("outcome", "")).strip().lower() for t in toks]
        if sorted(labels) == ["no", "yes"]:
            toks.sort(key=lambda t: str(t.get("outcome", "")).strip().lower() != "yes")
        prices = None
        if toks:
            if any(t.get("winner") for t in toks):
                prices = ["1" if t.get("winner") else "0" for t in toks]
            else:
                prices = [str(t.get("price", "")) for t in toks]
        out.append({
            "conditionId": m.get("condition_id"),
            "question": m.get("question"),
            "slug": m.get("market_slug"),
            "endDate": m.get("end_date_iso"),
            "outcomePrices": prices,
            "volumeNum": None,
            "category": m.get("category") or m.get("tags"),
        })
    return out


def resolve_username(client: "PolymarketClient", name: str) -> list[str]:
    """
    Map a Polymarket display name to candidate wallet addresses.

    Needed because accounts are discussed by username and never by address.
    Tries the leaderboard surfaces, since those are the only public endpoints
    that carry name-to-address pairs; there is no documented username search.

    A note on using the leaderboard here, given `discover_by_leaderboard`
    deliberately raises. The difference is what it is used FOR. Seeding a
    candidate population from the leaderboard selects on the outcome variable
    and corrupts everything downstream. Resolving one identifier that someone
    already named is a lookup, and it ranks nobody.

    It is still not unbiased, and the distinction is worth keeping sharp: an
    account reaches you because somebody celebrated its record, so you inherit
    their selection without seeing its size. That is exactly why named-wallet
    mode reports both the pre-specified bar and the inherited-search bar.
    """
    # The working endpoint, found by probing rather than guessing. The
    # `search_profiles=true` flag is the whole trick: without it the same URL
    # answers 200 with an empty pagination object and nothing else, which reads
    # exactly like "no such user" rather than "you did not ask for profiles".
    endpoints = [
        (f"{GAMMA}/public-search", {"q": name, "search_profiles": "true"}),
        (f"{GAMMA}/public-search", {"q": name, "search_profiles": "true",
                                    "events_status": "all",
                                    "limit_per_type": 20}),
    ]
    target = name.strip().lower()
    out: list[str] = []
    for url, params in endpoints:
        try:
            payload = client._get(url, params)
        except Exception:                                    # noqa: BLE001
            continue
        rows = payload if isinstance(payload, list) else [payload]
        # The profile list is nested under a key, not returned at the top level.
        if len(rows) == 1 and isinstance(rows[0], dict):
            for key in ("profiles", "leaderboard", "data", "results", "users"):
                v = rows[0].get(key)
                if isinstance(v, list):
                    rows = v
                    break
        for r in rows:
            if not isinstance(r, dict):
                continue
            label = " ".join(str(r.get(k, "")) for k in
                             ("name", "displayName", "username", "pseudonym",
                              "handle")).lower()
            if target not in label:
                continue
            for k in ("proxyWallet", "proxy_wallet", "wallet", "address",
                      "user", "userAddress", "walletAddress"):
                v = r.get(k)
                if isinstance(v, str) and v.startswith("0x") and len(v) >= 40:
                    out.append(v.lower())
        if out:
            break
    return sorted(set(out))


def fetch_full_histories(client: "PolymarketClient", candidates: list[str],
                         max_wallets: int = 400,
                         activity: dict[str, int] | None = None,
                         limit_per_wallet: int = 3000
                         ) -> tuple[pd.DataFrame, dict]:
    """
    Fetch each candidate's COMPLETE trade history, not just the slice that
    happened to fall inside the sampled markets.

    Without this a wallet is judged on whichever handful of its trades landed in
    our sample, which is both a tiny sample and the wrong one -- and it makes the
    persistence test impossible, since almost no wallet appears in both halves
    of a 200-market slice. The first live run scored 313 wallets and found
    exactly ONE active in both periods.

    Candidates are ranked by ACTIVITY (fill count), never by profit. Activity is
    outcome-independent, so narrowing by it does not pre-select winners the way
    a leaderboard would. The number of wallets whose performance is then
    examined -- `max_wallets` -- is the honest N for the luck correction, and it
    is returned as such.
    """
    order = sorted(candidates, key=lambda w: -(activity or {}).get(w, 0))
    chosen = order[:max_wallets]
    frames, failures = [], 0
    for w in chosen:
        try:
            raw = client.user_trades(w, limit=limit_per_wallet)
        except RuntimeError:
            failures += 1
            continue
        if raw:
            frames.append(pd.DataFrame(raw))
    meta = {"n_candidates": len(candidates), "n_fetched": len(frames),
            "n_wallets_examined": len(chosen), "fetch_failures": failures}
    if not frames:
        return pd.DataFrame(), meta
    return pd.concat(frames, ignore_index=True), meta


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

REQUIRED_TRADE_FIELDS = ("wallet", "market_id", "price", "size", "side")


def validate_trade_fields(raw: pd.DataFrame) -> dict[str, str]:
    """
    Resolve and record the trade field mapping, or fail loudly.

    A silently-missing field becomes a NaN column and then a plausible-looking
    result computed from nothing. This project has already produced three
    confident wrong answers from exactly that pattern; the guard is cheap.

    `outcome_index` and `timestamp` are not required: some responses name the
    taken side textually, and a missing timestamp only costs the persistence
    split, which falls back to resolution time anyway.
    """
    global TRADE_FIELDS
    TRADE_FIELDS = resolve_fields(raw, TRADE_FIELD_CANDIDATES,
                                  REQUIRED_TRADE_FIELDS, "Polymarket trades")
    return TRADE_FIELDS


def describe_response(raw: pd.DataFrame, label: str = "trades") -> str:
    """Human-readable report of what matched, for unattended log reading."""
    cands = (TRADE_FIELD_CANDIDATES if label == "trades"
             else MARKET_FIELD_CANDIDATES)
    req = REQUIRED_TRADE_FIELDS if label == "trades" else ("market_id",)
    try:
        found = resolve_fields(raw, cands, req, label)
    except ValueError as e:
        return f"[{label}] FIELD RESOLUTION FAILED\n{e}"
    lines = [f"[{label}] {len(raw)} rows, {len(raw.columns)} columns"]
    for k in cands:
        lines.append(f"    {k:15} -> {found.get(k, '(not found)')}")
    unused = sorted(set(map(str, raw.columns)) - set(found.values()))
    if unused:
        lines.append(f"    unused columns: {unused[:12]}")
    return "\n".join(lines)


def normalise_trades(raw: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw API trades into the schema `wallets.score_wallets` expects.

    Joins each trade to its market's resolution to produce the binary `outcome`
    from the trader's own perspective, and carries `resolved_at` so the
    persistence split can be made on resolution time rather than trade time --
    the difference between a valid test and a lookahead-contaminated one.
    """
    f = validate_trade_fields(raw)
    df = pd.DataFrame({
        "wallet": raw[f["wallet"]].astype(str).str.lower(),
        "market_id": raw[f["market_id"]].astype(str),
        "price": pd.to_numeric(raw[f["price"]], errors="coerce"),
        "size": pd.to_numeric(raw[f["size"]], errors="coerce"),
        "side": raw[f["side"]].astype(str).str.upper(),
    })
    df["timestamp"] = (pd.to_numeric(raw[f["timestamp"]], errors="coerce")
                       if "timestamp" in f else np.nan)
    if "outcome_index" in f:
        df["outcome_index"] = pd.to_numeric(raw[f["outcome_index"]],
                                            errors="coerce")
    elif "outcome" in raw.columns:
        # Some responses name the side taken ("Yes"/"No") instead of indexing it.
        df["outcome_index"] = (raw["outcome"].astype(str).str.strip().str.lower()
                               .map({"yes": 0, "no": 1}))
    else:
        raise ValueError(
            "No outcomeIndex and no outcome label in the trade response, so "
            "there is no way to tell which side a trade took. Refusing rather "
            "than guessing.")

    m = markets[["market_id", "winning_index", "resolved_at"]].copy()
    m["market_id"] = m["market_id"].astype(str)
    out = df.merge(m, on="market_id", how="inner")

    # The trade wins when the index it took is the index that resolved true.
    out["outcome"] = (out["outcome_index"] == out["winning_index"]).astype(float)
    out = out[out["winning_index"].notna()]
    out = out[(out["price"] > 0) & (out["price"] < 1) & (out["size"] > 0)]
    return out[["wallet", "market_id", "timestamp", "price", "size", "side",
                "outcome", "resolved_at"]].reset_index(drop=True)


def _markets_to_frame(rows: Iterable[dict]) -> pd.DataFrame:
    """Gamma markets -> market_id, winning_index, resolved_at, volume."""
    global MARKET_FIELDS
    rows = list(rows)
    if not rows:
        return pd.DataFrame(columns=["market_id", "question", "winning_index",
                                     "resolved_at", "volume"])
    MARKET_FIELDS = resolve_fields(pd.DataFrame(rows), MARKET_FIELD_CANDIDATES,
                                   ("market_id",), "Polymarket markets")
    f = MARKET_FIELDS
    recs = []
    for r in rows:
        prices = r.get(f.get("outcome_prices", "outcomePrices"))
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except json.JSONDecodeError:
                prices = None
        win = None
        if isinstance(prices, (list, tuple)) and len(prices) >= 2:
            try:
                vals = [float(p) for p in prices]
            except (TypeError, ValueError):
                vals = []
            # Tolerance matters more than it looks. Demanding exactly 1.0/0.0
            # rejected almost every market: three live runs sampled 44, then 8,
            # markets out of a requested 200, and every downstream symptom --
            # too few scorable wallets, a persistence test that never ran --
            # traced back to here. Resolved markets round, get stored as
            # strings, and occasionally settle at 0.995.
            if len(vals) >= 2 and max(vals) >= 0.95 and min(vals) <= 0.05:
                win = int(np.argmax(vals))
        if win is None:
            # Fallbacks for sources that report resolution separately from price.
            for key in ("umaResolutionStatus", "resolutionStatus", "winner",
                        "resolvedOutcome"):
                v = r.get(key)
                if isinstance(v, str) and v.strip().lower() in ("yes", "no"):
                    win = 0 if v.strip().lower() == "yes" else 1
                    break
        end = r.get(f.get("end_date", "endDate"))
        raw_cat = r.get(f.get("category", "category"))
        if isinstance(raw_cat, (list, tuple)):
            raw_cat = " ".join(str(x) for x in raw_cat)
        recs.append({
            "slug": r.get(f.get("slug", "slug")),
            "api_category": raw_cat,
            "market_id": str(r.get(f["market_id"], "")),
            "question": r.get(f.get("question", "question")),
            "winning_index": win,
            "resolved_at": pd.to_datetime(end, errors="coerce",
                                          utc=True).timestamp()
            if end else np.nan,
            "volume": pd.to_numeric(r.get(f.get("volume", "volume")),
                                    errors="coerce"),
        })
    df = pd.DataFrame(recs)
    df = df[df["market_id"] != ""]
    # Diagnostics for unattended runs: how many markets survive each stage, and
    # what the raw resolution field actually looked like. Without this a thin
    # pool is invisible and every downstream number is quietly starved.
    df.attrs["n_raw"] = len(rows)
    df.attrs["n_with_winner"] = int(df["winning_index"].notna().sum())
    sample = []
    for r in rows[:400]:
        v = r.get(f.get("outcome_prices", "outcomePrices"))
        if v is not None:
            sample.append(str(v)[:40])
    df.attrs["outcome_price_samples"] = sample[:8]
    return df


def sample_response_shapes() -> dict:
    """
    Recorded response shapes, for testing the adapter without network access.

    These mirror the documented public API. If the live shape has drifted,
    `validate_trade_fields` will say so on the first fetch in Colab -- which is
    the right place to find out, not three transformations downstream.
    """
    return {
        "trades": [{
            "proxyWallet": "0xABCdef0000000000000000000000000000000001",
            "conditionId": "0xmarket1", "timestamp": 1700000000,
            "price": "0.42", "size": "250", "side": "BUY",
            "outcomeIndex": 0, "outcome": "Yes", "title": "Example market",
        }, {
            "proxyWallet": "0xABCdef0000000000000000000000000000000002",
            "conditionId": "0xmarket1", "timestamp": 1700000100,
            "price": "0.58", "size": "100", "side": "SELL",
            "outcomeIndex": 0, "outcome": "Yes", "title": "Example market",
        }],
        "markets": [{
            "conditionId": "0xmarket1", "question": "Example market",
            "closed": True, "endDate": "2024-01-15T00:00:00Z",
            "outcomePrices": '["1", "0"]', "volumeNum": 250000.0,
        }],
    }


def markets_by_windows(client: "PolymarketClient", days_back: int = 730,
                       window_days: int = 14, max_depth: int = 6,
                       page_cap: int = 100) -> pd.DataFrame:
    """
    Crawl the market listing in date windows, splitting any window that truncates.

    Necessary because Gamma's listing cannot be paged. The probe that settled
    this measured two things:

      - `offset` is IGNORED. Five pages at offsets 0..2000 returned five
        identical pages: 500 rows each, 500 distinct markets in total. The
        loop in `resolved_markets` looked like it was crawling and was in fact
        re-reading page one, which is why a requested pool of 3,000 arrived as
        333 and the longshot calibration had an effective sample of 30 a band.
      - a query carrying date filters returns at most 100 rows whatever
        `limit` says.

    So the only way through is many narrow windows. The important part is the
    SPLIT: a window that comes back exactly at the cap is assumed truncated and
    is halved and re-asked, recursively. A window returning fewer than the cap
    is complete, and that is a property we can check rather than hope for --
    which matters, because a silently truncated window looks exactly like a
    quiet stretch of the calendar.
    """
    import datetime as dt

    now = dt.datetime.now(dt.timezone.utc)
    rows: list[dict] = []
    seen: set[str] = set()
    truncated = 0

    def fetch(lo: dt.datetime, hi: dt.datetime, depth: int) -> None:
        nonlocal truncated
        try:
            batch = client._get(f"{GAMMA}/markets", {
                "closed": "true", "limit": page_cap,
                "end_date_min": lo.strftime("%Y-%m-%d"),
                "end_date_max": hi.strftime("%Y-%m-%d"),
                "order": "endDate", "ascending": "false"})
        except Exception:                                    # noqa: BLE001
            return
        if not isinstance(batch, list):
            return
        # At the cap means "there was probably more". Split rather than accept
        # a silent partial answer -- the whole point of this function.
        if len(batch) >= page_cap and depth < max_depth and (hi - lo).days > 1:
            mid = lo + (hi - lo) / 2
            fetch(lo, mid, depth + 1)
            fetch(mid, hi, depth + 1)
            return
        if len(batch) >= page_cap:
            truncated += 1                 # bottomed out; window still full
        for r in batch:
            cid = str(r.get("conditionId") or r.get("condition_id") or "")
            if cid and cid not in seen:
                seen.add(cid)
                rows.append(r)

    hi = now
    while hi > now - dt.timedelta(days=days_back):
        lo = hi - dt.timedelta(days=window_days)
        fetch(lo, hi, 0)
        hi = lo

    df = _markets_to_frame(rows)
    df.attrs["n_windows_truncated"] = truncated
    df.attrs["crawl"] = (f"date windows, {window_days}d, {days_back}d back, "
                         f"cap {page_cap}")
    return df
