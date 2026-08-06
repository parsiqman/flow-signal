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

GAMMA = "https://gamma-api.polymarket.com"
DATA = "https://data-api.polymarket.com"

# Field names as documented for the public APIs. Response shapes do change, so
# these are isolated here and validated loudly on first fetch rather than being
# scattered through the code -- see `validate_trade_fields`.
TRADE_FIELDS = {
    "wallet": "proxyWallet",
    "market_id": "conditionId",
    "timestamp": "timestamp",
    "price": "price",
    "size": "size",
    "side": "side",
    "outcome_index": "outcomeIndex",
}
MARKET_FIELDS = {
    "market_id": "conditionId",
    "question": "question",
    "closed": "closed",
    "end_date": "endDate",
    "outcome_prices": "outcomePrices",
    "volume": "volumeNum",
}


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

        import urllib.parse
        import urllib.request

        wait = self.cfg.rate_limit_s - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)

        full = f"{url}?{urllib.parse.urlencode(params)}"
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
        rows, offset = [], 0
        while len(rows) < limit:
            batch = self._get(f"{GAMMA}/markets", {
                "closed": "true", "limit": self.cfg.page_size, "offset": offset,
                "order": "endDate", "ascending": "false"})
            if not batch:
                break
            rows.extend(batch)
            offset += self.cfg.page_size
            if len(batch) < self.cfg.page_size:
                break
        df = _markets_to_frame(rows)
        if min_volume:
            df = df[df["volume"].fillna(0) >= min_volume]
        return df.head(limit).reset_index(drop=True)

    def market_trades(self, market_id: str, limit: int = 5000) -> list[dict]:
        """Every trade in one market. This is the unbiased discovery unit."""
        out, offset = [], 0
        while len(out) < limit:
            batch = self._get(f"{DATA}/trades", {
                "market": market_id, "limit": self.cfg.page_size,
                "offset": offset, "takerOnly": "false"})
            if not batch:
                break
            out.extend(batch)
            offset += self.cfg.page_size
            if len(batch) < self.cfg.page_size:
                break
        return out[:limit]

    def user_trades(self, wallet: str, limit: int = 5000) -> list[dict]:
        """Full history for one wallet, across all markets."""
        out, offset = [], 0
        while len(out) < limit:
            batch = self._get(f"{DATA}/trades", {
                "user": wallet, "limit": self.cfg.page_size, "offset": offset,
                "takerOnly": "false"})
            if not batch:
                break
            out.extend(batch)
            offset += self.cfg.page_size
            if len(batch) < self.cfg.page_size:
                break
        return out[:limit]


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
    wallet_col = TRADE_FIELDS["wallet"]
    meta = {
        "n_markets_sampled": len(sample),
        "n_markets_fetched": len(frames),
        "n_raw_trades": len(raw_trades),
        "n_wallets_discovered": int(raw_trades[wallet_col].nunique())
        if wallet_col in raw_trades else 0,
        "failures": failures,
    }
    return raw_trades, meta


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def validate_trade_fields(raw: pd.DataFrame) -> None:
    """
    Fail loudly if the API shape has drifted.

    A silently-missing field becomes a NaN column and then a plausible-looking
    result computed from nothing. This project has already produced three
    confident wrong answers from exactly that pattern; the guard is cheap.
    """
    missing = [k for k, v in TRADE_FIELDS.items() if v not in raw.columns]
    if missing:
        raise ValueError(
            f"Polymarket trade response is missing {missing} "
            f"(expected columns {[TRADE_FIELDS[m] for m in missing]}).\n"
            f"Got: {sorted(raw.columns)[:25]}\n"
            f"The API shape has probably changed -- update TRADE_FIELDS in "
            f"client.py rather than working around it downstream.")


def normalise_trades(raw: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw API trades into the schema `wallets.score_wallets` expects.

    Joins each trade to its market's resolution to produce the binary `outcome`
    from the trader's own perspective, and carries `resolved_at` so the
    persistence split can be made on resolution time rather than trade time --
    the difference between a valid test and a lookahead-contaminated one.
    """
    validate_trade_fields(raw)
    f = TRADE_FIELDS
    df = pd.DataFrame({
        "wallet": raw[f["wallet"]].astype(str).str.lower(),
        "market_id": raw[f["market_id"]].astype(str),
        "timestamp": pd.to_numeric(raw[f["timestamp"]], errors="coerce"),
        "price": pd.to_numeric(raw[f["price"]], errors="coerce"),
        "size": pd.to_numeric(raw[f["size"]], errors="coerce"),
        "side": raw[f["side"]].astype(str).str.upper(),
        "outcome_index": pd.to_numeric(raw[f["outcome_index"]], errors="coerce"),
    })

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
    f = MARKET_FIELDS
    recs = []
    for r in rows:
        prices = r.get(f["outcome_prices"])
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except json.JSONDecodeError:
                prices = None
        win = None
        if isinstance(prices, (list, tuple)) and len(prices) >= 2:
            vals = [float(p) for p in prices]
            # A resolved binary market prices its outcomes at exactly 1 and 0.
            if max(vals) > 0.99 and min(vals) < 0.01:
                win = int(np.argmax(vals))
        end = r.get(f["end_date"])
        recs.append({
            "market_id": str(r.get(f["market_id"], "")),
            "question": r.get(f["question"]),
            "winning_index": win,
            "resolved_at": pd.to_datetime(end, errors="coerce",
                                          utc=True).timestamp()
            if end else np.nan,
            "volume": pd.to_numeric(r.get(f["volume"]), errors="coerce"),
        })
    df = pd.DataFrame(recs)
    return df[df["market_id"] != ""]


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
