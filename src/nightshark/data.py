"""Price data: a candle container and two feeds behind one interface.

The no-lookahead rule is enforced here rather than left to callers. Every feed
takes an `as_of` timestamp and returns only candles that had already CLOSED at
that instant. A 1-minute candle stamped 12:03:00 covers 12:03:00-12:03:59 and
is therefore not available until 12:04:00. Asking a feed for data is the only
way the rest of the bot sees prices, so a lookahead bug has to get past this
file first.

Kalshi's BTC markets settle against the CF Benchmarks BRTI, which is a licensed
feed. `RestFeed` pulls a public exchange's candles as a stand-in; the prices
track BRTI closely but are not identical, and near a strike that difference
decides trades. Treat live RestFeed output as an approximation of settlement.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

MINUTE = 60


@dataclass(frozen=True)
class CandleSeries:
    """OHLCV candles in parallel arrays. `ts` is the candle's OPEN time (UTC epoch seconds)."""

    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    interval_s: int

    def __len__(self) -> int:
        return int(self.ts.size)

    def __getitem__(self, sl: slice) -> "CandleSeries":
        return CandleSeries(
            self.ts[sl], self.open[sl], self.high[sl], self.low[sl],
            self.close[sl], self.volume[sl], self.interval_s,
        )

    def last(self, n: int) -> "CandleSeries":
        return self[max(0, len(self) - n):]

    def resample(self, minutes: int) -> "CandleSeries":
        """Aggregate to a coarser interval, aligned to the epoch.

        Epoch alignment matters: it makes 15-minute buckets fall on :00/:15/:30/:45,
        the same boundaries Kalshi's windows use. A trailing partial bucket is
        dropped -- an unclosed candle is exactly the lookahead this module exists
        to prevent.
        """
        step = minutes * MINUTE
        if step % self.interval_s:
            raise ValueError(f"{minutes}m is not a multiple of the {self.interval_s}s base interval")
        if len(self) == 0:
            return self

        bucket = (self.ts // step) * step
        edges = np.flatnonzero(np.r_[True, bucket[1:] != bucket[:-1]])
        starts, ends = edges, np.r_[edges[1:], len(self)]

        per_bucket = step // self.interval_s
        complete = (ends - starts) == per_bucket
        starts, ends = starts[complete], ends[complete]
        if starts.size == 0:
            return CandleSeries(*(np.array([]) for _ in range(6)), interval_s=step)

        return CandleSeries(
            ts=bucket[starts],
            open=self.open[starts],
            high=np.array([self.high[s:e].max() for s, e in zip(starts, ends)]),
            low=np.array([self.low[s:e].min() for s, e in zip(starts, ends)]),
            close=self.close[ends - 1],
            volume=np.array([self.volume[s:e].sum() for s, e in zip(starts, ends)]),
            interval_s=step,
        )


class PriceFeed:
    """Interface every feed implements."""

    def history(self, as_of: datetime, minutes: int) -> CandleSeries:
        raise NotImplementedError

    def spot(self, as_of: datetime) -> float:
        """Last CLOSED price at `as_of`. Not the live tick -- see module docstring."""
        h = self.history(as_of, 5)
        if len(h) == 0:
            raise RuntimeError("feed returned no candles")
        return float(h.close[-1])


class SyntheticFeed(PriceFeed):
    """Deterministic simulated BTC tape. Offline, seeded, reproducible.

    Log-returns with AR(1) stochastic volatility, so the tape has vol clustering
    and fat-ish tails instead of clean Gaussian noise. Each 1-minute candle is
    built from sub-ticks, giving genuine wicks for the candle-structure features
    to read.

    This is a test fixture, not a market simulator. It has no order flow, no
    news, and no reason for direction to be predictable. A strategy that looks
    profitable on this tape has found a property of the generator.
    """

    def __init__(self, seed: int = 7, start_price: float = 68000.0,
                 span_days: int = 45, anchor: datetime | None = None,
                 annual_vol: float = 0.55, subticks: int = 6):
        self.seed = seed
        self.start_price = start_price
        self.subticks = subticks
        anchor = anchor or datetime.now(timezone.utc)
        self._t0 = int((anchor - timedelta(days=span_days)).timestamp()) // MINUTE * MINUTE
        n = span_days * 2 * 24 * 60
        self._build(n, annual_vol)

    def _build(self, n: int, annual_vol: float) -> None:
        rng = np.random.default_rng(self.seed)
        per_min = annual_vol / np.sqrt(365 * 24 * 60)

        # AR(1) log-vol: persistent quiet and busy regimes.
        log_v = np.zeros(n)
        for i in range(1, n):
            log_v[i] = 0.995 * log_v[i - 1] + rng.normal(0, 0.05)
        vol = per_min * np.exp(log_v - log_v.var() / 2)

        k = self.subticks
        steps = rng.normal(0, np.repeat(vol, k) / np.sqrt(k))
        path = self.start_price * np.exp(np.cumsum(steps))
        p = path.reshape(n, k)

        self._ts = self._t0 + np.arange(n) * MINUTE
        self._open = np.r_[self.start_price, p[:-1, -1]]
        self._close = p[:, -1]
        self._high = np.maximum(p.max(axis=1), self._open)
        self._low = np.minimum(p.min(axis=1), self._open)
        # Volume correlates with volatility, as it does in the real tape.
        self._vol = np.abs(rng.normal(0, 1, n)) * 40 * (vol / per_min) + 5

    def history(self, as_of: datetime, minutes: int) -> CandleSeries:
        cutoff = int(as_of.timestamp())
        # A candle opening at t closes at t + 60; it is only usable once closed.
        usable = np.flatnonzero(self._ts + MINUTE <= cutoff)
        if usable.size == 0:
            raise RuntimeError(f"synthetic feed has no data at or before {as_of:%Y-%m-%d %H:%M}Z")
        end = usable[-1] + 1
        start = max(0, end - minutes)
        sl = slice(start, end)
        return CandleSeries(self._ts[sl], self._open[sl], self._high[sl],
                            self._low[sl], self._close[sl], self._vol[sl], MINUTE)


class RestFeed(PriceFeed):
    """1-minute candles from a public REST endpoint (Coinbase shape by default).

    Coinbase `/products/{id}/candles` returns rows of
    [time, low, high, open, close, volume], newest first. Point `url` elsewhere
    and pass a `parser` to use a different venue or a licensed CF Benchmarks feed.
    """

    def __init__(self, url: str, symbol: str = "BTC-USD", parser=None, timeout: float = 15.0):
        self.url = url
        self.symbol = symbol
        self.timeout = timeout
        self.parser = parser or self._parse_coinbase

    @staticmethod
    def _parse_coinbase(payload) -> CandleSeries:
        rows = sorted(payload, key=lambda r: r[0])
        a = np.array(rows, dtype=float)
        return CandleSeries(ts=a[:, 0], low=a[:, 1], high=a[:, 2], open=a[:, 3],
                            close=a[:, 4], volume=a[:, 5], interval_s=MINUTE)

    def history(self, as_of: datetime, minutes: int) -> CandleSeries:
        cutoff = int(as_of.timestamp()) // MINUTE * MINUTE
        start = cutoff - minutes * MINUTE
        url = (f"{self.url}?granularity=60"
               f"&start={datetime.fromtimestamp(start, timezone.utc).isoformat()}"
               f"&end={datetime.fromtimestamp(cutoff, timezone.utc).isoformat()}")
        req = urllib.request.Request(url, headers={"User-Agent": "nightshark/0.1"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            series = self.parser(json.loads(r.read().decode()))
        # Belt and braces: drop anything that had not closed by `as_of`.
        keep = np.flatnonzero(series.ts + MINUTE <= int(as_of.timestamp()))
        if keep.size == 0:
            raise RuntimeError("REST feed returned no closed candles")
        return series[slice(0, int(keep[-1]) + 1)]


def build_feed(cfg) -> PriceFeed:
    if cfg.feed == "synthetic":
        return SyntheticFeed(seed=cfg.synthetic_seed)
    if cfg.feed == "rest":
        return RestFeed(cfg.feed_url, cfg.feed_symbol)
    raise ValueError(f"unknown feed {cfg.feed!r} (expected 'synthetic' or 'rest')")
