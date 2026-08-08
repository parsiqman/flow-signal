"""
What the order book actually costs, in the bands the rule trades.

The walk-forward result is 4.41c/share net, but that "net" rests on an ASSUMED
1c half-spread. The break-even is 5.41c. Everything therefore turns on a number
that was typed in rather than measured, and in the one place where a typed
number is most dangerous: the rule's average fill price is 89c, and a book that
is 2c wide there quietly halves the edge.

So this module measures three things off live books, in the same price bands:

  - **half-spread**, what crossing costs per share
  - **depth at the touch**, which is the capacity answer -- the counterparty
    here is recreational money and it is not deep
  - **how much of the edge survives**, per band, against that band's own
    measured spread rather than an average

Two honest limitations, stated because they bound what this can conclude:

  1. These are books for markets open NOW, while the edge was measured on
     markets that have since resolved. Liquidity drifts, so this is an estimate
     of current cost, not a replay of historical cost.
  2. Best bid/ask is what a marketable order pays. A patient limit order inside
     the spread pays less and sometimes does not get filled at all; that
     trade-off is a live-trading question this cannot settle.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .client import CLOB, GAMMA
from .longshot import DEFAULT_BANDS, _band_labels


def _token_ids(market: dict) -> list[str]:
    raw = market.get("clobTokenIds") or market.get("clob_token_ids")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    return [str(t) for t in raw] if isinstance(raw, (list, tuple)) else []


def open_markets(client, limit: int = 500, min_volume: float = 1000.0
                 ) -> list[dict]:
    """Currently-tradeable markets, which are the only ones with a live book."""
    out = []
    try:
        batch = client._get(f"{GAMMA}/markets",
                            {"closed": "false", "active": "true",
                             "limit": min(limit, 100), "order": "volumeNum",
                             "ascending": "false"})
    except Exception:                                        # noqa: BLE001
        return out
    if not isinstance(batch, list):
        return out
    for m in batch:
        if not isinstance(m, dict):
            continue
        vol = pd.to_numeric(m.get("volumeNum"), errors="coerce")
        if pd.notna(vol) and vol < min_volume:
            continue
        out.append(m)
    return out


def sample_books(client, markets: list[dict], max_books: int = 400) -> pd.DataFrame:
    """
    Best bid, best ask and size at the touch, one row per token.

    A token with no bid or no ask is kept with a NaN spread rather than
    dropped. Those are exactly the illiquid corners where a naive filter would
    quietly delete the worst cases and flatter the average -- the same shape of
    error as `require_bid=False` silently doing nothing in the options work.
    """
    rows, seen = [], 0
    for m in markets:
        for tid in _token_ids(m):
            if seen >= max_books:
                break
            seen += 1
            try:
                b = client._get(f"{CLOB}/book", {"token_id": tid})
            except Exception:                                # noqa: BLE001
                continue
            if not isinstance(b, dict):
                continue
            bids = b.get("bids") or []
            asks = b.get("asks") or []

            def _best(levels, want_max):
                vals = []
                for lv in levels:
                    try:
                        vals.append((float(lv["price"]), float(lv["size"])))
                    except (KeyError, TypeError, ValueError):
                        continue
                if not vals:
                    return (np.nan, np.nan)
                return max(vals) if want_max else min(vals)

            bid, bid_sz = _best(bids, True)
            ask, ask_sz = _best(asks, False)
            mid = (bid + ask) / 2 if np.isfinite(bid) and np.isfinite(ask) else np.nan
            rows.append({
                "token_id": tid, "question": m.get("question"),
                "bid": bid, "ask": ask, "mid": mid,
                "half_spread_cents": (ask - bid) / 2 * 100
                if np.isfinite(mid) else np.nan,
                "depth_at_touch_usd": (bid * bid_sz if np.isfinite(bid) else 0.0)
                + (ask * ask_sz if np.isfinite(ask) else 0.0),
                "one_sided": not (np.isfinite(bid) and np.isfinite(ask)),
            })
    return pd.DataFrame(rows)


def cost_by_band(books: pd.DataFrame, bands: tuple[float, ...] = DEFAULT_BANDS
                 ) -> pd.DataFrame:
    """Measured cost and depth per price band, alongside how many books quoted."""
    if books.empty:
        return pd.DataFrame(columns=["band", "n_tokens", "median_half_spread_cents"])
    labels = _band_labels(bands)
    b = books.assign(band=pd.cut(books["mid"], bins=list(bands), labels=labels,
                                 include_lowest=True))
    b = b[b["band"].notna()]
    out = (b.groupby("band", observed=True)
            .agg(n_tokens=("token_id", "size"),
                 one_sided=("one_sided", "sum"),
                 median_half_spread_cents=("half_spread_cents", "median"),
                 p90_half_spread_cents=("half_spread_cents",
                                        lambda s: s.quantile(0.90)),
                 median_depth_usd=("depth_at_touch_usd", "median"))
            .reset_index())
    return out.round(3)


def edge_after_measured_cost(oos_by_band: dict[str, float],
                             costs: pd.DataFrame) -> pd.DataFrame:
    """
    Per-band gross edge against that band's OWN measured half-spread.

    Charging one average spread across bands hides the case that matters: the
    cheap bands carry the widest proportional spreads and are exactly where a
    longshot rule wants to trade.
    """
    rows = []
    for _, r in costs.iterrows():
        band = str(r["band"])
        gross = oos_by_band.get(band)
        if gross is None:
            continue
        cost = float(r["median_half_spread_cents"])
        rows.append({
            "band": band,
            "gross_edge_cents": round(gross, 2),
            "measured_half_spread_cents": round(cost, 2),
            "net_edge_cents": round(gross - cost, 2),
            "survives": bool(gross - cost > 0),
            "median_depth_usd": r["median_depth_usd"],
        })
    return pd.DataFrame(rows)
