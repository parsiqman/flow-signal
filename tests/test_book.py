"""
Tests for the order-book cost measurement.

The rule's whole result turns on one number -- what crossing costs -- and the
walk-forward used an assumed 1c against a 6.27c break-even. These tests guard
the ways a measured spread can flatter itself.

    python tests/test_book.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polymarket import book  # noqa: E402


class _Books:
    """A venue whose favourites are tight and whose longshots are not."""

    def __init__(self):
        self.markets = [{"question": f"q{i}", "volumeNum": 50_000.0,
                         "clobTokenIds": f'["t{i}a", "t{i}b"]'} for i in range(12)]

    def _get(self, url, params):
        if "/markets" in url:
            return self.markets
        tid = params["token_id"]
        cheap = tid.endswith("b")
        if cheap:                       # a longshot: wide, thin
            return {"bids": [{"price": "0.06", "size": "300"}],
                    "asks": [{"price": "0.12", "size": "300"}]}
        return {"bids": [{"price": "0.88", "size": "4000"}],   # favourite: tight
                "asks": [{"price": "0.90", "size": "4000"}]}


def test_spread_is_measured_per_band_not_averaged_across_them():
    """
    One average spread hides the case that decides the rule: cheap bands carry
    the widest proportional spreads and are exactly where a longshot rule wants
    to trade.
    """
    api = _Books()
    books = book.sample_books(api, api.markets)
    costs = book.cost_by_band(books)
    cheap = costs[costs["band"] == "0.05-0.10"]
    dear = costs[costs["band"] == "0.80-0.90"]
    assert float(cheap["median_half_spread_cents"].iloc[0]) == 3.0
    assert float(dear["median_half_spread_cents"].iloc[0]) == 1.0


def test_a_wide_band_can_fail_while_a_tight_one_passes():
    api = _Books()
    costs = book.cost_by_band(book.sample_books(api, api.markets))
    net = book.edge_after_measured_cost({"0.05-0.10": 2.0, "0.80-0.90": 8.54},
                                        costs)
    by = {r["band"]: r for _, r in net.iterrows()}
    assert by["0.05-0.10"]["survives"] is False      # 2.0c edge, 3.0c spread
    assert by["0.80-0.90"]["survives"] is True       # 8.54c edge, 1.0c spread


def test_one_sided_books_are_kept_and_counted_not_silently_dropped():
    """
    Dropping tokens with no bid deletes the worst liquidity from the average
    and flatters it. The same shape of error as a spread filter that silently
    removed the zero-bid rows it was supposed to keep.
    """
    class _Missing(_Books):
        def _get(self, url, params):
            if "/markets" in url:
                return self.markets
            return {"bids": [], "asks": [{"price": "0.10", "size": "100"}]}

    api = _Missing()
    books = book.sample_books(api, api.markets)
    assert len(books) > 0
    assert books["one_sided"].all()
    assert books["half_spread_cents"].isna().all()


def test_depth_at_the_touch_is_reported_because_capacity_is_the_real_limit():
    api = _Books()
    costs = book.cost_by_band(book.sample_books(api, api.markets))
    assert "median_depth_usd" in costs.columns
    dear = costs[costs["band"] == "0.80-0.90"]["median_depth_usd"].iloc[0]
    cheap = costs[costs["band"] == "0.05-0.10"]["median_depth_usd"].iloc[0]
    assert dear > cheap, "favourites should show more money at the touch"


def test_books_are_capped_so_a_probe_cannot_run_away():
    api = _Books()
    books = book.sample_books(api, api.markets, max_books=5)
    assert len(books) <= 5


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
