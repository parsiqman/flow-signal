"""
The canonical option-chain schema every loader must produce.

One schema, so that vendor quirks are handled once at the edge and nothing
downstream ever asks "which format is this?". Adding a vendor means writing an
adapter, not touching the strategy.

Real option chains carry traps that synthetic data does not, and most of them
are silent. The three that matter most, encoded in this schema:

  - `is_adjusted`: options on a stock that split or paid a special dividend
    have non-standard deliverables (e.g. 100 shares plus cash). Their quoted
    prices are not comparable to standard contracts and no pricing model
    applies. They must be dropped, not modelled.
  - `bid` and `ask` are kept separate from any mid. A zero bid means the
    contract cannot be sold at all -- which a mid price hides completely, and
    which would let a backtest book credit it could never have collected.
  - `underlying_price` is stored per row rather than joined later, because the
    snapshot time of the chain and the close of the underlying are often not
    the same instant, and joining them silently introduces lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# One row per (date, underlying, expiry, strike, right).
CONTRACT_COLUMNS = {
    "date": "datetime64[ns]",      # snapshot date
    "underlying": "object",        # ticker
    "expiry": "datetime64[ns]",
    "strike": "float64",
    "right": "object",             # 'C' or 'P'
    "bid": "float64",
    "ask": "float64",
    "volume": "float64",
    "open_interest": "float64",
    "iv": "float64",               # vendor-computed implied vol, may be NaN
    "underlying_price": "float64",
    "is_adjusted": "bool",         # non-standard deliverable
}

REQUIRED = tuple(CONTRACT_COLUMNS)

# The panel the strategy pipeline consumes. Deliberately identical to what
# `options_alpha.synthetic.generate_market` emits, so real and synthetic data
# are interchangeable and every downstream test applies to both.
PANEL_COLUMNS = ("day", "name", "spot", "atm_iv", "days_to_earnings")


@dataclass
class ChainStats:
    rows: int
    underlyings: int
    dates: int
    first_date: pd.Timestamp | None
    last_date: pd.Timestamp | None
    years: float

    def __str__(self) -> str:
        return (f"{self.rows:,} rows | {self.underlyings} underlyings | "
                f"{self.dates:,} dates | {self.first_date.date() if self.first_date else '?'}"
                f" to {self.last_date.date() if self.last_date else '?'} "
                f"({self.years:.1f}y)")


def conform(df: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """
    Coerce a loaded frame to the canonical schema, or say exactly what is wrong.

    `strict=False` fills genuinely optional columns (volume, open_interest, iv,
    is_adjusted) with sensible defaults, because some cheap sources omit them.
    It never invents bid, ask, strike, expiry or underlying_price -- a backtest
    without those is not a backtest.
    """
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]

    optional_defaults = {
        "volume": 0.0, "open_interest": 0.0, "iv": np.nan, "is_adjusted": False,
    }
    essential = [c for c in REQUIRED if c not in optional_defaults]

    missing_essential = [c for c in essential if c not in out.columns]
    if missing_essential:
        raise ValueError(
            f"Chain data is missing essential columns {missing_essential}. "
            f"Present: {sorted(out.columns)}. These cannot be defaulted -- a "
            f"backtest without real bids, asks and expiries is not a backtest.")

    for col, default in optional_defaults.items():
        if col not in out.columns:
            if strict:
                raise ValueError(
                    f"Column '{col}' is missing. Pass strict=False to default it "
                    f"to {default!r}, but know that you are doing so.")
            out[col] = default

    out["date"] = pd.to_datetime(out["date"])
    out["expiry"] = pd.to_datetime(out["expiry"])
    out["right"] = out["right"].astype(str).str.upper().str[0]
    for c in ("strike", "bid", "ask", "volume", "open_interest", "iv",
              "underlying_price"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["is_adjusted"] = out["is_adjusted"].astype(bool)
    out["underlying"] = out["underlying"].astype(str).str.upper()

    bad_right = ~out["right"].isin(["C", "P"])
    if bad_right.any():
        raise ValueError(f"{bad_right.sum()} rows have a right that is neither "
                         f"C nor P: {sorted(out.loc[bad_right, 'right'].unique())[:5]}")

    return out[list(REQUIRED)]


def dte(df: pd.DataFrame) -> pd.Series:
    """Calendar days to expiry. Trading-day conversion happens downstream."""
    return (df["expiry"] - df["date"]).dt.days


def mid(df: pd.DataFrame) -> pd.Series:
    """
    Mid price, NaN where the market is unusable.

    Returns NaN rather than a number for crossed markets and zero bids. A mid
    computed from a zero bid is the single most common way a backtest books
    credit it could never have received: the contract had no buyer at any
    price, but (0 + ask)/2 looks like a perfectly tradeable quote.
    """
    m = (df["bid"] + df["ask"]) / 2.0
    unusable = (df["bid"] <= 0) | (df["ask"] <= 0) | (df["bid"] > df["ask"])
    return m.where(~unusable, np.nan)


def relative_spread(df: pd.DataFrame) -> pd.Series:
    """Bid-ask spread as a fraction of mid. Infinite where unusable."""
    m = mid(df)
    return ((df["ask"] - df["bid"]) / m).replace([np.inf, -np.inf], np.nan)


def describe(df: pd.DataFrame) -> ChainStats:
    dates = df["date"].nunique()
    lo = df["date"].min() if len(df) else None
    hi = df["date"].max() if len(df) else None
    years = ((hi - lo).days / 365.25) if (lo is not None and hi is not None) else 0.0
    return ChainStats(rows=len(df), underlyings=df["underlying"].nunique(),
                      dates=dates, first_date=lo, last_date=hi, years=years)
