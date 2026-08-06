"""
Vendor adapters, and a fixture generator that produces realistically dirty data.

Adapters are thin by design: rename columns, fix conventions, hand off to
`schema.conform`. All the judgement lives in `quality.py`, so a new vendor
costs a dict rather than a review of the whole pipeline.

The fixture generator matters more than it sounds. Every defect `quality.py`
checks for has to be reproducible on demand, or the checks are untested
assertions about data nobody has seen. `synthetic_chain(dirty=True)` injects
zero bids, crossed markets, stale quotes, adjusted contracts, parity breaks and
a survivorship filter -- so the gate is tested against the failures it claims
to catch rather than against clean data that happens to pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import conform

# --- column maps -------------------------------------------------------------
# Keys are canonical names, values are the vendor's spelling.

DOLTHUB_OPTIONS = {
    "date": "date", "underlying": "act_symbol", "expiry": "expiration",
    "strike": "strike", "right": "call_put", "bid": "bid", "ask": "ask",
    "volume": "volume", "open_interest": "open_interest", "iv": "vol",
    "underlying_price": "underlying_price",
}

ORATS_EOD = {
    "date": "trade_date", "underlying": "ticker", "expiry": "expir_date",
    "strike": "strike", "bid": "cbid", "ask": "cask",
    "volume": "cvolume", "open_interest": "coi", "iv": "cmiv",
    "underlying_price": "stkpx",
}

POLYGON_FLAT = {
    "date": "window_start", "underlying": "underlying_ticker",
    "expiry": "expiration_date", "strike": "strike_price",
    "right": "contract_type", "bid": "bid_price", "ask": "ask_price",
    "volume": "volume", "open_interest": "open_interest",
    "iv": "implied_volatility", "underlying_price": "underlying_price",
}

VENDORS = {"dolthub": DOLTHUB_OPTIONS, "orats": ORATS_EOD, "polygon": POLYGON_FLAT}


def load_csv(path: str | Path, vendor: str | None = None,
             column_map: dict | None = None, *, strict: bool = False,
             **read_kwargs) -> pd.DataFrame:
    """
    Read a vendor CSV into the canonical schema.

    `strict=False` by default because most cheap sources omit at least one of
    volume, open interest, IV or the adjusted flag. The quality gate will warn
    about whatever is missing -- which is the right place for that judgement,
    since a missing adjusted flag is survivable and a missing bid is not.
    """
    if column_map is None:
        if vendor is None:
            raise ValueError("pass either vendor= or column_map=")
        if vendor not in VENDORS:
            raise ValueError(f"unknown vendor '{vendor}'; known: {sorted(VENDORS)}")
        column_map = VENDORS[vendor]
    raw = pd.read_csv(path, **read_kwargs)
    return from_frame(raw, column_map, strict=strict)


def from_frame(raw: pd.DataFrame, column_map: dict,
               *, strict: bool = False) -> pd.DataFrame:
    """Apply a column map to an in-memory frame. Used by every adapter."""
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename = {v.lower(): k for k, v in column_map.items() if v.lower() in df.columns}
    df = df.rename(columns=rename)

    # Vendors spell the option right half a dozen ways.
    if "right" in df.columns:
        df["right"] = (df["right"].astype(str).str.upper()
                       .str.replace("CALL", "C", regex=False)
                       .str.replace("PUT", "P", regex=False).str[0])
    return conform(df, strict=strict)


def load_orats_pair(raw: pd.DataFrame) -> pd.DataFrame:
    """
    ORATS ships calls and puts as columns on one row; unstack them.

    Vendors that put both rights on a single row are a common shape and a
    common source of silent error: mapping only the call columns produces a
    dataset that looks complete and contains no puts at all.
    """
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    frames = []
    for right, (b, a, v, oi, iv) in {
        "C": ("cbid", "cask", "cvolume", "coi", "cmiv"),
        "P": ("pbid", "pask", "pvolume", "poi", "pmiv"),
    }.items():
        need = {"trade_date", "ticker", "expir_date", "strike", "stkpx", b, a}
        if not need.issubset(df.columns):
            continue
        part = pd.DataFrame({
            "date": df["trade_date"], "underlying": df["ticker"],
            "expiry": df["expir_date"], "strike": df["strike"],
            "right": right, "bid": df[b], "ask": df[a],
            "volume": df.get(v, 0.0), "open_interest": df.get(oi, 0.0),
            "iv": df.get(iv, np.nan), "underlying_price": df["stkpx"],
        })
        frames.append(part)
    if not frames:
        raise ValueError("no ORATS call or put columns found; check the export")
    return conform(pd.concat(frames, ignore_index=True), strict=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def synthetic_chain(n_underlyings: int = 6, n_days: int = 120,
                    seed: int = 0, dirty: bool = False,
                    survivorship_filtered: bool = False,
                    vrp: float = 0.0, term_slope: float = 0.0) -> pd.DataFrame:
    """
    A small, structurally valid option chain, optionally corrupted on purpose.

    Clean output is arbitrage-consistent by construction (priced with
    Black-Scholes off a real vol path), so anything the quality gate flags on
    `dirty=False` is a false positive in the gate.

    `dirty=True` injects, in order of how much damage each does in the wild:
      zero bids on far-OTM contracts, crossed markets, stale zero-activity
      quotes, adjusted (post-split) contracts, and put-call parity breaks
      caused by a mismatched underlying timestamp.

    `survivorship_filtered=True` deletes every underlying that would have left
    the universe, reproducing the single most flattering bias for a short-vol
    strategy.

    `vrp` inflates quoted implied vol relative to the volatility that actually
    generates returns, so `chain.measure_vrp` can be tested in both directions:
    at vrp=0 a seller loses the spread and capture must be negative; at vrp=0.2
    capture must come out clearly positive. Without both cases the kill
    criterion is an untested assertion.

    `term_slope` makes implied vol depend on tenor, which is what makes
    constant-maturity interpolation do anything at all. At zero it is a no-op
    and any test of the interpolation is vacuous.
    """
    rng = np.random.default_rng(seed)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from options_alpha.synthetic import bs_price_scalar

    start = pd.Timestamp("2021-01-04")
    dates = pd.bdate_range(start, periods=n_days)
    names = [f"UND{i}" for i in range(n_underlyings)]
    rows = []

    spots = {n: float(rng.uniform(40, 180)) for n in names}
    vols = {n: float(rng.uniform(0.20, 0.55)) for n in names}
    # A name that leaves the universe partway through, as real ones do.
    delisted = names[-1] if n_underlyings > 2 else None
    delist_at = n_days // 2

    for di, d in enumerate(dates):
        for n in names:
            # Survivorship bias removes the name from the WHOLE history, not
            # just from the point it delisted -- because the universe was
            # defined as "names optionable today" and this one is not. That is
            # precisely why the bias is invisible: there is no gap to notice.
            if delisted == n:
                if survivorship_filtered:
                    continue
                if di >= delist_at:
                    continue
            vols[n] = float(np.clip(vols[n] * np.exp(rng.normal(0, 0.03)), 0.1, 1.5))
            spots[n] = float(spots[n] * np.exp(rng.normal(0, vols[n] / 16)))
            s, v = spots[n], vols[n]

            for tenor in (21, 49):
                expiry = d + pd.Timedelta(days=tenor)
                t = tenor / 365.0
                step = 2.5 if s < 100 else 5.0
                base = round(s / step) * step
                # Quoted vol carries the premium and the term slope; the vol
                # that actually generates returns does not. That gap IS the
                # variance risk premium being measured downstream.
                quoted_base = v * (1.0 + vrp) * (1.0 + term_slope * (tenor - 30) / 30)
                for off in range(-4, 5):
                    k = base + off * step
                    if k <= 0:
                        continue
                    # Skew: downside strikes carry higher implied vol.
                    kv = quoted_base * (1.0 - 0.35 * np.log(k / s))
                    kv = float(np.clip(kv, 0.05, 3.0))
                    for right in ("C", "P"):
                        theo = bs_price_scalar(s, k, t, kv, right == "C", 0.04)
                        half = max(0.02, theo * 0.03)
                        rows.append({
                            "date": d, "underlying": n, "expiry": expiry,
                            "strike": k, "right": right,
                            "bid": max(theo - half, 0.0), "ask": theo + half,
                            "volume": float(rng.integers(0, 500)),
                            "open_interest": float(rng.integers(0, 5000)),
                            "iv": kv, "underlying_price": s,
                            "is_adjusted": False,
                        })

    df = pd.DataFrame(rows)
    if not dirty:
        return conform(df, strict=True)

    n = len(df)
    idx = df.index.to_numpy()

    # 1. Zero bids on cheap far-OTM contracts -- the ones a wing would use.
    cheap = df.index[df["bid"] < 0.15]
    df.loc[rng.choice(cheap, size=min(len(cheap), n // 12), replace=False),
           "bid"] = 0.0

    # 2. Crossed markets.
    crossed = rng.choice(idx, size=n // 200, replace=False)
    df.loc[crossed, "bid"] = df.loc[crossed, "ask"] + 0.05

    # 3. Stale: no volume, no open interest.
    stale = rng.choice(idx, size=n // 6, replace=False)
    df.loc[stale, ["volume", "open_interest"]] = 0.0

    # 4. Adjusted contracts on one name.
    adj_name = df["underlying"].unique()[0]
    adj = df.index[(df["underlying"] == adj_name)
                   & (df["date"] > dates[n_days // 3])]
    df.loc[adj, "is_adjusted"] = True
    df.loc[adj, "strike"] = df.loc[adj, "strike"] * 0.7    # odd deliverable

    # 5. Parity breaks from a mismatched underlying timestamp.
    bad_day = dates[n_days // 4]
    m = df["date"] == bad_day
    df.loc[m, "underlying_price"] = df.loc[m, "underlying_price"] * 1.06

    return conform(df, strict=True)


def write_fixture(path: str | Path, **kwargs) -> Path:
    """Persist a fixture chain to CSV, for notebook and integration use."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    synthetic_chain(**kwargs).to_csv(p, index=False)
    return p
