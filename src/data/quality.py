"""
Data quality gate for real option chains.

Real chains are filthy in ways that do not raise exceptions. Every check below
corresponds to a failure that produces a backtest which runs cleanly, looks
plausible, and is wrong -- the same class of bug that made the synthetic
generator report a losing strategy for the wrong reason.

Ranked by how much damage they do, worst first:

1. **Zero bids counted as sellable.** A contract with bid 0.00 has no buyer at
   any price. Booking (0 + ask)/2 as credit invents money. This is worst on
   exactly the far-OTM wings a defined-risk strategy trades.
2. **Survivorship.** A universe defined as "names optionable today" has
   silently deleted every company that went bankrupt or was acquired. For a
   SHORT VOLATILITY strategy this bias is favourable and large: the names that
   blew up are the ones missing.
3. **Adjusted options.** After a split or special dividend, contracts deliver
   non-standard amounts. No pricing model applies. They must be dropped.
4. **Stale quotes.** Illiquid contracts carry yesterday's quote, or last week's.
   The strategy sees a price nobody would trade at.
5. **Arbitrage violations.** Put-call parity breaches and non-monotonic strike
   ladders indicate bad data, bad timestamps, or a mismatched underlying price.
6. **Timestamp misalignment.** Chain snapshot and underlying close taken at
   different instants creates a small, systematic, free lookahead.

The gate returns a report and, at `severity='block'`, refuses to let the
backtest run. That is deliberate: a warning gets read once and then ignored on
every subsequent run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import dte, mid, relative_spread


@dataclass
class Finding:
    check: str
    severity: str          # 'block' | 'warn' | 'info'
    detail: str
    affected_rows: int = 0
    affected_frac: float = 0.0

    def __str__(self) -> str:
        tag = {"block": "BLOCK", "warn": "WARN ", "info": "info "}[self.severity]
        pct = f" ({self.affected_frac:.1%})" if self.affected_frac else ""
        return f"[{tag}] {self.check}: {self.detail}{pct}"


@dataclass
class QualityReport:
    findings: list[Finding] = field(default_factory=list)
    rows_in: int = 0
    rows_out: int = 0

    @property
    def blocked(self) -> bool:
        return any(f.severity == "block" for f in self.findings)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "check": f.check, "severity": f.severity, "rows": f.affected_rows,
            "frac": round(f.affected_frac, 4), "detail": f.detail,
        } for f in self.findings])

    def summary(self) -> str:
        head = (f"{self.rows_in:,} rows in, {self.rows_out:,} usable "
                f"({self.rows_out / max(self.rows_in, 1):.1%})")
        blocks = [f for f in self.findings if f.severity == "block"]
        if blocks:
            return head + f"\nBLOCKED by {len(blocks)} check(s):\n" + \
                "\n".join(f"  {b}" for b in blocks)
        warns = sum(1 for f in self.findings if f.severity == "warn")
        return head + f"\nUsable. {warns} warning(s)."


def check_chain(df: pd.DataFrame, *, risk_free: float = 0.04,
                max_unusable_frac: float = 0.60,
                min_years: float = 0.0) -> QualityReport:
    """
    Run every check. Does not modify the data -- see `clean()` for that.

    `max_unusable_frac` blocks when most of the chain is untradeable, which
    usually means the wrong file, the wrong column mapping, or a source that
    quotes only last-trade prices rather than a live market.
    """
    rep = QualityReport(rows_in=len(df))
    if df.empty:
        rep.findings.append(Finding("empty", "block", "no rows at all"))
        return rep

    n = len(df)

    # --- 1. unusable quotes -------------------------------------------------
    zero_bid = (df["bid"] <= 0).sum()
    crossed = (df["bid"] > df["ask"]).sum()
    no_ask = (df["ask"] <= 0).sum()
    rep.findings.append(Finding(
        "zero_bid", "info",
        f"{zero_bid:,} contracts have no bid and cannot be sold at any price; "
        f"they are excluded from sellable structures", zero_bid, zero_bid / n))
    if crossed:
        rep.findings.append(Finding(
            "crossed_market", "warn",
            f"{crossed:,} rows have bid > ask, which is not a real market",
            crossed, crossed / n))
    if no_ask:
        rep.findings.append(Finding(
            "no_ask", "info", f"{no_ask:,} rows have no ask", no_ask, no_ask / n))

    unusable = ((df["bid"] <= 0) | (df["ask"] <= 0) | (df["bid"] > df["ask"])).mean()
    if unusable > max_unusable_frac:
        rep.findings.append(Finding(
            "mostly_unusable", "block",
            f"{unusable:.1%} of quotes are unusable (above the "
            f"{max_unusable_frac:.0%} limit). Likely the wrong column mapping, "
            f"or a source that carries last-trade rather than quoted markets",
            int(unusable * n), unusable))

    # --- 2. survivorship ----------------------------------------------------
    rep.findings.append(_survivorship(df))

    # --- 3. adjusted contracts ---------------------------------------------
    adj = int(df["is_adjusted"].sum())
    if adj:
        rep.findings.append(Finding(
            "adjusted_options", "info",
            f"{adj:,} contracts have non-standard deliverables (post-split or "
            f"special dividend); no pricing model applies and they are dropped",
            adj, adj / n))
    elif df["underlying"].nunique() > 20:
        rep.findings.append(Finding(
            "adjusted_options", "warn",
            "no contracts flagged as adjusted across a large universe -- either "
            "the source does not report the flag, or it was not mapped. Splits "
            "happen; zero is suspicious"))

    # --- 4. stale quotes ----------------------------------------------------
    rep.findings.append(_staleness(df))

    # --- 5. structural sanity ----------------------------------------------
    rep.findings.extend(_arbitrage_checks(df, risk_free))

    # --- 6. coverage --------------------------------------------------------
    d = dte(df)
    neg = int((d < 0).sum())
    if neg:
        rep.findings.append(Finding(
            "expired_rows", "warn",
            f"{neg:,} rows have an expiry before their snapshot date", neg, neg / n))

    span_years = (df["date"].max() - df["date"].min()).days / 365.25
    if span_years < min_years:
        rep.findings.append(Finding(
            "insufficient_history", "block",
            f"only {span_years:.1f}y of data; {min_years:.1f}y required. A "
            f"short-volatility backtest without a crisis in it measures the "
            f"good half of the distribution"))
    else:
        rep.findings.append(Finding(
            "history_span", "info", f"{span_years:.1f} years of chain data"))

    iv = df["iv"]
    if iv.notna().any():
        wild = int(((iv > 5.0) | (iv < 0.01)).sum())
        if wild:
            rep.findings.append(Finding(
                "implausible_iv", "warn",
                f"{wild:,} rows have implied vol outside 1%-500%; vendor IV is "
                f"unreliable for deep ITM/OTM strikes", wild, wild / n))
    else:
        rep.findings.append(Finding(
            "no_iv", "warn",
            "no implied vol in the source; it will have to be solved from "
            "prices, which needs a dividend and rate assumption"))

    rep.rows_out = int((~_drop_mask(df)).sum())
    return rep


def _survivorship(df: pd.DataFrame) -> Finding:
    """
    Did any underlying ever leave the universe?

    A dataset where names only ever appear, never disappear, has been filtered
    to survivors. For a short-vol strategy that is a favourable bias of exactly
    the wrong kind: the deleted names are the ones that went to zero.
    """
    if df["date"].nunique() < 60:
        return Finding("survivorship", "info",
                       "too few dates to assess survivorship")
    last_date = df["date"].max()
    first_date = df["date"].min()
    early = set(df.loc[df["date"] <= first_date + pd.Timedelta(days=30),
                       "underlying"])
    late = set(df.loc[df["date"] >= last_date - pd.Timedelta(days=30),
                      "underlying"])
    if not early:
        return Finding("survivorship", "info", "insufficient early coverage")
    dropped = early - late
    frac = len(dropped) / len(early)
    if frac == 0 and len(early) > 20:
        return Finding(
            "survivorship", "block",
            f"not one of {len(early)} underlyings left the universe over the "
            f"whole sample. Real universes lose names to bankruptcy, "
            f"acquisition and delisting -- this one has been filtered to "
            f"survivors, which flatters a short-volatility strategy badly")
    return Finding(
        "survivorship", "info",
        f"{len(dropped)} of {len(early)} early underlyings absent at the end "
        f"({frac:.1%}) -- consistent with real attrition")


def _staleness(df: pd.DataFrame) -> Finding:
    """Contracts with no volume and no open interest are not real markets."""
    dead = ((df["volume"].fillna(0) == 0) & (df["open_interest"].fillna(0) == 0))
    frac = float(dead.mean())
    sev = "warn" if frac > 0.5 else "info"
    return Finding(
        "stale_contracts", sev,
        f"{int(dead.sum()):,} contracts have zero volume AND zero open "
        f"interest; their quotes are indicative at best",
        int(dead.sum()), frac)


def _arbitrage_checks(df: pd.DataFrame, risk_free: float) -> list[Finding]:
    """
    Put-call parity and intrinsic-value floors, on a sample.

    Violations mean the underlying price does not belong to the same instant as
    the chain, or the data is simply wrong. Either way the backtest inherits a
    free, invisible edge -- and free edges in backtests are always data errors.
    """
    out = []
    usable = df[mid(df).notna() & (dte(df) > 0)]
    if usable.empty:
        return [Finding("arbitrage", "warn", "no usable quotes to check")]

    m = mid(usable)
    s = usable["underlying_price"]
    k = usable["strike"]
    t = dte(usable) / 365.0
    disc = np.exp(-risk_free * t)

    is_call = usable["right"] == "C"
    intrinsic = np.where(is_call, np.maximum(s - k * disc, 0.0),
                         np.maximum(k * disc - s, 0.0))
    # 1c tolerance for rounding in reported prices.
    below = (m < intrinsic - 0.01)
    frac = float(below.mean())
    sev = "block" if frac > 0.05 else ("warn" if frac > 0.005 else "info")
    out.append(Finding(
        "below_intrinsic", sev,
        f"{int(below.sum()):,} quotes are below intrinsic value, which is "
        f"impossible in a real market and usually means the underlying price "
        f"is from a different instant than the chain",
        int(below.sum()), frac))

    # Put-call parity on matched pairs.
    piv = (usable.assign(m=m)
           .pivot_table(index=["date", "underlying", "expiry", "strike"],
                        columns="right", values="m", aggfunc="first"))
    if {"C", "P"}.issubset(piv.columns):
        pair = piv.dropna()
        if len(pair) > 50:
            # Everything below is numpy, deliberately: mixing a MultiIndexed
            # Series with a derived array lets pandas try to align on index
            # names and fail, or worse, align on partial overlap and silently
            # produce garbage.
            idx = pair.index.to_frame(index=False)
            tt = ((idx["expiry"] - idx["date"]).dt.days / 365.0).to_numpy()
            strikes = idx["strike"].to_numpy()
            spot = (usable.groupby(["date", "underlying", "expiry", "strike"])
                    ["underlying_price"].first().reindex(pair.index).to_numpy())
            resid = (pair["C"].to_numpy() - pair["P"].to_numpy()
                     - (spot - strikes * np.exp(-risk_free * tt)))
            # Scale by spot: dividends legitimately shift parity a little.
            rel = np.abs(resid) / np.maximum(spot, 1e-9)
            bad = float((rel > 0.02).mean())
            sev = "block" if bad > 0.20 else ("warn" if bad > 0.05 else "info")
            out.append(Finding(
                "put_call_parity", sev,
                f"{bad:.1%} of matched call/put pairs breach parity by more "
                f"than 2% of spot (dividends explain small deviations, not this)",
                int(bad * len(pair)), bad))
    return out


def _drop_mask(df: pd.DataFrame) -> pd.Series:
    """Rows that must never reach a backtest."""
    return (df["is_adjusted"]
            | (dte(df) < 0)
            | df["underlying_price"].isna()
            | (df["underlying_price"] <= 0)
            | (df["bid"] > df["ask"]))


def clean(df: pd.DataFrame, *, max_relative_spread: float | None = 1.0,
          require_bid: bool = True) -> pd.DataFrame:
    """
    Return only rows a strategy may legitimately trade.

    `require_bid=True` drops zero-bid contracts. Keep it on for anything that
    SELLS options: a contract with no bid cannot be sold, and pretending
    otherwise manufactures credit. Turn it off only when buying, where a zero
    bid is merely unhelpful rather than disqualifying.
    """
    out = df[~_drop_mask(df)].copy()
    if require_bid:
        out = out[out["bid"] > 0]
    out = out[out["ask"] > 0]
    if max_relative_spread is not None:
        rs = relative_spread(out)
        keep = rs.notna() & (rs <= max_relative_spread)
        if not require_bid:
            # A zero-bid contract can still be BOUGHT at the ask. Its relative
            # spread is undefined (there is no mid), so the spread test simply
            # does not apply -- dropping it here would silently make
            # require_bid=False identical to require_bid=True, which is exactly
            # the bug this branch exists to avoid.
            keep = keep | ((out["bid"] <= 0) & (out["ask"] > 0))
        out = out[keep]
    return out.reset_index(drop=True)


def assert_usable(df: pd.DataFrame, **kwargs) -> QualityReport:
    """
    Gate. Raises on any blocking finding.

    Mirrors `test_premium_is_actually_harvestable`: check the data BEFORE the
    strategy touches it, and refuse rather than warn. A warning printed at the
    top of a long run is read exactly once.
    """
    rep = check_chain(df, **kwargs)
    if rep.blocked:
        raise ValueError("Chain data failed the quality gate.\n" + rep.summary())
    return rep
