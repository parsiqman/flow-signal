"""
Which Polymarket wallets are actually skilled, and which just got lucky?

This is the whole problem. Everything else in copy trading is plumbing.

The trap, quantified: scan 100,000 wallets, rank by profit, take the top 50, and
you have selected for luck with near-certainty. A wallet with zero edge making
100 bets shows a measured edge with a standard error around 5 cents per share.
The best of 100,000 such wallets sits roughly 4.8 standard errors out -- about
24 cents per share of *apparent* edge, produced entirely by chance. That looks
like a once-in-a-generation trader and it is noise.

So the headline metric here is never raw profit. It is profit measured against
what the search itself would produce from a population with no skill at all.

Three further traps specific to prediction markets:

**Win rate is meaningless.** A trader who only buys favourites at 90c wins 90%
of the time with exactly zero edge. The only thing that matters is whether they
beat the price they paid, so every metric below is (outcome - price), never
win/loss.

**Favourite-longshot bias.** Longshots are systematically overpriced in
prediction markets -- a documented, decades-old effect. A wallet that
mechanically fades longshots will look skilled while harvesting a known bias.
That is a real edge, but you should harvest it directly rather than pay latency
and slippage to copy someone else doing it. `bias_attribution` separates the two.

**Persistence is the only real test.** Rank on one period, measure on the next.
If top-decile wallets do not outperform out of sample, copy trading is dead
regardless of how good the in-sample numbers look.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# One row per fill. This is the shape Polymarket's Data API /trades returns,
# reduced to what actually matters for skill measurement.
TRADE_COLUMNS = (
    "wallet",        # trader address
    "market_id",
    "timestamp",
    "price",         # 0..1, price per share paid or received
    "size",          # shares
    "side",          # 'BUY' or 'SELL'
    "outcome",       # 1 or 0 once resolved, NaN while open
    "resolved_at",
)


def normalise_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce fills to signed long-YES exposure so every trade is comparable.

    A SELL of YES at price p is economically a BUY of NO at (1 - p). Collapsing
    both into one signed representation means the edge metric does not silently
    ignore half a wallet's activity -- which it would if only BUYs were counted,
    and that omission is invisible in the output.
    """
    out = df.copy()
    out["side"] = out["side"].astype(str).str.upper()
    is_sell = out["side"] == "SELL"
    # For a sell, the trader profits when the outcome is 0, at an effective
    # entry price of (1 - p) on the complementary claim.
    out["eff_price"] = np.where(is_sell, 1.0 - out["price"], out["price"])
    out["eff_outcome"] = np.where(is_sell, 1.0 - out["outcome"], out["outcome"])
    out["stake"] = out["eff_price"] * out["size"]
    out["profit"] = (out["eff_outcome"] - out["eff_price"]) * out["size"]
    return out


@dataclass
class WalletRecord:
    wallet: str
    n_trades: int
    n_eff: float               # Kish effective sample size, after size weighting
    n_markets: int
    total_stake: float
    total_profit: float
    edge_per_share: float      # mean(outcome - price), the only honest metric
    edge_se: float
    t_stat: float
    roi: float
    avg_price: float           # reveals a favourite or longshot preference
    first_ts: float
    last_ts: float


def score_wallets(trades: pd.DataFrame, min_trades: int = 20,
                  min_markets: int = 10) -> pd.DataFrame:
    """
    One row per wallet, with edge measured against the prices they paid.

    **Trades are aggregated to the MARKET before anything is computed**, and
    that is the whole correctness story here. A market resolves exactly once,
    so every trade a wallet makes in it shares one outcome: 42 trades in a
    single market is ONE independent bet, not 42. Treating them as 42
    observations inflates the t-statistic by sqrt(42), roughly 6.5x.

    This was not a hypothetical. The first live scan produced a t-statistic of
    124 and a "54 cents/share edge", and 7 of the 11 wallets that cleared the
    luck bar had two or fewer distinct markets between them. Synthetic fixtures
    never caught it because they spread each wallet's trades evenly across
    hundreds of markets; real traders pile into a handful.

    So: compute a size-weighted edge per (wallet, market), then treat each
    market as one observation. `min_markets` is the binding filter -- a wallet
    with nine resolved markets cannot be told apart from luck no matter how
    many times it traded them.
    """
    t = normalise_trades(trades)
    t = t[t["eff_outcome"].notna()]        # resolved markets only
    if t.empty:
        return pd.DataFrame(columns=[f.name for f in
                                     WalletRecord.__dataclass_fields__.values()])

    t["_edge"] = t["eff_outcome"] - t["eff_price"]

    # Step 1: collapse to one observation per (wallet, market). Vectorised --
    # a groupby.apply with a Python callable is ~100x slower here and made the
    # scan time out on realistic wallet counts.
    t["_we"] = t["size"] * t["_edge"]
    t["_wp"] = t["size"] * t["eff_price"]
    mk = (t.groupby(["wallet", "market_id"], sort=False)
          .agg(_we=("_we", "sum"), _wp=("_wp", "sum"), _sz=("size", "sum"),
               stake=("stake", "sum"), profit=("profit", "sum"),
               n_fills=("size", "size"))
          .reset_index())
    mk = mk[mk["_sz"] > 0]
    mk["edge"] = mk["_we"] / mk["_sz"]
    mk["price"] = mk["_wp"] / mk["_sz"]

    rows = []
    for wallet, g in mk.groupby("wallet", sort=False):
        n_fills = int(g["n_fills"].sum())
        n_mkts = len(g)
        if n_fills < min_trades or n_mkts < min_markets:
            continue
        # Weight by SIZE (shares), not stake (dollars). Stake is price x size,
        # and price is correlated with the mispricing being measured, so
        # stake-weighting drags the estimate negative by roughly sigma^2/E[p] --
        # about 12 cents at the dispersions used in the fixtures. Size carries
        # no such correlation. Caught by
        # test_fixture_actually_contains_the_edge_it_advertises.
        w = g["_sz"].to_numpy(dtype=float)
        edges = g["edge"].to_numpy(dtype=float)
        stake = float(g["stake"].sum())
        if stake <= 0 or w.sum() <= 0 or not np.isfinite(edges).all():
            continue
        edge = float(np.average(edges, weights=w))
        # Kish effective sample size over MARKETS, not fills.
        n_eff = float(w.sum() ** 2 / np.sum(w ** 2))
        sd = (float(np.sqrt(np.average((edges - edge) ** 2, weights=w)))
              if n_mkts > 1 else np.nan)
        # Variance floor. A wallet whose market-level edges happen to be nearly
        # identical produces a near-zero standard error and an astronomical
        # t-statistic from no real information. Binary outcomes cannot be more
        # precise than the binomial bound at the prices actually traded.
        price = float(np.average(g["price"], weights=w))
        floor = np.sqrt(max(price * (1 - price), 0.01)) * 0.25
        sd = max(sd, floor) if np.isfinite(sd) else floor
        se = sd / np.sqrt(n_eff) if n_eff > 1 else np.nan

        rows.append(WalletRecord(
            wallet=wallet, n_trades=n_fills, n_eff=round(n_eff, 1),
            n_markets=n_mkts,
            total_stake=stake, total_profit=float(g["profit"].sum()),
            edge_per_share=edge, edge_se=float(se) if se else np.nan,
            t_stat=float(edge / se) if se and se > 0 else np.nan,
            roi=float(g["profit"].sum() / stake) if stake > 0 else np.nan,
            avg_price=price,
            first_ts=float(t.loc[t["wallet"] == wallet, "timestamp"].min()),
            last_ts=float(t.loc[t["wallet"] == wallet, "timestamp"].max()),
        ))
    return pd.DataFrame([r.__dict__ for r in rows])


# ---------------------------------------------------------------------------
# The luck baseline
# ---------------------------------------------------------------------------

def expected_best_edge(n_wallets: int, trades_per_wallet: int,
                       avg_price: float = 0.5) -> float:
    """
    Apparent edge of the BEST wallet in a population with zero true skill.

    Any candidate must beat this before it is interesting. Under the null, a
    wallet's measured edge is approximately normal with standard error
    sqrt(p(1-p)/n), and the maximum of N draws sits about sqrt(2 ln N) standard
    errors out.

    The consequence is brutal and worth internalising: scanning more wallets
    does not improve your chances of finding skill, it improves your chances of
    finding luck that looks like skill. The bar rises with the size of the
    search, exactly as it does with a strategy parameter sweep.
    """
    if n_wallets <= 1 or trades_per_wallet < 2:
        return 0.0
    se = np.sqrt(avg_price * (1 - avg_price) / trades_per_wallet)
    return float(se * np.sqrt(2.0 * np.log(n_wallets)))


def luck_threshold_t(n_wallets: int, quantile: float = 0.95) -> float:
    """
    The t-statistic a wallet must beat, in a population of `n_wallets`, before
    luck stops being a sufficient explanation.

    Works in t-units rather than cents deliberately. Real wallet populations are
    wildly heterogeneous -- one trader has 20 fills, another has 2,000 -- and a
    threshold expressed in cents is not comparable across them. The 20-fill
    wallet needs a huge apparent edge to be interesting; the 2,000-fill wallet
    needs a modest one. Dividing by each wallet's own standard error puts them
    on one scale, and only then does a max-of-N argument apply.

    Getting this wrong was an actual error in the first version of this file:
    the per-wallet cents threshold let low-trade-count wallets through on noise.
    """
    if n_wallets <= 1:
        return 0.0
    n = float(n_wallets)
    root = np.sqrt(2.0 * np.log(n))
    a_n = root - (np.log(np.log(n)) + np.log(4 * np.pi)) / (2 * root)
    b_n = 1.0 / root
    return float(a_n + b_n * (-np.log(-np.log(quantile))))


def identifiable_edge_cents(n_wallets: int, trades_per_wallet: int,
                            avg_price: float = 0.5,
                            quantile: float = 0.95) -> float:
    """
    How big must a real edge be before it can be TOLD APART from the luckiest
    wallet in the population?

    This is the number that decides whether copy trading is possible at all, and
    it is uncomfortable. It rises with the number of wallets scanned and falls
    only with the square root of a trader's history, so scanning harder actively
    makes identification harder while doing nothing to make traders better.
    """
    if n_wallets <= 1 or trades_per_wallet < 2:
        return 0.0
    se = np.sqrt(avg_price * (1 - avg_price) / trades_per_wallet)
    return float(100.0 * se * luck_threshold_t(n_wallets, quantile))


def luck_threshold(n_wallets: int, trades_per_wallet: int,
                   avg_price: float = 0.5, quantile: float = 0.95) -> float:
    """
    The level a wallet must beat before luck is an implausible explanation.

    NOT the expected maximum -- that is the wrong bar and using it is a real
    error. The observed maximum exceeds its own mean roughly half the time, so
    gating on the expectation lets through false positives at about a 50% rate.
    Measured on a 3,000-wallet zero-skill population, the expected-max bar
    passed three wallets; every one of them was noise.

    The maximum of N normals is asymptotically Gumbel, so the correct bar is a
    high quantile of that distribution:

        a_N = sqrt(2 ln N) - (ln ln N + ln 4pi) / (2 sqrt(2 ln N))
        b_N = 1 / sqrt(2 ln N)
        threshold = a_N + b_N * (-ln(-ln q))

    Verified against simulation in test_luck_threshold_matches_simulation.
    """
    if n_wallets <= 1 or trades_per_wallet < 2:
        return 0.0
    se = np.sqrt(avg_price * (1 - avg_price) / trades_per_wallet)
    n = float(n_wallets)
    root = np.sqrt(2.0 * np.log(n))
    a_n = root - (np.log(np.log(n)) + np.log(4 * np.pi)) / (2 * root)
    b_n = 1.0 / root
    z = a_n + b_n * (-np.log(-np.log(quantile)))
    return float(se * z)


def luck_adjusted_ranking(scored: pd.DataFrame,
                          n_wallets_scanned: int | None = None) -> pd.DataFrame:
    """
    Attach the luck baseline to every wallet and flag which clear it.

    `n_wallets_scanned` should be the size of the population you SEARCHED, not
    the number that survived filtering. Filtering first and counting after is
    the most common way this correction gets quietly understated.
    """
    if scored.empty:
        return scored.assign(luck_baseline=np.nan, clears_luck=False)
    n = n_wallets_scanned if n_wallets_scanned is not None else len(scored)
    out = scored.copy()
    out["luck_expected_max"] = [
        expected_best_edge(n, int(r.n_trades), float(r.avg_price))
        for r in out.itertuples()
    ]
    # Gate on the t-statistic, not on cents, and at the 95th percentile of the
    # maximum rather than its mean. Both choices were errors in the first
    # version: a cents threshold is not comparable across wallets with different
    # history lengths, and the expected max is exceeded by chance half the time.
    out["luck_threshold"] = [
        luck_threshold(n, int(r.n_trades), float(r.avg_price), 0.95)
        for r in out.itertuples()
    ]
    out["t_needed"] = luck_threshold_t(n, 0.95)
    out["clears_luck"] = out["t_stat"] > out["t_needed"]
    return out.sort_values("edge_per_share", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# The decisive test
# ---------------------------------------------------------------------------

def persistence_test(trades: pd.DataFrame, split_ts: float | None = None,
                     min_trades_each: int = 15, top_frac: float = 0.10,
                     split_on: str = "resolved_at",
                     min_markets_each: int = 5) -> dict:
    """
    Rank wallets on the first period, measure them on the second.

    This is the experiment that decides whether copy trading is viable at all,
    and it is cheap. If wallets selected for past performance do not outperform
    the population afterwards, then past Polymarket profit carries no
    information about future profit, and no amount of execution engineering
    rescues the idea.

    `split_on='resolved_at'` is the default and matters on real data. Splitting
    on trade timestamp is lookahead: a trade PLACED in period A on a market that
    RESOLVES in period B has an outcome nobody could have known when ranking at
    the end of A. Including it means the ranking is built partly from the
    future, which is exactly the bias the test exists to rule out. A trade
    belongs to the period in which its market settled, not the one in which it
    was entered.

    Returns the out-of-sample edge of the selected group, the edge of everyone
    else, and the gap. A gap indistinguishable from zero is a kill signal.
    """
    t = normalise_trades(trades)
    t = t[t["eff_outcome"].notna()]
    if t.empty:
        return {"verdict": "no resolved trades", "n_selected": 0}

    if split_on not in t.columns:
        raise ValueError(
            f"split_on='{split_on}' is not a column. Splitting on trade time "
            f"instead of resolution time introduces lookahead; if that is "
            f"genuinely intended, pass split_on='timestamp' explicitly.")
    key = t[split_on]
    if split_ts is None:
        split_ts = float(key.quantile(0.5))

    early = t[key <= split_ts]
    late = t[key > split_ts]
    if early.empty or late.empty:
        return {"verdict": "split produced an empty period", "n_selected": 0}

    a = score_wallets(early.rename(columns={"eff_outcome": "_"}).assign(
        outcome=early["eff_outcome"], price=early["eff_price"], side="BUY"),
        min_trades=min_trades_each, min_markets=min_markets_each)
    b = score_wallets(late.rename(columns={"eff_outcome": "_"}).assign(
        outcome=late["eff_outcome"], price=late["eff_price"], side="BUY"),
        min_trades=min_trades_each, min_markets=min_markets_each)
    if a.empty or b.empty:
        return {"verdict": "too few wallets active in both periods",
                "n_selected": 0}

    both = a.merge(b, on="wallet", suffixes=("_a", "_b"))
    if len(both) < 20:
        return {"verdict": f"only {len(both)} wallets active in both periods; "
                           f"too few to conclude anything", "n_selected": len(both)}

    k = max(1, int(len(both) * top_frac))
    ranked = both.sort_values("edge_per_share_a", ascending=False)
    selected = ranked.head(k)
    rest = ranked.tail(len(ranked) - k)

    sel_edge = float(np.average(selected["edge_per_share_b"],
                                weights=selected["n_trades_b"]))
    rest_edge = float(np.average(rest["edge_per_share_b"],
                                 weights=rest["n_trades_b"])) if len(rest) else np.nan
    gap = sel_edge - rest_edge

    # Standard error of the gap, from the selected group's own dispersion.
    sd = float(np.std(selected["edge_per_share_b"], ddof=1)) if k > 1 else np.nan
    se = sd / np.sqrt(k) if np.isfinite(sd) and sd > 0 else np.nan
    t_stat = gap / se if se and se > 0 else np.nan

    # Does past rank predict future rank at all? Spearman computed directly --
    # pandas routes method="spearman" through scipy, which is not a dependency
    # here and would fail only at runtime, deep inside a long analysis.
    ra = both["edge_per_share_a"].rank().to_numpy()
    rb = both["edge_per_share_b"].rank().to_numpy()
    rank_corr = (float(np.corrcoef(ra, rb)[0, 1])
                 if ra.std() > 0 and rb.std() > 0 else np.nan)

    return {
        "n_wallets_both_periods": len(both),
        "n_selected": k,
        "selected_oos_edge": round(sel_edge, 4),
        "everyone_else_oos_edge": round(rest_edge, 4),
        "gap": round(gap, 4),
        "gap_t_stat": round(float(t_stat), 2) if np.isfinite(t_stat) else np.nan,
        "rank_correlation": round(rank_corr, 4),
        "verdict": ("past performance predicts future performance"
                    if np.isfinite(t_stat) and t_stat > 2.0 and gap > 0
                    else "NO EVIDENCE that past performance predicts future "
                         "performance -- copy trading has nothing to copy"),
    }


def bias_attribution(trades: pd.DataFrame, wallet: str) -> dict:
    """
    Is this wallet skilled, or just fading longshots?

    Longshots are systematically overpriced in prediction markets -- documented
    for decades across racetracks and betting exchanges. A wallet that
    mechanically buys favourites and sells longshots harvests that bias and
    looks talented doing it.

    That distinction is not academic. If the edge is bias harvesting, you should
    run it directly as a rule, not pay latency and slippage to copy someone
    else's version of it. Copying is a strictly worse wrapper around the same
    trade.
    """
    t = normalise_trades(trades)
    t = t[(t["wallet"] == wallet) & t["eff_outcome"].notna()]
    if t.empty:
        return {"verdict": "no resolved trades for this wallet"}

    bins = [0.0, 0.15, 0.35, 0.65, 0.85, 1.0]
    labels = ["longshot", "underdog", "toss-up", "favourite", "near-certain"]
    t = t.assign(band=pd.cut(t["eff_price"], bins=bins, labels=labels,
                             include_lowest=True))
    by_band = t.groupby("band", observed=True).apply(
        lambda g: pd.Series({
            "n": len(g),
            "stake_share": g["stake"].sum(),
            "edge": float(np.average(g["eff_outcome"] - g["eff_price"],
                                     weights=g["size"])),
        }), include_groups=False)
    if by_band.empty:
        return {"verdict": "no usable price bands"}
    by_band["stake_share"] = by_band["stake_share"] / by_band["stake_share"].sum()

    overall = float(np.average(t["eff_outcome"] - t["eff_price"], weights=t["size"]))
    # Concentration: how much of the edge comes from the extreme price bands,
    # where the documented bias lives.
    extreme = by_band.reindex(["longshot", "near-certain"]).dropna()
    extreme_share = float(extreme["stake_share"].sum()) if len(extreme) else 0.0

    return {
        "overall_edge": round(overall, 4),
        "by_price_band": by_band.round(4).to_dict("index"),
        "stake_in_extreme_bands": round(extreme_share, 3),
        "verdict": ("edge is concentrated in extreme prices -- likely harvesting "
                    "favourite-longshot bias, which you should run directly "
                    "rather than copy"
                    if extreme_share > 0.5 else
                    "edge is spread across price bands -- not obviously a "
                    "bias-harvesting rule"),
    }
