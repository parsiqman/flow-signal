"""
The favourite-longshot rule, run directly instead of copied.

The wallet scan found a real edge and then told us not to rent it: 83% of that
trader's stake sat in the extreme price bands, which is favourite-longshot
harvesting -- a documented structural bias, not private information. Copying it
paid 2.93c of cross-venue slippage against a 4.45c edge. Running the rule
yourself pays the spread once and keeps the rest.

**Who is on the other side, and why they keep taking it.** Longshots are
lottery-shaped: a small stake with a large, salient, low-probability payoff.
People overpay for that shape, consistently, and have done so in racetrack
betting since Griffith (1949) and in every betting exchange studied since. The
counterparty is not a mispricing that arbitrage will close -- it is a
preference. That is why it persists, and it is also why it is capacity-limited:
the money on the other side is small and recreational.

**What this module refuses to do.** It does not fit a curve and report the fit.
The calibration is estimated on one period and evaluated, untouched, on a later
one, and the evaluation is reported net of the spread you would actually cross.
An in-sample calibration curve of prediction-market prices always looks
profitable, because prices are noisy and the noise is mean-reverting by
construction.

Two hazards specific to this measurement, both of which produce a confident
positive if ignored:

  1. **Market clustering.** A market resolves once, so every fill in it shares
     one outcome. 400 fills in one market is ONE observation. Ignoring this
     inflated a t-statistic to 124 earlier in this project.
  2. **The spread is proportionally brutal where the edge is.** A 1c spread on
     a 5c token is 20% of the notional. An edge measured on mid prices in the
     longshot band can be entirely fictional at the touch. `evaluate` therefore
     takes a cost model and reports the break-even spread alongside the edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .wallets import normalise_trades, _kish_n_eff

# Bands are fixed here, in code, BEFORE looking at any result. Choosing cut
# points after seeing which ones pay is the purest form of the search problem
# this repo exists to control: with free choice of boundaries a null tape can
# be made to show any edge you like.
DEFAULT_BANDS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50,
                                    0.65, 0.80, 0.90, 0.95, 1.0)


def _band_labels(edges: tuple[float, ...]) -> list[str]:
    return [f"{a:.2f}-{b:.2f}" for a, b in zip(edges, edges[1:])]


def calibrate(trades: pd.DataFrame,
              bands: tuple[float, ...] = DEFAULT_BANDS) -> pd.DataFrame:
    """
    Realised payoff frequency against price paid, one observation per market.

    Each fill is a claim bought at `eff_price` that paid `eff_outcome`. If
    prices were unbiased, the average payoff inside a price band would equal
    the average price in it. Favourite-longshot bias predicts a specific
    signature: payoffs BELOW price at the cheap end (longshots overpriced) and
    payoffs ABOVE price at the dear end (favourites underpriced).

    Fills are collapsed to one observation per (market, band) first. Without
    that, a single heavily-traded market contributes hundreds of correlated
    rows and every standard error below is wrong by an order of magnitude.
    """
    t = normalise_trades(trades)
    t = t[t["eff_outcome"].notna() & t["size"].gt(0)]
    t = t[(t["eff_price"] > 0) & (t["eff_price"] < 1)]
    if t.empty:
        return pd.DataFrame(columns=["band", "n_markets", "n_eff", "avg_price",
                                     "realised_freq", "edge", "se", "t_stat"])

    labels = _band_labels(bands)
    t = t.assign(band=pd.cut(t["eff_price"], bins=list(bands), labels=labels,
                             include_lowest=True))
    t = t[t["band"].notna()]
    t["_wp"] = t["eff_price"] * t["size"]
    t["_wo"] = t["eff_outcome"] * t["size"]

    per = (t.groupby(["market_id", "band"], observed=True)
             .agg(_wp=("_wp", "sum"), _wo=("_wo", "sum"), _sz=("size", "sum"))
             .reset_index())
    per["price"] = per["_wp"] / per["_sz"]
    per["payoff"] = per["_wo"] / per["_sz"]

    rows = []
    for band, g in per.groupby("band", observed=True):
        w = g["_sz"].to_numpy(dtype=float)
        price = float(np.average(g["price"], weights=w))
        payoff = float(np.average(g["payoff"], weights=w))
        n_eff = _kish_n_eff(w)
        # Binomial standard error at the observed frequency, on the EFFECTIVE
        # sample. A floor on the variance stops a band whose markets all
        # resolved the same way from reporting an infinite t-statistic.
        var = max(payoff * (1.0 - payoff), 0.01)
        se = float(np.sqrt(var / max(n_eff, 1.0)))
        edge = payoff - price
        rows.append({
            "band": band, "n_markets": int(len(g)), "n_eff": round(n_eff, 1),
            "avg_price": round(price, 4), "realised_freq": round(payoff, 4),
            "edge": round(edge, 4), "se": round(se, 4),
            "t_stat": round(edge / se, 2) if se > 0 else np.nan,
        })
    out = pd.DataFrame(rows)
    return out.sort_values("avg_price").reset_index(drop=True)


@dataclass
class LongshotRule:
    """
    Which price bands to take, and in which direction.

    `min_t` is the bar a band's in-sample edge must clear to be included, and
    it is corrected for the number of bands examined. Fitting ten bands and
    keeping whichever look good is ten tests, not one.
    """

    bands: tuple[float, ...]
    side: dict[str, int] = field(default_factory=dict)   # band -> +1 buy, -1 sell
    fitted_edge: dict[str, float] = field(default_factory=dict)
    min_t: float = 2.0
    n_bands_tested: int = 0

    def is_empty(self) -> bool:
        return not self.side

    def describe(self) -> str:
        if self.is_empty():
            return "no band cleared the in-sample bar; the rule trades nothing"
        parts = [f"{'BUY ' if s > 0 else 'SELL'} {b} "
                 f"(fitted {self.fitted_edge[b] * 100:+.2f}c)"
                 for b, s in sorted(self.side.items())]
        return "; ".join(parts)


def fit(train: pd.DataFrame, bands: tuple[float, ...] = DEFAULT_BANDS,
        quantile: float = 0.95) -> LongshotRule:
    """
    Select the bands whose in-sample edge survives the multiple-testing bar.

    A band is taken LONG when claims in it pay more often than they cost, and
    SHORT when they pay less. Bands that do not clear the bar are left alone
    rather than traded small: a band with no measurable edge contributes only
    spread cost, and including it purely to look diversified is a way of paying
    to add noise.
    """
    from .wallets import luck_threshold_t

    cal = calibrate(train, bands)
    n_tested = int(len(cal))
    bar = float(luck_threshold_t(max(n_tested, 1), quantile=quantile))
    rule = LongshotRule(bands=bands, min_t=bar, n_bands_tested=n_tested)
    for _, r in cal.iterrows():
        t = r["t_stat"]
        if not np.isfinite(t) or abs(t) < bar:
            continue
        rule.side[str(r["band"])] = 1 if r["edge"] > 0 else -1
        rule.fitted_edge[str(r["band"])] = float(r["edge"])
    return rule


def evaluate(rule: LongshotRule, test: pd.DataFrame,
             half_spread_cents: float = 1.0,
             fee_cents: float = 0.0) -> dict:
    """
    Apply a FIXED rule to data it was not fitted on, net of execution cost.

    `half_spread_cents` is what you give up crossing to get filled, in cents
    per share. It is charged on entry only, because a claim held to resolution
    is settled by the exchange rather than sold back -- which is the one
    genuinely nice feature of prediction markets as a venue.

    Returns the out-of-sample edge before and after cost, the t-statistic on
    the after-cost number, and the break-even spread. If the break-even spread
    is below what the book actually shows in these bands, the rule does not
    exist no matter how good the gross edge looks.
    """
    if rule.is_empty():
        return {"verdict": "rule is empty; nothing to evaluate", "n_markets": 0}

    t = normalise_trades(test)
    t = t[t["eff_outcome"].notna() & t["size"].gt(0)]
    t = t[(t["eff_price"] > 0) & (t["eff_price"] < 1)]
    if t.empty:
        return {"verdict": "no resolved trades in the test period", "n_markets": 0}

    labels = _band_labels(rule.bands)
    t = t.assign(band=pd.cut(t["eff_price"], bins=list(rule.bands), labels=labels,
                             include_lowest=True))
    t = t[t["band"].astype(str).isin(rule.side)]
    if t.empty:
        return {"verdict": "no test trades fell in the rule's bands",
                "n_markets": 0}

    t["_side"] = t["band"].astype(str).map(rule.side).astype(float)
    # Taking a band SHORT means buying the complementary claim: price -> 1-p,
    # payoff -> 1-payoff. Same arithmetic the wallet normalisation uses.
    t["_p"] = np.where(t["_side"] > 0, t["eff_price"], 1.0 - t["eff_price"])
    t["_o"] = np.where(t["_side"] > 0, t["eff_outcome"], 1.0 - t["eff_outcome"])
    t["_pnl"] = t["_o"] - t["_p"]

    per = (t.assign(_wp=t["_p"] * t["size"], _wo=t["_o"] * t["size"])
            .groupby("market_id", observed=True)
            .agg(_wp=("_wp", "sum"), _wo=("_wo", "sum"), _sz=("size", "sum"))
            .reset_index())
    w = per["_sz"].to_numpy(dtype=float)
    price = per["_wp"].to_numpy() / w
    payoff = per["_wo"].to_numpy() / w
    gross = float(np.average(payoff - price, weights=w))

    cost = (half_spread_cents + fee_cents) / 100.0
    net = gross - cost
    n_eff = _kish_n_eff(w)
    per_market = payoff - price
    sd = float(np.sqrt(np.average((per_market - gross) ** 2, weights=w)))
    sd = max(sd, 0.05)                      # variance floor, as in score_wallets
    se = sd / np.sqrt(max(n_eff, 1.0))
    avg_price = float(np.average(price, weights=w))

    return {
        "n_markets": int(len(per)),
        "n_eff": round(n_eff, 1),
        "avg_price": round(avg_price, 4),
        "gross_edge_cents": round(gross * 100, 2),
        "cost_cents": round(cost * 100, 2),
        "net_edge_cents": round(net * 100, 2),
        "t_stat_net": round(net / se, 2) if se > 0 else np.nan,
        "roi_net": round(net / avg_price, 4) if avg_price > 0 else np.nan,
        "breakeven_half_spread_cents": round(gross * 100, 2),
        "verdict": ("net edge survives the spread" if net > 0
                    else "the spread eats the entire edge"),
    }


def null_check(rule: LongshotRule, test: pd.DataFrame, n_draws: int = 200,
               half_spread_cents: float = 1.0, seed: int = 0) -> dict:
    """
    Re-run the rule on tapes where prices are HONEST by construction.

    The null hypothesis has to be "prices are well calibrated", not "price and
    outcome are unrelated". The first attempt here permuted outcomes across
    markets, which destroys calibration entirely: a rule buying favourites at
    93c then wins about half the time instead of 93% of the time, so the null
    distribution sits near -43 cents and the real edge beats it by a mile no
    matter what the real edge is. A test the strategy cannot fail measures
    nothing.

    So each market's outcome is redrawn as Bernoulli(price paid). Price
    distribution, sizes and market clustering are all preserved; the only thing
    removed is the bias being hunted. The rule's own band selection is held
    fixed, because re-fitting on each draw would test a different question.
    """
    rng = np.random.default_rng(seed)
    real = evaluate(rule, test, half_spread_cents=half_spread_cents)
    if "net_edge_cents" not in real:
        return {"verdict": "rule could not be evaluated on the real tape"}

    # Outcomes are a property of the MARKET, and a binary market has TWO
    # tokens whose fates are opposite. Fills within a market split into at most
    # two groups by which token they bought, and `outcome` is constant inside a
    # group. Drawing one Bernoulli per market and broadcasting it to every fill
    # -- the first version here -- gave both tokens the same fate, which is
    # impossible, and put the null mean at -38 cents instead of zero. That is
    # the SECOND time this null has been centred somewhere the strategy cannot
    # fail; the giveaway both times was a null mean far from zero.
    t = normalise_trades(test)
    t = t[t["size"].gt(0) & t["outcome"].notna()]
    if t.empty:
        return {"verdict": "no resolved trades to build a null from"}

    # Recover the two token groups per market and each group's traded price.
    grp = (t.assign(_wp=t["price"] * t["size"])
            .groupby(["market_id", "outcome"])
            .agg(_wp=("_wp", "sum"), _sz=("size", "sum")))
    grp["price"] = grp["_wp"] / grp["_sz"]

    winners: dict = {}
    for mid, g in grp.groupby(level=0):
        legs = g.reset_index()
        # Probability the actually-winning token wins, under honest prices.
        row = legs[legs["outcome"] == 1.0]
        p_win = float(row["price"].iloc[0]) if len(row) else float(
            1.0 - legs["price"].iloc[0])
        winners[mid] = min(max(p_win, 0.001), 0.999)

    mids = list(winners)
    probs = np.array([winners[m] for m in mids], dtype=float)

    draws = []
    for _ in range(n_draws):
        # One draw per market decides whether the token that really won wins
        # again. Every fill keeps its own side, so the two tokens stay
        # opposite and the price distribution is untouched.
        keeps = pd.Series((rng.random(len(mids)) < probs), index=mids)
        fake = test.copy()
        flip = ~fake["market_id"].map(keeps).astype(bool)
        fake["outcome"] = np.where(flip, 1.0 - fake["outcome"], fake["outcome"])
        r = evaluate(rule, fake, half_spread_cents=half_spread_cents)
        if "net_edge_cents" in r:
            draws.append(r["net_edge_cents"])
    if not draws:
        return {"verdict": "null draws produced nothing evaluable"}
    draws = np.asarray(draws, dtype=float)
    p = float((draws >= real["net_edge_cents"]).mean())
    return {
        "real_net_edge_cents": real["net_edge_cents"],
        "null_mean_cents": round(float(draws.mean()), 3),
        "null_sd_cents": round(float(draws.std(ddof=1)), 3),
        "p_value": round(p, 4),
        "n_draws": int(len(draws)),
        "verdict": ("edge is within what honest prices produce by chance"
                    if p > 0.05 else
                    "edge exceeds what honest prices produce by chance"),
    }


def walk_forward(trades: pd.DataFrame, bands: tuple[float, ...] = DEFAULT_BANDS,
                 half_spread_cents: float = 1.0,
                 split_on: str = "resolved_at") -> dict:
    """
    Fit on the first half of history by resolution time, evaluate on the second.

    Splitting on resolution rather than trade time for the reason stated
    throughout this repo: a trade entered before the cut on a market that
    settles after it has an outcome nobody could have known at fitting time.
    """
    if split_on not in trades.columns:
        raise ValueError(f"{split_on!r} not in trades; have {list(trades.columns)}")
    per_market = trades.groupby("market_id")[split_on].min().sort_values()
    if len(per_market) < 40:
        return {"verdict": f"only {len(per_market)} markets; too few to split"}
    cut = float(per_market.iloc[len(per_market) // 2])
    early = set(per_market[per_market < cut].index)
    train = trades[trades["market_id"].isin(early)]
    test = trades[~trades["market_id"].isin(early)]

    rule = fit(train, bands)
    cal_train = calibrate(train, bands)
    cal_test = calibrate(test, bands)
    out = {
        "split_ts": cut,
        "n_markets_train": int(len(early)),
        "n_markets_test": int(len(per_market) - len(early)),
        "rule": rule.describe(),
        "bar_used": round(rule.min_t, 2),
        "n_bands_tested": rule.n_bands_tested,
        "calibration_train": cal_train,
        "calibration_test": cal_test,
    }
    out["out_of_sample"] = evaluate(rule, test,
                                    half_spread_cents=half_spread_cents)
    return out


# Favourite-longshot bias, as measured in the literature, expressed as the
# pricing error at the extremes. Racetrack and exchange studies put it in the
# low single digits of a cent to under ten cents. This is the effect the test
# has to be able to SEE; a run that cannot resolve it has not tested anything.
DOCUMENTED_EFFECT_CENTS = (2.0, 8.0)


def minimum_detectable_edge(cal: pd.DataFrame, bar: float) -> pd.DataFrame:
    """
    The smallest edge each band could have detected, in cents.

    This is the question that has to be asked BEFORE reading a null result,
    and not asking it is how this project has repeatedly manufactured
    confident negatives. A band with an effective sample of 30 has a standard
    error near 9 cents; against a corrected bar of 2.57 it cannot see anything
    smaller than 22 cents. Favourite-longshot bias is 2-8 cents. Such a band
    reporting "no edge" has reported nothing at all.
    """
    out = cal.copy()
    out["mde_cents"] = (bar * out["se"] * 100).round(2)
    out["can_see_documented_effect"] = out["mde_cents"] <= DOCUMENTED_EFFECT_CENTS[1]
    return out


def power_verdict(cal: pd.DataFrame, bar: float) -> dict:
    """
    Is a null result from this sample informative, or just small?

    Returns `underpowered=True` when the typical band cannot resolve even the
    upper end of the documented effect. In that case the only honest report is
    that the question was not answered -- NOT that the bias is absent.
    """
    mde = minimum_detectable_edge(cal, bar)
    if mde.empty:
        return {"underpowered": True, "verdict": "no bands to assess"}
    med = float(mde["mde_cents"].median())
    n_ok = int(mde["can_see_documented_effect"].sum())
    under = med > DOCUMENTED_EFFECT_CENTS[1]
    return {
        "median_mde_cents": round(med, 2),
        "bands_that_can_see_the_effect": n_ok,
        "n_bands": int(len(mde)),
        "documented_effect_cents": list(DOCUMENTED_EFFECT_CENTS),
        "underpowered": bool(under),
        "verdict": (f"UNDERPOWERED: the typical band cannot resolve an edge "
                    f"below {med:.1f}c, and the effect being hunted is "
                    f"{DOCUMENTED_EFFECT_CENTS[0]:.0f}-"
                    f"{DOCUMENTED_EFFECT_CENTS[1]:.0f}c. A null here means the "
                    f"sample is too small, not that the bias is absent."
                    if under else
                    f"adequately powered: the typical band resolves down to "
                    f"{med:.1f}c, inside the {DOCUMENTED_EFFECT_CENTS[0]:.0f}-"
                    f"{DOCUMENTED_EFFECT_CENTS[1]:.0f}c effect range"),
    }


def loss_correlation(rule: LongshotRule, trades: pd.DataFrame,
                     bucket_seconds: float = 86_400.0,
                     group_on: str = "resolved_at") -> dict:
    """
    Intraclass correlation of per-market rule P&L, within resolution buckets.

    This is the single parameter the capital model turns on. Ruin arrives at a
    correlation near 0.07, and everything above that number is arithmetic, so
    measuring it is the difference between "ruled out" and "ruled out for a
    reason somebody checked".

    Resolution DAY is the grouping because that is how the risk actually
    arrives: a book of long favourites does not lose gradually, it loses on the
    days when upsets happen to cluster -- one election night, one weekend of
    sport, one crypto move that settles a dozen derived markets at once.

    One-way ANOVA on per-market P&L:

        ICC = (MSB - MSW) / (MSB + (m - 1) * MSW)

    with m the average bucket size. This is the same rho the Gaussian copula in
    `capacity.drawdown_simulation` takes, so the two connect directly.

    Also returns the effective number of INDEPENDENT bets,
    n / (1 + (m - 1) * ICC). When that number is far below the trade count, the
    breadth argument -- 1,294 bets, therefore diversified -- is false, and it
    is false in the direction that flatters the strategy.
    """
    if rule.is_empty():
        return {"verdict": "rule is empty; no P&L to correlate"}
    t = normalise_trades(trades)
    t = t[t["eff_outcome"].notna() & t["size"].gt(0)]
    t = t[(t["eff_price"] > 0) & (t["eff_price"] < 1)]
    if t.empty or group_on not in t.columns:
        return {"verdict": f"no usable trades, or {group_on!r} missing"}

    labels = _band_labels(rule.bands)
    t = t.assign(band=pd.cut(t["eff_price"], bins=list(rule.bands), labels=labels,
                             include_lowest=True))
    t = t[t["band"].astype(str).isin(rule.side)]
    if t.empty:
        return {"verdict": "no trades fell in the rule's bands"}

    side = t["band"].astype(str).map(rule.side).astype(float)
    t = t.assign(
        _p=np.where(side > 0, t["eff_price"], 1.0 - t["eff_price"]),
        _o=np.where(side > 0, t["eff_outcome"], 1.0 - t["eff_outcome"]))

    per = (t.assign(_wp=t["_p"] * t["size"], _wo=t["_o"] * t["size"])
            .groupby("market_id")
            .agg(_wp=("_wp", "sum"), _wo=("_wo", "sum"), _sz=("size", "sum"),
                 ts=(group_on, "min"))
            .reset_index())
    per["pnl"] = per["_wo"] / per["_sz"] - per["_wp"] / per["_sz"]
    per["bucket"] = (per["ts"] // bucket_seconds).astype("Int64")
    per = per[per["bucket"].notna()]
    if len(per) < 30:
        return {"verdict": f"only {len(per)} markets; too few to estimate"}

    groups = [g["pnl"].to_numpy() for _, g in per.groupby("bucket") if len(g) >= 1]
    k = len(groups)
    n = int(sum(len(g) for g in groups))
    if k < 2 or n <= k:
        return {"verdict": "not enough distinct resolution buckets"}

    grand = float(np.mean(np.concatenate(groups)))
    ssb = float(sum(len(g) * (g.mean() - grand) ** 2 for g in groups))
    ssw = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    msb = ssb / (k - 1)
    msw = ssw / (n - k)
    sizes = np.array([len(g) for g in groups], dtype=float)
    # Unbiased average group size for unbalanced designs.
    m = (n - (sizes ** 2).sum() / n) / (k - 1) if k > 1 else 1.0
    icc = (msb - msw) / (msb + (m - 1) * msw) if (msb + (m - 1) * msw) > 0 else 0.0
    icc = float(max(icc, 0.0))
    n_eff = n / (1.0 + (m - 1) * icc) if m > 1 else float(n)

    return {
        "n_markets": n,
        "n_buckets": k,
        "avg_bucket_size": round(float(m), 2),
        "icc": round(icc, 4),
        "effective_independent_bets": round(float(n_eff), 1),
        "breadth_lost_pct": round(100.0 * (1 - n_eff / n), 1),
        "ruin_threshold_icc": 0.07,
        "verdict": (f"MEASURED correlation {icc:.3f} is ABOVE the ~0.07 that "
                    f"wipes out the capital base; the breadth argument is "
                    f"false and the strategy is ruled out on risk"
                    if icc > 0.07 else
                    f"MEASURED correlation {icc:.3f} is BELOW the ~0.07 ruin "
                    f"threshold; the capital model's pessimistic case does not "
                    f"hold and this needs re-examining"),
    }
