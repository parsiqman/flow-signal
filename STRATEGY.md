# Defined-Risk Variance Risk Premium Harvest

**Status: works on synthetic data. Has never touched a real option chain.**

```bash
python src/options_alpha/run.py --quick   # full pipeline, ~70s
python tests/test_options_alpha.py        # 23 tests
```

---

## 1. Why this strategy and not another

Seven options strategy families were scored on breadth, friction, edge
durability, data cost, and tail behaviour (`src/options_alpha/families.py`).
**All seven failed the screen**, for two complementary reasons:

- **Index short vol** has the strongest edge in options — a 25-year-documented
  risk premium — and is unsurvivable naked: ~15 sigma of tail loss ends the
  account.
- **Cross-sectional VRP** is survivable and broad, but vega-neutralising it
  hedges away the premium itself, leaving a forecast signal too weak to prove.

The failures are complementary, which points at a strategy none of the seven
describes: **take the premium, cap the tail structurally, spread it thin.**
That synthesis is what got built.

One distinction did most of the work. A **documented risk premium** does not
need rediscovering from your own short history — the research problem is
implementation and survival. A **mispricing** with no published track record
must be established from scratch. Conflating the two is how people spend two
years "validating" the equity risk premium. This project is an *implementation*
problem, which is why the signal logic is deliberately plain.

---

## 2. The strategy

| | |
|---|---|
| **Universe** | ~60 liquid optionable underlyings |
| **Structure** | Iron condor — short strangle with bought wings, defined risk |
| **Short strikes** | 10-delta, chosen by inverting Black-Scholes (not a fixed % offset) |
| **Wings** | 0.8 × the 1-sigma expected move, so width scales with vol |
| **Tenor** | 35 trading days |
| **Entry** | Weekly, so expiries ladder instead of clustering |
| **Exit** | Close at 50% of credit captured, else hold to expiration |
| **Sizing** | 1% of equity at risk per position, 25% portfolio cap |
| **Excluded** | Any name with earnings inside the option's life |

Three design choices carry the strategy, and each falls out of an analysis
rather than a preference:

**Strikes are set by delta, not by percentage.** A fixed 5%-OTM strike means
something completely different at 20% vol than at 60% vol — it would silently
load up on risk in exactly the regimes where risk should be coming off.

**Wings are bought, always.** They cost roughly a quarter of the edge and
remove the failure mode that ends the project. Position sizing cannot do this
job, because in a vol shock every short-vol position loses at once —
diversification smooths ordinary days, the wings are the actual protection. If
implied vol is high enough that the wing strike would land at or below zero,
the trade is **refused**, not clamped: a put spread whose long leg can't be
bought is a naked short put in disguise.

**Positions are held to expiry where possible.** Expiration costs nothing to
close; an early exit pays the spread a second time. This single choice is worth
more than any signal refinement — the top of every in-sample grid was occupied
by hold-to-expiry variants before anything else mattered.

---

## 3. Results (synthetic only)

Walk-forward: grid searched on the first 6 years, one config frozen, run **once**
on the last 6.

| | In-sample (artefact) | Out-of-sample |
|---|---|---|
| CAGR | 9.6% | **12.3%** |
| Sharpe | 1.70 | 2.52 |
| Max drawdown | −6.6% | −7.5% |
| Worst month | −6.1% | −4.2% |
| Win rate | — | 94.8% |

Out-of-sample beat in-sample, so there was no overfitting collapse. The median
in-sample config scored 4.4% on the same test half, so the tuning added real
signal rather than noise.

**Frozen config on 14 independently generated markets** (never tuned on):
mean CAGR **+11.6%**, profitable in **100%**, mean max drawdown −7.1%, worst
−10.7%.

**Crisis stress**, shocks forced every ~2 years:

| Shock severity | Mean CAGR | Worst drawdown |
|---|---|---|
| mild (2.8× vol, −9% gap) | +4.9% | −16.5% |
| severe (5× vol, −18% gap) | +5.3% | −16.3% |
| 2008-scale (7× vol, −28% gap) | +5.8% | −18.4% |

**Execution sensitivity** — CAGR falls from 8.0% at mid fills to 6.4% crossing
the spread every time. Friction matters but is not decisive here, which was a
surprise: the four legs partly offset, so a full spread cross costs <10% of the
credit, not the ~4% of capital-at-risk the family model assumed.

**Ablation** — what is actually load-bearing:

| Removed | CAGR effect |
|---|---|
| Earnings exclusion | **−4.3pp** — the single most important filter |
| Profit target (hold to expiry instead) | **−3.3pp** |
| Regime filter | **+0.9pp** — it *hurts*; now off by default |
| Richness floor | +0.5pp — not load-bearing |

The last two matter. The regime filter and the rich-vol selection were both in
the original design on reasoning that turned out to be wrong: the wings already
cap the loss, so standing down mid-shock just forgoes the richest premium of the
cycle. Both are kept in the code but disabled, because the synthetic tail is
gentler than a real one — they get retested on real 2008/2020 data before being
discarded for good.

---

## 4. What this does NOT establish

**The Sharpe ratio is not believable and should be discounted hard.** Real
short-volatility books run 0.8–1.5. A synthetic 2.5 means the model's world is
too kind, in three specific ways: liquidity never disappears, spreads never gap,
and early assignment never happens at the worst moment. All three occur in a
real crisis and all three hurt precisely when the book is already losing.

**The premium here is a parameter, not a measurement.** It was set at +16%
relative and validated at ~9% frictionless straddle capture. Real single-name
VRP is smaller and varies across names and regimes.

**Nothing has touched a real option chain.** Every number above describes a
market I wrote.

---

## 5. Three bugs worth remembering

Each produced a confident, plausible, completely wrong backtest. None would
have been caught by testing the strategy.

1. **Exploding volatility.** The log-vol process folded its own common factor
   back into the per-name state each step, compounding to 107% average vol.
   Caught by eyeballing the generated market — which is why the pipeline prints
   its inputs before its results.

2. **A premium that wasn't there.** Implied vol was priced off the latent vol
   process, which knows nothing about earnings jumps or crash gaps. The nominal
   "+16% premium" was actually **−2% real**, so frictionless straddle selling
   lost money and every strategy result was measuring my parameters rather than
   the strategy. This is the dangerous class of bug: nothing crashed, the
   backtest just confidently said the strategy loses.

3. **Trading days vs calendar days.** Volatility annualised with √252 but time
   to expiry computed as `dte/365`, where `dte` counts *trading* days. Every
   option priced for 83% of its real time exposure — a systematic ~17%
   underpricing that swamped the effect being measured.

The guard against all three is `test_premium_is_actually_harvestable`, which
checks the market rewards vol selling **frictionlessly** before any strategy is
allowed to run on it. `run.py` performs the same check at step 2 and aborts if
it fails. **A backtest is only as trustworthy as the market it runs in, and a
broken market fails silently.**

---

## 6. Next steps, in order

1. **Buy deep end-of-day option chain history.** ~15–19 years, back to the
   mid-2000s, covering 2008, 2018, 2020 and 2024. This is the cheapest options
   data that exists (roughly a few hundred dollars one-time, versus $250/month
   for trade-level flow) and it is the only way to see the strategy's own tail.
   A short-vol backtest that doesn't contain a crisis is measuring the good half
   of the distribution.
2. **Re-run this exact pipeline against it.** The code is data-source agnostic:
   swap `generate_market()` for a real loader producing the same panel columns
   (`day, name, spot, atm_iv, days_to_earnings`).
3. **Re-run the generator validation on real data too** — measure actual
   straddle capture per name and per year. If real VRP is much below the ~9%
   assumed here, the economics change and the strategy may not clear costs.
4. **Retest the regime filter and richness floor** on real crises before
   discarding them.
5. **Paper trade** for one full quarter before any capital. The gap between a
   backtest and a fill is where this kind of strategy dies.

Only after 1–5: a Render cron placing orders, sized far below the backtest.

---

## 7. Honest summary

A defensible, tested, mechanically sound implementation of a real and
well-documented market premium, which survives its own tail in every synthetic
market it has seen — and which has not yet been shown to work anywhere outside
a market I generated myself. The design reasoning is sound and the engineering
is checked. The evidence is not yet real.
