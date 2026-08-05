# Project: Options Trading Algo — Variance Risk Premium Harvest

## What this is
Design, test and build a systematic options strategy. Currently: a defined-risk
short-volatility book that harvests the variance risk premium across ~60 liquid
underlyings, validated on synthetic data only.

Read **STRATEGY.md** first — it is the current state of the work.

## Established (do not re-litigate; build on it)

1. **Market: US equity options.** Chosen over crypto/DeFi in DECISION.md;
   crypto lost on statistical power, not fees. That analysis stands.
2. **Strategy family: defined-risk VRP harvest.** All seven naive options
   families failed the screen in `families.py`. Index short vol has the best
   edge and is unsurvivable naked; cross-sectional VRP is survivable but
   vega-neutralising hedges the premium away. The synthesis — take the premium,
   cap the tail with bought wings, spread it thin — is what got built.
3. **This is an IMPLEMENTATION problem, not a DISCOVERY problem.** The variance
   risk premium is documented back to the early 2000s. Do not build elaborate
   signal machinery on top of a known effect; it will fit noise and break out of
   sample. The research question is survival and execution, not alpha discovery.
4. **Three design choices carry the strategy**, each derived not preferred:
   strikes by delta (not fixed %), wings always bought (refuse the trade if the
   wing strike would be <= 0), and hold to expiry where possible.
5. **The regime filter and richness floor are NOT load-bearing** — the ablation
   shows the regime filter actively costs ~1pp of CAGR. Both are disabled by
   default but retained, pending a retest on real crisis data.

## Results so far (synthetic ONLY — see STRATEGY.md §3)
Walk-forward OOS: +12.3% CAGR, max drawdown -7.5%. Across 14 fresh markets:
+11.6% mean CAGR, profitable in 100%. Survives 2008-scale shocks every 2 years.
**The Sharpe of ~2.5 is not believable** and must be discounted — the synthetic
tail is too gentle (liquidity never vanishes, spreads never gap, no assignment).

## Repo layout
- `STRATEGY.md` — the strategy, results, bugs, and next steps. Start here.
- `DECISION.md` — why equities over crypto/DeFi (settled, still valid).
- `src/options_alpha/`
  - `families.py` — why this strategy and not the other six
  - `synthetic.py` — the market generator + Black-Scholes
  - `strategy.py` — signal, position construction, sizing, regime defence
  - `backtest.py` — engine + tail-aware diagnostics
  - `research.py` — walk-forward split, sweep, ablation, multi-seed
  - `run.py` — entry point; runs the whole chain
- `tests/test_options_alpha.py` — 23 tests, weighted toward the GENERATOR
- `tests/test_market_selection.py` — 20 tests for the DECISION.md model
- `src/market_selection/` — the venue model behind DECISION.md
- `docs/ARCHIVE-ma-flow-thesis.md` — the superseded M&A options-flow project
- `src/flow_backtest.py` — that project's code. Parked, not deleted.

## Hard rules
- **Validate the market before trusting any backtest.** A market that does not
  reward vol selling frictionlessly cannot evaluate a vol-selling strategy, and
  that failure is SILENT — it produces a confident, losing, meaningless result.
  `test_premium_is_actually_harvestable` and `run.py` step 2 both guard this.
  This bug already happened once; see STRATEGY.md §5.
- **Trading days, not calendar days.** Vol is annualised with sqrt(252) and
  `dte` counts trading days, so year fractions use `/TRADING_DAYS`. Using /365
  underprices every option by ~17% and silently swamps the effect measured.
- **Don't tune on the test period. Ever.** Fix the split in code before looking
  at results. `research.walk_forward()` does this; use it.
- **No lookahead.** The detector at day t may use data up to and including t
  only. `test_no_lookahead` enforces this by truncation.
- **Never sell naked.** If defined risk cannot be constructed, skip the trade.
- Report negative results as readily as positive ones. The most likely honest
  outcome of any strategy research is "smaller than it looked".
- This is research tooling, not financial advice.
- User environment: Chromebook + Colab + Render. No local dev machine. Anything
  requiring local execution is a non-starter.

## Priorities (in order)
0. **Buy deep EOD option chain history** (~15-19y, covering 2008/2018/2020/2024).
   Cheapest options data there is. Without a crisis in the sample, a short-vol
   backtest measures only the good half of the distribution.
1. **Re-run this pipeline on real data.** Swap `generate_market()` for a loader
   producing the same panel columns; everything downstream is source-agnostic.
2. **Re-run the generator validation on real chains** — measure actual straddle
   capture per name and year. If real VRP is well below the ~9% assumed here,
   the economics change materially.
3. **Retest the regime filter and richness floor** against real crises.
4. **Paper trade one full quarter** before any capital.
5. Only then: a Render cron placing orders, sized far below the backtest.
