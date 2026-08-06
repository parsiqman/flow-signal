# Project: Options Trading Algo — Variance Risk Premium Harvest

## What this is
A research platform for scouting, testing, validating and iterating on trading
strategies — plus the first candidate put through it.

**The platform is the deliverable, not the strategy.** An LLM trained on public
text is a poor source of proprietary alpha; the useful contribution is making
sure that when an idea is tested, the answer is trustworthy.

Read **LAB.md** first (the machine), then **STRATEGY.md** (the first candidate).

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

## Results so far (synthetic ONLY)
The VRP strategy passes 4 of the platform's 5 gates and is **BLOCKED** at
`validated` on deflated Sharpe (0.81 vs a 0.95 bar). Its out-of-sample Sharpe
clears the luck baseline for a 36-config search, but not by enough once the
negative skew of a short-vol return stream is accounted for.

That is the platform working. Do not "fix" this by loosening the threshold or
by searching harder — searching harder RAISES the bar it must clear. The two
legitimate routes are a genuinely better strategy, or more data.

## Repo layout
- `LAB.md` — the research platform. **Start here.**
- `src/lab/`
  - `protocol.py` — the interface every candidate implements
  - `registry.py` — hypothesis pre-registration + automatic trial ledger
  - `validation.py` — deflated Sharpe, PBO/CSCV, null tests, the gauntlet
  - `scout.py` — idea catalogue, ordered by why someone loses money to you
  - `pipeline.py` — promotion stages that cannot be skipped
  - `run_lab.py` — the machine demonstrated end to end
- `tests/test_lab.py` — 34 tests, incl. Monte Carlo checks of the statistics
- `DATA.md` + `src/data/` — real option-chain ingestion: canonical schema,
  quality gate, constant-maturity IV, strike snapping, `measure_vrp`
- `notebooks/real_option_chains.ipynb` — the Colab pull (this sandbox cannot
  reach ANY options vendor; all return 403 on CONNECT)
- `tests/test_data.py` — 26 tests, every defect reproduced on purpose
- `STRATEGY.md` — the first candidate: results, bugs, next steps.
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
- **The search is the enemy.** Every backtest run raises the bar the winner
  must clear. Record every run to the TrialLedger, including exploratory ones;
  an undercounted trial log silently inflates every significance number.
- **No idea without a named counterparty.** If you cannot say who is on the
  other side and why they keep taking it, it is a pattern in a dataset.
- **All gates must pass; never average them.** A weighted score lets a strong
  return outvote a failed overfitting test. That trade destroys accounts.
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
0. **Run the kill criterion on free data.** DoltHub `post-no-preference/options`
   is free and covers ~2019-now (Mar-2020, 2022, Aug-2024 = three vol regimes).
   Run `notebooks/real_option_chains.ipynb` in Colab and check
   `chain.measure_vrp`. STRATEGY.md commits to abandoning the thesis below 3%
   capture. Honour that. Spend NO money before this answers yes.
1. **Only then buy deeper history** (ORATS, 2007+) so 2008 is in the sample.
   Without a crisis, a short-vol backtest measures the good half of the
   distribution.
2. **Run the real panel through the gauntlet.** The panel is schema-identical to
   the synthetic one, so the backtest and gauntlet run unchanged. Expect real
   data to be harder than synthetic, not easier.
3. **Retest the regime filter and richness floor** against real crises.
4. **Paper trade one full quarter** before any capital.
5. Only then: a Render cron placing orders, sized far below the backtest.
