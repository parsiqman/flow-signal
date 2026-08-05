# Project: Unusual Options Flow → M&A Signal Detection

## What this is
Research pipeline testing whether unusual options flow (large ask-side sweeps on
short-dated OTM calls, volume >> open interest) can flag acquisition targets
BEFORE announcement — inspired by the PYPL/Stripe pattern ($294k of 5% OTM
weekly calls bought the day before the deal news, +3000% next day).

Origin: built in a claude.ai conversation (July 2026). This file carries that
context so any Claude Code session starts fully briefed.

## Established so far (do not re-litigate; build on it)
1. **The phenomenon is real but the naive version loses money.** Synthetic
   backtest showed: an unfiltered "unusual activity" screener fires ~700
   alerts/year at ~2% precision, ROI ≈ -43%. Survivorship bias is the trap —
   the PYPL screenshot is one winner from a pile of look-alikes.
2. **Catalyst exclusion is the biggest lever.** Dropping flow within 7 days of
   scheduled events (earnings, FDA dates) is what separates "unexplained
   urgency" (the actual tell) from ordinary speculation. In the synthetic
   sweep it tripled precision and flipped ROI positive.
3. **Realistic ceiling: ~6-10% precision even with good filters.** Rumor
   bursts and hidden spread legs look identical to informed flow. Economics
   depend entirely on 10x-40x payoffs on hits covering ~90% near-total losses.
4. **Recall has a ceiling too**: only some fraction of deals leak (academic
   estimates ~25-60%; see Augustin, Brenner & Subrahmanyam, NYU, on informed
   options trading ahead of M&A).

## Market selection: SETTLED 2026-08-05 (do not re-litigate; see DECISION.md)
Target market is **US cash equities** — cross-sectional, dollar-neutral, 1-3 day
holding period, ~1500-name liquid universe via Alpaca. Crypto and DeFi were
evaluated seriously across six venues and lost:
- Not on fees. Crypto's volatility pays for most of its fee disadvantage
  (19x the round-trip cost, but only ~6x the cost-to-noise ratio), and L2 gas
  is a rounding error (~0.06bp on a $5k clip) next to AMM pool fees.
- **On statistical power.** Retail crypto fees force 13-22 day minimum holding
  periods; long horizons plus small, highly-correlated universes yield 37-162
  independent bets/year vs ~4,000 in equities. Free history covers 0.41x of
  what equities needs to validate and 0.004-0.010x of what crypto needs.
- Two 2026 rule changes finished it: the PDT $25k rule was eliminated
  (2026-06-04), removing crypto's capital-access edge; and Hyperliquid, the one
  crypto venue strong on both fees and breadth, geoblocks US persons.
Equities won 96.6% of 20,000 random criterion weightings and all 7 stress tests.
Reopen the question only if: CFTC onshores DeFi-style perps to US persons, a
sub-10bp round-trip crypto fee tier becomes reachable, or capital exceeds ~$500k.

**Sobering corollary, do not lose it:** even the winner needs ~12 years to reach
t=2 at IC=0.04. It only becomes validatable within 10 years of free history at
IC>=0.05. The equity path lives or dies on beating the published-anomaly
baseline slightly. Plan research accordingly.

## Repo layout
- `DECISION.md` — the market-selection memo: recommendation, evidence,
  sensitivities, sources, and what would change the answer.
- `src/market_selection/` — the runnable decision model behind it:
  - `venues.py` — ALL parameters that drive the conclusion, in one file.
    Disagree here and re-run; nothing is hardcoded elsewhere.
  - `economics.py` — cost-to-noise, minimum viable holding period, cost drag
  - `power.py` — Grinold net of costs: breadth -> IR -> years-to-validate,
    plus Monte Carlo false-positive rates
  - `scorecard.py` — hard gates, weighted score, 20k-draw weight sensitivity
  - `run_analysis.py` — entry point; prints the full chain of reasoning
- `tests/test_market_selection.py` — 20 tests. These check the *machinery*
  (analytic IR matches simulation, monotonicities, gates actually veto), not
  the assumptions. Run them after touching venues.py.
- `src/flow_backtest.py` — core framework, fully working:
  - `SignalConfig` — filter thresholds (premium, vol/OI, moneyness, DTE,
    ask-side %, catalyst window, multi-day accumulation)
  - `detect_signals()` — flow tape → ticker-day alerts
  - `evaluate()` — precision/recall vs. announcement dates
  - `simulate_pnl()` — lottery-ticket economics (ROUGH; see Priorities #2)
  - `generate_synthetic_market()` — demo data incl. earnings-spec noise and
    no-event rumor bursts (both are essential; do not remove them)
  - `sweep_thresholds()` — config grid → precision/recall/ROI table
- `notebooks/flow_signal_backtest.ipynb` — Colab-ready version of the same,
  with real-data upload cells. Keep in sync with src/ if logic changes.
- `data/` — real CSVs go here (gitignored). Schemas in README.md.

## Priorities (in order) — reset by the market-selection decision
0. **Verify the venue parameters from Colab.** Nothing in `venues.py` was
   measured; this sandbox's egress policy blocks every market-data host
   (Binance, Coinbase, Kraken, Hyperliquid, DefiLlama, sec.gov all 403 on
   CONNECT). Colab is not restricted. Confirm, in order of uncertainty:
   US onshore perp fees, blended equity half-spread, and `residual_corr` per
   venue (that one drives breadth, which drives the whole conclusion).
1. **Equity data ingestion.** Alpaca: ~10y of free 1-minute bars, ~1500 names
   above $5M ADV. Build the universe with point-in-time membership — a
   universe defined by today's liquidity is survivorship bias.
2. **Fix the walk-forward split BEFORE any signal work.** Period A for tuning,
   period B frozen. Write the split into code and commit it before looking at
   a single result. This was priority #1 for the options work too and never
   got done; do not repeat that.
3. **Cross-sectional signal research, 1-3 day horizon.** Target IC >= 0.05.
   Below 0.04 the strategy is not validatable within 10 years of history, so
   IC is the go/no-go metric, not backtest ROI.
4. **Alert/execution service on Render** (only after 1-3 prove an edge): one
   daily cron after the 4pm close. Market hours are the batch boundary — this
   is why equities beat 24/7 crypto on infra fit for a machine-less setup.

### Parked: the options-flow M&A thesis
Not abandoned — it has the highest per-bet edge of the six venues (IC≈0.14
implied by the established 6-10% precision at 10-40x payoffs). It fails on
data access alone: historical flow + NBBO is paid-only, no free tier is deep
enough for walk-forward. Revisit if the project ever agrees to pay ~$250/mo.
Its original priorities, still valid if resumed: walk-forward split; replace
`simulate_pnl` with real historical option prices (entry at alert close, exit
at announcement or expiry); real ingestion (Polygon.io trades or a flow-service
export; SEC EDGAR for announcement dates, not close dates; free earnings
calendars for catalysts; compute ask_side_pct from trade price vs NBBO).

## Hard rules
- **No lookahead bias.** Detector may only use data available at alert time.
  Open interest updates overnight — same-day OI is cheating; lag it one day.
- **Don't tune on the test period.** Ever.
- Following public options tape is legal; the framework detects *possible*
  informed trading, it does not establish it. Keep framing factual.
- This is research tooling, not financial advice. Its most likely honest
  output is "the edge is smaller than the screenshot implies." Report negative
  results as readily as positive ones.
- User environment: Chromebook + Colab + Render. No local dev machine.
  Anything requiring local execution is a non-starter; keep everything
  runnable in this sandbox, Colab, or Render.
