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
5. **Walk-forward (synthetic) shows variance, not overfit, is the problem.**
   `src/walk_forward.py`: 2-year market, tune 72-config grid on year 1, score
   frozen on year 2, repeated over 10 seeds. Mean OOS precision ~5% vs ~4.5%
   in-sample — no measurable overfitting penalty, because the grid is nearly
   flat: informed-pattern flow passes any sane thresholds, and the catalyst
   window is the only filter that moves alert counts materially. The real
   lesson: per-year ROI on an *identical* process swings ~-80% to +120%
   (12 events/yr, payoffs dominated by 1-2 big jumps), so any single-period
   ROI figure (incl. the +33% from the original sweep) is noise. Judge configs
   on multi-period aggregates only.

## Repo layout
- `src/flow_backtest.py` — core framework, fully working:
  - `SignalConfig` — filter thresholds (premium, vol/OI, moneyness, DTE,
    ask-side %, catalyst window, multi-day accumulation)
  - `detect_signals()` — flow tape → ticker-day alerts
  - `evaluate()` — precision/recall vs. announcement dates
  - `simulate_pnl()` — lottery-ticket economics (ROUGH; see Priorities #2)
  - `generate_synthetic_market()` — demo data incl. earnings-spec noise and
    no-event rumor bursts (both are essential; do not remove them)
  - `sweep_thresholds()` — config grid → precision/recall/ROI table
- `src/walk_forward.py` — Priority #1 done: chronological tune/test split with
  boundary embargo, grid tuning on period A only, frozen scoring on period B,
  multi-seed aggregation
- `notebooks/flow_signal_backtest.ipynb` — Colab-ready version of the same,
  with real-data upload cells. Keep in sync with src/ if logic changes.
- `data/` — real CSVs go here (gitignored). Schemas in README.md.

## Priorities (in order)
1. **Walk-forward validation.** Current sweep tunes and tests on the same
   synthetic year — overfit by construction. Split: tune thresholds on
   period A, evaluate frozen on period B.
2. **Replace simulate_pnl with real option prices.** Current payoff model is
   order-of-magnitude only. For flagged contracts, pull actual historical
   option prices (entry at alert close, exit at announcement or expiry).
3. **Real data ingestion.** Flow: Polygon.io options trades or a flow-service
   export (Bullflow/Unusual Whales). Events: SEC EDGAR merger announcements
   (announcement dates, not close dates). Catalysts: free earnings calendars.
   Compute ask_side_pct from trade price vs NBBO if building from raw trades.
4. **Alert service on Render** (only after 1-3 prove an edge exists): daily
   job pulling flow, applying the frozen config, notifying on alerts.

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
