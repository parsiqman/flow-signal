# Flow Signal — trading algo research

Two things live here:

1. **Market selection** (`src/market_selection/`) — which market to build a
   trading algo in. Answered: **US cash equities**, cross-sectional, 1–3 day
   horizon. Crypto and DeFi lose on statistical power, not on fees. Full
   reasoning and sources in **[DECISION.md](DECISION.md)**.
2. **Unusual options flow → M&A** (`src/flow_backtest.py`) — the original
   thesis: is pre-announcement informed options flow detectable and tradeable?
   Parked, not deleted: it has the highest per-bet edge of any venue evaluated
   but no free historical data to validate against. See `CLAUDE.md`.

## Quick start
```bash
pip install -r requirements.txt

python src/market_selection/run_analysis.py   # the market decision, end to end
python tests/test_market_selection.py         # 20 tests on the decision model
python src/flow_backtest.py                   # synthetic demo + threshold sweep
```
Or open `notebooks/flow_signal_backtest.ipynb` in Colab (zero install).

To disagree with the market decision, edit the parameter table in
`src/market_selection/venues.py` and re-run — every number that drives the
conclusion is in that one file.

## Real data schemas (put CSVs in data/)

**flow.csv** — one row per sweep/block print:
```
date,ticker,opt_type,strike,spot,dte,volume,open_interest,premium,ask_side_pct,exec_type
2026-07-13,PYPL,C,60.0,57.1,4,4200,310,91000,0.95,sweep
```

**events.csv** — announcements to test against:
```
ticker,announce_date,jump_pct
PYPL,2026-07-14,0.42
```

**catalysts.csv** — scheduled events to exclude (earnings, FDA):
```
ticker,date
PYPL,2026-07-28
```

## Data sources
- Flow: Bullflow / Unusual Whales / Cheddar Flow exports (paid), or build from
  Polygon.io options trades / CBOE DataShop (ask_side_pct = trade vs NBBO).
- Events: SEC EDGAR full-text search for merger agreements (announcement dates).
- Catalysts: free earnings calendars (Nasdaq, FMP free tier).

Minimum for a meaningful backtest: 1-2 years of flow, few hundred liquid
names, 30+ announcement events.
