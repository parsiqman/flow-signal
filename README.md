# Flow Signal — Unusual Options Flow → M&A Backtest

Tests whether pre-announcement informed options flow is detectable and tradeable.
See `CLAUDE.md` for full project context and priorities.

## Quick start
```bash
pip install -r requirements.txt
python src/flow_backtest.py        # runs synthetic demo + threshold sweep
python src/walk_forward.py         # walk-forward validation, 10 seeds (~2 min)
```
Or open `notebooks/flow_signal_backtest.ipynb` in Colab (zero install).

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
