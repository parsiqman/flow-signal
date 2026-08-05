# Options Trading Algo — research repo

**Current work: a defined-risk variance risk premium harvest.** Sell volatility
across ~60 liquid underlyings, cap every position with bought wings, hold to
expiry. Validated on synthetic data only — see **[STRATEGY.md](STRATEGY.md)**.

Synthetic results: +12.3% CAGR out-of-sample, −7.5% max drawdown, profitable in
14 of 14 independently generated markets, survives 2008-scale shocks. The
Sharpe of ~2.5 is **not believable** and is discounted accordingly; nothing here
has touched a real option chain yet.

## Quick start
```bash
pip install -r requirements.txt

python src/options_alpha/run.py --quick    # the whole chain, ~70s
python tests/test_options_alpha.py         # 23 tests
```

## What's here
| Path | What |
|---|---|
| [STRATEGY.md](STRATEGY.md) | The strategy, results, the bugs, next steps. **Start here.** |
| [DECISION.md](DECISION.md) | Why equities over crypto/DeFi. Settled, still valid. |
| `src/options_alpha/` | The strategy: families, generator, backtest, research harness |
| `src/market_selection/` | The venue model behind DECISION.md |
| `docs/ARCHIVE-ma-flow-thesis.md` | Superseded M&A options-flow project (`src/flow_backtest.py`) |

## One thing worth knowing before touching this
A backtest is only as trustworthy as the market it runs in, and a broken market
fails **silently**. During development, a generator bug made implied vol sit
*below* true realised vol — so frictionless option selling lost money and every
strategy result was measuring my parameters rather than the strategy. Nothing
crashed; the backtest just confidently reported a loss.

`run.py` therefore validates the market before any strategy runs on it, and
aborts if frictionless straddle selling isn't profitable. Keep that guard.

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
