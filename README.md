# Trading Strategy Research Platform

**The deliverable is the machine, not a strategy.** `src/lab/` is infrastructure
for scouting, testing, validating and iterating on candidates — with
multiple-testing correction built into the foundation, where it can't be
skipped. Backtest 100 strategies and keep the best, and you get a beautiful
Sharpe ratio even if none of them has an edge; the platform computes that luck
baseline explicitly and makes every candidate clear it.

The first candidate through it is **blocked**: 4 of 5 gates passed, failing
deflated Sharpe at 0.81 against a 0.95 bar. That is the demonstration.

See **[LAB.md](LAB.md)**.

---

## The first candidate: a defined-risk variance risk premium harvest

Sell volatility across ~60 liquid underlyings, cap every position with bought
wings, hold to expiry. Synthetic data only — see **[STRATEGY.md](STRATEGY.md)**.

Synthetic results: +12.3% CAGR out-of-sample, −7.5% max drawdown, profitable in
14 of 14 independently generated markets, survives 2008-scale shocks. The
Sharpe of ~2.5 is **not believable** and is discounted accordingly; nothing here
has touched a real option chain yet.

## Quick start
```bash
pip install -r requirements.txt

python src/lab/run_lab.py                  # the platform, end to end
python tests/test_lab.py                   # 34 tests

python src/options_alpha/run.py --quick    # the candidate strategy, ~70s
python tests/test_options_alpha.py         # 23 tests
python tests/test_data.py                  # 26 tests, real-chain ingestion
```

Real chains run in **Colab**, not here: this sandbox's egress policy blocks every
options vendor. See [DATA.md](DATA.md).

## What's here
| Path | What |
|---|---|
| [LAB.md](LAB.md) | The research platform. **Start here.** |
| `src/lab/` | Protocol, registry + trial ledger, validation gauntlet, scout, pipeline |
| [STRATEGY.md](STRATEGY.md) | The first candidate: results, the bugs, next steps |
| [DATA.md](DATA.md) | Real option chains: what to buy, the quality gate, the kill criterion |
| `src/data/` + `notebooks/real_option_chains.ipynb` | Ingestion, quality gate, Colab pull |
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
