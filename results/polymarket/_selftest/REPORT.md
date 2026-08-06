# Polymarket wallet scan — 2026-08-06 09:36 UTC

## MAYBE. 20 wallet(s) clear the luck bar AND past performance predicts future performance. Check bias attribution before building anything.

| | |
|---|---|
| mode | offline |
| wallets discovered | 1,200 |
| wallets scored (>= 20 trades) | 1,200 |
| t-stat needed to clear luck | 3.96 |
| best t observed | 5.65 |
| best edge observed | 32.7 c/share |
| **wallets clearing the bar** | **20** |

## Persistence (the decisive test)

- `n_wallets_both_periods`: 1200
- `n_selected`: 120
- `selected_oos_edge`: 0.0691
- `everyone_else_oos_edge`: 0.0026
- `gap`: 0.0664
- `gap_t_stat`: 6.34
- `rank_correlation`: 0.0693
- `verdict`: past performance predicts future performance

## Copy economics

- `leader_edge_cents`: 32.7
- `slippage_cents`: 2.93
- `net_edge_cents`: 29.77
- `edge_retained_pct`: 91.0
- `breakeven_leader_edge_cents`: 2.93
- `return_per_trade_pct`: 59.54
- `copies_landed_per_year`: 110
- `stake_per_trade_usd`: 1250
- `est_annual_return_pct`: 327.5
- `capacity_usd_per_trade`: 2500
- `verdict`: copying retains a usable share of the edge

## How to read this

A low wallet count clearing the bar is the expected result. The
measured false-positive rate of that gate is ~20% of populations,
so clearing it is necessary, not sufficient — persistence is the
test that matters. See POLYMARKET.md.
