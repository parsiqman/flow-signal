# Polymarket wallet scan — 2026-08-06 11:40 UTC

## MAYBE. 6 wallet(s) clear the luck bar AND past performance predicts future performance. Check bias attribution before building anything.

| | |
|---|---|
| mode | offline |
| wallets discovered | 1,200 |
| wallets scored (>=20 fills, >=10 markets) | 1,200 |
| t-stat needed to clear luck | 3.96 |
| best t observed | 4.73 |
| best edge observed | 25.23 c/share |
| **wallets clearing the bar (uniform)** | **6** |
| **wallet-category pairs clearing (specialists)** | **0** |
| concentrated-edge wallets surfaced | 0 |

## Persistence (the decisive test)

- `n_wallets_both_periods`: 1200
- `n_selected`: 120
- `selected_oos_edge`: 0.0396
- `everyone_else_oos_edge`: 0.0052
- `gap`: 0.0344
- `gap_t_stat`: 3.49
- `rank_correlation`: 0.0919
- `verdict`: past performance predicts future performance

## Copy economics

- `leader_edge_cents`: 22.28
- `slippage_cents`: 2.93
- `net_edge_cents`: 19.35
- `edge_retained_pct`: 86.8
- `breakeven_leader_edge_cents`: 2.93
- `return_per_trade_pct`: 38.7
- `copies_landed_per_year`: 110
- `stake_per_trade_usd`: 1250
- `est_annual_return_pct`: 212.8
- `capacity_usd_per_trade`: 2500
- `verdict`: copying retains a usable share of the edge

## How to read this

A low wallet count clearing the bar is the expected result. The
measured false-positive rate of that gate is ~20% of populations,
so clearing it is necessary, not sufficient — persistence is the
test that matters. See POLYMARKET.md.
