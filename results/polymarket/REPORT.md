# Polymarket wallet scan — 2026-08-06 09:37 UTC

## NO. 11 wallet(s) cleared the luck bar but past performance does NOT predict future performance -- consistent with those wallets having been lucky.

| | |
|---|---|
| mode | live |
| wallets discovered | 6,025 |
| wallets scored (>= 20 trades) | 313 |
| t-stat needed to clear luck | 4.32 |
| best t observed | 124.5 |
| best edge observed | 54.34 c/share |
| **wallets clearing the bar** | **11** |

## Persistence (the decisive test)

- `verdict`: only 1 wallets active in both periods; too few to conclude anything
- `n_selected`: 1

## Copy economics

- `leader_edge_cents`: 54.34
- `slippage_cents`: 2.93
- `net_edge_cents`: 51.41
- `edge_retained_pct`: 94.6
- `breakeven_leader_edge_cents`: 2.93
- `return_per_trade_pct`: 102.82
- `copies_landed_per_year`: 110
- `stake_per_trade_usd`: 1250
- `est_annual_return_pct`: 565.5
- `capacity_usd_per_trade`: 2500
- `verdict`: copying retains a usable share of the edge

## How to read this

A low wallet count clearing the bar is the expected result. The
measured false-positive rate of that gate is ~20% of populations,
so clearing it is necessary, not sufficient — persistence is the
test that matters. See POLYMARKET.md.
