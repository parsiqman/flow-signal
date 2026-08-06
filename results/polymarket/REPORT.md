# Polymarket wallet scan — 2026-08-06 23:52 UTC

## NO. 1 wallet(s) cleared the luck bar but past performance does NOT predict future performance -- consistent with those wallets having been lucky.

| | |
|---|---|
| mode | named |
| wallets discovered | 1 |
| wallets scored (>=15 fills, >=10 markets) | 1 |
| t-stat needed to clear luck | 1.64 |
| best t observed | 10.19 |
| best edge observed | 4.45 c/share |
| **wallets clearing the bar (uniform)** | **1** |
| **wallet-category pairs clearing (specialists)** | **1** |
| concentrated-edge wallets surfaced | 0 |

## Persistence (the decisive test)

- `verdict`: only 1 wallets active in both periods; too few to conclude anything
- `n_selected`: 1

## Copy economics

- `leader_edge_cents`: 4.45
- `slippage_cents`: 2.93
- `net_edge_cents`: 1.52
- `edge_retained_pct`: 34.2
- `breakeven_leader_edge_cents`: 2.93
- `return_per_trade_pct`: 3.05
- `copies_landed_per_year`: 110
- `stake_per_trade_usd`: 1250
- `est_annual_return_pct`: 16.8
- `capacity_usd_per_trade`: 2500
- `verdict`: copying retains a usable share of the edge

## How to read this

A low wallet count clearing the bar is the expected result. The
measured false-positive rate of that gate is ~20% of populations,
so clearing it is necessary, not sufficient — persistence is the
test that matters. See POLYMARKET.md.
