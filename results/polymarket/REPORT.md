# Polymarket wallet scan — 2026-08-07 00:00 UTC

## MAYBE. 1 wallet(s) clear the luck bar, and the edge is positive in both halves of their own history. The cross-sectional persistence test could NOT be run (too few wallets) -- this is weaker evidence than it looks. Check bias attribution before building anything.

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

### Within-wallet split — `0x6011655c4afb76f36dd1b08a137a1ba73466b31e`

- `split_ts`: 1782777600.0
- `early`: {'n_markets': 897, 'n_eff': 457.2, 'edge_per_share': 0.0431, 't_stat': 7.41, 'roi': 0.0952}
- `late`: {'n_markets': 934, 'n_eff': 372.5, 'edge_per_share': 0.0458, 't_stat': 7.1, 'roi': 0.101}
- `edge_decay`: 0.0027
- `verdict`: edge is present and significant in BOTH halves of this wallet's own history

### Bias attribution — `0x6011655c4afb76f36dd1b08a137a1ba73466b31e`

- `overall_edge`: 0.0445
- `extreme_band_stake`: 0.832
- `verdict`: edge is concentrated in extreme prices -- likely harvesting favourite-longshot bias, which you should run directly rather than copy

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
