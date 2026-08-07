# Polymarket wallet scan — 2026-08-07 01:30 UTC

## NO. Nothing clears the luck bar, and no out-of-sample test could be run on this sample.

| | |
|---|---|
| mode | named |
| wallets discovered | 1 |
| wallets scored (>=15 fills, >=10 markets) | 1 |
| t-stat needed to clear luck | 1.64 |
| best t observed | 1.13 |
| best edge observed | 1.35 c/share |
| **wallets clearing the bar (uniform)** | **0** |
| **wallet-category pairs clearing (specialists)** | **0** |
| concentrated-edge wallets surfaced | 0 |

## Persistence (the decisive test)

- `verdict`: only 1 wallets active in both periods; too few to conclude anything
- `n_selected`: 1

### Within-wallet split — `0x4ebc2722adc772bde8680792d0a6fdf15499a33d`

- `split_ts`: 1785715200.0
- `early`: {'n_markets': 86, 'n_eff': 45.4, 'edge_per_share': 0.0084, 't_stat': 0.45, 'roi': 0.018}
- `late`: {'n_markets': 104, 'n_eff': 64.1, 'edge_per_share': 0.0179, 't_stat': 1.15, 'roi': 0.038}
- `edge_decay`: 0.0095
- `verdict`: edge is positive in both halves but significant in at most one; suggestive, not established

### Bias attribution — `0x4ebc2722adc772bde8680792d0a6fdf15499a33d`

- `overall_edge`: 0.0135
- `extreme_band_stake`: 0.377
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Trade style — `0x4ebc2722adc772bde8680792d0a6fdf15499a33d`

- `n_markets`: 190
- `both_sides_frac`: 0.995
- `median_fills_per_market`: 45.0
- `median_span_hours`: 1.034
- `extreme_band_stake`: 0.377
- `avg_size_per_market`: 12457.8
- `style`: market maker / latency
- `copyable`: UNCOPYABLE. The edge is in being first to the book. A copier is by construction the slower side of it.

## How to read this

A low wallet count clearing the bar is the expected result. The
measured false-positive rate of that gate is ~20% of populations,
so clearing it is necessary, not sufficient — persistence is the
test that matters. See POLYMARKET.md.
