# Polymarket wallet scan — 2026-10-25 11:03 UTC

## NO. No wallet is distinguishable from the luckiest of the population, and past performance does not predict future performance. There is nothing here safe to copy.

| | |
|---|---|
| mode | live |
| wallets discovered | 400 |
| wallets scored (>=20 fills, >=10 markets) | 122 |
| t-stat needed to clear luck | 3.7 |
| best t observed | 3.08 |
| best edge observed | 22.36 c/share |
| **wallets clearing the bar (uniform)** | **0** |
| **wallet-category pairs clearing (specialists)** | **1** |
| concentrated-edge wallets surfaced | 1 |

## Persistence (the decisive test)

- `n_wallets_both_periods`: 21
- `n_selected`: 2
- `selected_oos_edge`: -0.0128
- `everyone_else_oos_edge`: 0.0069
- `gap`: -0.0197
- `gap_t_stat`: -0.92
- `rank_correlation`: 0.126
- `verdict`: NO EVIDENCE that past performance predicts future performance -- copy trading has nothing to copy

### Bias attribution — `0xfe911a4e80ee71b47cd1ee690733ac4062e970ff`

- `overall_edge`: 0.2236
- `extreme_band_stake`: 0.353
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0xa9d71818dadc207f9ea3d3f46ff0b12e497025e9`

- `overall_edge`: 0.1164
- `extreme_band_stake`: 0.424
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0x17525891a4ed52520ad75dac8ed5f32a58be7993`

- `overall_edge`: 0.1149
- `extreme_band_stake`: 0.74
- `verdict`: edge is concentrated in extreme prices -- likely harvesting favourite-longshot bias, which you should run directly rather than copy

### Bias attribution — `0xb74711992caf6d04fa55eecc46b8efc95311b050`

- `overall_edge`: 0.1012
- `extreme_band_stake`: 0.301
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0x22271273b28d0d198a8e5ff9e7f9f75d0c24b572`

- `overall_edge`: 0.0743
- `extreme_band_stake`: 0.498
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Trade style — `0xfe911a4e80ee71b47cd1ee690733ac4062e970ff`

- `n_markets`: 50
- `both_sides_frac`: 0.12
- `median_fills_per_market`: 3.0
- `median_span_hours`: 4.013
- `extreme_band_stake`: 0.353
- `avg_size_per_market`: 3022.4
- `style`: position taker
- `copyable`: COPYABLE IN PRINCIPLE. Check the slippage arithmetic before believing it survives execution.

### Trade style — `0xa9d71818dadc207f9ea3d3f46ff0b12e497025e9`

- `n_markets`: 73
- `both_sides_frac`: 0.027
- `median_fills_per_market`: 5.0
- `median_span_hours`: 0.987
- `extreme_band_stake`: 0.424
- `avg_size_per_market`: 928.4
- `style`: mixed / unclear
- `copyable`: UNCLEAR. The signature does not match a clean style; inspect the fills before drawing any conclusion.

### Trade style — `0x17525891a4ed52520ad75dac8ed5f32a58be7993`

- `n_markets`: 28
- `both_sides_frac`: 0.107
- `median_fills_per_market`: 2.5
- `median_span_hours`: 0.691
- `extreme_band_stake`: 0.74
- `avg_size_per_market`: 1719.6
- `style`: bias harvester
- `copyable`: RUN THE RULE. Structural, available to anyone posting the same orders, and cheaper without the copy latency.

### Trade style — `0xb74711992caf6d04fa55eecc46b8efc95311b050`

- `n_markets`: 17
- `both_sides_frac`: 0.176
- `median_fills_per_market`: 4.0
- `median_span_hours`: 5.916
- `extreme_band_stake`: 0.301
- `avg_size_per_market`: 7555.3
- `style`: mixed / unclear
- `copyable`: UNCLEAR. The signature does not match a clean style; inspect the fills before drawing any conclusion.

### Trade style — `0x22271273b28d0d198a8e5ff9e7f9f75d0c24b572`

- `n_markets`: 38
- `both_sides_frac`: 0.316
- `median_fills_per_market`: 2.0
- `median_span_hours`: 0.659
- `extreme_band_stake`: 0.498
- `avg_size_per_market`: 3962.4
- `style`: mixed / unclear
- `copyable`: UNCLEAR. The signature does not match a clean style; inspect the fills before drawing any conclusion.

## How to read this

A low wallet count clearing the bar is the expected result. The
measured false-positive rate of that gate is ~20% of populations,
so clearing it is necessary, not sufficient — persistence is the
test that matters. See POLYMARKET.md.
