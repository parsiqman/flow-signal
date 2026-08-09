# Polymarket wallet scan — 2026-08-09 07:31 UTC

## NO. No wallet is distinguishable from the luckiest of the population, and past performance does not predict future performance. There is nothing here safe to copy.

| | |
|---|---|
| mode | live |
| wallets discovered | 400 |
| wallets scored (>=20 fills, >=10 markets) | 141 |
| t-stat needed to clear luck | 3.7 |
| best t observed | 3.04 |
| best edge observed | 28.27 c/share |
| **wallets clearing the bar (uniform)** | **0** |
| **wallet-category pairs clearing (specialists)** | **1** |
| concentrated-edge wallets surfaced | 1 |

## Persistence (the decisive test)

- `n_wallets_both_periods`: 21
- `n_selected`: 2
- `selected_oos_edge`: 0.0674
- `everyone_else_oos_edge`: -0.0056
- `gap`: 0.073
- `gap_t_stat`: 1.91
- `rank_correlation`: 0.1506
- `verdict`: NO EVIDENCE that past performance predicts future performance -- copy trading has nothing to copy

### Bias attribution — `0xa6a856a8c8a7f14fd9be6ae11c367c7cbb755009`

- `overall_edge`: 0.2827
- `extreme_band_stake`: 0.003
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0x59ee6c6a56d7b00223f0c30f8002c4df762b684d`

- `overall_edge`: 0.2551
- `extreme_band_stake`: 0.301
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0x27abdfc9393c72a6330a3be987da4b46c726e521`

- `overall_edge`: 0.1641
- `extreme_band_stake`: 0.033
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0xa9d71818dadc207f9ea3d3f46ff0b12e497025e9`

- `overall_edge`: 0.1202
- `extreme_band_stake`: 0.404
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0xfe911a4e80ee71b47cd1ee690733ac4062e970ff`

- `overall_edge`: 0.1125
- `extreme_band_stake`: 0.676
- `verdict`: edge is concentrated in extreme prices -- likely harvesting favourite-longshot bias, which you should run directly rather than copy

### Trade style — `0xa6a856a8c8a7f14fd9be6ae11c367c7cbb755009`

- `n_markets`: 10
- `both_sides_frac`: 0.0
- `median_fills_per_market`: 4.5
- `median_span_hours`: 0.617
- `extreme_band_stake`: 0.003
- `avg_size_per_market`: 3896.7
- `style`: mixed / unclear
- `copyable`: UNCLEAR. The signature does not match a clean style; inspect the fills before drawing any conclusion.

### Trade style — `0x59ee6c6a56d7b00223f0c30f8002c4df762b684d`

- `n_markets`: 16
- `both_sides_frac`: 0.062
- `median_fills_per_market`: 19.5
- `median_span_hours`: 6.066
- `extreme_band_stake`: 0.301
- `avg_size_per_market`: 82075.8
- `style`: mixed / unclear
- `copyable`: UNCLEAR. The signature does not match a clean style; inspect the fills before drawing any conclusion.

### Trade style — `0x27abdfc9393c72a6330a3be987da4b46c726e521`

- `n_markets`: 11
- `both_sides_frac`: 0.0
- `median_fills_per_market`: 2.0
- `median_span_hours`: 0.103
- `extreme_band_stake`: 0.033
- `avg_size_per_market`: 358.8
- `style`: position taker
- `copyable`: COPYABLE IN PRINCIPLE. Check the slippage arithmetic before believing it survives execution.

### Trade style — `0xa9d71818dadc207f9ea3d3f46ff0b12e497025e9`

- `n_markets`: 70
- `both_sides_frac`: 0.014
- `median_fills_per_market`: 5.0
- `median_span_hours`: 0.846
- `extreme_band_stake`: 0.404
- `avg_size_per_market`: 942.5
- `style`: mixed / unclear
- `copyable`: UNCLEAR. The signature does not match a clean style; inspect the fills before drawing any conclusion.

### Trade style — `0xfe911a4e80ee71b47cd1ee690733ac4062e970ff`

- `n_markets`: 87
- `both_sides_frac`: 0.172
- `median_fills_per_market`: 3.0
- `median_span_hours`: 4.144
- `extreme_band_stake`: 0.676
- `avg_size_per_market`: 3916.8
- `style`: bias harvester
- `copyable`: RUN THE RULE. Structural, available to anyone posting the same orders, and cheaper without the copy latency.

## How to read this

A low wallet count clearing the bar is the expected result. The
measured false-positive rate of that gate is ~20% of populations,
so clearing it is necessary, not sufficient — persistence is the
test that matters. See POLYMARKET.md.
