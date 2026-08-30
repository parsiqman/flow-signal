# Polymarket wallet scan — 2026-08-30 12:19 UTC

## NO. Nothing clears the luck bar, and no out-of-sample test could be run on this sample.

| | |
|---|---|
| mode | live |
| wallets discovered | 400 |
| wallets scored (>=20 fills, >=10 markets) | 139 |
| t-stat needed to clear luck | 3.7 |
| best t observed | 3.04 |
| best edge observed | 28.27 c/share |
| **wallets clearing the bar (uniform)** | **0** |
| **wallet-category pairs clearing (specialists)** | **1** |
| concentrated-edge wallets surfaced | 2 |

## Persistence (the decisive test)

- `verdict`: only 10 wallets active in both periods; too few to conclude anything
- `n_selected`: 10

### Bias attribution — `0xa6a856a8c8a7f14fd9be6ae11c367c7cbb755009`

- `overall_edge`: 0.2827
- `extreme_band_stake`: 0.003
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0x38e59b36aae31b164200d0cad7c3fe5e0ee795e7`

- `overall_edge`: 0.2113
- `extreme_band_stake`: 0.689
- `verdict`: edge is concentrated in extreme prices -- likely harvesting favourite-longshot bias, which you should run directly rather than copy

### Bias attribution — `0xecaa8806a9a05049d7d5260a33dc924220e377a9`

- `overall_edge`: 0.1179
- `extreme_band_stake`: 0.185
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0xfe911a4e80ee71b47cd1ee690733ac4062e970ff`

- `overall_edge`: 0.1125
- `extreme_band_stake`: 0.676
- `verdict`: edge is concentrated in extreme prices -- likely harvesting favourite-longshot bias, which you should run directly rather than copy

### Bias attribution — `0x364c0e95e6126e28907b91aff028d5f0caa2e701`

- `overall_edge`: 0.1065
- `extreme_band_stake`: 0.264
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Trade style — `0xa6a856a8c8a7f14fd9be6ae11c367c7cbb755009`

- `n_markets`: 10
- `both_sides_frac`: 0.0
- `median_fills_per_market`: 4.5
- `median_span_hours`: 0.617
- `extreme_band_stake`: 0.003
- `avg_size_per_market`: 3896.7
- `style`: mixed / unclear
- `copyable`: UNCLEAR. The signature does not match a clean style; inspect the fills before drawing any conclusion.

### Trade style — `0x38e59b36aae31b164200d0cad7c3fe5e0ee795e7`

- `n_markets`: 24
- `both_sides_frac`: 0.458
- `median_fills_per_market`: 4.5
- `median_span_hours`: 50.638
- `extreme_band_stake`: 0.689
- `avg_size_per_market`: 2197.0
- `style`: bias harvester
- `copyable`: RUN THE RULE. Structural, available to anyone posting the same orders, and cheaper without the copy latency.

### Trade style — `0xecaa8806a9a05049d7d5260a33dc924220e377a9`

- `n_markets`: 13
- `both_sides_frac`: 0.231
- `median_fills_per_market`: 38.0
- `median_span_hours`: 2.319
- `extreme_band_stake`: 0.185
- `avg_size_per_market`: 19852.1
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

### Trade style — `0x364c0e95e6126e28907b91aff028d5f0caa2e701`

- `n_markets`: 12
- `both_sides_frac`: 0.0
- `median_fills_per_market`: 1.0
- `median_span_hours`: 0.0
- `extreme_band_stake`: 0.264
- `avg_size_per_market`: 75.1
- `style`: position taker
- `copyable`: COPYABLE IN PRINCIPLE. Check the slippage arithmetic before believing it survives execution.

## How to read this

A low wallet count clearing the bar is the expected result. The
measured false-positive rate of that gate is ~20% of populations,
so clearing it is necessary, not sufficient — persistence is the
test that matters. See POLYMARKET.md.
