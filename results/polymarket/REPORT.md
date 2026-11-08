# Polymarket wallet scan — 2026-11-08 12:54 UTC

## NO. Nothing clears the luck bar, and no out-of-sample test could be run on this sample.

| | |
|---|---|
| mode | live |
| wallets discovered | 400 |
| wallets scored (>=20 fills, >=10 markets) | 74 |
| t-stat needed to clear luck | 3.7 |
| best t observed | 3.08 |
| best edge observed | 22.78 c/share |
| **wallets clearing the bar (uniform)** | **0** |
| **wallet-category pairs clearing (specialists)** | **1** |
| concentrated-edge wallets surfaced | 0 |

## Persistence (the decisive test)

- `verdict`: only 17 wallets active in both periods; too few to conclude anything
- `n_selected`: 17

### Bias attribution — `0x17525891a4ed52520ad75dac8ed5f32a58be7993`

- `overall_edge`: 0.2278
- `extreme_band_stake`: 0.421
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0xfe911a4e80ee71b47cd1ee690733ac4062e970ff`

- `overall_edge`: 0.2236
- `extreme_band_stake`: 0.353
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0xcdc495702d5caabf60301025c5a2dd4c1a66a880`

- `overall_edge`: 0.1074
- `extreme_band_stake`: 0.078
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0x22271273b28d0d198a8e5ff9e7f9f75d0c24b572`

- `overall_edge`: 0.071
- `extreme_band_stake`: 0.463
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0x53d2d3c78597a78402d4db455a680da7ef560c3f`

- `overall_edge`: 0.0558
- `extreme_band_stake`: 0.946
- `verdict`: edge is concentrated in extreme prices -- likely harvesting favourite-longshot bias, which you should run directly rather than copy

### Trade style — `0x17525891a4ed52520ad75dac8ed5f32a58be7993`

- `n_markets`: 19
- `both_sides_frac`: 0.211
- `median_fills_per_market`: 2.0
- `median_span_hours`: 1.203
- `extreme_band_stake`: 0.421
- `avg_size_per_market`: 1165.6
- `style`: mixed / unclear
- `copyable`: UNCLEAR. The signature does not match a clean style; inspect the fills before drawing any conclusion.

### Trade style — `0xfe911a4e80ee71b47cd1ee690733ac4062e970ff`

- `n_markets`: 50
- `both_sides_frac`: 0.12
- `median_fills_per_market`: 3.0
- `median_span_hours`: 4.013
- `extreme_band_stake`: 0.353
- `avg_size_per_market`: 3022.4
- `style`: position taker
- `copyable`: COPYABLE IN PRINCIPLE. Check the slippage arithmetic before believing it survives execution.

### Trade style — `0xcdc495702d5caabf60301025c5a2dd4c1a66a880`

- `n_markets`: 34
- `both_sides_frac`: 0.265
- `median_fills_per_market`: 3.0
- `median_span_hours`: 0.301
- `extreme_band_stake`: 0.078
- `avg_size_per_market`: 1698.6
- `style`: mixed / unclear
- `copyable`: UNCLEAR. The signature does not match a clean style; inspect the fills before drawing any conclusion.

### Trade style — `0x22271273b28d0d198a8e5ff9e7f9f75d0c24b572`

- `n_markets`: 37
- `both_sides_frac`: 0.297
- `median_fills_per_market`: 2.0
- `median_span_hours`: 0.028
- `extreme_band_stake`: 0.463
- `avg_size_per_market`: 3882.1
- `style`: mixed / unclear
- `copyable`: UNCLEAR. The signature does not match a clean style; inspect the fills before drawing any conclusion.

### Trade style — `0x53d2d3c78597a78402d4db455a680da7ef560c3f`

- `n_markets`: 39
- `both_sides_frac`: 0.231
- `median_fills_per_market`: 2.0
- `median_span_hours`: 0.057
- `extreme_band_stake`: 0.946
- `avg_size_per_market`: 3879.8
- `style`: bias harvester
- `copyable`: RUN THE RULE. Structural, available to anyone posting the same orders, and cheaper without the copy latency.

## How to read this

A low wallet count clearing the bar is the expected result. The
measured false-positive rate of that gate is ~20% of populations,
so clearing it is necessary, not sufficient — persistence is the
test that matters. See POLYMARKET.md.
