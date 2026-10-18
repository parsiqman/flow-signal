# Polymarket wallet scan — 2026-10-18 10:45 UTC

## NO. Nothing clears the luck bar, and no out-of-sample test could be run on this sample.

| | |
|---|---|
| mode | live |
| wallets discovered | 400 |
| wallets scored (>=20 fills, >=10 markets) | 129 |
| t-stat needed to clear luck | 3.7 |
| best t observed | 3.08 |
| best edge observed | 30.75 c/share |
| **wallets clearing the bar (uniform)** | **0** |
| **wallet-category pairs clearing (specialists)** | **2** |
| concentrated-edge wallets surfaced | 1 |

## Persistence (the decisive test)

- `verdict`: only 19 wallets active in both periods; too few to conclude anything
- `n_selected`: 19

### Bias attribution — `0x9b3dcd99eec7fe11602e6534e6302c0f318d7422`

- `overall_edge`: 0.3075
- `extreme_band_stake`: 0.0
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

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

### Bias attribution — `0x15e06a47d9c665bda1703ff965ea76f8e56e8d86`

- `overall_edge`: 0.0973
- `extreme_band_stake`: 0.001
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Trade style — `0x9b3dcd99eec7fe11602e6534e6302c0f318d7422`

- `n_markets`: 22
- `both_sides_frac`: 0.0
- `median_fills_per_market`: 3.0
- `median_span_hours`: 0.053
- `extreme_band_stake`: 0.0
- `avg_size_per_market`: 62784.9
- `style`: position taker
- `copyable`: COPYABLE IN PRINCIPLE. Check the slippage arithmetic before believing it survives execution.

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

### Trade style — `0x15e06a47d9c665bda1703ff965ea76f8e56e8d86`

- `n_markets`: 36
- `both_sides_frac`: 0.139
- `median_fills_per_market`: 4.5
- `median_span_hours`: 0.313
- `extreme_band_stake`: 0.001
- `avg_size_per_market`: 64420.5
- `style`: mixed / unclear
- `copyable`: UNCLEAR. The signature does not match a clean style; inspect the fills before drawing any conclusion.

## How to read this

A low wallet count clearing the bar is the expected result. The
measured false-positive rate of that gate is ~20% of populations,
so clearing it is necessary, not sufficient — persistence is the
test that matters. See POLYMARKET.md.
