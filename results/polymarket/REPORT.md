# Polymarket wallet scan — 2026-10-11 12:44 UTC

## WEAK SIGNAL. No individual wallet clears the luck bar, but past performance does predict future performance in aggregate. Worth a larger scan.

| | |
|---|---|
| mode | live |
| wallets discovered | 400 |
| wallets scored (>=20 fills, >=10 markets) | 111 |
| t-stat needed to clear luck | 3.7 |
| best t observed | 3.08 |
| best edge observed | 22.78 c/share |
| **wallets clearing the bar (uniform)** | **0** |
| **wallet-category pairs clearing (specialists)** | **1** |
| concentrated-edge wallets surfaced | 0 |

## Persistence (the decisive test)

- `n_wallets_both_periods`: 23
- `n_selected`: 2
- `selected_oos_edge`: 0.0133
- `everyone_else_oos_edge`: 0.0032
- `gap`: 0.0102
- `gap_t_stat`: 4.11
- `rank_correlation`: 0.1057
- `verdict`: past performance predicts future performance

### Bias attribution — `0x17525891a4ed52520ad75dac8ed5f32a58be7993`

- `overall_edge`: 0.2278
- `extreme_band_stake`: 0.421
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0xfe911a4e80ee71b47cd1ee690733ac4062e970ff`

- `overall_edge`: 0.2236
- `extreme_band_stake`: 0.353
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0x83b9f9e2d28ab875b9c9f73cdd5535c4c8926124`

- `overall_edge`: 0.2099
- `extreme_band_stake`: 0.001
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0xa9d71818dadc207f9ea3d3f46ff0b12e497025e9`

- `overall_edge`: 0.1919
- `extreme_band_stake`: 0.435
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0xcdc495702d5caabf60301025c5a2dd4c1a66a880`

- `overall_edge`: 0.1074
- `extreme_band_stake`: 0.078
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

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

### Trade style — `0x83b9f9e2d28ab875b9c9f73cdd5535c4c8926124`

- `n_markets`: 17
- `both_sides_frac`: 0.0
- `median_fills_per_market`: 5.0
- `median_span_hours`: 0.484
- `extreme_band_stake`: 0.001
- `avg_size_per_market`: 9602.2
- `style`: mixed / unclear
- `copyable`: UNCLEAR. The signature does not match a clean style; inspect the fills before drawing any conclusion.

### Trade style — `0xa9d71818dadc207f9ea3d3f46ff0b12e497025e9`

- `n_markets`: 71
- `both_sides_frac`: 0.028
- `median_fills_per_market`: 5.0
- `median_span_hours`: 1.087
- `extreme_band_stake`: 0.435
- `avg_size_per_market`: 1021.7
- `style`: mixed / unclear
- `copyable`: UNCLEAR. The signature does not match a clean style; inspect the fills before drawing any conclusion.

### Trade style — `0xcdc495702d5caabf60301025c5a2dd4c1a66a880`

- `n_markets`: 34
- `both_sides_frac`: 0.265
- `median_fills_per_market`: 3.0
- `median_span_hours`: 0.301
- `extreme_band_stake`: 0.078
- `avg_size_per_market`: 1698.6
- `style`: mixed / unclear
- `copyable`: UNCLEAR. The signature does not match a clean style; inspect the fills before drawing any conclusion.

## How to read this

A low wallet count clearing the bar is the expected result. The
measured false-positive rate of that gate is ~20% of populations,
so clearing it is necessary, not sufficient — persistence is the
test that matters. See POLYMARKET.md.
