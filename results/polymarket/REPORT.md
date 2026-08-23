# Polymarket wallet scan — 2026-08-23 06:54 UTC

## NO. No wallet is distinguishable from the luckiest of the population, and past performance does not predict future performance. There is nothing here safe to copy.

| | |
|---|---|
| mode | live |
| wallets discovered | 400 |
| wallets scored (>=20 fills, >=10 markets) | 155 |
| t-stat needed to clear luck | 3.7 |
| best t observed | 3.04 |
| best edge observed | 29.51 c/share |
| **wallets clearing the bar (uniform)** | **0** |
| **wallet-category pairs clearing (specialists)** | **2** |
| concentrated-edge wallets surfaced | 1 |

## Persistence (the decisive test)

- `n_wallets_both_periods`: 27
- `n_selected`: 2
- `selected_oos_edge`: 0.0056
- `everyone_else_oos_edge`: 0.0099
- `gap`: -0.0042
- `gap_t_stat`: -0.34
- `rank_correlation`: -0.022
- `verdict`: NO EVIDENCE that past performance predicts future performance -- copy trading has nothing to copy

### Bias attribution — `0x9b3dcd99eec7fe11602e6534e6302c0f318d7422`

- `overall_edge`: 0.2951
- `extreme_band_stake`: 0.0
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0xa6a856a8c8a7f14fd9be6ae11c367c7cbb755009`

- `overall_edge`: 0.2827
- `extreme_band_stake`: 0.003
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0x59ee6c6a56d7b00223f0c30f8002c4df762b684d`

- `overall_edge`: 0.2551
- `extreme_band_stake`: 0.301
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0xc23b2190e56399fae83048dea976e13d83cd24f9`

- `overall_edge`: 0.2084
- `extreme_band_stake`: 0.198
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0x509587cbb541251c74f261df3421f1fcc9fdc97c`

- `overall_edge`: 0.1577
- `extreme_band_stake`: 0.212
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Trade style — `0x9b3dcd99eec7fe11602e6534e6302c0f318d7422`

- `n_markets`: 22
- `both_sides_frac`: 0.0
- `median_fills_per_market`: 4.5
- `median_span_hours`: 0.127
- `extreme_band_stake`: 0.0
- `avg_size_per_market`: 64924.3
- `style`: mixed / unclear
- `copyable`: UNCLEAR. The signature does not match a clean style; inspect the fills before drawing any conclusion.

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

### Trade style — `0xc23b2190e56399fae83048dea976e13d83cd24f9`

- `n_markets`: 15
- `both_sides_frac`: 0.133
- `median_fills_per_market`: 2.0
- `median_span_hours`: 0.001
- `extreme_band_stake`: 0.198
- `avg_size_per_market`: 109047.4
- `style`: position taker
- `copyable`: COPYABLE IN PRINCIPLE. Check the slippage arithmetic before believing it survives execution.

### Trade style — `0x509587cbb541251c74f261df3421f1fcc9fdc97c`

- `n_markets`: 13
- `both_sides_frac`: 0.308
- `median_fills_per_market`: 7.0
- `median_span_hours`: 26.25
- `extreme_band_stake`: 0.212
- `avg_size_per_market`: 8950.4
- `style`: mixed / unclear
- `copyable`: UNCLEAR. The signature does not match a clean style; inspect the fills before drawing any conclusion.

## How to read this

A low wallet count clearing the bar is the expected result. The
measured false-positive rate of that gate is ~20% of populations,
so clearing it is necessary, not sufficient — persistence is the
test that matters. See POLYMARKET.md.
