# Polymarket wallet scan — 2026-08-16 06:58 UTC

## NO. No wallet is distinguishable from the luckiest of the population, and past performance does not predict future performance. There is nothing here safe to copy.

| | |
|---|---|
| mode | live |
| wallets discovered | 400 |
| wallets scored (>=20 fills, >=10 markets) | 156 |
| t-stat needed to clear luck | 3.7 |
| best t observed | 3.08 |
| best edge observed | 35.22 c/share |
| **wallets clearing the bar (uniform)** | **0** |
| **wallet-category pairs clearing (specialists)** | **1** |
| concentrated-edge wallets surfaced | 4 |

## Persistence (the decisive test)

- `n_wallets_both_periods`: 24
- `n_selected`: 2
- `selected_oos_edge`: -0.0272
- `everyone_else_oos_edge`: 0.0079
- `gap`: -0.0351
- `gap_t_stat`: -1.51
- `rank_correlation`: -0.0861
- `verdict`: NO EVIDENCE that past performance predicts future performance -- copy trading has nothing to copy

### Bias attribution — `0x4a9a87962893e6e119f1cc7f67c61d0287e0f675`

- `overall_edge`: 0.3522
- `extreme_band_stake`: 0.184
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0xa6a856a8c8a7f14fd9be6ae11c367c7cbb755009`

- `overall_edge`: 0.2827
- `extreme_band_stake`: 0.003
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0x59ee6c6a56d7b00223f0c30f8002c4df762b684d`

- `overall_edge`: 0.2551
- `extreme_band_stake`: 0.301
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0x509587cbb541251c74f261df3421f1fcc9fdc97c`

- `overall_edge`: 0.1577
- `extreme_band_stake`: 0.212
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0xb48b9192dc52eed724fa58c66fa8926d06a3648e`

- `overall_edge`: 0.1553
- `extreme_band_stake`: 0.754
- `verdict`: edge is concentrated in extreme prices -- likely harvesting favourite-longshot bias, which you should run directly rather than copy

### Trade style — `0x4a9a87962893e6e119f1cc7f67c61d0287e0f675`

- `n_markets`: 23
- `both_sides_frac`: 0.217
- `median_fills_per_market`: 1.0
- `median_span_hours`: 0.0
- `extreme_band_stake`: 0.184
- `avg_size_per_market`: 144285.4
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

### Trade style — `0x509587cbb541251c74f261df3421f1fcc9fdc97c`

- `n_markets`: 13
- `both_sides_frac`: 0.308
- `median_fills_per_market`: 7.0
- `median_span_hours`: 26.25
- `extreme_band_stake`: 0.212
- `avg_size_per_market`: 8950.4
- `style`: mixed / unclear
- `copyable`: UNCLEAR. The signature does not match a clean style; inspect the fills before drawing any conclusion.

### Trade style — `0xb48b9192dc52eed724fa58c66fa8926d06a3648e`

- `n_markets`: 127
- `both_sides_frac`: 0.055
- `median_fills_per_market`: 2.0
- `median_span_hours`: 1.599
- `extreme_band_stake`: 0.754
- `avg_size_per_market`: 4291.9
- `style`: bias harvester
- `copyable`: RUN THE RULE. Structural, available to anyone posting the same orders, and cheaper without the copy latency.

## How to read this

A low wallet count clearing the bar is the expected result. The
measured false-positive rate of that gate is ~20% of populations,
so clearing it is necessary, not sufficient — persistence is the
test that matters. See POLYMARKET.md.
