# Polymarket wallet scan — 2026-09-06 10:51 UTC

## NO. No wallet is distinguishable from the luckiest of the population, and past performance does not predict future performance. There is nothing here safe to copy.

| | |
|---|---|
| mode | live |
| wallets discovered | 400 |
| wallets scored (>=20 fills, >=10 markets) | 148 |
| t-stat needed to clear luck | 3.7 |
| best t observed | 3.04 |
| best edge observed | 32.73 c/share |
| **wallets clearing the bar (uniform)** | **0** |
| **wallet-category pairs clearing (specialists)** | **1** |
| concentrated-edge wallets surfaced | 3 |

## Persistence (the decisive test)

- `n_wallets_both_periods`: 27
- `n_selected`: 2
- `selected_oos_edge`: 0.0056
- `everyone_else_oos_edge`: 0.0057
- `gap`: -0.0
- `gap_t_stat`: -0.0
- `rank_correlation`: 0.3101
- `verdict`: NO EVIDENCE that past performance predicts future performance -- copy trading has nothing to copy

### Bias attribution — `0x208e7d17e83f757e68bc0d0aecfca9d67c3ebcb3`

- `overall_edge`: 0.3273
- `extreme_band_stake`: 0.444
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0x59ee6c6a56d7b00223f0c30f8002c4df762b684d`

- `overall_edge`: 0.2551
- `extreme_band_stake`: 0.301
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0x38e59b36aae31b164200d0cad7c3fe5e0ee795e7`

- `overall_edge`: 0.2113
- `extreme_band_stake`: 0.689
- `verdict`: edge is concentrated in extreme prices -- likely harvesting favourite-longshot bias, which you should run directly rather than copy

### Bias attribution — `0xc23b2190e56399fae83048dea976e13d83cd24f9`

- `overall_edge`: 0.2084
- `extreme_band_stake`: 0.198
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Bias attribution — `0xa9d71818dadc207f9ea3d3f46ff0b12e497025e9`

- `overall_edge`: 0.1202
- `extreme_band_stake`: 0.404
- `verdict`: edge is spread across price bands -- not obviously a bias-harvesting rule

### Trade style — `0x208e7d17e83f757e68bc0d0aecfca9d67c3ebcb3`

- `n_markets`: 10
- `both_sides_frac`: 0.3
- `median_fills_per_market`: 48.5
- `median_span_hours`: 57.901
- `extreme_band_stake`: 0.444
- `avg_size_per_market`: 36095.0
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

### Trade style — `0x38e59b36aae31b164200d0cad7c3fe5e0ee795e7`

- `n_markets`: 24
- `both_sides_frac`: 0.458
- `median_fills_per_market`: 4.5
- `median_span_hours`: 50.638
- `extreme_band_stake`: 0.689
- `avg_size_per_market`: 2197.0
- `style`: bias harvester
- `copyable`: RUN THE RULE. Structural, available to anyone posting the same orders, and cheaper without the copy latency.

### Trade style — `0xc23b2190e56399fae83048dea976e13d83cd24f9`

- `n_markets`: 15
- `both_sides_frac`: 0.133
- `median_fills_per_market`: 2.0
- `median_span_hours`: 0.001
- `extreme_band_stake`: 0.198
- `avg_size_per_market`: 109047.4
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

## How to read this

A low wallet count clearing the bar is the expected result. The
measured false-positive rate of that gate is ~20% of populations,
so clearing it is necessary, not sufficient — persistence is the
test that matters. See POLYMARKET.md.
