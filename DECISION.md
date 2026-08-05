# Market Selection: Crypto/DeFi vs Traditional Equities

**Decision date:** 2026-08-05
**Reproduce with:** `python src/market_selection/run_analysis.py`
**Verify the machinery:** `python tests/test_market_selection.py` (20 tests)

---

## Recommendation

**Build on US cash equities.** Cross-sectional, dollar-neutral, 1–3 day holding
period, ~1500-name liquid universe, via Alpaca.

Equities won 96.6% of 20,000 random criterion weightings and all seven stress
scenarios. This is not a close call, and it is not an artifact of how the
scorecard was weighted — the weight sensitivity test exists specifically to
catch that failure, and it cleared.

**Crypto DeFi is ruled out — but not for the reason it usually is.** Gas is a
non-issue. The reasons are cost floor, statistical power, and two 2026
regulatory changes that removed crypto's structural advantages.

---

## The question needed reframing first

"Crypto DeFi or traditional stocks" is a two-way framing of a five-way choice,
and the interesting variation turns out to be *within* each camp, not between
them. Offshore DeFi perps and on-chain AMM swaps have almost nothing in common
economically; neither do cash equities and short-dated options. Collapsing them
hides the decision. The six venues actually evaluated:

| Venue | Camp |
|---|---|
| US cash equities (liquid universe) | traditional |
| US equity options, short-dated event-driven — *the current repo's thesis* | traditional |
| Crypto spot on a US CEX (Kraken Pro / Coinbase Advanced) | crypto |
| Crypto perps, CFTC-regulated US (Kraken/Bitnomial, Coinbase FM) | crypto |
| On-chain DeFi (Uniswap/Aerodrome on Base/Arbitrum) | crypto-defi |
| Offshore DeFi perps (Hyperliquid) | crypto-defi |

---

## Three findings that decide it

### 1. Crypto's cost floor forces long holding periods, and long holding periods destroy statistical power

The naive comparison — "crypto fees are 19x equity fees" —
is true but misleading. What matters is friction relative to the move you are forecasting,
because a 40bp fee in a market with 4% daily vol is proportionally cheaper than
a 2bp fee in a market with 1.4% daily vol. Crypto's volatility genuinely pays
for a large part of its fees:

| | round-trip cost | cost ÷ move size (5-day hold) |
|---|---|---|
| US equities | 5.0 bps | 0.018 |
| Crypto spot (US CEX) | 94.0 bps | 0.105 |
| ratio | **18.8x worse** | **5.8x worse** |

So crypto is penalised less than the fee headline suggests. It is still
penalised, and the penalty lands somewhere specific — the **minimum viable
holding period**, the shortest horizon at which a signal of given skill still
clears costs:

| Venue | min horizon @ IC=0.05 |
|---|---|
| US cash equities | **13 hours** |
| Offshore DeFi perps (Hyperliquid) | 18 hours |
| US crypto perps | 3 days |
| On-chain DeFi | 13.5 days |
| Crypto spot (US CEX) | **22.5 days** |

At retail fee tiers you cannot trade crypto spot faster than roughly monthly.
That is not a trading algo; it is a slow rebalancing scheme. And it feeds
directly into the finding below, because holding period is the denominator of
bets per year.

### 2. The statistical-power gap is 40–100x, and power is the binding constraint

CLAUDE.md's priority #1 is walk-forward validation: tune on period A, evaluate
frozen on period B. That is only possible in a venue that supplies enough
independent bets to make period B mean anything.

Running Grinold's law net of costs — `IR = TC × (IC − cost/σ) × √breadth`, with
breadth discounted for cross-asset correlation and a 0.5 transfer coefficient —
and granting **each venue its own most favourable skill assumption**:

| Venue | assumed IC | bets/yr | net IR | yrs to t=2 | free history covers |
|---|---|---|---|---|---|
| US cash equities | 0.04 | 4,067 | 0.57 | 12.2 | **0.41x** |
| US equity options (event) | 0.14 | 100 | 0.49 | 16.8 | 0.00x |
| On-chain DeFi | 0.06 | 47 | 0.13 | 252 | 0.010x |
| Offshore DeFi perps | 0.05 | 162 | 0.12 | 260 | 0.008x |
| Crypto spot (US CEX) | 0.05 | 37 | 0.06 | 1,043 | 0.004x |
| US crypto perps | 0.04 | 44 | 0.01 | 37,468 | 0.000x |

Crypto's inefficiency premium is granted here at face value — DeFi is given a
*higher* IC than equities — and it still loses by two orders of magnitude. The
mechanism is mechanical and hard to argue with: **high fees force long horizons,
long horizons mean few bets, few bets mean no proof.** US crypto perps are the
extreme case — cheap fees, but a four-instrument universe at 0.8 correlation
provides essentially no breadth, and √breadth never rescues it.

A related consequence, from the Monte Carlo: after two years of data, a
**zero-edge** strategy clears t=2 anyway 1.1% of the time in crypto spot versus
0.0% in equities. Crypto backtests don't just prove less — they lie more, and
for the same reason (scarce independent bets).

### 3. Two 2026 rule changes removed crypto's structural advantages

- **The PDT rule is gone.** The SEC approved FINRA's amendments to Rule 4210 on
  2026-04-14, effective 2026-06-04, eliminating the $25,000 minimum equity
  requirement and the pattern-day-trader designation. The single biggest
  structural reason a small account chose crypto over equities no longer exists.
- **The best DeFi venue is unavailable.** Hyperliquid — the only crypto venue
  combining low fees (4.5bp taker) with a deep 150-instrument universe — treats
  US persons as Restricted Persons, enforces it by geofencing, and its terms
  forbid the VPN workaround. It holds no CFTC/SEC registration to offer
  leveraged perps to US persons. It is the strongest crypto candidate on merit
  and it is a hard gate.

  Worth noting: **the gate is not what decides against it.** With gates
  disabled, Hyperliquid still never reaches the top two on the evidence alone
  (asserted in `test_gates_can_be_disabled_for_counterfactuals`).

---

## Supporting findings

**Gas is not why DeFi loses.** Post-EIP-4844, an L2 swap costs $0.01–0.05 —
0.06bp on a $5,000 clip. AMM pool fees (~25bp) plus price impact (~20bp) are
roughly **700x larger** than gas. Halving DeFi pool fees in the stress test
doesn't change the ranking; gas is a rounding error. Anyone ruling out DeFi on
gas costs is reaching the right conclusion by the wrong route.

**Equity data is now free and deep, which neutralises crypto's oldest
advantage.** Alpaca provides ~10 years of 1-minute US equity bars free to open
accounts, including unfunded ones. Crypto's "free complete history" pitch used
to be decisive against $199/mo Polygon-style equity feeds. It no longer is.

**24/7 markets are worse for this setup, not better.** Counterintuitive but
firm: a 4pm close is a free batch boundary. Equities need one Render cron per
day. A 24/7 market has no natural boundary, needs an always-on worker, and —
with leverage — needs liquidation monitoring a cron cannot provide. For a
Chromebook + Colab + Render user with no local machine, market hours are a
feature.

**Annual cost drag at $25,000, each venue at its own best horizon:** equities
4.5%/yr, crypto spot 5.4%, on-chain DeFi 6.4%, US perps 12.8%, Hyperliquid
22.5%. Equities is cheapest *despite* turning over 84x/year versus crypto
spot's 5.8x — that is the cost floor advantage compounding.

---

## What this does *not* say

- **It does not say crypto is unprofitable.** It says a solo researcher cannot
  *prove* an edge there within the available history, which is a different and
  narrower claim. A well-capitalised desk paying 5bp fee tiers faces entirely
  different economics.
- **It does not say equities is easy.** The honest reading of the power table is
  that at realistic skill (IC 0.04), even the winner needs ~12 years to reach
  t=2 and free history covers only 0.41x of that. Equities is the only venue
  within striking distance of feasible — not a venue where success is likely.
  The gap closes at IC≈0.05 (4.8 years, fully covered by 10 years of history),
  so **the equity path lives or dies on finding a signal slightly better than
  the published-anomaly baseline.**
- **It is not a backtest.** It is a decision model over parameters taken from
  published fee schedules and market-structure reporting. The parameters are
  the argument; they live in one file (`src/market_selection/venues.py`) so
  they can be disagreed with directly.

---

## What would change the answer

| If this turned out to be true | Effect |
|---|---|
| CFTC onshores Hyperliquid-style perps to US persons (Chair Selig has said this is planned) | Reopens the strongest crypto venue; re-run required |
| You reach a maker-only crypto fee tier below ~10bp round trip | Crypto spot's minimum horizon drops from 22.5d toward days; breadth recovers |
| Your realistic equity IC is below 0.03 | Equities becomes unvalidatable too (43+ years); reconsider everything |
| You will pay $250/mo for options flow data | Reopens the existing repo's thesis — it has the highest per-bet edge of any venue |
| Capital rises above ~$500k | Cost floors matter less; the ranking compresses |

Stress-tested and **did not** change the answer: US perp fees 3x worse, crypto
vol 50% higher, equity spreads 2x worse, crypto all-maker fills, equity breadth
halved, DeFi pool fees halved.

---

## What happens to the existing M&A options work

**Park it. Do not delete it.** It is the highest-scoring venue on repo carryover
(10/10) and carries the highest plausible per-bet edge of all six (IC≈0.14,
derived from CLAUDE.md's own established 6–10% precision with 10–40x payoffs —
and reproduced by `src/flow_backtest.py`, which still prints 6.1% precision and
+33% ROI under the catalyst filter).

It fails on exactly one criterion: **data**. Historical options flow plus NBBO
is paid-only, with no free tier deep enough for a walk-forward backtest. Its
`history_cover` is 0.00x — not marginal, absent. CLAUDE.md's priority #1 cannot
be executed against it at any price the project has agreed to pay.

What transfers to the equity work: `detect_signals` / `evaluate` /
`sweep_thresholds` are a generic signal→alert→precision/recall harness. The
config-sweep discipline, the catalyst-exclusion idea, and the synthetic-market
generator's insistence on modelling look-alike false positives all carry over
directly. Only the options-specific filters go.

---

## Honest limitation of this analysis

**No parameter here was measured; all were sourced from published figures.**
This sandbox's egress policy blocks every market-data host (Binance, Coinbase,
Kraken, Hyperliquid, DefiLlama and sec.gov all returned 403 on CONNECT), so
live verification was impossible here. Colab does not have this restriction.

**First action item: verify the parameter table from Colab before building on
it.** The three least-certain inputs, in order:

1. `CRYPTO_PERPS_US.commission_bps = 6.0` — the US onshore perp fee schedule is
   new and was not directly confirmed. (Stress-tested at 18bp; ranking held.)
2. `US_EQUITIES.half_spread_bps = 2.0` — blended across a 1500-name universe;
   measurable directly from Alpaca quote data.
3. `residual_corr` for every venue — drives breadth, and breadth drives the
   headline conclusion. Measurable from returns data in an afternoon.

---

## Next task

Cross-sectional equity signal research on Alpaca data, 1–3 day horizon, with
period A/period B walk-forward split fixed **before** any tuning, per CLAUDE.md
hard rule #2. Target: an IC of 0.05 or better, which is the threshold at which
10 years of free history becomes sufficient to validate honestly.

---

## Sources

Fee schedules, market structure, and regulatory status (accessed 2026-08-05):

- [FINRA Regulatory Notice 26-10 — intraday margin requirements](https://www.finra.org/rules-guidance/notices/26-10) and [FINRA: Understanding the New Intraday Margin Requirements](https://www.finra.org/investors/insights/intraday-margin-requirements) — PDT elimination
- [SEC approval order SR-FINRA-2025-017](https://www.sec.gov/files/rules/sro/finra/2026/34-105226.pdf)
- [Kraken fees guide 2026](https://www.cryptoryancy.com/kraken-fees-complete-guide-2026/) and [Coinbase Advanced Trade fees 2026](https://tokenecho.io/guides/coinbase-advanced-trade-fees/)
- [Hyperliquid vs Binance/Bybit perp fees 2026](https://bitsgap.com/blog/hyperliquid-fees-vs-binance-and-bybit-whats-actually-cheaper)
- [Is Hyperliquid available in the US? (2026)](https://www.datawallet.com/crypto/is-hyperliquid-available-in-the-usa) and [How US traders can engage with Hyperliquid](https://www.buildix.trade/blog/how-to-trade-hyperliquid-us-access-options-2026)
- [Kraken: CFTC-regulated perpetual futures for US traders](https://blog.kraken.com/product/kraken-derivatives/announcing-cftc-regulated-us-perps) and [Coinbase: perpetual futures have arrived in the US](https://www.coinbase.com/blog/perpetual-futures-have-arrived-in-the-us)
- [L2 gas fee statistics 2026](https://coinlaw.io/gas-fee-markets-on-layer-2-statistics/) and [DEX fees explained 2026](https://www.alphaexcapital.com/cryptocurrencies/defi-web3-and-nfts/decentralized-exchanges-and-swaps/dex-fees-explained)
- [Alpaca market data](https://alpaca.markets/data) and [Alpaca algorithmic trading](https://alpaca.markets/algotrading)
- [Best market data APIs for algorithmic trading 2026](https://www.alphanume.com/blog/best-market-data-apis-for-algorithmic-trading-in-2026) — Polygon free-tier limits
- [CryptoDataDownload free historical OHLCV](https://www.cryptodatadownload.com/data/) and [Kraken downloadable OHLCVT archives](https://support.kraken.com/articles/360047124832-downloadable-historical-ohlcvt-open-high-low-close-volume-trades-data)
- [DeFiLlama free API](https://eco.com/support/en/articles/14800367-defillama-free-tvl-and-defi-analytics) and [Dune Analytics guide 2026](https://www.dextools.io/tutorials/how-to-use-dune-analytics-on-chain-dashboard-tutorial-2026)
- [CLARITY Act status 2026](https://en.cryptonomist.ch/2026/08/03/clarity-act-crypto-regulation-4/) — ~30% passage odds per Galaxy Research
- [IRS Instructions for Form 1099-DA (2026)](https://www.irs.gov/instructions/i1099da) — digital asset basis reporting phase-in
- [A Trend Factor for the Cross Section of Cryptocurrency Returns, JFQA](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/trend-factor-for-the-cross-section-of-cryptocurrency-returns/4C1509ACBA33D5DCAF0AC24379148178) and [Cryptocurrency Market Efficiency Revisited](https://www.worldscientific.com/doi/10.1142/S2424786325500215) — crypto inefficiency evidence
