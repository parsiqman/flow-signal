# Polymarket Copy Trading — feasibility

```bash
python tests/test_polymarket.py     # 45 tests
```

**Status: run against live data.** See "The first real wallet" below. Everything
else is either a checked fact about the venue or a result on populations with
known ground truth.

---

## The first real wallet (2026-08-07)

One account was named from outside this dataset — a weather specialist,
`0x6011655c4afb76f36dd1b08a137a1ba73466b31e` — and scored on its complete
record. It is the first thing in this repo to clear its pre-registered bar.

| | |
|---|---|
| fills / markets | 5,307 / 1,831 (100% of the record matched) |
| edge | **4.45 c/share**, ROI 9.8% |
| t-statistic | **10.19** |
| bar if genuinely pre-specified (N=1) | 1.64 — cleared |
| bar if inheriting someone's search (N=1000) | 3.92 — cleared |
| effective sample (Kish) | 810 markets |
| own-history split | early t=7.41, late t=7.10, decay +0.003 |

Both halves of its own history clear even the inherited-search bar
independently, so this is not a streak.

**And you should still not copy it.** 83% of stake sits in the extreme price
bands: this is favourite-longshot harvesting, a documented structural bias, not
private information. Copying pays 2.93c of cross-venue slippage against a 4.45c
edge — 34% retained, ~17%/yr — to rent a rule you can run yourself. The
deliverable here is the rule, not the wallet.

Three pipeline bugs stood between the first run and this number, all of the
same species: **a data path that returns the wrong thing looks exactly like a
negative result.** Gamma answers 200 with its default listing for a parameter
it does not recognise (1,840 politics markets for a weather trader, 0 fills
matched); then it answered for only 13 of 1,831 ids, enough to print a
confident "NO" off an effective sample of 4.7; then the cross-sectional
persistence test, which cannot run on a single wallet, had its "too few
wallets" non-answer read as a failed test. Nothing about the trader changed
between the first verdict and this one.

---

## The headline finding

Two numbers, computed independently, that collide:

| | Leader's edge required |
|---|---|
| To make you **20% a year** after all copying costs | **4.8 cents/share** |
| To be **distinguishable from luck** among 10,000 wallets | **15.7 cents/share** |

**The identification bar is three times the profitability bar.** There is a wide
band of traders who would genuinely make you money and whom you cannot reliably
tell apart from lucky ones. That gap — not fees, not latency — is what makes
copy trading hard.

And anyone clearing the identification bar has a ~31% return per trade sustained
over hundreds of trades, which is extraordinary enough that the data deserves
suspicion before the trader deserves capital.

---

## Why the luck problem is so severe

Concrete, from `fixtures.generate_no_skill_population`: a population of **3,000
wallets where every single one has exactly zero edge**. The best performer shows

```
edge +31.9 cents/share | ROI +74.6% | t-stat 5.02
```

That trader does not exist. It is the maximum of 3,000 random walks, and it
looks exactly like genius.

The bar rises with the size of the search, so scanning harder makes this worse,
not better:

| Wallets scanned | Leader trades | Edge needed to be identifiable |
|---|---|---|
| 1,000 | 200 | 13.8c |
| 10,000 | 200 | 15.7c |
| 100,000 | 200 | 17.3c |
| 10,000 | 1,000 | 7.0c |

The only thing that helps is a **long history per trader**, not a wide search.

---

## Three prediction-market traps

**Win rate is meaningless.** A wallet buying only 90c favourites wins 90% of the
time with zero edge. Every metric here is `(outcome − price)`, never win/loss.
Asserted in `test_win_rate_is_not_edge`.

**Favourite-longshot bias.** Longshots are systematically overpriced — documented
for decades. A wallet mechanically fading them looks skilled while harvesting a
known effect. That is a real edge, but **you should run it directly rather than
pay latency and slippage to copy someone else's version of it.**
`bias_attribution` separates the two.

**Persistence is the only real test.** Rank wallets on one period, measure them
on the next. Measured power: the test detects a 15c edge cleanly (t = 6.4) and
correctly reports nothing on a null population — but **cannot resolve a 6c edge**
at 1,200 wallets with 60–240 trades each. That is a true statement about
resolution, not a broken test.

---

## The regulatory structure (checked, August 2026)

- **Polymarket US (QCX LLC)** — CFTC-regulated DCM, live since December 2025,
  legal in 40+ states. Full KYC, USD settlement via FCMs. This is the only
  Polymarket product a US person may lawfully use.
- **Polymarket Global** — geo-blocked for US persons since the January 2022 CFTC
  settlement. VPN access violates the terms.

**The wallet histories live on Global. You would have to trade on US.** Signal
and fill are on different venues, with different order books, and Polymarket US
flow is very likely not on-chain at all. This is a structural feature of the 2026
position, not something to engineer around.

Cost of that split, from `execution.sensitivity`:

| Scenario | Slippage | Edge kept | Est. annual (8c leader) |
|---|---|---|---|
| Cross-venue (US-legal) | 2.93c | 63% | 56% |
| Same venue (non-US person) | 1.93c | 76% | 109% |
| Thin books (small markets) | 5.15c | 36% | 13% |

Roughly a third of the edge, plus a market-match rate near 55% — you can only
copy trades in markets that exist on both venues.

---

## What is genuinely good here

Worth stating, because it is the opposite of the options project:

- **Data is free, complete and keyless.** Polymarket's Data API returns any
  wallet's full trade history, positions and P&L; a Goldsky subgraph indexes
  every trade. The constraint that killed the M&A options thesis is absent.
- **Resolution is unambiguous.** Markets settle to 0 or 1. No mark-to-market
  judgement, no ambiguity about who was right.
- **The counterparty is identifiable.** Recreational bettors with a rooting
  interest, partisans, and people trading for entertainment. That is a real
  answer to "who is on the other side and why do they keep losing" — the
  question that the M&A flow thesis could never answer.
- **It is capacity-constrained**, which favours a small account. Estimated
  capacity is ~$2,500 per trade at baseline depth. Useless to a fund, fine for
  $25k. This is the category most likely to clear a 20% bar.

---

## What would have to be true

1. Wallets with **>15c/share edge over 200+ trades** actually exist on Polymarket.
2. Their edge **persists** out of sample (the `persistence_test`, run on real data).
3. Their edge is **not** just favourite-longshot bias (`bias_attribution`) — and
   if it is, run the bias directly instead.
4. Enough of their markets **exist on Polymarket US** to copy at all.

Point 1 is empirical and free to check. Points 2–4 only matter if 1 holds.

---

## Two bugs found while building this

Both are the same species as the synthetic-options-market bug, which makes three
occurrences in this project — the pattern is worth naming.

1. **The fixture did not contain the effect it advertised.** `skill_edge` only
   controlled *when* a skilled trader acted; the mispricing available to act on
   was a penny of noise. "Skilled" wallets had no edge, so the evaluator found
   nothing and looked broken while working correctly.
2. **The t-statistic mixed a size-weighted mean with an unweighted standard
   error.** Trade sizes are lognormal, so effective sample size is far below
   trade count. Every t-stat was inflated and noise walked through the gate.
   Fixed with Kish's effective sample size.

The recurring lesson: **a test harness that does not contain the effect it
claims to measure fails silently and looks like a negative result.**

---

## Honest calibration note

The luck gate is set at a nominal 95th percentile but its measured false-positive
rate is ~20% of populations, because per-share P&L is bimodal and wallet
t-statistics have fatter tails than normal. Clearing it is **necessary, not
sufficient** — the same status as a low PBO in `lab/validation.py`. It is
documented rather than tuned away.

---

## Building it: Global first

`notebooks/polymarket_wallet_skill.ipynb` (Colab) runs the crawl and answers the
identification question. Execution and the geo split are deferred deliberately —
they only matter if a wallet clears the bar.

**The one rule that must not be broken:** wallets are discovered by *market
participation*, never from the public leaderboard. Leaderboard seeding selects on
the outcome variable — ranking traders by past profit inside a set already
filtered for past profit — and produces excellent, meaningless numbers with no
visible failure anywhere. `client.discover_by_leaderboard()` exists and raises.

**Lookahead guard:** the persistence split is on **resolution time, not trade
time**. A trade placed in period A on a market resolving in period B has an
outcome nobody knew when ranking at the end of A. `persistence_test` defaults to
`split_on='resolved_at'` and refuses an unknown column with an explanation.

**Shape drift:** `validate_trade_fields` fails loudly if the API response no
longer carries an expected field, rather than letting it become a NaN column and
then a plausible result computed from nothing.

Read the answer as follows:

| Outcome | What it means |
|---|---|
| Nothing clears the luck bar | Complete answer, reached free. Most of what a leaderboard shows is the maximum of thousands of random walks. |
| Something clears it but does not persist | Same answer. Persistence is the decisive test. |
| Persists, but concentrated in extreme price bands | Favourite-longshot bias. Build the rule directly; copying is a worse wrapper around the same trade. |
| Persists and is not bias | Now the geo problem is worth solving — reformulate around Polymarket US. |
