# Real Option Chains

```bash
python tests/test_data.py                    # 26 tests
```
Then run **`notebooks/real_option_chains.ipynb`** in Colab — it does the pull.

---

## Why Colab and not here

This sandbox's egress policy blocks every options vendor. Measured, not assumed:

```
hist.databento.com  403      api.orats.io       403
api.databento.com   403      datashop.cboe.com  403
thetadata.us        403      dolthub.com        403
api.polygon.io      403      data.alpaca.markets 403
```

All are `403 to CONNECT` — organisation policy denials, not network faults. Only
GitHub is reachable. Colab is unrestricted, so the ingestion layer is built and
tested here against fixtures, and the pull runs there.

---

## What to buy (staged, so you spend nothing until the thesis survives)

| Source | History | Cost | Verdict |
|---|---|---|---|
| **DoltHub `post-no-preference/options`** | ~2019 → now | **free** | EOD chains, bids/asks/greeks, ~2100 symbols. Covers Mar-2020, 2022 and Aug-2024 — three vol regimes. **Start here.** |
| [ORATS near-EOD](https://orats.com/near-eod-data) | 2007 → now | paid | The only cheap route to 2008. Buy *only* if the free data clears the kill criterion. |
| CBOE DataShop | deep | paid per dataset | Authoritative, pricier. |
| Databento OPRA | ~2021 → now | pay-per-GB, $125 free credit | Tick-level. Does **not** reach 2008 — verify coverage before buying it for tail work. |

The logic: the free dataset is enough to answer *"is there a premium at all?"*.
Only pay for depth once the answer is yes.

---

## The kill criterion comes first

`STRATEGY.md` pre-registered it:

> Abandon the thesis if frictionless straddle capture on real chains is below 3%.

`chain.measure_vrp()` decides it — sell an ATM straddle at the **bid** (what a
seller actually receives), hold to expiry, compare premium against payout. It is
tested in both directions: on a fixture with no premium it must report negative
capture, and on one with a 25% premium it must find it. Testing only the positive
case would let a broken measurement pass forever.

Run this **before** any strategy work. If it fails, the direction is dead and no
amount of implementation quality rescues it.

---

## The quality gate

Real chains are filthy, and every defect below fails **silently** — the backtest
runs clean and reports a confident wrong number. Ranked by damage:

| # | Defect | Why it is expensive |
|---|---|---|
| 1 | **Zero bids treated as sellable** | No buyer at any price, but `(0+ask)/2` looks tradeable. Worst on exactly the far-OTM wings a defined-risk strategy trades — it manufactures credit that could never have been collected. |
| 2 | **Survivorship** | A universe of "names optionable today" has deleted every company that went bankrupt or was acquired. For a *short-vol* strategy this bias is favourable and large: the missing names are the ones that blew up. |
| 3 | **Adjusted options** | Post-split contracts deliver non-standard amounts. No pricing model applies; they must be dropped, not modelled. |
| 4 | **Stale quotes** | Zero volume *and* zero open interest means the quote is indicative at best. |
| 5 | **Arbitrage violations** | Sub-intrinsic prices and parity breaks mean the underlying price came from a different instant than the chain — a free, invisible edge. Free edges in backtests are always data errors. |

`quality.assert_usable()` **raises** rather than warns. A warning at the top of a
long run is read exactly once.

Survivorship detection is the one worth calling out: it blocks when *no*
underlying ever leaves the universe over a long sample. Real universes lose names
constantly; a dataset where none disappear has been filtered to survivors.

---

## Chain mathematics

**Constant-maturity implied vol.** Listed expiries jump around — today the
nearest 30-day option is 28 days out, next week 21, then 35. Raw nearest-expiry
IV partly measures the expiry calendar rather than the market. So IV is
interpolated to a fixed tenor in **total variance** (`w = iv² × T`), which is what
is additive in time. Interpolating linearly in vol is a common shortcut that
biases the term structure, and a richness signal built on it partly measures that
bias.

**Real strikes.** The synthetic backtest could place a strike wherever the maths
wanted. Real chains offer $1, $2.50 or $5 increments, so a 10-delta target lands
on whatever is listed nearby — and on a cheap underlying that may be 14-delta
instead. `select_strikes` returns what actually exists and reports the gap rather
than hiding it. A backtest using the ideal strike is trading a contract that never
existed.

---

## Layout

| File | Job |
|---|---|
| `schema.py` | Canonical contract schema. Refuses to invent bid/ask/expiry; `mid()` returns NaN where a contract cannot be traded. |
| `quality.py` | The gate: every check above, plus `clean()`. |
| `chain.py` | ATM IV, constant-maturity interpolation, strike snapping, panel build, `measure_vrp`. |
| `loaders.py` | Vendor adapters (dolthub/orats/polygon) + deliberately dirty fixtures. |

Adding a vendor costs a column-map dict. All judgement lives in `quality.py`.

The panel output is **schema-identical** to `options_alpha.synthetic.generate_market`,
so real and synthetic data are interchangeable and every existing test applies to both.

---

## On testing the gate

Every defect is reproducible on demand via `loaders.synthetic_chain(dirty=True)`,
because a gate validated only against clean data is an assertion, not a test. Both
properties are checked with equal weight:

- it **catches** the defects (dirty fixture blocks), and
- it does **not** fire on good data (clean fixture passes).

The second matters as much as the first. A gate that blocks everything is useless
in a different way, and would be quietly disabled within a week.
