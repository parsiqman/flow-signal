# The Research Platform

```bash
python src/lab/run_lab.py        # the whole machine, demonstrated end to end
python tests/test_lab.py         # 34 tests
```

**What this is:** infrastructure for scouting, testing, validating and iterating
on strategies. Not a strategy. The strategy in `STRATEGY.md` is the first thing
put through it — and the platform **blocks it**, which is the demonstration.

---

## Why this and not more strategies

An LLM trained on public text is a poor source of proprietary alpha. Everything
I can propose is, by construction, something many people have already read. So
the useful contribution is not the idea list — it is making sure that when an
idea *is* tested, the answer is trustworthy.

That reframing has teeth. The previous session's real contribution wasn't a
strategy, it was catching three bugs that each produced a confident, plausible,
wrong backtest. This package generalises that: it is a set of machines for not
fooling yourself, applied uniformly to every candidate.

---

## The central problem: the search is the enemy

Backtest 100 strategies, keep the best, and you will find a beautiful Sharpe
ratio **even if not one of them has any edge**. This is not a small correction.

The platform computes the luck baseline explicitly. From the demo run:

```
configurations run: 36
best in-sample Sharpe:   3.30

>> Expected best Sharpe from 36 ZERO-EDGE strategies: 1.35
   Any candidate below this line is indistinguishable from the search.
```

A raw Sharpe is not evidence. Sharpe *relative to what your own search would
produce from noise* is evidence. Everything in `validation.py` computes that.

---

## The five gates

Every candidate faces the same battery. All must pass — there is deliberately
**no weighted score**, because a weighted score lets a strong return number
outvote a failed overfitting test, and that is precisely the trade that
destroys accounts.

| Gate | Question | Threshold |
|---|---|---|
| `deflated_sharpe` | Does it beat what N random trials would produce by luck? | DSR ≥ 0.95 |
| `beats_luck_baseline` | Does raw Sharpe exceed the zero-edge best-of-N? | > E[max SR] |
| `probability_of_overfit` | Does the *selection procedure* generalise, not just the winner? | PBO ≤ 0.25 |
| `null_data_test` | Does it stop working when the claimed effect is removed? | p ≤ 0.05 |
| `cross_sample_stability` | Does it profit in most independent samples? | ≥ 70% |

The null test is the one that catches what cross-validation cannot: a strategy
that isn't exploiting the effect you think it is. If a volatility-premium
harvester earns just as much on data containing **no** volatility premium, then
whatever it's capturing, the reasoning that justified it is wrong — even if the
P&L is real.

---

## Result of the demo: the strategy is blocked

```
                  gate result  observed threshold
       deflated_sharpe   FAIL    0.8102   >= 0.95
   beats_luck_baseline   PASS    1.8370   > 1.351
probability_of_overfit   PASS    0.0000   <= 0.25
        null_data_test   PASS    0.0000   <= 0.05
cross_sample_stability   PASS    1.0000    >= 0.7

vrp-001: 4/5 gates passed -- BLOCKED on: deflated_sharpe
-> validated: False
-> live:      False (cannot skip from tested to live)
```

The VRP strategy passes four of five and fails the one that matters most. Its
out-of-sample Sharpe of 1.84 does clear the 1.35 luck baseline — but not by
enough, once the negative skew of a short-volatility return stream is accounted
for. Deflated Sharpe of 0.81 against a 0.95 bar.

**This is the platform working.** Without it, the honest-looking summary would
have been "+9.5% CAGR out of sample, Sharpe 1.84, profitable in 100% of fresh
markets" — all true, and not sufficient.

---

## The pieces

| Module | Job |
|---|---|
| `protocol.py` | One interface every candidate implements, so all are tested identically and a new idea costs a signal function, not a project |
| `registry.py` | Pre-registration + the automatic trial ledger |
| `validation.py` | DSR, PBO/CSCV, minimum backtest length, null tests, the gauntlet |
| `scout.py` | Idea catalogue organised by *why someone loses money to you* |
| `pipeline.py` | Promotion stages that cannot be skipped |

### Pre-registration
An idea is written down **before** testing: thesis, prediction, success
criteria, kill criteria, and the field that does the most work — **who is on the
other side and why they keep taking it**. Fields under 25 characters are
rejected outright. If you can't name the counterparty, you have found a pattern
in a dataset, and patterns in datasets are free, plentiful and worth nothing.

Amendments are allowed but recorded, so the log shows what was decided before
results and what after.

### The trial ledger
Every backtest is counted automatically and persisted, because the deflation
math is a function of how many things you tried. Humans reliably undercount —
the twenty runs spent "just getting it working" don't feel like trials. They
are. The ledger counts them regardless.

It also tracks **research debt**: how many years of data your search has now
made necessary. That number rises with every run, which puts the price of one
more parameter tweak in front of you at the moment you're tempted to make it.

### Scouting
Ideas ranked by the durability of the counterparty's motive, not by expected
return — expected return at the idea stage is imagination, a named counterparty
is a fact you can check. Four sources, best first: **risk premium** (someone
pays to shed a risk), **constraint** (someone must trade regardless of price),
**behavioural** (someone predictably errs, and decays once known),
**information** (someone knows first — not retail-accessible).

The catalogue is all public knowledge, stated plainly. Its job is making sure
the obvious ground is covered *systematically and honestly* before anyone hunts
for the exotic.

---

## Two calibration findings worth keeping

**PBO on pure noise is not 0.5, and is not stable.** Measured over 16 datasets:
a genuine edge gives PBO 0.000 every time; identical-noise configurations
scatter from 0.10 to 0.86 (mean 0.54). Train and test halves are exact
complements of one sample, so a winner selected on split luck must give it back.

**Therefore a low PBO is weaker evidence than it looks.** Noise produced 0.10 in
one of 16 draws — roughly a 5–10% chance a single reading under the 0.25 gate is
luck. A *high* reading is the trustworthy signal. This is exactly why the
gauntlet requires PBO alongside DSR and a null test rather than on its own.

Both were found by testing the validator against Monte Carlo ground truth
rather than against itself. `validation.py` is the one module where an error
wouldn't produce a wrong backtest — it would produce a wrong *belief about every
backtest*, silently and permanently.

---

## How to add a candidate

1. Add an `Idea` to `scout.CATALOGUE` with a named counterparty.
2. `scout.to_hypothesis(...)` — write prediction, success and kill criteria by
   hand. These are the commitments; auto-generating them defeats the point.
3. `registry.register(h)`, then promote to `SCREENED`.
4. Implement `Strategy.generate`. Declare `required_columns` — the harness
   refuses data that lacks them rather than producing NaN-shaped results.
5. Search, recording every run to the ledger.
6. `validation.run_gauntlet(...)` on out-of-sample returns.
7. `pipeline.promote(...)`. It will refuse if the gauntlet failed or if your
   search was larger than your sample can support.

---

## Honest limits

- The gauntlet reduces false positives; it cannot manufacture an edge. A
  disciplined process applied to public ideas most likely yields "no edge here",
  and that is a real, useful, money-saving answer.
- Thresholds (0.95, 0.25, 70%, 63 paper days) are conventional, not derived.
  They are in one place and should be argued with.
- Everything so far runs on synthetic data. The platform's value is only
  realised when pointed at real chains.
- `minimum_backtest_length` is an approximation and tends to be optimistic;
  treat it as a floor, not a target.
