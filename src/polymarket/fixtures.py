"""
Synthetic wallet populations with KNOWN ground truth.

The evaluator's whole job is telling skill from luck. That claim cannot be
checked against real Polymarket data, because nobody knows which real wallets
are skilled -- that is the question. So it is checked here instead, on a
population where the answer is recorded in advance:

  - `skilled_frac` of wallets genuinely beat the price by `skill_edge`
  - the rest have exactly zero edge and differ only by luck
  - every wallet's true status is returned alongside its trades

That lets both failure modes be measured directly: how many real edges the
ranking finds (recall), and how many lucky no-edge wallets it promotes
(false discovery). A selection rule tested only against real data can be
evaluated on neither.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_wallets(n_wallets: int = 2000,
                     skilled_frac: float = 0.02,
                     skill_edge: float = 0.04,
                     trades_per_wallet: tuple[int, int] = (20, 200),
                     n_markets: int = 800,
                     longshot_bias: float = 0.0,
                     seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a population of traders and their fills.

    Returns (trades, truth). `truth` carries the `is_skilled` flag and each
    wallet's true edge, so the evaluator can be scored rather than admired.

    `skill_edge` is in probability units: 0.04 means a skilled wallet buys when
    the true probability exceeds the price by 4 cents on average. That is a
    large edge for a real market and is deliberately generous -- if the
    evaluator cannot find an edge this big, it will not find a realistic one.

    `longshot_bias` mispricies cheap outcomes upward, reproducing the documented
    favourite-longshot effect. With it switched on, wallets that merely fade
    longshots earn money with no skill of their own, which is exactly the
    confound `wallets.bias_attribution` exists to separate.
    """
    rng = np.random.default_rng(seed)

    # Market true probabilities, and the prices actually quoted.
    true_p = rng.uniform(0.05, 0.95, n_markets)
    quoted = true_p.copy()
    if longshot_bias:
        # Cheap outcomes quoted above their true probability.
        quoted = np.where(true_p < 0.30, true_p + longshot_bias * (0.30 - true_p),
                          quoted)
        quoted = np.where(true_p > 0.70, true_p - longshot_bias * (true_p - 0.70),
                          quoted)
    # Real, exploitable mispricing must actually EXIST for skill to be possible.
    # Scaled so the target `skill_edge` is attainable: a perfectly informed
    # trader earns E|mispricing| = 0.8 * sigma, so sigma is set from the target.
    mis_sd = max(0.06, skill_edge / 0.6)
    mispricing = rng.normal(0, mis_sd, n_markets)
    quoted = np.clip(quoted + mispricing, 0.02, 0.98)
    mispricing = quoted - true_p          # after clipping, this is the truth
    # Probability a skilled trader takes the profitable side, solved from the
    # target edge: edge = E|mispricing| * (2p - 1).
    e_abs = float(np.mean(np.abs(mispricing)))
    p_correct = float(np.clip(0.5 + skill_edge / (2 * max(e_abs, 1e-9)), 0.5, 0.999))
    outcomes = (rng.random(n_markets) < true_p).astype(int)

    n_skilled = int(n_wallets * skilled_frac)
    skilled = set(rng.choice(n_wallets, size=n_skilled, replace=False).tolist())

    rows, truth = [], []
    lo, hi = trades_per_wallet
    for w in range(n_wallets):
        wallet = f"0x{w:040x}"
        n = int(rng.integers(lo, hi + 1))
        is_skilled = w in skilled
        picks = rng.choice(n_markets, size=n, replace=True)

        for i, mkt in enumerate(picks):
            p, q, o = quoted[mkt], true_p[mkt], outcomes[mkt]
            if is_skilled:
                # Takes the profitable side with probability p_correct. The
                # price is too HIGH when quoted > true, so the profitable side
                # is SELL; too low means BUY.
                correct_side = "SELL" if mispricing[mkt] > 0 else "BUY"
                wrong_side = "BUY" if correct_side == "SELL" else "SELL"
                side = correct_side if rng.random() < p_correct else wrong_side
            else:
                side = "BUY" if rng.random() < 0.5 else "SELL"

            rows.append({
                "wallet": wallet, "market_id": int(mkt),
                "timestamp": float(rng.uniform(0, 1000)),
                "price": float(p), "size": float(rng.lognormal(4.5, 0.9)),
                "side": side, "outcome": float(o),
                "resolved_at": 1000.0,
            })

        truth.append({"wallet": wallet, "is_skilled": is_skilled,
                      "n_trades": n,
                      "true_edge": skill_edge if is_skilled else 0.0})

    trades = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return trades, pd.DataFrame(truth)


def generate_no_skill_population(n_wallets: int = 5000, seed: int = 1,
                                 **kwargs) -> pd.DataFrame:
    """
    A population where NOBODY has an edge.

    The most important fixture in the file. Any selection rule run over this
    will still produce a top decile with impressive-looking returns, because
    that is what ranking noise does. If a rule reports skill here, it reports
    skill everywhere, and its output on real data means nothing.
    """
    trades, _ = generate_wallets(n_wallets=n_wallets, skilled_frac=0.0,
                                 seed=seed, **kwargs)
    return trades


def generate_persistent_and_transient(n_wallets: int = 1500,
                                      seed: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Skilled wallets whose edge persists, in a population where luck does not.

    Used to check that `persistence_test` can detect real persistence when it
    exists -- the necessary companion to checking that it reports nothing on a
    no-skill population. A test that only ever says "no" is not a test.
    """
    trades, truth = generate_wallets(n_wallets=n_wallets, skilled_frac=0.05,
                                     skill_edge=0.06, seed=seed,
                                     trades_per_wallet=(60, 240))
    return trades, truth
