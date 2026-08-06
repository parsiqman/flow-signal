"""
Can the trade actually be copied? Modelling the gap between seeing and filling.

Skill identification decides whether there is anything worth copying. This
module decides whether what is left after copying it is still worth having.
Three costs, and the second is the one that quietly kills copy trading:

1. **Latency.** The trade is visible only after it settles on chain. By then the
   price has already moved -- their fill moved it. You are buying what they just
   bought, from the book they just emptied.

2. **Adverse selection in what you CAN fill.** This is the subtle one. The
   trades still available at the old price are, disproportionately, the ones
   nobody else wanted. If the informed trade was genuinely informative, the
   market moved and your copy is expensive. If the price did not move, either
   the trade was not informative or nobody believed it. Either way, the subset
   you can fill cheaply is adversely selected against the subset worth copying.

3. **Cross-venue basis.** US persons may legally trade Polymarket US (QCX LLC,
   CFTC-regulated) but NOT Polymarket Global, which is where the on-chain wallet
   histories live. So the signal comes from one venue and the fill happens on
   another, with a different order book, possibly a different market universe,
   and no guarantee the two prices agree. This is a structural feature of the
   2026 regulatory position, not a detail to engineer around.

Prediction-market prices are bounded in [0, 1], so a "1 cent" move is 1% of a
50c contract but 10% of a 10c longshot. Everything below is in cents per share,
and relative cost is computed against the price actually paid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ExecutionModel:
    """Assumptions about copying one trade. Argue with these directly."""

    # --- latency ---------------------------------------------------------
    detection_lag_s: float = 8.0        # block confirmation + poll interval
    decision_lag_s: float = 2.0
    submit_lag_s: float = 3.0

    # --- price impact ----------------------------------------------------
    # How much of the leader's own impact has NOT decayed by the time you fill.
    impact_persistence: float = 0.65
    # Their impact, in cents per share, for a trade of typical size against a
    # typical Polymarket book.
    leader_impact_cents: float = 1.2
    # Your own impact when you follow.
    own_impact_cents: float = 0.4

    # --- spread ----------------------------------------------------------
    half_spread_cents: float = 0.75     # thin books; wider on small markets

    # --- cross-venue -----------------------------------------------------
    cross_venue: bool = True            # signal on Global, fill on Polymarket US
    basis_cents: float = 1.0            # typical price divergence between venues
    market_match_rate: float = 0.55     # fraction of Global markets that exist on US

    # --- capacity --------------------------------------------------------
    typical_book_depth_usd: float = 25_000.0
    max_participation: float = 0.10     # of resting depth, before impact explodes

    def total_slippage_cents(self) -> float:
        """Cents per share given up between their fill and yours."""
        c = (self.leader_impact_cents * self.impact_persistence
             + self.own_impact_cents + self.half_spread_cents)
        if self.cross_venue:
            c += self.basis_cents
        return c

    def net_edge_cents(self, leader_edge_cents: float) -> float:
        """What is left of their edge after you have paid to follow it."""
        return leader_edge_cents - self.total_slippage_cents()

    def breakeven_leader_edge_cents(self) -> float:
        """How good the leader must be before copying them earns anything."""
        return self.total_slippage_cents()

    def capacity_usd(self) -> float:
        return self.typical_book_depth_usd * self.max_participation


def copy_economics(leader_edge_cents: float, avg_price_cents: float = 50.0,
                   model: ExecutionModel | None = None,
                   trades_per_year: int = 200,
                   capital_usd: float = 25_000.0) -> dict:
    """
    Full economics of copying one trader, in the units that matter.

    `leader_edge_cents` is their measured edge per share in cents -- exactly
    what `wallets.score_wallets` produces, multiplied by 100.

    Note what a small absolute number means here. An edge of 3 cents per share
    on a 50c contract is a 6% return per trade, which is enormous. But the
    slippage of copying is also measured in cents, so the two are the same order
    of magnitude, and that is the entire problem: copy trading takes a
    cents-scale edge and pays a cents-scale cost to acquire it.
    """
    m = model or ExecutionModel()
    slip = m.total_slippage_cents()
    net = leader_edge_cents - slip
    stake_per_trade = min(capital_usd * 0.05, m.capacity_usd())

    # Return per trade on capital staked, then scaled by how many copies land.
    ret_per_trade = net / avg_price_cents
    fill_rate = m.market_match_rate if m.cross_venue else 0.9
    effective_trades = trades_per_year * fill_rate
    annual_return = (ret_per_trade * stake_per_trade * effective_trades) / capital_usd

    return {
        "leader_edge_cents": round(leader_edge_cents, 2),
        "slippage_cents": round(slip, 2),
        "net_edge_cents": round(net, 2),
        "edge_retained_pct": (round(100 * net / leader_edge_cents, 1)
                              if leader_edge_cents > 0 else np.nan),
        "breakeven_leader_edge_cents": round(m.breakeven_leader_edge_cents(), 2),
        "return_per_trade_pct": round(100 * ret_per_trade, 2),
        "copies_landed_per_year": round(effective_trades),
        "stake_per_trade_usd": round(stake_per_trade),
        "est_annual_return_pct": round(100 * annual_return, 1),
        "capacity_usd_per_trade": round(m.capacity_usd()),
        "verdict": ("copying retains a usable share of the edge" if net > 0
                    else "SLIPPAGE EXCEEDS THE LEADER'S ENTIRE EDGE"),
    }


def sensitivity(leader_edge_cents: float = 4.0,
                capital_usd: float = 25_000.0) -> pd.DataFrame:
    """
    How the answer moves with the assumptions least worth trusting.

    The cross-venue row is the one to read first. It is not a modelling choice
    -- it is imposed by the 2026 regulatory position, and it roughly doubles
    the cost of every copy.
    """
    from dataclasses import replace
    base = ExecutionModel()
    scenarios = {
        "baseline (cross-venue, US-legal)": base,
        "same venue (non-US person)": replace(base, cross_venue=False),
        "faster detection (2s total)": replace(base, detection_lag_s=1.0,
                                               decision_lag_s=0.5,
                                               submit_lag_s=0.5,
                                               impact_persistence=0.85),
        "impact decays fully before fill": replace(base, impact_persistence=0.15),
        "thick books (5x depth, half spread)": replace(
            base, half_spread_cents=0.35, leader_impact_cents=0.5,
            typical_book_depth_usd=125_000),
        "thin books (small markets)": replace(
            base, half_spread_cents=1.8, leader_impact_cents=3.0,
            typical_book_depth_usd=5_000),
    }
    rows = []
    for label, m in scenarios.items():
        e = copy_economics(leader_edge_cents, model=m, capital_usd=capital_usd)
        rows.append({
            "scenario": label,
            "slippage_c": e["slippage_cents"],
            "net_edge_c": e["net_edge_cents"],
            "edge_kept_%": e["edge_retained_pct"],
            "est_annual_%": e["est_annual_return_pct"],
            "capacity_$": e["capacity_usd_per_trade"],
        })
    return pd.DataFrame(rows)


def required_leader_edge(target_annual_pct: float = 20.0,
                         model: ExecutionModel | None = None,
                         trades_per_year: int = 200,
                         avg_price_cents: float = 50.0,
                         capital_usd: float = 25_000.0) -> dict:
    """
    Invert the economics: how good must a leader be to hit a return target?

    Then compare that against the luck baseline from `wallets.expected_best_edge`.
    If the edge required to hit the target is smaller than the apparent edge a
    zero-skill wallet shows by chance, then you cannot tell the traders who
    would get you there from the ones who merely look like they would -- and
    the strategy is unimplementable regardless of whether such traders exist.
    """
    m = model or ExecutionModel()
    stake = min(capital_usd * 0.05, m.capacity_usd())
    fill_rate = m.market_match_rate if m.cross_venue else 0.9
    effective = trades_per_year * fill_rate
    if effective <= 0 or stake <= 0:
        return {"verdict": "no capacity to trade"}

    # target = net_cents/price * stake * effective / capital
    net_needed = (target_annual_pct / 100.0) * capital_usd * avg_price_cents \
        / (stake * effective)
    return {
        "target_annual_pct": target_annual_pct,
        "net_edge_needed_cents": round(net_needed, 2),
        "slippage_cents": round(m.total_slippage_cents(), 2),
        "leader_edge_needed_cents": round(net_needed + m.total_slippage_cents(), 2),
        "stake_per_trade_usd": round(stake),
        "copies_per_year": round(effective),
    }
