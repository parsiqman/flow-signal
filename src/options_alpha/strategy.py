"""
The strategy: defined-risk variance risk premium harvest.

Thesis, in one line: implied volatility sits above subsequently realised
volatility on average because option buyers are paying for insurance, and a
seller can collect that premium -- provided they never let one position, or one
regime, end the account.

The design follows from three constraints the family screen produced:

  1. Do NOT hedge the premium away. Vega-neutralising a VRP book removes the
     very thing that pays; what is left is a weak cross-sectional forecast.
     So the book is deliberately net short vol.
  2. Cap every position structurally. Bought wings cost roughly a quarter of
     the edge and remove the failure mode that ends the project. Position
     sizing alone cannot do this, because the tail arrives everywhere at once.
  3. Spread across many underlyings and stagger expiries, for whatever breadth
     is available -- but do not pretend that diversification is protection.
     In a vol shock every short-vol position loses together. The wings are the
     protection; the diversification only smooths the ordinary days.

Everything below is deliberately mechanical. This is an implementation problem,
not a discovery problem: the premium is documented back to the early 2000s, so
elaborate signal engineering would be fitting noise on top of a known effect.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .synthetic import (MarketConfig, bs_price_scalar, smile_iv, spread_frac,
                        strike_for_delta, TRADING_DAYS)  # noqa: F401


@dataclass
class StrategyConfig:
    """Every knob, in one place. Defaults are the ones argued for in STRATEGY.md."""

    # --- position structure ----------------------------------------------
    dte: int = 35                    # 30-45 is the premium/gamma sweet spot
    short_delta: float = 0.16        # ~1 sd short strikes
    wing_width_frac: float = 0.60    # wing distance, in units of the 1sd move
    wings_scale_with_vol: bool = True
    profit_target: float = 0.50      # close at this fraction of credit; 1.0 = hold to expiry
    both_sides: bool = True          # iron condor vs put-spread only
    # Fraction of the half-spread actually paid. 1.0 = cross to the touch every
    # time; 0.0 = always filled at mid. Working limit orders on a non-urgent
    # systematic entry realistically lands around 0.3-0.5. This is the most
    # consequential single assumption in the whole model -- see run.py step 4.
    fill_quality: float = 1.0

    # --- selection --------------------------------------------------------
    n_positions: int = 20            # concurrent positions per expiry cycle
    min_richness: float = 0.0        # only sell vol that is actually rich
    exclude_earnings: bool = True    # skip prints inside the option's life

    # --- sizing -----------------------------------------------------------
    risk_per_position: float = 0.010   # 1% of equity at risk per position
    max_portfolio_risk: float = 0.25   # 25% of equity at risk in total

    # --- regime defence ---------------------------------------------------
    # OFF by default, on the evidence: the ablation shows it costs ~1.7pp of
    # CAGR and improves no risk measure. The bought wings already cap the loss,
    # so standing down mid-shock just forgoes the richest premium of the cycle.
    # Kept in the code because the synthetic tail is gentler than a real one --
    # retest on real 2008/2020 data before discarding it for good.
    use_regime_filter: bool = False
    regime_lookback_fast: int = 10
    regime_lookback_slow: int = 60
    regime_stop_ratio: float = 1.35    # fast/slow vol above this -> stand down
    regime_size_floor: float = 0.25    # size multiplier when standing down

    # --- rebalance --------------------------------------------------------
    entry_every_days: int = 7          # weekly, so expiries ladder naturally


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

def forecast_realised_vol(spot_history: np.ndarray) -> np.ndarray:
    """
    HAR-style realised volatility forecast (Corsi 2009): blend short, medium
    and long realised vol.

    Deliberately simple. The forecast only has to be roughly right -- the edge
    comes from the premium, not from out-forecasting the market. A complicated
    model here would be fitting noise on top of a known effect, and would be
    the first thing to break out of sample.
    """
    rets = np.diff(np.log(spot_history), axis=0)
    if len(rets) < 63:
        return np.std(rets, axis=0) * np.sqrt(TRADING_DAYS)
    rv5 = np.std(rets[-5:], axis=0) * np.sqrt(TRADING_DAYS)
    rv21 = np.std(rets[-21:], axis=0) * np.sqrt(TRADING_DAYS)
    rv63 = np.std(rets[-63:], axis=0) * np.sqrt(TRADING_DAYS)
    return 0.35 * rv5 + 0.40 * rv21 + 0.25 * rv63


def richness(atm_iv: np.ndarray, forecast_rv: np.ndarray) -> np.ndarray:
    """
    How much premium is on offer, in relative terms.

    Relative rather than absolute (IV - RV) so that a 20%-vol name and a
    90%-vol name are comparable. An absolute spread would load the entire book
    into high-vol names, which is a concentrated bet on those names not moving
    rather than a diversified harvest.
    """
    return atm_iv / np.maximum(forecast_rv, 1e-6) - 1.0


# ---------------------------------------------------------------------------
# Position construction
# ---------------------------------------------------------------------------

@dataclass
class Position:
    name: str
    entry_day: int
    expiry_day: int
    spot_at_entry: float
    short_put: float
    long_put: float
    short_call: float
    long_call: float
    credit: float          # per unit, after entry costs
    max_loss: float        # per unit
    units: float
    richness: float

    def capital_at_risk(self) -> float:
        return self.max_loss * self.units


def build_condor(spot: float, atm_iv: float, cfg: StrategyConfig,
                 mkt: MarketConfig) -> tuple[dict, float, float] | None:
    """
    Construct one defined-risk short-vol position and price it net of spreads.

    Returns (strikes, net_credit_per_unit, max_loss_per_unit), or None when the
    structure is not worth trading -- which happens when the spread eats too
    much of the credit. Refusing to trade is a feature: the family screen showed
    friction is the binding constraint, so the discipline has to live in code.
    """
    # Trading-day year fraction, to match the sqrt(252) vol annualisation.
    t = cfg.dte / TRADING_DAYS
    sp = strike_for_delta(spot, t, atm_iv, -cfg.short_delta, is_call=False)
    sc = strike_for_delta(spot, t, atm_iv, cfg.short_delta, is_call=True)
    # Wing distance scales with the actual expected move, not with spot. A
    # fixed percentage of spot means something completely different at 20% vol
    # than at 60% vol, and would silently load up on risk in exactly the
    # regimes where risk should be coming off.
    sigma_move = atm_iv * np.sqrt(t) * spot
    width = cfg.wing_width_frac * (sigma_move if cfg.wings_scale_with_vol else spot)
    lp, lc = sp - width, sc + width
    # At very high implied vol a wing priced in sigma units can land at or below
    # zero, which is not a strike that exists. Refuse the trade rather than
    # clamping: a put spread whose long leg cannot be bought is a naked short
    # put wearing a disguise, and defined risk is the point of the whole design.
    if lp <= 0.01 * spot or sp <= lp:
        return None

    spec = [(sp, False, True), (lp, False, False)]
    if cfg.both_sides:
        spec += [(sc, True, True), (lc, True, False)]

    legs = []
    for strike, is_call, is_short in spec:
        iv_k = float(smile_iv(atm_iv, strike / spot, mkt))
        mid = bs_price_scalar(spot, strike, t, iv_k, is_call, mkt.risk_free)
        # Wings are far OTM and cheap, so their proportional spread is worst.
        is_wing = not is_short
        half_spread = mid * spread_frac(atm_iv, is_wing) / 2.0 * cfg.fill_quality
        # Selling: receive bid (mid - half). Buying: pay ask (mid + half).
        fill = (mid - half_spread) if is_short else (mid + half_spread)
        legs.append((strike, is_call, is_short, max(fill, 0.0), mid))

    credit = sum(f if short else -f for _, _, short, f, _ in legs)
    gross = sum(m if short else -m for _, _, short, _, m in legs)
    if credit <= 0 or gross <= 0:
        return None
    # If crossing the spread costs more than a third of the theoretical credit,
    # the trade is a donation to the market maker.
    if credit < 0.67 * gross:
        return None

    max_loss = width - credit
    if max_loss <= 0:
        return None
    strikes = {"short_put": sp, "long_put": lp}
    if cfg.both_sides:
        strikes.update({"short_call": sc, "long_call": lc})
    return strikes, credit, max_loss


def condor_payoff(spot_at_expiry: float, k: dict) -> float:
    """What the seller owes at expiration. Always >= 0."""
    s = spot_at_expiry
    owed = max(0.0, k["short_put"] - s) - max(0.0, k["long_put"] - s)
    if "short_call" in k:
        owed += max(0.0, s - k["short_call"]) - max(0.0, s - k["long_call"])
    return owed


# ---------------------------------------------------------------------------
# Regime defence
# ---------------------------------------------------------------------------

def regime_multiplier(market_spot_history: np.ndarray,
                      cfg: StrategyConfig) -> float:
    """
    Stand down when volatility is accelerating.

    Compares fast to slow realised vol on an equal-weighted market proxy. This
    is the cheap stand-in for a VIX term-structure signal: when short-dated vol
    runs above long-dated, the premium is no longer compensation for risk, it
    IS the risk, and sellers get run over.

    Deliberately reduces rather than eliminates exposure. A binary on/off would
    be fitted to whichever shocks happen to be in the sample, and would flip
    the strategy into a market-timing bet it has no edge in.
    """
    if not cfg.use_regime_filter or len(market_spot_history) < cfg.regime_lookback_slow + 2:
        return 1.0
    rets = np.diff(np.log(market_spot_history))
    fast = np.std(rets[-cfg.regime_lookback_fast:])
    slow = np.std(rets[-cfg.regime_lookback_slow:])
    if slow <= 0:
        return 1.0
    ratio = fast / slow
    if ratio <= 1.0:
        return 1.0
    if ratio >= cfg.regime_stop_ratio:
        return cfg.regime_size_floor
    # linear taper between the two
    span = cfg.regime_stop_ratio - 1.0
    return float(1.0 - (1.0 - cfg.regime_size_floor) * (ratio - 1.0) / span)


def select_candidates(snapshot: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Rank the universe and take the richest names that pass the filters.

    The earnings exclusion is the important one, and it is the same insight the
    old M&A work landed on from the other direction: implied vol before a
    scheduled event is not a premium being overpaid, it is compensation for a
    jump that genuinely happens. Selling it is not harvesting VRP, it is
    selling a fair-priced lottery ticket and calling it income.
    """
    df = snapshot.copy()
    if cfg.exclude_earnings:
        df = df[df["days_to_earnings"] > cfg.dte]
    df = df[df["richness"] > cfg.min_richness]
    return df.nlargest(cfg.n_positions, "richness")
