"""
Synthetic options market, built to be hostile to the strategy that trades it.

A short-volatility backtest is only informative if the data contains the events
that kill short volatility. Generating a well-behaved lognormal market and
discovering that selling options makes money proves nothing -- it is true by
construction. So this generator deliberately includes:

  - stochastic volatility with a strong market-wide common factor, so the
    positions co-move exactly when you would least like them to
  - rare volatility regime shocks: market-wide, persistent, with the price gap
    that accompanies them. These are the Feb-2018 / Mar-2020 / Aug-2024 events
  - fat-tailed daily returns even outside shocks
  - earnings dates that inflate implied vol *legitimately*, so that a naive
    richness screen walks straight into them
  - a volatility smile, so that selling puts collects skew premium and pays for
    it in a crash
  - bid-ask spreads that widen with illiquidity and with volatility

The variance risk premium is real in this market (implied sits above expected
realised on average), so a well-built strategy should profit. The question the
backtest answers is not "is there an edge" but "does the implementation survive
the regimes where the edge turns against you".
"""

from __future__ import annotations
from dataclasses import dataclass
import math as _math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class MarketConfig:
    n_names: int = 60
    n_days: int = 252 * 12          # 12 years, long enough to contain shocks
    seed: int = 17

    # volatility process (log-OU, mean reverting)
    vol_mean: float = 0.28          # long-run annualised vol
    vol_kappa: float = 3.0          # mean reversion speed (per year)
    vol_of_vol: float = 0.55
    common_vol_share: float = 0.55  # fraction of vol shocks that are market-wide

    # the tail
    shock_per_year: float = 1 / 6   # a real vol regime roughly every 6 years
    shock_vol_mult: float = 2.8     # vol multiplier at the peak
    shock_decay_days: int = 25
    shock_gap_pct: float = -0.09    # the price gap that arrives with it

    # variance risk premium: implied vol is this much above expected realised
    vrp_mean: float = 0.16          # +16% relative, i.e. 28% RV -> ~32.5% IV
    vrp_noise: float = 0.08
    smile_slope: float = 0.35       # OTM puts richer than ATM (negative skew)

    # earnings
    earnings_per_year: int = 4
    earnings_jump_vol: float = 0.055  # the actual move it is compensating for

    risk_free: float = 0.04


# ---------------------------------------------------------------------------
# Option pricing
# ---------------------------------------------------------------------------

_SQRT2 = _math.sqrt(2.0)


def _norm_cdf(x):
    """Standard normal CDF via erf, avoiding a scipy dependency."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return 0.5 * (1.0 + _math.erf(float(arr) / _SQRT2))
    return 0.5 * (1.0 + _ERF(arr / _SQRT2))


# np.vectorize on every scalar call dominated the backtest runtime, so the
# array path gets a closed-form erf (Abramowitz & Stegun 7.1.26, ~1e-7) and
# the scalar path goes straight to math.erf.
def _ERF(z):
    z = np.asarray(z, dtype=float)
    sign = np.sign(z)
    a = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return sign * y


def bs_price_scalar(spot: float, strike: float, t_years: float, iv: float,
                    is_call: bool, r: float = 0.04) -> float:
    """Pure-Python Black-Scholes for the single-option hot path."""
    t = max(t_years, 1e-9)
    iv = max(iv, 1e-6)
    sqrt_t = _math.sqrt(t)
    d1 = (_math.log(spot / strike) + (r + 0.5 * iv * iv) * t) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    nd1 = 0.5 * (1.0 + _math.erf(d1 / _SQRT2))
    nd2 = 0.5 * (1.0 + _math.erf(d2 / _SQRT2))
    disc = _math.exp(-r * t)
    if is_call:
        return spot * nd1 - strike * disc * nd2
    return strike * disc * (1.0 - nd2) - spot * (1.0 - nd1)


def bs_price(spot, strike, t_years, iv, is_call: bool, r: float = 0.04):
    """Black-Scholes. Vectorised over any broadcastable inputs."""
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    t = np.maximum(np.asarray(t_years, dtype=float), 1e-9)
    iv = np.maximum(np.asarray(iv, dtype=float), 1e-6)

    sqrt_t = np.sqrt(t)
    d1 = (np.log(spot / strike) + (r + 0.5 * iv ** 2) * t) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    disc = np.exp(-r * t)
    if is_call:
        return spot * _norm_cdf(d1) - strike * disc * _norm_cdf(d2)
    return strike * disc * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_delta(spot, strike, t_years, iv, is_call: bool, r: float = 0.04):
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    t = np.maximum(np.asarray(t_years, dtype=float), 1e-9)
    iv = np.maximum(np.asarray(iv, dtype=float), 1e-6)
    d1 = (np.log(spot / strike) + (r + 0.5 * iv ** 2) * t) / (iv * np.sqrt(t))
    return _norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0


def strike_for_delta(spot, t_years, iv, target_delta: float, is_call: bool,
                     r: float = 0.04) -> float:
    """
    Invert Black-Scholes for the strike at a given delta.

    Used instead of a fixed percentage offset because a fixed offset means
    something completely different at 20% vol than at 80% vol -- the strategy
    would silently take far more risk in exactly the regimes it should be
    taking less.
    """
    z = _inv_norm(target_delta if is_call else 1.0 - abs(target_delta))
    return float(spot * np.exp(-z * iv * np.sqrt(t_years)
                               + (r + 0.5 * iv ** 2) * t_years))


def _inv_norm(p: float) -> float:
    """Acklam's inverse normal CDF approximation. Good to ~1e-9."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r_ = q * q
    return (((((a[0]*r_+a[1])*r_+a[2])*r_+a[3])*r_+a[4])*r_+a[5])*q / \
           (((((b[0]*r_+b[1])*r_+b[2])*r_+b[3])*r_+b[4])*r_+1)


def smile_iv(atm_iv: float | np.ndarray, moneyness: float | np.ndarray,
             cfg: MarketConfig) -> np.ndarray:
    """
    Implied vol across strikes. log(K/S) < 0 (downside puts) gets a higher IV.

    This matters more than it looks: it is why selling puts pays better than
    selling calls, and also why a crash hurts more than the delta suggests.
    """
    if np.isscalar(moneyness) and np.isscalar(atm_iv):
        return atm_iv * (1.0 - cfg.smile_slope * _math.log(moneyness))
    log_m = np.log(np.asarray(moneyness, dtype=float))
    return np.asarray(atm_iv, dtype=float) * (1.0 - cfg.smile_slope * log_m)


# ---------------------------------------------------------------------------
# Market generation
# ---------------------------------------------------------------------------

def generate_market(cfg: MarketConfig | None = None) -> pd.DataFrame:
    """
    Returns a daily panel with one row per (day, name):
      day, name, spot, rv_true (annualised instantaneous vol), atm_iv,
      earnings_in_days, is_shock

    `rv_true` is the vol that actually generates returns. `atm_iv` is what the
    market charges for it. The gap between them is the premium being harvested,
    and it is deliberately *not* constant -- it compresses before shocks and
    spikes during them, which is what makes the timing hard.
    """
    cfg = cfg or MarketConfig()
    rng = np.random.default_rng(cfg.seed)
    n, days = cfg.n_names, cfg.n_days
    dt = 1.0 / TRADING_DAYS

    # --- volatility paths: common factor + idiosyncratic ------------------
    # Two mean-reverting log-vol deviations tracked separately -- a shared
    # market factor and a per-name idiosyncratic one. They must be kept apart:
    # folding the common factor back into the per-name state re-adds it every
    # step, which compounds into an explosive vol path rather than a
    # mean-reverting one.
    common = np.zeros(days)                  # market-wide log-vol deviation
    idio = np.zeros((days, n))               # per-name log-vol deviation
    sqrt_dt = np.sqrt(dt)
    xi_c = cfg.vol_of_vol * np.sqrt(cfg.common_vol_share)
    xi_i = cfg.vol_of_vol * np.sqrt(1 - cfg.common_vol_share)
    for t in range(1, days):
        common[t] = (common[t - 1] - cfg.vol_kappa * common[t - 1] * dt
                     + xi_c * rng.normal(0, sqrt_dt))
        idio[t] = (idio[t - 1] - cfg.vol_kappa * idio[t - 1] * dt
                   + xi_i * rng.normal(0, sqrt_dt, n))
    log_v = np.log(cfg.vol_mean) + idio + common[:, None]

    # --- volatility regime shocks (the tail) ------------------------------
    shock_mult = np.ones(days)
    shock_gap = np.zeros(days)
    p_shock = cfg.shock_per_year / TRADING_DAYS
    shock_days = []
    t = TRADING_DAYS  # no shock in the first year, so warm-up is clean
    while t < days:
        if rng.random() < p_shock:
            shock_days.append(t)
            decay = np.exp(-np.arange(cfg.shock_decay_days) / (cfg.shock_decay_days / 3))
            end = min(days, t + cfg.shock_decay_days)
            shock_mult[t:end] = np.maximum(
                shock_mult[t:end],
                1 + (cfg.shock_vol_mult - 1) * decay[:end - t])
            shock_gap[t] = cfg.shock_gap_pct * rng.uniform(0.7, 1.6)
            t += cfg.shock_decay_days
        t += 1

    rv = np.exp(log_v) * shock_mult[:, None]
    rv = np.clip(rv, 0.06, 3.0)

    # --- earnings calendar ------------------------------------------------
    period = TRADING_DAYS // cfg.earnings_per_year
    earnings = np.zeros((days, n), dtype=bool)
    for j in range(n):
        offset = int(rng.integers(0, period))
        earnings[offset::period, j] = True

    # --- price paths ------------------------------------------------------
    spot = np.zeros((days, n))
    spot[0] = rng.uniform(30, 250, n)
    for t in range(1, days):
        sig_d = rv[t - 1] / np.sqrt(TRADING_DAYS)
        # Student-t innovations: fat tails even without a regime shock.
        z = rng.standard_t(df=4, size=n) / np.sqrt(4 / (4 - 2))
        ret = sig_d * z + shock_gap[t] * rng.uniform(0.6, 1.4, n)
        ret += np.where(earnings[t], rng.normal(0, cfg.earnings_jump_vol, n), 0.0)
        spot[t] = spot[t - 1] * np.exp(np.clip(ret, -0.6, 0.6))

    # --- implied vol: expected forward RV, plus the premium ---------------
    # Built from the volatility that returns ACTUALLY realise, measured from the
    # generated price path itself -- not from the latent vol process.
    #
    # This distinction is the whole ballgame and getting it wrong invalidates
    # everything downstream. Earnings jumps and shock gaps inject real variance
    # that the OU vol process knows nothing about. Pricing options off the
    # latent process leaves implied vol systematically below true realised vol,
    # so a "+16% premium" is nominal only and frictionless straddle selling
    # loses money -- which makes every strategy result an artefact of the
    # generator rather than a fact about volatility selling. Measuring realised
    # vol from the path guarantees the premium is genuinely there to harvest.
    # (Validated by test_premium_is_actually_harvestable.)
    horizon = 21
    log_ret = np.zeros_like(spot)
    log_ret[1:] = np.log(spot[1:] / spot[:-1])
    fwd_rv = np.zeros_like(rv)
    trail_rv = np.zeros_like(rv)
    for t in range(days):
        f_end = min(days, t + 1 + horizon)
        seg = log_ret[t + 1:f_end]
        fwd_rv[t] = (seg.std(axis=0) * np.sqrt(TRADING_DAYS)
                     if len(seg) > 1 else rv[t])
        lo = max(0, t - horizon)
        seg2 = log_ret[lo:t + 1]
        trail_rv[t] = (seg2.std(axis=0) * np.sqrt(TRADING_DAYS)
                       if len(seg2) > 1 else rv[t])
    market_view = 0.55 * fwd_rv + 0.45 * trail_rv    # partial foresight
    vrp = cfg.vrp_mean + rng.normal(0, cfg.vrp_noise, (days, n))
    # The premium compresses right before shocks and spikes during them --
    # exactly the dynamic that makes naive richness screens dangerous.
    for s in shock_days:
        lo, hi = max(0, s - 12), s
        vrp[lo:hi] -= 0.14
        vrp[s:min(days, s + cfg.shock_decay_days)] += 0.30
    atm_iv = np.clip(market_view * (1.0 + vrp), 0.05, 3.5)

    days_to_earnings = np.full((days, n), 999, dtype=int)
    for j in range(n):
        idx = np.where(earnings[:, j])[0]
        for e in idx:
            lo = max(0, e - 60)
            days_to_earnings[lo:e + 1, j] = np.arange(e - lo, -1, -1)

    frames = []
    for j in range(n):
        frames.append(pd.DataFrame({
            "day": np.arange(days),
            "name": f"SYN{j:03d}",
            "spot": spot[:, j],
            "rv_true": rv[:, j],
            "atm_iv": atm_iv[:, j],
            "days_to_earnings": days_to_earnings[:, j],
            "is_shock": shock_mult > 1.05,
        }))
    panel = pd.concat(frames, ignore_index=True)
    panel.attrs["shock_days"] = shock_days
    panel.attrs["config"] = cfg
    return panel


def spread_frac(iv: float, is_wing: bool = False) -> float:
    """
    Bid-ask as a fraction of an option's mid price.

    Widens with volatility (market makers charge more when they are less sure)
    and is worse on far-OTM wings, where the premium is small and the
    proportional spread is brutal. Both effects punish the strategy precisely
    when it is most active, which is the point of modelling them.
    """
    base = 0.025 + 0.030 * min(iv, 1.5)
    return base * (2.2 if is_wing else 1.0)
