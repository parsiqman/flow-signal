"""
Backtest engine, and the diagnostics that matter for a negatively skewed book.

Sharpe ratio is close to useless here and is reported only because its absence
would be conspicuous. A short-volatility strategy manufactures a high Sharpe by
converting a small chance of a large loss into a steady drip of small gains --
which is exactly what the ratio is bad at seeing. The numbers to read are:

  - worst single month, and the drawdown around each shock
  - what fraction of total profit the worst 1% of days gives back
  - whether the strategy survives with the regime filter switched OFF, which
    is the honest test of whether the wings alone are enough

Positions are held to expiration unless the profit target is hit, so the exit
spread is mostly avoided. That single design choice is worth more than any
signal refinement, and it falls straight out of the cost analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import (Position, StrategyConfig, build_condor, condor_payoff,
                       forecast_realised_vol, regime_multiplier, richness,
                       select_candidates)
from .synthetic import (MarketConfig, bs_price_scalar, smile_iv, spread_frac,
                        TRADING_DAYS)


def run_backtest(panel: pd.DataFrame, cfg: StrategyConfig | None = None,
                 mkt: MarketConfig | None = None,
                 starting_equity: float = 100_000.0,
                 warmup_days: int = 70) -> dict:
    """
    Walk the panel forward one day at a time, opening and settling positions.

    No lookahead: the forecast at day t uses prices up to and including t, and
    positions opened at t are priced with day-t implied vol only.
    """
    cfg = cfg or StrategyConfig()
    mkt = mkt or MarketConfig()

    wide_spot = panel.pivot(index="day", columns="name", values="spot")
    wide_iv = panel.pivot(index="day", columns="name", values="atm_iv")
    wide_earn = panel.pivot(index="day", columns="name", values="days_to_earnings")
    names = list(wide_spot.columns)
    days = list(wide_spot.index)
    # Scalar .loc lookups dominate the runtime once there are hundreds of open
    # positions marked every day, so the panel is dropped to plain arrays and
    # indexed positionally from here on.
    spot_a = wide_spot.to_numpy()
    iv_a = wide_iv.to_numpy()
    earn_a = wide_earn.to_numpy()
    col = {n: i for i, n in enumerate(names)}
    market_proxy = spot_a.mean(axis=1)

    equity = starting_equity
    open_positions: list[tuple[Position, dict]] = []
    equity_curve, daily_pnl, trade_log = [], [], []

    for t in days:
        realised = 0.0

        # --- settle anything expiring today ------------------------------
        still_open = []
        for pos, strikes in open_positions:
            if pos.expiry_day > t:
                still_open.append((pos, strikes))
                continue
            s_exp = float(spot_a[t, col[pos.name]])
            owed = condor_payoff(s_exp, strikes)
            pnl = (pos.credit - owed) * pos.units
            realised += pnl
            trade_log.append({
                "name": pos.name, "entry_day": pos.entry_day, "exit_day": t,
                "exit": "expiry", "richness": pos.richness,
                "capital_at_risk": pos.capital_at_risk(), "pnl": pnl,
                "return_on_risk": pnl / max(pos.capital_at_risk(), 1e-9),
            })
        open_positions = still_open

        # --- take profits early where the target is hit -------------------
        if open_positions:
            keep = []
            for pos, strikes in open_positions:
                t_rem = max((pos.expiry_day - t) / TRADING_DAYS, 1e-6)
                j = col[pos.name]
                s_now = float(spot_a[t, j])
                iv_now = float(iv_a[t, j])
                cost_to_close = _condor_mark(s_now, strikes, t_rem, iv_now, mkt,
                                             include_spread=True,
                                             fill_quality=cfg.fill_quality)
                captured = pos.credit - cost_to_close
                if captured >= cfg.profit_target * pos.credit:
                    pnl = captured * pos.units
                    realised += pnl
                    trade_log.append({
                        "name": pos.name, "entry_day": pos.entry_day, "exit_day": t,
                        "exit": "target", "richness": pos.richness,
                        "capital_at_risk": pos.capital_at_risk(), "pnl": pnl,
                        "return_on_risk": pnl / max(pos.capital_at_risk(), 1e-9),
                    })
                else:
                    keep.append((pos, strikes))
            open_positions = keep

        equity += realised

        # --- open new positions on the entry schedule ---------------------
        is_entry_day = (t >= warmup_days) and (t % cfg.entry_every_days == 0)
        if is_entry_day and t + cfg.dte <= days[-1]:
            hist = spot_a[max(0, t - 63):t + 1]
            fc = forecast_realised_vol(hist)
            iv_now = iv_a[t]
            snap = pd.DataFrame({
                "name": names,
                "spot": spot_a[t],
                "atm_iv": iv_now,
                "forecast_rv": fc,
                "richness": richness(iv_now, fc),
                "days_to_earnings": earn_a[t],
            })
            picks = select_candidates(snap, cfg)

            size_mult = regime_multiplier(market_proxy[:t + 1], cfg)
            committed = sum(p.capital_at_risk() for p, _ in open_positions)
            room = max(0.0, cfg.max_portfolio_risk * equity - committed)

            for _, row in picks.iterrows():
                built = build_condor(float(row["spot"]), float(row["atm_iv"]),
                                     cfg, mkt)
                if built is None:
                    continue
                strikes, credit, max_loss = built
                budget = min(cfg.risk_per_position * equity * size_mult, room)
                if budget <= 0:
                    break
                units = budget / max_loss
                if units <= 0:
                    continue
                pos = Position(
                    name=row["name"], entry_day=t, expiry_day=t + cfg.dte,
                    spot_at_entry=float(row["spot"]),
                    short_put=strikes["short_put"], long_put=strikes["long_put"],
                    short_call=strikes.get("short_call", float("nan")),
                    long_call=strikes.get("long_call", float("nan")),
                    credit=credit, max_loss=max_loss, units=units,
                    richness=float(row["richness"]),
                )
                open_positions.append((pos, strikes))
                room -= pos.capital_at_risk()

        # --- mark the book ------------------------------------------------
        unrealised = 0.0
        for pos, strikes in open_positions:
            t_rem = max((pos.expiry_day - t) / TRADING_DAYS, 1e-6)
            j = col[pos.name]
            s_now = float(spot_a[t, j])
            iv_now = float(iv_a[t, j])
            mark = _condor_mark(s_now, strikes, t_rem, iv_now, mkt,
                                include_spread=False)
            unrealised += (pos.credit - mark) * pos.units

        equity_curve.append({"day": t, "equity": equity,
                             "marked_equity": equity + unrealised,
                             "n_open": len(open_positions),
                             "at_risk": sum(p.capital_at_risk()
                                            for p, _ in open_positions)})
        daily_pnl.append(realised)

    curve = pd.DataFrame(equity_curve).set_index("day")
    trades = pd.DataFrame(trade_log)
    return {"curve": curve, "trades": trades,
            "stats": summarise(curve, trades, starting_equity, panel)}


def _condor_mark(spot: float, k: dict, t_years: float, atm_iv: float,
                 mkt: MarketConfig, include_spread: bool,
                 fill_quality: float = 1.0) -> float:
    """Cost to buy the position back. Spread included only for real exits."""
    spec = [(k["short_put"], False, True), (k["long_put"], False, False)]
    if "short_call" in k:
        spec += [(k["short_call"], True, True), (k["long_call"], True, False)]
    total = 0.0
    for strike, is_call, is_short in spec:
        iv_k = float(smile_iv(atm_iv, strike / spot, mkt))
        mid = bs_price_scalar(spot, strike, t_years, iv_k, is_call, mkt.risk_free)
        if include_spread:
            half = mid * spread_frac(atm_iv, is_wing=not is_short) / 2.0 * fill_quality
            # Closing reverses each leg: buy back shorts at ask, sell longs at bid.
            mid = (mid + half) if is_short else max(mid - half, 0.0)
        total += mid if is_short else -mid
    return total


def summarise(curve: pd.DataFrame, trades: pd.DataFrame,
              starting_equity: float, panel: pd.DataFrame) -> dict:
    """Headline numbers, with the tail diagnostics given equal billing."""
    eq = curve["marked_equity"]
    rets = eq.pct_change().dropna()
    years = len(curve) / TRADING_DAYS
    total_ret = eq.iloc[-1] / starting_equity - 1
    cagr = (eq.iloc[-1] / starting_equity) ** (1 / max(years, 1e-9)) - 1

    running_max = eq.cummax()
    dd = eq / running_max - 1
    max_dd = float(dd.min())

    sharpe = (float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS))
              if rets.std() > 0 else np.nan)
    downside = rets[rets < 0]
    sortino = (float(rets.mean() / downside.std() * np.sqrt(TRADING_DAYS))
               if len(downside) > 1 and downside.std() > 0 else np.nan)

    # How much of the gain does the worst 1% of days hand back? For a short-vol
    # book this is the number that decides whether the strategy is viable.
    worst_1pct = rets.nsmallest(max(1, len(rets) // 100)).sum()

    # Monthly aggregation, which is how a drawdown is actually experienced.
    monthly = eq.iloc[::21].pct_change().dropna()

    win_rate = float((trades["pnl"] > 0).mean()) if len(trades) else np.nan
    shock_days = panel.attrs.get("shock_days", [])
    shock_dd = []
    for s in shock_days:
        window = eq.loc[max(0, s - 5):min(eq.index[-1], s + 40)]
        if len(window) > 2:
            shock_dd.append(float(window.min() / window.iloc[0] - 1))

    return {
        "years": round(years, 1),
        "total_return": round(float(total_ret), 3),
        "cagr": round(float(cagr), 4),
        "sharpe": round(sharpe, 2) if sharpe == sharpe else np.nan,
        "sortino": round(sortino, 2) if sortino == sortino else np.nan,
        "max_drawdown": round(max_dd, 3),
        "worst_month": round(float(monthly.min()), 3) if len(monthly) else np.nan,
        "worst_1pct_days_give_back": round(float(worst_1pct), 3),
        "n_trades": int(len(trades)),
        "win_rate": round(win_rate, 3) if win_rate == win_rate else np.nan,
        "avg_return_on_risk": (round(float(trades["return_on_risk"].mean()), 4)
                               if len(trades) else np.nan),
        "n_shocks_in_sample": len(shock_days),
        "worst_shock_drawdown": (round(min(shock_dd), 3) if shock_dd else np.nan),
        "final_equity": round(float(eq.iloc[-1]), 0),
    }
