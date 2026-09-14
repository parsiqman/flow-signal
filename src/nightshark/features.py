"""Turn a raw candle tape into a compact description of market state.

Four families, as specified: volatility, trend, momentum, and multi-timeframe
candle structure. The output is deliberately small. Every token in this summary
is paid for on every decision, four times an hour, forever -- and a model given
600 numbers reasons worse than one given 40 that matter.

Two rules shape the format:

1. Numbers are normalized (basis points, ratios, percentiles) rather than raw.
   "close 84326.04" forces the model to do arithmetic to learn anything;
   "ret15m +23bp" is the same fact already useful, and the model cannot get the
   subtraction wrong.
2. Nothing here reads a candle that had not closed by `as_of`. The feed enforces
   that, and this module never looks at anything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .data import CandleSeries

ANNUALIZE_1M = np.sqrt(365 * 24 * 60)


# --------------------------------------------------------------------------
# indicator primitives
# --------------------------------------------------------------------------
def ema(x: np.ndarray, span: int) -> np.ndarray:
    if x.size == 0:
        return x
    a = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, x.size):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rsi(close: np.ndarray, period: int = 14) -> float:
    """Wilder's RSI. Returns 50.0 (neutral) when there is not enough history."""
    if close.size < period + 1:
        return 50.0
    d = np.diff(close)
    gain, loss = np.clip(d, 0, None), -np.clip(d, None, 0)
    ag, al = gain[:period].mean(), loss[:period].mean()
    for i in range(period, d.size):
        ag = (ag * (period - 1) + gain[i]) / period
        al = (al * (period - 1) + loss[i]) / period
    if al == 0:
        return 100.0 if ag > 0 else 50.0
    return float(100 - 100 / (1 + ag / al))


def atr(s: CandleSeries, period: int = 14) -> float:
    """Average true range as a fraction of price, in basis points."""
    if len(s) < 2:
        return 0.0
    prev = s.close[:-1]
    tr = np.maximum(s.high[1:] - s.low[1:],
                    np.maximum(np.abs(s.high[1:] - prev), np.abs(s.low[1:] - prev)))
    n = min(period, tr.size)
    return float(tr[-n:].mean() / s.close[-1] * 10_000)


def realized_vol_bp(close: np.ndarray, n: int) -> float:
    """Annualized realized vol from the last n 1-minute log returns, in %."""
    if close.size < n + 1:
        n = close.size - 1
    if n < 2:
        return 0.0
    r = np.diff(np.log(close[-(n + 1):]))
    return float(r.std(ddof=1) * ANNUALIZE_1M * 100)


def slope_bp_per_min(close: np.ndarray, n: int) -> float:
    """Least-squares trend of log price over the last n bars, in bp per minute."""
    n = min(n, close.size)
    if n < 3:
        return 0.0
    y = np.log(close[-n:])
    x = np.arange(n, dtype=float)
    b = np.polyfit(x, y, 1)[0]
    return float(b * 10_000)


def r_squared(close: np.ndarray, n: int) -> float:
    """How straight that trend is. Direction without this is half the picture:
    a clean 20bp drift and a whipsaw averaging 20bp are different markets."""
    n = min(n, close.size)
    if n < 3:
        return 0.0
    y = np.log(close[-n:])
    x = np.arange(n, dtype=float)
    fit = np.polyval(np.polyfit(x, y, 1), x)
    ss_res = float(((y - fit) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 0.0 if ss_tot == 0 else round(1 - ss_res / ss_tot, 2)


def candle_shape(s: CandleSeries, i: int) -> str:
    """One candle as `<dir><body>/<upper wick>/<lower wick>`, each in bp of price.

    Compact on purpose: `+12/3/8` is a green candle with a 12bp body, a 3bp upper
    wick and an 8bp lower wick -- rejection from below, buyers in control.
    """
    o, h, l, c = float(s.open[i]), float(s.high[i]), float(s.low[i]), float(s.close[i])
    scale = 10_000 / c if c else 0
    body = (c - o) * scale
    upper = (h - max(o, c)) * scale
    lower = (min(o, c) - l) * scale
    return f"{body:+.0f}/{upper:.0f}/{lower:.0f}"


# --------------------------------------------------------------------------
# feature bundle
# --------------------------------------------------------------------------
@dataclass
class Features:
    as_of: datetime
    spot: float
    window_end: datetime | None
    minutes_left: float
    values: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "spot": round(self.spot, 2),
            "window_end": self.window_end.isoformat() if self.window_end else None,
            "minutes_left": round(self.minutes_left, 2),
            **self.values,
        }

    def to_prompt_text(self) -> str:
        """The token-efficient rendering that actually goes to the model."""
        v = self.values
        lines = [
            f"T {self.as_of:%H:%M}Z  spot {self.spot:,.0f}  "
            f"settles {self.window_end:%H:%M}Z in {self.minutes_left:.0f}m"
            if self.window_end else f"T {self.as_of:%H:%M}Z  spot {self.spot:,.0f}",

            f"VOL rv15 {v['rv_15m']:.0f}% rv60 {v['rv_60m']:.0f}% rv240 {v['rv_240m']:.0f}% "
            f"atr1m {v['atr_1m_bp']:.0f}bp atr5m {v['atr_5m_bp']:.0f}bp regime {v['vol_regime']}",

            f"TREND slope5m {v['slope_5m_bp']:+.1f} slope30m {v['slope_30m_bp']:+.1f} "
            f"r2_30m {v['r2_30m']:.2f} ema9v21 {v['ema_9_21_bp']:+.0f}bp "
            f"ema21v50 {v['ema_21_50_bp']:+.0f}bp vwap {v['vs_vwap_bp']:+.0f}bp",

            f"MOM rsi1m {v['rsi_1m']:.0f} rsi5m {v['rsi_5m']:.0f} "
            f"ret5m {v['ret_5m_bp']:+.0f}bp ret15m {v['ret_15m_bp']:+.0f}bp "
            f"ret60m {v['ret_60m_bp']:+.0f}bp macd5m {v['macd_hist_5m_bp']:+.0f}bp",

            f"RANGE pos_60m {v['range_pos_60m']:.2f} up_min_20 {v['up_minute_frac_20']:.2f} "
            f"streak {v['streak']:+d} vs_window_open {v['vs_window_open_bp']:+.0f}bp",

            "CANDLES body/upwick/lowwick in bp, oldest->newest",
            f"  1m  {' '.join(v['candles_1m'])}",
            f"  5m  {' '.join(v['candles_5m'])}",
            f"  15m {' '.join(v['candles_15m'])}",
        ]
        return "\n".join(lines)


def extract(s1m: CandleSeries, as_of: datetime, window_end: datetime | None = None,
            window_open_price: float | None = None) -> Features:
    """Build the feature bundle from 1-minute candles closed at or before `as_of`."""
    if len(s1m) < 60:
        raise ValueError(f"need >=60 1m candles to extract features, got {len(s1m)}")

    s5, s15 = s1m.resample(5), s1m.resample(15)
    c = s1m.close
    spot = float(c[-1])

    def ret_bp(n: int) -> float:
        n = min(n, c.size - 1)
        return float((c[-1] / c[-1 - n] - 1) * 10_000) if n > 0 else 0.0

    rv15, rv60, rv240 = (realized_vol_bp(c, n) for n in (15, 60, 240))
    # Vol regime as a percentile of the last 24h of rolling 15m vol: tells the
    # model whether "60% annualized" is quiet or frantic for this tape right now.
    tail = c[-min(c.size, 1440):]
    if tail.size > 30:
        r = np.diff(np.log(tail))
        roll = np.array([r[i:i + 15].std(ddof=1) for i in range(0, r.size - 15, 5)])
        pct = float((roll < (rv15 / 100 / ANNUALIZE_1M)).mean()) if roll.size else 0.5
    else:
        pct = 0.5
    vol_regime = "low" if pct < 0.33 else ("high" if pct > 0.67 else "mid")

    e9, e21, e50 = (ema(s5.close, n) for n in (9, 21, 50)) if len(s5) >= 50 else (
        ema(s5.close, 9), ema(s5.close, 21), ema(s5.close, min(50, max(2, len(s5)))))
    macd = ema(s5.close, 12) - ema(s5.close, 26)
    macd_hist = macd - ema(macd, 9)

    typical = (s1m.high + s1m.low + s1m.close) / 3
    vwap_n = min(240, len(s1m))
    vw = float((typical[-vwap_n:] * s1m.volume[-vwap_n:]).sum() / max(s1m.volume[-vwap_n:].sum(), 1e-9))

    hi60, lo60 = float(s1m.high[-60:].max()), float(s1m.low[-60:].min())
    range_pos = 0.5 if hi60 == lo60 else (spot - lo60) / (hi60 - lo60)

    d = np.sign(np.diff(c[-21:]))
    streak = 0
    for x in d[::-1]:
        if x == 0 or (streak and np.sign(streak) != x):
            break
        streak += int(x) if x else 0

    minutes_left = (window_end - as_of).total_seconds() / 60 if window_end else 0.0
    vs_open = ((spot / window_open_price - 1) * 10_000) if window_open_price else 0.0

    values = {
        "rv_15m": round(rv15, 1), "rv_60m": round(rv60, 1), "rv_240m": round(rv240, 1),
        "atr_1m_bp": round(atr(s1m, 14), 1), "atr_5m_bp": round(atr(s5, 14), 1),
        "vol_regime": vol_regime, "vol_pctile": round(pct, 2),

        "slope_5m_bp": round(slope_bp_per_min(c, 5), 2),
        "slope_30m_bp": round(slope_bp_per_min(c, 30), 2),
        "r2_30m": r_squared(c, 30),
        "ema_9_21_bp": round(float((e9[-1] / e21[-1] - 1) * 10_000), 1),
        "ema_21_50_bp": round(float((e21[-1] / e50[-1] - 1) * 10_000), 1),
        "vs_vwap_bp": round(float((spot / vw - 1) * 10_000), 1),

        "rsi_1m": round(rsi(c, 14), 1), "rsi_5m": round(rsi(s5.close, 14), 1),
        "ret_5m_bp": round(ret_bp(5), 1), "ret_15m_bp": round(ret_bp(15), 1),
        "ret_60m_bp": round(ret_bp(60), 1),
        "macd_hist_5m_bp": round(float(macd_hist[-1] / spot * 10_000), 1),

        "range_pos_60m": round(float(range_pos), 2),
        "up_minute_frac_20": round(float((np.diff(c[-21:]) > 0).mean()), 2),
        "streak": int(streak),
        "vs_window_open_bp": round(float(vs_open), 1),

        "candles_1m": [candle_shape(s1m, i) for i in range(len(s1m) - 6, len(s1m))],
        "candles_5m": [candle_shape(s5, i) for i in range(max(0, len(s5) - 6), len(s5))],
        "candles_15m": [candle_shape(s15, i) for i in range(max(0, len(s15) - 4), len(s15))],
    }
    return Features(as_of=as_of, spot=spot, window_end=window_end,
                    minutes_left=minutes_left, values=values)
