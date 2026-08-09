"""
Weather markets against public ensemble forecasts.

The one lead in this project with a named counterparty AND direct evidence
somebody is already collecting on it: a wallet trading 1,831 weather markets at
t=10.19, whose edge survives its own out-of-sample split.

**The thesis, and why it is not a pattern in a dataset.** Numerical weather
prediction is one of the genuinely great forecasting achievements. NOAA's GFS
and ECMWF run physics-based ensembles, dozens of perturbed members, published
free. Polymarket's temperature-band markets are priced by people reading a
single headline number off a phone app, or guessing. The counterparty is not
making a subtle error -- they are not using the best available public forecast,
because pulling and calibrating an ensemble is work rather than insight. That
is the most durable kind of retail edge: available to anyone willing to do the
annoying part, and not competed away by being written about.

**The trap that would make all of this fake.** A backtest must use the forecast
AS IT STOOD when the market was trading. A reanalysis, or a forecast issued
after the trade, contains the answer. Open-Meteo archives historical FORECASTS
separately from historical observations, and this module refuses to score any
trade whose forecast was issued after it -- see `assert_no_lookahead`. Without
that, this measures nothing but the weather having happened.

**The second trap: ensembles are underdispersed.** A raw ensemble is
overconfident -- the members cluster tighter than reality. Reading band
probabilities straight off member counts produces a forecast that looks sharper
than the market and is not. `calibration_report` checks the forecast against
outcomes BEFORE any edge is claimed, because an uncalibrated forecast beating a
market price is an artefact, not an edge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_HIST_FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPEN_METEO_ENSEMBLE = "https://ensemble-api.open-meteo.com/v1/ensemble"

# Cities Polymarket runs temperature markets on, with coordinates. Kept as a
# constant rather than geocoded at runtime: a silent geocoding miss would
# forecast the wrong city and look like a bad forecast rather than a bug.
CITIES = {
    "nyc": (40.7128, -74.0060), "new york": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437), "la": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298), "houston": (29.7604, -95.3698),
    "miami": (25.7617, -80.1918), "denver": (39.7392, -104.9903),
    "seattle": (47.6062, -122.3321), "boston": (42.3601, -71.0589),
    "washington": (38.9072, -77.0369), "dc": (38.9072, -77.0369),
    "philadelphia": (39.9526, -75.1652), "atlanta": (33.7490, -84.3880),
    "phoenix": (33.4484, -112.0740), "dallas": (32.7767, -96.7970),
    "london": (51.5074, -0.1278), "paris": (48.8566, 2.3522),
    "moscow": (55.7558, 37.6173), "tokyo": (35.6762, 139.6503),
}


@dataclass
class TempMarket:
    """A parsed temperature-band market."""
    market_id: str
    city: str
    lat: float
    lon: float
    date: pd.Timestamp
    lo_f: float          # band lower bound, inclusive, Fahrenheit
    hi_f: float          # band upper bound, exclusive
    raw: str


def parse_temperature_market(market_id: str, question: str,
                             end_date=None) -> TempMarket | None:
    """
    Turn a market question into a city, a date and a temperature band.

    Deliberately conservative: anything it cannot parse UNAMBIGUOUSLY returns
    None rather than a guess. A mis-parsed band silently scores the wrong
    outcome, which would show up as a forecast that cannot predict -- the
    failure mode this project has hit four times in other guises.
    """
    if not question:
        return None
    q = str(question).lower()
    if "temperature" not in q and "temp" not in q and "degrees" not in q:
        return None

    city = next((c for c in sorted(CITIES, key=len, reverse=True) if c in q), None)
    if city is None:
        return None
    lat, lon = CITIES[city]

    # Band forms seen on Polymarket, most specific first.
    lo = hi = None
    m = re.search(r"(\d{1,3})\s*(?:-|to|–)\s*(\d{1,3})\s*(?:°|degrees|f\b)", q)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2)) + 1.0
    if lo is None:
        m = re.search(r"(?:above|over|greater than|higher than|\bat least\b)\s*"
                      r"(\d{1,3})", q)
        if m:
            lo, hi = float(m.group(1)), np.inf
    if lo is None:
        m = re.search(r"(?:below|under|less than|lower than)\s*(\d{1,3})", q)
        if m:
            lo, hi = -np.inf, float(m.group(1))
    if lo is None:
        return None

    date = pd.to_datetime(end_date, errors="coerce", utc=True) if end_date is not None else pd.NaT
    if pd.isna(date):
        return None
    return TempMarket(str(market_id), city, lat, lon, date, lo, hi, str(question))


def ensemble_band_probability(members_f: np.ndarray, lo_f: float, hi_f: float,
                              spread_inflation: float = 1.0) -> float:
    """
    Probability the day's high lands in [lo, hi), from ensemble members.

    `spread_inflation` widens the member spread about its mean. Raw ensembles
    are systematically underdispersed -- members agree with each other more
    than they agree with reality -- so reading probabilities straight off member
    counts yields a forecast that LOOKS sharper than the market while being
    overconfident. The multiplier is a fitted correction, and it must be fitted
    on data the edge is not then measured on.
    """
    x = np.asarray(members_f, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if spread_inflation != 1.0:
        mu = x.mean()
        x = mu + (x - mu) * spread_inflation
    return float(((x >= lo_f) & (x < hi_f)).mean())


def fit_spread_inflation(members: list[np.ndarray], observed_f: np.ndarray,
                         grid=np.arange(0.6, 3.01, 0.05)) -> float:
    """
    Choose the inflation that makes the ensemble HONEST, not the one that pays.

    Fitted by minimising the continuous ranked probability score against
    observations -- a proper scoring rule, so it cannot be gamed by widening or
    narrowing to taste. Note this is a fit, so it is a researcher degree of
    freedom and belongs in the training period only.
    """
    obs = np.asarray(observed_f, dtype=float)
    best, best_score = 1.0, np.inf
    for k in grid:
        total = 0.0
        for mem, o in zip(members, obs):
            x = np.asarray(mem, dtype=float)
            x = x[np.isfinite(x)]
            if x.size == 0 or not np.isfinite(o):
                continue
            mu = x.mean()
            x = mu + (x - mu) * k
            # CRPS via the empirical-ensemble formula.
            term1 = np.abs(x - o).mean()
            term2 = np.abs(x[:, None] - x[None, :]).mean() / 2.0
            total += term1 - term2
        if total < best_score:
            best, best_score = float(k), total
    return best


def assert_no_lookahead(signals: pd.DataFrame) -> None:
    """
    Refuse any row whose forecast was issued after the trade it is scoring.

    The single most dangerous line in this module. A forecast issued after the
    market traded contains information nobody had, and a backtest built on it
    produces a large, confident, entirely fictional edge.
    """
    need = {"forecast_issued_at", "traded_at"}
    missing = need - set(signals.columns)
    if missing:
        raise ValueError(f"cannot check lookahead without {sorted(missing)}")
    bad = signals["forecast_issued_at"] > signals["traded_at"]
    if bool(bad.any()):
        raise ValueError(
            f"{int(bad.sum())} of {len(signals)} rows use a forecast issued "
            f"AFTER the trade. That is lookahead and would fabricate an edge.")


def calibration_report(signals: pd.DataFrame, bins=10) -> pd.DataFrame:
    """
    Is the FORECAST itself calibrated? Asked before any edge is claimed.

    When the forecast says 30%, does it happen 30% of the time? An
    uncalibrated forecast can beat a market price on paper purely by being
    overconfident in the right direction on a small sample, and that is not an
    edge. If this table is not close to the diagonal, nothing downstream means
    anything.
    """
    s = signals.dropna(subset=["p_forecast", "outcome"])
    if s.empty:
        return pd.DataFrame(columns=["bucket", "n", "mean_forecast", "realised"])
    cut = pd.cut(s["p_forecast"], bins=np.linspace(0, 1, bins + 1),
                 include_lowest=True)
    out = (s.groupby(cut, observed=True)
            .agg(n=("outcome", "size"), mean_forecast=("p_forecast", "mean"),
                 realised=("outcome", "mean")).reset_index())
    out.columns = ["bucket", "n", "mean_forecast", "realised"]
    out["gap"] = (out["realised"] - out["mean_forecast"]).round(4)
    return out


def edge_vs_market(signals: pd.DataFrame, min_disagreement: float = 0.10,
                   half_spread_cents: float = 1.0) -> dict:
    """
    Trade only where the forecast DISAGREES with the price, and score it.

    The signal is the disagreement, not the forecast. Buying when the forecast
    merely agrees with the market earns the spread cost and nothing else, so
    `min_disagreement` is what turns a forecast into a strategy.

    Direction: buy the band when the forecast says it is likelier than the price
    implies, sell when it says the opposite. Outcome is scored against what the
    weather actually did.
    """
    s = signals.dropna(subset=["p_forecast", "p_market", "outcome"]).copy()
    if s.empty:
        return {"verdict": "no scorable signals", "n": 0}
    s["disagreement"] = s["p_forecast"] - s["p_market"]
    s = s[s["disagreement"].abs() >= min_disagreement]
    if s.empty:
        return {"verdict": f"no signal disagreed by >= {min_disagreement}",
                "n": 0}

    side = np.sign(s["disagreement"])
    entry = np.where(side > 0, s["p_market"], 1.0 - s["p_market"])
    payoff = np.where(side > 0, s["outcome"], 1.0 - s["outcome"])
    gross = float(np.mean(payoff - entry))
    cost = half_spread_cents / 100.0
    net = gross - cost
    n = int(len(s))
    sd = float(np.std(payoff - entry, ddof=1)) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 and sd > 0 else np.nan
    return {
        "n": n,
        "n_markets": int(s["market_id"].nunique()) if "market_id" in s else n,
        "avg_disagreement": round(float(s["disagreement"].abs().mean()), 4),
        "gross_edge_cents": round(gross * 100, 2),
        "cost_cents": round(cost * 100, 2),
        "net_edge_cents": round(net * 100, 2),
        "t_stat": round(net / se, 2) if se and np.isfinite(se) and se > 0 else np.nan,
        "verdict": ("forecast beats the market price net of cost" if net > 0
                    else "no edge after cost"),
    }
