"""
Which options strategy is worth building? Decided on numbers, not taste.

Seven families are on the table. They are compared on the things that actually
determine whether a solo researcher can build, prove, and survive a strategy:

  1. How many independent bets per year does it hand you (can you ever prove it)
  2. How much friction per bet, relative to the bet's own volatility
  3. Is the edge a risk premium (durable) or a mispricing (decays)
  4. What data does it need, and what does that cost
  5. How badly is it skewed, and how often does the bad tail actually arrive

Point 5 is the one that makes options different from equities, and it is where
most retail options backtests quietly lie. A short-volatility strategy has a
wonderful Sharpe ratio right up until the day it does not, and a t-statistic
computed on normally-distributed assumptions cannot see that coming. So tail
frequency is a first-class criterion here, not a footnote.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Same implementation-loss haircut used throughout: what fraction of a paper
# signal actually reaches the portfolio after constraints, sizing and skipped
# trades. Grinold & Kahn's transfer coefficient.
TRANSFER_COEFFICIENT = 0.5
T_TARGET = 2.0


@dataclass
class Family:
    """One options strategy family, described in decision-relevant terms."""

    name: str
    thesis: str

    # --- breadth ---------------------------------------------------------
    n_underlyings: int
    residual_corr: float        # co-movement of the bets after hedging
    cycles_per_year: float      # entries per underlying per year

    # --- friction, as a fraction of capital at risk per position ---------
    entry_cost_frac: float
    exit_cost_frac: float       # 0 when the position is held to expiration
    bet_sigma_frac: float       # return sd of one position

    # --- edge ------------------------------------------------------------
    plausible_ic: float         # per-bet Sharpe of a good signal here
    edge_type: str              # 'risk_premium' (durable) or 'mispricing' (decays)
    # Decades of published, out-of-sample, third-party evidence that the edge
    # exists. This is the difference between discovering something and
    # implementing something already known -- and it changes the burden of
    # proof completely. You do not need to re-derive the equity risk premium
    # from your own five years of data, and the same applies to the variance
    # risk premium.
    published_evidence_years: float = 0.0
    # How much of a paper signal survives implementation. Low for complex
    # multi-constraint books, high for a mechanical premium harvest.
    transfer_coefficient: float = 0.5

    # --- tail ------------------------------------------------------------
    skew: str = "negative"      # 'negative', 'positive', 'symmetric'
    tail_loss_frac: float = 1.0   # loss in the bad regime, x normal position sd
    tail_every_years: float = 0.0  # how often that regime shows up

    # --- practicalities --------------------------------------------------
    data_needed: str = ""
    data_cost_usd_month: float = 0.0
    free_history_years: float = 0.0
    # History you can BUY, cheaply and once. End-of-day option chains go back
    # to the mid-2000s and are the cheapest options data that exists; tick-level
    # tape with quotes does not, at any price a solo project would pay. This
    # distinction decides whether a strategy's tail can be studied at all.
    deep_history_years: float = 0.0
    deep_history_cost_usd: float = 0.0
    min_capital_usd: float = 0.0
    notes: str = ""

    # ------------------------------------------------------------------
    def round_trip_cost_frac(self) -> float:
        return self.entry_cost_frac + self.exit_cost_frac

    def cost_to_noise(self) -> float:
        """Friction as a fraction of the position's own return volatility."""
        return self.round_trip_cost_frac() / self.bet_sigma_frac

    def net_bet_sharpe(self, ic: float | None = None) -> float:
        return (self.plausible_ic if ic is None else ic) - self.cost_to_noise()

    def effective_breadth(self) -> float:
        """
        Independent bets per year, discounted for co-movement.

        N_eff = N / (1 + (N-1) * rho). Volatility is more correlated across
        names than returns are, so options strategies lose more breadth to
        this term than equity strategies do -- an implied-vol shock hits every
        name at once.
        """
        n, rho = self.n_underlyings, self.residual_corr
        n_eff = n / (1 + (n - 1) * rho) if n > 1 else 1.0
        return n_eff * self.cycles_per_year

    def annual_ir(self, ic: float | None = None) -> float:
        return (self.transfer_coefficient * self.net_bet_sharpe(ic)
                * np.sqrt(self.effective_breadth()))

    def years_to_prove(self, ic: float | None = None) -> float:
        ir = self.annual_ir(ic)
        return np.inf if ir <= 0 else (T_TARGET / ir) ** 2

    def tail_blind_spot(self) -> float:
        """
        How much of the strategy's own worst case your data has NOT seen.

        Ratio of the tail's recurrence interval to the history available. Above
        1.0 means the blowup regime is, on average, absent from your backtest --
        so the backtest is measuring the good half of the distribution only.
        This is the single most common way an options backtest flatters itself.
        """
        available = max(self.free_history_years, self.deep_history_years)
        if self.skew != "negative" or available <= 0:
            return 0.0
        return self.tail_every_years / available

    def proof_burden(self) -> str:
        """
        What question does this family actually pose?

        A documented risk premium does not need rediscovering from your own
        short history -- the research problem is implementation and survival.
        A mispricing with no published record has to be established from
        scratch, against your own data, with all the statistical difficulty
        that implies. Conflating the two is how people spend two years
        'validating' the equity risk premium.
        """
        if self.edge_type == "risk_premium" and self.published_evidence_years >= 15:
            return "implement"   # edge is established; risk engineering is the job
        return "discover"        # you must prove it yourself, from your own data

    def survivable(self) -> bool:
        """
        Can one bad regime end the project?

        A strategy whose tail loss exceeds ~8 normal position sigmas is not
        risk-managed by position sizing alone -- it needs structurally defined
        risk (bought wings), or it is a bet on not being unlucky.
        """
        return self.tail_loss_frac <= 8.0


# ---------------------------------------------------------------------------
# The candidate set.
#
# Parameters are assumptions, stated here so they can be argued with directly.
# Costs are expressed as a fraction of capital at risk per position, which puts
# a defined-risk credit spread and a long lottery ticket on one scale.
# ---------------------------------------------------------------------------

CROSS_SECTIONAL_VRP = Family(
    name="Cross-sectional VRP (single names)",
    thesis="Sell implied vol where it is richest vs forecast realised vol, buy "
           "where cheapest, vega-neutral across ~400 liquid names.",
    n_underlyings=400,
    residual_corr=0.20,      # vol shocks are common; less idiosyncratic than returns
    cycles_per_year=12,      # monthly, 30-45 DTE
    entry_cost_frac=0.017,   # ~4% of premium on a defined-risk spread
    exit_cost_frac=0.004,    # mostly held to expiry; only early exits pay
    bet_sigma_frac=0.50,
    plausible_ic=0.05,
    edge_type="risk_premium",
    skew="negative",
    tail_loss_frac=4.0,      # wings cap the loss - this is why they are worth buying
    tail_every_years=6.0,
    data_needed="EOD option chains with IV + underlying history",
    data_cost_usd_month=60.0,
    free_history_years=2.0,
    min_capital_usd=10_000,
    published_evidence_years=20.0,   # VRP documented since Bakshi-Kapadia (2003)
    transfer_coefficient=0.5,        # complex book, many constraints
        deep_history_years=19.0,
    deep_history_cost_usd=600.0,
    notes="The options analogue of cross-sectional equity stat-arb. Note that\n           vega-neutralising strips out the very premium that makes short vol\n           pay, leaving only the weaker cross-sectional dispersion signal.",
)

INDEX_SHORT_VOL = Family(
    name="Index short volatility (SPX/SPY)",
    thesis="Harvest the index variance risk premium directly. The single "
           "best-documented edge in derivatives.",
    n_underlyings=2,
    residual_corr=0.95,
    cycles_per_year=12,
    entry_cost_frac=0.006,   # index options are the tightest markets available
    exit_cost_frac=0.002,
    bet_sigma_frac=0.50,
    plausible_ic=0.25,       # large and reliable - it is compensation, not a forecast
    edge_type="risk_premium",
    skew="negative",
    tail_loss_frac=15.0,     # Feb 2018, Mar 2020, Aug 2024. Undefended, this ends you.
    tail_every_years=6.0,
    data_needed="SPX/SPY chains + VIX term structure",
    data_cost_usd_month=0.0,
    free_history_years=5.0,
    min_capital_usd=25_000,
    published_evidence_years=25.0,   # Carr-Wu, Bakshi-Kapadia, CBOE PUT index
    transfer_coefficient=0.8,        # mechanical: sell monthly, little slippage
        deep_history_years=19.0,
    deep_history_cost_usd=300.0,
    notes="Highest per-bet edge of any family, and almost no breadth.",
)

EARNINGS_VOL_CRUSH = Family(
    name="Earnings volatility crush",
    thesis="Implied vol inflates into an earnings date and collapses after. "
           "Sell the inflation, close the day after the print.",
    n_underlyings=500,
    residual_corr=0.05,      # earnings outcomes are genuinely idiosyncratic
    cycles_per_year=4,
    entry_cost_frac=0.030,   # wide pre-earnings markets, and you MUST pay to exit
    exit_cost_frac=0.030,
    bet_sigma_frac=0.60,
    plausible_ic=0.06,
    edge_type="mispricing",
    skew="negative",
    tail_loss_frac=6.0,
    tail_every_years=3.0,
    data_needed="EOD chains + earnings calendar (calendar is free)",
    data_cost_usd_month=60.0,
    free_history_years=2.0,
    min_capital_usd=10_000,
    published_evidence_years=8.0,    # documented, but crowded and decaying
    transfer_coefficient=0.6,
        deep_history_years=19.0,
    deep_history_cost_usd=600.0,
    notes="Cannot hold to expiry, so it pays the spread twice. That is the tax.",
)

TERM_STRUCTURE = Family(
    name="Volatility term structure / calendars",
    thesis="Trade the slope of the vol curve: sell rich front-month against "
           "cheap back-month, or the reverse in backwardation.",
    n_underlyings=150,
    residual_corr=0.30,      # curve shape is a strongly common factor
    cycles_per_year=12,
    entry_cost_frac=0.025,   # two expiries, two spreads
    exit_cost_frac=0.015,
    bet_sigma_frac=0.40,
    plausible_ic=0.05,
    edge_type="risk_premium",
    skew="negative",
    tail_loss_frac=5.0,
    tail_every_years=6.0,
    data_needed="EOD chains, multiple expiries per name",
    data_cost_usd_month=60.0,
    free_history_years=2.0,
    min_capital_usd=15_000,
    published_evidence_years=15.0,
    transfer_coefficient=0.6,
        deep_history_years=19.0,
    deep_history_cost_usd=600.0,
    notes="Structurally similar to cross-sectional VRP but with less breadth.",
)

FLOW_EVENT_DRIVEN = Family(
    name="Informed-flow / event-driven (M&A)",
    thesis="Detect unusual pre-announcement call buying and buy the same "
           "lottery tickets.",
    n_underlyings=100,       # bounded by deals per year, not by universe size
    residual_corr=0.0,       # deals are genuinely independent of each other
    cycles_per_year=1,
    entry_cost_frac=0.060,   # short-dated OTM markets are 8%+ wide
    exit_cost_frac=0.045,
    bet_sigma_frac=2.50,     # mostly -100%, occasionally +10x
    plausible_ic=0.14,
    edge_type="mispricing",
    skew="positive",         # the one family whose tail helps you
    tail_loss_frac=1.0,
    tail_every_years=0.0,
    data_needed="Trade-level tape WITH quotes, to compute ask-side share",
    data_cost_usd_month=250.0,
    free_history_years=0.0,  # no free historical options tape exists
    min_capital_usd=5_000,
    published_evidence_years=0.0,    # academic interest, no tradeable track record
    transfer_coefficient=0.5,
        deep_history_years=0.0,    # tick tape with quotes is not sold as deep history
    deep_history_cost_usd=0.0,
    notes="Highest per-bet edge, lowest breadth, priciest data.",
)

ZERO_DTE = Family(
    name="0DTE / intraday index options",
    thesis="Harvest same-day decay or trade intraday gamma.",
    n_underlyings=3,
    residual_corr=0.95,
    cycles_per_year=252,
    entry_cost_frac=0.020,   # small premiums make the spread proportionally huge
    exit_cost_frac=0.020,
    bet_sigma_frac=0.55,
    plausible_ic=0.05,
    edge_type="mispricing",
    skew="negative",
    tail_loss_frac=10.0,
    tail_every_years=2.0,
    data_needed="Intraday tick chains - expensive and large",
    data_cost_usd_month=150.0,
    free_history_years=1.0,
    min_capital_usd=25_000,
    published_evidence_years=2.0,
    transfer_coefficient=0.4,        # fills are the whole game and you lose them
        deep_history_years=4.0,    # intraday chain history is thin and costly
    deep_history_cost_usd=3000.0,
    notes="Competing directly with market makers on their home turf.",
)

COVERED_CALL_WHEEL = Family(
    name="Covered calls / the wheel",
    thesis="Sell calls against stock, sell puts to acquire it.",
    n_underlyings=200,
    residual_corr=0.60,      # it is mostly long equity beta in a costume
    cycles_per_year=12,
    entry_cost_frac=0.012,
    exit_cost_frac=0.004,
    bet_sigma_frac=0.45,
    plausible_ic=0.02,       # very little alpha once beta is stripped out
    edge_type="risk_premium",
    skew="negative",
    tail_loss_frac=5.0,
    tail_every_years=8.0,
    data_needed="EOD chains",
    data_cost_usd_month=60.0,
    free_history_years=2.0,
    min_capital_usd=25_000,
    published_evidence_years=20.0,
    transfer_coefficient=0.8,
        deep_history_years=19.0,
    deep_history_cost_usd=600.0,
    notes="Popular because it feels safe. Mostly a levered long position.",
)


# ---------------------------------------------------------------------------
# The synthesis.
#
# None of the seven naive families above survives its own screen, and the two
# failure modes are complementary rather than shared:
#
#   - Index short vol has by far the strongest edge (a 25-year-documented risk
#     premium) and no survivability: 15 sigma of tail loss ends the account.
#   - Cross-sectional VRP is survivable and broad, but vega-neutralising it
#     hedges away the premium itself, leaving a forecast signal too weak to
#     ever prove.
#
# Which suggests taking the premium (not hedging it away), capping the tail
# structurally (bought wings, not position sizing), and spreading it over many
# underlyings for whatever breadth is available. That is a different strategy
# from any of the seven, so it is scored as its own candidate rather than
# assumed to inherit their properties.
# ---------------------------------------------------------------------------

DEFINED_RISK_VRP = Family(
    name="Defined-risk VRP harvest (index + single names)",
    thesis="Stay net short volatility to collect the premium, cap every "
           "position's loss with bought wings, spread across ~60 liquid "
           "underlyings, and tilt size toward the richest vol.",
    n_underlyings=60,
    residual_corr=0.50,      # net short vol, so positions co-move by design
    cycles_per_year=12,
    entry_cost_frac=0.020,   # four legs instead of two - wings are not free
    exit_cost_frac=0.005,    # held to expiry except for profit-taking
    bet_sigma_frac=0.50,
    plausible_ic=0.18,       # premium, minus what the wings cost to buy
    edge_type="risk_premium",
    published_evidence_years=25.0,
    transfer_coefficient=0.7,
    skew="negative",
    tail_loss_frac=4.0,      # structurally capped: max loss = width - credit
    tail_every_years=6.0,
    data_needed="EOD option chains with IV + underlying history",
    data_cost_usd_month=60.0,
    free_history_years=2.0,
    min_capital_usd=15_000,
        deep_history_years=19.0,   # EOD chains back to ~2007: covers 2008, 2018, 2020, 2024
    deep_history_cost_usd=600.0,
    notes="Buying the wings costs roughly a quarter of the edge and removes "
          "the failure mode that ends the project. That is the trade.",
)


ALL_FAMILIES = [
    DEFINED_RISK_VRP,
    CROSS_SECTIONAL_VRP,
    INDEX_SHORT_VOL,
    EARNINGS_VOL_CRUSH,
    TERM_STRUCTURE,
    FLOW_EVENT_DRIVEN,
    ZERO_DTE,
    COVERED_CALL_WHEEL,
]


def comparison_table(families: list[Family] | None = None) -> pd.DataFrame:
    """The whole decision on one page."""
    families = families or ALL_FAMILIES
    rows = []
    for f in families:
        yrs = f.years_to_prove()
        rows.append({
            "family": f.name,
            "bets/yr": round(f.effective_breadth()),
            "cost/noise": round(f.cost_to_noise(), 3),
            "IC": f.plausible_ic,
            "net_IR": round(f.annual_ir(), 2),
            "yrs_to_prove": "never" if np.isinf(yrs) else round(yrs, 1),
            "edge": "durable" if f.edge_type == "risk_premium" else "decays",
            "skew": f.skew,
            "tail_blind_spot": round(f.tail_blind_spot(), 1),
            "survivable": "yes" if f.survivable() else "NO",
            "data_$/mo": f.data_cost_usd_month,
        })
    return (pd.DataFrame(rows)
            .sort_values("net_IR", ascending=False)
            .reset_index(drop=True))


def screen(families: list[Family] | None = None) -> pd.DataFrame:
    """
    Apply the disqualifiers before ranking anything.

    Three of these are hard. A strategy that cannot be proven within available
    history is not a strategy, it is a belief. A strategy whose bad regime is
    absent from its own backtest will be discovered the expensive way. And a
    strategy that one regime can end does not get to compound.
    """
    families = families or ALL_FAMILIES
    rows = []
    for f in families:
        yrs = f.years_to_prove()
        needed = yrs * 2  # tune half, test half
        fails = []
        if np.isinf(yrs):
            fails.append("no net edge after costs")
        elif f.proof_burden() == "discover":
            # Must be established from your own data, so history depth binds.
            if f.free_history_years <= 0:
                fails.append("no free history to test against")
            elif needed > f.free_history_years * 3:
                fails.append(f"needs ~{needed:.0f}y history, has "
                             f"{f.free_history_years:.0f}y")
        # A documented risk premium is exempt from the provability screen -- but
        # only that one. It still has to survive its own tail, and the tail is
        # where short-vol strategies actually die.
        if not f.survivable():
            fails.append(f"tail loss {f.tail_loss_frac:.0f}x sigma - one regime ends it")
        if f.tail_blind_spot() > 2.0:
            fails.append(f"backtest cannot see its own tail ({f.tail_blind_spot():.1f}x)")
        rows.append({
            "family": f.name,
            "verdict": "PASS" if not fails else "FAIL",
            "why": "; ".join(fails) if fails else "clears every disqualifier",
        })
    return pd.DataFrame(rows)
