"""
Where candidate strategies come from.

Not a signal generator. Ideas are organised by the *economic reason someone
loses money to you*, because that is the only property that predicts whether an
edge survives contact with the future. A pattern found by searching data has no
such property and is indistinguishable from noise at the moment of discovery.

Four sources, in descending order of how well they survive:

  1. RISK PREMIUM      someone pays you to carry a risk they do not want.
                       Durable, because the discomfort is permanent. Capacity
                       is large. This is where a solo researcher should live.
  2. CONSTRAINT        someone must trade regardless of price -- a mandate, an
                       index rule, a margin call, a tax year-end. Durable while
                       the rule exists, and the rule is usually public.
  3. BEHAVIOURAL       someone predictably errs. Real, but decays as it becomes
                       known, and the good ones are already known.
  4. INFORMATION       someone knows first. Not accessible to retail, and the
                       version that is accessible is usually illegal or already
                       priced. Listed for completeness and to be avoided.

An idea with no identified counterparty is not on this list, which is the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .registry import Hypothesis

SOURCES = ("risk_premium", "constraint", "behavioural", "information")


@dataclass
class Idea:
    """A candidate before it earns the right to be a Hypothesis."""
    title: str
    source: str
    who_pays: str
    why_they_persist: str
    retail_accessible: bool
    data_needed: str
    est_capacity: str
    decay_risk: str          # 'low' | 'medium' | 'high'
    notes: str = ""

    def score(self) -> float:
        """
        Crude triage, used only to order a queue -- never to decide anything.

        Weighted toward durability and accessibility rather than expected
        return, because expected return at the idea stage is imagination.
        """
        s = {"risk_premium": 4.0, "constraint": 3.0,
             "behavioural": 1.5, "information": 0.5}[self.source]
        s += {"low": 2.0, "medium": 1.0, "high": 0.0}[self.decay_risk]
        s += 2.0 if self.retail_accessible else -3.0
        return s


# ---------------------------------------------------------------------------
# The starting catalogue.
#
# All of these are publicly documented. That is deliberate and worth stating
# plainly: an LLM trained on public text is a poor source of proprietary alpha,
# and pretending otherwise would be the most expensive kind of self-flattery.
# What this catalogue is for is making sure the *obvious* ground is covered
# systematically and tested honestly, before anyone goes hunting for the exotic.
# The edge, if any appears, will come from implementation quality and from
# the discipline of the validation battery -- not from the idea list.
# ---------------------------------------------------------------------------

CATALOGUE = [
    Idea("Variance risk premium (index and single-name)", "risk_premium",
         who_pays="Investors and funds buying downside protection, plus "
                  "structured-product desks hedging their own books.",
         why_they_persist="Insurance demand is structural and price-insensitive; "
                          "mandates require hedging regardless of whether it is "
                          "expensive this month.",
         retail_accessible=True,
         data_needed="EOD option chains with IV, 15+ years",
         est_capacity="large", decay_risk="low",
         notes="Currently implemented. See STRATEGY.md."),

    Idea("Index rebalance front-running (add/delete effect)", "constraint",
         who_pays="Index-tracking funds that must buy at the close on a "
                  "published date, at whatever price is there.",
         why_they_persist="Tracking-error mandates make them price-insensitive; "
                          "the rule is public and they cannot deviate from it.",
         retail_accessible=True,
         data_needed="Index change announcements + daily prices; both free",
         est_capacity="small", decay_risk="medium",
         notes="Heavily studied and partly arbitraged, but the constraint is real "
               "and cannot be removed without changing the mandate."),

    Idea("Overnight vs intraday return separation", "risk_premium",
         who_pays="Investors unwilling to hold gap risk through the close.",
         why_they_persist="Overnight gap risk is unhedgeable for many mandates, "
                          "so compensation for bearing it persists.",
         retail_accessible=True,
         data_needed="Daily OHLC, free from Alpaca",
         est_capacity="medium", decay_risk="medium"),

    Idea("Post-earnings-announcement drift", "behavioural",
         who_pays="Investors who under-react to earnings surprises and adjust "
                  "their positions gradually over subsequent weeks.",
         why_they_persist="Attention is limited and institutional rebalancing is "
                          "slow; the effect has weakened but not vanished.",
         retail_accessible=True,
         data_needed="Earnings dates + surprise + daily prices",
         est_capacity="medium", decay_risk="high",
         notes="Documented since 1968 and decaying steadily. Test the decay "
               "explicitly rather than the average effect."),

    Idea("Tax-loss selling / January reversal", "constraint",
         who_pays="Taxable investors dumping losers before year-end for reasons "
                  "that have nothing to do with value.",
         why_they_persist="The tax code sets the deadline; it is not a belief "
                          "anyone can be argued out of.",
         retail_accessible=True,
         data_needed="Daily prices + a calendar",
         est_capacity="small", decay_risk="medium"),

    Idea("Volatility term-structure carry", "risk_premium",
         who_pays="Hedgers buying longer-dated protection at a premium to "
                  "short-dated.",
         why_they_persist="Same insurance demand as VRP, expressed along the "
                          "maturity axis instead of the strike axis.",
         retail_accessible=True,
         data_needed="Option chains across multiple expiries",
         est_capacity="medium", decay_risk="low"),

    Idea("Pre-announcement informed options flow (M&A)", "information",
         who_pays="Nobody predictable -- you are trying to detect someone else's "
                  "information, not to be compensated for a risk.",
         why_they_persist="Leaks recur, but detection is drowned by look-alike "
                          "rumour flow and hidden spread legs.",
         retail_accessible=False,
         data_needed="Trade-level tape WITH quotes, $250/mo, no free history",
         est_capacity="small", decay_risk="high",
         notes="The project's original thesis. Archived: highest per-bet edge, "
               "lowest breadth, priciest data, and no counterparty who is "
               "structurally willing to keep losing."),
]


def queue(catalogue: list[Idea] | None = None) -> pd.DataFrame:
    """Triage order. Ordering only -- the gauntlet does the deciding."""
    items = catalogue or CATALOGUE
    return pd.DataFrame([{
        "idea": i.title, "source": i.source, "score": i.score(),
        "retail": "yes" if i.retail_accessible else "NO",
        "decay_risk": i.decay_risk, "capacity": i.est_capacity,
        "data": i.data_needed,
    } for i in items]).sort_values("score", ascending=False).reset_index(drop=True)


def to_hypothesis(idea: Idea, hid: str, prediction: str,
                  success_criteria: dict, kill_criteria: str) -> Hypothesis:
    """
    Promote an idea to a registered hypothesis.

    The three arguments that are NOT carried over from the Idea -- prediction,
    success criteria and kill criteria -- must be written by hand, on purpose.
    They are the commitments, and auto-generating them would defeat the point
    of pre-registration entirely.
    """
    return Hypothesis(
        id=hid, title=idea.title, source=idea.source,
        thesis=f"{idea.title}. {idea.notes}".strip(),
        who_pays=idea.who_pays, why_they_persist=idea.why_they_persist,
        prediction=prediction, success_criteria=success_criteria,
        kill_criteria=kill_criteria, data_required=idea.data_needed,
        capacity_note=idea.est_capacity, decay_risk=idea.decay_risk,
    )
