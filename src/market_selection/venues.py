"""
Venue parameter table: what it actually costs, and what you are allowed to do,
in each candidate market as of August 2026.

Every number here is an *input assumption*, not a result. They are sourced from
public fee schedules and market-structure reporting (see DECISION.md for the
citation list) and deliberately kept in one place so the whole conclusion can be
re-run against different beliefs. `sensitivity.py` exists because several of
these are uncertain enough to deserve a stress test rather than a footnote.

Cost convention: all costs are basis points of *traded notional*, one-way,
unless the field name says otherwise. Options are the exception and are
expressed in bps of *premium*, since premium is the capital at risk.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Venue:
    """One tradeable market, described in the terms that decide feasibility."""

    name: str
    style: str                    # 'cross_sectional' or 'directional'
    notes: str = ""

    # --- execution cost, one-way, bps of notional -------------------------
    commission_bps: float = 0.0   # exchange/broker fee
    half_spread_bps: float = 0.0  # cost of crossing to the touch
    slippage_bps: float = 0.0     # impact + adverse selection at target clip
    fixed_cost_usd: float = 0.0   # per-transaction dollar cost (gas)
    regulatory_fee_bps: float = 0.0

    # --- carry, bps of notional per calendar day held ---------------------
    carry_bps_per_day: float = 0.0   # funding, borrow, theta

    # --- risk / opportunity ----------------------------------------------
    daily_vol_bps: float = 100.0     # typical asset's daily return vol
    bet_sigma_override_bps: float | None = None  # for non-sqrt-scaling payoffs
    n_assets: int = 1                # tradeable names with adequate liquidity
    avg_pairwise_corr: float = 0.3   # raw cross-sectional co-movement
    residual_corr: float = 0.05      # co-movement AFTER hedging the common factor
    sessions_per_year: int = 252     # rebalance opportunities per year
    max_bets_per_year: float | None = None  # hard cap (e.g. event-driven)
    plausible_ic: float = 0.05       # per-bet Sharpe a good signal here can reach

    # --- practical constraints -------------------------------------------
    typical_clip_usd: float = 5_000.0
    min_capital_usd: float = 2_000.0
    data_cost_usd_per_month: float = 0.0
    history_years_free: float = 0.0  # free backtestable history depth

    # --- hard gates: True means "this is a real blocker" ------------------
    gate_legal_us_retail: bool = False     # cannot legally/practically access
    gate_data_unavailable: bool = False    # cannot get research data at all
    gate_infra_incompatible: bool = False  # cannot run on Colab/Render
    gate_reasons: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def round_trip_cost_bps(self, clip_usd: float | None = None) -> float:
        """Total cost to enter and exit one position, excluding carry."""
        clip = clip_usd or self.typical_clip_usd
        variable = self.commission_bps + self.half_spread_bps + self.slippage_bps
        fixed = (self.fixed_cost_usd / clip) * 10_000 if clip else 0.0
        return 2 * (variable + fixed) + self.regulatory_fee_bps

    def holding_cost_bps(self, holding_days: float,
                         clip_usd: float | None = None) -> float:
        """Round-trip cost plus carry over the holding period."""
        return self.round_trip_cost_bps(clip_usd) + self.carry_bps_per_day * holding_days

    def bet_sigma_bps(self, holding_days: float) -> float:
        """Return volatility of one position over its holding period."""
        if self.bet_sigma_override_bps is not None:
            return self.bet_sigma_override_bps
        return self.daily_vol_bps * (holding_days ** 0.5)

    def cost_to_noise(self, holding_days: float,
                      clip_usd: float | None = None) -> float:
        """
        The single most decision-relevant ratio: friction as a fraction of the
        opportunity's own standard deviation.

        This is what makes naive fee comparisons misleading. A 40bp fee in a
        market with 4% daily vol is *proportionally cheaper* than a 2bp fee in a
        market with 0.15% daily vol. Any forecast must beat this ratio in
        information-coefficient terms before it earns a cent.
        """
        return self.holding_cost_bps(holding_days, clip_usd) / self.bet_sigma_bps(holding_days)

    def effective_breadth(self, holding_days: float) -> float:
        """
        Independent bets available per year.

        Naive Grinold -- breadth = names x rebalances -- is wildly optimistic:
        it hands a 1500-name daily equity book ~378,000 bets/year and predicts
        information ratios above 20, which no equity market-neutral fund has
        ever produced. The correction is that positions co-move, so N names buy
        far fewer than N independent bets:

            N_eff = N / (1 + (N-1) * corr)

        Which correlation to use depends on what is being forecast:
          - directional/timing: the common factor IS the bet, so raw pairwise
            correlation applies and a small universe collapses to ~1/rho.
          - cross-sectional (dollar-neutral): the common factor is hedged out,
            so the much smaller *residual* correlation applies. Residuals are
            close to independent in equities and markedly less so in crypto,
            where sector rotation and 'alt season' move the whole tail together.
        """
        n = self.n_assets
        rho = self.avg_pairwise_corr if self.style == "directional" else self.residual_corr
        n_eff = n / (1 + (n - 1) * rho) if n > 1 else 1.0
        rebalances = self.sessions_per_year / max(holding_days, 1e-9)
        breadth = n_eff * rebalances
        if self.max_bets_per_year is not None:
            breadth = min(breadth, self.max_bets_per_year)
        return breadth

    def is_gated(self) -> bool:
        return (self.gate_legal_us_retail or self.gate_data_unavailable
                or self.gate_infra_incompatible)


# How much of a paper signal actually reaches the portfolio, after position
# limits, lot sizes, capital constraints, and the trades you skip because they
# are not worth the friction. Grinold & Kahn's transfer coefficient; empirically
# 0.3-0.6 for constrained real-money books. Applied to every venue equally, so
# it shifts all results but changes no ranking -- it is here to keep the
# absolute IR numbers honest rather than to tilt the comparison.
TRANSFER_COEFFICIENT = 0.5


# ----------------------------------------------------------------------------
# The candidate set.
#
# "Crypto DeFi vs traditional stocks" is not actually a two-way choice, and
# collapsing it to one hides the decision. Splitting it five ways is the point:
# the interesting variation is *within* each camp, not between them.
# ----------------------------------------------------------------------------

US_EQUITIES = Venue(
    name="US cash equities (liquid universe)",
    style="cross_sectional",
    notes="Alpaca/IBKR, commission-free. ~1500 names above $5M ADV.",
    commission_bps=0.0,
    half_spread_bps=2.0,       # blended large+mid cap; large caps are ~0.5bp
    slippage_bps=0.5,
    regulatory_fee_bps=0.05,   # SEC 31 fee + TAF, sell side only
    carry_bps_per_day=0.10,    # short borrow on the short leg
    daily_vol_bps=140.0,       # ~22% annualized
    n_assets=1500,
    avg_pairwise_corr=0.30,
    residual_corr=0.02,      # post-factor equity residuals are nearly independent
    sessions_per_year=252,
    plausible_ic=0.04,       # published cross-sectional anomalies live at 0.02-0.05
    typical_clip_usd=5_000,
    min_capital_usd=2_000,     # PDT $25k floor removed June 2026
    data_cost_usd_per_month=0.0,
    history_years_free=10.0,   # Alpaca: 10y of 1-minute bars, free
)

US_EQUITY_OPTIONS = Venue(
    name="US equity options (short-dated, event-driven)",
    style="cross_sectional",
    notes="The current repo's thesis. Costs are bps of PREMIUM, not notional.",
    commission_bps=30.0,       # ~$0.35/contract on a ~$120 premium contract
    half_spread_bps=400.0,     # 8%-wide markets on short-dated OTM are normal
    slippage_bps=100.0,
    carry_bps_per_day=0.0,     # theta is inside the payoff distribution below
    bet_sigma_override_bps=25_000.0,  # ~250% sd: mostly -100%, rarely +10x
    n_assets=1500,
    avg_pairwise_corr=0.30,
    residual_corr=0.02,
    sessions_per_year=252,
    max_bets_per_year=100,
    # Implied by CLAUDE.md's own established numbers: ~8% precision with ~20x
    # payoffs on hits and -85% on misses gives E[r]=+0.74 with sd 5.4 per bet.
    plausible_ic=0.14,     # binding constraint: filtered M&A alerts per year
    typical_clip_usd=1_000,
    min_capital_usd=5_000,
    data_cost_usd_per_month=250.0,  # flow feed (Unusual Whales/Bullflow tier)
    history_years_free=0.0,         # no free historical options flow, at all
    gate_reasons=["Historical options flow + NBBO is paid-only; no free tier "
                  "supports a walk-forward backtest."],
)

CRYPTO_SPOT_CEX = Venue(
    name="Crypto spot on US CEX (Kraken Pro / Coinbase Advanced)",
    style="cross_sectional",
    notes="Retail base fee tier. This is where most 'crypto algo' attempts start.",
    commission_bps=34.0,       # Kraken base 25bp maker / 40bp taker, blended
    half_spread_bps=8.0,       # majors ~1bp, alts 10-30bp
    slippage_bps=5.0,
    carry_bps_per_day=0.0,
    daily_vol_bps=400.0,       # ~76% annualized, blended majors + alts
    n_assets=120,
    avg_pairwise_corr=0.70,    # everything is a levered bet on BTC
    residual_corr=0.15,        # residuals still co-move: sector rotation, 'alt season'
    sessions_per_year=365,
    plausible_ic=0.05,         # less efficient market, but noisier signals too
    typical_clip_usd=5_000,
    min_capital_usd=500,
    data_cost_usd_per_month=0.0,
    history_years_free=9.0,    # CryptoDataDownload/Kraken CSV archives
)

CRYPTO_PERPS_US = Venue(
    name="Crypto perps, CFTC-regulated US (Kraken/Bitnomial, Coinbase FM)",
    style="directional",
    notes="Launched onshore in 2026. Cheap and legal, but a tiny universe.",
    commission_bps=6.0,        # LEAST CERTAIN INPUT - see sensitivity.py
    half_spread_bps=2.0,
    slippage_bps=2.0,
    carry_bps_per_day=1.5,     # funding, one-directional exposure
    daily_vol_bps=300.0,       # majors only, ~57% annualized
    n_assets=8,                # BTC, ETH, SOL, XRP + a short tail
    avg_pairwise_corr=0.80,
    residual_corr=0.30,
    sessions_per_year=365,
    plausible_ic=0.04,         # majors are the most heavily quantified crypto assets
    typical_clip_usd=5_000,
    min_capital_usd=1_000,
    data_cost_usd_per_month=0.0,
    history_years_free=6.0,    # offshore perp history is a usable proxy
)

DEFI_ONCHAIN_L2 = Venue(
    name="On-chain DeFi (Uniswap/Aerodrome on Base/Arbitrum)",
    style="cross_sectional",
    notes="Post-EIP-4844 gas is genuinely negligible; the pool fee is not.",
    commission_bps=25.0,       # blended AMM pool fee (5/30/100bp tiers)
    half_spread_bps=20.0,      # price impact on a $5k clip in a mid-depth pool
    slippage_bps=10.0,         # residual MEV/adverse selection w/ private RPC
    fixed_cost_usd=0.03,       # L2 gas per swap - 0.06bp on a $5k clip
    carry_bps_per_day=0.0,
    daily_vol_bps=600.0,       # ~115% annualized, long-tail tokens
    n_assets=200,              # tokens with >$500k pool depth on a major L2
    avg_pairwise_corr=0.65,
    residual_corr=0.12,
    sessions_per_year=365,
    plausible_ic=0.06,         # least efficient venue, but MEV taxes the fast signals
    typical_clip_usd=5_000,
    min_capital_usd=500,
    data_cost_usd_per_month=0.0,
    history_years_free=5.0,    # Dune/DefiLlama/subgraph, free and complete
    gate_reasons=["US market-structure law unresolved (CLARITY Act ~30% odds "
                  "in 2026); no 1099-DA basis reporting, so tax accounting is "
                  "self-serve across every swap."],
)

DEFI_PERPS_OFFSHORE = Venue(
    name="Offshore DeFi perps (Hyperliquid)",
    style="directional",
    notes="Cheapest venue on paper. Also the one a US person may not use.",
    commission_bps=4.5,        # 0.045% taker
    half_spread_bps=1.0,
    slippage_bps=1.5,
    carry_bps_per_day=1.5,
    daily_vol_bps=350.0,
    n_assets=150,              # deep perp universe, the real DeFi advantage
    avg_pairwise_corr=0.75,
    residual_corr=0.20,
    sessions_per_year=365,
    plausible_ic=0.05,
    typical_clip_usd=5_000,
    min_capital_usd=500,
    data_cost_usd_per_month=0.0,
    history_years_free=4.0,
    gate_legal_us_retail=True,
    gate_reasons=["US persons are Restricted Persons under Hyperliquid's terms; "
                  "geofenced, and the terms forbid VPN workarounds. No CFTC/SEC "
                  "registration to offer leveraged perps to US persons."],
)


ALL_VENUES = [
    US_EQUITIES,
    US_EQUITY_OPTIONS,
    CRYPTO_SPOT_CEX,
    CRYPTO_PERPS_US,
    DEFI_ONCHAIN_L2,
    DEFI_PERPS_OFFSHORE,
]

CAMP = {
    US_EQUITIES.name: "traditional",
    US_EQUITY_OPTIONS.name: "traditional",
    CRYPTO_SPOT_CEX.name: "crypto",
    CRYPTO_PERPS_US.name: "crypto",
    DEFI_ONCHAIN_L2.name: "crypto-defi",
    DEFI_PERPS_OFFSHORE.name: "crypto-defi",
}
