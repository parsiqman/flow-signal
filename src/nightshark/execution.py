"""Turn UP/DOWN into an order.

The translation is the part that is easy to get wrong. Kalshi does not list an
"UP" contract. It lists strike markets -- "BTC above $84,250 at 16:15" -- that
pay $1.00 if true and $0 if not. So:

    UP   -> buy YES on the strike nearest spot
    DOWN -> buy NO  on the same strike

Picking the NEAREST strike is what makes this a direction bet. A strike far
above spot is a bet on magnitude, not direction, and a correct UP call still
loses. `select_contract` enforces that.

Two brokers share one interface. `PaperBroker` needs no credentials and no
network, and prices contracts from a binary-option model plus a spread and the
real fee formula -- so paper PnL includes the costs that decide whether a
50%-accurate bot makes or loses money. `KalshiBroker` talks to the exchange.

WARNING, and it is not a small one: the live endpoint shapes, the auth signing
detail and the series ticker below were written from documentation, NOT verified
against the API -- this sandbox's egress policy blocks api.elections.kalshi.com.
Run `--list-markets` against your own key and reconcile before trading real money.
"""

from __future__ import annotations

import base64
import json
import math
import time
import urllib.error
import urllib.request
import uuid
from urllib.parse import urlparse
from dataclasses import dataclass, field
from datetime import datetime, timezone

YES, NO = "yes", "no"


def fee_cents(price_cents: float, count: int, rate: float = 0.07) -> int:
    """Kalshi trading fee: ceil(rate * C * P * (1-P)), P in dollars, result in cents.

    Maximal at 50c -- exactly where a direction bet on the nearest strike trades.
    Confirm `rate` against the current fee schedule; it varies by market.
    """
    p = price_cents / 100.0
    return int(math.ceil(rate * count * p * (1 - p) * 100))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def binary_fair_value(spot: float, strike: float, minutes_left: float,
                      annual_vol_pct: float) -> float:
    """P(spot_at_expiry > strike), as a probability in [0, 1].

    Driftless lognormal -- over 15 minutes the drift term is noise next to the
    diffusion term, and pretending to know it would be the same self-deception
    this bot is meant to test for.
    """
    if minutes_left <= 0:
        return 1.0 if spot > strike else 0.0
    sigma = max(annual_vol_pct, 1e-6) / 100.0
    t = minutes_left / (365 * 24 * 60)
    denom = sigma * math.sqrt(t)
    if denom <= 0:
        return 1.0 if spot > strike else 0.0
    d2 = (math.log(spot / strike) - 0.5 * sigma ** 2 * t) / denom
    return _norm_cdf(d2)


@dataclass
class Contract:
    ticker: str
    strike: float
    close_time: datetime
    yes_ask: int = 0          # cents
    no_ask: int = 0
    yes_bid: int = 0
    no_bid: int = 0

    def ask_for(self, side: str) -> int:
        return self.yes_ask if side == YES else self.no_ask


@dataclass
class Fill:
    ticker: str
    side: str
    count: int
    price_cents: int
    fee_cents: int
    order_id: str
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    strike: float = 0.0
    close_time: datetime | None = None

    @property
    def cost(self) -> float:
        """Dollars out the door, fees included."""
        return (self.count * self.price_cents + self.fee_cents) / 100.0

    def settle(self, settlement_price: float) -> float:
        """Realized PnL in dollars given the underlying's settlement price."""
        yes_won = settlement_price > self.strike
        won = yes_won if self.side == YES else not yes_won
        return (self.count * 100) / 100.0 - self.cost if won else -self.cost

    def to_dict(self) -> dict:
        return {"at": self.at.isoformat(), "ticker": self.ticker, "side": self.side,
                "count": self.count, "price_cents": self.price_cents,
                "fee_cents": self.fee_cents, "strike": self.strike,
                "cost": round(self.cost, 2), "order_id": self.order_id}


def side_for(direction: str) -> str:
    """UP -> YES, DOWN -> NO, on a strike near spot."""
    if direction == "UP":
        return YES
    if direction == "DOWN":
        return NO
    raise ValueError(f"not a tradeable direction: {direction!r}")


def select_contract(contracts, spot: float, window_end: datetime | None = None):
    """The nearest strike in the target window -- the closest thing to a pure
    direction bet the exchange lists. Returns None if nothing qualifies."""
    pool = list(contracts)
    if window_end is not None:
        matched = [c for c in pool if abs((c.close_time - window_end).total_seconds()) <= 60]
        pool = matched or []
    if not pool:
        return None
    return min(pool, key=lambda c: abs(c.strike - spot))


class PaperBroker:
    """Simulated fills. No key, no network, costs included."""

    def __init__(self, spread_cents: int = 2, strike_step: float = 250.0,
                 fee_rate: float = 0.07):
        self.spread_cents = spread_cents
        self.strike_step = strike_step
        self.fee_rate = fee_rate
        self.orders: list[Fill] = []

    def list_contracts(self, series_ticker: str = "", window_end: datetime | None = None,
                       spot: float = 0.0, minutes_left: float = 0.0,
                       annual_vol_pct: float = 60.0, n_strikes: int = 5):
        """Synthesize a strike ladder around spot, priced off the binary model.

        Shares KalshiBroker's signature so the runner never branches on broker type.
        """
        window_end = window_end or datetime.now(timezone.utc)
        base = round(spot / self.strike_step) * self.strike_step
        out = []
        for k in range(-(n_strikes // 2), n_strikes // 2 + 1):
            strike = base + k * self.strike_step
            p = binary_fair_value(spot, strike, minutes_left, annual_vol_pct)
            mid = p * 100
            half = self.spread_cents / 2
            yes_ask = int(min(99, max(1, round(mid + half))))
            yes_bid = int(min(99, max(1, round(mid - half))))
            out.append(Contract(
                ticker=f"PAPER-BTC-{window_end:%y%b%d%H%M}-T{int(strike)}".upper(),
                strike=float(strike), close_time=window_end,
                yes_ask=yes_ask, yes_bid=yes_bid,
                no_ask=100 - yes_bid, no_bid=100 - yes_ask,
            ))
        return out

    def buy(self, contract: Contract, side: str, count: int, max_price_cents: int):
        price = contract.ask_for(side)
        if price > max_price_cents:
            return None, f"ask {price}c above cap {max_price_cents}c"
        fill = Fill(ticker=contract.ticker, side=side, count=count, price_cents=price,
                    fee_cents=fee_cents(price, count, self.fee_rate),
                    order_id=f"paper-{uuid.uuid4().hex[:12]}",
                    strike=contract.strike, close_time=contract.close_time)
        self.orders.append(fill)
        return fill, ""


class KalshiBroker:
    """Live Kalshi REST client. Signs every request with the account's RSA key.

    Kalshi authenticates API keys by RSA-PSS over `timestamp + METHOD + path`.
    Requires `cryptography`.
    """

    def __init__(self, base_url: str, key_id: str, private_key_path: str,
                 timeout: float = 15.0, fee_rate: float = 0.07):
        if not key_id or not private_key_path:
            raise ValueError(
                "live trading needs KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH")
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        self._hashes, self._padding = hashes, padding
        self.base_url = base_url.rstrip("/")
        self.key_id = key_id
        self.timeout = timeout
        self.fee_rate = fee_rate
        with open(private_key_path, "rb") as f:
            self.private_key = serialization.load_pem_private_key(f.read(), password=None)

    def _headers(self, method: str, path: str) -> dict:
        ts = str(int(time.time() * 1000))
        # Sign the path WITHOUT the query string.
        msg = (ts + method.upper() + path).encode()
        sig = self.private_key.sign(
            msg,
            self._padding.PSS(
                mgf=self._padding.MGF1(self._hashes.SHA256()),
                salt_length=self._padding.PSS.DIGEST_LENGTH,
            ),
            self._hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(self, method: str, path: str, query: str = "", body: dict | None = None):
        url = f"{self.base_url}{path}" + (f"?{query}" if query else "")
        data = json.dumps(body).encode() if body is not None else None
        # The signed path is the full path as it appears in the URL (API prefix
        # included, query string excluded).
        signed_path = urlparse(url).path
        req = urllib.request.Request(url, data=data, method=method.upper(),
                                     headers=self._headers(method, signed_path))
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:400]
            raise RuntimeError(f"kalshi {method} {path} -> {e.code}: {detail}") from e

    def list_contracts(self, series_ticker: str = "", window_end: datetime | None = None,
                       spot: float = 0.0, minutes_left: float = 0.0,
                       annual_vol_pct: float = 0.0, limit: int = 200):
        """Open markets in the series. `spot`/`minutes_left`/`annual_vol_pct` are
        accepted and ignored -- the real exchange quotes its own prices."""
        q = f"series_ticker={series_ticker}&status=open&limit={limit}"
        payload = self._request("GET", "/markets", q)
        out = []
        for m in payload.get("markets", []):
            strike = m.get("floor_strike", m.get("strike_price", m.get("cap_strike")))
            if strike is None:
                continue
            out.append(Contract(
                ticker=m["ticker"], strike=float(strike),
                close_time=datetime.fromisoformat(
                    m["close_time"].replace("Z", "+00:00")),
                yes_ask=int(m.get("yes_ask") or 0), yes_bid=int(m.get("yes_bid") or 0),
                no_ask=int(m.get("no_ask") or 0), no_bid=int(m.get("no_bid") or 0),
            ))
        return out

    def buy(self, contract: Contract, side: str, count: int, max_price_cents: int):
        price = contract.ask_for(side)
        if not price or price > max_price_cents:
            return None, f"ask {price}c above cap {max_price_cents}c"
        body = {
            "action": "buy", "side": side, "ticker": contract.ticker,
            "count": count, "type": "limit",
            "client_order_id": str(uuid.uuid4()),
            # Marketable limit at the ask: crosses the spread but never pays more.
            ("yes_price" if side == YES else "no_price"): price,
        }
        resp = self._request("POST", "/portfolio/orders", body=body)
        order = resp.get("order", {})
        filled = int(order.get("count", count))
        fill = Fill(ticker=contract.ticker, side=side, count=filled, price_cents=price,
                    fee_cents=fee_cents(price, filled, self.fee_rate),
                    order_id=order.get("order_id", "unknown"),
                    strike=contract.strike, close_time=contract.close_time)
        return fill, ""

    def balance(self) -> float:
        return self._request("GET", "/portfolio/balance").get("balance", 0) / 100.0


def build_broker(cfg):
    if cfg.dry_run:
        return PaperBroker()
    return KalshiBroker(cfg.kalshi_base_url, cfg.kalshi_key_id,
                        cfg.kalshi_private_key_path)
