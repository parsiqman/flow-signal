"""Ask Claude for a direction on the next settlement window.

Contract: the model returns UP or DOWN plus a confidence and a one-line reason.
The reason is not decoration -- a bot that logs only "DOWN" cannot be debugged
after a losing week, and the line costs a handful of output tokens.

Every failure path in this module returns ABSTAIN, never a guess. A timeout, a
rate limit, a malformed response or a refusal all mean "no trade this window".
A trading bot that treats an API error as a coin flip is a bot that trades on
coin flips.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import anthropic

UP, DOWN, ABSTAIN = "UP", "DOWN", "ABSTAIN"

# Stable across every call -- keep it that way. Anything that varies per window
# belongs in the user message, or it invalidates the cached prefix.
SYSTEM_PROMPT = """\
You are a short-horizon directional classifier for Bitcoin.

You will receive a compact summary of BTC market state computed strictly from \
candles that had already closed at time T. You must judge whether the price at \
the end of the current settlement window will be ABOVE (UP) or BELOW (DOWN) the \
reference price given in the summary.

Reading the summary:
- bp = basis points (1bp = 0.01%). rv* = annualized realized volatility, %.
- slope*_bp = least-squares drift of log price, bp per minute, over that lookback.
- r2_30m = how linear the 30m trend is (0 = noise, 1 = a straight line).
- ema9v21 / ema21v50 = EMA spreads in bp; positive means the faster EMA is above.
- vwap = spot minus VWAP in bp. range_pos_60m = where spot sits in the 60m range \
(0 = at the low, 1 = at the high). streak = consecutive up(+) or down(-) minutes.
- CANDLES give body/upper-wick/lower-wick in bp, oldest to newest. A small body \
with a long wick is rejection of that extreme.

How to weigh it:
- Over 15 minutes, BTC is close to a random walk. The honest base rate is near \
50/50 and you should expect to be wrong often. Say so through `confidence`, not \
by refusing to pick.
- `confidence` is the probability you assign to your own call. Use 0.50 when the \
state is genuinely uninformative, and reserve values above 0.60 for setups where \
the evidence is unusually one-sided. Systematically overstating confidence will \
size this bot into losses.
- Mean reversion and momentum both appear at this horizon. Vol regime and r2 are \
what separate them: strong linear trends in high vol tend to persist for minutes; \
exhausted streaks against long wicks at a range extreme tend to snap back.
- Less time left in the window means less room for a move to develop; weight the \
most recent structure accordingly.

Return only the structured object. No preamble."""

SCHEMA = {
    "type": "object",
    "properties": {
        "direction": {"type": "string", "enum": [UP, DOWN]},
        "confidence": {"type": "number", "minimum": 0.5, "maximum": 1.0},
        "reason": {"type": "string", "maxLength": 160},
    },
    "required": ["direction", "confidence", "reason"],
    "additionalProperties": False,
}


@dataclass
class Signal:
    direction: str                 # UP | DOWN | ABSTAIN
    confidence: float = 0.0
    reason: str = ""
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    error: str = ""
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def tradeable(self) -> bool:
        return self.direction in (UP, DOWN)

    def to_dict(self) -> dict:
        return {
            "at": self.at.isoformat(), "direction": self.direction,
            "confidence": round(self.confidence, 3), "reason": self.reason,
            "latency_s": round(self.latency_s, 2), "model": self.model,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "error": self.error,
        }


class ClaudeDirectionModel:
    """Wraps one Messages API call per settlement window."""

    def __init__(self, model: str = "claude-opus-5", effort: str = "medium",
                 max_tokens: int = 2048, timeout_s: float = 60.0, client=None):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        # Zero-arg client resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
        # `ant auth login` profile, in that order.
        self.client = client or anthropic.Anthropic()

    def predict(self, feature_text: str, extra_context: str = "") -> Signal:
        user = feature_text if not extra_context else f"{feature_text}\n{extra_context}"
        t0 = time.monotonic()
        try:
            resp = self.client.with_options(timeout=self.timeout_s).messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                # Adaptive thinking; `effort` trades reasoning depth against cost
                # and latency. Both live under output_config alongside the schema.
                thinking={"type": "adaptive"},
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": SCHEMA},
                },
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    # Harmless if the prefix is below the model's minimum cacheable
                    # size (it currently is); it starts paying the moment the
                    # system prompt grows.
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.APIStatusError as e:
            # Carry the server's own message, not just the status code. A bare
            # "api_status_400" is indistinguishable between a bad schema, an
            # expired card and a rate limit -- and this bot's response to all
            # three is a silent abstain, so the log line is the only evidence
            # anyone will have that it stopped trading and why.
            detail = ""
            try:
                detail = (e.body or {}).get("error", {}).get("message", "")
            except (AttributeError, TypeError):
                pass
            return self._abstain(
                f"api_status_{e.status_code}: {detail or e}"[:240], t0)
        except anthropic.APIConnectionError as e:
            return self._abstain(f"api_connection: {e}", t0)
        except Exception as e:                      # never let the loop die on one call
            return self._abstain(f"unexpected: {type(e).__name__}: {e}", t0)

        latency = time.monotonic() - t0

        if resp.stop_reason == "refusal":
            cat = getattr(resp.stop_details, "category", None) if resp.stop_details else None
            return self._abstain(f"refusal:{cat}", t0, latency=latency)
        if resp.stop_reason == "max_tokens":
            return self._abstain("truncated: raise max_tokens", t0, latency=latency)

        try:
            text = next(b.text for b in resp.content if b.type == "text")
            data = json.loads(text)
            direction = data["direction"]
            if direction not in (UP, DOWN):
                raise ValueError(f"bad direction {direction!r}")
        except (StopIteration, json.JSONDecodeError, KeyError, ValueError) as e:
            return self._abstain(f"unparseable: {e}", t0, latency=latency)

        return Signal(
            direction=direction,
            confidence=float(data.get("confidence", 0.5)),
            reason=str(data.get("reason", ""))[:160],
            latency_s=latency,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=resp.model,
        )

    def _abstain(self, error: str, t0: float, latency: float | None = None) -> Signal:
        return Signal(direction=ABSTAIN, reason="no trade",
                      latency_s=latency if latency is not None else time.monotonic() - t0,
                      model=self.model, error=error)


class ScriptedModel:
    """Offline stand-in used by tests and by --feed synthetic dry runs.

    Deliberately dumb (last-5m momentum). It exists to exercise the plumbing
    without spending money or needing a key -- not as a fallback strategy.
    """

    def __init__(self, sequence=None):
        self.sequence = list(sequence or [])
        self.calls = 0

    def predict(self, feature_text: str, extra_context: str = "") -> Signal:
        self.calls += 1
        if self.sequence:
            d = self.sequence[(self.calls - 1) % len(self.sequence)]
        else:
            up = "ret5m +" in feature_text
            d = UP if up else DOWN
        return Signal(direction=d, confidence=0.55, reason="scripted",
                      model="scripted", latency_s=0.0)
