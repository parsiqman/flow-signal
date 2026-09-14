"""NightShark-style 15-minute BTC direction bot for Kalshi.

Pipeline, in order: data -> features -> llm -> execution, with risk as a gate
that can halt the whole thing. `runner.py` wires them together.

Nothing here constitutes financial advice, and nothing here has been shown to
be profitable. Run it in paper mode and measure before it touches real money.
"""

__all__ = ["config", "data", "features", "llm", "execution", "risk", "runner"]
