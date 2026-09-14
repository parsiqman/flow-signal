# NightShark-style Kalshi BTC bot

An automated directional bot for Kalshi's 15-minute BTC markets. Claude reads a
compact summary of market state once per window and answers UP or DOWN; the bot
turns that into an order on the nearest strike, and a risk loop stops everything
if drawdown gets out of hand.

```
data.py       live/historical BTC candles, no-lookahead by construction
features.py   volatility, trend, momentum, multi-timeframe candle structure
              -> ~130-token text summary
llm.py        one Messages API call per window -> UP | DOWN | ABSTAIN
execution.py  UP/DOWN -> YES/NO on the nearest strike; paper + live brokers
risk.py       PnL ledger, sticky drawdown kill switch, persisted to disk
runner.py     the 15-minute loop, and the simulator
```

## Run it

```bash
pip install -r requirements.txt
export PYTHONPATH=src

# End-to-end on synthetic data, no API key, no network, no money:
python -m nightshark.runner --simulate 20 --scripted

# Measure a strategy over a full sample (risk limits lifted, paper only):
python -m nightshark.runner --simulate 400 --scripted --measure

# Claude actually makes the calls (needs ANTHROPIC_API_KEY):
python -m nightshark.runner --simulate 20

# State of the account and the kill switch:
python -m nightshark.runner --status
python -m nightshark.runner --reset-halt
```

Everything defaults to **paper mode**. Live trading takes three deliberate steps:

```bash
export KALSHI_KEY_ID=...            # from the Kalshi API keys page
export KALSHI_PRIVATE_KEY_PATH=...  # the RSA private key .pem
export NIGHTSHARK_DRY_RUN=false
python -m nightshark.runner --list-markets   # verify the series ticker FIRST
python -m nightshark.runner --live
```

## Configuration

Every field in `config.py` is settable as `NIGHTSHARK_<FIELD>`. The ones that
matter most:

| Variable | Default | What it does |
|---|---|---|
| `NIGHTSHARK_DRY_RUN` | `true` | Paper broker. Set `false` for real money. |
| `NIGHTSHARK_CONTRACTS_PER_TRADE` | `10` | Fixed size. No conviction scaling. |
| `NIGHTSHARK_MAX_DRAWDOWN` | `200` | Dollars below peak equity before the bot halts. |
| `NIGHTSHARK_DAILY_LOSS_LIMIT` | `100` | Dollars lost in a UTC day before it stops for the day. |
| `NIGHTSHARK_MAX_PRICE_CENTS` | `70` | Never pay more than this per contract. |
| `NIGHTSHARK_MIN_CONFIDENCE` | `0` | Skip windows the model isn't sure about. |
| `NIGHTSHARK_FEED` | `synthetic` | `rest` pulls real candles from `FEED_URL`. |
| `NIGHTSHARK_MODEL` | `claude-opus-5` | |
| `NIGHTSHARK_EFFORT` | `medium` | `low`..`max`. Trades reasoning depth for cost. |

## Design decisions worth knowing about

**The model is never told the contract's price.** That price is the market's own
probability estimate and is better informed than the feature summary. Show it to
the model and the model learns to echo it — the bot looks calibrated and has no
independent signal. Direction and price are judged separately: Claude answers
direction, the runner mechanically decides whether the offered price is worth
paying.

**Failures abstain; they never guess.** A timeout, a rate limit, a malformed
response, or a refusal all produce `ABSTAIN` and no trade.

**The kill switch is sticky and survives restarts.** Drawdown is measured against
the all-time equity peak on disk. `--reset-halt` re-baselines that peak to
current equity — otherwise clearing the flag would just re-trip on the next check.

**Paper mode charges you.** The paper broker prices contracts from a binary-option
model, adds a spread, and applies Kalshi's real fee formula. A 50%-accurate bot
loses money in paper mode, exactly as it would live.

## Before you risk anything

Two numbers decide whether this works, and neither is the hit rate on its own:

- At a 50c entry, fees and spread put **breakeven near 52-53%**. `--simulate`
  prints your average entry price, the implied breakeven, and your actual hit
  rate side by side.
- Over 15 minutes BTC is close to a random walk. A genuine edge here would be
  small, and the null hypothesis — that there is none — is the one to beat.

Run `--simulate --measure` on real candles (`NIGHTSHARK_FEED=rest`) over a few
hundred windows before going live. `--measure` lifts the risk limits for the
duration of the run: live you want to stop after six losses, but measuring, that
stop truncates the sample exactly at a losing streak and the hit rate you read
back is conditioned on having stopped. Without it a 400-window run can report on
9 trades. Synthetic-tape results measure the plumbing, not the
strategy: that tape has no reason for direction to be predictable, so anything
profitable on it is a property of the generator.

## Unverified against the live exchange

This was built where `api.elections.kalshi.com` is blocked by network policy, so
the live paths are written from documentation and **not** confirmed against the
API. Check these before trading:

1. **The series ticker** (`KXBTC` default) — confirm with `--list-markets`.
2. **The auth signature** — RSA-PSS over `timestamp + METHOD + path`, salt length
   = digest length.
3. **Order and market field names** — `floor_strike`, `yes_ask`, `client_order_id`.
4. **The fee rate** (0.07) — it varies by market.

The paper broker, the features, the risk loop, and the model integration are all
exercised by `tests/test_nightshark.py` and do not depend on any of the above.

Research tooling, not financial advice. Nothing here has been shown to be
profitable, and the most likely honest result is that the edge is smaller than
the framing implies.
