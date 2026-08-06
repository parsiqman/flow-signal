# Polymarket scan FAILED — 2026-08-06 11:40 UTC

```
Traceback (most recent call last):
  File "/home/runner/work/flow-signal/flow-signal/scripts/polymarket_scan.py", line 374, in main
    trades, meta = collect(args)
                   ^^^^^^^^^^^^^
  File "/home/runner/work/flow-signal/flow-signal/scripts/polymarket_scan.py", line 66, in collect
    markets = api.resolved_markets(limit=args.market_pool,
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/flow-signal/flow-signal/src/polymarket/client.py", line 191, in resolved_markets
    batch = self._get(f"{GAMMA}/markets", {
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/flow-signal/flow-signal/src/polymarket/client.py", line 172, in _get
    raise RuntimeError(f"GET failed after {self.cfg.max_retries} tries: "
RuntimeError: GET failed after 4 tries: https://gamma-api.polymarket.com/markets?closed=true&limit=500&offset=2500&order=endDate&ascending=false
HTTP Error 422: Unprocessable Entity

```

If this is a field-resolution error, the API shape has drifted: add the correct spelling to the candidate lists in `src/polymarket/client.py`. The `--offline` run in the same workflow tells you whether the analysis code itself still works.
