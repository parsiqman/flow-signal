# Polymarket scan FAILED — 2026-11-22 14:17 UTC

```
Traceback (most recent call last):
  File "/home/runner/work/flow-signal/flow-signal/scripts/polymarket_scan.py", line 1548, in main
    trades, meta = collect(args)
                   ^^^^^^^^^^^^^
  File "/home/runner/work/flow-signal/flow-signal/scripts/polymarket_scan.py", line 139, in collect
    raise RuntimeError("no trades returned from any sampled market")
RuntimeError: no trades returned from any sampled market

```

If this is a field-resolution error, the API shape has drifted: add the correct spelling to the candidate lists in `src/polymarket/client.py`. The `--offline` run in the same workflow tells you whether the analysis code itself still works.
