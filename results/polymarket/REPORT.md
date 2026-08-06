# Polymarket scan FAILED — 2026-08-06 22:55 UTC

```
Traceback (most recent call last):
  File "/home/runner/work/flow-signal/flow-signal/scripts/polymarket_scan.py", line 482, in main
    trades, meta = collect(args)
                   ^^^^^^^^^^^^^
  File "/home/runner/work/flow-signal/flow-signal/scripts/polymarket_scan.py", line 68, in collect
    return collect_named_wallets(args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/flow-signal/flow-signal/scripts/polymarket_scan.py", line 207, in collect_named_wallets
    raise RuntimeError(
RuntimeError: no wallet addresses to evaluate. Username lookup failed; pass the 0x address from the profile URL with --wallets.

```

If this is a field-resolution error, the API shape has drifted: add the correct spelling to the candidate lists in `src/polymarket/client.py`. The `--offline` run in the same workflow tells you whether the analysis code itself still works.
