# GX1 Engine

GX1 is an offline XAUUSD research and evidence pipeline for one learned trading
system. It is not currently an admitted or profitable trading bot.

## Architecture

```text
native OANDA M1/M5
  -> same 8 causal feature owners on separate native M5/M1 clocks
  -> shared encoder
       |-- Entry: local M5 + M15/H1/H4/D1 -> LONG / SHORT / FLAT
       |-- Exit: local M1 + M5/M15/H1/H4/D1 -> HOLD / EXIT_NOW
       `-- learned sizing/evidence heads
  -> calibration -> untouched TEST -> same-bundle replay
```

Entry consumes 96 local M5 bars with 538 ordered signals (34 base + 371
mandatory causal + 133 TRAIN-ranked over 16 mandatory families — the V29
event surface with level/trendline registries and per-timeframe event
primitives), 164 continuous context fields, 5 categorical fields and closed
M15/H1/H4/D1 context. Exit
consumes the same definitions and TRAIN normalization on 480 local M1 bars,
closed M5/M15/H1/H4/D1 context, the frozen Entry representation and its causal
in-trade path. Higher-timeframe OHLCV closes before feature computation;
computed M1 indicators are never resampled upward.

Entry's 538/164/5 tensors are read from one immutable, hash-bound native M5
surface and sliced by exact timestamp across all three splits. Exit uses the
corresponding native M1 surface. There is no split-local alternate feature
builder or cross-resolution value copy.

The model is the only decision authority. Exact top-logit ties, missing paths,
stale artifacts and lineage mismatches fail closed. There are no active
handwritten direction rules, fallbacks or alternate replay selectors.

## Current status

The source architecture and contracts are substantially connected and tested,
but the system is not empirically finished:

- the current 2026-08-11 tree passes repo-wide lint/compile/shell checks and
  all 2,078 collected tests under the 4G cgroup;
- fresh native M1/M5 V4 sources and canonical pair generation
  `9b18e215...077232cd` (2026-08-09) are published and hash-bound;
- the V28 dataset chain ran GREEN end to end on the repaired feature
  substrate; it is the frozen baseline for the pre-registered V29-vs-V28
  evaluation, not an admitted training substrate;
- the V29 event surface (level/trendline registries, per-TF event
  primitives; 538 signals after the V30 fidelity wave) is committed in source; its dataset rebuild has
  not run yet;
- no accepted Entry/Exit checkpoint or calibrated bundle;
- no untouched-TEST edge, historical PnL or win-rate proof;
- no same-candidate full-TEST unified Exit proof.

Every dataset built on the retired 513 surface is invalid as substrate for
new training. A fresh V29 rebuild is required.

## Start here

```bash
bash scripts/gx1_handover.sh --check
bash scripts/gx1_handover.sh
bash scripts/entry_next_edge_control.sh --help
```

Read `GX1_RULES.md`, `AGENTS.md`, `SYSTEM_MAP.md` and
`HANDOVER_XAU_DIRECTION_REPAIR_20260714.md`. Large immutable data lives under
`/home/andre2/GX1_DATA`; code lives in this repository. Never select artifacts
by `latest`, mtime or glob order. `requirements.txt` pins the small direct
runtime and verification surface used by the current Python 3.10.12 checkout.

## Resource safety

All heavy work runs through `scripts/gx1_capped_run.sh`, one job at a time.
Audits/tests are capped at 4G; canonical training is capped at 20G; swap is
512 MiB and CPU affinity is 0-1. Feature producers use one worker and model
DataLoaders use zero subprocess workers. Training has one deterministic FP32
path; compile, autocast, TF32 and runtime-selected fast modes are forbidden.
Partial or cap-killed output is not reusable.

Live, paper, broker, daemon, promotion and drift/adaptation work are outside
the frozen scope. See `GX1_RULES.md` for the complete binding rules.
