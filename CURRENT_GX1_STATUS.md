<!-- GX1_DOCUMENT_CLASS: CANONICAL | current status -->
# Current GX1 status

Current prepared source: 0e81f5b88fd28ea0b80b3bb5314599989f357e15 (/home/andre2/src/GX1_VAL_MARKET_CACHE_V37).

The user rejected the long VAL runtime. Read-only source inspection and the
existing short profile found repeated market-only local/MTF encoding at the
same absolute M1 row (330/341 model stack samples in that branch). This
successor reuses those outputs within one frozen EMA VAL invocation. Entry
token, trade path, lifetime MAE/MFE and Q/Exit evaluation remain per position.
All June opportunities and both sides remain evaluated; no trade-age cap,
confidence cutoff, precision reduction or feature/data omission was added.
The cache is recreated for every invocation/epoch and excluded from TRAIN.

Eleven focused checks passed: cached/uncached Q/actions/routes, changed trade
inputs with the same market row, mixed cache hits/misses, unchanged model
state and TRAIN gradients, preserved checkpoint state, source restrictions,
and full-cohort pause/resume with separate cache instances. Actual first-batch
GPU cache-hit versus uncached comparison must also pass with identical actions
and the existing absolute 0.0001 Bps limit before admitting production actions.
Measured end-to-end speed is still pending. Do not claim a shorter finish time
from the old linear closure-rate estimate; that estimate was withdrawn.

The previous f40ec16f campaign was deliberately disabled/stopped. Preserved
TRAIN remains checkpoint 309 / 19,588 steps; the actual f40ec16f checkpoint SHA
is 8079953fc0ffd3ea6fbf6a91bed8875adfb30128b3a8431431dde57f2d3c455d.
Its archived partial VAL had 13,152 forwards / 1,659,288 views, progress SHA
dbc413312f3e27b32ef7e55678ab020d126436ee6e39b90aea175dd3af7b7fc7.
Original 2959cd09 TRAIN and its verified Mac backup are unchanged. The bound
recipe explicitly permits only the inference-cache model-source addition
when restoring that TRAIN. New VAL accumulators start fresh; no old partial
VAL is represented as evaluated by the new source. See
handover_snapshot/VAL_MARKET_CACHE_PREPARED.json and CURRENT_NATIVE_RUN.json.

Observe the current runtime with scripts/gx1_handover.sh. TEST remains sealed; no final main-model June Bps or live readiness has been established.
