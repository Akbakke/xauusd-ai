<!-- GX1_DOCUMENT_CLASS: CANONICAL | current status -->
# Current GX1 status

Current prepared source: 1548dd7c47d7f5a83ede4ccca1ef635b443d51f0 at /home/andre2/src/GX1_VAL_HOTPATH_V39.
Branch fix/native-val-hotpath-20260913 is pushed with exact remote HEAD verified.
The user authorized further performance work after items 1–4 on September 13.
Use CURRENT_NATIVE_RUN.json for all exact runtime and source bindings.

This successor validates a private copy of frozen normalization once, indexes
immutable entry metadata once, converts the active mask only when saving,
and encodes an identical LONG/SHORT price history once during frozen VAL.
Side-specific summaries, fusion and Q decisions remain separate. TRAIN's
forward and gradients, data, FP32, features, five timeframes/eight families,
all 5,508 June entries/both sides, costs and batch sizes remain unchanged.

V38 invocation 3 durably saved 13,176,595 views / 104,181 forwards at cursor
2579/3992, then failed with CANDIDATE_EXECUTION_PAUSE_RECEIPT_CONFLICT.
It must not be described as an outer guard PASS. Its complete VAL progress
file is retained and explicitly bound as the new recipe's resume origin.
TRAIN remains checkpoint 309 / 19,588 optimizer steps, using the same immutable
EMA validation snapshot. The collision came from identical budget contents
while the TRAIN pointer stayed fixed. Native VAL receipt names now also bind
the verified saved progress-file SHA; existing receipts are never overwritten.

Fifteen targeted CPU cases and mandatory commit hooks passed. They cover
short/full-length shared paths, side outputs/actions, TRAIN gradients,
costs/slice hashes, pause/resume and identity drift, and three successive
VAL receipts with the same budget and TRAIN pointer. Actual GPU parity,
startup/resume and sustained throughput still require runtime observation.

The controller is prepared for the new source after the old process exited.
No TRAIN/VAL prefix is repeated. Keep one heavy job, four CPU workers,
19 available WSL threads, normal priority, all existing resource/GPU guards,
and sparse observations. Max 30 epochs, June VAL each epoch, patience 5.
TEST remains sealed; positive full-policy net Bps and live readiness unproven.
