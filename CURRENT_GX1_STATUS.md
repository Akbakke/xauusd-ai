<!-- GX1_DOCUMENT_CLASS: CANONICAL | current status -->
# GX1 status — 2026-09-13

The 267bb0c8 first-batch measurement (2026-09-13 08:14 UTC) used both
TF32 flags disabled: 128-row inference 0.435893 s versus eight 16-row calls
0.691886 s, 1.5873x. All 256 HOLD/EXIT choices matched. Maximum Q difference
was 0.0000491143 Bps; local/MTF intermediate differences were <7.2e-7,
path/summary differences zero. The default torch comparison nevertheless
stopped before any VAL progress was committed. No TRAIN steps were lost.

Successor f40ec16f corrects that diagnostic: absolute Q limit 0.0001 Bps,
zero relative allowance, still exact HOLD/EXIT action agreement. This is an
explicit tolerance change based on the actual FP32 comparison, not a claim
of bitwise equivalence. It continues to reject the earlier 0.03 Bps error.
Three targeted cases pass: observed rounding accepted, larger drift rejected,
changed action rejected even within the numeric allowance. Git hooks pass.
PyTorch documents FP32 batch-versus-slice differences at
https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html#batched-computations-or-slice-computations .

The 1.5873x figure is one comparison of model inference, not full-VAL speed.
New live progress and resource measurements remain required. The prepared
campaign is handover_snapshot/VAL128_ROUNDING_PREPARED.json. The existing
TRAIN checkpoint is restored again; no completed TRAIN or full smoke repeats.

Exact current binding: CURRENT_NATIVE_RUN.json. Read CURRENT_HANDOVER.md and use the read-only handover observer for current runtime status. TEST remains sealed. No final June score or live readiness is established.
