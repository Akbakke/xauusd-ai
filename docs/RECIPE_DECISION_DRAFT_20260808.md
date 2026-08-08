# DRAFT — next immutable recipe decision (awaiting owner approval)

Status: **PROPOSAL ONLY.** This document carries no authority. The recipe owner
is `gx1/contracts/entry_model_native_train_recipe_v1.py`; nothing changes until
the owner adopts a decision and the recipe audit re-binds. Drafted 2026-08-08
from the six-lane headmaster review and the measured HEAD smoke attribution.

## Why a new decision is needed

The 2026-08-07/08 smokes produced the first honest post-repair evidence:

- FLAT collapse is gone (anti-collapse active at batch ≥64): epoch-1 VAL
  predictions 53.9/5.9/40.2 — three classes, not 100 % FLAT.
- The remaining failure was **machinery, not model**: (a) the pooled
  directional CE weight passed the TRAIN prior through as a 0.80-nat/step
  anti-SHORT tilt (fixed in code, d631d64e, per-class weights); (b) the
  admission gate `ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL=0.35` is
  arithmetically unreachable at epochs 1–2 under any weighting, so
  `TRAIN_FAIL_NO_BEST_STATE` was overdetermined by schedule.
- Step-time attribution on HEAD bytes (2026-08-08 07:16, batches 19/20):
  0.5 s assembly + 11.0 s forward + 17.8 s backward + 0.05 s step = 29.4 s at
  batch 64 on CUDA, GPU 50 %, VRAM 5.2/24 GB. The compute lives in the
  32 sequential 8-row exit-action chunks
  (`UNIFIED_EXIT_ACTION_FORWARD_CHUNK_ROWS=8`), a constant derived for the 10G
  **CPU RSS** ceiling and mis-scoped on CUDA where the attention transients
  live in VRAM.

## Proposed decision items

1. **Admission schedule.** Add `ENTRY_CKPT_ADMISSION_MIN_EPOCHS=<N>` gating the
   hard `MIN_PRED_TO_LABEL` requirement: before epoch N the balance guard
   records but does not veto; from epoch N it vetoes exactly as today. Origin
   for N: the existing schedule convention
   `ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS=6` — the recipe already
   recognizes that slice verdicts before epoch 6 are noise. Same N, same
   rationale, one convention.

2. **Device-scoped exit-action chunk.** Promote
   `UNIFIED_EXIT_ACTION_FORWARD_CHUNK_ROWS` from trainer constant to recipe
   key, with the CPU value 8 (derived from the 10G RSS ceiling, measured) and
   a CUDA value derived from measured VRAM headroom (5.2/24 GB at chunk 8;
   attention transients scale linearly in chunk rows). Chunking is
   training-equivalent, not bit-identical (dropout RNG order), so the value is
   decision-affecting and belongs here, not in code.

3. **Adopt into the recipe record:** the per-class directional CE weight
   (already in code at d631d64e; reverses the pooled half of the 2026-05-26
   decision, keeps its sqrt-softening half) and the specialist-movement
   admission extension (contract v1: 6 → 8 parameters).

4. **Explicitly deferred, recorded here so they are not silently dropped:**
   anchoring-weight sweep (balance 0.5 / prior-match 8.0 / min-pred-rate 12.0 —
   no legitimate origin for new magnitudes without a sweep on real data);
   V28 divergence/M5-RSI feature owners (gated behind a smoke that passes
   admission on V27); y_side FLAT-parking rewrite (proven masked, zero effect
   today, rebuild-coupled).

## What this decision does NOT do

No anchoring weights change. No feature surface change. No threshold moves
without its sampling-error bound stated (rule 2f). No live-route reopening.
