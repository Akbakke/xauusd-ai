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

## ADOPTED 2026-08-11 — logit-adjusted direction CE (tau=1.0)

Adopted into the recipe owner (`gx1/contracts/entry_model_native_train_recipe_v1.py`,
key `ENTRY_DIRECTION_LOGIT_ADJUST_TAU=1.0`, env count 163 → 164) and the trainer
(`gx1/models/entry_v10/entry_v10_ctx_train_v3.py`).

- **Decision:** every direction CE training loss (main CE, tail-direction CE
  reuse, mtf-dir aux CE) consumes `logits + tau*log(TRAIN priors)`; the same
  adjustment is applied in `validate()`'s CE for val-loss comparability only.
  Emitted/serving logits, all battery probabilities (prior-match, balance,
  min-pred-rate, slice terms) and every acc/pred-rate metric stay on raw
  logits (rule 6 train==serve; rule 3 one decision authority). `tau=0.0` is
  the exact-compatibility switch (no offset built, bit-identical CE graph).
- **Origin:** tau=1.0 is the standard value of the published method (Menon et
  al. 2021, "Long-tail learning via logit adjustment"). Priors = physical
  TRAIN label rates computed in-trainer at dataset load; a zero/non-finite
  rate fails closed (`ENTRY_DIRECTION_LOGIT_ADJUST_PRIOR_INVALID`).
- **Class weights under adjustment:** with tau>0, long/short/flat direction CE
  class weights are exactly 1.0 (Menon et al.: adjustment replaces
  reweighting; combining both double-corrects). tau==0 keeps the 2026-08-08
  sqrt-softened construction unchanged. The old comment's claim that
  sqrt-softened `w_k * r_k` "equalizes by construction" is corrected in both
  paths: `w_k * r_k = sqrt(r_k(1-r_k))` still orders by class rate (measured
  2026-08-11 on V28 TRAIN rates: +3.8% CE mass on LONG vs SHORT).
- **Evidence basis (measured 2026-08-10/11, V28):** training oscillates/
  collapses class-wise under the anti-collapse penalty battery; the collapse
  direction flips with seed; no static loss asymmetry explains it; the battery
  anchors TRAIN-batch marginals while plain CE mildly prefers collapse.
  Handling the prior inside the CE removes the need for the battery to fight
  the CE. Anchoring weights themselves are unchanged (no origin for new
  magnitudes without a sweep on real data — deferred item 4 above stands).
- **Also adopted (log-only + record completeness):** short-side
  DEAD/TEASER/HARD_NEG rate proofs mirroring the long-side lines (rule 2e —
  the short negatives were applied under `ENTRY_SYMMETRIC_NEGATIVES` but never
  recorded); `[ENTRY_HARD_NEG_RECIPE]` now states the short-side application;
  `[ENTRY_TRAIN_RECIPE]`, `[ENTRY_DIRECTION_BALANCE_PROOF]` and the bundle
  metadata now record `direction_logit_adjust_tau`.
