# DRAFT — next immutable recipe decision (awaiting owner approval)

Status: **SUPERSEDED HISTORICAL PROPOSAL ONLY.** This document carries no authority. The recipe owner
is `gx1/contracts/entry_model_native_train_recipe_v1.py`; nothing changes until
the owner adopts a decision and the recipe audit re-binds. Drafted 2026-08-08
from the six-lane headmaster review and the measured HEAD smoke attribution.

> Current-status note, 2026-08-30: V46's recipe owner and hash-bound recipe
> audit are current. This draft must not change V46 features, objectives,
> batch geometry or the guarded candidate-learning-validation plan recorded in
> the handover. The technical CPU parity/journal and candidate Exit-evidence
> binding repair change no recipe authority and do not authorise full or
> external training.

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

## ADOPTED 2026-08-13 — two stability dampers (V30 package 5)

- **Evidence basis (measured 2026-08-12/13, three seeds, identical recipe:
  batch 64 x accum 10, 8 epochs, lr 3e-4, 25k rows, logit-adjusted CE):**
  s1337 guard-OK 4/7, no collapse, best 0.238; s1338 guard-OK 1/7, FLAT drift,
  hard-red; s1339 guard-OK 4/6, best 0.256 then LONG collapse at epoch 6,
  hard-red. The balanced checkpoint score oscillates epoch to epoch and the
  late epochs lean hard — a limit cycle at a fixed step size. The two dampers
  below address the *dynamics*; neither claims to create edge.
- **`ENTRY_TRAIN_LR_COSINE_DECAY=1` (ADOPTED ON).** A switch, not a magnitude.
  `1` selects `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
  T_max=<the declared --epochs budget>, eta_min=0.0)` — the library's own
  standard cosine anneal (Loshchilov & Hutter 2017, SGDR) without restarts and
  without warmup. Its only two inputs are the declared epoch budget and the
  library default 0.0, so no number is chosen here. The scheduler steps once
  per epoch AFTER that epoch's training, so epoch 0 trains at the declared
  `lr` and the schedule reaches exactly 0 at the end of the budget.
  Rationale: a step size decaying to zero shrinks the oscillation amplitude of
  exactly this limit-cycle class in the late epochs where the leans occur.
  `0` reproduces the fixed-LR behaviour bit-identically — no scheduler object
  is constructed and no `param_group` is ever written.
- **`ENTRY_TRAIN_WEIGHT_EMA_DECAY=epoch` (DECLARED ON — operator decision
  taken 2026-08-13, V30 package 6).** An exponential moving average of the
  model weights is maintained across optimizer steps; VALIDATION and
  checkpoint selection read the averaged weights (so the gate judges the
  weights that will ship) while the raw weights keep training.
  **Origin: the key declares a HORIZON, not a magnitude.** `epoch` selects
  `derive_weight_ema_decay` in the recipe owner — the averaging horizon is
  exactly ONE declared epoch of optimizer steps, so
  `steps_per_epoch = ceil(train_rows / (batch_size * grad_accum_steps))` and
  `decay = 1 - 1/steps_per_epoch`. For the declared smoke budget (25,000 rows,
  batch 64, accumulation 10) that is 40 steps per epoch and **decay = 0.975**.
  The textbook `0.999` was rejected on measurement, not taste: it is a
  PER-STEP constant with a ~1000-step horizon, while this profile's entire run
  is 40 x 8 = 320 optimizer steps, so a 0.999 shadow would still be dominated
  by its initialization at the last checkpoint — and it has no in-repo named
  constant to adopt (searched 2026-08-13: no `0.999`, `ema_decay`,
  `EMA_DECAY`, `AveragedModel`, `polyak`, `swa`), so pinning it would be an
  invented magnitude (rule 2a/2b). A derivation also follows batch/
  accumulation/row-budget changes automatically, which a pinned float cannot.
  Exactly two values are declared (`MODEL_NATIVE_WEIGHT_EMA_DECAY_DECLARED_VALUES
  = ("0.0", "epoch")`); a bare decay fails closed. `0.0` remains the
  exact-compatibility sentinel: it resolves to 0.0 without consulting the
  budget, no shadow state is allocated, no `state_dict` is swapped, and the
  checkpoint expression is the pre-package-5 raw clone character for
  character. The tau precedent governs the split (the recipe owner owns the
  rule and the formula; the run-dependent quantity is resolved in-trainer from
  that owner's function, never from an ambient default — rule 14).
- **Env contract:** 164 -> 166 pre-V29 keys
  (`_PRE_V29_RECIPE_ENV_KEY_COUNT`); the total stays derived as that base plus
  the declared V29 registry key tuple.
- **Recorded:** `[ENTRY_TRAIN_RECIPE]`, `[TRAIN_STABILITY_DAMPERS]` and
  `[TRAIN_WEIGHT_EMA_DERIVATION]` log the resolved values, and bundle metadata
  records `train_lr_cosine_decay`, `train_weight_ema_decay` and the complete
  `train_weight_ema_derivation` (declared token, row budget, batch geometry,
  steps per epoch) so the number can be reproduced from the bundle alone.
- **Unproved:** neither damper has been run end to end on the V30 substrate.
  The three-seed measurement above is the *cause* evidence; that these two
  changes remove the limit cycle is a hypothesis until a fresh multi-seed
  smoke says otherwise. Existing readiness/recipe artifacts bind the old
  trainer bytes and correctly fail closed; re-materialization precedes the
  next run.
