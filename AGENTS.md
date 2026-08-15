# GX1 agent handover

Read `GX1_RULES.md` first. It is binding.

## Current truth

- Scope is offline XAUUSD only.
- Architecture is fixed: the same eight code-owned feature implementations run
  independently on local M5 for Entry and local M1 for Exit, in one model and
  shared encoder. There is no combined pre-owner M1/M5 package.
- Entry is one M5 sequence plus continuous and categorical context. **Every
  dimension derives from `gx1/contracts/entry_model_native_signal_v1.py`; this
  document restates none of them** (CLAUDE.md rule 4 — every count restated in
  this repository has gone stale within days, and on 2026-08-15 all eight of
  them in this file were wrong at once). Read them with:
  `MODEL_NATIVE_SIGNAL_DIM`, `MODEL_NATIVE_BASE_SIGNAL_DIM`,
  `MODEL_NATIVE_MANDATORY_SELECTED_FIELDS`,
  `MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS`,
  `model_native_context_contract_metadata()`, `MODEL_NATIVE_SEQ_LEN`, and
  `htf_features.MULTI_TF_FEATURE_COUNT_V4`. The shape is: a frozen base block +
  the mandatory causal families + the complete code-owned candidate remainder.
  The contract keeps the raw local/MTF evidence and retires handwritten
  scorebooks, the five regime composites, the `tf_agreement` auxiliary
  objective/head and `signed_vol_z_20`.
- Signal binds the exact causal candle geometry/relation/carry owner on local
  and per-TF clocks; its width derives from
  `gx1/features/entry_candle_primitives_v1.CANDLE_PRIMITIVE_FEATURE_NAMES`. The retained local SMC event block has six
  exact displacement/depth/event/age outputs; neither is a scorebook vote.
- Entry consumes one immutable native M5 feature surface across all splits;
  exact contiguous timestamp views are required. Never restore per-split
  inline reconstruction of the specialist fields.
- Exit is the same feature contract at M1, a 480-bar M1 sequence, a dedicated
  learned Entry-decision token (one learned 609-to-128 projection of the exact
  six-block pre-argmax decision source) and the additive 15-field causal path.
  Every result binds the frozen float32 token bytes and exact
  M1/five-TF tensor bytes, clocks, cache, side, quotes, path and trade identity.
- Entry context is closed M15/H1/H4/D1. Exit context is closed
  M5/M15/H1/H4/D1. Build closed OHLCV bars before features; never resample
  already computed M1 indicator values into a higher timeframe.
- Every MTF lane has `MULTI_TF_FEATURE_COUNT_V4` ordered fields. Its three raw
  tick-volume primitives
  (`vol_z_20`, `vol_ratio_5_20`, `vol_pct_96`) are computed by the same volume
  owner from that timeframe's closed OHLCV; volume aggregates by sum. The
  local slices require earlier owner rows for the volume window, so Entry and
  Exit request more native source rows than they slice; the exact counts derive
  from the sequence lengths and the volume owner's window. No zero-filled warmup
  or resampling of computed volume features is allowed.
- The MTF matrix, cache-manifest, liveness, signal-split and mandatory-stack
  schema versions are owned by their contracts and printed by
  `bash scripts/gx1_handover.sh` under `feature_contracts:` — never restated
  here. One
  UTC trading-session clock phases H4 bars on 22/02/06/10/14/18 UTC and D1 at
  22:00 UTC; the retired H4 00/04/... and D1 midnight grids are not
  current-contract inputs.
- Unique model argmax is the only Entry/Exit authority; ties fail closed.
- Training-objective v6 and 46-key recipe schema v5 use plain unweighted CE for
  main/MTF/side classification and plain unweighted BCE for the hierarchy's
  binary tasks. Waves A/B retired direction and hierarchical distribution
  forcing. Fixed auxiliary task weights, rank margins and gate regularization
  remain for Wave C; do not claim every static objective magnitude is gone.
- There is no admitted model, recipe, edge, win-rate or PnL proof, and no
  dataset. The V28 (513) and V29J (592) chains both ran GREEN but were retired
  on 2026-08-14 through the retention owner: nothing was ever trained on
  either, so neither could serve as the comparison baseline it was named as.
  The evaluation reference is the coin-flip null (-13.16 bps TRAIN /
  -18.58 bps VAL). No V30 dataset exists yet.
- No tick-resolution feature, dataset, Exit evaluation or trading claim exists;
  the current Exit input clock is native closed M1.
- The TRAIN-fit squeeze owner and fail-closed six-clock artifact plumbing are
  production-integrated in source. No production squeeze artifact exists yet;
  separately fit immutable TRAIN artifacts for M1/M5/M15/H1/H4/D1, then rebuild
  caches/surfaces/dataset and retrain before making any model or edge claim.
- Fresh native M1/M5 V4 sources and canonical pair generation
  `9b18e215...077232cd` (2026-08-09) exist; the 2026-08-04 parent
  `64d62c1f...a11b84c` is untouched history. They are source authority only.
- The current-contract rebuild chain requires the explicit registry-fit window
  inputs `--registry-fit-train-start`, `--registry-fit-train-end` and
  `--registry-fit-inner-end`. The fit population is the closed interval
  [start, end] with the inner boundary strictly inside it; the chain proves all
  three equal its own split authority by exact timestamp before the next step
  consumes either lane. `--level-tol-quantile-q` was retired and no longer
  exists — passing it aborts the chain. Registry fits freeze into the hash-bound
  build manifests with their exact TRAIN source provenance, including one
  immutable-file-checked pointer to the pair generation; both lanes fail closed
  without it. The level-registry runtime-population shadow is an
  observation-only support check through the same state machine, not a second
  implementation or a live/shadow-trading route.
- V18 was invalid because training `run_id` equalled `dataset_run_id`; it was
  stopped safely. Source/lifecycle fixes invalidate V8/V13/V18 as resume input.

## Takeover sequence

```bash
bash scripts/gx1_handover.sh --check
bash scripts/gx1_handover.sh
git status --short --untracked-files=all
git worktree list --porcelain
```

The executable handover is the status owner. Do not infer state from old run
directories. A registered dirty nested worktree may belong to another agent;
do not modify or delete it.

## Implementation discipline

- Extend an existing owner instead of adding another implementation.
- One formula, one ordered field contract and one normalization state must serve
  native M5 and native M1. The exact Entry signal-manifest hash and TRAIN-rank
  state must bind the Exit surface. Resolution-specific values stay separate.
- Fit that rank state from the pair-bound canonical M5 market fields, then
  require exact canonical-to-final-M5 market identity through TRAIN. Never
  revive the circular rank-from-downstream-source path.
- No ambient environment flag may change feature bytes, dimensions, sampling,
  objectives or model decisions. Recipe-owned values are exact and audited.
- There is one deterministic FP32 trainer path. Feature producers use one
  worker and DataLoaders use zero subprocess workers; do not add compile,
  autocast, TF32, hardware-derived workers or soft performance fallbacks.
- Candidate training uses the full TRAIN population. Smoke subsampling is an
  explicit storage/compute profile and cannot authorize a candidate.
- Calibration events are immutable, split-bound and retained across direction,
  path and sizing stages. TEST cannot fit anything.
- Position-size supervision is the masked exact ECDF rank of selected-side
  path evidence fitted on TRAIN tradable rows only. VAL/TEST apply the frozen
  ECDF; unmasked training is forbidden, and sizing has no direction authority.
- M5 label horizons remain M5 bars even when outcomes are reconstructed from
  M1; never count M1 rows as M5 bars.
- Exit state probes must remain label-independent. Full non-tied long/short
  trajectories are materializable in bounded chunks; probe-only checkpoint
  validation cannot authorize a candidate.
- Missing source, context, path, model or provenance is an error, never a
  neutral value or fallback.

## Machine safety

Every heavy command uses `scripts/gx1_capped_run.sh`: 4G for audits/tests and
at most 20G for the canonical trainer (raised from 10G 2026-08-09 on real
batch=640 RSS measurement, see CLAUDE.md), 512 MiB swap, CPU 0-1, one job at a
time. Communicate before any run lasting more than a minute. Never launch live,
paper, broker, dashboard, collector, notifier or adaptation work. Do not stop
pre-existing processes unless the user explicitly authorizes that action.

The verified takeover environment is CPython 3.10.12 with the direct packages
in `requirements.txt`. Do not reproduce the workstation by freezing unrelated
packages from `.venv`.

## Required verification before commit

```bash
git diff --check
bash -n scripts/*.sh
.venv/bin/python -m compileall -q gx1 tests
scripts/gx1_capped_run.sh --class audit --mem 4G --swap 512M -- \
  .venv/bin/python -m pytest -q
```

Inspect the exact diff and preserve unrelated user changes. No destructive Git
commands. Generated-run cleanup must use the retention contract, not `rm`.

## Next implementation sequence

1. Verify the audited producer commit with the executable handover.
2. Use the published V4 native/canonical pair and run the current-pair rebuild
   chain under a new dataset run ID.
3. Rebuild both resolution surfaces, then pass the exact M5 surface to Entry
   and the exact M1 surface to lifecycle/Exit before building the splits.
4. Materialize a distinct training run ID and bounded smoke recipe.
5. Train/audit smoke, then full candidate if every gate passes.
6. Fit allowed calibration, freeze the candidate, evaluate untouched TEST and
   run the same bundle's unified Exit replay.

Run `bash scripts/gx1_handover.sh` whenever authority or status changes.
