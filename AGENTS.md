# GX1 agent handover

Read `GX1_RULES.md` first. It is binding.

## Current truth

**Operational override, 2026-08-28:**
[`docs/CURRENT_AUDIT_STATUS_20260828.md`](docs/CURRENT_AUDIT_STATUS_20260828.md)
is the short current-state companion to this historical handover. It controls
the next action: retain V46, keep background services stopped, finish the
current-contract audit, and do not start full training, TEST, demo or live.

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
  learned Entry-decision token (one learned projection of the exact six-block
  pre-argmax decision source; widths owned by
  `gx1/contracts/entry_decision_token_v1.py`, never restated here) and the
  additive 15-field causal path.
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
- Unique model argmax is the only Entry/Exit authority; ties fail closed. What
  is argmaxed is `entry_action_q_bps` / `unified_exit_action` — expected return
  in basis points, not calibrated probabilities (`entry_fitted_q_v1.py`).
- The objective and recipe owners
  (`gx1/contracts/entry_model_native_training_objective_v1.py`,
  `gx1/contracts/entry_model_native_train_recipe_v1.py`) own their schema
  versions, keys and flags; **execute them, this file restates none of them**.
  Proven from source 2026-08-19: the sole decision loss is masked raw-bps MSE
  on fitted-Q, task weights are learned by trainable homoscedastic
  log-variance, and no cross-entropy holds decision authority — one masked BCE
  survives on the `trendline_event` auxiliary head. The retired "objective v6 /
  46-key recipe schema v5 / unweighted CE / pending Wave C" description was
  false by 2026-08-19. Whether every static magnitude in the trainer is gone is
  **not examined**; do not claim it either way.
- There is no admitted model, recipe, edge, win-rate or PnL proof, and no
  admitted dataset. V46 is the explicitly hash-bound audited research dataset:
  its data, liveness, all-eight-specialist, source-backed sequence and
  M1-decision-to-fill causality PASS evidence is verified by
  `current_audited_dataset_evidence` in the launch state. It remains blocked
  from admission because fitted-Q production economics lacks immutable
  executable bid/ask, cost, financing, gap/terminal and portfolio evidence. The V28 (513)
  and V29J (592) chains both ran GREEN but were retired
  on 2026-08-14 through the retention owner: nothing was ever trained on
  either, so neither could serve as the comparison baseline it was named as.
  The evaluation reference is the coin-flip null; its magnitude is
  substrate-specific and lives in
  `HANDOVER_XAU_DIRECTION_REPAIR_20260714.md`, not here. V36 reached the
  pre-dataset cross-surface audit and ended RED; cache v30 and overlap policy v3
  repaired the scalar-byte and representation-alias defects it exposed. The
  V37 then exposed an untrimmed honest Group-A causal warmup prefix; the shared
  fail-closed prefix owner repaired it. V38 then exposed an unbounded lifecycle
  representation; its compact pointer repair reconstructs the unchanged state
  population from the hash-bound M1 clock. Fresh V39 proved that repair on the
  full chain: TRAIN/VAL/TEST and compact lifecycle completed, and full-input
  liveness passed. Post-build pretrain audit v5 ended RED because the retired
  selected-side `y_bad_path` diagnostic is structurally zero whenever the
  positive-PnL direction policy is coherent. Both side-specific sources were
  live and the scalar had zero copy mismatches. Audit v6 makes this exemption
  explicit while still requiring finite scalar bytes, exact copy identity and
  live side sources; adversarial tests and a diagnostic V39 replay pass.
  V35/V36/V37/V38/V39 products remain invalid resume/consumer input (rule 7).
  No admitted dataset exists.
- The current V46 feature/data/sequence evidence has passed technical
  training-pipeline proof, but cannot claim prediction quality or edge. The
  historical 32-row smoke successfully exercised CUDA bundle plumbing; it is
  superseded as learning evidence by the fresh batch-8, 60-step one-epoch
  probe. Its first materialization exposed an exact strict-reload defect: EMA
  averaged every floating buffer, including immutable input-normalization
  state. Commit `fad763af` confines EMA averaging to named parameters and
  copies buffers exactly. Its focused CPU regression executes 60 updates,
  exports, strictly reloads and validates the complete normalization state; a
  second production partition test proves all eight local/MTF family branches
  and every learned specialist-gate logit receive gradient after the
  zero-initialized correction opens. The repaired probe completed under the
  automatic guard and produced a fresh atomic bundle. The exact evaluator wrote
  all 70,880 VAL predictions at 55 C / 155.63 W / 727 MiB. The bundle audit
  passes data, lineage, shape, liveness and active-head checks but fails an old
  *quality* heuristic: five families never top-rank and structure/swing averages
  0.007619497 below a 0.01 floor. Neither condition is a valid prerequisite for
  the first full candidate because a softmax has only one top rank per row and a
  small smoke cannot establish economic relevance. The technical start contract
  instead requires every family to be finite, positive, dynamic and connected
  by actual gradients. Keep the strict quality audit for later candidate/OOS
  qualification, but assess contribution with ablation and regime-sliced OOS,
  not forced equal gate shares. The preregistered selective-edge decision is
  FAIL. This proves no edge. The clean-source reports emitted 2026-08-28 now
  explicitly authorize candidate research, never activation: smoke readiness,
  trainability readiness and candidate readiness all pass. The source now
  contains a hash-bound, two-slot candidate resume protocol for the
  same 20-minute 220 W / 70 C / 12 GiB guarded window. It persists model,
  fixed fitted-Q target, optimizer, EMA, scheduler, deterministic order/RNG,
  selection state and an in-flight full-VAL accumulator. The protocol has
  regression coverage but has not yet been accepted with a full candidate CUDA
  run. It must first be committed, recipe-audited and dry-run against the exact
  V46 evidence. Do not bypass the guard or mislabel a partial session as
  candidate evidence. Resolve exact logs and the next decision from the
  handover, never from an old run directory.
- Before candidate execution, a pre-candidate integration may use only the
  explicit attended-smoke time window `[2024-12-01T00:00:00Z,
  2025-06-01T00:00:00Z)` from V46 TRAIN. It is 32,289 chronological rows,
  never a uniform TRAIN sample. Its recipe/session binds the UTC boundaries;
  its terminal private report must prove exact liveness, all joint-task
  supervision/gradients and parameter movement. It produces no bundle and has
  no validation, TEST, backtest, candidate, demo or live authority.
- No tick-resolution feature, dataset, Exit evaluation or trading claim exists;
  the current Exit input clock is native closed M1.
- The TRAIN-fit squeeze owner and fail-closed six-clock artifact plumbing are
  production-integrated in source, and six immutable TRAIN artifacts have been
  fitted and admitted since 2026-08-18. **No manifest path or hash is restated
  here** (rule 13): several six-clock sets exist on disk with different
  `contract_sha256`, this file named a superseded one until 2026-08-19, and the
  binding must be resolved from the run's own V4 cache manifest. The 2026-08-15
  set fails closed at load. Rebuild caches/surfaces/dataset and retrain before
  making any model or edge claim.
- Canonical pair generation `53cba459...4668f7` (2026-08-20) is current source
  authority. Resolve its complete identity and generation-local manifest from
  `PROJECT_STATE_xau_direction_launch.json`; this file deliberately restates no
  path or hash. Earlier generations remain history, not current build input.
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
  stopped safely. Source/lifecycle fixes invalidate V8/V13/V18, and terminal
  V34/V35/V36/V37/V38/V39 products are also forbidden resume input.

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
Canonical CUDA additionally stops above 70 C core, 220 W actual draw or 12 GiB
resident VRAM from pinned native WSL telemetry. The Windows-host driver rejected
a physical lower power limit from WSL, so 220 W is a one-second process stop,
not a throttle. The completed V46 batch-8 smoke published a diagnostic bundle;
the exact VAL evaluator inherits the same guard before its smoke-bundle audit.
A remote GPU
may be prepared only as offline research from the frozen commit and V46 hashes,
and only after explicit cost approval; it receives no broker credential or live
route.

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
2. Preserve V46's sealed TEST. The immutable VAL prediction and CPU audit now
   exist; repair the remaining smoke learning-evidence gate only with a bounded
   preflighted learning-validation probe. Train a full research candidate only
   if that follow-up audit proves safe and valid.
3. Evaluate the candidate on the historical research path (VAL and then the
   untouched TEST only where its seal permits it); report that result as
   research-only, never as a production edge claim.
4. Separately bind causal executable prices, costs, financing, gap/terminal
   treatment and portfolio constraints before any production-net claim, demo,
   paper or live route. Do not rebuild V46 merely to change a report-only
   consumer.
5. Fit allowed calibration, freeze the candidate and run the same bundle's
   unified Exit replay only after the relevant research gates pass.

Run `bash scripts/gx1_handover.sh` whenever authority or status changes.
