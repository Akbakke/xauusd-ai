# Entry Foundation Audit - 2026-06-28

Status: active seq146 foundation is activated and post-apply refreshed. Entry
smoke/candidate/IQL evidence is available, but shadow/live/promotion remain
closed. Exit-side work is now in report-only foundation mode: active
Entry-bound per-bar reconstruction, state/reward contract and split/leakage
audit are ready; Exit Transformer/IQL training remains closed until explicit
Exit architecture, training and replay evidence gates are added and pass. The
active Exit model dataset/readiness gate is also ready, with train-only
normalization metadata and train/val/test shards. The active Exit Transformer
architecture/readiness contract is ready, and the active Exit Transformer
training-plan/readiness manifest plus fail-closed trainer-wrapper readiness are
ready for trainer-core/pretrain-manifest implementation review. No Exit trainer
is approved or allowed.

The original rule still applies: foundation pass required before any more Transformer training.

## Decision

Do not run generic model training, candidate training, replay, IQL, shadow or
live from old artifacts.

Do not treat foundation work as feature-count decoration. The active foundation
must become one coherent multi-timeframe market-state contract across structure,
SMC/liquidity, trend, volatility, momentum, session/regime, spread/ATR,
support/resistance and Exit-relevant context. The goal is tradable edge proven
by replay/PnL, drawdown, MAE, bad-path, path quality and regime/session slices,
not broad direction averages.

The product objective is a fully automated XAUUSD bot that can enter long or
short at high-quality points and exit near maximum profit opportunity. That
requires the Entry Transformer "eyes", Entry IQL "brain", Exit Transformer
timing model and Exit IQL policy to share one grounded multi-timeframe feature
language. HH/HL/LH/LL, SMC/liquidity, momentum/flow, trend/EMA,
volatility/compression, session/regime, spread/ATR, support/resistance and
Entry-to-Exit path context must be calibrated as one system, not optimized as
isolated signals. All new gates must measure tradable edge and failure slices;
no broad average may hide weak session, regime, side, bad-path or tail behavior.

`<id>` in `--vedtak <id>` is a deliberate human approval token, not a report id
that the scripts invent automatically. Activation apply requires the
`ENTRY_FOUNDATION_ACTIVATE_` prefix, for example
`ENTRY_FOUNDATION_ACTIVATE_20260629_DIRECTIONAL_SMC_V1`. Post-apply refresh
requires a separate `ENTRY_FOUNDATION_POST_APPLY_` token. These vedtak ids open
only the specific reviewed mutation they are passed to; they do not approve
training, staging, replay, IQL, shadow, live or promotion.

The previous Transformer runs proved that the pipeline can consume `seq=(96,89)`,
but they did not prove that the feature/target foundation is sufficient for
lower drawdown. Those artifacts are frozen as diagnostic history.

The active path is now:

`handover -> verify -> selftest -> foundation-guardrails -> readiness-report -> Entry evidence gates -> Entry/IQL replay and slice evidence -> Entry-to-Exit handoff -> active Exit per-bar reconstruction -> active Exit state/reward contract -> active Exit split/leakage audit -> active Exit model dataset/readiness gates -> active Exit Transformer architecture/readiness -> active Exit Transformer training plan/readiness -> fail-closed Exit Transformer trainer-wrapper readiness -> only then consider Exit trainer-core/pretrain-manifest/replay/IQL evidence gates`

The historical activation path was:

`handover -> verify -> selftest -> foundation-guardrails -> foundation-adoption-candidate -> foundation-activation-plan -> foundation-activation-apply --dry-run -> worktree-hygiene -> optional stage-foundation-cleanup --apply --vedtak <id> -> train-readiness -> if foundation_activation_required_before_smoke=true: foundation-activation-apply --apply --vedtak <id> -> foundation-activation-post-apply --apply --vedtak <id> -> train-readiness -> optional smoke-manifest --vedtak <id> -> smoke-train --vedtak <id> --require-edge-audit`

Use `scripts/entry_next_edge_control.sh handover` as the default orientation
command for a fresh session. It prints the active seq146 operating point, runs
foundation verification, summarizes readiness, and keeps the historical legacy
handover behind the explicit `GX1_ALLOW_LEGACY_HANDOVER=20260627_ALLOW_LEGACY_HANDOVER`
token.
Use `scripts/entry_next_edge_control.sh readiness-report` when you need a
non-training status refresh across train-readiness, candidate-readiness,
replay-readiness, the latest IQL distillation contract, IQL replay evidence and
IQL replay comparison. It is report-only: it must not stage, train, replay,
distill, shadow or touch live paths. It also
prints the latest worktree-hygiene dirty/stage/hold counts, stage-ready/safe
flags, critical-gate path coverage, post-stage status and stage/hold path lists
so cleanup can be reviewed without changing the git index. It prints the canonical
`stage-foundation-cleanup --dry-run` and `stage-foundation-cleanup --apply --vedtak <id>`
commands first; the raw `git add --pathspec-from-file=...` command is audit
detail only.
The same report ends with an `allowed now` / `optional proof commands` /
`blocked now` policy summary so a fresh session can distinguish commands that
can be executed immediately from proof-only commands that still need an
explicit vedtak and real training gates without inferring from individual
failures. Use
`scripts/entry_next_edge_control.sh readiness-report --json` for the same
status as machine-readable JSON. The JSON includes `status_summary` booleans
for foundation readiness, smoke-manifest proof eligibility, real smoke-train
eligibility, foundation activation requirement and exact vedtak-gated
activation apply command, candidate/IQL availability, IQL replay-evidence
readiness, promotion-review eligibility and current blockers.
Both text and JSON readiness reports are allowed report-only commands while
training remains blocked.
The JSON also includes a `commands` object with clean `argv` arrays and
`allowed`/`mode` metadata for automation. It covers the safe now-commands
(`verify`, `selftest`, `foundation-guardrails`, readiness reports and stage
dry-run) and the blocked downstream paths (`candidate-train`,
`selective-edge`, `replay-evidence`, `iql-distill`, IQL replay/compare,
shadow and live). `execution_allowed_now=true` means the command can be run
immediately with no vedtak placeholder and no side effect.
`allowed_after_explicit_vedtak=true` means the structural gate is open after a
real vedtak id is supplied. Inline comments in human-readable command lists
must not be parsed as executable command text. Each command also declares
whether it requires vedtak, requires clean git, mutates the git index, starts a
trainer, starts replay, starts IQL distillation or touches shadow/live paths,
plus `not_executable_now_reason` when it is deliberately not a now-command.
Use `scripts/entry_next_edge_control.sh smoke-manifest --vedtak <id>` when you
need the pre-train manifest/provenance proof while the worktree is still dirty.
It runs the real manifest preflight and writes the manifest, but must stop
before trainer start.

`train-readiness` emits `READY_FOR_VEDTAK_SMOKE_TRAIN` only when the foundation
contract and execution hygiene both pass. In a dirty worktree it emits
`READY_FOR_VEDTAK_SMOKE_TRAIN_AFTER_GIT_CLEAN`; that means the foundation
contract is ready, but real trainer start is still blocked until git is clean.
When the active canonical foundation is stale but a green adoption candidate,
activation plan and activation-apply dry-run exist, it remains `NOT_READY` and
sets `foundation_activation_required_before_smoke=true`. In that state the
next command is the vedtak-gated canonical path switch:

`scripts/entry_next_edge_control.sh foundation-activation-apply --plan-json <activation-plan> --apply --vedtak <id>`

This activation apply does not train. It mutates only the canonical foundation
dataset path, then `foundation-activation-post-apply --apply --vedtak <id>`
must run the recorded post-apply commands to refresh the canonical feature,
target and specialist audits, materialize the canonical smoke dataset, run
`verify --quiet`, and then rerun `train-readiness`. If activation apply has
already reported
`APPLIED_ALIAS_SWITCH` but post-apply refresh is not completed,
`train-readiness` must keep `NOT_READY` and point `next_allowed_command` at
`foundation-activation-post-apply --apply --vedtak <id>`.
The next real command after both vedtak and clean-git readiness is:

`scripts/entry_next_edge_control.sh smoke-train --vedtak <id> --require-edge-audit`

This does not approve a candidate, replay, IQL distillation, shadow or live
path. Those gates remain closed until the required post-smoke bundle audit,
candidate-readiness, selective-edge/no-XGB evidence, replay-readiness,
   distillation contract, IQL replay evidence, IQL comparison, IQL slice/tail
   audit and Entry-to-Exit handoff readiness gates pass.

After IQL replay comparison, `scripts/entry_next_edge_control.sh
iql-slice-audit` must prove that the edge survives supported session, regime
and side slices, and must record exit-opportunity diagnostics from the exact
candidate/IQL replay trade logs. `scripts/entry_next_edge_control.sh
entry-exit-materialize` may materialize the active Entry-bound per-bar
HOLD/EXIT_NOW substrate from IQL replay trades and canonical M5 bid/ask bars,
with optional hashed M1-to-M5 supplement for real price gaps. It must never
synthesize missing bars; trades with unresolved per-bar gaps must be excluded
into an explicit gap manifest. It must not train, replay, distill, shadow or
touch live paths.
`scripts/entry_next_edge_control.sh entry-exit-handoff` is report-only and
must keep Exit Transformer/IQL training closed until that substrate exists with
explicit trade, bar-state, entry-context and replay-identity fields and has
been audited.

## Current Operating Status - 2026-06-30

Active status:

- Foundation activation apply completed.
- Foundation post-apply refresh completed.
- Worktree hygiene: `PASS_CLEAN_GIT`.
- Train-readiness: `READY_FOR_VEDTAK_SMOKE_TRAIN`.
- Entry-to-Exit per-bar handoff: `PASS_WITH_EXPLICIT_GAP_EXCLUSIONS`.
  The active materializer fills missing canonical M5 `atr_bps` deterministically
  from closed-bar bid/ask OHLC true range and excludes unresolved/non-contiguous
  price gaps instead of synthesizing bars.
- Active Exit per-bar reconstruction audit:
  `READY_FOR_EXIT_STATE_REWARD_CONTRACT_REVIEW`.
- Active Exit state/reward contract:
  `ENTRY_EXIT_STATE_REWARD_CONTRACT_READY`.
- Active Exit split/leakage audit:
  `ENTRY_EXIT_SPLIT_LEAKAGE_AUDIT_READY`.
- Active Exit model dataset/readiness:
  `ENTRY_EXIT_MODEL_DATASET_READY_FOR_EXIT_TRANSFORMER_READINESS_REVIEW`.
- Active Exit Transformer architecture/readiness:
  `ENTRY_EXIT_TRANSFORMER_ARCHITECTURE_READY_FOR_TRAINING_PLAN_REVIEW`.
- Active Exit Transformer training plan/readiness:
  `ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READY_FOR_VEDTAK_REVIEW`.
  It pins 5906 rows, 413 episodes, the exact Exit heads
  `exit_now_logit`, `hold_value_bps`, `exit_now_reward_bps`,
  `giveback_risk_bps`, `mfe_capture_ratio`, shard hashes, train vedtak prefix
  `ENTRY_EXIT_TRANSFORMER_TRAIN_` and RAM guardrails with `num_workers=0`,
  initial batch size 32 and max process RSS 8 GiB.
- Active Exit Transformer trainer wrapper readiness:
  `ENTRY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READY_FOR_IMPLEMENTATION_REVIEW`.
  The wrapper is executable and fail-closed: missing vedtak and wrong vedtak
  prefix both reject before side effects, the implementation flag remains
  disabled, and the future train path declares cgroup RAM caps plus
  `--num-workers 0`.
- Exit Transformer training, Exit IQL, shadow, live and promotion remain
  closed. The next safe work is implementing active Exit Transformer trainer
  core plus a pretrain-manifest audit; it remains closed until those gates,
  clean-git checks, RAM guard and an explicit Exit train vedtak exist.

## Previous Operating Status - 2026-06-29

Directional-SMC candidate status:

- Candidate dataset:
  `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_foundation_seq146_neutral_20260629_directional_smc`
- Candidate smoke dataset:
  `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_foundation_seq146_smoke_20260629_directional_smc`
- Candidate feature, target and specialist audits are PASS under
  `foundation_seq146_20260629_directional_smc`.
- Adoption candidate report:
  `/home/andre2/GX1_DATA/reports/entry_foundation_adoption_candidate_20260629_v1/foundation_seq146_20260629_directional_smc/ENTRY_FOUNDATION_ADOPTION_CANDIDATE_latest.json`
  - Decision: `PASS`.
  - `candidate_ready_for_activation=true`.
  - `activation_allowed_without_vedtak=false`.
  - Candidate dataset, feature audit, target audit, specialist audit, smoke
    dataset and artifact-fingerprint gates must all be `PASS`.
- Activation plan report:
  `/home/andre2/GX1_DATA/reports/entry_foundation_activation_plan_20260629_v1/foundation_seq146_20260629_directional_smc/ENTRY_FOUNDATION_ACTIVATION_PLAN_latest.json`
  - Decision: `READY_FOR_VEDTAK_ACTIVATION`.
  - Strategy: `canonical_active_alias_then_canonical_audit_refresh`.
  - Must re-hash the adoption feature-audit, target-audit, specialist-audit and
    smoke-dataset-manifest fingerprints against the current files.
  - Must include source-pointer checks for activation apply, post-apply refresh
    and `entry_next_edge_control.sh`, plus an activation step that runs
    `verify --quiet` before `train-readiness`.
- Activation apply dry-run report:
  `/home/andre2/GX1_DATA/reports/entry_foundation_activation_apply_20260629_v1/foundation_seq146_20260629_directional_smc/ENTRY_FOUNDATION_ACTIVATION_APPLY_latest.json`
  - Decision: `READY_FOR_VEDTAK_APPLY`.
  - `mutation_performed=false`.
  - Rejects stale activation plans that do not include the active verify step or
    the current adoption artifact contract and apply/post-apply/control
    source-pointer checks.
  - `post_apply_commands` contains canonical feature audit refresh, target
    audit refresh, specialist audit refresh, canonical smoke dataset refresh
    `verify --quiet` and `train-readiness`.
- Post-apply refresh command:
  `scripts/entry_next_edge_control.sh foundation-activation-post-apply --activation-apply-json <activation-apply-report> --apply --vedtak <id>`
  - Requires its own `ENTRY_FOUNDATION_POST_APPLY_<id>` vedtak.
  - Refuses apply until activation apply has `decision=APPLIED_ALIAS_SWITCH`
    and `mutation_performed=true`.
  - Does not train, replay, distill, shadow or touch live paths.
  - `train-readiness` must require guardrail proof that post-apply dry-run is
    allowed now and post-apply apply remains blocked without activation apply
    plus explicit post-apply vedtak.

Canonical active status:

- Active foundation dataset path remains:
  `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_foundation_seq146_neutral`
- The active path is still a real directory, not the activation symlink.
- Active feature audit is intentionally `FAIL` until activation because the
  active split manifests lack
  `entry_foundation_structure_v1_20260629_directional_smc_pressure` metadata.
- `train-readiness` decision is `NOT_READY` with
  `foundation_activation_required_before_smoke=true`,
  `foundation_activation_apply_required_before_smoke=true`,
  `foundation_activation_post_apply_required_before_smoke=false`,
  `activation_apply_ready=true`, `activation_apply_mutation_performed=false`
  and `smoke_training_allowed_with_explicit_vedtak=false`.

## Previous Operating Status - 2026-06-28

Active foundation dataset:

- `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_foundation_seq146_neutral`
- Sequence/snap input dimension: `146`.
- Chart-structure extension dimension: `105`.
- Context: `ctx_cont=142`, `ctx_cat=5`.
- XGB bridge: neutral compatibility bridge only; not an active edge source.

Machine PASS artifacts:

- Feature audit:
  `/home/andre2/GX1_DATA/reports/entry_feature_foundation_audit_20260628_v1/foundation_seq146/ENTRY_FEATURE_FOUNDATION_AUDIT_latest.json`
- Target audit:
  `/home/andre2/GX1_DATA/reports/entry_target_foundation_audit_20260628_v1/foundation_seq146/ENTRY_TARGET_FOUNDATION_AUDIT_latest.json`
- Active feature and target audit latest files are intentionally scoped under
  `foundation_seq146`; top-level/legacy diagnostic latest files must not be used
  for smoke-readiness.
- Specialist feature group audit:
  `/home/andre2/GX1_DATA/reports/entry_specialist_feature_group_audit_20260628_v1/ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.json`
- Foundation guardrails:
  `/home/andre2/GX1_DATA/reports/entry_foundation_guardrails_20260628_v1/ENTRY_FOUNDATION_GUARDRAILS_latest.json`
- Worktree hygiene:
  `/home/andre2/GX1_DATA/reports/entry_foundation_worktree_hygiene_20260628_v1/ENTRY_FOUNDATION_WORKTREE_HYGIENE_latest.json`
  - Stage candidates:
    `/home/andre2/GX1_DATA/reports/entry_foundation_worktree_hygiene_20260628_v1/ENTRY_FOUNDATION_WORKTREE_STAGE_PATHS_latest.txt`
  - Review/hold paths:
    `/home/andre2/GX1_DATA/reports/entry_foundation_worktree_hygiene_20260628_v1/ENTRY_FOUNDATION_WORKTREE_REVIEW_HOLD_PATHS_latest.txt`
  - Stage status table:
    `/home/andre2/GX1_DATA/reports/entry_foundation_worktree_hygiene_20260628_v1/ENTRY_FOUNDATION_WORKTREE_STAGE_STATUS_latest.tsv`
  - Review/hold status table:
    `/home/andre2/GX1_DATA/reports/entry_foundation_worktree_hygiene_20260628_v1/ENTRY_FOUNDATION_WORKTREE_REVIEW_HOLD_STATUS_latest.tsv`
  - Git add dry-run proof:
    `/home/andre2/GX1_DATA/reports/entry_foundation_worktree_hygiene_20260628_v1/ENTRY_FOUNDATION_WORKTREE_GIT_ADD_DRY_RUN_latest.txt`

The active seq146 contract includes the roadmap families requested for the
foundation rebuild:

- HH/HL/LH/LL state.
- BOS/CHoCH age and break recency.
- Sweep reclaim and false-breakout follow-through.
- Compression-expansion triggers.
- Impulse/pullback phase.
- Session x structure interactions.
- Machine liveness/audit by train/val/test split.

No actual smoke training, candidate training, replay, IQL distillation, shadow
or live process is approved by this document alone. The control surface remains
`scripts/entry_next_edge_control.sh`.

## Archived Pre-Foundation Feature Contract

Dataset:

- Run: `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_seq_structure_neutral`
- Status: archived/frozen after audit. The dataset contract below documents
  what was inspected; the original parquets, manifests and diagnostic bundles
  have been moved under `_LEGACY_PRE_FOUNDATION_DO_NOT_USE_20260628/`.
- Manifest created: `2026-06-28T09:23:33Z`
- Build git commit: `36296e162220f1779b2a19b3ea067f21b9e5f6c2`
- Train: 2020-11-09 to 2025-09-30, 350,219 rows
- Val: 2025-10-01 to 2025-12-31, 17,646 rows
- Test: 2026-01-02 to 2026-06-26, 34,030 rows
- Sequence shape: `96 x 89`
- Context shape: `ctx_cont=142`, `ctx_cat=5`
- XGB bridge: neutral compatibility bridge, not an active edge source
- Manifest feature contract says `XGB_SIGNAL_BRIDGE_V2`; exported Transformer
  bundle remains runtime-compatible as `XGB_SIGNAL_BRIDGE_V1` while carrying
  `neutral_xgb_bridge=true`.

The emitted 89 sequence features include:

- Base sequence: 41 fields.
- Chart-structure extension: 48 fields.
- SMC base: `smc_swing_state`, `smc_bos_up`, `smc_bos_down`, `smc_choch`,
  `smc_sweep_up`, `smc_sweep_down`, `smc_sweep_size_atr`,
  `smc_bars_since_sweep`, `smc_premium_discount`.
- Wick/price/volatility: `wick_asym`, `body_pct`, returns, volatility z-scores,
  EMA position/slope, squeeze and bandwidth features.
- Chart-extension features: 48 additional interaction features including BOS,
  wick-level, EU-session, level-proximity, pullback, liquidity, D1/H1 vol/ATR and
  compression interactions.

Notable emitted chart-extension examples:

- `chart.eu_x_hh`, `chart.eu_x_bos`, `chart.bos_x_level_prox`,
  `chart.bos_x_ema50_200`, `chart.bos_x_price_vs_ema200`,
  `chart.bos_x_tf_agreement`, `chart.bos_x_d1_atr_pct`.
- `chart.wick_level_x_level_prox`, `chart.wick_x_major_level`,
  `chart.wick_level_x_h1_vol_pct`, `chart.wick_level_x_vol_stack`.
- `chart.is_eu_only_x_pullback`, `chart.eu_x_pullback`,
  `ctx_cont.struct_pullback_depth_m5_v3`.
- `ctx_cont.H1_range_compression_ratio`,
  `ctx_cont.smc_sweep_bull_pressure_last48`,
  `ctx_cont.liquidity_hi_nearest_abs_atr`.

The 142 context features include:

- Session: Asia/EU/US/overlap timing and tradability flags.
- Multi-TF structure: continuation, pullback, bounce, pullback-depth for M5,
  M15, H1, H4, D1.
- SMC aggregations: CHoCH recency, BOS pressure, sweep pressure/recency,
  premium/discount extremes.
- Liquidity/pivots: nearest support/resistance, R/S proximity, liquidity high/low
  distance, D1/H1/H4/M15 structure.
- Regime: ATR ratios, trend age, D1 regime transitions, TF agreement/divergence.

Observed liveness from snap arrays:

- No non-finite values in train/val/test.
- One expected constant slot in train: `margin_top1_top2` from neutral bridge.
- Six constant neutral bridge slots in val/test: `p_long`, `p_short`, `p_flat`,
  `p_hat`, `uncertainty_score`, `margin_top1_top2`.
- BOS/CHoCH/sweep features are non-empty:
  - Train `smc_bos_up=19.6%`, `smc_bos_down=17.0%`, `smc_choch=0.11%`,
    sweeps about `5.6-5.7%`.
  - Val `smc_bos_up=22.7%`, `smc_bos_down=16.5%`, `smc_choch=0.02%`,
    sweeps about `2.9-3.1%`.
  - Test `smc_bos_up=18.1%`, `smc_bos_down=16.7%`, `smc_choch=0.03%`,
    sweeps about `6.2-6.3%`.
- `smc_choch` is very sparse, so CHoCH should probably be represented with
  age/recency/pressure features rather than only a one-bar event flag.
- `chart.sweep_size_x_vol_stack` is effectively always non-zero in val/test
  because it is an interaction with continuous volatility state; it should be
  checked for interpretability before being trusted as a pure sweep feature.

## Pre-Foundation Gaps That Triggered Seq146 Rebuild

The current feature layer is not empty or naive, but it is not yet a complete
chart-structure foundation.

Missing or under-explicit sequence features:

- Direct HH/HL/LH/LL state as separate model inputs, not only compressed
  `smc_swing_state` and interaction features.
- BOS age, CHoCH age, and bars-since-last-break per side.
- Explicit sweep-reclaim / false-breakout follow-through features in the emitted
  sequence list.
- Compression-to-expansion trigger features, not only compression ratios.
- Impulse/pullback phase and impulse age as first-class sequence inputs.
- Session x structure features for all major sessions, not mostly EU-focused
  interactions.
- Feature liveness report should be a required artifact before training.

## Current Targets

Direction target:

- `y_direction`: 0=long, 1=short, 2=flat.
- Final emitted label uses spread-aware final PnL at `H=24` bars.
- Long/short if side PnL at horizon is at least `15 bps`; otherwise flat.

Path targets:

- `path_quality_bps`: `MFE - MAE` over `H=10`.
- `y_bad_path`: losing side at `H=10`.
- `clean_edge`: MFE >= 14 bps, MAE <= 4 bps, path >= 16 bps.
- `survival`: MFE >= 8 bps, MAE <= 6 bps, path >= 8 bps.

Observed label rates:

- Train direction: long `19.3%`, short `17.6%`, flat `63.1%`.
- Val direction: long `33.0%`, short `25.2%`, flat `41.8%`.
- Test direction: long `32.5%`, short `32.0%`, flat `35.5%`.
- `y_bad_path` rises from `11.6%` train to `19.9%` test.
- `y_bad_path` correctly has negative rank correlation with path quality:
  train `-0.128`, val `-0.186`, test `-0.201`.
- Tradable rate: train `36.9%`, val `58.2%`, test `64.5%`.
- `path_quality_bps` distribution:
  - Train mean/p10/p50/p90: `4.497 / 0.000 / 0.000 / 19.940`.
  - Val mean/p10/p50/p90: `9.193 / -8.542 / 0.000 / 37.795`.
  - Test mean/p10/p50/p90: `14.058 / -11.278 / 0.000 / 50.764`.
- This is a material distribution shift: val/test are much more directional
  and tradable than train. That must be handled explicitly in validation.

## Training Diagnostics From Aborted Path

These runs are diagnostic only. They are not approved candidates.

- Neutral sequence run, stopped after epoch 3:
  - Contract: `seq=(512,96,89)`, `ctx_cont=142`, neutral XGB bridge.
  - Best observed epoch: val loss `5.205`, dir acc `0.408`.
  - Issue: `bad_path_w=0.000`, so tail-risk was not actively trained.
  - Issue: bad-path diagnostic was anti-targeted, positive correlation with
    path quality.
- Tail-risk v1, 4 epochs:
  - Settings: `ENTRY_AUX_BAD_PATH_WEIGHT=0.80`, strong bad-path weighting,
    clean/survival/rank still active.
  - Final val loss `6.121`, dir acc `0.421`.
  - Issue: bad-path still anti-targeted, Spearman `+0.097` versus path quality.
  - Issue: clean-edge and survival collapsed/redundant, cross-head rho about
    `+0.979`.
- Tail-risk v2 symmetric, 4 epochs:
  - Settings: `ENTRY_SYMMETRIC_NEGATIVES=1`, bad-path weight `0.35`,
    bad-path pos-weight cap `2.0`, clean/survival/rank disabled.
  - Final val loss `4.610`, dir acc `0.454`.
  - Export/strict-load/liveness passed.
  - Issue: bad-path still anti-targeted, Spearman `+0.099`; do not use this
    head as a gate.

Interpretation:

- The current sequence pipeline is technically alive.
- XGB is not the active edge in these runs.
- The first blocking problem is foundation quality: feature contract,
  target contract and replay objective alignment.
- The second blocking problem is tail-risk target/head design. The observed
  bad-path label is semantically valid in the dataset, but the model head learned
  the wrong sign in short training runs.

## Legacy Cleanup - 2026-06-28

Purpose: remove old pre-foundation artifacts from active discovery paths so a
script cannot accidentally train, evaluate, replay or promote the wrong bundle.

No files were deleted in this pass. Artifacts were moved into explicit legacy
directories with `README_DO_NOT_USE.md` markers.

Run-root freeze markers:

- `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/README_FOUNDATION_FREEZE.md`
- `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626/README_FOUNDATION_FREEZE.md`
- `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_seq_structure_neutral/README_FOUNDATION_FREEZE.md`
- `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/README_FOUNDATION_FREEZE.md`

Legacy locations:

- `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/_LEGACY_PRE_FOUNDATION_DO_NOT_USE_20260628/`
  - old `entry_iql_*` runs/logs
  - old `forward_outcome_*` artifacts/logs
  - old `phase6_lam50`
  - old root `xgb_v7`
  - old root `v10_bundle_clean`
  - old root dataset build logs
  - old `v10_6yr_rebuild_20260626_bodyfix`
- `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_seq_structure_neutral/_LEGACY_PRE_FOUNDATION_DO_NOT_USE_20260628/`
  - old `v10_dataset_seq_structure_neutral__HOLD_03B_*` parquets/manifests
  - old `DATASET_BUILD_PROOF.json`
  - diagnostic Transformer bundles:
    `transformer_seq_structure_neutral_smoke`,
    `transformer_seq_structure_neutral_tailrisk_bp080_e4_seed1337`,
    `transformer_seq_structure_neutral_tailrisk_v2_sym_e4_seed1337`
- `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626/_LEGACY_PRE_FOUNDATION_DO_NOT_USE_20260628/`
  - old `v10_bundle_6yr`
  - old `v10_dataset_6yr`
  - old `v10_dataset_6yr_corrupt_1782468340`
- `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix/_LEGACY_PRE_FOUNDATION_DO_NOT_USE_20260628/`
  - old `v10_bundle_*`
  - old `v10_dataset_*`
  - old `xgb_v7_fixed_h24_15bps*`
- `/home/andre2/GX1_DATA/reports/_LEGACY_ENTRY_PRE_FOUNDATION_DO_NOT_USE_20260628/`
  - old `entry_chart_structure*`
  - old `entry_exit_retimer*`
  - old `entry_feature_drawdown_atlas*`
  - old `entry_feature_exit_retimer*`
  - old `entry_transformer_feature_audit*`
  - old `entry_transformer_feature_diagnostics*`
  - old `entry_feature_interaction_tail*`
  - old `entry_feature_tail_exit_deep_audit*`
  - old `entry_residual_tail_fast*`
  - old `entry_risk_overlay_hh_bos*`
  - old `entry_selective_edge_20260627*`
  - old `entry_tabular_no_xgb_candidates`
  - old `sequence_structure_builder_smoke_20260628_v1`
  - old `online_iql`
  - old `truth_e2e_sanity`
  - old `v12_paper_runs` report tree containing pre-foundation Entry inference
    and candidate-gate outputs
- `/home/andre2/GX1_DATA/data/_LEGACY_ENTRY_PRE_FOUNDATION_DO_NOT_USE_20260628/`
  - old `entry_v10_ctx_datasets`
  - old `data/data/training/entry_v10_ctx`
  - old `data/data/trainsets/entry_v10_ctx_6x6`
- `/home/andre2/GX1_DATA/models/_LEGACY_ENTRY_PRE_FOUNDATION_DO_NOT_USE_20260628/`
  - old `models/entry_v10_ctx`
  - old `models/xgb_v7_base80_20260526T052210Z`
- `/home/andre2/GX1_DATA/_LEGACY_ENTRY_PRE_FOUNDATION_DO_NOT_USE_20260628/`
  - old root `ENTRY_DECISION_DEEP_ANALYSIS_2026-05-21.md`
- `/home/andre2/GX1_DATA/runs/FASE2B_CLEAN_20260608/_LEGACY_ENTRY_PRE_FOUNDATION_DO_NOT_USE_20260628/`
  - old `entry_decisions*`
  - old `entry_iql_*.log`
- `/home/andre2/GX1_DATA/runs/TROUGH_REBUILD_20260611/_LEGACY_ENTRY_PRE_FOUNDATION_DO_NOT_USE_20260628/`
  - old `entry_iql_trough_20260611`
  - old `entry_iql_trough.log`
- `/home/andre2/GX1_DATA/runs/_OPENMORE_LADDER_20260613/_LEGACY_ENTRY_PRE_FOUNDATION_DO_NOT_USE_20260628/`
  - old `entry_decisions_takeall`
- `/home/andre2/GX1_DATA/runs/_REPLAY_2026_20260613/_LEGACY_ENTRY_PRE_FOUNDATION_DO_NOT_USE_20260628/`
  - old `ENTRY_IQL_INFERENCE_FOR_V12_*`

Cleanup logs:

- `/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/LEGACY_CLEANUP_20260628.log`
- `/home/andre2/GX1_DATA/reports/LEGACY_ENTRY_REPORTS_CLEANUP_20260628.log`

Remaining active inputs:

- Source/cache files needed for inspection or rebuild were left in place, for
  example `FULL_PLUS_CTX_v3src.parquet`, `cv3`, canonical feature files and
  `MULTI_TF_V2_CACHE`.
- These are not approved training bundles. They are rebuild inputs only until
  the feature and target foundation gates pass.

Post-cleanup verification:

- A recursive check under `/home/andre2/GX1_DATA`, excluding legacy paths,
  returned no matches for the highest-risk old artifact names:
  `entry_*`, `ENTRY_*`, `entry_v10*`, `v10_bundle*`, `v10_dataset*`,
  `transformer_seq_structure*`, and `*tailrisk*`.
- The current report root still contains non-Entry or source/foundation reports
  such as `sequence_feature_promotion_20260628_v1`,
  `sequence_structure_feature_layer_20260628_v1`,
  `sequential_feature_coverage_20260628_v1`, `v12_live_data`,
  `online_replay`, `exit_feature_alignment_20260628_v1` and
  `xgb_fixed_label_smoke_20260626`.
- No active Entry training/eval/replay process was found before the second move
  pass; `pgrep` only matched the verification commands themselves.

## Target Gaps

The labels are defined, but the business objective is not yet encoded tightly enough.

Main issue:

- Training loss optimizes direction/classification and auxiliary path heads.
- The real objective is 2026 replay quality: positive expectancy, lower drawdown,
  stable session behavior, controlled tail losses.
- Therefore validation accuracy alone is not a go/no-go metric.

Required acceptance gates:

- Feature gate: no non-finite inputs, no unexpected constant features, all declared
  chart-structure features have liveness by split, and no lookahead leakage.
- Target gate: direction, bad-path, path-quality, clean-edge and survival rates must
  be reported by split, side and session.
- Distribution-shift gate: train/val/test shifts in tradable rate, direction mix,
  bad-path rate and path-quality distribution must be explicit in the report.
- Model gate: test/replay must beat majority-label and neutral-skip baselines, and
  must not rely on neutral XGB bridge slots.
- Tail-risk gate: `bad_path_prob` must rank-correlate negatively with realized
  `path_quality_bps` and separate high-MAE/tail-loss rows before it can be used
  as a pre-gate or sizing gate.
- Trading gate: 2026 offline replay must improve max drawdown and tail loss versus
  the current baseline; win rate alone is not sufficient.
- Robustness gate: results must be split by Asia/EU/US/overlap, side, volatility
  regime and trend regime before any shadow/live work.

## Next Work Order

1. Do not use archived seq96/89 pre-foundation datasets or bundles.
2. Keep using `scripts/entry_next_edge_control.sh` as the canonical control
   surface. Start fresh sessions with
   `scripts/entry_next_edge_control.sh handover`; it must report
   `active Entry foundation seq146` and must not fall through to the historical legacy
   handover unless the explicit legacy env token is set.
   `scripts/entry_next_edge_control.sh verify`/`selftest` must also cover the
   readiness-policy snapshot and critical-gate path coverage source contracts.
   Use `scripts/entry_next_edge_control.sh readiness-report` for a full
   fail-open readiness snapshot while the current step is still blocked.
3. Run `scripts/entry_next_edge_control.sh foundation-guardrails` to prove
   legacy shadow/live entrypoints fail closed. It must also parse a
   non-refreshing readiness-policy JSON snapshot and prove safe audit/dry-run
   commands are `execution_allowed_now=true` while candidate train,
   replay-evidence, IQL, shadow and live paths remain blocked with explicit
   side-effect metadata.
4. Run `scripts/entry_next_edge_control.sh worktree-hygiene --no-fail-on-dirty`
   to classify the current dirty worktree into active foundation, legacy
   tombstone cleanup, entry-related review and unrelated review paths. Real
   smoke training still requires the final decision `PASS_CLEAN_GIT`. Use
   `ENTRY_FOUNDATION_WORKTREE_STAGE_PATHS_latest.txt` as the foundation cleanup
   candidate list and keep `ENTRY_FOUNDATION_WORKTREE_REVIEW_HOLD_PATHS_latest.txt`
   out of the cleanup commit unless separately reviewed. The audit also writes
   `ENTRY_FOUNDATION_WORKTREE_GIT_ADD_DRY_RUN_latest.txt` to prove the stage
   pathspec is valid without changing the git index. Treat the stage list as
   usable only when the JSON report has `stage_plan_safe=true`, zero stage/hold
   overlap and zero dry-run hold overlap. Review
   `ENTRY_FOUNDATION_WORKTREE_STAGE_STATUS_latest.tsv` and
   `ENTRY_FOUNDATION_WORKTREE_REVIEW_HOLD_STATUS_latest.tsv` for file status,
   size and line-count summaries before any actual staging. The same JSON must
   also have `foundation_cleanup_review_decision=PASS`, proving all required
   foundation cleanup paths are either clean or included in the stage candidate
   list. It must also report `foundation_cleanup_critical_gate_review` for the
   docs, control surface, smoke bundle audit, candidate-readiness,
   replay-readiness, IQL distillation/evidence/comparison gates and their
   tests, so those gate files cannot be omitted from the staged foundation
   cleanup.
   Only use the reported `foundation_cleanup_stage_command` when
   `foundation_cleanup_stage_ready=true`. After any actual staging, rerun
   `worktree-hygiene --no-fail-on-dirty` and require
   `foundation_cleanup_post_stage_verification.decision=PASS_STAGED`; before
   staging the expected decision is `NOT_STAGED`.
   The control-surface wrapper
   `scripts/entry_next_edge_control.sh stage-foundation-cleanup --dry-run`
   prints the safe stage command without changing the index. Actual staging
   requires `--apply --vedtak <id>` and reruns the post-stage verification.
5. Run `scripts/entry_next_edge_control.sh train-readiness` before any real
   smoke attempt and require `READY_FOR_VEDTAK_SMOKE_TRAIN`. If the decision is
   `READY_FOR_VEDTAK_SMOKE_TRAIN_AFTER_GIT_CLEAN`, the foundation contract is
   ready but the worktree must be cleaned/committed before trainer start. The
   2026-06-29 directional-SMC path may instead report `NOT_READY` with
   `foundation_activation_required_before_smoke=true`; in that case the next
   action is only the explicit vedtak-gated
   `foundation-activation-apply --apply --vedtak <id>` path switch, followed by
   the recorded post-apply audit refresh commands, `verify --quiet` and another
   train-readiness run. If the path switch is already applied,
   `train-readiness` must route the next action directly to
   `foundation-activation-post-apply --apply --vedtak <id>`. It does not approve
   smoke training by itself. The
   smoke-train wrapper reruns both `foundation-guardrails --quiet` and
   `train-readiness --quiet` before real training. The train-readiness report
   also carries the latest foundation guardrail artifact and execution-hygiene
   gate as PASS/FAIL evidence, and must reject a guardrail artifact that lacks
   the readiness JSON command-policy proof, including the post-apply dry-run and
   post-apply apply blocking checks. The smoke-dataset manifest must carry
   `entry_foundation_smoke_dataset_audit_provenance_v1` with SHA256s for the
   active feature, target and specialist audits, and train-readiness must reject
   the sample if those hashes no longer match the active audit artifacts. It
   must also re-hash the smoke split source manifests, output parquets and
   output manifests so stale or mutated sample files cannot pass by carrying
   old hash strings. It must also prove exact objective liveness for
   HH/HL/LH/LL, BOS/CHoCH age, sweep reclaim, compression-expansion,
   impulse/pullback and session x structure on train/val/test, raw
   source-field liveness for every required `snap.*`/`ctx_cont.*` foundation
   input on train/val/test, plus live input features for every required
   specialist encoder on train/val/test. It must also prove the trainer's own
   specialist-fusion contract loader accepts the current specialist audit,
   loads the exact required trainable specialist set at `seq_input_dim=146`,
   and excludes neutral bridge, price-action and unmapped groups from trainable
   specialist indices until those groups have separate liveness/role gates.
   The manifest must also preserve `specialist_model_contract_valid=true`, the
   exact trainable specialist model contract, and exact owned-roadmap-objective
   mapping for the six trainable specialist AIs.
   The specialist architecture contract must also match the target-head
   contract exactly: `hold_horizon` remains blocked and cannot appear in the
   active specialist-fusion head list.
6. Optional proof step before training:
   `scripts/entry_next_edge_control.sh smoke-manifest --vedtak <id>`. This is
   the canonical alias for the smoke wrapper's `--manifest-only` path. It may
   run while git is dirty if `foundation_contract_ready_for_smoke=true`, writes
   the pre-train manifest, and must stop before trainer start.
7. The next real action requires explicit user vedtak:
   `scripts/entry_next_edge_control.sh smoke-train --vedtak <id> --require-edge-audit`.
8. The smoke-train wrapper writes
   `ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST_<UTC>.json` before any real
   trainer process starts. The manifest must record audit artifact SHA256s plus
   feature objective coverage, exact objective liveness, foundation source-field
   coverage and liveness, target-head contract, specialist exact objective
   routing, specialist input liveness, the trainer loader's exact trainable and
   excluded specialist groups, smoke-dataset audit provenance/split hashes and
   worktree hygiene summary, including the critical-gate path review. The
   post-smoke bundle audit must reject a pre-train manifest whose smoke-dataset
   provenance no longer matches the active feature, target and specialist audit
   hashes, or whose worktree hygiene summary does not prove critical gate paths
   are complete.
   The trained bundle metadata must also preserve the exact
   `SPECIALIST_MODEL_CONTRACT` as `bundle_specialist_model_contract`: exact six
   trainable specialist roles, exact owned roadmap objectives, support heads,
   signal families and model roles. The bundle audit must fail if this
   bundle-level contract is missing, reports failures, or diverges from the
   audited specialist registry.
   Real trainer start also requires a clean git worktree; use `--dry-run` or
   `--manifest-only` for inspection/proof while the repo is intentionally
   dirty. `--manifest-only` still requires
   `foundation_contract_ready_for_smoke=true`; it only bypasses the
   execution-hygiene trainer-start blocker.
9. Treat smoke output as plumbing and learning evidence only after the automatic
   post-train bundle audit passes strict load, specialist fusion, exact
   target-head contract, pre-train manifest provenance and edge diagnostics.
   The target-head contract must
   match the active/blocked head sets exactly across bundle capabilities,
   `train_recipe.active_heads`, state_dict heads and forward outputs; extra
   unsupported or experimental heads are not allowed. Specialist fusion is not
   considered live unless the exact required specialist group set is present,
   the gate is normalized and entropic, and each required specialist has mean
   gate weight above 1% on every audited split. Extra ungated specialist groups
   such as price-action must keep the bundle audit closed until promoted by a
   separate gate. Candidate-readiness and replay-readiness must preserve the
   same exact-head, exact-specialist-set, bundle-level specialist model
   contract and per-required-specialist gate-liveness requirements before
   candidate training, replay evidence or IQL distillation can open. The
   candidate train pre-manifest must carry forward
   the smoke edge audit's exact required specialist set, smoke-dataset
   audit-provenance flags, worktree critical-gate proof, and the candidate
   trainer loader's exact trainable and excluded specialist groups.
   Replay-readiness must also
   reject candidate bundle audits whose
   pre-train manifest contract did not validate raw source-field liveness,
   specialist active/blocked head parity and smoke-dataset audit-provenance
   hashes, or whose bundle-level specialist model contract is not preserved.
10. IQL distillation must preserve replay-readiness evidence identity and hash
   provenance. The distillation contract records SHA256s for replay-readiness,
   candidate-readiness, candidate bundle audit, selective-edge summary/metrics
   and candidate replay manifest/metrics/monthly/trades. IQL replay evidence
   must re-check those hashes before it can pass, and `iql-compare` requires the
   IQL replay manifest to prove that validation. The distillation contract must
   also re-check the replay-readiness gate proving candidate pre-train manifest
   provenance, including raw source-field liveness, specialist active/blocked
   head parity, exact smoke specialist-set preservation, valid specialist model
   contract preservation, exact owned-roadmap-objective mapping and the
   smoke-dataset audit-provenance flags. It must also preserve the separate
   replay-readiness gate proving the candidate bundle's own
   `bundle_specialist_model_contract` remained valid after training.
   IQL replay
   evidence and the final IQL comparison gate must independently reject any
   distillation contract where any pre-train, smoke-dataset, specialist-set,
   specialist-model, bundle-specialist-model or replay-artifact provenance
   contract is not `ok`.
11. Candidate training, replay-readiness, IQL distillation, IQL replay evidence,
   promotion review, shadow and live remain closed until their separate gates
   pass.

The phase plan below records how the foundation was intended to be repaired.
Phases 1-3 have corresponding seq146 machine-audit artifacts listed above; the
current active step is the vedtak-gated smoke train.

## Execution Plan

This is the required work order for the next person touching Entry training.
Do not skip phases. A later phase may start only when the previous phase has a
written artifact and a PASS/FAIL conclusion.

### Phase 0 - Freeze Training

Goal: prevent more model churn before the foundation is fixed.

Actions:

- Do not start any new Transformer/IQL/full replay training from the current
  `v10_dataset_seq_structure_neutral` dataset.
- Do not glob old Entry/IQL/XGB/report artifacts from the frozen run and report
  roots. Pre-foundation artifacts now live under explicit `_LEGACY...DO_NOT_USE`
  directories.
- Treat existing bundles as diagnostic artifacts only:
  - `transformer_seq_structure_neutral_tailrisk_bp080_e4_seed1337`
  - `transformer_seq_structure_neutral_tailrisk_v2_sym_e4_seed1337`
- Keep XGB bridge neutral until a new feature/target contract is approved.

Done when:

- This audit is the active source of truth for the next Entry work.
- Any running training/eval processes are stopped or explicitly documented.
- Old pre-foundation datasets, bundles, reports and training outputs are moved
  out of active discovery paths into legacy directories.

### Phase 1 - Feature Preflight Script

Goal: make feature quality machine-checkable before training.

Build a script, suggested path:

- `gx1/scripts/audit_entry_foundation_features_v1.py`

Inputs:

- dataset manifest
- train/val/test parquet paths
- optional bundle metadata path

Required checks:

- Verify `seq`, `snap`, `ctx_cont`, `ctx_cat` dimensions against manifest/bundle.
- Print ordered sequence feature names and context names.
- Report non-finite count by feature and split.
- Report constant and near-constant features by split.
- Report liveness rates for all chart/SMC/structure/liquidity features.
- Detect neutral XGB bridge slots and allowlist only those constants.
- Report train/val/test distribution drift for important features.
- Fail if declared chart features are missing, non-finite, unexpectedly constant,
  or not present in the emitted sequence/context arrays.

Required output artifact:

- `ENTRY_FEATURE_FOUNDATION_AUDIT_<timestamp>.json`
- `ENTRY_FEATURE_FOUNDATION_AUDIT_<timestamp>.md`

Pass criteria:

- No non-finite values.
- No unexpected constant features.
- Neutral XGB bridge constants are explicitly detected and allowlisted.
- BOS, CHoCH, sweep, liquidity, pullback, compression and structure families
  have liveness reported by train/val/test.
- Foundation objective coverage is exact: HH/HL/LH/LL `5/5`,
  BOS/CHoCH age `8/8`, sweep reclaim / false breakout `5/5`,
  compression-expansion `5/5`, impulse/pullback `6/6`, and session x
  structure `28/28`.
- Foundation source dependencies are exact: all 46 raw `snap.*`/`ctx_cont.*`
  source fields used by the foundation structure layer must be present in
  train, val and test emitted contracts.
- Foundation source liveness is exact: all 46 raw foundation source fields must
  have finite, non-constant, active values in train, val and test. The current
  gate uses minimum active-rate `0.0001` and minimum active-count `1`, which
  keeps sparse CHoCH events eligible while still rejecting dead raw inputs.
- Specialist routing is exact: every required foundation objective feature must
  be mapped to its expected specialist encoder with zero missing or misrouted
  features.
- Any sparse but intended event features, especially CHoCH, are marked as such
  and paired with a recency/age/pressure representation.

### Phase 2 - Add Missing Structure Features

Goal: make chart structure explicit, not only implied through interactions.

Add first-class sequence features for:

- HH/HL/LH/LL state as separate numeric inputs.
- BOS up/down age and bars-since-last-break.
- CHoCH age and direction of last character change.
- Sweep-reclaim up/down and failed-sweep follow-through.
- Compression-to-expansion trigger and expansion direction.
- Impulse direction, impulse age, pullback phase and pullback depth.
- Session x structure for Asia, EU, US and overlap, not EU-only.

Implementation rules:

- Features must be lookahead-safe.
- Pivot-confirmed features must only activate after confirmation lag.
- Every new feature must have an entry in the manifest/contract.
- Every new feature must appear in the feature preflight report.
- Do not add magic-rule labels here; this phase is inputs only.

Required output artifact:

- updated dataset builder or feature module
- updated manifest contract
- feature audit PASS report

Pass criteria:

- New sequence dimension is documented.
- All new features have non-finite/constant/liveness stats.
- No feature uses future bars beyond its documented confirmation lag.

### Phase 3 - Target Audit

Goal: prove the labels match the trading problem before optimizing them.

Build a script, suggested path:

- `gx1/scripts/audit_entry_foundation_targets_v1.py`

Required checks by split, side, session and regime:

- `y_direction` rates.
- `y_tradable` rates.
- `y_bad_path` rates.
- `path_quality_bps` mean/p10/p50/p90.
- MFE/MAE distributions.
- clean-edge and survival rates.
- Correlation between `y_bad_path` and `path_quality_bps`.
- Target drift train -> val -> test.
- Majority-label baseline and neutral-skip baseline.

Required decision:

- Confirm whether `H=24, >=15 bps` is still the correct direction target.
- Confirm whether bad-path should be a separate head, a sizing target, a
  direction penalty, or a replay-only diagnostic.
- Confirm whether path-quality horizon `H=10` aligns with the exit/replay horizon.

Required output artifact:

- `ENTRY_TARGET_FOUNDATION_AUDIT_<timestamp>.json`
- `ENTRY_TARGET_FOUNDATION_AUDIT_<timestamp>.md`

Pass criteria:

- Label definitions are explicit and approved.
- Target drift is quantified.
- Bad-path target is semantically valid and has the expected negative relation
  to realized path quality.
- Trading objective is stated as replay/PnL/drawdown/tail-risk, not val accuracy.

### Phase 4 - Rebuild Dataset

Goal: create a clean dataset only after features and targets pass.

Actions:

- Rebuild train/val/test parquet with the new feature contract.
- Emit manifest with:
  - ordered sequence names
  - ordered context names
  - structure-extension feature names
  - target contract
  - neutral bridge status
  - source git commit
  - source parquet hashes
- Run Phase 1 and Phase 3 audits on the rebuilt dataset.

Pass criteria:

- Feature audit PASS.
- Target audit PASS.
- Manifest and bundle/runtime contracts agree on all dimensions.

### Phase 5 - Smoke Model Only

Goal: verify plumbing and obvious learning behavior, not produce a candidate.

Actions:

- Train a short smoke Transformer on a limited epoch budget.
- Keep XGB bridge neutral.
- Track direction, tradable, path-quality and tail-risk diagnostics.
- Do not run full replay yet.

Pass criteria:

- Strict bundle load passes.
- Feature liveness audit passes after export.
- Direction beats majority baseline on val/test.
- `bad_path_prob` is not anti-targeted. It must have negative rank correlation
  with `path_quality_bps` before it can influence gating/sizing.

### Phase 6 - Candidate Training

Goal: train only after smoke behavior is sane.

Actions:

- Train the candidate Transformer/IQL variant.
- Save full training config and environment overrides.
- Export bundle metadata with all active heads and loss weights.
- Run feature and target audits against the exact bundle/dataset used.

Pass criteria:

- Strict load PASS.
- No ignored live inputs.
- Model does not rely on neutral bridge slots.
- Direction, tradable and path heads are calibrated enough for offline evaluation.

### Phase 7 - 2026 Offline Evaluation And Replay

Goal: decide with trading metrics, not training metrics.

Required reports:

- test/eval report on 2026 parquet
- offline replay/backtest report
- split by:
  - side
  - session
  - volatility regime
  - trend regime
  - month

Pass criteria:

- Positive expectancy versus baseline.
- Lower max drawdown versus baseline.
- Better tail-loss profile versus baseline.
- No hidden degradation in either long or short side.
- No session-specific blow-up hidden by aggregate PnL.
- Win rate alone is not accepted as a pass.

### Phase 8 - Promotion Decision

Goal: make an explicit go/no-go decision.

Promotion is allowed only if:

- Feature audit PASS.
- Target audit PASS.
- 2026 offline replay PASS.
- Tail-risk gate PASS if tail-risk is used for gating/sizing.
- Bundle contract is reproducible from manifest and git commit.

If any gate fails:

- Do not shadow/live.
- Record the failed gate and return to the relevant earlier phase.
