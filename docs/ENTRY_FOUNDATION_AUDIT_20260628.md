# Entry Foundation Audit - 2026-06-28

Status: active seq146 foundation is activated and post-apply refreshed. Entry
smoke/candidate/IQL evidence is available, but shadow/live/promotion remain
closed. Exit-side work is now in report-only foundation mode: active
Entry-bound per-bar reconstruction, state/reward contract and split/leakage
audit are ready. The active Exit model dataset/readiness gate is ready, with
train-only normalization metadata and train/val/test shards. The active
Entry-to-Exit feature alignment audit is also ready after carrying Entry
HH/SMC/trend/momentum/MTF mechanism fields plus exact specialist-gate outputs
into Exit model state. Downstream Exit Transformer architecture, training-plan,
wrapper, pretrain-manifest, slice robustness, train-execution review and
post-train audit contract reports are ready in report-only mode. Exit
Transformer/IQL training remains closed until a separate explicit train
enablement vedtak package is reviewed. The Exit Transformer supervised trainer
core exists, but the wrapper keeps it disabled by default and no Exit trainer is
approved to run without that enablement package.

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

Active Objective Rule: all Entry and Exit features with multi-timeframe context
must share one calibrated market-state language before any promotion path is
discussed. The Entry Transformer is the directional evidence layer, Entry IQL is
the entry policy layer, Exit Transformer is the exit-timing evidence layer and
Exit IQL is the exit policy layer. The system must coordinate structure,
liquidity, momentum, trend, volatility, regime, session, spread/ATR,
support/resistance, chart geometry and price action as one replay-proven
trading policy, not as isolated feature families or direction-accuracy scores.

Feature Harmony Rule: every active Entry/Exit input must either be routed into
the shared market-state contract or explicitly excluded with a recorded reason.
The roughly 200 features are not a menu of standalone indicators; they are the
numeric language the sequential AI uses to reconstruct how a manual trader
would read structure, liquidity, support/resistance, Fibonacci/geometry,
candles, EMA/trend pressure, momentum, ATR/spread and session/regime context
across timeframes. Smart layers may summarize these mechanisms, but they do
not replace provenance: each smart feature must be traceable to source fields,
hashes, routing ownership, finite/non-collapsed liveness and replay evidence.
Entry work comes first, but it must preserve the exact state needed by Exit so
profit capture can be learned from the same Entry decisions. Keep the repo
clean at every transition; prefer extending existing modules and control
surfaces, and create new files only for genuinely new artifacts, gates or model
components.

Full Bot Objective Lock: the requested end state is a fully automated XAUUSD
policy that enters long or short at the highest-quality Entry points and exits
near maximum profit opportunity. Entry is the first active build track: the
Entry Transformer must learn the full multi-timeframe picture as the evidence
"eyes", and Entry IQL may only become the policy "brain" after exact replay
evidence proves tradable edge. Exit follows after Entry evidence: Exit
Transformer must learn exit-timing evidence from the exact Entry policy traces,
and Exit IQL must learn profit capture from those replay-bound states. The
roughly 200 Entry/Exit inputs are one shared market-state language, not a bag
of independent indicators; HH/HL/LH/LL, SMC/liquidity, momentum/flow,
trend/EMA, volatility/compression, regime/session, support/resistance,
spread/ATR, chart geometry, price action and all multi-timeframe context must
be calibrated together before promotion, shadow or live can be discussed. Keep
repo work inside existing modules, reports and control surfaces; create new
files only for genuinely new artifacts, gates or model components that cannot
fit the existing ownership boundaries. Every step must preserve clean-git
hygiene before real training and must fail closed on missing provenance, hashes,
liveness, exact contracts or weak replay slices.

Codex Objective Lock - 2026-07-01: every session must continue toward one
coherent Entry-to-Exit trading system, not isolated feature experiments. The
existing foundation inputs, smart-layer inputs and multi-timeframe context must
all stay routed, live, hashed and explainable by market mechanism. Smart layers
are not replacements for the old foundation features; they are calibrated
mechanism summaries layered on top of the same source language. Entry
Transformer training must improve the evidence layer first, Entry IQL may only
consume replay-proven Entry evidence, and Exit Transformer/IQL work must reuse
the exact Entry policy traces to learn profit capture. Raw direction accuracy is
only a diagnostic; the acceptance proof is replay net PnL, drawdown, MAE,
bad-path avoidance, path-quality calibration, selected-tail direction quality
and session/regime/side/tail robustness.

Feature Orchestration Rule - 2026-07-01: no feature family is considered
"done" merely because it is present in the dataset or routed to a specialist.
Structure, SMC/liquidity, momentum/flow, trend/EMA, volatility/compression,
session/regime, spread/ATR, support/resistance, chart geometry, price action
and every multi-timeframe interaction must be tested as cooperating evidence
for the same Entry decision surface. The old foundation features, reused
sequence extensions, chart/candle challengers, smart-layer summaries and
context embeddings must remain available to the Transformer unless an exact
ablation plus replay/slice proof shows a removal improves tradable edge. Any
new smart specialist must feed calibrated evidence into the Entry Transformer
and preserve the Entry-to-Exit state needed for Exit Transformer/IQL profit
capture; isolated feature polish that cannot improve replay evidence is not
approved work.

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

`handover -> verify -> selftest -> foundation-guardrails -> readiness-report -> Entry evidence gates -> Entry/IQL replay and slice evidence -> Entry-to-Exit handoff -> active Exit per-bar reconstruction -> active Exit state/reward contract -> active Exit split/leakage audit -> active Exit model dataset/readiness gates -> active Entry-to-Exit feature alignment -> active Exit Transformer architecture/readiness -> active Exit Transformer training plan/readiness -> fail-closed Exit Transformer trainer-wrapper readiness -> active Exit Transformer pretrain manifest -> active Exit model dataset slice robustness -> active Exit Transformer train-execution review -> active Exit Transformer post-train audit contract -> explicit active Exit Transformer train-enablement package -> only then consider capped Exit train wrapper execution/replay/IQL evidence gates`

The historical activation path was:

`handover -> verify -> selftest -> foundation-guardrails -> foundation-adoption-candidate -> foundation-activation-plan -> foundation-activation-apply --dry-run -> worktree-hygiene -> optional stage-foundation-cleanup --apply --vedtak <id> -> train-readiness -> if foundation_activation_required_before_smoke=true: foundation-activation-apply --apply --vedtak <id> -> foundation-activation-post-apply --apply --vedtak <id> -> train-readiness -> optional smoke-manifest --vedtak <id> -> smoke-train --vedtak <id> --require-edge-audit`

Use `scripts/entry_next_edge_control.sh handover` as the default orientation
command for a fresh session. It prints the active seq146 operating point, runs
foundation verification, summarizes readiness, and keeps the historical legacy
handover behind the explicit `GX1_ALLOW_LEGACY_HANDOVER=20260627_ALLOW_LEGACY_HANDOVER`
token.
Use `scripts/entry_next_edge_control.sh readiness-report` when you need a
fast non-training light refresh: it refreshes worktree hygiene and
train-readiness, then reads the latest candidate-readiness, replay-readiness,
IQL distillation, IQL replay and Exit reports. It is report-only: it must not
stage, train, replay, distill, shadow or touch live paths. Use
`scripts/entry_next_edge_control.sh readiness-report --snapshot` only when a
strict latest-report read is required, and use
`scripts/entry_next_edge_control.sh readiness-report --refresh` only when an
explicit full report refresh is needed; that path may rerun large smart hash
and fullscan gates, but still must not stage, train, replay, distill, shadow or
touch live paths. The report also
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

## Current Operating Status - 2026-07-01

Active status:

- Worktree hygiene is `PASS_CLEAN_GIT`, and
  `scripts/entry_next_edge_control.sh verify --quiet` passes.
- Feature harmony is current: 843 active/generated inputs are accounted for,
  772 are specialist-routed, 71 are explicitly excluded with recorded reasons
  and `unmapped_input_count=0`. The active categorical context is
  `ctx_cat=5`: `session_id`, `vol_regime_id`, `atr_bucket`,
  `spread_bucket` and `H4_trend_sign_cat`; these are context embeddings, not
  a reduction of the full signal surface.
- Feature orchestration is now machine-gated inside
  `feature_harmony_contract`: all eight required mechanism specialists, all
  required input surfaces and all ten smart-layer source contracts must be
  present before inventory readiness can pass. The latest inventory has
  `feature_orchestration_ready=true`, no missing required mechanism
  specialists, no missing required input surfaces and no missing required
  smart layers.
- Smart rebuild preflight now consumes that contract directly. The latest
  `ENTRY_SMART_REBUILD_PREFLIGHT_latest.json` has PASS checks for both
  `inventory feature harmony contract is ready` and
  `inventory feature orchestration contract is ready`; smart rebuild review
  cannot open from feature counts or source coverage alone.
- Smart post-rebuild readiness now consumes the smart rebuild preflight report
  directly. It must fail closed unless the preflight report proves feature
  harmony/orchestration, has no failures, matches the audited dataset directory
  and matches the smart manifest by hash. Smoke/trainability authority cannot
  be based on a rebuilt dataset with missing orchestration provenance.
- Smart smoke-manifest and smoke-readiness now require that post-rebuild
  orchestration provenance to be preserved in the latest smoke-manifest
  readiness report. A stale smoke-manifest report that predates the
  orchestration checks must block smart smoke-readiness until the proof-only
  smart smoke-manifest gate is regenerated with an explicit vedtak.
- The smart candidate contract is `smart_seq520_candidate`: expected
  seq/snap width 520, made from 41 base signal fields, 105 reused foundation
  extension fields, 41 chart-geometry fields, 28 candlestick fields and 305
  smart-layer fields across ten layers. All ten smart layers have required
  source coverage and zero missing required source fields.
- Smart trainability is structurally wired under
  `specialist_contract_mode=smart_seq520_candidate`, but it must remain
  fail-closed whenever smart smoke-readiness is blocked by stale
  smoke-manifest/post-rebuild provenance. Smart candidate-readiness is
  intentionally `NOT_READY_FOR_CANDIDATE_TRAINING` until a newly trained smart
  smoke bundle carries `path_calibration_recipe_contract=PASS`; the latest old
  smart smoke bundle has `path_calibration_recipe_contract=null`.
  Candidate-readiness also requires direction distribution coverage across
  active LONG/SHORT/FLAT classes, so a bundle that beats majority while
  collapsing away from a common class remains blocked.
- Future smart smoke/candidate bundles must carry
  `tail_direction_recipe_contract=PASS` with
  `ENTRY_TAIL_DIRECTION_CE_WEIGHT=0.35`,
  `ENTRY_TAIL_DIRECTION_QUALITY_QUANTILE=0.70` and
  `ENTRY_TAIL_DIRECTION_MIN_BATCH=8`. The trainer applies this extra
  direction CE only to directional, tradable, clean-path rows in the top
  path-quality tail, so low broad direction accuracy and weak selected-tail
  direction precision are attacked during training, not just diagnosed later.
- The broad/default smart replay is not approved for IQL because drawdown is
  too high: the latest default replay-readiness remains
  `NOT_READY_FOR_IQL_DISTILLATION` with max drawdown about 1342 bps against
  the 650 bps bound. This is an explicit fail-closed result, not a hidden
  failure.
- The selected smart replay policy
  `smart_seq520_stop_tp_mfe_protect_act1_sl45_tp90_top10_cost00` passes replay
  evidence with 655 trades, net about 3439 bps, profit factor about 3.08, max
  drawdown about 418 bps and max loss -45 bps. Its direction precision is only
  about 0.478, so the policy is edge-positive through selective ranking and
  risk/exit controls, not through strong broad direction accuracy.
- Replay-readiness now treats selected-tail direction quality as a hard
  contract: candidate top-5/top-10 selected tails and supported
  session/side/vol-regime slices must clear the configured direction precision
  threshold before IQL distillation authority can open. PnL filtering may not
  hide a weak long/short direction model.
- The selected smart IQL replay
  `smart_seq520_iql_student_stop_tp_mfe_protect_act1_sl45_broad_net_min190_v2`
  passes with 735 trades, net about 3886 bps, profit factor about 3.13, max
  drawdown 315 bps and max loss -45 bps. IQL comparison is
  `READY_FOR_PROMOTION_REVIEW_VEDTAK` with about +447 bps net lift versus the
  selected candidate, but `promotion_shadow_live_allowed=false`.
- Smart selected IQL slice audit is now fail-closed on path-signal
  calibration. The report decision is `FAIL` because `path_quality_pred` is
  wrong-signed for net PnL and stop-loss behavior on both candidate and IQL,
  and `bad_path_prob` is wrong-signed versus stop-loss behavior. Supported
  regression counts remain diagnostic=5, drawdown=2, edge=3 and p90 MAE=8.
  Worst supported edge regressions include volatility regime 3, SHORT side and
  ASIA session. The next required gate is to repair weak slices or path-signal
  calibration before promotion review; shadow/live remain blocked. Exit
  opportunity diagnostics still show large remaining giveback/peak oracle
  slack, including about 12895 bps IQL peak-oracle lift, which is the current
  evidence for prioritizing Exit Transformer/hazard/IQL work after the Entry
  calibration issue is handled.
- Commit `e7aa6762` repairs the next Entry training recipe for this known
  path-signal defect: `entry_v10_ctx_train_v3.py` now has a full-batch
  `path_quality_pred` ranking loss against realized `path_quality_bps`, and
  the smoke/candidate wrappers pass the matching `ENTRY_PATH_QUALITY_RANK_*`
  recipe into future vedtak-gated runs. Existing smart replay artifacts are
  still old and must remain failed until a new capped smart smoke/candidate
  bundle proves corrected calibration in replay.
- Commit `38170e56` locks that repair into the proof gates: smoke/candidate
  bundle audit and replay-readiness now require a passing
  `path_calibration_recipe_contract` with full-batch path-quality ranking
  before a candidate bundle can be replay/IQL authority. Old bundle-audit
  reports without this recipe are fail-closed until regenerated from a newly
  trained bundle.
- Commit `20677b4d` makes the selected smart replay-readiness effective status
  fail closed when the selected IQL slice/path-signal calibration report is not
  ready, even if an older selected replay-readiness artifact still says
  `READY_FOR_IQL_DISTILLATION_VEDTAK`. New training/replay authority requires a
  freshly trained bundle with the path-calibration recipe and a refreshed
  replay/slice audit that passes.
- Smart smoke-manifest, smoke-readiness and trainability gates now require the
  future smart train contract to declare the exact full-batch
  `path_calibration_recipe_contract` and the six
  `ENTRY_*_QUALITY_RANK_*` env values, and trainability checks that both
  smoke and candidate wrappers expose those envs. The smart lane must fail
  closed if the path-quality/bad-path ranking repair is not actually carried
  into the next capped trainer command.
- The next smart smoke/candidate training recipe must also preserve
  `direction_balance_recipe_contract=PASS`: `ENTRY_PRED_BALANCE_ALPHA=0.05`,
  `ENTRY_PRED_BALANCE_TARGET=label`, positive `ENTRY_DIRECTION_CE_SCALE` and
  `GX1_V10_CKPT_MONITOR=dir_acc`. This turns weak broad direction accuracy and
  class-collapse risk into a train contract and bundle-audit gate, not a
  cosmetic metric.
- Candidate-readiness now also requires the smoke bundle audit itself to carry
  `path_calibration_recipe_contract=PASS` with active path-quality and bad-path
  heads, full-batch path-quality ranking and positive rank weights/margins.
  It also requires active LONG/SHORT/FLAT direction distribution coverage on
  audited splits; old smoke bundles without these contracts cannot open
  candidate training, even if their older aggregate direction/PnL diagnostics
  looked acceptable.
- Smoke bundle audit and candidate-readiness must also prove direction context
  slice diagnostics across available categorical buckets such as session,
  volatility regime, ATR bucket, spread bucket and H4 trend sign. Audited
  slices with enough rows and label diversity must beat their own majority
  baseline and preserve active class distribution, so broad accuracy around
  0.40 cannot hide a weak regime/session bucket.
- Smart smoke-manifest, smoke-readiness and trainability must declare this
  `direction_context_slice_contract` before a future smart smoke train can be
  considered cleanly wired; the proof still happens in the post-smoke bundle
  audit from the actual trained bundle.
- Raw smoke/bundle direction accuracy around 0.40 is not an acceptance metric by
  itself. It is only a sanity diagnostic against the majority baseline; Entry
  acceptance remains replay/PnL, drawdown, MAE, bad-path, class-distribution
  calibration and session/regime/side/tail slice evidence.
- Exit report-only gates remain ready through feature alignment, architecture,
  training plan, wrapper readiness, pretrain manifest, slice robustness,
  train-execution review and post-train audit contract. Exit Transformer/IQL
  training is still blocked by
  `BLOCKED_BY_EXIT_TRANSFORMER_TRAIN_ENABLEMENT_PACKAGE`; no Exit training may
  run without a separate explicit `ENTRY_EXIT_TRANSFORMER_TRAIN_` package.
- Shadow, live and promotion remain closed.

## Current Operating Status - 2026-06-30

Active status:

- Foundation activation apply completed.
- Foundation post-apply refresh completed.
- Worktree hygiene: `PASS_CLEAN_GIT`.
- Train-readiness: `READY_FOR_VEDTAK_SMOKE_TRAIN`.
- Seq215 challenger foundation: materialized and audited as a separate
  8-specialist challenger contract with 215 seq/snap signal inputs, but not yet
  trained as evidence. It is opened only by
  `scripts/entry_next_edge_control.sh smoke-train-seq215 --vedtak <id> --require-edge-audit`
  with an explicit `SEQ215` vedtak id. `candidate-readiness-seq215` must remain
  `NOT_READY_FOR_CANDIDATE_TRAINING` until the real seq215 smoke bundle edge
  audit exists and passes.
- Smart-layer candidate: report-only/dormant. The optional
  `materialize_entry_specialist_challenger_extension_manifest_v1 --include-smart-layers`
  path currently combines the audited seq215 extension with 305 extra dormant smart-layer
  features for trend/EMA, SMC/liquidity quality, structure/swing derivations,
  momentum/flow, session/regime interactions, volatility/compression,
  chart-geometry smart2, price-action/candle smart3, support/resistance level
  memory and multi-timeframe confluence. With the current feature list this is
  a `smart_seq520_candidate` manifest candidate, not an active
  dataset. It writes separate `SMART` manifest/report latest files and must
  not overwrite the seq215 latest contract. The Entry feature AI inventory now
  writes `feature_harmony_contract`: every active/generated Entry input is
  either specialist-routed or explicitly excluded with reason, with zero
  unmapped fields required. It also proves smart-source coverage for all ten
  smart layers, including OHLC source provenance for price-action/candle
  smart3, and the inline sequence-extension builder can materialize the
  requested 305 smart-layer fields when a capped rebuild is explicitly opened.
  `smart-rebuild-preflight` is the report-only
  gate that binds the latest smart manifest, inventory, active source parquet,
  hashes and 4G-capped rebuild command before any dataset mutation. It is still
  not train-ready and requires a separate dataset rebuild, feature audit, specialist audit,
  liveness/non-collapse proof and train-readiness gate before any
  smoke/candidate/replay/IQL step.
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
- Active Entry-to-Exit feature alignment:
  `ENTRY_EXIT_FEATURE_ALIGNMENT_READY_FOR_EXIT_TRANSFORMER_TRAINING_REVIEW`.
  The active Exit state now carries Entry score/probabilities, path-quality,
  bad-path context, path/session/regime/spread/ATR state, 43 Entry
  snapshot/context mechanism fields and the exact six specialist-gate weights:
  `structure_swing`, `smc_liquidity`, `trend_ema`, `vol_compression`,
  `momentum_flow` and `session_regime`. Missing required alignment families:
  zero.
  This six-gate alignment is valid only for the active `foundation_seq146`
  contract. `challenger_seq215` must not reuse the six-gate Exit alignment; it
  must carry `chart_geometry_encoder` and `price_action_candle_encoder` state
  plus all eight specialist-gate weights before any Exit train/replay/IQL step.
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
  disabled, it requires the active train-execution review report and
  post-train audit contract, and the future train path declares cgroup RAM caps
  plus `--num-workers 0`.
- Active Exit Transformer pretrain manifest:
  `ENTRY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST_READY_FOR_TRAIN_EXECUTION_REVIEW`.
  It imports the active trainer core, runs a CPU-only finite forward preflight
  on four train episodes, proves the exact five output heads are live, records
  405559 parameters, 35 valid tokens and `optimizer_steps=0`, and keeps all
  train/replay/IQL/shadow/live side effects closed.
- Active Exit model dataset slice robustness:
  `ENTRY_EXIT_MODEL_DATASET_SLICE_ROBUSTNESS_READY_WITH_WEAK_SLICE_DISCLOSURE`.
  Train/val/test split-level labels, rewards and state features are live.
  The audit found zero unsupported session/regime/side slices and 20 weak
  slices that must be explicitly accounted for before any Exit train-execution
  review; examples include sparse long slices in EU/US/overlap and low-count
  volatility-regime slices. Numeric feature liveness remains strict on train;
  finite but constant val/test context fields are disclosed separately. The
  current disclosure is `entry_ctx_d1_regime_class_id_v2` in val/test, with
  train live and zero blocking numeric feature failures.
- Active Exit Transformer train-execution review:
  `ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_READY_FOR_EXPLICIT_VEDTAK_PACKAGE`.
  It binds the active training plan, fail-closed wrapper, pretrain manifest,
  weak-slice policy and RAM guardrails into one report. It requires the
  `ENTRY_EXIT_TRANSFORMER_TRAIN_` vedtak prefix for any future training
  discussion, preserves `num_workers=0`, max process RSS 8 GiB and abort below
  8 GiB available RAM, and requires post-train reporting for session/regime/side,
  direction/tail and weak slices. It still sets
  `exit_training_allowed=false` and
  `exit_training_allowed_with_explicit_vedtak=false`.
- Active Exit Transformer post-train audit contract:
  `ENTRY_EXIT_TRANSFORMER_POST_TRAIN_AUDIT_CONTRACT_READY`.
  It locks the future bundle audit before any Exit train enablement package:
  exact five heads, strict load/finite forward, train-only normalization hash,
  weak-slice disclosure, session/regime/side/tail diagnostics, net reward,
  MAE/drawdown, giveback risk and MFE capture. Replay, Exit-IQL, shadow, live
  and promotion remain blocked until a trained bundle later passes that audit.
- Exit Transformer training, Exit IQL, shadow, live and promotion remain
  closed. The next boundary is an explicit Exit train-execution enablement
  vedtak package; it must account for weak slices, clean-git checks, RAM guard
  and the active post-train audit contract before any Exit trainer can run.

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
   Use `scripts/entry_next_edge_control.sh readiness-report` for a fast
   fail-open light refresh of worktree/train-readiness plus latest downstream
   reports while the current step is still blocked; use
   `readiness-report --snapshot` only for strict latest-report reads and
   `readiness-report --refresh` only when an explicit full report refresh is
   intended.
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
   specialist-fusion contract loader accepts the current specialist audit and
   loads the exact required trainable specialist set for the declared
   `specialist_contract_mode`: six specialists at `seq_input_dim=146` for
   `foundation_seq146`, or eight specialists at `seq_input_dim=215` for
   `challenger_seq215`. Neutral bridge and unmapped groups remain excluded from
   trainable specialist indices in both modes.
   The manifest must also preserve `specialist_model_contract_valid=true`, the
   exact trainable specialist model contract for the declared mode, and exact
   owned-roadmap-objective mapping for every trainable specialist AI.
   The specialist architecture contract must also match the target-head
   contract exactly: `hold_horizon` remains blocked and cannot appear in the
   active specialist-fusion head list.
6. Optional proof step before training:
   `scripts/entry_next_edge_control.sh smoke-manifest --vedtak <id>`. This is
   the canonical alias for the smoke wrapper's `--manifest-only` path. It may
   run while git is dirty if `foundation_contract_ready_for_smoke=true`, writes
   the pre-train manifest, and must stop before trainer start.
   For the seq215 challenger, use
   `scripts/entry_next_edge_control.sh smoke-manifest-seq215 --vedtak <id>`
   with an explicit vedtak id containing `SEQ215`; it must preserve
   `specialist_contract_mode=challenger_seq215`, `seq_input_dim=215`, the
   seq215 smoke dataset and seq215 specialist audit.
7. The next real action requires explicit user vedtak:
   `scripts/entry_next_edge_control.sh smoke-train --vedtak <id> --require-edge-audit`.
   For the seq215 challenger, the only real-train command is
   `scripts/entry_next_edge_control.sh smoke-train-seq215 --vedtak <id> --require-edge-audit`.
   It requires a `SEQ215` vedtak id, clean git, the same train-readiness gate,
   and a post-smoke audit written under
   `entry_foundation_smoke_bundle_audit_20260628_v1/challenger_seq215_20260630`.
   It does not open candidate training, replay, IQL, shadow, live or promotion.
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
   The trained bundle metadata must also preserve the exact specialist model
   contract for the declared mode as `bundle_specialist_model_contract`: exact
   six trainable specialist roles for `foundation_seq146` or exact eight
   trainable specialist roles for `challenger_seq215`, exact owned roadmap
   objectives, support heads, signal families and model roles. The bundle audit
   must fail if this bundle-level contract is missing, reports failures, or
   diverges from the audited specialist registry.
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
   considered live unless the exact required specialist group set for the
   declared contract mode is present, the gate is normalized and entropic, and
   each required specialist has mean gate weight above 1% on every audited
   split. Extra ungated specialist groups outside the selected mode must keep
   the bundle audit closed until promoted by a separate gate. Candidate-readiness
   and replay-readiness must preserve the same exact-head, exact-specialist-set,
   bundle-level specialist model
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
- For any chart-geometry challenger build, report liveness for numeric
  trendline, support/resistance-channel, Fibonacci-zone, EMA-cross and
  chart-pattern proxy families separately from SMC.
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
- Challenger chart geometry: numeric support/resistance line proximity,
  channel position, Fibonacci retracement/extension proximity, EMA-cross
  pressure, trendline-break pressure and triangle/flag pattern proxies. These
  remain excluded from active `foundation_seq146`, but are included in the
  gated `challenger_seq215` 8-specialist trainable contract after the seq215
  rebuild and specialist audit. They still require seq215 smoke edge audit,
  candidate-readiness, replay and IQL evidence before they can influence any
  promotion discussion.
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
