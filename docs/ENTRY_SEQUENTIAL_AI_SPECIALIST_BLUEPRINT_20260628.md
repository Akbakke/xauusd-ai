# Entry Sequential AI Specialist Blueprint

Status: machine-audited feature grouping contracts exist. The active canonical
Entry foundation remains `foundation_seq146`, and `train-readiness` is green
for a vedtak-gated smoke run. The `challenger_seq215` path is now an audited
8-specialist challenger contract, but it is not candidate/replay/IQL evidence
until its own real smoke train and post-smoke edge audit pass. Do not promote
from either path until smoke training, replay and explicit promotion gates pass.

Machine contract:

- Registry: `gx1/features/entry_specialist_feature_groups_v1.py`
  - `SPECIALIST_MODEL_CONTRACT` is the machine-readable role contract for the
    six `foundation_seq146` trainable specialist AIs: owned roadmap objectives,
    primary signal families and supported active heads.
  - `CHALLENGER_SEQ215_SPECIALIST_MODEL_CONTRACT` is the machine-readable role
    contract for the eight `challenger_seq215` trainable specialist AIs. It
    extends the six foundation specialists with `chart_geometry_encoder` and
    `price_action_candle_encoder`.
  - `specialist_model_contract_for_mode()` and
    `required_training_specialists_for_mode()` are the authority. Documentation,
    wrappers, audits and bundle metadata must name the active contract mode
    instead of assuming the six-specialist base contract.
- Audit: `gx1/scripts/audit_entry_specialist_feature_groups_v1.py`
  - The audit writes `specialist_model_contract`,
    `specialist_model_contract_valid` and
    `specialist_model_contract_failures`.
- Post-smoke bundle audit:
  `gx1/scripts/audit_entry_foundation_smoke_bundle_v1.py`
  - Must validate both the pre-train manifest specialist model contract and the
    trained bundle metadata's preserved `bundle_specialist_model_contract` for
    the declared contract mode.
- Train-readiness gate:
  `gx1/scripts/verify_entry_training_readiness_v1.py`
- Worktree-hygiene gate:
  `gx1/scripts/audit_entry_foundation_worktree_hygiene_v1.py`
  - Critical-gate path coverage includes docs/control, smoke bundle audit,
    candidate-readiness, replay-readiness, IQL distillation/evidence/comparison
    gates and their tests.
- Feature AI inventory and harmony gate:
  `gx1/scripts/materialize_entry_feature_ai_inventory_v1.py`
  - Writes `feature_harmony_contract` as the machine-readable authority for
    the Feature Harmony Rule: every active/generated Entry input must be
    specialist-routed or explicitly excluded with a recorded reason, smart
    layers must preserve source coverage, and unmapped fields keep downstream
    training/replay gates closed.
  - Also writes `feature_orchestration_ready` inside that contract. This proves
    the full Entry decision surface has all required mechanism specialists,
    all required input surfaces and all smart-layer source contracts present
    before the feature inventory can be treated as ready.
- Smart rebuild preflight:
  `gx1/scripts/materialize_entry_smart_seq520_rebuild_preflight_v1.py`
  - Consumes `feature_harmony_contract` and must fail closed unless both
    `feature_harmony_ready=true` and `feature_orchestration_ready=true` with no
    missing mechanism specialists, input surfaces or smart layers. Smart
    dataset rebuild review cannot open from counts/source coverage alone.
- Smart post-rebuild readiness:
  `gx1/scripts/audit_entry_smart_dataset_post_rebuild_readiness_v1.py`
  - Consumes the smart rebuild preflight report and must fail closed unless
    that preflight proves feature harmony and feature orchestration, has no
    failures, matches the audited dataset directory and matches the smart
    manifest by hash. A rebuilt dataset cannot become smoke/trainability
    authority if orchestration provenance is missing.
- Candidate-readiness gate:
  `gx1/scripts/verify_entry_candidate_readiness_v1.py`
  - Supports `foundation_seq146` and `challenger_seq215`. The seq215 report
    lives under `entry_candidate_readiness_20260628_v1/challenger_seq215_20260630`
    and must remain `NOT_READY_FOR_CANDIDATE_TRAINING` until the matching
    seq215 smoke bundle edge audit exists and passes.
- Candidate-train wrapper:
  `scripts/run_entry_foundation_seq146_candidate_train.sh`
- Candidate selective-edge evaluator:
  `gx1/scripts/evaluate_entry_candidate_selective_edge_v1.py`
- Candidate replay evidence materializer:
  `gx1/scripts/materialize_entry_candidate_replay_evidence_v1.py`
- Replay/distillation-readiness gate:
  `gx1/scripts/verify_entry_replay_readiness_v1.py`
- IQL distillation contract:
  `gx1/scripts/materialize_entry_iql_distillation_contract_v1.py`
- IQL distillation wrapper:
  `scripts/run_entry_foundation_iql_distill.sh`
- IQL replay evidence materializer:
  `gx1/scripts/materialize_entry_iql_replay_evidence_v1.py`
- Post-distillation replay comparison gate:
  `gx1/scripts/verify_entry_iql_replay_comparison_v1.py`
- Post-comparison slice/tail audit:
  `gx1/scripts/audit_entry_iql_replay_slices_v1.py`
  - Must prove session/regime/side edge slices, direction/bad-path/tail
    diagnostics and exit-opportunity slack from the exact replay trade logs.
- Entry-to-Exit handoff readiness audit:
  `gx1/scripts/audit_entry_exit_handoff_readiness_v1.py`
  - Must preserve active Entry/IQL replay evidence and keep Exit Transformer/IQL
    training blocked until an active Entry-bound per-bar Exit substrate exists.
- Entry-bound Exit per-bar substrate materializer:
  `gx1/scripts/materialize_entry_exit_per_bar_handoff_v1.py`
  - Builds the active per-bar HOLD/EXIT_NOW substrate from IQL replay trades and
    canonical M5 bid/ask bars, with hashed M1-to-M5 supplement when available
    and explicit gap exclusions when price coverage is unresolved. It keeps
    Exit Transformer/IQL training closed for separate review gates.
- Active Exit per-bar reconstruction audit:
  `gx1/scripts/audit_entry_exit_per_bar_reconstruction_v1.py`
  - Proves the active Entry-bound per-bar substrate has live ATR/spread,
    finite state, exact terminal rows, contiguous M5 timelines, provenance and
    explicit gap exclusions before state/reward work.
- Active Exit state/reward contract:
  `gx1/scripts/materialize_entry_exit_state_reward_contract_v1.py`
  - Materializes HOLD/EXIT_NOW state, reward and next-row pointer semantics
    from the active reconstruction. It keeps reward/outcome fields out of state
    and keeps Exit training/IQL closed.
- Active Exit split/leakage audit:
  `gx1/scripts/audit_entry_exit_split_leakage_v1.py`
  - Assigns deterministic time-ordered train/val/test episode splits and
    proves no episode, next-row pointer, reward field or shortcut state feature
    leaks across model boundaries.
- Active Exit model dataset/readiness:
  `gx1/scripts/materialize_entry_exit_model_dataset_readiness_v1.py`
  - Writes active train/val/test Exit model shards, feature schema and
    train-only numeric/categorical normalization metadata from the split/leakage
    dataset. It keeps Exit Transformer/IQL training closed until architecture,
    training and replay-evidence gates exist and pass.
- Active Entry-to-Exit feature alignment:
  `gx1/scripts/audit_entry_exit_feature_alignment_v1.py`
  - Audits whether the Exit model state carries Entry policy context plus the
    required HH/SMC/liquidity, trend/EMA, compression/expansion, momentum/flow,
    multi-timeframe and specialist-gate mechanism families. It is fail-closed:
    the active Exit state is ready only when those Entry snapshot families and
    exact specialist-gate outputs are materialized as model state, not just
    broad score/probability fields.
  - For `foundation_seq146`, the exact Entry specialist-gate set is the six base
    specialists. For `challenger_seq215`, Exit alignment must carry
    `chart_geometry_encoder` and `price_action_candle_encoder` state plus all
    eight specialist-gate weights before any Exit train/replay/IQL step.
- Active Exit Transformer architecture/readiness:
  `gx1/scripts/audit_entry_exit_transformer_architecture_readiness_v1.py`
  - Locks the active `exit_sequence_transformer_v1` architecture contract
    against the model dataset: causal masked Transformer encoder, train-only
    normalization, planned sequence length, exact output heads and no
    train/replay/IQL/shadow/live side effects.
- Active Exit Transformer training plan/readiness:
  `gx1/scripts/materialize_entry_exit_transformer_training_plan_readiness_v1.py`
  - Pins the future trainer command contract, active architecture/dataset shard
    hashes, exact Exit output heads, explicit `ENTRY_EXIT_TRANSFORMER_TRAIN_`
    vedtak requirement, clean-git/pretrain-manifest requirements and RAM
    guardrails. It remains report-only and keeps trainer/replay/IQL/shadow/live
    side effects closed.
- Active Exit Transformer trainer wrapper readiness:
  `gx1/scripts/audit_entry_exit_transformer_trainer_wrapper_readiness_v1.py`
  - Audits `scripts/run_entry_exit_transformer_train.sh` as a fail-closed
    future train wrapper: vedtak prefix rejection, implementation disabled,
    train-execution review and post-train audit contract requirements, cgroup
    RAM cap declaration, `--num-workers 0` and no train/replay/IQL/shadow/live
    side effects.
- Active Exit Transformer trainer core:
  `gx1/models/exit_sequence_transformer/train_v1.py`
  - Defines the active causal masked `ExitSequenceTransformerV1` with exact
    output heads, a CPU-safe `--preflight-only` path and the supervised
    per-bar Exit training loop. Non-preflight training remains fail-closed
    unless `--enable-training`, an `ENTRY_EXIT_TRANSFORMER_TRAIN_` vedtak,
    ready train-execution/post-train/feature-alignment reports and `--num-workers 0`
    are all provided.
- Active Exit Transformer pretrain manifest:
  `gx1/scripts/materialize_entry_exit_transformer_pretrain_manifest_v1.py`
  - Runs a finite forward preflight on active train episodes, writes the
    pretrain manifest, records zero optimizer steps and keeps Exit
    training/replay/IQL/shadow/live closed until train-execution review.
- Active Exit model dataset slice robustness:
  `gx1/scripts/audit_entry_exit_model_dataset_slice_robustness_v1.py`
  - Audits train/val/test label, reward and state-feature liveness plus
    session/regime/side slices. Train numeric state features must be finite and
    live. Finite but constant non-train context fields are disclosed separately
    only when train is live. It discloses weak slices explicitly so
    train-execution review cannot hide sparse side/session/regime behavior
    behind broad averages.
- Active Exit Transformer train-execution review:
  `gx1/scripts/audit_entry_exit_transformer_train_execution_review_v1.py`
  - Binds the training plan, fail-closed wrapper, pretrain manifest, RAM
    guardrails and weak-slice policy into one report. It keeps Exit training
    closed and requires a separate explicit train-execution vedtak package
    before any trainer can run.
- Active Exit Transformer post-train audit contract:
  `gx1/scripts/audit_entry_exit_transformer_post_train_contract_v1.py`
  - Locks the required future bundle audit before any Exit trainer enablement:
    exact active heads, strict load/finite forward, train-only normalization
    hash, weak-slice disclosure, session/regime/side/tail diagnostics, net
    reward, MAE/drawdown, giveback risk and MFE capture. It keeps
    training/replay/IQL/shadow/live closed.
- Active Exit Transformer train-enablement package:
  `gx1/scripts/materialize_entry_exit_transformer_train_enablement_package_v1.py`
  - Binds feature alignment, training plan, wrapper readiness, train-execution
    review and post-train audit contract into one explicit
    `ENTRY_EXIT_TRANSFORMER_TRAIN_` package. It exercises only the wrapper
    dry-run path, records the exact capped future training command and keeps
    trainer/replay/IQL/shadow/live closed.
- Latest report:
  `/home/andre2/GX1_DATA/reports/entry_specialist_feature_group_audit_20260628_v1/ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.json`
- Current decision: `PASS` on the active seq146 foundation dataset.

## Current Evidence Status - 2026-07-01

The 2026-07-01 operating point is feature-harmony clean but not promotion-ready:

- Worktree hygiene is `PASS_CLEAN_GIT`, and `verify --quiet` passes.
- The active feature harmony report accounts for 843 inputs: 772 routed,
  71 explicitly excluded and `unmapped_input_count=0`.
- `ctx_cat=5` means the five categorical context embeddings
  `session_id`, `vol_regime_id`, `atr_bucket`, `spread_bucket` and
  `H4_trend_sign_cat`; it does not replace the 520 smart seq/snap signal
  surface or the 142 continuous context fields.
- `smart_seq520_candidate` is the active smart evidence contract for the smart
  path: 520 seq/snap signals, 8 required trainable specialists and 10 smart
  layers with 305 generated mechanism features. Required smart-layer source
  coverage is complete.
- Default broad smart replay remains fail-closed because max drawdown is about
  1342 bps, above the 650 bps bound. Do not use the default replay as IQL
  authority.
- The selected smart candidate replay policy with SL45/TP90/MFE-protect passes:
  655 trades, net about 3439 bps, profit factor about 3.08, max drawdown about
  418 bps and max loss -45 bps.
- The selected smart IQL replay passes: 735 trades, net about 3886 bps, profit
  factor about 3.13, max drawdown 315 bps and max loss -45 bps. It beats the
  selected candidate by about +447 bps net in the comparison gate, while
  `promotion_shadow_live_allowed=false`.
- Commit `e7aa6762` repairs the next Entry training recipe for the known
  path-signal calibration defect: the trainer now has full-batch
  `path_quality_pred` ranking loss against realized `path_quality_bps`, and
  smoke/candidate wrappers pass the matching `ENTRY_PATH_QUALITY_RANK_*`
  recipe into future vedtak-gated runs. Existing selected smart replay
  artifacts are still old and must remain failed until a new capped smart
  smoke/candidate bundle proves corrected calibration in replay.
- Commit `38170e56` makes that repair a gate, not just trainer code:
  smoke/candidate bundle audit and replay-readiness now require
  `path_calibration_recipe_contract=PASS` with full-batch path-quality ranking
  before any candidate bundle can be replay/IQL authority.
- Commit `20677b4d` makes old selected smart replay-readiness stale when the
  selected IQL slice/path-signal calibration gate is not ready; stale selected
  replay can no longer open IQL distillation authority in the readiness
  summary.
- Smart smoke-manifest, smoke-readiness and trainability gates now require the
  future smart train contract to declare the exact full-batch
  `path_calibration_recipe_contract` plus the six `ENTRY_*_QUALITY_RANK_*`
  env values. Trainability also checks that smoke and candidate wrappers expose
  those envs, so smart training readiness cannot pass unless the repaired
  path-quality/bad-path ranking recipe is actually wired into the next capped
  trainer command.
- Smart smoke/candidate training must also carry an exact direction-balance
  recipe: `ENTRY_PRED_BALANCE_ALPHA=0.05`, `ENTRY_PRED_BALANCE_TARGET=label`,
  positive `ENTRY_DIRECTION_CE_SCALE` and `GX1_V10_CKPT_MONITOR=dir_acc`.
  Bundle audit and candidate-readiness must fail closed when this
  `direction_balance_recipe_contract` is missing or stale, because broad
  direction accuracy around 0.40 is not acceptable without class-distribution
  calibration proof.
- Smart smoke/candidate training must also carry the tail-direction recipe:
  `ENTRY_TAIL_DIRECTION_CE_WEIGHT=0.35`,
  `ENTRY_TAIL_DIRECTION_QUALITY_QUANTILE=0.70` and
  `ENTRY_TAIL_DIRECTION_MIN_BATCH=8`. This focuses extra direction CE on
  directional, tradable, clean-path rows in the top path-quality tail, so the
  next bundle directly optimizes the selected-tail direction weakness instead
  of only reporting it after replay.
- Candidate-readiness now requires the smoke bundle audit to preserve
  `path_calibration_recipe_contract=PASS` before any candidate-train vedtak can
  be considered. This closes old smoke bundles that lack full-batch
  path-quality ranking or positive bad-path/path-quality rank weights, even
  when their older broad metrics still pass.
- Candidate-readiness also requires active LONG/SHORT/FLAT direction
  distribution coverage on audited smoke splits. A bundle that beats majority
  by collapsing away from a common class, including underpredicting FLAT and
  overtrading LONG/SHORT, must remain blocked before candidate training.
- Smoke bundle audit now carries direction context slice diagnostics for
  categorical context buckets. Candidate-readiness must fail closed when an
  audited session/regime/ATR/spread/H4-trend slice with enough rows and label
  diversity fails its own majority baseline or active class-distribution
  coverage. This prevents broad direction accuracy from hiding a broken bucket.
- Smart smoke-manifest, smoke-readiness and trainability reports must declare
  the same `direction_context_slice_contract` as a future post-smoke audit
  requirement before the smart smoke train surface is treated as wired.
- Raw direction accuracy around 0.40 is a weak sanity diagnostic, not the
  primary objective. The smart selected replay has direction precision only
  about 0.478 for candidate and 0.452 for IQL, so the next improvement must
  target class-distribution calibration, selected-tail direction quality,
  bad-path/path-quality semantics and exit timing rather than celebrating broad
  accuracy.
- Replay-readiness now fails closed on weak selected-tail direction quality:
  candidate top-5/top-10 selected tails and supported session/side/vol-regime
  slices must clear the configured direction precision threshold before IQL
  distillation authority can open. PnL filtering may not hide a weak long/short
  direction model.
- Smart selected slice audit now fails closed on path-signal calibration:
  `path_quality_pred` is wrong-signed for net PnL and stop-loss behavior on
  both candidate and IQL, and `bad_path_prob` is wrong-signed versus stop-loss
  behavior. Supported IQL-vs-candidate regressions still exist in volatility
  regime 3, SHORT side and ASIA session, with additional p90 MAE and
  diagnostic regressions. These heads must not be used as policy gates until
  repaired, and smart selected promotion review is blocked by this audit.
- Exit diagnostics show large remaining profit-capture opportunity; IQL peak
  oracle lift is about 12895 bps. This supports Exit Transformer/hazard/IQL as
  the next major improvement after Entry review, but Exit training remains
  closed until an explicit `ENTRY_EXIT_TRANSFORMER_TRAIN_` enablement package.

## Goal

Find tradable XAUUSD edge with sequence models that understand different market
mechanisms, then fuse them into one entry policy and one exit policy trained
against replay/PnL, drawdown and tail-risk objectives. The end state is a fully
automated XAUUSD bot that enters long or short at high-quality points and exits
near maximum profit opportunity while respecting session/regime/tail risk.

Active Objective Rule: all Entry and Exit features with multi-timeframe context
must share one calibrated market-state language before any promotion path is
discussed. The Entry Transformer is the directional evidence layer, Entry IQL is
the entry policy layer, Exit Transformer is the exit-timing evidence layer and
Exit IQL is the exit policy layer. The system must coordinate structure,
liquidity, momentum, trend, volatility, regime, session, spread/ATR,
support/resistance, chart geometry and price action as one replay-proven
trading policy, not as isolated feature families or direction-accuracy scores.

Feature Harmony Rule: every active input must be accounted for before training
or replay can be trusted. Each Entry/Exit feature must either be routed into a
specialist mechanism contract or explicitly excluded with a recorded reason.
The roughly 200 inputs are the shared numeric language for the sequential AI to
read the whole multi-timeframe chart: HH/HL/LH/LL, SMC/liquidity, support and
resistance, Fibonacci/geometry, candle patterns, EMA/trend pressure, momentum,
ATR/spread and session/regime context. Smart specialist layers may compress
and expose stronger mechanism signals to the Transformer, but they do not
erase the base features or their provenance; every smart layer must preserve
source-field coverage, routing ownership, hashes, finite/non-collapsed
liveness and replay evidence. Entry remains first, then Exit must reuse the
same calibrated state from the exact Entry policy traces. Keep the repo clean
at every transition, reuse existing modules and controls where possible, and
create new files only for genuinely new artifacts, gates or model components.

Codex Operating Rule - 2026-07-01: continue the build as one full
Entry-to-Exit bot objective. The 520 smart seq/snap surface, the 142 continuous
context inputs and the five categorical context embeddings are one market-state
language, not competing feature sets. The older foundation features remain
source truth and provenance; smart features are higher-level calibrated
summaries for the specialist Transformer to fuse. Do not remove or bypass a
base feature family just because a smart layer exists until ablation and replay
prove the replacement is better. Entry Transformer must first become the
calibrated evidence layer, Entry IQL the replay-proven entry policy, then Exit
Transformer and Exit IQL must learn profit capture from the exact Entry traces.
If path-quality, bad-path, selected-tail direction, session/regime/side or tail
calibration fails, the system must fail closed even when broad accuracy or old
aggregate replay metrics look acceptable.

Feature Orchestration Rule - 2026-07-01: specialist readiness is not complete
when features are merely counted or routed. Structure, SMC/liquidity,
momentum/flow, trend/EMA, volatility/compression, session/regime, spread/ATR,
support/resistance, chart geometry, price action and multi-timeframe
interactions must be treated as cooperating evidence for one Entry decision
surface. The foundation features, reused sequence extensions, chart/candle
challengers, smart-layer summaries and context embeddings stay available to
the Transformer unless an exact ablation plus replay/slice proof shows removal
improves tradable edge. Each smart specialist must expose calibrated evidence
to the Entry Transformer and preserve the Entry-to-Exit state required for Exit
Transformer/IQL profit capture; isolated feature polish without replay impact
is outside the active objective.

## Operating Doctrine

The objective is a live-equivalent XAUUSD policy, not another isolated
direction-accuracy experiment. Entry is handled first, Exit second, and both
must eventually work as one system across Entry, Entry-IQL, Exit, Exit-IQL,
overlays, sizing and portfolio constraints.

The Entry specialist Transformer is the sequence "eyes": it reads the full
multi-timeframe market-state picture and exposes calibrated specialist evidence.
Entry IQL is the policy "brain": it may only learn from replay evidence tied to
the exact candidate bundle, trade log, artifact hashes and gate identities. The
same Transformer/IQL split is the intended pattern for Exit timing and
profit-capture work after Entry is proven.

Full Bot Objective Lock: the system is being built toward a fully automated
XAUUSD trading bot that enters long or short at high-quality points and exits
near maximum profit opportunity. Entry remains the first active build track:
the Entry Transformer must learn the full multi-timeframe market picture as the
evidence layer, then Entry IQL may become the policy layer only after exact
replay evidence proves tradable edge. Exit work must reuse that same calibrated
language later: Exit Transformer learns exit-timing evidence from the exact
Entry policy traces, and Exit IQL learns profit capture from replay-bound Exit
state. The roughly 200 Entry/Exit inputs are one coordinated market-state
language, not a bag of indicators. HH/HL/LH/LL, SMC/liquidity, momentum/flow,
trend/EMA, volatility/compression, session/regime, support/resistance,
spread/ATR, chart geometry, price action and all multi-timeframe context must
be grounded, routed, hashed, live, non-collapsed and cross-compatible before
any promotion, shadow or live path is discussed.

Foundation features are not decorations and not independent one-off signals.
The roughly 200 Entry/Exit inputs, including HH/HL/LH/LL, SMC/liquidity,
momentum/flow, trend/EMA, volatility/compression, session/regime,
support/resistance, spread/ATR and multi-timeframe context, must be grounded,
audited and cross-compatible so specialist gates can agree, disagree or abstain
under recorded provenance. Broad averages cannot hide weak slices; session,
regime, direction, bad-path and tail behavior remain first-class gates.
Exit features must speak the same calibrated language as Entry: Entry score,
direction probabilities, bad-path probability, path-quality prediction, session,
regime, spread/ATR and realized path state must all be preserved with replay
identity so Exit Transformer/IQL can learn profit capture from the exact Entry
policy that produced the trades.

Keep implementation work inside existing modules, reports and control surfaces
unless genuinely new functionality needs a new file. Create new files only for
genuinely new artifacts, gates or model components that cannot fit existing
ownership boundaries. Do not build UI, docs or model "pynt" ahead of foundation
activation, post-apply verification, train-readiness, exact contracts and clean
worktree/staging evidence.

## Specialist Encoders

Use one shared 96-bar M5 timeline and separate encoders by feature family:

- `structure_swing_encoder`: HH/HL/LH/LL state, BOS/CHoCH age, swing distance,
  pullback phase and structure break recency.
- `smc_liquidity_encoder`: SMC BOS/CHoCH/sweep, sweep reclaim, false breakout,
  premium/discount, support/resistance and wick/liquidity proximity.
- `trend_ema_encoder`: M5/M15/H1/H4/D1 EMA stack, slope, trend age, regime
  agreement and trend exhaustion.
- `vol_compression_encoder`: ATR percentile, range compression, squeeze,
  compression-release and expansion direction.
- `momentum_flow_encoder`: recent returns, impulse direction, acceleration,
  MFE/MAE-conditioned momentum and volatility-adjusted follow-through.
- `session_regime_encoder`: Asia/EU/US/overlap, session age/boundary, vol
  regime, spread bucket and session x structure interactions.
- `chart_geometry_encoder`: challenger specialist for numeric trendlines,
  support/resistance channels, Fibonacci pullback/extension zones, EMA-cross
  pressure and chart-pattern proxies. This is excluded from the active
  `foundation_seq146` contract, but included as a trainable specialist in the
  gated `challenger_seq215` contract after the seq215 rebuild and specialist
  audit.
- `price_action_candle_encoder`: body/wick/range shape, doji/indecision,
  hammer/shooting-star rejection, engulfing/two-candle reversal, inside/outside
  bars and three-candle continuation/reversal pressure that is not already
  assigned to liquidity/structure. This is excluded from active
  `foundation_seq146`, but included in gated `challenger_seq215`.
- `neutral_bridge_anchor`: allowlisted neutral XGB bridge fields only; keep as
  explicit priors until a new bridge is approved.

The trainable specialist set is selected by `specialist_contract_mode`:

- `foundation_seq146`: the six base specialists
  `structure_swing_encoder`, `smc_liquidity_encoder`, `trend_ema_encoder`,
  `vol_compression_encoder`, `momentum_flow_encoder` and
  `session_regime_encoder`.
- `challenger_seq215`: the six base specialists plus
  `chart_geometry_encoder` and `price_action_candle_encoder`.

`neutral_bridge_anchor` and `unmapped` stay excluded from trainable specialist
indices in both modes. Any specialist outside the selected mode is diagnostic
only and must keep smoke/candidate gates closed if it appears as trainable
without matching liveness, role, active-head support and bundle-preservation
evidence.

Current audited seq146 coverage:

- `neutral_bridge_anchor`: 7 signal fields.
- `structure_swing_encoder`: 37 signal fields, 33 selected extension features.
- `smc_liquidity_encoder`: 24 signal fields, 19 selected extension features.
- `trend_ema_encoder`: 6 signal fields.
- `vol_compression_encoder`: 21 signal fields, 9 selected extension features.
- `momentum_flow_encoder`: 5 signal fields, 1 selected extension feature.
- `session_regime_encoder`: 43 signal fields, 43 selected extension features.
- `price_action_candle_encoder`: 3 signal fields.
- Unmapped fields: 0.

Current audited seq215 challenger coverage:

- Base seq: 41 signal fields.
- Reused foundation sequence extension: 105 signal fields.
- New chart geometry challenger: 41 signal fields.
- New candlestick challenger: 28 signal fields.
- Total seq/snap width: 215 signal fields.
- Context width: `ctx_cont=142`, `ctx_cat=5`, sequence length 96.
- Specialist routing: 215/215 signal fields mapped, `unmapped=[]`, no nonfinite
  values found in fullscan snap or ctx_cont.
- `chart_geometry_encoder`: 41 signal fields covering trendline/channel/SR
  geometry, Fibonacci zones, EMA-cross pressure and triangle/flag/compression
  proxies.
- `price_action_candle_encoder`: 31 signal fields, made from 3 existing
  body/wick/range fields plus 28 candlestick challenger fields.

Current report-only smart-layer candidate:

- The optional
  `materialize_entry_specialist_challenger_extension_manifest_v1 --include-smart-layers`
  path preserves the existing seq215 latest files and writes separate `SMART`
  manifest/report latest files.
- It currently adds 305 dormant smart-layer features on top of the 174-feature
  seq215 extension: trend/EMA 20, SMC/liquidity quality 24,
  structure/swing derivations 28, momentum/flow 26, session/regime
  interactions 68, volatility/compression 28, chart-geometry smart2 13,
  price-action/candle smart3 32, support/resistance level memory 34 and
  multi-timeframe confluence 32.
- Expected rebuilt seq/snap width is currently 520 signal fields: 41 base
  signal fields plus the 479-feature combined extension. This is a dataset rebuild candidate,
  not active training evidence.
- The Entry feature AI inventory proves all ten smart layers have required
  source coverage. The inline sequence-extension builder is wired to materialize
  these smart fields from the existing emitted signal/context fields plus the
  active source parquet. `smart-rebuild-preflight` must bind the latest smart
  manifest, inventory, source parquet, hashes and 4G-capped rebuild command
  before any smart dataset rebuild is reviewed.
  audited OHLC source parquet for candle smart3 during a later capped rebuild.
- These smart layers are diagnostic/report-only until a separate rebuild,
  feature audit, specialist audit, liveness/non-collapse proof and
  train-readiness gate explicitly promote them into a trainable contract.

Foundation requirement mapping is audited as:

- HH/HL/LH/LL -> `structure_swing_encoder`.
- BOS/CHoCH age -> `structure_swing_encoder`.
- Sweep reclaim / false breakout -> `smc_liquidity_encoder`.
- Compression-expansion -> `vol_compression_encoder`.
- Impulse/pullback phase -> `structure_swing_encoder`.
- Session x structure -> `session_regime_encoder`.

The audit also enforces exact objective routing: every required foundation
feature from the objective coverage contract must be present and mapped to its
expected specialist, with zero missing or misrouted features.
It also enforces the `SPECIALIST_MODEL_CONTRACT`: every trainable specialist
must have a model role, non-empty primary signal families, active-head support
only, and exact ownership of its roadmap objectives.

## Fusion And Heads

Fuse specialist embeddings with a gated mixture layer:

- Gate inputs: session, volatility regime, spread bucket, trend regime and
  structure state.
- Output heads: direction, tradable, bad-path/tail-risk, path quality,
  position-size, dip/forecast/timing, forward volatility and MTF-direction.
  Hold-horizon remains blocked until its target is non-constant in every split.
- The specialist audit's `architecture_contract.recommended_fusion` must match
  the target-head contract exactly: all active heads are listed under
  `active_heads`, and `hold_horizon` is listed only under `blocked_heads`.
- The trainer's specialist-fusion loader must return the exact trainable set
  for the declared `specialist_contract_mode`: six specialists for
  `foundation_seq146`, eight specialists for `challenger_seq215`. Extra
  classified groups are diagnostics only until promoted by a separate gate.
- Keep neutral XGB bridge inputs explicit and allowlisted until a new bridge is
  intentionally approved.

## Training Sequence

0. Start fresh sessions with `scripts/entry_next_edge_control.sh handover` and
   require it to report active Entry foundation seq146 before touching training,
   replay, IQL, shadow or live paths. `verify`/`selftest` must cover the
   readiness-policy snapshot and critical-gate path coverage contracts. Use
   `scripts/entry_next_edge_control.sh readiness-report` for a report-only
   light refresh of worktree/train-readiness plus latest candidate, replay,
   IQL and Exit reports, or add `--json` for a machine-readable report. Use
   `readiness-report --snapshot` only for a strict latest-report read and
   `readiness-report --refresh` only when an explicit full report refresh is
   intended; full refresh may rerun large smart hash/fullscan gates but still
   must not stage, train, replay, distill, shadow or touch live paths.
   The report includes worktree stage/hold counts and critical-gate path
   coverage for cleanup review. Machine agents must use
   `commands.*.execution_allowed_now` for autonomous report/audit/dry-run
   actions and `commands.*.allowed_after_explicit_vedtak` only after a real
   vedtak id is supplied; placeholder commands containing `--vedtak <id>` are
   not autonomous now-commands. The JSON command policy must explicitly show
   candidate training, replay evidence, IQL distillation, shadow and live paths
   as blocked until their gates pass.
1. Run feature foundation audit and target foundation audit on the rebuilt
   dataset. The feature audit must prove each required roadmap family is live
   in train/val/test: HH/HL/LH/LL, BOS/CHoCH age, sweep reclaim,
   compression-expansion, impulse/pullback and session x structure. It must
   also prove every foundation source field used by these features is present
   in each emitted split contract and live as raw `snap.*`/`ctx_cont.*` input
   on train/val/test.
2. Run specialist feature group audit and require zero unmapped seq/snap fields
   plus exact foundation objective routing to the expected specialist encoders.
3. Run `scripts/entry_next_edge_control.sh foundation-guardrails` and require
   PASS. This proves old no-XGB shadow/live entrypoints fail closed under the
   active foundation-freeze and verifies the readiness JSON command policy so
   only safe audit/dry-run commands are executable now.
4. Run `scripts/entry_next_edge_control.sh worktree-hygiene --no-fail-on-dirty`
   while cleaning up. It classifies active foundation, legacy tombstone,
   entry-related review and unrelated dirty paths; final real training requires
   `PASS_CLEAN_GIT`. The audit writes explicit stage-candidate and review-hold
   path lists plus a `git add --dry-run` proof so the foundation cleanup can be
   isolated before smoke training without changing the index. Use the stage
   list only when the audit reports `stage_plan_safe=true`, and review the
   generated status TSV files before actual staging. The audit must also report
   `foundation_cleanup_review_decision=PASS`,
   `foundation_cleanup_critical_gate_review` for docs/control/guardrail,
   smoke-bundle, candidate-readiness, replay-readiness, IQL
   distillation/evidence/comparison gate files and their tests, and
   `foundation_cleanup_stage_ready=true`. If the stage command is later run,
   rerun the audit and require `PASS_STAGED` in the post-stage verification.
   The canonical wrapper is
   `scripts/entry_next_edge_control.sh stage-foundation-cleanup`; it defaults to
   dry-run and requires `--apply --vedtak <id>` for actual staging.
5. Run `scripts/entry_next_edge_control.sh train-readiness` and require
   `READY_FOR_VEDTAK_SMOKE_TRAIN` before real trainer start.
   `READY_FOR_VEDTAK_SMOKE_TRAIN_AFTER_GIT_CLEAN` means the foundation contract
   is ready but execution hygiene still blocks real training. The readiness
   contract also covers the current 2026-06-29 activation bridge state:
   `NOT_READY` with `foundation_activation_required_before_smoke=true` means a
   green directional-SMC candidate, activation plan and activation-apply dry-run
   exist, but canonical active paths have not been switched. The next action is
   only `foundation-activation-apply --apply --vedtak <id>`, followed by the
   post-apply audit refresh commands, `verify --quiet` and another
   `train-readiness`; it does not approve trainer start. The activation plan
   must also prove the adoption gates are all PASS, re-hash the adoption audit
   and smoke-manifest fingerprints, and prove current apply, post-apply and
   control-surface source-pointer checks before activation apply is accepted.
   The smoke-train wrapper also reruns `foundation-guardrails` and
   `train-readiness` before real
   training; dry-run and materialize-only paths only print those preflight
   commands and do not recurse through readiness. Manifest-only may run while
   git is dirty, but only if `foundation_contract_ready_for_smoke=true`.
   Train-readiness must also require the guardrail artifact's readiness JSON
   command-policy proof, so stale guardrails cannot open trainer start.
   The smoke-dataset manifest must also preserve
   `entry_foundation_smoke_dataset_audit_provenance_v1` with SHA256s for the
   active feature, target and specialist audits, and train-readiness must reject
   it when those hashes no longer match the active audit artifacts. It must also
   re-hash the smoke split source manifests, output parquets and output
   manifests before smoke training is opened.
6. Optionally run
   `scripts/entry_next_edge_control.sh smoke-manifest --vedtak <id>` to write
   the pre-train manifest/provenance proof without starting the trainer.
   For the seq215 challenger, use
   `scripts/entry_next_edge_control.sh smoke-manifest-seq215 --vedtak <id>`
   with an explicit vedtak id containing `SEQ215`. It must write the manifest
   for `specialist_contract_mode=challenger_seq215`, `seq_input_dim=215`, the
   seq215 smoke dataset and the seq215 specialist audit, but still stop before
   trainer start.
7. Train a short smoke model with frozen XGB-neutral bridge and all heads
   enabled according to the machine target-head contract. Current active heads:
   direction, tradable, path quality, MFE-first-N, bad-path, clean-edge,
   survival, TF-agreement, path-quality variance, position-size, dip, forecast,
   timing, tail-risk, vol-forecast and MTF-direction. `hold_horizon` is blocked
   because `y_hold_horizon_target` is constant in train/val/test.
8. Let
   `scripts/entry_next_edge_control.sh smoke-train --vedtak <id> --require-edge-audit`
   run from a clean git worktree, write its pre-train run manifest, then run
   its automatic post-train `audit-smoke-bundle` gate to prove strict runtime
   load, finite forward pass, required specialist metadata,
   non-collapsed specialist-gate liveness, active target-head contract,
   pre-train manifest provenance, direction baselines and bad-path/path-quality
   diagnostics. The audit must
   match the target audit's active heads to bundle capabilities,
   `train_recipe.active_heads`, state_dict heads, forward output keys and
   expected output shapes, while keeping blocked heads absent. The active and
   blocked head sets must be exact: extra unsupported or experimental heads are
   not allowed in capabilities, `train_recipe.active_heads`, state_dict heads or
   forward outputs. The pre-train manifest must include audit artifact SHA256s
   and compact summaries of feature objective coverage, exact objective
   liveness on train/val/test, foundation source-field coverage and liveness,
   target-head contract, specialist objective routing, specialist input
   liveness, the trainer loader's exact trainable and excluded specialist
   groups, smoke-dataset audit provenance, smoke split hashes and worktree
   hygiene, including critical-gate path review.
   It must also report the specialist-fusion gate contract: the exact required
   training specialist group set, minimum active specialist count, minimum gate
   entropy, and each required specialist's mean gate weight above 1% on every
   audited split. Extra ungated specialist groups keep the audit closed. The
   trainer must also preserve the exact specialist model contract for the
   declared mode in trained bundle metadata, and the post-smoke bundle audit
   must emit PASS for `bundle_specialist_model_contract`: exact six trainable
   specialist AIs for `foundation_seq146` or exact eight trainable specialist
   AIs for `challenger_seq215`, with owned roadmap objectives, signal families,
   support heads and model roles. The pre-train contract must preserve the exact
   active/blocked head split, with `hold_horizon` blocked. Edge
   diagnostics are the smoke-wrapper default; `--no-require-edge-audit` is an
   explicit plumbing-only opt-out and `--require-edge-audit` is intentionally
   incompatible with `--skip-smoke-audit`.
   The seq215 smoke path is
   `scripts/entry_next_edge_control.sh smoke-train-seq215 --vedtak <id> --require-edge-audit`.
   It requires an explicit `SEQ215` vedtak id, uses the seq215 smoke dataset,
   `specialist_contract_mode=challenger_seq215`, the 8-specialist audit, and
   writes the post-smoke audit under
   `entry_foundation_smoke_bundle_audit_20260628_v1/challenger_seq215_20260630`.
   It does not approve candidate training, replay, IQL, shadow, live or
   promotion.
9. Require direction to beat majority baseline and bad-path probability to be
   negatively related to path quality before treating the smoke as learning
   evidence. Plumbing-only smoke audits may pass without `--require-edge`.
10. Run `scripts/entry_next_edge_control.sh candidate-readiness` and require
   `READY_FOR_CANDIDATE_TRAINING_VEDTAK`. For seq215, run
   `scripts/entry_next_edge_control.sh candidate-readiness-seq215` and require
   the same decision under `contract_mode=challenger_seq215`. It must stay
   `NOT_READY_FOR_CANDIDATE_TRAINING` until the real seq215 smoke bundle edge
   audit exists, is readable, matches `seq_input_dim=215`, preserves the exact
   eight specialists and proves non-collapsed gate liveness.
11. Train the candidate specialist-fusion Transformer through
   `scripts/entry_next_edge_control.sh candidate-train --vedtak <id>`. The
   wrapper must then strict-load the candidate bundle on the full foundation
   val/test splits and require the same active-head/output-shape contract before
   replay evidence is considered. The candidate pre-train manifest must carry
   forward the smoke edge audit's exact required specialist set and
   smoke-dataset audit-provenance flags, plus worktree critical-gate proof and
   the candidate trainer loader's exact trainable and excluded specialist groups,
   so the candidate bundle audit
   can prove the candidate still descends from the audited smoke sample and did
   not train an ungated specialist. The candidate bundle audit must also prove
   the trained bundle still preserves a valid `bundle_specialist_model_contract`;
   pre-train manifest provenance alone is not enough. Replay-readiness
   must consume this
   `entry_candidate_bundle_audit_20260628_v1` artifact and require PASS before
   IQL distillation can open. It must also require the candidate bundle audit's
   pre-train manifest contract to have validated feature objective coverage,
   exact objective liveness, raw source-field liveness, specialist routing,
   specialist input liveness, specialist active/blocked head parity and
   smoke-dataset audit-provenance hashes, plus the bundle-level specialist model
   contract preservation check.
   For seq215, use
   `scripts/entry_next_edge_control.sh candidate-train-seq215 --vedtak <id>`
   only after `candidate-readiness-seq215` is green and the vedtak id contains
   `SEQ215`; it remains closed before that even when base seq146 candidate
   readiness or historical IQL evidence exists.
12. Run selective-edge evaluation with no-XGB ablation through
   `scripts/entry_next_edge_control.sh selective-edge --bundle-dir <candidate> --no-xgb-bundle-dir <ablation>`.
   This writes `summary.json` and `selective_edge_metrics.csv` for
   `replay-readiness`. The replay-readiness gate must match this summary's
   `bundle_dir` back to the candidate bundle audit artifact so selective-edge
   evidence cannot be mixed across candidates.
13. Run 2026 offline replay and materialize its explicit trade log through
   `scripts/entry_next_edge_control.sh replay-evidence --trades-path <csv|parquet>`.
   This writes `replay_policy_trades.csv`, `replay_policy_metrics.csv` and
   `replay_policy_monthly.csv`. The trade log must include IQL transition
   minimum fields: `entry_time`, `policy_id`, `session`, `side`, `score`,
   `p_long`, `p_short`, `p_flat`, `net_pnl_bps`, `mfe_bps`, `mae_bps` and
   `held_bars`. The replay evidence manifest must also record the candidate
   bundle audit and selective-edge summary paths, and the `bundle_dir` identity
   must match across those artifacts.
14. Require
   `scripts/entry_next_edge_control.sh replay-readiness`.
15. Open IQL distillation only through
   `scripts/entry_next_edge_control.sh iql-distill --vedtak <id>`. This writes
   the foundation distillation contract and remains research-only until a
   post-distillation replay comparison exists. The contract must preserve the
   replay-readiness evidence identity and SHA256s for the replay-readiness
   report, candidate-readiness report, candidate bundle audit, selective-edge
   summary/metrics, and candidate replay manifest/metrics/monthly/trades, so the
   IQL student remains tied to the exact candidate bundle and replay evidence.
   It must also re-check that replay-readiness preserved the candidate
   pre-train manifest provenance for feature/source liveness, specialist
   active/blocked head parity, exact smoke specialist-set preservation, valid
   specialist model contract preservation, exact owned-roadmap-objective mapping
   and the smoke-dataset audit-provenance flags. It must separately re-check the
   replay-readiness gate proving the candidate bundle's
   `bundle_specialist_model_contract` remained valid after training; a ready
   decision without those explicit checks is not enough to open IQL.
   IQL replay evidence and the final comparison gate must reject any
   distillation contract where any pre-train, smoke-dataset, specialist-set,
   specialist-model, bundle-specialist-model or replay-artifact provenance
   contract is not `ok`.
16. Distill the candidate into the IQL entry policy using replay rewards:
   realized PnL, drawdown, MAE tail, path quality, trade duration and missed
   opportunity.
17. Materialize IQL-student replay evidence from an explicit trade log through
   `scripts/entry_next_edge_control.sh iql-replay-evidence --trades-path <csv|parquet>`.
   This writes the IQL student's `replay_policy_trades.csv`,
   `replay_policy_metrics.csv`, `replay_policy_monthly.csv` and
   `REPLAY_EVIDENCE_MANIFEST.json`, preserving the distillation contract's
   `evidence_identity`. The trade log's `policy_id` must match the requested
   IQL student policy, and the distillation contract's artifact hashes plus
   candidate replay manifest must still validate.
18. Require `scripts/entry_next_edge_control.sh iql-compare`. The IQL student
   must beat the candidate on replay net PnL, not degrade profit factor,
   drawdown or max loss, and have no negative months before promotion review can
   even be discussed. Both candidate and IQL replay directories must carry
   `REPLAY_EVIDENCE_MANIFEST.json` identity matching the distillation
   contract's `evidence_identity`; the IQL manifest must also prove it validated
   the distillation artifact hash contract, so a replay CSV from another
   candidate cannot pass the comparison gate.
19. Continue improvement with offline replay logs only; no online self-learning
   or live promotion without explicit gate reports.

## Promotion Gates

- Feature audit PASS on exact dataset used for training.
- Target audit PASS on exact dataset used for training.
- Strict bundle load PASS.
- 2026 replay improves net PnL, max drawdown and tail loss, not just accuracy.
- Slice reports by session, side, volatility regime and trend regime.

## Blueprint V2 Delta

This delta extends the current sequential specialist blueprint with the audit
findings that are not yet fully captured above. It does not replace the current
foundation gate sequence. It tightens what must be proven before Entry, IQL,
Exit alignment or live/shadow promotion can be discussed.

### Artifact Contract And Activation

- Treat the legacy Entry contract as closed until active artifact paths resolve
  to existing, strict-loadable foundation artifacts.
- Add a contract-path existence gate for every active role in
  `PROJECT_STATE_artifacts.json`: `xgb`, `v10_entry`, `v3_exit`,
  `entry_iql` and `exit_iql`.
- The gate must fail if any active path points to a legacy tombstone,
  missing directory or non-loadable bundle, even if historical reports exist.
- Activation remains vedtak-gated: foundation activation apply, post-apply
  audit refresh, `verify --quiet`, then `train-readiness`.
- No training, replay, shadow or live promotion may use a candidate path that
  is not tied back to the exact feature, target, specialist and bundle audits.

### Metrics That Matter

- Do not optimize full-sample direction accuracy as the primary goal.
- Required entry metrics: selected-tail direction accuracy, trade rate,
  net bps per trade, profit factor, max drawdown, max loss, bad-path avoidance,
  tail loss, calibration and month-by-month stability.
- Required policy metrics: cap-3 portfolio behavior, same-side clustering,
  correlated-entry drawdown, rejected-trade opportunity cost and time in market.
- Required slice metrics: session, side, volatility regime, trend regime,
  ATR percentile, spread bucket, compression state, BOS/CHoCH state,
  sweep/reclaim state, false-breakout state and MTF agreement state.

### Specialist Extensions

- Keep `price_action_candle_encoder` excluded from active `foundation_seq146`.
  In `challenger_seq215`, it is already a gated trainable specialist with its
  own role contract and target-head support, but it still has no candidate
  authority until seq215 smoke edge audit, candidate-readiness, replay and IQL
  gates prove tradable edge.
- Expand price-action features with wick rejection, body expansion, inside/outside
  bars, engulfing behavior, close-location value, wick-to-ATR ratios, rejection
  at support/resistance and candle shape conditioned on SMC context.
- Strengthen `momentum_flow_encoder`; current audited coverage is intentionally
  thin and must not be treated as complete momentum intelligence.
- Add momentum features for multi-horizon return slope, acceleration,
  volatility-normalized impulse, exhaustion, signed volume/flow proxy when
  available, return skew and momentum follow-through after pullback.
- Preserve the six `foundation_seq146` trainable specialists as the base
  canonical fusion contract. Preserve the eight `challenger_seq215` trainable
  specialists as a separate challenger contract. New specialists or promoted
  diagnostic groups require separate liveness, routing, active-head support,
  contract-mode plumbing and bundle-preservation gates.

### Auxiliary Specialist Models

Auxiliary models are allowed only as audited heads, frozen features or gated
specialists inside the foundation contract. They must not become independent
uncontrolled entry bots.

- `regime_classifier`: trend, range, chop, breakout, compression and reversal.
- `momentum_continuation_model`: continuation probability over short forward
  horizons after impulse or pullback.
- `breakout_failure_model`: BOS/CHoCH plus sweep, wick and level proximity into
  false-breakout probability.
- `vol_expansion_model`: compression release, ATR expansion and directional
  volatility breakout risk.
- `liquidity_sweep_reclaim_model`: sweep-and-reclaim success/failure
  probability by session and structure state.
- `session_behavior_model`: Asia, EU, US and overlap drift, volatility and
  spread behavior.
- Each auxiliary model must report calibration, liveness, feature coverage,
  split stability and contribution in ablation before it can influence policy.

### Feature Interaction Contracts

The candidate trainer and replay reports should preserve explicit interaction
families so the model can be audited by market mechanism, not just by raw
feature names.

- BOS/CHoCH plus EMA stack plus H4/D1 agreement -> trend continuation bias.
- Sweep reclaim plus wick rejection plus level proximity plus session ->
  reversal or false-breakout filter.
- Compression plus ATR percentile plus spread bucket plus session boundary ->
  breakout readiness.
- Impulse strength plus pullback depth plus trend age -> continuation versus
  late-chase risk.
- Direction margin plus bad-path probability plus tail-risk plus path-quality
  variance -> uncertainty-adjusted entry threshold and position size.
- Session x structure already exists; add session x volatility expansion x
  spread as an explicit audited interaction family.

### Exit Alignment

- Exit must consume enough Entry foundation context to manage trades according
  to why they were entered.
- At minimum, Exit/V3/Exit-IQL alignment must cover structure state, BOS/CHoCH,
  sweep/reclaim, false breakout, compression/expansion, EMA/trend agreement,
  ATR regime, session regime and the Entry specialist gate outputs.
- The `challenger_seq215` path must not inherit the six-gate Exit alignment from
  `foundation_seq146`; it requires chart geometry state, price action/candle
  state and all eight specialist gate weights.
- Add an exit feature-alignment gate that fails when new Entry foundation
  features are missing from the Exit state without an explicit waiver.
- Evaluate exit policy separately for sweep-reversal entries, BOS-continuation
  entries, compression-breakout entries and mean-reversion/chop entries.
- Distilled Exit-IQL Q heads in V3 may be promoted only after replay comparison
  proves they improve net PnL, drawdown and tail loss without degrading
  profit-factor or negative-month count.

### Replay And Live Parity

- Add Entry live-vs-replay parity tests for action, conviction gate, no-XGB
  ablation behavior, DIPFIX, skip-ASIA, ATR sizing, margin sizing and cap-3
  portfolio constraints.
- Add Exit live-vs-replay parity tests for V3 window construction, M5 phase,
  M1 ATR, regime one-hots, trade-state overlay, Strategy-F, let-winners-run and
  hold-horizon expiration.
- Replay evidence must be generated from explicit trade logs and must preserve
  candidate identity, policy id, artifact hashes and all gate identities.
- Candidate and IQL comparison must use the same live-equivalent overlays that
  would actually run in shadow/live. A replay that omits production overlays is
  research-only evidence.

### Split, Leakage And Robustness Tests

- Require chronological walk-forward splits with purge/embargo for both Entry
  and Exit training paths unless a gate explicitly proves an equivalent split.
- Add a test proving the main Exit-IQL materialization path uses the intended
  chronological group-purged resolver, not a stratified shortcut.
- Add adversarial validation between train/val/test and between historical years
  to expose distribution drift before candidate promotion.
- Add feature ablation and permutation tests by specialist group: structure,
  SMC/liquidity, trend/EMA, volatility/compression, momentum/flow and
  session/regime.
- Add calibration tests for direction, tradable, bad-path, path-quality,
  tail-risk, position-size and auxiliary-model outputs.
- Add stress tests for spread widening, slippage, high-volatility periods,
  consecutive-loss runs, large MAE paths, rollover/news windows when data is
  available and missing/late feature updates.

### Model Zoo And Challenger Baselines

Transformer plus IQL remains the primary architecture, but it must beat strong
baselines under the same replay and cost assumptions.

- Maintain tabular challenger baselines: XGBoost, LightGBM or CatBoost for
  direction, tradable, bad-path, path-quality and final policy gating.
- Maintain sequence challenger baselines where practical: PatchTST, TSMixer,
  TimesNet, Temporal Fusion Transformer or N-BEATS-style forecasters.
- Maintain policy challenger baselines: behavior cloning, advantage-weighted
  regression/classification, CQL, Decision Transformer and survival/hazard
  timing models.
- A challenger model may be promoted only if it passes the same artifact,
  replay, calibration, ablation, slice and live-parity gates as the primary
  specialist Transformer/IQL stack.
- If the specialist Transformer wins only on accuracy but loses on replay PnL,
  drawdown, tail loss or calibration, it is not promotion-ready.

### Production Readiness Rule

The full bot is not ready because one component has a good isolated metric. It
is ready only when Entry, Entry-IQL, Exit, Exit-IQL, overlays, sizing and
portfolio constraints are evaluated as one live-equivalent policy with strict
artifact identity, calibrated probabilities, robust regime slices, acceptable
drawdown and no untested production-only behavior.
