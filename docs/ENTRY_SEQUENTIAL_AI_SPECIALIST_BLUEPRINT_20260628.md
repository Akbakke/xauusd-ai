# Entry Sequential AI Specialist Blueprint

Status: machine-audited feature grouping contract exists. Do not promote from
this until smoke training, replay and explicit promotion gates pass.

Machine contract:

- Registry: `gx1/features/entry_specialist_feature_groups_v1.py`
  - `SPECIALIST_MODEL_CONTRACT` is the machine-readable role contract for the
    six trainable specialist AIs: owned roadmap objectives, primary signal
    families and supported active heads.
- Audit: `gx1/scripts/audit_entry_specialist_feature_groups_v1.py`
  - The audit writes `specialist_model_contract`,
    `specialist_model_contract_valid` and
    `specialist_model_contract_failures`.
- Post-smoke bundle audit:
  `gx1/scripts/audit_entry_foundation_smoke_bundle_v1.py`
  - Must validate both the pre-train manifest specialist model contract and the
    trained bundle metadata's preserved `bundle_specialist_model_contract`.
- Train-readiness gate:
  `gx1/scripts/verify_entry_training_readiness_v1.py`
- Worktree-hygiene gate:
  `gx1/scripts/audit_entry_foundation_worktree_hygiene_v1.py`
  - Critical-gate path coverage includes docs/control, smoke bundle audit,
    candidate-readiness, replay-readiness, IQL distillation/evidence/comparison
    gates and their tests.
- Candidate-readiness gate:
  `gx1/scripts/verify_entry_candidate_readiness_v1.py`
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
    cgroup RAM cap declaration, `--num-workers 0` and no
    train/replay/IQL/shadow/live side effects.
- Active Exit Transformer trainer core:
  `gx1/models/exit_sequence_transformer/train_v1.py`
  - Defines the active causal masked `ExitSequenceTransformerV1` with exact
    output heads and a CPU-only `--preflight-only` path. Non-preflight training
    exits fail-closed.
- Active Exit Transformer pretrain manifest:
  `gx1/scripts/materialize_entry_exit_transformer_pretrain_manifest_v1.py`
  - Runs a finite forward preflight on active train episodes, writes the
    pretrain manifest, records zero optimizer steps and keeps Exit
    training/replay/IQL/shadow/live closed until train-execution review.
- Latest report:
  `/home/andre2/GX1_DATA/reports/entry_specialist_feature_group_audit_20260628_v1/ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.json`
- Current decision: `PASS` on the active seq146 foundation dataset.

## Goal

Find tradable XAUUSD edge with sequence models that understand different market
mechanisms, then fuse them into one entry policy and one exit policy trained
against replay/PnL, drawdown and tail-risk objectives. The end state is a fully
automated XAUUSD bot that enters long or short at high-quality points and exits
near maximum profit opportunity while respecting session/regime/tail risk.

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
unless genuinely new functionality needs a new file. Do not build UI, docs or
model "pynt" ahead of foundation activation, post-apply verification,
train-readiness and clean worktree/staging evidence.

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
- `price_action_candle_encoder`: body/wick/range shape that is not already
  assigned to liquidity/structure.
- `neutral_bridge_anchor`: allowlisted neutral XGB bridge fields only; keep as
  explicit priors until a new bridge is approved.

Only the six required training specialists above `price_action_candle_encoder`
are trainable in the current fusion contract. `price_action_candle_encoder`,
`neutral_bridge_anchor` and `unmapped` stay excluded from trainable specialist
indices until they have their own liveness and role gates.

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
- The trainer's specialist-fusion loader must return the exact trainable set:
  structure/swing, SMC/liquidity, trend/EMA, vol/compression, momentum/flow and
  session/regime. Extra classified groups are diagnostics only until promoted by
  a separate gate.
- Keep neutral XGB bridge inputs explicit and allowlisted until a new bridge is
  intentionally approved.

## Training Sequence

0. Start fresh sessions with `scripts/entry_next_edge_control.sh handover` and
   require it to report active Entry foundation seq146 before touching training,
   replay, IQL, shadow or live paths. `verify`/`selftest` must cover the
   readiness-policy snapshot and critical-gate path coverage contracts. Use
   `scripts/entry_next_edge_control.sh readiness-report` for a report-only
   snapshot across train, candidate, replay, IQL distillation, IQL replay
   evidence and IQL comparison gates, or add `--json` for a machine-readable
   snapshot. The report includes worktree stage/hold counts and critical-gate
   path coverage for cleanup review. Machine agents must use
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
   trainer must also preserve the exact `SPECIALIST_MODEL_CONTRACT` in trained
   bundle metadata, and the post-smoke bundle audit must emit PASS for
   `bundle_specialist_model_contract`: exact six trainable specialist AIs,
   owned roadmap objectives, signal families, support heads and model roles. The
   pre-train contract must preserve the exact active/blocked head split, with
   `hold_horizon` blocked. Edge
   diagnostics are the smoke-wrapper default; `--no-require-edge-audit` is an
   explicit plumbing-only opt-out and `--require-edge-audit` is intentionally
   incompatible with `--skip-smoke-audit`.
9. Require direction to beat majority baseline and bad-path probability to be
   negatively related to path quality before treating the smoke as learning
   evidence. Plumbing-only smoke audits may pass without `--require-edge`.
10. Run `scripts/entry_next_edge_control.sh candidate-readiness` and require
   `READY_FOR_CANDIDATE_TRAINING_VEDTAK`.
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

- Promote `price_action_candle_encoder` only after its own liveness, role and
  target-support gates pass.
- Expand price-action features with wick rejection, body expansion, inside/outside
  bars, engulfing behavior, close-location value, wick-to-ATR ratios, rejection
  at support/resistance and candle shape conditioned on SMC context.
- Strengthen `momentum_flow_encoder`; current audited coverage is intentionally
  thin and must not be treated as complete momentum intelligence.
- Add momentum features for multi-horizon return slope, acceleration,
  volatility-normalized impulse, exhaustion, signed volume/flow proxy when
  available, return skew and momentum follow-through after pullback.
- Preserve the six current trainable specialists as the base fusion contract.
  New specialists or promoted diagnostic groups require separate liveness,
  routing, active-head support and bundle-preservation gates.

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
