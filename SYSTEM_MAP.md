# GX1 current system map

This is the current architecture map for XAUUSD Entry direction. Retired
rule-gated, anchored and Entry-RL flows are not documented as alternatives
because they have no Entry authority.

The immutable data front end is:

```text
native OANDA M1 v3 + native OANDA M5 v3
        |
        v
entry_next_edge_control.sh model-native-canonical-pair
        |
        +-- canonical-v3: persisted model-agnostic M5 feature surface
        +-- raw BASE28: exactly 13 native M1 fields
        +-- pair lineage: native/source/code/formula/timing/schema/coverage
        |
        v
chronological split -> TRAIN-only rank reference -> Entry and Exit consumers
```

M5-derived evidence becomes available at `bar_start + 5 minutes`. The same
decision timestamp owns HTF joins, cyclic time and session state. TRAIN-fit
ATR/spread buckets are forbidden from the raw pair and mutable global bucket
files have no authority.

## Decision flow

```text
canonical XAU tapes + derived market state
        |
        v
exact split manifests and chronological leak-safe rows
        |
        +-- 34 genuine base price-state fields
        +-- 479 specialist fields
        |     +-- 378 mandatory outputs from 12 causal layers
        |     +-- 101 deterministic TRAIN-only ranked fields
        +-- 142 continuous + 5 categorical context fields
        |     +-- exact one-owner routing into the eight specialist tokens
        +-- M5/M15/H1/H4/D1 sequences with recipe-owned lengths
              +-- terminal V7 used 16/16/16/8/8
        |
        v
full physical-TRAIN robust normalization + exact MTF V2 cache-byte binding
        |
        v
full-field liveness + field order/hash + feature/target/specialist audits
        |
        v
eight signal+owned-context specialist encoders + temporal/MTF/cross-TF/FiLM fusion
        |
        v
22 supervised evidence heads
        |
        v
exact 26-group / 96-dimension evidence fusion -> 128 hidden -> 3 logits
        |
        v
immutable calibration on declared calibration data
        |
        v
shared exact runtime evidence contract
        |
        +-- calibrated logits [LONG, SHORT, FLAT]
        +-- hierarchy/path/utility/MTF/specialist evidence
        +-- mandatory learned position_size evidence
        |
        +-- argmax is the only direction authority
        +-- learned sizing is separately calibrated and proof-bound
        |
        v
immutable prediction evidence -> unit-normalized direction replay -> pocket/slice audits
                                      |
                                      +-- separate sizing OOS + exact active-Exit replay
                                          + post-adoption runtime parity
        |
        v
launch contract ALLOW only after newest complete evidence passes
```

Any missing or inconsistent edge in this flow stops the path. There is no
secondary Entry model, hand-written filter, default decision or stale bundle
selector.

## Input ownership

`gx1/contracts/entry_model_native_signal_v1.py` owns the exact static Entry
shape and forbids the seven retired bridge fields:

- contract mode `xau_seq513_model_native_direction_v4`;
- direction mode `model_native`;
- 34 base + 479 specialist = 513 signal fields;
- the specialist surface is 378 mandatory code-owned causal-layer outputs plus
  101 deterministic TRAIN-only ranked fields;
- sequence length 96;
- 142 continuous + 5 categorical context fields;
- no bridge source and no anchor source.

This contract is the one exact owner of the 34 base, 142 continuous-context
and 5 categorical-context identities. Active Entry has zero imports from
`signal_bridge_v1` or `signal_bridge_v3`; those modules cannot supply,
reorder or repair Entry state.

The 21-field V1 context prefix is likewise partitioned only here: source6,
micro5, swing5 and session5. `micro_structure_v1` and `swing_structure_v1`
own the causal formulas used by prebuild, dataset build and serving. During the
dataset join canonical-v2 exclusively owns base30 + context-v2-19, while the
exact source prebuilt exclusively owns context-v3-5 + regime-source15 + raw
volume. A duplicated or missing owner field is a hard failure; there is no
source/canonical preference fallback.

The complete 479-field order is manifest-owned and hash-bound. The first 378
positions must exactly equal the immutable causal-layer registry; the final
101 come from one validated deterministic TRAIN-only ranking. It cannot be
reconstructed from a mutable registry at train or serve time.

`gx1/features/entry_model_native_feature_layers_v1.py` owns the twelve mandatory
causal layers: the 57-field foundation cross-family layer, trend/EMA,
SMC/liquidity, structure/swing, momentum/flow,
session/regime, volatility/compression, chart geometry,
price-action/candles, support/resistance memory, MTF confluence and exact M5
EMA50/200 state/cross evidence.
`gx1/contracts/entry_structural_aux_label_signal_v1.py` owns the 19 named
current-bar signal requirements consumed by structural auxiliary-label
construction. Every requirement must resolve inside the mandatory 378-field
prefix. The dataset builder imports this registry directly; optional ranking
cannot decide whether target construction is possible.
`gx1/contracts/entry_pretrain_polarity_signal_v1.py` owns the four exact
geometry inputs used to prove support/resistance channel polarity. They are
also mandatory and embedded in the signal contract; audit target consistency
is evaluated independently so missing polarity cannot hide target failures.
`gx1/features/entry_volatility_semantics_v1.py` owns the sign-preserving
conversion of canonical ATR14/ATR100 ratios and relative Bollinger bandwidth
into separate compression and expansion pressure. Feature families may not
interpret those raw sources independently.
The empirical field-by-field foundation routing and duplicate audit is recorded
in `docs/FOUNDATION_FEATURE_ROUTING_AUDIT_20260722.md`.
`gx1/features/entry_specialist_feature_groups_v1.py` routes the complete
479-field surface into the exact eight-way learned specialist partition:

1. structure/swing;
2. SMC/liquidity;
3. trend/EMA;
4. volatility/compression;
5. momentum/flow;
6. session/regime;
7. chart geometry;
8. price-action/candles.

Features may inform several learned interactions, but each emitted field has
one exact ordered input identity. Genuine trend/session evidence is mandatory;
only disconnected direction filters are retired.

This means every genuine form of structure, multi-timeframe trend, liquidity,
volatility, momentum, session/regime, chart geometry and candle/price action is
retained in the eight-specialist learned path. Path quality and utility are
supervised evidence in the same fused model. No downstream component may
recreate these as an independent veto, flip, threshold or substitute direction.

"Full stack" means complete causal coverage with exact liveness and influence,
not an unlimited pile of correlated indicator aliases. A proposed trend or
session field belongs only if it is available at the decision timestamp,
non-degenerate with sufficient TRAIN support, uniquely identified and
demonstrably connected to the learned path. An untouched OOS window may
truthfully occupy one learned regime state. Redundant, stale or future-leaking
variants make the system less robust and are rejected rather than counted as
extra evidence.

`gx1/contracts/entry_full_input_liveness_v1.py` validates every field on train,
validation and test. TRAIN must prove variability/activity or an explicit
sparse-event support floor. VAL/TEST are exact untouched state observations and
may contain one genuine regime state, but non-finite data, unseen categorical
values, forbidden fields, wrong order or identity mismatches fail. ATR shift is
preserved as diagnostic evidence for later OOS edge gates. There is no
constant-pass-through allowlist and no fabricated OOS variation.

## Model and objective ownership

`gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py` owns the Entry
architecture. The active architecture consumes the exact model-native state,
uses temporal and multi-timeframe representations, learned timeframe scaling,
cross-timeframe attention, positional encoding, FiLM conditioning and eight
specialist encoders before producing the public direction surface and its
supporting heads.

`_checkpoint_admission_ok` in the trainer owns checkpoint admission for both
profiles (user vedtak 2026-07-25). Candidate requires auxiliary head health,
active head health and cooperation gate health, unchanged. Smoke admits on
active-head liveness plus non-degenerate class support and keeps auxiliary and
cooperation health as logged diagnostics, so a trainability run produces a
comparable bundle instead of a binary refusal. A smoke bundle has zero edge,
promotion or launch authority; every downstream acceptance contract is
unchanged and only candidate bundles enter the acceptance chain.

`gx1/contracts/entry_model_native_readiness_v1.py` owns the exact specialist
and head declaration. `gx1/contracts/entry_model_native_training_objective_v1.py`
requires every advertised objective to have a positive loss weight. A head
cannot be retained as an unsupervised decoration.

`gx1/contracts/entry_model_native_aux_targets_v3.py` is the sole owner of the
schema-v4 46-column future-target surface and the exact 12-value turning-point
layout. For LONG, adverse-turn timing means the low before the favorable peak
(`BOTTOM`); for SHORT it means the high before the favorable trough (`TOP`).
The surface also includes nine spread-aware, full-counterfactual
LONG/SHORT/FLAT path-utility targets at K12/K48/K96.
`gx1/contracts/entry_model_native_offline_rl_v1.py` owns the schema-v2
action/horizon order, reward scaling, expectile and ranking math. Ambiguous
reward ties are excluded from ranking rather than silently assigned by action
order. There is no logged-behavior/AWR objective, Bellman backup, replay policy
or separate Entry-IQL runtime.

Direction training includes the public three-class objective, MTF direction,
tail and slice behavior, utility margins/triad, trade/side hierarchy and
validity. Supporting objectives include path quality, MFE, tradability,
bad-path, clean-edge, survival, timing, tail/volatility, TF agreement,
trendline rail and position size.

Internal contextual-bandit objectives directly regress `Q(s,a,K)` to all nine
counterfactual rewards, train `V(s,K)` toward the expectile of detached
`max_a Q`, and enforce a small reward-defined action-ranking margin.
`Adv(s,a,K)=Q(s,a,K)-V(s,K)` is derived exactly. Q, V and Advantage are learned
evidence in the final fusion; none is an independent direction selector.

The final direction layer is one exact learned fusion. Its ordered input is 26
evidence groups / 96 values, followed by `LayerNorm(96)`, a learned
`96 -> 128` projection, GELU and a learned `128 -> 3` projection. Immutable
calibration is applied after those raw `LONG/SHORT/FLAT` logits; exact argmax
remains the only public direction selection.

The position-size target is exactly `sigmoid((MFE-MAE)/(2*ATR_bps))`. MFE is
selected-side, spread-aware and signed; path quality is also signed. MAE is a
non-negative adverse magnitude. Validator, scaling, train loss and validation
loss preserve those domains without clipping or parked-zero substitution.
`FLAT` has a neutral training target and zero executable units. Prediction is
exported and journaled, but it can become capital authority only through the
separate immutable learned-size calibration and OOS exposure/drawdown/parity
contract. Missing or rejected sizing proof emits no order; it is never
converted to multiplier `1.0`.

## Runtime evidence ownership

`gx1/contracts/entry_model_native_runtime_evidence_v1.py` owns the complete
serve-time evidence schema. The same validator is called by the model-native
decision adapter, `TradeState` persistence/recovery, `TradeJournal` writes and
reloads, and the daily trade review. It requires exact calibrated direction,
trade/flat and side parity plus all hierarchy, path, utility, calibration,
MTF, eight-specialist, geometry, learned-size and exact Q/V/Advantage evidence.
It also validates `Adv=Q-V`. Missing, unexpected, non-finite, inconsistent or
retired overlay fields fail closed; consumers do not fill or infer them.

`position_size_logit` and `position_size_pred` are mandatory learned evidence.
The sizing diagnostic additionally binds its calibration bytes, XAU instrument
constraints, account scenarios, monotone capacity transform, step-rounded
units and actual-1-unit/equal-total-allocation controls. This label-horizon
proof does not grant capital authority. Entry paper/live remains blocked until
a full-TEST joint sizing-only replay binds every per-M1 HOLD-to-`EXIT_NOW` trace,
learned sizing is adopted, and a fresh post-adoption broker shadow
runtime-parity event passes. Strict finalizers and row-recomputing validators
now exist, but no real current-contract events do. Artifact guard
binds them to the accepted bundle and complete serve gates. A historical
fixed-1x comparison cannot satisfy launch authority or rescue a failed proof.

## Dataset and launch ownership

`gx1/scripts/build_entry_v10_ctx_training_dataset_v3.py` is the canonical
Entry dataset builder. Its filename is historical; its accepted behavior is
model-native only. It must emit exact split manifests and all direction/path/
utility/sizing targets without invoking Entry XGB inference.

The rebuild wrapper, TRAIN-only rank-reference producer and dataset builder
all require the same `--run-id`. The rank NPZ embeds it, the rank sidecar
repeats it, and the state contract/build proof/split manifests bind both
artifact hashes.
Missing or unequal IDs invalidate the chain. The ID is immutable provenance,
not an approval or direction authority.

Training owns a separate output lineage. Public train wrappers accept only the
new training `--run-id`; `entry_model_native_seq513_train_launch_contract_v3`
derives `dataset_run_id` from the exact post-rebuild event and independently
requires `extra.entry_run_id` plus
`model_native_state_contract.entry_run_id` in TRAIN/VAL/TEST to equal it.
Recipe audit v2 records both roles. The launch validator emits the dataset ID,
the wrapper forwards it explicitly, the trainer compares CLI/environment/all
manifests, and exported metadata plus lock bind one exact
`entry_model_native_training_run_lineage_v1`. A missing, operator-supplied,
collapsed or split-brain lineage fails closed before usable bundle authority.

All retained OANDA backfill writers also validate an explicit `--vedtak`
before creating or modifying files. A missing or invalid decision fails before
side effects; backfill output never supplies Entry direction authority.

The feature-ranking JSON and its derived seq513 signal manifest are explicit
immutable inputs, never files selected by glob, mtime or lexical "latest".
Preflight, wrapper and builder revalidate the nested ranking lineage, run ID,
source hash and exact requested TRAIN start/end. A ranking produced for a
different split cannot authorize a build even when its schema and 378+101
shape are otherwise valid.

`gx1/contracts/entry_model_native_train_recipe_v1.py` owns the exact 162-key
decision-affecting trainer environment. It covers every active trainer setting
plus checkpoint monitor; ambient values, pass-throughs and wrapper defaults
cannot supply direction behavior.

`gx1/scripts/materialize_entry_model_native_seq513_train_recipe_audit_v1.py`
constructs one immutable recipe event. It delegates validation to
`gx1/contracts/entry_model_native_train_launch_v1.py`, which binds explicit
train/val/test manifests, data files, source tape, liveness, feature, target,
specialist, real split-native pretrain and readiness audits. The launch
contract also binds the exact bytes of the recipe owner, producer, control
surface, wrapper, trainer and capped runner. The recorded source commit must be
an existing ancestor; current execution is admitted only while every bound
executable source hash still matches. Mutable `latest`, symlinks, missing
hashes, unlisted environment overrides and pre-existing output directories
fail.

Both wrappers pass the exact TRAIN/VAL/TEST manifest and parquet paths to the
trainer, together with recipe-bound hashes. The trainer revalidates all six;
it has no dataset-directory glob, TRAIN-stem VAL/TEST inference or optional
split fallback.

`gx1/contracts/entry_dataset_split_artifacts_v1.py` owns the common foundation
audit boundary. For each TRAIN/VAL/TEST split it requires an explicit canonical
manifest path, manifest SHA-256 and parquet SHA-256; the parquet path may come
only from the hash-bound manifest. Feature, target and specialist audits emit
the normalized four-field identity, and smoke/adoption gates compare it with
the candidate manifest. Extra files in the dataset directory are inert.

The only model-native train wrappers are:

- `scripts/run_entry_model_native_seq513_smoke_train.sh`;
- `scripts/run_entry_model_native_seq513_candidate_train.sh`.

They do not grant authority by themselves. One valid training `--run-id`, one
launch-derived immutable `dataset_run_id` and all bound prerequisite evidence
are required; evidence gates, not either ID, admit execution.

## Bundle, calibration and evidence ownership

`gx1/scripts/audit_entry_foundation_smoke_bundle_v1.py` is the canonical exact
bundle audit despite its historical filename. It strict-loads the state and
proves metadata/lock/state hashes, objective identity, dimensions, 22 active
heads, eight specialists, exact learned fusion, learned-component movement,
MTF/cross-TF/positional/FiLM/timeframe-scale wiring and immutable validation/
test behavior. Direction claims require exact support, confusion counts and
recomputed Wilson lower bounds globally, per class and in declared context
slices on both validation and test.

Foundation target audit schema v2 requires all 46 aux targets in every split,
including `time_to_mfe` and all Q targets. LONG/SHORT Q targets must be live,
FLAT must be exact zero reward, and every horizon must contain non-collapsed
unique LONG/SHORT/FLAT best actions. Prediction evidence schema v2 and smoke
bundle audit schema v4 then bind the exact bundle commit and prove on both
VAL and TEST that all 12 timing
outputs align with their targets, learned near-BOTTOM LONG and near-TOP SHORT
pockets meet immutable precision/support/Wilson floors, Q aligns with reward
targets, Q ranking selects the reward-best action, V tracks max-Q, and
`Advantage=Q-V`. These are audit-only evidence contracts; none is a live rule
or a second direction authority.

Smoke v3 distinguishes prediction/gate/component liveness from causal
direction influence and has no activation authority. Serve-parity v4 is the
later fail-closed influence owner: both exact input-family masking and encoder
hook ablation must move raw and calibrated class margins for all eight
specialists, while immutable VAL-mean replacement of each exact fusion slice
must move both surfaces for all 26 groups on deterministic TEST states. Exact
zero-mask ablations impose the same requirement on continuous/categorical
context and M5/M15/H1/H4/D1. Layout, reference, bundle and lock hashes are
admission-bound; an old event or one passive input/group blocks launch.

`gx1/scripts/fit_entry_direction_calibration_v1.py` may fit only the declared
calibration artifact. Trained, calibrated and sizing-finalized bundles share
the exact inventory/commit owner in
`entry_model_native_bundle_commit_v1.py`; publication occurs only after hidden
staging, strict load, `fsync` and atomic no-replace rename.
`gx1/contracts/immutable_event_authority_v1.py` likewise stages and fsyncs a
complete event before its final name becomes visible. A mutable report path or
a copied `PASS` decision is insufficient.

Selective-edge prediction carries exact VAL/TEST manifest and parquet
identities into its immutable report. Candidate replay and serve parity consume
those report-bound identities directly. Learned sizing provenance does the
same for TRAIN/VAL fitting and TEST diagnostics; it neither scans the dataset
directory nor grants sizing any direction authority.

`gx1/execution/model_native_entry_replay_v1.py` owns neutral source-tape and
offline label-horizon primitives. Replay consumes the model's final direction;
it does not decide direction. Candidate replay records the calibrated logits,
decision, supporting heads, target/outcome alignment and a unit-normalized
price-path outcome. It is an offline direction diagnostic with
`position_size_applied=false`, not an order or capital simulation. Executable
learned sizing must be proven separately against its OOS controls, the exact
adopted active Exit replay and post-adoption runtime parity. The diagnostic
contracts now require replay schema v7: immutable label-horizon bid/ask facts
remain separate from active-Exit decisions and fills, while every action binds
one committed closed bar to the following exact fresh quote. They also require
a file-bound contiguous per-M1 Exit trace,
row-recomputed bid/ask results, exact canonical OOS row identity, exact
registry identity, a recursive path/size/SHA inventory of every regular file
under all three active Exit artifacts, adopted bundle hashes and fresh
broker-shadow parity with zero orders. The compatibility joint finalizer still
validates caller-supplied trace bytes and is explicitly rejected by launch and
runtime. The canonical operation in the same existing owner instead iterates
every TEST row through `V12Pipeline.make_exit_decision`, emits explicit FLAT
no-order evidence, and cross-binds runtime heads, SourceTape, frozen pair,
active Exit artifacts, source inventory and exact published replay/trace
files. No current real chain has passed, so capital authority remains
`BLOCK`; live/paper emits `NO_ORDER` whenever that sizing authority is missing
or red.

`gx1/contracts/entry_model_native_adaptation_drift_v1.py` owns the only market
adaptation trigger. It row-recomputes immutable same-bundle candidate-TEST and
settled broker-shadow probabilities/outcomes, global LONG/SHORT edge and
direction-specific session/volatility slices. It can return only `STABLE` or
terminal `DRIFT`; it never selects direction, trains or submits an order.

`gx1/scripts/verify_entry_replay_readiness_v1.py` schema v2 binds current
bundle bytes and hands off with zero activation authority to
`gx1/contracts/entry_model_native_adaptation_lifecycle_v1.py`. The lifecycle
owns initial admission, monitor refresh, drift block, offline challenger,
zero-order shadow, explicit promotion and prior-incumbent rollback.
`gx1/contracts/entry_model_native_adaptation_shadow_v1.py` is the mandatory
promotion-comparison owner: identical paired paths, both exact argmax outputs,
bid/ask-recomputed outcomes, absolute challenger side edge and positive
lower-95% paired improvement globally and per supported direction/context.
Newest
terminal evidence wins; failed drift, replay or transition refresh invalidates
older green evidence. `gx1_guards/artifacts.py` accepts launch `ALLOW` only
from a fresh activating lifecycle event cross-bound to the exact accepted
bundle, serve gates, joint Exit proof, sizing runtime parity and the newest
immutable one-time launch approval. The approval binds the complete
launch-state payload and exact bundle commit; environment text is not
authority. The runner binds a startup lease and revalidates unchanged
launch/registry bytes and all freshness gates before every new exposure.

`gx1/scripts/finalize_entry_model_native_launch_v1.py` is the one canonical
candidate→launch transaction producer, reached only through
`scripts/entry_next_edge_control.sh model-native-finalize-launch`. It accepts
no caller-selected registry, state or event roots. It requires a pre-existing
identity-bound vedtak, validates the exact accepted bundle/commit and
serve/sizing/lifecycle prerequisites, derives an exact active-Exit projection,
and binds the same single-exposure operating point into registry and state.
`gx1/contracts/entry_model_native_launch_transaction_v1.py` owns the exact
COMMIT/FAIL shape, target hashes, local immutable backups and recovery rules.
A stable cross-process lock plus registry compare-and-swap serializes writers;
both target replacements are post-validated, and any partial error restores
both pre-transaction bytes before publishing terminal FAIL. A newer malformed
or red event blocks older green evidence. This closes the audited source
transaction gap but creates no artifact or launch authority by itself.
`require_canonical_active_exit_replay_launch_authority` blocks caller-supplied
joint Exit rows before vedtak consumption or target mutation, but accepts a
v7 proof only when its nested producer evidence revalidates every exact input,
active Exit byte and output binding. The canonical producer reuses
`V12Pipeline.make_exit_decision` over full TEST with the complete frozen Entry
snapshot and hash-bound M1/canonical/BASE28/MTF state. The existing
`load_active_exit_replay` factory already loads one atomic frozen prebuilt pair,
explicit XGB/V3/Exit-IQL artifacts, pinned environment and exact SourceTape
without loading SmartEntry. The producer is exposed through the existing
`model-native-canonical-active-exit-replay` control route; missing cadence,
state, inference or source exhaustion terminalizes red, never a synthetic HOLD
or horizon pass. Current launch remains `BLOCK` because no compliant fresh
chain has executed it.

## Evidence retention and cleanup ownership

`gx1/contracts/evidence_retention_v1.py` is the byte-identity and authority
contract for destructive evidence cleanup. `gx1/scripts/cleanup_gx1_evidence_v1.py`
is the only admitted `GX1_DATA` deletion route. It accepts exact disjoint leaf
targets only; pins the canonical artifact registry, XAU launch contract and
deletion-incident record; writes a per-entry JSONL byte/topology manifest; and
forbids exclusions, symlinks and mount crossings. Immutable plan and separate
approval events precede explicit execution. Execution atomically stages each
target inside a new same-filesystem wrapper, proves the staged inventory again,
then writes durable staged/terminal evidence. Repository source cleanup remains
a reviewed source change and cannot use this data-deletion route.

`PROJECT_STATE_entry_iql_delete_incident.json` is the compact incident record
for the 2026-07-07 exclusion-path failure. Salvaged metadata is diagnostic only:
it contains no original row/model bytes and has no direction, comparison or
launch authority.

## Serving boundary

`gx1/execution/v12_model_native_state_live.py` builds the exact serve state.
`gx1/execution/v12_smart_entry_live.py` is a historical filename for the
model-native decision adapter; it loads only a launch-admitted bundle and
validates the shared exact runtime evidence contract before returning a result.
`gx1_guards/artifacts.py` requires both the artifact registry and
`PROJECT_STATE_xau_direction_launch.json` to agree on the exact bundle,
metadata hash, sizing/Exit/lifecycle evidence, bundle commit and immutable
launch approval. Current launch state is `BLOCK`, so resolution must fail.

Serving must reproduce the same ordered features, normalization, timeframe
alignment, architecture, calibration and final logits as immutable replay on
identical bars. A parity mismatch blocks launch.

Live direction must never be reconstructed from specialist votes, utility,
confidence, session, trend or a threshold. Those are learned evidence inside
the model, not downstream authorities.

`gx1/execution/v12_pipeline.py` owns immutable Entry freshness. It accepts
exactly 96 rows ending at the latest closed M5 bar. That row becomes available
five minutes after its bar-start; the decision must occur within the next 90
seconds, and canonical cutoff age may not exceed 390 seconds. These limits are
not environment-configurable. A missing/wrong window, older row or exceeded
limit raises structured unavailability and emits no `LONG`, `SHORT` or `FLAT`;
there is no cached-row substitution or backlog execution. Exit retains its
separate admitted freshness semantics.

The same fail-closed rule applies before inference to the 142+5 context and
session identity. Missing, invalid or session-inconsistent values, including a
fabricated ASIA flag, return
`MODEL_NATIVE_ENTRY_CONTEXT_NO_DIRECTION`; the runtime never manufactures
context, invokes a bridge or translates the failure to synthetic `FLAT`.

## Exit boundary

Exit is separate from Entry, but the retained June/July Exit selection is no
longer admissible. Its Exit-IQL summary is explicitly research-only and
non-production, the registry has no unique serving fold, and its checkpoints
lack the newly required ordered-feature binding. Runtime therefore fails
closed before loading it. The old per-bar substrate is also causally invalid:
fill/first-state timing predates the exact T+5 contract, four of five M1 phases
joined a not-yet-closed M5 row, and V3 trade overlay was written one M1 row
early. Rebuild, exact V3 rescore and Exit retraining are mandatory.

The repaired live path derives every required closed-M5 key from the actual
512-M1 window, requires complete unique finite canonical/XGB coverage, stages
the entire M1→V3→Exit transition on a cloned TradeState and commits only after
the complete decision validates. Persisted state binds the exact last M1,
rejects duplicate/gapped cadence and preserves Strategy-F deferral. The five
M1 microfeatures share one log-return/RMS-volatility owner between train and
serve. No partial state, zero-filled model input, implicit fold or
environment-selected V3 artifact is admissible.

Exit IO V8 is 173 fields with a per-M1 historical closed-M5 cadence. Its 78
market-context fields are never copied from the terminal row across the full
window. The shared M1→closed-M5 mapping, 95-row volume prefix, XGB session/
probability bridge and exact SourceTape open quote are single owners. The V3
training reader revalidates the `N x 173` float32 matrix, UTC minute index,
zero base trade-state slots, XGB bridge values, overlays, exact 240-row trade
records and teacher equality; byte bindings alone are insufficient.

The retained V3 XGB bridge is an Exit-only owner of real ordered 7/41-field
validation for two active Exit consumers. Its import and order checks fail
closed and are intentionally not deleted as Entry residue. Conversely, the
retired Entry-IQL record in `PROJECT_STATE_artifacts.json` has `path=null` and
status `RETIRED_ARTIFACT_ABSENT`; it cannot be resolved as a fallback.

## Control and current status

Use:

```bash
bash scripts/gx1_handover.sh
bash scripts/gx1_handover.sh --check
.venv/bin/python -m json.tool PROJECT_STATE_xau_direction_launch.json
scripts/entry_next_edge_control.sh --help
```

`scripts/entry_next_edge_control.sh` is the single Entry control surface. It
must not expose direct legacy trainers, generic upstream pass-throughs or old
activation routes. `scripts/gx1_handover.sh` is the only handover script.

Current facts:

- source has one chain-owned ranker/dataset path, a host-wide exclusive
  capped-job lock, immutable bounded Group-A chunks, one exact checkpoint
  retry and schema-v4 immutable terminal events;
- V24 terminalized GREEN under the pre-V7 dataset gates and binds XAU data
  through `2026-07-22T12:05:00Z`, 369,081/5,904/4,115 split rows and the
  exact 513+142+5 field surface. The post-V7 audit rejects it for reuse because
  all six signed dip-MFE targets are clipped to a non-negative domain;
- V24 post-rebuild, liveness, pretrain, foundation feature, complete target,
  specialist, smoke-readiness and trainability reviews historically passed the same split
  bytes. TRAIN has zero dead signals, exact duplicate groups or unmapped
  fields; the only exact OOS duplicate group is six D1 regime-state fields in
  June VAL and is recorded, not waived;
- V22 proved and failed on two TRAIN duplicate liquidity/SR pairs plus sparse
  event semantics. V23 proved their repair, then exposed the missing explicit
  `iql_distillation=false` preflight side-effect declaration. V24 proves both
  repairs and all six side-effect keys;
- trainability source wiring validates that downstream owners import and use
  `MODEL_NATIVE_CONTRACT_MODE` and `MODEL_NATIVE_SIGNAL_DIM` from the exact
  signal contract. Resolved-literal grep is not contract proof;
- no model training is running. V7 completed six TRAIN/VAL epochs and failed
  hard-red with no checkpoint/bundle. The terminal output path is absent and
  its temporary memmap is cleaned;
- the full audit in `PIPELINE_AUDIT_XAU_20260723.md` proves selected-side
  bad-path LONG bias, signed dip-MFE corruption, replacement-sampler coverage
  loss, auxiliary target/weight/metric mismatches, partial checkpoint
  admission and incomplete MTF/scaler/context/fusion/launch contracts. All
  first-wave faults and records the later Exit/incremental re-audit. Several
  first-wave faults are source-repaired. ATR/ROC/VWAP, dependent normalized
  VWAP, published SMC ATR and H1/H4 decision-time alignment now have shared
  owners. Canonical-v3 and BASE28 publish through one immutable, content-bound,
  atomic generation pointer; the updater recomputes canonical-v2 over the
  complete verified native-M5 history instead of resetting bounded state.
  Native M1/M5 ownership now includes the actual immutable producer in the
  existing historical OANDA owner, not only its validator: fixed three-day M1
  or 15-day M5 exact MBA requests, retained response chunks, complete-only
  typed rederivation, streamed year output, source/Git inventory and atomic
  no-replace publication. Its fixed routes are
  `model-native-native-m1-source` and `model-native-native-m5-source`; neither
  has run on production data. Admission verifies closure, per-year SHA, row
  count and exact Arrow schema. Raw BASE28 owns only the 13 native-M1 market
  fields; phase/volume transforms are causal derivatives. Realized slippage
  is removed from the causal feature contract
  because no pre-decision observable owner exists; replay stress remains an
  explicit evaluation input. XGB and V3 require bundle-owned exact feature
  contracts and one recursive, reproducible cross-role lineage identity. The
  existing V3 dataset owner now loads only exact runtime-head prediction
  evidence, strict chronological SourceTape, one frozen canonical-v3/BASE28
  pair and one exact XGB identity. It derives the 173-field matrix through the
  shared serving builder, creates exact T+5 overlays/records, writes a
  byte-bound PASS event and publishes by atomic no-replace rename. Callers
  cannot supply matrix, overlay or record members. The canonical active-Exit
  full-TEST producer and fresh artifacts remain open; V24/V7 and the old Exit
  artifacts remain rejected;
- rejected V21/V22/V23 large split parquets are deleted; their terminal,
  manifest and audit evidence is retained;
- zero-reachability Entry adapters, critics, duplicate journal schemas,
  detached feature modules, manual sizing modules and stale research launchers
  have been deleted rather than retained as alternatives;
- no practical-precision or trading-edge claim exists without new immutable OOS
  and live-like proof;
- H1/H4 has one causal full-array alignment owner. Leading warmup remains NaN,
  no completed HTF evidence fails closed, and the unreachable stateful branch
  with conflicting shift semantics has been removed;
- Entry candidate, replay, paper/demo/live and promotion remain blocked.

## Pipeline- og ingredienskart (seq513-datakjeden)

Oppdatert 2026-07-23 etter V7-terminalfeil og full-pipeline-audit. Alle 19
strukturelle krav og alle fire polaritetskrav er obligatoriske. V24 beviste
378+101-flaten, separat likviditets-/S/R-semantikk, sparse-event-livlighet og
den komplette seksnøklers preflightkontrakten, men er nå eksplisitt avvist for
rebuild på grunn av signed-target- og training-contract-feil.
Kartet beskriver den herdede, påkrevde artefakt-DAG-en og kolonne-eierskapet.
Les dette FØR du rg-jakter i builderen.

### Artefakt-DAG (produsent → output)

```text
kanonisk M1 bid/ask  GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL/year=*/
  └─(repair ved tape-defekt) gx1/scripts/repair_m5_tape_dec2024_from_m1_v1.py
        --vedtak --m5-tape-root --m1-tape-root --out-root → EVENT/m5_tape_repaired_*/
kanonisk M5 bid/ask  .../xauusd_m5_bid_ask__CANONICAL/year=*/  (OANDA-native; IKKE M1-aggregert:
        high/low == M1-agg eksakt, open/close/volume kan avvike grense-ticks <=0.06)
  └─ gx1.scripts.materialize_build_canonical_features_v2  --m5-root --out-path
        → canonical_features_v2.parquet
      └─ gx1.scripts.materialize_canonical_v3_augment  --input --output-dir
            → cv3/ (113 filkolonner = cv2 − 11 redundante + 6 nye)
          └─ gx1.scripts.materialize_cv3_modelrange_v1 --run-id --cv3
                --canonical-v2 --out (eksakt radakse, bare `atr` fra cv2,
                eksplisitt vindustrim og hashbundet provenance; 109 kol)
              └─ gx1.scripts.add_ctx_cont_columns_to_prebuilt
                    --prebuilt_parquet --output_parquet --raw_m5_parquet <7 år-parter>
                    --tape-root --rank-reference <eksakt TRAIN-only NPZ>
                    (eksakt ctx16 + session5/cat5; ingen alternative dimensjoner)
                    → FULL_PLUS_CTX_v3src.parquet (188 kol; aktiv kontrakt) + manifester
cv3 ─ gx1.scripts.prebuild_multi_tf_cache_v2 --m5-prebuilt --out-dir
        → MULTI_TF_V2_CACHE/ (builder_version må matche HTF_V2_CACHE_BUILDER_VERSION)
reparert tape + cv2 + cv3 + modelrange + cache + FULL_PLUS
  └─ gx1.scripts.audit_seq513_source_cascade_v1 --run-id --event-root --out
        --required-history-start --expected-full-time-min --expected-full-time-max
        → fersk schema-v5 hashbundet SOURCE_CASCADE_PROOF.json; alle self-paths må være
          event-lokale og den ferdige finite flaten må dekke common-history-start
FULL_PLUS + cache ─ scripts/run_seq513_rebuild_chain_v1.sh
      └─ materialize_model_native_train_rank_reference_v2
        --run-id --source-parquet --history-start --fit-start --fit-end --out
        → model_native_train_rank_reference_v4.npz + sidecar
          (retained filename; payload schema model_native_train_rank_reference_v5)
      └─ materialize_entry_model_native_train_feature_ranker_v1
        --run-id --source-parquet --mtf-cache-dir --rank-reference-npz
        --history-start --train-start --train-end --out
        (bounded Group-A chunks + EVENT/_ranker_checkpoint.npz)
        → ENTRY_MODEL_NATIVE_TRAIN_FEATURE_RANKING_<stamp>.json
  └─ materialize_entry_model_native_seq513_signal_manifest_v1 --feature-ranking-json --out --run-id
      └─ entry_next_edge_control.sh model-native-rebuild-preflight
            <eksplisitte ranking-/manifeststier og øvrige flagg>
          └─ rebuild_entry_model_native_seq513_dataset.sh --run-id
                <samme eksplisitte ranking-/manifestidentitet og øvrige flagg>
                (capped 30G)
                → dataset/*__HOLD_03B_{train,val,test}.parquet + DATASET_BUILD_PROOF.json
Kjede-driver: scripts/run_seq513_rebuild_chain_v1.sh --run-id --event-root
--feature-ranking-json --preflight-out-dir. Rankingstien og preflight-mappen
må være nye; kjeden produserer og binder rankingen selv og allokerer den eksakte
timestampede manifeststien først ved manifest-produsentgrensen. Ingen inferred
resume, glob/mtime eller leksikalsk latest.
Kun én eksakt hash-bundet checkpoint-retry er tillatt etter capped feil.
Telegram-ping er kun operasjonell status; validerte split-manifester er
terminal autoritet.
V24-terminal + preflight + liveness + pretrain + seks split-filer/manifester
  └─ entry_next_edge_control.sh model-native-post-rebuild-readiness
        → immutable post-rebuild readiness; source_dataset_dir == smoke_dataset_dir
      └─ foundation feature/target/specialist audits
          └─ model-native-smoke-manifest → smoke-readiness → trainability/recipe
              → model-native-smoke-train --dry-run → capped --execute
```

V24 nådde og bestod den daværende `trainability`-reviewen. V1–V6 feilet uten bundle: først
på emitted aux-target proof, deretter fordi trainer feilaktig sammenlignet
smoke-run-ID med dataset-build-ID, så fordi signert spread-aware MFE feilaktig
ble validert som ikke-negativ, og til slutt fordi MTF-direction-headens
liveness-sjekk krevde et redundant `y_direction` batch-alias i stedet for det
kanoniske `y`-tensoren. V4 bygget full 72.71 GB TRAIN-tensor, subsamplet 10 000
rader og nådde første model-forward; ingen optimizer-step fullførte.
`entry_model_native_seq513_train_recipe_audit_v2` produseres nå av én kanonisk
eier med 162 eksakte treningsverdier, separat training/dataset-lineage og
bytebinding til hele kjørebanen. Commit `c9e2569f` bevarer valgt-side signert
MFE og path quality gjennom aktiv target-validering samt train/val-loss, mens
MAE fortsatt er en ikke-negativ adverse magnitude. Den separate seksfelts
dip-MFE-produsenten ble senere bevist å fortsatt klippe negative verdier.
Commit `f05b3390` binder MTF-headen til
samme kanoniske klasse-target i train og val. V5 fullførte deretter en hel
train/val-epoke med optimizer-steg, men 23 direction-slice-feil og
tradable/bad-path AUC 0.509/0.482 mot fast gulv 0.52 avviste checkpointen.
Ingen bundle ble skrevet. V6 kjørte seks fulle epoker og viste at global balanse
kunne bli god i epoke 4, men lokale slices og auxiliary-health passerte aldri.
Ved epoke 6 var LONG-andelen 0.058943 og clean-edge/path-quality-headene hadde
kollapset til Spearman +0.959 mot bare +0.699 mellom VAL-targetene. Commit
`37128985` samler derfor eksakt epokevid gatebruk for specialist-, TF- og
family×TF-portene og gjør alle tre checkpoint-blokkerende ved uendret gulv
0.01. Den retning-nøytrale balansevekten økes 0.05→0.50; ingen retnings-,
AUC-, slice- eller promotionsterskel senkes. V7-recipe og offentlig dry-run
passerte med 25 000 stratifierte rader og åtte epoker/patience åtte. V7
fullførte seks epoker før hard-red-stop. Accuracy toppet 0.403455 gjennom
85.1118% FLAT; sluttpunktet hadde 71.4092% SHORT, 32 slice-feil, svak
bad-path/survival AUC, seks cross-head-kollapser og sultede specialist-/
family×TF-porter. Ingen bundle ble skrevet.

2026-07-25: den ferske XAU-only-rebuilden ER utført på native tape. V26
kjørte hele kjeden GRØNN (fersk event-lokal native-v3 M5-tape → kaskade →
kjede → alle audits → smoke-stige → immutabelt V8-recipe), og V8 fullførte
seks epoker på det fullt reparerte substratet før hard-red-stoppen:
`TRAIN_FAIL_NO_BEST_STATE` med total FLAT-kollaps (VAL 100 % FLAT fra epoke
3, 58 slice-feil, path-aux på sjansenivå, sultet family×TF-gate). Sytten
aldri-eksekverte post-audit-grenser ble reparert i eksisterende eiere
underveis (DECISION_LOG 2026-07-25), inkludert: modelrange-projeksjonen
ekskluderer de ni ctx-eide `add_session_features`-kolonnene (126→115+atr;
FULL_PLUS er nå 194 kolonner), recipen binder MTF-cache-manifestet og
emitterer cache-katalogen som validert env-rad, trener/normalisering beviser
cachen mot dens egen deklarerte cv3-kilde, signert dip-MFE admitteres,
klippe-cappen holdes ved eksakt skala-eskalering, og train/serve deler én
identisk clamp. Neste smoke krever et nytt immutabelt recipe-vedtak
(V7-æraens hyperparametre var stilt for rå unormaliserte input); ingen
empirisk aksept-terskel endres.
Håndskrevet JSON eller direkte scriptkall er fortsatt ikke tillatt.

### Kolonne-/feature-eierskap (base 34 + ctx 142)

- Base 30/34: kilde-parquet direkte. Base 4 volum (vol_z_20, vol_ratio_5_20,
  vol_pct_96, signed_vol_z_20): `gx1/features/volume_features.py::add_volume_features`.
- ctx 62 kilde-bårne: FULL_PLUS direkte.
- ctx ~80 GROUP_A_PARITY + DIP_STRUCT (dist_to_*, dip_*, struct_*, atr_ratio_*,
  vol_pct_*_1yr): `augment_forward_outcome_v2` via `build_attach_context` +
  `compute_attach_rows` + `finalize_attach_columns` (attach_group_a… er
  komposisjonen). Nullkopiert måling: 4096 komplette rader på 1,99 s
  (~2062 rader/s). Beslutningsrammen og full kausal M5-kontekst er separate:
  hver beslutningsrad må ha eksakt lik timestamp/OHLC i konteksten. Dette gir
  Group-A-warmup=0 ved Jan-5 i stedet for den feilaktige 13 714-raders
  60-D1-resetten. Parallellisér med full-serie-kontekst én gang + disjunkte
  rad-loop over workers. 1-års-persentilene ligger i konteksten →
  chunk-med-overlapp er FEIL tilnærming. De eksakte
  radområdene checkpointes uten overlapp i 4096-raders immutable NPZ-chunks;
  schema v2 binder decision-frame, full M5-kontekst, MTF, felt og run/window.
- ctx 13 ENTRY_SMART_DERIVED (smc_*_pressure, sr_*):
  `gx1/features/entry_smart_context.py::add_entry_smart_context_features`.
- ctx is_ASIA: builder ~linje 1799, `(session_id==0).astype(int8)`.
- ctx_cat 5: ctx-adderen emitterer; ORDERED_CTX_CAT_NAMES_V3.
- 378 obligatoriske specialist-felt: entry_model_native_feature_layers_v1 + de
  ni entry_*-lagmodulene, inkludert 57 foundation cross-family-felt og 11
  eksakte M5 EMA50/200-felt samt alle strukturelle aux-label- og
  pretrain-polaritetsforutsetninger;
  kandidat-
  ekstra fra samme modulers
  *_FEATURE_NAMES-konstanter; alt beregnes av builderens
  `_build_inline_seq_structure_extension` (~linje 674) som krever
  base+ctx142+cat5 i rammen.

### Builder-linjekart (build_entry_v10_ctx_training_dataset_v3.py)

~594 ctx-gate (CTX6CAT6, v1) · ~628 manifest-kontrakt · ~674 inline-extension ·
~1799 is_ASIA · ~1835 ctx-navn V3 · ~1982 df_ctx_cont · ~2096 merged3-assembly ·
~2115 cv2-lasteliste · ~2244 GROUP_A-attach (krever env
GX1_V10_MULTI_TF_V2_CACHE_DIR) · ~2330 smart-context · ~2337 ctx-komplett-sjekk ·
~3440 args.

### Kjente feller (verifisert 2026-07-18/19)

- July-19-forsøkene gjenbrukte en feature-ranking med TRAIN-start 2020-11-13
  mot aktiv TRAIN-start 2021-03-16. Den gamle preflighten sjekket ikke det
  nestede vinduet og ga derfor falsk GREEN. Forsøkene er terminert/ugyldige;
  schema-v2 `CHAIN_STATUS.json` er nå terminal `RED` med årsak
  `FEATURE_RANKING_TRAIN_WINDOW_MISMATCH` og bundne hasher.
- Des-2024-tapedefekten finnes FORTSATT i kanonisk M5-rot og live-prebuilt:
  3 430 rader har umulig OHLC-geometri og 2 799 av dem er helgerader. Ren M1
  støtter 5 757 rebuildede desember-bøtter; 3 459 kanoniske M5-rader mangler
  M1-backing. Kun event-kopien er reparert. Ingen canonical/live-data ble
  endret; reparasjon og paritetsbevis er en egen åpen beslutning.
- Skall-cwd kan resettes mellom kall: alltid `cd /home/andre2/src/GX1_ENGINE &&`
  først (rg gir ellers stille tomme treff); capped_run arver cwd og
  `python -m gx1...` krever repo-cwd.
- V20 feilet før split-publisering fordi
  `chart.geometry_channel_position_low_to_high` ble brukt av strukturell
  aux-label-bygging uten å være obligatorisk. V21 passerte dette, men feilet
  senere fordi `chart.geometry_support_minus_resistance_stack` manglet i
  pretrain-polaritetsflaten. Den beviste V24-vinduskontrakten
  har rå model-range-start 2020-11-13 og ferdig
  common-history-start 2021-01-05. V15 beviste framtidsstempel-fellen og V16
  beviste at model-range-start 2021-01-04 gir for sen finite start 2021-01-14;
  begge er terminal RED. Kjeden avviser nå begge feil før ranking. TRAIN er
  2021-03-16..2026-05-31, VAL 2026-06-01..2026-06-30 og TEST
  2026-07-01 til snapshotets eksplisitte siste lukkede M5-bar. Alle syv
  grenser er obligatoriske kjede-input; ingen sluttdato har fallback/default.
  Rankingens TRAIN-vindu må matche eksakt.

### Post-V7 source-repair routing checkpoint (2026-07-23)

The target/trainer path now preserves signed dip-MFE, applies selected-side
bad-path symmetrically, samples without replacement, binds exact recipe and
all M5/MTF component bytes, normalizes raw-bps forward heads to a 20-bps model
unit and gates checkpoints on conditional path skill plus all 22 active heads'
target/output liveness and final-fusion influence.

One immutable normalization contract fits all 513+142 continuous surfaces and
the five 25-field timeframe surfaces on the complete physical TRAIN population
before sampling. Every statistic, categorical domain, selected causal source
row and alias is hash-bound in bundle state. All 142+5 context fields have one
family owner and enter family-specific projections before specialist
cross-attention. Current-bar aliases are derived from the actual ordered
signal names, must be bit-identical and reuse context-owned statistics; `82`
is V24 fixture evidence, not a hard-coded architecture constant.

Bundle/event publication, recursive active-Exit byte identity, structured
launch approval, per-entry runtime lease recheck and missing-trade-ID close
behavior are source-repaired. The transactional candidate→promotion→launch
producer, identity-bound vedtak and exact target transaction/recovery are
source-repaired too. A hard launch/runtime barrier gives caller-supplied Exit
replay diagnostics zero authority. The later Exit/incremental audit led to
source closure for atomic canonical-v3/BASE28 generations, complete-history
canonical-v2 recomputation, strict shared native-M1/M5
closure/schema/hash ownership and its immutable OANDA source-bundle producer,
removal of non-observable slippage features, reproducible XGB-bound V3 lineage,
a hash-bound historical closed-M1 provider for Exit and the exact V3
training-dataset writer/event in the existing owner. Formula and HTF
decision-alignment ownership is also repaired. BASE28 ownership is now exact:
its 13 physical market fields come only from native M1 in canonical source
order, while timestamp phase and volume transforms are derived causally at the
training/serve boundary. The complete snapshot-driven
native→canonical-v3/raw-BASE28 producer now exists in that same pair owner and
is routed as `model-native-canonical-pair`; it accepts no old-pair copy and
binds native/source/code/formula/timing/schema/coverage lineage. Native M1/M5
production and the first pair generation (`077e5419…`, 2019-roots, 2026-07-24)
are executed; the exit-chain TRAIN-rank identity binding is source-complete
(loader attach + bucket derivation, V3 dataset and joint-replay routes take
mandatory `--train-rank-reference-npz`/`--train-rank-reference-sha256`, live
requires an ACTIVE `train_rank_reference` registry entry, trainer copies the
block into bundle lineage). Still open are producing the immutable TRAIN-only
rank reference and executing the bound routes, a fresh V3 dataset on accepted
Entry prediction evidence, execution of the canonical active-Exit full-TEST
loop/event, fresh XGB/V3/Exit artifacts (no XGB trainer exists in the
repository), empirical dataset/model/edge proof, and the Entry-cascade
decision to rewire its source from the December-2024-defective old canonical
roots to the fresh native roots.

Update this map whenever ownership or the active call graph changes. Remove
obsolete facts instead of appending a second historical architecture.
