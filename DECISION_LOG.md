# GX1 current decisions

Historical implementation narratives were removed because they repeatedly
acted as stale authority. Git history retains them. This file records only
decisions that constrain the current model-native Entry work.

## 2026-07-21 — one durable rebuild owner and one heavy job at a time

V3 did not fail with a model/data exception: its log stayed inside the silent
Group-A row loop after canonical join. A separately launched V4 ranker then
overlapped that still-running builder before the host environment stopped.
Neither attempt wrote admissible terminal dataset evidence.

The rebuild chain now owns the fresh TRAIN-rank reference and ranking as well
as manifest, preflight and dataset construction. Every heavy stage enters through the
capped runner's host-wide nonblocking lock, so overlapping ranker/builder/
trainer/replay jobs fail immediately. Group-A keeps one full causal context,
fans only disjoint 4096-row ranges, writes immutable per-range checkpoints and
binds completion to frame, MTF arrays, field order and run/window identity.
Worker allocations are range-sized instead of full-frame-sized. The ranker and
builder may make one exact checkpoint retry; changed, partial, unexpected or
unbound checkpoint bytes fail closed. Normal and trapped chain exits publish
immutable schema-v4 terminal events with boot and process identity.

Focused and causal integration tests prove this source contract. V11 exercised
the durable owner/checkpoint path and wrote terminal RED evidence, then exposed
the separate full-history reset fixed by checkpoint schema v2. V1-V11 remain
rejected, and launch remains `BLOCK`.
The unreachable pre-rebuild abstention verifier, its control route, contract
and sole-purpose tests are deleted because the required historical bytes do
not exist; future selection evidence starts from fresh candidate TEST rows.

## 2026-07-20 — adaptation is an immutable offline lifecycle, never live learning

The model-native adaptation source boundary is now implemented. Drift schema
v1 independently recomputes same-bundle candidate-TEST versus settled
broker-shadow rows, exact LONG/SHORT/FLAT probabilities and distribution,
absolute LONG/SHORT edge, and direction-specific session/volatility slices.
Every source and bundle byte is hash-bound; missing support, stale rows,
submitted orders, weak lower-95% PnL or changed bytes returns terminal `DRIFT`.

Replay-readiness schema v2 has zero activation authority and hands the exact
bundle bytes to lifecycle v1. A crashed refresh publishes a newer terminal red
event. Lifecycle v1 enforces the only transition graph: initial admission or
stable monitoring; drift block; offline challenger replay; zero-order shadow;
explicit promotion; and rollback only to a prior incumbent. Live gradients,
post-model direction rules and replay-to-launch pass-through are forbidden.
The launch guard now requires a fresh activating lifecycle event cross-bound to
the same serve, joint Exit, learned-sizing and runtime-parity evidence.

Shadow schema v1 is the promotion comparison owner. It replays incumbent and
challenger on identical immutable paths, recomputes both outcomes from the same
bid/ask prices, requires exact model argmax and zero orders, and admits only a
challenger with absolute LONG/SHORT edge plus positive lower-95% paired
improvement globally and across supported direction/session/volatility slices.

This is source governance, not empirical edge. No real drift rows, challenger,
paired shadow, admission, promotion or rollback event was produced, and no heavy run
was authorized. The current launch decision remains `BLOCK`.

## 2026-07-20 — learned sizing has a complete source admission path

Capital sizing is no longer permanently source-blocked. Label-horizon OOS
proof remains diagnostic. Adoption schema v3 additionally requires an
immutable full-TEST joint Entry plus registry-selected active Exit proof. Every
non-FLAT row must bind a complete per-M1 HOLD-to-`EXIT_NOW` trace; horizon caps,
failed or discontinuous traces, incomplete LONG/SHORT support or registry drift
fail closed. All exposure, drawdown and paired utility controls are recomputed
from the resulting bid/ask exits.

After adoption, runtime-parity v1 requires at least 32 fresh broker-live shadow
observations with LONG/SHORT/FLAT support, multiple transaction IDs, exact
size-transform equality, frozen bundle/model/adoption hashes, unchanged model
direction and zero submitted orders. Artifact resolution now invokes the
previously orphaned serve-gate validator and admits `ALLOW` only when serve
parity, direction pockets, joint Exit sizing, adoption and fresh runtime parity
all bind the same bundle. No real current evidence was produced; launch stays
`BLOCK` and fixed 1x remains non-executable.

## 2026-07-20 — causal influence is explicit at launch, not inferred from smoke

Smoke bundle audit v3 proves prediction, specialist-gate and strict-component
liveness only and has zero activation authority. The misleading duplicate
`zero_init_pass_through_absent` assertion was deleted; loading a component is
not proof that it affects direction.

Serve-parity schema v4 is the launch boundary for causal model influence. On
the same 16 deterministic TEST states, both exact specialist-family input
masking and specialist-encoder hook ablation must move class-centred raw and
final calibrated logits for all eight specialists. Separately replacing each
of the exact 26 learned-fusion slices with its hash-bound candidate-VAL mean
must move both raw and final class margins above epsilon on at least 8/16 rows.
Zero-masking continuous context, categorical context and each of
M5/M15/H1/H4/D1 must satisfy the same raw-and-final movement contract.
The exact group layout, reference, bundle metadata and transformer lock remain
hash-bound. Any missing group, raw metric, insufficient movement or pre-v4
event blocks launch; smoke liveness can never substitute for this proof.

## 2026-07-20 — top/bottom and Q/V evidence must be target-aligned

Non-constant head output is not sufficient learned evidence. Target foundation
audit v2 requires the complete canonical schema-v4 46-target surface in every
split. The timing layout explicitly maps LONG adverse-turn timing to `BOTTOM`
and SHORT adverse-turn timing to `TOP`; `time_to_mfe` is mandatory rather than
an optional column. LONG/SHORT counterfactual Q targets must be live, FLAT Q
targets must be exact zero reward, and all three unique reward-best actions
must have non-collapsed support at every horizon.

Immutable prediction evidence v2 and smoke bundle audit v3 require VAL and
TEST timing-to-target Spearman/MAE, precise supported model-claimed near-turn
LONG/BOTTOM and SHORT/TOP pockets, Q-to-reward alignment and action-ranking,
V/max-Q alignment and exact `Advantage=Q-V`. Thresholds are hash-bound in
foundation audit policy v3 and cannot be changed by CLI or environment. Reward
ties are excluded from Q ranking instead of inheriting LONG from array order.
These metrics audit the one final model path; they do not threshold or replace
live direction. This source hardening authorizes no rebuild or training.

## 2026-07-20 — one model-native offline-RL fusion, not a second IQL policy

The deleted legacy Entry-IQL runtime is not restored as an Entry authority.
Its useful offline-RL math is selectively ported as internal expectile-V and
counterfactual LONG/SHORT/FLAT action-value heads on the same seq513 encoder.
Those positively trained outputs become evidence inside one learned final
fusion; only the final calibrated three-class logits and argmax may select
direction. Distillation requires an independently proven immutable teacher.
The required adaptation mode is the immutable offline
challenger/replay/shadow/promotion/rollback lifecycle described above; live
online gradients and post-model rules are forbidden. Implemented source
contracts grant no authority without fresh byte-bound evidence.

The exact source contract uses action order `LONG, SHORT, FLAT`, horizons
K12/K48/K96 and FLAT reward zero. LONG/SHORT rewards are executable bid/ask
final PnL plus the canonical path utility (`0.35*MFE - 1.15*MAE +
0.25*(MFE-MAE)`). Q is direct full-counterfactual regression, V is a 0.8
expectile of detached max-action Q, and ranking follows the reward argmax.
There is no behavior/AWR loss because no legitimate logged Entry action exists.
Q/V/Advantage add 21 values to the final learned fusion, making its exact
surface 26 groups / 96 values; no Q argmax has public authority.

The exact deleted historical row/model benchmark is not present in Git or the
salvage inventory and cannot be reconstructed from JSON/MD metadata. It is not
silently substituted. Future admission uses a fresh immutable proxy comparison
and absolute untouched OOT/cost/live-like gates, failing closed if they do not
prove edge.

## 2026-07-20 — destructive evidence cleanup has one fail-closed owner

The 2026-07-07 executor deleted 58,512,455,902 bytes, including protected
Entry-IQL data, because its dry-run used unresolved literal `...` exclusion
paths. `PROJECT_STATE_entry_iql_delete_incident.json` binds the surviving
dry-run, execution and salvage identities. Exclusion-based parent deletion is
permanently forbidden.

All future destructive `GX1_DATA` cleanup must use the pinned exact-target
contract and sole CLI owner. It requires a reconstructible per-entry
byte/topology inventory, protection of every active/retired/history/launch and
incident-evidence path, immutable plan, separate immutable approval, explicit
execute, same-device atomic quarantine, staged revalidation and terminal
evidence. Source cleanup remains continuous, but is performed as reviewed
source edits rather than through a data-deletion script.

## 2026-07-19 — one exact split identity survives every Entry evidence stage

Foundation feature, target and specialist audits must resolve TRAIN/VAL/TEST
parquets only through explicit canonical manifests plus declared manifest and
parquet SHA-256 values. They publish the normalized identities; smoke and
adoption gates require byte-for-byte equality with the candidate declaration.
Selective-edge prediction, replay, serve parity and learned sizing consume the
same immutable report bindings. Directory uniqueness, glob selection,
TRAIN-stem derivation and unbound decoy files have no authority.

Repository cleanup is a permanent token/credit discipline. Proven-unused code
and sole-purpose baggage are deleted when encountered after one bounded owner/
caller/process/evidence check. Uncertain ownership is recorded precisely; it
does not justify repeated repository-wide scans or renamed dead copies.

## 2026-07-19 — exact split and serving identities precede execution

Smoke/candidate wrappers and the trainer must receive explicit TRAIN/VAL/TEST
manifest and parquet paths. The validated recipe binds all six hashes; the
trainer rechecks regular canonical paths, hash bytes, manifest self-path,
vedtak lineage, distinctness and common signal/state contracts. Split globbing
and TRAIN-stem inference are retired.

Live serving must load parity and direction-audit events from the exact
launch-declared path/SHA before checking newest immutable-event authority. A
failed model decision is `MODEL_DECISION_UNAVAILABLE`, not `FLAT` or `SKIP`.
Downstream execution may refuse an order for safety/sizing facts, but it cannot
threshold, veto, flip or replace model direction.

## 2026-07-19 — invalidated rebuild authority cannot be recycled

`XAU_SEQ513_REBUILD_20260718_V1` is historical RED evidence only. Its window
values remain useful as the intended next-build window, but its ID cannot
authorize a new ranking, rebuild, smoke or candidate run. A new explicit
vedtak is required after the abstention-baseline decision; no name is inferred.

## 2026-07-19 — Entry route is bridge-free and context fails closed

Active Entry has zero imports from `signal_bridge_v1` or `signal_bridge_v3`.
`entry_model_native_signal_v1` is the single exact owner of the 34 base,
142 continuous-context and 5 categorical-context fields. Missing, invalid or
session-inconsistent context — including a fabricated ASIA flag — yields
`MODEL_NATIVE_ENTRY_CONTEXT_NO_DIRECTION`; it is not converted to `FLAT` or a
legacy bridge decision.

The retained V3 XGB bridge is Exit-only because it still owns real 7/41 field
validation used by two active Exit consumers. Its import and field order fail
closed. Entry cleanup does not authorize weakening or deleting that Exit
contract. Every retained OANDA backfill writer now validates an explicit
`--vedtak` before any side effect. The retired Entry-IQL artifact registry
entry has `path=null` and status `RETIRED_ARTIFACT_ABSENT`; it has no fallback
or compatibility authority.

## 2026-07-19 — mismatched ranking invalidates the rebuild attempts

The seq513 attempts under vedtak `XAU_SEQ513_REBUILD_20260718_V1` were
terminated and invalidated. Their reused feature-ranking JSON covered TRAIN
`2020-11-13..2026-03-31`, while the active dataset request covered TRAIN
`2021-03-16..2026-03-31`. The then-current preflight checked the outer build
window but omitted the nested ranking-window comparison and therefore emitted
a false GREEN. No resulting dataset, signal manifest, bundle or edge evidence
is accepted; partial files are non-authoritative. No rebuild process is
running. Event-local schema-v2 `CHAIN_STATUS.json` is terminal `RED` with
reason `FEATURE_RANKING_TRAIN_WINDOW_MISMATCH` and exact vedtak/git/ranking/
manifest/preflight hashes.

Preflight, wrapper and builder must receive explicit feature-ranking and signal
manifest paths and validate ranking lineage, vedtak, source hash and exact
TRAIN start/end. Directory glob, mtime and lexical-latest selection are
forbidden. Launch remains `BLOCK`; no practical-precision claim exists.

## 2026-07-19 — rebuild window contract and one-owner version constants

Under vedtak `XAU_SEQ513_REBUILD_20260718_V1`: split windows are source-exact
(history 2021-01-05 = source first row; train 2021-03-16 leaves 277 clean rows
for the 96-bar sequence window after the 13,439-row GROUP_A warmup; test end
2026-06-14T23:55 = source last bar). Contract version strings consumed at two
boundaries must have ONE code owner (the loader's constant) — the preflight's
pinned MTF-cache literal made the two contracts mutually unsatisfiable after
the 07-17 re-versioning. The Dec-2024 tape defect is repaired ONLY in the
event copy; canonical root and live prebuilt repairs are a separate open
decision.

This historical clean-row estimate is explicitly superseded by the later V11
decision below; it must not be used as a rebuild premise.

## 2026-07-17 — Entry success criterion is abstention quality, not direction precision

User vedtak (explicit choice over "restore promoted chain" and "continue
unchanged"): the model-native seq513 architecture stays, but the primary
empirical admission criterion is reframed. The June 2026 falsification
campaign proved directional accuracy has a hard information ceiling (~0.62,
12+ tracks refuted strict-OOT); the proven Entry-side edge was SELECTION (the
historical Entry-IQL raw_adv gate). Therefore:

1. The learned `FLAT` class / abstention surface must match or beat the
   historical Entry-IQL gate's OOT selection quality (EV-per-take at
   comparable coverage) before any candidate can advance. This is the primary
   Entry edge gate; direction precision beyond the proven ceiling is not a
   success metric and "near-perfect direction" is not claimable.
2. Before any large rebuild/training spend, a cheap decisive diagnostic runs
   on existing immutable evidence: can a learned abstention head reproduce or
   beat the historical gate's take/skip separation OOT? A failed diagnostic
   stops the lane for re-evaluation instead of burning a full training run.
3. Flat-starvation (zero FLAT predictions — the failure mode of every
   July 8-16 smoke) is the central training problem to solve, not a slice
   detail. Entry-IQL as a separate post-model stage remains retired; its
   selection ROLE lives inside the model.

## 2026-07-17 — audit wave: zero-reachability deletions and contract hardening

User-approved (git history is the backup). Deleted verified-dead surfaces:
`gx1/execution/telemetry.py`, `gx1/execution/v12_live_features.py`,
`gx1/features/feature_manifest.py`, `gx1/utils/pnl.py`,
`gx1/contracts/signal_bridge_v2.py` (+ its sole dead consumer
`assert_canonical_v2_schema`), `gx1/runtime/column_collision_guard.py` (+ its
sole-purpose test), the dead `PrebuiltStateLoader.get_latest_row` method, the
dead `GX1_SIGNAL_BRIDGE_VERSION` env write, and all empty migration-residue
directories (runtime paths `data/`, `runs/`, `gx1/live/logs` preserved). The
absence guard test now pins these as ModuleNotFound.

Hardened in the same wave: the mandatory causal-layer prefix ORDER is
now validated at every manifest consumer (membership alone no longer passes);
the five previously unvalidated `required_*` partition constants in the launch
JSON are enforced against code constants on the ALLOW path; the 90-second Entry
latency limit has one numeric owner (the runtime evidence contract).

## 2026-07-17 — historical Exit truth_e2e evidence producers are retired from test scope

The immutable April-2026 `truth_e2e_sanity` input locks were deleted from
GX1_DATA, so `materialize_exit_hold_exit_now_mdp_reward_contract_v1.write_artifacts`
and `materialize_exit_off_policy_eval_harness_v1.write_artifacts` can never run
again as-is, and per the handover the historical producer chain must not be
restored. Their two `write_artifacts` tests are removed; the modules REMAIN as
live library owners for the retained Exit-IQL chain
(`FORBIDDEN_STATE_FIELDS_V1`, `evaluate_policy`, `_exit_index_realized_exit`).
Fresh Exit evidence requires the new exact builder
(`BLOCKED_PENDING_NEW_EXACT_BUILDER`).

## 2026-07-17 — full-stack families cannot be ranked away

The 479-field specialist surface is now exactly 316 code-owned mandatory
outputs from eleven registered causal feature layers followed by 163 fields from
deterministic TRAIN-only ranking. The emitted exact order and both components
are hash-bound. This retains genuine trend, session, liquidity, structure,
volatility, momentum, price-action, support/resistance and MTF evidence while
still rejecting redundant, dead, unrouteable or future-leaking aliases.

## 2026-07-17 — rebuild authority is artifact-bound

An Entry rebuild decision must survive beyond the shell wrapper. One validated
`--vedtak` is now required by both writing Python producers and is bound into
the rank NPZ, its sidecar, the dataset build proof, the model-native state
contract and every split manifest. Missing, placeholder or unequal IDs fail
closed. This source change did not by itself authorize or start a rebuild.

## 2026-07-17 — learned sizing is required execution evidence

The learned sizing head requires immutable calibration, XAU instrument/account
capacity inputs and TEST utility/exposure/drawdown diagnostics. Label-horizon
results alone do not grant capital authority. Paper/live remains blocked until
an exact joint adopted-active-Exit sizing replay and fresh post-adoption broker
runtime parity both pass. Historical fixed 1x is a comparison baseline only
and can neither satisfy admission nor act as fallback.

## 2026-07-16 — Entry launch is blocked

No existing bundle is accepted for current XAU direction. Fresh seq513 data,
bundle, calibration, immutable prediction/replay and train==serve evidence are
required; older evidence has no compatibility or launch authority.

## 2026-07-16 — one model-native direction path

Final calibrated `LONG/SHORT/FLAT` logits and their argmax are the sole Entry
direction authority. XGB anchors, neutral bridges, Entry-IQL, hand-written
trend/session/confidence/utility filters and compatibility fallbacks are not
permitted.

## 2026-07-16 — preserve the full evidence stack

Retiring a filter does not retire its information. Genuine multi-timeframe
trend, structure, liquidity, volatility, momentum, session/regime, chart,
candle, path and utility evidence must remain as live model inputs, targets or
supervised heads.

## 2026-07-16 — sizing is learned but not automatically capital authority

The position-size head is mandatory, trained, parity-checked and journaled.
Its calibration and label-horizon OOS controls are diagnostic. It receives
capital authority only after a separate exact joint adopted-Exit sizing replay
and fresh post-adoption broker runtime parity pass and are explicitly admitted;
otherwise Entry emits no order.

## 2026-07-16 — exact evidence or fail closed

Every authority boundary uses explicit immutable paths, hashes, exact schemas
and newest-terminal-event precedence. Missing, stale, mutable, malformed or
mismatched evidence blocks the path. Unit tests prove source contracts only;
they do not prove trading edge.

## 2026-07-16 — continuous source cleanup

Disconnected scripts, archived code, stale configs, sole-purpose tests and
obsolete Markdown are deleted after active-call/process/evidence checks. Active
Exit behavior and persistent data processes remain outside Entry cleanup scope.

## 2026-07-16 — one exact runtime evidence contract

The model-native decision, `TradeState`, trade journal and daily review must
validate the same complete immutable snapshot. Direction/logit/probability
parity, hierarchy, path, utility, calibration, MTF, all eight specialists and
the learned size head are mandatory. No consumer may fill missing fields or
accept retired Entry overlay evidence.

## 2026-07-16 — Entry freshness is immutable

Entry consumes exactly 96 rows ending at the latest closed M5 bar. The row has
a fixed five-minute availability lag, then a 90-second decision limit; the
canonical-cutoff age cap is therefore 390 seconds. The limits have no Entry
runtime override. Missing, wrong or late state yields no model direction, not
synthetic `FLAT`, an older cached row or backlog execution. Exit freshness is a
separate contract.

## 2026-07-16 — source completion does not prove edge

Legacy Entry branches and zero-reachability adapters, critics, duplicate
journal schemas, detached feature modules and manual sizing implementations
are physically removed. Launch remains `BLOCK` across later invalid rebuild
attempts, and no practical-precision claim is allowed before new immutable
OOS, live-like, cost and train==serve evidence passes.

## 2026-07-20 — Entry run identity is lineage, not manual approval

User selected `XAU_SEQ513_REBUILD_20260720_V2` as the next immutable Entry run
lineage and removed the separate manual rebuild/training approval requirement.
The active Entry CLI is `--run-id`; `entry_run_id` is hash-bound through rank,
manifest, state, split, train, calibration, sizing and adaptation artifacts.
It grants no authority. Exact evidence contracts alone decide whether a stage
may execute or must fail closed. Live/paper launch, promotion and destructive
GX1_DATA cleanup retain their separate safety boundaries.

The deleted historical Entry-IQL row/model benchmark is not a satisfiable
pre-rebuild gate. A fresh seq513 rebuild and candidate are required to produce
learned OOT rows; selection/abstention admission then uses a fresh immutable
proxy comparison plus absolute OOT support/confidence, cost and live-like
gates. Historical metadata can never substitute for those bytes.

## 2026-07-21 — one feature state before ranking, dataset and serve

The immutable TRAIN-rank reference is now an upstream feature-computation
input. The chain creates it before ranking; ranking, manifest, preflight,
dataset and serve bind the same NPZ bytes, sidecar bytes, source hash,
common-history start and TRAIN window. The dataset wrapper accepts only an
already-bound reference.

The model-native specialist partition remains 479 fields but is now 316
mandatory outputs from eleven causal families plus 163 deterministic
TRAIN-ranked fields. The new mandatory family is the exact 11-field M5
EMA50/200 state/cross/slope/location layer, routed to the learned trend
specialist. Price-derived features use only common-history `close` and
recomputed `atr`; schema fallback to `mid` or older ATR aliases is forbidden.
Signed BOS and sweep pressures are directionally symmetric. The partial live
MTF splice is retired because exact M5/H4/D1 parity was unproven; any positive
context age now yields no direction until one full five-timeframe refresh is
complete. None of these source repairs proves trading edge, so launch remains
`BLOCK` pending fresh rebuild and empirical gates.

## 2026-07-21 — full causal history is a bound Group-A input

V11 terminally disproved the previous common-history assumption. Although its
source cascade, TRAIN ranking, 513 manifest and preflight passed, every one of
the 60 Group-A/dip/structure outputs stayed unavailable for 13,714 rows because
the 60-D1 liquidity window was rebuilt from the Jan-5 decision slice. Only one
clean row remained before TRAIN; seq_len=96 requires 95.

Decision rows and long-memory context are now separate explicit inputs. The
full M5 prefix must end causally, contain every decision timestamp with exact
high/low/close, and is SHA-bound into checkpoint schema v2 together with the
five-TF arrays and ordered fields. Dataset and live preparation use this same
owner path; live HTF/REGIME_V4 computes on full cv3 before slicing. No V11
partial may be resumed after the code change. V12 later proved this repair but
was aborted for a stale cutoff; V13 is required, and launch remains `BLOCK`
until downstream empirical gates pass.

## 2026-07-21 — stale V12 cutoff rejected; V13 snapshots current collector

V12 proved the full causal history repair and built all three seq513 splits,
but its inherited source/test end remained 2026-06-14. Once complete OANDA M1
through 2026-07-21 was available, V12 was interrupted during liveness and
terminalized `ABORTED`; no liveness artifact had been emitted. Its files are
diagnostic only and cannot be resumed into V13.

Read-only comparison found 47,086 collector/canonical overlap timestamps with
zero difference across mid/bid/ask/volume, no conflicting duplicates,
nonfinite values or invalid geometry. V13 therefore snapshots exact collector
bytes event-locally, aggregates only provably complete M5 bars and requires
bit-exact overlap with the repaired M5 tape before atomic publication. The
model-range end and all seven split boundaries are explicit required inputs.
The active rolling split is TRAIN through 2026-05-31, June VAL and July TEST
through the snapshot's last closed M5 bar.

The liveness reader's Python-object conversion was also removed. Direct Arrow
child-buffer extraction validates all offsets and was bit-exact on a real
512-row `96 x 513` batch (max difference zero), reducing that conversion from
6.49 seconds to 0.00073 seconds. The exhaustive finiteness, shape,
seq/snap-parity and 660-field statistics remain unchanged.
