# GX1 current decisions

Historical implementation narratives were removed because they repeatedly
acted as stale authority. Git history retains them. This file records only
decisions that constrain the current model-native Entry work. Later entries
supersede earlier event-specific state; historical headings below describe the
boundary at which each decision was made, not current artifact authority.

## 2026-07-23 — one-epoch V5 is empirical red; V6 preserves the gates

V5 crossed every source and runtime wall, built the complete TRAIN tensor,
trained the exact 10,000-row subsample for one epoch with optimizer steps and
completed full validation. LONG/SHORT/FLAT prediction rates were
0.575542/0.226287/0.198171, so the class-balance guard passed. The direction
evidence did not: validation accuracy was 0.324187, direction-slice score was
-0.914416 with 23 failures, tradable AUC was 0.509 and bad-path AUC was 0.482
against the fixed auxiliary-health floor 0.52.

Checkpoint admission requires auxiliary health. V5 therefore emitted
`TRAIN_FAIL_NO_BEST_STATE`, wrote no checkpoint and created no bundle. The
result is retained as model-quality evidence; no threshold, loss weight,
feature, source binding or fail-closed condition is relaxed.

The recipe contract permits an explicit multi-epoch smoke, while the existing
direction hard-red policy is configured to begin at epoch six. V6 therefore
keeps exact V24 data, all 162 environment settings, thresholds, loss weights,
10,000-row cap and 30G/2G caps, changing only the declared horizon to eight
epochs and patience six. It records repository commit `87b0cec2`; its
executable source bindings remain the exact `f05b3390` repair bytes. Recipe SHA-256
`470b6abb287a9ebb23d2b897555217466b3cbabc1c2593271d41bb82493b1d1b`
is PASS for `XAU_SEQ513_SMOKE_20260723_V6`; the public dry-run passes. No V6
execution or bundle exists at this boundary.

## 2026-07-23 — active head checks bind canonical batch targets

V4 crossed the signed-target wall, built the complete 369,081-row
`96 x 513` TRAIN tensor (72.71 GB disk-backed), stratified the exact 10,000-row
smoke subset, loaded VAL and all five timeframe tables, constructed every
mandatory specialist/head and entered its first model forward. It then failed
before loss completion or optimizer step with
`ENTRY_MODEL_NATIVE_ACTIVE_HEAD_TARGET_MISSING`: the MTF direction-head check
required `y_direction`, although the Dataset deliberately converts that
immutable parquet field once to the canonical class-index batch tensor `y`.

Commit `f05b3390` makes both train and validation MTF checks require `y`.
Adding a second alias would have weakened the no-alias/no-default contract and
is not allowed. All other active head checks were reviewed against emitted
batch keys; no other mismatch was found.

Fresh smoke-readiness SHA-256
`fa44e809e28599b9c9d4fa897fafaccd15cdb80f3bdee9948665cd1c1b283650`
and trainability SHA-256
`6908796b6e289c708d0d1b1bd942c10ef9482391fd3cfb3aba2168cfbc88e312`
are READY. V5 recipe SHA-256
`9e9ae299332b29360c7434e0d237aadfe55e817e1c447e4a97c88ad1d1cd903a`
is PASS for `XAU_SEQ513_SMOKE_20260723_V5`, binds dataset V24 and source
commit `f05b3390`, and its exact public dry-run passes. V5 had not started at
that recorded boundary; its later empirical failure is recorded in the newer
decision above.

## 2026-07-23 — signed forward-outcome domains are exact

V3 crossed the V1 aux-emission and V2 run-lineage walls, started the trainer
and completed M5/M15/H1/H4/D1 prebuild. It then failed before the first batch
with `ENTRY_V10_CTX_MODEL_NATIVE_ACTIVE_TARGET_CONTRACT_INVALID` because the
trainer incorrectly required `mfe_first_n_bps >= 0`. Selected-side MFE is
spread-aware and legitimately negative when price never earns back the entry
spread: V24 contains 1,952 / 52 / 31 such TRAIN / VAL / TEST rows. MAE remains
a non-negative adverse magnitude.

The audit found a second mismatch in the same domain: train and validation
losses silently clipped signed MFE and `path_quality_bps` to zero. Among
tradable V24 rows, 12,965 / 413 / 216 negative path-quality outcomes were
therefore being rewritten. Commit `c9e2569f` removes the invalid MFE
non-negativity check and preserves exact signed scaling in both losses, with
regression tests keeping negative MAE invalid.

Fresh smoke-readiness SHA-256
`ca27425ea3250cb878f786f5441c4d9c208271a2b64466f14ff613f3940fbb24`
and trainability SHA-256
`960282bbbed0889d06b818abcf5e9ef9ef47b0b44a50d3d32763ad327411d66e`
are READY. V4 recipe SHA-256
`d07b4af58bc019277d4501cd396d3e091b8ee9642dd9fcdb2d73649c554b0083`
is PASS for `XAU_SEQ513_SMOKE_20260723_V4`, binds dataset V24 and source
commit `c9e2569f`, and its exact public dry-run passes. At that recorded
boundary V4 had not started; its later failure is recorded in the newer
decision above.

## 2026-07-23 — dataset-build and training-output lineage are distinct

The first two real V24 smoke executions both failed closed before training and
created no bundle. V1 found that the trainer rejected the producer's stronger
aux-target emission proof because it demanded equality with the static
46-target subset. Commit `9459babe` introduced one exact static-plus-four-row-
counter validator shared by launch and trainer.

V2 crossed that wall and found a second mismatch:
`ENTRY_TRAIN_SPLIT_RUN_ID_LINEAGE_MISMATCH`. V24's three manifests correctly
carry `XAU_SEQ513_REBUILD_20260722_V24`; the smoke output correctly requested
`XAU_SEQ513_SMOKE_20260723_V2`. Treating those two lifecycle roles as one ID
would either reject every reused immutable dataset or erase provenance by
renaming its build lineage.

Commit `b986c8db` establishes the fail-closed separation. The recipe keeps
`run_id` as training/output lineage and adds `dataset_run_id`, derived only
from the bound post-rebuild evidence and all three split manifests. The launch
validator emits it, both public wrappers forward it, the trainer requires exact
CLI/environment/manifest/state agreement, and successful metadata plus lock
must carry one exact `entry_model_native_training_run_lineage_v1`. Collapsed
roles, missing values and meta/lock split-brain are invalid.

The resulting V3 recipe and dry-run passed, then the next real execution
exposed the signed-target mismatch recorded in the newer decision above.

## 2026-07-23 — exact smoke recipe is executable, edge remains unproved

Commits `f08cd904`, `b5a61e21` and `bf5c61a0` establish one canonical
model-native training recipe. The producer owns all 162 decision-affecting
trainer environment values, validates the real split-native pretrain schema,
and binds the exact control, contract, producer, wrapper, trainer and capped
runner bytes. A later documentation commit is permitted because the recorded
source commit must be an existing ancestor, while any executable-source byte
change still invalidates the recipe.

Fresh V24 smoke-readiness and trainability events are READY. Recipe SHA-256
`fa2404603a435d8dc47e26fb2d7345e25b3a2d81b3760e9a0a6c7cf1078ec040`
is PASS for run `XAU_SEQ513_SMOKE_20260723_V1`, and the exact public wrapper
dry-run passes with one epoch, 10,000 rows and 30G/2G memory/swap caps. The
declared bundle directory remains absent. This removes the source blocker but
does not create a model, prediction evidence, edge proof or launch authority.
The next gate is capped execution followed immediately by the exact public
smoke-bundle audit.

## 2026-07-22 — V24 is the current audited dataset, not model authority

V24 rebuilt the complete XAU source cascade through the last complete M5 bar
at `2026-07-22T12:05:00Z` and terminalized GREEN at the designed smoke gate.
Its 369,081 TRAIN / 5,904 June VAL / 4,115 July TEST rows bind the exact
513+142+5 surface. Exhaustive liveness, pretrain, post-rebuild, foundation
feature, complete 46-target, specialist, smoke-manifest, smoke-readiness and
trainability reviews pass on the same six split bytes.

The specialist audit proves zero TRAIN dead signals, zero TRAIN exact duplicate
groups, zero unmapped signal/context fields and all eight model contracts. One
six-field D1 duplicate group is observed only in June VAL because the short
chronological window occupies one regime state; OOS state is recorded rather
than fabricated. V24 is therefore the only current dataset input to the next
smoke gate. It is not an accepted bundle and proves no direction edge.

## 2026-07-22 — liquidity pools, sparse events and preflight keys are exact

V22 failed the specialist audit because five genuine sparse TRAIN events were
evaluated by a generic activity floor and because SMC liquidity-pool low/high
proximity were exact duplicates of S/R support/resistance proximity. Sparse
events now use the canonical liveness support table. SMC pool proximity is a
distinct causal blend of dedicated liquidity proximity (55%), recent swing
(25%) and M5/M15/H1/H4/D1 level clustering (20%); S/R retains repeated-level
memory. V23 and V24 prove zero TRAIN duplicate groups.

V23 then reached smoke readiness and failed only because preflight omitted the
required explicit `iql_distillation` side-effect key. The producer now emits
the exact six-key map with every value false. V24 proves it in a fresh artifact.

## 2026-07-22 — source-wiring proof follows contract ownership

The first V24 trainability review incorrectly required four downstream source
files to contain the resolved contract-mode string and numeric width. Those
files correctly imported `MODEL_NATIVE_CONTRACT_MODE` and
`MODEL_NATIVE_SIGNAL_DIM` from the one signal-contract owner, so duplicating
literals would have weakened the architecture. Commit `0f2b9468` replaces raw
text matching with AST proof that both exact constants are imported from the
owner and used. A hard-coded literal without the import now fails. The
corrected immutable trainability review is READY.

At that boundary smoke training was not executable because the immutable
recipe-audit had no canonical producer and the exact post-smoke bundle audit
was not exposed in the single control surface. The 2026-07-23 decision above
supersedes that source status; hand-authored evidence and direct route bypasses
remain forbidden.

## 2026-07-22 — rejected V21/V22/V23 large splits are removed

The user explicitly requested deletion of large failed artifacts. V22 and V23
split parquets were removed while their small terminal/manifests/audits were
retained. V21 cleanup later used the sole evidence-retention owner with exact
three-file plan, immutable approval, same-device staging/revalidation and
terminal `DELETE_COMPLETE`; it removed 75,648,062,117 bytes. V24 and canonical
XAU sources were not touched.

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

## 2026-07-17 — full-stack families cannot be ranked away (v1, superseded)

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
pre-rebuild gate. At that boundary a fresh seq513 rebuild and candidate were
required to produce learned OOT rows; V24 now satisfies the dataset half only.
Selection/abstention admission still requires a fresh immutable proxy
comparison plus absolute OOT support/confidence, cost and live-like gates.
Historical metadata can never substitute for those bytes.

## 2026-07-21 — one feature state before ranking, dataset and serve (v1, superseded)

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

The first V13 M1-to-M5 seam attempt then exposed an important distinction:
the 47,086-row exact result was collector M1 versus canonical M1, not proof
that every aggregated M5 bucket was complete. Across 9,451 native-M5 overlap
buckets, all 9,404 buckets containing offsets `[0,1,2,3,4]` were bit-exact.
Another 35 daily 22:00 UTC reopen buckets containing only offset `[4]` were
also bit-exact. Twelve other partial buckets were collector holes and differed
from native M5, including large price and volume omissions. V13 therefore
admits only the two overlap-proven coverage forms. Unsupported partial buckets
are omitted and enumerated, not filled; any mismatch among admitted overlap
rows or loss of the declared final M5 bar aborts publication. This is a source
coverage contract and has no direction authority.

## 2026-07-21 — V13 MTF source rejected; V14 source cascade passes

V13 snapshotting passed, but its MTF cache was accidentally built from the
trimmed `cv3_modelrange` artifact instead of full canonical-v3. The cache had
plausible shapes but the source-identity contract would reject it. V13 stopped
before context, source audit, ranking or dataset construction and is diagnostic
only; its partial is not moved, renamed or resumed.

V14 rebuilt the entire source event fresh. Its schema-v3 source proof is PASS
through `2026-07-21T17:00:00Z`: 392,959 rows x 188 columns, all 187 numeric
fields live, zero constants, exact duplicate groups or nonfinite fields, and
all five MTF caches bound to full canonical-v3. This is current clean input,
not a direction-edge claim; rank, dataset, liveness and empirical model gates
remain pending.

The Entry trainer audit also found no generic recency/time-decay objective.
Regime FiLM and specialists can condition on current state, but the loss does
not automatically prioritize 2026 over 2021. Candidate work must compare the
full-history baseline against an immutable TRAIN-only recent-regime adaptation
phase, with June validation and July untouched TEST. Any later refresh remains
an offline challenger requiring paired zero-order shadow before promotion.

## 2026-07-21 — V14 liveness was a real policy FAIL; V15 must be fresh

V14 completed exact TRAIN-only ranking and materialized 369,081 TRAIN, 5,904
VAL and 3,898 TEST rows with 513 ordered signals, 142 continuous context fields
and 5 categorical context fields. Its exhaustive liveness scan completed; it
was not killed. The immutable schema-v2 artifact reported 40 failures. Direct
comparison with `FULL_PLUS_CTX_v3src.parquet` proved that the June D1 zeros and
July state transition were source-exact. EMA50/200 crosses had 1,114 events per
direction on TRAIN, while the generic one-percent policy incorrectly called
them inactive. CHoCH and D1 transition flags were likewise genuine sparse
impulses. Current ATR was shifted against the aggregate five-year TRAIN mean
but covered by recent TRAIN regimes (for example, trailing-one-year D1 ATR
mean 85.23 versus June 98.55 and July 89.58).

The schema-v3 contract therefore makes TRAIN the strict learnability surface:
every numeric field needs real variation/activity or an exact declared
sparse-event support floor, and every categorical field needs multiple TRAIN
states. VAL/TEST remain exhaustive immutable chronological observations; one
genuine state is recorded as `OBSERVED_SINGLE_STATE`, while NaN/Inf, shape,
order, seq/snapshot parity and categorical values outside the TRAIN vocabulary
still fail closed. ATR shift remains recorded but cannot substitute for the
later untouched OOS edge/calibration/cost gates. This is not a constant
allowlist and does not create synthetic OOS activity.

The chain also incorrectly retried the whole wrapper after V14 had already
written immutable split/audit outputs. Retry is now permitted only when exact
checkpoints exist and no split, manifest or audit output has begun. Otherwise
the event terminalizes RED and a fresh lineage is mandatory. V14 remains
non-authoritative; V15 must rebuild under the repaired source revision.

## 2026-07-21 — V15/V16 expose early provenance and history-window gaps

V15 rebuilt a current source through 18:10 UTC, but its caller named the fresh
ranking target with timestamp 18:31 while the manifest boundary occurred at
18:30:37. The manifest correctly rejected the impossible ordering. V15 is
terminal RED and was not renamed or resumed.

V16 rebuilt independently through 18:25 UTC. Its source, ranking and exact 513
manifest passed, but preflight rejected `covers_history_start=false`: raw
model-range began 2021-01-04 and the context producer's causal warmup made the
first finite row 2021-01-14, later than the required 2021-01-05 common-history
boundary. V14 had proven the correct upstream start, 2020-11-13, yielding a
finite surface from 2021-01-04. This is a source-window error, not permission to
relax warmup or fill values.

Source-cascade audit schema v4 now requires and binds the common-history start.
The one-shot chain independently scans the exact source time column during its
initial contract validation, requiring ordered unique coverage of history and
TEST plus at least 96 pre-TRAIN rows. It also rejects a future timestamp in the
requested fresh ranking filename. Both failures therefore occur before rank
state/ranking in the next lineage. V17 must rebuild from raw model-range start
2020-11-13; V14-V16 remain non-authoritative terminal evidence.

## 2026-07-21 — V17 proved input liveness; later superseded

V17 rebuilt independently through the last complete M5 bar at 18:40 UTC. Its
source, TRAIN-only ranking, exact 513-field manifest, preflight, all three
datasets and exhaustive schema-v3 input-liveness contract passed. This proves
that 513 signals plus 142 continuous and five categorical context fields are
present, ordered and active under their declared TRAIN/event contracts. It is
not yet evidence of model precision or profitable OOS direction selection.

The next pretrain gate failed solely because it searched the event-local tape
directory name for an instrument substring. That heuristic is deleted from
both audit and trainer. One shared fail-closed contract now traces exact
`XAU_USD` identity to the canonical M1/M5 manifests and binds run id, producer
method, repaired/current manifests, geometry, overlap, every yearly tape hash
and every collector-snapshot hash. Dataset manifests capture that proof and
consumers revalidate it. The repair and current-snapshot schemas advanced to
v2, source-cascade proof to v5 and pretrain audit to v2; older artifacts could
not pass. V17 remains terminal RED. At that historical boundary only a fresh
V18 lineage could proceed; V24 is now current.

The command hook is now GX1-local and retains only its Git/secret safety floor.
Runtime market identity belongs to the model/data contracts, not path-name
blacklists. Explicit references to instruments outside GX1 were removed from
the repository and active hook; negative tests use an abstract invalid token.

## 2026-07-22 — V18 clock-order RED; V19 temporarily admitted, later rejected

V18 rebuilt current source successfully but could not cross the fresh-ranking
boundary after the host clock moved backwards by roughly 22 seconds. The
requested immutable ranking filename then appeared to be from the future. The
contract correctly terminalized the lineage RED; no V18 partial may be reused.

V19 rebuilt independently under source revision
`d2c44fa94c447b9c7d9bf23740b0a811d02c8c62`. Its chain terminal is GREEN and
states `stopped at smoke gate`. The lineage binds exact XAU tape provenance,
392,995 x 188 finite/live source rows through 2026-07-21T20:00:00Z, full-v3
five-timeframe caches, TRAIN-only ranking, exact 513 manifest and 369,081 /
5,904 / 3,934 chronological TRAIN/VAL/TEST rows. Exhaustive input liveness
validates all 1,980 field/split records; pretrain validates all 34 auxiliary
targets and zero consistency mismatches. June/July H4/D1 ATR shift was retained
as `SHIFT_OBSERVED`. V19 was temporarily admitted as dataset evidence only
under that contract. The later foundation audit below rejected it; it never
contained a trained model or granted direction-edge, bundle or launch authority.

## 2026-07-22 — post-rebuild smoke uses the canonical dataset or fails closed

The post-V19 ownership audit found an impossible downstream route: smoke
required a legacy readiness artifact whose producer had been deleted and a
separate smoke-split schema no active producer could create. A new immutable
post-rebuild readiness producer now validates the exact V19 green terminal,
preflight, full-input liveness, pretrain audit and six split files/manifests.
It requires `source_dataset_dir == smoke_dataset_dir`; copied, parallel or
schema-divergent smoke datasets are rejected. Smoke manifest, readiness and
trainability consume this contract through the single Entry control surface.
The launch validator, trainer, smoke-bundle audit and adoption-readiness now
all require the same canonical `entry_model_native_seq513_split_manifest_v2`;
the unproducible smoke-only split schema is removed end to end.

The first real V19 post-rebuild attempt then published an immutable BLOCK. It
proved all six artifact content hashes but exposed two adapter mistakes: the
producer read nonexistent `feature_contract.signal_dim` instead of validating
the canonical ordered `signal_bridge_fields`, and passed six-artifact key names
to the liveness validator where its API requires `{path, sha256}` manifest
bindings. The producer now validates the complete model-native signal contract,
exact 513-field order and the validator's native binding shape. No V19 bytes or
admission policy were changed; the BLOCK event remains diagnostic evidence.

## 2026-07-22 — unavailable HTF evidence is not a neutral zero

V19 source logs disclosed 12 leading M5 rows before completed H1/H4 evidence.
They were outside every admitted model split, but the legacy builder represented
them as zero. Future construction instead keeps the historical warmup prefix
as NaN so the causal trim contract owns it, and live construction raises when
no completed HTF evidence exists. The repository also contained an unreachable
stateful alignment branch with no constructor/caller and with `j-1` HTF-bar
semantics, while the active path shifted one M5 row. That branch and its unused
state fields are deleted. H1 and H4 now share one causal alignment owner, so
dataset/serve parity cannot silently depend on which branch happened to run.

## 2026-07-22 — V19 is rejected; all 57 foundation fields become mandatory

The corrected post-rebuild producer passed V19 with immutable event
`ENTRY_MODEL_NATIVE_SEQ513_POST_REBUILD_READINESS_20260722T070548397061Z.json`
and SHA-256
`405f827162779ecea57e01c361a0c73e2d933501dd5f9e3dcfabc977ee3dd920`.
The next real foundation-feature audit then failed closed. V19 contains none of
the 57 implemented `chart.foundation_*` fields, and its split metadata cannot
claim the required foundation version/count/all-selected contract.

The root mismatch was deterministic: `build_chart_layer()` computes the full
foundation layer and the downstream structure/SMC/volatility/session layers
consume it, but the mandatory registry omitted it. The TRAIN ranker's
reflective candidate-universe scan also did not expose
`FOUNDATION_STRUCTURE_FEATURE_NAMES`, so none of the fields could enter the
163 ranking-owned positions. Tests had used synthetic manifests that manually
inserted the fields and therefore did not exercise the real producer path.

An independent scan of all 5,808 post-warmup V19 validation rows proved every
foundation field active. No foundation field was exactly equal to any of the
prior 316 mandatory fields. Expected semantic correlations are high for some
derived pairs, but 28 explicit session-by-structure interactions and the raw
HH/HL/LH/LL, BOS/CHoCH age, sweep/reclaim, compression-release and
impulse/pullback surfaces retain distinct values.

The active signal contract is therefore
`xau_seq513_model_native_direction_v2`: 34 base fields plus 479 specialist
fields, partitioned as 373 mandatory outputs from twelve code-owned layers and
106 deterministic TRAIN-only ranked fields. Signal, mandatory-stack, ranking
and manifest schemas are versioned forward. All 57 foundation fields route to
one of four learned specialists (19 structure, 5 SMC/liquidity, 5 volatility,
28 session/regime), while the final direction remains model-only. V19 and all
older artifacts fail the new contract and cannot be trained; V20 must rebuild
the full ranking/manifest/split chain from immutable inputs.

The same formula audit found a separate sign mismatch shared by four active
builders. `H1_range_compression_ratio` and
`M15_range_compression_ratio` are `ATR14 / ATR100`, so values below one mean
compression and values above one mean expansion. `_v1_bb_squeeze_20_2` is
`bandwidth / mean_bandwidth - 1`, so negative values mean squeeze. Foundation,
volatility, chart-geometry, chart-core and deep-interaction code had treated
the raw positive value as compression. One strict owner now converts both
source contracts into separate [0,1] compression/expansion pressures, and
release uses lagged compression followed by positive expansion acceleration.
Impossible non-positive ATR ratios fail closed. On the 5,808 post-warmup V19
validation rows, all 57 corrected foundation fields remain finite and
non-constant; this diagnostic does not rehabilitate V19 or select a feature.

## 2026-07-22 — V20 is structural-label RED; all target prerequisites become mandatory

V20 rebuilt all source artifacts from canonical roots through the last complete
M5 bar at 2026-07-22T07:35:00Z. Its 393,122 x 188 FULL_PLUS source audit passed
with all 187 numeric fields finite/live and no constants, exact duplicates or
fallback. A fresh TRAIN-only ranking, real 513-field manifest and rebuild
preflight also passed. No V19 artifact was reused.

Dataset construction then failed closed before any split publication because
`chart.geometry_channel_position_low_to_high` is consumed by the structural
auxiliary-label producer but had remained optional ranking evidence. It did not
win V20 selection. The one exact checkpoint retry failed identically. V20's
immutable terminal is RED at `dataset-rebuild-exact-checkpoint-resume`, SHA-256
`0b60ceda8b72f45cc76d83c3e4bb681bc5f190f1b0200a67391140e0a293e606`.
Nothing from V20 may be resumed or reused.

All 19 current-bar structural-label signal requirements now have one code-owned
registry imported by both the model-native signal contract and dataset builder.
Every requirement must resolve to a mandatory selected field. The four geometry
prerequisites are promoted into the chart-geometry layer, changing the exact
partition from 373 mandatory + 106 ranked to 377 mandatory + 102 ranked while
preserving 34 + 479 = 513 total fields. Signal, split-manifest, ranking,
signal-manifest, preflight and foundation-policy identities are versioned
forward. This repaired ownership and construction reachability only; V21 was
still required to prove dataset validity and empirical direction edge.

## 2026-07-22 — V21 is pretrain-polarity RED; audit dependencies are contract-bound

V21 rebuilt independently from current canonical roots through the last closed
M5 bar at `2026-07-22T08:05:00Z`. It produced 369,081 TRAIN, 5,904 VAL and
4,067 TEST rows with exact 513+142+5 inputs. LONG/SHORT/FLAT and path-quality
targets were non-degenerate, and the exhaustive full-input liveness artifact
passed all 1,980 field/split records.

The post-build pretrain audit then failed because
`chart.geometry_support_minus_resistance_stack` was a required input to its
support/resistance channel-polarity proof but was not mandatory and had not won
the 102-field TRAIN-ranked remainder. Because the audit returned immediately
on missing polarity, it also reported target consistency unavailable. V21
terminalized RED at dataset-rebuild. Its terminal SHA-256 is
`4c6186eb37992c8b576ba334bc02375c18b73a60ab80af4b3826f07d4c01e2d8`;
no V21 artifact may be resumed or reused.

The repaired audit was then run once against V21 as explicitly non-authoritative
diagnostic evidence. It remained RED on the old v3 identity and missing field,
but all required target columns were live in all splits and every target
consistency mismatch counter was zero. The diagnostic artifact SHA-256 is
`351e60c0e7f03063fdf03ded7cb5fd716b7c54830e5193c2ccf59b0bff094cbe`.
V21's TRAIN ranking contained the missing field at rank 123 with nonzero score;
it was live but fell outside the 102 optional positions.

The polarity requirements now have one contract imported by both the signal
identity and pretrain audit. Support-minus-resistance joins the mandatory
chart-geometry family, changing the partition to 378 mandatory + 101 ranked
while retaining 479 specialist and 513 total signals. The pretrain audit
computes target liveness and consistency even when polarity evidence is
missing; missing polarity still independently fails closed. All dependent
signal, mandatory-stack, split-manifest, ranking, manifest, preflight and
foundation-policy identities were versioned forward. At that historical
boundary V22 had to be wholly fresh; V24 now proves the repaired partition.

At the operator's request, 18 large parquet splits from rejected V12, V14,
V17, V19 and the obsolete pre-foundation seq520 lineage were deleted after
exact path/status/reference checks. The deletion removed 386,704,608,791
bytes. Small terminal, manifest and audit records remain as failure evidence;
active V21 and canonical XAU sources were not touched.
