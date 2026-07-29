# GX1 current decisions

Historical implementation narratives were removed because they repeatedly
acted as stale authority. Git history retains them. This file records only
decisions that constrain the current model-native Entry work. Later entries
supersede earlier event-specific state; historical headings below describe the
boundary at which each decision was made, not current artifact authority.

## 2026-07-23 — first audit repair checkpoint; later superseded

The post-V7 findings were reproduced before repair. Source now closes signed
dip-MFE, selected-side bad-path symmetry, no-replacement sampling,
bidirectional clean/survival weighting, exact 162-value trainer recipe, M5
path/hash authority, one causal MTF cache identity, all-22-head checkpoint
liveness/influence, raw-bps forward-head units, CLI gradient accumulation and
stale loader compatibility inputs. Focused integration completed with 391
tests passing and two declared runtime skips.

This historical checkpoint was later superseded the same day by the complete
TRAIN-fit normalization and 142+5 one-owner context-routing decisions at the
end of this file. The 82-alias observation remains V24 fixture evidence only,
not a current architecture constant.

Launch design found a separate bootstrap conflict. Joint Exit/sizing now binds
the actual recursive Exit artifact bytes, and consumer-side immutable approval
is repaired. The transactional launch producer remains open.

Decision:

- preserve all completed repairs and their regression contracts;
- preserve the later normalization, routing, MTF, bundle/event, Exit-byte and
  runtime fail-close repairs before fresh rebuild;
- implement the transactional promotion/launch producer;
- keep launch `BLOCK`; V24/V7 remain immutable failed evidence.

## 2026-07-23 — V7 is terminal RED; pipeline audit requires rebuild

V7 ran on the exact immutable 25,000-row recipe and completed six full
TRAIN/VAL epochs. The hard-red slice policy then stopped epochs seven/eight
with `TRAIN_FAIL_NO_BEST_STATE`. No checkpoint or bundle was written and the
72.71 GB temporary memmap was cleaned.

Raw VAL accuracy peaked at 0.403455 only through 85.1118% FLAT. The final
epoch predicted 71.4092% SHORT, failed 32 direction-slice checks, retained
bad-path/survival AUC 0.478/0.514, six cross-head collapse pairs and
specialist/family×TF minimum means 0.000054/0.000300. The TF gate itself
passed at 0.024166, proving that one healthy aggregate gate cannot substitute
for full cooperation and directional evidence.

Three independent read-only audits then reviewed the complete data, model and
runtime pipeline. `PIPELINE_AUDIT_XAU_20260723.md` is the detailed record.
Two findings are P0:

1. the selected-side `y_bad_path` probability penalty always suppresses LONG,
   including all 425 SHORT bad-path rows in the exact V7 cap;
2. six signed spread-aware dip-MFE targets are clipped to zero, corrupting V24
   and requiring a fresh rebuild.

P1 findings include 62% unique-row coverage from replacement sampling,
LONG-only positive weights for bidirectional clean/survival targets, global
path AUC leakage through tradable-versus-FLAT, partial checkpoint coverage,
taxonomy-only context specialist routing, incomplete MTF/scaler identity,
incompatible fusion units and missing transactional launch authority.

Decision:

- V24 and V7 are immutable failure evidence and cannot be retrained or
  promoted.
- No threshold is relaxed and no V8 is materialized.
- Repair target semantics and objective symmetry first, then sampling,
  conditional metrics/all-head influence, MTF/scaler/context/fusion contracts,
  atomic bundle/promotion/vedtak and handover lifecycle.
- Rebuild all XAU-only splits only after focused and full source proof.
- Launch remains `BLOCK`.

## 2026-07-23 — V6 proves objective instability; cooperation becomes an admission gate

V6 completed six full train/VAL epochs with optimizer steps and no runtime
contract failure. Epoch 4 was the strongest direction candidate: accuracy
0.383469, direction checkpoint score 0.361111, prediction rates
0.256267/0.373645/0.370088 and near-label global balance. It still failed 15
local context slices, while bad-path/clean-edge/survival AUC was
0.472/0.517/0.501. By epoch 6, LONG support had fallen to 0.058943, slice
failures rose to 29, bad-path/survival AUC remained 0.474/0.509, and
clean-edge/path-quality predictions collapsed to Spearman +0.959. The
corresponding VAL targets correlate only +0.699, so this is learned
redundancy, not target identity. No epoch admitted a checkpoint; early
stopping emitted `TRAIN_FAIL_NO_BEST_STATE` and no bundle.

The review found a separate observability gap: VAL computed regularization for
`specialist_gate`, `tf_gate` and `family_tf_cooperation_gate` but discarded
their use statistics. Checkpoint admission therefore depended on auxiliary
health but not on direct evidence that every cooperation path remained live.
Commit `37128985` now aggregates exact epoch-wide mean weights and entropy for
all three gates and blocks checkpoint admission when any token mean is at or
below the existing 0.01 minimum or entropy misses its existing training floor.
The direction-neutral gate-balance weight rises from 0.05 to 0.50 to oppose
the starvation observed in V6. No direction, AUC, slice or promotion threshold
is reduced.

Fresh smoke-readiness SHA-256
`395d76f9dbe58e7c5a2c9a7488de32d320487efa0942908fcc39a57219034ebb`
and trainability SHA-256
`9f05c6970e7ee17fd8dba5c5583a6332fc068c59d34068e0e49a218079048e77`
are READY. V7 recipe SHA-256
`fc012059594f5a197fdf145c86487e74ddfeba997f2604fa6759a0378416568d`
is PASS for `XAU_SEQ513_SMOKE_20260723_V7`, binds source `37128985`, and its
public dry-run passes. V7 declares 25,000 stratified TRAIN rows, eight epochs
and patience eight so rare bad-path support rises from roughly 334 to roughly
835 positive rows. Exact V24 bytes, seed, learning rate and all empirical
acceptance thresholds remain fixed. V7 has not executed at this boundary.

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

## 2026-07-23 — full-stack input cooperation is contract-owned

The post-V7 audit proved that field presence was insufficient: raw scales
spanned millions, 147 context fields did not enter their claimed specialist
tokens, disk MTF components were replaceable and global auxiliary metrics
could pass without conditional path skill.

The active source now fits one ordered immutable normalization contract on the
complete physical TRAIN population before sampling for all 513 signals, 142
continuous context values and each five-timeframe 25-field surface. Binary and
categorical semantics remain exact; every continuous statistic, categorical
domain, causal TF source-row hash and alias owner is bound into metadata, lock
and persistent model state. VAL, TEST, replay and serve cannot refit.

All 142 continuous and five categorical fields have exactly one of eight
family owners. Family projections enter the owned specialist token before
cross-attention; categorical fields use separate field/domain embeddings. The
signal/context alias set is derived from the actual ordered signal names and
must be bit-identical. V24's 82 aliases are fixture evidence, not a hard-coded
constant. RSI, percentage change and rate-of-change evidence routes to
momentum. Conditional tradable/LONG/SHORT path metrics and incremental lift,
all 22 active heads and all 26 fusion groups are checkpoint-blocking.

Optional TRAIN ranking now uses exact spread-aware LONG utility minus SHORT
utility with final PnL, MFE, MAE and path-quality terms, not H24 mid-close
return. These changes prove one learned cooperation path in code; they do not
prove OOS direction edge.

## 2026-07-23 — MTF, bundle, Exit and runtime identities fail closed

Admitted training requires one exact V2 disk cache bound to its M5 source,
ten component array hashes/sizes, exact 11-file inventory and aggregate
identity. Source-build fallback is not an admitted trainer mode.

Trained, calibrated and sizing-finalized bundles use a shared exact commit
manifest. They are built under a hidden sibling path, fsynced, strict-loaded
before visibility and atomically published with no replacement. Immutable
JSON events follow the same hidden-stage/fsync/no-replace visibility rule.
Smoke bundle audit schema v4 binds the commit.

Joint Entry-sizing/Exit proof schema v2 recursively binds path, size and
SHA-256 for every regular file under the selected XGB, V3 Exit and Exit-IQL
artifacts; registry JSON alone is no longer sufficient. Runtime retains file
stats and rejects in-place changes.

A missing OANDA trade ID is unresolved exposure and can no longer send an
opposite market order on a hedging account. Journal writes are locked and
fsynced.

## 2026-07-23 — environment text is not launch authority

`GX1_SMART_LAUNCH_VEDTAK` is removed from both launcher and runner. A future
ALLOW consumer requires the newest immutable one-time approval with an exact
ID, event SHA, complete launch-state payload hash and bundle-commit hash.
The runner captures an exact launch/registry lease at startup and revalidates
the unchanged identity and all freshness evidence before every new exposure.

No canonical transactional candidate→promotion→launch producer exists yet.
That remains a source P0; consumer hardening does not manufacture a safe
completion path. Launch remains `BLOCK`. No rebuild, training, calibration,
replay, paper/demo/live execution or promotion was run, and V24/V7 remain
immutable failure evidence.

## 2026-07-23 — transactional launch source P0 is closed; launch stays BLOCK

The missing bounded authority now exists as
`gx1/scripts/finalize_entry_model_native_launch_v1.py` with its exact
transaction contract in
`gx1/contracts/entry_model_native_launch_transaction_v1.py`. This is a
genuinely separate transaction authority, not a minor-version copy or
compatibility wrapper, and it is reachable only through the existing
`scripts/entry_next_edge_control.sh model-native-finalize-launch` surface.
The public CLI fixes the repository registry/state and canonical
`GX1_DATA/reports/entry_model_native_launch_authority` roots.

The finalizer cannot mint its own approval. A pre-existing one-time vedtak
must bind the exact bundle commit, transaction ID, canonical targets,
operating point and all prerequisite evidence. Exact same-byte/no-symlink
reads, recursive active-Exit projection, TEST-replay portfolio-capacity proof,
a stable process lock, registry compare-and-swap, local immutable backups and
strict COMMIT/FAIL validation form one recoverable registry/state
transaction. Partial replacement restores both original targets and publishes
newer terminal FAIL evidence. Runtime revalidates the accepted lease before
each new exposure.

Read-only data inspection corrected the December-2024 boundary: canonical M5
and live-prebuilt share 3,430 impossible-geometry rows, 2,799 on weekends.
Clean M1 supports 5,757 rebuilt December M5 buckets and leaves 3,459 canonical
rows without M1 backing. No canonical or live data were modified.

Decision:

- `remaining_source_p0=[]`; preserve all source repairs;
- keep launch `BLOCK`: no accepted fresh dataset, model, bundle, replay,
  lifecycle chain, real vedtak or launch transaction exists;
- repair and prove canonical/live December-2024 parity before rebuild;
- then rebuild and train from fresh XAU-only bytes, comparing a declared
  full-history baseline with a TRAIN-only recent-regime challenger while
  preserving final TEST;
- extend the existing script/contract owner for minor changes. A new file is
  permitted only for a genuinely new bounded authority and must remain routed
  through the existing public control surface.

## 2026-07-23 — adversarial Exit replay re-audit reopens one source P0

The transaction mechanics above remain repaired, but the assumption that the
joint Exit proof was model-produced was false. The existing sizing finalizer
hashes the selected XGB/V3/Exit-IQL artifacts and validates caller-supplied
replay/trace parquets; it never executes those models. A synthetic linear
price trace could therefore satisfy the diagnostic schema without proving
that the active Exit stack caused any action.

Containment is now exact and fail-closed:

- `entry_fill_time` is explicit and exactly decision T+5;
- replay rows must extend the exact canonical OOS TEST rows;
- every trace-step PnL is recomputed from bid/ask;
- simultaneous exposure is fixed to one until shared
  equity/margin/drawdown replay exists;
- cached sizing authority rehashes every bound byte on every application;
- runner admission reconciles exact broker/local XAU trade IDs and requires
  the same exposure transaction snapshot immediately before a new order;
- launch finalizer and runtime artifact guard reject caller-supplied Exit
  replay before vedtak consumption or authority mutation.

Decision:

- `remaining_source_p0=["canonical_full_test_active_exit_replay_producer"]`;
- reuse `V12Pipeline.make_exit_decision`; do not implement a second Exit
  policy;
- the producer must own full TEST iteration, complete frozen Entry snapshots,
  T+5 fills, immutable M1/canonical/BASE28/MTF state, active model/config bytes
  and complete per-M1 input/output/action traces;
- fallback, floor-only approximation, missing bar/state and horizon-cap are
  terminal red;
- launch remains `BLOCK`; the repaired transaction finalizer must not be
  confused with completed upstream evidence.

## 2026-07-23 — full feature/Exit audit supersedes the single-P0 view

Three parallel read-only audits traced the data/feature producer, model
training and live inference/launch paths field by field. They proved that the
old active Exit selection is not a valid incumbent: its per-bar data predate
the exact T+5 fill, about 80% of sampled M1 rows join a still-forming M5 bar,
the V3 trade overlay is shifted one M1 row, the fixed live M5 fetch leaves part
of a real 512-M1 window zero-filled, and five M1 microfeatures use different
train/serve formulas. TradeState also persisted partial transitions and lost
cadence/deferral identity on restart.

The Exit artifact contract was independently invalid. Its summary declares
research-only/non-production, ordered feature names are not bound into the old
checkpoints, and three folds have no explicit serving fold. The V3 lineage is
not reproducible from its declared deleted input. No code-only waiver can
rehabilitate these bytes.

Decision:

- keep Entry, candidate replay and all paper/demo/live launch `BLOCK`;
- extend existing owners only; no minor-version or one-off scripts;
- require exact T+5 fill, first closed fill-bar state with
  `bars_in_trade=1`, exact last-closed-M5 joins and exact V3 overlay timestamps;
- require full 512-M1-derived M5 coverage, complete finite feature state and
  one shared M1 microfeature implementation;
- stage M1→V3→Exit on a cloned TradeState, persist exact last M1 and
  Strategy-F deferral, and commit only a complete decision;
- require production flags, explicit serving fold, all ordered features and
  summary↔checkpoint feature SHA before Exit load;
- rebuild/rescore/retrain Exit after data/source repair; old outputs have zero
  launch authority;
- close in source: complete-history canonical-v2 recomputation, atomic
  immutable canonical-v3/BASE28 generation identity, native-M5
  market-closure/schema/hash ownership, removal of non-observable
  slippage-derived decision fields and reproducible V3 lineage bound to the
  exact XGB bridge;
- retain as open P0s: the exact model-native V3 training-dataset writer,
  canonical active-Exit full-TEST producer, native-M5 materialization/pair
  bootstrap and fresh XGB/V3/Exit artifact rebuild/rescore/retrain;
- close the formula/alignment P0 in source: PLUS5 ATR/ROC/VWAP and dependent
  normalized VWAP have one shared owner, SMC consumes the published ATR, and
  H1/H4 selects the state available at M5 decision time without an extra lag;
- keep current prebuilts blocked: manifest-bound loader admission proves path,
  SHA, rows and Arrow schema, and the still-running legacy updater has already
  produced a canonical parquet whose bytes no longer match its manifest;
- treat the 2,375 invalid prebuilt OHLC rows in late 2024 as a separate data
  rebuild/quarantine blocker. No canonical/live data were modified.

## 2026-07-23 — exact V3 training-dataset producer is source-complete

The strict V3 reader/materializer previously proved storage semantics but did
not own construction or publication of a complete dataset. That left a false
authority boundary: an external caller could assemble the matrix, overlays and
records and then ask only for validation.

Decision:

- extend `gx1/exits/training/thin_record_dataset.py`, the existing V3 dataset
  owner; do not add a versioned builder script;
- expose the operation through the existing
  `scripts/entry_next_edge_control.sh` route
  `model-native-v3-exit-dataset`;
- accept only explicit immutable prediction evidence, Entry bundle/dataset,
  chronological SourceTape, XGB bundle and frozen canonical-v3/BASE28 pair
  identities;
- derive every 173-field base row through the shared serving builder and every
  T+5/240-bar overlay/record through the shared model-native materializer;
- bind all input bytes, producer source bytes, XGB identity, runtime-head
  trade identities, direction support and dataset members into one exact PASS
  event and manifest;
- publish by fsynced atomic no-replace rename, then re-open and validate the
  final bytes;
- reject unsorted source rows instead of silently reordering them, and recheck
  the frozen pair manifest/files before exposing its frames;
- keep launch `BLOCK`: this source proof creates no production dataset,
  artifact, OOS edge or live authority. The remaining source P0 is the
  canonical full-TEST active-Exit producer.

## 2026-07-23 — canonical full-TEST active-Exit producer is source-complete

The retained joint finalizer remains a validator for caller-supplied diagnostic
rows and therefore still has zero launch authority. The missing production
boundary is now implemented by extending that same sizing/replay owner rather
than adding another script.

Decision:

- add replay schema v7 with nested canonical producer evidence;
- expose `produce-canonical-joint-exit-proof` through the existing
  `entry_next_edge_control.sh` route
  `model-native-canonical-active-exit-replay`;
- derive the exact canonical OOS TEST row set and runtime-head direction
  evidence internally; callers cannot supply actions, fills, trace rows,
  horizon caps or matrices;
- emit explicit `FLAT_NO_ORDER` for model FLAT and open an independent
  unit-normalized research TradeState at exact T+5 for each LONG/SHORT row;
- call `V12Pipeline.make_exit_decision` on each consecutive M1 step until
  actual `EXIT_NOW`; missing cadence/state/inference or SourceTape exhaustion
  is terminal red;
- bind and revalidate prediction/report provenance, canonical OOS rows,
  SourceTape, frozen canonical-v3/BASE28 generation, recursive active Exit
  artifacts, transitive producer source and exact immutable replay/trace
  outputs;
- keep the public caller-parquet compatibility operation diagnostic-only and
  require canonical producer evidence before vedtak consumption or launch
  mutation;
- cover LONG, SHORT and FLAT plus forged output/tape binding rejection in the
  end-to-end contract test;
- preserve the existing-owner rule: no new versioned script was created;
- include replay-readiness and shadow-event timestamps when selecting the
  lifecycle event floor, closing a separate 77 ms future-evidence race exposed
  by the broader regression run;
- keep launch `BLOCK`: no accepted fresh Entry or causal production Exit chain
  exists, current native-M5/pair data are noncompliant and the canonical/live
  December-2024 parity issue remains unresolved.

## 2026-07-24 — native OANDA M5 producer completed and made efficient

Audit found that the declared native-M5 owner was only a validator contract.
The actual script still mutated one current-year parquet in place, silently
continued after failed request chunks, could return empty success, exposed an
alternate-provider repair path, wrote no canonical manifest and had no atomic
root publication. No process, systemd unit, cron entry or repository caller
used those legacy modes.

Decision:

- extend `gx1/scripts/backfill_xauusd_m5_from_oanda.py`; do not add another
  backfill script;
- remove direct canonical-year mutation, gap-repair, alternate-provider,
  synthesis and empty-success behavior;
- retain each normalized OANDA MBA response as deterministic compressed
  evidence and admit only literal `complete=true` XAU_USD M5 rows;
- require exact UTC/M5 ordering, positive finite MBA geometry, integer
  non-negative volume, strict 14-column Arrow schema and source-defined market
  closure;
- bind request closure, response chunks, clean Git/source inventory, typed row
  digest and every year hash/count/bounds in
  `xau_canonical_m5_source_v2`;
- independently stream-rederive source and parquet digests before fsynced
  atomic no-replace publication;
- use at most 15 days/4,320 M5 slots per request and no fixed sleep; the shared
  OANDA client remains the sole retry/429/backoff owner;
- vectorize the stable typed-row digest byte-identically and stream both
  producer output and validator state, avoiding multiple full-history frame
  copies;
- expose only `model-native-native-m5-source` with explicit vedtak, interval,
  chunk size and fresh absolute output root;
- keep launch `BLOCK` and do not run the external-data operation in this
  checkpoint;
- record the next seam honestly: `v12_canonical_incremental` can extend or
  copy an already admitted pair, but no complete initial
  native→canonical-v3/BASE28 builder is yet exposed. Copying the invalid old
  pair is not a compliant bootstrap.

## 2026-07-24 — unify native M1/M5 ownership and narrow BASE28

A three-way pipeline audit found that M1 still had several mutable writers and
a shallow descriptor while M5 alone had the strict source contract. It also
found split feature authority: broad BASE28 duplicated closed-M5/canonical
fields, consumers resolved duplicates differently, and the global
`regime_bucket_edges_v1.json` was mutable, stale and not fitted only on TRAIN.

Decision:

- extend the same historical OANDA owner and source validator to strict native
  M1 and M5 schema v3; do not create a second producer;
- fix source policy at three days for M1 and 15 days for M5, each 4,320
  theoretical slots, with no caller chunk or granularity override;
- require exact 14-column physical order for both timeframes and validate both
  descriptors through the complete response↔parquet rederivation contract;
- expose the two fixed control routes `model-native-native-m1-source` and
  `model-native-native-m5-source`;
- remove the three unreachable mutable M1 writer sources
  `materialize_backfill_xauusd_m1_2020_2024_v1.py`,
  `materialize_backfill_xauusd_m1_repair_v1.py` and
  `v12_backfill_to_present.py`; their history remains recoverable in Git;
- make raw BASE28 own only the 13 physical native-M1 market fields in exact
  source order; derive phase and volume transforms causally at train/serve;
- derive `m5_phase_0..4` only from the real M1 timestamp and define phase 4 as
  the first decision timestamp that can observe the just-closed M5 bar;
- replace the per-row/per-field BASE routing loop with one vectorized frame
  mapping; 100,000 rows × 54 fields measured about 0.033 seconds locally;
- prohibit TRAIN-fit `atr_bucket` and `spread_bucket` from the raw pair. The
  future builder must publish a separate immutable TRAIN-only rank reference
  and bind it through dataset, bundle, replay and live;
- keep launch `BLOCK`. Neither strict native source route, the complete raw
  pair bootstrap nor any training/live operation was executed.

## 2026-07-24 — make the raw pair path complete, causal and load-efficient

The follow-up field/consumer audit proved four remaining mismatches:

- M15/H1/H4/D1, cyclic and session features used the M5 bar-start label
  instead of the decision-availability timestamp. At an M15/D1 boundary this
  delayed newly closed evidence by one M5 bar;
- Exit's mutable full-history bucket file disagreed with the exact TRAIN-only
  ECDF on 13.577428% of ATR rows and 30.579820% of spread rows;
- bootstrap copied caller-supplied prebuilts and its v1 pair manifest bound
  output bytes/schema, but not the native inputs, formulas or producer source;
- a complete persisted canonical surface was recomputed during every
  load/refresh. Local measurements attributed roughly 17 seconds to repeated
  raw-BASE augment work and roughly 95 seconds to compatibility augmentation.

Decision:

- extend `gx1.execution.v12_canonical_incremental`; do not create a replacement
  builder. It now accepts only two immutable native snapshot roots plus an
  explicit vedtak, rebuilds the complete model-agnostic canonical surface,
  derives raw BASE28 as exactly the 13 native M1 fields, and publishes one
  immutable atomic pair;
- bind native manifest hashes, source environment/interval, clean commit,
  producer source inventory, formula/timing declaration, coverage and output
  schemas in pair manifest v2. Frozen-pair consumers expose the exact lineage
  and its digest to downstream evidence;
- expose the same owner as `model-native-canonical-pair` through the existing
  capped control surface. Remove copy/bootstrap compatibility and daemon-loop
  behavior;
- use one shared M5 decision-availability function (`bar_start + 5 minutes`)
  for HTF merges, cyclic features and sessions;
- keep `atr` as the canonical Wilder ATR and `atr_bps` as the normalized
  market value. Do not let context augmentation overwrite canonical `atr`;
- make model-agnostic augmentation remove TRAIN buckets. Derive ATR/spread
  buckets only from an explicit immutable `TrainRankReferenceV2`; delete the
  obsolete mutable global bucket writer/reader;
- validate the complete persisted canonical surface once. Compatibility
  augmenters must return without recomputation when every required field is
  already present;
- keep launch `BLOCK`. No native source, pair, rank, dataset, training, replay
  or live operation was executed. Exit dataset/bundle/replay/live still need
  one exact bound TRAIN-rank identity and all historical Exit artifacts remain
  invalid.

## 2026-07-24 — native production executed, first pair generation, exit rank binding

Both strict native source routes ran for real under vedtak
`XAU_NATIVE_PAIR_BOOTSTRAP_20260724_V1`. The accepted roots cover
`[2019-01-01, 2026-07-24T12:00)` and were produced from clean commit
`1751c931`: `XAU_M1_NATIVE_2019_20260724_V2` (2,652,244 rows, 921 chunks)
and `XAU_M5_NATIVE_2019_20260724_V2` (535,978 rows, 185 chunks). The
first 2020-11-01 roots are superseded diagnostics only: the live D1
convergence floors (`D1_PCTL252_MIN_BARS=270`, `D1_EMA200_MIN_BARS=220`)
make a 2020-11 start structurally unable to warm the pair before the
2021 model range.

The first full-history executions exposed real latent defects behind the
new fail-closed validators, each repaired at its exact owner:

- `build_basic_v1` silently skipped session features without a `ts`
  column, then the mandatory session-volatility-pressure block failed on
  missing `_v1_is_US`. Session evidence is now mandatory with an exact
  time-source requirement; `add_session_features` no longer emits
  synthetic zero sessions.
- The candle-share block masked zero range to NaN; 102 real quiet-tape
  bars produced mid-series gaps. Zero-range bars now yield exact-zero
  body/wick shares; non-degenerate bars are bit-identical.
- Cross-TF momentum required finite H1 ATR everywhere in the legacy
  zero-warmup convention; it now carries exactly one leading NaN prefix
  for the causal trim owner and fails closed on interior gaps.
- The pair Group-A attach ran without `context_m5`, so 60-D1 liquidity
  and trailing-1yr state would rebuild from the trim boundary — the
  exact V11 failure mode. The attach now receives the full native
  prehistory.
- The per-candidate Group-A emission has its own convergence warmup at
  the context boundary (whole-row NaN via `CausalContextWarmupError`);
  the bootstrap now trims exactly the attached columns through the
  shared trim owner before the immutable all-column finiteness gate.

A 135,000-row real-data probe validated the repaired chain end to end,
and the full run published the first immutable pair generation
`077e5419…` from commit `aaeb0f82`: 468,267 canonical rows and 2,326,495
BASE28 rows covering 2019-12-15T23:00 through 2026-07-24T11:55, 67,711
warmup rows trimmed, `rank_fit_fields_absent=true`, full causal timing
contract and both native manifests hash-bound, at the canonical serving
pointer and generation root.

The exit chain now binds one immutable TRAIN-rank identity end to end:
one identity projection and fail-closed validator in the state-contract
owner; `PrebuiltStateLoader.attach_train_rank_reference` derives
`atr_bucket`/`spread_bucket` through the single formula owner and
re-derives them on every async refresh before the atomic swap; the V3
exit dataset producer and `produce-canonical-joint-exit-proof` take
mandatory `--train-rank-reference-npz`/`--train-rank-reference-sha256`
and bind the identity into manifest, producer event and replay-v7
evidence; `load_active_exit_replay` requires the loaded reference,
`load_default` requires an ACTIVE `train_rank_reference` registry entry,
and the per-M1 exit canonical fails `EXIT_TRAIN_RANK_REFERENCE_UNBOUND`
when unbound; the exit trainer copies the block verbatim into bundle
lineage. The registry intentionally has no such entry yet, so live stays
fail-closed.

Operational record: the producer's clean-repository contract correctly
aborted three runs (an in-repo harness worktree surfaced through
`--untracked-files=all`, now excluded via repo-local `.git/info/exclude`;
one commit landed during a running production and tripped
`REPOSITORY_COMMIT_CHANGED_BEFORE_PUBLISH`). Source edits land between
productions, never during. The full test suite is green at `aaeb0f82`.

Decision:

- the 2019 native roots and pair generation `077e5419…` are the only
  current pair inputs; the 2020-11 roots are superseded diagnostics and
  may be deleted only under an explicit cleanup decision;
- fresh XGB/V3/Exit-IQL artifacts remain open, not failed: no XGB
  trainer exists in the repository, and V3 dataset production requires
  accepted Entry prediction evidence;
- the Entry rebuild source cascade still reads the old canonical roots
  carrying the December-2024 defect; rewiring it to the fresh native
  roots supersedes that repair and is the recommended next decision;
- the legacy `gx1-canonical-incremental` daemon still runs pre-removal
  loop code whose output is already blocked by manifest admission;
  stopping it is an operator action
  (`sudo systemctl disable --now gx1-canonical-incremental.service`);
- launch remains `BLOCK`; nothing here is model, edge or launch
  evidence.

## 2026-07-24 — XGB is cut; the Entry cascade accepts native tape

User vedtak (explicit): XGB is cut entirely and will not be rebuilt. No XGB
trainer is added. Entry-side XGB authority was already forbidden; the
retained Exit-side XGB (79-field input, 7-dim bridge inside the V8 173-field
matrix) will not be reproduced in the fresh Exit chain — the
direction-probability evidence role moves to the accepted model-native Entry
bundle's calibrated outputs when the successor Exit IO contract is defined.
The current XGB runtime, contracts and registry entry remain only as the
already-blocked historical serving surface and are deleted together with the
V8 IO contract in the Exit-rebuild wave. "Fresh 79-field XGB
rebuild/rescore" is removed from the open gates. The exit-chain TRAIN-rank
binding stays: it guards the current fail-closed surface, and the successor
Exit IO decision must re-evaluate the bucket need explicitly.

The Entry seq513 chain now accepts a strict native-v3 M5 root as complete
tape provenance. `validate_xau_tape_provenance_v1` dispatches on the actual
manifest identity: `REPAIR_MANIFEST.json` keeps the exact legacy
repair/current-snapshot branches, while a root with `MANIFEST.json` of
schema `xau_canonical_native_source_v3` is validated through the full
canonical descriptor (response evidence, source↔parquet identity, year
hashes) and returns its hash-bound provenance; run identity is bound by the
consuming event and currency is enforced against the exact `time_max_utc`.
The source-cascade audit and the chain driver accept exactly one event-local
tape identity — `m5_tape_native_v3` or the legacy
`m5_tape_repaired_dec2024`; both present fails closed. New lineages produce
a fresh native M5 root event-locally through the last closed bar, which
supersedes both the December-2024 repair and the collector-snapshot step for
Entry rebuilds. Historical events keep their legacy contracts unchanged.

Decision:

- no XGB trainer, rescore or fresh XGB artifact is ever built; the Exit
  successor contract replaces the bridge evidence with accepted Entry
  calibrated outputs;
- next Entry lineages use an event-local native-v3 M5 tape; the old
  canonical roots and the Dec-2024 repair path remain historical evidence
  only;
- launch remains `BLOCK`; this authorizes the next fresh rebuild lineage on
  native tape but proves no dataset, model or edge.

## 2026-07-25 — first native-tape lineage runs end to end; V8 is the honest FLAT-collapse verdict

The XGB-cut vedtak's authorization was executed: V25 and V26 are the first
Entry lineages on an event-local strict native-v3 M5 tape. Seventeen
never-executed post-audit boundaries failed closed on first real contact and
were repaired in their existing owners, each with regression tests:

1. the cascade-audit report block hashed `REPAIR_MANIFEST.json`
   unconditionally; it now binds the actual tape manifest and has a native
   full-run test (`a406cdf7`);
2. the Entry modelrange projection still pinned pre-07-24 widths; it now
   admits the honest 126-column cv3, excludes the nine ctx-adder-owned
   `add_session_features` columns (one-owner rule) and moves the audit
   constants to 131/125/115/194 (`a3ee94ab`);
3. the builder assigned the two session-minutes context fields by pandas
   label alignment against a DatetimeIndex-keyed Series — all-NaN on the
   window-filtered frame; assignment is positional and the nonfinite guard
   now names offending columns (`fb033fbc`); V25's chain RED at this wall is
   immutable failure evidence and its event root is retained;
4. post-rebuild readiness pinned the legacy snapshot tape schema and
   manifest-embedded run id; it now dispatches on the native identity exactly
   like the chain driver and cascade audit (`ba345f09`);
5. a WSL NTP step moved the clock ~25 s backward between stamp generation
   and chain validation (the V15/V18 trap); chain launches now backdate the
   ranking stamp by 60 s;
6. the trainer demands `GX1_V10_MULTI_TF_V2_CACHE_DIR` but no owner in the
   fail-closed environment pipeline supplied it; the launch contract now
   binds `multi_tf_cache_manifest_json` as a common artifact and emits the
   exact cache-directory row (`d84b9728`);
7. the trainer ambient guard rejected that same row; it is a named runtime
   identity beside the dataset rows (`62150a40`);
8. + 10. the trainer and normalization fitter required cache source ==
   `--m5-prebuilt-path`; the cascade-audited cache is built from
   full-history canonical-v3, so both owners now prove the cache against its
   own declared source bytes (`53ae4874`, `746c0ffd`);
9. the trainer's active-target contract still listed the six dip-MFE targets
   as non-negative and rejected exactly the signed evidence the P0 repair
   preserves; only the dip-MAE half is non-negative (`f49876b8`);
11.–13. the 2% TRAIN clip cap rejected 72 genuine sparse-burst/heavy-tail
   fields on first data contact; scales now escalate deterministically to
   the exact order statistic so the cap holds by construction, with the new
   scale sources admitted by the same contract's validator
   (`74898cf7`, `d632b26e`, plus allowlist);
14.–15. specialist-keyed maps arrive alphabetized from sort_keys JSON
   events; the context-routing validator returns canonical registry order
   after proving content identity (`6273daed`);
16.–17. the eval-mode zero-tolerance OOD raise contradicted the declared 2%
   clip design; train and serve now apply the one identical clamp
   (`4dbfbdc0`).

V26 then ran green end to end: fresh native tape (536,086 rows,
2019-01-01→2026-07-24T20:55), full source cascade PASS, chain terminal GREEN
at the smoke gate, splits 369,303/5,904/4,776, post-rebuild readiness READY,
foundation feature/target/specialist PASS with zero failures, smoke
manifest/readiness/trainability READY, immutable V8 recipe PASS (162 env
values, exact V7-era hyperparameters frozen for comparability).

`XAU_SEQ513_SMOKE_20260725_V8` executed six full TRAIN/VAL epochs on the
fully repaired substrate and stopped hard-red with
`TRAIN_FAIL_NO_BEST_STATE`. The signature is total FLAT collapse: VAL
prediction rates 0/0/100% from epoch 3 (epoch 2 briefly mixed direction),
raw accuracy pinned at the 0.3858 FLAT label rate, 58 slice failures,
path-auxiliary conditional AUC at chance with negative incremental lift,
clean-edge~tradable Spearman +0.976 and a starved family×TF gate. No
checkpoint or bundle was written.

Decision:

- V25 (chain RED) and V26/V8 are immutable evidence; V24/V7 remain
  historical. The V26 dataset bytes stay GREEN for the next evidence gate;
  they admit no model or launch.
- The V8 hyperparameters were tuned for raw inputs spanning a 6.4-million×
  scale range. On the normalized substrate they drive the optimizer into the
  all-FLAT corner. A future smoke needs a new immutable recipe decision
  (hyperparameters are recipe values, not acceptance thresholds); rerunning
  the proven-red V8 recipe is forbidden waste.
- No empirical acceptance threshold is changed. Launch remains `BLOCK`.

## 2026-07-25 — V9 reproduces the collapse; the probe isolates the objective stack

V9 changed exactly two recipe values from proven-red V8 — 25,000→50,000
stratified rows and 8→12 epochs — with every other recipe value, objective
weight and acceptance threshold identical. It reproduced the same
class-degenerate failure: epochs 1 and 4 all FLAT, epochs 2/3/5 marginal side
leakage, epoch 6 flipped to heavy SHORT with accuracy falling to 0.3017, 54
slice failures, `best_epoch=-1`, hard-red stop at the same epoch-6 boundary
and `TRAIN_FAIL_NO_BEST_STATE`. More data and longer training do not change
the collapse mode.

A scratchpad probe (zero authority, no event bytes) then isolated where the
problem is not. A plain 256-128 MLP with unweighted cross-entropy, trained on
the current-bar surface only — 513 snapshot plus 142 continuous context, no
seq96 and none of the five MTF branches — reached VAL accuracy 0.4021 against
the 0.3858 majority baseline with all three classes alive (prediction rates
0.153/0.284/0.563, per-class recall 0.173/0.302/0.649) and tradable VAL AUC
0.5833. Inverse-frequency class weighting made it worse (0.3547 and FLAT
recall collapsing to 0.07), which is itself informative about aggressive
rebalancing.

Conclusion: the V26 substrate and the current-bar evidence support
non-degenerate three-class output and carry a real tradability ranking. The
full training objective — direction CE scale 4.0 competing with roughly
fifteen slice/prior/margin/conviction penalties at weights 2.0-12.0, a
transition cost matrix that prices LONG/SHORT→FLAT at 0.45 against
FLAT→LONG/SHORT at 1.60, and prediction-balance class weights 1/1/4 favouring
FLAT — does not converge to it. Two independent recipe points now support
this, plus one contrasting probe.

Decision:

- V8 and V9 are immutable empirical failure evidence. Rerunning proven-red
  recipe values is forbidden waste at roughly 1.5-3.5 GPU-hours per attempt.
- The next attempt requires an explicit user decision, because both candidate
  paths change something the operator owns: either separate the smoke gate
  from the acceptance gate so smoke admits a checkpoint on liveness plus
  non-degenerate class support while every acceptance threshold stays fixed at
  candidate stage, or rebalance the smoke objective weights (recipe values,
  never `ENTRY_CKPT_*`, slice policy or AUC floors) with the probe as the
  documented rationale.
- No empirical acceptance threshold is changed by this entry. Launch remains
  `BLOCK`.

## 2026-07-25 — user vedtak: profile-separated admission and a rebalanced objective

The user chose both paths offered above. Neither weakens an acceptance gate.

**Profile-separated checkpoint admission.** `_checkpoint_admission_ok` is now
the one owner of the decision. `candidate` is byte-for-byte unchanged:
auxiliary head health, active head health and cooperation gate health all
block admission. `smoke` answers the trainability question it is named for and
admits on active-head liveness plus non-degenerate class support
(`direction_class_balance_guard_ok`, itself governed by the unchanged
`ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE=0.05` and
`ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL=0.35`). Auxiliary and cooperation
health are still computed, logged and journaled identically at smoke; they no
longer veto. The purpose is a progress ratchet: smoke runs now produce a
measurable bundle instead of a binary refusal, so each 1.5-3.5 GPU-hour
attempt yields comparable evidence.

A smoke bundle gains no authority from this. The smoke bundle audit, candidate
readiness, selective-edge prediction, replay, serve parity v4, learned sizing,
joint Exit proof, adaptation lifecycle and the transactional launch finalizer
are all unchanged and still require the complete evidence set. Only a
candidate bundle can enter the acceptance chain, and launch still requires the
newest immutable approval bound to the accepted bundle commit.

**Rebalanced training objective.** Five objective weights change in the
canonical recipe owner; the key count stays 162 and the contract hash advances
as designed:

- `ENTRY_DIRECTION_CE_SCALE` 4.00 -> 12.00;
- `ENTRY_PRED_BALANCE_CLASS_WEIGHTS` `1.0,1.0,4.0` -> `1.0,1.0,1.0`;
- `ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT` 8.00 -> 2.00;
- `ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT` 8.00 -> 2.00;
- `ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT` 6.00 -> 2.00.

Rationale from the V8/V9 loss summaries and the probe: direction
cross-entropy contributed roughly 10.4 of 88.9 total training loss, about
12%, while a tower of distributional and conviction penalties at weights
2.0-12.0 dominated. The probe showed that plain unweighted cross-entropy on
the same substrate produces exactly the non-degenerate output the full stack
cannot reach, and that aggressive inverse-frequency rebalancing makes it worse
(FLAT recall 0.07). The single coherent hypothesis is therefore that
discrimination must dominate and the distributional/conviction terms must be
secondary. Direction cross-entropy now outweighs every individual penalty.

Explicitly unchanged: every `ENTRY_CKPT_*` value, every slice policy, minimum
row count, label-rate floor, tolerance and prediction-rate floor, the
specialist gate floor, both hard-red stop settings, and the full transition
cost matrix (`LONG/SHORT->FLAT` 0.45 against `FLAT->LONG/SHORT` 1.60), which
encodes real trading economics and the abstention-quality goal.

Decision:

- adopt both changes for the next smoke lineage (V10) on the unchanged V26
  dataset bytes;
- keep V8 and V9 as immutable evidence; do not rerun their recipe values;
- if V10 still collapses, the next hypothesis is the balanced sampler's
  train/validation prior mismatch (`ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER`),
  which the V8/V9 loss split already suggests: balance penalties were small on
  sampled TRAIN batches and large on the real VAL prior;
- launch remains `BLOCK`; a smoke bundle proves trainability only and never
  edge, promotion or launch.

## 2026-07-26 — V10 isolates the blocker: the ratchet earned its keep

V10 ran the first lineage with both 2026-07-25 vedtak active: profile-separated
smoke admission and the discrimination-dominant objective (direction CE scale
12.00, prediction-balance class weights `1.0,1.0,1.0`, triad/trade/side
conviction weights 2.00). It failed closed with `TRAIN_FAIL_NO_BEST_STATE`
after six epochs and wrote no bundle.

The failure is different from V8 and V9 in a way that matters. V8 collapsed to
FLAT after briefly trying direction; V9 oscillated to heavy SHORT. V10 is
pinned at 100% FLAT from epoch 1 and never moves: VAL prediction rates
0.000/0.000/1.000 in every epoch and accuracy exactly the 0.3858 FLAT label
rate.

The new admission diagnostics did their job and isolated the blocker exactly:

    [ENTRY_CHECKPOINT_ADMISSION_BLOCKED] epoch=1..6 profile=smoke
    aux_head_health_ok=0 active_head_health_ok=1
    cooperation_gate_health_ok=1 class_support_ok=0

`active_head_health_ok=1` throughout and `cooperation_gate_health_ok=1` from
epoch 2 onward. The specialist, timeframe and family-by-timeframe cooperation
gates are healthy on this substrate — that was never true in V5 through V9 and
is genuine new information. The sole blocker is the degenerate public
three-class output.

Two optimization facts point at the next hypothesis:

- train direction cross-entropy did not fall at all across six epochs — 29.40
  at epoch 1 against 30.36 at epoch 6 — even though its scale was tripled. The
  objective rebalance reached the loss and changed nothing about learning;
- total train loss moved only 95.69 to 91.04, about 5%, while VAL loss rose to
  213-247 against 151-159 in V8.

A model that cannot reduce its dominant loss term over six epochs is not
mis-weighted, it is not learning. With total loss magnitude near 91 across
fifteen-plus terms and `grad_clip_norm` at 1.0, every step is clipped to a
small fraction of the raw gradient, and the surviving direction is whichever
term happens to dominate that batch. The leading hypothesis is therefore
optimization throughput — gradient clipping and learning rate against a
large-magnitude multi-term loss — ahead of the previously recorded
balanced-sampler prior mismatch.

Decision:

- V10 is immutable empirical failure evidence; do not rerun its recipe values.
- Both 2026-07-25 vedtak are retained. The ratchet delivered exactly what it
  was adopted for: it converted a binary refusal into a precise blocker
  identification, and it proved the cooperation gates healthy.
- The next attempt tests optimization throughput first, using recipe/CLI
  values only (`grad_clip_norm`, learning rate, and if needed
  `ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER`). No `ENTRY_CKPT_*` value, slice
  policy, floor, gate threshold or cost-matrix entry may move.
- Launch remains `BLOCK`.

## 2026-07-26 — V11 clears gradient clipping; the fusion path is the mechanism

V11 was a deliberately short throughput probe: two epochs,
`grad_clip_norm` 1.0 -> 10.0, learning rate 3e-4 -> 1e-3, everything else
identical to V10. Cost about 40 minutes instead of 3.5 hours.

Result: train direction cross-entropy 29.72 -> 30.19, still flat. VAL
prediction rates 0.000/0.000/1.000. And the decisive detail — `raw_acc`
0.385840 and direction checkpoint score -0.728320 are **bit-identical to
V10** despite triple the learning rate and ten times the clipping headroom.
Gradient clipping is eliminated as the mechanism.

Bit-identical validation metrics under different optimizer settings can only
mean the public direction output does not meaningfully depend on the input. A
scratchpad probe (zero authority) instantiated the real architecture and
measured cross-sample spread through the fusion path:

- inputs vary strongly: cross-sample standard deviation about 1.0 on `seq_x`,
  `snap_x`, `ctx_cont` and all five timeframe tensors;
- at `evidence_fusion_in` the cross-sample standard deviation has fallen to
  0.111 while within-sample spread is 0.883, so the 22-head / 26-group path
  loses roughly an order of magnitude of sample discrimination;
- at the direction logits the cross-sample standard deviation is 0.066 while
  the spread between the three classes inside one row is about 2.9, so the
  sample-dependent component is on the order of two percent of the
  class-offset it must overcome. Every one of the 16 distinct inputs produced
  the same argmax;
- `evidence_fusion_norm` reports a per-sample mean range of 6.8e-08. This is
  exact and shape-independent: `LayerNorm(96)` removes the per-sample mean of
  the evidence vector, so any signal carried by the overall level of evidence
  — all heads leaning the same way — is deleted before the direction
  projection. Only the relative pattern across the 96 channels survives.

Caveat recorded honestly: the probe ran on the small test instantiation of the
architecture with synthetic inputs, so the exact ratios are indicative and not
production magnitudes. The LayerNorm mean-deletion measurement is exact and
holds at any shape.

This explains the whole V5-V11 series. Direction can only reach the logits
through 22 heads that are themselves being trained from scratch; early in
training their outputs are weakly sample-dependent, LayerNorm then discards
the shared level, and the surviving signal is far too small to flip any row's
argmax. The model therefore predicts one class for every row, which class is
set by the prior, and the tiny gradient it receives competes with fifteen
distributional penalties. It is a curriculum and amplitude problem in the
fusion path, not a data, feature, information-ceiling or clipping problem.

Decision:

- V11 is immutable evidence; gradient clipping and learning rate are cleared.
- The next probe raises the amplitude of sample-dependent evidence reaching
  the fusion using recipe/CLI values only: `--specialist-fusion-scale`
  0.25 -> 1.00 and `--multi-tf-scale` 0.50 -> 1.00, two epochs. Both are
  recipe values; no gate, floor, threshold, cost or architecture contract
  moves.
- If amplitude alone does not restore sample dependence, the remaining
  candidates are architectural and require an explicit user decision, because
  they touch the pinned fusion contract: whether `LayerNorm(96)` may keep the
  per-sample level, and whether head pre-training is admissible as a
  curriculum inside one run.
- Launch remains `BLOCK`.

## 2026-07-26 — V13 refutes the zero-init hypothesis on real data

The specialist-branch measurement now runs inside the trainer's own validated
path, so it uses real V26 bytes, real 96x513 dimensions, the real normalization
contract with full lineage and the real specialist routing. V13 was a
deliberately small read-the-meter run: 12,000 rows, three epochs.

Result, and it refutes the hypothesis I held:

    epoch=1 specialist_out_weight_norm=0.000000e+00 nonzero=0/16384
    epoch=2 specialist_out_weight_norm=2.360016e-01 nonzero=16384/16384
    epoch=3 specialist_out_weight_norm=3.277482e-01 nonzero=16384/16384

The branch is closed at initialization exactly as the source and the algebra
say, but it opens by itself within one epoch and keeps growing. Once the weight
is non-zero, gradient reaches all 129 upstream specialist tensors by
construction. The eight-specialist path is therefore **not** disconnected from
the decision during real training. The zero-init is a legitimate residual-gate
pattern that works as intended, and no initialization change is warranted.

Correction to my own instrumentation, recorded honestly: the first version of
this log also printed upstream gradient norms and reported zero. That was a
placement artifact — the log sits at the start of the epoch, after the previous
epoch cleared gradients — not evidence. The line now reports only what is valid
at that point (weight norm, non-zero count, branch-open flag) and states that
upstream gradient flow follows exactly from a non-zero weight.

Two earlier probe conclusions are also downgraded. They ran on the small test
instantiation with synthetic inputs, so their magnitudes prove nothing about
production: the 0.111/0.066 fusion-variance figures and the "roughly two
percent of the class offset" ratio are withdrawn as production claims. What
survives from that work is the LayerNorm mean-deletion property, which is
definitional, and the observation that the architecture can overfit a small
batch, which is a plumbing check only.

V13 also showed real direction output early: `short_to_long_val` was 0.144166
at epoch 1 and 0.094986 at epoch 2 with accuracy 0.3869 and 0.3914, then epoch
3 collapsed back to zero and 0.385840. Combined with V8's brief epoch-2
direction and V9's epoch-6 SHORT flip, the honest picture is that the model
does produce direction early and the objective then drives it into the FLAT
corner and keeps it there.

Decision:

- The zero-init hypothesis is refuted; do not change `specialist_out`
  initialization.
- The collapse is objective and optimization dynamics, not a disconnected
  path. The remaining recorded candidate is the balanced-sampler prior
  mismatch (`ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER`): balance penalties were
  small on sampled TRAIN batches and large on the real VAL prior across every
  run, which is the signature of a decision boundary fitted to a balanced world
  the model never meets at validation.
- Keep the branch-health log. It is cheap, it settled this question in ten
  minutes, and it converts a future recurrence into a one-line read.
- Launch remains `BLOCK`.

## 2026-07-26 — value-origin audit: shadow contract removed, absorbers closed, register opened

Two independent read-only audits swept the active model, feature, dataset and
runtime paths for invented values, fallbacks and silent substitutions. They
returned 45 findings. Per rule 24 the repairs were limited to what is in the
active Entry path and closable without changing a data lineage; everything else
is registered rather than hot-patched. Every finding below was re-verified
first-hand before acting, and that verification changed the picture twice.

### Repaired

**The trainer's shadow contract is deleted.** Own measurement: 160 `_env_str`
call sites carried a literal default, 62 had drifted from the canonical recipe
owner, and 30 of those drifted to zero — including
`ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE`,
`ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL`,
`ENTRY_CKPT_DIRECTION_SLICE_GUARD` and the whole direction-balance,
prior-match and slice-guard weight family. A zero there silently deletes an
entire loss family. `main()`'s 162-key guard meant a canonical train could not
consume the drift, but other importers had no guard at all. The repair is a
deletion: `_env_str(name)` now sources the value from the recipe owner, all 160
literals are gone, drift is impossible by construction, and an unknown key
raises. Two regression tests forbid a second origin returning.

**Two target absorbers now fail closed.** `clamp_min(0.0)` on
`y_long/short_expected_mae_bps` in the trainer, and the `np.maximum` /
`np.minimum` domain clamp in the dataset writer's `_store_prefix`. These are
precisely the mechanism that let V24's signed-target corruption reach training
undetected: a corrupt target was rewritten into a plausible one. Both now raise
with the observed extreme and the declared bound.

**The session ASIA fallback is closed.** `get_session_id_vectorized` ended in
`fillna(0)`, turning any unmapped session into ASIA for three active consumers:
the canonical builder, the Entry context adder and the live context augmenter.
The Entry context contract forbids a fabricated ASIA flag outright. Both the
scalar and vectorized helpers now raise and name the offending sessions.

### Verified and corrected the audit

`_v1_atr_regime_id` really is constant 1.0 for every bar — the legacy chained
indexing at `basic_v1.py:842-844` writes a temporary copy, and the 2026-06-11
fix sits behind `GX1_ATR_REGIME_FIX`, which is not one of the 162 recipe keys
and would therefore be rejected by the trainer's ambient guard. But the audit
did not check the consumer side: the field is already listed in
`ENTRY_DEAD_CONSTANT_COLUMNS` and dropped from the Entry projection, so it does
not reach the Entry model. Severity is dead-code cleanup under rule 10, not
live corruption. The env gate is unreachable and the live cement it preserved
parity with is retired, so the buggy branch and its gate should be deleted at
the next touch of that owner.

### Registered, not repaired

Roughly forty findings sit in the Exit chain — `materialize_build_exit_iql_v2`,
`materialize_run_exit_iql_with_v2_state_and_reward_variants_v1`,
`disk_labeled_dataset`, `exit_transformer_v0`, `v12_paper_runner` and
`v12_exit_iql_live`. Those artifacts are already contractually invalid and must
be rebuilt from scratch on accepted Entry evidence, and runtime already fails
closed before loading them. Repairing code slated for replacement would be
waste. They are recorded here so the Exit-rebuild wave starts from them rather
than rediscovering them:

- a guessed 1.5 bps spread inside the Exit reward target, applied per row and
  for a wholly absent column;
- Exit reward components NaN-filled to 0.0, which is a legitimate learnable
  reward value;
- a missing required state column zero-filled into the model matrix, which also
  defeats the NaN-fraction guard below it;
- TRAIN-median imputation of every continuous Exit feature, in a module whose
  own evidence payload claims "No fabricated values";
- constant features passing normalization via `std or 1.0`;
- forward-horizon Exit targets fabricated as 0.0 where no future exists;
- missing Exit labels defaulting to the negative class while a working
  ignore-index mechanism exists in the same file;
- a bad timestamp routed into TRAIN by a swallowed parse exception;
- a synthetic broker fill price and zeroed broker facts in the paper runner;
- split boundaries and artifact selection supplied as argparse defaults.

Two structural notes worth more than any single item: `v3_m1` received three
fail-closed guards in June that `v2` never received, and four findings are that
one gap. And "a correct guard defeated by an earlier fill" recurs — worth
grepping as its own class rather than reviewing guards in isolation.

Also registered for the next Entry lineage, not hot-patched: `basic_v1`
substitutes invented neutral values during indicator warmup (RSI 50.0, CLV 0.5,
ratio 1.0, regime 1.0). Removing them changes where the causal warmup trim
falls, which is a data-lineage change that must be validated by a rebuild
rather than patched under a running lineage. The same applies to the invented
winsorization ladder and ATR floor on the active V2 MTF surface, which should
come from TRAIN-fitted quantiles the way every other surface already does.

Decision:

- adopt the four repairs above; they are deletions and fail-closed conversions
  with no new constant introduced;
- carry the register into the Exit-rebuild wave and the next Entry lineage;
- rules 2a-2e and rule 24 now govern this class of defect, and the audit itself
  demonstrated why 2d matters: two of the loudest findings were weaker than
  first stated once verified.
- launch remains `BLOCK`.

## 2026-07-26 — V14: the balanced sampler was the collapse mechanism

`ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER` went 1 to 0 as a recipe value, so
training sees the real class prior instead of artificially balanced batches.
V14 was a small read-the-meter run: 12,000 rows, three epochs.

    epoch 1  acc 0.338584  balance_score +0.127033  pred 0.474/0.313/0.213  guard_ok=1  BEST_CHECKPOINT
    epoch 2  acc 0.382114  balance_score +0.251355  pred 0.340/0.405/0.255  guard_ok=1  BEST_CHECKPOINT
    epoch 3  acc 0.348916  balance_score -0.563516  pred 0.153/0.764/0.083  guard_ok=0

These are the first admitted checkpoints since the 2026-07-14 reset. The class
balance guard passed — it had been 0 in every run from V5 through V13 — the
model produced a genuine three-class distribution, and the direction slice score
improved between epochs (-0.731459 to -0.287875). Epoch 3 swung to heavy SHORT,
so instability is not solved; the model is simply no longer frozen.

Five candidate mechanisms were settled by measurement rather than argument:
objective weights (V10), gradient clipping and learning rate (V11), fusion
amplitude (V12), the specialist zero-init (V13, refuted on real data), and the
balanced-sampler prior mismatch (V14, confirmed). The signature had been visible
in every run — balance penalties small on sampled TRAIN batches, large on the
real VAL prior — and it is exactly what a decision boundary fitted to a
balanced world the model never meets at validation looks like.

### The ratchet had a second gate

V14 still refused to write a bundle: `TRAIN_FAIL_DIRECTION_SLICE_GUARD` blocks
bundle publication when the best checkpoint fails the direction slice contract,
and the 2026-07-25 vedtak had only separated checkpoint admission. Without this
repair a full smoke would have hit the same wall after hours of compute, and the
ratchet would never deliver the measurable artifact it was adopted for.

Closed with the same one-owner pattern, with three deliberate limits:

- only the direction-slice gate is profile-separated. The class-balance bundle
  guard stays unconditional, because non-degenerate three-class output is
  precisely what smoke requires, so a smoke best-checkpoint always satisfies it;
- the slice failure evidence file is written for both profiles, so the failure
  stays immutable and visible; only the raise is separated;
- at smoke the bundle is logged explicitly as trainability evidence with no
  edge, promotion or launch authority. Every downstream acceptance contract is
  unchanged and only candidate bundles enter the acceptance chain.

Decision:

- keep the sampler at 0 for the next lineage; V14 is the evidence.
- adopt the bundle-write separation as the completion of the 2026-07-25 vedtak,
  not a new relaxation.
- the next run is a full smoke on the unchanged V26 bytes to produce the first
  measurable bundle for `model-native-smoke-bundle-audit` and an abstention
  reading. Instability across epochs is expected and is the next question.
- launch remains `BLOCK`; no acceptance threshold changed.

## 2026-07-26 — measured: the V26 training rows carry no synthetic substitution

A direct question — is every value real? — was answered by measurement on all
369,303 real TRAIN rows rather than by reading code and assuming. Two claims of
mine were withdrawn in the process, which is the point of rule 2d.

### Withdrawn claim

I reported that `momentum.flow_mtf_bull_confirmation` and
`flow_mtf_bear_confirmation` sharing exactly 37,576 rows at 0.5 proved a shared
"undefined -> 0.5" default path. That inference was wrong. `_sign_agreement`
computes bull and bear as complementary fractions of valid timeframe
components, so `bull + bear = 1` and `bull == 0.5` is mathematically equivalent
to `bear == 0.5`. Identical rows are required by the definition. Those 37,576
rows are genuine even disagreement between timeframes — real evidence.

Measurement confirms the helper is clean in practice as well: `bull + bear` is
exactly 1.0 in **100.0%** of TRAIN rows, and the case I suspected next — the
`max(valid_count, 1.0)` placeholder denominator turning "no valid timeframe"
into a neutral zero — activates in **0 rows**.

### Second suspicion also refuted

`chart.foundation_pullback_depth_norm` sits at exactly 0.5 in 72,881 rows
(19.7%). All 72,881 occur while a real pullback phase is active, and **none**
occur in the 24,931 rows with no phase active. The field has 200,002 unique
values. A mass at 0.5 is the classic 50% retracement, i.e. market behaviour,
not a placeholder.

### Upgraded from suspected to measured

The `basic_v1` warmup neutrals were registered on 2026-07-26 as unfixed with
the assumption that the causal warmup trim removes them before TRAIN. That is
now measured, not assumed: scanning every one of the 513 signal and 142 context
fields for the exact substitution values found **no RSI field at exactly 50.0
and no CLV field at exactly 0.5**. The only exact-50.0 and exact-25.0 masses
sit on minute and bar counters, where those values occur naturally once per
session or once per episode. The substitutions do not reach the training rows.

The `±25.0` default clip touches one field: `chart.m5_ema50_200_spread_atr`
saturates at the bound in 56 of 369,303 rows, with 368,377 unique values. The
age counters reach 96 and 104 with 105,442 values above 25, proving they are
not clipped. Code hygiene, not corruption.

MTF clip-ladder saturations are legitimate: `regime_class_id` and
`ema_stack_aligned_v2` at 0 are declared semantic categoricals, the wick shares
at exactly 0 are the deliberate 2026-07-24 zero-range-bar fix, and
`session_change_flag` at 98.5% zero matches four session changes against 288 M5
bars per day exactly.

Decision:

- the V26 training rows are real market evidence; no synthetic substitution
  reaches them. This is measured on the full population, not sampled.
- remaining items in this class are code hygiene for the next lineage: the
  unreachable `GX1_ATR_REGIME_FIX` gate and its dead constant feature, the
  `±25.0` default clip, and the invented MTF winsorization ladder, which should
  come from TRAIN-fitted quantiles like every other surface.
- the withdrawn claim stands as recorded evidence that identical row counts
  between complementary features prove complementarity, not a shared default.
- launch remains `BLOCK`.

## 2026-07-26 — the liveness gate measured the wrong rows four times

The post-export feature-liveness gate blocked every bundle from the 07-14 reset
onward. Four separate defects were found, all of the same class: the instrument
reported a verdict from a place where that verdict was not valid. None of them
was a fault in the model or the data.

1. **Gradient norms read after the epoch cleared them.** My own
   `[ENTRY_SPECIALIST_BRANCH_HEALTH]` log sat at epoch start and printed
   upstream norms as zero. Corrected to report only fields valid at that point.
2. **MTF sampled from row 0.** Row 0 is indicator warmup, where NaN is the
   correct causal value and no model row ever looks. Every timeframe field
   reported as silently dead. The sample now starts at the cache's own declared
   `causal_warmup_rows`; all five timeframes declare 27, exactly 27 non-finite
   rows exist, and zero after.
3. **The sample index ignored the dataset's row selection.** The index was built
   as `linspace(0, len(_live_ds)-1)` but applied to `_np_seq`/`_np_snap`/
   `_np_ctx_cont`, which are the full parquet arrays. After a stratified
   subsample `len()` is the subsampled count, so the gate read the first
   contiguous rows. Measured: `d1_trend_age_mature_flag_v3` has std 0.4007 over
   TRAIN and 0.4006 over the stratified subsample, but exactly 0.0000 over the
   first 50,000 rows. A field with 74,185 non-zero rows was called dead.
4. **`DEAD_STD` applied to a 1024-row sample.** Measured on the full 369,303-row
   population, all six fields V17 called dead are alive:
   `d1_regime_changed_flag_v3` std 1.53e-02 nunique 2;
   `session_trend_structure_liquidity_long_score` std 2.31e-04 nunique 15,351;
   `session_vol_spread_breakout_readiness` std 5.60e-05 nunique 137,844;
   `eu_structure_breakout_readiness` std 3.44e-05 nunique 60,936.

Defect 4 has two independent causes, so the verdict needs two scales and no new
constant. `DEAD_STD` settles sparse impulses — a flag firing on 0.024% of rows
is far above the bar yet absent from almost every sample, and this exact field
was already documented as alive on a full scan in 2026-07-05.
`LIVE_TAIL_REF_MIN_NUNIQUE`, this owner's existing was-varying convention,
settles richly-varying fields of small magnitude, where failing on std measures
scale rather than liveness. Both halves are load-bearing on the real data: four
fields clear on nunique, two on std.

A sample below `DEAD_STD` is therefore now a suspicion, not a verdict. The gate
escalates that one field to its complete declared population and rules there.
Escalation replaces exemption: no field is excused, each is measured. The
model-native gate still refuses `KNOWN_ALLOWED_DEAD` entirely, and that refusal
is now sustainable because the correct measurement is available instead.

The sequence surface is ruled on the snap population by a proof from source, not
an assumption. The builder cuts both surfaces from one matrix in one column
order (`build_entry_v10_ctx_training_dataset_v3.py:3468-3469`), so per column the
seq population is a union of trailing windows and therefore a superset of snap's.
Non-constant in snap implies non-constant in seq, which is the only direction
needed to clear a sequence flag. Measuring `_np_seq` directly would stride-read
all 72.75 GB of the memmap per field and prove nothing more.

The three `session_regime` readiness scores were traced to their owner
(`entry_session_regime_interactions_v1.py`). They are unfloored multiplicative
AND-chains of five to eight terms each bounded to `[0,1]`, so a near-zero
product is the design, not a computation fault. Sibling fields in the same
68-field family deliberately carry a `0.50 +` additive floor and a `hi=2.0`
ceiling to avoid the collapse. Whether the unfloored substyle earns its place is
a feature-quality question that cannot be answered without a dataset rebuild,
because the values are baked into the parquet.

Decision:

- deadness is a property of the complete declared population; a sample may only
  raise the suspicion. Any future liveness surface must state which rows it
  measured and why that is where the verdict is valid.
- the four repairs are permanent contract behaviour with behavioural regression
  tests through the owner, not raw-source text assertions.
- the readiness-score combination style and the earlier-registered `basic_v1`
  hygiene items belong to the next Entry lineage, not to a hot patch inside a
  built dataset.
- smoke accuracy is trainability evidence only. Direction quality is not
  claimed from a 50,000-row run and remains unproved.
- launch remains `BLOCK`.

## 2026-07-27 — V18 published the first bundle; the ladder's own gate is unreachable

V18 published the first Entry bundle of this lineage under its final name by
atomic no-replace rename, with no staging remnant. Trainability is proven end to
end: strict state load at 513/513/142/5, all 22 heads present, all 8 specialist
encoders constructed, and every one of the 655 normalized fields carrying a named
fitted origin — 458 `iqr`, 61 `iqr_clip_cap_quantile`, 60
`median_positive_abs_deviation`, 57 `binary_identity`, 11 combined and 8
`categorical_embedding_identity`. No field received an invented scale.

That census also closes the `session_regime` readiness question. All three
scores were fitted through the sparse branch, with scales near 1e-05 against raw
population standard deviations of 3.4e-05 to 2.3e-04, so their typical non-zero
magnitudes reach the model at order one and the extreme tail is clipped at
`CLIP_ABS` within the declared 2% rate. `d1_regime_changed_flag_v3` correctly
received `binary_identity` at scale 1.0 and stays exact 0/1 evidence. The
multiplicative AND-chain compresses dynamic range; the fitted normalization
restores it. This is measured on the published artifact, not argued.

The bundle carries no direction content, and its own diagnostic says so. Exactly
one checkpoint was ever admitted — epoch 1, essentially untrained, `dir_acc`
0.3438 against the 0.3858 majority baseline. In every epoch, 17 to 20 of the 20
audited context slices sat below their own slice majority. The run stopped at
epoch 6 of 12 on `ENTRY_DIR_HARD_RED_STOP` with `epochs_since_improve=5`,
`patience=3`, `min_epochs=6`, while train loss was still falling from 101.37 to
93.10 and validation loss from 116.36 to 112.28. The model was still learning
when the slice admission criterion ended the run, on 50,000 of 369,303 rows.

A separate and more serious finding: the candidate ladder is gated behind
thresholds that this project has never approached. `smoke_edge_pockets` in
`entry_foundation_audit_policy_v1.py` sets `min_direction_accuracy` and
`min_balanced_accuracy` to 0.90 and `min_trade_direction_precision` to 0.98.
Despite the key name, these are not pocket-scoped: the smoke bundle audit
evaluates them over the complete split
(`audit_entry_foundation_smoke_bundle_v1.py:1981-1982` scopes only by `split`,
with no confidence selection, and lines 1081-1089 apply the accuracy bars outside
the `support_scope` branch, so even per-context slices must clear 0.90). The
current majority baseline is 0.3858 and the best figure ever measured on this
substrate is a probe MLP at 0.4021, so the gate demands roughly 2.3 times the
highest number the project has recorded. The thresholds were introduced on
2026-07-19, and no document reconciles them with the retirement of the "97%"
goal in favour of honest bps, win-rate, MAE and cost gates.

Decision:

- V18 is trainability evidence only. No direction, calibration or abstention
  quality is claimed from it, and the smoke-bundle audit was deliberately NOT
  run on it: auditing a checkpoint that is below the majority baseline in every
  slice would measure noise.
- the first honest direction measurement comes from V19 on the complete 369,303
  TRAIN rows at the same 12 epochs, which is 14.8 times V18's total learning
  signal. The change set is exactly two items — a new run id and
  `--subsample-rows 0`, the owner's own expression for the full population, with
  the recipe and every threshold untouched. No magnitude was invented.
- the hard-red stop policy stays as bound. If it fires at epoch 6 again, that
  run still carries 7.4 times V18's signal and the stop policy can then be
  judged on evidence rather than anticipation.
- the 0.90/0.98 threshold question is DEFERRED by user vedtak until V19 produces
  a real number, so acceptance criteria are set from measurement rather than
  before it. The thresholds are recorded here as an identified structural
  blocker and are left untouched; the candidate ladder therefore remains closed.
- launch remains `BLOCK`.

## 2026-07-27 — the objective enforced a class prior that has drifted 39 points

The FLAT collapse that has blocked this project since V8 is not an optimization
pathology, a fusion defect, an initialization problem or a sampler artifact. Five
separate investigations chased those. The mechanism is an explicit loss term.

`_direction_global_prior_match_term` penalizes the model whenever its mean
predicted class rates deviate more than `TOLERANCE=0.02` from the current
batch's actual label rates, at `WEIGHT=8.00`. Two sibling terms do the same for
the hierarchy trade and side heads at 4.00 each, so 16.00 of total objective
weight pins the model's output distribution to the TRAIN marginal within two
percentage points.

That marginal is not stationary. Measured over the full V26 tape by quarter, the
FLAT label share runs from 0.390 (2026Q1) to 0.782 (2023Q3) — a 39-point spread
with a clear structural direction, because the label threshold is an absolute
bps number (`flat_threshold_bps`) while gold's volatility has risen. TRAIN
averages 0.612 FLAT; VAL is 0.386 and TEST 0.466. The model is therefore trained,
at weight 16.00, to emit a distribution that is 22 points wrong where it is
scored. The whole direction edge budget is a few points.

The LONG/SHORT balance, by contrast, is stable: the per-quarter L/S ratio stays
inside 0.85 to 1.39. So what drifted is not *which way* but *how often there is
anything there at all*.

V19 measured the trade-off directly before being stopped. Epoch 1 reached
accuracy 0.3962 — the first time in this lineage above the 0.3858 VAL majority —
with 20 of 20 audited slices failing the prediction-rate requirement. By epoch 3
the model had satisfied the prior term, cutting prediction-rate failures to 3,
and accuracy fell to 0.3560 with all 20 slices below their own majority. The
model traded four points of accuracy for prior compliance. That is measured on
real data at full contract dimensions, not argued from source.

Anti-collapse does not depend on these terms. `_direction_min_pred_rate_term`
(weight 12.00) and its slice sibling (weight 8.00) are one-sided: they require
each class to reach `max(0.50 × label_rate, 0.05)` and penalize only
under-prediction. They remove the degenerate-collapse failure mode completely
without pinning anything to a marginal. The two-sided prior-match terms add no
protection the floors do not already give.

Decision:

- `ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT`,
  `ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT` and
  `ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT` are set to 0.00 in the canonical
  recipe owner. This removes a mechanism rather than inventing a magnitude, which
  rule 2b permits explicitly, and rule 24 requires because the floors already
  remove the real failure mode. The recipe still binds exactly 162 keys and every
  one-sided floor, tolerance and acceptance threshold is untouched.
- the current prior belongs at serve time, not in the trunk. The existing
  calibration owner already fits temperature plus a zero-sum per-class bias on a
  declared split, which is exactly a label-shift correction, and its
  `--direction-odds-cap` guard constrains only the LONG/SHORT asymmetry. Held near
  1.0 it permits the FLAT correction while forbidding any manufactured directional
  tilt, satisfying rule 3. Fitting a LONG/SHORT tilt from one month of VAL is
  explicitly rejected: the L/S ratio is stable and recency fits have been refuted.
- this is NOT the refuted recency or more-data hypothesis. Those tested
  conditional ranking skill. The class prior is a different quantity and had
  never been measured until now.
- V19 is immutable diagnostic evidence and was stopped at epoch 3 once it had
  established the accuracy-versus-prior trade-off. Its log is retained. It never
  wrote a bundle.
- V20 reruns the full 369,303-row population with the prior pinning removed and
  everything else identical, so the delta is attributable.
- the absolute-bps label threshold is NOT changed. It is economically correct —
  a move must clear costs to be worth taking — and rule 16 forbids target
  rewrites. The non-stationarity it produces is handled as prior, not by
  redefining the target.
- launch remains `BLOCK`.

## 2026-07-28/29 — the model overfits before it balances

V21C is the first run with every repair in place at once: prior tolerance floored
on batch sampling noise, all five per-timeframe windows declared and verified end
to end, D1 at six weeks instead of eight days, acceptance reduced to the majority
baseline, parallel data loading, and the four liveness-gate defects fixed. It ran
eight epochs on the complete 369,303-row population before being stopped, because
the curve was already unambiguous.

| epoch | train | val | accuracy | short/long | admitted |
|-------|-------|-----|----------|------------|----------|
| 1     | 89.41 | 106.19 | 0.4048 | 0.000 | no - class_support_ok=0 |
| 3     | 83.90 | 115.94 | 0.3438 | 0.212 | yes |
| 5     | 70.44 | 167.91 | 0.3211 | 0.369 | yes |
| 7     | 57.16 | 206.87 | 0.3269 | 0.434 | yes |
| 8     |       | 212.18 | 0.3328 |       | yes |

Train loss falls 36% while validation loss doubles. The model passes from
degenerate straight into overfitting without a usable state in between: epoch 1
generalizes best and beats the 0.3858 majority baseline at 0.4048, but predicts
zero SHORT and is correctly refused; every epoch balanced enough to admit sits
4 to 5 points BELOW the baseline with validation loss rising monotonically.

The checkpoint monitor selects on `dir_acc`, so it admitted epoch 8 at validation
loss 212 over epoch 3 at 116. Switching `GX1_V10_CKPT_MONITOR` to `val_loss`
would not rescue this: the best-validation epoch is the degenerate one.

The gap is now precise and small. Balanced accuracy is 0.3438 against a 0.3858
baseline and against 0.4021 from a plain unweighted-cross-entropy probe on the
same substrate - a probe that had all three classes alive and 0.5833 tradable
AUC. A simple model achieves what the 22-head model cannot, so the substrate is
not the constraint and roughly six points separate the current balanced state
from where the signal demonstrably is.

Everything else has been eliminated by measurement rather than argument: the
prior pinning is gone, four liveness-gate defects were instrument errors, seven
hidden wrapper defaults are closed, no synthetic substitution reaches the TRAIN
rows, and every contracted field is alive. One diagnosis remains.

Decision:

- the cause is capacity against available signal, not data, features or targets.
  `weight_decay` is 1e-05 and `dropout` is 0.05 for a model with 22 heads, eight
  specialist encoders and five timeframe branches over 369,303 rows.
- no regularization magnitude is chosen here. No stronger value is declared
  anywhere in the repository, so picking one would be an invented magnitude
  (rule 2b). The legitimate routes are an explicit user vedtak or a sweep that
  selects on validation loss, which gives the value the origin of a statistic
  fitted on real declared data (rule 2a).
- `dropout` is currently a model-class default with no CLI or recipe input - the
  seventh hidden default of this campaign - and must become declarable before any
  sweep is possible.
- V21C is immutable diagnostic evidence. Its log is retained; it wrote no bundle.
- launch remains `BLOCK`.

## 2026-07-29 — scratch that never cleaned itself

Four orphaned per-run memmap directories held 295 GB between them, one per
interrupted run. `TemporaryDirectory` cleans up through a finalizer that a killed
process never executes, and the trainer had no other cleanup path, so every
interrupted run leaked a 69 GB mirror of the TRAIN parquet.

The scratch name carries the creating PID, so an orphan is provable rather than
guessed. The Dataset now sweeps dead-PID scratch before allocating its own, and
leaves live PIDs untouched so a concurrent run is safe.

Decision:

- released through the cleanup owner with a hash-bound plan, approval and
  execution, never by direct removal (rule 9).
- combined with the V24 split-parquet release earlier in the campaign, 371 GB
  returned while every manifest, audit and terminal event was retained.
