# Pre-training repair and architecture decision — 2026-09-06

## Authority and objective

**Approved implementation items 1/2/3/5, the measured retention-reference
repair, exact active-candidate transfer closure and the bounded source/state
successor route are deployed. Complete canonical CPU regression passes all
4,062 tests, with actual pre-TEST native and downstream TRAIN/VAL owners also
verified. Item 4 remains deferred. Large-training readiness is not established;
no training, cloud purchase, TEST or promotion authority is granted.**
The operator requested the known defects fixed and an independent review of
architecture, all eight families, models, missing essentials and unnecessary
complexity before considering paid compute. The current launch-state
`pretraining_review_hold` overrides historical continuation instructions.

Verified repository: `/home/andre2/src/GX1_ENGINE`, WSL Ubuntu-22.04, user
`andre2`, branch `audit/v9-premiere-20260905`, base commit
`d4d459c13ec1235ed46fbb5bceb7a72f97057368`. Implementation is in this repository,
not the deleted Mac project. Mac SSH commands explicitly enter this WSL path;
opening an app alone is not proof of a connected Linux working directory.

The goal is bounded technical repairs, validated evidence and explicit next
decisions. It is not to declare profitability, replace the research objective
without a successor contract, or make every speculative enhancement mandatory.

## Canonical serialization repair — 2026-09-06

The runner previously selected different heavy-job lock files depending on
`XDG_RUNTIME_DIR`; its nested path proved cgroup limits but not lock ownership.
The existing runner and capped-execution owner now use one protected
`/run/user/<real UID>/gx1-heavy-job.lock`, with no XDG-selected location or `/tmp`
fallback. Existing safe 0644 lock files inside the owned 0700 runtime directory
remain valid; no lock is truncated, unlinked or replaced by this change.

Both new-scope entry and nested dispatch now verify kernel ancestry after the
existing cgroup checks. The outermost actual canonical-runner ancestor must
hold FD9 on that exact inode with a whole-file exclusive FLOCK; descendants
must share one actual scope and UID. PID/start identity, ancestry, scope, lock
path and descriptor evidence are rechecked. The observed FLOCK PID zero is
valid: the holding descriptor, not that PID field or an environment token,
establishes the witness. Nested subprocesses may close their inherited FDs.

A cwd-changed upstream Bash wrapper is not mistaken for a missing canonical
runner. Only a missing relative, differently named operand may be classified
as an unrelated script through its actual FD255: regular, absolute existing
target, matching basename and device/inode, distinct from the canonical runner.
Missing, deleted, aliased-to-canonical or mismatched script evidence rejects.
This bounded handling follows the observed supported Bash behavior; it is not
a general missing-file exception or a new caller-controlled authority.

The helper runs as an isolated stdlib-only file through the canonical Python.
Existing target validation, CUDA guards, capacities, numerical-thread limits
and the three existing CPU/CUDA/cgroup proof functions remain unchanged; the
latter are AST-identical. No new host profile, supervisor or telemetry policy
is introduced. These are entry-boundary observations, not continuous ancestry
monitoring or proof of multi-day host reliability.

Five source/test paths are deployed. Final frozen-source verification passes
**248 tests** in 2.638 seconds; the **482-test canonical integration** passes in
92.945 seconds, both with zero failures/errors/skips. Integration exercises
closed inherited FDs, unchanged/unset/alternate XDG, a real cwd-changing wrapper
and refusal of another top-level dispatch before its stubbed capacity manager.
No second heavy scope or model is launched by those fixtures. Evidence:

- `/var/tmp/gx1-readiness-canonical-lock-final-unit-20260906.xml`, SHA-256
  `511b296d067bada7d8a7452b0ec3f1f9ecceada1cefa66b5481c96f4f638d157`.
- `/var/tmp/gx1-readiness-canonical-lock-integration-20260906.xml`, SHA-256
  `fb42dc6bfddd353a7f92953fac19b0840a2f35612f81fe802a62089cd4b94e38`.

An earlier staging run's sole failure was an omitted extensionless pre-commit
hook in the snapshot, not a production failure. The final 457-file snapshot
includes that exact existing hook; its manifest SHA-256 is
`62cef750a9ee3b73e0ce0234a027badd80d1501705beefeb3dcb1daf6cb46a63`.
The rebuild-script assertion now binds the wrapper's real lock owner rather
than demanding an inlined filename; its existing flock/exit/scope checks stay.
Ruff and shell syntax pass. Independent review's cwd-change issue is closed
by the bounded fix and executed integration, not by suppressing an error.
Complete canonical regression now passes **3,912 tests**, zero failures/errors/
skips, in **1,030.030 seconds**, through the repaired runner at the unchanged
4G/512M cap: `/var/tmp/gx1-readiness-canonical-lock-full-20260906.xml`, SHA-256
`f136d5243ec5d11f68d033a8f73f708be61c8db76d873295897116c38f7d1068`.
Final source/session and kernel-lock identity are recorded in
`/var/tmp/gx1-readiness-canonical-lock-preservation-20260906.json`.
Training and migration remain held; this is not GPU or candidate-quality evidence.

## Actual checkpoint/source disposition — 2026-09-06

**Unchanged resume is now known to be incompatible, not merely untested.**
The existing session owner opened the actual retained checkpoint 152 / 9,664
steps in explicit read-only mode under the unchanged 4G/512M CPU cap. Online,
frozen-target and EMA states each contain 794 tensors, including 36 tensors
under the four removed static Exit registrations: 562,688 elements and
2,250,752 tensor bytes per state. Current model source has none of those four
attributes. Existing strict model/EMA restoration must therefore reject the
unchanged schema; no real model was constructed or loaded to demonstrate this.
The online/target digests equal the earlier read-only measurement. All five
session files and six directory/file stat identities remain unchanged.

Evidence: `/var/tmp/gx1-readiness-checkpoint-schema-compatibility-20260906.json`,
SHA-256 `97b4a8d4061ec0c2ecd0eeaf1b589e6730774710c2bfe6139c38cad78f7d95c3`.
Peak RSS is 786,268 KiB. No forward, gradient, optimizer load/step, RNG restore,
dataset opening, checkpoint write or migration is included.

The actual optimizer contains groups of 748 and ten parameter IDs, with 722
populated states and 36 IDs without state. It has no explicit parameter names.
The equal count of 36 is **not** a name-to-ID mapping and does not authorize
dropping or remapping optimizer entries. Current restore mutates models,
optimizer and RNG; it is not a read-only compatibility API.

Historical source is not missing: all **105** recipe-bound blobs match their
declared sizes and SHA-256 values at the recipe's actual source commit
`7b1ccdaae6a7c5ef8a2d25e6b65f11146929936e`. No checkout, commit, branch or
historical-code import was performed. Evidence:
`/var/tmp/gx1-readiness-historical-source-availability-20260906.json`, SHA-256
`a07996b251ae6d6437dfce37d7e02b0a5a22c9b7338e604a2551806a94409cd5`.
This establishes recoverable source bytes, not reconstructed parameter order
or numerical continuation parity.

The public-integration preservation snapshot has 109 source roles, eleven
changed existing bindings and four additions. Its changes separate as follows:

| Change category | Consequence |
| --- | --- |
| Model scale reuse and retired registrations/metadata | Active fixture forward/gradient checks pass, but state keys and constructor RNG consumption change; real unchanged-trajectory parity is not established. |
| Trainer diagnostic batching and read-only session access | CPU transport and evidence-access changes remain source-bound; they do not create a compatibility waiver. |
| Spread validation and lifecycle file admission | Full real spread bytes remain identical on the measured inputs; complete VAL file re-admission passes. This is not whole-model/state equivalence. |
| Event/retention, capped proof and public/source-control wiring | Authority and provenance changes are mandatory bindings even where they do not change learning tensors. |
| Added native usefulness and selective-evaluator closure | The old recipe's binding set is incomplete for current code; retaining its old digest cannot admit the new producer. |

The existing guard-recovery owner permits only the guard binding to change,
with identical role sets, paths and learning/data settings. It is not a
migration route for this transition. Preserve the checkpoint as a research
reference. A separately bound successor needs an explicit state/numerical
disposition; neither silently filtered keys nor resetting training from zero
is authorized here. Source remains uncommitted and the review hold remains.

### Original ordering and installed serialization owner

The next read-only session measurement preserves all 794 original ordered state
entries for online, target and EMA as lists, plus both original ordered optimizer
ID arrays (748 and ten entries). All three canonical tensor digests match the
earlier schema measurement, and all five session files/six stat identities remain
unchanged. Evidence:
`/var/tmp/gx1-readiness-checkpoint-order-witness-20260906.json`, SHA-256
`f80a02e6cd5259aa4618599f26fda3d9b41e9162885227f806b55d8c40b6b051`.
Peak RSS is 785,092 KiB. No model is constructed or loaded; there is no forward,
gradient, optimizer restore/step, RNG restore, dataset opening or state conversion.

Independent source review also checks the installed Torch `Module`,
`ParameterDict`, `Transformer` and `Optimizer` implementations. Their four exact
file hashes and sizes match installed RECORD entries and are bound in that
report. This proves those installed file identities, not downloaded-wheel
authenticity. Parameter enumeration follows registration order and deduplicates
objects; state serialization additionally includes buffers and permits hooks.
`ParameterDict.update` sorts an ordinary mapping, so `JOINT_TASK_NAMES` tuple
order is not itself the task optimizer order. Optimizer restoration pairs saved
IDs with current group positions, not independently verified parameter names.

The ordered witness therefore resolves a missing observation, **not the complete
name-to-ID mapping**. Buffer classification, aliases, hooks/overrides and the
exact historical construction/group order still need reconciliation before
assigning any names or removing IDs. Matching shapes/counts or the absence of
optimizer moments cannot substitute for that proof. Numerical-continuation and
explicit successor authority remain separate requirements even after mapping.

### Source-derived parameter correspondence — 2026-09-07

The next bounded measurement derives all **758 parameter names / saved IDs**
from the original state order and source-proven persistent-buffer declarations.
The exact historical normalization owner supplies the seven surface names;
constructor AST inspection expands the four per-surface declarations and eight
literal declarations into exactly **36 persistent buffers**. These buffers are
distinct from the 36 retired parameters. Filtering them and separating the
source-owned task parameters reconciles the original **748 + 10** groups.
The current constructor AST equals the historical constructor after removing
exactly the four approved static-Exit registrations, with no other constructor
change. This establishes the source-derived proposed **712 + 10** grouping,
not observed enumeration from a newly constructed current model.

All **722 populated AdamW states** reconcile by ordered name/ID, tensor shape,
dtype and finiteness. Both moments are CPU tensors matching their parameter;
counters are finite float32 CPU scalars with positive bounded integer values.
Every observed counter is 9,664. The 36 names under the removed registrations
map exactly to the IDs without populated optimizer state; this is no longer a
guess based solely on equal counts. The retained `exit_side_embedding.weight`
maps to saved ID 603 and has populated state at step 9,664. Never remove a
contiguous ID range or all entries based on their numeric position.

Evidence: `/var/tmp/gx1-readiness-checkpoint-parameter-map-reviewed-20260907.json`,
SHA-256 `1a0ab73e9d5035f93140e58930d8349eb9d5558ef52b2d841576a2c163e851f8`.
It binds the full correspondence, original and retained-subset online/target/EMA
digests and all named optimizer moment/counter tensor bytes. Peak RSS is
796,280 KiB under the unchanged capped/locked CPU audit. The earlier measurement
overgeneralized counter dtype/device checks in its scope wording; this reviewed
successor checks both explicitly. Exact correspondence and tensor digests agree
between both immutable reports; neither report nor checkpoint is overwritten.

Independent review of the exact historical model/trainer finds fresh module
registration, no project serialization/enumeration overrides or hook installs,
no post-construction registration/group-membership edits and direct unsorted
state capture. EMA value swaps use strict loading, and target models are separate
objects rather than additional children of the online model. Together with the
installed runtime source evidence, this supports correspondence within the
declared producer/runtime contract. It is not proof against unrecorded historical
external monkeypatching, nor an actual current-model enumeration/restore test.

The checkpoint, source and all learning-state bytes remain untouched. No named
state map is applied, no model is constructed/loaded, and no forward, gradient,
optimizer step, numerical-continuation comparison or migration runs. Full
source-identity checking also confirms 12 changed existing bindings plus four
new roles: the existing exact validator rejects the old recipe's role set.
Physical inode/mtime fields remain part of that contract; retaining a path/hash
alone cannot authorize a different checkout/host. A fresh source-bound successor
and an explicit numerical/state disposition remain necessary before execution.

### Exact normalization metadata recovered without refitting — 2026-09-07

The private session binds its input-normalization hash but does not persist the
full JSON contract. That metadata is nevertheless available in the exact smoke
bundle referenced by the retained candidate gate and hash-bound post-run audit.
The existing signal and normalization owners validate its ordered fields,
statistics, categorical domains, aliases, surface hashes and complete contract.
Metadata and lock contain identical normalization and fit-population proof;
the contract hash exactly equals the retained candidate session's
`efa41d82c7ba9464b9df3dbb51382979f7cb3c86d5a646057b88afb37901313d`.
TRAIN parquet/manifest, M5 source, MTF cache and timeframe-length bindings match
the candidate recipe, as does its TRAIN sequence-reconstruction audit reference.
The persisted lineage records zero VAL/TEST fit rows. The fit-proof hash is
recomputed, but the physical fit population is not reread or refitted.

Evidence: `/var/tmp/gx1-readiness-normalization-metadata-witness-20260907.json`,
SHA-256 `33d8962417ef2f738df187c6f13fa42e39c747c82189ab04227704e542882af8`.
Peak RSS is 547,752 KiB under the unchanged capped/locked CPU audit. All session
file hashes and stat identities are preserved. No dataset/cache rebuild,
checkpoint deserialization, model construction/loading or optimizer restoration
runs. This removes the need to invent normalization metadata or refit TRAIN
merely to obtain it for a later compatibility investigation. It does not change
the trainer's startup behavior, admit the smoke bundle under current source,
prove numerical continuation or authorize a source/state successor.

### Current model structure measured without loading weights — 2026-09-07

A fresh CPU instance now executes the exact current canonical constructor AST,
not a reduced-shape fixture or private test sentinel. Inputs come from the
candidate recipe, its specialist audit through the existing trainer owner,
code-owned MTF routing and the hash-matching normalization metadata above.
The trainer's three exact parameter-grouping assignments also execute without
constructing an optimizer. Actual enumeration contains **722 parameters** in
**712 + 10** groups and **758 persistent state entries**, including 36 buffers.
Names, order, shapes, dtypes and group membership match the previously derived
retained checkpoint schema for online, frozen target and EMA. Parameter
registration order and optimizer group order are compared separately.

Evidence: `/var/tmp/gx1-readiness-current-model-structure-20260907.json`, SHA-256
`b53677d914e58a120e32f2027bb8501f5c05d22d30baaa314b5dcf173b06dabe`.
Peak RSS is 647,416 KiB under the unchanged capped/locked CPU audit. These are
fresh initialization tensors, not restored candidate weights; values and
numerical continuation are not compared. No checkpoint deserialization, forward,
gradient, optimizer creation/restoration, TRAIN refit, dataset rebuild or state
conversion runs. Checkpoint hashes/stat identities remain unchanged. This closes
the actual-current-enumeration gap, not the strict state/source transition or
large-training readiness gap.

## Measured retention-reference repair — 2026-09-07

The first-hop scan of 31 explicitly selected, hash-verified JSON inputs exposed
real type mismatches: descriptive target/head/fusion text was treated as a
filesystem path; native producer repository labels were resolved relative to
the data bundle rather than their declared snapshots; the native API endpoint
was treated as an absolute filesystem dependency. The original diagnostic is
`/var/tmp/gx1-readiness-transfer-reference-frontier-20260907.json`, SHA-256
`945d3a4e71c21367587742185b5a302411d74ee6c38777aac3130f9fd559fb5e`.

Existing owners now expose the exact descriptive metadata and validate the
complete native producer inventory before retention adapts those known types.
Snapshot-relative paths retain their declared SHA/size bindings; all other
native dependencies remain traversable. Unknown, changed, extra-field,
list-wrapped and sealed-TEST lookalikes are not exempt. No metadata bytes,
features, outputs or source artifacts are rewritten. The complete native owner
retains its independent snapshot byte/hash, symlink, layout and data checks;
retention itself still protects binary references without hashing their bytes.

The reviewed immutable source snapshot passes Ruff and **289 synthetic tests**
in **16.377 seconds**, zero failures/errors/skips:
`/var/tmp/gx1-readiness-retention-reference-reviewed-staging-20260907.xml`,
SHA-256 `871dd1ffe5ca7363b876dafff2f851eb7b976114b0da606383b447c5a5ec9b7b`.
Independent review's list-wrapped exemption defect was fixed before this run.
The actual read-only post-deployment scan applies five exact semantic adapters
and both native adapters. Both complete M1/M5 native owner descriptors remain
identical to the previous real-data evidence: 2,628,372 and 531,190 closed rows,
in 86.366 and 17.624 seconds respectively, at 892,196 KiB peak process RSS.
Evidence: `/var/tmp/gx1-readiness-retention-reference-real-20260907.json`,
SHA-256 `1bf165f8c962c1943115f7797903edb3e97326a824c87f5dc55cc0398781cf82`.

All 1,185 selected input stat identities and all five session file hashes remain
unchanged. This was not yet complete transfer or retention closure: 59 unresolved
reference mentions remained unopened and 113 references outside the data root
were not followed. At that checkpoint, the 109-role current source closure
differed from the recipe in 15 existing byte bindings plus four added roles.
The actual public source preparation owner rejects the dirty worktree. Complete
canonical CPU regression
now passes **4,034 tests** in **1,082.165 seconds**, zero failures/errors/skips,
under the unchanged 4G/512M audit cap:
`/var/tmp/gx1-readiness-retention-reference-canonical-full-20260907.xml`,
SHA-256 `f830e9f72448cc7ea8a707a2fc0eaf7e9f3cf7559fb6e42b964f528269450bb4`.
Case identities include every previous 3,912-case baseline test and all 289
focused cases. No cleanup, model restoration, training, TEST, migration or
upload is authorized. The broad opaque-root protection below remains intact.

## Active candidate transfer closure and state successor — 2026-09-07

The ranking-target retention adapter no longer imports a materializer script.
Its exact semantic mapping is owned by
`entry_causal_m1_target_policy_v1.py`; the materializer retains its public
constant as a projection of that contract. The resulting training source
closure is restored from the accidental 116 roles to exactly **109**. Against
the historical 105-role recipe it now has **17 changed existing roles and four
additions**, with no removed role. The exact delta is locked into the new
source-state successor route; an extra, missing or path-changed role rejects.

Existing public owners now validate the remaining active-candidate dependencies
instead of treating every JSON string as a generic path. Direct pre-TEST M1/M5
source manifests, source-cascade proof, M1/M5 enriched sidecars, cross-surface
overlap, pre-TEST guard and TRAIN/VAL unified Exit lifecycle all pass on the
actual bound artifacts. A new read-only Group-A completion validator checks
canonical paths, manifest/completion identity, exact directory contents and
every NPZ chunk's hash, size, metadata, bounds, dtype and time grid. The actual
TRAIN checkpoint passes with 114 chunks / 464,244 source rows; VAL passes with
115 chunks / 470,269 source rows. Evidence:
`/var/tmp/gx1-readiness-group-a-real-owner-20260907.json`, SHA-256
`f6231a38f9f87f2288065681b847c9d8f83ccf83ba6143dbe2f0064bacd396a0`.

The deduplicated transfer closure contains **1,434 regular files** and
**12,146,382,873 bytes**. It includes selected TRAIN/VAL inputs, both complete
native roots, 229 Group-A chunks and their four manifests/completions, downstream
owner proofs, audit sidecars and all five private session files. Previously
hashed large files retain their complete stat identity; newly resolved leaves
were stably SHA-256 hashed in this owner run. TEST was not resolved, statted or
opened. Evidence:
`/var/tmp/gx1-readiness-active-candidate-transfer-closure-20260907.json`,
SHA-256 `75b6dc12e8ed9ab0ff4f66ea882d4c4019ba514609cd7590f6ce31b4a8e7e293`.
This closes the active candidate's non-TEST transfer inventory, not the broad
historical V46 cleanup root; that root remains opaque and deletion stays blocked.

`verify_candidate_checkpoint_resume_v1.py` now has a separate
`--prepare-source-state-successor` mode. It does not weaken ordinary strict
restore or guard-only recovery. The transition admits unchanged learning/data
settings with a new run/output/source identity and exactly the measured source
delta. It removes only the 36 named stateless static-Exit tensors/IDs, preserves
`exit_side_embedding.weight` as optimizer ID 603 -> 579, remaps all 722 populated
AdamW states into 712 + 10 groups, and carries target, EMA, scheduler, RNG,
epoch order, progress and pointer into a new no-replace session. Structural,
corruption and cloned AdamW one-step tests pass; the combined checkpoint/session
regression passes **55 tests** with Ruff clean:
`/var/tmp/gx1-readiness-source-state-successor-exact-delta-20260907.xml`,
SHA-256 `41301e7a9ce298100f050e67083735435435a12db4d0e1d2d6026703e68ae62d`.

No actual checkpoint migration was executed. The old recipe correctly fails
current source preparation because the worktree is dirty. Remaining required
steps are: review/commit this source, materialize a new source-bound candidate
recipe/run, execute the CPU-only successor into a fresh session, prove actual
next-batch forward/loss/mask/target/gradient/AdamW/EMA equivalence, then rehash
the 1,434-file inventory on the selected destination and qualify that host.
Neither the structural route nor this transfer inventory grants launch authority.

The final executable source passes the complete canonical CPU suite:
**4,062 tests**, zero failures/errors/skips, in **1,074.934 seconds**, under the
unchanged 4G/512M cap. Evidence:
`/var/tmp/gx1-readiness-final-canonical-full-20260907.xml`, SHA-256
`3f2ce3e974fe66364fa1cadd127e6b9f366c78945ef92840d85b9d1099df1aa5`.
Subsequent changes to this document and the linked review are documentation-only
and receive the focused handover/Markdown regression below.

## Historical root authority clarification — 2026-09-06

The broad historical V46 root still has none of the retention owner's six
direct manifest names. Two explicitly bound reports were read through that
existing owner and their hashes match: the 41,566,606-byte rebuild terminal and
15,746-byte post-rebuild report. Their schemas identify a completed historical
rebuild and its declared TRAIN/VAL bindings, not exhaustive reachability of
all later artifacts below that root. Its launch-state status remains
`AUDITED_REPORT_ONLY_PRODUCTION_ECONOMICS_BLOCKED`.

Evidence: `/var/tmp/gx1-readiness-v46-root-authority-20260906.json`, SHA-256
`209ba59d6471aeb5fd573f02ac06cd043a527a6c93c93427ff07b303b32eb1d9`.
The initial conservative 4-MiB read limit rejected before decoding the larger
report; the successful inspection uses the owner's existing 128-MiB JSON limit
without changing it, within the unchanged 4-GiB process cap (639,796 KiB peak).
No recursive listing, binary payload, physical TEST path or cleanup is opened.
Do not manufacture a generic root manifest, ignore `root_dir` or rebuild the
corrected candidate dataset merely to make global retention green. The broad
root remains protected/opaque; future transfer must declare the exact current
non-TEST dependency closure, not upload this historical tree indiscriminately.

## Explicit non-TEST transfer inputs — 2026-09-06

A read-only, role-bound inventory now verifies **59 distinct files**, totaling
**10,515,905,851 bytes (9.794 GiB)**. It combines the retained recipe's 15
artifact roots, the earlier VAL reader's 29 direct bindings, both TRAIN/VAL
lifecycle files, the exact cache arrays and distinct cache M5 source, six-clock
squeeze parameter/fit-source bindings, TRAIN-fitted sizing ECDF, signal/ranker
and explicitly referenced quote/native metadata. All 59 current file hashes
match their declarations; duplicate roles agree on one path/hash identity.
At that measurement, all earlier VAL file stat identities and the six checkpoint/
session stat identities remain unchanged; all 65 changed-source/document bytes
still match the canonical-lock preservation snapshot. The following documentation
update does not rewrite that frozen snapshot.

Evidence: `/var/tmp/gx1-readiness-transfer-input-inventory-20260906.json`, SHA-256
`2d41c182a9002cb5f4bc719d2355bfa41feb5feb21fae63da4846da6a3432bcf`.
The instrument reuses the bounded metadata owner and the actual canonical
capped/lock proofs; peak RSS is 625,240 KiB under the unchanged 4G/512M cap.
Payloads are streamed for byte hashes, not decoded as dataframes, feature
arrays or models. Neither checkpoint deserialization nor TEST access occurs.
No data, training state, source, recipe, gate or destination path is rewritten.

This is an **explicit input subset, not complete transitive transfer authority**.
Native source chunks and deeper generation/evidence/event-history dependencies
still require their existing owners. The broad historical V46 root remains
opaque/protected; this report cannot authorize cleanup, indiscriminate upload,
path relocation or reusing an inode-bound session on another host.

The largest selected input is the 7,361,839,090-byte TRAIN parquet, followed by
the 1,462,639,286-byte M1 feature surface. These measurements describe stored
bytes, not resident tensors, GPU memory or training speed. Preserve the existing
compact lifecycle/cache representations; do not materialize a population-wide
window tape or rebuild this dataset merely to prepare later transfer. The final
transfer set must add proven ancestry, source/runtime and the explicitly chosen
checkpoint transition rather than treating these 59 files as sufficient.

The subsequent exact native-source check closes the two selected pre-TEST M1/M5
bundle components through `canonical_xau_source_descriptor_v1`, not through a
replacement validator. It fully checks 913 M1 and 183 M5 response chunks, their
14 producer-source snapshots, 16 yearly parquet partitions and two manifests.
The existing owner rederives 2,628,372 M1 and 531,190 M5 closed rows, validates
chronology and exact filesystem surfaces, and agrees with the bound manifests.
Both are schema-v3 bootstrap bundles with no successor-parent chain. Every
declared request ends no later than the sealed 2026-07-01 UTC TEST boundary;
no TEST root or row is consumed. These upstream native row counts are not the
later feature-surface or candidate population counts.

Evidence: `/var/tmp/gx1-readiness-transfer-native-inventory-20260906.json`, SHA-256
`301aee4b15c981b51e6c1ddd81482be5ff5c29e44dd6cd2dcd2ecd8cce28555d`.
The two owner calls take 81.997 and 16.721 seconds, respectively; peak RSS is
882,984 KiB under the same 4G/512M cap. The 1,128 native files union with the
59-file input inventory produces **1,185 distinct files / 10,795,866,746 bytes
(10.054 GiB)**, with two shared manifest paths counted once. All measured input,
source and checkpoint/session stat identities remain unchanged.

This is complete evidence for those two native bundles, not every downstream
generation/source/report/event-history dependency. No broad retention closure,
data relocation, upload, source/session migration or destination qualification
is inferred. Those remaining references still need their existing owners.

## Public CPU usefulness integration — 2026-09-06

`scripts/entry_next_edge_control.sh model-native-feature-usefulness` now invokes
the existing audit module through the existing audit/4G/512M runner. It requires
bare `--execute`, exact `--device cpu`, the selected bundle/session/recipe hashes,
explicit batch/retained-array/episode/forward-call budgets and `--out-json`.
The output parent must already exist, be empty and lie outside the source tree,
bundle and candidate session. Relative paths, ancestor symlinks, duplicate or
missing flags, CUDA/auto, TEST and partial-layout/sampling overrides reject.
The audit module's separate `--validate-json` mode is preserved. No CUDA
producer whitelist, training launcher, watchdog limit or feature family changes.

The existing capped-execution owner now verifies actual cgroup max/high/swap/
task limits against runner declarations, kernel-reported CPU affinity within
0–7 and every one-thread library marker. The public producer additionally
establishes one-thread deterministic FP32 on CPU, using the recipe seed and
CPU generator only. It does not call the trainer's eight-thread setup or query
CUDA. Numerical drift and a review hold reject before publication. Cgroup and
affinity checks are boundary observations, not a new continuous watchdog.

Existing read-only recipe/source and selected-pair owners remain authoritative.
The audit module is an explicit root of the existing static Python source
closure, which includes its native helper and capped proof. All source/recipe/
hold validators are otherwise AST-identical. The complete committed bundle is
checked through the existing selective-edge integrity owner before loading and
after consumption, including its inventory, commit, lock, metadata and model
bytes. The selected commit must match. VAL owners are rechecked and their
scratch is closed before the existing atomic no-replace report writer runs.
No report is published after a failed owner, cap, hold, cleanup or final-file
check; a competing output is preserved.

**Measured before canonical deployment:** 329 focused command/containment/
publication cases pass in 13.984 seconds, with zero failures/errors/skips:
`/var/tmp/gx1-readiness-native-public-control-20260906.xml`, SHA-256
`a1c105be2a3db312a835b21d56dc8c2f7286f7afd13e4c48807c1a9f509d1c63`.
The subsequent 16-file frozen-source run passes 1,141 of 1,166 cases in
404.421 seconds. Its 25 failures are test-environment boundaries: 18 existing
wrapper fixtures require Git metadata and seven require a `.venv` that the
source-only snapshot intentionally does not contain. Every failure is checked
in the XML; no source guard or test expectation is relaxed. The unchanged
repo-dependent tests are rerun in the canonical WSL checkout, not made green
by fabricating snapshot Git/Python identities. Evidence:
`/var/tmp/gx1-readiness-native-public-integration-20260906.xml`, SHA-256
`17f95f263138b1f3285c3eda3de5f759021d93af6087e2fbf8858c2b241dd314`.

The canonical source/recipe/launcher/control/capped-owner run now passes all
**377 tests**, zero failures/errors/skips, in **49.475 seconds**, including the
25 snapshot-environment failures with no change to their expectations:
`/var/tmp/gx1-readiness-native-public-canonical-launch-20260906.xml`, SHA-256
`8f8d41133a6db01b2c2848b7e12be30db14b2b65db30ae02ee27f957ad598f65`.
Before the later lock repair, the complete canonical suite passed **3,831 tests**, zero failures/errors/
skips, in **1,068.264 seconds**, under the unchanged 4G/512M CPU audit cap:
`/var/tmp/gx1-readiness-native-public-canonical-full-20260906.xml`, SHA-256
`6c5692438f411025d7f490248957e3c7b0dca8181a1d460f784287cacbe65902`.
This run validates the deployed native/public integration, not measured GPU
speedup, actual full-candidate usefulness or continuation of the old checkpoint.

The new proof also passes against the actual capped process: max/high 4 GiB,
swap 512 MiB, 64 tasks, affinity 0–7 and numerical-thread markers one:
`/var/tmp/gx1-readiness-native-public-cpu-proof-20260906.json`, SHA-256
`7d2851a6bfe27878968a99df799f1ab48bddc2a81aca9fc3ec488faeb9cc58d9`.
This process proof does not measure model execution or full-producer headroom.
Ruff has no new findings; the unchanged outcome test retains its pre-existing
F841. Independent review found no additional actionable public-boundary defect.

**Correction to the prior retained-checkpoint concern:** the current checkpoint
policy admits exactly one top-k record. The session owner requires that exact
policy and rejects a second record before deserializing retained checkpoints;
the sole admissible record is the already bound selected checkpoint. A new
synthetic state/pointer tamper test exercises that refusal. No unnecessary
general retained-file map or compatibility path is introduced.

All fifteen related native/lifecycle/cache/control/source-proof changes are now
deployed. Earlier sections below retain their chronological then-staged status;
this section and the current-disposition table own deployment status. Full
candidate model loading, peak RSS and exhaustive audit runtime remain unmeasured.
The completed-session/selected-bundle requirement stays strict and is not a
circular prerequisite for first training. No real selected model, GPU, training,
TEST, dataset rebuild, feature removal, checkpoint migration or cloud spend ran.

## Operator-approved efficiency follow-up — 2026-09-06

The operator explicitly selected optimization items 1, 2, 3 and 5 now. Item 4,
active encoder/fusion simplification, waits for documented feature-effect
evidence. No family, active head, data row, history length, precision policy or
decision authority is removed. This does not authorize training or cloud spend.

Items 1, 2 and 5 are implemented in the existing model/trainer/contract owners;
their focused regressions and the existing Dataset reuse proof are deployed.
The native usefulness orchestration for item 3 is now wired through the public
CPU route above; its earlier staged implementation is described here:

- **1: repeated scale work.** Native episode and incremental Exit evaluate the
  same learned effective scale once per timeframe per forward, rather than
  twice per family. Nothing is cached across optimizer steps or detached from
  the learned scale. The original finite check remains in the shared owner.
- **2: diagnostic transport.** The trainer keeps the exact ordered learned
  loss expression and per-task finite guards. Its three diagnostic scalars per
  active task are copied to CPU together. Seven Exit statistic reductions are
  transported together as int64, without changing tie/action-mask formulas.
  This reduces source-level transfer points, not yet measured training time.
- **3: compact reuse.** The existing Dataset already shares overlapping Entry
  source windows and gives each tensor consumer an independent writable copy.
  New regressions protect that boundary. The staged usefulness consumer now
  applies the complete Entry/Exit layout one batch/paired native episode at a
  time, reuses the frozen baseline targets/tokens, and streams report hashes
  and paired statistics. No persistent dataset or population-wide feature
  window tape is created. An explicit forward-call budget must cover the whole
  layout before baseline inference; a smaller budget rejects, not samples.
- **5: dead registration.** Four uncalled static Exit modules and the obsolete
  path-encoder-layer metadata are removed from source. Active episode/carry
  encoders remain. The strict loader rejects legacy extra state/metadata; it
  does not silently discard old checkpoint keys.

Removing constructors changes fresh-initialization RNG consumption and state
keys. Sharing scale computation can change FP32 gradient accumulation order.
Neither nominally equal seeds nor passing small CPU fixtures establishes
bit-identical candidate continuation. Preserve the existing checkpoint bytes;
source/recipe rebinding and an explicit state-migration decision remain required
before any resume. No migration or checkpoint rewrite is performed here.

The first 15 focused synthetic orchestration/transport tests pass in 14.67 s:
`/var/tmp/gx1-readiness-native-interventions-transport-20260906.xml`.
The combined staged run then passes 650 of 652 tests in 366.39 seconds:
`/var/tmp/gx1-readiness-efficiency-integration-20260906.xml`. Its only two
failures expected the older baseline error prefix, but the new exact clock/index
identity check correctly rejects those mutations before baseline collection.
The ten-case confirmation passes in 1.88 seconds after correcting only those
test expectations: `/var/tmp/gx1-readiness-efficiency-confirm-20260906.xml`.
XML case reconciliation proves 652 unique cases pass across both runs; the
first combined run alone is not all-green. The two 414-file staging manifests
differ only in that test file, not any production source. Ruff and syntax pass.
The model cases prove fewer scale-owner calls, equal full-episode fixture
outputs at identical active weights, finite gradients and close prefix/carry
gradients, retained nonfinite stops and strict rejection of legacy keys.
The gradient comparisons are not bit-identical continuation evidence. No GPU
throughput, full-horizon gradient parity on real data, feature utility or
trading result is claimed.

The subsequent canonical full suite passes **3,200 tests**, zero failures,
errors or skips, in **751.030 seconds** under the 4G/512M CPU audit cap:
`/var/tmp/gx1-readiness-efficiency-canonical-full-20260906.xml`, SHA-256
`ee6cc38d27b877cb4ee6d68ba7b5eb71dccb0c2573c003ebec1242348fac706b`.
This covers deployed model/transport/retired-metadata changes and all existing
canonical tests, not the five staged native usefulness files. Handover still
returns the required review-hold exit 2; no real trainer process is present.

The exhaustive usefulness analysis is diagnostic research, not a new circular
prerequisite that demands a completed trained candidate before first training.
Its strict selected-pair reader intentionally rejects the present incomplete
TRAIN checkpoint. Implementation/integration obligations remain open, but no
actual full 1,419-variant Exit audit is launched by this work.

Independent source inspection found that complete consumed-input stability
must include more than the reader's direct `files` map: cache feature/clock/
scalar arrays, M1/VAL lifecycle backing, squeeze provenance and conditional
native-source ancestry. The subsequent staged repair re-enters the existing
file-admission owners without rebuilding the Dataset or lifecycle corpus; its
842-case combined synthetic verification is recorded below. It does not stat
or open unselected TRAIN/TEST files to expand this closure. Public CPU execution
and final producer publication are now integrated as described above; neither
file-readmission fixtures nor source wiring grant current execution authority.

## Consumed-input stability — pre-deployment evidence

`require_native_val_inputs_unchanged` pins the direct opening hashes, re-enters
the original selected lifecycle admission, uses the cold V4 cache loader for
every feature/clock/scalar NPY and its existing transitive dependencies, and
then rechecks the original hashes and Entry index/clock arrays. It compares
the lifecycle evidence against an opening deep copy. The original cache/VAL
manifest bindings must match the direct file map; a retained trainer cache is
not fresh file-admission evidence. Checks run before baseline allocation and
forwarding, after the baseline and after the complete intervention loop.

The existing `UnifiedExitLifecycleCorpus` owns both initial and repeated file
admission. Small opening manifests are pinned before following their references
again and after admission. The original legacy/pre-TEST source authority,
feature-surface and selected split validators still execute. Rechecking does
not construct another M1 corpus, rebuild temporary feature arrays, reread the
selected lifecycle/Entry rows or inspect unselected split files. Source-ancestry
validators may still allocate their existing temporary DataFrames. This is not
an atomic filesystem snapshot or a zero-IO claim. The original selected-row,
clock, pointer and episode construction body is AST-identical to the deployed
owner, as are all other top-level functions in that module.

Regression covers direct-file changes, immutable cache arrays and inventory,
all six-clock squeeze source/parameter bindings, lifecycle/native-source
ancestry, manifest changes before/during admission and mutation of retained
evidence. Tests prove the same Dataset/corpus/cache backing and M1 scratch are
retained, with no unselected TRAIN/TEST path access. All mutation fixtures use
private source copies; the common squeeze fixture retains its original default
provenance so the existing cross-fit invariance assertion is unchanged.

The first integrated snapshot ran 842 cases: 778 passed, six failed and 58
fixture setups errored. The snapshot omitted the native producer's declared
`gx1_guards/gates.py`; its real source-inventory owner correctly rejected those
fixtures. One existing squeeze fit-invariance case also caught an overbroad
test-fixture provenance change. Both were corrected in the test packaging and
fixtures, without modifying production source or relaxing an assertion.

The complete 416-file frozen snapshot is
`/var/tmp/gx1-native-stability-complete-source-stage-20260906`, with manifest
SHA-256 `2086bc04e26d1c8dae0a718fc3fb2f3eda9cd2566643f124063b97222a6853e3`.
Its affected run passes **819 tests**, zero failures/errors/skips, in
**331.759 seconds**:
`/var/tmp/gx1-readiness-native-stability-complete-source-20260906.xml`, SHA-256
`7b7e5e4c15f7a498f0f817fdd3d75397a7c43b48891608ff0dc8e782fb2254c9`.
The 23 deselected native baseline/intervention arithmetic cases passed in the
preceding run, with their test module and all shared production source bytes
unchanged. XML case reconciliation verifies **842 unique passing cases across
the two runs**, not a single all-green 842-case run. Ruff adds no findings;
one pre-existing F841 remains inside an unchanged outcome-inventory test.
All 416 frozen source hashes still match after testing. The first focused
mutation attempt against a read-only snapshot was blocked before writing;
that original test-support file also still matches its frozen hash.

**Real VAL measurement, 18:01:16 UTC.** The exact staged owners construct all
5,509 Entry rows and 5,632,000 paired Exit states, then re-enter the complete
VAL file-admission chain once. Opening takes about 149.935 seconds; the fresh
file check takes 145.782 seconds; together they take 295.717 seconds. Process
peak RSS is 3,456,404 KiB (3.296 GiB), already reached during opening and not
exceeded by the recheck. This is not proof of zero temporary allocation or
cgroup-wide peak usage. The process completes under the unchanged 4G/512M
audit cap, preserving the same Dataset, corpus and M1 scratch. The full original
indices, source clocks, direct file stats and compact state-population hash
remain exact. Evidence:
`/var/tmp/gx1-readiness-native-stability-owners-real-20260906.json`, SHA-256
`98aeea2ed069ad0a1cee8a88ca6db9e2bcb5ae17738984b1936df8dd8140c9a9`.
No selected model, baseline array population or real forward is loaded/run.
This measures input-owner headroom only, not full-producer headroom or faster
training, and grants no changed-source recipe exception.

The five native usefulness source/test files and four additional existing
lifecycle/cache/fixture files were still staged at this checkpoint. They are
now deployed through the public integration above. The older canonical
3,200-test result does not validate this new producer. No source/recipe waiver, data rebuild,
real model forward, feature removal, checkpoint rewrite or training occurred.

**Public integration decision, independently reviewed and now implemented.** The smallest route
is an explicit CPU-only execution mode in the existing audit module, dispatched
through the existing control script's audit/4G cap. Preserve validation-only
mode; reject implicit devices, TEST, sampling and partial-layout publication.
Prove actual cgroup limits through the existing capped-execution owner before
model loading, not merely environment markers. Establish deterministic FP32
with one thread; the trainer's setup hardcodes eight and cannot be called
unchanged. Bind the audit module as an explicit root of the existing recipe
source closure; a shell string does not bind its Python imports.

The public route now reuses the committed-bundle integrity owner around the
selected pair's six-role map. The suspected extra retained-checkpoint gap was
ruled out by the actual single-record policy and adversarial session test;
no additional inventory mechanism is needed. The existing atomic no-replace report
writer remains the publication owner. No CUDA whitelist or cap expansion is
justified by this CPU file-reader measurement. The strict selected loader needs
a completed candidate session and its selected bundle, so neither the whole
audit nor its resource measurement becomes a circular first-training gate.

## Current disposition

| Area | Status and evidence |
| --- | --- |
| Chronological HAC | Implemented; 26 focused CPU tests pass. Coverage membership is chosen by score, then selected trades are restored to chronological order for HAC. Schema v2 binds the observation order and rejects old VAL references. |
| Real retained smoke VAL | Hash-bound 5,509-row CPU recomputation rejects the old 1%-coverage primary PASS; none of the frozen coverages qualifies. No new inference or TEST. This is the four-step smoke, not the partial full candidate. |
| Seed identity | Implemented; combined seed/handover run passes all 151 tests. Actual recipe bytes, source/data bindings and metadata/lock provenance must agree; only seed and run-specific envelope fields vary. Current metrics are recomputed from bound predictions. |
| Immutable events | Publication-order witnesses, explicit shared scopes, five producer migrations, no-replace sidecars and bounded read-only inventory are implemented. All 236 event/evaluator/seed tests pass together. Wall-clock/mtime order cannot establish authority; no existing event is rewritten. |
| Retention | Transitive metadata and event-history/witness protection are implemented. The measured typed-reference repair passes 289 focused cases and both actual native source owners, and is included in the 4,034-test full suite. Shared-owner budgets and scope rechecks remain enforced. Unresolved references, opaque roots and sealed TEST still block complete closure/cleanup; no deletion is executed. |
| Eight-family connectivity | All 42 model-shape tests pass, including Entry/Exit gradient reachability and episode/incremental carry checks. Mechanical fixtures do not establish usefulness or trading benefit. |
| Usefulness core | Schema v9, native selected-pair/input/baseline/intervention owners and CPU-only public publication are deployed and included in the all-green 3,912-test canonical suite. The earlier 25 snapshot-environment failures also pass canonically without weakened checks. Actual full-candidate usefulness and integrated resource headroom remain unmeasured; the incomplete checkpoint is not admitted. |
| TRAIN diagnostics | Ranker outcome-end containment implemented; all 19 tests pass, including future-suffix invariance and checkpoint-key invalidation. Actual policy fitter and dataset builder already enforce stronger containment; this does not establish model-training leakage. |
| Spread aliases | Invalid raw quotes and contradictory aliases now reject instead of being hidden. All 156 tests pass. A full hash-bound real-data comparison additionally accepts all 470,558 M5 and 2,628,372 M1 source rows and proves identical old/current spread-output bytes. This is not whole-model or resume compatibility. |
| Static preflight | Schema v3 removes accelerator probes, validates architecture through its frozen owner, and explicitly limits row-boundary evidence. All 101 static/retired-state/rebuild-chain tests pass. No outcome-containment or launch authority is inferred. |
| Documentation | Retired ATR/spread rank-NPZ requirements removed. Distinct current normalization, availability diagnostics and sizing ECDF owners retained. Historical VAL-power statement explicitly does not apply to June-2026 VAL. Diagnostic/feature/Exit clocks distinguished. |
| Windows clock | Windows Time changed from stopped/manual to running/automatic; NTP synchronization succeeded. Later status remains running with no leap warning. Long-run stability is not established by a short sample. |
| WSL lifecycle | `general.instanceIdleTimeout=-1` is activated and a 90-second no-client idle probe passes with identical PID-1 start time. Activation occurred only after Windows reported zero running distributions; no pre-existing distribution or user job was stopped. This is a bounded lifetime check, not a multi-day training proof. |
| Heavy-job serialization | One canonical per-UID lock and actual ancestor/descriptor/scope proof are deployed. All 482 integrated cases pass, including real closed-FD, changed-cwd and XDG boundaries. This is a launch-boundary proof, not a new continuous supervisor or generic-host qualification. |
| Full-suite/source readiness | Complete canonical CPU regression passes all 4,034 tests in 1,082.165 seconds, zero failures/errors/skips, including the native/public, lock and retention-reference repairs. Source remains uncommitted: 15 changed existing recipe byte bindings and four added roles. The old recipe/gate/checkpoint identity does not authorize changed source or dropped registrations. |

First-wave full-suite evidence is
`/var/tmp/gx1-readiness-full-final-20260906.xml`, started at 11:41:41 UTC on
2026-09-06. It runs under the existing audit cap (4G memory, 512M swap), CPUs
0–7, one numerical thread and task limit 64. Shell syntax and `git diff --check`
pass. Real-state handover still returns the intended exit 2 / review-hold BLOCK,
with `cuda_authority: NONE`; that is the correct disposition, not a launch PASS.

The first integration attempt stopped on ten immutable-publication fixture
failures; a bounded remainder run exposed one tuple/list return-type regression.
Fixtures now publish legitimate successors for semantic variants and separately
test tampering. Producers preserve their existing in-memory return types while
publishing normalized JSON. No hash/order guard or assertion was weakened to
obtain the final independent full-suite PASS.

## Native usefulness streaming follow-up

The existing `audit_task_feature_usefulness` now routes whole-vector summaries
through the shared bounded streaming owner and evaluates all variants inside
each batch. Component loss/margin vectors are retained for that batch only;
the former population-sized collection of every component is gone. Full input
arrays and baseline/supervision arrays still exist in this API, so this repair
alone does **not** remove the measured multi-terabyte flat-state problem.
Array hashing also uses bounded buffers rather than a complete `.tobytes()`
copy. Canonical vector hashes, order, counts and signs are exact; shifted Chan
moments and compensated sums may round differently from NumPy or another
partition. IID standard error remains descriptive, not dependence-adjusted
evidence. No numerical tolerance, clipping or selection threshold was added
to the production result.

All 218 initial streaming/core tests pass in 22.315 seconds in
`/var/tmp/gx1-readiness-usefulness-streaming-20260906.xml`. Cases include
singleton/uneven batches, empty margin subsets within a batch, canonical hash
parity, invalid/incomplete/overflow streams and bounded scratch/input lifetime.
The subsequent donor-geometry/handover integration passes 288 tests in
`/var/tmp/gx1-readiness-usefulness-native-geometry-20260906.xml`; later report-
binding regressions are included in the final integrated rerun, not this count.

The independent integrated rerun passes all 3,143 tests, zero failures/errors/
skips, in 712.501 seconds, started 13:23:07 UTC:
`/var/tmp/gx1-readiness-streaming-integrated-full-20260906.xml`, SHA-256
`fdf7a7f37f6f6f727028992fd9e62791a9781fe2d23b5ac55cd3a93d6720f243`.
Ruff reports no diagnostics for the four deployed streaming/core Python files.
This result excludes the separately staged compact plan/adapter and the pending
read-only session-opener repair; it is not the completed F2 producer.

**Real native geometry, not guessed pairing.** At 13:13:10 UTC, all 5,500 VAL
entry-episode pairs were inspected through the actual Dataset compact-MTF
history method using the recipe-bound lifecycle pointers, M1 clock and genuine
read-only cache arrays. This covers 11,000 side episodes / 5,632,000 states.
There are 131 combined history-shape/gather-index geometry groups, containing
21–87 entry pairs each, with **zero singleton groups**. A simple adjacent-entry
rotation matches geometry in **zero of 5,500 pairs**. The existing donor owner
therefore now accepts exact block geometry, groups on it together with within-
block positions, and fails if any group cannot be deranged; no row is omitted.
Report v8/donor-plan v2 bind that additional input and reject missing/invalid
bindings and older report claims. Actual end-to-end donor-plan publication is
still pending; this census is not an inference or usefulness report.

Evidence: `/var/tmp/gx1-readiness-native-donor-geometry-20260906.json`.
The exact trainer owner SHA remains
`fb4aa9744518ebda31e633145bd60463d53b532e639f09cc2f1c8d72247dcf7c`.
The census does not read TRAIN/TEST episode populations, fit features, load a
model or invoke CUDA. The actual compact MTF history/gather payload is
492,160–495,680 bytes per entry pair; this excludes local/path/context/token
arrays, model memory and inference activations and is not a full RAM estimate.

**Adapter staging is separate from deployed source.** A second subagent wrote
`gx1/scripts/entry_exit_feature_usefulness_native_v1.py` and its native tests in
the local implementation stage. They are intentionally not installed in the
canonical repository until the real report producer owns their inputs. A
temporary WSL import overlay tests the same adapter against the canonical model
and teacher/Bellman helpers under the 4G CPU cap: 39 tests pass in 22.89 seconds
in `/var/tmp/gx1-readiness-native-adapter-staging-corrected-20260906.xml`.
The first attempt had nine fixture failures because a previously sealed
synthetic donor was sent back into the sealer with its old digest. Corrected
fixtures create a new unsigned synthetic donor before sealing; no episode hash
validator or model check was weakened. The initial failed result is retained.

The adapter forwards complete compact episodes from origin, swaps coupled
local/context fields once, requires exact MTF geometry, preserves fixed
baseline supervision and obtains the opposite-side intervention by swapping
the genuine native output side axis. It does not reset hidden state per row,
reseal counterfactual inputs or replace the frozen teacher. Native parity on
synthetic inputs is not candidate quality, file provenance or production
integration. Remaining F2 work includes the selected online/teacher state pair,
immutable VAL dataset/index/lifecycle and token bindings, compact full-population
donor/side-plan hashing, a native report identity, and the existing public
control surface. The current candidate is still an incomplete TRAIN checkpoint,
not a selected completed-VAL model. Neither the staging adapter nor this census
grants training, resume, TEST or cloud authority.

## Native structural plans and read-only checkpoint access

At 13:57 UTC the deployed `build_native_exit_structure_plans` owner constructed
the actual full VAL donor and opposite-side plans: 5,500 Entry episode pairs,
5,632,000 state rows, 131 geometry groups, no omitted rows and every donor a
distinct geometry-compatible block. Its expanded-row hashes are streamed with
O(Entry-count) retained plan memory. This is now actual owner execution, not
only a feasibility census. The bound census inputs remain unchanged. The run
takes 4.340 seconds and peaks at 852,464 KiB process RSS; those measurements
include this structural census only, not model inference or the full audit.
Evidence: `/var/tmp/gx1-readiness-native-compact-plans-real-20260906.json`.
All 238 usefulness/streaming tests pass after deployment, including exact compact
versus expanded plan-hash parity on mechanical fixtures.

The existing `_CandidateTrainingSession` now has an explicit `read_only=True`
opener. It requires an existing exact session contract, rejects symlink
traversal and forbids both checkpoint writers before argument handling. It can
read a retained session beside an already published bundle without granting
writer/resume authority. Default writer behavior and serialized formats remain
unchanged. An AST comparison with HEAD proves every trainer node outside this
session class is unchanged; this is not a source-identity waiver.

At 14:02 UTC, this deployed owner deserialized the actual retained incomplete
TRAIN checkpoint 152 / 9,664 optimizer steps under the 4G CPU cap. Both writer
methods rejected, every session file's bytes/mode/mtime remained identical,
and no output bundle appeared. Online and frozen-target state digests were
computed without a forward or optimizer step. Evidence:
`/var/tmp/gx1-readiness-read-only-real-checkpoint-20260906.json`.
All 332 affected canonical session/usefulness/handover tests pass in 28.092
seconds, with zero failures/errors/skips, in
`/var/tmp/gx1-readiness-native-reader-canonical-20260906.xml`; Ruff reports no
diagnostics on the session owner/tests and compact-plan owner/tests.

The subsequent full canonical suite passes all 3,172 tests, zero failures,
errors or skips, in 714.228 seconds, started at 14:03:48 UTC:
`/var/tmp/gx1-readiness-native-reader-full-20260906.xml` (SHA-256
`2448f76c16a80ee49c0019c53655f822af1af68c51cde32afb5b35f9e69c6838`).
This covers the deployed session reader and compact structural plans, not the
staged native adapter, selected-pair/VAL-input readers or the subsequent report
normalization-binding repair. Handover still gives the expected review-hold
BLOCK, exit 2, with no CUDA authority.

The staging native adapter now obtains Exit Bellman targets and raw teacher
first-state side values for Entry through one genuine teacher forward. Its
selected-pair loader binds retained best online/teacher states even when the
best epoch precedes the final early-stopped epoch, checks embedded normalization
and preserves strict recipe/source checks. All 87 combined staging tests pass
in 34.939 seconds in `/var/tmp/gx1-readiness-native-reader-staging-20260906.xml`.
File-loader fixtures stub external strict bundle/recipe admission; they do not
prove a real completed candidate exists. The two new adapter/test files remain
undeployed until the existing public producer owns their inputs.

F2 still needs the actual immutable VAL Dataset/index/lifecycle and token
bindings, native report identity, complete streaming orchestration and public
control wiring. The current incomplete checkpoint is not a selected completed
VAL pair. No new dataset, model weights, recipe or checkpoint was produced.
Rehashing the 105 recipe source bindings now finds five mismatches: the previous
four plus the session owner. Preserve the hold; do not relabel the old recipe
or gate to resume modified source.

## Native VAL input-reader evidence

At 14:25 UTC, the actual canonical `EntryV10CtxDataset` and
`UnifiedExitLifecycleCorpus(splits=("val",))` construct the complete existing
VAL population under the 4G audit cap. All 5,509 Entry indices equal the exact
immutable parquet row order and source-reconstruction clock. Native bar opens
run from 2026-05-31 23:55 UTC to 2026-06-30 14:55 UTC; their Entry decision clock
is 2026-06-01 00:00 UTC to 2026-06-30 15:00 UTC. The lifecycle validates
5,632,000 state rows, with 30,023 unique decision times and the unchanged
population hash. All bound file metadata remains unchanged after the read.

Evidence: `/var/tmp/gx1-readiness-native-val-owners-real-20260906.json`, SHA-256
`80814c8f64713b4167527fb6ebd9a7bfe0a5e0125a3b28c53483bfd36c332d08`.
Construction takes 152.437 seconds and peaks at 3,252,120 KiB process RSS.
This measures real data-owner construction only: it does not load a selected
model pair, predict, fit normalization, rebuild a dataset, read TRAIN/TEST
datasets or prove every materialized model input. Existing owners use transient
read buffers and temporary M1 mmaps; no new persistent dataset is produced.
The peak is not a headroom guarantee for the eventual models/baseline buffers;
measure the integrated envelope without raising the 4G cap as a workaround.

The existing audit owner now also has staged `open_native_val_inputs`, binding
the strict selected pair to the exact pre-TEST candidate recipe, VAL manifests,
source reconstruction, cache and lifecycle owners. It checks complete Entry
indices and clocks, selects VAL only, restores the cache environment and
releases only owner-created scratch. The combined session/native-adapter/core/
streaming staging regression passes 359 tests in 55.800 seconds in
`/var/tmp/gx1-readiness-native-val-reader-staging-20260906.xml`. The file-reader
tests explicitly mock external IO/admission and do not turn the independent
real-owner probe into end-to-end selected-model validation. A test-only lambda
was replaced with a named function for Ruff; all 34 directly affected reader/
normalization tests then pass, with 63 deselected, in
`/var/tmp/gx1-readiness-native-val-reader-corrected-20260906.xml`.

A source review found that report identity could advertise a different
normalization contract from its internally consistent Entry/Exit teacher state,
even after both public hashes were recomputed. The staged validator now binds
those identities, and the core validates the teacher/normalization before any
audit prediction. This does not compare online and target model digests as if
they must be equal; those are genuinely separate states.
The final affected core rerun passes all 103 tests in 21.02 seconds in
`/var/tmp/gx1-readiness-native-val-reader-final-corrected-20260906.xml`, including
six pre-prediction rejection cases. The first such run had two fixture setup
failures: the missing-teacher mutation also broke the fixture's normalization
identity constructor before reaching the audit. Keeping the valid identity
fixed while corrupting only the teacher repairs the test; no production guard
or expected error was weakened. Ruff passes on the final staged files.

**Required native identity, not fabricated runtime fills.** The deployed v8
report schema demands runtime Entry-fill snapshots and per-state Exit envelopes.
`build_entry_decision_token_snapshot` requires an asserted LONG/SHORT model
direction and a trade identity; the research population instead contains both
hypothetical sides and also Entry rows whose model might choose FLAT. Never
manufacture claimed model directions or trades to satisfy that schema. The
native producer must explicitly bind its real compact episode-pack population,
online/teacher tensor streams, projection contract, immutable row/decision
clocks, normalization and selected states, without granting runtime-fill
authority. The staged v9 identity repair below addresses the report contract;
actual stream production and public control wiring remain incomplete. All of
this reader/normalization/native-identity work remains staging-only; the deployed
3,172-test result does not cover it.

## Native report identity and selected-state roles — staging only

Source proof identifies two different objects: a runtime snapshot asserts a
model-selected trade/fill, while a research episode contains both hypothetical
sides whether or not Entry chooses FLAT. Staged report v9 therefore removes
the runtime snapshot/envelope-set fields rather than fabricating trades. It
binds separate online/teacher FP32 token-stream hashes, immutable Entry indices,
the compact episode-pack contract/population, eligible Entry indices and native
MTF geometry. No expanded float64 per-state token tape is required by the
report. This is a structural report contract, not proof that actual token or
episode streams have already been produced.

The report cross-checks the frozen teacher against its selected epoch, TRAIN
split, lifecycle lineage and embedded normalization. The exact native state
count, terminal count and donor-block count derive from the episode owner and
must agree with the complete pair population. Rehashing an inconsistent report
does not bypass these comparisons. Online and teacher model states remain
distinct, and a selected best epoch before the final early-stop epoch is valid.
The shared selected-teacher comparison also runs before core audit prediction.

The selected-pair loader now returns explicit path/SHA roles for session
contract, active pointer, active state, selected checkpoint, bundle metadata
and recipe audit. Equal file hashes cannot collapse different roles; the active
slot comes from the validated pointer, not a filename guess. It repeats the
existing clean-checkout/exact-source provenance check after online/teacher
loading, rejecting a source change during load without any compatibility lane.
Normalization is bound to its real embedded bundle metadata, not an invented
separate normalization file. The report validator is metadata-only; it does
not replace the strict file/model loader or post-consumption stability proof.

All **399** affected staging tests pass, zero failures/errors/skips, in
**80.057 seconds** under the unchanged 4 GiB / 512 MiB audit cap. They cover
the report/core, native adapter/selected loader, streaming owners and session
reader together. Ruff passes on all five pending source/test files.
Evidence: `/var/tmp/gx1-readiness-native-report-identity-corrected-20260906.xml`,
SHA-256 `1c8830479f33c1ede429f3a69eb52679646c9e1840ebce699af8cd158b72c9f9`.
The first 123-case run had 122 passes and one test-expectation failure: changing
the metadata digest now fails its explicit artifact binding before reaching
the outer identity hash. That outer-hash regression now mutates the independent
online-state digest; the new metadata-binding regression remains strict.
The failed XML is retained, not relabelled green.

These are mechanical fixtures, including native-model arithmetic on synthetic
inputs, not a current-candidate usefulness result. The report fixture uses the
owner's complete episode state count but short mock histories; it proves report
arithmetic only, not real model-input admission. No GPU, TRAIN/TEST dataset,
candidate forward, training update or new persistent dataset is involved.

The subsequent baseline collector below implements the compact pack/online/
teacher stream identity and fixed supervision. Remaining integration must
stream all planned interventions, complete consumed-input stability coverage,
assemble the native report and wire atomic publication through the existing
public control/guard. Current incomplete TRAIN state cannot satisfy
the selected completed-VAL loader. Full audit cost and integrated memory
headroom remain unmeasured; the earlier data-only RSS is not that evidence.
The five pending native source/test files are not deployed until this public
path is genuinely integrated. Canonical executable source and the prior
3,172-test full-suite result are unchanged by this staging wave.

## Complete native baseline collector — staging only

`collect_native_usefulness_baseline` now walks every immutable Entry row through
the existing Dataset mapping, with exact sequence lengths, field aliases,
categorical domains and FP32 tensors. Separate online and frozen-teacher Entry
forwards produce independent token streams. The existing compact Exit adapter
and Bellman owner run once per model per eligible paired episode; the existing
Entry target owner consumes the teacher's raw first-state side values, not its
Bellman target at state zero. All Entry rows remain present, including model
FLAT outputs and genuine lifecycle-ineligible rows with the owner's FLAT-only
supervision mask. A missing eligible episode or fabricated ineligible episode
rejects rather than reducing the population.

The result retains small read-only Q/target/mask/clock arrays and O(Entry-count)
tokens/seals. Complete feature histories and per-effect loss vectors are not
cached over the population. Original pack seals, Entry-fill bindings, per-row
Entry input digests, geometry-compatible donors and both native token streams
feed the staged report identity. No runtime fill snapshot is fabricated and
no report or persistent dataset is published by this collector.

The shared selected-pair preservation check runs before allocation/forward and
after collection. It validates exact role/file closure, nested metadata, actual
online/teacher tensor digests, unchanged normalization, FP32, eval/frozen state
and clean/exact source provenance, without checkpoint deserialization, model
movement or repair. This is boundary verification, not a lock; the public
producer still must serialize changes. The collector also rehashes the input
reader's explicit file set, which does not yet cover the whole consumed native
M1/cache closure. That missing closure must be completed before publication.

All **472** combined affected staging tests pass, zero failures/errors/skips,
in **310.393 seconds** under the unchanged 4 GiB / 512 MiB CPU audit cap.
Evidence: `/var/tmp/gx1-readiness-native-baseline-corrected-20260906.xml`,
SHA-256 `842d7387cce1ab772c097123a018ccfca8036676a0980f8ed732e0dfbbdc8f1b`.
Ruff and whitespace checks pass. One new integration test runs real native
Entry/Exit/teacher arithmetic on a synthetic full-dimension population with
noncontiguous eligible Entry indices, then exactly compares the stored Exit Q,
Bellman targets and Entry bridge against the existing owners. File/selected-
candidate admission is explicitly mocked: this is not real VAL inference or
evidence of model quality. Other cases reject budget overflow before retained
payload allocation, missing pairs, clock/seal changes, wrong bridge masks and
dtype/sequence/alias mismatches.

The first 17-case focused run had 16 passes and one fixture assertion failure:
changing an arbitrary teacher parameter did not change its projected tokens.
The corrected fixture changes one explicit learned token-projection bias;
exact owner comparisons and distinct-token assertions then pass. No production
check is relaxed, and the failed XML is retained. Different state digests alone
are not evidence of different outputs or feature usefulness.

Source arithmetic through the staged shape owner, using the earlier verified
5,509 Entry rows and 5,500 episode pairs, gives **146,694,486 bytes** of retained
baseline ndarray payload including compact donor indices. Evidence:
`/var/tmp/gx1-readiness-native-baseline-capacity-20260906.json`, SHA-256
`d638a9916b8abd6305f410d35c2f4d7087d8bac6c31067e4727b84a33fc99977`.
No full-size arrays or model were allocated by that arithmetic probe. The
number excludes Dataset/corpus owners, models, Entry batches, native activations,
Python metadata and effect metrics. It is neither integrated peak RSS nor a
claim that the complete audit fits or finishes within the current guard.

The five native source/test files remain staged, not deployed. The current
candidate, TEST, canonical executable source and training hold are untouched;
only the two review documents are updated canonically in this wave. The full
native intervention loop, report assembly, complete consumed-input stability
and public guarded command remain required; passing baseline tests is not F2
completion or large-training authority.

## Follow-up: actual compatibility and remaining integration work

The previous goal turn made verified progress, not merely a status restatement.
On re-entry, the same WSL repository/HEAD, current source hashes, absent trainer
process and active review-hold BLOCK were checked before more work.

**Additional source repairs.** The usefulness writer formerly exposed its final
filename before writing/fsync completed. It now uses the existing shared atomic
text writer, with no-replace publication, complete-file-before-link and directory
fsync. Existing/dangling leaf symlinks reject before resolution; parent-symlink
resolution retains the existing API, not a new general filesystem-hardening
guarantee. A directory-fsync failure can leave a complete published final whose
durability is unproven, never a partially written final. Regression cases cover
failures and competing publication without deleting the competing file.

The core also previously accepted missing timestamps, reordered/duplicate Entry
clocks, and Exit state-index gaps such as 0,7 as if they covered an episode from
its origin. Schema v7 requires finite clocks, strictly increasing Entry times,
contiguous state indices starting at zero for each Exit episode/side, strictly
increasing time within that group, and identical time for opposite-side state
pairs. Overlapping episodes and legitimate opposite-side duplicate clocks remain
allowed. These necessary mechanical conditions do not prove real dataset
membership, native full-episode forwarding or loaded-model/teacher identity.
The corrected small fixtures remain explicitly mechanical, not market evidence.

All 200 affected usefulness/handover/retention tests pass, zero failures/errors/
skips, in `/var/tmp/gx1-readiness-usefulness-clock-final-20260906.xml`. An initial
new test expected a schema-specific error string, while the existing validator
correctly returns its combined policy error; the assertion now names that exact
owner error. No production check was relaxed.

The subsequent independent complete suite passes all 2,953 tests, zero failures,
errors or skips, in 694.498 seconds:
`/var/tmp/gx1-readiness-followup-full-20260906.xml`, started 12:29:37 UTC.
The three follow-up Python files introduce no Ruff diagnostics relative to HEAD;
syntax and diff-whitespace checks pass. Rehashing all 105 recipe source bindings
still finds exactly the same four changed files. Real-state handover retains
the review-hold BLOCK and no CUDA authority; PID-1 start ticks remain 214.

**Real source-compatibility measurement.** At 12:20:07 UTC, the current spread
owners were compared with the exact historical recipe-bound source functions
on the complete native source populations. All 470,558 bound M5 prebuilt rows
and 2,628,372 bound M1 quote rows pass the new validation. Both `spread_bps` and
basic-owner `spread_pct` are byte-identical after the same float64 representation
used by their API. Input files are hash-verified and unchanged across the scan;
their footer maxima prove the files end before the declared July-1 TEST boundary.
Only quote/time columns are evaluated, with no inference, target fitting or
TEST data. Evidence:
`/var/tmp/gx1-readiness-real-spread-compatibility-20260906.json`.
This closes the real-data question for those row-local spread owners. It does
not certify every generated feature, all source migration or checkpoint resume.

**Real retention attempt.** At 12:13:53 UTC, the repaired owner was run read-only
against all three actual root authorities. It returns BLOCK because the active
`V46_20260825T170935Z_CHAIN` directory lacks a direct manifest recognized by the
declared traversal. No data JSON or binary payload was opened, no cleanup plan
was produced and nothing was deleted. Evidence:
`/var/tmp/gx1-readiness-real-retention-20260906.json`.
Therefore real transitive closure remains incomplete, not silently approved by
the synthetic tests. Root rules/data documentation now distinguish that fact
from the implemented traversal and the historical three-path implementation.

**The missing adapter is not just a CLI flag.** The hash-verified current VAL
lifecycle declares 11,000 side-episode rows and 5,632,000 Exit state rows. Source-
owned widths and the bound recipe's sequence lengths imply a 4,651,220,992,000-
byte (4.2303 TiB) lower bound for fully materialized float32 local/MTF row windows
alone. The current audit core would also retain 296 component loss vectors,
requiring 13,336,576,000 bytes before margins/temporary arrays. Its full declared
Exit layout requires 1,419 forward variants. These are actual-manifest/source
arithmetic, not observed RSS, GPU demand, wall time or an invitation to allocate
them. The existing production model uses compact full episodes; this explains
why a naive adapter to the flat diagnostic interface is unsuitable, not the
model's training-memory requirement. Evidence:
`/var/tmp/gx1-readiness-usefulness-resource-shape-20260906.json`.

The next implementation must extend the existing usefulness owner with compact
episode streaming and streamed paired statistics, not expand all rolling windows,
reset recurrent carry for each row, or cache 296 full-population vectors. The
acceptance boundary is:

1. Bind the exact selected online/frozen-teacher states through existing
   checkpoint/bundle owners; never substitute inference-only or online weights
   for the target. The current incomplete epoch is not a selected VAL checkpoint.
2. Reuse the exact VAL Dataset index mapping, native history/cache/gather owners
   and clock semantics. Verify the complete population before publishing; no
   sampling or silent omission of donor blocks with unsupported geometry.
3. Obtain baseline Entry and Exit Bellman targets/masks through their actual
   fitted-Q owners once, and hold them fixed across perturbations. Use genuine
   separate online/teacher Entry tokens. Replay every perturbed Exit episode
   from its origin through the existing causal episode/carry implementation.
4. Retain the declared coupled aliases, categories, family/timeframe effects and
   interactions. If a smaller family-only stage is proposed to reduce diagnostic
   cost, give it an explicit partial scope; never relabel it as the complete
   existing report. Measure cost before authorizing all variants.
5. Prove whole-population/batch-partition consistency, bounded memory, exact
   immutable input identities and no model/teacher mutation, then wire the real
   producer through the existing public control surface. Until then, the
   end-to-end usefulness adapter remains unimplemented and launch stays blocked.

The new review documents are included in the explicit handover fingerprint;
the all-Markdown coverage regression passes. Event compatibility is deliberately
fail-closed: multiple unwitnessed legacy events cannot be ordered by timestamps,
and independent authority roots cannot be silently merged or narrowed. Preserve
historical JSON and its adjacent `.order` witnesses together. Reused old evaluator
roots containing uppercase `ENTRY_CANDIDATE_SELECTIVE_EDGE_SUMMARY_*` collide
with the authoritative prefix grammar; use a fresh explicitly declared output
for a new approved publication, never delete/rename old evidence to obtain PASS.

The retained session is independently hash-checked after repairs: checkpoint
152, epoch 0, TRAIN, optimizer step/next offset 9,664, incomplete. The active
200,910,510-byte state matches pointer SHA
`3e1d1476e55bb83829055528c8c243f9b985e864a4d1a1ba555ea4e5a8e8da12`.
Recipe, session contract and resume pointer remain unchanged. This used streamed
file hashing, not deserialization or GPU. Evidence:
`/var/tmp/gx1-readiness-preserved-checkpoint-20260906.json`.

## What the real HAC regression means

Retained prediction SHA-256:
`5a0dec1133ef11d56fbb86d958c38bdbfb56905c2158b286bed6dd9177146443`.
At 1% coverage, 56 selected rows include 55 trades. Mean advantage is unchanged;
HAC SE rises from 13.9014462920 to 21.9185822268 bps, making the two-SE floor
43.8371644536 bps rather than 27.8028925840. Mean advantage, approximately
29.551248 bps, does not clear the corrected floor. The old null and cohort
membership are not retuned. This demonstrates an evaluation-order defect, not
the final candidate's quality. Technical report completion is not an economic
hypothesis PASS.

Installed-source recomputation evidence:
`/var/tmp/gx1-readiness-real-val-hac-final-20260906.json`, recorded 11:39:30 UTC,
evaluator SHA-256
`488950bf5d9a2b9c11da1f82ff50874fcf621f1a93bc5de8aa9b4b3ea9faab37`.
It also verifies full-coverage score-rank invariance. Historical published
prediction/report files are unchanged; this is a separate CPU regression record.

## Architecture verdict: preserve, measure, then simplify

The design is coherent but complex relative to its current empirical evidence:
one shared encoder, native M5 Entry and M1 Exit, eight genuinely routed families,
learned cross-family interactions, shared Entry-to-Exit representation and ten
learned task influences. Nothing in the review establishes a disconnected
family or a shortage of indicators. Dense experts still compute even when a
gate weight is small; softmax gating is not sparse compute.

Keep the current model and partial checkpoint as a research reference. Do not
buy a larger model, force equal family weights, remove a family because it never
wins top rank, or introduce an external direction router. No model is proven
better for this particular XAUUSD task by its name or architecture alone.

Before model expansion, predeclare a small comparison using existing numerical
dependencies: a regularized linear baseline, histogram gradient boosting and a
small temporal/MLP baseline. Compare matching available inputs, target/economics,
chronological splits and compute budgets. Teacher-fitting baselines are
diagnostics, not automatically an independent end-to-end trading policy.
Do not use unlimited VAL search or open TEST for this design decision.
[The pinned scikit-learn estimator is already available](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).

## Missing essentials before a production-oriented large experiment

1. **Executable reward and terminal specification.** Current research Exit
   marks at the observed M1 close and has a capacity-forced terminal at 512;
   runtime can continue beyond that storage window. Next executable fills,
   commissions/financing, missing quote handling, economic termination and one
   shared capital/exposure ledger are explicit unready contracts. Preserve that
   BLOCK. A corrected economic objective is a separately bound successor, not
   an unchanged resume of current weights/optimizer state.
2. **Meaningful-checkpoint usefulness evidence.** The native adapter and CPU
   public producer are now implemented and tested as recorded above. Actual
   execution still needs a completed selected online/frozen-teacher pair, not
   the current incomplete checkpoint. Full VAL and episode-origin carry remain
   mandatory; target/mask baselines remain fixed across coupled perturbations.
   Integrated headroom, runtime and dependence-aware feature-effect support
   remain unmeasured. This evidence precedes feature removal, not first training.
3. **Qualification support.** June-2026 VAL is not the old 13-month power case.
   Five seeds and untouched TEST remain required. The declared July-1 TEST start
   cannot provide six distinct calendar months as of September 6; the contract's
   minimum is not reduced. Candidate completion, research qualification and
   production admission are different milestones.
4. **Measured performance, not estimates from unstable clocks.** Profile a
   later authorized meaningful window before targeting GPU count or spend.
   Repeated timeframe-scale work and scalar CPU/GPU synchronizations are source
   opportunities, not measured speedups. Preserve FP32 and gradient parity;
   do not turn on TF32/autocast/compile/workers as an unreviewed shortcut.
   [PyTorch explains synchronization costs](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization).

## Build / retain / remove decisions

- **Implemented now:** the exact native usefulness adapter/public CPU route,
  shared histories, scale reuse, batched diagnostics and dead Exit registration
  removal. No additional implementation remains for approved items 1/2/3/5.
- **Separate future research:** a prospective bounded baseline protocol and
  executable economics as a successor design, not a silent patch.
- **Retain:** all eight families, source/recipe/input bindings, native-clock
  contracts, checkpoints, TEST seals, capped execution and watchdog limits.
- **Measure later:** actual throughput and integrated resource use of the
  implemented scale/transport changes. Fewer calls in fixtures are not a GPU
  speedup measurement or proof of bit-identical candidate continuation.
- **Already removed:** retired static Exit registrations and their metadata,
  not active families. The state-key/RNG compatibility boundary remains strict.
- **Unexecuted cleanup candidates:** the uncalled launch transaction owner and
  packages required only by readiness. They are not removed by this wave;
  PyYAML/sklearn remain enforced by readiness. Item 4 stays deferred.
- **Do not add now:** more indicator families, an LLM/PPO/model zoo, tick/order-
  flow claims from tick-count or independent quote-extrema proxies, or a second
  implementation of existing feature/target/normalization owners.

## Host and future transfer boundary

The following is the earlier read-only-session source snapshot, not current
deployment identity. The final native/public preservation record is
`/var/tmp/gx1-readiness-native-public-preservation-20260906.json`; it records
current source closure, all changed paths and exact unchanged checkpoint bytes.
At this earlier checkpoint, rehashing the recipe's 105 bindings found five
changed source files:

- `gx1/contracts/evidence_retention_v1.py`
- `gx1/contracts/immutable_event_authority_v1.py`
- `gx1/features/basic_v1.py`
- `gx1/features/model_native_market_context_v1.py`
- `gx1/models/entry_v10/entry_v10_ctx_train_v3.py`

At that earlier checkpoint all other bound source bytes matched. The trainer
change was confined to the session class; every AST node outside it and the
entire model source matched HEAD. The subsequent efficiency and public-source
changes above supersede that comparison; it cannot authorize current source.
The recipe SHA remains
`d22ece49d589bf4387e2df566bf3a9d802238b20e9815c88c8456de8c8dd3d52`.
This is not a claim that the old recipe accepts the current worktree. A reviewed
source/session compatibility decision and fresh bound evidence must precede
any later resume; never weaken the validator or reuse the guard-only recovery
as an exception for these additional changes. No commit is created in this wave.

Windows `.wslconfig` backup:
`C:\Users\Andre\.wslconfig.gx1-readiness-20260906.bak`.
Original SHA-256:
`86fb1da64eab230e8c269272143e0dc75873c391443d5b087d55320875267548`.
Prepared SHA-256:
`9a94ecbe7afd9fe32f7c28dc78a5455b9c2b67a708529bdaa70deffe006481c3`.
RAM 32GB, swap 4GB, processors 19, VM idle timeout and reclaim settings are
unchanged. `instanceIdleTimeout` and `vmIdleTimeout` are different controls.
The initial probe before VM activation failed. At 10:47:56 UTC, Windows reported
zero running distributions, permitting shutdown of the already empty WSL VM.
After restart the 90-second no-client probe retained PID-1 start ticks 214.
At its end, WSL UTC 10:49:41.387 was between Windows readings 10:49:41.357 and
10:49:41.419. No active distribution or user job was stopped. Long-run stability
and the original historical shutdown initiator remain unproven.
[Microsoft documents startup-time configuration application](https://learn.microsoft.com/en-us/windows/wsl/wsl-config).

A narrowly scoped installed-runtime inventory resolves all 11 requirements to
50 installed distributions with no version mismatch. It records dependency
metadata and installed RECORD hashes, not verified wheel bytes, and imports no
Torch/GPU module. Evidence: `/var/tmp/gx1-readiness-runtime-inventory-20260906.json`.
No package is installed, upgraded or removed by this inventory.

No cloud resource is ordered, data uploaded or generic host profile invented.
Later transfer requires the exact source plus resolved runtime inventory,
transitive TRAIN/VAL source/cache/manifests, and checkpoint online/teacher/
optimizer/EMA/RNG/cursor identities. Path/inode-bound state cannot be copied
and falsely called the same session. Existing guard-only recovery is not a
generic cloud migration owner. TEST payloads, credentials and unrelated home
files do not belong in a convenience upload.

The actual host must supply equivalent enforced memory/task/CPU limits,
trustworthy thermal/power telemetry, a supported persistent execution lifetime
and deterministic numerical validation. The current Windows/3090 hard-coded
profile is not a cloud profile. Hardware-specific limits and a one-GPU pilot
belong to the later selected machine, not guessed specifications today.

## Review coverage and evidence limits

Independent reviews:
- [Architecture, model and simplification](PRETRAIN_ARCHITECTURE_REVIEW_20260906.md).
- [Data, eight families, causality and dependencies](PRETRAIN_DATA_REVIEW_20260906.md).

The source inventory covers 410 Python files structurally; no claim is made
that every line of every file or every Markdown document was manually audited.
Source proof, generated mechanical tests and actual bound-data measurements
are distinguished in both reports. No agent measured current full-candidate
feature utility, same-bundle live parity or profitability. No artifact cleanup,
GPU inference, large training, TEST evaluation, purchase or model replacement
is part of this repair wave.
