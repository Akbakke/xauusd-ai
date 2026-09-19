# Pre-training code review — 2026-09-05

> **Authoritative status — 2026-09-08:** Read
> [`CURRENT_CLOUD_TRAINING_STATUS_20260908.md`](CURRENT_CLOUD_TRAINING_STATUS_20260908.md). The canonical
> handover currently returns `BLOCK`; the review hold is active, no trainer is
> running, and CUDA, TEST, paper/live and spending authority are all `NONE`.
> The cloud-control package is uncommitted and has known failing CPU tests.
> This block supersedes every lower runtime/status statement; lower dated text
> remains architecture, policy or historical evidence only.

## Current operator override — local guarded smoke complete, 2026-09-08

The one authorized source-current 32-row CUDA smoke completed successfully on
the local GX1 RTX 3090. Recipe SHA-256
`21d117236cdb4d81131b5fa0da9319df62c76f366513c04c8dab58e714b3d4e0`
binds source commit `efa99b2b3105d2fb44a041de404d5a66b41158f8` and source
bindings SHA-256
`6691dcd22f7c1de7452515910862213182ebc59e54a44b95586e7a87c53e485d`.
The canonical guard exited with child status 0 after 350 seconds and 286 signed
samples, peaking at 50 C core, 50 C memory, 158.83 W and 9417 MiB VRAM. The
first profiled TRAIN batch took 2.115376 seconds and peaked at 6376 MiB CUDA
memory. The published bundle commit SHA-256 is
`09e6c247a13b52caade71d5bc5e2bdf1af3376bf8fe8e5a0927a73430ff88448`
under `LOCAL_3090_SOURCE_CURRENT_SMOKE_20260908T084056Z`.

The smoke proves source-current local runtime plumbing and guard headroom only;
its 32-row loss/PnL is not quality or edge evidence. `pretraining_review_hold`
is restored. Candidate continuation, another smoke, VAL, TEST, paper/live,
cloud purchase and model/data changes remain blocked pending explicit review.

## Previous operator override — readiness repair, 2026-09-06

Final local disposition, 2026-09-07: the requested source repairs and
multi-agent architecture/data review are complete. The repaired source is
explicitly committed; a fresh source-bound successor recipe and CPU-migrated
session preserve the checkpoint, and production-path next-batch equivalence
passes between historical and successor source. The full CPU suite is green.
Both sessions and TEST remain preserved; no CUDA, training or cloud purchase
was performed by the completion work.

The `pretraining_review_hold` remains active with no activation authority, but
its reason is now external qualification rather than unfinished CPU repair. A
selected destination must receive and rehash the exact non-TEST closure, verify
the complete Git bundle, run a guarded source-current smoke and publish fresh
trainability/readiness plus an exact recipe-bound gate. The immutable external
evidence beneath `SOURCE_STATE_SUCCESSOR_20260907T213749Z`, created after the
final clean commit, owns exact paths and hashes. This document does not grant
launch authority. Preserve the candidate, original recovery evidence and TEST;
do not resume, restart from zero, redesign the active architecture or buy
compute from this review.

The final independent repository-wide lint gate found one real audit defect:
the sequence-source reconstruction audit referenced the prior timestamp before
initialization when a split crossed an Arrow batch boundary. The repair
initializes that boundary state explicitly and a forced two-batch regression
proves the production loop. Twelve additional unused-import/local diagnostics
were removed without changing behavior. Repository-wide Ruff and compileall
are clean; all 4,069 tests pass in 1,075.888 seconds with zero failures, errors
or skips under the canonical 4G/512M CPU audit cap. The immutable JUnit report
is `/var/tmp/gx1-authoritative-lint-repair-full-20260907.xml`, SHA-256
`da6d1aab2d0ad6dc36ec9ac4a1c36733bfc245a8d66c0c44bac6014984c32e14`.

The subsequent operator decision explicitly approves efficiency items 1, 2, 3
and 5: eliminate repeated timeframe-scale computation, batch diagnostic CPU
transfers, retain compact shared data histories, and remove genuinely unused
static Exit registration. Active fusion/encoder simplification (item 4) waits
for feature-effect evidence. Items 1, 2 and 5 and the existing Dataset reuse
regression are implemented canonically after staged CPU verification; the new
native usefulness orchestration is now integrated through the CPU-only public
route described below.
The combined verification covers 652 unique cases across a 650-pass/2-fail
run and a ten-pass confirmation of the corrected error-prefix expectations.
No production check is weakened. The subsequent canonical full suite passes
all 3,200 tests in 751.030 seconds, zero failures/errors/skips, under the CPU
audit cap (`/var/tmp/gx1-readiness-efficiency-canonical-full-20260906.xml`,
SHA-256 `ee6cc38d27b877cb4ee6d68ba7b5eb71dccb0c2573c003ebec1242348fac706b`).
That older complete suite excludes the five then-staged native usefulness files.
No feature family or checkpoint bytes are removed. Constructor RNG/state keys
and source identities change, so neither a same-seed claim nor silent legacy
key dropping may authorize continuation. See the repair record's efficiency
follow-up for exact scope and remaining migration/performance evidence.

The next staged repair closes the VAL-input file-stability gap through the
existing lifecycle and cold V4 cache admission owners, without reconstructing
the Dataset/M1 corpus or opening unselected TRAIN/TEST files. It pins opening
manifests, direct inputs, NPY bytes, squeeze/native-source dependencies and
original clock/index/evidence identities before and after consumption. The
latest affected run passes 819 tests in 331.759 seconds, with 23 unchanged
native baseline/intervention cases retained from the preceding run. XML
reconciliation proves 842 unique passing synthetic cases; it is not one
all-green 842-case run. Initial failures were frozen-snapshot inventory and
private-provenance fixture defects, not waived production checks. Details and
exact hashes are in the repair record's consumed-input stability section.
At that checkpoint all nine related changes were staged. They are now deployed
through the public integration below, but real full-producer resource evidence
remains open. The older canonical 3,200-test result does not cover these changes.
The actual 5,509-row VAL Dataset/corpus plus one full file re-admission completes
in 295.717 seconds at 3.296 GiB process peak RSS under the unchanged 4G cap.
The recheck takes 145.782 seconds without exceeding opening-time peak RSS;
same owner/scratch identities, source clocks and state population are preserved.
No model, baseline prediction population or real forward is included.

The subsequent public integration is now deployed: exact CPU-only dispatch,
actual capped-process/affinity proof, one-thread deterministic FP32 without CUDA
queries, clean exact source closure, full committed-bundle checks and atomic
publication only after input scratch closes. The actual single-record top-k
policy rules out the suspected additional retained-file gap; an adversarial
state/pointer test proves the refusal. No general compatibility map is added.
All 329 focused public-boundary tests pass. The subsequent 1,166-case staged
run has 25 environment-only failures (no snapshot Git or `.venv`); all 377
canonical source/recipe/launcher/control/capped tests then pass, including those
25 unchanged cases. The complete canonical CPU suite now passes all **3,831
tests**, zero failures/errors/skips, in **1,068.264 seconds**, under the unchanged
4G/512M cap: `/var/tmp/gx1-readiness-native-public-canonical-full-20260906.xml`,
SHA-256 `6c5692438f411025d7f490248957e3c7b0dca8181a1d460f784287cacbe65902`.
Approved implementation items 1/2/3/5 are complete; item 4 remains deferred.
Full-candidate loading, exhaustive audit runtime, GPU speedup and integrated
resource headroom remain unmeasured. Old-checkpoint continuation and large
training remain unauthorized; no active feature family is removed.

The subsequent serialization repair is deployed through the existing runner
and capped-execution owner. One canonical per-UID runtime lock replaces ambient
XDG/fallback selection; real kernel ancestry/FD9/FLOCK evidence now precedes
target dispatch in new and nested scopes. Closed child descriptors remain supported. A narrow
actual-FD255 check handles cwd-changed upstream wrappers without accepting
unresolved runner aliases. Existing caps, numerical settings, target validation
and GPU guards are unchanged. All 248 final unit/static cases and 482 canonical
integration cases pass. Complete canonical CPU regression of this additional
source now passes **3,912 tests** in **1,030.030 seconds**, zero failures/errors/
skips: `/var/tmp/gx1-readiness-canonical-lock-full-20260906.xml`, SHA-256
`f136d5243ec5d11f68d033a8f73f708be61c8db76d873295897116c38f7d1068`.
The preceding 3,831-test result remains the older source baseline.
Exact evidence and boundary limitations are in the repair record.

The subsequent actual checkpoint schema audit proves that unchanged resume is
incompatible: online, target and EMA each retain 36 removed static-Exit tensors.
All checkpoint bytes and stat identities are preserved. Its 36 optimizer IDs
without populated state cannot be assumed to identify those tensors; no name
mapping, model loading or migration is performed. All 105 historical source
blobs are nevertheless available exactly at the declared recipe commit.
Guard-only recovery cannot admit the changed role set or learning source.
Exact evidence and the source-change disposition are in the repair record.

The subsequent read-only ordering witness now preserves complete original
online/target/EMA key order and both optimizer ID arrays. Installed Torch
enumeration/serialization source hashes match RECORD. It confirms the positional
optimizer-loading boundary and ordinary-mapping `ParameterDict` sorting, but
does not yet map saved IDs to source-proven parameter names. Buffer/alias/hook
and construction-order reconciliation remains necessary; no actual model or
optimizer is loaded and checkpoint bytes are unchanged.

The subsequent 2026-09-07 source-derived correspondence now reconciles all 758
original parameter names/IDs after identifying the separate 36 persistent
buffers. All 722 populated AdamW states pass moment shape/dtype/finite checks
and float32 CPU counter checks; every counter is 9,664. The 36 retired parameter
names map exactly to stateless IDs, while the retained side embedding at ID 603
has populated state. Current constructor AST equals historical construction
minus the four approved registrations. Independent review finds no conflicting
alias, hook, registration or group rewrite in those exact project owners.
This is conditional source/runtime correspondence, not actual model enumeration,
numerical continuation or migration authority. Full exact source binding still
rejects the old recipe: 12 changed existing roles and four additions. The repair
record binds the reviewed evidence and its explicit limits; all checkpoint bytes
remain preserved.

The exact persisted normalization metadata is now located and validated through
the existing owners: the gate-bound smoke metadata/lock contract hash equals
the retained candidate session hash, with matching recipe input lineage and
timeframe lengths. No TRAIN refit or dataset reconstruction is needed merely to
recover this metadata. Its immutable witness and limits are in the repair record.
This is not current bundle admission, model restoration or migration authority;
the source/state successor and numerical disposition remain outstanding.

The subsequent bounded CPU measurement executes the actual current constructor
and parameter-grouping code with those exact inputs. Its 722 freshly initialized
parameters, 712 + 10 groups and 758 persistent state entries match the retained
historical names/order/shapes/dtypes. This closes actual structural enumeration,
not weight restoration or numerical parity. No optimizer, forward, gradient,
checkpoint deserialization or state conversion runs; preserved session bytes
remain unchanged. The repair record binds the measurement and its limits.

The subsequent 2026-09-07 first-hop inspection identifies and repairs real
retention reference-type mismatches in existing owners: exact descriptive
metadata, native API labels and native producer snapshot bindings. Strict
unknown/extra/list/TEST handling and native byte/hash checks remain intact.
All 289 focused synthetic tests and Ruff pass; both actual pre-TEST native
owner descriptors remain identical after deployment (2,628,372 M1 and 531,190
M5 closed rows). Checkpoint bytes and 1,185 input stat identities are preserved.
Full canonical regression of this new source passes all **4,034 tests** in
**1,082.165 seconds**, zero failures/errors/skips; the 3,912-case result is the
preceding baseline. At that checkpoint, current source had 15 changed existing
recipe byte bindings and four additional roles; public source preparation still rejects
the dirty worktree. No remaining transitive closure, migration or training
authority is implied. Exact immutable evidence and limits are in the repair
record's measured retention-reference section.

The final follow-up removes the retention layer's accidental materializer-script
dependency and restores the intended 109-role source closure. The current exact
delta is 17 changed existing roles plus four additions and no removals. Existing
owners plus a new read-only Group-A completion owner close the active candidate's
non-TEST transfer inventory at 1,434 files / 12,146,382,873 bytes; the broad V46
cleanup root remains opaque and protected. TRAIN's 114 and VAL's 115 Group-A
chunks pass exact manifest/hash/NPZ/time-grid validation. The separate
`--prepare-source-state-successor` route is locked to this source delta and the
36 source-proven stateless Exit parameters. It preserves ID 603 as 579 and all
722 populated AdamW states, with no ordinary-restore or guard-only exception.
No actual checkpoint migration is run. The complete executable CPU suite passes
all **4,062 tests** in **1,074.934 seconds**, zero failures/errors/skips:
`/var/tmp/gx1-readiness-final-canonical-full-20260907.xml`, SHA-256
`3f2ce3e974fe66364fa1cadd127e6b9f366c78945ef92840d85b9d1099df1aa5`.
Transfer and successor evidence, exact hashes and remaining clean-commit/new-
recipe/actual-equivalence/destination-host blockers are in the repair record.
Item 4 remains deferred; no family, TEST seal, checkpoint or safety owner is
removed and no training/cloud authority is granted.

The broad V46 retention root remains deliberately BLOCK: none of its six
recognized direct manifests exists. Its hash-bound rebuild and post-rebuild
JSON reports are readable under the existing owner limits, but they do not
certify every later descendant. No filename alias, synthetic manifest, TEST
traversal, dataset rebuild or cleanup is used to hide that missing closure.

The subsequent explicit non-TEST input inventory hashes 59 distinct files,
10,515,905,851 bytes, under the unchanged capped/locked CPU audit. Every hash
matches its declared role; previous VAL direct-file and checkpoint/session stat
identities remain unchanged. It covers direct recipe/reader inputs and named
lifecycle/cache/squeeze/ECDF/signal/native metadata additions, not complete
transitive provenance or a cloud upload authorization. Native chunks, deeper
generation/evidence dependencies, source/state transition and actual destination
qualification remain outstanding. No feature/data rebuild or TEST access occurs.
Exact scope and immutable evidence are in the repair record's transfer section.
The subsequent existing native owner fully verifies the two exact pre-TEST
M1/M5 bundles, including 1,096 response chunks, producer snapshots and yearly
parquets. Their union with the selected input inventory is 1,185 files /
10,795,866,746 bytes. This closes those two native components, not all downstream
generation/report/event-history references or the source/state transition.

Implemented and verified in this repair wave:

- Selective-edge HAC receives chronological selected trades, not score-ranked
  trades. Preregistered schema v2 explicitly binds that order and rejects legacy
  VAL references. All 26 focused evaluator tests pass under the 4G CPU cap.
- Real, hash-verified retained smoke VAL predictions reproduce the consequence:
  the former 1%-coverage primary PASS becomes FAIL. On its 55 trades, HAC SE is
  21.9185822268 bps, versus 13.9014462920 in the old score-order calculation.
  None of the preregistered coverages qualifies. This is the smoke model, not
  evaluation of the partial full candidate; no GPU inference or TEST was run.
- All 42 existing model-shape/connectivity tests pass under the CPU cap,
  including all-eight-family gradient reachability and incremental Exit carry.
  These mechanical fixtures prove tested connectivity, not market utility.
- Windows Time was stopped/manual; it is now running/automatic and reports a
  successful NTP synchronization. Repeated Windows/WSL clock disagreement was
  observed before repair. The subsequent bounded checks pass; multi-day clock
  stability remains unproven.
- The WSL host config now declares `general.instanceIdleTimeout=-1`, with an
  exact backup retained and existing VM/RAM/CPU/swap limits unchanged. This is
  the distribution idle setting, distinct from the existing VM idle timeout.
  After Windows reported zero running distributions, the empty VM was shut
  down to activate the config. A 90-second no-client idle probe now preserves
  PID-1 start identity; Windows/WSL clocks align. No active distribution or
  user job was stopped. Long-run stability and the old initiator remain unproven.

Current integrated status and independent reviews are in
[`PRETRAIN_READINESS_REPAIR_20260906.md`](PRETRAIN_READINESS_REPAIR_20260906.md).

Seed identity, witnessed publication chronology and transitive retention repairs
are implemented and tested, together with TRAIN diagnostic containment, spread
alias consistency, passive static preflight and explicit usefulness-uncertainty
scope. Six subagents contributed implementation or independent architecture,
feature/data, baseline and simplification review; their findings are recorded
in the linked repair and companion reports.

First-wave complete CPU regression passes all 2,929 tests, zero failures/errors/skips,
in 698.050 seconds: `/var/tmp/gx1-readiness-full-final-20260906.xml`. All 28
changed Python files parse; Ruff introduces no diagnostics relative to HEAD.
Shell syntax and diff whitespace checks pass. This validates the repair wave,
not profitability, checkpoint qualification or large-training readiness.

The follow-up adds atomic usefulness-report publication and schema-v7 clock/
episode-origin checks. Its 200 affected tests and subsequent complete 2,953-test
suite pass, with zero failures/errors/skips; the full suite takes 694.498 seconds
(`/var/tmp/gx1-readiness-followup-full-20260906.xml`). A real native-quote scan preserves exact
spread output bytes on all 470,558 M5 and 2,628,372 M1 rows. A real retention
attempt remains BLOCK on an unrecognized directory-manifest boundary. The
complete usefulness adapter needs compact native-episode streaming and streamed
paired statistics; it is not implemented merely by passing these core checks.

The next repair replaces population-sized component-vector retention with
batchwise paired-summary/effect/synergy accumulation in the existing core.
Report v8/donor-plan v2 also bind native-MTF geometry: the actual 5,500 VAL entry
episode pairs form 131 compatible groups with no singletons, whereas adjacent-
entry rotation matches none. The first combined streaming/geometry/handover
regression passes 288 tests. A compact adapter passes 39 separate CPU staging
tests, including real native-model arithmetic on synthetic inputs, but is not
deployed or production-wired. File-backed online/teacher/population identity and
the full producer remain open; do not describe F2 or pretraining readiness as
complete. Exact evidence and final integrated status belong to the repair record.
The deployed streaming/geometry code subsequently passes the complete 3,143-test
CPU suite in 712.501 seconds, with zero failures/errors/skips, in
`/var/tmp/gx1-readiness-streaming-integrated-full-20260906.xml`. Pending compact
plan/native producer/read-only session work is not included in this result.

Subsequently the compact structural-plan owner is deployed and exercised on all
5,500 real VAL episode pairs / 5,632,000 states, with 131 compatible groups and
no omitted donors. An explicit read-only opener in the existing session owner
also reads the actual checkpoint 152 / 9,664 steps without changing any session
file bytes, modes or mtimes; both writers reject. The 332 affected canonical
tests pass in 28.092 seconds. The staged native adapter/selected-pair reader
passes 87 separate mechanical tests, but remains undeployed and not production-
wired. Exact evidence and limitations are in the repair record's native-plan/
read-only section. None of this creates a new dataset or selected candidate.
The subsequent full deployed suite passes all 3,172 tests, zero failures/errors/
skips, in 714.228 seconds (`/var/tmp/gx1-readiness-native-reader-full-20260906.xml`).
Staged native producer/reader and subsequent normalization-binding work is not
included; handover retains the review-hold BLOCK and no CUDA authority.

A subsequent read-only construction through the real VAL Dataset/lifecycle
owners confirms all 5,509 immutable Entry indices/source clocks and 5,632,000
Exit state rows, without a model forward or new dataset. Its 152.437-second,
3,252,120-KiB peak-RSS result is data-reader evidence only, not integrated model
headroom. The native input-reader and normalization-identity repairs remain
staging-only: their combined 359-test regression and final 103-test affected
core rerun pass. A runtime fill snapshot must not be fabricated for the two-
sided research population. The subsequent staged v9 report contract replaces
those runtime-fill fields with native episode/token-population and selected-
teacher identity, and binds normalization to embedded bundle metadata. The
selected loader preserves six explicit artifact roles and repeats its strict
source check after loading. All 399 combined affected staging tests pass in
80.057 seconds, zero failures/errors/skips, in
`/var/tmp/gx1-readiness-native-report-identity-corrected-20260906.xml`.
The following staged collector now builds the complete native baseline through
the existing Entry/Exit/teacher/target owners, retaining every Entry row and
separate frozen tokens with exact compact pack/donor identity. A boundary check
rehashes selected-state artifacts, actual model states and source provenance
without reselection or model mutation. All 472 affected staging tests pass in
310.393 seconds in `/var/tmp/gx1-readiness-native-baseline-corrected-20260906.xml`,
including native arithmetic on synthetic full-dimension input. Source-derived
baseline ndarray payload is 146,694,486 bytes for the earlier real population
counts, not a measured integrated memory requirement.
This remains mechanical verification, not an actual candidate usefulness result.
The complete intervention/report producer, full consumed-input stability proof
and public control wiring remain unfinished; all five pending native files
remain undeployed. The canonical 3,172-test result does not include them.
Exact evidence and limitations are in the repair record.

The recipe, session contract, resume pointer and active checkpoint-152 bytes
remain unchanged and hash-verified. Five of the recipe's 105 source bindings
now differ: retention, immutable event authority, basic features, native
market context and the trainer's read-only session owner. Source remains
uncommitted. Shared contract changes invalidate
the old source closure even without altering learned mathematics; no old
source-bound gate is permission to run modified source. Future sky work
still requires a reviewed native host/telemetry profile, complete dependency
inventory, explicit checkpoint lineage, a measured one-GPU pilot and cost
approval. Do not invent hardware thresholds before selecting a real host.

## Historical operator-approved continuation (superseded)

2026-09-06: the operator approved controlled checkpoint recovery and continued
training, retaining all limits. The real CPU transfer, current-source recipes,
readiness and new candidate gate have passed. The launch hold is cleared only
for that exact recovered continuation; clean source and fresh signed 160 W
telemetry are still required. The original session is untouched. The new
standard recipe/session has bit-identical learning state; only the reviewed
guard differs in the 105-file recipe source closure. No production trainer,
model, source validator, session protocol or runtime compatibility lane changed.

Recovery preparation evidence, 2026-09-06:

- Source proof: rehashing all 105 historical recipe bindings still finds only
  `trainer_safety_guard` changed. The new recovery utility and handover renderer
  are outside that unchanged learning closure. No source-check exception was
  added to the canonical launcher or session owner.
- Measured on the real preserved checkpoint: strict CPU session loading passes
  at checkpoint 125 / 7936 steps; the epoch order has all 313399 TRAIN rows;
  online, target, optimizer and EMA tensors are finite. The pointer file SHA is
  `645cd547316bf43f6b84a09d4b83ee98cc282771bece658a5efdc76dc52a4a51`.
- Exact incident logs: optimizer/EMA step 7936 completed at
  `2026-09-05T19:56:30.184Z`, guard exit logged at `19:56:32Z`, and checkpoint
  serialization completed at `19:56:32.468Z`. The unchanged `train_epoch`
  executes optimizer plus EMA before `step_done`, then synchronously calls the
  checkpoint hook before another batch. The saved update predates guard exit;
  this does not prove continuous signed telemetry after guard exit.
- Mechanical CPU tests: 29 checkpoint/transfer tests passed, including a real
  two-slot atomic transfer with mocked recipe/Git gates. This proves the
  transfer mechanics, not market-data or CUDA behavior. Negative cases reject
  learning/data/run/path changes, late/ambiguous updates, nonfinite tensors,
  existing successor directories and mixed CLI modes.
- Affected verification: 93 tests passed
  (`/tmp/gx1-guard-recovery-focused-fixed-20260906.xml`). The first affected
  run found the newly requested `passord.md` missing from the handover
  fingerprint inventory. Both the real inventory and its assertion now
  include it; the marker's content was not changed. The previously reported
  2477-test run predated that marker addition and was not a complete check of
  the final previous commit.
- All shell syntax, capped Python compilation and diff whitespace checks pass.
- Complete CPU regression: all 2496 tests passed, zero failures/errors/skips,
  659.532 seconds (`/tmp/gx1-guard-recovery-full-20260906.xml`), under the existing
  4G audit cap. Handover's marker inventory was aligned with the test's explicit
  path order; the subsequent real-state handover must also prove this after
  the hold is cleared. No CUDA has started during this preparation.


### Real guard-recovery transfer and CPU gates — 2026-09-06

The committed verifier transferred the real state through the unchanged
canonical session owner. It checked the original recipe against its frozen Git
bytes, required the reviewed guard commit, proved the new recipe's current
clean source, reconstructed both real session contracts with the trainer's own
builder, and rehashed every typed checkpoint component after atomic publication.
All 15 non-contract components are identical, including online/target weights,
optimizer, EMA, scheduler, RNG, epoch order, progress and selection state.
Original checkpoint/contract/pointer and incident-log hashes were rechecked and
remain unchanged. No CUDA ran during transfer.

- Transfer: `/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/GUARD_RECOVERY_20260906T061447Z/RECOVERY_PROOF/CANDIDATE_GUARD_RECOVERY_20260906T062738799702Z.json`
  SHA-256 `38b5a5d730acd6de00a5220b7c6938aad544ff2a42f79923644670438d3d5ec4`.
- New session contract: `333ef6bcef013f39d317ecc28bdc093b7ab75af04c688fb6ab2c694d494767cf`.
- Initial transferred state: `5472a2f98bcb1a6571325f046f50adf3a16fc7f40574f8f89fda391d0b1c1317`,
  checkpoint 125, TRAIN epoch 0, offset/global step 7936, complete=false.
- Trainability: `/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/GUARD_RECOVERY_20260906T061447Z/TRAINABILITY/ENTRY_PRETEST_TRAINABILITY_READINESS_20260906T062904212387Z.json`,
  `READY_FOR_PRETEST_CANDIDATE_TRAINABILITY_REVIEW`, zero failures.
- Candidate readiness: `/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/GUARD_RECOVERY_20260906T061447Z/CANDIDATE_READINESS/ENTRY_CANDIDATE_READINESS_20260906T062943428108Z.json`,
  SHA-256 `d07c8e31d57d56fe7fd09b92ee4e4990b2697b26ea869a3d51e61cb99fa9afa6`,
  `READY_FOR_CANDIDATE_TRAINING`, zero failures.
- New gate: `/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/GUARD_RECOVERY_20260906T061447Z/LAUNCH_GATE/ENTRY_PRETEST_CANDIDATE_LAUNCH_GATE_20260906T063105728686Z.json`,
  SHA-256 `e10ce617e4ba02e96218f674d50aec21a1989c49189005a9fb5dc66e891d3ed7`,
  `READY_FOR_PRETEST_CANDIDATE_TRAINING`, zero failures; independently reloaded
  through the normal gate validator.

The original completed smoke/VAL audit is retained as evidence for unchanged
learning code and data. A CPU-only smoke recipe rebind supplies the current
source half of the static trainability contract; it was not executed and is
not represented as a new bundle. The current-source status now selects the
candidate recipe, not that unused smoke output. No source freshness check was
waived, no original recipe/checkpoint was relabelled, and no training started
from zero. All 173 affected current-state controls subsequently passed, with
zero failures/errors/skips in 70.611 seconds:
`/tmp/gx1-guard-recovery-current-state-20260906.xml`. The 2496-test full-suite
result remains applicable to unchanged executable code; this final wave changes
only state references and documentation. Clean commit/preflight precede CUDA.

### Historical terminal safety failure — 2026-09-05

At 2026-09-05T19:56:32Z, window 3 logged `event=stop reason=guard_exit`.
On recovery from an interrupted observation, tool handle 12222 was missing,
guard PID 601907 was absent, and exact candidate PID/PGID 601955 was still
running with PPID 172. This is a real loss of the mandatory guard, not an
observation timeout or an expected 7200-second boundary. The agent sent TERM
to the verified owned group at 19:57:39Z; it became defunct, then disappeared.
That event did not authorize a retry. The later, explicitly approved and
verified recovery above is the separate continuation authority.

Last signed heartbeat: 19:56:05Z, 60 C core / 68 C memory / 158.64 W draw /
160 W limit. The unguarded interval has no continuous signed evidence.
Latest preserved checkpoint: 125, 7936 TRAIN optimizer steps, epoch index 0,
complete=false, slot 0 SHA-256
`40f1de6617784fb5519ee849443fe179d0fdbe2f59cca963a3683f90b187a685`.
The active session reference remains unchanged. Exact third-window sidecars
are recorded in the current handoff; no artifact or log was deleted.

Source at the incident: `terminate_child_group` in `gx1_guarded_trainer_exec.sh`
recorded its stop event and wrote to stderr before issuing TERM; it also logged
before KILL. The real unmodified guard was exercised in a bounded 4G CPU test
with a real closed stderr pipe and owned child process. Both cases left the
child executing after the guard exited: normal TERM and TERM-ignoring child
(`/tmp/gx1-guard-closed-stderr-before-20260905.xml`, two expected failures).
This reproduces the failure mechanism, not the uninstrumented reason that the
original observation service disconnected.

The same owner now ignores SIGPIPE (ordinary failed writes still unwind under
errexit), sends TERM before diagnostic I/O, treats cleanup diagnostics as
best-effort, and sends KILL before its diagnostic. Both exact new regressions
pass (`/tmp/gx1-guard-closed-stderr-after-20260905.xml`). No thermal/power/memory
threshold, model, target, dataset or recipe value was changed. The guard's
source bytes have changed, so the historical recipe's executable closure is
not current launch authority. Old recipes, gates, checkpoints and logs remain
untouched; source-safe recovery is a separate unresolved boundary.
The existing fail-closed hold is restored:
`GUARD_EXIT_ORPHANED_CUDA_NO_RETRY`. Ordinary resumption instructions below
are superseded until the guard failure and safe source-bound recovery are resolved.

Recovery boundary, proven from the existing owners: `recipe_source_binding_paths`
includes `trainer_safety_guard`, and `_validate_source_bindings` requires its
exact bytes. `_candidate_training_session_contract` includes that recipe's
source provenance; `_CandidateTrainingSession` rejects a different immutable
contract. Its legacy bridge only adds previously absent provenance for the
same certified source; it does not allow a repaired guard or source migration.
Therefore neither a new recipe alone nor a documentation-only commit can
resume this checkpoint under the repair. No existing source-change recovery
route was found in these launch/session owners. A reviewed recovery must
preserve the original session and certify unchanged learning state and inputs;
it must also resolve checkpoint 125's overlap with the guard-exit timestamp.
Do not silently choose the inactive slot or reset TRAIN. This is a real
provenance/safety decision, not another routine smoke approval.

Focused verification of the repaired source passed all 127 tests with no
failures, errors or skips in 89.903 seconds
(`/tmp/gx1-guard-exit-repair-focused-20260905.xml`). The prior 2474-test full
result below predates the executable guard repair and does not cover it.
The executable handover's new hold reason originally fell through to the old
target/data-rebuild instruction. Its existing renderer now names CPU guard
repair and verified source-bound checkpoint recovery instead, while returning
the same BLOCK/exit 2 and no CUDA/TEST/paper/live authority. This adds no launch
or recovery route. The final focused verification also covers this status-only
branch; the complete suite was already collected when this case was added.

The first repaired-source full suite completed in 725.470 seconds: 2475
passed, one failed, zero errors/skips
(`/tmp/gx1-guard-exit-repair-full-20260905.xml`). The failure was the real-state
integration test in `test_entry_model_native_train_recipe.py`: it treated any
review hold as the old target-correction boundary and expected the old causality
error. The current evidence owner instead selects the newer pre-TEST TRAIN/VAL
recipe and correctly rejects the retained V46 dataset by identity first. The
test now verifies the current recipe identity with the real CPU evidence owner,
proves an in-memory hold removal cannot change that selection, and still
requires exact rejection of the old dataset. No production validator changed;
the actual hold was never removed.

Final affected verification: all 170 tests passed, zero errors/failures/skips,
135.009 seconds (`/tmp/gx1-guard-exit-repair-final-focused-20260905.xml`). This
includes the new handover reason, the corrected real-state assertion, and both
closed-pipe guard regressions. Shell syntax, capped Python compilation and diff
whitespace checks passed. Independently hashing all 105 recipe-bound source
files found exactly one changed binding: `trainer_safety_guard`. The historical
recipe hash still matches. Model, optimizer/trainer, feature and data-contract
source bytes are unchanged; this fact alone does not authorize recovery.

Final complete CPU suite: all 2477 tests passed, zero errors/failures/skips,
725.317 seconds (`/tmp/gx1-guard-exit-repair-final-full-20260905.xml`). The
existing job survived interrupted observation and was followed by the same
handle and confirmed OS process; no duplicate suite was started because of
an observation interruption. This supersedes the first repaired-source suite's
one failed test assertion, not the CUDA safety failure. The actual handover
and official candidate dry-run both returned BLOCK/exit 2 under the preserved
hold. No new CUDA run, checkpoint rewrite, data rebuild or cache deletion was
performed. The operator-requested root `passord.md` is only a coordination
marker and grants no execution authority.

### Historical successful launch and first normal boundary

The full five-year candidate started from clean launch commit
`d4376b3c`, with the existing source `e25a8cb6` recipe and exact new gate below.
The official dry-run passed with profile=candidate, epochs=30, batch=8,
patience=5, subsample_rows=0 and test_accessed=false. A fresh signed launch
response was `43,48,30.8,160,325`. First optimizer step completed at
2026-09-05T16:03:38.640Z; checkpoint 2 persisted 64 steps at 16:05:16Z.
Independent read-only verification matched the 200910446-byte state to SHA-256
`9565a97575fc3bfbe08a532c86be921b91bc6ae0f385a40912cc3a272969a5ff`
and session contract `6ff73e8607347049f453977c4f144bfec7d913322e72680f1c345b18bf68002d`.
The run is not complete and no full-epoch VAL result exists yet.

Exact active session/log paths and the 15-minute follow-up / guarded-window
resume protocol are recorded in the current handoff. The active launch-state
session reference is now this candidate; the old V9 session is retained.
The first guard window ended solely at `wall_clock_limit_7200s` at
2026-09-05T17:47:06Z, exit 75. All 5646 signed samples remained within policy:
peaks 62 C core, 68 C memory, 164.5 W draw and 9460 MiB VRAM. There was no
additional stop/failure or forced KILL, and the exact process group was gone.
Handover rehashed checkpoint 62 (3904 TRAIN steps, epoch index 0, incomplete),
slot 1 SHA-256 `8d9f6bbb91feb0b6d67c84510df9a61abfe18936a96baf571d6c7b3339bf7398`.
This verifies the normal time-boundary prerequisite for same-session resumption;
it does not prove a fresh-process reload until `resumed=1` and later progress
are actually observed. The status-only verification/commit belongs between
windows, without executable changes or a bypass of the capped pre-commit hook.

At the first boundary all 68 handover/current-dataset tests passed in 24.874 s
with zero failures/errors/skips
(`/tmp/gx1-first-candidate-window-handover-20260905.xml`). Shell syntax, capped
Python compilation and `git diff --check` passed. The existing 2474-test full
regression result below covers the unchanged executable source; this status-only
change does not repeat it, the completed smoke or the completed VAL pass.

After the completed smoke, the operator explicitly authorized the remaining
local progression without separate approvals between ordinary steps: exact
bundle guarded VAL prediction, CPU post-run audit/readiness, candidate launch
gate and, only if those gates pass, the unchanged five-year full TRAIN/VAL
candidate (batch 8, maximum 30 epochs, patience 5). The review hold is
transitioned off for that sequence; global admission stays BLOCK. The next
step is VAL prediction, not another smoke. Existing signed 160 W telemetry,
source closure and safety limits are unchanged. A genuine safety/data/model
failure stops progression; hardware failures never receive automatic retries.
No TEST release, candidate acceptance, paper/live, external spend or material
model change is authorized. Older per-step approval/hold statements below
describe the preceding state, not a requirement to ask again at every step.

## Completed canonical smoke — 2026-09-05

### Successor VAL and candidate gate completed

- VAL report: `/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/ENTRY_V9_FIVE_YEAR_HANDOVER_GATE_SMOKE_20260905T153048Z_VAL_PREDICTIONS/ENTRY_CANDIDATE_SELECTIVE_EDGE_20260905T155407025645Z.json`, SHA-256 `168c340687b98df097471229fcd051396b59888f45e31aeb3bbb76a6c5807ebd`.
- Predictions: `/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/ENTRY_V9_FIVE_YEAR_HANDOVER_GATE_SMOKE_20260905T153048Z_VAL_PREDICTIONS/selective_edge_predictions_20260905T155407025645Z.parquet`, SHA-256 `5a0dec1133ef11d56fbb86d958c38bdbfb56905c2158b286bed6dd9177146443`; all 5509 VAL rows, no TEST, no trainer started.
- Post-run audit: `/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/ENTRY_V9_FIVE_YEAR_HANDOVER_GATE_SMOKE_20260905T153048Z_POSTRUN_AUDIT/ENTRY_MODEL_NATIVE_SMOKE_BUNDLE_AUDIT_20260905T155439237233Z.json`, SHA-256 `15e45ed5a31c711eaf0a4440fe4342f632861a626defc610b9b8cd758e537f58`.
- Candidate readiness: `/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/ENTRY_V9_FIVE_YEAR_HANDOVER_CANDIDATE_20260905T153048Z_READINESS/ENTRY_CANDIDATE_READINESS_20260905T155504876246Z.json`, SHA-256 `9959edc3941cf49a5c6da7b83415198a5cf4bdbe4c4a01308f2db23f82829fa1`; READY_FOR_CANDIDATE_TRAINING, all six checks PASS.
- Candidate launch gate: `/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/ENTRY_V9_FIVE_YEAR_HANDOVER_CANDIDATE_20260905T153048Z_LAUNCH_GATE/ENTRY_PRETEST_CANDIDATE_LAUNCH_GATE_20260905T155520511753Z.json`, SHA-256 `4f9003275f930e3aa2de6eefa0cd0ad656a4d865c03199ed1368fb0c17e16678`; READY_FOR_PRETEST_CANDIDATE_TRAINING, zero failures, TEST/paper/live authority false.
- VAL guard: `/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/.ENTRY_V9_FIVE_YEAR_HANDOVER_GATE_SMOKE_20260905T153048Z_VAL_PREDICTIONS.guard.GXQh4E5R.log`; trainer output: `/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/.ENTRY_V9_FIVE_YEAR_HANDOVER_GATE_SMOKE_20260905T153048Z_VAL_PREDICTIONS.trainer.hPYy9mu8.log`.

Exactly one guarded VAL pass exited 0. Terminal guard aggregates: 93 signed
samples, peaks 56 C core, 56 C memory junction, 152.72 W draw, 726 MiB VRAM,
with the 160 W physical limit enforced. The strict smoke audit has seven
failures, all never-top-ranked specialists. Every active output is live;
strict bundle components pass; all eight specialist weights are finite,
positive and state-varying on full VAL. The existing technical start contract
therefore passes without changing or waiving any checks. This is not a
production-quality, model-acceptance or trading-edge result. The selected
candidate remains the unchanged source `e25a8cb6` recipe already recorded.

The CPU-only evaluator dry-run first rejected `--device cuda` under an audit
scope before allocation; it then passed with `--device cpu`. The actual CUDA
pass used the allow-listed producer guard. The first documentation commit
attempt was blocked because the focused CPU tests still held the exclusive
job lock; it succeeded after all 68 tests completed. Neither event caused a
CUDA retry or bypass. No executable source changed in this continuation.

The terminal gate transition initially passed 67/68 focused tests; the one
failure was the snapshot test still allowing only the pre-VAL statuses. It
now validates the actual immutable gate and its bound audit/readiness, and the
active session against its own recipe instead of a historical run-name literal.
All 68 tests then passed (`/tmp/gx1-candidate-gate-transition-confirm-20260905.xml`).
This test-only update does not change the recipe-bound executable closure.

### Prior canonical smoke boundary

Exactly one authorized canonical smoke completed with process/watchdog exit 0.
No automatic retry, separate VAL prediction pass, full candidate, TEST,
acceptance, paper or live execution occurred. The execution hold is restored.

- Recipe: `/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/ENTRY_V9_FIVE_YEAR_HANDOVER_GATE_SMOKE_20260905T153048Z_RECIPE.json`,
  SHA-256 `cd4c65c68547e1d985b0864eaa53c51a86c9ddeb5ca07769006620b363e12de7`.
- Source: `e25a8cb6377eec496d1863c667c0d6cff785d86a`; clean launch checkout `8c3760b0`.
- Bundle: `/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/ENTRY_V9_FIVE_YEAR_HANDOVER_GATE_SMOKE_20260905T153048Z_BUNDLE`.
- Commit-manifest file SHA-256: `26e6ba3e61e915ec9cdf299680a7f07ab83453b4ddfce5d42546518201bfb036`.
- Bundle commit SHA-256: `8a9b197b805e6c373f0e17a3cf4bf0f7559cd43936431ce03cb325ce3c7bfc5d`.
- Bundle metadata SHA-256: `2372aeb7fd90136811d455b74325a97ddc4c39ddb2ab2ec3ac2716c0d1cfd9df`.
- Watchdog: `/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/.ENTRY_V9_FIVE_YEAR_HANDOVER_GATE_SMOKE_20260905T153048Z_BUNDLE.guard.y5LogBp8.log`.
- Trainer output: `/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/.ENTRY_V9_FIVE_YEAR_HANDOVER_GATE_SMOKE_20260905T153048Z_BUNDLE.trainer.7fNwucFq.log`.

The trainer completed epoch 1 with batch 8 and 32 TRAIN / 32 VAL rows,
strictly reloaded its staged bundle and atomically published it. Independent
CPU verification rehashed the bundle commit and matched recipe/source
provenance in metadata and lock. Both normal handover modes then verified the
executed reference and bundle before the hold was restored.

Measured watchdog terminal aggregates: 311 samples; peaks 61 C core, 60 C
memory junction, 158.56 W draw, 9458 MiB VRAM. Published power-limit samples
were 160 W and the guard enforced that ceiling throughout. No stop, kill or
fatal event occurred. Use the terminal aggregate, not the initial
`event=telemetry` row or sparse heartbeats, for these maxima.

All ten joint tasks were supervised, received gradients and moved from
neutral; Entry fitted-Q and Exit component movement passed. Technical smoke
admission was `strict_ok=0 technical_ok=1` under the existing smoke policy.
Strict warnings were `trendline_event` diagnostic support
`rows_by_column=[3,12,32,32]` and constant/dead Exit cooperation-gate indices
`[166,175]`. These are not all-clear candidate-quality evidence and were not
waived or repaired during this run. Selected smoke VAL PnL was
`-15.70416259765625` bps; the tiny trained sample establishes no trading edge.
Full-population normalization used 313399 Entry TRAIN decisions and zero
VAL/TEST fit rows.

Next execution requires new scoped authority for guarded VAL predictions from
this exact bundle, then post-run audit/readiness and an exact candidate gate.
Full candidate training and every admission/TEST/paper/live route stay closed.

After recording the executed bundle and restoring the hold, all 68 focused
handover/current-dataset tests passed with zero failures/errors/skips
(`/tmp/gx1-smoke-terminal-handoff-20260905.xml`). No executable source changed
after the smoke; the terminal update contains status and documentation only.

## Handover repair and pre-launch record

The paragraphs below record the preparation states before the completed run;
they are retained history, not a new permission to execute.

Source repair is committed as `e25a8cb6`. Its new canonical smoke recipe is
`/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/ENTRY_V9_FIVE_YEAR_HANDOVER_GATE_SMOKE_20260905T153048Z_RECIPE.json`,
SHA-256 `cd4c65c68547e1d985b0864eaa53c51a86c9ddeb5ca07769006620b363e12de7`.
The paired, unexecuted candidate recipe SHA-256 is
`7923b9ea638de50ffb06a82a7f601bec96f86f96a8f5f4ab89122cbccc4f094f`; new CPU readiness is
`/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/ENTRY_V9_FIVE_YEAR_HANDOVER_TRAINABILITY_20260905T153048Z/ENTRY_PRETEST_TRAINABILITY_READINESS_20260905T153150593398Z.json`,
SHA-256 `8489d179562a3cd2d04b8bf4f864fbfe5f3fecad0551ace81b6e5f5d0269bbaf` (all five checks PASS).
Actual direct-dataset handover validation passed on these files while the
old execution hold was still present. The hold was then cleared only for the
operator's one smoke. Both normal handover modes passed from clean commit
`dea9246a`, and the exact official launcher dry-run passed: canonical smoke,
batch 8, one epoch, 32 rows per split, no candidate gate and TEST unopened.
All admission fields remained BLOCK/empty. No CUDA attempt had started at
that pre-launch checkpoint.
All 68 real handover/current-dataset transition tests now pass
(`/tmp/gx1-smoke-real-handover-transition-confirm-20260905.xml`), after
updating two stale historical status-text expectations. No production check
was weakened. The current smoke reference is checked against its actual
immutable recipe; executed references also require the real bundle commit.

After CPU preparation, the operator authorized one canonical gate-smoke and
then explicitly authorized repairing its handover linkage. Pre-launch
inspection reproduced `CURRENT_AUDITED_DATASET_EXECUTION_CAUSALITY_EXPECTATION_INVALID`:
normal handover still required the historical V46 report collection despite
the valid direct TRAIN/VAL successor. The earlier CPU completion did not prove
this execution transition.

The existing current-dataset owner now accepts an explicitly bound
`current_pretest_trainability_readiness` reference. Its existing readiness
owner revalidates the immutable report, both nested recipes, exact selected
recipe/dataset/run identity, zero-failure pretrain proof and small JSON
audit/manifest hashes. The current entry-notional causality contract is still
required. Admission fields stay BLOCK/empty; legacy V46 evidence is retained
as history, not fabricated or upgraded. Missing/malformed direct evidence
fails closed instead of falling back to V46.

The handover renderer derives the recipe status instead of always claiming a
candidate gate is ready. Model, feature, target data, trainer settings and
hardware limits are unchanged. The source change requires newly materialized
recipes/readiness, but no data rebuild. Twelve mutation/control cases cover
this route; all 85 focused handover/readiness/current-dataset tests passed.
The full CPU regression suite passed all 2474 tests in 693.649 seconds, with
zero failures/errors/skips (`/tmp/gx1-smoke-handover-full-20260905.xml`).
After adding an explicit dry-run-pending handover status, all 85 affected
tests passed again (`/tmp/gx1-smoke-handover-final-confirm-20260905.xml`).
Capped compileall, all shell syntax checks and diff whitespace checks pass.
This authorization covers exactly one canonical 32-row-per-split smoke;
separate VAL predictions, full candidate, TEST, acceptance, paper and live
remain unauthorized.

## Historical CPU successor completion — 2026-09-05

The authorized CPU goal is complete: corrected TRAIN-only target policy/ranking
and sizing, new TRAIN/VAL datasets and lifecycle, all affected CPU audits,
two new source-bound recipes and direct CPU trainability-readiness.
The active execution hold remains BLOCK; its reason is now
`SUCCESSOR_CPU_READY_REQUIRES_SCOPED_CUDA_AUTHORIZATION_AND_RUNTIME_EVIDENCE`.
No successor CUDA, full candidate training, physical TEST access, candidate
acceptance, paper or live activity occurred during that CPU phase. The launch
state's then-retained `current_source_technical_recipe` and old gate were
historical records, not authority for this successor.

All paths in the following inventories are relative to:

`/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z`

| Artifact | Relative path | File SHA-256 |
| --- | --- | --- |
| TRAIN parquet | `dataset/entry_dataset__ENTRY_FITTED_Q_train.parquet` | `dec6216669864e807dc0c58e66dc53def84b515efb4d1f3c0fd0b48d1c1777af` |
| TRAIN manifest | `dataset/entry_dataset__ENTRY_FITTED_Q_train.manifest.json` | `9311e6d8617c92b7b3dec1c6fab8042002b9610431828266c39976460689f7e6` |
| VAL parquet | `dataset/entry_dataset__ENTRY_FITTED_Q_val.parquet` | `2de2fac108ec7313307376fa0eed7daad27b9f90f9eaf38cba4b4ab02e8a013e` |
| VAL manifest | `dataset/entry_dataset__ENTRY_FITTED_Q_val.manifest.json` | `3eb41d343b6a6461f59396ea7c225246aaf5a1a7ae307ff02304909da690c76e` |
| Unopened TEST guard | `ENTRY_MODEL_NATIVE_PRETEST_TEST_GUARD_20260905T144343Z.json` | `521e6f8315a92822b64ce4114799c899067f2a8031e73ca6486393da50a9a73c` |
| Canonical smoke recipe | `ENTRY_V9_FIVE_YEAR_ENTRY_NOTIONAL_GATE_SMOKE_20260905T144446Z_RECIPE.json` | `cafabf1f7b87a3feb630b58a1b78ffc53d027129d60d41c7fe88f643b68d4476` |
| Full candidate recipe | `ENTRY_V9_FIVE_YEAR_ENTRY_NOTIONAL_CANDIDATE_20260905T144446Z_RECIPE.json` | `e1441e9a5fbe693b348a27ff5b13dfb251830c5e72433281d3c2409c87e779e0` |
| CPU trainability readiness | `ENTRY_V9_FIVE_YEAR_PRETEST_TRAINABILITY_20260905T144558Z/ENTRY_PRETEST_TRAINABILITY_READINESS_20260905T144627440261Z.json` | `c577b80cd2bd15f527640a105e20fceae4b1e8fd660f67b949d0a7811d2c2a86` |

The recipes both bind source `406c732e19369180e94f72702967e52740a3a0cd` and
identical artifact-bindings SHA-256
`dd61724a6cb610b373ab38b7f1c3a1fff664b54bc3cecf75f350bc578974a637`.
They are report-only, with activation false and no execution side effects.
Neither declared output bundle exists: these are recipes, not executed runs.
Their complete trainer configurations match the exact rehashed historical
recipes; no architecture, feature formula, learning objective or early-stop
change was introduced. Candidate: batch 8, at most 30 epochs, patience 5,
minimum epoch 1, no row subsample. Canonical gate-smoke: batch 8, one epoch,
32 deterministic rows. Historical smoke/VAL results cannot be relabelled as
new-source or new-data execution.

Actual TRAIN population is 313399 rows, observed from 2021-06-01T00:00Z to
2026-05-29T14:50Z; actual VAL is 5509 rows, observed from
2026-05-31T23:55Z to 2026-06-30T14:55Z. Declared emission windows remain TRAIN
2021-06-01T00:00Z through 2026-05-31T23:50Z and VAL
2026-05-31T23:55Z through 2026-06-30T23:55Z. Physical TEST starts
2026-07-01T00:00Z and was not opened. A capped parquet rehash/footer/time-column
check confirmed the actual counts, sorted unique disjoint clocks and matching
TRAIN-only direction/sizing policy identities. Sizing policy SHA-256:
`793b305565848827675a421838d8aff0a302c77613b30c3df454ab5e0f59430b`.

The new full-population pretrain audit is
`dataset_audits/XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_20260905T143350408713Z.json`.
It reports PASS, verifies large-artifact hashes, covers all 313399 TRAIN and
5509 VAL rows, and records zero target mismatches/invalid target modes with
target finite rates of 1. Cross-surface, full-input liveness, foundation
feature/target, specialist, execution causality and both sequence reconstruction
audits also PASS. No physical TEST population was loaded.

| Bound audit/proof | Relative path | File SHA-256 |
| --- | --- | --- |
| dataset_build_proof | `dataset/DATASET_BUILD_PROOF.json` | `5b4ca02d67dcfe7db857f8c769c8db6b0f4f1d08e6ac25f382278e98055756df` |
| execution_causality_audit | `dataset/ENTRY_EXECUTION_CAUSALITY_AUDIT_20260905T143732Z.json` | `1844da8718e377ee19e92ed00e3ae0ad68fbd2f3ebd2a7b75a7fc8e3dca02547` |
| feature_audit | `dataset/AUDIT_FOUNDATION_FEATURES_20260905T143544Z/ENTRY_FEATURE_FOUNDATION_AUDIT_20260905T143626Z.json` | `27b6e15a71d07f63b675b81ba6cde53c463a9c23412724a4cc17816a6ff8a607` |
| full_input_liveness | `dataset/ENTRY_FULL_INPUT_LIVENESS_CONTRACT_20260905T140138802080Z.json` | `70a28b91a69261e87b7bc79a4b6bf9cd6b8b3ea3664f95d4f2a42f48cfbf5108` |
| specialist_audit | `dataset/AUDIT_SPECIALIST_FEATURES_20260905T143544Z/ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_20260905T143705Z.json` | `43e63401dd17babc43836f77fc35d364e05d533c89de848eccf27b77eb203b97` |
| target_audit | `dataset/AUDIT_FOUNDATION_TARGETS_20260905T143544Z/ENTRY_TARGET_FOUNDATION_AUDIT_20260905T143645Z.json` | `50f06a04edb4a7c2616a76a661a323df6ca4948479b08a4b355282c2aae0bb0e` |
| train_sequence_source_reconstruction | `dataset/ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_TRAIN_20260905T143732Z.json` | `53f836d0bee89409636ad58f988ada44d61633bfd62c10014bef4e5f2f68ef29` |
| unified_exit_lifecycle_manifest | `dataset/UNIFIED_EXIT_LIFECYCLE/UNIFIED_EXIT_LIFECYCLE_MANIFEST.json` | `3fa95fb054b9e593708533e714189939f2e48fb10270dbb48efb5cc8aefdae52` |
| val_sequence_source_reconstruction | `dataset/ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_VAL_20260905T143732Z.json` | `9536e442bc5840674b1d9efc21edd1257d512bfaeba38d7ad3707f6a3a193523` |

The existing lifecycle consumer loaded and validated the real new corpus
under the audit cap, including exact Entry parquet/manifest bindings, clocks,
windows and mapping proofs. TRAIN has 626780 episodes / 320911360 states;
VAL has 11000 episodes / 5632000 states, with equal long/short state counts.
Verification log: `lifecycle_verification.log`.

CPU readiness decision is
`READY_FOR_PRETEST_CANDIDATE_TRAINABILITY_REVIEW`, with all five checks true
and no failures: exact shared dataset, direct artifact rehashes, full five-year
pretrain PASS, clean current candidate source closure, and clean current smoke
source closure. It explicitly records `candidate_training_allowed=false`,
`activation_authority=false` and `promotion_shadow_live_allowed=false`.
An individual causality audit's local `training_authorized=true` certifies
its causal-data contract only; it does not override the hold or grant
operator/CUDA authority.

Remaining execution work is outside this completed CPU goal: obtain new
explicit scoped authorization for the successor canonical gate-smoke, record
the reviewed hold transition, and require clean handover/exact preflight plus
fresh signed 160 W telemetry before execution. Then obtain separately scoped
successor guarded VAL/runtime evidence and its new candidate gate. Full
candidate training requires that gate and explicit full-candidate authority.
Do not clear the hold or start any of these automatically.

Final handoff/hold and versioned-hook verification passed all 67 cases under
the 4 GiB / 512 MiB audit cap, with zero failures/errors/skips:
`/tmp/gx1-cpu-successor-handoff-20260905.xml`. The 15 small JSON artifacts in
the inventories above were independently rehashed against this document;
the large parquets had already been rehashed during production/readiness.
Launch state remains within its 14000-byte limit with the exact five-key
BLOCK hold; historical recipe/gate records were not rewritten.

## Authorised CPU continuation and verification history

The operator authorised completion of the corrected TRAIN/VAL successor on
2026-09-05. The shared goal is to rebuild its TRAIN-only policy/ranking,
signal lineage, target data and affected audits, then materialize new
source-bound recipes and report exactly which launch evidence is still
missing. No CUDA, physical TEST, candidate acceptance, paper or live work is
authorised by this continuation. The explicit training hold stays active.

The existing rebuild wrapper now has an explicit `--pretest-only` route;
it rejects legacy tape/pair-generation/TEST arguments before path access,
uses source-only hygiene independent of training readiness, retains source
lineage and split windows, and writes fresh TRAIN/VAL outputs. The existing
feature reattestation owner may reuse unchanged feature bytes only after its
full exact-contract and source checks pass. Historical targets and policy
checkpoints are not reusable under the corrected return contract.

Recovery-route verification: all 2447 tests passed in one full capped CPU run
(642.560 seconds; zero failures, errors or skips), including the new
source-only hygiene and pre-TEST argument-isolation regressions. All shell
syntax checks, capped `compileall`, and `git diff --check` passed. All 11
installed direct dependency pins match and import as required. Local test
record: `/tmp/gx1-cpu-successor-source-20260905.xml`. This verifies code and
orchestration; actual replacement dataset evidence is recorded above. Runtime
evidence for the successor remains pending.

The corrected TRAIN policy/ranking has now been emitted successfully at
`/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/ENTRY_MODEL_NATIVE_TRAIN_FEATURE_RANKING_20260905T132535Z.json`
(file SHA-256 `147d70cc79cddf35b8a3b95968f6092d0a998f98967e3acf23185e0dec1efe92`).
Its target contract is `entry_causal_m1_outcomes_v2_entry_notional_pnl`;
the new policy SHA-256 is
`208fa387f4054594aa3a38f4f538f828dcfa9d005e5fb158eb0c047d6a77f8a2`.
Feature availability is fitted on 354569 TRAIN rows, excluding 93922 history
rows; all 67 candidate fields remain available. The target ranking contains
313475 complete target rows. These are source/fit populations, not the final
emitted sequence-dataset row counts. VAL, TEST and future rows were not used
for fitting. The selected diagnostic horizon remains 19 M5 bars.

A preparation-only integration defect was also repaired: direct CPU
trainability-readiness used the execution-provenance API, so the review hold
prevented even a non-authorizing source report. The existing owner now shares
all recipe/hash/ancestry/clean-source checks through a read-only source API;
the execution API still checks the hold first. Readiness verifies both smoke
and candidate source closures and keeps every training/activation authority
false. The live numerical ranker/policy code was unchanged during this repair.

The subsequent full CPU suite completed 2457 cases: 2456 passed and the
TERM-ignoring-descendant watchdog case timed out. Its isolated three-case
rerun passed, which does not establish the original timeout's cause. Source
inspection exposed three independently reproduced hazards: waiting
indefinitely when the child exists before its process group, measuring elapsed
time with a realtime clock that can move backwards, and relying on an external
process for PID/group checks and signals. The guard now checks both the owned
PID and group, never waits indefinitely after forced termination, measures
deadlines with Linux uptime and uses Bash's explicit `builtin kill`. Job
control is disabled before `setsid`, and an established group gets one TERM
before the existing grace/KILL. No power, thermal, memory or stop-grace limit
was changed. Regression fixtures demonstrate the hazards independently of
market data; they do not prove a cure for the physical host hang.
Diagnostic records: `/tmp/gx1-cpu-successor-readiness-20260905.xml` and
`/tmp/gx1-watchdog-repro-20260905.xml`.

The complete watchdog confirmation passed 54 cases. After the final builtin
change, all 35 watchdog/attended cases and the one affected static source case
passed again. Exact XML case-identity reconciliation against the 2457-case
fullsuite gives **2462 passing cases, zero unresolved failures/errors/skips**;
this is not a claim that the last fullsuite alone was all-green. Final records:
`/tmp/gx1_watchdog_complete_final_20260905.xml`,
`/tmp/gx1_watchdog_builtin_final_20260905.xml`, and
`/tmp/gx1_watchdog_builtin_static_final_20260905.xml`.
Capped `compileall`, shell syntax and diff whitespace checks also pass.

## Verdict

**LOCAL PRE-CLOUD PACKAGE COMPLETE; EXECUTION REMAINS BLOCKED.** The corrected
code/data/state continuation is committed, source-bound, CPU-migrated and
next-batch equivalent. The 4 September smoke, guarded VAL inference, technical
readiness and candidate gate remain historical evidence and cannot authorize
the successor source. `PROJECT_STATE_xau_direction_launch.json` records an
explicit external-qualification hold;
`scripts/gx1_handover.sh` returns nonzero and explains the hold before resolving
any dataset. Shared execution-provenance validation also rejects any present
hold, including null/malformed values, before direct trainer execution. TEST,
candidate acceptance, paper and live remain unauthorised.

This review was requested by the operator and divided between three agents
(trainer/model, data/causality, cleanup/integration), with the main agent
reviewing gate validation and integrating/validating repairs. No CUDA was
started and no physical TEST input was inspected.

## Confirmed defects and repairs

| Finding | Repair / significance |
| --- | --- |
| Short auxiliary PnL used exit-price notional, whereas lifecycle Exit reward and short MFE/MAE use entry notional. Entry 100 / exit 90 yielded 1111.11 bps versus 1000 bps. | Use entry notional in both causal outcome paths. Version the causal target contract and reject v1 policy/causality evidence. This requires successor TRAIN/VAL target data, not just a recipe rebind. |
| Reconstructing a DataLoader on resume consumed the global Torch CPU RNG even with zero workers. | Private generator for exact-order loaders. Regression exercises real loader iterations across TRAIN and EMA-VAL interruption, comparing future epoch order, model, optimizer, EMA and RNG with uninterrupted execution. |
| Nonfinite gradient norms could reach optimizer/EMA before the next forward detected damaged weights. | Both full and remainder optimizer boundaries fail before optimizer/EMA mutation using finite-gradient clipping. |
| Lifecycle split end was self-attested, not bound to the Entry split's allowed window. | Require hash-bound canonical Entry manifests, exact lifecycle/emission end agreement, row counts and sorted unique observed clocks. Actual retained five-year manifest ends agree; this was a validation hole, not measured leakage. |
| Candidate launch gate could skip malformed trainability bindings and trusted intact outer reports despite changed transitive files. | Require direct exact-recipe readiness, rehash nested evidence, and enforce the existing technical pipeline contract. Legacy readiness cannot substitute for direct pre-TEST readiness. |
| Source closure omitted the signed telemetry query and pre-TEST launcher/gate import chain. | Bind these executable inputs in new recipes. Old source-bound recipes are historical. |
| The wrapper checked the candidate gate, but a direct trainer call did not. | Pass the separate gate binding to the trainer and revalidate it before CUDA/device/training. No recipe/gate circular hash dependency is introduced. |
| Watchdog termination polled only the group leader; a surviving descendant could escape shutdown. | Check the launched process group through TERM/KILL escalation and reject/clean orphaned descendants after leader exit. Limits and grace period are unchanged. |
| Handover/docs simultaneously claimed “gate ready” and “no gate”; emission bounds were described as observed data coverage. | Correct active documentation and record the explicit current hold. Do not rewrite historical immutable reports. |

## What liveness means here

For starting research training, the smoke must demonstrate real inputs,
working active heads, strict bundle loading and positive, dynamic routing
through all eight specialists. It need not have every specialist ranked first.
The retained VAL proof had positive varying weights for all eight; seven were
never top-ranked. That prevents the stronger quality claim, not the narrower
technical connectivity claim. Neither proves edge or profitability.

## Historical dataset diagnosis and completed recovery plan

The old retained five-year dataset has 313399 TRAIN rows and 5509 VAL rows.
Its declared emission windows end 2026-05-31T23:50Z and 2026-06-30T23:55Z;
observed final rows are 2026-05-29T14:50Z and 2026-06-30T14:55Z. Windows are not
assertions of a continuous, unpurged training population.

A capped time-column-only read verified both actual populations are sorted,
unique and disjoint, matching their hash-bound manifests. TRAIN row counts by
calendar year are: 2021: 30511; 2022: 62137; 2023: 63212; 2024: 65210;
2025: 65615; 2026: 26714. The actual bound old policy was rehashed and rejected
by the corrected policy owner with `ENTRY_CAUSAL_M1_TARGET_POLICY_CONTRACT_INVALID`.

The dataset proof binds the old `entry_causal_m1_outcomes_v1` policy through
the frozen policy, ranking and signal lineage. The corrected contract must
not be attached to those old bytes. The recorded recovery plan below has now
completed its CPU steps (1–3 and the recipe/readiness portion of 4); runtime
evidence and the authorized execution transition remain pending:

1. Finish CPU regression verification of the corrected code.
2. Fit a new TRAIN-only target policy/ranking and sizing ECDF, and build a
   successor TRAIN/VAL label dataset using corrected short returns. Keep
   physical TEST closed and keep all old artifacts intact. Preserve registered
   fit windows and split boundaries; do not change them just to pass a gate.
3. Rebind lifecycle to the new parquet hashes and regenerate the affected
   data, causal, reconstruction, liveness and pretrain evidence. Raw M1/M5
   quote sources, MTF feature formulas and model architecture are unchanged.
   The existing feature-surface reattestation route may reuse unchanged
   feature bytes if exact contract equality is proven; do not assume reuse.
4. Bind the frozen reviewed source and successor data into a new recipe and
   readiness chain. Determine exactly which runtime evidence the changed
   dataset requires; do not relabel the old smoke as a new-data smoke.
5. Clear the review hold only with the replacement evidence recorded. Require
   clean handover, fresh signed 160 W telemetry and explicit scoped CUDA
   authorisation before any CUDA execution. Full candidate remains at most
   30 epochs with existing early-stop policy; no architecture change is implied.

The successor used `rebuild_entry_model_native_seq513_dataset.sh` with its
explicit `--pretest-only` route, source-only hygiene and existing capped CPU
owners. This route rejects legacy tape/pair-generation/TEST arguments before
path access. The legacy three-split invocation is not the recovery route and
must not be used. The review hold was not temporarily removed for production.

## Coverage and limitations

- Static AST/import coverage: 411 tracked Python files in the initial tree; no missing internal
  import modules found. All 11 direct installed dependency pins matched.
- Targeted review: data/target generation and train-only fitting; MTF/local
  clocks and feature routing; Entry/Exit/lifecycle; optimizer, EMA, checkpoint,
  selection and resume; immutable launch/evidence binding; signed telemetry,
  capped execution, process cleanup and offline entrypoints.
- All 14 GX1 user services/timers were masked, with no loaded GX1 units observed.
- Kept legacy-named V12/OANDA modules that still have offline consumers, and
  unused model parameters needed for strict historical bundle compatibility.
- The old M5-only sizing fitter is used only by historical test fixtures, not
  the current dataset producer. Its v1 arithmetic is retained with those
  fixtures; current sizing uses the causal M1 owner and cannot admit that
  legacy fit as new training evidence.
- This is not a proof that every possible defect is eliminated. No new full
  five-year epoch, production economics, OOS edge or actual host-hang cure has
  been demonstrated. 160 W is a guarded operating limit, not proof of hardware
  stability.

## Cleanup

Removed only the empty root `=3`, `gx1/features/array_utils.py` and
`gx1/features/rolling_timer.py` after reference/import/dynamic-loader and
source-closure checks. They are recoverable from Git. No model, data, audit,
environment or registered worktree was deleted. Details:
[`REPO_CLEANUP_CANDIDATES_20260903.md`](REPO_CLEANUP_CANDIDATES_20260903.md).

## Initial review verification record (before successor production)

Completed under the 4 GiB / 512 MiB audit cap, with heavy jobs serialized:

- 21 focused causal-outcome/target-policy/execution-causality/sizing tests passed.
- The first completed fullsuite passed 2423 cases; its sole failure was the
  launch-state size guard (14038 bytes against a 14000-byte limit). Stale
  narrative was replaced with the current hold explanation; the limit was
  not relaxed.
- The last fullsuite ran all 2432 cases in 602.15 seconds: 2428 passed and four
  test expectations failed. Three still expected historical V46/text to be
  current after the semantic revocation; one expected the wrong exception
  text in the new old-audit rejection test. These test expectations were
  corrected without weakening production checks.
- A final 109-case confirmation run passed in 34.77 seconds, including all
  four corrected cases and complete affected readiness/causality/handover/
  recipe/direct-launch test files. XML case-identity reconciliation verifies
  **2432 cases passed across the fullsuite and confirmation, zero unresolved
  failures**. This is not a claim that the last fullsuite alone was all-green.
- `compileall gx1 tests`, `git diff --check` and individual `bash -n` checks
  for every top-level shell script passed.
- Actual TRAIN/VAL clock and old-policy rejection checks passed. Handover
  intentionally exits 2 with the named review hold. No CUDA, new dataset build
  or physical TEST access occurred.

Local test logs (diagnostics, not training authority):
`/tmp/gx1-premiere-review-full-20260905.xml`,
`/tmp/gx1-premiere-review-final-20260905.xml`,
`/tmp/gx1-premiere-review-confirm-20260905.xml`.

The early mixed-source baseline was interrupted and is excluded from the
final verification claim. Code review and corrected TRAIN/VAL production are
now complete as recorded above; successor execution evidence and scoped
authorization remain required before large training.
