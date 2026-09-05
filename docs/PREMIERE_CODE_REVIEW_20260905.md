# Pre-training code review — 2026-09-05

## Completed CPU successor — 2026-09-05

The authorized CPU goal is complete: corrected TRAIN-only target policy/ranking
and sizing, new TRAIN/VAL datasets and lifecycle, all affected CPU audits,
two new source-bound recipes and direct CPU trainability-readiness.
The active execution hold remains BLOCK; its reason is now
`SUCCESSOR_CPU_READY_REQUIRES_SCOPED_CUDA_AUTHORIZATION_AND_RUNTIME_EVIDENCE`.
No successor CUDA, full candidate training, physical TEST access, candidate
acceptance, paper or live activity occurred. The launch state's retained
`current_source_technical_recipe` and old gate are historical records, not
authority for this successor.

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

**BLOCKED: do not start the five-year CUDA candidate yet.** The 4 September
smoke, guarded VAL inference, technical readiness and candidate gate remain
historical evidence. They do not certify the corrected code and target semantics.
`PROJECT_STATE_xau_direction_launch.json` records an explicit review hold;
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

The successor used the repaired `run_seq513_rebuild_chain_v1.sh` with its
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
