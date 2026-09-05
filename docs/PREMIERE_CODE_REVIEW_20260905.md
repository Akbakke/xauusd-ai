# Pre-training code review — 2026-09-05

## Authorised CPU continuation

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
orchestration, not yet the replacement dataset or runtime evidence.

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

## Dataset consequences and next work

The retained five-year dataset has 313399 TRAIN rows and 5509 VAL rows.
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
not be attached to those old bytes. Required continuation:

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

The legacy `run_seq513_rebuild_chain_v1.sh` is not this recovery route: it
requires TEST windows and successful training handover. Do not remove the hold
temporarily or run that three-split chain. The successor orchestration must
use the existing dataset builder's `--pretest-only` mode (which rejects TEST
arguments), the existing owners above, fresh output identities and capped CPU
execution. It needs source-hygiene checks independent of old training readiness.

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

## Verification record

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
final verification claim. Code review is complete; corrected TRAIN/VAL data
and successor execution evidence are still required before large training.
