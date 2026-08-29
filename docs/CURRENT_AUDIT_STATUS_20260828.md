# Current audit status — 2026-08-29

This document is the short, current-state override for operational decisions.
It complements the historical design documents; it does not grant execution
authority. `scripts/gx1_handover.sh` remains the executable status owner.

## Current verdict

The project is offline research only. There is no admitted production model,
predictive edge, realised PnL, win rate, MAE/MFE result, calibration,
untouched-TEST result, demo account, paper route or live route.

### Strict pre-full-train override — 2026-08-29

**NO-GO for external full training. Do not rent or start a remote machine.**
The new strict preflight read only the explicit TRAIN and VAL sequence
parquets: 248,028 and 70,880 rows respectively, with no duplicate timestamps,
no timestamp at or after the TEST boundary, and no TEST fit rows in the
TRAIN-only normalization lineage. `TEST ACCESSED: NO` for that preflight.

It also found a critical boundary defect in the current V4 MTF cache: its
metadata declares M5/M15/H1/H4/D1 content through 2026-08-02/04, after the
sealed 2026-07-01 TEST boundary. The standard cache loader verifies every
array byte before serving a VAL window, so it would read TEST-era cache bytes.
Under the strict preflight rule that is not allowed. The earlier 70,880-row
technical VAL prediction artifact therefore remains historical technical
evidence, but is not evidence for this strict preflight.

Do not work around this by slicing, hashing or inspecting the current cache in
the VAL run. First provide a separately immutable, byte-bound five-timeframe
cache whose declared final timestamp is strictly before 2026-07-01, then rerun
the static preflight, full all-head VAL, bundle reload and VAL plumbing. The
current report is
`artifacts/pre_fulltrain_preflight_20260829T143010Z/preflight_manifest.json`.
The candidate checkpoint fixes below are green, but they cannot override the
data-boundary NO-GO.

### Failed-safe reconstruction check — 2026-08-29

An attempted offline reconstruction of a pre-TEST M5 source was also rejected
before any new cache was published. The available M1 reports omit the direct
M5 rollover candle at 22:00 UTC (the first mismatch is 2026-06-11 22:05 in the
M1-derived axis versus 22:00 in the existing M5 cache). Therefore M1 resampling
is not a byte- and axis-equivalent substitute for the direct M5 tape. The
attempted candidate and its rejection note are retained under
`artifacts/pretest_mtf_source_20260829T230000Z/`; neither is training input.
No TEST row was loaded for the check, and no MTF cache was published.

The remaining critical prerequisite is consequently precise: a separately
immutable, direct-M5 artifact ending before 2026-07-01, including the missing
rollover candles, must be made available under the existing offline policy.
Do not work around this with synthetic candles, Parquet filtering of a
cross-boundary row group, or an external machine. Once that artifact exists,
the prefix-cache publisher will recheck every safe timestamp before a full VAL
is allowed.

The narrow research-candidate admission is now technically green, but it is
not an edge or trading admission. Do not start an unpredeclared CUDA job, TEST
evaluation, OANDA activity, collector, Telegram notifier or dashboard. The
collector/dashboard/notifier/self-check services remain deliberately inactive.

The current V46 source data are retained and are not to be rebuilt merely to
obtain another smoke run. Full-input liveness and its entire downstream
report-only chain have been refreshed from the retained immutable TRAIN/VAL
bytes. Historical normalization evidence still predates the repaired Exit MTF
timing contract and is not candidate authority; regenerate only that evidence
if a later, explicit candidate preflight requires it.

There is no persistent historical normalizer selected by the active trainer.
For every future run it fits a new TRAIN-only normalization contract from the
exact source-backed Entry M5 surface, the closed M1 Exit lifecycle and the
validated five-clock V4 cache, before compute sampling or optimization.

## What the current source now proves

- All eight families have exact local/context/MTF routing. Entry consumes local
  M5 and M15/H1/H4/D1 context; Exit consumes local M1 and all five MTF clocks.
  This is a structural/causal proof, not an edge result.
- A candidate export requires selected-checkpoint Entry VAL input-influence
  evidence across individual fields, all local/context families and all Entry
  MTF family routes. It also requires movement in every local and MTF family
  encoder component.
- Exit lifecycle loading recomputes the complete causally eligible Entry-row
  population from the immutable Entry and M1 clocks. A compact episode file
  cannot silently omit a serviceable episode.
- TRAIN normalization now binds the source-backed M5 feature surface and its
  exact sequence-reconstruction audit, in addition to the selected physical
  fit rows. A candidate cannot use an unaudited reconstructed sequence source.
- Recipe source closure now includes every executed Python package
  `__init__.py`. The trainer independently revalidates the byte-bound recipe
  at its own boundary and persists the identical recipe/source map in bundle
  metadata and lock; readiness and five-seed comparison reject omissions or
  split-brain provenance.
- Recipe production now checks the small immutable profile/run-lineage evidence
  before hashing the multi-gigabyte TRAIN/VAL inputs. A stale smoke run ID
  fails before any large-input binding or recipe publication. This is a CPU
  efficiency and provenance control only; it produces neither a bundle nor
  authority to run CUDA.
- From clean source commit `78b00d66`, the fresh CUDA-intended smoke recipe
  passes at
  `train_recipe_audit_current_export_cuda_smoke_20260828T195400Z/`
  `ENTRY_MODEL_NATIVE_SEQ513_TRAIN_RECIPE_AUDIT_20260828T195622479693Z.json`
  (SHA-256 `54548f69325ae79111364e7e4e8e99eec92660a4fe47b217a35fbdc60eda41b8`).
  It binds the current V46 evidence and bounded canonical geometry (batch 8,
  deterministic 32-row smoke, 10 GiB host cap). Runtime now accepts a clean
  descendant only when every byte-bound execution file remains exact, so a
  documentation-only commit no longer forces another large-input rehash.
  Actual export remains a separately authorised guarded operation.
- The separately authorised guarded export completed on 2026-08-29 from
  `train_recipe_audit_guarded_cuda_bundle_export_20260829T001005Z/`
  `ENTRY_MODEL_NATIVE_SEQ513_TRAIN_RECIPE_AUDIT_20260829T001008366250Z.json`
  (PASS). Its atomic bundle is
  `v10_entry_model_native_seq513_guarded_cuda_bundle_export_20260829T001005Z`.
  It completed the full source/normalization preflight, four bounded CUDA
  optimizer steps, strict reload and post-export liveness under the guard
  (66 C / 214.59 W / 8,763 MiB peak). Both bundle metadata and lock bind the
  same current recipe-source provenance. This removes the legacy-bundle
  provenance defect. Its guarded VAL evaluator subsequently wrote an immutable
  70,880-row prediction artifact and report
  `val_predictions_guarded_cuda_bundle_export_20260829T001005Z/`
  `ENTRY_CANDIDATE_SELECTIVE_EDGE_20260829T122405807858Z.json` (technical
  prediction decision `PASS`; preregistered selective-edge decision `FAIL`).
  The CPU-only smoke-bundle audit then recorded finite, normalized and
  state-varying routes for all eight specialists. Three families were never
  the largest softmax weight in this four-step smoke, so its strict
  post-training-quality decision is correctly `FAIL`; it is not a missing,
  constant or disconnected feature route. The narrower technical admission
  contract independently verified positive, non-constant routing for every
  family and emitted
  `candidate_readiness_guarded_cuda_bundle_export_20260829T001005Z/`
  `ENTRY_CANDIDATE_READINESS_20260829T122708592909Z.json` with
  `READY_FOR_CANDIDATE_TRAINING`. That event binds the current bundle, all
  70,880 VAL rows, 238 signal fields, all eight specialists and all five
  MTF clocks. It keeps activation, shadow/paper/demo/live promotion and
  economic authority false.
- The offline champion/challenger review loop is rebuilt and regression-tested
  (`tests/test_entry_offline_challenger.py`: four passing tests). It accepts
  only two immutable, independently completed rolling-OOS candidate-result
  events for the exact same unseen window and economic contracts, then reports
  deltas for human review. It cannot train, schedule, change weights, select a
  winner automatically or activate a model. See
  `docs/OFFLINE_CHAMPION_CHALLENGER_V1.md`. No synthetic OOS events were
  manufactured for this verification.
- The future serve adapter now opens only the exact V4 cache bound by the
  bundle, injects its frozen artifacts directly, and refuses a context snapshot
  from another immutable pair generation even if its timestamp is identical.
  Serve-parity evidence records the actual cache and full pair component hashes.
- The current offline boundary rejects direct paper-runner recovery and direct
  OANDA history ingestion before credentials, network, state reconciliation or
  writes. This preserves future code without granting present authority.
- The TEST evaluator is hard closed. A future one-time immutable TEST-release
  authority is required before any TEST file can be opened. This is intentional
  and does not block TRAIN/VAL audit work.
- The deep source audit traced the retained V46 feature chain end to end. Its
  latest liveness evidence covers 310 qualified local fields across all eight
  families on both native clocks (M5 Entry and M1 Exit), and the V4 cache
  covers all five MTF clocks with no constant field or exact duplicate pair.
  This proves wiring and variation, not predictive edge.
- The direct M1 fallback now derives the level/trendline registry clock from
  its declared local timeframe. Canonical V46 already supplied its exact M1
  layers, so this closes a latent fallback defect without rebuilding data.
- Report-only selective-edge summaries now read the emitted pre-registered
  metric scope and preserve boolean pass/fail values. This repairs reporting;
  it creates neither an edge claim nor TEST authority.
- CUDA-capable non-trainer paths now fail closed unless they are inside the
  capped/guarded producer scope; proof-only smoke-bundle audits and the
  synthetic Exit benchmark are CPU-only. Evidence cleanup checks for an open
  writer before moving any target into quarantine.
- A fresh V46 full-input-liveness v10 scan passed on 2026-08-28. It scanned
  all 248,028 TRAIN and 70,880 VAL rows, checked all 7,286,409,984 sequence
  values for finiteness and exact last-step/snapshot parity, and revalidated
  the complete five-clock MTF cache. It did not open TEST.
- The downstream V46 post-rebuild, smoke-manifest, smoke-readiness and
  trainability reports were then re-bound to that v10 scan. Their 6, 26, 6 and
  32 checks respectively passed with no failures. The final CPU-only recipe
  preflight rehashed the actual large TRAIN/VAL files and passed, while keeping
  TEST byte-opaque, CUDA unused and all trainer side effects false.
- Targeted normalization, Entry/Exit shared-base, event/parity and recipe
  regression tests also pass. Therefore there is no separate stale
  normalization artifact to rebuild before the first explicitly authorized
  learning-validation run.
- A CPU recipe carrying a chronological time window was found during the
  wrapper dry-run check. The wrapper correctly reserves such a window for the
  attended CUDA lane; the recipe contract now rejects that invalid CPU
  combination before publication, with a regression test. The current CPU
  recipe has no time window and is hash-bound to the repaired source.
- The explicitly authorised guarded learning-validation session reached its
  terminal state without a restart: the immutable resume pointer is
  `complete=true` at 4,037/4,037 batch-8 optimizer steps over the exact
  `[2024-12-01T00:00:00Z, 2025-06-01T00:00:00Z)` 32,289-row TRAIN window.
  Its terminal report is `PASS_TECHNICAL_INTEGRATION_NOT_EDGE`: all ten joint
  tasks recorded supervision and gradients, and model parameters moved. It
  wrote no bundle and ran no VAL, TEST, OOS, PnL, win-rate, MAE/MFE or trading
  step. The report also correctly keeps candidate authority false: ten Exit
  feature gates were exactly neutral because their correctly routed H4/D1 raw
  fields were constant over this deliberately narrow six-month window. The
  retained full-V46 liveness proof, not artificial gate noise, establishes
  whether those fields vary across candidate training; a fresh full-candidate
  gate audit remains required.
- A current CPU-only candidate-readiness recheck then correctly returned
  `NOT_READY_FOR_CANDIDATE_TRAINING`. The historical diagnostic smoke bundle
  has no `recipe_source_provenance` in its metadata or lock, which is now an
  exact candidate requirement. Its input, prediction-evidence, wrapper and
  eight-specialist checks remain green; this is a provenance block, not a
  feature, target or gradient-path failure. Do not relabel or patch the old
  bundle: a fresh, fully exported bundle from the current source contract is
  required before candidate training can be considered. The historical
  60-step smoke's separate strict-quality `FAIL` (some families were never
  top-ranked) is deliberately not a candidate-start condition: the actual
  admission contract requires eight finite, positive, non-constant routes,
  not a tiny smoke to prove market edge. A regression test now proves both
  this technical/quality separation and that the legacy provenance condition
  fails closed.

Recent safety/source commits: `34659e36`, `c3b67b6f`, `6ee59296`, `f2d4862b`.
The current V46 report pointers in `PROJECT_STATE_xau_direction_launch.json`
bind the new v10 evidence chain. No dataset rebuild, CUDA, broker or TEST
operation was started while refreshing it.

## Required next sequence

1. Preserve the historical provenance block; do not repair or relabel the old
   bundle. The fresh, current-provenance bundle has now separately passed
   technical candidate-readiness. Before a candidate CUDA launch, materialize
   its exact immutable recipe and pass its no-GPU dry-run; do not turn this
   readiness event into an unbounded or unattended run.
2. Candidate training remains offline research only. Its later rolling-OOS
   result must use a frozen cost/replay contract; only then can the prepared
   champion/challenger review loop compare two candidates. It cannot promote
   either one automatically.
3. The active normalizer has been confirmed to fit current TRAIN-only source
   inputs at run time; do not rebuild V46 or manufacture a replacement
   normalization artifact.
4. Inspect the terminal technical-epoch report and guard logs. They prove live
   task/gradient paths and safe bounded execution, but not predictive
   performance. Do not portray them as VAL, backtest or edge evidence.
5. Do not resume the completed technical epoch. The next model job, if
   separately launched, is the declared candidate recipe rather than another
   exploratory smoke; TEST stays sealed.
6. Keep TEST sealed until a single candidate has passed VAL and an immutable
   release event is designed and reviewed. Demo/OANDA comes only after
   backtests and the separate executable-economics/risk gates.

## Safety boundary

Every audit/test runs through `scripts/gx1_capped_run.sh` under the audit cap.
No CUDA or producer job is authorised by this document. The host GPU power
limit is not a safety control from WSL; any future CUDA work must use the
existing attended telemetry guard and be manually observed.
