# GX1 current re-entry — updated 2026-09-05

This is the short human handoff after a lost chat, reboot, or context reset.
It is an index, not execution authority. The executable authority is
`bash scripts/gx1_handover.sh`; `GX1_RULES.md` remains binding.

## Read this first

```bash
cd /home/andre2/src/GX1_ENGINE
git status --short --untracked-files=all
bash scripts/gx1_handover.sh --check
bash scripts/gx1_handover.sh
```

Do not infer authority from an old checkpoint, a terminal scrollback, or a
run-directory timestamp. Do not start TRAIN from this document.

## Current truth

Exactly one authorized successor canonical gate-smoke completed on 2026-09-05.
The process and watchdog exited 0, and the immutable bundle passed strict
reload/publication and a separate CPU commit/provenance verification. This is
technical smoke evidence only, not candidate acceptance or an edge result.

The run used batch 8, one epoch and 32 deterministic TRAIN / 32 VAL rows from
the unchanged five-year dataset. All ten joint tasks had supervision,
gradients and parameter movement. The existing technical smoke admission
passed; strict candidate-quality checks did not all pass: sparse
`trendline_event` support and two constant Exit cooperation-gate indices
(166, 175) were reported. Small-sample selected VAL PnL was
`-15.70416259765625` bps, not a backtest or production result.

Current smoke recipe: `ENTRY_V9_FIVE_YEAR_HANDOVER_GATE_SMOKE_20260905T153048Z_RECIPE.json`,
SHA-256 `cd4c65c68547e1d985b0864eaa53c51a86c9ddeb5ca07769006620b363e12de7`.
Source: `e25a8cb6377eec496d1863c667c0d6cff785d86a`.
Bundle: `/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/ENTRY_V9_FIVE_YEAR_HANDOVER_GATE_SMOKE_20260905T153048Z_BUNDLE`.
Bundle commit SHA-256: `8a9b197b805e6c373f0e17a3cf4bf0f7559cd43936431ce03cb325ce3c7bfc5d`.
Exact sidecar paths and remaining bindings are in
[the review/execution report](PREMIERE_CODE_REVIEW_20260905.md).

The watchdog's terminal aggregate records 311 verified samples, no stop/fatal
event, and peaks of 61 C core, 60 C memory junction, 158.56 W draw and 9458 MiB
VRAM; the physical limit was 160 W. These are the full-run guard aggregates,
not just the less frequent heartbeat snapshots.

The execution hold is restored:
`SUCCESSOR_CPU_READY_REQUIRES_SCOPED_CUDA_AUTHORIZATION_AND_RUNTIME_EVIDENCE`.
The selected recipe status is
`EXECUTED_TECHNICAL_SMOKE__POSTRUN_AUDIT_PENDING__NO_CANDIDATE_AUTHORITY`.
Do not repeat this smoke or launch full training automatically. The next
separately authorized runtime step is guarded VAL prediction from this exact
bundle, followed by its post-run audit/readiness and candidate gate. TEST,
candidate acceptance, paper and live remain unauthorized.

New CPU readiness remains report-only, with all five checks passing:
`/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/ENTRY_V9_FIVE_YEAR_HANDOVER_TRAINABILITY_20260905T153048Z/ENTRY_PRETEST_TRAINABILITY_READINESS_20260905T153150593398Z.json`,
SHA-256 `8489d179562a3cd2d04b8bf4f864fbfe5f3fecad0551ace81b6e5f5d0269bbaf`.
The paired candidate recipe is unexecuted:
`/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z/ENTRY_V9_FIVE_YEAR_HANDOVER_CANDIDATE_20260905T153048Z_RECIPE.json`,
SHA-256 `7923b9ea638de50ffb06a82a7f601bec96f86f96a8f5f4ab89122cbccc4f094f`.
Both recipes bind the unchanged successor TRAIN/VAL artifact set.

### Completed CPU preparation before handover rebind

The [2026-09-05 review report](PREMIERE_CODE_REVIEW_20260905.md) records the
repairs, actual production evidence and exact artifact hashes. The corrected
TRAIN-only policy/ranking/sizing and five-year TRAIN/VAL data have been rebuilt;
all affected CPU audits pass. Actual populations are 313399 TRAIN and 5509 VAL
rows, with original split boundaries preserved and physical TEST unopened.
The contract is `entry_causal_m1_outcomes_v2_entry_notional_pnl`; old fitted
policy/target bytes were not relabelled. Model architecture, feature formulas
and candidate configuration remain unchanged.

The successor root is:
`/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/V9_FIVE_YEAR_ENTRY_NOTIONAL_20260905T132535Z`.
Under that root:

- Dataset: `dataset/`, with passing full-population pretrain, feature/target,
  specialist, causality, reconstruction, liveness and lifecycle verification.
- Canonical smoke recipe:
  `ENTRY_V9_FIVE_YEAR_ENTRY_NOTIONAL_GATE_SMOKE_20260905T144446Z_RECIPE.json`,
  SHA-256 `cafabf1f7b87a3feb630b58a1b78ffc53d027129d60d41c7fe88f643b68d4476`.
- Candidate recipe:
  `ENTRY_V9_FIVE_YEAR_ENTRY_NOTIONAL_CANDIDATE_20260905T144446Z_RECIPE.json`,
  SHA-256 `e1441e9a5fbe693b348a27ff5b13dfb251830c5e72433281d3c2409c87e779e0`.
  It retains batch 8, at most 30 epochs and early-stop patience 5.
- CPU readiness:
  `ENTRY_V9_FIVE_YEAR_PRETEST_TRAINABILITY_20260905T144558Z/ENTRY_PRETEST_TRAINABILITY_READINESS_20260905T144627440261Z.json`,
  SHA-256 `c577b80cd2bd15f527640a105e20fceae4b1e8fd660f67b949d0a7811d2c2a86`.
  Decision `READY_FOR_PRETEST_CANDIDATE_TRAINABILITY_REVIEW`; all five checks
  pass, with candidate training and activation explicitly false.

These two preparation recipes bind source `406c732e` and the same new artifact
set. Neither executed; both were superseded by the `e25a8cb6` handover rebind
above. No full five-year candidate has started.
The exact September 4 smoke, guarded VAL pass and candidate gate remain
historical. Their old source/data cannot certify this successor.

Verification includes a 2447-case all-green recovery-route fullsuite, then a
2457-case integration suite with one watchdog timeout and passing focused
confirmations after repairs. Exact case-identity reconciliation gives 2462
passing cases and no unresolved failures; the last fullsuite alone was not
all-green. The report preserves that distinction and its XML records.

During the completed CPU phase, normal handover deliberately exited 2 while
the execution hold was active. That hold was cleared only for the one
explicitly authorized smoke and has now been restored. It never authorized
full candidate or separate VAL execution.
`bash scripts/gx1_handover.sh --source-only` checks source hygiene without
authorizing training or consulting old runtime evidence. Do not remove the
hold merely to obtain a green normal handover.

### Historical five-year pre-TEST evidence completed on 2026-09-04

- A new TRAIN/VAL-only dataset has been materialized under
  `V9_FIVE_YEAR_PRETEST_BOUNDARYFIX_20260903T155921Z/`
  `FIVE_YEAR_DATASET_20260903T170055Z`. Its declared emission windows are
  TRAIN `2021-06-01T00:00:00Z` through `2026-05-31T23:50:00Z` and VAL
  `2026-05-31T23:55:00Z` through `2026-06-30T23:55:00Z`. These are window
  bounds, not the first/last timestamps actually emitted. TRAIN has 313,399
  rows with observed maximum `2026-05-29T14:50:00Z`; VAL has 5,509 rows with
  observed maximum `2026-06-30T14:55:00Z`. Do not infer a gap-free population
  from adjacent declared windows; market closures and target-completeness
  rules affect the emitted rows.
  The physical TEST boundary remains `2026-07-01T00:00:00Z` and was not read.
- The new M1 and M5 shared feature surfaces, cross-surface audit, full-input
  liveness, foundation feature/target audits, specialist audit, execution
  causality audit, and both source-reconstruction audits all report `PASS`.
  The dataset CPU preflight also passed with `test_accessed=false`.
- The unopened-TEST guard is
  `ENTRY_MODEL_NATIVE_PRETEST_TEST_GUARD_20260903T173738Z.json`, SHA-256
  `7d69d20f1ce50538cddfc18c438c4f0965c223d272ba05bae43ec9bc7e85ed0b`.
  Its control-plane proof reports no TEST dataset/manifest/metric bytes read
  and no TEST paths resolved or statted.
- The historical offline candidate recipe validated on 2026-09-04 is:
  `ENTRY_V9_FIVE_YEAR_CANDIDATE_20260904T201433Z_RECIPE.json`, SHA-256
  `e660167098c8bbd1bb33261051324341a3da547cb7289f5ac359c635c9905ae5`.
  It is `report_only`, has `activation_authority=false`, and binds source
  commit `b414fc36`. It specifies batch 8, **at most 30 epochs**, validation
  every epoch, patience 5, minimum epoch 1, and `save_top_k=1`. Its old source
  hashes do not match the 2026-09-05 review repairs. The successor recipe above
  now binds the rebuilt short-return target/policy chain and fresh CPU audits;
  it does not inherit this old recipe's execution evidence.
- `ENTRY_V9_FIVE_YEAR_CANDIDATE_CHECKPOINT_POLICY_PROOF_20260903T174100Z.json`
  is `PASS`: an improving synthetic run reaches epoch 30; five consecutive
  non-improvements after epoch 1 stop at epoch 6. This is a policy proof, not
  a training result.
- The old candidate-readiness and smoke-audit pair was deliberately passed to
  the five-year gate materializer on 2026-09-04 and rejected as incompatible;
  it is bound to `PRETEST_DATASET_V4_20260829T224438Z`, not this dataset, and
  its smoke audit is `FAIL`. No gate output was written. The correct successor
  is the immutable, exact-data recipe
  `ENTRY_V9_FIVE_YEAR_GATE_SMOKE_20260904T124806Z_RECIPE.json`, SHA-256
  `2cab27cd57100553e3a9d0b741310c7726a25d840b9c234ea237b686e1fc057f`.
  Its guarded launcher dry-run is `PASS`, with `test_accessed=false`; it binds
  a canonical CUDA smoke of one epoch, batch 8 and 32 deterministic rows.
  It was executed once on 2026-09-04 after fresh signed 160 W telemetry. The
  immutable bundle is `ENTRY_V9_FIVE_YEAR_GATE_SMOKE_20260904T124806Z_BUNDLE`,
  with bundle-commit SHA-256
  `85401bebf8b8571c19db91964aba737636518c8cb3ebf0e3a9a82a28fea4fce8`.
  The guarded one-epoch run exited 0; it touched TRAIN/VAL only and did not
  create TEST, candidate, paper or live authority.
- The 2026-09-04 direct V9 trainability readiness is
  `ENTRY_PRETEST_TRAINABILITY_READINESS_20260904T201527266562Z.json`, SHA-256
  `7d97f3651fdf2860244dd75a38b6eab9db98183280eec8b4c3c6d93db8cfa6b5`.
  It rehashes the exact smoke/candidate recipes and all direct TRAIN/VAL
  bindings against the new zero-failure five-year pretrain audit. It remains
  non-authorizing (`candidate_training_allowed=false`).
- One separately authorized guarded CUDA VAL prediction pass completed from
  that exact bundle on 2026-09-04 with exit 0. It wrote
  `ENTRY_V9_FIVE_YEAR_GATE_SMOKE_20260904T124806Z_VAL_PREDICTIONS/`
  `ENTRY_CANDIDATE_SELECTIVE_EDGE_20260904T202209781302Z.json`, reporting
  `PASS`. The signed watchdog recorded 63 samples, at most 60 C core,
  66 C memory junction, 161.17 W draw and approximately 1.05 GiB VRAM.
- Its immutable post-run smoke audit remains `FAIL` on seven families never
  taking the largest specialist gate weight. This is the strict top-rank
  quality heuristic. Technical liveness requires finite, positive,
  state-varying routes through all eight specialists and the contract's
  training-connectivity evidence; it does not require all eight to win top
  rank. Calling that failure a proven dead model path was incorrect. Neither
  liveness nor the top-rank histogram proves economic edge.
- Technical candidate readiness was materialized as
  `ENTRY_CANDIDATE_READINESS_20260904T202543050922Z.json`, SHA-256
  `a5db7a4ca0c4bb62976b2dc6805db451b70bde47900f53483a6b2375521591a5`.
  The launch gate
  `ENTRY_PRETEST_CANDIDATE_LAUNCH_GATE_20260904T202913201537Z.json`, SHA-256
  `25d76042cbac9afaa2ec36cc8e71a59db343164a73c17460babb4364bb517ad7`,
  was READY and passed CPU dry-run at `55de7b82`. These historical artifacts
  remain immutable; their readiness must not be carried across the current
  source changes or the corrected short-return contract without the affected
  dataset rebuild, new bindings and verification.
- Do not substitute `model-native-attended-hardware-smoke`: that route is a
  synthetic architecture diagnostic with no dataset, bundle or candidate
  output, and therefore cannot produce the smoke audit required by a
  candidate gate. The exact-data canonical smoke and VAL pass above have
  already run; do not automatically repeat either during this CPU review.

### Resource and authority boundary for the new recipe

- Observed CPU-only preparation: the actual dataset build completed under the
  16 GiB cap with approximately 5.2 GiB observed process RSS and produced
  approximately 7.5 GiB of TRAIN+VAL parquet. At preparation time, 840 GiB of
  disk was free. These are preparation measurements, not CUDA throughput.
- The five-year TRAIN set has 313,399 rows: 39,175 optimizer steps per epoch
  at batch 8, or at most 1,175,250 steps across 30 epochs. Actual CUDA
  wall-time remains intentionally unestimated: no full five-year CUDA epoch
  has run, and the historical VAL timing was taken before the present 160 W
  operating rule.
- The older pre-five-year readiness and gate bind a different dataset and
  cannot substitute for the five-year evidence. A five-year gate now exists,
  but its recipe/source binding predates the 2026-09-05 repairs. No full
  five-year candidate, TEST evaluation, acceptance, promotion, paper, broker
  or live action has been started by this preparation.
- Before any future CUDA request, require a fresh clean-worktree handover,
  fresh signed 160 W telemetry, explicit operator authorization, and a
  separately reviewed gate for this exact recipe. This handoff is not that
  authorization.

### Historical V9 one-epoch technical result

- V9 (`V9_ONE_EPOCH_CANDIDATE_20260901T213444Z`) completed one full technical
  epoch: 248,028 TRAIN rows / 31,004 optimizer steps, then 70,880 VAL rows /
  8,860 batches. Its terminal state is `phase=validation`, `complete=true`,
  `global_optimizer_steps=31004`, with state SHA-256
  `e3c10500549656456765ee6fe32f0022feb3612682ba3c3652b9a43e2460a371`.
- The immutable technical bundle is
  `ENTRY_V9_ONE_EPOCH_CANDIDATE_20260901T213444Z_BUNDLE`. Selected checkpoint:
  `top_k/epoch_0001.pt`, SHA-256
  `65de701e8787f160ab9e09ff587984f7110661940695632a9bf8d8ec4c972a2d`.
  Bundle commit SHA-256 is
  `87dcb4fd55c5ab7a91de5043b99f18feece8f5b545f715d4779c3002dafff224`.
- V9 is a **technical TRAIN+VAL result only**, not an accepted candidate. Its
  selection monitor was
  `entry_policy_realized_gross_spread_inclusive_pnl_bps_mean=-0.6577958464622498`
  bps. TEST remains unread; candidate acceptance, promotion, paper, broker and
  live authority are all false.
- Two post-run control defects were fixed and tested in `98cf85b8`: terminal
  `complete=true` / `phase=validation` handling in the epoch seal, and the
  comparison of per-side Exit evidence with an incorrectly combined population.
  A fresh-source CPU-only launch dry-run then passed at `98cf85b8`. After a
  clean handover/preflight and fresh signed 160 W telemetry, exactly one
  canonical 32-row technical CUDA smoke was executed. It published
  `ENTRY_V9_POSTRUN_SOURCE_REBIND_20260903T013249Z_BUNDLE`, bundle-commit
  SHA-256 `d5026848d1637363351d821f837ea781cb1235c1ba04929517013c358623e92e`.
  Its CPU-only post-run audit was `FAIL`: three specialist gates were never
  top-ranked in the small smoke sample. The technical pipeline, inputs and
  hashes passed the then-current technical contract, so candidate readiness was
  `READY_FOR_CANDIDATE_TRAINING` and a hash-bound candidate launch gate passed
  CPU-only dry-run for that historical recipe. This is preparation only, not candidate acceptance or CUDA,
  TEST, promotion, paper, broker or live authority.

## Current host gate

The physical PC was restarted after the prior 3090 host hang, invalidating the
old signed 160 W response. The temporary 390 W state was repaired on
2026-09-04 with elevated native Windows sensor setup: it set 160 W and
installed the `GX1GpuPowerLimit` SYSTEM task, which reapplies and verifies the
cap at boot and every 15 minutes. A fresh source-bound nonce/RSA bridge query
then returned `52,56,39.05,160,392` (core C, memory-junction C, draw W,
physical limit W, VRAM MiB) for the expected GPU UUID. It proves signed
telemetry and the 160 W host prerequisite at that instant, but is not a
candidate gate or CUDA authorisation. Obtain a new signed bridge response
immediately before any proposed launch. The review itself starts no trainer.

Today's watchdog stops above 65 C core, 80 C memory junction, 160 W configured
physical limit, 170 W actual draw or 12 GiB resident VRAM, and fails closed on
missing/invalid signed telemetry. The persistent Windows cap and the one-second
process watchdog serve separate purposes. Historical 70 C / 220 W runs do not
define current limits.

The corrected CPU target/policy/audit chain and new recipes/readiness are now
complete, and the one authorized canonical smoke has completed. No further
CUDA work is authorized by that result. Remaining execution sequence:

1. Do not repeat the completed smoke. Obtain new explicit scoped authorization
   for guarded VAL predictions from its exact bundle. Record the reviewed hold
   transition and require clean executable handover/exact launch preflight.
   Re-probe signed Windows telemetry immediately before launching and require
   the physical-limit field to remain 160 W.
2. Complete successor post-run audit/readiness and materialize the new candidate
   gate from that exact source/data/bundle. Do not substitute the historical
   September 4 gate or treat strict quality warnings as resolved by exit 0.
3. Full candidate launch needs the successor runtime gate, clean launch
   preflight, fresh signed 160 W telemetry and explicit full-candidate authority.

The hold stays active until an authorized, reviewed execution transition;
source-only hygiene and CPU readiness do not clear it. Preserve old immutable
artifacts, and never overwrite hashes or relabel old execution as new-source
execution. TEST, candidate acceptance, paper and live remain unauthorized.

## Historical one-epoch immutable paths

All paths below live under
`/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/`:

- V9 recipe: `V9_ONE_EPOCH_CANDIDATE_20260901T213444Z_RECIPE.json`, SHA-256
  `61f90a5eed4a1b21f87e96770d43fecba7978a42b3e305c57ff064ed645cf9b1`.
- V9 active-session directory:
  `.gx1-candidate-training-session.ENTRY_V9_ONE_EPOCH_CANDIDATE_20260901T213444Z_BUNDLE`.
- V9 published bundle:
  `ENTRY_V9_ONE_EPOCH_CANDIDATE_20260901T213444Z_BUNDLE`.
- Historical source recipe and executed 32-row technical-smoke bundle:
  `V9_POSTRUN_SOURCE_REBIND_20260903T013249Z_RECIPE.json`, SHA-256
  `570a4baefb999d406f5d39b994bbed9a408244409ce9448e44fbc3e425c40372`; bundle
  `ENTRY_V9_POSTRUN_SOURCE_REBIND_20260903T013249Z_BUNDLE`, commit-manifest
  SHA-256 `26113018d79efe3075a9d1e8c1e87dbedb74fa8adb5207aa9c46e0d4c27e2ee9`.
- Post-run audit:
  `V9_POSTRUN_SOURCE_REBIND_20260903T013249Z_SMOKE_POSTRUN_AUDIT_20260903T102800Z/ENTRY_MODEL_NATIVE_SMOKE_BUNDLE_AUDIT_20260903T102647715422Z.json`,
  SHA-256 `812df4844a3dc485ca1d4f562a9de340f92d9c45be2eb98ae32e1805672a6756`.
- Frozen candidate recipe and launch gate (both unexecuted):
  `V9_POSTRUN_CANDIDATE_20260903T102900Z_RECIPE.json`, SHA-256
  `2983d413b2324be4e471461153d5f8ff59a35348281e1baac359cd7ef4153ccc`; gate
  `V9_POSTRUN_CANDIDATE_20260903T102900Z_LAUNCH_GATE/ENTRY_PRETEST_CANDIDATE_LAUNCH_GATE_20260903T102817399764Z.json`,
  SHA-256 `14d42cc6de1665ab9e18f11c7d750078eab84576cd77d9dbe765257cc02c66e7`.

Keep these immutable artifacts. They are evidence, not disposable cache.

## Authority map

| Question | Owner |
| --- | --- |
| What is the live verified session and source closure? | `scripts/gx1_handover.sh` |
| What is allowed? | `GX1_RULES.md` |
| How should an agent work safely? | `AGENTS.md` and `CLAUDE.md` |
| What did V9 prove? | this file and `HANDOVER_XAU_DIRECTION_REPAIR_20260714.md` |
| How is the Windows telemetry bridge required? | `docs/CANONICAL_HOST_GPU_TELEMETRY_BRIDGE_CONTRACT.md` |
| What may later be removed? | `docs/REPO_CLEANUP_CANDIDATES_20260903.md` |

Every other Markdown file is either a contract, design record, audit, or
historical evidence. It may constrain work, but it does not override this
re-entry state or authorise a command.
