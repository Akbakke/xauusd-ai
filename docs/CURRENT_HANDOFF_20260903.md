# GX1 current re-entry — 2026-09-03

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

### Five-year pre-TEST candidate preparation (2026-09-03)

- A new TRAIN/VAL-only dataset has been materialized under
  `V9_FIVE_YEAR_PRETEST_BOUNDARYFIX_20260903T155921Z/`
  `FIVE_YEAR_DATASET_20260903T170055Z`. Its immutable split is:
  TRAIN `2021-06-01T00:00:00Z` through
  `2026-05-31T23:50:00Z` (313,399 rows), followed without a missing M5 bar by
  VAL `2026-05-31T23:55:00Z` through `2026-06-30T23:55:00Z` (5,509 rows).
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
- The active offline candidate recipe is the 2026-09-04 source rebind:
  `ENTRY_V9_FIVE_YEAR_CANDIDATE_20260904T201433Z_RECIPE.json`, SHA-256
  `e660167098c8bbd1bb33261051324341a3da547cb7289f5ac359c635c9905ae5`.
  It is `report_only`, has `activation_authority=false`, and binds source
  commit `b414fc36`. It specifies batch 8, **at most 30 epochs**, validation
  every epoch, patience 5, minimum epoch 1, and `save_top_k=1`.
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
- The current direct V9 readiness is
  `ENTRY_PRETEST_TRAINABILITY_READINESS_20260904T201527266562Z.json`, SHA-256
  `7d97f3651fdf2860244dd75a38b6eab9db98183280eec8b4c3c6d93db8cfa6b5`.
  It rehashes the exact smoke/candidate recipes and all direct TRAIN/VAL
  bindings against the new zero-failure five-year pretrain audit. It remains
  non-authorizing (`candidate_training_allowed=false`).
- A gate still cannot be materialized: the smoke-bundle contract additionally
  requires immutable VAL-only prediction evidence from this exact bundle. That
  is one separate guarded CUDA inference pass, not a retraining pass. It needs
  a separate explicit operator authorization.
- Do not substitute `model-native-attended-hardware-smoke`: that route is a
  synthetic architecture diagnostic with no dataset, bundle or candidate
  output, and therefore cannot produce the smoke audit required by a
  candidate gate. The exact-data canonical smoke above needs separate explicit
  CUDA operator authorisation before `--execute`.

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
- The older candidate-readiness and launch-gate artifacts bind the old dataset
  and do **not** authorize this five-year recipe. No new five-year candidate
  gate, candidate run, TEST evaluation, acceptance, promotion, paper, broker
  or live action has been created.
- Before any future CUDA request, require a fresh clean-worktree handover,
  fresh signed 160 W telemetry, explicit operator authorization, and a
  separately reviewed gate for this exact recipe. This handoff is not that
  authorization.

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
  Its CPU-only post-run audit is `FAIL`: three specialist gates were never
  top-ranked in the small smoke sample. The technical pipeline, inputs and
  hashes remain proven, so immutable candidate readiness is
  `READY_FOR_CANDIDATE_TRAINING` and a hash-bound candidate launch gate passed
  CPU-only dry-run. This is preparation only, not candidate acceptance or CUDA,
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
immediately before any proposed launch; no trainer is active.

No further CUDA work, including another 31,004-step TRAIN, is authorised. It
remains blocked until both of the following are true:

1. The executable handover and the exact source/recipe preflight pass on a
   clean, reviewed worktree. Re-probe the signed Windows bridge immediately
   before the launch and require its physical-limit field to remain 160 W.
2. The operator explicitly authorises a new CUDA launch. This is intentionally
   separate from this technical result and from any old chat instruction.

The physical-limit change and signed bridge verification are
safety-precondition repairs, not CUDA authorisation. The next non-CUDA action
is to materialize a five-year candidate gate only if its exact preflight
artefacts exist; old smoke/readiness evidence is dataset-bound to a different
surface and must fail closed.

## Relevant immutable paths

All paths below live under
`/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/`:

- V9 recipe: `V9_ONE_EPOCH_CANDIDATE_20260901T213444Z_RECIPE.json`, SHA-256
  `61f90a5eed4a1b21f87e96770d43fecba7978a42b3e305c57ff064ed645cf9b1`.
- V9 active-session directory:
  `.gx1-candidate-training-session.ENTRY_V9_ONE_EPOCH_CANDIDATE_20260901T213444Z_BUNDLE`.
- V9 published bundle:
  `ENTRY_V9_ONE_EPOCH_CANDIDATE_20260901T213444Z_BUNDLE`.
- Current-source recipe and executed 32-row technical-smoke bundle:
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
