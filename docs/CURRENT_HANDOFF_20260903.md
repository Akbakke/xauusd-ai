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
  Its CPU-only post-run audit is pending; this grants no candidate, TEST,
  promotion, paper or live authority.

## Current host gate

The physical PC was restarted after the prior 3090 host hang. The last proven
Windows sensor setup had a 160 W physical GPU limit and 52 C memory junction,
but that installation evidence was invalidated by the restart. A fresh signed
WSL bridge query to `http://172.30.224.1:38128/gx1/v1/telemetry/` now succeeds
and reported 57 C core, 64 C memory junction, 127.49 W draw, **160 W configured
physical limit**, and 442 MiB VRAM for the expected GPU UUID immediately before
the smoke launch. Both host-safety telemetry gates passed, the guarded smoke
completed with exit code 0, and no trainer is active.

No further CUDA work, including another 31,004-step TRAIN, is authorised. It
remains blocked until both of the following are true:

1. The executable handover and the exact source/recipe preflight pass on a
   clean, reviewed worktree. Re-probe the signed Windows bridge immediately
   before the launch and require its physical-limit field to remain 160 W.
2. The operator explicitly authorises a new CUDA launch. This is intentionally
   separate from this technical result and from any old chat instruction.

The physical-limit change was performed from native elevated Windows PowerShell
and followed by the signed bridge query above. It is a safety-precondition
repair, not CUDA authorisation.

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
