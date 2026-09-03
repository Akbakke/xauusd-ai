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
  A fresh-source CPU-only launch dry-run then passed at `98cf85b8`; it did not
  execute CUDA or create a new bundle.

## Current host gate

The physical PC was restarted after the prior 3090 host hang. The last proven
Windows sensor setup had a 160 W physical GPU limit and 52 C memory junction,
but that installation evidence was invalidated by the restart. A fresh signed
WSL bridge query to `http://172.30.224.1:38128/gx1/v1/telemetry/` now succeeds
and reported 56 C core, 64 C memory junction, 127.93 W draw, **390 W configured
physical limit**, and 409 MiB VRAM for the expected GPU UUID. The bridge is
therefore healthy, but the 160 W physical limit is not; no trainer is active.

New CUDA work, including another 31,004-step TRAIN, is blocked until all of the
following are true:

1. The signed Windows bridge continues to return a fresh valid response from
   the configured WSL endpoint, including numeric core temperature, memory
   junction, power and VRAM for the expected RTX 3090 UUID.
2. The physical GPU limit is set to, and freshly verified as, exactly 160 W
   after the restart. The last signed response reports 390 W, so this gate is
   currently failing.
3. The executable handover and the exact source/recipe preflight pass on a
   clean, reviewed worktree.
4. The operator explicitly authorises a new CUDA launch. This is intentionally
   separate from this technical result and from any old chat instruction.

The physical-limit change must be performed from native elevated Windows
PowerShell, not from WSL, using the reviewed installer with
`-Install -SetPowerLimitWatts 160`; immediately follow it with a new signed
bridge query. That administrative action is a safety-precondition repair, not
CUDA authorisation.

## Relevant immutable paths

All paths below live under
`/home/andre2/GX1_DATA/data/data/prebuilt/V46_20260825T170935Z_CHAIN/artifacts/`:

- V9 recipe: `V9_ONE_EPOCH_CANDIDATE_20260901T213444Z_RECIPE.json`, SHA-256
  `61f90a5eed4a1b21f87e96770d43fecba7978a42b3e305c57ff064ed645cf9b1`.
- V9 active-session directory:
  `.gx1-candidate-training-session.ENTRY_V9_ONE_EPOCH_CANDIDATE_20260901T213444Z_BUNDLE`.
- V9 published bundle:
  `ENTRY_V9_ONE_EPOCH_CANDIDATE_20260901T213444Z_BUNDLE`.
- Current-source CPU dry-run recipe:
  `V9_POSTRUN_SOURCE_REBIND_20260903T013249Z_RECIPE.json`, SHA-256
  `570a4baefb999d406f5d39b994bbed9a408244409ce9448e44fbc3e425c40372`.

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
