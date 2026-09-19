# GX1 current cloud-training status — 2026-09-08

Status verified at `2026-09-08T12:45:53Z` (`14:45:53 CEST`) against the
canonical WSL repository, Git, process table and executable handover check.
This document is the current human-readable status owner for cloud-training
preparation. It grants no execution authority.

## Executive status

- Canonical repository: `/home/andre2/src/GX1_ENGINE` on Ubuntu-22.04 as user
  `andre2`.
- Branch: `audit/v9-premiere-20260905`.
- Committed `HEAD`: `cf246b37f5521e6e1041fad7dfb2fe3647918c1f` (`Add explicit
  Hopper BF16 training policy`).
- The worktree is deliberately dirty while the cloud host, telemetry,
  deadline, smoke-measurement and capacity-gate package is being completed.
  These changes are not yet committed and therefore are not launchable source.
- No trainer, guarded trainer or CUDA-training process was active at the
  verification time.
- `bash scripts/gx1_handover.sh --check` exits `2` with `decision: BLOCK`,
  `pretraining_review_hold: ACTIVE`, `cuda_authority: NONE` and
  `test_paper_live_authority: NONE`.
- Exact blocker:
  `LOCAL_PRE_CLOUD_READINESS_COMPLETE__EXTERNAL_HOST_AND_FRESH_GATE_REQUIRED`.
- No cloud host has been purchased or provisioned. No cloud smoke, cloud
  capacity PASS, large candidate run, VAL run, TEST access, paper run or live
  run has started.

## What is already proved

- The canonical pre-TEST data/model package, source-repair lineage, CPU
  checkpoint migration and next-batch equivalence evidence remain preserved.
- The local RTX 3090 source-current smoke from source commit
  `efa99b2b3105d2fb44a041de404d5a66b41158f8` completed under the signed 160 W
  guard. Its immutable bundle is
  `LOCAL_3090_SOURCE_CURRENT_SMOKE_20260908T084056Z`, with bundle-commit
  SHA-256
  `09e6c247a13b52caade71d5bc5e2bdf1af3376bf8fe8e5a0927a73430ff88448`.
- That smoke proves local runtime plumbing and RTX 3090 guard headroom only.
  It predates the current committed Hopper policy and cannot qualify a cloud
  host, predict cloud completion time, or authorize candidate continuation.
- Both preserved candidate sessions and their checkpoints remain evidence.
  Neither is current training authority.
- No new dataset is being built. The cloud-preparation work does not change the
  model architecture, targets, feature surface, TRAIN/VAL split or sealed TEST
  boundary.
- The eight feature families remain wired through the existing model-native
  contracts. Structural routing is not evidence that every family adds
  economic value; feature removal remains deferred until measured usefulness
  evidence exists.

## Cloud package now in progress

The uncommitted package is intended to add these fail-closed controls:

1. An exact source- and host-bound H100/H200 Hopper profile with BF16 policy,
   GPU identity, cgroup limits, budget and deadline evidence.
2. Root-owned, nonce-bound, signed Linux GPU telemetry for the selected cloud
   host.
3. A local hard-deadline timer that invokes a provider deletion command and
   forces local shutdown. Actual provider-side deletion/TTL evidence must still
   be verified on the selected provider; a local timer alone is not treated as
   independent billing protection.
4. A bounded Hopper smoke recipe using 32 warm-up optimizer steps and 256
   measured optimizer steps at batch 32 or 64, with synchronized TRAIN timing,
   representative VAL timing, preflight timing and a real fsync'd checkpoint.
5. An immutable smoke measurement embedded identically in bundle metadata and
   lock files and bound through the bundle-commit manifest.
6. A capacity gate that projects the full physical TRAIN and VAL populations,
   30-epoch worst case, checkpoint overhead, two restart reserves, a 43.2-hour
   admission ceiling, a 48-hour hard host deadline and a buffered NOK 2,500
   cost ceiling.
7. Runner and guard bindings for exact source commit, host profile, GPU UUID,
   cgroup, telemetry owner and capacity gate. The existing local RTX 3090 FP32
   path remains separate.

## Current validation state

- A targeted capped CPU regression run completed without starting CUDA or
  training.
- Four tests currently fail in the uncommitted cloud package: three capacity
  fixture/projection expectations and one Hopper smoke-geometry fixture.
- These failures mean the package is not ready to commit, not ready to bind
  into a new recipe and not ready for cloud execution.
- Additional dedicated tests are still required for the smoke-measurement
  contract, benchmark materializer, integrated candidate-gate chain, remaining
  provider deadline and trainer measurement output.
- No documentation statement may convert this work-in-progress package into
  execution authority.

## Required path to the large run

The order is fixed and fail-closed:

1. Finish the cloud contracts and resolve all targeted CPU regressions.
2. Run the broader relevant CPU suite and inspect the complete diff.
3. Commit one clean, explicit source revision.
4. Materialize a new source-bound cloud smoke recipe from that exact commit.
5. Select and provision one exact H100 80 GB or H200 host only after provider
   identity, price, FX input and deletion/deadline controls are recorded.
6. Transfer the committed source and required artifacts, then rehash them on
   the destination host.
7. Materialize and verify the destination host profile and fresh signed
   telemetry.
8. Run one bounded, guarded Hopper smoke. TEST remains sealed.
9. Materialize the benchmark from the committed smoke bundle and require a
   fresh `PASS_CAPACITY_QUALIFIED` gate under both time and cost ceilings.
10. Materialize a new candidate recipe bound to the same source, host profile
    and capacity gate. Recheck remaining provider deadline before launch.
11. Start the large candidate only through the canonical capped runner and
    guard. Any mismatch or stale evidence returns to `BLOCK`.

## Authority and truth order

Use sources in this order:

1. Actual processes and host state.
2. `bash scripts/gx1_handover.sh --check`.
3. `PROJECT_STATE_xau_direction_launch.json`.
4. Exact immutable recipe, source bindings, host profile, session contract,
   checkpoint pointer, smoke bundle and capacity gate.
5. This document and `docs/CURRENT_HANDOFF_20260903.md` for human context.
6. Dated reviews and incident reports as historical evidence only.

Prose never grants CUDA, training, TEST, paper, live, promotion or spending
authority. A historical PASS never substitutes for fresh host-bound evidence.

## Safe read-only re-entry

```bash
cd /home/andre2/src/GX1_ENGINE
git status --short
git branch --show-current
git rev-parse HEAD
bash scripts/gx1_handover.sh --check
```

The expected current result is `BLOCK`. Do not replace `--check` with a launch
command until the complete cloud qualification chain above exists and passes.

## Markdown synchronization scope

All 28 canonical project Markdown files known to Git on 2026-09-08 point to
this status boundary. Package-manager files under `.venv`, pytest caches and
documents inside the separate ignored `.claude/worktrees/...` checkout are not
canonical project documentation and are intentionally not rewritten.
