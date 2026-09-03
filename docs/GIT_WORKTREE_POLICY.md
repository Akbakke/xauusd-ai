# GX1 working-tree policy

> **2026-09-03 re-entry note:** use
> [`CURRENT_HANDOFF_20260903.md`](CURRENT_HANDOFF_20260903.md) for runtime
> truth. In particular, do not remove a nested worktree or V9 evidence while
> cleaning caches; current CUDA authority is separately blocked pending clean
> preflight and explicit operator authorisation, despite fresh signed 160 W
> physical-limit telemetry.

The canonical repository is `/home/andre2/src/GX1_ENGINE`. Verify it with
`git rev-parse --show-toplevel`; never hard-code an old workstation path.

A dirty tree is expected during the active repair. Existing modifications and
untracked files may belong to the user or another agent. Do not reset,
checkout, overwrite, stage or delete unrelated changes. Coordinate overlapping
files and inspect the exact diff before editing.

`bash scripts/gx1_handover.sh --check` is the read-only worktree/status
verifier. It reports prunable registered worktrees separately and treats one
as a fail-closed source-identity condition; inspect its path and reachability
before an owner explicitly runs any prune/cleanup command. The fingerprint
covers tracked changes and non-ignored untracked paths. The handover permits
only the declared local `.env`, canonical `.venv`, registered
`.claude/worktrees` and regenerable Python/pytest/ruff caches outside that
fingerprint; every other ignored path is a fail-closed source-identity block.
Never mistake a green tracked diff for proof about ignored files.
`scripts/run_seq513_rebuild_chain_v1.sh` consumes the same handover check
before it can start a heavyweight producer.

> **2026-08-30 checkpoint:** the active repair contains technical
> checkpoint-parity, VAL-journal and candidate Exit-evidence binding changes,
> plus the checkpoint-640 fresh-process resume evidence. They must pass focused
> regressions and be committed with their status docs before any new candidate
> declaration. No heavy training is allowed from an uncommitted or ambiguous
> worktree.

Tests and read-only audits may run in a dirty tree. A heavy dataset build,
training/replay evidence run or launch must bind the exact source revision and
worktree state required by its immutable run contract; it must not pretend a
dirty source is a clean committed artifact.

Do not use `git reset --hard`, destructive checkout, force-push or automated
cleanup of unknown files. Repository cleanup requires reachability checks.
Cleanup under `/home/andre2/GX1_DATA` is a separate explicitly authorized act.

The authoritative status files (`PROJECT_STATE_xau_direction_launch.json`, the
handover and every Markdown file fingerprinted by `scripts/gx1_handover.sh`)
must be updated together in one reviewed commit. They describe a partial,
non-admitted candidate session, but its live position is derived only by the
handover's runtime checks of the launch-state reference, recipe/source closure,
contract, pointer and state SHA. Do not leave a stale recipe, stale driver-cap
assertion or old execution plan in one document while changing another. The
next local CUDA operation must be explicitly declared against the frozen
source/recipe/session; candidate VAL, TEST, demo and live remain blocked.
