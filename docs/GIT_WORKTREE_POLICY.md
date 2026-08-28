# GX1 working-tree policy

The canonical repository is `/home/andre2/src/GX1_ENGINE`. Verify it with
`git rev-parse --show-toplevel`; never hard-code an old workstation path.

A dirty tree is expected during the active repair. Existing modifications and
untracked files may belong to the user or another agent. Do not reset,
checkout, overwrite, stage or delete unrelated changes. Coordinate overlapping
files and inspect the exact diff before editing.

Tests and read-only audits may run in a dirty tree. A heavy dataset build,
training/replay evidence run or launch must bind the exact source revision and
worktree state required by its immutable run contract; it must not pretend a
dirty source is a clean committed artifact.

Do not use `git reset --hard`, destructive checkout, force-push or automated
cleanup of unknown files. Repository cleanup requires reachability checks.
Cleanup under `/home/andre2/GX1_DATA` is a separate explicitly authorized act.

The authoritative status files (`PROJECT_STATE_xau_direction_launch.json`, the
handover and every Markdown file fingerprinted by `scripts/gx1_handover.sh`)
must be updated together in one reviewed commit. They describe the 2026-08-28
bounded-smoke status: four 220 W CUDA steps/validation completed, then no
bundle was allowed because a retired static Exit path was named in the
movement proof. Do not leave a stale run recipe or old execution plan in one
document while changing another. Commit `a77ebb6e` repairs that proof; its
fresh recipe-bound/dry-run-checked repeat remains the only local continuation.
