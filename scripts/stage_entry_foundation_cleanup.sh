#!/usr/bin/env bash
set -euo pipefail

# Stage only the audited Entry foundation cleanup pathspec.
# Default mode is dry-run/report-only. Actual git index mutation requires both
# --apply and --vedtak.

REPO=/home/andre2/src/GX1_ENGINE
DATA=/home/andre2/GX1_DATA
PY=$REPO/.venv/bin/python
HYGIENE_JSON=$DATA/reports/entry_foundation_worktree_hygiene_20260628_v1/ENTRY_FOUNDATION_WORKTREE_HYGIENE_latest.json

APPLY=0
VEDTAK=""

usage() {
  cat <<'EOF'
Usage:
  scripts/stage_entry_foundation_cleanup.sh [--dry-run]
  scripts/stage_entry_foundation_cleanup.sh --apply --vedtak <id>

Default mode is dry-run. It refreshes worktree-hygiene and prints the exact
stage command only if foundation_cleanup_stage_ready=true.

--apply mutates the git index with the audited foundation stage pathspec, then
reruns worktree-hygiene and requires post-stage verification PASS_STAGED. It
does not commit, train, replay, shadow, promote, or touch live paths.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) APPLY=0; shift ;;
    --apply) APPLY=1; shift ;;
    --vedtak) VEDTAK="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "FATAL: unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "$REPO"

"$REPO/scripts/entry_next_edge_control.sh" worktree-hygiene --no-fail-on-dirty --quiet

mapfile -t INFO < <("$PY" - "$HYGIENE_JSON" <<'PY'
import json
import shlex
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
if report.get("foundation_cleanup_review_decision") != "PASS":
    print("FATAL: foundation cleanup review is not PASS.", file=sys.stderr)
    raise SystemExit(2)
if not report.get("stage_plan_safe"):
    print("FATAL: stage_plan_safe is not true.", file=sys.stderr)
    raise SystemExit(2)
stage_ready = bool(report.get("foundation_cleanup_stage_ready"))
dirty_count = int(report.get("dirty_count") or 0)
clean_git = report.get("decision") == "PASS_CLEAN_GIT" and dirty_count == 0
if not stage_ready and not clean_git:
    print("FATAL: foundation_cleanup_stage_ready is not true.", file=sys.stderr)
    raise SystemExit(2)
cmd = report.get("foundation_cleanup_stage_command") or []
if stage_ready and cmd[:2] != ["git", "add"]:
    print(f"FATAL: unexpected stage command: {cmd}", file=sys.stderr)
    raise SystemExit(2)
stage_file = ""
for item in cmd:
    if str(item).startswith("--pathspec-from-file="):
        stage_file = str(item).split("=", 1)[1]
if stage_ready and not stage_file:
    print("FATAL: stage command is missing --pathspec-from-file.", file=sys.stderr)
    raise SystemExit(2)
post = report.get("foundation_cleanup_post_stage_verification") or {}
print(stage_file)
print(" ".join(shlex.quote(str(part)) for part in cmd))
print(report.get("foundation_cleanup_dirty_count"))
print(report.get("review_before_stage_dirty_count"))
print(post.get("decision"))
print("true" if stage_ready else "false")
print("true" if clean_git else "false")
PY
)

STAGE_FILE="${INFO[0]}"
STAGE_COMMAND="${INFO[1]}"
FOUNDATION_DIRTY="${INFO[2]}"
REVIEW_DIRTY="${INFO[3]}"
POST_STAGE_DECISION="${INFO[4]}"
STAGE_READY="${INFO[5]}"
CLEAN_GIT="${INFO[6]}"

if [[ "$APPLY" != "1" ]]; then
  echo "Foundation cleanup stage-ready: $STAGE_READY"
  echo "Foundation cleanup dirty paths: $FOUNDATION_DIRTY"
  echo "Review/hold dirty paths: $REVIEW_DIRTY"
  echo "Current post-stage verification: $POST_STAGE_DECISION"
  if [[ "$STAGE_READY" == "true" ]]; then
    echo "Stage command:"
    echo "  $STAGE_COMMAND"
  elif [[ "$CLEAN_GIT" == "true" ]]; then
    echo "Clean git: true"
    echo "No foundation cleanup paths to stage."
  fi
  echo "Dry-run only; no git index changes made."
  exit 0
fi

if [[ -z "$VEDTAK" ]]; then
  echo "FATAL: --apply requires --vedtak <id>." >&2
  exit 2
fi

if [[ "$STAGE_READY" != "true" ]]; then
  if [[ "$CLEAN_GIT" == "true" ]]; then
    echo "Foundation cleanup already clean under vedtak=$VEDTAK"
    echo "post_stage_verification=$POST_STAGE_DECISION"
    exit 0
  fi
  echo "FATAL: foundation_cleanup_stage_ready is not true." >&2
  exit 2
fi

git add --pathspec-from-file="$STAGE_FILE"
"$REPO/scripts/entry_next_edge_control.sh" worktree-hygiene --no-fail-on-dirty --quiet
"$PY" - "$HYGIENE_JSON" "$VEDTAK" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
vedtak = sys.argv[2]
report = json.loads(path.read_text(encoding="utf-8"))
post = report.get("foundation_cleanup_post_stage_verification") or {}
if post.get("decision") != "PASS_STAGED":
    print(f"FATAL: post-stage verification is {post.get('decision')}, expected PASS_STAGED.", file=sys.stderr)
    print(f"cached_not_in_stage_count={post.get('cached_not_in_stage_count')}", file=sys.stderr)
    print(f"cached_hold_overlap_count={post.get('cached_hold_overlap_count')}", file=sys.stderr)
    print(f"stage_missing_from_cached_count={post.get('stage_missing_from_cached_count')}", file=sys.stderr)
    raise SystemExit(2)
print(f"Foundation cleanup staged under vedtak={vedtak}")
print(f"cached_count={post.get('cached_count')}")
print(f"expected_stage_count={post.get('expected_stage_count')}")
print(f"post_stage_verification={post.get('decision')}")
PY
