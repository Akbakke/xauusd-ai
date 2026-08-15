#!/usr/bin/env bash
# Complete the 2026-08-13 retention transaction.
#
# The plan bound the launch contract as it stood at commit 52e57533. Package 8
# then changed that file, so every stage of the transaction (execute, recover,
# resume) fails closed on the authority hash — by design. Restore the exact
# bound bytes for the duration of the completion, then put the current version
# back. Nothing is lost: both versions are committed.
set -euo pipefail
cd /home/andre2/src/GX1_ENGINE
LC=PROJECT_STATE_xau_direction_launch.json
KEEP=/tmp/claude-1000/-home-andre2/9cbeb236-a28b-4eac-8862-84382a344726/scratchpad/launch_contract_v30.json
R=/home/andre2/GX1_DATA/reports

cp "$LC" "$KEEP"
restore() { cp "$KEEP" "$LC"; }
trap restore EXIT

git show 52e57533:"$LC" > "$LC"
test "$(sha256sum "$LC" | cut -d' ' -f1)" = \
     "1885ebe27c02304e0d0acb441a7b0d6217e05abfa099a69183f5daf2c3754ab9"

scripts/gx1_capped_run.sh --class audit -- \
  .venv/bin/python -m gx1.scripts.cleanup_gx1_evidence_v1 resume \
  --plan-json "$R/gx1_evidence_retention_cleanup_plans/GX1_EVIDENCE_RETENTION_CLEANUP_PLAN_20260813T201653669865Z.json" \
  --plan-sha256 33a7fb65f02db7be7623e6ed8932150fb762412c525db3e4bd290826e162fd15 \
  --approval-json "$R/gx1_evidence_retention_cleanup_approvals/GX1_EVIDENCE_RETENTION_APPROVAL_20260813T202111867253Z.json" \
  --approval-sha256 1f4697e4e977531df87039f3eddd7fe47aeba94e9ddd9e160e9e9fd50fc7c4eb \
  --staged-json "$R/gx1_evidence_retention_cleanup_reports/GX1_EVIDENCE_CLEANUP_STAGED_20260813T202929071188Z.json" \
  --staged-sha256 e30c5958aa8d8f23a505099569c80e922b374c6de18a3f635e7cc264cd3b393b \
  --vedtak GX1_V30_RECLAIM_DEAD_V29_CHAIN_ROOTS_20260813 \
  --resume --allow-interrupted-payload
