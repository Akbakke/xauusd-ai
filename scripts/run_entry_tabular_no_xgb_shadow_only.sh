#!/usr/bin/env bash
set -euo pipefail

# Historical pre-foundation tabular no-XGB live-shadow launcher.
#
# The active Entry path is foundation seq146 smoke-readiness. Shadow/live are
# closed until the foundation train, candidate, replay and IQL gates produce
# explicit PASS/READY artifacts. This script is retained only as a tombstone for
# the old 2026-06-27 plan and must fail before any runner/preflight logic.

REPO=/home/andre2/src/GX1_ENGINE
cd "$REPO"

cat >&2 <<'EOF'
FATAL: blocked by active Entry foundation-freeze.

The 2026-06-27 tabular no-XGB shadow plan is historical pre-foundation evidence,
not the active operating point. Do not run preview-shadow, start-shadow,
verify-shadow, direct v12_paper_runner shadow, or legacy shadow launchers.

Run:
  scripts/entry_next_edge_control.sh verify
  scripts/entry_next_edge_control.sh selftest
  scripts/entry_next_edge_control.sh foundation-guardrails
  scripts/entry_next_edge_control.sh train-readiness

Next real action, only after explicit user vedtak:
  scripts/entry_next_edge_control.sh smoke-train --vedtak <id> --require-edge-audit
EOF
exit 2
