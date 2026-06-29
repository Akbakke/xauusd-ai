#!/usr/bin/env bash
# Guard legacy live/practice/OANDA surfaces while the Entry next-edge plan is active.
#
# Source from old launchers before they can start archived runners or place
# practice orders. The caller may override the ack var/value for a narrower
# surface, but the default blocks old Entry live-practice launch paths.
set -euo pipefail

SURFACE="${ENTRY_NEXT_EDGE_LEGACY_LIVE_SURFACE:-legacy live/practice surface}"
ACK_VAR="${ENTRY_NEXT_EDGE_LEGACY_LIVE_ACK_VAR:-GX1_ALLOW_LEGACY_ENTRY_LIVE_PRACTICE}"
ACK_VALUE="${ENTRY_NEXT_EDGE_LEGACY_LIVE_ACK_VALUE:-20260627_ALLOW_LEGACY_ENTRY_LIVE_PRACTICE}"
CURRENT_ACK="${!ACK_VAR:-}"

if [[ "$CURRENT_ACK" != "$ACK_VALUE" ]]; then
    cat >&2 <<EOF
[ABORT] Active Entry next-edge plan blocks ${SURFACE}.
        Current path is Entry foundation seq146 cleanup/audit/smoke-readiness.
        Use:
          scripts/entry_next_edge_control.sh verify
          scripts/entry_next_edge_control.sh selftest
          scripts/entry_next_edge_control.sh foundation-guardrails
          scripts/entry_next_edge_control.sh worktree-hygiene
          scripts/entry_next_edge_control.sh stage-foundation-cleanup --dry-run
          scripts/entry_next_edge_control.sh materialize-smoke
          scripts/entry_next_edge_control.sh train-readiness
        Override only with:
          ${ACK_VAR}=${ACK_VALUE} <command>
EOF
    exit 2
fi
