#!/usr/bin/env bash
# Source this from legacy Entry V10/V10.1 launchers.
#
# Active Entry path is foundation seq146 cleanup/audit/smoke-readiness. Old
# XGB/V10/ET research commands are intentionally blocked unless a deliberate
# legacy ack is provided.

_ENTRY_LEGACY_RESEARCH_ACK_REQUIRED=20260627_ALLOW_LEGACY_ENTRY_V10_RESEARCH
_ENTRY_NEXT_EDGE_LEGACY_SURFACE="${ENTRY_NEXT_EDGE_LEGACY_SURFACE:-legacy Entry V10/V10.1 research launcher}"

if [[ "${GX1_ALLOW_LEGACY_ENTRY_V10_RESEARCH:-}" == "$_ENTRY_LEGACY_RESEARCH_ACK_REQUIRED" ]]; then
  echo "[entry-next-edge] legacy override acknowledged for $_ENTRY_NEXT_EDGE_LEGACY_SURFACE: $_ENTRY_LEGACY_RESEARCH_ACK_REQUIRED" >&2
  return 0 2>/dev/null || exit 0
fi

cat >&2 <<EOF
[ABORT] Active Entry next-edge plan blocks $_ENTRY_NEXT_EDGE_LEGACY_SURFACE.
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
          GX1_ALLOW_LEGACY_ENTRY_V10_RESEARCH=$_ENTRY_LEGACY_RESEARCH_ACK_REQUIRED <command>
EOF

exit 2
