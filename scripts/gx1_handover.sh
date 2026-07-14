#!/usr/bin/env bash
# Thin viewer for the single current XAUUSD direction-repair handover.
set -euo pipefail

REPO=/home/andre2/src/GX1_ENGINE
HANDOVER="$REPO/HANDOVER_XAU_DIRECTION_REPAIR_20260714.md"

if [[ ! -f "$HANDOVER" ]]; then
  echo "FATAL: current handover missing: $HANDOVER" >&2
  exit 2
fi

exec sed -n '1,260p' "$HANDOVER"
