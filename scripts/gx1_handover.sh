#!/usr/bin/env bash
# Current takeover viewer for the XAUUSD direction-repair handover.
set -euo pipefail

REPO=/home/andre2/src/GX1_ENGINE
HANDOVER="$REPO/HANDOVER_XAU_DIRECTION_REPAIR_20260714.md"

if [[ ! -f "$HANDOVER" ]]; then
  echo "FATAL: current handover missing: $HANDOVER" >&2
  exit 2
fi

cd "$REPO"

echo "## GX1 XAU Direction Repair Takeover"
echo
echo "### Git"
git status --short
echo
echo "### Disk"
df -h /home/andre2/GX1_DATA
echo
echo "### RAM"
free -h
echo
echo "### Python Processes"
echo "Note: if this is run inside a sandbox, host runtime processes may require an approved host ps check."
if ps -C python -C python3 -o pid,ppid,stat,%cpu,%mem,etime,cmd --sort=-%cpu; then
  :
else
  echo "(no python/python3 processes matched in this process namespace)"
fi
echo
echo "### Handover"
sed -n '1,380p' "$HANDOVER"
