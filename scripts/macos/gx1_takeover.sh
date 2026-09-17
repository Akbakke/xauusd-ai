#!/usr/bin/env bash
# Use current_work for the current policy/session; COMPLETED_RUN is history.
# current_work.handover_resume_point and latest_model_correction identify the stop point.
# This read-only entrypoint never starts training. Require measured learning first.
set -euo pipefail
[[ $# -le 1 ]] || { echo "Usage: ./handover.sh [--check|--verbose|--source-only]" >&2; exit 2; }
mode="${1:---check}"
case "$mode" in --check|--verbose|--source-only) ;;
  -h|--help) echo "Usage: ./handover.sh [--check|--verbose|--source-only] (read-only)"; exit 0 ;;
  *) echo "Unsupported handover argument" >&2; exit 2 ;; esac
encoded=$(printf 'cd /home/andre2/src/GX1_CURRENT && exec bash scripts/gx1_handover.sh %s\n' "$mode" | base64 | tr -d '\n')
exec ssh -o BatchMode=yes -o ConnectTimeout=5 gx1-3090-lan "wsl.exe -d Ubuntu-22.04 -u andre2 -- /bin/bash -c \"echo $encoded | base64 -d | /bin/bash\""
