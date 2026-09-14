#!/usr/bin/env bash
# The Mac workspace has one read-only entrypoint to the current Linux source.
set -euo pipefail
[[ $# -le 1 ]] || { echo "Usage: ./handover.sh [--check|--verbose]" >&2; exit 2; }
mode="${1:---check}"
case "$mode" in --check|--verbose) ;; *) echo "Unsupported handover argument" >&2; exit 2 ;; esac
encoded=$(printf 'cd /home/andre2/src/GX1_CURRENT && exec bash scripts/gx1_handover.sh %s\n' "$mode" | base64 | tr -d '\n')
exec ssh -o BatchMode=yes -o ConnectTimeout=5 gx1-3090-lan "wsl.exe -d Ubuntu-22.04 -u andre2 -- /bin/bash -c \"echo $encoded | base64 -d | /bin/bash\""
