#!/usr/bin/env bash
set -euo pipefail

# Campaign bookkeeping may verify bound checkpoint bytes. Keep that work
# inside the same exclusive, resource-capped audit route as other GX1 audits.
repo=/home/andre2/src/GX1_CURRENT
if [[ "$(realpath "$0")" != "$repo/scripts/gx1_capped_campaign_control.sh" ]] ||
   [[ $# -lt 3 || "$1" != '-m' || "$2" != 'gx1.scripts.local_random_access_campaign_v2' ]]; then
  echo 'FATAL: invalid capped campaign-control entry point' >&2
  exit 64
fi
case "$3" in
  inspect|begin|record) ;;
  *) echo 'FATAL: campaign-control command is not allowed' >&2; exit 64 ;;
esac
exec "$repo/scripts/gx1_capped_run.sh" --class audit --mem 4G --swap 512M -- \
  "$repo/.venv/bin/python" "$@"
