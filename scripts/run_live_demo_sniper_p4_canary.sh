#!/bin/bash
# LEGACY REMOVED — Tombstone
#
# This script has been archived as part of Trial 160 migration.
# Use Trial 160 pipeline instead.
#
# Archived: 2026-01-16
# Archive location: archive/legacy_20260116/run_live_demo_sniper_p4_canary.sh

set -euo pipefail

if [ "${ALLOW_LEGACY:-}" != "1" ]; then
    echo "❌ ERROR: Legacy script removed."
    echo ""
    echo "This script has been archived as part of Trial 160 migration."
    echo "Use Trial 160 pipeline instead:"
    echo "  scripts/run_fullyear_trial160_prebuilt.sh (replay)"
    echo "  scripts/run_live_trial160.sh (live - TODO: implementer)"
    echo ""
    echo "To override (NOT RECOMMENDED):"
    echo "  ALLOW_LEGACY=1 ./scripts/run_live_demo_sniper_p4_canary.sh"
    exit 1
fi

# If ALLOW_LEGACY=1, load from archive
ARCHIVE_SCRIPT="archive/legacy_20260116/run_live_demo_sniper_p4_canary.sh"
if [ ! -f "$ARCHIVE_SCRIPT" ]; then
    echo "❌ ERROR: Archived script not found: $ARCHIVE_SCRIPT"
    exit 1
fi

exec bash "$ARCHIVE_SCRIPT" "$@"
