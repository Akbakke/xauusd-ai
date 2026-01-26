#!/usr/bin/env bash
# SNIPER Resume After Sleep
#
# Convenience wrapper to resume SNIPER after Mac sleep/wake.
# Backfills missing candles, optionally backfills trades, then starts SNIPER.
#
# Usage:
#   ./scripts/resume_sniper_after_sleep.sh [--skip-trade-backfill]

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Load .env if it exists
if [ -f .env ]; then
    echo -e "${BLUE}Loading .env file...${NC}"
    set -a
    source .env
    set +a
else
    echo -e "${YELLOW}Warning: .env file not found${NC}"
fi

# Run resume script
echo -e "${BLUE}=== SNIPER Resume After Sleep ===${NC}"
echo ""

python gx1/scripts/resume_sniper_after_sleep.py "$@"












