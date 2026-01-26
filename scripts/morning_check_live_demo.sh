#!/bin/bash
# Morning Check Script for Live DEMO
#
# Kjører etter natten for å verifisere:
# - SNIPER: 0 trades i natt (ASIA session)
# - FARM: 0-n trades (ASIA session)
# - Journal integrity (entry_snapshot, execution_events, exit_summary)
#
# Usage:
#   ./scripts/morning_check_live_demo.sh [--farm-dir DIR] [--sniper-dir DIR]

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
FARM_DIR=""
SNIPER_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --farm-dir)
            FARM_DIR="$2"
            shift 2
            ;;
        --sniper-dir)
            SNIPER_DIR="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=== MORNING CHECK - LIVE DEMO ===${NC}"
echo ""

# Auto-detect if not provided
if [ -z "$FARM_DIR" ]; then
    FARM_DIR=$(ls -td runs/live_demo/FARM_* 2>/dev/null | head -1)
fi

if [ -z "$SNIPER_DIR" ]; then
    SNIPER_DIR=$(ls -td runs/live_demo/SNIPER_* 2>/dev/null | head -1)
fi

if [ -z "$FARM_DIR" ] || [ ! -d "$FARM_DIR" ]; then
    echo -e "${RED}ERROR: FARM directory not found${NC}"
    exit 1
fi

if [ -z "$SNIPER_DIR" ] || [ ! -d "$SNIPER_DIR" ]; then
    echo -e "${RED}ERROR: SNIPER directory not found${NC}"
    exit 1
fi

echo "FARM dir: ${FARM_DIR}"
echo "SNIPER dir: ${SNIPER_DIR}"
echo ""

# Check trade counts
echo -e "${BLUE}=== TRADE COUNTS ===${NC}"

# SNIPER trades (should be 0 in ASIA)
SNIPER_TRADES=$(find "${SNIPER_DIR}" -name "*.json" -path "*/trade_journal/trades/*" 2>/dev/null | wc -l | tr -d ' ')
echo "SNIPER trades: ${SNIPER_TRADES}"
if [ "$SNIPER_TRADES" -eq 0 ]; then
    echo -e "${GREEN}✓ SNIPER: 0 trades (expected in ASIA)${NC}"
else
    echo -e "${YELLOW}⚠ SNIPER: ${SNIPER_TRADES} trades (unexpected in ASIA)${NC}"
fi

# FARM trades (0-n expected)
FARM_TRADES=$(find "${FARM_DIR}" -name "*.json" -path "*/trade_journal/trades/*" 2>/dev/null | wc -l | tr -d ' ')
echo "FARM trades: ${FARM_TRADES}"
if [ "$FARM_TRADES" -ge 0 ]; then
    echo -e "${GREEN}✓ FARM: ${FARM_TRADES} trades (expected 0-n)${NC}"
else
    echo -e "${RED}✗ FARM: Invalid trade count${NC}"
fi

echo ""

# Check journal integrity
echo -e "${BLUE}=== JOURNAL INTEGRITY ===${NC}"

python3 << PYEOF
import json
from pathlib import Path
from collections import Counter

farm_dir = Path("${FARM_DIR}")
sniper_dir = Path("${SNIPER_DIR}")

def check_journal_integrity(run_dir, engine_name):
    trades_dir = run_dir / "trade_journal" / "trades"
    if not trades_dir.exists():
        print(f"{engine_name}: No trades directory found")
        return
    
    json_files = list(trades_dir.glob("*.json"))
    if len(json_files) == 0:
        print(f"{engine_name}: No trade JSON files found")
        return
    
    print(f"{engine_name}: Checking {len(json_files)} trade(s)...")
    
    has_entry_snapshot = 0
    has_execution_events = 0
    has_exit_summary = 0
    missing_fields = []
    
    for json_file in json_files[:10]:  # Check first 10
        try:
            with open(json_file, 'r') as f:
                d = json.load(f)
            
            if d.get("entry_snapshot"):
                has_entry_snapshot += 1
            else:
                missing_fields.append(f"{json_file.name}: missing entry_snapshot")
            
            if d.get("execution_events"):
                has_execution_events += 1
            else:
                missing_fields.append(f"{json_file.name}: missing execution_events")
            
            if d.get("exit_summary"):
                has_exit_summary += 1
            else:
                missing_fields.append(f"{json_file.name}: missing exit_summary")
        except Exception as e:
            missing_fields.append(f"{json_file.name}: error - {e}")
    
    print(f"  entry_snapshot: {has_entry_snapshot}/{min(10, len(json_files))}")
    print(f"  execution_events: {has_execution_events}/{min(10, len(json_files))}")
    print(f"  exit_summary: {has_exit_summary}/{min(10, len(json_files))}")
    
    if missing_fields:
        print(f"  ⚠ Missing fields in {len(missing_fields)} file(s)")
        for mf in missing_fields[:5]:
            print(f"    - {mf}")
    else:
        print(f"  ✓ All required fields present")

check_journal_integrity(farm_dir, "FARM")
print("")
check_journal_integrity(sniper_dir, "SNIPER")

PYEOF

echo ""

# Final summary
echo -e "${BLUE}=== SUMMARY ===${NC}"
echo "SNIPER: ${SNIPER_TRADES} trades (expected: 0 in ASIA)"
echo "FARM: ${FARM_TRADES} trades (expected: 0-n in ASIA)"
echo ""
echo "If unexpected results, check:"
echo "  - docs/ops/INCIDENT_PLAYBOOK.md"
echo "  - Logs in ${FARM_DIR}/farm_runtime.log"
echo "  - Logs in ${SNIPER_DIR}/sniper_runtime.log"
echo ""

