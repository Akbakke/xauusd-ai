#!/bin/bash
# Quick status check for SNIPER CANARY replays (P1, P2, P4)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== SNIPER CANARY Replay Status ==="
echo ""

for policy in "SNIPER_P1_PLONG_080" "SNIPER_P2_REGIME_BLOCKS" "SNIPER_P4_COMBINED"; do
    dir="runs/replay_shadow/$policy"
    log="$dir/replay.log"
    
    echo "--- $policy ---"
    
    if [ -f "$log" ]; then
        # Get last few lines
        echo "Last log lines:"
        tail -5 "$log" | sed 's/^/  /'
        
        # Check if still running
        if pgrep -f "replay.*$policy" > /dev/null; then
            echo "  Status: RUNNING"
        else
            echo "  Status: COMPLETED or STOPPED"
        fi
        
        # Check for results.json
        if [ -f "$dir/results.json" ]; then
            echo "  Results: Found results.json"
            # Try to extract trade count if possible
            if command -v python3 > /dev/null; then
                python3 -c "
import json
import sys
try:
    with open('$dir/results.json') as f:
        data = json.load(f)
        if 'total_trades' in data:
            print(f\"  Total trades: {data['total_trades']}\")
        if 'win_rate' in data:
            print(f\"  Win rate: {data['win_rate']:.2%}\")
except:
    pass
" 2>/dev/null || true
            fi
        else
            echo "  Results: Not yet available"
        fi
    else
        echo "  Log file: Not found (may not have started yet)"
        if pgrep -f "replay.*$policy" > /dev/null; then
            echo "  Status: Process running, waiting for log..."
        else
            echo "  Status: Not running"
        fi
    fi
    echo ""
done

echo "To monitor live:"
echo "  tail -f runs/replay_shadow/SNIPER_P1_PLONG_080/replay.log"
echo "  tail -f runs/replay_shadow/SNIPER_P2_REGIME_BLOCKS/replay.log"
echo "  tail -f runs/replay_shadow/SNIPER_P4_COMBINED/replay.log"

