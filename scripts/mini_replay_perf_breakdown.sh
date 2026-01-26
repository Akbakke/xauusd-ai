#!/usr/bin/env bash
# Mini replay for performance breakdown analysis (1 week, 1 worker)
#
# Mål: Stabilitet + perf-breakdown, ikke "maks hastighet"
# Output: Perf-rapport med %-fordeling av tid (journaling, features, inference, overhead)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

POLICY_THRESHOLD018="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_0_18.yaml"
DATA_FILE="data/raw/xauusd_m5_2025_bid_ask.parquet"

# 1 week test (UTC) - normal activity week (not Jan 1-7)
START_TS="2025-01-15T00:00:00Z"
END_TS="2025-01-22T00:00:00Z"  # 1 week

OUTPUT_BASE="data/replay/sniper/entry_v10_1_flat_threshold0_18/2025"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$OUTPUT_BASE/MINI_PERF_BREAKDOWN_${TIMESTAMP}"
LOG_FILE="/tmp/mini_replay_perf_${TIMESTAMP}.log"

echo "=================================================================================="
echo "MINI REPLAY - PERFORMANCE BREAKDOWN ANALYSIS"
echo "=================================================================================="
echo ""
echo "Configuration:"
echo "  - Policy: $POLICY_THRESHOLD018"
echo "  - Entry: ENTRY_V10.1 HYBRID (variant=v10_1, seq_len=90)"
echo "  - Threshold: min_prob_long=0.18"
echo "  - Period: 1 week (${START_TS} → ${END_TS})"
echo "  - Workers: 1 (single worker, no parallel chunks)"
echo "  - Device: CPU (GX1_FORCE_TORCH_DEVICE=cpu)"
echo "  - Fast replay: ON"
echo "  - Journaling: STD (not FULL)"
echo "  - Perf timing: ON"
echo "  - Log level: INFO"
echo "  - Log file: $LOG_FILE"
echo "  - Output dir: $OUTPUT_DIR"
echo ""

# Set environment variables
export GX1_LOGLEVEL=INFO
export GX1_FORCE_TORCH_DEVICE=cpu

# Validate files exist
if [ ! -f "$POLICY_THRESHOLD018" ]; then
    echo "❌ ERROR: Policy file not found: $POLICY_THRESHOLD018"
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "❌ ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Prepare test data (1 week, EU/OVERLAP/US only)
echo "[1/4] Preparing test data (1 week, EU/OVERLAP/US only)..."
TEST_DATA_DIR="data/temp/mini_perf_${TIMESTAMP}"
mkdir -p "$TEST_DATA_DIR"

python3 -c "
import pandas as pd
from pathlib import Path
import sys

data_file = sys.argv[1]
start_ts = sys.argv[2]
end_ts = sys.argv[3]
output_file = sys.argv[4]

from gx1.execution.live_features import infer_session_tag

df = pd.read_parquet(data_file)
df.index = pd.to_datetime(df.index, utc=True)

df = df[(df.index >= start_ts) & (df.index < end_ts)].copy()

# Filter to SNIPER sessions
sessions = ['EU', 'OVERLAP', 'US']
mask = df.index.map(lambda ts: infer_session_tag(ts) in sessions)
df = df[mask]

print(f'✅ Test data: {len(df):,} rows')
print(f'   Date range: {df.index.min()} → {df.index.max()}')

df.to_parquet(output_file)
print(f'✅ Saved to: {output_file}')
" "$DATA_FILE" "$START_TS" "$END_TS" "$TEST_DATA_DIR/test_data.parquet" 2>&1 | tee -a "$LOG_FILE"

TEST_DATA="$TEST_DATA_DIR/test_data.parquet"

# Run replay (single worker, no parallel chunks)
echo ""
echo "[2/4] Running replay with performance timing (single worker)..."
echo "   This should take ~5-15 minutes..."
echo ""

mkdir -p "$OUTPUT_DIR"

python3 scripts/run_mini_replay_perf.py "$POLICY_THRESHOLD018" "$TEST_DATA" "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"

REPLAY_EXIT_CODE=${PIPESTATUS[0]}

# Check for performance summary
echo ""
echo "[3/4] Checking for performance summary..."
echo ""

if grep -q "\[REPLAY_PERF_SUMMARY\]" "$LOG_FILE"; then
    echo "✅ Found [REPLAY_PERF_SUMMARY] in log"
    echo ""
    echo "Performance Summary:"
    grep "\[REPLAY_PERF_SUMMARY\]" "$LOG_FILE" | tail -1
    echo ""
else
    echo "❌ ERROR: No [REPLAY_PERF_SUMMARY] found in log"
    echo "   Check log file: $LOG_FILE"
    exit 1
fi

# Verify replay completed successfully
if [ $REPLAY_EXIT_CODE -ne 0 ]; then
    echo "❌ ERROR: Replay exited with code $REPLAY_EXIT_CODE"
    echo "   Check log file: $LOG_FILE"
    exit 1
fi

# Check that trade journals exist
if [ ! -d "$OUTPUT_DIR/trade_journal" ]; then
    echo "❌ ERROR: Trade journal directory not found: $OUTPUT_DIR/trade_journal"
    exit 1
fi

echo "✅ Trade journal directory exists: $OUTPUT_DIR/trade_journal"
echo ""

# Generate performance report
echo "[4/4] Generating performance breakdown report..."
echo ""

python3 << PYEOF 2>&1 | tee -a "$LOG_FILE"
import sys
from pathlib import Path
import re

log_file = Path("$LOG_FILE")
output_dir = Path("$OUTPUT_DIR")
report_path = Path("reports/rl/entry_v10/ENTRY_V10_1_MINI_REPLAY_PERF_BREAKDOWN_${TIMESTAMP}.md")

# Parse REPLAY_PERF_SUMMARY line
summary_line = None
with open(log_file, "r") as f:
    for line in f:
        if "[REPLAY_PERF_SUMMARY]" in line:
            summary_line = line.strip()
            break

if not summary_line:
    print("❌ ERROR: Could not find [REPLAY_PERF_SUMMARY] in log")
    sys.exit(1)

# Extract numbers from summary line
# Format: [REPLAY_PERF_SUMMARY] bars=%d trades=%d total=%.1fs bars_per_sec=%.2f feat=%.1fs (%.1f%%) model=%.1fs (%.1f%%) journal=%.1fs (%.1f%%) other=%.1fs (%.1f%%)
pattern = r"bars=(\d+) trades=(\d+) total=([\d.]+)s bars_per_sec=([\d.]+) feat=([\d.]+)s \(([\d.]+)%\) model=([\d.]+)s \(([\d.]+)%\) journal=([\d.]+)s \(([\d.]+)%\) other=([\d.]+)s \(([\d.]+)%\)"
match = re.search(pattern, summary_line)

if not match:
    print(f"❌ ERROR: Could not parse summary line: {summary_line}")
    sys.exit(1)

n_bars = int(match.group(1))
n_trades = int(match.group(2))
total_time = float(match.group(3))
bars_per_sec = float(match.group(4))
feat_time = float(match.group(5))
feat_pct = float(match.group(6))
model_time = float(match.group(7))
model_pct = float(match.group(8))
journal_time = float(match.group(9))
journal_pct = float(match.group(10))
other_time = float(match.group(11))
other_pct = float(match.group(12))

# Verify percentages sum to ~100%
total_pct = feat_pct + model_pct + journal_pct + other_pct
if abs(total_pct - 100.0) > 1.0:
    print(f"⚠️  WARNING: Percentages sum to {total_pct:.1f}% (expected ~100%)")

# Generate report
report_path.parent.mkdir(parents=True, exist_ok=True)

report_content = f"""# ENTRY_V10.1 Mini Replay - Performance Breakdown

**Date:** $(date +'%Y-%m-%d %H:%M:%S')
**Period:** 1 week ({START_TS} → {END_TS})
**Policy:** GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_0_18.yaml
**Workers:** 1 (single worker)
**Device:** CPU (GX1_FORCE_TORCH_DEVICE=cpu)
**Fast Replay:** ON
**Journaling:** STD

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Bars Processed | {n_bars:,} |
| Total Trades Created | {n_trades:,} |
| Total Time | {total_time:.1f} seconds ({total_time/60:.1f} minutes) |
| Bars per Second | {bars_per_sec:.2f} |

## Time Breakdown (% of Total)

| Component | Time (seconds) | Percentage |
|-----------|----------------|------------|
| **Feature Building** | {feat_time:.1f} | {feat_pct:.1f}% |
| **Model Inference** | {model_time:.1f} | {model_pct:.1f}% |
| **Journaling/I/O** | {journal_time:.1f} | {journal_pct:.1f}% |
| **Other/Overhead** | {other_time:.1f} | {other_pct:.1f}% |
| **Total** | {total_time:.1f} | {total_pct:.1f}% |

## Top-10 Bottleneck Analysis

Based on percentage of total time:

1. **{'Journaling/I/O' if journal_pct >= max(feat_pct, model_pct, other_pct) else 'Feature Building' if feat_pct >= max(model_pct, other_pct) else 'Model Inference' if model_pct >= other_pct else 'Other/Overhead'}**: {max(journal_pct, feat_pct, model_pct, other_pct):.1f}%
2. **{'Journaling/I/O' if journal_pct >= max([x for x in [feat_pct, model_pct, other_pct] if x != max(journal_pct, feat_pct, model_pct, other_pct)]) else 'Feature Building' if feat_pct >= max([x for x in [journal_pct, model_pct, other_pct] if x != max(journal_pct, feat_pct, model_pct, other_pct)]) else 'Model Inference' if model_pct >= max([x for x in [journal_pct, feat_pct, other_pct] if x != max(journal_pct, feat_pct, model_pct, other_pct)]) else 'Other/Overhead'}**: {sorted([journal_pct, feat_pct, model_pct, other_pct], reverse=True)[1]:.1f}%
3. **{'Journaling/I/O' if journal_pct >= sorted([x for x in [journal_pct, feat_pct, model_pct, other_pct] if x != max(journal_pct, feat_pct, model_pct, other_pct) and x != sorted([journal_pct, feat_pct, model_pct, other_pct], reverse=True)[1]])[0] else 'Feature Building' if feat_pct >= sorted([x for x in [journal_pct, feat_pct, model_pct, other_pct] if x != max(journal_pct, feat_pct, model_pct, other_pct) and x != sorted([journal_pct, feat_pct, model_pct, other_pct], reverse=True)[1]])[0] else 'Model Inference' if model_pct >= sorted([x for x in [journal_pct, feat_pct, model_pct, other_pct] if x != max(journal_pct, feat_pct, model_pct, other_pct) and x != sorted([journal_pct, feat_pct, model_pct, other_pct], reverse=True)[1]])[0] else 'Other/Overhead'}**: {sorted([journal_pct, feat_pct, model_pct, other_pct], reverse=True)[2]:.1f}%
4. **{'Journaling/I/O' if journal_pct == min(journal_pct, feat_pct, model_pct, other_pct) else 'Feature Building' if feat_pct == min(journal_pct, feat_pct, model_pct, other_pct) else 'Model Inference' if model_pct == min(journal_pct, feat_pct, model_pct, other_pct) else 'Other/Overhead'}**: {min(journal_pct, feat_pct, model_pct, other_pct):.1f}%

## Recommendations for Optimization

Based on the bottleneck analysis above:

"""
PYEOF

# Generate simpler report
python3 << PYEOF
import sys
from pathlib import Path
import re
from datetime import datetime

log_file = Path("$LOG_FILE")
report_path = Path("reports/rl/entry_v10/ENTRY_V10_1_MINI_REPLAY_PERF_BREAKDOWN_${TIMESTAMP}.md")

# Parse REPLAY_PERF_SUMMARY line
summary_line = None
with open(log_file, "r") as f:
    for line in f:
        if "[REPLAY_PERF_SUMMARY]" in line:
            summary_line = line.strip()
            break

if not summary_line:
    print("❌ ERROR: Could not find [REPLAY_PERF_SUMMARY] in log")
    sys.exit(1)

# Extract numbers
pattern = r"bars=(\d+) trades=(\d+) total=([\d.]+)s bars_per_sec=([\d.]+) feat=([\d.]+)s \(([\d.]+)%\) model=([\d.]+)s \(([\d.]+)%\) journal=([\d.]+)s \(([\d.]+)%\) other=([\d.]+)s \(([\d.]+)%\)"
match = re.search(pattern, summary_line)

if not match:
    print(f"❌ ERROR: Could not parse summary line")
    sys.exit(1)

n_bars = int(match.group(1))
n_trades = int(match.group(2))
total_time = float(match.group(3))
bars_per_sec = float(match.group(4))
feat_time = float(match.group(5))
feat_pct = float(match.group(6))
model_time = float(match.group(7))
model_pct = float(match.group(8))
journal_time = float(match.group(9))
journal_pct = float(match.group(10))
other_time = float(match.group(11))
other_pct = float(match.group(12))

# Sort components by percentage
components = [
    ("Journaling/I/O", journal_time, journal_pct),
    ("Feature Building", feat_time, feat_pct),
    ("Model Inference", model_time, model_pct),
    ("Other/Overhead", other_time, other_pct),
]
components_sorted = sorted(components, key=lambda x: x[2], reverse=True)

# Generate report
report_path.parent.mkdir(parents=True, exist_ok=True)

with open(report_path, "w") as f:
    f.write(f"# ENTRY_V10.1 Mini Replay - Performance Breakdown\n\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"**Period:** 1 week ({START_TS} → {END_TS})\n")
    f.write(f"**Policy:** GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_0_18.yaml\n")
    f.write(f"**Workers:** 1 (single worker)\n")
    f.write(f"**Device:** CPU (GX1_FORCE_TORCH_DEVICE=cpu)\n")
    f.write(f"**Fast Replay:** ON\n")
    f.write(f"**Journaling:** STD\n\n")
    
    f.write("## Performance Summary\n\n")
    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")
    f.write(f"| Total Bars Processed | {n_bars:,} |\n")
    f.write(f"| Total Trades Created | {n_trades:,} |\n")
    f.write(f"| Total Time | {total_time:.1f} seconds ({total_time/60:.1f} minutes) |\n")
    f.write(f"| Bars per Second | {bars_per_sec:.2f} |\n\n")
    
    f.write("## Time Breakdown (% of Total)\n\n")
    f.write("| Component | Time (seconds) | Percentage |\n")
    f.write("|-----------|----------------|------------|\n")
    for name, time_val, pct in components_sorted:
        f.write(f"| **{name}** | {time_val:.1f} | {pct:.1f}% |\n")
    f.write(f"| **Total** | {total_time:.1f} | {sum(c[2] for c in components):.1f}% |\n\n")
    
    f.write("## Top Bottleneck Analysis\n\n")
    f.write("Components sorted by percentage of total time:\n\n")
    for i, (name, time_val, pct) in enumerate(components_sorted, 1):
        f.write(f"{i}. **{name}**: {pct:.1f}% ({time_val:.1f}s)\n")
    f.write("\n")
    
    # Recommendations
    f.write("## Recommendations for Optimization\n\n")
    top_component = components_sorted[0][0]
    top_pct = components_sorted[0][2]
    
    if top_component == "Journaling/I/O":
        f.write(f"**Primary Bottleneck: Journaling/I/O ({top_pct:.1f}%)\n\n")
        f.write("Recommended optimizations:\n")
        f.write("- Batch JSON writes → Parquet format (5-30× speedup)\n")
        f.write("- Async writer thread (2-5× speedup)\n")
        f.write("- Reduce journaling verbosity (STD → MINIMAL if acceptable)\n\n")
    elif top_component == "Feature Building":
        f.write(f"**Primary Bottleneck: Feature Building ({top_pct:.1f}%)\n\n")
        f.write("Recommended optimizations:\n")
        f.write("- Precompute/cache rolling features (2-10× speedup)\n")
        f.write("- Vectorize feature computation (2-5× speedup)\n")
        f.write("- Reduce feature set if possible (1.5-3× speedup)\n\n")
    elif top_component == "Model Inference":
        f.write(f"**Primary Bottleneck: Model Inference ({top_pct:.1f}%)\n\n")
        f.write("Recommended optimizations:\n")
        f.write("- Increase batch size (2-10× speedup)\n")
        f.write("- Reduce model calls (cache predictions when appropriate)\n")
        f.write("- Optimize transformer sequence length if possible\n\n")
    else:
        f.write(f"**Primary Bottleneck: Other/Overhead ({top_pct:.1f}%)\n\n")
        f.write("Further investigation needed to identify specific overhead sources.\n\n")
    
    f.write(f"**Output Directory:** `$OUTPUT_DIR`\n")
    f.write(f"**Log File:** `$LOG_FILE`\n")

print(f"✅ Performance report saved to: {report_path}")
PYEOF

echo ""
echo "=================================================================================="
echo "✅ MINI REPLAY PERFORMANCE BREAKDOWN COMPLETE"
echo "=================================================================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Report: reports/rl/entry_v10/ENTRY_V10_1_MINI_REPLAY_PERF_BREAKDOWN_${TIMESTAMP}.md"
echo ""
echo "Next steps:"
echo "  1. Review performance breakdown report"
echo "  2. Identify primary bottleneck"
echo "  3. Apply targeted optimizations based on recommendations"
echo ""

