#!/usr/bin/env bash
# SNIPER ENTRY_V10.1 FLAT THRESHOLD 0.18 FULLYEAR 2025 Performance Profiling
#
# Runs FULLYEAR 2025 replay with ENTRY_V10.1 in FLAT mode with threshold=0.18
# and performance profiling enabled.
#
# This script profiles the replay to identify bottlenecks:
# - Feature building time
# - Model inference time
# - Trade journaling time
#
# Usage:
#   bash scripts/profile_sniper_entry_v10_1_flat_threshold018_2025.sh [--n-workers N]
#
# Output:
# - Log file: /tmp/profile_v10_1_threshold018_2025.log
# - Performance report: reports/rl/entry_v10/ENTRY_V10_1_REPLAY_PERF_2025_THRESHOLD_0_18.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Parse arguments
N_WORKERS=7  # Default 7 workers
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-workers)
            N_WORKERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

POLICY_THRESHOLD018="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_0_18.yaml"
DATA_FILE="data/raw/xauusd_m5_2025_bid_ask.parquet"

# FULLYEAR 2025 (UTC)
START_TS="2025-01-01T00:00:00Z"
END_TS="2026-01-01T00:00:00Z"  # exclusive

echo "=================================================================================="
echo "SNIPER ENTRY_V10.1 FLAT THRESHOLD 0.18 FULLYEAR 2025 Performance Profiling"
echo "=================================================================================="
echo ""
echo "Configuration:"
echo "  - Policy: $POLICY_THRESHOLD018"
echo "  - Entry: ENTRY_V10.1 HYBRID (variant=v10_1, seq_len=90)"
echo "  - Mode: FLAT with threshold=0.18"
echo "  - Data source: Same as P4.1 replays"
echo "  - Date range: $START_TS → $END_TS"
echo "  - Workers: $N_WORKERS (parallel replay)"
echo "  - Log level: INFO (minimal logging for profiling)"
echo ""

# Validate config exists
if [ ! -f "$POLICY_THRESHOLD018" ]; then
    echo "❌ ERROR: THRESHOLD 0.18 policy file not found: $POLICY_THRESHOLD018"
    exit 1
fi

# Validate data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Set log level to INFO (minimal logging)
export GX1_LOGLEVEL=INFO

# Log file for profiling output
LOG_FILE="/tmp/profile_v10_1_threshold018_2025.log"
echo "Log file: $LOG_FILE"
echo ""

# Run replay (using existing script but with profiling)
echo "[1/3] Running FULLYEAR 2025 replay with performance profiling..."
echo "   This may take a while for FULLYEAR data..."
echo ""

bash scripts/run_sniper_entry_v10_1_flat_threshold018_2025.sh --n-workers "$N_WORKERS" 2>&1 | tee "$LOG_FILE"

echo ""
echo "[2/3] Extracting performance metrics from log..."
echo ""

# Extract performance summary from log
PERF_SUMMARY=$(grep -E "^\[REPLAY_PERF_SUMMARY\]" "$LOG_FILE" | tail -1 || echo "")

if [ -z "$PERF_SUMMARY" ]; then
    echo "⚠️  WARNING: No [REPLAY_PERF_SUMMARY] line found in log"
    echo "   Check log file: $LOG_FILE"
    exit 1
fi

echo "Performance Summary:"
echo "$PERF_SUMMARY"
echo ""

echo "[3/3] Generating performance report..."
echo ""

# Generate markdown report
REPORT_PATH="reports/rl/entry_v10/ENTRY_V10_1_REPLAY_PERF_2025_THRESHOLD_0_18.md"
mkdir -p "$(dirname "$REPORT_PATH")"

python3 << PYEOF
import re
from pathlib import Path
from datetime import datetime

log_file = Path("$LOG_FILE")
report_path = Path("$REPORT_PATH")

# Parse performance summary line
# Format: [REPLAY_PERF_SUMMARY] bars=X trades=Y total=Z bars_per_sec=W feat=A (B%) model=C (D%) journal=E (F%) other=G (H%)
perf_summary_line = """$PERF_SUMMARY"""

lines = [
    "# ENTRY_V10.1 FLAT THRESHOLD 0.18 FULLYEAR 2025 Replay Performance Profile",
    "",
    f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "**Configuration:**",
    "- Policy: GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_0_18",
    "- Entry: ENTRY_V10.1 HYBRID (variant=v10_1, seq_len=90)",
    "- Threshold: min_prob_long=0.18, p_side_min.long=0.18",
    "- Sizing: FLAT (baseline sizing, no aggressive overlays)",
    "- Exit: ExitCritic V1 + RULE5/RULE6A",
    "- Workers: $N_WORKERS (parallel replay)",
    "",
    "## Performance Summary",
    "",
]

# Parse performance metrics
pattern = r"bars=(\d+)\s+trades=(\d+)\s+total=([\d.]+)s\s+bars_per_sec=([\d.]+)\s+feat=([\d.]+)s\s+\(([\d.]+)%\)\s+model=([\d.]+)s\s+\(([\d.]+)%\)\s+journal=([\d.]+)s\s+\(([\d.]+)%\)\s+other=([\d.]+)s\s+\(([\d.]+)%\)"
match = re.search(pattern, perf_summary_line)

if match:
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
    
    lines.extend([
        "| Component | Time (seconds) | Percentage |",
        "|-----------|---------------|------------|",
        f"| **Feature Building** | {feat_time:.1f} | {feat_pct:.1f}% |",
        f"| **Model Inference** | {model_time:.1f} | {model_pct:.1f}% |",
        f"| **Trade Journaling** | {journal_time:.1f} | {journal_pct:.1f}% |",
        f"| **Other** | {other_time:.1f} | {other_pct:.1f}% |",
        f"| **Total** | {total_time:.1f} | 100.0% |",
        "",
        "## Metrics",
        "",
        f"- **Total Bars Processed:** {n_bars:,}",
        f"- **Total Trades:** {n_trades:,}",
        f"- **Total Time:** {total_time:.1f} seconds ({total_time/60:.1f} minutes)",
        f"- **Bars per Second:** {bars_per_sec:.2f}",
        "",
        "## Analysis",
        "",
    ])
    
    # Identify bottlenecks
    bottlenecks = []
    if journal_pct > 30:
        bottlenecks.append(f"⚠️  **Trade Journaling** is the primary bottleneck ({journal_pct:.1f}% of total time)")
    if model_pct > 30:
        bottlenecks.append(f"⚠️  **Model Inference** is a significant bottleneck ({model_pct:.1f}% of total time)")
    if feat_pct > 30:
        bottlenecks.append(f"⚠️  **Feature Building** is a significant bottleneck ({feat_pct:.1f}% of total time)")
    
    if bottlenecks:
        lines.append("### Bottlenecks Identified:")
        lines.append("")
        for bottleneck in bottlenecks:
            lines.append(f"- {bottleneck}")
        lines.append("")
    else:
        lines.append("✅ No single component dominates execution time (>30%).")
        lines.append("")
        lines.append("Performance appears well-distributed across components.")
        lines.append("")
    
    lines.extend([
        "## Recommendations",
        "",
        "Based on the performance profile:",
    ])
    
    if journal_pct > 30:
        lines.extend([
            "- **Trade Journaling**: Consider batching JSON writes or reducing JSON size",
            "  - Use binary formats (parquet) for trade journal instead of JSON",
            "  - Batch writes instead of writing per-trade",
            "  - Reduce logging verbosity for replay mode",
        ])
    if model_pct > 30:
        lines.extend([
            "- **Model Inference**: Consider optimizing model loading or batching predictions",
            "  - Pre-load models in memory",
            "  - Batch predictions if possible",
            "  - Use model quantization or optimization",
        ])
    if feat_pct > 30:
        lines.extend([
            "- **Feature Building**: Consider optimizing feature computation",
            "  - Cache computed features where possible",
            "  - Optimize pandas operations (use vectorization)",
            "  - Profile specific feature calculations",
        ])
    
    lines.append("")
else:
    lines.append("⚠️  Failed to parse performance summary from log.")
    lines.append("")
    lines.append(f"Raw log line: `{perf_summary_line}`")
    lines.append("")

lines.extend([
    "## Log File",
    "",
    f"Full replay log: `$LOG_FILE`",
    "",
    "To extract performance metrics manually:",
    "```bash",
    "grep '\[REPLAY_PERF_SUMMARY\]' $LOG_FILE",
    "```",
    "",
])

# Write report
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"✅ Performance report saved to: {report_path}")
PYEOF

echo ""
echo "=================================================================================="
echo "✅ Performance Profiling Complete!"
echo "=================================================================================="
echo ""
echo "Log File: $LOG_FILE"
echo "Report: $REPORT_PATH"
echo ""
echo "Next steps:"
echo "  1. Review performance report: cat $REPORT_PATH"
echo "  2. Check for bottlenecks (>30% in any component)"
echo "  3. Implement optimizations based on recommendations"
echo ""

