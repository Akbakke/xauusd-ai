#!/usr/bin/env bash
# ENTRY_V10.1 Threshold Sweep Test 2025 - LOWZONE (0.10-0.16)
#
# Tests lower thresholds to find optimal baseline under 0.18.
# Tests thresholds: [0.10, 0.12, 0.14, 0.16]
#
# For each threshold:
#   - Creates config file
#   - Runs FULLYEAR 2025 replay with 7 workers
#   - Generates report with key metrics
#   - Collects statistics
#
# After all runs, generates combined summary report comparing all thresholds.
#
# Usage:

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
test "$(pwd)" = "$ROOT" || { echo "❌ FAIL: Not in repo-root (pwd=$(pwd), root=$ROOT)"; exit 1; }
echo "[RUN_CTX] root=$ROOT"
echo "[RUN_CTX] head=$(git rev-parse --short HEAD)"
echo "[RUN_CTX] whoami=$(whoami) host=$(hostname)"
#   bash scripts/run_entry_v10_1_threshold_sweep_lowzone_2025.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Thresholds to test (LOWZONE: under 0.18)
THRESHOLDS=(0.10 0.12 0.14 0.16)
DATA_FILE="data/raw/xauusd_m5_2025_bid_ask.parquet"
START_TS="2025-01-01T00:00:00Z"
END_TS="2026-01-01T00:00:00Z"

echo "=================================================================================="
echo "ENTRY_V10.1 THRESHOLD SWEEP TEST 2025 - LOWZONE (0.10-0.16)"
echo "=================================================================================="
echo ""
echo "Thresholds to test: ${THRESHOLDS[*]}"
echo "Mode: SEQUENTIAL (7 workers per threshold)"
echo "Purpose: Find optimal baseline threshold under 0.18"
echo ""

# Validate data file
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Function to create config for a threshold (same as main sweep script)
create_threshold_config() {
    local threshold=$1
    local threshold_str=$(printf "%.2f" "$threshold" | tr '.' '_')
    
    local entry_config="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/ENTRY_V10_1_SNIPER_FLAT_THRESHOLD_${threshold_str}.yaml"
    local main_config="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_${threshold_str}.yaml"
    
    echo "📝 Creating config for threshold ${threshold}..."
    
    # Create entry config
    cat > "$entry_config" << EOF
# ENTRY_V10.1_SNIPER Policy Configuration (FLAT THRESHOLD ${threshold}) - LOWZONE sweep
#
# SNIPER entry policy with ENTRY_V10.1 HYBRID (XGB + Transformer).
# FLAT mode: No aggressive sizing, uses baseline sizing from P4.1.
# THRESHOLD ${threshold}: LOWZONE threshold sweep test value.
#
# Configuration:
#   - Entry: ENTRY_V10.1 HYBRID (variant=v10_1, seq_len=90)
#   - Same regime blocks as P4.1 (TREND_UP×LOW, NEUTRAL×HIGH, DOWN×HIGH)
#   - Same session filters (EU/OVERLAP/US)
#   - Baseline sizing (same lot/risk logic as P4.1)
#   - Threshold: min_prob_long=${threshold}, p_side_min.long=${threshold}
#
# Status: OFFLINE_ONLY (LOWZONE threshold sweep test)

entry_v9_policy_sniper:
  enabled: true
  min_prob_long: ${threshold}  # LOWZONE threshold sweep test value
  min_prob_profitable: 0.0
  enable_profitable_filter: false
  require_trend_up: false
  
  allow_low_vol: true
  allow_medium_vol: true
  allow_high_vol: true
  allow_extreme_vol: false
  
  allow_short: false

entry_models:
  v9:
    enabled: false
  
  v10:
    enabled: true
    variant: v10_1
    model_path: "models/entry_v10/entry_v10_1_transformer.pt"
    meta_path: "models/entry_v10/entry_v10_1_transformer_meta.json"
    feature_meta_path: "gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json"
  
  xgb:
    enabled: true
    eu_model_path: "models/entry_v10/xgb_entry_EU_v10.joblib"
    us_model_path: "models/entry_v10/xgb_entry_US_v10.joblib"
    overlap_model_path: "models/entry_v10/xgb_entry_OVERLAP_v10.joblib"

require_v9_for_entry: false

meta_model:
  enabled: true
  model_path: gx1/models/farm_entry_meta/baseline_model.pkl
  feature_cols_path: gx1/models/farm_entry_meta/feature_cols.json

allowed_sessions: ["EU", "OVERLAP", "US"]
allowed_vol_regimes: ["LOW", "MEDIUM", "HIGH"]
allowed_trend_regimes: []

guard:
  enabled: true

entry_gating:
  p_side_min:
    long: ${threshold}  # LOWZONE threshold sweep test value
    short: 1.0
  margin_min:
    long: 0.50
    short: 0.50
  side_ratio_min: 1.25
  sticky_bars: 1
  
  stage0:
    enabled: true
    allowed_sessions: ["EU", "OVERLAP", "US"]
    allowed_vol_regimes: ["LOW", "MEDIUM", "HIGH"]
    
    block_trend_vol_combos:
      - trend: "TREND_NEUTRAL"
        vol: "HIGH"
      - trend: "TREND_DOWN"
        vol: "HIGH"
      - trend: "TREND_UP"
        vol: "LOW"
EOF

    # Create main config
    cat > "$main_config" << EOF
# GX1 v1.2 → SNIPER ENTRY_V10.1 FLAT THRESHOLD ${threshold} (OFFLINE_ONLY)
#
# Entry: ENTRY_V10.1 HYBRID (variant=v10_1, seq_len=90) with baseline sizing
# Exit: HYBRID_ROUTER_V3 - ML decision tree with range features
# Mode: REPLAY (OFFLINE_ONLY - LOWZONE threshold sweep test)
#
# Configuration:
#   - Entry: ENTRY_V10.1 HYBRID (deeper Transformer, longer sequences)
#   - FLAT sizing: Same baseline sizing as P4.1 (no aggressive size overlays)
#   - THRESHOLD ${threshold}: LOWZONE threshold sweep test value
#   - Exit: ExitCritic V1 + RULE5/RULE6A (same as P4.1)
#
# ⚠️  OFFLINE_ONLY: This config is for LOWZONE threshold sweep testing only.

meta:
  router_version: V3_RANGE
  name: "SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_${threshold_str}"
  role: OFFLINE_ONLY
  description: >
    SNIPER with ENTRY_V10.1 HYBRID (variant=v10_1, seq_len=90) in FLAT mode.
    Threshold ${threshold} (LOWZONE threshold sweep test).
    Baseline sizing (no aggressive size overlays).
  bundle_location: sniper_snapshot/2025_SNIPER_V10_1
  notes: "LOWZONE threshold sweep test (threshold=${threshold})"

policy_name: GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_${threshold_str}
version: "GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_${threshold_str}"
mode: "REPLAY"
run_mode: "PROD"
instrument: "XAU_USD"
timeframe: "M5"
warmup_bars: 288

entry_config: gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/ENTRY_V10_1_SNIPER_FLAT_THRESHOLD_${threshold_str}.yaml

exit_config: gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_A_P4_1.yaml

exit_policies:
  rule5: gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_A_P4_1.yaml
  rule6a: gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_ADAPTIVE.yaml

hybrid_exit_router:
  version: HYBRID_ROUTER_V3
  model_path: gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/exit_router_models_v3/exit_router_v3_tree.pkl
  v3_range_edge_cutoff: 1.0
  documentation: |
    HYBRID_ROUTER_V3_RANGE provides ML-based routing using decision tree trained on range features.

execution:
  dry_run: true
  max_open_trades: 3
  min_time_between_trades_sec: 180

risk_guard:
  config_path: gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/SNIPER_RISK_GUARD_V1.yaml

logging:
  level: INFO

backfill:
  enabled: false

exit_control:
  allowed_loss_closers:
    - RULE_A_PROFIT
    - RULE_A_TRAILING
    - RULE_B_FAST_LOSS
    - RULE_C_TIMEOUT
    - RULE_FORCE_TIMEOUT
    - BROKER_SL
    - SL_TICK
    - REPLAY_EOF
  require_trade_open: true

replay:
  close_open_trades_at_end: true
  eof_close_reason: "REPLAY_EOF"
  eof_price_source: "mid"

exit_critic:
  enabled: true
  model_path: models/exit_critic/exit_critic_xgb_v1.json
  metadata_path: models/exit_critic/exit_critic_xgb_v1.meta.json
  
  exit_now_threshold: 0.60
  scalp_threshold: 0.40
  
  guards:
    min_bars_held: 5
    apply_for_loss_leq: -40.0
    apply_for_profit_ge: 40.0
    allowed_sessions:
      - EU
      - OVERLAP
      - US
    allowed_vol_regimes:
      - LOW
      - MEDIUM
      - HIGH
    allowed_trend_regimes:
      - TREND_UP
      - TREND_NEUTRAL
      - TREND_DOWN
EOF

    echo "✅ Created configs for threshold ${threshold}"
}

# Function to extract metrics from trade journal (same as main sweep script)
extract_metrics() {
    local output_dir=$1
    local threshold=$2
    
    python3 << PYEOF
import json
import pandas as pd
import os
from datetime import timedelta

output_dir = '$output_dir'
chunks_dir = os.path.join(output_dir, 'parallel_chunks')

all_trades = []

for i in range(7):
    trades_dir = os.path.join(chunks_dir, f'chunk_{i}', 'trade_journal', 'trades')
    if not os.path.exists(trades_dir):
        continue
    
    json_files = sorted([f for f in os.listdir(trades_dir) if f.endswith('.json')])
    
    for json_file in json_files:
        json_path = os.path.join(trades_dir, json_file)
        try:
            with open(json_path, 'r') as f:
                trade = json.load(f)
            
            exit_sum = trade.get('exit_summary')
            if exit_sum:
                pnl_bps = exit_sum.get('realized_pnl_bps')
                exit_time_str = exit_sum.get('exit_time')
                
                if pnl_bps is not None and exit_time_str:
                    exit_events = trade.get('exit_events', [])
                    bars_held = exit_events[0].get('bars_held') if exit_events else None
                    
                    entry_time = None
                    if bars_held is not None:
                        try:
                            exit_dt = pd.to_datetime(exit_time_str, utc=True)
                            entry_dt = exit_dt - timedelta(minutes=int(bars_held) * 5)
                            entry_time = entry_dt.isoformat()
                        except:
                            pass
                    
                    if not entry_time:
                        entry_time = exit_time_str
                    
                    all_trades.append({
                        'trade_id': trade.get('trade_id'),
                        'entry_time': entry_time,
                        'exit_time': exit_time_str,
                        'pnl_bps': pnl_bps,
                        'exit_reason': exit_sum.get('exit_reason'),
                    })
        except Exception:
            continue

if len(all_trades) > 0:
    df = pd.DataFrame(all_trades)
    df['pnl_bps'] = pd.to_numeric(df['pnl_bps'], errors='coerce')
    df = df[df['pnl_bps'].notna()].copy()
    
    if len(df) > 0:
        pnl = df['pnl_bps']
        
        # Calculate cumulative PnL for drawdown
        df_sorted = df.sort_values('exit_time')
        df_sorted['cumulative_pnl'] = df_sorted['pnl_bps'].cumsum()
        df_sorted['running_max'] = df_sorted['cumulative_pnl'].cummax()
        df_sorted['drawdown'] = df_sorted['cumulative_pnl'] - df_sorted['running_max']
        max_drawdown = df_sorted['drawdown'].min() if len(df_sorted) > 0 else 0
        
        metrics = {
            'trade_count': len(df),
            'mean_bps': float(pnl.mean()),
            'median_bps': float(pnl.median()),
            'winrate': float((pnl > 0).mean()),
            'p01_bps': float(pnl.quantile(0.01)),
            'p05_bps': float(pnl.quantile(0.05)),
            'p95_bps': float(pnl.quantile(0.95)),
            'p99_bps': float(pnl.quantile(0.99)),
            'max_drawdown_bps': float(max_drawdown),
            'total_pnl_bps': float(pnl.sum()),
            'std_bps': float(pnl.std()),
            'min_bps': float(pnl.min()),
            'max_bps': float(pnl.max()),
        }
        
        import json
        print(json.dumps(metrics))
    else:
        print('{"error": "No valid trades"}')
else:
    print('{"error": "No trades found"}')
PYEOF
}

# Function to run replay for a threshold (same structure as main sweep script)
run_threshold_replay() {
    local threshold=$1
    local threshold_str=$(printf "%.2f" "$threshold" | tr '.' '_')
    
    echo ""
    echo "=================================================================================="
    echo "Running replay for threshold ${threshold}"
    echo "=================================================================================="
    echo ""
    
    local policy_config="gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1/GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_${threshold_str}.yaml"
    local output_base="data/replay/sniper/entry_v10_1_flat_threshold${threshold_str}/2025"
    mkdir -p "$output_base"
    local run_tag="THRESHOLD_${threshold_str}_$(date +%Y%m%d_%H%M%S)"
    local output_dir="$output_base/$run_tag"
    mkdir -p "$output_dir"
    
    echo "Policy: $policy_config"
    echo "Output: $output_dir"
    echo ""
    
    # Prepare filtered data (reuse if exists)
    local filtered_data="$output_base/fullyear_2025_filtered.parquet"
    if [ ! -f "$filtered_data" ]; then
        echo "[1/5] Preparing FULLYEAR 2025 data..."
        python3 -c "
import pandas as pd
import sys
from gx1.execution.live_features import infer_session_tag

data_file = sys.argv[1]
start_ts = sys.argv[2]
end_ts = sys.argv[3]
output_file = sys.argv[4]

df = pd.read_parquet(data_file)
df.index = pd.to_datetime(df.index, utc=True)
df = df[(df.index >= start_ts) & (df.index < end_ts)].copy()

sessions = ['EU', 'OVERLAP', 'US']
mask = df.index.map(lambda ts: infer_session_tag(ts) in sessions)
df = df[mask]

df.to_parquet(output_file)
print(f'✅ Filtered data: {len(df):,} rows')
" "$DATA_FILE" "$START_TS" "$END_TS" "$filtered_data"
    else
        echo "[1/5] Using existing filtered data"
    fi
    
    # Split into chunks
    echo "[2/5] Splitting into 7 chunks..."
    local chunk_dir="$output_dir/chunks"
    mkdir -p "$chunk_dir"
    
    python3 -c "
import pandas as pd
from pathlib import Path
import sys

data_file = sys.argv[1]
chunk_dir = Path(sys.argv[2])

df = pd.read_parquet(data_file)
chunk_size = (len(df) + 7 - 1) // 7

for i in range(7):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(df))
    if start_idx >= len(df):
        break
    chunk_df = df.iloc[start_idx:end_idx]
    chunk_file = chunk_dir / f'chunk_{i}.parquet'
    chunk_df.to_parquet(chunk_file)
    print(f'  Chunk {i}: {len(chunk_df):,} bars')
" "$filtered_data" "$chunk_dir"
    
    # Run parallel replays
    echo "[3/5] Running replay with 7 workers..."
    mkdir -p "$output_dir/parallel_chunks"
    mkdir -p "$output_dir/logs"
    
    for i in {0..6}; do
        local chunk_file="$chunk_dir/chunk_${i}.parquet"
        if [ ! -f "$chunk_file" ]; then
            continue
        fi
        
        local chunk_output="$output_dir/parallel_chunks/chunk_${i}"
        mkdir -p "$chunk_output"
        
        (
            export OMP_NUM_THREADS=1
            export OPENBLAS_NUM_THREADS=1
            export MKL_NUM_THREADS=1
            export VECLIB_MAXIMUM_THREADS=1
            export NUMEXPR_NUM_THREADS=1
            export GX1_XGB_THREADS=1
            
            python3 -c "
import sys
from pathlib import Path
from gx1.execution.oanda_demo_runner import GX1DemoRunner

policy_path = Path(sys.argv[1])
chunk_file = Path(sys.argv[2])
output_dir = Path(sys.argv[3])

runner = GX1DemoRunner(
    policy_path,
    dry_run_override=True,
    replay_mode=True,
    fast_replay=True,
    output_dir=output_dir,
)

runner.run_replay(chunk_file)
" "$policy_config" "$chunk_file" "$chunk_output" > "$output_dir/logs/chunk_${i}.log" 2>&1
        ) &
    done
    
    wait
    echo "✅ All chunks completed"
    
    # Extract metrics
    echo "[4/5] Extracting metrics..."
    local metrics_json=$(extract_metrics "$output_dir" "$threshold")
    echo "$metrics_json" > "$output_dir/metrics.json"
    
    # Generate report
    echo "[5/5] Generating report..."
    local report_path="reports/rl/entry_v10/ENTRY_V10_1_FLAT_${threshold_str}_LOWZONE_FULLYEAR.md"
    python3 << PYEOF
import json
from pathlib import Path
from datetime import datetime

metrics = json.loads('''$metrics_json''')
threshold = $threshold
report_path = Path('$report_path')
output_dir = '$output_dir'

if 'error' in metrics:
    lines = [
        f'# ENTRY_V10.1 FLAT THRESHOLD {threshold} LOWZONE FULLYEAR 2025 Replay Results',
        '',
        f'**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        f'**Error:** {metrics["error"]}',
        '',
        f'**Output Directory:** `{output_dir}`',
    ]
else:
    lines = [
        f'# ENTRY_V10.1 FLAT THRESHOLD {threshold} LOWZONE FULLYEAR 2025 Replay Results',
        '',
        f'**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        '**Configuration:**',
        f'- Policy: GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_{threshold:.2f}'.replace('.', '_'),
        f'- Threshold: min_prob_long={threshold}, p_side_min.long={threshold}',
        '- Sizing: FLAT (baseline sizing, no aggressive overlays)',
        '',
        '## Results',
        '',
        '| Metric | Value |',
        '|--------|-------|',
        f'| Total Trades | {metrics["trade_count"]:,} |',
        f'| Mean PnL | {metrics["mean_bps"]:.2f} bps |',
        f'| Median PnL | {metrics["median_bps"]:.2f} bps |',
        f'| Win Rate | {metrics["winrate"]*100:.1f}% |',
        f'| P01 (Tail Risk) | {metrics["p01_bps"]:.2f} bps |',
        f'| P05 (Tail Risk) | {metrics["p05_bps"]:.2f} bps |',
        f'| Max Drawdown | {metrics["max_drawdown_bps"]:.2f} bps |',
        f'| Total PnL | {metrics["total_pnl_bps"]:.2f} bps |',
        f'| Std PnL | {metrics["std_bps"]:.2f} bps |',
        f'| Min PnL | {metrics["min_bps"]:.2f} bps |',
        f'| Max PnL | {metrics["max_bps"]:.2f} bps |',
        '',
        f'**Output Directory:** `{output_dir}`',
        '',
    ]

report_path.parent.mkdir(parents=True, exist_ok=True)
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f'✅ Report saved to: {report_path}')
PYEOF

    echo "✅ Threshold ${threshold} replay complete"
    echo "$metrics_json"
}

# Main execution
echo "[PREP] Creating configs for all thresholds..."
for threshold in "${THRESHOLDS[@]}"; do
    create_threshold_config "$threshold"
done

echo ""
echo "✅ All configs created"
echo ""

# Run replays sequentially
echo "[RUN] Running replays sequentially..."
for threshold in "${THRESHOLDS[@]}"; do
    echo ""
    echo "Running threshold ${threshold}..."
    run_threshold_replay "$threshold" 2>&1 | tee "/tmp/threshold_lowzone_${threshold}_replay.log"
    echo ""
done

echo "✅ All replays completed"

# Generate combined summary
echo ""
echo "=================================================================================="
echo "Generating combined LOWZONE summary report"
echo "=================================================================================="
echo ""

python3 << 'PYEOF'
import json
import os
from pathlib import Path
from datetime import datetime

thresholds = [0.10, 0.12, 0.14, 0.16]
base_dir = Path('data/replay/sniper')

# Collect metrics for all thresholds
all_metrics = {}

for threshold in thresholds:
    threshold_str = f'{threshold:.2f}'.replace('.', '_')
    output_base = base_dir / f'entry_v10_1_flat_threshold{threshold_str}' / '2025'
    
    if output_base.exists():
        runs = sorted([d for d in output_base.iterdir() if d.is_dir()], 
                     key=lambda x: x.stat().st_mtime, reverse=True)
        if runs:
            metrics_file = runs[0] / 'metrics.json'
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    all_metrics[threshold] = metrics
                except:
                    pass

# Generate summary report
report_path = Path('reports/rl/entry_v10/ENTRY_V10_1_THRESHOLD_SWEEP_LOWZONE_2025.md')

lines = [
    '# ENTRY_V10.1 THRESHOLD SWEEP SUMMARY 2025 - LOWZONE (0.10-0.16)',
    '',
    f'**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
    '',
    '**Purpose:** Find optimal baseline threshold under 0.18 for ENTRY_V10.1 FLAT configuration',
    '',
    '## Comparison Table',
    '',
    '| Threshold | Trade Count | Mean BPS | Median BPS | Win Rate | P01 | P05 | Max DD | Total PnL |',
    '|-----------|-------------|----------|------------|----------|-----|-----|--------|-----------|',
]

for threshold in sorted(all_metrics.keys()):
    m = all_metrics[threshold]
    if 'error' not in m:
        lines.append(
            f'| {threshold:.2f} | {m["trade_count"]:,} | {m["mean_bps"]:.2f} | {m["median_bps"]:.2f} | '
            f'{m["winrate"]*100:.1f}% | {m["p01_bps"]:.2f} | {m["p05_bps"]:.2f} | '
            f'{m["max_drawdown_bps"]:.2f} | {m["total_pnl_bps"]:.2f} |'
        )

lines.extend([
    '',
    '## Key Metrics Explained',
    '',
    '- **Threshold**: min_prob_long and p_side_min.long value',
    '- **Trade Count**: Total number of trades generated',
    '- **Mean BPS**: Average PnL per trade',
    '- **Median BPS**: Median PnL per trade (robust to outliers)',
    '- **Win Rate**: Percentage of profitable trades',
    '- **P01/P05**: 1st and 5th percentile (tail risk indicators)',
    '- **Max DD**: Maximum drawdown (largest peak-to-trough decline)',
    '- **Total PnL**: Sum of all trade PnL',
    '',
    '## Recommendations',
    '',
    'Compare thresholds based on:',
    '1. **Trade Count**: Lower threshold = more trades (less selectivity)',
    '2. **Mean/Median BPS**: Higher is better (profitability)',
    '3. **Win Rate**: Higher is better (consistency)',
    '4. **Tail Risk (P01/P05)**: Less negative is better (risk management)',
    '5. **Max Drawdown**: Less negative is better (risk management)',
    '',
    'Optimal threshold balances:',
    '- Sufficient trade volume for consistency',
    '- Positive expected value (mean/median BPS)',
    '- Acceptable risk profile (tail risk, drawdown)',
    '',
    '**Note:** These LOWZONE thresholds (<0.18) are compared with higher thresholds (0.18-0.28)',
    'to find the optimal baseline entry threshold.',
    '',
])

report_path.parent.mkdir(parents=True, exist_ok=True)
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f'✅ Combined LOWZONE summary saved to: {report_path}')
PYEOF

echo ""
echo "=================================================================================="
echo "✅ THRESHOLD SWEEP TEST LOWZONE COMPLETE"
echo "=================================================================================="
echo ""
echo "Summary report: reports/rl/entry_v10/ENTRY_V10_1_THRESHOLD_SWEEP_LOWZONE_2025.md"
echo ""

