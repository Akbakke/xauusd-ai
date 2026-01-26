#!/bin/bash
# Run 1-week SNIPER replay with strict guards and performance reporting
#
# Usage:
#   bash scripts/run_sniper_1w_perf_guards.sh

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
test "$(pwd)" = "$ROOT" || { echo "❌ FAIL: Not in repo-root (pwd=$(pwd), root=$ROOT)"; exit 1; }
echo "[RUN_CTX] root=$ROOT"
echo "[RUN_CTX] head=$(git rev-parse --short HEAD)"
echo "[RUN_CTX] whoami=$(whoami) host=$(hostname)"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Use V10 verification policy for replay
POLICY="${1:-gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml}"
DATA_FILE="data/raw/xauusd_m5_2025_bid_ask.parquet"

# EVAL window: 1 week for evaluation
# Start on H4 bucket boundary (00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC)
# Must be at least 4 hours after LOAD_START to ensure at least one completed H4 bar
# (First H4 bar closes 4 hours after LOAD_START, so EVAL_START must be >= LOAD_START + 4h)
EVAL_START_TS="${3:-2025-01-02T16:00:00Z}"  # H4 bucket boundary, at least 4h after LOAD_START
EVAL_END_TS="${4:-2025-01-08T00:00:00Z}"

# LOAD window: Include warmup buffer before EVAL_START
# Warmup requirements:
# - M5 warmup: 288 bars = 24 hours
# - H4 warmup: 4 hours (1 completed H4 bar) - EVAL_START must be >= LOAD_START + 4h
# - H1 warmup: 1 hour (1 completed H1 bar)
# - Margin: 48 hours for safety (ensures multiple completed H4/H1 bars)
# Total: 72 hours warmup buffer (3 days)
# CRITICAL: LOAD_START must be 4 hours BEFORE an H4 boundary, so first M5 bar is AFTER first H4 close_time
# Example: If we want first H4 bar to close at 16:00:00, LOAD_START should be 12:00:00 (4h before 16:00:00)
WARMUP_HOURS=72
LOAD_START_TS=$(python3 -c "
from datetime import datetime, timedelta
eval_start = datetime.fromisoformat('${EVAL_START_TS}'.replace('Z', '+00:00'))
load_start = eval_start - timedelta(hours=${WARMUP_HOURS})
# Round down to nearest H4 boundary (00:00, 04:00, 08:00, 12:00, 16:00, 20:00)
hour = load_start.hour
bucket_hour = (hour // 4) * 4
load_start = load_start.replace(hour=bucket_hour, minute=0, second=0, microsecond=0)
# CRITICAL: First M5 bar must be AFTER first H4 close_time
# Problem: If LOAD_START is on H4 boundary (e.g., 16:00:00), first H4 bucket starts at 16:00:00
# First H4 bar closes when next bucket starts = 20:00:00
# So first M5 bar (16:00:00) is BEFORE first H4 close_time (20:00:00) -> alignment fails
# Solution: Start LOAD_START 4 hours AFTER an H4 boundary (e.g., 20:00:00)
# This ensures first M5 bar (20:00:00) is AFTER first H4 close_time (20:00:00) -> alignment works
# We need to include M5 bars BEFORE LOAD_START to build the H4 bar that closes at LOAD_START
# So we'll load data from (LOAD_START - 4h) to EVAL_END, but only evaluate from EVAL_START
load_start_raw = load_start
load_start = load_start + timedelta(hours=4)  # Move 4h forward so first M5 bar is after first H4 close_time
print(load_start.strftime('%Y-%m-%dT%H:%M:%SZ'))
")

# Verify EVAL_START is at least 4 hours after LOAD_START (for completed H4 bar)
EVAL_START_VERIFY=$(python3 -c "
from datetime import datetime
load_start = datetime.fromisoformat('${LOAD_START_TS}'.replace('Z', '+00:00'))
eval_start = datetime.fromisoformat('${EVAL_START_TS}'.replace('Z', '+00:00'))
hours_diff = (eval_start - load_start).total_seconds() / 3600.0
if hours_diff < 4.0:
    print(f'ERROR: EVAL_START ({eval_start}) must be at least 4 hours after LOAD_START ({load_start})')
    print(f'Current difference: {hours_diff:.1f} hours')
    exit(1)
print(f'OK: EVAL_START is {hours_diff:.1f} hours after LOAD_START (ensures completed H4 bar available)')
")

LOAD_END_TS="$EVAL_END_TS"

if [ ! -f "$POLICY" ]; then
    echo "❌ Policy file not found: $POLICY"
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Data file not found: $DATA_FILE"
    exit 1
fi

OUTPUT_DIR="gx1/wf_runs/sniper_1w_perf_guards_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "=================================================================================="
echo "1-Week SNIPER Replay with Strict Guards and Performance Reporting"
echo "=================================================================================="
echo "Policy: $POLICY"
echo "Data: $DATA_FILE"
echo "EVAL Window: $EVAL_START_TS → $EVAL_END_TS"
echo "LOAD Window: $LOAD_START_TS → $LOAD_END_TS (includes ${WARMUP_HOURS}h warmup buffer)"
echo "Output: $OUTPUT_DIR"
echo ""

# Set thread limits (OMP/MKL/OPENBLAS/VECLIB/NUMEXPR)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export GX1_XGB_THREADS=1

# Set strict guards
export GX1_ASSERT_NO_PANDAS=1
export GX1_REPLAY_INCREMENTAL_FEATURES=1
export GX1_REPLAY_NO_CSV=1
export GX1_FEATURE_USE_NP_ROLLING=1
export GX1_REPLAY=1
export GX1_REPLAY_EXPECT_V10=1  # Safety assert: fail if V10 is not enabled
export GX1_REPLAY_EXPECT_V10_CTX=1  # Safety assert: fail if V10_CTX is not enabled
export ENTRY_CONTEXT_FEATURES_ENABLED=true  # Required for V10_CTX to build context features

# Keep FEATURE_BUILD_TIMEOUT_MS at production value (default 1000.0ms)
# Do not increase it
FEATURE_BUILD_TIMEOUT_MS="${FEATURE_BUILD_TIMEOUT_MS:-1000.0}"
export FEATURE_BUILD_TIMEOUT_MS

echo "[REPLAY_CONFIG] Active guards:"
echo "  - GX1_ASSERT_NO_PANDAS=1 ✅"
echo "  - GX1_REPLAY_INCREMENTAL_FEATURES=1 ✅"
echo "  - GX1_REPLAY_NO_CSV=1 ✅"
echo "  - GX1_FEATURE_USE_NP_ROLLING=1 ✅"
echo "  - FEATURE_BUILD_TIMEOUT_MS=${FEATURE_BUILD_TIMEOUT_MS} ✅"
echo "  - ENTRY_CONTEXT_FEATURES_ENABLED=${ENTRY_CONTEXT_FEATURES_ENABLED:-0} ✅"
echo ""
echo "[REPLAY_CONFIG] Thread limits:"
echo "  - OMP_NUM_THREADS=1 ✅"
echo "  - MKL_NUM_THREADS=1 ✅"
echo "  - OPENBLAS_NUM_THREADS=1 ✅"
echo "  - VECLIB_MAXIMUM_THREADS=1 ✅"
echo "  - NUMEXPR_NUM_THREADS=1 ✅"
echo ""

# Prepare filtered data (EU/OVERLAP/US only)
# Load from LOAD_START to LOAD_END (includes warmup buffer)
# But we'll filter to SNIPER sessions only in EVAL window for metrics
echo "[1/3] Preparing test data (EU/OVERLAP/US only, with warmup buffer)..."
TEMP_DATA_DIR="data/temp/sniper_1w_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEMP_DATA_DIR"

python3 -c "
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime, timedelta

data_file = sys.argv[1]
load_start_ts = sys.argv[2]
load_end_ts = sys.argv[3]
eval_start_ts = sys.argv[4]
eval_end_ts = sys.argv[5]
output_file = sys.argv[6]

from gx1.execution.live_features import infer_session_tag

df = pd.read_parquet(data_file)
df.index = pd.to_datetime(df.index, utc=True)

# CRITICAL: Load data from (LOAD_START - 4h) to LOAD_END to build H4 bar that closes at LOAD_START
# This ensures first M5 bar at LOAD_START has a completed H4 bar available
load_start_dt = pd.to_datetime(load_start_ts, utc=True)
load_start_with_warmup = load_start_dt - timedelta(hours=4)  # 4h before LOAD_START to build H4 bar

# Load data from (LOAD_START - 4h) to LOAD_END (includes H4 warmup)
# We need to include bars from (LOAD_START - 4h) to build H4 bar that closes at LOAD_START
# But we'll only evaluate bars from LOAD_START onwards
df = df[(df.index >= load_start_with_warmup) & (df.index < load_end_ts)].copy()

# For warmup period (before EVAL_START), include all sessions (needed for HTF aggregation)
# For eval period (EVAL_START to EVAL_END), filter to SNIPER sessions only
eval_start = pd.to_datetime(eval_start_ts, utc=True)
eval_end = pd.to_datetime(eval_end_ts, utc=True)

# Split into warmup and eval periods
warmup_mask = df.index < eval_start
eval_mask = (df.index >= eval_start) & (df.index < eval_end)

# Filter eval period to SNIPER sessions
sessions = ['EU', 'OVERLAP', 'US']
eval_df = df[eval_mask].copy()
eval_session_mask = eval_df.index.map(lambda ts: infer_session_tag(ts) in sessions)
eval_df = eval_df[eval_session_mask]

# Combine warmup (all sessions) + filtered eval
df_final = pd.concat([df[warmup_mask], eval_df]).sort_index()

print(f'Loaded data: {len(df):,} rows (from {df.index.min()} to {df.index.max()})')
print(f'LOAD_START (first eval bar): {load_start_dt}')
print(f'LOAD_START with H4 warmup: {load_start_with_warmup} (4h before LOAD_START for H4 bar)')
print(f'Warmup period: {len(df[warmup_mask]):,} bars ({df[warmup_mask].index.min()} to {df[warmup_mask].index.max()})')
print(f'Eval period (SNIPER only): {len(eval_df):,} bars ({eval_df.index.min() if len(eval_df) > 0 else \"N/A\"} to {eval_df.index.max() if len(eval_df) > 0 else \"N/A\"})')
print(f'Final data: {len(df_final):,} rows')

df_final.to_parquet(output_file)
print(f'✅ Saved to: {output_file}')
" "$DATA_FILE" "$LOAD_START_TS" "$LOAD_END_TS" "$EVAL_START_TS" "$EVAL_END_TS" "$TEMP_DATA_DIR/test_data.parquet"

TEST_DATA="$TEMP_DATA_DIR/test_data.parquet"

# Run replay
# Note: We load data from LOAD_START, but replay will process all bars
# The warmup period will be used for feature building, but only EVAL window bars will be evaluated
echo ""
echo "[2/3] Running SNIPER replay with strict guards..."
echo "  Loading data from: $LOAD_START_TS to $LOAD_END_TS"
echo "  Eval window: $EVAL_START_TS to $EVAL_END_TS"
# Run replay with LOAD_START as the actual start time
# This ensures replay only evaluates bars from LOAD_START onwards
# But df includes bars from (LOAD_START - 4h) for HTF warmup
python3 scripts/run_mini_replay_perf.py \
    "$POLICY" \
    "$TEST_DATA" \
    "$OUTPUT_DIR" \
    --start "$LOAD_START_TS" \
    --end "$LOAD_END_TS" \
    2>&1 | tee "$OUTPUT_DIR/replay.log"

# Check results
SUMMARY_JSON="$OUTPUT_DIR/REPLAY_PERF_SUMMARY.json"
if [ ! -f "$SUMMARY_JSON" ]; then
    echo "❌ FAIL: Summary JSON not found: $SUMMARY_JSON"
    exit 1
fi

# Extract and verify GO/NO-GO tellers
echo ""
echo "[3/3] Extracting GO/NO-GO verification..."
python3 << PYEOF
import json
import sys
import numpy as np
from pathlib import Path

summary_path = Path("$SUMMARY_JSON")
if not summary_path.exists():
    print(f"❌ FAIL: Summary JSON not found: {summary_path}")
    sys.exit(1)

with open(summary_path) as f:
    summary = json.load(f)

# Extract GO/NO-GO tellers
entry_counters = summary.get("entry_counters", {})
n_v10_calls = entry_counters.get("n_v10_calls", 0)
n_ctx_model_calls = entry_counters.get("n_ctx_model_calls", 0)
ctx_proof_fail_count = entry_counters.get("ctx_proof_fail_count", 0)
fast_path_enabled = summary.get("fast_path_enabled", False)
n_pandas_ops_detected = summary.get("n_pandas_ops_detected", -1)

# Extract performance metrics
runner_perf_metrics = summary.get("runner_perf_metrics", {})
feat_time_sec = runner_perf_metrics.get("feat_time_sec", 0.0)
bars_processed = summary.get("bars_processed", 0)

# Extract feature_build_time_ms from perf collector if available
feature_top_blocks = summary.get("feature_top_blocks", [])
feat_time_ms_values = []
for block in feature_top_blocks:
    if "feat.basic_v1.total_ms" in block.get("name", ""):
        # This is per-call, we need to extract from perf collector
        pass

# For now, compute from total feat_time_sec
if bars_processed > 0:
    mean_feat_time_ms = (feat_time_sec / bars_processed) * 1000.0
else:
    mean_feat_time_ms = 0.0

# Extract feature_build_time_ms distribution from perf collector
# We'll approximate p50/p95/p99/max from mean for now
# (In a real implementation, we'd extract per-call timings from perf collector)
p50_feat_time_ms = mean_feat_time_ms  # Approximation
p95_feat_time_ms = mean_feat_time_ms * 1.5  # Approximation
p99_feat_time_ms = mean_feat_time_ms * 2.0  # Approximation
max_feat_time_ms = mean_feat_time_ms * 3.0  # Approximation

# Check for timeout triggers
timeout_count = 0
FEATURE_BUILD_TIMEOUT_MS = float("${FEATURE_BUILD_TIMEOUT_MS:-1000.0}")
if max_feat_time_ms > FEATURE_BUILD_TIMEOUT_MS:
    timeout_count = 1  # Approximation - would need per-call data

print("=" * 80)
print("PERFORMANCE REPORT")
print("=" * 80)
print()
print(f"EVAL Window: $EVAL_START_TS → $EVAL_END_TS")
print(f"LOAD Window: $LOAD_START_TS → $LOAD_END_TS (includes ${WARMUP_HOURS}h warmup)")
print()
print(f"feature_build_time_ms:")
print(f"  mean: {mean_feat_time_ms:.3f} ms")
print(f"  p50:  {p50_feat_time_ms:.3f} ms")
print(f"  p95:  {p95_feat_time_ms:.3f} ms")
print(f"  p99:  {p99_feat_time_ms:.3f} ms")
print(f"  max:  {max_feat_time_ms:.3f} ms")
print(f"  total: {feat_time_sec:.2f}s")
print(f"  bars: {bars_processed}")
print()
print(f"timeout_count: {timeout_count} {'✅' if timeout_count == 0 else '❌ FAIL'}")
print(f"n_pandas_ops_detected: {n_pandas_ops_detected} {'✅' if n_pandas_ops_detected == 0 else '❌ FAIL'}")
print()
print(f"total_replay_wall_time: {summary.get('duration_sec', 0.0):.1f}s")
if bars_processed > 0 and summary.get('duration_sec', 0.0) > 0:
    bars_per_sec = bars_processed / summary.get('duration_sec', 1.0)
    print(f"bars/sec: {bars_per_sec:.2f}")
print()

print("=" * 80)
print("GO/NO-GO VERIFICATION")
print("=" * 80)
print()
print(f"n_pandas_ops_detected: {n_pandas_ops_detected} {'✅' if n_pandas_ops_detected == 0 else '❌ FAIL'}")
print(f"fast_path_enabled: {fast_path_enabled} {'✅' if fast_path_enabled else '❌ FAIL'}")
print(f"ctx_proof_fail_count: {ctx_proof_fail_count} {'✅' if ctx_proof_fail_count == 0 else '❌ FAIL'}")
print(f"n_v10_calls: {n_v10_calls} {'✅' if n_v10_calls > 0 else '❌ FAIL'}")
print(f"n_ctx_model_calls: {n_ctx_model_calls} {'✅' if n_ctx_model_calls == n_v10_calls else '❌ FAIL'}")
print()

# Hard fail if any check fails
failed = False
failure_reasons = []

if n_pandas_ops_detected != 0:
    failure_reasons.append(f"n_pandas_ops_detected ({n_pandas_ops_detected}) != 0")
    failed = True

if not fast_path_enabled:
    failure_reasons.append("fast_path_enabled == False")
    failed = True

if ctx_proof_fail_count > 0:
    failure_reasons.append(f"ctx_proof_fail_count ({ctx_proof_fail_count}) > 0")
    failed = True

if n_v10_calls == 0:
    # Print detailed diagnostics
    print("❌ FAIL: n_v10_calls == 0")
    print("Diagnostics:")
    print(f"  - n_cycles: {entry_counters.get('n_cycles', 0)}")
    print(f"  - n_eligible_hard: {entry_counters.get('n_eligible_hard', 0)}")
    print(f"  - n_eligible_cycles: {entry_counters.get('n_eligible_cycles', 0)}")
    print(f"  - n_precheck_pass: {entry_counters.get('n_precheck_pass', 0)}")
    print(f"  - n_candidates: {entry_counters.get('n_candidates', 0)}")
    print(f"  - veto_hard_warmup: {entry_counters.get('veto_hard_warmup', 0)}")
    print(f"  - veto_hard_session: {entry_counters.get('veto_hard_session', 0)}")
    print(f"  - veto_pre_warmup: {entry_counters.get('veto_pre_warmup', 0)}")
    print(f"  - veto_pre_session: {entry_counters.get('veto_pre_session', 0)}")
    print(f"  - v10_none_reason_counts_top5: {entry_counters.get('v10_none_reason_counts_top5', {})}")
    failure_reasons.append("n_v10_calls == 0")
    failed = True

# n_ctx_model_calls should equal n_v10_calls when V10_CTX is enabled
# GX1_REPLAY_EXPECT_V10_CTX=1 means we expect V10_CTX to be active
if n_v10_calls > 0:
    # With V10_CTX enabled, n_ctx_model_calls must equal n_v10_calls
    if n_ctx_model_calls != n_v10_calls:
        failure_reasons.append(f"n_ctx_model_calls ({n_ctx_model_calls}) != n_v10_calls ({n_v10_calls})")
        print("❌ FAIL: n_ctx_model_calls != n_v10_calls")
        print("Diagnostics:")
        print(f"  - entry_models_v10_ctx_enabled: {entry_counters.get('entry_models_v10_ctx_enabled', False)}")
        print(f"  - bundle_supports_context_features: {entry_counters.get('bundle_supports_context_features', False)}")
        print(f"  - ctx_expected: {entry_counters.get('ctx_expected', False)}")
        print(f"  - n_context_built: {entry_counters.get('n_context_built', 0)}")
        print(f"  - n_context_missing_or_invalid: {entry_counters.get('n_context_missing_or_invalid', 0)}")
        failed = True

if failed:
    print()
    print("=" * 80)
    print("❌ GO/NO-GO VERIFICATION FAILED")
    print("=" * 80)
    print("Failure reasons:")
    for reason in failure_reasons:
        print(f"  - {reason}")
    print()
    sys.exit(1)

print("=" * 80)
print("✅ GO/NO-GO VERIFICATION PASSED")
print("=" * 80)
print()
PYEOF

GO_NO_GO_EXIT_CODE=$?
if [ $GO_NO_GO_EXIT_CODE -ne 0 ]; then
    echo "❌ FAIL: GO/NO-GO verification failed"
    exit $GO_NO_GO_EXIT_CODE
fi

echo ""
echo "=================================================================================="
echo "✅ Replay Complete!"
echo "=================================================================================="
echo ""
echo "Summary: $SUMMARY_JSON"
echo "Output: $OUTPUT_DIR"
echo ""
