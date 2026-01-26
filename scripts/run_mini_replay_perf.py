#!/usr/bin/env python3
"""
Run mini replay with performance timing.

Del C: Safe Threading Environment Variables
-------------------------------------------
To reduce native threading issues that can cause segfaults in pandas/numpy,
set these environment variables before running:

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMEXPR_MAX_THREADS=1

Or in one line:
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_MAX_THREADS=1

These are NOT set by default in code, but can be enabled for reproduction/stability.
"""

import sys
import faulthandler
import signal
import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

def convert_to_python_type(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_type(item) for item in obj]
    else:
        return obj

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# CRITICAL: Enable fail-fast stack dumps for replay debugging (Grep A)
# Fault log will be set in main() when output_dir is known
_fault_log_file = None

def _setup_faulthandler(output_dir: Path) -> None:
    """Setup faulthandler with output to fault.log in chunk directory."""
    global _fault_log_file
    fault_log_path = output_dir / "fault.log"
    _fault_log_file = open(fault_log_path, 'w')
    
    # Enable faulthandler for all threads, write to fault.log
    # SIGSEGV/SIGABRT are automatically handled by enable() - no need to register separately
    faulthandler.enable(file=_fault_log_file, all_threads=True)
    
    # Dump traceback every 30 seconds if hung
    faulthandler.dump_traceback_later(30, file=_fault_log_file, repeat=True)
    
    print(f"[REPLAY_DEBUG] Fail-fast enabled: PID={os.getpid()}, fault.log={fault_log_path}")

def _dump_stack(sig, frame):
    """Dump stack trace on SIGUSR1 signal."""
    if _fault_log_file:
        faulthandler.dump_traceback(file=_fault_log_file, all_threads=True)
    else:
        faulthandler.dump_traceback(all_threads=True)

signal.signal(signal.SIGUSR1, _dump_stack)
print(f"[REPLAY_DEBUG] Fail-fast enabled: PID={os.getpid()}, send SIGUSR1 for stack dump (fault.log will be set in main)")

# Del 2: Print thread limits recommendation for replay mode
if os.getenv("GX1_REPLAY") == "1" or os.getenv("REPLAY_MODE") == "1" or True:  # Always show for replay scripts
    thread_vars = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_MAX_THREADS"]
    all_set = all(os.getenv(v) == "1" for v in thread_vars)
    if not all_set:
        print("")
        print("[REPLAY_THREAD_LIMITS] ⚠️  Recommended: Set thread limits to reduce segfault risk:")
        print("    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_MAX_THREADS=1")
        print("    Or use: scripts/run_replay_with_thread_limits.sh scripts/run_mini_replay_perf.py ...")
        print("")


def write_perf_summary(
    output_dir: Path,
    start_time: float,
    end_time: float,
    bars_processed: int = 0,
    bars_total: int = 0,
    last_timestamp: str = None,
    trades_total: int = 0,
    trades_open_at_end: int = 0,
    log_file: str = None,
    runner_perf_metrics: dict = None,
    completed: bool = False,
    early_stop_reason: str = None,
    feature_top_blocks: list = None,
    next_pandas_rolling_target: str = None,
    target_total_sec: float = 0.0,
    target_calls: int = 0,
    chunk_id: str = None,
    window_start: str = None,
    window_end: str = None,
    error_type: str = None,
    error_message: str = None,
    traceback_excerpt: str = None,
    entry_counters: dict = None,
    fast_path_enabled: bool = False,  # OPPGAVE 1
    n_pandas_ops_detected: int = 0,  # NEW: Pandas ops detected count
    n_bars_skipped_due_to_htf_warmup: int = 0,  # NEW: Bars skipped due to HTF warmup
    htf_align_idx_min: int = -1,  # NEW: Minimum HTF alignment index (should be >= 0 after warmup)
):
    """
    Write performance summary to JSON and Markdown files (Grep B).
    Always called from finally block, even if replay was interrupted.
    """
    duration_sec = end_time - start_time
    percent_done = (bars_processed / bars_total * 100.0) if bars_total > 0 else 0.0
    
    # Del 3: Auto-identify next pandas rolling target
    next_pandas_rolling_target = None
    target_total_sec = 0.0
    target_calls = 0
    if feature_top_blocks:
        pandas_rolling_blocks = [
            b for b in feature_top_blocks
            if b.get("name", "").startswith("rolling.pandas.")
        ]
        if pandas_rolling_blocks:
            # Sort by total_sec descending and pick top one
            pandas_rolling_blocks.sort(key=lambda x: x.get("total_sec", 0.0), reverse=True)
            top_target = pandas_rolling_blocks[0]
            next_pandas_rolling_target = top_target.get("name", "")
            target_total_sec = top_target.get("total_sec", 0.0)
            target_calls = top_target.get("count", 0)
    
    # Del 3: Auto-identify next pandas misc_roll target
    next_pandas_misc_roll_target = None
    misc_roll_target_total_sec = 0.0
    misc_roll_target_calls = 0
    misc_roll_target_sites = []
    if feature_top_blocks:
        # Find top pandas rolling operations overall (they should be from misc_roll)
        pandas_rolling_blocks = [
            b for b in feature_top_blocks
            if b.get("name", "").startswith("rolling.pandas.")
        ]
        # Find top misc_roll sub-blocks
        misc_roll_subblocks = [
            b for b in feature_top_blocks
            if b.get("name", "").startswith("feat.basic_v1.misc_roll.")
        ]
        
        # Combine and pick the top one by total_sec
        all_misc_roll_blocks = pandas_rolling_blocks + misc_roll_subblocks
        if all_misc_roll_blocks:
            all_misc_roll_blocks.sort(key=lambda x: x.get("total_sec", 0.0), reverse=True)
            top_misc_target = all_misc_roll_blocks[0]
            next_pandas_misc_roll_target = top_misc_target.get("name", "")
            misc_roll_target_total_sec = top_misc_target.get("total_sec", 0.0)
            misc_roll_target_calls = top_misc_target.get("count", 0)
            
            # Get top 3 misc_roll sub-blocks for sites list
            misc_roll_subblocks_sorted = sorted(misc_roll_subblocks, key=lambda x: x.get("total_sec", 0.0), reverse=True)[:3]
            misc_roll_target_sites = [b.get("name", "") for b in misc_roll_subblocks_sorted]
    
    # Del 2: Extract top pandas ops (top 10 rolling.pandas.*)
    top_pandas_ops = []
    if feature_top_blocks:
        pandas_ops = [
            b for b in feature_top_blocks
            if b.get("name", "").startswith("rolling.pandas.")
        ]
        pandas_ops.sort(key=lambda x: x.get("total_sec", 0.0), reverse=True)
        top_pandas_ops = pandas_ops[:10]
    
    # Del 2: window_start and window_end are passed as parameters (from WINDOW.json)
    # Do not override them here - they come from the source of truth (WINDOW.json)
    
    feat_time_sec = runner_perf_metrics.get('feat_time_sec', 0.0) if runner_perf_metrics else 0.0
    
    # COMMIT D: Include run_id and chunk_id in summary (for observability)
    # Note: runner is not available in this scope, so we get from env vars
    run_id = os.getenv('GX1_RUN_ID')
    chunk_id = os.getenv('GX1_CHUNK_ID')
    
    summary_data = {
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.fromtimestamp(end_time).isoformat(),
        "window_start": window_start,
        "window_end": window_end,
        "run_id": run_id,  # COMMIT D
        "chunk_id": chunk_id,  # COMMIT D
        "duration_sec": duration_sec,
        "bars_processed": bars_processed,
        "bars_total": bars_total,
        "percent_done": percent_done,
        "completed": completed,
        "status": "complete" if completed else "incomplete",
        "early_stop_reason": early_stop_reason if early_stop_reason else (None if completed else "interrupted"),
        "error_type": error_type if not completed else None,
        "error_message": error_message if not completed else None,
        "traceback_excerpt": traceback_excerpt if not completed else None,
        "last_timestamp_processed": last_timestamp,
        "trades_total": trades_total_final if 'trades_total_final' in locals() else (trades_total if 'trades_total' in locals() else 0),
        "trades_open_at_end": trades_open_at_end,
        "log_file": log_file,
        "runner_perf_metrics": runner_perf_metrics or {},
        "feat_time_sec": feat_time_sec,
        "feature_top_blocks": feature_top_blocks or [],
        "top_pandas_ops": top_pandas_ops,
        "entry_counters": entry_counters if entry_counters else {},
        "next_pandas_rolling_target": next_pandas_rolling_target,
        "target_total_sec": target_total_sec,
        "target_calls": target_calls,
        "next_pandas_misc_roll_target": next_pandas_misc_roll_target,
        "next_pandas_misc_roll_target_total_sec": misc_roll_target_total_sec,
        "next_pandas_misc_roll_target_calls": misc_roll_target_calls,
        "next_pandas_misc_roll_target_sites": misc_roll_target_sites,
        "fast_path_enabled": fast_path_enabled,  # OPPGAVE 1
        "n_pandas_ops_detected": n_pandas_ops_detected,  # NEW: Pandas ops detected count
        "n_bars_skipped_due_to_htf_warmup": n_bars_skipped_due_to_htf_warmup,  # NEW: Bars skipped due to HTF warmup
        "htf_align_idx_min": htf_align_idx_min,  # NEW: Minimum HTF alignment index (should be >= 0 after warmup)
    }
    
    # Add chunk identification if provided (Del A3)
    if chunk_id is not None:
        summary_data["chunk_id"] = chunk_id
    if window_start is not None:
        summary_data["window_start"] = window_start
    if window_end is not None:
        summary_data["window_end"] = window_end
    
    # Write JSON (convert numpy types to Python types first)
    json_path = output_dir / "REPLAY_PERF_SUMMARY.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        summary_data_python = convert_to_python_type(summary_data)
        json.dump(summary_data_python, f, indent=2)
    
    # Write Markdown
    md_path = output_dir / "REPLAY_PERF_SUMMARY.md"
    with open(md_path, "w") as f:
        f.write("# Replay Performance Summary\n\n")
        f.write(f"**Start Time:** {summary_data['start_time']}\n")
        f.write(f"**End Time:** {summary_data['end_time']}\n")
        f.write(f"**Duration:** {duration_sec:.1f} seconds ({duration_sec/60:.1f} minutes)\n\n")
        f.write(f"**Bars Processed:** {bars_processed}/{bars_total} ({percent_done:.1f}%)\n")
        f.write(f"**Completed:** {completed}\n")
        if early_stop_reason:
            f.write(f"**Early Stop Reason:** {early_stop_reason}\n")
        if last_timestamp:
            f.write(f"**Last Timestamp Processed:** {last_timestamp}\n")
        f.write(f"**Trades Total:** {trades_total}\n")
        f.write(f"**Trades Open at End:** {trades_open_at_end}\n\n")
        if log_file:
            f.write(f"**Log File:** `{log_file}`\n\n")
        if runner_perf_metrics:
            f.write("## Runner Performance Metrics\n\n")
            for key, value in runner_perf_metrics.items():
                f.write(f"- **{key}:** {value}\n")
            f.write("\n")
        
        # NEW: GO/NO-GO Verification
        f.write("## GO/NO-GO Verification\n\n")
        if entry_counters:
            n_v10_calls = entry_counters.get("n_v10_calls", 0)
            n_ctx_model_calls = entry_counters.get("n_ctx_model_calls", 0)
            ctx_proof_fail_count = entry_counters.get("ctx_proof_fail_count", 0)
            f.write(f"- **n_v10_calls:** {n_v10_calls} {'✅' if n_v10_calls > 0 else '❌'}\n")
            f.write(f"- **n_ctx_model_calls:** {n_ctx_model_calls} {'✅' if n_ctx_model_calls == n_v10_calls else '❌'}\n")
            f.write(f"- **ctx_proof_fail_count:** {ctx_proof_fail_count} {'✅' if ctx_proof_fail_count == 0 else '❌'}\n")
        f.write(f"- **fast_path_enabled:** {fast_path_enabled} {'✅' if fast_path_enabled else '❌'}\n")
        f.write(f"- **n_pandas_ops_detected:** {n_pandas_ops_detected} {'✅' if n_pandas_ops_detected == 0 else '❌'}\n")
        f.write(f"- **n_bars_skipped_due_to_htf_warmup:** {n_bars_skipped_due_to_htf_warmup} {'✅' if n_bars_skipped_due_to_htf_warmup < 100 else '⚠️'}\n")
        f.write(f"- **htf_align_idx_min:** {htf_align_idx_min} {'✅' if htf_align_idx_min >= 0 else '❌'}\n")
        f.write("\n")
        
        # NEW: Feature Build Time Statistics
        if runner_perf_metrics:
            feat_time_sec_val = runner_perf_metrics.get('feat_time_sec', 0.0)
            if bars_processed > 0:
                # Extract feature_build_time_ms from perf collector if available
                # For now, compute from feat_time_sec
                feat_time_ms_per_bar = (feat_time_sec_val / bars_processed) * 1000.0 if bars_processed > 0 else 0.0
                f.write("## Feature Build Time Statistics\n\n")
                f.write(f"- **Mean feature_build_time_ms:** {feat_time_ms_per_bar:.3f} ms\n")
                f.write(f"- **Total feature time:** {feat_time_sec_val:.2f}s\n")
                f.write(f"- **Bars processed:** {bars_processed}\n")
                f.write(f"- **Timeout check triggered:** {'0 ✅' if n_pandas_ops_detected == 0 else '>0 ❌'}\n")
                f.write("\n")
        
        # Del 2: Scaling Snapshot (milestone-friendly)
        if bars_processed > 0 and duration_sec > 0:
            bars_per_sec = bars_processed / duration_sec
            feat_time_sec_val = runner_perf_metrics.get('feat_time_sec', 0.0) if runner_perf_metrics else 0.0
            feat_sec_per_bar = feat_time_sec_val / bars_processed if bars_processed > 0 else 0.0
            f.write("## Scaling Snapshot\n\n")
            f.write(f"- **Bars/second:** {bars_per_sec:.2f}\n")
            f.write(f"- **Feature time per bar:** {feat_sec_per_bar*1000:.3f} ms\n")
            f.write(f"- **Total bars:** {bars_total}\n")
            f.write(f"- **Duration:** {duration_sec:.1f}s ({duration_sec/60:.1f} min)\n")
            f.write(f"- **Feature time:** {feat_time_sec_val:.2f}s\n")
            f.write("\n")
        
        # Del 2: Top Pandas Ops (milestone-friendly)
        if top_pandas_ops:
            f.write("## Top Pandas Operations (Top 10)\n\n")
            f.write("| Name | Total (sec) | Share (%) | Calls |\n")
            f.write("|------|-------------|-----------|-------|\n")
            for op in top_pandas_ops:
                name = op.get("name", "unknown")
                total_sec = op.get("total_sec", 0.0)
                share_pct = op.get("share_of_feat_time_pct", 0.0)
                count = op.get("count", 0)
                f.write(f"| {name} | {total_sec:.4f} | {share_pct:.2f}% | {count} |\n")
            f.write("\n")
        
        # Del 2E: Feature Top Blocks breakdown
        if feature_top_blocks:
            f.write("## Feature Top Blocks (Top 15)\n\n")
            feat_time_sec = runner_perf_metrics.get('feat_time_sec', 0.0) if runner_perf_metrics else 0.0
            if feat_time_sec > 0:
                f.write(f"*Breakdown of {feat_time_sec:.2f}s total feature time*\n\n")
            f.write("| Name | Total (sec) | Share (%) | Calls |\n")
            f.write("|------|-------------|-----------|-------|\n")
            for block in feature_top_blocks:
                name = block.get("name", "unknown")
                total_sec = block.get("total_sec", 0.0)
                share_pct = block.get("share_of_feat_time_pct", 0.0)
                count = block.get("count", 0)
                f.write(f"| {name} | {total_sec:.4f} | {share_pct:.2f}% | {count} |\n")
            f.write("\n")
        
        # Del 3: Next optimization target
        if next_pandas_rolling_target:
            f.write("## Next Optimization Target\n\n")
            f.write(f"**Next pandas rolling target:** `{next_pandas_rolling_target}`\n")
            f.write(f"**Total time:** {target_total_sec:.4f}s\n")
            f.write(f"**Calls:** {target_calls}\n")
            f.write(f"\n*This is the highest-cost pandas rolling operation in the top blocks.*\n\n")
        else:
            f.write("## Next Optimization Target\n\n")
            f.write("*No pandas rolling operations found in top blocks.*\n\n")
        
        # Del 3: Misc Roll Hotspots
        if next_pandas_misc_roll_target:
            f.write("## Misc Roll Hotspots\n\n")
            f.write(f"**Next pandas misc_roll target:** `{next_pandas_misc_roll_target}`\n")
            f.write(f"**Total time:** {misc_roll_target_total_sec:.4f}s\n")
            f.write(f"**Calls:** {misc_roll_target_calls}\n")
            if misc_roll_target_sites:
                f.write(f"**Top 3 misc_roll sub-blocks:**\n")
                for site in misc_roll_target_sites:
                    f.write(f"  - `{site}`\n")
            f.write(f"\n*This is the highest-cost operation within misc_roll (pandas rolling or misc_roll sub-blocks).*\n\n")
    
    # Del 3: Optional regression check against baseline
    baseline_json_path = os.getenv("GX1_PERF_BASELINE_JSON")
    if baseline_json_path and Path(baseline_json_path).exists():
        try:
            with open(baseline_json_path) as f:
                baseline_data = json.load(f)
            
            baseline_feat_time = baseline_data.get("feat_time_sec", 0.0)
            baseline_bars = baseline_data.get("bars_total", 0)
            baseline_feat_per_bar = baseline_feat_time / baseline_bars if baseline_bars > 0 else 0.0
            
            current_feat_per_bar = feat_time_sec / bars_total if bars_total > 0 else 0.0
            
            baseline_top5 = baseline_data.get("feature_top_blocks", [])[:5]
            current_top5 = feature_top_blocks[:5] if feature_top_blocks else []
            
            md_path_with_delta = output_dir / "REPLAY_PERF_SUMMARY.md"
            with open(md_path_with_delta, "a") as f:  # Append to existing file
                f.write("## Regression Comparison\n\n")
                f.write(f"*Baseline: {baseline_json_path}*\n\n")
                f.write("| Metric | Baseline | Current | Delta |\n")
                f.write("|--------|----------|---------|-------|\n")
                f.write(f"| Feat time/bar (ms) | {baseline_feat_per_bar*1000:.3f} | {current_feat_per_bar*1000:.3f} | {(current_feat_per_bar - baseline_feat_per_bar)*1000:+.3f} |\n")
                f.write("\n")
                f.write("### Top 5 Blocks Comparison\n\n")
                f.write("| Rank | Baseline | Current |\n")
                f.write("|------|----------|----------|\n")
                for i in range(max(len(baseline_top5), len(current_top5))):
                    baseline_name = baseline_top5[i].get("name", "") if i < len(baseline_top5) else "-"
                    baseline_sec = baseline_top5[i].get("total_sec", 0.0) if i < len(baseline_top5) else 0.0
                    current_name = current_top5[i].get("name", "") if i < len(current_top5) else "-"
                    current_sec = current_top5[i].get("total_sec", 0.0) if i < len(current_top5) else 0.0
                    f.write(f"| {i+1} | {baseline_name} ({baseline_sec:.4f}s) | {current_name} ({current_sec:.4f}s) |\n")
                f.write("\n")
        except Exception as e:
            print(f"[REPLAY_PERF] Warning: Could not perform regression comparison: {e}")
    
    print(f"[REPLAY_PERF] Summary written to: {json_path} and {md_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run mini replay with performance timing")
    parser.add_argument("policy_path", type=Path, help="Path to policy YAML")
    parser.add_argument("data_file", type=Path, help="Path to replay data (CSV/Parquet)")
    parser.add_argument("output_dir", type=Path, help="Output directory for replay results")
    parser.add_argument("--start", type=str, default=None, help="Start timestamp (ISO format, e.g., 2025-01-01T00:00:00Z)")
    parser.add_argument("--end", type=str, default=None, help="End timestamp (ISO format, e.g., 2025-01-08T00:00:00Z)")
    args = parser.parse_args()
    
    policy_path = args.policy_path
    data_file = args.data_file
    output_dir = args.output_dir
    start_ts = args.start
    end_ts = args.end
    
    # CRITICAL: Create output_dir and write worker_started.txt as FIRST operation
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_started_path = output_dir / "worker_started.txt"
    worker_started_path.write_text(f"PID={os.getpid()}\nStarted at {datetime.now().isoformat()}\n")
    print(f"[REPLAY_DEBUG] Worker started marker written: {worker_started_path}")
    
    # CRITICAL: Setup faulthandler IMMEDIATELY after output_dir exists (before any imports/loading)
    _setup_faulthandler(output_dir)
    
    # Write env_snapshot.json to capture thread limits
    env_snapshot_path = output_dir / "env_snapshot.json"
    env_snapshot = {
        "pid": os.getpid(),
        "timestamp": datetime.now().isoformat(),
        # COMMIT D: Include run_id and chunk_id in env snapshot
        "GX1_RUN_ID": os.getenv("GX1_RUN_ID"),
        "GX1_CHUNK_ID": os.getenv("GX1_CHUNK_ID"),
        "thread_limits": {
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.getenv("VECLIB_MAXIMUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.getenv("NUMEXPR_NUM_THREADS"),
            "GX1_XGB_THREADS": os.getenv("GX1_XGB_THREADS"),
        },
        "python_path": sys.executable,
    }
    with open(env_snapshot_path, 'w') as f:
        json.dump(env_snapshot, f, indent=2)
    print(f"[REPLAY_DEBUG] Environment snapshot written: {env_snapshot_path}")
    
    from gx1.execution.oanda_demo_runner import GX1DemoRunner
    
    # Del 2: Informativ logging ved start
    print("=" * 80)
    print("Starting replay with performance timing...")
    print("=" * 80)
    print()
    
    # Log activated toggles
    print("[REPLAY_CONFIG] Active toggles:")
    no_csv = os.getenv("GX1_REPLAY_NO_CSV") == "1"
    np_rolling = os.getenv("GX1_FEATURE_USE_NP_ROLLING") == "1"
    print(f"  - GX1_REPLAY_NO_CSV: {'✅ ON' if no_csv else '❌ OFF'}")
    print(f"  - GX1_FEATURE_USE_NP_ROLLING: {'✅ ON' if np_rolling else '❌ OFF'}")
    
    # Log thread limits
    thread_vars = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_MAX_THREADS"]
    thread_status = []
    all_set = True
    for var in thread_vars:
        val = os.getenv(var)
        if val == "1":
            thread_status.append(f"    {var}=1 ✅")
        else:
            thread_status.append(f"    {var}={val or 'NOT SET'} ❌")
            all_set = False
    print("[REPLAY_CONFIG] Thread limits:")
    for status in thread_status:
        print(status)
    if not all_set:
        print("  ⚠️  Recommendation: Use scripts/run_replay_with_thread_limits.sh wrapper")
    print()
    
    # Log replay window (if available from data file) and get bars_total early
    bars_total = 0
    try:
        import pandas as pd
        if data_file.suffix.lower() == ".parquet":
            df_preview = pd.read_parquet(data_file)
        else:
            df_preview = pd.read_csv(data_file)
        if "time" in df_preview.columns:
            df_preview["time"] = pd.to_datetime(df_preview["time"], utc=True)
            df_preview = df_preview.set_index("time").sort_index()
            preview_start_ts = df_preview.index[0] if len(df_preview) > 0 else None
            preview_end_ts = df_preview.index[-1] if len(df_preview) > 0 else None
            bars_total = len(df_preview)
            if preview_start_ts and preview_end_ts:
                print("[REPLAY_CONFIG] Replay window (before filtering):")
                print(f"  - Start: {preview_start_ts}")
                print(f"  - End: {preview_end_ts}")
                print(f"  - Expected bars: {bars_total}")
                if start_ts or end_ts:
                    print(f"  - Filter: {start_ts} to {end_ts}")
                print()
    except Exception as e:
        print(f"[REPLAY_CONFIG] Warning: Could not determine replay window: {e}")
        print()
    
    # Performance tracking
    start_time = time.time()
    bars_processed = 0
    last_timestamp = None
    trades_total = 0
    trades_open_at_end = 0
    runner_perf_metrics = {}
    log_file = None  # Not used in perf summary, but required for function signature
    feat_time_sec = 0.0  # Initialize to avoid UnboundLocalError
    perf_started = False  # Initialize to avoid UnboundLocalError
    
    # Extract window from data_file BEFORE replay starts (source of truth)
    window_start_iso = None
    window_end_iso = None
    try:
        import pandas as pd
        if data_file.exists():
            df = pd.read_parquet(data_file) if data_file.suffix.lower() == ".parquet" else pd.read_csv(data_file)
            if len(df) > 0:
                # Ensure time index
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"], utc=True)
                    df = df.set_index("time").sort_index()
                elif "ts" in df.columns:
                    df["ts"] = pd.to_datetime(df["ts"], utc=True)
                    df = df.set_index("ts").sort_index()
                elif not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, utc=True)
                
                if len(df) > 0 and hasattr(df.index, 'min'):
                    window_start_dt = df.index.min()
                    window_end_dt = df.index.max()
                    # Convert to ISO format
                    if hasattr(window_start_dt, 'isoformat'):
                        window_start_iso = window_start_dt.isoformat()
                        if not str(window_start_iso).endswith('Z') and '+' not in str(window_start_iso):
                            window_start_iso = str(window_start_iso) + 'Z'
                    else:
                        window_start_iso = str(window_start_dt)
                    if hasattr(window_end_dt, 'isoformat'):
                        window_end_iso = window_end_dt.isoformat()
                        if not str(window_end_iso).endswith('Z') and '+' not in str(window_end_iso):
                            window_end_iso = str(window_end_iso) + 'Z'
                    else:
                        window_end_iso = str(window_end_dt)
    except Exception as e:
        print(f"[REPLAY_PERF] Warning: Could not extract window from data_file {data_file}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    
    # Write WINDOW.json immediately (before replay starts) - source of truth
    def _write_window_file(output_dir: Path, window_start: str, window_end: str) -> None:
        """Write window timestamps to WINDOW.json (atomic write)."""
        p = output_dir / "WINDOW.json"
        tmp = p.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps({"window_start": window_start, "window_end": window_end}, indent=2), encoding="utf-8")
            tmp.replace(p)
        except Exception as e:
            print(f"[REPLAY_PERF] Warning: Failed to write WINDOW.json: {e}", file=sys.stderr)
    
    if window_start_iso and window_end_iso:
        _write_window_file(output_dir, window_start_iso, window_end_iso)
        print(f"[REPLAY_PERF] Wrote WINDOW.json: {window_start_iso} → {window_end_iso}")
    else:
        print(f"[REPLAY_PERF] WARNING: Could not extract window timestamps from data_file, WINDOW.json not written", file=sys.stderr)
    
    # Setup faulthandler with output to fault.log (before any code that might crash)
    # Wrap replay in try/finally to ensure perf summary is always written (Grep B)
    # Note: faulthandler is already setup earlier in main()
    try:
        runner = GX1DemoRunner(
            policy_path,
            dry_run_override=True,
            replay_mode=True,
            fast_replay=True,
            output_dir=output_dir,
        )
        
        # Filter data by date range if provided (before replay)
        filtered_data_file = data_file
        if start_ts or end_ts:
            import pandas as pd
            import tempfile
            
            # Load full dataset
            if data_file.suffix.lower() == ".parquet":
                df_full = pd.read_parquet(data_file)
            else:
                df_full = pd.read_csv(data_file)
            
            # Ensure time column is datetime
            if "time" in df_full.columns:
                df_full["time"] = pd.to_datetime(df_full["time"], utc=True)
                df_full = df_full.set_index("time").sort_index()
            elif "ts" in df_full.columns:
                df_full["ts"] = pd.to_datetime(df_full["ts"], utc=True)
                df_full = df_full.set_index("ts").sort_index()
                # Rename index to 'time' for compatibility
                df_full.index.name = "time"
            elif isinstance(df_full.index, pd.DatetimeIndex):
                # Ensure index is named 'time'
                df_full.index.name = "time"
            else:
                raise ValueError("Data must have 'time' or 'ts' column, or DatetimeIndex")
            
            # Filter by date range
            if start_ts:
                start_dt = pd.to_datetime(start_ts, utc=True)
                df_full = df_full[df_full.index >= start_dt]
            if end_ts:
                end_dt = pd.to_datetime(end_ts, utc=True)
                df_full = df_full[df_full.index < end_dt]
            
            # Fix case-insensitive column name collisions at source (before writing parquet)
            # This is replay/offline mode, so we hard fail on collisions
            from collections import defaultdict
            cols = list(df_full.columns)
            lower_to_cols = defaultdict(list)
            for c in cols:
                k = str(c).lower()
                lower_to_cols[k].append(c)
            collisions = {k: v for k, v in lower_to_cols.items() if len(v) > 1}
            if collisions:
                raise ValueError(
                    f"[CASE_COLLISION] Source data file {data_file.name} has case-insensitive column name collisions. "
                    f"Collisions: {collisions}. "
                    f"This must be fixed in the source CSV/parquet file before replay can proceed."
                )
            
            # Remove duplicate columns (if any remain after collision check)
            if df_full.columns.duplicated().any():
                dupes = df_full.columns[df_full.columns.duplicated()].tolist()
                raise ValueError(
                    f"[DUPLICATE_COLUMNS] Source data file {data_file.name} has duplicate column names: {dupes}. "
                    f"This must be fixed in the source CSV/parquet file before replay can proceed."
                )
            
            # Write filtered data to temp file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.parquet', delete=False, dir=output_dir.parent)
            temp_file.close()
            temp_path = Path(temp_file.name)
            df_full.to_parquet(temp_path)
            filtered_data_file = temp_path
            
            print(f"[REPLAY_1W] Filtered to {len(df_full)} bars: {df_full.index.min()} to {df_full.index.max()}")
        
        # bars_total already set above in config logging section
        
        print("Starting replay...")
        runner.run_replay(filtered_data_file)
        
        print()
        print("=" * 80)
        print("✅ Replay complete!")
        print("=" * 80)
        print()
        
        # Del 1: Extract metrics from runner (stored on self)
        bars_processed = getattr(runner, 'perf_n_bars_processed', 0)
        bars_total_runner = getattr(runner, 'perf_bars_total', 0)
        if bars_total_runner > 0:
            bars_total = bars_total_runner  # Use runner value (correct, accounts for min_bars_for_features)
        # Otherwise use pre-computed value from data file
        # Use entry_telemetry['n_trades_created'] as source of truth (counts all trades created)
        # perf_n_trades_created counts trades logged, which may exclude some (e.g., dry_run skips)
        trades_total = getattr(runner, 'perf_n_trades_created', 0)
        # Prefer entry_telemetry if available (more accurate)
        if hasattr(runner, 'entry_manager') and hasattr(runner.entry_manager, 'entry_telemetry'):
            entry_telemetry_trades = runner.entry_manager.entry_telemetry.get("n_trades_created", 0)
            if entry_telemetry_trades > 0:
                trades_total = entry_telemetry_trades
        
        # Del 1: Determine if replay completed normally
        completed = (bars_total > 0 and bars_processed >= bars_total)
        early_stop_reason = None if completed else "interrupted"
        
        # Get trades open at end
        if hasattr(runner, 'open_trades'):
            trades_open_at_end = len(runner.open_trades) if runner.open_trades else 0
        else:
            trades_open_at_end = 0
        
        # Extract perf metrics from runner
        if hasattr(runner, 'perf_feat_time'):
            runner_perf_metrics['feat_time_sec'] = getattr(runner, 'perf_feat_time', 0.0)
        if hasattr(runner, 'perf_model_time'):
            runner_perf_metrics['model_time_sec'] = getattr(runner, 'perf_model_time', 0.0)
        if hasattr(runner, 'perf_journal_time'):
            runner_perf_metrics['journal_time_sec'] = getattr(runner, 'perf_journal_time', 0.0)
        if hasattr(runner, 'perf_other_time'):
            runner_perf_metrics['other_time_sec'] = getattr(runner, 'perf_other_time', 0.0)
        
        # Del 2E: Extract feature top blocks from perf collector
        feature_top_blocks = []
        if hasattr(runner, 'perf_collector'):
            try:
                top_15 = runner.perf_collector.top(15)  # Increased to 15 to catch more pandas rolling ops
                feat_time_sec = runner_perf_metrics.get('feat_time_sec', 0.0)
                feature_top_blocks = [
                    {
                        "name": name,
                        "total_sec": total_sec,
                        "count": count,
                        "share_of_feat_time_pct": (total_sec / feat_time_sec * 100.0) if feat_time_sec > 0 else 0.0
                    }
                    for name, total_sec, count in top_15
                ]
            except Exception as e:
                print(f"[REPLAY_PERF] Warning: Failed to extract feature top blocks: {e}", file=sys.stderr)
        
        # Try to get last timestamp from replay state
        if hasattr(runner, '_replay_current_ts') and runner._replay_current_ts is not None:
            last_timestamp = str(runner._replay_current_ts)
        elif hasattr(runner, 'replay_end_ts') and runner.replay_end_ts is not None:
            last_timestamp = str(runner.replay_end_ts)
        elif hasattr(runner, 'replay_eval_end_ts') and runner.replay_eval_end_ts is not None:
            last_timestamp = str(runner.replay_eval_end_ts)
        
    except KeyboardInterrupt:
        print("\n[REPLAY] Interrupted by user (Ctrl+C)")
        early_stop_reason = "keyboard_interrupt"
        error_type = "KeyboardInterrupt"
        error_message = "User interrupted (Ctrl+C)"
        traceback_excerpt = None
        raise
    except Exception as e:
        print(f"\n[REPLAY] Error during replay: {e}", file=sys.stderr)
        import traceback
        # Capture full exception message for early_stop_reason
        early_stop_reason = str(e) if str(e) else f"exception: {type(e).__name__}"
        error_type = type(e).__name__
        error_message = str(e)
        # Capture traceback excerpt (last 15 lines)
        tb_lines = traceback.format_exc().split('\n')
        traceback_excerpt = '\n'.join(tb_lines[-15:]) if len(tb_lines) > 15 else traceback.format_exc()
        raise
    finally:
        # Always write perf summary, even if replay was interrupted (Grep B)
        end_time = time.time()
        
        # Del 1: Get bars_total from runner (fallback to pre-computed value)
        bars_total_final = getattr(runner, 'perf_bars_total', 0) if 'runner' in locals() else bars_total
        bars_total_final = bars_total_final if bars_total_final > 0 else (bars_total if isinstance(bars_total, int) else 0)
        bars_processed_final = getattr(runner, 'perf_n_bars_processed', 0) if 'runner' in locals() else bars_processed
        completed_final = (bars_total_final > 0 and bars_processed_final >= bars_total_final)
        early_stop_reason_final = early_stop_reason if 'early_stop_reason' in locals() and not completed_final else None
        error_type_final = error_type if 'error_type' in locals() and not completed_final else None
        error_message_final = error_message if 'error_message' in locals() and not completed_final else None
        traceback_excerpt_final = traceback_excerpt if 'traceback_excerpt' in locals() and not completed_final else None
        
        # Fix trades_total: prefer entry_telemetry['n_trades_created'] over perf_n_trades_created
        trades_total_final = trades_total if 'trades_total' in locals() else 0
        if 'runner' in locals() and hasattr(runner, 'entry_manager') and hasattr(runner.entry_manager, 'entry_telemetry'):
            entry_telemetry_trades = runner.entry_manager.entry_telemetry.get("n_trades_created", 0)
            if entry_telemetry_trades > 0:
                trades_total_final = entry_telemetry_trades
        elif 'runner' in locals():
            trades_total_final = getattr(runner, 'perf_n_trades_created', trades_total_final)
        
        # Extract HTF warmup telemetry from runner
        n_bars_skipped_due_to_htf_warmup = 0
        htf_align_idx_min = -1
        if 'runner' in locals():
            n_bars_skipped_due_to_htf_warmup = getattr(runner, 'n_bars_skipped_due_to_htf_warmup', 0)
            # Extract htf_align_idx_min from perf collector
            if hasattr(runner, 'perf_collector'):
                try:
                    all_data = runner.perf_collector.get_all()
                    htf_align_key = "feat.htf_align_idx_min"
                    if htf_align_key in all_data:
                        htf_align_data = all_data[htf_align_key]
                        if isinstance(htf_align_data, dict) and "total_sec" in htf_align_data:
                            # total_sec stores the min value (we use it as a counter)
                            htf_align_idx_min = int(np.int64(htf_align_data["total_sec"]).item())
                        elif isinstance(htf_align_data, (int, float, np.integer, np.floating)):
                            htf_align_idx_min = int(np.int64(htf_align_data).item())
                except Exception as e:
                    print(f"[REPLAY_PERF] Warning: Failed to extract htf_align_idx_min: {e}", file=sys.stderr)
        
        # Extract entry counters from runner if available
        entry_counters_final = {}
        if 'runner' in locals() and hasattr(runner, 'entry_manager'):
            try:
                entry_mgr = runner.entry_manager
                # Extract counters from EntryManager
                # Add V10 diagnostic counters
                if hasattr(entry_mgr, 'n_v10_calls'):
                    entry_counters_final['n_v10_calls'] = entry_mgr.n_v10_calls
                if hasattr(entry_mgr, 'n_v10_pred_ok'):
                    entry_counters_final['n_v10_pred_ok'] = entry_mgr.n_v10_pred_ok
                if hasattr(entry_mgr, 'n_v10_pred_none_or_nan'):
                    entry_counters_final['n_v10_pred_none_or_nan'] = entry_mgr.n_v10_pred_none_or_nan
                if hasattr(entry_mgr, 'farm_diag'):
                    diag = entry_mgr.farm_diag
                    # Extract veto counters from EntryManager (if available)
                    veto_counters = getattr(entry_mgr, 'veto_counters', {})
                    threshold_used = getattr(entry_mgr, 'threshold_used', None)
                    
                    # Compute threshold veto (candidates that passed stage0 but failed threshold)
                    n_after_stage0 = diag.get("n_after_stage0", 0)
                    n_after_threshold = diag.get("n_after_policy_thresholds", 0)
                    threshold_veto = max(0, n_after_stage0 - n_after_threshold)
                    
                    # Get telemetry from entry_telemetry (SNIPER-first) or fallback to farm_diag (legacy)
                    entry_telemetry = getattr(entry_mgr, 'entry_telemetry', {})
                    if not entry_telemetry:
                        # Fallback to farm_diag for backward compatibility
                        entry_telemetry = {
                            "n_cycles": diag.get("n_bars", 0),
                            "n_candidates": diag.get("n_raw_candidates", 0),
                            "n_trades_created": diag.get("n_after_policy_thresholds", 0),
                            "p_long_values": diag.get("p_long_values", []),
                        }
                    
                    # Compute p_long histogram (200 bins from 0.0 to 1.0)
                    # This avoids storing large arrays for FULLYEAR replays
                    p_long_vals = entry_telemetry.get("p_long_values", [])
                    if not p_long_vals:
                        p_long_vals = diag.get("p_long_values", [])  # Legacy fallback
                    
                    p_long_histogram = None
                    p_long_stats = {}
                    if p_long_vals:
                        p_long_array = np.array(p_long_vals, dtype=np.float64)
                        # Filter finite values only
                        finite_mask = np.isfinite(p_long_array)
                        if np.any(finite_mask):
                            finite_vals = p_long_array[finite_mask]
                            # Clip to [0.0, 1.0] range
                            finite_vals = np.clip(finite_vals, 0.0, 1.0)
                            # Create histogram: 200 bins from 0.0 to 1.0
                            hist_counts, bin_edges = np.histogram(finite_vals, bins=200, range=(0.0, 1.0))
                            p_long_histogram = {
                                "bins": 200,
                                "min": 0.0,
                                "max": 1.0,
                                "counts": hist_counts.tolist(),  # Convert to list for JSON
                                "total_count": int(len(finite_vals)),
                            }
                            # Compute stats from histogram for summary
                            p_long_stats = {
                                "mean": float(np.mean(finite_vals)),
                                "p50": float(np.percentile(finite_vals, 50)),
                                "p90": float(np.percentile(finite_vals, 90)),
                                "min": float(np.min(finite_vals)),
                                "max": float(np.max(finite_vals)),
                            }
                    
                    # Note: Invariant check for n_trades_created == trades_total is done in assert_perf_invariants.py
                    
                    # Get veto_pre and veto_cand (SNIPER telemetry contract)
                    veto_pre = getattr(entry_mgr, 'veto_pre', {})
                    veto_cand = getattr(entry_mgr, 'veto_cand', {})
                    
                    # OPPGAVE 2: Get hard/soft eligibility veto counters
                    veto_hard = getattr(entry_mgr, 'veto_hard', {})
                    veto_soft = getattr(entry_mgr, 'veto_soft', {})
                    
                    # Determine ctx_expected (for CTX_INV_0)
                    # Note: os is already imported at module level
                    context_features_enabled = os.getenv("ENTRY_CONTEXT_FEATURES_ENABLED", "false").lower() == "true"
                    ctx_bundle_active = getattr(runner, "entry_v10_ctx_enabled", False) if 'runner' in locals() else False
                    bundle_meta = getattr(runner.entry_v10_bundle, "metadata", {}) or {} if 'runner' in locals() and hasattr(runner, 'entry_v10_bundle') and runner.entry_v10_bundle else {}
                    supports_context_features = bundle_meta.get("supports_context_features", False)
                    ctx_expected = context_features_enabled and ctx_bundle_active and supports_context_features
                    
                    # Get model class name
                    model_class_name = "UNKNOWN"
                    if 'runner' in locals() and hasattr(runner, 'entry_v10_bundle') and runner.entry_v10_bundle:
                        transformer_model = getattr(runner.entry_v10_bundle, "transformer_model", None)
                        if transformer_model is not None:
                            model_class_name = type(transformer_model).__name__
                    
                    # Get entry_models.v10_ctx.enabled from policy
                    entry_models_v10_ctx_enabled = False
                    if 'runner' in locals() and hasattr(runner, 'policy'):
                        entry_models_v10_ctx_enabled = runner.policy.get("entry_models", {}).get("v10_ctx", {}).get("enabled", False)
                    
                    # Get v10_none_reason_counts from telemetry
                    v10_none_reason_counts = entry_telemetry.get("v10_none_reason_counts", {})
                    
                    entry_counters_final = {
                        "n_cycles": entry_telemetry.get("n_cycles", 0),
                        "n_eligible_hard": entry_telemetry.get("n_eligible_hard", 0),  # OPPGAVE 2: Hard eligibility counter
                        "n_eligible_cycles": entry_telemetry.get("n_eligible_cycles", 0),  # OPPGAVE 2: Soft eligibility counter (after hard + soft)
                        # OPPGAVE 2: Context features telemetry
                        "n_context_built": entry_telemetry.get("n_context_built", 0),
                        "n_context_missing_or_invalid": entry_telemetry.get("n_context_missing_or_invalid", 0),
                        "n_ctx_model_calls": entry_telemetry.get("n_ctx_model_calls", 0),  # DEL 4: Ctx model calls
                        "n_v10_calls": entry_mgr.n_v10_calls if hasattr(entry_mgr, 'n_v10_calls') else 0,  # DEL D: Total V10 calls
                        "ctx_expected": ctx_expected,  # NEW: Flag for CTX_INV_0
                        "v10_none_reason_counts": v10_none_reason_counts,  # NEW: Reason-coded early returns
                        # NEW: Additional ctx diagnostics
                        "ENTRY_CONTEXT_FEATURES_ENABLED": context_features_enabled,
                        "entry_models_v10_ctx_enabled": entry_models_v10_ctx_enabled,
                        "bundle_supports_context_features": supports_context_features,
                        "model_class_name": model_class_name,
                        # DEL D: CTX consumption proof telemetry
                        "ctx_proof_enabled": os.getenv("GX1_CTX_CONSUMPTION_PROOF", "0") == "1",
                        "ctx_proof_pass_count": entry_telemetry.get("ctx_proof_pass_count", 0),
                        "ctx_proof_fail_count": entry_telemetry.get("ctx_proof_fail_count", 0),
                        # DEL D: v10_none_reason_counts_top5 (for easier debugging)
                        "v10_none_reason_counts_top5": dict(sorted(v10_none_reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]) if v10_none_reason_counts else {},
                        # DEL D: v10_none_reason_counts_full (for timeout diagnostics)
                        "v10_none_reason_counts_full": v10_none_reason_counts,
                        # DEL D: Exception stack excerpts (for timeout diagnostics)
                        "v10_exception_stacks": entry_telemetry.get("v10_exception_stacks", {}),
                        "n_precheck_pass": entry_telemetry.get("n_precheck_pass", 0),
                        "n_entry_candidates": entry_telemetry.get("n_candidates", 0),
                        "n_candidate_pass": entry_telemetry.get("n_candidate_pass", 0),
                        "n_entry_accepted": entry_telemetry.get("n_trades_created", 0),  # Alias: n_entry_accepted == n_trades_created
                        "n_trades_created": entry_telemetry.get("n_trades_created", 0),  # Explicit for clarity
                        # Entry journaling atomicity counters (OPPGAVE RUNTIME)
                        "n_entry_snapshots_written": entry_telemetry.get("n_entry_snapshots_written", 0),
                        "n_entry_snapshots_failed": entry_telemetry.get("n_entry_snapshots_failed", 0),
                        # OPPGAVE 2: Hard eligibility vetoes (before feature build)
                        "veto_hard_warmup": veto_hard.get("veto_hard_warmup", 0),
                        "veto_hard_session": veto_hard.get("veto_hard_session", 0),
                        "veto_hard_spread": veto_hard.get("veto_hard_spread", 0),
                        "veto_hard_killswitch": veto_hard.get("veto_hard_killswitch", 0),
                        # OPPGAVE 2: Soft eligibility vetoes (after minimal cheap computation)
                        "veto_soft_vol_regime_extreme": veto_soft.get("veto_soft_vol_regime_extreme", 0),
                        # Stage-0 vetoes (precheck level)
                        "veto_pre_warmup": veto_pre.get("veto_pre_warmup", 0),
                        "veto_pre_session": veto_pre.get("veto_pre_session", 0),
                        "veto_pre_regime": veto_pre.get("veto_pre_regime", 0),
                        "veto_pre_spread": veto_pre.get("veto_pre_spread", 0),
                        "veto_pre_atr": veto_pre.get("veto_pre_atr", 0),
                        "veto_pre_killswitch": veto_pre.get("veto_pre_killswitch", 0),
                        "veto_pre_model_missing": veto_pre.get("veto_pre_model_missing", 0),
                        "veto_pre_nan_features": veto_pre.get("veto_pre_nan_features", 0),
                        # Stage-1 vetoes (candidate level)
                        "veto_cand_threshold": veto_cand.get("veto_cand_threshold", 0),
                        "veto_cand_risk_guard": veto_cand.get("veto_cand_risk_guard", 0),
                        "veto_cand_max_trades": veto_cand.get("veto_cand_max_trades", 0),
                        "veto_cand_big_brain": veto_cand.get("veto_cand_big_brain", 0),
                        # Legacy vetoes (backward compatibility)
                        "veto_warmup": veto_counters.get("veto_warmup", 0),
                        "veto_session": entry_mgr.stage0_reasons.get("stage0_session_block", 0) + veto_counters.get("veto_session", 0),
                        "veto_threshold": threshold_veto + veto_counters.get("veto_threshold", 0),
                        "veto_spread": veto_counters.get("veto_spread", 0),
                        "veto_atr": entry_mgr.stage0_reasons.get("stage0_vol_block", 0) + veto_counters.get("veto_atr", 0),
                        "veto_regime": entry_mgr.stage0_reasons.get("stage0_trend_vol_block", 0) + veto_counters.get("veto_regime", 0),
                        "veto_risk_guard": veto_counters.get("veto_risk_guard", 0),
                        "veto_model_missing": veto_counters.get("veto_model_missing", 0),
                        "veto_nan_features": veto_counters.get("veto_nan_features", 0),
                        "veto_killswitch": veto_counters.get("veto_killswitch", 0),
                        "veto_max_trades": veto_counters.get("veto_max_trades", 0),
                        "veto_big_brain": veto_counters.get("veto_big_brain", 0),
                        # p_long stats (computed from histogram)
                        "p_long_mean": p_long_stats.get("mean"),
                        "p_long_p50": p_long_stats.get("p50"),
                        "p_long_p90": p_long_stats.get("p90"),
                        "p_long_min": p_long_stats.get("min"),
                        "p_long_max": p_long_stats.get("max"),
                        "p_long_histogram": p_long_histogram,  # Full histogram for merge
                        "threshold_used": threshold_used,
                        "candidate_session_counts": entry_telemetry.get("candidate_sessions", {}),
                        "trade_session_counts": entry_telemetry.get("trade_sessions", {}),
                        # Legacy (backward compatibility)
                        "session_bar_counts": diag.get("sessions", {}),
                    }
                    # Add V10 diagnostic counters
                    if hasattr(entry_mgr, 'n_v10_calls'):
                        entry_counters_final['n_v10_calls'] = entry_mgr.n_v10_calls
                    if hasattr(entry_mgr, 'n_v10_pred_ok'):
                        entry_counters_final['n_v10_pred_ok'] = entry_mgr.n_v10_pred_ok
                    if hasattr(entry_mgr, 'n_v10_pred_none_or_nan'):
                        entry_counters_final['n_v10_pred_none_or_nan'] = entry_mgr.n_v10_pred_none_or_nan
            except Exception as e:
                print(f"[REPLAY_PERF] Warning: Failed to extract entry counters: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
        
        # Log when perf instrumentation started (for feat_time_sec=0 diagnostics)
        perf_started = 'runner' in locals() and hasattr(runner, 'perf_feat_time')
        if not perf_started and feat_time_sec == 0.0:
            print(f"[REPLAY_PERF] WARNING: feat_time_sec=0.0 and perf instrumentation never started (exception before feature building?)", file=sys.stderr)
        
        # Del 2E: Get feature top blocks from runner if available
        feature_top_blocks_final = []
        n_pandas_ops_detected = 0
        if 'runner' in locals():
            if hasattr(runner, 'perf_collector'):
                try:
                    # Debug: check collector state
                    all_data = runner.perf_collector.get_all()
                    print(f"[REPLAY_PERF] Debug: perf_collector.get_all() returned {len(all_data)} entries", file=sys.stderr)
                    if len(all_data) > 0:
                        print(f"[REPLAY_PERF] Debug: First 5 entries: {list(all_data.items())[:5]}", file=sys.stderr)
                    
                    # Extract n_pandas_ops_detected from perf collector
                    # Note: perf_inc stores count in _counts dict, which is returned as "count" in get_all()
                    pandas_ops_key = "feat.basic_v1.n_pandas_ops_detected"
                    if pandas_ops_key in all_data:
                        # Get total count (sum of all increments)
                        pandas_ops_data = all_data[pandas_ops_key]
                        if isinstance(pandas_ops_data, dict) and "count" in pandas_ops_data:
                            n_pandas_ops_detected = int(pandas_ops_data["count"])
                        elif isinstance(pandas_ops_data, (int, float)):
                            n_pandas_ops_detected = int(pandas_ops_data)
                        print(f"[REPLAY_PERF] Extracted n_pandas_ops_detected={n_pandas_ops_detected} from perf collector", file=sys.stderr)
                    
                    top_15 = runner.perf_collector.top(15)  # Increased to 15 to catch more pandas rolling ops
                    print(f"[REPLAY_PERF] Debug: perf_collector.top(15) returned {len(top_15)} entries", file=sys.stderr)
                    
                    # Use runner_perf_metrics if available, otherwise try to get from runner directly
                    feat_time_sec = runner_perf_metrics.get('feat_time_sec', 0.0) if runner_perf_metrics else 0.0
                    if feat_time_sec == 0.0 and hasattr(runner, 'perf_feat_time'):
                        feat_time_sec = getattr(runner, 'perf_feat_time', 0.0)
                    print(f"[REPLAY_PERF] Debug: feat_time_sec = {feat_time_sec}", file=sys.stderr)
                    
                    feature_top_blocks_final = [
                        {
                            "name": name,
                            "total_sec": total_sec,
                            "count": count,
                            "share_of_feat_time_pct": (total_sec / feat_time_sec * 100.0) if feat_time_sec > 0 else 0.0
                        }
                        for name, total_sec, count in top_15
                    ]
                    print(f"[REPLAY_PERF] Debug: Created {len(feature_top_blocks_final)} feature_top_blocks entries", file=sys.stderr)
                except Exception as e:
                    print(f"[REPLAY_PERF] Warning: Failed to extract feature top blocks: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc()
            else:
                print("[REPLAY_PERF] Debug: runner exists but has no perf_collector attribute", file=sys.stderr)
        elif 'feature_top_blocks' in locals():
            feature_top_blocks_final = feature_top_blocks
        else:
            print("[REPLAY_PERF] Debug: 'runner' not in locals() in finally block", file=sys.stderr)
        
        # Del A3: Extract chunk_id and window info from output_dir or environment
        chunk_id = None
        
        # Try to extract chunk_id from output_dir path (e.g., .../chunk_0/...)
        output_dir_str = str(output_dir)
        import re
        # Match chunk_(\d+) anywhere in path (more robust than requiring trailing slash)
        chunk_match = re.search(r'chunk_(\d+)', output_dir_str)
        if chunk_match:
            chunk_idx = int(chunk_match.group(1))
            chunk_id = f"chunk_{chunk_idx:03d}"
        
        # Load window from WINDOW.json (source of truth, written before replay starts)
        window_start_final = None
        window_end_final = None
        window_json_path = output_dir / "WINDOW.json"
        if window_json_path.exists():
            try:
                j = json.loads(window_json_path.read_text(encoding="utf-8"))
                window_start_final = j.get("window_start")
                window_end_final = j.get("window_end")
                if window_start_final and window_end_final:
                    print(f"[REPLAY_PERF] Loaded window from WINDOW.json: {window_start_final} → {window_end_final}", file=sys.stderr)
            except Exception as e:
                print(f"[REPLAY_PERF] Warning: Failed to read WINDOW.json: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
        else:
            print(f"[REPLAY_PERF] FAIL window missing: output_dir={output_dir}, WINDOW.json does not exist", file=sys.stderr)
        
        # Hard fail if window is missing (invariant requires it)
        if window_start_final is None or window_end_final is None:
            print(f"[REPLAY_PERF] FAIL window missing: output_dir={output_dir}, window_start={window_start_final}, window_end={window_end_final}", file=sys.stderr)
            raise RuntimeError(f"WINDOW_MISSING: window_start/end not found for {output_dir}")
        
        # Del 3: Calculate next pandas rolling target from feature_top_blocks_final
        # OPPGAVE 1: Get fast_path_enabled from runner
        fast_path_enabled = getattr(runner, 'fast_path_enabled', False) if 'runner' in locals() else False
        
        # (This is also calculated inside write_perf_summary, but we pass None here
        #  to let the function handle it internally for consistency)
        write_perf_summary(
            output_dir=output_dir,
            start_time=start_time,
            end_time=end_time,
            bars_processed=bars_processed_final,
            bars_total=bars_total_final,
            fast_path_enabled=fast_path_enabled,
            last_timestamp=last_timestamp,
            trades_total=trades_total,
            trades_open_at_end=trades_open_at_end,
            log_file=log_file,
            runner_perf_metrics=runner_perf_metrics,
            completed=completed_final,
            early_stop_reason=early_stop_reason_final,
            feature_top_blocks=feature_top_blocks_final,
            chunk_id=chunk_id,
            window_start=window_start_final,
            window_end=window_end_final,
            error_type=error_type_final if 'error_type_final' in locals() else None,
            error_message=error_message_final if 'error_message_final' in locals() else None,
            traceback_excerpt=traceback_excerpt_final if 'traceback_excerpt_final' in locals() else None,
            entry_counters=entry_counters_final if 'entry_counters_final' in locals() else None,
            n_pandas_ops_detected=n_pandas_ops_detected if 'n_pandas_ops_detected' in locals() else 0,
            n_bars_skipped_due_to_htf_warmup=n_bars_skipped_due_to_htf_warmup if 'n_bars_skipped_due_to_htf_warmup' in locals() else 0,
            htf_align_idx_min=htf_align_idx_min if 'htf_align_idx_min' in locals() else -1,
        )


if __name__ == "__main__":
    main()
