#!/usr/bin/env python3
"""
Merge performance summaries from parallel replay chunks.

Fail-fast: raises SystemExit if any chunk summary is missing.
Deterministic merge: sums/aggregates metrics from all chunks.
"""
import json
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Add project root to Python path
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def load_chunk_summaries(outdir: Path) -> List[Dict[str, Any]]:
    """
    Load all chunk perf summaries.
    
    Parameters
    ----------
    outdir : Path
        Root output directory containing parallel_chunks/
    
    Returns
    -------
    List[Dict[str, Any]]
        List of chunk summary dictionaries
    
    Raises
    ------
    SystemExit
        If any chunk directory is missing a summary
    """
    # Match only numeric chunk directories (chunk_0, chunk_1, etc.), not chunk_0.log files
    all_items = list(outdir.glob("parallel_chunks/chunk_*"))
    chunk_dirs = sorted([d for d in all_items if d.is_dir() and d.name.startswith("chunk_") and (len(d.name) == 7 and d.name[6:].isdigit())])
    if not chunk_dirs:
        raise SystemExit(f"No chunk directories found in {outdir}/parallel_chunks/")
    
    summaries = []
    missing = []
    
    incomplete = []
    for chunk_dir in chunk_dirs:
        summary_file = chunk_dir / "REPLAY_PERF_SUMMARY.json"
        if not summary_file.exists():
            missing.append(chunk_dir)
        else:
            with open(summary_file) as f:
                chunk_data = json.load(f)
                summaries.append(chunk_data)
                # Fail-fast on incomplete chunks (status != "complete")
                chunk_status = chunk_data.get("status", chunk_data.get("completed", True))
                if chunk_status not in ("complete", True):
                    incomplete.append((chunk_dir, chunk_data.get("chunk_id", "unknown"), chunk_data.get("early_stop_reason", "unknown")))
    
    if missing:
        missing_str = ", ".join(str(m.name) for m in missing[:5])
        if len(missing) > 5:
            missing_str += f" ... ({len(missing)} total)"
        raise SystemExit(f"Missing perf summary in {len(missing)} chunks: {missing_str}")
    
    if incomplete:
        incomplete_str = ", ".join(f"{chunk_id} ({chunk_dir.name})" for chunk_dir, chunk_id, _ in incomplete[:5])
        if len(incomplete) > 5:
            incomplete_str += f" ... ({len(incomplete)} total)"
        reasons_str = "; ".join(f"{chunk_id}: {reason}" for _, chunk_id, reason in incomplete[:3])
        raise SystemExit(f"Incomplete chunks (status != 'complete'): {incomplete_str}. Reasons: {reasons_str}")
    
    return summaries


def merge_summaries(summaries: List[Dict[str, Any]], wallclock_duration: float = None) -> Dict[str, Any]:
    """
    Merge chunk summaries into global summary.
    
    Parameters
    ----------
    summaries : List[Dict[str, Any]]
        List of chunk summary dictionaries
    wallclock_duration : float, optional
        Total wall-clock duration (if available from parent process)
    
    Returns
    -------
    Dict[str, Any]
        Merged summary dictionary
    """
    if not summaries:
        raise ValueError("No summaries to merge")
    
    # Basic aggregation
    bars_total = sum(s.get("bars_total", 0) for s in summaries)
    bars_processed = sum(s.get("bars_processed", 0) for s in summaries)
    trades_total = sum(s.get("trades_total", 0) for s in summaries)
    trades_open_at_end = sum(s.get("trades_open_at_end", 0) for s in summaries)
    
    # Duration: use wallclock if provided, otherwise max (parallel execution)
    if wallclock_duration is not None:
        duration_sec = wallclock_duration
    else:
        duration_sec_sum = sum(s.get("duration_sec", 0.0) for s in summaries)
        duration_sec_max = max((s.get("duration_sec", 0.0) for s in summaries), default=0.0)
        duration_sec = duration_sec_max  # Use max for parallel execution
    
    # Completion status
    all_completed = all(s.get("completed", False) for s in summaries)
    early_stop_reasons = []
    for s in summaries:
        if not s.get("completed", False):
            chunk_id = s.get("chunk_id", "unknown")
            reason = s.get("early_stop_reason", "unknown")
            early_stop_reasons.append(f"{chunk_id}: {reason}")
    
    # Runner perf metrics (sum)
    runner_perf = defaultdict(float)
    for s in summaries:
        chunk_perf = s.get("runner_perf_metrics", {})
        for key, value in chunk_perf.items():
            if isinstance(value, (int, float)):
                runner_perf[key] += value
    
    # Feature top blocks (merge by name, sum total_sec and count)
    feature_blocks = defaultdict(lambda: {"total_sec": 0.0, "count": 0})
    for s in summaries:
        for block in s.get("feature_top_blocks", []):
            name = block.get("name")
            if name:
                feature_blocks[name]["total_sec"] += block.get("total_sec", 0.0)
                feature_blocks[name]["count"] += block.get("count", 0)
    
    # Sort by total_sec descending, take top 15
    feat_time_sec = runner_perf.get("feat_time_sec", 0.0)
    feature_top_blocks = [
        {
            "name": name,
            "total_sec": data["total_sec"],
            "count": data["count"],
            "share_of_feat_time_pct": (data["total_sec"] / feat_time_sec * 100.0) if feat_time_sec > 0 else 0.0,
        }
        for name, data in sorted(feature_blocks.items(), key=lambda x: x[1]["total_sec"], reverse=True)[:15]
    ]
    
    # Top pandas ops (merge by name, sum total_sec and count)
    pandas_ops = defaultdict(lambda: {"total_sec": 0.0, "count": 0})
    for s in summaries:
        for op in s.get("top_pandas_ops", []):
            name = op.get("name")
            if name:
                pandas_ops[name]["total_sec"] += op.get("total_sec", 0.0)
                pandas_ops[name]["count"] += op.get("count", 0)
    
    top_pandas_ops = [
        {
            "name": name,
            "total_sec": data["total_sec"],
            "count": data["count"],
        }
        for name, data in sorted(pandas_ops.items(), key=lambda x: x[1]["total_sec"], reverse=True)[:10]
    ]
    
    # Window info (from first and last chunks)
    window_start = summaries[0].get("window_start") if summaries else None
    window_end = summaries[-1].get("window_end") if summaries else None
    
    # Timestamps
    start_time = summaries[0].get("start_time") if summaries else None
    end_time = summaries[-1].get("end_time") if summaries else None
    last_timestamp = summaries[-1].get("last_timestamp_processed") if summaries else None
    
    # Chunk completion table
    chunk_completion = [
        {
            "chunk_id": s.get("chunk_id", "unknown"),
            "completed": s.get("completed", False),
            "bars_processed": s.get("bars_processed", 0),
            "bars_total": s.get("bars_total", 0),
            "duration_sec": s.get("duration_sec", 0.0),
            "early_stop_reason": s.get("early_stop_reason"),
        }
        for s in summaries
    ]
    
    # Merge entry counters (SNIPER telemetry contract)
    entry_counters_merged = defaultdict(int)
    threshold_used_values = []  # Collect all threshold values
    candidate_session_counts_merged = defaultdict(int)
    trade_session_counts_merged = defaultdict(int)
    session_bar_counts_merged = defaultdict(int)  # Legacy
    
    # Histogram merge: aggregate p_long histograms from chunks
    merged_histogram_counts = None
    histogram_total_count = 0
    histogram_bins = 200
    histogram_min = 0.0
    histogram_max = 1.0
    
    for s in summaries:
        counters = s.get("entry_counters", {})
        if counters:
            # Sum core counters (Stage-0 and Stage-1)
            for key in ["n_cycles", "n_precheck_pass", "n_predictions", "n_entry_candidates", 
                       "n_candidate_pass", "n_entry_accepted", "n_trades_created",
                       # Stage-0 vetoes
                       "veto_pre_warmup", "veto_pre_session", "veto_pre_regime", 
                       "veto_pre_spread", "veto_pre_atr", "veto_pre_killswitch",
                       "veto_pre_model_missing", "veto_pre_nan_features",
                       # Stage-1 vetoes
                       "veto_cand_threshold", "veto_cand_risk_guard", 
                       "veto_cand_max_trades", "veto_cand_big_brain",
                       # Legacy vetoes (backward compatibility)
                       "veto_warmup", "veto_session", "veto_threshold", "veto_spread",
                       "veto_atr", "veto_regime", "veto_risk_guard", "veto_model_missing",
                       "veto_nan_features", "veto_killswitch", "veto_max_trades", "veto_big_brain"]:
                entry_counters_merged[key] += counters.get(key, 0)
            
            # Merge p_long histogram
            hist = counters.get("p_long_histogram")
            if hist and isinstance(hist, dict):
                counts = hist.get("counts", [])
                total = hist.get("total_count", 0)
                bins = hist.get("bins", 200)
                hmin = hist.get("min", 0.0)
                hmax = hist.get("max", 1.0)
                
                # Initialize merged histogram on first chunk
                if merged_histogram_counts is None:
                    merged_histogram_counts = [0] * bins
                    histogram_bins = bins
                    histogram_min = hmin
                    histogram_max = hmax
                
                # Sum histogram counts (must have same bin count)
                if len(counts) == histogram_bins:
                    for i in range(histogram_bins):
                        merged_histogram_counts[i] += counts[i]
                    histogram_total_count += total
            
            # Track threshold_used (collect all unique values, cap at 20)
            thresh = counters.get("threshold_used")
            if thresh is not None and thresh not in threshold_used_values and len(threshold_used_values) < 20:
                threshold_used_values.append(thresh)
            
            # Merge session counts (candidate and trade sessions)
            candidate_sessions = counters.get("candidate_session_counts", {})
            for session, count in candidate_sessions.items():
                candidate_session_counts_merged[session] += count
            
            trade_sessions = counters.get("trade_session_counts", {})
            for session, count in trade_sessions.items():
                trade_session_counts_merged[session] += count
            
            # Legacy session_bar_counts
            session_counts = counters.get("session_bar_counts", {})
            for session, count in session_counts.items():
                session_bar_counts_merged[session] += count
    
    # Compute p_long stats from merged histogram
    entry_counters_final = dict(entry_counters_merged)
    if merged_histogram_counts and histogram_total_count > 0:
        # Convert histogram to approximate quantiles
        # Build cumulative distribution
        cumsum = 0
        cumsums = []
        for count in merged_histogram_counts:
            cumsum += count
            cumsums.append(cumsum)
        
        # Compute quantiles from cumulative distribution
        bin_width = (histogram_max - histogram_min) / histogram_bins
        q50_idx = None
        q90_idx = None
        
        for i, cs in enumerate(cumsums):
            pct = cs / histogram_total_count
            if q50_idx is None and pct >= 0.50:
                q50_idx = i
            if q90_idx is None and pct >= 0.90:
                q90_idx = i
                break
        
        # Compute quantile values (bin center)
        if q50_idx is not None:
            q50_bin_center = histogram_min + (q50_idx + 0.5) * bin_width
            entry_counters_final["p_long_p50"] = float(q50_bin_center)
        if q90_idx is not None:
            q90_bin_center = histogram_min + (q90_idx + 0.5) * bin_width
            entry_counters_final["p_long_p90"] = float(q90_bin_center)
        
        # Compute weighted mean from histogram
        bin_centers = [histogram_min + (i + 0.5) * bin_width for i in range(histogram_bins)]
        weighted_sum = sum(cent * count for cent, count in zip(bin_centers, merged_histogram_counts))
        entry_counters_final["p_long_mean"] = weighted_sum / histogram_total_count
        
        # Store merged histogram for potential future use
        entry_counters_final["p_long_histogram"] = {
            "bins": histogram_bins,
            "min": histogram_min,
            "max": histogram_max,
            "counts": merged_histogram_counts,
            "total_count": histogram_total_count,
        }
    
    # Threshold summary
    if threshold_used_values:
        if len(threshold_used_values) == 1:
            entry_counters_final["threshold_used"] = threshold_used_values[0]
        else:
            entry_counters_final["threshold_used"] = threshold_used_values[0]  # Use first as primary
            entry_counters_final["threshold_used_unique"] = threshold_used_values[:20]  # Cap at 20
    
    entry_counters_final["candidate_session_counts"] = dict(candidate_session_counts_merged)
    entry_counters_final["trade_session_counts"] = dict(trade_session_counts_merged)
    entry_counters_final["session_bar_counts"] = dict(session_bar_counts_merged)  # Legacy
    
    merged = {
        "start_time": start_time,
        "end_time": end_time,
        "duration_sec": duration_sec,
        "window_start": window_start,
        "window_end": window_end,
        "bars_total": bars_total,
        "bars_processed": bars_processed,
        "completed": all_completed,
        "early_stop_reasons": early_stop_reasons if early_stop_reasons else None,
        "last_timestamp_processed": last_timestamp,
        "trades_total": trades_total,
        "trades_open_at_end": trades_open_at_end,
        "runner_perf_metrics": dict(runner_perf),
        "feature_top_blocks": feature_top_blocks,
        "top_pandas_ops": top_pandas_ops,
        "chunk_completion": chunk_completion,
        "entry_counters": entry_counters_final,
    }
    
    return merged


def format_merged_md(data: Dict[str, Any]) -> str:
    """
    Format merged summary as Markdown with chunk completion table.
    """
    from gx1.utils.perf_summary_io import format_md_summary
    
    # Start with base format
    md = format_md_summary(data)
    
    # Add chunk completion table if present
    chunk_completion = data.get("chunk_completion", [])
    if chunk_completion:
        md += "\n## Chunk Completion Status\n\n"
        md += "| Chunk ID | Completed | Bars Processed | Bars Total | Duration (sec) | Early Stop |\n"
        md += "|----------|-----------|----------------|------------|----------------|------------|\n"
        for chunk in chunk_completion:
            chunk_id = chunk.get("chunk_id", "unknown")
            completed = "✅" if chunk.get("completed") else "❌"
            bars_proc = chunk.get("bars_processed", 0)
            bars_tot = chunk.get("bars_total", 0)
            duration = chunk.get("duration_sec", 0.0)
            early_stop = chunk.get("early_stop_reason", "-")
            md += f"| {chunk_id} | {completed} | {bars_proc:,} | {bars_tot:,} | {duration:.1f} | {early_stop} |\n"
        md += "\n"
    
    # Add early stop reasons if any
    if data.get("early_stop_reasons"):
        md += "## Early Stop Reasons\n\n"
        for reason in data["early_stop_reasons"]:
            md += f"- {reason}\n"
        md += "\n"
    
    return md


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: merge_perf_summaries.py <output_dir> [wallclock_duration_sec]")
        return 1
    
    outdir = Path(sys.argv[1])
    wallclock_duration = float(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not outdir.exists():
        print(f"ERROR: Output directory does not exist: {outdir}")
        return 1
    
    try:
        # Load chunk summaries (fail-fast if any missing)
        summaries = load_chunk_summaries(outdir)
        print(f"✅ Loaded {len(summaries)} chunk summaries")
        
        # Merge summaries
        merged = merge_summaries(summaries, wallclock_duration)
        
        # Write merged JSON
        json_path = outdir / "REPLAY_PERF_SUMMARY.json"
        with open(json_path, "w") as f:
            json.dump(merged, f, indent=2, sort_keys=True, default=str)
        print(f"✅ Wrote merged summary: {json_path}")
        
        # Write merged Markdown
        md_path = outdir / "REPLAY_PERF_SUMMARY.md"
        md_content = format_merged_md(merged)
        with open(md_path, "w") as f:
            f.write(md_content)
        print(f"✅ Wrote merged markdown: {md_path}")
        
        return 0
        
    except SystemExit as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: Failed to merge summaries: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

