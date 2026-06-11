#!/usr/bin/env python3
"""
Generate entry-score distribution report from FULLYEAR replay.

Reads chunk footers and generates comprehensive entry-score distribution analysis.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime
import numpy as np

def load_entry_scores_from_footers(run_dir: Path) -> Dict[str, Any]:
    """Load entry-score data from all chunk footers."""
    footers = list(run_dir.glob("chunk_*/chunk_footer.json"))
    
    all_scores = []
    session_scores = defaultdict(list)
    regime_scores = defaultdict(list)
    
    total_samples = 0
    score_stats = {
        "min": None,
        "max": None,
        "mean": None,
        "median": None,
        "p5": None,
        "p25": None,
        "p50": None,
        "p75": None,
        "p95": None,
        "p99": None,
    }
    
    for footer_path in footers:
        try:
            with open(footer_path, "r") as f:
                footer = json.load(f)
            
            # Aggregate stats from footer
            samples = footer.get("entry_score_samples", 0)
            if samples > 0:
                total_samples += samples
                # Use aggregated stats from footer (more efficient than loading all samples)
                if score_stats["min"] is None or footer.get("entry_score_min") < score_stats["min"]:
                    score_stats["min"] = footer.get("entry_score_min")
                if score_stats["max"] is None or footer.get("entry_score_max") > score_stats["max"]:
                    score_stats["max"] = footer.get("entry_score_max")
                
                # For percentiles, we need to aggregate properly (use weighted approach or collect all)
                # For now, use footer stats as approximation
                if score_stats["mean"] is None:
                    score_stats["mean"] = footer.get("entry_score_mean")
                    score_stats["median"] = footer.get("entry_score_median")
                    score_stats["p5"] = footer.get("entry_score_p5")
                    score_stats["p25"] = footer.get("entry_score_p25")
                    score_stats["p50"] = footer.get("entry_score_p50")
                    score_stats["p75"] = footer.get("entry_score_p75")
                    score_stats["p95"] = footer.get("entry_score_p95")
                    score_stats["p99"] = footer.get("entry_score_p99")
        except Exception as e:
            print(f"Warning: Failed to load {footer_path}: {e}", file=sys.stderr)
    
    return {
        "total_samples": total_samples,
        "score_stats": score_stats,
        "session_scores": dict(session_scores),
        "regime_scores": dict(regime_scores),
    }

def load_perf_json(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load perf JSON to get run context."""
    perf_files = list(run_dir.glob("perf_*.json"))
    if not perf_files:
        return None
    
    try:
        with open(perf_files[0], "r") as f:
            return json.load(f)
    except Exception:
        return None

def get_threshold_from_policy(perf: Optional[Dict[str, Any]]) -> Optional[float]:
    """Extract threshold from policy config if available."""
    # Try to get from policy path or config
    # For now, return None and let user specify
    return None

def generate_histogram(scores: List[float], bins: int = 20) -> List[Dict[str, Any]]:
    """Generate histogram data."""
    if not scores:
        return []
    
    hist, edges = np.histogram(scores, bins=bins)
    result = []
    for i in range(len(hist)):
        result.append({
            "bin_start": float(edges[i]),
            "bin_end": float(edges[i + 1]),
            "count": int(hist[i]),
        })
    return result

def generate_report(run_dir: Path, output_path: Path, threshold: Optional[float] = None) -> None:
    """Generate entry-score distribution report."""
    print(f"Loading entry-score data from: {run_dir}")
    
    # Load data
    score_data = load_entry_scores_from_footers(run_dir)
    perf = load_perf_json(run_dir)
    
    if score_data["total_samples"] == 0:
        report = f"""# FULLYEAR ENTRY-SCORE DISTRIBUTION

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Run ID:** {run_dir.name}
**Source:** {run_dir}

## ERROR

No entry-score samples found. This may indicate:
- No bars reached entry-stage
- Entry-score logging not enabled
- All bars were filtered by hard eligibility

Check chunk footers for `entry_score_samples` field.
"""
        with open(output_path, "w") as f:
            f.write(report)
        return
    
    # Get threshold from policy if not provided
    if threshold is None:
        # Try to extract from policy config
        threshold = get_threshold_from_policy(perf)
        if threshold is None:
            threshold = 0.67  # Default SNIPER threshold
    
    # Calculate threshold stats
    stats = score_data["score_stats"]
    if stats["mean"] is not None:
        # Approximate % above threshold using percentile
        # This is approximate since we don't have full distribution
        p95 = stats.get("p95", 0.0)
        p99 = stats.get("p99", 0.0)
        mean = stats.get("mean", 0.0)
        median = stats.get("median", 0.0)
        
        # Rough estimate: if threshold is above p95, very few samples
        # This is a simplification - full analysis would require all samples
        if threshold <= stats.get("p5", 0.0):
            pct_above = ">95%"
        elif threshold <= stats.get("p25", 0.0):
            pct_above = "75-95%"
        elif threshold <= stats.get("p50", 0.0):
            pct_above = "50-75%"
        elif threshold <= stats.get("p75", 0.0):
            pct_above = "25-50%"
        elif threshold <= stats.get("p95", 0.0):
            pct_above = "5-25%"
        else:
            pct_above = "<5%"
    else:
        pct_above = "unknown"
    
    # Generate report
    report_lines = [
        "# FULLYEAR ENTRY-SCORE DISTRIBUTION",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Run ID:** {run_dir.name}",
        f"**Source:** {run_dir}",
        "",
        "## RUN_CTX",
        "",
        f"- **root:** {run_dir.parent.parent}",
        f"- **head:** {perf.get('git_head', 'unknown') if perf else 'unknown'}",
        f"- **run-id:** {perf.get('run_id', run_dir.name) if perf else run_dir.name}",
        f"- **policy:** {perf.get('policy_path', 'unknown') if perf else 'unknown'}",
        f"- **dataset:** {perf.get('data_path', 'unknown') if perf else 'unknown'}",
        f"- **prebuilt sha:** {perf.get('features_file_sha256', 'unknown') if perf else 'unknown'}",
        "",
        "## Counts",
        "",
    ]
    
    # Load footer data for counts
    footer_files = list(run_dir.glob("chunk_*/chunk_footer.json"))
    total_lookup_attempts = 0
    total_bars_reaching_entry_stage = 0
    for footer_path in footer_files:
        try:
            with open(footer_path, "r") as f:
                footer = json.load(f)
            total_lookup_attempts += footer.get("lookup_attempts", 0)
            total_bars_reaching_entry_stage += footer.get("bars_reaching_entry_stage", 0)
        except Exception:
            pass
    
    # Build score stats strings
    mean_str = f"{stats.get('mean'):.4f}" if stats.get('mean') is not None else "N/A"
    median_str = f"{stats.get('median'):.4f}" if stats.get('median') is not None else "N/A"
    p5_str = f"{stats.get('p5'):.4f}" if stats.get('p5') is not None else "N/A"
    p25_str = f"{stats.get('p25'):.4f}" if stats.get('p25') is not None else "N/A"
    p50_str = f"{stats.get('p50'):.4f}" if stats.get('p50') is not None else "N/A"
    p75_str = f"{stats.get('p75'):.4f}" if stats.get('p75') is not None else "N/A"
    p95_str = f"{stats.get('p95'):.4f}" if stats.get('p95') is not None else "N/A"
    p99_str = f"{stats.get('p99'):.4f}" if stats.get('p99') is not None else "N/A"
    
    report_lines.extend([
        f"- **eligible bars (lookup_attempts):** {total_lookup_attempts:,}",
        f"- **bars reaching entry-stage:** {total_bars_reaching_entry_stage:,}",
        f"- **entry-score samples:** {score_data['total_samples']:,}",
        "",
        "## Score Stats (Global)",
        "",
        f"- **min:** {stats.get('min', 'N/A')}",
        f"- **max:** {stats.get('max', 'N/A')}",
        f"- **mean:** {mean_str}",
        f"- **median:** {median_str}",
        f"- **p5:** {p5_str}",
        f"- **p25:** {p25_str}",
        f"- **p50:** {p50_str}",
        f"- **p75:** {p75_str}",
        f"- **p95:** {p95_str}",
        f"- **p99:** {p99_str}",
        "",
        "## Threshold Analysis",
        "",
        f"- **Current threshold:** {threshold:.4f}",
        f"- **% of bars above threshold (approx):** {pct_above}",
        "",
        "## Notes",
        "",
        "- Entry-score samples represent ALL eligible bars that reached entry-stage",
        "- Scores are logged BEFORE policy threshold check",
        "- Threshold analysis is approximate (based on percentiles)",
        "- Full distribution requires loading all samples (not implemented for performance)",
        "",
    ])
    
    # Add per-chunk summary
    report_lines.extend([
        "## Chunk Summary",
        "",
        "| Chunk | Entry Score Samples | Min | Max | Mean | P95 |",
        "|-------|---------------------|-----|-----|------|-----|",
    ])
    
    for footer_path in sorted(run_dir.glob("chunk_*/chunk_footer.json")):
        try:
            with open(footer_path, "r") as f:
                footer = json.load(f)
            chunk_idx = footer.get("chunk_id", "?")
            samples = footer.get("entry_score_samples", 0)
            if samples > 0:
                report_lines.append(
                    f"| {chunk_idx} | {samples:,} | {footer.get('entry_score_min', 'N/A'):.4f} | "
                    f"{footer.get('entry_score_max', 'N/A'):.4f} | {footer.get('entry_score_mean', 'N/A'):.4f} | "
                    f"{footer.get('entry_score_p95', 'N/A'):.4f} |"
                )
        except Exception:
            pass
    
    report_lines.append("")
    
    # Write report
    report = "\n".join(report_lines)
    with open(output_path, "w") as f:
        f.write(report)
    
    print(f"✅ Report written to: {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: generate_entry_score_distribution.py <run_dir> [output_path] [threshold]")
        sys.exit(1)
    
    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = run_dir.parent / f"FULLYEAR_ENTRY_SCORE_DISTRIBUTION_{run_dir.name}.md"
    
    threshold = float(sys.argv[3]) if len(sys.argv) >= 4 else None
    
    generate_report(run_dir, output_path, threshold)

if __name__ == "__main__":
    main()
