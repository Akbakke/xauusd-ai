#!/usr/bin/env python3
"""
SNIPER Live p_long Distribution Analysis

Reads SNIPER runtime log and extracts p_long distribution from [SNIPER_CYCLE] logs.
Produces percentile analysis and near-miss rate.

Output:
- reports/sniper/diagnostics/SNIPER_LIVE_P_LONG_DISTRIBUTION_<timestamp>.md
- CSV with raw datapoints
"""

from __future__ import annotations

import re
import logging
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Thresholds
MIN_PROB_LONG = 0.67  # SNIPER threshold
NEAR_MISS_MIN = 0.60
NEAR_MISS_MAX = 0.67


def parse_sniper_cycle_line(line: str) -> Optional[Dict]:
    """Parse a [SNIPER_CYCLE] log line."""
    # Pattern: [SNIPER_CYCLE] ts=... session=... in_scope=... ... p_long=... reason=... eval=...
    # More flexible pattern that handles optional fields
    if '[SNIPER_CYCLE]' not in line:
        return None
    
    # Extract key fields using regex
    ts_match = re.search(r'ts=([^\s]+)', line)
    session_match = re.search(r'session=([^\s]+)', line)
    in_scope_match = re.search(r'in_scope=([^\s]+)', line)
    p_long_match = re.search(r'p_long=([0-9.]+)', line)
    reason_match = re.search(r'reason=([^\s]+)', line)
    eval_match = re.search(r'eval=([^\s]+)', line)
    
    # p_long is required for this analysis
    if not p_long_match:
        return None
    
    try:
        return {
            'ts': ts_match.group(1) if ts_match else None,
            'session': session_match.group(1) if session_match else None,
            'in_scope': int(in_scope_match.group(1)) if in_scope_match else None,
            'p_long': float(p_long_match.group(1)),
            'reason': reason_match.group(1) if reason_match else None,
            'eval': int(eval_match.group(1)) if eval_match else None,
        }
    except (ValueError, IndexError, AttributeError):
        return None


def load_sniper_log(log_path: Path) -> List[Dict]:
    """Load and parse SNIPER_CYCLE logs from runtime log."""
    rows = []
    
    if not log_path.exists():
        log.error(f"Log file not found: {log_path}")
        return rows
    
    with open(log_path, 'r') as f:
        for line in f:
            if '[SNIPER_CYCLE]' in line:
                parsed = parse_sniper_cycle_line(line)
                if parsed:
                    rows.append(parsed)
    
    return rows


def analyze_p_long_distribution(df: pd.DataFrame) -> Dict:
    """Analyze p_long distribution."""
    if len(df) == 0:
        return {}
    
    p_long_vals = df['p_long'].dropna()
    if len(p_long_vals) == 0:
        return {}
    
    return {
        'count': len(p_long_vals),
        'mean': float(p_long_vals.mean()),
        'median': float(p_long_vals.median()),
        'std': float(p_long_vals.std()),
        'min': float(p_long_vals.min()),
        'max': float(p_long_vals.max()),
        'percentiles': {
            'p10': float(p_long_vals.quantile(0.10)),
            'p25': float(p_long_vals.quantile(0.25)),
            'p50': float(p_long_vals.quantile(0.50)),
            'p75': float(p_long_vals.quantile(0.75)),
            'p90': float(p_long_vals.quantile(0.90)),
            'p95': float(p_long_vals.quantile(0.95)),
            'p99': float(p_long_vals.quantile(0.99)),
        },
        'above_threshold': int((p_long_vals >= MIN_PROB_LONG).sum()),
        'near_miss': int(((p_long_vals >= NEAR_MISS_MIN) & (p_long_vals < NEAR_MISS_MAX)).sum()),
    }


def analyze_by_session(df: pd.DataFrame) -> Dict:
    """Analyze p_long distribution by session."""
    results = {}
    
    for session in ['EU', 'OVERLAP', 'US', 'ASIA']:
        session_df = df[df['session'] == session]
        if len(session_df) == 0:
            continue
        
        p_long_vals = session_df['p_long'].dropna()
        if len(p_long_vals) == 0:
            continue
        
        results[session] = {
            'count': len(p_long_vals),
            'mean': float(p_long_vals.mean()),
            'median': float(p_long_vals.median()),
            'above_threshold': int((p_long_vals >= MIN_PROB_LONG).sum()),
            'near_miss': int(((p_long_vals >= NEAR_MISS_MIN) & (p_long_vals < NEAR_MISS_MAX)).sum()),
        }
    
    return results


def analyze_by_reason(df: pd.DataFrame) -> Dict:
    """Analyze p_long distribution by reason."""
    results = {}
    
    for reason in df['reason'].unique():
        reason_df = df[df['reason'] == reason]
        if len(reason_df) == 0:
            continue
        
        p_long_vals = reason_df['p_long'].dropna()
        if len(p_long_vals) == 0:
            continue
        
        results[reason] = {
            'count': len(p_long_vals),
            'mean': float(p_long_vals.mean()),
            'median': float(p_long_vals.median()),
            'above_threshold': int((p_long_vals >= MIN_PROB_LONG).sum()),
            'near_miss': int(((p_long_vals >= NEAR_MISS_MIN) & (p_long_vals < NEAR_MISS_MAX)).sum()),
        }
    
    return results


def generate_report(
    df: pd.DataFrame,
    overall_stats: Dict,
    session_stats: Dict,
    reason_stats: Dict,
    log_path: Path,
    output_path: Path
) -> None:
    """Generate markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    with open(output_path, 'w') as f:
        f.write("# SNIPER Live p_long Distribution Analysis\n\n")
        f.write(f"**Generated:** {timestamp}\n")
        f.write(f"**Source Log:** `{log_path}`\n\n")
        f.write("---\n\n")
        
        f.write("## Summary\n\n")
        if overall_stats:
            f.write(f"- **Total Cycles with p_long:** {overall_stats['count']}\n")
            f.write(f"- **Mean p_long:** {overall_stats['mean']:.4f}\n")
            f.write(f"- **Median p_long:** {overall_stats['median']:.4f}\n")
            f.write(f"- **Above Threshold (≥0.67):** {overall_stats['above_threshold']} ({overall_stats['above_threshold']/overall_stats['count']*100:.1f}%)\n")
            f.write(f"- **Near-Miss [0.60, 0.67):** {overall_stats['near_miss']} ({overall_stats['near_miss']/overall_stats['count']*100:.1f}%)\n\n")
        else:
            f.write("No p_long data available.\n\n")
        
        f.write("## p_long Distribution\n\n")
        if overall_stats:
            f.write("### Overall Statistics\n\n")
            f.write(f"- **Count:** {overall_stats['count']}\n")
            f.write(f"- **Mean:** {overall_stats['mean']:.4f}\n")
            f.write(f"- **Median:** {overall_stats['median']:.4f}\n")
            f.write(f"- **Std:** {overall_stats['std']:.4f}\n")
            f.write(f"- **Range:** [{overall_stats['min']:.4f}, {overall_stats['max']:.4f}]\n\n")
            f.write("Percentiles:\n")
            for p_name, p_val in overall_stats['percentiles'].items():
                f.write(f"- {p_name}: {p_val:.4f}\n")
            f.write("\n")
        
        f.write("### By Session\n\n")
        if session_stats:
            f.write("| Session | Count | Mean | Median | Above Threshold | Near-Miss |\n")
            f.write("|---------|-------|------|--------|-----------------|-----------|\n")
            for session, stats in session_stats.items():
                f.write(f"| {session} | {stats['count']} | {stats['mean']:.4f} | {stats['median']:.4f} | "
                       f"{stats['above_threshold']} ({stats['above_threshold']/stats['count']*100:.1f}%) | "
                       f"{stats['near_miss']} ({stats['near_miss']/stats['count']*100:.1f}%) |\n")
        else:
            f.write("No session data available.\n")
        
        f.write("\n### By Reason\n\n")
        if reason_stats:
            f.write("| Reason | Count | Mean | Median | Above Threshold | Near-Miss |\n")
            f.write("|--------|-------|------|--------|-----------------|-----------|\n")
            for reason, stats in reason_stats.items():
                f.write(f"| {reason} | {stats['count']} | {stats['mean']:.4f} | {stats['median']:.4f} | "
                       f"{stats['above_threshold']} ({stats['above_threshold']/stats['count']*100:.1f}%) | "
                       f"{stats['near_miss']} ({stats['near_miss']/stats['count']*100:.1f}%) |\n")
        else:
            f.write("No reason data available.\n")


def main():
    """Main analysis function."""
    import sys
    
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        # Find newest SNIPER log
        base_path = Path(".")
        sniper_logs = list((base_path / "runs" / "live_demo").glob("SNIPER_*/sniper_runtime.log"))
        if not sniper_logs:
            log.error("No SNIPER logs found")
            return
        log_path = max(sniper_logs, key=lambda p: p.stat().st_mtime)
    
    log.info(f"Analyzing log: {log_path}")
    
    # Load and parse logs
    rows = load_sniper_log(log_path)
    if not rows:
        log.warning("No [SNIPER_CYCLE] logs found")
        return
    
    log.info(f"Parsed {len(rows)} SNIPER_CYCLE log entries")
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Filter for cycles with p_long (eval=1)
    df_with_p_long = df[df['eval'] == 1].copy()
    log.info(f"Cycles with eval=1: {len(df_with_p_long)}")
    
    if len(df_with_p_long) == 0:
        log.warning("No cycles with eval=1 found")
        return
    
    # Analyze
    overall_stats = analyze_p_long_distribution(df_with_p_long)
    session_stats = analyze_by_session(df_with_p_long)
    reason_stats = analyze_by_reason(df_with_p_long)
    
    # Generate report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path("reports/sniper/diagnostics") / f"SNIPER_LIVE_P_LONG_DISTRIBUTION_{timestamp}.md"
    generate_report(df_with_p_long, overall_stats, session_stats, reason_stats, log_path, output_path)
    log.info(f"Report generated: {output_path}")
    
    # Save CSV
    csv_path = Path("reports/sniper/diagnostics") / f"SNIPER_LIVE_P_LONG_DISTRIBUTION_{timestamp}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_with_p_long.to_csv(csv_path, index=False)
    log.info(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()

