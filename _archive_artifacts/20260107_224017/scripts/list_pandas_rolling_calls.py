#!/usr/bin/env python3
"""
List all pandas .rolling() calls in basic_v1.py with optional perf prioritization.

Del 4: Can take a perf JSON to cross-reference and prioritize by actual cost.
"""

import re
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple


def load_perf_data(perf_json_path: Path) -> Dict:
    """Load perf data from JSON file."""
    try:
        with open(perf_json_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load perf JSON: {e}", file=sys.stderr)
        return {}


def extract_perf_rolling_data(perf_data: Dict) -> Dict[str, Dict]:
    """
    Extract rolling.pandas.* entries from perf data.
    Returns dict mapping name -> {total_sec, count, share_pct}
    """
    rolling_data = {}
    feature_top_blocks = perf_data.get("feature_top_blocks", [])
    for block in feature_top_blocks:
        name = block.get("name", "")
        if name.startswith("rolling.pandas."):
            rolling_data[name] = {
                "total_sec": block.get("total_sec", 0.0),
                "count": block.get("count", 0),
                "share_pct": block.get("share_of_feat_time_pct", 0.0),
            }
    return rolling_data


def parse_rolling_call(line: str, line_num: int) -> Optional[Dict]:
    """
    Parse a .rolling() call from a line.
    Returns dict with: window, operation, min_periods (if found), full_expr
    """
    # Match patterns like:
    # .rolling(48, min_periods=12)
    # .rolling(20).mean()
    # .rolling(10, min_periods=5).std(ddof=0)
    pattern = r'\.rolling\s*\(\s*(\d+)\s*(?:,\s*min_periods\s*=\s*(\d+))?\s*\)\s*\.(\w+)\s*\('
    match = re.search(pattern, line)
    if match:
        window = int(match.group(1))
        min_periods = int(match.group(2)) if match.group(2) else None
        operation = match.group(3)
        return {
            "window": window,
            "operation": operation,
            "min_periods": min_periods,
            "line": line_num,
            "full_expr": line.strip(),
        }
    return None


def list_rolling_calls(file_path: Path, perf_json_path: Optional[Path] = None):
    """
    List all pandas .rolling() calls in file_path.
    
    If perf_json_path is provided, cross-reference with perf data and prioritize.
    """
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return
    
    # Load perf data if provided
    perf_data = {}
    rolling_perf = {}
    if perf_json_path and perf_json_path.exists():
        perf_data = load_perf_data(perf_json_path)
        rolling_perf = extract_perf_rolling_data(perf_data)
        print(f"Loaded perf data from: {perf_json_path}")
        print(f"Found {len(rolling_perf)} pandas rolling entries in perf data\n")
    
    # Read file and find rolling calls
    with open(file_path) as f:
        lines = f.readlines()
    
    rolling_calls = []
    for i, line in enumerate(lines, 1):
        parsed = parse_rolling_call(line, i)
        if parsed:
            # Try to match with perf data
            # Format: rolling.pandas.<op>.w<window>
            perf_key = f"rolling.pandas.{parsed['operation']}.w{parsed['window']}"
            if perf_key in rolling_perf:
                parsed["perf_total_sec"] = rolling_perf[perf_key]["total_sec"]
                parsed["perf_count"] = rolling_perf[perf_key]["count"]
                parsed["perf_share_pct"] = rolling_perf[perf_key]["share_pct"]
            else:
                parsed["perf_total_sec"] = None
                parsed["perf_count"] = None
                parsed["perf_share_pct"] = None
            
            rolling_calls.append(parsed)
    
    if not rolling_calls:
        print("No pandas .rolling() calls found.")
        return
    
    # Sort by window size (or by perf_total_sec if available)
    if any(c.get("perf_total_sec") is not None for c in rolling_calls):
        rolling_calls.sort(key=lambda x: x.get("perf_total_sec") or 0.0, reverse=True)
        print("=" * 90)
        print("PANDAS ROLLING CALLS - PRIORITIZED BY PERFORMANCE")
        print("=" * 90)
    else:
        rolling_calls.sort(key=lambda x: x["window"], reverse=True)
        print("=" * 90)
        print("PANDAS ROLLING CALLS - SORTED BY WINDOW SIZE")
        print("=" * 90)
    
    print()
    print(f"{'#':<4} {'Window':<8} {'Op':<8} {'MinP':<6} {'Line':<6} {'Perf (sec)':<12} {'Calls':<8} {'Share %':<10}")
    print("-" * 90)
    
    for i, call in enumerate(rolling_calls[:10], 1):  # Top 10
        window = call["window"]
        op = call["operation"]
        minp = call.get("min_periods", "N/A")
        line_num = call["line"]
        perf_sec = call.get("perf_total_sec")
        perf_count = call.get("perf_count")
        perf_share = call.get("perf_share_pct")
        
        perf_sec_str = f"{perf_sec:.4f}" if perf_sec is not None else "N/A"
        perf_count_str = f"{perf_count}" if perf_count is not None else "N/A"
        perf_share_str = f"{perf_share:.2f}%" if perf_share is not None else "N/A"
        
        print(f"{i:<4} {window:<8} {op:<8} {minp!s:<6} {line_num:<6} {perf_sec_str:<12} {perf_count_str:<8} {perf_share_str:<10}")
    
    if len(rolling_calls) > 10:
        print(f"\n... and {len(rolling_calls) - 10} more calls")
    
    print()
    print("=" * 90)
    print("DETAILED CALLS (with context)")
    print("=" * 90)
    print()
    
    for call in rolling_calls[:10]:
        print(f"Line {call['line']}: {call['full_expr']}")
        if call.get("perf_total_sec") is not None:
            print(f"  → Perf: {call['perf_total_sec']:.4f}s ({call.get('perf_share_pct', 0):.2f}%), {call.get('perf_count', 0)} calls")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="List pandas .rolling() calls in basic_v1.py with optional perf prioritization."
    )
    parser.add_argument(
        "--perf",
        type=Path,
        help="Path to REPLAY_PERF_SUMMARY.json for performance prioritization"
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("gx1/features/basic_v1.py"),
        help="File to scan (default: gx1/features/basic_v1.py)"
    )
    
    args = parser.parse_args()
    
    list_rolling_calls(args.file, args.perf)


if __name__ == "__main__":
    main()
