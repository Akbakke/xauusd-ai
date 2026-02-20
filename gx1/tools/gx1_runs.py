#!/usr/bin/env python3
"""
CLI tool to query and display run index entries.

Reads from $GX1_DATA/reports/_index.jsonl and displays recent runs.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from gx1.utils.output_dir import resolve_gx1_data_root
from gx1.utils.run_index import read_run_index


def format_timestamp(ts_utc: str) -> str:
    """Format ISO8601 timestamp to shorter format."""
    try:
        dt = datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts_utc[:19] if len(ts_utc) >= 19 else ts_utc


def format_value(value, default="—"):
    """Format a value for display, using default if None or missing."""
    if value is None:
        return default
    if isinstance(value, float):
        return f"{value:.2f}" if abs(value) < 10000 else f"{value:.0f}"
    return str(value)


def main():
    # Check if "verify" is the first argument (simple command dispatch)
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        # Import and run verify_run_index
        try:
            from gx1.scripts.verify_run_index import main as verify_main
            exit_code = verify_main()
            return exit_code
        except ImportError as e:
            print(f"ERROR: Failed to import verify_run_index: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"ERROR: Verification failed: {e}", file=sys.stderr)
            return 2
    
    # Default: list command
    parser = argparse.ArgumentParser(
        description="Query and display run index entries from _index.jsonl"
    )
    parser.add_argument(
        "--last",
        type=int,
        default=20,
        help="Show last N runs (default: 20)",
    )
    parser.add_argument(
        "--kind",
        type=str,
        default=None,
        help="Filter by kind (e.g., 'replay_eval', 'us_disabled_proof')",
    )
    parser.add_argument(
        "--status",
        type=str,
        default="any",
        choices=["any", "completed", "failed", "unknown"],
        help="Filter by status (default: any)",
    )
    parser.add_argument(
        "--since-hours",
        type=int,
        default=None,
        help="Only show runs from last N hours",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON entries instead of table",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug information (partial lines ignored, invalid lines count)",
    )
    
    args = parser.parse_args()
    
    # Resolve GX1_DATA root
    try:
        gx1_data_root = resolve_gx1_data_root()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    
    # Handle list command (or default)
    # Read run index with explicit partial-line handling
    reports_root = gx1_data_root / "reports"
    index_path = reports_root / "_index.jsonl"
    
    entries = []
    partial_last_line_ignored = False
    invalid_lines_ignored = 0
    
    try:
        if not index_path.exists():
            if args.debug:
                print(f"[DEBUG] Index file does not exist: {index_path}", file=sys.stderr)
        else:
            with open(index_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
                # Process all lines except possibly the last one
                for i, line in enumerate(lines):
                    is_last_line = (i == len(lines) - 1)
                    line_stripped = line.rstrip("\n\r")
                    
                    # Check if last line doesn't end with '\n' (partial line)
                    if is_last_line:
                        if not line.endswith("\n"):
                            partial_last_line_ignored = True
                            if args.debug:
                                print(f"[DEBUG] Ignoring partial last line (no newline)", file=sys.stderr)
                            continue
                    
                    if not line_stripped:
                        continue
                    
                    try:
                        entry = json.loads(line_stripped)
                        entries.append(entry)
                    except json.JSONDecodeError as e:
                        # If this is the last line and JSON parse fails, ignore it
                        if is_last_line:
                            partial_last_line_ignored = True
                            if args.debug:
                                print(f"[DEBUG] Ignoring partial last line (JSON parse error): {e}", file=sys.stderr)
                        else:
                            # Non-last line with parse error - count as invalid
                            invalid_lines_ignored += 1
                            if args.debug:
                                print(f"[DEBUG] Invalid line {i+1} (non-last): {e}", file=sys.stderr)
                        continue
    except Exception as e:
        print(f"ERROR: Failed to read run index: {e}", file=sys.stderr)
        return 1
    
    # Show debug stats if requested
    if args.debug:
        debug_info = {
            "partial_last_line_ignored": partial_last_line_ignored,
            "invalid_lines_ignored": invalid_lines_ignored,
        }
        print(json.dumps(debug_info, indent=2), file=sys.stderr)
    
    if not entries:
        print("No entries found in run index.", file=sys.stderr)
        return 0
    
    # Filter entries
    filtered = []
    now = datetime.now(timezone.utc)
    
    for entry in entries:
        # Filter by kind
        if args.kind and entry.get("kind") != args.kind:
            continue
        
        # Filter by status
        if args.status != "any":
            status_lower = entry.get("status", "UNKNOWN").lower()
            if args.status == "completed" and status_lower != "completed":
                continue
            if args.status == "failed" and status_lower != "failed":
                continue
            if args.status == "unknown" and status_lower != "unknown":
                continue
        
        # Filter by time
        if args.since_hours:
            try:
                ts_str = entry.get("ts_utc", "")
                if ts_str:
                    entry_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if (now - entry_dt) > timedelta(hours=args.since_hours):
                        continue
            except Exception:
                # Skip entries with invalid timestamps
                continue
        
        filtered.append(entry)
    
    # Sort by ts_utc descending (most recent first)
    filtered.sort(
        key=lambda e: e.get("ts_utc", ""),
        reverse=True,
    )
    
    # Limit to last N
    filtered = filtered[: args.last]
    
    # Output
    if args.json:
        # Raw JSON output
        for entry in filtered:
            print(json.dumps(entry, sort_keys=True))
    else:
        # Table output
        print(f"{'ts_utc':<20} {'status':<12} {'kind':<20} {'run_id':<30} {'pnl_bps':<10} {'max_dd_bps':<12} {'trades':<8} {'output_mode':<10}")
        print("-" * 140)
        
        for entry in filtered:
            ts = format_timestamp(entry.get("ts_utc", ""))
            status = entry.get("status", "UNKNOWN")
            kind = entry.get("kind", "unknown")
            run_id = entry.get("run_id", "")
            pnl_bps = format_value(entry.get("total_pnl_bps"))
            max_dd_bps = format_value(entry.get("max_dd_bps"))
            trades = format_value(entry.get("trades"))
            output_mode = entry.get("output_mode", "—")
            
            # Truncate long run_id
            if len(run_id) > 28:
                run_id = run_id[:25] + "..."
            
            print(
                f"{ts:<20} {status:<12} {kind:<20} {run_id:<30} {pnl_bps:<10} {max_dd_bps:<12} {trades:<8} {output_mode:<10}"
            )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
