#!/usr/bin/env python3
"""
Exit coverage audit for FULLYEAR replay output.

Counts:
- trades_total
- n_with_exit_summary
- n_open_at_end (left_open / eof_close)
- n_missing_exit_summary

Diagnoses root cause by sampling trades without exit_summary.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

def load_trades_from_jsonl(jsonl_path: Path) -> List[Dict]:
    """Load all trades from merged_trade_index.jsonl."""
    trades = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line))
    return trades

def load_trade_json(trade_json_path: Path) -> Optional[Dict]:
    """Load individual trade JSON file."""
    try:
        with open(trade_json_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def audit_exit_coverage(run_dir: Path, sample_size: int = 20) -> Dict[str, Any]:
    """
    Audit exit coverage.
    
    Returns:
        Dict with audit results
    """
    jsonl_path = run_dir / "trade_journal" / "merged_trade_index.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL index not found: {jsonl_path}")
    
    # Load trades from JSONL
    trades = load_trades_from_jsonl(jsonl_path)
    print(f"Loaded {len(trades)} trades from JSONL")
    
    # Find trade JSON files
    chunks_dir = run_dir / "parallel_chunks"
    trade_json_files = {}
    
    if chunks_dir.exists():
        for chunk_dir in chunks_dir.glob("chunk_*"):
            trades_dir = chunk_dir / "trade_journal" / "trades"
            if trades_dir.exists():
                for json_file in trades_dir.glob("*.json"):
                    trade_json = load_trade_json(json_file)
                    if trade_json:
                        trade_uid = trade_json.get("trade_uid")
                        if trade_uid:
                            trade_json_files[trade_uid] = json_file
    
    print(f"Found {len(trade_json_files)} trade JSON files")
    
    # Audit each trade
    trades_total = len(trades)
    n_with_exit_summary = 0
    n_open_at_end = 0
    n_missing_exit_summary = 0
    missing_trade_uids = []
    
    for trade in trades:
        trade_uid = trade.get("trade_uid")
        if not trade_uid:
            continue
        
        trade_json_path = trade_json_files.get(trade_uid)
        if not trade_json_path:
            n_missing_exit_summary += 1
            missing_trade_uids.append(trade_uid)
            continue
        
        trade_json = load_trade_json(trade_json_path)
        if not trade_json:
            n_missing_exit_summary += 1
            missing_trade_uids.append(trade_uid)
            continue
        
        exit_summary = trade_json.get("exit_summary")
        if exit_summary:
            n_with_exit_summary += 1
            # Check if it's EOF-close
            exit_reason = exit_summary.get("exit_reason", "")
            if "EOF" in exit_reason.upper() or "LEFT_OPEN" in exit_reason.upper():
                n_open_at_end += 1
        else:
            n_missing_exit_summary += 1
            missing_trade_uids.append(trade_uid)
    
    # Sample missing trades for diagnosis
    sample_trades = []
    for trade_uid in missing_trade_uids[:sample_size]:
        trade_json_path = trade_json_files.get(trade_uid)
        if trade_json_path:
            trade_json = load_trade_json(trade_json_path)
            if trade_json:
                entry_snapshot = trade_json.get("entry_snapshot")
                exit_events = trade_json.get("exit_events", [])
                exit_summary = trade_json.get("exit_summary")
                
                # Try to find last seen timestamp
                last_seen_ts = None
                if exit_events:
                    last_seen_ts = exit_events[-1].get("timestamp")
                elif entry_snapshot:
                    last_seen_ts = entry_snapshot.get("entry_time")
                
                # Try to find bars_in_trade
                bars_in_trade = None
                if exit_events:
                    bars_in_trade = exit_events[-1].get("bars_held")
                elif exit_summary:
                    bars_in_trade = exit_summary.get("bars_held")
                
                sample_trades.append({
                    "trade_uid": trade_uid,
                    "has_entry_snapshot": entry_snapshot is not None,
                    "has_exit_events": len(exit_events) > 0,
                    "has_exit_summary": exit_summary is not None,
                    "bars_in_trade": bars_in_trade,
                    "last_seen_ts": last_seen_ts,
                    "exit_reason": exit_summary.get("exit_reason") if exit_summary else None,
                })
    
    return {
        "trades_total": trades_total,
        "n_with_exit_summary": n_with_exit_summary,
        "n_open_at_end": n_open_at_end,
        "n_missing_exit_summary": n_missing_exit_summary,
        "coverage_rate": n_with_exit_summary / trades_total if trades_total > 0 else 0.0,
        "sample_missing_trades": sample_trades,
        "missing_trade_uids": missing_trade_uids[:100],  # First 100 for reference
    }

def check_eof_close_activity(run_dir: Path) -> Dict[str, Any]:
    """Check if EOF-close is active per chunk."""
    chunks_dir = run_dir / "parallel_chunks"
    eof_info = {}
    
    if chunks_dir.exists():
        for chunk_dir in sorted(chunks_dir.glob("chunk_*")):
            chunk_id = chunk_dir.name
            log_file = chunk_dir / f"{chunk_id}.log"
            
            eof_found = False
            eof_close_count = 0
            
            if log_file.exists():
                with open(log_file, 'r') as f:
                    content = f.read()
                    if "EOF" in content or "left_open" in content.lower():
                        eof_found = True
                        # Count EOF closes
                        eof_close_count = content.lower().count("eof") + content.lower().count("left_open")
            
            eof_info[chunk_id] = {
                "eof_activity_found": eof_found,
                "eof_close_count": eof_close_count,
            }
    
    return eof_info

def main():
    parser = argparse.ArgumentParser(description="Audit exit coverage")
    parser.add_argument("run_dir", type=Path, help="Run directory with trade_journal/")
    parser.add_argument("--sample-size", type=int, default=20, help="Sample size for diagnosis (default: 20)")
    parser.add_argument("--output", type=Path, help="Output JSON file (default: run_dir/exit_coverage_audit.json)")
    
    args = parser.parse_args()
    
    if not args.run_dir.exists():
        print(f"ERROR: Run directory not found: {args.run_dir}")
        sys.exit(1)
    
    output_file = args.output or (args.run_dir / "exit_coverage_audit.json")
    
    print("=" * 80)
    print("EXIT COVERAGE AUDIT")
    print("=" * 80)
    print(f"Run directory: {args.run_dir}")
    print()
    
    # Audit
    audit_result = audit_exit_coverage(args.run_dir, sample_size=args.sample_size)
    
    # Check EOF activity
    eof_info = check_eof_close_activity(args.run_dir)
    audit_result["eof_activity"] = eof_info
    
    # Write JSON
    with open(output_file, 'w') as f:
        json.dump(audit_result, f, indent=2)
    print(f"✅ Wrote audit: {output_file}")
    
    # Print summary
    print()
    print("=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)
    print(f"Total trades: {audit_result['trades_total']}")
    print(f"With exit_summary: {audit_result['n_with_exit_summary']}")
    print(f"Open at end (EOF/left_open): {audit_result['n_open_at_end']}")
    print(f"Missing exit_summary: {audit_result['n_missing_exit_summary']}")
    print(f"Coverage rate: {audit_result['coverage_rate']*100:.1f}%")
    print()
    
    print("EOF Activity per chunk:")
    for chunk_id, info in sorted(eof_info.items()):
        print(f"  {chunk_id}: EOF activity={info['eof_activity_found']}, count={info['eof_close_count']}")
    print()
    
    if audit_result['sample_missing_trades']:
        print(f"Sample of {len(audit_result['sample_missing_trades'])} missing trades:")
        for trade in audit_result['sample_missing_trades'][:10]:
            print(f"  {trade['trade_uid']}:")
            print(f"    has_entry_snapshot: {trade['has_entry_snapshot']}")
            print(f"    has_exit_events: {trade['has_exit_events']}")
            print(f"    has_exit_summary: {trade['has_exit_summary']}")
            print(f"    bars_in_trade: {trade['bars_in_trade']}")
            print(f"    last_seen_ts: {trade['last_seen_ts']}")
            print(f"    exit_reason: {trade['exit_reason']}")
            print()

if __name__ == "__main__":
    main()



