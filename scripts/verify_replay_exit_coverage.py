#!/usr/bin/env python3
"""
Replay contract verification: Every trade must have exit_summary.

Hard fail in replay mode if any trade is missing exit_summary.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

def load_trades_from_jsonl(jsonl_path: Path) -> List[Dict]:
    """Load all trades from merged_trade_index.jsonl."""
    trades = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line))
    return trades

def load_trade_json(trade_json_path: Path) -> Dict:
    """Load individual trade JSON file."""
    with open(trade_json_path, 'r') as f:
        return json.load(f)

def verify_replay_exit_coverage(run_dir: Path) -> Dict[str, any]:
    """
    Verify replay contract: all trades must have exit_summary.
    
    Returns:
        Dict with verification results
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
    
    # Verify each trade
    missing_exit_summary = []
    
    for trade in trades:
        trade_uid = trade.get("trade_uid")
        if not trade_uid:
            continue
        
        trade_json_path = trade_json_files.get(trade_uid)
        if not trade_json_path:
            missing_exit_summary.append({
                "trade_uid": trade_uid,
                "trade_id": trade.get("trade_id"),
                "reason": "trade_json_file_not_found",
            })
            continue
        
        trade_json = load_trade_json(trade_json_path)
        exit_summary = trade_json.get("exit_summary")
        
        if not exit_summary:
            missing_exit_summary.append({
                "trade_uid": trade_uid,
                "trade_id": trade.get("trade_id"),
                "reason": "exit_summary_missing",
            })
    
    return {
        "total_trades": len(trades),
        "n_missing_exit_summary": len(missing_exit_summary),
        "missing_trades": missing_exit_summary,
        "all_have_exit_summary": len(missing_exit_summary) == 0,
    }

def main():
    parser = argparse.ArgumentParser(description="Verify replay exit coverage contract")
    parser.add_argument("run_dir", type=Path, help="Run directory with trade_journal/")
    parser.add_argument("--fail-fast", action="store_true", help="Fail fast if any trade missing exit_summary")
    
    args = parser.parse_args()
    
    if not args.run_dir.exists():
        print(f"ERROR: Run directory not found: {args.run_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("REPLAY EXIT COVERAGE CONTRACT VERIFICATION")
    print("=" * 80)
    print(f"Run directory: {args.run_dir}")
    print()
    
    # Verify
    result = verify_replay_exit_coverage(args.run_dir)
    
    print(f"Total trades: {result['total_trades']}")
    print(f"Missing exit_summary: {result['n_missing_exit_summary']}")
    print()
    
    if result['all_have_exit_summary']:
        print("✅ PASS: All trades have exit_summary")
        sys.exit(0)
    else:
        print("❌ FAIL: Some trades are missing exit_summary")
        print()
        print("First 20 missing trades:")
        for trade in result['missing_trades'][:20]:
            print(f"  {trade['trade_uid']} ({trade['trade_id']}): {trade['reason']}")
        
        if args.fail_fast:
            print()
            print("MISSING_EXIT_SUMMARY_REPLAY: Hard contract violation in replay mode")
            sys.exit(1)
        else:
            sys.exit(0)

if __name__ == "__main__":
    main()



