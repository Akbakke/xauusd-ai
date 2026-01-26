#!/usr/bin/env python3
"""
Diagnose trade journal collisions in FULLYEAR output.

Prints detailed information about collision keys, sources, and trade details.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

def load_trades_from_journal_root(journal_root: Path, source_label: str) -> List[Dict]:
    """Load all trades from a journal root directory."""
    trades = []
    trades_dir = journal_root / "trades"
    if not trades_dir.exists():
        return trades
    
    for trade_file in sorted(trades_dir.glob("*.json")):
        try:
            with open(trade_file) as f:
                trade = json.load(f)
                trade["_source_file"] = str(trade_file)
                trade["_source_chunk"] = source_label
                trades.append(trade)
        except Exception as e:
            print(f"WARNING: Failed to load {trade_file}: {e}", file=sys.stderr)
    
    return trades


def diagnose_collisions(run_dir: Path) -> None:
    """Diagnose trade journal collisions."""
    run_dir = Path(run_dir)
    
    # Discover journal roots
    journal_roots = []
    parallel_chunks_dir = run_dir / "parallel_chunks"
    if parallel_chunks_dir.exists():
        for chunk_dir in sorted(parallel_chunks_dir.glob("chunk_*")):
            if chunk_dir.is_dir():
                journal_dir = chunk_dir / "trade_journal"
                if journal_dir.exists():
                    journal_roots.append((journal_dir, chunk_dir.name))
    else:
        # Single-run layout
        journal_dir = run_dir / "trade_journal"
        if journal_dir.exists():
            journal_roots.append((journal_dir, "single"))
    
    if not journal_roots:
        print(f"ERROR: No trade journals found in {run_dir}")
        sys.exit(1)
    
    # Load all trades
    all_trades = []
    for journal_root, source_label in journal_roots:
        trades = load_trades_from_journal_root(journal_root, source_label)
        all_trades.extend(trades)
        print(f"Loaded {len(trades)} trades from {source_label}")
    
    print(f"\nTotal trades loaded: {len(all_trades)}")
    
    # Group by trade_id to find collisions
    trades_by_id: Dict[str, List[Dict]] = defaultdict(list)
    for trade in all_trades:
        trade_id = trade.get("trade_id")
        if trade_id:
            trades_by_id[trade_id].append(trade)
    
    # Find collisions
    collisions = {tid: trades for tid, trades in trades_by_id.items() if len(trades) > 1}
    
    print(f"\nCollisions found: {len(collisions)}")
    print(f"Total duplicate trade_ids: {sum(len(trades) - 1 for trades in collisions.values())}")
    
    if not collisions:
        print("✅ No collisions found!")
        return
    
    # Sort by collision count (descending)
    sorted_collisions = sorted(collisions.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"\n=== TOP 20 COLLISION KEYS ===")
    for i, (trade_id, trades) in enumerate(sorted_collisions[:20], 1):
        print(f"\n{i}. trade_id: {trade_id}")
        print(f"   Occurrences: {len(trades)}")
        
        for j, trade in enumerate(trades, 1):
            entry = trade.get("entry_snapshot") or {}
            exit_ = trade.get("exit_summary") or {}
            entry_time = entry.get("entry_time") or entry.get("timestamp") or "N/A"
            exit_time = exit_.get("exit_time") or exit_.get("timestamp") or "N/A"
            side = entry.get("side") or trade.get("side") or "N/A"
            exit_reason = exit_.get("exit_reason") or "N/A"
            
            source_chunk = trade.get("_source_chunk", "unknown")
            source_file = trade.get("_source_file", "unknown")
            
            print(f"   [{j}] Source: {source_chunk}")
            print(f"       File: {source_file}")
            print(f"       Entry time: {entry_time}")
            print(f"       Exit time: {exit_time}")
            print(f"       Side: {side}")
            print(f"       Exit reason: {exit_reason}")
    
    # Summary statistics
    print(f"\n=== COLLISION SUMMARY ===")
    collision_by_source = defaultdict(lambda: defaultdict(int))
    for trade_id, trades in collisions.items():
        for trade in trades:
            source = trade.get("_source_chunk", "unknown")
            collision_by_source[source][trade_id] += 1
    
    print("\nCollisions per source chunk:")
    for source in sorted(collision_by_source.keys()):
        count = len(collision_by_source[source])
        print(f"  {source}: {count} unique trade_ids involved")
    
    # Check if collisions are always between specific chunks
    print("\nCollision patterns:")
    chunk_pairs = defaultdict(int)
    for trade_id, trades in collisions.items():
        sources = sorted(set(t.get("_source_chunk", "unknown") for t in trades))
        if len(sources) > 1:
            # Count pairs
            for i, s1 in enumerate(sources):
                for s2 in sources[i+1:]:
                    chunk_pairs[(s1, s2)] += 1
    
    for (s1, s2), count in sorted(chunk_pairs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {s1} <-> {s2}: {count} collisions")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/diagnose_trade_journal_collisions.py <run_dir>")
        sys.exit(1)
    
    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"ERROR: Directory not found: {run_dir}")
        sys.exit(1)
    
    diagnose_collisions(run_dir)

