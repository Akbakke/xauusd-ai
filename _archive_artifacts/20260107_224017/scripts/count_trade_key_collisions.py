#!/usr/bin/env python3
"""
Count collisions on trade_key in trade_journal_index.csv (COMMIT D).

Usage:
    python3 scripts/count_trade_key_collisions.py <index_csv_path>
"""
import csv
import sys
from pathlib import Path
from collections import Counter


def count_trade_key_collisions(index_csv_path: Path) -> tuple[int, list[tuple[str, int]]]:
    """
    Count collisions on trade_key in trade_journal_index.csv (COMMIT D).
    
    Returns:
        (collisions_count, top_duplicates)
        collisions_count: total_rows - unique(trade_key)
        top_duplicates: list of (trade_key, count) tuples for duplicate keys (max 20)
    """
    if not index_csv_path.exists():
        return 0, []
    
    trade_keys = []
    with open(index_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trade_key = row.get('trade_key', '').strip()
            if trade_key:
                trade_keys.append(trade_key)
    
    total_rows = len(trade_keys)
    unique_keys = len(set(trade_keys))
    collisions_count = total_rows - unique_keys
    
    # Find top duplicates
    key_counts = Counter(trade_keys)
    duplicates = [(key, count) for key, count in key_counts.items() if count > 1]
    duplicates.sort(key=lambda x: x[1], reverse=True)
    
    return collisions_count, duplicates[:20]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: count_trade_key_collisions.py <index_csv_path>")
        sys.exit(1)
    
    index_path = Path(sys.argv[1])
    collisions_count, top_duplicates = count_trade_key_collisions(index_path)
    
    print(f"collisions_count={collisions_count}")
    if top_duplicates:
        print(f"unique_duplicate_keys={len(set([k for k, _ in top_duplicates]))}")
        print("top_duplicates:")
        for key, count in top_duplicates:
            print(f"  {key}: {count}")
    else:
        print("unique_duplicate_keys=0")

