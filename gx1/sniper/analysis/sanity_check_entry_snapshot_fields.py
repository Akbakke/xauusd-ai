#!/usr/bin/env python3
"""
Sanity check for entry_snapshot fields in trade journals.

This script reads trade_journal/trades/*.json and reports coverage (% non-null)
for critical entry_snapshot fields:
- entry_snapshot (top-level)
- entry_snapshot.base_units
- entry_snapshot.units
- entry_snapshot.sniper_overlays
- entry_snapshot.session
- entry_snapshot.trend_regime
- entry_snapshot.vol_regime

Usage:
    python gx1/sniper/analysis/sanity_check_entry_snapshot_fields.py --run-dir gx1/wf_runs/SNIPER_OBS_Q4_2025_baseline_*
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

SNIPER_BASE = Path("gx1/wf_runs")


def check_entry_snapshot_fields(run_dir: Path) -> Dict[str, Any]:
    """
    Check entry_snapshot field coverage in trade journals.
    
    Returns dict with:
    - total_trades: int
    - has_entry_snapshot: int
    - coverage: dict mapping field names to (count, percentage)
    """
    trades_dir = run_dir / "trade_journal" / "trades"
    
    if not trades_dir.exists():
        return {
            "total_trades": 0,
            "has_entry_snapshot": 0,
            "coverage": {},
            "error": "trades directory does not exist",
        }
    
    total_trades = 0
    has_entry_snapshot = 0
    coverage: Dict[str, int] = {
        "base_units": 0,
        "units": 0,
        "sniper_overlays": 0,
        "session": 0,
        "trend_regime": 0,
        "vol_regime": 0,
    }
    
    for json_file in sorted(trades_dir.glob("*.json")):
        total_trades += 1
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                trade = json.load(f)
            
            entry_snapshot = trade.get("entry_snapshot")
            if entry_snapshot is not None and isinstance(entry_snapshot, dict):
                has_entry_snapshot += 1
                
                # Check each field
                if entry_snapshot.get("base_units") is not None:
                    coverage["base_units"] += 1
                if entry_snapshot.get("units") is not None:
                    coverage["units"] += 1
                if entry_snapshot.get("sniper_overlays") is not None:
                    coverage["sniper_overlays"] += 1
                if entry_snapshot.get("session") is not None:
                    coverage["session"] += 1
                if entry_snapshot.get("trend_regime") is not None:
                    coverage["trend_regime"] += 1
                if entry_snapshot.get("vol_regime") is not None:
                    coverage["vol_regime"] += 1
        except Exception as e:
            print(f"WARNING: Failed to read {json_file.name}: {e}")
            continue
    
    # Calculate percentages
    coverage_pct: Dict[str, tuple[int, float]] = {}
    for field, count in coverage.items():
        pct = (count / total_trades * 100.0) if total_trades > 0 else 0.0
        coverage_pct[field] = (count, pct)
    
    return {
        "total_trades": total_trades,
        "has_entry_snapshot": has_entry_snapshot,
        "entry_snapshot_pct": (has_entry_snapshot / total_trades * 100.0) if total_trades > 0 else 0.0,
        "coverage": coverage_pct,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sanity check entry_snapshot fields in trade journals"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to SNIPER run directory (e.g., gx1/wf_runs/SNIPER_OBS_Q4_2025_baseline_*)",
    )
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: Run directory does not exist: {run_dir}")
        return 0  # Exit 0 as requested
    
    print(f"Checking entry_snapshot fields in: {run_dir.name}")
    print()
    
    results = check_entry_snapshot_fields(run_dir)
    
    if results.get("error"):
        print(f"ERROR: {results['error']}")
        return 0
    
    total = results["total_trades"]
    has_entry = results["has_entry_snapshot"]
    entry_pct = results["entry_snapshot_pct"]
    
    print("=" * 60)
    print("ENTRY_SNAPSHOT FIELD COVERAGE")
    print("=" * 60)
    print()
    print(f"Total trades: {total}")
    print(f"Trades with entry_snapshot: {has_entry} / {total} ({entry_pct:.1f}%)")
    print()
    
    if has_entry == 0:
        print("⚠️  WARNING: No trades have entry_snapshot!")
        print()
        return 0
    
    print("Field Coverage (within trades that have entry_snapshot):")
    print()
    print(f"{'Field':<20} {'Count':<10} {'% of Total':<15} {'% of Has Entry':<15}")
    print("-" * 60)
    
    for field, (count, pct_total) in results["coverage"].items():
        pct_has_entry = (count / has_entry * 100.0) if has_entry > 0 else 0.0
        print(f"{field:<20} {count:<10} {pct_total:>6.1f}%        {pct_has_entry:>6.1f}%")
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Check critical fields
    critical_fields = ["base_units", "units", "sniper_overlays"]
    all_critical_present = all(
        results["coverage"][field][0] > 0 for field in critical_fields
    )
    
    if all_critical_present:
        print("✅ All critical fields (base_units, units, sniper_overlays) are present")
    else:
        print("⚠️  Some critical fields are missing:")
        for field in critical_fields:
            count, _ = results["coverage"][field]
            if count == 0:
                print(f"   - {field}: 0 trades")
    
    print()
    return 0


if __name__ == "__main__":
    exit(main())

