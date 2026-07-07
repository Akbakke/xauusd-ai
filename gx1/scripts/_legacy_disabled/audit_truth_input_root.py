#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit Truth Decomposition Input Root

DEL 1: Discover years, count trades, detect paths for debugging.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Any

# Add workspace root to path
import sys
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def discover_years_and_trades(input_root: Path) -> Dict[str, Any]:
    """
    Recursively scan input_root for year directories and trade files.
    
    Returns dict with:
    - detected_years: List[int]
    - year_counts: Dict[int, int]
    - discovered_paths: List[str] (top 20 relevant)
    - year_paths: Dict[int, List[str]]
    """
    detected_years: Set[int] = set()
    year_counts: Dict[int, int] = {}
    year_paths: Dict[int, List[str]] = {}
    discovered_paths: List[str] = []
    
    # Pattern 1: Direct year directories {2020, 2021, ...}
    for year_dir in sorted(input_root.glob("[0-9][0-9][0-9][0-9]")):
        try:
            year = int(year_dir.name)
            if 2020 <= year <= 2025:
                detected_years.add(year)
                year_paths[year] = []
                
                # Look for trade files
                trade_dirs = [
                    year_dir / "chunk_0" / "trade_journal" / "trades",
                    year_dir / "chunk_1" / "trade_journal" / "trades",
                ]
                
                for trade_dir in trade_dirs:
                    if trade_dir.exists():
                        trade_files = list(trade_dir.glob("*.json"))
                        count = len(trade_files)
                        year_counts[year] = year_counts.get(year, 0) + count
                        path_str = str(trade_dir.relative_to(input_root))
                        year_paths[year].append(path_str)
                        if len(discovered_paths) < 20:
                            discovered_paths.append(path_str)
                        log.info(f"Found {count} trades in {path_str}")
        
        except ValueError:
            continue
    
    # Pattern 2: YEAR_{year} directories
    for year_dir in sorted(input_root.glob("YEAR_[0-9][0-9][0-9][0-9]")):
        try:
            year = int(year_dir.name.split("_")[1])
            if 2020 <= year <= 2025:
                detected_years.add(year)
                if year not in year_paths:
                    year_paths[year] = []
                
                trade_dirs = [
                    year_dir / "chunk_0" / "trade_journal" / "trades",
                ]
                
                for trade_dir in trade_dirs:
                    if trade_dir.exists():
                        trade_files = list(trade_dir.glob("*.json"))
                        count = len(trade_files)
                        year_counts[year] = year_counts.get(year, 0) + count
                        path_str = str(trade_dir.relative_to(input_root))
                        year_paths[year].append(path_str)
                        if len(discovered_paths) < 20:
                            discovered_paths.append(path_str)
        
        except (ValueError, IndexError):
            continue
    
    # Pattern 3: Nested YEAR_{year} (e.g., archive/REPLAY_RAW_*/TRIAL160_YEARLY/{year})
    for nested_dir in sorted(input_root.rglob("YEAR_[0-9][0-9][0-9][0-9]")):
        try:
            year = int(nested_dir.name.split("_")[1])
            if 2020 <= year <= 2025:
                detected_years.add(year)
                if year not in year_paths:
                    year_paths[year] = []
                
                trade_dirs = [
                    nested_dir / "chunk_0" / "trade_journal" / "trades",
                ]
                
                for trade_dir in trade_dirs:
                    if trade_dir.exists():
                        trade_files = list(trade_dir.glob("*.json"))
                        count = len(trade_files)
                        year_counts[year] = year_counts.get(year, 0) + count
                        path_str = str(trade_dir.relative_to(input_root))
                        year_paths[year].append(path_str)
                        if len(discovered_paths) < 20:
                            discovered_paths.append(path_str)
        
        except (ValueError, IndexError):
            continue
    
    # Pattern 4: {year}_CANARY_* or similar
    for year_dir in sorted(input_root.glob("[0-9][0-9][0-9][0-9]_*")):
        try:
            year = int(year_dir.name.split("_")[0])
            if 2020 <= year <= 2025:
                detected_years.add(year)
                if year not in year_paths:
                    year_paths[year] = []
                
                trade_dirs = [
                    year_dir / "chunk_0" / "trade_journal" / "trades",
                ]
                
                for trade_dir in trade_dirs:
                    if trade_dir.exists():
                        trade_files = list(trade_dir.glob("*.json"))
                        count = len(trade_files)
                        year_counts[year] = year_counts.get(year, 0) + count
                        path_str = str(trade_dir.relative_to(input_root))
                        year_paths[year].append(path_str)
                        if len(discovered_paths) < 20:
                            discovered_paths.append(path_str)
        
        except (ValueError, IndexError):
            continue
    
    return {
        "detected_years": sorted(list(detected_years)),
        "year_counts": {str(k): v for k, v in sorted(year_counts.items())},
        "discovered_paths": discovered_paths,
        "year_paths": {str(k): v for k, v in sorted(year_paths.items())},
    }


def main():
    parser = argparse.ArgumentParser(description="Audit Truth Decomposition Input Root")
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory to scan for year directories and trade files",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    workspace_root = Path(__file__).parent.parent.parent
    if not args.input_root.is_absolute():
        args.input_root = workspace_root / args.input_root
    
    if not args.input_root.exists():
        log.error(f"Input root does not exist: {args.input_root}")
        return 1
    
    log.info("=" * 60)
    log.info("AUDIT TRUTH DECOMPOSITION INPUT ROOT")
    log.info("=" * 60)
    log.info(f"Input root: {args.input_root}")
    log.info("")
    
    # Discover years and trades
    audit_data = discover_years_and_trades(args.input_root)
    
    detected_years = audit_data["detected_years"]
    year_counts = audit_data["year_counts"]
    
    log.info(f"Detected years: {detected_years}")
    log.info(f"Year counts: {year_counts}")
    log.info("")
    
    # ============================================================================
    # INVARIANT: Year Coverage
    # ============================================================================
    # FATAL if < 2 years (prevents "only 2023" scenarios)
    # ============================================================================
    
    # FATAL if less than 2 years
    if len(detected_years) < 2:
        log.error(f"❌ FATAL: Only {len(detected_years)} year(s) detected. Need at least 2 years.")
        log.error(f"   Detected: {detected_years}")
        log.error("   This invariant prevents accidental single-year analysis.")
        return 1
    
    # Warnings for missing years
    expected_years = [2020, 2021, 2022, 2023, 2024, 2025]
    missing_years = [y for y in expected_years if y not in detected_years]
    if missing_years:
        log.warning(f"⚠️  Missing years: {missing_years}")
    
    # Create output directory
    output_dir = workspace_root / "reports" / "truth_decomp" / "_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write JSON report
    json_path = output_dir / "AUDIT_INPUT_ROOT.json"
    with open(json_path, "w") as f:
        json.dump(audit_data, f, indent=2)
    log.info(f"✅ Wrote JSON report: {json_path}")
    
    # Write Markdown report
    md_path = output_dir / "AUDIT_INPUT_ROOT.md"
    with open(md_path, "w") as f:
        f.write("# Truth Decomposition Input Root Audit\n\n")
        f.write(f"**Input Root:** `{args.input_root}`\n\n")
        f.write(f"**Generated:** {Path(__file__).stat().st_mtime}\n\n")
        f.write("## Detected Years\n\n")
        f.write(f"- **Years Found:** {len(detected_years)}\n")
        f.write(f"- **Years:** {', '.join(map(str, detected_years))}\n\n")
        f.write("## Year Trade Counts\n\n")
        f.write("| Year | Trade Count |\n")
        f.write("|------|-------------|\n")
        for year in sorted(detected_years):
            count = year_counts.get(str(year), 0)
            f.write(f"| {year} | {count:,} |\n")
        f.write("\n")
        
        if missing_years:
            f.write("## ⚠️ Missing Years\n\n")
            f.write(f"- {', '.join(map(str, missing_years))}\n\n")
        
        f.write("## Discovered Paths (Top 20)\n\n")
        for i, path in enumerate(audit_data["discovered_paths"][:20], 1):
            f.write(f"{i}. `{path}`\n")
        f.write("\n")
        
        f.write("## Year Paths\n\n")
        for year in sorted(detected_years):
            f.write(f"### {year}\n\n")
            paths = audit_data["year_paths"].get(str(year), [])
            if paths:
                for path in paths:
                    f.write(f"- `{path}`\n")
            else:
                f.write("- *No paths found*\n")
            f.write("\n")
    
    log.info(f"✅ Wrote Markdown report: {md_path}")
    
    log.info("")
    log.info("=" * 60)
    log.info("✅ AUDIT COMPLETE")
    log.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
