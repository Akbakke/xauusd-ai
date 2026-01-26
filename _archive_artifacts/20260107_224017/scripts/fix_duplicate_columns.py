#!/usr/bin/env python3
"""
Fix duplicate column names in data files (deterministic repair).

This script:
1. Detects case-insensitive column name collisions (e.g., 'close' and 'CLOSE')
2. Checks if colliding columns have identical content
3. If identical: drops the duplicate (keeps first occurrence by name order)
4. If different: hard fails with detailed diff stats

Usage:
    python3 scripts/fix_duplicate_columns.py <input_file> [--output <output_file>] [--dry-run]

If --output is not specified, overwrites input file (after backup).
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import hashlib
import json
from datetime import datetime


def detect_collisions(df: pd.DataFrame) -> dict:
    """Detect case-insensitive column name collisions."""
    cols = list(df.columns)
    lower_to_cols = defaultdict(list)
    for c in cols:
        k = str(c).lower()
        lower_to_cols[k].append(c)
    collisions = {k: v for k, v in lower_to_cols.items() if len(v) > 1}
    return collisions


def check_columns_identical(df: pd.DataFrame, col1: str, col2: str) -> tuple[bool, dict]:
    """
    Check if two columns have identical content.
    
    Returns:
        (is_identical: bool, stats: dict)
    """
    try:
        # Handle NaN/None comparisons
        arr1 = df[col1].to_numpy()
        arr2 = df[col2].to_numpy()
        
        # Check if arrays are identical (handles NaN correctly)
        is_identical = np.array_equal(arr1, arr2, equal_nan=True)
        
        if is_identical:
            return True, {}
        
        # Compute diff stats
        mask = np.isfinite(arr1) & np.isfinite(arr2)
        if mask.any():
            diff = arr1[mask] - arr2[mask]
            stats = {
                "n_different": np.sum(~np.isclose(arr1[mask], arr2[mask], equal_nan=True)),
                "max_abs_diff": float(np.max(np.abs(diff))),
                "mean_abs_diff": float(np.mean(np.abs(diff))),
                "min_col1": float(np.nanmin(arr1)),
                "max_col1": float(np.nanmax(arr1)),
                "min_col2": float(np.nanmin(arr2)),
                "max_col2": float(np.nanmax(arr2)),
            }
        else:
            stats = {
                "n_different": len(arr1),
                "note": "No finite values to compare",
            }
        
        return False, stats
    except Exception as e:
        return False, {"error": str(e)}


def fix_duplicate_columns(df: pd.DataFrame, dry_run: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """
    Fix duplicate columns deterministically.
    
    Returns:
        (fixed_df, removed_columns)
    """
    removed_cols = []
    df_fixed = df.copy()
    
    # Step 1: Detect case-insensitive collisions
    collisions = detect_collisions(df_fixed)
    
    if not collisions:
        # Check for exact duplicates too
        if df_fixed.columns.duplicated().any():
            dupes = df_fixed.columns[df_fixed.columns.duplicated()].tolist()
            raise ValueError(
                f"Exact duplicate columns found: {dupes}. "
                "This should not happen in pandas DataFrames. Fix upstream."
            )
        return df_fixed, removed_cols
    
    print(f"Found {len(collisions)} case-insensitive collision(s):")
    for norm_name, orig_cols in collisions.items():
        print(f"  '{norm_name}': {orig_cols}")
    
    # Step 2: For each collision group, check if columns are identical
    for norm_name, orig_cols in collisions.items():
        # Sort columns to ensure deterministic order (keep first alphabetically)
        orig_cols_sorted = sorted(orig_cols)
        keep_col = orig_cols_sorted[0]
        drop_cols = orig_cols_sorted[1:]
        
        print(f"\nChecking collision group '{norm_name}':")
        print(f"  Keeping: '{keep_col}' (first alphabetically)")
        
        for drop_col in drop_cols:
            print(f"  Checking '{drop_col}' vs '{keep_col}'...")
            is_identical, stats = check_columns_identical(df_fixed, keep_col, drop_col)
            
            if is_identical:
                print(f"    ✅ Identical - will drop '{drop_col}'")
                if not dry_run:
                    df_fixed = df_fixed.drop(columns=[drop_col])
                removed_cols.append(drop_col)
            else:
                error_msg = (
                    f"\n❌ Columns '{keep_col}' and '{drop_col}' differ!\n"
                    f"Stats: {stats}\n"
                    f"\nThis file requires manual repair. The columns are not identical, "
                    f"so automatic deduplication is not safe."
                )
                raise ValueError(error_msg)
    
    # Step 3: Rename remaining columns to lowercase for consistency
    # (Only if we fixed collisions, to avoid unnecessary changes)
    if removed_cols and not dry_run:
        # Normalize all columns to lowercase
        col_mapping = {c: c.lower() for c in df_fixed.columns}
        df_fixed = df_fixed.rename(columns=col_mapping)
        print(f"\n✅ Normalized all columns to lowercase")
    
    return df_fixed, removed_cols


def main():
    parser = argparse.ArgumentParser(
        description="Fix duplicate column names in data files (deterministic repair)"
    )
    parser.add_argument("input_file", type=Path, help="Input data file (CSV or Parquet)")
    parser.add_argument("--output", type=Path, default=None, help="Output file (default: overwrite input)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run: don't write changes")
    parser.add_argument("--backup", action="store_true", default=True, help="Create backup before overwriting")
    
    args = parser.parse_args()
    
    input_file = args.input_file
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    # Determine output file
    output_file = args.output if args.output else input_file
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Read file
    print("Reading file (this may take a moment for large files)...")
    try:
        if input_file.suffix.lower() == ".parquet":
            df = pd.read_parquet(input_file)
        else:
            df = pd.read_csv(input_file)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)[:10]}...")
    print()
    
    # Fix duplicates
    try:
        df_fixed, removed_cols, report_data = fix_duplicate_columns(df, dry_run=args.dry_run)
        
        if removed_cols:
            print(f"\n✅ Removed {len(removed_cols)} duplicate column(s): {removed_cols}")
        else:
            print("\n✅ No duplicates found - file is clean")
        
        # Write report
        if not args.dry_run and (removed_cols or report_data.get("collisions_detected")):
            report_dir = Path("reports/data_integrity")
            report_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = report_dir / f"fix_duplicate_columns_{timestamp}.md"
            
            with open(report_path, 'w') as f:
                f.write(f"# Duplicate Column Fix Report\n\n")
                f.write(f"**Timestamp:** {datetime.now().isoformat()}\n")
                f.write(f"**Input File:** {input_file}\n")
                f.write(f"**Output File:** {output_file}\n\n")
                
                f.write(f"## Summary\n\n")
                f.write(f"- **Collisions Detected:** {len(report_data.get('collisions_detected', {}))}\n")
                f.write(f"- **Columns Removed:** {len(report_data.get('columns_removed', []))}\n")
                f.write(f"- **Columns Kept:** {len(report_data.get('columns_kept', []))}\n\n")
                
                f.write(f"## Before Fix\n\n")
                before = report_data.get("before", {})
                f.write(f"- **Rows:** {before.get('n_rows', 0):,}\n")
                f.write(f"- **Columns:** {before.get('n_columns', 0)}\n")
                f.write(f"- **Columns Hash:** `{before.get('columns_hash', 'N/A')}`\n")
                f.write(f"- **Column Names:** {', '.join(before.get('column_names', [])[:20])}")
                if len(before.get('column_names', [])) > 20:
                    f.write(f" ... ({len(before.get('column_names', [])) - 20} more)")
                f.write(f"\n\n")
                
                if report_data.get("collisions_detected"):
                    f.write(f"## Collisions Detected\n\n")
                    for norm_name, orig_cols in report_data["collisions_detected"].items():
                        f.write(f"- **`{norm_name}`:** {orig_cols}\n")
                    f.write(f"\n")
                
                if report_data.get("columns_removed"):
                    f.write(f"## Columns Removed\n\n")
                    for item in report_data["columns_removed"]:
                        f.write(f"- **`{item['column']}`** (kept `{item['kept']}`, normalized: `{item['normalized_name']}`)\n")
                    f.write(f"\n")
                
                f.write(f"## After Fix\n\n")
                after = report_data.get("after", {})
                f.write(f"- **Rows:** {after.get('n_rows', 0):,}\n")
                f.write(f"- **Columns:** {after.get('n_columns', 0)}\n")
                f.write(f"- **Columns Hash:** `{after.get('columns_hash', 'N/A')}`\n")
                f.write(f"- **Column Names:** {', '.join(after.get('column_names', [])[:20])}")
                if len(after.get('column_names', [])) > 20:
                    f.write(f" ... ({len(after.get('column_names', [])) - 20} more)")
                f.write(f"\n\n")
                
                f.write(f"## Data Integrity\n\n")
                f.write(f"- **Rows unchanged:** {before.get('n_rows', 0) == after.get('n_rows', 0)}\n")
                f.write(f"- **Columns reduced by:** {before.get('n_columns', 0) - after.get('n_columns', 0)}\n")
                f.write(f"- **All columns normalized to lowercase:** Yes\n\n")
                
                f.write(f"## Full Report Data (JSON)\n\n")
                f.write(f"```json\n")
                f.write(json.dumps(report_data, indent=2, default=str))
                f.write(f"\n```\n")
            
            print(f"\n📄 Report written to: {report_path}")
        
        # Write output
        if not args.dry_run:
            # Create backup if overwriting
            if output_file == input_file and args.backup:
                backup_file = input_file.with_suffix(input_file.suffix + ".backup")
                print(f"\nCreating backup: {backup_file}")
                if input_file.suffix.lower() == ".parquet":
                    pd.read_parquet(input_file).to_parquet(backup_file)
                else:
                    pd.read_csv(input_file).to_csv(backup_file, index=False)
            
            print(f"\nWriting fixed file to: {output_file}")
            if output_file.suffix.lower() == ".parquet":
                df_fixed.to_parquet(output_file, index=False)
            else:
                df_fixed.to_csv(output_file, index=False)
            
            print("✅ Done!")
        else:
            print("\n✅ Dry run complete - no changes written")
            
    except ValueError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

