#!/usr/bin/env python3
"""
Build XGB Input Contract Report.

Reads prebuilt parquet for multiple years, compares schemas,
and produces a feature matrix report for XGB input contract validation.

Usage:
    python3 gx1/scripts/build_xgb_input_contract_report.py --years 2020 2021 2022 2023 2024 2025 --n-bars 20000
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

# Constants
FEATURE_MANIFEST_PATHS = [
    WORKSPACE_ROOT / "gx1" / "feature_manifest_v1.json",
    WORKSPACE_ROOT / "gx1" / "features" / "feature_manifest_v1.json",
    WORKSPACE_ROOT / "gx1" / "models" / "entry_v9" / "nextgen_2020_2025_clean" / "entry_v9_feature_meta.json",
]

# Reserved columns (not features)
RESERVED_COLUMNS = {
    "ts", "timestamp", "time", "date", "datetime",
    "open", "high", "low", "close", "volume",
    "p_long_xgb", "p_hat_xgb", "uncertainty_score",
    "session", "session_id", "trading_session",
    "year", "month", "day", "hour", "minute",
    "index", "Unnamed: 0",
}


def resolve_gx1_data_dir() -> Path:
    """Resolve GX1_DATA directory."""
    if "GX1_DATA_ROOT" in os.environ:
        path = Path(os.environ["GX1_DATA_ROOT"])
        if path.exists():
            return path
    
    # Fallback
    default = WORKSPACE_ROOT.parent / "GX1_DATA"
    return default


def resolve_prebuilt_for_year(year: int, gx1_data: Path) -> Optional[Path]:
    """Resolve prebuilt parquet path for a given year."""
    # Priority order (updated based on actual GX1_DATA structure)
    candidates = [
        # TRIAL160 prebuilt (primary)
        gx1_data / "data" / "data" / "prebuilt" / "TRIAL160" / str(year) / f"xauusd_m5_{year}_features_v10_ctx.parquet",
        # Canonical features path
        gx1_data / "data" / "data" / "features" / f"xauusd_m5_{year}_features_v10_ctx.parquet",
        # Legacy prebuilt paths
        gx1_data / "prebuilt" / "prebuilt_v3" / f"prebuilt_{year}_v3.parquet",
        gx1_data / "prebuilt" / "prebuilt_v2" / f"prebuilt_{year}_v2.parquet",
        gx1_data / "prebuilt" / f"prebuilt_{year}.parquet",
        gx1_data / "prebuilt" / f"prebuilt_{year}_v3.parquet",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    # Glob search in data/data/prebuilt
    prebuilt_root = gx1_data / "data" / "data" / "prebuilt"
    if prebuilt_root.exists():
        for match in prebuilt_root.rglob(f"*{year}*features*.parquet"):
            return match
    
    # Glob search in prebuilt
    prebuilt_root = gx1_data / "prebuilt"
    if prebuilt_root.exists():
        for match in prebuilt_root.rglob(f"*{year}*.parquet"):
            return match
    
    return None


def load_feature_manifest() -> Optional[Dict[str, Any]]:
    """Load feature manifest from known paths."""
    for path in FEATURE_MANIFEST_PATHS:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    return None


def compute_schema_hash(columns: List[str]) -> str:
    """Compute hash of sorted column names."""
    sorted_cols = sorted(columns)
    cols_str = "|".join(sorted_cols)
    return hashlib.sha256(cols_str.encode()).hexdigest()[:16]


def get_numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get numeric columns that are likely features (not reserved)."""
    numeric_cols = []
    for col in df.columns:
        if col.lower() in RESERVED_COLUMNS:
            continue
        if df[col].dtype in [np.float32, np.float64, np.int32, np.int64]:
            numeric_cols.append(col)
    return sorted(numeric_cols)


def compute_feature_stats(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute statistics for each feature column."""
    stats = {}
    for col in feature_cols:
        if col not in df.columns:
            stats[col] = {"present": 0.0, "nan_pct": 100.0}
            continue
        
        series = df[col]
        n_total = len(series)
        n_nan = series.isna().sum()
        nan_pct = (n_nan / n_total) * 100 if n_total > 0 else 0.0
        
        valid = series.dropna()
        if len(valid) == 0:
            stats[col] = {
                "present": 100.0,
                "nan_pct": nan_pct,
                "mean": np.nan,
                "std": np.nan,
                "p1": np.nan,
                "p50": np.nan,
                "p99": np.nan,
                "min": np.nan,
                "max": np.nan,
            }
        else:
            stats[col] = {
                "present": 100.0,
                "nan_pct": nan_pct,
                "mean": float(valid.mean()),
                "std": float(valid.std()),
                "p1": float(np.percentile(valid, 1)),
                "p50": float(np.percentile(valid, 50)),
                "p99": float(np.percentile(valid, 99)),
                "min": float(valid.min()),
                "max": float(valid.max()),
            }
    
    return stats


def check_range_flags(stats: Dict[str, Dict[str, float]], year: int) -> List[Dict[str, Any]]:
    """Check for absurd ranges or high NaN rates."""
    flags = []
    
    for col, col_stats in stats.items():
        # High NaN rate
        nan_pct = col_stats.get("nan_pct", 0)
        if nan_pct > 0.1:
            flags.append({
                "year": year,
                "feature": col,
                "issue": "HIGH_NAN",
                "value": nan_pct,
                "details": f"NaN rate {nan_pct:.2f}% > 0.1%",
            })
        
        # Absurd range
        max_val = col_stats.get("max", 0)
        min_val = col_stats.get("min", 0)
        if not np.isnan(max_val) and abs(max_val) > 1e6:
            flags.append({
                "year": year,
                "feature": col,
                "issue": "ABSURD_MAX",
                "value": max_val,
                "details": f"Max value {max_val:.2e} > 1e6",
            })
        if not np.isnan(min_val) and abs(min_val) > 1e6:
            flags.append({
                "year": year,
                "feature": col,
                "issue": "ABSURD_MIN",
                "value": min_val,
                "details": f"Min value {min_val:.2e} > 1e6",
            })
        
        # Inf values (detected via p99 or mean)
        mean_val = col_stats.get("mean", 0)
        if not np.isnan(mean_val) and np.isinf(mean_val):
            flags.append({
                "year": year,
                "feature": col,
                "issue": "INF_VALUES",
                "value": mean_val,
                "details": "Infinite values detected",
            })
    
    return flags


def main():
    parser = argparse.ArgumentParser(
        description="Build XGB Input Contract Report"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2020, 2021, 2022, 2023, 2024, 2025],
        help="Years to analyze"
    )
    parser.add_argument(
        "--n-bars",
        type=int,
        default=20000,
        help="Number of bars to sample per year"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: reports/xgb_contract/)"
    )
    parser.add_argument(
        "--require-schema-match",
        type=int,
        default=1,
        help="Hard fail if schema mismatch between years (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    gx1_data = resolve_gx1_data_dir()
    print(f"GX1_DATA: {gx1_data}")
    
    # Load feature manifest
    manifest = load_feature_manifest()
    if manifest:
        print(f"Feature manifest loaded: {len(manifest.get('feature_cols', []))} features")
        manifest_features = set(manifest.get("feature_cols", []))
    else:
        print("WARNING: No feature manifest found, will infer from data")
        manifest_features = None
    
    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = WORKSPACE_ROOT / "reports" / "xgb_contract" / f"CONTRACT_REPORT_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Collect data per year
    year_data: Dict[int, Dict[str, Any]] = {}
    all_features: Set[str] = set()
    schema_hashes: Dict[int, str] = {}
    all_flags: List[Dict[str, Any]] = []
    
    for year in args.years:
        print(f"\n{'='*60}")
        print(f"Processing year {year}...")
        print(f"{'='*60}")
        
        # Find prebuilt
        prebuilt_path = resolve_prebuilt_for_year(year, gx1_data)
        if not prebuilt_path:
            print(f"  WARNING: No prebuilt found for {year}, skipping")
            continue
        
        print(f"  Prebuilt: {prebuilt_path}")
        
        # Load data
        try:
            df = pd.read_parquet(prebuilt_path)
            print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Sample if needed
            if len(df) > args.n_bars:
                # Sample evenly across the year
                step = len(df) // args.n_bars
                df = df.iloc[::step][:args.n_bars]
                print(f"  Sampled to {len(df)} rows")
            
            # Get numeric feature columns
            feature_cols = get_numeric_feature_columns(df)
            print(f"  Numeric feature columns: {len(feature_cols)}")
            
            # Compute schema hash
            schema_hash = compute_schema_hash(feature_cols)
            schema_hashes[year] = schema_hash
            print(f"  Schema hash: {schema_hash}")
            
            # Track all features
            all_features.update(feature_cols)
            
            # Compute stats
            stats = compute_feature_stats(df, feature_cols)
            
            # Check for range flags
            flags = check_range_flags(stats, year)
            all_flags.extend(flags)
            print(f"  Range flags: {len(flags)}")
            
            year_data[year] = {
                "prebuilt_path": str(prebuilt_path),
                "n_rows": len(df),
                "n_features": len(feature_cols),
                "schema_hash": schema_hash,
                "feature_cols": feature_cols,
                "stats": stats,
            }
            
        except Exception as e:
            print(f"  ERROR: Failed to process {year}: {e}")
            continue
    
    # Check schema consistency
    print(f"\n{'='*60}")
    print("Schema Consistency Check")
    print(f"{'='*60}")
    
    unique_hashes = set(schema_hashes.values())
    if len(unique_hashes) == 1:
        print(f"✅ All years have consistent schema (hash: {list(unique_hashes)[0]})")
    else:
        print(f"⚠️  Schema mismatch detected! {len(unique_hashes)} different schemas:")
        for year, hash_val in sorted(schema_hashes.items()):
            n_features = year_data.get(year, {}).get("n_features", 0)
            print(f"  {year}: {hash_val} ({n_features} features)")
        
        # Find missing/extra features between years
        all_year_features = {year: set(year_data.get(year, {}).get("feature_cols", [])) for year in args.years if year in year_data}
        reference_year = min(all_year_features.keys())
        reference_features = all_year_features[reference_year]
        
        for year in sorted(all_year_features.keys()):
            if year == reference_year:
                continue
            year_features = all_year_features[year]
            missing = reference_features - year_features
            extra = year_features - reference_features
            if missing:
                print(f"  {year} missing from {reference_year}: {list(missing)[:5]}...")
            if extra:
                print(f"  {year} extra vs {reference_year}: {list(extra)[:5]}...")
        
        if args.require_schema_match:
            print()
            print("=" * 60)
            print("FATAL: Schema mismatch and --require-schema-match=1")
            print("=" * 60)
            return 1
    
    # Build feature matrix
    print(f"\n{'='*60}")
    print("Building XGB_INPUT_CANDIDATE_MATRIX.csv")
    print(f"{'='*60}")
    
    all_features_sorted = sorted(all_features)
    matrix_rows = []
    
    for feature in all_features_sorted:
        row = {"feature": feature}
        for year in sorted(year_data.keys()):
            stats = year_data[year].get("stats", {}).get(feature, {})
            row[f"{year}_present"] = stats.get("present", 0)
            row[f"{year}_nan_pct"] = stats.get("nan_pct", 100)
            row[f"{year}_mean"] = stats.get("mean", np.nan)
            row[f"{year}_std"] = stats.get("std", np.nan)
            row[f"{year}_p1"] = stats.get("p1", np.nan)
            row[f"{year}_p50"] = stats.get("p50", np.nan)
            row[f"{year}_p99"] = stats.get("p99", np.nan)
        matrix_rows.append(row)
    
    matrix_df = pd.DataFrame(matrix_rows)
    matrix_path = output_dir / "XGB_INPUT_CANDIDATE_MATRIX.csv"
    matrix_df.to_csv(matrix_path, index=False)
    print(f"  Wrote: {matrix_path}")
    print(f"  Features: {len(all_features_sorted)}")
    
    # Write schema fingerprints
    fingerprints = {
        "generated_at": datetime.datetime.now().isoformat(),
        "years_analyzed": sorted(year_data.keys()),
        "n_features_total": len(all_features_sorted),
        "schema_hashes": schema_hashes,
        "schema_match": len(unique_hashes) == 1,
        "year_details": {
            year: {
                "prebuilt_path": data.get("prebuilt_path"),
                "n_rows": data.get("n_rows"),
                "n_features": data.get("n_features"),
                "schema_hash": data.get("schema_hash"),
            }
            for year, data in year_data.items()
        },
    }
    
    fingerprints_path = output_dir / "SCHEMA_FINGERPRINTS.json"
    with open(fingerprints_path, "w") as f:
        json.dump(fingerprints, f, indent=2)
    print(f"  Wrote: {fingerprints_path}")
    
    # Write range flags
    flags_path = output_dir / "RANGE_FLAGS.md"
    with open(flags_path, "w") as f:
        f.write("# XGB Input Range Flags\n\n")
        f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")
        
        if all_flags:
            f.write(f"## Summary\n\n")
            f.write(f"Total flags: {len(all_flags)}\n\n")
            
            # Group by issue type
            by_issue = {}
            for flag in all_flags:
                issue = flag["issue"]
                if issue not in by_issue:
                    by_issue[issue] = []
                by_issue[issue].append(flag)
            
            for issue, flags in sorted(by_issue.items()):
                f.write(f"### {issue} ({len(flags)} flags)\n\n")
                f.write("| Year | Feature | Value | Details |\n")
                f.write("|------|---------|-------|--------|\n")
                for flag in sorted(flags, key=lambda x: (x["year"], x["feature"])):
                    value = flag.get("value", "N/A")
                    if isinstance(value, float):
                        value = f"{value:.4f}"
                    f.write(f"| {flag['year']} | {flag['feature']} | {value} | {flag['details']} |\n")
                f.write("\n")
        else:
            f.write("## No flags detected ✅\n\n")
            f.write("All features within expected ranges.\n")
    
    print(f"  Wrote: {flags_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Years analyzed: {sorted(year_data.keys())}")
    print(f"Total features: {len(all_features_sorted)}")
    print(f"Schema match: {'✅ Yes' if len(unique_hashes) == 1 else '❌ No'}")
    print(f"Range flags: {len(all_flags)}")
    print(f"\nReports written to: {output_dir}")
    
    # Suggest next steps
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print("1. Review XGB_INPUT_CANDIDATE_MATRIX.csv for feature coverage")
    print("2. Check RANGE_FLAGS.md for problematic features")
    print("3. Create gx1/xgb/contracts/xgb_input_features_v1.json with:")
    print(f"   - features: [list of {len(all_features_sorted)} feature names]")
    print("   - expected_count: " + str(len(all_features_sorted)))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
