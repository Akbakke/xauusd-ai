#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Units and Ranges Validator

Validates feature units and ranges against Feature Control Plane contracts.
Hard-fails on: bps outside reasonable bounds, ratio <0 when it doesn't make sense, extreme zscore.

Usage:
    python3 gx1/scripts/validate_feature_units_ranges.py \
        --prebuilt-parquet data/features/xauusd_m5_2025_features_v10_ctx.parquet \
        --manifest-json gx1/feature_manifest_v1.json \
        --out-root ../GX1_DATA/reports/feature_validation
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def get_git_sha() -> Optional[str]:
    """Get git SHA if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:8]
    except Exception:
        return None


def validate_units_ranges(
    prebuilt_parquet_path: Path,
    manifest_json_path: Path,
    output_dir: Path,
    fail_fast: bool = True,
) -> Dict[str, Any]:
    """
    Validate feature units and ranges.
    
    Args:
        prebuilt_parquet_path: Path to prebuilt features parquet
        manifest_json_path: Path to feature manifest JSON
        output_dir: Output directory for reports
        fail_fast: If True, hard-fail on first error
    
    Returns:
        Validation summary dict
    """
    log.info("=" * 60)
    log.info("FEATURE UNITS AND RANGES VALIDATOR")
    log.info("=" * 60)
    
    # Load manifest
    log.info(f"Loading manifest: {manifest_json_path}")
    with open(manifest_json_path, "r") as f:
        manifest = json.load(f)
    
    # Load prebuilt parquet (sample for efficiency)
    log.info(f"Loading prebuilt parquet: {prebuilt_parquet_path}")
    df = pd.read_parquet(prebuilt_parquet_path)
    
    log.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Initialize validation results
    validation_results = {
        "timestamp": datetime.now().isoformat(),
        "prebuilt_parquet_path": str(prebuilt_parquet_path.resolve()),
        "manifest_json_path": str(manifest_json_path.resolve()),
        "sys_executable": sys.executable,
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "git_sha": get_git_sha(),
        "rows_checked": len(df),
        "errors": [],
        "warnings": [],
        "passed": True,
        "feature_stats": {},
    }
    
    manifest_features = {f["name"]: f for f in manifest.get("features", [])}
    
    # Define reasonable bounds per units_contract
    UNITS_BOUNDS = {
        "bps": (-10000, 10000),  # -100% to +100% in bps (very wide, but catches extreme errors)
        "ratio": (-10, 10),  # Ratios can be negative (returns) or >1 (multiples)
        "zscore": (-10, 10),  # Z-scores should be within reasonable range
        "price": (0, None),  # Prices should be positive (no upper bound)
        "flag": (0, 1),  # Flags should be 0 or 1
        "int": (None, None),  # No bounds for int (context-dependent)
    }
    
    log.info("[VALIDATION] Checking units and ranges...")
    
    for feature_name in sorted(df.columns):
        if feature_name not in manifest_features:
            continue
        
        feature_info = manifest_features[feature_name]
        units_contract = feature_info.get("units_contract", "unknown")
        
        if units_contract == "unknown":
            continue  # Skip unknown units
        
        col = df[feature_name]
        
        # Skip model outputs (may have different ranges)
        if feature_info.get("stage") == "derived_from_model":
            continue
        
        # Compute stats
        stats = {
            "min": float(col.min()) if col.dtype in [np.float64, np.float32, np.int64, np.int32] else None,
            "max": float(col.max()) if col.dtype in [np.float64, np.float32, np.int64, np.int32] else None,
            "mean": float(col.mean()) if col.dtype in [np.float64, np.float32] else None,
            "std": float(col.std()) if col.dtype in [np.float64, np.float32] else None,
            "p5": float(col.quantile(0.05)) if col.dtype in [np.float64, np.float32] else None,
            "p95": float(col.quantile(0.95)) if col.dtype in [np.float64, np.float32] else None,
        }
        validation_results["feature_stats"][feature_name] = stats
        
        # Validate bounds
        if units_contract in UNITS_BOUNDS:
            min_bound, max_bound = UNITS_BOUNDS[units_contract]
            
            if min_bound is not None and stats["min"] is not None and stats["min"] < min_bound:
                error_msg = f"Feature {feature_name} (units_contract={units_contract}) has min={stats['min']:.4f} < {min_bound}"
                log.error(f"[UNITS_RANGE_FAIL] {error_msg}")
                validation_results["errors"].append(error_msg)
                validation_results["passed"] = False
                if fail_fast:
                    raise RuntimeError(f"[UNITS_RANGE_FAIL] {error_msg}")
            
            if max_bound is not None and stats["max"] is not None and stats["max"] > max_bound:
                error_msg = f"Feature {feature_name} (units_contract={units_contract}) has max={stats['max']:.4f} > {max_bound}"
                log.error(f"[UNITS_RANGE_FAIL] {error_msg}")
                validation_results["errors"].append(error_msg)
                validation_results["passed"] = False
                if fail_fast:
                    raise RuntimeError(f"[UNITS_RANGE_FAIL] {error_msg}")
        
        # Special validations
        if units_contract == "ratio":
            # Ratios that should be >= 0 (e.g., body_pct, vol_ratio)
            if feature_name in ["body_pct", "vol_ratio", "comp3_ratio"]:
                if stats["min"] is not None and stats["min"] < 0:
                    error_msg = f"Feature {feature_name} (units_contract=ratio) has min={stats['min']:.4f} < 0 (should be >= 0)"
                    log.error(f"[UNITS_RANGE_FAIL] {error_msg}")
                    validation_results["errors"].append(error_msg)
                    validation_results["passed"] = False
                    if fail_fast:
                        raise RuntimeError(f"[UNITS_RANGE_FAIL] {error_msg}")
        
        elif units_contract == "zscore":
            # Z-scores should be within reasonable range (most within -5 to +5)
            if stats["p5"] is not None and stats["p5"] < -10:
                warning_msg = f"Feature {feature_name} (units_contract=zscore) has p5={stats['p5']:.4f} < -10 (extreme)"
                log.warning(f"[UNITS_RANGE_WARN] {warning_msg}")
                validation_results["warnings"].append(warning_msg)
            
            if stats["p95"] is not None and stats["p95"] > 10:
                warning_msg = f"Feature {feature_name} (units_contract=zscore) has p95={stats['p95']:.4f} > 10 (extreme)"
                log.warning(f"[UNITS_RANGE_WARN] {warning_msg}")
                validation_results["warnings"].append(warning_msg)
        
        elif units_contract == "flag":
            # Flags should be exactly 0 or 1
            unique_values = col.unique()
            if len(unique_values) > 2 or not all(v in [0, 1, True, False] for v in unique_values):
                error_msg = f"Feature {feature_name} (units_contract=flag) has values other than 0/1: {unique_values[:10]}"
                log.error(f"[UNITS_RANGE_FAIL] {error_msg}")
                validation_results["errors"].append(error_msg)
                validation_results["passed"] = False
                if fail_fast:
                    raise RuntimeError(f"[UNITS_RANGE_FAIL] {error_msg}")
    
    # Write summary
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "UNITS_RANGES_VALIDATION_SUMMARY.json"
    with open(summary_path, "w") as f:
        json.dump(validation_results, f, indent=2)
    
    # Write markdown report
    md_path = output_dir / "UNITS_RANGES_VALIDATION_REPORT.md"
    md_content = f"""# Feature Units and Ranges Validation Report

**Generated:** {validation_results['timestamp']}  
**Prebuilt Parquet:** `{prebuilt_parquet_path}`  
**Manifest JSON:** `{manifest_json_path}`  
**Status:** {'✅ PASS' if validation_results['passed'] else '❌ FAIL'}

## Summary

- **Rows Checked:** {validation_results['rows_checked']:,}
- **Features Checked:** {len(validation_results['feature_stats'])}
- **Errors:** {len(validation_results['errors'])}
- **Warnings:** {len(validation_results['warnings'])}

## Errors

"""
    
    if validation_results["errors"]:
        for error in validation_results["errors"]:
            md_content += f"- ❌ {error}\n"
    else:
        md_content += "No errors.\n"
    
    md_content += "\n## Warnings\n\n"
    if validation_results["warnings"]:
        for warning in validation_results["warnings"]:
            md_content += f"- ⚠️ {warning}\n"
    else:
        md_content += "No warnings.\n"
    
    md_content += "\n## Feature Statistics (Top 20 by Range)\n\n"
    md_content += "| Feature | Units | Min | Max | Mean | Std | P5 | P95 |\n"
    md_content += "|---------|-------|-----|-----|------|-----|----|----|\n"
    
    # Sort by range (max - min)
    sorted_features = sorted(
        validation_results["feature_stats"].items(),
        key=lambda x: (x[1]["max"] or 0) - (x[1]["min"] or 0),
        reverse=True,
    )[:20]
    
    for feature_name, stats in sorted_features:
        feature_info = manifest_features.get(feature_name, {})
        units_contract = feature_info.get("units_contract", "unknown")
        md_content += f"| `{feature_name}` | {units_contract} | {stats['min']:.4f if stats['min'] is not None else 'N/A'} | {stats['max']:.4f if stats['max'] is not None else 'N/A'} | {stats['mean']:.4f if stats['mean'] is not None else 'N/A'} | {stats['std']:.4f if stats['std'] is not None else 'N/A'} | {stats['p5']:.4f if stats['p5'] is not None else 'N/A'} | {stats['p95']:.4f if stats['p95'] is not None else 'N/A'} |\n"
    
    md_content += f"""
## Metadata

- **sys.executable:** `{validation_results['sys_executable']}`
- **cwd:** `{validation_results['cwd']}`
- **git_sha:** `{validation_results['git_sha'] or 'N/A'}`
"""
    
    with open(md_path, "w") as f:
        f.write(md_content)
    
    if validation_results["passed"]:
        log.info("✅ Units and ranges validation PASSED")
    else:
        log.error(f"❌ Units and ranges validation FAILED: {len(validation_results['errors'])} errors")
        if fail_fast:
            raise RuntimeError(f"[UNITS_RANGES_VALIDATION_FAIL] {len(validation_results['errors'])} errors found")
    
    return validation_results


def main():
    parser = argparse.ArgumentParser(description="Feature Units and Ranges Validator")
    parser.add_argument(
        "--prebuilt-parquet",
        type=Path,
        required=True,
        help="Path to prebuilt features parquet",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=Path("gx1/feature_manifest_v1.json"),
        help="Path to feature manifest JSON",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root directory (default: GX1_REPORTS_ROOT/feature_validation)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=True,
        help="Hard-fail on first error (default: True)",
    )
    
    args = parser.parse_args()
    
    # Resolve output directory
    if args.out_root is None:
        default_reports_root = Path(os.getenv("GX1_REPORTS_ROOT", "../GX1_DATA/reports"))
        args.out_root = default_reports_root / "feature_validation"
    
    args.out_root.mkdir(parents=True, exist_ok=True)
    
    if not args.prebuilt_parquet.exists():
        raise FileNotFoundError(f"Prebuilt parquet not found: {args.prebuilt_parquet}")
    
    if not args.manifest_json.exists():
        raise FileNotFoundError(f"Manifest JSON not found: {args.manifest_json}")
    
    validation_results = validate_units_ranges(
        prebuilt_parquet_path=args.prebuilt_parquet,
        manifest_json_path=args.manifest_json,
        output_dir=args.out_root,
        fail_fast=args.fail_fast,
    )
    
    log.info(f"✅ Validation complete: {args.out_root}")
    log.info(f"   Summary: {args.out_root / 'UNITS_RANGES_VALIDATION_SUMMARY.json'}")
    log.info(f"   Report: {args.out_root / 'UNITS_RANGES_VALIDATION_REPORT.md'}")


if __name__ == "__main__":
    main()
