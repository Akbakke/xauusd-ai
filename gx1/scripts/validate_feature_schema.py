#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Schema Validator

Validates prebuilt feature parquet schema against Feature Control Plane contracts.
Hard-fails on: index not DatetimeIndex UTC, missing columns, NaN/Inf in features, dtype mismatch.

Usage:
    python3 gx1/scripts/validate_feature_schema.py \
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
from typing import Dict, List, Any, Optional

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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
        return result.stdout.strip()[:8]  # First 8 chars
    except Exception:
        return None


def validate_feature_schema(
    prebuilt_parquet_path: Path,
    manifest_json_path: Path,
    output_dir: Path,
    fail_fast: bool = True,
) -> Dict[str, Any]:
    """
    Validate feature schema.
    
    Args:
        prebuilt_parquet_path: Path to prebuilt features parquet
        manifest_json_path: Path to feature manifest JSON
        output_dir: Output directory for reports
        fail_fast: If True, hard-fail on first error
    
    Returns:
        Validation summary dict
    """
    log.info("=" * 60)
    log.info("FEATURE SCHEMA VALIDATOR")
    log.info("=" * 60)
    
    # Load manifest
    log.info(f"Loading manifest: {manifest_json_path}")
    with open(manifest_json_path, "r") as f:
        manifest = json.load(f)
    
    # Load prebuilt parquet
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
        "rows_loaded": len(df),
        "columns_loaded": len(df.columns),
        "errors": [],
        "warnings": [],
        "passed": True,
    }
    
    # 1. Validate index is DatetimeIndex UTC
    log.info("[VALIDATION] Checking index type and timezone...")
    if not isinstance(df.index, pd.DatetimeIndex):
        error_msg = f"Index is not DatetimeIndex, got: {type(df.index)}"
        log.error(f"[SCHEMA_FAIL] {error_msg}")
        validation_results["errors"].append(error_msg)
        validation_results["passed"] = False
        if fail_fast:
            raise RuntimeError(f"[SCHEMA_FAIL] {error_msg}")
    
    if df.index.tz is None:
        error_msg = "Index timezone is None (must be UTC)"
        log.error(f"[SCHEMA_FAIL] {error_msg}")
        validation_results["errors"].append(error_msg)
        validation_results["passed"] = False
        if fail_fast:
            raise RuntimeError(f"[SCHEMA_FAIL] {error_msg}")
    
    if str(df.index.tz) != "UTC":
        error_msg = f"Index timezone is not UTC, got: {df.index.tz}"
        log.error(f"[SCHEMA_FAIL] {error_msg}")
        validation_results["errors"].append(error_msg)
        validation_results["passed"] = False
        if fail_fast:
            raise RuntimeError(f"[SCHEMA_FAIL] {error_msg}")
    
    log.info("✅ Index is DatetimeIndex UTC")
    
    # 2. Validate required columns from manifest
    log.info("[VALIDATION] Checking required columns from manifest...")
    manifest_features = {f["name"]: f for f in manifest.get("features", [])}
    expected_features = set(manifest_features.keys())
    actual_features = set(df.columns)
    
    missing_features = expected_features - actual_features
    if missing_features:
        error_msg = f"Missing {len(missing_features)} features from manifest: {sorted(list(missing_features))[:10]}"
        log.error(f"[SCHEMA_FAIL] {error_msg}")
        validation_results["errors"].append(error_msg)
        validation_results["passed"] = False
        if fail_fast:
            raise RuntimeError(f"[SCHEMA_FAIL] {error_msg}")
    
    extra_features = actual_features - expected_features
    if extra_features:
        warning_msg = f"Extra {len(extra_features)} features not in manifest: {sorted(list(extra_features))[:10]}"
        log.warning(f"[SCHEMA_WARN] {warning_msg}")
        validation_results["warnings"].append(warning_msg)
    
    log.info(f"✅ Column count match: {len(expected_features)} expected, {len(actual_features)} actual")
    
    # 3. Validate dtype for flag/int features
    log.info("[VALIDATION] Checking dtype for flag/int features...")
    dtype_errors = []
    for feature_name, feature_info in manifest_features.items():
        if feature_name not in df.columns:
            continue
        
        units = feature_info.get("units", "unknown")
        units_contract = feature_info.get("units_contract", units)
        
        if units_contract == "flag":
            # Flags should be int or bool
            dtype = df[feature_name].dtype
            if dtype not in [np.int64, np.int32, np.int16, np.int8, bool]:
                error_msg = f"Feature {feature_name} has units_contract=flag but dtype={dtype} (expected int or bool)"
                log.error(f"[SCHEMA_FAIL] {error_msg}")
                dtype_errors.append(error_msg)
                validation_results["errors"].append(error_msg)
                validation_results["passed"] = False
                if fail_fast:
                    raise RuntimeError(f"[SCHEMA_FAIL] {error_msg}")
        
        elif units_contract == "int":
            # Int features should be int
            dtype = df[feature_name].dtype
            if dtype not in [np.int64, np.int32, np.int16, np.int8]:
                error_msg = f"Feature {feature_name} has units_contract=int but dtype={dtype} (expected int)"
                log.error(f"[SCHEMA_FAIL] {error_msg}")
                dtype_errors.append(error_msg)
                validation_results["errors"].append(error_msg)
                validation_results["passed"] = False
                if fail_fast:
                    raise RuntimeError(f"[SCHEMA_FAIL] {error_msg}")
    
    if not dtype_errors:
        log.info("✅ Dtype validation passed")
    
    # 4. Validate NaN/Inf in features
    log.info("[VALIDATION] Checking for NaN/Inf in features...")
    nan_inf_errors = []
    for feature_name in sorted(df.columns):
        if feature_name not in manifest_features:
            continue
        
        feature_info = manifest_features[feature_name]
        units_contract = feature_info.get("units_contract", "unknown")
        
        # Model outputs and some features may have NaN (check manifest for explicit NaN tolerance)
        if feature_info.get("stage") == "derived_from_model":
            continue  # Skip model outputs (may have NaN)
        
        col = df[feature_name]
        nan_count = col.isna().sum()
        inf_count = np.isinf(col).sum() if col.dtype in [np.float64, np.float32] else 0
        
        if nan_count > 0:
            error_msg = f"Feature {feature_name} has {nan_count} NaN values (units_contract={units_contract})"
            log.error(f"[SCHEMA_FAIL] {error_msg}")
            nan_inf_errors.append(error_msg)
            validation_results["errors"].append(error_msg)
            validation_results["passed"] = False
            if fail_fast:
                raise RuntimeError(f"[SCHEMA_FAIL] {error_msg}")
        
        if inf_count > 0:
            error_msg = f"Feature {feature_name} has {inf_count} Inf values (units_contract={units_contract})"
            log.error(f"[SCHEMA_FAIL] {error_msg}")
            nan_inf_errors.append(error_msg)
            validation_results["errors"].append(error_msg)
            validation_results["passed"] = False
            if fail_fast:
                raise RuntimeError(f"[SCHEMA_FAIL] {error_msg}")
    
    if not nan_inf_errors:
        log.info("✅ NaN/Inf validation passed")
    
    # Write summary
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "SCHEMA_VALIDATION_SUMMARY.json"
    with open(summary_path, "w") as f:
        json.dump(validation_results, f, indent=2)
    
    # Write markdown report
    md_path = output_dir / "SCHEMA_VALIDATION_REPORT.md"
    md_content = f"""# Feature Schema Validation Report

**Generated:** {validation_results['timestamp']}  
**Prebuilt Parquet:** `{prebuilt_parquet_path}`  
**Manifest JSON:** `{manifest_json_path}`  
**Status:** {'✅ PASS' if validation_results['passed'] else '❌ FAIL'}

## Summary

- **Rows Loaded:** {validation_results['rows_loaded']:,}
- **Columns Loaded:** {validation_results['columns_loaded']}
- **Errors:** {len(validation_results['errors'])}
- **Warnings:** {len(validation_results['warnings'])}

## Validation Results

### Index Validation
- **Type:** DatetimeIndex ✅
- **Timezone:** UTC ✅

### Column Validation
- **Expected Features:** {len(expected_features)}
- **Actual Features:** {len(actual_features)}
- **Missing Features:** {len(missing_features)}
- **Extra Features:** {len(extra_features)}

### Dtype Validation
- **Flag Features:** {'✅ PASS' if not dtype_errors else '❌ FAIL'}
- **Int Features:** {'✅ PASS' if not dtype_errors else '❌ FAIL'}

### NaN/Inf Validation
- **NaN Check:** {'✅ PASS' if not nan_inf_errors else '❌ FAIL'}
- **Inf Check:** {'✅ PASS' if not nan_inf_errors else '❌ FAIL'}

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
    
    md_content += f"""
## Metadata

- **sys.executable:** `{validation_results['sys_executable']}`
- **cwd:** `{validation_results['cwd']}`
- **git_sha:** `{validation_results['git_sha'] or 'N/A'}`
"""
    
    with open(md_path, "w") as f:
        f.write(md_content)
    
    if validation_results["passed"]:
        log.info("✅ Schema validation PASSED")
    else:
        log.error(f"❌ Schema validation FAILED: {len(validation_results['errors'])} errors")
        if fail_fast:
            raise RuntimeError(f"[SCHEMA_VALIDATION_FAIL] {len(validation_results['errors'])} errors found")
    
    return validation_results


def main():
    parser = argparse.ArgumentParser(description="Feature Schema Validator")
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
    
    validation_results = validate_feature_schema(
        prebuilt_parquet_path=args.prebuilt_parquet,
        manifest_json_path=args.manifest_json,
        output_dir=args.out_root,
        fail_fast=args.fail_fast,
    )
    
    log.info(f"✅ Validation complete: {args.out_root}")
    log.info(f"   Summary: {args.out_root / 'SCHEMA_VALIDATION_SUMMARY.json'}")
    log.info(f"   Report: {args.out_root / 'SCHEMA_VALIDATION_REPORT.md'}")


if __name__ == "__main__":
    main()
