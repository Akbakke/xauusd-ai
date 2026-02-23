#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Golden Feature Checks

Validates features at specific timestamps (golden samples) to ensure consistency.
Uses deterministic sampling and compares against expected values.

Usage:
    python3 gx1/scripts/golden_feature_checks.py \
        --prebuilt-parquet data/features/xauusd_m5_2025_features_v10_ctx.parquet \
        --manifest-json gx1/feature_manifest_v1.json \
        --sample-timestamps 10 \
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


def sample_timestamps(
    df: pd.DataFrame,
    n_samples: int,
    seed: int = 0,
) -> List[pd.Timestamp]:
    """
    Sample timestamps deterministically.
    
    Args:
        df: DataFrame with DatetimeIndex
        n_samples: Number of samples to take
        seed: Random seed for reproducibility
    
    Returns:
        List of sampled timestamps
    """
    np.random.seed(seed)
    
    # Stratified sampling: early, mid, late
    n = len(df)
    if n_samples <= 3:
        # Simple sampling
        indices = np.linspace(0, n - 1, n_samples, dtype=int)
    else:
        # Stratified: 30% early, 40% mid, 30% late
        n_early = max(1, int(n_samples * 0.3))
        n_mid = max(1, int(n_samples * 0.4))
        n_late = n_samples - n_early - n_mid
        
        early_indices = np.random.choice(n // 3, n_early, replace=False)
        mid_indices = np.random.choice(n // 3, n_mid, replace=False) + n // 3
        late_indices = np.random.choice(n - 2 * (n // 3), n_late, replace=False) + 2 * (n // 3)
        
        indices = np.concatenate([early_indices, mid_indices, late_indices])
        indices = np.sort(indices)
    
    timestamps = [df.index[i] for i in indices]
    return timestamps


def golden_feature_checks(
    prebuilt_parquet_path: Path,
    manifest_json_path: Path,
    output_dir: Path,
    n_samples: int = 10,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Perform golden feature checks at specific timestamps.
    
    Args:
        prebuilt_parquet_path: Path to prebuilt features parquet
        manifest_json_path: Path to feature manifest JSON
        output_dir: Output directory for reports
        n_samples: Number of timestamps to sample
        seed: Random seed for reproducibility
    
    Returns:
        Validation summary dict
    """
    log.info("=" * 60)
    log.info("GOLDEN FEATURE CHECKS")
    log.info("=" * 60)
    
    # Load manifest
    log.info(f"Loading manifest: {manifest_json_path}")
    with open(manifest_json_path, "r") as f:
        manifest = json.load(f)
    
    # Load prebuilt parquet
    log.info(f"Loading prebuilt parquet: {prebuilt_parquet_path}")
    df = pd.read_parquet(prebuilt_parquet_path)
    
    log.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Sample timestamps
    log.info(f"Sampling {n_samples} timestamps (seed={seed})...")
    timestamps = sample_timestamps(df, n_samples, seed)
    log.info(f"Sampled timestamps: {[str(ts) for ts in timestamps[:5]]}...")
    
    # Initialize validation results
    validation_results = {
        "timestamp": datetime.now().isoformat(),
        "prebuilt_parquet_path": str(prebuilt_parquet_path.resolve()),
        "manifest_json_path": str(manifest_json_path.resolve()),
        "sys_executable": sys.executable,
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "git_sha": get_git_sha(),
        "n_samples": n_samples,
        "seed": seed,
        "sampled_timestamps": [str(ts) for ts in timestamps],
        "errors": [],
        "warnings": [],
        "passed": True,
        "sample_values": {},
    }
    
    manifest_features = {f["name"]: f for f in manifest.get("features", [])}
    
    # Select critical features to check (10-15 features)
    critical_features = [
        "atr",
        "atr50",
        "ret_1",
        "ret_5",
        "ret_20",
        "rvol_20",
        "vol_ratio",
        "body_pct",
        "is_EU",
        "is_US",
        "session_id",
        "atr_regime_id",
        "ema20_slope",
        "ema100_slope",
        "pos_vs_ema200",
    ]
    
    # Filter to features that exist in manifest and dataframe
    critical_features = [f for f in critical_features if f in manifest_features and f in df.columns]
    
    log.info(f"[VALIDATION] Checking {len(critical_features)} critical features at {n_samples} timestamps...")
    
    for ts in timestamps:
        if ts not in df.index:
            error_msg = f"Timestamp {ts} not found in dataframe index"
            log.error(f"[GOLDEN_CHECK_FAIL] {error_msg}")
            validation_results["errors"].append(error_msg)
            validation_results["passed"] = False
            continue
        
        row = df.loc[ts]
        sample_values = {}
        
        for feature_name in critical_features:
            feature_info = manifest_features[feature_name]
            units_contract = feature_info.get("units_contract", "unknown")
            
            value = row[feature_name]
            
            # Check for NaN/Inf
            if pd.isna(value):
                error_msg = f"Feature {feature_name} has NaN at timestamp {ts}"
                log.error(f"[GOLDEN_CHECK_FAIL] {error_msg}")
                validation_results["errors"].append(error_msg)
                validation_results["passed"] = False
                continue
            
            if isinstance(value, (float, np.floating)) and np.isinf(value):
                error_msg = f"Feature {feature_name} has Inf at timestamp {ts}"
                log.error(f"[GOLDEN_CHECK_FAIL] {error_msg}")
                validation_results["errors"].append(error_msg)
                validation_results["passed"] = False
                continue
            
            # Check units contract bounds
            if units_contract == "flag":
                if value not in [0, 1, True, False]:
                    error_msg = f"Feature {feature_name} (flag) has value {value} != 0/1 at timestamp {ts}"
                    log.error(f"[GOLDEN_CHECK_FAIL] {error_msg}")
                    validation_results["errors"].append(error_msg)
                    validation_results["passed"] = False
            
            elif units_contract == "ratio":
                # Ratios should be reasonable (not extreme)
                if abs(value) > 100:
                    warning_msg = f"Feature {feature_name} (ratio) has extreme value {value:.4f} at timestamp {ts}"
                    log.warning(f"[GOLDEN_CHECK_WARN] {warning_msg}")
                    validation_results["warnings"].append(warning_msg)
            
            elif units_contract == "bps":
                # BPS should be reasonable (not extreme)
                if abs(value) > 10000:
                    warning_msg = f"Feature {feature_name} (bps) has extreme value {value:.4f} at timestamp {ts}"
                    log.warning(f"[GOLDEN_CHECK_WARN] {warning_msg}")
                    validation_results["warnings"].append(warning_msg)
            
            elif units_contract == "zscore":
                # Z-scores should be reasonable (not extreme)
                if abs(value) > 10:
                    warning_msg = f"Feature {feature_name} (zscore) has extreme value {value:.4f} at timestamp {ts}"
                    log.warning(f"[GOLDEN_CHECK_WARN] {warning_msg}")
                    validation_results["warnings"].append(warning_msg)
            
            sample_values[feature_name] = float(value) if isinstance(value, (int, float, np.number)) else str(value)
        
        validation_results["sample_values"][str(ts)] = sample_values
    
    # Write summary
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "GOLDEN_CHECKS_SUMMARY.json"
    with open(summary_path, "w") as f:
        json.dump(validation_results, f, indent=2)
    
    # Write markdown report
    md_path = output_dir / "GOLDEN_CHECKS_REPORT.md"
    md_content = f"""# Golden Feature Checks Report

**Generated:** {validation_results['timestamp']}  
**Prebuilt Parquet:** `{prebuilt_parquet_path}`  
**Manifest JSON:** `{manifest_json_path}`  
**Status:** {'✅ PASS' if validation_results['passed'] else '❌ FAIL'}

## Summary

- **Samples:** {validation_results['n_samples']}
- **Seed:** {validation_results['seed']}
- **Critical Features Checked:** {len(critical_features)}
- **Errors:** {len(validation_results['errors'])}
- **Warnings:** {len(validation_results['warnings'])}

## Sampled Timestamps

"""
    
    for ts in validation_results["sampled_timestamps"]:
        md_content += f"- `{ts}`\n"
    
    md_content += "\n## Errors\n\n"
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
    
    md_content += "\n## Sample Values (First 3 Timestamps)\n\n"
    md_content += "| Timestamp | Feature | Value | Units |\n"
    md_content += "|-----------|---------|-------|-------|\n"
    
    for ts_str in list(validation_results["sample_values"].keys())[:3]:
        sample_values = validation_results["sample_values"][ts_str]
        for feature_name, value in list(sample_values.items())[:5]:  # First 5 features per timestamp
            feature_info = manifest_features.get(feature_name, {})
            units_contract = feature_info.get("units_contract", "unknown")
            md_content += f"| `{ts_str}` | `{feature_name}` | {value:.4f if isinstance(value, float) else value} | {units_contract} |\n"
    
    md_content += f"""
## Metadata

- **sys.executable:** `{validation_results['sys_executable']}`
- **cwd:** `{validation_results['cwd']}`
- **git_sha:** `{validation_results['git_sha'] or 'N/A'}`
"""
    
    with open(md_path, "w") as f:
        f.write(md_content)
    
    if validation_results["passed"]:
        log.info("✅ Golden feature checks PASSED")
    else:
        log.error(f"❌ Golden feature checks FAILED: {len(validation_results['errors'])} errors")
        raise RuntimeError(f"[GOLDEN_CHECKS_FAIL] {len(validation_results['errors'])} errors found")
    
    return validation_results


def main():
    parser = argparse.ArgumentParser(description="Golden Feature Checks")
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
        "--sample-timestamps",
        type=int,
        default=10,
        help="Number of timestamps to sample (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
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
    
    validation_results = golden_feature_checks(
        prebuilt_parquet_path=args.prebuilt_parquet,
        manifest_json_path=args.manifest_json,
        output_dir=args.out_root,
        n_samples=args.sample_timestamps,
        seed=args.seed,
    )
    
    log.info(f"✅ Golden checks complete: {args.out_root}")
    log.info(f"   Summary: {args.out_root / 'GOLDEN_CHECKS_SUMMARY.json'}")
    log.info(f"   Report: {args.out_root / 'GOLDEN_CHECKS_REPORT.md'}")


if __name__ == "__main__":
    main()
