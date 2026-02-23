#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Leakage and Causality Validator

Validates features for lookahead leakage and causality by recomputing critical features
at specific timestamps and comparing against prebuilt values.

Uses "slow & clear" implementations locally (no runtime feature-building modules).

Usage:
    python3 gx1/scripts/validate_feature_leakage_and_causality.py \
        --prebuilt-parquet data/features/xauusd_m5_2025_features_v10_ctx.parquet \
        --candles-parquet data/oanda/years/2025.parquet \
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


def sample_timestamps(
    df: pd.DataFrame,
    n_samples: int,
    seed: int = 0,
) -> List[pd.Timestamp]:
    """Sample timestamps deterministically."""
    np.random.seed(seed)
    n = len(df)
    if n_samples <= 3:
        indices = np.linspace(0, n - 1, n_samples, dtype=int)
    else:
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


def compute_atr_simple(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute ATR using simple implementation (no lookahead)."""
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    # Convert to bps (assuming close is in price units)
    atr_bps = (atr / close) * 10000
    
    return atr_bps


def compute_return(close: np.ndarray, period: int = 1) -> np.ndarray:
    """Compute return (no lookahead)."""
    n = len(close)
    ret = np.zeros(n)
    ret[period:] = (close[period:] - close[:-period]) / close[:-period]
    return ret


def compute_session_flag(timestamp: pd.Timestamp) -> Tuple[bool, bool]:
    """Compute session flags (EU/US) from timestamp."""
    hour = timestamp.hour
    # Simple session detection (can be improved)
    is_eu = 7 <= hour < 16  # 7:00-16:00 UTC
    is_us = 13 <= hour < 22  # 13:00-22:00 UTC
    return is_eu, is_us


def recompute_critical_features(
    candles_df: pd.DataFrame,
    timestamp: pd.Timestamp,
    feature_name: str,
) -> Optional[float]:
    """
    Recompute a critical feature at a specific timestamp.
    
    Uses "slow & clear" implementations locally (no runtime feature-building modules).
    """
    if timestamp not in candles_df.index:
        return None
    
    idx = candles_df.index.get_loc(timestamp)
    
    # Get data up to and including timestamp (causal)
    df_slice = candles_df.iloc[:idx + 1]
    
    if len(df_slice) < 2:
        return None  # Not enough data
    
    high = df_slice["high"].values
    low = df_slice["low"].values
    close = df_slice["close"].values
    open_price = df_slice["open"].values
    
    # Recompute based on feature name
    if feature_name == "atr":
        atr_bps = compute_atr_simple(high, low, close, period=14)
        return float(atr_bps[-1]) if len(atr_bps) > 0 and not np.isnan(atr_bps[-1]) else None
    
    elif feature_name == "ret_1":
        ret = compute_return(close, period=1)
        return float(ret[-1]) if len(ret) > 0 and not np.isnan(ret[-1]) else None
    
    elif feature_name == "ret_5":
        ret = compute_return(close, period=5)
        return float(ret[-1]) if len(ret) > 0 and not np.isnan(ret[-1]) else None
    
    elif feature_name == "ret_20":
        ret = compute_return(close, period=20)
        return float(ret[-1]) if len(ret) > 0 and not np.isnan(ret[-1]) else None
    
    elif feature_name == "is_EU":
        is_eu, _ = compute_session_flag(timestamp)
        return 1.0 if is_eu else 0.0
    
    elif feature_name == "is_US":
        _, is_us = compute_session_flag(timestamp)
        return 1.0 if is_us else 0.0
    
    elif feature_name == "body_pct":
        if len(df_slice) < 1:
            return None
        body = abs(close[-1] - open_price[-1])
        range_price = high[-1] - low[-1]
        if range_price > 0:
            return float(body / range_price)
        return 0.0
    
    elif feature_name == "vol_ratio":
        if len(df_slice) < 20:
            return None
        # Simple volatility ratio: std of returns over last 20 bars
        ret_20 = compute_return(close, period=1)
        if len(ret_20) >= 20:
            vol_recent = np.std(ret_20[-20:])
            vol_long = np.std(ret_20[-60:]) if len(ret_20) >= 60 else vol_recent
            if vol_long > 0:
                return float(vol_recent / vol_long)
        return None
    
    # Add more feature recomputations as needed
    return None


def validate_leakage_and_causality(
    prebuilt_parquet_path: Path,
    candles_parquet_path: Optional[Path],
    manifest_json_path: Path,
    output_dir: Path,
    n_samples: int = 10,
    seed: int = 0,
    tolerance: float = 1e-3,
) -> Dict[str, Any]:
    """
    Validate feature leakage and causality.
    
    Args:
        prebuilt_parquet_path: Path to prebuilt features parquet
        candles_parquet_path: Path to candles parquet (optional, for recomputation)
        manifest_json_path: Path to feature manifest JSON
        output_dir: Output directory for reports
        n_samples: Number of timestamps to sample
        seed: Random seed for reproducibility
        tolerance: Tolerance for value comparison
    
    Returns:
        Validation summary dict
    """
    log.info("=" * 60)
    log.info("FEATURE LEAKAGE AND CAUSALITY VALIDATOR")
    log.info("=" * 60)
    
    # Load manifest
    log.info(f"Loading manifest: {manifest_json_path}")
    with open(manifest_json_path, "r") as f:
        manifest = json.load(f)
    
    # Load prebuilt parquet
    log.info(f"Loading prebuilt parquet: {prebuilt_parquet_path}")
    df_prebuilt = pd.read_parquet(prebuilt_parquet_path)
    
    log.info(f"Loaded {len(df_prebuilt):,} rows, {len(df_prebuilt.columns)} columns")
    
    # Load candles if provided
    df_candles = None
    if candles_parquet_path and candles_parquet_path.exists():
        log.info(f"Loading candles parquet: {candles_parquet_path}")
        df_candles = pd.read_parquet(candles_parquet_path)
        log.info(f"Loaded {len(df_candles):,} candle rows")
    else:
        log.warning("No candles parquet provided, skipping recomputation checks")
    
    # Sample timestamps
    log.info(f"Sampling {n_samples} timestamps (seed={seed})...")
    timestamps = sample_timestamps(df_prebuilt, n_samples, seed)
    log.info(f"Sampled timestamps: {[str(ts) for ts in timestamps[:5]]}...")
    
    # Initialize validation results
    validation_results = {
        "timestamp": datetime.now().isoformat(),
        "prebuilt_parquet_path": str(prebuilt_parquet_path.resolve()),
        "candles_parquet_path": str(candles_parquet_path.resolve()) if candles_parquet_path else None,
        "manifest_json_path": str(manifest_json_path.resolve()),
        "sys_executable": sys.executable,
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "git_sha": get_git_sha(),
        "n_samples": n_samples,
        "seed": seed,
        "tolerance": tolerance,
        "sampled_timestamps": [str(ts) for ts in timestamps],
        "errors": [],
        "warnings": [],
        "passed": True,
        "recomputation_results": {},
    }
    
    manifest_features = {f["name"]: f for f in manifest.get("features", [])}
    
    # Select critical features to check (10-15 features)
    critical_features = [
        "atr",
        "ret_1",
        "ret_5",
        "ret_20",
        "vol_ratio",
        "body_pct",
        "is_EU",
        "is_US",
    ]
    
    # Filter to features that exist in manifest and dataframe
    critical_features = [f for f in critical_features if f in manifest_features and f in df_prebuilt.columns]
    
    log.info(f"[VALIDATION] Checking {len(critical_features)} critical features at {n_samples} timestamps...")
    
    if df_candles is None:
        log.warning("Skipping recomputation checks (no candles parquet)")
        validation_results["warnings"].append("No candles parquet provided, skipping recomputation checks")
    else:
        # Ensure timezone alignment
        if df_candles.index.tz != df_prebuilt.index.tz:
            log.warning(f"Timezone mismatch: candles={df_candles.index.tz}, prebuilt={df_prebuilt.index.tz}")
            if df_candles.index.tz is None:
                df_candles.index = df_candles.index.tz_localize("UTC")
            elif df_prebuilt.index.tz is None:
                df_prebuilt.index = df_prebuilt.index.tz_localize("UTC")
        
        for ts in timestamps:
            if ts not in df_prebuilt.index:
                error_msg = f"Timestamp {ts} not found in prebuilt dataframe"
                log.error(f"[LEAKAGE_CHECK_FAIL] {error_msg}")
                validation_results["errors"].append(error_msg)
                validation_results["passed"] = False
                continue
            
            if ts not in df_candles.index:
                warning_msg = f"Timestamp {ts} not found in candles dataframe"
                log.warning(f"[LEAKAGE_CHECK_WARN] {warning_msg}")
                validation_results["warnings"].append(warning_msg)
                continue
            
            recomputation_results = {}
            
            for feature_name in critical_features:
                feature_info = manifest_features[feature_name]
                
                # Get prebuilt value
                prebuilt_value = df_prebuilt.loc[ts, feature_name]
                
                if pd.isna(prebuilt_value):
                    continue  # Skip NaN values
                
                # Recompute value
                recomputed_value = recompute_critical_features(df_candles, ts, feature_name)
                
                if recomputed_value is None:
                    continue  # Skip if recomputation failed
                
                # Compare values
                diff = abs(prebuilt_value - recomputed_value)
                relative_diff = diff / abs(prebuilt_value) if abs(prebuilt_value) > 1e-10 else diff
                
                recomputation_results[feature_name] = {
                    "prebuilt": float(prebuilt_value),
                    "recomputed": float(recomputed_value),
                    "diff": float(diff),
                    "relative_diff": float(relative_diff),
                }
                
                # Check tolerance
                if relative_diff > tolerance:
                    error_msg = (
                        f"Feature {feature_name} mismatch at {ts}: "
                        f"prebuilt={prebuilt_value:.6f}, recomputed={recomputed_value:.6f}, "
                        f"relative_diff={relative_diff:.6f} > tolerance={tolerance}"
                    )
                    log.error(f"[LEAKAGE_CHECK_FAIL] {error_msg}")
                    validation_results["errors"].append(error_msg)
                    validation_results["passed"] = False
            
            validation_results["recomputation_results"][str(ts)] = recomputation_results
    
    # Write summary
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "LEAKAGE_CAUSALITY_VALIDATION_SUMMARY.json"
    with open(summary_path, "w") as f:
        json.dump(validation_results, f, indent=2)
    
    # Write markdown report
    md_path = output_dir / "LEAKAGE_CAUSALITY_VALIDATION_REPORT.md"
    md_content = f"""# Feature Leakage and Causality Validation Report

**Generated:** {validation_results['timestamp']}  
**Prebuilt Parquet:** `{prebuilt_parquet_path}`  
**Candles Parquet:** `{candles_parquet_path or 'N/A'}`  
**Manifest JSON:** `{manifest_json_path}`  
**Status:** {'✅ PASS' if validation_results['passed'] else '❌ FAIL'}

## Summary

- **Samples:** {validation_results['n_samples']}
- **Seed:** {validation_results['seed']}
- **Tolerance:** {validation_results['tolerance']}
- **Critical Features Checked:** {len(critical_features)}
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
    
    if validation_results["recomputation_results"]:
        md_content += "\n## Recomputation Results (First 3 Timestamps)\n\n"
        md_content += "| Timestamp | Feature | Prebuilt | Recomputed | Diff | Relative Diff |\n"
        md_content += "|-----------|---------|----------|------------|------|---------------|\n"
        
        for ts_str in list(validation_results["recomputation_results"].keys())[:3]:
            recomputation_results = validation_results["recomputation_results"][ts_str]
            for feature_name, result in list(recomputation_results.items())[:5]:
                md_content += f"| `{ts_str}` | `{feature_name}` | {result['prebuilt']:.6f} | {result['recomputed']:.6f} | {result['diff']:.6f} | {result['relative_diff']:.6f} |\n"
    
    md_content += f"""
## Metadata

- **sys.executable:** `{validation_results['sys_executable']}`
- **cwd:** `{validation_results['cwd']}`
- **git_sha:** `{validation_results['git_sha'] or 'N/A'}`
"""
    
    with open(md_path, "w") as f:
        f.write(md_content)
    
    if validation_results["passed"]:
        log.info("✅ Leakage and causality validation PASSED")
    else:
        log.error(f"❌ Leakage and causality validation FAILED: {len(validation_results['errors'])} errors")
        raise RuntimeError(f"[LEAKAGE_CAUSALITY_VALIDATION_FAIL] {len(validation_results['errors'])} errors found")
    
    return validation_results


def main():
    parser = argparse.ArgumentParser(description="Feature Leakage and Causality Validator")
    parser.add_argument(
        "--prebuilt-parquet",
        type=Path,
        required=True,
        help="Path to prebuilt features parquet",
    )
    parser.add_argument(
        "--candles-parquet",
        type=Path,
        default=None,
        help="Path to candles parquet (optional, for recomputation)",
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
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Tolerance for value comparison (default: 1e-3)",
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
    
    validation_results = validate_leakage_and_causality(
        prebuilt_parquet_path=args.prebuilt_parquet,
        candles_parquet_path=args.candles_parquet,
        manifest_json_path=args.manifest_json,
        output_dir=args.out_root,
        n_samples=args.sample_timestamps,
        seed=args.seed,
        tolerance=args.tolerance,
    )
    
    log.info(f"✅ Validation complete: {args.out_root}")
    log.info(f"   Summary: {args.out_root / 'LEAKAGE_CAUSALITY_VALIDATION_SUMMARY.json'}")
    log.info(f"   Report: {args.out_root / 'LEAKAGE_CAUSALITY_VALIDATION_REPORT.md'}")


if __name__ == "__main__":
    main()
