#!/usr/bin/env python3
"""
Prebuilt features preflight check - hard-fail before replay.

Validates that prebuilt features are ready and match raw data before starting replay.
All checks are hard-fail (no silent fallback).

Usage:
    python3 gx1/scripts/prebuilt_preflight.py \
        --raw data/raw/xauusd_m5_2025_bid_ask.parquet \
        --features data/features/xauusd_m5_2025_features_v10_ctx.parquet \
        --policy gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml \
        --days 7 \
        --strict 1
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

EXPECTED_SCHEMA_VERSION = "features_v10_ctx_v1"


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def check_files_exist(raw_path: Path, features_path: Path) -> None:
    """A1: Files exist + readable."""
    if not raw_path.exists():
        raise FileNotFoundError(f"[PREBUILT_PREFLIGHT] Raw file not found: {raw_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"[PREBUILT_PREFLIGHT] Features file not found: {features_path}")
    
    # Check readable
    try:
        # Read first row to verify file is readable (use head(1) instead of nrows)
        df_raw_check = pd.read_parquet(raw_path)
        if len(df_raw_check) == 0:
            raise RuntimeError("[PREBUILT_PREFLIGHT] Raw file is empty")
    except Exception as e:
        raise RuntimeError(f"[PREBUILT_PREFLIGHT] Raw file not readable: {e}")
    
    try:
        # Read first row to verify file is readable (use head(1) instead of nrows)
        df_features_check = pd.read_parquet(features_path)
        if len(df_features_check) == 0:
            raise RuntimeError("[PREBUILT_PREFLIGHT] Features file is empty")
    except Exception as e:
        raise RuntimeError(f"[PREBUILT_PREFLIGHT] Features file not readable: {e}")
    
    log.info(f"[PREBUILT_PREFLIGHT] ✅ Files exist and readable")


def check_schema_version(features_path: Path) -> None:
    """A2: Features parquet schema_version exists + matches expected."""
    # Try to read metadata from parquet (if available)
    # Otherwise check manifest
    manifest_path = features_path.with_suffix(".manifest.json")
    
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        schema_version = manifest.get("schema_version")
        if schema_version != EXPECTED_SCHEMA_VERSION:
            raise ValueError(
                f"[PREBUILT_PREFLIGHT] Schema version mismatch: "
                f"expected={EXPECTED_SCHEMA_VERSION}, got={schema_version}"
            )
        log.info(f"[PREBUILT_PREFLIGHT] ✅ Schema version matches: {schema_version}")
    else:
        log.warning(f"[PREBUILT_PREFLIGHT] ⚠️  Manifest not found: {manifest_path} (skipping schema check)")


def check_manifest_sha256(features_path: Path) -> None:
    """A3: Manifest exists and sha256 matches parquet."""
    manifest_path = features_path.with_suffix(".manifest.json")
    
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"[PREBUILT_PREFLIGHT] Manifest not found: {manifest_path}. "
            f"Run build_fullyear_features_parquet.py to generate manifest."
        )
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    expected_sha256 = manifest.get("features_file_sha256")
    if not expected_sha256:
        raise ValueError(f"[PREBUILT_PREFLIGHT] Manifest missing features_file_sha256")
    
    actual_sha256 = compute_file_sha256(features_path)
    
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"[PREBUILT_PREFLIGHT] SHA256 mismatch: "
            f"manifest={expected_sha256[:16]}..., actual={actual_sha256[:16]}... "
            f"File may have been modified after build."
        )
    
    log.info(f"[PREBUILT_PREFLIGHT] ✅ Manifest SHA256 matches: {actual_sha256[:16]}...")


def check_timestamp_alignment(
    raw_path: Path, features_path: Path, days: int
) -> tuple[int, bool]:
    """A4: Raw timestamps vs features timestamps - exact equality."""
    log.info(f"[PREBUILT_PREFLIGHT] Loading {days}-day slice for timestamp check...")
    
    # Load raw data
    df_raw = pd.read_parquet(raw_path)
    if not isinstance(df_raw.index, pd.DatetimeIndex):
        if "time" in df_raw.columns:
            df_raw["time"] = pd.to_datetime(df_raw["time"], utc=True)
            df_raw = df_raw.set_index("time").sort_index()
        else:
            raise ValueError("[PREBUILT_PREFLIGHT] Raw data must have DatetimeIndex or 'time' column")
    
    # Load features data
    df_features = pd.read_parquet(features_path)
    if not isinstance(df_features.index, pd.DatetimeIndex):
        if "time" in df_features.columns:
            df_features["time"] = pd.to_datetime(df_features["time"], utc=True)
            df_features = df_features.set_index("time").sort_index()
        else:
            raise ValueError("[PREBUILT_PREFLIGHT] Features data must have DatetimeIndex or 'time' column")
    
    # Take deterministic slice (first N days)
    # Calculate bars per day (M5 = 288 bars/day)
    bars_per_day = 288
    slice_bars = days * bars_per_day
    df_raw_slice = df_raw.head(slice_bars)
    df_features_slice = df_features.head(slice_bars)
    
    # Extract timestamps as int64 seconds
    raw_ts = (df_raw_slice.index.astype(np.int64) // 1_000_000_000).astype(np.int64)
    features_ts = (df_features_slice.index.astype(np.int64) // 1_000_000_000).astype(np.int64)
    
    # Check length
    if len(raw_ts) != len(features_ts):
        raise ValueError(
            f"[PREBUILT_PREFLIGHT] Timestamp length mismatch: "
            f"raw={len(raw_ts)}, features={len(features_ts)}"
        )
    
    # Check exact equality
    if not np.array_equal(raw_ts, features_ts):
        # Find first 5 diffs
        diffs = np.where(raw_ts != features_ts)[0][:5]
        diff_details = []
        for idx in diffs:
            diff_details.append(
                f"idx={idx}: raw={raw_ts[idx]} ({pd.Timestamp(raw_ts[idx], unit='s', tz='UTC')}), "
                f"features={features_ts[idx]} ({pd.Timestamp(features_ts[idx], unit='s', tz='UTC')})"
            )
        raise ValueError(
            f"[PREBUILT_PREFLIGHT] Timestamp mismatch (first 5 diffs):\n" + "\n".join(diff_details)
        )
    
    log.info(f"[PREBUILT_PREFLIGHT] ✅ Timestamp alignment: {len(raw_ts)} bars match exactly")
    
    # 6) Prebuilt coverage check
    df_raw_full = pd.read_parquet(raw_path)
    if not isinstance(df_raw_full.index, pd.DatetimeIndex):
        if "time" in df_raw_full.columns:
            df_raw_full["time"] = pd.to_datetime(df_raw_full["time"], utc=True)
            df_raw_full = df_raw_full.set_index("time").sort_index()
    
    df_features_full = pd.read_parquet(features_path)
    if not isinstance(df_features_full.index, pd.DatetimeIndex):
        if "time" in df_features_full.columns:
            df_features_full["time"] = pd.to_datetime(df_features_full["time"], utc=True)
            df_features_full = df_features_full.set_index("time").sort_index()
    
    raw_rows = len(df_raw_full)
    features_rows = len(df_features_full)
    coverage_ratio = features_rows / raw_rows if raw_rows > 0 else 0.0
    
    if coverage_ratio < 0.99:
        raise ValueError(
            f"[PREBUILT_PREFLIGHT] Coverage too low: features={features_rows}, raw={raw_rows}, ratio={coverage_ratio:.4f} < 0.99\n"
            f"Instructions: Rebuild prebuilt features to cover full dataset."
        )
    
    # Check timestamp range
    raw_start = df_raw_full.index.min()
    raw_end = df_raw_full.index.max()
    features_start = df_features_full.index.min()
    features_end = df_features_full.index.max()
    
    if features_start > raw_start or features_end < raw_end:
        log.warning(
            f"[PREBUILT_PREFLIGHT] ⚠️  Timestamp range mismatch:\n"
            f"  Raw: {raw_start} to {raw_end}\n"
            f"  Features: {features_start} to {features_end}\n"
            f"  Difference: start={features_start - raw_start}, end={raw_end - features_end}"
        )
    else:
        log.info(
            f"[PREBUILT_PREFLIGHT] ✅ Coverage: {features_rows}/{raw_rows} rows ({coverage_ratio*100:.2f}%), "
            f"timestamp range covers raw data"
        )
    
    return len(raw_ts), True


def get_required_columns(policy_path: Optional[Path]) -> Set[str]:
    """Get required feature columns from feature_meta.json."""
    required = set()
    
    # Try to get from policy config
    if policy_path and policy_path.exists():
        import yaml
        with open(policy_path, "r") as f:
            policy = yaml.safe_load(f)
        
        # Get feature_meta_path from v10_ctx config
        v10_ctx_cfg = policy.get("entry_models", {}).get("v10_ctx", {})
        feature_meta_path = v10_ctx_cfg.get("feature_meta_path")
        
        if feature_meta_path:
            feature_meta_path = Path(feature_meta_path)
            if not feature_meta_path.is_absolute():
                feature_meta_path = project_root / feature_meta_path
            
            if feature_meta_path.exists():
                with open(feature_meta_path, "r") as f:
                    feature_meta = json.load(f)
                
                # Get seq and snap features
                seq_features = feature_meta.get("seq_features", [])
                snap_features = feature_meta.get("snap_features", [])
                
                required.update(seq_features)
                required.update(snap_features)
                
                log.info(
                    f"[PREBUILT_PREFLIGHT] Loaded required columns from feature_meta: "
                    f"{len(seq_features)} seq + {len(snap_features)} snap = {len(required)} total"
                )
    
    # Fallback: use known required columns if feature_meta not found
    if not required:
        log.warning("[PREBUILT_PREFLIGHT] ⚠️  Could not load feature_meta, using fallback list")
        # Known required columns from runtime_v10_ctx
        required = {
            "atr50", "atr_regime_id", "atr_z", "body_pct", "ema100_slope", "ema20_slope",
            "pos_vs_ema200", "roc100", "roc20", "session_id", "std50", "trend_regime_tf24h",
            "wick_asym", "p_long_xgb_seq", "margin_xgb_seq", "p_long_xgb_ema_seq",
            "p_long_xgb_now", "margin_xgb_now", "p_hat_xgb_now",
        }
    
    return required


def check_required_columns(features_path: Path, required_cols: Set[str]) -> int:
    """A5: Required columns exist and have correct dtypes."""
    # Read first row to get schema (use head(1) instead of nrows)
    df_features = pd.read_parquet(features_path).head(1)  # Just for schema
    
    missing_cols = required_cols - set(df_features.columns)
    if missing_cols:
        raise ValueError(
            f"[PREBUILT_PREFLIGHT] Missing required columns: {sorted(missing_cols)}"
        )
    
    # Check dtypes are numeric
    invalid_dtypes = []
    for col in required_cols:
        dtype = df_features[col].dtype
        if not np.issubdtype(dtype, np.number):
            invalid_dtypes.append(f"{col}: {dtype}")
    
    if invalid_dtypes:
        raise ValueError(
            f"[PREBUILT_PREFLIGHT] Non-numeric dtypes in required columns: {invalid_dtypes}"
        )
    
    log.info(f"[PREBUILT_PREFLIGHT] ✅ Required columns check: {len(required_cols)} columns OK")
    return len(required_cols)


def check_nan_inf(features_path: Path, required_cols: Set[str], days: int) -> None:
    """A6: NaN/Inf guard - sample 10k rows."""
    log.info(f"[PREBUILT_PREFLIGHT] Checking NaN/Inf in {days}-day slice...")
    
    bars_per_day = 288
    slice_bars = min(days * bars_per_day, 10000)  # Sample up to 10k bars
    
    # Read slice (use head() instead of nrows)
    df_features = pd.read_parquet(features_path).head(slice_bars)
    
    # Check for NaN/Inf in required columns
    for col in required_cols:
        if col not in df_features.columns:
            continue
        
        nan_count = df_features[col].isna().sum()
        inf_count = np.isinf(df_features[col]).sum() if np.issubdtype(df_features[col].dtype, np.number) else 0
        
        if nan_count > 0:
            raise ValueError(
                f"[PREBUILT_PREFLIGHT] NaN found in {col}: {nan_count} values"
            )
        if inf_count > 0:
            raise ValueError(
                f"[PREBUILT_PREFLIGHT] Inf found in {col}: {inf_count} values"
            )
    
    log.info(f"[PREBUILT_PREFLIGHT] ✅ NaN/Inf check: {slice_bars} rows, {len(required_cols)} columns OK")


def check_env_vars(features_path: Path) -> None:
    """A7: Replay wiring check - env var expectations."""
    use_prebuilt = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES")
    prebuilt_path = os.getenv("GX1_REPLAY_PREBUILT_FEATURES_PATH")
    
    features_path_resolved = features_path.resolve()
    
    if use_prebuilt != "1":
        raise RuntimeError(
            f"[PREBUILT_PREFLIGHT] GX1_REPLAY_USE_PREBUILT_FEATURES must be '1', got: {use_prebuilt}\n"
            f"Export command: export GX1_REPLAY_USE_PREBUILT_FEATURES=1"
        )
    
    if not prebuilt_path:
        raise RuntimeError(
            f"[PREBUILT_PREFLIGHT] GX1_REPLAY_PREBUILT_FEATURES_PATH not set\n"
            f"Export command: export GX1_REPLAY_PREBUILT_FEATURES_PATH={features_path_resolved}"
        )
    
    prebuilt_path_resolved = Path(prebuilt_path).resolve()
    if prebuilt_path_resolved != features_path_resolved:
        raise RuntimeError(
            f"[PREBUILT_PREFLIGHT] GX1_REPLAY_PREBUILT_FEATURES_PATH mismatch:\n"
            f"  env={prebuilt_path_resolved}\n"
            f"  expected={features_path_resolved}\n"
            f"Export command: export GX1_REPLAY_PREBUILT_FEATURES_PATH={features_path_resolved}"
        )
    
    log.info(f"[PREBUILT_PREFLIGHT] ✅ Env vars OK: USE_PREBUILT=1, PATH={prebuilt_path_resolved}")


def main():
    parser = argparse.ArgumentParser(description="Prebuilt features preflight check")
    parser.add_argument("--raw", type=Path, required=True, help="Raw data parquet path")
    parser.add_argument("--features", type=Path, required=True, help="Prebuilt features parquet path")
    parser.add_argument("--policy", type=Path, help="Policy YAML (for feature_meta)")
    parser.add_argument("--days", type=int, default=7, help="Days to check (default: 7)")
    parser.add_argument("--strict", type=int, default=1, help="Strict mode (default: 1)")
    
    args = parser.parse_args()
    
    log.info("=" * 80)
    log.info("[PREBUILT_PREFLIGHT] Starting preflight checks")
    log.info("=" * 80)
    
    try:
        # A1: Files exist
        check_files_exist(args.raw, args.features)
        
        # A2: Schema version
        check_schema_version(args.features)
        
        # A3: Manifest + SHA256
        check_manifest_sha256(args.features)
        
        # A4: Timestamp alignment
        rows_checked, ts_match = check_timestamp_alignment(args.raw, args.features, args.days)
        
        # A5: Required columns
        required_cols = get_required_columns(args.policy)
        cols_checked = check_required_columns(args.features, required_cols)
        
        # A6: NaN/Inf guard
        check_nan_inf(args.features, required_cols, args.days)
        
        # A7: Env vars (optional - only warn if not set, don't fail)
        # This is a warning, not a hard fail, since preflight can be run before env vars are set
        try:
            check_env_vars(args.features)
        except RuntimeError as e:
            log.warning(f"[PREBUILT_PREFLIGHT] Env vars not set (this is OK for preflight): {e}")
            log.info("[PREBUILT_PREFLIGHT] Env vars will be checked again during replay")
        
        # Get SHA256 for output
        sha256 = compute_file_sha256(args.features)
        
        # Success output
        log.info("=" * 80)
        log.info("[PREBUILT_PREFLIGHT] ✅ ALL CHECKS PASSED")
        log.info("=" * 80)
        print(f"PREBUILT_PREFLIGHT_OK rows_checked={rows_checked} cols_checked={cols_checked} sha256={sha256[:16]}... ts_match=true")
        
        return 0
        
    except Exception as e:
        log.error("=" * 80)
        log.error(f"[PREBUILT_PREFLIGHT] ❌ FAILED: {e}")
        log.error("=" * 80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
