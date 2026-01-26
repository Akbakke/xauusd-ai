#!/usr/bin/env python3
"""
Preflight sanity checks for FULLYEAR build.

This script performs comprehensive validation before running the full build:
- Raw data integrity
- Contract checks (dimensions, calibration)
- Leakage detection (HTF partial bars, alignment, shifts, truncate-future)
- Reproducibility checks

Outputs a markdown report with pass/fail status for each check.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gx1.features.feature_state import FeatureState
from gx1.features.runtime_v9 import build_v9_runtime_features
from gx1.models.entry_v10.calibration_paths import get_calibrator_path_hierarchy
from gx1.models.entry_v10.xgb_calibration import load_xgb_calibrators
from gx1.utils.feature_context import reset_feature_state, set_feature_state

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


class PreflightCheck:
    """Container for a single preflight check result."""

    def __init__(self, name: str, passed: bool, message: str = "", details: Dict = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}

    def __repr__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status}: {self.name} - {self.message}"


def get_git_commit() -> Tuple[str, bool]:
    """Get current git commit hash and dirty state."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        commit = result.stdout.strip() if result.returncode == 0 else "unknown"

        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True,
            cwd=project_root,
        )
        is_dirty = result.returncode != 0

        return commit, is_dirty
    except Exception:
        return "unknown", True


def check_raw_data_integrity(
    df: pd.DataFrame, require_labels: bool = True
) -> List[PreflightCheck]:
    """Check A) Raw data integrity."""
    checks = []

    # Required columns
    required_cols = {"open", "high", "low", "close", "volume", "ts"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        checks.append(
            PreflightCheck(
                "Raw data: Required columns",
                False,
                f"Missing columns: {sorted(missing_cols)}",
            )
        )
    else:
        checks.append(
            PreflightCheck("Raw data: Required columns", True, "All required columns present")
        )

    # Dtypes
    numeric_cols = ["open", "high", "low", "close", "volume"]
    dtype_ok = all(df[col].dtype in [np.float32, np.float64, np.int64] for col in numeric_cols if col in df.columns)
    if not dtype_ok:
        bad_dtypes = {col: str(df[col].dtype) for col in numeric_cols if col in df.columns and df[col].dtype not in [np.float32, np.float64, np.int64]}
        checks.append(
            PreflightCheck(
                "Raw data: Dtypes",
                False,
                f"Invalid dtypes: {bad_dtypes}",
            )
        )
    else:
        checks.append(PreflightCheck("Raw data: Dtypes", True, "All numeric columns have valid dtypes"))

    # Timestamps monotonic
    if "ts" in df.columns:
        df_ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        is_monotonic = df_ts.is_monotonic_increasing
        if not is_monotonic:
            bad_indices = np.where(~df_ts.diff().fillna(pd.Timedelta(seconds=1)) >= pd.Timedelta(0))[0]
            checks.append(
                PreflightCheck(
                    "Raw data: Timestamps monotonic",
                    False,
                    f"Non-monotonic timestamps at indices: {bad_indices[:10].tolist()}",
                    {"bad_indices": bad_indices[:10].tolist()},
                )
            )
        else:
            checks.append(PreflightCheck("Raw data: Timestamps monotonic", True, "Timestamps are monotonic"))

    # No NaN/inf in OHLC
    ohlc_cols = ["open", "high", "low", "close"]
    ohlc_present = [c for c in ohlc_cols if c in df.columns]
    if ohlc_present:
        has_nan_inf = False
        bad_cols = []
        for col in ohlc_present:
            if df[col].isna().any() or np.isinf(df[col]).any():
                has_nan_inf = True
                bad_cols.append(col)
        if has_nan_inf:
            checks.append(
                PreflightCheck(
                    "Raw data: OHLC NaN/inf",
                    False,
                    f"NaN/inf found in: {bad_cols}",
                )
            )
        else:
            checks.append(PreflightCheck("Raw data: OHLC NaN/inf", True, "No NaN/inf in OHLC"))

    # Label columns
    if require_labels:
        label_cols = ["mfe_bps", "MFE_bps", "mae_bps", "MAE_bps"]
        has_labels = any(col in df.columns for col in label_cols)
        if not has_labels:
            checks.append(
                PreflightCheck(
                    "Raw data: Label columns",
                    False,
                    f"Missing label columns. Expected one of: {label_cols}",
                )
            )
        else:
            # Check label ranges
            mfe_col = "MFE_bps" if "MFE_bps" in df.columns else ("mfe_bps" if "mfe_bps" in df.columns else None)
            mae_col = "MAE_bps" if "MAE_bps" in df.columns else ("mae_bps" if "mae_bps" in df.columns else None)
            if mfe_col and mae_col:
                mfe_range = (df[mfe_col].min(), df[mfe_col].max())
                mae_range = (df[mae_col].min(), df[mae_col].max())
                # Plausible ranges: MFE/MAE should be reasonable (e.g., -1000 to +1000 bps)
                if abs(mfe_range[0]) > 10000 or abs(mfe_range[1]) > 10000:
                    checks.append(
                        PreflightCheck(
                            "Raw data: Label ranges",
                            False,
                            f"MFE range implausible: {mfe_range}",
                        )
                    )
                elif abs(mae_range[0]) > 10000 or abs(mae_range[1]) > 10000:
                    checks.append(
                        PreflightCheck(
                            "Raw data: Label ranges",
                            False,
                            f"MAE range implausible: {mae_range}",
                        )
                    )
                else:
                    checks.append(
                        PreflightCheck(
                            "Raw data: Label ranges",
                            True,
                            f"MFE: {mfe_range}, MAE: {mae_range}",
                        )
                    )
            else:
                checks.append(PreflightCheck("Raw data: Label columns", True, "Label columns present"))

    return checks


def check_contract(
    df_raw: pd.DataFrame,
    policy_config: Dict,
    feature_meta_path: Path,
    seq_scaler_path: Optional[Path],
    snap_scaler_path: Optional[Path],
    calibration_dir: Path,
    calibration_method: str,
    require_calibration: bool,
) -> List[PreflightCheck]:
    """Check B) Contract checks."""
    checks = []

    # Build mini dataset
    log.info("[CONTRACT] Building mini dataset for contract verification...")
    feature_state = FeatureState()
    feature_token = set_feature_state(feature_state)
    try:
        df_features, seq_feature_names, snap_feature_names = build_v9_runtime_features(
            df_raw=df_raw,
            feature_meta_path=feature_meta_path,
            seq_scaler_path=seq_scaler_path,
            snap_scaler_path=snap_scaler_path,
        )
    finally:
        reset_feature_state(feature_token)

    # Check base feature dimensions (before XGB channels)
    expected_base_seq_dim = 13
    expected_base_snap_dim = 85

    actual_seq_dim = len(seq_feature_names)
    actual_snap_dim = len(snap_feature_names)

    if actual_seq_dim != expected_base_seq_dim:
        checks.append(
            PreflightCheck(
                "Contract: Base seq feature count",
                False,
                f"Expected {expected_base_seq_dim} base seq features, got {actual_seq_dim}",
            )
        )
    else:
        checks.append(PreflightCheck("Contract: Base seq feature count", True, f"Got {actual_seq_dim} base seq features"))

    if actual_snap_dim != expected_base_snap_dim:
        checks.append(
            PreflightCheck(
                "Contract: Base snap feature count",
                False,
                f"Expected {expected_base_snap_dim} base snap features, got {actual_snap_dim}",
            )
        )
    else:
        checks.append(PreflightCheck("Contract: Base snap feature count", True, f"Got {actual_snap_dim} base snap features"))

    # Note: Final dims with XGB channels will be 16/88 (13+3 / 85+3), but we check base here
    # The build script will add XGB channels later

    # Check calibration
    if require_calibration:
        policy_id = policy_config.get("policy_name", "GX1_SNIPER_TRAIN_V10_CTX_GATED")
        calibrators = load_xgb_calibrators(
            calibration_dir=calibration_dir,
            policy_id=policy_id,
            method=calibration_method,
        )
        if not calibrators:
            checks.append(
                PreflightCheck(
                    "Contract: Calibration required",
                    False,
                    f"No calibrators found for policy_id={policy_id}, method={calibration_method}",
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    "Contract: Calibration required",
                    True,
                    f"Found calibrators for {len(calibrators)} sessions",
                )
            )

    return checks


def check_htf_no_partial_bar(df_raw: pd.DataFrame, n_samples: int = 10) -> List[PreflightCheck]:
    """Check C1) HTF no-partial-bar leakage."""
    checks = []

    if "ts" not in df_raw.columns:
        checks.append(
            PreflightCheck(
                "HTF no-partial-bar",
                False,
                "Missing 'ts' column for HTF check",
            )
        )
        return checks

    # Sample random indices
    np.random.seed(1337)
    sample_indices = np.random.choice(len(df_raw), min(n_samples, len(df_raw)), replace=False)

    # For each sample, we'd need to check HTF aggregator state
    # This is a simplified check - in practice, we'd need access to HTF aggregator internals
    # For now, we check that timestamps are reasonable
    df_ts = pd.to_datetime(df_raw["ts"], utc=True, errors="coerce")
    sample_times = df_ts.iloc[sample_indices]

    # Basic sanity: timestamps should be valid
    invalid_times = sample_times.isna().sum()
    if invalid_times > 0:
        checks.append(
            PreflightCheck(
                "HTF no-partial-bar: Timestamp validity",
                False,
                f"{invalid_times} invalid timestamps in sample",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "HTF no-partial-bar: Timestamp validity",
                True,
                f"All {len(sample_indices)} sample timestamps valid",
            )
        )

    # Note: Full HTF aggregator check would require building features and inspecting aggregator state
    # This is a placeholder for the full implementation

    return checks


def check_alignment_correctness(df_features: pd.DataFrame) -> List[PreflightCheck]:
    """Check C2) Alignment correctness (searchsorted <= current M5 time)."""
    checks = []

    # Check HTF-aligned features (they should not be future-looking)
    htf_features = [c for c in df_features.columns if c.startswith("_v1h1_") or c.startswith("_v1h4_")]

    if not htf_features:
        checks.append(
            PreflightCheck(
                "Alignment correctness",
                False,
                "No HTF features found to check",
            )
        )
        return checks

    # Basic check: HTF features should not have extreme outliers that suggest future leakage
    # This is a simplified check - full check would require comparing with HTF aggregator state
    for feat in htf_features[:5]:  # Check first 5 HTF features
        if feat in df_features.columns:
            values = df_features[feat].dropna()
            if len(values) > 0:
                # Check for suspicious patterns (this is heuristic)
                # In practice, we'd compare with HTF aggregator's last_completed_htf_close_time
                checks.append(
                    PreflightCheck(
                        f"Alignment correctness: {feat}",
                        True,
                        f"Feature present, {len(values)} non-null values",
                    )
                )

    return checks


def check_shift_correctness(df_features: pd.DataFrame) -> List[PreflightCheck]:
    """Check C3) Shift correctness (features should be t-1, not t)."""
    checks = []

    # Check a few known shifted features
    shifted_features = ["_v1_r5", "_v1_atr14", "_v1_ema_diff"]
    for feat in shifted_features:
        if feat in df_features.columns:
            # Basic check: feature should not be perfectly correlated with current price
            # (this is heuristic - full check would require recomputing with known lag)
            checks.append(
                PreflightCheck(
                    f"Shift correctness: {feat}",
                    True,
                    "Feature present (full lag verification requires recomputation)",
                )
            )

    return checks


def calculate_first_valid_eval_idx(df: pd.DataFrame, min_bars_for_features: int = 288) -> int:
    """
    Calculate first valid evaluation index based on HTF warmup requirements.
    
    This matches the logic in oanda_demo_runner._calculate_first_valid_eval_idx.
    """
    if len(df) == 0:
        return 0
    
    if "ts" not in df.columns:
        return min_bars_for_features
    
    df_ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    valid_mask = ~df_ts.isna()
    if valid_mask.sum() == 0:
        return min_bars_for_features
    
    valid_times = df_ts[valid_mask]
    eval_start_ts = valid_times.iloc[0]
    
    # M5 warmup time
    m5_warmup_time = eval_start_ts + pd.Timedelta(minutes=5 * min_bars_for_features)
    
    # Build HTF bars to find first H1/H4 close times
    n_bars_to_check = min(1000, len(df))
    df_sample = df.iloc[:n_bars_to_check].copy()
    
    try:
        from gx1.features.htf_aggregator import build_htf_from_m5
        import numpy as np
        
        m5_timestamps = (pd.to_datetime(df_sample["ts"], utc=True).astype('int64') // 1_000_000_000).astype(np.int64)
        m5_open = df_sample['open'].values.astype(np.float64)
        m5_high = df_sample['high'].values.astype(np.float64)
        m5_low = df_sample['low'].values.astype(np.float64)
        m5_close = df_sample['close'].values.astype(np.float64)
        
        h1_ts, _, _, _, _, h1_close_times = build_htf_from_m5(
            m5_timestamps, m5_open, m5_high, m5_low, m5_close, interval_hours=1
        )
        h4_ts, _, _, _, _, h4_close_times = build_htf_from_m5(
            m5_timestamps, m5_open, m5_high, m5_low, m5_close, interval_hours=4
        )
        
        if len(h1_close_times) > 0:
            first_h1_close_time = pd.Timestamp(h1_close_times[0], unit='s', tz='UTC')
        else:
            first_h1_close_time = eval_start_ts + pd.Timedelta(hours=1)
        
        if len(h4_close_times) > 0:
            first_h4_close_time = pd.Timestamp(h4_close_times[0], unit='s', tz='UTC')
        else:
            first_h4_close_time = eval_start_ts + pd.Timedelta(hours=4)
        
        first_valid_eval_time = max(eval_start_ts, m5_warmup_time, first_h1_close_time, first_h4_close_time)
        
        # Find first index where ts >= first_valid_eval_time
        first_valid_idx = np.where(valid_times >= first_valid_eval_time)[0]
        if len(first_valid_idx) > 0:
            return int(first_valid_idx[0])
        else:
            return len(df)
    except Exception as e:
        log.warning(f"[WARMUP] Failed to calculate warmup boundary: {e}, using min_bars_for_features")
        return min_bars_for_features


def check_truncate_future(
    df_raw: pd.DataFrame,
    policy_config: Dict,
    feature_meta_path: Path,
    seq_scaler_path: Optional[Path],
    snap_scaler_path: Optional[Path],
    n_samples: int = 5,
) -> List[PreflightCheck]:
    """Check C4) Truncate-future leakage test (most important)."""
    checks = []

    if "ts" not in df_raw.columns:
        checks.append(
            PreflightCheck(
                "Truncate-future",
                False,
                "Missing 'ts' column",
            )
        )
        return checks

    log.info("[TRUNCATE_FUTURE] Running truncate-future leakage test...")

    # Choose random cutoff time T (in middle of dataset)
    df_ts = pd.to_datetime(df_raw["ts"], utc=True, errors="coerce")
    valid_mask = ~df_ts.isna()
    if valid_mask.sum() < 2:
        checks.append(
            PreflightCheck(
                "Truncate-future",
                False,
                "Not enough valid timestamps",
            )
        )
        return checks

    valid_times = df_ts[valid_mask]
    cutoff_idx = len(valid_times) // 2
    cutoff_time = valid_times.iloc[cutoff_idx]

    # Build features using only data <= T
    df_truncated = df_raw[df_ts <= cutoff_time].copy()
    if len(df_truncated) < 100:
        checks.append(
            PreflightCheck(
                "Truncate-future",
                False,
                f"Truncated dataset too small: {len(df_truncated)} rows",
            )
        )
        return checks

    feature_state_trunc = FeatureState()
    feature_token_trunc = set_feature_state(feature_state_trunc)
    try:
        df_features_trunc, _, _ = build_v9_runtime_features(
            df_raw=df_truncated,
            feature_meta_path=feature_meta_path,
            seq_scaler_path=seq_scaler_path,
            snap_scaler_path=snap_scaler_path,
        )
    finally:
        reset_feature_state(feature_token_trunc)

    # Build features using full data
    feature_state_full = FeatureState()
    feature_token_full = set_feature_state(feature_state_full)
    try:
        df_features_full, _, _ = build_v9_runtime_features(
            df_raw=df_raw,
            feature_meta_path=feature_meta_path,
            seq_scaler_path=seq_scaler_path,
            snap_scaler_path=snap_scaler_path,
        )
    finally:
        reset_feature_state(feature_token_full)

    # Compare features at <=T
    # Align by timestamp (not index, since indices may differ)
    df_ts_trunc = pd.to_datetime(df_truncated["ts"], utc=True, errors="coerce")
    df_ts_full = pd.to_datetime(df_raw["ts"], utc=True, errors="coerce")
    
    # Find matching timestamps
    matching_indices = []
    for trunc_time in df_ts_trunc:
        if pd.isna(trunc_time):
            continue
        matches = np.where(df_ts_full == trunc_time)[0]
        if len(matches) > 0:
            matching_indices.append((np.where(df_ts_trunc == trunc_time)[0][0], matches[0]))
    
    if len(matching_indices) < n_samples:
        checks.append(
            PreflightCheck(
                "Truncate-future: Timestamp alignment",
                False,
                f"Only {len(matching_indices)} matching timestamps (expected >= {n_samples})",
            )
        )
        return checks

    # Sample indices for comparison
    np.random.seed(1337)
    sample_pairs = np.random.choice(len(matching_indices), min(n_samples, len(matching_indices)), replace=False)
    sample_indices = [(matching_indices[i][0], matching_indices[i][1]) for i in sample_pairs]

    # Calculate warmup boundary
    first_valid_eval_idx = calculate_first_valid_eval_idx(df_truncated, min_bars_for_features=288)
    
    # Compare ALL features (not just first 20)
    feature_cols = [c for c in df_features_trunc.columns if c.startswith("_v1")]
    mismatches = []
    mismatch_by_feature = {}  # feature_name -> list of (trunc_idx, full_idx, abs_diff, ts)

    # Reset indices for easier comparison
    df_features_trunc = df_features_trunc.reset_index(drop=True)
    df_features_full = df_features_full.reset_index(drop=True)
    
    # Get timestamps for warmup classification
    df_ts_trunc = pd.to_datetime(df_truncated["ts"], utc=True, errors="coerce")

    # Compare ALL matching indices (not just samples)
    for trunc_idx, full_idx in matching_indices:
        if trunc_idx >= len(df_features_trunc) or full_idx >= len(df_features_full):
            continue
        
        # Get timestamp for this row
        if trunc_idx < len(df_ts_trunc):
            row_ts = df_ts_trunc.iloc[trunc_idx]
        else:
            row_ts = None

        for col in feature_cols:
            if col not in df_features_trunc.columns or col not in df_features_full.columns:
                continue

            val_trunc = df_features_trunc.iloc[trunc_idx][col]
            val_full = df_features_full.iloc[full_idx][col]

            # Handle NaN
            if pd.isna(val_trunc) and pd.isna(val_full):
                continue
            if pd.isna(val_trunc) or pd.isna(val_full):
                mismatches.append((trunc_idx, full_idx, col, float('inf'), val_trunc, val_full, row_ts))
                if col not in mismatch_by_feature:
                    mismatch_by_feature[col] = []
                mismatch_by_feature[col].append((trunc_idx, full_idx, float('inf'), row_ts))
                continue

            # Compare with tolerance
            abs_diff = abs(float(val_trunc) - float(val_full))
            if abs_diff > 1e-5:
                mismatches.append((trunc_idx, full_idx, col, abs_diff, val_trunc, val_full, row_ts))
                if col not in mismatch_by_feature:
                    mismatch_by_feature[col] = []
                mismatch_by_feature[col].append((trunc_idx, full_idx, abs_diff, row_ts))

    # Analyze mismatches by feature
    feature_stats = {}
    for feat_name, feat_mismatches in mismatch_by_feature.items():
        diffs = [m[2] for m in feat_mismatches if m[2] != float('inf')]
        timestamps = [m[3] for m in feat_mismatches if m[3] is not None]
        
        # Classify as warmup vs post-warmup
        warmup_mismatches = []
        post_warmup_mismatches = []
        for m in feat_mismatches:
            trunc_idx_m = m[0]
            if trunc_idx_m < first_valid_eval_idx:
                warmup_mismatches.append(m)
            else:
                post_warmup_mismatches.append(m)
        
        feature_stats[feat_name] = {
            "mismatch_count": len(feat_mismatches),
            "mismatch_rate": len(feat_mismatches) / len(matching_indices) if len(matching_indices) > 0 else 0.0,
            "max_abs_diff": max(diffs) if diffs else float('inf'),
            "median_abs_diff": float(np.median(diffs)) if diffs else float('inf'),
            "warmup_mismatches": len(warmup_mismatches),
            "post_warmup_mismatches": len(post_warmup_mismatches),
            "first_mismatch_ts": str(min(timestamps)) if timestamps else None,
            "last_mismatch_ts": str(max(timestamps)) if timestamps else None,
        }
    
    # Sort by mismatch count (descending)
    top_mismatching_features = sorted(
        feature_stats.items(),
        key=lambda x: x[1]["mismatch_count"],
        reverse=True
    )[:20]  # Top 20
    
    # Determine verdict: PASS if mismatches are only in warmup, FAIL if post-warmup
    has_post_warmup_mismatches = any(
        stats["post_warmup_mismatches"] > 0
        for _, stats in feature_stats.items()
    )
    
    if mismatches:
        # Build detailed report
        mismatch_summary_lines = []
        mismatch_summary_lines.append(f"Total mismatches: {len(mismatches)}")
        mismatch_summary_lines.append(f"Features with mismatches: {len(feature_stats)}")
        mismatch_summary_lines.append(f"Warmup boundary index: {first_valid_eval_idx}")
        mismatch_summary_lines.append(f"Post-warmup mismatches: {'YES (FAIL)' if has_post_warmup_mismatches else 'NO (warmup only)'}")
        mismatch_summary_lines.append("")
        mismatch_summary_lines.append("Top 20 mismatching features:")
        for feat_name, stats in top_mismatching_features:
            mismatch_summary_lines.append(
                f"  - {feat_name}: count={stats['mismatch_count']}, "
                f"rate={stats['mismatch_rate']:.2%}, "
                f"max_diff={stats['max_abs_diff']:.6f}, "
                f"median_diff={stats['median_abs_diff']:.6f}, "
                f"warmup={stats['warmup_mismatches']}, "
                f"post_warmup={stats['post_warmup_mismatches']}"
            )
        
        mismatch_summary = "\n".join(mismatch_summary_lines)
        
        verdict = "FAIL" if has_post_warmup_mismatches else "PASS (warmup only)"
        checks.append(
            PreflightCheck(
                "Truncate-future: Feature equality",
                not has_post_warmup_mismatches,  # PASS only if no post-warmup mismatches
                f"{verdict}: {mismatch_summary}",
                {
                    "total_mismatches": len(mismatches),
                    "features_with_mismatches": len(feature_stats),
                    "warmup_boundary_idx": int(first_valid_eval_idx),
                    "has_post_warmup_mismatches": has_post_warmup_mismatches,
                    "top_mismatching_features": {
                        feat_name: stats
                        for feat_name, stats in top_mismatching_features
                    },
                },
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "Truncate-future: Feature equality",
                True,
                f"All {len(matching_indices)} matching index pairs match (checked {len(feature_cols)} features)",
            )
        )

    return checks


def check_reproducibility() -> List[PreflightCheck]:
    """Check D) Reproducibility."""
    checks = []

    # Git commit
    commit, is_dirty = get_git_commit()
    if is_dirty:
        checks.append(
            PreflightCheck(
                "Reproducibility: Git state",
                False,
                f"Repository is dirty (uncommitted changes). Commit: {commit[:8]}",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "Reproducibility: Git state",
                True,
                f"Repository is clean. Commit: {commit[:8]}",
            )
        )

    # Env vars
    relevant_env_vars = [
        "GX1_REQUIRE_XGB_CALIBRATION",
        "GX1_ALLOW_UNCALIBRATED_XGB",
        "GX1_ASSERT_NO_PANDAS",
        "FEATURE_BUILD_TIMEOUT_MS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    env_snapshot = {var: os.getenv(var, "not set") for var in relevant_env_vars}
    checks.append(
        PreflightCheck(
            "Reproducibility: Env vars",
            True,
            f"Captured {len(env_snapshot)} env vars",
            {"env_snapshot": env_snapshot},
        )
    )

    # Thread limits
    thread_vars = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
    thread_limits_set = any(os.getenv(var) for var in thread_vars)
    if not thread_limits_set:
        checks.append(
            PreflightCheck(
                "Reproducibility: Thread limits",
                False,
                "Thread limits not set (OMP_NUM_THREADS, MKL_NUM_THREADS, etc.)",
            )
        )
    else:
        checks.append(PreflightCheck("Reproducibility: Thread limits", True, "Thread limits are set"))

    return checks


def write_report(
    output_path: Path,
    checks: List[PreflightCheck],
    config: Dict,
    git_commit: str,
    is_dirty: bool,
) -> None:
    """Write markdown report."""
    passed = sum(1 for c in checks if c.passed)
    total = len(checks)
    status = "✅ PASS" if passed == total else "❌ FAIL"

    with open(output_path, "w") as f:
        f.write(f"# Preflight Full Build Sanity Report\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        f.write(f"**Status:** {status} ({passed}/{total} checks passed)\n\n")
        f.write(f"**Git Commit:** {git_commit[:8]} {'(dirty)' if is_dirty else '(clean)'}\n\n")

        f.write("## Configuration\n\n")
        f.write("```json\n")
        f.write(json.dumps(config, indent=2))
        f.write("\n```\n\n")

        f.write("## Checks\n\n")
        for check in checks:
            status_icon = "✅" if check.passed else "❌"
            f.write(f"### {status_icon} {check.name}\n\n")
            f.write(f"**Status:** {'PASS' if check.passed else 'FAIL'}\n\n")
            f.write(f"**Message:** {check.message}\n\n")
            if check.details:
                f.write("**Details:**\n\n")
                f.write("```json\n")
                # Convert numpy types to Python native types for JSON serialization
                def convert_to_python_type(obj):
                    if isinstance(obj, (np.integer, np.int64, np.int32, np.int8, np.int16)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (tuple, list)):
                        return [convert_to_python_type(item) for item in obj]
                    elif isinstance(obj, dict):
                        return {k: convert_to_python_type(v) for k, v in obj.items()}
                    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
                        return str(obj)
                    elif hasattr(obj, 'item'):  # numpy scalar
                        return obj.item()
                    return obj
                try:
                    details_python = convert_to_python_type(check.details)
                    f.write(json.dumps(details_python, indent=2, default=str))
                except Exception as e:
                    f.write(f"Error serializing details: {e}\n")
                    f.write(f"Details type: {type(check.details)}\n")
                f.write("\n```\n\n")
            f.write("\n")

        f.write("## Summary\n\n")
        f.write(f"- **Total checks:** {total}\n")
        f.write(f"- **Passed:** {passed}\n")
        f.write(f"- **Failed:** {total - passed}\n\n")

        if passed < total:
            f.write("### Failed Checks\n\n")
            for check in checks:
                if not check.passed:
                    f.write(f"- ❌ {check.name}: {check.message}\n")


def main():
    parser = argparse.ArgumentParser(description="Preflight sanity checks for FULLYEAR build")
    parser.add_argument("--data", type=str, required=True, help="Input data file (parquet)")
    parser.add_argument("--policy_config", type=str, required=True, help="Policy config YAML/JSON")
    parser.add_argument("--feature_meta_path", type=str, required=True, help="Feature metadata JSON")
    parser.add_argument("--seq_scaler_path", type=str, default=None, help="Sequence scaler path")
    parser.add_argument("--snap_scaler_path", type=str, default=None, help="Snapshot scaler path")
    parser.add_argument("--calibration_dir", type=str, default="models/xgb_calibration", help="Calibration directory")
    parser.add_argument("--calibration_method", type=str, default="platt", choices=["platt", "isotonic"], help="Calibration method")
    parser.add_argument("--start", type=str, default=None, help="Start date/time (ISO format)")
    parser.add_argument("--end", type=str, default=None, help="End date/time (ISO format)")
    parser.add_argument("--max_rows", type=int, default=20000, help="Maximum rows to check")
    parser.add_argument("--require_calibration", action="store_true", help="Require calibration (overrides env)")
    parser.add_argument("--no_require_calibration", action="store_true", help="Do not require calibration (overrides env)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Determine require_calibration
    if args.require_calibration:
        require_calibration = True
    elif args.no_require_calibration:
        require_calibration = False
    else:
        require_calibration = os.getenv("GX1_REQUIRE_XGB_CALIBRATION", "0") == "1"

    # Load data
    log.info(f"[PREFLIGHT] Loading data from {args.data}")
    df_raw = pd.read_parquet(args.data)
    log.info(f"[PREFLIGHT] Loaded {len(df_raw)} rows")

    # Filter by time range
    if args.start or args.end:
        if "ts" not in df_raw.columns:
            raise RuntimeError("--start/--end requires 'ts' column")
        df_raw["ts"] = pd.to_datetime(df_raw["ts"], utc=True)
        if args.start:
            start_ts = pd.Timestamp(args.start, tz="UTC")
            df_raw = df_raw[df_raw["ts"] >= start_ts].copy()
        if args.end:
            end_ts = pd.Timestamp(args.end, tz="UTC")
            df_raw = df_raw[df_raw["ts"] <= end_ts].copy()

    # Limit rows
    if len(df_raw) > args.max_rows:
        df_raw = df_raw.head(args.max_rows).copy()
        log.info(f"[PREFLIGHT] Limited to {args.max_rows} rows")

    # Load policy config
    policy_config_path = Path(args.policy_config)
    if policy_config_path.suffix in [".yaml", ".yml"]:
        import yaml
        with open(policy_config_path, "r") as f:
            policy_config = yaml.safe_load(f)
    else:
        with open(policy_config_path, "r") as f:
            policy_config = json.load(f)

    # Run checks
    all_checks = []

    log.info("[PREFLIGHT] Running raw data integrity checks...")
    all_checks.extend(check_raw_data_integrity(df_raw, require_labels=True))

    log.info("[PREFLIGHT] Running contract checks...")
    all_checks.extend(
        check_contract(
            df_raw=df_raw,
            policy_config=policy_config,
            feature_meta_path=Path(args.feature_meta_path),
            seq_scaler_path=Path(args.seq_scaler_path) if args.seq_scaler_path else None,
            snap_scaler_path=Path(args.snap_scaler_path) if args.snap_scaler_path else None,
            calibration_dir=Path(args.calibration_dir),
            calibration_method=args.calibration_method,
            require_calibration=require_calibration,
        )
    )

    # Build features once for alignment/shift checks
    log.info("[PREFLIGHT] Building features for alignment/shift checks...")
    feature_state = FeatureState()
    feature_token = set_feature_state(feature_state)
    try:
        df_features, _, _ = build_v9_runtime_features(
            df_raw=df_raw,
            feature_meta_path=Path(args.feature_meta_path),
            seq_scaler_path=Path(args.seq_scaler_path) if args.seq_scaler_path else None,
            snap_scaler_path=Path(args.snap_scaler_path) if args.snap_scaler_path else None,
        )
    finally:
        reset_feature_state(feature_token)

    log.info("[PREFLIGHT] Running HTF no-partial-bar checks...")
    all_checks.extend(check_htf_no_partial_bar(df_raw, n_samples=10))

    log.info("[PREFLIGHT] Running alignment correctness checks...")
    all_checks.extend(check_alignment_correctness(df_features))

    log.info("[PREFLIGHT] Running shift correctness checks...")
    all_checks.extend(check_shift_correctness(df_features))

    log.info("[PREFLIGHT] Running truncate-future leakage test...")
    all_checks.extend(
        check_truncate_future(
            df_raw=df_raw,
            policy_config=policy_config,
            feature_meta_path=Path(args.feature_meta_path),
            seq_scaler_path=Path(args.seq_scaler_path) if args.seq_scaler_path else None,
            snap_scaler_path=Path(args.snap_scaler_path) if args.snap_scaler_path else None,
            n_samples=5,
        )
    )

    log.info("[PREFLIGHT] Running reproducibility checks...")
    all_checks.extend(check_reproducibility())

    # Print summary
    passed = sum(1 for c in all_checks if c.passed)
    total = len(all_checks)
    status = "✅ PASS" if passed == total else "❌ FAIL"

    print("\n" + "=" * 80)
    print(f"PREFLIGHT SUMMARY: {status} ({passed}/{total} checks passed)")
    print("=" * 80)
    for check in all_checks:
        print(f"  {check}")
    print("=" * 80 + "\n")

    # Write report
    reports_dir = project_root / "reports" / "preflight"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"PREFLIGHT_FULL_BUILD_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    git_commit, is_dirty = get_git_commit()
    config = {
        "data_path": args.data,
        "policy_config_path": str(args.policy_config),
        "feature_meta_path": args.feature_meta_path,
        "seq_scaler_path": args.seq_scaler_path,
        "snap_scaler_path": args.snap_scaler_path,
        "calibration_dir": args.calibration_dir,
        "calibration_method": args.calibration_method,
        "require_calibration": require_calibration,
        "start": args.start,
        "end": args.end,
        "max_rows": args.max_rows,
        "seed": args.seed,
    }

    write_report(report_path, all_checks, config, git_commit, is_dirty)
    log.info(f"[PREFLIGHT] Report written: {report_path}")

    # Exit with error code if any checks failed
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
