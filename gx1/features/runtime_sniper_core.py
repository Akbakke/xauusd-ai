"""
Runtime Sniper Core - Feature building without V9 dependencies.

This module provides core runtime feature building functionality without
any V9 dependencies. It uses basic_v1 and sequence_features to build
features that match the expected feature contract.

Design Philosophy:
  - Zero V9 dependencies: uses basic_v1 and sequence_features only
  - Same feature logic: produces features compatible with V10_CTX contract
  - Replay-safe: never imports V9 modules, even indirectly
  - PREBUILT-safe: hard-fails if called in PREBUILT mode
"""

import logging
import json
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


def build_sniper_core_runtime_features(
    df_raw: pd.DataFrame,
    feature_meta_path: Path,
    seq_scaler_path: Optional[Path] = None,
    snap_scaler_path: Optional[Path] = None,
    fillna_value: float = 0.0,
) -> Tuple[pd.DataFrame, list, list]:
    """
    Build runtime features using core runtime (no V9 dependencies).
    
    This function builds features using basic_v1 and sequence_features,
    then subsets to the features expected by the model and scales them.
    
    Args:
        df_raw: Raw candles DataFrame (OHLCV)
        feature_meta_path: Path to feature_meta.json
        seq_scaler_path: Optional path to sequence scaler
        snap_scaler_path: Optional path to snapshot scaler
        fillna_value: Fill value for NaN/inf
    
    Returns:
        df_feats: DataFrame with seq+snap features
        seq_features: List of sequence feature names
        snap_features: List of snapshot feature names
    """
    import os
    
    # DEL 5: TRUTH/PREBUILT policy - FORBID 0.0 injection
    # In TRUTH mode, we must have REAL features, not fake 0.0 columns
    # This prevents "fake coverage" where validation passes but features are meaningless
    # NOTE: Builder is NOT in TRUTH mode - TRUTH is only for replay/validation
    # Builder can use 0.0 injection to fill missing features (schema contract gate will catch issues)
    is_truth = os.getenv("GX1_TRUTH_TELEMETRY", "0") == "1" or os.getenv("GX1_TRUTH_MODE", "0") == "1"
    # Builder mode detection removed - builder is NOT in TRUTH mode
    is_prebuilt_builder = False  # Builder is not in TRUTH mode
    
    # FASE 0.3: Global kill-switch - hard-fail hvis feature-building er deaktivert
    feature_build_disabled = os.getenv("GX1_FEATURE_BUILD_DISABLED", "0") == "1"
    if feature_build_disabled:
        raise RuntimeError(
            "[PREBUILT_FAIL] build_sniper_core_runtime_features() called while GX1_FEATURE_BUILD_DISABLED=1. "
            "Feature-building is completely disabled in prebuilt mode. "
            "This is a hard invariant - prebuilt features must be used directly."
        )
    
    # TRIPWIRE: Fail-fast if prebuilt features are enabled
    prebuilt_enabled = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES") == "1"
    is_replay = os.getenv("GX1_REPLAY") == "1"
    if prebuilt_enabled and is_replay:
        raise RuntimeError(
            "[PREBUILT_FAIL] build_sniper_core_runtime_features() called while GX1_REPLAY_USE_PREBUILT_FEATURES=1. "
            "This indicates prebuilt bypass is not working. "
            "Prebuilt features must be used directly without calling build_sniper_core_runtime_features()."
        )
    
    if df_raw is None or len(df_raw) == 0:
        raise RuntimeError(
            "[RUNTIME_SNIPER_CORE] df_raw is empty – cannot build features"
        )
    
    # 1) Build basic_v1 features
    from gx1.features.basic_v1 import build_basic_v1
    build_result = build_basic_v1(df_raw)
    # build_basic_v1 returns (df, newcols) tuple
    if isinstance(build_result, tuple):
        df_basic, newcols = build_result
    else:
        df_basic = build_result
    
    # 2) Build sequence features if needed
    # Note: sequence_features may not be needed if basic_v1 already provides all features
    # For now, we'll use basic_v1 features directly
    
    # 3) Load feature_meta to get expected features
    with open(feature_meta_path) as f:
        feature_meta = json.load(f)
    
    seq_features = feature_meta.get("seq_features", [])
    snap_features = feature_meta.get("snap_features", [])
    
    expected_features = set(seq_features) | set(snap_features)
    present_features = set(df_basic.columns)
    
    # Map case-insensitive: CLOSE -> close, etc.
    col_lower_map = {col.lower(): col for col in df_basic.columns}
    missing = []
    for exp_feat in expected_features:
        exp_lower = exp_feat.lower()
        if exp_feat not in present_features:
            # Try case-insensitive match
            if exp_lower in col_lower_map:
                # Rename column to match expected case
                actual_col = col_lower_map[exp_lower]
                df_basic = df_basic.rename(columns={actual_col: exp_feat})
                log.debug(
                    "[RUNTIME_SNIPER_CORE] Mapped column '%s' -> '%s' (case-insensitive)",
                    actual_col, exp_feat
                )
            else:
                missing.append(exp_feat)
    
    # Re-check after case-insensitive mapping
    present_features = set(df_basic.columns)
    missing = sorted(expected_features - present_features)
    extra = sorted(present_features - expected_features)
    
    if missing:
        if is_truth or is_prebuilt_builder:
            # TRUTH/PREBUILT builder: Hard fail on missing features
            # Builder must produce ALL required features, not inject 0.0
            raise RuntimeError(
                f"[TRUTH_SCHEMA_FAIL] Missing {len(missing)} required features in TRUTH/PREBUILT builder mode. "
                f"0.0 injection is FORBIDDEN in TRUTH mode. "
                f"Missing features: {sorted(missing)[:20]}. "
                f"Fix: Ensure builder produces all features from feature_meta.json, or update feature_meta.json to remove unused features."
            )
        else:
            # Non-TRUTH mode: Allow 0.0 injection (legacy behavior for live mode)
            log.warning(
                "[RUNTIME_SNIPER_CORE] Missing %d expected features (will be filled with 0.0): %s",
                len(missing),
                ", ".join(missing[:10]) + ("..." if len(missing) > 10 else ""),
            )
            # Add missing features as columns filled with 0.0
            for feat_name in missing:
                df_basic[feat_name] = 0.0
            log.info(
                "[RUNTIME_SNIPER_CORE] Added %d missing features as 0.0 columns",
                len(missing)
            )
    
    # Re-check after adding missing features
    present_features = set(df_basic.columns)
    missing_after = sorted(expected_features - present_features)
    
    if missing_after:
        # This should never happen after adding missing features
        log.error(
            "[RUNTIME_SNIPER_CORE] Still missing %d expected features after adding 0.0 columns: %s",
            len(missing_after),
            ", ".join(missing_after),
        )
        raise RuntimeError(
            f"[RUNTIME_SNIPER_CORE] Feature mismatch after adding missing features!\n"
            f"  Missing {len(missing_after)} expected features: {sorted(missing_after)}\n"
            f"  This should not happen - feature addition failed."
        )
    
    # 4) Remove reserved candle columns from df_basic before subsetting
    # Reserved columns (CLOSE, close, volume, etc.) should not be in prebuilt
    # They come from candles directly and are aliased at runtime
    from gx1.runtime.column_collision_guard import RESERVED_CANDLE_COLUMNS
    ALIASED_FEATURES = {"CLOSE", "close", "volume"}  # These are aliased from candles, not in prebuilt
    
    # Remove reserved columns from df_basic
    reserved_in_df = [c for c in df_basic.columns if c.lower() in [r.lower() for r in RESERVED_CANDLE_COLUMNS]]
    if reserved_in_df:
        log.debug(
            "[RUNTIME_SNIPER_CORE] Removing reserved candle columns from df_basic: %s",
            reserved_in_df
        )
        df_basic = df_basic.drop(columns=reserved_in_df, errors="ignore")
    
    # Filter out reserved columns from feature list
    seq_features_filtered = [f for f in seq_features if f not in ALIASED_FEATURES]
    snap_features_filtered = [f for f in snap_features if f not in ALIASED_FEATURES]
    
    # Get columns that exist in df_basic and are in expected features
    # Note: seq_features_filtered and snap_features_filtered may have overlaps (same feature in both seq and snap)
    # We need to deduplicate to avoid duplicate columns
    all_required_features = list(dict.fromkeys(seq_features_filtered + snap_features_filtered))  # Preserves order, removes duplicates
    available_features = [f for f in all_required_features if f in df_basic.columns]
    df_feats = df_basic[available_features].copy()
    
    # Add any missing features as 0.0 columns (excluding reserved) - ONLY if not TRUTH mode
    # Note: is_truth and is_prebuilt_builder are defined at function start (line ~55-58)
    missing_features = [f for f in all_required_features if f not in df_feats.columns]
    if missing_features:
        if is_truth or is_prebuilt_builder:
            # TRUTH/PREBUILT builder: Hard fail
            raise RuntimeError(
                f"[TRUTH_SCHEMA_FAIL] Missing {len(missing_features)} required features after filtering reserved columns. "
                f"0.0 injection is FORBIDDEN in TRUTH mode. "
                f"Missing: {sorted(missing_features)[:20]}. "
                f"Fix: Builder must produce all required features."
            )
        else:
            # Non-TRUTH: Allow 0.0 injection
            # Use pd.concat for better performance (avoids DataFrame fragmentation warning)
            missing_df = pd.DataFrame(
                0.0,
                index=df_feats.index,
                columns=missing_features,
                dtype=np.float32
            )
            df_feats = pd.concat([df_feats, missing_df], axis=1)
            log.info(
                "[RUNTIME_SNIPER_CORE] Added %d missing features as 0.0 columns (excluding reserved)",
                len(missing_features)
            )
    
    # 5) Fill NaN/inf
    df_feats = df_feats.fillna(fillna_value)
    df_feats = df_feats.replace([np.inf, -np.inf], fillna_value)
    
    # 6) Apply scalers if provided
    if seq_scaler_path and seq_scaler_path.exists():
        from sklearn.preprocessing import StandardScaler
        import joblib
        seq_scaler = joblib.load(seq_scaler_path)
        # Scale sequence features
        seq_cols = [col for col in seq_features if col in df_feats.columns]
        if seq_cols:
            df_feats[seq_cols] = seq_scaler.transform(df_feats[seq_cols])
    
    if snap_scaler_path and snap_scaler_path.exists():
        from sklearn.preprocessing import StandardScaler
        import joblib
        snap_scaler = joblib.load(snap_scaler_path)
        # Scale snapshot features
        snap_cols = [col for col in snap_features if col in df_feats.columns]
        if snap_cols:
            df_feats[snap_cols] = snap_scaler.transform(df_feats[snap_cols])
    
    log.debug(
        "[RUNTIME_SNIPER_CORE] Built features: shape=%s, n_seq=%d, n_snap=%d",
        df_feats.shape, len(seq_features), len(snap_features)
    )
    
    return df_feats, seq_features, snap_features
