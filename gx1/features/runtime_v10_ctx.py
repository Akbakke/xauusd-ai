"""
Runtime V10_CTX Feature Builder - Uses core runtime without V9 dependencies.

This module provides V10_CTX identity while using core runtime feature logic
that has zero V9 dependencies. All feature building is identical to the original
V9 runtime, but without any V9 imports.

Design Philosophy:
  - Zero V9 dependencies: uses runtime_sniper_core (no V9 imports)
  - Same feature logic: uses build_basic_v1 and build_sequence_features (not V9-specific)
  - V10_CTX identity for logging and provenance
  - Replay-safe: never imports V9 modules, even indirectly
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd

# DEL 3: PREBUILT mode fix - move runtime_sniper_core import to lazy (baseline-only)
# runtime_sniper_core is forbidden in PREBUILT mode, so we only import it when needed (baseline mode only)
# Do NOT import at top level - this module can be imported in PREBUILT mode

log = logging.getLogger(__name__)


def build_v10_ctx_runtime_features(
    df_raw: pd.DataFrame,
    feature_meta_path: Path,
    seq_scaler_path: Optional[Path] = None,
    snap_scaler_path: Optional[Path] = None,
    fillna_value: float = 0.0,
) -> Tuple[pd.DataFrame, list, list]:
    """
    Build V10_CTX runtime features using core runtime (no V9 dependencies).
    
    This wrapper uses core runtime logic that has zero V9 dependencies.
    All feature building is identical to the original V9 runtime, but without
    any V9 imports or logging.
    
    DEL 3: This eliminates V9 runtime warnings in replay by using core runtime.
    
    FASE 0.3: Global kill-switch - GX1_FEATURE_BUILD_DISABLED=1 forbydder ALL feature-building.
    
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
    # FASE 0.3: Global kill-switch - hard-fail hvis feature-building er deaktivert
    import os
    feature_build_disabled = os.getenv("GX1_FEATURE_BUILD_DISABLED", "0") == "1"
    if feature_build_disabled:
        raise RuntimeError(
            "[PREBUILT_FAIL] build_v10_ctx_runtime_features() called while GX1_FEATURE_BUILD_DISABLED=1. "
            "Feature-building is completely disabled in prebuilt mode. "
            "This is a hard invariant - prebuilt features must be used directly."
        )
    
    # DEL 3: Use core runtime (no V9 dependencies) - lazy import to avoid PREBUILT import leak
    # Import here (lazy) - only called in baseline mode (PREBUILT mode hard-fails above)
    from gx1.features.runtime_sniper_core import build_sniper_core_runtime_features
    df_feats, seq_features, snap_features = build_sniper_core_runtime_features(
        df_raw=df_raw,
        feature_meta_path=feature_meta_path,
        seq_scaler_path=seq_scaler_path,
        snap_scaler_path=snap_scaler_path,
        fillna_value=fillna_value,
    )
    
    # Log with V10_CTX identity
    log.debug(
        "[ENTRY_V10_CTX] Runtime features built (via core): shape=%s, n_seq=%d, n_snap=%d",
        df_feats.shape, len(seq_features), len(snap_features)
    )
    
    return df_feats, seq_features, snap_features
