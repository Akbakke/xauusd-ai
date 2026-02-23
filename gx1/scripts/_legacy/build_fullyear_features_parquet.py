#!/usr/bin/env python3
"""
Build prebuilt features for FULLYEAR replay speedup.

This script builds ALL features required for replay (build_basic_v1 + HTF + ctx)
and saves them to a parquet file. Replay can then skip feature-building entirely
by loading this prebuilt file.

Mål: FULLYEAR replay ned fra ~4.7h til < 1h.

Usage:
    python3 gx1/scripts/build_fullyear_features_parquet.py \
        --input data/raw/xauusd_m5_2025_bid_ask.parquet \
        --output data/features/xauusd_m5_2025_features_v10_ctx.parquet \
        --feature-meta models/entry_v10_ctx/feature_meta.json \
        --seq-scaler models/entry_v10_ctx/seq_scaler.pkl \
        --snap-scaler models/entry_v10_ctx/snap_scaler.pkl
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path (same as other scripts)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

# Set replay mode for feature building (ensures stateful HTF aligner is used)
os.environ["GX1_REPLAY"] = "1"
os.environ["GX1_REPLAY_INCREMENTAL_FEATURES"] = "1"
os.environ["GX1_FEATURE_USE_NP_ROLLING"] = "1"
# Disable timeout for batch feature building (we're building for entire dataset, not per-bar)
os.environ["FEATURE_BUILD_TIMEOUT_MS"] = "60000"  # 60 seconds (batch building can take longer)

# Import feature builders
from gx1.features.runtime_v10_ctx import build_v10_ctx_runtime_features
from gx1.utils.feature_context import set_feature_state, reset_feature_state, get_feature_state
from gx1.features.feature_state import FeatureState
from gx1.features.htf_align_state import HTFAligner
# DEL 3: Import SMC pack (PREBUILT-only, not used in runtime replay)
from gx1.features.smc_pack_v1 import build_smc_pack_v1

log = logging.getLogger(__name__)


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def build_features_chunk(
    df_chunk: pd.DataFrame,
    feature_meta_path: Path,
    seq_scaler_path: Optional[Path],
    snap_scaler_path: Optional[Path],
    feature_state: FeatureState,
) -> pd.DataFrame:
    """
    Build features for a chunk of data (same semantics as replay).
    
    Args:
        df_chunk: Chunk of raw OHLCV data
        feature_meta_path: Path to feature_meta.json
        seq_scaler_path: Path to sequence scaler (optional)
        snap_scaler_path: Path to snapshot scaler (optional)
        feature_state: Persistent feature state (for HTF aligners and quantiles)
    
    Returns:
        DataFrame with all features (seq + snap + all _v1_* features)
    """
    # Update quantile state incrementally for all bars in chunk (required for _v1_r1_q10_48 and _v1_r1_q90_48)
    if feature_state.r1_quantiles_state is not None:
        for close_price in df_chunk["close"].values:
            feature_state.r1_quantiles_state.update(float(close_price))
    
    # Set feature state in context (required for stateful HTF aligner)
    token = set_feature_state(feature_state)
    try:
        # Build features using same pipeline as replay
        # This returns seq+snap features (scaled and ready for model)
        df_feats, seq_features, snap_features = build_v10_ctx_runtime_features(
            df_raw=df_chunk,
            feature_meta_path=feature_meta_path,
            seq_scaler_path=seq_scaler_path,
            snap_scaler_path=snap_scaler_path,
            fillna_value=0.0,
        )
        
        # DEL 3: Add SMC pack features (PREBUILT-only, deterministic, causal)
        # SMC features are added to prebuilt parquet but NOT used by model (yet)
        # They are available for analysis and future model training
        log.debug("[PREBUILT] Building SMC pack v1 features...")
        df_feats = build_smc_pack_v1(
            df_feats,
            L=5,  # Left window for pivot detection
            R=5,  # Right window for pivot confirmation
            k_atr=0.5,  # Minimum amplitude filter in ATR units
            eq_tol_atr=0.3,  # EQ level tolerance in ATR units
            use_htf_atr=True,  # Use H1 ATR if available
        )
        log.debug(f"[PREBUILT] SMC features added: {len([c for c in df_feats.columns if c.startswith('smc_')])} columns")
        
        # Note: df_feats now contains all required features (seq + snap + smc_*)
        # These are the features that the model expects, plus SMC features for analysis
        
        # DEL 1: Filter out reserved candle columns (CLOSE, open, high, low, etc.)
        # These should never be in prebuilt parquet - they come from candles directly
        df_feats, sanitize_metadata = sanitize_feature_columns(df_feats)
    finally:
        reset_feature_state(token)
    
    return df_feats


def sanitize_feature_columns(df_features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Sanitize feature DataFrame by removing reserved candle columns.
    
    This function filters out reserved candle columns (CLOSE, open, high, low, etc.)
    that should never appear in prebuilt parquet. CLOSE is dropped silently
    (will be aliased from candles), other reserved columns cause hard-fail.
    
    Args:
        df_features: Feature DataFrame that may contain reserved columns
    
    Returns:
        Tuple of (sanitized DataFrame, metadata dict with dropped columns info)
    
    Raises:
        RuntimeError: If reserved columns other than CLOSE are found
    """
    from gx1.runtime.column_collision_guard import RESERVED_CANDLE_COLUMNS, check_reserved_candle_columns
    
    reserved_found = check_reserved_candle_columns(df_features, context="prebuilt builder")
    metadata = {"dropped_columns": [], "reserved_found": reserved_found}
    
    if reserved_found:
        # Special handling for CLOSE: drop it silently (will be aliased from candles)
        if "CLOSE" in reserved_found:
            log.warning(
                "[PREBUILT_SCHEMA_FIX] Dropped CLOSE from prebuilt features. "
                "CLOSE will be aliased from candles.close in transformer input assembly."
            )
            df_features = df_features.drop(columns=["CLOSE"], errors="ignore")
            metadata["dropped_columns"].append("CLOSE")
            reserved_found = [c for c in reserved_found if c != "CLOSE"]
        
        # Hard-fail for any other reserved columns
        if reserved_found:
            raise RuntimeError(
                f"[PREBUILT_SCHEMA_FAIL] Prebuilt features contain reserved candle columns: {reserved_found}. "
                f"Reserved columns (case-insensitive): {sorted(RESERVED_CANDLE_COLUMNS)}. "
                f"These must not appear in prebuilt parquet - they come from candles directly."
            )
    
    return df_features, metadata


def validate_features(df: pd.DataFrame) -> None:
    """
    Hard-fail on NaN/shape mismatch.
    
    Args:
        df: Features DataFrame
    
    Raises:
        ValueError: If NaN found or shape mismatch
    """
    # Check for NaN
    nan_cols = []
    for col in df.columns:
        if df[col].isna().any():
            nan_count = df[col].isna().sum()
            nan_cols.append(f"{col}: {nan_count} NaNs")
    
    if nan_cols:
        raise ValueError(
            f"Features contain NaN values (hard-fail):\n" + "\n".join(nan_cols[:20])
        )
    
    # Check for inf
    inf_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if np.isinf(df[col]).any():
            inf_count = np.isinf(df[col]).sum()
            inf_cols.append(f"{col}: {inf_count} infs")
    
    if inf_cols:
        raise ValueError(
            f"Features contain inf values (hard-fail):\n" + "\n".join(inf_cols[:20])
        )
    
    # Check for duplicate indices
    if df.index.duplicated().any():
        dup_count = df.index.duplicated().sum()
        raise ValueError(
            f"Features contain {dup_count} duplicate indices (hard-fail)"
        )
    
    # Check for monotonic index
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Features index is not monotonic increasing (hard-fail)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Build prebuilt features for FULLYEAR replay"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input parquet file (raw OHLCV data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output parquet file (prebuilt features)",
    )
    parser.add_argument(
        "--feature-meta",
        type=Path,
        required=True,
        help="Path to feature_meta.json",
    )
    parser.add_argument(
        "--seq-scaler",
        type=Path,
        help="Path to sequence scaler (optional)",
    )
    parser.add_argument(
        "--snap-scaler",
        type=Path,
        help="Path to snapshot scaler (optional)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Chunk size for processing (default: 10000 bars)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    # Validate inputs
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not args.feature_meta.exists():
        raise FileNotFoundError(f"Feature meta not found: {args.feature_meta}")
    if args.seq_scaler and not args.seq_scaler.exists():
        raise FileNotFoundError(f"Sequence scaler not found: {args.seq_scaler}")
    if args.snap_scaler and not args.snap_scaler.exists():
        raise FileNotFoundError(f"Snapshot scaler not found: {args.snap_scaler}")
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    log.info("=" * 80)
    log.info("Building prebuilt features for FULLYEAR replay")
    log.info("=" * 80)
    log.info(f"Input: {args.input}")
    log.info(f"Output: {args.output}")
    log.info(f"Feature meta: {args.feature_meta}")
    log.info(f"Chunk size: {args.chunk_size:,} bars")
    log.info("")
    
    # Load raw data
    log.info(f"Loading raw data from {args.input}...")
    df_raw = pd.read_parquet(args.input)
    log.info(f"Loaded {len(df_raw):,} bars")
    
    # Ensure DatetimeIndex
    if not isinstance(df_raw.index, pd.DatetimeIndex):
        if "time" in df_raw.columns:
            df_raw["time"] = pd.to_datetime(df_raw["time"], utc=True)
            df_raw = df_raw.set_index("time").sort_index()
        else:
            raise ValueError("DataFrame must have DatetimeIndex or 'time' column")
    
    # Initialize feature state (for stateful HTF aligner)
    # Note: HTF aligners will be initialized automatically during feature building
    # when build_v10_ctx_runtime_features calls build_basic_v1
    feature_state = FeatureState()
    
    # Initialize incremental quantile state (required for _v1_r1_q10_48 and _v1_r1_q90_48)
    from gx1.features.rolling_state_numba import RollingR1Quantiles48State
    feature_state.r1_quantiles_state = RollingR1Quantiles48State()
    
    # Process in chunks (to avoid memory issues)
    log.info("Building features in chunks...")
    all_chunks = []
    n_chunks = (len(df_raw) + args.chunk_size - 1) // args.chunk_size
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * args.chunk_size
        end_idx = min((chunk_idx + 1) * args.chunk_size, len(df_raw))
        df_chunk = df_raw.iloc[start_idx:end_idx].copy()
        
        log.info(
            f"Processing chunk {chunk_idx + 1}/{n_chunks}: "
            f"bars {start_idx:,}-{end_idx:,} ({len(df_chunk):,} bars)"
        )
        
        # Build features for chunk
        df_feats_chunk = build_features_chunk(
            df_chunk=df_chunk,
            feature_meta_path=args.feature_meta,
            seq_scaler_path=args.seq_scaler,
            snap_scaler_path=args.snap_scaler,
            feature_state=feature_state,
        )
        
        # Validate chunk
        validate_features(df_feats_chunk)
        
        all_chunks.append(df_feats_chunk)
        
        # Progress update
        if (chunk_idx + 1) % 10 == 0:
            log.info(f"Progress: {chunk_idx + 1}/{n_chunks} chunks completed")
    
    # Concatenate all chunks
    log.info("Concatenating chunks...")
    df_features = pd.concat(all_chunks, ignore_index=False)
    df_features = df_features.sort_index()
    
    # Final validation
    log.info("Validating final features...")
    validate_features(df_features)
    
    # Add metadata
    schema_version = "features_v10_ctx_v1"
    metadata = {
        "schema_version": schema_version,
        "input_file": str(args.input),
        "input_file_sha256": compute_file_sha256(args.input),
        "feature_meta": str(args.feature_meta),
        "n_bars": len(df_features),
        "n_features": len(df_features.columns),
        "feature_columns": list(df_features.columns),
        "index_start": df_features.index[0].isoformat() if len(df_features) > 0 else None,
        "index_end": df_features.index[-1].isoformat() if len(df_features) > 0 else None,
    }
    
    # Save to parquet
    log.info(f"Saving features to {args.output}...")
    # Save without metadata (metadata saved in separate manifest file)
    df_features.to_parquet(
        args.output,
        engine="pyarrow",
        compression="snappy",
        index=True,
    )
    
    # Compute output file SHA256
    output_sha256 = compute_file_sha256(args.output)
    log.info(f"Output file SHA256: {output_sha256}")
    
    # Save manifest
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest = {
        "schema_version": schema_version,
        "features_file": str(args.output),
        "features_file_sha256": output_sha256,
        "input_file": str(args.input),
        "input_file_sha256": metadata["input_file_sha256"],
        "feature_meta": str(args.feature_meta),
        "n_bars": len(df_features),
        "n_features": len(df_features.columns),
        "index_start": df_features.index[0].isoformat() if len(df_features) > 0 else None,
        "index_end": df_features.index[-1].isoformat() if len(df_features) > 0 else None,
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    log.info(f"Manifest saved to {manifest_path}")
    
    log.info("=" * 80)
    log.info("✅ Feature building complete!")
    log.info(f"   Features: {args.output}")
    log.info(f"   Manifest: {manifest_path}")
    log.info(f"   SHA256: {output_sha256}")
    log.info(f"   Bars: {len(df_features):,}")
    log.info(f"   Features: {len(df_features.columns):,}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
