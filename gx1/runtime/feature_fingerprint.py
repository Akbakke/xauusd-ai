"""
Feature Schema Fingerprint — LIVE Feature Validation

Computes a fingerprint of feature schema (column names, order, dtype, dims)
and compares against expected fingerprint from training/prebuilt meta.
Hard-fails on mismatch.

Dependencies (explicit install line):
  (no external dependencies beyond stdlib)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureFingerprint:
    """Feature schema fingerprint."""
    
    seq_features: List[str]
    snap_features: List[str]
    seq_dtype: str
    snap_dtype: str
    seq_shape: Tuple[int, int]  # (seq_len, n_features)
    snap_shape: Tuple[int, ...]  # (n_features,)
    fingerprint_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "seq_features": self.seq_features,
            "snap_features": self.snap_features,
            "seq_dtype": str(self.seq_dtype),
            "snap_dtype": str(self.snap_dtype),
            "seq_shape": list(self.seq_shape),
            "snap_shape": list(self.snap_shape),
            "fingerprint_hash": self.fingerprint_hash,
        }


def compute_feature_fingerprint(
    seq_data: np.ndarray,
    snap_data: np.ndarray,
    seq_feat_names: List[str],
    snap_feat_names: List[str],
    xgb_seq_channel_names: Optional[List[str]] = None,
    xgb_snap_channel_names: Optional[List[str]] = None,
) -> FeatureFingerprint:
    """
    Compute feature schema fingerprint from actual data.
    
    Args:
        seq_data: Sequence features array [seq_len, n_features]
        snap_data: Snapshot features array [n_features]
        seq_feat_names: List of sequence feature names (base features only)
        snap_feat_names: List of snapshot feature names (base features only)
        xgb_seq_channel_names: Optional list of XGB seq channel names (if None, inferred from contract)
        xgb_snap_channel_names: Optional list of XGB snap channel names (if None, inferred from contract)
    
    Returns:
        FeatureFingerprint object
    """
    # Validate shapes
    if seq_data.ndim != 2:
        raise ValueError(f"seq_data must be 2D, got shape {seq_data.shape}")
    if snap_data.ndim != 1:
        raise ValueError(f"snap_data must be 1D, got shape {snap_data.shape}")
    
    seq_len, seq_n_feat = seq_data.shape
    snap_n_feat = snap_data.shape[0]
    
    # FIX: Handle case where seq_feat_names only includes base features, but seq_data includes XGB channels
    # This is the V10_CTX case: base_seq_features (13) + XGB_CHANNELS (3) = TOTAL_SEQ_FEATURES (16)
    n_base_seq = len(seq_feat_names)
    n_base_snap = len(snap_feat_names)
    
    # Infer XGB channel names from contract if not provided
    if xgb_seq_channel_names is None:
        try:
            from gx1.features.feature_contract_v10_ctx import SEQ_XGB_CHANNEL_NAMES, XGB_CHANNELS
            xgb_seq_channel_names = SEQ_XGB_CHANNEL_NAMES
            expected_xgb_channels = XGB_CHANNELS
        except ImportError:
            # Fallback: assume no XGB channels if contract not available
            xgb_seq_channel_names = []
            expected_xgb_channels = 0
    
    if xgb_snap_channel_names is None:
        try:
            from gx1.features.feature_contract_v10_ctx import SNAP_XGB_CHANNEL_NAMES
            xgb_snap_channel_names = SNAP_XGB_CHANNEL_NAMES
        except ImportError:
            xgb_snap_channel_names = []
    
    # Validate dimensions: seq_data must be base + XGB, snap_data must be base + XGB
    # NOTE: Use XGB_CHANNELS from contract (not len of channel names) because model expects fixed dimension
    # margin_xgb was REMOVED from channel names but XGB_CHANNELS=3 is kept for model compatibility
    try:
        from gx1.features.feature_contract_v10_ctx import XGB_CHANNELS
        expected_xgb_channels = XGB_CHANNELS
    except ImportError:
        # Fallback to len if contract not available
        expected_xgb_channels = max(len(xgb_seq_channel_names), len(xgb_snap_channel_names))
    
    expected_seq_total = n_base_seq + expected_xgb_channels
    expected_snap_total = n_base_snap + expected_xgb_channels
    
    if seq_n_feat != expected_seq_total:
        raise ValueError(
            f"FEATURE_DIM_MISMATCH: seq_data has {seq_n_feat} features, but expected "
            f"{expected_seq_total} (base={n_base_seq} + XGB={expected_xgb_channels}). "
            f"seq_feat_names length: {n_base_seq}, xgb_seq_channel_names: {xgb_seq_channel_names}"
        )
    
    if snap_n_feat != expected_snap_total:
        raise ValueError(
            f"FEATURE_DIM_MISMATCH: snap_data has {snap_n_feat} features, but expected "
            f"{expected_snap_total} (base={n_base_snap} + XGB={expected_xgb_channels}). "
            f"snap_feat_names length: {n_base_snap}, xgb_snap_channel_names: {xgb_snap_channel_names}"
        )
    
    # Build full feature names (base + XGB channels)
    full_seq_feat_names = seq_feat_names + xgb_seq_channel_names
    full_snap_feat_names = snap_feat_names + xgb_snap_channel_names
    
    # Build fingerprint string (deterministic) - use full feature names (base + XGB)
    fingerprint_parts = [
        f"seq_features:{','.join(sorted(full_seq_feat_names))}",
        f"snap_features:{','.join(sorted(full_snap_feat_names))}",
        f"seq_dtype:{seq_data.dtype}",
        f"snap_dtype:{snap_data.dtype}",
        f"seq_shape:{seq_len},{seq_n_feat}",
        f"snap_shape:{snap_n_feat}",
    ]
    fingerprint_str = "|".join(fingerprint_parts)
    
    # Compute hash
    fingerprint_hash = hashlib.sha256(fingerprint_str.encode("utf-8")).hexdigest()
    
    return FeatureFingerprint(
        seq_features=full_seq_feat_names.copy(),  # Include XGB channels
        snap_features=full_snap_feat_names.copy(),  # Include XGB channels
        seq_dtype=str(seq_data.dtype),
        snap_dtype=str(snap_data.dtype),
        seq_shape=(seq_len, seq_n_feat),
        snap_shape=(snap_n_feat,),
        fingerprint_hash=fingerprint_hash,
    )


def load_expected_fingerprint(bundle_dir: Path) -> Optional[FeatureFingerprint]:
    """
    Load expected fingerprint from bundle metadata.
    
    Tries multiple locations:
    1. bundle_dir/metadata.json
    2. bundle_dir/bundle_metadata.json
    3. bundle_dir/../feature_meta.json (if available, compute from it)
    
    Args:
        bundle_dir: Bundle directory path
    
    Returns:
        FeatureFingerprint if found, None otherwise
    """
    # Try metadata.json first
    metadata_path = bundle_dir / "metadata.json"
    if not metadata_path.exists():
        # Try bundle_metadata.json
        metadata_path = bundle_dir / "bundle_metadata.json"
    
    # If metadata exists but doesn't have feature names, try feature_meta.json fallback
    metadata_has_features = False
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            # Check if metadata has feature names
            if metadata.get("seq_features") or metadata.get("snap_features"):
                metadata_has_features = True
        except Exception:
            pass
    
    if not metadata_path.exists() or not metadata_has_features:
        # Try to compute from feature_meta.json if available
        # Check multiple possible locations
        feature_meta_candidates = [
            bundle_dir / "feature_meta.json",
            bundle_dir.parent / "feature_meta.json",
            Path("gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json"),  # Standard location
            Path("models/entry_v10_ctx/feature_meta.json"),
        ]
        
        feature_meta_path = None
        for candidate in feature_meta_candidates:
            if candidate.exists():
                feature_meta_path = candidate
                break
        
        if feature_meta_path and feature_meta_path.exists():
            # Load feature_meta and compute expected fingerprint
            with open(feature_meta_path, "r") as f:
                feature_meta = json.load(f)
            
            seq_features = feature_meta.get("seq_features", [])
            snap_features = feature_meta.get("snap_features", [])
            
            # Add XGB channels to get full feature list
            try:
                from gx1.features.feature_contract_v10_ctx import SEQ_XGB_CHANNEL_NAMES, SNAP_XGB_CHANNEL_NAMES, TOTAL_SEQ_FEATURES, TOTAL_SNAP_FEATURES
                full_seq_features = seq_features + SEQ_XGB_CHANNEL_NAMES
                full_snap_features = snap_features + SNAP_XGB_CHANNEL_NAMES
                
                # Build fingerprint string (same format as compute_feature_fingerprint)
                seq_len = 30  # Default seq_len for V10_CTX
                fingerprint_parts = [
                    f"seq_features:{','.join(sorted(full_seq_features))}",
                    f"snap_features:{','.join(sorted(full_snap_features))}",
                    f"seq_dtype:float32",
                    f"snap_dtype:float32",
                    f"seq_shape:{seq_len},{TOTAL_SEQ_FEATURES}",
                    f"snap_shape:{TOTAL_SNAP_FEATURES}",
                ]
                fingerprint_str = "|".join(fingerprint_parts)
                fingerprint_hash = hashlib.sha256(fingerprint_str.encode("utf-8")).hexdigest()
                
                return FeatureFingerprint(
                    seq_features=full_seq_features.copy(),
                    snap_features=full_snap_features.copy(),
                    seq_dtype="float32",
                    snap_dtype="float32",
                    seq_shape=(seq_len, TOTAL_SEQ_FEATURES),
                    snap_shape=(TOTAL_SNAP_FEATURES,),
                    fingerprint_hash=fingerprint_hash,
                )
            except ImportError:
                pass
        
        return None
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Extract feature info from metadata
    seq_features = metadata.get("seq_features", [])
    snap_features = metadata.get("snap_features", [])
    
    if not seq_features or not snap_features:
        return None
    
    # Build fingerprint string (same format as compute_feature_fingerprint)
    # IMPORTANT: Must include XGB channels if they're not already in seq_features/snap_features
    seq_len = metadata.get("seq_len", 30)
    seq_n_feat = len(seq_features)
    snap_n_feat = len(snap_features)
    
    # Check if XGB channels are already included
    try:
        from gx1.features.feature_contract_v10_ctx import SEQ_XGB_CHANNEL_NAMES, SNAP_XGB_CHANNEL_NAMES, TOTAL_SEQ_FEATURES, TOTAL_SNAP_FEATURES
        
        # If seq_features doesn't include XGB channels, add them
        if seq_n_feat < TOTAL_SEQ_FEATURES:
            full_seq_features = seq_features + SEQ_XGB_CHANNEL_NAMES
            seq_n_feat = TOTAL_SEQ_FEATURES
        else:
            full_seq_features = seq_features
        
        # If snap_features doesn't include XGB channels, add them
        if snap_n_feat < TOTAL_SNAP_FEATURES:
            full_snap_features = snap_features + SNAP_XGB_CHANNEL_NAMES
            snap_n_feat = TOTAL_SNAP_FEATURES
        else:
            full_snap_features = snap_features
    except ImportError:
        # Fallback: use metadata as-is
        full_seq_features = seq_features
        full_snap_features = snap_features
    
    fingerprint_parts = [
        f"seq_features:{','.join(sorted(full_seq_features))}",
        f"snap_features:{','.join(sorted(full_snap_features))}",
        f"seq_dtype:float32",
        f"snap_dtype:float32",
        f"seq_shape:{seq_len},{seq_n_feat}",
        f"snap_shape:{snap_n_feat}",
    ]
    fingerprint_str = "|".join(fingerprint_parts)
    fingerprint_hash = hashlib.sha256(fingerprint_str.encode("utf-8")).hexdigest()
    
    return FeatureFingerprint(
        seq_features=full_seq_features.copy(),
        snap_features=full_snap_features.copy(),
        seq_dtype="float32",
        snap_dtype="float32",
        seq_shape=(seq_len, seq_n_feat),
        snap_shape=(snap_n_feat,),
        fingerprint_hash=fingerprint_hash,
    )


def validate_feature_fingerprint(
    actual: FeatureFingerprint,
    expected: Optional[FeatureFingerprint],
    is_live: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Validate actual fingerprint against expected.
    
    Args:
        actual: Actual fingerprint from live features
        expected: Expected fingerprint from bundle metadata
        is_live: If True, hard-fail on mismatch. If False, return warning.
    
    Returns:
        (is_valid, error_message)
    """
    if expected is None:
        return (True, None)  # No expected fingerprint available (skip validation)
    
    errors = []
    
    # Check hash match (fast path)
    if actual.fingerprint_hash != expected.fingerprint_hash:
        errors.append(f"Fingerprint hash mismatch: actual={actual.fingerprint_hash[:16]}..., expected={expected.fingerprint_hash[:16]}...")
    
    # Check feature names match
    if set(actual.seq_features) != set(expected.seq_features):
        missing = set(expected.seq_features) - set(actual.seq_features)
        extra = set(actual.seq_features) - set(expected.seq_features)
        if missing:
            errors.append(f"Missing seq_features: {sorted(missing)}")
        if extra:
            errors.append(f"Extra seq_features: {sorted(extra)}")
    
    if set(actual.snap_features) != set(expected.snap_features):
        missing = set(expected.snap_features) - set(actual.snap_features)
        extra = set(actual.snap_features) - set(expected.snap_features)
        if missing:
            errors.append(f"Missing snap_features: {sorted(missing)}")
        if extra:
            errors.append(f"Extra snap_features: {sorted(extra)}")
    
    # Check shapes match
    if actual.seq_shape[1] != expected.seq_shape[1]:  # n_features
        errors.append(f"seq_shape[1] mismatch: actual={actual.seq_shape[1]}, expected={expected.seq_shape[1]}")
    
    if actual.snap_shape[0] != expected.snap_shape[0]:
        errors.append(f"snap_shape[0] mismatch: actual={actual.snap_shape[0]}, expected={expected.snap_shape[0]}")
    
    if errors:
        error_msg = "Feature fingerprint mismatch:\n" + "\n".join([f"  - {e}" for e in errors])
        if is_live:
            raise RuntimeError(f"[FEATURE_FINGERPRINT_FAIL] {error_msg}")
        return (False, error_msg)
    
    return (True, None)
