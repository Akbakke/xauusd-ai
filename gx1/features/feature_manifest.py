#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature manifest for runtime vs training validation.

Exports feature manifest (name + order + dtype) to JSON and validates
runtime features against manifest to ensure consistency between training and runtime.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def export_feature_manifest(
    features_df: pd.DataFrame,
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Export feature manifest to JSON.
    
    Args:
        features_df: DataFrame with features (columns = feature names)
        output_path: Path to save manifest JSON
        metadata: Optional metadata (model version, training date, etc.)
    """
    manifest = {
        "features": [],
        "metadata": metadata or {},
    }
    
    for col in features_df.columns:
        dtype_str = str(features_df[col].dtype)
        manifest["features"].append({
            "name": col,
            "dtype": dtype_str,
            "order": len(manifest["features"]),  # 0-indexed
        })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info("[FEATURE_MANIFEST] Exported manifest to %s (%d features)", output_path, len(manifest["features"]))


def load_feature_manifest(manifest_path: Path) -> Dict[str, Any]:
    """
    Load feature manifest from JSON.
    
    Args:
        manifest_path: Path to manifest JSON
        
    Returns:
        Manifest dict with "features" and "metadata" keys
    """
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    logger.debug("[FEATURE_MANIFEST] Loaded manifest from %s (%d features)", manifest_path, len(manifest.get("features", [])))
    return manifest


def validate_runtime_features(
    runtime_features: pd.DataFrame,
    manifest: Dict[str, Any],
    prod_mode: bool = False,
) -> tuple[bool, List[str]]:
    """
    Validate runtime features against manifest.
    
    Args:
        runtime_features: DataFrame with runtime features
        manifest: Feature manifest dict (from load_feature_manifest)
        prod_mode: If True, treat mismatches as errors (block trading)
                   If False, log warnings only
        
    Returns:
        (is_valid, error_messages)
        - is_valid: True if features match manifest
        - error_messages: List of validation error messages
    """
    errors = []
    manifest_features = manifest.get("features", [])
    
    # Check feature count
    expected_count = len(manifest_features)
    actual_count = len(runtime_features.columns)
    if expected_count != actual_count:
        errors.append(
            f"Feature count mismatch: expected {expected_count}, got {actual_count}"
        )
    
    # Check feature names and order
    expected_names = [f["name"] for f in manifest_features]
    actual_names = list(runtime_features.columns)
    
    if expected_names != actual_names:
        errors.append(
            f"Feature name/order mismatch:\n"
            f"  Expected: {expected_names}\n"
            f"  Actual: {actual_names}"
        )
    
    # Check dtypes (loose check: allow compatible types)
    for i, feat_spec in enumerate(manifest_features):
        feat_name = feat_spec["name"]
        expected_dtype = feat_spec["dtype"]
        
        if feat_name not in runtime_features.columns:
            errors.append(f"Missing feature: {feat_name}")
            continue
        
        actual_dtype = str(runtime_features[feat_name].dtype)
        
        # Loose dtype check: allow compatible types (e.g., int64 vs int32)
        if not _dtypes_compatible(expected_dtype, actual_dtype):
            errors.append(
                f"Feature {feat_name}: dtype mismatch (expected {expected_dtype}, got {actual_dtype})"
            )
    
    is_valid = len(errors) == 0
    
    if not is_valid:
        error_msg = "Feature manifest validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        if prod_mode:
            logger.error("[FEATURE_MANIFEST] %s", error_msg)
            logger.error("[FEATURE_MANIFEST] PROD mode: Trading blocked due to feature mismatch")
        else:
            logger.warning("[FEATURE_MANIFEST] %s", error_msg)
    
    return is_valid, errors


def _dtypes_compatible(expected: str, actual: str) -> bool:
    """
    Check if two dtypes are compatible (loose check).
    
    Allows:
    - int64 vs int32 vs int (all integers)
    - float64 vs float32 vs float (all floats)
    - object vs string (both strings)
    """
    # Normalize dtype strings
    expected_norm = expected.lower().replace("_", "")
    actual_norm = actual.lower().replace("_", "")
    
    # Integer types
    if "int" in expected_norm and "int" in actual_norm:
        return True
    
    # Float types
    if "float" in expected_norm and "float" in actual_norm:
        return True
    
    # String types
    if expected_norm in ("object", "string", "str") and actual_norm in ("object", "string", "str"):
        return True
    
    # Exact match
    if expected_norm == actual_norm:
        return True
    
    return False


def compute_feature_hash(features_df: pd.DataFrame) -> str:
    """
    Compute hash of feature columns (for drift detection).
    
    Args:
        features_df: DataFrame with features
        
    Returns:
        SHA256 hash of feature column names (sorted)
    """
    import hashlib
    
    feature_names = sorted(features_df.columns.tolist())
    feature_str = ",".join(feature_names)
    return hashlib.sha256(feature_str.encode("utf-8")).hexdigest()[:16]

