"""
Stub implementation of feature_manifest for compatibility.

This module provides minimal implementations of load_manifest() and align_features()
to maintain compatibility with existing code that references gx1.tuning.feature_manifest.

The original tuning module has been removed, but these functions are still needed
for feature alignment in live/replay execution.
"""

from typing import Any, Dict, Optional
import pandas as pd
import logging

log = logging.getLogger(__name__)


def load_manifest() -> Dict[str, Any]:
    """
    Load feature manifest (stub implementation).
    
    Returns
    -------
    Dict[str, Any]
        Empty manifest dict with minimal structure.
    """
    return {
        "feature_cols": [],
        "training_stats": {},
        "version": "stub_1.0",
    }


def align_features(
    df: pd.DataFrame,
    manifest: Optional[Dict[str, Any]] = None,
    training_stats: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Align features with manifest (stub implementation - no-op).
    
    This is a no-op stub that simply returns the input DataFrame unchanged.
    The original align_features() would have normalized/scaled features to match
    training data, but for replay purposes, we assume features are already compatible.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input feature DataFrame.
    manifest : Optional[Dict[str, Any]], optional
        Feature manifest (ignored in stub).
    training_stats : Optional[Dict[str, Any]], optional
        Training statistics (ignored in stub).
    
    Returns
    -------
    pd.DataFrame
        Input DataFrame unchanged (no-op).
    """
    # No-op: return input unchanged
    # In the original implementation, this would normalize/scale features
    # to match training data statistics, but for replay we assume compatibility
    return df.copy()

