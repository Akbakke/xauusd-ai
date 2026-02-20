"""
Feature Contract Utilities - SSoT for effective required prebuilt features.

This module provides canonical functions for determining which features
are required in prebuilt parquet files vs. which are aliased from raw candles.
"""

from typing import Dict, List, Set, Tuple
import logging

log = logging.getLogger(__name__)


def get_effective_required_prebuilt_features(
    feature_meta: Dict,
    aliased_features: Set[str] = None
) -> Tuple[List[str], List[str]]:
    """
    Get effective required features for prebuilt parquet (excluding aliased features).
    
    This is the SSoT function for determining which features must be present
    in prebuilt parquet files. Features that are aliased from raw candles
    (e.g., close, volume) are excluded from the required set but returned
    separately for reporting.
    
    Args:
        feature_meta: Feature metadata dict with "seq_features" and "snap_features" keys
        aliased_features: Set of feature names that are aliased from candles (default: {"close", "volume", "CLOSE"})
    
    Returns:
        Tuple of:
            required_prebuilt_features: List of feature names that must be in prebuilt parquet
            aliased_features_list: List of feature names that are required by model but sourced from candles
    
    Example:
        >>> feature_meta = {"seq_features": ["close", "atr"], "snap_features": ["close", "volume", "rsi"]}
        >>> required, aliased = get_effective_required_prebuilt_features(feature_meta)
        >>> required
        ['atr', 'rsi']
        >>> aliased
        ['close', 'volume', 'CLOSE']
    """
    if aliased_features is None:
        # Default: close, volume, CLOSE are aliased from candles
        aliased_features = {"close", "volume", "CLOSE"}
    
    # Get all required features from feature_meta
    seq_features = feature_meta.get("seq_features", [])
    snap_features = feature_meta.get("snap_features", [])
    
    # Combine and deduplicate (preserve order)
    all_required = list(dict.fromkeys(seq_features + snap_features))
    
    # Separate into prebuilt-required and aliased
    required_prebuilt = []
    aliased_list = []
    
    for feat in all_required:
        if feat in aliased_features:
            aliased_list.append(feat)
        else:
            required_prebuilt.append(feat)
    
    # Sort for consistency
    required_prebuilt = sorted(required_prebuilt)
    aliased_list = sorted(aliased_list)
    
    log.debug(
        "[FEATURE_CONTRACT] Effective required prebuilt features: %d total, %d prebuilt-required, %d aliased",
        len(all_required), len(required_prebuilt), len(aliased_list)
    )
    
    return required_prebuilt, aliased_list
