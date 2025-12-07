"""
FARM Meta-Model Feature Builder

Single source of truth for building meta-model features in runtime.
Matches the exact feature construction used in training.

Training dataset (farm_entry_dataset_v1.parquet) characteristics:
- 100% LONG-only trades (side_sign=1.0, is_long=1.0, is_short=0.0)
- p_long = entry_prob_long (mean=0.85, range=0.80-0.985)
- entry_prob_short (mean=0.15, range=0.015-0.20)
- atr_bps (mean=4.23, range=1.28-14.95)

This module ensures runtime features match training exactly.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Feature columns expected by meta-model (in order)
META_FEATURE_COLS = [
    "side_sign",
    "p_long",
    "entry_prob_long",
    "entry_prob_short",
    "atr_bps",
    "is_long",
    "is_short",
]


def build_meta_feature_vector(row: pd.Series, config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Build meta-model feature vector from runtime row.
    
    Matches exact feature construction from training:
    - Training was 100% LONG-only, so we set side_sign=1.0, is_long=1.0, is_short=0.0
    - p_long = entry_prob_long (from training dataset)
    - entry_prob_long/entry_prob_short come from ENTRY_V9 predictions
    - atr_bps comes from _v1_atr14_bps or atr_bps column
    
    Args:
        row: Runtime row (DataFrame row or Series) with ENTRY_V9 features
        config: Optional config dict with:
            - entry_v9_policy_farm_v2.allow_short (bool): Whether SHORT trades are allowed (default: False)
            - For now, SHORT support is infrastructure-only (not enabled)
    
    Returns:
        Dictionary with feature values (matching META_FEATURE_COLS order)
    """
    features = {}
    
    # ============================================================================
    # 1. side_sign, is_long, is_short: FARM training was LONG-only
    # ============================================================================
    # Training dataset: 100% LONG (side_sign=1.0, is_long=1.0, is_short=0.0)
    # For FARM_V2, we match training by always using LONG-only features
    # even though ENTRY_V9 might predict SHORT in some cases.
    # 
    # SHORT-infra (prepared but not enabled):
    #   - If allow_short == True (future): Could use side-aware mapping
    #   - For now: Always LONG-only to match training dataset
    allow_short = False
    if config is not None:
        policy_cfg = config.get("entry_v9_policy_farm_v2", {})
        allow_short = policy_cfg.get("allow_short", False)
    
    if allow_short:
        # TODO: Future SHORT support
        # - Could use side-aware meta-model (separate model for SHORT)
        # - Or use side-normalized features (side_sign = +1 for LONG, -1 for SHORT)
        # - For now, still use LONG-only features (matches training)
        logger.warning("[FARM_META_FEATURES] allow_short=True detected, but SHORT support not yet implemented. Using LONG-only features.")
        features["side_sign"] = 1.0
        features["is_long"] = 1.0
        features["is_short"] = 0.0
    else:
        # Current behavior: Always LONG-only (matches training)
        features["side_sign"] = 1.0  # Always LONG for FARM (matches training)
        features["is_long"] = 1.0    # Always LONG for FARM (matches training)
        features["is_short"] = 0.0   # Never SHORT for FARM (matches training)
    
    # ============================================================================
    # 2. p_long: Should match entry_prob_long from training
    # ============================================================================
    # Training: p_long = entry_prob_long (correlation = 1.0, same values)
    # Mean in training: 0.8496, range: 0.80-0.985
    # 
    # In runtime, we use entry_prob_long directly (or prob_long if entry_prob_long not available)
    if "entry_prob_long" in row.index:
        entry_prob_long = float(row["entry_prob_long"])
    elif "prob_long" in row.index:
        entry_prob_long = float(row["prob_long"])
    elif "p_long" in row.index:
        entry_prob_long = float(row["p_long"])
    else:
        logger.warning("No entry_prob_long/prob_long/p_long found, using 0.5")
        entry_prob_long = 0.5
    
    features["p_long"] = entry_prob_long
    features["entry_prob_long"] = entry_prob_long
    
    # ============================================================================
    # 3. entry_prob_short: From ENTRY_V9 predictions
    # ============================================================================
    # Training: mean=0.1504, range=0.015-0.20
    # In runtime, use entry_prob_short or prob_short
    if "entry_prob_short" in row.index:
        entry_prob_short = float(row["entry_prob_short"])
    elif "prob_short" in row.index:
        entry_prob_short = float(row["prob_short"])
    else:
        # Default: 1 - entry_prob_long (if we only have LONG prob)
        entry_prob_short = 1.0 - entry_prob_long
        logger.debug(f"No entry_prob_short found, computed as 1 - entry_prob_long = {entry_prob_short:.4f}")
    
    features["entry_prob_short"] = entry_prob_short
    
    # ============================================================================
    # 4. atr_bps: ATR in basis points
    # ============================================================================
    # Training: mean=4.23, range=1.28-14.95
    # Training dataset: atr_bps was computed as (atr / close) * 10000
    # 
    # In runtime, EntryFeatureBundle.compute_atr_bps() computes:
    #   atr_bps = (atr / close) * 10000
    # 
    # However, if runtime has atr_bps that's too high (e.g., 13920), it might be
    # computed incorrectly (e.g., _v1_atr14 * 10000 without dividing by close).
    # 
    # We need to recompute atr_bps correctly from _v1_atr14 and close:
    #   atr_bps = (_v1_atr14 / close) * 10000
    #
    # Priority:
    #   1. If _v1_atr14 and close are available, recompute: (_v1_atr14 / close) * 10000
    #   2. If atr_bps exists and is reasonable (< 100), use it directly
    #   3. Otherwise, try to infer from _v1_atr14 alone (fallback)
    
    if "_v1_atr14" in row.index and "close" in row.index:
        # Recompute atr_bps correctly: (atr / close) * 10000
        _v1_atr14 = float(row["_v1_atr14"])
        close = float(row["close"])
        if close > 0:
            atr_bps = (_v1_atr14 / close) * 10000.0
            logger.debug(f"Recomputed atr_bps from _v1_atr14 and close: {_v1_atr14:.6f} / {close:.2f} * 10000 = {atr_bps:.2f}")
        else:
            logger.warning("close is 0 or NaN, cannot compute atr_bps, using 0.0")
            atr_bps = 0.0
    elif "atr_bps" in row.index:
        atr_bps_raw = float(row["atr_bps"])
        # Check if atr_bps is reasonable (training range: 1.28-14.95)
        # If it's way too high (e.g., > 100), it's likely wrong and we should try to recompute
        if atr_bps_raw > 100.0:
            # Likely wrong - try to recompute from _v1_atr14 if available
            if "_v1_atr14" in row.index:
                _v1_atr14 = float(row["_v1_atr14"])
                # Try to infer close from entry_price or current price
                close = row.get("close", row.get("entry_price", row.get("price", 3350.0)))
                if close > 0:
                    atr_bps = (_v1_atr14 / close) * 10000.0
                    logger.warning(f"atr_bps={atr_bps_raw:.2f} seems too high, recomputed from _v1_atr14: {atr_bps:.2f}")
                else:
                    atr_bps = atr_bps_raw  # Fallback to original (wrong) value
                    logger.warning(f"atr_bps={atr_bps_raw:.2f} seems too high but cannot recompute (no close), using as-is")
            else:
                atr_bps = atr_bps_raw
                logger.warning(f"atr_bps={atr_bps_raw:.2f} seems too high but _v1_atr14 not available, using as-is")
        else:
            # Reasonable value, use directly
            atr_bps = atr_bps_raw
    elif "_v1_atr14" in row.index:
        # Fallback: try to infer from _v1_atr14 alone
        _v1_atr14 = float(row["_v1_atr14"])
        # If _v1_atr14 is very small (< 0.1), it might already be in bps (unlikely)
        # Otherwise, assume it's in pip and we need close to convert properly
        if _v1_atr14 < 0.1:
            # Likely already in bps (but this is unusual)
            atr_bps = _v1_atr14
            logger.warning(f"_v1_atr14={_v1_atr14:.6f} is very small, assuming it's already in bps (unusual)")
        else:
            # Need close to convert properly - try to infer
            close = row.get("close", row.get("entry_price", row.get("price", 3350.0)))
            if close > 0:
                atr_bps = (_v1_atr14 / close) * 10000.0
                logger.debug(f"Computed atr_bps from _v1_atr14 (inferred close={close:.2f}): {atr_bps:.2f}")
            else:
                logger.warning("Cannot compute atr_bps: _v1_atr14 available but no close/price, using 0.0")
                atr_bps = 0.0
    else:
        logger.warning("No atr_bps/_v1_atr14_bps/_v1_atr14 found, using 0.0")
        atr_bps = 0.0
    
    features["atr_bps"] = atr_bps
    
    # ============================================================================
    # Validate feature ranges (sanity checks)
    # ============================================================================
    if not (0.0 <= features["p_long"] <= 1.0):
        logger.warning(f"p_long out of range: {features['p_long']:.4f}")
    if not (0.0 <= features["entry_prob_short"] <= 1.0):
        logger.warning(f"entry_prob_short out of range: {features['entry_prob_short']:.4f}")
    if features["atr_bps"] < 0.0:
        logger.warning(f"atr_bps negative: {features['atr_bps']:.4f}")
    
    return features


def build_meta_feature_matrix(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Build meta-model feature matrix for multiple rows.
    
    Args:
        df: DataFrame with runtime rows
        config: Optional config dict
    
    Returns:
        DataFrame with feature columns (matching META_FEATURE_COLS)
    """
    feature_dicts = []
    
    for idx in df.index:
        row = df.loc[idx]
        features = build_meta_feature_vector(row, config)
        feature_dicts.append(features)
    
    feature_df = pd.DataFrame(feature_dicts, index=df.index)
    
    # Ensure columns are in correct order
    feature_df = feature_df[[col for col in META_FEATURE_COLS if col in feature_df.columns]]
    
    return feature_df

