"""
ENTRY_V9 Policy FARM_V2B

FARM Entry AI+ Policy V2B: p_long-driven entry (NO meta-filter)
- ASIA + (LOW ∪ MEDIUM) volatilitet (enforced by brutal guard V2)
- p_long >= min_prob_long (ENTRY_V9 probability threshold)
- p_profitable computed and logged, but NOT used for filtering
- No trend filter (require_trend_up: false)

Design Philosophy:
  - Remove meta-model as HARD gate (too restrictive)
  - Use p_long + regime as primary filter
  - Log p_profitable for offline analysis
  - Re-introduce meta-filter later as "soft gate" or "priority scorer"
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
from datetime import datetime
import os

from gx1.policy.farm_meta_features import (
    build_meta_feature_vector,
    build_meta_feature_matrix,
    META_FEATURE_COLS,
)

logger = logging.getLogger(__name__)


def apply_entry_v9_policy_farm_v2b(
    df_signals: pd.DataFrame,
    config: Dict[str, Any],
    meta_model: Optional[object] = None,
    meta_feature_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Apply ENTRY_V9 Policy FARM_V2B - p_long-driven entry (NO meta-filter).
    
    Args:
        df_signals: DataFrame with columns:
            - "prob_long" (required): ENTRY_V9 prediction probability
            - "session" or "_v1_session_tag" or "session_tag" (optional): Session identifier
            - "vol_regime" or "brain_vol_regime" or "atr_regime_id" (optional): Volatility regime
            - Additional feature columns for meta-model prediction (optional, for logging)
        config: Config dict with "entry_v9_policy_farm_v2b" section
        meta_model: Optional trained meta-model for p_profitable prediction (computed but not used for filtering)
        meta_feature_cols: Optional list of feature columns for meta-model
    
    Returns:
        DataFrame with additional columns:
            - "entry_v9_policy_farm_v2b" (bool): Whether this row passes policy
            - "policy_score" (float): prob_long (for future use)
            - "p_profitable" (float): Meta-model prediction (if available, for logging only)
    """
    policy_cfg = config.get("entry_v9_policy_farm_v2b", {})
    
    if not policy_cfg.get("enabled", False):
        # Policy disabled: all signals pass
        df_signals = df_signals.copy()
        df_signals["entry_v9_policy_farm_v2b"] = True
        df_signals["policy_score"] = df_signals.get("prob_long", 0.5)
        logger.info("[ENTRY_V9_POLICY_FARM_V2B] Policy disabled - all signals pass")
        return df_signals
    
    df = df_signals.copy()
    n_original = len(df)
    
    # Get parameters
    min_prob_long = float(policy_cfg.get("min_prob_long", 0.72))
    min_prob_short = float(policy_cfg.get("min_prob_short", 0.72))  # Default same as long
    enable_profitable_filter = policy_cfg.get("enable_profitable_filter", False)
    require_trend_up = policy_cfg.get("require_trend_up", False)
    allow_short = policy_cfg.get("allow_short", False)
    
    logger.info(
        f"[POLICY_FARM_V2B] Policy params: min_prob_long={min_prob_long}, min_prob_short={min_prob_short}, "
        f"allow_short={allow_short}, enable_profitable_filter={enable_profitable_filter}, require_trend_up={require_trend_up}"
    )
    
    # ============================================================================
    # STEP 0: BRUTAL FARM_V2 GATING - Must be ASIA + (LOW ∪ MEDIUM) BEFORE any other checks
    # This is the FIRST and ONLY filter - nothing else matters if not ASIA+(LOW|MEDIUM)
    # Uses centralized farm_brutal_guard_v2 for consistency
    # ============================================================================
    
    # Import centralized guard V2
    from gx1.policy.farm_guards import farm_brutal_guard_v2, get_farm_entry_metadata_v2
    
    # Get allow_medium_vol from config (allow_short already retrieved above)
    allow_medium_vol = policy_cfg.get("allow_medium_vol", True)
    
    # Apply brutal guard to ALL rows - filter out non-ASIA+(LOW|MEDIUM) BEFORE any other logic
    n_before_guard = len(df)
    
    # Normalize columns for guard
    if "session" not in df.columns:
        for col in ["_v1_session_tag", "session_tag"]:
            if col in df.columns:
                df["session"] = df[col]
                break
        if "session" not in df.columns and "session_id" in df.columns:
            session_map = {0: "EU", 1: "OVERLAP", 2: "US"}
            df["session"] = df["session_id"].map(session_map).fillna("UNKNOWN")
    
    if "vol_regime" not in df.columns:
        if "atr_regime" in df.columns:
            df["vol_regime"] = df["atr_regime"]
        elif "_v1_atr_regime_id" in df.columns or "atr_regime_id" in df.columns:
            ATR_ID_TO_VOL = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "EXTREME"}
            atr_id_col = "_v1_atr_regime_id" if "_v1_atr_regime_id" in df.columns else "atr_regime_id"
            df["vol_regime"] = df[atr_id_col].map(ATR_ID_TO_VOL).fillna("UNKNOWN")
    
    # Apply brutal guard V2 to each row
    guard_passed_mask = pd.Series([False] * len(df), index=df.index)
    for idx in df.index:
        try:
            farm_brutal_guard_v2(df.loc[idx], context="policy_v2b", allow_medium_vol=allow_medium_vol)
            guard_passed_mask.loc[idx] = True
        except AssertionError as e:
            logger.debug(f"[POLICY_FARM_V2B] Row {idx} rejected by brutal guard V2: {e}")
            guard_passed_mask.loc[idx] = False
    
    # Filter to only rows that passed guard
    df = df[guard_passed_mask].copy()
    n_after_guard = len(df)
    
    vol_regimes_allowed = "LOW" + ("+MEDIUM" if allow_medium_vol else "")
    logger.info(
        f"[POLICY_FARM_V2B] BRUTAL GUARD V2: {n_before_guard} -> {n_after_guard} rows passed (ASIA+{vol_regimes_allowed} only)"
    )
    
    # If no rows remain after guard, reject all
    if len(df) == 0:
        logger.warning(
            "[POLICY_FARM_V2B] BRUTAL GUARD: No rows passed ASIA+(LOW|MEDIUM) filter. Rejecting all signals."
        )
        result_df = df_signals.copy()
        result_df["entry_v9_policy_farm_v2b"] = False
        result_df["policy_score"] = df_signals.get("prob_long", 0.0)
        result_df["p_profitable"] = 0.0
        return result_df
    
    # ============================================================================
    # STEP 1: Optional trend filter (if require_trend_up is True)
    # ============================================================================
    if require_trend_up:
        trend_col = None
        for col in ["trend_regime", "brain_trend_regime"]:
            if col in df.columns:
                trend_col = col
                break
        
        if trend_col is None:
            if "trend_regime_tf24h" in df.columns:
                def map_trend(val):
                    try:
                        v = float(val)
                        if v > 0.001:
                            return "TREND_UP"
                        elif v < -0.001:
                            return "TREND_DOWN"
                        else:
                            return "RANGE"
                    except:
                        return "UNKNOWN"
                df["trend_regime"] = df["trend_regime_tf24h"].apply(map_trend)
                trend_col = "trend_regime"
        
        if trend_col:
            n_before_trend = len(df)
            df = df[df[trend_col] == "TREND_UP"].copy()
            n_after_trend = len(df)
            logger.info(
                f"[POLICY_FARM_V2B] After TREND_UP filter: {n_before_trend} -> {n_after_trend} rows"
            )
        else:
            logger.warning("[POLICY_FARM_V2B] No trend regime column found, skipping trend filter")
    
    # ============================================================================
    # STEP 2: Apply side-aware probability thresholds (p_long OR p_short)
    # ============================================================================
    if "prob_long" not in df.columns:
        raise KeyError("df_signals must have 'prob_long' column for ENTRY_V9_POLICY_FARM_V2B")
    
    # Use prob_long as p_long (alias for consistency)
    if "p_long" not in df.columns:
        df["p_long"] = df["prob_long"]
    
    # Ensure prob_short exists (may be missing in some datasets)
    if "prob_short" not in df.columns:
        if "p_short" in df.columns:
            df["prob_short"] = df["p_short"]
        else:
            # If prob_short missing, compute as 1 - prob_long (assuming normalized probabilities)
            df["prob_short"] = 1.0 - df["prob_long"]
            logger.debug("[POLICY_FARM_V2B] prob_short not found, computed as 1 - prob_long")
    
    if "p_short" not in df.columns:
        df["p_short"] = df["prob_short"]
    
    n_before_prob = len(df)
    
    # Side-aware filtering: allow long if p_long >= min_prob_long, allow short if p_short >= min_prob_short
    long_mask = df["p_long"] >= min_prob_long
    short_mask = df["p_short"] >= min_prob_short if allow_short else pd.Series([False] * len(df), index=df.index)
    
    # Signal passes if EITHER long OR short threshold is met
    prob_mask = long_mask | short_mask
    
    # Determine which side to use (for logging/analysis)
    df["_policy_side"] = "none"
    df.loc[long_mask & ~short_mask, "_policy_side"] = "long"
    df.loc[short_mask & ~long_mask, "_policy_side"] = "short"
    df.loc[long_mask & short_mask, "_policy_side"] = "both"  # Both thresholds met (use max)
    
    n_after_prob = prob_mask.sum()
    n_long_signals = long_mask.sum()
    n_short_signals = short_mask.sum()
    
    logger.info(
        f"[POLICY_FARM_V2B] After side-aware thresholds (long>={min_prob_long}, short>={min_prob_short}): "
        f"{n_before_prob} -> {n_after_prob} signals (long={n_long_signals}, short={n_short_signals})"
    )
    
    # ============================================================================
    # STEP 3: Compute p_profitable (for logging/analysis, NOT for filtering)
    # ============================================================================
    
    # Try to compute p_profitable if meta-model is available (for logging)
    p_profitable_computed = False
    if meta_model is not None and meta_feature_cols is not None:
        try:
            # Use centralized feature builder (matches training exactly)
            X_meta = build_meta_feature_matrix(df, config=config)
            
            # Ensure all required features are present (in correct order)
            for train_col in meta_feature_cols:
                if train_col not in X_meta.columns:
                    X_meta[train_col] = 0.0
                    logger.warning(f"[POLICY_FARM_V2B] Feature '{train_col}' missing after mapping, using 0.0")

            # Reorder columns to match meta_feature_cols order
            X_meta = X_meta[[col for col in meta_feature_cols if col in X_meta.columns]]

            # Fill NaNs with median (robust to missing data)
            for col in X_meta.columns:
                if pd.api.types.is_numeric_dtype(X_meta[col]):
                    median_val = X_meta[col].median()
                    if pd.notna(median_val):
                        X_meta[col] = X_meta[col].fillna(median_val)
                    else:
                        X_meta[col] = X_meta[col].fillna(0.0)
            
            # Predict probabilities for ALL candidates (for logging only)
            p_profitable = meta_model.predict_proba(X_meta.values)[:, 1]
            df["p_profitable"] = p_profitable
            p_profitable_computed = True
            
            logger.info(
                f"[POLICY_FARM_V2B] Computed p_profitable for {len(df)} candidates (for logging only): "
                f"mean={p_profitable.mean():.4f}, min={p_profitable.min():.4f}, max={p_profitable.max():.4f}"
            )
        except Exception as e:
            logger.warning(
                f"[POLICY_FARM_V2B] Failed to compute p_profitable (non-fatal): {e}. "
                f"Continuing without p_profitable logging."
            )
            df["p_profitable"] = 0.0
    else:
        # Meta-model not available - set p_profitable to 0.0
        df["p_profitable"] = 0.0
        logger.debug("[POLICY_FARM_V2B] Meta-model not available - p_profitable not computed")
    
    # ============================================================================
    # STEP 4: Apply filtering (p_long only, NOT p_profitable)
    # ============================================================================
    
    if enable_profitable_filter:
        # This should not be used in V2B, but we support it for future flexibility
        min_prob_profitable = float(policy_cfg.get("min_prob_profitable", 0.0))
        if min_prob_profitable > 0.0 and p_profitable_computed:
            profitable_mask = df["p_profitable"] >= min_prob_profitable
            final_mask = prob_mask & profitable_mask
            logger.warning(
                f"[POLICY_FARM_V2B] WARNING: enable_profitable_filter=True is enabled. "
                f"Filtering on p_profitable>={min_prob_profitable}. "
                f"This is not recommended for V2B (use V2 instead)."
            )
        else:
            final_mask = prob_mask
    else:
        # V2B: Only use p_long filter, NOT p_profitable
        final_mask = prob_mask
    
    n_final = final_mask.sum()
    
    # Set policy flag
    df["entry_v9_policy_farm_v2b"] = final_mask.astype(bool)
    
    # Policy score: use max(p_long, p_short) for side-aware scoring
    df["policy_score"] = df[["p_long", "p_short"]].max(axis=1)
    
    # Final coverage
    final_coverage = df["entry_v9_policy_farm_v2b"].sum() / n_original if n_original > 0 else 0.0
    
    # Count signals by side
    n_long_final = (df["entry_v9_policy_farm_v2b"] & (df["_policy_side"].isin(["long", "both"]))).sum()
    n_short_final = (df["entry_v9_policy_farm_v2b"] & (df["_policy_side"].isin(["short", "both"]))).sum()
    
    logger.info(
        f"[POLICY_FARM_V2B] ===== FINAL RESULT ====="
        f"coverage={final_coverage:.4f}, min_prob_long={min_prob_long}, min_prob_short={min_prob_short}, "
        f"allow_short={allow_short}, enable_profitable_filter={enable_profitable_filter}, "
        f"n_signals={df['entry_v9_policy_farm_v2b'].sum()}/{n_original} (long={n_long_final}, short={n_short_final})"
    )
    
    # CRITICAL: Return same structure as FARM_V1/V2 - must preserve all original rows
    # Merge back with original df_signals to preserve structure
    result_df = df_signals.copy()
    result_df["entry_v9_policy_farm_v2b"] = False
    result_df["policy_score"] = df_signals.get("prob_long", 0.0)
    
    # CRITICAL: Always include p_profitable in result_df (even if signal rejected)
    # This is needed for logging/analysis even when trade is not created
    if "p_profitable" in df.columns and len(df) > 0:
        # Copy p_profitable to result_df for all rows (even rejected ones)
        for idx in df.index:
            if idx in result_df.index:
                result_df.loc[idx, "p_profitable"] = df.loc[idx, "p_profitable"]
        # If result_df has rows not in df, fill p_profitable with NaN or 0.0
        missing_idx = result_df.index.difference(df.index)
        if len(missing_idx) > 0:
            result_df.loc[missing_idx, "p_profitable"] = 0.0
    else:
        # No p_profitable computed - set to 0.0
        result_df["p_profitable"] = 0.0
    
    # Update rows that passed policy (including side information)
    if len(df) > 0:
        for idx in df.index:
            if idx in result_df.index:
                result_df.loc[idx, "entry_v9_policy_farm_v2b"] = df.loc[idx, "entry_v9_policy_farm_v2b"]
                result_df.loc[idx, "policy_score"] = df.loc[idx, "policy_score"]
                # Add side information if available
                if "_policy_side" in df.columns:
                    result_df.loc[idx, "_policy_side"] = df.loc[idx, "_policy_side"]
    
    return result_df


def get_entry_policy_v9_farm_v2b():
    """
    Get ENTRY_V9_FARM_V2B policy function.
    
    Returns:
        Callable that applies FARM_V2B policy
    """
    return apply_entry_v9_policy_farm_v2b

