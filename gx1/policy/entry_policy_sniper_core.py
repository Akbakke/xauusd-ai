"""
SNIPER Entry Policy Core - Session/volatility guards and threshold logic.

This is a core policy implementation without V9 dependencies.
Used by V10_CTX wrapper to avoid importing V9 modules in replay mode.

Design Philosophy:
  - Zero V9 dependencies: no imports from entry_v9_policy_* or runtime_v9
  - Same logic as entry_v9_policy_sniper.py, but with neutral names
  - EU/OVERLAP/US + (LOW ∪ MEDIUM ∪ HIGH) volatility guard
  - Side-aware probability thresholds (p_long OR p_short)
  - Optional profitable filter and trend_up soft gate
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging

# Use farm_guards (not V9-specific, safe to import)
from gx1.policy.farm_guards import sniper_guard_v1, _extract_session_vol_regime

logger = logging.getLogger(__name__)


@dataclass
class SniperPolicyParams:
    """Parameters for SNIPER entry policy."""
    min_prob_long: float = 0.68
    min_prob_short: float = 0.72
    enable_profitable_filter: bool = False
    min_prob_profitable: float = 0.0
    require_trend_up: bool = False  # Soft gate only
    allow_short: bool = False
    allow_high_vol: bool = True
    allow_extreme_vol: bool = False
    enabled: bool = True


def run_sniper_policy(
    df_signals: pd.DataFrame,
    config: Dict[str, Any],
    policy_flag_col_name: str = "entry_policy_sniper_v10_ctx",
    meta_model: Optional[object] = None,
    meta_feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run SNIPER entry policy - EU/London/NY sessions with HIGH volatility allowed.
    
    This is the core policy logic without V9 dependencies.
    Identical to entry_v9_policy_sniper but with neutral names and no V9 imports.
    
    Args:
        df_signals: DataFrame with columns:
            - "prob_long" (required): Entry prediction probability
            - "session" or "_v1_session_tag" or "session_tag" (optional): Session identifier
            - "vol_regime" or "brain_vol_regime" or "atr_regime_id" (optional): Volatility regime
            - Additional feature columns for meta-model prediction (optional, for logging)
        config: Config dict with policy section (typically "entry_v9_policy_sniper" for backward compat)
        policy_flag_col_name: Name of output policy flag column (default: "entry_policy_sniper_v10_ctx")
        meta_model: Optional trained meta-model for p_profitable prediction (computed but not used for filtering)
        meta_feature_cols: Optional list of feature columns for meta-model
    
    Returns:
        DataFrame with additional columns:
            - policy_flag_col_name (bool): Whether this row passes policy
            - "policy_score" (float): max(p_long, p_short) for side-aware scoring
            - "p_profitable" (float): Meta-model prediction (if available, for logging only)
            - "_policy_side" (str): "long", "short", "both", or "none"
    """
    # Extract policy config (support both "entry_v9_policy_sniper" for backward compat and generic "sniper_policy")
    policy_cfg = config.get("entry_v9_policy_sniper", {}) or config.get("sniper_policy", {})
    
    # Parse parameters
    params = SniperPolicyParams(
        min_prob_long=float(policy_cfg.get("min_prob_long", 0.68)),
        min_prob_short=float(policy_cfg.get("min_prob_short", 0.72)),
        enable_profitable_filter=policy_cfg.get("enable_profitable_filter", False),
        min_prob_profitable=float(policy_cfg.get("min_prob_profitable", 0.0)),
        require_trend_up=policy_cfg.get("require_trend_up", False),
        allow_short=policy_cfg.get("allow_short", False),
        allow_high_vol=policy_cfg.get("allow_high_vol", True),
        allow_extreme_vol=policy_cfg.get("allow_extreme_vol", False),
        enabled=policy_cfg.get("enabled", True),
    )
    
    if not params.enabled:
        # Policy disabled: all signals pass
        df_signals = df_signals.copy()
        df_signals[policy_flag_col_name] = True
        df_signals["policy_score"] = df_signals.get("prob_long", 0.5)
        df_signals["p_profitable"] = 0.0
        df_signals["_policy_side"] = "none"
        logger.info(f"[POLICY_SNIPER_CORE] Policy disabled - all signals pass")
        return df_signals
    
    df = df_signals.copy()
    n_original = len(df)
    
    logger.info(
        f"[POLICY_SNIPER_CORE] Policy params: min_prob_long={params.min_prob_long}, min_prob_short={params.min_prob_short}, "
        f"allow_short={params.allow_short}, allow_high_vol={params.allow_high_vol}, allow_extreme_vol={params.allow_extreme_vol}, "
        f"enable_profitable_filter={params.enable_profitable_filter}, require_trend_up={params.require_trend_up}"
    )
    
    # ============================================================================
    # STEP 0: SNIPER GUARD - Must be EU/OVERLAP/US + (LOW ∪ MEDIUM ∪ HIGH) BEFORE any other checks
    # ============================================================================
    
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
    
    # Apply SNIPER guard to each row
    n_before_guard = len(df)
    guard_passed_mask = pd.Series([False] * len(df), index=df.index)
    for idx in df.index:
        try:
            sniper_guard_v1(
                df.loc[idx], 
                context="policy_sniper_core", 
                allow_high_vol=params.allow_high_vol,
                allow_extreme_vol=params.allow_extreme_vol
            )
            guard_passed_mask.loc[idx] = True
        except AssertionError as e:
            logger.debug(f"[POLICY_SNIPER_CORE] Row {idx} rejected by sniper guard: {e}")
            guard_passed_mask.loc[idx] = False
    
    # Filter to only rows that passed guard
    df = df[guard_passed_mask].copy()
    n_after_guard = len(df)
    
    vol_regimes_allowed = "LOW+MEDIUM" + ("+HIGH" if params.allow_high_vol else "") + ("+EXTREME" if params.allow_extreme_vol else "")
    logger.info(
        f"[POLICY_SNIPER_CORE] SNIPER GUARD: {n_before_guard} -> {n_after_guard} rows passed (EU/OVERLAP/US+{vol_regimes_allowed} only)"
    )
    
    # If no rows remain after guard, reject all
    if len(df) == 0:
        logger.warning(
            "[POLICY_SNIPER_CORE] SNIPER GUARD: No rows passed EU/OVERLAP/US+(LOW|MEDIUM|HIGH) filter. Rejecting all signals."
        )
        result_df = df_signals.copy()
        result_df[policy_flag_col_name] = False
        result_df["policy_score"] = df_signals.get("prob_long", 0.0)
        result_df["p_profitable"] = 0.0
        result_df["_policy_side"] = "none"
        return result_df
    
    # ============================================================================
    # STEP 1: Optional trend filter (soft gate only - log but don't hard-require)
    # ============================================================================
    if params.require_trend_up:
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
            df_trend_up = df[df[trend_col] == "TREND_UP"].copy()
            n_after_trend = len(df_trend_up)
            logger.info(
                f"[POLICY_SNIPER_CORE] After TREND_UP filter (soft): {n_before_trend} -> {n_after_trend} rows"
            )
            # In v1: log trend but don't hard-filter (soft gate only)
        else:
            logger.warning("[POLICY_SNIPER_CORE] No trend regime column found, skipping trend filter")
    
    # ============================================================================
    # STEP 2: Apply side-aware probability thresholds (p_long OR p_short)
    # ============================================================================
    if "prob_long" not in df.columns:
        raise KeyError(f"df_signals must have 'prob_long' column for SNIPER policy")
    
    # Use prob_long as p_long (alias for consistency)
    if "p_long" not in df.columns:
        df["p_long"] = df["prob_long"]
    
    # Ensure prob_short exists
    if "prob_short" not in df.columns:
        if "p_short" in df.columns:
            df["prob_short"] = df["p_short"]
        else:
            df["prob_short"] = 1.0 - df["prob_long"]
            logger.debug("[POLICY_SNIPER_CORE] prob_short not found, computed as 1 - prob_long")
    
    if "p_short" not in df.columns:
        df["p_short"] = df["prob_short"]
    
    n_before_prob = len(df)
    
    # Side-aware filtering: allow long if p_long >= min_prob_long, allow short if p_short >= min_prob_short
    long_mask = df["p_long"] >= params.min_prob_long
    short_mask = df["p_short"] >= params.min_prob_short if params.allow_short else pd.Series([False] * len(df), index=df.index)
    
    # Signal passes if EITHER long OR short threshold is met
    prob_mask = long_mask | short_mask
    
    # Determine which side to use (for logging/analysis)
    df["_policy_side"] = "none"
    df.loc[long_mask & ~short_mask, "_policy_side"] = "long"
    df.loc[short_mask & ~long_mask, "_policy_side"] = "short"
    df.loc[long_mask & short_mask, "_policy_side"] = "both"
    
    n_after_prob = prob_mask.sum()
    n_long_signals = long_mask.sum()
    n_short_signals = short_mask.sum()
    
    logger.info(
        f"[POLICY_SNIPER_CORE] After side-aware thresholds (long>={params.min_prob_long}, short>={params.min_prob_short}): "
        f"{n_before_prob} -> {n_after_prob} signals (long={n_long_signals}, short={n_short_signals})"
    )
    
    # ============================================================================
    # STEP 3: Compute p_profitable (for logging/analysis, NOT for filtering)
    # ============================================================================
    p_profitable_computed = False
    if meta_model is not None and meta_feature_cols is not None:
        try:
            from gx1.policy.farm_meta_features import build_meta_feature_matrix
            X_meta = build_meta_feature_matrix(df, config=config)
            
            for train_col in meta_feature_cols:
                if train_col not in X_meta.columns:
                    X_meta[train_col] = 0.0
                    logger.warning(f"[POLICY_SNIPER_CORE] Feature '{train_col}' missing after mapping, using 0.0")
            
            X_meta = X_meta[[col for col in meta_feature_cols if col in X_meta.columns]]
            
            for col in X_meta.columns:
                if pd.api.types.is_numeric_dtype(X_meta[col]):
                    median_val = X_meta[col].median()
                    if pd.notna(median_val):
                        X_meta[col] = X_meta[col].fillna(median_val)
                    else:
                        X_meta[col] = X_meta[col].fillna(0.0)
            
            p_profitable = meta_model.predict_proba(X_meta.values)[:, 1]
            df["p_profitable"] = p_profitable
            p_profitable_computed = True
            
            logger.info(
                f"[POLICY_SNIPER_CORE] Computed p_profitable for {len(df)} candidates (for logging only): "
                f"mean={p_profitable.mean():.4f}, min={p_profitable.min():.4f}, max={p_profitable.max():.4f}"
            )
        except Exception as e:
            logger.warning(
                f"[POLICY_SNIPER_CORE] Failed to compute p_profitable (non-fatal): {e}. "
                f"Continuing without p_profitable logging."
            )
            df["p_profitable"] = 0.0
    else:
        df["p_profitable"] = 0.0
        logger.debug("[POLICY_SNIPER_CORE] Meta-model not available - p_profitable not computed")
    
    # ============================================================================
    # STEP 4: Apply filtering (p_long only, NOT p_profitable)
    # ============================================================================
    if params.enable_profitable_filter:
        if params.min_prob_profitable > 0.0 and p_profitable_computed:
            profitable_mask = df["p_profitable"] >= params.min_prob_profitable
            final_mask = prob_mask & profitable_mask
            logger.warning(
                f"[POLICY_SNIPER_CORE] WARNING: enable_profitable_filter=True is enabled. "
                f"Filtering on p_profitable>={params.min_prob_profitable}."
            )
        else:
            final_mask = prob_mask
    else:
        final_mask = prob_mask
    
    n_final = final_mask.sum()
    
    # Set policy flag (use provided column name)
    df[policy_flag_col_name] = final_mask.astype(bool)
    
    # Policy score: use max(p_long, p_short) for side-aware scoring
    df["policy_score"] = df[["p_long", "p_short"]].max(axis=1)
    
    # Final coverage
    final_coverage = df[policy_flag_col_name].sum() / n_original if n_original > 0 else 0.0
    
    # Count signals by side
    n_long_final = (df[policy_flag_col_name] & (df["_policy_side"].isin(["long", "both"]))).sum()
    n_short_final = (df[policy_flag_col_name] & (df["_policy_side"].isin(["short", "both"]))).sum()
    
    logger.info(
        f"[POLICY_SNIPER_CORE] ===== FINAL RESULT ====="
        f"coverage={final_coverage:.4f}, min_prob_long={params.min_prob_long}, min_prob_short={params.min_prob_short}, "
        f"allow_short={params.allow_short}, enable_profitable_filter={params.enable_profitable_filter}, "
        f"n_signals={df[policy_flag_col_name].sum()}/{n_original} (long={n_long_final}, short={n_short_final})"
    )
    
    # Merge back with original df_signals to preserve structure
    result_df = df_signals.copy()
    result_df[policy_flag_col_name] = False
    result_df["policy_score"] = df_signals.get("prob_long", 0.0)
    result_df["_policy_side"] = "none"
    
    # Always include p_profitable in result_df
    if "p_profitable" in df.columns and len(df) > 0:
        for idx in df.index:
            if idx in result_df.index:
                result_df.loc[idx, "p_profitable"] = df.loc[idx, "p_profitable"]
        missing_idx = result_df.index.difference(df.index)
        if len(missing_idx) > 0:
            result_df.loc[missing_idx, "p_profitable"] = 0.0
    else:
        result_df["p_profitable"] = 0.0
    
    # Update rows that passed policy
    if len(df) > 0:
        for idx in df.index:
            if idx in result_df.index:
                result_df.loc[idx, policy_flag_col_name] = df.loc[idx, policy_flag_col_name]
                result_df.loc[idx, "policy_score"] = df.loc[idx, "policy_score"]
                if "_policy_side" in df.columns:
                    result_df.loc[idx, "_policy_side"] = df.loc[idx, "_policy_side"]
    
    return result_df
