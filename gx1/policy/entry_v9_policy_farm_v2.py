"""
ENTRY_V9 Policy FARM_V2

FARM Entry AI+ Policy V2: Meta-model enhanced entry selection
- Kun ASIA + LOW volatilitet (enforced by brutal guard)
- p_long >= min_prob_long (ENTRY_V9 probability threshold)
- p_profitable >= min_prob_profitable (Meta-model prediction threshold)
- No trend filter (require_trend_up: false)

Selected from offline policy simulation (Phase 3).
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

# Global feature logging buffer (thread-safe for single-threaded replay)
_feature_log_buffer: List[Dict[str, Any]] = []
_feature_log_count = 0
_feature_log_flush_interval = 500  # Flush every N candidates
_feature_log_max_rows = 10000  # Hard cap per run
_feature_log_output_dir = Path("gx1/wf_runs/FARM_ENTRY_FEATURE_PROBE")
_feature_log_enabled = os.getenv("FARM_FEATURE_LOGGING", "false").lower() == "true"

# Global policy debug counters (for instrumentation)
_policy_debug_call_count = 0
_policy_debug_log_interval = 100  # Log every N calls


def apply_entry_v9_policy_farm_v2(
    df_signals: pd.DataFrame,
    config: Dict[str, Any],
    meta_model: Optional[object] = None,
    meta_feature_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Apply ENTRY_V9 Policy FARM_V2 - Meta-model enhanced entry selection.
    
    Args:
        df_signals: DataFrame with columns:
            - "prob_long" (required): ENTRY_V9 prediction probability
            - "session" or "_v1_session_tag" or "session_tag" (optional): Session identifier
            - "vol_regime" or "brain_vol_regime" or "atr_regime_id" (optional): Volatility regime
            - Additional feature columns for meta-model prediction
        config: Config dict with "entry_v9_policy_farm_v2" section
        meta_model: Optional trained meta-model for p_profitable prediction
        meta_feature_cols: Optional list of feature columns for meta-model
    
    Returns:
        DataFrame with additional columns:
            - "entry_v9_policy_farm_v2" (bool): Whether this row passes policy
            - "policy_score" (float): prob_long (for future use)
            - "p_profitable" (float): Meta-model prediction (if available)
    """
    policy_cfg = config.get("entry_v9_policy_farm_v2", {})
    
    if not policy_cfg.get("enabled", False):
        # Policy disabled: all signals pass
        df_signals = df_signals.copy()
        df_signals["entry_v9_policy_farm_v2"] = True
        df_signals["policy_score"] = df_signals.get("prob_long", 0.5)
        logger.info("[ENTRY_V9_POLICY_FARM_V2] Policy disabled - all signals pass")
        return df_signals
    
    df = df_signals.copy()
    n_original = len(df)
    
    # Get parameters
    min_prob_long = float(policy_cfg.get("min_prob_long", 0.75))
    min_prob_profitable = float(policy_cfg.get("min_prob_profitable", 0.50))
    require_trend_up = policy_cfg.get("require_trend_up", False)
    
    logger.info(
        f"[POLICY_FARM_V2] Policy params: min_prob_long={min_prob_long}, "
        f"min_prob_profitable={min_prob_profitable}, require_trend_up={require_trend_up}"
    )
    
    # ============================================================================
    # STEP 0: BRUTAL FARM_V2 GATING - Must be ASIA + (LOW ∪ MEDIUM) BEFORE any other checks
    # This is the FIRST and ONLY filter - nothing else matters if not ASIA+(LOW|MEDIUM)
    # Uses centralized farm_brutal_guard_v2 for consistency
    # ============================================================================
    
    # Import centralized guard V2
    from gx1.policy.farm_guards import farm_brutal_guard_v2, get_farm_entry_metadata_v2
    
    # Get allow_medium_vol from config
    allow_medium_vol = policy_cfg.get("allow_medium_vol", True)
    allow_short = policy_cfg.get("allow_short", False)
    
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
            farm_brutal_guard_v2(df.loc[idx], context="policy_v2", allow_medium_vol=allow_medium_vol)
            guard_passed_mask.loc[idx] = True
        except AssertionError as e:
            logger.debug(f"[POLICY_FARM_V2] Row {idx} rejected by brutal guard V2: {e}")
            guard_passed_mask.loc[idx] = False
    
    # Filter to only rows that passed guard
    df = df[guard_passed_mask].copy()
    n_after_guard = len(df)
    
    vol_regimes_allowed = "LOW" + ("+MEDIUM" if allow_medium_vol else "")
    logger.info(
        f"[POLICY_FARM_V2] BRUTAL GUARD V2: {n_before_guard} -> {n_after_guard} rows passed (ASIA+{vol_regimes_allowed} only)"
    )
    
    # Initialize debug counters
    debug_stats = {
        "n_rows_input": n_after_guard,  # After guard (ASIA+LOW/MED)
        "n_pass_session_vol": n_after_guard,  # Should match n_rows_input
        "n_pass_p_long": 0,
        "n_pass_p_profitable": 0,
        "n_pass_both": 0,
        "p_long_mean": 0.0,
        "p_long_p50": 0.0,
        "p_long_p90": 0.0,
        "p_profitable_mean": 0.0,
        "p_profitable_p50": 0.0,
        "p_profitable_p90": 0.0,
        "session_dist": {},
        "vol_dist": {},
    }
    
    # Capture session/vol distribution
    if len(df) > 0:
        if "session" in df.columns:
            debug_stats["session_dist"] = df["session"].value_counts().to_dict()
        if "vol_regime" in df.columns:
            debug_stats["vol_dist"] = df["vol_regime"].value_counts(dropna=False).to_dict()
    
    # If no rows remain after guard, reject all
    if len(df) == 0:
        logger.warning(
            "[POLICY_FARM_V2] BRUTAL GUARD: No rows passed ASIA+LOW filter. Rejecting all signals."
        )
        result_df = df_signals.copy()
        result_df["entry_v9_policy_farm_v2"] = False
        result_df["policy_score"] = df_signals.get("prob_long", 0.0)
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
                f"[POLICY_FARM_V2] After TREND_UP filter: {n_before_trend} -> {n_after_trend} rows"
            )
        else:
            logger.warning("[POLICY_FARM_V2] No trend regime column found, skipping trend filter")
    
    # ============================================================================
    # STEP 2: Apply p_long threshold
    # ============================================================================
    if "prob_long" not in df.columns:
        raise KeyError("df_signals must have 'prob_long' column for ENTRY_V9_POLICY_FARM_V2")
    
    # Use prob_long as p_long (alias for consistency)
    if "p_long" not in df.columns:
        df["p_long"] = df["prob_long"]
    
    n_before_prob = len(df)
    prob_mask = df["p_long"] >= min_prob_long
    n_after_prob = prob_mask.sum()
    
    # Update debug stats
    if len(df) > 0:
        debug_stats["p_long_mean"] = float(df["p_long"].mean())
        debug_stats["p_long_p50"] = float(df["p_long"].median())
        debug_stats["p_long_p90"] = float(df["p_long"].quantile(0.90))
        debug_stats["n_pass_p_long"] = int(n_after_prob)
    
    logger.info(
        f"[POLICY_FARM_V2] After p_long>={min_prob_long}: {n_before_prob} -> {n_after_prob} signals"
    )
    
    # ============================================================================
    # STEP 3: Apply p_profitable threshold (meta-model prediction)
    # ============================================================================
    
    # FARM_V2 REQUIRES meta-model - no silent fallback
    if meta_model is None or meta_feature_cols is None:
        raise RuntimeError(
            "FARM_V2 requires meta-model for p_profitable prediction. "
            "Meta-model is missing or not loaded. "
            "FARM_V2 cannot run without meta-model."
        )
    
    # Generate p_profitable predictions for ALL candidates (before filtering)
    # This allows logging stats even if policy rejects all signals
    try:
        # Use centralized feature builder (matches training exactly)
        # This ensures runtime features match training dataset construction:
        # - Training was 100% LONG-only (side_sign=1.0, is_long=1.0, is_short=0.0)
        # - p_long = entry_prob_long (mean=0.85, range=0.80-0.985)
        # - entry_prob_short (mean=0.15, range=0.015-0.20)
        # - atr_bps (mean=4.23, range=1.28-14.95)
        X_meta = build_meta_feature_matrix(df, config=config)
        
        # Ensure all required features are present (in correct order)
        for train_col in meta_feature_cols:
            if train_col not in X_meta.columns:
                X_meta[train_col] = 0.0
                logger.warning(f"[POLICY_FARM_V2] Feature '{train_col}' missing after mapping, using 0.0")

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
        
        # ============================================================================
        # FEATURE LOGGING: Log features as they are sent to the model
        # ============================================================================
        if _feature_log_enabled and len(X_meta) > 0:
            _log_runtime_features(X_meta, df, config)
        
        # Predict probabilities for ALL candidates (before any filtering)
        p_profitable = meta_model.predict_proba(X_meta.values)[:, 1]
        df["p_profitable"] = p_profitable
        
        # Update debug stats with p_profitable distribution
        if len(df) > 0:
            debug_stats["p_profitable_mean"] = float(p_profitable.mean())
            debug_stats["p_profitable_p50"] = float(np.percentile(p_profitable, 50))
            debug_stats["p_profitable_p90"] = float(np.percentile(p_profitable, 90))
        
        # Log stats for input candidates (before policy filtering)
        logger.info(
            f"[POLICY_FARM_V2] Generated p_profitable for {len(df)} input candidates: "
            f"mean={p_profitable.mean():.4f}, min={p_profitable.min():.4f}, max={p_profitable.max():.4f}, "
            f"p>={min_prob_profitable}={(p_profitable >= min_prob_profitable).sum()}/{len(p_profitable)}"
        )
    except Exception as e:
        raise RuntimeError(
            f"FARM_V2 failed to generate p_profitable predictions: {e}. "
            f"FARM_V2 cannot run without p_profitable."
        ) from e
    
    # Apply p_profitable threshold
    n_before_profitable = prob_mask.sum()
    profitable_mask = df["p_profitable"] >= min_prob_profitable
    final_mask = prob_mask & profitable_mask
    n_after_profitable = final_mask.sum()
    
    # Update debug stats
    debug_stats["n_pass_p_profitable"] = int(profitable_mask.sum())
    debug_stats["n_pass_both"] = int(n_after_profitable)
    
    logger.info(
        f"[POLICY_FARM_V2] After p_profitable>={min_prob_profitable}: "
        f"{n_before_profitable} -> {n_after_profitable} signals"
    )
    
    # Set policy flag
    df["entry_v9_policy_farm_v2"] = final_mask.astype(bool)
    df["policy_score"] = df["p_long"]
    
    # Final coverage
    final_coverage = df["entry_v9_policy_farm_v2"].sum() / n_original if n_original > 0 else 0.0
    
    logger.info(
        f"[POLICY_FARM_V2] ===== FINAL RESULT ====="
        f"coverage={final_coverage:.4f}, min_prob_long={min_prob_long}, "
        f"min_prob_profitable={min_prob_profitable}, "
        f"n_signals={df['entry_v9_policy_farm_v2'].sum()}/{n_original}"
    )
    
    # Periodic debug logging (every N calls)
    global _policy_debug_call_count
    _policy_debug_call_count += 1
    
    if _policy_debug_call_count % _policy_debug_log_interval == 0:
        # Format session/vol distributions (limit to top 3)
        session_str = ", ".join([f"{k}:{v}" for k, v in list(debug_stats["session_dist"].items())[:3]])
        vol_str = ", ".join([f"{k}:{v}" for k, v in list(debug_stats["vol_dist"].items())[:3]])
        
        logger.info(
            f"[FARM_V2_POLICY_DEBUG] call={_policy_debug_call_count} | "
            f"input={debug_stats['n_rows_input']}, "
            f"pass_p_long={debug_stats['n_pass_p_long']}, "
            f"pass_p_prof={debug_stats['n_pass_p_profitable']}, "
            f"pass_both={debug_stats['n_pass_both']}, "
            f"p_long_mean={debug_stats['p_long_mean']:.3f}, "
            f"p_prof_mean={debug_stats['p_profitable_mean']:.3f}, "
            f"session_dist=[{session_str}], "
            f"vol_dist=[{vol_str}]"
        )
    
    # CRITICAL: Return same structure as FARM_V1 - must preserve all original rows
    # Merge back with original df_signals to preserve structure
    result_df = df_signals.copy()
    result_df["entry_v9_policy_farm_v2"] = False
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
        # No p_profitable computed - this should not happen
        result_df["p_profitable"] = 0.0
    
    # Update rows that passed policy
    if len(df) > 0:
        for idx in df.index:
            if idx in result_df.index:
                result_df.loc[idx, "entry_v9_policy_farm_v2"] = df.loc[idx, "entry_v9_policy_farm_v2"]
                result_df.loc[idx, "policy_score"] = df.loc[idx, "policy_score"]
    
    return result_df


def _log_runtime_features(
    X_meta: pd.DataFrame,
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> None:
    """
    Log runtime features to buffer for later comparison with training data.
    
    Only logs for ASIA+LOW candidates (FARM regime).
    Flushes buffer periodically to parquet file.
    """
    global _feature_log_buffer, _feature_log_count
    
    # Only log if enabled and under cap
    if not _feature_log_enabled or _feature_log_count >= _feature_log_max_rows:
        return
    
    # Extract metadata
    run_id = config.get("run_id", "unknown")
    
    # For each candidate in X_meta
    for idx in X_meta.index:
        if _feature_log_count >= _feature_log_max_rows:
            break
        
        # Get feature vector (as sent to model)
        feature_dict = X_meta.loc[idx].to_dict()
        
        # Add metadata
        feature_dict["ts"] = df.loc[idx].get("time", df.loc[idx].get("ts", pd.NaT)) if idx in df.index else pd.NaT
        feature_dict["run_id"] = run_id
        
        # Add session/vol_regime for sanity
        if idx in df.index:
            row = df.loc[idx]
            feature_dict["session"] = row.get("session", row.get("_v1_session_tag", row.get("session_tag", "UNKNOWN")))
            feature_dict["vol_regime"] = row.get("vol_regime", row.get("atr_regime", row.get("brain_vol_regime", "UNKNOWN")))
        else:
            feature_dict["session"] = "UNKNOWN"
            feature_dict["vol_regime"] = "UNKNOWN"
        
        # Only log ASIA+LOW (FARM regime)
        if feature_dict.get("session") == "ASIA" and feature_dict.get("vol_regime") == "LOW":
            _feature_log_buffer.append(feature_dict)
            _feature_log_count += 1
    
    # Flush buffer if interval reached
    if len(_feature_log_buffer) >= _feature_log_flush_interval:
        _flush_feature_log_buffer()


def _flush_feature_log_buffer() -> None:
    """Flush feature log buffer to parquet file."""
    global _feature_log_buffer
    
    if len(_feature_log_buffer) == 0:
        return
    
    try:
        # Create output directory
        _feature_log_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = _feature_log_output_dir / f"runtime_features_{timestamp}.parquet"
        
        # Convert to DataFrame
        df_log = pd.DataFrame(_feature_log_buffer)
        
        # Save to parquet
        df_log.to_parquet(output_path, index=False)
        
        logger.info(f"[FEATURE_LOG] Flushed {len(_feature_log_buffer)} feature vectors to {output_path}")
        
        # Clear buffer
        _feature_log_buffer = []
    except Exception as e:
        logger.error(f"[FEATURE_LOG] Failed to flush buffer: {e}")


def flush_feature_log_final() -> None:
    """Flush remaining feature log buffer (call at end of run)."""
    global _feature_log_buffer
    if len(_feature_log_buffer) > 0:
        _flush_feature_log_buffer()


def get_entry_policy_v9_farm_v2():
    """
    Get ENTRY_V9_FARM_V2 policy function.
    
    Returns:
        Callable that applies FARM_V2 policy
    """
    return apply_entry_v9_policy_farm_v2

