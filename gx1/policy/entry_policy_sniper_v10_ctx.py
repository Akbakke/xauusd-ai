"""
ENTRY_V10_CTX Policy SNIPER - Wrapper for V10_CTX identity.

SNIPER Entry Policy wrapper for V10_CTX using core policy logic (no V9 dependencies).

This wrapper provides V10_CTX identity while using the core policy implementation
that has zero V9 dependencies. All thresholds, gates, and parameters are identical
to the original V9 policy, but without any V9 imports.

Design Philosophy:
  - Zero V9 dependencies: uses entry_policy_sniper_core (no V9 imports)
  - Explicit V10_CTX identity for replay provenance/auditing
  - Same entry model (V10_CTX instead of V9), same policy thresholds
  - Replay-safe: never imports V9 modules, even indirectly
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

# DEL 1: Import core policy (no V9 dependencies)
from gx1.policy.entry_policy_sniper_core import run_sniper_policy

logger = logging.getLogger(__name__)


def apply_entry_policy_sniper_v10_ctx(
    df_signals: pd.DataFrame,
    config: Dict[str, Any],
    meta_model: Optional[object] = None,
    meta_feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Apply ENTRY_V10_CTX Policy SNIPER - EU/London/NY sessions with HIGH volatility allowed.
    
    This wrapper uses core policy logic (no V9 dependencies) with V10_CTX identity.
    All scoring, thresholds, and gates are identical to the original V9 policy,
    but implemented without any V9 imports.
    
    Args:
        df_signals: DataFrame with columns:
            - "prob_long" (required): Entry prediction probability (from V10_CTX model)
            - "session" or "_v1_session_tag" or "session_tag" (optional): Session identifier
            - "vol_regime" or "brain_vol_regime" or "atr_regime_id" (optional): Volatility regime
            - Additional feature columns for meta-model prediction (optional, for logging)
        config: Config dict with "entry_v9_policy_sniper" section (backward compatible config key)
        meta_model: Optional trained meta-model for p_profitable prediction (computed but not used for filtering)
        meta_feature_cols: Optional list of feature columns for meta-model
    
    Returns:
        DataFrame with additional columns:
            - "entry_policy_sniper_v10_ctx" (bool): Whether this row passes policy
            - "policy_score" (float): max(p_long, p_short) for side-aware scoring
            - "p_profitable" (float): Meta-model prediction (if available, for logging only)
            - "_policy_side" (str): "long", "short", "both", or "none"
    """
    # DEL 1: Use core policy (no V9 dependencies) - safe for replay mode
    df_result = run_sniper_policy(
        df_signals=df_signals,
        config=config,
        policy_flag_col_name="entry_policy_sniper_v10_ctx",  # V10_CTX identity
        meta_model=meta_model,
        meta_feature_cols=meta_feature_cols,
    )
    
    logger.debug(
        f"[POLICY_SNIPER_V10_CTX] Applied V10_CTX policy (via core): "
        f"{df_result['entry_policy_sniper_v10_ctx'].sum()}/{len(df_result)} signals passed"
    )
    
    return df_result


def get_entry_policy_sniper_v10_ctx():
    """
    Get ENTRY_V10_CTX_SNIPER policy function.
    
    Returns:
        Callable that applies SNIPER policy with V10_CTX identity
    """
    return apply_entry_policy_sniper_v10_ctx
