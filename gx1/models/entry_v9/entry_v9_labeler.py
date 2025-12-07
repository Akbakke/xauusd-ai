"""
ENTRY_V9 Label Generator

Generates three labels for multi-task learning:
1. y_direction: Binary direction label (1=long, 0=short/neutral)
2. y_early_move: Binary label (1 if MFE before MAE within horizon, 0 otherwise)
3. y_quality_score: Regression label (normalized MFE-MAE gap, clipped to [-1, 1])
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def generate_entry_v9_labels(
    df: pd.DataFrame,
    horizon_bars: int = 10,
    mfe_col: str = "MFE_bps",
    mae_col: str = "MAE_bps",
    first_hit_col: Optional[str] = "first_hit",
    y_direction_col: Optional[str] = "y",
    target_mfe_bps: float = 20.0,
) -> pd.DataFrame:
    """
    Generate all three labels for ENTRY_V9 training.

    Args:
        df: DataFrame with MFE_bps, MAE_bps, and optionally first_hit, y columns
        horizon_bars: Number of bars to check for early_move (default: 10)
        mfe_col: Column name for MFE in bps
        mae_col: Column name for MAE in bps
        first_hit_col: Column name for first_hit (if exists, uses it; otherwise computes)
        y_direction_col: Column name for direction label (if exists, uses it; otherwise computes from MFE/MAE)
        target_mfe_bps: Target MFE in bps for direction label (default: 20.0)

    Returns:
        DataFrame with added columns:
            - y_direction: [0, 1] (1 if MFE >= target_mfe_bps and MFE > MAE, else 0)
            - y_early_move: [0, 1] (1 if MFE before MAE within horizon_bars)
            - y_quality_score: [-1, 1] (normalized (MFE - MAE) / 100.0, clipped)
    """
    df = df.copy()

    # Ensure MFE and MAE columns exist
    if mfe_col not in df.columns:
        raise KeyError(f"Missing required column: {mfe_col}")
    if mae_col not in df.columns:
        raise KeyError(f"Missing required column: {mae_col}")

    # ============================================================
    # 1. y_direction: Binary direction label
    # ============================================================
    if y_direction_col and y_direction_col in df.columns:
        # Use existing direction label if available
        df["y_direction"] = (df[y_direction_col] > 0.5).astype(int)
    else:
        # Compute from MFE/MAE: 1 if MFE >= target and MFE > MAE
        df["y_direction"] = (
            (df[mfe_col] >= target_mfe_bps) & (df[mfe_col] > df[mae_col])
        ).astype(int)

    # ============================================================
    # 2. y_early_move: Binary label (MFE before MAE within horizon)
    # ============================================================
    if first_hit_col and first_hit_col in df.columns:
        # Use existing first_hit column if available
        # first_hit == "MFE" means MFE happened before MAE
        df["y_early_move"] = (df[first_hit_col] == "MFE").astype(int)
    else:
        # Compute from MFE/MAE: 1 if MFE > MAE (simplified, assumes MFE before MAE if MFE > MAE)
        # In practice, we'd need first_hit information, but for now use MFE > MAE as proxy
        df["y_early_move"] = (df[mfe_col] > df[mae_col]).astype(int)

    # ============================================================
    # 3. y_quality_score: Regression label (normalized MFE-MAE gap)
    # ============================================================
    # Compute MFE - MAE gap in bps
    gap_bps = df[mfe_col] - df[mae_col]

    # Normalize: divide by 100.0 to get reasonable scale, then clip to [-1, 1]
    # This means:
    #   - gap = +100 bps -> quality_score = +1.0
    #   - gap = -100 bps -> quality_score = -1.0
    #   - gap = 0 bps -> quality_score = 0.0
    df["y_quality_score"] = np.clip(gap_bps / 100.0, -1.0, 1.0).astype(np.float32)

    # Handle NaN values (set to 0.0)
    df["y_direction"] = df["y_direction"].fillna(0).astype(int)
    df["y_early_move"] = df["y_early_move"].fillna(0).astype(int)
    df["y_quality_score"] = df["y_quality_score"].fillna(0.0).astype(np.float32)

    return df


def compute_label_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute statistics for generated labels.

    Returns:
        Dict with label statistics (coverage, mean, std, etc.)
    """
    stats = {}

    if "y_direction" in df.columns:
        stats["direction_coverage"] = float(df["y_direction"].mean())
        stats["direction_positive"] = int(df["y_direction"].sum())
        stats["direction_total"] = len(df)

    if "y_early_move" in df.columns:
        stats["early_move_coverage"] = float(df["y_early_move"].mean())
        stats["early_move_positive"] = int(df["y_early_move"].sum())
        stats["early_move_total"] = len(df)

    if "y_quality_score" in df.columns:
        stats["quality_score_mean"] = float(df["y_quality_score"].mean())
        stats["quality_score_std"] = float(df["y_quality_score"].std())
        stats["quality_score_min"] = float(df["y_quality_score"].min())
        stats["quality_score_max"] = float(df["y_quality_score"].max())
        stats["quality_score_positive"] = int((df["y_quality_score"] > 0).sum())
        stats["quality_score_total"] = len(df)

    return stats

