"""
Canonical EXIT_IO_V1_CTX35 feature list (ordered, length = 35).
This list is the single source of truth for runtime and artifact packaging.
"""

EXIT_IO_V1_CTX35_FEATURES = [
    # --- existing 19 ---
    "p_long",
    "p_short",
    "p_flat",
    "p_hat",
    "uncertainty_score",
    "margin_top1_top2",
    "entropy",
    "p_long_entry",
    "p_hat_entry",
    "uncertainty_entry",
    "entropy_entry",
    "margin_entry",
    "pnl_bps_now",
    "mfe_bps",
    "mae_bps",
    "dd_from_mfe_bps",
    "bars_held",
    "time_since_mfe_bars",
    "atr_bps_now",
    # --- ctx_cont (11 base+micro) ---
    "atr_bps",
    "spread_bps",
    "D1_dist_from_ema200_atr",
    "H1_range_compression_ratio",
    "D1_atr_percentile_252",
    "M15_range_compression_ratio",
    "micro_momentum_3",
    "micro_momentum_5",
    "micro_acceleration",
    "wick_ratio",
    "distance_ema_fast",
    # --- swing (5) ---
    "dist_last_swing_high_atr",
    "dist_last_swing_low_atr",
    "bars_since_swing_high",
    "bars_since_swing_low",
    "retracement_from_last_impulse",
]

EXIT_IO_V1_CTX35_FEATURE_COUNT = 35
