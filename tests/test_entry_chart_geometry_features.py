import numpy as np

from gx1.features.entry_chart_geometry_v1 import (
    CHART_GEOMETRY_FEATURE_NAMES,
    CHART_GEOMETRY_SOURCE_FIELDS,
    build_entry_chart_geometry_layer,
    missing_chart_geometry_source_fields,
)
from gx1.features.entry_specialist_feature_groups_v1 import classify_entry_specialist_feature


def _matrix(names: list[str], n: int = 6) -> np.ndarray:
    x = np.zeros((n, len(names)), dtype=np.float32)
    idx = {name: i for i, name in enumerate(names)}

    def set_col(name: str, values) -> None:
        x[:, idx[name]] = np.asarray(values, dtype=np.float32)

    trend_down_up = [-2.0, 2.0, 2.5, 2.0, -1.5, -2.0]
    for name in (
        "snap._v1_ema_diff",
        "snap.ema20_slope",
        "snap.pos_vs_ema200",
        "ctx_cont._v1h1_ema_diff",
        "ctx_cont._v1h1_slope5",
        "ctx_cont._v1h4_ema_diff",
        "ctx_cont._v1h4_slope5",
        "ctx_cont.d1_ema_slope_20_canon_v2",
        "ctx_cont.m15_trend_sign_canon_v2",
    ):
        set_col(name, trend_down_up)
    set_col("ctx_cont.regime_stack_sum_v3", [-3, 3, 3, 2, -2, -3])
    set_col("ctx_cont.regime_tf_agreement_v3", [0.2, 1.0, 1.0, 0.9, 0.8, 0.8])
    set_col("ctx_cont.regime_divergence_flag_v3", [0, 0, 0, 0, 1, 1])

    set_col("ctx_cont.dist_last_swing_high_atr", [3, 1, 0.3, 2, 0.2, 0.1])
    set_col("ctx_cont.dist_last_swing_low_atr", [0.2, 0.1, 0.2, 0.1, 2, 3])
    set_col("ctx_cont.bars_since_swing_high", [20, 10, 3, 8, 1, 1])
    set_col("ctx_cont.bars_since_swing_low", [1, 1, 2, 1, 8, 10])
    set_col("ctx_cont.sr_nearest_pivot_abs_atr", [1, 0.5, 0.2, 0.1, 0.2, 0.1])
    set_col("ctx_cont.sr_support_proximity_exp", [0.8, 1.0, 0.9, 1.0, 0.1, 0.1])
    set_col("ctx_cont.sr_resistance_proximity_exp", [0.2, 0.2, 0.6, 0.3, 1.0, 1.0])
    set_col("ctx_cont.sr_support_minus_resistance_prox", [0.5, 0.8, 0.3, 0.7, -0.8, -0.9])
    for name in ("ctx_cont.dist_to_S1_atr", "ctx_cont.dist_to_S2_atr", "ctx_cont.dist_to_h1_lo_atr", "ctx_cont.dist_to_h4_lo_atr", "ctx_cont.dist_to_d1_lo_atr"):
        set_col(name, [0.2, 0.1, 0.2, 0.1, 3, 3])
    for name in ("ctx_cont.dist_to_R1_atr", "ctx_cont.dist_to_R2_atr", "ctx_cont.dist_to_h1_hi_atr", "ctx_cont.dist_to_h4_hi_atr", "ctx_cont.dist_to_d1_hi_atr"):
        set_col(name, [3, 3, 0.4, 3, 0.1, 0.1])

    set_col("snap.smc_premium_discount", [0.2, 0.4, 0.5, 0.618, 0.8, 0.8])
    set_col("ctx_cont.retracement_from_last_impulse", [0.2, 0.4, 0.5, 0.618, 0.8, 0.8])
    set_col("ctx_cont.d1_close_pct_in_20day_range_canon_v2", [0.2, 0.4, 0.5, 0.618, 0.8, 0.8])
    set_col("snap.smc_bos_up", [0, 1, 1, 1, 0, 0])
    set_col("snap.smc_bos_down", [0, 0, 0, 0, 1, 1])
    set_col("ctx_cont.smc_bos_pressure_last12", [-1, 1, 1, 1, -1, -1])
    set_col("ctx_cont.smc_bos_pressure_last48", [-1, 1, 1, 1, -1, -1])
    set_col("snap.smc_choch", [0, 0, 0, 0, 1, 1])
    set_col("ctx_cont.smc_choch_recent_tau12", [0, 0, 0, 0, 1, 1])
    set_col("ctx_cont.smc_choch_recent_tau24", [0, 0, 0, 0, 1, 1])
    set_col("snap.smc_sweep_up", [0, 0, 0, 0, 1, 1])
    set_col("snap.smc_sweep_down", [0, 1, 0, 1, 0, 0])
    set_col("ctx_cont.smc_sweep_bull_pressure_last12", [0, 1, 0, 1, -1, -1])
    set_col("ctx_cont.smc_sweep_bull_pressure_last48", [0, 1, 0, 1, -1, -1])
    set_col("snap.smc_sweep_size_atr", [0, 1, 0, 1, 1, 1])
    set_col("ctx_cont.smc_sweep_size_recent_tau12", [0, 1, 0, 1, 1, 1])
    set_col("ctx_cont.smc_sweep_recency_tau24", [0, 1, 0.5, 1, 1, 1])
    set_col("ctx_cont.wick_ratio", [0.1, 1, 0.2, 0.8, 1, 1])
    set_col("snap.wick_asym", [0, -0.5, 0, -0.2, 0.5, 0.5])
    set_col("ctx_cont.H1_range_compression_ratio", [0.2, 0.8, 0.9, 0.9, 0.8, 0.8])
    set_col("ctx_cont.M15_range_compression_ratio", [0.2, 0.8, 0.9, 0.9, 0.8, 0.8])
    set_col("snap._v1_bb_squeeze_20_2", [0.1, 0.8, 0.9, 0.9, 0.8, 0.8])
    set_col("snap.atr_z", [0.1, 0.5, 1, 2, 1, 1])
    set_col("snap.rvol_20", [0.1, 0.5, 1, 2, 1, 1])
    set_col("snap.vol_ratio_5_20", [0.1, 0.5, 1, 2, 1, 1])
    set_col("ctx_cont.h1_trend_age_bars_norm_v2", [0.1, 0.2, 0.3, 0.4, 0.9, 1.0])
    set_col("ctx_cont.h4_trend_age_bars_norm_v2", [0.1, 0.2, 0.3, 0.4, 0.9, 1.0])
    set_col("ctx_cont.D1_atr_percentile_252", [0.2, 0.3, 0.4, 0.5, 0.9, 1.0])
    return x


def test_chart_geometry_layer_builds_manual_trader_proxies() -> None:
    names = list(CHART_GEOMETRY_SOURCE_FIELDS)
    x = _matrix(names)
    out, out_names = build_entry_chart_geometry_layer(x, names)
    idx = {name: i for i, name in enumerate(out_names)}

    assert out.shape == (6, len(CHART_GEOMETRY_FEATURE_NAMES))
    assert tuple(out_names) == CHART_GEOMETRY_FEATURE_NAMES
    assert np.isfinite(out).all()
    assert out[1, idx["chart.geometry_ema_cross_up_pressure"]] > 0.0
    assert out[3, idx["chart.geometry_fib_retracement_618_proximity"]] > 0.99
    assert out[3, idx["chart.geometry_fib_pullback_long_pressure"]] > out[3, idx["chart.geometry_fib_pullback_short_pressure"]]
    assert out[3, idx["chart.geometry_trendline_break_up_pressure"]] > 0.0
    assert out[5, idx["chart.geometry_failed_breakout_high_reversal_pressure"]] > 0.0


def test_chart_geometry_source_contract_and_specialist_routing() -> None:
    assert missing_chart_geometry_source_fields(CHART_GEOMETRY_SOURCE_FIELDS) == []
    missing = missing_chart_geometry_source_fields(
        name for name in CHART_GEOMETRY_SOURCE_FIELDS if name != "ctx_cont.dist_to_R1_atr"
    )
    assert missing == ["ctx_cont.dist_to_R1_atr"]
    assert classify_entry_specialist_feature("chart.geometry_fib_golden_zone_proximity") == "chart_geometry_encoder"
    assert classify_entry_specialist_feature("chart.geometry_ascending_triangle_pressure") == "chart_geometry_encoder"
