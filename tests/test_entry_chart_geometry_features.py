import numpy as np
import pytest

from gx1.contracts.entry_pretrain_polarity_signal_v1 import (
    PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS,
)
from gx1.features.entry_chart_geometry_v1 import (
    CHART_GEOMETRY_FEATURE_NAMES,
    CHART_GEOMETRY_SMART2_FEATURE_NAMES,
    CHART_GEOMETRY_MODEL_NATIVE_FEATURE_NAMES,
    CHART_GEOMETRY_SOURCE_FIELDS,
    build_entry_chart_geometry_layer,
    missing_chart_geometry_source_fields,
)
from gx1.features.entry_specialist_feature_groups_v1 import classify_entry_specialist_feature
from gx1.features.entry_model_native_feature_layers_v1 import add_chart_feature


EXPECTED_CHART_GEOMETRY_FEATURE_COUNT = 58
SMART2_CHART_GEOMETRY_FEATURE_NAMES = (
    "chart.geometry_trendline_channel_confluence_pressure",
    "chart.geometry_channel_edge_rejection_pressure",
    "chart.geometry_rising_support_rail_long_pressure",
    "chart.geometry_rising_support_rail_short_trap_pressure",
    "chart.geometry_falling_resistance_rail_short_pressure",
    "chart.geometry_falling_resistance_rail_long_trap_pressure",
    "chart.geometry_fib_support_confluence_long_pressure",
    "chart.geometry_fib_resistance_confluence_short_pressure",
    "chart.geometry_fib_extension_exhaustion_risk",
    "chart.geometry_ema_cross_mtf_bull_confirmation",
    "chart.geometry_ema_cross_mtf_bear_confirmation",
    "chart.geometry_triangle_apex_compression_pressure",
    "chart.geometry_flag_breakout_readiness_pressure",
    "chart.geometry_mtf_channel_breakout_up_quality",
    "chart.geometry_mtf_channel_breakout_down_quality",
    "chart.geometry_mtf_channel_retest_long_quality",
    "chart.geometry_mtf_channel_retest_short_quality",
)

STRUCTURAL_AUX_GEOMETRY_FEATURE_NAMES = (
    "chart.geometry_support_line_proximity_stack",
    "chart.geometry_resistance_line_proximity_stack",
    "chart.geometry_channel_position_low_to_high",
    "chart.geometry_channel_edge_pressure",
)
MANDATORY_RAW_GEOMETRY_FEATURE_NAMES = (
    *PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS,
    "chart.geometry_channel_edge_pressure",
)


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
    set_col("snap.body_pct", [0.8, 0.1, 0.8, 0.2, 0.1, 0.1])
    set_col("ctx_cont.H1_range_compression_ratio", [1.0, 0.8, 0.5, 0.5, 0.8, 0.8])
    set_col("ctx_cont.M15_range_compression_ratio", [1.0, 0.8, 0.5, 0.5, 0.8, 0.8])
    set_col("snap._v1_bb_squeeze_20_2", [0.0, -0.2, -0.9, -0.9, -0.2, -0.2])
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

    assert len(CHART_GEOMETRY_FEATURE_NAMES) == EXPECTED_CHART_GEOMETRY_FEATURE_COUNT
    assert out.shape == (6, EXPECTED_CHART_GEOMETRY_FEATURE_COUNT)
    assert tuple(out_names) == CHART_GEOMETRY_FEATURE_NAMES
    assert CHART_GEOMETRY_SMART2_FEATURE_NAMES == SMART2_CHART_GEOMETRY_FEATURE_NAMES
    assert tuple(out_names[-len(SMART2_CHART_GEOMETRY_FEATURE_NAMES) :]) == CHART_GEOMETRY_SMART2_FEATURE_NAMES
    assert len(CHART_GEOMETRY_MODEL_NATIVE_FEATURE_NAMES) == 18
    assert CHART_GEOMETRY_MODEL_NATIVE_FEATURE_NAMES[:5] == (
        MANDATORY_RAW_GEOMETRY_FEATURE_NAMES
    )
    assert set(STRUCTURAL_AUX_GEOMETRY_FEATURE_NAMES).issubset(
        set(MANDATORY_RAW_GEOMETRY_FEATURE_NAMES)
    )
    assert set(CHART_GEOMETRY_MODEL_NATIVE_FEATURE_NAMES[5:]).issubset(
        set(CHART_GEOMETRY_SMART2_FEATURE_NAMES)
    )
    assert (
        "chart.geometry_rising_support_rail_short_trap_pressure"
        in CHART_GEOMETRY_MODEL_NATIVE_FEATURE_NAMES
    )
    assert (
        "chart.geometry_falling_resistance_rail_long_trap_pressure"
        in CHART_GEOMETRY_MODEL_NATIVE_FEATURE_NAMES
    )
    assert np.isfinite(out).all()
    assert out[1, idx["chart.geometry_ema_cross_up_pressure"]] > 0.0
    assert out[3, idx["chart.geometry_fib_retracement_618_proximity"]] > 0.99
    assert out[3, idx["chart.geometry_fib_pullback_long_pressure"]] > out[3, idx["chart.geometry_fib_pullback_short_pressure"]]
    assert out[3, idx["chart.geometry_trendline_break_up_pressure"]] > 0.0
    assert out[5, idx["chart.geometry_failed_breakout_high_reversal_pressure"]] > 0.0
    assert out[1, idx["chart.geometry_ema_cross_mtf_bull_confirmation"]] > 0.0
    assert out[4, idx["chart.geometry_ema_cross_mtf_bear_confirmation"]] > 0.0
    assert out[3, idx["chart.geometry_mtf_channel_breakout_up_quality"]] > 0.0
    assert out[5, idx["chart.geometry_mtf_channel_retest_short_quality"]] > 0.0
    assert out[3, idx["chart.geometry_rising_support_rail_long_pressure"]] > 0.0
    assert out[3, idx["chart.geometry_rising_support_rail_short_trap_pressure"]] > 0.0
    assert out[5, idx["chart.geometry_falling_resistance_rail_short_pressure"]] > 0.0
    assert out[5, idx["chart.geometry_falling_resistance_rail_long_trap_pressure"]] > 0.0
    assert out[3, idx["chart.geometry_fib_support_confluence_long_pressure"]] > out[3, idx["chart.geometry_fib_resistance_confluence_short_pressure"]]
    assert out[5, idx["chart.geometry_fib_resistance_confluence_short_pressure"]] > out[5, idx["chart.geometry_fib_support_confluence_long_pressure"]]
    assert not np.array_equal(
        out[:, idx["chart.geometry_fib_pullback_long_pressure"]],
        out[:, idx["chart.geometry_fib_support_confluence_long_pressure"]],
    )
    assert not np.array_equal(
        out[:, idx["chart.geometry_fib_pullback_short_pressure"]],
        out[:, idx["chart.geometry_fib_resistance_confluence_short_pressure"]],
    )
    assert out[2, idx["chart.geometry_triangle_apex_compression_pressure"]] > out[0, idx["chart.geometry_triangle_apex_compression_pressure"]]
    assert out[3, idx["chart.geometry_flag_breakout_readiness_pressure"]] > 0.0
    assert out[1, idx["chart.geometry_channel_position_low_to_high"]] < 0.42
    assert out[5, idx["chart.geometry_channel_position_low_to_high"]] > 0.58


def test_chart_geometry_source_contract_and_specialist_routing() -> None:
    assert missing_chart_geometry_source_fields(CHART_GEOMETRY_SOURCE_FIELDS) == []
    missing = missing_chart_geometry_source_fields(
        name for name in CHART_GEOMETRY_SOURCE_FIELDS if name != "ctx_cont.dist_to_R1_atr"
    )
    assert missing == ["ctx_cont.dist_to_R1_atr"]
    assert classify_entry_specialist_feature("chart.geometry_fib_golden_zone_proximity") == "chart_geometry_encoder"
    assert classify_entry_specialist_feature("chart.geometry_ascending_triangle_pressure") == "chart_geometry_encoder"
    assert classify_entry_specialist_feature("chart.geometry_mtf_channel_retest_long_quality") == "chart_geometry_encoder"
    assert classify_entry_specialist_feature("chart.geometry_rising_support_rail_short_trap_pressure") == "chart_geometry_encoder"


def test_chart_geometry_always_rejects_missing_source_fields() -> None:
    names = [name for name in CHART_GEOMETRY_SOURCE_FIELDS if name != "ctx_cont.dist_to_R1_atr"]
    x = np.zeros((3, len(names)), dtype=np.float32)

    with pytest.raises(RuntimeError, match="CHART_GEOMETRY_SOURCE_FIELDS_MISSING"):
        build_entry_chart_geometry_layer(x, names)


def test_chart_geometry_rejects_nonfinite_sources_and_duplicate_names() -> None:
    names = list(CHART_GEOMETRY_SOURCE_FIELDS)
    for value in (np.nan, np.inf, -np.inf):
        x = _matrix(names)
        x[1, names.index("ctx_cont.dist_to_R1_atr")] = value
        with pytest.raises(RuntimeError, match="CHART_GEOMETRY_SOURCE_NONFINITE"):
            build_entry_chart_geometry_layer(x, names)

    duplicate_names = [*names, names[0]]
    with pytest.raises(RuntimeError, match="CHART_GEOMETRY_FEATURE_NAMES_DUPLICATE"):
        build_entry_chart_geometry_layer(
            np.zeros((2, len(duplicate_names)), dtype=np.float32),
            duplicate_names,
        )


def test_generated_chart_features_keep_constant_columns_for_manifest_contract() -> None:
    arrays: list[np.ndarray] = []
    names: list[str] = []

    add_chart_feature(arrays, names, "lh_x_ema50_200", np.zeros(6, dtype=np.float32))

    assert names == ["chart.lh_x_ema50_200"]
    assert len(arrays) == 1
    assert arrays[0].shape == (6,)
    assert np.all(arrays[0] == 0.0)
