import numpy as np

from gx1.features.entry_specialist_feature_groups_v1 import classify_entry_specialist_feature
from gx1.features.entry_support_resistance_memory_v1 import (
    SUPPORT_RESISTANCE_MEMORY_FEATURE_NAMES,
    SUPPORT_RESISTANCE_MEMORY_OPTIONAL_FIELDS,
    SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS,
    build_entry_support_resistance_memory_layer,
    missing_support_resistance_memory_source_fields,
)


EXPECTED_SUPPORT_RESISTANCE_MEMORY_FEATURE_COUNT = 34


def _matrix(names: list[str], n: int = 8) -> np.ndarray:
    x = np.zeros((n, len(names)), dtype=np.float32)
    idx = {name: i for i, name in enumerate(names)}

    def set_col(name: str, values) -> None:
        if name in idx:
            x[:, idx[name]] = np.asarray(values, dtype=np.float32)

    support = [8.0, 0.12, 0.18, 0.14, 4.0, 4.0, 4.0, 8.0]
    resistance = [8.0, 4.0, 4.0, 4.0, 0.20, 0.10, 0.12, 8.0]

    set_col("ctx_cont.sr_nearest_pivot_abs_atr", [8.0, 0.10, 0.12, 0.15, 0.20, 0.10, 0.12, 8.0])
    set_col("ctx_cont.sr_support_proximity_exp", [0.05, 1.0, 0.95, 0.95, 0.20, 0.10, 0.12, 0.05])
    set_col("ctx_cont.sr_resistance_proximity_exp", [0.05, 0.12, 0.15, 0.10, 0.90, 1.0, 0.95, 0.05])
    set_col("ctx_cont.sr_support_minus_resistance_prox", [0.0, 0.85, 0.70, 0.60, -0.70, -0.85, -0.65, 0.0])
    set_col("ctx_cont.liquidity_lo_nearest_abs_atr", [8.0, 0.08, 0.10, 0.12, 5.0, 4.0, 4.0, 8.0])
    set_col("ctx_cont.liquidity_hi_nearest_abs_atr", [8.0, 4.0, 4.0, 4.0, 0.12, 0.08, 0.10, 8.0])
    set_col("ctx_cont.liquidity_lo_minus_hi_prox", [0.0, 0.80, 0.75, 0.65, -0.70, -0.80, -0.75, 0.0])
    for name in ("ctx_cont.dist_to_S1_atr", "ctx_cont.dist_to_S2_atr", "ctx_cont.dist_to_m5_lo_atr", "ctx_cont.dist_to_m15_lo_atr", "ctx_cont.dist_to_h1_lo_atr", "ctx_cont.dist_to_h4_lo_atr", "ctx_cont.dist_to_d1_lo_atr"):
        set_col(name, support)
    for name in ("ctx_cont.dist_to_R1_atr", "ctx_cont.dist_to_R2_atr", "ctx_cont.dist_to_m5_hi_atr", "ctx_cont.dist_to_m15_hi_atr", "ctx_cont.dist_to_h1_hi_atr", "ctx_cont.dist_to_h4_hi_atr", "ctx_cont.dist_to_d1_hi_atr"):
        set_col(name, resistance)
    set_col("ctx_cont.dist_last_swing_low_atr", support)
    set_col("ctx_cont.dist_last_swing_high_atr", resistance)
    set_col("ctx_cont.bars_since_swing_low", [96, 1, 2, 3, 20, 21, 22, 96])
    set_col("ctx_cont.bars_since_swing_high", [96, 20, 21, 22, 2, 1, 2, 96])

    set_col("snap.smc_sweep_down", [0, 1, 0, 0, 0, 0, 0, 0])
    set_col("snap.smc_sweep_up", [0, 0, 0, 0, 1, 1, 0, 0])
    set_col("snap.smc_sweep_size_atr", [0, 1.4, 0.8, 0.2, 1.2, 1.5, 0.6, 0])
    set_col("snap.smc_bars_since_sweep", [96, 0, 1, 5, 0, 0, 2, 96])
    set_col("ctx_cont.smc_sweep_bull_pressure_last12", [0, 1.0, 0.5, 0.0, -0.8, -1.0, -0.4, 0])
    set_col("ctx_cont.smc_sweep_bull_pressure_last48", [0, 0.8, 0.4, 0.0, -0.6, -0.8, -0.4, 0])
    set_col("ctx_cont.smc_sweep_size_recent_tau12", [0, 0.9, 0.5, 0.1, 0.7, 0.9, 0.4, 0])
    set_col("ctx_cont.smc_sweep_recency_tau24", [0, 1.0, 0.8, 0.3, 1.0, 1.0, 0.5, 0])
    set_col("snap.smc_bos_down", [0, 0, 0, 1, 0, 0, 0, 0])
    set_col("snap.smc_bos_up", [0, 0, 0, 0, 0, 0, 1, 0])
    set_col("ctx_cont.smc_bos_pressure_last12", [0, 0.2, 0.2, -1.0, 0.2, 0.2, 1.0, 0])
    set_col("ctx_cont.smc_bos_pressure_last48", [0, 0.1, 0.1, -0.8, 0.1, 0.1, 0.8, 0])
    set_col("snap.smc_choch", [0, 0, 0, 1, 0, 1, 0, 0])
    set_col("ctx_cont.smc_choch_recent_tau12", [0, 0.1, 0.1, 0.8, 0.1, 0.8, 0.2, 0])
    set_col("ctx_cont.smc_choch_recent_tau24", [0, 0.1, 0.1, 0.7, 0.1, 0.7, 0.2, 0])
    set_col("snap.smc_premium_discount", [0.5, 0.18, 0.22, 0.25, 0.82, 0.86, 0.72, 0.5])

    set_col("snap.body_pct", [0.0, 0.55, 0.35, -0.75, -0.40, -0.55, 0.80, 0.0])
    set_col("snap._v1_body_share_1", [0.0, 0.60, 0.40, 0.70, 0.45, 0.55, 0.75, 0.0])
    set_col("snap._v1_clv", [0.0, 0.90, 0.70, -0.85, -0.65, -0.90, 0.90, 0.0])
    set_col("snap.wick_asym", [0.0, -0.85, -0.45, 0.10, 0.60, 0.85, -0.10, 0.0])
    set_col("ctx_cont.wick_ratio", [0.5, 0.08, 0.20, 0.88, 0.80, 0.92, 0.10, 0.5])

    for name in ("ctx_cont._v1h1_ema_diff", "ctx_cont._v1h4_ema_diff", "ctx_cont.d1_ema_slope_20_canon_v2", "ctx_cont.m15_trend_sign_canon_v2"):
        set_col(name, [0.0, 1.5, 1.2, -1.0, -1.0, -1.5, 1.5, 0.0])
    set_col("ctx_cont.regime_stack_sum_v3", [0.0, 2.0, 1.5, -2.0, -1.0, -2.0, 2.0, 0.0])

    set_col("chart.smc_liquidity_reclaim_confirmation_long", [0, 0.8, 0.4, 0, 0, 0, 0, 0])
    set_col("chart.smc_liquidity_reclaim_confirmation_short", [0, 0, 0, 0, 0.5, 0.8, 0, 0])
    set_col("chart.smc_liquidity_false_breakout_quality_long", [0, 0.8, 0.4, 0, 0, 0, 0, 0])
    set_col("chart.smc_liquidity_false_breakout_quality_short", [0, 0, 0, 0, 0.5, 0.8, 0, 0])

    return x


def test_support_resistance_memory_layer_builds_causal_level_features() -> None:
    names = list(SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS) + list(SUPPORT_RESISTANCE_MEMORY_OPTIONAL_FIELDS)
    x = _matrix(names)
    out, out_names = build_entry_support_resistance_memory_layer(x, names)
    idx = {name: i for i, name in enumerate(out_names)}

    assert tuple(out_names) == SUPPORT_RESISTANCE_MEMORY_FEATURE_NAMES
    assert len(out_names) == EXPECTED_SUPPORT_RESISTANCE_MEMORY_FEATURE_COUNT
    assert len(set(out_names)) == EXPECTED_SUPPORT_RESISTANCE_MEMORY_FEATURE_COUNT
    assert out.shape == (8, len(out_names))
    assert np.isfinite(out).all()

    assert out[1, idx["chart.sr_memory_nearest_pivot_proximity"]] > out[0, idx["chart.sr_memory_nearest_pivot_proximity"]]
    assert out[1, idx["chart.sr_memory_support_level_proximity_stack"]] > out[0, idx["chart.sr_memory_support_level_proximity_stack"]]
    assert out[5, idx["chart.sr_memory_resistance_level_proximity_stack"]] > out[0, idx["chart.sr_memory_resistance_level_proximity_stack"]]
    assert out[3, idx["chart.sr_memory_support_repeated_test_slow"]] > out[1, idx["chart.sr_memory_support_repeated_test_slow"]]
    assert out[6, idx["chart.sr_memory_resistance_repeated_test_slow"]] > out[4, idx["chart.sr_memory_resistance_repeated_test_slow"]]
    assert out[1, idx["chart.sr_memory_support_reclaim_pressure_long"]] > out[0, idx["chart.sr_memory_support_reclaim_pressure_long"]]
    assert out[5, idx["chart.sr_memory_resistance_reclaim_pressure_short"]] > out[0, idx["chart.sr_memory_resistance_reclaim_pressure_short"]]
    assert out[3, idx["chart.sr_memory_support_break_pressure_down"]] > out[0, idx["chart.sr_memory_support_break_pressure_down"]]
    assert out[6, idx["chart.sr_memory_resistance_break_pressure_up"]] > out[0, idx["chart.sr_memory_resistance_break_pressure_up"]]
    assert out[1, idx["chart.sr_memory_mtf_h1_h4_d1_support_confluence"]] > out[0, idx["chart.sr_memory_mtf_h1_h4_d1_support_confluence"]]
    assert out[5, idx["chart.sr_memory_mtf_h1_h4_d1_resistance_confluence"]] > out[0, idx["chart.sr_memory_mtf_h1_h4_d1_resistance_confluence"]]
    assert out[1, idx["chart.sr_memory_liquidity_low_level_rejection_long"]] > out[0, idx["chart.sr_memory_liquidity_low_level_rejection_long"]]
    assert out[5, idx["chart.sr_memory_liquidity_high_level_rejection_short"]] > out[0, idx["chart.sr_memory_liquidity_high_level_rejection_short"]]
    assert out[6, idx["chart.sr_memory_liquidity_resistance_break_continuation_long"]] > out[0, idx["chart.sr_memory_liquidity_resistance_break_continuation_long"]]


def test_support_resistance_memory_layer_sanitizes_nonfinite_inputs() -> None:
    names = list(SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS) + list(SUPPORT_RESISTANCE_MEMORY_OPTIONAL_FIELDS)
    x = _matrix(names)
    x[1, names.index("ctx_cont.sr_support_proximity_exp")] = np.nan
    x[2, names.index("ctx_cont.dist_to_h1_lo_atr")] = np.inf
    x[5, names.index("ctx_cont.dist_to_h1_hi_atr")] = -np.inf

    out, out_names = build_entry_support_resistance_memory_layer(x, names)

    assert len(out_names) == EXPECTED_SUPPORT_RESISTANCE_MEMORY_FEATURE_COUNT
    assert out.shape == (8, EXPECTED_SUPPORT_RESISTANCE_MEMORY_FEATURE_COUNT)
    assert np.isfinite(out).all()


def test_support_resistance_memory_layer_is_future_row_invariant() -> None:
    names = list(SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS)
    x = _matrix(names)
    changed_future = x.copy()
    changed_future[5:, :] = changed_future[5:, :] * -9.0 + 4.0

    out, names_a = build_entry_support_resistance_memory_layer(x, names)
    changed, names_b = build_entry_support_resistance_memory_layer(changed_future, names)

    assert names_a == names_b
    np.testing.assert_allclose(out[:5], changed[:5], rtol=0.0, atol=1e-7)


def test_support_resistance_memory_source_contract_and_routing() -> None:
    assert len(SUPPORT_RESISTANCE_MEMORY_FEATURE_NAMES) == EXPECTED_SUPPORT_RESISTANCE_MEMORY_FEATURE_COUNT
    assert len(SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS) == len(set(SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS))
    assert missing_support_resistance_memory_source_fields(SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS) == []
    assert missing_support_resistance_memory_source_fields(["ctx_cont.sr_nearest_pivot_abs_atr"])[:1] == [
        "ctx_cont.sr_support_proximity_exp"
    ]

    routing = {name: classify_entry_specialist_feature(name) for name in SUPPORT_RESISTANCE_MEMORY_FEATURE_NAMES}
    routing_gaps = {
        name: specialist
        for name, specialist in routing.items()
        if specialist not in {"smc_liquidity_encoder", "chart_geometry_encoder"}
    }

    assert routing_gaps == {}
    assert set(routing.values()) == {"smc_liquidity_encoder"}
