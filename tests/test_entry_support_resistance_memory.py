import numpy as np
import pytest

from gx1.features.entry_specialist_feature_groups_v1 import classify_entry_specialist_feature
from gx1.features.entry_support_resistance_memory_v1 import (
    SR_MEMORY_WARMUP_ROWS,
    SUPPORT_RESISTANCE_MEMORY_FEATURE_NAMES,
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

    set_col("candle.pattern_body_direction", [0, 1, 1, -1, -1, -1, 1, 0])
    set_col("candle.pattern_body_share", [0.0, 0.60, 0.40, 0.70, 0.45, 0.55, 0.75, 0.0])
    set_col("candle.pattern_upper_wick_share", [0.5, 0.05, 0.15, 0.20, 0.45, 0.40, 0.10, 0.5])
    set_col("candle.pattern_lower_wick_share", [0.5, 0.35, 0.45, 0.10, 0.10, 0.05, 0.15, 0.5])
    set_col("candle.pattern_close_location", [0.5, 0.95, 0.80, 0.10, 0.20, 0.08, 0.90, 0.5])

    for name in ("ctx_cont._v1h1_ema_diff", "ctx_cont._v1h4_ema_diff", "ctx_cont.d1_ema_slope_20_canon_v2", "ctx_cont.m15_trend_sign_canon_v2"):
        set_col(name, [0.0, 1.5, 1.2, -1.0, -1.0, -1.5, 1.5, 0.0])
    set_col("ctx_cont.regime_stack_sum_v3", [0.0, 2.0, 1.5, -2.0, -1.0, -2.0, 2.0, 0.0])

    set_col("chart.smc_liquidity_reclaim_confirmation_long", [0, 0.8, 0.4, 0, 0, 0, 0, 0])
    set_col("chart.smc_liquidity_reclaim_confirmation_short", [0, 0, 0, 0, 0.5, 0.8, 0, 0])
    set_col("chart.smc_liquidity_false_breakout_quality_long", [0, 0.8, 0.4, 0, 0, 0, 0, 0])
    set_col("chart.smc_liquidity_false_breakout_quality_short", [0, 0, 0, 0, 0.5, 0.8, 0, 0])

    return x


def test_support_resistance_memory_layer_builds_causal_level_features() -> None:
    names = list(SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS)
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
    # Side distinction: with the shared min(r_abs, s_abs) pivot row removed
    # from both maxes, the stacks must decorrelate on this mixed scenario
    # (support-near rows 1-3, resistance-near rows 4-6), not merely differ.
    support_stack = out[:, idx["chart.sr_memory_support_level_proximity_stack"]]
    resistance_stack = out[:, idx["chart.sr_memory_resistance_level_proximity_stack"]]
    assert float(np.corrcoef(support_stack, resistance_stack)[0, 1]) < 0.99


def test_support_resistance_memory_layer_rejects_nonfinite_inputs() -> None:
    names = list(SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS)
    for field, value in (
        ("ctx_cont.sr_support_proximity_exp", np.nan),
        ("ctx_cont.dist_to_h1_lo_atr", np.inf),
        ("ctx_cont.dist_to_h1_hi_atr", -np.inf),
    ):
        x = _matrix(names)
        x[2, names.index(field)] = value
        with pytest.raises(RuntimeError, match="SUPPORT_RESISTANCE_MEMORY_SOURCE_NONFINITE"):
            build_entry_support_resistance_memory_layer(x, names)


def test_support_resistance_memory_layer_rejects_missing_generated_evidence() -> None:
    missing_field = "chart.geometry_support_line_proximity_stack"
    names = [name for name in SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS if name != missing_field]
    with pytest.raises(RuntimeError, match="SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS_MISSING"):
        build_entry_support_resistance_memory_layer(_matrix(names), names)


def test_support_resistance_memory_layer_is_future_row_invariant() -> None:
    names = list(SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS)
    x = _matrix(names)
    changed_future = x.copy()
    changed_future[5:, :] = changed_future[5:, :] * -9.0 + 4.0

    out, names_a = build_entry_support_resistance_memory_layer(x, names)
    changed, names_b = build_entry_support_resistance_memory_layer(changed_future, names)

    assert names_a == names_b
    np.testing.assert_allclose(out[:5], changed[:5], rtol=0.0, atol=1e-7)


def test_support_resistance_dominated_max_rows_removal_is_bit_identical_on_producer_consistent_inputs() -> None:
    """The 2026-08-09 stack repair removed max() rows dominated under the
    producer contract (entry_smart_context.py:140-184).  On inputs that satisfy
    that contract the reduced max must equal, bit for bit, the previous max
    over the full row set minus only the shared pivot row (whose removal is the
    deliberate value change)."""
    names = list(SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS)
    x = _matrix(names)
    idx = {name: i for i, name in enumerate(names)}

    low_fields = (
        "ctx_cont.dist_to_m5_lo_atr",
        "ctx_cont.dist_to_m15_lo_atr",
        "ctx_cont.dist_to_h1_lo_atr",
        "ctx_cont.dist_to_h4_lo_atr",
        "ctx_cont.dist_to_d1_lo_atr",
    )
    high_fields = (
        "ctx_cont.dist_to_m5_hi_atr",
        "ctx_cont.dist_to_m15_hi_atr",
        "ctx_cont.dist_to_h1_hi_atr",
        "ctx_cont.dist_to_h4_hi_atr",
        "ctx_cont.dist_to_d1_hi_atr",
    )
    x[:, idx["ctx_cont.dist_to_S1_atr"]] = [0.10, 0.60, 2.0, 4.0, 0.30, 1.2, 5.0, 8.0]
    x[:, idx["ctx_cont.dist_to_S2_atr"]] = [0.40, 0.20, 3.0, 5.0, 0.90, 0.7, 6.0, 9.0]
    x[:, idx["ctx_cont.dist_to_R1_atr"]] = [0.50, 3.0, 0.25, 1.0, 4.0, 0.15, 2.0, 7.0]
    x[:, idx["ctx_cont.dist_to_R2_atr"]] = [0.80, 4.0, 0.45, 2.0, 5.0, 0.35, 3.0, 8.0]
    for offset, field in enumerate(low_fields):
        x[:, idx[field]] = np.asarray(
            [0.2, 1.5, 0.4, 6.0, 2.5, 0.9, 3.5, 7.0], dtype=np.float32
        ) + 0.1 * offset
    for offset, field in enumerate(high_fields):
        x[:, idx[field]] = np.asarray(
            [0.6, 2.5, 0.3, 1.4, 4.5, 0.2, 2.4, 6.0], dtype=np.float32
        ) + 0.1 * offset

    # Derive the aggregates exactly as the producer does
    # (gx1/features/entry_smart_context.py:140-184).
    s_abs = np.minimum(
        np.abs(x[:, idx["ctx_cont.dist_to_S1_atr"]]),
        np.abs(x[:, idx["ctx_cont.dist_to_S2_atr"]]),
    )
    r_abs = np.minimum(
        np.abs(x[:, idx["ctx_cont.dist_to_R1_atr"]]),
        np.abs(x[:, idx["ctx_cont.dist_to_R2_atr"]]),
    )
    lo_abs = np.min(np.abs(np.stack([x[:, idx[f]] for f in low_fields])), axis=0)
    hi_abs = np.min(np.abs(np.stack([x[:, idx[f]] for f in high_fields])), axis=0)
    x[:, idx["ctx_cont.sr_support_proximity_exp"]] = np.exp(-np.minimum(s_abs, 20.0))
    x[:, idx["ctx_cont.sr_resistance_proximity_exp"]] = np.exp(-np.minimum(r_abs, 20.0))
    x[:, idx["ctx_cont.sr_nearest_pivot_abs_atr"]] = np.minimum(r_abs, s_abs)
    x[:, idx["ctx_cont.liquidity_lo_nearest_abs_atr"]] = lo_abs
    x[:, idx["ctx_cont.liquidity_hi_nearest_abs_atr"]] = hi_abs

    def prox(values: np.ndarray) -> np.ndarray:
        return (1.0 / (1.0 + np.abs(values))).astype(np.float32)

    def clip01(values: np.ndarray) -> np.ndarray:
        return np.clip(values, 0.0, 1.0).astype(np.float32)

    def recency(values: np.ndarray) -> np.ndarray:
        return (1.0 / (1.0 + np.maximum(values, 0.0))).astype(np.float32)

    def col(name: str) -> np.ndarray:
        return x[:, idx[name]]

    swing_low_row = clip01(
        prox(col("ctx_cont.dist_last_swing_low_atr"))
        * (0.50 + recency(col("ctx_cont.bars_since_swing_low")))
    )
    swing_high_row = clip01(
        prox(col("ctx_cont.dist_last_swing_high_atr"))
        * (0.50 + recency(col("ctx_cont.bars_since_swing_high")))
    )
    kept_support = [
        prox(col("ctx_cont.dist_to_S1_atr")),
        prox(col("ctx_cont.dist_to_S2_atr")),
        prox(col("ctx_cont.liquidity_lo_nearest_abs_atr")),
        swing_low_row,
        clip01(col("chart.geometry_support_line_proximity_stack")),
    ]
    dominated_support = [clip01(col("ctx_cont.sr_support_proximity_exp"))] + [
        prox(col(field)) for field in low_fields
    ]
    kept_resistance = [
        prox(col("ctx_cont.dist_to_R1_atr")),
        prox(col("ctx_cont.dist_to_R2_atr")),
        prox(col("ctx_cont.liquidity_hi_nearest_abs_atr")),
        swing_high_row,
        clip01(col("chart.geometry_resistance_line_proximity_stack")),
    ]
    dominated_resistance = [clip01(col("ctx_cont.sr_resistance_proximity_exp"))] + [
        prox(col(field)) for field in high_fields
    ]

    reduced_support = np.maximum.reduce(kept_support)
    reduced_resistance = np.maximum.reduce(kept_resistance)
    # Bit-identity of the dominated-row removal on producer-consistent inputs.
    np.testing.assert_array_equal(
        reduced_support, np.maximum.reduce(kept_support + dominated_support)
    )
    np.testing.assert_array_equal(
        reduced_resistance, np.maximum.reduce(kept_resistance + dominated_resistance)
    )

    out, out_names = build_entry_support_resistance_memory_layer(x, names)
    out_idx = {name: i for i, name in enumerate(out_names)}
    np.testing.assert_array_equal(
        out[:, out_idx["chart.sr_memory_support_level_proximity_stack"]], reduced_support
    )
    np.testing.assert_array_equal(
        out[:, out_idx["chart.sr_memory_resistance_level_proximity_stack"]], reduced_resistance
    )


def test_support_resistance_memory_source_contract_and_routing() -> None:
    assert len(SUPPORT_RESISTANCE_MEMORY_FEATURE_NAMES) == EXPECTED_SUPPORT_RESISTANCE_MEMORY_FEATURE_COUNT
    assert len(SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS) == len(set(SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS))
    assert missing_support_resistance_memory_source_fields(SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS) == []
    assert missing_support_resistance_memory_source_fields(["ctx_cont.sr_nearest_pivot_abs_atr"])[:1] == [
        "ctx_cont.sr_support_proximity_exp"
    ]
    assert "chart.geometry_support_line_proximity_stack" in SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS
    assert "chart.smc_liquidity_reclaim_confirmation_long" in SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS
    # Declared cold-start contamination bound exported for callers
    # (origin: 5 * tau_slow at the declared slow decay 0.96, declared as 125).
    assert SR_MEMORY_WARMUP_ROWS == 125

    routing = {name: classify_entry_specialist_feature(name) for name in SUPPORT_RESISTANCE_MEMORY_FEATURE_NAMES}
    routing_gaps = {
        name: specialist
        for name, specialist in routing.items()
        if specialist not in {"smc_liquidity_encoder", "chart_geometry_encoder"}
    }

    assert routing_gaps == {}
    assert set(routing.values()) == {"smc_liquidity_encoder"}
