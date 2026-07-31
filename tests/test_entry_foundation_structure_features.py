import numpy as np
import pytest

from gx1.features.entry_foundation_structure_v1 import (
    FOUNDATION_STRUCTURE_SOURCE_FIELDS,
    build_entry_foundation_structure_layer,
    missing_foundation_structure_source_fields,
)
from gx1.features.entry_model_native_feature_layers_v1 import (
    CHART_LAYER_SOURCE_FIELDS,
    build_chart_layer,
)


def _matrix(names: list[str], n: int = 8) -> np.ndarray:
    x = np.zeros((n, len(names)), dtype=np.float32)
    idx = {name: i for i, name in enumerate(names)}

    def set_col(name: str, values) -> None:
        if name not in idx:
            return
        x[:, idx[name]] = np.asarray(values, dtype=np.float32)

    set_col("snap.smc_swing_state", [0, 0, 1, 4, 2, 3, 3, 0])
    set_col("snap.smc_bos_up", [0, 0, 1, 0, 0, 0, 0, 0])
    set_col("snap.smc_bos_down", [0, 0, 0, 0, 0, 1, 0, 0])
    set_col("snap.smc_choch", [0, 0, 0, 0, 1, 0, 0, 0])
    set_col("ctx_cont.smc_choch_recent_tau12", [0, 0, 0, 0, 1, 0.5, 0.25, 0])
    set_col("ctx_cont.smc_choch_recent_tau24", [0, 0, 0, 0, 1, 0.7, 0.4, 0.1])
    set_col("ctx_cont.smc_bos_pressure_last12", [0, 0.1, 1, 0.5, 0, 0.3, 0.1, 0])
    set_col("ctx_cont.smc_bos_pressure_last48", [0, 0.2, 1, 0.7, 0.1, 0.5, 0.2, 0])
    set_col("ctx_cont.dist_last_swing_high_atr", [0.2, 0.3, 0.1, 1.0, 0.2, 0.1, 0.3, 0.4])
    set_col("ctx_cont.dist_last_swing_low_atr", [1.0, 0.1, 0.3, 0.5, 0.4, 0.2, 0.1, 0.3])
    set_col("ctx_cont.bars_since_swing_high", [2, 3, 1, 8, 2, 1, 3, 4])
    set_col("ctx_cont.bars_since_swing_low", [8, 1, 2, 5, 4, 2, 1, 3])
    set_col("ctx_cont.retracement_from_last_impulse", [0.1, 0.4, 0.3, 0.8, 0.5, 0.2, 0.4, 0.1])
    set_col("ctx_cont.struct_pullback_depth_h1_v3", [0.1, 0.5, 0.3, 0.8, 0.4, 0.2, 0.5, 0.1])
    set_col("ctx_cont.struct_pullback_depth_h4_v3", [0.2, 0.4, 0.2, 0.7, 0.5, 0.3, 0.4, 0.2])
    set_col("ctx_cont._v1h1_ema_diff", [-2, -1, 2, 1, -1, -2, -1, 2])
    set_col("ctx_cont._v1h4_ema_diff", [-1, -1, 2, 1, -2, -2, -1, 2])
    set_col("ctx_cont.d1_ema_slope_20_canon_v2", [-1, 0, 1, 1, -1, -1, -1, 1])
    set_col("ctx_cont.m15_trend_sign_canon_v2", [-1, 1, 1, 1, -1, -1, -1, 1])
    set_col("ctx_cont.regime_stack_sum_v3", [-2, 1, 2, 1, -2, -3, -2, 2])
    set_col("snap.smc_sweep_up", [0, 0, 0, 0, 0, 0, 1, 0])
    set_col("snap.smc_sweep_down", [0, 1, 0, 0, 0, 0, 0, 0])
    set_col("snap.smc_bars_since_sweep", [96, 0, 1, 2, 3, 4, 0, 1])
    set_col("snap.smc_sweep_size_atr", [0, 2, 1, 0, 0, 0, 1.5, 0.5])
    set_col("ctx_cont.smc_sweep_bull_pressure_last12", [0, 0.5, 0.2, 0, 0, 0, 1, 0.5])
    set_col("ctx_cont.smc_sweep_bull_pressure_last48", [0, 1, 0.5, 0, 0, 0, 0.2, 0.1])
    set_col("ctx_cont.smc_sweep_size_recent_tau12", [0, 1, 0.5, 0, 0, 0, 0.5, 0.2])
    set_col("ctx_cont.smc_sweep_recency_tau24", [0, 1, 0.8, 0.5, 0.2, 0.1, 1, 0.8])
    set_col("snap.body_pct", [0.8, 0.1, 0.5, 0.8, 0.8, 0.7, 0.1, 0.6])
    set_col("ctx_cont.sr_support_proximity_exp", [0.1, 1, 0.6, 0.2, 0.1, 0.2, 0.3, 0.4])
    set_col("ctx_cont.sr_resistance_proximity_exp", [0.2, 0.2, 0.3, 0.5, 0.7, 1, 1, 0.4])
    set_col("ctx_cont.H1_range_compression_ratio", [1.0, 0.8, 0.5, 0.4, 0.7, 0.9, 0.6, 0.8])
    set_col("ctx_cont.M15_range_compression_ratio", [1.0, 0.7, 0.5, 0.4, 0.6, 0.9, 0.6, 0.8])
    set_col("snap._v1_bb_squeeze_20_2", [0.0, -0.2, -0.8, -1.0, -0.4, -0.2, -0.8, -0.2])
    set_col("snap.atr_z", [0.1, 0.2, 2, 3, 0.5, 0.3, 2, 0.2])
    set_col("snap.rvol_20", [0.1, 0.3, 2, 3, 0.5, 0.3, 2, 0.2])
    set_col("snap.vol_ratio_5_20", [0.1, 0.3, 2, 3, 0.5, 0.3, 2, 0.2])
    set_col("ctx_cont.m15_trend_age_bars_norm_v2", [0, 0.1, 0.2, 0.4, 0.5, 0.8, 0.9, 0.1])
    set_col("ctx_cont.h1_trend_age_bars_norm_v2", [0, 0.2, 0.3, 0.5, 0.7, 0.9, 1, 0.2])
    set_col("ctx_cont.h4_trend_age_bars_norm_v2", [0, 0.1, 0.2, 0.4, 0.5, 0.8, 0.9, 0.1])
    set_col("ctx_cont.is_ASIA", [1, 1, 0, 0, 0, 0, 0, 0])
    set_col("ctx_cont.is_asia_eu_overlap", [0, 0, 1, 0, 0, 0, 0, 0])
    set_col("ctx_cont.is_eu_us_overlap", [0, 0, 0, 1, 0, 0, 0, 0])
    set_col("ctx_cont.is_eu_only", [0, 0, 0, 0, 1, 0, 0, 0])
    set_col("ctx_cont.is_us_only", [0, 0, 0, 0, 0, 1, 1, 0])
    return x


def test_foundation_structure_layer_exposes_required_families_and_age_logic() -> None:
    names = [
        "snap.smc_swing_state",
        "snap.smc_bos_up",
        "snap.smc_bos_down",
        "snap.smc_choch",
        "ctx_cont.smc_choch_recent_tau12",
        "ctx_cont.smc_choch_recent_tau24",
        "ctx_cont.smc_bos_pressure_last12",
        "ctx_cont.smc_bos_pressure_last48",
        "ctx_cont.dist_last_swing_high_atr",
        "ctx_cont.dist_last_swing_low_atr",
        "ctx_cont.bars_since_swing_high",
        "ctx_cont.bars_since_swing_low",
        "ctx_cont.retracement_from_last_impulse",
        "ctx_cont.struct_pullback_depth_h1_v3",
        "ctx_cont.struct_pullback_depth_h4_v3",
        "ctx_cont._v1h1_ema_diff",
        "ctx_cont._v1h4_ema_diff",
        "ctx_cont.d1_ema_slope_20_canon_v2",
        "ctx_cont.m15_trend_sign_canon_v2",
        "ctx_cont.regime_stack_sum_v3",
        "snap.smc_sweep_up",
        "snap.smc_sweep_down",
        "snap.smc_bars_since_sweep",
        "snap.smc_sweep_size_atr",
        "ctx_cont.smc_sweep_bull_pressure_last12",
        "ctx_cont.smc_sweep_bull_pressure_last48",
        "ctx_cont.smc_sweep_size_recent_tau12",
        "ctx_cont.smc_sweep_recency_tau24",
        "snap.body_pct",
        "ctx_cont.sr_support_proximity_exp",
        "ctx_cont.sr_resistance_proximity_exp",
        "ctx_cont.H1_range_compression_ratio",
        "ctx_cont.M15_range_compression_ratio",
        "snap._v1_bb_squeeze_20_2",
        "snap.atr_z",
        "snap.rvol_20",
        "snap.vol_ratio_5_20",
        "ctx_cont.m15_trend_age_bars_norm_v2",
        "ctx_cont.h1_trend_age_bars_norm_v2",
        "ctx_cont.h4_trend_age_bars_norm_v2",
        "ctx_cont.is_ASIA",
        "ctx_cont.is_asia_eu_overlap",
        "ctx_cont.is_eu_us_overlap",
        "ctx_cont.is_eu_only",
        "ctx_cont.is_us_only",
    ]
    x = _matrix(names)
    out, out_names = build_entry_foundation_structure_layer(x, names)
    idx = {name: i for i, name in enumerate(out_names)}

    assert out.shape == (8, len(out_names))
    assert np.isfinite(out).all()
    for required in [
        "chart.foundation_hh_state",
        "chart.foundation_hl_state",
        "chart.foundation_lh_state",
        "chart.foundation_ll_state",
        "chart.foundation_bos_up_age_bars",
        "chart.foundation_bos_down_age_bars",
        "chart.foundation_choch_age_bars",
        "chart.foundation_sweep_low_reclaim_up_proxy",
        "chart.foundation_sweep_high_reclaim_down_proxy",
        "chart.foundation_compression_release_trigger",
        "chart.foundation_impulse_direction",
        "chart.foundation_pullback_phase_up",
        "chart.foundation_asia_x_hh_state",
        "chart.foundation_eu_x_hl_state",
        "chart.foundation_us_x_ll_state",
        "chart.foundation_overlap_x_lh_state",
    ]:
        assert required in idx

    np.testing.assert_allclose(out[:, idx["chart.foundation_bos_up_age_bars"]], [96, 96, 0, 1, 2, 3, 4, 5])
    np.testing.assert_allclose(out[:, idx["chart.foundation_bos_down_age_bars"]], [96, 96, 96, 96, 96, 0, 1, 2])
    np.testing.assert_allclose(out[:, idx["chart.foundation_choch_age_bars"]], [96, 96, 96, 96, 0, 1, 2, 3])
    assert out[1, idx["chart.foundation_sweep_low_reclaim_up_proxy"]] > 0.0
    assert out[6, idx["chart.foundation_sweep_high_reclaim_down_proxy"]] > 0.0
    assert out[2, idx["chart.foundation_compression_release_trigger"]] > 0.0


def test_foundation_structure_layer_uses_signed_smc_pressure_directionally() -> None:
    names = list(FOUNDATION_STRUCTURE_SOURCE_FIELDS)
    x = np.zeros((4, len(names)), dtype=np.float32)
    idx = {name: i for i, name in enumerate(names)}

    def set_col(name: str, values) -> None:
        x[:, idx[name]] = np.asarray(values, dtype=np.float32)

    set_col("ctx_cont.smc_bos_pressure_last12", [1.0, -1.0, 0.0, 0.0])
    set_col("ctx_cont.smc_bos_pressure_last48", [1.0, -1.0, 0.0, 0.0])
    set_col("ctx_cont.smc_sweep_bull_pressure_last12", [0.0, 0.0, 1.0, -1.0])
    set_col("ctx_cont.smc_sweep_bull_pressure_last48", [0.0, 0.0, 1.0, -1.0])
    set_col("snap.smc_bars_since_sweep", [96.0, 96.0, 0.0, 0.0])
    set_col("ctx_cont.smc_sweep_recency_tau24", [0.0, 0.0, 1.0, 1.0])
    set_col("snap.smc_sweep_size_atr", [0.0, 0.0, 1.0, 1.0])
    set_col("ctx_cont.smc_sweep_size_recent_tau12", [0.0, 0.0, 1.0, 1.0])
    set_col("snap.body_pct", [1.0, 1.0, 0.0, 0.0])
    set_col("ctx_cont.sr_support_proximity_exp", [0.0, 0.0, 1.0, 1.0])
    set_col("ctx_cont.sr_resistance_proximity_exp", [0.0, 0.0, 1.0, 1.0])
    set_col("ctx_cont.H1_range_compression_ratio", [1.0, 1.0, 1.0, 1.0])
    set_col("ctx_cont.M15_range_compression_ratio", [1.0, 1.0, 1.0, 1.0])

    out, out_names = build_entry_foundation_structure_layer(x, names)
    out_idx = {name: i for i, name in enumerate(out_names)}

    bos_balance = out[:, out_idx["chart.foundation_bos_recent_balance"]]
    sweep_balance = out[:, out_idx["chart.foundation_sweep_reclaim_balance_proxy"]]
    assert bos_balance[0] > 0.0
    assert bos_balance[1] < 0.0
    assert sweep_balance[2] > 0.0
    assert sweep_balance[3] < 0.0


def test_foundation_structure_source_dependency_contract_is_explicit() -> None:
    assert len(FOUNDATION_STRUCTURE_SOURCE_FIELDS) == len(set(FOUNDATION_STRUCTURE_SOURCE_FIELDS))
    assert "snap.smc_bos_up" in FOUNDATION_STRUCTURE_SOURCE_FIELDS
    assert "ctx_cont.smc_choch_recent_tau24" in FOUNDATION_STRUCTURE_SOURCE_FIELDS
    assert "ctx_cont.H1_range_compression_ratio" in FOUNDATION_STRUCTURE_SOURCE_FIELDS
    assert "ctx_cont.is_eu_us_overlap" in FOUNDATION_STRUCTURE_SOURCE_FIELDS

    assert missing_foundation_structure_source_fields(FOUNDATION_STRUCTURE_SOURCE_FIELDS) == []
    missing = missing_foundation_structure_source_fields(
        name for name in FOUNDATION_STRUCTURE_SOURCE_FIELDS if name != "snap.smc_bos_up"
    )
    assert missing == ["snap.smc_bos_up"]


def test_foundation_structure_layer_fails_closed_on_missing_or_nonfinite_sources() -> None:
    missing_names = [
        name for name in FOUNDATION_STRUCTURE_SOURCE_FIELDS if name != "snap.smc_bos_up"
    ]
    with pytest.raises(RuntimeError, match="FOUNDATION_STRUCTURE_SOURCE_FIELDS_MISSING"):
        build_entry_foundation_structure_layer(
            np.zeros((3, len(missing_names)), dtype=np.float32),
            missing_names,
        )

    names = list(FOUNDATION_STRUCTURE_SOURCE_FIELDS)
    for value in (np.nan, np.inf, -np.inf):
        x = _matrix(names)
        x[2, names.index("ctx_cont.sr_support_proximity_exp")] = value
        with pytest.raises(RuntimeError, match="FOUNDATION_STRUCTURE_SOURCE_NONFINITE"):
            build_entry_foundation_structure_layer(x, names)


def test_foundation_structure_layer_rejects_duplicate_or_misaligned_names() -> None:
    names = list(FOUNDATION_STRUCTURE_SOURCE_FIELDS)
    duplicate_names = [*names, names[0]]
    with pytest.raises(RuntimeError, match="FOUNDATION_STRUCTURE_FEATURE_NAMES_DUPLICATE"):
        build_entry_foundation_structure_layer(
            np.zeros((2, len(duplicate_names)), dtype=np.float32),
            duplicate_names,
        )
    with pytest.raises(RuntimeError, match="FOUNDATION_STRUCTURE_FEATURE_NAME_COUNT_MISMATCH"):
        build_entry_foundation_structure_layer(
            np.zeros((2, len(names) + 1), dtype=np.float32),
            names,
        )


def test_chart_layer_includes_foundation_features() -> None:
    names = list(CHART_LAYER_SOURCE_FIELDS)
    x = _matrix(names)
    chart_x, chart_names = build_chart_layer(x, names)

    assert chart_x.shape[0] == x.shape[0]
    assert "chart.foundation_hh_state" in chart_names
    assert "chart.foundation_bos_up_age_bars" in chart_names
    assert np.isfinite(chart_x).all()
