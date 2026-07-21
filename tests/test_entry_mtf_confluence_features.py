from collections import Counter

import numpy as np
import pytest

from gx1.features.entry_mtf_confluence_v1 import (
    MTF_CONFLUENCE_FEATURE_NAMES,
    MTF_CONFLUENCE_SOURCE_FIELDS,
    build_entry_mtf_confluence_layer,
    missing_mtf_confluence_source_fields,
)
from gx1.features.entry_specialist_feature_groups_v1 import classify_entry_specialist_feature


EXPECTED_MTF_CONFLUENCE_FEATURE_NAMES = (
    "trend.mtf_confluence_trend_direction_score",
    "trend.mtf_confluence_trend_tf_agreement",
    "trend.mtf_confluence_trend_tf_conflict",
    "trend.mtf_confluence_trend_m5_m15_h1_h4_d1_alignment",
    "trend.mtf_confluence_long_trend_bias",
    "trend.mtf_confluence_short_trend_bias",
    "chart.structure_swing_mtf_confluence_structure_direction_score",
    "chart.structure_swing_mtf_confluence_bos_alignment_up",
    "chart.structure_swing_mtf_confluence_bos_alignment_down",
    "chart.structure_swing_mtf_confluence_structure_conflict",
    "chart.structure_swing_mtf_confluence_pullback_abstain_pressure",
    "chart.smc_liquidity_mtf_confluence_sweep_reclaim_long",
    "chart.smc_liquidity_mtf_confluence_sweep_reclaim_short",
    "chart.smc_liquidity_mtf_confluence_false_breakout_long",
    "chart.smc_liquidity_mtf_confluence_false_breakout_short",
    "chart.smc_liquidity_mtf_confluence_liquidity_conflict",
    "chart.smc_liquidity_mtf_confluence_premium_discount_alignment",
    "chart.geometry_mtf_confluence_fib_sr_long_proximity",
    "chart.geometry_mtf_confluence_fib_sr_short_proximity",
    "chart.geometry_mtf_confluence_sr_balance",
    "chart.geometry_mtf_confluence_major_level_density",
    "session_regime.mtf_confluence_session_permission",
    "session_regime.mtf_confluence_regime_agreement",
    "session_regime.mtf_confluence_regime_conflict",
    "session_regime.mtf_confluence_spread_vol_abstain",
    "session_regime.mtf_confluence_session_regime_tradable_long",
    "session_regime.mtf_confluence_session_regime_tradable_short",
    "session_regime.mtf_confluence_long_agreement_score",
    "session_regime.mtf_confluence_short_agreement_score",
    "session_regime.mtf_confluence_direction_balance",
    "session_regime.mtf_confluence_conflict_score",
    "session_regime.mtf_confluence_abstain_score",
)


def _matrix(names: list[str], n: int = 8) -> np.ndarray:
    x = np.zeros((n, len(names)), dtype=np.float32)
    idx = {name: i for i, name in enumerate(names)}

    def set_col(name: str, values) -> None:
        x[:, idx[name]] = np.asarray(values, dtype=np.float32)

    bullish = [-1.2, 0.3, 2.0, 1.6, 1.5, -2.0, -1.6, 0.2]
    bearish = [1.2, -0.2, 0.4, 0.5, -1.4, -2.0, -1.8, 0.1]
    for name in (
        "snap._v1_ema_diff",
        "snap.ema20_slope",
        "snap.pos_vs_ema200",
        "ctx_cont.m15_trend_sign_canon_v2",
        "ctx_cont._v1h1_ema_diff",
        "ctx_cont._v1h1_slope5",
        "ctx_cont._v1h4_ema_diff",
        "ctx_cont._v1h4_slope5",
        "ctx_cont.d1_ema_slope_20_canon_v2",
        "ctx_cont.d1_pct_change_5_canon_v2",
    ):
        set_col(name, bullish)
    set_col("ctx_cont._v1h4_ema_diff", [-1.2, 0.3, 2.0, 1.6, -1.5, -2.0, -1.6, 0.2])
    set_col("ctx_cont.d1_ema_slope_20_canon_v2", bearish)
    set_col("ctx_cont.regime_tf_agreement_v3", [0.2, 0.5, 1.0, 0.9, 0.1, 1.0, 0.9, 0.4])
    set_col("ctx_cont.regime_stack_sum_v3", [-2, 1, 3, 2, 0, -3, -2, 0])
    set_col("ctx_cont.regime_divergence_flag_v3", [0.2, 0.1, 0.0, 0.0, 1.0, 0.0, 0.1, 0.2])

    set_col("chart.foundation_hh_state", [0, 0.5, 1, 1, 0.8, 0, 0, 0.2])
    set_col("chart.foundation_hl_state", [0, 0.5, 1, 1, 0.8, 0, 0, 0.2])
    set_col("chart.foundation_lh_state", [0.5, 0, 0, 0, 0.8, 1, 1, 0.2])
    set_col("chart.foundation_ll_state", [0.5, 0, 0, 0, 0.8, 1, 1, 0.2])
    set_col("chart.foundation_structure_up_minus_down", [-1, 0.4, 2, 1.4, 0.0, -2, -1.6, 0])
    set_col("chart.foundation_bos_up_age_bars", [96, 8, 0, 1, 2, 96, 96, 12])
    set_col("chart.foundation_bos_down_age_bars", [8, 96, 96, 96, 1, 0, 1, 12])
    set_col("chart.foundation_bos_up_recent_tau24", [0, 0.4, 1, 0.8, 0.8, 0, 0, 0.2])
    set_col("chart.foundation_bos_down_recent_tau24", [0.4, 0, 0, 0, 0.8, 1, 0.8, 0.2])
    set_col("chart.foundation_bos_recent_balance", [-0.5, 0.3, 1.0, 0.8, 0.0, -1.0, -0.8, 0.0])
    set_col("chart.foundation_choch_recent_tau24", [0.4, 0.1, 0.0, 0.0, 1.0, 0.0, 0.1, 0.2])
    set_col("chart.foundation_pullback_phase_up", [0, 0.3, 0.4, 0.7, 0.8, 0, 0, 0.2])
    set_col("chart.foundation_pullback_phase_down", [0.3, 0, 0, 0, 0.8, 0.5, 0.7, 0.2])
    set_col("chart.foundation_pullback_depth_norm", [0.2, 0.3, 0.4, 0.618, 0.9, 0.4, 0.618, 0.2])
    for tf in ("m5", "m15", "h1", "h4", "d1"):
        set_col(f"ctx_cont.struct_continuation_up_{tf}_v3", [0, 0.4, 1.0, 0.9, 0.7, 0, 0, 0.2])
        set_col(f"ctx_cont.struct_continuation_down_{tf}_v3", [0.4, 0, 0, 0, 0.7, 1.0, 0.9, 0.2])

    set_col("chart.foundation_sweep_low_reclaim_up_proxy", [0, 1, 5, 4, 4, 0, 0, 0])
    set_col("chart.foundation_sweep_high_reclaim_down_proxy", [1, 0, 0, 0, 4, 5, 4, 0])
    set_col("chart.foundation_false_breakout_low_followthrough_up_proxy", [0, 1, 4, 5, 2, 0, 0, 0])
    set_col("chart.foundation_false_breakout_high_followthrough_down_proxy", [1, 0, 0, 0, 2, 5, 4, 0])
    set_col("chart.foundation_sweep_reclaim_balance_proxy", [-1, 1, 5, 4, 0, -5, -4, 0])
    set_col("snap.smc_sweep_up", [0, 0, 0, 0, 1, 1, 1, 0])
    set_col("snap.smc_sweep_down", [0, 0, 1, 1, 1, 0, 0, 0])
    set_col("snap.smc_sweep_size_atr", [0, 0.4, 1, 1, 1, 1, 1, 0])
    set_col("snap.smc_bars_since_sweep", [96, 12, 1, 2, 1, 1, 2, 24])
    set_col("snap.smc_premium_discount", [0.5, 0.4, 0.2, 0.382, 0.5, 0.8, 0.618, 0.5])
    set_col("ctx_cont.smc_sweep_recency_tau24", [0, 0.3, 1, 0.8, 1, 1, 0.8, 0.2])
    set_col("ctx_cont.smc_sweep_size_recent_tau12", [0, 0.4, 1, 1, 1, 1, 1, 0])

    set_col("ctx_cont.sr_nearest_pivot_abs_atr", [2, 1, 0.2, 0.2, 0.1, 0.2, 0.2, 1])
    set_col("ctx_cont.sr_support_proximity_exp", [0.2, 0.5, 1, 0.9, 0.8, 0.1, 0.1, 0.3])
    set_col("ctx_cont.sr_resistance_proximity_exp", [0.5, 0.2, 0.1, 0.2, 0.8, 1, 0.9, 0.3])
    set_col("ctx_cont.sr_support_minus_resistance_prox", [-0.2, 0.2, 0.8, 0.6, 0.0, -0.8, -0.6, 0.0])
    for name in ("ctx_cont.dist_to_S1_atr", "ctx_cont.dist_to_S2_atr"):
        set_col(name, [2, 1, 0.1, 0.2, 0.2, 3, 3, 1])
    for name in ("ctx_cont.dist_to_R1_atr", "ctx_cont.dist_to_R2_atr"):
        set_col(name, [1, 2, 3, 3, 0.2, 0.1, 0.2, 1])
    for tf in ("m5", "m15", "h1", "h4", "d1"):
        set_col(f"ctx_cont.dist_to_{tf}_lo_atr", [2, 1, 0.1, 0.2, 0.2, 3, 3, 1])
        set_col(f"ctx_cont.dist_to_{tf}_hi_atr", [1, 2, 3, 3, 0.2, 0.1, 0.2, 1])
    set_col("ctx_cont.liquidity_lo_nearest_abs_atr", [2, 1, 0.1, 0.2, 0.2, 3, 3, 1])
    set_col("ctx_cont.liquidity_hi_nearest_abs_atr", [1, 2, 3, 3, 0.2, 0.1, 0.2, 1])
    set_col("ctx_cont.retracement_from_last_impulse", [0.2, 0.4, 0.618, 0.618, 0.5, 0.618, 0.618, 0.2])
    set_col("ctx_cont.d1_close_pct_in_20day_range_canon_v2", [0.3, 0.4, 0.618, 0.618, 0.5, 0.618, 0.618, 0.3])

    set_col("ctx_cont.minutes_since_session_open", [0, 60, 120, 180, 5, 120, 180, 240])
    set_col("ctx_cont.minutes_to_next_session_boundary", [10, 120, 180, 120, 5, 180, 120, 120])
    set_col("ctx_cont.session_change_flag", [1, 0, 0, 0, 1, 0, 0, 0])
    set_col("ctx_cont.session_tradable", [0, 1, 1, 1, 1, 1, 1, 1])
    set_col("ctx_cont.is_ASIA", [1, 0, 0, 0, 0, 0, 0, 0])
    set_col("ctx_cont.is_asia_eu_overlap", [0, 0, 0, 0, 0, 0, 0, 0])
    set_col("ctx_cont.is_eu_us_overlap", [0, 0, 0, 1, 0, 0, 1, 0])
    set_col("ctx_cont.is_eu_only", [0, 1, 1, 0, 1, 0, 0, 0])
    set_col("ctx_cont.is_us_only", [0, 0, 0, 0, 0, 1, 0, 1])
    set_col("ctx_cont.spread_bps", [8, 2, 1, 1, 12, 1, 1, 2])
    set_col("ctx_cont.atr_bps", [20, 20, 20, 20, 20, 20, 20, 20])
    set_col("ctx_cat.spread_bucket", [2, 0, 0, 0, 2, 0, 0, 0])
    set_col("ctx_cat.vol_regime_id", [2, 1, 1, 1, 3, 1, 1, 1])
    set_col("ctx_cont.vol_pct_m5_1yr", [0.8, 0.3, 0.3, 0.4, 0.95, 0.3, 0.4, 0.3])
    set_col("ctx_cont.vol_pct_h1_1yr", [0.8, 0.3, 0.3, 0.4, 0.95, 0.3, 0.4, 0.3])
    set_col("ctx_cont.D1_atr_percentile_252", [0.8, 0.4, 0.4, 0.5, 0.95, 0.4, 0.5, 0.4])
    set_col("ctx_cont.d1_regime_changed_flag_v3", [1, 0, 0, 0, 1, 0, 0, 0])
    set_col("ctx_cont.bars_since_d1_regime_change_v3", [0, 0.5, 0.9, 0.8, 0, 0.9, 0.8, 0.6])
    return x


def test_mtf_confluence_feature_contract_is_stable() -> None:
    assert len(MTF_CONFLUENCE_FEATURE_NAMES) == 32
    assert MTF_CONFLUENCE_FEATURE_NAMES == EXPECTED_MTF_CONFLUENCE_FEATURE_NAMES


def test_mtf_confluence_layer_builds_family_agreement_features() -> None:
    names = list(MTF_CONFLUENCE_SOURCE_FIELDS)
    out, out_names = build_entry_mtf_confluence_layer(_matrix(names), names)
    idx = {name: i for i, name in enumerate(out_names)}

    assert out.shape == (8, len(MTF_CONFLUENCE_FEATURE_NAMES))
    assert tuple(out_names) == MTF_CONFLUENCE_FEATURE_NAMES
    assert np.isfinite(out).all()
    assert out[2, idx["session_regime.mtf_confluence_long_agreement_score"]] > out[2, idx["session_regime.mtf_confluence_short_agreement_score"]]
    assert out[5, idx["session_regime.mtf_confluence_short_agreement_score"]] > out[5, idx["session_regime.mtf_confluence_long_agreement_score"]]
    assert out[2, idx["trend.mtf_confluence_trend_tf_agreement"]] > out[4, idx["trend.mtf_confluence_trend_tf_agreement"]]
    assert out[4, idx["trend.mtf_confluence_trend_tf_conflict"]] > out[2, idx["trend.mtf_confluence_trend_tf_conflict"]]
    assert out[2, idx["chart.structure_swing_mtf_confluence_bos_alignment_up"]] > 0.50
    assert out[5, idx["chart.structure_swing_mtf_confluence_bos_alignment_down"]] > 0.50
    assert out[2, idx["chart.smc_liquidity_mtf_confluence_sweep_reclaim_long"]] > out[5, idx["chart.smc_liquidity_mtf_confluence_sweep_reclaim_long"]]
    assert out[5, idx["chart.smc_liquidity_mtf_confluence_sweep_reclaim_short"]] > out[2, idx["chart.smc_liquidity_mtf_confluence_sweep_reclaim_short"]]
    assert out[2, idx["chart.geometry_mtf_confluence_fib_sr_long_proximity"]] > out[5, idx["chart.geometry_mtf_confluence_fib_sr_long_proximity"]]
    assert out[5, idx["chart.geometry_mtf_confluence_fib_sr_short_proximity"]] > out[2, idx["chart.geometry_mtf_confluence_fib_sr_short_proximity"]]
    assert out[4, idx["session_regime.mtf_confluence_conflict_score"]] > out[2, idx["session_regime.mtf_confluence_conflict_score"]]
    assert out[4, idx["session_regime.mtf_confluence_abstain_score"]] > out[2, idx["session_regime.mtf_confluence_abstain_score"]]


def test_mtf_confluence_layer_rejects_nonfinite_inputs() -> None:
    names = list(MTF_CONFLUENCE_SOURCE_FIELDS)
    for field, value in (
        ("snap._v1_ema_diff", np.nan),
        ("ctx_cont._v1h1_ema_diff", np.inf),
        ("ctx_cont.spread_bps", -np.inf),
    ):
        x = _matrix(names)
        x[2, names.index(field)] = value
        with pytest.raises(RuntimeError, match="MTF_CONFLUENCE_SOURCE_NONFINITE"):
            build_entry_mtf_confluence_layer(x, names)


def test_mtf_confluence_layer_rejects_invalid_atr_and_spread() -> None:
    names = list(MTF_CONFLUENCE_SOURCE_FIELDS)
    zero_atr = _matrix(names)
    zero_atr[4, names.index("ctx_cont.atr_bps")] = 0.0
    with pytest.raises(RuntimeError, match="MTF_CONFLUENCE_ATR_BPS_NOT_POSITIVE"):
        build_entry_mtf_confluence_layer(zero_atr, names)

    negative_spread = _matrix(names)
    negative_spread[4, names.index("ctx_cont.spread_bps")] = -0.1
    with pytest.raises(RuntimeError, match="MTF_CONFLUENCE_SPREAD_BPS_NEGATIVE"):
        build_entry_mtf_confluence_layer(negative_spread, names)


def test_mtf_confluence_layer_does_not_read_future_rows() -> None:
    names = list(MTF_CONFLUENCE_SOURCE_FIELDS)
    x = _matrix(names)
    baseline, _ = build_entry_mtf_confluence_layer(x, names)

    for future_start in range(1, x.shape[0]):
        changed = x.copy()
        changed[future_start:, :] = 99.0
        mutated, _ = build_entry_mtf_confluence_layer(changed, names)

        np.testing.assert_allclose(mutated[:future_start], baseline[:future_start], rtol=0.0, atol=0.0)


def test_mtf_confluence_source_contract_requires_exact_names_and_routes_outputs() -> None:
    assert missing_mtf_confluence_source_fields(MTF_CONFLUENCE_SOURCE_FIELDS) == []
    alias_substitutions = missing_mtf_confluence_source_fields(
        [
            "spread_bps" if name == "ctx_cont.spread_bps" else
            "spread_bucket" if name == "ctx_cat.spread_bucket" else
            "_v1_ema_diff" if name == "snap._v1_ema_diff" else
            name
            for name in MTF_CONFLUENCE_SOURCE_FIELDS
        ]
    )
    assert set(alias_substitutions) == {
        "snap._v1_ema_diff",
        "ctx_cont.spread_bps",
        "ctx_cat.spread_bucket",
    }
    alias_names = [
        "spread_bps" if name == "ctx_cont.spread_bps" else name
        for name in MTF_CONFLUENCE_SOURCE_FIELDS
    ]
    with pytest.raises(RuntimeError, match="MTF_CONFLUENCE_SOURCE_FIELDS_MISSING"):
        build_entry_mtf_confluence_layer(
            _matrix(list(MTF_CONFLUENCE_SOURCE_FIELDS)),
            alias_names,
        )
    missing = missing_mtf_confluence_source_fields(
        name for name in MTF_CONFLUENCE_SOURCE_FIELDS if name != "ctx_cont._v1h4_slope5"
    )
    assert missing == ["ctx_cont._v1h4_slope5"]

    route_counts = Counter(classify_entry_specialist_feature(name) for name in MTF_CONFLUENCE_FEATURE_NAMES)
    assert route_counts == {
        "trend_ema_encoder": 6,
        "structure_swing_encoder": 5,
        "smc_liquidity_encoder": 6,
        "chart_geometry_encoder": 4,
        "session_regime_encoder": 11,
    }
