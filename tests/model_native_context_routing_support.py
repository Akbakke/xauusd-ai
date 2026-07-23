"""Deterministic context-routing fixtures.

V24's 82 temporal aliases are fixture evidence, not a production constant.
Production derives the overlap from each artifact's ordered signal names.
"""
from __future__ import annotations

import json

from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT,
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    classify_entry_specialist_feature,
    model_native_context_temporal_alias_policy,
)


V24_TEMPORAL_ALIAS_CTX_CONT_FIELDS = (
    "atr_bps",
    "spread_bps",
    "D1_dist_from_ema200_atr",
    "D1_atr_percentile_252",
    "micro_momentum_3",
    "micro_momentum_5",
    "wick_ratio",
    "distance_ema_fast",
    "dist_last_swing_high_atr",
    "dist_last_swing_low_atr",
    "is_ASIA",
    "minutes_since_session_open",
    "minutes_to_next_session_boundary",
    "session_tradable",
    "_v1h1_atr",
    "_v1h4_ema_diff",
    "_v1h4_atr",
    "_v1h4_rsi14_z",
    "d1_atr14_canon_v2",
    "d1_rsi14_canon_v2",
    "d1_ema_slope_20_canon_v2",
    "d1_close_pct_in_20day_range_canon_v2",
    "d1_pct_change_5_canon_v2",
    "m15_rsi14_canon_v2",
    "m15_trend_sign_canon_v2",
    "hour_sin",
    "hour_cos",
    "dow_cos",
    "is_us_only",
    "vol_pct_m5_1yr",
    "vol_pct_h1_1yr",
    "dist_to_R1_atr",
    "dist_to_m5_hi_atr",
    "dist_to_m15_hi_atr",
    "dist_to_m15_lo_atr",
    "dist_to_h1_lo_atr",
    "dist_to_h4_hi_atr",
    "dist_to_h4_lo_atr",
    "dist_to_d1_lo_atr",
    "dip_confirmed_m5_v3",
    "dip_confirmed_m15_v3",
    "dip_proximity_h1_v3",
    "dip_confirmed_h1_v3",
    "dip_proximity_h4_v3",
    "struct_continuation_up_m5_v3",
    "struct_continuation_down_m5_v3",
    "struct_pullback_depth_m5_v3",
    "struct_continuation_up_m15_v3",
    "struct_continuation_down_m15_v3",
    "struct_bounce_in_downtrend_m15_v3",
    "struct_pullback_depth_m15_v3",
    "struct_continuation_down_h1_v3",
    "struct_continuation_down_h4_v3",
    "struct_continuation_up_d1_v3",
    "struct_pullback_in_uptrend_d1_v3",
    "struct_continuation_down_d1_v3",
    "struct_bounce_in_downtrend_d1_v3",
    "struct_pullback_depth_d1_v3",
    "smc_choch_recent_tau12",
    "smc_choch_recent_tau24",
    "smc_bos_pressure_last12",
    "smc_sweep_bull_pressure_last12",
    "smc_sweep_bull_pressure_last48",
    "smc_sweep_size_recent_tau12",
    "smc_sweep_recency_tau24",
    "sr_resistance_proximity_exp",
    "liquidity_hi_nearest_abs_atr",
    "dip_confirmed_mean_5tf",
    "dip_confirmed_max_5tf",
    "dip_proximity_mean_h1h4d1",
    "m15_regime_class_id_v2",
    "h4_regime_class_id_v2",
    "d1_regime_class_id_v2",
    "m5_regime_class_id_v2",
    "m15_trend_age_bars_norm_v2",
    "h4_trend_age_bars_norm_v2",
    "d1_trend_age_bars_norm_v2",
    "regime_tf_agreement_v3",
    "regime_stack_sum_v3",
    "regime_divergence_flag_v3",
    "d1_dist_to_boundary_v3",
    "d1_trend_age_mature_flag_v3",
)
V24_TEMPORAL_ALIAS_SIGNAL_FIELDS = tuple(
    f"ctx_cont.{field}" for field in V24_TEMPORAL_ALIAS_CTX_CONT_FIELDS
)


_FILLER_PREFIX_BY_SPECIALIST = {
    "structure_swing_encoder": "structure.swing_fixture_",
    "smc_liquidity_encoder": "smc.liquidity_fixture_",
    "trend_ema_encoder": "trend.ema_fixture_",
    "vol_compression_encoder": "volatility.atr_fixture_",
    "momentum_flow_encoder": "momentum.flow_fixture_",
    "session_regime_encoder": "session.regime_fixture_",
    "chart_geometry_encoder": "chart.line_pattern_fixture_",
    "price_action_candle_encoder": "price_action.candle_fixture_",
}


def ordered_signal_names_for_specialist_indices(
    specialist_indices: dict[str, list[int]],
    *,
    temporal_alias_signal_fields: tuple[str, ...] = (
        V24_TEMPORAL_ALIAS_SIGNAL_FIELDS
    ),
) -> list[str]:
    width = sum(len(indices) for indices in specialist_indices.values())
    fields: list[str | None] = [None] * width
    available = {
        name: list(indices) for name, indices in specialist_indices.items()
    }
    for signal_field in temporal_alias_signal_fields:
        owner = classify_entry_specialist_feature(signal_field)
        fields[available[owner].pop(0)] = signal_field
    for specialist in MODEL_NATIVE_TRAINING_SPECIALISTS:
        for ordinal, index in enumerate(available[specialist]):
            fields[index] = f"{_FILLER_PREFIX_BY_SPECIALIST[specialist]}{ordinal:03d}"
    if any(field is None for field in fields):
        raise RuntimeError("MODEL_NATIVE_TEST_SIGNAL_FIELD_GAP")
    return [str(field) for field in fields]


def context_routing_for_ordered_signal_names(
    ordered_signal_names: list[str],
) -> dict:
    routing = json.loads(
        json.dumps(MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT)
    )
    routing["temporal_alias_policy"] = (
        model_native_context_temporal_alias_policy(ordered_signal_names)
    )
    return routing
