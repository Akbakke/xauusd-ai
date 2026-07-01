"""Entry feature-to-specialist grouping contract.

The goal is to make the sequential specialist-AI design operational before
training: every emitted seq/snap field gets one primary encoder group, and the
foundation structure families are explicitly assigned to the specialist that
should learn that market mechanism.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable


SPECIALIST_GROUPS: "OrderedDict[str, dict[str, str]]" = OrderedDict(
    [
        (
            "neutral_bridge_anchor",
            {
                "encoder": "neutral_bridge_anchor",
                "role": "Frozen/allowlisted neutral XGB bridge priors until a new bridge is approved.",
            },
        ),
        (
            "structure_swing_encoder",
            {
                "encoder": "structure_swing_encoder",
                "role": "HH/HL/LH/LL, BOS/CHoCH age, swing structure, impulse and pullback phase.",
            },
        ),
        (
            "smc_liquidity_encoder",
            {
                "encoder": "smc_liquidity_encoder",
                "role": "Sweep/reclaim, false breakout, S/R, level proximity, wick liquidity and premium/discount.",
            },
        ),
        (
            "trend_ema_encoder",
            {
                "encoder": "trend_ema_encoder",
                "role": "EMA stack, slope, trend proxy, price-vs-EMA and multi-timeframe trend agreement.",
            },
        ),
        (
            "vol_compression_encoder",
            {
                "encoder": "vol_compression_encoder",
                "role": "ATR, volatility regime, range compression, squeeze, expansion and release.",
            },
        ),
        (
            "momentum_flow_encoder",
            {
                "encoder": "momentum_flow_encoder",
                "role": "Returns, impulse follow-through, dip confirmation, CLV and recent directional flow.",
            },
        ),
        (
            "session_regime_encoder",
            {
                "encoder": "session_regime_encoder",
                "role": "Asia/EU/US/overlap, session boundary, time pockets and session x structure interactions.",
            },
        ),
        (
            "chart_geometry_encoder",
            {
                "encoder": "chart_geometry_encoder",
                "role": "Research challenger for numeric trendlines, support/resistance channels, Fibonacci zones, EMA crosses and chart patterns.",
            },
        ),
        (
            "price_action_candle_encoder",
            {
                "encoder": "price_action_candle_encoder",
                "role": "Candle body/wick/range shape when it is not already a liquidity or structure interaction.",
            },
        ),
    ]
)


REQUIRED_TRAINING_SPECIALISTS = (
    "structure_swing_encoder",
    "smc_liquidity_encoder",
    "trend_ema_encoder",
    "vol_compression_encoder",
    "momentum_flow_encoder",
    "session_regime_encoder",
)

CHALLENGER_SEQ215_TRAINING_SPECIALISTS = (
    *REQUIRED_TRAINING_SPECIALISTS,
    "chart_geometry_encoder",
    "price_action_candle_encoder",
)

SMART_SEQ520_CANDIDATE_SPECIALISTS = CHALLENGER_SEQ215_TRAINING_SPECIALISTS
SMART_SEQ520_EXPECTED_SIGNAL_DIM = 520
SMART_SEQ520_EXPECTED_SELECTED_FEATURE_COUNT = 479
SMART_SEQ520_EXPECTED_SMART_FEATURE_COUNT = 305

SPECIALIST_FUSION_ACTIVE_HEADS = (
    "direction",
    "tradable",
    "path_quality",
    "mfe_first_n",
    "bad_path",
    "clean_edge",
    "survival",
    "tf_agreement",
    "path_quality_log_var",
    "position_size",
    "dip",
    "forecast",
    "timing",
    "tail_risk",
    "vol_forecast",
    "mtf_direction",
)

SPECIALIST_FUSION_BLOCKED_HEADS = ("hold_horizon",)

NEUTRAL_BRIDGE_FIELDS = {
    "p_long",
    "p_short",
    "p_flat",
    "p_hat",
    "uncertainty_score",
    "margin_top1_top2",
    "entropy",
}

CONTEXT_FEATURE_SPECIALIST_OVERRIDES = {
    "ctx_cont.spread_bps": "session_regime_encoder",
    "ctx_cat.spread_bucket": "session_regime_encoder",
    "spread_bps": "session_regime_encoder",
    "spread_bucket": "session_regime_encoder",
    "ctx_cont.is_us_only": "session_regime_encoder",
    "ctx_cont.is_eu_only": "session_regime_encoder",
    "ctx_cont.is_asia_eu_overlap": "session_regime_encoder",
    "ctx_cont.is_eu_us_overlap": "session_regime_encoder",
    "is_us_only": "session_regime_encoder",
    "is_eu_only": "session_regime_encoder",
    "is_asia_eu_overlap": "session_regime_encoder",
    "is_eu_us_overlap": "session_regime_encoder",
    "ctx_cat.session_id": "session_regime_encoder",
    "ctx_cat.vol_regime_id": "session_regime_encoder",
    "ctx_cont.m5_regime_class_id_v2": "session_regime_encoder",
    "ctx_cont.m15_regime_class_id_v2": "session_regime_encoder",
    "ctx_cont.h1_regime_class_id_v2": "session_regime_encoder",
    "ctx_cont.h4_regime_class_id_v2": "session_regime_encoder",
    "ctx_cont.d1_regime_class_id_v2": "session_regime_encoder",
    "ctx_cont.regime_tf_agreement_v3": "session_regime_encoder",
    "ctx_cont.regime_stack_sum_v3": "session_regime_encoder",
    "ctx_cont.regime_divergence_flag_v3": "session_regime_encoder",
    "ctx_cont.d1_dist_to_boundary_v3": "session_regime_encoder",
    "ctx_cont.d1_regime_changed_flag_v3": "session_regime_encoder",
    "ctx_cont.bars_since_d1_regime_change_v3": "session_regime_encoder",
    "session_id": "session_regime_encoder",
    "vol_regime_id": "session_regime_encoder",
    "m5_regime_class_id_v2": "session_regime_encoder",
    "m15_regime_class_id_v2": "session_regime_encoder",
    "h1_regime_class_id_v2": "session_regime_encoder",
    "h4_regime_class_id_v2": "session_regime_encoder",
    "d1_regime_class_id_v2": "session_regime_encoder",
    "regime_tf_agreement_v3": "session_regime_encoder",
    "regime_stack_sum_v3": "session_regime_encoder",
    "regime_divergence_flag_v3": "session_regime_encoder",
    "d1_dist_to_boundary_v3": "session_regime_encoder",
    "d1_regime_changed_flag_v3": "session_regime_encoder",
    "bars_since_d1_regime_change_v3": "session_regime_encoder",
    "ctx_cat.atr_bucket": "vol_compression_encoder",
    "atr_bucket": "vol_compression_encoder",
    "ctx_cont.dip_proximity_h1_v3": "momentum_flow_encoder",
    "ctx_cont.dip_proximity_h4_v3": "momentum_flow_encoder",
    "ctx_cont.dip_proximity_d1_v3": "momentum_flow_encoder",
    "ctx_cont.dip_proximity_mean_h1h4d1": "momentum_flow_encoder",
    "dip_proximity_h1_v3": "momentum_flow_encoder",
    "dip_proximity_h4_v3": "momentum_flow_encoder",
    "dip_proximity_d1_v3": "momentum_flow_encoder",
    "dip_proximity_mean_h1h4d1": "momentum_flow_encoder",
}

FOUNDATION_REQUIREMENT_PATTERNS = OrderedDict(
    [
        (
            "hh_hl_lh_ll",
            {
                "expected_specialist": "structure_swing_encoder",
                "tokens": ("foundation_hh_state", "foundation_hl_state", "foundation_lh_state", "foundation_ll_state"),
            },
        ),
        (
            "bos_choch_age",
            {
                "expected_specialist": "structure_swing_encoder",
                "tokens": ("foundation_bos_", "foundation_choch_", "foundation_bars_since_structure_break"),
            },
        ),
        (
            "sweep_reclaim",
            {
                "expected_specialist": "smc_liquidity_encoder",
                "tokens": ("foundation_sweep", "foundation_false_breakout"),
            },
        ),
        (
            "compression_expansion",
            {
                "expected_specialist": "vol_compression_encoder",
                "tokens": ("foundation_compression", "foundation_expansion"),
            },
        ),
        (
            "impulse_pullback_phase",
            {
                "expected_specialist": "structure_swing_encoder",
                "tokens": ("foundation_impulse", "foundation_pullback"),
            },
        ),
        (
            "session_x_structure",
            {
                "expected_specialist": "session_regime_encoder",
                "tokens": ("foundation_asia_x_", "foundation_eu_x_", "foundation_us_x_", "foundation_overlap_x_"),
            },
        ),
    ]
)

FOUNDATION_OBJECTIVE_SPECIALISTS = OrderedDict(
    [
        ("hh_hl_lh_ll", "structure_swing_encoder"),
        ("bos_choch_age", "structure_swing_encoder"),
        ("sweep_reclaim_false_breakout", "smc_liquidity_encoder"),
        ("compression_expansion", "vol_compression_encoder"),
        ("impulse_pullback_phase", "structure_swing_encoder"),
        ("session_x_structure", "session_regime_encoder"),
    ]
)

SPECIALIST_MODEL_CONTRACT = OrderedDict(
    [
        (
            "structure_swing_encoder",
            {
                "model_role": "market_structure_sequence_ai",
                "owned_objectives": ("hh_hl_lh_ll", "bos_choch_age", "impulse_pullback_phase"),
                "primary_signal_families": (
                    "HH/HL/LH/LL state",
                    "BOS/CHoCH age",
                    "swing distance",
                    "impulse/pullback phase",
                    "structure break recency",
                ),
                "supports_heads": ("direction", "tradable", "timing", "mtf_direction", "path_quality"),
            },
        ),
        (
            "smc_liquidity_encoder",
            {
                "model_role": "smc_liquidity_sequence_ai",
                "owned_objectives": ("sweep_reclaim_false_breakout",),
                "primary_signal_families": (
                    "sweep reclaim",
                    "false breakout",
                    "support/resistance proximity",
                    "wick liquidity",
                    "premium/discount",
                ),
                "supports_heads": ("tradable", "bad_path", "tail_risk", "path_quality", "position_size"),
            },
        ),
        (
            "trend_ema_encoder",
            {
                "model_role": "trend_alignment_sequence_ai",
                "owned_objectives": (),
                "primary_signal_families": (
                    "EMA stack",
                    "EMA slope",
                    "price-vs-EMA",
                    "trend age",
                    "multi-timeframe agreement",
                ),
                "supports_heads": ("direction", "tf_agreement", "mtf_direction", "forecast"),
            },
        ),
        (
            "vol_compression_encoder",
            {
                "model_role": "volatility_regime_sequence_ai",
                "owned_objectives": ("compression_expansion",),
                "primary_signal_families": (
                    "ATR percentile",
                    "range compression",
                    "squeeze",
                    "compression release",
                    "expansion direction",
                ),
                "supports_heads": ("vol_forecast", "tail_risk", "position_size", "path_quality_log_var"),
            },
        ),
        (
            "momentum_flow_encoder",
            {
                "model_role": "momentum_followthrough_sequence_ai",
                "owned_objectives": (),
                "primary_signal_families": (
                    "recent returns",
                    "impulse direction",
                    "acceleration",
                    "CLV",
                    "volatility-adjusted follow-through",
                ),
                "supports_heads": ("direction", "dip", "forecast", "clean_edge", "mfe_first_n"),
            },
        ),
        (
            "session_regime_encoder",
            {
                "model_role": "session_regime_sequence_ai",
                "owned_objectives": ("session_x_structure",),
                "primary_signal_families": (
                    "Asia/EU/US/overlap",
                    "session boundary",
                    "session age",
                    "spread bucket",
                    "session x structure interactions",
                ),
                "supports_heads": ("tradable", "timing", "bad_path", "survival", "position_size"),
            },
        ),
    ]
)

CHALLENGER_SEQ215_SPECIALIST_MODEL_CONTRACT = OrderedDict(
    [
        *SPECIALIST_MODEL_CONTRACT.items(),
        (
            "chart_geometry_encoder",
            {
                "model_role": "chart_geometry_line_fib_pattern_sequence_ai",
                "owned_objectives": (),
                "primary_signal_families": (
                    "support/resistance line proximity",
                    "trendline/channel break pressure",
                    "Fibonacci retracement/extension zones",
                    "EMA cross pressure",
                    "triangle/flag/compression chart-pattern proxies",
                ),
                "supports_heads": ("direction", "tradable", "timing", "path_quality", "tail_risk"),
            },
        ),
        (
            "price_action_candle_encoder",
            {
                "model_role": "closed_bar_candlestick_pattern_sequence_ai",
                "owned_objectives": (),
                "primary_signal_families": (
                    "single-candle body/wick shape",
                    "doji/indecision",
                    "hammer/shooting-star rejection",
                    "engulfing and two-candle reversal",
                    "inside/outside bar compression and expansion",
                    "three-candle continuation/reversal patterns",
                ),
                "supports_heads": ("direction", "tradable", "timing", "bad_path", "tail_risk"),
            },
        ),
    ]
)

SMART_SEQ520_SPECIALIST_MODEL_CONTRACT = CHALLENGER_SEQ215_SPECIALIST_MODEL_CONTRACT

SMART_SEQ520_SMART_FAMILY_CONTRACT = OrderedDict(
    [
        (
            "trend_ema_smart_layer",
            {
                "expected_feature_count": 20,
                "expected_specialist_counts": {"trend_ema_encoder": 20},
                "owned_specialists": ("trend_ema_encoder",),
                "purpose": "Trend/EMA stack, slope, inflection, exhaustion and MTF trend pressure.",
            },
        ),
        (
            "smc_liquidity_quality_layer",
            {
                "expected_feature_count": 24,
                "expected_specialist_counts": {"smc_liquidity_encoder": 24},
                "owned_specialists": ("smc_liquidity_encoder",),
                "purpose": "Sweep/reclaim, false-breakout and premium/discount quality scoring.",
            },
        ),
        (
            "structure_swing_derivation_layer",
            {
                "expected_feature_count": 28,
                "expected_specialist_counts": {"structure_swing_encoder": 28},
                "owned_specialists": ("structure_swing_encoder",),
                "purpose": "HH/HL/LH/LL consistency, swing-leg quality, BOS/CHoCH and pullback derivations.",
            },
        ),
        (
            "momentum_flow_smart_layer",
            {
                "expected_feature_count": 26,
                "expected_specialist_counts": {"momentum_flow_encoder": 26},
                "owned_specialists": ("momentum_flow_encoder",),
                "purpose": "Vol-adjusted returns, impulse, follow-through and clean-edge momentum pressure.",
            },
        ),
        (
            "session_regime_interaction_layer",
            {
                "expected_feature_count": 68,
                "expected_specialist_counts": {"session_regime_encoder": 68},
                "owned_specialists": ("session_regime_encoder",),
                "purpose": "Session age/boundaries, regime agreement and session x structure/liquidity interactions.",
            },
        ),
        (
            "vol_compression_smart_layer",
            {
                "expected_feature_count": 28,
                "expected_specialist_counts": {"vol_compression_encoder": 28},
                "owned_specialists": ("vol_compression_encoder",),
                "purpose": "ATR percentile, squeeze state, compression-release and volatility forecast confidence.",
            },
        ),
        (
            "chart_geometry_smart2_layer",
            {
                "expected_feature_count": 13,
                "expected_specialist_counts": {"chart_geometry_encoder": 13},
                "owned_specialists": ("chart_geometry_encoder",),
                "purpose": "Smart2 trendline/channel/Fibonacci/EMA-cross/chart-pattern geometry fields.",
            },
        ),
        (
            "price_action_candle_smart3_layer",
            {
                "expected_feature_count": 32,
                "expected_specialist_counts": {"price_action_candle_encoder": 32},
                "owned_specialists": ("price_action_candle_encoder",),
                "purpose": "Smart3 closed-bar candle body/wick/reversal/continuation pattern fields.",
            },
        ),
        (
            "support_resistance_memory_layer",
            {
                "expected_feature_count": 34,
                "expected_specialist_counts": {"smc_liquidity_encoder": 34},
                "owned_specialists": ("smc_liquidity_encoder",),
                "purpose": "Support/resistance level memory, repeated tests, reclaim/break pressure and trap risk.",
            },
        ),
        (
            "mtf_confluence_layer",
            {
                "expected_feature_count": 32,
                "expected_specialist_counts": {
                    "trend_ema_encoder": 6,
                    "structure_swing_encoder": 5,
                    "smc_liquidity_encoder": 6,
                    "chart_geometry_encoder": 4,
                    "session_regime_encoder": 11,
                },
                "owned_specialists": (
                    "trend_ema_encoder",
                    "structure_swing_encoder",
                    "smc_liquidity_encoder",
                    "chart_geometry_encoder",
                    "session_regime_encoder",
                ),
                "purpose": "Cross-family MTF confluence and disagreement features that route back to their mechanism owners.",
            },
        ),
    ]
)

TRAINABLE_SPECIALIST_CONTRACT_MODES = ("foundation_seq146", "challenger_seq215", "smart_seq520_candidate")
SPECIALIST_CONTRACT_MODES = TRAINABLE_SPECIALIST_CONTRACT_MODES
SPECIALIST_AUDIT_CONTRACT_MODES = SPECIALIST_CONTRACT_MODES
SPECIALIST_CONTRACT_TRAINING_ALLOWED = {
    "foundation_seq146": True,
    "challenger_seq215": True,
    "smart_seq520_candidate": True,
}


def required_training_specialists_for_mode(mode: str = "foundation_seq146") -> tuple[str, ...]:
    normalized = str(mode or "foundation_seq146").strip()
    if normalized == "foundation_seq146":
        return REQUIRED_TRAINING_SPECIALISTS
    if normalized == "challenger_seq215":
        return CHALLENGER_SEQ215_TRAINING_SPECIALISTS
    if normalized == "smart_seq520_candidate":
        return SMART_SEQ520_CANDIDATE_SPECIALISTS
    raise ValueError(f"unknown specialist contract mode: {mode}")


def specialist_model_contract_for_mode(mode: str = "foundation_seq146") -> "OrderedDict[str, dict[str, object]]":
    normalized = str(mode or "foundation_seq146").strip()
    if normalized == "foundation_seq146":
        return SPECIALIST_MODEL_CONTRACT
    if normalized == "challenger_seq215":
        return CHALLENGER_SEQ215_SPECIALIST_MODEL_CONTRACT
    if normalized == "smart_seq520_candidate":
        return SMART_SEQ520_SPECIALIST_MODEL_CONTRACT
    raise ValueError(f"unknown specialist contract mode: {mode}")


def specialist_contract_training_allowed_for_mode(mode: str = "foundation_seq146") -> bool:
    normalized = str(mode or "foundation_seq146").strip()
    if normalized not in SPECIALIST_AUDIT_CONTRACT_MODES:
        raise ValueError(f"unknown specialist contract mode: {mode}")
    return bool(SPECIALIST_CONTRACT_TRAINING_ALLOWED.get(normalized, False))


def smart_family_contract_for_mode(mode: str = "foundation_seq146") -> "OrderedDict[str, dict[str, object]]":
    normalized = str(mode or "foundation_seq146").strip()
    if normalized == "smart_seq520_candidate":
        return SMART_SEQ520_SMART_FAMILY_CONTRACT
    if normalized in SPECIALIST_CONTRACT_MODES:
        return OrderedDict()
    raise ValueError(f"unknown specialist contract mode: {mode}")


def _norm(name: str) -> str:
    return str(name or "").strip().lower()


def _contains_any(name: str, tokens: Iterable[str]) -> bool:
    n = _norm(name)
    return any(token in n for token in tokens)


def classify_entry_specialist_feature(name: str) -> str:
    """Return the primary specialist group for one emitted seq/snap feature."""
    n = _norm(name)
    bare = n
    for prefix in ("chart.", "ctx_cont.", "ctx_cat.", "snap.", "seq."):
        if bare.startswith(prefix):
            bare = bare[len(prefix) :]
            break

    if bare in NEUTRAL_BRIDGE_FIELDS:
        return "neutral_bridge_anchor"
    if n.startswith("momentum.flow_") or bare.startswith("momentum.flow_"):
        return "momentum_flow_encoder"
    if n in CONTEXT_FEATURE_SPECIALIST_OVERRIDES:
        return CONTEXT_FEATURE_SPECIALIST_OVERRIDES[n]
    if bare in CONTEXT_FEATURE_SPECIALIST_OVERRIDES:
        return CONTEXT_FEATURE_SPECIALIST_OVERRIDES[bare]

    if n.startswith("candle.pattern_") or bare.startswith("candle.pattern_"):
        return "price_action_candle_encoder"

    if n.startswith("chart.structure_swing_") or bare.startswith("structure_swing_"):
        return "structure_swing_encoder"

    if _contains_any(
        n,
        (
            "chart.geometry_",
            "geometry_",
            "fib_",
            "fibonacci",
            "trendline",
            "channel_",
            "triangle",
            "flag_pullback",
            "ema_cross",
            "line_pattern",
        ),
    ):
        return "chart_geometry_encoder"

    # Session x structure is intentionally owned by the session encoder even
    # when the name also contains BOS/HH/sweep. The point is regime conditioning.
    if _contains_any(
        n,
        (
            "foundation_asia_x_",
            "foundation_eu_x_",
            "foundation_us_x_",
            "foundation_overlap_x_",
            "session_regime.",
            "session_regime_",
            "is_eu_only",
            "is_asia",
            "asia",
            "eu_x_",
            "us_x_",
            "overlap",
            "session",
            "session_id",
            "hour_",
            "dow_",
            "is_us_only",
            "spread",
            "regime",
            "dist_to_boundary",
        ),
    ):
        return "session_regime_encoder"

    if _contains_any(
        n,
        (
            "foundation_sweep",
            "foundation_false_breakout",
            "sweep",
            "liquidity",
            "sr_",
            "support",
            "resistance",
            "premium_discount",
            "premium_state",
            "premium_extreme",
            "level",
            "pivot",
            "dist_to_r",
            "dist_to_s",
            "dist_to_m5_hi",
            "dist_to_m5_lo",
            "dist_to_m15_hi",
            "dist_to_m15_lo",
            "dist_to_h1_hi",
            "dist_to_h1_lo",
            "dist_to_h4_hi",
            "dist_to_h4_lo",
            "dist_to_d1_hi",
            "dist_to_d1_lo",
            "wick_level",
            "major_level",
        ),
    ):
        return "smc_liquidity_encoder"

    if _contains_any(
        n,
        (
            "foundation_hh_state",
            "foundation_hl_state",
            "foundation_lh_state",
            "foundation_ll_state",
            "foundation_structure",
            "foundation_bos",
            "foundation_choch",
            "foundation_bars_since_structure_break",
            "foundation_impulse",
            "foundation_pullback",
            "retracement_from_last_impulse",
            "smc_swing",
            "smc_bos",
            "smc_choch",
            "bos",
            "choch",
            "swing",
            "pullback",
            "struct_",
            "near_recent_swing",
            "lh_x_",
            "hh_x_",
            "hl_x_",
            "ll_x_",
        ),
    ):
        return "structure_swing_encoder"

    if _contains_any(
        n,
        (
            "foundation_compression",
            "foundation_expansion",
            "atr",
            "vol",
            "rvol",
            "range_compression",
            "range_z",
            "squeeze",
            "bb_bandwidth",
            "sigma",
            "kurt",
        ),
    ):
        return "vol_compression_encoder"

    if _contains_any(
        n,
        (
            "ema",
            "kama",
            "tema",
            "trend",
            "slope",
            "pos_vs_ema",
            "price_vs_ema",
            "tf_agreement",
            "rsi",
            "pct_change",
            "dist_roc",
        ),
    ):
        return "trend_ema_encoder"

    if _contains_any(
        n,
        (
            "ret_",
            "mom",
            "momentum",
            "dip_confirmed",
            "dip_proximity",
            "acceleration",
            "clv",
            "signed_vol",
            "followthrough",
        ),
    ):
        return "momentum_flow_encoder"

    if _contains_any(n, ("body", "wick", "range", "candle")):
        return "price_action_candle_encoder"

    return "unmapped"


def group_features_by_specialist(features: Iterable[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {name: [] for name in SPECIALIST_GROUPS}
    grouped["unmapped"] = []
    for feature in features:
        group = classify_entry_specialist_feature(str(feature))
        grouped.setdefault(group, []).append(str(feature))
    return grouped
