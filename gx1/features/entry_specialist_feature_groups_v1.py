"""Exact model-native seq513 feature-to-specialist contract.

Every seq/snap field is owned by one of eight trainable evidence specialists.
The retired seven-field XGBoost bridge is detected only to fail closed; it is
never an encoder group, model input, prior, or compatibility route.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from gx1.contracts.entry_model_native_signal_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_FIELDS,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.features.entry_model_native_feature_layers_v1 import (
    MODEL_NATIVE_MANDATORY_FAMILY_FEATURES,
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
)


SPECIALIST_GROUPS: "OrderedDict[str, dict[str, str]]" = OrderedDict(
    [
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
                "role": "Numeric trendlines, support/resistance channels, Fibonacci zones, EMA crosses and chart patterns.",
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

MODEL_NATIVE_TRAINING_SPECIALISTS = (
    "structure_swing_encoder",
    "smc_liquidity_encoder",
    "trend_ema_encoder",
    "vol_compression_encoder",
    "momentum_flow_encoder",
    "session_regime_encoder",
    "chart_geometry_encoder",
    "price_action_candle_encoder",
)

MODEL_NATIVE_EXPECTED_SIGNAL_DIM = MODEL_NATIVE_SIGNAL_DIM
MODEL_NATIVE_EXPECTED_SELECTED_FEATURE_COUNT = MODEL_NATIVE_SELECTED_FEATURE_COUNT
MODEL_NATIVE_EXPECTED_SPECIALIST_FEATURE_COUNT = (
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT
)

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
SPECIALIST_SHARED_REACHABLE_HEADS = tuple(
    head for head in SPECIALIST_FUSION_ACTIVE_HEADS if head != "mtf_direction"
) + (
    "trade_side_hierarchy",
    "trendline_rail",
    "side_validity",
    "offline_rl_action_value",
    "offline_rl_expectile_value",
    "model_native_evidence_fusion",
)

FORBIDDEN_LEGACY_BRIDGE_SPECIALIST = "forbidden_legacy_bridge"
_FORBIDDEN_LEGACY_BRIDGE_FIELDS = frozenset(FORBIDDEN_LEGACY_BRIDGE_FIELDS)

CONTEXT_FEATURE_SPECIALIST_OVERRIDES = {
    # Distance from the D1 EMA200 is trend alignment evidence.  The ``_atr``
    # suffix denotes normalization only and must not transfer semantic
    # ownership to the volatility specialist.
    "ctx_cont.d1_dist_from_ema200_atr": "trend_ema_encoder",
    "d1_dist_from_ema200_atr": "trend_ema_encoder",
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

MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT = OrderedDict(
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
                "supports_heads": SPECIALIST_SHARED_REACHABLE_HEADS,
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
                "supports_heads": SPECIALIST_SHARED_REACHABLE_HEADS,
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
                "supports_heads": SPECIALIST_SHARED_REACHABLE_HEADS,
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
                "supports_heads": SPECIALIST_SHARED_REACHABLE_HEADS,
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
                "supports_heads": SPECIALIST_SHARED_REACHABLE_HEADS,
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
                "supports_heads": SPECIALIST_SHARED_REACHABLE_HEADS,
            },
        ),
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
                "supports_heads": SPECIALIST_SHARED_REACHABLE_HEADS,
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
                "supports_heads": SPECIALIST_SHARED_REACHABLE_HEADS,
            },
        ),
    ]
)

MODEL_NATIVE_SMART_FAMILY_CONTRACT = OrderedDict(
    [
        (
            "foundation_cross_family_layer",
            {
                "expected_feature_count": 57,
                "expected_specialist_counts": {
                    "structure_swing_encoder": 19,
                    "smc_liquidity_encoder": 5,
                    "vol_compression_encoder": 5,
                    "session_regime_encoder": 28,
                },
                "owned_specialists": (
                    "structure_swing_encoder",
                    "smc_liquidity_encoder",
                    "vol_compression_encoder",
                    "session_regime_encoder",
                ),
                "purpose": (
                    "Primitive HH/HL/LH/LL, BOS/CHoCH age, sweep/reclaim, "
                    "compression-release, impulse/pullback and explicit "
                    "session-by-structure evidence retained alongside its "
                    "higher-order specialist derivations."
                ),
            },
        ),
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
                "expected_feature_count": 18,
                "expected_specialist_counts": {"chart_geometry_encoder": 18},
                "owned_specialists": ("chart_geometry_encoder",),
                "purpose": "Mandatory structural-label and pretrain-polarity geometry inputs plus curated trendline/channel/Fibonacci rail-trap evidence.",
            },
        ),
        (
            "price_action_candle_smart3_layer",
            {
                "expected_feature_count": 32,
                "expected_specialist_counts": {"price_action_candle_encoder": 32},
                "owned_specialists": ("price_action_candle_encoder",),
                "purpose": "Closed-bar candle body/wick/reversal/continuation pattern fields.",
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
        (
            "price_ema50_200_layer",
            {
                "expected_feature_count": 11,
                "expected_specialist_counts": {"trend_ema_encoder": 11},
                "owned_specialists": ("trend_ema_encoder",),
                "purpose": "Exact M5 EMA50/200 state, crosses, slopes, acceleration and price location.",
            },
        ),
    ]
)

if tuple(SPECIALIST_GROUPS) != MODEL_NATIVE_TRAINING_SPECIALISTS:
    raise RuntimeError("MODEL_NATIVE_SPECIALIST_GROUP_ORDER_MISMATCH")
if tuple(MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT) != MODEL_NATIVE_TRAINING_SPECIALISTS:
    raise RuntimeError("MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT_ORDER_MISMATCH")
if sum(
    int(spec["expected_feature_count"])
    for spec in MODEL_NATIVE_SMART_FAMILY_CONTRACT.values()
) != MODEL_NATIVE_EXPECTED_SPECIALIST_FEATURE_COUNT:
    raise RuntimeError("MODEL_NATIVE_SPECIALIST_FEATURE_COUNT_MISMATCH")
if tuple(MODEL_NATIVE_SMART_FAMILY_CONTRACT) != tuple(
    family for family, _features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
):
    raise RuntimeError("MODEL_NATIVE_SPECIALIST_FAMILY_REGISTRY_ORDER_MISMATCH")
for _family, _features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES:
    if int(MODEL_NATIVE_SMART_FAMILY_CONTRACT[_family]["expected_feature_count"]) != len(
        _features
    ):
        raise RuntimeError(
            f"MODEL_NATIVE_SPECIALIST_FAMILY_COUNT_MISMATCH: {_family}"
        )

SPECIALIST_CONTRACT_MODES = (MODEL_NATIVE_CONTRACT_MODE,)


def require_model_native_specialist_contract_mode(mode: object) -> str:
    """Return the exact active mode or reject historical/blank aliases."""

    if not isinstance(mode, str) or mode != MODEL_NATIVE_CONTRACT_MODE:
        raise ValueError(
            "model-native specialist contract mode required: "
            f"observed={mode!r} expected={MODEL_NATIVE_CONTRACT_MODE!r}"
        )
    return mode


def required_training_specialists_for_mode(mode: str) -> tuple[str, ...]:
    require_model_native_specialist_contract_mode(mode)
    return MODEL_NATIVE_TRAINING_SPECIALISTS


def specialist_model_contract_for_mode(
    mode: str,
) -> "OrderedDict[str, dict[str, object]]":
    require_model_native_specialist_contract_mode(mode)
    return MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT


def specialist_contract_training_allowed_for_mode(mode: str) -> bool:
    require_model_native_specialist_contract_mode(mode)
    return True


def smart_family_contract_for_mode(
    mode: str,
) -> "OrderedDict[str, dict[str, object]]":
    require_model_native_specialist_contract_mode(mode)
    return MODEL_NATIVE_SMART_FAMILY_CONTRACT


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

    if bare in _FORBIDDEN_LEGACY_BRIDGE_FIELDS:
        return FORBIDDEN_LEGACY_BRIDGE_SPECIALIST
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
    if bare.startswith(
        ("m5_ema50_200_", "m5_price_vs_ema", "m5_ema50_", "m5_ema200_")
    ):
        return "trend_ema_encoder"

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
    grouped[FORBIDDEN_LEGACY_BRIDGE_SPECIALIST] = []
    grouped["unmapped"] = []
    for feature in features:
        group = classify_entry_specialist_feature(str(feature))
        grouped.setdefault(group, []).append(str(feature))
    return grouped


for _family, _features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES:
    _observed_specialist_counts: dict[str, int] = {}
    for _feature in _features:
        _specialist = classify_entry_specialist_feature(_feature)
        _observed_specialist_counts[_specialist] = (
            _observed_specialist_counts.get(_specialist, 0) + 1
        )
    _expected_specialist_counts = {
        str(name): int(count)
        for name, count in (
            MODEL_NATIVE_SMART_FAMILY_CONTRACT[_family].get(
                "expected_specialist_counts"
            )
            or {}
        ).items()
    }
    if _observed_specialist_counts != _expected_specialist_counts:
        raise RuntimeError(
            "MODEL_NATIVE_MANDATORY_FAMILY_SPECIALIST_ROUTING_MISMATCH: "
            f"family={_family} observed={_observed_specialist_counts} "
            f"expected={_expected_specialist_counts}"
        )
