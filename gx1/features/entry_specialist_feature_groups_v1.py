"""Exact model-native seq513 feature-to-specialist contract.

Every seq/snap field is owned by one of eight trainable evidence specialists.
The retired seven-field external bridge is detected only to fail closed; it is
never an encoder group, model input, prior, or compatibility route.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Iterable, Mapping

from gx1.contracts.entry_model_native_signal_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_FIELDS,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CAT_DOMAINS,
    MODEL_NATIVE_CTX_CAT_FIELDS_SHA256,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS_SHA256,
    RETIRED_MODEL_NATIVE_SIGNAL_FIELDS,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.features.entry_model_native_feature_layers_v1 import (
    CANDLE_PRIMITIVE_MANDATORY_FEATURE_NAMES,
    FOUNDATION_STRUCTURE_FEATURE_NAMES,
    LEVEL_REGISTRY_M5_LAYER_FEATURE_NAMES,
    MODEL_NATIVE_MANDATORY_FAMILY_FEATURES,
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
    MOMENTUM_EVENT_M5_LAYER_FEATURE_NAMES,
    PRICE_DERIVED_FEATURE_NAMES,
    RAW_MTF_TREND_LAYER_FEATURE_NAMES,
    SMC_LOCAL_EVENT_LAYER_FEATURE_NAMES,
    SWING_EVENT_LAYER_FEATURE_NAMES,
    TRENDLINE_REGISTRY_M5_LAYER_FEATURE_NAMES,
    VOLATILITY_SQUEEZE_LOCAL_LAYER_FEATURE_NAMES,
)
from gx1.features.micro_structure_v1 import MICRO_FEATURE_NAMES_V1
from gx1.features.volume_features import VOLUME_FEATURE_NAMES


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
                "role": "Identified trendlines, channels and level-registry geometry.",
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

MODEL_NATIVE_TRAINING_SPECIALISTS = tuple(SPECIALIST_GROUPS)

# Exact all-eight-family routing for the per-resolution V4 surface.  This is a
# second use of the same eight semantic owners, not a parallel specialist
# taxonomy.  The HTF feature contract owns ordered emission; this owner binds
# every emitted name to exactly one specialist and rejects older 6/8 surfaces
# when explicit family×timeframe routing is requested.
from gx1.features.htf_features import (  # noqa: E402
    LOCAL_MOMENTUM_V30_PRIMITIVE_FEATURES,
    MULTI_TF_PER_BAR_FEATURES_V4,
    MULTI_TF_V4_CANDLE_PRIMITIVE_FEATURES,
    MULTI_TF_V4_LEVEL_REGISTRY_FEATURES,
    MULTI_TF_V4_MOMENTUM_EVENT_FEATURES,
    MULTI_TF_V4_SWING_FEATURES,
    MULTI_TF_V4_TREND_EVENT_FEATURES,
    MULTI_TF_V4_TRENDLINE_REGISTRY_FEATURES,
    MULTI_TF_V4_VOLUME_FEATURES,
    MULTI_TF_V4_VOLATILITY_SQUEEZE_FEATURES,
)
from gx1.features.smc_v1 import (  # noqa: E402
    SMC_MTF_FEATURE_NAMES_V1,
    SMC_MTF_GEOMETRY_FEATURE_NAMES_V1,
)

MULTI_TF_SPECIALIST_ROUTING_SCHEMA_VERSION = (
    "entry_multi_tf_eight_family_specialist_routing_v8"
)
MULTI_TF_SPECIALIST_FEATURE_GROUPS_V4 = OrderedDict(
    [
        (
            "structure_swing_encoder",
            tuple(MULTI_TF_V4_SWING_FEATURES),
        ),
        (
            "smc_liquidity_encoder",
            tuple(SMC_MTF_FEATURE_NAMES_V1)
            # V29 Phase A: the per-TF immutable pivot-anchor registry block
            # (design doc §1.4 routing: level identity/touch/break/retest is
            # liquidity-level evidence).
            + tuple(MULTI_TF_V4_LEVEL_REGISTRY_FEATURES),
        ),
        (
            "trend_ema_encoder",
            (
                "ema20_dist_atr",
                "ema50_dist_atr",
                "ema100_dist_atr",
                "ema200_dist_atr",
                "ema20_slope_atr",
                "ema50_slope_atr",
                "ema200_slope_atr",
                "ema_stack_aligned_v2",
                "adx14",
                # V30 (2026-08-13): signed DI spread from the same _adx14
                # producer as raw adx14 — trend-direction evidence.
                "di_spread_signed",
                "trend_state_age_bars",
            )
            # V29 Phase A: EMA50/200 spread/state/cross events + price-vs-EMA
            # cross events and ages (trend_ema GAP-1/2/3).
            + tuple(MULTI_TF_V4_TREND_EVENT_FEATURES),
        ),
        (
            "vol_compression_encoder",
            (
                "atr_bps_14",
                "bb_width_atr",
            )
            + tuple(MULTI_TF_V4_VOLATILITY_SQUEEZE_FEATURES),
        ),
        (
            "momentum_flow_encoder",
            (
                "rsi14_centered",
                # V30 (2026-08-13): raw Wilder RSI 5-bar velocity — momentum
                # evidence beside its rsi14_centered sibling.
                "rsi14_delta_5",
                "mom_5_atr",
                "mom_20_atr",
                "bb_position",
            )
            # V29 Phase A: RSI threshold/divergence and mom20 sign-flip
            # events (momentum_flow G1/G2).
            + tuple(MULTI_TF_V4_MOMENTUM_EVENT_FEATURES)
            # Raw tick-count activity on each TF's own closed OHLCV bars.
            + tuple(MULTI_TF_V4_VOLUME_FEATURES),
        ),
        (
            "session_regime_encoder",
            (
                "vwap_local_cycle_dist_atr",
                "vwap20_dist_atr",
                "vwap96_dist_atr",
                "vwap_local_cycle_slope_atr",
            ),
        ),
        (
            "chart_geometry_encoder",
            tuple(SMC_MTF_GEOMETRY_FEATURE_NAMES_V1)
            # V29 Phase A: the per-TF trendline/channel registry block
            # (design doc §2: sloped-line evidence is chart geometry).
            + tuple(MULTI_TF_V4_TRENDLINE_REGISTRY_FEATURES),
        ),
        (
            "price_action_candle_encoder",
            (
                "close_open_atr",
                "body_pct",
                *MULTI_TF_V4_CANDLE_PRIMITIVE_FEATURES,
            ),
        ),
    ]
)


def require_multi_tf_specialist_routing_v4(
    feature_names: Iterable[str],
) -> "OrderedDict[str, tuple[int, ...]]":
    """Bind the exact V4 field order to all eight non-empty specialists."""
    ordered = tuple(str(name) for name in feature_names)
    if ordered != tuple(MULTI_TF_PER_BAR_FEATURES_V4):
        raise RuntimeError(
            "ENTRY_MULTI_TF_SPECIALIST_FEATURE_CONTRACT_INVALID"
        )
    if tuple(MULTI_TF_SPECIALIST_FEATURE_GROUPS_V4) != (
        MODEL_NATIVE_TRAINING_SPECIALISTS
    ):
        raise RuntimeError("ENTRY_MULTI_TF_SPECIALIST_ORDER_INVALID")
    flattened = tuple(
        name
        for names in MULTI_TF_SPECIALIST_FEATURE_GROUPS_V4.values()
        for name in names
    )
    if (
        len(flattened) != len(set(flattened))
        or set(flattened) != set(ordered)
        or any(not names for names in MULTI_TF_SPECIALIST_FEATURE_GROUPS_V4.values())
    ):
        raise RuntimeError("ENTRY_MULTI_TF_SPECIALIST_COVERAGE_INVALID")
    index = {name: position for position, name in enumerate(ordered)}
    return OrderedDict(
        (
            specialist,
            tuple(index[name] for name in names),
        )
        for specialist, names in MULTI_TF_SPECIALIST_FEATURE_GROUPS_V4.items()
    )


# Import-time proof: a contract edit cannot silently orphan, duplicate or
# reorder one family×timeframe input.
MULTI_TF_SPECIALIST_INDICES_V4 = require_multi_tf_specialist_routing_v4(
    MULTI_TF_PER_BAR_FEATURES_V4
)

MODEL_NATIVE_EXPECTED_SIGNAL_DIM = MODEL_NATIVE_SIGNAL_DIM
MODEL_NATIVE_EXPECTED_SELECTED_FEATURE_COUNT = MODEL_NATIVE_SELECTED_FEATURE_COUNT
MODEL_NATIVE_EXPECTED_SPECIALIST_FEATURE_COUNT = (
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT
)
MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_SCHEMA_VERSION = (
    "entry_model_native_context_specialist_routing_v7"
)
MODEL_NATIVE_CONTEXT_TEMPORAL_ALIAS_POLICY_SCHEMA_VERSION = (
    "entry_model_native_context_temporal_alias_policy_v1"
)
MODEL_NATIVE_NOMINAL_CTX_CONT_FIELDS: tuple[str, ...] = ()
MODEL_NATIVE_NOMINAL_CTX_CONT_CARDINALITY = 0
SPECIALIST_FUSION_ACTIVE_HEADS = (
    "direction",
    "tradable",
    "path_quality",
    "mfe_first_n",
    "bad_path",
    "clean_edge",
    "survival",
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
    # Raw one-observation change in EMA200 distance is directional momentum;
    # ATR only supplies the unit and must not route it to volatility.
    "ctx_cont.d1_dist_change_1bar_atr_v4": "momentum_flow_encoder",
    "d1_dist_change_1bar_atr_v4": "momentum_flow_encoder",
    "ctx_cont.m15_ema5_20_spread_atr_canon_v2": "trend_ema_encoder",
    "m15_ema5_20_spread_atr_canon_v2": "trend_ema_encoder",
    "ctx_cont.h4_mid_ema50_dist_atr_canon_v2": "trend_ema_encoder",
    "h4_mid_ema50_dist_atr_canon_v2": "trend_ema_encoder",
    # D1 close location inside the trailing 20-day range is range-location
    # structure evidence: the chart-geometry layer declares it as a source
    # field and consumes it as its D1 range-location input.  The trailing
    # ``range`` token names the structure, not candle body/wick shape, and
    # must not fall through to the price-action owner.
    "ctx_cont.d1_close_pct_in_20day_range_canon_v2": "chart_geometry_encoder",
    "d1_close_pct_in_20day_range_canon_v2": "chart_geometry_encoder",
    "ctx_cont.spread_bps": "session_regime_encoder",
    "spread_bps": "session_regime_encoder",
    # V30 package 4 (2026-08-13): the quote/spread-dynamics block joins its
    # level sibling ``spread_bps`` in the execution/session-regime specialist.
    # Two of the three names would reach that encoder through the broad
    # ``spread`` lexical rule anyway; ``quote_range_asymmetry_bps`` would not
    # (``range`` is claimed earlier by the volatility owner), so all three are
    # routed EXPLICITLY here rather than depending on keyword precedence.
    # Semantics: how expensive and how disorderly the quote is at the decision
    # bar — abstention/execution-regime evidence, never direction evidence.
    "ctx_cont.spread_bps_delta_1": "session_regime_encoder",
    "spread_bps_delta_1": "session_regime_encoder",
    "ctx_cont.spread_intrabar_range_bps": "session_regime_encoder",
    "spread_intrabar_range_bps": "session_regime_encoder",
    "ctx_cont.quote_range_asymmetry_bps": "session_regime_encoder",
    "quote_range_asymmetry_bps": "session_regime_encoder",
    # V30 package 8A (2026-08-13): the two swing-structure "level intact"
    # flags (swing_structure_v1.SWING_V29_ADDITION_NAMES_V1) name the state of
    # the last CONFIRMED SWING PIVOT, i.e. structure, not a horizontal S/R
    # level.  The substring ``level`` is claimed earlier by the liquidity
    # owner, so — exactly like the quote/spread block above — they are routed
    # EXPLICITLY here instead of depending on keyword precedence.
    "swing_high_level_intact": "structure_swing_encoder",
    "swing_low_level_intact": "structure_swing_encoder",
    "ctx_cont.swing_high_level_intact": "structure_swing_encoder",
    "ctx_cont.swing_low_level_intact": "structure_swing_encoder",
    "ctx_cat.session_id": "session_regime_encoder",
    "session_id": "session_regime_encoder",
}

FOUNDATION_REQUIREMENT_PATTERNS = OrderedDict(
    [
        (
            "bos_choch_age",
            {
                "expected_specialist": "structure_swing_encoder",
                "tokens": ("foundation_bos_", "foundation_choch_", "foundation_bars_since_structure_break"),
            },
        ),
    ]
)

FOUNDATION_OBJECTIVE_SPECIALISTS = OrderedDict(
    [
        ("bos_choch_age", "structure_swing_encoder"),
    ]
)

MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT = OrderedDict(
    [
        (
            "structure_swing_encoder",
            {
                "model_role": "market_structure_sequence_ai",
                "owned_objectives": ("bos_choch_age",),
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
                "owned_objectives": (),
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
                "owned_objectives": (),
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
                "owned_objectives": (),
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
                "model_role": "identified_chart_object_sequence_ai",
                "owned_objectives": (),
                "primary_signal_families": (
                    "identified support/resistance levels",
                    "persistent trendline state",
                    "line and level touch history",
                    "break and retest events",
                    "channel geometry",
                ),
                "supports_heads": SPECIALIST_SHARED_REACHABLE_HEADS,
            },
        ),
        (
            "price_action_candle_encoder",
            {
                "model_role": "raw_closed_bar_geometry_sequence_ai",
                "owned_objectives": (),
                "primary_signal_families": (
                    "signed body geometry",
                    "upper/lower wick shares",
                    "close location and zero-range identity",
                    "open/close change versus previous bar",
                    "high/low/range/body change versus previous bar",
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
                "expected_feature_count": len(FOUNDATION_STRUCTURE_FEATURE_NAMES),
                "expected_specialist_counts": {
                    "structure_swing_encoder": len(
                        FOUNDATION_STRUCTURE_FEATURE_NAMES
                    ),
                },
                "owned_specialists": ("structure_swing_encoder",),
                "purpose": (
                    "Exact stateful local BOS-up, BOS-down and CHOCH ages; "
                    "all former foundation scorebooks are retired."
                ),
            },
        ),
        (
            "raw_mtf_trend_layer",
            {
                "expected_feature_count": len(RAW_MTF_TREND_LAYER_FEATURE_NAMES),
                "expected_specialist_counts": {
                    "trend_ema_encoder": len(RAW_MTF_TREND_LAYER_FEATURE_NAMES)
                },
                "owned_specialists": ("trend_ema_encoder",),
                "purpose": (
                    "Raw M15/H1/H4/D1 trend direction evidence; independently "
                    "computed per closed timeframe with no hand-weighted score."
                ),
            },
        ),
        (
            "price_action_candle_raw_layer",
            {
                "expected_feature_count": len(
                    CANDLE_PRIMITIVE_MANDATORY_FEATURE_NAMES
                ),
                "expected_specialist_counts": {
                    "price_action_candle_encoder": len(
                        CANDLE_PRIMITIVE_MANDATORY_FEATURE_NAMES
                    ),
                },
                "owned_specialists": ("price_action_candle_encoder",),
                "purpose": (
                    "Raw closed-bar body/wick/location and one-bar relative "
                    "geometry; the temporal model learns named patterns."
                ),
            },
        ),
        (
            "price_ema50_200_layer",
            {
                # V30 (2026-08-13): counts DERIVED from the owner tuple (the
                # layer gained chart.local_kama_efficiency_30, then the GAP-2/3
                # age fields and the four price-vs-EMA cross events; prefer
                # derive-from-owner over restated literals).
                "expected_feature_count": len(PRICE_DERIVED_FEATURE_NAMES),
                "expected_specialist_counts": {
                    "trend_ema_encoder": len(PRICE_DERIVED_FEATURE_NAMES),
                },
                "owned_specialists": ("trend_ema_encoder",),
                "purpose": (
                    "Exact local-resolution EMA50/200 state, crosses, "
                    "price-vs-EMA crosses, side/cross durations, slopes, "
                    "acceleration, price location and the window-30 Kaufman "
                    "efficiency ratio; M5 for Entry and M1 for Exit."
                ),
            },
        ),
        # ── V29 Phase A stage 2 event families (counts DERIVED from the
        # declared owner tuples; docs/V29_EVENT_SURFACE_DESIGN_20260811.md) ──
        (
            "level_registry_m5_layer",
            {
                "expected_feature_count": len(
                    LEVEL_REGISTRY_M5_LAYER_FEATURE_NAMES
                ),
                "expected_specialist_counts": {
                    "smc_liquidity_encoder": len(
                        LEVEL_REGISTRY_M5_LAYER_FEATURE_NAMES
                    ),
                },
                "owned_specialists": ("smc_liquidity_encoder",),
                "purpose": (
                    "Persistent immutable-pivot level identity: nearest "
                    "above/below slots, touch/reaction memory, break and "
                    "signed retest events."
                ),
            },
        ),
        (
            "trendline_registry_m5_layer",
            {
                "expected_feature_count": len(
                    TRENDLINE_REGISTRY_M5_LAYER_FEATURE_NAMES
                ),
                "expected_specialist_counts": {
                    "chart_geometry_encoder": len(
                        TRENDLINE_REGISTRY_M5_LAYER_FEATURE_NAMES
                    ),
                },
                "owned_specialists": ("chart_geometry_encoder",),
                "purpose": (
                    "Real two-point-anchored trendline/channel registry on the "
                    "entry M5 clock: slot projections, touch/break/retest "
                    "events and channel/triangle state."
                ),
            },
        ),
        (
            "swing_structure_event_layer",
            {
                "expected_feature_count": len(SWING_EVENT_LAYER_FEATURE_NAMES),
                "expected_specialist_counts": {
                    "structure_swing_encoder": len(
                        SWING_EVENT_LAYER_FEATURE_NAMES
                    ),
                },
                "owned_specialists": ("structure_swing_encoder",),
                "purpose": (
                    "Confirmed-swing break events with displacement, break "
                    "ages and pivot-sequence deltas/run counts (G1/G2/G4)."
                ),
            },
        ),
        (
            "momentum_event_m5_layer",
            {
                "expected_feature_count": len(
                    MOMENTUM_EVENT_M5_LAYER_FEATURE_NAMES
                ),
                "expected_specialist_counts": {
                    "momentum_flow_encoder": len(
                        MOMENTUM_EVENT_M5_LAYER_FEATURE_NAMES
                    ),
                },
                "owned_specialists": ("momentum_flow_encoder",),
                "purpose": (
                    "RSI threshold crosses, extreme age, mom20 sign flips and "
                    "confirmed-pivot RSI divergence plus continuous momentum "
                    "on the native M5 Entry or M1 Exit clock (G1/G2)."
                ),
            },
        ),
        (
            "smc_local_event_layer",
            {
                "expected_feature_count": len(
                    SMC_LOCAL_EVENT_LAYER_FEATURE_NAMES
                ),
                "expected_specialist_counts": {
                    "structure_swing_encoder": 1,
                    "smc_liquidity_encoder": (
                        len(SMC_LOCAL_EVENT_LAYER_FEATURE_NAMES) - 1
                    ),
                },
                "owned_specialists": (
                    "structure_swing_encoder",
                    "smc_liquidity_encoder",
                ),
                "purpose": (
                    "Native M5/M1 BOS displacement, sided sweep depth, "
                    "and level-identity sweep events; raw sweep age/seen "
                    "live in the canonical local SMC owner."
                ),
            },
        ),
        (
            "volatility_squeeze_local_layer",
            {
                "expected_feature_count": len(
                    VOLATILITY_SQUEEZE_LOCAL_LAYER_FEATURE_NAMES
                ),
                "expected_specialist_counts": {
                    "vol_compression_encoder": len(
                        VOLATILITY_SQUEEZE_LOCAL_LAYER_FEATURE_NAMES
                    ),
                },
                "owned_specialists": ("vol_compression_encoder",),
                "purpose": (
                    "TRAIN-fitted native-clock squeeze state, raw duration, "
                    "one-shot release edge and release memory."
                ),
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


def _norm(name: str) -> str:
    return str(name or "").strip().lower()


def _contains_any(name: str, tokens: Iterable[str]) -> bool:
    n = _norm(name)
    return any(token in n for token in tokens)


_PRICE_DERIVED_TREND_FIELDS = frozenset(
    _norm(field) for field in PRICE_DERIVED_FEATURE_NAMES
)
_BASIC_V1_TREND_FIELDS = frozenset(
    {
        "_v1_ema3_ema6_spread_frac",
        "_v1_kama30_change_5_atr",
        "_v1_tema20_change_3_atr",
    }
)

# The tick-volume family is declared PARTICIPATION evidence by its owner
# (gx1.features.volume_features): surge detection, fast-vs-slow activity,
# percentile rank of activity and signed participation are order-flow
# quantities, not unsigned volatility magnitudes.  Routing them through the
# lexical "vol" matcher handed them to the volatility specialist; the exact
# declared field set belongs to the momentum/flow owner.
_VOLUME_PARTICIPATION_FIELDS = frozenset(
    _norm(field) for field in VOLUME_FEATURE_NAMES
)
_RETIRED_MODEL_NATIVE_SIGNAL_FIELDS = frozenset(
    _norm(field) for field in RETIRED_MODEL_NATIVE_SIGNAL_FIELDS
)
_LOCAL_NATIVE_MOMENTUM_FIELDS = frozenset(
    _norm(field) for field in LOCAL_MOMENTUM_V30_PRIMITIVE_FEATURES
)
_LOCAL_MICRO_STRUCTURE_FIELDS = frozenset(
    _norm(field) for field in MICRO_FEATURE_NAMES_V1
)


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
    if bare in _RETIRED_MODEL_NATIVE_SIGNAL_FIELDS:
        return FORBIDDEN_LEGACY_BRIDGE_SPECIALIST
    if n.startswith("momentum.flow_") or bare.startswith("momentum.flow_"):
        return "momentum_flow_encoder"
    if n in CONTEXT_FEATURE_SPECIALIST_OVERRIDES:
        return CONTEXT_FEATURE_SPECIALIST_OVERRIDES[n]
    if bare in CONTEXT_FEATURE_SPECIALIST_OVERRIDES:
        return CONTEXT_FEATURE_SPECIALIST_OVERRIDES[bare]

    if n.startswith("candle.raw_") or bare.startswith("candle.raw_"):
        return "price_action_candle_encoder"

    # These eleven fields are one local-resolution EMA formula family.  Route
    # the exact emitted fields before broad lexical rules: four names contain
    # ``spread`` as an EMA spread, not execution/session spread evidence.
    if n in _PRICE_DERIVED_TREND_FIELDS:
        return "trend_ema_encoder"
    # Exact basic-v1 trend formulas. Their fidelity names expose either the
    # fractional spread or ATR normalization; broad lexical routing would
    # otherwise mistake ``spread`` for execution context and ``atr`` for a
    # volatility target even though ATR is only the unit denominator.
    if bare in _BASIC_V1_TREND_FIELDS:
        return "trend_ema_encoder"
    # The five local-price primitives form one exact micro/chart-geometry
    # block. Route their declared names before lexical EMA/range/momentum
    # matchers so one formula family cannot be split across specialists merely
    # because its honest field names expose their units.
    if bare in _LOCAL_MICRO_STRUCTURE_FIELDS:
        return "chart_geometry_encoder"

    if n.startswith("chart.structure_swing_") or bare.startswith("structure_swing_"):
        return "structure_swing_encoder"

    if _contains_any(
        n,
        (
            "geometry_",
            "fib_",
            "fibonacci",
            "trendline",
            "channel_",
            "triangle",
            "flag_pullback",
            "ema_cross",
            "line_pattern",
            # V29 trendline/channel registry emissions (design doc §2): the
            # The local M5 lane carries them as chart.geomline_*/chart.geomchan_*.
            "geomline_",
            "geomchan_",
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
            "is_asia",
            "asia",
            "eu_x_",
            "us_x_",
            "overlap",
            "session",
            "session_id",
            "hour_",
            "dow_",
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
            # V29 structure_swing G4 run counts (swing_structure_v1
            # SWING_V29_ADDITION_NAMES_V1): pivot-sequence structure evidence.
            # V30 package 8A completed the set with the two MISSING counters.
            "consecutive_higher_lows",
            "consecutive_lower_highs",
            "consecutive_higher_highs",
            "consecutive_lower_lows",
        ),
    ):
        return "structure_swing_encoder"

    # The three raw tick-volume activity primitives are flow evidence. Keep
    # this exact owner set before the generic "vol" matcher; the retired
    # signed-volume composite is rejected above rather than routed.
    if bare in _VOLUME_PARTICIPATION_FIELDS:
        return "momentum_flow_encoder"
    if bare in _LOCAL_NATIVE_MOMENTUM_FIELDS:
        return "momentum_flow_encoder"
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
            "bandwidth",
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
            "followthrough",
            "rsi",
            "pct_change",
            "dist_roc",
            # V29 momentum_flow G1 divergence events (bear/bull divergence and
            # age).
            "divergence",
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


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def model_native_context_specialist_routing_contract() -> dict[str, object]:
    """Return the exact one-owner routing for every declared context field."""

    ctx_cont_indices = {name: [] for name in MODEL_NATIVE_TRAINING_SPECIALISTS}
    ctx_cat_indices = {name: [] for name in MODEL_NATIVE_TRAINING_SPECIALISTS}
    ctx_cont_names = {name: [] for name in MODEL_NATIVE_TRAINING_SPECIALISTS}
    ctx_cat_names = {name: [] for name in MODEL_NATIVE_TRAINING_SPECIALISTS}
    for index, field in enumerate(MODEL_NATIVE_CTX_CONT_FIELDS):
        owner = classify_entry_specialist_feature(f"ctx_cont.{field}")
        if owner not in ctx_cont_indices:
            raise RuntimeError(
                "MODEL_NATIVE_CONTEXT_CONT_SPECIALIST_OWNER_INVALID: "
                f"field={field} owner={owner}"
            )
        ctx_cont_indices[owner].append(index)
        ctx_cont_names[owner].append(field)
    for index, field in enumerate(MODEL_NATIVE_CTX_CAT_FIELDS):
        owner = classify_entry_specialist_feature(f"ctx_cat.{field}")
        if owner not in ctx_cat_indices:
            raise RuntimeError(
                "MODEL_NATIVE_CONTEXT_CAT_SPECIALIST_OWNER_INVALID: "
                f"field={field} owner={owner}"
            )
        ctx_cat_indices[owner].append(index)
        ctx_cat_names[owner].append(field)

    nominal_set = set(MODEL_NATIVE_NOMINAL_CTX_CONT_FIELDS)
    ctx_cont_numeric_indices = {
        name: [
            index
            for index in indices
            if MODEL_NATIVE_CTX_CONT_FIELDS[index] not in nominal_set
        ]
        for name, indices in ctx_cont_indices.items()
    }
    ctx_cont_nominal_indices = {
        name: [
            index
            for index in indices
            if MODEL_NATIVE_CTX_CONT_FIELDS[index] in nominal_set
        ]
        for name, indices in ctx_cont_indices.items()
    }
    routing_payload = {
        "ctx_cont_indices": ctx_cont_indices,
        "ctx_cat_indices": ctx_cat_indices,
        "ctx_cont_names": ctx_cont_names,
        "ctx_cat_names": ctx_cat_names,
        "ctx_cont_numeric_indices": ctx_cont_numeric_indices,
        "ctx_cont_nominal_indices": ctx_cont_nominal_indices,
    }
    return {
        "schema_version": MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_SCHEMA_VERSION,
        "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        "ctx_cont_fields_sha256": MODEL_NATIVE_CTX_CONT_FIELDS_SHA256,
        "ctx_cat_fields_sha256": MODEL_NATIVE_CTX_CAT_FIELDS_SHA256,
        **routing_payload,
        "routing_sha256": _canonical_sha256(routing_payload),
        "family_context_injection": "pre_cross_specialist_token",
        "global_context_source": "family_context_tokens",
        "independent_raw_global_context_projection": False,
        "all_context_fields_have_exactly_one_owner": True,
        "nominal_ctx_cont_fields": list(MODEL_NATIVE_NOMINAL_CTX_CONT_FIELDS),
        "nominal_ctx_cont_cardinality": (
            MODEL_NATIVE_NOMINAL_CTX_CONT_CARDINALITY
        ),
        "nominal_ctx_cont_representation": "none",
        "nominal_ctx_cont_hard_integer_domain_check": True,
        "nominal_ctx_cont_excluded_from_numeric_projection": True,
        "ctx_cat_domains": {
            name: list(domain)
            for name, domain in MODEL_NATIVE_CTX_CAT_DOMAINS.items()
        },
        "ctx_cat_embedding_scope": "separate_table_per_field",
        "numeric_ctx_cont_pre_projection_normalization": (
            "frozen_train_per_field_robust_only"
        ),
        "family_ctx_cont_layer_norm": False,
        "forward_identity_guards": [
            "seq_last_equals_snap_bit_identical",
            "snap_alias_equals_ctx_cont_bit_identical",
        ],
    }


MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT = (
    model_native_context_specialist_routing_contract()
)


def model_native_context_temporal_alias_policy(
    ordered_signal_names: Iterable[str],
) -> dict[str, object]:
    """Derive the exact temporal/context overlap owned by one signal artifact."""

    signal_fields = tuple(str(field) for field in ordered_signal_names)
    if len(signal_fields) != len(set(signal_fields)):
        raise RuntimeError("MODEL_NATIVE_CONTEXT_TEMPORAL_ALIAS_SIGNAL_DUPLICATE")
    ctx_index = {
        field: index for index, field in enumerate(MODEL_NATIVE_CTX_CONT_FIELDS)
    }
    aliases: list[dict[str, object]] = []
    for signal_index, signal_field in enumerate(signal_fields):
        if not signal_field.startswith("ctx_cont."):
            continue
        ctx_field = signal_field.removeprefix("ctx_cont.")
        if ctx_field not in ctx_index:
            continue
        owner = classify_entry_specialist_feature(f"ctx_cont.{ctx_field}")
        if owner not in MODEL_NATIVE_TRAINING_SPECIALISTS:
            raise RuntimeError(
                "MODEL_NATIVE_CONTEXT_TEMPORAL_ALIAS_OWNER_INVALID: "
                f"field={signal_field} owner={owner}"
            )
        aliases.append(
            {
                "signal_field": signal_field,
                "signal_index": signal_index,
                "ctx_cont_field": ctx_field,
                "ctx_cont_index": ctx_index[ctx_field],
                "specialist": owner,
            }
        )
    alias_fields = [str(alias["signal_field"]) for alias in aliases]
    return {
        "schema_version": (
            MODEL_NATIVE_CONTEXT_TEMPORAL_ALIAS_POLICY_SCHEMA_VERSION
        ),
        "derivation": (
            "ordered_signal_names_intersection_exact_ordered_ctx_cont_names"
        ),
        "alias_count": len(aliases),
        "signal_fields": alias_fields,
        "signal_fields_sha256": _canonical_sha256(alias_fields),
        "signal_indices": [int(alias["signal_index"]) for alias in aliases],
        "ctx_cont_indices": [int(alias["ctx_cont_index"]) for alias in aliases],
        "aliases": aliases,
        "aliases_sha256": _canonical_sha256(aliases),
        "signal_role": "temporal_sequence_only",
        "generic_snap_projection_excludes_aliases": True,
        "current_bar_context_source": "owner_family_context_token",
        # The executable owner is ``share_temporal_alias_stats_from_signal``
        # (entry_v10_input_normalization -> entry_model_native_input
        # _normalization_v1): the shared local signal population fits each
        # duplicated temporal field once and the ctx_cont surface receives a
        # bit-identical copy.  The declaration must name that direction.
        "statistics_owner": "signal",
        "signal_alias_statistics_policy": (
            "bit_identical_copy_from_signal_train_stats"
        ),
        "forward_identity_guard": (
            "snap_alias_equals_ctx_cont_bit_identical_before_transform"
        ),
    }


def require_model_native_context_specialist_routing(
    value: object,
    *,
    ordered_signal_names: Iterable[str],
    context: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError(f"[{context}_CONTEXT_SPECIALIST_ROUTING_MISSING]")
    observed = dict(value)
    expected = MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT
    for key in (
        "schema_version",
        "ctx_cont_dim",
        "ctx_cat_dim",
        "ctx_cont_fields_sha256",
        "ctx_cat_fields_sha256",
        "ctx_cont_indices",
        "ctx_cat_indices",
        "ctx_cont_names",
        "ctx_cat_names",
        "ctx_cont_numeric_indices",
        "ctx_cont_nominal_indices",
        "routing_sha256",
        "family_context_injection",
        "global_context_source",
        "independent_raw_global_context_projection",
        "all_context_fields_have_exactly_one_owner",
        "nominal_ctx_cont_fields",
        "nominal_ctx_cont_cardinality",
        "nominal_ctx_cont_representation",
        "nominal_ctx_cont_hard_integer_domain_check",
        "nominal_ctx_cont_excluded_from_numeric_projection",
        "ctx_cat_domains",
        "ctx_cat_embedding_scope",
        "numeric_ctx_cont_pre_projection_normalization",
        "family_ctx_cont_layer_norm",
        "forward_identity_guards",
    ):
        if observed.get(key) != expected.get(key):
            raise RuntimeError(
                f"[{context}_CONTEXT_SPECIALIST_ROUTING_MISMATCH] field={key}"
            )
    alias_policy_raw = observed.get("temporal_alias_policy")
    if not isinstance(alias_policy_raw, Mapping):
        raise RuntimeError(f"[{context}_CONTEXT_TEMPORAL_ALIAS_POLICY_MISSING]")
    expected_alias_policy = model_native_context_temporal_alias_policy(
        ordered_signal_names
    )
    if dict(alias_policy_raw) != expected_alias_policy:
        raise RuntimeError(f"[{context}_CONTEXT_TEMPORAL_ALIAS_POLICY_INVALID]")
    normalized = json.loads(json.dumps(observed))
    # Immutable JSON events serialize with sorted keys, so a mapping keyed by
    # specialist cannot carry the canonical registry order through disk. The
    # equality checks above prove exact content identity against the
    # code-owned contract; re-key the specialist-owned index maps in exact
    # registry order so every consumer receives the one canonical ordering.
    for key in (
        "ctx_cont_indices",
        "ctx_cat_indices",
        "ctx_cont_nominal_indices",
    ):
        mapping = normalized.get(key)
        if isinstance(mapping, dict):
            normalized[key] = {
                str(name): mapping[str(name)]
                for name in MODEL_NATIVE_TRAINING_SPECIALISTS
            }
    return normalized


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
