"""Entry seq146 feature-to-specialist grouping contract.

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


def _norm(name: str) -> str:
    return str(name or "").strip().lower()


def _contains_any(name: str, tokens: Iterable[str]) -> bool:
    n = _norm(name)
    return any(token in n for token in tokens)


def classify_entry_specialist_feature(name: str) -> str:
    """Return the primary specialist group for one emitted seq/snap feature."""
    n = _norm(name)
    bare = n
    for prefix in ("chart.", "ctx_cont.", "snap.", "seq."):
        if bare.startswith(prefix):
            bare = bare[len(prefix) :]
            break

    if bare in NEUTRAL_BRIDGE_FIELDS:
        return "neutral_bridge_anchor"

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
            "is_eu_only",
            "is_asia",
            "asia",
            "eu_x_",
            "us_x_",
            "overlap",
            "session",
            "hour_",
            "dow_",
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
