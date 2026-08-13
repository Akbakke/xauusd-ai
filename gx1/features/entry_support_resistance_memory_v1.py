"""Canonical support/resistance level-memory features for Entry specialists.

This smart layer derives causal numeric memory around already-materialized
support/resistance, pivot, liquidity and SMC fields for the model-native
train/serve feature path.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from gx1.features.entry_smc_liquidity_quality_v1 import (
    TREND_COMPOSITE_WEIGHTS,
    WICK_REJECTION_CLOSE_LOCATION_BASE,
    WICK_REJECTION_STRENGTH_ALGEBRAIC_MAX,
)


SUPPORT_RESISTANCE_MEMORY_FEATURE_VERSION = (
    "entry_support_resistance_memory_v4_20260813_removed_geometry_composite_inputs"
)
SUPPORT_RESISTANCE_MEMORY_FEATURE_PREFIX = "chart.sr_memory_"

# Declared output contract in exact _add order.  The build asserts the emitted
# names against this tuple (SUPPORT_RESISTANCE_MEMORY_FEATURE_NAME_DRIFT), the
# same hard drift guard the SMC/liquidity layer carries; the previous
# import-time execution of the builder on synthetic zeros is removed.
SUPPORT_RESISTANCE_MEMORY_FEATURE_SUFFIXES = (
    "nearest_pivot_proximity",
    "nearest_level_proximity",
    "support_level_proximity_stack",
    "resistance_level_proximity_stack",
    "support_minus_resistance_level_balance",
    "support_repeated_test_fast",
    "resistance_repeated_test_fast",
    "support_repeated_test_slow",
    "resistance_repeated_test_slow",
    "support_repeated_test_pressure",
    "resistance_repeated_test_pressure",
    "repeated_test_balance",
    "support_respect_pressure_long",
    "resistance_respect_pressure_short",
    "support_break_pressure_down",
    "resistance_break_pressure_up",
    "support_reclaim_pressure_long",
    "resistance_reclaim_pressure_short",
    "respect_minus_break_balance",
    "reclaim_minus_break_balance",
    "mtf_h1_h4_d1_support_confluence",
    "mtf_h1_h4_d1_resistance_confluence",
    "mtf_h1_h4_d1_level_confluence",
    "mtf_h1_h4_d1_support_minus_resistance",
    "liquidity_low_level_rejection_long",
    "liquidity_high_level_rejection_short",
    "liquidity_level_rejection_balance",
    "liquidity_resistance_break_continuation_long",
    "liquidity_support_break_continuation_short",
    "liquidity_level_continuation_balance",
    "two_sided_liquidity_level_pressure",
    "level_memory_exhaustion_risk",
    "major_level_trap_risk_long",
    "major_level_trap_risk_short",
)
SUPPORT_RESISTANCE_MEMORY_FEATURE_NAMES = tuple(
    f"{SUPPORT_RESISTANCE_MEMORY_FEATURE_PREFIX}{name}"
    for name in SUPPORT_RESISTANCE_MEMORY_FEATURE_SUFFIXES
)

SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS = (
    "ctx_cont.sr_nearest_pivot_abs_atr",
    "ctx_cont.sr_support_proximity_exp",
    "ctx_cont.sr_resistance_proximity_exp",
    "ctx_cont.sr_support_minus_resistance_prox",
    "ctx_cont.liquidity_hi_nearest_abs_atr",
    "ctx_cont.liquidity_lo_nearest_abs_atr",
    "ctx_cont.liquidity_lo_minus_hi_prox",
    "ctx_cont.dist_to_R1_atr",
    "ctx_cont.dist_to_R2_atr",
    "ctx_cont.dist_to_S1_atr",
    "ctx_cont.dist_to_S2_atr",
    "ctx_cont.dist_to_m5_hi_atr",
    "ctx_cont.dist_to_m5_lo_atr",
    "ctx_cont.dist_to_m15_hi_atr",
    "ctx_cont.dist_to_m15_lo_atr",
    "ctx_cont.dist_to_h1_hi_atr",
    "ctx_cont.dist_to_h1_lo_atr",
    "ctx_cont.dist_to_h4_hi_atr",
    "ctx_cont.dist_to_h4_lo_atr",
    "ctx_cont.dist_to_d1_hi_atr",
    "ctx_cont.dist_to_d1_lo_atr",
    "ctx_cont.dist_last_swing_high_atr",
    "ctx_cont.dist_last_swing_low_atr",
    "ctx_cont.bars_since_swing_high",
    "ctx_cont.bars_since_swing_low",
    "snap.smc_sweep_up",
    "snap.smc_sweep_down",
    "snap.smc_sweep_size_atr",
    "snap.smc_bars_since_sweep",
    "ctx_cont.smc_sweep_bull_pressure_last12",
    "ctx_cont.smc_sweep_bull_pressure_last48",
    "ctx_cont.smc_sweep_size_recent_tau12",
    "ctx_cont.smc_sweep_recency_tau24",
    "snap.smc_bos_up",
    "snap.smc_bos_down",
    "ctx_cont.smc_bos_pressure_last12",
    "ctx_cont.smc_bos_pressure_last48",
    "snap.smc_choch",
    "ctx_cont.smc_choch_recent_tau12",
    "ctx_cont.smc_choch_recent_tau24",
    "snap.smc_premium_discount",
    "candle.pattern_body_direction",
    "candle.pattern_body_share",
    "candle.pattern_upper_wick_share",
    "candle.pattern_lower_wick_share",
    "candle.pattern_close_location",
    "ctx_cont._v1h1_ema_diff",
    "ctx_cont._v1h4_ema_diff",
    "ctx_cont.d1_ema_slope_20_canon_v2",
    "ctx_cont.m15_trend_sign_canon_v2",
    "ctx_cont.regime_stack_sum_v3",
    "chart.geometry_support_line_proximity_stack",
    "chart.geometry_resistance_line_proximity_stack",
    # V30 package 7 (2026-08-13): `chart.geometry_support_minus_resistance_stack`,
    # `..._support_bounce_long_pressure`, `..._resistance_reject_short_pressure`
    # and `..._trendline_break_up/down_pressure` were removed from their
    # producer (exact-affine duplicate of the two stacks + one ctx field, and
    # three NAME-ONLY composites — the "trendline break" contained no line).
    # This layer keeps the two sided proximity stacks, which are the only
    # geometry inputs it actually needed.
    "chart.smc_liquidity_reclaim_confirmation_long",
    "chart.smc_liquidity_reclaim_confirmation_short",
    "chart.smc_liquidity_false_breakout_quality_long",
    "chart.smc_liquidity_false_breakout_quality_short",
)


def _name_index(names: Iterable[str]) -> dict[str, int]:
    values = list(names)
    invalid = [name for name in values if not isinstance(name, str) or not name]
    if invalid:
        raise RuntimeError(f"SUPPORT_RESISTANCE_MEMORY_FEATURE_NAMES_INVALID: {invalid[:10]}")
    duplicates = sorted({name for name in values if values.count(name) > 1})
    if duplicates:
        raise RuntimeError(f"SUPPORT_RESISTANCE_MEMORY_FEATURE_NAMES_DUPLICATE: {duplicates[:10]}")
    return {name: i for i, name in enumerate(values)}


def missing_support_resistance_memory_source_fields(feature_names: Iterable[str]) -> list[str]:
    available = set(feature_names)
    return [name for name in SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS if name not in available]


def _require_source_matrix(
    x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    try:
        matrix = np.asarray(x, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("SUPPORT_RESISTANCE_MEMORY_INPUT_NOT_NUMERIC") from exc
    if matrix.ndim != 2:
        raise RuntimeError(f"SUPPORT_RESISTANCE_MEMORY_INPUT_NOT_2D: shape={matrix.shape}")
    if matrix.shape[0] == 0:
        raise RuntimeError("SUPPORT_RESISTANCE_MEMORY_INPUT_EMPTY")
    if len(feature_names) != matrix.shape[1]:
        raise RuntimeError(
            "SUPPORT_RESISTANCE_MEMORY_FEATURE_NAME_COUNT_MISMATCH: "
            f"names={len(feature_names)} columns={matrix.shape[1]}"
        )
    index = _name_index(feature_names)
    missing = missing_support_resistance_memory_source_fields(feature_names)
    if missing:
        raise RuntimeError(
            "SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS_MISSING: "
            f"{missing[:30]} total={len(missing)}"
        )
    if not np.isfinite(matrix).all():
        bad = np.argwhere(~np.isfinite(matrix))[0]
        row, column = int(bad[0]), int(bad[1])
        raise RuntimeError(
            "SUPPORT_RESISTANCE_MEMORY_SOURCE_NONFINITE: "
            f"row={row} field={feature_names[column]}"
        )
    return matrix, index


def _col(x: np.ndarray, index: dict[str, int], name: str) -> np.ndarray:
    try:
        column = index[name]
    except KeyError as exc:
        raise RuntimeError(f"SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELD_MISSING: {name}") from exc
    arr = np.asarray(x[:, column], dtype=np.float32)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise RuntimeError(f"SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELD_INVALID: {name}")
    return arr


def _clip(arr: np.ndarray, lo: float = -25.0, hi: float = 25.0) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError(
            f"SUPPORT_RESISTANCE_MEMORY_DERIVED_VALUE_INVALID: shape={values.shape}"
        )
    return np.clip(values, lo, hi).astype(np.float32, copy=False)


def _clip01(arr: np.ndarray) -> np.ndarray:
    return _clip(arr, 0.0, 1.0)


def _pos(arr: np.ndarray) -> np.ndarray:
    return np.maximum(arr, 0.0).astype(np.float32, copy=False)


def _neg(arr: np.ndarray) -> np.ndarray:
    return np.maximum(-arr, 0.0).astype(np.float32, copy=False)


def _prox_abs(arr: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.abs(arr))).astype(np.float32, copy=False)


def _recency(arr: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.maximum(arr, 0.0))).astype(np.float32, copy=False)


def _tanh(arr: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return np.tanh(arr / max(float(scale), 1e-6)).astype(np.float32, copy=False)


SUPPORT_RESISTANCE_MEMORY_STATE_KEYS = (
    "support_fast",
    "support_slow",
    "resistance_fast",
    "resistance_slow",
)

# Exponential level-memory decays (per-M5-bar retention).  Values unchanged
# from the existing owner; named so the declared warmup bound below has a
# stated origin.
SR_MEMORY_FAST_DECAY = 0.82
SR_MEMORY_SLOW_DECAY = 0.96

# Declared contaminated-prefix length for a cold start (initial_carry = 0.0,
# no persisted memory state).  Origin: 5 * tau_slow, where
# tau_slow = -1 / ln(SR_MEMORY_SLOW_DECAY) ~= 24.5 bars, so 5 * tau ~= 122.5,
# declared upward to 125 rows per the 2026-08-09 repair directive (a declared
# prefix may exceed the derived bound; it may not undercut it).  The fast
# memory's tau ~= 5.0 bars is covered by the same prefix.  Callers that must
# exclude cold-start contamination drop this many leading rows or pass a
# persisted initial carry.
SR_MEMORY_WARMUP_ROWS = 125

# Algebraic maximum of the reclaim-pressure weighted sum
#   stack * liquidity * sweep * sweep_recent
#     * (0.45 + sweep_size + wick_reject + close_location)
#     * (0.70 + 0.30 * side)
#   + 0.35 * smc_liquidity_reclaim_confirmation:
# every non-bracket factor and every bracket input is bounded <= 1 (clip01,
# prox, or normalized wick rejection), so the first bracket is
# <= 0.45 + 1 + 1 + 1 = 3.45, the second <= 1.0, the product <= 3.45, and the
# additive confirmation term (producer hi=1.0) adds <= 0.35: total 3.80.
# Dividing by this bound before the [0, 1] clip maps the algebraic range onto
# the declared domain instead of saturating it.
SR_RECLAIM_PRESSURE_ALGEBRAIC_MAX = 3.80

# Algebraic maximum of the level-rejection weighted sum
#   liquidity * stack * (0.40 + sweep + sweep_recent)
#     * (0.35 + wick_reject + choch) * (0.80 + 0.20 * side)
#   + 0.30 * smc_liquidity_false_breakout_quality:
# brackets are <= 2.40, <= 2.35 and <= 1.0 with all other factors <= 1, so the
# product is <= 5.64; the single additive term (rescaled to [0, 1]) adds
# <= 0.30: total 5.94.  (History: 6.69 before the 2026-08-09 [0,5]-domain fix,
# 6.09 after it, and 5.94 from V30 package 7 on 2026-08-13, when the
# `+ 0.15 * (geometry_pressure / 5.0)` term was dropped with its removed
# producer.  The divisor is DERIVED from the expression it normalizes and must
# always match it — it is not a tunable.)
SR_LEVEL_REJECTION_ALGEBRAIC_MAX = 5.94


def _decayed_memory(
    arr: np.ndarray,
    decay: float,
    *,
    initial_carry: float = 0.0,
) -> tuple[np.ndarray, np.float32]:
    """Causal exponential memory with explicit cold-start semantics.

    carry[i] = decay * carry[i-1] + (1 - decay) * value[i], seeded with
    ``initial_carry``.  A cold start (initial_carry = 0.0) reads as "no S/R
    evidence yet" until real rows drive the recursion.  That is deliberate:
    mid-series NaN is forbidden, so the warmup prefix is emitted and instead
    DECLARED contaminated.  At the slow decay SR_MEMORY_SLOW_DECAY = 0.96 the
    time constant is tau = -1/ln(0.96) ~= 24.5 bars, so the cold-start bias has
    decayed to e^-3..e^-5 (~5%..0.7%) after ~3-5 tau ~= 75-125 bars; the
    declared bound exported for callers is SR_MEMORY_WARMUP_ROWS = 125.
    """
    values = _clip01(arr)
    out = np.empty_like(values, dtype=np.float32)
    carry = np.float32(initial_carry)
    if not np.isfinite(carry) or carry < 0.0 or carry > 1.0:
        raise RuntimeError("SUPPORT_RESISTANCE_MEMORY_STATE_INVALID")
    alpha = np.float32(1.0 - float(decay))
    decay32 = np.float32(decay)
    for i, value in enumerate(values):
        carry = decay32 * carry + alpha * np.float32(value)
        out[i] = carry
    clean = _clip01(out)
    return clean, np.float32(clean[-1])


def _add(
    arrays: list[np.ndarray],
    names: list[str],
    name: str,
    arr: np.ndarray,
    *,
    lo: float = -25.0,
    hi: float = 25.0,
) -> None:
    clean = _clip(np.asarray(arr, dtype=np.float32), lo, hi)
    if clean.ndim != 1:
        raise RuntimeError(f"support/resistance memory feature {name} is not 1D: {clean.shape}")
    if not np.isfinite(clean).all():
        raise RuntimeError(f"support/resistance memory feature {name} contains non-finite values")
    arrays.append(clean)
    names.append(f"{SUPPORT_RESISTANCE_MEMORY_FEATURE_PREFIX}{name}")


def build_entry_support_resistance_memory_layer(
    x: np.ndarray,
    feature_names: list[str],
    *,
    memory_state: Mapping[str, float] | None = None,
    return_memory_state: bool = False,
) -> (
    tuple[np.ndarray, list[str]]
    | tuple[np.ndarray, list[str], dict[str, np.float32]]
):
    """Build deterministic support/resistance memory from exact sources."""
    x, idx = _require_source_matrix(x, feature_names)
    if memory_state is None:
        state = {
            name: np.float32(0.0)
            for name in SUPPORT_RESISTANCE_MEMORY_STATE_KEYS
        }
    else:
        if set(memory_state) != set(SUPPORT_RESISTANCE_MEMORY_STATE_KEYS):
            raise RuntimeError("SUPPORT_RESISTANCE_MEMORY_STATE_SCHEMA_INVALID")
        state = {
            name: np.float32(memory_state[name])
            for name in SUPPORT_RESISTANCE_MEMORY_STATE_KEYS
        }
        if any(
            not np.isfinite(value) or value < 0.0 or value > 1.0
            for value in state.values()
        ):
            raise RuntimeError("SUPPORT_RESISTANCE_MEMORY_STATE_INVALID")
    arrays: list[np.ndarray] = []
    names: list[str] = []

    def c(name: str) -> np.ndarray:
        return _col(x, idx, name)

    h1_trend = _tanh(c("ctx_cont._v1h1_ema_diff"))
    h4_trend = _tanh(c("ctx_cont._v1h4_ema_diff"))
    d1_slope = _tanh(c("ctx_cont.d1_ema_slope_20_canon_v2"))
    m15_trend = _tanh(c("ctx_cont.m15_trend_sign_canon_v2"))
    regime_stack = _tanh(c("ctx_cont.regime_stack_sum_v3"), scale=3.0)
    # Single-owner trend composite (rule 19): weights imported from the
    # SMC/liquidity layer, whose constant cites
    # entry_model_native_feature_layers_v1.py:557-563 as the origin.
    w_h1, w_h4, w_d1, w_m15, w_regime = TREND_COMPOSITE_WEIGHTS
    trend = _clip(
        w_h1 * h1_trend + w_h4 * h4_trend + w_d1 * d1_slope + w_m15 * m15_trend + w_regime * regime_stack
    )
    trend_up = _pos(trend)
    trend_down = _neg(trend)

    pivot_proximity = _prox_abs(c("ctx_cont.sr_nearest_pivot_abs_atr"))
    s1 = _prox_abs(c("ctx_cont.dist_to_S1_atr"))
    s2 = _prox_abs(c("ctx_cont.dist_to_S2_atr"))
    r1 = _prox_abs(c("ctx_cont.dist_to_R1_atr"))
    r2 = _prox_abs(c("ctx_cont.dist_to_R2_atr"))
    h1_lo = _prox_abs(c("ctx_cont.dist_to_h1_lo_atr"))
    h1_hi = _prox_abs(c("ctx_cont.dist_to_h1_hi_atr"))
    h4_lo = _prox_abs(c("ctx_cont.dist_to_h4_lo_atr"))
    h4_hi = _prox_abs(c("ctx_cont.dist_to_h4_hi_atr"))
    d1_lo = _prox_abs(c("ctx_cont.dist_to_d1_lo_atr"))
    d1_hi = _prox_abs(c("ctx_cont.dist_to_d1_hi_atr"))
    swing_low = _prox_abs(c("ctx_cont.dist_last_swing_low_atr")) * (0.50 + _recency(c("ctx_cont.bars_since_swing_low")))
    swing_high = _prox_abs(c("ctx_cont.dist_last_swing_high_atr")) * (0.50 + _recency(c("ctx_cont.bars_since_swing_high")))

    # Support/resistance proximity stacks (2026-08-09 repair; the SMC/liquidity
    # layer carries the full derivation comment for the identical repair).
    # The max() previously ran over 12 rows; under the producer contract
    # (gx1/features/entry_smart_context.py:140-184) these max inputs were
    # algebraically dead and are removed (bit-identical, since _prox_abs of the
    # producer's min equals the componentwise max of _prox_abs and
    # exp(-x) <= 1/(1+x)):
    #   removed (dominated) row               dominator kept in the max
    #   _prox_abs(dist_to_{m5,m15,h1,h4,d1}_lo_atr)
    #                                         _prox_abs(liquidity_lo_nearest_abs_atr)
    #   _clip01(sr_support_proximity_exp)     max(s1, s2) = 1/(1+s_abs)
    # and the mirrored hi/R rows for the resistance stack.  Separately the
    # shared row pivot_proximity = _prox_abs(min(r_abs, s_abs)) appeared in
    # BOTH stacks, flooring each side's proximity by the other side's distance
    # and making the stacks bit-equal whenever it won; it is removed from both
    # maxes (VALUE CHANGE) — the side-specific rows s1/s2 and r1/r2 remain and
    # now carry the side information.  It stays an emitted feature below.
    support_stack = np.maximum.reduce(
        [
            s1,
            s2,
            _prox_abs(c("ctx_cont.liquidity_lo_nearest_abs_atr")),
            _clip01(swing_low),
            _clip01(c("chart.geometry_support_line_proximity_stack")),
        ]
    ).astype(np.float32)
    resistance_stack = np.maximum.reduce(
        [
            r1,
            r2,
            _prox_abs(c("ctx_cont.liquidity_hi_nearest_abs_atr")),
            _clip01(swing_high),
            _clip01(c("chart.geometry_resistance_line_proximity_stack")),
        ]
    ).astype(np.float32)
    nearest_level = np.maximum(support_stack, resistance_stack).astype(np.float32)
    # V30 package 7 (2026-08-13): the removed
    # `chart.geometry_support_minus_resistance_stack` term was itself
    # `clip(sr_support_minus_resistance_prox + geometry_support_stack -
    # geometry_resistance_stack)`, and both geometry stacks are already folded
    # into this layer's own `support_stack`/`resistance_stack` above — so the
    # term was a double count of exactly this expression.  Dropping it leaves
    # the same three quantities, each still a model input.
    support_minus_resistance = _clip(
        c("ctx_cont.sr_support_minus_resistance_prox") + support_stack - resistance_stack,
        -2.0,
        2.0,
    )

    support_mtf = _clip01(0.34 * h1_lo + 0.33 * h4_lo + 0.33 * d1_lo)
    resistance_mtf = _clip01(0.34 * h1_hi + 0.33 * h4_hi + 0.33 * d1_hi)
    mtf_level_confluence = _clip01(np.maximum(support_mtf, resistance_mtf) * (0.70 + 0.30 * nearest_level))
    mtf_balance = _clip(support_mtf - resistance_mtf, -1.0, 1.0)

    low_liquidity = np.maximum(_prox_abs(c("ctx_cont.liquidity_lo_nearest_abs_atr")), support_stack).astype(np.float32)
    high_liquidity = np.maximum(_prox_abs(c("ctx_cont.liquidity_hi_nearest_abs_atr")), resistance_stack).astype(np.float32)
    liquidity_balance = _clip(c("ctx_cont.liquidity_lo_minus_hi_prox") + low_liquidity - high_liquidity, -2.0, 2.0)
    two_sided_liquidity = _clip01(np.minimum(low_liquidity, high_liquidity) * (0.50 + 0.50 * nearest_level))

    support_touch_event = _clip01(0.45 * support_stack + 0.25 * support_mtf + 0.20 * low_liquidity + 0.10 * _pos(support_minus_resistance))
    resistance_touch_event = _clip01(0.45 * resistance_stack + 0.25 * resistance_mtf + 0.20 * high_liquidity + 0.10 * _neg(support_minus_resistance))
    support_fast, support_fast_state = _decayed_memory(
        support_touch_event,
        SR_MEMORY_FAST_DECAY,
        initial_carry=state["support_fast"],
    )
    support_slow, support_slow_state = _decayed_memory(
        support_touch_event,
        SR_MEMORY_SLOW_DECAY,
        initial_carry=state["support_slow"],
    )
    resistance_fast, resistance_fast_state = _decayed_memory(
        resistance_touch_event,
        SR_MEMORY_FAST_DECAY,
        initial_carry=state["resistance_fast"],
    )
    resistance_slow, resistance_slow_state = _decayed_memory(
        resistance_touch_event,
        SR_MEMORY_SLOW_DECAY,
        initial_carry=state["resistance_slow"],
    )
    support_repeated = _clip01((0.62 * support_fast + 0.38 * support_slow) * (0.50 + support_stack))
    resistance_repeated = _clip01((0.62 * resistance_fast + 0.38 * resistance_slow) * (0.50 + resistance_stack))
    repeated_balance = _clip(support_repeated - resistance_repeated, -1.0, 1.0)

    sweep_up = _clip01(
        c("snap.smc_sweep_up")
        + 0.50 * _neg(c("ctx_cont.smc_sweep_bull_pressure_last12"))
        + 0.25 * _neg(c("ctx_cont.smc_sweep_bull_pressure_last48"))
    )
    sweep_down = _clip01(
        c("snap.smc_sweep_down")
        + 0.50 * _pos(c("ctx_cont.smc_sweep_bull_pressure_last12"))
        + 0.25 * _pos(c("ctx_cont.smc_sweep_bull_pressure_last48"))
    )
    sweep_recent = _clip01(_recency(c("snap.smc_bars_since_sweep")) + c("ctx_cont.smc_sweep_recency_tau24"))
    sweep_size = _clip01(c("snap.smc_sweep_size_atr") + c("ctx_cont.smc_sweep_size_recent_tau12"))
    bos_up = _clip01(c("snap.smc_bos_up") + 0.50 * _pos(c("ctx_cont.smc_bos_pressure_last12")) + 0.25 * _pos(c("ctx_cont.smc_bos_pressure_last48")))
    bos_down = _clip01(c("snap.smc_bos_down") + 0.50 * _neg(c("ctx_cont.smc_bos_pressure_last12")) + 0.25 * _neg(c("ctx_cont.smc_bos_pressure_last48")))
    choch = _clip01(c("snap.smc_choch") + c("ctx_cont.smc_choch_recent_tau12") + 0.50 * c("ctx_cont.smc_choch_recent_tau24"))

    premium = _clip01(c("snap.smc_premium_discount"))
    discount = _clip01(1.0 - premium)
    body_direction = _clip(c("candle.pattern_body_direction"), -1.0, 1.0)
    body_share = _clip01(c("candle.pattern_body_share"))
    body_bull = _pos(body_direction)
    body_bear = _neg(body_direction)
    lower_wick = _clip01(c("candle.pattern_lower_wick_share"))
    upper_wick = _clip01(c("candle.pattern_upper_wick_share"))
    close_near_high = _clip01(c("candle.pattern_close_location"))
    close_near_low = _clip01(1.0 - close_near_high)
    # Wick rejection: identical expression to the SMC/liquidity layer, which is
    # the single owner of the base term (this file's 0.55 copy had no separate
    # origin and is retired, rule 19); normalized by the shared derived
    # algebraic maximum before the [0, 1] clip.
    wick_reject_long = _clip01(
        lower_wick
        * (WICK_REJECTION_CLOSE_LOCATION_BASE + close_near_high + 0.25 * body_bull)
        * (0.75 + 0.25 * body_share)
        / WICK_REJECTION_STRENGTH_ALGEBRAIC_MAX
    )
    wick_reject_short = _clip01(
        upper_wick
        * (WICK_REJECTION_CLOSE_LOCATION_BASE + close_near_low + 0.25 * body_bear)
        * (0.75 + 0.25 * body_share)
        / WICK_REJECTION_STRENGTH_ALGEBRAIC_MAX
    )

    support_respect_long = _clip01(
        support_stack
        * (0.45 + 0.25 * support_repeated + 0.20 * support_mtf + 0.10 * discount)
        * (0.45 + 0.35 * wick_reject_long + 0.20 * body_bull)
    )
    resistance_respect_short = _clip01(
        resistance_stack
        * (0.45 + 0.25 * resistance_repeated + 0.20 * resistance_mtf + 0.10 * premium)
        * (0.45 + 0.35 * wick_reject_short + 0.20 * body_bear)
    )
    support_break_down = _clip(
        support_stack
        * (0.45 + support_repeated + support_mtf)
        * (0.45 + bos_down + trend_down + body_bear)
        * (1.0 - 0.50 * wick_reject_long),
        0.0,
        5.0,
    )
    resistance_break_up = _clip(
        resistance_stack
        * (0.45 + resistance_repeated + resistance_mtf)
        * (0.45 + bos_up + trend_up + body_bull)
        * (1.0 - 0.50 * wick_reject_short),
        0.0,
        5.0,
    )
    # Normalized by the derived algebraic maximum before the [0, 1] clip (see
    # SR_RECLAIM_PRESSURE_ALGEBRAIC_MAX derivation at module level).
    support_reclaim_long = _clip01(
        (
            support_stack
            * low_liquidity
            * sweep_down
            * sweep_recent
            * (0.45 + sweep_size + wick_reject_long + close_near_high)
            * (0.70 + 0.30 * discount)
            + 0.35 * c("chart.smc_liquidity_reclaim_confirmation_long")
        )
        / SR_RECLAIM_PRESSURE_ALGEBRAIC_MAX
    )
    resistance_reclaim_short = _clip01(
        (
            resistance_stack
            * high_liquidity
            * sweep_up
            * sweep_recent
            * (0.45 + sweep_size + wick_reject_short + close_near_low)
            * (0.70 + 0.30 * premium)
            + 0.35 * c("chart.smc_liquidity_reclaim_confirmation_short")
        )
        / SR_RECLAIM_PRESSURE_ALGEBRAIC_MAX
    )
    respect_minus_break = _clip(support_respect_long + resistance_respect_short - 0.50 * (support_break_down + resistance_break_up), -2.0, 2.0)
    reclaim_minus_break = _clip(support_reclaim_long + resistance_reclaim_short - 0.50 * (support_break_down + resistance_break_up), -2.0, 2.0)

    # The geometry pressures are produced on a declared [0, 5] domain
    # (entry_chart_geometry_v1.py:418-449, hi=5.0); rescale them to [0, 1]
    # before blending with [0, 1] inputs — the same /5.0 convention the
    # SMC/liquidity layer already applies to its foundation/geometry
    # consumptions.  The whole weighted sum is then normalized by its derived
    # algebraic maximum before the [0, 1] clip (see
    # SR_LEVEL_REJECTION_ALGEBRAIC_MAX derivation at module level).
    low_level_rejection_long = _clip01(
        (
            low_liquidity
            * support_stack
            * (0.40 + sweep_down + sweep_recent)
            * (0.35 + wick_reject_long + choch)
            * (0.80 + 0.20 * discount)
            + 0.30 * c("chart.smc_liquidity_false_breakout_quality_long")
        )
        / SR_LEVEL_REJECTION_ALGEBRAIC_MAX
    )
    high_level_rejection_short = _clip01(
        (
            high_liquidity
            * resistance_stack
            * (0.40 + sweep_up + sweep_recent)
            * (0.35 + wick_reject_short + choch)
            * (0.80 + 0.20 * premium)
            + 0.30 * c("chart.smc_liquidity_false_breakout_quality_short")
        )
        / SR_LEVEL_REJECTION_ALGEBRAIC_MAX
    )
    rejection_balance = _clip(low_level_rejection_long - high_level_rejection_short, -1.0, 1.0)
    resistance_continuation_long = _clip(
        high_liquidity
        * resistance_stack
        * (0.45 + resistance_repeated + resistance_mtf)
        * (0.45 + bos_up + trend_up + body_bull)
        # V30 package 7 (2026-08-13): the `+ 0.20 *
        # (chart.geometry_trendline_break_up_pressure / 5.0)` term is gone with
        # its removed producer (that field contained no line: its "break" was
        # `smc_bos_up`, which this expression already reads directly as
        # `bos_up`).  The surviving factors are unchanged - no weight is
        # renormalized, because inventing a replacement magnitude is forbidden.
        * (1.0 - 0.40 * wick_reject_short),
        0.0,
        5.0,
    )
    support_continuation_short = _clip(
        low_liquidity
        * support_stack
        * (0.45 + support_repeated + support_mtf)
        * (0.45 + bos_down + trend_down + body_bear)
        # V30 package 7 (2026-08-13): mirror of the long side above - the
        # removed `chart.geometry_trendline_break_down_pressure` term carried
        # `smc_bos_down`, which this expression already reads as `bos_down`.
        * (1.0 - 0.40 * wick_reject_long),
        0.0,
        5.0,
    )
    continuation_balance = _clip(resistance_continuation_long - support_continuation_short, -2.0, 2.0)
    exhaustion_risk = _clip(
        two_sided_liquidity
        * nearest_level
        * (0.40 + support_repeated + resistance_repeated)
        * (0.40 + sweep_size + choch)
        * (0.80 + 0.20 * np.abs(liquidity_balance)),
        0.0,
        5.0,
    )
    trap_risk_long = _clip(high_liquidity * resistance_stack * premium * (0.50 + sweep_up + wick_reject_short) * (1.0 - support_reclaim_long), 0.0, 5.0)
    trap_risk_short = _clip(low_liquidity * support_stack * discount * (0.50 + sweep_down + wick_reject_long) * (1.0 - resistance_reclaim_short), 0.0, 5.0)

    _add(arrays, names, "nearest_pivot_proximity", pivot_proximity, lo=0.0, hi=1.0)
    _add(arrays, names, "nearest_level_proximity", nearest_level, lo=0.0, hi=1.0)
    _add(arrays, names, "support_level_proximity_stack", support_stack, lo=0.0, hi=1.0)
    _add(arrays, names, "resistance_level_proximity_stack", resistance_stack, lo=0.0, hi=1.0)
    _add(arrays, names, "support_minus_resistance_level_balance", support_minus_resistance, lo=-2.0, hi=2.0)
    _add(arrays, names, "support_repeated_test_fast", support_fast, lo=0.0, hi=1.0)
    _add(arrays, names, "resistance_repeated_test_fast", resistance_fast, lo=0.0, hi=1.0)
    _add(arrays, names, "support_repeated_test_slow", support_slow, lo=0.0, hi=1.0)
    _add(arrays, names, "resistance_repeated_test_slow", resistance_slow, lo=0.0, hi=1.0)
    _add(arrays, names, "support_repeated_test_pressure", support_repeated, lo=0.0, hi=1.0)
    _add(arrays, names, "resistance_repeated_test_pressure", resistance_repeated, lo=0.0, hi=1.0)
    _add(arrays, names, "repeated_test_balance", repeated_balance, lo=-1.0, hi=1.0)
    _add(arrays, names, "support_respect_pressure_long", support_respect_long, lo=0.0, hi=1.0)
    _add(arrays, names, "resistance_respect_pressure_short", resistance_respect_short, lo=0.0, hi=1.0)
    _add(arrays, names, "support_break_pressure_down", support_break_down, lo=0.0, hi=5.0)
    _add(arrays, names, "resistance_break_pressure_up", resistance_break_up, lo=0.0, hi=5.0)
    _add(arrays, names, "support_reclaim_pressure_long", support_reclaim_long, lo=0.0, hi=1.0)
    _add(arrays, names, "resistance_reclaim_pressure_short", resistance_reclaim_short, lo=0.0, hi=1.0)
    _add(arrays, names, "respect_minus_break_balance", respect_minus_break, lo=-2.0, hi=2.0)
    _add(arrays, names, "reclaim_minus_break_balance", reclaim_minus_break, lo=-2.0, hi=2.0)
    _add(arrays, names, "mtf_h1_h4_d1_support_confluence", support_mtf, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_h1_h4_d1_resistance_confluence", resistance_mtf, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_h1_h4_d1_level_confluence", mtf_level_confluence, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_h1_h4_d1_support_minus_resistance", mtf_balance, lo=-1.0, hi=1.0)
    _add(arrays, names, "liquidity_low_level_rejection_long", low_level_rejection_long, lo=0.0, hi=1.0)
    _add(arrays, names, "liquidity_high_level_rejection_short", high_level_rejection_short, lo=0.0, hi=1.0)
    _add(arrays, names, "liquidity_level_rejection_balance", rejection_balance, lo=-1.0, hi=1.0)
    _add(arrays, names, "liquidity_resistance_break_continuation_long", resistance_continuation_long, lo=0.0, hi=5.0)
    _add(arrays, names, "liquidity_support_break_continuation_short", support_continuation_short, lo=0.0, hi=5.0)
    _add(arrays, names, "liquidity_level_continuation_balance", continuation_balance, lo=-2.0, hi=2.0)
    _add(arrays, names, "two_sided_liquidity_level_pressure", two_sided_liquidity, lo=0.0, hi=1.0)
    _add(arrays, names, "level_memory_exhaustion_risk", exhaustion_risk, lo=0.0, hi=5.0)
    _add(arrays, names, "major_level_trap_risk_long", trap_risk_long, lo=0.0, hi=5.0)
    _add(arrays, names, "major_level_trap_risk_short", trap_risk_short, lo=0.0, hi=5.0)

    out = np.column_stack(arrays).astype(np.float32, copy=False) if arrays else np.empty((x.shape[0], 0), dtype=np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("support/resistance memory layer contains non-finite values")
    if len(set(names)) != len(names):
        dupes = sorted({name for name in names if names.count(name) > 1})
        raise RuntimeError(f"support/resistance memory layer has duplicate names: {dupes[:10]}")
    if tuple(names) != SUPPORT_RESISTANCE_MEMORY_FEATURE_NAMES:
        raise RuntimeError("SUPPORT_RESISTANCE_MEMORY_FEATURE_NAME_DRIFT")
    if return_memory_state:
        return out, names, {
            "support_fast": support_fast_state,
            "support_slow": support_slow_state,
            "resistance_fast": resistance_fast_state,
            "resistance_slow": resistance_slow_state,
        }
    return out, names
