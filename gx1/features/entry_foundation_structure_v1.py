"""Explicit Entry foundation structure features.

This layer turns the existing causal ``snap.*`` and ``ctx_cont.*`` fields into
first-class model-native sequence evidence. It does not
read future prices and it intentionally keeps proxy names explicit where the
current contract does not expose the exact raw event.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterable

import numpy as np

from gx1.features.entry_volatility_semantics_v1 import (
    atr_ratio_compression_pressure,
    atr_ratio_expansion_pressure,
    bollinger_expansion_pressure,
    bollinger_squeeze_pressure,
)


FOUNDATION_STRUCTURE_FEATURE_VERSION = (
    "entry_foundation_structure_v5_20260813_swing_state_enum_decomposition_repair"
)
FOUNDATION_STRUCTURE_FEATURE_PREFIX = "chart.foundation_"
FOUNDATION_EVENT_AGE_CAP = 96
FOUNDATION_EVENT_AGE_CARRY_KEYS = (
    "bos_up_age_bars",
    "bos_down_age_bars",
    "choch_age_bars",
)
FOUNDATION_STRUCTURE_REQUIRED_FAMILIES = (
    "hh_hl_lh_ll_state",
    "bos_choch_age",
    "sweep_reclaim_false_breakout",
    "compression_expansion",
    "impulse_pullback_phase",
    "session_x_structure",
)
FOUNDATION_STRUCTURE_SOURCE_FIELDS = (
    "ctx_cont._v1h1_ema_diff",
    "ctx_cont._v1h4_ema_diff",
    "ctx_cont.d1_ema_slope_20_canon_v2",
    "ctx_cont.m15_trend_sign_canon_v2",
    "ctx_cont.regime_stack_sum_v3",
    "ctx_cont.dist_last_swing_high_atr",
    "ctx_cont.dist_last_swing_low_atr",
    "ctx_cont.bars_since_swing_high",
    "ctx_cont.bars_since_swing_low",
    "snap.smc_swing_state",
    "snap.smc_bos_up",
    "snap.smc_bos_down",
    "ctx_cont.smc_bos_pressure_last12",
    "ctx_cont.smc_bos_pressure_last48",
    "snap.smc_choch",
    "ctx_cont.smc_choch_recent_tau12",
    "ctx_cont.smc_choch_recent_tau24",
    "ctx_cont.struct_pullback_depth_h1_v3",
    "ctx_cont.struct_pullback_depth_h4_v3",
    "ctx_cont.retracement_from_last_impulse",
    "snap.smc_sweep_up",
    "snap.smc_sweep_down",
    "snap.smc_bars_since_sweep",
    "ctx_cont.smc_sweep_bull_pressure_last12",
    "ctx_cont.smc_sweep_bull_pressure_last48",
    "snap.smc_sweep_size_atr",
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
)


def _name_index(names: Iterable[str]) -> dict[str, int]:
    values = list(names)
    invalid = [name for name in values if not isinstance(name, str) or not name]
    if invalid:
        raise RuntimeError(f"FOUNDATION_STRUCTURE_FEATURE_NAMES_INVALID: {invalid[:10]}")
    duplicates = sorted({name for name in values if values.count(name) > 1})
    if duplicates:
        raise RuntimeError(f"FOUNDATION_STRUCTURE_FEATURE_NAMES_DUPLICATE: {duplicates[:10]}")
    return {name: i for i, name in enumerate(values)}


def missing_foundation_structure_source_fields(feature_names: Iterable[str]) -> list[str]:
    available = set(feature_names)
    return [name for name in FOUNDATION_STRUCTURE_SOURCE_FIELDS if name not in available]


def _require_source_matrix(
    x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    try:
        matrix = np.asarray(x, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("FOUNDATION_STRUCTURE_INPUT_NOT_NUMERIC") from exc
    if matrix.ndim != 2:
        raise RuntimeError(f"FOUNDATION_STRUCTURE_INPUT_NOT_2D: shape={matrix.shape}")
    if matrix.shape[0] == 0:
        raise RuntimeError("FOUNDATION_STRUCTURE_INPUT_EMPTY")
    if len(feature_names) != matrix.shape[1]:
        raise RuntimeError(
            "FOUNDATION_STRUCTURE_FEATURE_NAME_COUNT_MISMATCH: "
            f"names={len(feature_names)} columns={matrix.shape[1]}"
        )
    index = _name_index(feature_names)
    missing = missing_foundation_structure_source_fields(feature_names)
    if missing:
        raise RuntimeError(
            "FOUNDATION_STRUCTURE_SOURCE_FIELDS_MISSING: "
            f"{missing[:30]} total={len(missing)}"
        )
    if not np.isfinite(matrix).all():
        bad = np.argwhere(~np.isfinite(matrix))[0]
        row, column = int(bad[0]), int(bad[1])
        raise RuntimeError(
            "FOUNDATION_STRUCTURE_SOURCE_NONFINITE: "
            f"row={row} field={feature_names[column]}"
        )
    return matrix, index


def _col(x: np.ndarray, index: dict[str, int], name: str) -> np.ndarray:
    try:
        column = index[name]
    except KeyError as exc:
        raise RuntimeError(f"FOUNDATION_STRUCTURE_SOURCE_FIELD_MISSING: {name}") from exc
    arr = np.asarray(x[:, column], dtype=np.float32)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise RuntimeError(f"FOUNDATION_STRUCTURE_SOURCE_FIELD_INVALID: {name}")
    return arr


def _clip(arr: np.ndarray, lo: float = -25.0, hi: float = 25.0) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError(f"FOUNDATION_STRUCTURE_DERIVED_VALUE_INVALID: shape={values.shape}")
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


def _lag1(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr, dtype=np.float32)
    if arr.size:
        out[0] = 0.0
        out[1:] = arr[:-1]
    return out


def _require_event_age_value(value: object, *, cap: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise RuntimeError("FOUNDATION_EVENT_AGE_CARRY_STATE_INVALID")
    numeric = float(value)
    if (
        not np.isfinite(numeric)
        or not numeric.is_integer()
        or numeric < 0.0
        or numeric > float(cap)
    ):
        raise RuntimeError("FOUNDATION_EVENT_AGE_CARRY_STATE_INVALID")
    return int(numeric)


def _require_event_age_carry_state(
    state: Mapping[str, object] | None,
) -> dict[str, int]:
    if state is None:
        return {
            name: FOUNDATION_EVENT_AGE_CAP
            for name in FOUNDATION_EVENT_AGE_CARRY_KEYS
        }
    if not isinstance(state, Mapping) or set(state) != set(
        FOUNDATION_EVENT_AGE_CARRY_KEYS
    ):
        raise RuntimeError("FOUNDATION_EVENT_AGE_CARRY_STATE_INVALID")
    return {
        name: _require_event_age_value(
            state[name],
            cap=FOUNDATION_EVENT_AGE_CAP,
        )
        for name in FOUNDATION_EVENT_AGE_CARRY_KEYS
    }


class _FoundationEventAgeCarryScope:
    __slots__ = ("initial_state", "tail_replay_rows", "next_state")

    def __init__(self, initial_state: dict[str, int], tail_replay_rows: int) -> None:
        self.initial_state = initial_state
        self.tail_replay_rows = tail_replay_rows
        self.next_state: dict[str, int] = {}


_FOUNDATION_EVENT_AGE_CARRY_SCOPE: ContextVar[
    _FoundationEventAgeCarryScope | None
] = ContextVar(
    "foundation_event_age_carry_scope",
    default=None,
)


@contextmanager
def foundation_event_age_carry_scope(
    state: Mapping[str, object] | None,
    *,
    tail_replay_rows: int,
) -> Iterator[_FoundationEventAgeCarryScope]:
    """Bound event-age continuation to one synchronous feature build."""

    if _FOUNDATION_EVENT_AGE_CARRY_SCOPE.get() is not None:
        raise RuntimeError("FOUNDATION_EVENT_AGE_CARRY_SCOPE_NESTED")
    if (
        isinstance(tail_replay_rows, bool)
        or not isinstance(tail_replay_rows, int)
        or tail_replay_rows < 0
    ):
        raise RuntimeError("FOUNDATION_EVENT_AGE_CARRY_REPLAY_ROWS_INVALID")
    scope = _FoundationEventAgeCarryScope(
        _require_event_age_carry_state(state),
        tail_replay_rows,
    )
    token = _FOUNDATION_EVENT_AGE_CARRY_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _FOUNDATION_EVENT_AGE_CARRY_SCOPE.reset(token)


def _bars_since_event(
    event: np.ndarray,
    *,
    cap: int = FOUNDATION_EVENT_AGE_CAP,
    carry_key: str,
) -> np.ndarray:
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise RuntimeError("FOUNDATION_EVENT_AGE_CAP_INVALID")
    flags = np.asarray(event, dtype=bool)
    if flags.ndim != 1:
        raise RuntimeError("FOUNDATION_EVENT_AGE_FLAGS_INVALID")
    scope = _FOUNDATION_EVENT_AGE_CARRY_SCOPE.get()
    if scope is not None and (
        cap != FOUNDATION_EVENT_AGE_CAP
        or carry_key not in FOUNDATION_EVENT_AGE_CARRY_KEYS
        or carry_key in scope.next_state
    ):
        raise RuntimeError("FOUNDATION_EVENT_AGE_CARRY_CAPTURE_INVALID")
    out = np.empty(flags.shape[0], dtype=np.float32)
    age = cap if scope is None else scope.initial_state[carry_key]
    for i, hit in enumerate(flags):
        if bool(hit):
            age = 0
        else:
            age = min(age + 1, cap)
        out[i] = float(age)
    if scope is not None:
        carry_index = flags.size - scope.tail_replay_rows - 1
        scope.next_state[carry_key] = (
            scope.initial_state[carry_key]
            if carry_index < 0
            else int(out[carry_index])
        )
    return out


def _state_eq(state: np.ndarray, value: int) -> np.ndarray:
    return (np.rint(state).astype(np.int16, copy=False) == int(value)).astype(np.float32)


def _add(arrays: list[np.ndarray], names: list[str], name: str, arr: np.ndarray, *, lo: float = -25.0, hi: float = 25.0) -> None:
    clean = _clip(np.asarray(arr, dtype=np.float32), lo, hi)
    if clean.ndim != 1:
        raise RuntimeError(f"foundation feature {name} is not 1D: {clean.shape}")
    if not np.isfinite(clean).all():
        raise RuntimeError(f"foundation feature {name} contains non-finite values")
    arrays.append(clean)
    names.append(f"{FOUNDATION_STRUCTURE_FEATURE_PREFIX}{name}")


def build_entry_foundation_structure_layer(
    x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build explicit foundation structure evidence from exact contract fields."""
    x, idx = _require_source_matrix(x, feature_names)
    arrays: list[np.ndarray] = []
    names: list[str] = []

    def c(name: str) -> np.ndarray:
        return _col(x, idx, name)

    # Unit assumptions (owners): _v1h*_ema_diff and d1_ema_slope_20_canon_v2
    # are emitted in ATR-multiples by htf_features (unit-conversion owner), so
    # tanh at scale=1.0 is dimensionally sane; m15_trend_sign_canon_v2 is a
    # dimensionless sign in {-1,0,1} (htf_features), scale=1.0 sane.
    h1_trend = _tanh(c("ctx_cont._v1h1_ema_diff"))
    h4_trend = _tanh(c("ctx_cont._v1h4_ema_diff"))
    d1_slope = _tanh(c("ctx_cont.d1_ema_slope_20_canon_v2"))
    m15_trend = _tanh(c("ctx_cont.m15_trend_sign_canon_v2"))
    regime_stack = _tanh(c("ctx_cont.regime_stack_sum_v3"), scale=3.0)
    trend = _clip(0.35 * h1_trend + 0.30 * h4_trend + 0.20 * d1_slope + 0.10 * m15_trend + 0.05 * regime_stack)
    trend_up = _pos(trend)
    trend_down = _neg(trend)

    near_high = _prox_abs(c("ctx_cont.dist_last_swing_high_atr"))
    near_low = _prox_abs(c("ctx_cont.dist_last_swing_low_atr"))
    recent_high = _recency(c("ctx_cont.bars_since_swing_high"))
    recent_low = _recency(c("ctx_cont.bars_since_swing_low"))
    high_context = _clip01(near_high * (0.50 + recent_high))
    low_context = _clip01(near_low * (0.50 + recent_low))

    # ``snap.smc_swing_state`` enum (owner: gx1/features/smc_v1.py, the
    # ``swing_state`` block): 0 = HH+HL (clean up), 1 = HH+LL (two-sided
    # expansion), 2 = LH+HL (contraction/inside), 3 = LH+LL (clean down),
    # 4 = exact tie or warmup.  The four indicators below are named after the
    # enum's own two-sided decomposition, never after a direction bias.
    #
    # V30 package 8A (2026-08-13) — LIVE-BUG REPAIR, no new magnitude
    # (docs/INDICATOR_FIDELITY_AUDIT_20260813.md §4a).  States 1 and 2 were
    # read as ``up_bias``/``down_bias``: state 1 was weighted 0.55 into
    # ``hl_state`` although state 1 carries a LOWER low (evidence AGAINST a
    # higher low), and state 2 was weighted 0.45 into ``ll_state`` although
    # state 2 carries a HIGHER low.  Before the 2026-08-09 smc partition
    # repair both states were reachable only on exact float ties, so the
    # mislabel was inert; that repair made them live.  The repair here is the
    # literal decomposition of the enum — each of the four HH/HL/LH/LL
    # indicators is the sum of the two states that actually contain it:
    #     HH in {0, 1}   HL in {0, 2}   LH in {1, 3}   LL in {2, 3}
    # Every weight below is UNCHANGED and keeps its original role (the
    # "clean" state of the pair keeps the clean-state weight, the mixed state
    # keeps the mixed-state weight); only the state each weight multiplies is
    # corrected.  Re-weighting would invent magnitudes (rule 2b).  Net effect:
    # ``hh_state`` and ``ll_state`` were already the correct pairs and are
    # byte-identical; ``hl_state`` swaps state 1 -> state 2 and ``lh_state``
    # swaps state 2 -> state 1.
    swing_state = c("snap.smc_swing_state")
    state_hh_hl = _state_eq(swing_state, 0)
    state_hh_ll = _state_eq(swing_state, 1)
    state_lh_hl = _state_eq(swing_state, 2)
    state_lh_ll = _state_eq(swing_state, 3)

    bos_up_event = _clip01(c("snap.smc_bos_up"))
    bos_down_event = _clip01(c("snap.smc_bos_down"))
    bos_pressure12 = c("ctx_cont.smc_bos_pressure_last12")
    bos_pressure48 = c("ctx_cont.smc_bos_pressure_last48")
    bos_up_pressure = _clip01(bos_up_event + 0.5 * _pos(bos_pressure12) + 0.25 * _pos(bos_pressure48))
    bos_down_pressure = _clip01(bos_down_event + 0.5 * _neg(bos_pressure12) + 0.25 * _neg(bos_pressure48))
    choch_event = _clip01(c("snap.smc_choch"))
    choch_recent_contract = _clip01(c("ctx_cont.smc_choch_recent_tau12") + 0.5 * c("ctx_cont.smc_choch_recent_tau24"))
    pullback = _clip01(0.60 * c("ctx_cont.struct_pullback_depth_h1_v3") + 0.40 * c("ctx_cont.struct_pullback_depth_h4_v3"))
    retracement = _clip01(c("ctx_cont.retracement_from_last_impulse"))

    # HH in {0, 1}: unchanged pairing (0.85 clean / 0.45 mixed).
    hh_state = _clip01(0.85 * state_hh_hl + 0.45 * state_hh_ll + 0.50 * high_context * trend_up + 0.35 * bos_up_pressure)
    # Pullback/retracement amplifier normalized by its algebraic max: pullback
    # and retracement are each _clip01-bounded, so (1+pullback+retracement) is
    # in [1,3]; dividing by 3.0 maps it to [1/3,1] so the amplified term can no
    # longer saturate _clip01 exactly in the informative region.
    # HL in {0, 2}: the 0.55 mixed-state weight moves off state 1 (HH+LL) onto
    # state 2 (LH+HL), the state that actually carries a higher low.
    hl_state = _clip01(0.75 * state_hh_hl + 0.55 * state_lh_hl + 0.60 * low_context * trend_up * ((1.0 + pullback + retracement) / 3.0))
    # LH in {1, 3}: the 0.55 mixed-state weight moves off state 2 (LH+HL) onto
    # state 1 (HH+LL), the state that actually carries a lower high.
    lh_state = _clip01(0.55 * state_hh_ll + 0.45 * state_lh_ll + 0.60 * high_context * trend_down * ((1.0 + pullback + retracement) / 3.0))
    # LL in {2, 3}: unchanged pairing (0.85 clean / 0.45 mixed).
    ll_state = _clip01(0.85 * state_lh_ll + 0.45 * state_lh_hl + 0.50 * low_context * trend_down + 0.35 * bos_down_pressure)
    _add(arrays, names, "hh_state", hh_state, lo=0.0, hi=1.0)
    _add(arrays, names, "hl_state", hl_state, lo=0.0, hi=1.0)
    _add(arrays, names, "lh_state", lh_state, lo=0.0, hi=1.0)
    _add(arrays, names, "ll_state", ll_state, lo=0.0, hi=1.0)
    _add(arrays, names, "structure_up_minus_down", (hh_state + hl_state) - (lh_state + ll_state), lo=-2.0, hi=2.0)

    bos_up_age = _bars_since_event(
        bos_up_event > 0.5,
        carry_key="bos_up_age_bars",
    )
    bos_down_age = _bars_since_event(
        bos_down_event > 0.5,
        carry_key="bos_down_age_bars",
    )
    choch_age = _bars_since_event(
        choch_event > 0.5,
        carry_key="choch_age_bars",
    )
    bos_up_recent = np.exp(-bos_up_age / 24.0).astype(np.float32) * (0.5 + 0.5 * bos_up_pressure)
    bos_down_recent = np.exp(-bos_down_age / 24.0).astype(np.float32) * (0.5 + 0.5 * bos_down_pressure)
    choch_recent = _clip01(np.exp(-choch_age / 24.0).astype(np.float32) * (0.5 + 0.5 * choch_recent_contract))
    _add(arrays, names, "bos_up_age_bars", bos_up_age, lo=0.0, hi=96.0)
    _add(arrays, names, "bos_down_age_bars", bos_down_age, lo=0.0, hi=96.0)
    _add(arrays, names, "bos_up_recent_tau24", bos_up_recent, lo=0.0, hi=1.0)
    _add(arrays, names, "bos_down_recent_tau24", bos_down_recent, lo=0.0, hi=1.0)
    _add(arrays, names, "bos_recent_balance", bos_up_recent - bos_down_recent, lo=-1.0, hi=1.0)
    _add(arrays, names, "choch_age_bars", choch_age, lo=0.0, hi=96.0)
    _add(arrays, names, "choch_recent_tau24", choch_recent, lo=0.0, hi=1.0)
    _add(arrays, names, "bars_since_structure_break_min", np.minimum(np.minimum(bos_up_age, bos_down_age), choch_age), lo=0.0, hi=96.0)

    sweep_bull_pressure12 = c("ctx_cont.smc_sweep_bull_pressure_last12")
    sweep_bull_pressure48 = c("ctx_cont.smc_sweep_bull_pressure_last48")
    sweep_up = _clip01(c("snap.smc_sweep_up") + 0.5 * _neg(sweep_bull_pressure12) + 0.25 * _neg(sweep_bull_pressure48))
    sweep_down = _clip01(c("snap.smc_sweep_down") + 0.5 * _pos(sweep_bull_pressure12) + 0.25 * _pos(sweep_bull_pressure48))
    sweep_recent = _clip01(_recency(c("snap.smc_bars_since_sweep")) + c("ctx_cont.smc_sweep_recency_tau24"))
    sweep_size = _clip01(c("snap.smc_sweep_size_atr") + c("ctx_cont.smc_sweep_size_recent_tau12"))
    wick_level = _clip01(1.0 - c("snap.body_pct"))
    support_prox = _clip01(c("ctx_cont.sr_support_proximity_exp"))
    resistance_prox = _clip01(c("ctx_cont.sr_resistance_proximity_exp"))
    sweep_low_reclaim_up = _clip(sweep_down * sweep_recent * (0.50 + sweep_size) * (0.50 + wick_level) * (0.50 + support_prox + low_context + trend_up), 0.0, 5.0)
    sweep_high_reclaim_down = _clip(sweep_up * sweep_recent * (0.50 + sweep_size) * (0.50 + wick_level) * (0.50 + resistance_prox + high_context + trend_down), 0.0, 5.0)
    false_high_follow_down = _clip(sweep_up * resistance_prox * wick_level * (0.50 + trend_down + choch_recent), 0.0, 5.0)
    false_low_follow_up = _clip(sweep_down * support_prox * wick_level * (0.50 + trend_up + choch_recent), 0.0, 5.0)
    _add(arrays, names, "sweep_low_reclaim_up_proxy", sweep_low_reclaim_up, lo=0.0, hi=5.0)
    _add(arrays, names, "sweep_high_reclaim_down_proxy", sweep_high_reclaim_down, lo=0.0, hi=5.0)
    _add(arrays, names, "false_breakout_high_followthrough_down_proxy", false_high_follow_down, lo=0.0, hi=5.0)
    _add(arrays, names, "false_breakout_low_followthrough_up_proxy", false_low_follow_up, lo=0.0, hi=5.0)
    _add(arrays, names, "sweep_reclaim_balance_proxy", sweep_low_reclaim_up - sweep_high_reclaim_down, lo=-5.0, hi=5.0)

    h1_ratio = c("ctx_cont.H1_range_compression_ratio")
    m15_ratio = c("ctx_cont.M15_range_compression_ratio")
    bb_relative_width = c("snap._v1_bb_squeeze_20_2")
    h1_compression = atr_ratio_compression_pressure(h1_ratio)
    m15_compression = atr_ratio_compression_pressure(m15_ratio)
    squeeze = bollinger_squeeze_pressure(bb_relative_width)
    h1_expansion = atr_ratio_expansion_pressure(h1_ratio)
    m15_expansion = atr_ratio_expansion_pressure(m15_ratio)
    bb_expansion = bollinger_expansion_pressure(bb_relative_width)
    atr_z = _pos(_tanh(c("snap.atr_z"), scale=2.0))
    rvol = _pos(_tanh(c("snap.rvol_20"), scale=2.0))
    vol_ratio = _pos(_tanh(c("snap.vol_ratio_5_20"), scale=2.0))
    compression = _clip01(0.45 * h1_compression + 0.35 * m15_compression + 0.20 * squeeze)
    expansion = _clip01(
        0.20 * h1_expansion
        + 0.15 * m15_expansion
        + 0.15 * bb_expansion
        + 0.20 * atr_z
        + 0.15 * rvol
        + 0.15 * vol_ratio
    )
    expansion_delta = _pos(expansion - _lag1(expansion))
    # Honest declared bound: _lag1(compression) and expansion_delta are each in
    # [0,1] (compression/expansion are _clip01-bounded), so the product is
    # provably in [0,1]; the former hi=5.0 declared a domain the value cannot
    # reach. The value itself is unchanged — its smallness is intrinsic.
    release_trigger = _clip(_lag1(compression) * expansion_delta, 0.0, 1.0)
    _add(arrays, names, "compression_state", compression, lo=0.0, hi=1.0)
    _add(arrays, names, "expansion_state", expansion, lo=0.0, hi=1.0)
    _add(arrays, names, "compression_release_trigger", release_trigger, lo=0.0, hi=1.0)
    _add(arrays, names, "compression_release_up", release_trigger * trend_up, lo=0.0, hi=5.0)
    _add(arrays, names, "compression_release_down", release_trigger * trend_down, lo=0.0, hi=5.0)

    trend_delta = _clip(trend - _lag1(trend), lo=-2.0, hi=2.0)
    trend_age_m15 = _clip01(c("ctx_cont.m15_trend_age_bars_norm_v2"))
    trend_age_h1 = _clip01(c("ctx_cont.h1_trend_age_bars_norm_v2"))
    trend_age_h4 = _clip01(c("ctx_cont.h4_trend_age_bars_norm_v2"))
    impulse_direction = _clip(trend + 0.35 * (bos_up_recent - bos_down_recent) + 0.20 * trend_delta, lo=-2.0, hi=2.0)
    impulse_age_proxy = _clip01((trend_age_m15 + trend_age_h1 + trend_age_h4) / 3.0)
    pullback_phase_up = _clip01(trend_up * (0.50 * retracement + 0.50 * pullback) * (1.0 - 0.50 * choch_recent))
    pullback_phase_down = _clip01(trend_down * (0.50 * retracement + 0.50 * pullback) * (1.0 - 0.50 * choch_recent))
    _add(arrays, names, "impulse_direction", impulse_direction, lo=-2.0, hi=2.0)
    _add(arrays, names, "impulse_age_proxy", impulse_age_proxy, lo=0.0, hi=1.0)
    _add(arrays, names, "pullback_phase_up", pullback_phase_up, lo=0.0, hi=1.0)
    _add(arrays, names, "pullback_phase_down", pullback_phase_down, lo=0.0, hi=1.0)
    _add(arrays, names, "pullback_depth_norm", _clip01(0.50 * retracement + 0.50 * pullback), lo=0.0, hi=1.0)
    _add(arrays, names, "impulse_pullback_alignment", impulse_direction * (0.50 * retracement + 0.50 * pullback), lo=-2.0, hi=2.0)

    asia = _clip01(c("ctx_cont.is_ASIA"))
    asia_eu = _clip01(c("ctx_cont.is_asia_eu_overlap"))
    eu_us = _clip01(c("ctx_cont.is_eu_us_overlap"))
    eu = _clip01(c("ctx_cont.is_eu_only") + asia_eu + eu_us)
    us = _clip01(c("ctx_cont.is_us_only") + eu_us)
    overlap = _clip01(asia_eu + eu_us)
    session_groups = (
        ("asia", asia),
        ("eu", eu),
        ("us", us),
        ("overlap", overlap),
    )
    structure_signals = (
        ("hh_state", hh_state),
        ("hl_state", hl_state),
        ("lh_state", lh_state),
        ("ll_state", ll_state),
        ("bos_balance", bos_up_recent - bos_down_recent),
        ("choch_recent", choch_recent),
        ("sweep_reclaim_balance", sweep_low_reclaim_up - sweep_high_reclaim_down),
    )
    for session_name, session_signal in session_groups:
        for signal_name, signal in structure_signals:
            _add(arrays, names, f"{session_name}_x_{signal_name}", session_signal * signal, lo=-5.0, hi=5.0)

    out = np.column_stack(arrays).astype(np.float32, copy=False) if arrays else np.empty((x.shape[0], 0), dtype=np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("foundation structure layer contains non-finite values")
    if len(set(names)) != len(names):
        dupes = sorted({name for name in names if names.count(name) > 1})
        raise RuntimeError(f"foundation structure layer has duplicate names: {dupes[:10]}")
    return out, names


_FOUNDATION_NAME_PROBE = np.zeros(
    (1, len(FOUNDATION_STRUCTURE_SOURCE_FIELDS)), dtype=np.float32
)
for _ratio_field in (
    "ctx_cont.H1_range_compression_ratio",
    "ctx_cont.M15_range_compression_ratio",
):
    _FOUNDATION_NAME_PROBE[0, FOUNDATION_STRUCTURE_SOURCE_FIELDS.index(_ratio_field)] = 1.0
FOUNDATION_STRUCTURE_FEATURE_NAMES = tuple(
    build_entry_foundation_structure_layer(
        _FOUNDATION_NAME_PROBE,
        list(FOUNDATION_STRUCTURE_SOURCE_FIELDS),
    )[1]
)
del _FOUNDATION_NAME_PROBE, _ratio_field
