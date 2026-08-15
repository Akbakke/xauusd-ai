"""One-truth causal local price/quote primitives for Entry and Exit.

The price block runs independently on native closed M5 bars for Entry and
native closed M1 bars for Exit.  Missing history remains unavailable (NaN)
until the exact lag/EMA seed exists; callers trim that single causal prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

from gx1.features.model_native_market_context_v1 import derive_observed_spread_bps
from gx1.features.technical_indicators_v1 import (
    TECHNICAL_INDICATOR_FORMULA_SHA256,
    TECHNICAL_INDICATOR_PRIMITIVES_VERSION,
    classic_ema,
)


MICRO_FEATURE_NAMES_V1 = (
    "close_return_3_bps",
    "close_return_5_bps",
    "close_return_acceleration_1_bps",
    "close_distance_below_high_range_fraction",
    "close_range_observed",
    "close_distance_from_ema5_bps",
)
MICRO_EMA_SPAN_V1 = 5
MICRO_CAUSAL_WARMUP_ROWS_V1 = 5
MICRO_WARMUP_PREFIX_FIELDS_V1 = (
    "close_return_3_bps",
    "close_return_5_bps",
    "close_return_acceleration_1_bps",
    "close_distance_from_ema5_bps",
)
if set(MICRO_WARMUP_PREFIX_FIELDS_V1) - set(MICRO_FEATURE_NAMES_V1):
    raise RuntimeError("MICRO_WARMUP_PREFIX_FIELDS_UNKNOWN")


@dataclass(frozen=True)
class MicroStructureCarryV1:
    """Bounded causal state for exact chunked local-price computation."""

    rows_seen: int = 0
    close_history: tuple[float, ...] = ()
    ema_seed_values: tuple[float, ...] = ()
    ema_value: float | None = None


def _require_micro_carry(carry: MicroStructureCarryV1) -> None:
    if not isinstance(carry, MicroStructureCarryV1):
        raise RuntimeError("MICRO_STRUCTURE_CARRY_TYPE_INVALID")
    if (
        isinstance(carry.rows_seen, bool)
        or not isinstance(carry.rows_seen, int)
        or carry.rows_seen < 0
        or len(carry.close_history) != min(carry.rows_seen, 5)
        or len(carry.ema_seed_values) >= MICRO_EMA_SPAN_V1
    ):
        raise RuntimeError("MICRO_STRUCTURE_CARRY_SHAPE_INVALID")
    history = np.asarray(carry.close_history, dtype=np.float64)
    seed = np.asarray(carry.ema_seed_values, dtype=np.float64)
    if (
        (history.size and (not np.isfinite(history).all() or np.any(history <= 0.0)))
        or (seed.size and (not np.isfinite(seed).all() or np.any(seed <= 0.0)))
        or (
            carry.ema_value is not None
            and (
                not np.isfinite(float(carry.ema_value))
                or float(carry.ema_value) <= 0.0
                or carry.ema_seed_values
                or carry.rows_seen < MICRO_EMA_SPAN_V1
            )
        )
        or (
            carry.ema_value is None
            and len(carry.ema_seed_values) != carry.rows_seen
        )
    ):
        raise RuntimeError("MICRO_STRUCTURE_CARRY_VALUES_INVALID")


def compute_micro_structure_features_chunk(
    high,
    low,
    close,
    *,
    carry: MicroStructureCarryV1 | None = None,
) -> tuple[dict[str, np.ndarray], MicroStructureCarryV1]:
    """Compute one non-empty chunk and return the exact bounded next state.

    All calculations use only the current closed bar, prior closes and the
    prior classic-EMA state.  The EMA recurrence itself is delegated to the
    shared :func:`classic_ema` formula owner; a synthetic constant seed equal
    to the prior EMA value resumes that same recurrence without retaining the
    full history.
    """

    state = MicroStructureCarryV1() if carry is None else carry
    _require_micro_carry(state)

    h = np.asarray(high, dtype=np.float64)
    low_values = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    if h.ndim != 1 or low_values.ndim != 1 or c.ndim != 1:
        raise RuntimeError(
            "MICRO_STRUCTURE_SOURCE_NOT_1D: "
            f"high={h.shape} low={low_values.shape} close={c.shape}"
        )
    if not (len(h) == len(low_values) == len(c)) or len(c) == 0:
        raise RuntimeError(
            "MICRO_STRUCTURE_SOURCE_LENGTH_INVALID: "
            f"high={len(h)} low={len(low_values)} close={len(c)}"
        )
    if (
        not np.isfinite(h).all()
        or not np.isfinite(low_values).all()
        or not np.isfinite(c).all()
    ):
        raise RuntimeError("MICRO_STRUCTURE_SOURCE_NONFINITE")
    if np.any(h <= 0.0) or np.any(low_values <= 0.0) or np.any(c <= 0.0):
        raise RuntimeError("MICRO_STRUCTURE_SOURCE_NONPOSITIVE")
    if np.any(h < low_values) or np.any(h < c) or np.any(low_values > c):
        raise RuntimeError("MICRO_STRUCTURE_SOURCE_GEOMETRY_INVALID")

    history = np.asarray(state.close_history, dtype=np.float64)
    lag_source = np.concatenate((history, c))
    history_rows = len(history)
    return_3 = np.full(len(c), np.nan, dtype=np.float64)
    return_5 = np.full(len(c), np.nan, dtype=np.float64)
    return_acceleration_1 = np.full(len(c), np.nan, dtype=np.float64)
    local_rows = np.arange(len(c), dtype=np.int64)
    global_rows = int(state.rows_seen) + local_rows
    source_rows = history_rows + local_rows
    valid_3 = global_rows >= 3
    valid_5 = global_rows >= 5
    valid_acceleration = global_rows >= 2
    return_3[valid_3] = (
        c[valid_3] / lag_source[source_rows[valid_3] - 3] - 1.0
    ) * 1e4
    return_5[valid_5] = (
        c[valid_5] / lag_source[source_rows[valid_5] - 5] - 1.0
    ) * 1e4
    prior_close = lag_source[source_rows[valid_acceleration] - 1]
    return_acceleration_1[valid_acceleration] = (
        c[valid_acceleration] / prior_close
        - prior_close / lag_source[source_rows[valid_acceleration] - 2]
    ) * 1e4

    bar_range = h - low_values
    close_distance_below_high = np.zeros(len(c), dtype=np.float64)
    positive_range = bar_range > 0.0
    close_distance_below_high[positive_range] = (
        (h[positive_range] - c[positive_range]) / bar_range[positive_range]
    )

    if state.ema_value is None:
        ema_prefix = np.asarray(state.ema_seed_values, dtype=np.float64)
        ema_input = np.concatenate((ema_prefix, c))
        ema_start = len(ema_prefix)
    else:
        ema_prefix = np.full(
            MICRO_EMA_SPAN_V1,
            float(state.ema_value),
            dtype=np.float64,
        )
        ema_input = np.concatenate((ema_prefix, c))
        ema_start = MICRO_EMA_SPAN_V1
    ema_all = classic_ema(pd.Series(ema_input, dtype=np.float64), MICRO_EMA_SPAN_V1)
    ema_chunk = ema_all.to_numpy(dtype=np.float64)[ema_start:]
    distance_from_ema = (c - ema_chunk) / c * 1e4

    result = {
        "close_return_3_bps": return_3.astype(np.float32),
        "close_return_5_bps": return_5.astype(np.float32),
        "close_return_acceleration_1_bps": return_acceleration_1.astype(
            np.float32
        ),
        "close_distance_below_high_range_fraction": close_distance_below_high.astype(
            np.float32
        ),
        "close_range_observed": positive_range.astype(np.float32),
        "close_distance_from_ema5_bps": distance_from_ema.astype(np.float32),
    }
    if tuple(result) != MICRO_FEATURE_NAMES_V1 or any(
        values.shape != c.shape or np.isinf(values).any()
        for values in result.values()
    ):
        raise RuntimeError("MICRO_STRUCTURE_OUTPUT_INVALID")
    if (
        not np.array_equal(
            result["close_range_observed"],
            positive_range.astype(np.float32),
        )
        or np.any(
            result["close_distance_below_high_range_fraction"]
            [~positive_range]
            != 0.0
        )
        or np.any(
            result["close_distance_below_high_range_fraction"]
            [positive_range]
            < 0.0
        )
        or np.any(
            result["close_distance_below_high_range_fraction"]
            [positive_range]
            > 1.0
        )
    ):
        raise RuntimeError("MICRO_STRUCTURE_RANGE_PRESENCE_INVALID")

    finite_ema = ema_chunk[np.isfinite(ema_chunk)]
    if finite_ema.size:
        next_ema_value: float | None = float(finite_ema[-1])
        next_ema_seed: tuple[float, ...] = ()
    else:
        next_ema_value = None
        next_ema_seed = tuple(float(value) for value in ema_input)
    next_history_values = lag_source[-5:]
    next_state = MicroStructureCarryV1(
        rows_seen=state.rows_seen + len(c),
        close_history=tuple(float(value) for value in next_history_values),
        ema_seed_values=next_ema_seed,
        ema_value=next_ema_value,
    )
    _require_micro_carry(next_state)
    return result, next_state

# V30 package 4 (2026-08-13) — quote/spread dynamics.
#
# PURPOSE: abstention and execution-regime evidence.  These three fields say
# how expensive and how disorderly the quote is at the decision bar; they are
# NOT a direction signal and must never be read as one: orthogonal
# microstructure sources were OOS-refuted for DIRECTION on the retired chain,
# and that fence stands.  The model may learn to abstain or to size down when
# execution conditions are hostile; no post-model rule may consume them
# (rule 3: argmax stays the only decision authority).
#
# Why they exist at all: the ctx surface carried exactly ONE spread field
# (``spread_bps``, a level) and no quote-dynamics evidence, while the legacy
# The retired canonical spread z-score died constant. Measured on
# the complete declared native M5 tape 2026-08-13 (N=537,861 rows): spread_bps
# mean 2.23 / std 2.61 / p1 1.11 / p99 14.66 with ~3261 distinct rounded
# values; the 1-bar change is nonzero on 99.93% of rows with std 1.161; the
# intrabar quote envelope (ask_high - bid_low) is 10.21 bps mean / 8.64 std;
# the quote-range asymmetry is nonzero on 87.85% of rows with std 0.969 bps.
# Three fields, no thresholds, no clips, no new magnitudes.
SPREAD_DYNAMICS_FEATURE_NAMES_V1 = (
    "spread_bps_delta_1",
    "spread_intrabar_range_bps",
    "quote_range_asymmetry_bps",
)
# Exact source columns.  All seven are members of
# ``xau_tape_provenance_v1.CANONICAL_NATIVE_REQUIRED_COLUMNS`` (plus the mid
# ``close``), so every offline ctx producer's frame carries them by contract.
SPREAD_DYNAMICS_SOURCE_COLUMNS_V1 = (
    "close",
    "bid_close",
    "ask_close",
    "bid_high",
    "bid_low",
    "ask_high",
    "ask_low",
)
# ``spread_bps_delta_1`` needs one preceding row and has no value on the first
# row of any frame.  That row is an honest NaN, never a parked zero that would
# read as "the spread did not move" (rule 2e); the ctx warmup-trim contract
# removes it, exactly as it removes the swing pivot-delta prefixes.
SPREAD_DYNAMICS_CAUSAL_WARMUP_ROWS_V1 = 1
# Exactly the emitted fields that carry that NaN prefix.  The ctx producers
# join this tuple into their causal warmup-trim lists; the other two fields are
# defined on every row and must NOT be listed (a field with no prefix in a trim
# list is a silent no-op today and a wrong trim after any later change).
SPREAD_DYNAMICS_WARMUP_PREFIX_FIELDS_V1 = ("spread_bps_delta_1",)
if set(SPREAD_DYNAMICS_WARMUP_PREFIX_FIELDS_V1) - set(
    SPREAD_DYNAMICS_FEATURE_NAMES_V1
):
    raise RuntimeError("SPREAD_DYNAMICS_WARMUP_PREFIX_FIELDS_UNKNOWN")


@dataclass(frozen=True)
class SpreadDynamicsCarryV1:
    """One-row quote state required by the spread-change primitive."""

    rows_seen: int = 0
    previous_spread_bps: float | None = None


MICRO_STRUCTURE_SCHEMA_VERSION = (
    "micro_structure_primitives_v4_lag_return_range_presence_20260814"
)
MICRO_PRICE_FORMULA_CONTRACT = (
    "close_return_3_bps[t]=(close[t]/close[t-3]-1)*10000",
    "close_return_5_bps[t]=(close[t]/close[t-5]-1)*10000",
    "close_return_acceleration_1_bps[t]=((close[t]/close[t-1]-1)-(close[t-1]/close[t-2]-1))*10000",
    "close_distance_below_high_range_fraction[t]=(high[t]-close[t])/(high[t]-low[t])_when_range_positive_else_storage_zero_with_range_observed_zero",
    "close_range_observed[t]=1_iff_high[t]>low[t]_else_0",
    "close_distance_from_ema5_bps[t]=(close[t]-classic_sma_seeded_ema5[t])/close[t]*10000",
    "warmup=causal_nan_prefix_no_partial_window_no_zero_or_midpoint_fill",
    "transform=no_epsilon_no_clip_no_neutral_midpoint_fill",
    "clock=same_owner_independent_native_closed_m5_entry_and_m1_exit",
)
SPREAD_DYNAMICS_FORMULA_CONTRACT = (
    "spread_bps[t]=(ask_close[t]-bid_close[t])/bid_close[t]*10000",
    "spread_bps_delta_1[t]=spread_bps[t]-spread_bps[t-1]",
    "spread_intrabar_range_bps[t]=(ask_high[t]-bid_low[t])/close[t]*10000",
    "quote_range_asymmetry_bps[t]=((ask_high[t]-ask_low[t])-(bid_high[t]-bid_low[t]))/close[t]*10000",
    "warmup=causal_nan_prefix_for_spread_delta_only",
    "transform=no_epsilon_no_clip_no_static_boundary_value",
)
MICRO_CARRY_CONTRACT = (
    "price_carry=rows_seen_last_five_closes_ema_seed_or_last_classic_ema",
    "price_chunk_output=exact_batch_output_for_every_partition",
    "spread_carry=rows_seen_previous_raw_spread_bps",
    "spread_chunk_output=exact_batch_output_for_every_partition",
    "carry=bounded_past_only_no_future_or_full_history_retention",
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def micro_structure_contract_metadata() -> dict[str, object]:
    """Return the complete immutable price/quote formula identity."""

    price_names = list(MICRO_FEATURE_NAMES_V1)
    spread_names = list(SPREAD_DYNAMICS_FEATURE_NAMES_V1)
    formula_contract = {
        "price": list(MICRO_PRICE_FORMULA_CONTRACT),
        "spread": list(SPREAD_DYNAMICS_FORMULA_CONTRACT),
    }
    carry_contract = list(MICRO_CARRY_CONTRACT)
    payload: dict[str, object] = {
        "schema_version": MICRO_STRUCTURE_SCHEMA_VERSION,
        "price_feature_names": price_names,
        "price_feature_names_sha256": _canonical_sha256(price_names),
        "spread_feature_names": spread_names,
        "spread_feature_names_sha256": _canonical_sha256(spread_names),
        "ordered_feature_names_sha256": _canonical_sha256(
            price_names + spread_names
        ),
        "price_warmup_rows_until_all_fields_finite": (
            MICRO_CAUSAL_WARMUP_ROWS_V1
        ),
        "price_field_nan_prefix_rows": {
            "close_return_3_bps": 3,
            "close_return_5_bps": 5,
            "close_return_acceleration_1_bps": 2,
            "close_distance_below_high_range_fraction": 0,
            "close_range_observed": 0,
            "close_distance_from_ema5_bps": 4,
        },
        "price_warmup_prefix_fields": list(MICRO_WARMUP_PREFIX_FIELDS_V1),
        "spread_field_nan_prefix_rows": {
            "spread_bps_delta_1": SPREAD_DYNAMICS_CAUSAL_WARMUP_ROWS_V1,
            "spread_intrabar_range_bps": 0,
            "quote_range_asymmetry_bps": 0,
        },
        "spread_warmup_prefix_fields": list(
            SPREAD_DYNAMICS_WARMUP_PREFIX_FIELDS_V1
        ),
        "classic_ema": {
            "span": MICRO_EMA_SPAN_V1,
            "owner_schema_version": TECHNICAL_INDICATOR_PRIMITIVES_VERSION,
            "owner_formula_sha256": TECHNICAL_INDICATOR_FORMULA_SHA256,
            "seed": "arithmetic_mean_of_first_exact_span_observations",
        },
        "formula_contract": formula_contract,
        "formula_sha256": _canonical_sha256(formula_contract),
        "carry_contract": carry_contract,
        "carry_contract_sha256": _canonical_sha256(carry_contract),
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    return payload


def require_micro_structure_contract_metadata(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Require byte-semantic equality with the active micro owner contract."""

    if not isinstance(value, Mapping):
        raise RuntimeError("MICRO_STRUCTURE_CONTRACT_MISSING")
    observed = dict(value)
    expected = micro_structure_contract_metadata()
    if observed != expected:
        raise RuntimeError("MICRO_STRUCTURE_CONTRACT_MISMATCH")
    return observed


def compute_micro_structure_features(
    high,
    low,
    close,
) -> dict[str, np.ndarray]:
    """Return the exact causal six-field local-price surface."""

    result, _ = compute_micro_structure_features_chunk(high, low, close)
    return result


def compute_spread_dynamics_features_chunk(
    frame: pd.DataFrame,
    *,
    carry: SpreadDynamicsCarryV1 | None = None,
) -> tuple[dict[str, np.ndarray], SpreadDynamicsCarryV1]:
    """Return one exact causal quote/spread-dynamics chunk and next state.

    Causality: every input is an aggregate of the decision bar's OWN closed
    quotes.  ``bid_close``/``ask_close`` are the same two columns the ctx
    ``spread_bps`` owner reads (``entry_model_native_state_v2
    .compute_causal_market_rank_inputs`` builds ``frame[["bid_close",
    "ask_close"]]`` and hands it to ``derive_observed_spread_bps``); the four
    extremes are that bar's own high/low of the bid and ask series.  This is
    the identical closed-bar convention every other current-bar ctx field uses,
    so the block exposes nothing at ``t`` that the surface does not already
    expose at ``t``.  ``spread_bps_delta_1`` additionally reads row ``t-1``
    only.  No forward row participates anywhere.

    Units: bps of the current mid close, this file's own ``/ close * 1e4``
    convention (``close_return_*``/``close_return_acceleration_1_bps``/
    ``close_distance_from_ema5_bps``) and
    ``derive_observed_spread_bps``'s ``* 1e4``.
    No threshold, no clip and no new magnitude is introduced: this owner clips
    nothing, and neither does this block.  ``quote_range_asymmetry_bps`` stays
    SIGNED (an ask range wider than the bid range is different evidence from
    the reverse).

    One spread owner: the level is NOT recomputed here.  It is produced by
    ``derive_observed_spread_bps`` from the same two columns and the same
    ``(ask - bid) / bid * 1e4`` formula that puts ``spread_bps`` on the ctx
    surface, so ``spread_bps_delta_1[t] == spread_bps[t] - spread_bps[t-1]``
    holds exactly on the emitted ctx column.
    """

    state = SpreadDynamicsCarryV1() if carry is None else carry
    if not isinstance(state, SpreadDynamicsCarryV1):
        raise RuntimeError("SPREAD_DYNAMICS_CARRY_TYPE_INVALID")
    if (
        isinstance(state.rows_seen, bool)
        or not isinstance(state.rows_seen, int)
        or state.rows_seen < 0
        or (state.rows_seen == 0) != (state.previous_spread_bps is None)
        or (
            state.previous_spread_bps is not None
            and not np.isfinite(float(state.previous_spread_bps))
        )
    ):
        raise RuntimeError("SPREAD_DYNAMICS_CARRY_INVALID")
    if not isinstance(frame, pd.DataFrame):
        raise RuntimeError(
            f"SPREAD_DYNAMICS_SOURCE_FRAME_INVALID: {type(frame).__name__}"
        )
    if len(frame) == 0:
        raise RuntimeError("SPREAD_DYNAMICS_SOURCE_ROWS_EMPTY")
    missing = [
        name
        for name in SPREAD_DYNAMICS_SOURCE_COLUMNS_V1
        if name not in frame.columns
    ]
    if missing:
        raise RuntimeError(f"SPREAD_DYNAMICS_SOURCE_FIELDS_MISSING: {missing}")

    values: dict[str, np.ndarray] = {}
    for name in SPREAD_DYNAMICS_SOURCE_COLUMNS_V1:
        try:
            column = pd.to_numeric(frame[name], errors="raise").to_numpy(
                dtype=np.float64
            )
        except Exception as exc:
            raise RuntimeError(
                f"SPREAD_DYNAMICS_SOURCE_COLUMN_INVALID: {name}"
            ) from exc
        if column.ndim != 1:
            raise RuntimeError(
                f"SPREAD_DYNAMICS_SOURCE_NOT_1D: {name} shape={column.shape}"
            )
        if not np.isfinite(column).all():
            raise RuntimeError(f"SPREAD_DYNAMICS_SOURCE_NONFINITE: {name}")
        if np.any(column <= 0.0):
            raise RuntimeError(f"SPREAD_DYNAMICS_SOURCE_NONPOSITIVE: {name}")
        values[name] = column

    close = values["close"]
    bid_high = values["bid_high"]
    bid_low = values["bid_low"]
    ask_high = values["ask_high"]
    ask_low = values["ask_low"]
    # A quote bar is valid only when each side's own high dominates its low and
    # the ask series dominates the bid series.  Both are data-integrity facts of
    # a two-sided quote, not chosen tolerances; a violation is a broken tape and
    # fails closed.  Together they also prove ask_high >= bid_low, so the
    # intrabar envelope below is non-negative by construction.
    if (
        np.any(bid_high < bid_low)
        or np.any(ask_high < ask_low)
        or np.any(ask_high < bid_high)
        or np.any(ask_low < bid_low)
    ):
        raise RuntimeError("SPREAD_DYNAMICS_SOURCE_QUOTE_GEOMETRY_INVALID")

    spread_bps = np.asarray(
        derive_observed_spread_bps(frame[["bid_close", "ask_close"]].copy()),
        dtype=np.float64,
    )
    if spread_bps.shape != close.shape:
        raise RuntimeError(
            f"SPREAD_DYNAMICS_SPREAD_SHAPE_INVALID: {spread_bps.shape}"
        )
    spread_delta_1 = np.empty_like(spread_bps)
    spread_delta_1[0] = (
        np.nan
        if state.previous_spread_bps is None
        else spread_bps[0] - float(state.previous_spread_bps)
    )
    spread_delta_1[1:] = spread_bps[1:] - spread_bps[:-1]

    intrabar_range_bps = (ask_high - bid_low) / close * 1e4
    quote_range_asymmetry_bps = (
        (ask_high - ask_low) - (bid_high - bid_low)
    ) / close * 1e4

    result = {
        "spread_bps_delta_1": spread_delta_1.astype(np.float32),
        "spread_intrabar_range_bps": intrabar_range_bps.astype(np.float32),
        "quote_range_asymmetry_bps": quote_range_asymmetry_bps.astype(np.float32),
    }
    if tuple(result) != SPREAD_DYNAMICS_FEATURE_NAMES_V1 or any(
        array.shape != close.shape for array in result.values()
    ):
        raise RuntimeError("SPREAD_DYNAMICS_OUTPUT_INVALID")
    delta = result["spread_bps_delta_1"]
    if state.rows_seen == 0 and not np.isnan(delta[0]):
        raise RuntimeError("SPREAD_DYNAMICS_WARMUP_PREFIX_INVALID")
    if state.rows_seen > 0 and not np.isfinite(delta[0]):
        raise RuntimeError("SPREAD_DYNAMICS_CARRY_OUTPUT_INVALID")
    if not np.isfinite(delta[1:]).all():
        raise RuntimeError("SPREAD_DYNAMICS_WARMUP_PREFIX_INVALID")
    for name in ("spread_intrabar_range_bps", "quote_range_asymmetry_bps"):
        if not np.isfinite(result[name]).all():
            raise RuntimeError(f"SPREAD_DYNAMICS_OUTPUT_NONFINITE: {name}")
    if np.any(result["spread_intrabar_range_bps"] < 0.0):
        raise RuntimeError("SPREAD_DYNAMICS_INTRABAR_RANGE_NEGATIVE")
    next_state = SpreadDynamicsCarryV1(
        rows_seen=state.rows_seen + len(frame),
        previous_spread_bps=float(spread_bps[-1]),
    )
    return result, next_state


def compute_spread_dynamics_features(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return the batch quote/spread surface through the chunk owner."""

    result, _ = compute_spread_dynamics_features_chunk(frame)
    return result
