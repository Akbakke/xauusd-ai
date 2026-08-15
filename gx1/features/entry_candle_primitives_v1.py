"""Raw causal candle geometry shared by local Entry/Exit and every MTF lane.

The owner deliberately emits no named candlestick pattern and contains no
market threshold, vote or direction decision.  A temporal model can learn
doji, rejection, engulfing, inside/outside and multi-bar patterns from these
continuous measurements without inheriting a hand-written definition.

Row ``t`` describes the bar that closes at that row's authoritative close.
Relational fields need row ``t-1`` and therefore preserve one honest NaN
prefix.  A zero-range bar is explicitly identified; its otherwise undefined
close location uses the neutral encoding 0.5 only together with that flag.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


CANDLE_PRIMITIVE_FEATURE_VERSION = (
    "entry_candle_primitives_v2_20260814_exact_relations_and_carry"
)
CANDLE_PRIMITIVE_FEATURE_PREFIX = "candle.raw_"
CANDLE_PRIMITIVE_SOURCE_FIELDS = ("time", "open", "high", "low", "close")
CANDLE_PRIMITIVE_FEATURE_NAMES = (
    "candle.raw_body_signed_range",
    "candle.raw_upper_wick_share",
    "candle.raw_lower_wick_share",
    "candle.raw_close_location",
    "candle.raw_zero_range_flag",
    # These six use the exact two-bar geometry envelope (ranges and absolute
    # OHLC/body displacements).  The retired *_prev_range spelling falsely
    # claimed an exact previous-range denominator, including after a
    # zero-range bar.
    "candle.raw_open_gap_local_geometry",
    "candle.raw_close_change_local_geometry",
    "candle.raw_high_change_local_geometry",
    "candle.raw_low_change_local_geometry",
    "candle.raw_range_change_local_geometry",
    "candle.raw_body_change_local_geometry",
    # Exact two-bar geometry. Magnitudes are zero off-event and accompanied
    # by either an algebraically readable sign or an explicit flag.
    "candle.raw_open_above_previous_high_local_geometry",
    "candle.raw_open_below_previous_low_local_geometry",
    "candle.raw_body_overlap_previous_local_geometry",
    "candle.raw_body_contains_previous_flag",
    "candle.raw_body_contained_by_previous_flag",
    "candle.raw_range_contains_previous_flag",
    "candle.raw_range_contained_by_previous_flag",
    "candle.raw_bull_body_covers_previous_bear_body_event",
    "candle.raw_bear_body_covers_previous_bull_body_event",
    "candle.raw_high_rejection_previous_high_event",
    "candle.raw_low_rejection_previous_low_event",
    "candle.raw_high_rejection_depth_local_geometry",
    "candle.raw_low_rejection_depth_local_geometry",
    # Raw observed-bar duration of the current exact state.  No cap, tanh or
    # threshold is applied; TRAIN-only input normalization owns scale.
    "candle.raw_observed_body_direction_duration_bars",
    "candle.raw_observed_range_relation_duration_bars",
)
CANDLE_PRIMITIVE_RELATIONAL_FEATURE_NAMES = CANDLE_PRIMITIVE_FEATURE_NAMES[5:]
CANDLE_PRIMITIVE_FEATURE_NAMES_SHA256 = hashlib.sha256(
    "\n".join(CANDLE_PRIMITIVE_FEATURE_NAMES).encode("utf-8")
).hexdigest()


def candle_primitive_contract_metadata() -> dict[str, object]:
    """Return the immutable identity embedded in local signal artifacts."""

    return {
        "owner": "gx1.features.entry_candle_primitives_v1",
        "feature_version": CANDLE_PRIMITIVE_FEATURE_VERSION,
        "feature_count": len(CANDLE_PRIMITIVE_FEATURE_NAMES),
        "ordered_feature_names_sha256": CANDLE_PRIMITIVE_FEATURE_NAMES_SHA256,
        "ordered_feature_names": list(CANDLE_PRIMITIVE_FEATURE_NAMES),
    }


@dataclass(frozen=True)
class CandlePrimitiveCarryState:
    """Exact continuation state for native-clock chunk evaluation."""

    feature_version: str = CANDLE_PRIMITIVE_FEATURE_VERSION
    previous_open: float | None = None
    previous_high: float | None = None
    previous_low: float | None = None
    previous_close: float | None = None
    previous_body_direction: int | None = None
    body_direction_duration_bars: int = 0
    previous_range_relation: int | None = None
    range_relation_duration_bars: int = 0
    rows_seen: int = 0
    last_timestamp_ns: int | None = None


def missing_candle_primitive_source_fields(columns: object) -> list[str]:
    available = set(columns)
    return [name for name in CANDLE_PRIMITIVE_SOURCE_FIELDS if name not in available]


def _time_index(frame: object) -> pd.DatetimeIndex:
    try:
        index = pd.DatetimeIndex(frame["time"])  # type: ignore[index]
    except Exception as exc:
        raise RuntimeError("CANDLE_PRIMITIVE_TIME_INVALID") from exc
    if index.empty or index.hasnans:
        raise RuntimeError("CANDLE_PRIMITIVE_TIME_INVALID")
    if index.tz is None or str(index.tz) != "UTC":
        raise RuntimeError("CANDLE_PRIMITIVE_TIME_NOT_UTC")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise RuntimeError("CANDLE_PRIMITIVE_TIME_NOT_STRICT")
    return index


def _numeric_column(frame: object, name: str) -> np.ndarray:
    try:
        raw = frame[name]  # type: ignore[index]
    except Exception as exc:
        raise RuntimeError(f"CANDLE_PRIMITIVE_SOURCE_MISSING: {name}") from exc
    values = (
        raw.to_numpy(dtype=np.float64)
        if hasattr(raw, "to_numpy")
        else np.asarray(raw, dtype=np.float64)
    )
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError(f"CANDLE_PRIMITIVE_SOURCE_INVALID: {name}")
    return values


def compute_entry_candle_primitive_chunk(
    frame: object,
    *,
    carry: CandlePrimitiveCarryState | None = None,
) -> tuple[np.ndarray, list[str], CandlePrimitiveCarryState]:
    columns = getattr(frame, "columns", None)
    if columns is None:
        try:
            columns = frame.keys()  # type: ignore[union-attr]
        except Exception as exc:
            raise RuntimeError("CANDLE_PRIMITIVE_SOURCE_COLUMNS_INVALID") from exc
    missing = missing_candle_primitive_source_fields(columns)
    if missing:
        raise RuntimeError(f"CANDLE_PRIMITIVE_SOURCE_FIELDS_MISSING: {missing}")

    times = _time_index(frame)
    open_ = _numeric_column(frame, "open")
    high = _numeric_column(frame, "high")
    low = _numeric_column(frame, "low")
    close = _numeric_column(frame, "close")
    n_rows = len(close)
    if not (len(times) == len(open_) == len(high) == len(low) == n_rows):
        raise RuntimeError("CANDLE_PRIMITIVE_SOURCE_LENGTH_MISMATCH")
    if np.any(open_ <= 0.0) or np.any(high <= 0.0) or np.any(low <= 0.0) or np.any(close <= 0.0):
        raise RuntimeError("CANDLE_PRIMITIVE_SOURCE_PRICE_NONPOSITIVE")
    invalid_geometry = (
        (high < low)
        | (high < open_)
        | (high < close)
        | (low > open_)
        | (low > close)
    )
    if invalid_geometry.any():
        raise RuntimeError(
            "CANDLE_PRIMITIVE_SOURCE_GEOMETRY_INVALID: "
            f"row={int(np.flatnonzero(invalid_geometry)[0])}"
        )

    bar_range = high - low
    zero_range = bar_range == 0.0
    positive_range = ~zero_range
    body_signed = close - open_
    body = np.abs(body_signed)
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low

    body_signed_range = np.zeros(n_rows, dtype=np.float64)
    upper_wick_share = np.zeros(n_rows, dtype=np.float64)
    lower_wick_share = np.zeros(n_rows, dtype=np.float64)
    close_location = np.full(n_rows, 0.5, dtype=np.float64)
    body_signed_range[positive_range] = (
        body_signed[positive_range] / bar_range[positive_range]
    )
    upper_wick_share[positive_range] = (
        upper_wick[positive_range] / bar_range[positive_range]
    )
    lower_wick_share[positive_range] = (
        lower_wick[positive_range] / bar_range[positive_range]
    )
    close_location[positive_range] = (
        (close[positive_range] - low[positive_range])
        / bar_range[positive_range]
    )

    if carry is None:
        state = CandlePrimitiveCarryState()
    else:
        if (
            not isinstance(carry, CandlePrimitiveCarryState)
            or carry.feature_version != CANDLE_PRIMITIVE_FEATURE_VERSION
            or carry.rows_seen < 0
            or carry.body_direction_duration_bars < 0
            or carry.range_relation_duration_bars < 0
            or carry.previous_body_direction not in {None, -1, 0, 1}
            or carry.previous_range_relation not in {None, -1, 0, 1, 2}
            or any(
                value is not None and (not np.isfinite(value) or value <= 0.0)
                for value in (
                    carry.previous_open,
                    carry.previous_high,
                    carry.previous_low,
                    carry.previous_close,
                )
            )
        ):
            raise RuntimeError("CANDLE_PRIMITIVE_CARRY_INVALID")
        previous_values = (
            carry.previous_open,
            carry.previous_high,
            carry.previous_low,
            carry.previous_close,
        )
        if any(value is None for value in previous_values) != all(
            value is None for value in previous_values
        ):
            raise RuntimeError("CANDLE_PRIMITIVE_CARRY_INVALID")
        is_empty_state = carry.rows_seen == 0
        if is_empty_state != all(value is None for value in previous_values):
            raise RuntimeError("CANDLE_PRIMITIVE_CARRY_INVALID")
        if is_empty_state:
            if (
                carry.last_timestamp_ns is not None
                or carry.previous_body_direction is not None
                or carry.body_direction_duration_bars != 0
                or carry.previous_range_relation is not None
                or carry.range_relation_duration_bars != 0
            ):
                raise RuntimeError("CANDLE_PRIMITIVE_CARRY_INVALID")
        else:
            previous_open, previous_high, previous_low, previous_close = (
                float(value) for value in previous_values
            )
            if (
                carry.last_timestamp_ns is None
                or previous_high < previous_low
                or previous_high < max(previous_open, previous_close)
                or previous_low > min(previous_open, previous_close)
                or carry.previous_body_direction
                != int(np.sign(previous_close - previous_open))
                or carry.body_direction_duration_bars < 1
                or (carry.rows_seen == 1)
                != (carry.previous_range_relation is None)
                or (carry.rows_seen == 1)
                != (carry.range_relation_duration_bars == 0)
                or (
                    carry.rows_seen > 1
                    and carry.range_relation_duration_bars < 1
                )
            ):
                raise RuntimeError("CANDLE_PRIMITIVE_CARRY_INVALID")
        if (
            carry.last_timestamp_ns is not None
            and int(times[0].value) <= carry.last_timestamp_ns
        ):
            raise RuntimeError("CANDLE_PRIMITIVE_CARRY_TIME_OVERLAP")
        state = carry

    matrix = np.full(
        (n_rows, len(CANDLE_PRIMITIVE_FEATURE_NAMES)),
        np.nan,
        dtype=np.float64,
    )
    matrix[:, 0] = body_signed_range
    matrix[:, 1] = upper_wick_share
    matrix[:, 2] = lower_wick_share
    matrix[:, 3] = close_location
    matrix[:, 4] = zero_range.astype(np.float64)

    previous_open_value = state.previous_open
    previous_high_value = state.previous_high
    previous_low_value = state.previous_low
    previous_close_value = state.previous_close
    previous_body_direction = state.previous_body_direction
    body_direction_duration = int(state.body_direction_duration_bars)
    previous_range_relation = state.previous_range_relation
    range_relation_duration = int(state.range_relation_duration_bars)

    for row in range(n_rows):
        body_direction = int(np.sign(body_signed[row]))
        if previous_body_direction == body_direction:
            body_direction_duration += 1
        else:
            body_direction_duration = 1
        matrix[row, 24] = float(body_direction_duration)

        if previous_close_value is not None:
            previous_range = float(previous_high_value - previous_low_value)
            previous_body = abs(previous_close_value - previous_open_value)
            price_scale = max(
                abs(open_[row]),
                abs(high[row]),
                abs(low[row]),
                abs(close[row]),
                abs(previous_open_value),
                abs(previous_high_value),
                abs(previous_low_value),
                abs(previous_close_value),
                1.0,
            )
            local_geometry_scale = max(
                float(bar_range[row]),
                previous_range,
                abs(open_[row] - previous_close_value),
                abs(close[row] - previous_close_value),
                abs(high[row] - previous_high_value),
                abs(low[row] - previous_low_value),
                abs(float(bar_range[row]) - previous_range),
                abs(float(body[row]) - previous_body),
                price_scale * np.finfo(np.float64).eps,
            )
            matrix[row, 5] = (open_[row] - previous_close_value) / local_geometry_scale
            matrix[row, 6] = (close[row] - previous_close_value) / local_geometry_scale
            matrix[row, 7] = (high[row] - previous_high_value) / local_geometry_scale
            matrix[row, 8] = (low[row] - previous_low_value) / local_geometry_scale
            matrix[row, 9] = (bar_range[row] - previous_range) / local_geometry_scale
            matrix[row, 10] = (body[row] - previous_body) / local_geometry_scale
            matrix[row, 11] = max(open_[row] - previous_high_value, 0.0) / local_geometry_scale
            matrix[row, 12] = max(previous_low_value - open_[row], 0.0) / local_geometry_scale

            body_low = min(open_[row], close[row])
            body_high = max(open_[row], close[row])
            previous_body_low = min(previous_open_value, previous_close_value)
            previous_body_high = max(previous_open_value, previous_close_value)
            body_overlap = max(
                min(body_high, previous_body_high)
                - max(body_low, previous_body_low),
                0.0,
            )
            matrix[row, 13] = body_overlap / local_geometry_scale
            body_equal = (
                body_low == previous_body_low
                and body_high == previous_body_high
            )
            body_contains = (
                body_low <= previous_body_low
                and body_high >= previous_body_high
                and not body_equal
            )
            body_contained = (
                body_low >= previous_body_low
                and body_high <= previous_body_high
                and not body_equal
            )
            matrix[row, 14] = float(body_contains)
            matrix[row, 15] = float(body_contained)

            range_equal = (
                high[row] == previous_high_value
                and low[row] == previous_low_value
            )
            range_contains = (
                high[row] >= previous_high_value
                and low[row] <= previous_low_value
                and not range_equal
            )
            range_contained = (
                high[row] <= previous_high_value
                and low[row] >= previous_low_value
                and not range_equal
            )
            matrix[row, 16] = float(range_contains)
            matrix[row, 17] = float(range_contained)

            previous_direction = int(previous_close_value > previous_open_value) - int(
                previous_close_value < previous_open_value
            )
            bull_cover = body_contains and body_direction == 1 and previous_direction == -1
            bear_cover = body_contains and body_direction == -1 and previous_direction == 1
            matrix[row, 18] = float(bull_cover)
            matrix[row, 19] = float(bear_cover)

            high_rejection = high[row] > previous_high_value and close[row] <= previous_high_value
            low_rejection = low[row] < previous_low_value and close[row] >= previous_low_value
            matrix[row, 20] = float(high_rejection)
            matrix[row, 21] = float(low_rejection)
            matrix[row, 22] = (
                (high[row] - previous_high_value) / local_geometry_scale
                if high_rejection
                else 0.0
            )
            matrix[row, 23] = (
                (previous_low_value - low[row]) / local_geometry_scale
                if low_rejection
                else 0.0
            )

            range_relation = (
                2
                if range_equal
                else 1
                if range_contains
                else -1
                if range_contained
                else 0
            )
            if previous_range_relation == range_relation:
                range_relation_duration += 1
            else:
                range_relation_duration = 1
            matrix[row, 25] = float(range_relation_duration)
            previous_range_relation = range_relation

        previous_open_value = float(open_[row])
        previous_high_value = float(high[row])
        previous_low_value = float(low[row])
        previous_close_value = float(close[row])
        previous_body_direction = body_direction

    matrix = matrix.astype(np.float32, copy=False)
    if matrix.shape != (n_rows, len(CANDLE_PRIMITIVE_FEATURE_NAMES)):
        raise RuntimeError("CANDLE_PRIMITIVE_OUTPUT_SHAPE_INVALID")
    if np.isinf(matrix).any():
        raise RuntimeError("CANDLE_PRIMITIVE_OUTPUT_INFINITY")
    complete = np.isfinite(matrix).all(axis=1)
    expected_first_complete = 0 if state.previous_close is not None else 1
    if complete.any():
        first_complete = int(np.argmax(complete))
        if first_complete != expected_first_complete or not complete[first_complete:].all():
            raise RuntimeError("CANDLE_PRIMITIVE_OUTPUT_AVAILABILITY_INVALID")
    elif n_rows > expected_first_complete:
        raise RuntimeError("CANDLE_PRIMITIVE_OUTPUT_UNAVAILABLE")
    next_carry = CandlePrimitiveCarryState(
        previous_open=previous_open_value,
        previous_high=previous_high_value,
        previous_low=previous_low_value,
        previous_close=previous_close_value,
        previous_body_direction=previous_body_direction,
        body_direction_duration_bars=body_direction_duration,
        previous_range_relation=previous_range_relation,
        range_relation_duration_bars=range_relation_duration,
        rows_seen=state.rows_seen + n_rows,
        last_timestamp_ns=int(times[-1].value),
    )
    return matrix, list(CANDLE_PRIMITIVE_FEATURE_NAMES), next_carry


def build_entry_candle_primitive_layer(
    frame: object,
) -> tuple[np.ndarray, list[str]]:
    """Batch compatibility route through the exact chunk/carry owner."""

    matrix, names, _carry = compute_entry_candle_primitive_chunk(frame)
    return matrix, names


__all__ = [
    "CANDLE_PRIMITIVE_FEATURE_NAMES",
    "CANDLE_PRIMITIVE_FEATURE_NAMES_SHA256",
    "CANDLE_PRIMITIVE_FEATURE_PREFIX",
    "CANDLE_PRIMITIVE_FEATURE_VERSION",
    "CANDLE_PRIMITIVE_RELATIONAL_FEATURE_NAMES",
    "CANDLE_PRIMITIVE_SOURCE_FIELDS",
    "CandlePrimitiveCarryState",
    "build_entry_candle_primitive_layer",
    "candle_primitive_contract_metadata",
    "compute_entry_candle_primitive_chunk",
    "missing_candle_primitive_source_fields",
]
