"""One causal owner for raw event ages and discrete-state durations.

Both quantities are measured in observed rows on the caller's native clock.
They are never capped, log transformed, zero-filled before an event, or
inferred from elapsed wall time.  Callers that project a slower clock onto a
faster decision clock must first reduce the input to genuine new observations
of that slower clock.
"""
from __future__ import annotations

import hashlib

import numpy as np


RAW_AGE_OWNER_SCHEMA_VERSION = "gx1_raw_event_and_state_age_v1"
RAW_AGE_FORMULA_CONTRACT = (
    "clock=caller_native_observed_rows_only",
    "event_age=nan_until_first_true_then_zero_and_uncapped_integer_increment",
    "state_age=nan_outside_valid_prefix_then_zero_on_first_or_changed_state",
    "transform=none_no_clip_no_log_no_sentinel",
)
RAW_AGE_FORMULA_SHA256 = hashlib.sha256(
    "\n".join(RAW_AGE_FORMULA_CONTRACT).encode("utf-8")
).hexdigest()


def raw_age_contract_metadata() -> dict[str, object]:
    return {
        "owner": "gx1.features.event_age_v1",
        "schema_version": RAW_AGE_OWNER_SCHEMA_VERSION,
        "formula_contract": list(RAW_AGE_FORMULA_CONTRACT),
        "formula_sha256": RAW_AGE_FORMULA_SHA256,
    }


def _require_one_suffix(valid_mask: np.ndarray, *, context: str) -> int:
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 1:
        raise RuntimeError(f"{context}_VALIDITY_NOT_1D")
    if not valid.any():
        return len(valid)
    first = int(np.argmax(valid))
    if not valid[first:].all():
        raise RuntimeError(f"{context}_VALIDITY_NOT_ONE_PREFIX")
    return first


def raw_event_age_bars(
    event_mask,
    valid_mask=None,
    *,
    initial_age_bars: int | None = None,
) -> np.ndarray:
    """Return raw uncapped observed-row age since the latest true event.

    Rows before the first genuine event are NaN.  ``initial_age_bars`` is an
    explicit continuation state: it is the age at the observed row directly
    preceding this chunk, never a default or a fabricated observation.
    """

    event = np.asarray(event_mask)
    if event.ndim != 1 or event.dtype != np.bool_:
        raise RuntimeError("RAW_EVENT_AGE_FLAGS_MUST_BE_1D_BOOL")
    valid = (
        np.ones(event.shape[0], dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != event.shape:
        raise RuntimeError("RAW_EVENT_AGE_VALIDITY_SHAPE_MISMATCH")
    first_valid = _require_one_suffix(valid, context="RAW_EVENT_AGE")
    if initial_age_bars is not None and (
        isinstance(initial_age_bars, (bool, np.bool_))
        or not isinstance(initial_age_bars, (int, np.integer))
        or int(initial_age_bars) < 0
    ):
        raise RuntimeError("RAW_EVENT_AGE_INITIAL_STATE_INVALID")
    if initial_age_bars is not None and first_valid > 0:
        raise RuntimeError("RAW_EVENT_AGE_CARRY_WITH_INVALID_PREFIX")

    ages = np.full(event.shape[0], np.nan, dtype=np.float64)
    age = None if initial_age_bars is None else int(initial_age_bars)
    for row in range(first_valid, event.shape[0]):
        if bool(event[row]):
            age = 0
        elif age is not None:
            age += 1
        if age is not None:
            ages[row] = float(age)
    return ages


def raw_event_age_from_last_observed_row(
    current_row: int,
    last_event_row: int | None,
) -> float:
    """Scalar counterpart for incremental owners with an exact row index."""

    if (
        isinstance(current_row, (bool, np.bool_))
        or not isinstance(current_row, (int, np.integer))
        or int(current_row) < 0
    ):
        raise RuntimeError("RAW_EVENT_AGE_CURRENT_ROW_INVALID")
    if last_event_row is None:
        return float("nan")
    if (
        isinstance(last_event_row, (bool, np.bool_))
        or not isinstance(last_event_row, (int, np.integer))
        or int(last_event_row) > int(current_row)
    ):
        raise RuntimeError("RAW_EVENT_AGE_LAST_EVENT_ROW_INVALID")
    if int(last_event_row) < 0:
        return float("nan")
    return float(int(current_row) - int(last_event_row))


def raw_state_age_bars(state_values, valid_mask=None) -> np.ndarray:
    """Return raw uncapped duration of the current discrete native state."""

    state = np.asarray(state_values)
    if state.ndim != 1:
        raise RuntimeError("RAW_STATE_AGE_VALUES_NOT_1D")
    if valid_mask is None:
        if np.issubdtype(state.dtype, np.number):
            valid = np.isfinite(state.astype(np.float64, copy=False))
        else:
            valid = np.ones(state.shape[0], dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != state.shape:
        raise RuntimeError("RAW_STATE_AGE_VALIDITY_SHAPE_MISMATCH")
    first_valid = _require_one_suffix(valid, context="RAW_STATE_AGE")
    ages = np.full(state.shape[0], np.nan, dtype=np.float64)
    if first_valid == len(state):
        return ages
    age = 0
    ages[first_valid] = 0.0
    previous = state[first_valid]
    for row in range(first_valid + 1, len(state)):
        current = state[row]
        if current == previous:
            age += 1
        else:
            age = 0
        ages[row] = float(age)
        previous = current
    return ages


__all__ = [
    "RAW_AGE_FORMULA_CONTRACT",
    "RAW_AGE_FORMULA_SHA256",
    "RAW_AGE_OWNER_SCHEMA_VERSION",
    "raw_age_contract_metadata",
    "raw_event_age_bars",
    "raw_event_age_from_last_observed_row",
    "raw_state_age_bars",
]
