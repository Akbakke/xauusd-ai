"""Causal local BOS/CHOCH event-age features.

The former foundation layer emitted 57 hand-weighted state, proxy and
cross-session scores. Those scorebooks were retired in v13. This owner keeps
only three genuine stateful event ages. Their uncapped memory extends beyond
the local model window, and pre-event rows are honestly unavailable instead
of encoded with eventually constant ever-seen masks. Raw swing,
SMC, volatility, trend and session evidence is routed independently to the
learned specialists.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterable

import numpy as np

from gx1.features.event_age_v1 import raw_event_age_bars


FOUNDATION_STRUCTURE_FEATURE_VERSION = (
    "entry_foundation_event_ages_v9_20260814_raw_uncapped_honest_prefix"
)
FOUNDATION_STRUCTURE_FEATURE_PREFIX = "chart.foundation_"
FOUNDATION_EVENT_AGE_CARRY_KEYS = (
    "bos_up_event_age_bars",
    "bos_down_event_age_bars",
    "choch_event_age_bars",
)
FOUNDATION_STRUCTURE_REQUIRED_FAMILIES = ("bos_choch_age",)
FOUNDATION_STRUCTURE_SOURCE_FIELDS = (
    "snap.smc_bos_up",
    "snap.smc_bos_down",
    "snap.smc_choch",
)
FOUNDATION_STRUCTURE_FEATURE_NAMES = tuple(
    f"{FOUNDATION_STRUCTURE_FEATURE_PREFIX}{name}"
    for name in FOUNDATION_EVENT_AGE_CARRY_KEYS
)
FOUNDATION_STRUCTURE_FORMULA_CONTRACT = (
    "source=exact_binary_closed_bar_smc_bos_up_bos_down_choch_events",
    "age=raw_uncapped_observed_rows_since_last_event",
    "unobserved_storage=nan_until_first_genuine_event",
    "observed_storage=raw_uncapped_native_bar_age",
    "carry=none_until_first_event_then_exact_nonnegative_integer_age",
)
FOUNDATION_STRUCTURE_FORMULA_SHA256 = hashlib.sha256(
    "\n".join(FOUNDATION_STRUCTURE_FORMULA_CONTRACT).encode("utf-8")
).hexdigest()
FOUNDATION_STRUCTURE_FEATURE_NAMES_SHA256 = hashlib.sha256(
    "\n".join(FOUNDATION_STRUCTURE_FEATURE_NAMES).encode("utf-8")
).hexdigest()


def foundation_structure_contract_metadata() -> dict[str, object]:
    return {
        "owner": "gx1.features.entry_foundation_structure_v1",
        "feature_version": FOUNDATION_STRUCTURE_FEATURE_VERSION,
        "formula_sha256": FOUNDATION_STRUCTURE_FORMULA_SHA256,
        "formula_contract": list(FOUNDATION_STRUCTURE_FORMULA_CONTRACT),
        "feature_count": len(FOUNDATION_STRUCTURE_FEATURE_NAMES),
        "ordered_feature_names_sha256": (
            FOUNDATION_STRUCTURE_FEATURE_NAMES_SHA256
        ),
        "ordered_feature_names": list(FOUNDATION_STRUCTURE_FEATURE_NAMES),
    }


def missing_foundation_structure_source_fields(
    feature_names: Iterable[str],
) -> list[str]:
    available = set(feature_names)
    return [
        name for name in FOUNDATION_STRUCTURE_SOURCE_FIELDS if name not in available
    ]


def _require_event_age_value(value: object) -> int | None:
    if value is None:
        return None
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
    ):
        raise RuntimeError("FOUNDATION_EVENT_AGE_CARRY_STATE_INVALID")
    return int(numeric)


def _require_event_age_carry_state(
    state: Mapping[str, object] | None,
) -> dict[str, int | None]:
    if state is None:
        return {name: None for name in FOUNDATION_EVENT_AGE_CARRY_KEYS}
    if not isinstance(state, Mapping) or set(state) != set(
        FOUNDATION_EVENT_AGE_CARRY_KEYS
    ):
        raise RuntimeError("FOUNDATION_EVENT_AGE_CARRY_STATE_INVALID")
    return {
        name: _require_event_age_value(state[name])
        for name in FOUNDATION_EVENT_AGE_CARRY_KEYS
    }


class _FoundationEventAgeCarryScope:
    __slots__ = ("initial_state", "tail_replay_rows", "next_state")

    def __init__(
        self,
        initial_state: dict[str, int | None],
        tail_replay_rows: int,
    ) -> None:
        self.initial_state = initial_state
        self.tail_replay_rows = tail_replay_rows
        self.next_state: dict[str, int | None] = {}


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
    """Bind event-age continuation to one synchronous feature build."""

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


def foundation_event_age_carry_scope_is_active() -> bool:
    """Tell orchestration owners whether an exact continuation is bound."""

    return _FOUNDATION_EVENT_AGE_CARRY_SCOPE.get() is not None


def _bars_since_event(
    event: np.ndarray,
    *,
    carry_key: str,
) -> np.ndarray:
    """Return shared raw event age and capture an exact continuation state."""
    flags = np.asarray(event, dtype=bool)
    if flags.ndim != 1:
        raise RuntimeError("FOUNDATION_EVENT_AGE_FLAGS_INVALID")
    scope = _FOUNDATION_EVENT_AGE_CARRY_SCOPE.get()
    if scope is not None and (
        carry_key not in FOUNDATION_EVENT_AGE_CARRY_KEYS
        or carry_key in scope.next_state
    ):
        raise RuntimeError("FOUNDATION_EVENT_AGE_CARRY_CAPTURE_INVALID")
    ages = raw_event_age_bars(
        flags.astype(np.bool_),
        initial_age_bars=(
            None if scope is None else scope.initial_state[carry_key]
        ),
    )
    if scope is not None:
        carry_index = flags.size - scope.tail_replay_rows - 1
        scope.next_state[carry_key] = (
            scope.initial_state[carry_key]
            if carry_index < 0
            else (
                None
                if not np.isfinite(ages[carry_index])
                else int(ages[carry_index])
            )
        )
    return ages


def build_entry_foundation_structure_layer(
    x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build three raw local event ages with honest pre-event NaN prefixes."""

    matrix = np.asarray(x, dtype=np.float32)
    if matrix.ndim != 2:
        raise RuntimeError(
            f"FOUNDATION_STRUCTURE_INPUT_NOT_2D: shape={matrix.shape}"
        )
    if matrix.shape[0] == 0:
        raise RuntimeError("FOUNDATION_STRUCTURE_INPUT_EMPTY")
    if matrix.shape[1] != len(feature_names):
        raise RuntimeError(
            "FOUNDATION_STRUCTURE_FEATURE_NAME_COUNT_MISMATCH: "
            f"names={len(feature_names)} columns={matrix.shape[1]}"
        )
    if len(feature_names) != len(set(feature_names)):
        raise RuntimeError("FOUNDATION_STRUCTURE_FEATURE_NAMES_DUPLICATE")
    missing = missing_foundation_structure_source_fields(feature_names)
    if missing:
        raise RuntimeError(
            "FOUNDATION_STRUCTURE_SOURCE_FIELDS_MISSING: "
            f"{missing}"
        )
    if not np.isfinite(matrix).all():
        raise RuntimeError("FOUNDATION_STRUCTURE_SOURCE_NONFINITE")
    index = {name: position for position, name in enumerate(feature_names)}
    ages = []
    for source_name, carry_key in zip(
        FOUNDATION_STRUCTURE_SOURCE_FIELDS,
        FOUNDATION_EVENT_AGE_CARRY_KEYS,
    ):
        source = matrix[:, index[source_name]]
        if not np.isin(source, (0.0, 1.0)).all():
            raise RuntimeError(
                "FOUNDATION_STRUCTURE_EVENT_SOURCE_NOT_EXACT_BINARY: "
                f"{source_name}"
            )
        age = _bars_since_event(
            source == 1.0,
            carry_key=carry_key,
        )
        ages.append(age)
    out = np.column_stack(ages).astype(np.float32, copy=False)
    return out, list(FOUNDATION_STRUCTURE_FEATURE_NAMES)


__all__ = [
    "FOUNDATION_EVENT_AGE_CARRY_KEYS",
    "FOUNDATION_STRUCTURE_FEATURE_NAMES",
    "FOUNDATION_STRUCTURE_FEATURE_NAMES_SHA256",
    "FOUNDATION_STRUCTURE_FORMULA_CONTRACT",
    "FOUNDATION_STRUCTURE_FORMULA_SHA256",
    "FOUNDATION_STRUCTURE_FEATURE_PREFIX",
    "FOUNDATION_STRUCTURE_FEATURE_VERSION",
    "FOUNDATION_STRUCTURE_REQUIRED_FAMILIES",
    "FOUNDATION_STRUCTURE_SOURCE_FIELDS",
    "build_entry_foundation_structure_layer",
    "foundation_structure_contract_metadata",
    "foundation_event_age_carry_scope",
    "missing_foundation_structure_source_fields",
]
