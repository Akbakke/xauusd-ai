"""Small deterministic fixtures for the exact model-native signal contract."""
from __future__ import annotations

from collections.abc import Iterable

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
)


def canonical_model_native_selected_fields(
    *,
    required_fields: Iterable[str] = (),
    remainder_prefix: str = "session_regime.ranked_fixture",
) -> list[str]:
    selected = list(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
    for raw_name in required_fields:
        name = str(raw_name)
        if name not in selected:
            selected.append(name)
    expected_count = (
        len(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
        + MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT
    )
    if len(selected) > expected_count:
        raise RuntimeError("MODEL_NATIVE_TEST_REQUIRED_FIELDS_EXCEED_RANKED_CAPACITY")
    index = 0
    while len(selected) < expected_count:
        name = f"{remainder_prefix}_{index:03d}"
        index += 1
        if name not in selected:
            selected.append(name)
    if len(selected) != len(set(selected)):
        raise RuntimeError("MODEL_NATIVE_TEST_SELECTED_FIELDS_DUPLICATE")
    return selected
