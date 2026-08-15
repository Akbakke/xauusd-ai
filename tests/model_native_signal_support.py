"""Small deterministic fixtures for the exact model-native signal contract."""
from __future__ import annotations

from collections.abc import Iterable

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
)


def canonical_model_native_selected_fields(
    *,
    required_fields: Iterable[str] = (),
    remainder_prefix: str = "session_regime.ranked_fixture",
) -> list[str]:
    del remainder_prefix  # retired fixed-capacity fixture surface
    selected = [
        *MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
        *MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
    ]
    missing = sorted(set(str(name) for name in required_fields) - set(selected))
    if missing:
        raise RuntimeError(
            f"MODEL_NATIVE_TEST_REQUIRED_FIELDS_NOT_CODE_OWNED: {missing}"
        )
    if len(selected) != len(set(selected)):
        raise RuntimeError("MODEL_NATIVE_TEST_SELECTED_FIELDS_DUPLICATE")
    return selected
