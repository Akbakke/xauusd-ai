from __future__ import annotations

import numpy as np
import pytest

from gx1.features.entry_foundation_structure_v1 import (
    FOUNDATION_EVENT_AGE_CARRY_KEYS,
    FOUNDATION_STRUCTURE_FEATURE_NAMES,
    FOUNDATION_STRUCTURE_FORMULA_SHA256,
    FOUNDATION_STRUCTURE_SOURCE_FIELDS,
    build_entry_foundation_structure_layer,
    foundation_structure_contract_metadata,
    foundation_event_age_carry_scope,
)


def test_foundation_emits_raw_uncapped_ages_with_honest_prefixes() -> None:
    matrix = np.zeros((8, 3), dtype=np.float32)
    matrix[2, 0] = 1.0
    matrix[5, 1] = 1.0
    matrix[4, 2] = 1.0
    out, names = build_entry_foundation_structure_layer(
        matrix,
        list(FOUNDATION_STRUCTURE_SOURCE_FIELDS),
    )
    assert tuple(names) == FOUNDATION_STRUCTURE_FEATURE_NAMES
    np.testing.assert_array_equal(
        out[:, 0], [np.nan, np.nan, 0, 1, 2, 3, 4, 5]
    )
    np.testing.assert_array_equal(
        out[:, 1], [np.nan, np.nan, np.nan, np.nan, np.nan, 0, 1, 2]
    )
    np.testing.assert_array_equal(
        out[:, 2], [np.nan, np.nan, np.nan, np.nan, 0, 1, 2, 3]
    )


def test_foundation_age_is_not_capped_at_local_sequence_length() -> None:
    rows = 211
    matrix = np.zeros((rows, 3), dtype=np.float32)
    matrix[1, :] = 1.0
    out, _ = build_entry_foundation_structure_layer(
        matrix,
        list(FOUNDATION_STRUCTURE_SOURCE_FIELDS),
    )
    assert out[-1, 0] == 209.0
    assert out[-1, 1] == 209.0
    assert out[-1, 2] == 209.0


def test_foundation_contract_binds_formula_names_and_rejects_nonbinary_event() -> None:
    contract = foundation_structure_contract_metadata()
    assert contract["formula_sha256"] == FOUNDATION_STRUCTURE_FORMULA_SHA256
    assert contract["ordered_feature_names"] == list(
        FOUNDATION_STRUCTURE_FEATURE_NAMES
    )
    matrix = np.zeros((4, 3), dtype=np.float32)
    matrix[2, 0] = 0.75
    with pytest.raises(RuntimeError, match="EVENT_SOURCE_NOT_EXACT_BINARY"):
        build_entry_foundation_structure_layer(
            matrix,
            list(FOUNDATION_STRUCTURE_SOURCE_FIELDS),
        )


def test_foundation_age_carry_is_chunk_exact() -> None:
    matrix = np.zeros((11, 3), dtype=np.float32)
    matrix[[1, 8], 0] = 1.0
    matrix[[4], 1] = 1.0
    matrix[[6], 2] = 1.0
    expected, _ = build_entry_foundation_structure_layer(
        matrix,
        list(FOUNDATION_STRUCTURE_SOURCE_FIELDS),
    )
    state = None
    pieces = []
    for start, stop in ((0, 5), (5, 8), (8, 11)):
        with foundation_event_age_carry_scope(
            state,
            tail_replay_rows=0,
        ) as scope:
            piece, _ = build_entry_foundation_structure_layer(
                matrix[start:stop],
                list(FOUNDATION_STRUCTURE_SOURCE_FIELDS),
            )
        assert tuple(scope.next_state) == FOUNDATION_EVENT_AGE_CARRY_KEYS
        state = scope.next_state
        pieces.append(piece)
    np.testing.assert_array_equal(np.concatenate(pieces), expected)


@pytest.mark.parametrize(
    "state",
    [
        {},
        {name: -1 for name in FOUNDATION_EVENT_AGE_CARRY_KEYS},
        {**{name: None for name in FOUNDATION_EVENT_AGE_CARRY_KEYS}, FOUNDATION_EVENT_AGE_CARRY_KEYS[0]: np.nan},
        {**{name: None for name in FOUNDATION_EVENT_AGE_CARRY_KEYS}, FOUNDATION_EVENT_AGE_CARRY_KEYS[1]: 1.5},
    ],
)
def test_foundation_age_carry_rejects_invalid_state(state: dict[str, object]) -> None:
    with pytest.raises(RuntimeError, match="FOUNDATION_EVENT_AGE_CARRY_STATE_INVALID"):
        with foundation_event_age_carry_scope(state, tail_replay_rows=0):
            pass
