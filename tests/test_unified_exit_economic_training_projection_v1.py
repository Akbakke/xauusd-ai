from __future__ import annotations

import numpy as np
import pytest

from gx1.contracts.unified_exit_dataset_adapter_v2 import (
    ECONOMIC_TRAINING_PROJECTION_SCHEMA_VERSION,
    require_economic_training_projection,
    seal_economic_training_projection,
)


def _manifest() -> dict[str, str]:
    return {
        "economic_step_model_sha256": "a" * 64,
        "economic_step_source_manifest_sha256": "b" * 64,
    }


def _projection() -> dict:
    return seal_economic_training_projection(
        {
            "schema_version": ECONOMIC_TRAINING_PROJECTION_SCHEMA_VERSION,
            "entry_row_index": 7,
            "side_index": 0,
            "start_state_index": 3,
            "stop_state_index": 6,
            "hold_stop_state_index": 5,
            "exit_event_kind_index": np.asarray([0, 0, 2], dtype="u1"),
            "exit_reward_bps": np.asarray([1.5, -2.0, 3.25], dtype="<f8"),
            "hold_event_kind_index": np.asarray([1, 1], dtype="u1"),
            "hold_reward_bps": np.asarray([-0.1, -0.1], dtype="<f8"),
            **_manifest(),
        }
    )


def _require(value):
    return require_economic_training_projection(
        value,
        entry_row_index=7,
        side_index=0,
        start_state_index=3,
        stop_state_index=6,
        hold_stop_state_index=5,
        economic_manifest=_manifest(),
    )


def test_projection_seals_exact_read_only_array_bytes() -> None:
    projection = _projection()
    checked = _require(projection)
    assert checked["exit_reward_bps"].tobytes() == np.asarray(
        [1.5, -2.0, 3.25], dtype="<f8"
    ).tobytes()
    assert not checked["exit_reward_bps"].flags.writeable
    assert len(checked["exit_stream_sha256"]) == 64
    assert len(checked["hold_stream_sha256"]) == 64
    assert len(checked["projection_sha256"]) == 64


def test_projection_fails_closed_on_mutable_or_tampered_arrays() -> None:
    projection = _projection()
    mutable = dict(projection)
    mutable["exit_reward_bps"] = projection["exit_reward_bps"].copy()
    with pytest.raises(RuntimeError, match="PROJECTION_ARRAY_INVALID"):
        _require(mutable)

    tampered = dict(projection)
    tampered["exit_stream_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="PROJECTION_HASH_INVALID"):
        _require(tampered)
