from __future__ import annotations

import pytest

from gx1.contracts.entry_model_native_learned_component_movement_v1 import (
    COMPONENT_PARAMETERS,
    ENCODER_COMPONENT_PREFIXES,
    PARAMETER_SHAPES,
    REFERENCE,
    SCHEMA_VERSION,
    require_learned_component_movement_metadata,
)


def _passing_movement() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "reference": REFERENCE,
        "selected_checkpoint_epoch": 1,
        "parameter_deltas": {
            name: {
                "shape": shape,
                "max_abs_delta": 1.0,
                "l2_delta": 1.0,
                "changed": True,
            }
            for name, shape in PARAMETER_SHAPES.items()
        },
        "component_changed": {
            name: True for name in COMPONENT_PARAMETERS
        },
        "encoder_component_movement": {
            name: {
                "parameter_count": 2,
                "changed_parameter_count": 1,
                "max_abs_delta": 1.0,
                "l2_delta": 1.0,
                "changed": True,
            }
            for name in ENCODER_COMPONENT_PREFIXES
        },
        "output_rows_distinct": True,
        "decision": "PASS",
    }


def test_candidate_movement_requires_every_local_and_mtf_family_encoder() -> None:
    value = _passing_movement()
    normalized = require_learned_component_movement_metadata(
        value,
        context="TEST",
    )
    assert set(normalized["encoder_component_movement"]) == set(
        ENCODER_COMPONENT_PREFIXES
    )

    inactive = next(iter(ENCODER_COMPONENT_PREFIXES))
    value["encoder_component_movement"][inactive]["changed"] = False  # type: ignore[index]
    with pytest.raises(RuntimeError, match="ENCODER_INACTIVE"):
        require_learned_component_movement_metadata(value, context="TEST")
