from __future__ import annotations

import pytest

from gx1.contracts.entry_model_native_direction_evidence_fusion_v1 import (
    FUSION_MODE,
    HIDDEN_DIM,
    INPUT_DIM,
    INPUTS,
    INPUTS_SHA256,
    ORDERED_INPUT_LAYOUT,
    OUTPUT_DIM,
    direction_evidence_fusion_metadata,
    require_direction_evidence_fusion_metadata,
)


def test_direction_evidence_fusion_layout_is_exact_and_hash_bound() -> None:
    assert len(INPUTS) == 26
    assert sum(width for _, width in INPUTS) == INPUT_DIM == 96
    assert HIDDEN_DIM == 128
    assert OUTPUT_DIM == 3
    assert FUSION_MODE == "sole_learned_acyclic_96x128x3"
    assert INPUTS_SHA256 == (
        "eaebf8a66c1ae45efde3a0b2d8149e1d16eb782a53db2e3903b061fc0f6f9be6"
    )
    assert ORDERED_INPUT_LAYOUT[0] == {
        "name": "model_native_logits",
        "width": 3,
        "start": 0,
        "stop": 3,
    }
    assert ORDERED_INPUT_LAYOUT[-1]["stop"] == 96


def test_direction_evidence_fusion_metadata_has_no_soft_compatibility() -> None:
    expected = direction_evidence_fusion_metadata()
    assert expected["sole_direction_path"] is True
    assert expected["raw_pre_aux_calibration"] is True
    assert expected["no_direction_derived_inputs"] is True
    assert expected["no_detach"] is True
    assert expected["additive_direction_overrides"] is False
    assert expected["residual_direction_path"] is False
    assert expected["manual_direction_cap"] is False
    assert require_direction_evidence_fusion_metadata(
        expected, context="TEST"
    ) == expected

    stale = dict(expected)
    stale["residual_direction_path"] = True
    with pytest.raises(RuntimeError, match="TEST_DIRECTION_EVIDENCE_FUSION_INVALID"):
        require_direction_evidence_fusion_metadata(stale, context="TEST")
