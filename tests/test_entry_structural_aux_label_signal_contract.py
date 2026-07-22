from __future__ import annotations

import ast
from pathlib import Path

import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_STRUCTURAL_AUX_LABEL_SIGNAL_CONTRACT,
    model_native_signal_contract_metadata,
)
from gx1.contracts.entry_structural_aux_label_signal_v1 import (
    STRUCTURAL_AUX_LABEL_SIGNAL_REQUIREMENTS,
    structural_aux_label_signal_contract_metadata,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields


PROMOTED_GEOMETRY_REQUIREMENTS = {
    "chart.geometry_support_line_proximity_stack",
    "chart.geometry_resistance_line_proximity_stack",
    "chart.geometry_channel_position_low_to_high",
    "chart.geometry_channel_edge_pressure",
}


def test_every_structural_aux_requirement_is_code_owned_and_mandatory() -> None:
    mandatory = set(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
    assert PROMOTED_GEOMETRY_REQUIREMENTS <= mandatory
    assert MODEL_NATIVE_STRUCTURAL_AUX_LABEL_SIGNAL_CONTRACT == (
        structural_aux_label_signal_contract_metadata(
            MODEL_NATIVE_MANDATORY_SELECTED_FIELDS
        )
    )
    for candidates in STRUCTURAL_AUX_LABEL_SIGNAL_REQUIREMENTS.values():
        assert mandatory.intersection(candidates)

    signal_contract = model_native_signal_contract_metadata(
        canonical_model_native_selected_fields()
    )
    assert signal_contract["structural_aux_label_signal_contract"] == (
        MODEL_NATIVE_STRUCTURAL_AUX_LABEL_SIGNAL_CONTRACT
    )


def test_structural_aux_contract_rejects_ranking_owned_prerequisite() -> None:
    mandatory = list(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
    mandatory.remove("chart.geometry_channel_position_low_to_high")
    with pytest.raises(
        RuntimeError,
        match="STRUCTURAL_AUX_LABEL_SIGNAL_REQUIREMENTS_NOT_MANDATORY",
    ):
        structural_aux_label_signal_contract_metadata(mandatory)


def test_dataset_builder_uses_the_complete_named_requirement_registry() -> None:
    source = Path(
        "gx1/scripts/build_entry_v10_ctx_training_dataset_v3.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    used = {
        str(call.args[0].value)
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "_structural_signal"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    }
    assert used == set(STRUCTURAL_AUX_LABEL_SIGNAL_REQUIREMENTS)

    direct_sig_calls = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "_sig_col"
    ]
    assert len(direct_sig_calls) == 1
