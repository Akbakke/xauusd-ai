from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_SIGNAL_DIM,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.features.entry_model_native_feature_layers_v1 import (
    MODEL_NATIVE_MANDATORY_FAMILY_COUNT,
)
from gx1.contracts.entry_model_native_sizing_authority_v1 import (
    MODEL_NATIVE_SIZING_MODE_LEARNED,
)
from gx1_guards import artifacts


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _allow_state(**updates: object) -> dict:
    state = {
        "schema_version": "gx1_xau_direction_launch_state_v1",
        "project": artifacts.THIS_PROJECT,
        "decision": "ALLOW",
        "latest_terminal_event_id": "UNIT_DIRECTION_PASS",
        "latest_terminal_event_decision": "PASS",
        "decision_surface": "model_direction_argmax",
        "public_trade_flat_surface": "public_trade_flat_decision_logits",
        "required_contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "required_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "required_base_signal_dim": MODEL_NATIVE_BASE_SIGNAL_DIM,
        "required_selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "required_mandatory_causal_layer_feature_count": (
            MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT
        ),
        "required_train_ranked_remainder_feature_count": (
            MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT
        ),
        "required_mandatory_causal_layer_count": MODEL_NATIVE_MANDATORY_FAMILY_COUNT,
        "required_ctx_cont_dim": artifacts.REQUIRED_XAU_CTX_CONT_DIM,
        "required_ctx_cat_dim": artifacts.REQUIRED_XAU_CTX_CAT_DIM,
        "sizing_adoption_mode": MODEL_NATIVE_SIZING_MODE_LEARNED,
        "accepted_via_vedtak": "UNIT_DIRECTION_VEDTAK",
        "dataset_event_id": "UNIT_DATASET_EVENT",
    }
    state.update(updates)
    return state


def test_missing_xau_launch_contract_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing_launch.json"
    monkeypatch.setattr(artifacts, "XAU_DIRECTION_LAUNCH_CONTRACT", missing)

    with pytest.raises(artifacts.ArtifactGuardError, match="contract missing"):
        artifacts._check_v10_entry_launch_contract(tmp_path / "bundle")


def test_explicit_block_state_reports_terminal_blockers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch = tmp_path / "PROJECT_STATE_xau_direction_launch.json"
    _write_json(
        launch,
        {
            "schema_version": "gx1_xau_direction_launch_state_v1",
            "project": artifacts.THIS_PROJECT,
            "decision": "BLOCK",
            "latest_terminal_event_id": "UNIT_HARD_RED",
            "blockers": ["joint active Exit proof absent"],
        },
    )
    monkeypatch.setattr(artifacts, "XAU_DIRECTION_LAUNCH_CONTRACT", launch)

    with pytest.raises(artifacts.ArtifactGuardError, match="UNIT_HARD_RED"):
        artifacts._check_v10_entry_launch_contract(tmp_path / "bundle")


@pytest.mark.parametrize(
    "field_name",
    [
        "required_base_signal_dim",
        "required_selected_feature_count",
        "required_mandatory_causal_layer_feature_count",
        "required_train_ranked_remainder_feature_count",
        "required_mandatory_causal_layer_count",
    ],
)
def test_allow_state_rejects_wrong_full_stack_partition_constants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
) -> None:
    launch = tmp_path / "PROJECT_STATE_xau_direction_launch.json"
    _write_json(launch, _allow_state(**{field_name: 999}))
    monkeypatch.setattr(artifacts, "XAU_DIRECTION_LAUNCH_CONTRACT", launch)

    with pytest.raises(artifacts.ArtifactGuardError, match=field_name):
        artifacts._check_v10_entry_launch_contract(tmp_path / "bundle")


@pytest.mark.parametrize(
    "missing_evidence",
    ["joint_exit_execution_proof_evidence", "sizing_runtime_parity_evidence"],
)
def test_allow_state_cannot_omit_capital_sizing_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_evidence: str,
) -> None:
    launch = tmp_path / "PROJECT_STATE_xau_direction_launch.json"
    state = _allow_state(
        joint_exit_execution_proof_evidence={},
        sizing_runtime_parity_evidence={},
    )
    del state[missing_evidence]
    _write_json(launch, state)
    monkeypatch.setattr(artifacts, "XAU_DIRECTION_LAUNCH_CONTRACT", launch)

    with pytest.raises(artifacts.ArtifactGuardError, match=missing_evidence):
        artifacts._check_v10_entry_launch_contract(tmp_path / "bundle")


def test_allow_state_remains_structurally_blocked_until_producers_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch = tmp_path / "PROJECT_STATE_xau_direction_launch.json"
    _write_json(
        launch,
        _allow_state(
            joint_exit_execution_proof_evidence={"decision": "PASS"},
            sizing_runtime_parity_evidence={"decision": "PASS"},
        ),
    )
    monkeypatch.setattr(artifacts, "XAU_DIRECTION_LAUNCH_CONTRACT", launch)

    with pytest.raises(artifacts.ArtifactGuardError, match="structurally BLOCKED"):
        artifacts._check_v10_entry_launch_contract(tmp_path / "bundle")
