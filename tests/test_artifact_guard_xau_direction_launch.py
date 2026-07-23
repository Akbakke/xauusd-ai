from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
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
    learned_sizing_authority_contract_metadata,
)
from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    require_bundle_commit_manifest,
)
from gx1.contracts.entry_model_native_launch_approval_v1 import (
    EVENT_PREFIX as LAUNCH_APPROVAL_EVENT_PREFIX,
    SCHEMA_VERSION as LAUNCH_APPROVAL_SCHEMA_VERSION,
    launch_state_approval_payload_sha256,
)
from gx1.contracts.model_native_serve_gate_v1 import (
    DIRECTION_POCKET_MAX_BAD_SIDE_RATE,
    DIRECTION_POCKET_MAX_BAD_SIDE_WILSON_UPPER_95,
    DIRECTION_POCKET_MIN_MEAN_PROXY_PNL_BPS_EXCLUSIVE,
    DIRECTION_POCKET_MIN_SELECTED_ROWS,
    DIRECTION_POCKET_SPREAD_AWARE_PROXY_CONTRACT,
    DIRECTION_POCKET_WILSON_CONFIDENCE_LEVEL,
    MODEL_NATIVE_DIRECTION_POCKET_SCHEMA_VERSION,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
)
from tests.model_native_serve_gate_support import passing_direction_repair_pockets
from tests.model_native_sizing_support import write_passing_runtime_sizing_parity
from tests.model_native_adaptation_support import (
    adaptation_bundle_identity,
    write_initial_adaptation_lifecycle,
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
        "accepted_via_vedtak": None,
        "dataset_event_id": "UNIT_DATASET_EVENT",
        "adaptation_lifecycle_evidence": {},
    }
    state.update(updates)
    return state


def _attach_launch_approval(
    state: dict,
    *,
    root: Path,
    bundle_dir: Path,
) -> None:
    vedtak_id = "UNIT_DIRECTION_VEDTAK"
    commit = require_bundle_commit_manifest(bundle_dir.resolve())
    event_path, _ = write_immutable_json_event(
        root,
        LAUNCH_APPROVAL_EVENT_PREFIX,
        {
            "schema_version": LAUNCH_APPROVAL_SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "decision": "ALLOW",
            "project": artifacts.THIS_PROJECT,
            "vedtak_id": vedtak_id,
            "accepted_bundle_dir": str(bundle_dir.resolve()),
            "bundle_commit_sha256": commit["commit_sha256"],
            "launch_state_payload_sha256": (
                launch_state_approval_payload_sha256(state)
            ),
        },
    )
    state["accepted_via_vedtak"] = {
        "schema_version": LAUNCH_APPROVAL_SCHEMA_VERSION,
        "vedtak_id": vedtak_id,
        "event_path": str(event_path),
        "event_sha256": hashlib.sha256(event_path.read_bytes()).hexdigest(),
    }


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
    [
        "joint_exit_execution_proof_evidence",
        "sizing_runtime_parity_evidence",
        "adaptation_lifecycle_evidence",
    ],
)
def test_allow_state_cannot_omit_launch_authority_evidence(
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


def test_allow_state_rejects_unbound_sizing_authority_before_runtime_parity(
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

    with pytest.raises(artifacts.ArtifactGuardError, match="sizing authority invalid"):
        artifacts._check_v10_entry_launch_contract(tmp_path / "bundle")


def test_allow_state_requires_complete_serve_exit_sizing_and_adaptation_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = write_passing_runtime_sizing_parity(tmp_path)
    bundle_dir = evidence["bundle_dir"]
    parity_binding = evidence["oos_source"][
        "model_head_serve_parity_artifact"
    ]
    parity_path = Path(parity_binding["json_path"])
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    direction_path, _ = write_immutable_json_event(
        tmp_path / "direction_pockets",
        "MODEL_NATIVE_DIRECTION_POCKET_AUDIT",
        {
            "schema_version": MODEL_NATIVE_DIRECTION_POCKET_SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "decision": "PASS",
            "failures": [],
            **{
                field: parity[field]
                for field in (
                    "contract_version",
                    "split",
                    "model_name",
                    "bundle_dir",
                    "dataset_dir",
                    "dataset_parquet",
                    "dataset_parquet_sha256",
                    "prediction_evidence",
                    "prediction_report_evidence",
                    "test_coverage",
                )
            },
            "max_bad_side_rate": DIRECTION_POCKET_MAX_BAD_SIDE_RATE,
            "max_bad_side_wilson_upper_95": (
                DIRECTION_POCKET_MAX_BAD_SIDE_WILSON_UPPER_95
            ),
            "wilson_confidence_level": DIRECTION_POCKET_WILSON_CONFIDENCE_LEVEL,
            "min_selected_rows": DIRECTION_POCKET_MIN_SELECTED_ROWS,
            "min_mean_proxy_pnl_bps_exclusive": (
                DIRECTION_POCKET_MIN_MEAN_PROXY_PNL_BPS_EXCLUSIVE
            ),
            "spread_aware_proxy_pnl_contract": (
                DIRECTION_POCKET_SPREAD_AWARE_PROXY_CONTRACT
            ),
            "predictions_parquet": parity["pinned_predictions"],
            "required_selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
            "observed_selection_score_modes": [MODEL_DIRECTION_SELECTION_MODE],
            "pockets": passing_direction_repair_pockets(),
        },
    )
    direction_binding = {
        "json_path": str(direction_path),
        "sha256": hashlib.sha256(direction_path.read_bytes()).hexdigest(),
    }
    state = _allow_state(
        accepted_bundle_dir=str(bundle_dir),
        bundle_metadata_sha256=hashlib.sha256(
            (bundle_dir / "bundle_metadata.json").read_bytes()
        ).hexdigest(),
        sizing_authority_contract=learned_sizing_authority_contract_metadata(
            adoption_artifact=evidence["adoption_artifact"]
        ),
        joint_exit_execution_proof_evidence=evidence[
            "joint_exit_proof_artifact"
        ],
        sizing_runtime_parity_evidence=evidence[
            "runtime_sizing_parity_artifact"
        ],
        serve_gate_evidence={
            "model_native_serve_parity": parity_binding,
            "model_native_direction_pocket_audit": direction_binding,
        },
    )
    lifecycle = write_initial_adaptation_lifecycle(
        tmp_path / "adaptation",
        bundle=bundle_dir,
        identity=adaptation_bundle_identity(bundle_dir),
        admission_evidence={
            "serve_gate_evidence": state["serve_gate_evidence"],
            "joint_exit_execution_proof_evidence": state[
                "joint_exit_execution_proof_evidence"
            ],
            "sizing_runtime_parity_evidence": state[
                "sizing_runtime_parity_evidence"
            ],
        },
    )
    state["adaptation_lifecycle_evidence"] = lifecycle["artifact"]
    _attach_launch_approval(
        state,
        root=tmp_path / "launch_approval",
        bundle_dir=bundle_dir,
    )
    launch = tmp_path / "PROJECT_STATE_xau_direction_launch.json"
    _write_json(launch, state)
    monkeypatch.setattr(artifacts, "XAU_DIRECTION_LAUNCH_CONTRACT", launch)

    validated = artifacts._check_v10_entry_launch_contract(bundle_dir)

    assert validated["decision"] == "ALLOW"
    assert validated["joint_exit_execution_proof_evidence"] == evidence[
        "joint_exit_proof_artifact"
    ]

    original_dataset_event_id = state["dataset_event_id"]
    state["dataset_event_id"] = "UNIT_TAMPERED_DATASET_EVENT"
    _write_json(launch, state)
    with pytest.raises(artifacts.ArtifactGuardError, match="approval invalid"):
        artifacts._check_v10_entry_launch_contract(bundle_dir)
    state["dataset_event_id"] = original_dataset_event_id

    state["serve_gate_evidence"]["model_native_direction_pocket_audit"][
        "sha256"
    ] = "0" * 64
    _attach_launch_approval(
        state,
        root=tmp_path / "launch_approval",
        bundle_dir=bundle_dir,
    )
    _write_json(launch, state)
    with pytest.raises(artifacts.ArtifactGuardError, match="hash mismatch"):
        artifacts._check_v10_entry_launch_contract(bundle_dir)
