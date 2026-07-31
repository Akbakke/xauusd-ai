from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    require_bundle_commit_manifest,
)
from gx1.contracts.entry_model_native_launch_approval_v1 import (
    VEDTAK_EVENT_PREFIX,
    VEDTAK_SCHEMA_VERSION,
    launch_vedtak_request,
)
from gx1.contracts.entry_model_native_launch_transaction_v1 import (
    EVENT_PREFIX as LAUNCH_TRANSACTION_EVENT_PREFIX,
    SCHEMA_VERSION as LAUNCH_TRANSACTION_SCHEMA_VERSION,
)
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
from gx1.contracts.immutable_event_authority_v1 import (
    select_latest_immutable_event,
    write_immutable_json_event,
)
from gx1.contracts.live_tail_publication_v1 import (
    live_tail_launch_authority,
    publish_live_tail_admission_event,
    publish_live_tail_publication_event,
)
from gx1.contracts.model_native_serve_gate_v1 import (
    DIRECTION_POCKET_MAX_SELECTED_LABEL_ERROR_RATE,
    DIRECTION_POCKET_MAX_SELECTED_LABEL_ERROR_WILSON_UPPER_95,
    DIRECTION_POCKET_MIN_MEAN_PROXY_PNL_BPS_EXCLUSIVE,
    DIRECTION_POCKET_MIN_SELECTED_ROWS,
    DIRECTION_POCKET_SPREAD_AWARE_PROXY_CONTRACT,
    DIRECTION_POCKET_WILSON_CONFIDENCE_LEVEL,
    MODEL_NATIVE_DIRECTION_POCKET_SCHEMA_VERSION,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
)
from gx1.scripts import finalize_entry_model_native_launch_v1 as launch_finalizer
from gx1_guards import artifacts
from tests import test_v12_state_from_prebuilt_refresh as prebuilt_refresh
from tests.model_native_adaptation_support import (
    adaptation_bundle_identity,
    write_initial_adaptation_lifecycle,
)
from tests.model_native_serve_gate_support import passing_direction_repair_pockets
from tests.model_native_sizing_support import write_passing_runtime_sizing_parity


@pytest.fixture(autouse=True)
def _pin_lifecycle_contract_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_lifecycle = artifacts.require_launch_adaptation_authority
    original_live_tail = (
        launch_finalizer.require_newest_live_tail_runtime_authority
    )

    def require_with_event_clock(binding: dict, **kwargs):
        event_path = Path(str(binding.get("json_path") or ""))
        payload = json.loads(event_path.read_text(encoding="utf-8"))
        return original_lifecycle(
            binding,
            now_utc=payload["created_utc"],
            **kwargs,
        )

    def require_live_tail_with_event_clock(authority: dict, **kwargs):
        admission_path = Path(
            str(authority["launch_admission"]["json_path"])
        )
        admission = json.loads(admission_path.read_text(encoding="utf-8"))
        return original_live_tail(
            authority,
            now_utc=admission["created_utc"],
            **kwargs,
        )

    monkeypatch.setattr(
        artifacts,
        "require_launch_adaptation_authority",
        require_with_event_clock,
    )
    monkeypatch.setattr(
        launch_finalizer,
        "require_newest_live_tail_runtime_authority",
        require_live_tail_with_event_clock,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _passing_live_tail_fixture(tmp_path: Path) -> dict:
    paths = prebuilt_refresh._prebuilt_fixture(tmp_path / "live-tail-pair")
    pair_manifest = Path(paths["pair_manifest"])
    generation_root = Path(paths["generation_root"])
    publication_root = tmp_path / "live-tail-publications"
    admission_root = tmp_path / "live-tail-admissions"

    canonical_one, base28_one = prebuilt_refresh._successor_frames()
    prebuilt_refresh._publish_live_tail_fixture_pair(
        paths,
        canonical=canonical_one,
        base28=base28_one,
        created_utc="2026-07-16T12:20:00Z",
    )
    publication_one_path, publication_one = (
        publish_live_tail_publication_event(
            event_root=publication_root,
            pair_manifest_path=pair_manifest,
            generation_root=generation_root,
            created_utc="2026-07-16T12:20:30Z",
        )
    )
    assert publication_one["decision"] == "PASS"

    next_time = pd.Timestamp("2026-07-16T12:20:00Z")
    canonical_two = pd.concat(
        [
            canonical_one,
            pd.DataFrame(
                {
                    "time": [next_time],
                    "open": np.asarray([2404.0], dtype=np.float64),
                    "signal": np.asarray([5.0], dtype=np.float32),
                }
            ),
        ],
        ignore_index=True,
    )
    base_row = pd.DataFrame(
        {
            name: np.asarray([2404.0 + offset], dtype=np.float64)
            for offset, name in enumerate(
                prebuilt_refresh.incremental.RAW_BASE28_COLUMNS
            )
        },
        index=pd.DatetimeIndex([next_time], name="time"),
    )
    base28_two = pd.concat([base28_one, base_row])
    prebuilt_refresh._publish_live_tail_fixture_pair(
        paths,
        canonical=canonical_two,
        base28=base28_two,
        created_utc="2026-07-16T12:25:00Z",
    )
    publication_one_sha256 = _sha256(publication_one_path)
    publication_two_path, publication_two = (
        publish_live_tail_publication_event(
            event_root=publication_root,
            pair_manifest_path=pair_manifest,
            generation_root=generation_root,
            previous_publication_json=publication_one_path,
            previous_publication_sha256=publication_one_sha256,
            created_utc="2026-07-16T12:25:30Z",
        )
    )
    assert publication_two["decision"] == "PASS"
    admission_path, admission = publish_live_tail_admission_event(
        event_root=admission_root,
        parent_publication_json=publication_one_path,
        parent_publication_sha256=publication_one_sha256,
        child_publication_json=publication_two_path,
        child_publication_sha256=_sha256(publication_two_path),
        pair_manifest_path=pair_manifest,
        generation_root=generation_root,
        created_utc="2026-07-16T12:25:31Z",
    )
    assert admission["decision"] == "PASS"
    admission_sha256 = _sha256(admission_path)
    authority = live_tail_launch_authority(
        admission_path,
        expected_sha256=admission_sha256,
    )
    return {
        "admission_path": admission_path,
        "admission_binding": {
            "json_path": str(admission_path),
            "sha256": admission_sha256,
        },
        "authority": authority,
    }


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
            "blockers": ["joint unified Exit proof absent"],
        },
    )
    monkeypatch.setattr(artifacts, "XAU_DIRECTION_LAUNCH_CONTRACT", launch)

    with pytest.raises(artifacts.ArtifactGuardError, match="UNIT_HARD_RED"):
        artifacts._check_v10_entry_launch_contract(tmp_path / "bundle")


@pytest.mark.parametrize("decision", ["allow", " ALLOW ", "Allow"])
def test_launch_decision_requires_exact_allow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
) -> None:
    launch = tmp_path / "PROJECT_STATE_xau_direction_launch.json"
    _write_json(launch, _allow_state(decision=decision))
    monkeypatch.setattr(artifacts, "XAU_DIRECTION_LAUNCH_CONTRACT", launch)

    with pytest.raises(artifacts.ArtifactGuardError, match="fail-closed"):
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
    live_tail = _passing_live_tail_fixture(tmp_path)
    launch = tmp_path / "PROJECT_STATE_xau_direction_launch.json"
    _write_json(
        launch,
        _allow_state(
            joint_exit_execution_proof_evidence={"decision": "PASS"},
            sizing_runtime_parity_evidence={"decision": "PASS"},
            new_entry_live_tail_authority=live_tail["authority"],
        ),
    )
    monkeypatch.setattr(artifacts, "XAU_DIRECTION_LAUNCH_CONTRACT", launch)

    with pytest.raises(artifacts.ArtifactGuardError, match="sizing authority invalid"):
        artifacts._check_v10_entry_launch_contract(tmp_path / "bundle")


def _passing_launch_prerequisites(tmp_path: Path) -> dict:
    live_tail = _passing_live_tail_fixture(tmp_path)
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
            "max_selected_label_error_rate": (
                DIRECTION_POCKET_MAX_SELECTED_LABEL_ERROR_RATE
            ),
            "max_selected_label_error_wilson_upper_95": (
                DIRECTION_POCKET_MAX_SELECTED_LABEL_ERROR_WILSON_UPPER_95
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
        new_entry_live_tail_authority=live_tail["authority"],
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
    return {
        "evidence": evidence,
        "bundle_dir": bundle_dir,
        "parity_path": parity_path,
        "direction_path": direction_path,
        "state": state,
        "lifecycle": lifecycle,
        "live_tail": live_tail,
    }


def test_allow_state_without_static_live_tail_authority_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _passing_launch_prerequisites(tmp_path)
    state = json.loads(json.dumps(chain["state"]))
    state.pop("new_entry_live_tail_authority")
    launch = tmp_path / "PROJECT_STATE_xau_direction_launch.json"
    _write_json(launch, state)
    monkeypatch.setattr(artifacts, "XAU_DIRECTION_LAUNCH_CONTRACT", launch)

    with pytest.raises(
        artifacts.ArtifactGuardError,
        match="static live-tail authority invalid",
    ):
        artifacts._check_v10_entry_launch_contract(chain["bundle_dir"])


def test_allow_state_with_tampered_static_live_tail_authority_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _passing_launch_prerequisites(tmp_path)
    state = json.loads(json.dumps(chain["state"]))
    state["new_entry_live_tail_authority"]["launch_anchor"][
        "pair_generation_id"
    ] = "PAIR_TAMPERED"
    launch = tmp_path / "PROJECT_STATE_xau_direction_launch.json"
    _write_json(launch, state)
    monkeypatch.setattr(artifacts, "XAU_DIRECTION_LAUNCH_CONTRACT", launch)

    with pytest.raises(
        artifacts.ArtifactGuardError,
        match="static live-tail authority invalid",
    ):
        artifacts._check_v10_entry_launch_contract(chain["bundle_dir"])


def test_launch_fails_closed_without_canonical_unified_replay_producer(
    tmp_path: Path,
) -> None:
    chain = _passing_launch_prerequisites(tmp_path)
    launch = tmp_path / "PROJECT_STATE_xau_direction_launch.json"
    with pytest.raises(
        launch_finalizer.EntryLaunchFinalizationError,
        match="canonical unified replay producer is absent",
    ):
        _run_launch_finalizer(
            chain,
            root=tmp_path,
            transaction_id="UNIT_MISSING_EXIT_REPLAY_PRODUCER",
        )
    assert json.loads(launch.read_text(encoding="utf-8"))["decision"] == "BLOCK"


def _run_launch_finalizer(
    chain: dict,
    *,
    root: Path,
    transaction_id: str,
    artifact_registry_path: Path | None = None,
    max_trades: int = 1,
    allow_diagnostic_joint_exit_fixture: bool = False,
    diagnostic_sizing_snapshot=None,
) -> tuple[Path, dict]:
    evidence = chain["evidence"]
    launch_path = root / "PROJECT_STATE_xau_direction_launch.json"
    registry_path = (
        evidence["artifact_registry_path"]
        if artifact_registry_path is None
        else artifact_registry_path
    )
    _write_json(
        launch_path,
        {
            "schema_version": "gx1_xau_direction_launch_state_v1",
            "project": "XAUUSD",
            "decision": "BLOCK",
            "accepted_via_vedtak": None,
        },
    )
    transaction_request = launch_vedtak_request(
        transaction_id=transaction_id,
        accepted_bundle_dir=chain["bundle_dir"],
        bundle_commit_sha256=require_bundle_commit_manifest(
            chain["bundle_dir"]
        )["commit_sha256"],
        target_registry_path=registry_path,
        target_launch_state_path=launch_path,
        operating_point={
            "selection_score": MODEL_DIRECTION_SELECTION_MODE,
            "max_trades": max_trades,
        },
        evidence={
            "sizing_adoption": evidence["adoption_artifact"],
            "joint_exit_sizing_proof": evidence[
                "joint_exit_proof_artifact"
            ],
            "sizing_runtime_parity": evidence[
                "runtime_sizing_parity_artifact"
            ],
            "model_native_serve_parity": {
                "json_path": str(chain["parity_path"]),
                "sha256": hashlib.sha256(
                    chain["parity_path"].read_bytes()
                ).hexdigest(),
            },
            "model_native_direction_pocket_audit": {
                "json_path": str(chain["direction_path"]),
                "sha256": hashlib.sha256(
                    chain["direction_path"].read_bytes()
                ).hexdigest(),
            },
            "adaptation_lifecycle": chain["lifecycle"]["artifact"],
            "live_tail_admission": chain["live_tail"][
                "admission_binding"
            ],
        },
    )
    vedtak_path, _ = write_immutable_json_event(
        root / "launch_vedtak",
        VEDTAK_EVENT_PREFIX,
        {
            "schema_version": VEDTAK_SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "decision": "AUTHORIZE",
            "project": artifacts.THIS_PROJECT,
            "vedtak_id": f"VEDTAK_{transaction_id}",
            "launch_request": transaction_request,
            "launch_request_sha256": hashlib.sha256(
                json.dumps(
                    transaction_request,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
        },
    )
    kwargs = {
        "accepted_bundle_dir": chain["bundle_dir"],
        "sizing_adoption_path": Path(
            evidence["adoption_artifact"]["json_path"]
        ),
        "joint_exit_proof_path": Path(
            evidence["joint_exit_proof_artifact"]["json_path"]
        ),
        "sizing_runtime_parity_path": Path(
            evidence["runtime_sizing_parity_artifact"]["json_path"]
        ),
        "serve_parity_path": chain["parity_path"],
        "direction_pocket_path": chain["direction_path"],
        "adaptation_lifecycle_path": chain["lifecycle"]["event_path"],
        "live_tail_admission_path": chain["live_tail"]["admission_path"],
        "launch_vedtak_path": vedtak_path,
        "transaction_id": transaction_id,
        "max_trades": max_trades,
        "artifact_registry_path": registry_path,
        "launch_state_path": launch_path,
        "approval_event_dir": root / "launch_approval",
        "transaction_event_dir": root / "launch_transaction",
    }
    if not allow_diagnostic_joint_exit_fixture:
        return launch_finalizer.finalize_entry_model_native_launch(**kwargs)
    # Transaction-mechanics tests isolate commit/rollback behavior from the
    # separately blocked candidate-bound unified Exit replay prerequisite.
    if diagnostic_sizing_snapshot is None:
        diagnostic_sizing_snapshot = artifacts.prepare_model_native_sizing_authority(
            learned_sizing_authority_contract_metadata(
                adoption_artifact=evidence["adoption_artifact"]
            ),
            context="UNIT_DIAGNOSTIC_SIZING_SNAPSHOT",
        )
    with (
        patch.object(
            launch_finalizer,
            "require_canonical_unified_replay_launch_authority",
            return_value=None,
        ),
        patch.object(
            artifacts,
            "require_canonical_unified_replay_launch_authority",
            return_value=None,
        ),
        patch.object(
            artifacts,
            "prepare_model_native_sizing_authority",
            return_value=diagnostic_sizing_snapshot,
        ),
    ):
        return launch_finalizer.finalize_entry_model_native_launch(**kwargs)


def test_transactional_launch_finalizer_commits_both_authority_files(
    tmp_path: Path,
) -> None:
    chain = _passing_launch_prerequisites(tmp_path)
    diagnostic_sizing_snapshot = artifacts.prepare_model_native_sizing_authority(
        learned_sizing_authority_contract_metadata(
            adoption_artifact=chain["evidence"]["adoption_artifact"]
        ),
        context="UNIT_FINALIZER_DIAGNOSTIC_SIZING_SNAPSHOT",
    )
    event_path, event = _run_launch_finalizer(
        chain,
        root=tmp_path,
        transaction_id="UNIT_FINALIZER_TRANSACTION",
        allow_diagnostic_joint_exit_fixture=True,
        diagnostic_sizing_snapshot=diagnostic_sizing_snapshot,
    )
    registry_path = chain["evidence"]["artifact_registry_path"]
    launch_path = tmp_path / "PROJECT_STATE_xau_direction_launch.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    state = json.loads(launch_path.read_text(encoding="utf-8"))

    assert event["decision"] == "COMMIT"
    assert Path(event["json_path"]) == event_path
    assert registry["active"]["v10_entry"]["path"] == str(
        chain["bundle_dir"].resolve()
    )
    assert state["decision"] == "ALLOW"
    assert state["blockers"] == []
    assert (
        state["new_entry_live_tail_authority"]
        == chain["live_tail"]["authority"]
    )
    evidence = chain["evidence"]
    retry_kwargs = {
        "accepted_bundle_dir": chain["bundle_dir"],
        "sizing_adoption_path": Path(
            evidence["adoption_artifact"]["json_path"]
        ),
        "joint_exit_proof_path": Path(
            evidence["joint_exit_proof_artifact"]["json_path"]
        ),
        "sizing_runtime_parity_path": Path(
            evidence["runtime_sizing_parity_artifact"]["json_path"]
        ),
        "serve_parity_path": chain["parity_path"],
        "direction_pocket_path": chain["direction_path"],
        "adaptation_lifecycle_path": chain["lifecycle"]["event_path"],
        "live_tail_admission_path": chain["live_tail"]["admission_path"],
        "launch_vedtak_path": Path(
            state["accepted_via_vedtak"]["vedtak_authority"]["json_path"]
        ),
        "transaction_id": "UNIT_FINALIZER_TRANSACTION",
        "max_trades": 1,
        "artifact_registry_path": registry_path,
        "launch_state_path": launch_path,
        "approval_event_dir": tmp_path / "launch_approval",
        "transaction_event_dir": tmp_path / "launch_transaction",
    }
    retry_path, retry_event = launch_finalizer.finalize_entry_model_native_launch(
        **retry_kwargs
    )
    assert retry_path == event_path
    assert retry_event == event
    with pytest.raises(
        launch_finalizer.EntryLaunchFinalizationError,
        match="differs from retry request",
    ):
        launch_finalizer.finalize_entry_model_native_launch(
            **{**retry_kwargs, "max_trades": 2}
        )
    with patch.object(
        artifacts,
        "require_canonical_unified_replay_launch_authority",
        return_value=None,
    ), patch.object(
        artifacts,
        "prepare_model_native_sizing_authority",
        return_value=diagnostic_sizing_snapshot,
    ):
        validated = artifacts._check_v10_entry_launch_contract(
            chain["bundle_dir"],
            launch_contract_path=launch_path,
            selection_contract_path=registry_path,
            target_launch_contract_path=launch_path,
            target_selection_contract_path=registry_path,
        )
    assert validated == state

    registry["note"] = "tampered after commit"
    _write_json(registry_path, registry)
    with (
        patch.object(
            artifacts,
            "require_canonical_unified_replay_launch_authority",
            return_value=None,
        ),
        patch.object(
            artifacts,
            "prepare_model_native_sizing_authority",
            return_value=diagnostic_sizing_snapshot,
        ),
        pytest.raises(artifacts.ArtifactGuardError, match="transaction invalid"),
    ):
        artifacts._check_v10_entry_launch_contract(
            chain["bundle_dir"],
            launch_contract_path=launch_path,
            selection_contract_path=registry_path,
            target_launch_contract_path=launch_path,
            target_selection_contract_path=registry_path,
        )


def test_transactional_launch_finalizer_rolls_back_partial_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _passing_launch_prerequisites(tmp_path)
    diagnostic_sizing_snapshot = artifacts.prepare_model_native_sizing_authority(
        learned_sizing_authority_contract_metadata(
            adoption_artifact=chain["evidence"]["adoption_artifact"]
        ),
        context="UNIT_ROLLBACK_DIAGNOSTIC_SIZING_SNAPSHOT",
    )
    registry_path = chain["evidence"]["artifact_registry_path"]
    launch_path = tmp_path / "PROJECT_STATE_xau_direction_launch.json"
    original_registry = registry_path.read_bytes()
    original_replace = launch_finalizer._replace_target
    calls = 0

    def fail_second_replace(stage: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected launch-state replace failure")
        original_replace(stage, target)

    monkeypatch.setattr(
        launch_finalizer,
        "_replace_target",
        fail_second_replace,
    )
    with pytest.raises(
        launch_finalizer.EntryLaunchFinalizationError,
        match="injected launch-state replace failure",
    ):
        _run_launch_finalizer(
            chain,
            root=tmp_path,
            transaction_id="UNIT_ROLLBACK_TRANSACTION",
            allow_diagnostic_joint_exit_fixture=True,
            diagnostic_sizing_snapshot=diagnostic_sizing_snapshot,
        )

    assert registry_path.read_bytes() == original_registry
    assert json.loads(launch_path.read_text(encoding="utf-8"))["decision"] == "BLOCK"
    newest = select_latest_immutable_event(
        tmp_path / "launch_transaction",
        LAUNCH_TRANSACTION_EVENT_PREFIX,
    )
    assert newest is not None
    failure = json.loads(newest.read_text(encoding="utf-8"))
    assert failure["decision"] == "FAIL"
    assert failure["rollback_complete"] is True

    monkeypatch.setattr(
        launch_finalizer,
        "_replace_target",
        original_replace,
    )
    event_path, event = _run_launch_finalizer(
        chain,
        root=tmp_path,
        transaction_id="UNIT_ROLLBACK_RETRY_TRANSACTION",
        allow_diagnostic_joint_exit_fixture=True,
        diagnostic_sizing_snapshot=diagnostic_sizing_snapshot,
    )
    assert event["decision"] == "COMMIT"
    assert Path(event["json_path"]) == event_path
    with patch.object(
        artifacts,
        "require_canonical_unified_replay_launch_authority",
        return_value=None,
    ), patch.object(
        artifacts,
        "prepare_model_native_sizing_authority",
        return_value=diagnostic_sizing_snapshot,
    ):
        artifacts._check_v10_entry_launch_contract(
            chain["bundle_dir"],
            launch_contract_path=launch_path,
            selection_contract_path=registry_path,
            target_launch_contract_path=launch_path,
            target_selection_contract_path=registry_path,
        )


def test_transactional_launch_finalizer_records_precommit_side_effect_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _passing_launch_prerequisites(tmp_path)
    registry_path = chain["evidence"]["artifact_registry_path"]
    launch_path = tmp_path / "PROJECT_STATE_xau_direction_launch.json"
    original_registry = registry_path.read_bytes()
    original_publish_backup = launch_finalizer._publish_backup
    calls = 0

    def fail_second_backup(path: Path, encoded: bytes) -> dict[str, str]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected launch-state backup failure")
        return original_publish_backup(path, encoded)

    monkeypatch.setattr(
        launch_finalizer,
        "_publish_backup",
        fail_second_backup,
    )
    with pytest.raises(
        launch_finalizer.EntryLaunchFinalizationError,
        match="injected launch-state backup failure",
    ):
        _run_launch_finalizer(
            chain,
            root=tmp_path,
            transaction_id="UNIT_PRECOMMIT_FAILURE_TRANSACTION",
            allow_diagnostic_joint_exit_fixture=True,
        )

    assert registry_path.read_bytes() == original_registry
    assert json.loads(launch_path.read_text(encoding="utf-8"))["decision"] == "BLOCK"
    newest = select_latest_immutable_event(
        tmp_path / "launch_transaction",
        LAUNCH_TRANSACTION_EVENT_PREFIX,
    )
    assert newest is not None
    failure = json.loads(newest.read_text(encoding="utf-8"))
    assert failure["decision"] == "FAIL"
    assert failure["commit_event"] is None
    assert failure["rollback_complete"] is True


def test_joint_replay_authority_is_candidate_bound_not_registry_bound(
    tmp_path: Path,
) -> None:
    chain = _passing_launch_prerequisites(tmp_path)
    snapshot = artifacts.prepare_model_native_sizing_authority(
        learned_sizing_authority_contract_metadata(
            adoption_artifact=chain["evidence"]["adoption_artifact"]
        ),
        context="UNIT_CANDIDATE_NOT_REGISTRY_AUTHORITY",
    )
    assert snapshot.candidate_bundle_authority["bundle_dir"] == str(
        chain["bundle_dir"]
    )
    assert not {
        "path",
        "active_exit_entries",
        "projection_sha256",
    }.intersection(
        snapshot.candidate_bundle_authority
    )


def test_transactional_launch_rejects_unproven_concurrency_cap(
    tmp_path: Path,
) -> None:
    chain = _passing_launch_prerequisites(tmp_path)
    with pytest.raises(
        launch_finalizer.EntryLaunchFinalizationError,
        match="outside the portfolio replay contract",
    ):
        _run_launch_finalizer(
            chain,
            root=tmp_path,
            transaction_id="UNIT_UNPROVEN_MAX_TRADES_TRANSACTION",
            max_trades=2,
            allow_diagnostic_joint_exit_fixture=True,
        )
    assert json.loads(
        (tmp_path / "PROJECT_STATE_xau_direction_launch.json").read_text(
            encoding="utf-8"
        )
    )["decision"] == "BLOCK"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema_version": LAUNCH_TRANSACTION_SCHEMA_VERSION,
            "decision": "COMMIT",
        },
        {
            "schema_version": (
                "entry_model_native_launch_transaction_terminal_failure_v1"
            ),
            "decision": "FAIL",
            "failures": ["forged"],
            "project": "NOT_XAUUSD",
            "transaction_id": "UNIT_FORGED_TRANSACTION",
            "commit_event": None,
            "rollback_complete": True,
        },
    ],
)
def test_transactional_launch_rejects_forged_newest_recovery_event(
    tmp_path: Path,
    payload: dict,
) -> None:
    chain = _passing_launch_prerequisites(tmp_path)
    registry_path = chain["evidence"]["artifact_registry_path"]
    original_registry = registry_path.read_bytes()
    write_immutable_json_event(
        tmp_path / "launch_transaction",
        LAUNCH_TRANSACTION_EVENT_PREFIX,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        },
    )

    with pytest.raises(
        launch_finalizer.EntryLaunchFinalizationError,
        match="commit event keys mismatch|failure event is malformed",
    ):
        _run_launch_finalizer(
            chain,
            root=tmp_path,
            transaction_id="UNIT_FORGED_RECOVERY_REJECT",
        )

    assert registry_path.read_bytes() == original_registry
    assert json.loads(
        (tmp_path / "PROJECT_STATE_xau_direction_launch.json").read_text(
            encoding="utf-8"
        )
    )["decision"] == "BLOCK"
