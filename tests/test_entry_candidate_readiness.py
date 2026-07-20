from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gx1.contracts.entry_foundation_audit_policy_v1 import (
    FOUNDATION_AUDIT_DATA_SPLITS,
    foundation_audit_policy_binding,
    foundation_audit_policy_enforcement,
    foundation_audit_policy_metadata,
)
from gx1.contracts.entry_model_native_readiness_v1 import (
    MODEL_NATIVE_ACTIVE_HEADS,
    MODEL_NATIVE_BLOCKED_HEADS,
    MODEL_NATIVE_REQUIRED_SPECIALISTS,
    model_native_readiness_contract_metadata,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SIGNAL_DIM,
    model_native_signal_contract_metadata,
)
from gx1.contracts.entry_model_native_smoke_bundle_audit_v1 import (
    require_smoke_bundle_audit_contract,
)
from gx1.contracts.entry_model_native_train_launch_v1 import (
    MODEL_NATIVE_RECIPE_ENV_KEYS,
    RECIPE_AUDIT_SCHEMA,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    REQUIRED_POSITIVE_LOSS_WEIGHTS,
    SCHEMA_VERSION as TRAINING_OBJECTIVE_SCHEMA,
    training_objective_contract_metadata,
)
from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
from gx1.models.entry_v10.direction_decision_contract import (
    model_direction_decision_contract_metadata,
)
from gx1.scripts import verify_entry_candidate_readiness_v1 as readiness
from tests.entry_model_native_smoke_audit_support import passing_smoke_audit_splits
from tests.model_native_signal_support import canonical_model_native_selected_fields


STAMP_TIME = datetime(2026, 7, 16, 12, 0, 0, 123456, tzinfo=timezone.utc)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path.resolve()


def _objective() -> dict:
    return training_objective_contract_metadata(
        {key: 1.0 for key in REQUIRED_POSITIVE_LOSS_WEIGHTS}
    )


def _signal_contract() -> dict:
    return model_native_signal_contract_metadata(
        canonical_model_native_selected_fields(
            remainder_prefix="session_regime.candidate_readiness_fixture"
        )
    )


def _event(
    root: Path,
    prefix: str,
    payload: dict,
    *,
    offset_seconds: int = 0,
) -> tuple[Path, dict]:
    created = STAMP_TIME.replace(second=STAMP_TIME.second + offset_seconds)
    return write_immutable_json_event(
        root,
        prefix,
        {**payload, "created_utc": created.isoformat()},
    )


def _fixture(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    bundle = (tmp_path / "bundle").resolve()
    dataset = (tmp_path / "dataset").resolve()
    evidence = (tmp_path / "evidence").resolve()
    bundle.mkdir()
    dataset.mkdir()
    evidence.mkdir()

    objective = _objective()
    signal = _signal_contract()
    metadata = {
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "seq_len": 96,
        "model_native_signal_contract": signal,
        "model_native_training_objective": objective,
        "direction_decision_contract": model_direction_decision_contract_metadata(),
    }
    lock = {
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "model_native_signal_contract": signal,
        "model_native_training_objective": objective,
    }
    metadata_path = _write_json(bundle / "bundle_metadata.json", metadata)
    lock_path = _write_json(bundle / "MASTER_TRANSFORMER_LOCK.json", lock)
    state_path = bundle / "model_state_dict.pt"
    state_path.write_bytes(b"exact model state")

    specialist_payload = {
        "schema_version": "entry_specialist_feature_group_audit_v1",
        **foundation_audit_policy_binding(),
        "foundation_audit_policy_enforcement": (
            foundation_audit_policy_enforcement("specialist")
        ),
        "decision": "PASS",
        "failures": [],
        "data_splits": list(FOUNDATION_AUDIT_DATA_SPLITS),
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "signal_field_count": MODEL_NATIVE_SIGNAL_DIM,
        "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "required_training_specialists": list(MODEL_NATIVE_REQUIRED_SPECIALISTS),
        "specialist_model_contract_valid": True,
        "signal_routing_all_mapped": True,
        "specialist_input_liveness_all_live": True,
    }
    specialist_path, specialist_event = _event(
        evidence,
        "ENTRY_SPECIALIST_AUDIT",
        specialist_payload,
    )
    target_path = _write_json(
        evidence / "ENTRY_TARGET_AUDIT_20260716T115959123456Z.json",
        {
            "schema_version": "entry_target_foundation_audit_v2",
            **foundation_audit_policy_binding(),
            "foundation_audit_policy_enforcement": (
                foundation_audit_policy_enforcement("target")
            ),
            "decision": "PASS",
            "failures": [],
            "data_splits": list(FOUNDATION_AUDIT_DATA_SPLITS),
        },
    )
    pretrain_path = _write_json(
        evidence / "XAU_PRETRAIN_AUDIT_20260716T115958123456Z.json",
        {
            "schema_version": "xau_direction_repair_pretrain_audit_v1",
            "decision": "PASS",
            "failures": [],
        },
    )
    prediction_report = _write_json(
        evidence / "ENTRY_CANDIDATE_SELECTIVE_EDGE_20260716T120002123456Z.json",
        {"fixture": True},
    )

    def binding(path: Path) -> dict[str, str]:
        return readiness._artifact_binding(path)

    report = {
        "schema_version": "entry_foundation_smoke_bundle_audit_v3",
        **foundation_audit_policy_binding(),
        "decision": "PASS",
        "failures": [],
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "sequence_length": 96,
        "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "bundle_dir": str(bundle),
        "dataset_dir": str(dataset),
        "data_splits": ["val", "test"],
        "model_native_readiness_contract": model_native_readiness_contract_metadata(),
        "direction_decision_contract": model_direction_decision_contract_metadata(),
        "bundle_artifacts": {
            "bundle_metadata": binding(metadata_path),
            "master_transformer_lock": binding(lock_path),
            "model_state_dict": binding(state_path),
        },
        "input_audits": {
            "target": {
                **binding(target_path),
                **foundation_audit_policy_binding(),
                "foundation_audit_policy_enforcement": (
                    foundation_audit_policy_enforcement("target")
                ),
                "schema_version": "entry_target_foundation_audit_v2",
                "decision": "PASS",
                "failures": [],
                "data_splits": list(FOUNDATION_AUDIT_DATA_SPLITS),
            },
            "specialist": {
                **binding(specialist_path),
                **foundation_audit_policy_binding(),
                "foundation_audit_policy_enforcement": (
                    foundation_audit_policy_enforcement("specialist")
                ),
                "schema_version": "entry_specialist_feature_group_audit_v1",
                "decision": "PASS",
                "failures": [],
                "data_splits": list(FOUNDATION_AUDIT_DATA_SPLITS),
            },
            "pretrain": {
                **binding(pretrain_path),
                "schema_version": "xau_direction_repair_pretrain_audit_v1",
                "decision": "PASS",
                "failures": [],
            },
        },
        "model_native_training_objective_contract": {
            "decision": "PASS",
            "failures": [],
            "meta_lock_exact": True,
            "objective": objective,
            "metadata_path": str(metadata_path),
            "metadata_sha256": binding(metadata_path)["sha256"],
            "lock_path": str(lock_path),
            "lock_sha256": binding(lock_path)["sha256"],
        },
        "head_contract": {
            "decision": "PASS",
            "failures": [],
            "active_heads": list(MODEL_NATIVE_ACTIVE_HEADS),
            "blocked_heads": list(MODEL_NATIVE_BLOCKED_HEADS),
        },
        "specialist_contract": {
            "decision": "PASS",
            "failures": [],
            "specialists": list(MODEL_NATIVE_REQUIRED_SPECIALISTS),
            "gate_liveness_proven": True,
        },
        "liveness_contract": {
            "decision": "PASS",
            "failures": [],
            "all_active_head_predictions_live": True,
            "all_specialist_gates_live": True,
            "strict_bundle_components_live": True,
        },
        "edge_contract": {
            "decision": "PASS",
            "failures": [],
            "direction_edge_proven": True,
            "context_slice_edge_proven": True,
            "path_quality_edge_proven": True,
            "bad_path_edge_proven": True,
            "turning_point_edge_proven": True,
            "offline_rl_edge_proven": True,
        },
        "splits": passing_smoke_audit_splits(),
        "prediction_evidence": {
            "schema_version": "entry_candidate_model_direction_prediction_evidence_v2",
            "authoritative": True,
            "path": str(evidence / "selective_edge_predictions_20260716T120002123456Z.parquet"),
        },
        "prediction_report_json": str(prediction_report),
        "prediction_report_sha256": binding(prediction_report)["sha256"],
        "promotion_shadow_live_allowed": False,
        "activation_authority": False,
    }
    paths = {
        "bundle": bundle,
        "dataset": dataset,
        "evidence": evidence,
        "specialist": specialist_path,
    }
    return report, paths


def test_exact_smoke_consumer_contract_accepts_only_full_seq513_proof(
    tmp_path: Path,
) -> None:
    report, _ = _fixture(tmp_path)

    normalized = require_smoke_bundle_audit_contract(report, context="TEST")

    assert normalized["contract_mode"] == MODEL_NATIVE_CONTRACT_MODE
    assert normalized["sequence_length"] == 96
    assert normalized["signal_dim"] == 513
    assert normalized["model_native_training_objective_contract"][
        "meta_lock_exact"
    ] is True
    policy = foundation_audit_policy_metadata()["smoke_edge_pockets"]
    assert set(normalized["splits"]) == {"val", "test"}
    for split in normalized["splits"].values():
        direction = split["direction"]
        assert direction["support_scope"] == "global"
        assert direction["minimum_trade_rows"] == policy["min_trade_rows"]
        assert direction["minimum_prediction_rows_per_class"] == policy[
            "min_prediction_rows_per_class"
        ]
        assert direction["minimum_trade_precision_wilson_lower"] == policy[
            "min_trade_precision_wilson_lower"
        ]
        assert direction["minimum_class_precision_wilson_lower"] == policy[
            "min_class_precision_wilson_lower"
        ]
        context_contract = split["context_slice_contract"]
        assert context_contract["minimum_trade_rows_per_slice"] == policy[
            "min_context_trade_rows"
        ]
        assert context_contract["minimum_trade_precision_wilson_lower"] == policy[
            "min_context_trade_precision_wilson_lower"
        ]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report.update({"contract_mode": "smart_seq520_candidate"}),
        lambda report: report.update({"sequence_length": 30}),
        lambda report: report["head_contract"]["active_heads"].pop(),
        lambda report: report["specialist_contract"].update(
            {"gate_liveness_proven": False}
        ),
        lambda report: report["liveness_contract"].update(
            {"strict_bundle_components_live": False}
        ),
        lambda report: report["edge_contract"].update(
            {"direction_edge_proven": False}
        ),
        lambda report: report["splits"].pop("test"),
        lambda report: report["splits"]["val"]["direction"].update(
            {"minimum_trade_rows": 1}
        ),
        lambda report: report["splits"]["test"]["direction"].update(
            {"trade_direction_precision_wilson_lower": 1.0}
        ),
        lambda report: report["splits"]["val"]["context_slice_contract"].update(
            {"minimum_trade_rows_per_slice": 1}
        ),
    ],
)
def test_exact_smoke_consumer_contract_rejects_soft_or_missing_proof(
    tmp_path: Path,
    mutation,
) -> None:
    report, _ = _fixture(tmp_path)
    mutation(report)

    with pytest.raises(RuntimeError):
        require_smoke_bundle_audit_contract(report, context="TEST")


def test_bundle_rehash_rejects_objective_meta_lock_split_brain(tmp_path: Path) -> None:
    report, paths = _fixture(tmp_path)
    normalized = require_smoke_bundle_audit_contract(report, context="TEST")
    assert readiness._bundle_file_check(normalized)["ok"] is True

    lock_path = paths["bundle"] / "MASTER_TRANSFORMER_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["model_native_training_objective"]["configurable_positive_loss_weights"][
        REQUIRED_POSITIVE_LOSS_WEIGHTS[0]
    ] = 0.0
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    assert readiness._bundle_file_check(normalized)["ok"] is False


def test_candidate_readiness_run_uses_only_exact_immutable_inputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    smoke, paths = _fixture(tmp_path)
    smoke_path, _ = _event(
        paths["evidence"],
        "ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT",
        smoke,
        offset_seconds=3,
    )
    future = {
        "control_route": "model-native-smoke-train",
        "wrapper_path": "scripts/run_entry_model_native_seq513_smoke_train.sh",
        "recipe_audit_schema": RECIPE_AUDIT_SCHEMA,
        "training_objective_schema": TRAINING_OBJECTIVE_SCHEMA,
        "recipe_env_keys": list(MODEL_NATIVE_RECIPE_ENV_KEYS),
        "required_positive_loss_weights": list(REQUIRED_POSITIVE_LOSS_WEIGHTS),
    }
    trainability_path, _ = _event(
        paths["evidence"],
        "ENTRY_MODEL_NATIVE_SEQ513_TRAINABILITY_READINESS",
        {
            "schema_version": "entry_model_native_seq513_trainability_readiness_v1",
            "decision": "READY_FOR_MODEL_NATIVE_SEQ513_TRAINABILITY_REVIEW",
            "failures": [],
            "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
            "expected_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
            "required_training_specialists": list(MODEL_NATIVE_REQUIRED_SPECIALISTS),
            "future_train_contract": future,
        },
        offset_seconds=4,
    )
    monkeypatch.setattr(
        readiness,
        "_prediction_evidence_check",
        lambda contract: readiness._check(
            "immutable prediction evidence rehashes and is model-native", True
        ),
    )

    report = readiness.run(
        readiness.build_parser().parse_args(
            [
                "--smoke-bundle-audit-json",
                str(smoke_path),
                "--specialist-audit-json",
                str(paths["specialist"]),
                "--trainability-readiness-json",
                str(trainability_path),
                "--expected-smoke-dataset-dir",
                str(paths["dataset"]),
                "--out-dir",
                str(tmp_path / "out"),
                "--quiet",
            ]
        )
    )

    assert report["decision"] == "READY_FOR_CANDIDATE_TRAINING"
    assert report["failures"] == []
    assert report["candidate_training_allowed"] is True
    assert report["promotion_shadow_live_allowed"] is False
    assert Path(report["json_path"]).is_file()


def test_parser_rejects_generic_upstream_and_retired_aliases() -> None:
    parser = readiness.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(
        [
            "--smoke-bundle-audit-json",
            "/tmp/smoke.json",
            "--specialist-audit-json",
            "/tmp/specialist.json",
            "--trainability-readiness-json",
            "/tmp/trainability.json",
            "--expected-smoke-dataset-dir",
            "/tmp/dataset",
            "--out-dir",
            "/tmp/out",
        ]
    )
    assert args.trainability_readiness_json == "/tmp/trainability.json"
    assert not hasattr(args, "fail_on_not_ready")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--smoke-bundle-audit-json",
                "/tmp/smoke.json",
                "--specialist-audit-json",
                "/tmp/specialist.json",
                "--trainability-readiness-json",
                "/tmp/trainability.json",
                "--expected-smoke-dataset-dir",
                "/tmp/dataset",
                "--out-dir",
                "/tmp/out",
                "--fail-on-not-ready",
            ]
        )
    for stale in ("--upstream-readiness-json", "--challenger-seq215"):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "--smoke-bundle-audit-json",
                    "/tmp/smoke.json",
                    "--specialist-audit-json",
                    "/tmp/specialist.json",
                    "--trainability-readiness-json",
                    "/tmp/trainability.json",
                    "--expected-smoke-dataset-dir",
                    "/tmp/dataset",
                    "--out-dir",
                    "/tmp/out",
                    stale,
                    "/tmp/stale.json",
                ]
            )
