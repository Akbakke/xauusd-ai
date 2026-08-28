from __future__ import annotations

import json
import copy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gx1.contracts.entry_model_native_smoke_bundle_audit_v1 import (
    PRETRAIN_AUDIT_SCHEMA,
)
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
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
    model_native_signal_contract_metadata,
)
from gx1.contracts.entry_model_native_smoke_bundle_audit_v1 import (
    SCHEMA_VERSION as SMOKE_BUNDLE_AUDIT_SCHEMA_VERSION,
    require_smoke_bundle_audit_contract,
    require_smoke_bundle_training_pipeline_contract,
)
from gx1.contracts.entry_model_native_train_launch_v1 import (
    MODEL_NATIVE_RECIPE_ENV_KEYS,
    RECIPE_AUDIT_SCHEMA,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    SCHEMA_VERSION as TRAINING_OBJECTIVE_SCHEMA,
    training_objective_contract_metadata,
)
from gx1.contracts.entry_model_native_joint_task_weighting_v1 import (
    JOINT_TASK_NAMES,
)
from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    CORE_ARTIFACTS as BUNDLE_COMMIT_CORE_ARTIFACTS,
    write_bundle_commit_manifest,
)
from gx1.models.entry_v10.direction_decision_contract import (
    model_direction_decision_contract_metadata,
)
from gx1.scripts import verify_entry_candidate_readiness_v1 as readiness
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    PREDICTION_EVIDENCE_SCHEMA_VERSION,
)
from tests.entry_model_native_smoke_audit_support import passing_smoke_audit_splits
from tests.model_native_signal_support import canonical_model_native_selected_fields


STAMP_TIME = datetime(2026, 7, 16, 12, 0, 0, 123456, tzinfo=timezone.utc)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path.resolve()


def _objective() -> dict:
    return training_objective_contract_metadata()


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
    write_bundle_commit_manifest(
        bundle_dir=bundle,
        artifact_names=BUNDLE_COMMIT_CORE_ARTIFACTS,
        bundle_kind="trained",
        created_at_utc=STAMP_TIME.isoformat(),
    )

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
            "schema_version": "entry_target_foundation_audit_v4",
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
            "schema_version": PRETRAIN_AUDIT_SCHEMA,
            "decision": "PASS",
            "failures": [],
        },
    )
    prediction_report = _write_json(
        evidence / "ENTRY_CANDIDATE_SELECTIVE_EDGE_20260716T120002123456Z.json",
        {"fixture": True},
    )
    prediction_path = (
        evidence / "selective_edge_predictions_20260716T120002123456Z.parquet"
    )
    prediction_path.write_bytes(b"candidate-readiness-prediction-evidence")

    def binding(path: Path) -> dict[str, str]:
        return readiness._artifact_binding(path)

    report = {
        "schema_version": SMOKE_BUNDLE_AUDIT_SCHEMA_VERSION,
        **foundation_audit_policy_binding(),
        "decision": "PASS",
        "failures": [],
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "sequence_length": MODEL_NATIVE_SEQ_LEN,
        "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "bundle_dir": str(bundle),
        "dataset_dir": str(dataset),
        "data_splits": ["val"],
        "model_native_readiness_contract": model_native_readiness_contract_metadata(),
        "direction_decision_contract": model_direction_decision_contract_metadata(),
        "bundle_artifacts": {
            "bundle_commit": binding(
                bundle / "ENTRY_MODEL_NATIVE_BUNDLE_COMMIT.json"
            ),
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
                "schema_version": "entry_target_foundation_audit_v4",
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
                "schema_version": PRETRAIN_AUDIT_SCHEMA,
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
        # The six per-family edge proofs are retired with the handwritten edge
        # scorebook; the producer now emits the fitted-Q block. The two
        # economics booleans stay False exactly as the producer pins them —
        # the audit may run clean while an edge CLAIM remains disallowed.
        "edge_contract": {
            "decision": "PASS",
            "failures": [],
            "raw_entry_q_structure_proven": True,
            "production_economics_ready": False,
            "edge_claim_allowed": False,
        },
        "splits": passing_smoke_audit_splits(),
        "prediction_evidence": {
            "schema_version": PREDICTION_EVIDENCE_SCHEMA_VERSION,
            "evidence_stage": "pre_calibration",
            "authoritative": False,
            "runtime_head_evidence_authoritative": False,
            "path": str(prediction_path),
            "sha256": binding(prediction_path)["sha256"],
            "splits": ["val"],
            "models": ["entry_model_native_smoke"],
        },
        "prediction_evidence_stage": "pre_calibration",
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


def _technical_only_smoke(report: dict) -> None:
    """Make a real pipeline proof that deliberately lacks model-quality proof."""

    report["decision"] = "FAIL"
    report["failures"] = [
        "split/val: specialist family is not top-ranked in tiny smoke",
    ]
    report["specialist_contract"] = {
        "decision": "FAIL",
        "failures": ["specialist gate liveness is unproven"],
        "specialists": list(MODEL_NATIVE_REQUIRED_SPECIALISTS),
        "gate_liveness_proven": False,
    }
    report["liveness_contract"] = {
        "decision": "FAIL",
        "failures": ["one or more specialists lack strict quality evidence"],
        "all_active_head_predictions_live": True,
        "all_specialist_gates_live": False,
        "strict_bundle_components_live": True,
    }
    report["splits"]["val"]["decision"] = "FAIL"
    report["splits"]["val"]["failures"] = list(report["failures"])
    weights = {
        name: 1.0 / len(MODEL_NATIVE_REQUIRED_SPECIALISTS)
        for name in MODEL_NATIVE_REQUIRED_SPECIALISTS
    }
    report["splits"]["val"]["specialist_gate"] = {
        "decision": "FAIL",
        "failures": list(report["failures"]),
        "finite": True,
        "row_sum_max_abs_error": 0.0,
        "entropy_mean": 1.0,
        "mean_weight": weights,
        "std_weight": {
            name: 0.01 for name in MODEL_NATIVE_REQUIRED_SPECIALISTS
        },
        "top_rank_count": {
            name: 0 for name in MODEL_NATIVE_REQUIRED_SPECIALISTS
        },
    }


def test_exact_smoke_consumer_contract_accepts_only_full_seq513_proof(
    tmp_path: Path,
) -> None:
    report, _ = _fixture(tmp_path)

    normalized = require_smoke_bundle_audit_contract(report, context="TEST")

    assert normalized["contract_mode"] == MODEL_NATIVE_CONTRACT_MODE
    assert normalized["sequence_length"] == MODEL_NATIVE_SEQ_LEN
    assert normalized["signal_dim"] == MODEL_NATIVE_SIGNAL_DIM
    assert normalized["model_native_training_objective_contract"][
        "meta_lock_exact"
    ] is True
    policy = foundation_audit_policy_metadata()["smoke_edge_pockets"]
    assert set(normalized["splits"]) == {"val"}
    for split in normalized["splits"].values():
        direction = split["direction"]
        assert direction["support_scope"] == "global"
        assert direction["minimum_trade_rows"] == policy["min_trade_rows"]
        assert direction["minimum_prediction_rows_per_class"] == policy[
            "min_prediction_rows_per_class"
        ]
        # Smoke gates on validity only: support, all three classes emitted and
        # beating the majority baseline. Precision bars are proved on untouched
        # TEST, where the sample supports the claim, so the smoke artifact no
        # longer publishes them as minimums.
        for retired in (
            "minimum_direction_accuracy",
            "minimum_balanced_accuracy",
            "minimum_trade_direction_precision",
            "minimum_trade_precision_wilson_lower",
            "minimum_class_precision",
            "minimum_class_precision_wilson_lower",
        ):
            assert retired not in direction
        context_contract = split["context_slice_contract"]
        assert context_contract["minimum_trade_rows_per_slice"] == policy[
            "min_context_trade_rows"
        ]


def test_candidate_start_uses_technical_pipeline_not_smoke_quality_result(
    tmp_path: Path,
) -> None:
    report, _ = _fixture(tmp_path)
    _technical_only_smoke(report)

    with pytest.raises(RuntimeError):
        require_smoke_bundle_audit_contract(report, context="STRICT")

    technical = require_smoke_bundle_training_pipeline_contract(
        report,
        context="TRAIN_START",
    )
    assert technical["training_pipeline_ready"] is True
    assert technical["qualification_decision"] == "FAIL"
    assert technical["specialist_gate_connectivity"]["finite"] is True

    broken = copy.deepcopy(report)
    broken["splits"]["val"]["specialist_gate"]["std_weight"][
        MODEL_NATIVE_REQUIRED_SPECIALISTS[0]
    ] = 0.0
    with pytest.raises(RuntimeError, match="ROUTE_CONSTANT"):
        require_smoke_bundle_training_pipeline_contract(
            broken,
            context="TRAIN_START",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report.update({"contract_mode": "smart_seq520_candidate"}),
        lambda report: report.update(
            {"sequence_length": MODEL_NATIVE_SEQ_LEN - 1}
        ),
        lambda report: report["head_contract"]["active_heads"].pop(),
        lambda report: report["specialist_contract"].update(
            {"gate_liveness_proven": False}
        ),
        lambda report: report["liveness_contract"].update(
            {"strict_bundle_components_live": False}
        ),
        lambda report: report["edge_contract"].update(
            {"raw_entry_q_structure_proven": False}
        ),
        lambda report: report["splits"].pop("val"),
        lambda report: report["splits"]["val"]["direction"].update(
            {"minimum_trade_rows": 1}
        ),
        lambda report: report["splits"]["val"]["direction"].update(
            {"minimum_trade_direction_precision": 0.01}
        ),
        lambda report: report["splits"]["val"]["direction"].update(
            {"trade_direction_precision_wilson_lower": 1.0}
        ),
        lambda report: report["splits"]["val"]["context_slice_contract"].update(
            {"minimum_trade_rows_per_slice": 1}
        ),
        lambda report: report["prediction_evidence"].update(
            {"schema_version": "entry_candidate_model_direction_prediction_evidence_v2"}
        ),
        lambda report: report["prediction_evidence"].update(
            {"authoritative": True}
        ),
        lambda report: report["prediction_evidence"].update(
            {"splits": ["val", "test"]}
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
    lock["model_native_training_objective"]["fixed_relative_task_weights"] = True
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    assert readiness._bundle_file_check(normalized)["ok"] is False


def test_prediction_evidence_check_accepts_only_policy_owned_val(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_path = _write_json(tmp_path / "prediction_report.json", {"ok": True})
    predictions_path = tmp_path / "predictions.parquet"
    predictions_path.write_bytes(b"prediction-evidence")
    evidence = {
        "path": str(predictions_path),
        "sha256": readiness._sha256_file(predictions_path),
        "splits": ["val"],
        "models": ["entry_model_native_smoke"],
    }
    smoke = {
        "prediction_report_json": str(report_path),
        "prediction_report_sha256": readiness._sha256_file(report_path),
        "prediction_evidence": evidence,
        "bundle_dir": str(tmp_path / "bundle"),
        "dataset_dir": str(tmp_path / "dataset"),
    }
    resolver_calls: list[dict] = []

    def fake_resolver(*_, **kwargs):
        resolver_calls.append(kwargs)
        return predictions_path, {"decision": "PASS"}, evidence

    monkeypatch.setattr(
        readiness,
        "resolve_and_validate_prediction_evidence",
        fake_resolver,
    )

    assert readiness._prediction_evidence_check(smoke)["ok"] is True
    assert resolver_calls == [
        {
            "expected_sha256": evidence["sha256"],
            "prediction_report_path": report_path,
            "bundle_dir": Path(smoke["bundle_dir"]),
            "dataset_dir": Path(smoke["dataset_dir"]),
            "expected_stage": "pre_calibration",
            "expected_splits": ("val",),
            "expected_model": "entry_model_native_smoke",
        }
    ]

    evidence["splits"] = ["val", "test"]
    failed = readiness._prediction_evidence_check(smoke)
    assert failed["ok"] is False
    assert "policy-owned smoke splits" in failed["details"]["error"]


def test_candidate_readiness_run_uses_only_exact_immutable_inputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    smoke, paths = _fixture(tmp_path)
    _technical_only_smoke(smoke)
    smoke_path, _ = _event(
        paths["evidence"],
        "ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT",
        smoke,
        offset_seconds=3,
    )
    future = {
        "profile": "smoke",
        "control_route": "model-native-smoke-train",
        "wrapper_path": "scripts/run_entry_model_native_seq513_train.sh",
        "recipe_audit_schema": RECIPE_AUDIT_SCHEMA,
        "training_objective_schema": TRAINING_OBJECTIVE_SCHEMA,
        "recipe_env_keys": list(MODEL_NATIVE_RECIPE_ENV_KEYS),
        "joint_task_names": list(JOINT_TASK_NAMES),
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
    assert report["dataset_dir"] == str(paths["dataset"].resolve())
    assert report["smoke_bundle_dataset_dir"] == str(paths["dataset"].resolve())
    assert report["checks"][0]["details"]["qualification_decision"] == "FAIL"
    assert Path(report["json_path"]).is_file()


def test_candidate_readiness_refuses_to_write_evidence_inside_dataset(
    monkeypatch,
    tmp_path,
) -> None:
    smoke, paths = _fixture(tmp_path)
    _technical_only_smoke(smoke)
    smoke_path, _ = _event(
        paths["evidence"],
        "ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT",
        smoke,
        offset_seconds=3,
    )
    future = {
        "profile": "smoke",
        "control_route": "model-native-smoke-train",
        "wrapper_path": "scripts/run_entry_model_native_seq513_train.sh",
        "recipe_audit_schema": RECIPE_AUDIT_SCHEMA,
        "training_objective_schema": TRAINING_OBJECTIVE_SCHEMA,
        "recipe_env_keys": list(MODEL_NATIVE_RECIPE_ENV_KEYS),
        "joint_task_names": list(JOINT_TASK_NAMES),
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
    with pytest.raises(RuntimeError, match="out_dir must not be the dataset"):
        readiness.run(
            readiness.build_parser().parse_args(
                [
                    "--smoke-bundle-audit-json", str(smoke_path),
                    "--specialist-audit-json", str(paths["specialist"]),
                    "--trainability-readiness-json", str(trainability_path),
                    "--expected-smoke-dataset-dir", str(paths["dataset"]),
                    "--out-dir", str(paths["dataset"] / "evidence"),
                    "--quiet",
                ]
            )
        )


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
