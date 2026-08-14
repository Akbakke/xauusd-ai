import argparse
import hashlib
import json
from pathlib import Path

import pytest

from gx1.contracts.entry_full_input_liveness_v1 import SCHEMA_VERSION as LIVENESS_SCHEMA
from gx1.contracts.entry_foundation_audit_policy_v1 import (
    FOUNDATION_AUDIT_DATA_SPLITS,
    foundation_audit_policy_binding,
    foundation_audit_policy_enforcement,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.scripts import materialize_entry_model_native_seq513_smoke_manifest_v1 as manifest_gate
from gx1.scripts import verify_entry_model_native_seq513_trainability_readiness_v1 as trainability_gate
from gx1.scripts import verify_entry_model_native_seq513_smoke_readiness_v1 as readiness
from tests.model_native_context_routing_support import (
    context_routing_for_ordered_signal_names,
    ordered_signal_names_for_specialist_indices,
)
from tests.entry_full_input_liveness_support import write_full_input_liveness_fixture
from tests.model_native_sizing_support import (
    model_native_target_audit_evidence,
)


def _write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_contract() -> dict:
    return json.loads(json.dumps(readiness.EXPECTED_MODEL_CONTRACT))


def test_smart_direction_repair_contract_is_consistent_across_gates() -> None:
    expected = readiness.DIRECTION_DIAGNOSTIC_RECIPE_CONTRACT
    assert expected == manifest_gate.DIRECTION_DIAGNOSTIC_RECIPE_CONTRACT
    assert expected == trainability_gate.DIRECTION_DIAGNOSTIC_RECIPE_CONTRACT
    expected_env = readiness.DIRECTION_DIAGNOSTIC_ENV_TEMPLATE
    assert expected_env == manifest_gate.DIRECTION_DIAGNOSTIC_ENV_TEMPLATE
    assert expected_env == trainability_gate.DIRECTION_DIAGNOSTIC_ENV_TEMPLATE
    assert (
        readiness.DIRECTION_CONTEXT_SLICE_CONTRACT
        == manifest_gate.DIRECTION_CONTEXT_SLICE_CONTRACT
    )
    assert (
        readiness.DIRECTION_CONTEXT_SLICE_CONTRACT
        == trainability_gate.DIRECTION_CONTEXT_SLICE_CONTRACT
    )


def _build_fixture(tmp_path: Path, *, smoke_manifest_provenance: bool = True) -> argparse.Namespace:
    dataset_run_id = "MODEL_NATIVE_SEQ513_DATASET_READINESS_PYTEST"
    smart_dataset_dir = tmp_path / "v10_dataset_smart_candidate_20260630"
    smart_smoke_dataset_dir = smart_dataset_dir
    smart_dataset_dir.mkdir()
    full_input_liveness_path, full_input_liveness, _ = write_full_input_liveness_fixture(
        tmp_path / "full_input_liveness",
        dataset_dir=smart_dataset_dir,
    )
    stamped_liveness_path = (
        tmp_path
        / "ENTRY_FULL_INPUT_LIVENESS_CONTRACT_20260716T120000123456Z.json"
    )
    stamped_liveness_path.write_bytes(full_input_liveness_path.read_bytes())
    full_input_liveness_path = stamped_liveness_path
    m5_prebuilt_path = tmp_path / "FULL_PLUS_CTX_v3src.parquet"
    m5_prebuilt_path.write_bytes(b"exact-m5-model-source")
    cache_manifest_path = _write_json(
        tmp_path / "MULTI_TF_V4_CACHE" / "manifest.json",
        {"schema_version": "pytest-cache", "source": str(m5_prebuilt_path)},
    )
    exit_lifecycle_dir = tmp_path / "exit_lifecycle"
    _write_json(
        exit_lifecycle_dir / "UNIFIED_EXIT_LIFECYCLE_MANIFEST.json",
        {"decision": "PASS", "entry_run_id": dataset_run_id},
    )
    pretrain_path = _write_json(
        tmp_path / "XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_20260716T120000223456Z.json",
        {"decision": "PASS", "entry_run_id": dataset_run_id},
    )
    split_artifacts: dict[str, dict[str, str]] = {}
    for split in ("train", "val", "test"):
        parquet_path = (
            smart_smoke_dataset_dir
            / f"v10_smart_seq513_model_native_smoke__DIR_TRAIN_FIT_{split}.parquet"
        )
        manifest_path = (
            smart_smoke_dataset_dir
            / f"v10_smart_seq513_model_native_smoke__DIR_TRAIN_FIT_{split}.manifest.json"
        )
        parquet_path.write_bytes(f"{split}-parquet".encode("utf-8"))
        manifest_path.write_text(f'{{"split":"{split}"}}\n', encoding="utf-8")
        split_artifacts[split] = {
            "out_parquet": str(parquet_path),
            "out_manifest": str(manifest_path),
            "out_parquet_sha256": _sha256(parquet_path),
            "out_manifest_sha256": _sha256(manifest_path),
        }

    _write_json(
        tmp_path
        / "ENTRY_MODEL_NATIVE_SEQ513_REBUILD_PREFLIGHT_20260716T120001123456Z.json",
        {
            "decision": "READY_FOR_MODEL_NATIVE_SEQ513_REBUILD",
            "report_only": True,
            "training_allowed": False,
            "dataset_rebuild_allowed": True,
            "side_effects_started": {
                "dataset_rebuild": False,
                "training": False,
                "replay": False,
                "iql_distillation": False,
                "shadow": False,
                "live": False,
            },
            "counts": {
                "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
                "expected_seq_snap_width": MODEL_NATIVE_SIGNAL_DIM,
            },
            "inputs": {
                "source_parquet": {
                    "exists": True,
                    "path": str(m5_prebuilt_path),
                    "sha256": _sha256(m5_prebuilt_path),
                    "size_bytes": m5_prebuilt_path.stat().st_size,
                },
                "multi_tf_cache": {
                    "manifest": {
                        "exists": True,
                        "path": str(cache_manifest_path),
                        "sha256": _sha256(cache_manifest_path),
                        "size_bytes": cache_manifest_path.stat().st_size,
                    }
                },
                "exit_lifecycle_dir": str(exit_lifecycle_dir),
            },
        },
    )
    _write_json(
        tmp_path / "ENTRY_SMART_DATASET_POST_REBUILD_READINESS_20260716T120002123456Z.json",
        {
            "schema_version": readiness.POST_REBUILD_SCHEMA_VERSION,
            "decision": readiness.POST_REBUILD_READY_DECISION,
            "entry_run_id": dataset_run_id,
            "dataset_dir": str(smart_dataset_dir),
            "post_rebuild_refresh_command_contract": {
                "smoke_dataset_dir": str(smart_smoke_dataset_dir),
            },
            "full_input_liveness_contract": {
                "path": str(full_input_liveness_path),
                "sha256": _sha256(full_input_liveness_path),
                "schema_version": LIVENESS_SCHEMA,
                "decision": full_input_liveness["decision"],
                "field_order_sha256": full_input_liveness["field_order_sha256"],
                "field_counts": full_input_liveness["expected_field_counts"],
                "atr_ood_status": full_input_liveness["atr_ood_drift"]["status"],
            },
            "pretrain_audit": {
                "decision": "PASS",
                "path": str(pretrain_path),
                "sha256": _sha256(pretrain_path),
            },
            "test_isolation": {
                "authority": {
                    "path": str(
                        (
                            tmp_path
                            / "rebuild_authority"
                            / "ENTRY_MODEL_NATIVE_SEQ513_UNTOUCHED_TEST_SEAL_20260716T120002123456Z.json"
                        ).resolve()
                    ),
                    "sha256": "9" * 64,
                }
            },
        },
    )
    _write_json(
        tmp_path / "ENTRY_FEATURE_FOUNDATION_AUDIT_20260716T120003123456Z.json",
        {
            "decision": "PASS",
            "dataset_dir": str(smart_dataset_dir),
            "failures": [],
            "foundation_objective_liveness_all_live": True,
            "foundation_source_field_liveness_all_live": True,
            "foundation_objective_liveness": [
                {
                    "split": "train",
                    "objective": "hh_hl_lh_ll",
                    "missing_count": 0,
                    "nonfinite_count": 0,
                    "near_constant_count": 0,
                }
            ],
            "foundation_source_field_liveness": [
                {
                    "split": "train",
                    "source_field": "chart.foundation_bos_up_event_age_bars",
                    "observed": True,
                    "nonfinite_count": 0,
                    "near_constant": False,
                }
            ],
        },
    )
    _write_json(
        tmp_path / "ENTRY_TARGET_FOUNDATION_AUDIT_20260716T120004123456Z.json",
        {
            "schema_version": "entry_target_foundation_audit_v3",
            **foundation_audit_policy_binding(),
            "foundation_audit_policy_enforcement": (
                foundation_audit_policy_enforcement("target")
            ),
            "decision": "PASS",
            "dataset_dir": str(smart_dataset_dir),
            "failures": [],
            "data_splits": list(FOUNDATION_AUDIT_DATA_SPLITS),
            **model_native_target_audit_evidence(),
        },
    )
    specialist_indices = {name: [] for name in readiness.REQUIRED_SPECIALISTS}
    for index in range(MODEL_NATIVE_SIGNAL_DIM):
        specialist_indices[
            readiness.REQUIRED_SPECIALISTS[index % len(readiness.REQUIRED_SPECIALISTS)]
        ].append(index)
    specialist_liveness = [
        {
            "split": split,
            "specialist": name,
            "feature_count": 2,
            "live_feature_count": 2,
            "min_required_live_feature_count": 1,
            "nonfinite_count": 0,
            "near_constant_count": 0,
            "mean_active_rate": 0.25,
        }
        for split in FOUNDATION_AUDIT_DATA_SPLITS
        for name in readiness.REQUIRED_SPECIALISTS
    ]
    ordered_signal_names = ordered_signal_names_for_specialist_indices(
        specialist_indices
    )
    context_routing = context_routing_for_ordered_signal_names(
        ordered_signal_names
    )
    _write_json(
        tmp_path / "ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_20260716T120005123456Z.json",
        {
            "decision": "PASS",
            "dataset_dir": str(smart_dataset_dir),
            "failures": [],
            "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
            "signal_field_count": MODEL_NATIVE_SIGNAL_DIM,
            "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
            "required_training_specialists": list(readiness.REQUIRED_SPECIALISTS),
            "specialist_model_contract": _model_contract(),
            "specialist_model_contract_valid": True,
            "specialist_model_contract_failures": [],
            "signal_routing_all_mapped": True,
            "signal_unmapped_count": 0,
            "context_specialist_routing_all_mapped": True,
            "context_specialist_routing_failure_count": 0,
            "specialist_input_liveness_all_live": True,
            "specialist_input_liveness": specialist_liveness,
            "feature_rows": [
                {"index": index, "feature": field}
                for index, field in enumerate(ordered_signal_names)
            ],
            "architecture_contract": {
                "input_dim": MODEL_NATIVE_SIGNAL_DIM,
                "specialist_input_indices": specialist_indices,
                "context_specialist_routing": context_routing,
                "recommended_fusion": {
                    "active_heads": list(readiness.SPECIALIST_FUSION_ACTIVE_HEADS),
                    "blocked_heads": list(readiness.SPECIALIST_FUSION_BLOCKED_HEADS),
                },
            },
        },
    )
    smoke_manifest = {
            "schema_version": "entry_model_native_seq513_smoke_dataset_v3",
            "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
            "expected_seq_snap_width": MODEL_NATIVE_SIGNAL_DIM,
            "out_dir": str(smart_smoke_dataset_dir),
            "splits": {
                split: {
                    "rows": 16,
                    **split_artifacts[split],
                    "split_manifest_schema_version": (
                        readiness.SMOKE_SPLIT_MANIFEST_SCHEMA
                    ),
                }
                for split in readiness.PREFREEZE_SPLITS
            },
        }
    smoke_manifest_path = (
        tmp_path / "ENTRY_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST_20260716T120006123456Z.json"
    )
    _write_json(
        smoke_manifest_path,
        {
            "schema_version": "entry_model_native_seq513_smoke_manifest_v3",
            "decision": "READY_FOR_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST_REVIEW",
            "report_only": True,
            "manifest_embedded": True,
            "manifest_sha256": readiness._sha256_json(smoke_manifest),
            "smoke_manifest": smoke_manifest,
            "side_effects_started": {
                "dataset_rebuild": False,
                "training": False,
                "replay": False,
                "iql_distillation": False,
                "shadow": False,
                "live": False,
            },
            "checks": (
                [
                    {
                        "name": name,
                        "ok": True,
                        "details": {},
                    }
                    for name in readiness.REQUIRED_SMOKE_MANIFEST_PROVENANCE_CHECKS
                ]
                if smoke_manifest_provenance
                else []
            ),
        },
    )
    return argparse.Namespace(
        model_native_rebuild_preflight_json=str(
            tmp_path
            / "ENTRY_MODEL_NATIVE_SEQ513_REBUILD_PREFLIGHT_20260716T120001123456Z.json"
        ),
        smart_post_rebuild_readiness_json=str(
            tmp_path
            / "ENTRY_SMART_DATASET_POST_REBUILD_READINESS_20260716T120002123456Z.json"
        ),
        full_input_liveness_json=str(full_input_liveness_path),
        smart_dataset_dir=str(smart_dataset_dir),
        smart_smoke_dataset_dir=str(smart_smoke_dataset_dir),
        smart_feature_audit_json=str(
            tmp_path / "ENTRY_FEATURE_FOUNDATION_AUDIT_20260716T120003123456Z.json"
        ),
        smart_target_audit_json=str(
            tmp_path / "ENTRY_TARGET_FOUNDATION_AUDIT_20260716T120004123456Z.json"
        ),
        smart_specialist_audit_json=str(
            tmp_path
            / "ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_20260716T120005123456Z.json"
        ),
        smoke_manifest_event_json=str(smoke_manifest_path),
        repo_dir=str(tmp_path),
        out_dir=str(tmp_path / "reports"),
        memory_cap="10G",
        swap_cap="512M",
        quiet=True,
    )


def _run_blocked(args: argparse.Namespace) -> dict:
    with pytest.raises(SystemExit) as exc_info:
        readiness.run(args)
    assert exc_info.value.code == 1
    paths = list(Path(args.out_dir).glob(f"{readiness.EVENT_PREFIX}_*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


def test_model_native_seq513_smoke_readiness_passes_as_report_only(monkeypatch, tmp_path: Path) -> None:
    args = _build_fixture(tmp_path)
    monkeypatch.setattr(readiness, "_git_status_short", lambda repo: [])

    report = readiness.run(args)

    assert report["decision"] == "READY_FOR_MODEL_NATIVE_SEQ513_SMOKE_READINESS_REVIEW"
    assert report["report_only"] is True
    assert report["training_allowed"] is False
    assert report["smart_smoke_training_allowed"] is False
    assert report["smart_trainability_readiness_required_before_training"] is True
    assert report["execution_allowed_now"] is False
    assert report["control_surface_mutated"] is False
    assert not any(report["side_effects_started"].values())
    assert report["full_input_liveness_validation"]["ok"] is True
    assert report["full_input_liveness_validation"]["field_counts"] == {
        "signal": MODEL_NATIVE_SIGNAL_DIM,
        "ctx_cont": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat": 5,
    }

    train_contract = report["future_command_contracts"]["smart_smoke_train"]
    assert train_contract["implemented_in_control_surface"] is True
    assert train_contract["execution_allowed_now"] is False
    assert train_contract["requires_clean_git"] is True
    assert train_contract["requires_ram_cap"] is True
    assert train_contract["ram_cap_runner"] == "scripts/gx1_capped_run.sh"
    assert train_contract["num_workers"] == 0
    assert train_contract["prefreeze_test_seal_lineage_required"] is True
    assert train_contract["requires_edge_audit"] is True
    assert train_contract["recipe_audit_control_route_exposed"] is True
    assert train_contract["recipe_audit_control_route"] == "model-native-train-recipe-audit"
    assert train_contract["recipe_audit_argv_template"][:2] == [
        "scripts/entry_next_edge_control.sh",
        "model-native-train-recipe-audit",
    ]
    assert train_contract["post_smoke_prediction_control_route_exposed"] is True
    assert (
        train_contract["post_smoke_prediction_control_route"]
        == "model-native-selective-edge"
    )
    prediction_argv = train_contract["post_smoke_prediction_argv_template"]
    assert prediction_argv[:2] == [
        "scripts/entry_next_edge_control.sh",
        "model-native-selective-edge",
    ]
    assert prediction_argv[prediction_argv.index("--splits") + 1] == "val"
    assert (
        prediction_argv[prediction_argv.index("--evidence-stage") + 1]
        == "pre_calibration"
    )
    smoke_manifest_event = json.loads(
        Path(args.smoke_manifest_event_json).read_text(encoding="utf-8")
    )
    val_artifacts = smoke_manifest_event["smoke_manifest"]["splits"]["val"]
    assert (
        prediction_argv[prediction_argv.index("--val-manifest-sha256") + 1]
        == val_artifacts["out_manifest_sha256"]
    )
    assert (
        prediction_argv[prediction_argv.index("--val-parquet-sha256") + 1]
        == val_artifacts["out_parquet_sha256"]
    )
    assert "--test-manifest-json" not in prediction_argv
    assert train_contract["post_smoke_audit_control_route_exposed"] is True
    assert train_contract["post_smoke_audit_control_route"] == "model-native-smoke-bundle-audit"
    assert train_contract["post_smoke_audit_argv_template"][:2] == [
        "scripts/entry_next_edge_control.sh",
        "model-native-smoke-bundle-audit",
    ]
    assert "--test-manifest-json" not in train_contract[
        "post_smoke_audit_argv_template"
    ]
    audit_argv = train_contract["post_smoke_audit_argv_template"]
    assert audit_argv[audit_argv.index("--pretrain-audit-json") + 1].endswith(
        "XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_20260716T120000223456Z.json"
    )
    assert train_contract["control_route"] == "model-native-smoke-train"
    assert train_contract["profile"] == "smoke"
    assert (
        train_contract["wrapper_path"]
        == "scripts/run_entry_model_native_seq513_train.sh"
    )
    assert train_contract["specialist_contract_mode"] == MODEL_NATIVE_CONTRACT_MODE
    assert train_contract["expected_signal_dim"] == MODEL_NATIVE_SIGNAL_DIM
    assert train_contract[
        "requires_direction_diagnostic_recipe_contract"
    ] is True
    assert train_contract[
        "direction_diagnostic_recipe_contract"
    ] == readiness.DIRECTION_DIAGNOSTIC_RECIPE_CONTRACT
    assert set(train_contract["recipe_env_keys"]) == set(
        readiness.MODEL_NATIVE_RECIPE_ENV_KEYS
    )
    assert set(train_contract["joint_task_names"]) == set(
        readiness.JOINT_TASK_NAMES
    )
    assert train_contract["requires_exact_model_native_training_objective"] is True
    assert train_contract["requires_direction_context_slice_contract"] is True
    assert train_contract["direction_context_slice_contract"] == readiness.DIRECTION_CONTEXT_SLICE_CONTRACT
    assert train_contract["requires_canonical_direction_decision_contract"] is True
    assert (
        train_contract["canonical_direction_decision_contract"]
        == readiness.CANONICAL_DIRECTION_DECISION_CONTRACT
    )
    train_argv = " ".join(train_contract["wrapper_argv_template"])
    assert train_contract["wrapper_argv_template"] == train_contract["argv_template"]
    assert train_contract["wrapper_argv_template"][:2] == [
        "scripts/entry_next_edge_control.sh",
        "model-native-smoke-train",
    ]
    assert str(Path(args.smart_smoke_dataset_dir)) in train_argv
    out_idx = train_contract["wrapper_argv_template"].index("--out-bundle-dir") + 1
    assert str(Path(args.smart_dataset_dir).parent) in train_contract[
        "wrapper_argv_template"
    ][out_idx]
    assert "gx1.models.entry_v10.entry_v10_ctx_train_v3" not in train_argv
    assert "audit-smoke-bundle" not in train_argv
    assert "--recipe-audit-json" in train_contract["wrapper_argv_template"]
    assert "--post-rebuild-readiness-json" in train_contract["wrapper_argv_template"]
    assert "--pretrain-audit-json" in train_contract["wrapper_argv_template"]
    for flag in (
        "--unified-exit-lifecycle-manifest-json",
        "--m5-prebuilt-path",
        "--multi-tf-cache-manifest-json",
        "--dropout",
        "--num-workers",
        "--multi-tf-num-layers",
        "--specialist-num-layers",
        "--grad-accum-steps",
        "--per-tf-seq-len-m5",
        "--per-tf-seq-len-m15",
        "--per-tf-seq-len-h1",
        "--per-tf-seq-len-h4",
        "--per-tf-seq-len-d1",
        "--cross-family-fusion-scale",
    ):
        assert flag in train_contract["wrapper_argv_template"]
    worker_index = train_contract["wrapper_argv_template"].index("--num-workers")
    assert train_contract["wrapper_argv_template"][worker_index + 1] == "0"
    assert Path(report["json_path"]).exists()


def test_model_native_seq513_smoke_readiness_fails_closed_on_dirty_git_and_wrong_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path)
    specialist_path = Path(args.smart_specialist_audit_json)
    specialist = json.loads(specialist_path.read_text(encoding="utf-8"))
    specialist["contract_mode"] = "foundation_seq146"
    specialist["signal_field_count"] = MODEL_NATIVE_SIGNAL_DIM + 6
    _write_json(specialist_path, specialist)
    monkeypatch.setattr(readiness, "_git_status_short", lambda repo: [" M gx1/example.py"])

    report = _run_blocked(args)

    blockers = "\n".join(report["blockers"])
    assert report["decision"] == "BLOCKED_MODEL_NATIVE_SEQ513_SMOKE_READINESS"
    assert report["training_allowed"] is False
    assert report["smart_smoke_training_allowed"] is False
    assert "clean git required before smart smoke train" in blockers
    assert "smart specialist audit uses model-native seq513 contract mode" in blockers
    assert "smart specialist audit has exact smart signal width" in blockers
    assert "trainer loader accepts exact smart specialist contract" in blockers


def test_model_native_seq513_smoke_readiness_fails_closed_on_liveness_hash_tamper(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path)
    liveness_path = Path(args.full_input_liveness_json)
    liveness_path.write_text(liveness_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    monkeypatch.setattr(readiness, "_git_status_short", lambda repo: [])

    report = _run_blocked(args)

    assert report["decision"] == "BLOCKED_MODEL_NATIVE_SEQ513_SMOKE_READINESS"
    assert "model_native_rebuild_preflight: full-input liveness artifact hash schema fields and ATR shift observation validate" in report["blockers"]
    assert any(
        row["code"] == "artifact_sha256_mismatch"
        for row in report["full_input_liveness_validation"]["failures"]
    )


def test_model_native_seq513_smoke_readiness_fails_closed_on_nonfinite_liveness(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path)
    feature_path = Path(args.smart_feature_audit_json)
    feature = json.loads(feature_path.read_text(encoding="utf-8"))
    feature["foundation_objective_liveness"][0]["nonfinite_count"] = 1
    _write_json(feature_path, feature)

    specialist_path = Path(args.smart_specialist_audit_json)
    specialist = json.loads(specialist_path.read_text(encoding="utf-8"))
    specialist["specialist_input_liveness"][0]["nan_count"] = 1
    _write_json(specialist_path, specialist)
    monkeypatch.setattr(readiness, "_git_status_short", lambda repo: [])

    report = _run_blocked(args)

    blockers = "\n".join(report["blockers"])
    assert report["decision"] == "BLOCKED_MODEL_NATIVE_SEQ513_SMOKE_READINESS"
    assert report["training_allowed"] is False
    assert "smart feature audit proves finite live features" in blockers
    assert "smart specialist input has no NaN inf or liveness collapse" in blockers


def test_model_native_seq513_smoke_readiness_allows_specialist_near_constant_when_live_count_passes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path)
    specialist_path = Path(args.smart_specialist_audit_json)
    specialist = json.loads(specialist_path.read_text(encoding="utf-8"))
    row = specialist["specialist_input_liveness"][0]
    row["near_constant_count"] = 1
    row["near_constant_features"] = ["session_regime.h4_d1_regime_sign_agreement"]
    row["live_feature_count"] = max(
        int(row.get("live_feature_count") or 0),
        int(row.get("min_required_live_feature_count") or 1),
    )
    specialist["specialist_input_liveness_all_live"] = True
    _write_json(specialist_path, specialist)
    monkeypatch.setattr(readiness, "_git_status_short", lambda repo: [])

    report = readiness.run(args)

    blockers = "\n".join(report["blockers"])
    assert report["decision"] == "READY_FOR_MODEL_NATIVE_SEQ513_SMOKE_READINESS_REVIEW"
    assert "smart specialist input has no NaN inf or liveness collapse" not in blockers


def test_model_native_seq513_smoke_readiness_fails_closed_on_blocked_manifest_readiness(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path)
    readiness_path = Path(args.smoke_manifest_event_json)
    payload = json.loads(readiness_path.read_text(encoding="utf-8"))
    payload["decision"] = "BLOCKED_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST_READINESS"
    payload["manifest_embedded"] = False
    _write_json(readiness_path, payload)
    monkeypatch.setattr(readiness, "_git_status_short", lambda repo: [])

    report = _run_blocked(args)

    blockers = "\n".join(report["blockers"])
    assert report["decision"] == "BLOCKED_MODEL_NATIVE_SEQ513_SMOKE_READINESS"
    assert report["training_allowed"] is False
    assert "model-native smoke manifest event is ready" in blockers


def test_model_native_seq513_smoke_readiness_rejects_stale_target_audit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path)
    target_path = Path(args.smart_target_audit_json)
    target = json.loads(target_path.read_text(encoding="utf-8"))
    target["schema_version"] = "entry_target_foundation_audit_v1"
    target.pop("model_native_aux_target_contract")
    _write_json(target_path, target)
    monkeypatch.setattr(readiness, "_git_status_short", lambda repo: [])

    report = _run_blocked(args)

    blockers = "\n".join(report["blockers"])
    assert report["decision"] == "BLOCKED_MODEL_NATIVE_SEQ513_SMOKE_READINESS"
    assert "smart target audit proves exact aux-v3 and offline-RL targets" in blockers


def test_model_native_seq513_smoke_readiness_fails_closed_on_stale_manifest_provenance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path, smoke_manifest_provenance=False)
    monkeypatch.setattr(readiness, "_git_status_short", lambda repo: [])

    report = _run_blocked(args)

    blockers = "\n".join(report["blockers"])
    assert report["decision"] == "BLOCKED_MODEL_NATIVE_SEQ513_SMOKE_READINESS"
    assert report["training_allowed"] is False
    assert "model-native smoke manifest proves post-rebuild orchestration provenance" in blockers


def test_model_native_seq513_smoke_readiness_fails_closed_on_stale_manifest_hash(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path)
    readiness_path = Path(args.smoke_manifest_event_json)
    payload = json.loads(readiness_path.read_text(encoding="utf-8"))
    payload["manifest_sha256"] = "0" * 64
    _write_json(readiness_path, payload)
    monkeypatch.setattr(readiness, "_git_status_short", lambda repo: [])

    report = _run_blocked(args)

    blockers = "\n".join(report["blockers"])
    assert report["decision"] == "BLOCKED_MODEL_NATIVE_SEQ513_SMOKE_READINESS"
    assert report["training_allowed"] is False
    assert "model-native smoke manifest event hash-binds its embedded manifest" in blockers


def test_model_native_seq513_smoke_readiness_fails_closed_on_split_artifact_hash_mismatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path)
    event = json.loads(Path(args.smoke_manifest_event_json).read_text(encoding="utf-8"))
    manifest = event["smoke_manifest"]
    train_parquet = Path(manifest["splits"]["train"]["out_parquet"])
    train_parquet.write_bytes(b"changed after manifest")
    monkeypatch.setattr(readiness, "_git_status_short", lambda repo: [])

    report = _run_blocked(args)

    blockers = "\n".join(report["blockers"])
    assert report["decision"] == "BLOCKED_MODEL_NATIVE_SEQ513_SMOKE_READINESS"
    assert report["training_allowed"] is False
    assert "smart smoke split artifact files exist and hashes match manifest" in blockers


def test_model_native_seq513_smoke_readiness_fails_closed_on_stale_split_schema(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path)
    manifest_path = Path(args.smoke_manifest_event_json)
    event = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = event["smoke_manifest"]
    manifest["splits"]["train"]["split_manifest_schema_version"] = "entry_foundation_smoke_split_manifest_v1"
    event["smoke_manifest"] = manifest
    event["manifest_sha256"] = readiness._sha256_json(manifest)
    _write_json(manifest_path, event)
    monkeypatch.setattr(readiness, "_git_status_short", lambda repo: [])

    report = _run_blocked(args)

    blockers = "\n".join(report["blockers"])
    assert report["decision"] == "BLOCKED_MODEL_NATIVE_SEQ513_SMOKE_READINESS"
    assert report["training_allowed"] is False
    assert "smart smoke split manifests pin model-native seq513 split schema" in blockers


def test_parser_and_source_require_explicit_evidence_and_publish_one_event() -> None:
    parser = readiness.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    assert "_latest.json" not in parser.format_help()

    source = Path(readiness.__file__).read_text(encoding="utf-8")
    assert source.count("write_immutable_json_event(") == 1
    assert "replace_latest_json_mirror" not in source
    assert ".md\"" not in source
    assert "smart_seq520" not in source.lower()
    assert "ENTRY_HIER_POCKET" not in source
    assert "ENTRY_TRENDLINE_RAIL_WRONG" not in source
    assert "fail-on-not-ready" not in source
