import argparse
import hashlib
import json
from pathlib import Path

from gx1.scripts import verify_entry_smart_seq520_smoke_readiness_v1 as readiness


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_contract() -> dict:
    return json.loads(json.dumps(readiness.EXPECTED_MODEL_CONTRACT))


def _build_fixture(tmp_path: Path) -> argparse.Namespace:
    smart_dataset_dir = tmp_path / "v10_dataset_smart_candidate_20260630"
    smart_smoke_dataset_dir = tmp_path / "v10_dataset_smart_seq520_smoke_20260630"
    smart_dataset_dir.mkdir()
    smart_smoke_dataset_dir.mkdir()
    split_artifacts: dict[str, dict[str, str]] = {}
    for split in ("train", "val", "test"):
        parquet_path = smart_smoke_dataset_dir / f"v10_smart_seq520_smoke__HOLD_03B_{split}.parquet"
        manifest_path = smart_smoke_dataset_dir / f"v10_smart_seq520_smoke__HOLD_03B_{split}.manifest.json"
        parquet_path.write_bytes(f"{split}-parquet".encode("utf-8"))
        manifest_path.write_text(f'{{"split":"{split}"}}\n', encoding="utf-8")
        split_artifacts[split] = {
            "out_parquet": str(parquet_path),
            "out_manifest": str(manifest_path),
            "out_parquet_sha256": _sha256(parquet_path),
            "out_manifest_sha256": _sha256(manifest_path),
        }

    _write_json(
        tmp_path / "smart_rebuild_preflight.json",
        {
            "decision": "READY_FOR_SMART_REBUILD_VEDTAK_REVIEW",
            "report_only": True,
            "training_allowed": False,
            "dataset_rebuild_allowed_without_vedtak": False,
            "side_effects_started": {
                "dataset_rebuild": False,
                "training": False,
                "replay": False,
                "iql_distillation": False,
                "shadow": False,
                "live": False,
            },
            "counts": {
                "manifest_variant": "smart_seq520_candidate",
                "expected_seq_snap_width": 520,
            },
        },
    )
    _write_json(
        tmp_path / "feature_audit.json",
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
                    "source_field": "chart.foundation_hh_state",
                    "observed": True,
                    "nonfinite_count": 0,
                    "near_constant": False,
                }
            ],
        },
    )
    _write_json(
        tmp_path / "target_audit.json",
        {
            "decision": "PASS",
            "dataset_dir": str(smart_dataset_dir),
            "failures": [],
            "target_head_contract": {
                "active_training_heads": list(readiness.SPECIALIST_FUSION_ACTIVE_HEADS),
                "blocked_heads": list(readiness.SPECIALIST_FUSION_BLOCKED_HEADS),
            },
        },
    )
    specialist_indices = {
        name: [20 + idx, 120 + idx]
        for idx, name in enumerate(readiness.REQUIRED_SPECIALISTS)
    }
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
        for split in ("train", "val", "test")
        for name in readiness.REQUIRED_SPECIALISTS
    ]
    _write_json(
        tmp_path / "specialist_audit.json",
        {
            "decision": "PASS",
            "dataset_dir": str(smart_dataset_dir),
            "failures": [],
            "contract_mode": "smart_seq520_candidate",
            "signal_field_count": 520,
            "selected_feature_count": 479,
            "required_training_specialists": list(readiness.REQUIRED_SPECIALISTS),
            "specialist_model_contract": _model_contract(),
            "specialist_model_contract_valid": True,
            "specialist_model_contract_failures": [],
            "signal_routing_all_mapped": True,
            "signal_unmapped_count": 0,
            "context_routing_all_mapped": True,
            "context_routing_unmapped_count": 0,
            "specialist_input_liveness_all_live": True,
            "specialist_input_liveness": specialist_liveness,
            "architecture_contract": {
                "input_dim": 520,
                "specialist_input_indices": specialist_indices,
                "recommended_fusion": {
                    "active_heads": list(readiness.SPECIALIST_FUSION_ACTIVE_HEADS),
                    "blocked_heads": list(readiness.SPECIALIST_FUSION_BLOCKED_HEADS),
                },
            },
        },
    )
    smoke_manifest_path = tmp_path / "smoke_manifest.json"
    _write_json(
        smoke_manifest_path,
        {
            "schema_version": "entry_smart_seq520_smoke_dataset_v1",
            "manifest_variant": "smart_seq520_candidate",
            "expected_seq_snap_width": 520,
            "out_dir": str(smart_smoke_dataset_dir),
            "splits": {
                split: {
                    "rows": 16,
                    **split_artifacts[split],
                    "split_manifest_schema_version": "entry_smart_seq520_smoke_split_manifest_v1",
                }
                for split in ("train", "val", "test")
            },
        },
    )
    _write_json(
        tmp_path / "smoke_manifest_readiness.json",
        {
            "schema_version": "entry_smart_seq520_smoke_manifest_readiness_v1",
            "decision": "READY_FOR_SMART_SEQ520_SMOKE_MANIFEST_REVIEW",
            "report_only": True,
            "manifest_written": True,
            "manifest_path": str(smoke_manifest_path),
            "manifest_sha256": _sha256(smoke_manifest_path),
            "side_effects_started": {
                "dataset_rebuild": False,
                "training": False,
                "replay": False,
                "iql_distillation": False,
                "shadow": False,
                "live": False,
            },
        },
    )
    return argparse.Namespace(
        smart_rebuild_preflight_json=str(tmp_path / "smart_rebuild_preflight.json"),
        smart_dataset_dir=str(smart_dataset_dir),
        smart_smoke_dataset_dir=str(smart_smoke_dataset_dir),
        smart_feature_audit_json=str(tmp_path / "feature_audit.json"),
        smart_target_audit_json=str(tmp_path / "target_audit.json"),
        smart_specialist_audit_json=str(tmp_path / "specialist_audit.json"),
        smart_smoke_dataset_manifest_json=str(smoke_manifest_path),
        smart_smoke_dataset_manifest_readiness_json=str(tmp_path / "smoke_manifest_readiness.json"),
        repo_dir=str(tmp_path),
        out_dir=str(tmp_path / "reports"),
        memory_cap="22G",
        swap_cap="2G",
        quiet=True,
        no_fail_on_not_ready=True,
    )


def test_smart_seq520_smoke_readiness_passes_as_report_only(monkeypatch, tmp_path: Path) -> None:
    args = _build_fixture(tmp_path)
    monkeypatch.setattr(readiness, "_git_status_short", lambda repo: [])

    report = readiness.run(args)

    assert report["decision"] == "READY_FOR_SMART_SEQ520_SMOKE_MANIFEST_REVIEW"
    assert report["report_only"] is True
    assert report["training_allowed"] is False
    assert report["smart_smoke_training_allowed_without_vedtak"] is False
    assert report["smart_smoke_training_allowed_after_explicit_vedtak_and_gates"] is False
    assert report["smart_trainability_readiness_required_before_training"] is True
    assert report["execution_allowed_now"] is False
    assert report["control_surface_mutated"] is False
    assert not any(report["side_effects_started"].values())

    train_contract = report["future_command_contracts"]["smart_smoke_train"]
    assert train_contract["implemented_in_control_surface"] is True
    assert train_contract["execution_allowed_now"] is False
    assert train_contract["requires_clean_git"] is True
    assert train_contract["requires_ram_cap"] is True
    assert train_contract["ram_cap_runner"] == "scripts/gx1_capped_run.sh"
    assert train_contract["num_workers"] == 0
    assert train_contract["requires_edge_audit"] is True
    assert "--require-edge" in train_contract["post_smoke_audit_argv_template"]
    assert train_contract["specialist_contract_mode"] == "smart_seq520_candidate"
    assert train_contract["expected_signal_dim"] == 520
    assert train_contract["requires_path_calibration_recipe_contract"] is True
    assert train_contract["path_calibration_recipe_contract"] == readiness.PATH_CALIBRATION_RECIPE_CONTRACT
    assert train_contract["path_calibration_env_template"] == readiness.PATH_CALIBRATION_ENV_TEMPLATE
    assert train_contract["requires_direction_balance_recipe_contract"] is True
    assert train_contract["direction_balance_recipe_contract"] == readiness.DIRECTION_BALANCE_RECIPE_CONTRACT
    assert train_contract["direction_balance_env_template"] == readiness.DIRECTION_BALANCE_ENV_TEMPLATE
    assert train_contract["requires_direction_context_slice_contract"] is True
    assert train_contract["direction_context_slice_contract"] == readiness.DIRECTION_CONTEXT_SLICE_CONTRACT
    train_argv = " ".join(train_contract["inner_train_argv_template"])
    for key, value in readiness.PATH_CALIBRATION_ENV_TEMPLATE.items():
        assert f"{key}={value}" in train_argv
    for key, value in readiness.DIRECTION_BALANCE_ENV_TEMPLATE.items():
        assert f"{key}={value}" in train_argv
    assert Path(report["json_path"]).exists()


def test_smart_seq520_smoke_readiness_fails_closed_on_dirty_git_and_wrong_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path)
    specialist_path = Path(args.smart_specialist_audit_json)
    specialist = json.loads(specialist_path.read_text(encoding="utf-8"))
    specialist["contract_mode"] = "foundation_seq146"
    specialist["signal_field_count"] = 519
    _write_json(specialist_path, specialist)
    monkeypatch.setattr(readiness, "_git_status_short", lambda repo: [" M gx1/example.py"])

    report = readiness.run(args)

    blockers = "\n".join(report["blockers"])
    assert report["decision"] == "BLOCKED_SMART_SEQ520_SMOKE_READINESS"
    assert report["training_allowed"] is False
    assert report["smart_smoke_training_allowed_after_explicit_vedtak_and_gates"] is False
    assert "clean git required before smart smoke train" in blockers
    assert "smart specialist audit uses smart_seq520_candidate contract mode" in blockers
    assert "smart specialist audit has exact smart signal width" in blockers
    assert "trainer loader accepts exact smart specialist contract" in blockers


def test_smart_seq520_smoke_readiness_fails_closed_on_nonfinite_liveness(
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

    report = readiness.run(args)

    blockers = "\n".join(report["blockers"])
    assert report["decision"] == "BLOCKED_SMART_SEQ520_SMOKE_READINESS"
    assert report["training_allowed"] is False
    assert "smart feature audit proves finite live features" in blockers
    assert "smart specialist input has no NaN inf or liveness collapse" in blockers


def test_smart_seq520_smoke_readiness_fails_closed_on_blocked_manifest_readiness(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path)
    readiness_path = Path(args.smart_smoke_dataset_manifest_readiness_json)
    payload = json.loads(readiness_path.read_text(encoding="utf-8"))
    payload["decision"] = "BLOCKED_SMART_SEQ520_SMOKE_MANIFEST_READINESS"
    payload["manifest_written"] = False
    _write_json(readiness_path, payload)
    monkeypatch.setattr(readiness, "_git_status_short", lambda repo: [])

    report = readiness.run(args)

    blockers = "\n".join(report["blockers"])
    assert report["decision"] == "BLOCKED_SMART_SEQ520_SMOKE_READINESS"
    assert report["training_allowed"] is False
    assert "latest smart smoke manifest readiness report is ready" in blockers


def test_smart_seq520_smoke_readiness_fails_closed_on_stale_manifest_hash(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path)
    readiness_path = Path(args.smart_smoke_dataset_manifest_readiness_json)
    payload = json.loads(readiness_path.read_text(encoding="utf-8"))
    payload["manifest_sha256"] = "0" * 64
    _write_json(readiness_path, payload)
    monkeypatch.setattr(readiness, "_git_status_short", lambda repo: [])

    report = readiness.run(args)

    blockers = "\n".join(report["blockers"])
    assert report["decision"] == "BLOCKED_SMART_SEQ520_SMOKE_READINESS"
    assert report["training_allowed"] is False
    assert "latest smart smoke manifest readiness points at this manifest" in blockers


def test_smart_seq520_smoke_readiness_fails_closed_on_split_artifact_hash_mismatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path)
    manifest = json.loads(Path(args.smart_smoke_dataset_manifest_json).read_text(encoding="utf-8"))
    train_parquet = Path(manifest["splits"]["train"]["out_parquet"])
    train_parquet.write_bytes(b"changed after manifest")
    monkeypatch.setattr(readiness, "_git_status_short", lambda repo: [])

    report = readiness.run(args)

    blockers = "\n".join(report["blockers"])
    assert report["decision"] == "BLOCKED_SMART_SEQ520_SMOKE_READINESS"
    assert report["training_allowed"] is False
    assert "smart smoke split artifact files exist and hashes match manifest" in blockers


def test_smart_seq520_smoke_readiness_fails_closed_on_stale_split_schema(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path)
    manifest_path = Path(args.smart_smoke_dataset_manifest_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["splits"]["train"]["split_manifest_schema_version"] = "entry_foundation_smoke_split_manifest_v1"
    _write_json(manifest_path, manifest)
    readiness_path = Path(args.smart_smoke_dataset_manifest_readiness_json)
    readiness_payload = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness_payload["manifest_sha256"] = _sha256(manifest_path)
    _write_json(readiness_path, readiness_payload)
    monkeypatch.setattr(readiness, "_git_status_short", lambda repo: [])

    report = readiness.run(args)

    blockers = "\n".join(report["blockers"])
    assert report["decision"] == "BLOCKED_SMART_SEQ520_SMOKE_READINESS"
    assert report["training_allowed"] is False
    assert "smart smoke split manifests pin smart seq520 split schema" in blockers
