import argparse
import json
from pathlib import Path

from gx1.scripts import verify_entry_smart_seq520_smoke_readiness_v1 as readiness


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _model_contract() -> dict:
    return json.loads(json.dumps(readiness.EXPECTED_MODEL_CONTRACT))


def _build_fixture(tmp_path: Path) -> argparse.Namespace:
    smart_dataset_dir = tmp_path / "v10_dataset_smart_candidate_20260630"
    smart_smoke_dataset_dir = tmp_path / "v10_dataset_smart_seq520_smoke_20260630"
    smart_dataset_dir.mkdir()
    smart_smoke_dataset_dir.mkdir()

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
    _write_json(
        tmp_path / "smoke_manifest.json",
        {
            "schema_version": "entry_smart_seq520_smoke_dataset_v1",
            "manifest_variant": "smart_seq520_candidate",
            "expected_seq_snap_width": 520,
            "out_dir": str(smart_smoke_dataset_dir),
            "splits": {
                split: {
                    "rows": 16,
                    "out_parquet_sha256": "a" * 64,
                    "out_manifest_sha256": "b" * 64,
                }
                for split in ("train", "val", "test")
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
        smart_smoke_dataset_manifest_json=str(tmp_path / "smoke_manifest.json"),
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
    assert report["smart_smoke_training_allowed_after_explicit_vedtak_and_gates"] is True
    assert report["execution_allowed_now"] is False
    assert report["control_surface_mutated"] is False
    assert not any(report["side_effects_started"].values())

    train_contract = report["future_command_contracts"]["smart_smoke_train"]
    assert train_contract["implemented_in_control_surface"] is False
    assert train_contract["requires_clean_git"] is True
    assert train_contract["requires_ram_cap"] is True
    assert train_contract["ram_cap_runner"] == "scripts/gx1_capped_run.sh"
    assert train_contract["num_workers"] == 0
    assert train_contract["requires_edge_audit"] is True
    assert "--require-edge" in train_contract["post_smoke_audit_argv_template"]
    assert train_contract["specialist_contract_mode"] == "smart_seq520_candidate"
    assert train_contract["expected_signal_dim"] == 520
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
