import argparse
import json
from pathlib import Path

from gx1.features.entry_foundation_structure_v1 import (
    FOUNDATION_STRUCTURE_FEATURE_NAMES,
    FOUNDATION_STRUCTURE_FEATURE_VERSION,
)
from gx1.scripts.verify_entry_foundation_adoption_candidate_v1 import run
from gx1.scripts.verify_entry_training_readiness_v1 import REQUIRED_SPECIALISTS

REPO = Path("/home/andre2/src/GX1_ENGINE")


def _sha(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _write_split(root: Path, split: str) -> None:
    parquet = root / f"candidate_{split}.parquet"
    parquet.write_bytes(f"{split}-parquet".encode("utf-8"))
    fields = [f"f{i}" for i in range(146)]
    manifest = {
        "extra": {
            "signal_bridge": {
                "fields": fields,
                "seq_input_dim": 146,
                "snap_input_dim": 146,
                "seq_structure_extension_dim": 105,
                "seq_structure_extension_v1": {
                    "foundation_structure_feature_version": FOUNDATION_STRUCTURE_FEATURE_VERSION,
                    "foundation_structure_feature_count": len(FOUNDATION_STRUCTURE_FEATURE_NAMES),
                    "foundation_structure_missing_feature_count": 0,
                    "foundation_structure_all_required_selected": True,
                },
            }
        }
    }
    parquet.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_audits(root: Path, dataset: Path) -> dict[str, Path]:
    audits = root / "audits"
    audits.mkdir()
    feature = _write_json(
        audits / "feature.json",
        {
            "schema_version": "entry_feature_foundation_audit_v1",
            "decision": "PASS",
            "failures": [],
            "dataset_dir": str(dataset),
            "foundation_structure_feature_version": FOUNDATION_STRUCTURE_FEATURE_VERSION,
            "foundation_missing_from_manifest_count": 0,
            "manifest_foundation_all_required_selected": True,
            "foundation_objective_coverage_all_present": True,
            "foundation_objective_liveness_all_live": True,
            "foundation_source_field_liveness_all_live": True,
        },
    )
    target = _write_json(
        audits / "target.json",
        {
            "schema_version": "entry_target_foundation_audit_v1",
            "decision": "PASS",
            "failures": [],
            "dataset_dir": str(dataset),
            "target_head_contract": {"active_training_heads": ["direction"]},
        },
    )
    specialist = _write_json(
        audits / "specialist.json",
        {
            "schema_version": "entry_specialist_feature_group_audit_v1",
            "decision": "PASS",
            "failures": [],
            "dataset_dir": str(dataset),
            "signal_field_count": 146,
            "selected_feature_count": 105,
            "required_training_specialists": list(REQUIRED_SPECIALISTS),
            "specialist_input_liveness_all_live": True,
            "foundation_objective_routing_all_present_and_expected": True,
        },
    )
    return {"feature_audit": feature, "target_audit": target, "specialist_audit": specialist}


def _write_smoke(root: Path, dataset: Path, audit_paths: dict[str, Path]) -> Path:
    smoke = root / "smoke"
    smoke.mkdir()
    splits = {}
    for split in ("train", "val", "test"):
        out_parquet = smoke / f"smoke_{split}.parquet"
        out_manifest = smoke / f"smoke_{split}.manifest.json"
        out_parquet.write_bytes(f"{split}-smoke-parquet".encode("utf-8"))
        out_manifest.write_text(json.dumps({"split": split}), encoding="utf-8")
        source_manifest = dataset / f"candidate_{split}.manifest.json"
        splits[split] = {
            "rows": 6,
            "label_counts": {"0": 2, "1": 2, "2": 2},
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": _sha(source_manifest),
            "out_path": str(out_parquet),
            "out_parquet_sha256": _sha(out_parquet),
            "out_manifest": str(out_manifest),
            "out_manifest_sha256": _sha(out_manifest),
        }
    manifest = {
        "schema_version": "entry_foundation_seq146_smoke_dataset_v1",
        "source_dir": str(dataset),
        "splits": splits,
        "audit_provenance": {
            "schema_version": "entry_foundation_smoke_dataset_audit_provenance_v1",
            "artifacts": {
                name: {
                    "path": str(path),
                    "exists": True,
                    "sha256": _sha(path),
                }
                for name, path in audit_paths.items()
            },
        },
    }
    (smoke / "SMOKE_DATASET_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return smoke


def _args(tmp_path: Path, dataset: Path, audits: dict[str, Path], smoke: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_dir=str(dataset),
        feature_audit_json=str(audits["feature_audit"]),
        target_audit_json=str(audits["target_audit"]),
        specialist_audit_json=str(audits["specialist_audit"]),
        smoke_dataset_dir=str(smoke),
        out_dir=str(tmp_path / "reports"),
        expected_smoke_train_rows=6,
        expected_smoke_val_rows=6,
        expected_smoke_test_rows=6,
        quiet=True,
    )


def test_entry_foundation_adoption_candidate_passes_for_consistent_bundle(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for split in ("train", "val", "test"):
        _write_split(dataset, split)
    audits = _write_audits(tmp_path, dataset)
    smoke = _write_smoke(tmp_path, dataset, audits)

    report = run(_args(tmp_path, dataset, audits, smoke))

    assert report["decision"] == "PASS"
    assert report["candidate_ready_for_activation"] is True
    assert report["training_allowed"] is False
    assert report["activation_allowed_without_vedtak"] is False
    assert report["failures"] == []
    assert Path(report["json_path"]).exists()


def test_entry_foundation_adoption_candidate_fails_on_audit_dataset_mismatch(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for split in ("train", "val", "test"):
        _write_split(dataset, split)
    audits = _write_audits(tmp_path, dataset)
    feature = json.loads(audits["feature_audit"].read_text(encoding="utf-8"))
    feature["dataset_dir"] = str(tmp_path / "other_dataset")
    audits["feature_audit"].write_text(json.dumps(feature), encoding="utf-8")
    smoke = _write_smoke(tmp_path, dataset, audits)

    report = run(_args(tmp_path, dataset, audits, smoke))

    assert report["decision"] == "NOT_READY"
    assert report["candidate_ready_for_activation"] is False
    assert any(
        row["gate"] == "feature_audit" and row["check"] == "feature audit points at candidate dataset"
        for row in report["failures"]
    )


def test_foundation_state_allows_adoption_candidate_report_root() -> None:
    verifier = (REPO / "gx1/scripts/verify_entry_foundation_state_v1.py").read_text(encoding="utf-8")

    assert "entry_foundation_adoption_candidate_20260629_v1" in verifier
    assert "entry_foundation_activation_plan_20260629_v1" in verifier
    assert "entry_foundation_activation_apply_20260629_v1" in verifier
