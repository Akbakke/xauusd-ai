import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gx1.scripts.materialize_entry_foundation_smoke_dataset_v1 import run


def _write_audit(path: Path, *, schema: str, decision: str = "PASS") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": schema,
                "decision": decision,
                "dataset_dir": "/tmp/foundation",
                "created_utc": "2026-06-29T00:00:00+00:00",
                "json_path": str(path),
            }
        ),
        encoding="utf-8",
    )


def _write_split(root: Path, split: str, labels: list[int]) -> None:
    path = root / f"sample_{split}.parquet"
    table = pa.table(
        {
            "time": pa.array(pd.date_range("2026-01-01", periods=len(labels), freq="5min", tz="UTC")),
            "y_direction": pa.array(labels, type=pa.int64()),
            "value": pa.array([float(i) for i in range(len(labels))], type=pa.float32()),
        }
    )
    pq.write_table(table, path)
    manifest = {
        "output_data_path": str(path),
        "extra": {
            "rows": len(labels),
            "signal_bridge": {
                "fields": ["p_long", "p_short", "p_flat"],
                "seq_input_dim": 3,
                "snap_input_dim": 3,
            },
        },
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_split_with_snap(root: Path, split: str, labels: list[int], rare_values: list[float]) -> None:
    path = root / f"sample_{split}.parquet"
    flat = []
    for value in rare_values:
        flat.extend([0.0, float(value)])
    table = pa.table(
        {
            "time": pa.array(pd.date_range("2026-01-01", periods=len(labels), freq="5min", tz="UTC")),
            "y_direction": pa.array(labels, type=pa.int64()),
            "snap": pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), 2),
        }
    )
    pq.write_table(table, path)
    manifest = {
        "output_data_path": str(path),
        "extra": {
            "rows": len(labels),
            "signal_bridge": {
                "fields": ["neutral", "rare_feature"],
                "seq_input_dim": 2,
                "snap_input_dim": 2,
            },
        },
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_materialize_entry_foundation_smoke_dataset_rewrites_manifests(tmp_path: Path) -> None:
    source = tmp_path / "source"
    out = tmp_path / "out"
    audits = tmp_path / "audits"
    source.mkdir()
    audits.mkdir()
    feature_audit = audits / "feature.json"
    target_audit = audits / "target.json"
    specialist_audit = audits / "specialist.json"
    _write_audit(feature_audit, schema="entry_feature_foundation_audit_v1")
    _write_audit(target_audit, schema="entry_target_foundation_audit_v1")
    _write_audit(specialist_audit, schema="entry_specialist_feature_group_audit_v1")
    for split in ("train", "val", "test"):
        _write_split(source, split, [0, 1, 2, 0, 1, 2, 0, 1, 2])

    report = run(
        argparse.Namespace(
            source_dir=str(source),
            out_dir=str(out),
            stem="smoke",
            train_rows=6,
            val_rows=6,
            test_rows=6,
            batch_size=4,
            feature_audit_json=str(feature_audit),
            target_audit_json=str(target_audit),
            specialist_audit_json=str(specialist_audit),
            split_schema_version="",
            quiet=True,
        )
    )

    assert report["splits"]["train"]["rows"] == 6
    assert report["splits"]["train"]["label_counts"] == {"0": 2, "1": 2, "2": 2}
    assert (out / "smoke_train.parquet").exists()
    split_manifest = json.loads((out / "smoke_train.manifest.json").read_text(encoding="utf-8"))
    assert split_manifest["output_data_path"] == str(out / "smoke_train.parquet")
    assert split_manifest["extra"]["rows"] == 6
    assert split_manifest["foundation_smoke_dataset_v1"]["source_rows"] == 9
    assert report["splits"]["train"]["source_manifest"] == str(source / "sample_train.manifest.json")
    assert len(report["splits"]["train"]["source_manifest_sha256"]) == 64
    assert len(report["splits"]["train"]["out_parquet_sha256"]) == 64
    assert len(report["splits"]["train"]["out_manifest_sha256"]) == 64
    manifest_path = out / "SMOKE_DATASET_MANIFEST.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = manifest["audit_provenance"]
    assert provenance["schema_version"] == "entry_foundation_smoke_dataset_audit_provenance_v1"
    assert provenance["all_artifacts_present"] is True
    assert provenance["all_artifact_hashes_present"] is True
    assert provenance["artifacts"]["feature_audit"]["path"] == str(feature_audit)
    assert len(provenance["artifacts"]["feature_audit"]["sha256"]) == 64


def test_materialize_entry_foundation_smoke_dataset_can_pin_split_schema(tmp_path: Path) -> None:
    source = tmp_path / "source"
    out = tmp_path / "out"
    audits = tmp_path / "audits"
    source.mkdir()
    audits.mkdir()
    feature_audit = audits / "feature.json"
    target_audit = audits / "target.json"
    specialist_audit = audits / "specialist.json"
    _write_audit(feature_audit, schema="entry_feature_foundation_audit_v1")
    _write_audit(target_audit, schema="entry_target_foundation_audit_v1")
    _write_audit(specialist_audit, schema="entry_specialist_feature_group_audit_v1")
    for split in ("train", "val", "test"):
        _write_split(source, split, [0, 1, 2])

    run(
        argparse.Namespace(
            source_dir=str(source),
            out_dir=str(out),
            stem="v10_smart_seq520_smoke__HOLD_03B",
            train_rows=3,
            val_rows=3,
            test_rows=3,
            batch_size=4,
            feature_audit_json=str(feature_audit),
            target_audit_json=str(target_audit),
            specialist_audit_json=str(specialist_audit),
            schema_version="entry_smart_seq520_smoke_dataset_v1",
            split_schema_version="entry_smart_seq520_smoke_split_manifest_v1",
            manifest_variant="smart_seq520_candidate",
            expected_seq_snap_width=520,
            quiet=True,
        )
    )

    split_manifest = json.loads(
        (out / "v10_smart_seq520_smoke__HOLD_03B_train.manifest.json").read_text(encoding="utf-8")
    )
    assert split_manifest["schema_version"] == "entry_smart_seq520_smoke_split_manifest_v1"
    assert split_manifest["manifest_variant"] == "smart_seq520_candidate"
    assert split_manifest["expected_seq_snap_width"] == 520
    assert split_manifest["output_data_path"] == str(out / "v10_smart_seq520_smoke__HOLD_03B_train.parquet")
    manifest = json.loads((out / "SMOKE_DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "entry_smart_seq520_smoke_dataset_v1"
    assert manifest["manifest_variant"] == "smart_seq520_candidate"
    assert manifest["expected_seq_snap_width"] == 520


def test_materialize_entry_foundation_smoke_dataset_can_force_extreme_snap_rows(tmp_path: Path) -> None:
    source = tmp_path / "source"
    out = tmp_path / "out"
    audits = tmp_path / "audits"
    source.mkdir()
    audits.mkdir()
    feature_audit = audits / "feature.json"
    target_audit = audits / "target.json"
    specialist_audit = audits / "specialist.json"
    _write_audit(feature_audit, schema="entry_feature_foundation_audit_v1")
    _write_audit(target_audit, schema="entry_target_foundation_audit_v1")
    _write_audit(specialist_audit, schema="entry_specialist_feature_group_audit_v1")
    labels = [0, 1, 2] * 10
    rare_values = [0.0] * 28 + [10.0, 20.0]
    for split in ("train", "val", "test"):
        _write_split_with_snap(source, split, labels, rare_values)

    report = run(
        argparse.Namespace(
            source_dir=str(source),
            out_dir=str(out),
            stem="smoke",
            train_rows=6,
            val_rows=6,
            test_rows=6,
            batch_size=4,
            extreme_snap_feature=["rare_feature"],
            extreme_snap_rows=2,
            feature_audit_json=str(feature_audit),
            target_audit_json=str(target_audit),
            specialist_audit_json=str(specialist_audit),
            split_schema_version="",
            quiet=True,
        )
    )

    assert report["splits"]["train"]["forced_extreme_snap_features"] == ["rare_feature"]
    assert report["splits"]["train"]["forced_extreme_snap_rows_selected"] == 2
    table = pq.read_table(out / "smoke_train.parquet", columns=["snap"])
    snap = table.column("snap").to_pylist()
    rare = [float(row[1]) for row in snap]
    assert max(rare) == 20.0
    assert 10.0 in rare
