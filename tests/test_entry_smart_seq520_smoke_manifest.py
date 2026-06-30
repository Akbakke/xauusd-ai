import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from gx1.scripts import materialize_entry_smart_seq520_smoke_manifest_v1 as manifest_gate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_split(dataset_dir: Path, split: str, *, width: int = 520, write_manifest: bool = True) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    parquet = dataset_dir / f"v10_smart_seq520_smoke__HOLD_03B_{split}.parquet"
    seq_values = [
        [[float(row + col + tick) for col in range(width)] for tick in range(2)]
        for row in range(3)
    ]
    snap_values = [[float(row + col + 1) for col in range(width)] for row in range(3)]
    table = pa.table(
        {
            "seq": pa.array(seq_values, type=pa.list_(pa.list_(pa.float32()))),
            "snap": pa.array(snap_values, type=pa.list_(pa.float32())),
            "y_direction": pa.array([0, 1, 2], type=pa.int64()),
        }
    )
    pq.write_table(table, parquet)
    if write_manifest:
        fields = [f"signal_{idx:03d}" for idx in range(width)]
        _write_json(
            parquet.with_suffix(".manifest.json"),
            {
                "schema_version": "entry_smart_seq520_smoke_split_manifest_v1",
                "manifest_variant": "smart_seq520_candidate",
                "expected_seq_snap_width": width,
                "output_data_path": str(parquet),
                "extra": {
                    "signal_bridge": {
                        "fields": fields,
                        "seq_input_dim": width,
                        "snap_input_dim": width,
                    }
                },
            },
        )


def _build_dataset(tmp_path: Path, *, missing_manifest_split: str | None = None, width: int = 520) -> Path:
    dataset_dir = tmp_path / "v10_dataset_smart_seq520_smoke_20260630"
    for split in ("train", "val", "test"):
        _write_split(dataset_dir, split, width=width, write_manifest=split != missing_manifest_split)
    return dataset_dir


def _args(tmp_path: Path, dataset_dir: Path, *, vedtak_id: str = "SMART_SEQ520_SMOKE_PYTEST") -> argparse.Namespace:
    return argparse.Namespace(
        smart_smoke_dataset_dir=str(dataset_dir),
        out_dir=str(tmp_path / "reports"),
        vedtak_id=vedtak_id,
        memory_cap="22G",
        swap_cap="2G",
        sample_rows=2,
        batch_size=2,
        fail_on_not_ready=False,
        quiet=True,
    )


def test_materialize_smart_seq520_smoke_manifest_report_only(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)

    report = manifest_gate.run(_args(tmp_path, dataset_dir))

    assert report["decision"] == "READY_FOR_SMART_SEQ520_SMOKE_MANIFEST_REVIEW"
    assert report["report_only"] is True
    assert report["manifest_written"] is True
    assert report["manifest_variant"] == "smart_seq520_candidate"
    assert report["expected_seq_snap_width"] == 520
    assert report["training_allowed"] is False
    assert report["replay_allowed"] is False
    assert report["iql_allowed"] is False
    assert report["shadow_live_allowed"] is False
    assert report["control_surface_mutated"] is False
    assert report["mutations_outside_report_dir"] is False
    assert not any(report["side_effects_started"].values())

    smoke_manifest = json.loads(Path(report["manifest_path"]).read_text(encoding="utf-8"))
    assert smoke_manifest["schema_version"] == "entry_smart_seq520_smoke_dataset_v1"
    assert smoke_manifest["explicit_vedtak_id"] == "SMART_SEQ520_SMOKE_PYTEST"
    assert set(smoke_manifest["splits"]) == {"train", "val", "test"}
    for split in ("train", "val", "test"):
        row = smoke_manifest["splits"][split]
        assert row["rows"] == 3
        assert row["seq_input_dim"] == 520
        assert row["snap_input_dim"] == 520
        assert row["field_count"] == 520
        assert len(row["out_parquet_sha256"]) == 64
        assert len(row["out_manifest_sha256"]) == 64

    train_contract = report["future_command_contracts"]["smart_smoke_train"]
    assert train_contract["requires_explicit_vedtak"] is True
    assert train_contract["explicit_vedtak_id"] == "SMART_SEQ520_SMOKE_PYTEST"
    assert train_contract["implemented_in_control_surface"] is False
    assert train_contract["requires_trainer_surface_enablement"] is True
    assert train_contract["specialist_contract_mode"] == "smart_seq520_candidate"
    assert "<" not in " ".join(train_contract["argv_template"])
    assert train_contract["requires_ram_cap"] is True
    assert train_contract["ram_cap_runner"] == "scripts/gx1_capped_run.sh"
    assert train_contract["num_workers"] == 0
    assert train_contract["started_by_this_report"] is False
    assert train_contract["starts_replay"] is False
    assert train_contract["starts_iql_distillation"] is False
    assert train_contract["touches_shadow_or_live"] is False


def test_smart_seq520_smoke_manifest_fails_closed_when_split_manifest_missing(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path, missing_manifest_split="val")

    report = manifest_gate.run(_args(tmp_path, dataset_dir))

    assert report["decision"] == "BLOCKED_SMART_SEQ520_SMOKE_MANIFEST_READINESS"
    assert report["manifest_written"] is False
    assert not Path(report["manifest_path"]).exists()
    assert "exact train val test split manifests exist" in report["blockers"]
    assert report["training_allowed"] is False
    assert not any(report["side_effects_started"].values())


def test_smart_seq520_smoke_manifest_requires_explicit_vedtak_id(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)

    report = manifest_gate.run(_args(tmp_path, dataset_dir, vedtak_id="<SMART_SEQ520_SMOKE_VEDTAK_ID>"))

    assert report["decision"] == "BLOCKED_SMART_SEQ520_SMOKE_MANIFEST_READINESS"
    assert report["manifest_written"] is False
    assert report["explicit_vedtak_id"] is None
    assert "explicit smart seq520 smoke vedtak id is provided" in report["blockers"]
    assert not Path(report["manifest_path"]).exists()


def test_smart_seq520_smoke_manifest_blocks_wrong_width(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path, width=519)

    report = manifest_gate.run(_args(tmp_path, dataset_dir))

    assert report["decision"] == "BLOCKED_SMART_SEQ520_SMOKE_MANIFEST_READINESS"
    assert report["manifest_written"] is False
    assert "split signal_bridge seq and snap dims are 520" in report["blockers"]
    assert "split parquet seq and snap samples have width 520" in report["blockers"]
