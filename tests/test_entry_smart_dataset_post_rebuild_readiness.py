import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gx1.scripts import audit_entry_smart_dataset_post_rebuild_readiness_v1 as gate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _smart_fields(width: int = 520) -> list[str]:
    fields = [
        "p_long",
        "p_short",
        "p_flat",
        "p_hat",
        "uncertainty_score",
        "margin_top1_top2",
        "entropy",
    ]
    templates = (
        "foundation_hh_state_{i:03d}",
        "foundation_sweep_reclaim_{i:03d}",
        "ema_trend_feature_{i:03d}",
        "atr_vol_feature_{i:03d}",
        "momentum.flow_feature_{i:03d}",
        "session_regime_feature_{i:03d}",
        "chart.geometry_feature_{i:03d}",
        "candle.pattern_feature_{i:03d}",
    )
    i = 0
    while len(fields) < width:
        fields.append(templates[i % len(templates)].format(i=i))
        i += 1
    return fields


def _write_split_parquet(path: Path, *, split_offset: float, width: int, rows: int, nonfinite: bool = False) -> None:
    seq_rows = []
    snap_rows = []
    ctx_cont_rows = []
    ctx_cat_rows = []
    for row in range(rows):
        base = (np.arange(width, dtype=np.float32) * 0.001) + np.float32(split_offset + row * 0.1)
        seq = np.stack([base + np.float32(step * 0.0001) for step in range(96)]).astype(np.float32)
        snap = seq[-1].copy()
        if nonfinite and row == 1:
            snap[10] = np.inf
        seq_rows.append(seq.tolist())
        snap_rows.append(snap.tolist())
        ctx_cont_rows.append([float(row + 1), float(split_offset + row), float(row) / 10.0])
        ctx_cat_rows.append([row % 4, row % 3])
    table = pa.table(
        {
            "time": pa.array(pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")),
            "seq": pa.array(seq_rows, type=pa.list_(pa.list_(pa.float32()))),
            "snap": pa.array(snap_rows, type=pa.list_(pa.float32())),
            "ctx_cont": pa.array(ctx_cont_rows, type=pa.list_(pa.float32())),
            "ctx_cat": pa.array(ctx_cat_rows, type=pa.list_(pa.int64())),
            "y_direction": pa.array([row % 3 for row in range(rows)], type=pa.int64()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _build_fixture(tmp_path: Path, *, nonfinite_train_snap: bool = False, create_dataset: bool = True) -> argparse.Namespace:
    fields = _smart_fields()
    selected = fields[41:]
    source_file = tmp_path / "sources" / "foundation_manifest.json"
    source_code = tmp_path / "sources" / "smart_layer.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text('{"selected": true}\n', encoding="utf-8")
    source_code.write_text("SMART_LAYER_VERSION = 'fixture'\n", encoding="utf-8")
    source_parquet = tmp_path / "sources" / "canonical_v3_FULL_PLUS_CTX.parquet"
    source_parquet.write_bytes(b"source parquet placeholder")
    smart_manifest = tmp_path / "ENTRY_SPECIALIST_CHALLENGER_SMART_EXTENSION_MANIFEST_latest.json"
    _write_json(
        smart_manifest,
        {
            "schema_version": "entry_specialist_challenger_extension_manifest_v1",
            "manifest_variant": "smart_seq520_candidate",
            "base_signal_feature_count": 41,
            "expected_seq_snap_width": 520,
            "selected_feature_count": len(selected),
            "selected_features": selected,
            "source_manifests": {
                "foundation_sequence_extension": {
                    "label": "foundation_sequence_extension",
                    "path": str(source_file),
                    "sha256": _sha256(source_file),
                },
                "smart_candidate_layers": {
                    "trend_ema_smart_layer": {
                        "label": "trend_ema_smart_layer",
                        "path": str(source_code),
                        "sha256": _sha256(source_code),
                    }
                },
            },
            "dataset_rebuild_required_before_training": True,
            "training_allowed": False,
        },
    )
    smart_report = tmp_path / "ENTRY_SPECIALIST_CHALLENGER_SMART_EXTENSION_REPORT_latest.json"
    _write_json(
        smart_report,
        {
            "decision": "READY_FOR_SMART_CHALLENGER_DATASET_REBUILD_MANIFEST",
            "counts": {
                "base_signal_features": 41,
                "combined_selected_features": len(selected),
                "expected_seq_snap_width": 520,
            },
            "training_allowed": False,
        },
    )
    dataset_dir = tmp_path / "v10_dataset_smart_candidate_20260630"
    splits = {
        "train": {"start": "2025-01-01T00:00:00+00:00", "end": "2025-09-30T23:59:59+00:00"},
        "val": {"start": "2025-10-01T00:00:00+00:00", "end": "2025-11-30T23:59:59+00:00"},
        "test": {"start": "2025-12-01T00:00:00+00:00", "end": "2025-12-31T23:59:59+00:00"},
    }
    if create_dataset:
        for idx, split in enumerate(("train", "val", "test")):
            parquet = dataset_dir / f"v10_smart_seq520_candidate__HOLD_03B_{split}.parquet"
            _write_split_parquet(
                parquet,
                split_offset=float(idx + 1),
                width=len(fields),
                rows=5,
                nonfinite=bool(nonfinite_train_snap and split == "train"),
            )
            _write_json(
                parquet.with_suffix(".manifest.json"),
                {
                    "output_data_path": str(parquet),
                    "splits": splits,
                    "ts_min_max_by_split": {
                        split: {
                            "ts_min": "2026-01-01 00:00:00+00:00",
                            "ts_max": "2026-01-01 00:20:00+00:00",
                        }
                    },
                    "extra": {
                        "base28_manifest": {
                            "path": "/dev/null",
                            "parquet_path": str(source_parquet),
                            "parquet_sha256": _sha256(source_parquet),
                        },
                        "signal_bridge": {
                            "id": "XGB_SIGNAL_BRIDGE_V3",
                            "fields": fields,
                            "snap_fields": fields,
                            "seq_input_dim": len(fields),
                            "snap_input_dim": len(fields),
                            "base_seq_input_dim": 41,
                            "seq_structure_extension_dim": len(selected),
                            "neutral_xgb_bridge": True,
                            "seq_structure_extension_v1": {
                                "enabled": True,
                                "mode": "inline_from_merged3",
                                "features": selected,
                                "feature_count": len(selected),
                                "manifest_path": str(smart_manifest),
                                "manifest_selected_feature_count": len(selected),
                                "source_parquet_for_price_features": str(source_parquet),
                            },
                        },
                        "ctx_contract": {
                            "tag": "CTX_FIXTURE",
                            "ctx_cont_dim": 3,
                            "ctx_cat_dim": 2,
                            "ctx_cont_names": ["spread_bps", "atr_bps", "regime_stack_sum_v3"],
                            "ctx_cat_names": ["session_id", "spread_bucket"],
                        },
                    },
                },
            )
    return argparse.Namespace(
        dataset_dir=str(dataset_dir),
        smart_manifest=str(smart_manifest),
        smart_report=str(smart_report),
        out_dir=str(tmp_path / "reports"),
        contract_mode="smart_seq520_candidate",
        expected_seq_len=96,
        sample_rows=3,
        batch_size=2,
        fullscan=True,
        verify_source_parquet_hashes=True,
        min_live_features_per_specialist=1,
        min_live_feature_fraction=0.05,
        fail_on_not_ready=False,
        quiet=True,
    )


def test_smart_dataset_post_rebuild_readiness_passes_fullscan_fixture(tmp_path: Path) -> None:
    report = gate.run(_build_fixture(tmp_path))

    assert report["decision"] == gate.READY_DECISION
    assert report["expected_contract"]["expected_seq_snap_width"] == 520
    assert report["scan_policy"]["fullscan"] is True
    assert report["training_allowed"] is False
    assert report["replay_allowed"] is False
    assert report["iql_allowed"] is False
    assert not any(report["side_effects_started"].values())
    assert report["signal_routing"]["unmapped_count"] == 0
    assert report["split_scans"]["train"]["total_rows"] == 5
    assert report["split_scans"]["train"]["all_scanned_values_finite"] is True
    assert Path(report["json_path"]).exists()


def test_smart_dataset_post_rebuild_readiness_fails_closed_when_dataset_missing(tmp_path: Path) -> None:
    report = gate.run(_build_fixture(tmp_path, create_dataset=False))

    assert report["decision"] == gate.BLOCKED_DECISION
    assert any(failure["check"] == "smart dataset directory exists" for failure in report["failures"])
    assert report["training_allowed"] is False
    assert not any(report["side_effects_started"].values())


def test_smart_dataset_post_rebuild_readiness_blocks_nonfinite_sample(tmp_path: Path) -> None:
    report = gate.run(_build_fixture(tmp_path, nonfinite_train_snap=True))

    assert report["decision"] == gate.BLOCKED_DECISION
    assert report["split_scans"]["train"]["nonfinite_counts"]["snap"] == 1
    assert any(failure["check"] == "parquet scan loaded finite seq/snap/ctx sample" for failure in report["failures"])
    assert report["training_allowed"] is False
