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


def _build_fixture(
    tmp_path: Path,
    *,
    nonfinite_train_snap: bool = False,
    create_dataset: bool = True,
    preflight_orchestration: bool = True,
) -> argparse.Namespace:
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
    smart_rebuild_preflight = tmp_path / "ENTRY_SMART_REBUILD_PREFLIGHT_latest.json"
    _write_json(
        smart_rebuild_preflight,
        {
            "decision": "READY_FOR_SMART_REBUILD_VEDTAK_REVIEW",
            "failures": [] if preflight_orchestration else [{"check": "inventory feature orchestration contract is ready"}],
            "training_allowed": False,
            "dataset_rebuild_allowed_without_vedtak": False,
            "dataset_rebuild_allowed_after_explicit_vedtak_review": preflight_orchestration,
            "side_effects_started": {
                "dataset_rebuild": False,
                "training": False,
                "replay": False,
                "iql_distillation": False,
                "shadow": False,
                "live": False,
            },
            "inputs": {
                "smart_manifest": {
                    "path": str(smart_manifest),
                    "exists": True,
                    "sha256": _sha256(smart_manifest),
                }
            },
            "rebuild_command_contract": {
                "planned_dataset_dir": str(dataset_dir),
            },
            "checks": [
                {
                    "name": "inventory feature harmony contract is ready",
                    "ok": True,
                    "details": {"feature_harmony_ready": True, "unmapped_input_count": 0},
                },
                {
                    "name": "inventory feature orchestration contract is ready",
                    "ok": preflight_orchestration,
                    "details": {
                        "feature_orchestration_ready": preflight_orchestration,
                        "missing_required_smart_layers": [] if preflight_orchestration else ["mtf_confluence_layer"],
                    },
                },
            ],
        },
    )
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
        smart_rebuild_preflight=str(smart_rebuild_preflight),
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


def _with_scan_policy(
    args: argparse.Namespace,
    *,
    fullscan: bool | None = None,
    verify_source_parquet_hashes: bool | None = None,
) -> argparse.Namespace:
    if fullscan is not None:
        args.fullscan = fullscan
    if verify_source_parquet_hashes is not None:
        args.verify_source_parquet_hashes = verify_source_parquet_hashes
    return args


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
    assert report["smart_rebuild_preflight"]["exists"] is True
    check_names = {row["name"]: row for row in report["checks"]}
    assert check_names["smart rebuild preflight proves feature harmony"]["ok"] is True
    assert check_names["smart rebuild preflight proves feature orchestration"]["ok"] is True
    assert check_names["smart rebuild preflight planned dataset matches audited dataset"]["ok"] is True
    contract = report["post_rebuild_refresh_command_contract"]
    assert contract["ordered_steps"] == [
        "smart_feature_audit",
        "smart_target_audit",
        "smart_specialist_audit",
        "smart_smoke_dataset",
        "smart_smoke_manifest",
        "smart_smoke_readiness",
    ]
    assert contract["all_commands_avoid_training_replay_iql_shadow_live"] is True
    assert contract["commands"]["smart_feature_audit"]["argv"][:6] == [
        "scripts/gx1_capped_run.sh",
        "--mem",
        "4G",
        "--swap",
        "1G",
        "--",
    ]
    assert "--source-parquet" in contract["commands"]["smart_feature_audit"]["argv"]
    assert str(tmp_path / "sources" / "canonical_v3_FULL_PLUS_CTX.parquet") in contract["commands"]["smart_feature_audit"]["argv"]
    smoke_dataset_command = contract["commands"]["smart_smoke_dataset"]
    assert smoke_dataset_command["argv"] == [
        "scripts/entry_next_edge_control.sh",
        "smart-post-rebuild-refresh",
        "--apply",
        "--vedtak",
        "<SMART_SEQ520_POST_REBUILD_REFRESH_VEDTAK_ID>",
    ]
    assert smoke_dataset_command["implemented_in_control_surface"] is True
    smoke_dataset_argv = smoke_dataset_command["inner_argv"]
    assert smoke_dataset_argv[:6] == [
        "scripts/gx1_capped_run.sh",
        "--mem",
        "8G",
        "--swap",
        "1G",
        "--",
    ]
    assert smoke_dataset_command["requires_ram_cap"] is True
    assert smoke_dataset_command["ram_cap_mem"] == "8G"
    assert smoke_dataset_command["ram_cap_swap"] == "1G"
    assert "--schema-version" in smoke_dataset_argv
    assert "entry_smart_seq520_smoke_dataset_v1" in smoke_dataset_argv
    assert "--split-schema-version" in smoke_dataset_argv
    assert "entry_smart_seq520_smoke_split_manifest_v1" in smoke_dataset_argv
    assert "--manifest-variant" in smoke_dataset_argv
    assert "smart_seq520_candidate" in smoke_dataset_argv
    assert "--expected-seq-snap-width" in smoke_dataset_argv
    assert "520" in smoke_dataset_argv
    assert "--batch-size" in smoke_dataset_argv
    assert "256" in smoke_dataset_argv
    assert "--extreme-snap-feature" in smoke_dataset_argv
    assert "session_regime.session_trend_structure_liquidity_long_score" in smoke_dataset_argv
    assert "--extreme-snap-rows" in smoke_dataset_argv
    assert "64" in smoke_dataset_argv
    assert smoke_dataset_command["requires_explicit_vedtak"] is True
    assert contract["requires_explicit_smoke_dataset_refresh_vedtak"] is True
    assert contract["commands"]["smart_smoke_manifest"]["requires_explicit_vedtak"] is True
    for command in contract["commands"].values():
        assert command["starts_trainer"] is False
        assert command["starts_replay"] is False
        assert command["starts_iql_distillation"] is False
        assert command["touches_shadow_or_live"] is False
    assert Path(report["json_path"]).exists()


def test_smart_post_rebuild_refresh_control_uses_report_contract() -> None:
    control = Path("scripts/entry_next_edge_control.sh").read_text(encoding="utf-8")
    refresh_block = control.split("smart-post-rebuild-refresh)", 1)[1].split("smart-smoke-manifest)", 1)[0]

    assert "post_rebuild_refresh_command_contract" in refresh_block
    assert "--post-rebuild-readiness-json" in refresh_block
    assert "v10_6yr_rebuild_20260626_spreadfix" not in refresh_block


def test_smart_dataset_post_rebuild_readiness_fails_closed_when_dataset_missing(tmp_path: Path) -> None:
    report = gate.run(_build_fixture(tmp_path, create_dataset=False))

    assert report["decision"] == gate.BLOCKED_DECISION
    assert any(failure["check"] == "smart dataset directory exists" for failure in report["failures"])
    assert report["training_allowed"] is False
    assert not any(report["side_effects_started"].values())
    assert report["post_rebuild_refresh_command_contract"]["all_commands_avoid_training_replay_iql_shadow_live"] is True


def test_smart_dataset_post_rebuild_readiness_fails_closed_without_orchestration_preflight(tmp_path: Path) -> None:
    report = gate.run(_build_fixture(tmp_path, preflight_orchestration=False))

    assert report["decision"] == gate.BLOCKED_DECISION
    failure_names = {failure["check"] for failure in report["failures"]}
    assert "smart rebuild preflight decision is ready" in failure_names
    assert "smart rebuild preflight proves feature orchestration" in failure_names
    assert report["training_allowed"] is False
    assert not any(report["side_effects_started"].values())


def test_smart_dataset_post_rebuild_readiness_blocks_nonfinite_sample(tmp_path: Path) -> None:
    report = gate.run(_build_fixture(tmp_path, nonfinite_train_snap=True))

    assert report["decision"] == gate.BLOCKED_DECISION
    assert report["split_scans"]["train"]["nonfinite_counts"]["snap"] == 1
    assert any(failure["check"] == "parquet scan loaded finite seq/snap/ctx sample" for failure in report["failures"])
    assert report["training_allowed"] is False


def test_smart_dataset_post_rebuild_readiness_blocks_corrupt_partial_parquet(tmp_path: Path) -> None:
    args = _build_fixture(tmp_path)
    train_parquet = Path(args.dataset_dir) / "v10_smart_seq520_candidate__HOLD_03B_train.parquet"
    train_parquet.write_bytes(b"partial parquet from killed rebuild\n")

    report = gate.run(args)

    assert report["decision"] == gate.BLOCKED_DECISION
    assert any("parquet read error" in error for error in report["split_scans"]["train"]["errors"])
    assert any(failure["check"] == "parquet scan loaded finite seq/snap/ctx sample" for failure in report["failures"])
    assert report["training_allowed"] is False


def test_smart_dataset_post_rebuild_readiness_requires_fullscan_and_source_hashes(tmp_path: Path) -> None:
    args = _with_scan_policy(
        _build_fixture(tmp_path),
        fullscan=False,
        verify_source_parquet_hashes=False,
    )

    report = gate.run(args)

    failure_names = {failure["check"] for failure in report["failures"]}
    assert report["decision"] == gate.BLOCKED_DECISION
    assert "post-rebuild scan is fullscan" in failure_names
    assert "source parquet hashes are explicitly verified" in failure_names
    assert "train source parquet observed hash matches recorded" in failure_names
    assert report["training_allowed"] is False
