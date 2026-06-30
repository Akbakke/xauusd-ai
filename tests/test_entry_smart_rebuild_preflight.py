import argparse
import hashlib
import json
from pathlib import Path

from gx1.scripts import materialize_entry_smart_seq520_rebuild_preflight_v1 as preflight


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_fixture(
    tmp_path: Path,
    *,
    source_coverage: bool = True,
    verify_large_input_hashes: bool = True,
) -> argparse.Namespace:
    source = tmp_path / "FULL_PLUS_CTX_v3src.parquet"
    source.write_bytes(b"dummy parquet placeholder")
    output_dir = tmp_path / "foundation_dataset"
    output_dir.mkdir(parents=True)
    xgb_bundle = tmp_path / "xgb_bundle"
    xgb_bundle.mkdir(parents=True)
    (xgb_bundle / "xgb_universal_multihead_v2.joblib").write_bytes(b"dummy model")
    source_sha = _sha256(source)

    for split in ("train", "val", "test"):
        parquet = output_dir / f"v10_foundation_seq146__HOLD_03B_{split}.parquet"
        parquet.write_bytes(f"{split}\n".encode("utf-8"))
        _write_json(
            output_dir / f"v10_foundation_seq146__HOLD_03B_{split}.manifest.json",
            {
                "output_data_path": str(parquet),
                "splits": {
                    "train": {
                        "start": "2020-11-09 00:00:00+00:00",
                        "end": "2025-09-30 23:59:59+00:00",
                    },
                    "val": {
                        "start": "2025-10-01 00:00:00+00:00",
                        "end": "2025-12-31 23:59:59+00:00",
                    },
                    "test": {
                        "start": "2026-01-01 00:00:00+00:00",
                        "end": "2026-06-26 03:25:00+00:00",
                    },
                },
                "extra": {
                    "base28_manifest": {
                        "path": "/dev/null",
                        "parquet_path": str(source),
                        "parquet_sha256": source_sha,
                    },
                    "xgb_bundle": str(xgb_bundle),
                    "signal_bridge": {
                        "id": "XGB_SIGNAL_BRIDGE_V3",
                        "seq_input_dim": 146,
                        "snap_input_dim": 146,
                        "base_seq_input_dim": 41,
                        "seq_structure_extension_dim": 105,
                        "neutral_xgb_bridge": True,
                        "fields": [f"f{i}" for i in range(146)],
                        "seq_structure_extension_v1": {
                            "enabled": True,
                            "feature_count": 105,
                            "manifest_path": "/tmp/sequence_structure_feature_layer_manifest.json",
                            "manifest_selected_feature_count": 105,
                            "source_parquet_for_price_features": str(source),
                        },
                    },
                    "ctx_contract": {
                        "tag": "CTX6CAT5",
                        "ctx_cont_dim": 142,
                        "ctx_cat_dim": 5,
                        "ctx_cat_names": [
                            "session_id",
                            "vol_regime_id",
                            "atr_bucket",
                            "spread_bucket",
                            "H4_trend_sign_cat",
                        ],
                        "allow_zero_ctx": False,
                    },
                },
            },
        )

    smart_manifest = tmp_path / "ENTRY_SPECIALIST_CHALLENGER_SMART_EXTENSION_MANIFEST.json"
    selected = [f"smart.feature_{i:03d}" for i in range(479)]
    _write_json(
        smart_manifest,
        {
            "decision": "READY_FOR_SMART_CHALLENGER_DATASET_REBUILD_MANIFEST",
            "manifest_variant": "smart_seq520_candidate",
            "selected_features": selected,
            "dataset_rebuild_required_before_training": True,
            "training_allowed": False,
        },
    )
    smart_report = tmp_path / "ENTRY_SPECIALIST_CHALLENGER_SMART_EXTENSION_REPORT_latest.json"
    _write_json(
        smart_report,
        {
            "decision": "READY_FOR_SMART_CHALLENGER_DATASET_REBUILD_MANIFEST",
            "training_allowed": False,
            "manifest": {"manifest_json_path": str(smart_manifest)},
            "counts": {
                "base_signal_features": 41,
                "foundation_sequence_extension_features": 105,
                "chart_geometry_challenger_features": 41,
                "candlestick_challenger_features": 28,
                "smart_candidate_features": 305,
                "combined_selected_features": 479,
                "expected_seq_snap_width": 520,
                "duplicate_feature_count": 0,
            },
        },
    )
    inventory = tmp_path / "ENTRY_FEATURE_AI_INVENTORY_latest.json"
    _write_json(
        inventory,
        {
            "decision": "READY_FOR_SPECIALIST_AI_DESIGN_REVIEW",
            "training_allowed": False,
            "side_effects_started": {
                "training": False,
                "replay": False,
                "iql_distillation": False,
                "shadow": False,
                "live": False,
                "promotion": False,
            },
            "smart_candidate": {
                "manifest_variant": "smart_seq520_candidate",
                "expected_signal_dim": 520,
                "smart_layer_features": 305,
                "source_coverage_all_required_available": source_coverage,
                "missing_required_source_field_layers": [] if source_coverage else ["session_regime_interaction_layer"],
                "dataset_rebuild_required_before_training": True,
                "training_allowed": False,
            },
        },
    )
    return argparse.Namespace(
        smart_report=str(smart_report),
        inventory_report=str(inventory),
        foundation_dataset_dir=str(output_dir),
        planned_dataset_dir=str(tmp_path / "smart_dataset"),
        out_dir=str(tmp_path / "reports"),
        verify_large_input_hashes=verify_large_input_hashes,
        quiet=True,
        no_fail_on_audit_fail=False,
    )


def test_smart_rebuild_preflight_accepts_dynamic_smart_seq_width(tmp_path: Path) -> None:
    args = _build_fixture(tmp_path)

    report = preflight.run(args)

    assert report["decision"] == "READY_FOR_SMART_REBUILD_VEDTAK_REVIEW"
    assert report["training_allowed"] is False
    assert report["dataset_rebuild_allowed_without_vedtak"] is False
    assert not any(report["side_effects_started"].values())
    assert report["counts"]["expected_seq_snap_width"] == 520
    assert report["counts"]["smart_layer_features"] == 305
    argv = report["rebuild_command_contract"]["argv"]
    assert argv[:6] == ["scripts/gx1_capped_run.sh", "--mem", "22G", "--swap", "2G", "--"]
    assert "--source-parquet-override" in argv
    assert "--seq-structure-compute-inline" in argv
    assert argv[argv.index("--train_start") + 1] == "2020-11-09 00:00:00+00:00"
    assert argv[argv.index("--train_end") + 1] == "2025-09-30 23:59:59+00:00"
    assert argv[argv.index("--val_start") + 1] == "2025-10-01 00:00:00+00:00"
    assert argv[argv.index("--val_end") + 1] == "2025-12-31 23:59:59+00:00"
    assert argv[argv.index("--test_start") + 1] == "2026-01-01 00:00:00+00:00"
    assert argv[argv.index("--test_end") + 1] == "2026-06-26 03:25:00+00:00"
    assert report["rebuild_command_contract"]["split_schedule"]["train"]["start"] == "2020-11-09 00:00:00+00:00"
    assert report["rebuild_command_contract"]["allowed_without_vedtak"] is False
    assert report["rebuild_command_contract"]["requires_explicit_rebuild_vedtak"] is True
    assert report["rebuild_command_contract"]["requires_clean_git_before_execution"] is True
    assert report["rebuild_command_contract"]["uses_legacy_guarded_builder"] is True
    assert report["rebuild_command_contract"]["required_environment"] == {
        "GX1_DATA": "/home/andre2/GX1_DATA",
        "GX1_ALLOW_LEGACY_ENTRY_V10_RESEARCH": "20260627_ALLOW_LEGACY_ENTRY_V10_RESEARCH"
    }
    assert report["rebuild_command_contract"]["starts_trainer"] is False
    assert report["rebuild_command_contract"]["starts_replay"] is False
    assert report["rebuild_command_contract"]["starts_iql_distillation"] is False
    assert report["rebuild_command_contract"]["touches_shadow_or_live"] is False
    assert Path(report["json_path"]).exists()
    assert _sha256(Path(report["inputs"]["smart_report"]["path"])) == report["inputs"]["smart_report"]["sha256"]


def test_smart_rebuild_preflight_fails_closed_on_missing_source_coverage(tmp_path: Path) -> None:
    args = _build_fixture(tmp_path, source_coverage=False)

    report = preflight.run(args)

    assert report["decision"] == "BLOCKED_SMART_REBUILD_PREFLIGHT"
    assert any(row["name"] == "inventory required source coverage is complete" for row in report["failures"])
    assert report["dataset_rebuild_allowed_after_explicit_vedtak_review"] is False
    assert report["side_effects_started"] == {
        "dataset_rebuild": False,
        "training": False,
        "replay": False,
        "iql_distillation": False,
        "shadow": False,
        "live": False,
    }


def test_smart_rebuild_preflight_fails_closed_without_large_hash_verification(tmp_path: Path) -> None:
    args = _build_fixture(tmp_path, verify_large_input_hashes=False)

    report = preflight.run(args)

    assert report["decision"] == "BLOCKED_SMART_REBUILD_PREFLIGHT"
    failure_names = {row["name"] for row in report["failures"]}
    assert "large source parquet hashes are explicitly verified" in failure_names
    assert "train source parquet observed hash matches recorded" in failure_names
    assert report["dataset_rebuild_allowed_after_explicit_vedtak_review"] is False
