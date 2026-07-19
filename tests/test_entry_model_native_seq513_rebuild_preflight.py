import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_SIGNAL_DIM,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_MANDATORY_FAMILY_FEATURES,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
    model_native_mandatory_full_stack_metadata,
    model_native_signal_contract_metadata,
)
from gx1.contracts.entry_model_native_state_v2 import (
    MODEL_NATIVE_HISTORY_MODE,
    MODEL_NATIVE_RANK_TRANSFORM,
    MODEL_NATIVE_STATE_SCHEMA_VERSION,
    MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    group_features_by_specialist,
)
from gx1.features.htf_features import (
    MULTI_TF_FEATURE_COUNT_V2,
    MULTI_TF_PER_BAR_FEATURES_V2,
    MULTI_TF_SHIFT,
)
from gx1.scripts import (
    materialize_entry_model_native_seq513_rebuild_preflight_v1 as preflight,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields


STAMP = "20260716T120000123456Z"
CREATED = "2026-07-16T12:00:00.123456+00:00"
SPLITS = {
    "history_start": "2020-11-08T00:00:00+00:00",
    "train_start": "2020-11-09T00:00:00+00:00",
    "train_end": "2025-09-30T23:59:59+00:00",
    "val_start": "2025-10-01T00:00:00+00:00",
    "val_end": "2025-12-31T23:59:59+00:00",
    "test_start": "2026-01-01T00:00:00+00:00",
    "test_end": "2026-06-26T03:25:00+00:00",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_parquet(path: Path, payload: dict[str, list]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(payload), path)
    return path


def _selected_features() -> list[str]:
    return canonical_model_native_selected_fields()


def _build_fixture(
    tmp_path: Path,
    *,
    mutable_manifest: bool = False,
    break_signal_contract: bool = False,
    break_specialist_coverage: bool = False,
    break_source_manifest_hash: bool = False,
    break_mtf: bool = False,
    missing_tape_year: int | None = None,
    missing_tape_column: str | None = None,
) -> argparse.Namespace:
    history_base = datetime.fromisoformat(SPLITS["history_start"])
    history_times = [history_base + timedelta(minutes=5 * index) for index in range(289)]
    source_times = history_times + [datetime.fromisoformat(SPLITS["test_end"])]
    n_source = len(source_times)
    close = np.linspace(1800.0, 2300.0, n_source)
    source = _write_parquet(
        tmp_path / "inputs/source.parquet",
        {
            "time": source_times,
            "open": close.tolist(),
            "high": (close + 1.0).tolist(),
            "low": (close - 1.0).tolist(),
            "close": close.tolist(),
            "bid_close": (close - 0.05).tolist(),
            "ask_close": (close + 0.05).tolist(),
        },
    )
    canonical = _write_parquet(
        tmp_path / "inputs/canonical_v2.parquet",
        {
            "time": [
                datetime(2020, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 6, 26, 3, 25, tzinfo=timezone.utc),
            ],
            "canonical_feature": [1.0, 2.0],
        },
    )

    source_layer = _write_json(
        tmp_path / f"inputs/MODEL_NATIVE_SIGNAL_LAYER_{STAMP}.json",
        {"created_utc": CREATED, "proof": True},
    )
    selected = _selected_features()
    if break_specialist_coverage:
        selected[-1] = "unmapped_fixture_field"
    declared_contract = model_native_signal_contract_metadata(selected)
    if break_signal_contract:
        declared_contract["seq_input_dim"] = 512
    grouped = group_features_by_specialist(selected)
    manifest_name = (
        "MODEL_NATIVE_SIGNAL_MANIFEST_latest.json"
        if mutable_manifest
        else f"MODEL_NATIVE_SIGNAL_MANIFEST_{STAMP}.json"
    )
    manifest_path = tmp_path / "inputs" / manifest_name
    manifest = {
        "schema_version": "entry_specialist_challenger_extension_manifest_v1",
        "created_utc": CREATED,
        "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
        "selected_features": selected,
        "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "base_signal_feature_count": MODEL_NATIVE_BASE_SIGNAL_DIM,
        "expected_seq_snap_width": MODEL_NATIVE_SIGNAL_DIM,
        "model_native_signal_contract": declared_contract,
        "mandatory_full_stack": model_native_mandatory_full_stack_metadata(),
        "smart_layer_feature_counts": {
            family: len(features)
            for family, features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
        },
        "source_feature_counts": {
            "smart_candidate_layers": 305,
            "mandatory_full_stack": 305,
            "ranked_remainder": 174,
        },
        "features_by_specialist": grouped,
        "required_training_specialists": list(MODEL_NATIVE_TRAINING_SPECIALISTS),
        "source_manifests": {
            "model_native_signal_layer": {
                "path": str(source_layer.resolve()),
                "sha256": "0" * 64 if break_source_manifest_hash else _sha256(source_layer),
            }
        },
        "manifest_json_path": str(manifest_path.resolve()),
    }
    _write_json(manifest_path, manifest)

    test_end_ns = int(
        datetime.fromisoformat(SPLITS["test_end"]).timestamp() * 1_000_000_000
    )
    first_ns = int(
        datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp() * 1_000_000_000
    )
    mtf_cache = tmp_path / "inputs/mtf_cache"
    mtf_cache.mkdir(parents=True)
    tf_rows: dict[str, dict] = {}
    for tf in ("M5", "M15", "H1", "H4", "D1"):
        feats_name = f"{tf}_feats.npy"
        ts_name = f"{tf}_ts.npy"
        np.save(
            mtf_cache / feats_name,
            np.zeros((2, MULTI_TF_FEATURE_COUNT_V2), dtype=np.float32),
        )
        np.save(mtf_cache / ts_name, np.array([first_ns, test_end_ns], dtype=np.int64))
        tf_rows[tf] = {
            "n_bars": 2,
            "feature_count": MULTI_TF_FEATURE_COUNT_V2,
            "feats_npy": feats_name,
            "ts_npy": ts_name,
            "first_ts_ns": first_ns,
            "last_ts_ns": test_end_ns,
        }
    mtf_manifest = {
        "feature_count": MULTI_TF_FEATURE_COUNT_V2,
        "feature_names": list(MULTI_TF_PER_BAR_FEATURES_V2),
        "shift_contract": {tf: str(MULTI_TF_SHIFT[tf]) for tf in tf_rows},
        "builder_version": preflight.EXPECTED_MTF_BUILDER_VERSION,
        "m5_prebuilt_source": str(source.resolve()),
        "m5_prebuilt_source_sha256": _sha256(source),
        "tfs": tf_rows,
    }
    if break_mtf:
        mtf_manifest["feature_names"] = list(MULTI_TF_PER_BAR_FEATURES_V2[:-1])
    _write_json(mtf_cache / "manifest.json", mtf_manifest)

    tape_root = tmp_path / "inputs/tape"
    for year in range(2020, 2027):
        if year == missing_tape_year:
            continue
        tape_payload = {
                "time": [datetime(year, 1, 2, tzinfo=timezone.utc)],
                "open": [1800.05 + year],
                "high": [1800.25 + year],
                "low": [1799.85 + year],
                "close": [1800.05 + year],
                "bid_close": [1800.0 + year],
                "bid_high": [1800.2 + year],
                "bid_low": [1799.8 + year],
                "ask_close": [1800.1 + year],
                "ask_high": [1800.3 + year],
                "ask_low": [1799.9 + year],
        }
        if missing_tape_column and year == 2023:
            tape_payload.pop(missing_tape_column)
        _write_parquet(tape_root / f"year={year}/part-000.parquet", tape_payload)

    return argparse.Namespace(
        source_parquet=str(source),
        canonical_v2_parquet=str(canonical),
        signal_manifest=str(manifest_path),
        rank_reference_npz=str(tmp_path / "run/rank/model_native_rank.npz"),
        mtf_cache_dir=str(mtf_cache),
        tape_root=str(tape_root),
        output=str(tmp_path / "run/dataset/model_native__HOLD_03B.parquet"),
        audit_out_dir=str(tmp_path / "run/pretrain_audit"),
        out_dir=str(tmp_path / "reports"),
        **SPLITS,
    )


def test_preflight_binds_exact_wrapper_inputs_and_fails_closed_without_vedtak(
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path)

    report = preflight.run(args)

    assert report["decision"] == preflight.READY_DECISION
    assert report["schema_version"] == "entry_model_native_seq513_rebuild_preflight_v1"
    assert not report["failures"]
    assert report["counts"] == {
        "base_signal_features": MODEL_NATIVE_BASE_SIGNAL_DIM,
        "selected_features": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "expected_seq_snap_width": MODEL_NATIVE_SIGNAL_DIM,
        "seq_len": MODEL_NATIVE_SEQ_LEN,
        "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        "required_specialist_count": 8,
        "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
    }
    assert report["required_model_native_contract"] == {
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "seq_len": MODEL_NATIVE_SEQ_LEN,
        "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        "base_signal_dim": MODEL_NATIVE_BASE_SIGNAL_DIM,
        "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "bridge_dim": 0,
        "bridge_source": None,
        "anchor_source": None,
    }
    assert report["specialist_contract"]["all_eight_covered"] is True
    assert report["specialist_contract"]["mandatory_full_stack_exact"] is True
    assert report["inputs"]["multi_tf_cache"]["exact"] is True
    assert report["inputs"]["tape"]["exact"] is True
    assert report["inputs"]["source_parquet"]["sha256"] == _sha256(
        Path(args.source_parquet)
    )
    assert report["inputs"]["canonical_v2_parquet"]["sha256"] == _sha256(
        Path(args.canonical_v2_parquet)
    )

    command = report["rebuild_command_contract"]
    argv = command["argv_template"]
    assert argv[:3] == [
        "scripts/rebuild_entry_model_native_seq513_dataset.sh",
        "--vedtak",
        "<EXPLICIT_VEDTAK_ID>",
    ]
    for flag in (
        "--source-parquet",
        "--canonical-v2-parquet",
        "--signal-manifest",
        "--rank-reference-npz",
        "--mtf-cache-dir",
        "--tape-root",
        "--output",
        "--audit-out-dir",
        "--history-start",
        "--train-start",
        "--train-end",
        "--val-start",
        "--val-end",
        "--test-start",
        "--test-end",
    ):
        assert argv.count(flag) == 1
    assert command["requires_explicit_vedtak"] is True
    assert command["executable_without_replacing_vedtak_placeholder"] is False
    assert command["rank_reference_contract"] == {
        "producer": "gx1.scripts.materialize_model_native_train_rank_reference_v2",
        "schema_version": MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
        "source_parquet": str(Path(args.source_parquet).resolve()),
        "source_parquet_sha256": _sha256(Path(args.source_parquet)),
        "output_npz": str(Path(args.rank_reference_npz).resolve()),
        "sidecar_json": str(Path(args.rank_reference_npz).resolve()) + ".json",
        "materialized_before_dataset_builder": True,
        "feature_history_start_utc": SPLITS["history_start"],
        "fit_start_utc": SPLITS["train_start"],
        "fit_end_utc": SPLITS["train_end"],
        "fit_scope": "train_only",
        "rank_transform": MODEL_NATIVE_RANK_TRANSFORM,
        "contains_validation_or_test_rows": False,
        "contains_per_row_state": False,
        "sidecar_source_sha256_must_match": True,
        "sidecar_npz_sha256_required": True,
        "builder_must_verify_npz_and_sidecar": True,
        "requires_explicit_vedtak": True,
        "vedtak_bound_in_npz_and_sidecar": True,
        "dataset_builder_requires_same_explicit_vedtak": True,
    }
    assert command["fixed_builder_contract"]["state_schema_version"] == MODEL_NATIVE_STATE_SCHEMA_VERSION
    assert command["fixed_builder_contract"]["direction_target_mode"] == "path_utility_v2"
    assert command["fixed_builder_contract"]["explicit_vedtak_required"] is True
    assert command["fixed_builder_contract"]["rank_reference_vedtak_match_required"] is True
    aux_contract = command["fixed_builder_contract"]["aux_head_target_contract"]
    assert aux_contract["schema_version"] == "entry_model_native_aux_targets_v2"
    assert len(aux_contract["columns"]) == 37
    assert aux_contract["max_future_horizon_bars"] == 96
    assert aux_contract["spread_aware_risk_magnitudes_required"] is True
    assert aux_contract["incomplete_rows_may_be_emitted"] is False
    assert command["fixed_builder_contract"]["feature_history_mode"] == MODEL_NATIVE_HISTORY_MODE
    assert command["fixed_builder_contract"]["split_reset_allowed"] is False
    assert report["inputs"]["source_time_contract"]["exact"] is True
    assert report["dataset_rebuild_allowed_without_vedtak"] is False
    assert report["training_allowed"] is False
    assert not any(report["side_effects_started"].values())
    assert Path(report["json_path"]).name.startswith(
        "ENTRY_MODEL_NATIVE_SEQ513_REBUILD_PREFLIGHT_"
    )
    assert not list(Path(report["json_path"]).parent.glob("*_latest.json"))

    serialized = json.dumps(report, sort_keys=True).lower()
    for retired in (
        "smart_report",
        "inventory_report",
        "source_dataset_dir",
        "planned_dataset_dir",
        "neutral_xgb_bridge",
        "allow_zero_ctx",
    ):
        assert retired not in serialized


@pytest.mark.parametrize(
    ("fixture_kwargs", "failure_name"),
    [
        (
            {"break_signal_contract": True},
            "signal manifest proves exact ordered 34+479=513 and 142/5 intent",
        ),
        (
            {"break_specialist_coverage": True},
            "all 479 selected features map across the exact eight specialists",
        ),
        (
            {"break_source_manifest_hash": True},
            "declared signal source manifests are present and hash-bound",
        ),
        (
            {"break_mtf": True},
            "explicit MTF cache has exact five-TF files/schema/source hash/coverage",
        ),
        (
            {"missing_tape_year": 2023},
            "explicit tape root has exact spread-aware OHLC coverage for every split year",
        ),
        (
            {"missing_tape_column": "ask_low"},
            "explicit tape root has exact spread-aware OHLC coverage for every split year",
        ),
    ],
)
def test_preflight_blocks_incomplete_contracts(
    tmp_path: Path, fixture_kwargs: dict, failure_name: str
) -> None:
    report = preflight.run(_build_fixture(tmp_path, **fixture_kwargs))

    assert report["decision"] == preflight.BLOCKED_DECISION
    assert failure_name in {row["name"] for row in report["failures"]}
    assert report["dataset_rebuild_allowed_after_explicit_vedtak_review"] is False


def test_preflight_rejects_479_field_manifest_that_swaps_one_mandatory_member(
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path)
    manifest_path = Path(args.signal_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = list(manifest["selected_features"])
    victim = MODEL_NATIVE_MANDATORY_SELECTED_FIELDS[0]
    replacement = "trend.ema_adversarial_replacement_fixture"
    before = {
        name: len(fields)
        for name, fields in group_features_by_specialist(selected).items()
    }
    selected[selected.index(victim)] = replacement
    after_groups = group_features_by_specialist(selected)
    after = {name: len(fields) for name, fields in after_groups.items()}
    assert len(selected) == len(set(selected)) == MODEL_NATIVE_SELECTED_FEATURE_COUNT
    assert before == after
    manifest["selected_features"] = selected
    manifest["features_by_specialist"] = after_groups
    _write_json(manifest_path, manifest)

    report = preflight.run(args)

    assert report["decision"] == preflight.BLOCKED_DECISION
    assert report["specialist_contract"]["all_eight_covered"] is True
    assert report["specialist_contract"]["mandatory_full_stack_exact"] is False
    assert "all code-owned full-stack family fields are retained exactly" in {
        row["name"] for row in report["failures"]
    }


def test_preflight_rejects_stale_duplicate_family_count_metadata(
    tmp_path: Path,
) -> None:
    args = _build_fixture(tmp_path)
    manifest_path = Path(args.signal_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["smart_layer_feature_counts"]["trend_ema_smart_layer"] -= 1
    _write_json(manifest_path, manifest)

    report = preflight.run(args)

    assert report["decision"] == preflight.BLOCKED_DECISION
    assert "declared mandatory family counts match recomputed selected names" in {
        row["name"] for row in report["failures"]
    }


def test_preflight_rejects_mutable_manifest_and_existing_outputs(tmp_path: Path) -> None:
    mutable = preflight.run(_build_fixture(tmp_path / "mutable", mutable_manifest=True))
    assert mutable["decision"] == preflight.BLOCKED_DECISION
    assert "signal manifest is an explicit timestamped immutable input" in {
        row["name"] for row in mutable["failures"]
    }

    args = _build_fixture(tmp_path / "existing")
    rank = Path(args.rank_reference_npz)
    rank.parent.mkdir(parents=True)
    rank.write_bytes(b"stale")
    output = Path(args.output)
    output.parent.mkdir(parents=True)
    output.with_name(f"{output.stem}_train.manifest.json").write_text("{}\n")
    Path(args.audit_out_dir).mkdir(parents=True)
    report = preflight.run(args)
    failure_names = {row["name"] for row in report["failures"]}
    assert report["decision"] == preflight.BLOCKED_DECISION
    assert "rank-reference NPZ and sidecar paths are fresh" in failure_names
    assert "dataset output path and derived split artifacts are fresh" in failure_names
    assert "audit output directory is fresh" in failure_names


def test_preflight_blocks_overlapping_split_windows(tmp_path: Path) -> None:
    args = _build_fixture(tmp_path)
    args.val_start = args.train_end

    report = preflight.run(args)

    assert report["decision"] == preflight.BLOCKED_DECISION
    assert "explicit train/val/test split windows are ordered and non-overlapping" in {
        row["name"] for row in report["failures"]
    }
    assert report["rebuild_command_contract"] == {}


def _explicit_cli_args(tmp_path: Path) -> list[str]:
    args = [
        "--source-parquet",
        str(tmp_path / "source.parquet"),
        "--canonical-v2-parquet",
        str(tmp_path / "canonical.parquet"),
        "--signal-manifest",
        str(tmp_path / f"MODEL_NATIVE_SIGNAL_MANIFEST_{STAMP}.json"),
        "--rank-reference-npz",
        str(tmp_path / "rank.npz"),
        "--mtf-cache-dir",
        str(tmp_path / "mtf"),
        "--tape-root",
        str(tmp_path / "tape"),
        "--output",
        str(tmp_path / "dataset/model_native__HOLD_03B.parquet"),
        "--audit-out-dir",
        str(tmp_path / "audit"),
    ]
    for name, value in SPLITS.items():
        args.extend([f"--{name.replace('_', '-')}", value])
    args.extend(["--out-dir", str(tmp_path / "reports")])
    return args


def test_parser_requires_exact_rebuild_inputs_and_rejects_retired_arguments(
    tmp_path: Path,
) -> None:
    parser = preflight.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    for retired in (
        ["--smart-report", str(tmp_path / "report.json")],
        ["--inventory-report", str(tmp_path / "inventory.json")],
        ["--source-dataset-dir", str(tmp_path / "old_dataset")],
        ["--gx1-data-root", str(tmp_path / "GX1_DATA")],
        ["--verify-large-input-hashes"],
        ["--neutral-xgb-bridge"],
        ["--allow-zero-ctx"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(_explicit_cli_args(tmp_path) + retired)

    parsed = parser.parse_args(_explicit_cli_args(tmp_path))
    assert parsed.source_parquet == str(tmp_path / "source.parquet")
    assert parsed.rank_reference_npz == str(tmp_path / "rank.npz")


def test_run_rejects_old_namespace_without_explicit_source(tmp_path: Path) -> None:
    old = argparse.Namespace(
        smart_report=str(tmp_path / "old.json"),
        inventory_report=str(tmp_path / "old_inventory.json"),
        source_dataset_dir=str(tmp_path / "old_dataset"),
        out_dir=str(tmp_path / "reports"),
    )

    with pytest.raises(RuntimeError, match="explicit --source-parquet is required"):
        preflight.run(old)
