import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_SIGNAL_DIM,
    model_native_signal_contract_metadata,
)
from gx1.contracts.entry_model_native_readiness_v1 import (
    model_native_readiness_contract_metadata,
)
from gx1.scripts import materialize_entry_model_native_seq513_smoke_manifest_v1 as gate
from tests.model_native_signal_support import canonical_model_native_selected_fields


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_split(
    dataset_dir: Path,
    split: str,
    *,
    width: int = MODEL_NATIVE_SIGNAL_DIM,
    write_manifest: bool = True,
    schema_version: str = gate.SPLIT_SCHEMA_VERSION,
    exact_signal_contract: bool = True,
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    parquet = dataset_dir / f"{gate.DEFAULT_STEM}_{split}.parquet"
    seq_values = [
        [[float(row + column + tick) for column in range(width)] for tick in range(2)]
        for row in range(3)
    ]
    snap_values = [
        [float(row + column + 1) for column in range(width)] for row in range(3)
    ]
    pq.write_table(
        pa.table(
            {
                "seq": pa.array(seq_values, type=pa.list_(pa.list_(pa.float32()))),
                "snap": pa.array(snap_values, type=pa.list_(pa.float32())),
                "y_direction": pa.array([0, 1, 2], type=pa.int64()),
            }
        ),
        parquet,
    )
    if not write_manifest:
        return
    contract = model_native_signal_contract_metadata(
        canonical_model_native_selected_fields(
            remainder_prefix="session_regime.smoke_manifest_fixture"
        )
    )
    fields = contract["fields"] if exact_signal_contract else [f"signal_{i}" for i in range(width)]
    _write_json(
        parquet.with_suffix(".manifest.json"),
        {
            "schema_version": schema_version,
            "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
            "expected_seq_snap_width": width,
            "output_data_path": str(parquet),
            "extra": {
                "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
                "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
                "neutral_xgb_bridge": False,
                "model_native_signal_contract": contract,
                "signal_bridge": {
                    "fields": fields,
                    "seq_input_dim": width,
                    "snap_input_dim": width,
                },
            },
        },
    )


def _dataset(
    tmp_path: Path,
    *,
    width: int = MODEL_NATIVE_SIGNAL_DIM,
    missing_manifest_split: str | None = None,
    schema_version: str = gate.SPLIT_SCHEMA_VERSION,
    exact_signal_contract: bool = True,
) -> Path:
    dataset_dir = tmp_path / "model_native_seq513_smoke_dataset"
    for split in gate.SPLITS:
        _write_split(
            dataset_dir,
            split,
            width=width,
            write_manifest=split != missing_manifest_split,
            schema_version=schema_version,
            exact_signal_contract=exact_signal_contract,
        )
    return dataset_dir


def _post_rebuild(
    tmp_path: Path,
    dataset_dir: Path,
    *,
    decision: str = gate.POST_REBUILD_READY_DECISION,
    side_effects: bool = True,
    provenance: bool = True,
    latest_name: bool = False,
) -> Path:
    name = (
        "ENTRY_SMART_DATASET_POST_REBUILD_READINESS_latest.json"
        if latest_name
        else "ENTRY_SMART_DATASET_POST_REBUILD_READINESS_20260716T120000123456Z.json"
    )
    payload = {
        "decision": decision,
        "post_rebuild_refresh_command_contract": {
            "smart_smoke_dataset_dir": str(dataset_dir),
            "all_commands_avoid_training_replay_iql_shadow_live": True,
        },
        "checks": (
            [{"name": name, "ok": True, "details": {}}
             for name in gate.REQUIRED_POST_REBUILD_ORCHESTRATION_CHECKS]
            if provenance
            else []
        ),
    }
    if side_effects:
        payload["side_effects_started"] = {
            "dataset_rebuild": False,
            "training": False,
            "replay": False,
            "iql_distillation": False,
            "shadow": False,
            "live": False,
        }
    return _write_json(tmp_path / name, payload)


def _args(
    tmp_path: Path,
    dataset_dir: Path,
    *,
    post_rebuild: Path | None = None,
    vedtak_id: str = "MODEL_NATIVE_SEQ513_SMOKE_PYTEST",
) -> argparse.Namespace:
    specialist = _write_json(
        tmp_path / "ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_20260716T120001123456Z.json",
        {"decision": "PASS"},
    )
    return argparse.Namespace(
        smart_smoke_dataset_dir=str(dataset_dir),
        post_rebuild_readiness_json=str(post_rebuild or _post_rebuild(tmp_path, dataset_dir)),
        smart_specialist_audit_json=str(specialist),
        out_dir=str(tmp_path / "reports"),
        vedtak_id=vedtak_id,
        memory_cap="22G",
        swap_cap="2G",
        sample_rows=2,
        batch_size=2,
        quiet=True,
    )


def _run_blocked(args: argparse.Namespace) -> dict:
    with pytest.raises(SystemExit) as exc_info:
        gate.run(args)
    assert exc_info.value.code == 2
    paths = list(Path(args.out_dir).glob(f"{gate.EVENT_PREFIX}_*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


def test_parser_requires_explicit_evidence_dataset_and_output_paths() -> None:
    parser = gate.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--vedtak", "MODEL_NATIVE_SEQ513_SMOKE_PYTEST"])
    help_text = parser.format_help()
    assert "_latest.json" not in help_text
    assert "--post-rebuild-readiness-json" in help_text
    assert "--smart-specialist-audit-json" in help_text


def test_materializes_one_hash_bound_immutable_manifest_event(tmp_path: Path) -> None:
    dataset_dir = _dataset(tmp_path)

    report = gate.run(_args(tmp_path, dataset_dir))

    assert report["decision"] == "READY_FOR_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST_REVIEW"
    assert report["manifest_embedded"] is True
    assert len(report["manifest_sha256"]) == 64
    assert len(report["evidence_binding_sha256"]) == 64
    assert report["manifest_variant"] == MODEL_NATIVE_CONTRACT_MODE
    assert report["expected_seq_snap_width"] == MODEL_NATIVE_SIGNAL_DIM
    assert (
        report["model_native_readiness_contract"]
        == model_native_readiness_contract_metadata()
    )
    assert report["training_allowed"] is False
    assert not any(report["side_effects_started"].values())

    event_path = Path(report["json_path"])
    assert event_path.name.startswith(f"{gate.EVENT_PREFIX}_")
    assert event_path.is_file()
    assert list((tmp_path / "reports").iterdir()) == [event_path]
    assert not list((tmp_path / "reports").glob("*.md"))

    manifest = report["smoke_manifest"]
    assert manifest["schema_version"] == gate.SCHEMA_VERSION
    assert manifest["explicit_vedtak_id"] == "MODEL_NATIVE_SEQ513_SMOKE_PYTEST"
    assert set(manifest["splits"]) == set(gate.SPLITS)
    for row in manifest["splits"].values():
        assert row["rows"] == 3
        assert row["seq_input_dim"] == MODEL_NATIVE_SIGNAL_DIM
        assert row["snap_input_dim"] == MODEL_NATIVE_SIGNAL_DIM
        assert row["field_count"] == MODEL_NATIVE_SIGNAL_DIM
        assert row["split_manifest_schema_version"] == gate.SPLIT_SCHEMA_VERSION
        assert len(row["out_parquet_sha256"]) == 64
        assert len(row["out_manifest_sha256"]) == 64


def test_rejects_mutable_latest_input_before_artifact_reads(tmp_path: Path) -> None:
    dataset_dir = _dataset(tmp_path)
    latest = _post_rebuild(tmp_path, dataset_dir, latest_name=True)

    with pytest.raises(RuntimeError, match="explicit timestamped JSON evidence"):
        gate.run(_args(tmp_path, dataset_dir, post_rebuild=latest))

    assert not (tmp_path / "reports").exists()


@pytest.mark.parametrize(
    ("dataset_kwargs", "blocker"),
    [
        ({"missing_manifest_split": "val"}, "exact train val test split manifests exist"),
        ({"width": 512}, "split signal seq and snap dims are 513"),
        ({"schema_version": "stale_split_schema_v1"}, "split manifests use model-native seq513 split schema"),
        ({"exact_signal_contract": False}, "split manifests carry exact model-native signal contract"),
    ],
)
def test_manifest_acceptance_fails_closed(
    tmp_path: Path,
    dataset_kwargs: dict,
    blocker: str,
) -> None:
    dataset_dir = _dataset(tmp_path, **dataset_kwargs)

    report = _run_blocked(_args(tmp_path, dataset_dir))

    assert report["decision"] == "BLOCKED_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST_READINESS"
    assert report["manifest_embedded"] is False
    assert report["smoke_manifest"] == {}
    assert blocker in report["blockers"]
    assert len(list((tmp_path / "reports").glob("*.json"))) == 1


@pytest.mark.parametrize("missing", ["side_effects", "provenance"])
def test_upstream_acceptance_proof_is_mandatory(tmp_path: Path, missing: str) -> None:
    dataset_dir = _dataset(tmp_path)
    post = _post_rebuild(
        tmp_path,
        dataset_dir,
        side_effects=missing != "side_effects",
        provenance=missing != "provenance",
    )

    report = _run_blocked(_args(tmp_path, dataset_dir, post_rebuild=post))

    assert report["decision"] == "BLOCKED_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST_READINESS"
    assert report["manifest_embedded"] is False


def test_source_has_one_immutable_writer_and_no_duplicate_outputs() -> None:
    source = Path(gate.__file__).read_text(encoding="utf-8")
    assert source.count("write_immutable_json_event(") == 1
    assert "replace_latest_json_mirror" not in source
    assert "_latest.json\").write_text" not in source
    assert ".md\"" not in source
    assert "SMOKE_DATASET_MANIFEST.json" not in source
    assert "smart_seq520" not in source.lower()
    assert "fail-on-not-ready" not in source
