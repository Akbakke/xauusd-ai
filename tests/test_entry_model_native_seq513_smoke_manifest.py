import argparse
import hashlib
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
        "schema_version": gate.POST_REBUILD_SCHEMA_VERSION,
        "decision": decision,
        "post_rebuild_refresh_command_contract": {
            "smoke_dataset_dir": str(dataset_dir),
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
    run_id: str = "MODEL_NATIVE_SEQ513_SMOKE_PYTEST",
) -> argparse.Namespace:
    specialist = _write_json(
        tmp_path / "ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_20260716T120001123456Z.json",
        {"decision": "PASS"},
    )
    values: dict[str, object] = {
        f"{split}_{kind}": str(
            dataset_dir
            / (
                f"{gate.DEFAULT_STEM}_{split}.parquet"
                if kind == "parquet"
                else f"{gate.DEFAULT_STEM}_{split}.manifest.json"
            )
        )
        for split in gate.SPLITS
        for kind in ("parquet", "manifest_json")
    }
    for split in gate.SPLITS:
        for kind in ("parquet", "manifest"):
            path = Path(values[f"{split}_{kind if kind == 'parquet' else 'manifest_json'}"])
            values[f"{split}_{kind}_sha256"] = (
                hashlib.sha256(path.read_bytes()).hexdigest()
                if path.is_file()
                else "0" * 64
            )
    return argparse.Namespace(
        smart_smoke_dataset_dir=str(dataset_dir),
        post_rebuild_readiness_json=str(post_rebuild or _post_rebuild(tmp_path, dataset_dir)),
        smart_specialist_audit_json=str(specialist),
        out_dir=str(tmp_path / "reports"),
        run_id=run_id,
        memory_cap="10G",
        swap_cap="512M",
        sample_rows=2,
        batch_size=2,
        quiet=True,
        **values,
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
        parser.parse_args(["--run-id", "MODEL_NATIVE_SEQ513_SMOKE_PYTEST"])
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
    train_contract = report["future_command_contracts"]["smart_smoke_train"]
    argv = train_contract["argv_template"]
    assert argv == train_contract["wrapper_argv_template"]
    assert argv[:2] == [
        "scripts/entry_next_edge_control.sh",
        "model-native-smoke-train",
    ]
    for flag in (
        "--train-manifest-json",
        "--val-manifest-json",
        "--test-manifest-json",
        "--train-parquet",
        "--val-parquet",
        "--test-parquet",
    ):
        assert flag in argv
    joined = " ".join(argv)
    assert "gx1.models.entry_v10.entry_v10_ctx_train_v3" not in joined
    assert "--dataset_dir" not in argv
    assert "--dataset_train_parquet" not in argv

    event_path = Path(report["json_path"])
    assert event_path.name.startswith(f"{gate.EVENT_PREFIX}_")
    assert event_path.is_file()
    assert list((tmp_path / "reports").iterdir()) == [event_path]
    assert not list((tmp_path / "reports").glob("*.md"))

    manifest = report["smoke_manifest"]
    assert manifest["schema_version"] == gate.SCHEMA_VERSION
    assert manifest["entry_run_id"] == "MODEL_NATIVE_SEQ513_SMOKE_PYTEST"
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
        ({"missing_manifest_split": "val"}, "exact train val test split artifacts exist"),
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


def test_explicit_split_hash_and_six_way_identity_fail_closed(tmp_path: Path) -> None:
    dataset_dir = _dataset(tmp_path)
    bad_hash = _args(tmp_path, dataset_dir)
    bad_hash.val_parquet_sha256 = "0" * 64

    hash_report = _run_blocked(bad_hash)

    assert "caller-bound split hashes match train val test bytes" in hash_report["blockers"]

    second_root = tmp_path / "second"
    second_root.mkdir()
    second_dataset = _dataset(second_root)
    duplicate = _args(second_root, second_dataset)
    duplicate.val_parquet = duplicate.train_parquet
    duplicate.val_parquet_sha256 = duplicate.train_parquet_sha256

    duplicate_report = _run_blocked(duplicate)

    assert (
        "train val test split paths are explicit canonical and distinct"
        in duplicate_report["blockers"]
    )


def test_source_has_one_immutable_writer_and_no_duplicate_outputs() -> None:
    source = Path(gate.__file__).read_text(encoding="utf-8")
    assert source.count("write_immutable_json_event(") == 1
    assert "replace_latest_json_mirror" not in source
    assert "_latest.json\").write_text" not in source
    assert ".md\"" not in source
    assert "SMOKE_DATASET_MANIFEST.json" not in source
    assert "smart_seq520" not in source.lower()
    assert "fail-on-not-ready" not in source
    assert "_split_candidates" not in source
    assert 'glob(f"*_{split}' not in source


def test_control_route_requires_all_explicit_split_paths_and_hashes() -> None:
    source = Path("scripts/entry_next_edge_control.sh").read_text(encoding="utf-8")
    route = source[
        source.index("  model-native-smoke-manifest)") :
        source.index("  model-native-smoke-readiness)")
    ]
    for split in gate.SPLITS:
        for suffix in (
            "parquet",
            "parquet-sha256",
            "manifest-json",
            "manifest-sha256",
        ):
            assert f"--{split}-{suffix}" in route
