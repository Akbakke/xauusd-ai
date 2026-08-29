from __future__ import annotations

import json
from pathlib import Path

from gx1.contracts.entry_model_native_post_rebuild_v1 import (
    PRETEST_TEST_GUARD_EVENT_PREFIX,
)
from gx1.scripts.materialize_entry_pretest_test_guard_v1 import (
    materialize_pretest_test_guard,
)


RUN_ID = "PRETEST_GUARD_MATERIALIZER_PYTEST_V1"
BOUNDARY = "2026-07-01T00:00:00+00:00"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_materializer_binds_only_green_train_val_liveness(tmp_path: Path) -> None:
    dataset = (tmp_path / "dataset").resolve()
    authority = (tmp_path / "authority").resolve()
    dataset.mkdir()
    authority.mkdir()
    paths = {
        "train_manifest": dataset / "entry_train.manifest.json",
        "train_parquet": dataset / "entry_train.parquet",
        "val_manifest": dataset / "entry_val.manifest.json",
        "val_parquet": dataset / "entry_val.parquet",
        "proof": dataset / "DATASET_BUILD_PROOF.json",
        "liveness": dataset / "ENTRY_FULL_INPUT_LIVENESS_20260830T010000Z.json",
    }
    paths["train_parquet"].write_bytes(b"train")
    paths["val_parquet"].write_bytes(b"val")
    for split in ("train", "val"):
        _write_json(
            paths[f"{split}_manifest"],
            {
                "output_data_path": str(paths[f"{split}_parquet"]),
                "extra": {
                    "entry_run_id": RUN_ID,
                    "pretest_only": True,
                    "pretest_test_guard": {
                        "test_accessed": False,
                        "test_boundary_utc": BOUNDARY,
                    },
                },
            },
        )
    _write_json(
        paths["proof"],
        {
            "entry_run_id": RUN_ID,
            "pretest_only": True,
            "pretest_test_guard": {
                "test_accessed": False,
                "test_boundary_utc": BOUNDARY,
            },
        },
    )
    from gx1.scripts.materialize_entry_pretest_test_guard_v1 import _sha256_file

    train_manifest_sha = _sha256_file(paths["train_manifest"])
    val_manifest_sha = _sha256_file(paths["val_manifest"])
    train_parquet_sha = "a" * 64
    val_parquet_sha = "b" * 64
    _write_json(
        paths["liveness"],
        {
            "decision": "PASS",
            "dataset_dir": str(dataset),
            "input_bindings": {
                "split_manifests": {
                    "train": {
                        "path": str(paths["train_manifest"]),
                        "sha256": train_manifest_sha,
                        "observed_sha256": train_manifest_sha,
                    },
                    "val": {
                        "path": str(paths["val_manifest"]),
                        "sha256": val_manifest_sha,
                        "observed_sha256": val_manifest_sha,
                    },
                },
                "fullscan_proof": {
                    "train": {
                        "parquet_path": str(paths["train_parquet"]),
                        "parquet_sha256": train_parquet_sha,
                        "fullscan": True,
                        "scan_complete": True,
                        "scanned_rows": 3,
                        "total_rows": 3,
                    },
                    "val": {
                        "parquet_path": str(paths["val_parquet"]),
                        "parquet_sha256": val_parquet_sha,
                        "fullscan": True,
                        "scan_complete": True,
                        "scanned_rows": 2,
                        "total_rows": 2,
                    },
                },
            },
        },
    )
    out = authority / f"{PRETEST_TEST_GUARD_EVENT_PREFIX}_20260830T010203000000Z.json"
    result = materialize_pretest_test_guard(
        dataset_dir=dataset,
        train_manifest=paths["train_manifest"],
        train_parquet=paths["train_parquet"],
        train_parquet_sha256=train_parquet_sha,
        val_manifest=paths["val_manifest"],
        val_parquet=paths["val_parquet"],
        val_parquet_sha256=val_parquet_sha,
        dataset_build_proof=paths["proof"],
        full_input_liveness=paths["liveness"],
        out_json=out,
        created_utc="2026-08-30T01:02:03+00:00",
    )
    assert result["path"] == str(out)
    event = json.loads(out.read_text(encoding="utf-8"))
    assert event["test_accessed"] is False
    assert event["test_materialized"] is False
    assert "test_manifest" not in event and "test_parquet" not in event
