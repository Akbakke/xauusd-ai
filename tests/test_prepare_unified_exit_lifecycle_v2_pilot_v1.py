from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from gx1.scripts.prepare_unified_exit_lifecycle_v2_pilot_v1 import (
    PILOT_RUN_ID,
    STAGE_RECEIPT_SCHEMA_VERSION,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    _canonical_sha256,
    build_pilot_readiness,
)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _source_fixture(tmp_path: Path) -> tuple[Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    m1 = tmp_path / "m1.parquet"
    pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2025-05-31T23:59:00Z", "2025-06-01T00:00:00Z"]
            )
        }
    ).to_parquet(m1, index=False)
    m1_manifest = tmp_path / "m1.manifest.json"
    _write_json(m1_manifest, {"source": "synthetic-train-val-only"})
    m1_authority = {
        "m1_source_path": str(m1),
        "m1_source_sha256": _file_sha(m1),
        "m1_source_manifest_path": str(m1_manifest),
        "m1_source_manifest_sha256": _file_sha(m1_manifest),
        "test_accessed": False,
    }
    paths: dict[str, Path] = {}
    clocks = {
        "train": [
            "2025-05-31T23:55:00Z",
            "2025-06-01T00:00:00Z",
            "2026-05-31T23:55:00Z",
            "2026-06-01T00:00:00Z",
        ],
        "val": ["2026-06-01T00:00:00Z", "2026-06-30T23:55:00Z"],
    }
    for split in ("train", "val"):
        parquet = tmp_path / f"{split}.parquet"
        pd.DataFrame({"time": pd.to_datetime(clocks[split])}).to_parquet(
            parquet, index=False
        )
        manifest = tmp_path / f"{split}.manifest.json"
        _write_json(
            manifest,
            {
                "output_data_path": str(parquet),
                "extra": {
                    "entry_run_id": "SOURCE_RUN",
                    "pretest_test_guard": {"test_accessed": False},
                    "unified_exit_lifecycle": {"m1_authority": m1_authority},
                },
            },
        )
        paths[f"{split}_parquet"] = parquet
        paths[f"{split}_manifest"] = manifest
    lifecycle = tmp_path / "UNIFIED_EXIT_LIFECYCLE_MANIFEST.json"
    _write_json(lifecycle, {"historical": True})
    paths["unified_exit_lifecycle_manifest"] = lifecycle
    recipe = tmp_path / "source.recipe.json"
    _write_json(
        recipe,
        {
            "artifact_bindings": {
                name: {"path": str(path), "sha256": _file_sha(path)}
                for name, path in paths.items()
            }
        },
    )
    return recipe, _file_sha(recipe)


def _build(tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    recipe, digest = _source_fixture(tmp_path)
    return build_pilot_readiness(
        source_recipe_path=recipe,
        source_recipe_sha256=digest,
        pilot_root=tmp_path / "pilot-output-must-not-exist",
        **kwargs,
    )


def _stage_receipt(
    tmp_path: Path, *, stage: str, pilot_binding_sha256: str
) -> Path:
    artifact = tmp_path / f"{stage}.artifact"
    artifact.write_bytes(stage.encode())
    payload = {
        "schema_version": STAGE_RECEIPT_SCHEMA_VERSION,
        "decision": "PASS",
        "stage": stage,
        "pilot_binding_sha256": pilot_binding_sha256,
        "artifact_bindings": {
            "output": {"path": str(artifact), "sha256": _file_sha(artifact)}
        },
        "test_accessed": False,
    }
    payload["receipt_sha256"] = _canonical_sha256(payload)
    path = tmp_path / f"{stage}.receipt.json"
    _write_json(path, payload)
    return path


def test_dry_readiness_is_exact_blocked_and_does_not_write(tmp_path: Path) -> None:
    pilot_root = tmp_path / "pilot-output-must-not-exist"
    report = _build(tmp_path)

    assert report["decision"] == "BLOCKED"
    assert report["pilot_run_id"] == PILOT_RUN_ID
    assert report["windows"] == {
        "train": {"start_utc": TRAIN_START, "end_utc_exclusive": TRAIN_END},
        "val": {"start_utc": VAL_START, "end_utc_exclusive": VAL_END},
    }
    assert report["selection_bindings"]["train"]["selected_rows"] == 2
    assert report["selection_bindings"]["val"]["selected_rows"] == 2
    assert report["normalization_policy"] == {
        "fit_splits": ["train"],
        "reuse_five_year_normalization": False,
        "val_may_fit": False,
    }
    validate = report["commands"]["lifecycle_validate_no_publish"]
    assert "--publish" not in validate
    assert "--m1-source-parquet" in validate
    assert "--m1-source-manifest" in validate
    assert "--execute" not in validate
    assert report["commands"]["lifecycle_publish_after_validate_pass"][-1] == (
        "--publish"
    )
    assert "--execute" not in report["commands"]["canonical_launch_dry_run"]
    assert report["commands"]["canonical_launch_dry_run"][-1] == "--dry-run"
    assert report["cuda_execution_authorized"] is False
    assert not pilot_root.exists()


def test_source_parquet_swap_is_rejected(tmp_path: Path) -> None:
    recipe, digest = _source_fixture(tmp_path)
    recipe_value = json.loads(recipe.read_text(encoding="utf-8"))
    train = Path(recipe_value["artifact_bindings"]["train_parquet"]["path"])
    train.write_bytes(b"swapped")

    with pytest.raises(RuntimeError, match="SOURCE_BINDING_HASH_MISMATCH_TRAIN_PARQUET"):
        build_pilot_readiness(
            source_recipe_path=recipe,
            source_recipe_sha256=digest,
            pilot_root=tmp_path / "pilot",
        )


def test_unrelated_or_out_of_order_receipt_cannot_advance(tmp_path: Path) -> None:
    recipe, digest = _source_fixture(tmp_path)
    report = build_pilot_readiness(
        source_recipe_path=recipe,
        source_recipe_sha256=digest,
        pilot_root=tmp_path / "pilot",
    )
    wrong = _stage_receipt(
        tmp_path,
        stage="entry_window_adoption",
        pilot_binding_sha256="0" * 64,
    )
    with pytest.raises(RuntimeError, match="ENTRY_WINDOW_ADOPTION_RECEIPT_NOT_PASS"):
        build_pilot_readiness(
            source_recipe_path=recipe,
            source_recipe_sha256=digest,
            pilot_root=tmp_path / "pilot",
            entry_window_adoption_receipt=wrong,
        )

    later = _stage_receipt(
        tmp_path,
        stage="lifecycle_validate_no_publish",
        pilot_binding_sha256=report["pilot_binding_sha256"],
    )
    with pytest.raises(RuntimeError, match="PILOT_STAGE_ORDER_INVALID"):
        build_pilot_readiness(
            source_recipe_path=recipe,
            source_recipe_sha256=digest,
            pilot_root=tmp_path / "pilot",
            lifecycle_validate_receipt=later,
        )
