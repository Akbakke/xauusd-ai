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


def _chronological_fixture(tmp_path: Path) -> tuple[Path, str, Path]:
    """Synthetic clocks exercise admission only, never trading evidence."""
    import numpy as np

    recipe_path, _ = _source_fixture(tmp_path)
    recipe = json.loads(recipe_path.read_text())
    sources = recipe["artifact_bindings"]
    clocks = {
        "train": ["2011-06-01T00:00:00Z", "2025-05-30T12:55:00Z"],
        "val": ["2025-06-01T22:00:00Z", "2026-06-30T14:55:00Z"],
    }
    windows = {
        "train": {"start": "2011-06-01T00:00:00+00:00", "end": "2025-05-31T23:59:59+00:00"},
        "val": {"start": "2025-06-01T00:00:00+00:00", "end": "2026-06-30T23:59:59+00:00"},
    }
    populations = {}
    for split in ("train", "val"):
        parquet = Path(sources[f"{split}_parquet"]["path"])
        clock = pd.DatetimeIndex(pd.to_datetime(clocks[split], utc=True)).as_unit("ns")
        pd.DataFrame({"time": clock}).to_parquet(parquet, index=False)
        manifest = Path(sources[f"{split}_manifest"]["path"])
        value = json.loads(manifest.read_text())
        value["splits"] = windows
        _write_json(manifest, value)
        for kind in ("manifest", "parquet"):
            sources[f"{split}_{kind}"]["sha256"] = _file_sha(
                Path(sources[f"{split}_{kind}"]["path"])
            )
        populations[split] = {
            "manifest": sources[f"{split}_manifest"],
            "parquet": sources[f"{split}_parquet"],
            "declared_window": windows[split],
            "physical_rows": len(clock),
            "clock_sha256": hashlib.sha256(
                np.asarray(clock.asi8, dtype="<i8").tobytes()
            ).hexdigest(),
        }
    design = tmp_path / "DESIGN.json"
    _write_json(design, {
        "schema_version": "gx1_frozen_chronological_learning_design_v1",
        "calendar": {
            "physical_source_splits": {"train": "train", "control": "val"},
            "physical_coordinate_namespaces_are_separate": True,
            "train_entry_start_inclusive": windows["train"]["start"],
            "train_control_cutoff": windows["val"]["start"],
            "development_control_entry_end_exclusive": "2026-07-01T00:00:00+00:00",
            "source_bindings": populations,
        },
    })
    recipe["lifecycle_v2_data_scope"] = {
        "schema_version": "gx1_lifecycle_v2_full_train_data_scope_v1",
        "train_coverage": "entire_bound_train",
        "run_id": "FRESH_BOUND_CALENDAR",
        "chronological_learning_design": {"path": str(design), "sha256": _file_sha(design)},
    }
    _write_json(recipe_path, recipe)
    return recipe_path, _file_sha(recipe_path), design


def _rebind_design(recipe_path: Path, design: Path) -> str:
    recipe = json.loads(recipe_path.read_text())
    recipe["lifecycle_v2_data_scope"]["chronological_learning_design"]["sha256"] = _file_sha(design)
    _write_json(recipe_path, recipe)
    return _file_sha(recipe_path)


def test_frozen_physical_calendar_uses_both_real_source_windows(tmp_path: Path) -> None:
    recipe, digest, design = _chronological_fixture(tmp_path)
    plan = build_pilot_readiness(
        source_recipe_path=recipe, source_recipe_sha256=digest,
        pilot_root=tmp_path / "pilot",
    )
    assert plan["windows"] == {
        "train": {"start_utc": "2011-06-01T00:00:00+00:00", "end_utc_exclusive": "2025-06-01T00:00:00+00:00"},
        "val": {"start_utc": "2025-06-01T00:00:00+00:00", "end_utc_exclusive": "2026-07-01T00:00:00+00:00"},
    }
    assert plan["pilot_binding"]["chronological_learning_design"] == {
        "path": str(design), "sha256": _file_sha(design),
    }
    assert all(p["selected_rows"] == p["source_rows"] == 2 for p in plan["selection_bindings"].values())
    assert plan["cuda_execution_authorized"] is False
    assert plan["decision"] == "BLOCKED"
    assert not (tmp_path / "pilot").exists()


@pytest.mark.parametrize("change", ["wrong_physical_role", "swapped_source", "overlap", "naive_clock"])
def test_frozen_calendar_rejects_role_source_and_time_mismatches(tmp_path: Path, change: str) -> None:
    recipe, _, design = _chronological_fixture(tmp_path)
    value = json.loads(design.read_text())
    calendar = value["calendar"]
    if change == "wrong_physical_role":
        calendar["physical_source_splits"]["control"] = "train"
    elif change == "swapped_source":
        calendar["source_bindings"]["val"]["parquet"] = calendar["source_bindings"]["train"]["parquet"]
    elif change == "overlap":
        calendar["train_control_cutoff"] = calendar["train_entry_start_inclusive"]
    else:
        calendar["train_control_cutoff"] = "2025-06-01"
    _write_json(design, value)
    with pytest.raises(RuntimeError, match="CHRONOLOGICAL_(DESIGN|SOURCE_WINDOW)_INVALID"):
        build_pilot_readiness(
            source_recipe_path=recipe, source_recipe_sha256=_rebind_design(recipe, design),
            pilot_root=tmp_path / "pilot",
        )


@pytest.mark.parametrize("change", ["row_count", "clock"])
def test_frozen_calendar_checks_population_not_only_dates(tmp_path: Path, change: str) -> None:
    recipe, _, design = _chronological_fixture(tmp_path)
    value = json.loads(design.read_text())
    population = value["calendar"]["source_bindings"]["val"]
    population["physical_rows" if change == "row_count" else "clock_sha256"] = (
        3 if change == "row_count" else "0" * 64
    )
    _write_json(design, value)
    with pytest.raises(RuntimeError, match="CHRONOLOGICAL_POPULATION_MISMATCH"):
        build_pilot_readiness(
            source_recipe_path=recipe, source_recipe_sha256=_rebind_design(recipe, design),
            pilot_root=tmp_path / "pilot",
        )


def test_frozen_calendar_design_hash_cannot_be_replaced(tmp_path: Path) -> None:
    recipe, digest, design = _chronological_fixture(tmp_path)
    design.write_bytes(design.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="CHRONOLOGICAL_DESIGN_HASH_MISMATCH"):
        build_pilot_readiness(
            source_recipe_path=recipe, source_recipe_sha256=digest,
            pilot_root=tmp_path / "pilot",
        )


def test_legacy_full_train_calendar_keeps_original_endpoints(tmp_path: Path) -> None:
    from gx1.scripts.prepare_unified_exit_lifecycle_v2_pilot_v1 import _entry_window_scope

    scope = _entry_window_scope(
        {"lifecycle_v2_data_scope": {
            "schema_version": "gx1_lifecycle_v2_full_train_data_scope_v1",
            "train_coverage": "entire_bound_train", "run_id": "HISTORICAL",
        }},
        {"splits": {"train": {"start": "2021-06-01T00:00:00+00:00", "end": "2026-05-31T23:59:59+00:00"}}},
        dataset_run_id="HISTORICAL_SOURCE",
    )
    assert scope["windows"]["train"] == {
        "start_utc": "2021-06-01T00:00:00+00:00", "end_utc_exclusive": TRAIN_END,
    }
    assert scope["windows"]["val"] == {"start_utc": VAL_START, "end_utc_exclusive": VAL_END}
    assert "chronological_learning_design" not in scope
