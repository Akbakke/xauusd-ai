#!/usr/bin/env python3
"""Build a read-only, fail-closed plan for the local lifecycle-v2 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.unified_exit_lifecycle_v2 import terminal_state_counts_sha256
from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import (
    require_no_cap_authority,
    require_no_cap_authority_sources,
)


SCHEMA_VERSION = "gx1_unified_exit_lifecycle_v2_local_pilot_readiness_v1"
PILOT_RUN_ID = "GX1_EXIT_LIFECYCLE_V2_LOCAL_PILOT_20260910"
TRAIN_START = "2025-06-01T00:00:00+00:00"
TRAIN_END = "2026-06-01T00:00:00+00:00"
VAL_START = TRAIN_END
VAL_END = "2026-07-01T00:00:00+00:00"
PILOT_EPOCHS = 1
STAGE_RECEIPT_SCHEMA_VERSION = "gx1_lifecycle_v2_pilot_stage_receipt_v1"
REQUIRED_SOURCE_BINDINGS = (
    "train_manifest",
    "train_parquet",
    "val_manifest",
    "val_parquet",
    "unified_exit_lifecycle_manifest",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _regular_absolute(path: Path, label: str) -> Path:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve() != path
    ):
        raise RuntimeError(f"PILOT_{label}_FILE_INVALID")
    return path


def _read_bound_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    path = _regular_absolute(path, label)
    if _sha256_file(path) != expected_sha256:
        raise RuntimeError(f"PILOT_{label}_HASH_MISMATCH")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"PILOT_{label}_JSON_INVALID") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"PILOT_{label}_JSON_INVALID")
    return value


def _source_binding(recipe: Mapping[str, Any], name: str) -> tuple[Path, str]:
    bindings = recipe.get("artifact_bindings")
    value = bindings.get(name) if isinstance(bindings, Mapping) else None
    if not isinstance(value, Mapping):
        raise RuntimeError(f"PILOT_SOURCE_BINDING_MISSING_{name.upper()}")
    path = Path(str(value.get("path") or ""))
    digest = value.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError(f"PILOT_SOURCE_BINDING_INVALID_{name.upper()}")
    _regular_absolute(path, f"SOURCE_{name.upper()}")
    if _sha256_file(path) != digest:
        raise RuntimeError(f"PILOT_SOURCE_BINDING_HASH_MISMATCH_{name.upper()}")
    return path, digest


def _window_selection(
    path: Path, *, start: str, end: str, label: str
) -> dict[str, Any]:
    values = pd.DatetimeIndex(
        pd.to_datetime(pd.read_parquet(path, columns=["time"])["time"], utc=True)
    ).as_unit("ns")
    if values.empty or values.hasnans or not values.is_unique or not values.is_monotonic_increasing:
        raise RuntimeError(f"PILOT_{label}_CLOCK_INVALID")
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    selected = np.flatnonzero((values >= start_time) & (values < end_time)).astype(
        np.int64
    )
    if selected.size == 0:
        raise RuntimeError(f"PILOT_{label}_WINDOW_EMPTY")
    selected_times = np.asarray(values.asi8[selected], dtype="<i8")
    return {
        "source_rows": len(values),
        "selected_rows": int(selected.size),
        "first_source_row_index": int(selected[0]),
        "last_source_row_index": int(selected[-1]),
        "observed_first_utc": values[selected[0]].isoformat(),
        "observed_last_utc": values[selected[-1]].isoformat(),
        "source_row_indices_sha256": hashlib.sha256(
            np.ascontiguousarray(selected, dtype="<i8").tobytes()
        ).hexdigest(),
        "selected_clock_sha256": hashlib.sha256(selected_times.tobytes()).hexdigest(),
    }


def _m1_binding(
    train_manifest: Mapping[str, Any], val_manifest: Mapping[str, Any]
) -> dict[str, str]:
    train_authority = (
        train_manifest.get("extra", {})
        .get("unified_exit_lifecycle", {})
        .get("m1_authority", {})
    )
    val_authority = (
        val_manifest.get("extra", {})
        .get("unified_exit_lifecycle", {})
        .get("m1_authority", {})
    )
    keys = {
        "parquet": ("m1_source_path", "m1_source_sha256"),
        "manifest": ("m1_source_manifest_path", "m1_source_manifest_sha256"),
    }
    result: dict[str, str] = {}
    for name, (path_key, sha_key) in keys.items():
        path = Path(str(train_authority.get(path_key) or ""))
        digest = train_authority.get(sha_key)
        if (
            not isinstance(digest, str)
            or val_authority.get(path_key) != str(path)
            or val_authority.get(sha_key) != digest
            or _sha256_file(_regular_absolute(path, f"M1_SOURCE_{name.upper()}"))
            != digest
        ):
            raise RuntimeError("PILOT_SOURCE_M1_LINEAGE_INVALID")
        result[f"{name}_path"] = str(path)
        result[f"{name}_sha256"] = digest
    if train_authority.get("test_accessed") is not False or val_authority.get(
        "test_accessed"
    ) is not False:
        raise RuntimeError("PILOT_SOURCE_M1_LINEAGE_INVALID")
    return result


def _optional_stage_receipt(
    path: Path | None, *, stage: str, pilot_binding_sha256: str
) -> dict[str, Any] | None:
    if path is None:
        return None
    label = f"{stage.upper()}_RECEIPT"
    path = _regular_absolute(path, label)
    digest = _sha256_file(path)
    value = _read_bound_json(path, digest, label)
    expected_keys = {
        "schema_version",
        "decision",
        "stage",
        "pilot_binding_sha256",
        "artifact_bindings",
        "test_accessed",
        "receipt_sha256",
    }
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    bindings = value.get("artifact_bindings")
    if (
        set(value) != expected_keys
        or value.get("schema_version") != STAGE_RECEIPT_SCHEMA_VERSION
        or value.get("decision") != "PASS"
        or value.get("stage") != stage
        or value.get("pilot_binding_sha256") != pilot_binding_sha256
        or value.get("test_accessed") is not False
        or value.get("receipt_sha256") != _canonical_sha256(unsigned)
        or not isinstance(bindings, Mapping)
        or not bindings
    ):
        raise RuntimeError(f"PILOT_{label}_NOT_PASS")
    for name, binding in bindings.items():
        if not isinstance(name, str) or not isinstance(binding, Mapping):
            raise RuntimeError(f"PILOT_{label}_ARTIFACT_BINDING_INVALID")
        if set(binding) != {"path", "sha256"}:
            raise RuntimeError(f"PILOT_{label}_ARTIFACT_BINDING_INVALID")
        artifact = Path(str(binding.get("path") or ""))
        artifact_sha = binding.get("sha256")
        if (
            not isinstance(artifact_sha, str)
            or _sha256_file(_regular_absolute(artifact, f"{label}_{name.upper()}"))
            != artifact_sha
        ):
            raise RuntimeError(f"PILOT_{label}_ARTIFACT_BINDING_INVALID")
    return {"path": str(path), "sha256": digest, "payload": value}


def _optional_economic_authority(
    path: Path | None,
    *,
    split: str,
    dataset_run_id: str,
    selected_rows: int,
) -> dict[str, Any] | None:
    if path is None:
        return None
    label = f"{split.upper()}_ECONOMICS_AUTHORITY"
    path = _regular_absolute(path, label)
    digest = _sha256_file(path)
    value = _read_bound_json(path, digest, label)
    terminal_hash = terminal_state_counts_sha256(
        {
            (entry_row, side_index): None
            for entry_row in range(selected_rows)
            for side_index in (0, 1)
        }
    )
    checked = require_no_cap_authority(
        value,
        expected_split=split,
        expected_dataset_run_id=dataset_run_id,
        expected_terminal_state_counts_sha256=terminal_hash,
    )
    require_no_cap_authority_sources(checked)
    return {"path": str(path), "sha256": digest, "payload": checked}


def _optional_child_admission(
    path: Path | None, *, plan: Mapping[str, Any]
) -> dict[str, Any] | None:
    if path is None:
        return None
    path = _regular_absolute(path, "CHILD_VIEW_ADMISSION")
    digest = _sha256_file(path)
    value = _read_bound_json(path, digest, "CHILD_VIEW_ADMISSION")
    splits = value.get("splits")
    if (
        value.get("schema_version")
        != "gx1_lifecycle_v2_pilot_child_view_admission_v1"
        or value.get("decision") != "PASS"
        or value.get("pilot_run_id") != plan["pilot_run_id"]
        or value.get("parent_dataset_run_id") != plan["dataset_run_id"]
        or value.get("child_dataset_run_id") != plan["pilot_dataset_run_id"]
        or value.get("pilot_binding_sha256") != plan["pilot_binding_sha256"]
        or value.get("parent_v1_root_path")
        != plan["source_bindings"]["unified_exit_lifecycle_manifest"]["path"]
        or value.get("parent_v1_root_sha256")
        != plan["source_bindings"]["unified_exit_lifecycle_manifest"]["sha256"]
        or value.get("m1_source_binding") != plan["m1_source_binding"]
        or value.get("test_accessed") is not False
        or value.get("witness_sha256")
        != _canonical_sha256(
            {key: item for key, item in value.items() if key != "witness_sha256"}
        )
        or not isinstance(splits, Mapping)
        or set(splits) != {"train", "val"}
    ):
        raise RuntimeError("PILOT_CHILD_VIEW_ADMISSION_NOT_PASS")
    for split in ("train", "val"):
        observed = splits[split]
        selection = plan["selection_bindings"][split]
        if (
            not isinstance(observed, Mapping)
            or observed.get("rows") != selection["selected_rows"]
            or observed.get("source_row_indices_sha256")
            != selection["source_row_indices_sha256"]
            or observed.get("clock_sha256") != selection["selected_clock_sha256"]
        ):
            raise RuntimeError("PILOT_CHILD_VIEW_ADMISSION_NOT_PASS")
        for kind in ("parquet", "manifest"):
            artifact = Path(str(observed.get(f"{kind}_path") or ""))
            expected_sha = observed.get(
                "parquet_sha256" if kind == "parquet" else "manifest_file_sha256"
            )
            if _sha256_file(_regular_absolute(artifact, "CHILD_VIEW_ARTIFACT")) != expected_sha:
                raise RuntimeError("PILOT_CHILD_VIEW_ADMISSION_NOT_PASS")
    return {"path": str(path), "sha256": digest, "payload": value}


def build_pilot_readiness(
    *,
    source_recipe_path: Path,
    source_recipe_sha256: str,
    pilot_root: Path,
    entry_window_adoption_receipt: Path | None = None,
    child_view_admission: Path | None = None,
    train_economics_authority: Path | None = None,
    val_economics_authority: Path | None = None,
    lifecycle_validate_receipt: Path | None = None,
    normalization_receipt: Path | None = None,
    first_state_receipt: Path | None = None,
    launch_recipe: Path | None = None,
) -> dict[str, Any]:
    """Inspect source bytes and return commands; never write or execute them."""

    source_recipe_path = source_recipe_path.expanduser().resolve()
    pilot_root = pilot_root.expanduser().resolve()
    recipe = _read_bound_json(
        source_recipe_path, source_recipe_sha256, "SOURCE_RECIPE"
    )
    source: dict[str, dict[str, str]] = {}
    for name in REQUIRED_SOURCE_BINDINGS:
        path, digest = _source_binding(recipe, name)
        source[name] = {"path": str(path), "sha256": digest}
    train_manifest = _read_bound_json(
        Path(source["train_manifest"]["path"]),
        source["train_manifest"]["sha256"],
        "SOURCE_TRAIN_MANIFEST",
    )
    val_manifest = _read_bound_json(
        Path(source["val_manifest"]["path"]),
        source["val_manifest"]["sha256"],
        "SOURCE_VAL_MANIFEST",
    )
    dataset_run_id = train_manifest.get("extra", {}).get("entry_run_id")
    if (
        not isinstance(dataset_run_id, str)
        or not dataset_run_id
        or val_manifest.get("extra", {}).get("entry_run_id") != dataset_run_id
        or train_manifest.get("output_data_path") != source["train_parquet"]["path"]
        or val_manifest.get("output_data_path") != source["val_parquet"]["path"]
        or train_manifest.get("extra", {}).get("pretest_test_guard", {}).get(
            "test_accessed"
        )
        is not False
        or val_manifest.get("extra", {}).get("pretest_test_guard", {}).get(
            "test_accessed"
        )
        is not False
    ):
        raise RuntimeError("PILOT_SOURCE_ENTRY_LINEAGE_INVALID")
    selections = {
        "train": _window_selection(
            Path(source["train_parquet"]["path"]),
            start=TRAIN_START,
            end=TRAIN_END,
            label="TRAIN",
        ),
        "val": _window_selection(
            Path(source["val_parquet"]["path"]),
            start=VAL_START,
            end=VAL_END,
            label="VAL",
        ),
    }
    m1_source = _m1_binding(train_manifest, val_manifest)
    pilot_dataset_run_id = f"{dataset_run_id}__PILOT_20250601_20260630"
    pilot_binding = {
        "schema_version": SCHEMA_VERSION,
        "pilot_run_id": PILOT_RUN_ID,
        "pilot_dataset_run_id": pilot_dataset_run_id,
        "source_recipe_sha256": source_recipe_sha256,
        "source_bindings": source,
        "m1_source_binding": m1_source,
        "windows": {
            "train": {"start_utc": TRAIN_START, "end_utc_exclusive": TRAIN_END},
            "val": {"start_utc": VAL_START, "end_utc_exclusive": VAL_END},
        },
        "selection_bindings": selections,
        "planned_epochs": PILOT_EPOCHS,
    }
    pilot_binding_sha256 = _canonical_sha256(pilot_binding)
    plan_context = {
        **pilot_binding,
        "dataset_run_id": dataset_run_id,
        "pilot_binding_sha256": pilot_binding_sha256,
    }
    receipts = {
        "entry_window_adoption": _optional_stage_receipt(
            entry_window_adoption_receipt,
            stage="entry_window_adoption",
            pilot_binding_sha256=pilot_binding_sha256,
        ),
        "child_view_admission": _optional_child_admission(
            child_view_admission, plan=plan_context
        ),
        "train_economics": _optional_economic_authority(
            train_economics_authority,
            split="train",
            dataset_run_id=pilot_dataset_run_id,
            selected_rows=selections["train"]["selected_rows"],
        ),
        "val_economics": _optional_economic_authority(
            val_economics_authority,
            split="val",
            dataset_run_id=pilot_dataset_run_id,
            selected_rows=selections["val"]["selected_rows"],
        ),
        "lifecycle_validate": _optional_stage_receipt(
            lifecycle_validate_receipt,
            stage="lifecycle_validate_no_publish",
            pilot_binding_sha256=pilot_binding_sha256,
        ),
        "normalization": _optional_stage_receipt(
            normalization_receipt,
            stage="train_only_normalization",
            pilot_binding_sha256=pilot_binding_sha256,
        ),
        "first_state": _optional_stage_receipt(
            first_state_receipt,
            stage="first_state_train_val",
            pilot_binding_sha256=pilot_binding_sha256,
        ),
        "launch_recipe": _optional_stage_receipt(
            launch_recipe,
            stage="canonical_launch_recipe",
            pilot_binding_sha256=pilot_binding_sha256,
        ),
    }
    required_order = list(receipts)
    first_incomplete = next(
        (index for index, name in enumerate(required_order) if receipts[name] is None),
        len(required_order),
    )
    if any(receipts[name] is not None for name in required_order[first_incomplete + 1 :]):
        raise RuntimeError("PILOT_STAGE_ORDER_INVALID")
    blocked = required_order[first_incomplete:]
    entry_dir = pilot_root / "ENTRY_WINDOW"
    lifecycle_dir = pilot_root / "LIFECYCLE_V2"
    economics_dir = pilot_root / "ECONOMICS"
    validate_command = [
        "python",
        "-m",
        "gx1.scripts.materialize_unified_exit_lifecycle_v2_pilot_child_v1",
        "--source-recipe",
        str(source_recipe_path),
        "--source-recipe-sha256",
        source_recipe_sha256,
        "--pilot-root",
        str(pilot_root),
        "--child-root",
        str(entry_dir / "ENTRY_WINDOW_ROOT.json"),
        "--output-dir",
        str(lifecycle_dir),
        "--train-economic-authority",
        str(economics_dir / "train.authority.json"),
        "--val-economic-authority",
        str(economics_dir / "val.authority.json"),
        "--m1-source-parquet",
        m1_source["parquet_path"],
        "--m1-source-manifest",
        m1_source["manifest_path"],
        "--planned-epochs",
        str(PILOT_EPOCHS),
    ]
    launch_dry_run = [
        "python",
        "gx1/scripts/run_entry_model_native_pretest_technical_train_v1.py",
        "--recipe-json",
        str(pilot_root / "LAUNCH" / "pilot.recipe.json"),
        "--recipe-sha256",
        "<sha256-from-launch-receipt>",
        "--dry-run",
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS" if not blocked else "BLOCKED",
        "pilot_run_id": PILOT_RUN_ID,
        "source_recipe": {
            "path": str(source_recipe_path),
            "sha256": source_recipe_sha256,
            "historical_recipe_is_launch_authority": False,
        },
        "dataset_run_id": dataset_run_id,
        "pilot_dataset_run_id": pilot_dataset_run_id,
        "windows": {
            "train": {"start_utc": TRAIN_START, "end_utc_exclusive": TRAIN_END},
            "val": {"start_utc": VAL_START, "end_utc_exclusive": VAL_END},
        },
        "source_bindings": source,
        "m1_source_binding": m1_source,
        "selection_bindings": selections,
        "pilot_binding": pilot_binding,
        "pilot_binding_sha256": pilot_binding_sha256,
        "pilot_epochs": PILOT_EPOCHS,
        "normalization_policy": {
            "fit_splits": ["train"],
            "reuse_five_year_normalization": False,
            "val_may_fit": False,
        },
        "first_state_gate": {
            "entry_row_index": 0,
            "splits": ["train", "val"],
            "requires_lifecycle_and_normalization_sha256": True,
        },
        "stage_order": required_order,
        "receipts": receipts,
        "missing_or_blocked_stages": blocked,
        "commands": {
            "lifecycle_validate_no_publish": validate_command,
            "lifecycle_publish_after_validate_pass": [
                *validate_command,
                "--publish",
            ],
            "canonical_launch_dry_run": launch_dry_run,
        },
        "cuda_execution_authorized": False,
        "test_accessed": False,
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-recipe", required=True, type=Path)
    parser.add_argument("--source-recipe-sha256", required=True)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--entry-window-adoption-receipt", type=Path)
    parser.add_argument("--child-view-admission", type=Path)
    parser.add_argument("--train-economics-authority", type=Path)
    parser.add_argument("--val-economics-authority", type=Path)
    parser.add_argument("--lifecycle-validate-receipt", type=Path)
    parser.add_argument("--normalization-receipt", type=Path)
    parser.add_argument("--first-state-receipt", type=Path)
    parser.add_argument("--launch-recipe", type=Path)
    parser.add_argument("--dry-run", action="store_true", required=True)
    args = parser.parse_args()
    report = build_pilot_readiness(
        source_recipe_path=args.source_recipe,
        source_recipe_sha256=args.source_recipe_sha256,
        pilot_root=args.pilot_root,
        entry_window_adoption_receipt=args.entry_window_adoption_receipt,
        child_view_admission=args.child_view_admission,
        train_economics_authority=args.train_economics_authority,
        val_economics_authority=args.val_economics_authority,
        lifecycle_validate_receipt=args.lifecycle_validate_receipt,
        normalization_receipt=args.normalization_receipt,
        first_state_receipt=args.first_state_receipt,
        launch_recipe=args.launch_recipe,
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
