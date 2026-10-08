#!/usr/bin/env python3
"""Verify a pilot Entry child view against its exact parent v1 admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gx1.contracts.immutable_event_authority_v1 import _fsync_directory, _publish_file_noreplace
from gx1.contracts.unified_exit_lifecycle_v1 import (
    UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION,
)
from gx1.scripts.materialize_unified_exit_lifecycle_v2 import (
    _require_full_v1_admission,
)
from gx1.scripts.prepare_unified_exit_lifecycle_v2_pilot_v1 import (
    _canonical_sha256,
    _read_bound_json,
    _regular_absolute,
    _sha256_file,
    build_pilot_readiness,
)


SCHEMA_VERSION = "gx1_lifecycle_v2_pilot_child_view_admission_v1"


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = _regular_absolute(path.expanduser().resolve(), label)
    return _read_bound_json(path, _sha256_file(path), label)


def _require_child_split(
    *,
    split: str,
    root_path: Path,
    root: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    raw_binding = root.get("splits", {}).get(split)
    if not isinstance(raw_binding, Mapping) or set(raw_binding) != {
        "parquet",
        "manifest",
        "manifest_contract_sha256",
    }:
        raise RuntimeError("PILOT_CHILD_VIEW_SPLIT_BINDING_INVALID")
    parquet_binding = raw_binding["parquet"]
    manifest_binding = raw_binding["manifest"]
    if (
        not isinstance(parquet_binding, Mapping)
        or set(parquet_binding) != {"path", "sha256"}
        or not isinstance(manifest_binding, Mapping)
        or set(manifest_binding) != {"path", "sha256"}
    ):
        raise RuntimeError("PILOT_CHILD_VIEW_SPLIT_BINDING_INVALID")
    parquet_path = _regular_absolute(
        Path(str(parquet_binding["path"] or "")), f"CHILD_{split.upper()}_PARQUET"
    )
    manifest_path = _regular_absolute(
        Path(str(manifest_binding["path"] or "")), f"CHILD_{split.upper()}_MANIFEST"
    )
    reuse_complete = "pretest_m1_rebinding" in plan["pilot_binding"]
    expected_parquet = (
        Path(plan["source_bindings"][f"{split}_parquet"]["path"])
        if reuse_complete else root_path.parent / f"{split}.parquet"
    )
    if (
        parquet_path != expected_parquet
        or manifest_path.parent != root_path.parent
        or manifest_path.name != f"{split}.manifest.json"
        or _sha256_file(parquet_path) != parquet_binding["sha256"]
        or _sha256_file(manifest_path) != manifest_binding["sha256"]
    ):
        raise RuntimeError("PILOT_CHILD_VIEW_SPLIT_HASH_INVALID")
    manifest = _read_json(manifest_path, f"CHILD_{split.upper()}_MANIFEST")
    unsigned_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    selection = plan["selection_bindings"][split]
    expected_window = plan["windows"][split]
    source_parquet = plan["source_bindings"][f"{split}_parquet"]
    source_manifest = plan["source_bindings"][f"{split}_manifest"]
    if (
        manifest.get("decision") != "PASS"
        or manifest.get("split") != split
        or manifest.get("pilot_run_id") != plan["pilot_run_id"]
        or manifest.get("pilot_dataset_run_id") != plan["pilot_dataset_run_id"]
        or manifest.get("pilot_binding_sha256") != plan["pilot_binding_sha256"]
        or manifest.get("window_start_utc") != expected_window["start_utc"]
        or manifest.get("window_end_utc_exclusive")
        != expected_window["end_utc_exclusive"]
        or manifest.get("rows") != selection["selected_rows"]
        or manifest.get("clock_sha256") != selection["selected_clock_sha256"]
        or manifest.get("source_row_indices_sha256")
        != selection["source_row_indices_sha256"]
        or manifest.get("source_parquet") != source_parquet
        or manifest.get("source_manifest") != source_manifest
        or manifest.get("output_parquet_path") != str(parquet_path)
        or manifest.get("output_parquet_sha256") != parquet_binding["sha256"]
        or manifest.get("manifest_sha256") != _canonical_sha256(unsigned_manifest)
        or raw_binding["manifest_contract_sha256"] != manifest["manifest_sha256"]
        or manifest.get("test_accessed") is not False
    ):
        raise RuntimeError("PILOT_CHILD_VIEW_MANIFEST_INVALID")
    child_clock = pd.DatetimeIndex(
        pd.to_datetime(pd.read_parquet(parquet_path, columns=["time"])["time"], utc=True)
    ).as_unit("ns")
    parent_clock = pd.DatetimeIndex(
        pd.to_datetime(
            pd.read_parquet(Path(source_parquet["path"]), columns=["time"])["time"],
            utc=True,
        )
    ).as_unit("ns")
    indices = np.arange(
        selection["first_source_row_index"],
        selection["last_source_row_index"] + 1,
        dtype=np.int64,
    )
    selected = parent_clock[indices]
    if (
        len(indices) != selection["selected_rows"]
        or (reuse_complete and (
            len(indices) != len(parent_clock)
            or parquet_binding != source_parquet
            or indices[0] != 0 or indices[-1] != len(parent_clock) - 1
        ))
        or not child_clock.equals(selected)
        or pq.ParquetFile(parquet_path).metadata.num_rows != len(indices)
    ):
        raise RuntimeError("PILOT_CHILD_VIEW_CLOCK_IDENTITY_INVALID")
    return {
        "parquet_path": str(parquet_path),
        "parquet_sha256": parquet_binding["sha256"],
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": manifest_binding["sha256"],
        "manifest_contract_sha256": manifest["manifest_sha256"],
        "rows": len(indices),
        "source_row_indices_sha256": selection["source_row_indices_sha256"],
        "clock_sha256": selection["selected_clock_sha256"],
    }


def _require_parent_entry_admission(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Adopt Entry bytes; never carry old M1 state coordinates into the new source."""

    rebinding = plan["pilot_binding"].get("pretest_m1_rebinding")
    if rebinding is None:
        return _require_full_v1_admission(
            entry_paths={
                split: Path(plan["source_bindings"][f"{split}_parquet"]["path"])
                for split in ("train", "val")
            },
            entry_manifest_paths={
                split: Path(plan["source_bindings"][f"{split}_manifest"]["path"])
                for split in ("train", "val")
            },
            dataset_run_id=plan["dataset_run_id"],
        )
    if (
        "chronological_learning_design" not in plan["pilot_binding"]
        or not isinstance(plan["pilot_binding"].get("source_test_guard_lineage"), Mapping)
        or rebinding.get("physical_row_indices_reusable") is not False
    ):
        raise RuntimeError("PILOT_CHILD_PARENT_ENTRY_SCOPE_INVALID")
    binding = plan["source_bindings"]["unified_exit_lifecycle_manifest"]
    path = Path(binding["path"])
    root = _read_bound_json(path, binding["sha256"], "PARENT_ENTRY_LINEAGE")
    original_sha = rebinding["original_m1_authority_sha256"]
    if (
        root.get("schema_version") != UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION
        or root.get("decision") != "PASS"
        or root.get("entry_run_id") != plan["dataset_run_id"]
        or root.get("m1_authority_sha256") != original_sha
        or _canonical_sha256(root.get("m1_authority")) != original_sha
        or not isinstance(root.get("splits"), Mapping)
        or any(
            root["splits"].get(split, {}).get("entry_dataset_path")
            != plan["source_bindings"][f"{split}_parquet"]["path"]
            or root["splits"].get(split, {}).get("entry_dataset_sha256")
            != plan["source_bindings"][f"{split}_parquet"]["sha256"]
            for split in ("train", "val")
        )
    ):
        raise RuntimeError("PILOT_CHILD_PARENT_ENTRY_LINEAGE_INVALID")
    return {"root_manifest_path": path, "root_manifest_sha256": binding["sha256"]}


def require_pilot_child_calendar(admission: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the frozen physical calendar for the explicit new preparation route."""

    binding = admission.get("chronological_learning_design")
    if binding is None:
        return None
    if (
        not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}
        or admission.get("decision") != "PASS"
        or admission.get("test_accessed") is not False
        or admission.get("parent_admission_scope") != "frozen_entry_bytes_only_no_parent_m1_states"
        or admission.get("witness_sha256") != _canonical_sha256(
            {k: v for k, v in admission.items() if k != "witness_sha256"}
        )
    ):
        raise RuntimeError("PILOT_CHILD_CALENDAR_ADMISSION_INVALID")
    design = _read_bound_json(Path(binding["path"]), binding["sha256"], "CHILD_CALENDAR_DESIGN")
    calendar = design.get("calendar", {})
    if (
        design.get("schema_version") != "gx1_frozen_chronological_learning_design_v1"
        or calendar.get("physical_source_splits") != {"train": "train", "control": "val"}
        or calendar.get("physical_coordinate_namespaces_are_separate") is not True
    ):
        raise RuntimeError("PILOT_CHILD_CALENDAR_DESIGN_INVALID")
    boundaries = [
        pd.Timestamp(calendar[key]) for key in (
            "train_entry_start_inclusive", "train_control_cutoff",
            "development_control_entry_end_exclusive",
        )
    ]
    if (
        any(pd.isna(t) or t.tzinfo is None for t in boundaries)
        or not boundaries[0] < boundaries[1] < boundaries[2]
    ):
        raise RuntimeError("PILOT_CHILD_CALENDAR_BOUNDARIES_INVALID")
    windows = {
        split: {"start_utc": boundaries[i].tz_convert("UTC").isoformat(),
                "end_utc_exclusive": boundaries[i + 1].tz_convert("UTC").isoformat()}
        for i, split in enumerate(("train", "val"))
    }
    if admission.get("windows") != windows:
        raise RuntimeError("PILOT_CHILD_CALENDAR_WINDOWS_CHANGED")
    for split in ("train", "val"):
        child = admission["splits"][split]
        physical = calendar["source_bindings"][split]
        # This route reuses the entire parent parquet: feature bytes as well
        # as row counts and clocks remain bound to the frozen design.
        if (
            child["rows"] != physical["physical_rows"]
            or child["clock_sha256"] != physical["clock_sha256"]
            or {"path": child["parquet_path"], "sha256": child["parquet_sha256"]}
            != physical["parquet"]
        ):
            raise RuntimeError("PILOT_CHILD_CALENDAR_PHYSICAL_SOURCE_CHANGED")
    return windows


def validate_pilot_child_view(
    *,
    source_recipe_path: Path,
    source_recipe_sha256: str,
    pilot_root: Path,
    child_root_path: Path,
) -> dict[str, Any]:
    """Re-open parent admission and every child byte; return an exact witness."""

    plan = build_pilot_readiness(
        source_recipe_path=source_recipe_path,
        source_recipe_sha256=source_recipe_sha256,
        pilot_root=pilot_root,
    )
    parent = _require_parent_entry_admission(plan)
    root_path = _regular_absolute(
        child_root_path.expanduser().resolve(), "CHILD_ROOT"
    )
    if root_path != pilot_root.expanduser().resolve() / "ENTRY_WINDOW" / root_path.name:
        raise RuntimeError("PILOT_CHILD_VIEW_ROOT_PATH_INVALID")
    root = _read_json(root_path, "CHILD_ROOT")
    if (
        root.get("decision") != "PASS"
        or root.get("published") is not True
        or root.get("pilot_run_id") != plan["pilot_run_id"]
        or root.get("pilot_dataset_run_id") != plan["pilot_dataset_run_id"]
        or root.get("pilot_binding_sha256") != plan["pilot_binding_sha256"]
        or root.get("source_recipe") != plan["source_recipe"]
        or root.get("source_bindings") != plan["source_bindings"]
        or root.get("selection_bindings") != plan["selection_bindings"]
        or root.get("test_accessed") is not False
        or root.get("contract_sha256")
        != _canonical_sha256(
            {key: value for key, value in root.items() if key != "contract_sha256"}
        )
    ):
        raise RuntimeError("PILOT_CHILD_VIEW_ROOT_INVALID")
    splits = {
        split: _require_child_split(
            split=split, root_path=root_path, root=root, plan=plan
        )
        for split in ("train", "val")
    }
    witness = {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS",
        "pilot_run_id": plan["pilot_run_id"],
        "parent_dataset_run_id": plan["dataset_run_id"],
        "child_dataset_run_id": plan["pilot_dataset_run_id"],
        "pilot_binding_sha256": plan["pilot_binding_sha256"],
        "parent_v1_root_path": str(parent["root_manifest_path"]),
        "parent_v1_root_sha256": parent["root_manifest_sha256"],
        "child_root_path": str(root_path),
        "child_root_sha256": _sha256_file(root_path),
        "splits": splits,
        "m1_source_binding": plan["m1_source_binding"],
        "test_accessed": False,
    }
    if "pretest_m1_rebinding" in plan["pilot_binding"]:
        witness.update({
            "chronological_learning_design": plan["pilot_binding"]["chronological_learning_design"],
            "windows": plan["windows"],
            "parent_admission_scope": "frozen_entry_bytes_only_no_parent_m1_states",
        })
    witness["witness_sha256"] = _canonical_sha256(witness)
    require_pilot_child_calendar(witness)
    return witness


def publish_pilot_child_view_admission(
    *,
    source_recipe_path: Path,
    source_recipe_sha256: str,
    pilot_root: Path,
    child_root_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Publish one immutable witness only after all parent/child bytes pass."""

    root = pilot_root.expanduser().resolve()
    output = output_path.expanduser().resolve()
    if output != root / "ADMISSION" / "CHILD_VIEW_ADMISSION.json":
        raise RuntimeError("PILOT_CHILD_VIEW_ADMISSION_OUTPUT_PATH_INVALID")
    witness = validate_pilot_child_view(
        source_recipe_path=source_recipe_path,
        source_recipe_sha256=source_recipe_sha256,
        pilot_root=root,
        child_root_path=child_root_path,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    raw = json.dumps(witness, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    # Strict-load and preserve failed staging before existing no-replace publication.
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    if temporary.read_bytes() != raw or json.loads(temporary.read_text()) != witness:
        raise RuntimeError("PILOT_CHILD_VIEW_ADMISSION_STAGING_INVALID")
    _publish_file_noreplace(temporary, output)
    _fsync_directory(output.parent)
    if _sha256_file(output) != hashlib.sha256(raw).hexdigest():
        raise RuntimeError("PILOT_CHILD_VIEW_ADMISSION_WRITE_INVALID")
    return {
        "path": str(output),
        "sha256": _sha256_file(output),
        "witness": witness,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-recipe", required=True, type=Path)
    parser.add_argument("--source-recipe-sha256", required=True)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--child-root", required=True, type=Path)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()
    if args.out_json is None:
        report = validate_pilot_child_view(
            source_recipe_path=args.source_recipe,
            source_recipe_sha256=args.source_recipe_sha256,
            pilot_root=args.pilot_root,
            child_root_path=args.child_root,
        )
    else:
        report = publish_pilot_child_view_admission(
            source_recipe_path=args.source_recipe,
            source_recipe_sha256=args.source_recipe_sha256,
            pilot_root=args.pilot_root,
            child_root_path=args.child_root,
            output_path=args.out_json,
        )
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
