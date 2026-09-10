#!/usr/bin/env python3
"""Materialize the exact TRAIN/VAL Entry window for the lifecycle-v2 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gx1.scripts.prepare_unified_exit_lifecycle_v2_pilot_v1 import (
    STAGE_RECEIPT_SCHEMA_VERSION,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    _canonical_sha256,
    _sha256_file,
    build_pilot_readiness,
)


SCHEMA_VERSION = "gx1_lifecycle_v2_pilot_entry_window_v1"
ROOT_SCHEMA_VERSION = "gx1_lifecycle_v2_pilot_entry_window_root_v1"


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
        + b"\n"
    )


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _validate_clock(values: pd.Series, *, start: str, end: str, split: str) -> None:
    clock = pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit("ns")
    if (
        clock.empty
        or clock.hasnans
        or not clock.is_unique
        or not clock.is_monotonic_increasing
        or clock[0] < pd.Timestamp(start)
        or clock[-1] >= pd.Timestamp(end)
    ):
        raise RuntimeError(f"PILOT_ENTRY_WINDOW_{split.upper()}_CLOCK_INVALID")


def _materialize_split(
    *,
    source_path: Path,
    output_path: Path,
    start: str,
    end: str,
    expected_rows: int,
) -> tuple[int, str, str]:
    source = pq.ParquetFile(source_path)
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for batch in source.iter_batches(batch_size=2048):
            table = pa.Table.from_batches([batch])
            clock = pd.DatetimeIndex(
                pd.to_datetime(table["time"].to_pandas(), utc=True)
            )
            keep = np.asarray(
                (clock >= pd.Timestamp(start)) & (clock < pd.Timestamp(end)),
                dtype=np.bool_,
            )
            if not keep.any():
                continue
            selected = table.filter(pa.array(keep))
            if writer is None:
                writer = pq.ParquetWriter(output_path, selected.schema)
            writer.write_table(selected)
            rows += selected.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None or rows != expected_rows:
        raise RuntimeError("PILOT_ENTRY_WINDOW_ROW_COUNT_INVALID")
    reopened = pq.ParquetFile(output_path)
    if reopened.metadata.num_rows != expected_rows or reopened.schema_arrow != source.schema_arrow:
        raise RuntimeError("PILOT_ENTRY_WINDOW_OUTPUT_INVALID")
    _validate_clock(
        pd.read_parquet(output_path, columns=["time"])["time"],
        start=start,
        end=end,
        split=output_path.stem,
    )
    return rows, _sha256_file(output_path), _schema_sha256(reopened.schema_arrow)


def materialize_pilot_entry_window(
    *,
    source_recipe_path: Path,
    source_recipe_sha256: str,
    pilot_root: Path,
    output_dir: Path,
    publish: bool,
) -> dict[str, Any]:
    """Validate exact source bytes and optionally publish an atomic subset."""

    if type(publish) is not bool:
        raise RuntimeError("PILOT_ENTRY_WINDOW_PUBLISH_INVALID")
    plan = build_pilot_readiness(
        source_recipe_path=source_recipe_path,
        source_recipe_sha256=source_recipe_sha256,
        pilot_root=pilot_root,
    )
    output = output_dir.expanduser().resolve()
    if output != pilot_root.expanduser().resolve() / "ENTRY_WINDOW":
        raise RuntimeError("PILOT_ENTRY_WINDOW_OUTPUT_PATH_INVALID")
    windows = {
        "train": (TRAIN_START, TRAIN_END),
        "val": (VAL_START, VAL_END),
    }
    preview = {
        "schema_version": ROOT_SCHEMA_VERSION,
        "decision": "READY_TO_MATERIALIZE",
        "published": False,
        "output_dir": str(output),
        "pilot_run_id": plan["pilot_run_id"],
        "pilot_dataset_run_id": plan["pilot_dataset_run_id"],
        "pilot_binding_sha256": plan["pilot_binding_sha256"],
        "source_recipe": plan["source_recipe"],
        "source_bindings": plan["source_bindings"],
        "selection_bindings": plan["selection_bindings"],
        "test_accessed": False,
    }
    preview["contract_sha256"] = _canonical_sha256(preview)
    if not publish:
        return preview
    if output.exists() or output.is_symlink():
        raise RuntimeError("PILOT_ENTRY_WINDOW_OUTPUT_EXISTS")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        split_bindings: dict[str, dict[str, Any]] = {}
        for split, (start, end) in windows.items():
            parquet = staging / f"{split}.parquet"
            source_parquet = Path(plan["source_bindings"][f"{split}_parquet"]["path"])
            rows, digest, schema_sha = _materialize_split(
                source_path=source_parquet,
                output_path=parquet,
                start=start,
                end=end,
                expected_rows=plan["selection_bindings"][split]["selected_rows"],
            )
            final_parquet = output / parquet.name
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "decision": "PASS",
                "split": split,
                "pilot_run_id": plan["pilot_run_id"],
                "pilot_dataset_run_id": plan["pilot_dataset_run_id"],
                "pilot_binding_sha256": plan["pilot_binding_sha256"],
                "window_start_utc": start,
                "window_end_utc_exclusive": end,
                "rows": rows,
                "clock_sha256": plan["selection_bindings"][split][
                    "selected_clock_sha256"
                ],
                "source_row_indices_sha256": plan["selection_bindings"][split][
                    "source_row_indices_sha256"
                ],
                "source_parquet": plan["source_bindings"][f"{split}_parquet"],
                "source_manifest": plan["source_bindings"][f"{split}_manifest"],
                "output_parquet_path": str(final_parquet),
                "output_parquet_sha256": digest,
                "output_schema_sha256": schema_sha,
                "test_accessed": False,
            }
            manifest["manifest_sha256"] = _canonical_sha256(manifest)
            manifest_name = f"{split}.manifest.json"
            (staging / manifest_name).write_bytes(_json_bytes(manifest))
            split_bindings[split] = {
                "parquet": {"path": str(final_parquet), "sha256": digest},
                "manifest": {
                    "path": str(output / manifest_name),
                    "sha256": _sha256_file(staging / manifest_name),
                },
                "manifest_contract_sha256": manifest["manifest_sha256"],
            }
        root = {
            **{key: value for key, value in preview.items() if key != "contract_sha256"},
            "decision": "PASS",
            "published": True,
            "splits": split_bindings,
        }
        root["contract_sha256"] = _canonical_sha256(root)
        root_path = staging / "ENTRY_WINDOW_ROOT.json"
        root_path.write_bytes(_json_bytes(root))
        final_root = output / root_path.name
        receipt = {
            "schema_version": STAGE_RECEIPT_SCHEMA_VERSION,
            "decision": "PASS",
            "stage": "entry_window_adoption",
            "pilot_binding_sha256": plan["pilot_binding_sha256"],
            "artifact_bindings": {
                "root": {"path": str(final_root), "sha256": _sha256_file(root_path)},
                **{
                    f"{split}_{kind}": binding[kind]
                    for split, binding in split_bindings.items()
                    for kind in ("parquet", "manifest")
                },
            },
            "test_accessed": False,
        }
        receipt["receipt_sha256"] = _canonical_sha256(receipt)
        (staging / "ENTRY_WINDOW_ADOPTION_RECEIPT.json").write_bytes(
            _json_bytes(receipt)
        )
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    for binding in receipt["artifact_bindings"].values():
        if _sha256_file(Path(binding["path"])) != binding["sha256"]:
            raise RuntimeError("PILOT_ENTRY_WINDOW_POST_PUBLISH_HASH_MISMATCH")
    return {"root": root, "receipt": receipt}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-recipe", required=True, type=Path)
    parser.add_argument("--source-recipe-sha256", required=True)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    result = materialize_pilot_entry_window(
        source_recipe_path=args.source_recipe,
        source_recipe_sha256=args.source_recipe_sha256,
        pilot_root=args.pilot_root,
        output_dir=args.output_dir,
        publish=args.publish,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
