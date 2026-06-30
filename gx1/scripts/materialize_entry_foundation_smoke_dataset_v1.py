#!/usr/bin/env python3
"""Materialize a small Entry foundation seq146 smoke dataset.

This does not rebuild features or labels. It samples rows from the already
audited foundation seq146 train/val/test parquets, copies their split manifests,
and rewrites manifest paths/row counts so the canonical trainer can run a fast
plumbing smoke without loading the full 11GB train split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gx1.scripts.verify_entry_foundation_state_v1 import (
    FEATURE_AUDIT_LATEST,
    FOUNDATION_DATASET_DIR,
    RUN_ROOT,
    SPECIALIST_AUDIT_LATEST,
    TARGET_AUDIT_LATEST,
)


DEFAULT_OUT_DIR = (
    RUN_ROOT / "v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_foundation_seq146_smoke"
)
DEFAULT_STEM = "v10_foundation_seq146_smoke__HOLD_03B"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": None,
            "mtime_ns": None,
            "sha256": None,
        }
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(path),
    }


def _audit_summary(path: Path) -> dict[str, Any]:
    row = _artifact_fingerprint(path)
    if not path.exists():
        row.update({"schema_version": None, "decision": None, "dataset_dir": None, "created_utc": None})
        return row
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        row.update({"read_error": f"{type(exc).__name__}: {exc}"})
        return row
    row.update(
        {
            "schema_version": report.get("schema_version"),
            "decision": report.get("decision"),
            "dataset_dir": report.get("dataset_dir"),
            "created_utc": report.get("created_utc"),
            "json_path": report.get("json_path"),
        }
    )
    return row


def _audit_provenance(args: argparse.Namespace, source_dir: Path) -> dict[str, Any]:
    artifacts = {
        "feature_audit": _audit_summary(Path(getattr(args, "feature_audit_json", FEATURE_AUDIT_LATEST)).expanduser()),
        "target_audit": _audit_summary(Path(getattr(args, "target_audit_json", TARGET_AUDIT_LATEST)).expanduser()),
        "specialist_audit": _audit_summary(
            Path(getattr(args, "specialist_audit_json", SPECIALIST_AUDIT_LATEST)).expanduser()
        ),
    }
    return {
        "schema_version": "entry_foundation_smoke_dataset_audit_provenance_v1",
        "source_dataset_dir": str(source_dir),
        "artifacts": artifacts,
        "all_artifacts_present": all(bool(row.get("exists")) for row in artifacts.values()),
        "all_artifact_hashes_present": all(
            isinstance(row.get("sha256"), str) and len(str(row.get("sha256"))) == 64
            for row in artifacts.values()
        ),
    }


def _source_split_path(source_dir: Path, split: str) -> Path:
    matches = sorted(source_dir.glob(f"*_{split}.parquet"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one source {split} parquet under {source_dir}, got {matches}")
    return matches[0]


def _split_manifest_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(".manifest.json")


def _spread_positions(positions: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or len(positions) == 0:
        return np.asarray([], dtype=np.int64)
    if count >= len(positions):
        return positions.astype(np.int64, copy=False)
    pick = np.linspace(0, len(positions) - 1, num=count, dtype=np.int64)
    return positions[pick].astype(np.int64, copy=False)


def _selected_global_indices(pf: pq.ParquetFile, *, max_rows: int, batch_size: int) -> np.ndarray:
    label_chunks: list[np.ndarray] = []
    for batch in pf.iter_batches(batch_size=batch_size, columns=["y_direction"]):
        labels = batch.column(batch.schema.get_field_index("y_direction")).to_numpy(zero_copy_only=False)
        label_chunks.append(np.asarray(labels, dtype=np.int64))
    if not label_chunks:
        return np.asarray([], dtype=np.int64)

    labels = np.concatenate(label_chunks)
    per_class_target = max(1, int(max_rows // 3))
    selected: list[np.ndarray] = []
    selected_set: set[int] = set()
    for label in (0, 1, 2):
        positions = np.flatnonzero(labels == label).astype(np.int64, copy=False)
        picked = _spread_positions(positions, min(per_class_target, len(positions)))
        selected.append(picked)
        selected_set.update(int(i) for i in picked)

    if selected:
        out = np.unique(np.concatenate(selected)).astype(np.int64, copy=False)
    else:
        out = np.asarray([], dtype=np.int64)
    if len(out) < max_rows:
        remaining = np.asarray([i for i in range(len(labels)) if i not in selected_set], dtype=np.int64)
        fill = _spread_positions(remaining, max_rows - len(out))
        out = np.unique(np.concatenate([out, fill])).astype(np.int64, copy=False)
    return np.sort(out[:max_rows]).astype(np.int64, copy=False)


def _sample_split(source_path: Path, out_path: Path, *, max_rows: int, batch_size: int) -> dict[str, Any]:
    pf = pq.ParquetFile(source_path)
    if "y_direction" not in pf.schema_arrow.names:
        raise RuntimeError(f"source split lacks y_direction: {source_path}")
    selected_indices = _selected_global_indices(pf, max_rows=max_rows, batch_size=batch_size)
    if len(selected_indices) == 0:
        raise RuntimeError(f"no rows selected from {source_path}")
    selected_set = set(int(i) for i in selected_indices)
    selected: list[pa.Table] = []
    offset = 0

    for batch in pf.iter_batches(batch_size=batch_size):
        batch_len = int(batch.num_rows)
        local_idx = [
            int(global_idx - offset)
            for global_idx in selected_indices
            if offset <= int(global_idx) < offset + batch_len
        ]
        if local_idx:
            table = pa.Table.from_batches([batch])
            selected.append(table.take(pa.array(local_idx, type=pa.int64())))
        offset += batch_len
        if offset > int(selected_indices[-1]):
            break

    if not selected:
        raise RuntimeError(f"no rows selected from {source_path}")
    out = pa.concat_tables(selected, promote_options="default")
    if out.num_rows > max_rows:
        out = out.slice(0, max_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, out_path, compression="zstd")

    labels = np.asarray(out.column("y_direction").to_pylist(), dtype=np.int64)
    time_min = None
    time_max = None
    if "time" in out.column_names and out.num_rows:
        times = out.column("time").to_pylist()
        time_min = str(min(times))
        time_max = str(max(times))
    return {
        "source_path": str(source_path),
        "out_path": str(out_path),
        "source_rows": int(pf.metadata.num_rows),
        "rows": int(out.num_rows),
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        "selection_strategy": "class_balanced_global_time_spread_v2",
        "selected_source_index_min": int(selected_indices[0]),
        "selected_source_index_max": int(selected_indices[-1]),
        "time_min": time_min,
        "time_max": time_max,
    }


def _copy_split_manifest(
    source_manifest: Path,
    out_manifest: Path,
    *,
    out_parquet: Path,
    split: str,
    sample: dict[str, Any],
    split_schema_version: str | None = None,
    manifest_variant: str | None = None,
    expected_seq_snap_width: int | None = None,
) -> None:
    data = json.loads(source_manifest.read_text(encoding="utf-8"))
    if split_schema_version:
        data["schema_version"] = split_schema_version
    if manifest_variant:
        data["manifest_variant"] = manifest_variant
    if expected_seq_snap_width and expected_seq_snap_width > 0:
        data["expected_seq_snap_width"] = int(expected_seq_snap_width)
    data["output_data_path"] = str(out_parquet)
    data["foundation_smoke_dataset_v1"] = {
        "source_manifest": str(source_manifest),
        "source_output_data_path": str(sample["source_path"]),
        "sample_rows": int(sample["rows"]),
        "source_rows": int(sample["source_rows"]),
        "split": split,
        "label_counts": sample["label_counts"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    extra = data.setdefault("extra", {})
    if isinstance(extra, dict):
        extra["rows"] = int(sample["rows"])
        extra["foundation_smoke_dataset_v1"] = data["foundation_smoke_dataset_v1"]
    data["ts_min_max_by_split"] = {
        split: {
            "min": sample.get("time_min"),
            "max": sample.get("time_max"),
        }
    }
    out_manifest.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = Path(args.source_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    stem = str(args.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    row_limits = {
        "train": int(args.train_rows),
        "val": int(args.val_rows),
        "test": int(args.test_rows),
    }
    manifest_variant = str(getattr(args, "manifest_variant", "") or "")
    expected_seq_snap_width = int(getattr(args, "expected_seq_snap_width", 0) or 0)
    samples: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        source_path = _source_split_path(source_dir, split)
        out_path = out_dir / f"{stem}_{split}.parquet"
        sample = _sample_split(source_path, out_path, max_rows=row_limits[split], batch_size=int(args.batch_size))
        source_manifest = _split_manifest_path(source_path)
        if not source_manifest.exists():
            raise RuntimeError(f"source split manifest missing: {source_manifest}")
        out_manifest = out_path.with_suffix(".manifest.json")
        _copy_split_manifest(
            source_manifest,
            out_manifest,
            out_parquet=out_path,
            split=split,
            sample=sample,
            split_schema_version=str(getattr(args, "split_schema_version", "") or ""),
            manifest_variant=manifest_variant,
            expected_seq_snap_width=expected_seq_snap_width,
        )
        sample.update(
            {
                "source_manifest": str(source_manifest),
                "source_manifest_sha256": _sha256_file(source_manifest),
                "out_manifest": str(out_manifest),
                "out_manifest_sha256": _sha256_file(out_manifest),
                "out_parquet_sha256": _sha256_file(out_path),
            }
        )
        samples[split] = sample

    manifest = {
        "schema_version": str(getattr(args, "schema_version", "entry_foundation_seq146_smoke_dataset_v1")),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "out_dir": str(out_dir),
        "stem": stem,
        "splits": samples,
        "audit_provenance": _audit_provenance(args, source_dir),
    }
    if manifest_variant:
        manifest["manifest_variant"] = manifest_variant
    if expected_seq_snap_width > 0:
        manifest["expected_seq_snap_width"] = expected_seq_snap_width
    manifest_path = out_dir / "SMOKE_DATASET_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    if not args.quiet:
        print(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", default=str(FOUNDATION_DATASET_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--stem", default=DEFAULT_STEM)
    ap.add_argument("--train-rows", type=int, default=4095)
    ap.add_argument("--val-rows", type=int, default=1536)
    ap.add_argument("--test-rows", type=int, default=1536)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--feature-audit-json", default=str(FEATURE_AUDIT_LATEST))
    ap.add_argument("--target-audit-json", default=str(TARGET_AUDIT_LATEST))
    ap.add_argument("--specialist-audit-json", default=str(SPECIALIST_AUDIT_LATEST))
    ap.add_argument("--schema-version", default="entry_foundation_seq146_smoke_dataset_v1")
    ap.add_argument("--split-schema-version", default="")
    ap.add_argument("--manifest-variant", default="")
    ap.add_argument("--expected-seq-snap-width", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
