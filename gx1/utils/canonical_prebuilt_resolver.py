from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict

import pyarrow.parquet as pq


def resolve_base28_canonical_from_manifest(manifest_path: str) -> Dict[str, object]:
    """
    Resolve BASE28 canonical prebuilt strictly from manifest.

    Requirements:
    - kind == "BASE28_CANONICAL_MANIFEST"
    - parquet_path exists
    - sha256(parquet_path) == parquet_sha256 in manifest
    Returns dict with parquet_path, parquet_sha256, rows, cols_total.
    """
    mp = Path(manifest_path).expanduser().resolve()
    if not mp.exists():
        raise RuntimeError(f"BASE28_MANIFEST_INVALID: manifest not found: {mp}")

    try:
        obj = json.loads(mp.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"BASE28_MANIFEST_INVALID: cannot read/parse {mp}: {e}")

    if obj.get("kind") != "BASE28_CANONICAL_MANIFEST":
        raise RuntimeError(f"BASE28_MANIFEST_INVALID: kind != BASE28_CANONICAL_MANIFEST (kind={obj.get('kind')})")

    parquet_path = Path(obj.get("parquet_path") or "").expanduser().resolve()
    parquet_sha = obj.get("parquet_sha256") or ""
    cols_total = obj.get("cols_total")
    rows = obj.get("rows")

    if not parquet_path.exists():
        raise RuntimeError(f"BASE28_PARQUET_NOT_FOUND: {parquet_path}")
    if not parquet_sha:
        raise RuntimeError("BASE28_MANIFEST_INVALID: parquet_sha256 missing")

    # Compute sha256
    h = hashlib.sha256()
    with open(parquet_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    actual_sha = h.hexdigest()
    if actual_sha != parquet_sha:
        raise RuntimeError(f"BASE28_SHA_MISMATCH: expected={parquet_sha} got={actual_sha} path={parquet_path}")

    # Sanity: schema + rows/cols
    # Canonical BASE28 parquet on disk: time as column (UTC), index=False.
    # Forbidden index-like columns: never allow hidden index artifacts.
    forbidden_index_columns = {"__index_level_0__", "index_level_0", "timestamp", "datetime"}
    schema_names = pq.ParquetFile(parquet_path).schema_arrow.names
    bad = [c for c in schema_names if c in forbidden_index_columns]
    if bad:
        raise RuntimeError(
            f"BASE28_CANONICAL_FORBIDDEN_TIME_COLUMN: found={bad} (forbidden index columns). "
            f"Canonical parquet must have 'time' as column (UTC) and no __index_level_0__/index_level_0/timestamp/datetime."
        )

    md = pq.read_metadata(parquet_path)
    cols = len(schema_names)  # explicit: all columns on disk (includes time)
    num_rows = md.num_rows
    if num_rows <= 0:
        raise RuntimeError(f"BASE28_MANIFEST_INVALID: parquet_rows={num_rows} (must be > 0) path={parquet_path}")
    if cols_total is None:
        raise RuntimeError("BASE28_MANIFEST_INVALID: cols_total missing")
    if cols_total != cols:
        raise RuntimeError(f"BASE28_MANIFEST_INVALID: cols_total={cols_total} parquet_cols={cols}")

    return {
        "parquet_path": str(parquet_path),
        "parquet_sha256": actual_sha,
        "rows": num_rows,
        "cols_total": cols,
    }
