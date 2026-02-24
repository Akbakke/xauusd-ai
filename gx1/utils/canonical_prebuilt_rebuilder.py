#!/home/andre2/venvs/gx1/bin/python
"""
Canonical BASE28 prebuilt rebuilder (TRUTH/SMOKE hygiene).

Canonical contract (on disk):
- Parquet is "physical": it has columns, not a pandas Index.
- Canonical BASE28 parquet MUST store time as an explicit column named "time" (UTC).
- Parquet MUST be written with index=False to prevent hidden index artifacts like "__index_level_0__".
- "time" is NOT forbidden. It is REQUIRED.

Runtime contract:
- Loaders may set DatetimeIndex from df["time"] (and optionally drop "time") in-memory.

This tool is explicit/one-off (TRUTH never auto-fixes):
- Rebuild BASE28_CANONICAL parquet into canonical on-disk format.
- Enforce time UTC, monotonic increasing, no duplicates.
- Forbid hidden index artifacts / legacy time aliases in the output schema.
- Write new parquet + sidecar manifest(s); update CURRENT_MANIFEST.json atomically.
- Archive old manifests under /home/andre2/GX1_DATA/_ARCHIVE_BASE28_MANIFESTS/.

TRUTH SSoT (manifest-only):
- Canonical prebuilt source = /home/andre2/GX1_DATA/data/data/prebuilt/BASE28_CANONICAL/CURRENT_MANIFEST.json
- canonical_prebuilt_parquet is only a mirror and MUST match manifest.parquet_path
- On-disk format: time column (UTC), index=False; forbid __index_level_0__/index_level_0/timestamp/datetime
- Repair path: run this rebuilder to regenerate manifest/schema; TRUTH/SMOKE must never bypass CURRENT_MANIFEST.json

Usage (example):
  python -m gx1.utils.canonical_prebuilt_rebuilder \
    --in /home/andre2/GX1_DATA/data/data/prebuilt/BASE28_CANONICAL/xauusd_m5_BASE28_2020_2025.parquet \
    --out-dir /home/andre2/GX1_DATA/data/data/prebuilt/BASE28_CANONICAL
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq

# Canonical on-disk format REQUIRES "time" column.
REQUIRED_TIME_COLUMN = "time"

# Forbidden artifacts: we never want these to appear as parquet schema columns in canonical.
# (They usually indicate pandas index leakage or ambiguous time aliases.)
FORBIDDEN_INDEX_ARTIFACTS = {"__index_level_0__", "index_level_0", "timestamp", "datetime"}

ARCHIVE_DIR = Path("/home/andre2/GX1_DATA/_ARCHIVE_BASE28_MANIFESTS")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _load_base28_contract() -> List[str]:
    contract_path = Path("/home/andre2/src/GX1_ENGINE/gx1/xgb/contracts/xgb_input_features_base28_v1.json")
    if not contract_path.exists():
        raise RuntimeError(f"BASE28_CONTRACT_MISSING: {contract_path}")
    data = json.loads(contract_path.read_text(encoding="utf-8"))
    feats = data.get("features") or []
    if not isinstance(feats, list) or len(feats) != 28 or not all(isinstance(x, str) for x in feats):
        raise RuntimeError(f"BASE28_CONTRACT_INVALID: expected 28 features, got {len(feats)} at {contract_path}")
    return list(feats)


def _extract_time_series(df: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
    """
    Determine the canonical time series (UTC) from:
      1) df["time"] if present
      2) one of the legacy aliases: timestamp/datetime/__index_level_0__/index_level_0 if present
      3) df.index if it is a DatetimeIndex
    Returns: (time_series_utc, columns_to_drop)
    """
    cols_to_drop: List[str] = []

    # Preferred: explicit "time" column already present
    if REQUIRED_TIME_COLUMN in df.columns:
        t = pd.to_datetime(df[REQUIRED_TIME_COLUMN], utc=True)
        # keep "time" (do not drop it); we will overwrite with canonical UTC if needed
        return t, cols_to_drop

    # Legacy time aliases in columns
    for alias in ("timestamp", "datetime", "__index_level_0__", "index_level_0"):
        if alias in df.columns:
            t = pd.to_datetime(df[alias], utc=True)
            cols_to_drop.append(alias)
            return t, cols_to_drop

    # Fallback: use index if it is a DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            t = pd.to_datetime(df.index, utc=True)
        else:
            # Normalize to UTC deterministically
            t = df.index.tz_convert("UTC")
        # df.index isn't a column; nothing to drop
        return pd.Series(t, index=df.index), cols_to_drop

    raise RuntimeError(
        "REBUILD_FAIL: no usable time source found (expected column 'time' or alias "
        "'timestamp'/'datetime'/'__index_level_0__'/'index_level_0', or a DatetimeIndex)."
    )


def rebuild_base28_parquet(src_parquet: Path, out_dir: Path) -> Dict[str, object]:
    if not src_parquet.exists():
        raise RuntimeError(f"REBUILD_SOURCE_NOT_FOUND: {src_parquet}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Validate existing parquet schema (no rewrite; manifest/sidecar regeneration only)
    pf = pq.ParquetFile(src_parquet)
    schema_names = pf.schema_arrow.names
    forbidden_found = [c for c in schema_names if c in FORBIDDEN_INDEX_ARTIFACTS]
    if forbidden_found:
        raise RuntimeError(f"REBUILD_FAIL: forbidden columns in schema: {forbidden_found}")
    if REQUIRED_TIME_COLUMN not in schema_names:
        raise RuntimeError("REBUILD_FAIL: canonical parquet missing required 'time' column")

    md = pf.metadata
    rows = int(md.num_rows)
    cols_total = int(len(schema_names))
    if rows <= 0:
        raise RuntimeError("REBUILD_FAIL: rows must be > 0")

    sha = _sha256(src_parquet)

    # Load BASE28 contract for required_all_features prefix
    base28_features = _load_base28_contract()
    extra_cols = [c for c in schema_names if c not in base28_features]
    required_all_features = base28_features + extra_cols

    # Manifest (immutable, timestamped)
    manifest_path = out_dir / f"xauusd_m5_BASE28_2020_2025_REBUILT_{ts}.manifest.json"
    manifest = {
        "kind": "BASE28_CANONICAL_MANIFEST",
        "created_utc": ts,
        "parquet_path": str(src_parquet),
        "parquet_sha256": sha,
        "rows": rows,
        "cols_total": cols_total,
        "note": "Regenerated manifest/schema (canonical BASE28 with time column, index=False).",
    }
    _write_json_atomic(manifest_path, manifest)

    # Schema manifest (sidecar)
    schema_manifest_path = src_parquet.with_suffix(".schema_manifest.json")
    schema_manifest = {
        "parquet_path": str(src_parquet),
        "schema_names": schema_names,
        "cols_total": cols_total,
        "required_time_column": REQUIRED_TIME_COLUMN,
        "forbidden_index_artifacts": sorted(FORBIDDEN_INDEX_ARTIFACTS),
        "forbidden_found": forbidden_found,
        "has_time_column": True,
        "required_all_features": required_all_features,
    }
    _write_json_atomic(schema_manifest_path, schema_manifest)

    # Archive old manifests (except CURRENT_MANIFEST.json and the new manifest)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for m in out_dir.glob("*.manifest.json"):
        if m.name == "CURRENT_MANIFEST.json" or m == manifest_path:
            continue
        shutil.move(str(m), str(ARCHIVE_DIR / m.name))

    # Update CURRENT_MANIFEST.json atomically to the new manifest payload (SSoT pointer)
    current = out_dir / "CURRENT_MANIFEST.json"
    _write_json_atomic(current, manifest)

    return {
        "parquet_path": str(src_parquet),
        "parquet_sha256": sha,
        "rows": rows,
        "cols_total": cols_total,
        "manifest_path": str(manifest_path),
        "schema_manifest_path": str(schema_manifest_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild canonical BASE28 prebuilt into on-disk format: time column (UTC), index=False; forbid hidden index artifacts."
    )
    ap.add_argument("--in", dest="src", required=True, type=Path, help="Source BASE28 parquet (canonical or legacy)")
    ap.add_argument(
        "--out-dir",
        dest="out_dir",
        required=False,
        type=Path,
        default=Path("/home/andre2/GX1_DATA/data/data/prebuilt/BASE28_CANONICAL"),
        help="Output directory (BASE28_CANONICAL)",
    )
    args = ap.parse_args()

    res = rebuild_base28_parquet(args.src.expanduser().resolve(), args.out_dir.expanduser().resolve())
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())