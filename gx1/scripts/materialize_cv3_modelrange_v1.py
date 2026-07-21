"""Materialize the exact common-history cv3 model-range source.

The active Entry source cascade needs the causal ``atr`` normalization source
that canonical-v3 omits from its model feature surface. Ten retired duplicate/
XGB-only columns are not restored, and five constant legacy cost/regime fields
are removed from the Entry projection. This producer joins the one required
column from exact row-aligned canonical-v2 bytes, trims the declared causal
history window, and binds every input/output byte plus the Entry run lineage in
one immutable provenance sidecar. Existing outputs, symlinks, row-local joins
and implicit/latest discovery are forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id


SCHEMA_VERSION = "cv3_modelrange_provenance_v5"
PRODUCER = "gx1.scripts.materialize_cv3_modelrange_v1"
PRODUCER_VERSION = "v4"
EXPECTED_CV3_COLUMN_COUNT = 113
EXPECTED_OUTPUT_COLUMN_COUNT = 109
DEFAULT_START_UTC = "2020-11-13T00:00:00Z"
EXTRA_COLUMNS_FROM_CANONICAL_V2 = (
    "atr",
)
ENTRY_DEAD_CONSTANT_COLUMNS = (
    "_v1_atr_regime_id",
    "_v1_spread_p",
    "_v1_slip_bps",
    "_v1_spread_z",
    "_v1_cost_bps_est",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc(value: Any, *, field: str) -> pd.Timestamp:
    try:
        parsed = pd.to_datetime(value, utc=True, errors="raise")
    except Exception as exc:
        raise RuntimeError(f"CV3_MODELRANGE_TIMESTAMP_INVALID: {field}={value!r}") from exc
    if pd.isna(parsed):
        raise RuntimeError(f"CV3_MODELRANGE_TIMESTAMP_INVALID: {field}={value!r}")
    return pd.Timestamp(parsed)


def _require_regular_input(path: Path, *, label: str) -> Path:
    candidate = path.expanduser().resolve()
    if path.is_symlink() or not candidate.is_file():
        raise RuntimeError(f"CV3_MODELRANGE_{label}_MISSING_OR_SYMLINK: {candidate}")
    return candidate


def _require_fresh_output(path: Path) -> tuple[Path, Path]:
    candidate = path.expanduser().resolve()
    parent = candidate.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise RuntimeError(f"CV3_MODELRANGE_OUTPUT_PARENT_SYMLINK: {parent}")
    sidecar = candidate.with_suffix(".provenance.json")
    if candidate.exists() or candidate.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise RuntimeError(f"CV3_MODELRANGE_OUTPUT_NOT_FRESH: {candidate}")
    return candidate, sidecar


def _load_exact_frame(path: Path, *, label: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "time" not in frame.columns:
        raise RuntimeError(f"CV3_MODELRANGE_{label}_TIME_MISSING")
    if frame.columns.duplicated().any():
        raise RuntimeError(f"CV3_MODELRANGE_{label}_DUPLICATE_COLUMNS")
    parsed = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    if parsed.isna().any():
        raise RuntimeError(f"CV3_MODELRANGE_{label}_TIME_INVALID")
    if parsed.duplicated().any() or not parsed.is_monotonic_increasing:
        raise RuntimeError(f"CV3_MODELRANGE_{label}_TIME_ORDER_INVALID")
    out = frame.copy()
    out["time"] = parsed
    return out


def _require_finite_numeric(frame: pd.DataFrame) -> None:
    for name in frame.columns:
        if name == "time":
            continue
        try:
            values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=np.float64)
        except Exception as exc:
            raise RuntimeError(f"CV3_MODELRANGE_COLUMN_NONNUMERIC: {name}") from exc
        if not np.isfinite(values).all():
            raise RuntimeError(f"CV3_MODELRANGE_COLUMN_NONFINITE: {name}")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o644)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write: {temporary}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    entry_run_id = require_entry_run_id(getattr(args, "run_id", ""))
    cv3_path = _require_regular_input(Path(args.cv3), label="CV3")
    canonical_v2_path = _require_regular_input(
        Path(args.canonical_v2), label="CANONICAL_V2"
    )
    output_path, sidecar_path = _require_fresh_output(Path(args.out))
    start = _parse_utc(args.start, field="start")
    end = _parse_utc(args.end, field="end")
    if end < start:
        raise RuntimeError("CV3_MODELRANGE_WINDOW_INVALID")

    cv3_sha = _sha256_file(cv3_path)
    canonical_v2_sha = _sha256_file(canonical_v2_path)
    cv3 = _load_exact_frame(cv3_path, label="CV3")
    canonical_v2 = _load_exact_frame(canonical_v2_path, label="CANONICAL_V2")
    if len(cv3.columns) != EXPECTED_CV3_COLUMN_COUNT:
        raise RuntimeError(
            "CV3_MODELRANGE_CV3_WIDTH_INVALID: "
            f"got={len(cv3.columns)} expected={EXPECTED_CV3_COLUMN_COUNT}"
        )
    missing_dead = [name for name in ENTRY_DEAD_CONSTANT_COLUMNS if name not in cv3]
    if missing_dead:
        raise RuntimeError(f"CV3_MODELRANGE_DEAD_COLUMNS_MISSING: {missing_dead}")
    missing = [name for name in EXTRA_COLUMNS_FROM_CANONICAL_V2 if name not in canonical_v2]
    collisions = [name for name in EXTRA_COLUMNS_FROM_CANONICAL_V2 if name in cv3]
    if missing:
        raise RuntimeError(f"CV3_MODELRANGE_CANONICAL_V2_COLUMNS_MISSING: {missing}")
    if collisions:
        raise RuntimeError(f"CV3_MODELRANGE_CV3_COLUMN_COLLISIONS: {collisions}")
    if len(cv3) != len(canonical_v2) or not np.array_equal(
        cv3["time"].array.asi8, canonical_v2["time"].array.asi8
    ):
        raise RuntimeError("CV3_MODELRANGE_SOURCE_TIME_ALIGNMENT_MISMATCH")

    merged = cv3.drop(columns=list(ENTRY_DEAD_CONSTANT_COLUMNS)).copy()
    for name in EXTRA_COLUMNS_FROM_CANONICAL_V2:
        merged[name] = canonical_v2[name].to_numpy(copy=True)
    selected = merged.loc[(merged["time"] >= start) & (merged["time"] <= end)].copy()
    if selected.empty:
        raise RuntimeError("CV3_MODELRANGE_WINDOW_EMPTY")
    if len(selected.columns) != EXPECTED_OUTPUT_COLUMN_COUNT:
        raise RuntimeError(
            "CV3_MODELRANGE_OUTPUT_WIDTH_INVALID: "
            f"got={len(selected.columns)} expected={EXPECTED_OUTPUT_COLUMN_COUNT}"
        )
    _require_finite_numeric(selected)

    temporary = output_path.with_name(
        f".{output_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        selected.to_parquet(temporary, index=False)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    if _sha256_file(cv3_path) != cv3_sha or _sha256_file(canonical_v2_path) != canonical_v2_sha:
        raise RuntimeError("CV3_MODELRANGE_SOURCE_CHANGED_DURING_BUILD")

    report = {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "producer_version": PRODUCER_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "entry_run_id": entry_run_id,
        "inputs": {
            "cv3": str(cv3_path),
            "cv3_sha256": cv3_sha,
            "canonical_v2": str(canonical_v2_path),
            "canonical_v2_sha256": canonical_v2_sha,
        },
        "extra_columns_from_canonical_v2": list(EXTRA_COLUMNS_FROM_CANONICAL_V2),
        "entry_dead_constant_columns_removed": list(ENTRY_DEAD_CONSTANT_COLUMNS),
        "model_range": {"start_utc": start.isoformat(), "end_utc": end.isoformat()},
        "rows": int(len(selected)),
        "columns": int(len(selected.columns)),
        "time_min_utc": pd.Timestamp(selected["time"].iloc[0]).isoformat(),
        "time_max_utc": pd.Timestamp(selected["time"].iloc[-1]).isoformat(),
        "output": str(output_path),
        "output_sha256": _sha256_file(output_path),
        "no_row_local_join": True,
        "exact_source_time_alignment_required": True,
        "all_output_features_finite": True,
    }
    _atomic_write_json(sidecar_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cv3", type=Path, required=True)
    parser.add_argument("--canonical-v2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default=DEFAULT_START_UTC)
    parser.add_argument(
        "--end",
        required=True,
        help="Explicit inclusive UTC model-range cutoff; no stale default is allowed.",
    )
    return parser


def main() -> int:
    report = run(build_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
