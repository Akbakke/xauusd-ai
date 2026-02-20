#!/usr/bin/env python3
"""
Prebuilt Feature Schema Gate for GX1 (TRUTH/SMOKE).

Purpose:
- Validate that the PREBUILT feature parquet passed via `--prebuilt-parquet` is structurally sane.
- Prebuilt features do NOT need OHLC candles; raw candles come from `--data`.

Checks (when truth_or_smoke=True):
- File exists and schema can be read
- Has a timestamp column (canonicalized to "ts") either via:
  - existing "ts" column, or
  - "time"/"timestamp" column (renamed to "ts" by ts_utils.ensure_ts_column), or
  - DatetimeIndex (handled at runtime; this gate only requires a timestamp column in schema)
- Has a minimal required feature set (SSoT guardrail)
- FORBIDS: any column starting with "prob_" (and writes fatal capsule)

On failure:
- Writes PREBUILT_SCHEMA_FATAL.json to output_dir (or /tmp fallback)
- Exits with code 2
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None


def _truth_or_smoke_env() -> bool:
    mode = os.getenv("GX1_RUN_MODE", "").upper()
    return os.getenv("GX1_TRUTH_MODE", "0") == "1" or mode in {"TRUTH", "SMOKE"}


def _resolve_bundle_dir_for_gate() -> Optional[Path]:
    """
    Resolve bundle_dir for strict schema gate.

    Priority:
    - GX1_CANONICAL_BUNDLE_DIR (TRUTH/SMOKE)
    - GX1_BUNDLE_DIR (worker/replay)
    """
    canonical = os.getenv("GX1_CANONICAL_BUNDLE_DIR") or ""
    if canonical:
        return Path(canonical)
    bundle_dir = os.getenv("GX1_BUNDLE_DIR") or ""
    if bundle_dir:
        return Path(bundle_dir)
    return None


def _load_lock_expected_features(bundle_dir: Path) -> Tuple[List[str], Dict[str, Any]]:
    """
    Load MASTER_MODEL_LOCK.json and extract the ordered feature list for XGB inputs.
    """
    lock_path = bundle_dir / "MASTER_MODEL_LOCK.json"
    if not lock_path.exists():
        raise RuntimeError(f"MASTER_MODEL_LOCK.json missing in bundle_dir: {bundle_dir}")
    obj = json.loads(lock_path.read_text(encoding="utf-8"))
    ordered = obj.get("ordered_features") or obj.get("feature_list") or obj.get("features_ordered") or None
    if not ordered or not isinstance(ordered, list):
        raise RuntimeError("MASTER_MODEL_LOCK.json missing ordered feature list (ordered_features/feature_list)")
    ordered_features = [str(x) for x in ordered]
    if not ordered_features or len(set(ordered_features)) != len(ordered_features):
        raise RuntimeError("MASTER_MODEL_LOCK ordered feature list invalid (empty or duplicates)")
    return ordered_features, obj


def _now_utc_compact() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _safe_mkdir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def _fallback_dir() -> Path:
    root = Path(tempfile.gettempdir()) / "gx1_prebuilt_schema_gate"
    if not _safe_mkdir(root):
        root = Path("/tmp")
    run_dir = root / f"run_{_now_utc_compact()}_{os.getpid()}"
    _safe_mkdir(run_dir)
    return run_dir


def _choose_report_dir(output_dir: Path) -> Path:
    try:
        if output_dir is not None and _safe_mkdir(output_dir):
            return output_dir
    except Exception:
        pass
    return _fallback_dir()


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _read_schema_columns(prebuilt_path: Path) -> Optional[List[str]]:
    if pq is None:
        return None
    try:
        parquet_file = pq.ParquetFile(prebuilt_path)
        return parquet_file.schema.names
    except Exception:
        return None


def run_prebuilt_feature_schema_gate_or_fatal(output_dir: Path, truth_or_smoke: bool, prebuilt_path: Path) -> None:
    if not truth_or_smoke:
        return

    report_dir = _choose_report_dir(output_dir)
    fatal_path = report_dir / "PREBUILT_SCHEMA_FATAL.json"

    if prebuilt_path is None:
        capsule = {
            "status": "FAIL",
            "message": "Prebuilt path is None (expected --prebuilt-parquet)",
            "prebuilt_path": None,
            "sys.executable": sys.executable,
            "sys.version": sys.version,
            "cwd": str(Path.cwd()),
            "report_dir": str(report_dir),
        }
        _write_json_atomic(fatal_path, capsule)
        raise SystemExit(2)

    if not prebuilt_path.exists():
        capsule = {
            "status": "FAIL",
            "message": f"Prebuilt parquet does not exist: {prebuilt_path}",
            "prebuilt_path": str(prebuilt_path),
            "sys.executable": sys.executable,
            "sys.version": sys.version,
            "cwd": str(Path.cwd()),
            "report_dir": str(report_dir),
        }
        _write_json_atomic(fatal_path, capsule)
        raise SystemExit(2)

    columns = _read_schema_columns(prebuilt_path)
    if not columns:
        capsule = {
            "status": "FAIL",
            "message": f"Failed to read parquet schema for prebuilt: {prebuilt_path}",
            "prebuilt_path": str(prebuilt_path),
            "sys.executable": sys.executable,
            "sys.version": sys.version,
            "cwd": str(Path.cwd()),
            "report_dir": str(report_dir),
        }
        _write_json_atomic(fatal_path, capsule)
        raise SystemExit(2)

    banned_features_found = sorted([c for c in columns if c.startswith("prob_")])
    if banned_features_found:
        capsule = {
            "status": "FAIL",
            "message": "Prebuilt schema contains forbidden prob_* columns (prebuilt must not contain model outputs).",
            "prebuilt_path": str(prebuilt_path),
            "banned_features_found": banned_features_found[:50],
            "columns_sample": sorted(columns)[:50],
            "sys.executable": sys.executable,
            "sys.version": sys.version,
            "cwd": str(Path.cwd()),
            "report_dir": str(report_dir),
        }
        _write_json_atomic(fatal_path, capsule)
        raise SystemExit(2)

    # Timestamp presence (schema-level): accept ts/time/timestamp/time-like.
    cols_lower = {c.lower() for c in columns}
    has_tsish = any(c in cols_lower for c in ("ts", "time", "timestamp", "__index_level_0__"))
    if not has_tsish:
        capsule = {
            "status": "FAIL",
            "message": "Prebuilt schema missing a recognizable timestamp column (expected ts/time/timestamp).",
            "prebuilt_path": str(prebuilt_path),
            "missing_required_features": [],
            "banned_features_found": [],
            "columns_sample": sorted(columns)[:50],
            "sys.executable": sys.executable,
            "sys.version": sys.version,
            "cwd": str(Path.cwd()),
            "report_dir": str(report_dir),
        }
        _write_json_atomic(fatal_path, capsule)
        raise SystemExit(2)

    # SIGNAL-ONLY ARCHITECTURE: strict gate on XGB input feature availability
    #
    # In TRUTH/SMOKE we validate the prebuilt parquet against the active bundle's MASTER_MODEL_LOCK
    # ordered feature list (order-sensitive). This replaces any v10_ctx-indicator minimal-set checks.
    bundle_dir = _resolve_bundle_dir_for_gate()
    if bundle_dir is None:
        capsule = {
            "status": "FAIL",
            "message": "Missing bundle dir for schema gate (expected GX1_CANONICAL_BUNDLE_DIR or GX1_BUNDLE_DIR in TRUTH/SMOKE).",
            "prebuilt_path": str(prebuilt_path),
            "env": {
                "GX1_CANONICAL_BUNDLE_DIR": os.getenv("GX1_CANONICAL_BUNDLE_DIR"),
                "GX1_BUNDLE_DIR": os.getenv("GX1_BUNDLE_DIR"),
                "GX1_RUN_MODE": os.getenv("GX1_RUN_MODE"),
                "GX1_TRUTH_MODE": os.getenv("GX1_TRUTH_MODE"),
            },
            "columns_sample": sorted(columns)[:50],
            "sys.executable": sys.executable,
            "sys.version": sys.version,
            "cwd": str(Path.cwd()),
            "report_dir": str(report_dir),
        }
        _write_json_atomic(fatal_path, capsule)
        raise SystemExit(2)

    try:
        expected_features, master_lock = _load_lock_expected_features(bundle_dir)
    except Exception as e:
        capsule = {
            "status": "FAIL",
            "message": f"Failed to load MASTER_MODEL_LOCK feature list for schema gate: {e}",
            "bundle_dir": str(bundle_dir),
            "prebuilt_path": str(prebuilt_path),
            "columns_sample": sorted(columns)[:50],
            "sys.executable": sys.executable,
            "sys.version": sys.version,
            "cwd": str(Path.cwd()),
            "report_dir": str(report_dir),
        }
        _write_json_atomic(fatal_path, capsule)
        raise SystemExit(2)

    # Allow timestamp-ish columns. Require prebuilt to start with lock ordered features (prefix match; allows ctx+2 extra columns).
    tsish_cols = {"ts", "time", "timestamp", "__index_level_0__"}
    allowed_non_feature = {c for c in columns if c.lower() in tsish_cols}
    expected_set = set(expected_features)

    feature_cols_ordered = [c for c in columns if c not in allowed_non_feature]
    prefix_match = len(feature_cols_ordered) >= len(expected_features) and feature_cols_ordered[: len(expected_features)] == expected_features
    missing = [c for c in expected_features if c not in set(columns)]
    extras = [c for c in feature_cols_ordered if c not in expected_set]

    if missing or not prefix_match:
        capsule = {
            "status": "FAIL",
            "message": "Prebuilt schema does not match MASTER_MODEL_LOCK ordered features (signal-only truth).",
            "bundle_dir": str(bundle_dir),
            "prebuilt_path": str(prebuilt_path),
            "expected_n_features": len(expected_features),
            "actual_n_columns": len(columns),
            "allowed_non_feature": sorted(list(allowed_non_feature)),
            "missing_features": missing[:50],
            "extra_columns": extras[:50],
            "order_match": prefix_match,
            "expected_features_head": expected_features[:25],
            "actual_features_head": feature_cols_ordered[:25],
            "master_lock_feature_contract_id": master_lock.get("feature_contract_id"),
            "master_lock_feature_list_sha256": master_lock.get("feature_list_sha256"),
            "sys.executable": sys.executable,
            "sys.version": sys.version,
            "cwd": str(Path.cwd()),
            "report_dir": str(report_dir),
        }
        _write_json_atomic(fatal_path, capsule)
        raise SystemExit(2)

