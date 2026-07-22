#!/usr/bin/env python3
"""Emit an immutable, report-only preflight for one seq513 dataset rebuild.

This preflight binds the exact inputs consumed by
``scripts/rebuild_entry_model_native_seq513_dataset.sh``.  It does not infer
paths, select mutable mirrors, inspect an older built dataset, or mix artifacts
from a different immutable run lineage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_state_v2 import (
    MODEL_NATIVE_HISTORY_MODE,
    MODEL_NATIVE_RANK_TRANSFORM,
    MODEL_NATIVE_STATE_SCHEMA_VERSION,
    MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
    validate_train_rank_reference_lineage_v2,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_SIGNAL_DIM,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_MANDATORY_FAMILY_FEATURES,
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
    model_native_mandatory_full_stack_metadata,
    require_model_native_manifest,
)
from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    group_features_by_specialist,
)
from gx1.features.htf_features import (
    HTF_V2_CACHE_BUILDER_VERSION,
    MULTI_TF_FEATURE_COUNT_V2,
    MULTI_TF_PER_BAR_FEATURES_V2,
    MULTI_TF_SHIFT,
)
from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    model_native_aux_target_contract_metadata,
)
from gx1.scripts.materialize_entry_model_native_seq513_signal_manifest_v1 import (
    validate_signal_manifest_training_lineage,
)
from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id


REPO = Path(__file__).resolve().parents[2]
REBUILD_WRAPPER = REPO / "scripts/rebuild_entry_model_native_seq513_dataset.sh"
EVENT_PREFIX = "ENTRY_MODEL_NATIVE_SEQ513_REBUILD_PREFLIGHT"
READY_DECISION = "READY_FOR_MODEL_NATIVE_SEQ513_REBUILD"
BLOCKED_DECISION = "BLOCKED_MODEL_NATIVE_SEQ513_REBUILD_PREFLIGHT"

EXPECTED_MTF_TFS = ("M5", "M15", "H1", "H4", "D1")
# ONE numeric owner: the loader (htf_features) enforces this exact version at
# consumption time; pinning a separate literal here made the two contracts
# mutually unsatisfiable when the cache builder was re-versioned 2026-07-17.
EXPECTED_MTF_BUILDER_VERSION = HTF_V2_CACHE_BUILDER_VERSION
FULL_INPUT_LIVENESS_OUTPUT_PATTERN = (
    "ENTRY_FULL_INPUT_LIVENESS_CONTRACT_<UTC_TIMESTAMP>.json"
)
EXACT_TAPE_COLUMNS = frozenset(
    {
        "time",
        "open",
        "high",
        "low",
        "close",
        "bid_close",
        "bid_high",
        "bid_low",
        "ask_close",
        "ask_high",
        "ask_low",
    }
)
RANK_SOURCE_COLUMNS = (
    "time",
    "high",
    "low",
    "close",
    "bid_close",
    "ask_close",
)
SOURCE_MARKET_COLUMNS = ("open",)
_TIMESTAMPED_JSON_RE = re.compile(r".+_\d{8}T\d{6}(?:\d{6})?Z\.json")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def _required_path_arg(args: argparse.Namespace, name: str, option: str) -> Path:
    raw = getattr(args, name, None)
    if raw is None or not str(raw).strip():
        raise RuntimeError(f"explicit {option} is required")
    return Path(str(raw)).expanduser().resolve()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_meta(path: Path, *, sha256: str | None = None) -> dict[str, Any]:
    exists = path.is_file() and not path.is_symlink()
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": sha256 if sha256 is not None else (_sha256_file(path) if exists else None),
    }


def _read_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file() or path.is_symlink():
        return {}, f"missing regular JSON file: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {}, f"invalid JSON {path}: {exc}"
    if not isinstance(payload, dict):
        return {}, f"JSON root must be an object: {path}"
    return payload, None


def _check(checks: list[dict[str, Any]], name: str, ok: bool, details: Any = None) -> None:
    checks.append({"name": name, "ok": bool(ok), "details": details})


def _parse_utc(raw: object, *, label: str) -> tuple[datetime | None, str | None]:
    text = str(raw or "").strip()
    if not text:
        return None, f"{label} is empty"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        return None, f"{label} is not ISO-8601: {exc}"
    if parsed.tzinfo is None:
        return None, f"{label} must declare UTC timezone"
    parsed = parsed.astimezone(timezone.utc)
    return parsed, None


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _split_contract(args: argparse.Namespace) -> tuple[dict[str, dict[str, str]], list[str]]:
    names = (
        "history_start",
        "train_start",
        "train_end",
        "val_start",
        "val_end",
        "test_start",
        "test_end",
    )
    values: dict[str, datetime] = {}
    failures: list[str] = []
    for name in names:
        value, error = _parse_utc(getattr(args, name, None), label=f"--{name.replace('_', '-')}")
        if error:
            failures.append(error)
        elif value is not None:
            values[name] = value
    if len(values) == len(names) and not (
        values["history_start"]
        < values["train_start"]
        <= values["train_end"]
        < values["val_start"]
        <= values["val_end"]
        < values["test_start"]
        <= values["test_end"]
    ):
        failures.append("split windows must be ordered and non-overlapping")
    if failures:
        return {}, failures
    return {
        "history": {
            "start": _iso_utc(values["history_start"]),
            "end": _iso_utc(values["test_end"]),
        },
        "train": {
            "start": _iso_utc(values["train_start"]),
            "end": _iso_utc(values["train_end"]),
        },
        "val": {
            "start": _iso_utc(values["val_start"]),
            "end": _iso_utc(values["val_end"]),
        },
        "test": {
            "start": _iso_utc(values["test_start"]),
            "end": _iso_utc(values["test_end"]),
        },
    }, []


def _parquet_columns(path: Path) -> tuple[list[str], str | None]:
    if not path.is_file() or path.is_symlink():
        return [], f"missing regular parquet file: {path}"
    try:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names), None
    except Exception as exc:  # pragma: no cover - exact backend exception is version-specific
        return [], f"cannot read parquet schema {path}: {exc}"


def _source_time_contract(
    path: Path,
    *,
    history_start: datetime | None,
    train_start: datetime | None,
    test_end: datetime | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "first_utc": None,
        "last_utc": None,
        "row_count": 0,
        "rows_from_history_start_before_train": 0,
        "strictly_increasing": False,
        "covers_history_start": False,
        "covers_test_end": False,
        "has_required_pre_train_history_rows": False,
        "error": None,
        "exact": False,
    }
    if not path.is_file() or path.is_symlink():
        result["error"] = "source parquet is missing, non-regular, or a symlink"
        return result
    try:
        import pyarrow.parquet as pq

        raw = pq.read_table(path, columns=["time"]).column("time").to_pandas()
        time = pd.DatetimeIndex(pd.to_datetime(raw, utc=True, errors="coerce"))
    except Exception as exc:  # pragma: no cover - backend details vary
        result["error"] = str(exc)
        return result
    result["row_count"] = int(len(time))
    if len(time) == 0 or time.hasnans:
        result["error"] = "time column is empty or contains invalid timestamps"
        return result
    strictly_increasing = bool(time.is_monotonic_increasing and time.is_unique)
    first = time[0].to_pydatetime()
    last = time[-1].to_pydatetime()
    result["first_utc"] = _iso_utc(first)
    result["last_utc"] = _iso_utc(last)
    result["strictly_increasing"] = strictly_increasing
    if history_start is not None:
        result["covers_history_start"] = first <= history_start <= last
    if test_end is not None:
        result["covers_test_end"] = first <= test_end <= last
    if history_start is not None and train_start is not None:
        history_rows = int(((time >= history_start) & (time < train_start)).sum())
        result["rows_from_history_start_before_train"] = history_rows
        result["has_required_pre_train_history_rows"] = history_rows >= MODEL_NATIVE_SEQ_LEN
    result["exact"] = bool(
        strictly_increasing
        and result["covers_history_start"]
        and result["covers_test_end"]
        and result["has_required_pre_train_history_rows"]
    )
    return result


def _manifest_timestamp_matches_created(path: Path, payload: dict[str, Any]) -> bool:
    match = re.search(r"(\d{8}T\d{6}(?:\d{6})?Z)\.json$", path.name)
    raw_created = payload.get("created_utc")
    if match is None or not isinstance(raw_created, str):
        return False
    stamp = match.group(1)
    # Parse the terminal ``Z`` as a literal. Passing it through strptime's
    # timezone handling made valid second-resolution names fail on this runtime.
    body = stamp[:-1]
    fmt = "%Y%m%dT%H%M%S%f" if len(stamp) == 22 else "%Y%m%dT%H%M%S"
    try:
        name_time = datetime.strptime(body, fmt).replace(tzinfo=timezone.utc)
        created = datetime.fromisoformat(raw_created.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return False
    if len(stamp) != 22:
        created = created.replace(microsecond=0)
    return name_time == created


def _manifest_specialist_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    selected = [str(name) for name in (manifest.get("selected_features") or [])]
    selected_set = set(selected)
    grouped = group_features_by_specialist(selected)
    required = tuple(MODEL_NATIVE_TRAINING_SPECIALISTS)
    observed = tuple(name for name in required if grouped.get(name))
    unmapped = list(grouped.get("unmapped") or [])
    forbidden = list(grouped.get("forbidden_legacy_bridge") or [])
    declared_groups = manifest.get("features_by_specialist")
    declared_group_match: bool | None = None
    if isinstance(declared_groups, dict):
        declared_group_match = all(
            list(declared_groups.get(name) or []) == list(grouped.get(name) or [])
            for name in required
        )
    declared_required = manifest.get("required_training_specialists")
    declared_required_match: bool | None = None
    if isinstance(declared_required, list):
        declared_required_match = tuple(str(name) for name in declared_required) == required
    mandatory_rows: list[dict[str, Any]] = []
    for family, expected_features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES:
        expected_set = set(expected_features)
        observed_features = [name for name in selected if name in expected_set]
        missing_features = [name for name in expected_features if name not in selected_set]
        mandatory_rows.append(
            {
                "family": family,
                "expected_feature_count": len(expected_features),
                "observed_feature_count": len(observed_features),
                "missing_features": missing_features,
                "exact": not missing_features
                and len(observed_features) == len(expected_features),
            }
        )
    missing_mandatory = [
        name for name in MODEL_NATIVE_MANDATORY_SELECTED_FIELDS if name not in selected_set
    ]
    ranked_remainder = [
        name for name in selected if name not in set(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
    ]
    declared_mandatory = manifest.get("mandatory_full_stack")
    canonical_mandatory = model_native_mandatory_full_stack_metadata()
    recomputed_family_counts = {
        str(row["family"]): int(row["observed_feature_count"])
        for row in mandatory_rows
    }
    expected_family_counts = {
        family: len(features)
        for family, features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
    }
    declared_family_counts = manifest.get("smart_layer_feature_counts")
    declared_source_counts = manifest.get("source_feature_counts")
    expected_source_counts = {
        "smart_candidate_layers": MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
        "mandatory_full_stack": MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
        "ranked_remainder": MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
    }
    return {
        "required_specialists": list(required),
        "observed_specialists": list(observed),
        "feature_counts": {name: len(grouped.get(name) or []) for name in required},
        "unmapped_features": unmapped,
        "forbidden_features": forbidden,
        "all_eight_covered": observed == required and not unmapped and not forbidden,
        "mandatory_family_rows": mandatory_rows,
        "mandatory_feature_count": MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
        "mandatory_missing_features": missing_mandatory,
        "ranked_remainder_feature_count": len(ranked_remainder),
        "mandatory_full_stack_exact": (
            not missing_mandatory
            and all(bool(row["exact"]) for row in mandatory_rows)
            and len(ranked_remainder) == MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT
        ),
        "declared_mandatory_full_stack_present": isinstance(
            declared_mandatory, dict
        ),
        "declared_mandatory_full_stack_match": (
            isinstance(declared_mandatory, dict)
            and declared_mandatory == canonical_mandatory
        ),
        "recomputed_mandatory_family_counts": recomputed_family_counts,
        "declared_mandatory_family_counts_present": isinstance(
            declared_family_counts, dict
        ),
        "declared_mandatory_family_counts_match": (
            isinstance(declared_family_counts, dict)
            and declared_family_counts == expected_family_counts
        ),
        "declared_source_feature_counts_present": isinstance(
            declared_source_counts, dict
        ),
        "declared_source_feature_counts_match": (
            isinstance(declared_source_counts, dict)
            and declared_source_counts == expected_source_counts
        ),
        "declared_groups_present": isinstance(declared_groups, dict),
        "declared_groups_match": declared_group_match,
        "declared_required_specialists_present": isinstance(declared_required, list),
        "declared_required_specialists_match": declared_required_match,
    }


def _source_manifest_rows(value: Any, *, prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(value, dict):
        return rows
    if "path" in value or "sha256" in value:
        raw_path = str(value.get("path") or "").strip()
        path = Path(raw_path).expanduser().resolve() if raw_path else Path("/")
        declared_sha = str(value.get("sha256") or "")
        observed_sha = _sha256_file(path) if raw_path and path.is_file() and not path.is_symlink() else None
        rows.append(
            {
                "label": prefix,
                "path": raw_path,
                "declared_sha256": declared_sha,
                "observed_sha256": observed_sha,
                "valid": bool(
                    raw_path
                    and len(declared_sha) == 64
                    and observed_sha == declared_sha
                    and not path.is_symlink()
                ),
            }
        )
        return rows
    for name, child in value.items():
        child_prefix = f"{prefix}.{name}" if prefix else str(name)
        rows.extend(_source_manifest_rows(child, prefix=child_prefix))
    return rows


def _mtf_cache_contract(
    cache_dir: Path,
    *,
    history_start: datetime | None,
    test_end: datetime | None,
) -> dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    manifest, read_error = _read_json(manifest_path)
    expected_feature_names = list(MULTI_TF_PER_BAR_FEATURES_V2)
    expected_shift = {tf: str(MULTI_TF_SHIFT[tf]) for tf in EXPECTED_MTF_TFS}
    observed_tfs = manifest.get("tfs") if isinstance(manifest.get("tfs"), dict) else {}
    rows: dict[str, dict[str, Any]] = {}
    file_names: set[str] = set()
    for tf in EXPECTED_MTF_TFS:
        info = observed_tfs.get(tf) if isinstance(observed_tfs.get(tf), dict) else {}
        feats_name = str(info.get("feats_npy") or "")
        ts_name = str(info.get("ts_npy") or "")
        feats_path = cache_dir / feats_name if feats_name else cache_dir / "<missing>"
        ts_path = cache_dir / ts_name if ts_name else cache_dir / "<missing>"
        if feats_name:
            file_names.add(feats_name)
        if ts_name:
            file_names.add(ts_name)
        errors: list[str] = []
        feats_shape: tuple[int, ...] = ()
        ts_shape: tuple[int, ...] = ()
        feats_dtype = ""
        ts_dtype = ""
        first_ts_ns: int | None = None
        last_ts_ns: int | None = None
        monotonic = False
        if feats_path.is_file() and ts_path.is_file() and not feats_path.is_symlink() and not ts_path.is_symlink():
            try:
                feats = np.load(feats_path, mmap_mode="r", allow_pickle=False)
                ts = np.load(ts_path, mmap_mode="r", allow_pickle=False)
                feats_shape = tuple(int(x) for x in feats.shape)
                ts_shape = tuple(int(x) for x in ts.shape)
                feats_dtype = str(feats.dtype)
                ts_dtype = str(ts.dtype)
                if ts.ndim == 1 and len(ts):
                    first_ts_ns = int(ts[0])
                    last_ts_ns = int(ts[-1])
                    monotonic = bool(len(ts) == 1 or np.all(ts[1:] > ts[:-1]))
            except Exception as exc:  # pragma: no cover - corrupt numpy variants differ
                errors.append(str(exc))
        else:
            errors.append("feature or timestamp npy is missing/non-regular")

        expected_first_max = None
        expected_last_min = None
        covers_train_history = False
        covers_test_end = False
        shift = MULTI_TF_SHIFT[tf]
        if history_start is not None:
            expected_first_max = int(
                (history_start - timedelta(seconds=shift.total_seconds() * MODEL_NATIVE_SEQ_LEN)).timestamp()
                * 1_000_000_000
            )
            covers_train_history = first_ts_ns is not None and first_ts_ns <= expected_first_max
        if test_end is not None:
            expected_last_min = int(
                (test_end - timedelta(seconds=shift.total_seconds())).timestamp()
                * 1_000_000_000
            )
            covers_test_end = last_ts_ns is not None and last_ts_ns >= expected_last_min

        n_bars = int(info.get("n_bars") or 0)
        exact = bool(
            not errors
            and feats_name == f"{tf}_feats.npy"
            and ts_name == f"{tf}_ts.npy"
            and len(feats_shape) == 2
            and feats_shape[0] == n_bars
            and feats_shape[1] == MULTI_TF_FEATURE_COUNT_V2
            and ts_shape == (n_bars,)
            and feats_dtype == "float32"
            and ts_dtype == "int64"
            and int(info.get("feature_count") or 0) == MULTI_TF_FEATURE_COUNT_V2
            and first_ts_ns == int(info.get("first_ts_ns") or -1)
            and last_ts_ns == int(info.get("last_ts_ns") or -1)
            and monotonic
            and covers_train_history
            and covers_test_end
        )
        rows[tf] = {
            "exact": exact,
            "n_bars": n_bars,
            "feature_shape": list(feats_shape),
            "timestamp_shape": list(ts_shape),
            "feature_dtype": feats_dtype,
            "timestamp_dtype": ts_dtype,
            "first_ts_ns": first_ts_ns,
            "last_ts_ns": last_ts_ns,
            "expected_first_ts_ns_at_most": expected_first_max,
            "expected_last_ts_ns_at_least": expected_last_min,
            "covers_train_history": covers_train_history,
            "covers_test_end": covers_test_end,
            "timestamps_strictly_increasing": monotonic,
            "feature_file": _artifact_meta(feats_path),
            "timestamp_file": _artifact_meta(ts_path),
            "errors": errors,
        }

    mtf_source_raw = str(manifest.get("m5_prebuilt_source") or "").strip()
    mtf_source = Path(mtf_source_raw).expanduser().resolve() if mtf_source_raw else Path("/")
    declared_source_sha = str(manifest.get("m5_prebuilt_source_sha256") or "")
    observed_source_sha = (
        _sha256_file(mtf_source)
        if mtf_source_raw and mtf_source.is_file() and not mtf_source.is_symlink()
        else None
    )
    expected_files = {
        *(f"{tf}_feats.npy" for tf in EXPECTED_MTF_TFS),
        *(f"{tf}_ts.npy" for tf in EXPECTED_MTF_TFS),
    }
    observed_npy = {path.name for path in cache_dir.glob("*.npy") if path.is_file()}
    return {
        "cache_dir": str(cache_dir),
        "manifest": _artifact_meta(manifest_path),
        "manifest_read_error": read_error,
        "feature_count": manifest.get("feature_count"),
        "feature_names": manifest.get("feature_names"),
        "shift_contract": manifest.get("shift_contract"),
        "builder_version": manifest.get("builder_version"),
        "tf_order": list(observed_tfs),
        "tf_rows": rows,
        "files_declared": sorted(file_names),
        "files_observed": sorted(observed_npy),
        "files_exact": file_names == expected_files and observed_npy == expected_files,
        "source": {
            "path": mtf_source_raw,
            "declared_sha256": declared_source_sha,
            "observed_sha256": observed_source_sha,
            "hash_matches": bool(
                len(declared_source_sha) == 64
                and observed_source_sha == declared_source_sha
                and not mtf_source.is_symlink()
            ),
        },
        "exact": bool(
            read_error is None
            and cache_dir.is_dir()
            and not cache_dir.is_symlink()
            and manifest.get("feature_count") == MULTI_TF_FEATURE_COUNT_V2
            and manifest.get("feature_names") == expected_feature_names
            and manifest.get("shift_contract") == expected_shift
            and manifest.get("builder_version") == EXPECTED_MTF_BUILDER_VERSION
            and set(observed_tfs) == set(EXPECTED_MTF_TFS)
            and file_names == expected_files
            and observed_npy == expected_files
            and all(row["exact"] for row in rows.values())
            and len(declared_source_sha) == 64
            and observed_source_sha == declared_source_sha
            and not mtf_source.is_symlink()
        ),
    }


def _tape_contract(tape_root: Path, *, start_year: int | None, end_year: int | None) -> dict[str, Any]:
    all_files = sorted(path for path in tape_root.rglob("*.parquet") if path.is_file()) if tape_root.is_dir() else []
    required_years = list(range(start_year, end_year + 1)) if start_year is not None and end_year is not None else []
    files_by_year: dict[str, list[str]] = {}
    schema_failures: list[dict[str, Any]] = []
    for year in required_years:
        year_dir = tape_root / f"year={year}"
        year_files = sorted(path for path in year_dir.glob("*.parquet") if path.is_file()) if year_dir.is_dir() else []
        files_by_year[str(year)] = [str(path) for path in year_files]
        for path in year_files:
            columns, error = _parquet_columns(path)
            colset = set(columns)
            missing_columns = sorted(EXACT_TAPE_COLUMNS - colset)
            if error or missing_columns:
                schema_failures.append(
                    {
                        "path": str(path),
                        "columns": columns,
                        "missing_exact_columns": missing_columns,
                        "error": error,
                    }
                )
    return {
        "root": str(tape_root),
        "required_years": required_years,
        "files_by_year": files_by_year,
        "total_parquet_files": len(all_files),
        "schema_failures": schema_failures,
        "exact": bool(
            tape_root.is_dir()
            and not tape_root.is_symlink()
            and required_years
            and all(files_by_year.get(str(year)) for year in required_years)
            and not schema_failures
        ),
    }


def _freshness_contract(
    *, rank_reference: Path, output: Path, audit_out_dir: Path
) -> dict[str, Any]:
    output_stem = output.stem
    derived = [
        output,
        output.parent / "DATASET_BUILD_PROOF.json",
        *(
            output.parent / f"{output_stem}_{split}{suffix}"
            for split in ("train", "val", "test")
            for suffix in (".parquet", ".manifest.json")
        ),
    ]
    existing = [str(path) for path in derived if path.exists() or path.is_symlink()]
    rank_sidecar = rank_reference.with_suffix(rank_reference.suffix + ".json")
    return {
        "rank_reference_npz": str(rank_reference),
        "rank_reference_sidecar": str(rank_sidecar),
        "rank_reference_suffix_valid": rank_reference.suffix == ".npz",
        "rank_reference_present": rank_reference.is_file()
        and not rank_reference.is_symlink(),
        "rank_reference_sidecar_present": rank_sidecar.is_file()
        and not rank_sidecar.is_symlink(),
        "output": str(output),
        "output_suffix_valid": output.name.endswith("__HOLD_03B.parquet"),
        "existing_output_artifacts": existing,
        "output_fresh": not existing,
        "audit_out_dir": str(audit_out_dir),
        "audit_out_dir_fresh": not audit_out_dir.exists() and not audit_out_dir.is_symlink(),
        "full_input_liveness_output": str(
            audit_out_dir / FULL_INPUT_LIVENESS_OUTPUT_PATTERN
        ),
        "full_input_liveness_output_fresh": not any(
            audit_out_dir.glob("ENTRY_FULL_INPUT_LIVENESS_CONTRACT_*.json")
        ),
    }


def _command_contract(
    *,
    entry_run_id: str,
    source_parquet: Path,
    canonical_v2_parquet: Path,
    signal_manifest: Path,
    feature_ranking_json: Path,
    rank_reference_npz: Path,
    mtf_cache_dir: Path,
    tape_root: Path,
    output: Path,
    audit_out_dir: Path,
    split_schedule: dict[str, dict[str, str]],
    model_native_signal_contract: dict[str, Any],
    source_parquet_sha256: str,
) -> dict[str, Any]:
    argv_template = [
        "scripts/rebuild_entry_model_native_seq513_dataset.sh",
        "--run-id",
        entry_run_id,
        "--source-parquet",
        str(source_parquet),
        "--canonical-v2-parquet",
        str(canonical_v2_parquet),
        "--signal-manifest",
        str(signal_manifest),
        "--feature-ranking-json",
        str(feature_ranking_json),
        "--rank-reference-npz",
        str(rank_reference_npz),
        "--existing-rank-reference",
        "--mtf-cache-dir",
        str(mtf_cache_dir),
        "--tape-root",
        str(tape_root),
        "--output",
        str(output),
        "--audit-out-dir",
        str(audit_out_dir),
        "--history-start",
        split_schedule["history"]["start"],
        "--train-start",
        split_schedule["train"]["start"],
        "--train-end",
        split_schedule["train"]["end"],
        "--val-start",
        split_schedule["val"]["start"],
        "--val-end",
        split_schedule["val"]["end"],
        "--test-start",
        split_schedule["test"]["start"],
        "--test-end",
        split_schedule["test"]["end"],
    ]
    return {
        "wrapper": _artifact_meta(REBUILD_WRAPPER),
        "argv_template": argv_template,
        "run_lineage_required": True,
        "entry_run_id": entry_run_id,
        "run_id_validated": True,
        "starts_dataset_rebuild": True,
        "starts_training": False,
        "starts_replay": False,
        "touches_shadow_or_live": False,
        "mandatory_post_build_gates": [
            {
                "producer": "gx1.scripts.materialize_entry_full_input_liveness_v1",
                "validator": (
                    "gx1.contracts.entry_full_input_liveness_v1."
                    "validate_full_input_liveness_artifact"
                ),
                "output_pattern": str(
                    audit_out_dir / FULL_INPUT_LIVENESS_OUTPUT_PATTERN
                ),
                "fullscan": True,
                "fail_closed": True,
            },
            {
                "producer": "gx1.scripts.audit_xau_direction_repair_pretrain_v1",
                "fail_closed": True,
            },
        ],
        "model_native_signal_contract": model_native_signal_contract,
        "split_schedule": split_schedule,
        "rank_reference_contract": {
            "producer": "gx1.scripts.materialize_model_native_train_rank_reference_v2",
            "schema_version": MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
            "source_parquet": str(source_parquet),
            "source_parquet_sha256": source_parquet_sha256,
            "output_npz": str(rank_reference_npz),
            "sidecar_json": str(
                rank_reference_npz.with_suffix(rank_reference_npz.suffix + ".json")
            ),
            "materialized_before_dataset_builder": True,
            "materialized_before_feature_ranker": True,
            "feature_history_start_utc": split_schedule["history"]["start"],
            "fit_start_utc": split_schedule["train"]["start"],
            "fit_end_utc": split_schedule["train"]["end"],
            "fit_scope": "train_only",
            "rank_transform": MODEL_NATIVE_RANK_TRANSFORM,
            "contains_validation_or_test_rows": False,
            "contains_per_row_state": False,
            "sidecar_source_sha256_must_match": True,
            "sidecar_npz_sha256_required": True,
            "builder_must_verify_npz_and_sidecar": True,
            "run_lineage_required": True,
            "run_id_bound_in_npz_and_sidecar": True,
            "dataset_builder_requires_same_run_id": True,
            "preflight_validates_exact_existing_reference": True,
        },
        "fixed_builder_contract": {
            "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
            "base_signal_dim": MODEL_NATIVE_BASE_SIGNAL_DIM,
            "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
            "seq_len": MODEL_NATIVE_SEQ_LEN,
            "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
            "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
            "hold_bars": 3,
            "direction_target_mode": "path_utility_v2",
            "aux_head_target_contract": model_native_aux_target_contract_metadata(),
            "inline_selected_features": True,
            "rank_reference_required": True,
            "run_lineage_required": True,
            "rank_reference_run_id_match_required": True,
            "closed_bar_multi_tf_required": True,
            "state_schema_version": MODEL_NATIVE_STATE_SCHEMA_VERSION,
            "feature_history_mode": MODEL_NATIVE_HISTORY_MODE,
            "split_reset_allowed": False,
            "rank_fit_scope": "train_only",
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    entry_run_id = require_entry_run_id(getattr(args, "run_id", None))
    source_parquet = _required_path_arg(args, "source_parquet", "--source-parquet")
    canonical_v2_parquet = _required_path_arg(
        args, "canonical_v2_parquet", "--canonical-v2-parquet"
    )
    signal_manifest_path = _required_path_arg(
        args, "signal_manifest", "--signal-manifest"
    )
    feature_ranking_path = _required_path_arg(
        args, "feature_ranking_json", "--feature-ranking-json"
    )
    rank_reference_npz = _required_path_arg(
        args, "rank_reference_npz", "--rank-reference-npz"
    )
    mtf_cache_dir = _required_path_arg(args, "mtf_cache_dir", "--mtf-cache-dir")
    tape_root = _required_path_arg(args, "tape_root", "--tape-root")
    output = _required_path_arg(args, "output", "--output")
    audit_out_dir = _required_path_arg(args, "audit_out_dir", "--audit-out-dir")
    out_dir = _required_path_arg(args, "out_dir", "--out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, Any]] = []
    source_sha = _sha256_file(source_parquet)
    canonical_sha = _sha256_file(canonical_v2_parquet)
    source_columns, source_schema_error = _parquet_columns(source_parquet)
    canonical_columns, canonical_schema_error = _parquet_columns(canonical_v2_parquet)
    _check(
        checks,
        "explicit source parquet is a regular readable parquet",
        source_schema_error is None and not source_parquet.is_symlink(),
        {"path": str(source_parquet), "error": source_schema_error},
    )
    _check(
        checks,
        "source parquet carries market and raw TRAIN-rank source columns",
        set(SOURCE_MARKET_COLUMNS + RANK_SOURCE_COLUMNS).issubset(source_columns),
        {"required": list(SOURCE_MARKET_COLUMNS + RANK_SOURCE_COLUMNS), "columns": source_columns},
    )
    _check(
        checks,
        "explicit canonical-v2 parquet is a regular readable parquet",
        canonical_schema_error is None and not canonical_v2_parquet.is_symlink(),
        {"path": str(canonical_v2_parquet), "error": canonical_schema_error},
    )
    _check(
        checks,
        "canonical-v2 parquet has an explicit time key",
        "time" in canonical_columns,
        canonical_columns,
    )
    _check(checks, "source parquet hash is bound", source_sha is not None, source_sha)
    _check(checks, "canonical-v2 parquet hash is bound", canonical_sha is not None, canonical_sha)

    manifest, manifest_read_error = _read_json(signal_manifest_path)
    _check(
        checks,
        "signal manifest is an explicit timestamped immutable input",
        bool(
            manifest_read_error is None
            and not signal_manifest_path.is_symlink()
            and _TIMESTAMPED_JSON_RE.fullmatch(signal_manifest_path.name)
            and "_latest" not in signal_manifest_path.name
        ),
        {"path": str(signal_manifest_path), "error": manifest_read_error},
    )
    _check(
        checks,
        "signal manifest filename timestamp matches created_utc",
        _manifest_timestamp_matches_created(signal_manifest_path, manifest),
        manifest.get("created_utc"),
    )
    declared_path = str(manifest.get("json_path") or "").strip()
    _check(
        checks,
        "signal manifest declares its exact immutable self-path",
        bool(declared_path)
        and Path(declared_path).expanduser().resolve() == signal_manifest_path,
        declared_path,
    )
    try:
        signal_contract = require_model_native_manifest(
            manifest, context="MODEL_NATIVE_SEQ513_REBUILD_PREFLIGHT"
        )
        signal_contract_failures: list[str] = []
    except (RuntimeError, TypeError, ValueError) as exc:
        signal_contract = {}
        signal_contract_failures = [str(exc)]
    _check(
        checks,
        "signal manifest proves exact ordered 34+479=513 and 142/5 intent",
        not signal_contract_failures,
        signal_contract_failures,
    )

    specialist = _manifest_specialist_contract(manifest)
    _check(
        checks,
        "all 479 selected features map across the exact eight specialists",
        specialist["all_eight_covered"],
        specialist,
    )
    _check(
        checks,
        "all code-owned full-stack family fields are retained exactly",
        specialist["mandatory_full_stack_exact"],
        specialist,
    )
    _check(
        checks,
        "declared mandatory full-stack metadata matches recomputed registry",
        specialist["declared_mandatory_full_stack_match"] is True,
        specialist,
    )
    _check(
        checks,
        "declared mandatory family counts match recomputed selected names",
        specialist["declared_mandatory_family_counts_match"] is True,
        specialist,
    )
    _check(
        checks,
        "declared source feature counts match mandatory and ranked partitions",
        specialist["declared_source_feature_counts_match"] is True,
        specialist,
    )
    if specialist["declared_groups_present"]:
        _check(
            checks,
            "declared specialist feature groups match derived routing",
            specialist["declared_groups_match"] is True,
            specialist,
        )
    if specialist["declared_required_specialists_present"]:
        _check(
            checks,
            "declared required specialist set/order is exact",
            specialist["declared_required_specialists_match"] is True,
            specialist,
        )
    source_manifest_rows = _source_manifest_rows(manifest.get("source_manifests"))
    if source_manifest_rows:
        _check(
            checks,
            "declared signal source manifests are present and hash-bound",
            all(row["valid"] for row in source_manifest_rows),
            source_manifest_rows,
        )

    split_schedule, split_failures = _split_contract(args)
    _check(
        checks,
        "explicit train/val/test split windows are ordered and non-overlapping",
        not split_failures,
        {"schedule": split_schedule, "failures": split_failures},
    )
    train_start, _ = _parse_utc(
        split_schedule.get("train", {}).get("start"), label="train start"
    )
    history_start, _ = _parse_utc(
        split_schedule.get("history", {}).get("start"), label="history start"
    )
    test_end, _ = _parse_utc(
        split_schedule.get("test", {}).get("end"), label="test end"
    )

    signal_lineage: dict[str, Any] = {}
    signal_lineage_failures: list[str] = []
    if split_schedule and source_sha:
        try:
            signal_lineage = validate_signal_manifest_training_lineage(
                manifest_path=signal_manifest_path,
                feature_ranking_path=feature_ranking_path,
                expected_run_id=entry_run_id,
                expected_source_sha256=source_sha,
                expected_train_start_utc=split_schedule["train"]["start"],
                expected_train_end_utc=split_schedule["train"]["end"],
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            signal_lineage_failures.append(str(exc))
    else:
        signal_lineage_failures.append(
            "source hash and valid split schedule are required before lineage validation"
        )
    _check(
        checks,
        "signal manifest binds the explicit ranking, run_id, source hash, and exact TRAIN window",
        not signal_lineage_failures,
        {"lineage": signal_lineage, "failures": signal_lineage_failures},
    )

    source_time = _source_time_contract(
        source_parquet,
        history_start=history_start,
        train_start=train_start,
        test_end=test_end,
    )
    _check(
        checks,
        "source has ordered common history, >=96 pre-TRAIN rows, and TEST coverage",
        source_time["exact"],
        source_time,
    )

    mtf = _mtf_cache_contract(
        mtf_cache_dir, history_start=history_start, test_end=test_end
    )
    _check(
        checks,
        "explicit MTF cache has exact five-TF files/schema/source hash/coverage",
        mtf["exact"],
        mtf,
    )
    tape = _tape_contract(
        tape_root,
        start_year=train_start.year if train_start is not None else None,
        end_year=test_end.year if test_end is not None else None,
    )
    _check(
        checks,
        "explicit tape root has exact spread-aware OHLC coverage for every split year",
        tape["exact"],
        tape,
    )

    freshness = _freshness_contract(
        rank_reference=rank_reference_npz,
        output=output,
        audit_out_dir=audit_out_dir,
    )
    _check(
        checks,
        "rank-reference NPZ and sidecar are explicit existing immutable inputs",
        freshness["rank_reference_suffix_valid"]
        and freshness["rank_reference_present"]
        and freshness["rank_reference_sidecar_present"],
        freshness,
    )
    rank_reference_lineage: dict[str, Any] = {}
    rank_reference_failures: list[str] = []
    if split_schedule and source_sha:
        try:
            reference = validate_train_rank_reference_lineage_v2(
                rank_reference_npz,
                expected_run_id=entry_run_id,
                expected_source_parquet=source_parquet,
                expected_source_sha256=source_sha,
                expected_history_start_utc=split_schedule["history"]["start"],
                expected_fit_start_utc=split_schedule["train"]["start"],
                expected_fit_end_utc=split_schedule["train"]["end"],
            )
            rank_reference_lineage = {
                "path": str(reference.path),
                "sha256": reference.sha256,
                "sidecar_sha256": reference.sidecar_sha256,
                "schema_version": MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
                "fit_start_utc": reference.fit_start_utc.isoformat(),
                "fit_end_utc": reference.fit_end_utc.isoformat(),
                "fit_row_count": reference.fit_row_count,
            }
        except (RuntimeError, TypeError, ValueError) as exc:
            rank_reference_failures.append(str(exc))
    else:
        rank_reference_failures.append(
            "source hash and valid split schedule are required before rank-reference validation"
        )
    _check(
        checks,
        "rank-reference binds the exact run_id, source hash, history, and TRAIN window",
        not rank_reference_failures,
        {
            "lineage": rank_reference_lineage,
            "failures": rank_reference_failures,
        },
    )
    _check(
        checks,
        "dataset output path and derived split artifacts are fresh",
        freshness["output_suffix_valid"] and freshness["output_fresh"],
        freshness,
    )
    _check(
        checks,
        "audit output directory is fresh",
        freshness["audit_out_dir_fresh"],
        freshness,
    )
    _check(
        checks,
        "rebuild wrapper is a regular executable control artifact",
        REBUILD_WRAPPER.is_file()
        and not REBUILD_WRAPPER.is_symlink()
        and bool(REBUILD_WRAPPER.stat().st_mode & 0o111),
        str(REBUILD_WRAPPER),
    )

    command_contract: dict[str, Any] = {}
    if split_schedule:
        command_contract = _command_contract(
            entry_run_id=entry_run_id,
            source_parquet=source_parquet,
            canonical_v2_parquet=canonical_v2_parquet,
            signal_manifest=signal_manifest_path,
            feature_ranking_json=feature_ranking_path,
            rank_reference_npz=rank_reference_npz,
            mtf_cache_dir=mtf_cache_dir,
            tape_root=tape_root,
            output=output,
            audit_out_dir=audit_out_dir,
            split_schedule=split_schedule,
            model_native_signal_contract=signal_contract,
            source_parquet_sha256=str(source_sha or ""),
        )

    failures = [row for row in checks if not row["ok"]]
    created_utc = datetime.now(timezone.utc)
    report = {
        "schema_version": "entry_model_native_seq513_rebuild_preflight_v5",
        "created_utc": created_utc.isoformat(),
        "decision": READY_DECISION if not failures else BLOCKED_DECISION,
        "report_only": True,
        "entry_run_id": entry_run_id,
        "training_allowed": False,
        "dataset_rebuild_allowed": not failures,
        "side_effects_started": {
            "dataset_rebuild": False,
            "training": False,
            "replay": False,
            "shadow": False,
            "live": False,
        },
        "counts": {
            "base_signal_features": MODEL_NATIVE_BASE_SIGNAL_DIM,
            "selected_features": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
            "expected_seq_snap_width": MODEL_NATIVE_SIGNAL_DIM,
            "seq_len": MODEL_NATIVE_SEQ_LEN,
            "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
            "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
            "required_specialist_count": len(MODEL_NATIVE_TRAINING_SPECIALISTS),
            "manifest_variant": manifest.get("manifest_variant"),
        },
        "required_model_native_contract": {
            "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
            "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
            "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
            "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
            "seq_len": MODEL_NATIVE_SEQ_LEN,
            "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
            "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
            "base_signal_dim": MODEL_NATIVE_BASE_SIGNAL_DIM,
            "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
            "bridge_dim": 0,
            "bridge_source": None,
            "anchor_source": None,
        },
        "inputs": {
            "source_parquet": _artifact_meta(source_parquet, sha256=source_sha),
            "canonical_v2_parquet": _artifact_meta(
                canonical_v2_parquet, sha256=canonical_sha
            ),
            "signal_manifest": _artifact_meta(signal_manifest_path),
            "feature_ranking_json": _artifact_meta(feature_ranking_path),
            "rank_and_output_freshness": freshness,
            "rank_reference_lineage": rank_reference_lineage,
            "source_time_contract": source_time,
            "multi_tf_cache": mtf,
            "tape": tape,
        },
        "specialist_contract": specialist,
        "signal_source_manifest_rows": source_manifest_rows,
        "signal_training_lineage": signal_lineage,
        "rebuild_command_contract": command_contract,
        "checks": checks,
        "failures": failures,
        "next_required_gate": (
            "review and execute only the exact run_id-bound capped rebuild command, then prove "
            "feature/target/specialist/liveness "
            "contracts before any training review"
        ),
    }
    strict_report = json.loads(
        json.dumps(report, sort_keys=True, allow_nan=False, default=_json_default)
    )
    _, published = write_immutable_json_event(out_dir, EVENT_PREFIX, strict_report)
    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report-only exact model-native seq513 rebuild preflight."
    )
    parser.add_argument("--source-parquet", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--canonical-v2-parquet", required=True)
    parser.add_argument("--signal-manifest", required=True)
    parser.add_argument("--feature-ranking-json", required=True)
    parser.add_argument("--rank-reference-npz", required=True)
    parser.add_argument("--mtf-cache-dir", required=True)
    parser.add_argument("--tape-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-out-dir", required=True)
    parser.add_argument("--history-start", required=True)
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--val-start", required=True)
    parser.add_argument("--val-end", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    if report["decision"] != READY_DECISION:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
