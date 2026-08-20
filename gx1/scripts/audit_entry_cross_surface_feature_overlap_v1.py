#!/usr/bin/env python3
"""Full-scan the actual Entry/Exit inputs for cross-surface duplicates.

This is deliberately a pre-dataset gate.  It reads the immutable M5 Entry and
M1 Exit feature surfaces in bounded Arrow batches, projects exactly the MTF
last-closed values each decision route consumes, and hashes every timestamped
float32 field sequence.  Equality is consequently a claim about all actual
decision rows, never a correlation or a sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gx1.contracts.entry_cross_surface_overlap_v1 import (
    DECISION_ROUTES,
    POLICY_VERSION,
    SCHEMA_VERSION,
    classify_active_duplicate_pairs,
    require_eight_family_coverage,
)
from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_DECISION_BAR_SECONDS,
    EXIT_DECISION_BAR_SECONDS,
    require_entry_exit_feature_surface_identity,
)
from gx1.contracts.entry_exit_feature_surface_v1 import (
    ENTRY_EXIT_FEATURE_SURFACE_COLUMNS,
    ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION,
    ENTRY_EXIT_M5_FEATURE_SURFACE_SCHEMA_VERSION,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_SIGNAL_DIM,
    require_model_native_manifest,
)
from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id
from gx1.features.htf_features import (
    MULTI_TF_PER_BAR_FEATURES_V4,
    MULTI_TF_SHIFT,
    load_multi_tf_v4_cache,
)
from gx1.utils.artifact_primitives_v1 import canonical_json_sha256, sha256_file


OUTPUT_PREFIX = "ENTRY_CROSS_SURFACE_INPUT_OVERLAP"
OUTPUT_RE = re.compile(rf"{OUTPUT_PREFIX}_\d{{8}}T\d{{6}}(?:\d{{6}})?Z\.json")
PRODUCER_SCHEMA_VERSION = "entry_cross_surface_feature_overlap_audit_v1"
DEFAULT_BATCH_SIZE = 4096
_HASH_DOMAIN = b"entry_cross_surface_timestamped_float32_values_v1\0"


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label}_MISSING_REGULAR_FILE: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label}_INVALID_JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label}_ROOT_NOT_OBJECT: {path}")
    return value


def _require_regular_absolute(path: Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise RuntimeError(f"{label}_PATH_NOT_ABSOLUTE: {candidate}")
    if candidate.is_symlink() or not candidate.is_file() or candidate.resolve() != candidate:
        raise RuntimeError(f"{label}_PATH_INVALID: {candidate}")
    return candidate


def _load_signal_contract(path: Path) -> tuple[list[str], str]:
    manifest_path = _require_regular_absolute(path, label="CROSS_SURFACE_SIGNAL_MANIFEST")
    manifest = _read_json(manifest_path, label="CROSS_SURFACE_SIGNAL_MANIFEST")
    contract = require_model_native_manifest(manifest, context="CROSS_SURFACE")
    fields = [str(name) for name in contract["fields"]]
    if len(fields) != MODEL_NATIVE_SIGNAL_DIM or len(fields) != len(set(fields)):
        raise RuntimeError("CROSS_SURFACE_SIGNAL_FIELDS_INVALID")
    return fields, sha256_file(manifest_path)


def _validate_surface(
    *,
    surface_path: Path,
    signal_manifest_path: Path,
    signal_manifest_sha256: str,
    signal_fields: list[str],
    entry_run_id: str,
    timeframe: str,
) -> dict[str, Any]:
    path = _require_regular_absolute(surface_path, label=f"CROSS_SURFACE_{timeframe}")
    sidecar = _require_regular_absolute(
        Path(f"{path}.manifest.json"), label=f"CROSS_SURFACE_{timeframe}_MANIFEST"
    )
    manifest = _read_json(sidecar, label=f"CROSS_SURFACE_{timeframe}_MANIFEST")
    surface_sha256 = sha256_file(path)
    manifest_sha256 = sha256_file(sidecar)
    payload = dict(manifest)
    declared_hash = payload.pop("manifest_sha256", None)
    expected_schema = (
        ENTRY_EXIT_M5_FEATURE_SURFACE_SCHEMA_VERSION
        if timeframe == "M5"
        else ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION
    )
    if (
        manifest.get("schema_version") != expected_schema
        or manifest.get("decision") != "PASS"
        or manifest.get("dataset_run_id") != entry_run_id
        or manifest.get("output_parquet") != str(path)
        or manifest.get("output_parquet_sha256") != surface_sha256
        or declared_hash != canonical_json_sha256(payload)
    ):
        raise RuntimeError(f"CROSS_SURFACE_{timeframe}_MANIFEST_CONTRACT_INVALID")
    require_entry_exit_feature_surface_identity(
        manifest,
        expected_timeframe=timeframe,
        expected_ordered_fields=signal_fields,
        expected_signal_manifest_path=str(signal_manifest_path),
        expected_signal_manifest_sha256=signal_manifest_sha256,
        context=f"CROSS_SURFACE_{timeframe}",
    )
    return {
        "path": str(path),
        "sha256": surface_sha256,
        "manifest_path": str(sidecar),
        "manifest_sha256": manifest_sha256,
        "pair_generation_id": str(manifest.get("pair_generation_id") or ""),
        "rows": int(manifest.get("rows") or 0),
        "source_parquet": str(manifest.get("source_parquet") or ""),
        "source_sha256": str(manifest.get("source_sha256") or ""),
    }


def _batch_matrix(batch: Any, *, name: str, width: int, dtype: np.dtype[Any]) -> np.ndarray:
    index = batch.schema.get_field_index(name)
    if index < 0:
        raise RuntimeError(f"CROSS_SURFACE_COLUMN_MISSING: {name}")
    column = batch.column(index)
    if not hasattr(column, "values") or column.null_count:
        raise RuntimeError(f"CROSS_SURFACE_COLUMN_INVALID: {name}")
    values = np.asarray(column.values.to_numpy(zero_copy_only=False), dtype=dtype)
    rows = int(batch.num_rows)
    if values.shape != (rows * width,):
        raise RuntimeError(
            f"CROSS_SURFACE_COLUMN_WIDTH_INVALID: {name}: "
            f"got={values.shape} expected={(rows * width,)}"
        )
    matrix = values.reshape(rows, width)
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"CROSS_SURFACE_COLUMN_NONFINITE: {name}")
    return np.ascontiguousarray(matrix, dtype=dtype)


def _batch_times_ns(batch: Any, *, previous: int | None, seconds: int) -> np.ndarray:
    index = batch.schema.get_field_index("time")
    if index < 0:
        raise RuntimeError("CROSS_SURFACE_TIME_MISSING")
    try:
        times = pd.DatetimeIndex(
            pd.to_datetime(batch.column(index).to_pandas(), utc=True, errors="coerce")
        ).as_unit("ns")
    except Exception as exc:
        raise RuntimeError("CROSS_SURFACE_TIME_INVALID") from exc
    values = np.asarray(times.asi8, dtype=np.int64)
    if (
        values.size == 0
        or np.any(values == np.iinfo(np.int64).min)
        or np.any(values % int(seconds * 1_000_000_000) != 0)
        or np.any(np.diff(values) <= 0)
        or (previous is not None and int(values[0]) <= previous)
    ):
        raise RuntimeError("CROSS_SURFACE_TIME_INVALID")
    return values


class _TimestampedColumnHashes:
    """Incremental exact hashes of all columns on one decision population."""

    def __init__(self, names: Iterable[str]) -> None:
        ordered = tuple(str(name) for name in names)
        self._hashers = {name: hashlib.sha256(_HASH_DOMAIN) for name in ordered}
        if not self._hashers or len(self._hashers) != len(ordered):
            raise RuntimeError("CROSS_SURFACE_HASH_FIELDS_INVALID")
        self.rows = 0

    def update(self, *, timestamps_ns: np.ndarray, values: np.ndarray, names: list[str]) -> None:
        timestamps = np.asarray(timestamps_ns, dtype="<i8")
        matrix = np.asarray(values, dtype="<f4")
        if (
            timestamps.ndim != 1
            or matrix.ndim != 2
            or matrix.shape != (timestamps.size, len(names))
            or tuple(names) != tuple(self._hashers)
            or not np.isfinite(matrix).all()
        ):
            raise RuntimeError("CROSS_SURFACE_HASH_UPDATE_INVALID")
        timestamp_bytes = np.ascontiguousarray(timestamps).tobytes()
        for index, name in enumerate(names):
            self._hashers[name].update(timestamp_bytes)
            self._hashers[name].update(np.ascontiguousarray(matrix[:, index]).tobytes())
        self.rows += int(timestamps.size)

    def result(self) -> dict[str, str]:
        if self.rows <= 0:
            raise RuntimeError("CROSS_SURFACE_HASH_EMPTY")
        return {name: hasher.hexdigest() for name, hasher in self._hashers.items()}


def _project_mtf_batch(
    *,
    cache: Mapping[str, pd.DataFrame],
    target_times_ns: np.ndarray,
    decision_seconds: int,
    timeframe: str,
) -> np.ndarray:
    frame = cache[timeframe]
    timestamps = np.asarray(frame.attrs.get("ts_int64"), dtype=np.int64)
    values = np.asarray(frame.attrs.get("feats_np"), dtype=np.float32)
    if timestamps.ndim != 1 or values.shape != (timestamps.size, len(MULTI_TF_PER_BAR_FEATURES_V4)):
        raise RuntimeError(f"CROSS_SURFACE_MTF_CACHE_ARRAY_INVALID: {timeframe}")
    cutoff = np.asarray(target_times_ns, dtype=np.int64) + int(
        decision_seconds * 1_000_000_000
    ) - int(MULTI_TF_SHIFT[timeframe].value)
    positions = np.searchsorted(timestamps, cutoff, side="right") - 1
    if np.any(positions < 0):
        raise RuntimeError(f"CROSS_SURFACE_MTF_HISTORY_INSUFFICIENT: {timeframe}")
    projected = np.ascontiguousarray(values[positions], dtype=np.float32)
    if not np.isfinite(projected).all():
        raise RuntimeError(f"CROSS_SURFACE_MTF_PROJECTED_NONFINITE: {timeframe}")
    return projected


def _hash_intersections(
    *, local: Mapping[str, str], mtf: Mapping[str, str]
) -> list[dict[str, str]]:
    local_by_hash: dict[str, list[str]] = {}
    mtf_by_hash: dict[str, list[str]] = {}
    for name, digest in local.items():
        local_by_hash.setdefault(str(digest), []).append(str(name))
    for name, digest in mtf.items():
        mtf_by_hash.setdefault(str(digest), []).append(str(name))
    return [
        {"local_field": lhs, "mtf_field": rhs, "values_sha256": digest}
        for digest in sorted(set(local_by_hash) & set(mtf_by_hash))
        for lhs in sorted(local_by_hash[digest])
        for rhs in sorted(mtf_by_hash[digest])
    ]


def _scan_decision_surface(
    *,
    decision: str,
    surface_path: Path,
    decision_seconds: int,
    signal_fields: list[str],
    cache: Mapping[str, pd.DataFrame],
    batch_size: int,
) -> dict[str, Any]:
    route = DECISION_ROUTES[decision]
    active_tfs = tuple(str(value) for value in route["active_mtf_timeframes"])
    local_names = [
        *(f"local.signal.{name}" for name in signal_fields),
        *(f"local.ctx_cont.{name}" for name in MODEL_NATIVE_CTX_CONT_FIELDS),
    ]
    local_hashes = _TimestampedColumnHashes(local_names)
    active_mtf_hashers = {
        timeframe: _TimestampedColumnHashes(
            f"mtf.{timeframe.lower()}.{name}" for name in MULTI_TF_PER_BAR_FEATURES_V4
        )
        for timeframe in active_tfs
    }
    inactive_m5_hasher = (
        _TimestampedColumnHashes(
            f"mtf.m5.{name}" for name in MULTI_TF_PER_BAR_FEATURES_V4
        )
        if decision == "entry"
        else None
    )
    try:
        parquet = pq.ParquetFile(surface_path)
    except Exception as exc:
        raise RuntimeError(f"CROSS_SURFACE_PARQUET_INVALID: {surface_path}") from exc
    if tuple(parquet.schema_arrow.names) != ENTRY_EXIT_FEATURE_SURFACE_COLUMNS:
        raise RuntimeError(f"CROSS_SURFACE_PARQUET_SCHEMA_INVALID: {surface_path}")
    previous_time: int | None = None
    first_time: int | None = None
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=["time", "signal", "ctx_cont"], use_threads=False
    ):
        times = _batch_times_ns(batch, previous=previous_time, seconds=decision_seconds)
        signal = _batch_matrix(
            batch, name="signal", width=len(signal_fields), dtype=np.dtype(np.float32)
        )
        context = _batch_matrix(
            batch,
            name="ctx_cont",
            width=len(MODEL_NATIVE_CTX_CONT_FIELDS),
            dtype=np.dtype(np.float32),
        )
        local_hashes.update(
            timestamps_ns=times,
            values=np.column_stack((signal, context)),
            names=local_names,
        )
        for timeframe, hasher in active_mtf_hashers.items():
            hasher.update(
                timestamps_ns=times,
                values=_project_mtf_batch(
                    cache=cache,
                    target_times_ns=times,
                    decision_seconds=decision_seconds,
                    timeframe=timeframe,
                ),
                names=list(hasher._hashers),
            )
        if inactive_m5_hasher is not None:
            inactive_m5_hasher.update(
                timestamps_ns=times,
                values=_project_mtf_batch(
                    cache=cache,
                    target_times_ns=times,
                    decision_seconds=decision_seconds,
                    timeframe="M5",
                ),
                names=list(inactive_m5_hasher._hashers),
            )
        previous_time = int(times[-1])
        if first_time is None:
            first_time = int(times[0])
    if first_time is None:
        raise RuntimeError(f"CROSS_SURFACE_NO_ROWS: {surface_path}")
    active_flat = {
        name: digest
        for hasher in active_mtf_hashers.values()
        for name, digest in hasher.result().items()
    }
    result = {
        "decision": decision,
        "local_timeframe": route["local_timeframe"],
        "active_mtf_timeframes": list(active_tfs),
        "row_count": local_hashes.rows,
        "first_time_ns": first_time,
        "last_time_ns": previous_time,
        "local_field_hashes": local_hashes.result(),
        "active_mtf_field_hashes": active_flat,
    }
    if inactive_m5_hasher is not None:
        result["inactive_entry_m5_field_hashes"] = inactive_m5_hasher.result()
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    entry_run_id = require_entry_run_id(args.run_id)
    out_path = Path(args.out_json).expanduser().resolve()
    if OUTPUT_RE.fullmatch(out_path.name) is None:
        raise RuntimeError(f"CROSS_SURFACE_OUTPUT_FILENAME_INVALID: {out_path.name}")
    if out_path.exists() or out_path.is_symlink() or out_path.parent.is_symlink():
        raise RuntimeError(f"CROSS_SURFACE_OUTPUT_NOT_FRESH: {out_path}")
    if not out_path.parent.is_dir():
        raise RuntimeError(f"CROSS_SURFACE_OUTPUT_PARENT_MISSING: {out_path.parent}")
    batch_size = int(args.batch_size)
    if not 1 <= batch_size <= 32_768:
        raise RuntimeError(f"CROSS_SURFACE_BATCH_SIZE_INVALID: {batch_size}")

    signal_manifest = _require_regular_absolute(
        Path(args.signal_manifest), label="CROSS_SURFACE_SIGNAL_MANIFEST"
    )
    signal_fields, signal_manifest_sha = _load_signal_contract(signal_manifest)
    m1_surface = _validate_surface(
        surface_path=Path(args.m1_feature_base_parquet),
        signal_manifest_path=signal_manifest,
        signal_manifest_sha256=signal_manifest_sha,
        signal_fields=signal_fields,
        entry_run_id=entry_run_id,
        timeframe="M1",
    )
    m5_surface = _validate_surface(
        surface_path=Path(args.m5_feature_base_parquet),
        signal_manifest_path=signal_manifest,
        signal_manifest_sha256=signal_manifest_sha,
        signal_fields=signal_fields,
        entry_run_id=entry_run_id,
        timeframe="M5",
    )
    if not m1_surface["pair_generation_id"] or m1_surface["pair_generation_id"] != m5_surface["pair_generation_id"]:
        raise RuntimeError("CROSS_SURFACE_PAIR_GENERATION_MISMATCH")
    cache_dir = Path(args.mtf_cache_dir).expanduser().resolve()
    if cache_dir.is_symlink() or not cache_dir.is_dir():
        raise RuntimeError(f"CROSS_SURFACE_MTF_CACHE_INVALID: {cache_dir}")
    cache = load_multi_tf_v4_cache(cache_dir)
    cache_binding = {
        "path": str(cache_dir),
        "manifest_sha256": str(cache.manifest_sha256),
        "cache_identity_sha256": str(cache.cache_identity_sha256),
        "m5_prebuilt_source": str(cache.m5_prebuilt_source),
        "m5_prebuilt_source_sha256": str(cache.m5_prebuilt_source_sha256),
    }
    if (
        cache_binding["m5_prebuilt_source"] != m5_surface["source_parquet"]
        or cache_binding["m5_prebuilt_source_sha256"] != m5_surface["source_sha256"]
    ):
        raise RuntimeError("CROSS_SURFACE_M5_CACHE_SOURCE_MISMATCH")
    local_fields = [
        *signal_fields,
        *(f"ctx_cont.{name}" for name in MODEL_NATIVE_CTX_CONT_FIELDS),
    ]
    family_coverage = require_eight_family_coverage(
        local_fields=local_fields, mtf_feature_names=MULTI_TF_PER_BAR_FEATURES_V4
    )
    entry = _scan_decision_surface(
        decision="entry",
        surface_path=Path(m5_surface["path"]),
        decision_seconds=ENTRY_DECISION_BAR_SECONDS,
        signal_fields=signal_fields,
        cache=cache,
        batch_size=batch_size,
    )
    exit_ = _scan_decision_surface(
        decision="exit",
        surface_path=Path(m1_surface["path"]),
        decision_seconds=EXIT_DECISION_BAR_SECONDS,
        signal_fields=signal_fields,
        cache=cache,
        batch_size=batch_size,
    )
    if entry["row_count"] != m5_surface["rows"] or exit_["row_count"] != m1_surface["rows"]:
        raise RuntimeError("CROSS_SURFACE_SURFACE_ROW_COUNT_MISMATCH")
    del cache
    entry_classification = classify_active_duplicate_pairs(
        decision="entry",
        local_field_hashes=entry["local_field_hashes"],
        active_mtf_field_hashes=entry["active_mtf_field_hashes"],
    )
    exit_classification = classify_active_duplicate_pairs(
        decision="exit",
        local_field_hashes=exit_["local_field_hashes"],
        active_mtf_field_hashes=exit_["active_mtf_field_hashes"],
    )
    inactive_entry_m5_pairs = _hash_intersections(
        local=entry["local_field_hashes"],
        mtf=entry["inactive_entry_m5_field_hashes"],
    )
    failures: list[dict[str, Any]] = []
    for decision, classification in (("entry", entry_classification), ("exit", exit_classification)):
        missing_declared = classification["missing_declared_context_mtf_alias_pairs"]
        if missing_declared:
            failures.append(
                {
                    "code": "declared_context_mtf_alias_not_exact",
                    "decision": decision,
                    "pairs": missing_declared,
                }
            )
        unexpected = classification["unexpected_active_exact_duplicate_pairs"]
        if unexpected:
            failures.append(
                {
                    "code": "unexpected_active_exact_cross_surface_duplicate",
                    "decision": decision,
                    "pairs": unexpected,
                }
            )
    report = {
        "schema_version": SCHEMA_VERSION,
        "producer_schema_version": PRODUCER_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "entry_run_id": entry_run_id,
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "policy": {
            "version": POLICY_VERSION,
            "comparison": "full_population_timestamped_float32_sha256",
            "active_unexpected_duplicate_action": "fail_closed_before_dataset_build",
            "declared_context_mtf_alias_action": "report_only_explicit_projection_contract",
            "inactive_entry_m5_action": "report_only_route_excluded_from_entry_mtf",
            "decision_routes": {
                decision: {
                    "local_timeframe": route["local_timeframe"],
                    "active_mtf_timeframes": list(route["active_mtf_timeframes"]),
                }
                for decision, route in DECISION_ROUTES.items()
            },
        },
        "input_bindings": {
            "signal_manifest": {"path": str(signal_manifest), "sha256": signal_manifest_sha},
            "m1_feature_surface": m1_surface,
            "m5_feature_surface": m5_surface,
            "mtf_cache": cache_binding,
        },
        "eight_family_coverage": family_coverage,
        "entry": {
            **entry,
            **entry_classification,
            "inactive_entry_m5_exact_duplicate_pairs": inactive_entry_m5_pairs,
        },
        "exit": {
            **exit_,
            **exit_classification,
        },
    }
    try:
        with out_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RuntimeError(f"CROSS_SURFACE_OUTPUT_NOT_FRESH: {out_path}") from exc
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--signal-manifest", required=True)
    parser.add_argument("--m1-feature-base-parquet", required=True)
    parser.add_argument("--m5-feature-base-parquet", required=True)
    parser.add_argument("--mtf-cache-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if report["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
