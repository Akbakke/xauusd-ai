#!/usr/bin/env python3
"""Full-scan the actual native M5 and M1 feature surfaces for liveness.

The split-dataset liveness gate proves what reached the Entry training tensor.
This companion audit closes the other half of the unified architecture: every
locally computed feature on the immutable native M5 Entry *and* M1 Exit
surfaces must be finite and vary on the exact post-history population.  It
does not substitute a summary statistic for a scan, nor does it permit a
family to be declared present merely because one of its columns is live.

The existing cross-surface overlap artifact is required as an immutable input.
That binds this liveness proof to the same causal decision population and to
the prior full-population MTF duplicate/alias proof.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gx1.contracts.entry_cross_surface_overlap_v1 import (
    validate_cross_surface_overlap_report,
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
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_SIGNAL_DIM,
    require_model_native_manifest,
)
from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    classify_entry_specialist_feature,
)
from gx1.utils.artifact_primitives_v1 import canonical_json_sha256, sha256_file


OUTPUT_PREFIX = "ENTRY_FEATURE_SURFACE_LIVENESS"
OUTPUT_RE = re.compile(rf"{OUTPUT_PREFIX}_\d{{8}}T\d{{6}}(?:\d{{6}})?Z\.json")
SCHEMA_VERSION = "entry_feature_surface_liveness_v1"
POLICY_VERSION = "entry_feature_surface_liveness_policy_v1"
PRODUCER_SCHEMA_VERSION = "entry_feature_surface_liveness_audit_v1"
DEFAULT_BATCH_SIZE = 4096
LIVENESS_EPSILON = 1e-7
NEAR_CONSTANT_STD = 1e-9
REQUIRED_COLUMNS = ("time", "signal", "ctx_cont", "ctx_cat")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise RuntimeError(f"{label}_MISSING_REGULAR_FILE: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label}_INVALID_JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label}_ROOT_NOT_OBJECT: {path}")
    return payload


def _require_regular_absolute(path: Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or not candidate.is_file()
        or candidate.resolve() != candidate
    ):
        raise RuntimeError(f"{label}_PATH_INVALID: {candidate}")
    return candidate


def _load_signal_contract(path: Path) -> tuple[list[str], str, int]:
    manifest_path = _require_regular_absolute(path, label="SURFACE_LIVENESS_SIGNAL_MANIFEST")
    manifest = _read_json(manifest_path, label="SURFACE_LIVENESS_SIGNAL_MANIFEST")
    contract = require_model_native_manifest(manifest, context="SURFACE_LIVENESS")
    fields = [str(name) for name in contract["fields"]]
    if len(fields) != MODEL_NATIVE_SIGNAL_DIM or len(fields) != len(set(fields)):
        raise RuntimeError("SURFACE_LIVENESS_SIGNAL_FIELDS_INVALID")
    source_cascade = (
        manifest.get("feature_ranking", {}).get("source_cascade")
        if isinstance(manifest.get("feature_ranking"), Mapping)
        else None
    )
    history_start = (
        source_cascade.get("history_start_utc")
        if isinstance(source_cascade, Mapping)
        else None
    )
    if not isinstance(history_start, str):
        raise RuntimeError("SURFACE_LIVENESS_HISTORY_START_MISSING")
    try:
        timestamp = pd.Timestamp(history_start)
    except Exception as exc:
        raise RuntimeError("SURFACE_LIVENESS_HISTORY_START_INVALID") from exc
    if timestamp.tzinfo is None:
        raise RuntimeError("SURFACE_LIVENESS_HISTORY_START_INVALID")
    return fields, sha256_file(manifest_path), int(timestamp.tz_convert("UTC").value)


def _validate_surface(
    *,
    surface_path: Path,
    signal_manifest_path: Path,
    signal_manifest_sha256: str,
    signal_fields: list[str],
    entry_run_id: str,
    timeframe: str,
) -> dict[str, Any]:
    path = _require_regular_absolute(surface_path, label=f"SURFACE_LIVENESS_{timeframe}")
    sidecar = _require_regular_absolute(
        Path(f"{path}.manifest.json"), label=f"SURFACE_LIVENESS_{timeframe}_MANIFEST"
    )
    manifest = _read_json(sidecar, label=f"SURFACE_LIVENESS_{timeframe}_MANIFEST")
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
        raise RuntimeError(f"SURFACE_LIVENESS_{timeframe}_MANIFEST_CONTRACT_INVALID")
    require_entry_exit_feature_surface_identity(
        manifest,
        expected_timeframe=timeframe,
        expected_ordered_fields=signal_fields,
        expected_signal_manifest_path=str(signal_manifest_path),
        expected_signal_manifest_sha256=signal_manifest_sha256,
        context=f"SURFACE_LIVENESS_{timeframe}",
    )
    return {
        "path": str(path),
        "sha256": surface_sha256,
        "manifest_path": str(sidecar),
        "manifest_sha256": manifest_sha256,
        "rows": int(manifest.get("rows") or 0),
        "timeframe": timeframe,
    }


def _batch_matrix(
    batch: Any, *, name: str, width: int, dtype: np.dtype[Any]
) -> np.ndarray:
    index = batch.schema.get_field_index(name)
    if index < 0:
        raise RuntimeError(f"SURFACE_LIVENESS_COLUMN_MISSING: {name}")
    column = batch.column(index)
    if not hasattr(column, "values") or column.null_count:
        raise RuntimeError(f"SURFACE_LIVENESS_COLUMN_INVALID: {name}")
    values = np.asarray(column.values.to_numpy(zero_copy_only=False), dtype=dtype)
    rows = int(batch.num_rows)
    if values.shape != (rows * width,):
        raise RuntimeError(
            f"SURFACE_LIVENESS_COLUMN_WIDTH_INVALID: {name}: "
            f"got={values.shape} expected={(rows * width,)}"
        )
    return np.ascontiguousarray(values.reshape(rows, width), dtype=dtype)


def _batch_times_ns(
    batch: Any, *, previous: int | None, seconds: int
) -> np.ndarray:
    index = batch.schema.get_field_index("time")
    if index < 0:
        raise RuntimeError("SURFACE_LIVENESS_TIME_MISSING")
    try:
        times = pd.DatetimeIndex(
            pd.to_datetime(batch.column(index).to_pandas(), utc=True, errors="coerce")
        ).as_unit("ns")
    except Exception as exc:
        raise RuntimeError("SURFACE_LIVENESS_TIME_INVALID") from exc
    values = np.asarray(times.asi8, dtype=np.int64)
    if (
        values.size == 0
        or np.any(values == np.iinfo(np.int64).min)
        or np.any(values % int(seconds * 1_000_000_000) != 0)
        or np.any(np.diff(values) <= 0)
        or (previous is not None and int(values[0]) <= previous)
    ):
        raise RuntimeError("SURFACE_LIVENESS_TIME_INVALID")
    return values


class _StreamingFieldStats:
    """Bounded, numerically stable full-population statistics per field."""

    def __init__(self, width: int, *, categorical: bool = False) -> None:
        self.width = int(width)
        self.categorical = bool(categorical)
        self.row_count = 0
        self.finite_count = np.zeros(self.width, dtype=np.int64)
        self.nonfinite_count = np.zeros(self.width, dtype=np.int64)
        self.mean = np.zeros(self.width, dtype=np.float64)
        self.m2 = np.zeros(self.width, dtype=np.float64)
        self.minimum = np.full(self.width, np.inf, dtype=np.float64)
        self.maximum = np.full(self.width, -np.inf, dtype=np.float64)
        self.active_count = np.zeros(self.width, dtype=np.int64)
        self.integer_like_count = np.zeros(self.width, dtype=np.int64)
        self.unique_values = [set() for _ in range(self.width)] if categorical else []

    def update(self, values: np.ndarray) -> None:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.width:
            raise RuntimeError("SURFACE_LIVENESS_STATS_SHAPE_INVALID")
        rows = int(matrix.shape[0])
        if rows <= 0:
            raise RuntimeError("SURFACE_LIVENESS_STATS_EMPTY_BATCH")
        self.row_count += rows
        finite = np.isfinite(matrix)
        batch_count = finite.sum(axis=0, dtype=np.int64)
        previous_count = self.finite_count.copy()
        self.finite_count += batch_count
        self.nonfinite_count += rows - batch_count
        self.active_count += (
            finite & (np.abs(matrix) > float(LIVENESS_EPSILON))
        ).sum(axis=0, dtype=np.int64)
        safe = np.where(finite, matrix, 0.0)
        batch_sum = safe.sum(axis=0, dtype=np.float64)
        batch_mean = np.divide(
            batch_sum, batch_count, out=np.zeros(self.width), where=batch_count > 0
        )
        centered = np.where(finite, matrix - batch_mean, 0.0)
        batch_m2 = np.square(centered).sum(axis=0, dtype=np.float64)
        combined_count = previous_count + batch_count
        delta = batch_mean - self.mean
        valid = batch_count > 0
        self.mean[valid] += delta[valid] * batch_count[valid] / combined_count[valid]
        self.m2[valid] += (
            batch_m2[valid]
            + np.square(delta[valid])
            * previous_count[valid]
            * batch_count[valid]
            / combined_count[valid]
        )
        self.minimum = np.minimum(
            self.minimum, np.min(np.where(finite, matrix, np.inf), axis=0)
        )
        self.maximum = np.maximum(
            self.maximum, np.max(np.where(finite, matrix, -np.inf), axis=0)
        )
        if self.categorical:
            rounded = np.rint(matrix)
            self.integer_like_count += (
                finite & (np.abs(matrix - rounded) <= 1e-9)
            ).sum(axis=0, dtype=np.int64)
            for index in range(self.width):
                if batch_count[index]:
                    self.unique_values[index].update(
                        int(value) for value in matrix[finite[:, index], index]
                    )

    def finalize(self, fields: Sequence[str]) -> dict[str, dict[str, Any]]:
        if len(fields) != self.width or self.row_count <= 0:
            raise RuntimeError("SURFACE_LIVENESS_STATS_FINALIZE_INVALID")
        result: dict[str, dict[str, Any]] = {}
        for index, field in enumerate(fields):
            count = int(self.finite_count[index])
            minimum = float(self.minimum[index]) if count else 0.0
            maximum = float(self.maximum[index]) if count else 0.0
            std = math.sqrt(max(float(self.m2[index]) / count, 0.0)) if count else 0.0
            row: dict[str, Any] = {
                "row_count": int(self.row_count),
                "finite_count": count,
                "nonfinite_count": int(self.nonfinite_count[index]),
                "std": std,
                "min": minimum,
                "max": maximum,
                "value_range": max(maximum - minimum, 0.0),
                "active_count": int(self.active_count[index]),
                "active_rate": float(self.active_count[index]) / self.row_count,
            }
            if self.categorical:
                row.update(
                    {
                        "integer_like_count": int(self.integer_like_count[index]),
                        "unique_count": len(self.unique_values[index]),
                        "unique_values": sorted(self.unique_values[index]),
                    }
                )
            result[str(field)] = row
        return result


def _field_liveness(
    *, stats: Mapping[str, Any], categorical: bool
) -> tuple[bool, list[str]]:
    rows = int(stats.get("row_count") or 0)
    reasons: list[str] = []
    if rows <= 0:
        reasons.append("empty_population")
    if int(stats.get("finite_count") or 0) != rows or int(stats.get("nonfinite_count") or 0):
        reasons.append("nonfinite")
    if categorical:
        if int(stats.get("integer_like_count") or 0) != rows:
            reasons.append("non_integer_categorical")
        if int(stats.get("unique_count") or 0) < 2:
            reasons.append("single_category")
    else:
        if float(stats.get("std") or 0.0) < float(NEAR_CONSTANT_STD):
            reasons.append("near_constant_std")
        if float(stats.get("value_range") or 0.0) < float(NEAR_CONSTANT_STD):
            reasons.append("near_constant_range")
        if int(stats.get("active_count") or 0) <= 0:
            reasons.append("no_active_value")
    return not reasons, reasons


def _scan_surface(
    *,
    surface_path: Path,
    timeframe: str,
    decision_seconds: int,
    signal_fields: list[str],
    audit_start_ns: int,
    batch_size: int,
) -> dict[str, Any]:
    try:
        parquet = pq.ParquetFile(surface_path)
    except Exception as exc:
        raise RuntimeError(f"SURFACE_LIVENESS_PARQUET_INVALID: {surface_path}") from exc
    if tuple(parquet.schema_arrow.names) != ENTRY_EXIT_FEATURE_SURFACE_COLUMNS:
        raise RuntimeError(f"SURFACE_LIVENESS_PARQUET_SCHEMA_INVALID: {surface_path}")
    signal_stats = _StreamingFieldStats(len(signal_fields))
    context_stats = _StreamingFieldStats(len(MODEL_NATIVE_CTX_CONT_FIELDS))
    category_stats = _StreamingFieldStats(len(MODEL_NATIVE_CTX_CAT_FIELDS), categorical=True)
    previous_time: int | None = None
    source_first_time: int | None = None
    first_time: int | None = None
    source_rows = 0
    excluded_rows = 0
    for batch in parquet.iter_batches(
        batch_size=int(batch_size), columns=list(REQUIRED_COLUMNS), use_threads=False
    ):
        times = _batch_times_ns(batch, previous=previous_time, seconds=decision_seconds)
        signal = _batch_matrix(batch, name="signal", width=len(signal_fields), dtype=np.dtype(np.float32))
        context = _batch_matrix(
            batch, name="ctx_cont", width=len(MODEL_NATIVE_CTX_CONT_FIELDS), dtype=np.dtype(np.float32)
        )
        category = _batch_matrix(
            batch, name="ctx_cat", width=len(MODEL_NATIVE_CTX_CAT_FIELDS), dtype=np.dtype(np.int64)
        )
        source_rows += int(times.size)
        if source_first_time is None:
            source_first_time = int(times[0])
        previous_time = int(times[-1])
        mask = times >= int(audit_start_ns)
        excluded_rows += int(np.count_nonzero(~mask))
        if not np.any(mask):
            continue
        signal_stats.update(signal[mask])
        context_stats.update(context[mask])
        category_stats.update(category[mask])
        if first_time is None:
            first_time = int(times[mask][0])
    if first_time is None or source_first_time is None or previous_time is None:
        raise RuntimeError(f"SURFACE_LIVENESS_NO_ROWS: {surface_path}")
    field_stats: dict[str, dict[str, Any]] = {}
    for prefix, stats, fields, categorical in (
        ("local.signal.", signal_stats, signal_fields, False),
        ("local.ctx_cont.", context_stats, MODEL_NATIVE_CTX_CONT_FIELDS, False),
        ("local.ctx_cat.", category_stats, MODEL_NATIVE_CTX_CAT_FIELDS, True),
    ):
        for name, row in stats.finalize(fields).items():
            live, reasons = _field_liveness(stats=row, categorical=categorical)
            field_stats[f"{prefix}{name}"] = {
                **row,
                "categorical": categorical,
                "live": live,
                "liveness_failures": reasons,
            }
    if source_rows != int(parquet.metadata.num_rows or 0):
        raise RuntimeError("SURFACE_LIVENESS_SOURCE_ROW_COUNT_INVALID")
    return {
        "timeframe": timeframe,
        "row_count": signal_stats.row_count,
        "source_row_count": source_rows,
        "excluded_pre_history_row_count": excluded_rows,
        "audit_start_time_ns": int(audit_start_ns),
        "source_first_time_ns": source_first_time,
        "first_time_ns": first_time,
        "last_time_ns": previous_time,
        "field_stats": field_stats,
    }


def _family_liveness(
    *, field_stats: Mapping[str, Mapping[str, Any]], signal_fields: Sequence[str]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    family_fields: dict[str, list[str]] = {family: [] for family in MODEL_NATIVE_TRAINING_SPECIALISTS}
    invalid: list[dict[str, Any]] = []
    keys = [
        *(f"local.signal.{field}" for field in signal_fields),
        *(f"local.ctx_cont.{field}" for field in MODEL_NATIVE_CTX_CONT_FIELDS),
        *(f"local.ctx_cat.{field}" for field in MODEL_NATIVE_CTX_CAT_FIELDS),
    ]
    if set(keys) != set(field_stats):
        raise RuntimeError("SURFACE_LIVENESS_FIELD_SET_INVALID")
    for key in keys:
        if key.startswith("local.signal."):
            owner_name = key.removeprefix("local.signal.")
        elif key.startswith("local.ctx_cont."):
            owner_name = "ctx_cont." + key.removeprefix("local.ctx_cont.")
        else:
            owner_name = "ctx_cat." + key.removeprefix("local.ctx_cat.")
        family = classify_entry_specialist_feature(owner_name)
        if family not in family_fields:
            invalid.append({"field": key, "classifier_result": family})
            continue
        family_fields[family].append(key)
    rows: dict[str, dict[str, Any]] = {}
    for family in MODEL_NATIVE_TRAINING_SPECIALISTS:
        fields = family_fields[family]
        dead = [field for field in fields if field_stats[field].get("live") is not True]
        rows[family] = {
            "field_count": len(fields),
            "live_field_count": len(fields) - len(dead),
            "all_fields_live": bool(fields) and not dead,
            "dead_fields": dead,
        }
        if not fields:
            invalid.append({"family": family, "reason": "empty_family"})
    return rows, invalid


def _cross_population_matches(
    *, scan: Mapping[str, Any], cross_row: Mapping[str, Any]
) -> bool:
    keys = (
        "row_count",
        "source_row_count",
        "excluded_pre_history_row_count",
        "audit_start_time_ns",
        "source_first_time_ns",
        "first_time_ns",
        "last_time_ns",
    )
    return all(scan.get(key) == cross_row.get(key) for key in keys)


def run(args: argparse.Namespace) -> dict[str, Any]:
    entry_run_id = require_entry_run_id(args.run_id)
    out_path = Path(args.out_json).expanduser().resolve()
    if OUTPUT_RE.fullmatch(out_path.name) is None:
        raise RuntimeError(f"SURFACE_LIVENESS_OUTPUT_FILENAME_INVALID: {out_path.name}")
    if out_path.exists() or out_path.is_symlink() or out_path.parent.is_symlink() or not out_path.parent.is_dir():
        raise RuntimeError(f"SURFACE_LIVENESS_OUTPUT_NOT_FRESH: {out_path}")
    batch_size = int(args.batch_size)
    if not 1 <= batch_size <= 32_768:
        raise RuntimeError("SURFACE_LIVENESS_BATCH_SIZE_INVALID")
    signal_manifest = _require_regular_absolute(Path(args.signal_manifest), label="SURFACE_LIVENESS_SIGNAL_MANIFEST")
    signal_fields, signal_manifest_sha, history_start_ns = _load_signal_contract(signal_manifest)
    m1_surface = _validate_surface(
        surface_path=Path(args.m1_feature_base_parquet), signal_manifest_path=signal_manifest,
        signal_manifest_sha256=signal_manifest_sha, signal_fields=signal_fields,
        entry_run_id=entry_run_id, timeframe="M1",
    )
    m5_surface = _validate_surface(
        surface_path=Path(args.m5_feature_base_parquet), signal_manifest_path=signal_manifest,
        signal_manifest_sha256=signal_manifest_sha, signal_fields=signal_fields,
        entry_run_id=entry_run_id, timeframe="M5",
    )
    cross_path = _require_regular_absolute(Path(args.cross_surface_overlap_json), label="SURFACE_LIVENESS_CROSS_SURFACE")
    cross = validate_cross_surface_overlap_report(
        cross_path,
        expected_entry_run_id=entry_run_id,
        expected_input_bindings={
            "signal_manifest": {"path": str(signal_manifest), "sha256": signal_manifest_sha},
            "m1_feature_surface": {"path": m1_surface["path"], "sha256": m1_surface["sha256"]},
            "m5_feature_surface": {"path": m5_surface["path"], "sha256": m5_surface["sha256"]},
        },
    )
    cross_payload = _read_json(cross_path, label="SURFACE_LIVENESS_CROSS_SURFACE")
    if (
        int(cross_payload["entry"].get("audit_start_time_ns") or -1) != history_start_ns
        or int(cross_payload["exit"].get("audit_start_time_ns") or -1) != history_start_ns
    ):
        raise RuntimeError("SURFACE_LIVENESS_CROSS_POPULATION_START_MISMATCH")
    entry = _scan_surface(
        surface_path=Path(m5_surface["path"]), timeframe="M5", decision_seconds=ENTRY_DECISION_BAR_SECONDS,
        signal_fields=signal_fields, audit_start_ns=history_start_ns, batch_size=batch_size,
    )
    exit_ = _scan_surface(
        surface_path=Path(m1_surface["path"]), timeframe="M1", decision_seconds=EXIT_DECISION_BAR_SECONDS,
        signal_fields=signal_fields, audit_start_ns=history_start_ns, batch_size=batch_size,
    )
    entry_families, entry_routing_issues = _family_liveness(field_stats=entry["field_stats"], signal_fields=signal_fields)
    exit_families, exit_routing_issues = _family_liveness(field_stats=exit_["field_stats"], signal_fields=signal_fields)
    failures: list[dict[str, Any]] = []
    for decision, scan, cross_row, families, routing_issues in (
        ("entry", entry, cross_payload["entry"], entry_families, entry_routing_issues),
        ("exit", exit_, cross_payload["exit"], exit_families, exit_routing_issues),
    ):
        dead = [name for name, stats in scan["field_stats"].items() if stats.get("live") is not True]
        if dead:
            failures.append({"code": "dead_or_nonfinite_local_feature", "decision": decision, "fields": dead})
        if routing_issues:
            failures.append({"code": "eight_family_routing_invalid", "decision": decision, "issues": routing_issues})
        failed_families = [family for family, row in families.items() if row.get("all_fields_live") is not True]
        if failed_families:
            failures.append({"code": "family_not_fully_live", "decision": decision, "families": failed_families})
        if not _cross_population_matches(scan=scan, cross_row=cross_row):
            failures.append({"code": "population_differs_from_cross_surface_proof", "decision": decision})
    report = {
        "schema_version": SCHEMA_VERSION,
        "producer_schema_version": PRODUCER_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "entry_run_id": entry_run_id,
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "policy": {
            "version": POLICY_VERSION,
            "population": "manifest_bound_history_start_through_surface_end",
            "numeric_liveness": {
                "active_abs_threshold": LIVENESS_EPSILON,
                "near_constant_std": NEAR_CONSTANT_STD,
                "require_finite": True,
            },
            "categorical_liveness": "finite_integer_and_more_than_one_observed_category",
            "family_rule": "every_locally_consumed_field_of_each_of_eight_families_must_be_live",
        },
        "input_bindings": {
            "signal_manifest": {"path": str(signal_manifest), "sha256": signal_manifest_sha},
            "m1_feature_surface": m1_surface,
            "m5_feature_surface": m5_surface,
            "cross_surface_overlap": cross,
        },
        "entry": {**entry, "eight_family_liveness": entry_families},
        "exit": {**exit_, "eight_family_liveness": exit_families},
    }
    with out_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--signal-manifest", required=True)
    parser.add_argument("--m1-feature-base-parquet", required=True)
    parser.add_argument("--m5-feature-base-parquet", required=True)
    parser.add_argument("--cross-surface-overlap-json", required=True)
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
