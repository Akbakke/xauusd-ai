#!/usr/bin/env python3
"""Build pack-independent inputs for the one-year pilot normalization fit.

The artifacts produced here bind physical TRAIN feature rows.  They deliberately
contain no transition-sampler, chunk, reward, Q-value, or outcome selection.
The view binds the immutable lifetime-summary registry and is then ready for a
separate outcome-blind TRAIN-only normalization fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.entry_sequence_source_reconstruction_v1 import (
    feature_surface_binding_from_split_manifest,
    require_sequence_source_reconstruction_audit,
)
from gx1.contracts.unified_exit_lifetime_summary_v1 import (
    LIFETIME_SUMMARY_FIELD_ORDER,
    lifetime_summary_registry,
)
from gx1.contracts.unified_exit_market_closure_authority_v1 import (
    MARKET_CLOSURE_AUTHORITY_SCHEMA_VERSION,
    m1_clock_sha256,
    require_exact_market_schedule,
    require_market_closure_authority,
)
from gx1.scripts.audit_entry_sequence_source_reconstruction_v1 import (
    _feature_surface_from_manifest,
    _load_surface_signal,
)
from gx1.scripts.prepare_unified_exit_lifecycle_v2_pilot_v1 import TRAIN_END


BUNDLE_SCHEMA_VERSION = "gx1_unified_exit_pilot_normalization_inputs_v1"
SEQUENCE_AUDIT_SCHEMA_VERSION = (
    "gx1_unified_exit_pilot_child_sequence_reconstruction_audit_v1"
)
POPULATION_SCHEMA_VERSION = "gx1_unified_exit_pilot_train_normalization_population_v1"
VIEW_SCHEMA_VERSION = "gx1_unified_exit_pilot_child_normalization_view_v1"
EXPECTED_CHILD_ADMISSION_SCHEMA = "gx1_lifecycle_v2_pilot_child_view_admission_v1"
EXPECTED_TRAIN_ROWS = 65_295
EXPECTED_VAL_ROWS = 5_508
EXPECTED_SUMMARY_REGISTRY_SCHEMA = "gx1_unified_exit_lifetime_summary_v1"
ENTRY_BAR_NS = 5 * 60 * 1_000_000_000
EXIT_BAR_NS = 60 * 1_000_000_000
ARROW_BATCH_ROWS = 256


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"PILOT_NORMALIZATION_{label}_SHA_INVALID")
    return value


def _exact_file(raw: Path, label: str) -> Path:
    supplied = Path(raw).expanduser()
    if (
        not supplied.is_absolute()
        or supplied.is_symlink()
        or any(parent.is_symlink() for parent in supplied.parents)
    ):
        raise RuntimeError(f"PILOT_NORMALIZATION_{label}_PATH_INVALID")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"PILOT_NORMALIZATION_{label}_PATH_INVALID") from exc
    if resolved != supplied or not resolved.is_file():
        raise RuntimeError(f"PILOT_NORMALIZATION_{label}_PATH_INVALID")
    return resolved


def _exact_dir(raw: Path, label: str) -> Path:
    supplied = Path(raw).expanduser()
    if (
        not supplied.is_absolute()
        or supplied.is_symlink()
        or any(parent.is_symlink() for parent in supplied.parents)
    ):
        raise RuntimeError(f"PILOT_NORMALIZATION_{label}_PATH_INVALID")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"PILOT_NORMALIZATION_{label}_PATH_INVALID") from exc
    if resolved != supplied or not resolved.is_dir():
        raise RuntimeError(f"PILOT_NORMALIZATION_{label}_PATH_INVALID")
    return resolved


def _read_json(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(key)
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(f"PILOT_NORMALIZATION_{label}_JSON_INVALID") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"PILOT_NORMALIZATION_{label}_JSON_INVALID")
    return value


def _clock_hash(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype="<i8").tobytes()
    ).hexdigest()


def _time_ns(column: Any, label: str) -> np.ndarray:
    try:
        values = column.to_numpy(zero_copy_only=False).astype("datetime64[ns]")
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"PILOT_NORMALIZATION_{label}_TIME_INVALID") from exc
    values = values.astype(np.int64, copy=False)
    if np.any(values == np.iinfo(np.int64).min):
        raise RuntimeError(f"PILOT_NORMALIZATION_{label}_TIME_INVALID")
    return values


def _require_child_admission(
    path: Path,
    *,
    expected_train_rows: int,
    expected_val_rows: int,
) -> tuple[dict[str, Any], str]:
    admission_path = _exact_file(path, "CHILD_ADMISSION")
    observed = _read_json(admission_path, "CHILD_ADMISSION")
    claimed = observed.get("witness_sha256")
    if (
        observed.get("schema_version") != EXPECTED_CHILD_ADMISSION_SCHEMA
        or observed.get("decision") != "PASS"
        or observed.get("test_accessed") is not False
        or not isinstance(observed.get("splits"), Mapping)
        or observed["splits"].get("train", {}).get("rows") != expected_train_rows
        or observed["splits"].get("val", {}).get("rows") != expected_val_rows
        or claimed
        != _canonical_sha256(
            {key: value for key, value in observed.items() if key != "witness_sha256"}
        )
    ):
        raise RuntimeError("PILOT_NORMALIZATION_CHILD_ADMISSION_INVALID")
    for split in ("train", "val"):
        item = observed["splits"][split]
        if not isinstance(item, Mapping):
            raise RuntimeError("PILOT_NORMALIZATION_CHILD_ADMISSION_INVALID")
        parquet = _exact_file(Path(item["parquet_path"]), f"{split.upper()}_PARQUET")
        manifest = _exact_file(Path(item["manifest_path"]), f"{split.upper()}_MANIFEST")
        if _sha256_file(parquet) != _require_sha(
            item["parquet_sha256"], split
        ) or _sha256_file(manifest) != _require_sha(
            item["manifest_file_sha256"], f"{split}_manifest"
        ):
            raise RuntimeError("PILOT_NORMALIZATION_CHILD_BYTES_CHANGED")
    return observed, _sha256_file(admission_path)


def _parent_sources(
    child: Mapping[str, Any],
    *,
    parent_sequence_audit_path: Path,
    mtf_cache_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    child_manifest_path = Path(child["splits"]["train"]["manifest_path"])
    child_manifest = _read_json(child_manifest_path, "CHILD_TRAIN_MANIFEST")
    source = child_manifest.get("source_manifest")
    if not isinstance(source, Mapping):
        raise RuntimeError("PILOT_NORMALIZATION_PARENT_MANIFEST_BINDING_INVALID")
    parent_path = _exact_file(Path(source["path"]), "PARENT_TRAIN_MANIFEST")
    parent_sha = _sha256_file(parent_path)
    if parent_sha != _require_sha(source["sha256"], "PARENT_TRAIN_MANIFEST"):
        raise RuntimeError("PILOT_NORMALIZATION_PARENT_MANIFEST_BINDING_INVALID")
    parent = _read_json(parent_path, "PARENT_TRAIN_MANIFEST")
    source_parquet = child_manifest.get("source_parquet")
    if (
        parent.get("extra", {}).get("entry_run_id")
        != child.get("parent_dataset_run_id")
        or not isinstance(parent.get("feature_contract"), Mapping)
        or not isinstance(source_parquet, Mapping)
        or parent.get("output_data_path") != source_parquet.get("path")
    ):
        raise RuntimeError("PILOT_NORMALIZATION_PARENT_MANIFEST_INVALID")

    parent_audit_path = _exact_file(parent_sequence_audit_path, "PARENT_SEQUENCE_AUDIT")
    parent_audit = _read_json(parent_audit_path, "PARENT_SEQUENCE_AUDIT")
    try:
        require_sequence_source_reconstruction_audit(
            parent_audit,
            expected_parquet_path=Path(parent["output_data_path"]),
            expected_manifest_path=parent_path,
            expected_parquet_sha256=_require_sha(
                source_parquet.get("sha256"), "PARENT_TRAIN_PARQUET"
            ),
            expected_manifest_sha256=parent_sha,
            expected_feature_surface=parent,
            expected_rows=int(parent_audit["rows"]),
            expected_seq_len=MODEL_NATIVE_SEQ_LEN,
            expected_signal_dim=MODEL_NATIVE_SIGNAL_DIM,
        )
    except Exception as exc:
        raise RuntimeError("PILOT_NORMALIZATION_PARENT_SEQUENCE_AUDIT_INVALID") from exc

    binding = parent.get("extra", {}).get("multi_tf_cache_binding")
    if not isinstance(binding, Mapping):
        raise RuntimeError("PILOT_NORMALIZATION_MTF_BINDING_INVALID")
    cache_manifest_path = _exact_file(mtf_cache_manifest_path, "MTF_CACHE_MANIFEST")
    if str(cache_manifest_path) != binding.get("manifest_path") or _sha256_file(
        cache_manifest_path
    ) != _require_sha(binding.get("manifest_sha256"), "MTF_CACHE_MANIFEST"):
        raise RuntimeError("PILOT_NORMALIZATION_MTF_BINDING_INVALID")
    cache_manifest = _read_json(cache_manifest_path, "MTF_CACHE_MANIFEST")
    if (
        cache_manifest.get("cache_identity_sha256")
        != binding.get("cache_identity_sha256")
        or cache_manifest.get("m5_prebuilt_source") != binding.get("m5_prebuilt_source")
        or cache_manifest.get("m5_prebuilt_source_sha256")
        != binding.get("m5_prebuilt_source_sha256")
        or cache_manifest.get("v29_registry_constants")
        != binding.get("v29_registry_constants")
        or cache_manifest.get("volatility_squeeze_artifact_set")
        != binding.get("volatility_squeeze_artifact_set")
    ):
        raise RuntimeError("PILOT_NORMALIZATION_MTF_BINDING_INVALID")
    cache_dir = _exact_dir(Path(binding["cache_dir"]), "MTF_CACHE_DIR")
    if cache_manifest.get("tfs", {}).keys() != {"M5", "M15", "H1", "H4", "D1"}:
        raise RuntimeError("PILOT_NORMALIZATION_MTF_BINDING_INVALID")
    for tf, item in cache_manifest["tfs"].items():
        if not isinstance(item, Mapping):
            raise RuntimeError("PILOT_NORMALIZATION_MTF_BINDING_INVALID")
        for stem in ("feats_npy", "ts_npy", "model_native_scalars_npy"):
            relative = item.get(stem)
            if (
                not isinstance(relative, str)
                or Path(relative).name != relative
                or not relative
            ):
                raise RuntimeError("PILOT_NORMALIZATION_MTF_BINDING_INVALID")
            artifact = _exact_file(cache_dir / relative, f"MTF_{tf}_{stem}")
            if artifact.stat().st_size != item.get(
                f"{stem}_size_bytes"
            ) or _sha256_file(artifact) != _require_sha(
                item.get(f"{stem}_sha256"), f"MTF_{tf}_{stem}"
            ):
                raise RuntimeError("PILOT_NORMALIZATION_MTF_BINDING_INVALID")
    m5_source = _exact_file(
        Path(binding["m5_prebuilt_source"]), "MTF_M5_PREBUILT_SOURCE"
    )
    if _sha256_file(m5_source) != _require_sha(
        binding["m5_prebuilt_source_sha256"], "MTF_M5_PREBUILT_SOURCE"
    ):
        raise RuntimeError("PILOT_NORMALIZATION_MTF_BINDING_INVALID")
    return parent, parent_audit, dict(binding)


def build_child_sequence_reconstruction_audit(
    *,
    child_admission: Mapping[str, Any],
    child_admission_file_sha256: str,
    parent_manifest: Mapping[str, Any],
    parent_sequence_audit: Mapping[str, Any],
    parent_sequence_audit_path: Path,
) -> dict[str, Any]:
    child = child_admission["splits"]["train"]
    child_path = Path(child["parquet_path"])
    try:
        surface_binding, surface_path, surface_manifest_path = (
            _feature_surface_from_manifest(dict(parent_manifest))
        )
    except Exception as exc:
        raise RuntimeError("PILOT_NORMALIZATION_M5_FEATURE_SURFACE_CHANGED") from exc
    source_times, source_signal = _load_surface_signal(
        surface_path, expected_rows=int(surface_binding["rows"])
    )

    parquet = pq.ParquetFile(child_path)
    if not {"time", "seq", "snap"}.issubset(parquet.schema_arrow.names):
        raise RuntimeError("PILOT_NORMALIZATION_CHILD_SEQUENCE_COLUMNS_MISSING")
    expected_rows = int(child["rows"])
    if parquet.metadata.num_rows != expected_rows:
        raise RuntimeError("PILOT_NORMALIZATION_CHILD_SEQUENCE_ROWS_INVALID")
    stream = hashlib.sha256()
    stream.update(b"gx1_unified_exit_pilot_child_sequence_reconstruction_v1\0")
    history_offsets = np.arange(MODEL_NATIVE_SEQ_LEN, dtype=np.int64) - (
        MODEL_NATIVE_SEQ_LEN - 1
    )
    observed = 0
    prior: int | None = None
    mapped_positions: list[np.ndarray] = []
    for batch in parquet.iter_batches(
        batch_size=ARROW_BATCH_ROWS,
        columns=["time", "seq", "snap"],
        use_threads=False,
    ):
        count = batch.num_rows
        times = _time_ns(batch.column("time"), "CHILD_SEQUENCE")
        positions = np.searchsorted(source_times, times).astype(np.int64, copy=False)
        if (
            np.any(positions < MODEL_NATIVE_SEQ_LEN - 1)
            or np.any(positions >= len(source_times))
            or not np.array_equal(source_times[positions], times)
            or (prior is not None and int(times[0]) <= prior)
            or (count > 1 and np.any(np.diff(times) <= 0))
        ):
            raise RuntimeError(
                "PILOT_NORMALIZATION_CHILD_SEQUENCE_TIME_MAPPING_INVALID"
            )
        sequence = (
            batch.column("seq")
            .flatten()
            .flatten()
            .to_numpy(zero_copy_only=False)
            .reshape(count, MODEL_NATIVE_SEQ_LEN, MODEL_NATIVE_SIGNAL_DIM)
            .astype(np.float32, copy=False)
        )
        snapshot = (
            batch.column("snap")
            .flatten()
            .to_numpy(zero_copy_only=False)
            .reshape(count, MODEL_NATIVE_SIGNAL_DIM)
            .astype(np.float32, copy=False)
        )
        expected = source_signal[positions[:, None] + history_offsets[None, :]]
        if (
            not np.isfinite(sequence).all()
            or not np.isfinite(snapshot).all()
            or not np.array_equal(sequence, expected)
            or not np.array_equal(snapshot, source_signal[positions])
        ):
            raise RuntimeError("PILOT_NORMALIZATION_CHILD_SEQUENCE_VALUE_MISMATCH")
        stream.update(np.ascontiguousarray(times, dtype="<i8").tobytes())
        stream.update(np.ascontiguousarray(positions, dtype="<i8").tobytes())
        stream.update(np.ascontiguousarray(sequence, dtype="<f4").tobytes())
        stream.update(np.ascontiguousarray(snapshot, dtype="<f4").tobytes())
        mapped_positions.append(positions.copy())
        prior = int(times[-1])
        observed += count
    if observed != expected_rows:
        raise RuntimeError("PILOT_NORMALIZATION_CHILD_SEQUENCE_ROWS_INVALID")
    all_positions = np.concatenate(mapped_positions)
    audit = {
        "schema_version": SEQUENCE_AUDIT_SCHEMA_VERSION,
        "decision": "PASS",
        "scope": "child_train_only",
        "child_dataset_run_id": child_admission["child_dataset_run_id"],
        "child_admission_file_sha256": child_admission_file_sha256,
        "child_admission_witness_sha256": child_admission["witness_sha256"],
        "child_train_parquet_path": str(child_path),
        "child_train_parquet_sha256": child["parquet_sha256"],
        "child_train_manifest_path": child["manifest_path"],
        "child_train_manifest_sha256": child["manifest_file_sha256"],
        "child_train_rows": expected_rows,
        "child_train_clock_sha256": child["clock_sha256"],
        "parent_dataset_run_id": child_admission["parent_dataset_run_id"],
        "parent_sequence_audit_path": str(
            _exact_file(parent_sequence_audit_path, "PARENT_SEQUENCE_AUDIT")
        ),
        "parent_sequence_audit_file_sha256": _sha256_file(
            Path(parent_sequence_audit_path)
        ),
        "parent_sequence_audit_contract_sha256": _canonical_sha256(
            parent_sequence_audit
        ),
        "m5_feature_surface_path": str(surface_path),
        "m5_feature_surface_sha256": surface_binding["sha256"],
        "m5_feature_surface_manifest_path": str(surface_manifest_path),
        "m5_feature_surface_manifest_sha256": surface_binding["manifest_sha256"],
        "m5_feature_surface_rows": int(surface_binding["rows"]),
        "source_position_stream_sha256": _clock_hash(all_positions),
        "sequence_value_stream_sha256": stream.hexdigest(),
        "sequence_shape": [
            expected_rows,
            MODEL_NATIVE_SEQ_LEN,
            MODEL_NATIVE_SIGNAL_DIM,
        ],
        "snapshot_shape": [expected_rows, MODEL_NATIVE_SIGNAL_DIM],
        "test_rows_scanned": 0,
        "val_rows_scanned": 0,
        "test_accessed": False,
    }
    audit["contract_sha256"] = _canonical_sha256(audit)
    return audit


def _merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((int(left), int(right)) for left, right in intervals)
    if not ordered or any(left < 0 or right <= left for left, right in ordered):
        raise RuntimeError("PILOT_NORMALIZATION_POPULATION_INTERVAL_INVALID")
    merged: list[tuple[int, int]] = []
    for left, right in ordered:
        if not merged or left > merged[-1][1]:
            merged.append((left, right))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
    return merged


def _indices_from_intervals(intervals: Sequence[tuple[int, int]]) -> np.ndarray:
    return np.concatenate(
        [np.arange(left, right, dtype=np.int64) for left, right in intervals]
    )


def _selected_surface_hash(
    *,
    surface_path: Path,
    source_positions: np.ndarray,
    source_times: np.ndarray,
    namespace: bytes,
) -> str:
    parquet = pq.ParquetFile(surface_path)
    if not {"time", "signal", "ctx_cont", "ctx_cat"}.issubset(
        parquet.schema_arrow.names
    ):
        raise RuntimeError("PILOT_NORMALIZATION_FEATURE_SURFACE_SCHEMA_INVALID")
    selected = np.asarray(source_positions, dtype=np.int64)
    if selected.ndim != 1 or selected.size < 1 or np.any(np.diff(selected) <= 0):
        raise RuntimeError("PILOT_NORMALIZATION_FEATURE_SURFACE_SELECTION_INVALID")
    digest = hashlib.sha256()
    digest.update(namespace + b"\0")
    offset = 0
    consumed = 0
    cursor = 0
    for batch in parquet.iter_batches(
        batch_size=4096,
        columns=["time", "signal", "ctx_cont", "ctx_cat"],
        use_threads=False,
    ):
        count = int(batch.num_rows)
        right = int(np.searchsorted(selected, offset + count, side="left"))
        positions = selected[cursor:right]
        if positions.size:
            local = positions - offset
            times = _time_ns(batch.column("time"), "FEATURE_SURFACE")
            if not np.array_equal(times[local], source_times[positions]):
                raise RuntimeError("PILOT_NORMALIZATION_FEATURE_SURFACE_TIME_MISMATCH")
            digest.update(np.ascontiguousarray(positions, dtype="<i8").tobytes())
            digest.update(np.ascontiguousarray(times[local], dtype="<i8").tobytes())
            for name, dtype in (
                ("signal", "<f4"),
                ("ctx_cont", "<f4"),
                ("ctx_cat", "<i8"),
            ):
                values = batch.column(name)
                if not hasattr(values, "values"):
                    raise RuntimeError(
                        "PILOT_NORMALIZATION_FEATURE_SURFACE_DECODE_INVALID"
                    )
                width = int(values.type.list_size)
                matrix = values.values.to_numpy(zero_copy_only=False).reshape(
                    count, width
                )[local]
                if name != "ctx_cat" and not np.isfinite(matrix).all():
                    raise RuntimeError("PILOT_NORMALIZATION_FEATURE_SURFACE_NONFINITE")
                digest.update(np.ascontiguousarray(matrix, dtype=dtype).tobytes())
            consumed += len(positions)
        cursor = right
        offset += count
    if consumed != len(selected) or offset != parquet.metadata.num_rows:
        raise RuntimeError("PILOT_NORMALIZATION_FEATURE_SURFACE_SCAN_INCOMPLETE")
    return digest.hexdigest()


def build_train_normalization_population_witness(
    *,
    child_admission: Mapping[str, Any],
    child_admission_file_sha256: str,
    child_sequence_audit: Mapping[str, Any],
    m1_source_path: Path,
    m1_source_manifest_path: Path,
    m1_feature_base_path: Path,
    m1_feature_base_manifest_path: Path,
    market_closure_authority_path: Path,
    parent_manifest: Mapping[str, Any],
    mtf_cache_binding: Mapping[str, Any],
    mtf_cache_manifest_path: Path,
    train_end: str = TRAIN_END,
) -> dict[str, Any]:
    m1_path = _exact_file(m1_source_path, "M1_SOURCE")
    m1_manifest_path = _exact_file(m1_source_manifest_path, "M1_SOURCE_MANIFEST")
    m1_sha = _sha256_file(m1_path)
    m1_manifest_sha = _sha256_file(m1_manifest_path)
    expected_m1 = child_admission["m1_source_binding"]
    if expected_m1 != {
        "parquet_path": str(m1_path),
        "parquet_sha256": m1_sha,
        "manifest_path": str(m1_manifest_path),
        "manifest_sha256": m1_manifest_sha,
    }:
        raise RuntimeError("PILOT_NORMALIZATION_M1_BINDING_INVALID")
    m1_times = pd.DatetimeIndex(
        pd.read_parquet(m1_path, columns=["time"])["time"]
    ).as_unit("ns")
    if (
        m1_times.empty
        or m1_times.hasnans
        or not m1_times.is_unique
        or not m1_times.is_monotonic_increasing
    ):
        raise RuntimeError("PILOT_NORMALIZATION_M1_CLOCK_INVALID")
    clock_ns = np.asarray(m1_times.asi8, dtype=np.int64)
    clock_sha = m1_clock_sha256(m1_times)

    closure_path = _exact_file(
        market_closure_authority_path, "MARKET_CLOSURE_AUTHORITY"
    )
    closure_file_sha = _sha256_file(closure_path)
    closure_raw = _read_json(closure_path, "MARKET_CLOSURE_AUTHORITY")
    closure = require_market_closure_authority(
        closure_raw,
        expected_m1_source_sha256=m1_sha,
        expected_m1_clock_sha256=clock_sha,
    )
    if (
        closure["schema_version"] != MARKET_CLOSURE_AUTHORITY_SCHEMA_VERSION
        or closure["m1_source_path"] != str(m1_path)
        or closure["m1_source_manifest_path"] != str(m1_manifest_path)
        or closure["m1_source_manifest_sha256"] != m1_manifest_sha
    ):
        raise RuntimeError("PILOT_NORMALIZATION_MARKET_CLOSURE_AUTHORITY_INVALID")
    schedule_path = _exact_file(
        Path(closure["schedule_path"]), "MARKET_CLOSURE_SCHEDULE"
    )
    if _sha256_file(schedule_path) != closure["schedule_file_sha256"]:
        raise RuntimeError("PILOT_NORMALIZATION_MARKET_CLOSURE_AUTHORITY_INVALID")
    schedule = require_exact_market_schedule(
        _read_json(schedule_path, "MARKET_CLOSURE_SCHEDULE")
    )
    if schedule["schedule_sha256"] != closure["schedule_sha256"]:
        raise RuntimeError("PILOT_NORMALIZATION_MARKET_CLOSURE_AUTHORITY_INVALID")
    unknown_gap_rows = np.asarray(
        [
            int(item["gap_after_m1_row"])
            for item in closure["intervals"]
            if not item["successor_across_gap_allowed"]
        ],
        dtype=np.int64,
    )

    child_train = child_admission["splits"]["train"]
    entry_times = pd.DatetimeIndex(
        pd.read_parquet(child_train["parquet_path"], columns=["time"])["time"]
    ).as_unit("ns")
    entry_ns = np.asarray(entry_times.asi8, dtype=np.int64)
    if (
        len(entry_times) != child_train["rows"]
        or _clock_hash(entry_ns) != child_train["clock_sha256"]
    ):
        raise RuntimeError("PILOT_NORMALIZATION_CHILD_TRAIN_CLOCK_INVALID")
    starts = np.searchsorted(clock_ns, entry_ns + ENTRY_BAR_NS, side="left")
    exact = starts < len(clock_ns)
    positions = np.flatnonzero(exact)
    exact[positions] &= (
        clock_ns[starts[positions]] == entry_ns[positions] + ENTRY_BAR_NS
    )
    if not exact.all():
        raise RuntimeError("PILOT_NORMALIZATION_ENTRY_OPEN_M1_MISSING")
    split_end_ns = int(pd.Timestamp(train_end).value)
    available_stop = int(
        np.searchsorted(clock_ns, split_end_ns - EXIT_BAR_NS, side="right")
    )
    exit_intervals: list[tuple[int, int]] = []
    for raw_start in starts.tolist():
        start = int(raw_start)
        gap_pos = int(np.searchsorted(unknown_gap_rows, start, side="left"))
        stop = available_stop
        if gap_pos < len(unknown_gap_rows):
            gap_after = int(unknown_gap_rows[gap_pos])
            if gap_after < available_stop - 1:
                stop = gap_after + 1
        if stop <= start:
            raise RuntimeError("PILOT_NORMALIZATION_EXIT_POPULATION_EMPTY")
        exit_intervals.append((start, stop))
    merged_exit = _merge_intervals(exit_intervals)
    exit_indices = _indices_from_intervals(merged_exit)

    source_binding = feature_surface_binding_from_split_manifest(parent_manifest)
    m5_times, _ = _load_surface_signal(
        _exact_file(Path(source_binding["path"]), "M5_FEATURE_SURFACE"),
        expected_rows=int(source_binding["rows"]),
    )
    entry_positions = np.searchsorted(m5_times, entry_ns)
    if (
        np.any(entry_positions < MODEL_NATIVE_SEQ_LEN - 1)
        or np.any(entry_positions >= len(m5_times))
        or not np.array_equal(m5_times[entry_positions], entry_ns)
    ):
        raise RuntimeError("PILOT_NORMALIZATION_ENTRY_M5_MAPPING_INVALID")
    entry_intervals = _merge_intervals(
        [
            (int(position) - MODEL_NATIVE_SEQ_LEN + 1, int(position) + 1)
            for position in entry_positions.tolist()
        ]
    )
    entry_local_indices = _indices_from_intervals(entry_intervals)

    feature_path = _exact_file(m1_feature_base_path, "M1_FEATURE_BASE")
    feature_manifest_path = _exact_file(
        m1_feature_base_manifest_path, "M1_FEATURE_BASE_MANIFEST"
    )
    feature_sha = _sha256_file(feature_path)
    feature_manifest_sha = _sha256_file(feature_manifest_path)
    feature_manifest = _read_json(feature_manifest_path, "M1_FEATURE_BASE_MANIFEST")
    if (
        feature_manifest.get("output_parquet") != str(feature_path)
        or feature_manifest.get("output_parquet_sha256") != feature_sha
        or feature_manifest.get("alignment_parquet") != str(m1_path)
        or feature_manifest.get("alignment_sha256") != m1_sha
        or feature_manifest.get("signal_dim") != MODEL_NATIVE_SIGNAL_DIM
        or feature_manifest.get("ctx_cont_dim") != MODEL_NATIVE_CTX_CONT_DIM
        or feature_manifest.get("ctx_cat_dim") != MODEL_NATIVE_CTX_CAT_DIM
    ):
        raise RuntimeError("PILOT_NORMALIZATION_M1_FEATURE_BASE_INVALID")
    feature_parquet = pq.ParquetFile(feature_path)
    feature_times = _time_ns(
        feature_parquet.read(columns=["time"]).column("time"),
        "M1_FEATURE_BASE",
    )
    if feature_parquet.metadata.num_rows != feature_manifest.get("rows") or len(
        feature_times
    ) != feature_manifest.get("rows"):
        raise RuntimeError("PILOT_NORMALIZATION_M1_FEATURE_BASE_INVALID")
    feature_positions = np.searchsorted(feature_times, clock_ns[exit_indices])
    if np.any(feature_positions >= len(feature_times)) or not np.array_equal(
        feature_times[feature_positions], clock_ns[exit_indices]
    ):
        raise RuntimeError("PILOT_NORMALIZATION_M1_FEATURE_MAPPING_INVALID")
    exit_values_sha = _selected_surface_hash(
        surface_path=feature_path,
        source_positions=feature_positions,
        source_times=feature_times,
        namespace=b"gx1_pilot_normalization_unique_exit_m1_rows_v1",
    )

    mtf_manifest = _exact_file(mtf_cache_manifest_path, "MTF_CACHE_MANIFEST")
    if (
        str(mtf_manifest) != mtf_cache_binding["manifest_path"]
        or _sha256_file(mtf_manifest) != mtf_cache_binding["manifest_sha256"]
    ):
        raise RuntimeError("PILOT_NORMALIZATION_MTF_BINDING_INVALID")
    witness = {
        "schema_version": POPULATION_SCHEMA_VERSION,
        "decision": "PASS",
        "fit_scope": "train_unique_physical_rows_only",
        "selection_independent_of_transition_pack": True,
        "selection_basis": "physical_train_clocks_and_closure_authority_only",
        "child_dataset_run_id": child_admission["child_dataset_run_id"],
        "child_admission_file_sha256": child_admission_file_sha256,
        "child_admission_witness_sha256": child_admission["witness_sha256"],
        "child_sequence_audit_sha256": child_sequence_audit["contract_sha256"],
        "train_entry_decision_rows": len(entry_times),
        "train_entry_clock_sha256": child_train["clock_sha256"],
        "entry_m5_local_unique_rows": len(entry_local_indices),
        "entry_m5_local_intervals": [
            {"start_row": left, "end_row_exclusive": right}
            for left, right in entry_intervals
        ],
        "entry_m5_local_indices_sha256": _clock_hash(entry_local_indices),
        "exit_m1_current_unique_rows": len(exit_indices),
        "exit_m1_current_intervals": [
            {"start_row": left, "end_row_exclusive": right}
            for left, right in merged_exit
        ],
        "exit_m1_current_indices_sha256": _clock_hash(exit_indices),
        "exit_m1_feature_positions_sha256": _clock_hash(feature_positions),
        "exit_m1_selected_values_sha256": exit_values_sha,
        "m1_source": {
            "path": str(m1_path),
            "sha256": m1_sha,
            "manifest_path": str(m1_manifest_path),
            "manifest_sha256": m1_manifest_sha,
            "clock_sha256": clock_sha,
            "rows": len(m1_times),
        },
        "m1_feature_base": {
            "path": str(feature_path),
            "sha256": feature_sha,
            "manifest_path": str(feature_manifest_path),
            "manifest_sha256": feature_manifest_sha,
            "rows": int(feature_manifest["rows"]),
            "feature_field_order_sha256": feature_manifest[
                "feature_field_order_sha256"
            ],
        },
        "market_closure_authority": {
            "path": str(closure_path),
            "file_sha256": closure_file_sha,
            "artifact_sha256": closure["artifact_sha256"],
            "schema_version": closure["schema_version"],
            "known_market_closure_count": closure["known_market_closure_count"],
            "unknown_source_gap_count": closure["unknown_source_gap_count"],
        },
        "mtf_cache": {
            "cache_dir": mtf_cache_binding["cache_dir"],
            "manifest_path": str(mtf_manifest),
            "manifest_sha256": mtf_cache_binding["manifest_sha256"],
            "cache_identity_sha256": mtf_cache_binding["cache_identity_sha256"],
        },
        "train_end_utc_exclusive": pd.Timestamp(train_end).isoformat(),
        "val_fit_rows": 0,
        "test_fit_rows": 0,
        "test_accessed": False,
    }
    witness["contract_sha256"] = _canonical_sha256(witness)
    return witness


def build_normalization_inputs(
    *,
    pilot_root: Path,
    output_dir: Path,
    child_admission_path: Path,
    parent_sequence_audit_path: Path,
    m1_source_path: Path,
    m1_source_manifest_path: Path,
    m1_feature_base_path: Path,
    m1_feature_base_manifest_path: Path,
    mtf_cache_manifest_path: Path,
    market_closure_authority_path: Path,
    publish: bool,
    expected_train_rows: int = EXPECTED_TRAIN_ROWS,
    expected_val_rows: int = EXPECTED_VAL_ROWS,
) -> dict[str, Any]:
    if type(publish) is not bool:
        raise RuntimeError("PILOT_NORMALIZATION_INVOCATION_INVALID")
    root = Path(pilot_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output != root / "NORMALIZATION_INPUTS":
        raise RuntimeError("PILOT_NORMALIZATION_OUTPUT_PATH_INVALID")
    admission, admission_file_sha = _require_child_admission(
        child_admission_path,
        expected_train_rows=expected_train_rows,
        expected_val_rows=expected_val_rows,
    )
    parent, parent_audit, mtf_binding = _parent_sources(
        admission,
        parent_sequence_audit_path=parent_sequence_audit_path,
        mtf_cache_manifest_path=mtf_cache_manifest_path,
    )
    sequence_audit = build_child_sequence_reconstruction_audit(
        child_admission=admission,
        child_admission_file_sha256=admission_file_sha,
        parent_manifest=parent,
        parent_sequence_audit=parent_audit,
        parent_sequence_audit_path=parent_sequence_audit_path,
    )
    population = build_train_normalization_population_witness(
        child_admission=admission,
        child_admission_file_sha256=admission_file_sha,
        child_sequence_audit=sequence_audit,
        m1_source_path=m1_source_path,
        m1_source_manifest_path=m1_source_manifest_path,
        m1_feature_base_path=m1_feature_base_path,
        m1_feature_base_manifest_path=m1_feature_base_manifest_path,
        market_closure_authority_path=market_closure_authority_path,
        parent_manifest=parent,
        mtf_cache_binding=mtf_binding,
        mtf_cache_manifest_path=mtf_cache_manifest_path,
    )
    sequence_path = output / "CHILD_TRAIN_SEQUENCE_RECONSTRUCTION_AUDIT.json"
    population_path = output / "TRAIN_NORMALIZATION_POPULATION_WITNESS.json"
    sequence_bytes = _json_bytes(sequence_audit)
    population_bytes = _json_bytes(population)
    parent_manifest_path = Path(
        _read_json(
            Path(admission["splits"]["train"]["manifest_path"]),
            "CHILD_TRAIN_MANIFEST",
        )["source_manifest"]["path"]
    )
    summary_registry = lifetime_summary_registry()
    if (
        summary_registry["schema_version"] != EXPECTED_SUMMARY_REGISTRY_SCHEMA
        or summary_registry["field_order"] != list(LIFETIME_SUMMARY_FIELD_ORDER)
    ):
        raise RuntimeError("PILOT_NORMALIZATION_SUMMARY_REGISTRY_INVALID")
    view = {
        "schema_version": VIEW_SCHEMA_VERSION,
        "decision": "PASS",
        "blockers": [],
        "child_dataset_run_id": admission["child_dataset_run_id"],
        "child_admission": {
            "path": str(Path(child_admission_path)),
            "file_sha256": admission_file_sha,
            "witness_sha256": admission["witness_sha256"],
        },
        "child_train": dict(admission["splits"]["train"]),
        "parent_train_manifest": {
            "path": str(parent_manifest_path),
            "sha256": _sha256_file(parent_manifest_path),
            "feature_contract_sha256": _canonical_sha256(parent["feature_contract"]),
            "feature_surface_binding_sha256": _canonical_sha256(
                feature_surface_binding_from_split_manifest(parent)
            ),
            "multi_tf_cache_binding_sha256": _canonical_sha256(mtf_binding),
        },
        "child_sequence_reconstruction_audit": {
            "path": str(sequence_path),
            "file_sha256": hashlib.sha256(sequence_bytes).hexdigest(),
            "contract_sha256": sequence_audit["contract_sha256"],
        },
        "train_normalization_population_witness": {
            "path": str(population_path),
            "file_sha256": hashlib.sha256(population_bytes).hexdigest(),
            "contract_sha256": population["contract_sha256"],
        },
        "lifetime_summary_registry": {
            "schema_version": summary_registry["schema_version"],
            "status": "BOUND",
            "registry_sha256": summary_registry["registry_sha256"],
            "field_order": summary_registry["field_order"],
            "field_order_sha256": summary_registry["field_order_sha256"],
            "dimension": summary_registry["dimension"],
        },
        "normalization_fit_status": "READY_FOR_TRAIN_ONLY_FIT",
        "final_normalization_published": False,
        "first_state_witness_published": False,
        "val_fit_rows": 0,
        "test_fit_rows": 0,
        "test_accessed": False,
    }
    view["contract_sha256"] = _canonical_sha256(view)
    if not publish:
        return {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "mode": "validate_no_publish",
            "decision": "PASS",
            "published": False,
            "output_dir": str(output),
            "sequence_audit": sequence_audit,
            "population_witness": population,
            "normalization_view": view,
        }
    if output.exists() or output.is_symlink():
        raise RuntimeError("PILOT_NORMALIZATION_OUTPUT_EXISTS")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent)
    )
    try:
        (staging / sequence_path.name).write_bytes(sequence_bytes)
        (staging / population_path.name).write_bytes(population_bytes)
        (staging / "CHILD_NORMALIZATION_VIEW.json").write_bytes(_json_bytes(view))
        for path in staging.iterdir():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "mode": "publish_intermediate_inputs",
        "decision": "PASS",
        "published": True,
        "output_dir": str(output),
        "normalization_view_contract_sha256": view["contract_sha256"],
        "normalization_fit_status": "READY_FOR_TRAIN_ONLY_FIT",
        "test_accessed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "build pack-independent one-year pilot normalization inputs"
    )
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--child-admission", type=Path, required=True)
    parser.add_argument("--parent-sequence-audit", type=Path, required=True)
    parser.add_argument("--m1-source", type=Path, required=True)
    parser.add_argument("--m1-source-manifest", type=Path, required=True)
    parser.add_argument("--m1-feature-base", type=Path, required=True)
    parser.add_argument("--m1-feature-base-manifest", type=Path, required=True)
    parser.add_argument("--mtf-cache-manifest", type=Path, required=True)
    parser.add_argument("--market-closure-authority", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_normalization_inputs(
        pilot_root=args.pilot_root,
        output_dir=args.output_dir,
        child_admission_path=args.child_admission,
        parent_sequence_audit_path=args.parent_sequence_audit,
        m1_source_path=args.m1_source,
        m1_source_manifest_path=args.m1_source_manifest,
        m1_feature_base_path=args.m1_feature_base,
        m1_feature_base_manifest_path=args.m1_feature_base_manifest,
        mtf_cache_manifest_path=args.mtf_cache_manifest,
        market_closure_authority_path=args.market_closure_authority,
        publish=args.publish,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
