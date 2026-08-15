"""Build causal native M1/M5 frames through the same feature owners.

One implementation runs at an explicit one-minute Exit clock or five-minute
Entry clock.  Each route consumes only its own native closed rows, then builds
closed MTF OHLCV before features.  It never combines M1/M5 values in front of
the owners or resamples computed M1 features upward.  The older
``EXPANDED_BASE34_CTX16CAT6`` artifact is intentionally not an input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import stat
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.contracts.entry_exit_feature_base_v1 import (  # noqa: E402
    ENTRY_DECISION_BAR_SECONDS,
    ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION,
    EXIT_DECISION_BAR_SECONDS,
    entry_exit_shared_feature_base_contract,
    require_entry_exit_shared_feature_base_contract,
)
from gx1.contracts.entry_exit_production_architecture_v1 import (  # noqa: E402
    current_entry_exit_architecture_observation,
    require_entry_exit_production_architecture,
)
from gx1.contracts.entry_model_native_signal_v1 import (  # noqa: E402
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
)
from gx1.contracts.xau_tape_provenance_v1 import (  # noqa: E402
    CANONICAL_NATIVE_REQUIRED_COLUMNS,
)
from gx1.contracts.gx1_scope_v1 import require_offline_scope  # noqa: E402
from gx1.execution.v12_ctx_augment_live import (  # noqa: E402
    augment_canonical_v3_model_agnostic_from_v4,
)
from gx1.features.htf_features import (  # noqa: E402
    HTF_V4_MATRIX_CONTRACT,
    MULTI_TF_PER_BAR_FEATURES_V4,
    MULTI_TF_RESAMPLE_RULES,
    attach_model_native_mtf_scalars_v4,
    bind_model_native_mtf_scalar_owner_v4,
    build_multi_tf_v4_closed_timestamp_indices,
    build_multi_tf_per_bar_features_v4,
    load_multi_tf_v4_cache,
    validate_causal_feature_matrix,
)
from gx1.time.session_detector import decision_availability  # noqa: E402
from gx1.scripts.augment_forward_outcome_v2 import (  # noqa: E402
    attach_group_a_ctx_columns_parallel,
)
from gx1.scripts.materialize_build_canonical_features_v2 import (  # noqa: E402
    build_canonical_v2,
)
from gx1.scripts.materialize_canonical_v3_augment import (  # noqa: E402
    add_cross_tf_momentum,
)
from gx1.scripts.prebuild_multi_tf_cache_v4 import (  # noqa: E402
    _rename_dir_noreplace,
    publish_multi_tf_v4_cache,
)


TIMEFRAME_SPECS = {
    "M1": {
        "duration": pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS),
        "seconds": EXIT_DECISION_BAR_SECONDS,
        "lineage_key": "m1",
        "label": "m1",
    },
    "M5": {
        "duration": pd.Timedelta(seconds=ENTRY_DECISION_BAR_SECONDS),
        "seconds": ENTRY_DECISION_BAR_SECONDS,
        "lineage_key": "m5",
        "label": "m5",
    },
}
RAW_COLUMNS = tuple(CANONICAL_NATIVE_REQUIRED_COLUMNS)
SOURCE_BATCH_ROWS = 32768
OUTPUT_COLUMNS = tuple(
    dict.fromkeys(
        (
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "atr",
            "m5h1_momentum",
            *MODEL_NATIVE_BASE_FIELDS,
            *MODEL_NATIVE_CTX_CONT_FIELDS,
            *MODEL_NATIVE_CTX_CAT_FIELDS,
        )
    )
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _fsync_file(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError(f"ENTRY_EXIT_ENRICHED_FSYNC_NOT_REGULAR: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_identity(path: Path) -> tuple[int, int]:
    observed = os.lstat(path)
    if not stat.S_ISREG(observed.st_mode):
        raise RuntimeError(f"ENTRY_EXIT_ENRICHED_ARTIFACT_NOT_REGULAR: {path}")
    return int(observed.st_dev), int(observed.st_ino)


def _directory_identity(path: Path) -> tuple[int, int]:
    observed = os.lstat(path)
    if not stat.S_ISDIR(observed.st_mode):
        raise RuntimeError(f"ENTRY_EXIT_ENRICHED_CACHE_NOT_DIRECTORY: {path}")
    return int(observed.st_dev), int(observed.st_ino)


def _unlink_if_owned(path: Path, identity: tuple[int, int]) -> None:
    try:
        if _file_identity(path) == identity:
            os.unlink(path)
    except FileNotFoundError:
        return


def _publish_file_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one immutable file without replacing a target."""

    try:
        os.link(source, destination, follow_symlinks=False)
    except FileExistsError as exc:
        raise RuntimeError(
            f"ENTRY_EXIT_ENRICHED_PUBLISH_TARGET_EXISTS: {destination}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"ENTRY_EXIT_ENRICHED_PUBLISH_FAILED: {destination}"
        ) from exc


def _new_absent_file_path(parent: Path, *, prefix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=prefix, dir=parent)
    os.close(descriptor)
    path = Path(raw_path)
    path.unlink()
    return path


def _new_absent_directory_path(parent: Path, *, prefix: str) -> Path:
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    path.rmdir()
    return path


def _write_frame_parquet_bounded(
    frame: pd.DataFrame,
    output: Path,
    *,
    columns: Sequence[str],
    index_name: str | None,
    chunk_rows: int,
) -> None:
    """Write one frame through bounded Arrow batches to an immutable path."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    names = tuple(str(name) for name in columns)
    if (
        frame.empty
        or isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, int)
        or chunk_rows <= 0
        or not names
        or len(names) != len(set(names))
    ):
        raise RuntimeError("ENTRY_EXIT_ENRICHED_BOUNDED_WRITE_INPUT_INVALID")
    output = Path(output)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise RuntimeError("ENTRY_EXIT_ENRICHED_BOUNDED_WRITE_TARGET_EXISTS")
    if index_name is None:
        if tuple(frame.columns) != names:
            raise RuntimeError("ENTRY_EXIT_ENRICHED_BOUNDED_WRITE_SCHEMA_INVALID")
    elif names[0] != index_name or any(
        list(frame.columns).count(name) != 1 for name in names[1:]
    ):
        raise RuntimeError("ENTRY_EXIT_ENRICHED_BOUNDED_WRITE_SCHEMA_INVALID")

    partial = _new_absent_file_path(
        output.parent,
        prefix=f".{output.name}.partial-",
    )
    writer: pq.ParquetWriter | None = None
    expected_schema: pa.Schema | None = None
    output_identity: tuple[int, int] | None = None
    try:
        for lo in range(0, len(frame), chunk_rows):
            hi = min(lo + chunk_rows, len(frame))
            if index_name is None:
                table = pa.Table.from_pandas(
                    frame.iloc[lo:hi].loc[:, list(names)],
                    preserve_index=False,
                )
            else:
                arrays = [pa.array(frame.index[lo:hi], from_pandas=True)]
                arrays.extend(
                    pa.array(frame[name].iloc[lo:hi], from_pandas=True)
                    for name in names[1:]
                )
                table = pa.Table.from_arrays(arrays, names=list(names))
            if expected_schema is None:
                expected_schema = table.schema
                writer = pq.ParquetWriter(
                    partial,
                    expected_schema,
                    compression="snappy",
                    use_dictionary=True,
                    write_statistics=True,
                )
            elif not table.schema.equals(expected_schema, check_metadata=False):
                raise RuntimeError(
                    "ENTRY_EXIT_ENRICHED_BOUNDED_WRITE_BATCH_SCHEMA_DRIFT"
                )
            assert writer is not None
            writer.write_table(table, row_group_size=hi - lo)
        if writer is None or expected_schema is None:
            raise RuntimeError("ENTRY_EXIT_ENRICHED_BOUNDED_WRITE_EMPTY")
        writer.close()
        writer = None
        _fsync_file(partial)
        _fsync_directory(partial.parent)
        partial_identity = _file_identity(partial)
        _publish_file_noreplace(partial, output)
        output_identity = partial_identity
        _fsync_directory(output.parent)

        parquet = pq.ParquetFile(output)
        if (
            parquet.metadata.num_rows != len(frame)
            or tuple(parquet.schema_arrow.names) != names
            or any(
                parquet.metadata.row_group(index).num_rows > chunk_rows
                for index in range(parquet.metadata.num_row_groups)
            )
        ):
            raise RuntimeError("ENTRY_EXIT_ENRICHED_BOUNDED_WRITE_PROOF_FAILED")
    except BaseException:
        if output_identity is not None:
            _unlink_if_owned(output, output_identity)
            _fsync_directory(output.parent)
        raise
    finally:
        if writer is not None:
            writer.close()
        if partial.exists() and not partial.is_symlink():
            partial.unlink()


def _write_output_parquet_bounded(
    frame: pd.DataFrame,
    output: Path,
    *,
    chunk_rows: int = 32768,
) -> None:
    """Write the ordered enriched surface without one full Arrow copy."""

    _write_frame_parquet_bounded(
        frame,
        output,
        columns=OUTPUT_COLUMNS,
        index_name="time",
        chunk_rows=chunk_rows,
    )


def _require_regular(path: Path, *, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"M1_ENRICHED_{label}_INVALID: {path}")
    return path


def _source_manifest_identity(root: Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"M1_ENRICHED_SOURCE_ROOT_INVALID: {root}")
    manifest_path = root / "MANIFEST.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError(f"M1_ENRICHED_SOURCE_MANIFEST_INVALID: {manifest_path}")
    parts = sorted(root.glob("year=*/part-*.parquet"))
    if not parts:
        raise RuntimeError(f"M1_ENRICHED_SOURCE_PARTS_MISSING: {root}")
    invalid = [
        path
        for path in parts
        if path.is_symlink() or path.parent.is_symlink() or not path.is_file()
    ]
    if invalid:
        raise RuntimeError(f"M1_ENRICHED_SOURCE_PART_INVALID: {invalid[0]}")
    return {
        "root": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "part_paths": [str(path) for path in parts],
        "part_sha256": {str(path): _sha256_file(path) for path in parts},
    }


def _materialize_native_source_bounded(
    root: Path,
    *,
    timeframe: str,
    output: Path,
    batch_rows: int = SOURCE_BATCH_ROWS,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    """Validate and spool ordered native parts without a read-all/concat copy."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    spec = TIMEFRAME_SPECS[timeframe]
    label = spec["label"].upper()
    if (
        isinstance(batch_rows, bool)
        or not isinstance(batch_rows, int)
        or batch_rows <= 0
        or batch_rows > SOURCE_BATCH_ROWS
    ):
        raise RuntimeError(f"{label}_ENRICHED_SOURCE_BATCH_ROWS_INVALID")
    identity = _source_manifest_identity(root)
    output = Path(output)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise RuntimeError(f"{label}_ENRICHED_SOURCE_STAGE_INVALID")

    schema = pa.schema(
        [pa.field("time", pa.timestamp("ns", tz="UTC"), nullable=False)]
        + [
            pa.field(name, pa.float64(), nullable=False)
            for name in RAW_COLUMNS[1:-1]
        ]
        + [pa.field("volume", pa.int64(), nullable=False)]
    )
    partial = _new_absent_file_path(
        output.parent,
        prefix=f".{output.name}.partial-",
    )
    writer: pq.ParquetWriter | None = None
    output_identity: tuple[int, int] | None = None
    rows = 0
    first_time: pd.Timestamp | None = None
    previous_time_ns: int | None = None
    max_observed_batch = 0
    try:
        writer = pq.ParquetWriter(
            partial,
            schema,
            compression="snappy",
            use_dictionary=True,
            write_statistics=True,
        )
        for raw_path in identity["part_paths"]:
            try:
                parquet = pq.ParquetFile(raw_path)
            except Exception as exc:
                raise RuntimeError(
                    f"{label}_ENRICHED_SOURCE_PARQUET_INVALID: {raw_path}"
                ) from exc
            if tuple(parquet.schema_arrow.names) != RAW_COLUMNS:
                raise RuntimeError(
                    f"{label}_ENRICHED_SOURCE_SCHEMA_INVALID: "
                    f"{raw_path} expected={RAW_COLUMNS} "
                    f"observed={tuple(parquet.schema_arrow.names)}"
                )
            part_rows = 0
            for batch in parquet.iter_batches(
                batch_size=batch_rows,
                columns=list(RAW_COLUMNS),
                use_threads=False,
            ):
                frame = batch.to_pandas()
                times = pd.DatetimeIndex(
                    pd.to_datetime(frame["time"], utc=True, errors="coerce")
                ).as_unit("ns")
                if times.isna().any():
                    raise RuntimeError(f"{label}_ENRICHED_SOURCE_TIME_INVALID")
                if (
                    times.has_duplicates
                    or not times.is_monotonic_increasing
                    or (
                        previous_time_ns is not None
                        and int(times.asi8[0]) <= previous_time_ns
                    )
                ):
                    raise RuntimeError(
                        f"{label}_ENRICHED_SOURCE_TIME_ORDER_INVALID"
                    )
                if np.any(times.asi8 % int(spec["duration"].value) != 0):
                    raise RuntimeError(
                        f"{label}_ENRICHED_SOURCE_OFF_{timeframe}_GRID"
                    )
                numeric = frame.loc[:, list(RAW_COLUMNS[1:])].apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                values = numeric.to_numpy(dtype=np.float64)
                if not np.isfinite(values).all():
                    raise RuntimeError(f"{label}_ENRICHED_SOURCE_NONFINITE")
                high = values[:, RAW_COLUMNS[1:].index("high")]
                low = values[:, RAW_COLUMNS[1:].index("low")]
                open_ = values[:, RAW_COLUMNS[1:].index("open")]
                close = values[:, RAW_COLUMNS[1:].index("close")]
                volume = numeric["volume"].to_numpy(dtype=np.float64)
                if (
                    np.any(open_ <= 0.0)
                    or np.any(low <= 0.0)
                    or np.any(high < low)
                    or np.any(high < open_)
                    or np.any(high < close)
                    or np.any(low > open_)
                    or np.any(low > close)
                    or np.any(volume <= 0.0)
                    or np.any(volume != volume.astype(np.int64))
                ):
                    raise RuntimeError(
                        f"{label}_ENRICHED_SOURCE_OHLCV_GEOMETRY_INVALID"
                    )
                arrays = [pa.array(times, type=schema.field("time").type)]
                arrays.extend(
                    pa.array(
                        numeric[name].to_numpy(dtype=np.float64),
                        type=pa.float64(),
                    )
                    for name in RAW_COLUMNS[1:-1]
                )
                arrays.append(pa.array(volume.astype(np.int64), type=pa.int64()))
                normalized = pa.Table.from_arrays(arrays, schema=schema)
                writer.write_table(normalized, row_group_size=len(frame))
                if first_time is None:
                    first_time = pd.Timestamp(times[0])
                previous_time_ns = int(times.asi8[-1])
                rows += len(frame)
                part_rows += len(frame)
                max_observed_batch = max(max_observed_batch, len(frame))
            if part_rows != int(parquet.metadata.num_rows):
                raise RuntimeError(
                    f"{label}_ENRICHED_SOURCE_PART_ROW_COUNT_MISMATCH"
                )
        if rows <= 0 or first_time is None or previous_time_ns is None:
            raise RuntimeError(f"{label}_ENRICHED_SOURCE_EMPTY")
        writer.close()
        writer = None
        _fsync_file(partial)
        _fsync_directory(partial.parent)
        partial_identity = _file_identity(partial)
        _publish_file_noreplace(partial, output)
        output_identity = partial_identity
        _fsync_directory(output.parent)
        observed = pq.ParquetFile(output)
        if (
            observed.metadata.num_rows != rows
            or not observed.schema_arrow.equals(schema, check_metadata=False)
            or any(
                observed.metadata.row_group(index).num_rows > batch_rows
                for index in range(observed.metadata.num_row_groups)
            )
        ):
            raise RuntimeError(f"{label}_ENRICHED_SOURCE_STAGE_PROOF_FAILED")
    except BaseException:
        if output_identity is not None:
            _unlink_if_owned(output, output_identity)
            _fsync_directory(output.parent)
        raise
    finally:
        if writer is not None:
            writer.close()
        if partial.exists() and not partial.is_symlink():
            partial.unlink()

    last_time = pd.Timestamp(previous_time_ns, unit="ns", tz="UTC")
    summary = {
        "row_count": rows,
        "time_min_utc": first_time.isoformat(),
        "time_max_utc": last_time.isoformat(),
    }
    bounded_io = {
        "configured_batch_rows": batch_rows,
        "maximum_batch_rows": SOURCE_BATCH_ROWS,
        "maximum_observed_batch_rows": max_observed_batch,
    }
    return identity, summary, bounded_io


def _checkpoint_key(
    *,
    source_identity: dict[str, Any],
    dataset_run_id: str,
    pair_generation_id: str,
    timeframe: str,
) -> str:
    spec = TIMEFRAME_SPECS[timeframe]
    return _canonical_sha256(
        {
            "schema_version": "gx1_entry_exit_enriched_frame_checkpoint_key_v1",
            "timeframe": timeframe,
            "source_manifest_sha256": source_identity["manifest_sha256"],
            "part_sha256": source_identity["part_sha256"],
            "dataset_run_id": dataset_run_id,
            "pair_generation_id": pair_generation_id,
            "base_bar_seconds": spec["seconds"],
            "shared_contract": entry_exit_shared_feature_base_contract(),
        }
    )


def _require_pair_binding(
    *,
    pair_manifest_path: Path,
    expected_pair_manifest_sha256: str,
    pair_generation_id: str,
    source_identity: dict[str, Any],
    native_summary: Mapping[str, Any],
    timeframe: str = "M1",
) -> dict[str, Any]:
    spec = TIMEFRAME_SPECS[timeframe]
    label = spec["label"].upper()
    expected_summary_keys = {
        "row_count",
        "time_min_utc",
        "time_max_utc",
    }
    if (
        not isinstance(native_summary, Mapping)
        or set(native_summary) != expected_summary_keys
        or isinstance(native_summary.get("row_count"), bool)
        or not isinstance(native_summary.get("row_count"), int)
        or int(native_summary["row_count"]) <= 0
        or not isinstance(native_summary.get("time_min_utc"), str)
        or not isinstance(native_summary.get("time_max_utc"), str)
    ):
        raise RuntimeError(f"{label}_ENRICHED_NATIVE_SUMMARY_INVALID")
    manifest_path = _require_regular(pair_manifest_path, label="PAIR_MANIFEST")
    # The chain validated this manifest at its pair-authority step and carries
    # the exact hash; the producer may not substitute its own observation for
    # the declared one (rule 2a: an explicit CLI input, never a self-measured
    # default).
    manifest_sha256 = _sha256_file(manifest_path)
    if (
        not isinstance(expected_pair_manifest_sha256, str)
        or len(expected_pair_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_pair_manifest_sha256
        )
        or manifest_sha256 != expected_pair_manifest_sha256
    ):
        raise RuntimeError(f"{label}_ENRICHED_PAIR_MANIFEST_SHA256_MISMATCH")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label}_ENRICHED_PAIR_MANIFEST_INVALID") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label}_ENRICHED_PAIR_MANIFEST_INVALID")
    if payload.get("pair_generation_id") != pair_generation_id:
        raise RuntimeError(f"{label}_ENRICHED_PAIR_GENERATION_ID_MANIFEST_MISMATCH")
    lineage = payload.get("lineage")
    native_sources = lineage.get("native_sources") if isinstance(lineage, dict) else None
    bound_source = (
        native_sources.get(spec["lineage_key"])
        if isinstance(native_sources, dict)
        else None
    )
    if not isinstance(bound_source, dict):
        raise RuntimeError(
            f"{label}_ENRICHED_PAIR_NATIVE_{spec['lineage_key'].upper()}_LINEAGE_MISSING"
        )
    expected = {
        "root": source_identity["root"],
        "manifest_path": source_identity["manifest_path"],
        "manifest_sha256": source_identity["manifest_sha256"],
        "row_count": int(native_summary["row_count"]),
        "time_min_utc": native_summary["time_min_utc"],
        "time_max_utc": native_summary["time_max_utc"],
    }
    observed = {name: bound_source.get(name) for name in expected}
    if observed != expected:
        raise RuntimeError(
            f"{label}_ENRICHED_PAIR_NATIVE_{spec['lineage_key'].upper()}_BINDING_MISMATCH: "
            f"observed={observed} expected={expected}"
        )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "pair_generation_id": pair_generation_id,
        f"native_{spec['lineage_key']}": expected,
    }


def _load_bound_m5_cache_context(
    *,
    cache_dir: Path,
    dataset_run_id: str,
    pair_generation_id: str,
) -> tuple[pd.DataFrame, dict, dict[str, Any]]:
    """Load the one verified V4 cache and its exact pair-bound M5 source."""

    cache_path = Path(cache_dir).expanduser().resolve()
    if cache_path.is_symlink() or not cache_path.is_dir():
        raise RuntimeError("M1_ENRICHED_MULTI_TF_CACHE_INVALID")
    multi_tf = load_multi_tf_v4_cache(cache_path)
    source = Path(str(multi_tf.m5_prebuilt_source)).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError("M1_ENRICHED_M5_CONTEXT_SOURCE_INVALID")
    source_sha256 = _sha256_file(source)
    if source_sha256 != str(multi_tf.m5_prebuilt_source_sha256):
        raise RuntimeError("M1_ENRICHED_M5_CONTEXT_SOURCE_HASH_MISMATCH")

    source_manifest = Path(f"{source}.manifest.json")
    if source_manifest.is_symlink() or not source_manifest.is_file():
        raise RuntimeError("M1_ENRICHED_M5_CONTEXT_MANIFEST_INVALID")
    try:
        payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("M1_ENRICHED_M5_CONTEXT_MANIFEST_INVALID") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("M1_ENRICHED_M5_CONTEXT_MANIFEST_INVALID")
    declared_payload_sha256 = payload.get("manifest_sha256")
    payload_without_hash = dict(payload)
    payload_without_hash.pop("manifest_sha256", None)
    if (
        payload.get("schema_version")
        != ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION
        or payload.get("decision") != "PASS"
        or payload.get("timeframe") != "M5"
        or payload.get("dataset_run_id") != dataset_run_id
        or payload.get("pair_generation_id") != pair_generation_id
        or Path(str(payload.get("output_parquet") or "")).expanduser().resolve()
        != source
        or payload.get("output_parquet_sha256") != source_sha256
        or declared_payload_sha256 != _canonical_sha256(payload_without_hash)
    ):
        raise RuntimeError("M1_ENRICHED_M5_CONTEXT_BINDING_MISMATCH")
    require_entry_exit_shared_feature_base_contract(
        payload.get("shared_feature_base_contract"),
        context="M1_ENRICHED_M5_CONTEXT",
    )

    context = pd.read_parquet(
        source,
        columns=["time", "open", "high", "low", "close", "volume"],
    )
    context["time"] = pd.to_datetime(context["time"], utc=True, errors="raise")
    context = context.set_index("time").sort_index(kind="mergesort")
    times = pd.DatetimeIndex(context.index).as_unit("ns")
    values = context[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    if (
        context.empty
        or times.has_duplicates
        or not times.is_monotonic_increasing
        or np.any(times.asi8 % (ENTRY_DECISION_BAR_SECONDS * 1_000_000_000) != 0)
        or not np.isfinite(values).all()
        or np.any(values[:, 4] <= 0.0)
    ):
        raise RuntimeError("M1_ENRICHED_M5_CONTEXT_GEOMETRY_INVALID")
    context.index = times
    bind_model_native_mtf_scalar_owner_v4(
        multi_tf,
        context[["open", "high", "low", "close", "volume"]],
    )
    return context, multi_tf, {
        "cache_dir": str(cache_path),
        "cache_manifest_path": str(cache_path / "manifest.json"),
        "cache_manifest_sha256": str(multi_tf.manifest_sha256),
        "cache_identity_sha256": str(multi_tf.cache_identity_sha256),
        "m5_context_source": str(source),
        "m5_context_source_sha256": source_sha256,
        "m5_context_manifest_path": str(source_manifest),
        "m5_context_manifest_sha256": _sha256_file(source_manifest),
    }


def _rss_gib() -> float:
    """Resident set size in GiB. Always valid wherever it is called."""

    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        return -1.0
    return -1.0


# The producer ceiling is 10 GiB and a cgroup kill leaves no traceback, no
# checkpoint and no partial output. Failing loudly just under it turns that
# silent death into a named error that says which step was holding the memory.
_RSS_CEILING_GIB = 9.0


def _log_rss(label: str, *, ceiling_gib: float = _RSS_CEILING_GIB) -> None:
    """Emit the resident set at a named producer step and fail before the kill.

    A cgroup kill leaves no traceback and no checkpoint, so the only way to
    locate the peak is to record it as the stage advances.
    """

    rss = _rss_gib()
    print(f"[m1_enriched_rss] {label} rss_gib={rss:.2f}", flush=True)
    if rss >= ceiling_gib:
        raise RuntimeError(
            f"M1_ENRICHED_RSS_CEILING_EXCEEDED: {label} rss_gib={rss:.2f} "
            f"ceiling_gib={ceiling_gib:.2f}"
        )


def _complete_v4_owned_context(
    canonical_holder: "list[pd.DataFrame] | pd.DataFrame",
    *,
    multi_tf: dict,
    decision_bar_duration: pd.Timedelta,
) -> pd.DataFrame:
    """Run the sole MTF owner before every dependent context transform.

    Accepts either a frame or a single-element holder list. The holder form
    transfers ownership: the element is popped so the caller no longer keeps the
    input frame alive while the successor frames are built.
    """

    if isinstance(canonical_holder, list):
        canonical = canonical_holder.pop()
    else:
        canonical = canonical_holder
    _log_rss("context_start")

    attach_model_native_mtf_scalars_v4(
        canonical,
        multi_tf=multi_tf,
        decision_bar_duration=decision_bar_duration,
    )
    _log_rss("after_mtf_scalars")
    out = augment_canonical_v3_model_agnostic_from_v4(
        canonical,
        base_bar_duration=decision_bar_duration,
    )
    # augment_canonical_v3_from_v4 and add_cross_tf_momentum each return a new
    # frame. On the native M1 clock one frame is ~2.7 GB, so holding the input
    # and both results at once exceeds the 10G producer ceiling. Release each
    # frame as soon as its successor exists; the caller hands ownership over in
    # a holder so no caller-side reference keeps the input alive.
    del canonical
    _log_rss("after_augment_v3")
    out = add_cross_tf_momentum(
        out,
        decision_bar_duration=decision_bar_duration,
    )
    _log_rss("after_cross_tf_momentum")
    if tuple(name for name in out.columns if name == "m5h1_momentum") != (
        "m5h1_momentum",
    ):
        raise RuntimeError("ENTRY_EXIT_V4_MOMENTUM_OWNER_INVALID")
    return out


def _slice_multi_tf_to_output_source(
    multi_tf: Mapping[str, pd.DataFrame],
    output_index: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """Retain exact V4 bytes for the emitted M5 source timestamp geometry."""

    expected_indices = build_multi_tf_v4_closed_timestamp_indices(output_index)
    sliced: dict[str, pd.DataFrame] = {}
    for timeframe in MULTI_TF_RESAMPLE_RULES:
        source = multi_tf[timeframe]
        expected = expected_indices[timeframe]
        positions = source.index.get_indexer(expected)
        source_values = np.asarray(source.attrs.get("feats_np"))
        if (
            len(expected) == 0
            or np.any(positions < 0)
            or source_values.dtype != np.dtype(np.float32)
            or source_values.shape
            != (len(source), len(MULTI_TF_PER_BAR_FEATURES_V4))
        ):
            raise RuntimeError(
                f"M5_ENRICHED_MULTI_TF_OUTPUT_SLICE_INVALID: {timeframe}"
            )
        values = np.ascontiguousarray(source_values[positions])
        frame = pd.DataFrame(
            values,
            index=expected,
            columns=MULTI_TF_PER_BAR_FEATURES_V4,
            copy=False,
        )
        frame_values = frame.to_numpy(dtype=np.float32, copy=False)
        warmup_rows = validate_causal_feature_matrix(
            frame_values,
            expected_width=len(MULTI_TF_PER_BAR_FEATURES_V4),
            context=f"M5_ENRICHED_MULTI_TF_OUTPUT_{timeframe}",
        )
        frame.attrs["ts_int64"] = np.ascontiguousarray(
            expected.asi8.astype(np.int64, copy=False)
        )
        frame.attrs["feats_np"] = frame_values
        frame.attrs["causal_warmup_rows"] = warmup_rows
        frame.attrs["htf_feature_contract"] = HTF_V4_MATRIX_CONTRACT
        sliced[timeframe] = frame
    return sliced


def _load_canonical_stage(path: Path, *, timeframe: str) -> pd.DataFrame:
    label = TIMEFRAME_SPECS[timeframe]["label"].upper()
    canonical = pd.read_parquet(path)
    if canonical.empty or tuple(name for name in canonical.columns if name == "time") != (
        "time",
    ):
        raise RuntimeError(f"{label}_ENRICHED_CANONICAL_STAGE_INVALID")
    canonical_time = pd.DatetimeIndex(
        pd.to_datetime(canonical.pop("time"), utc=True, errors="raise")
    ).as_unit("ns")
    if canonical_time.has_duplicates or not canonical_time.is_monotonic_increasing:
        raise RuntimeError(f"{label}_ENRICHED_CANONICAL_TIME_ORDER_INVALID")
    canonical.index = canonical_time
    return canonical


def _build_canonical_stage(
    *,
    native_stage: Path,
    canonical_stage: Path,
    timeframe: str,
    chunk_rows: int,
) -> dict[str, Any]:
    """Build canonical local owners in an isolated process, then release raw."""

    spec = TIMEFRAME_SPECS[timeframe]
    label = spec["label"].upper()
    raw = pd.read_parquet(native_stage)
    if raw.empty or tuple(raw.columns) != RAW_COLUMNS:
        raise RuntimeError(f"{label}_ENRICHED_NATIVE_STAGE_SCHEMA_INVALID")
    canonical = build_canonical_v2(
        raw,
        base_bar_duration=spec["duration"],
    )
    del raw
    if canonical.empty or tuple(name for name in canonical.columns if name == "time") != (
        "time",
    ):
        raise RuntimeError(f"{label}_ENRICHED_CANONICAL_STAGE_INVALID")
    columns = tuple(canonical.columns)
    _write_frame_parquet_bounded(
        canonical,
        canonical_stage,
        columns=columns,
        index_name=None,
        chunk_rows=chunk_rows,
    )
    return {
        "rows": int(len(canonical)),
        "columns": list(columns),
        "stage_sha256": _sha256_file(canonical_stage),
    }


def _finish_model_native_surface(
    enriched: pd.DataFrame,
    *,
    timeframe: str,
) -> pd.DataFrame:
    spec = TIMEFRAME_SPECS[timeframe]
    label = spec["label"].upper()
    decision_ts = decision_availability(
        enriched.index,
        bar_duration=spec["duration"],
        context=f"{label}_ENRICHED_TIME_FEATURES",
    )
    hour = decision_ts.hour.to_numpy(dtype=np.float32)
    dow = decision_ts.dayofweek.to_numpy(dtype=np.float32)
    enriched["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
    enriched["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)
    enriched["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0).astype(np.float32)
    enriched["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0).astype(np.float32)
    if not enriched.index.is_unique or not enriched.index.is_monotonic_increasing:
        raise RuntimeError(f"{label}_ENRICHED_OUTPUT_TIME_ORDER_INVALID")
    missing = [
        name
        for name in OUTPUT_COLUMNS
        if name != "time" and name not in enriched.columns
    ]
    if missing:
        raise RuntimeError(f"{label}_ENRICHED_OUTPUT_FIELDS_MISSING: {missing}")
    for name in OUTPUT_COLUMNS[1:]:
        values = pd.to_numeric(
            enriched[name], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{label}_ENRICHED_OUTPUT_NONFINITE: {name}")
    cats = enriched[list(MODEL_NATIVE_CTX_CAT_FIELDS)].to_numpy(dtype=np.float64)
    if not np.equal(cats, np.rint(cats)).all():
        raise RuntimeError(f"{label}_ENRICHED_OUTPUT_CTX_CAT_NONINTEGER")
    return enriched


def _build_enriched_stage(
    *,
    native_stage: Path,
    canonical_stage: Path,
    output_stage: Path,
    temporary_cache_dir: Path | None,
    source_cache_dir: Path,
    timeframe: str,
    checkpoint_dir: Path,
    checkpoint_key: str,
    checkpoint_chunk_rows: int,
    dataset_run_id: str,
    pair_generation_id: str,
    registry_fit_train_start: str | None = None,
    registry_fit_train_end: str | None = None,
    registry_fit_inner_end: str | None = None,
    registry_fit_source_provenance_by_clock: Mapping[str, Mapping[str, Any]] | None = None,
    volatility_squeeze_manifest: Path | None = None,
    expected_volatility_squeeze_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Complete MTF/context owners after the raw/canonical process has exited."""

    spec = TIMEFRAME_SPECS[timeframe]
    label = spec["label"].upper()
    if (
        volatility_squeeze_manifest is None
        or expected_volatility_squeeze_manifest_sha256 is None
    ):
        raise RuntimeError("ENTRY_EXIT_ENRICHED_VOLATILITY_SQUEEZE_ARTIFACT_REQUIRED")
    from gx1.features.volatility_squeeze_state_v1 import (
        load_volatility_squeeze_artifact_manifest,
    )

    squeeze_artifacts = load_volatility_squeeze_artifact_manifest(
        Path(volatility_squeeze_manifest).expanduser().resolve(strict=True),
        expected_sha256=expected_volatility_squeeze_manifest_sha256,
    )
    v29_registry_m1_lane_params: dict[str, Any] | None = None
    if timeframe == "M5":
        if temporary_cache_dir is None:
            raise RuntimeError("M5_ENRICHED_TEMPORARY_CACHE_MISSING")
        native_ohlcv = pd.read_parquet(
            native_stage,
            columns=["time", "open", "high", "low", "close", "volume"],
        )
        native_ohlcv["time"] = pd.to_datetime(
            native_ohlcv["time"], utc=True, errors="raise"
        )
        native_ohlcv = native_ohlcv.set_index("time")
        # The M5 enriched route is the first producer in the rebuild chain, so
        # the V29 registry constants are TRAIN-fitted here (rule 18: once, on
        # the declared chronological inner/outer TRAIN windows) and frozen
        # into the temporary cache manifest published below.
        if (
            registry_fit_train_start is None
            or registry_fit_train_end is None
            or registry_fit_inner_end is None
            or registry_fit_source_provenance_by_clock is None
        ):
            raise RuntimeError(
                "M5_ENRICHED_V29_REGISTRY_FIT_INPUTS_REQUIRED: "
                "immutable source lineage and chronological TRAIN split are required"
            )
        from gx1.contracts.entry_exit_production_architecture_v1 import (
            PRODUCTION_MTF_PER_TF_WINDOW_BARS,
        )
        from gx1.contracts.entry_model_native_signal_v1 import (
            MODEL_NATIVE_SEQ_LEN,
        )
        from gx1.features.htf_features import (
            fit_v29_registry_constants_from_m5,
        )

        v29_registry_constants = fit_v29_registry_constants_from_m5(
            native_ohlcv,
            declared_train_window_start=registry_fit_train_start,
            declared_train_window_end=registry_fit_train_end,
            declared_inner_fit_window_end=registry_fit_inner_end,
            source_provenance_by_clock=registry_fit_source_provenance_by_clock,
            per_tf_seq_lens=dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS),
            entry_m5_seq_len=MODEL_NATIVE_SEQ_LEN,
        )
        multi_tf = build_multi_tf_per_bar_features_v4(
            native_ohlcv,
            v29_registry_constants=v29_registry_constants,
            volatility_squeeze_artifacts=squeeze_artifacts,
        )
        context_m5 = native_ohlcv[["high", "low", "close"]].copy()
        del native_ohlcv
        multi_tf_binding: dict[str, Any] | None = None
    else:
        if temporary_cache_dir is not None:
            raise RuntimeError("M1_ENRICHED_TEMPORARY_CACHE_FORBIDDEN")
        # The M1 enriched route owns the Exit local lane's V29 registry fit
        # (rule 18: once, on the declared M1 chronological TRAIN split) and
        # freezes it into the hash-bound M1 manifest
        # published below — mirroring the M5 route's cache-manifest freeze.
        if (
            registry_fit_train_start is None
            or registry_fit_train_end is None
            or registry_fit_inner_end is None
            or registry_fit_source_provenance_by_clock is None
            or set(registry_fit_source_provenance_by_clock) != {"M1"}
        ):
            raise RuntimeError(
                "M1_ENRICHED_V29_REGISTRY_FIT_INPUTS_REQUIRED: "
                "immutable source lineage and chronological TRAIN split are required"
            )
        from gx1.contracts.entry_exit_feature_base_v1 import (
            EXIT_FEATURE_SEQUENCE_BARS,
        )
        from gx1.features.htf_features import (
            fit_v29_registry_m1_lane_params_from_m1,
        )

        native_m1_ohlcv = pd.read_parquet(
            native_stage,
            columns=["time", "open", "high", "low", "close", "volume"],
        )
        native_m1_ohlcv["time"] = pd.to_datetime(
            native_m1_ohlcv["time"], utc=True, errors="raise"
        )
        native_m1_ohlcv = native_m1_ohlcv.set_index("time")
        v29_registry_m1_lane_params = fit_v29_registry_m1_lane_params_from_m1(
            native_m1_ohlcv,
            declared_train_window_start=registry_fit_train_start,
            declared_train_window_end=registry_fit_train_end,
            declared_inner_fit_window_end=registry_fit_inner_end,
            source_provenance=registry_fit_source_provenance_by_clock["M1"],
            exit_m1_seq_len=EXIT_FEATURE_SEQUENCE_BARS,
        )
        del native_m1_ohlcv
        context_m5, multi_tf, multi_tf_binding = _load_bound_m5_cache_context(
            cache_dir=source_cache_dir,
            dataset_run_id=dataset_run_id,
            pair_generation_id=pair_generation_id,
        )
        if multi_tf.volatility_squeeze_artifacts.binding() != squeeze_artifacts.binding():
            raise RuntimeError("M1_ENRICHED_VOLATILITY_SQUEEZE_BINDING_MISMATCH")

    canonical = _complete_v4_owned_context(
        [_load_canonical_stage(canonical_stage, timeframe=timeframe)],
        multi_tf=multi_tf,
        decision_bar_duration=spec["duration"],
    )
    _log_rss("before_group_a_attach")
    enriched = attach_group_a_ctx_columns_parallel(
        canonical,
        multi_tf=multi_tf,
        journal_label=f"{spec['label']}_enriched_frame",
        workers=1,
        checkpoint_dir=checkpoint_dir,
        checkpoint_key=checkpoint_key,
        checkpoint_chunk_rows=checkpoint_chunk_rows,
        context_m5=context_m5,
        base_bar_duration=spec["duration"],
    )
    del canonical, context_m5
    _log_rss("after_group_a_attach")
    enriched = _finish_model_native_surface(enriched, timeframe=timeframe)
    _log_rss("after_finish_surface")
    _write_output_parquet_bounded(
        enriched,
        output_stage,
        chunk_rows=checkpoint_chunk_rows,
    )
    output_sha256 = _sha256_file(output_stage)

    if timeframe == "M5":
        cache_features = _slice_multi_tf_to_output_source(
            multi_tf,
            pd.DatetimeIndex(enriched.index).as_unit("ns"),
        )
        publish_multi_tf_v4_cache(
            out_dir=temporary_cache_dir,
            m5_prebuilt=output_stage,
            expected_source_sha256=output_sha256,
            features=cache_features,
            v29_registry_constants=v29_registry_constants,
            volatility_squeeze_artifacts=squeeze_artifacts,
        )
        verified = load_multi_tf_v4_cache(temporary_cache_dir)
        if (
            Path(verified.m5_prebuilt_source) != output_stage
            or verified.m5_prebuilt_source_sha256 != output_sha256
        ):
            raise RuntimeError("M5_ENRICHED_TEMPORARY_CACHE_BINDING_MISMATCH")
        del cache_features, verified
    if timeframe == "M1" and multi_tf_binding is None:
        raise RuntimeError(f"{label}_ENRICHED_MULTI_TF_BINDING_MISSING")
    if timeframe == "M1" and v29_registry_m1_lane_params is None:
        raise RuntimeError("M1_ENRICHED_V29_REGISTRY_M1_LANE_PARAMS_MISSING")
    del multi_tf
    return {
        "rows": int(len(enriched)),
        "output_sha256": output_sha256,
        "multi_tf_binding": multi_tf_binding,
        "v29_registry_m1_lane_params": v29_registry_m1_lane_params,
    }


def _write_stage_report(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(dict(payload), sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _isolated_stage_entrypoint(
    stage_name: str,
    kwargs: dict[str, Any],
    report_path: Path,
) -> None:
    try:
        if stage_name == "canonical":
            result = _build_canonical_stage(**kwargs)
        elif stage_name == "enriched":
            result = _build_enriched_stage(**kwargs)
        else:
            raise RuntimeError("ENTRY_EXIT_ENRICHED_STAGE_NAME_INVALID")
        _write_stage_report(
            report_path,
            {"decision": "PASS", "result": result},
        )
    except BaseException as exc:
        try:
            _write_stage_report(
                report_path,
                {
                    "decision": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=20),
                },
            )
        finally:
            raise SystemExit(1) from exc


def _run_isolated_stage(
    stage_name: str,
    *,
    kwargs: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    """Run one full-frame phase in a child so its resident set is reclaimed."""

    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("ENTRY_EXIT_ENRICHED_PROCESS_ISOLATION_UNAVAILABLE")
    if report_path.exists() or report_path.is_symlink():
        raise RuntimeError("ENTRY_EXIT_ENRICHED_STAGE_REPORT_EXISTS")
    process = mp.get_context("fork").Process(
        target=_isolated_stage_entrypoint,
        args=(stage_name, kwargs, report_path),
        name=f"gx1-{stage_name}-stage",
    )
    process.start()
    process.join()
    if report_path.is_symlink() or not report_path.is_file():
        raise RuntimeError(
            f"ENTRY_EXIT_ENRICHED_{stage_name.upper()}_STAGE_DIED: "
            f"exitcode={process.exitcode}"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"ENTRY_EXIT_ENRICHED_{stage_name.upper()}_STAGE_REPORT_INVALID"
        ) from exc
    if (
        process.exitcode != 0
        or not isinstance(report, dict)
        or report.get("decision") != "PASS"
        or not isinstance(report.get("result"), dict)
    ):
        raise RuntimeError(
            f"ENTRY_EXIT_ENRICHED_{stage_name.upper()}_STAGE_FAILED: "
            f"exitcode={process.exitcode} error={report.get('error')!r}"
        )
    return dict(report["result"])


def _write_manifest_stage(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != dict(payload):
        raise RuntimeError("ENTRY_EXIT_ENRICHED_MANIFEST_STAGE_VERIFY_FAILED")
    without_hash = dict(observed)
    declared_hash = without_hash.pop("manifest_sha256", None)
    if declared_hash != _canonical_sha256(without_hash):
        raise RuntimeError("ENTRY_EXIT_ENRICHED_MANIFEST_STAGE_HASH_MISMATCH")
    _fsync_directory(path.parent)
    return _sha256_file(path)


def _prepare_m5_cache_stage(
    *,
    temporary_cache_dir: Path,
    output_stage: Path,
    output_final: Path,
    cache_final: Path,
    output_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    """Rebind verified temporary V4 bytes to the final immutable source path."""

    if output_final.exists() or output_final.is_symlink():
        raise RuntimeError("M5_ENRICHED_OUTPUT_ALREADY_EXISTS")
    if cache_final.exists() or cache_final.is_symlink():
        raise RuntimeError("M5_ENRICHED_MULTI_TF_CACHE_OUTPUT_INVALID")
    output_identity = _file_identity(output_stage)
    cache_stage = _new_absent_directory_path(
        cache_final.parent,
        prefix=f".{cache_final.name}.prepared-",
    )
    output_linked = False
    try:
        _publish_file_noreplace(output_stage, output_final)
        output_linked = True
        _fsync_directory(output_final.parent)
        temporary_features = load_multi_tf_v4_cache(temporary_cache_dir)
        # The rebound final cache carries the exact TRAIN-fitted registry
        # constants already frozen in the verified temporary cache manifest
        # (one truth: fit happened once, upstream; this is a byte rebind,
        # never a refit).
        publish_multi_tf_v4_cache(
            out_dir=cache_stage,
            m5_prebuilt=output_final,
            expected_source_sha256=output_sha256,
            features=temporary_features,
            v29_registry_constants=temporary_features.v29_registry_constants,
            volatility_squeeze_artifacts=(
                temporary_features.volatility_squeeze_artifacts
            ),
        )
        verified = load_multi_tf_v4_cache(cache_stage)
        if (
            Path(verified.m5_prebuilt_source) != output_final
            or verified.m5_prebuilt_source_sha256 != output_sha256
            or verified.v29_registry_constants
            != temporary_features.v29_registry_constants
        ):
            raise RuntimeError("M5_ENRICHED_PREPARED_CACHE_BINDING_MISMATCH")
        del temporary_features
        binding = {
            "cache_dir": str(cache_final),
            "cache_manifest_path": str(cache_final / "manifest.json"),
            "cache_manifest_sha256": str(verified.manifest_sha256),
            "cache_identity_sha256": str(verified.cache_identity_sha256),
            "m5_context_source": str(output_final),
            "m5_context_source_sha256": output_sha256,
        }
        del verified
        return cache_stage, binding
    except BaseException:
        if cache_stage.exists() and not cache_stage.is_symlink():
            shutil.rmtree(cache_stage)
        raise
    finally:
        if output_linked:
            _unlink_if_owned(output_final, output_identity)
            _fsync_directory(output_final.parent)


def _publish_prepared_generation(
    *,
    output_stage: Path,
    output_final: Path,
    manifest_stage: Path,
    manifest_final: Path,
    cache_stage: Path | None = None,
    cache_final: Path | None = None,
) -> None:
    """Commit parquet/cache first and the validating sidecar strictly last."""

    if (cache_stage is None) != (cache_final is None):
        raise RuntimeError("ENTRY_EXIT_ENRICHED_CACHE_PUBLICATION_PAIR_INVALID")
    if (
        output_final.exists()
        or output_final.is_symlink()
        or manifest_final.exists()
        or manifest_final.is_symlink()
        or (cache_final is not None and (cache_final.exists() or cache_final.is_symlink()))
    ):
        raise RuntimeError("ENTRY_EXIT_ENRICHED_PUBLISH_TARGET_EXISTS")
    output_identity = _file_identity(output_stage)
    manifest_identity = _file_identity(manifest_stage)
    cache_identity = (
        _directory_identity(cache_stage) if cache_stage is not None else None
    )
    output_published = False
    cache_published = False
    manifest_published = False
    try:
        _publish_file_noreplace(output_stage, output_final)
        output_published = True
        _fsync_directory(output_final.parent)
        if cache_stage is not None and cache_final is not None:
            if cache_stage.parent != cache_final.parent:
                raise RuntimeError("ENTRY_EXIT_ENRICHED_CACHE_STAGE_PARENT_INVALID")
            _rename_dir_noreplace(cache_stage, cache_final)
            cache_published = True
            _fsync_directory(cache_final.parent)
        _publish_file_noreplace(manifest_stage, manifest_final)
        manifest_published = True
        _fsync_directory(manifest_final.parent)
        if (
            _file_identity(output_final) != output_identity
            or _file_identity(manifest_final) != manifest_identity
            or (
                cache_final is not None
                and _directory_identity(cache_final) != cache_identity
            )
        ):
            raise RuntimeError("ENTRY_EXIT_ENRICHED_PUBLISHED_IDENTITY_MISMATCH")
    except BaseException:
        if manifest_published:
            _unlink_if_owned(manifest_final, manifest_identity)
        if cache_published and cache_stage is not None and cache_final is not None:
            if _directory_identity(cache_final) == cache_identity:
                if cache_stage.exists() or cache_stage.is_symlink():
                    shutil.rmtree(cache_final)
                else:
                    os.rename(cache_final, cache_stage)
        if output_published:
            _unlink_if_owned(output_final, output_identity)
        for parent in {
            output_final.parent,
            manifest_final.parent,
            *(set() if cache_final is None else {cache_final.parent}),
        }:
            _fsync_directory(parent)
        raise


def _build_enriched_frame(
    *,
    native_root: Path,
    timeframe: str,
    pair_manifest_path: Path,
    expected_pair_manifest_sha256: str,
    multi_tf_cache_dir: Path,
    output_parquet: Path,
    manifest_path: Path,
    checkpoint_dir: Path,
    dataset_run_id: str,
    pair_generation_id: str,
    workers: int,
    checkpoint_chunk_rows: int = 4096,
    registry_fit_train_start: str | None = None,
    registry_fit_train_end: str | None = None,
    registry_fit_inner_end: str | None = None,
    registry_fit_tape_manifest: Path | None = None,
    expected_registry_fit_tape_manifest_sha256: str | None = None,
    volatility_squeeze_manifest: Path | None = None,
    expected_volatility_squeeze_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if timeframe not in TIMEFRAME_SPECS:
        raise RuntimeError(f"ENTRY_EXIT_ENRICHED_TIMEFRAME_INVALID: {timeframe}")
    require_entry_exit_production_architecture(
        current_entry_exit_architecture_observation(),
        context=f"ENTRY_EXIT_{timeframe}_ENRICHED_BUILD",
    )
    spec = TIMEFRAME_SPECS[timeframe]
    label = spec["label"].upper()
    require_offline_scope("featurebase_build")
    contract = require_entry_exit_shared_feature_base_contract(
        entry_exit_shared_feature_base_contract(),
        context=f"{label}_ENRICHED_PRODUCER",
    )
    if not isinstance(dataset_run_id, str) or not dataset_run_id:
        raise RuntimeError(f"{label}_ENRICHED_DATASET_RUN_ID_INVALID")
    if (
        not isinstance(pair_generation_id, str)
        or len(pair_generation_id) != 64
        or any(ch not in "0123456789abcdef" for ch in pair_generation_id)
    ):
        raise RuntimeError(f"{label}_ENRICHED_PAIR_GENERATION_ID_INVALID")
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers != 1
    ):
        raise RuntimeError(f"{label}_ENRICHED_WORKERS_MUST_EQUAL_ONE")
    if (
        isinstance(checkpoint_chunk_rows, bool)
        or not isinstance(checkpoint_chunk_rows, int)
        or checkpoint_chunk_rows <= 0
        or checkpoint_chunk_rows > SOURCE_BATCH_ROWS
    ):
        raise RuntimeError(f"{label}_ENRICHED_CHECKPOINT_CHUNK_ROWS_INVALID")
    output = Path(output_parquet).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    cache_dir = Path(multi_tf_cache_dir).expanduser().resolve()
    if (
        output.exists()
        or output.is_symlink()
        or manifest.exists()
        or manifest.is_symlink()
    ):
        raise RuntimeError(f"{label}_ENRICHED_OUTPUT_ALREADY_EXISTS")
    if manifest != Path(f"{output}.manifest.json"):
        raise RuntimeError(f"{label}_ENRICHED_MANIFEST_SIDECAR_PATH_INVALID")
    if timeframe == "M5":
        if (
            cache_dir.exists()
            or cache_dir.is_symlink()
            or not cache_dir.parent.is_dir()
        ):
            raise RuntimeError("M5_ENRICHED_MULTI_TF_CACHE_OUTPUT_INVALID")
    elif cache_dir.is_symlink() or not cache_dir.is_dir():
        raise RuntimeError("M1_ENRICHED_MULTI_TF_CACHE_INPUT_INVALID")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    if checkpoint.is_symlink():
        raise RuntimeError(f"{label}_ENRICHED_CHECKPOINT_SYMLINK")
    checkpoint.mkdir(parents=True, exist_ok=True)

    cache_stage: Path | None = None
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.build-",
        dir=output.parent,
    ) as raw_work_dir:
        work_dir = Path(raw_work_dir)
        native_stage = work_dir / "native.parquet"
        canonical_stage = work_dir / "canonical.parquet"
        output_stage = work_dir / "enriched.parquet"
        temporary_cache = (
            work_dir / "MULTI_TF_V4_CACHE"
            if timeframe == "M5"
            else None
        )
        source_identity, native_summary, source_bounded_io = (
            _materialize_native_source_bounded(
                native_root,
                timeframe=timeframe,
                output=native_stage,
            )
        )
        pair_binding = _require_pair_binding(
            pair_manifest_path=pair_manifest_path,
            expected_pair_manifest_sha256=expected_pair_manifest_sha256,
            pair_generation_id=pair_generation_id,
            source_identity=source_identity,
            native_summary=native_summary,
            timeframe=timeframe,
        )
        if (
            registry_fit_train_start is None
            or registry_fit_train_end is None
            or registry_fit_inner_end is None
            or registry_fit_tape_manifest is None
            or expected_registry_fit_tape_manifest_sha256 is None
        ):
            raise RuntimeError(
                f"{label}_ENRICHED_REGISTRY_FIT_LINEAGE_REQUIRED"
            )
        train_start = pd.Timestamp(registry_fit_train_start)
        inner_end = pd.Timestamp(registry_fit_inner_end)
        train_end = pd.Timestamp(registry_fit_train_end)
        if any(
            stamp.tzinfo is None or stamp.utcoffset() != pd.Timedelta(0)
            for stamp in (train_start, inner_end, train_end)
        ) or not train_start < inner_end < train_end:
            raise RuntimeError(f"{label}_ENRICHED_REGISTRY_FIT_SPLIT_INVALID")
        train_start_label = train_start.isoformat()
        inner_end_label = inner_end.isoformat()
        train_end_label = train_end.isoformat()
        tape_manifest = _require_regular(
            registry_fit_tape_manifest,
            label="REGISTRY_FIT_TAPE_MANIFEST",
        )
        tape_sha256 = _sha256_file(tape_manifest)
        if tape_sha256 != expected_registry_fit_tape_manifest_sha256:
            raise RuntimeError(
                f"{label}_ENRICHED_REGISTRY_FIT_TAPE_HASH_MISMATCH"
            )
        fit_clocks = ("M1",) if timeframe == "M1" else ("M5", "M15", "H1", "H4", "D1")
        registry_fit_source_provenance_by_clock = {
            clock: {
                "source_artifact": source_identity["manifest_path"],
                "source_sha256": source_identity["manifest_sha256"],
                "source_schema_version": "native_ohlcv_manifest_bound_frame_v1",
                "source_lane": clock,
                "tape_manifest_artifact": str(tape_manifest),
                "tape_manifest_sha256": tape_sha256,
                # The hash-bound pointer to the pair generation this fit read.
                # It was carried under the false name ``split_manifest_*``
                # until 2026-08-15; the binding is real (it is re-checked on
                # every V4 cache load), only the name was wrong.
                "pair_manifest_artifact": pair_binding["manifest_path"],
                "pair_manifest_sha256": pair_binding["manifest_sha256"],
                "train_split_id": f"{pair_generation_id}:TRAIN",
                "declared_train_window_start": train_start_label,
                "declared_train_window_end": train_end_label,
            }
            for clock in fit_clocks
        }
        checkpoint_key = _checkpoint_key(
            source_identity=source_identity,
            dataset_run_id=dataset_run_id,
            pair_generation_id=pair_generation_id,
            timeframe=timeframe,
        )

        canonical_result = _run_isolated_stage(
            "canonical",
            kwargs={
                "native_stage": native_stage,
                "canonical_stage": canonical_stage,
                "timeframe": timeframe,
                "chunk_rows": checkpoint_chunk_rows,
            },
            report_path=work_dir / "canonical-stage.json",
        )
        if canonical_result.get("rows") != native_summary["row_count"]:
            raise RuntimeError(f"{label}_ENRICHED_CANONICAL_ROW_COUNT_MISMATCH")

        enriched_result = _run_isolated_stage(
            "enriched",
            kwargs={
                "native_stage": native_stage,
                "canonical_stage": canonical_stage,
                "output_stage": output_stage,
                "temporary_cache_dir": temporary_cache,
                "source_cache_dir": cache_dir,
                "timeframe": timeframe,
                "checkpoint_dir": checkpoint,
                "checkpoint_key": checkpoint_key,
                "checkpoint_chunk_rows": checkpoint_chunk_rows,
                "dataset_run_id": dataset_run_id,
                "pair_generation_id": pair_generation_id,
                "registry_fit_train_start": train_start_label,
                "registry_fit_train_end": train_end_label,
                "registry_fit_inner_end": inner_end_label,
                "registry_fit_source_provenance_by_clock": (
                    registry_fit_source_provenance_by_clock
                ),
                "volatility_squeeze_manifest": volatility_squeeze_manifest,
                "expected_volatility_squeeze_manifest_sha256": (
                    expected_volatility_squeeze_manifest_sha256
                ),
            },
            report_path=work_dir / "enriched-stage.json",
        )
        rows = enriched_result.get("rows")
        output_sha256 = enriched_result.get("output_sha256")
        if (
            isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows <= 0
            or not isinstance(output_sha256, str)
            or output_sha256 != _sha256_file(output_stage)
        ):
            raise RuntimeError(f"{label}_ENRICHED_STAGE_RESULT_INVALID")

        if timeframe == "M5":
            assert temporary_cache is not None
            cache_stage, multi_tf_binding = _prepare_m5_cache_stage(
                temporary_cache_dir=temporary_cache,
                output_stage=output_stage,
                output_final=output,
                cache_final=cache_dir,
                output_sha256=output_sha256,
            )
        else:
            observed_binding = enriched_result.get("multi_tf_binding")
            if not isinstance(observed_binding, dict):
                raise RuntimeError("M1_ENRICHED_MULTI_TF_BINDING_MISSING")
            multi_tf_binding = observed_binding

        v29_registry_m1_lane_params: dict[str, Any] | None = None
        if timeframe == "M1":
            # Freeze the declared M1-lane TRAIN fit into the hash-bound M1
            # manifest (rule 2f provenance shape validated by the one owner).
            from gx1.features.htf_features import (
                require_v29_registry_m1_lane_params,
            )

            v29_registry_m1_lane_params = require_v29_registry_m1_lane_params(
                enriched_result.get("v29_registry_m1_lane_params")
            )

        result = {
            "schema_version": ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION,
            "decision": "PASS",
            "shared_feature_base_contract": contract,
            "dataset_run_id": dataset_run_id,
            "pair_generation_id": pair_generation_id,
            "timeframe": timeframe,
            "base_bar_seconds": spec["seconds"],
            f"native_{spec['lineage_key']}_source": source_identity,
            "pair_binding": pair_binding,
            "multi_tf_cache_binding": multi_tf_binding,
            **(
                {}
                if v29_registry_m1_lane_params is None
                else {
                    "v29_registry_m1_lane_params": v29_registry_m1_lane_params
                }
            ),
            "checkpoint_dir": str(checkpoint),
            "checkpoint_key": checkpoint_key,
            "output_parquet": str(output),
            "output_parquet_sha256": output_sha256,
            "rows": rows,
            "columns": list(OUTPUT_COLUMNS),
            "required_base_fields": list(MODEL_NATIVE_BASE_FIELDS),
            "required_context_cont_fields": list(MODEL_NATIVE_CTX_CONT_FIELDS),
            "required_context_cat_fields": list(MODEL_NATIVE_CTX_CAT_FIELDS),
            "bounded_io": {
                **source_bounded_io,
                "native_parts_read_sequentially": True,
                "pandas_read_all_concat": False,
                "canonical_stage_process_isolated": True,
                "enriched_stage_process_isolated": True,
                "raw_canonical_mtf_enriched_co_resident": False,
                "arrow_use_threads": False,
            },
            "causal_contract": {
                "future_rows_used": False,
                f"decision_uses_closed_{spec['label']}_bar": True,
                "same_feature_owner_as_entry": True,
                "same_specialist_stack_as_entry": True,
                "native_resolution_values": True,
                "cross_resolution_value_copy": False,
                "computed_m1_feature_resampling": False,
                "sole_mtf_feature_owner": "native_m5_v4",
                "legacy_basic_h1_h4_reached": False,
                "legacy_canonical_d1_m15_reached": False,
                "legacy_live_htf_augment_reached": False,
                "old_m1_artifacts_consumed": False,
                "missing_field_fill": False,
                "frame_relative_bucket_fallback": False,
            },
        }
        result["manifest_sha256"] = _canonical_sha256(result)
        manifest_stage = work_dir / "manifest.json"
        try:
            _write_manifest_stage(manifest_stage, result)
            _publish_prepared_generation(
                output_stage=output_stage,
                output_final=output,
                manifest_stage=manifest_stage,
                manifest_final=manifest,
                cache_stage=cache_stage,
                cache_final=cache_dir if timeframe == "M5" else None,
            )
        finally:
            if cache_stage is not None and cache_stage.exists():
                if cache_stage.is_symlink() or not cache_stage.is_dir():
                    raise RuntimeError("M5_ENRICHED_CACHE_STAGE_CLEANUP_INVALID")
                shutil.rmtree(cache_stage)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    route = parser.add_mutually_exclusive_group(required=True)
    route.add_argument("--native-m1-root", type=Path)
    route.add_argument("--native-m5-root", type=Path)
    parser.add_argument("--pair-manifest", required=True, type=Path)
    parser.add_argument(
        "--expected-pair-manifest-sha256",
        required=True,
        help=(
            "Exact lowercase SHA-256 of --pair-manifest as validated by the "
            "chain's pair-authority step; the registry TRAIN fit freezes this "
            "hash-bound pointer into its source provenance"
        ),
    )
    parser.add_argument("--multi-tf-cache-dir", required=True, type=Path)
    parser.add_argument("--output-parquet", required=True, type=Path)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--dataset-run-id", required=True)
    parser.add_argument("--pair-generation-id", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--checkpoint-chunk-rows", type=int, default=4096)
    parser.add_argument("--registry-fit-train-start", required=True)
    parser.add_argument(
        "--registry-fit-train-end",
        default=None,
        help=(
            "Declared TRAIN window end (UTC) for the V29 registry TRAIN "
            "fit (no default exists; required on both native routes)"
        ),
    )
    parser.add_argument(
        "--registry-fit-inner-end",
        required=True,
        help="Chronological inner-TRAIN fit/selection boundary (UTC)",
    )
    parser.add_argument(
        "--registry-fit-tape-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-registry-fit-tape-manifest-sha256",
        required=True,
    )
    parser.add_argument("--volatility-squeeze-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-volatility-squeeze-manifest-sha256",
        required=True,
    )
    args = parser.parse_args()
    timeframe = "M1" if args.native_m1_root is not None else "M5"
    if args.registry_fit_train_end is None:
        parser.error(
            "--registry-fit-train-end is required on both native routes"
        )
    native_root = args.native_m1_root or args.native_m5_root
    result = _build_enriched_frame(
        native_root=native_root,
        timeframe=timeframe,
        pair_manifest_path=args.pair_manifest,
        expected_pair_manifest_sha256=args.expected_pair_manifest_sha256,
        multi_tf_cache_dir=args.multi_tf_cache_dir,
        output_parquet=args.output_parquet,
        manifest_path=args.manifest_path,
        checkpoint_dir=args.checkpoint_dir,
        dataset_run_id=args.dataset_run_id,
        pair_generation_id=args.pair_generation_id,
        workers=args.workers,
        checkpoint_chunk_rows=args.checkpoint_chunk_rows,
        registry_fit_train_start=args.registry_fit_train_start,
        registry_fit_train_end=args.registry_fit_train_end,
        registry_fit_inner_end=args.registry_fit_inner_end,
        registry_fit_tape_manifest=args.registry_fit_tape_manifest,
        expected_registry_fit_tape_manifest_sha256=(
            args.expected_registry_fit_tape_manifest_sha256
        ),
        volatility_squeeze_manifest=args.volatility_squeeze_manifest,
        expected_volatility_squeeze_manifest_sha256=(
            args.expected_volatility_squeeze_manifest_sha256
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
