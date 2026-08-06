"""Materialize native M1/M5 surfaces through one feature-owner implementation.

This producer accepts an already enriched causal frame at one explicit native
clock. It reuses the exact model-native signal-layer owner used by the Entry
dataset builder; it never copies values across M1/M5, synthesizes missing
fields, fills absent context, or resamples computed M1 features upward. M1 also
receives the exact closed-M1 source timeline used by the Exit lifecycle.
"""
from __future__ import annotations

import argparse
import gc
import json
import mmap as mmap_module
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.contracts.entry_exit_feature_base_v1 import (  # noqa: E402
    ENTRY_DECISION_BAR_SECONDS,
    ENTRY_FEATURE_SEQUENCE_BARS,
    EXIT_DECISION_BAR_SECONDS,
    EXIT_FEATURE_SEQUENCE_BARS,
    require_entry_exit_enriched_source_binding,
)
from gx1.contracts.entry_exit_production_architecture_v1 import (  # noqa: E402
    current_entry_exit_architecture_observation,
    require_entry_exit_production_architecture,
)
from gx1.contracts.entry_model_native_signal_v1 import (  # noqa: E402
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CAT_MIN_MAX,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_SIGNAL_DIM,
    require_model_native_manifest,
)
from gx1.contracts.entry_exit_feature_surface_v1 import (  # noqa: E402
    ENTRY_EXIT_FEATURE_SURFACE_COLUMNS,
    build_entry_exit_feature_surface_manifest,
)
from gx1.contracts.gx1_scope_v1 import require_offline_scope  # noqa: E402
from gx1.features.entry_foundation_structure_v1 import (  # noqa: E402
    FOUNDATION_EVENT_AGE_CARRY_KEYS,
    foundation_event_age_carry_scope,
)


def _fixed_size_list_column(
    values: np.ndarray,
    width: int,
    *,
    dtype: pa.DataType,
) -> pa.Array:
    matrix = np.ascontiguousarray(values)
    if matrix.ndim != 2 or matrix.shape[1] != width:
        raise RuntimeError(
            "M1_FEATURE_BASE_FIXED_LIST_SHAPE_INVALID: "
            f"shape={matrix.shape} width={width}"
        )
    return pa.FixedSizeListArray.from_arrays(
        pa.array(matrix.reshape(-1), type=dtype),
        width,
    )


_BOUNDED_CAUSAL_OVERLAP_ROWS = 8
_BOUNDED_PARQUET_ROW_GROUPS_PER_BATCH = 64


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _admit_new_file(stage: Path, destination: Path) -> None:
    """Atomically admit one fsynced immutable file without overwriting."""

    with stage.open("rb") as handle:
        os.fsync(handle.fileno())
    try:
        os.link(stage, destination, follow_symlinks=False)
    except FileExistsError as exc:
        raise RuntimeError(
            f"ENTRY_EXIT_FEATURE_BASE_ADMISSION_TARGET_EXISTS: {destination}"
        ) from exc
    _fsync_directory(destination.parent)


def _validate_staged_feature_surface(
    path: Path,
    *,
    expected_times: pd.DatetimeIndex,
    batch_rows: int,
) -> None:
    """Stream-validate staged bytes without allocating another full matrix."""

    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise RuntimeError(
            "ENTRY_EXIT_FEATURE_BASE_STAGED_PARQUET_INVALID"
        ) from exc
    if (
        parquet.metadata is None
        or parquet.metadata.num_rows != len(expected_times)
        or tuple(parquet.schema_arrow.names)
        != ENTRY_EXIT_FEATURE_SURFACE_COLUMNS
    ):
        raise RuntimeError("ENTRY_EXIT_FEATURE_BASE_STAGED_SCHEMA_INVALID")

    expected_types = {
        "signal": pa.list_(pa.float32(), MODEL_NATIVE_SIGNAL_DIM),
        "ctx_cont": pa.list_(
            pa.float32(),
            len(MODEL_NATIVE_CTX_CONT_FIELDS),
        ),
        "ctx_cat": pa.list_(
            pa.int64(),
            len(MODEL_NATIVE_CTX_CAT_FIELDS),
        ),
    }
    for name, expected_type in expected_types.items():
        if parquet.schema_arrow.field(name).type != expected_type:
            raise RuntimeError(
                f"ENTRY_EXIT_FEATURE_BASE_STAGED_{name.upper()}_SCHEMA_INVALID"
            )

    offset = 0
    try:
        batches = parquet.iter_batches(
            batch_size=batch_rows,
            columns=list(ENTRY_EXIT_FEATURE_SURFACE_COLUMNS),
            use_threads=False,
        )
        for batch in batches:
            rows = int(batch.num_rows)
            if rows <= 0 or offset + rows > len(expected_times):
                raise RuntimeError(
                    "ENTRY_EXIT_FEATURE_BASE_STAGED_ROW_COUNT_INVALID"
                )
            observed_times = pd.DatetimeIndex(
                pd.to_datetime(
                    batch.column(0).to_pandas(),
                    utc=True,
                    errors="coerce",
                )
            ).as_unit("ns")
            if not observed_times.equals(expected_times[offset : offset + rows]):
                raise RuntimeError(
                    "ENTRY_EXIT_FEATURE_BASE_STAGED_TIME_MISMATCH"
                )

            for column_index, (name, expected_type) in enumerate(
                expected_types.items(),
                start=1,
            ):
                column = batch.column(column_index)
                width = expected_type.list_size
                if column.null_count or column.values.null_count:
                    raise RuntimeError(
                        f"ENTRY_EXIT_FEATURE_BASE_STAGED_{name.upper()}_NULL"
                    )
                values = np.asarray(
                    column.values.to_numpy(zero_copy_only=False)
                )
                if values.shape != (rows * width,):
                    raise RuntimeError(
                        f"ENTRY_EXIT_FEATURE_BASE_STAGED_{name.upper()}_WIDTH_INVALID"
                    )
                values = values.reshape(rows, width)
                if not np.isfinite(values).all():
                    raise RuntimeError(
                        f"ENTRY_EXIT_FEATURE_BASE_STAGED_{name.upper()}_NONFINITE"
                    )
                if name == "ctx_cat":
                    for index, (lower, upper) in enumerate(
                        MODEL_NATIVE_CTX_CAT_MIN_MAX.values()
                    ):
                        if np.any(values[:, index] < lower) or np.any(
                            values[:, index] > upper
                        ):
                            raise RuntimeError(
                                "ENTRY_EXIT_FEATURE_BASE_STAGED_CTX_CAT_DOMAIN_INVALID"
                            )
            offset += rows
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "ENTRY_EXIT_FEATURE_BASE_STAGED_DECODE_INVALID"
        ) from exc
    if offset != len(expected_times):
        raise RuntimeError("ENTRY_EXIT_FEATURE_BASE_STAGED_ROW_COUNT_INVALID")


def _drop_memmap_pages(values: np.ndarray) -> None:
    if not isinstance(values, np.memmap):
        return
    values.flush()
    handle = getattr(values, "_mmap", None)
    if handle is not None and hasattr(handle, "madvise"):
        handle.madvise(mmap_module.MADV_DONTNEED)


def _close_memmap(values: np.ndarray) -> None:
    if not isinstance(values, np.memmap):
        return
    _drop_memmap_pages(values)
    handle = getattr(values, "_mmap", None)
    if handle is not None:
        handle.close()


def _persist_float32_matrix(
    path: Path,
    values: np.ndarray,
) -> np.memmap:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise RuntimeError("M1_FEATURE_BASE_PRECOMPUTED_LAYER_INVALID")
    stored = np.memmap(
        path,
        dtype=np.float32,
        mode="w+",
        shape=matrix.shape,
    )
    stored[:] = matrix
    _drop_memmap_pages(stored)
    shape = stored.shape
    handle = getattr(stored, "_mmap", None)
    if handle is not None:
        handle.close()
    del stored
    return np.memmap(
        path,
        dtype=np.float32,
        mode="r",
        shape=shape,
    )


def _read_positioned_source_rows(
    parquet: pq.ParquetFile,
    *,
    row_group_offsets: np.ndarray,
    source_positions: np.ndarray,
    columns: list[str],
) -> pd.DataFrame:
    positions = np.asarray(source_positions, dtype=np.int64)
    if (
        positions.ndim != 1
        or positions.size == 0
        or np.any(positions < 0)
        or np.any(positions[1:] <= positions[:-1])
        or positions[-1] >= row_group_offsets[-1]
    ):
        raise RuntimeError("M1_FEATURE_BASE_SOURCE_POSITIONS_INVALID")
    groups = np.searchsorted(
        row_group_offsets[1:],
        positions,
        side="right",
    )
    tables: list[pa.Table] = []
    for group in np.unique(groups):
        group_mask = groups == group
        local_positions = positions[group_mask] - row_group_offsets[group]
        table = parquet.read_row_group(
            int(group),
            columns=columns,
            use_threads=False,
        )
        tables.append(
            table.take(pa.array(local_positions, type=pa.int64()))
        )
    selected = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    frame = selected.to_pandas()
    if len(frame) != len(positions):
        raise RuntimeError("M1_FEATURE_BASE_SOURCE_POSITION_ROW_COUNT_INVALID")
    return frame


def _build_bounded_extension_chunk(
    frame: pd.DataFrame,
    *,
    source_parquet: Path,
    requested_features: list[str],
    ctx_cont_names: list[str],
    ctx_cat_names: list[str],
    base_signal_fields: list[str],
    price_layer: np.ndarray,
    price_names: list[str],
    candle_layer: np.ndarray,
    candle_names: list[str],
    emit_offset: int,
    support_memory_state: dict[str, np.float32] | None,
    foundation_event_age_state: Mapping[str, object] | None,
    source_contract_label: str,
) -> tuple[
    np.ndarray,
    list[str],
    dict[str, Any],
    dict[str, np.float32],
    dict[str, int],
]:
    """Use the single model-native owner path on one bounded causal slice."""

    from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
        _build_inline_seq_structure_extension,
    )

    with foundation_event_age_carry_scope(
        foundation_event_age_state,
        tail_replay_rows=min(
            _BOUNDED_CAUSAL_OVERLAP_ROWS,
            len(frame),
        ),
    ) as event_age_scope:
        result = _build_inline_seq_structure_extension(
            frame,
            requested_features=requested_features,
            ctx_cont_names=ctx_cont_names,
            ctx_cat_names=ctx_cat_names,
            source_parquet=source_parquet,
            source_contract_label=source_contract_label,
            base_signal_fields=base_signal_fields,
            precomputed_price_layer=(price_layer, price_names),
            precomputed_candle_layer=(candle_layer, candle_names),
            emit_offset=emit_offset,
            support_memory_state=support_memory_state,
            return_support_memory_state=True,
        )
        if tuple(event_age_scope.next_state) != FOUNDATION_EVENT_AGE_CARRY_KEYS:
            raise RuntimeError("FOUNDATION_EVENT_AGE_CARRY_NOT_CAPTURED")
        next_event_age_state = dict(event_age_scope.next_state)
    return (*result, next_event_age_state)


def _materialize_bounded_feature_surface(
    *,
    source: Path,
    alignment: Path | None,
    output: Path,
    contract: dict[str, Any],
    timeframe: str,
    bar_seconds: int,
    sequence_bars: int,
) -> dict[str, Any]:
    """Write one native surface without a full source/513-wide RAM frame."""

    if (timeframe == "M1") != (alignment is not None):
        raise RuntimeError("ENTRY_EXIT_FEATURE_BASE_NATIVE_ROUTE_INVALID")

    from gx1.features.entry_model_native_feature_layers_v1 import (
        PRICE_DERIVED_FEATURE_NAMES,
        build_candlestick_derived_layer,
        build_price_derived_layer,
    )
    from gx1.features.entry_candlestick_patterns_v1 import (
        CANDLESTICK_PATTERN_FEATURE_NAMES,
    )
    from gx1.features.entry_support_resistance_memory_v1 import (
        SUPPORT_RESISTANCE_MEMORY_STATE_KEYS,
    )

    source_parquet = pq.ParquetFile(source)
    source_columns = tuple(source_parquet.schema_arrow.names)
    required = {
        "time",
        "open",
        "high",
        "low",
        "close",
        "atr",
        *MODEL_NATIVE_BASE_FIELDS,
        *MODEL_NATIVE_CTX_CONT_FIELDS,
        *MODEL_NATIVE_CTX_CAT_FIELDS,
    }
    missing = sorted(required - set(source_columns))
    if missing:
        raise RuntimeError(
            "M1_FEATURE_BASE_ENRICHED_CAUSAL_FRAME_FIELDS_MISSING: "
            f"{missing[:30]}"
        )
    source_time_frame = pd.read_parquet(source, columns=["time"])
    source_times = pd.DatetimeIndex(
        pd.to_datetime(
            source_time_frame["time"],
            utc=True,
            errors="coerce",
        )
    ).as_unit("ns")
    if (
        len(source_times) == 0
        or source_times.hasnans
        or not source_times.is_unique
        or not source_times.is_monotonic_increasing
        or not source_times.floor(f"{bar_seconds}s").equals(source_times)
    ):
        raise RuntimeError("M1_FEATURE_BASE_TIME_GEOMETRY_INVALID")
    del source_time_frame

    pre_source_rows = 0
    if alignment is None:
        surface_times = source_times
        positions = np.arange(len(source_times), dtype=np.int64)
    else:
        alignment_time_frame = pd.read_parquet(alignment, columns=["time"])
        surface_times = pd.DatetimeIndex(
            pd.to_datetime(
                alignment_time_frame["time"],
                utc=True,
                errors="coerce",
            )
        ).as_unit("ns")
        if (
            len(surface_times) == 0
            or surface_times.hasnans
            or not surface_times.is_unique
            or not surface_times.is_monotonic_increasing
            or not surface_times.floor(f"{bar_seconds}s").equals(
                surface_times
            )
        ):
            raise RuntimeError(
                "M1_FEATURE_BASE_ALIGNMENT_TIME_GEOMETRY_INVALID"
            )
        del alignment_time_frame
        # Every emitted M1 row must carry a complete higher-timeframe context,
        # and the widest of those is the 252-bar D1 window - about 353 calendar
        # days, built by resampling the native M5 tape. The first M1 row that can
        # own a full D1 window therefore lands roughly a year after the M5 source
        # begins. Alignment rows before that point cannot be produced with valid
        # D1 context at all; a pair generation that predates this rule asks for
        # them anyway. Those leading rows are excluded, and the subset
        # requirement still holds exactly over the span the surface covers, so a
        # gap INSIDE the window is still a hard failure.
        covered = surface_times >= source_times[0]
        pre_source_rows = int(np.count_nonzero(~covered))  # noqa: F841 - manifest
        if pre_source_rows:
            if not bool(covered[pre_source_rows:].all()):
                raise RuntimeError(
                    "M1_FEATURE_BASE_ALIGNMENT_PRE_SOURCE_ROWS_NOT_LEADING"
                )
            surface_times = surface_times[covered]
            if len(surface_times) == 0:
                raise RuntimeError(
                    "M1_FEATURE_BASE_ALIGNMENT_NO_ROWS_WITHIN_SOURCE_SPAN"
                )
        positions = source_times.get_indexer(surface_times)
    if np.any(positions < 0):
        raise RuntimeError(
            "M1_FEATURE_BASE_ALIGNMENT_TIME_NOT_SUBSET: "
            f"missing_rows={int(np.count_nonzero(positions < 0))}"
        )
    if np.any(positions[1:] <= positions[:-1]):
        raise RuntimeError("M1_FEATURE_BASE_ALIGNMENT_POSITION_ORDER_INVALID")

    row_group_offsets = np.zeros(
        source_parquet.metadata.num_row_groups + 1,
        dtype=np.int64,
    )
    for index in range(source_parquet.metadata.num_row_groups):
        row_group_offsets[index + 1] = (
            row_group_offsets[index]
            + source_parquet.metadata.row_group(index).num_rows
        )
    if row_group_offsets[-1] != len(source_times):
        raise RuntimeError("M1_FEATURE_BASE_SOURCE_METADATA_ROW_COUNT_INVALID")

    read_columns = list(
        dict.fromkeys(
            [
                "time",
                *MODEL_NATIVE_BASE_FIELDS,
                *MODEL_NATIVE_CTX_CONT_FIELDS,
                *MODEL_NATIVE_CTX_CAT_FIELDS,
            ]
        )
    )
    requested_features = list(contract["selected_fields"])
    expected_fields = tuple(contract["fields"])
    batch_rows = sequence_bars * _BOUNDED_PARQUET_ROW_GROUPS_PER_BATCH
    source_contract_label = f"causal_enriched_{timeframe.lower()}_frame_v1"

    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.generation.",
        dir=str(output.parent),
    ) as temporary_storage:
        storage = Path(temporary_storage)
        sample_times = pd.DataFrame({"time": surface_times})
        price_values, price_names = build_price_derived_layer(
            sample_times,
            source,
        )
        if tuple(price_names) != tuple(PRICE_DERIVED_FEATURE_NAMES):
            raise RuntimeError("M1_FEATURE_BASE_PRICE_LAYER_ORDER_INVALID")
        price_matrix = _persist_float32_matrix(
            storage / "price_layer.mmap",
            price_values,
        )
        del price_values
        gc.collect()

        candle_values, candle_names = build_candlestick_derived_layer(
            sample_times,
            source,
        )
        if tuple(candle_names) != tuple(CANDLESTICK_PATTERN_FEATURE_NAMES):
            raise RuntimeError("M1_FEATURE_BASE_CANDLE_LAYER_ORDER_INVALID")
        candle_matrix = _persist_float32_matrix(
            storage / "candle_layer.mmap",
            candle_values,
        )
        del candle_values, sample_times
        gc.collect()

        partial_output = storage / output.name
        writer: pq.ParquetWriter | None = None
        emitted_rows = 0
        extension_meta: dict[str, Any] | None = None
        extension_names: list[str] | None = None
        support_state: dict[str, np.float32] | None = None
        foundation_event_age_state: dict[str, int] | None = None
        try:
            for start in range(0, len(surface_times), batch_rows):
                stop = min(start + batch_rows, len(surface_times))
                prefix = max(
                    0,
                    start - _BOUNDED_CAUSAL_OVERLAP_ROWS,
                )
                emit_offset = start - prefix
                frame = _read_positioned_source_rows(
                    source_parquet,
                    row_group_offsets=row_group_offsets,
                    source_positions=positions[prefix:stop],
                    columns=read_columns,
                )
                frame_times = pd.DatetimeIndex(
                    pd.to_datetime(
                        frame["time"],
                        utc=True,
                        errors="coerce",
                    )
                ).as_unit("ns")
                expected_times = surface_times[prefix:stop]
                if not frame_times.equals(expected_times):
                    raise RuntimeError(
                        "M1_FEATURE_BASE_BATCH_TIME_ALIGNMENT_INVALID"
                    )
                for name in (
                    *MODEL_NATIVE_BASE_FIELDS,
                    *MODEL_NATIVE_CTX_CONT_FIELDS,
                ):
                    values = pd.to_numeric(
                        frame[name],
                        errors="coerce",
                    ).to_numpy(dtype=np.float32)
                    if not np.isfinite(values).all():
                        raise RuntimeError(
                            f"M1_FEATURE_BASE_NONFINITE_FIELD: {name}"
                        )
                raw_ctx_cat = frame[
                    list(MODEL_NATIVE_CTX_CAT_FIELDS)
                ].apply(
                    pd.to_numeric,
                    errors="coerce",
                ).to_numpy(dtype=np.float64)
                if (
                    not np.isfinite(raw_ctx_cat).all()
                    or not np.equal(raw_ctx_cat, np.rint(raw_ctx_cat)).all()
                ):
                    raise RuntimeError("M1_FEATURE_BASE_CTX_CAT_NONINTEGER")

                (
                    extension,
                    names,
                    meta,
                    support_state,
                    foundation_event_age_state,
                ) = _build_bounded_extension_chunk(
                    frame,
                    source_parquet=source,
                    requested_features=requested_features,
                    ctx_cont_names=list(MODEL_NATIVE_CTX_CONT_FIELDS),
                    ctx_cat_names=list(MODEL_NATIVE_CTX_CAT_FIELDS),
                    base_signal_fields=list(MODEL_NATIVE_BASE_FIELDS),
                    price_layer=price_matrix[prefix:stop],
                    price_names=list(price_names),
                    candle_layer=candle_matrix[prefix:stop],
                    candle_names=list(candle_names),
                    emit_offset=emit_offset,
                    support_memory_state=support_state,
                    foundation_event_age_state=foundation_event_age_state,
                    source_contract_label=source_contract_label,
                )
                if set(support_state) != set(
                    SUPPORT_RESISTANCE_MEMORY_STATE_KEYS
                ):
                    raise RuntimeError(
                        "M1_FEATURE_BASE_SUPPORT_STATE_SCHEMA_INVALID"
                    )
                if tuple(foundation_event_age_state) != FOUNDATION_EVENT_AGE_CARRY_KEYS:
                    raise RuntimeError(
                        "M1_FEATURE_BASE_FOUNDATION_EVENT_AGE_STATE_SCHEMA_INVALID"
                    )
                if extension_names is None:
                    extension_names = names
                    extension_meta = meta
                elif names != extension_names or meta != extension_meta:
                    raise RuntimeError(
                        "M1_FEATURE_BASE_BATCH_EXTENSION_CONTRACT_DRIFT"
                    )
                if tuple([*MODEL_NATIVE_BASE_FIELDS, *names]) != expected_fields:
                    raise RuntimeError(
                        "M1_FEATURE_BASE_SIGNAL_FIELD_ORDER_MISMATCH"
                    )

                emitted = frame.iloc[emit_offset:]
                signal = np.empty(
                    (stop - start, MODEL_NATIVE_SIGNAL_DIM),
                    dtype=np.float32,
                )
                for column, name in enumerate(MODEL_NATIVE_BASE_FIELDS):
                    signal[:, column] = emitted[name].to_numpy(
                        dtype=np.float32,
                        copy=False,
                    )
                signal[:, len(MODEL_NATIVE_BASE_FIELDS) :] = extension
                if signal.shape != (
                    stop - start,
                    MODEL_NATIVE_SIGNAL_DIM,
                ) or not np.isfinite(signal).all():
                    raise RuntimeError("M1_FEATURE_BASE_SIGNAL_SHAPE_INVALID")
                ctx_cont = emitted[
                    list(MODEL_NATIVE_CTX_CONT_FIELDS)
                ].astype(np.float32).to_numpy()
                ctx_cat = raw_ctx_cat[emit_offset:].astype(np.int64)
                output_table = pa.Table.from_arrays(
                    [
                        pa.array(surface_times[start:stop]),
                        _fixed_size_list_column(
                            signal,
                            MODEL_NATIVE_SIGNAL_DIM,
                            dtype=pa.float32(),
                        ),
                        _fixed_size_list_column(
                            ctx_cont,
                            len(MODEL_NATIVE_CTX_CONT_FIELDS),
                            dtype=pa.float32(),
                        ),
                        _fixed_size_list_column(
                            ctx_cat,
                            len(MODEL_NATIVE_CTX_CAT_FIELDS),
                            dtype=pa.int64(),
                        ),
                    ],
                    names=list(ENTRY_EXIT_FEATURE_SURFACE_COLUMNS),
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        partial_output,
                        output_table.schema,
                        compression="snappy",
                        use_dictionary=False,
                    )
                writer.write_table(
                    output_table,
                    row_group_size=sequence_bars,
                )
                emitted_rows += stop - start
                del (
                    output_table,
                    ctx_cat,
                    ctx_cont,
                    signal,
                    extension,
                    emitted,
                    raw_ctx_cat,
                    frame,
                )
                gc.collect()
                _drop_memmap_pages(price_matrix)
                _drop_memmap_pages(candle_matrix)
        finally:
            if writer is not None:
                writer.close()
            _close_memmap(price_matrix)
            _close_memmap(candle_matrix)

        if (
            emitted_rows != len(surface_times)
            or extension_meta is None
            or extension_names is None
            or not partial_output.is_file()
        ):
            raise RuntimeError("M1_FEATURE_BASE_BOUNDED_WRITE_INCOMPLETE")
        _validate_staged_feature_surface(
            partial_output,
            expected_times=surface_times,
            batch_rows=sequence_bars,
        )
        _admit_new_file(partial_output, output)
        return {
            "rows": emitted_rows,
            "alignment_rows_before_source_span": pre_source_rows,
            "first_surface_row_utc": str(surface_times[0]),
            "extension": extension_meta,
            "materialization": {
                "mode": (
                    f"bounded_native_{timeframe.lower()}_owner_batches_"
                    "v2_event_age_carry"
                ),
                "batch_rows": batch_rows,
                "causal_overlap_rows": _BOUNDED_CAUSAL_OVERLAP_ROWS,
                "recursive_state_fields": [
                    *SUPPORT_RESISTANCE_MEMORY_STATE_KEYS,
                    *(
                        f"foundation_event_age.{name}"
                        for name in FOUNDATION_EVENT_AGE_CARRY_KEYS
                    ),
                ],
                "full_signal_matrix_in_ram": False,
                "full_extension_matrix_in_ram": False,
            },
        }


def _publish_feature_base_manifest(
    *,
    timeframe: str,
    dataset_run_id: str,
    pair_generation_id: str,
    source: Path,
    source_binding: dict[str, Any],
    alignment: Path | None,
    seq_structure_manifest: Path,
    output: Path,
    rows: int,
    contract: dict[str, Any],
    extension_meta: dict[str, Any],
    materialization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if (
        not output.is_file()
        or output.is_symlink()
        or manifest_path.exists()
        or manifest_path.is_symlink()
    ):
        raise RuntimeError(
            "ENTRY_EXIT_FEATURE_BASE_MANIFEST_ADMISSION_STATE_INVALID"
        )
    manifest = build_entry_exit_feature_surface_manifest(
        timeframe=timeframe,
        dataset_run_id=dataset_run_id,
        pair_generation_id=pair_generation_id,
        source=source,
        source_binding=source_binding,
        alignment=alignment,
        seq_structure_manifest=seq_structure_manifest,
        output=output,
        rows=rows,
        signal_contract=contract,
        extension=extension_meta,
        materialization=materialization,
    )
    encoded = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.generation.",
        dir=str(output.parent),
    ) as publish_storage:
        partial = Path(publish_storage) / manifest_path.name
        with partial.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            staged_manifest = json.loads(partial.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "ENTRY_EXIT_FEATURE_BASE_STAGED_MANIFEST_INVALID"
            ) from exc
        if staged_manifest != manifest:
            raise RuntimeError(
                "ENTRY_EXIT_FEATURE_BASE_STAGED_MANIFEST_MISMATCH"
            )
        _admit_new_file(partial, manifest_path)
    return manifest


def _materialize_feature_base(
    *,
    source_parquet: Path,
    alignment_parquet: Path | None,
    seq_structure_manifest: Path,
    output_parquet: Path,
    dataset_run_id: str,
    pair_generation_id: str,
    timeframe: str,
) -> dict[str, Any]:
    if timeframe not in {"M1", "M5"}:
        raise RuntimeError("ENTRY_EXIT_FEATURE_BASE_TIMEFRAME_INVALID")
    if (timeframe == "M1") != (alignment_parquet is not None):
        raise RuntimeError("ENTRY_EXIT_FEATURE_BASE_NATIVE_ROUTE_INVALID")
    require_entry_exit_production_architecture(
        current_entry_exit_architecture_observation(),
        context=f"ENTRY_EXIT_{timeframe}_FEATURE_BASE_BUILD",
    )
    bar_seconds = (
        EXIT_DECISION_BAR_SECONDS if timeframe == "M1" else ENTRY_DECISION_BAR_SECONDS
    )
    sequence_bars = (
        EXIT_FEATURE_SEQUENCE_BARS
        if timeframe == "M1"
        else ENTRY_FEATURE_SEQUENCE_BARS
    )
    require_offline_scope("featurebase_build")
    source_input = Path(source_parquet).expanduser()
    alignment_input = (
        None
        if alignment_parquet is None
        else Path(alignment_parquet).expanduser()
    )
    manifest_input = Path(seq_structure_manifest).expanduser()
    output_input = Path(output_parquet).expanduser()
    if (
        source_input.is_symlink()
        or (alignment_input is not None and alignment_input.is_symlink())
        or manifest_input.is_symlink()
        or output_input.is_symlink()
    ):
        raise RuntimeError("M1_FEATURE_BASE_INPUT_OR_OUTPUT_INVALID")
    source = source_input.resolve()
    alignment = None if alignment_input is None else alignment_input.resolve()
    manifest_path = manifest_input.resolve()
    output = output_input.resolve()
    output_manifest = output.with_suffix(output.suffix + ".manifest.json")
    if (
        not source.is_file()
        or (alignment is not None and not alignment.is_file())
        or not manifest_path.is_file()
        or output.exists()
        or output.is_symlink()
        or output_manifest.exists()
        or output_manifest.is_symlink()
        or not output.parent.is_dir()
    ):
        raise RuntimeError("M1_FEATURE_BASE_INPUT_OR_OUTPUT_INVALID")
    if not isinstance(dataset_run_id, str) or not dataset_run_id:
        raise RuntimeError("M1_FEATURE_BASE_DATASET_RUN_ID_INVALID")
    if not isinstance(pair_generation_id, str) or not pair_generation_id:
        raise RuntimeError("M1_FEATURE_BASE_PAIR_GENERATION_ID_INVALID")

    source_binding = require_entry_exit_enriched_source_binding(
        source,
        dataset_run_id=dataset_run_id,
        pair_generation_id=pair_generation_id,
        timeframe=timeframe,
        context="ENTRY_EXIT_FEATURE_BASE",
    )

    contract = require_model_native_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        context="M1_FEATURE_BASE",
    )
    if tuple(contract["base_fields"]) != MODEL_NATIVE_BASE_FIELDS:
        raise RuntimeError("M1_FEATURE_BASE_BASE_FIELD_ORDER_INVALID")
    if int(contract["seq_input_dim"]) != MODEL_NATIVE_SIGNAL_DIM:
        raise RuntimeError("M1_FEATURE_BASE_SIGNAL_DIM_INVALID")

    bounded = _materialize_bounded_feature_surface(
        source=source,
        alignment=alignment,
        output=output,
        contract=contract,
        timeframe=timeframe,
        bar_seconds=bar_seconds,
        sequence_bars=sequence_bars,
    )
    return _publish_feature_base_manifest(
        timeframe=timeframe,
        dataset_run_id=dataset_run_id,
        pair_generation_id=pair_generation_id,
        source=source,
        source_binding=source_binding,
        alignment=alignment,
        seq_structure_manifest=manifest_path,
        output=output,
        rows=int(bounded["rows"]),
        contract=contract,
        extension_meta=dict(bounded["extension"]),
        materialization=(
            dict(bounded["materialization"])
            if timeframe == "M1"
            else None
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", required=True, type=Path)
    parser.add_argument("--alignment-parquet", type=Path)
    parser.add_argument("--seq-structure-manifest", required=True, type=Path)
    parser.add_argument("--output-parquet", required=True, type=Path)
    parser.add_argument("--dataset-run-id", required=True)
    parser.add_argument("--pair-generation-id", required=True)
    args = parser.parse_args()
    timeframe = "M1" if args.alignment_parquet is not None else "M5"
    manifest = _materialize_feature_base(
        source_parquet=args.source_parquet,
        alignment_parquet=args.alignment_parquet,
        seq_structure_manifest=args.seq_structure_manifest,
        output_parquet=args.output_parquet,
        dataset_run_id=args.dataset_run_id,
        pair_generation_id=args.pair_generation_id,
        timeframe=timeframe,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
