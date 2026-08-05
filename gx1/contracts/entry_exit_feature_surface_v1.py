"""Validated row-level native feature surfaces used by Entry and Exit.

M5 carries Entry's model input and M1 carries Exit's higher-resolution model
input.  Both surfaces use the same ordered semantic contract while retaining
their independently computed native-resolution values.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
    require_model_native_manifest,
)
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_CTX_CAT_MIN_MAX
from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_EXIT_FEATURE_BASE_SCHEMA_VERSION,
    EXIT_DECISION_BAR_SECONDS,
    EXIT_FEATURE_SEQUENCE_BARS,
    entry_exit_shared_feature_base_contract,
    require_entry_exit_enriched_source_binding,
    require_entry_exit_feature_surface_identity,
)
from gx1.contracts.entry_exit_production_architecture_v1 import (
    current_entry_exit_architecture_observation,
    require_entry_exit_production_architecture,
)
from gx1.utils.artifact_primitives_v1 import (
    canonical_json_sha256,
    require_immutable_artifact,
    sha256_file,
)


ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION = (
    "gx1_entry_exit_m1_feature_surface_v1"
)
ENTRY_EXIT_M5_FEATURE_SURFACE_SCHEMA_VERSION = (
    "gx1_entry_exit_m5_feature_surface_v1"
)
ENTRY_M5_FEATURE_SURFACE_CONSUMPTION_MODE = (
    "exact_hash_bound_native_m5_feature_surface_v1"
)
ENTRY_EXIT_FEATURE_SURFACE_COLUMNS = ("time", "signal", "ctx_cont", "ctx_cat")

def _read_json_manifest(path: Path, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{context}_JSON_INVALID") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context}_JSON_OBJECT_REQUIRED")
    return payload


def build_entry_exit_feature_surface_manifest(
    *,
    timeframe: str,
    dataset_run_id: str,
    pair_generation_id: str,
    source: Path,
    source_binding: Mapping[str, str],
    alignment: Path | None,
    seq_structure_manifest: Path,
    output: Path,
    rows: int,
    signal_contract: Mapping[str, Any],
    extension: Mapping[str, Any],
    materialization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the only Entry/Exit feature-surface manifest shape."""

    if (
        timeframe not in {"M1", "M5"}
        or (timeframe == "M1") != (materialization is not None)
        or (timeframe == "M1" and alignment is None)
    ):
        raise RuntimeError("ENTRY_EXIT_FEATURE_SURFACE_BUILD_CONTRACT_INVALID")
    schema_version = (
        ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION
        if timeframe == "M1"
        else ENTRY_EXIT_M5_FEATURE_SURFACE_SCHEMA_VERSION
    )
    fields = list(signal_contract["fields"])
    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "decision": "PASS",
        "feature_base_contract_schema_version": (
            ENTRY_EXIT_FEATURE_BASE_SCHEMA_VERSION
        ),
        "shared_feature_base_contract": entry_exit_shared_feature_base_contract(),
        "dataset_run_id": dataset_run_id,
        "pair_generation_id": pair_generation_id,
        "anchor_timeframe": timeframe,
        "source_parquet": str(source),
        "source_sha256": source_binding["source_sha256"],
        "source_manifest": source_binding["manifest_path"],
        "source_manifest_sha256": source_binding["manifest_sha256"],
        "source_manifest_schema_version": source_binding["schema_version"],
        "rank_reference_npz": source_binding["rank_reference_npz"],
        "rank_reference_sha256": source_binding["rank_reference_sha256"],
        "alignment_parquet": None if alignment is None else str(alignment),
        "alignment_sha256": (
            None if alignment is None else sha256_file(alignment)
        ),
        "seq_structure_manifest": str(seq_structure_manifest),
        "seq_structure_manifest_sha256": sha256_file(seq_structure_manifest),
        "output_parquet": str(output),
        "output_parquet_sha256": sha256_file(output),
        "rows": rows,
        "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        "feature_field_order": fields,
        "feature_field_order_sha256": canonical_json_sha256(fields),
        "extension": dict(extension),
        "causal_contract": {
            "future_rows_used": False,
            "closed_decision_bar_required": True,
            "m1_closed_bar_required": timeframe == "M1",
            "m5_row_reuse": False,
            "native_resolution_values": True,
            "cross_resolution_value_copy": False,
            "computed_m1_feature_resampling": False,
            "duplicate_feature_implementation": False,
            "exact_closed_source_timestamp_subset": alignment is not None,
        },
    }
    if materialization is not None:
        manifest["materialization"] = dict(materialization)
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


def require_exact_m1_feature_surface_manifest(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    expected_parquet_path: Path,
    expected_parquet_sha256: str,
    expected_dataset_run_id: str,
    expected_pair_generation_id: str,
    expected_rows: int,
    expected_m1_source_path: Path,
    expected_m1_source_sha256: str,
    context: str,
) -> dict[str, Any]:
    """Bind the exact M1 surface before lifecycle allocates either matrix."""

    if (
        not isinstance(expected_dataset_run_id, str)
        or not expected_dataset_run_id
        or not isinstance(expected_pair_generation_id, str)
        or not expected_pair_generation_id
        or isinstance(expected_rows, bool)
        or not isinstance(expected_rows, int)
        or expected_rows <= 0
    ):
        raise RuntimeError(f"{context}_EXPECTED_IDENTITY_INVALID")
    manifest_path = require_immutable_artifact(
        Path(manifest_path),
        expected_sha256=expected_manifest_sha256,
        context=f"{context}_MANIFEST",
    )
    parquet_path = require_immutable_artifact(
        Path(expected_parquet_path),
        expected_sha256=expected_parquet_sha256,
        context=f"{context}_PARQUET",
    )
    m1_source_path = require_immutable_artifact(
        Path(expected_m1_source_path),
        expected_sha256=expected_m1_source_sha256,
        context=f"{context}_M1_SOURCE",
    )
    if manifest_path != Path(f"{parquet_path}.manifest.json"):
        raise RuntimeError(f"{context}_MANIFEST_SIDECAR_INVALID")

    payload = _read_json_manifest(manifest_path, context=context)
    source_raw = payload.get("source_parquet")
    if not isinstance(source_raw, str):
        raise RuntimeError(f"{context}_SOURCE_PATH_INVALID")
    source_binding = require_entry_exit_enriched_source_binding(
        Path(source_raw),
        dataset_run_id=expected_dataset_run_id,
        pair_generation_id=expected_pair_generation_id,
        timeframe="M1",
        context=context,
    )
    signal_raw = payload.get("seq_structure_manifest")
    signal_sha = payload.get("seq_structure_manifest_sha256")
    if not isinstance(signal_raw, str) or not isinstance(signal_sha, str):
        raise RuntimeError(f"{context}_SIGNAL_MANIFEST_IDENTITY_INVALID")
    signal_path = require_immutable_artifact(
        Path(signal_raw),
        expected_sha256=signal_sha,
        context=f"{context}_SIGNAL_MANIFEST",
    )
    signal_contract = require_model_native_manifest(
        _read_json_manifest(signal_path, context=f"{context}_SIGNAL_MANIFEST"),
        context=f"{context}_SIGNAL_MANIFEST",
    )
    require_entry_exit_feature_surface_identity(
        payload,
        expected_timeframe="M1",
        expected_ordered_fields=signal_contract["fields"],
        expected_signal_manifest_path=str(signal_path),
        expected_signal_manifest_sha256=signal_sha,
        expected_rank_reference_sha256=source_binding[
            "rank_reference_sha256"
        ],
        context=context,
    )
    extension = payload.get("extension")
    materialization = payload.get("materialization")
    if not isinstance(extension, Mapping) or not isinstance(
        materialization,
        Mapping,
    ):
        raise RuntimeError(f"{context}_MANIFEST_COMPONENT_INVALID")
    expected_manifest = build_entry_exit_feature_surface_manifest(
        timeframe="M1",
        dataset_run_id=expected_dataset_run_id,
        pair_generation_id=expected_pair_generation_id,
        source=Path(source_raw),
        source_binding=source_binding,
        alignment=m1_source_path,
        seq_structure_manifest=signal_path,
        output=parquet_path,
        rows=expected_rows,
        signal_contract=signal_contract,
        extension=extension,
        materialization=materialization,
    )
    if payload != expected_manifest:
        raise RuntimeError(f"{context}_IDENTITY_INVALID")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(parquet_path)
    except Exception as exc:
        raise RuntimeError(f"{context}_PARQUET_SCHEMA_INVALID") from exc
    if (
        parquet.metadata is None
        or parquet.metadata.num_rows != expected_rows
        or tuple(parquet.schema_arrow.names)
        != ENTRY_EXIT_FEATURE_SURFACE_COLUMNS
    ):
        raise RuntimeError(f"{context}_PARQUET_SCHEMA_INVALID")
    expected_types = (
        ("signal", MODEL_NATIVE_SIGNAL_DIM, pa.float32()),
        ("ctx_cont", MODEL_NATIVE_CTX_CONT_DIM, pa.float32()),
        ("ctx_cat", MODEL_NATIVE_CTX_CAT_DIM, pa.int64()),
    )
    for name, width, dtype in expected_types:
        observed = parquet.schema_arrow.field(name).type
        if (
            not pa.types.is_fixed_size_list(observed)
            or observed.list_size != width
            or observed.value_type != dtype
        ):
            raise RuntimeError(f"{context}_{name.upper()}_SCHEMA_INVALID")
    return payload


def require_m1_feature_window(
    value: Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Validate one atomic live M1 feature window before model invocation."""

    require_entry_exit_production_architecture(
        current_entry_exit_architecture_observation(),
        context=f"{context}_M1_FEATURE_WINDOW",
    )

    required = {
        "schema_version",
        "decision_time",
        "dataset_run_id",
        "feature_base_sha256",
        "sequence_bars",
        "signal",
        "snap",
        "ctx_cont",
        "ctx_cat",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise RuntimeError(f"{context}_M1_FEATURE_WINDOW_SCHEMA_INVALID")
    if value["schema_version"] != ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION:
        raise RuntimeError(f"{context}_M1_FEATURE_WINDOW_VERSION_INVALID")
    if (
        isinstance(value["dataset_run_id"], bool)
        or not isinstance(value["dataset_run_id"], str)
        or not value["dataset_run_id"]
        or not isinstance(value["feature_base_sha256"], str)
        or len(value["feature_base_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in value["feature_base_sha256"])
        or value["sequence_bars"] != EXIT_FEATURE_SEQUENCE_BARS
    ):
        raise RuntimeError(f"{context}_M1_FEATURE_WINDOW_IDENTITY_INVALID")
    signal = np.asarray(value["signal"], dtype=np.float32)
    snap = np.asarray(value["snap"], dtype=np.float32)
    ctx_cont = np.asarray(value["ctx_cont"], dtype=np.float32)
    ctx_cat_raw = np.asarray(value["ctx_cat"], dtype=np.float64)
    if (
        ctx_cat_raw.shape != (MODEL_NATIVE_CTX_CAT_DIM,)
        or not np.isfinite(ctx_cat_raw).all()
        or not np.array_equal(ctx_cat_raw, np.rint(ctx_cat_raw))
    ):
        raise RuntimeError(f"{context}_M1_FEATURE_WINDOW_CTX_CAT_INVALID")
    ctx_cat = ctx_cat_raw.astype(np.int64)
    expected = {
        "signal": (EXIT_FEATURE_SEQUENCE_BARS, MODEL_NATIVE_SIGNAL_DIM),
        "snap": (MODEL_NATIVE_SIGNAL_DIM,),
        "ctx_cont": (MODEL_NATIVE_CTX_CONT_DIM,),
        "ctx_cat": (MODEL_NATIVE_CTX_CAT_DIM,),
    }
    observed = {
        "signal": signal,
        "snap": snap,
        "ctx_cont": ctx_cont,
        "ctx_cat": ctx_cat,
    }
    for name, array in observed.items():
        if array.shape != expected[name] or not np.isfinite(array).all():
            raise RuntimeError(f"{context}_M1_FEATURE_WINDOW_{name.upper()}_INVALID")
    if not np.array_equal(signal[-1], snap):
        raise RuntimeError(f"{context}_M1_FEATURE_WINDOW_SNAP_ALIAS_INVALID")
    for index, (lower, upper) in enumerate(MODEL_NATIVE_CTX_CAT_MIN_MAX.values()):
        if np.any(ctx_cat[index] < lower) or np.any(ctx_cat[index] > upper):
            raise RuntimeError(
                f"{context}_M1_FEATURE_WINDOW_CTX_CAT_DOMAIN_INVALID"
            )
    return {
        **value,
        "signal": np.ascontiguousarray(signal),
        "snap": np.ascontiguousarray(snap),
        "ctx_cont": np.ascontiguousarray(ctx_cont),
        "ctx_cat": np.ascontiguousarray(ctx_cat),
    }


def load_m1_feature_surface(
    path: Path,
    *,
    context: str,
    storage_dir: Path | None = None,
    expected_bar_seconds: int = EXIT_DECISION_BAR_SECONDS,
) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    """Load and validate one exact row-level M1 feature-base artifact.

    The M1 surface is intentionally shared by Entry and Exit, but it is much
    larger than a normal training parquet because it contains one row per
    minute.  Read the nested fixed-size lists in bounded Arrow batches so the
    contract does not transiently materialize a second object-heavy pandas
    copy of the full feature base.
    """

    require_entry_exit_production_architecture(
        current_entry_exit_architecture_observation(),
        context=f"{context}_M1_FEATURE_SURFACE_LOAD",
    )
    if (
        isinstance(expected_bar_seconds, bool)
        or not isinstance(expected_bar_seconds, int)
        or expected_bar_seconds <= 0
    ):
        raise RuntimeError(f"{context}_FEATURE_SURFACE_BAR_SECONDS_INVALID")
    resolved = Path(path).expanduser().absolute()
    if (
        not resolved.is_absolute()
        or resolved.is_symlink()
        or not resolved.is_file()
    ):
        raise RuntimeError(f"{context}_M1_FEATURE_SURFACE_PATH_INVALID")
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(resolved)
        columns = tuple(parquet.schema_arrow.names)
    except Exception as exc:
        raise RuntimeError(f"{context}_M1_FEATURE_SURFACE_SCHEMA_INVALID") from exc
    if columns != ENTRY_EXIT_FEATURE_SURFACE_COLUMNS:
        raise RuntimeError(
            f"{context}_M1_FEATURE_SURFACE_SCHEMA_INVALID: "
            f"columns={columns}"
        )
    try:
        time_frame = pd.read_parquet(resolved, columns=["time"])
        times = pd.DatetimeIndex(
            pd.to_datetime(time_frame["time"], utc=True, errors="coerce")
        ).as_unit("ns")
    except Exception as exc:
        raise RuntimeError(f"{context}_M1_FEATURE_SURFACE_TIME_INVALID") from exc
    if (
        len(times) == 0
        or times.hasnans
        or not times.is_unique
        or not times.is_monotonic_increasing
        or not times.floor(f"{expected_bar_seconds}s").equals(times)
    ):
        raise RuntimeError(f"{context}_M1_FEATURE_SURFACE_TIME_INVALID")

    backing_root: Path | None = None
    if storage_dir is not None:
        backing_root = Path(storage_dir).expanduser().absolute()
        if backing_root.exists() and backing_root.is_symlink():
            raise RuntimeError(
                f"{context}_M1_FEATURE_SURFACE_STORAGE_PATH_INVALID"
            )
        backing_root.mkdir(parents=True, exist_ok=True)

    def _allocate(name: str, shape: tuple[int, int], dtype: Any) -> np.ndarray:
        if backing_root is None:
            return np.empty(shape, dtype=dtype)
        return np.memmap(
            backing_root / f"{name}.mmap",
            dtype=dtype,
            mode="w+",
            shape=shape,
        )

    arrays = {
        "signal": _allocate(
            "signal",
            (len(times), MODEL_NATIVE_SIGNAL_DIM),
            np.float32,
        ),
        "ctx_cont": _allocate(
            "ctx_cont",
            (len(times), MODEL_NATIVE_CTX_CONT_DIM),
            np.float32,
        ),
        "ctx_cat": _allocate(
            "ctx_cat",
            (len(times), MODEL_NATIVE_CTX_CAT_DIM),
            np.int64,
        ),
    }
    widths = {
        "signal": MODEL_NATIVE_SIGNAL_DIM,
        "ctx_cont": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat": MODEL_NATIVE_CTX_CAT_DIM,
    }
    offsets = 0
    try:
        batches = parquet.iter_batches(
            batch_size=8192,
            columns=["signal", "ctx_cont", "ctx_cat"],
            use_threads=False,
        )
        for batch in batches:
            row_count = int(batch.num_rows)
            if row_count <= 0:
                continue
            if offsets + row_count > len(times):
                raise RuntimeError(
                    f"{context}_M1_FEATURE_SURFACE_ROW_COUNT_INVALID"
                )
            for name, width in widths.items():
                column = batch.column(batch.schema.get_field_index(name))
                if not hasattr(column, "values"):
                    raise RuntimeError(
                        f"{context}_M1_FEATURE_SURFACE_{name.upper()}_DECODE_INVALID"
                    )
                flat = np.asarray(
                    column.values.to_numpy(zero_copy_only=False),
                    dtype=np.float64 if name == "ctx_cat" else np.float32,
                )
                if flat.shape != (row_count * width,):
                    raise RuntimeError(
                        f"{context}_M1_FEATURE_SURFACE_{name.upper()}_WIDTH_INVALID: "
                        f"shape={flat.shape} expected=({row_count * width},)"
                    )
                values = flat.reshape(row_count, width)
                if name == "ctx_cat":
                    if not np.isfinite(values).all() or not np.array_equal(
                        values, np.rint(values)
                    ):
                        raise RuntimeError(
                            f"{context}_M1_FEATURE_SURFACE_CTX_CAT_NONINTEGER"
                        )
                    cast_values = values.astype(np.int64)
                else:
                    if not np.isfinite(values).all():
                        raise RuntimeError(
                            f"{context}_M1_FEATURE_SURFACE_NONFINITE"
                        )
                    cast_values = values.astype(np.float32, copy=False)
                arrays[name][offsets : offsets + row_count] = cast_values
            offsets += row_count
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"{context}_M1_FEATURE_SURFACE_DECODE_INVALID"
        ) from exc
    if offsets != len(times):
        raise RuntimeError(
            f"{context}_M1_FEATURE_SURFACE_ROW_COUNT_INVALID: "
            f"rows={offsets} expected={len(times)}"
        )

    for index, (lower, upper) in enumerate(MODEL_NATIVE_CTX_CAT_MIN_MAX.values()):
        values = arrays["ctx_cat"][:, index]
        if np.any(values < lower) or np.any(values > upper):
            raise RuntimeError(
                f"{context}_M1_FEATURE_SURFACE_CTX_CAT_DOMAIN_INVALID: index={index}"
            )
    if backing_root is not None:
        for values in arrays.values():
            if isinstance(values, np.memmap):
                values.flush()
                mmap_handle = getattr(values, "_mmap", None)
                if mmap_handle is not None and hasattr(mmap_handle, "madvise"):
                    import mmap

                    mmap_handle.madvise(mmap.MADV_DONTNEED)
    return times, {
        name: values
        for name, values in arrays.items()
    }


def load_m1_feature_surface_times(
    path: Path,
    *,
    context: str,
    expected_bar_seconds: int = EXIT_DECISION_BAR_SECONDS,
) -> pd.DatetimeIndex:
    """Validate only one ordered native clock without loading feature arrays.

    Dataset/lifecycle producers that do not consume the feature vectors must
    not deserialize the full nested ``signal``/``ctx`` columns.  The immutable
    manifest and the full-surface preflight remain responsible for validating
    vector widths, finiteness and categorical domains; this helper owns only
    the exact parquet schema and causal timestamp geometry needed by such a
    producer.
    """

    require_entry_exit_production_architecture(
        current_entry_exit_architecture_observation(),
        context=f"{context}_M1_FEATURE_SURFACE_TIME_LOAD",
    )
    if (
        isinstance(expected_bar_seconds, bool)
        or not isinstance(expected_bar_seconds, int)
        or expected_bar_seconds <= 0
    ):
        raise RuntimeError(f"{context}_FEATURE_SURFACE_BAR_SECONDS_INVALID")
    resolved = Path(path).expanduser().absolute()
    if (
        not resolved.is_absolute()
        or resolved.is_symlink()
        or not resolved.is_file()
    ):
        raise RuntimeError(f"{context}_M1_FEATURE_SURFACE_PATH_INVALID")
    try:
        import pyarrow.parquet as pq

        columns = tuple(pq.read_schema(resolved).names)
    except Exception as exc:
        raise RuntimeError(f"{context}_M1_FEATURE_SURFACE_SCHEMA_INVALID") from exc
    if columns != ENTRY_EXIT_FEATURE_SURFACE_COLUMNS:
        raise RuntimeError(
            f"{context}_M1_FEATURE_SURFACE_SCHEMA_INVALID: columns={columns}"
        )
    try:
        frame = pd.read_parquet(resolved, columns=["time"])
        times = pd.DatetimeIndex(
            pd.to_datetime(frame["time"], utc=True, errors="coerce")
        ).as_unit("ns")
    except Exception as exc:
        raise RuntimeError(f"{context}_M1_FEATURE_SURFACE_TIME_INVALID") from exc
    if (
        len(times) == 0
        or times.hasnans
        or not times.is_unique
        or not times.is_monotonic_increasing
        or not times.floor(f"{expected_bar_seconds}s").equals(times)
    ):
        raise RuntimeError(f"{context}_M1_FEATURE_SURFACE_TIME_INVALID")
    return times
