"""Validated row-level feature-base surface used by unified Exit lifecycle.

The surface stores one causal M1 row per timestamp.  Lifecycle sampling builds
the higher-resolution sequence by slicing these rows; it never rebuilds a
second feature taxonomy or substitutes the M5 Entry sequence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_CTX_CAT_MIN_MAX
from gx1.contracts.entry_exit_feature_base_v1 import EXIT_DECISION_BAR_SECONDS
from gx1.contracts.entry_exit_feature_base_v1 import EXIT_FEATURE_SEQUENCE_BARS


ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION = (
    "gx1_entry_exit_m1_feature_surface_v1"
)
ENTRY_EXIT_FEATURE_SURFACE_COLUMNS = ("time", "signal", "ctx_cont", "ctx_cat")


def require_m1_feature_window(
    value: Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Validate one atomic live M1 feature window before model invocation."""

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
) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    """Load and validate one exact row-level M1 feature-base artifact.

    The M1 surface is intentionally shared by Entry and Exit, but it is much
    larger than a normal training parquet because it contains one row per
    minute.  Read the nested fixed-size lists in bounded Arrow batches so the
    contract does not transiently materialize a second object-heavy pandas
    copy of the full feature base.
    """

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
        or not times.floor(f"{EXIT_DECISION_BAR_SECONDS}s").equals(times)
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
) -> pd.DatetimeIndex:
    """Validate only the ordered M1 clock without materializing feature arrays.

    Dataset/lifecycle producers that do not consume the feature vectors must
    not deserialize the full nested ``signal``/``ctx`` columns.  The immutable
    manifest and the full-surface preflight remain responsible for validating
    vector widths, finiteness and categorical domains; this helper owns only
    the exact parquet schema and causal timestamp geometry needed by such a
    producer.
    """

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
        or not times.floor(f"{EXIT_DECISION_BAR_SECONDS}s").equals(times)
    ):
        raise RuntimeError(f"{context}_M1_FEATURE_SURFACE_TIME_INVALID")
    return times
