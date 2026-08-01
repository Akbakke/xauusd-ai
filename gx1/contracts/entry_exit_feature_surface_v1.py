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
) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    """Load and validate one exact row-level M1 feature-base artifact."""

    resolved = Path(path).expanduser().absolute()
    if (
        not resolved.is_absolute()
        or resolved.is_symlink()
        or not resolved.is_file()
    ):
        raise RuntimeError(f"{context}_M1_FEATURE_SURFACE_PATH_INVALID")
    frame = pd.read_parquet(resolved)
    if tuple(frame.columns) != ENTRY_EXIT_FEATURE_SURFACE_COLUMNS:
        raise RuntimeError(
            f"{context}_M1_FEATURE_SURFACE_SCHEMA_INVALID: "
            f"columns={tuple(frame.columns)}"
        )
    times = pd.DatetimeIndex(
        pd.to_datetime(frame["time"], utc=True, errors="coerce")
    ).as_unit("ns")
    if (
        len(times) == 0
        or times.hasnans
        or not times.is_unique
        or not times.is_monotonic_increasing
        or not times.floor(f"{EXIT_DECISION_BAR_SECONDS}s").equals(times)
    ):
        raise RuntimeError(f"{context}_M1_FEATURE_SURFACE_TIME_INVALID")

    def _matrix(name: str, width: int, dtype: Any) -> np.ndarray:
        rows = frame[name].tolist()
        try:
            values = np.asarray(rows, dtype=dtype)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{context}_M1_FEATURE_SURFACE_{name.upper()}_DECODE_INVALID"
            ) from exc
        if values.shape != (len(frame), width):
            raise RuntimeError(
                f"{context}_M1_FEATURE_SURFACE_{name.upper()}_WIDTH_INVALID: "
                f"shape={values.shape} expected=({len(frame)},{width})"
            )
        return values

    signal = _matrix("signal", MODEL_NATIVE_SIGNAL_DIM, np.float32)
    ctx_cont = _matrix("ctx_cont", MODEL_NATIVE_CTX_CONT_DIM, np.float32)
    ctx_cat_raw = _matrix("ctx_cat", MODEL_NATIVE_CTX_CAT_DIM, np.float64)
    if not np.array_equal(ctx_cat_raw, np.rint(ctx_cat_raw)):
        raise RuntimeError(f"{context}_M1_FEATURE_SURFACE_CTX_CAT_NONINTEGER")
    ctx_cat = ctx_cat_raw.astype(np.int64)
    if not np.isfinite(signal).all() or not np.isfinite(ctx_cont).all():
        raise RuntimeError(f"{context}_M1_FEATURE_SURFACE_NONFINITE")
    for index, (lower, upper) in enumerate(MODEL_NATIVE_CTX_CAT_MIN_MAX.values()):
        values = ctx_cat[:, index]
        if np.any(values < lower) or np.any(values > upper):
            raise RuntimeError(
                f"{context}_M1_FEATURE_SURFACE_CTX_CAT_DOMAIN_INVALID: index={index}"
            )
    return times, {
        "signal": np.ascontiguousarray(signal),
        "ctx_cont": np.ascontiguousarray(ctx_cont),
        "ctx_cat": np.ascontiguousarray(ctx_cat),
    }
