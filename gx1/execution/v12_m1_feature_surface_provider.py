"""Hash-bound causal M1 feature-surface provider for unified Exit.

The provider is deliberately separate from feature construction.  A producer
must first publish one immutable row-level M1 surface and bind it to the exact
dataset run and serving pair generation.  Runtime then reads only the 480
closed M1 rows needed for one Exit decision; it never recomputes ATR, SMC,
context, or specialist features and never substitutes the M5 Entry surface.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gx1.contracts.entry_exit_feature_base_v1 import (
    EXIT_DECISION_BAR_SECONDS,
    EXIT_FEATURE_SEQUENCE_BARS,
    require_entry_exit_shared_feature_base_contract,
)
from gx1.contracts.entry_exit_feature_surface_v1 import (
    ENTRY_EXIT_FEATURE_SURFACE_COLUMNS,
    ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION,
    require_m1_feature_window,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
)

MAX_RUNTIME_ROWGROUP_ROWS = EXIT_FEATURE_SEQUENCE_BARS * 4


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("M1_FEATURE_PROVIDER_MANIFEST_PATH_INVALID")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("M1_FEATURE_PROVIDER_MANIFEST_READ_INVALID") from exc
    if not isinstance(value, dict):
        raise RuntimeError("M1_FEATURE_PROVIDER_MANIFEST_SCHEMA_INVALID")
    return value


@dataclass(frozen=True)
class _SurfaceBinding:
    parquet_path: Path
    parquet_sha256: str
    manifest_path: Path
    manifest_sha256: str
    dataset_run_id: str
    pair_generation_id: str


class M1SharedFeatureSurfaceProvider:
    """Serve exact causal windows from one admitted M1 feature artifact."""

    def __init__(self, binding: _SurfaceBinding) -> None:
        self._binding = binding
        parquet = pq.ParquetFile(str(binding.parquet_path))
        if tuple(parquet.schema_arrow.names) != ENTRY_EXIT_FEATURE_SURFACE_COLUMNS:
            raise RuntimeError("M1_FEATURE_PROVIDER_PARQUET_SCHEMA_INVALID")
        if parquet.metadata is None or parquet.metadata.num_rows <= 0:
            raise RuntimeError("M1_FEATURE_PROVIDER_PARQUET_EMPTY")

        # The timestamp column is the only column read at construction.  It
        # gives a cheap exact row index while keeping the 513-wide surface
        # out of memory until a live Exit decision asks for one window.
        times = pd.DatetimeIndex(
            pd.to_datetime(
                parquet.read(columns=["time"])["time"].to_pandas(),
                utc=True,
                errors="coerce",
            )
        ).as_unit("ns")
        if (
            len(times) == 0
            or times.hasnans
            or not times.is_unique
            or not times.is_monotonic_increasing
            or not times.floor(f"{EXIT_DECISION_BAR_SECONDS}s").equals(times)
        ):
            raise RuntimeError("M1_FEATURE_PROVIDER_TIME_GEOMETRY_INVALID")
        self._times_ns = np.ascontiguousarray(times.asi8, dtype=np.int64)

        row_groups = parquet.metadata.num_row_groups
        offsets = [0]
        for index in range(row_groups):
            row_count = parquet.metadata.row_group(index).num_rows
            if row_count > MAX_RUNTIME_ROWGROUP_ROWS:
                raise RuntimeError("M1_FEATURE_PROVIDER_ROW_GROUP_TOO_LARGE")
            offsets.append(offsets[-1] + row_count)
        if offsets[-1] != len(self._times_ns):
            raise RuntimeError("M1_FEATURE_PROVIDER_ROW_GROUP_INDEX_INVALID")
        self._parquet = parquet
        self._row_group_offsets = np.asarray(offsets, dtype=np.int64)

    @classmethod
    def from_admitted_artifact(
        cls,
        *,
        parquet_path: Path,
        manifest_path: Path,
        dataset_run_id: str,
        pair_generation_id: str,
    ) -> "M1SharedFeatureSurfaceProvider":
        parquet = Path(parquet_path).expanduser().resolve()
        manifest = Path(manifest_path).expanduser().resolve()
        if (
            parquet.is_symlink()
            or not parquet.is_file()
            or manifest.is_symlink()
            or not manifest.is_file()
        ):
            raise RuntimeError("M1_FEATURE_PROVIDER_ARTIFACT_PATH_INVALID")
        if not isinstance(dataset_run_id, str) or not dataset_run_id:
            raise RuntimeError("M1_FEATURE_PROVIDER_DATASET_RUN_ID_INVALID")
        if not isinstance(pair_generation_id, str) or not pair_generation_id:
            raise RuntimeError("M1_FEATURE_PROVIDER_PAIR_GENERATION_ID_INVALID")

        feature_sha = _sha256_file(parquet)
        manifest_sha = _sha256_file(manifest)
        payload = _read_manifest(manifest)
        if (
            payload.get("schema_version") != ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION
            or payload.get("decision") != "PASS"
            or payload.get("dataset_run_id") != dataset_run_id
            or payload.get("pair_generation_id") != pair_generation_id
            or payload.get("output_parquet") != str(parquet)
            or payload.get("output_parquet_sha256") != feature_sha
            or payload.get("signal_dim") != MODEL_NATIVE_SIGNAL_DIM
            or payload.get("ctx_cont_dim") != MODEL_NATIVE_CTX_CONT_DIM
            or payload.get("ctx_cat_dim") != MODEL_NATIVE_CTX_CAT_DIM
        ):
            raise RuntimeError("M1_FEATURE_PROVIDER_MANIFEST_BINDING_INVALID")
        require_entry_exit_shared_feature_base_contract(
            payload.get("shared_feature_base_contract"),
            context="M1_FEATURE_PROVIDER",
        )
        return cls(
            _SurfaceBinding(
                parquet_path=parquet,
                parquet_sha256=feature_sha,
                manifest_path=manifest,
                manifest_sha256=manifest_sha,
                dataset_run_id=dataset_run_id,
                pair_generation_id=pair_generation_id,
            )
        )

    @property
    def feature_base_sha256(self) -> str:
        return self._binding.parquet_sha256

    @property
    def dataset_run_id(self) -> str:
        return self._binding.dataset_run_id

    @property
    def pair_generation_id(self) -> str:
        return self._binding.pair_generation_id

    def __call__(
        self,
        *,
        decision_time: Any,
        prebuilt_snapshot: Any,
    ) -> Mapping[str, Any]:
        timestamp = pd.Timestamp(decision_time)
        if (
            pd.isna(timestamp)
            or timestamp.tzinfo is None
            or timestamp.utcoffset() != pd.Timedelta(0)
            or timestamp != timestamp.floor(f"{EXIT_DECISION_BAR_SECONDS}s")
        ):
            raise RuntimeError("M1_FEATURE_PROVIDER_DECISION_TIME_INVALID")
        timestamp = timestamp.tz_convert("UTC").as_unit("ns")
        snapshot_pair_id = getattr(prebuilt_snapshot, "pair_generation_id", None)
        if snapshot_pair_id != self._binding.pair_generation_id:
            raise RuntimeError("M1_FEATURE_PROVIDER_PAIR_GENERATION_MISMATCH")

        end = int(np.searchsorted(self._times_ns, timestamp.value, side="left"))
        start = end - EXIT_FEATURE_SEQUENCE_BARS
        if (
            end >= len(self._times_ns)
            or self._times_ns[end] != timestamp.value
            or start < 0
        ):
            raise RuntimeError("M1_FEATURE_PROVIDER_CAUSAL_WINDOW_UNAVAILABLE")
        selected_times = self._times_ns[start:end]
        expected_step = int(pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS).value)
        if (
            len(selected_times) != EXIT_FEATURE_SEQUENCE_BARS
            or np.any(np.diff(selected_times) != expected_step)
        ):
            raise RuntimeError("M1_FEATURE_PROVIDER_CADENCE_GAP")

        first_group = int(
            np.searchsorted(self._row_group_offsets, start, side="right") - 1
        )
        last_group = int(
            np.searchsorted(self._row_group_offsets, end - 1, side="right") - 1
        )
        table = self._parquet.read_row_groups(
            list(range(first_group, last_group + 1)),
            columns=list(ENTRY_EXIT_FEATURE_SURFACE_COLUMNS),
        )
        frame = table.to_pandas()
        local_start = start - int(self._row_group_offsets[first_group])
        local_end = local_start + EXIT_FEATURE_SEQUENCE_BARS
        window = frame.iloc[local_start:local_end]
        if len(window) != EXIT_FEATURE_SEQUENCE_BARS:
            raise RuntimeError("M1_FEATURE_PROVIDER_WINDOW_READ_INVALID")

        def matrix(name: str, width: int) -> np.ndarray:
            values = np.asarray(window[name].tolist(), dtype=np.float32)
            if values.shape != (EXIT_FEATURE_SEQUENCE_BARS, width):
                raise RuntimeError(f"M1_FEATURE_PROVIDER_{name.upper()}_SHAPE_INVALID")
            return np.ascontiguousarray(values)

        signal = matrix("signal", MODEL_NATIVE_SIGNAL_DIM)
        ctx_cont_all = matrix("ctx_cont", MODEL_NATIVE_CTX_CONT_DIM)
        ctx_cat_all = np.asarray(window["ctx_cat"].tolist(), dtype=np.int64)
        if ctx_cat_all.shape != (EXIT_FEATURE_SEQUENCE_BARS, MODEL_NATIVE_CTX_CAT_DIM):
            raise RuntimeError("M1_FEATURE_PROVIDER_CTX_CAT_SHAPE_INVALID")
        value = {
            "schema_version": ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION,
            "decision_time": timestamp.isoformat(),
            "dataset_run_id": self._binding.dataset_run_id,
            "feature_base_sha256": self._binding.parquet_sha256,
            "sequence_bars": EXIT_FEATURE_SEQUENCE_BARS,
            "signal": signal,
            "snap": signal[-1].copy(),
            "ctx_cont": ctx_cont_all[-1].copy(),
            "ctx_cat": ctx_cat_all[-1].copy(),
        }
        return require_m1_feature_window(
            value,
            context="M1_FEATURE_PROVIDER_WINDOW",
        )
