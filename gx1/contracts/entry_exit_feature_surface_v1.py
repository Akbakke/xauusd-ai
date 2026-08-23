"""Validated row-level native feature surfaces used by Entry and Exit.

M5 carries Entry's model input and M1 carries Exit's higher-resolution model
input.  Both surfaces use the same ordered semantic contract while retaining
their independently computed native-resolution values.
"""
from __future__ import annotations

import json
import mmap
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
    "gx1_entry_exit_m1_feature_surface_v10"
)
ENTRY_EXIT_M5_FEATURE_SURFACE_SCHEMA_VERSION = (
    "gx1_entry_exit_m5_feature_surface_v8"
)
ENTRY_M5_FEATURE_SURFACE_CONSUMPTION_MODE = (
    "exact_hash_bound_native_m5_feature_surface_v1"
)
ENTRY_EXIT_FEATURE_SURFACE_COLUMNS = ("time", "signal", "ctx_cont", "ctx_cat")
_M1_FEATURE_SURFACE_BATCH_ROWS = 8192
_M1_FEATURE_SURFACE_DISK_SYNC_ROWS = 262_144
_REGISTRY_FIT_BINDING_KEYS = frozenset(
    {
        "lane",
        "artifact_path",
        "artifact_sha256",
        "params_schema_version",
        "params_module",
        "params_contract_sha256",
    }
)

def _read_json_manifest(path: Path, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{context}_JSON_INVALID") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context}_JSON_OBJECT_REQUIRED")
    return payload


def _require_registry_fit_binding(
    value: Mapping[str, Any] | Any,
    *,
    timeframe: str,
    expected_source: Path,
    expected_source_sha256: str,
) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _REGISTRY_FIT_BINDING_KEYS
        or value.get("lane") != timeframe
    ):
        raise RuntimeError("ENTRY_EXIT_FEATURE_SURFACE_REGISTRY_BINDING_INVALID")
    artifact_raw = value.get("artifact_path")
    if not isinstance(artifact_raw, str):
        raise RuntimeError("ENTRY_EXIT_FEATURE_SURFACE_REGISTRY_BINDING_INVALID")
    artifact = Path(artifact_raw).expanduser()
    if (
        not artifact.is_absolute()
        or artifact.is_symlink()
        or not artifact.is_file()
        or artifact.resolve(strict=True) != artifact
    ):
        raise RuntimeError("ENTRY_EXIT_FEATURE_SURFACE_REGISTRY_BINDING_INVALID")
    from gx1.features.htf_features import (
        load_v29_registry_constants_manifest,
        load_v29_registry_m1_lane_params_manifest,
    )

    params = (
        load_v29_registry_constants_manifest(artifact)
        if timeframe == "M5"
        else load_v29_registry_m1_lane_params_manifest(artifact)
    )
    container = _read_json_manifest(
        artifact,
        context="ENTRY_EXIT_FEATURE_SURFACE_REGISTRY_ARTIFACT",
    )
    source_path_key = (
        "m5_prebuilt_source" if timeframe == "M5" else "output_parquet"
    )
    source_sha_key = (
        "m5_prebuilt_source_sha256"
        if timeframe == "M5"
        else "output_parquet_sha256"
    )
    if (
        container.get(source_path_key) != str(expected_source)
        or container.get(source_sha_key) != expected_source_sha256
    ):
        raise RuntimeError("ENTRY_EXIT_FEATURE_SURFACE_REGISTRY_SOURCE_MISMATCH")
    provenance = params.get("provenance")
    expected = {
        "lane": timeframe,
        "artifact_path": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "params_schema_version": str(params["schema_version"]),
        "params_module": str(provenance["module"]),
        "params_contract_sha256": str(params["contract_sha256"]),
    }
    if dict(value) != expected:
        raise RuntimeError("ENTRY_EXIT_FEATURE_SURFACE_REGISTRY_BINDING_INVALID")
    return expected


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
    registry_fit_binding: Mapping[str, Any],
    volatility_squeeze_artifact_binding: Mapping[str, Any],
    materialization: Mapping[str, Any] | None,
    causal_warmup: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the only Entry/Exit feature-surface manifest shape.

    ``causal_warmup`` declares the surface's leading-row exclusion: the fixed
    price-derived prefix plus the measured V29 layer warmup floors (data-
    dependent statistics on the exact declared source bytes). Downstream
    exactness checks bind these declared values instead of re-deriving a
    fixed prefix.
    """

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
    exact_registry_fit_binding = _require_registry_fit_binding(
        registry_fit_binding,
        timeframe=timeframe,
        expected_source=source,
        expected_source_sha256=source_binding["source_sha256"],
    )
    from gx1.features.volatility_squeeze_state_v1 import (
        require_volatility_squeeze_artifact_binding,
    )

    exact_squeeze_artifacts = require_volatility_squeeze_artifact_binding(
        volatility_squeeze_artifact_binding
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
        "registry_fit_binding": exact_registry_fit_binding,
        "volatility_squeeze_artifact_set": exact_squeeze_artifacts.binding(),
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
    if causal_warmup is not None:
        manifest["causal_warmup"] = dict(causal_warmup)
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
        context=context,
    )
    extension = payload.get("extension")
    registry_fit_binding = payload.get("registry_fit_binding")
    volatility_squeeze_artifact_binding = payload.get(
        "volatility_squeeze_artifact_set"
    )
    materialization = payload.get("materialization")
    if not isinstance(extension, Mapping) or not isinstance(
        materialization,
        Mapping,
    ) or not isinstance(registry_fit_binding, Mapping) or not isinstance(
        volatility_squeeze_artifact_binding,
        Mapping,
    ):
        raise RuntimeError(f"{context}_MANIFEST_COMPONENT_INVALID")
    # The declared leading-exclusion block (V29 measured warmup floors) is
    # part of the identity exactly like extension/materialization: lift the
    # observed block and let the byte-exact reconstruction comparison prove
    # it. Pre-V29 manifests carry no block and reconstruct identically.
    causal_warmup = payload.get("causal_warmup")
    if causal_warmup is not None and not isinstance(causal_warmup, Mapping):
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
        registry_fit_binding=registry_fit_binding,
        volatility_squeeze_artifact_binding=(
            volatility_squeeze_artifact_binding
        ),
        materialization=materialization,
        causal_warmup=causal_warmup,
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
        "sequence_first_time",
        "sequence_last_time",
        "dataset_run_id",
        "pair_generation_id",
        "feature_base_sha256",
        "feature_manifest_sha256",
        "feature_field_order_sha256",
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
    try:
        decision_time = pd.Timestamp(value["decision_time"])
    except Exception as exc:
        raise RuntimeError(
            f"{context}_M1_FEATURE_WINDOW_DECISION_TIME_INVALID"
        ) from exc
    if (
        pd.isna(decision_time)
        or decision_time.tzinfo is None
        or decision_time.utcoffset() != pd.Timedelta(0)
        or decision_time != decision_time.floor(
            f"{EXIT_DECISION_BAR_SECONDS}s"
        )
        or value["decision_time"]
        != decision_time.tz_convert("UTC").isoformat()
    ):
        raise RuntimeError(
            f"{context}_M1_FEATURE_WINDOW_DECISION_TIME_INVALID"
        )
    sequence_times: list[pd.Timestamp] = []
    for field in ("sequence_first_time", "sequence_last_time"):
        try:
            timestamp = pd.Timestamp(value[field])
        except Exception as exc:
            raise RuntimeError(
                f"{context}_M1_FEATURE_WINDOW_SEQUENCE_TIME_INVALID"
            ) from exc
        if (
            pd.isna(timestamp)
            or timestamp.tzinfo is None
            or timestamp.utcoffset() != pd.Timedelta(0)
            or timestamp != timestamp.floor(f"{EXIT_DECISION_BAR_SECONDS}s")
            or value[field] != timestamp.tz_convert("UTC").isoformat()
        ):
            raise RuntimeError(
                f"{context}_M1_FEATURE_WINDOW_SEQUENCE_TIME_INVALID"
            )
        sequence_times.append(timestamp)
    if not sequence_times[0] < sequence_times[1] or sequence_times[1] != decision_time:
        raise RuntimeError(
            f"{context}_M1_FEATURE_WINDOW_SEQUENCE_CLOCK_INVALID"
        )
    if (
        isinstance(value["dataset_run_id"], bool)
        or not isinstance(value["dataset_run_id"], str)
        or not value["dataset_run_id"]
        or not isinstance(value["pair_generation_id"], str)
        or not value["pair_generation_id"]
        or not isinstance(value["feature_base_sha256"], str)
        or len(value["feature_base_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in value["feature_base_sha256"])
        or not isinstance(value["feature_manifest_sha256"], str)
        or len(value["feature_manifest_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in value["feature_manifest_sha256"])
        or not isinstance(value["feature_field_order_sha256"], str)
        or len(value["feature_field_order_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in value["feature_field_order_sha256"])
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

    def _flush_and_discard_disk_backing() -> None:
        """Release clean shared pages without changing the decoded surface.

        The arrays are deliberately complete: the normalizer and Exit
        lifecycle must consume every owner-declared TRAIN row. On a bounded
        smoke host, retaining every clean page after decoding makes the file
        cache compete with the model for the same cgroup allowance. These are
        MAP_SHARED temporary files, so flushing commits the exact decoded
        bytes and MADV_DONTNEED only releases clean cache pages.
        """

        if backing_root is None:
            return
        advice = getattr(mmap, "MADV_DONTNEED", None)
        for values in arrays.values():
            if not isinstance(values, np.memmap):
                raise RuntimeError(
                    f"{context}_M1_FEATURE_SURFACE_DISK_BACKING_INVALID"
                )
            values.flush()
            mmap_handle = getattr(values, "_mmap", None)
            if (
                advice is not None
                and mmap_handle is not None
                and hasattr(mmap_handle, "madvise")
            ):
                mmap_handle.madvise(advice)

    # Bound dirty shared pages while decoding the complete immutable surface.
    # This is storage management only: every batch still receives the same
    # exact dtype, width, finite and categorical-domain validation.
    next_disk_sync = _M1_FEATURE_SURFACE_DISK_SYNC_ROWS
    try:
        batches = parquet.iter_batches(
            batch_size=_M1_FEATURE_SURFACE_BATCH_ROWS,
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
            if backing_root is not None and offsets >= next_disk_sync:
                _flush_and_discard_disk_backing()
                next_disk_sync = (
                    offsets + _M1_FEATURE_SURFACE_DISK_SYNC_ROWS
                )
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

    # Do not form a full-surface boolean matrix here. The canonical M1
    # feature surface is multi-gigabyte once decoded; bounded validation keeps
    # the same exhaustive domain check compatible with the 4 GiB attended
    # trainer envelope.
    for start in range(0, len(times), _M1_FEATURE_SURFACE_BATCH_ROWS):
        stop = min(len(times), start + _M1_FEATURE_SURFACE_BATCH_ROWS)
        for index, (lower, upper) in enumerate(
            MODEL_NATIVE_CTX_CAT_MIN_MAX.values()
        ):
            values = arrays["ctx_cat"][start:stop, index]
            if np.any(values < lower) or np.any(values > upper):
                raise RuntimeError(
                    f"{context}_M1_FEATURE_SURFACE_CTX_CAT_DOMAIN_INVALID: "
                    f"index={index}"
                )
        if backing_root is not None:
            _flush_and_discard_disk_backing()
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
