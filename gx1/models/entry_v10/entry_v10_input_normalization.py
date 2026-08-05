"""Exact shared TRAIN-only fit helper for model-native Entry and Exit.

This module owns population selection and artifact lineage only.  The
normalization math and immutable schema remain owned by
``entry_model_native_input_normalization_v1``.  In particular:

* the shared local 513 surface fits unique physical M5 and M1 rows once;
* ctx_cont/ctx_cat fit Entry decisions plus unique Exit M1 decisions once;
* current-bar signal/context aliases are derived from the routing contract,
  checked bit-for-bit, and inherit local-population statistics;
* MTF fits union Entry +5m and Exit +1m route consumption per cache row;
* VAL and TEST contribute exactly zero observations.

The helper deliberately has no subsample argument.  A training sampler is a
downstream concern and therefore cannot change the fitted population.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_input_normalization_v1 import (
    CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS,
    EXPECTED_SURFACES,
    EXPECTED_TFS,
    MatrixPopulationPart,
    MTF_SEMANTIC_CATEGORICAL_DOMAINS,
    build_input_normalization_contract,
    fit_ctx_cat_contract,
    fit_surface_normalization,
    share_temporal_alias_stats_from_signal,
)
from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_DECISION_BAR_SECONDS,
    ENTRY_MTF_CONTEXT_TIMEFRAMES,
    EXIT_DECISION_BAR_SECONDS,
    EXIT_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT,
    model_native_context_temporal_alias_policy,
    require_model_native_context_specialist_routing,
)
from gx1.features.htf_features import (
    HTF_V4_CACHE_SCHEMA_VERSION,
    HTF_V4_MATRIX_CONTRACT,
    MULTI_TF_FEATURE_COUNT_V4,
    MULTI_TF_PER_BAR_FEATURES_V4,
    MULTI_TF_SHIFT,
    MultiTFV4DiskCache,
    load_multi_tf_v4_cache,
)


FIT_HELPER_SCHEMA_VERSION = "entry_v10_train_input_normalization_fit_v1"
FIT_POPULATION_PROOF_SCHEMA_VERSION = (
    "entry_v10_train_input_normalization_population_proof_v1"
)
ENTRY_TARGET_AVAILABILITY_SHIFT = pd.Timedelta(
    seconds=ENTRY_DECISION_BAR_SECONDS
)
EXIT_TARGET_AVAILABILITY_SHIFT = pd.Timedelta(
    seconds=EXIT_DECISION_BAR_SECONDS
)
DEFAULT_ROW_CHUNK = 4096
_SHA256_BUFFER_SIZE = 1024 * 1024
MODEL_NATIVE_MTF_CACHE_BINDING_KEYS = frozenset(
    {
        "cache_dir",
        "manifest_path",
        "manifest_sha256",
        "cache_identity_sha256",
        "m5_prebuilt_source",
        "m5_prebuilt_source_sha256",
    }
)


@dataclass(frozen=True)
class TrainNormalizationArtifacts:
    """Exact immutable files which authorize one normalization fit."""

    dataset_run_id: str
    train_parquet_path: Path
    train_manifest_path: Path
    m5_prebuilt_path: Path
    mtf_cache_dir: Path


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


def _field_names_sha256(field_names: Sequence[str]) -> str:
    return _canonical_sha256([str(name) for name in field_names])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_SHA256_BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_regular_file(raw_path: Path, *, label: str) -> Path:
    supplied = Path(raw_path).expanduser()
    if not supplied.is_absolute():
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_ARTIFACT_PATH_INVALID] "
            f"label={label} reason=not_absolute"
        )
    if supplied.is_symlink() or any(parent.is_symlink() for parent in supplied.parents):
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_ARTIFACT_PATH_INVALID] "
            f"label={label} reason=symlink"
        )
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_ARTIFACT_PATH_INVALID] label={label}"
        ) from exc
    if resolved != supplied or not resolved.is_file():
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_ARTIFACT_PATH_INVALID] label={label}"
        )
    return resolved


def _json_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_without_duplicate_keys,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_JSON_INVALID] label={label}"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_JSON_INVALID] label={label}"
        )
    return value


def _exact_sha256(value: Any, *, context: str, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RuntimeError(
            f"[{context}_INVALID] {field} must be an exact lowercase SHA-256"
        )
    return value


def _exact_directory(raw_path: Path, *, context: str) -> Path:
    supplied = Path(raw_path).expanduser()
    if (
        not supplied.is_absolute()
        or supplied.is_symlink()
        or any(parent.is_symlink() for parent in supplied.parents)
    ):
        raise RuntimeError(f"[{context}_INVALID] cache_dir is not canonical")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"[{context}_INVALID] cache_dir is missing") from exc
    if resolved != supplied or not resolved.is_dir():
        raise RuntimeError(f"[{context}_INVALID] cache_dir is not canonical")
    return resolved


def require_dataset_manifest_multi_tf_cache_binding(
    manifest: Mapping[str, Any],
    *,
    dataset_run_id: str,
    context: str,
) -> dict[str, str]:
    """Return one exact split-manifest declaration for the shared V4 cache."""

    if not isinstance(manifest, Mapping):
        raise RuntimeError(f"[{context}_INVALID] split manifest is not an object")
    extra = manifest.get("extra")
    if not isinstance(extra, Mapping) or extra.get("entry_run_id") != dataset_run_id:
        raise RuntimeError(f"[{context}_INVALID] dataset run lineage mismatch")
    raw = extra.get("multi_tf_cache_binding")
    if not isinstance(raw, Mapping) or set(raw) != MODEL_NATIVE_MTF_CACHE_BINDING_KEYS:
        raise RuntimeError(f"[{context}_INVALID] exact cache binding is missing")
    binding = {str(key): value for key, value in raw.items()}
    for field in (
        "manifest_sha256",
        "cache_identity_sha256",
        "m5_prebuilt_source_sha256",
    ):
        _exact_sha256(binding[field], context=context, field=field)
    for field in ("cache_dir", "manifest_path", "m5_prebuilt_source"):
        value = binding[field]
        if not isinstance(value, str) or not value or Path(value).expanduser() != Path(value):
            raise RuntimeError(f"[{context}_INVALID] {field} must be an exact path")
    return dict(binding)  # type: ignore[arg-type]


def require_multi_tf_v4_cache_binding_files(
    binding: Mapping[str, Any],
    *,
    expected_cache_dir: Path,
    context: str,
) -> dict[str, str]:
    """Bind a split declaration to the exact V4 manifest and source bytes."""

    if not isinstance(binding, Mapping) or set(binding) != MODEL_NATIVE_MTF_CACHE_BINDING_KEYS:
        raise RuntimeError(f"[{context}_INVALID] exact cache binding schema mismatch")
    data = {str(key): value for key, value in binding.items()}
    for field in (
        "manifest_sha256",
        "cache_identity_sha256",
        "m5_prebuilt_source_sha256",
    ):
        _exact_sha256(data[field], context=context, field=field)
    cache_dir = _exact_directory(expected_cache_dir, context=context)
    if data.get("cache_dir") != str(cache_dir):
        raise RuntimeError(f"[{context}_INVALID] cache_dir differs from launch")
    manifest_path = _exact_regular_file(
        Path(str(data.get("manifest_path") or "")),
        label=f"{context}_manifest",
    )
    if manifest_path != cache_dir / "manifest.json":
        raise RuntimeError(f"[{context}_INVALID] manifest path differs from cache_dir")
    if _sha256_file(manifest_path) != data["manifest_sha256"]:
        raise RuntimeError(f"[{context}_INVALID] manifest SHA-256 mismatch")
    source_path = _exact_regular_file(
        Path(str(data.get("m5_prebuilt_source") or "")),
        label=f"{context}_source",
    )
    if _sha256_file(source_path) != data["m5_prebuilt_source_sha256"]:
        raise RuntimeError(f"[{context}_INVALID] M5 source SHA-256 mismatch")
    cache_manifest = _read_json_object(manifest_path, label=f"{context}_manifest")
    if (
        cache_manifest.get("schema_version") != HTF_V4_CACHE_SCHEMA_VERSION
        or cache_manifest.get("cache_identity_sha256")
        != data["cache_identity_sha256"]
        or cache_manifest.get("m5_prebuilt_source") != str(source_path)
        or cache_manifest.get("m5_prebuilt_source_sha256")
        != data["m5_prebuilt_source_sha256"]
    ):
        raise RuntimeError(f"[{context}_INVALID] cache manifest lineage mismatch")
    return dict(data)  # type: ignore[arg-type]


def require_manifest_bound_multi_tf_v4_cache(
    manifest_path: Path,
    *,
    dataset_run_id: str,
    cache_dir: Path,
    cache: MultiTFV4DiskCache,
    context: str,
) -> dict[str, str]:
    """Prove dataset declaration, launch path and loaded cache are identical."""

    exact_manifest = _exact_regular_file(manifest_path, label=f"{context}_dataset")
    manifest = _read_json_object(exact_manifest, label=f"{context}_dataset")
    binding = require_dataset_manifest_multi_tf_cache_binding(
        manifest,
        dataset_run_id=dataset_run_id,
        context=context,
    )
    data = require_multi_tf_v4_cache_binding_files(
        binding,
        expected_cache_dir=cache_dir,
        context=context,
    )
    if (
        not isinstance(cache, MultiTFV4DiskCache)
        or tuple(cache) != EXPECTED_TFS
        or cache.manifest_sha256 != data["manifest_sha256"]
        or cache.cache_identity_sha256 != data["cache_identity_sha256"]
        or cache.m5_prebuilt_source != data["m5_prebuilt_source"]
        or cache.m5_prebuilt_source_sha256
        != data["m5_prebuilt_source_sha256"]
    ):
        raise RuntimeError(f"[{context}_INVALID] loaded cache differs from dataset")
    return data


def _as_utc_train_times_ns(values: Any, *, expected_rows: int) -> np.ndarray:
    try:
        index = (
            values
            if isinstance(values, pd.DatetimeIndex)
            else pd.DatetimeIndex(values)
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_TRAIN_TIMES_INVALID]"
        ) from exc
    if index.tz is None:
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_TRAIN_TIMES_NOT_UTC]")
    index = index.tz_convert("UTC")
    timestamps = np.asarray(index.asi8, dtype=np.int64)
    if (
        timestamps.shape != (int(expected_rows),)
        or timestamps.size < 2
        or np.any(np.diff(timestamps) <= 0)
    ):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_TRAIN_TIMES_INVALID]"
        )
    return np.ascontiguousarray(timestamps)


def _timestamp_iso_utc(timestamp_ns: int) -> str:
    return pd.Timestamp(int(timestamp_ns), unit="ns", tz="UTC").isoformat()


def _hash_int64_indices(indices: np.ndarray, *, namespace: str) -> str:
    values = np.ascontiguousarray(indices, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(b"entry_v10_normalization_selected_indices_v1\0")
    digest.update(str(namespace).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray([len(values)], dtype="<i8").tobytes())
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _hash_selected_mtf_rows(
    *,
    tf: str,
    indices: np.ndarray,
    timestamps_ns: np.ndarray,
    values: np.ndarray,
    field_names: Sequence[str],
) -> str:
    """Hash only consumed MTF evidence, not unselected future cache rows."""

    selected_indices = np.ascontiguousarray(indices, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(b"entry_v10_normalization_selected_mtf_rows_v1\0")
    digest.update(str(tf).encode("ascii"))
    digest.update(b"\0")
    digest.update(bytes.fromhex(_field_names_sha256(field_names)))
    digest.update(
        np.asarray(
            [selected_indices.size, int(values.shape[1])],
            dtype="<i8",
        ).tobytes()
    )
    for start in range(0, int(selected_indices.size), DEFAULT_ROW_CHUNK):
        stop = min(int(selected_indices.size), start + DEFAULT_ROW_CHUNK)
        block_indices = selected_indices[start:stop]
        digest.update(block_indices.tobytes(order="C"))
        digest.update(
            np.ascontiguousarray(
                timestamps_ns[block_indices], dtype="<i8"
            ).tobytes()
        )
        digest.update(
            np.ascontiguousarray(
                values[block_indices], dtype="<f4"
            ).tobytes()
        )
    return digest.hexdigest()


def _hash_train_decision_rows(
    *,
    train_times_ns: np.ndarray,
    snap: np.ndarray,
    ctx_cont: np.ndarray,
    ctx_cat: np.ndarray,
    row_chunk: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"entry_v10_normalization_full_train_decision_rows_v1\0")
    digest.update(np.asarray(snap.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(ctx_cont.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(ctx_cat.shape, dtype="<i8").tobytes())
    for start in range(0, int(snap.shape[0]), int(row_chunk)):
        stop = min(int(snap.shape[0]), start + int(row_chunk))
        digest.update(
            np.ascontiguousarray(train_times_ns[start:stop], dtype="<i8").tobytes()
        )
        digest.update(
            np.ascontiguousarray(snap[start:stop], dtype="<f4").tobytes()
        )
        digest.update(
            np.ascontiguousarray(ctx_cont[start:stop], dtype="<f4").tobytes()
        )
        digest.update(
            np.ascontiguousarray(ctx_cat[start:stop], dtype="<i8").tobytes()
        )
    return digest.hexdigest()


def _derive_temporal_aliases(
    ordered_signal_names: Sequence[str],
) -> list[dict[str, Any]]:
    routing = json.loads(
        json.dumps(MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT)
    )
    routing["temporal_alias_policy"] = model_native_context_temporal_alias_policy(
        ordered_signal_names
    )
    validated = require_model_native_context_specialist_routing(
        routing,
        ordered_signal_names=ordered_signal_names,
        context="ENTRY_INPUT_NORMALIZATION_FIT",
    )
    aliases = validated["temporal_alias_policy"]["aliases"]
    if not isinstance(aliases, list):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_ALIASES_INVALID]")
    return [dict(alias) for alias in aliases]


def _validate_full_train_inputs(
    *,
    train_seq: Any,
    train_snap: Any,
    train_ctx_cont: Any,
    train_ctx_cat: Any,
    ordered_signal_names: Sequence[str],
    temporal_aliases: Sequence[Mapping[str, Any]],
    row_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(row_chunk, bool) or not isinstance(
        row_chunk, (int, np.integer)
    ) or int(row_chunk) < 1:
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_ROW_CHUNK_INVALID]")
    seq = np.asarray(train_seq)
    snap = np.asarray(train_snap)
    ctx_cont = np.asarray(train_ctx_cont)
    ctx_cat = np.asarray(train_ctx_cat)
    names = [str(name) for name in ordered_signal_names]
    if (
        seq.dtype != np.dtype(np.float32)
        or snap.dtype != np.dtype(np.float32)
        or ctx_cont.dtype != np.dtype(np.float32)
        or not np.issubdtype(ctx_cat.dtype, np.integer)
    ):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_TRAIN_DTYPE_INVALID]")
    rows = int(seq.shape[0]) if seq.ndim == 3 else -1
    if (
        rows < 2
        or seq.shape != (rows, MODEL_NATIVE_SEQ_LEN, MODEL_NATIVE_SIGNAL_DIM)
        or snap.shape != (rows, MODEL_NATIVE_SIGNAL_DIM)
        or ctx_cont.shape != (rows, len(MODEL_NATIVE_CTX_CONT_FIELDS))
        or ctx_cat.shape != (rows, len(MODEL_NATIVE_CTX_CAT_FIELDS))
        or names != list(dict.fromkeys(names))
        or len(names) != MODEL_NATIVE_SIGNAL_DIM
    ):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_TRAIN_SHAPE_INVALID] "
            f"seq={seq.shape} snap={snap.shape} ctx_cont={ctx_cont.shape} "
            f"ctx_cat={ctx_cat.shape} fields={len(names)}"
        )

    alias_signal_indices = np.asarray(
        [int(alias["signal_index"]) for alias in temporal_aliases],
        dtype=np.int64,
    )
    alias_ctx_indices = np.asarray(
        [int(alias["ctx_cont_index"]) for alias in temporal_aliases],
        dtype=np.int64,
    )
    for start in range(0, rows, int(row_chunk)):
        stop = min(rows, start + int(row_chunk))
        seq_block = seq[start:stop]
        snap_block = snap[start:stop]
        ctx_block = ctx_cont[start:stop]
        if (
            not np.isfinite(seq_block).all()
            or not np.isfinite(snap_block).all()
            or not np.isfinite(ctx_block).all()
        ):
            raise RuntimeError(
                "[ENTRY_INPUT_NORMALIZATION_TRAIN_NONFINITE] "
                f"rows={start}:{stop}"
            )
        if not np.array_equal(seq_block[:, -1, :], snap_block):
            raise RuntimeError(
                "[ENTRY_INPUT_NORMALIZATION_SEQ_LAST_SNAP_MISMATCH] "
                f"rows={start}:{stop}"
            )
        if alias_signal_indices.size and not np.array_equal(
            snap_block[:, alias_signal_indices],
            ctx_block[:, alias_ctx_indices],
        ):
            raise RuntimeError(
                "[ENTRY_INPUT_NORMALIZATION_ALIAS_VALUE_MISMATCH] "
                f"rows={start}:{stop}"
            )
    return seq, snap, ctx_cont, ctx_cat


def _load_entry_m5_source_times(path: Path) -> np.ndarray:
    try:
        frame = pd.read_parquet(path, columns=["time"])
    except Exception as exc:
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_M5_TIME_READ_FAILED]"
        ) from exc
    parsed = pd.DatetimeIndex(
        pd.to_datetime(frame["time"], utc=True, errors="coerce")
    )
    values = np.asarray(parsed.asi8, dtype=np.int64)
    if (
        values.size < MODEL_NATIVE_SEQ_LEN
        or parsed.hasnans
        or np.any(np.diff(values) <= 0)
    ):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_M5_TIME_INDEX_INVALID]"
        )
    return values


def _update_physical_rows_hash(
    digest: Any,
    *,
    physical_indices: np.ndarray,
    timestamps_ns: np.ndarray,
    values: np.ndarray,
) -> None:
    indices = np.asarray(physical_indices, dtype=np.int64)
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    matrix = np.asarray(values)
    if (
        indices.ndim != 1
        or timestamps.shape != indices.shape
        or matrix.ndim != 2
        or matrix.shape[0] != indices.size
    ):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_PHYSICAL_HASH_INPUT_INVALID]"
        )
    for start in range(0, int(indices.size), DEFAULT_ROW_CHUNK):
        stop = min(int(indices.size), start + DEFAULT_ROW_CHUNK)
        digest.update(
            np.ascontiguousarray(indices[start:stop], dtype="<i8").tobytes()
        )
        digest.update(
            np.ascontiguousarray(timestamps[start:stop], dtype="<i8").tobytes()
        )
        digest.update(
            np.ascontiguousarray(matrix[start:stop], dtype="<f4").tobytes()
        )


def _select_entry_local_population(
    *,
    train_seq: np.ndarray,
    train_snap: np.ndarray,
    train_times_ns: np.ndarray,
    m5_source_times_ns: np.ndarray,
    signal_names: Sequence[str],
) -> tuple[list[MatrixPopulationPart], dict[str, Any]]:
    """Select each physical M5 row consumed by any Entry window once."""

    source_positions = np.searchsorted(
        m5_source_times_ns,
        train_times_ns,
        side="left",
    ).astype(np.int64, copy=False)
    if (
        np.any(source_positions >= len(m5_source_times_ns))
        or not np.array_equal(
            m5_source_times_ns[source_positions], train_times_ns
        )
        or np.any(np.diff(source_positions) <= 0)
    ):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_ENTRY_M5_POINTER_INVALID]"
        )
    left_edges = source_positions - MODEL_NATIVE_SEQ_LEN + 1
    if int(left_edges[0]) < 0:
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_ENTRY_M5_HISTORY_INSUFFICIENT]"
        )

    digest = hashlib.sha256()
    digest.update(b"entry_exit_shared_local_entry_m5_rows_v1\0")
    digest.update(bytes.fromhex(_field_names_sha256(signal_names)))
    parts: list[MatrixPopulationPart] = []
    singleton_train_rows: list[int] = []
    singleton_physical_rows: list[int] = []
    non_single_hash_blocks: list[tuple[np.ndarray, np.ndarray]] = []
    merged_intervals: list[tuple[int, int]] = []
    covered_right = -1
    for train_row, (left, right) in enumerate(
        zip(left_edges.tolist(), source_positions.tolist())
    ):
        new_left = max(int(left), covered_right + 1)
        if new_left > int(right):
            continue
        local_start = new_left - int(left)
        physical = np.arange(new_left, int(right) + 1, dtype=np.int64)
        values = train_seq[train_row, local_start:]
        if values.shape[0] != physical.size:
            raise RuntimeError(
                "[ENTRY_INPUT_NORMALIZATION_ENTRY_LOCAL_SELECTION_INVALID]"
            )
        if not merged_intervals or new_left > merged_intervals[-1][1]:
            merged_intervals.append((new_left, int(right) + 1))
        else:
            merged_intervals[-1] = (
                merged_intervals[-1][0],
                max(merged_intervals[-1][1], int(right) + 1),
            )
        if local_start == MODEL_NATIVE_SEQ_LEN - 1:
            singleton_train_rows.append(train_row)
            singleton_physical_rows.append(int(right))
        else:
            non_single_hash_blocks.append((physical, values))
            parts.append(
                MatrixPopulationPart(
                    train_seq[train_row],
                    row_indices=np.arange(
                        local_start,
                        MODEL_NATIVE_SEQ_LEN,
                        dtype=np.int64,
                    ),
                    source=f"entry_m5_window_{train_row}",
                )
            )
        covered_right = int(right)
    for physical, values in non_single_hash_blocks:
        _update_physical_rows_hash(
            digest,
            physical_indices=physical,
            timestamps_ns=m5_source_times_ns[physical],
            values=values,
        )
    if singleton_train_rows:
        singleton_rows = np.asarray(singleton_train_rows, dtype=np.int64)
        singleton_physical = np.asarray(
            singleton_physical_rows,
            dtype=np.int64,
        )
        for start in range(0, int(singleton_rows.size), DEFAULT_ROW_CHUNK):
            stop = min(int(singleton_rows.size), start + DEFAULT_ROW_CHUNK)
            selected_train = singleton_rows[start:stop]
            selected_physical = singleton_physical[start:stop]
            _update_physical_rows_hash(
                digest,
                physical_indices=selected_physical,
                timestamps_ns=m5_source_times_ns[selected_physical],
                values=train_snap[selected_train],
            )
        parts.append(
            MatrixPopulationPart(
                train_snap,
                row_indices=singleton_rows,
                source="entry_m5_decision_snapshots",
            )
        )
    selected_count = sum(right - left for left, right in merged_intervals)
    selected_indices = np.empty(selected_count, dtype=np.int64)
    selected_offset = 0
    for left, right in merged_intervals:
        count = right - left
        selected_indices[selected_offset : selected_offset + count] = np.arange(
            left,
            right,
            dtype=np.int64,
        )
        selected_offset += count
    if (
        selected_offset != selected_count
        or selected_indices.size < MODEL_NATIVE_SEQ_LEN
        or np.any(np.diff(selected_indices) <= 0)
    ):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_ENTRY_LOCAL_POPULATION_INVALID]"
        )
    proof = {
        "route": "entry_m5_local",
        "selection": "union_of_entry_train_windows_each_physical_m5_row_once",
        "sequence_bars": MODEL_NATIVE_SEQ_LEN,
        "decision_row_count": int(train_times_ns.size),
        "selected_unique_row_count": int(selected_indices.size),
        "selected_row_indices_sha256": _hash_int64_indices(
            selected_indices,
            namespace="entry_local_m5_rows",
        ),
        "selected_row_values_sha256": digest.hexdigest(),
        "time_min_utc": _timestamp_iso_utc(
            m5_source_times_ns[selected_indices[0]]
        ),
        "time_max_utc": _timestamp_iso_utc(
            m5_source_times_ns[selected_indices[-1]]
        ),
    }
    proof["selection_proof_sha256"] = _canonical_sha256(proof)
    return parts, proof


def _hash_selected_surface_rows(
    *,
    namespace: str,
    matrix: np.ndarray,
    row_indices: np.ndarray,
    timestamps_ns: np.ndarray,
    field_names: Sequence[str],
    dtype: str,
) -> str:
    indices = np.asarray(row_indices, dtype=np.int64)
    source_times = np.asarray(timestamps_ns, dtype=np.int64)
    values = np.asarray(matrix)
    if (
        indices.ndim != 1
        or indices.size < 1
        or np.any(np.diff(indices) <= 0)
        or int(indices[0]) < 0
        or int(indices[-1]) >= values.shape[0]
        or source_times.shape != (values.shape[0],)
    ):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_SELECTED_SURFACE_HASH_INVALID]"
        )
    digest = hashlib.sha256()
    digest.update(namespace.encode("utf-8") + b"\0")
    digest.update(bytes.fromhex(_field_names_sha256(field_names)))
    for start in range(0, int(indices.size), DEFAULT_ROW_CHUNK):
        stop = min(int(indices.size), start + DEFAULT_ROW_CHUNK)
        selected = indices[start:stop]
        digest.update(np.ascontiguousarray(selected, dtype="<i8").tobytes())
        digest.update(
            np.ascontiguousarray(source_times[selected], dtype="<i8").tobytes()
        )
        digest.update(
            np.ascontiguousarray(values[selected], dtype=dtype).tobytes()
        )
    return digest.hexdigest()


_MTF_PER_BAR_CONTRACTS = {
    HTF_V4_MATRIX_CONTRACT: (MULTI_TF_FEATURE_COUNT_V4, MULTI_TF_PER_BAR_FEATURES_V4),
}


def resolve_mtf_per_bar_contract(source: object, *, tf: str) -> tuple[int, tuple]:
    """Return (width, ordered names) for the contract this source declares.

    Normalization must be fitted over the surface the model reads. Pinning these
    to V2 is what held the higher timeframes at 25 generic features, so the
    source declares its contract and this resolves it; an undeclared or unknown
    contract fails closed rather than defaulting.
    """
    declared = getattr(source, "attrs", {}).get("htf_feature_contract")
    if declared not in _MTF_PER_BAR_CONTRACTS:
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_MTF_CONTRACT_UNKNOWN] tf={tf} "
            f"declared={declared!r} known={sorted(_MTF_PER_BAR_CONTRACTS)}"
        )
    return _MTF_PER_BAR_CONTRACTS[declared]


def _extract_mtf_source(
    source: Any,
    *,
    tf: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    if not isinstance(source, pd.DataFrame):
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_MTF_SOURCE_INVALID] tf={tf}"
        )
    _mtf_count, _mtf_names = resolve_mtf_per_bar_contract(source, tf=tf)
    if (
        list(source.columns) != list(_mtf_names)
        or int(source.shape[1]) != _mtf_count
    ):
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_MTF_FIELDS_INVALID] tf={tf}"
        )
    timestamps = np.asarray(source.attrs.get("ts_int64"))
    values = np.asarray(source.attrs.get("feats_np"))
    warmup = source.attrs.get("causal_warmup_rows")
    if (
        timestamps.dtype != np.dtype(np.int64)
        or timestamps.shape != (len(source),)
        or values.dtype != np.dtype(np.float32)
        or values.shape != (len(source), _mtf_count)
        or len(source) < 2
        or np.any(np.diff(timestamps) <= 0)
        or isinstance(warmup, bool)
        or not isinstance(warmup, (int, np.integer))
        or not 0 <= int(warmup) < len(source)
    ):
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_MTF_SOURCE_INVALID] tf={tf}"
        )
    return (
        np.ascontiguousarray(timestamps, dtype=np.int64),
        values,
        int(warmup),
    )


def select_causal_mtf_fit_population(
    *,
    tf: str,
    source: pd.DataFrame,
    train_times_ns: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Compatibility wrapper for one Entry-clock route.

    Canonical fitting uses :func:`select_shared_causal_mtf_fit_population`.
    This wrapper remains a narrow deterministic unit-test helper.
    """

    population, window, proof = select_shared_causal_mtf_fit_population(
        tf=tf,
        source=source,
        entry_train_times_ns=np.asarray(train_times_ns),
        exit_train_times_ns=np.asarray([], dtype=np.int64),
        seq_len=seq_len,
        entry_route_timeframes=EXPECTED_TFS,
        require_exit_route=False,
    )
    matrix = np.asarray(population.values)
    indices = np.asarray(population.row_indices, dtype=np.int64)
    return (
        np.ascontiguousarray(matrix[indices], dtype=np.float32),
        window,
        proof,
    )


def _validate_route_times(
    values: Any,
    *,
    route: str,
    allow_empty: bool,
) -> np.ndarray:
    times = np.asarray(values)
    if times.dtype != np.dtype(np.int64) or times.ndim != 1:
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_ROUTE_TIMES_INVALID] route={route}"
        )
    if times.size == 0 and allow_empty:
        return times
    if times.size < 1 or np.any(np.diff(times) <= 0):
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_ROUTE_TIMES_INVALID] route={route}"
        )
    return times


def select_shared_causal_mtf_fit_population(
    *,
    tf: str,
    source: pd.DataFrame,
    entry_train_times_ns: np.ndarray,
    exit_train_times_ns: np.ndarray,
    seq_len: int,
    entry_route_timeframes: Sequence[str] = ENTRY_MTF_CONTEXT_TIMEFRAMES,
    require_exit_route: bool = True,
) -> tuple[MatrixPopulationPart, dict[str, Any], dict[str, Any]]:
    """Union actual Entry +5m and Exit +1m TRAIN cache consumption."""

    if tf not in EXPECTED_TFS:
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_MTF_TF_INVALID] tf={tf}"
        )
    if isinstance(seq_len, bool) or not isinstance(
        seq_len, (int, np.integer)
    ) or int(seq_len) < 1:
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_MTF_SEQ_LEN_INVALID] tf={tf}"
        )
    entry_tfs = tuple(str(value) for value in entry_route_timeframes)
    if (
        len(entry_tfs) != len(set(entry_tfs))
        or any(value not in EXPECTED_TFS for value in entry_tfs)
    ):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_ENTRY_MTF_ROUTE_INVALID]"
        )
    entry_ts = _validate_route_times(
        entry_train_times_ns,
        route="entry",
        allow_empty=tf not in entry_tfs,
    )
    exit_ts = _validate_route_times(
        exit_train_times_ns,
        route="exit",
        allow_empty=not require_exit_route,
    )
    source_ts, source_values, warmup = _extract_mtf_source(source, tf=tf)
    shift_ns = int(MULTI_TF_SHIFT[tf].value)
    coverage_delta = np.zeros(len(source_ts) + 1, dtype=np.int64)
    route_proofs: dict[str, dict[str, Any]] = {}
    route_specs = (
        (
            "entry",
            entry_ts if tf in entry_tfs else np.asarray([], dtype=np.int64),
            ENTRY_TARGET_AVAILABILITY_SHIFT,
            tf in entry_tfs,
        ),
        (
            "exit",
            exit_ts,
            EXIT_TARGET_AVAILABILITY_SHIFT,
            tf in EXIT_MTF_CONTEXT_TIMEFRAMES and exit_ts.size > 0,
        ),
    )
    for route, decision_ts, availability, enabled in route_specs:
        if not enabled:
            route_proofs[route] = {
                "enabled": False,
                "decision_row_count": 0,
                "target_availability_shift_seconds": int(
                    availability.total_seconds()
                ),
                "selected_unique_row_count": 0,
                "selected_row_indices_sha256": _hash_int64_indices(
                    np.asarray([], dtype=np.int64),
                    namespace=f"mtf:{tf}:{route}",
                ),
            }
            continue
        availability_ns = int(availability.value)
        if np.any(decision_ts > np.iinfo(np.int64).max - availability_ns):
            raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_TIME_OVERFLOW]")
        cutoffs = decision_ts + availability_ns - shift_ns
        right = np.searchsorted(source_ts, cutoffs, side="right").astype(
            np.int64,
            copy=False,
        )
        left = right - int(seq_len)
        invalid = np.flatnonzero(
            (right < int(seq_len)) | (left < int(warmup))
        )
        if invalid.size:
            row = int(invalid[0])
            raise RuntimeError(
                "[ENTRY_INPUT_NORMALIZATION_MTF_WARMUP_INCOMPLETE] "
                f"tf={tf} route={route} train_row={row} "
                f"left={int(left[row])} right={int(right[row])} "
                f"warmup={warmup} seq_len={int(seq_len)}"
            )
        route_delta = np.zeros(len(source_ts) + 1, dtype=np.int64)
        np.add.at(route_delta, left, 1)
        np.add.at(route_delta, right, -1)
        route_indices = np.flatnonzero(
            np.cumsum(route_delta[:-1], dtype=np.int64) > 0
        ).astype(np.int64, copy=False)
        np.add.at(coverage_delta, left, 1)
        np.add.at(coverage_delta, right, -1)
        route_proofs[route] = {
            "enabled": True,
            "decision_row_count": int(decision_ts.size),
            "decision_times_sha256": _hash_int64_indices(
                decision_ts,
                namespace=f"mtf:{tf}:{route}:decision_times",
            ),
            "target_availability_shift_seconds": int(
                availability.total_seconds()
            ),
            "selected_unique_row_count": int(route_indices.size),
            "selected_row_indices_sha256": _hash_int64_indices(
                route_indices,
                namespace=f"mtf:{tf}:{route}",
            ),
            "selected_row_values_sha256": _hash_selected_mtf_rows(
                tf=f"{tf}:{route}",
                indices=route_indices,
                timestamps_ns=source_ts,
                values=source_values,
                field_names=resolve_mtf_per_bar_contract(source, tf=tf)[1],
            ),
        }
    selected_indices = np.flatnonzero(
        np.cumsum(coverage_delta[:-1], dtype=np.int64) > 0
    ).astype(np.int64, copy=False)
    if selected_indices.size < 2:
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_MTF_POPULATION_EMPTY] tf={tf}"
        )
    _sel_count, _sel_names = resolve_mtf_per_bar_contract(source, tf=tf)
    ema_stack_index = list(_sel_names).index("ema_stack_aligned_v2")
    regime_index = list(_sel_names).index("regime_class_id")
    for start in range(0, int(selected_indices.size), DEFAULT_ROW_CHUNK):
        block_indices = selected_indices[
            start : start + DEFAULT_ROW_CHUNK
        ]
        block = np.asarray(source_values[block_indices], dtype=np.float32)
        if not np.isfinite(block).all():
            raise RuntimeError(
                f"[ENTRY_INPUT_NORMALIZATION_MTF_SELECTED_NONFINITE] tf={tf}"
            )
        if not np.isin(block[:, ema_stack_index], (-1.0, 0.0, 1.0)).all():
            raise RuntimeError(
                f"[ENTRY_INPUT_NORMALIZATION_MTF_EMA_STACK_DOMAIN_INVALID] tf={tf}"
            )
        if not np.isin(
            block[:, regime_index],
            MTF_SEMANTIC_CATEGORICAL_DOMAINS["regime_class_id"],
        ).all():
            raise RuntimeError(
                f"[ENTRY_INPUT_NORMALIZATION_MTF_REGIME_DOMAIN_INVALID] tf={tf}"
            )

    indices_hash = _hash_int64_indices(selected_indices, namespace=f"mtf:{tf}")
    values_hash = _hash_selected_mtf_rows(
        tf=tf,
        indices=selected_indices,
        timestamps_ns=source_ts,
        values=source_values,
        field_names=_sel_names,
    )
    window = {
        "left_index_inclusive": int(selected_indices[0]),
        "right_index_exclusive": int(selected_indices[-1]) + 1,
        "selected_unique_row_count": int(selected_indices.size),
        "selected_row_indices_sha256": indices_hash,
        "selected_row_values_sha256": values_hash,
        "time_min_utc": _timestamp_iso_utc(source_ts[selected_indices[0]]),
        "time_max_utc": _timestamp_iso_utc(source_ts[selected_indices[-1]]),
    }
    proof = {
        "tf": tf,
        "selection": (
            "union_of_entry_plus5_exit_plus1_train_windows_each_cache_row_once"
        ),
        "target_availability_shift_seconds": None,
        "route_target_availability_shift_seconds": {
            "entry": int(ENTRY_TARGET_AVAILABILITY_SHIFT.total_seconds()),
            "exit": int(EXIT_TARGET_AVAILABILITY_SHIFT.total_seconds()),
        },
        "tf_shift_seconds": int(MULTI_TF_SHIFT[tf].total_seconds()),
        "seq_len": int(seq_len),
        "source_row_count": int(len(source_ts)),
        "source_warmup_rows": int(warmup),
        "routes": route_proofs,
        **window,
    }
    proof["selection_proof_sha256"] = _canonical_sha256(proof)
    return (
        MatrixPopulationPart(
            source_values,
            row_indices=selected_indices,
            source=f"shared_v4_mtf_{tf.lower()}",
        ),
        window,
        proof,
    )


def _verify_artifacts_and_load_mtf(
    *,
    artifacts: TrainNormalizationArtifacts,
    ordered_signal_names: Sequence[str],
) -> tuple[dict[str, Any], MultiTFV4DiskCache]:
    dataset_run_id = str(artifacts.dataset_run_id).strip()
    if not dataset_run_id:
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_DATASET_RUN_ID_MISSING]"
        )
    train_parquet = _exact_regular_file(
        artifacts.train_parquet_path, label="train_parquet"
    )
    train_manifest = _exact_regular_file(
        artifacts.train_manifest_path, label="train_manifest"
    )
    m5_prebuilt = _exact_regular_file(
        artifacts.m5_prebuilt_path, label="m5_prebuilt"
    )
    manifest = _read_json_object(train_manifest, label="train_manifest")
    feature_contract = manifest.get("feature_contract")
    manifest_run_id = (
        manifest.get("extra", {}).get("entry_run_id")
        if isinstance(manifest.get("extra"), Mapping)
        else None
    )
    try:
        declared_output = Path(str(manifest["output_data_path"])).resolve(
            strict=True
        )
    except (KeyError, OSError) as exc:
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_TRAIN_MANIFEST_INVALID]"
        ) from exc
    if (
        not isinstance(feature_contract, Mapping)
        or manifest_run_id != dataset_run_id
        or declared_output != train_parquet
        or feature_contract.get("signal_bridge_fields")
        != list(ordered_signal_names)
        or feature_contract.get("ctx_cont_names")
        != list(MODEL_NATIVE_CTX_CONT_FIELDS)
        or feature_contract.get("ctx_cat_names")
        != list(MODEL_NATIVE_CTX_CAT_FIELDS)
    ):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_TRAIN_MANIFEST_MISMATCH]"
        )

    cache_dir = Path(artifacts.mtf_cache_dir).expanduser()
    if not cache_dir.is_absolute():
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_MTF_CACHE_PATH_INVALID]"
        )
    cache = load_multi_tf_v4_cache(cache_dir)
    if not isinstance(cache, MultiTFV4DiskCache):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_MTF_CACHE_IDENTITY_MISSING]"
        )
    cache_binding = require_manifest_bound_multi_tf_v4_cache(
        train_manifest,
        dataset_run_id=dataset_run_id,
        cache_dir=cache_dir,
        cache=cache,
        context="ENTRY_INPUT_NORMALIZATION_MTF_BINDING",
    )
    # The cache's full-history M5 source and the model-range seq/snapshot
    # source are deliberately distinct identities, both now bound exactly.
    m5_sha256 = _sha256_file(m5_prebuilt)
    cache_manifest = Path(cache_binding["manifest_path"])
    manifest_sha256 = cache_binding["manifest_sha256"]
    _mtf_manifest_declared = json.loads(cache_manifest.read_text(encoding="utf-8"))
    for _key in ("builder_version", "feature_names", "feature_count"):
        if _key not in _mtf_manifest_declared:
            raise RuntimeError(
                f"[ENTRY_INPUT_NORMALIZATION_MTF_MANIFEST_INCOMPLETE] missing {_key}"
            )
    if int(_mtf_manifest_declared["feature_count"]) != len(
        _mtf_manifest_declared["feature_names"]
    ):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_MTF_MANIFEST_WIDTH_MISMATCH]"
        )
    base_lineage = {
        "dataset_run_id": dataset_run_id,
        "train_parquet_path": str(train_parquet),
        "train_parquet_sha256": _sha256_file(train_parquet),
        "train_manifest_path": str(train_manifest),
        "train_manifest_sha256": _sha256_file(train_manifest),
        "m5_prebuilt_path": str(m5_prebuilt),
        "m5_prebuilt_sha256": m5_sha256,
        "mtf_cache_manifest_path": str(cache_manifest),
        "mtf_cache_manifest_sha256": manifest_sha256,
        # Builder version and field-name hash come from the manifest the cache
        # actually published, not from whichever contract this module imports.
        # Recording V2's names beside a V3 cache would be a lineage that lies.
        "mtf_builder_version": _mtf_manifest_declared["builder_version"],
        "mtf_feature_names_sha256": _field_names_sha256(
            tuple(_mtf_manifest_declared["feature_names"])
        ),
    }
    return base_lineage, cache


def _validate_exit_train_population(
    lifecycle: Any,
    *,
    temporal_aliases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not hasattr(lifecycle, "train_normalization_population"):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_EXIT_LIFECYCLE_REQUIRED]"
        )
    raw = lifecycle.train_normalization_population()
    expected_keys = {
        "signal",
        "ctx_cont",
        "ctx_cat",
        "local_row_indices",
        "current_row_indices",
        "current_decision_times_ns",
        "source_times_ns",
        "local_merged_intervals",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_EXIT_POPULATION_SCHEMA_INVALID]"
        )
    signal = np.asarray(raw["signal"])
    ctx_cont = np.asarray(raw["ctx_cont"])
    ctx_cat = np.asarray(raw["ctx_cat"])
    local_indices = np.asarray(raw["local_row_indices"])
    current_indices = np.asarray(raw["current_row_indices"])
    current_times = np.asarray(raw["current_decision_times_ns"])
    source_times = np.asarray(raw["source_times_ns"])
    rows = int(signal.shape[0]) if signal.ndim == 2 else -1
    if (
        signal.dtype != np.dtype(np.float32)
        or ctx_cont.dtype != np.dtype(np.float32)
        or not np.issubdtype(ctx_cat.dtype, np.integer)
        or signal.shape != (rows, MODEL_NATIVE_SIGNAL_DIM)
        or ctx_cont.shape != (rows, len(MODEL_NATIVE_CTX_CONT_FIELDS))
        or ctx_cat.shape != (rows, len(MODEL_NATIVE_CTX_CAT_FIELDS))
        or local_indices.dtype != np.dtype(np.int64)
        or current_indices.dtype != np.dtype(np.int64)
        or current_times.dtype != np.dtype(np.int64)
        or source_times.dtype != np.dtype(np.int64)
        or local_indices.ndim != 1
        or current_indices.ndim != 1
        or current_times.shape != current_indices.shape
        or source_times.shape != (rows,)
        or local_indices.size < MODEL_NATIVE_SEQ_LEN
        or current_indices.size < 1
        or int(local_indices[0]) < 0
        or int(local_indices[-1]) >= rows
        or int(current_indices[0]) < 0
        or int(current_indices[-1]) >= rows
        or np.any(np.diff(local_indices) <= 0)
        or np.any(np.diff(current_indices) <= 0)
        or np.any(np.diff(current_times) <= 0)
        or np.any(np.diff(source_times) <= 0)
        or not np.array_equal(source_times[current_indices], current_times)
        or not np.isin(current_indices, local_indices).all()
    ):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_EXIT_POPULATION_INVALID]"
        )
    alias_signal_indices = np.asarray(
        [int(alias["signal_index"]) for alias in temporal_aliases],
        dtype=np.int64,
    )
    alias_ctx_indices = np.asarray(
        [int(alias["ctx_cont_index"]) for alias in temporal_aliases],
        dtype=np.int64,
    )
    for start in range(0, int(current_indices.size), DEFAULT_ROW_CHUNK):
        selected = current_indices[start : start + DEFAULT_ROW_CHUNK]
        if (
            not np.isfinite(signal[selected]).all()
            or not np.isfinite(ctx_cont[selected]).all()
            or (
                alias_signal_indices.size
                and not np.array_equal(
                    signal[selected][:, alias_signal_indices],
                    ctx_cont[selected][:, alias_ctx_indices],
                )
            )
        ):
            raise RuntimeError(
                "[ENTRY_INPUT_NORMALIZATION_EXIT_ALIAS_OR_FINITE_INVALID]"
            )
    return {
        "signal": signal,
        "ctx_cont": ctx_cont,
        "ctx_cat": ctx_cat,
        "local_row_indices": local_indices,
        "current_row_indices": current_indices,
        "current_decision_times_ns": current_times,
        "source_times_ns": source_times,
        "local_merged_intervals": tuple(raw["local_merged_intervals"]),
    }


def fit_entry_v10_train_input_normalization(
    *,
    train_seq: Any,
    train_snap: Any,
    train_ctx_cont: Any,
    train_ctx_cat: Any,
    train_times: Any,
    train_exit_lifecycle: Any,
    ordered_signal_names: Sequence[str],
    per_tf_seq_lens: Mapping[str, int],
    artifacts: TrainNormalizationArtifacts,
    row_chunk: int = DEFAULT_ROW_CHUNK,
) -> dict[str, Any]:
    """Fit and bind the exact full-TRAIN model-native input transform.

    The returned mapping contains the immutable contract plus an auxiliary
    population proof.  No file is written and no model/trainer is mutated.
    """

    signal_names = [str(name) for name in ordered_signal_names]
    if len(signal_names) != MODEL_NATIVE_SIGNAL_DIM:
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_SIGNAL_FIELDS_INVALID]"
        )
    if (
        not isinstance(per_tf_seq_lens, Mapping)
        or tuple(per_tf_seq_lens) != EXPECTED_TFS
        or any(
            isinstance(per_tf_seq_lens[tf], bool)
            or not isinstance(per_tf_seq_lens[tf], (int, np.integer))
            or int(per_tf_seq_lens[tf]) < 1
            for tf in EXPECTED_TFS
        )
    ):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_MTF_SEQ_LENS_INVALID]"
        )
    aliases = _derive_temporal_aliases(signal_names)
    seq, snap, ctx_cont, ctx_cat = _validate_full_train_inputs(
        train_seq=train_seq,
        train_snap=train_snap,
        train_ctx_cont=train_ctx_cont,
        train_ctx_cat=train_ctx_cat,
        ordered_signal_names=signal_names,
        temporal_aliases=aliases,
        row_chunk=row_chunk,
    )
    train_times_ns = _as_utc_train_times_ns(
        train_times, expected_rows=int(snap.shape[0])
    )
    base_lineage, multi_tf_sources = _verify_artifacts_and_load_mtf(
        artifacts=artifacts,
        ordered_signal_names=signal_names,
    )
    row_count = int(snap.shape[0])
    manifest = _read_json_object(
        Path(base_lineage["train_manifest_path"]), label="train_manifest"
    )
    split_times = manifest.get("ts_min_max_by_split", {}).get("train")
    if not isinstance(split_times, Mapping):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_TRAIN_MANIFEST_TIMES_MISSING]"
        )
    manifest_min = pd.Timestamp(split_times.get("ts_min"))
    manifest_max = pd.Timestamp(split_times.get("ts_max"))
    if (
        manifest_min.tzinfo is None
        or manifest_max.tzinfo is None
        or int(manifest_min.tz_convert("UTC").value) != int(train_times_ns[0])
        or int(manifest_max.tz_convert("UTC").value) != int(train_times_ns[-1])
    ):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_TRAIN_MANIFEST_TIMES_MISMATCH]"
        )

    exit_population = _validate_exit_train_population(
        train_exit_lifecycle,
        temporal_aliases=aliases,
    )
    m5_source_times_ns = _load_entry_m5_source_times(
        Path(base_lineage["m5_prebuilt_path"])
    )
    entry_local_parts, entry_local_proof = _select_entry_local_population(
        train_seq=seq,
        train_snap=snap,
        train_times_ns=train_times_ns,
        m5_source_times_ns=m5_source_times_ns,
        signal_names=signal_names,
    )
    exit_local_indices = exit_population["local_row_indices"]
    exit_current_indices = exit_population["current_row_indices"]
    exit_source_times_ns = exit_population["source_times_ns"]
    exit_decision_times_ns = exit_population["current_decision_times_ns"]
    exit_local_proof = {
        "route": "exit_m1_local",
        "selection": "union_of_exit_train_480_windows_each_physical_m1_row_once",
        "sequence_bars": 480,
        "decision_row_count": int(exit_current_indices.size),
        "selected_unique_row_count": int(exit_local_indices.size),
        "selected_row_indices_sha256": _hash_int64_indices(
            exit_local_indices,
            namespace="exit_local_m1_rows",
        ),
        "selected_row_values_sha256": _hash_selected_surface_rows(
            namespace="entry_exit_shared_local_exit_m1_rows_v1",
            matrix=exit_population["signal"],
            row_indices=exit_local_indices,
            timestamps_ns=exit_source_times_ns,
            field_names=signal_names,
            dtype="<f4",
        ),
        "time_min_utc": _timestamp_iso_utc(
            exit_source_times_ns[exit_local_indices[0]]
        ),
        "time_max_utc": _timestamp_iso_utc(
            exit_source_times_ns[exit_local_indices[-1]]
        ),
    }
    exit_local_proof["selection_proof_sha256"] = _canonical_sha256(
        exit_local_proof
    )
    local_parts = [
        *entry_local_parts,
        MatrixPopulationPart(
            exit_population["signal"],
            row_indices=exit_local_indices,
            source="exit_m1_local",
        ),
    ]
    context_cont_parts = [
        MatrixPopulationPart(ctx_cont, source="entry_m5_decisions"),
        MatrixPopulationPart(
            exit_population["ctx_cont"],
            row_indices=exit_current_indices,
            source="exit_m1_decisions",
        ),
    ]
    context_cat_parts = [
        MatrixPopulationPart(ctx_cat, source="entry_m5_decisions"),
        MatrixPopulationPart(
            exit_population["ctx_cat"],
            row_indices=exit_current_indices,
            source="exit_m1_decisions",
        ),
    ]
    local_row_count = int(
        entry_local_proof["selected_unique_row_count"]
    ) + int(exit_local_indices.size)
    context_row_count = row_count + int(exit_current_indices.size)

    signal_surface = fit_surface_normalization(
        local_parts,
        surface="signal",
        field_names=signal_names,
        row_count=local_row_count,
        semantic_categorical_domains={
            f"ctx_cont.{name}": domain
            for name, domain in (
                CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS.items()
            )
            if f"ctx_cont.{name}" in signal_names
        },
    )
    ctx_surface_raw = fit_surface_normalization(
        context_cont_parts,
        surface="ctx_cont",
        field_names=MODEL_NATIVE_CTX_CONT_FIELDS,
        row_count=context_row_count,
        semantic_categorical_domains=(
            CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS
        ),
    )
    ctx_surface = share_temporal_alias_stats_from_signal(
        ctx_surface_raw,
        signal_surface,
        temporal_aliases=aliases,
        ctx_cont_values=context_cont_parts,
    )
    surfaces: dict[str, Mapping[str, Any]] = {
        "signal": signal_surface,
        "ctx_cont": ctx_surface,
    }
    tf_windows: dict[str, dict[str, Any]] = {}
    tf_population_proofs: dict[str, dict[str, Any]] = {}
    for tf in EXPECTED_TFS:
        selected_population, window, selection_proof = (
            select_shared_causal_mtf_fit_population(
                tf=tf,
                source=multi_tf_sources[tf],
                entry_train_times_ns=train_times_ns,
                exit_train_times_ns=exit_decision_times_ns,
                seq_len=int(per_tf_seq_lens[tf]),
            )
        )
        surface_name = f"mtf_{tf.lower()}"
        _fit_count, _fit_names = resolve_mtf_per_bar_contract(
            multi_tf_sources[tf], tf=tf
        )
        surfaces[surface_name] = fit_surface_normalization(
            selected_population,
            surface=surface_name,
            field_names=_fit_names,
            row_count=int(window["selected_unique_row_count"]),
            semantic_categorical_domains=MTF_SEMANTIC_CATEGORICAL_DOMAINS,
        )
        tf_windows[tf] = window
        tf_population_proofs[tf] = selection_proof
    if tuple(surfaces) != EXPECTED_SURFACES:
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_SURFACE_ORDER_INVALID]"
        )

    fit_start_utc = _timestamp_iso_utc(
        min(int(train_times_ns[0]), int(exit_decision_times_ns[0]))
    )
    fit_end_utc = _timestamp_iso_utc(
        max(int(train_times_ns[-1]), int(exit_decision_times_ns[-1]))
    )
    lineage = {
        **base_lineage,
        "train_row_count": context_row_count,
        "entry_train_decision_row_count": row_count,
        "exit_train_decision_row_count": int(exit_current_indices.size),
        "local_fit_row_count": local_row_count,
        "context_fit_row_count": context_row_count,
        "val_fit_row_count": 0,
        "test_fit_row_count": 0,
        "train_time_min_utc": fit_start_utc,
        "train_time_max_utc": fit_end_utc,
        "per_tf_seq_lens": {
            tf: int(per_tf_seq_lens[tf]) for tf in EXPECTED_TFS
        },
        "per_tf_shift_seconds": {
            tf: int(MULTI_TF_SHIFT[tf].total_seconds()) for tf in EXPECTED_TFS
        },
        "per_tf_fit_windows": tf_windows,
    }
    ctx_cat_contract = fit_ctx_cat_contract(
        context_cat_parts,
        field_names=MODEL_NATIVE_CTX_CAT_FIELDS,
    )
    normalization_contract = build_input_normalization_contract(
        fit_start_utc=fit_start_utc,
        fit_end_utc=fit_end_utc,
        surfaces=surfaces,
        ctx_cat=ctx_cat_contract,
        lineage=lineage,
        temporal_aliases=aliases,
    )

    entry_train_indices = np.arange(row_count, dtype=np.int64)
    entry_context_values_sha256 = _hash_train_decision_rows(
        train_times_ns=train_times_ns,
        snap=snap,
        ctx_cont=ctx_cont,
        ctx_cat=ctx_cat,
        row_chunk=int(row_chunk),
    )
    exit_context_values_sha256 = _canonical_sha256(
        {
            "signal": _hash_selected_surface_rows(
                namespace="exit_m1_decision_signal_v1",
                matrix=exit_population["signal"],
                row_indices=exit_current_indices,
                timestamps_ns=exit_source_times_ns,
                field_names=signal_names,
                dtype="<f4",
            ),
            "ctx_cont": _hash_selected_surface_rows(
                namespace="exit_m1_decision_ctx_cont_v1",
                matrix=exit_population["ctx_cont"],
                row_indices=exit_current_indices,
                timestamps_ns=exit_source_times_ns,
                field_names=MODEL_NATIVE_CTX_CONT_FIELDS,
                dtype="<f4",
            ),
            "ctx_cat": _hash_selected_surface_rows(
                namespace="exit_m1_decision_ctx_cat_v1",
                matrix=exit_population["ctx_cat"],
                row_indices=exit_current_indices,
                timestamps_ns=exit_source_times_ns,
                field_names=MODEL_NATIVE_CTX_CAT_FIELDS,
                dtype="<i8",
            ),
        }
    )
    population_proof = {
        "schema_version": FIT_POPULATION_PROOF_SCHEMA_VERSION,
        "fit_scope": "train_only",
        "signal_population": (
            "union_unique_physical_entry_m5_and_exit_m1_local_rows"
        ),
        "ctx_cont_population": (
            "entry_train_decisions_plus_unique_exit_m1_current_decisions"
        ),
        "ctx_cat_population": (
            "entry_train_decisions_plus_unique_exit_m1_current_decisions"
        ),
        "sequence_population": (
            "physical_window_union_each_source_row_once"
        ),
        "train_decision_row_count": context_row_count,
        "entry_train_decision_row_count": row_count,
        "exit_train_decision_row_count": int(exit_current_indices.size),
        "local_fit_row_count": local_row_count,
        "context_fit_row_count": context_row_count,
        "train_decision_row_indices_sha256": _canonical_sha256(
            {
                "entry": _hash_int64_indices(
                    entry_train_indices,
                    namespace="entry_train_decision_rows",
                ),
                "exit": _hash_int64_indices(
                    exit_current_indices,
                    namespace="exit_train_decision_rows",
                ),
            }
        ),
        "train_decision_row_values_sha256": _canonical_sha256(
            {
                "entry": entry_context_values_sha256,
                "exit": exit_context_values_sha256,
            }
        ),
        "val_fit_row_count": 0,
        "test_fit_row_count": 0,
        "temporal_alias_count": len(aliases),
        "temporal_aliases_sha256": normalization_contract[
            "temporal_aliases_sha256"
        ],
        "local_populations": {
            "entry": entry_local_proof,
            "exit": exit_local_proof,
        },
        "context_populations": {
            "entry": {
                "decision_row_count": row_count,
                "decision_times_sha256": _hash_int64_indices(
                    train_times_ns,
                    namespace="entry_context_decision_times",
                ),
                "values_sha256": entry_context_values_sha256,
            },
            "exit": {
                "decision_row_count": int(exit_current_indices.size),
                "decision_times_sha256": _hash_int64_indices(
                    exit_decision_times_ns,
                    namespace="exit_context_decision_times",
                ),
                "values_sha256": exit_context_values_sha256,
            },
        },
        "mtf_populations": tf_population_proofs,
    }
    population_proof["proof_sha256"] = _canonical_sha256(population_proof)
    return {
        "schema_version": FIT_HELPER_SCHEMA_VERSION,
        "normalization_contract": normalization_contract,
        "fit_population_proof": population_proof,
    }
