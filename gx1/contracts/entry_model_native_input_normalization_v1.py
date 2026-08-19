"""Immutable shared TRAIN-only input normalization for Entry and Exit.

The model consumes raw, finite XAU feature tensors. This contract fits the
shared local encoder on the deduplicated physical M5+M1 TRAIN union, context on
the Entry+Exit TRAIN decision union, and MTF surfaces on actual +5m/+1m route
consumption. A field is scaled as a nominal category only when a contract owner
DECLARES its exact integral domain; every other field -- including a field whose
fit window happens to show two values -- is median-centered, robustly scaled,
then mapped by an invertible ``asinh``. No tail observation is clipped,
saturated or collapsed onto a boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DOMAINS,
)
from gx1.features.htf_features import (
    MULTI_TF_TIMEFRAMES,
)

SCHEMA_VERSION = "entry_model_native_input_normalization_v7"
TRANSFORM = "shared_entry_exit_train_only_median_raw_iqr_asinh_v4"
FIT_POPULATION = "unique_physical_train_rows_entry_exit_union_v2"
CONTINUOUS_TRANSFORM = "asinh_affine_invertible_non_saturating"
FIT_COLUMN_CHUNK = 32
FIT_ROW_CHUNK = 4096
FIT_MAX_WORKING_BLOCK_BYTES = 128 * 1024 * 1024
EXPECTED_SURFACES = (
    "signal",
    "ctx_cont",
    "mtf_m5",
    "mtf_m15",
    "mtf_h1",
    "mtf_h4",
    "mtf_d1",
)
EXPECTED_TFS = MULTI_TF_TIMEFRAMES
CTX_CAT_DOMAINS = MODEL_NATIVE_CTX_CAT_DOMAINS
SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS = {
    "smc_swing_state": (0, 1, 2, 3, 4),
}
CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS: dict[str, tuple[int, ...]] = {}
MTF_SEMANTIC_CATEGORICAL_DOMAINS: dict[str, tuple[int, ...]] = {}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LINEAGE_KEYS = {
    "dataset_run_id",
    "train_parquet_path",
    "train_parquet_sha256",
    "train_manifest_path",
    "train_manifest_sha256",
    "train_row_count",
    "entry_train_decision_row_count",
    "exit_train_decision_row_count",
    "local_fit_row_count",
    "context_fit_row_count",
    "val_fit_row_count",
    "test_fit_row_count",
    "train_time_min_utc",
    "train_time_max_utc",
    "m5_prebuilt_path",
    "m5_prebuilt_sha256",
    "mtf_cache_manifest_path",
    "mtf_cache_manifest_sha256",
    "mtf_builder_version",
    "mtf_feature_names_sha256",
    "per_tf_seq_lens",
    "per_tf_shift_seconds",
    "per_tf_fit_windows",
}


@dataclass(frozen=True)
class MatrixPopulationPart:
    """One physical matrix plus an optional sorted unique row selection.

    A fit may contain several physically distinct sources (Entry M5 and Exit
    M1).  Keeping the source matrix and row indices separate lets the fit owner
    scan the exact union repeatedly without concatenating a multi-gigabyte
    matrix in RAM.
    """

    values: Any
    row_indices: Any = None
    source: str = ""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _field_names_sha256(field_names: Sequence[str]) -> str:
    return _canonical_sha256([str(name) for name in field_names])


def _parse_utc(value: Any, *, field: str) -> datetime:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_TIMESTAMP_INVALID] field={field}"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_TIMESTAMP_NOT_UTC] field={field}"
        )
    return parsed


def _require_lineage(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _LINEAGE_KEYS:
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_LINEAGE_SCHEMA_INVALID]")
    data = dict(value)
    if not str(data["dataset_run_id"]).strip():
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_DATASET_RUN_ID_MISSING]")
    for field in (
        "train_parquet_path",
        "train_manifest_path",
        "m5_prebuilt_path",
        "mtf_cache_manifest_path",
        "mtf_builder_version",
    ):
        if not str(data[field]).strip():
            raise RuntimeError(
                f"[ENTRY_INPUT_NORMALIZATION_LINEAGE_FIELD_MISSING] field={field}"
            )
    for field in (
        "train_parquet_sha256",
        "train_manifest_sha256",
        "m5_prebuilt_sha256",
        "mtf_cache_manifest_sha256",
        "mtf_feature_names_sha256",
    ):
        if _SHA256_RE.fullmatch(str(data[field])) is None:
            raise RuntimeError(
                f"[ENTRY_INPUT_NORMALIZATION_LINEAGE_SHA_INVALID] field={field}"
            )
    population_counts = (
        "train_row_count",
        "entry_train_decision_row_count",
        "exit_train_decision_row_count",
        "local_fit_row_count",
        "context_fit_row_count",
    )
    if any(
        isinstance(data[field], bool)
        or not isinstance(data[field], int)
        or int(data[field]) < (2 if field != "exit_train_decision_row_count" else 1)
        for field in population_counts
    ):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_TRAIN_ROWS_INVALID]")
    if (
        int(data["context_fit_row_count"])
        != int(data["entry_train_decision_row_count"])
        + int(data["exit_train_decision_row_count"])
        or int(data["train_row_count"]) != int(data["context_fit_row_count"])
        or int(data["local_fit_row_count"])
        < int(data["context_fit_row_count"])
    ):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_SHARED_POPULATION_COUNT_INVALID]"
        )
    if data["val_fit_row_count"] != 0 or data["test_fit_row_count"] != 0:
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_OOS_FIT_ROWS_FORBIDDEN]")
    time_min = _parse_utc(data["train_time_min_utc"], field="train_time_min_utc")
    time_max = _parse_utc(data["train_time_max_utc"], field="train_time_max_utc")
    if time_min > time_max:
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_TRAIN_TIME_ORDER_INVALID]")
    seq_lens = data["per_tf_seq_lens"]
    shifts = data["per_tf_shift_seconds"]
    windows = data["per_tf_fit_windows"]
    if (
        not isinstance(seq_lens, Mapping)
        or tuple(seq_lens) != EXPECTED_TFS
        or not isinstance(shifts, Mapping)
        or tuple(shifts) != EXPECTED_TFS
        or not isinstance(windows, Mapping)
        or tuple(windows) != EXPECTED_TFS
    ):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_MTF_LINEAGE_INVALID]")
    for tf in EXPECTED_TFS:
        if (
            isinstance(seq_lens[tf], bool)
            or int(seq_lens[tf]) < 1
            or isinstance(shifts[tf], bool)
            or int(shifts[tf]) < 1
        ):
            raise RuntimeError(
                f"[ENTRY_INPUT_NORMALIZATION_MTF_PARAMETER_INVALID] tf={tf}"
            )
        window = windows[tf]
        expected_window_keys = {
            "left_index_inclusive",
            "right_index_exclusive",
            "selected_unique_row_count",
            "selected_row_indices_sha256",
            "selected_row_values_sha256",
            "time_min_utc",
            "time_max_utc",
        }
        if not isinstance(window, Mapping) or set(window) != expected_window_keys:
            raise RuntimeError(
                f"[ENTRY_INPUT_NORMALIZATION_MTF_WINDOW_SCHEMA_INVALID] tf={tf}"
            )
        left = window["left_index_inclusive"]
        right = window["right_index_exclusive"]
        count = window["selected_unique_row_count"]
        if (
            isinstance(left, bool)
            or isinstance(right, bool)
            or isinstance(count, bool)
            or int(left) < 0
            or int(right) <= int(left)
            or int(count) < 2
            or int(count) > int(right) - int(left)
        ):
            raise RuntimeError(
                f"[ENTRY_INPUT_NORMALIZATION_MTF_WINDOW_INDEX_INVALID] tf={tf}"
            )
        if _SHA256_RE.fullmatch(
            str(window["selected_row_indices_sha256"])
        ) is None:
            raise RuntimeError(
                "[ENTRY_INPUT_NORMALIZATION_MTF_WINDOW_SELECTION_HASH_INVALID] "
                f"tf={tf}"
            )
        if _SHA256_RE.fullmatch(
            str(window["selected_row_values_sha256"])
        ) is None:
            raise RuntimeError(
                "[ENTRY_INPUT_NORMALIZATION_MTF_WINDOW_VALUES_HASH_INVALID] "
                f"tf={tf}"
            )
        window_min = _parse_utc(window["time_min_utc"], field=f"{tf}.time_min")
        window_max = _parse_utc(window["time_max_utc"], field=f"{tf}.time_max")
        if window_min > window_max:
            raise RuntimeError(
                f"[ENTRY_INPUT_NORMALIZATION_MTF_WINDOW_TIME_INVALID] tf={tf}"
            )
    return data


def _require_aliases(
    aliases: Sequence[Mapping[str, Any]],
    *,
    signal: Mapping[str, Any],
    ctx_cont: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(aliases, (list, tuple)):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_ALIASES_INVALID]")
    normalized: list[dict[str, Any]] = []
    expected_keys = {
        "signal_field",
        "signal_index",
        "ctx_cont_field",
        "ctx_cont_index",
        "specialist",
    }
    seen_signal: set[int] = set()
    seen_ctx: set[int] = set()
    signal_names = list(signal["field_names"])
    ctx_names = list(ctx_cont["field_names"])
    signal_center = np.asarray(signal["center"], dtype=np.float32)
    signal_scale = np.asarray(signal["scale"], dtype=np.float32)
    signal_binary = np.asarray(signal["binary_mask"], dtype=np.uint8)
    signal_categorical = np.asarray(signal["categorical_mask"], dtype=np.uint8)
    ctx_center = np.asarray(ctx_cont["center"], dtype=np.float32)
    ctx_scale = np.asarray(ctx_cont["scale"], dtype=np.float32)
    ctx_binary = np.asarray(ctx_cont["binary_mask"], dtype=np.uint8)
    ctx_categorical = np.asarray(
        ctx_cont["categorical_mask"], dtype=np.uint8
    )
    for raw in aliases:
        if not isinstance(raw, Mapping) or set(raw) != expected_keys:
            raise RuntimeError(
                "[ENTRY_INPUT_NORMALIZATION_ALIAS_SCHEMA_INVALID]"
            )
        alias = dict(raw)
        signal_index = alias["signal_index"]
        ctx_index = alias["ctx_cont_index"]
        if (
            isinstance(signal_index, bool)
            or isinstance(ctx_index, bool)
            or not isinstance(signal_index, int)
            or not isinstance(ctx_index, int)
            or signal_index < 0
            or signal_index >= len(signal_names)
            or ctx_index < 0
            or ctx_index >= len(ctx_names)
            or signal_index in seen_signal
            or ctx_index in seen_ctx
            or alias["signal_field"] != signal_names[signal_index]
            or alias["ctx_cont_field"] != ctx_names[ctx_index]
            or alias["signal_field"] != f"ctx_cont.{alias['ctx_cont_field']}"
            or not str(alias["specialist"]).strip()
        ):
            raise RuntimeError(
                "[ENTRY_INPUT_NORMALIZATION_ALIAS_IDENTITY_INVALID]"
            )
        if (
            signal_center[signal_index].tobytes() != ctx_center[ctx_index].tobytes()
            or signal_scale[signal_index].tobytes() != ctx_scale[ctx_index].tobytes()
            or signal_binary[signal_index] != ctx_binary[ctx_index]
            or signal_categorical[signal_index] != ctx_categorical[ctx_index]
            or signal["scale_source"][signal_index]
            != ctx_cont["scale_source"][ctx_index]
            or (
                signal["categorical_domains"].get(alias["signal_field"])
                != ctx_cont["categorical_domains"].get(alias["ctx_cont_field"])
            )
        ):
            raise RuntimeError(
                "[ENTRY_INPUT_NORMALIZATION_ALIAS_STATS_NOT_SHARED] "
                f"signal={alias['signal_field']} ctx={alias['ctx_cont_field']}"
            )
        seen_signal.add(signal_index)
        seen_ctx.add(ctx_index)
        normalized.append(alias)
    if normalized != sorted(normalized, key=lambda item: item["signal_index"]):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_ALIAS_ORDER_INVALID]")
    ctx_lookup = {name: index for index, name in enumerate(ctx_names)}
    expected_pairs = [
        (signal_index, ctx_lookup[signal_field.removeprefix("ctx_cont.")])
        for signal_index, signal_field in enumerate(signal_names)
        if signal_field.startswith("ctx_cont.")
        and signal_field.removeprefix("ctx_cont.") in ctx_lookup
    ]
    observed_pairs = [
        (int(alias["signal_index"]), int(alias["ctx_cont_index"]))
        for alias in normalized
    ]
    if observed_pairs != expected_pairs:
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_ALIAS_COVERAGE_INVALID] "
            f"observed={len(observed_pairs)} expected={len(expected_pairs)}"
        )
    return normalized


def _stats_sha256(
    *,
    field_names: Sequence[str],
    center: np.ndarray,
    scale: np.ndarray,
    binary_mask: np.ndarray,
    categorical_mask: np.ndarray,
    categorical_domains: Mapping[str, Sequence[int]],
    train_transformed_min: np.ndarray,
    train_transformed_max: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"entry_model_native_input_normalization_surface_v3_asinh\0")
    digest.update(CONTINUOUS_TRANSFORM.encode("ascii") + b"\0")
    digest.update(bytes.fromhex(_field_names_sha256(field_names)))
    digest.update(np.ascontiguousarray(center, dtype="<f4").tobytes(order="C"))
    digest.update(np.ascontiguousarray(scale, dtype="<f4").tobytes(order="C"))
    digest.update(np.ascontiguousarray(binary_mask, dtype=np.uint8).tobytes(order="C"))
    digest.update(
        np.ascontiguousarray(categorical_mask, dtype=np.uint8).tobytes(order="C")
    )
    digest.update(
        np.ascontiguousarray(
            train_transformed_min,
            dtype="<f8",
        ).tobytes(order="C")
    )
    digest.update(
        np.ascontiguousarray(
            train_transformed_max,
            dtype="<f8",
        ).tobytes(order="C")
    )
    digest.update(
        json.dumps(
            {
                str(name): [int(item) for item in domain]
                for name, domain in categorical_domains.items()
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def fit_ctx_cat_contract(
    values: Any,
    *,
    field_names: Sequence[str],
) -> dict[str, Any]:
    parts, population_rows, width = _as_population_parts(
        values,
        surface="ctx_cat",
    )
    names = [str(name) for name in field_names]
    if names != list(CTX_CAT_DOMAINS) or int(width) != len(names):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_CTX_CAT_FIELDS_INVALID]")
    observed_counts: dict[str, dict[str, int]] = {
        name: {str(value): 0 for value in CTX_CAT_DOMAINS[name]}
        for name in names
    }
    observed_rows = 0
    for numeric in _iter_population_column_blocks(
        parts,
        start=0,
        stop=width,
    ):
        if (
            not np.isfinite(numeric).all()
            or not np.equal(numeric, np.floor(numeric)).all()
        ):
            raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_CTX_CAT_VALUE_INVALID]")
        for index, name in enumerate(names):
            domain = CTX_CAT_DOMAINS[name]
            values_i = numeric[:, index].astype(np.int64)
            if not np.isin(values_i, domain).all():
                raise RuntimeError(
                    f"[ENTRY_INPUT_NORMALIZATION_CTX_CAT_DOMAIN_INVALID] field={name}"
                )
            for value in domain:
                observed_counts[name][str(value)] += int(
                    np.count_nonzero(values_i == value)
                )
        observed_rows += int(numeric.shape[0])
    if observed_rows != population_rows:
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_CTX_CAT_SCAN_MISMATCH]")
    payload = {
        "policy": "categorical_embedding_no_numeric_transform",
        "field_names": names,
        "field_names_sha256": _field_names_sha256(names),
        "fit_row_count": int(population_rows),
        "domains": {
            name: list(CTX_CAT_DOMAINS[name])
            for name in names
        },
        "observed_train_counts": observed_counts,
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    return payload


def require_ctx_cat_contract(
    value: Mapping[str, Any],
    *,
    field_names: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_CTX_CAT_CONTRACT_MISSING]")
    data = dict(value)
    expected_keys = {
        "policy",
        "field_names",
        "field_names_sha256",
        "fit_row_count",
        "domains",
        "observed_train_counts",
        "contract_sha256",
    }
    names = [str(name) for name in field_names]
    if (
        set(data) != expected_keys
        or data["policy"] != "categorical_embedding_no_numeric_transform"
        or names != list(CTX_CAT_DOMAINS)
        or data["field_names"] != names
        or data["field_names_sha256"] != _field_names_sha256(names)
        or int(data["fit_row_count"]) < 2
        or data["domains"]
        != {name: list(CTX_CAT_DOMAINS[name]) for name in names}
    ):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_CTX_CAT_CONTRACT_INVALID]")
    counts = data["observed_train_counts"]
    if not isinstance(counts, Mapping) or tuple(counts) != tuple(names):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_CTX_CAT_COUNTS_INVALID]")
    for name in names:
        expected_count_keys = {str(value) for value in CTX_CAT_DOMAINS[name]}
        field_counts = counts[name]
        if (
            not isinstance(field_counts, Mapping)
            or set(field_counts) != expected_count_keys
            or any(
                isinstance(count, bool) or int(count) < 0
                for count in field_counts.values()
            )
            or sum(int(count) for count in field_counts.values())
            != int(data["fit_row_count"])
        ):
            raise RuntimeError(
                f"[ENTRY_INPUT_NORMALIZATION_CTX_CAT_COUNTS_INVALID] field={name}"
            )
    without_hash = dict(data)
    observed_hash = str(without_hash.pop("contract_sha256") or "")
    if observed_hash != _canonical_sha256(without_hash):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_CTX_CAT_HASH_MISMATCH]")
    return data


def _as_population_parts(
    values: Any,
    *,
    surface: str,
) -> tuple[list[tuple[np.ndarray, np.ndarray | None, str]], int, int]:
    """Validate one bounded fit population without concatenating its sources."""

    if isinstance(values, MatrixPopulationPart):
        raw_parts = [values]
    elif (
        isinstance(values, (list, tuple))
        and values
        and all(isinstance(item, MatrixPopulationPart) for item in values)
    ):
        raw_parts = list(values)
    elif (
        isinstance(values, (list, tuple))
        and values
        and all(np.asarray(item).ndim == 2 for item in values)
    ):
        raw_parts = [
            MatrixPopulationPart(item, source=f"{surface}:{index}")
            for index, item in enumerate(values)
        ]
    else:
        raw_parts = [MatrixPopulationPart(values, source=surface)]

    normalized: list[tuple[np.ndarray, np.ndarray | None, str]] = []
    width: int | None = None
    total_rows = 0
    for part_index, part in enumerate(raw_parts):
        matrix = np.asarray(part.values)
        if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
            raise RuntimeError(
                f"[ENTRY_INPUT_NORMALIZATION_MATRIX_INVALID] "
                f"surface={surface} part={part_index} shape={tuple(matrix.shape)}"
            )
        if not np.issubdtype(matrix.dtype, np.number):
            raise RuntimeError(
                f"[ENTRY_INPUT_NORMALIZATION_MATRIX_DTYPE_INVALID] "
                f"surface={surface} part={part_index} dtype={matrix.dtype}"
            )
        if width is None:
            width = int(matrix.shape[1])
        elif int(matrix.shape[1]) != width:
            raise RuntimeError(
                f"[ENTRY_INPUT_NORMALIZATION_MATRIX_WIDTH_MISMATCH] "
                f"surface={surface} part={part_index} "
                f"observed={int(matrix.shape[1])} expected={width}"
            )
        indices: np.ndarray | None = None
        if part.row_indices is not None:
            raw_indices = np.asarray(part.row_indices)
            if (
                raw_indices.ndim != 1
                or raw_indices.size < 1
                or not np.issubdtype(raw_indices.dtype, np.integer)
            ):
                raise RuntimeError(
                    f"[ENTRY_INPUT_NORMALIZATION_ROW_SELECTION_INVALID] "
                    f"surface={surface} part={part_index}"
                )
            indices = np.asarray(raw_indices, dtype=np.int64)
            if (
                int(indices[0]) < 0
                or int(indices[-1]) >= int(matrix.shape[0])
                or np.any(np.diff(indices) <= 0)
            ):
                raise RuntimeError(
                    f"[ENTRY_INPUT_NORMALIZATION_ROW_SELECTION_INVALID] "
                    f"surface={surface} part={part_index}"
                )
            selected_rows = int(indices.size)
        else:
            selected_rows = int(matrix.shape[0])
        normalized.append((matrix, indices, str(part.source or surface)))
        total_rows += selected_rows
    if total_rows < 2 or width is None:
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_MATRIX_INVALID] "
            f"surface={surface} rows={total_rows}"
        )
    return normalized, total_rows, width


def _iter_population_column_blocks(
    parts: Sequence[tuple[np.ndarray, np.ndarray | None, str]],
    *,
    start: int,
    stop: int,
    row_chunk: int = FIT_ROW_CHUNK,
    dtype: np.dtype[Any] = np.dtype(np.float64),
):
    for matrix, indices, _source in parts:
        selected_rows = int(matrix.shape[0]) if indices is None else int(indices.size)
        for row_start in range(0, selected_rows, int(row_chunk)):
            row_stop = min(selected_rows, row_start + int(row_chunk))
            if indices is None:
                raw = matrix[row_start:row_stop, start:stop]
            else:
                raw = matrix[indices[row_start:row_stop], start:stop]
            yield np.asarray(raw, dtype=dtype)


def _materialize_population_columns(
    parts: Sequence[tuple[np.ndarray, np.ndarray | None, str]],
    *,
    total_rows: int,
    start: int,
    stop: int,
) -> np.ndarray:
    block = np.empty((int(total_rows), int(stop - start)), dtype=np.float64)
    offset = 0
    for source_block in _iter_population_column_blocks(
        parts,
        start=start,
        stop=stop,
    ):
        next_offset = offset + int(source_block.shape[0])
        block[offset:next_offset] = source_block
        offset = next_offset
    if offset != int(total_rows):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_POPULATION_SCAN_MISMATCH]")
    return block


def _population_transformed_extrema(
    parts: Sequence[tuple[np.ndarray, np.ndarray | None, str]],
    *,
    width: int,
    center: np.ndarray,
    scale: np.ndarray,
    binary_mask: np.ndarray,
    categorical_mask: np.ndarray,
    column_chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure exact TRAIN output support without changing any observation."""

    transformed_min = np.full(int(width), np.inf, dtype=np.float64)
    transformed_max = np.full(int(width), -np.inf, dtype=np.float64)
    for start in range(0, int(width), int(column_chunk)):
        stop = min(int(width), start + int(column_chunk))
        for block in _iter_population_column_blocks(
            parts,
            start=start,
            stop=stop,
        ):
            if not np.isfinite(block).all():
                raise RuntimeError(
                    "[ENTRY_INPUT_NORMALIZATION_NONFINITE] "
                    f"field_offset={start}"
                )
            transformed = (
                block - center[start:stop].astype(np.float64)
            ) / scale[start:stop].astype(np.float64)
            identity = np.logical_or(
                binary_mask[start:stop].astype(bool),
                categorical_mask[start:stop].astype(bool),
            )
            if identity.any():
                transformed[:, identity] = block[:, identity]
            continuous = ~identity
            if continuous.any():
                transformed[:, continuous] = np.arcsinh(
                    transformed[:, continuous]
                )
            if not np.isfinite(transformed).all():
                raise RuntimeError(
                    "[ENTRY_INPUT_NORMALIZATION_TRANSFORM_NONFINITE] "
                    f"field_offset={start}"
                )
            transformed_min[start:stop] = np.minimum(
                transformed_min[start:stop],
                transformed.min(axis=0),
            )
            transformed_max[start:stop] = np.maximum(
                transformed_max[start:stop],
                transformed.max(axis=0),
            )
    if (
        not np.isfinite(transformed_min).all()
        or not np.isfinite(transformed_max).all()
        or np.any(transformed_min > transformed_max)
    ):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_TRAIN_SUPPORT_INVALID]")
    return transformed_min, transformed_max


def fit_surface_normalization(
    values: Any,
    *,
    surface: str,
    field_names: Sequence[str],
    row_count: int | None = None,
    column_chunk: int = FIT_COLUMN_CHUNK,
    semantic_categorical_domains: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Fit an invertible robust transform without materializing the matrix.

    The input may be a memmap.  Columns are copied in small chunks so fitting
    the full TRAIN snapshot does not create another multi-gigabyte array.
    Continuous values use ``asinh((x - median) / scale)``. The scale is the
    raw TRAIN IQR, falling back only when that exact IQR is zero to the median
    strictly-positive deviation observed in TRAIN. There is no numeric floor,
    quantile cap, clipping or saturated-mask exception.
    """

    parts, population_rows, width = _as_population_parts(
        values,
        surface=surface,
    )
    names = [str(name) for name in field_names]
    if (
        not names
        or any(not name for name in names)
        or len(names) != len(set(names))
        or len(names) != int(width)
    ):
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_FIELDS_INVALID] "
            f"surface={surface} fields={len(names)} width={int(width)}"
        )
    if row_count is not None and int(row_count) != int(population_rows):
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_ROW_COUNT_MISMATCH] "
            f"surface={surface} declared={int(row_count)} "
            f"observed={int(population_rows)}"
        )
    if isinstance(column_chunk, bool) or int(column_chunk) < 1:
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_CHUNK_INVALID] {column_chunk!r}"
        )

    max_columns_by_budget = max(
        1,
        int(FIT_MAX_WORKING_BLOCK_BYTES) // (int(population_rows) * 8),
    )
    effective_column_chunk = min(int(column_chunk), max_columns_by_budget)
    center = np.empty(width, dtype=np.float32)
    scale = np.empty(width, dtype=np.float32)
    binary_mask = np.zeros(width, dtype=np.uint8)
    categorical_mask = np.zeros(width, dtype=np.uint8)
    scale_source: list[str] = [""] * width
    categorical_domains = {
        str(name): tuple(int(value) for value in domain)
        for name, domain in (semantic_categorical_domains or {}).items()
    }
    if (
        not set(categorical_domains).issubset(names)
        or any(
            not domain or len(domain) != len(set(domain))
            for domain in categorical_domains.values()
        )
        # A declared domain must be the exact 0-based range ``0..len-1``.
        # Origin (rule 2a): every consumer of a declared domain builds
        # ``nn.Embedding(len(domain), d_model)`` and indexes it with the RAW
        # field value cast by ``.long()`` -- entry_v10_ctx_hybrid_transformer
        # does this for the signal surface, for the per-specialist ctx_cont
        # nominal block and for every MTF lane. A domain such as (-1, 0, 1)
        # passes every other check here and in the runtime value guard (which
        # compares against ``min(domain)``/``max(domain)``), then indexes an
        # embedding table out of range. Fail at declaration instead.
        or any(
            tuple(domain) != tuple(range(len(domain)))
            for domain in categorical_domains.values()
        )
    ):
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_CATEGORICAL_FIELDS_INVALID] "
            f"surface={surface}"
        )

    for start in range(0, width, effective_column_chunk):
        stop = min(width, start + effective_column_chunk)
        block = _materialize_population_columns(
            parts,
            total_rows=population_rows,
            start=start,
            stop=stop,
        )
        if not np.isfinite(block).all():
            bad = np.argwhere(~np.isfinite(block))[0]
            raise RuntimeError(
                "[ENTRY_INPUT_NORMALIZATION_NONFINITE] "
                f"surface={surface} row={int(bad[0])} "
                f"field={names[start + int(bad[1])]}"
            )
        q25, median, q75 = np.quantile(block, [0.25, 0.5, 0.75], axis=0)
        for local in range(stop - start):
            index = start + local
            column = block[:, local]
            if names[index] in categorical_domains:
                domain = categorical_domains[names[index]]
                if (
                    not np.equal(column, np.floor(column)).all()
                    or not np.isin(column.astype(np.int64), domain).all()
                ):
                    raise RuntimeError(
                        "[ENTRY_INPUT_NORMALIZATION_CATEGORICAL_DOMAIN_INVALID] "
                        f"surface={surface} field={names[index]}"
                    )
                center[index] = np.float32(0.0)
                scale[index] = np.float32(1.0)
                categorical_mask[index] = np.uint8(1)
                scale_source[index] = "categorical_embedding_identity"
                continue
            # v7 (2026-08-19): the sample-inferred binary branch is REMOVED.
            # It read ``all(x in {0,1}) and any(x==0) and any(x==1)`` off the
            # fit window and stamped ``binary_mask``, which
            # ``apply_surface_normalization`` and the runtime encoder then
            # enforce as a hard {0,1} domain for the life of the bundle. That
            # made a window statistic into immutable bundle state with no
            # declared origin (rule 2a) and no way to fail closed (rule 18).
            #
            # PROVEN FROM SOURCE: ``ema_stack_aligned_v2`` emits exactly
            # {-1.0, 0.0, +1.0} post-warmup -- htf_features fills every defined
            # row with 0.0, then ``stack[bull] = 1`` / ``stack[bear] = -1`` over
            # two mutually exclusive strict EMA orderings. Three further signed
            # ternaries share that construction: ``mtf_level_retest_hold_signed``
            # and ``mtf_level_retest_fail_signed`` (level_registry_v1, a pure
            # ``(s > 0) - (s < 0)`` sign extraction), and the five ctx_cont
            # projections ``{tf}_ema_stack_aligned_v2`` of the first field.
            # REPORTED MEASUREMENT, not taken by this change (operator, on the
            # declared TRAIN window 2025-06-01..2026-05-31): the two
            # ``ema_stack`` names observed only {0, +1} there, were fitted as
            # binary, and the first D1 bear stack at serve raised
            # [ENTRY_INPUT_NORMALIZATION_BINARY_VALUE_INVALID] instead of
            # producing a decision -- so the system could not emit any Entry
            # action, not even FLAT, in a daily downtrend. Which of the signed
            # ternaries is captured depends only on which sign the window
            # happens to contain, which is the defect, not the field.
            #
            # No branch replaces it. A field is scaled as a nominal category
            # only where a contract owner DECLARES its exact domain above;
            # everything else -- including a genuinely two-valued flag -- takes
            # the continuous branch below. Two properties of that branch are
            # proven from the algebra, not assumed:
            #   * it is injective on two points (a positive affine map followed
            #     by the strictly increasing asinh), so no evidence is lost and
            #     the first Linear layer absorbs the change of units; and
            #   * it cannot newly raise UNSCALEABLE on a two-valued column. If
            #     at least 75% of the rows share one value the IQR is 0 and the
            #     median positive absolute deviation is the gap to the other
            #     value; otherwise the IQR is that same gap. Either way the
            #     scale is the gap, which is strictly positive whenever the
            #     column holds two distinct values.
            # Removing this exception is what rule 2b permits; inferring a
            # domain from a sample is what it forbids.
            field_center = np.float32(median[local])
            field_scale = np.float32(q75[local] - q25[local])
            source = "raw_iqr"
            if not np.isfinite(field_scale) or field_scale <= np.float32(0.0):
                deviations = np.abs(column - float(field_center))
                positive = deviations[deviations > 0.0]
                if positive.size:
                    field_scale = np.float32(np.median(positive))
                    source = "median_positive_abs_deviation"
            if not np.isfinite(field_scale) or field_scale <= np.float32(0.0):
                raise RuntimeError(
                    "[ENTRY_INPUT_NORMALIZATION_UNSCALEABLE] "
                    f"surface={surface} field={names[index]} "
                    f"center={float(field_center)!r} scale={float(field_scale)!r}"
                )
            center[index] = field_center
            scale[index] = field_scale
            scale_source[index] = source

    if (
        not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or not (scale > np.float32(0.0)).all()
    ):
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_FLOAT32_STATS_INVALID] surface={surface}"
        )
    transformed_min, transformed_max = _population_transformed_extrema(
        parts,
        width=width,
        center=center,
        scale=scale,
        binary_mask=binary_mask,
        categorical_mask=categorical_mask,
        column_chunk=effective_column_chunk,
    )
    stats_hash = _stats_sha256(
        field_names=names,
        center=center,
        scale=scale,
        binary_mask=binary_mask,
        categorical_mask=categorical_mask,
        categorical_domains=categorical_domains,
        train_transformed_min=transformed_min,
        train_transformed_max=transformed_max,
    )
    return {
        "surface": str(surface),
        "continuous_transform": CONTINUOUS_TRANSFORM,
        "field_count": width,
        "field_names": names,
        "field_names_sha256": _field_names_sha256(names),
        "fit_row_count": int(population_rows),
        "center": [float(value) for value in center],
        "scale": [float(value) for value in scale],
        "binary_mask": [int(value) for value in binary_mask],
        "binary_field_count": int(binary_mask.sum()),
        "categorical_mask": [int(value) for value in categorical_mask],
        "categorical_field_count": int(categorical_mask.sum()),
        "categorical_domains": {
            name: list(domain) for name, domain in categorical_domains.items()
        },
        "scale_source": scale_source,
        "train_transformed_min": [float(value) for value in transformed_min],
        "train_transformed_max": [float(value) for value in transformed_max],
        "stats_sha256": stats_hash,
    }


def require_surface_normalization(
    value: Mapping[str, Any],
    *,
    surface: str,
    field_names: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_SURFACE_MISSING] surface={surface}"
        )
    data = dict(value)
    expected_keys = {
        "surface",
        "continuous_transform",
        "field_count",
        "field_names",
        "field_names_sha256",
        "fit_row_count",
        "center",
        "scale",
        "binary_mask",
        "binary_field_count",
        "categorical_mask",
        "categorical_field_count",
        "categorical_domains",
        "scale_source",
        "train_transformed_min",
        "train_transformed_max",
        "stats_sha256",
    }
    if set(data) != expected_keys:
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_SURFACE_SCHEMA_INVALID] "
            f"surface={surface} missing={sorted(expected_keys - set(data))} "
            f"extra={sorted(set(data) - expected_keys)}"
        )
    names = [str(name) for name in field_names]
    if (
        data["surface"] != surface
        or data["continuous_transform"] != CONTINUOUS_TRANSFORM
        or data["field_names"] != names
        or int(data["field_count"]) != len(names)
        or data["field_names_sha256"] != _field_names_sha256(names)
        or int(data["fit_row_count"]) < 2
    ):
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_SURFACE_IDENTITY_INVALID] surface={surface}"
        )
    center = np.asarray(data["center"], dtype=np.float32)
    scale = np.asarray(data["scale"], dtype=np.float32)
    binary_mask = np.asarray(data["binary_mask"], dtype=np.uint8)
    categorical_mask = np.asarray(data["categorical_mask"], dtype=np.uint8)
    categorical_domains = data["categorical_domains"]
    scale_source = [str(item) for item in data["scale_source"]]
    transformed_min = np.asarray(data["train_transformed_min"], dtype=np.float64)
    transformed_max = np.asarray(data["train_transformed_max"], dtype=np.float64)
    expected_shape = (len(names),)
    if (
        center.shape != expected_shape
        or scale.shape != expected_shape
        or binary_mask.shape != expected_shape
        or categorical_mask.shape != expected_shape
        or len(scale_source) != len(names)
        or transformed_min.shape != expected_shape
        or transformed_max.shape != expected_shape
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or not (scale > 0.0).all()
        or not np.isin(binary_mask, [0, 1]).all()
        or not np.isin(categorical_mask, [0, 1]).all()
        or np.logical_and(binary_mask, categorical_mask).any()
        or int(binary_mask.sum()) != int(data["binary_field_count"])
        or int(categorical_mask.sum()) != int(data["categorical_field_count"])
        or not isinstance(categorical_domains, Mapping)
        or set(categorical_domains)
        != {names[index] for index in np.flatnonzero(categorical_mask)}
        or not np.isfinite(transformed_min).all()
        or not np.isfinite(transformed_max).all()
        or np.any(transformed_min > transformed_max)
    ):
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_SURFACE_VALUES_INVALID] surface={surface}"
        )
    # v7 (2026-08-19): no fit may stamp ``binary_mask`` any more (see the
    # removed inference in ``fit_surface_normalization``), so a surface that
    # carries one was fitted by the pre-v7 owner and its center/scale for that
    # field is an identity chosen from a window, not a fitted statistic. That
    # is stale immutable bundle state (rule 18) and is rejected here rather
    # than silently re-admitted. The key itself stays in the schema because the
    # runtime encoder registers an ``input_norm_{surface}_binary_mask`` buffer
    # from it and applies the same identity rule; retiring the key belongs to
    # that owner's wave, not this contract's.
    if int(binary_mask.sum()) != 0:
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_INFERRED_BINARY_MASK_FORBIDDEN] "
            f"surface={surface} "
            f"fields={[names[index] for index in np.flatnonzero(binary_mask)][:10]}"
        )
    for index in range(len(names)):
        is_categorical = bool(categorical_mask[index])
        if is_categorical:
            domain = categorical_domains.get(names[index])
            if (
                center[index] != np.float32(0.0)
                or scale[index] != np.float32(1.0)
                or scale_source[index] != "categorical_embedding_identity"
                or not isinstance(domain, list)
                or not domain
                or any(isinstance(item, bool) or not isinstance(item, int) for item in domain)
                or len(domain) != len(set(domain))
                # Same 0-based-range requirement the fit enforces: the runtime
                # encoder indexes an ``nn.Embedding(len(domain), d_model)`` with
                # the raw value, so a stored domain outside ``0..len-1`` is an
                # out-of-range table lookup, not a strict-load failure.
                or domain != list(range(len(domain)))
            ):
                raise RuntimeError(
                    f"[ENTRY_INPUT_NORMALIZATION_CATEGORICAL_CONTRACT_INVALID] "
                    f"surface={surface} field={names[index]}"
                )
            if (
                transformed_min[index] not in domain
                or transformed_max[index] not in domain
            ):
                raise RuntimeError(
                    "[ENTRY_INPUT_NORMALIZATION_CATEGORICAL_TRAIN_SUPPORT_INVALID] "
                    f"surface={surface} field={names[index]}"
                )
        if not is_categorical and scale_source[index] not in {
            "raw_iqr",
            "median_positive_abs_deviation",
        }:
            raise RuntimeError(
                f"[ENTRY_INPUT_NORMALIZATION_SCALE_SOURCE_INVALID] "
                f"surface={surface} field={names[index]}"
            )
        if (
            not is_categorical
            and transformed_min[index] >= transformed_max[index]
        ):
            raise RuntimeError(
                f"[ENTRY_INPUT_NORMALIZATION_CONTINUOUS_TRAIN_SUPPORT_INVALID] "
                f"surface={surface} field={names[index]}"
            )
    expected_hash = _stats_sha256(
        field_names=names,
        center=center,
        scale=scale,
        binary_mask=binary_mask,
        categorical_mask=categorical_mask,
        categorical_domains=categorical_domains,
        train_transformed_min=transformed_min,
        train_transformed_max=transformed_max,
    )
    if data["stats_sha256"] != expected_hash:
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_STATS_HASH_MISMATCH] surface={surface}"
        )
    return data


def share_temporal_alias_stats_from_signal(
    ctx_cont_surface: Mapping[str, Any],
    signal_surface: Mapping[str, Any],
    *,
    temporal_aliases: Sequence[Mapping[str, Any]],
    ctx_cont_values: Any,
) -> dict[str, Any]:
    """Make the shared local signal population own temporal-alias statistics.

    Entry/Exit local rows are a superset of their current decision rows.  The
    local encoder therefore owns each duplicated temporal field once; the
    ctx_cont surface reuses those exact center/scale values while recording
    its own TRAIN-only transformed support without modifying observations.
    """

    ctx_cont = dict(ctx_cont_surface)
    signal = dict(signal_surface)
    ctx_names = [str(name) for name in ctx_cont["field_names"]]
    signal_names = [str(name) for name in signal["field_names"]]
    center = np.asarray(ctx_cont["center"], dtype=np.float32).copy()
    scale = np.asarray(ctx_cont["scale"], dtype=np.float32).copy()
    binary = np.asarray(ctx_cont["binary_mask"], dtype=np.uint8).copy()
    categorical = np.asarray(
        ctx_cont["categorical_mask"], dtype=np.uint8
    ).copy()
    scale_source = [str(value) for value in ctx_cont["scale_source"]]
    categorical_domains = {
        str(name): [int(item) for item in domain]
        for name, domain in ctx_cont["categorical_domains"].items()
    }
    signal_center = np.asarray(signal["center"], dtype=np.float32)
    signal_scale = np.asarray(signal["scale"], dtype=np.float32)
    signal_binary = np.asarray(signal["binary_mask"], dtype=np.uint8)
    signal_categorical = np.asarray(
        signal["categorical_mask"], dtype=np.uint8
    )
    for raw in temporal_aliases:
        alias = dict(raw)
        signal_index = int(alias["signal_index"])
        ctx_index = int(alias["ctx_cont_index"])
        if (
            signal_names[signal_index] != alias["signal_field"]
            or ctx_names[ctx_index] != alias["ctx_cont_field"]
            or alias["signal_field"] != f"ctx_cont.{alias['ctx_cont_field']}"
        ):
            raise RuntimeError(
                "[ENTRY_INPUT_NORMALIZATION_ALIAS_IDENTITY_INVALID]"
            )
        center[ctx_index] = signal_center[signal_index]
        scale[ctx_index] = signal_scale[signal_index]
        binary[ctx_index] = signal_binary[signal_index]
        categorical[ctx_index] = signal_categorical[signal_index]
        scale_source[ctx_index] = signal["scale_source"][signal_index]
        categorical_domains.pop(alias["ctx_cont_field"], None)
        signal_domain = signal["categorical_domains"].get(
            alias["signal_field"]
        )
        if signal_domain is not None:
            categorical_domains[alias["ctx_cont_field"]] = [
                int(item) for item in signal_domain
            ]

    parts, population_rows, width = _as_population_parts(
        ctx_cont_values,
        surface="ctx_cont",
    )
    if (
        width != len(ctx_names)
        or population_rows != int(ctx_cont["fit_row_count"])
    ):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_ALIAS_CTX_POPULATION_MISMATCH]"
        )
    max_columns_by_budget = max(
        1,
        int(FIT_MAX_WORKING_BLOCK_BYTES) // (int(population_rows) * 8),
    )
    transformed_min, transformed_max = _population_transformed_extrema(
        parts,
        width=width,
        center=center,
        scale=scale,
        binary_mask=binary,
        categorical_mask=categorical,
        column_chunk=min(FIT_COLUMN_CHUNK, max_columns_by_budget),
    )
    ctx_cont.update(
        {
            "center": [float(value) for value in center],
            "scale": [float(value) for value in scale],
            "binary_mask": [int(value) for value in binary],
            "binary_field_count": int(binary.sum()),
            "categorical_mask": [int(value) for value in categorical],
            "categorical_field_count": int(categorical.sum()),
            "categorical_domains": categorical_domains,
            "scale_source": scale_source,
            "train_transformed_min": [
                float(value) for value in transformed_min
            ],
            "train_transformed_max": [
                float(value) for value in transformed_max
            ],
        }
    )
    ctx_cont["stats_sha256"] = _stats_sha256(
        field_names=ctx_names,
        center=center,
        scale=scale,
        binary_mask=binary,
        categorical_mask=categorical,
        categorical_domains=categorical_domains,
        train_transformed_min=transformed_min,
        train_transformed_max=transformed_max,
    )
    return ctx_cont


def build_input_normalization_contract(
    *,
    fit_start_utc: str,
    fit_end_utc: str,
    surfaces: Mapping[str, Mapping[str, Any]],
    ctx_cat: Mapping[str, Any],
    lineage: Mapping[str, Any],
    temporal_aliases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if tuple(surfaces) != EXPECTED_SURFACES:
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_SURFACES_INVALID] "
            f"observed={tuple(surfaces)} expected={EXPECTED_SURFACES}"
        )
    normalized_surfaces = {
        name: require_surface_normalization(
            surfaces[name],
            surface=name,
            field_names=list(surfaces[name].get("field_names") or []),
        )
        for name in EXPECTED_SURFACES
    }
    surface_hashes = {
        name: str(normalized_surfaces[name]["stats_sha256"])
        for name in EXPECTED_SURFACES
    }
    normalized_lineage = _require_lineage(lineage)
    normalized_ctx_cat = require_ctx_cat_contract(
        ctx_cat,
        field_names=list(CTX_CAT_DOMAINS),
    )
    if int(normalized_ctx_cat["fit_row_count"]) != int(
        normalized_lineage["context_fit_row_count"]
    ):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_CTX_CAT_LINEAGE_ROWS_MISMATCH]"
        )
    expected_shared_rows = {
        "signal": int(normalized_lineage["local_fit_row_count"]),
        "ctx_cont": int(normalized_lineage["context_fit_row_count"]),
    }
    for surface, expected_rows in expected_shared_rows.items():
        if int(normalized_surfaces[surface]["fit_row_count"]) != expected_rows:
            raise RuntimeError(
                "[ENTRY_INPUT_NORMALIZATION_SURFACE_LINEAGE_ROWS_MISMATCH] "
                f"surface={surface}"
            )
    for tf in EXPECTED_TFS:
        surface = f"mtf_{tf.lower()}"
        if int(normalized_surfaces[surface]["fit_row_count"]) != int(
            normalized_lineage["per_tf_fit_windows"][tf][
                "selected_unique_row_count"
            ]
        ):
            raise RuntimeError(
                "[ENTRY_INPUT_NORMALIZATION_MTF_LINEAGE_ROWS_MISMATCH] "
                f"tf={tf}"
            )
    if (
        _parse_utc(fit_start_utc, field="fit_start_utc")
        != _parse_utc(
            normalized_lineage["train_time_min_utc"],
            field="lineage.train_time_min_utc",
        )
        or _parse_utc(fit_end_utc, field="fit_end_utc")
        != _parse_utc(
            normalized_lineage["train_time_max_utc"],
            field="lineage.train_time_max_utc",
        )
    ):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_FIT_LINEAGE_MISMATCH]")
    normalized_aliases = _require_aliases(
        temporal_aliases,
        signal=normalized_surfaces["signal"],
        ctx_cont=normalized_surfaces["ctx_cont"],
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "transform": TRANSFORM,
        "fit_scope": "train_only",
        "fit_population": FIT_POPULATION,
        "fit_start_utc": str(fit_start_utc),
        "fit_end_utc": str(fit_end_utc),
        "continuous_transform": CONTINUOUS_TRANSFORM,
        "continuous_inverse": "x=sinh(z)*scale+center",
        "lineage": normalized_lineage,
        "ctx_cat": normalized_ctx_cat,
        "temporal_aliases": normalized_aliases,
        "temporal_aliases_sha256": _canonical_sha256(normalized_aliases),
        "surfaces": {
            name: dict(normalized_surfaces[name]) for name in EXPECTED_SURFACES
        },
        "surface_stats_sha256": surface_hashes,
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    return payload


def require_input_normalization_contract(
    value: Mapping[str, Any],
    *,
    expected_field_names: Mapping[str, Sequence[str]],
    expected_ctx_cat_names: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_CONTRACT_MISSING]")
    data = dict(value)
    expected_keys = {
        "schema_version",
        "transform",
        "fit_scope",
        "fit_population",
        "fit_start_utc",
        "fit_end_utc",
        "continuous_transform",
        "continuous_inverse",
        "lineage",
        "ctx_cat",
        "temporal_aliases",
        "temporal_aliases_sha256",
        "surfaces",
        "surface_stats_sha256",
        "contract_sha256",
    }
    if set(data) != expected_keys:
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_CONTRACT_SCHEMA_INVALID]")
    exact = {
        "schema_version": SCHEMA_VERSION,
        "transform": TRANSFORM,
        "fit_scope": "train_only",
        "fit_population": FIT_POPULATION,
        "continuous_transform": CONTINUOUS_TRANSFORM,
        "continuous_inverse": "x=sinh(z)*scale+center",
    }
    if any(data.get(key) != expected for key, expected in exact.items()):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_CONTRACT_IDENTITY_INVALID]")
    lineage = _require_lineage(data["lineage"])
    ctx_cat = require_ctx_cat_contract(
        data["ctx_cat"],
        field_names=expected_ctx_cat_names,
    )
    if int(ctx_cat["fit_row_count"]) != int(
        lineage["context_fit_row_count"]
    ):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_CTX_CAT_LINEAGE_ROWS_MISMATCH]"
        )
    if (
        _parse_utc(data["fit_start_utc"], field="fit_start_utc")
        != _parse_utc(lineage["train_time_min_utc"], field="train_time_min_utc")
        or _parse_utc(data["fit_end_utc"], field="fit_end_utc")
        != _parse_utc(lineage["train_time_max_utc"], field="train_time_max_utc")
        or not isinstance(data["surfaces"], Mapping)
        or tuple(data["surfaces"]) != EXPECTED_SURFACES
        or set(expected_field_names) != set(EXPECTED_SURFACES)
    ):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_CONTRACT_LINEAGE_INVALID]")
    normalized_surfaces: dict[str, Any] = {}
    for name in EXPECTED_SURFACES:
        normalized_surfaces[name] = require_surface_normalization(
            data["surfaces"][name],
            surface=name,
            field_names=expected_field_names[name],
        )
    expected_shared_rows = {
        "signal": int(lineage["local_fit_row_count"]),
        "ctx_cont": int(lineage["context_fit_row_count"]),
    }
    for surface, expected_rows in expected_shared_rows.items():
        if int(normalized_surfaces[surface]["fit_row_count"]) != expected_rows:
            raise RuntimeError(
                "[ENTRY_INPUT_NORMALIZATION_SURFACE_LINEAGE_ROWS_MISMATCH] "
                f"surface={surface}"
            )
    for tf in EXPECTED_TFS:
        surface = f"mtf_{tf.lower()}"
        if int(normalized_surfaces[surface]["fit_row_count"]) != int(
            lineage["per_tf_fit_windows"][tf]["selected_unique_row_count"]
        ):
            raise RuntimeError(
                "[ENTRY_INPUT_NORMALIZATION_MTF_LINEAGE_ROWS_MISMATCH] "
                f"tf={tf}"
            )
    expected_surface_hashes = {
        name: normalized_surfaces[name]["stats_sha256"]
        for name in EXPECTED_SURFACES
    }
    if data["surface_stats_sha256"] != expected_surface_hashes:
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_SURFACE_HASH_MAP_INVALID]"
        )
    normalized_aliases = _require_aliases(
        data["temporal_aliases"],
        signal=normalized_surfaces["signal"],
        ctx_cont=normalized_surfaces["ctx_cont"],
    )
    if (
        data["temporal_aliases"] != normalized_aliases
        or data["temporal_aliases_sha256"] != _canonical_sha256(normalized_aliases)
    ):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_ALIAS_HASH_INVALID]"
        )
    without_hash = dict(data)
    observed_hash = str(without_hash.pop("contract_sha256") or "")
    if observed_hash != _canonical_sha256(without_hash):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_CONTRACT_HASH_MISMATCH]")
    return data


def apply_surface_normalization(
    values: Any,
    surface_contract: Mapping[str, Any],
) -> np.ndarray:
    """Reference NumPy transform used by tests and offline parity audits."""

    matrix = np.asarray(values, dtype=np.float32)
    center = np.asarray(surface_contract["center"], dtype=np.float32)
    scale = np.asarray(surface_contract["scale"], dtype=np.float32)
    binary = np.asarray(surface_contract["binary_mask"], dtype=np.uint8).astype(bool)
    categorical = np.asarray(
        surface_contract["categorical_mask"], dtype=np.uint8
    ).astype(bool)
    field_names = [str(name) for name in surface_contract["field_names"]]
    if surface_contract.get("continuous_transform") != CONTINUOUS_TRANSFORM:
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_TRANSFORM_MISMATCH]")
    if matrix.shape[-1] != center.shape[0]:
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_APPLY_WIDTH_MISMATCH]")
    if not np.isfinite(matrix).all():
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_APPLY_NONFINITE]")
    # v7: no fit stamps ``binary_mask`` and ``require_surface_normalization``
    # rejects a surface that carries one, so this guard is reached only from a
    # caller that hands in an unvalidated payload. It stays because the key is
    # still in the schema and the runtime encoder still honours it.
    if binary.any():
        binary_values = matrix[..., binary]
        if not np.logical_or(binary_values == 0.0, binary_values == 1.0).all():
            raise RuntimeError(
                "[ENTRY_INPUT_NORMALIZATION_BINARY_VALUE_INVALID]"
            )
    if categorical.any():
        for index in np.flatnonzero(categorical):
            domain = surface_contract["categorical_domains"][field_names[index]]
            categorical_values = matrix[..., index]
            if (
                not np.equal(categorical_values, np.floor(categorical_values)).all()
                or not np.isin(categorical_values.astype(np.int64), domain).all()
            ):
                raise RuntimeError(
                    "[ENTRY_INPUT_NORMALIZATION_CATEGORICAL_VALUE_INVALID]"
                )
    # Compute the affine ratio in float64.  Both the observation and fitted
    # statistics are exact float32 values, but their mathematically finite
    # ratio may exceed float32 before ``asinh`` compresses it back into a
    # small finite number (for example a valid tail against a tiny learned
    # scale).  Float32 intermediate overflow would silently reintroduce a
    # numeric tail boundary even though the transform itself is non-saturating.
    normalized = (
        matrix.astype(np.float64) - center.astype(np.float64)
    ) / scale.astype(np.float64)
    identity = binary | categorical
    continuous = ~identity
    normalized[..., continuous] = np.arcsinh(normalized[..., continuous])
    normalized[..., identity] = matrix[..., identity].astype(np.float64)
    if not np.isfinite(normalized).all():
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_OUTPUT_NONFINITE]")
    return np.asarray(normalized, dtype=np.float32)


def invert_surface_normalization(
    normalized_values: Any,
    surface_contract: Mapping[str, Any],
) -> np.ndarray:
    """Invert the continuous TRAIN-fitted transform exactly up to FP32 error."""

    normalized = np.asarray(normalized_values, dtype=np.float32)
    center = np.asarray(surface_contract["center"], dtype=np.float32)
    scale = np.asarray(surface_contract["scale"], dtype=np.float32)
    binary = np.asarray(surface_contract["binary_mask"], dtype=np.uint8).astype(bool)
    categorical = np.asarray(
        surface_contract["categorical_mask"], dtype=np.uint8
    ).astype(bool)
    if surface_contract.get("continuous_transform") != CONTINUOUS_TRANSFORM:
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_TRANSFORM_MISMATCH]")
    if normalized.shape[-1] != center.shape[0]:
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_INVERSE_WIDTH_MISMATCH]")
    if not np.isfinite(normalized).all():
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_INVERSE_NONFINITE]")
    identity = binary | categorical
    continuous = ~identity
    # ``sinh(z) * scale`` is evaluated in float64 for the inverse reason: the
    # intermediate may exceed float32 while the final restored value remains
    # a valid float32 observation after multiplication by a tiny scale.
    restored = normalized.astype(np.float64)
    restored[..., continuous] = (
        np.sinh(restored[..., continuous])
        * scale[continuous].astype(np.float64)
        + center[continuous].astype(np.float64)
    )
    if not np.isfinite(restored).all():
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_INVERSE_OUTPUT_NONFINITE]")
    return np.asarray(restored, dtype=np.float32)
