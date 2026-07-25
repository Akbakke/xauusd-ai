"""Immutable TRAIN-only input-normalization contract for model-native Entry.

The model consumes raw, finite XAU feature tensors.  This contract fits one
robust transform per ordered field on TRAIN observations only and binds the
statistics directly into the model state and bundle metadata.  Binary fields
remain exact 0/1 evidence; every other field is median-centered, robustly
scaled, and clipped by one explicit model-owned bound.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "entry_model_native_input_normalization_v1"
TRANSFORM = "train_only_median_iqr_or_sparse_deviation_v1"
FIT_POPULATION = "unique_train_observation_rows_per_surface_v1"
CLIP_ABS = 12.0
SCALE_FLOOR = 1.0e-6
IQR_TO_SIGMA = 1.349
FIT_COLUMN_CHUNK = 32
MAX_TRAIN_CLIP_RATE = 0.02
EXPECTED_SURFACES = (
    "signal",
    "ctx_cont",
    "mtf_m5",
    "mtf_m15",
    "mtf_h1",
    "mtf_h4",
    "mtf_d1",
)
EXPECTED_TFS = ("M5", "M15", "H1", "H4", "D1")
CTX_CAT_DOMAINS = {
    "session_id": (0, 1, 2, 3),
    "vol_regime_id": (0, 1, 2, 3, 4),
    "atr_bucket": (0, 1, 2, 3, 4),
    "spread_bucket": (0, 1, 2, 3, 4),
    "H4_trend_sign_cat": (0, 1, 2),
}
CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS = {
    f"{tf}_regime_class_id_v2": (0, 1, 2, 3, 4)
    for tf in ("m15", "h1", "h4", "d1", "m5")
}
MTF_SEMANTIC_CATEGORICAL_DOMAINS = {
    "regime_class_id": (0, 1, 2, 3, 4),
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LINEAGE_KEYS = {
    "dataset_run_id",
    "train_parquet_path",
    "train_parquet_sha256",
    "train_manifest_path",
    "train_manifest_sha256",
    "train_row_count",
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
    if isinstance(data["train_row_count"], bool) or int(data["train_row_count"]) < 2:
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_TRAIN_ROWS_INVALID]")
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
) -> str:
    digest = hashlib.sha256()
    digest.update(b"entry_model_native_input_normalization_surface_v1\0")
    digest.update(bytes.fromhex(_field_names_sha256(field_names)))
    digest.update(np.ascontiguousarray(center, dtype="<f4").tobytes(order="C"))
    digest.update(np.ascontiguousarray(scale, dtype="<f4").tobytes(order="C"))
    digest.update(np.ascontiguousarray(binary_mask, dtype=np.uint8).tobytes(order="C"))
    digest.update(
        np.ascontiguousarray(categorical_mask, dtype=np.uint8).tobytes(order="C")
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
    matrix = _as_finite_matrix(values, surface="ctx_cat")
    names = [str(name) for name in field_names]
    if names != list(CTX_CAT_DOMAINS) or int(matrix.shape[1]) != len(names):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_CTX_CAT_FIELDS_INVALID]")
    numeric = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_CTX_CAT_VALUE_INVALID]")
    observed_counts: dict[str, dict[str, int]] = {}
    for index, name in enumerate(names):
        domain = CTX_CAT_DOMAINS[name]
        values_i = numeric[:, index].astype(np.int64)
        if not np.isin(values_i, domain).all():
            raise RuntimeError(
                f"[ENTRY_INPUT_NORMALIZATION_CTX_CAT_DOMAIN_INVALID] field={name}"
            )
        observed_counts[name] = {
            str(value): int(np.count_nonzero(values_i == value))
            for value in domain
        }
    payload = {
        "policy": "categorical_embedding_no_numeric_transform",
        "field_names": names,
        "field_names_sha256": _field_names_sha256(names),
        "fit_row_count": int(matrix.shape[0]),
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


def _as_finite_matrix(values: Any, *, surface: str) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_MATRIX_INVALID] "
            f"surface={surface} shape={tuple(matrix.shape)}"
        )
    if not np.issubdtype(matrix.dtype, np.number):
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_MATRIX_DTYPE_INVALID] "
            f"surface={surface} dtype={matrix.dtype}"
        )
    return matrix


def fit_surface_normalization(
    values: Any,
    *,
    surface: str,
    field_names: Sequence[str],
    row_count: int | None = None,
    column_chunk: int = FIT_COLUMN_CHUNK,
    semantic_categorical_domains: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Fit a bounded robust transform without materializing the full matrix.

    The input may be a memmap.  Columns are copied in small chunks so fitting
    the 513-wide TRAIN snapshot does not create another multi-gigabyte array.
    """

    matrix = _as_finite_matrix(values, surface=surface)
    names = [str(name) for name in field_names]
    if (
        not names
        or any(not name for name in names)
        or len(names) != len(set(names))
        or len(names) != int(matrix.shape[1])
    ):
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_FIELDS_INVALID] "
            f"surface={surface} fields={len(names)} width={int(matrix.shape[1])}"
        )
    if row_count is not None and int(row_count) != int(matrix.shape[0]):
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_ROW_COUNT_MISMATCH] "
            f"surface={surface} declared={int(row_count)} "
            f"observed={int(matrix.shape[0])}"
        )
    if isinstance(column_chunk, bool) or int(column_chunk) < 1:
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_CHUNK_INVALID] {column_chunk!r}"
        )

    width = int(matrix.shape[1])
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
    ):
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_CATEGORICAL_FIELDS_INVALID] "
            f"surface={surface}"
        )

    for start in range(0, width, int(column_chunk)):
        stop = min(width, start + int(column_chunk))
        block = np.asarray(matrix[:, start:stop], dtype=np.float64)
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
            is_binary = bool(
                np.logical_or(column == 0.0, column == 1.0).all()
                and (column == 0.0).any()
                and (column == 1.0).any()
            )
            if is_binary:
                center[index] = np.float32(0.0)
                scale[index] = np.float32(1.0)
                binary_mask[index] = np.uint8(1)
                scale_source[index] = "binary_identity"
                continue

            field_center = float(median[local])
            field_scale = float((q75[local] - q25[local]) / IQR_TO_SIGMA)
            source = "iqr"
            if not np.isfinite(field_scale) or field_scale <= SCALE_FLOOR:
                deviations = np.abs(column - field_center)
                positive = deviations[deviations > SCALE_FLOOR]
                if positive.size:
                    field_scale = float(np.median(positive))
                    source = "median_positive_abs_deviation"
            if not np.isfinite(field_scale) or field_scale <= SCALE_FLOOR:
                raise RuntimeError(
                    "[ENTRY_INPUT_NORMALIZATION_UNSCALEABLE] "
                    f"surface={surface} field={names[index]} "
                    f"center={field_center!r} scale={field_scale!r}"
                )
            # Sparse-event and heavy-tailed evidence families concentrate the
            # robust bulk on one value, so an IQR/MAD scale can put the
            # informative bursts beyond the clip boundary and reject the
            # mandatory feature surface wholesale. When the fitted scale
            # would clip more than the exact TRAIN cap, escalate it
            # deterministically to the smallest scale whose TRAIN clip rate
            # satisfies the cap by construction. The statistic is still
            # fitted once on the complete physical TRAIN population and
            # remains immutable bundle state; no value is rewritten.
            rounded_center = float(np.float32(field_center))
            deviations = np.abs(column - rounded_center)
            implied_clip_rate = float(
                (deviations > float(np.float32(field_scale)) * CLIP_ABS).mean()
            )
            if implied_clip_rate > MAX_TRAIN_CLIP_RATE:
                allowed = int(MAX_TRAIN_CLIP_RATE * deviations.size)
                order = deviations.size - 1 - allowed
                threshold = float(
                    np.partition(deviations, order)[order]
                )
                cap_scale = np.float32(threshold / CLIP_ABS)
                while (
                    np.isfinite(cap_scale)
                    and float(cap_scale) * CLIP_ABS < threshold
                ):
                    cap_scale = np.nextafter(cap_scale, np.float32(np.inf))
                if (
                    np.isfinite(cap_scale)
                    and float(cap_scale) > field_scale
                    and float(cap_scale) > SCALE_FLOOR
                ):
                    field_scale = float(cap_scale)
                    source = f"{source}_clip_cap_quantile"
            center[index] = np.float32(field_center)
            scale[index] = np.float32(field_scale)
            scale_source[index] = source

    if (
        not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or not (scale > np.float32(0.0)).all()
    ):
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_FLOAT32_STATS_INVALID] surface={surface}"
        )
    stats_hash = _stats_sha256(
        field_names=names,
        center=center,
        scale=scale,
        binary_mask=binary_mask,
        categorical_mask=categorical_mask,
        categorical_domains=categorical_domains,
    )
    clipped_count = np.zeros(width, dtype=np.int64)
    for start in range(0, width, int(column_chunk)):
        stop = min(width, start + int(column_chunk))
        block = np.asarray(matrix[:, start:stop], dtype=np.float64)
        normalized = (
            block - center[start:stop].astype(np.float64)
        ) / scale[start:stop].astype(np.float64)
        identity = np.logical_or(
            binary_mask[start:stop].astype(bool),
            categorical_mask[start:stop].astype(bool),
        )
        if identity.any():
            normalized[:, identity] = block[:, identity]
        clipped_count[start:stop] = np.count_nonzero(
            np.abs(normalized) > float(CLIP_ABS),
            axis=0,
        )
    clipped_rate = clipped_count.astype(np.float64) / float(matrix.shape[0])
    excessive = np.flatnonzero(clipped_rate > float(MAX_TRAIN_CLIP_RATE))
    if excessive.size:
        index = int(excessive[0])
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_TRAIN_CLIP_RATE_EXCESSIVE] "
            f"surface={surface} field={names[index]} "
            f"rate={float(clipped_rate[index]):.9f} "
            f"max={float(MAX_TRAIN_CLIP_RATE):.9f}"
        )
    return {
        "surface": str(surface),
        "field_count": width,
        "field_names": names,
        "field_names_sha256": _field_names_sha256(names),
        "fit_row_count": int(matrix.shape[0]),
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
        "train_clipped_count": [int(value) for value in clipped_count],
        "train_clipped_rate": [float(value) for value in clipped_rate],
        "max_train_clip_rate": float(MAX_TRAIN_CLIP_RATE),
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
        "train_clipped_count",
        "train_clipped_rate",
        "max_train_clip_rate",
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
    clipped_count = np.asarray(data["train_clipped_count"], dtype=np.int64)
    clipped_rate = np.asarray(data["train_clipped_rate"], dtype=np.float64)
    expected_shape = (len(names),)
    if (
        center.shape != expected_shape
        or scale.shape != expected_shape
        or binary_mask.shape != expected_shape
        or categorical_mask.shape != expected_shape
        or len(scale_source) != len(names)
        or clipped_count.shape != expected_shape
        or clipped_rate.shape != expected_shape
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
        or (clipped_count < 0).any()
        or not np.isfinite(clipped_rate).all()
        or (clipped_rate < 0.0).any()
        or (clipped_rate > float(MAX_TRAIN_CLIP_RATE)).any()
        or not np.allclose(
            clipped_rate,
            clipped_count.astype(np.float64) / float(data["fit_row_count"]),
            rtol=0.0,
            atol=1e-15,
        )
        or float(data["max_train_clip_rate"]) != float(MAX_TRAIN_CLIP_RATE)
    ):
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_SURFACE_VALUES_INVALID] surface={surface}"
        )
    for index, is_binary in enumerate(binary_mask.astype(bool)):
        if is_binary and (
            center[index] != np.float32(0.0)
            or scale[index] != np.float32(1.0)
            or scale_source[index] != "binary_identity"
        ):
            raise RuntimeError(
                f"[ENTRY_INPUT_NORMALIZATION_BINARY_CONTRACT_INVALID] "
                f"surface={surface} field={names[index]}"
            )
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
            ):
                raise RuntimeError(
                    f"[ENTRY_INPUT_NORMALIZATION_CATEGORICAL_CONTRACT_INVALID] "
                    f"surface={surface} field={names[index]}"
                )
        if not is_binary and not is_categorical and scale_source[index] not in {
            "iqr",
            "median_positive_abs_deviation",
        }:
            raise RuntimeError(
                f"[ENTRY_INPUT_NORMALIZATION_SCALE_SOURCE_INVALID] "
                f"surface={surface} field={names[index]}"
            )
    expected_hash = _stats_sha256(
        field_names=names,
        center=center,
        scale=scale,
        binary_mask=binary_mask,
        categorical_mask=categorical_mask,
        categorical_domains=categorical_domains,
    )
    if data["stats_sha256"] != expected_hash:
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_STATS_HASH_MISMATCH] surface={surface}"
        )
    return data


def share_temporal_alias_stats_from_ctx(
    signal_surface: Mapping[str, Any],
    ctx_cont_surface: Mapping[str, Any],
    *,
    temporal_aliases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Make ctx_cont the sole numerical-statistics owner for signal aliases."""

    signal = dict(signal_surface)
    ctx_cont = dict(ctx_cont_surface)
    signal_names = [str(name) for name in signal["field_names"]]
    ctx_names = [str(name) for name in ctx_cont["field_names"]]
    center = np.asarray(signal["center"], dtype=np.float32).copy()
    scale = np.asarray(signal["scale"], dtype=np.float32).copy()
    binary = np.asarray(signal["binary_mask"], dtype=np.uint8).copy()
    categorical = np.asarray(signal["categorical_mask"], dtype=np.uint8).copy()
    scale_source = [str(value) for value in signal["scale_source"]]
    clipped_count = [int(value) for value in signal["train_clipped_count"]]
    clipped_rate = [float(value) for value in signal["train_clipped_rate"]]
    categorical_domains = {
        str(name): [int(item) for item in domain]
        for name, domain in signal["categorical_domains"].items()
    }
    ctx_center = np.asarray(ctx_cont["center"], dtype=np.float32)
    ctx_scale = np.asarray(ctx_cont["scale"], dtype=np.float32)
    ctx_binary = np.asarray(ctx_cont["binary_mask"], dtype=np.uint8)
    ctx_categorical = np.asarray(ctx_cont["categorical_mask"], dtype=np.uint8)
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
        center[signal_index] = ctx_center[ctx_index]
        scale[signal_index] = ctx_scale[ctx_index]
        binary[signal_index] = ctx_binary[ctx_index]
        categorical[signal_index] = ctx_categorical[ctx_index]
        scale_source[signal_index] = ctx_cont["scale_source"][ctx_index]
        clipped_count[signal_index] = int(
            ctx_cont["train_clipped_count"][ctx_index]
        )
        clipped_rate[signal_index] = float(
            ctx_cont["train_clipped_rate"][ctx_index]
        )
        categorical_domains.pop(alias["signal_field"], None)
        ctx_domain = ctx_cont["categorical_domains"].get(alias["ctx_cont_field"])
        if ctx_domain is not None:
            categorical_domains[alias["signal_field"]] = [
                int(item) for item in ctx_domain
            ]
    signal.update(
        {
            "center": [float(value) for value in center],
            "scale": [float(value) for value in scale],
            "binary_mask": [int(value) for value in binary],
            "binary_field_count": int(binary.sum()),
            "categorical_mask": [int(value) for value in categorical],
            "categorical_field_count": int(categorical.sum()),
            "categorical_domains": categorical_domains,
            "scale_source": scale_source,
            "train_clipped_count": clipped_count,
            "train_clipped_rate": clipped_rate,
        }
    )
    signal["stats_sha256"] = _stats_sha256(
        field_names=signal_names,
        center=center,
        scale=scale,
        binary_mask=binary,
        categorical_mask=categorical,
        categorical_domains=categorical_domains,
    )
    return signal


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
        normalized_lineage["train_row_count"]
    ):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_CTX_CAT_LINEAGE_ROWS_MISMATCH]"
        )
    for surface in ("signal", "ctx_cont"):
        if int(normalized_surfaces[surface]["fit_row_count"]) != int(
            normalized_lineage["train_row_count"]
        ):
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
        "clip_abs": float(CLIP_ABS),
        "scale_floor": float(SCALE_FLOOR),
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
        "clip_abs",
        "scale_floor",
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
        "clip_abs": float(CLIP_ABS),
        "scale_floor": float(SCALE_FLOOR),
    }
    if any(data.get(key) != expected for key, expected in exact.items()):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_CONTRACT_IDENTITY_INVALID]")
    lineage = _require_lineage(data["lineage"])
    ctx_cat = require_ctx_cat_contract(
        data["ctx_cat"],
        field_names=expected_ctx_cat_names,
    )
    if int(ctx_cat["fit_row_count"]) != int(lineage["train_row_count"]):
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
    for surface in ("signal", "ctx_cont"):
        if int(normalized_surfaces[surface]["fit_row_count"]) != int(
            lineage["train_row_count"]
        ):
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
    *,
    clip_abs: float = CLIP_ABS,
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
    if matrix.shape[-1] != center.shape[0]:
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_APPLY_WIDTH_MISMATCH]")
    if not np.isfinite(matrix).all():
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_APPLY_NONFINITE]")
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
    normalized = (matrix - center) / scale
    normalized[..., binary] = matrix[..., binary]
    normalized[..., categorical] = matrix[..., categorical]
    normalized = np.clip(normalized, -float(clip_abs), float(clip_abs))
    if not np.isfinite(normalized).all():
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_OUTPUT_NONFINITE]")
    return np.asarray(normalized, dtype=np.float32)
