"""Exact TRAIN-only fit helper for model-native Entry input normalization.

This module owns population selection and artifact lineage only.  The
normalization math and immutable schema remain owned by
``entry_model_native_input_normalization_v1``.  In particular:

* all physical TRAIN decision rows are scanned before any sampler can act;
* signal statistics are fitted once on TRAIN snapshots, never on overlapping
  sequence windows;
* current-bar signal/context aliases are derived from the routing contract,
  checked bit-for-bit, and inherit their statistics from ``ctx_cont``;
* every MTF source bar used by at least one causal TRAIN window is fitted once;
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
    MTF_SEMANTIC_CATEGORICAL_DOMAINS,
    build_input_normalization_contract,
    fit_ctx_cat_contract,
    fit_surface_normalization,
    share_temporal_alias_stats_from_ctx,
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
    HTF_V2_MATRIX_CONTRACT,
    HTF_V3_MATRIX_CONTRACT,
    MULTI_TF_FEATURE_COUNT_V2,
    MULTI_TF_FEATURE_COUNT_V3,
    MULTI_TF_PER_BAR_FEATURES_V2,
    MULTI_TF_PER_BAR_FEATURES_V3,
    MULTI_TF_SHIFT,
    MultiTFV2DiskCache,
    load_multi_tf_v2_cache,
)


FIT_HELPER_SCHEMA_VERSION = "entry_v10_train_input_normalization_fit_v1"
FIT_POPULATION_PROOF_SCHEMA_VERSION = (
    "entry_v10_train_input_normalization_population_proof_v1"
)
TARGET_AVAILABILITY_SHIFT = pd.Timedelta(minutes=5)
DEFAULT_ROW_CHUNK = 4096
_SHA256_BUFFER_SIZE = 1024 * 1024


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
    selected_timestamps = np.ascontiguousarray(
        timestamps_ns[selected_indices], dtype="<i8"
    )
    selected_values = np.ascontiguousarray(values[selected_indices], dtype="<f4")
    digest = hashlib.sha256()
    digest.update(b"entry_v10_normalization_selected_mtf_rows_v1\0")
    digest.update(str(tf).encode("ascii"))
    digest.update(b"\0")
    digest.update(bytes.fromhex(_field_names_sha256(field_names)))
    digest.update(np.asarray(selected_values.shape, dtype="<i8").tobytes())
    digest.update(selected_indices.tobytes(order="C"))
    digest.update(selected_timestamps.tobytes(order="C"))
    digest.update(selected_values.tobytes(order="C"))
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


_MTF_PER_BAR_CONTRACTS = {
    HTF_V2_MATRIX_CONTRACT: (MULTI_TF_FEATURE_COUNT_V2, MULTI_TF_PER_BAR_FEATURES_V2),
    HTF_V3_MATRIX_CONTRACT: (MULTI_TF_FEATURE_COUNT_V3, MULTI_TF_PER_BAR_FEATURES_V3),
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
    """Return the exact union of source rows consumed by TRAIN MTF windows."""

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
    train_ts = np.asarray(train_times_ns)
    if (
        train_ts.dtype != np.dtype(np.int64)
        or train_ts.ndim != 1
        or train_ts.size < 2
        or np.any(np.diff(train_ts) <= 0)
    ):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_TRAIN_TIMES_INVALID]")
    source_ts, source_values, warmup = _extract_mtf_source(source, tf=tf)
    availability_ns = int(TARGET_AVAILABILITY_SHIFT.value)
    shift_ns = int(MULTI_TF_SHIFT[tf].value)
    if np.any(train_ts > np.iinfo(np.int64).max - availability_ns):
        raise RuntimeError("[ENTRY_INPUT_NORMALIZATION_TIME_OVERFLOW]")
    cutoffs = train_ts + availability_ns - shift_ns
    right = np.searchsorted(source_ts, cutoffs, side="right").astype(
        np.int64, copy=False
    )
    left = right - int(seq_len)
    invalid = np.flatnonzero((right < int(seq_len)) | (left < int(warmup)))
    if invalid.size:
        row = int(invalid[0])
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_MTF_WARMUP_INCOMPLETE] "
            f"tf={tf} train_row={row} left={int(left[row])} "
            f"right={int(right[row])} warmup={warmup} seq_len={int(seq_len)}"
        )

    coverage_delta = np.zeros(len(source_ts) + 1, dtype=np.int64)
    np.add.at(coverage_delta, left, 1)
    np.add.at(coverage_delta, right, -1)
    selected_indices = np.flatnonzero(
        np.cumsum(coverage_delta[:-1], dtype=np.int64) > 0
    ).astype(np.int64, copy=False)
    if selected_indices.size < 2:
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_MTF_POPULATION_EMPTY] tf={tf}"
        )
    selected_values = np.ascontiguousarray(
        source_values[selected_indices], dtype=np.float32
    )
    if not np.isfinite(selected_values).all():
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_MTF_SELECTED_NONFINITE] tf={tf}"
        )

    _sel_count, _sel_names = resolve_mtf_per_bar_contract(source, tf=tf)
    ema_stack_index = list(_sel_names).index("ema_stack_aligned_v2")
    if not np.isin(
        selected_values[:, ema_stack_index], (-1.0, 0.0, 1.0)
    ).all():
        raise RuntimeError(
            f"[ENTRY_INPUT_NORMALIZATION_MTF_EMA_STACK_DOMAIN_INVALID] tf={tf}"
        )
    regime_index = list(_sel_names).index("regime_class_id")
    if not np.isin(
        selected_values[:, regime_index],
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
            "union_of_causal_train_windows_each_source_row_once"
        ),
        "target_availability_shift_seconds": int(
            TARGET_AVAILABILITY_SHIFT.total_seconds()
        ),
        "tf_shift_seconds": int(MULTI_TF_SHIFT[tf].total_seconds()),
        "seq_len": int(seq_len),
        "source_row_count": int(len(source_ts)),
        "source_warmup_rows": int(warmup),
        **window,
    }
    proof["selection_proof_sha256"] = _canonical_sha256(proof)
    return selected_values, window, proof


def _verify_artifacts_and_load_mtf(
    *,
    artifacts: TrainNormalizationArtifacts,
    ordered_signal_names: Sequence[str],
) -> tuple[dict[str, Any], MultiTFV2DiskCache]:
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
    cache = load_multi_tf_v2_cache(cache_dir)
    if not isinstance(cache, MultiTFV2DiskCache):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_MTF_CACHE_IDENTITY_MISSING]"
        )
    # The verified V2 cache binds its own full-history canonical M5 source
    # (the cascade-audited canonical-v3 parquet); m5_prebuilt is the separate
    # model-range seq/snapshot source. Prove the cache against its own
    # declared source bytes — equality with m5_prebuilt would force the
    # five-timeframe surfaces onto the model-range-trimmed file.
    cache_source = Path(str(cache.m5_prebuilt_source)).expanduser()
    m5_sha256 = _sha256_file(m5_prebuilt)
    if (
        not cache_source.is_absolute()
        or not cache_source.is_file()
        or cache_source.is_symlink()
        or _sha256_file(cache_source) != cache.m5_prebuilt_source_sha256
        or tuple(cache) != EXPECTED_TFS
    ):
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_MTF_CACHE_LINEAGE_MISMATCH]"
        )
    cache_manifest = _exact_regular_file(
        cache_dir / "manifest.json", label="mtf_cache_manifest"
    )
    manifest_sha256 = _sha256_file(cache_manifest)
    if cache.manifest_sha256 != manifest_sha256:
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_MTF_CACHE_MANIFEST_SHA_MISMATCH]"
        )
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


def fit_entry_v10_train_input_normalization(
    *,
    train_seq: Any,
    train_snap: Any,
    train_ctx_cont: Any,
    train_ctx_cat: Any,
    train_times: Any,
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

    ctx_surface = fit_surface_normalization(
        ctx_cont,
        surface="ctx_cont",
        field_names=MODEL_NATIVE_CTX_CONT_FIELDS,
        row_count=row_count,
        semantic_categorical_domains=(
            CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS
        ),
    )
    signal_surface_raw = fit_surface_normalization(
        snap,
        surface="signal",
        field_names=signal_names,
        row_count=row_count,
    )
    signal_surface = share_temporal_alias_stats_from_ctx(
        signal_surface_raw,
        ctx_surface,
        temporal_aliases=aliases,
    )
    surfaces: dict[str, Mapping[str, Any]] = {
        "signal": signal_surface,
        "ctx_cont": ctx_surface,
    }
    tf_windows: dict[str, dict[str, Any]] = {}
    tf_population_proofs: dict[str, dict[str, Any]] = {}
    for tf in EXPECTED_TFS:
        selected_values, window, selection_proof = (
            select_causal_mtf_fit_population(
                tf=tf,
                source=multi_tf_sources[tf],
                train_times_ns=train_times_ns,
                seq_len=int(per_tf_seq_lens[tf]),
            )
        )
        surface_name = f"mtf_{tf.lower()}"
        _fit_count, _fit_names = resolve_mtf_per_bar_contract(
            multi_tf_sources[tf], tf=tf
        )
        surfaces[surface_name] = fit_surface_normalization(
            selected_values,
            surface=surface_name,
            field_names=_fit_names,
            row_count=int(selected_values.shape[0]),
            semantic_categorical_domains=MTF_SEMANTIC_CATEGORICAL_DOMAINS,
        )
        tf_windows[tf] = window
        tf_population_proofs[tf] = selection_proof
    if tuple(surfaces) != EXPECTED_SURFACES:
        raise RuntimeError(
            "[ENTRY_INPUT_NORMALIZATION_SURFACE_ORDER_INVALID]"
        )

    fit_start_utc = _timestamp_iso_utc(train_times_ns[0])
    fit_end_utc = _timestamp_iso_utc(train_times_ns[-1])
    lineage = {
        **base_lineage,
        "train_row_count": row_count,
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
        ctx_cat,
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

    train_indices = np.arange(row_count, dtype=np.int64)
    population_proof = {
        "schema_version": FIT_POPULATION_PROOF_SCHEMA_VERSION,
        "fit_scope": "train_only",
        "signal_population": (
            "one_full_train_snapshot_per_unique_decision_row"
        ),
        "ctx_cont_population": "one_full_train_row_per_decision_row",
        "ctx_cat_population": "one_full_train_row_per_decision_row",
        "sequence_population": (
            "validation_only_seq_last_equals_snap_not_flattened_for_fit"
        ),
        "train_decision_row_count": row_count,
        "train_decision_row_indices_sha256": _hash_int64_indices(
            train_indices, namespace="train_decision_rows"
        ),
        "train_decision_row_values_sha256": _hash_train_decision_rows(
            train_times_ns=train_times_ns,
            snap=snap,
            ctx_cont=ctx_cont,
            ctx_cat=ctx_cat,
            row_chunk=int(row_chunk),
        ),
        "val_fit_row_count": 0,
        "test_fit_row_count": 0,
        "temporal_alias_count": len(aliases),
        "temporal_aliases_sha256": normalization_contract[
            "temporal_aliases_sha256"
        ],
        "mtf_populations": tf_population_proofs,
    }
    population_proof["proof_sha256"] = _canonical_sha256(population_proof)
    return {
        "schema_version": FIT_HELPER_SCHEMA_VERSION,
        "normalization_contract": normalization_contract,
        "fit_population_proof": population_proof,
    }
