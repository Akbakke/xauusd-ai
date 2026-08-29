#!/usr/bin/env python3
"""Copy the causal pre-TEST prefix of a V4 MTF cache into a new cache.

This is deliberately not a normal ``load_multi_tf_v4_cache`` call.  The old
cache extends into TEST and the normal loader integrity-checks every array
byte, which would read TEST data.  Here the new sealed M5 source determines the
exact pre-TEST axis for every clock.  We memory-map the old arrays, read only
``[:safe_axis_length]`` and prove those timestamps equal the independently
derived safe axes before publishing a new immutable cache.

V4 features are causal and the copied prefix is therefore the same feature
history that was available at each pre-TEST timestamp.  No TEST array element
is read, hashed, inspected, or included in the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"PRETEST_MTF_CACHE_{label}_FILE_REQUIRED:{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"PRETEST_MTF_CACHE_{label}_JSON_INVALID:{path}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"PRETEST_MTF_CACHE_{label}_OBJECT_REQUIRED:{path}")
    return data


def _utc(value: str | pd.Timestamp, *, label: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise RuntimeError(f"PRETEST_MTF_CACHE_{label}_UTC_REQUIRED")
    return stamp.tz_convert("UTC")


def copy_safe_prefix(
    *,
    array_path: Path,
    safe_length: int,
    expected_shape_tail: tuple[int, ...],
    dtype: np.dtype,
    label: str,
) -> np.ndarray:
    """Copy only an explicitly bounded prefix from a memory-mapped array."""

    if safe_length <= 0:
        raise RuntimeError(f"PRETEST_MTF_CACHE_{label}_SAFE_LENGTH_INVALID")
    if not array_path.is_absolute() or array_path.is_symlink() or not array_path.is_file():
        raise RuntimeError(f"PRETEST_MTF_CACHE_{label}_ARRAY_REQUIRED:{array_path}")
    values = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if (
        values.dtype != dtype
        or values.ndim != len(expected_shape_tail) + 1
        or tuple(values.shape[1:]) != expected_shape_tail
        or values.shape[0] < safe_length
    ):
        raise RuntimeError(
            f"PRETEST_MTF_CACHE_{label}_ARRAY_SHAPE_INVALID:"
            f"observed={values.shape}/{values.dtype}:safe_length={safe_length}"
        )
    # Slicing ends at safe_length.  No boundary element and no TEST suffix is
    # faulted into memory by this code path.
    return np.ascontiguousarray(values[:safe_length], dtype=dtype)


def _safe_source_index(path: Path, *, test_start: pd.Timestamp) -> pd.DatetimeIndex:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"PRETEST_MTF_CACHE_SAFE_SOURCE_REQUIRED:{path}")
    source = pd.read_parquet(path, columns=["time"])
    index = pd.DatetimeIndex(pd.to_datetime(source["time"], utc=True, errors="raise"))
    if index.empty or not index.is_monotonic_increasing or not index.is_unique:
        raise RuntimeError("PRETEST_MTF_CACHE_SAFE_SOURCE_TIME_AXIS_INVALID")
    if (index >= test_start).any():
        raise RuntimeError("PRETEST_MTF_CACHE_SAFE_SOURCE_TEST_ROW_BLOCKED")
    return index


def _copy_pretest_frames(
    *,
    source_cache_dir: Path,
    safe_source_index: pd.DatetimeIndex,
    test_start: pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    from gx1.features.htf_features import (
        HTF_V4_CACHE_BUILDER_VERSION,
        HTF_V4_CACHE_SCHEMA_VERSION,
        HTF_V4_MATRIX_CONTRACT,
        MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4,
        MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4,
        MULTI_TF_FEATURE_COUNT_V4,
        MULTI_TF_PER_BAR_FEATURES_V4,
        MULTI_TF_RESAMPLE_RULES,
        build_multi_tf_v4_closed_timestamp_indices,
        validate_causal_feature_matrix,
    )

    source_dir = source_cache_dir.expanduser()
    if not source_dir.is_absolute() or source_dir.is_symlink() or not source_dir.is_dir():
        raise RuntimeError(f"PRETEST_MTF_CACHE_SOURCE_DIR_INVALID:{source_dir}")
    source_dir = source_dir.resolve(strict=True)
    manifest_path = source_dir / "manifest.json"
    manifest = _read_json(manifest_path, label="SOURCE_MANIFEST")
    if (
        manifest.get("schema_version") != HTF_V4_CACHE_SCHEMA_VERSION
        or manifest.get("builder_version") != HTF_V4_CACHE_BUILDER_VERSION
        or manifest.get("feature_names") != list(MULTI_TF_PER_BAR_FEATURES_V4)
        or manifest.get("feature_count") != MULTI_TF_FEATURE_COUNT_V4
    ):
        raise RuntimeError("PRETEST_MTF_CACHE_SOURCE_CONTRACT_INVALID")
    expected_axes = build_multi_tf_v4_closed_timestamp_indices(safe_source_index)
    frames: dict[str, pd.DataFrame] = {}
    evidence: dict[str, Any] = {}
    for timeframe in MULTI_TF_RESAMPLE_RULES:
        axis = expected_axes[timeframe]
        if axis.empty or (axis >= test_start).any():
            raise RuntimeError(f"PRETEST_MTF_CACHE_{timeframe}_SAFE_AXIS_INVALID")
        entry = manifest.get("tfs", {}).get(timeframe)
        if not isinstance(entry, dict):
            raise RuntimeError(f"PRETEST_MTF_CACHE_{timeframe}_MANIFEST_ENTRY_MISSING")
        scalar_fields = tuple(MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4[timeframe])
        if (
            entry.get("feature_count") != MULTI_TF_FEATURE_COUNT_V4
            or tuple(entry.get("model_native_scalar_fields") or ()) != scalar_fields
            or entry.get("model_native_scalar_contract") != MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4
        ):
            raise RuntimeError(f"PRETEST_MTF_CACHE_{timeframe}_MANIFEST_CONTRACT_INVALID")
        safe_length = len(axis)
        timestamps = copy_safe_prefix(
            array_path=source_dir / str(entry.get("ts_npy") or ""),
            safe_length=safe_length,
            expected_shape_tail=(),
            dtype=np.dtype(np.int64),
            label=f"{timeframe}_TIMESTAMPS",
        )
        expected_ns = axis.asi8.astype(np.int64, copy=False)
        if not np.array_equal(timestamps, expected_ns):
            raise RuntimeError(f"PRETEST_MTF_CACHE_{timeframe}_PREFIX_TIMESTAMP_MISMATCH")
        feature_values = copy_safe_prefix(
            array_path=source_dir / str(entry.get("feats_npy") or ""),
            safe_length=safe_length,
            expected_shape_tail=(MULTI_TF_FEATURE_COUNT_V4,),
            dtype=np.dtype(np.float32),
            label=f"{timeframe}_FEATURES",
        )
        scalar_values = copy_safe_prefix(
            array_path=source_dir / str(entry.get("model_native_scalars_npy") or ""),
            safe_length=safe_length,
            expected_shape_tail=(len(scalar_fields),),
            dtype=np.dtype(np.float32),
            label=f"{timeframe}_SCALARS",
        )
        warmup = validate_causal_feature_matrix(
            feature_values,
            expected_width=MULTI_TF_FEATURE_COUNT_V4,
            context=f"PRETEST_MTF_CACHE_{timeframe}",
        )
        frame = pd.DataFrame(
            feature_values,
            index=axis,
            columns=MULTI_TF_PER_BAR_FEATURES_V4,
            copy=False,
        )
        frame.attrs["ts_int64"] = timestamps
        frame.attrs["feats_np"] = feature_values
        frame.attrs["causal_warmup_rows"] = warmup
        frame.attrs["htf_feature_contract"] = HTF_V4_MATRIX_CONTRACT
        frame.attrs["model_native_mtf_scalars_np_v4"] = scalar_values
        frame.attrs["model_native_mtf_scalar_fields_v4"] = scalar_fields
        frame.attrs["model_native_mtf_scalar_contract_v4"] = MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4
        frame.attrs["model_native_mtf_scalar_warmup_rows_v4"] = int(
            entry.get("model_native_scalar_warmup_rows") or 0
        )
        frames[timeframe] = frame
        evidence[timeframe] = {
            "safe_rows_copied": safe_length,
            "time_min_utc": axis.min().isoformat(),
            "time_max_utc": axis.max().isoformat(),
            "test_rows_copied": 0,
        }
    return frames, manifest, evidence


def materialize_pretest_mtf_cache(
    *,
    source_cache_dir: Path,
    safe_m5_source: Path,
    test_start_utc: str | pd.Timestamp,
    volatility_squeeze_manifest: Path,
    expected_volatility_squeeze_manifest_sha256: str,
    out_dir: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    """Publish a verified V4 cache containing only the causal pre-TEST prefix."""

    from gx1.features.volatility_squeeze_state_v1 import (
        load_volatility_squeeze_artifact_manifest,
    )
    from gx1.scripts.prebuild_multi_tf_cache_v4 import publish_multi_tf_v4_cache

    test_start = _utc(test_start_utc, label="TEST_START")
    source = safe_m5_source.expanduser()
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise RuntimeError(f"PRETEST_MTF_CACHE_SAFE_SOURCE_REQUIRED:{source}")
    source = source.resolve(strict=True)
    index = _safe_source_index(source, test_start=test_start)
    frames, old_manifest, evidence = _copy_pretest_frames(
        source_cache_dir=source_cache_dir,
        safe_source_index=index,
        test_start=test_start,
    )
    constants = old_manifest.get("v29_registry_constants")
    if not isinstance(constants, dict):
        raise RuntimeError("PRETEST_MTF_CACHE_REGISTRY_CONSTANTS_MISSING")
    squeeze = load_volatility_squeeze_artifact_manifest(
        volatility_squeeze_manifest.expanduser().resolve(strict=True),
        expected_sha256=expected_volatility_squeeze_manifest_sha256,
    )
    published = publish_multi_tf_v4_cache(
        out_dir=out_dir.expanduser(),
        m5_prebuilt=source,
        expected_source_sha256=_sha256_file(source),
        features=frames,
        v29_registry_constants=constants,
        volatility_squeeze_artifacts=squeeze,
    )
    report = {
        "schema_version": "gx1_pretest_mtf_cache_copy_v1",
        "decision": "PASS",
        "test_accessed": False,
        "test_start_utc": test_start.isoformat(),
        "source_cache_manifest": str(Path(source_cache_dir).expanduser() / "manifest.json"),
        "safe_m5_source": str(source),
        "safe_m5_source_sha256": _sha256_file(source),
        "published_manifest": str(published),
        "published_manifest_sha256": _sha256_file(published),
        "timeframes": evidence,
    }
    evidence_path = evidence_path.expanduser()
    if not evidence_path.is_absolute() or evidence_path.exists() or evidence_path.is_symlink():
        raise RuntimeError(f"PRETEST_MTF_CACHE_EVIDENCE_PATH_INVALID:{evidence_path}")
    with evidence_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache-dir", type=Path, required=True)
    parser.add_argument("--safe-m5-source", type=Path, required=True)
    parser.add_argument("--test-start-utc", required=True)
    parser.add_argument("--volatility-squeeze-manifest", type=Path, required=True)
    parser.add_argument("--expected-volatility-squeeze-manifest-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    args = parser.parse_args()
    report = materialize_pretest_mtf_cache(
        source_cache_dir=args.source_cache_dir,
        safe_m5_source=args.safe_m5_source,
        test_start_utc=args.test_start_utc,
        volatility_squeeze_manifest=args.volatility_squeeze_manifest,
        expected_volatility_squeeze_manifest_sha256=args.expected_volatility_squeeze_manifest_sha256,
        out_dir=args.out_dir,
        evidence_path=args.evidence_path,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
