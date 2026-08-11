#!/usr/bin/env python3
"""Build and atomically publish the sole immutable V4 multi-timeframe cache.

The source must be one exact absolute native-M5 enriched parquet. The fixed
M5/M15/H1/H4/D1 cache contains ``MULTI_TF_FEATURE_COUNT_V4`` ordered float32
fields per timeframe (derived from the declared name tuples), exact int64
timestamps, byte hashes, full-input liveness evidence, the immutable
TRAIN-fitted V29 registry constants, and one canonical identity. There is no
contract switch and no historical publisher.

The V29 registry constants (level cluster tolerance / trendline band per TF)
are fitted here, once, on the declared TRAIN window (rule 18) with the
explicit recipe input ``--level-tol-quantile-q`` (the design doc declares no
default for the quantile; see ENTRY_LEVEL_REGISTRY_TOL_QUANTILE_Q=REQUIRED in
the recipe owner) and frozen with full provenance in the cache manifest.

Usage:
    python -m gx1.scripts.prebuild_multi_tf_cache_v4 \
        --m5-prebuilt /absolute/xauusd_m5_enriched.parquet \
        --expected-source-sha256 <exact-lowercase-sha256> \
        --level-tol-quantile-q <q in (0,1), explicit recipe input> \
        --registry-fit-train-end <UTC timestamp of the declared TRAIN end> \
        --out-dir /absolute/MULTI_TF_V4_CACHE
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_exact_sha256(value: object, *, label: str) -> str:
    observed = str(value)
    if len(observed) != 64 or any(
        ch not in "0123456789abcdef" for ch in observed
    ):
        raise RuntimeError(f"{label}: exact lowercase SHA-256 required")
    return observed


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _rename_dir_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("HTF_V4_CACHE_ATOMIC_PUBLISH_UNAVAILABLE")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise RuntimeError(
                f"HTF_V4_CACHE_OUTPUT_EXISTS: immutable output exists: {destination}"
            )
        raise RuntimeError(
            f"HTF_V4_CACHE_ATOMIC_PUBLISH_FAILED: {os.strerror(code)}"
        )


def _write_npy_fsync(path: Path, values: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_fsync(path: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())



def publish_multi_tf_v4_cache(
    *,
    out_dir: Path,
    m5_prebuilt: Path,
    expected_source_sha256: str,
    features: dict,
    v29_registry_constants: dict,
) -> Path:
    from gx1.features.htf_features import (
        HTF_V4_CACHE_BUILDER_VERSION,
        HTF_V4_CACHE_SCHEMA_VERSION,
        HTF_V4_MATRIX_CONTRACT,
        MULTI_TF_FEATURE_COUNT_V4,
        MULTI_TF_PER_BAR_FEATURES_V4,
        MULTI_TF_RESAMPLE_RULES,
        MULTI_TF_SHIFT,
        build_multi_tf_v4_closed_timestamp_indices,
        build_multi_tf_v4_liveness_contract,
        compute_htf_v4_cache_identity,
        require_multi_tf_v4_frames,
        require_v29_registry_constants,
        validate_causal_feature_matrix,
    )

    v29_registry_constants = require_v29_registry_constants(
        v29_registry_constants
    )

    SCHEMA_VERSION = HTF_V4_CACHE_SCHEMA_VERSION
    BUILDER_VERSION = HTF_V4_CACHE_BUILDER_VERSION
    MATRIX_CONTRACT = HTF_V4_MATRIX_CONTRACT
    FEATURE_COUNT = MULTI_TF_FEATURE_COUNT_V4
    FEATURE_NAMES = MULTI_TF_PER_BAR_FEATURES_V4

    source = m5_prebuilt.expanduser()
    expected_source_sha256 = _require_exact_sha256(
        expected_source_sha256,
        label="HTF_V4_CACHE_SOURCE_INVALID",
    )
    if (
        not source.is_absolute()
        or source.is_symlink()
        or not source.is_file()
        or source.resolve() != source
    ):
        raise RuntimeError(
            f"HTF_V4_CACHE_SOURCE_INVALID: exact absolute regular file required: {source}"
        )
    if _sha256_file(source) != expected_source_sha256:
        raise RuntimeError("HTF_V4_CACHE_SOURCE_CHANGED_DURING_BUILD")
    require_multi_tf_v4_frames(features)
    expected_v4_indices: dict[str, pd.DatetimeIndex]
    try:
        source_time = pd.read_parquet(source, columns=["time"])["time"]
    except Exception as exc:
        raise RuntimeError(
            "HTF_V4_CACHE_SOURCE_TIMESTAMP_GEOMETRY_INVALID: "
            f"cannot read exact time column: {exc}"
        ) from exc
    source_index = pd.DatetimeIndex(
        pd.to_datetime(source_time, utc=True, errors="coerce")
    )
    expected_v4_indices = build_multi_tf_v4_closed_timestamp_indices(
        source_index
    )

    requested_out = out_dir.expanduser()
    if not requested_out.is_absolute():
        raise RuntimeError("HTF_V4_CACHE_OUTPUT_INVALID: --out-dir must be absolute")
    if requested_out.exists() or requested_out.is_symlink():
        raise RuntimeError(
            f"HTF_V4_CACHE_OUTPUT_EXISTS: immutable output exists: {requested_out}"
        )
    if any(
        component.is_symlink()
        for component in (requested_out.parent, *requested_out.parent.parents)
    ):
        raise RuntimeError(
            "HTF_V4_CACHE_OUTPUT_INVALID: parent path traverses a symlink"
        )
    parent = requested_out.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise RuntimeError(
            f"HTF_V4_CACHE_OUTPUT_INVALID: parent must be a real directory: {parent}"
        )
    destination = parent / requested_out.name
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=str(parent))
    )
    published = False
    try:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "feature_count": int(FEATURE_COUNT),
            "feature_names": list(FEATURE_NAMES),
            "shift_contract": {
                tf: str(shift) for tf, shift in MULTI_TF_SHIFT.items()
            },
            "builder_version": BUILDER_VERSION,
            "m5_prebuilt_source": str(source),
            "m5_prebuilt_source_sha256": expected_source_sha256,
            "v29_registry_constants": v29_registry_constants,
            "tfs": {},
        }
        for tf in MULTI_TF_RESAMPLE_RULES:
            frame = features[tf]
            if not isinstance(frame, pd.DataFrame):
                raise RuntimeError(
                    f"HTF_V4_CACHE_FEATURE_SET_INVALID: {tf} is not a DataFrame"
                )
            feats_np = np.asarray(frame.attrs.get("feats_np"))
            ts_int64 = np.asarray(frame.attrs.get("ts_int64"))
            if (
                tuple(frame.columns) != tuple(FEATURE_NAMES)
                or not isinstance(frame.index, pd.DatetimeIndex)
                or frame.index.tz is None
                or not frame.index.is_unique
                or not frame.index.is_monotonic_increasing
                or feats_np.dtype != np.dtype(np.float32)
                or feats_np.ndim != 2
                or feats_np.shape[1] != FEATURE_COUNT
                or len(frame) != len(feats_np)
                or ts_int64.dtype != np.dtype(np.int64)
                or ts_int64.shape != (len(feats_np),)
                or len(ts_int64) == 0
                or np.any(np.diff(ts_int64) <= 0)
                or not np.array_equal(frame.index.asi8, ts_int64)
            ):
                raise RuntimeError(
                    f"HTF_V4_CACHE_FEATURE_SET_INVALID: malformed {tf} arrays"
                )
            if not frame.index.equals(expected_v4_indices[tf]):
                raise RuntimeError(
                    "HTF_V4_CACHE_SOURCE_TIMESTAMP_GEOMETRY_MISMATCH: "
                    f"{tf}"
                )
            warmup_rows = validate_causal_feature_matrix(
                feats_np,
                expected_width=FEATURE_COUNT,
                context=f"HTF_V4_CACHE_PUBLISH_{tf}",
            )
            if warmup_rows == len(feats_np):
                raise RuntimeError(
                    f"HTF_V4_CACHE_WARMUP_INCOMPLETE: {tf} has no complete row"
                )
            if (
                frame.attrs.get("causal_warmup_rows") != warmup_rows
                or frame.attrs.get("htf_feature_contract") != MATRIX_CONTRACT
            ):
                raise RuntimeError(
                    f"HTF_V4_CACHE_FEATURE_SET_INVALID: {tf} attrs mismatch"
                )

            feats_path = staging / f"{tf}_feats.npy"
            ts_path = staging / f"{tf}_ts.npy"
            _write_npy_fsync(feats_path, feats_np)
            _write_npy_fsync(ts_path, ts_int64)
            manifest["tfs"][tf] = {
                "n_bars": int(feats_np.shape[0]),
                "feature_count": int(feats_np.shape[1]),
                "feats_npy": feats_path.name,
                "feats_npy_sha256": _sha256_file(feats_path),
                "feats_npy_size_bytes": int(feats_path.stat().st_size),
                "ts_npy": ts_path.name,
                "ts_npy_sha256": _sha256_file(ts_path),
                "ts_npy_size_bytes": int(ts_path.stat().st_size),
                "first_ts_ns": int(ts_int64[0]),
                "last_ts_ns": int(ts_int64[-1]),
                "causal_warmup_rows": int(warmup_rows),
            }

        full_input_liveness = build_multi_tf_v4_liveness_contract(
            features
        )
        if full_input_liveness.get("decision") != "PASS":
            raise RuntimeError(
                "HTF_V4_CACHE_FULL_INPUT_LIVENESS_FAIL: "
                f"{full_input_liveness.get('failures')}"
            )
        manifest["full_input_liveness"] = full_input_liveness
        if _sha256_file(source) != expected_source_sha256:
            raise RuntimeError("HTF_V4_CACHE_SOURCE_CHANGED_DURING_BUILD")
        manifest["cache_identity_sha256"] = compute_htf_v4_cache_identity(manifest)
        expected_inventory = {
            "manifest.json",
            *(
                name
                for tf in MULTI_TF_RESAMPLE_RULES
                for name in (f"{tf}_feats.npy", f"{tf}_ts.npy")
            ),
        }
        _write_json_fsync(staging / "manifest.json", manifest)
        if {path.name for path in staging.iterdir()} != expected_inventory or any(
            path.is_symlink() or not path.is_file() for path in staging.iterdir()
        ):
            raise RuntimeError("HTF_V4_CACHE_STAGING_INVENTORY_INVALID")
        _fsync_directory(staging)
        _rename_dir_noreplace(staging, destination)
        published = True
        try:
            _fsync_directory(destination.parent)
        except OSError as exc:
            raise RuntimeError(
                "HTF_V4_CACHE_PUBLISHED_PARENT_FSYNC_FAILED: immutable output "
                f"is visible but not admissible: {destination}: {exc}"
            ) from exc
        return destination / "manifest.json"
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m5-prebuilt",
        type=Path,
        required=True,
        help="Exact native-M5 enriched OHLCV parquet",
    )
    parser.add_argument(
        "--expected-source-sha256",
        required=True,
        help="Predeclared exact lowercase SHA-256 of --m5-prebuilt",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="New immutable V4 cache directory",
    )
    parser.add_argument(
        "--level-tol-quantile-q",
        type=float,
        required=True,
        help=(
            "Explicit recipe input: quantile q for the level-registry "
            "cluster-tolerance TRAIN fit (design doc §8 item 4 declares no "
            "default; recipe key ENTRY_LEVEL_REGISTRY_TOL_QUANTILE_Q)"
        ),
    )
    parser.add_argument(
        "--registry-fit-train-end",
        required=True,
        help=(
            "Declared TRAIN window end (UTC timestamp); the V29 registry "
            "constants are fitted once on source rows at or before this "
            "boundary and frozen in the cache manifest (rule 18)"
        ),
    )
    args = parser.parse_args()

    from gx1.contracts.entry_exit_production_architecture_v1 import (
        PRODUCTION_MTF_PER_TF_WINDOW_BARS,
    )
    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_SEQ_LEN,
    )
    from gx1.features.htf_features import (
        MULTI_TF_FEATURE_COUNT_V4,
        build_multi_tf_per_bar_features_v4,
        fit_v29_registry_constants_from_m5,
    )
    import pyarrow.parquet as pq

    source = args.m5_prebuilt.expanduser()
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise RuntimeError(
            f"[CACHE_V4_SOURCE_CONTRACT] exact absolute non-symlink source required: {source}"
        )
    source = source.resolve(strict=True)
    out_dir = args.out_dir.expanduser()
    if not out_dir.is_absolute():
        raise RuntimeError(
            "[CACHE_V4_OUTPUT_CONTRACT] --out-dir must be absolute"
        )
    expected_source_sha256 = _require_exact_sha256(
        args.expected_source_sha256,
        label="CACHE_V4_EXPECTED_SOURCE_SHA256_INVALID",
    )
    if _sha256_file(source) != expected_source_sha256:
        raise RuntimeError(
            "[CACHE_V4_SOURCE_SHA256_MISMATCH] "
            f"source={source} expected={expected_source_sha256}"
        )

    columns = ["time", "open", "high", "low", "close", "volume"]
    missing = [
        name
        for name in columns
        if name not in pq.ParquetFile(source).schema_arrow.names
    ]
    if missing:
        raise RuntimeError(
            f"[CACHE_V4_SOURCE_CONTRACT] exact native-M5 OHLCV missing: {missing}"
        )
    m5 = pd.read_parquet(source, columns=columns)
    m5["time"] = pd.to_datetime(m5["time"], utc=True, errors="raise")
    m5 = m5.set_index("time")
    for name in ("open", "high", "low", "close", "volume"):
        m5[name] = m5[name].astype(np.float32)

    print(
        f"[CACHE_V4] source={source} rows={len(m5):,} "
        f"fields={MULTI_TF_FEATURE_COUNT_V4} out={out_dir}"
    )
    # TRAIN fit of the V29 registry constants (once, declared window; the
    # per-TF candidate windows are the production per-TF window bars and the
    # entry-M5 lane uses the Entry model sequence length — named contract
    # constants, not new numbers).
    registry_constants = fit_v29_registry_constants_from_m5(
        m5,
        level_tol_quantile_q=args.level_tol_quantile_q,
        declared_train_window_end=args.registry_fit_train_end,
        per_tf_seq_lens=dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS),
        entry_m5_seq_len=MODEL_NATIVE_SEQ_LEN,
    )
    print(
        "[CACHE_V4] v29 registry constants fitted: "
        f"level_tol_atr={registry_constants['level_tol_atr']} "
        f"trendline_band_atr={registry_constants['trendline_band_atr']} "
        f"entry_m5={registry_constants['entry_m5']}"
    )
    features = build_multi_tf_per_bar_features_v4(
        m5,
        v29_registry_constants=registry_constants,
    )
    manifest_path = publish_multi_tf_v4_cache(
        out_dir=out_dir,
        m5_prebuilt=source,
        expected_source_sha256=expected_source_sha256,
        features=features,
        v29_registry_constants=registry_constants,
    )
    for timeframe, frame in features.items():
        print(
            f"[CACHE_V4] {timeframe}: "
            f"{frame.attrs['feats_np'].shape}"
        )
    print(f"[CACHE_V4] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
