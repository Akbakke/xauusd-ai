#!/usr/bin/env python3
"""Build and atomically publish the sole immutable V4 multi-timeframe cache.

The source must be one exact absolute native-M5 enriched parquet. The fixed
M5/M15/H1/H4/D1 cache contains ``MULTI_TF_FEATURE_COUNT_V4`` ordered float32
fields per timeframe (derived from the declared name tuples), exact int64
timestamps, byte hashes, full-input liveness evidence, the immutable
TRAIN-fitted V29 registry constants, and one canonical identity. There is no
contract switch and no historical publisher.

The V29 registry constants (level cluster tolerance / trendline band and
lifecycle expiry per TF) are fitted here once on an immutable chronological
inner/outer TRAIN split and frozen with full provenance in the cache manifest.

Usage:
    python -m gx1.scripts.prebuild_multi_tf_cache_v4 \
        --m5-prebuilt /absolute/xauusd_m5_enriched.parquet \
        --expected-source-sha256 <exact-lowercase-sha256> \
        --registry-fit-train-start <UTC timestamp> \
        --registry-fit-inner-end <UTC timestamp> \
        --registry-fit-train-end <UTC timestamp of the declared TRAIN end> \
        --registry-fit-tape-manifest /absolute/TAPE_MANIFEST.json \
        --expected-registry-fit-tape-manifest-sha256 <exact-lowercase-sha256> \
        --registry-fit-pair-manifest /absolute/PAIR_MANIFEST.json \
        --expected-registry-fit-pair-manifest-sha256 <exact-lowercase-sha256> \
        --registry-fit-train-split-id <declared TRAIN split id> \
        --volatility-squeeze-manifest /absolute/SQUEEZE_MANIFEST.json \
        --expected-volatility-squeeze-manifest-sha256 <exact-lowercase-sha256> \
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
    volatility_squeeze_artifacts,
) -> Path:
    from gx1.features.htf_features import (
        HTF_V4_CACHE_BUILDER_VERSION,
        HTF_V4_CACHE_SCHEMA_VERSION,
        HTF_V4_MATRIX_CONTRACT,
        MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4,
        MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4,
        MULTI_TF_FEATURE_COUNT_V4,
        MULTI_TF_PER_BAR_FEATURES_V4,
        MULTI_TF_RESAMPLE_ORIGIN_CONTRACT,
        MULTI_TF_RESAMPLE_RULES,
        MULTI_TF_SHIFT,
        SMC_CAUSAL_REPLAY_SCHEMA_VERSION,
        build_multi_tf_v4_closed_timestamp_indices,
        build_multi_tf_v4_liveness_contract,
        compute_htf_v4_cache_identity,
        require_multi_tf_v4_frames,
        require_model_native_mtf_scalar_owner_v4,
        require_v29_registry_constants,
        validate_causal_feature_matrix,
    )
    from gx1.features.technical_indicators_v1 import (
        technical_indicator_contract_metadata,
    )
    from gx1.features.swing_structure_v1 import (
        swing_structure_contract_metadata,
    )
    from gx1.features.volatility_squeeze_state_v1 import (
        require_volatility_squeeze_artifact_set,
    )

    v29_registry_constants = require_v29_registry_constants(
        v29_registry_constants
    )
    volatility_squeeze_artifacts = require_volatility_squeeze_artifact_set(
        volatility_squeeze_artifacts
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
    require_model_native_mtf_scalar_owner_v4(features)
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
            # V30 package 3 (2026-08-13): the bin ORIGIN travels with the
            # cadence.  Taken verbatim from the one owner
            # (htf_features.MULTI_TF_RESAMPLE_ORIGIN_CONTRACT) so the published
            # cache records which daily clock produced its D1 bars.
            "resample_origin_contract": dict(MULTI_TF_RESAMPLE_ORIGIN_CONTRACT),
            "builder_version": BUILDER_VERSION,
            "smc_causal_replay_schema_version": (
                SMC_CAUSAL_REPLAY_SCHEMA_VERSION
            ),
            "technical_indicator_owner": technical_indicator_contract_metadata(),
            "swing_structure_owner": swing_structure_contract_metadata(),
            "m5_prebuilt_source": str(source),
            "m5_prebuilt_source_sha256": expected_source_sha256,
            "v29_registry_constants": v29_registry_constants,
            "volatility_squeeze_artifact_set": (
                volatility_squeeze_artifacts.binding()
            ),
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
            scalar_fields = MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4[tf]
            scalar_np = np.asarray(
                frame.attrs.get("model_native_mtf_scalars_np_v4")
            )
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
                or frame.attrs.get("model_native_mtf_scalar_fields_v4")
                != scalar_fields
                or scalar_np.dtype != np.dtype(np.float32)
                or scalar_np.shape != (len(feats_np), len(scalar_fields))
                or frame.attrs.get("model_native_mtf_scalar_contract_v4")
                != MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4
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
            scalar_warmup_rows = validate_causal_feature_matrix(
                scalar_np,
                expected_width=len(scalar_fields),
                context=f"HTF_V4_CACHE_PUBLISH_MODEL_NATIVE_SCALARS_{tf}",
            )
            if (
                frame.attrs.get("model_native_mtf_scalar_warmup_rows_v4")
                != scalar_warmup_rows
            ):
                raise RuntimeError(
                    f"HTF_V4_CACHE_FEATURE_SET_INVALID: {tf} scalar attrs mismatch"
                )

            feats_path = staging / f"{tf}_feats.npy"
            ts_path = staging / f"{tf}_ts.npy"
            scalar_path = staging / f"{tf}_model_native_scalars.npy"
            _write_npy_fsync(feats_path, feats_np)
            _write_npy_fsync(ts_path, ts_int64)
            _write_npy_fsync(scalar_path, scalar_np)
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
                "model_native_scalar_fields": list(scalar_fields),
                "model_native_scalar_count": int(len(scalar_fields)),
                "model_native_scalars_npy": scalar_path.name,
                "model_native_scalars_npy_sha256": _sha256_file(scalar_path),
                "model_native_scalars_npy_size_bytes": int(
                    scalar_path.stat().st_size
                ),
                "model_native_scalar_warmup_rows": int(scalar_warmup_rows),
                "model_native_scalar_contract": (
                    MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4
                ),
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
                for name in (
                    f"{tf}_feats.npy",
                    f"{tf}_ts.npy",
                    f"{tf}_model_native_scalars.npy",
                )
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
    parser.add_argument("--registry-fit-train-start", required=True)
    parser.add_argument("--registry-fit-inner-end", required=True)
    parser.add_argument(
        "--registry-fit-train-end",
        required=True,
        help=(
            "Declared TRAIN window end (UTC timestamp); the V29 registry "
            "constants are fitted once on source rows at or before this "
            "boundary and frozen in the cache manifest (rule 18)"
        ),
    )
    parser.add_argument(
        "--volatility-squeeze-manifest",
        type=Path,
        required=True,
        help="Exact immutable six-clock volatility-squeeze manifest",
    )
    parser.add_argument("--registry-fit-tape-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-registry-fit-tape-manifest-sha256", required=True
    )
    # A --registry-fit-split-manifest was required here until 2026-08-15. It
    # pointed at an artifact the rebuild chain only produces AFTER this cache
    # exists, so it could never be satisfied on a first build. The declared
    # TRAIN window below is the population authority instead, and the pair
    # manifest below restores the hash-bound pointer to the generation the fit
    # actually read (the split pointer was bound to the pair manifest under a
    # false name; removing it left no immutable-file binding at all).
    parser.add_argument("--registry-fit-pair-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-registry-fit-pair-manifest-sha256", required=True
    )
    parser.add_argument("--registry-fit-train-split-id", required=True)
    parser.add_argument(
        "--expected-volatility-squeeze-manifest-sha256",
        required=True,
        help="Predeclared exact file SHA-256 of the squeeze manifest",
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
    from gx1.features.volatility_squeeze_state_v1 import (
        load_volatility_squeeze_artifact_manifest,
    )

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

    def _bound_manifest(raw: Path, expected: str, *, label: str) -> tuple[Path, str]:
        path = raw.expanduser()
        digest = _require_exact_sha256(expected, label=f"{label}_SHA256_INVALID")
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
        ):
            raise RuntimeError(f"[{label}_INVALID] exact immutable file required")
        path = path.resolve(strict=True)
        if _sha256_file(path) != digest:
            raise RuntimeError(f"[{label}_HASH_MISMATCH]")
        return path, digest

    tape_manifest, tape_manifest_sha256 = _bound_manifest(
        args.registry_fit_tape_manifest,
        args.expected_registry_fit_tape_manifest_sha256,
        label="CACHE_V4_REGISTRY_TAPE_MANIFEST",
    )
    pair_manifest, pair_manifest_sha256 = _bound_manifest(
        args.registry_fit_pair_manifest,
        args.expected_registry_fit_pair_manifest_sha256,
        label="CACHE_V4_REGISTRY_PAIR_MANIFEST",
    )
    train_start = pd.Timestamp(args.registry_fit_train_start)
    inner_end = pd.Timestamp(args.registry_fit_inner_end)
    train_end = pd.Timestamp(args.registry_fit_train_end)
    if any(
        stamp.tzinfo is None or stamp.utcoffset() != pd.Timedelta(0)
        for stamp in (train_start, inner_end, train_end)
    ) or not train_start < inner_end < train_end:
        raise RuntimeError("[CACHE_V4_REGISTRY_TRAIN_SPLIT_INVALID]")
    source_provenance_by_clock = {
        clock: {
            "source_artifact": str(source),
            "source_sha256": expected_source_sha256,
            "source_schema_version": "native_m5_enriched_ohlcv_v1",
            "source_lane": clock,
            "tape_manifest_artifact": str(tape_manifest),
            "tape_manifest_sha256": tape_manifest_sha256,
            "pair_manifest_artifact": str(pair_manifest),
            "pair_manifest_sha256": pair_manifest_sha256,
            "train_split_id": args.registry_fit_train_split_id,
            "declared_train_window_start": train_start.isoformat(),
            "declared_train_window_end": train_end.isoformat(),
        }
        for clock in ("M5", "M15", "H1", "H4", "D1")
    }

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
        declared_train_window_start=train_start.isoformat(),
        declared_train_window_end=train_end.isoformat(),
        declared_inner_fit_window_end=inner_end.isoformat(),
        source_provenance_by_clock=source_provenance_by_clock,
        per_tf_seq_lens=dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS),
        entry_m5_seq_len=MODEL_NATIVE_SEQ_LEN,
    )
    squeeze_artifacts = load_volatility_squeeze_artifact_manifest(
        args.volatility_squeeze_manifest.expanduser().resolve(strict=True),
        expected_sha256=_require_exact_sha256(
            args.expected_volatility_squeeze_manifest_sha256,
            label="CACHE_V4_VOLATILITY_SQUEEZE_MANIFEST_SHA256_INVALID",
        ),
    )
    print(
        "[CACHE_V4] v29 registry constants fitted: "
        "level_recurrence_threshold_atr="
        f"{registry_constants['level_recurrence_threshold_atr']} "
        f"trendline_band_atr={registry_constants['trendline_band_atr']} "
        f"entry_m5={registry_constants['entry_m5']}"
    )
    features = build_multi_tf_per_bar_features_v4(
        m5,
        v29_registry_constants=registry_constants,
        volatility_squeeze_artifacts=squeeze_artifacts,
    )
    manifest_path = publish_multi_tf_v4_cache(
        out_dir=out_dir,
        m5_prebuilt=source,
        expected_source_sha256=expected_source_sha256,
        features=features,
        v29_registry_constants=registry_constants,
        volatility_squeeze_artifacts=squeeze_artifacts,
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
