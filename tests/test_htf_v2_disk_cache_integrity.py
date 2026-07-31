from __future__ import annotations

import hashlib
import io
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from gx1.features import htf_features as htf
from gx1.scripts import prebuild_multi_tf_cache_v2 as producer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _feature_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    width = htf.MULTI_TF_FEATURE_COUNT_V2
    for tf_offset, tf in enumerate(htf.MULTI_TF_RESAMPLE_RULES):
        values = (
            np.arange(4 * width, dtype=np.float32).reshape(4, width)
            + np.float32(tf_offset)
        )
        timestamps = np.asarray(
            [1_780_000_000_000_000_000 + tf_offset * 1_000_000 + row * 300_000_000_000
             for row in range(4)],
            dtype=np.int64,
        )
        frame = pd.DataFrame(
            values,
            index=pd.to_datetime(timestamps, unit="ns", utc=True),
            columns=htf.MULTI_TF_PER_BAR_FEATURES_V2,
        )
        frame.attrs["feats_np"] = values
        frame.attrs["ts_int64"] = timestamps
        frame.attrs["causal_warmup_rows"] = 0
        frame.attrs["htf_feature_contract"] = htf.HTF_V2_MATRIX_CONTRACT
        frames[tf] = frame
    return frames


def _feature_frames_v4(
    source_index: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    width = htf.MULTI_TF_FEATURE_COUNT_V4
    expected_indices = htf.build_multi_tf_v4_closed_timestamp_indices(
        source_index
    )
    for tf_offset, tf in enumerate(htf.MULTI_TF_RESAMPLE_RULES):
        index = expected_indices[tf]
        row = np.arange(len(index), dtype=np.float32).reshape(-1, 1)
        column = np.arange(width, dtype=np.float32).reshape(1, -1)
        values = row * (column + 1.0) + np.float32(tf_offset)
        timestamps = index.asi8.astype(np.int64, copy=True)
        frame = pd.DataFrame(
            values,
            index=index,
            columns=htf.MULTI_TF_PER_BAR_FEATURES_V4,
        )
        frame.attrs["feats_np"] = values
        frame.attrs["ts_int64"] = timestamps
        frame.attrs["causal_warmup_rows"] = 0
        frame.attrs["htf_feature_contract"] = htf.HTF_V4_MATRIX_CONTRACT
        frames[tf] = frame
    return frames


def _publish(tmp_path: Path) -> Path:
    source = tmp_path / "xauusd_m5_canonical.parquet"
    source.write_bytes(b"immutable XAUUSD source fixture")
    source = source.resolve(strict=True)
    cache_dir = tmp_path / "MULTI_TF_V2_CACHE_TEST"
    manifest_path = producer.publish_multi_tf_v2_cache(
        out_dir=cache_dir,
        m5_prebuilt=source,
        expected_source_sha256=_sha256(source),
        features=_feature_frames(),
        contract="v2",
    )
    assert manifest_path == cache_dir / "manifest.json"
    return cache_dir


def _publish_v4(tmp_path: Path) -> Path:
    source = tmp_path / "xauusd_m5_canonical_v4.parquet"
    source_index = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=8 * 288,
        freq="5min",
    )
    pd.DataFrame({"time": source_index}).to_parquet(source, index=False)
    source = source.resolve(strict=True)
    cache_dir = tmp_path / "MULTI_TF_V4_CACHE_TEST"
    producer.publish_multi_tf_v2_cache(
        out_dir=cache_dir,
        m5_prebuilt=source,
        expected_source_sha256=_sha256(source),
        features=_feature_frames_v4(source_index),
        contract="v4",
    )
    return cache_dir


def _manifest(cache_dir: Path) -> dict:
    return json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))


def test_cache_publisher_requires_explicit_contract() -> None:
    parameter = inspect.signature(producer.publish_multi_tf_v2_cache).parameters[
        "contract"
    ]
    assert parameter.default is inspect.Parameter.empty


def _all_array_names() -> tuple[str, ...]:
    return tuple(
        name
        for tf in htf.MULTI_TF_RESAMPLE_RULES
        for name in (f"{tf}_feats.npy", f"{tf}_ts.npy")
    )


def test_publisher_and_loader_bind_exact_inventory_hashes_and_cache_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = _publish(tmp_path)
    manifest = _manifest(cache_dir)
    expected_inventory = {"manifest.json", *_all_array_names()}

    assert set(path.name for path in cache_dir.iterdir()) == expected_inventory
    assert manifest["schema_version"] == htf.HTF_V2_CACHE_SCHEMA_VERSION
    assert manifest["builder_version"] == htf.HTF_V2_CACHE_BUILDER_VERSION
    assert manifest["cache_identity_sha256"] == htf.compute_htf_v2_cache_identity(
        manifest
    )
    for tf, info in manifest["tfs"].items():
        for kind in ("feats", "ts"):
            path = cache_dir / info[f"{kind}_npy"]
            assert info[f"{kind}_npy"] == f"{tf}_{kind}.npy"
            assert info[f"{kind}_npy_sha256"] == _sha256(path)
            assert info[f"{kind}_npy_size_bytes"] == path.stat().st_size

    real_np_load = np.load
    loaded_from_verified_bytes = 0

    def guarded_np_load(file, *args, **kwargs):
        nonlocal loaded_from_verified_bytes
        assert isinstance(file, io.BytesIO)
        loaded_from_verified_bytes += 1
        return real_np_load(file, *args, **kwargs)

    monkeypatch.setattr(htf.np, "load", guarded_np_load)
    loaded = htf.load_multi_tf_v2_cache(cache_dir)

    assert isinstance(loaded, dict)
    assert tuple(loaded) == tuple(htf.MULTI_TF_RESAMPLE_RULES)
    assert loaded.cache_identity_sha256 == manifest["cache_identity_sha256"]
    assert loaded_from_verified_bytes == len(_all_array_names())
    for tf in htf.MULTI_TF_RESAMPLE_RULES:
        np.testing.assert_array_equal(
            loaded[tf].attrs["feats_np"],
            _feature_frames()[tf].attrs["feats_np"],
        )
        np.testing.assert_array_equal(
            loaded[tf].attrs["ts_int64"],
            _feature_frames()[tf].attrs["ts_int64"],
        )


@pytest.mark.parametrize("array_name", _all_array_names())
def test_loader_rejects_same_size_byte_tamper_for_every_declared_array(
    tmp_path: Path,
    array_name: str,
) -> None:
    cache_dir = _publish(tmp_path)
    target = cache_dir / array_name
    payload = bytearray(target.read_bytes())
    payload[-1] ^= 0x01
    target.write_bytes(payload)

    with pytest.raises(RuntimeError, match="HTF_V2_CACHE_SHA256_MISMATCH"):
        htf.load_multi_tf_v2_cache(cache_dir)


@pytest.mark.parametrize("array_name", ("M5_feats.npy", "H4_ts.npy"))
def test_loader_rejects_missing_declared_array(
    tmp_path: Path,
    array_name: str,
) -> None:
    cache_dir = _publish(tmp_path)
    (cache_dir / array_name).unlink()

    with pytest.raises(RuntimeError, match="HTF_V2_CACHE_INVENTORY_MISMATCH"):
        htf.load_multi_tf_v2_cache(cache_dir)


def test_loader_rejects_unexpected_inventory_entry(tmp_path: Path) -> None:
    cache_dir = _publish(tmp_path)
    (cache_dir / "unbound.npy").write_bytes(b"extra")

    with pytest.raises(RuntimeError, match="HTF_V2_CACHE_INVENTORY_MISMATCH"):
        htf.load_multi_tf_v2_cache(cache_dir)


@pytest.mark.parametrize(
    "entry_name",
    ("manifest.json", "M15_feats.npy", "H4_ts.npy"),
)
def test_loader_rejects_symlinked_manifest_or_array(
    tmp_path: Path,
    entry_name: str,
) -> None:
    cache_dir = _publish(tmp_path)
    entry = cache_dir / entry_name
    outside = tmp_path / f"outside_{entry_name}"
    entry.replace(outside)
    entry.symlink_to(outside)

    with pytest.raises(RuntimeError, match="HTF_V2_CACHE_FILE_INVALID"):
        htf.load_multi_tf_v2_cache(cache_dir)


def test_loader_rejects_cache_directory_symlink(tmp_path: Path) -> None:
    cache_dir = _publish(tmp_path)
    alias = tmp_path / "cache_alias"
    alias.symlink_to(cache_dir, target_is_directory=True)

    with pytest.raises(RuntimeError, match="HTF_V2_CACHE_PATH_INVALID"):
        htf.load_multi_tf_v2_cache(alias)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("schema_version", "htf_v2_disk_cache_manifest_v1"),
        ("builder_version", "prebuild_multi_tf_cache_v2_old"),
    ),
)
def test_loader_rejects_old_schema_or_builder(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    cache_dir = _publish(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = _manifest(cache_dir)
    manifest[field] = replacement
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="HTF_V2_CACHE_CONTRACT_MISMATCH"):
        htf.load_multi_tf_v2_cache(cache_dir)


def test_loader_rejects_manifest_identity_tamper(tmp_path: Path) -> None:
    cache_dir = _publish(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = _manifest(cache_dir)
    manifest["cache_identity_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="HTF_V2_CACHE_IDENTITY_MISMATCH"):
        htf.load_multi_tf_v2_cache(cache_dir)


def test_loader_rejects_duplicate_manifest_keys(tmp_path: Path) -> None:
    cache_dir = _publish(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    payload = manifest_path.read_text(encoding="utf-8")
    payload = payload.replace(
        '"schema_version":',
        '"schema_version": "duplicate", "schema_version":',
        1,
    )
    manifest_path.write_text(payload, encoding="utf-8")

    with pytest.raises(RuntimeError, match="HTF_V2_CACHE_MANIFEST_INVALID"):
        htf.load_multi_tf_v2_cache(cache_dir)


def test_publisher_never_replaces_an_existing_cache(tmp_path: Path) -> None:
    cache_dir = _publish(tmp_path)
    manifest_sha_before = _sha256(cache_dir / "manifest.json")
    source = (tmp_path / "xauusd_m5_canonical.parquet").resolve(strict=True)

    with pytest.raises(RuntimeError, match="HTF_V2_CACHE_OUTPUT_EXISTS"):
        producer.publish_multi_tf_v2_cache(
            out_dir=cache_dir,
            m5_prebuilt=source,
            expected_source_sha256=_sha256(source),
            features=_feature_frames(),
            contract="v2",
        )

    assert _sha256(cache_dir / "manifest.json") == manifest_sha_before


def test_cli_rejects_wrong_predeclared_source_hash_before_parquet_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (tmp_path / "not-even-parquet.bin").resolve()
    source.write_bytes(b"exact source bytes")
    monkeypatch.setattr(
        pq,
        "ParquetFile",
        lambda *args, **kwargs: pytest.fail("parquet read started before hash gate"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prebuild_multi_tf_cache_v2",
            "--contract",
            "v4",
            "--m5-prebuilt",
            str(source),
            "--expected-source-sha256",
            "0" * 64,
            "--out-dir",
            str((tmp_path / "cache").resolve()),
        ],
    )

    with pytest.raises(RuntimeError, match="CACHE_V2_SOURCE_SHA256_MISMATCH"):
        producer.main()


@pytest.mark.parametrize(
    "invalid_sha256",
    ("", "abc", "A" * 64, "g" * 64, "0" * 63, "0" * 65),
)
def test_exact_source_hash_contract_rejects_noncanonical_values(
    invalid_sha256: str,
) -> None:
    with pytest.raises(RuntimeError, match="exact lowercase SHA-256 required"):
        producer._require_exact_sha256(invalid_sha256, label="TEST")


def test_v4_publisher_rejects_timestamp_arrays_not_derived_from_source(
    tmp_path: Path,
) -> None:
    source_index = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=8 * 288,
        freq="5min",
    )
    source = tmp_path / "xauusd_m5_canonical_v4.parquet"
    pd.DataFrame({"time": source_index}).to_parquet(source, index=False)
    frames = _feature_frames_v4(source_index)
    shifted = frames["H4"].copy()
    shifted.index = shifted.index + pd.Timedelta(minutes=5)
    shifted.attrs = dict(frames["H4"].attrs)
    shifted.attrs["ts_int64"] = shifted.index.asi8.astype(np.int64, copy=True)
    frames["H4"] = shifted

    with pytest.raises(
        RuntimeError,
        match="HTF_V4_CACHE_SOURCE_TIMESTAMP_GEOMETRY_MISMATCH: H4",
    ):
        producer.publish_multi_tf_v2_cache(
            out_dir=(tmp_path / "cache").resolve(),
            m5_prebuilt=source.resolve(),
            expected_source_sha256=_sha256(source),
            features=frames,
            contract="v4",
        )


def test_parent_fsync_failure_reports_visible_but_inadmissible_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "xauusd_m5_canonical.parquet"
    source.write_bytes(b"immutable XAUUSD source fixture")
    source = source.resolve(strict=True)
    destination = (tmp_path / "cache").resolve()
    real_fsync_directory = producer._fsync_directory
    calls = 0

    def fail_after_publish(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected parent fsync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(producer, "_fsync_directory", fail_after_publish)

    with pytest.raises(
        RuntimeError,
        match="HTF_V2_CACHE_PUBLISHED_PARENT_FSYNC_FAILED",
    ):
        producer.publish_multi_tf_v2_cache(
            out_dir=destination,
            m5_prebuilt=source,
            expected_source_sha256=_sha256(source),
            features=_feature_frames(),
            contract="v2",
        )

    assert destination.is_dir()
    assert (destination / "manifest.json").is_file()


def test_v4_cache_binds_and_recomputes_every_field_liveness(
    tmp_path: Path,
) -> None:
    cache_dir = _publish_v4(tmp_path)
    manifest = _manifest(cache_dir)

    assert manifest["full_input_liveness"]["decision"] == "PASS"
    assert (
        manifest["full_input_liveness"]["schema_version"]
        == htf.HTF_V4_FULL_INPUT_LIVENESS_SCHEMA_VERSION
    )
    assert set(manifest["full_input_liveness"]["timeframes"]) == set(
        htf.MULTI_TF_RESAMPLE_RULES
    )
    loaded = htf.load_multi_tf_v2_cache(cache_dir)
    assert tuple(loaded) == tuple(htf.MULTI_TF_RESAMPLE_RULES)


def test_v4_cache_rejects_hash_valid_but_false_liveness_claim(
    tmp_path: Path,
) -> None:
    cache_dir = _publish_v4(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = _manifest(cache_dir)
    manifest["full_input_liveness"]["timeframes"]["H4"]["fields"][
        htf.MULTI_TF_PER_BAR_FEATURES_V4[0]
    ]["std"] = 999.0
    manifest["cache_identity_sha256"] = htf.compute_htf_v2_cache_identity(
        manifest
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="HTF_V4_CACHE_FULL_INPUT_LIVENESS_INVALID"):
        htf.load_multi_tf_v2_cache(cache_dir)
