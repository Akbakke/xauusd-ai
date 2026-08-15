"""V4-only immutable multi-timeframe cache integrity tests."""
from __future__ import annotations

import hashlib
import inspect
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from gx1.features import htf_features as htf
from gx1.scripts import prebuild_multi_tf_cache_v4 as producer
from tests.htf_v29_registry_test_support import (
    synthetic_v29_registry_constants,
)
from tests.volatility_squeeze_test_support import (
    make_volatility_squeeze_artifact_set,
)

_V29_TEST_REGISTRY_CONSTANTS = synthetic_v29_registry_constants()
_SQUEEZE_TEST_ARTIFACTS = None


@pytest.fixture(scope="module", autouse=True)
def _bind_squeeze_artifacts(tmp_path_factory):
    global _SQUEEZE_TEST_ARTIFACTS
    _SQUEEZE_TEST_ARTIFACTS = make_volatility_squeeze_artifact_set(
        tmp_path_factory.mktemp("squeeze-artifacts")
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_array_names() -> tuple[str, ...]:
    return tuple(
        name
        for timeframe in htf.MULTI_TF_RESAMPLE_RULES
        for name in (
            f"{timeframe}_feats.npy",
            f"{timeframe}_ts.npy",
        )
    )


def _source_and_frames(
    tmp_path: Path,
) -> tuple[Path, pd.DatetimeIndex, dict[str, pd.DataFrame]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_index = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=8 * 288,
        freq="5min",
    )
    source = tmp_path / "xauusd_m5_enriched.parquet"
    pd.DataFrame({"time": source_index}).to_parquet(source, index=False)
    source = source.resolve(strict=True)

    expected_indices = htf.build_multi_tf_v4_closed_timestamp_indices(
        source_index
    )
    frames: dict[str, pd.DataFrame] = {}
    width = htf.MULTI_TF_FEATURE_COUNT_V4
    for offset, timeframe in enumerate(htf.MULTI_TF_RESAMPLE_RULES):
        index = expected_indices[timeframe]
        rows = np.arange(len(index), dtype=np.float32).reshape(-1, 1)
        columns = np.arange(width, dtype=np.float32).reshape(1, -1)
        values = rows * (columns + 1.0) + np.float32(offset)
        frame = pd.DataFrame(
            values,
            index=index,
            columns=htf.MULTI_TF_PER_BAR_FEATURES_V4,
            copy=False,
        )
        frame_values = frame.to_numpy(dtype=np.float32, copy=False)
        frame.attrs["feats_np"] = frame_values
        frame.attrs["ts_int64"] = index.asi8.astype(np.int64, copy=True)
        frame.attrs["causal_warmup_rows"] = 0
        frame.attrs["htf_feature_contract"] = htf.HTF_V4_MATRIX_CONTRACT
        frames[timeframe] = frame
    return source, source_index, frames


def _publish(tmp_path: Path) -> Path:
    source, _source_index, frames = _source_and_frames(tmp_path)
    cache_dir = tmp_path / "MULTI_TF_V4_CACHE_TEST"
    manifest_path = producer.publish_multi_tf_v4_cache(
        out_dir=cache_dir,
        m5_prebuilt=source,
        expected_source_sha256=_sha256(source),
        features=frames,
        v29_registry_constants=_V29_TEST_REGISTRY_CONSTANTS,
        volatility_squeeze_artifacts=_SQUEEZE_TEST_ARTIFACTS,
    )
    assert manifest_path == cache_dir / "manifest.json"
    return cache_dir


def _manifest(cache_dir: Path) -> dict:
    return json.loads(
        (cache_dir / "manifest.json").read_text(encoding="utf-8")
    )


def test_v4_publisher_has_no_contract_switch() -> None:
    parameters = inspect.signature(
        producer.publish_multi_tf_v4_cache
    ).parameters
    assert "contract" not in parameters


def test_publisher_and_loader_bind_exact_v4_inventory_and_verified_matrix_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _index, expected_frames = _source_and_frames(tmp_path)
    cache_dir = tmp_path / "cache"
    producer.publish_multi_tf_v4_cache(
        out_dir=cache_dir,
        m5_prebuilt=source,
        expected_source_sha256=_sha256(source),
        features=expected_frames,
        v29_registry_constants=_V29_TEST_REGISTRY_CONSTANTS,
        volatility_squeeze_artifacts=_SQUEEZE_TEST_ARTIFACTS,
    )
    manifest = _manifest(cache_dir)
    assert {path.name for path in cache_dir.iterdir()} == {
        "manifest.json",
        *_all_array_names(),
    }
    assert manifest["schema_version"] == htf.HTF_V4_CACHE_SCHEMA_VERSION
    assert manifest["builder_version"] == htf.HTF_V4_CACHE_BUILDER_VERSION
    assert manifest["smc_causal_replay_schema_version"] == (
        htf.SMC_CAUSAL_REPLAY_SCHEMA_VERSION
    )
    from gx1.features.technical_indicators_v1 import (
        technical_indicator_contract_metadata,
    )

    assert manifest["technical_indicator_owner"] == (
        technical_indicator_contract_metadata()
    )
    from gx1.features.swing_structure_v1 import (
        swing_structure_contract_metadata,
    )

    assert manifest["swing_structure_owner"] == (
        swing_structure_contract_metadata()
    )
    assert manifest["feature_count"] == htf.MULTI_TF_FEATURE_COUNT_V4
    assert manifest["feature_names"] == list(
        htf.MULTI_TF_PER_BAR_FEATURES_V4
    )
    # v8 binds the additive volume surface and the shared 22-anchored H4/D1
    # clock in the same immutable identity; neither repair can mask the other.
    assert set(htf.MULTI_TF_V4_VOLUME_FEATURES).issubset(
        manifest["feature_names"]
    )
    assert manifest["resample_origin_contract"]["H4"] == str(
        pd.Timedelta(hours=2)
    )
    assert manifest["resample_origin_contract"]["D1"] == str(
        pd.Timedelta(hours=22)
    )
    assert manifest["cache_identity_sha256"] == (
        htf.compute_htf_v4_cache_identity(manifest)
    )

    real_load = np.load
    verified_byte_loads = 0

    def guarded_load(file, *args, **kwargs):
        nonlocal verified_byte_loads
        assert isinstance(file, io.BytesIO)
        verified_byte_loads += 1
        return real_load(file, *args, **kwargs)

    monkeypatch.setattr(htf.np, "load", guarded_load)
    loaded = htf.load_multi_tf_v4_cache(cache_dir)
    assert isinstance(loaded, htf.MultiTFV4DiskCache)
    assert verified_byte_loads == len(_all_array_names())
    assert loaded.cache_identity_sha256 == manifest["cache_identity_sha256"]
    assert loaded.m5_prebuilt_source == str(source)
    assert loaded.m5_prebuilt_source_sha256 == _sha256(source)

    for timeframe in htf.MULTI_TF_RESAMPLE_RULES:
        np.testing.assert_array_equal(
            loaded[timeframe].index.asi8,
            expected_frames[timeframe].index.asi8,
        )
        frame_values = loaded[timeframe].to_numpy(
            dtype=np.float32,
            copy=False,
        )
        verified_values = loaded[timeframe].attrs["feats_np"]
        assert np.shares_memory(frame_values, verified_values)
        np.testing.assert_array_equal(frame_values, verified_values)
        np.testing.assert_array_equal(
            frame_values,
            expected_frames[timeframe].attrs["feats_np"],
        )


def test_registry_resolver_accepts_only_the_authoritative_cache_manifest(
    tmp_path: Path,
) -> None:
    cache_dir = _publish(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    assert htf.load_v29_registry_constants_manifest(manifest_path) == (
        _V29_TEST_REGISTRY_CONSTANTS
    )

    bare_payload = tmp_path / "bare-production-params.json"
    bare_payload.write_text(
        json.dumps(_V29_TEST_REGISTRY_CONSTANTS), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="CONTAINER_REQUIRED"):
        htf.load_v29_registry_constants_manifest(bare_payload)


def test_registry_resolver_rejects_cache_source_bytes_mismatch(
    tmp_path: Path,
) -> None:
    cache_dir = _publish(tmp_path)
    manifest = _manifest(cache_dir)
    source = Path(manifest["m5_prebuilt_source"])
    source.write_bytes(b"changed-after-cache-publication")
    with pytest.raises(RuntimeError, match="CACHE_SOURCE_IDENTITY_MISMATCH"):
        htf.load_v29_registry_constants_manifest(cache_dir / "manifest.json")


def test_loader_rejects_bound_squeeze_params_mutation(
    tmp_path: Path,
) -> None:
    artifacts = make_volatility_squeeze_artifact_set(
        tmp_path / "squeeze"
    )
    source, _source_index, frames = _source_and_frames(tmp_path / "source")
    cache_dir = tmp_path / "cache"
    producer.publish_multi_tf_v4_cache(
        out_dir=cache_dir,
        m5_prebuilt=source,
        expected_source_sha256=_sha256(source),
        features=frames,
        v29_registry_constants=_V29_TEST_REGISTRY_CONSTANTS,
        volatility_squeeze_artifacts=artifacts,
    )
    params_path = artifacts.manifest_path.parent / "m5_params.json"
    mutated = bytearray(params_path.read_bytes())
    mutated[-2] ^= 0x01
    params_path.write_bytes(mutated)

    with pytest.raises(
        RuntimeError,
        match="HTF_V4_CACHE_VOLATILITY_SQUEEZE_ARTIFACT_SET_INVALID",
    ):
        htf.load_multi_tf_v4_cache(cache_dir)


@pytest.mark.parametrize("array_name", _all_array_names())
def test_loader_rejects_same_size_byte_tamper(
    tmp_path: Path,
    array_name: str,
) -> None:
    cache_dir = _publish(tmp_path)
    target = cache_dir / array_name
    payload = bytearray(target.read_bytes())
    payload[-1] ^= 0x01
    target.write_bytes(payload)

    with pytest.raises(RuntimeError, match="HTF_V4_CACHE_SHA256_MISMATCH"):
        htf.load_multi_tf_v4_cache(cache_dir)


@pytest.mark.parametrize("array_name", ("M5_feats.npy", "H4_ts.npy"))
def test_loader_rejects_missing_declared_array(
    tmp_path: Path,
    array_name: str,
) -> None:
    cache_dir = _publish(tmp_path)
    (cache_dir / array_name).unlink()
    with pytest.raises(RuntimeError, match="HTF_V4_CACHE_INVENTORY_MISMATCH"):
        htf.load_multi_tf_v4_cache(cache_dir)


def test_loader_rejects_unexpected_inventory_entry(tmp_path: Path) -> None:
    cache_dir = _publish(tmp_path)
    (cache_dir / "unbound.npy").write_bytes(b"extra")
    with pytest.raises(RuntimeError, match="HTF_V4_CACHE_INVENTORY_MISMATCH"):
        htf.load_multi_tf_v4_cache(cache_dir)


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
    with pytest.raises(RuntimeError, match="HTF_V4_CACHE_FILE_INVALID"):
        htf.load_multi_tf_v4_cache(cache_dir)


def test_loader_rejects_cache_directory_symlink(tmp_path: Path) -> None:
    cache_dir = _publish(tmp_path)
    alias = tmp_path / "cache_alias"
    alias.symlink_to(cache_dir, target_is_directory=True)
    with pytest.raises(RuntimeError, match="HTF_V4_CACHE_PATH_INVALID"):
        htf.load_multi_tf_v4_cache(alias)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("schema_version", "htf_v4_disk_cache_manifest_v3"),
        ("builder_version", "prebuild_multi_tf_cache_v4_legacy"),
        ("smc_causal_replay_schema_version", "smc_causal_replay_v1"),
        ("technical_indicator_owner", {}),
        ("swing_structure_owner", {}),
    ),
)
def test_loader_rejects_legacy_schema_or_builder_before_use(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    cache_dir = _publish(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = _manifest(cache_dir)
    manifest[field] = replacement
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match="HTF_V4_CACHE_CONTRACT_REQUIRED|HTF_V4_CACHE_CONTRACT_MISMATCH",
    ):
        htf.load_multi_tf_v4_cache(cache_dir)


def test_loader_rejects_manifest_identity_tamper(tmp_path: Path) -> None:
    cache_dir = _publish(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = _manifest(cache_dir)
    manifest["cache_identity_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="HTF_V4_CACHE_IDENTITY_MISMATCH"):
        htf.load_multi_tf_v4_cache(cache_dir)


def test_loader_rejects_retired_midnight_h4_origin_even_with_rehashed_identity(
    tmp_path: Path,
) -> None:
    cache_dir = _publish(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = _manifest(cache_dir)
    manifest["resample_origin_contract"]["H4"] = str(pd.Timedelta(0))
    manifest["cache_identity_sha256"] = htf.compute_htf_v4_cache_identity(
        manifest
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="resample_origin_contract"):
        htf.load_multi_tf_v4_cache(cache_dir)


def test_loader_rejects_duplicate_manifest_keys(tmp_path: Path) -> None:
    cache_dir = _publish(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    payload = manifest_path.read_text(encoding="utf-8").replace(
        '"schema_version":',
        '"schema_version": "duplicate", "schema_version":',
        1,
    )
    manifest_path.write_text(payload, encoding="utf-8")
    with pytest.raises(RuntimeError, match="HTF_V4_CACHE_MANIFEST_INVALID"):
        htf.load_multi_tf_v4_cache(cache_dir)


def test_publisher_never_replaces_existing_cache(tmp_path: Path) -> None:
    cache_dir = _publish(tmp_path)
    before = _sha256(cache_dir / "manifest.json")
    source, _index, frames = _source_and_frames(tmp_path / "second")
    with pytest.raises(RuntimeError, match="HTF_V4_CACHE_OUTPUT_EXISTS"):
        producer.publish_multi_tf_v4_cache(
            out_dir=cache_dir,
            m5_prebuilt=source,
            expected_source_sha256=_sha256(source),
            features=frames,
            v29_registry_constants=_V29_TEST_REGISTRY_CONSTANTS,
            volatility_squeeze_artifacts=_SQUEEZE_TEST_ARTIFACTS,
        )
    assert _sha256(cache_dir / "manifest.json") == before


def test_cli_rejects_wrong_source_hash_before_parquet_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (tmp_path / "not-even-parquet.bin").resolve()
    source.write_bytes(b"exact source bytes")
    monkeypatch.setattr(
        pq,
        "ParquetFile",
        lambda *_args, **_kwargs: pytest.fail(
            "parquet read started before hash gate"
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prebuild_multi_tf_cache_v4",
            "--m5-prebuilt",
            str(source),
            "--expected-source-sha256",
            "0" * 64,
            "--out-dir",
            str((tmp_path / "cache").resolve()),
                "--registry-fit-train-start", "2026-01-01T00:00:00+00:00",
                "--registry-fit-inner-end", "2026-01-03T00:00:00+00:00",
                "--registry-fit-train-end", "2026-01-05T00:00:00+00:00",
                "--registry-fit-tape-manifest", str(source),
                "--expected-registry-fit-tape-manifest-sha256", "0" * 64,
                "--registry-fit-split-manifest", str(source),
                "--expected-registry-fit-split-manifest-sha256", "0" * 64,
                "--registry-fit-train-split-id", "synthetic:TRAIN",
                "--volatility-squeeze-manifest",
                str(_SQUEEZE_TEST_ARTIFACTS.manifest_path),
                "--expected-volatility-squeeze-manifest-sha256",
                _SQUEEZE_TEST_ARTIFACTS.manifest_file_sha256,
            ],
    )
    with pytest.raises(RuntimeError, match="CACHE_V4_SOURCE_SHA256_MISMATCH"):
        producer.main()


@pytest.mark.parametrize(
    "invalid_sha256",
    ("", "abc", "A" * 64, "g" * 64, "0" * 63, "0" * 65),
)
def test_exact_source_hash_rejects_noncanonical_values(
    invalid_sha256: str,
) -> None:
    with pytest.raises(RuntimeError, match="exact lowercase SHA-256 required"):
        producer._require_exact_sha256(invalid_sha256, label="TEST")


def test_publisher_rejects_timestamps_not_derived_from_native_m5(
    tmp_path: Path,
) -> None:
    source, _index, frames = _source_and_frames(tmp_path)
    shifted = frames["H4"].copy()
    shifted.index = shifted.index + pd.Timedelta(minutes=5)
    shifted_values = shifted.to_numpy(dtype=np.float32, copy=False)
    shifted.attrs["feats_np"] = shifted_values
    shifted.attrs["ts_int64"] = shifted.index.asi8.astype(np.int64, copy=True)
    frames["H4"] = shifted
    with pytest.raises(
        RuntimeError,
        match="HTF_V4_CACHE_SOURCE_TIMESTAMP_GEOMETRY_MISMATCH: H4",
    ):
        producer.publish_multi_tf_v4_cache(
            out_dir=tmp_path / "cache",
            m5_prebuilt=source,
            expected_source_sha256=_sha256(source),
            features=frames,
            v29_registry_constants=_V29_TEST_REGISTRY_CONSTANTS,
            volatility_squeeze_artifacts=_SQUEEZE_TEST_ARTIFACTS,
        )


def test_parent_fsync_failure_reports_visible_inadmissible_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _index, frames = _source_and_frames(tmp_path)
    destination = tmp_path / "cache"
    real_fsync = producer._fsync_directory
    calls = 0

    def fail_after_publish(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected parent fsync failure")
        real_fsync(path)

    monkeypatch.setattr(producer, "_fsync_directory", fail_after_publish)
    with pytest.raises(
        RuntimeError,
        match="HTF_V4_CACHE_PUBLISHED_PARENT_FSYNC_FAILED",
    ):
        producer.publish_multi_tf_v4_cache(
            out_dir=destination,
            m5_prebuilt=source,
            expected_source_sha256=_sha256(source),
            features=frames,
            v29_registry_constants=_V29_TEST_REGISTRY_CONSTANTS,
            volatility_squeeze_artifacts=_SQUEEZE_TEST_ARTIFACTS,
        )
    assert destination.is_dir()
    assert (destination / "manifest.json").is_file()


def test_v4_disk_projection_matches_in_memory_verified_bytes(
    tmp_path: Path,
) -> None:
    source, source_index, in_memory = _source_and_frames(tmp_path)
    cache_dir = tmp_path / "cache"
    producer.publish_multi_tf_v4_cache(
        out_dir=cache_dir,
        m5_prebuilt=source,
        expected_source_sha256=_sha256(source),
        features=in_memory,
        v29_registry_constants=_V29_TEST_REGISTRY_CONSTANTS,
        volatility_squeeze_artifacts=_SQUEEZE_TEST_ARTIFACTS,
    )
    from_disk = htf.load_multi_tf_v4_cache(cache_dir)
    targets = pd.date_range(
        source_index[-8],
        periods=6,
        freq="1min",
    ).asi8
    projection = (("verified_probe", htf.MULTI_TF_PER_BAR_FEATURES_V4[0]),)
    kwargs = {
        "target_ts_ns": targets,
        "per_tf_map": projection,
        "tfs": ("m5", "m15", "h1", "h4", "d1"),
        "decision_bar_duration": pd.Timedelta(minutes=1),
    }
    expected = htf.project_multi_tf_v4_scalars(in_memory, **kwargs)
    observed = htf.project_multi_tf_v4_scalars(from_disk, **kwargs)
    assert tuple(observed) == tuple(expected)
    for name in expected:
        np.testing.assert_array_equal(observed[name], expected[name])


def test_loader_rejects_legacy_manifest_before_array_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = _publish(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = _manifest(cache_dir)
    manifest["schema_version"] = "htf_v3_disk_cache_manifest_v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    array_load_attempted = False

    def forbidden_array_load(*_args, **_kwargs):
        nonlocal array_load_attempted
        array_load_attempted = True
        raise AssertionError("legacy arrays must not load")

    monkeypatch.setattr(htf, "_load_verified_cache_npy", forbidden_array_load)
    with pytest.raises(RuntimeError, match="HTF_V4_CACHE_CONTRACT_REQUIRED"):
        htf.load_multi_tf_v4_cache(cache_dir)
    assert array_load_attempted is False


def test_v4_cache_rejects_hash_valid_false_liveness_claim(
    tmp_path: Path,
) -> None:
    cache_dir = _publish(tmp_path)
    manifest_path = cache_dir / "manifest.json"
    manifest = _manifest(cache_dir)
    manifest["full_input_liveness"]["timeframes"]["H4"]["fields"][
        htf.MULTI_TF_PER_BAR_FEATURES_V4[0]
    ]["std"] = 999.0
    manifest["cache_identity_sha256"] = htf.compute_htf_v4_cache_identity(
        manifest
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match="HTF_V4_CACHE_FULL_INPUT_LIVENESS_INVALID",
    ):
        htf.load_multi_tf_v4_cache(cache_dir)
