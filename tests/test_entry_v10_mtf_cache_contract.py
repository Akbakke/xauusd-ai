from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.features import htf_features
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


def _write_m5_source(path: Path) -> None:
    time = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    close = np.asarray([2000.0, 2000.1, 2000.2], dtype=np.float32)
    pd.DataFrame(
        {
            "time": time,
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        }
    ).to_parquet(path, index=False)


def test_exact_mtf_prebuild_builds_once_and_reuses_one_cache_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m5_path = tmp_path / "xau_m5.parquet"
    _write_m5_source(m5_path)
    cache: dict[str, dict[str, pd.DataFrame]] = {}
    monkeypatch.setattr(trainer, "_MULTI_TF_CACHE", cache)

    index = pd.DatetimeIndex([pd.Timestamp("2026-01-01", tz="UTC")])
    expected = {
        tf: pd.DataFrame(np.zeros((1, 25), dtype=np.float32), index=index)
        for tf in ("M5", "M15", "H1", "H4", "D1")
    }
    calls = 0

    def fake_build(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        nonlocal calls
        calls += 1
        assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(frame.index, pd.DatetimeIndex)
        assert str(frame.index.tz) == "UTC"
        return expected

    monkeypatch.setattr(
        htf_features,
        "build_multi_tf_per_bar_features_v2",
        fake_build,
    )

    first = trainer._prebuild_multi_tf_v2_features_once(m5_path)
    second = trainer._prebuild_multi_tf_v2_features_once(m5_path)
    cache_key = trainer._multi_tf_cache_key(m5_path)

    assert calls == 1
    assert first is expected
    assert second is first
    assert cache == {cache_key: expected}
    assert cache[cache_key] is first


def test_in_process_mtf_cache_does_not_reuse_mutated_m5_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m5_path = tmp_path / "xau_m5.parquet"
    _write_m5_source(m5_path)
    monkeypatch.setattr(trainer, "_MULTI_TF_CACHE", {})
    monkeypatch.setattr(trainer, "_MULTI_TF_ACTIVE_CACHE_KEYS", {})
    calls = 0

    def fake_build(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        nonlocal calls
        calls += 1
        return {
            tf: pd.DataFrame(
                np.full((1, 25), calls, dtype=np.float32),
                index=pd.DatetimeIndex([pd.Timestamp("2026-01-01", tz="UTC")]),
            )
            for tf in ("M5", "M15", "H1", "H4", "D1")
        }

    monkeypatch.setattr(
        htf_features,
        "build_multi_tf_per_bar_features_v2",
        fake_build,
    )
    first = trainer._prebuild_multi_tf_v2_features_once(m5_path)
    changed = pd.read_parquet(m5_path)
    changed.loc[0, "close"] = float(changed.loc[0, "close"]) + 1.0
    changed.to_parquet(m5_path, index=False)
    second = trainer._prebuild_multi_tf_v2_features_once(m5_path)

    assert calls == 2
    assert second is not first
    assert len(trainer._MULTI_TF_CACHE) == 2


def test_prebuild_owner_uses_verified_disk_cache_and_binds_cache_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m5_path = (tmp_path / "xau_m5.parquet").resolve()
    _write_m5_source(m5_path)
    disk_cache = (tmp_path / "verified_cache").resolve()
    disk_cache.mkdir()
    monkeypatch.setenv("GX1_V10_MULTI_TF_V2_CACHE_DIR", str(disk_cache))
    monkeypatch.setattr(trainer, "_MULTI_TF_CACHE", {})
    monkeypatch.setattr(trainer, "_MULTI_TF_ACTIVE_CACHE_KEYS", {})
    source_sha256 = trainer._sha256_file(m5_path)
    index = pd.DatetimeIndex(
        pd.read_parquet(m5_path, columns=["time"])["time"]
    )

    class VerifiedCache(dict):
        cache_identity_sha256 = "a" * 64
        m5_prebuilt_source = str(m5_path)
        m5_prebuilt_source_sha256 = source_sha256

    verified = VerifiedCache(
        {
            tf: pd.DataFrame(
                np.zeros((len(index), 25), dtype=np.float32),
                index=index,
            )
            for tf in ("M5", "M15", "H1", "H4", "D1")
        }
    )
    load_calls = 0

    def fake_load(cache_dir: str) -> VerifiedCache:
        nonlocal load_calls
        load_calls += 1
        assert Path(cache_dir).resolve() == disk_cache
        return verified

    monkeypatch.setattr(htf_features, "load_multi_tf_v2_cache", fake_load)
    monkeypatch.setattr(
        htf_features,
        "build_multi_tf_per_bar_features_v2",
        lambda frame: pytest.fail("source builder bypassed verified disk cache"),
    )

    first = trainer._prebuild_multi_tf_v2_features_once(m5_path)
    second = trainer._prebuild_multi_tf_v2_features_once(m5_path)

    assert first is verified
    assert second is first
    assert load_calls == 1
    assert len(trainer._MULTI_TF_CACHE) == 1
    assert "cache_identity=" + ("a" * 64) in next(iter(trainer._MULTI_TF_CACHE))


@pytest.mark.parametrize(
    "invalid_mode",
    ("", "V1", "V2", "v2_causal", "V2_CAUSAL_COMPAT", None, True),
)
def test_mtf_cache_key_rejects_every_non_exact_contract_mode(
    tmp_path: Path,
    invalid_mode: object,
) -> None:
    with pytest.raises(RuntimeError, match="MULTI_TF_CACHE_CONTRACT_MODE_INVALID"):
        trainer._multi_tf_cache_key(
            tmp_path / "xau_m5.parquet",
            contract_mode=invalid_mode,  # type: ignore[arg-type]
        )


def test_active_mtf_prebuild_has_no_legacy_v1_builder_or_mode_lane() -> None:
    owner_source = inspect.getsource(trainer._prebuild_multi_tf_v2_features_once)
    train_source = inspect.getsource(trainer.run_train)

    assert "build_multi_tf_per_bar_features_v2" in owner_source
    assert "build_multi_tf_per_bar_features(" not in owner_source
    assert "v2_mode =" not in train_source
    assert "|v2=" not in train_source
