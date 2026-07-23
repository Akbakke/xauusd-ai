from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.execution import v12_state_from_prebuilt as state_module
from gx1.execution.v12_state_from_prebuilt import (
    PrebuiltIdentityError,
    PrebuiltStateLoader,
)


REPO = Path(__file__).resolve().parents[1]


def test_async_refresh_aborts_cv3_swap_when_mtf_refresh_fails() -> None:
    text = (REPO / "gx1/execution/v12_state_from_prebuilt.py").read_text(encoding="utf-8")

    assert "new-cv3/stale-mtf split-brain" in text
    assert "keeping stale" not in text


def test_async_refresh_reaugments_when_base28_advances() -> None:
    text = (REPO / "gx1/execution/v12_state_from_prebuilt.py").read_text(encoding="utf-8")

    assert "cv3_advanced or b28_advanced" in text


def test_active_prebuilt_augmentation_has_no_alternate_or_skip_path() -> None:
    text = (REPO / "gx1/execution/v12_state_from_prebuilt.py").read_text(encoding="utf-8")

    assert "sequential-fallback" not in text
    assert "augment skipped" not in text
    assert "falling back to on-disk cache" not in text
    assert "refusing a second transform path" in text


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(
    path: Path,
    parquet: Path,
    *,
    rows: int,
    cols_total: int,
    parquet_sha256: str | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "parquet_path": str(parquet.resolve()),
                "parquet_sha256": parquet_sha256 or _sha256(parquet),
                "rows": rows,
                "cols_total": cols_total,
            }
        ),
        encoding="utf-8",
    )


def _prebuilt_fixture(tmp_path: Path) -> dict[str, Path]:
    index = pd.date_range("2026-07-16T12:00:00Z", periods=3, freq="5min")
    cv3 = tmp_path / "canonical_v3.parquet"
    base28 = tmp_path / "base28.parquet"
    cv3_manifest = tmp_path / "cv3_CURRENT_MANIFEST.json"
    base28_manifest = tmp_path / "base28_CURRENT_MANIFEST.json"
    pd.DataFrame(
        {
            "time": index,
            "open": np.array([2400.0, 2401.0, 2402.0], dtype=np.float64),
            "signal": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        }
    ).to_parquet(cv3, index=False)
    pd.DataFrame(
        {"atr_bps": np.array([10.0, 11.0, 12.0], dtype=np.float32)},
        index=index.rename("time"),
    ).to_parquet(base28, index=True)
    _write_manifest(cv3_manifest, cv3, rows=3, cols_total=2)
    _write_manifest(base28_manifest, base28, rows=3, cols_total=1)
    return {
        "cv3": cv3,
        "base28": base28,
        "cv3_manifest": cv3_manifest,
        "base28_manifest": base28_manifest,
    }


def _disable_augmenters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _identity(
        self: PrebuiltStateLoader,
        cv3: pd.DataFrame | None = None,
    ) -> pd.DataFrame | None:
        return self._cv3 if cv3 is None else cv3

    for name in (
        "_augment_cv3_with_volume_features",
        "_augment_cv3_with_v2_mtf_scalars",
        "_augment_cv3_with_group_a_and_dip_struct",
        "_augment_cv3_with_v1_legacy",
        "_augment_cv3_with_regime_v4",
    ):
        monkeypatch.setattr(PrebuiltStateLoader, name, _identity)


def _loader(paths: dict[str, Path]) -> PrebuiltStateLoader:
    return PrebuiltStateLoader(
        canonical_v3_manifest_path=paths["cv3_manifest"],
        base28_manifest_path=paths["base28_manifest"],
    )


def test_initial_load_is_bound_to_both_current_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    _disable_augmenters(monkeypatch)
    loader = _loader(paths)

    loader.load()

    assert loader.canonical_v3_path == paths["cv3"].resolve()
    assert loader.base28_path == paths["base28"].resolve()
    assert loader._cv3_binding is not None
    assert loader._base28_binding is not None
    assert loader._cv3_binding.parquet_sha256 == _sha256(paths["cv3"])
    assert loader._base28_binding.parquet_sha256 == _sha256(paths["base28"])
    assert loader._cv3 is not None and loader._cv3.shape == (3, 2)
    assert loader._base28 is not None and loader._base28.shape == (3, 1)


@pytest.mark.parametrize(
    ("artifact", "manifest_change", "error"),
    [
        ("cv3", {"parquet_sha256": "0" * 64}, "CANONICAL_V3_PARQUET_SHA256_MISMATCH"),
        ("cv3", {"rows": 4}, "CANONICAL_V3_PARQUET_ROWS_MISMATCH"),
        ("base28", {"cols_total": 2}, "BASE28_PARQUET_SCHEMA_COUNT_MISMATCH"),
    ],
)
def test_initial_load_rejects_manifest_hash_rows_or_schema_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    manifest_change: dict[str, object],
    error: str,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    _disable_augmenters(monkeypatch)
    manifest_path = paths[f"{artifact}_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(manifest_change)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PrebuiltIdentityError, match=error):
        _loader(paths).load()


def test_explicit_path_is_only_an_assertion_not_a_manifest_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    _disable_augmenters(monkeypatch)
    alternate = tmp_path / "alternate_cv3.parquet"
    alternate.write_bytes(paths["cv3"].read_bytes())
    loader = _loader(paths)
    loader.canonical_v3_path = alternate

    with pytest.raises(
        PrebuiltIdentityError,
        match="CANONICAL_V3_MANIFEST_PARQUET_PATH_MISMATCH",
    ):
        loader.load()


def test_hot_refresh_manifest_identity_error_is_latched_and_blocks_old_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    _disable_augmenters(monkeypatch)
    loader = _loader(paths)
    loader.load()
    paths["base28_manifest"].write_text("{broken", encoding="utf-8")

    with pytest.raises(
        PrebuiltIdentityError,
        match="BASE28_CURRENT_MANIFEST_JSON_INVALID",
    ):
        loader.refresh_if_changed()
    with pytest.raises(PrebuiltIdentityError, match="PREBUILT_REFRESH_LATCHED"):
        _ = loader.cutoff_ts
    with pytest.raises(PrebuiltIdentityError, match="PREBUILT_REFRESH_LATCHED"):
        loader.get_window(pd.Timestamp("2026-07-16T12:10:00Z"))


def test_hot_refresh_rejects_exact_schema_drift_even_with_valid_new_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    _disable_augmenters(monkeypatch)
    loader = _loader(paths)
    loader.load()

    frame = pd.read_parquet(paths["cv3"])
    frame["signal"] = frame["signal"].astype(np.float64)
    frame.to_parquet(paths["cv3"], index=False)
    _write_manifest(paths["cv3_manifest"], paths["cv3"], rows=3, cols_total=2)

    with pytest.raises(
        PrebuiltIdentityError,
        match="CANONICAL_V3_PARQUET_SCHEMA_IDENTITY_MISMATCH",
    ):
        loader.refresh_if_changed()
    with pytest.raises(PrebuiltIdentityError, match="PREBUILT_REFRESH_LATCHED"):
        _ = loader.cutoff_ts


def test_async_identity_failure_is_latched_instead_of_serving_old_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    _disable_augmenters(monkeypatch)
    loader = _loader(paths)
    loader.load()

    frame = pd.read_parquet(paths["cv3"])
    frame["signal"] = frame["signal"] + 1.0
    frame.to_parquet(paths["cv3"], index=False)
    _write_manifest(paths["cv3_manifest"], paths["cv3"], rows=3, cols_total=2)

    class _InlineThread:
        def __init__(self, *, target, args, **_kwargs) -> None:
            self._target = target
            self._args = args

        def start(self) -> None:
            self._target(*self._args)

        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr(state_module.threading, "Thread", _InlineThread)

    def _identity_failure(*_args, **_kwargs):
        raise PrebuiltIdentityError("CANONICAL_V3_PARQUET_SHA256_MISMATCH")

    monkeypatch.setattr(
        state_module,
        "_load_verified_prebuilt",
        _identity_failure,
    )

    assert loader.refresh_if_changed() is True
    assert loader._refresh_error is not None
    with pytest.raises(PrebuiltIdentityError, match="PREBUILT_REFRESH_LATCHED"):
        loader.get_window(pd.Timestamp("2026-07-16T12:10:00Z"))
