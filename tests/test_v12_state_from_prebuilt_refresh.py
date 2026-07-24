from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.execution import v12_canonical_incremental as incremental
from gx1.execution import v12_state_from_prebuilt as state_module
from gx1.execution.v12_state_from_prebuilt import (
    PrebuiltIdentityError,
    PrebuiltStateLoader,
    read_prebuilt_pair_manifest,
)


REPO = Path(__file__).resolve().parents[1]


def test_async_refresh_aborts_cv3_swap_when_mtf_refresh_fails() -> None:
    text = (REPO / "gx1/execution/v12_state_from_prebuilt.py").read_text(
        encoding="utf-8"
    )

    assert "new-cv3/stale-mtf split-brain" in text
    assert "keeping stale" not in text


def test_async_refresh_reaugments_when_base28_advances() -> None:
    text = (REPO / "gx1/execution/v12_state_from_prebuilt.py").read_text(
        encoding="utf-8"
    )

    assert "cv3_advanced or b28_advanced" in text


def test_active_prebuilt_augmentation_has_no_alternate_or_skip_path() -> None:
    text = (REPO / "gx1/execution/v12_state_from_prebuilt.py").read_text(
        encoding="utf-8"
    )

    assert "sequential-fallback" not in text
    assert "augment skipped" not in text
    assert "falling back to on-disk cache" not in text
    assert "refusing a second transform path" in text


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frames(
    *,
    signal_dtype: np.dtype = np.dtype(np.float32),
    signal_offset: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2026-07-16T12:00:00Z", periods=3, freq="5min")
    canonical = pd.DataFrame(
        {
            "time": index,
            "open": np.array([2400.0, 2401.0, 2402.0], dtype=np.float64),
            "signal": np.array(
                [1.0 + signal_offset, 2.0 + signal_offset, 3.0 + signal_offset],
                dtype=signal_dtype,
            ),
        }
    )
    base28 = pd.DataFrame(
        {
            name: np.asarray(
                [2400.0 + offset, 2401.0 + offset, 2402.0 + offset],
                dtype=np.float64,
            )
            for offset, name in enumerate(incremental.RAW_BASE28_COLUMNS)
        },
        index=index.rename("time"),
    )
    return canonical, base28


def _lineage() -> dict[str, object]:
    source_files = [{"path": "fixture.py", "sha256": "1" * 64}]
    inventory_sha = hashlib.sha256(
        json.dumps(
            source_files,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    def native(timeframe: str) -> dict[str, object]:
        return {
            "root": f"/fixture/{timeframe.lower()}",
            "manifest_sha256": "2" * 64,
            "instrument": "XAU_USD",
            "timeframe": timeframe,
            "explicit_vedtak_id": "FIXTURE_VEDTAK_001",
            "source_environment": "practice",
            "source_base_url": "https://api-fxpractice.oanda.com/v3",
            "requested_start_utc": "2026-07-16T00:00:00+00:00",
            "requested_end_utc_exclusive": "2026-07-17T00:00:00+00:00",
            "row_count": 3,
            "time_min_utc": "2026-07-16T12:00:00+00:00",
            "time_max_utc": "2026-07-16T12:10:00+00:00",
            "canonical_rows_sha256": "3" * 64,
            "producer_git_commit": "4" * 40,
            "producer_source_inventory_sha256": "5" * 64,
            "manifest_payload_sha256": "6" * 64,
        }

    return {
        "schema_version": state_module.PREBUILT_PAIR_LINEAGE_SCHEMA_VERSION,
        "explicit_vedtak_id": "FIXTURE_VEDTAK_001",
        "producer_owner": "fixture",
        "producer_git_commit": "7" * 40,
        "producer_repository_clean": True,
        "producer_source_files": source_files,
        "producer_source_inventory_sha256": inventory_sha,
        "native_sources": {"m1": native("M1"), "m5": native("M5")},
        "derivation_contract": {
            "raw_base28_columns": list(incremental.RAW_BASE28_COLUMNS),
            "rank_fit_fields_absent": True,
            "m5_phase_owned_by_m1_time": True,
            "formula_contract": {"fixture": "v1"},
            "timing_contract": {"fixture": "v1"},
        },
        "coverage": {"fixture_rows": 3},
        "parent_pair_generation_id": None,
    }


def _write_staged_pair(
    staging_dir: Path,
    canonical: pd.DataFrame,
    base28: pd.DataFrame,
) -> None:
    incremental._write_candidate_parquet(
        canonical,
        staging_dir / incremental.PAIR_CANONICAL_FILENAME,
        index=False,
    )
    incremental._write_candidate_parquet(
        base28.reset_index(),
        staging_dir / incremental.PAIR_BASE28_FILENAME,
        index=False,
    )


def _prebuilt_fixture(tmp_path: Path) -> dict[str, Path | str]:
    generation_root = tmp_path / "generations"
    pair_manifest = tmp_path / "CANONICAL_V3_BASE28_CURRENT_PAIR_MANIFEST.json"
    canonical, base28 = _frames()
    staging_dir = incremental._candidate_staging_path(generation_root)
    _write_staged_pair(staging_dir, canonical, base28)
    generation_id = incremental._publish_prebuilt_pair_generation(
        staging_dir,
        pair_manifest_path=pair_manifest,
        generation_root=generation_root,
        expected_pair_generation_id=None,
        expected_manifest_sha256=None,
        lineage_contract=_lineage(),
        created_utc="2026-07-23T00:00:00Z",
    )
    binding = read_prebuilt_pair_manifest(
        pair_manifest,
        generation_root=generation_root,
    )
    return {
        "generation_root": generation_root,
        "pair_manifest": pair_manifest,
        "generation_id": generation_id,
        "cv3": binding.canonical_v3.parquet_path,
        "base28": binding.base28.parquet_path,
    }


def _publish_next_pair(
    paths: dict[str, Path | str],
    *,
    canonical: pd.DataFrame,
    base28: pd.DataFrame,
) -> str:
    generation_root = Path(paths["generation_root"])
    pair_manifest = Path(paths["pair_manifest"])
    current = read_prebuilt_pair_manifest(
        pair_manifest,
        generation_root=generation_root,
    )
    staging_dir = incremental._candidate_staging_path(generation_root)
    try:
        _write_staged_pair(staging_dir, canonical, base28)
        return incremental._publish_prebuilt_pair_generation(
            staging_dir,
            pair_manifest_path=pair_manifest,
            generation_root=generation_root,
            expected_pair_generation_id=current.pair_generation_id,
            expected_manifest_sha256=current.manifest_sha256,
            lineage_contract=_lineage(),
            created_utc="2026-07-23T00:05:00Z",
        )
    finally:
        incremental._discard_pair_staging_dir(
            staging_dir,
            generation_root=generation_root,
        )


def _disable_augmenters(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(
        state_module,
        "_require_persisted_model_agnostic_canonical",
        lambda _frame: None,
    )


def _loader(paths: dict[str, Path | str]) -> PrebuiltStateLoader:
    return PrebuiltStateLoader(
        pair_manifest_path=Path(paths["pair_manifest"]),
        generation_root=Path(paths["generation_root"]),
    )


def test_initial_load_is_bound_to_one_atomic_pair_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    _disable_augmenters(monkeypatch)
    loader = _loader(paths)

    loader.load()

    assert loader.canonical_v3_path == Path(paths["cv3"]).resolve()
    assert loader.base28_path == Path(paths["base28"]).resolve()
    assert loader._pair_binding is not None
    assert loader._cv3_binding is not None
    assert loader._base28_binding is not None
    assert loader._pair_binding.pair_generation_id == paths["generation_id"]
    assert loader._cv3_binding.pair_generation_id == paths["generation_id"]
    assert loader._base28_binding.pair_generation_id == paths["generation_id"]
    assert loader._cv3_binding.parquet_sha256 == _sha256(Path(paths["cv3"]))
    assert loader._base28_binding.parquet_sha256 == _sha256(Path(paths["base28"]))
    assert loader._cv3 is not None and loader._cv3.shape == (3, 2)
    assert loader._base28 is not None and loader._base28.shape == (3, 13)


def test_frozen_pair_load_disables_refresh_and_returns_exact_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    _disable_augmenters(monkeypatch)
    loader = _loader(paths)

    identity = loader.load_frozen_pair()

    assert identity["pair_generation_id"] == paths["generation_id"]
    assert identity["manifest_sha256"] == _sha256(
        Path(paths["pair_manifest"])
    )
    assert identity["canonical_v3"]["sha256"] == _sha256(Path(paths["cv3"]))
    assert identity["base28"]["sha256"] == _sha256(Path(paths["base28"]))
    assert identity["lineage"] == _lineage()
    assert identity["lineage_sha256"] == hashlib.sha256(
        json.dumps(
            _lineage(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert identity["refresh_enabled"] is False
    assert loader._refresh_enabled is False
    assert loader.refresh_if_changed() is False
    canonical, base, frame_identity = loader.frozen_pair_frames()
    assert canonical is loader._cv3
    assert base is loader._base28
    assert frame_identity == identity

    base_path = Path(paths["base28"])
    base_path.write_bytes(base_path.read_bytes() + b"tamper")
    with pytest.raises(
        PrebuiltIdentityError,
        match="PREBUILT_FROZEN_PAIR_FILES_CHANGED",
    ):
        loader.frozen_pair_frames()


def test_initial_load_rejects_artifact_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    _disable_augmenters(monkeypatch)
    cv3_path = Path(paths["cv3"])
    cv3_path.write_bytes(cv3_path.read_bytes() + b"tamper")

    with pytest.raises(
        PrebuiltIdentityError,
        match="CANONICAL_V3_PARQUET_SHA256_MISMATCH",
    ):
        _loader(paths).load()


def test_initial_load_rejects_artifact_row_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    _disable_augmenters(monkeypatch)
    manifest = json.loads(Path(paths["pair_manifest"]).read_text(encoding="utf-8"))
    declared_sha = manifest["artifacts"]["canonical_v3"]["parquet_sha256"]
    canonical, _base28 = _frames()
    canonical.iloc[:2].to_parquet(Path(paths["cv3"]), index=False)
    monkeypatch.setattr(state_module, "_sha256_file", lambda _path: declared_sha)

    with pytest.raises(
        PrebuiltIdentityError,
        match="CANONICAL_V3_PARQUET_ROWS_MISMATCH",
    ):
        _loader(paths).load()


def test_pair_manifest_path_cannot_escape_generation_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    _disable_augmenters(monkeypatch)
    alternate = tmp_path / "alternate_cv3.parquet"
    alternate.write_bytes(Path(paths["cv3"]).read_bytes())
    manifest_path = Path(paths["pair_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["canonical_v3"]["parquet_path"] = str(alternate.resolve())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        PrebuiltIdentityError,
        match="CANONICAL_V3_PAIR_PARQUET_PATH_NOT_GENERATION_EXACT",
    ):
        _loader(paths).load()


def test_mixed_generation_pair_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    _disable_augmenters(monkeypatch)
    old_manifest = json.loads(
        Path(paths["pair_manifest"]).read_text(encoding="utf-8")
    )
    canonical, base28 = _frames(signal_offset=10.0)
    _publish_next_pair(paths, canonical=canonical, base28=base28)
    manifest_path = Path(paths["pair_manifest"])
    mixed = json.loads(manifest_path.read_text(encoding="utf-8"))
    mixed["artifacts"]["base28"] = old_manifest["artifacts"]["base28"]
    manifest_path.write_text(json.dumps(mixed), encoding="utf-8")

    with pytest.raises(
        PrebuiltIdentityError,
        match="BASE28_PAIR_PARQUET_PATH_NOT_GENERATION_EXACT",
    ):
        _loader(paths).load()


def test_hot_refresh_pair_manifest_error_is_latched_and_blocks_old_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    _disable_augmenters(monkeypatch)
    loader = _loader(paths)
    loader.load()
    Path(paths["pair_manifest"]).write_text("{broken", encoding="utf-8")

    with pytest.raises(
        PrebuiltIdentityError,
        match="PREBUILT_PAIR_MANIFEST_JSON_INVALID",
    ):
        loader.refresh_if_changed()
    with pytest.raises(PrebuiltIdentityError, match="PREBUILT_REFRESH_LATCHED"):
        _ = loader.cutoff_ts
    with pytest.raises(PrebuiltIdentityError, match="PREBUILT_REFRESH_LATCHED"):
        loader.get_window(pd.Timestamp("2026-07-16T12:10:00Z"))


def test_hot_refresh_rejects_exact_schema_drift_in_valid_new_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    _disable_augmenters(monkeypatch)
    loader = _loader(paths)
    loader.load()
    canonical, base28 = _frames(signal_dtype=np.dtype(np.float64))
    _publish_next_pair(paths, canonical=canonical, base28=base28)

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
    canonical, base28 = _frames(signal_offset=1.0)
    _publish_next_pair(paths, canonical=canonical, base28=base28)

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


def test_pointer_replace_failure_never_serves_torn_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    manifest_path = Path(paths["pair_manifest"])
    generation_root = Path(paths["generation_root"])
    previous_bytes = manifest_path.read_bytes()
    current = read_prebuilt_pair_manifest(
        manifest_path,
        generation_root=generation_root,
    )
    generations_before = {
        item.name for item in generation_root.iterdir() if not item.name.startswith(".")
    }
    canonical, base28 = _frames(signal_offset=5.0)
    staging_dir = incremental._candidate_staging_path(generation_root)
    _write_staged_pair(staging_dir, canonical, base28)
    real_replace = incremental.os.replace

    def _fail_pointer_replace(source, destination) -> None:
        if Path(destination) == manifest_path:
            raise OSError("simulated pointer replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(incremental.os, "replace", _fail_pointer_replace)
    try:
        with pytest.raises(OSError, match="simulated pointer replacement failure"):
            incremental._publish_prebuilt_pair_generation(
                staging_dir,
                pair_manifest_path=manifest_path,
                generation_root=generation_root,
                expected_pair_generation_id=current.pair_generation_id,
                expected_manifest_sha256=current.manifest_sha256,
                lineage_contract=_lineage(),
            )
    finally:
        incremental._discard_pair_staging_dir(
            staging_dir,
            generation_root=generation_root,
        )

    assert manifest_path.read_bytes() == previous_bytes
    admitted = read_prebuilt_pair_manifest(
        manifest_path,
        generation_root=generation_root,
    )
    assert admitted.pair_generation_id == current.pair_generation_id
    generations_after = {
        item.name for item in generation_root.iterdir() if not item.name.startswith(".")
    }
    assert generations_after == generations_before
    assert not list(generation_root.glob(".staging-*"))
    assert not list(tmp_path.glob(f".{manifest_path.name}.*.tmp"))


def test_bootstrap_is_native_derived_and_one_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation_root = tmp_path / "generations"
    pair_manifest = tmp_path / "pair.json"
    checkpoint = tmp_path / "checkpoints"
    native_lineage = _lineage()["native_sources"]
    descriptors: dict[str, dict[str, object]] = {}
    for label in ("m1", "m5"):
        descriptor = dict(native_lineage[label])
        descriptor.update(
            {
                "manifest_path": f"/fixture/{label}/MANIFEST.json",
                "year_sha256": {"year=2026": "8" * 64},
                "year_rows": {"year=2026": 6 if label == "m1" else 2},
            }
        )
        descriptors[label] = descriptor
    descriptors["m1"]["row_count"] = 6
    descriptors["m5"]["row_count"] = 2

    m1_times = pd.date_range("2026-07-16T12:04:00Z", periods=6, freq="min")
    native_m1 = pd.DataFrame(
        {
            "time": m1_times,
            **{
                name: np.asarray(
                    [2400.0 + offset + row for row in range(6)],
                    dtype=np.float64,
                )
                for offset, name in enumerate(incremental.RAW_BASE28_COLUMNS)
            },
        }
    )
    native_m5 = pd.DataFrame(
        {
            "time": pd.DatetimeIndex(
                ["2026-07-16T12:00:00Z", "2026-07-16T12:05:00Z"]
            )
        }
    )
    canonical = pd.DataFrame(
        {
            "open": [2400.0, 2401.0],
            "signal": [1.0, 2.0],
        },
        index=pd.DatetimeIndex(native_m5["time"], name="time"),
    )

    monkeypatch.setattr(
        incremental,
        "_clean_repository_commit",
        lambda _root: "7" * 40,
    )
    monkeypatch.setattr(
        incremental,
        "canonical_xau_source_descriptor_v1",
        lambda _root, *, timeframe: descriptors[timeframe.lower()],
    )
    monkeypatch.setattr(
        incremental,
        "_load_native_source_frame",
        lambda _descriptor, *, timeframe: (
            native_m1.copy() if timeframe == "M1" else native_m5.copy()
        ),
    )
    monkeypatch.setattr(
        incremental,
        "_build_model_agnostic_canonical",
        lambda *_args, **_kwargs: canonical.copy(),
    )
    monkeypatch.setattr(
        incremental,
        "_pair_producer_source_inventory",
        lambda _root: [{"path": "fixture.py", "sha256": "1" * 64}],
    )

    generation_id = incremental.bootstrap_prebuilt_pair(
        native_m1_root=tmp_path / "native-m1",
        native_m5_root=tmp_path / "native-m5",
        vedtak_id="FIXTURE_VEDTAK_001",
        checkpoint_dir=checkpoint,
        pair_manifest_path=pair_manifest,
        generation_root=generation_root,
        repo_root=REPO,
    )
    admitted = read_prebuilt_pair_manifest(
        pair_manifest,
        generation_root=generation_root,
    )

    assert admitted.pair_generation_id == generation_id
    assert tuple(admitted.base28.arrow_schema[index][0] for index in range(1, 14)) == (
        incremental.RAW_BASE28_COLUMNS
    )
    assert admitted.lineage["native_sources"]["m1"]["timeframe"] == "M1"
    with pytest.raises(RuntimeError, match="active pointer already exists"):
        incremental.bootstrap_prebuilt_pair(
            native_m1_root=tmp_path / "native-m1",
            native_m5_root=tmp_path / "native-m5",
            vedtak_id="FIXTURE_VEDTAK_001",
            checkpoint_dir=checkpoint,
            pair_manifest_path=pair_manifest,
            generation_root=generation_root,
            repo_root=REPO,
        )
    assert not list(generation_root.glob(".staging-*"))


def test_copy_bootstrap_and_loop_control_are_removed() -> None:
    source = (REPO / "gx1/execution/v12_canonical_incremental.py").read_text(
        encoding="utf-8"
    )

    parameters = inspect.signature(incremental.bootstrap_prebuilt_pair).parameters
    assert "canonical_v3_path" not in parameters
    assert "base28_path" not in parameters
    assert "_copy_candidate_parquet" not in inspect.getsource(
        incremental.bootstrap_prebuilt_pair
    )
    assert "--loop" not in source
