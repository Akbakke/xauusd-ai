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
    require_prebuilt_successor_frame,
)
from gx1.contracts.live_tail_publication_v1 import (
    LiveTailAuthorityError,
    publish_live_tail_admission_event,
    publish_live_tail_publication_event,
    require_newest_live_tail_runtime_admission,
    require_live_tail_admission_event,
    require_live_tail_publication_event,
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


def _lineage(
    parent_pair_generation_id: str | None = None,
    parent_pair_manifest_sha256: str | None = None,
    *,
    artifact_rows: int = 3,
    canonical_columns: tuple[str, ...] = ("open", "signal"),
    time_min_utc: str = "2026-07-16T12:00:00+00:00",
    time_max_utc: str = "2026-07-16T12:10:00+00:00",
) -> dict[str, object]:
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
            "manifest_path": f"/fixture/{timeframe.lower()}/MANIFEST.json",
            "manifest_sha256": "2" * 64,
            "instrument": "XAU_USD",
            "timeframe": timeframe,
            "explicit_vedtak_id": "FIXTURE_VEDTAK_001",
            "source_environment": "practice",
            "source_base_url": "https://api-fxpractice.oanda.com/v3",
            "requested_start_utc": "2026-07-16T00:00:00+00:00",
            "requested_end_utc_exclusive": "2026-07-17T00:00:00+00:00",
            "row_count": artifact_rows,
            "time_min_utc": time_min_utc,
            "time_max_utc": time_max_utc,
            "canonical_rows_sha256": "3" * 64,
            "producer_git_commit": "4" * 40,
            "producer_source_inventory_sha256": "5" * 64,
            "manifest_payload_sha256": "6" * 64,
            "year_sha256": {"year=2026": "8" * 64},
            "year_rows": {"year=2026": artifact_rows},
        }

    return {
        "schema_version": state_module.PREBUILT_PAIR_LINEAGE_SCHEMA_VERSION,
        "explicit_vedtak_id": "FIXTURE_VEDTAK_001",
        "producer_owner": state_module.PREBUILT_PAIR_PRODUCER_OWNER,
        "producer_git_commit": "7" * 40,
        "producer_repository_clean": True,
        "producer_source_files": source_files,
        "producer_source_inventory_sha256": inventory_sha,
        "native_sources": {"m1": native("M1"), "m5": native("M5")},
        "derivation_contract": {
            "canonical_builder": (
                state_module.PREBUILT_CANONICAL_BUILDER_CONTRACT
            ),
            "canonical_ordered_columns_sha256": hashlib.sha256(
                json.dumps(
                    list(canonical_columns),
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "raw_base28_columns": list(incremental.RAW_BASE28_COLUMNS),
            "raw_base28_columns_sha256": hashlib.sha256(
                json.dumps(
                    list(incremental.RAW_BASE28_COLUMNS),
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "rank_fit_fields_absent": True,
            "m5_phase_owned_by_m1_time": True,
            "formula_contract": dict(
                state_module.PREBUILT_PAIR_FORMULA_CONTRACT
            ),
            "timing_contract": dict(
                state_module.PREBUILT_PAIR_TIMING_CONTRACT
            ),
        },
        "coverage": {
            "native_m1_rows": artifact_rows,
            "native_m5_rows": artifact_rows,
            "canonical_rows": artifact_rows,
            "base28_rows": artifact_rows,
            "canonical_time_min_utc": time_min_utc,
            "canonical_time_max_utc": time_max_utc,
            "base28_time_min_utc": time_min_utc,
            "base28_time_max_utc": time_max_utc,
            "canonical_warmup_prefix_rows_trimmed": 0,
        },
        "parent_pair_generation_id": parent_pair_generation_id,
        "parent_pair_manifest_sha256": parent_pair_manifest_sha256,
    }


def _frame_lineage(
    canonical: pd.DataFrame,
    base28: pd.DataFrame,
    *,
    parent_pair_generation_id: str | None = None,
    parent_pair_manifest_sha256: str | None = None,
) -> dict[str, object]:
    canonical_times = pd.DatetimeIndex(
        pd.to_datetime(
            canonical["time"]
            if "time" in canonical.columns
            else canonical.index,
            utc=True,
        )
    )
    base28_times = pd.DatetimeIndex(pd.to_datetime(base28.index, utc=True))
    if (
        len(canonical) != len(base28)
        or not canonical_times.equals(base28_times)
    ):
        raise AssertionError("fixture pair coverage must be exact")
    return _lineage(
        parent_pair_generation_id,
        parent_pair_manifest_sha256,
        artifact_rows=len(canonical),
        canonical_columns=tuple(
            name for name in canonical.columns if name != "time"
        ),
        time_min_utc=canonical_times[0].isoformat(),
        time_max_utc=canonical_times[-1].isoformat(),
    )


def _successor_frames(
    *,
    signal_dtype: np.dtype = np.dtype(np.float32),
    signal_value: float = 4.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical, base28 = _frames(signal_dtype=signal_dtype)
    successor_time = pd.Timestamp("2026-07-16T12:15:00Z")
    canonical = pd.concat(
        [
            canonical,
            pd.DataFrame(
                {
                    "time": [successor_time],
                    "open": np.asarray([2403.0], dtype=np.float64),
                    "signal": np.asarray([signal_value], dtype=signal_dtype),
                }
            ),
        ],
        ignore_index=True,
    )
    base_row = pd.DataFrame(
        {
            name: np.asarray([2403.0 + offset], dtype=np.float64)
            for offset, name in enumerate(incremental.RAW_BASE28_COLUMNS)
        },
        index=pd.DatetimeIndex([successor_time], name="time"),
    )
    base28 = pd.concat([base28, base_row])
    return canonical, base28


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
            lineage_contract=_frame_lineage(
                canonical,
                base28,
                parent_pair_generation_id=current.pair_generation_id,
                parent_pair_manifest_sha256=current.manifest_sha256,
            ),
            created_utc="2026-07-23T00:05:00Z",
        )
    finally:
        incremental._discard_pair_staging_dir(
            staging_dir,
            generation_root=generation_root,
        )


def _publish_live_tail_fixture_pair(
    paths: dict[str, Path | str],
    *,
    canonical: pd.DataFrame,
    base28: pd.DataFrame,
    created_utc: str,
) -> str:
    generation_root = Path(paths["generation_root"])
    pair_manifest = Path(paths["pair_manifest"])
    current = read_prebuilt_pair_manifest(
        pair_manifest,
        generation_root=generation_root,
    )
    lineage = _frame_lineage(
        canonical,
        base28,
        parent_pair_generation_id=current.pair_generation_id,
        parent_pair_manifest_sha256=current.manifest_sha256,
    )
    canonical_times = pd.DatetimeIndex(
        pd.to_datetime(canonical["time"], utc=True)
    )
    lineage["native_sources"]["m1"]["time_max_utc"] = (
        canonical_times[-1] + pd.Timedelta(minutes=4)
    ).isoformat()
    staging_dir = incremental._candidate_staging_path(generation_root)
    try:
        _write_staged_pair(staging_dir, canonical, base28)
        return incremental._publish_prebuilt_pair_generation(
            staging_dir,
            pair_manifest_path=pair_manifest,
            generation_root=generation_root,
            expected_pair_generation_id=current.pair_generation_id,
            expected_manifest_sha256=current.manifest_sha256,
            lineage_contract=lineage,
            created_utc=created_utc,
        )
    finally:
        incremental._discard_pair_staging_dir(
            staging_dir,
            generation_root=generation_root,
        )


def _admitted_live_tail_fixture(
    tmp_path: Path,
) -> dict[str, object]:
    paths = _prebuilt_fixture(tmp_path)
    pair_manifest = Path(paths["pair_manifest"])
    generation_root = Path(paths["generation_root"])
    publication_root = tmp_path / "live-tail-publications"
    admission_root = tmp_path / "live-tail-admissions"

    canonical_one, base28_one = _successor_frames()
    _publish_live_tail_fixture_pair(
        paths,
        canonical=canonical_one,
        base28=base28_one,
        created_utc="2026-07-16T12:20:00Z",
    )
    publication_one_path, publication_one = (
        publish_live_tail_publication_event(
            event_root=publication_root,
            pair_manifest_path=pair_manifest,
            generation_root=generation_root,
            created_utc="2026-07-16T12:20:30Z",
        )
    )

    next_time = pd.Timestamp("2026-07-16T12:20:00Z")
    canonical_two = pd.concat(
        [
            canonical_one,
            pd.DataFrame(
                {
                    "time": [next_time],
                    "open": np.asarray([2404.0], dtype=np.float64),
                    "signal": np.asarray([5.0], dtype=np.float32),
                }
            ),
        ],
        ignore_index=True,
    )
    base_row = pd.DataFrame(
        {
            name: np.asarray([2404.0 + offset], dtype=np.float64)
            for offset, name in enumerate(incremental.RAW_BASE28_COLUMNS)
        },
        index=pd.DatetimeIndex([next_time], name="time"),
    )
    base28_two = pd.concat([base28_one, base_row])
    _publish_live_tail_fixture_pair(
        paths,
        canonical=canonical_two,
        base28=base28_two,
        created_utc="2026-07-16T12:25:00Z",
    )
    publication_one_sha = _sha256(publication_one_path)
    publication_two_path, publication_two = (
        publish_live_tail_publication_event(
            event_root=publication_root,
            pair_manifest_path=pair_manifest,
            generation_root=generation_root,
            previous_publication_json=publication_one_path,
            previous_publication_sha256=publication_one_sha,
            created_utc="2026-07-16T12:25:30Z",
        )
    )
    publication_two_sha = _sha256(publication_two_path)
    admission_path, admission = publish_live_tail_admission_event(
        event_root=admission_root,
        parent_publication_json=publication_one_path,
        parent_publication_sha256=publication_one_sha,
        child_publication_json=publication_two_path,
        child_publication_sha256=publication_two_sha,
        pair_manifest_path=pair_manifest,
        generation_root=generation_root,
        created_utc="2026-07-16T12:25:31Z",
    )
    assert publication_one["decision"] == "PASS"
    assert publication_two["decision"] == "PASS"
    assert admission["decision"] == "PASS"
    return {
        "paths": paths,
        "pair_manifest": pair_manifest,
        "generation_root": generation_root,
        "publication_root": publication_root,
        "admission_path": admission_path,
        "publication_two_path": publication_two_path,
        "publication_two_sha": publication_two_sha,
    }


def _disable_augmenters(monkeypatch: pytest.MonkeyPatch) -> None:
    def _identity(
        self: PrebuiltStateLoader,
        cv3: pd.DataFrame | None = None,
    ) -> pd.DataFrame | None:
        return self._cv3 if cv3 is None else cv3

    for name in (
        "_augment_cv3_with_volume_features",
        "_augment_cv3_with_v4_mtf_scalars",
        "_augment_cv3_with_group_a",
        "_augment_cv3_with_v1_legacy",
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


def test_pair_pointer_is_exact_copy_of_immutable_generation_manifest(
    tmp_path: Path,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    pointer = Path(paths["pair_manifest"])
    binding = read_prebuilt_pair_manifest(
        pointer,
        generation_root=Path(paths["generation_root"]),
    )

    assert binding.generation_manifest_path is not None
    assert binding.generation_manifest_path.read_bytes() == pointer.read_bytes()


@pytest.mark.parametrize("mutation", ["missing", "different_bytes"])
def test_v3_pair_rejects_missing_or_divergent_generation_manifest(
    tmp_path: Path,
    mutation: str,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    pointer = Path(paths["pair_manifest"])
    generation_root = Path(paths["generation_root"])
    binding = read_prebuilt_pair_manifest(
        pointer,
        generation_root=generation_root,
    )
    generation_manifest = binding.generation_manifest_path
    assert generation_manifest is not None
    if mutation == "missing":
        generation_manifest.unlink()
        expected = "PREBUILT_PAIR_GENERATION_MANIFEST_MISSING"
    else:
        payload = json.loads(generation_manifest.read_text(encoding="utf-8"))
        payload["created_utc"] = "2026-07-23T00:00:01Z"
        generation_manifest.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        expected = "PREBUILT_PAIR_POINTER_GENERATION_MANIFEST_MISMATCH"

    with pytest.raises(PrebuiltIdentityError, match=expected):
        read_prebuilt_pair_manifest(
            pointer,
            generation_root=generation_root,
        )


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
    canonical, base28 = _successor_frames(signal_value=13.0)
    _publish_next_pair(paths, canonical=canonical, base28=base28)
    manifest_path = Path(paths["pair_manifest"])
    mixed = json.loads(manifest_path.read_text(encoding="utf-8"))
    mixed["artifacts"]["base28"] = old_manifest["artifacts"]["base28"]
    manifest_path.write_text(json.dumps(mixed), encoding="utf-8")

    with pytest.raises(
        PrebuiltIdentityError,
        match="PREBUILT_PAIR_GENERATION_ID_CONTENT_MISMATCH",
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


def test_publisher_rejects_exact_schema_drift_before_pointer_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    _disable_augmenters(monkeypatch)
    current = read_prebuilt_pair_manifest(
        Path(paths["pair_manifest"]),
        generation_root=Path(paths["generation_root"]),
    )
    canonical, base28 = _successor_frames(signal_dtype=np.dtype(np.float64))
    with pytest.raises(
        RuntimeError,
        match="CANONICAL_V3_SUCCESSOR_ARROW_SCHEMA_MISMATCH",
    ):
        _publish_next_pair(paths, canonical=canonical, base28=base28)
    admitted = read_prebuilt_pair_manifest(
        Path(paths["pair_manifest"]),
        generation_root=Path(paths["generation_root"]),
    )
    assert admitted.pair_generation_id == current.pair_generation_id


def test_async_identity_failure_is_latched_instead_of_serving_old_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    _disable_augmenters(monkeypatch)
    loader = _loader(paths)
    loader.load()
    canonical, base28 = _successor_frames(signal_value=4.0)
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
    canonical, base28 = _successor_frames(signal_value=8.0)
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
                lineage_contract=_frame_lineage(
                    canonical,
                    base28,
                    parent_pair_generation_id=current.pair_generation_id,
                    parent_pair_manifest_sha256=current.manifest_sha256,
                ),
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


@pytest.mark.parametrize("offered_parent", [None, "f" * 64])
def test_successor_publication_rejects_null_or_wrong_parent_without_pointer_move(
    tmp_path: Path,
    offered_parent: str | None,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    manifest_path = Path(paths["pair_manifest"])
    generation_root = Path(paths["generation_root"])
    previous_bytes = manifest_path.read_bytes()
    current = read_prebuilt_pair_manifest(
        manifest_path,
        generation_root=generation_root,
    )
    canonical, base28 = _successor_frames()
    staging_dir = incremental._candidate_staging_path(generation_root)
    _write_staged_pair(staging_dir, canonical, base28)
    try:
        with pytest.raises(
            PrebuiltIdentityError,
            match="PREBUILT_PAIR_PARENT_MISMATCH",
        ):
            incremental._publish_prebuilt_pair_generation(
                staging_dir,
                pair_manifest_path=manifest_path,
                generation_root=generation_root,
                expected_pair_generation_id=current.pair_generation_id,
                expected_manifest_sha256=current.manifest_sha256,
                lineage_contract=_frame_lineage(
                    canonical,
                    base28,
                    parent_pair_generation_id=offered_parent,
                    parent_pair_manifest_sha256=current.manifest_sha256
                    if offered_parent is not None
                    else None,
                ),
            )
    finally:
        incremental._discard_pair_staging_dir(
            staging_dir,
            generation_root=generation_root,
        )
    assert manifest_path.read_bytes() == previous_bytes


def test_successor_publication_requires_both_cas_identities(
    tmp_path: Path,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    manifest_path = Path(paths["pair_manifest"])
    generation_root = Path(paths["generation_root"])
    current = read_prebuilt_pair_manifest(
        manifest_path,
        generation_root=generation_root,
    )
    canonical, base28 = _successor_frames()
    staging_dir = incremental._candidate_staging_path(generation_root)
    _write_staged_pair(staging_dir, canonical, base28)
    try:
        with pytest.raises(RuntimeError, match="both expected pointer identities"):
            incremental._publish_prebuilt_pair_generation(
                staging_dir,
                pair_manifest_path=manifest_path,
                generation_root=generation_root,
                expected_pair_generation_id=current.pair_generation_id,
                expected_manifest_sha256=None,
                lineage_contract=_frame_lineage(
                    canonical,
                    base28,
                    parent_pair_generation_id=current.pair_generation_id,
                    parent_pair_manifest_sha256=current.manifest_sha256,
                ),
            )
    finally:
        incremental._discard_pair_staging_dir(
            staging_dir,
            generation_root=generation_root,
        )


@pytest.mark.parametrize("mutation", ["nonadvancing", "prefix"])
def test_successor_publication_rejects_nonadvance_or_rewritten_history(
    tmp_path: Path,
    mutation: str,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    manifest_path = Path(paths["pair_manifest"])
    generation_root = Path(paths["generation_root"])
    previous_bytes = manifest_path.read_bytes()
    if mutation == "nonadvancing":
        canonical, base28 = _frames()
        expected = "CANONICAL_V3_PREBUILT_SUCCESSOR_NOT_STRICTLY_ADVANCING"
    else:
        canonical, base28 = _successor_frames()
        canonical.loc[0, "signal"] = np.float32(99.0)
        expected = "CANONICAL_V3_PREBUILT_SUCCESSOR_PREFIX_MISMATCH"
    current = read_prebuilt_pair_manifest(
        manifest_path,
        generation_root=generation_root,
    )
    staging_dir = incremental._candidate_staging_path(generation_root)
    _write_staged_pair(staging_dir, canonical, base28)
    try:
        with pytest.raises(PrebuiltIdentityError, match=expected):
            incremental._publish_prebuilt_pair_generation(
                staging_dir,
                pair_manifest_path=manifest_path,
                generation_root=generation_root,
                expected_pair_generation_id=current.pair_generation_id,
                expected_manifest_sha256=current.manifest_sha256,
                lineage_contract=_frame_lineage(
                    canonical,
                    base28,
                    parent_pair_generation_id=current.pair_generation_id,
                    parent_pair_manifest_sha256=current.manifest_sha256,
                ),
            )
    finally:
        incremental._discard_pair_staging_dir(
            staging_dir,
            generation_root=generation_root,
        )
    assert manifest_path.read_bytes() == previous_bytes


def test_successor_prefix_comparison_is_ieee_bit_exact() -> None:
    index = pd.DatetimeIndex(["2026-07-16T12:00:00Z"], name="time")
    current = pd.DataFrame(
        {"value": np.asarray([0.0], dtype=np.float64)},
        index=index,
    )
    successor = pd.DataFrame(
        {"value": np.asarray([-0.0, 1.0], dtype=np.float64)},
        index=pd.DatetimeIndex(
            ["2026-07-16T12:00:00Z", "2026-07-16T12:05:00Z"],
            name="time",
        ),
    )
    with pytest.raises(
        PrebuiltIdentityError,
        match="PREBUILT_SUCCESSOR_PREFIX_MISMATCH",
    ):
        require_prebuilt_successor_frame(
            current,
            successor,
            label="CANONICAL_V3",
        )


def test_native_m1_m5_proof_accepts_source_exact_sparse_bucket_without_hour_rule() -> None:
    times = pd.DatetimeIndex(
        [
            "2026-01-05T23:00:00Z",
            "2026-01-05T23:02:00Z",
            "2026-01-05T23:05:00Z",
        ]
    )
    native_m1 = pd.DataFrame(
        {
            "time": times,
            **{
                name: np.asarray(
                    [2400.0 + offset + row for row in range(len(times))],
                    dtype=np.float64,
                )
                for offset, name in enumerate(incremental.RAW_BASE28_COLUMNS)
            },
        }
    )
    # The 23:00 bucket is sparse but separately sourced M5 agrees exactly.
    # The 23:05 M1 row is a forming tail after the last completed M5.
    native_m5 = (
        incremental._aggregate_native_m1_to_m5(native_m1)
        .iloc[:1]
        .reset_index()
    )

    incremental._require_native_m1_m5_aggregation_identity(
        native_m1,
        native_m5,
    )

    corrupted = native_m5.copy()
    corrupted.loc[0, "close"] += 0.25
    with pytest.raises(
        RuntimeError,
        match="PAIR_NATIVE_M1_M5_VALUE_IDENTITY_MISMATCH",
    ):
        incremental._require_native_m1_m5_aggregation_identity(
            native_m1,
            corrupted,
        )


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
    native_m5 = (
        incremental._aggregate_native_m1_to_m5(native_m1)
        .reset_index()
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


def test_full_history_successor_revalidates_native_prefix_and_publishes_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    pair_manifest = Path(paths["pair_manifest"])
    generation_root = Path(paths["generation_root"])
    parent = read_prebuilt_pair_manifest(
        pair_manifest,
        generation_root=generation_root,
    )
    parent_sources = parent.lineage["native_sources"]
    child_roots = {
        "M1": tmp_path / "native-m1-child",
        "M5": tmp_path / "native-m5-child",
    }
    child_descriptors: dict[str, dict[str, object]] = {}
    for timeframe in ("M1", "M5"):
        label = timeframe.lower()
        descriptor = dict(parent_sources[label])
        descriptor.update(
            {
                "schema_version": (
                    incremental.CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA
                ),
                "publication_mode": (
                    incremental.CANONICAL_NATIVE_SUCCESSOR_MODE
                ),
                "parent_source": {
                    "root": parent_sources[label]["root"],
                    "manifest_path": parent_sources[label]["manifest_path"],
                    "manifest_sha256": parent_sources[label][
                        "manifest_sha256"
                    ],
                },
                "root": str(child_roots[timeframe]),
                "manifest_path": str(
                    child_roots[timeframe] / "MANIFEST.json"
                ),
                "manifest_sha256": ("a" if timeframe == "M1" else "b") * 64,
                "requested_end_utc_exclusive": (
                    "2026-07-18T00:00:00+00:00"
                ),
                "row_count": 4,
                "time_max_utc": "2026-07-16T12:15:00+00:00",
                "canonical_rows_sha256": (
                    "c" if timeframe == "M1" else "d"
                )
                * 64,
                "manifest_payload_sha256": (
                    "e" if timeframe == "M1" else "f"
                )
                * 64,
                "year_sha256": {
                    "year=2026": (
                        "a" if timeframe == "M1" else "b"
                    )
                    * 64
                },
                "year_rows": {"year=2026": 4},
            }
        )
        child_descriptors[timeframe] = descriptor

    def native_frame(rows: int) -> pd.DataFrame:
        times = pd.date_range(
            "2026-07-16T12:00:00Z",
            periods=rows,
            freq="5min",
        )
        return pd.DataFrame(
            {
                "time": times,
                **{
                    name: np.asarray(
                        [2400.0 + offset + row for row in range(rows)],
                        dtype=np.float64,
                    )
                    for offset, name in enumerate(
                        incremental.RAW_BASE28_COLUMNS
                    )
                },
            }
        )

    parent_frames = {"M1": native_frame(3), "M5": native_frame(3)}
    child_frames = {"M1": native_frame(4), "M5": native_frame(4)}

    def descriptor_for(root: Path, *, timeframe: str) -> dict[str, object]:
        if Path(root) == child_roots[timeframe]:
            return dict(child_descriptors[timeframe])
        return dict(parent_sources[timeframe.lower()])

    def frame_for(
        descriptor: dict[str, object],
        *,
        timeframe: str,
    ) -> pd.DataFrame:
        if descriptor["root"] == str(child_roots[timeframe]):
            return child_frames[timeframe].copy()
        return parent_frames[timeframe].copy()

    successor_canonical, successor_base28 = _successor_frames()
    successor_canonical = successor_canonical.set_index("time")
    monkeypatch.setattr(
        incremental,
        "_clean_repository_commit",
        lambda _root: "7" * 40,
    )
    monkeypatch.setattr(
        incremental,
        "canonical_xau_source_descriptor_v1",
        descriptor_for,
    )
    monkeypatch.setattr(incremental, "_load_native_source_frame", frame_for)
    monkeypatch.setattr(
        incremental,
        "_native_bundle_cas_snapshot",
        lambda descriptor, *, timeframe: (
            str(descriptor["manifest_sha256"]),
            (),
            (),
        ),
    )
    monkeypatch.setattr(
        incremental,
        "_derive_pair_frames",
        lambda **_kwargs: (
            successor_canonical.copy(),
            successor_base28.copy(),
        ),
    )
    monkeypatch.setattr(
        incremental,
        "_pair_producer_source_inventory",
        lambda _root: [{"path": "fixture.py", "sha256": "1" * 64}],
    )

    child_id = incremental.publish_prebuilt_pair_successor(
        native_m1_root=child_roots["M1"],
        native_m5_root=child_roots["M5"],
        vedtak_id="FIXTURE_VEDTAK_001",
        checkpoint_dir=tmp_path / "checkpoints",
        expected_pair_generation_id=parent.pair_generation_id,
        expected_manifest_sha256=parent.manifest_sha256,
        pair_manifest_path=pair_manifest,
        generation_root=generation_root,
        repo_root=REPO,
    )
    admitted = read_prebuilt_pair_manifest(
        pair_manifest,
        generation_root=generation_root,
    )

    assert admitted.pair_generation_id == child_id
    assert (
        admitted.lineage["parent_pair_generation_id"]
        == parent.pair_generation_id
    )
    assert (
        admitted.lineage["parent_pair_manifest_sha256"]
        == parent.manifest_sha256
    )
    assert admitted.lineage["native_sources"]["m1"]["root"] == str(
        child_roots["M1"]
    )
    assert admitted.canonical_v3.rows == 4
    assert admitted.base28.rows == 4


@pytest.mark.parametrize(
    ("m1_schema", "m5_schema", "rejected_timeframe"),
    [
        ("xau_canonical_native_source_v3", "xau_canonical_native_source_v3", "M1"),
        (
            incremental.CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
            "xau_canonical_native_source_v3",
            "M5",
        ),
    ],
)
def test_pair_successor_rejects_v3_and_mixed_native_descriptors(
    m1_schema: str,
    m5_schema: str,
    rejected_timeframe: str,
) -> None:
    parent_sources = _lineage()["native_sources"]
    assert isinstance(parent_sources, dict)

    for label, timeframe, schema in (
        ("m1", "M1", m1_schema),
        ("m5", "M5", m5_schema),
    ):
        parent = parent_sources[label]
        assert isinstance(parent, dict)
        successor = {
            "schema_version": schema,
            "publication_mode": incremental.CANONICAL_NATIVE_SUCCESSOR_MODE,
            "parent_source": {
                "root": parent["root"],
                "manifest_path": parent["manifest_path"],
                "manifest_sha256": parent["manifest_sha256"],
            },
        }
        if timeframe == rejected_timeframe:
            with pytest.raises(
                RuntimeError,
                match=f"NATIVE_{timeframe}_SCHEMA_OR_MODE_INVALID",
            ):
                incremental._require_native_successor_descriptor_binding(
                    parent_descriptor=parent,
                    successor_descriptor=successor,
                    timeframe=timeframe,
                )
            break
        incremental._require_native_successor_descriptor_binding(
            parent_descriptor=parent,
            successor_descriptor=successor,
            timeframe=timeframe,
        )


def test_pair_successor_rejects_parent_source_not_bound_to_active_pair() -> None:
    parent = _lineage()["native_sources"]["m1"]
    successor = {
        "schema_version": incremental.CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
        "publication_mode": incremental.CANONICAL_NATIVE_SUCCESSOR_MODE,
        "parent_source": {
            "root": parent["root"],
            "manifest_path": parent["manifest_path"],
            "manifest_sha256": "f" * 64,
        },
    }

    with pytest.raises(RuntimeError, match="PARENT_SOURCE_MANIFEST_SHA256_MISMATCH"):
        incremental._require_native_successor_descriptor_binding(
            parent_descriptor=parent,
            successor_descriptor=successor,
            timeframe="M1",
        )


def test_runtime_blocks_newer_publication_block_after_valid_admission(
    tmp_path: Path,
) -> None:
    fixture = _admitted_live_tail_fixture(tmp_path)
    blocked_path, blocked = publish_live_tail_publication_event(
        event_root=Path(fixture["publication_root"]),
        pair_manifest_path=Path(fixture["pair_manifest"]),
        generation_root=Path(fixture["generation_root"]),
        created_utc="2026-07-16T12:31:31Z",
    )
    assert blocked["decision"] == "BLOCK"
    assert blocked_path.exists()

    with pytest.raises(
        LiveTailAuthorityError,
        match="LIVE_TAIL_PUBLICATION_NOT_PASS",
    ):
        require_newest_live_tail_runtime_admission(
            Path(fixture["admission_path"]),
            launch_admission_sha256=_sha256(
                Path(fixture["admission_path"])
            ),
            now_utc="2026-07-16T12:25:32Z",
        )


def test_runtime_blocks_newer_pass_publication_without_admission(
    tmp_path: Path,
) -> None:
    fixture = _admitted_live_tail_fixture(tmp_path)
    orphan_path, orphan = publish_live_tail_publication_event(
        event_root=Path(fixture["publication_root"]),
        pair_manifest_path=Path(fixture["pair_manifest"]),
        generation_root=Path(fixture["generation_root"]),
        created_utc="2026-07-16T12:25:40Z",
    )
    assert orphan["decision"] == "PASS"
    assert orphan_path != Path(fixture["publication_two_path"])

    with pytest.raises(
        LiveTailAuthorityError,
        match="LIVE_TAIL_RUNTIME_NEWEST_PUBLICATION_NOT_ADMITTED",
    ):
        require_newest_live_tail_runtime_admission(
            Path(fixture["admission_path"]),
            launch_admission_sha256=_sha256(
                Path(fixture["admission_path"])
            ),
            now_utc="2026-07-16T12:25:41Z",
        )


def test_runtime_rejects_replayed_admission_not_monotonic_from_prior_pass(
    tmp_path: Path,
) -> None:
    fixture = _admitted_live_tail_fixture(tmp_path)
    first_admission = Path(fixture["admission_path"])
    first_payload = json.loads(first_admission.read_text())
    replay_path, replay = publish_live_tail_admission_event(
        event_root=first_admission.parent,
        parent_publication_json=Path(
            first_payload["parent_publication"]["path"]
        ),
        parent_publication_sha256=first_payload[
            "parent_publication"
        ]["sha256"],
        child_publication_json=Path(
            first_payload["child_publication"]["path"]
        ),
        child_publication_sha256=first_payload[
            "child_publication"
        ]["sha256"],
        pair_manifest_path=Path(fixture["pair_manifest"]),
        generation_root=Path(fixture["generation_root"]),
        created_utc="2026-07-16T12:25:40Z",
    )
    assert replay["decision"] == "PASS"
    assert replay_path != first_admission

    with pytest.raises(
        LiveTailAuthorityError,
        match="ADMISSION_NOT_MONOTONIC_FROM_PRIOR_PASS",
    ):
        require_newest_live_tail_runtime_admission(
            first_admission,
            launch_admission_sha256=_sha256(first_admission),
            now_utc="2026-07-16T12:25:41Z",
        )


def test_live_tail_admission_requires_two_current_fresh_pair_successors(
    tmp_path: Path,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    pair_manifest = Path(paths["pair_manifest"])
    generation_root = Path(paths["generation_root"])
    publication_root = tmp_path / "live-tail-publications"
    admission_root = tmp_path / "live-tail-admissions"

    canonical_one, base28_one = _successor_frames()
    _publish_live_tail_fixture_pair(
        paths,
        canonical=canonical_one,
        base28=base28_one,
        created_utc="2026-07-16T12:20:00Z",
    )
    publication_one_path, publication_one = (
        publish_live_tail_publication_event(
            event_root=publication_root,
            pair_manifest_path=pair_manifest,
            generation_root=generation_root,
            created_utc="2026-07-16T12:20:30Z",
        )
    )
    assert publication_one["decision"] == "PASS"

    next_time = pd.Timestamp("2026-07-16T12:20:00Z")
    canonical_two = pd.concat(
        [
            canonical_one,
            pd.DataFrame(
                {
                    "time": [next_time],
                    "open": np.asarray([2404.0], dtype=np.float64),
                    "signal": np.asarray([5.0], dtype=np.float32),
                }
            ),
        ],
        ignore_index=True,
    )
    base_row = pd.DataFrame(
        {
            name: np.asarray([2404.0 + offset], dtype=np.float64)
            for offset, name in enumerate(incremental.RAW_BASE28_COLUMNS)
        },
        index=pd.DatetimeIndex([next_time], name="time"),
    )
    base28_two = pd.concat([base28_one, base_row])
    _publish_live_tail_fixture_pair(
        paths,
        canonical=canonical_two,
        base28=base28_two,
        created_utc="2026-07-16T12:25:00Z",
    )
    publication_one_sha = _sha256(publication_one_path)
    publication_two_path, publication_two = (
        publish_live_tail_publication_event(
            event_root=publication_root,
            pair_manifest_path=pair_manifest,
            generation_root=generation_root,
            previous_publication_json=publication_one_path,
            previous_publication_sha256=publication_one_sha,
            created_utc="2026-07-16T12:25:30Z",
        )
    )
    assert publication_two["decision"] == "PASS"
    publication_two_sha = _sha256(publication_two_path)

    admission_path, admission = publish_live_tail_admission_event(
        event_root=admission_root,
        parent_publication_json=publication_one_path,
        parent_publication_sha256=publication_one_sha,
        child_publication_json=publication_two_path,
        child_publication_sha256=publication_two_sha,
        pair_manifest_path=pair_manifest,
        generation_root=generation_root,
        created_utc="2026-07-16T12:25:31Z",
    )
    assert admission["decision"] == "PASS"
    admitted = require_live_tail_admission_event(
        admission_path,
        expected_sha256=_sha256(admission_path),
        pair_manifest_path=pair_manifest,
        generation_root=generation_root,
        now_utc="2026-07-16T12:25:32Z",
    )
    assert (
        admitted["anchor_pair"]["pair_generation_id"]
        == publication_two["pair"]["pair_generation_id"]
    )
    runtime = require_newest_live_tail_runtime_admission(
        admission_path,
        launch_admission_sha256=_sha256(admission_path),
        now_utc="2026-07-16T12:25:32Z",
    )
    assert (
        runtime["current_admission"]["pair_generation_id"]
        == publication_two["pair"]["pair_generation_id"]
    )
    exact_runtime = require_newest_live_tail_runtime_admission(
        admission_path,
        launch_admission_sha256=_sha256(admission_path),
        expected_pair_generation_id=publication_two["pair"][
            "pair_generation_id"
        ],
        expected_generation_manifest_sha256=publication_two["pair"][
            "generation_manifest"
        ]["sha256"],
        now_utc="2026-07-16T12:25:32Z",
    )
    assert (
        exact_runtime["current_admission"]["pair_generation_id"]
        == publication_two["pair"]["pair_generation_id"]
    )
    with pytest.raises(
        LiveTailAuthorityError,
        match="LIVE_TAIL_RUNTIME_DECISION_PAIR_MISMATCH",
    ):
        require_newest_live_tail_runtime_admission(
            admission_path,
            launch_admission_sha256=_sha256(admission_path),
            expected_pair_generation_id=publication_one["pair"][
                "pair_generation_id"
            ],
            expected_generation_manifest_sha256=publication_one["pair"][
                "generation_manifest"
            ]["sha256"],
            now_utc="2026-07-16T12:25:32Z",
        )

    with pytest.raises(
        LiveTailAuthorityError,
        match="LIVE_TAIL_ADMISSION_EXPIRED",
    ):
        require_live_tail_admission_event(
            admission_path,
            expected_sha256=_sha256(admission_path),
            pair_manifest_path=pair_manifest,
            generation_root=generation_root,
            now_utc="2026-07-16T12:31:31Z",
        )

    blocked_path, blocked = publish_live_tail_admission_event(
        event_root=admission_root,
        parent_publication_json=publication_one_path,
        parent_publication_sha256=publication_one_sha,
        child_publication_json=publication_two_path,
        child_publication_sha256=publication_two_sha,
        pair_manifest_path=pair_manifest,
        generation_root=generation_root,
        created_utc="2026-07-16T12:31:31Z",
    )
    assert blocked["decision"] == "BLOCK"
    assert blocked["failures"] == ["child_publication_is_stale"]
    assert (
        require_live_tail_admission_event(
            blocked_path,
            expected_sha256=_sha256(blocked_path),
            pair_manifest_path=pair_manifest,
            generation_root=generation_root,
            now_utc="2026-07-16T12:31:31Z",
            require_pass=False,
        )["decision"]
        == "BLOCK"
    )
    with pytest.raises(
        LiveTailAuthorityError,
        match="LIVE_TAIL_ADMISSION_NOT_PASS",
    ):
        require_live_tail_admission_event(
            blocked_path,
            expected_sha256=_sha256(blocked_path),
            pair_manifest_path=pair_manifest,
            generation_root=generation_root,
            now_utc="2026-07-16T12:31:31Z",
        )
    with pytest.raises(
        LiveTailAuthorityError,
        match="LIVE_TAIL_ADMISSION_NOT_PASS",
    ):
        require_newest_live_tail_runtime_admission(
            admission_path,
            launch_admission_sha256=_sha256(admission_path),
            now_utc="2026-07-16T12:31:31Z",
        )

    third_time = pd.Timestamp("2026-07-16T12:25:00Z")
    canonical_three = pd.concat(
        [
            canonical_two,
            pd.DataFrame(
                {
                    "time": [third_time],
                    "open": np.asarray([2405.0], dtype=np.float64),
                    "signal": np.asarray([6.0], dtype=np.float32),
                }
            ),
        ],
        ignore_index=True,
    )
    third_base_row = pd.DataFrame(
        {
            name: np.asarray([2405.0 + offset], dtype=np.float64)
            for offset, name in enumerate(incremental.RAW_BASE28_COLUMNS)
        },
        index=pd.DatetimeIndex([third_time], name="time"),
    )
    _publish_live_tail_fixture_pair(
        paths,
        canonical=canonical_three,
        base28=pd.concat([base28_two, third_base_row]),
        created_utc="2026-07-16T12:26:00Z",
    )
    with pytest.raises(
        LiveTailAuthorityError,
        match="LIVE_TAIL_ADMISSION_CURRENT_PAIR_MISMATCH",
    ):
        require_live_tail_admission_event(
            admission_path,
            expected_sha256=_sha256(admission_path),
            pair_manifest_path=pair_manifest,
            generation_root=generation_root,
            now_utc="2026-07-16T12:26:01Z",
        )


def test_live_tail_publication_writes_block_evidence_when_pair_is_stale(
    tmp_path: Path,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    canonical, base28 = _successor_frames()
    _publish_live_tail_fixture_pair(
        paths,
        canonical=canonical,
        base28=base28,
        created_utc="2026-07-16T12:20:00Z",
    )

    event_path, event = publish_live_tail_publication_event(
        event_root=tmp_path / "live-tail-publications",
        pair_manifest_path=Path(paths["pair_manifest"]),
        generation_root=Path(paths["generation_root"]),
        created_utc="2026-07-16T12:23:00Z",
    )

    assert event["decision"] == "BLOCK"
    assert event["failures"] == [
        "pair_publication_latency_exceeds_contract"
    ]
    assert event_path.is_file()
    with pytest.raises(
        LiveTailAuthorityError,
        match="LIVE_TAIL_PUBLICATION_NOT_PASS",
    ):
        require_live_tail_publication_event(
            event_path,
            expected_sha256=_sha256(event_path),
            require_pass=True,
        )


def test_canonical_incremental_owner_is_offline_only() -> None:
    source = (REPO / "gx1/execution/v12_canonical_incremental.py").read_text(
        encoding="utf-8"
    )

    for forbidden in (
        "live-tail-admission",
        "live_tail_publication",
        "live-tail-publication",
        "live_tail_admission",
        "live-tail-admission",
        "live_tail_publication_v1",
        "publish_live_tail_",
    ):
        assert forbidden not in source
    assert tuple(
        inspect.signature(incremental.publish_prebuilt_pair_successor).parameters
    ) == (
        "native_m1_root",
        "native_m5_root",
        "vedtak_id",
        "checkpoint_dir",
        "expected_pair_generation_id",
        "expected_manifest_sha256",
        "pair_manifest_path",
        "generation_root",
        "repo_root",
        "workers",
    )
    assert tuple(
        inspect.signature(incremental._publish_prebuilt_pair_generation).parameters
    ) == (
        "staging_dir",
        "pair_manifest_path",
        "generation_root",
        "expected_pair_generation_id",
        "expected_manifest_sha256",
        "lineage_contract",
        "created_utc",
    )
    assert "bootstrap" in source
    assert "successor" in source


def test_live_tail_publication_rejects_generation_manifest_as_pointer(
    tmp_path: Path,
) -> None:
    paths = _prebuilt_fixture(tmp_path)
    generation_root = Path(paths["generation_root"])
    current = read_prebuilt_pair_manifest(
        Path(paths["pair_manifest"]),
        generation_root=generation_root,
    )

    with pytest.raises(
        LiveTailAuthorityError,
        match="LIVE_TAIL_PAIR_POINTER_PATH_INVALID",
    ):
        publish_live_tail_publication_event(
            event_root=tmp_path / "publications",
            pair_manifest_path=current.generation_manifest_path,
            generation_root=generation_root,
            created_utc="2026-07-16T12:15:30Z",
        )


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


# ── Async refresh over a newly published pair generation ─────────────


def _atr_spread_frames(
    *,
    atr_offset: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical, base28 = _frames()
    canonical["atr_bps"] = np.asarray(
        [10.5 + atr_offset, 11.1 + atr_offset, 12.4 + atr_offset],
        dtype=np.float64,
    )
    canonical["spread_bps"] = np.asarray(
        [1.2 + atr_offset, 1.4 + atr_offset, 1.9 + atr_offset],
        dtype=np.float64,
    )
    return canonical, base28


def _atr_spread_successor_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical, base28 = _atr_spread_frames()
    successor_time = pd.Timestamp("2026-07-16T12:15:00Z")
    canonical = pd.concat(
        [
            canonical.reset_index(drop=True),
            pd.DataFrame(
                {
                    "time": [successor_time],
                    "open": np.asarray([2403.0], dtype=np.float64),
                    "signal": np.asarray([4.0], dtype=np.float32),
                    "atr_bps": np.asarray([17.4], dtype=np.float64),
                    "spread_bps": np.asarray([2.1], dtype=np.float64),
                }
            ),
        ],
        ignore_index=True,
    )
    base_row = pd.DataFrame(
        {
            name: np.asarray([2403.0 + offset], dtype=np.float64)
            for offset, name in enumerate(incremental.RAW_BASE28_COLUMNS)
        },
        index=pd.DatetimeIndex([successor_time], name="time"),
    )
    return canonical, pd.concat([base28, base_row])


def _atr_spread_prebuilt_fixture(tmp_path: Path) -> dict[str, Path | str]:
    generation_root = tmp_path / "generations"
    pair_manifest = tmp_path / "CANONICAL_V3_BASE28_CURRENT_PAIR_MANIFEST.json"
    canonical, base28 = _atr_spread_frames()
    staging_dir = incremental._candidate_staging_path(generation_root)
    _write_staged_pair(staging_dir, canonical, base28)
    generation_id = incremental._publish_prebuilt_pair_generation(
        staging_dir,
        pair_manifest_path=pair_manifest,
        generation_root=generation_root,
        expected_pair_generation_id=None,
        expected_manifest_sha256=None,
        lineage_contract=_frame_lineage(canonical, base28),
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


def test_async_refresh_swaps_in_the_newly_published_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful hot refresh serves the successor generation, not the old one.

    The retired TRAIN-rank attach used to re-derive atr/spread buckets during
    this swap. Those buckets are gone; the raw atr_bps/spread_bps evidence
    stays continuous, so what must still hold is that the refreshed frame is
    the newly published pair and no stale snapshot survives the swap.
    """
    import concurrent.futures

    paths = _atr_spread_prebuilt_fixture(tmp_path)
    _disable_augmenters(monkeypatch)
    loader = _loader(paths)
    loader.load()
    before = loader._cv3
    assert before is not None
    assert float(before["atr_bps"].iloc[-1]) == 12.4

    canonical, base28 = _atr_spread_successor_frames()
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

    # Run the exact refresh orchestration inline: same code path, but the two
    # heavy parallel augmenters are no-op'd (their real outputs are covered by
    # dedicated feature tests) and the pool executes in-process.
    class _InlineFuture:
        def __init__(self, value: pd.DataFrame) -> None:
            self._value = value

        def result(self, timeout: float | None = None) -> pd.DataFrame:
            return self._value

    class _InlinePool:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self) -> "_InlinePool":
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def submit(self, fn, *args):
            return _InlineFuture(fn(*args))

    monkeypatch.setattr(
        concurrent.futures,
        "ProcessPoolExecutor",
        _InlinePool,
    )
    monkeypatch.setattr(
        state_module,
        "_mp_v4_mtf_worker",
        lambda cv3: cv3.iloc[:, :0].copy(),
    )
    monkeypatch.setattr(
        state_module,
        "_mp_group_a_worker",
        lambda cv3: cv3.iloc[:, :0].copy(),
    )

    assert loader.refresh_if_changed() is True
    assert loader._refresh_error is None

    cv3 = loader._cv3
    assert cv3 is not None
    # The refreshed frame is the newly published generation, not the old one.
    assert float(cv3["atr_bps"].iloc[-1]) == 17.4
    assert float(cv3["spread_bps"].iloc[-1]) == 2.1
    assert len(cv3) == len(before) + 1
    # The retired rank-derived bucket columns must not reappear on the swap.
    assert "atr_bucket" not in cv3.columns
    assert "spread_bucket" not in cv3.columns


def test_prebuilt_lineage_requires_rank_fit_fields_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prebuilt pair must attest that no retired rank-fit field is present."""
    paths = _atr_spread_prebuilt_fixture(tmp_path)
    _disable_augmenters(monkeypatch)
    loader = _loader(paths)
    loader.load_frozen_pair()

    assert loader._cv3 is not None
    assert not {"atr_bucket", "spread_bucket", "vol_regime_id"} & set(
        loader._cv3.columns
    )


def test_superseded_derivation_contract_is_readable_only_when_explicitly_relaxed():
    """A parent being replaced must stay readable by the route that replaces it.

    Regression for a real deadlock, 2026-08-18. Commit a94f5c6e advanced the
    canonical builder contract (v2 -> v7): it stopped emitting
    ``smc_premium_state``/``smc_premium_discount``, both since moved into the
    contract's RETIRED tuples. The published 2026-08-09 pair generation had been
    built by the older builder, so ``validate_prebuilt_pair_lineage`` raised
    PREBUILT_PAIR_DERIVATION_CONTRACT_IDENTITY_MISMATCH on it -- and
    ``publish_prebuilt_pair_successor`` had to read that pointer before it could
    replace it. The chain could not advance its own source once the builder
    moved.

    The derivation contract describes how an artifact WAS built. Admission as a
    current source keeps the full check; reading a parent in order to supersede
    it does not. This test pins both halves: strict by default, relaxed only when
    asked.
    """

    from gx1.execution.v12_state_from_prebuilt import (
        PREBUILT_CANONICAL_BUILDER_CONTRACT,
        PrebuiltIdentityError,
        validate_prebuilt_pair_lineage,
    )

    fixture = _lineage()
    # Strict default accepts the current contract...
    validate_prebuilt_pair_lineage(fixture)

    superseded = json.loads(json.dumps(fixture))
    superseded["derivation_contract"]["canonical_builder"] = (
        PREBUILT_CANONICAL_BUILDER_CONTRACT.replace("_v7", "_v2")
    )
    assert (
        superseded["derivation_contract"]["canonical_builder"]
        != PREBUILT_CANONICAL_BUILDER_CONTRACT
    ), "fixture must actually differ, or this test proves nothing"

    # ...and rejects a superseded one. This is the non-vacuity half.
    with pytest.raises(PrebuiltIdentityError):
        validate_prebuilt_pair_lineage(superseded)

    # Relaxed accepts it, and returns the same envelope.
    relaxed = validate_prebuilt_pair_lineage(
        superseded, require_current_derivation_contract=False
    )
    assert relaxed["derivation_contract"]["canonical_builder"] != (
        PREBUILT_CANONICAL_BUILDER_CONTRACT
    )


def test_relaxing_the_derivation_contract_does_not_relax_anything_else():
    """The switch must be narrow: every other lineage rule still fails closed."""

    from gx1.execution.v12_state_from_prebuilt import (
        PrebuiltIdentityError,
        validate_prebuilt_pair_lineage,
    )

    broken = json.loads(json.dumps(_lineage()))
    broken["schema_version"] = "gx1_native_pair_lineage_v_does_not_exist"
    with pytest.raises(PrebuiltIdentityError):
        validate_prebuilt_pair_lineage(
            broken, require_current_derivation_contract=False
        )

    missing_sources = json.loads(json.dumps(_lineage()))
    missing_sources.pop("native_sources")
    with pytest.raises(PrebuiltIdentityError):
        validate_prebuilt_pair_lineage(
            missing_sources, require_current_derivation_contract=False
        )
