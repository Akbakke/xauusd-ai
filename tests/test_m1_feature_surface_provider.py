from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gx1.contracts.entry_exit_feature_surface_v1 as feature_surface_module
from gx1.contracts.entry_exit_feature_base_v1 import (
    EXIT_FEATURE_SEQUENCE_BARS,
    entry_exit_shared_feature_base_contract,
)
from gx1.contracts.entry_exit_feature_surface_v1 import (
    ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION,
    load_m1_feature_surface,
    load_m1_feature_surface_times,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.execution.v12_m1_feature_surface_provider import (
    M1SharedFeatureSurfaceProvider,
)
from gx1.time.session_detector import (
    m1_decision_availability,
    m5_decision_availability,
)


FEATURE_FIELD_ORDER = [
    f"signal_field_{index}"
    for index in range(MODEL_NATIVE_SIGNAL_DIM)
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _artifact(
    tmp_path: Path,
    *,
    market_gap: bool = False,
    rows: int | None = None,
) -> tuple[Path, Path, str, str]:
    if rows is None:
        rows = EXIT_FEATURE_SEQUENCE_BARS + 1
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    times = pd.date_range(start, periods=rows, freq="min")
    if market_gap:
        times = times.where(
            np.arange(rows) < 240,
            times + pd.Timedelta(days=2),
        )
    signal = np.arange(rows * MODEL_NATIVE_SIGNAL_DIM, dtype=np.float32).reshape(
        rows, MODEL_NATIVE_SIGNAL_DIM
    )
    frame = pd.DataFrame(
        {
            "time": times,
            "signal": [row.tolist() for row in signal],
            "ctx_cont": [
                np.full(MODEL_NATIVE_CTX_CONT_DIM, i, dtype=np.float32).tolist()
                for i in range(rows)
            ],
            "ctx_cat": [
                # session_id is the only categorical owner; width comes from
                # the contract, never from a restated literal.
                [1] * MODEL_NATIVE_CTX_CAT_DIM
                for _ in range(rows)
            ],
        }
    )
    parquet = tmp_path / "m1_feature_base.parquet"
    frame.to_parquet(parquet, index=False)
    dataset_run_id = "RUN_PROVIDER_TEST"
    pair_generation_id = "PAIR_PROVIDER_TEST"
    manifest = {
        "schema_version": ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION,
        "decision": "PASS",
        "shared_feature_base_contract": entry_exit_shared_feature_base_contract(),
        "dataset_run_id": dataset_run_id,
        "pair_generation_id": pair_generation_id,
        "output_parquet": str(parquet.resolve()),
        "output_parquet_sha256": _sha256(parquet),
        "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        "feature_field_order": FEATURE_FIELD_ORDER,
        "feature_field_order_sha256": _canonical_sha256(
            FEATURE_FIELD_ORDER
        ),
    }
    manifest_path = tmp_path / "m1_feature_base.parquet.manifest.json"
    manifest_path.write_text(
        __import__("json").dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    return parquet, manifest_path, dataset_run_id, pair_generation_id


def test_provider_reads_exact_causal_m1_window(tmp_path: Path) -> None:
    parquet, manifest, run_id, pair_id = _artifact(tmp_path)
    provider = M1SharedFeatureSurfaceProvider.from_admitted_artifact(
        parquet_path=parquet,
        manifest_path=manifest,
        dataset_run_id=run_id,
        pair_generation_id=pair_id,
        parquet_sha256=_sha256(parquet),
        manifest_sha256=_sha256(manifest),
        feature_field_order=FEATURE_FIELD_ORDER,
        feature_field_order_sha256=_canonical_sha256(FEATURE_FIELD_ORDER),
    )
    decision_time = pd.Timestamp("2026-01-01T08:00:00Z")
    value = provider(
        decision_time=decision_time,
        prebuilt_snapshot=SimpleNamespace(pair_generation_id=pair_id),
    )
    assert value["sequence_bars"] == EXIT_FEATURE_SEQUENCE_BARS
    assert value["signal"].shape == (
        EXIT_FEATURE_SEQUENCE_BARS,
        MODEL_NATIVE_SIGNAL_DIM,
    )
    assert np.array_equal(value["signal"][-1], value["snap"])
    assert np.array_equal(
        value["ctx_cont"],
        np.full(
            MODEL_NATIVE_CTX_CONT_DIM,
            EXIT_FEATURE_SEQUENCE_BARS,
            dtype=np.float32,
        ),
    )


def test_provider_uses_observed_rows_across_proven_market_gap(
    tmp_path: Path,
) -> None:
    parquet, manifest, run_id, pair_id = _artifact(
        tmp_path,
        market_gap=True,
    )
    provider = M1SharedFeatureSurfaceProvider.from_admitted_artifact(
        parquet_path=parquet,
        manifest_path=manifest,
        dataset_run_id=run_id,
        pair_generation_id=pair_id,
        parquet_sha256=_sha256(parquet),
        manifest_sha256=_sha256(manifest),
        feature_field_order=FEATURE_FIELD_ORDER,
        feature_field_order_sha256=_canonical_sha256(FEATURE_FIELD_ORDER),
    )
    decision_time = pd.read_parquet(parquet, columns=["time"])["time"].iloc[-1]

    value = provider(
        decision_time=pd.Timestamp(decision_time),
        prebuilt_snapshot=SimpleNamespace(pair_generation_id=pair_id),
    )

    assert value["signal"].shape[0] == EXIT_FEATURE_SEQUENCE_BARS
    assert np.array_equal(value["signal"][-1], value["snap"])


def test_provider_rejects_pair_mismatch(tmp_path: Path) -> None:
    parquet, manifest, run_id, pair_id = _artifact(tmp_path)
    provider = M1SharedFeatureSurfaceProvider.from_admitted_artifact(
        parquet_path=parquet,
        manifest_path=manifest,
        dataset_run_id=run_id,
        pair_generation_id=pair_id,
        parquet_sha256=_sha256(parquet),
        manifest_sha256=_sha256(manifest),
        feature_field_order=FEATURE_FIELD_ORDER,
        feature_field_order_sha256=_canonical_sha256(FEATURE_FIELD_ORDER),
    )
    with pytest.raises(RuntimeError, match="PAIR_GENERATION_MISMATCH"):
        provider(
            decision_time=pd.Timestamp("2026-01-01T08:00:00Z"),
            prebuilt_snapshot=SimpleNamespace(pair_generation_id="OTHER_PAIR"),
        )


def test_m1_and_m5_clocks_are_explicit() -> None:
    labels_m1 = pd.date_range("2026-01-01", periods=2, freq="min")
    labels_m5 = pd.date_range("2026-01-01", periods=2, freq="5min")
    assert list(m1_decision_availability(labels_m1)) == [
        pd.Timestamp("2026-01-01T00:01:00Z"),
        pd.Timestamp("2026-01-01T00:02:00Z"),
    ]
    assert list(m5_decision_availability(labels_m5)) == [
        pd.Timestamp("2026-01-01T00:05:00Z"),
        pd.Timestamp("2026-01-01T00:10:00Z"),
    ]


def test_time_only_surface_validation_preserves_exact_clock(tmp_path: Path) -> None:
    parquet, _manifest, _run_id, _pair_id = _artifact(tmp_path)

    times = load_m1_feature_surface_times(
        parquet,
        context="TEST",
    )

    assert len(times) == EXIT_FEATURE_SEQUENCE_BARS + 1
    assert times[0] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert times[-1] == pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(
        minutes=EXIT_FEATURE_SEQUENCE_BARS
    )


def test_full_surface_loader_preserves_fixed_list_values(tmp_path: Path) -> None:
    parquet, _manifest, _run_id, _pair_id = _artifact(tmp_path)

    times, arrays = load_m1_feature_surface(
        parquet,
        context="TEST",
    )

    assert len(times) == EXIT_FEATURE_SEQUENCE_BARS + 1
    assert arrays["signal"].shape == (
        EXIT_FEATURE_SEQUENCE_BARS + 1,
        MODEL_NATIVE_SIGNAL_DIM,
    )
    assert arrays["ctx_cont"].shape == (
        EXIT_FEATURE_SEQUENCE_BARS + 1,
        MODEL_NATIVE_CTX_CONT_DIM,
    )
    assert arrays["ctx_cat"].shape == (
        EXIT_FEATURE_SEQUENCE_BARS + 1,
        MODEL_NATIVE_CTX_CAT_DIM,
    )
    assert arrays["signal"].dtype == np.float32
    assert arrays["ctx_cat"].dtype == np.int64


def test_full_surface_loader_can_use_disk_backed_shared_storage(
    tmp_path: Path,
) -> None:
    parquet, _manifest, _run_id, _pair_id = _artifact(tmp_path)

    _times, arrays = load_m1_feature_surface(
        parquet,
        context="TEST",
        storage_dir=tmp_path / "shared_surface_storage",
    )

    assert all(isinstance(value, np.memmap) for value in arrays.values())
    assert np.array_equal(
        arrays["signal"][-1],
        np.arange(
            EXIT_FEATURE_SEQUENCE_BARS * MODEL_NATIVE_SIGNAL_DIM,
            (EXIT_FEATURE_SEQUENCE_BARS + 1) * MODEL_NATIVE_SIGNAL_DIM,
            dtype=np.float32,
        ),
    )


def test_disk_backed_loader_validates_full_surface_in_bounded_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_rows = 64
    rows = batch_rows * 3 + 7
    parquet, _manifest, _run_id, _pair_id = _artifact(tmp_path, rows=rows)
    monkeypatch.setattr(
        feature_surface_module,
        "_M1_FEATURE_SURFACE_BATCH_ROWS",
        batch_rows,
    )
    monkeypatch.setattr(
        feature_surface_module,
        "_M1_FEATURE_SURFACE_DISK_SYNC_ROWS",
        batch_rows * 2,
    )
    original_isfinite = np.isfinite
    observed_sizes: list[int] = []

    def _tracked_isfinite(values: object, *args: object, **kwargs: object):
        array = np.asarray(values)
        observed_sizes.append(int(array.size))
        return original_isfinite(values, *args, **kwargs)

    monkeypatch.setattr(feature_surface_module.np, "isfinite", _tracked_isfinite)

    _times, arrays = load_m1_feature_surface(
        parquet,
        context="TEST",
        storage_dir=tmp_path / "bounded_shared_surface_storage",
    )

    assert arrays["signal"].shape[0] == rows
    assert observed_sizes
    assert max(observed_sizes) <= batch_rows * MODEL_NATIVE_SIGNAL_DIM
