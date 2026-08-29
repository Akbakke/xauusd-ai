from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.scripts.run_pre_fulltrain_static_preflight_v1 import (
    PreflightError,
    TEST_BOUNDARY_UTC,
    inspect_bundle_normalization_binding,
    inspect_dataset_mtf_cache_binding,
    inspect_mtf_cache_test_boundary,
    scan_allowed_split,
)


def _write_split(path: Path, times: list[datetime], horizons: list[int]) -> None:
    pq.write_table(
        pa.table(
            {
                "time": pa.array(times, type=pa.timestamp("ns", tz="UTC")),
                "label_horizon_bars": pa.array(horizons, type=pa.int64()),
                "y_example": pa.array([1.0] * len(times), type=pa.float64()),
            }
        ),
        path,
    )


def test_allowed_val_scan_proves_bounds_without_test_access(tmp_path: Path) -> None:
    path = tmp_path / "val.parquet"
    _write_split(
        path,
        [
            datetime(2025, 6, 1, tzinfo=timezone.utc),
            datetime(2026, 6, 30, 23, 55, tzinfo=timezone.utc),
        ],
        [12, 96],
    )
    report = scan_allowed_split(
        path,
        label="val",
        nominal_start_utc="2025-06-01T00:00:00+00:00",
        nominal_end_utc=TEST_BOUNDARY_UTC,
    )
    assert report["rows"] == 2
    assert report["timestamps_at_or_after_test_boundary"] == 0
    assert report["max_label_horizon_bars_observed"] == 96


def test_scan_rejects_test_timestamp_and_test_like_path(tmp_path: Path) -> None:
    forbidden = tmp_path / "val.parquet"
    _write_split(
        forbidden,
        [datetime(2026, 7, 1, tzinfo=timezone.utc)],
        [1],
    )
    with pytest.raises(PreflightError, match="SPLIT_BOUNDARY_INVALID|SPLIT_INTEGRITY_INVALID"):
        scan_allowed_split(
            forbidden,
            label="val",
            nominal_start_utc="2025-06-01T00:00:00+00:00",
            nominal_end_utc=TEST_BOUNDARY_UTC,
        )


def test_mtf_cache_metadata_detects_test_exposure_without_reading_arrays(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "manifest.json"
    cache.write_text(
        """{
          "tfs": {
            "M5": {"last_ts_ns": 1785829800000000000},
            "M15": {"last_ts_ns": 1785828600000000000},
            "H1": {"last_ts_ns": 1785823200000000000},
            "H4": {"last_ts_ns": 1785808800000000000},
            "D1": {"last_ts_ns": 1785708000000000000}
          }
        }""",
        encoding="utf-8",
    )
    report = inspect_mtf_cache_test_boundary(cache)
    assert report["array_bytes_read"] == 0
    assert not report["safe_for_strict_preflight"]
    assert report["test_exposed_timeframes"] == ["D1", "H1", "H4", "M15", "M5"]

    path = tmp_path / "xau_test.parquet"
    _write_split(path, [datetime(2025, 6, 1, tzinfo=timezone.utc)], [1])
    with pytest.raises(PreflightError, match="TEST_PATH_REJECTED"):
        scan_allowed_split(
            path,
            label="val",
            nominal_start_utc="2025-06-01T00:00:00+00:00",
            nominal_end_utc=TEST_BOUNDARY_UTC,
        )


def test_dataset_manifest_must_bind_exact_inspected_mtf_cache(tmp_path: Path) -> None:
    manifest = tmp_path / "train.manifest.json"
    manifest.write_text(
        """{
          "extra": {
            "multi_tf_cache_binding": {
              "manifest_sha256": "manifest-new",
              "cache_identity_sha256": "identity-new",
              "m5_prebuilt_source_sha256": "source-new"
            }
          }
        }""",
        encoding="utf-8",
    )
    report = inspect_dataset_mtf_cache_binding(
        manifest,
        expected_manifest_sha256="manifest-new",
        expected_cache_identity_sha256="identity-new",
        expected_source_sha256="source-new",
    )
    assert report["matches_inspected_cache"]
    assert report["array_bytes_read"] == 0
    assert report["test_accessed"] is False

    manifest.write_text(
        """{
          "extra": {
            "multi_tf_cache_binding": {
              "manifest_sha256": "manifest-old",
              "cache_identity_sha256": "identity-old",
              "m5_prebuilt_source_sha256": "source-old"
            }
          }
        }""",
        encoding="utf-8",
    )
    mismatch = inspect_dataset_mtf_cache_binding(
        manifest,
        expected_manifest_sha256="manifest-new",
        expected_cache_identity_sha256="identity-new",
        expected_source_sha256="source-new",
    )
    assert not mismatch["matches_inspected_cache"]
    assert mismatch["mismatched_fields"] == [
        "manifest_sha256",
        "cache_identity_sha256",
        "m5_prebuilt_source_sha256",
    ]


def test_bundle_normalization_must_bind_exact_rebuilt_train_surface(
    tmp_path: Path,
) -> None:
    train_manifest = tmp_path / "train.manifest.json"
    train_manifest.write_text(
        """{
          "extra": {
            "entry_run_id": "PRETEST_RUN",
            "source_frame": {"parquet_sha256": "source-new"}
          }
        }""",
        encoding="utf-8",
    )
    bundle = {
        "run_lineage": {"dataset_run_id": "PRETEST_RUN"},
        "input_normalization": {
            "lineage": {
                "dataset_run_id": "PRETEST_RUN",
                "train_parquet_sha256": "train-new",
                "train_manifest_sha256": hashlib.sha256(
                    train_manifest.read_bytes()
                ).hexdigest(),
                "m5_prebuilt_sha256": "source-new",
                "mtf_cache_manifest_sha256": "cache-new",
            }
        },
    }
    matched = inspect_bundle_normalization_binding(
        bundle,
        train={"sha256": "train-new"},
        train_manifest=train_manifest,
        train_manifest_payload=json.loads(train_manifest.read_text()),
        mtf_cache_manifest_sha256="cache-new",
    )
    assert matched["matches_exact_train_surface"]

    bundle["input_normalization"]["lineage"]["mtf_cache_manifest_sha256"] = "cache-old"
    mismatch = inspect_bundle_normalization_binding(
        bundle,
        train={"sha256": "train-new"},
        train_manifest=train_manifest,
        train_manifest_payload=json.loads(train_manifest.read_text()),
        mtf_cache_manifest_sha256="cache-new",
    )
    assert not mismatch["matches_exact_train_surface"]
    assert mismatch["mismatched_fields"] == ["mtf_cache_manifest_sha256"]
