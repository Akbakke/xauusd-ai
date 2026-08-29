from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.scripts import materialize_direct_m5_pretest_source_v1 as producer


def _native_fixture(root: Path, timestamps: list[str], *, timeframe: str = "M5") -> None:
    root.mkdir()
    manifest = {
        "instrument": "XAU_USD",
        "timeframe": timeframe,
        "requested_start_utc": "2019-01-01T00:00:00+00:00",
        "requested_end_utc_exclusive": producer.TEST_BOUNDARY_UTC,
        "time_max_utc": "2026-06-30T23:55:00+00:00",
        "row_count": len(timestamps),
        "manifest_payload_sha256": "a" * 64,
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    part = root / "year=2026"
    part.mkdir()
    values = pd.to_datetime(timestamps, utc=True)
    table = pa.table(
        {
            "time": pa.array(values, type=pa.timestamp("ns", tz="UTC")),
            "open": pa.array([1.0] * len(values), type=pa.float64()),
            "high": pa.array([2.0] * len(values), type=pa.float64()),
            "low": pa.array([0.5] * len(values), type=pa.float64()),
            "close": pa.array([1.5] * len(values), type=pa.float64()),
            "volume": pa.array([10] * len(values), type=pa.int64()),
        }
    )
    pq.write_table(table, part / "part-000.parquet")


def _bypass_native_bundle_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        producer,
        "validate_canonical_native_source_bundle",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(producer, "_require_clean_repository", lambda _root: "b" * 40)


def test_materializes_direct_pretest_m5_without_test_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bypass_native_bundle_validation(monkeypatch)
    native = tmp_path / "native"
    _native_fixture(native, ["2026-06-30T23:50:00Z", "2026-06-30T23:55:00Z"])
    output = tmp_path / "published"
    repository = tmp_path / "repo"
    repository.mkdir()

    manifest = producer.materialize_direct_m5_pretest_source(
        native_m5_root=native,
        out_dir=output,
        repo_root=repository,
    )

    assert manifest["test_accessed"] is False
    assert manifest["row_count"] == 2
    assert manifest["time_max_utc"] == "2026-06-30T23:55:00+00:00"
    assert (output / "m5_ohlcv.parquet").is_file()
    assert (output / "manifest.json").is_file()


def test_rejects_native_row_at_test_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bypass_native_bundle_validation(monkeypatch)
    native = tmp_path / "native"
    _native_fixture(native, ["2026-07-01T00:00:00Z"])
    output = tmp_path / "published"
    repository = tmp_path / "repo"
    repository.mkdir()

    with pytest.raises(RuntimeError, match="DIRECT_M5_PRETEST_TIMESTAMP_BOUNDARY_INVALID"):
        producer.materialize_direct_m5_pretest_source(
            native_m5_root=native,
            out_dir=output,
            repo_root=repository,
        )
    assert not output.exists()


def test_materializes_direct_m1_with_its_own_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bypass_native_bundle_validation(monkeypatch)
    native = tmp_path / "native"
    _native_fixture(
        native,
        ["2026-06-30T23:58:00Z", "2026-06-30T23:59:00Z"],
        timeframe="M1",
    )
    output = tmp_path / "published"
    repository = tmp_path / "repo"
    repository.mkdir()

    manifest = producer.materialize_direct_m5_pretest_source(
        native_m5_root=native,
        out_dir=output,
        timeframe="M1",
        repo_root=repository,
    )

    assert manifest["timeframe"] == "M1"
    assert (output / "m1_ohlcv.parquet").is_file()
