from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_pretest_native_pair_lineage_v1 as producer


def _source_manifest(root: Path, *, timeframe: str, test_accessed: bool = False) -> Path:
    parquet = root / f"{timeframe.lower()}_ohlcv.parquet"
    parquet.write_bytes(f"{timeframe}-direct".encode("ascii"))
    manifest = {
        "schema_version": "gx1_direct_native_pretest_source_v2",
        "timeframe": timeframe,
        "test_boundary_utc": producer.TEST_BOUNDARY_UTC,
        "test_accessed": test_accessed,
        "source_requested_end_utc_exclusive": producer.TEST_BOUNDARY_UTC,
        "output_parquet": str(parquet),
        "output_parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "row_count": 2,
        "time_min_utc": "2026-06-30T23:50:00+00:00",
        "time_max_utc": "2026-06-30T23:59:00+00:00",
        "manifest_payload_sha256": "a" * 64,
    }
    path = root / f"{timeframe.lower()}_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_binds_exact_direct_pretest_m1_and_m5(tmp_path: Path) -> None:
    m1 = _source_manifest(tmp_path, timeframe="M1")
    m5 = _source_manifest(tmp_path, timeframe="M5")
    output = tmp_path / "pair.json"

    payload = producer.materialize_pretest_native_pair_lineage(
        m1_source_manifest=m1,
        m5_source_manifest=m5,
        output_json=output,
    )

    assert payload["test_accessed"] is False
    assert payload["m1"]["source_manifest_path"] == str(m1)
    assert payload["m5"]["source_manifest_path"] == str(m5)
    assert output.is_file()


def test_rejects_a_source_that_claims_test_access(tmp_path: Path) -> None:
    m1 = _source_manifest(tmp_path, timeframe="M1", test_accessed=True)
    m5 = _source_manifest(tmp_path, timeframe="M5")

    with pytest.raises(RuntimeError, match="PRETEST_NATIVE_PAIR_SOURCE_BOUNDARY_INVALID"):
        producer.materialize_pretest_native_pair_lineage(
            m1_source_manifest=m1,
            m5_source_manifest=m5,
            output_json=tmp_path / "pair.json",
        )
