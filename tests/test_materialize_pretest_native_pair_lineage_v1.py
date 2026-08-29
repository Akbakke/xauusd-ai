from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_pretest_native_pair_lineage_v1 as producer
from gx1.scripts import build_entry_exit_m1_enriched_frame_v1 as enriched_builder


def _source_manifest(root: Path, *, timeframe: str, test_accessed: bool = False) -> Path:
    parquet = root / f"{timeframe.lower()}_ohlcv.parquet"
    parquet.write_bytes(f"{timeframe}-direct".encode("ascii"))
    native_root = root / f"native_{timeframe.lower()}"
    native_root.mkdir()
    native_manifest = native_root / "MANIFEST.json"
    native_manifest.write_text(
        json.dumps({"timeframe": timeframe, "pretest": True}),
        encoding="utf-8",
    )
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
        "source_native_root": str(native_root),
        "source_native_manifest_path": str(native_manifest),
        "source_native_manifest_sha256": hashlib.sha256(
            native_manifest.read_bytes()
        ).hexdigest(),
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
    assert len(payload["pair_generation_id"]) == 64
    assert set(payload["pair_generation_id"]) <= set("0123456789abcdef")
    assert payload["m1"]["source_manifest_path"] == str(m1)
    assert payload["m5"]["source_manifest_path"] == str(m5)
    assert payload["lineage"]["native_sources"]["m1"]["root"] == str(
        tmp_path / "native_m1"
    )
    assert payload["lineage"]["native_sources"]["m5"]["root"] == str(
        tmp_path / "native_m5"
    )
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


def test_rejects_direct_source_with_unbound_native_manifest(tmp_path: Path) -> None:
    m1 = _source_manifest(tmp_path, timeframe="M1")
    m5 = _source_manifest(tmp_path, timeframe="M5")
    payload = json.loads(m1.read_text(encoding="utf-8"))
    payload["source_native_manifest_sha256"] = "f" * 64
    m1.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="PRETEST_NATIVE_PAIR_SOURCE_BOUNDARY_INVALID"):
        producer.materialize_pretest_native_pair_lineage(
            m1_source_manifest=m1,
            m5_source_manifest=m5,
            output_json=tmp_path / "pair.json",
        )


def test_pair_lineage_is_directly_usable_by_both_enriched_producers(
    tmp_path: Path,
) -> None:
    m1 = _source_manifest(tmp_path, timeframe="M1")
    m5 = _source_manifest(tmp_path, timeframe="M5")
    output = tmp_path / "pair.json"
    payload = producer.materialize_pretest_native_pair_lineage(
        m1_source_manifest=m1,
        m5_source_manifest=m5,
        output_json=output,
    )

    for timeframe in ("M1", "M5"):
        source = payload[timeframe.lower()]["native_source"]
        result = enriched_builder._require_pair_binding(
            pair_manifest_path=output,
            expected_pair_manifest_sha256=hashlib.sha256(
                output.read_bytes()
            ).hexdigest(),
            pair_generation_id=payload["pair_generation_id"],
            source_identity={
                "root": source["root"],
                "manifest_path": source["manifest_path"],
                "manifest_sha256": source["manifest_sha256"],
            },
            native_summary={
                "row_count": source["row_count"],
                "time_min_utc": source["time_min_utc"],
                "time_max_utc": source["time_max_utc"],
            },
            timeframe=timeframe,
        )
        assert result[f"native_{timeframe.lower()}"] == source
