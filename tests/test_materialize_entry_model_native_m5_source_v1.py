import hashlib
import json

import pytest

from gx1.contracts.xau_tape_provenance_v1 import (
    CANONICAL_NATIVE_SOURCE_SCHEMA,
    CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
)
from gx1.scripts import materialize_entry_model_native_m5_source_v1 as producer
from gx1.scripts.materialize_pretest_native_pair_lineage_v1 import (
    PAIR_LINEAGE_SCHEMA_VERSION,
    TEST_BOUNDARY_UTC,
)


def test_m5_source_accepts_only_authoritative_native_schema_versions() -> None:
    """The frozen pre-TEST V3 source and sealed V4 successor share one lane.

    The M5 source producer still performs the pair/hash/time/Arrow checks.  This
    test pins the narrow schema admission set so an arbitrary old or invented
    native manifest can never bypass those checks.
    """

    assert producer.NATIVE_SOURCE_SCHEMA_VERSIONS == frozenset(
        (
            CANONICAL_NATIVE_SOURCE_SCHEMA,
            CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
        )
    )
    assert "xau_canonical_native_source_v2" not in producer.NATIVE_SOURCE_SCHEMA_VERSIONS
    assert "xau_canonical_native_source_v5" not in producer.NATIVE_SOURCE_SCHEMA_VERSIONS


def _pretest_pair_payload() -> dict[str, object]:
    pair_id = "a" * 64
    native_m1 = {"root": "/native/m1"}
    native_m5 = {"root": "/native/m5"}
    payload: dict[str, object] = {
        "schema_version": PAIR_LINEAGE_SCHEMA_VERSION,
        "pair_generation_id": pair_id,
        "pair_symbol": "XAUUSD",
        "test_boundary_utc": TEST_BOUNDARY_UTC,
        "test_accessed": False,
        "m1": {"native_source": native_m1},
        "m5": {"native_source": native_m5},
        "lineage": {"native_sources": {"m1": native_m1, "m5": native_m5}},
    }
    payload["manifest_payload_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def test_m5_source_requires_sealed_pretest_pair_shape() -> None:
    payload = _pretest_pair_payload()
    bound_m5 = producer._require_pair_manifest_native_sources(
        payload,
        pair_generation_id="a" * 64,
    )
    # The compact pre-TEST leaf carries only source identity.  The producer
    # compares that entire mapping with its locally sealed native manifest;
    # detailed native fields remain authenticated by manifest_sha256.
    assert bound_m5 == {"root": "/native/m5"}

    payload["test_accessed"] = True
    with pytest.raises(
        RuntimeError,
        match="M5_SOURCE_PRETEST_PAIR_MANIFEST_CONTRACT_MISMATCH",
    ):
        producer._require_pair_manifest_native_sources(
            payload,
            pair_generation_id="a" * 64,
        )


@pytest.fixture
def canonical_pair_source(tmp_path):
    """Synthetic two-row binding fixture; it is not market/learning evidence."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from gx1.execution.v12_canonical_incremental import _native_pair_lineage_descriptor

    root = tmp_path / "native"
    year = root / "year=2020"
    year.mkdir(parents=True)
    times = pd.date_range("2020-01-01", periods=2, freq="5min", tz="UTC")
    columns = {
        "time": pa.array(times, type=pa.timestamp("ns", tz="UTC")),
        **{name: pa.array([1., 1.], type=pa.float64()) for name in producer.NATIVE_FLOAT_COLUMNS},
        "volume": pa.array([1, 1], type=pa.int64()),
    }
    part = year / "part-000.parquet"
    pq.write_table(pa.table({name: columns[name] for name in producer.NATIVE_SOURCE_COLUMNS}), part)
    source = {
        "schema_version": CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
        "out_root": str(root), "instrument": "XAU_USD", "timeframe": "M5",
        "bar_duration_seconds": 300, "decision_available_offset_seconds": 300,
        "schema_required_cols": list(producer.NATIVE_SOURCE_COLUMNS), "schema_optional_cols": [],
        "row_count": 2, "time_min_utc": times[0].isoformat(), "time_max_utc": times[1].isoformat(),
        "year_rows": {"year=2020": 2},
        "year_sha256": {"year=2020": hashlib.sha256(part.read_bytes()).hexdigest()},
        "explicit_vedtak_id": "UNIT_BINDING", "source_environment": "practice",
        "source_base_url": "https://api-fxpractice.oanda.com/v3",
        "requested_start_utc": times[0].isoformat(),
        "requested_end_utc_exclusive": (times[-1] + pd.Timedelta(minutes=5)).isoformat(),
        "canonical_rows_sha256": "b" * 64, "producer_git_commit": "c" * 40,
        "producer_source_inventory_sha256": "d" * 64,
    }
    source["manifest_payload_sha256"] = producer._canonical_sha256(source)
    manifest = root / "MANIFEST.json"
    manifest.write_text(json.dumps(source))
    native = _native_pair_lineage_descriptor({
        **source, "root": str(root), "manifest_path": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    })
    pair = {
        "schema_version": producer.PAIR_MANIFEST_SCHEMA_VERSION,
        "pair_generation_id": "a" * 64,
        "lineage": {"schema_version": producer.PAIR_LINEAGE_SCHEMA_VERSION,
                    "native_sources": {"m5": native}},
    }
    pair_path = tmp_path / "PAIR_MANIFEST.json"
    pair_path.write_text(json.dumps(pair))
    return root, pair_path, pair


def test_m5_preflight_accepts_publisher_complete_descriptor(canonical_pair_source):
    root, pair_path, _ = canonical_pair_source
    identity, binding, parts, seals = producer._preflight_native_source(
        native_root=root, pair_manifest_path=pair_path, pair_generation_id="a" * 64,
    )
    assert identity["manifest_path"] == str(root / "MANIFEST.json")
    assert binding["manifest_sha256"] == hashlib.sha256(pair_path.read_bytes()).hexdigest()
    assert len(parts) == 1 and parts[0].rows == 2
    assert len(seals) == 3


@pytest.mark.parametrize("field", [
    "explicit_vedtak_id", "source_environment", "source_base_url", "requested_start_utc",
    "requested_end_utc_exclusive", "producer_git_commit", "producer_source_inventory_sha256",
    "manifest_sha256", "unexpected_field",
])
def test_m5_preflight_rejects_changed_or_added_publisher_field(canonical_pair_source, field):
    root, pair_path, pair = canonical_pair_source
    pair["lineage"]["native_sources"]["m5"][field] = "changed"
    pair_path.write_text(json.dumps(pair))
    with pytest.raises(RuntimeError, match=f"M5_SOURCE_PAIR_NATIVE_M5_BINDING_MISMATCH: field={field}"):
        producer._preflight_native_source(
            native_root=root, pair_manifest_path=pair_path, pair_generation_id="a" * 64,
        )
