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
import hashlib
import json

import pytest
