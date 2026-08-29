from gx1.contracts.xau_tape_provenance_v1 import (
    CANONICAL_NATIVE_SOURCE_SCHEMA,
    CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
)
from gx1.scripts import materialize_entry_model_native_m5_source_v1 as producer


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
