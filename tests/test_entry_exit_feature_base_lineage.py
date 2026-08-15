from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION,
    entry_exit_shared_feature_base_contract,
    require_entry_exit_enriched_source_binding,
)
from gx1.utils.artifact_primitives_v1 import (
    canonical_json_sha256,
    sha256_file,
)


DATASET_RUN_ID = "ENTRY_EXIT_DATASET_PYTEST_V1"
PAIR_GENERATION_ID = "ENTRY_EXIT_PAIR_PYTEST_V1"


def _source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = (tmp_path / "enriched_m1.parquet").resolve()
    source.write_bytes(b"immutable-enriched-m1")
    payload = {
        "schema_version": ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION,
        "decision": "PASS",
        "shared_feature_base_contract": entry_exit_shared_feature_base_contract(),
        "dataset_run_id": DATASET_RUN_ID,
        "pair_generation_id": PAIR_GENERATION_ID,
        "timeframe": "M1",
        "base_bar_seconds": 60,
        "output_parquet": str(source),
        "output_parquet_sha256": sha256_file(source),
    }
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    manifest = source.with_suffix(source.suffix + ".manifest.json")
    manifest.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return source, manifest


def test_feature_base_requires_exact_enriched_source_lineage(
    tmp_path: Path,
) -> None:
    source, manifest = _source_fixture(tmp_path)

    binding = require_entry_exit_enriched_source_binding(
        source,
        dataset_run_id=DATASET_RUN_ID,
        pair_generation_id=PAIR_GENERATION_ID,
        timeframe="M1",
        context="TEST_ENTRY_EXIT_FEATURE_BASE",
    )

    assert binding["manifest_path"] == str(manifest)
    assert binding["manifest_sha256"] == sha256_file(manifest)
    assert binding["source_sha256"] == sha256_file(source)


@pytest.mark.parametrize(
    "field,value,error",
    (
        ("dataset_run_id", "RELABELLED_DATASET_RUN", "SOURCE_LINEAGE_INVALID"),
        ("pair_generation_id", "RELABELLED_PAIR", "SOURCE_LINEAGE_INVALID"),
        ("schema_version", "legacy_enriched_schema", "SOURCE_LINEAGE_INVALID"),
        ("timeframe", "M5", "SOURCE_LINEAGE_INVALID"),
        ("base_bar_seconds", 300, "SOURCE_LINEAGE_INVALID"),
        ("output_parquet", "/other/enriched_m1.parquet", "SOURCE_LINEAGE_INVALID"),
        ("output_parquet_sha256", "0" * 64, "SOURCE_LINEAGE_INVALID"),
    ),
)
def test_feature_base_rejects_relabelled_or_unbound_source(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    source, manifest = _source_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload[field] = value
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=error):
        require_entry_exit_enriched_source_binding(
            source,
            dataset_run_id=DATASET_RUN_ID,
            pair_generation_id=PAIR_GENERATION_ID,
            timeframe="M1",
            context="TEST_ENTRY_EXIT_FEATURE_BASE",
        )


def test_feature_base_rejects_source_bytes_changed_after_manifest(
    tmp_path: Path,
) -> None:
    source, _manifest = _source_fixture(tmp_path)
    source.write_bytes(b"changed-after-publication")

    with pytest.raises(RuntimeError, match="SOURCE_LINEAGE_INVALID"):
        require_entry_exit_enriched_source_binding(
            source,
            dataset_run_id=DATASET_RUN_ID,
            pair_generation_id=PAIR_GENERATION_ID,
            timeframe="M1",
            context="TEST_ENTRY_EXIT_FEATURE_BASE",
        )
