from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from gx1.contracts.entry_model_native_post_rebuild_v1 import (
    DATASET_REBUILD_TERMINAL_DECISION,
    DATASET_REBUILD_TERMINAL_EVENT_PREFIX,
    DATASET_REBUILD_TERMINAL_SCHEMA_VERSION,
    PREFREEZE_TEST_SEAL_LINEAGE_SCHEMA_VERSION,
    PREFREEZE_TEST_SEAL_VERIFICATION_MODE,
    TEST_SEAL_ACCESS_POLICY,
    TEST_SEAL_DECISION,
    TEST_SEAL_EVENT_PREFIX,
    TEST_SEAL_SCHEMA_VERSION,
    require_prefreeze_test_seal_lineage,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
)
from gx1.contracts.xau_tape_provenance_v1 import XAU_INSTRUMENT


STAMP = "20260730T010203123456Z"


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prefreeze_test_seal_lineage_fixture(
    *,
    dataset_run_id: str,
    dataset_dir: str = "/immutable/model_native_seq513_dataset",
) -> dict[str, Any]:
    root = Path(dataset_dir)
    authority = root.parent / "rebuild_authority"
    return {
        "schema_version": PREFREEZE_TEST_SEAL_LINEAGE_SCHEMA_VERSION,
        "verification_mode": PREFREEZE_TEST_SEAL_VERIFICATION_MODE,
        "seal_event": {
            "path": str(authority / f"{TEST_SEAL_EVENT_PREFIX}_{STAMP}.json"),
            "sha256": "1" * 64,
            "schema_version": TEST_SEAL_SCHEMA_VERSION,
            "decision": TEST_SEAL_DECISION,
            "created_utc": "2026-07-30T01:02:03.123456+00:00",
            "content_binding_sha256": "2" * 64,
        },
        "dataset_run_id": dataset_run_id,
        "dataset_dir": str(root),
        "split": "test",
        "access_policy": TEST_SEAL_ACCESS_POLICY,
        "disclosure_count": 0,
        "test_manifest": {
            "path": str(root / "xau_seq513_test.manifest.json"),
            "sha256": "3" * 64,
            "schema_version": MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
            "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
            "entry_run_id": dataset_run_id,
            "instrument": XAU_INSTRUMENT,
        },
        "test_parquet": {
            "path": str(root / "xau_seq513_test.parquet"),
            "sha256": "4" * 64,
        },
        "rows": 1,
        "pair_lineage_sha256": "5" * 64,
        "source_lineage_sha256": "6" * 64,
        "rebuild_terminal": {
            "path": str(
                authority
                / f"{DATASET_REBUILD_TERMINAL_EVENT_PREFIX}_{STAMP}.json"
            ),
            "sha256": "7" * 64,
            "schema_version": DATASET_REBUILD_TERMINAL_SCHEMA_VERSION,
            "decision": DATASET_REBUILD_TERMINAL_DECISION,
            "content_binding_sha256": "8" * 64,
        },
        "access_proof": {
            "seal_event_bytes_hash_validated": True,
            "test_dataset_bytes_read": False,
            "test_manifest_bytes_read": False,
            "test_metrics_read": False,
            "test_paths_resolved_or_statted": False,
        },
    }


def write_prefreeze_test_seal_fixture(
    *,
    authority_dir: Path,
    dataset_dir: Path,
    dataset_run_id: str,
    manifest_path: Path,
    manifest_sha256: str,
    parquet_path: Path,
    parquet_sha256: str,
    rows: int = 1,
) -> tuple[Path, str, dict[str, Any]]:
    authority = authority_dir.resolve()
    authority.mkdir(parents=True, exist_ok=True)
    dataset = dataset_dir.resolve()
    seal_path = authority / f"{TEST_SEAL_EVENT_PREFIX}_{STAMP}.json"
    terminal_path = (
        authority / f"{DATASET_REBUILD_TERMINAL_EVENT_PREFIX}_{STAMP}.json"
    )
    pair_lineage = {"fixture_pair_generation": "pair-generation-v1"}
    source_lineage = {"fixture_source_generation": "source-generation-v1"}
    rebuild_terminal = {
        "path": str(terminal_path),
        "sha256": "7" * 64,
        "schema_version": DATASET_REBUILD_TERMINAL_SCHEMA_VERSION,
        "decision": DATASET_REBUILD_TERMINAL_DECISION,
        "content_binding_sha256": "8" * 64,
    }
    payload = {
        "schema_version": TEST_SEAL_SCHEMA_VERSION,
        "decision": TEST_SEAL_DECISION,
        "created_utc": "2026-07-30T01:02:03.123456+00:00",
        "seal_path": str(seal_path),
        "entry_run_id": dataset_run_id,
        "dataset_dir": str(dataset),
        "split": "test",
        "access_policy": TEST_SEAL_ACCESS_POLICY,
        "disclosure_count": 0,
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
        "parquet": {"path": str(parquet_path), "sha256": parquet_sha256},
        "manifest_contract": {
            "schema_version": MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
            "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
            "entry_run_id": dataset_run_id,
            "instrument": XAU_INSTRUMENT,
        },
        "rows": rows,
        "pair_lineage": pair_lineage,
        "pair_lineage_sha256": canonical_json_sha256(pair_lineage),
        "source_lineage": source_lineage,
        "source_lineage_sha256": canonical_json_sha256(source_lineage),
        "rebuild_terminal": rebuild_terminal,
    }
    payload["content_binding_sha256"] = canonical_json_sha256(payload)
    seal_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    seal_sha256 = hashlib.sha256(seal_path.read_bytes()).hexdigest()
    lineage = require_prefreeze_test_seal_lineage(
        seal_path,
        seal_sha256,
        expected_dataset_run_id=dataset_run_id,
        expected_dataset_dir=dataset,
    )
    return seal_path, seal_sha256, lineage
