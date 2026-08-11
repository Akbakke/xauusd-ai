"""Contract tests for the retired-ancestor retention attestation.

A missing successor parent root is admitted ONLY when an executed
DELETE_COMPLETE retention event covers exactly that root and its
hash-verified per-file inventory attests the deleted MANIFEST.json with
exactly the child's recorded parent manifest sha256. Every weaker record
must return None (the caller fails closed).

All records here are synthetic (rule 2c): they prove the code contract.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gx1.contracts.xau_tape_provenance_v1 import (
    _retired_native_root_attestation,
)

_PARENT = Path("/synthetic/native_xau/XAU_M1_NATIVE_TEST_V3")
_MANIFEST_SHA = "ab" * 32


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_records(
    tmp_path: Path,
    *,
    decision: str = "DELETE_COMPLETE",
    target_path: str = str(_PARENT),
    inventory_sha: str = _MANIFEST_SHA,
    break_plan_hash: bool = False,
    break_inventory_hash: bool = False,
) -> Path:
    reports = tmp_path / "reports"
    plans = tmp_path / "plans"
    reports.mkdir()
    plans.mkdir()
    inventory = plans / "GX1_EVIDENCE_CLEANUP_INVENTORY_TEST_0001.jsonl"
    inventory.write_text(
        json.dumps(
            {
                "relative_path": "MANIFEST.json",
                "kind": "file",
                "sha256": inventory_sha,
                "size_bytes": 123,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    plan = plans / "GX1_EVIDENCE_RETENTION_CLEANUP_PLAN_TEST.json"
    plan.write_text(
        json.dumps(
            {
                "plan": {
                    "targets": [
                        {
                            "path": target_path,
                            "kind": "directory",
                            "inventory_jsonl": str(inventory),
                            "inventory_jsonl_sha256": (
                                "00" * 32
                                if break_inventory_hash
                                else _sha(inventory)
                            ),
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    execution = reports / "GX1_EVIDENCE_CLEANUP_EXECUTION_TEST.json"
    execution.write_text(
        json.dumps(
            {
                "schema_version": "gx1_evidence_cleanup_execution_v1",
                "decision": decision,
                "deleted": [target_path],
                "plan_json": str(plan),
                "plan_sha256": (
                    "00" * 32 if break_plan_hash else _sha(plan)
                ),
                "vedtak": "VEDTAK_TEST_CLEANUP_V1",
            }
        ),
        encoding="utf-8",
    )
    return reports


def test_exact_attestation_succeeds(tmp_path: Path) -> None:
    reports = _write_records(tmp_path)
    attestation = _retired_native_root_attestation(
        _PARENT,
        expected_manifest_sha256=_MANIFEST_SHA,
        reports_dir=reports,
    )
    assert attestation is not None
    assert attestation["parent_manifest_sha256"] == _MANIFEST_SHA
    assert attestation["vedtak"] == "VEDTAK_TEST_CLEANUP_V1"
    assert (
        attestation["schema_version"]
        == "gx1_retired_native_root_attestation_v1"
    )


def test_wrong_manifest_sha_fails(tmp_path: Path) -> None:
    reports = _write_records(tmp_path)
    assert (
        _retired_native_root_attestation(
            _PARENT,
            expected_manifest_sha256="cd" * 32,
            reports_dir=reports,
        )
        is None
    )


def test_incomplete_execution_fails(tmp_path: Path) -> None:
    reports = _write_records(tmp_path, decision="CLEANUP_PARTIAL_FAILURE")
    assert (
        _retired_native_root_attestation(
            _PARENT,
            expected_manifest_sha256=_MANIFEST_SHA,
            reports_dir=reports,
        )
        is None
    )


def test_uncovered_root_fails(tmp_path: Path) -> None:
    reports = _write_records(tmp_path, target_path="/synthetic/other/root")
    assert (
        _retired_native_root_attestation(
            _PARENT,
            expected_manifest_sha256=_MANIFEST_SHA,
            reports_dir=reports,
        )
        is None
    )


def test_tampered_plan_hash_fails(tmp_path: Path) -> None:
    reports = _write_records(tmp_path, break_plan_hash=True)
    assert (
        _retired_native_root_attestation(
            _PARENT,
            expected_manifest_sha256=_MANIFEST_SHA,
            reports_dir=reports,
        )
        is None
    )


def test_tampered_inventory_hash_fails(tmp_path: Path) -> None:
    reports = _write_records(tmp_path, break_inventory_hash=True)
    assert (
        _retired_native_root_attestation(
            _PARENT,
            expected_manifest_sha256=_MANIFEST_SHA,
            reports_dir=reports,
        )
        is None
    )


def test_missing_reports_dir_fails(tmp_path: Path) -> None:
    assert (
        _retired_native_root_attestation(
            _PARENT,
            expected_manifest_sha256=_MANIFEST_SHA,
            reports_dir=tmp_path / "absent",
        )
        is None
    )
