from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import gx1.contracts.entry_model_native_post_rebuild_v1 as guard
from gx1.models.entry_v10.entry_v10_bundle import (
    _require_current_prefreeze_test_seal_lineage,
)


DATASET_RUN_ID = "PRETEST_GUARD_PYTEST_V1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _event(tmp_path: Path) -> tuple[Path, str, dict[str, object], set[Path]]:
    dataset_dir = (tmp_path / "dataset").resolve()
    authority = (tmp_path / "rebuild_authority").resolve()
    dataset_dir.mkdir()
    authority.mkdir()
    train_manifest = dataset_dir / "entry_train.manifest.json"
    train_parquet = dataset_dir / "entry_train.parquet"
    val_manifest = dataset_dir / "entry_val.manifest.json"
    val_parquet = dataset_dir / "entry_val.parquet"
    proof = dataset_dir / "DATASET_BUILD_PROOF.json"
    liveness = dataset_dir / "ENTRY_FULL_INPUT_LIVENESS_20260830T010000Z.json"
    # Deliberately create no TEST path.  The guard event must be sufficient
    # without even naming one.
    for path in (train_manifest, train_parquet, val_manifest, val_parquet, proof, liveness):
        path.write_bytes(b"allowed-control-plane-fixture")
    event_path = authority / (
        f"{guard.PRETEST_TEST_GUARD_EVENT_PREFIX}_20260830T010203000000Z.json"
    )
    payload: dict[str, object] = {
        "schema_version": guard.PRETEST_TEST_GUARD_SCHEMA_VERSION,
        "decision": guard.PRETEST_TEST_GUARD_DECISION,
        "created_utc": "2026-08-30T01:02:03+00:00",
        "guard_path": str(event_path),
        "entry_run_id": DATASET_RUN_ID,
        "dataset_dir": str(dataset_dir),
        "split": "test",
        "access_policy": guard.PRETEST_TEST_GUARD_ACCESS_POLICY,
        "disclosure_count": 0,
        "test_boundary_utc": "2026-07-01T00:00:00+00:00",
        "test_accessed": False,
        "test_materialized": False,
        "train_manifest": {"path": str(train_manifest), "sha256": "a" * 64},
        "train_parquet": {"path": str(train_parquet), "sha256": "b" * 64},
        "val_manifest": {"path": str(val_manifest), "sha256": "c" * 64},
        "val_parquet": {"path": str(val_parquet), "sha256": "d" * 64},
        "dataset_build_proof": {"path": str(proof), "sha256": "e" * 64},
        "full_input_liveness": {"path": str(liveness), "sha256": "f" * 64},
    }
    payload["content_binding_sha256"] = guard._canonical_json_sha256(payload)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    event_path.write_bytes(raw)
    return event_path, _sha256_bytes(raw), payload, set()


def test_pretest_guard_never_opens_stats_or_names_a_test_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_path, sha256, payload, forbidden = _event(tmp_path)
    original_open = Path.open
    original_stat = Path.stat

    def guarded_open(self: Path, *args, **kwargs):
        assert self not in forbidden, f"TEST bytes opened: {self}"
        return original_open(self, *args, **kwargs)

    def guarded_stat(self: Path, *args, **kwargs):
        assert self not in forbidden, f"TEST path statted: {self}"
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    observed = guard.require_pretest_or_prefreeze_test_guard_lineage(
        event_path,
        sha256,
        expected_dataset_run_id=DATASET_RUN_ID,
        expected_dataset_dir=payload["dataset_dir"],
    )
    assert observed["schema_version"] == guard.PRETEST_TEST_GUARD_LINEAGE_SCHEMA_VERSION
    assert observed["access_proof"] == {
        "guard_event_bytes_hash_validated": True,
        "test_dataset_bytes_read": False,
        "test_manifest_bytes_read": False,
        "test_metrics_read": False,
        "test_paths_resolved_or_statted": False,
    }
    assert "test_parquet" not in observed and "test_manifest" not in observed
    assert _require_current_prefreeze_test_seal_lineage(
        {
            "run_lineage": {"dataset_run_id": DATASET_RUN_ID},
            "prefreeze_test_seal_lineage": observed,
        }
    ) == observed


def test_pretest_guard_rejects_any_claim_that_test_was_materialized(tmp_path: Path) -> None:
    event_path, _, payload, _ = _event(tmp_path)
    payload["test_materialized"] = True
    payload["content_binding_sha256"] = guard._canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "content_binding_sha256"}
    )
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    event_path.write_bytes(raw)
    with pytest.raises(guard.PrefreezeTestSealLineageError, match="identity/access mismatch"):
        guard.require_pretest_test_guard_lineage(
            event_path,
            _sha256_bytes(raw),
            expected_dataset_run_id=DATASET_RUN_ID,
            expected_dataset_dir=payload["dataset_dir"],
        )


def test_legacy_test_evaluation_validator_rejects_a_pretest_guard(tmp_path: Path) -> None:
    event_path, sha256, payload, _ = _event(tmp_path)
    with pytest.raises(guard.PrefreezeTestSealLineageError, match="keys are not exact"):
        guard.require_prefreeze_test_seal_lineage(
            event_path,
            sha256,
            expected_dataset_run_id=DATASET_RUN_ID,
            expected_dataset_dir=payload["dataset_dir"],
        )
