from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gx1.contracts.unified_exit_pilot_final_bindings_v1 import (
    build_composite_normalization_binding,
)
from gx1.contracts.unified_exit_pilot_normalization_v1 import (
    build_physical_summary_sample_authority,
    canonical_sha256,
    fit_lifetime_summary_normalization,
)
from gx1.contracts.unified_exit_random_access_cuda_smoke_v1 import (
    build_blocked_smoke_manifest,
    build_bootstrap_base_normalization,
    build_bootstrap_composite_normalization,
    build_bootstrap_source_receipt,
    file_sha256,
    require_bootstrap_base_normalization,
    require_bootstrap_source_receipt,
    require_smoke_manifest,
)
from tests.model_native_input_normalization_support import input_normalization_fixture


def _checkpoint(norm: str) -> dict:
    return {
        "state_path": "/immutable/state.pt",
        "state_file_sha256": "1" * 64,
        "pointer_path": "/immutable/pointer.json",
        "pointer_file_sha256": "2" * 64,
        "session_contract_path": "/immutable/session.json",
        "session_contract_file_sha256": "3" * 64,
        "session_contract_sha256": "3" * 64,
        "checkpoint_index": 152,
        "slot": 1,
        "phase": "train",
        "epoch_index": 0,
        "next_batch_offset": 9664,
        "global_optimizer_steps": 9664,
        "container_schema_version": "gx1_candidate_training_session_v1",
        "container_keyset_sha256": "4" * 64,
        "online_model_state_sha256": "5" * 64,
        "target_model_state_sha256": "6" * 64,
        "model_state_keyset_sha256": "7" * 64,
        "model_state_key_count": 794,
        "input_normalization_sha256": norm,
        "contains_random_access_v2_state": False,
    }


def _summary() -> dict:
    authority = build_physical_summary_sample_authority(
        successor_transition_count_by_entry=[8, 9], source_lineage_sha256="8" * 64
    )
    rows = authority["fit_row_count"]
    raw = np.arange(rows, dtype=np.float64)[:, None]
    return fit_lifetime_summary_normalization(
        values=np.concatenate([raw * (i + 1) + i for i in range(7)], axis=1),
        sample_authority=authority,
    )


def _base_contract() -> dict:
    return input_normalization_fixture(
        signal_names=["signal_a", "signal_b"], mtf_names=["mtf_a", "mtf_b"]
    )


def _legacy_contract() -> dict:
    value = _base_contract()
    value["schema_version"] = "entry_model_native_input_normalization_v7"
    value["transform"] = "shared_entry_exit_train_only_median_raw_iqr_asinh_v4"
    value.pop("contract_sha256")
    value["contract_sha256"] = canonical_sha256(value)
    return value


def _source_metadata(contract: dict) -> dict:
    return {
        "git_commit": "a" * 40,
        "input_normalization": contract,
        "input_normalization_fit_population_proof": {"proof_sha256": "c" * 64},
        "recipe_source_provenance": {
            "source_commit": "a" * 40,
            "source_bindings": {
                "python:gx1/contracts/entry_model_native_input_normalization_v1.py": {
                    "path": "/immutable/entry_model_native_input_normalization_v1.py",
                    "sha256": "b" * 64,
                }
            },
        },
    }


def test_checkpoint_source_receipt_is_v1_only_and_hash_bound() -> None:
    receipt = build_bootstrap_source_receipt(
        checkpoint=_checkpoint("9" * 64), source_commit="a" * 40
    )
    assert require_bootstrap_source_receipt(receipt) == receipt
    assert receipt["strict_v2_restore_required_after_bootstrap"] is True
    bad = copy.deepcopy(receipt)
    bad["checkpoint"]["state_file_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="PAYLOAD_INVALID"):
        require_bootstrap_source_receipt(bad)
    v2 = _checkpoint("9" * 64)
    v2["contains_random_access_v2_state"] = True
    with pytest.raises(RuntimeError, match="CHECKPOINT_INVALID"):
        build_bootstrap_source_receipt(checkpoint=v2, source_commit="a" * 40)


def test_bootstrap_base_preserves_checkpoint_normalization_and_blocks_val_fit() -> None:
    old = _legacy_contract()
    old["lineage"]["train_time_max_utc"] = "2026-05-31T23:45:00+00:00"
    old["fit_end_utc"] = "2026-05-31T23:45:00+00:00"
    # Recompute the contract because the lineage is hash-bound.
    old_without = dict(old)
    old_without.pop("contract_sha256")
    from gx1.contracts.unified_exit_pilot_normalization_v1 import canonical_sha256

    old["contract_sha256"] = canonical_sha256(old_without)
    source = _source_metadata(old)
    child_contract = _base_contract()
    child = {
        "schema_version": "gx1_unified_exit_pilot_base_normalization_v1",
        "decision": "PASS",
        "contract": child_contract,
        "contract_sha256": child_contract["contract_sha256"],
        "population_witness_sha256": "d" * 64,
        "val_fit_rows": 0,
        "test_fit_rows": 0,
        "test_accessed": False,
    }
    artifact = build_bootstrap_base_normalization(
        source_bundle_metadata=source,
        source_bundle_metadata_path="/immutable/source.json",
        source_bundle_metadata_file_sha256="e" * 64,
        checkpoint_input_normalization_sha256=old["contract_sha256"],
        child_base_artifact=child,
        pilot_val_start_utc="2026-06-01T00:00:00+00:00",
    )
    assert require_bootstrap_base_normalization(artifact) == artifact
    assert artifact["base_artifact"]["contract_sha256"] == old["contract_sha256"]
    leaking = copy.deepcopy(source)
    leaking["input_normalization"]["lineage"]["val_fit_row_count"] = 1
    with pytest.raises(RuntimeError):
        build_bootstrap_base_normalization(
            source_bundle_metadata=leaking,
            source_bundle_metadata_path="/immutable/source.json",
            source_bundle_metadata_file_sha256="e" * 64,
            checkpoint_input_normalization_sha256=old["contract_sha256"],
            child_base_artifact=child,
            pilot_val_start_utc="2026-06-01T00:00:00+00:00",
        )


def test_smoke_manifest_cannot_self_attest_sampler_selection(tmp_path: Path) -> None:
    contract = _legacy_contract()
    source = _source_metadata(contract)
    base = build_bootstrap_base_normalization(
        source_bundle_metadata=source,
        source_bundle_metadata_path="/immutable/source.json",
        source_bundle_metadata_file_sha256="e" * 64,
        checkpoint_input_normalization_sha256=contract["contract_sha256"],
        child_base_artifact={
            "contract": _base_contract(),
            "contract_sha256": _base_contract()["contract_sha256"],
        },
        pilot_val_start_utc="2027-01-01T00:00:00+00:00",
    )
    base_file = tmp_path / "base.json"
    base_file.write_text(json.dumps(base, sort_keys=True, indent=2) + "\n")
    summary = _summary()
    child_base_contract = _base_contract()
    child_base = {
        "schema_version": "gx1_unified_exit_pilot_base_normalization_v1",
        "decision": "PASS",
        "contract": child_base_contract,
        "contract_sha256": child_base_contract["contract_sha256"],
        "population_witness_sha256": "d" * 64,
        "val_fit_rows": 0,
        "test_fit_rows": 0,
        "test_accessed": False,
    }
    child_composite = build_composite_normalization_binding(
        base_artifact=child_base,
        base_path=str(base_file),
        base_file_sha256=file_sha256(base_file),
        summary_normalization=summary,
        summary_manifest_path="/immutable/summary.json",
        summary_manifest_file_sha256="f" * 64,
        summary_manifest_sha256="0" * 64,
    )
    child_composite_file = tmp_path / "child_composite.json"
    child_composite_file.write_text(
        json.dumps(child_composite, sort_keys=True, indent=2) + "\n"
    )
    composite = build_bootstrap_composite_normalization(
        bootstrap_base=base,
        base_path=str(base_file),
        base_file_sha256=file_sha256(base_file),
        child_composite=child_composite,
    )
    composite_file = tmp_path / "composite.json"
    composite_file.write_text(json.dumps(composite, sort_keys=True, indent=2) + "\n")
    dummy = tmp_path / "dummy.json"
    dummy.write_text("{}\n")
    artifacts = {
        name: {"path": str(dummy), "sha256": file_sha256(dummy)}
        for name in (
            "random_access_index_root",
            "final_bindings_bundle",
            "candidate_set",
            "train_closure_authority",
            "val_closure_authority",
            "economics_readiness",
            "train_cost_authority",
            "val_cost_authority",
            "state_view_source",
        )
    }
    artifacts["child_composite_normalization"] = {
        "path": str(child_composite_file),
        "sha256": file_sha256(child_composite_file),
    }
    artifacts["bootstrap_composite_normalization"] = {
        "path": str(composite_file),
        "sha256": file_sha256(composite_file),
    }
    coordinator = {
        name: {"path": str(dummy), "sha256": file_sha256(dummy)}
        for name in ("contract", "controller", "telemetry", "installer")
    }
    receipt = build_bootstrap_source_receipt(
        checkpoint=_checkpoint(contract["contract_sha256"]), source_commit="a" * 40
    )
    manifest = build_blocked_smoke_manifest(
        source_repo=str(tmp_path),
        source_commit="a" * 40,
        bootstrap_source=receipt,
        bootstrap_base=base,
        artifacts=artifacts,
        coordinator=coordinator,
        output_root=str(tmp_path / "out"),
    )
    assert require_smoke_manifest(manifest) == manifest
    bad = copy.deepcopy(manifest)
    bad["sampler_selection"] = {
        "status": "PASS",
        "receipt_path": "/fake",
        "receipt_file_sha256": "1" * 64,
        "selected_sampler_contract_sha256": "2" * 64,
    }
    bad.pop("manifest_sha256")
    from gx1.contracts.unified_exit_pilot_normalization_v1 import canonical_sha256

    bad["manifest_sha256"] = canonical_sha256(bad)
    with pytest.raises(RuntimeError, match="NOT_BLOCKED"):
        require_smoke_manifest(bad, verify_files=False)
