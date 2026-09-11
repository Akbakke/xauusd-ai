import copy
import hashlib
import json
from pathlib import Path

import pytest

from gx1.contracts.unified_exit_full_population_train_session_v1 import (
    build_full_population_train_session, require_full_population_train_session,
)
from gx1.contracts.unified_exit_random_access_sampler_v1 import (
    build_random_access_sampler_contract, canonical_sha256,
    schedule_random_access_full_population_epoch,
)
from gx1.scripts.run_unified_exit_random_access_fixed_step_v1 import _schedule_witness


def _binding(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _session(tmp_path, monkeypatch):
    # Mock only the separately tested tensor-authority boundary; exercise all new bindings.
    monkeypatch.setattr(
        "gx1.contracts.unified_exit_final_train_checkpoint_authority_v1.require_final_train_checkpoint_authority",
        lambda value, verify_files=True: dict(value),
    )
    contract = build_random_access_sampler_contract(
        split="train", source_lineage_sha256="a" * 64,
        transition_budget_per_epoch=320, transitions_per_entry=4,
        entry_pair_population=95,
    )
    _, anchors, schedule = schedule_random_access_full_population_epoch(
        sampler_contract=contract, epoch_index=0,
        successor_transition_count_by_entry=[100] * 95,
    )
    selected = _binding(tmp_path / "selected.json", {
        "selected_sampler_contract": contract,
        "selected_sampler_contract_sha256": contract["contract_sha256"],
    })
    launch = _binding(tmp_path / "launch.json", {"files": {"selected_sampler": selected}})
    selection = _binding(tmp_path / "selection.json", {
        "artifact_sha256": "b" * 64, "checkpoint_interval_optimizer_steps": 64,
    })
    state = _binding(tmp_path / "old" / "state.json", {"fixture": True})
    pointer_value = {
        "state_path": state["path"], "state_file_sha256": state["sha256"],
        "epoch_schedule_sha256": "c" * 64, "global_step": 5,
        "next_batch_offset": 5, "batch_size": 16,
    }
    pointer = _binding(tmp_path / "old" / "pointer.json", pointer_value)
    authority = _binding(tmp_path / "authority.json", {
        "source_commit": "d" * 40, "selected_batch_size": 16,
        "final_checkpoint_pointer": pointer, "launch_manifest": launch,
        "launch_manifest_sha256": "e" * 64, "gpu_batch_selection": selection,
    })
    proof = {
        "decision": "PASS_FULL_YEAR_COVERAGE_AND_COMPLETED_PREFIX",
        "prior_checkpoint_pointer": pointer, "prior_checkpoint_state": state,
        "full_population_schedule": schedule, "prefix_entry_pairs": 80,
        "full_epoch_entry_pairs": 95, "full_epoch_transitions": 380,
        "prefix_transition_and_anchor_bytes_equal": True,
        "prefix_parent_order_equal": True, "legacy_sampler_functions_ast_unchanged": True,
        "test_data_used": False,
    }
    proof["proof_sha256"] = canonical_sha256(proof)
    proof_binding = _binding(tmp_path / "proof.json", proof)
    value = build_full_population_train_session(
        source_repo=tmp_path, source_commit="f" * 40,
        checkpoint_dir=tmp_path / "new",
        prefix_authority_binding=authority, prefix_proof_binding=proof_binding,
    )
    return value, anchors


def _reseal(value):
    value.pop("manifest_sha256")
    value["manifest_sha256"] = canonical_sha256(value)
    return value


def test_continuation_counts_retained_prefix_and_partial_last_batch(tmp_path, monkeypatch):
    value, anchors = _session(tmp_path, monkeypatch)
    assert require_full_population_train_session(value) == value
    assert value["total_batches_per_epoch"] == 6
    assert value["initial_batch_offset"] == 5
    assert value["remaining_optimizer_steps"] == 1
    assert value["entry_pairs_per_epoch"] % value["selected_batch_size"] == 15
    order = [a["entry_row_index"] for a in anchors]
    witness = _schedule_witness(
        child_order=order, parent_order=[i + 1000 for i in order],
        batch_size=16, epoch_schedule_sha256="9" * 64,
        selected_sampler_artifact_sha256="8" * 64,
        full_population_schedule=value["full_population_schedule"],
    )
    assert witness["entry_pair_count"] == 95
    assert witness["full_population_schedule"] == value["full_population_schedule"]


def test_continuation_rejects_changed_budget_and_prefix_overwrite(tmp_path, monkeypatch):
    value, _ = _session(tmp_path, monkeypatch)
    changed = copy.deepcopy(value)
    changed["remaining_optimizer_steps"] += 1
    with pytest.raises(RuntimeError, match="PREFIX_BINDING_INVALID"):
        require_full_population_train_session(_reseal(changed))
    changed = copy.deepcopy(value)
    changed["checkpoint_dir"] = str(Path(value["prefix_checkpoint"]["path"]).parent)
    with pytest.raises(RuntimeError, match="PREFIX_OVERWRITE_FORBIDDEN"):
        require_full_population_train_session(_reseal(changed))
