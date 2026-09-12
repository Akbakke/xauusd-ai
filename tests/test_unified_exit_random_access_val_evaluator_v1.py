from __future__ import annotations

from gx1.contracts.unified_exit_entry_policy_evaluation_v1 import build_entry_policy_decisions

import json

import numpy as np
import pytest
import torch

from gx1.contracts.unified_exit_random_access_checkpoint_v1 import (
    canonical_sha256,
    file_sha256,
)
from gx1.contracts.unified_exit_random_access_model_v1 import (
    RANDOM_ACCESS_MODEL_SCHEMA_SHA256,
    RANDOM_ACCESS_MODEL_SCHEMA_VERSION,
)
from gx1.contracts.unified_exit_random_access_val_checkpoint_v1 import (
    VAL_CHECKPOINT_BINDING_SCHEMA_VERSION,
)
from gx1.contracts.unified_exit_random_access_val_evaluator_v1 import (
    PAUSE_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    run_resumable_random_access_val_evaluation_v1,
)
from tests.test_unified_exit_random_access_val_rollout_v1 import (
    VAL_ENTRY_COHORT_SIZE,
    _fixture,
)


def _with_route_outputs(model, seen_batch_sizes=None):
    original = model.forward_exit_random_access_batch

    def forward(**inputs):
        result = original(**inputs)
        batch = result["exit_action_q_bps"].shape[0]
        if seen_batch_sizes is not None:
            seen_batch_sizes.append(batch)
        result.update(
            {
                "exit_specialist_gate": torch.full((batch, 1, 3), 1.0 / 3.0),
                "exit_tf_gate": torch.full((batch, 1, 2), 0.5),
                "exit_family_tf_cooperation_gate": torch.full(
                    (batch, 1, 2, 3), 1.0 / 6.0
                ),
                "exit_family_tf_feature_gate": torch.full((batch, 1, 2, 3), 1.5),
            }
        )
        return result

    model.forward_exit_random_access_batch = forward
    return model


def _entry_policy(adapter, binding):
    return build_entry_policy_decisions(
        predicted_q_bps=np.tile(np.array([2, 1, 0], dtype=np.float32), (len(adapter.entries), 1)),
        entry_row_indices=[int(row["entry_row_index"]) for row in adapter.entries],
        checkpoint_binding_sha256=binding["binding_sha256"],
    )


def _checkpoint_binding(contract, adapter, tmp_path):
    checkpoint = tmp_path / "selected-v2.pt"
    pointer = tmp_path / "RESUME_POINTER.json"
    checkpoint.write_bytes(b"strict-v2-checkpoint")
    pointer.write_text("{}")
    checkpoint_sha = file_sha256(checkpoint)
    adapter.contract["checkpoint_file_sha256"] = checkpoint_sha
    adapter_contract = dict(adapter.contract)
    adapter_contract.pop("contract_sha256")
    adapter.contract["contract_sha256"] = canonical_sha256(adapter_contract)
    contract["checkpoint_file_sha256"] = checkpoint_sha
    contract["contract_sha256"] = adapter.contract["contract_sha256"]
    value = {
        "schema_version": VAL_CHECKPOINT_BINDING_SCHEMA_VERSION,
        "decision": "PASS",
        "model_variant": "weight_ema",
        "model_architecture_schema_version": RANDOM_ACCESS_MODEL_SCHEMA_VERSION,
        "model_architecture_sha256": RANDOM_ACCESS_MODEL_SCHEMA_SHA256,
        "model_state_sha256": contract["model_state_sha256"],
        "online_model_state_sha256": "1" * 64,
        "target_model_state_sha256": "2" * 64,
        "checkpoint_path": str(checkpoint),
        "checkpoint_file_sha256": checkpoint_sha,
        "checkpoint_pointer_path": str(pointer),
        "checkpoint_pointer_file_sha256": file_sha256(pointer),
        "checkpoint_pointer_sha256": "3" * 64,
        "launch_manifest_sha256": "4" * 64,
        "selected_sampler_artifact_sha256": "5" * 64,
        "bootstrap_source_receipt_sha256": "6" * 64,
        "base_normalization_sha256": "7" * 64,
        "summary_normalization_sha256": "8" * 64,
        "batch_size": 16,
        "epoch_schedule_sha256": "9" * 64,
        "epoch_index": 0,
        "next_batch_offset": 1024,
        "global_step": 1024,
        "weight_ema_decay": 1.0 - 1.0 / 1024.0,
        "weight_ema_steps": 1024,
        "weight_ema_parameter_names_sha256": "a" * 64,
        "online_buffers_preserved_exactly": True,
        "rng_mutated": False,
        "optimizer_or_scheduler_loaded": False,
        "test_data_used": False,
    }
    value["binding_sha256"] = canonical_sha256(value)
    return value


def test_full_cohort_pause_resume_is_semantically_exact(tmp_path) -> None:
    thresholds = np.ones((VAL_ENTRY_COHORT_SIZE, 2), dtype=np.float32)
    counts = np.full(VAL_ENTRY_COHORT_SIZE, 2, dtype=np.int64)
    model, representations, adapter, contract = _fixture(
        thresholds=thresholds,
        counts=counts,
    )
    seen_batch_sizes = []
    model = _with_route_outputs(model, seen_batch_sizes)
    binding = _checkpoint_binding(contract, adapter, tmp_path)
    progress = tmp_path / "progress.json"
    result = tmp_path / "result.json"

    paused = run_resumable_random_access_val_evaluation_v1(
        model=model,
        entry_decision_representations=representations,
        adapter=adapter,
        checkpoint_binding=binding,
        entry_policy_decisions=_entry_policy(adapter, binding),
        entry_route_diagnostics={
            "semantics": "observation_only",
            "causal_feature_importance_claimed": False,
        },
        progress_path=progress,
        result_path=result,
        max_forwards_this_invocation=1,
        policy_batch_size=16,
        progress_interval_forwards=1,
    )
    assert paused["schema_version"] == PAUSE_SCHEMA_VERSION
    assert paused["decision"] == "PAUSED_RESUMABLE"
    assert paused["next_state_index"] == 0
    assert paused["next_entry_scan_position"] == 16
    assert paused["completed_entry_pair_count"] == 0
    assert paused["completed_side_trade_count"] == 0
    assert not result.exists()

    with pytest.raises(RuntimeError, match="PROGRESS_INVALID"):
        run_resumable_random_access_val_evaluation_v1(
            model=model,
            entry_decision_representations=representations,
            adapter=adapter,
            checkpoint_binding=binding,
            entry_policy_decisions=_entry_policy(adapter, binding),
            entry_route_diagnostics={
                "semantics": "observation_only",
                "causal_feature_importance_claimed": False,
            },
            progress_path=progress,
            result_path=result,
            max_forwards_this_invocation=1,
            policy_batch_size=8,
            progress_interval_forwards=1,
        )

    completed = run_resumable_random_access_val_evaluation_v1(
        model=model,
        entry_decision_representations=representations,
        adapter=adapter,
        checkpoint_binding=binding,
        entry_policy_decisions=_entry_policy(adapter, binding),
        entry_route_diagnostics={
            "semantics": "observation_only",
            "causal_feature_importance_claimed": False,
        },
        progress_path=progress,
        result_path=result,
        max_forwards_this_invocation=700,
        policy_batch_size=16,
        progress_interval_forwards=64,
    )
    assert completed["schema_version"] == RESULT_SCHEMA_VERSION
    assert completed["decision"] == "PASS_COMPLETE"
    assert completed["entry_pair_cohort_size"] == VAL_ENTRY_COHORT_SIZE
    assert completed["side_trade_count"] == VAL_ENTRY_COHORT_SIZE * 2
    assert completed["exited_side_trade_count"] == VAL_ENTRY_COHORT_SIZE * 2
    assert completed["model_forward_count"] == 690
    assert max(seen_batch_sizes) == 16
    assert seen_batch_sizes[0] == 16
    assert min(seen_batch_sizes) == 4
    assert len(seen_batch_sizes) == 690
    assert completed["execution_contract"]["policy_batch_size"] == 16
    assert completed["exit_policy_diagnostics"]["hold_action_count"] == (
        VAL_ENTRY_COHORT_SIZE * 2
    )
    assert completed["exit_policy_diagnostics"]["exit_now_action_count"] == (
        VAL_ENTRY_COHORT_SIZE * 2
    )
    routes = completed["exit_gate_and_feature_route_diagnostics"]
    assert routes["causal_feature_importance_claimed"] is False
    assert set(routes["routes"]) == {
        "exit_specialist_gate",
        "exit_tf_gate",
        "exit_family_tf_cooperation_gate",
        "exit_family_tf_feature_gate",
    }
    persisted = json.loads(result.read_text())
    assert persisted["semantic_result_sha256"] == completed["semantic_result_sha256"]
    terminal = json.loads(progress.read_text())
    assert terminal["decision"] == "COMPLETE"
    assert terminal["result_file_sha256"]

    recovered = run_resumable_random_access_val_evaluation_v1(
        model=model,
        entry_decision_representations=representations,
        adapter=adapter,
        checkpoint_binding=binding,
        entry_policy_decisions=_entry_policy(adapter, binding),
        entry_route_diagnostics={
            "semantics": "observation_only",
            "causal_feature_importance_claimed": False,
        },
        progress_path=progress,
        result_path=result,
        max_forwards_this_invocation=1,
        policy_batch_size=16,
    )
    assert recovered == completed
    assert completed["entry_exit_policy_metrics"]["selected_trade_count"] == VAL_ENTRY_COHORT_SIZE
    assert completed["entry_exit_policy_metrics"]["full_cohort_authoritative"] is True


def test_checkpoint_variant_and_route_evidence_fail_closed(tmp_path) -> None:
    thresholds = np.zeros((VAL_ENTRY_COHORT_SIZE, 2), dtype=np.float32)
    counts = np.ones(VAL_ENTRY_COHORT_SIZE, dtype=np.int64)
    model, representations, adapter, contract = _fixture(
        thresholds=thresholds,
        counts=counts,
    )
    binding = _checkpoint_binding(contract, adapter, tmp_path)
    binding["model_variant"] = "online"
    with pytest.raises(RuntimeError, match="CHECKPOINT_BINDING"):
        run_resumable_random_access_val_evaluation_v1(
            model=model,
            entry_decision_representations=representations,
            adapter=adapter,
            checkpoint_binding=binding,
            entry_policy_decisions=_entry_policy(adapter, binding),
            entry_route_diagnostics={},
            progress_path=tmp_path / "progress.json",
            result_path=tmp_path / "result.json",
            max_forwards_this_invocation=1,
            policy_batch_size=16,
        )


    binding["model_variant"] = "weight_ema"
    binding.pop("binding_sha256")
    binding["binding_sha256"] = canonical_sha256(binding)
    with pytest.raises(RuntimeError, match="ROUTE_OUTPUT"):
        run_resumable_random_access_val_evaluation_v1(
            model=model,
            entry_decision_representations=representations,
            adapter=adapter,
            checkpoint_binding=binding,
            entry_policy_decisions=_entry_policy(adapter, binding),
            entry_route_diagnostics={},
            progress_path=tmp_path / "progress2.json",
            result_path=tmp_path / "result2.json",
            max_forwards_this_invocation=1,
            policy_batch_size=16,
        )



def test_wall_window_pauses_then_resumes_despite_prior_elapsed_time(tmp_path):
    model, representations, adapter, contract = _fixture(
        thresholds=np.zeros((VAL_ENTRY_COHORT_SIZE, 2), dtype=np.float32),
        counts=np.ones(VAL_ENTRY_COHORT_SIZE, dtype=np.int64),
    )
    model = _with_route_outputs(model)
    adapter.contract["compute_guard"]["wall_limit_scope"] = "invocation"
    adapter.contract["compute_guard"]["max_wall_seconds"] = 0.5
    unsigned = dict(adapter.contract)
    unsigned.pop("contract_sha256")
    adapter.contract["contract_sha256"] = canonical_sha256(unsigned)
    binding = _checkpoint_binding(contract, adapter, tmp_path)
    common = dict(
        model=model, entry_decision_representations=representations, adapter=adapter,
        checkpoint_binding=binding, entry_policy_decisions=_entry_policy(adapter, binding),
        entry_route_diagnostics={}, progress_path=tmp_path / "wall-progress.json",
        result_path=tmp_path / "wall-result.json", max_forwards_this_invocation=10_000,
        policy_batch_size=16,
    )
    clock_values = iter([0.0, 1.0])
    paused = run_resumable_random_access_val_evaluation_v1(
        **common, monotonic=lambda: next(clock_values, 1.0),
    )
    assert paused["decision"] == "PAUSED_RESUMABLE"
    assert paused["pause_reason"] == "invocation_wall_limit"
    assert not common["result_path"].exists()
    completed = run_resumable_random_access_val_evaluation_v1(
        **common, monotonic=lambda: 0.0,
    )
    assert completed["decision"] == "PASS_COMPLETE"
    assert completed["entry_exit_policy_metrics"]["full_cohort_authoritative"] is True


def test_route_diagnostics_preserve_feature_scaling_contract() -> None:
    from gx1.contracts.unified_exit_random_access_val_evaluator_v1 import (
        accumulate_route_diagnostics_v1,
    )

    routes = {
        "exit_specialist_gate": torch.tensor([[0.25, 0.75]]),
        "exit_tf_gate": torch.tensor([[0.4, 0.6]]),
        "exit_family_tf_cooperation_gate": torch.full((1, 2, 2), 0.25),
        "exit_family_tf_feature_gate": torch.tensor([[[0.5, 1.5], [1.25, 0.75]]]),
    }
    observed = {}
    accumulate_route_diagnostics_v1(observed, routes)
    assert observed["exit_family_tf_feature_gate"]["max"] == 1.5
    for invalid in (-0.1, 2.1, float("nan")):
        bad = {**routes, "exit_family_tf_feature_gate": torch.full((1, 2, 2), invalid)}
        with pytest.raises(RuntimeError, match="UNIFIED_EXIT_VAL_ROUTE_(RANGE|OUTPUT)_INVALID"):
            accumulate_route_diagnostics_v1({}, bad)
    with pytest.raises(RuntimeError, match="UNIFIED_EXIT_VAL_ROUTE_RANGE_INVALID:exit_tf_gate"):
        accumulate_route_diagnostics_v1({}, {**routes, "exit_tf_gate": torch.tensor([[0.4, 1.1]])})


def test_saturated_feature_gates_remain_visible_quality_failures() -> None:
    from gx1.contracts.unified_exit_random_access_val_evaluator_v1 import (
        accumulate_route_diagnostics_v1, finalize_route_diagnostics_v1,
    )

    routes = {
        "exit_specialist_gate": torch.tensor([[0.25, 0.75]]),
        "exit_tf_gate": torch.tensor([[0.4, 0.6]]),
        "exit_family_tf_cooperation_gate": torch.full((1, 2, 2), 0.25),
        "exit_family_tf_feature_gate": torch.tensor([[[0.0, 2.0], [1.25, 0.75]]]),
    }
    observed = {}
    accumulate_route_diagnostics_v1(observed, routes)
    result = finalize_route_diagnostics_v1(observed)
    feature = result["routes"]["exit_family_tf_feature_gate"]
    assert feature["min"] == 0.0 and feature["max"] == 2.0
    quality = feature["feature_gate_quality"]
    assert quality["saturated_lower_element_count"] == 1
    assert quality["saturated_upper_element_count"] == 1
    assert quality["saturated_element_fraction"] == 0.5
    assert quality["open_range_quality_pass"] is False
    assert quality["candidate_admission_claimed"] is False


def test_native_shared_routes_count_active_states_once() -> None:
    from gx1.contracts.unified_exit_random_access_val_evaluator_v1 import (
        accumulate_route_diagnostics_v1, finalize_route_diagnostics_v1,
    )
    routes = {
        "exit_specialist_gate": torch.full((4, 1, 8), 1.0 / 8),
        "exit_tf_gate": torch.full((4, 1, 5), 1.0 / 5),
        "exit_family_tf_cooperation_gate": torch.full((4, 1, 5, 8), 1.0 / 40),
        "exit_family_tf_feature_gate": torch.ones(4, 1, 5, 176),
    }
    mask = np.asarray([[True, True], [True, False], [False, True], [False, False]])
    accumulators = {}
    accumulate_route_diagnostics_v1(accumulators, routes, active_side_mask=mask)
    for raw in finalize_route_diagnostics_v1(accumulators)["routes"].values():
        assert raw["batch_row_count"] == 3
        assert raw["observation_unit"] == "active_entry_state_shared_by_sides"
    with pytest.raises(RuntimeError, match="ROUTE_SHARED_STATE_SHAPE_INVALID"):
        accumulate_route_diagnostics_v1({}, {**routes, "exit_tf_gate": torch.ones(4, 2, 5) / 5}, active_side_mask=mask)
