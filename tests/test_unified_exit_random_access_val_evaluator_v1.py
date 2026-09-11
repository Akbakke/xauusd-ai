from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from gx1.contracts.unified_exit_random_access_val_evaluator_v1 import (
    PAUSE_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    run_resumable_random_access_val_evaluation_v1,
)
from tests.test_unified_exit_random_access_val_rollout_v1 import (
    VAL_ENTRY_COHORT_SIZE,
    _fixture,
)


def _with_route_outputs(model):
    original = model.forward_exit_random_access_batch

    def forward(**inputs):
        result = original(**inputs)
        batch = result["exit_action_q_bps"].shape[0]
        result.update(
            {
                "exit_specialist_gate": torch.full((batch, 2, 3), 1.0 / 3.0),
                "exit_tf_gate": torch.full((batch, 2, 2), 0.5),
                "exit_family_tf_cooperation_gate": torch.full(
                    (batch, 2, 2, 3), 1.0 / 6.0
                ),
                "exit_family_tf_feature_gate": torch.full((batch, 2, 2, 3), 0.5),
            }
        )
        return result

    model.forward_exit_random_access_batch = forward
    return model


def _checkpoint_binding(contract):
    return {
        "schema_version": "gx1_test_selected_v2_checkpoint_binding_v1",
        "model_variant": "weight_ema",
        "model_state_sha256": contract["model_state_sha256"],
        "checkpoint_file_sha256": contract["checkpoint_file_sha256"],
        "checkpoint_path": "/immutable/selected-v2.pt",
        "global_step": 1024,
        "epoch_index": 0,
        "test_data_used": False,
    }


def test_full_cohort_pause_resume_is_semantically_exact(tmp_path) -> None:
    thresholds = np.ones((VAL_ENTRY_COHORT_SIZE, 2), dtype=np.float32)
    counts = np.full(VAL_ENTRY_COHORT_SIZE, 2, dtype=np.int64)
    model, representations, adapter, contract = _fixture(
        thresholds=thresholds,
        counts=counts,
    )
    model = _with_route_outputs(model)
    binding = _checkpoint_binding(contract)
    progress = tmp_path / "progress.json"
    result = tmp_path / "result.json"

    paused = run_resumable_random_access_val_evaluation_v1(
        model=model,
        entry_decision_representations=representations,
        adapter=adapter,
        checkpoint_binding=binding,
        entry_route_diagnostics={
            "semantics": "observation_only",
            "causal_feature_importance_claimed": False,
        },
        progress_path=progress,
        result_path=result,
        max_forwards_this_invocation=1,
        progress_interval_forwards=1,
    )
    assert paused["schema_version"] == PAUSE_SCHEMA_VERSION
    assert paused["decision"] == "PAUSED_RESUMABLE"
    assert paused["next_state_index"] == 1
    assert not result.exists()

    completed = run_resumable_random_access_val_evaluation_v1(
        model=model,
        entry_decision_representations=representations,
        adapter=adapter,
        checkpoint_binding=binding,
        entry_route_diagnostics={
            "semantics": "observation_only",
            "causal_feature_importance_claimed": False,
        },
        progress_path=progress,
        result_path=result,
        max_forwards_this_invocation=2,
        progress_interval_forwards=1,
    )
    assert completed["schema_version"] == RESULT_SCHEMA_VERSION
    assert completed["decision"] == "PASS_COMPLETE"
    assert completed["entry_pair_cohort_size"] == VAL_ENTRY_COHORT_SIZE
    assert completed["side_trade_count"] == VAL_ENTRY_COHORT_SIZE * 2
    assert completed["exited_side_trade_count"] == VAL_ENTRY_COHORT_SIZE * 2
    assert completed["model_forward_count"] == 2
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


def test_checkpoint_variant_and_route_evidence_fail_closed(tmp_path) -> None:
    thresholds = np.zeros((VAL_ENTRY_COHORT_SIZE, 2), dtype=np.float32)
    counts = np.ones(VAL_ENTRY_COHORT_SIZE, dtype=np.int64)
    model, representations, adapter, contract = _fixture(
        thresholds=thresholds,
        counts=counts,
    )
    binding = _checkpoint_binding(contract)
    binding["model_variant"] = "online"
    with pytest.raises(RuntimeError, match="EXECUTION_BINDING"):
        run_resumable_random_access_val_evaluation_v1(
            model=model,
            entry_decision_representations=representations,
            adapter=adapter,
            checkpoint_binding=binding,
            entry_route_diagnostics={},
            progress_path=tmp_path / "progress.json",
            result_path=tmp_path / "result.json",
            max_forwards_this_invocation=1,
        )

    binding["model_variant"] = "weight_ema"
    with pytest.raises(RuntimeError, match="ROUTE_OUTPUT"):
        run_resumable_random_access_val_evaluation_v1(
            model=model,
            entry_decision_representations=representations,
            adapter=adapter,
            checkpoint_binding=binding,
            entry_route_diagnostics={},
            progress_path=tmp_path / "progress2.json",
            result_path=tmp_path / "result2.json",
            max_forwards_this_invocation=1,
        )
