"""A completed cohort can include unselected, naturally censored alternatives."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gx1.contracts.unified_exit_entry_policy_evaluation_v1 import build_entry_policy_decisions
from gx1.contracts.unified_exit_random_access_val_evaluator_v1 import (
    _finalize_result, _new_progress, canonical_sha256, accumulate_route_diagnostics_v1,
    require_random_access_val_evaluation_result_v1,
)
from tests.test_unified_exit_random_access_val_evaluator_v1 import _checkpoint_binding
from gx1.contracts.unified_exit_random_access_val_rollout_v1 import VAL_ENTRY_COHORT_SIZE


@pytest.mark.parametrize("side,status,flat,authoritative", [
    (1, "RIGHT_CENSORED_SPLIT_END", False, True),
    (0, "RIGHT_CENSORED_SPLIT_END", False, False),
    (0, "RIGHT_CENSORED_SPLIT_END", True, True),
    (1, "TRUNCATED_COMPUTE_GUARD", False, False),
])
def test_actual_policy_censoring_round_trip(tmp_path, side, status, flat, authoritative):
    contract = {"model_state_sha256": "d" * 64, "contract_sha256": "e" * 64}
    adapter = SimpleNamespace(
        contract=contract,
        entries=[{"entry_row_index": i} for i in range(VAL_ENTRY_COHORT_SIZE)],
    )
    binding = _checkpoint_binding(contract, adapter, tmp_path)
    q = np.tile(np.array([2., 1., 0.], dtype=np.float32), (VAL_ENTRY_COHORT_SIZE, 1))
    if flat:
        q[0] = [0., 1., 2.]
    policy = build_entry_policy_decisions(
        predicted_q_bps=q, entry_row_indices=list(range(VAL_ENTRY_COHORT_SIZE)),
        checkpoint_binding_sha256=binding["binding_sha256"],
    )
    execution = {"entry_policy_sha256": policy["policy_sha256"]}
    execution["execution_contract_sha256"] = canonical_sha256(execution)
    progress = _new_progress(
        contract_sha256=contract["contract_sha256"],
        checkpoint_binding_sha256=binding["binding_sha256"],
    )
    accumulate_route_diagnostics_v1(progress["route_accumulators"], {
        "exit_specialist_gate": torch.full((1, 1, 2), 0.5),
        "exit_tf_gate": torch.full((1, 1, 2), 0.5),
        "exit_family_tf_cooperation_gate": torch.full((1, 1, 2, 2), 0.25),
        "exit_family_tf_feature_gate": torch.full((1, 1, 2, 2), 1.5),
    })
    for pair in progress["trade_accumulators"]:
        for trade in pair:
            trade.update(status="EXITED", exit_state_index=0, undiscounted_net_cash_pnl_bps=1.)
    progress["trade_accumulators"][0][side]["status"] = status
    result = _finalize_result(
        progress, adapter=adapter, checkpoint_binding=binding, execution_contract=execution,
        entry_route_diagnostics={}, entry_policy_decisions=policy, guard_reason=None,
    )
    checked = require_random_access_val_evaluation_result_v1(
        result, rollout_contract_sha256=contract["contract_sha256"],
        checkpoint_binding_sha256=binding["binding_sha256"],
        execution_contract_sha256=execution["execution_contract_sha256"],
    )
    assert checked["full_cohort_policy_metrics_authoritative"] is authoritative
    assert checked["policy_economics"]["authoritative_complete_policy_metric"] is False
    assert checked["entry_exit_policy_metrics"]["net_bps_sum"] == (
        float(VAL_ENTRY_COHORT_SIZE - int(flat)) if authoritative else None
    )
    assert checked["trade_outcomes"][side]["status"] == status
    assert checked["rollout_execution_complete"] is (not status.startswith("TRUNCATED_"))
