from __future__ import annotations

import copy

import pytest
import torch

from gx1.contracts import unified_exit_random_access_training_v1 as training
from tests.test_unified_exit_frozen_policy_trace_v1 import _batch, relative_fixture
from tests.test_liquidation_relative_learning_v1 import _TwoRowHead


@pytest.mark.parametrize("rows", [1, 80, 81, 336])
def test_frozen_target_chunking_preserves_nested_inputs_row_order_and_tail(rows):
    seen = []

    class Teacher:
        def forward_exit_random_access_batch(self, **inputs):
            assert not torch.is_grad_enabled()
            assert inputs["liquidation_relative_values"] is True
            ids = inputs["entry_decision_representation"][:, 0]
            torch.testing.assert_close(inputs["exit_mtf_histories"]["m5"][:, 0], ids + 10)
            torch.testing.assert_close(inputs["action_valid_mask"][:, 0, 0], ids.remainder(2).bool())
            seen.append(len(ids))
            return {"exit_action_q_bps": ids[:, None, None].expand(-1, 2, 2) + inputs["exit_mtf_histories"]["m5"][:, :, None]}

    ids = torch.arange(rows, dtype=torch.float32)
    mask = torch.ones(rows, 2, 2, dtype=torch.bool)
    mask[:, 0, 0] = ids.remainder(2).bool()
    inputs = {"entry_decision_representation": ids[:, None],
              "exit_mtf_histories": {"m5": (ids + 10)[:, None]},
              "action_valid_mask": mask, "liquidation_relative_values": True}
    with torch.no_grad():
        expected = Teacher().forward_exit_random_access_batch(**inputs)["exit_action_q_bps"]
        seen.clear()
        actual = training._frozen_exit_q_in_base_batches(target_model=Teacher(), model_inputs=inputs, rows_per_batch=80)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert sum(seen) == rows and max(seen) <= 80
    assert seen[-1] == ((rows - 1) % 80) + 1


@pytest.mark.parametrize("rows_per_batch", [0, -1, True])
def test_target_chunking_rejects_invalid_working_batch(rows_per_batch):
    with pytest.raises(RuntimeError, match="TARGET_BATCH_SIZE_INVALID"):
        training._frozen_exit_q_in_base_batches(target_model=None,
            model_inputs={"entry_decision_representation": torch.ones(2, 1)}, rows_per_batch=rows_per_batch)


def test_target_chunking_rejects_misaligned_nested_rows():
    with pytest.raises(RuntimeError, match="TARGET_BATCH_LAYOUT_INVALID"):
        training._frozen_exit_q_in_base_batches(target_model=None,
            model_inputs={"entry_decision_representation": torch.ones(2, 1),
                          "exit_mtf_histories": {"m5": torch.ones(1, 3)}}, rows_per_batch=1)


def test_chunking_preserves_trace_targets_entry_anchor_and_one_backward(relative_fixture, monkeypatch):
    batch, _ = _batch()
    outcomes, gradients = [], []
    for chunked in (True, False):
        if not chunked:
            monkeypatch.setattr(training, "_frozen_exit_q_in_base_batches",
                lambda *, target_model, model_inputs, rows_per_batch:
                    target_model.forward_exit_random_access_batch(**model_inputs)["exit_action_q_bps"])
        model = _TwoRowHead()
        target = copy.deepcopy(model).requires_grad_(False).eval()
        entry = torch.tensor([[2.], [3.], [4.]], requires_grad=True)
        result = training.run_random_access_training_step(model=model, target_model=target,
            entry_decision_representations=entry, target_entry_decision_representations=entry.detach(),
            batch=batch, grad_accum_steps=1)
        assert model.calls == 1 and result["backward_calls"] == 1
        assert target.calls == (3 if chunked else 1)
        assert all(p.grad is None for p in target.parameters())
        outcomes.append(result)
        gradients.append({k: p.grad.detach().clone() for k, p in model.named_parameters()})
    for key in ("targets", "entry_targets", "entry_gradients", "entry_valid_mask"):
        torch.testing.assert_close(outcomes[0][key], outcomes[1][key], rtol=0, atol=0)
    assert outcomes[0]["entry_bridge_binding"] == outcomes[1]["entry_bridge_binding"]
    for key in gradients[0]:
        torch.testing.assert_close(gradients[0][key], gradients[1][key], rtol=0, atol=0)
