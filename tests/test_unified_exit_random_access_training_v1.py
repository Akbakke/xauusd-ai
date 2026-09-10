from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn

from gx1.contracts.unified_exit_pilot_normalization_v1 import (
    build_physical_summary_sample_authority,
    fit_lifetime_summary_normalization,
)
from gx1.contracts.unified_exit_random_access_sampler_v1 import (
    schedule_random_access_entry_anchors,
)
from gx1.contracts.unified_exit_random_access_training_v1 import (
    collate_random_access_training_items,
    run_random_access_training_step,
)
from gx1.contracts.unified_exit_random_access_state_view_v1 import (
    _structured_sha256,
)
from tests.test_unified_exit_random_access_state_view_v1 import (
    _contract,
    _materialize,
    _sample,
)


def _normalization() -> dict:
    authority = build_physical_summary_sample_authority(
        successor_transition_count_by_entry=[600],
        source_lineage_sha256="e" * 64,
    )
    rows = authority["fit_row_count"]
    values = np.arange(rows * 7, dtype=np.float64).reshape(rows, 7)
    return fit_lifetime_summary_normalization(
        values=values, sample_authority=authority
    )


def _item(
    *, state_index: int = 0, successor_long_terminal: bool = False
) -> tuple[dict, dict]:
    contract = _contract()
    transition = _materialize(
        state_index,
        counts=((state_index + 2), 600),
        terminals=(successor_long_terminal, False),
    )
    anchor = _materialize(0, anchor=True)
    return contract, {
        "outer_batch_index": 1,
        "entry_row_index": 0,
        "transitions": [
            {"sample": _sample(contract, state_index), "state_view": transition}
        ],
        "anchor": {
            "sample": schedule_random_access_entry_anchors(
                sampler_contract=contract, epoch_index=0
            )[0],
            "state_view": anchor,
        },
        "entry_episode_binding_sha256": "b" * 64,
        "entry_fill_binding_sha256": "f" * 64,
        "first_state_bridge_witness_sha256": "a" * 64,
    }


def _collate(*, state_index: int = 0, successor_long_terminal: bool = False) -> dict:
    contract, item = _item(
        state_index=state_index,
        successor_long_terminal=successor_long_terminal,
    )
    view = item["transitions"][0]["state_view"]
    return collate_random_access_training_items(
        [item],
        outer_batch_size=3,
        sampler_contract=contract,
        normalization_artifact=_normalization(),
        expected_m1_source_sha256=view["m1_source_sha256"],
        expected_market_closure_authority_sha256=view[
            "market_closure_authority_sha256"
        ],
        expected_economic_step_manifest_sha256=view[
            "economic_step_manifest_sha256"
        ],
        expected_economics_objective_contract_sha256=view[
            "economics_objective_contract_sha256"
        ],
        device=torch.device("cpu"),
    )


class _ToyExitModel(nn.Module):
    def __init__(self, *, fixed: torch.Tensor | None = None) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.25))
        self.task_log_variances = nn.ParameterDict(
            {"unified_exit_action": nn.Parameter(torch.tensor(0.0))}
        )
        self.fixed = fixed
        self.calls = 0

    def forward_exit_random_access_batch(self, **kwargs):
        self.calls += 1
        token = kwargs["entry_decision_representation"]
        count = len(token)
        if self.fixed is not None:
            q = self.fixed[:count].to(token)
        else:
            q = (
                token[:, :1].reshape(count, 1, 1).expand(-1, 2, 2)
                + self.bias
            )
        return {
            "exit_action_q_bps": q,
            "exit_action_valid_mask": kwargs["action_valid_mask"],
        }


def test_collated_step_has_exact_call_counts_targets_masks_and_gradients() -> None:
    batch = _collate()
    model = _ToyExitModel()
    target = _ToyExitModel(
        fixed=torch.tensor(
            [
                [[10.0, 4.0], [8.0, 9.0]],
                [[5.0, 6.0], [7.0, 3.0]],
            ]
        )
    )
    target.requires_grad_(False).eval()
    entry = torch.tensor([[2.0], [3.0], [4.0]], requires_grad=True)
    result = run_random_access_training_step(
        model=model,
        target_model=target,
        entry_decision_representations=entry,
        target_entry_decision_representations=entry.detach().clone(),
        batch=batch,
        grad_accum_steps=1,
    )
    gamma = batch["elapsed_wall_clock_gamma"][0]
    expected = torch.tensor(
        [[[-0.1 + gamma * 10.0, 2.0], [-0.1 + gamma * 9.0, 3.0]]]
    )
    assert model.calls == 1
    assert target.calls == 1
    assert result["online_forward_calls"] == 1
    assert result["target_forward_calls"] == 1
    assert result["backward_calls"] == 1
    assert torch.allclose(result["targets"], expected, rtol=0.0, atol=1e-6)
    assert result["valid_mask"].all()
    assert model.bias.grad is not None and torch.isfinite(model.bias.grad)
    assert result["entry_gradients"][0].eq(0.0).all()
    assert result["entry_gradients"][2].eq(0.0).all()
    assert result["entry_gradients"][1].abs().sum() > 0.0
    assert result["entry_targets"].tolist() == [
        [0.0, 0.0, 0.0],
        [6.0, 7.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    assert result["entry_valid_mask"].tolist() == [
        [False, False, True],
        [True, True, True],
        [False, False, True],
    ]
    assert result["entry_bridge_binding"]["anchor_state_view_sha256"] == [
        batch["anchor_state_view_sha256"][0]
    ]


def test_target_must_be_frozen_and_eval() -> None:
    batch = _collate()
    model = _ToyExitModel()
    target = _ToyExitModel(fixed=torch.zeros(2, 2, 2))
    entry = torch.zeros(3, 1)
    with pytest.raises(RuntimeError, match="TARGET_MODEL_NOT_FROZEN_EVAL"):
        run_random_access_training_step(
            model=model,
            target_model=target,
            entry_decision_representations=entry,
            target_entry_decision_representations=entry,
            batch=batch,
            grad_accum_steps=1,
        )


def test_successor_terminal_hold_is_excluded_from_bellman_max() -> None:
    batch = _collate(state_index=530, successor_long_terminal=True)
    model = _ToyExitModel()
    target = _ToyExitModel(
        fixed=torch.tensor(
            [
                [[100.0, 4.0], [8.0, 9.0]],
                [[5.0, 6.0], [7.0, 3.0]],
            ]
        )
    )
    target.requires_grad_(False).eval()
    entry = torch.tensor([[2.0], [3.0], [4.0]])
    result = run_random_access_training_step(
        model=model,
        target_model=target,
        entry_decision_representations=entry,
        target_entry_decision_representations=entry,
        batch=batch,
        grad_accum_steps=1,
    )
    gamma = batch["elapsed_wall_clock_gamma"][0]
    assert torch.allclose(
        result["targets"][0, 0, 0], -0.1 + gamma * 4.0, atol=1e-6
    )


def test_collator_rejects_tampered_anchor() -> None:
    contract, item = _item()
    view = item["transitions"][0]["state_view"]
    tampered_normalization = _normalization()
    tampered_normalization["val_fit_rows"] = 1
    with pytest.raises(RuntimeError, match="NORMALIZATION"):
        collate_random_access_training_items(
            [item],
            outer_batch_size=3,
            sampler_contract=contract,
            normalization_artifact=tampered_normalization,
            expected_m1_source_sha256=view["m1_source_sha256"],
            expected_market_closure_authority_sha256=view[
                "market_closure_authority_sha256"
            ],
            expected_economic_step_manifest_sha256=view[
                "economic_step_manifest_sha256"
            ],
            expected_economics_objective_contract_sha256=view[
                "economics_objective_contract_sha256"
            ],
            device=torch.device("cpu"),
        )
    broken = copy.deepcopy(item)
    broken["anchor"]["state_view"]["loss_weight"] = 1.0
    with pytest.raises(RuntimeError, match="STATE_VIEW_.*INVALID"):
        collate_random_access_training_items(
            [broken],
            outer_batch_size=3,
            sampler_contract=contract,
            normalization_artifact=_normalization(),
            expected_m1_source_sha256=view["m1_source_sha256"],
            expected_market_closure_authority_sha256=view[
                "market_closure_authority_sha256"
            ],
            expected_economic_step_manifest_sha256=view[
                "economic_step_manifest_sha256"
            ],
            expected_economics_objective_contract_sha256=view[
                "economics_objective_contract_sha256"
            ],
            device=torch.device("cpu"),
        )
    forged = dict(item)
    forged_pair = dict(item["transitions"][0])
    forged_view = dict(forged_pair["state_view"])
    forged_current = dict(forged_view["current"])
    forged_successor = dict(forged_view["successor"])
    forged_current["state_index"] = 99
    forged_successor["state_index"] = 100
    forged_view["current"] = forged_current
    forged_view["successor"] = forged_successor
    forged_view.pop("state_view_sha256")
    forged_view["state_view_sha256"] = _structured_sha256(forged_view)
    forged_pair["state_view"] = forged_view
    forged["transitions"] = [forged_pair]
    with pytest.raises(RuntimeError, match="TRANSITION_LINKAGE_INVALID"):
        collate_random_access_training_items(
            [forged],
            outer_batch_size=3,
            sampler_contract=contract,
            normalization_artifact=_normalization(),
            expected_m1_source_sha256=view["m1_source_sha256"],
            expected_market_closure_authority_sha256=view[
                "market_closure_authority_sha256"
            ],
            expected_economic_step_manifest_sha256=view[
                "economic_step_manifest_sha256"
            ],
            expected_economics_objective_contract_sha256=view[
                "economics_objective_contract_sha256"
            ],
            device=torch.device("cpu"),
        )


def test_canonical_trainer_rejects_old_chunk_adapter_boundary() -> None:
    from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
        _episode_native_exit_train_v2,
    )

    class _Dataset:
        _unified_exit_lifecycle_v2 = object()

    with pytest.raises(RuntimeError, match="RANDOM_ACCESS_V2_ADAPTER_BOUNDARY_REQUIRED"):
        _episode_native_exit_train_v2(
            model=_ToyExitModel(),
            target_model=_ToyExitModel().requires_grad_(False).eval(),
            entry_decision_representations=torch.zeros(1, 1),
            target_entry_decision_representations=torch.zeros(1, 1),
            entry_row_indices=torch.zeros(1, dtype=torch.long),
            dataset=_Dataset(),
            device=torch.device("cpu"),
            grad_accum_steps=1,
            exit_cooperation_gate_epoch={},
            exit_feature_tf_gate_epoch={},
        )
