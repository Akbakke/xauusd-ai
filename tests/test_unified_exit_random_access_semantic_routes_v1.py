from __future__ import annotations

import pytest
import torch

from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_PATH_FEATURE_DIM,
)
from tests.test_entry_v10_ctx_model_shapes import (
    _make_exit_episode_inputs,
    _make_model,
)


def _random_access_inputs(batch_size: int = 2) -> dict:
    base = _make_exit_episode_inputs(state_count=1, batch_size=batch_size)
    tail_rows = 5
    lengths = torch.tensor([5, 3][:batch_size], dtype=torch.long)
    path = torch.randn(
        batch_size, 2, tail_rows, UNIFIED_EXIT_PATH_FEATURE_DIM
    )
    for index, length in enumerate(lengths.tolist()):
        path[index, :, length:] = 0.0
    return {
        "entry_decision_representation": base[
            "entry_decision_representation"
        ],
        "m1_local_history_x": base["exit_local_history_x"],
        "state_ctx_cat": base["exit_state_ctx_cat"][:, 0],
        "state_ctx_cont": base["exit_state_ctx_cont"][:, 0],
        "trade_path_tail_x": path,
        "trade_path_lengths": lengths,
        "normalized_lifetime_summary_x": torch.randn(batch_size, 2, 7),
        "exit_mtf_histories": base["exit_mtf_histories"],
        "exit_mtf_gathers": {
            name: gather[:, :1]
            for name, gather in base["exit_mtf_gathers"].items()
        },
        "exit_mtf_history_lengths": base["exit_mtf_history_lengths"],
        "action_valid_mask": torch.ones(
            batch_size, 2, 2, dtype=torch.bool
        ),
    }


def test_random_access_route_reaches_every_family_timeframe_and_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bounded state route must preserve every semantic evidence lane."""

    torch.manual_seed(20260911)
    model = _make_model(dropout=0.0).train()
    inputs = _random_access_inputs(batch_size=2)
    for name in (
        "entry_decision_representation",
        "m1_local_history_x",
        "state_ctx_cont",
        "trade_path_tail_x",
        "normalized_lifetime_summary_x",
    ):
        inputs[name] = inputs[name].detach().clone().requires_grad_(True)
    inputs["exit_mtf_histories"] = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in inputs["exit_mtf_histories"].items()
    }

    def forbidden_legacy_route(*_args, **_kwargs):
        raise AssertionError("random-access forward invoked a legacy prefix route")

    monkeypatch.setattr(model, "forward_exit_episode", forbidden_legacy_route)
    monkeypatch.setattr(
        model, "forward_exit_incremental_prefix", forbidden_legacy_route
    )
    calls: dict[str, int] = {}

    def counted(name: str):
        def observe(_module, _args, _output):
            calls[name] = calls.get(name, 0) + 1

        return observe

    handles = [
        model.exit_episode_global_gru.register_forward_hook(
            counted("local_global")
        ),
        model.exit_episode_path_gru.register_forward_hook(counted("path")),
        model.exit_random_access_summary_proj.register_forward_hook(
            counted("summary")
        ),
        model.exit_random_access_fuse.register_forward_hook(counted("fuse")),
        model.head_exit_action.register_forward_hook(counted("q_head")),
    ]
    for family in model._specialist_names:
        handles.extend(
            (
                model.exit_episode_family_gru[family].register_forward_hook(
                    counted(f"local/{family}")
                ),
                model.exit_episode_mtf_family_gru[family].register_forward_hook(
                    counted(f"mtf/{family}")
                ),
            )
        )
    try:
        output = model.forward_exit_random_access_batch(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    expected_calls = {
        "local_global",
        "path",
        "summary",
        "fuse",
        "q_head",
        *(f"local/{family}" for family in model._specialist_names),
        *(f"mtf/{family}" for family in model._specialist_names),
    }
    assert calls == {name: 1 for name in expected_calls}
    weights = torch.tensor(
        [
            [[1.0, -0.7], [0.4, 1.3]],
            [[-0.8, 0.6], [1.1, -0.2]],
        ]
    )
    (output["exit_action_q_bps"] * weights).sum().backward()

    for name in (
        "entry_decision_representation",
        "m1_local_history_x",
        "state_ctx_cont",
    ):
        gradient = inputs[name].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum().item() > 0.0
    for name in ("trade_path_tail_x", "normalized_lifetime_summary_x"):
        gradient = inputs[name].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        side_norm = gradient.abs().sum(dim=tuple(range(2, gradient.ndim)))
        assert torch.all(side_norm > 0.0)

    local_gradient = inputs["m1_local_history_x"].grad
    assert local_gradient is not None
    for family in model._specialist_names:
        local_indices = getattr(model, f"specialist_idx_{family}")
        assert (
            local_gradient.index_select(2, local_indices).abs().sum().item()
            > 0.0
        )
        for parameter in (
            model.exit_episode_family_gru[family].weight_ih_l0,
            model.exit_episode_mtf_family_gru[family].weight_ih_l0,
        ):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum().item() > 0.0

    for history in inputs["exit_mtf_histories"].values():
        assert history.grad is not None
        assert torch.isfinite(history.grad).all()
        for family in model._specialist_names:
            indices = getattr(model, f"multi_tf_specialist_idx_{family}")
            assert history.grad.index_select(2, indices).abs().sum().item() > 0.0

    cooperation = output["exit_family_tf_cooperation_gate"]
    assert cooperation.shape[2:] == (5, len(model._specialist_names))
    assert torch.all(cooperation > 0.0)
    assert torch.allclose(
        cooperation.sum(dim=(2, 3)),
        torch.ones_like(cooperation.sum(dim=(2, 3))),
    )
