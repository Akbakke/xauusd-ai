from __future__ import annotations

import ast
import copy
import inspect
import io
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from gx1.contracts.entry_model_native_joint_task_weighting_v1 import (
    FORMULA,
    JOINT_TASK_NAMES,
    joint_task_weighting_metadata,
    require_joint_task_weighting_metadata,
)
from gx1.contracts.unified_exit_fitted_q_v1 import (
    unified_exit_first_state_side_values,
)
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


ROOT = Path(__file__).resolve().parents[1]

# One episode, two sides, two Exit states, two actions (HOLD, EXIT_NOW).
# The final state is the forced terminal one: only EXIT_NOW is executable
# there, which is exactly what the fitted-Q policy replay owner requires.
_EPISODE_ACTION_VALID = np.array(
    [
        [[True, True], [False, True]],
        [[True, True], [False, True]],
    ],
    dtype=np.bool_,
)
_EPISODE_STATE_VALID = np.ones((2, 2), dtype=np.bool_)
_EPISODE_PACK_SHA256 = "6" * 64

# Frozen raw-bps target Q, shaped (episode, side, state, action).  Read against
# _EPISODE_ACTION_VALID the four flattened (side, state) rows are, in order:
#   side 0 state 0: both actions valid and exactly equal   -> target tie
#   side 0 state 1: terminal, EXIT_NOW only
#   side 1 state 0: both valid, EXIT_NOW strictly greater   -> unique argmax
#   side 1 state 1: terminal, EXIT_NOW only
# so exactly one row is target-equivalent, exactly one row has HOLD among the
# target-greedy actions, and all four rows have EXIT_NOW among them.
_TARGET_Q = torch.tensor(
    [[[[0.0, 0.0], [0.0, 2.0]], [[1.0, 3.0], [0.0, 5.0]]]]
)
_EXPECTED_POPULATION_ROWS = int(_EPISODE_STATE_VALID.sum())
_EXPECTED_Q_VALID_CELLS = int(_EPISODE_ACTION_VALID.sum())
_EXPECTED_TARGET_EQUIVALENT_ROWS = 1
_EXPECTED_HOLD_TARGET_GREEDY_ROWS = 1
_EXPECTED_EXIT_NOW_TARGET_GREEDY_ROWS = 4


class _TaskWeights(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.task_log_variances = torch.nn.ParameterDict(
            {
                name: torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
                for name in JOINT_TASK_NAMES
            }
        )


class _ExitDataset:
    def __init__(self, episode: dict[str, object]) -> None:
        self._unified_exit_lifecycle = object()
        self._episode = episode

    def materialize_full_exit_episode(self, entry_row_index: int):
        assert entry_row_index == 7
        return self._episode


class _ExitModel(_TaskWeights):
    def __init__(self) -> None:
        super().__init__()
        self.exit_head = torch.nn.Linear(4, 2)
        self.forwarded_rows = 0


def _episode_fixture() -> dict[str, object]:
    return {
        "snap": torch.tensor(
            [[[0.1, 0.2], [0.3, -0.1]], [[-0.2, 0.4], [0.5, 0.2]]]
        ),
        "exit_action_valid_mask": _EPISODE_ACTION_VALID.copy(),
        "exit_state_valid_mask": _EPISODE_STATE_VALID.copy(),
        # Raw-bps EXIT_NOW reward per side and Exit state.
        "exit_now_reward_bps": np.array(
            [[3.0, -1.0], [-2.0, 4.0]], dtype=np.float32
        ),
        "entry_row_index": 7,
        "episode_pack_sha256": _EPISODE_PACK_SHA256,
        "exit_state_row_time_ns": np.array(
            [1_700_000_000_000_000_000, 1_700_000_060_000_000_000],
            dtype=np.int64,
        ),
        "exit_entry_bid_ask": np.array(
            [[1999.9, 2000.1], [1999.9, 2000.1]], dtype=np.float32
        ),
    }


def _episode_masks(episode, device: torch.device):
    valid = torch.from_numpy(
        np.asarray(episode["exit_action_valid_mask"], dtype=np.bool_)
    ).unsqueeze(0).to(device)
    state_valid = torch.from_numpy(
        np.asarray(episode["exit_state_valid_mask"], dtype=np.bool_)
    ).unsqueeze(0).to(device)
    return valid, state_valid


def _patch_episode_owners(monkeypatch: pytest.MonkeyPatch, targets: torch.Tensor):
    def forward(*, model, entry_decision_representation, episode, **_kwargs):
        snap = episode["snap"].to(entry_decision_representation.device)
        token = entry_decision_representation.expand(4, -1)
        model.forwarded_rows += 4
        q = model.exit_head(torch.cat((token, snap.reshape(4, 2)), dim=1)).reshape(
            1, 2, 2, 2
        )
        valid, state_valid = _episode_masks(episode, q.device)
        terminal = torch.zeros_like(state_valid)
        lengths = torch.full((1, 2), 2, dtype=torch.long)
        return q, valid, state_valid, terminal, lengths

    def _target_masks(episodes):
        assert len(episodes) == 1
        return _episode_masks(episodes[0], targets.device)

    def fitted_targets(*, episode, **_kwargs):
        valid, _state_valid = _episode_masks(episode, targets.device)
        terminal = torch.zeros_like(targets[..., 0], dtype=torch.bool)
        return targets, valid, terminal

    def fitted_targets_batch(*, episodes, **_kwargs):
        valid, state_valid = _target_masks(episodes)
        terminal = torch.zeros_like(targets[..., 0], dtype=torch.bool)
        # The production owner derives the Entry bridge value through this
        # exact contract function; the fake must not invent its own bridge.
        first_side_values = unified_exit_first_state_side_values(
            frozen_target_q_bps=targets,
            action_valid_mask=valid,
            state_valid_mask=state_valid,
        )
        return targets, valid, terminal, first_side_values, state_valid[..., 0]

    monkeypatch.setattr(trainer, "_forward_unified_exit_episode_pack", forward)
    monkeypatch.setattr(trainer, "_fitted_q_targets_for_episode", fitted_targets)
    def forward_batch(*, model, entry_decision_representations, episodes, **kwargs):
        assert len(episodes) == 1
        return forward(
            model=model,
            entry_decision_representation=entry_decision_representations,
            episode=episodes[0],
            **kwargs,
        )

    monkeypatch.setattr(
        trainer, "_forward_unified_exit_episode_batch", forward_batch
    )
    monkeypatch.setattr(
        trainer, "_fitted_q_targets_for_episode_batch", fitted_targets_batch
    )


@pytest.mark.parametrize("exit_action_forward_chunk_rows", [None, 1])
def test_full_exit_streaming_matches_monolithic_objective_and_retains_ties(
    monkeypatch: pytest.MonkeyPatch,
    exit_action_forward_chunk_rows: int | None,
) -> None:
    torch.manual_seed(9)
    target_q = _TARGET_Q.clone()
    _patch_episode_owners(monkeypatch, target_q)
    episode = _episode_fixture()
    dataset = _ExitDataset(episode)
    streamed = _ExitModel()
    monolithic = _ExitModel()
    monolithic.load_state_dict(streamed.state_dict(), strict=True)
    streamed.task_log_variances["unified_exit_action"].data.fill_(0.2)
    monolithic.task_log_variances["unified_exit_action"].data.fill_(0.2)
    streamed_token = torch.tensor([[0.4, -0.3]], requires_grad=True)
    monolithic_token = streamed_token.detach().clone().requires_grad_(True)

    (
        entry_gradient,
        stats,
        entry_targets,
        entry_valid,
    ) = trainer._train_unified_exit_full_population(
        model=streamed,
        target_model=copy.deepcopy(streamed).eval(),
        entry_decision_representations=streamed_token,
        target_entry_decision_representations=streamed_token.detach(),
        entry_row_indices=torch.tensor([7]),
        dataset=dataset,
        device=torch.device("cpu"),
        grad_accum_steps=1,
        exit_cooperation_gate_epoch=None,
        exit_feature_tf_gate_epoch=None,
        exit_action_forward_chunk_rows=exit_action_forward_chunk_rows,
    )
    streamed_objective = (
        torch.exp(
            -streamed.task_log_variances["unified_exit_action"].detach()
        )
        * torch.tensor(stats["raw_loss"])
        + streamed.task_log_variances["unified_exit_action"]
        + (streamed_token * entry_gradient).sum()
    )
    streamed_objective.backward()

    snap = episode["snap"].reshape(4, 2)
    q_targets = target_q.reshape(4, 2)
    valid = torch.from_numpy(_EPISODE_ACTION_VALID.reshape(4, 2))
    q_values = monolithic.exit_head(
        torch.cat([monolithic_token.expand(4, -1), snap], dim=1)
    )
    raw_loss = torch.nn.functional.mse_loss(q_values[valid], q_targets[valid])
    exact = (
        torch.exp(-monolithic.task_log_variances["unified_exit_action"])
        * raw_loss
        + monolithic.task_log_variances["unified_exit_action"]
    )
    exact.backward()

    assert stats["population_rows"] == _EXPECTED_POPULATION_ROWS
    assert stats["q_valid_cells"] == _EXPECTED_Q_VALID_CELLS
    assert (
        stats["target_equivalent_action_rows"]
        == _EXPECTED_TARGET_EQUIVALENT_ROWS
    )
    assert streamed.forwarded_rows == _EXPECTED_POPULATION_ROWS
    # The streamed path must also hand back the frozen Entry bridge it derived
    # from the same episode, with FLAT always valid and both sides eligible.
    assert entry_valid.tolist() == [[True, True, True]]
    assert torch.allclose(
        entry_targets[:, :2],
        unified_exit_first_state_side_values(
            frozen_target_q_bps=target_q,
            action_valid_mask=torch.from_numpy(_EPISODE_ACTION_VALID).unsqueeze(0),
            state_valid_mask=torch.from_numpy(_EPISODE_STATE_VALID).unsqueeze(0),
        ),
    )
    assert not entry_targets.requires_grad
    assert torch.allclose(streamed_token.grad, monolithic_token.grad, atol=1e-6)
    assert torch.allclose(
        streamed.exit_head.weight.grad,
        monolithic.exit_head.weight.grad,
        atol=1e-6,
    )
    assert torch.allclose(
        streamed.task_log_variances["unified_exit_action"].grad,
        monolithic.task_log_variances["unified_exit_action"].grad,
        atol=1e-6,
    )


def test_attended_exit_chunking_streams_complete_episode_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _episode_fixture()
    second = {**_episode_fixture(), "entry_row_index": 8}

    class _TwoExitDataset:
        def __init__(self) -> None:
            self._unified_exit_lifecycle = object()
            self._episodes = {7: first, 8: second}

        def materialize_full_exit_episode(self, entry_row_index: int):
            return self._episodes[entry_row_index]

    forward_group_sizes: list[int] = []

    def forward_batch(*, model, entry_decision_representations, episodes, device, **_kwargs):
        forward_group_sizes.append(len(episodes))
        tokens = entry_decision_representations.repeat_interleave(4, dim=0)
        snapshots = torch.cat(
            [episode["snap"].reshape(4, 2) for episode in episodes], dim=0
        ).to(device)
        model.forwarded_rows += int(tokens.shape[0])
        q_values = model.exit_head(torch.cat((tokens, snapshots), dim=1)).reshape(
            len(episodes), 2, 2, 2
        )
        valid = torch.from_numpy(
            np.stack(
                [episode["exit_action_valid_mask"] for episode in episodes], axis=0
            )
        ).to(device)
        state_valid = torch.from_numpy(
            np.stack(
                [episode["exit_state_valid_mask"] for episode in episodes], axis=0
            )
        ).to(device)
        terminal = torch.zeros_like(state_valid)
        lengths = torch.full((len(episodes), 2), 2, dtype=torch.long, device=device)
        return q_values, valid, state_valid, terminal, lengths

    def fitted_targets_batch(*, episodes, device, **_kwargs):
        valid = torch.from_numpy(
            np.stack(
                [episode["exit_action_valid_mask"] for episode in episodes], axis=0
            )
        ).to(device)
        state_valid = torch.from_numpy(
            np.stack(
                [episode["exit_state_valid_mask"] for episode in episodes], axis=0
            )
        ).to(device)
        targets = _TARGET_Q.to(device).expand(len(episodes), -1, -1, -1)
        terminal = torch.zeros_like(state_valid)
        first_values = unified_exit_first_state_side_values(
            frozen_target_q_bps=targets,
            action_valid_mask=valid,
            state_valid_mask=state_valid,
        )
        return targets, valid, terminal, first_values, state_valid[..., 0]

    monkeypatch.setattr(trainer, "_forward_unified_exit_episode_batch", forward_batch)
    monkeypatch.setattr(
        trainer, "_fitted_q_targets_for_episode_batch", fitted_targets_batch
    )
    model = _ExitModel()
    model.task_log_variances["unified_exit_action"].data.fill_(0.2)
    token = torch.tensor([[0.4, -0.3], [0.2, 0.1]], requires_grad=True)
    (
        entry_gradient,
        stats,
        entry_targets,
        entry_valid,
    ) = trainer._train_unified_exit_full_population(
        model=model,
        target_model=copy.deepcopy(model).eval(),
        entry_decision_representations=token,
        target_entry_decision_representations=token.detach(),
        entry_row_indices=torch.tensor([7, 8]),
        dataset=_TwoExitDataset(),
        device=torch.device("cpu"),
        grad_accum_steps=1,
        exit_cooperation_gate_epoch=None,
        exit_feature_tf_gate_epoch=None,
        exit_action_forward_chunk_rows=1,
    )

    assert forward_group_sizes == [1, 1]
    assert model.forwarded_rows == 2 * _EXPECTED_POPULATION_ROWS
    assert stats["population_rows"] == 2 * _EXPECTED_POPULATION_ROWS
    assert stats["q_valid_cells"] == 2 * _EXPECTED_Q_VALID_CELLS
    assert entry_gradient.shape == token.shape
    assert entry_targets.shape == (2, 3)
    assert entry_valid.tolist() == [[True, True, True], [True, True, True]]


def test_full_exit_validation_uses_population_and_valid_denominators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(9)
    target_q = _TARGET_Q.clone()
    _patch_episode_owners(monkeypatch, target_q)
    episode = _episode_fixture()
    model = _ExitModel()
    token = torch.tensor([[0.4, -0.3]])

    (
        raw_loss,
        stats,
        entry_targets,
        entry_valid,
        entry_realized_pnl,
    ) = trainer._unified_exit_full_population_eval_loss(
        model=model,
        target_model=copy.deepcopy(model).eval(),
        entry_decision_representations=token,
        target_entry_decision_representations=token.clone(),
        entry_row_indices=torch.tensor([7]),
        dataset=_ExitDataset(episode),
        device=torch.device("cpu"),
    )

    assert stats["population_rows"] == _EXPECTED_POPULATION_ROWS
    assert stats["q_valid_cells"] == _EXPECTED_Q_VALID_CELLS
    assert (
        stats["target_equivalent_action_rows"]
        == _EXPECTED_TARGET_EQUIVALENT_ROWS
    )
    assert (
        stats["hold_target_greedy_rows"] == _EXPECTED_HOLD_TARGET_GREEDY_ROWS
    )
    assert (
        stats["exit_now_target_greedy_rows"]
        == _EXPECTED_EXIT_NOW_TARGET_GREEDY_ROWS
    )
    assert model.forwarded_rows == _EXPECTED_POPULATION_ROWS
    assert float(raw_loss) == pytest.approx(stats["raw_loss"])
    assert entry_valid.tolist() == [[True, True, True]]
    assert torch.allclose(
        entry_targets[:, :2],
        unified_exit_first_state_side_values(
            frozen_target_q_bps=target_q,
            action_valid_mask=torch.from_numpy(_EPISODE_ACTION_VALID).unsqueeze(0),
            state_valid_mask=torch.from_numpy(_EPISODE_STATE_VALID).unsqueeze(0),
        ),
    )
    # Every replayed side must realize one of that side's declared EXIT_NOW
    # rewards; the replay never invents a value and never skips termination.
    rewards = np.asarray(episode["exit_now_reward_bps"], dtype=np.float64)
    for side_index in range(2):
        assert float(entry_realized_pnl[0, side_index]) in set(
            rewards[side_index].tolist()
        )


def test_joint_task_formula_omits_absent_task_entirely() -> None:
    # Both names come from the owner tuple, never from a restated task label.
    present, absent = JOINT_TASK_NAMES[0], JOINT_TASK_NAMES[1]
    model = _TaskWeights()
    model.task_log_variances[present].data.fill_(0.3)
    raw = torch.tensor(2.0, requires_grad=True)

    loss, stats = trainer._joint_task_loss(model, {present: raw})
    expected = torch.exp(torch.tensor(-0.3)) * 2.0 + 0.3
    assert torch.allclose(loss, expected)
    assert stats[f"joint_task_raw_loss_{present}"] == 2.0

    loss.backward()
    assert model.task_log_variances[present].grad is not None
    assert model.task_log_variances[absent].grad is None


def test_all_declared_task_weights_get_gradient_move_and_round_trip() -> None:
    model = _TaskWeights()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    observed = {name: False for name in JOINT_TASK_NAMES}
    raw_losses = {
        name: torch.tensor(0.2 + 0.03 * index)
        for index, name in enumerate(JOINT_TASK_NAMES)
    }

    loss, _ = trainer._joint_task_loss(model, raw_losses)
    loss.backward()
    trainer._observe_joint_task_weight_gradients(model, observed)
    optimizer.step()

    selected = {
        name: float(model.task_log_variances[name].detach().item())
        for name in JOINT_TASK_NAMES
    }
    assert all(observed.values())
    assert all(value != 0.0 for value in selected.values())
    metadata = joint_task_weighting_metadata(
        selected,
        supervision_observed={name: True for name in JOINT_TASK_NAMES},
        gradient_observed=observed,
    )
    assert (
        require_joint_task_weighting_metadata(metadata, context="TEST")
        == metadata
    )

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = _TaskWeights()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    for name in JOINT_TASK_NAMES:
        assert torch.equal(
            restored.task_log_variances[name],
            model.task_log_variances[name],
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_supervision",
        "missing_gradient",
        "missing_movement",
        "hash_drift",
    ],
)
def test_joint_task_metadata_fails_closed_on_missing_train_proof(
    mutation: str,
) -> None:
    selected = {name: 0.1 for name in JOINT_TASK_NAMES}
    payload = joint_task_weighting_metadata(
        selected,
        supervision_observed={name: True for name in JOINT_TASK_NAMES},
        gradient_observed={name: True for name in JOINT_TASK_NAMES},
    )
    bad = copy.deepcopy(payload)
    task = JOINT_TASK_NAMES[0]
    if mutation == "missing_supervision":
        bad["tasks"][task]["supervision_observed"] = False
    elif mutation == "missing_gradient":
        bad["tasks"][task]["gradient_observed"] = False
    elif mutation == "missing_movement":
        bad["tasks"][task]["moved_from_neutral"] = False
    else:
        bad["selected_log_variances_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="JOINT_TASK_WEIGHTING"):
        require_joint_task_weighting_metadata(bad, context="TEST")


def test_trainer_ast_contains_the_exact_learned_formula() -> None:
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(trainer._joint_task_loss))
    )
    weighted_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "weighted"
            for target in node.targets
        )
    ]
    assert len(weighted_assignments) == 1
    assert ast.unparse(weighted_assignments[0].value) == (
        "torch.exp(-log_variance) * raw_loss + log_variance"
    )
    assert FORMULA == (
        "sum(exp(-s_i) * L_i + s_i for active exact-label tasks i)"
    )


def test_wave_c_and_full_exit_population_source_guards() -> None:
    trainer_source = (ROOT / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py").read_text(
        encoding="utf-8"
    )
    recipe_source = (ROOT / "gx1/contracts/entry_model_native_train_recipe_v1.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "REQUIRED_POSITIVE_LOSS_WEIGHTS",
        "ENTRY_PATH_RANK_WEIGHT",
        "ENTRY_PATH_RANK_MARGIN",
        "ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT",
        "ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT",
        "ENTRY_SPECIALIST_GATE_MIN_MEAN",
        "OFFLINE_RL_REWARD_SCALE_BPS",
        '["exit_sample_valid"]',
        '["exit_action_target"]',
        "selected_target_counts",
        "all_non_tied_states_both_sides_all_val_episodes",
    )
    for token in forbidden:
        assert token not in trainer_source
        assert token not in recipe_source
    assert ".sample(" not in trainer_source
    assert "iter_full_exit_trajectory_chunks(" not in trainer_source
    assert "forward_exit_action(" not in trainer_source
    assert "_forward_unified_exit_episode_batch(" in trainer_source
    assert "build_unified_exit_fitted_q_targets(" in trainer_source
    assert 'episode["exit_now_reward_bps"]' in trainer_source
    assert "cross_entropy(selected_logits" not in trainer_source
    assert 'episode["exit_state_valid_mask"]' in trainer_source
