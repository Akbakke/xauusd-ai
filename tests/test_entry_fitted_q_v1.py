from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from gx1.contracts.entry_fitted_q_v1 import (
    ENTRY_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION,
    build_entry_fitted_q_targets,
    entry_fill_binding_sha256,
    entry_fitted_q_contract,
    replay_entry_fitted_q_policy,
    require_entry_fitted_q_iteration_state,
)
from gx1.contracts.unified_exit_fitted_q_v1 import (
    unified_exit_fitted_q_contract,
)


_SHA = "a" * 64


def _exit_iteration() -> dict[str, object]:
    return {
        "schema_version": "gx1_unified_exit_fitted_q_iteration_state_v1",
        "iteration_index": 3,
        "target_model_state_sha256": "1" * 64,
        "train_split_sha256": "2" * 64,
        "train_fold_sha256": "3" * 64,
        "source_lineage_sha256": "4" * 64,
        "normalization_sha256": "5" * 64,
        "fitted_q_contract": unified_exit_fitted_q_contract(),
        "target_updated_from_val_or_test": False,
    }


def _entry_iteration() -> dict[str, object]:
    import hashlib
    import json

    exit_state = _exit_iteration()
    exit_sha = hashlib.sha256(
        json.dumps(
            exit_state,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return {
        "schema_version": ENTRY_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION,
        "iteration_index": 3,
        "entry_target_model_state_sha256": "1" * 64,
        "exit_target_model_state_sha256": "1" * 64,
        "exit_fitted_q_iteration_state_sha256": exit_sha,
        "train_split_sha256": "2" * 64,
        "train_fold_sha256": "3" * 64,
        "source_lineage_sha256": "4" * 64,
        "normalization_sha256": "5" * 64,
        "entry_fitted_q_contract": entry_fitted_q_contract(),
        "exit_fitted_q_contract": unified_exit_fitted_q_contract(),
        "target_updated_from_val_or_test": False,
    }


def test_entry_targets_are_frozen_exit_v0_and_flat_zero() -> None:
    exit_v0 = torch.tensor(
        [[7.5, -2.0], [-4.0, 3.25]], requires_grad=True
    )
    valid = torch.tensor([[True, True], [True, True]])
    targets, mask, binding = build_entry_fitted_q_targets(
        frozen_exit_first_state_values_bps=exit_v0,
        exit_side_valid_mask=valid,
        episode_pack_sha256=["6" * 64, "7" * 64],
        fill_binding_sha256=["8" * 64, "9" * 64],
    )
    assert torch.equal(
        targets,
        torch.tensor([[7.5, -2.0, 0.0], [-4.0, 3.25, 0.0]]),
    )
    assert bool(mask.all())
    assert not targets.requires_grad
    assert binding["eligible_side_cells"] == 4
    assert replay_entry_fitted_q_policy(
        predicted_q_bps=targets.numpy(), action_valid_mask=mask.numpy()
    ).tolist() == [0, 1]


def test_entry_negative_sides_choose_flat_and_ties_fail_closed() -> None:
    q = np.asarray([[-1.0, -2.0, 0.0]], dtype=np.float32)
    valid = np.ones_like(q, dtype=np.bool_)
    assert replay_entry_fitted_q_policy(
        predicted_q_bps=q, action_valid_mask=valid
    ).tolist() == [2]
    with pytest.raises(RuntimeError, match="TIED_ACTION"):
        replay_entry_fitted_q_policy(
            predicted_q_bps=np.asarray([[0.0, -1.0, 0.0]]),
            action_valid_mask=valid,
        )


def test_entry_target_masks_require_exact_episode_and_fill_bindings() -> None:
    values = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    valid = torch.tensor([[True, False], [False, False]])
    targets, mask, _ = build_entry_fitted_q_targets(
        frozen_exit_first_state_values_bps=values,
        exit_side_valid_mask=valid,
        episode_pack_sha256=["6" * 64, None],
        fill_binding_sha256=["7" * 64, None],
    )
    assert mask.tolist() == [[True, False, True], [False, False, True]]
    assert targets[:, 2].tolist() == [0.0, 0.0]
    with pytest.raises(RuntimeError, match="INELIGIBLE_ROW_HAS_BINDING"):
        build_entry_fitted_q_targets(
            frozen_exit_first_state_values_bps=values,
            exit_side_valid_mask=valid,
            episode_pack_sha256=["6" * 64, _SHA],
            fill_binding_sha256=["7" * 64, _SHA],
        )


def test_fill_binding_changes_with_side_quotes_episode_or_state_clock() -> None:
    kwargs = {
        "entry_row_index": 11,
        "episode_pack_sha256": _SHA,
        "first_exit_state_time_ns": 1_700_000_000_000_000_000,
        "exit_entry_bid_ask": np.asarray(
            [[1999.9, 2000.1], [1999.9, 2000.1]], dtype=np.float32
        ),
    }
    baseline = entry_fill_binding_sha256(**kwargs)
    for key, value in (
        ("episode_pack_sha256", "b" * 64),
        ("first_exit_state_time_ns", kwargs["first_exit_state_time_ns"] + 1),
        (
            "exit_entry_bid_ask",
            np.asarray([[1999.8, 2000.1], [1999.9, 2000.1]], dtype=np.float32),
        ),
    ):
        mutated = dict(kwargs)
        mutated[key] = value
        assert entry_fill_binding_sha256(**mutated) != baseline


def test_entry_iteration_is_exactly_bound_to_train_exit_teacher() -> None:
    observed = _entry_iteration()
    assert require_entry_fitted_q_iteration_state(
        observed,
        exit_fitted_q_iteration_state=_exit_iteration(),
        context="TEST",
    ) == observed
    for key, value in (
        ("target_updated_from_val_or_test", True),
        ("train_fold_sha256", "f" * 64),
        ("exit_target_model_state_sha256", "e" * 64),
    ):
        mutated = copy.deepcopy(observed)
        mutated[key] = value
        with pytest.raises(RuntimeError):
            require_entry_fitted_q_iteration_state(
                mutated,
                exit_fitted_q_iteration_state=_exit_iteration(),
                context="TEST",
            )
