"""Regression coverage for the executable unified-Exit movement proof."""

from __future__ import annotations

import pytest
import torch

from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
    _UNIFIED_EXIT_MOVEMENT_PREFIXES,
    _unified_exit_movement_proof,
)


def _active_component_states() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build one representative state key per active component prefix."""

    initial: dict[str, torch.Tensor] = {}
    selected: dict[str, torch.Tensor] = {}
    for component, prefixes in _UNIFIED_EXIT_MOVEMENT_PREFIXES.items():
        for ordinal, prefix in enumerate(prefixes):
            key = f"{prefix}movement_{component}_{ordinal}"
            initial[key] = torch.zeros(1)
            selected[key] = torch.ones(1)
    return initial, selected


def test_unified_exit_movement_proof_tracks_executable_episode_components() -> None:
    initial, selected = _active_component_states()

    evidence = _unified_exit_movement_proof(
        initial,
        selected,
        selected_checkpoint_epoch=1,
    )

    assert evidence["schema_version"] == "gx1_unified_exit_parameter_movement_v2"
    assert evidence["all_exit_components_moved"] is True
    assert set(evidence["component_max_abs_delta"]) == set(
        _UNIFIED_EXIT_MOVEMENT_PREFIXES
    )
    assert all(delta > 0.0 for delta in evidence["component_max_abs_delta"].values())


def test_unified_exit_movement_proof_rejects_a_dead_active_episode_component() -> None:
    initial, selected = _active_component_states()
    path_key = next(
        key
        for key in selected
        if key.startswith("exit_episode_path_gru.")
    )
    selected[path_key] = initial[path_key].clone()

    with pytest.raises(
        RuntimeError,
        match=r"UNIFIED_EXIT_SELECTED_CHECKPOINT_UNTRAINED.*episode_path_encoder",
    ):
        _unified_exit_movement_proof(
            initial,
            selected,
            selected_checkpoint_epoch=1,
        )


def test_unified_exit_movement_proof_excludes_retired_static_exit_branch() -> None:
    retired_prefixes = (
        "exit_path_encoder.",
        "exit_entry_path_attention.",
        "exit_entry_query_norm.",
        "exit_fuse.",
    )
    active_prefixes = tuple(
        prefix
        for prefixes in _UNIFIED_EXIT_MOVEMENT_PREFIXES.values()
        for prefix in prefixes
    )

    assert not any(
        prefix.startswith(retired)
        for prefix in active_prefixes
        for retired in retired_prefixes
    )
