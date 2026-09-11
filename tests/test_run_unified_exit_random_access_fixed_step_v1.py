from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from gx1.models.entry_v10.entry_v10_ctx_train_v3 import EntryV10CtxDataset
from gx1.scripts.run_unified_exit_random_access_fixed_step_v1 import (
    _FreshWeightEma,
    _ParentSampler,
    _absolute_optimizer_step,
)


def test_parent_sampler_converts_batch_cursor_to_parent_row_offset() -> None:
    rows = tuple(range(100, 200))
    sampler = _ParentSampler(rows, batch_offset=3, batch_size=16)
    assert list(sampler)[:16] == list(range(148, 164))
    assert len(sampler) == 52


def test_parent_child_entry_mapping_is_exact_and_fail_closed() -> None:
    dataset = EntryV10CtxDataset.__new__(EntryV10CtxDataset)
    dataset.df = pd.DataFrame({"x": range(6)})
    dataset._random_access_child_index_by_parent = None
    dataset.bind_random_access_entry_coordinate_mapping_v1(
        parent_entry_row_indices=[2, 3, 4], child_entry_row_indices=[0, 1, 2]
    )
    assert dataset._random_access_child_index_by_parent == {2: 0, 3: 1, 4: 2}
    with pytest.raises(RuntimeError, match="ENTRY_MAPPING_INVALID"):
        dataset.bind_random_access_entry_coordinate_mapping_v1(
            parent_entry_row_indices=[2], child_entry_row_indices=[0]
        )


def test_fresh_ema_round_trip_is_exact() -> None:
    model = torch.nn.Linear(3, 2)
    ema = _FreshWeightEma(model, 0.9)
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    state = ema.state_dict()
    restored = _FreshWeightEma(model, 0.9)
    restored.load_state_dict(state)
    assert restored.steps == 1
    assert all(
        torch.equal(restored.shadow[name], tensor)
        for name, tensor in state["shadow"].items()
    )


def test_capped_runner_allowlists_only_exact_fixed_step_module() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/gx1_capped_run.sh"
    ).read_text()
    assert (
        "RANDOM_ACCESS_FIXED_STEP_MODULE=gx1.scripts.run_unified_exit_random_access_fixed_step_v1"
        in source
    )
    assert 'if [[ "$module" == "$RANDOM_ACCESS_FIXED_STEP_MODULE" ]]' in source
    assert (
        "random-access fixed-step smoke/train requires the exact attended CUDA stage contract"
        in source
    )


def test_dataloader_iterator_does_not_advance_model_dropout_rng() -> None:
    torch.manual_seed(123)
    expected = torch.rand(4)
    torch.manual_seed(123)
    loader = DataLoader(
        TensorDataset(torch.arange(64)),
        batch_size=16,
        sampler=_ParentSampler(tuple(range(64)), batch_offset=0, batch_size=16),
        generator=torch.Generator().manual_seed(999),
    )
    next(iter(loader))
    assert torch.equal(torch.rand(4), expected)


def test_checkpoint_global_step_counts_optimizer_steps_not_writes() -> None:
    assert _absolute_optimizer_step(
        initial_global_step=77, start_batch_offset=128, next_batch_offset=192
    ) == 141
    with pytest.raises(RuntimeError, match="CHECKPOINT_CURSOR_INVALID"):
        _absolute_optimizer_step(
            initial_global_step=77, start_batch_offset=128, next_batch_offset=128
        )


def test_capped_runner_has_exact_non_attended_full_val_allowlist() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/gx1_capped_run.sh"
    ).read_text()
    assert "RANDOM_ACCESS_VAL_MODULE=gx1.scripts.run_unified_exit_random_access_val_v1" in source
    assert 'if [[ "$module" == "$RANDOM_ACCESS_VAL_MODULE" ]]' in source
    assert "random-access full VAL requires the exact guarded campaign contract" in source
    assert "GX1_CAMPAIGN_GUARD_LOG_PATH" in source
    assert "set -o noclobber" in source
    for flag in (
        "--launch-manifest",
        "--final-train-checkpoint-authority",
        "--final-train-checkpoint-authority-file-sha256",
        "--checkpoint-pointer",
        "--progress-path",
        "--rollout-progress-path",
        "--result-path",
        "--max-forwards-this-invocation",
        "--progress-interval-forwards",
        "--compute-guard-max-model-forwards",
        "--compute-guard-max-materialized-state-views",
        "--compute-guard-max-wall-seconds",
    ):
        assert flag in source
