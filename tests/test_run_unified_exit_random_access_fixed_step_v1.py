from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from gx1.models.entry_v10.entry_v10_ctx_train_v3 import EntryV10CtxDataset
from gx1.scripts.run_unified_exit_random_access_fixed_step_v1 import (
    _FreshWeightEma,
    _ParentSampler,
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
        "random-access fixed-step smoke requires the exact attended CUDA command contract"
        in source
    )
