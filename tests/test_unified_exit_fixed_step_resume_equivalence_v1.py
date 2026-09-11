from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gx1.contracts.unified_exit_fixed_step_resume_equivalence_v1 import (
    build_equivalence,
    canonical_sha256,
    require_equivalence,
)
from gx1.contracts.unified_exit_random_access_checkpoint_v1 import file_sha256


def _pointer(root: Path, name: str) -> Path:
    state = {
        "launch_manifest_sha256": "1" * 64,
        "selected_sampler_artifact_sha256": "3" * 64,
        "batch_size": 16,
        "epoch_schedule_sha256": "2" * 64,
        "global_step": 4,
        "next_batch_offset": 4,
        "epoch_index": 0,
        "online_model_state": {"w": torch.arange(3)},
        "target_model_state": {"w": torch.arange(3)},
        "optimizer_state": {"state": {}, "param_groups": []},
        "lr_scheduler_state": {"last_epoch": 0},
        "weight_ema_state": {"shadow": {"w": torch.arange(3)}},
        "python_rng_state": (3, (1, 2), None),
        "numpy_rng_state": {"keys": [1]},
        "torch_rng_state": torch.arange(4, dtype=torch.uint8),
        "cuda_rng_states": [],
    }
    state_path = root / f"{name}.pt"
    torch.save(state, state_path)
    pointer = {
        "state_path": str(state_path),
        "state_file_sha256": file_sha256(state_path),
    }
    pointer["pointer_sha256"] = canonical_sha256(pointer)
    path = root / f"{name}.json"
    path.write_text(json.dumps(pointer))
    return path


def _schedule(root: Path, name: str, *, shift: int = 0) -> Path:
    next_batch = {
        "batch_offset": 4,
        "batch_size": 16,
        "child_entry_row_indices": list(range(64 + shift, 80 + shift)),
        "parent_entry_row_indices": list(range(248168 + shift, 248184 + shift)),
    }
    next_batch["identity_sha256"] = canonical_sha256(next_batch)
    value = {
        "schema_version": "gx1_unified_exit_epoch_schedule_witness_v1",
        "epoch_index": 0,
        "batch_size": 16,
        "entry_pair_count": 16384,
        "transition_count": 65536,
        "selected_sampler_artifact_sha256": "3" * 64,
        "epoch_schedule_sha256": "2" * 64,
        "child_order_sha256": "4" * 64,
        "parent_order_sha256": "5" * 64,
        "next_batch_after_optimizer_step_4": next_batch,
        "test_data_used": False,
    }
    value["witness_sha256"] = canonical_sha256(value)
    path = root / f"{name}.schedule.json"
    path.write_text(json.dumps(value))
    return path


def _build(tmp_path: Path, *, split_shift: int = 0) -> dict:
    return build_equivalence(
        reference_pointer_path=_pointer(tmp_path, "reference"),
        split_pointer_path=_pointer(tmp_path, "split"),
        reference_schedule_path=_schedule(tmp_path, "reference"),
        split_schedule_path=_schedule(tmp_path, "split", shift=split_shift),
        expected_launch_manifest_sha256="1" * 64,
        expected_gpu_batch_selection_artifact_sha256="6" * 64,
        expected_selected_sampler_artifact_sha256="3" * 64,
        expected_batch_size=16,
        expected_epoch_schedule_sha256="2" * 64,
    )


def test_reference_four_matches_split_three_plus_fresh_resume_one(tmp_path: Path):
    value = _build(tmp_path)
    checked = require_equivalence(value)
    assert checked["next_batch_offset"] == 4
    assert checked["next_batch_identity"]["child_entry_row_indices"] == list(
        range(64, 80)
    )


def test_equivalence_rejects_different_next_batch_identity(tmp_path: Path):
    with pytest.raises(RuntimeError, match="MISMATCH"):
        _build(tmp_path, split_shift=1)
