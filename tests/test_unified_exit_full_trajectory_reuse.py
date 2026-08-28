from __future__ import annotations

import copy
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_MAX_PATH_BARS,
)
from gx1.scripts.benchmark_unified_exit_episode_vectorization_v1 import (
    _production_local_specialist_routing,
    _production_mtf_specialist_routing,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    require_multi_tf_specialist_routing_v4,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4


class _OneRowDataset:
    def __len__(self) -> int:
        return 1


def _one_full_episode() -> dict[str, np.ndarray]:
    rewards = np.zeros((2, UNIFIED_EXIT_MAX_PATH_BARS), dtype=np.float64)
    rewards[0, 0] = 1.25
    rewards[1, 0] = -0.75
    return {"exit_now_reward_bps": rewards}


def _one_full_batch(*, predicted_tie: bool) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    shape = (1, 2, UNIFIED_EXIT_MAX_PATH_BARS, 2)
    q_values = torch.zeros(shape, dtype=torch.float32)
    targets = torch.zeros(shape, dtype=torch.float32)
    targets[..., 1] = 1.0
    if not predicted_tie:
        q_values[..., 1] = 1.0
    valid = torch.ones(shape, dtype=torch.bool)
    # Terminal states admit only Exit Now, exactly as the full-trajectory
    # population contract expects: two invalid Hold cells per entry episode.
    valid[:, :, -1, 0] = False
    return q_values, targets, valid


def test_first_pass_full_trajectory_report_is_complete_and_hash_bound() -> None:
    model = torch.nn.Linear(3, 2)
    target_model = copy.deepcopy(model)
    accumulator = trainer._new_unified_exit_full_trajectory_accumulator(
        model=model,
        target_model=target_model,
    )
    accumulator["entry_rows_scanned"] += 1
    q_values, targets, valid = _one_full_batch(predicted_tie=False)

    trainer._accumulate_unified_exit_full_trajectory(
        accumulator,
        raw_entry_indices=[17],
        selected_positions=[0],
        episodes=[_one_full_episode()],
        q_values=q_values,
        targets=targets,
        valid=valid,
    )
    report = trainer._finalize_unified_exit_full_trajectory_validation(
        accumulator,
        dataset=_OneRowDataset(),
        exit_gate_stats={},
        exit_gate_failures=(),
    )

    assert report["decision"] == "PASS"
    assert report["population_rows"] == 2 * UNIFIED_EXIT_MAX_PATH_BARS
    assert report["q_valid_cells"] == 4 * UNIFIED_EXIT_MAX_PATH_BARS - 2
    assert report["long_population_rows"] == UNIFIED_EXIT_MAX_PATH_BARS
    assert report["short_population_rows"] == UNIFIED_EXIT_MAX_PATH_BARS
    assert report["fitted_q_bellman_mse_mean"] == pytest.approx(0.0)
    assert report["target_equivalent_action_rows"] == 0
    assert report["unique_target_action_agreement"] == pytest.approx(1.0)
    assert report["learned_policy_mean_realized_executable_pnl_bps"] == pytest.approx(
        0.25
    )
    assert report["immediate_exit_mean_realized_executable_pnl_bps"] == pytest.approx(
        0.25
    )
    assert report["terminal_exit_mean_realized_executable_pnl_bps"] == pytest.approx(
        0.0
    )
    assert report["learned_mean_exit_state_index"] == pytest.approx(0.0)
    assert report["online_model_state_sha256"] == trainer._model_state_sha256(model)
    assert report["target_model_state_sha256"] == trainer._model_state_sha256(
        target_model
    )
    reference_stream = hashlib.sha256(
        b"gx1_unified_exit_full_trajectory_stream_v7"
    ).hexdigest()
    q_np = q_values.double().numpy()
    row_bytes = np.ascontiguousarray(
        np.column_stack((
            np.repeat(17, 2 * UNIFIED_EXIT_MAX_PATH_BARS),
            np.repeat(np.arange(2), UNIFIED_EXIT_MAX_PATH_BARS),
            np.tile(np.arange(UNIFIED_EXIT_MAX_PATH_BARS), 2),
            q_np[0, ..., 0].reshape(-1),
            q_np[0, ..., 1].reshape(-1),
        ))
    ).tobytes()
    expected_stream = hashlib.sha256(
        bytes.fromhex(reference_stream) + hashlib.sha256(row_bytes).digest()
    ).hexdigest()
    assert report["state_prediction_stream_sha256"] == expected_stream


def test_first_pass_full_trajectory_rejects_exact_predicted_q_ties() -> None:
    model = torch.nn.Linear(3, 2)
    accumulator = trainer._new_unified_exit_full_trajectory_accumulator(
        model=model,
        target_model=copy.deepcopy(model),
    )
    accumulator["entry_rows_scanned"] += 1
    q_values, targets, valid = _one_full_batch(predicted_tie=True)
    with pytest.raises(RuntimeError, match="UNIFIED_EXIT_FITTED_Q_POLICY_TIED_ACTION"):
        trainer._accumulate_unified_exit_full_trajectory(
            accumulator,
            raw_entry_indices=[17],
            selected_positions=[0],
            episodes=[_one_full_episode()],
            q_values=q_values,
            targets=targets,
            valid=valid,
        )


def test_candidate_source_reuses_the_hash_bound_first_validation_pass() -> None:
    source = Path(trainer.__file__).read_text(encoding="utf-8")

    assert "_validate_unified_exit_full_trajectories" not in source
    assert 'collect_full_exit_trajectory=(profile == "candidate")' in source
    assert "best_unified_exit_full_trajectory_validation" in source
    assert "UNIFIED_EXIT_SELECTED_CHECKPOINT_FULL_VAL_STATE_MISMATCH" in source


def test_exit_profile_clock_is_cpu_safe_and_first_batch_only() -> None:
    before = trainer._synchronized_exit_profile_clock(torch.device("cpu"))
    after = trainer._synchronized_exit_profile_clock(torch.device("cpu"))
    source = Path(trainer.__file__).read_text(encoding="utf-8")

    assert after >= before
    assert "_profile_timing = not _first_batch_logged" in source
    assert "[UNIFIED_EXIT_PROFILE]" in source
    assert "[TRAIN_PROFILE]" in source


def test_exit_benchmark_uses_complete_contract_owned_specialist_routing() -> None:
    signal_names, local = _production_local_specialist_routing()
    tf_names = tuple(MULTI_TF_PER_BAR_FEATURES_V4)
    mtf = _production_mtf_specialist_routing(tf_names)

    assert len(signal_names) == trainer.MODEL_NATIVE_SIGNAL_DIM
    assert tuple(local) == MODEL_NATIVE_TRAINING_SPECIALISTS
    assert tuple(mtf) == MODEL_NATIVE_TRAINING_SPECIALISTS
    assert sorted(index for indices in local.values() for index in indices) == list(
        range(len(signal_names))
    )
    assert mtf == {
        name: list(indices)
        for name, indices in require_multi_tf_specialist_routing_v4(tf_names).items()
    }


def test_exit_vectorization_benchmark_rejects_direct_cuda_before_model_build() -> None:
    """The synthetic benchmark is CPU-only and must not become a GPU bypass."""

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gx1.scripts.benchmark_unified_exit_episode_vectorization_v1",
            "--batch",
            "1",
            "--device",
            "cuda",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr
