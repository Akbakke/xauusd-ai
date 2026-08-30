from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from gx1.contracts.entry_model_native_readiness_v1 import (
    MODEL_NATIVE_ACTIVE_HEADS,
)
from gx1.contracts.entry_model_native_train_recipe_v1 import (
    MODEL_NATIVE_RECIPE_ENV_KEYS,
)
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


TRAINER_PATH = Path(trainer.__file__)


def _live_active_head_epoch_accumulator() -> dict:
    rows = 32
    base = np.linspace(-1.0, 1.0, rows, dtype=np.float64)
    accumulator = trainer._new_active_head_epoch_accumulator()
    for head_name, components in trainer._ACTIVE_HEAD_TARGET_COMPONENTS.items():
        for component_name in components:
            width = int(trainer._ACTIVE_HEAD_COMPONENT_WIDTHS[component_name])
            prediction = np.stack(
                [base + 0.01 * column for column in range(width)],
                axis=1,
            )
            target = np.stack(
                [base[::-1] + 0.02 * column for column in range(width)],
                axis=1,
            )
            for column in trainer._ACTIVE_HEAD_STRUCTURAL_CONSTANT_COLUMNS.get(
                component_name, ()
            ):
                target[:, column] = 0.0
            accumulator["heads"][head_name]["components"][component_name] = {
                "prediction": [prediction],
                "target": [target],
            }
    return accumulator


def test_trainer_uses_direct_masked_raw_bps_entry_q_mse() -> None:
    source = TRAINER_PATH.read_text(encoding="utf-8")
    assert source.count(
        "nn.functional.mse_loss(\n"
        "                entry_action_q_bps[entry_action_q_valid],"
    ) == 1
    assert source.count(
        "nn.functional.mse_loss(\n"
        "            entry_action_q_bps[entry_action_q_valid],"
    ) == 1
    assert "nn.functional.cross_entropy(" not in source
    assert "entry_action_q_bps" in source
    assert "frozen_exit_first_state_values_bps" in source


def test_cuda_tf32_policy_is_explicitly_source_bound() -> None:
    cuda_policy = trainer._training_precision_metadata("cuda")
    cpu_policy = trainer._training_precision_metadata("cpu")
    assert cuda_policy == {
        "precision": "deterministic_fp32_tensors_tf32_matmul",
        "compile": False,
        "tf32": True,
        "autocast": False,
    }
    assert cpu_policy == {
        "precision": "deterministic_fp32",
        "compile": False,
        "tf32": False,
        "autocast": False,
    }
    assert "torch.backends.cuda.matmul.allow_tf32 = True" in TRAINER_PATH.read_text(
        encoding="utf-8"
    )


def test_exit_mtf_history_uses_m1_state_start_not_already_closed_clock() -> None:
    """The shared MTF route owns the single +60-second Exit availability shift."""

    source = TRAINER_PATH.read_text(encoding="utf-8")
    call_start = source.index("**self._get_exit_multi_tf_episode_histories(")
    call_end = source.index(")", call_start)
    call = source[call_start:call_end]
    assert 'core["exit_state_row_time_ns"]' in call
    assert 'core["exit_decision_time_ns"]' not in call
    history_start = source.index("def _get_exit_multi_tf_episode_histories(")
    history_end = source.index("def materialize_full_exit_episode(", history_start)
    history = source[history_start:history_end]
    assert "state_bar_start_time_ns" in history
    assert "availability_ns = state_start_ns + int(" in history


def test_retired_entry_authorities_have_no_trainer_surface() -> None:
    source = TRAINER_PATH.read_text(encoding="utf-8")
    forbidden = (
        "direction_logits",
        "direction_probs",
        "raw_direction_logits",
        "public_trade_flat",
        "evidence_fusion",
        "offline_rl",
        "head_direction",
        "head_mtf_direction",
        "head_action_value",
        "head_expectile",
        "hier_trade",
        "hier_side",
        "clean_edge",
        "survival",
        "side_validity",
        "y_direction",
    )
    assert [token for token in forbidden if token in source] == []


def test_active_head_contract_is_exact_and_q_has_sole_authority() -> None:
    assert tuple(trainer._ACTIVE_HEAD_OUTPUT_COMPONENTS) == tuple(
        MODEL_NATIVE_ACTIVE_HEADS
    )
    assert tuple(trainer._ACTIVE_HEAD_TARGET_COMPONENTS) == tuple(
        MODEL_NATIVE_ACTIVE_HEADS
    )
    assert trainer._ACTIVE_HEAD_ACTION_AUTHORITY_NONE == (
        frozenset(MODEL_NATIVE_ACTIVE_HEADS) - {"entry_action_q"}
    )
    assert trainer._active_head_contract_failures() == []


def test_every_current_active_head_has_live_target_and_output_evidence() -> None:
    metrics, failures = trainer._active_head_epoch_diagnostics(
        _live_active_head_epoch_accumulator()
    )
    assert failures == []
    assert metrics["active_head_health_ok"] is True
    assert tuple(metrics["active_head_diagnostics"]) == tuple(
        MODEL_NATIVE_ACTIVE_HEADS
    )
    assert (
        metrics["active_head_diagnostics"]["entry_action_q"][
            "entry_action_authority"
        ]
        == "sole_raw_bps_entry_q"
    )
    component = metrics["active_head_diagnostics"]["forecast"]["components"][
        "forecast_pred"
    ]
    assert component["validation_metrics"]["metric_type"] == "regression"
    assert "pearson" in component["validation_metrics"]


def test_primary_entry_q_diagnostics_are_chronological_and_deciled() -> None:
    rows = 20
    prediction = np.column_stack(
        (
            np.linspace(-2.0, 2.0, rows),
            np.linspace(-3.0, 1.0, rows),
            np.linspace(-4.0, 0.0, rows),
        )
    )
    target = prediction * 2.0
    dataset = SimpleNamespace(
        df=pd.DataFrame(
            {
                "time": pd.date_range(
                    "2025-06-01T00:00:00Z", periods=rows, freq="12h"
                )
            }
        )
    )
    observed = trainer._entry_action_q_primary_validation_diagnostics(
        prediction=prediction,
        target=target,
        valid=np.ones_like(prediction, dtype=bool),
        entry_row_indices=np.arange(rows, dtype=np.int64),
        dataset=dataset,
    )
    assert observed["primary_head"] == "entry_action_q"
    assert len(observed["deciles"]) == 10
    assert observed["top_decile_minus_bottom_decile_target_bps"] > 0.0
    assert observed["volatility_regime_stability"]["available"] is False


def test_joint_task_loss_evidence_covers_all_ten_tasks() -> None:
    accumulator = trainer._new_active_head_epoch_accumulator()
    task_losses = {
        name: torch.tensor(float(index + 1))
        for index, name in enumerate(trainer.JOINT_TASK_NAMES)
    }
    trainer._accumulate_joint_task_loss_evidence(
        accumulator,
        task_losses,
        active_head_supervised_cells={
            head_name: 4 for head_name in trainer._ACTIVE_HEAD_TO_JOINT_TASK
        },
        unified_exit_supervised_cells=3,
    )
    observed = trainer._finalize_joint_task_loss_evidence(accumulator)
    assert {
        key.removeprefix("joint_task_raw_loss_mean_")
        for key in observed
        if key.startswith("joint_task_raw_loss_mean_")
    } == set(trainer.JOINT_TASK_NAMES)
    assert observed["joint_task_raw_loss_mean_entry_action_q"] == 1.0
    assert observed["joint_task_supervised_cells_unified_exit_action"] == 3


def test_dead_current_head_blocks_checkpoint_health() -> None:
    accumulator = _live_active_head_epoch_accumulator()
    component = trainer._ACTIVE_HEAD_TARGET_COMPONENTS["forecast"][0]
    accumulator["heads"]["forecast"]["components"][component]["prediction"] = [
        np.zeros((32, trainer._ACTIVE_HEAD_COMPONENT_WIDTHS[component]))
    ]
    metrics, failures = trainer._active_head_epoch_diagnostics(accumulator)
    assert metrics["active_head_health_ok"] is False
    assert any("OUTPUT_DEAD" in failure for failure in failures)


def test_technical_smoke_handles_sparse_masked_event_without_weakening_candidate() -> None:
    """A uniform tiny smoke may see a real rare label fewer than 16 times."""

    accumulator = _live_active_head_epoch_accumulator()
    component = trainer._ACTIVE_HEAD_TARGET_COMPONENTS["trendline_event"][0]
    mask = np.ones((32, trainer._ACTIVE_HEAD_COMPONENT_WIDTHS[component]), dtype=bool)
    mask[7:, 0] = False
    mask[5:, 1] = False
    accumulator["heads"]["trendline_event"]["components"][component]["mask"] = [mask]

    strict_metrics, strict_failures = trainer._active_head_epoch_diagnostics(accumulator)
    assert strict_metrics["active_head_health_ok"] is False
    assert any("ROWS_INSUFFICIENT" in failure for failure in strict_failures)

    technical_metrics, technical_failures = trainer._active_head_epoch_diagnostics(
        accumulator,
        minimum_supervised_rows=trainer._ACTIVE_HEAD_TECHNICAL_SMOKE_MIN_ROWS,
    )
    assert technical_failures == []
    assert technical_metrics["active_head_health_ok"] is True
    validation_stats = {
        **strict_metrics,
        "active_head_technical_smoke_evidence": {
            "minimum_supervised_rows": (
                trainer._ACTIVE_HEAD_TECHNICAL_SMOKE_MIN_ROWS
            ),
            "health_ok": True,
        },
    }
    assert trainer._profiled_active_head_admission_health(
        profile="smoke",
        validation_stats=validation_stats,
    ) is True
    assert trainer._profiled_active_head_admission_health(
        profile="candidate",
        validation_stats=validation_stats,
    ) is False


def test_checkpoint_admission_uses_only_learned_head_and_gate_liveness() -> None:
    assert trainer._checkpoint_admission_ok(
        profile="candidate",
        active_head_health_ok=True,
        cooperation_gate_health_ok=True,
        exit_cooperation_gate_health_ok=True,
        candidate_exit_gate_health_provisional_ok=False,
    )
    assert not trainer._checkpoint_admission_ok(
        profile="candidate",
        active_head_health_ok=False,
        cooperation_gate_health_ok=True,
        exit_cooperation_gate_health_ok=True,
        candidate_exit_gate_health_provisional_ok=False,
    )
    assert trainer._checkpoint_admission_ok(
        profile="smoke",
        active_head_health_ok=True,
        cooperation_gate_health_ok=False,
        exit_cooperation_gate_health_ok=False,
        candidate_exit_gate_health_provisional_ok=False,
    )


def test_trainer_environment_reads_are_contract_owned() -> None:
    source = TRAINER_PATH.read_text(encoding="utf-8")
    for key in MODEL_NATIVE_RECIPE_ENV_KEYS:
        assert key in source or key in trainer.MODEL_NATIVE_RECIPE_ENV
    assert "ENTRY_SYMMETRIC_NEGATIVES" not in source


def test_subsampling_is_uniform_and_not_label_dependent() -> None:
    source = TRAINER_PATH.read_text(encoding="utf-8")
    assert "deterministic_uniform_subsample_indices" in source
    assert "subsample_rows" in source
    assert "y_direction" not in source
    assert "stratified" not in source.lower()


def test_entry_and_exit_share_one_frozen_target_snapshot_per_iteration() -> None:
    source = TRAINER_PATH.read_text(encoding="utf-8")
    assert "target_model = copy.deepcopy(model)" in source
    assert "target_model.requires_grad_(False)" in source
    assert "target_updated_from_val_or_test" in source
    assert "require_entry_fitted_q_iteration_state" in source
