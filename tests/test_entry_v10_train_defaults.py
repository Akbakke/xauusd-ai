from __future__ import annotations

import ast

import pytest
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


def test_cuda_memory_fence_and_strict_fp32_policy_are_source_bound() -> None:
    cuda_policy = trainer._training_precision_metadata("cuda")
    cpu_policy = trainer._training_precision_metadata("cpu")
    assert cuda_policy == {
        "precision": "deterministic_fp32",
        "compile": False,
        "tf32": False,
        "autocast": False,
        "cuda_memory_fraction": 0.45,
    }
    assert cpu_policy == {
        "precision": "deterministic_fp32",
        "compile": False,
        "tf32": False,
        "autocast": False,
        "cuda_memory_fraction": None,
    }
    source = TRAINER_PATH.read_text(encoding="utf-8")
    assert "torch.backends.cuda.matmul.allow_tf32 = False" in source
    assert "torch.cuda.set_per_process_memory_fraction(" in source
    assert 'tf32_matmul=false "' in source
    assert 'cuda_memory_fraction=%s "' in source


def test_hopper_policy_uses_bf16_forward_with_fp32_outputs() -> None:
    policy = trainer._training_precision_metadata(
        "cuda",
        "deterministic_bf16_hopper",
    )
    assert policy["autocast_dtype"] == "bfloat16"
    assert policy["loss_reduction_dtype"] == "float32"
    source = TRAINER_PATH.read_text(encoding="utf-8")
    assert 'torch.autocast(device_type="cuda", dtype=torch.bfloat16)' in source
    assert "torch.cuda.is_bf16_supported()" in source
    assert "GradScaler(" not in source


def test_candidate_override_skips_resumable_training_call() -> None:
    tree = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))
    run_train = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_train"
    )
    override_branch = next(
        node
        for node in ast.walk(run_train)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "candidate_result_override is not None"
    )
    calls = [
        node.func.id
        for statement in override_branch.orelse
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "_run_resumable_candidate_training" in calls
    assert all(
        "_run_resumable_candidate_training" not in {
            node.func.id
            for node in ast.walk(statement)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        for statement in override_branch.body
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


@pytest.mark.parametrize("capability,native_supported,accepted", [
    ((8, 6), True, True), ((8, 6), False, False),
    ((7, 5), True, False), ((9, 0), True, False),
])
def test_local_bf16_capability_probe_requires_native_3090_support(
    monkeypatch, capability, native_supported, accepted,
) -> None:
    monkeypatch.setattr(trainer.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(trainer.torch.cuda, "get_device_capability", lambda device: capability)
    def supported(*, including_emulation):
        assert including_emulation is False
        return native_supported
    monkeypatch.setattr(trainer.torch.cuda, "is_bf16_supported", supported)
    if accepted:
        trainer._require_local_bf16_3090_capability()
    else:
        with pytest.raises(RuntimeError, match="LOCAL_BF16_3090_NATIVE_CAPABILITY_REQUIRED"):
            trainer._require_local_bf16_3090_capability()


def test_local_bf16_forward_preserves_fp32_output_and_gradient_boundary(monkeypatch) -> None:
    """Exercise actual autocast/autograd on CPU; CUDA kernel proof is the smoke."""
    import torch
    from gx1.contracts.entry_training_precision_v1 import EXPERIMENTAL_BF16_3090
    real_autocast = torch.autocast
    requests = []
    def cpu_test_context(*, device_type, dtype):
        requests.append((device_type, dtype))
        return real_autocast(device_type="cpu", dtype=dtype)
    monkeypatch.setattr(trainer, "_TRAINING_PRECISION_POLICY", EXPERIMENTAL_BF16_3090)
    monkeypatch.setattr(trainer.torch, "autocast", cpu_test_context)
    model = torch.nn.Linear(8, 4)
    output = trainer._model_forward_fp32(model, torch.randn(3, 8))
    assert requests == [("cuda", torch.bfloat16)]
    assert output.dtype == torch.float32
    output.square().mean().backward()
    for parameter in model.parameters():
        assert parameter.dtype == torch.float32
        assert parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("capability,accepted", [((8, 6), True), ((9, 0), False), ((7, 5), False)])
def test_local_full_exit_batch_requires_the_local_architecture(monkeypatch, capability, accepted):
    monkeypatch.setattr(trainer.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(trainer.torch.cuda, "get_device_capability", lambda device: capability)
    if accepted:
        trainer._require_local_fp32_full_exit_batch_capability()
    else:
        with pytest.raises(RuntimeError, match="LOCAL_FP32_FULL_EXIT_BATCH_CAPABILITY_REQUIRED"):
            trainer._require_local_fp32_full_exit_batch_capability()


def test_local_full_exit_batch_does_not_enable_autocast(monkeypatch):
    from gx1.contracts.entry_training_precision_v1 import EXPERIMENTAL_FP32_3090_FULL_EXIT_BATCH
    monkeypatch.setattr(trainer, "_TRAINING_PRECISION_POLICY", EXPERIMENTAL_FP32_3090_FULL_EXIT_BATCH)
    def forbidden(*args, **kwargs):
        raise AssertionError("FP32 batch optimization must not invoke autocast")
    monkeypatch.setattr(trainer.torch, "autocast", forbidden)
    model = trainer.torch.nn.Linear(8, 4)
    result = trainer._model_forward_fp32(model, trainer.torch.randn(10, 8))
    assert result.dtype == trainer.torch.float32
    result.square().mean().backward()
    assert all(p.grad is not None and trainer.torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("policy,rows,device,max_steps,explicit_chunk,expected", [
    (trainer.EXPERIMENTAL_FP32_3090_FULL_EXIT_BATCH, 10, "cuda", None, None, 10),
    (trainer.EXPERIMENTAL_FP32_3090_FULL_EXIT_BATCH, 2, "cuda", None, None, 2),
    (trainer.DETERMINISTIC_FP32, 10, "cuda", None, None, 8),
    (trainer.DETERMINISTIC_FP32, 10, "cpu", None, None, None),
    (trainer.DETERMINISTIC_FP32, 10, "cuda", 1, 3, 3),
])
def test_smoke_exit_training_call_honors_policy_and_partial_batch(policy, rows, device, max_steps, explicit_chunk, expected):
    """Execute the actual train_epoch call-site expression, not a copied default."""
    tree = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))
    epoch = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train_epoch")
    calls = [node for node in ast.walk(epoch) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_train_unified_exit_full_population"]
    assert len(calls) == 1
    value = next(item.value for item in calls[0].keywords if item.arg == "exit_action_forward_chunk_rows")
    actual = eval(compile(ast.Expression(value), str(TRAINER_PATH), "eval"), vars(trainer), {
        "device": torch.device(device), "batch_rows": rows,
        "session_max_optimizer_steps": max_steps,
        "session_exit_action_forward_chunk_rows": explicit_chunk,
        "_TRAINING_PRECISION_POLICY": policy,
    })
    assert actual == expected


@pytest.mark.parametrize("policy,expected_q,expected_batch,expected_autocast", [
    ("deterministic_fp32", False, False, False),
    ("experimental_fp32_3090_no_uninitialized_fill", False, False, False),
    ("experimental_bf16_3090", False, False, True),
    ("experimental_bf16_3090_fp32_q_heads_no_fill", True, False, True),
    ("experimental_fp32_3090_batched_mtf_teacher_no_fill", False, True, False),
])
@pytest.mark.parametrize("raise_inside", [False, True])
def test_local_forward_context_activates_only_declared_experiment(
    monkeypatch, policy, expected_q, expected_batch, expected_autocast, raise_inside,
):
    from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as module
    real_autocast = torch.autocast

    def cpu_emulation(*, device_type, **kwargs):
        # Exercise actual context restoration on CPU; no CUDA capability claim.
        return real_autocast(device_type="cpu" if device_type == "cuda" else device_type, **kwargs)

    monkeypatch.setattr(torch, "autocast", cpu_emulation)
    monkeypatch.setattr(trainer, "_TRAINING_PRECISION_POLICY", policy)
    observed = []

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(1, 2, bias=False)
            with torch.no_grad():
                self.head.weight.copy_(torch.tensor([[1.0], [1.001]]))

        def forward(self, x):
            observed.append((module._RAW_Q_FP32_SCOPE.get(),
                             module._BATCH_EQUAL_LENGTH_MTF_EVAL.get(),
                             torch.is_autocast_enabled("cpu")))
            if raise_inside:
                raise RuntimeError("deliberate forward failure")
            return module._forward_raw_q_head(self.head, x)

    probe = Probe()
    x = torch.ones(1, 1, requires_grad=True)
    if raise_inside:
        with pytest.raises(RuntimeError, match="deliberate forward failure"):
            trainer._model_forward_fp32(probe, x)
    else:
        output = trainer._model_forward_fp32(probe, x)
        assert output.dtype == torch.float32
        if expected_autocast and not expected_q:
            assert output[0, 0] == output[0, 1]
        else:
            assert output[0, 0] < output[0, 1]
        output.sum().backward()
        assert torch.isfinite(x.grad).all() and torch.all(x.grad != 0)
        assert torch.isfinite(probe.head.weight.grad).all()
    assert observed == [(expected_q, expected_batch, expected_autocast)]
    assert module._RAW_Q_FP32_SCOPE.get() is False
    assert module._BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False
    assert torch.is_autocast_enabled("cpu") is False
