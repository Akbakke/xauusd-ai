"""Actual CLI boundaries: budgets do not replace recipe or candidate gates."""

import ast
import hashlib
import inspect
import json
import sys

import pytest

from gx1.contracts import entry_model_native_train_launch_v1 as launch_contract
from gx1.contracts import entry_pretest_candidate_launch_gate_v1 as gate_contract
from gx1.contracts import entry_training_precision_v1 as precision
from gx1.contracts.entry_model_native_pretest_technical_recipe_v1 import (
    canonical_json_sha256,
)
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.scripts import run_entry_model_native_pretest_technical_train_v1 as launcher
from tests.test_entry_pretest_technical_launch import _candidate_command


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(tmp_path, monkeypatch, *, selected=True):
    _, rp, _, gp, gs = _candidate_command(tmp_path, monkeypatch)
    recipe = json.loads(rp.read_bytes())
    if selected:
        recipe["trainer_cli"]["precision_policy"] = (
            precision.EXPERIMENTAL_FP32_3090_NO_UNINITIALIZED_FILL
        )
        recipe["trainer_cli_sha256"] = canonical_json_sha256(recipe["trainer_cli"])
        rp.write_text(json.dumps(recipe))
    rs = digest(rp)
    budget = dict(
        schema_version="gx1_candidate_execution_budget_v1",
        recipe_json=str(rp),
        recipe_sha256=rs,
        expected_active_pointer_sha256=None,
        stop_after_optimizer_steps=8,
        stop_after_completed_val_epochs=1,
        max_invocation_seconds=5400,
    )
    bp = tmp_path / "budget.json"
    bp.write_text(json.dumps(budget))
    kwargs = dict(
        recipe_path=rp,
        recipe_sha256=rs,
        candidate_gate_path=gp,
        candidate_gate_sha256=gs,
        candidate_execution_budget_path=bp,
        candidate_execution_budget_sha256=digest(bp),
    )
    return recipe, budget, kwargs


@pytest.mark.parametrize(
    "mode",
    ["valid", "missing", "missing_digest", "changed", "wrong_recipe", "missing_gate"],
)
def test_launcher_requires_valid_budget_and_existing_gate(tmp_path, monkeypatch, mode):
    recipe, budget, kwargs = setup(tmp_path, monkeypatch)
    if mode == "missing":
        kwargs.update(
            candidate_execution_budget_path=None, candidate_execution_budget_sha256=None
        )
    elif mode == "missing_digest":
        kwargs["candidate_execution_budget_sha256"] = None
    elif mode == "changed":
        kwargs["candidate_execution_budget_path"].write_text("{}")
    elif mode == "wrong_recipe":
        budget["recipe_sha256"] = "a" * 64
        kwargs["candidate_execution_budget_path"].write_text(json.dumps(budget))
        kwargs["candidate_execution_budget_sha256"] = digest(
            kwargs["candidate_execution_budget_path"]
        )
    elif mode == "missing_gate":
        kwargs["candidate_gate_path"] = None
    if mode != "valid":
        with pytest.raises(launcher.PretestTechnicalLaunchError):
            launcher.build_pretest_technical_launch(**kwargs)
        return
    before = kwargs["recipe_path"].read_bytes()
    command, environment, validated = launcher.build_pretest_technical_launch(**kwargs)
    assert validated == recipe and kwargs["recipe_path"].read_bytes() == before
    assert command[command.index("--candidate-execution-budget-json") + 1] == str(
        kwargs["candidate_execution_budget_path"]
    )
    assert (
        command[command.index("--candidate-execution-budget-sha256") + 1]
        == kwargs["candidate_execution_budget_sha256"]
    )
    assert "--candidate-gate-json" in command
    assert not any("BUDGET" in key for key in environment)
    assert validated["trainer_cli"]["epochs"] == 30


@pytest.mark.parametrize("mode", ["valid", "missing", "changed", "missing_gate"])
def test_actual_direct_cli_revalidates_before_cuda(tmp_path, monkeypatch, mode):
    _, _, kwargs = setup(tmp_path, monkeypatch)
    command, _, _ = launcher.build_pretest_technical_launch(**kwargs)
    module = "gx1.models.entry_v10.entry_v10_ctx_train_v3"
    argv = command[command.index(module) + 1 :]
    if mode in {"missing", "missing_gate"}:
        names = (
            ("--candidate-execution-budget-json", "--candidate-execution-budget-sha256")
            if mode == "missing"
            else ("--candidate-gate-json", "--candidate-gate-sha256")
        )
        for name in names:
            i = argv.index(name)
            del argv[i : i + 2]
    if mode == "changed":
        kwargs["candidate_execution_budget_path"].write_text("{}")
    monkeypatch.setattr(sys, "argv", [module, *argv])
    monkeypatch.setattr(trainer, "_require_trainer_cgroup_preflight", lambda: None)
    monkeypatch.setattr(trainer, "_enforce_canonical_train_env_contract", lambda: None)
    monkeypatch.setattr(
        launch_contract,
        "require_training_recipe_execution_provenance",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        gate_contract,
        "require_pretest_candidate_launch_gate",
        lambda *_args, **_kwargs: {},
    )

    def boundary(**_kwargs):
        raise RuntimeError("REACHED_CUDA_BOUNDARY")

    def forbidden(*_args, **_kwargs):
        pytest.fail("no CUDA or training in CLI contract test")

    monkeypatch.setattr(trainer, "_require_cuda_trainer_guard_execution", boundary)
    monkeypatch.setattr(trainer, "_resolve_device", forbidden)
    monkeypatch.setattr(trainer, "run_train", forbidden)
    error = (
        "REACHED_CUDA_BOUNDARY"
        if mode == "valid"
        else "CANDIDATE_GATE_REQUIRED"
        if mode == "missing_gate"
        else "CANDIDATE_EXECUTION_BUDGET"
    )
    with pytest.raises(RuntimeError, match=error):
        trainer.main()


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"profile": "smoke"},
        {"export_override": True},
        {"precision_policy": precision.DETERMINISTIC_FP32},
    ],
)
def test_run_train_budget_boundary_binds_provenance_and_scope(
    tmp_path, monkeypatch, change
):
    _, _, kwargs = setup(tmp_path, monkeypatch)
    args = dict(
        recipe_source_provenance={
            "recipe_audit_path": str(kwargs["recipe_path"]),
            "recipe_audit_sha256": kwargs["recipe_sha256"],
        },
        profile="candidate",
        precision_policy=precision.EXPERIMENTAL_FP32_3090_NO_UNINITIALIZED_FILL,
        export_override=False,
    )
    args.update(change)
    if change:
        with pytest.raises(RuntimeError, match="CANDIDATE_EXECUTION_BUDGET"):
            trainer._candidate_execution_budget_for_training(
                kwargs["candidate_execution_budget_path"],
                kwargs["candidate_execution_budget_sha256"],
                **args,
            )
    else:
        actual = trainer._candidate_execution_budget_for_training(
            kwargs["candidate_execution_budget_path"],
            kwargs["candidate_execution_budget_sha256"],
            **args,
        )
        assert actual["stop_after_optimizer_steps"] == 8
        kwargs["recipe_path"].write_text("{}")
        with pytest.raises(RuntimeError, match="RECIPE_CHANGED"):
            trainer._candidate_execution_budget_for_training(
                kwargs["candidate_execution_budget_path"],
                kwargs["candidate_execution_budget_sha256"],
                **args,
            )


def test_pause_catch_returns_before_candidate_export_and_catches_only_pause():
    tree = ast.parse(inspect.getsource(trainer.run_train))
    handlers = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ExceptHandler)
        and isinstance(n.type, ast.Name)
        and n.type.id == "_CandidateExecutionPaused"
    ]
    assert len(handlers) == 1
    handler = handlers[0]
    assert isinstance(handler.body[-1], ast.Return) and handler.body[-1].value is None
    assert [
        n.func.id
        for n in ast.walk(handler)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ] == ["_write_candidate_execution_pause_receipt"]


@pytest.mark.parametrize(
    "change",
    [{"subsample_rows": 512}, {"grad_accum_steps": 2}, {"subsample_rows": True}],
)
def test_selected_candidate_keeps_full_population_and_accumulation_one(change):
    kwargs = dict(profile="candidate", epochs=30, grad_accum_steps=1, subsample_rows=0)
    precision.require_local_precision_benchmark_geometry(
        precision.EXPERIMENTAL_FP32_3090_NO_UNINITIALIZED_FILL, **kwargs
    )
    kwargs.update(change)
    with pytest.raises(precision.TrainingPrecisionPolicyError):
        precision.require_local_precision_benchmark_geometry(
            precision.EXPERIMENTAL_FP32_3090_NO_UNINITIALIZED_FILL, **kwargs
        )
