"""Resume-only instrumentation: preserve durable state, EMA weights and RNG."""

import ast
import copy
import inspect
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from tests import test_candidate_execution_budget as budget_tests
from tests.test_candidate_execution_staging import Rows, equal_tree


@pytest.mark.parametrize(
    "rows,steps,valid",
    [
        (32, 64, True),
        (128, 128, True),
        (512, 64, True),
        (True, 64, False),
        (31, 64, False),
        (513, 64, False),
        (32, 129, False),
        (32, None, False),
    ],
)
def test_probe_budget_bounds(rows, steps, valid):
    fixture = budget_tests.BudgetTests()
    fixture.setUp()
    try:
        fixture.budget.update(
            resume_probe_val_rows=rows, stop_after_optimizer_steps=steps
        )
        if valid:
            assert fixture.load()["resume_probe_val_rows"] == rows
        else:
            with pytest.raises(ValueError, match="RESUME_PROBE_SCOPE_INVALID"):
                fixture.load()
    finally:
        fixture.doCleanups()


@pytest.mark.parametrize("ema", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_probe_restores_modes_weights_rng_and_leaves_checkpoint_untouched(
    tmp_path, monkeypatch, ema, failure
):
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(0.3))
    model.train()
    model[1].eval()  # Preserve mixed module modes as well.
    target = copy.deepcopy(model)
    target.requires_grad_(False)
    target.eval()
    weight_ema = trainer._WeightEma(model, 0.5) if ema else None
    if weight_ema is not None:
        with torch.no_grad():
            model[0].weight.add_(1)
        weight_ema.update(model)
    pointer = tmp_path / trainer._CANDIDATE_TRAINING_ACTIVE_FILENAME
    pointer.write_text('{"test_checkpoint":1}')
    pointer_before = pointer.read_bytes()
    model_before = copy.deepcopy(model.state_dict())
    target_before = copy.deepcopy(target.state_dict())
    modes = [module.training for root in (model, target) for module in root.modules()]
    rng = trainer._attended_session_rng_state(device=torch.device("cpu"))
    session = SimpleNamespace(directory=tmp_path, contract_sha256="c" * 64)
    observed = []

    def validate(model_arg, target_arg, loader, device_arg, **kwargs):
        assert model_arg is model and target_arg is target
        assert kwargs == {
            "collect_full_exit_trajectory": False,
            "candidate_allow_static_feature_gates": True,
        }
        observed.extend(index for batch in loader for index in batch.tolist())
        assert len(set(observed)) == 32
        equal_tree(
            model.state_dict(), weight_ema.state_dict_clone() if ema else model_before
        )
        model.eval()
        target.eval()
        random.random()
        np.random.random()
        torch.rand(5)
        if failure:
            raise RuntimeError("injected validation failure")
        return (
            2.0,
            float("nan"),
            0.5,
            float("nan"),
            {
                "entry_policy_realized_gross_spread_inclusive_pnl_bps_mean": 0.25,
                "unified_exit_raw_bps_q_mse_mean": 1.5,
                "unified_exit_unique_target_action_agreement": 0.75,
            },
        )

    monkeypatch.setattr(trainer, "validate", validate)
    kwargs = dict(
        model=model,
        target_model=target,
        weight_ema=weight_ema,
        val_ds=Rows(64),
        device=torch.device("cpu"),
        batch_size=8,
        seed=1337,
        requested_rows=32,
        session=session,
        execution_budget_sha256="e" * 64,
        pointer_sha256=trainer._sha256_file(pointer),
    )
    if failure:
        with pytest.raises(RuntimeError, match="injected validation failure"):
            trainer._candidate_resume_validation_probe(**kwargs)
        assert not list(tmp_path.glob("CANDIDATE_RESUME_VAL_PROBE_*.json"))
    else:
        result = trainer._candidate_resume_validation_probe(**kwargs)
        report = json.loads(Path(result["path"]).read_bytes())
        assert (
            report["metrics"]["val_loss"] == 2.0 and report["val_indices"] == observed
        )
        assert (
            report["candidate_selection_changed"] is False
            and report["full_val"] is False
        )
        assert trainer._candidate_resume_validation_probe(**kwargs) == result
        assert len(observed) == 32  # Reuse the bound immutable diagnostic.
    equal_tree(model_before, model.state_dict())
    equal_tree(target_before, target.state_dict())
    equal_tree(rng, trainer._attended_session_rng_state(device=torch.device("cpu")))
    assert modes == [
        module.training for root in (model, target) for module in root.modules()
    ]
    assert pointer.read_bytes() == pointer_before


def test_loss_gradient_and_order_log_is_exact_detached_read_only(caplog):
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    norm = trainer._optimizer_step_with_finite_gradients(model=model, optimizer=opt)
    state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(opt.state_dict())
    rng = trainer._attended_session_rng_state(device=torch.device("cpu"))
    with caplog.at_level("INFO"):
        trainer._record_candidate_resume_probe_step(
            batch_offset=4,
            loss=loss,
            gradient_norm=norm,
            entry_row_indices=torch.tensor([19, 7]),
        )
    event = next(
        record.message
        for record in caplog.records
        if "[CANDIDATE_RESUME_PROBE_STEP]" in record.message
    )
    value = json.loads(event.split("] ", 1)[1])
    assert value["loss"] == float(loss.detach()) and value[
        "gradient_norm_pre_clip"
    ] == float(norm)
    assert value["batch_offset"] == 4 and value["entry_row_indices"] == [19, 7]
    equal_tree(state, model.state_dict())
    equal_tree(optimizer_state, opt.state_dict())
    equal_tree(rng, trainer._attended_session_rng_state(device=torch.device("cpu")))


def test_probe_logs_after_real_optimizer_and_before_durable_pause_callback():
    tree = ast.parse(inspect.getsource(trainer.train_epoch))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]

    def one(name):
        return next(n for n in calls if n.func.id == name)

    assert (
        one("_optimizer_step_with_finite_gradients").lineno
        < one("_record_candidate_resume_probe_step").lineno
        < one("session_checkpoint_hook").lineno
    )
    branch = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name)
        and n.test.id == "session_resume_probe"
    )
    assert any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_record_candidate_resume_probe_step"
        for n in ast.walk(branch)
    )
