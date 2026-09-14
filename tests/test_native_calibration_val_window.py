"""CPU protocol tests for the native CUDA-window handoff; no GPU measurement."""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import random
import time

import numpy as np
import pytest
import torch

from gx1.scripts import run_unified_exit_random_access_full_train_v1 as wrapper
from tests.test_candidate_native_full_val import _epoch_boundary
from tests.test_candidate_economics_transition import _identical


@pytest.fixture
def window(tmp_path, monkeypatch):
    calibration = {"schema_version": "gx1_native_learning_calibration_run_v1",
                   "arm": "reference", "report_only_val": True}
    session, model, target, ema, state = _epoch_boundary(tmp_path, native_calibration=calibration)
    model.train()
    model[1].eval()  # Preserve mixed module modes, not just model.training.
    contract = wrapper.val._read(session._contract_path)
    recipe = wrapper.val._read(Path(contract["recipe_source_provenance"]["recipe_audit_path"]))
    context = {
        **recipe["val_limits"], "frame": object(), "state_factory": object(),
        "parent_coordinate_evidence": {"fixture": "parent"},
        "val_sequence_audit": tmp_path / "audit.json", "max_model_forwards": 100000,
        "max_state_views": 1000000,
    }
    components = {"model": model, "weight_ema": ema, "val_ds": object(), "native_val_context": context}
    pause = {"phase": "train", "epoch_index": 0, "next_batch_offset": 32,
             "global_optimizer_steps": 32, "session_directory": str(session.directory),
             "active_pointer_sha256": wrapper.val.file_sha256(session._active_path)}
    real_to = torch.nn.Module.to
    to_devices = []
    def cpu_only_to(module, *args, **kwargs):
        if args and isinstance(args[0], torch.device) and args[0].type == "cuda":
            to_devices.append(args[0])
            return module
        return real_to(module, *args, **kwargs)
    monkeypatch.setattr(torch.nn.Module, "to", cpu_only_to)
    capture = wrapper.trainer._attended_session_rng_state
    restore = wrapper.trainer._restore_attended_session_rng_state
    monkeypatch.setattr(wrapper.trainer, "_attended_session_rng_state",
                        lambda *, device: capture(device=torch.device("cpu")))
    monkeypatch.setattr(wrapper.trainer, "_restore_attended_session_rng_state",
                        lambda value, *, device: restore(value, device=torch.device("cpu")))
    calls = []
    signature = inspect.signature(wrapper.val.evaluate_bound_full_val_v1)
    def evaluator(**kwargs):
        signature.bind(**kwargs)  # Check the actual native evaluator API.
        calls.append(kwargs)
        assert kwargs["selected_batch_size"] == 16
        assert kwargs["exit_policy_batch_size"] == 256
        assert kwargs["cpu_pipeline_workers"] == 8
        assert kwargs["compute_guard_max_wall_seconds"] == 10800
        assert kwargs["progress_interval_forwards"] == 64
        assert kwargs["max_forwards_this_invocation"] == kwargs["compute_guard_max_model_forwards"] == 100000
        assert kwargs["compute_guard_max_materialized_state_views"] == 1000000
        assert kwargs["device"].type == "cuda"
        assert kwargs["model"] is model and kwargs["entry_dataset"] is components["val_ds"]
        assert kwargs["frame"] is context["frame"] and kwargs["state_factory"] is context["state_factory"]
        assert not any(module.training for module in model.modules())
        assert not kwargs["candidate_target_model"].training
        assert not any(parameter.requires_grad for parameter in kwargs["candidate_target_model"].parameters())
        binding = kwargs["checkpoint_binding"]
        assert binding["report_only"] is True and binding["immutable_epoch_snapshot"] is False
        assert wrapper.trainer.canonical_model_state_sha256(model.state_dict()) == binding["model_state_sha256"]
        assert wrapper.trainer.canonical_model_state_sha256(kwargs["candidate_target_model"].state_dict()) == binding["target_model_state_sha256"]
        with pytest.raises(RuntimeError, match="REPORT_ONLY_VAL_NOT_SELECTABLE"):
            wrapper.trainer._native_candidate_validation_stats({"checkpoint_binding": binding})
        # Exercise restoration even if an evaluator changes temporary EMA weights/modes/RNG.
        random.random(); np.random.random(); torch.rand(2)
        model.train()
        with torch.no_grad():
            next(model.parameters()).add_(7)
        progress = kwargs["rollout_progress_path"]
        progress.parent.mkdir(parents=True, exist_ok=True)
        progress.write_text(json.dumps({"model_forward_count": 3, "materialized_state_view_count": 768}))
        return {"decision": "PAUSED_RESUMABLE"}
    monkeypatch.setattr(wrapper.val, "evaluate_bound_full_val_v1", evaluator)
    return {"session": session, "model": model, "target": target, "ema": ema, "state": state,
            "recipe": recipe, "components": components, "pause": pause, "calls": calls,
            "to_devices": to_devices, "capture": capture, "evaluator": evaluator,
            "output": tmp_path.resolve() / "CANDIDATE"}


def _run(window, **overrides):
    inputs = dict(components=window["components"], recipe=window["recipe"], output=window["output"],
                  device=torch.device("cuda"), invocation_started=time.monotonic() - 10,
                  pause_evidence=window["pause"])
    inputs.update(overrides)
    return wrapper._run_native_calibration_validation(**inputs)


def test_native_report_window_uses_existing_evaluator_and_preserves_training_state(window):
    model, session = window["model"], window["session"]
    before_model = copy.deepcopy(model.state_dict())
    before_ema = copy.deepcopy(window["ema"].checkpoint_state())
    before_modes = [module.training for module in model.modules()]
    before_rng = window["capture"](device=torch.device("cpu"))
    before_pointer = session._active_path.read_bytes()
    before_state = session.load_checkpoint()
    result = _run(window)
    assert result["executed"] is True and result["report_only"] is True
    assert result["checkpoint_selection_advanced"] is False
    assert len(window["calls"]) == len(window["to_devices"]) == 1
    _identical(model.state_dict(), before_model)
    _identical(window["ema"].checkpoint_state(), before_ema)
    _identical(window["capture"](device=torch.device("cpu")), before_rng)
    _identical(session.load_checkpoint(), before_state)
    assert [module.training for module in model.modules()] == before_modes
    assert session._active_path.read_bytes() == before_pointer
    report = wrapper.val._read(Path(result["path"]))
    assert report["decision"] == "OBSERVATION_REQUIRES_REVIEW"
    assert report["model_forward_count"] == 3 and report["materialized_state_view_count"] == 768
    assert report["learning_calibrated"] is report["profitability_proven"] is False
    assert report["checkpoint_selection_advanced"] is False
    assert report["training_pointer_sha256"] == window["pause"]["active_pointer_sha256"]
    assert report["native_val_profile"] == window["recipe"]["val_limits"]


def test_evaluator_failure_restores_online_modes_and_rng(window, monkeypatch):
    before_model = copy.deepcopy(window["model"].state_dict())
    before_modes = [module.training for module in window["model"].modules()]
    before_rng = window["capture"](device=torch.device("cpu"))
    before_pointer = window["session"]._active_path.read_bytes()
    def fail(**kwargs):
        window["evaluator"](**kwargs)
        raise RuntimeError("fixture evaluator failure")
    monkeypatch.setattr(wrapper.val, "evaluate_bound_full_val_v1", fail)
    with pytest.raises(RuntimeError, match="fixture evaluator failure"):
        _run(window)
    _identical(window["model"].state_dict(), before_model)
    _identical(window["capture"](device=torch.device("cpu")), before_rng)
    assert [module.training for module in window["model"].modules()] == before_modes
    assert window["session"]._active_path.read_bytes() == before_pointer


@pytest.mark.parametrize("field,value", [("phase", "validation"), ("epoch_index", 1),
                                         ("next_batch_offset", 16), ("global_optimizer_steps", 31),
                                         ("epoch_index", False)])
def test_wrong_pause_cursor_is_rejected_before_evaluator(window, field, value):
    window["pause"][field] = value
    with pytest.raises(RuntimeError, match="VAL_SCOPE_INVALID"):
        _run(window)
    assert window["calls"] == window["to_devices"] == []


@pytest.mark.parametrize("field,value", [("policy_batch_size", 128), ("cpu_pipeline_workers", 4),
                                         ("max_wall_seconds", 4200)])
def test_context_cannot_override_bound_native_val_profile(window, field, value):
    window["components"]["native_val_context"][field] = value
    with pytest.raises(RuntimeError, match="VAL_PROFILE_MISMATCH"):
        _run(window)
    assert window["calls"] == window["to_devices"] == []


def test_requested_arm_must_match_session_bound_recipe(window):
    window["recipe"]["native_calibration"]["arm"] = "split"
    with pytest.raises(RuntimeError, match="VAL_RECIPE_MISMATCH"):
        _run(window)
    assert window["calls"] == window["to_devices"] == []


def test_wrong_training_pointer_is_rejected_before_evaluator(window):
    window["pause"]["active_pointer_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="VAL_POINTER_MISMATCH"):
        _run(window)
    assert window["calls"] == window["to_devices"] == []


def test_existing_report_prevents_repeating_val(window):
    path = window["session"].directory / "native_val/calibration_step_0032/CAPACITY_OBSERVATION.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="VAL_OBSERVATION_EXISTS"):
        _run(window)
    assert window["calls"] == window["to_devices"] == []


def test_insufficient_window_does_not_shorten_val_or_touch_training(window):
    before = window["session"]._active_path.read_bytes()
    result = _run(window, invocation_started=time.monotonic() - 2000)
    assert result["executed"] is False and result["reason"] == "insufficient_remaining_native_window"
    assert window["calls"] == window["to_devices"] == []
    assert window["session"]._active_path.read_bytes() == before
