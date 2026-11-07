"""CLI routing only: no data, CUDA, model, output, or policy claim."""
from __future__ import annotations

import builtins
import json
from pathlib import Path
import sys
import types

import pytest

from gx1.scripts import validate_entry_model_native_technical_checkpoint_v1 as technical

PLAN = "--market-state-five-endpoint-zero-step-plan"
SHA = "--market-state-five-endpoint-zero-step-plan-sha256"
MODULE = "gx1.audit.verify_market_state_five_endpoint_zero_step_v1"


def _args():
    return [PLAN, "/bound/plan.json", SHA, "a" * 64,
            "--device", "cuda", "--out-dir", "/bound/result"]


def _forbid_dispatch(monkeypatch):
    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == MODULE:
            raise AssertionError("worker imported before CLI rejection")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(technical, "run", lambda **kwargs:
                        (_ for _ in ()).throw(AssertionError("legacy route dispatched")))


@pytest.mark.parametrize("equals_form", [False, True])
def test_dispatch_forwards_only_four_bound_arguments(monkeypatch, capsys, equals_form):
    calls = []
    module = types.ModuleType(MODULE)

    def run(**kwargs):
        calls.append(kwargs)
        return {"status": "synthetic_dispatch_only"}

    module.run = run
    monkeypatch.setitem(sys.modules, MODULE, module)
    monkeypatch.setattr(technical, "run", lambda **kwargs:
                        (_ for _ in ()).throw(AssertionError("legacy route dispatched")))
    args = _args()
    if equals_form:
        args = [args[i] + "=" + args[i + 1] for i in range(0, len(args), 2)]
    assert technical.main(args) == 0
    assert calls == [{"plan_path": Path("/bound/plan.json"), "plan_sha256": "a" * 64,
                      "device": "cuda", "out_dir": Path("/bound/result")}]
    assert json.loads(capsys.readouterr().out) == {"status": "synthetic_dispatch_only"}


@pytest.mark.parametrize("option", [PLAN, SHA, "--device", "--out-dir"])
@pytest.mark.parametrize("equals_form", [False, True])
def test_duplicate_flag_rejected_before_worker_import(monkeypatch, option, equals_form):
    _forbid_dispatch(monkeypatch)
    args = _args()
    value = args[args.index(option) + 1]
    args += [option + "=" + value] if equals_form else [option, value]
    with pytest.raises(SystemExit) as exc:
        technical.main(args)
    assert exc.value.code == 2


@pytest.mark.parametrize("option", [PLAN, SHA, "--device", "--out-dir"])
def test_missing_flag_rejected_before_worker_import(monkeypatch, option):
    _forbid_dispatch(monkeypatch)
    args = _args()
    index = args.index(option)
    del args[index:index + 2]
    with pytest.raises(SystemExit) as exc:
        technical.main(args)
    assert exc.value.code == 2


@pytest.mark.parametrize("change", ["cpu", "relative_output", "extra_flag", "legacy_mode",
                                    "abbreviation", "help"])
def test_wrong_device_output_or_other_flags_rejected(monkeypatch, change):
    _forbid_dispatch(monkeypatch)
    args = _args()
    if change == "cpu":
        args[args.index("--device") + 1] = "cpu"
    elif change == "relative_output":
        args[args.index("--out-dir") + 1] = "relative/result"
    elif change == "extra_flag":
        args += ["--train"]
    elif change == "legacy_mode":
        args += ["--dynamic-compact-input-influence-plan", "/legacy/plan.json"]
    elif change == "abbreviation":
        args[args.index("--out-dir")] = "--out"
    else:
        args += ["--help"]
    with pytest.raises(SystemExit) as exc:
        technical.main(args)
    assert exc.value.code == 2


def test_sha_selector_first_and_sys_argv(monkeypatch):
    calls = []
    module = types.ModuleType(MODULE)
    module.run = lambda **kwargs: calls.append(kwargs) or {"status": "synthetic_dispatch_only"}
    monkeypatch.setitem(sys.modules, MODULE, module)
    args = _args()
    monkeypatch.setattr(sys, "argv", ["validator", *args[2:4], *args[0:2], *args[4:]])
    assert technical.main() == 0
    assert calls[0]["plan_path"] == Path("/bound/plan.json")
