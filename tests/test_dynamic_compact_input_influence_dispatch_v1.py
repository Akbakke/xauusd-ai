"""Dispatch-only tests: no input packet, state load, model or CUDA calls."""
from __future__ import annotations

import builtins
import json
from pathlib import Path
import sys
import types

import pytest

from gx1.scripts import validate_entry_model_native_technical_checkpoint_v1 as technical


_PLAN = "--dynamic-compact-input-influence-plan"
_SHA = "--dynamic-compact-input-influence-plan-sha256"
_MODULE = "gx1.audit.dynamic_compact_input_influence_v1"


def _args():
    return [_PLAN, "/bound/plan.json", _SHA, "a" * 64,
            "--device", "cuda", "--out-dir", "/bound/result"]


def _no_dispatch(monkeypatch):
    original = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == _MODULE:
            raise AssertionError("audit imported before CLI rejection")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    def legacy_run(**kwargs):
        raise AssertionError("legacy run was dispatched")
    monkeypatch.setattr(technical, "run", legacy_run)


@pytest.mark.parametrize("equals_form", [False, True])
def test_dynamic_dispatch_passes_only_explicit_bound_arguments(monkeypatch, capsys, equals_form):
    calls = []
    module = types.ModuleType(_MODULE)
    def run(**kwargs):
        calls.append(kwargs)
        return {"status": "synthetic_dispatch_only"}
    module.run = run
    monkeypatch.setitem(sys.modules, _MODULE, module)
    def forbidden(**kwargs):
        raise AssertionError("legacy validation must not run")
    monkeypatch.setattr(technical, "run", forbidden)
    args = _args()
    if equals_form:
        args = [args[i] + "=" + args[i + 1] for i in range(0, len(args), 2)]
    assert technical.main(args) == 0
    assert calls == [{"plan_path": Path("/bound/plan.json"), "plan_sha256": "a" * 64,
                      "device": "cuda", "out_dir": Path("/bound/result")}]
    assert json.loads(capsys.readouterr().out) == {"status": "synthetic_dispatch_only"}


@pytest.mark.parametrize("option", [_PLAN, _SHA, "--device", "--out-dir"])
@pytest.mark.parametrize("equals_form", [False, True])
def test_dynamic_dispatch_rejects_every_duplicate_before_import(monkeypatch, option, equals_form):
    _no_dispatch(monkeypatch)
    args = _args()
    value = args[args.index(option) + 1]
    args += [option + "=" + value] if equals_form else [option, value]
    with pytest.raises(SystemExit) as exc:
        technical.main(args)
    assert exc.value.code == 2


@pytest.mark.parametrize("option", [_PLAN, _SHA, "--device", "--out-dir"])
def test_dynamic_dispatch_rejects_missing_fields_before_import(monkeypatch, option):
    _no_dispatch(monkeypatch)
    args = _args()
    index = args.index(option)
    del args[index:index + 2]
    with pytest.raises(SystemExit) as exc:
        technical.main(args)
    assert exc.value.code == 2


@pytest.mark.parametrize("extra", [
    ["--train-parquet", "/legacy/train.parquet"],
    ["--batch-size", "16"], ["--dry-run"], ["--train"],
    ["--unknown"], ["--help"], ["--out", "/abbreviation"],
])
def test_dynamic_dispatch_rejects_legacy_unknown_and_abbreviated_flags(monkeypatch, extra):
    _no_dispatch(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        technical.main(_args() + extra)
    assert exc.value.code == 2


@pytest.mark.parametrize("defect", ["cpu", "relative_output"])
def test_dynamic_dispatch_requires_cuda_and_absolute_output_before_import(monkeypatch, defect):
    _no_dispatch(monkeypatch)
    args = _args()
    args[args.index("--device") + 1] = "cpu" if defect == "cpu" else "cuda"
    if defect == "relative_output":
        args[args.index("--out-dir") + 1] = "relative/result"
    with pytest.raises(SystemExit) as exc:
        technical.main(args)
    assert exc.value.code == 2


def test_dynamic_dispatch_accepts_sha_selector_first_and_sys_argv(monkeypatch):
    calls = []
    module = types.ModuleType(_MODULE)
    def run(**kwargs):
        calls.append(kwargs)
        return {"status": "synthetic_dispatch_only"}
    module.run = run
    monkeypatch.setitem(sys.modules, _MODULE, module)
    args = _args()
    monkeypatch.setattr(sys, "argv", ["validator", *args[2:4], *args[0:2], *args[4:]])
    assert technical.main() == 0
    assert calls[0]["plan_path"] == Path("/bound/plan.json")


def test_legacy_dispatch_keeps_existing_arguments_and_run(monkeypatch):
    calls = []
    def run(**kwargs):
        calls.append(kwargs)
        return {"decision": "synthetic_legacy_dispatch_only"}
    monkeypatch.setattr(technical, "run", run)
    original = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == _MODULE:
            raise AssertionError("dynamic audit was imported on legacy route")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    pairs = [("--bundle-dir", "/legacy/bundle"), ("--checkpoint-session-dir", "/legacy/session"),
             ("--train-parquet", "/legacy/train"), ("--val-parquet", "/legacy/val"),
             ("--m5-prebuilt", "/legacy/m5"), ("--multi-tf-cache-dir", "/legacy/cache"),
             ("--lifecycle-manifest", "/legacy/lifecycle"), ("--val-sequence-source-audit", "/legacy/audit"),
             ("--test-guard-json", "/legacy/guard"), ("--test-guard-sha256", "b" * 64),
             ("--device", "cpu"), ("--out-dir", "/legacy/output")]
    args = [part for pair in pairs for part in pair] + ["--dry-run"]
    assert technical.main(args) == 0
    assert len(calls) == 1
    assert calls[0] == {"bundle_dir": Path("/legacy/bundle"), "session_dir": Path("/legacy/session"),
        "train_parquet": Path("/legacy/train"), "val_parquet": Path("/legacy/val"),
        "m5_prebuilt": Path("/legacy/m5"), "multi_tf_cache_dir": Path("/legacy/cache"),
        "lifecycle_manifest": Path("/legacy/lifecycle"), "val_sequence_source_audit": Path("/legacy/audit"),
        "test_guard_json": Path("/legacy/guard"), "test_guard_sha256": "b" * 64,
        "device": "cpu", "batch_size": 8, "out_dir": Path("/legacy/output"),
        "max_validation_batches": None, "validation_session_dir": None, "dry_run": True}
