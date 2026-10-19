from __future__ import annotations

import ast
from collections import Counter
import hashlib
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

import gx1.contracts.local_random_access_campaign_v2 as owner


SOURCE = Path(__file__).resolve().parents[1] / "gx1/scripts/run_unified_exit_random_access_full_train_v1.py"


class _ModelBoundaryStop(RuntimeError):
    pass


class _PreflightFailure(RuntimeError):
    pass


def _load_guarded_invocation():
    """Execute only the actual entrypoint definition, without trainer/model imports."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    function = next(node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_guarded_native_candidate_invocation")
    scope_import = next(node for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "gx1.contracts.local_random_access_campaign_v2"
        and any(alias.name == "file_verification_scope" for alias in node.names))
    isolated = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        scope_import, function,
    ], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace = {"Path": Path, "time": time}
    exec(compile(isolated, str(SOURCE), "exec"), namespace)
    return namespace[function.name], namespace


@pytest.fixture
def physical_hash_calls(monkeypatch):
    calls: Counter[Path] = Counter()
    original = owner._file_sha256_uncached

    def traced(path, *, identity, scoped):
        calls[path] += 1
        return original(path, identity=identity, scoped=scoped)

    monkeypatch.setattr(owner, "_file_sha256_uncached", traced)
    return calls


def _preflight_fixture(tmp_path, monkeypatch, *, fault=None):
    path = tmp_path / "PHYSICAL.json"
    data = b'{"source": "bounded actual fixture bytes"}\n'
    path.write_bytes(data)
    binding = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
    events = []
    recipe = {"val_limits": {"max_wall_seconds": 60}, "trainer_cli": {},
        "out_bundle_dir": str(tmp_path / "unused-output")}
    budget = {"max_invocation_seconds": 12000}
    invoke, namespace = _load_guarded_invocation()

    def read(label):
        events.append(label)
        owner.require_binding(binding, label=label)
        if fault == label:
            raise _PreflightFailure(label)

    def cuda_guard(*, execution_tier):
        assert execution_tier == "canonical"
        read("cuda_guard")

    def recipe_boundary(recipe_path, recipe_file_sha256):
        assert recipe_path == tmp_path / "RECIPE.json"
        assert recipe_file_sha256 == "1" * 64
        read("recipe")
        return recipe, {"physical": path}, {"source": "fixture"}

    def budget_boundary(execution_budget_path, execution_budget_file_sha256, **kwargs):
        assert execution_budget_path == tmp_path / "BUDGET.json"
        assert execution_budget_file_sha256 == "2" * 64
        assert kwargs == {"recipe_path": tmp_path / "RECIPE.json",
            "recipe_sha256": "1" * 64, "recipe": recipe}
        read("budget")
        if fault == "wrong_expected_sha":
            owner.require_binding({**binding, "sha256": "0" * 64}, label="wrong expected SHA")
        if fault == "invalid_budget":
            return {**budget, "resume_probe_val_rows": 1}
        return budget

    def scope_boundary(value, *, execution_budget):
        assert value is recipe and execution_budget is budget
        read("scope")
        return {"validated": "fixture"}

    def before_device_and_model(device):
        assert device == "cuda"
        read("before_device_and_model")
        raise _ModelBoundaryStop("before any device/model/checkpoint execution")

    namespace.update(
        trainer=SimpleNamespace(_require_cuda_trainer_guard_execution=cuda_guard,
            _resolve_device=before_device_and_model),
        _require_native_full_train_recipe=recipe_boundary,
        launch_owner=SimpleNamespace(require_candidate_execution_budget=budget_boundary),
    )
    # The local import retains its source-owned name. Only its semantic admission
    # boundary is isolated; all file/SHA work above uses the genuine owners.
    scope_module = ModuleType("gx1.contracts.unified_exit_native_candidate_campaign_v1")
    scope_module.require_native_run_scope = scope_boundary
    monkeypatch.setitem(sys.modules, scope_module.__name__, scope_module)
    arguments = {"recipe_path": tmp_path / "RECIPE.json", "recipe_file_sha256": "1" * 64,
        "execution_budget_path": tmp_path / "BUDGET.json", "execution_budget_file_sha256": "2" * 64}
    return invoke, arguments, path, binding, events


def test_native_preflight_reuses_real_hashes_and_closes_before_device_model_execution(
    tmp_path, monkeypatch, physical_hash_calls
):
    invoke, arguments, path, binding, events = _preflight_fixture(tmp_path, monkeypatch)
    for operation in (1, 2):
        with pytest.raises(_ModelBoundaryStop, match="before any device/model/checkpoint"):
            invoke(**arguments)
        # Guard and device/model boundary each perform fresh reads outside the
        # scope. Recipe/budget/run-scope checks share one actual physical digest.
        assert physical_hash_calls[path] == operation * 3
        assert events[(operation - 1) * 5:operation * 5] == [
            "cuda_guard", "recipe", "budget", "scope", "before_device_and_model"]
    owner.require_binding(binding, label="outside both completed preflights")
    assert physical_hash_calls[path] == 7
    assert not (tmp_path / "unused-output").exists()


@pytest.mark.parametrize("fault", ["recipe", "budget", "scope", "wrong_expected_sha", "invalid_budget"])
def test_native_preflight_failure_preserves_guards_and_resets_scope(
    tmp_path, monkeypatch, physical_hash_calls, fault
):
    invoke, arguments, path, binding, events = _preflight_fixture(tmp_path, monkeypatch, fault=fault)
    expected = (owner.RandomAccessCampaignError if fault == "wrong_expected_sha"
        else RuntimeError if fault == "invalid_budget" else _PreflightFailure)
    match = ("SHA-256 mismatch" if fault == "wrong_expected_sha"
        else "NATIVE_FULL_TRAIN_INVOCATION_BUDGET_INVALID" if fault == "invalid_budget" else fault)
    for operation in (1, 2):
        with pytest.raises(expected, match=match):
            invoke(**arguments)
        assert physical_hash_calls[path] == operation * 3 - 1
        owner.require_binding(binding, label="outside failed preflight")
        assert physical_hash_calls[path] == operation * 3
    assert "before_device_and_model" not in events
    assert not (tmp_path / "unused-output").exists()
