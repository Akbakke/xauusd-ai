"""Window preflight integration over small real bindings, never native admission."""
from __future__ import annotations

import hashlib
import os
from collections import Counter
from pathlib import Path

import pytest

import gx1.contracts.local_random_access_campaign_v2 as hashes
from gx1.contracts.unified_exit_native_candidate_campaign_v1 import WINDOW_SCHEMA
from gx1.scripts import run_unified_exit_native_candidate_window_v1 as window
from tests.test_local_random_access_campaign_v2 import _write


class _BeforeModel(RuntimeError):
    pass


def _bind(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def bound_window(tmp_path, monkeypatch):
    """Keep actual context/window/GENESIS checks; isolate plan and run authority."""
    runtime, output = tmp_path / "runtime", tmp_path / "BUNDLE"
    recipe_path = tmp_path / "RECIPE.json"
    recipe = {"out_bundle_dir": str(output)}
    _write(recipe_path, recipe)
    recipe_binding = _bind(recipe_path)
    identifier = "invocation-0001"
    policy = {
        "schema_version": WINDOW_SCHEMA, "recipe": recipe_binding, "invocation_number": 1,
        "max_invocation_seconds": 12000,
        "budget_path": str(runtime / "budgets" / (identifier + ".json")),
        "progress_path": str(tmp_path / "PROGRESS.json"),
        "campaign_cursor_path": str(runtime / "native-candidate-cursor" / "RESUME_CURSOR.json"),
        "training_session_directory": str(output.parent / (
            window.native.trainer._CANDIDATE_TRAINING_SESSION_DIR_PREFIX + output.name)),
        "test_data_used": False,
    }
    policy["policy_sha256"] = hashes.canonical_sha256(policy)
    policy_path = tmp_path / "WINDOW.json"
    _write(policy_path, policy)
    policy_binding = _bind(policy_path)
    execution_path = tmp_path / "EXECUTION.json"
    _write(execution_path, {
        "prelaunch_manifest": recipe_binding, "train_session_manifest": policy_binding,
        "train_session_manifest_sha256": policy["policy_sha256"],
    })
    invocation = {
        "invocation_sha256": "1" * 64, "invocation_id": identifier,
        "kind": window.NATIVE_KIND, "invocation_number": 1,
        "progress_path": policy["progress_path"], "launcher_argv_sha256": "2" * 64,
        "checkpoint": {"pointer_path": policy["campaign_cursor_path"], "before_mode": "GENESIS"},
        "execution_manifest": _bind(execution_path),
    }
    plan = {"plan_sha256": "3" * 64, "phase": window.NATIVE_PHASE,
            "native_recipe": recipe_binding, "checked_invocations": [invocation],
            "runtime_root": str(runtime)}
    plan_path = tmp_path / "PLAN.json"
    _write(plan_path, {"isolated_plan_authority": True})
    marker = {"schema_version": hashes.ACTIVE_SCHEMA, "plan_sha256": plan["plan_sha256"],
              **{key: invocation[key] for key in (
                  "invocation_sha256", "invocation_number", "invocation_id", "kind", "launcher_argv_sha256")},
              "pointer_before_sha256": "GENESIS"}
    marker["marker_sha256"] = hashes.canonical_sha256(marker)
    marker_path = runtime / "ACTIVE_INVOCATION.json"
    _write(marker_path, marker)
    for key, value in {
        "GX1_CAMPAIGN_PLAN_PATH": str(plan_path),
        "GX1_CAMPAIGN_PLAN_FILE_SHA256": _bind(plan_path)["sha256"],
        "GX1_CAMPAIGN_PLAN_SHA256": plan["plan_sha256"],
        "GX1_CAMPAIGN_INVOCATION_SHA256": invocation["invocation_sha256"],
    }.items():
        monkeypatch.setenv(key, value)

    calls: Counter[Path] = Counter()
    real_hash = hashes._file_sha256_uncached

    def counted(path, *, identity, scoped):
        calls[path] += 1
        return real_hash(path, identity=identity, scoped=scoped)

    monkeypatch.setattr(hashes, "_file_sha256_uncached", counted)
    guarded = []

    def guard(**kwargs):
        assert kwargs == {"execution_tier": "canonical"}
        assert hashes._FILE_VERIFICATION_MEMO.get() is None
        guarded.append(True)

    def plan_authority(value, *, verify_files):
        assert value == {"isolated_plan_authority": True} and verify_files is True
        hashes.require_binding(recipe_binding, label="plan recipe")
        assert hashes.read_bound_json(recipe_path, recipe_binding["sha256"]) == recipe
        return plan

    def scope_authority(value, *, invocation_number):
        assert value == recipe and invocation_number == 1
        hashes.require_binding(recipe_binding, label="native scope recipe")
        return 0

    monkeypatch.setattr(window.native.trainer, "_require_cuda_trainer_guard_execution", guard)
    monkeypatch.setattr(window, "require_plan", plan_authority)
    monkeypatch.setattr(window, "require_native_run_scope", scope_authority)
    monkeypatch.setattr(window, "native_completed_val_ceiling", lambda _: None)
    return dict(policy=policy, recipe=recipe, recipe_binding=recipe_binding, recipe_path=recipe_path,
                policy_path=policy_path, policy_binding=policy_binding, marker_path=marker_path,
                calls=calls, guarded=guarded, plan_authority=plan_authority)


def _run(case):
    return window.run_window(policy_path=case["policy_path"],
                             policy_file_sha256=case["policy_binding"]["sha256"],
                             progress_path=Path(case["policy"]["progress_path"]))


def test_window_preflight_reuses_actual_hashes_and_ends_before_publication_and_model(
    bound_window, monkeypatch
):
    case = bound_window
    published = []
    real_publish = window.campaign._atomic_new

    def publish(path, value):
        assert hashes._FILE_VERIFICATION_MEMO.get() is None
        assert case["calls"][case["recipe_path"]] == 1
        assert value["stop_after_optimizer_steps"] == 0
        assert value["expected_active_pointer_sha256"] is None
        published.append(path)
        real_publish(path, value)

    def before_model(**kwargs):
        assert hashes._FILE_VERIFICATION_MEMO.get() is None
        assert published == [Path(case["policy"]["budget_path"])]
        # Outside preflight these two real verifications must read twice again.
        for _ in range(2):
            hashes.require_binding(case["recipe_binding"], label="execution boundary")
        assert case["calls"][case["recipe_path"]] == 3
        raise _BeforeModel("no actual model execution")

    monkeypatch.setattr(window.campaign, "_atomic_new", publish)
    monkeypatch.setattr(window.native, "run_guarded_native_candidate_invocation", before_model)
    with pytest.raises(_BeforeModel, match="no actual model execution"):
        _run(case)
    assert case["guarded"] == [True]
    assert case["calls"][case["marker_path"]] == 1
    assert hashes._FILE_VERIFICATION_MEMO.get() is None
    assert not Path(case["recipe"]["out_bundle_dir"]).exists()
    assert not Path(case["policy"]["progress_path"]).exists()


@pytest.mark.parametrize("fault", ["wrong_sha", "replaced_inode"])
def test_window_preflight_rechecks_binding_and_resets_after_failure(
    bound_window, tmp_path, monkeypatch, fault
):
    case = bound_window

    def invalid_plan(value, *, verify_files):
        plan = case["plan_authority"](value, verify_files=verify_files)
        if fault == "wrong_sha":
            hashes.require_binding({**case["recipe_binding"], "sha256": "0" * 64}, label="wrong recipe")
        else:
            before = case["recipe_path"].stat()
            data = case["recipe_path"].read_bytes()
            replacement = tmp_path / "replacement"
            replacement.write_bytes(data.replace(b"BUNDLE", b"BUNDLX"))
            os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
            replacement.replace(case["recipe_path"])
            assert case["recipe_path"].stat().st_size == before.st_size
            assert case["recipe_path"].stat().st_mtime_ns == before.st_mtime_ns
            assert case["recipe_path"].stat().st_ino != before.st_ino
        return plan

    def forbidden(**kwargs):
        pytest.fail("invalid preflight must not reach publication or model execution")

    monkeypatch.setattr(window, "require_plan", invalid_plan)
    monkeypatch.setattr(window.campaign, "_atomic_new", forbidden)
    monkeypatch.setattr(window.native, "run_guarded_native_candidate_invocation", forbidden)
    with pytest.raises(hashes.RandomAccessCampaignError, match="SHA-256 mismatch"):
        _run(case)
    assert case["guarded"] == [True]
    assert hashes._FILE_VERIFICATION_MEMO.get() is None
    expected_reads = 1 if fault == "wrong_sha" else 2
    assert case["calls"][case["recipe_path"]] == expected_reads
    hashes.require_binding(_bind(case["recipe_path"]), label="after failed operation")
    assert case["calls"][case["recipe_path"]] == expected_reads + 1
    assert not Path(case["policy"]["budget_path"]).exists()
    assert not Path(case["recipe"]["out_bundle_dir"]).exists()
