"""Synthetic readiness dispatch tests; authority semantics have their own tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gx1.contracts import entry_native_direction_research_campaign_v1 as research
from gx1.contracts import local_random_access_campaign_v2 as campaign
from scripts import collect_gx1_handover_readonly as collector


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _research_window(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    binding = _write_json(tmp_path / "window.json", {
        "schema_version": research.WINDOW_SCHEMA,
        "max_invocation_seconds": 12000,
        "test_data_used": False,
    })
    arguments = {"native_window_policy": Path(binding["path"]),
                 "native_window_policy_file_sha256": binding["sha256"]}
    return repo, binding, arguments


def _forbid_research_dispatch(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("research authority must not be dispatched")
    monkeypatch.setattr(research, "require_native_direction_research_readiness", unexpected)


def _legacy_policy_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    clock = repo / "clock.ps1"
    clock.write_text("# synthetic clock source\n")
    policy = {
        "schema_version": "gx1_next_native_run_policy_v1",
        "canonical_source_repo": str(repo), "canonical_branch": "work/gx1-current",
        "required_val_profile": {"policy_batch_size": 256, "cpu_pipeline_workers": 8,
                                 "max_wall_seconds": 10800, "progress_interval_forwards": 64},
        "native_invocation_seconds": 12000, "outer_guard_seconds": 13800,
        "train_batch_size": 16, "precision": "float32", "tf32_allowed": False,
        "training_module": "gx1.scripts.run_unified_exit_native_candidate_window_v1",
        "gpu_clock_launcher": {"path": str(clock),
                               "sha256": hashlib.sha256(clock.read_bytes()).hexdigest()},
        "training_enabled": True, "required_evidence": {},
    }
    _write_json(repo / "NEXT_RUN_POLICY.json", policy)

    def git(_repo, *args):
        assert _repo == repo
        return {("branch", "--show-current"): "work/gx1-current",
                ("rev-parse", "HEAD"): "d" * 40,
                ("status", "--porcelain=v1", "--untracked-files=all"): ""}[args]
    monkeypatch.setattr(collector, "_git", git)
    return repo, policy


def test_research_readiness_dispatches_only_after_exact_window_binding(tmp_path, monkeypatch):
    repo, binding, arguments = _research_window(tmp_path)
    expected = {"decision": "READY_SYNTHETIC_DISPATCH_WITNESS", "training_started": False}
    calls = []

    def authority(actual_repo, *, window_path, window_file_sha256):
        assert actual_repo == repo and window_path == Path(binding["path"])
        assert window_file_sha256 == hashlib.sha256(window_path.read_bytes()).hexdigest()
        calls.append((actual_repo, window_path, window_file_sha256))
        return expected
    monkeypatch.setattr(research, "require_native_direction_research_readiness", authority)
    assert collector.next_run_readiness(repo, **arguments) is expected
    assert calls == [(repo, Path(binding["path"]), binding["sha256"])]
    assert not (repo / "NEXT_RUN_POLICY.json").exists()  # Old-profile gates were not entered.


@pytest.mark.parametrize("fault", ["window_bytes", "digest"])
def test_research_readiness_rejects_changed_window_binding_before_authority(tmp_path, monkeypatch, fault):
    repo, binding, arguments = _research_window(tmp_path)
    _forbid_research_dispatch(monkeypatch)
    if fault == "window_bytes":
        Path(binding["path"]).write_text(json.dumps({"schema_version": research.WINDOW_SCHEMA,
                                                   "max_invocation_seconds": 1}))
    else:
        arguments["native_window_policy_file_sha256"] = "0" * 64
    with pytest.raises(campaign.RandomAccessCampaignError, match="SHA-256"):
        collector.next_run_readiness(repo, **arguments)


def test_research_readiness_propagates_authority_rejection_without_legacy_fallback(tmp_path, monkeypatch):
    repo, _binding, arguments = _research_window(tmp_path)

    def rejected(*args, **kwargs):
        raise RuntimeError("SYNTHETIC_RESEARCH_ADMISSION_REJECTED")
    monkeypatch.setattr(research, "require_native_direction_research_readiness", rejected)
    with pytest.raises(RuntimeError, match="SYNTHETIC_RESEARCH_ADMISSION_REJECTED"):
        collector.next_run_readiness(repo, **arguments)


def test_unbound_research_flags_do_not_create_readiness_exception(tmp_path, monkeypatch):
    repo, policy = _legacy_policy_repo(tmp_path, monkeypatch)
    policy["native_direction_research"] = {"execution_authorized": True}
    _write_json(repo / "NEXT_RUN_POLICY.json", policy)
    _forbid_research_dispatch(monkeypatch)
    result = collector.next_run_readiness(repo)
    assert result["decision"] == "BLOCKED" and result["training_started"] is False
    assert "native_window_binding_required" in result["blocked_reasons"]
    assert "resume_equivalence_missing" in result["blocked_reasons"]


def test_bound_legacy_window_keeps_legacy_readiness_route(tmp_path, monkeypatch):
    from gx1.contracts import unified_exit_native_candidate_campaign_v1 as native
    repo, _policy = _legacy_policy_repo(tmp_path, monkeypatch)
    _forbid_research_dispatch(monkeypatch)
    recipe = {"schema_version": "gx1_unified_exit_random_access_full_train_recipe_v1",
              "profile": "candidate", "test_data_used": False,
              "source_repo": str(repo), "source_commit": "d" * 40}
    recipe["recipe_sha256"] = native.native_sha256(recipe)
    window = {"schema_version": native.WINDOW_SCHEMA, "invocation_number": 1,
              "recipe": _write_json(tmp_path / "recipe.json", recipe)}
    binding = _write_json(tmp_path / "legacy_window.json", window)
    calls = []

    def legacy_window(value):
        assert value == window
        calls.append("window")
        return value

    def legacy_scope(value, *, invocation_number):
        assert value == recipe and invocation_number == 1
        calls.append("scope")
        return 2
    monkeypatch.setattr(native, "require_native_window_policy", legacy_window)
    monkeypatch.setattr(native, "require_native_run_scope", legacy_scope)
    result = collector.next_run_readiness(
        repo, native_window_policy=Path(binding["path"]),
        native_window_policy_file_sha256=binding["sha256"])
    assert calls == ["window", "scope"]
    assert result["decision"] == "READY_FOR_EXISTING_BOUND_CAMPAIGN_GATES"
    assert result["native_window_scope"]["optimizer_step_ceiling"] == 2
    assert result["training_started"] is False


@pytest.mark.parametrize("missing", ["path", "digest"])
def test_readiness_requires_complete_bound_window_pair(tmp_path, monkeypatch, missing):
    _forbid_research_dispatch(monkeypatch)
    arguments = {"native_window_policy": tmp_path / "missing.json",
                 "native_window_policy_file_sha256": "a" * 64}
    arguments.pop("native_window_policy" if missing == "path" else "native_window_policy_file_sha256")
    with pytest.raises(ValueError, match="BINDING_PAIR_REQUIRED"):
        collector.next_run_readiness(tmp_path, **arguments)
