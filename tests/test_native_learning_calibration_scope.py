from __future__ import annotations

import copy
import json

import pytest

from gx1.contracts import unified_exit_native_candidate_campaign_v1 as native


@pytest.fixture
def scope(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(native, "__file__", str(root / "gx1/contracts/native.py"))
    def write(name, value):
        path = root / name
        path.write_text(json.dumps(value))
        return {"path": str(path), "sha256": native.file_sha256(path)}
    profile = {"policy_batch_size": 256, "cpu_pipeline_workers": 8,
               "max_wall_seconds": 10800, "progress_interval_forwards": 64}
    risk = {"decision": "PASS", "test_data_used": False, "maximum_holding_seconds": None,
            "absolute_loss_limit_bps": None, "reward_accounting": "liquidation_advantage_v1",
            "economics_objective_schema": "gx1_unified_exit_economics_objective_v4"}
    policy = {"schema_version": "gx1_next_native_run_policy_v1", "canonical_source_repo": str(root),
              "canonical_branch": "work/gx1-current", "training_module": native.NATIVE_MODULE,
              "required_val_profile": profile, "train_batch_size": 16, "precision": "float32",
              "tf32_allowed": False, "native_invocation_seconds": 12000, "outer_guard_seconds": 13800,
              "required_evidence": {"risk_objective": write("risk.json", risk)}, "training_enabled": False,
              "native_learning_calibration": {"schema_version": "gx1_native_learning_calibration_scope_v1",
                  "optimizer_step_ceilings": [16, 32], "full_epoch_training_allowed": False, "test_data_used": False}}
    recipe = {"next_run_policy": write("NEXT_RUN_POLICY.json", policy), "val_limits": profile,
              "trainer_cli": {"batch_size": 16}, "source_bindings_sha256": "a" * 64,
              "candidate_resume_origin": {"schema_version": "gx1_candidate_economics_transition_origin_v1",
                  "contract": write("origin-contract.json", {"fixture": "contract"}),
                  "pointer": write("origin-pointer.json", {"fixture": "pointer"})},
              "files": {"economics_readiness": write("economics.json", {"economics_objective_contract": {
                  "schema_version": risk["economics_objective_schema"], "reward_accounting": risk["reward_accounting"],
                  "contract_sha256": "c" * 64}})}}
    def save():
        recipe["next_run_policy"] = write("NEXT_RUN_POLICY.json", policy)
    return policy, recipe, write, save


def test_stopped_project_permits_only_declared_native_calibration(scope):
    policy, recipe, _, _ = scope
    assert policy["training_enabled"] is False
    assert native.require_native_run_scope(recipe, invocation_number=1) == 16
    assert native.require_native_run_scope(recipe, invocation_number=2) == 32
    for count in (0, 3, True):
        with pytest.raises(RuntimeError, match="CALIBRATION_INVOCATION_INVALID"):
            native.require_native_run_scope(recipe, invocation_number=count)
    for steps in (None, 1, 17, 33, True):
        with pytest.raises(RuntimeError, match="CALIBRATION_STEP_CEILING_INVALID"):
            native.require_native_run_scope(recipe, execution_budget={"stop_after_optimizer_steps": steps})
    for steps in (16, 32):
        assert native.require_native_run_scope(recipe, execution_budget={
            "stop_after_optimizer_steps": steps, "stop_after_completed_val_epochs": None,
            "max_invocation_seconds": 12000}) == steps


@pytest.mark.parametrize("field,value", [("max_invocation_seconds", 60), ("stop_after_completed_val_epochs", 1), ("resume_probe_val_rows", 32)])
def test_calibration_cannot_enable_short_or_legacy_val_probe(scope, field, value):
    _, recipe, _, _ = scope
    budget = {"stop_after_optimizer_steps": 16, "stop_after_completed_val_epochs": None,
              "max_invocation_seconds": 12000, field: value}
    with pytest.raises(RuntimeError, match="NEXT_RUN_BUDGET_INVALID"):
        native.require_native_run_scope(recipe, execution_budget=budget)


def test_policy_file_changed_after_recipe_binding_is_rejected(scope):
    policy, recipe, write, _ = scope
    policy["training_enabled"] = True
    write("NEXT_RUN_POLICY.json", policy)  # Deliberately keep the old recipe digest.
    with pytest.raises(RuntimeError):
        native.require_native_run_scope(recipe, invocation_number=1)


def test_enabling_full_training_does_not_skip_missing_measurements(scope):
    policy, recipe, _, save = scope
    policy["training_enabled"] = True
    save()
    with pytest.raises(RuntimeError):
        native.require_native_run_scope(recipe, invocation_number=1)


def test_full_training_requires_each_measurement_bound_to_actual_objective_and_source(scope):
    policy, recipe, write, save = scope
    roles = ("checkpoint_transition", "learning_calibration", "gpu_batch256_parity",
             "end_to_end_throughput", "resume_equivalence")
    proof = {"decision": "PASS", "test_data_used": False,
             "economics_objective_contract_sha256": "c" * 64, "source_bindings_sha256": "a" * 64,
             "training_origin_pointer_sha256": recipe["candidate_resume_origin"]["pointer"]["sha256"], "native_val_profile": recipe["val_limits"]}
    for role in roles:
        policy["required_evidence"][role] = write(role + ".json", {**proof, "evidence_role": role})
    policy["training_enabled"] = True
    save()
    assert native.require_native_run_scope(recipe, invocation_number=1) is None
    for field in ("economics_objective_contract_sha256", "source_bindings_sha256",
                  "training_origin_pointer_sha256", "evidence_role", "native_val_profile"):
        changed = {**proof, "evidence_role": "gpu_batch256_parity", field: "wrong"}
        policy["required_evidence"]["gpu_batch256_parity"] = write("gpu_batch256_parity.json", changed)
        save()
        with pytest.raises(RuntimeError, match="NEXT_RUN_EVIDENCE_NOT_PASS"):
            native.require_native_run_scope(recipe, invocation_number=1)


def test_no_calibration_scope_means_no_training(scope):
    policy, recipe, _, save = scope
    policy.pop("native_learning_calibration")
    save()
    with pytest.raises(RuntimeError, match="TRAINING_BLOCKED_CALIBRATION_SCOPE_REQUIRED"):
        native.require_native_run_scope(recipe, invocation_number=1)


def test_old_economics_cannot_be_reinterpreted_as_v4(scope):
    _, recipe, write, _ = scope
    recipe["files"]["economics_readiness"] = write("old-economics.json", {"economics_objective_contract": {
        "schema_version": "gx1_unified_exit_economics_objective_v2"}})
    with pytest.raises(RuntimeError, match="NEXT_RUN_RISK_OBJECTIVE_INVALID"):
        native.require_native_run_scope(recipe, invocation_number=1)


@pytest.mark.parametrize("origin", [None, {}, {"pointer": None}])
def test_missing_origin_cannot_fall_back_to_seed(scope, origin):
    _, recipe, _, _ = scope
    recipe["candidate_resume_origin"] = origin
    with pytest.raises(RuntimeError, match="ECONOMICS_TRANSITION_ORIGIN_REQUIRED"):
        native.require_native_run_scope(recipe, invocation_number=1)


def test_declared_exit_baseline_is_bound_to_policy(scope):
    policy, recipe, _, save = scope
    recipe["candidate_resume_origin"]["exit_value_initialization"] = "close_now_baseline_v1"
    with pytest.raises(RuntimeError, match="EXIT_VALUE_INITIALIZATION_POLICY_MISMATCH"):
        native.require_native_run_scope(recipe, invocation_number=1)
    policy["exit_value_initialization"] = "close_now_baseline_v1"
    save()
    assert native.require_native_run_scope(recipe, invocation_number=1) == 16
    assert native.require_native_run_scope(recipe, invocation_number=2) == 32
    recipe["candidate_resume_origin"].pop("exit_value_initialization")
    with pytest.raises(RuntimeError, match="EXIT_VALUE_INITIALIZATION_POLICY_MISMATCH"):
        native.require_native_run_scope(recipe, invocation_number=1)


@pytest.mark.parametrize("value", [None, True, 0, {}, "unknown"])
def test_unknown_or_untyped_origin_initialization_is_rejected(scope, value):
    _, recipe, _, _ = scope
    recipe["candidate_resume_origin"]["exit_value_initialization"] = value
    with pytest.raises(RuntimeError, match="ECONOMICS_TRANSITION_ORIGIN_REQUIRED"):
        native.require_native_run_scope(recipe, invocation_number=1)


@pytest.mark.parametrize("value", [None, True, 0, {}, "unknown"])
def test_unknown_or_untyped_policy_initialization_is_rejected(scope, value):
    policy, recipe, _, save = scope
    policy["exit_value_initialization"] = value
    save()
    with pytest.raises(RuntimeError, match="EXIT_VALUE_INITIALIZATION_POLICY_MISMATCH"):
        native.require_native_run_scope(recipe, invocation_number=1)


def test_optional_initialization_does_not_allow_other_origin_fields(scope):
    _, recipe, _, _ = scope
    recipe["candidate_resume_origin"]["reset_encoder"] = True
    with pytest.raises(RuntimeError, match="ECONOMICS_TRANSITION_ORIGIN_REQUIRED"):
        native.require_native_run_scope(recipe, invocation_number=1)


def test_full_training_baseline_cannot_reuse_preserve_variant_evidence(scope):
    policy, recipe, write, save = scope
    policy["exit_value_initialization"] = "close_now_baseline_v1"
    recipe["candidate_resume_origin"]["exit_value_initialization"] = "close_now_baseline_v1"
    policy["training_enabled"] = True
    roles = ("checkpoint_transition", "learning_calibration", "gpu_batch256_parity",
             "end_to_end_throughput", "resume_equivalence")
    proof = {"decision": "PASS", "test_data_used": False,
             "economics_objective_contract_sha256": "c" * 64, "source_bindings_sha256": "a" * 64,
             "training_origin_pointer_sha256": recipe["candidate_resume_origin"]["pointer"]["sha256"],
             "native_val_profile": recipe["val_limits"]}
    for role in roles:
        policy["required_evidence"][role] = write(role + ".json", {**proof, "evidence_role": role})
    save()
    with pytest.raises(RuntimeError, match="NEXT_RUN_EVIDENCE_NOT_PASS"):
        native.require_native_run_scope(recipe, invocation_number=1)
    for role in roles:
        policy["required_evidence"][role] = write(role + ".json", {
            **proof, "evidence_role": role, "exit_value_initialization": "close_now_baseline_v1",
        })
    save()
    assert native.require_native_run_scope(recipe, invocation_number=1) is None
    for role in roles:
        policy["required_evidence"][role] = write(role + ".json", {**proof, "evidence_role": role})
        save()
        with pytest.raises(RuntimeError, match="NEXT_RUN_EVIDENCE_NOT_PASS"):
            native.require_native_run_scope(recipe, invocation_number=1)
        policy["required_evidence"][role] = write(role + ".json", {
            **proof, "evidence_role": role, "exit_value_initialization": "close_now_baseline_v1",
        })
