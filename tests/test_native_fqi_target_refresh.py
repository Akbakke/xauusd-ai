from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gx1.contracts import unified_exit_native_candidate_campaign_v1 as native
from gx1.contracts import unified_exit_random_access_val_checkpoint_v1 as ema
from tests.test_native_optimizer_procedure_scope import _write, optimizer_origin, optimizer_scope, optimizer_history, scope
from tests.test_native_training_continuation import continuation_origin, continuation_scope


@pytest.fixture
def fqi_origin(continuation_origin, tmp_path, monkeypatch):
    directory = tmp_path / "stopped91"
    directory.mkdir()
    recipe = _write(tmp_path / "stopped91-recipe.json", {"candidate_resume_origin": continuation_origin})
    contract = _write(directory / "CANDIDATE_TRAINING_SESSION_CONTRACT.json", {
        "recipe_source_provenance": {"recipe_audit_path": recipe["path"], "recipe_audit_sha256": recipe["sha256"]},
    })
    inherited = ema.bind_candidate_weight_ema_history_v1(
        session_contract_path=Path(continuation_origin["contract"]["path"]),
        session_contract_sha256=continuation_origin["contract"]["sha256"],
    )
    _write(directory / native.TRAINING_CONTINUATION_RECEIPT_NAME, {
        "schema_version": native.TRAINING_CONTINUATION_RECEIPT_SCHEMA,
        "origin": continuation_origin, "origin_cursor": native.TRAINING_CONTINUATION_ORIGIN_CURSOR,
        "origin_state_sha256": native.TRAINING_CONTINUATION_ORIGIN_STATE_SHA256,
        "destination_session_contract_sha256": contract["sha256"],
        "inherited_weight_ema_history": inherited, "ema_internal_steps": 25173,
        "global_optimizer_steps": 5265, "state_preserved": True,
        "optimizer_procedure_changed": False, "identical_future_trajectory_claimed": False,
        "preserved_state_fields": ["model_state", "target_model_state", "optimizer_state", "weight_ema_state",
                                   "lr_scheduler_state", "rng_state", "epoch_order", "training_progress"],
    })
    state = directory / "candidate_training_state_slot_0.pt"
    state.write_bytes(b"fixture stopped91; native scope does not deserialize tensors")
    state_sha = native.file_sha256(state)
    pointer = _write(directory / "CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json", {
        **native.FQI_TARGET_REFRESH_ORIGIN_CURSOR,
        "schema_version": "gx1_candidate_training_session_v1", "slot": 0,
        "session_contract_sha256": contract["sha256"], "state_sha256": state_sha,
    })
    monkeypatch.setattr(native, "FQI_TARGET_REFRESH_ORIGIN_CONTRACT_SHA256", contract["sha256"])
    monkeypatch.setattr(native, "FQI_TARGET_REFRESH_ORIGIN_POINTER_SHA256", pointer["sha256"])
    monkeypatch.setattr(native, "FQI_TARGET_REFRESH_ORIGIN_STATE_SHA256", state_sha)
    return {"schema_version": native.FQI_TARGET_REFRESH_SCHEMA,
            "contract": contract, "pointer": pointer,
            "exit_value_initialization": "close_now_baseline_v1",
            "train_population_scope": "latest_year_2025_2026_v1",
            "gradient_clipping_policy": native.OPTIMIZER_PROCEDURE_TRANSITION_POLICY}


@pytest.fixture
def fqi_scope(continuation_scope, fqi_origin):
    policy, recipe, write, save = continuation_scope
    recipe["candidate_resume_origin"] = fqi_origin
    policy["native_learning_calibration"]["schema_version"] = "gx1_native_fqi_target_refresh_scope_v1"
    save()
    return policy, recipe, write, save


def test_fqi_scope_admits_exact_91_and_only_one_5777_window(fqi_scope, continuation_origin):
    _, recipe, _, _ = fqi_scope
    assert native.require_native_run_scope(recipe) == 5777
    assert native.require_native_run_scope(recipe, invocation_number=1) == 5777
    assert native.require_native_run_scope(recipe, execution_budget={
        "stop_after_optimizer_steps": 5777, "stop_after_completed_val_epochs": None,
        "max_invocation_seconds": 12000,
    }) == 5777
    with pytest.raises(RuntimeError, match="INVOCATION_INVALID"):
        native.require_native_run_scope(recipe, invocation_number=2)
    with pytest.raises(RuntimeError, match="FQI_TARGET_REFRESH_ORIGIN_INVALID"):
        native.require_fqi_target_refresh_origin(continuation_origin)


@pytest.mark.parametrize("ceiling", [5521, 5778, 8162, True])
def test_fqi_scope_rejects_other_budgets(fqi_scope, ceiling):
    _, recipe, _, _ = fqi_scope
    with pytest.raises(RuntimeError, match="STEP_CEILING_INVALID"):
        native.require_native_run_scope(recipe, execution_budget={"stop_after_optimizer_steps": ceiling})


@pytest.mark.parametrize("fault", ["full_training", "old_scope", "extra_steps", "report_only_val"])
def test_fqi_scope_never_reuses_old_authority_or_enables_epoch(fqi_scope, fault):
    policy, recipe, _, save = fqi_scope
    if fault == "full_training":
        policy["training_enabled"] = True
    elif fault == "old_scope":
        policy["native_learning_calibration"]["schema_version"] = "gx1_native_training_continuation_scope_v1"
    elif fault == "extra_steps":
        policy["native_learning_calibration"]["additional_optimizer_step_ceilings"] = [256, 512]
    else:
        recipe["native_calibration"] = {"schema_version": "gx1_native_learning_calibration_run_v1",
                                        "arm": "reference", "report_only_val": True}
    save()
    with pytest.raises(RuntimeError, match="BOUNDED_TRAIN_ONLY_REQUIRED|FQI_TARGET_REFRESH_CALIBRATION_SCOPE_INVALID"):
        native.require_native_run_scope(recipe, invocation_number=1)


@pytest.mark.parametrize("artifact", ["contract", "pointer", "state"])
def test_fqi_origin_rejects_tampered_origin_files(fqi_origin, artifact):
    path = (Path(fqi_origin["pointer"]["path"]).parent / "candidate_training_state_slot_0.pt"
            if artifact == "state" else Path(fqi_origin[artifact]["path"]))
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RuntimeError):
        native.require_fqi_target_refresh_origin(fqi_origin)


def test_fqi_origin_rechecks_bound_cursor(fqi_origin, monkeypatch):
    changed = copy.deepcopy(fqi_origin)
    changed["pointer"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="ORIGIN_HASH_MISMATCH"):
        native.require_fqi_target_refresh_origin(changed, verify_files=False)
    path = Path(fqi_origin["pointer"]["path"])
    pointer = json.loads(path.read_text())
    pointer["global_optimizer_steps"] += 1
    fqi_origin["pointer"] = _write(path, pointer)
    monkeypatch.setattr(native, "FQI_TARGET_REFRESH_ORIGIN_POINTER_SHA256", fqi_origin["pointer"]["sha256"])
    with pytest.raises(RuntimeError, match="ORIGIN_CURSOR_INVALID"):
        native.require_fqi_target_refresh_origin(fqi_origin)


def test_fqi_history_retains_real_91_87_85_receipt_chain_and_declares_target_change(fqi_origin, tmp_path):
    directory = tmp_path / "fqi-session"
    directory.mkdir()
    recipe = _write(tmp_path / "fqi-recipe.json", {"candidate_resume_origin": fqi_origin})
    contract = _write(directory / "CANDIDATE_TRAINING_SESSION_CONTRACT.json", {
        "recipe_source_provenance": {"recipe_audit_path": recipe["path"], "recipe_audit_sha256": recipe["sha256"]},
    })
    inherited = ema.bind_candidate_weight_ema_history_v1(
        session_contract_path=Path(fqi_origin["contract"]["path"]),
        session_contract_sha256=fqi_origin["contract"]["sha256"],
    )
    receipt = {
        "schema_version": native.FQI_TARGET_REFRESH_RECEIPT_SCHEMA,
        "origin": fqi_origin, "origin_cursor": native.FQI_TARGET_REFRESH_ORIGIN_CURSOR,
        "origin_state_sha256": native.FQI_TARGET_REFRESH_ORIGIN_STATE_SHA256,
        "destination_session_contract_sha256": contract["sha256"],
        "inherited_weight_ema_history": inherited, "ema_internal_steps": 25429,
        "global_optimizer_steps": 5521, "state_preserved": False,
        "optimizer_procedure_changed": False, "identical_future_trajectory_claimed": False,
        "target_model_refreshed": True, "target_refresh_source": "origin_online_model_state",
        "original_target_model_state_sha256": "a" * 64,
        "refreshed_target_model_state_sha256": "b" * 64, "origin_online_model_state_sha256": "b" * 64,
        "training_progress_target_history": "retained_pre_refresh_history",
        "preserved_state_fields": ["model_state", "optimizer_state", "weight_ema_state",
                                   "lr_scheduler_state", "rng_state", "epoch_order", "training_progress"],
    }
    path = directory / native.FQI_TARGET_REFRESH_RECEIPT_NAME
    _write(path, receipt)
    kwargs = {"session_contract_path": Path(contract["path"]), "session_contract_sha256": contract["sha256"]}
    history = ema.bind_candidate_weight_ema_history_v1(**kwargs)
    assert history == {"optimizer_step_offset": 19908,
                       "transition_receipt": {"path": str(path), "sha256": native.file_sha256(path)}}
    assert ema._require_candidate_ema_history_offset(history, contract_path=Path(contract["path"])) == 19908
    for field, value in [("state_preserved", True), ("ema_internal_steps", 5521),
                         ("target_model_refreshed", False), ("refreshed_target_model_state_sha256", "c" * 64),
                         ("original_target_model_state_sha256", "unknown"),
                         ("preserved_state_fields", [*receipt["preserved_state_fields"], "target_model_state"])]:
        _write(path, {**receipt, field: value})
        with pytest.raises(RuntimeError, match="HISTORY_INVALID|FQI_TARGET_REFRESH_RECEIPT_INVALID"):
            ema.bind_candidate_weight_ema_history_v1(**kwargs)
    _write(path, receipt)
    # Changing even an unrelated field in the inherited continuation receipt is rejected.
    ancestor = Path(inherited["transition_receipt"]["path"])
    old = json.loads(ancestor.read_text())
    _write(ancestor, {**old, "unexpected_rewrite": True})
    with pytest.raises(RuntimeError, match="HISTORY_INVALID"):
        ema.bind_candidate_weight_ema_history_v1(**kwargs)


@pytest.mark.parametrize("count,expected", [(65295, "after_epoch_check"), (16, "cannot complete a TRAIN epoch")])
def test_fqi_materializer_checks_epoch_boundary_before_publication(fqi_scope, tmp_path, monkeypatch, count, expected):
    from gx1.scripts import materialize_local_random_access_campaign_v2 as materializer
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    _, recipe, _, _ = fqi_scope
    monkeypatch.setattr(materializer, "_source_commit", lambda _: "a" * 40)
    monkeypatch.setattr(native, "require_native_recipe_metadata", lambda *a, **kw: (recipe, count))
    def stop_before_prior(*_args, **_kwargs):
        raise RuntimeError("after_epoch_check")
    monkeypatch.setattr(campaign, "read_bound_json", stop_before_prior)
    with pytest.raises(RuntimeError, match=expected):
        materializer.materialize_native_candidate_campaign(
            repo=tmp_path, output=tmp_path / "uncreated", runtime=tmp_path / "uncreated-runtime",
            gpu_uuid="fixture", prepared_boot_path=tmp_path / "boot.json", prepared_boot_file_sha256="a" * 64,
            certificate_path=tmp_path / "certificate", prior_campaign_path=tmp_path / "prior.json",
            prior_campaign_file_sha256="a" * 64, selection_path=tmp_path / "selection.json",
            selection_file_sha256="a" * 64, recipe_path=tmp_path / "recipe.json",
            recipe_file_sha256="a" * 64, window_count=1,
        )
    assert not (tmp_path / "uncreated").exists() and not (tmp_path / "uncreated-runtime").exists()
