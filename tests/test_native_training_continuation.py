from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gx1.contracts import unified_exit_native_candidate_campaign_v1 as native
from gx1.contracts import unified_exit_random_access_val_checkpoint_v1 as ema
from tests.test_native_optimizer_procedure_scope import (
    _write, optimizer_origin, optimizer_scope, optimizer_history, scope,
)


@pytest.fixture
def continuation_origin(optimizer_history, monkeypatch):
    """Retain real economics -> clipping receipt lineage before stopped87."""
    contract, receipt_path, _ = optimizer_history
    state = receipt_path.parent / "candidate_training_state_slot_0.pt"
    state.write_bytes(b"fixture stopped87; origin validator does not deserialize tensors")
    state_sha = native.file_sha256(state)
    pointer = _write(receipt_path.parent / "CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json", {
        **native.TRAINING_CONTINUATION_ORIGIN_CURSOR,
        "schema_version": "gx1_candidate_training_session_v1", "slot": 0,
        "session_contract_sha256": contract["sha256"], "state_sha256": state_sha,
    })
    monkeypatch.setattr(native, "TRAINING_CONTINUATION_ORIGIN_CONTRACT_SHA256", contract["sha256"])
    monkeypatch.setattr(native, "TRAINING_CONTINUATION_ORIGIN_POINTER_SHA256", pointer["sha256"])
    monkeypatch.setattr(native, "TRAINING_CONTINUATION_ORIGIN_STATE_SHA256", state_sha)
    return {"schema_version": native.TRAINING_CONTINUATION_SCHEMA,
            "contract": contract, "pointer": pointer,
            "exit_value_initialization": "close_now_baseline_v1",
            "train_population_scope": "latest_year_2025_2026_v1",
            "gradient_clipping_policy": native.OPTIMIZER_PROCEDURE_TRANSITION_POLICY}


@pytest.fixture
def continuation_scope(optimizer_scope, continuation_origin):
    policy, recipe, write, save = optimizer_scope
    recipe["candidate_resume_origin"] = continuation_origin
    policy["native_learning_calibration"] = {
        "schema_version": "gx1_native_training_continuation_scope_v1",
        "additional_optimizer_step_ceilings": [256],
        "full_epoch_training_allowed": False, "test_data_used": False,
    }
    save()
    return policy, recipe, write, save


def test_continuation_admits_only_bound_87_and_one_absolute_5521_window(continuation_scope):
    _, recipe, _, _ = continuation_scope
    assert native.require_native_run_scope(recipe) == 5521
    assert native.require_native_run_scope(recipe, invocation_number=1) == 5521
    assert native.require_native_run_scope(recipe, execution_budget={
        "stop_after_optimizer_steps": 5521, "stop_after_completed_val_epochs": None,
        "max_invocation_seconds": 12000,
    }) == 5521
    with pytest.raises(RuntimeError, match="INVOCATION_INVALID"):
        native.require_native_run_scope(recipe, invocation_number=2)


@pytest.mark.parametrize("ceiling", [5265, 5522, 8162, True])
def test_continuation_rejects_other_step_budgets(continuation_scope, ceiling):
    _, recipe, _, _ = continuation_scope
    with pytest.raises(RuntimeError, match="STEP_CEILING_INVALID"):
        native.require_native_run_scope(recipe, execution_budget={"stop_after_optimizer_steps": ceiling})


@pytest.mark.parametrize("fault", ["full_training", "report_only_val", "extra_steps", "old_scope"])
def test_continuation_cannot_expand_or_reuse_finished_scope(continuation_scope, fault):
    policy, recipe, _, save = continuation_scope
    if fault == "full_training":
        policy["training_enabled"] = True
    elif fault == "report_only_val":
        recipe["native_calibration"] = {"schema_version": "gx1_native_learning_calibration_run_v1",
                                        "arm": "reference", "report_only_val": True}
    elif fault == "extra_steps":
        policy["native_learning_calibration"]["additional_optimizer_step_ceilings"] = [256, 512]
    else:
        policy["native_learning_calibration"].update(
            schema_version="gx1_native_optimizer_procedure_calibration_scope_v1",
            additional_optimizer_step_ceilings=[16, 32])
    save()
    with pytest.raises(RuntimeError, match="BOUNDED_TRAIN_ONLY_REQUIRED|CONTINUATION_CALIBRATION_SCOPE_INVALID"):
        native.require_native_run_scope(recipe, invocation_number=1)


def test_continuation_rejects_other_origin_and_modified_bound_cursor(continuation_origin, optimizer_origin, monkeypatch):
    with pytest.raises(RuntimeError, match="CONTINUATION_ORIGIN_INVALID"):
        native.require_training_continuation_origin(optimizer_origin)
    changed = copy.deepcopy(continuation_origin)
    changed["pointer"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="ORIGIN_HASH_MISMATCH"):
        native.require_training_continuation_origin(changed, verify_files=False)
    path = Path(continuation_origin["pointer"]["path"])
    pointer = json.loads(path.read_text())
    pointer["next_batch_offset"] += 1
    continuation_origin["pointer"] = _write(path, pointer)
    monkeypatch.setattr(native, "TRAINING_CONTINUATION_ORIGIN_POINTER_SHA256", continuation_origin["pointer"]["sha256"])
    with pytest.raises(RuntimeError, match="ORIGIN_CURSOR_INVALID"):
        native.require_training_continuation_origin(continuation_origin)


@pytest.mark.parametrize("artifact", ["contract", "pointer", "state"])
def test_continuation_rejects_changed_physical_origin(continuation_origin, artifact):
    path = (Path(continuation_origin["pointer"]["path"]).parent / "candidate_training_state_slot_0.pt"
            if artifact == "state" else Path(continuation_origin[artifact]["path"]))
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RuntimeError):
        native.require_training_continuation_origin(continuation_origin)


def test_continuation_history_preserves_both_prior_receipts_and_exact_ema_counter(continuation_origin, tmp_path):
    directory = tmp_path / "continuation-session"
    directory.mkdir()
    recipe = _write(tmp_path / "continuation-recipe.json", {"candidate_resume_origin": continuation_origin})
    contract = _write(directory / "CANDIDATE_TRAINING_SESSION_CONTRACT.json", {
        "recipe_source_provenance": {"recipe_audit_path": recipe["path"], "recipe_audit_sha256": recipe["sha256"]},
    })
    inherited = ema.bind_candidate_weight_ema_history_v1(
        session_contract_path=Path(continuation_origin["contract"]["path"]),
        session_contract_sha256=continuation_origin["contract"]["sha256"],
    )
    receipt = {
        "schema_version": native.TRAINING_CONTINUATION_RECEIPT_SCHEMA,
        "origin": continuation_origin, "origin_cursor": native.TRAINING_CONTINUATION_ORIGIN_CURSOR,
        "origin_state_sha256": native.TRAINING_CONTINUATION_ORIGIN_STATE_SHA256,
        "destination_session_contract_sha256": contract["sha256"],
        "inherited_weight_ema_history": inherited, "ema_internal_steps": 25173,
        "global_optimizer_steps": 5265, "state_preserved": True,
        "optimizer_procedure_changed": False, "identical_future_trajectory_claimed": False,
        "preserved_state_fields": ["model_state", "target_model_state", "optimizer_state", "weight_ema_state",
                                   "lr_scheduler_state", "rng_state", "epoch_order", "training_progress"],
    }
    path = directory / native.TRAINING_CONTINUATION_RECEIPT_NAME
    _write(path, receipt)
    kwargs = {"session_contract_path": Path(contract["path"]), "session_contract_sha256": contract["sha256"]}
    history = ema.bind_candidate_weight_ema_history_v1(**kwargs)
    assert history == {"optimizer_step_offset": 19908,
                       "transition_receipt": {"path": str(path), "sha256": native.file_sha256(path)}}
    assert ema._require_candidate_ema_history_offset(history, contract_path=Path(contract["path"])) == 19908
    for field, value in [("ema_internal_steps", 5265), ("optimizer_procedure_changed", True),
                         ("preserved_state_fields", ["model_state"])]:
        _write(path, {**receipt, field: value})
        with pytest.raises(RuntimeError, match="EMA_HISTORY_INVALID"):
            ema.bind_candidate_weight_ema_history_v1(**kwargs)
    _write(path, receipt)
    inherited_receipt = Path(inherited["transition_receipt"]["path"])
    old = json.loads(inherited_receipt.read_text())
    _write(inherited_receipt, {**old, "unexpected_rewrite": True})
    with pytest.raises(RuntimeError, match="EMA_HISTORY_INVALID"):
        ema.bind_candidate_weight_ema_history_v1(**kwargs)


@pytest.mark.parametrize("count,expected", [(65295, "after_epoch_check"), (16, "cannot complete a TRAIN epoch")])
def test_continuation_materializer_checks_epoch_boundary_before_publication(continuation_scope, tmp_path, monkeypatch, count, expected):
    from gx1.scripts import materialize_local_random_access_campaign_v2 as materializer
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    _, recipe, _, _ = continuation_scope
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
    assert not (tmp_path / "uncreated").exists()
    assert not (tmp_path / "uncreated-runtime").exists()
