from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gx1.contracts import unified_exit_native_candidate_campaign_v1 as native
from gx1.contracts import unified_exit_random_access_val_checkpoint_v1 as ema
from tests.test_native_learning_calibration_scope import scope


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return {"path": str(path), "sha256": native.file_sha256(path)}


@pytest.fixture
def optimizer_origin(tmp_path, monkeypatch):
    """Real JSON/hash owners; only production's fixed measured hashes vary."""
    old = tmp_path / "old-session"
    old.mkdir()
    v2_contract = _write(tmp_path / "v2-contract.json", {"fixture": "v2"})
    v2_cursor = {"checkpoint_index": 315, "phase": "train", "epoch_index": 1,
                 "next_batch_offset": 320, "global_optimizer_steps": 19908, "complete": False}
    v2_pointer = _write(tmp_path / "v2-pointer.json", {
        **v2_cursor, "session_contract_sha256": v2_contract["sha256"], "state_sha256": "a" * 64,
    })
    economics_origin = {"schema_version": "gx1_candidate_economics_transition_origin_v1",
                        "contract": v2_contract, "pointer": v2_pointer,
                        "exit_value_initialization": "close_now_baseline_v1"}
    old_recipe = _write(tmp_path / "old-recipe.json", {"candidate_resume_origin": economics_origin})
    contract = _write(old / "CANDIDATE_TRAINING_SESSION_CONTRACT.json", {
        "recipe_source_provenance": {"recipe_audit_path": old_recipe["path"], "recipe_audit_sha256": old_recipe["sha256"]},
    })
    _write(old / "CANDIDATE_ECONOMICS_TRANSITION.json", {
        "schema_version": "gx1_candidate_economics_transition_receipt_v1",
        "origin": economics_origin, "origin_cursor": v2_cursor, "origin_state_sha256": "a" * 64,
        "destination_session_contract_sha256": contract["sha256"], "ema_internal_steps": 19908,
        "new_objective_global_optimizer_steps": 0,
    })
    state = old / "candidate_training_state_slot_0.pt"
    state.write_bytes(b"bounded fake checkpoint bytes; scope owner does not deserialize tensors")
    state_sha = native.file_sha256(state)
    pointer = _write(old / "CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json", {
        **native.OPTIMIZER_PROCEDURE_ORIGIN_CURSOR,
        "schema_version": "gx1_candidate_training_session_v1", "slot": 0,
        "session_contract_sha256": contract["sha256"], "state_sha256": state_sha,
    })
    monkeypatch.setattr(native, "OPTIMIZER_PROCEDURE_ORIGIN_CONTRACT_SHA256", contract["sha256"])
    monkeypatch.setattr(native, "OPTIMIZER_PROCEDURE_ORIGIN_POINTER_SHA256", pointer["sha256"])
    monkeypatch.setattr(native, "OPTIMIZER_PROCEDURE_ORIGIN_STATE_SHA256", state_sha)
    return {
        "schema_version": native.OPTIMIZER_PROCEDURE_TRANSITION_SCHEMA,
        "contract": contract, "pointer": pointer,
        "exit_value_initialization": "close_now_baseline_v1",
        "train_population_scope": "latest_year_2025_2026_v1",
        "gradient_clipping_policy": native.OPTIMIZER_PROCEDURE_TRANSITION_POLICY,
    }


@pytest.fixture
def optimizer_scope(scope, optimizer_origin, tmp_path):
    from tests.test_unified_exit_latest_year_index_v1 import _fixture, _root
    policy, recipe, write, save = scope
    population_dir = tmp_path / "population"
    population_dir.mkdir()
    recipe["files"]["random_access_root"] = write("latest-root.json", _root(_fixture(population_dir)))
    recipe["candidate_resume_origin"] = optimizer_origin
    for name in ("exit_value_initialization", "train_population_scope", "gradient_clipping_policy"):
        policy[name] = optimizer_origin[name]
    policy["native_learning_calibration"] = {
        "schema_version": "gx1_native_optimizer_procedure_calibration_scope_v1",
        "additional_optimizer_step_ceilings": [16, 32],
        "full_epoch_training_allowed": False, "test_data_used": False,
    }
    save()
    return policy, recipe, write, save


def test_optimizer_scope_preserves_absolute_cursor_and_only_adds_16_or_32(optimizer_scope):
    _, recipe, _, _ = optimizer_scope
    assert native.require_native_run_scope(recipe) == 5265  # Metadata-only admission.
    for invocation, limit in [(1, 5249), (2, 5265)]:
        assert native.require_native_run_scope(recipe, invocation_number=invocation) == limit
        assert native.require_native_run_scope(recipe, execution_budget={
            "stop_after_optimizer_steps": limit, "stop_after_completed_val_epochs": None,
            "max_invocation_seconds": 12000,
        }) == limit


@pytest.mark.parametrize("value", [None, True, 16, 32, 5233, 5250, 8162])
def test_optimizer_scope_rejects_legacy_or_unbounded_absolute_ceiling(optimizer_scope, value):
    _, recipe, _, _ = optimizer_scope
    with pytest.raises(RuntimeError, match="CALIBRATION_STEP_CEILING_INVALID"):
        native.require_native_run_scope(recipe, execution_budget={"stop_after_optimizer_steps": value})


@pytest.mark.parametrize("value", [0, True, 3])
def test_optimizer_scope_rejects_extra_windows(optimizer_scope, value):
    _, recipe, _, _ = optimizer_scope
    with pytest.raises(RuntimeError, match="CALIBRATION_INVOCATION_INVALID"):
        native.require_native_run_scope(recipe, invocation_number=value)


@pytest.mark.parametrize("change", ["full_training", "report_only_val", "wrong_policy", "extra_steps", "old_scope"])
def test_optimizer_scope_cannot_expand_or_reuse_old_authority(optimizer_scope, change):
    policy, recipe, _, save = optimizer_scope
    if change == "full_training":
        policy["training_enabled"] = True
    elif change == "report_only_val":
        recipe["native_calibration"] = {"schema_version": "gx1_native_learning_calibration_run_v1",
                                        "arm": "reference", "report_only_val": True}
    elif change == "wrong_policy":
        policy["gradient_clipping_policy"] = "all_parameters_joint_v1"
    elif change == "extra_steps":
        policy["native_learning_calibration"]["additional_optimizer_step_ceilings"] = [16, 64]
    else:
        policy["native_learning_calibration"] = {"schema_version": "gx1_native_learning_calibration_scope_v1",
            "optimizer_step_ceilings": [16, 32], "full_epoch_training_allowed": False, "test_data_used": False}
    save()
    with pytest.raises(RuntimeError, match="OPTIMIZER_PROCEDURE"):
        native.require_native_run_scope(recipe, invocation_number=1)


@pytest.mark.parametrize("field,value", [("max_invocation_seconds", 60), ("stop_after_completed_val_epochs", 1), ("resume_probe_val_rows", 32)])
def test_optimizer_scope_retains_native_window_and_forbids_dummy_val(optimizer_scope, field, value):
    _, recipe, _, _ = optimizer_scope
    budget = {"stop_after_optimizer_steps": 5249, "stop_after_completed_val_epochs": None,
              "max_invocation_seconds": 12000, field: value}
    with pytest.raises(RuntimeError, match="NEXT_RUN_BUDGET_INVALID"):
        native.require_native_run_scope(recipe, execution_budget=budget)


@pytest.mark.parametrize("field,value", [("gradient_clipping_policy", True), ("schema_version", "unknown"), ("reset_encoder", True)])
def test_optimizer_origin_rejects_unknown_or_untyped_fields(optimizer_origin, field, value):
    origin = {**optimizer_origin, field: value}
    with pytest.raises(RuntimeError, match="ORIGIN_INVALID"):
        native.require_optimizer_procedure_origin(origin)


@pytest.mark.parametrize("artifact", ["contract", "pointer", "state"])
def test_optimizer_origin_rejects_changed_physical_files(optimizer_origin, artifact):
    path = (Path(optimizer_origin["pointer"]["path"]).parent / "candidate_training_state_slot_0.pt"
            if artifact == "state" else Path(optimizer_origin[artifact]["path"]))
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RuntimeError):
        native.require_optimizer_procedure_origin(optimizer_origin)


def test_optimizer_origin_rejects_changed_digest_even_without_io(optimizer_origin):
    origin = copy.deepcopy(optimizer_origin)
    origin["pointer"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="ORIGIN_HASH_MISMATCH"):
        native.require_optimizer_procedure_origin(origin, verify_files=False)


@pytest.mark.parametrize("field,value", [("global_optimizer_steps", 0), ("epoch_index", True), ("slot", 1)])
def test_optimizer_origin_rechecks_cursor_even_with_bound_fixture_hash(optimizer_origin, monkeypatch, field, value):
    path = Path(optimizer_origin["pointer"]["path"])
    pointer = json.loads(path.read_text())
    pointer[field] = value
    optimizer_origin["pointer"] = _write(path, pointer)
    monkeypatch.setattr(native, "OPTIMIZER_PROCEDURE_ORIGIN_POINTER_SHA256", optimizer_origin["pointer"]["sha256"])
    with pytest.raises(RuntimeError, match="ORIGIN_CURSOR_INVALID"):
        native.require_optimizer_procedure_origin(optimizer_origin)


@pytest.fixture
def optimizer_history(tmp_path, optimizer_origin):
    directory = tmp_path / "new-session"
    directory.mkdir()
    recipe = _write(tmp_path / "new-recipe.json", {"candidate_resume_origin": optimizer_origin})
    contract = _write(directory / "CANDIDATE_TRAINING_SESSION_CONTRACT.json", {
        "recipe_source_provenance": {"recipe_audit_path": recipe["path"], "recipe_audit_sha256": recipe["sha256"]},
    })
    inherited = ema.bind_candidate_weight_ema_history_v1(
        session_contract_path=Path(optimizer_origin["contract"]["path"]),
        session_contract_sha256=optimizer_origin["contract"]["sha256"],
    )
    receipt = {
        "schema_version": native.OPTIMIZER_PROCEDURE_TRANSITION_RECEIPT_SCHEMA,
        "origin": optimizer_origin, "origin_cursor": native.OPTIMIZER_PROCEDURE_ORIGIN_CURSOR,
        "origin_state_sha256": native.OPTIMIZER_PROCEDURE_ORIGIN_STATE_SHA256,
        "destination_session_contract_sha256": contract["sha256"],
        "inherited_weight_ema_history": inherited, "ema_internal_steps": 25141,
        "global_optimizer_steps": 5233, "state_preserved": True,
        "optimizer_procedure_changed": True, "identical_future_trajectory_claimed": False,
        "preserved_state_fields": ["model_state", "target_model_state", "optimizer_state", "weight_ema_state",
                                   "lr_scheduler_state", "rng_state", "epoch_order", "training_progress"],
    }
    receipt_path = directory / native.OPTIMIZER_PROCEDURE_TRANSITION_RECEIPT_NAME
    _write(receipt_path, receipt)
    return contract, receipt_path, receipt


def test_optimizer_history_binds_exact_retained_offset_to_new_receipt(optimizer_history):
    contract, receipt_path, _ = optimizer_history
    history = ema.bind_candidate_weight_ema_history_v1(
        session_contract_path=Path(contract["path"]), session_contract_sha256=contract["sha256"],
    )
    assert history == {"optimizer_step_offset": 19908,
                       "transition_receipt": {"path": str(receipt_path), "sha256": native.file_sha256(receipt_path)}}
    assert ema._require_candidate_ema_history_offset(history, contract_path=Path(contract["path"])) == 19908
    history["transition_receipt"]["path"] = str(receipt_path.parent.parent / receipt_path.name)
    with pytest.raises(RuntimeError, match="EMA_HISTORY_INVALID"):
        ema._require_candidate_ema_history_offset(history, contract_path=Path(contract["path"]))


@pytest.mark.parametrize("field,value", [
    ("ema_internal_steps", 5233), ("global_optimizer_steps", 0),
    ("inherited_weight_ema_history", None), ("state_preserved", False),
    ("optimizer_procedure_changed", False), ("identical_future_trajectory_claimed", True),
    ("preserved_state_fields", ["model_state"]), ("origin_state_sha256", "a" * 64),
    ("destination_session_contract_sha256", "b" * 64),
])
def test_optimizer_history_rejects_resets_and_unbound_preservation(optimizer_history, field, value):
    contract, receipt_path, receipt = optimizer_history
    _write(receipt_path, {**receipt, field: value})
    with pytest.raises(RuntimeError, match="EMA_HISTORY_INVALID"):
        ema.bind_candidate_weight_ema_history_v1(
            session_contract_path=Path(contract["path"]), session_contract_sha256=contract["sha256"],
        )


def test_optimizer_history_rejects_changed_inherited_receipt(optimizer_history):
    contract, _, receipt = optimizer_history
    old_path = Path(receipt["inherited_weight_ema_history"]["transition_receipt"]["path"])
    old = json.loads(old_path.read_text())
    _write(old_path, {**old, "unexpected_rewrite": True})
    with pytest.raises(RuntimeError, match="EMA_HISTORY_INVALID"):
        ema.bind_candidate_weight_ema_history_v1(
            session_contract_path=Path(contract["path"]), session_contract_sha256=contract["sha256"],
        )


@pytest.mark.parametrize("count,expected", [(65295, "after_epoch_check"), (16, "cannot complete a TRAIN epoch")])
def test_materializer_checks_same_epoch_end_without_running_or_mutating_campaign(optimizer_scope, tmp_path, monkeypatch, count, expected):
    from gx1.scripts import materialize_local_random_access_campaign_v2 as materializer
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    _, recipe, _, _ = optimizer_scope
    monkeypatch.setattr(materializer, "_source_commit", lambda _: "a" * 40)
    monkeypatch.setattr(native, "require_native_recipe_metadata", lambda *a, **kw: (recipe, count))
    # Real scope owner runs first; stop before prior campaign loading or writes.
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
            recipe_file_sha256="a" * 64, window_count=2,
        )
    assert not (tmp_path / "uncreated").exists()
    assert not (tmp_path / "uncreated-runtime").exists()
