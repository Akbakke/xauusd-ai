from __future__ import annotations

import copy
import json

import pytest

from gx1.contracts import unified_exit_native_candidate_campaign_v1 as native
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from tests.test_native_entry_learnability import (
    real_train_scope, learnability_scope, learnability_origin, fqi_scope, fqi_origin,
    continuation_scope, continuation_origin, optimizer_scope, optimizer_origin,
    optimizer_history, scope,
)
from tests.test_candidate_economics_transition import _optimizer_procedure_fixture, _identical


@pytest.fixture
def trace_scope(real_train_scope):
    policy, recipe, write, save = real_train_scope
    recipe["candidate_resume_origin"]["exit_backup_steps"] = 5
    recipe["exit_backup_steps"] = policy["exit_backup_steps"] = 5
    recipe["native_calibration"] = {"schema_version": "gx1_native_learning_calibration_run_v1",
                                    "arm": "split", "report_only_val": False}
    policy["native_learning_calibration"] = {
        "schema_version": "gx1_native_frozen_policy_trace_scope_v1",
        "additional_optimizer_step_ceilings": [16, 32],
        "reference_additional_optimizer_step_ceiling": 32,
        "full_epoch_training_allowed": False, "test_data_used": False,
    }
    save()
    return policy, recipe, write, save


@pytest.mark.parametrize("arm,ceilings", [("reference", [5809]), ("split", [5793, 5809])])
def test_trace_scope_is_exact_original95_plus32_or16plus16(trace_scope, arm, ceilings):
    _, recipe, _, _ = trace_scope
    recipe["native_calibration"]["arm"] = arm
    assert native.require_training_continuation_origin(recipe["candidate_resume_origin"])
    assert native.native_completed_val_ceiling(recipe) is None
    for number, ceiling in enumerate(ceilings, 1):
        assert native.require_native_run_scope(recipe, invocation_number=number) == ceiling
        assert native.require_native_run_scope(recipe, execution_budget={
            "stop_after_optimizer_steps": ceiling, "stop_after_completed_val_epochs": None,
            "max_invocation_seconds": 12000,
        }) == ceiling
    with pytest.raises(RuntimeError, match="INVOCATION_INVALID"):
        native.require_native_run_scope(recipe, invocation_number=len(ceilings)+1)
    for wrong in [5777, 5808, 5810, 6033, 8162, True]:
        with pytest.raises(RuntimeError):
            native.require_native_run_scope(recipe, execution_budget={"stop_after_optimizer_steps":wrong})


@pytest.mark.parametrize("fault", ["recipe_missing", "recipe_steps", "policy_missing", "policy_steps",
    "origin_missing", "origin_steps", "calibration_missing", "report_val", "full_training", "more_steps", "old_scope"])
def test_trace_scope_requires_every_explicit_binding_and_cannot_expand(trace_scope, fault):
    policy, recipe, _, save = trace_scope
    if fault == "recipe_missing": recipe.pop("exit_backup_steps")
    elif fault == "recipe_steps": recipe["exit_backup_steps"] = 2
    elif fault == "policy_missing": policy.pop("exit_backup_steps")
    elif fault == "policy_steps": policy["exit_backup_steps"] = 1
    elif fault == "origin_missing": recipe["candidate_resume_origin"].pop("exit_backup_steps")
    elif fault == "origin_steps": recipe["candidate_resume_origin"]["exit_backup_steps"] = True
    elif fault == "calibration_missing": recipe.pop("native_calibration")
    elif fault == "report_val": recipe["native_calibration"]["report_only_val"] = True
    elif fault == "full_training": policy["training_enabled"] = True
    elif fault == "more_steps": policy["native_learning_calibration"]["additional_optimizer_step_ceilings"] = [16,32,256]
    else: policy["native_learning_calibration"] = {"schema_version":"gx1_native_training_continuation_scope_v2",
        "stop_after_completed_val_epochs":2,"full_epoch_training_allowed":True,"test_data_used":False}
    save()
    with pytest.raises(RuntimeError): native.require_native_run_scope(recipe, invocation_number=1)


def test_trace_rejects_old87_even_with_matching_backup_flag(continuation_origin):
    origin = {**continuation_origin, "exit_backup_steps":5}
    with pytest.raises(RuntimeError, match="TRACE_BACKUP_ORIGIN_INVALID"):
        native.require_training_continuation_origin(origin)


def test_trace_transition_preserves_every_state_and_serialized_resume(tmp_path, monkeypatch):
    (old,new), origin = _optimizer_procedure_fixture(tmp_path,monkeypatch,trace_backup=True)
    before = old.load_checkpoint()
    hashes={p.name:trainer._sha256_file(p) for p in old.directory.iterdir() if p.is_file()}
    state=trainer._load_candidate_optimizer_procedure_successor_state(session=new,origin=origin)
    assert state["session_contract_sha256"] == new.contract_sha256
    assert new._contract["training"]["exit_backup_steps"] == 5
    for key in before.keys()-{"session_contract_sha256"}: _identical(before[key],state[key])
    receipt_path=new.directory/native.TRAINING_CONTINUATION_RECEIPT_NAME
    receipt=json.loads(receipt_path.read_text())
    assert receipt["exit_backup_policy_change"] == {"previous_backup_steps":1,"backup_steps":5,
        "teacher_preserved":True,"sampler_preserved":True,"holding_time_cap_introduced":False}
    assert receipt["state_preserved"] is True and receipt["optimizer_procedure_changed"] is False
    assert receipt["global_optimizer_steps"] == 5777
    _identical(state,trainer._load_candidate_optimizer_procedure_successor_state(session=new,origin=origin))
    new.save_checkpoint(state)
    _identical(state,new.load_checkpoint())
    assert hashes == {p.name:trainer._sha256_file(p) for p in old.directory.iterdir() if p.is_file()}
    with pytest.raises(RuntimeError,match="SESSION_BINDING_INVALID"):
        trainer._load_candidate_optimizer_procedure_successor_state(session=new,origin=origin)


@pytest.mark.parametrize("fault,error", [("recipe_backup","TRACE_BACKUP_POLICY_INVALID"),
    ("backup_steps","TRACE_BACKUP_POLICY_INVALID"),("calibration_val","TRACE_BACKUP_POLICY_INVALID"),
    ("model_source","UNAPPROVED_SOURCE_CHANGED"),("data","MODEL_DATA_OR_TRAINING_CHANGED"),
    ("batch","MODEL_DATA_OR_TRAINING_CHANGED"),("clip_cap","CLIPPING_POLICY_INVALID"),
    ("cursor","STOPPED_STATE_INVALID"),("ema_steps","EMA_HISTORY_INVALID")])
def test_trace_transition_rejects_unrelated_or_incomplete_changes(tmp_path,monkeypatch,fault,error):
    (_,new),origin=_optimizer_procedure_fixture(tmp_path,monkeypatch,fault,trace_backup=True)
    with pytest.raises(RuntimeError,match=error):
        trainer._load_candidate_optimizer_procedure_successor_state(session=new,origin=origin)
    assert not new._active_path.exists()
    assert not (new.directory/native.TRAINING_CONTINUATION_RECEIPT_NAME).exists()


def test_trace_history_requires_backup_receipt_and_preserved_teacher(trace_scope, tmp_path):
    from pathlib import Path
    from gx1.contracts import unified_exit_random_access_val_checkpoint_v1 as ema
    from tests.test_native_optimizer_procedure_scope import _write
    _, recipe, _, _ = trace_scope
    origin = recipe['candidate_resume_origin']
    directory = tmp_path/'trace-history'; directory.mkdir()
    recipe_binding = _write(tmp_path/'trace-history-recipe.json', recipe)
    contract_value = {'recipe_source_provenance': {'recipe_audit_path': recipe_binding['path'],
        'recipe_audit_sha256':recipe_binding['sha256']}, 'training':{'exit_backup_steps':5}}
    contract = _write(directory/'CANDIDATE_TRAINING_SESSION_CONTRACT.json', contract_value)
    inherited = ema.bind_candidate_weight_ema_history_v1(session_contract_path=Path(origin['contract']['path']),
        session_contract_sha256=origin['contract']['sha256'])
    receipt = {'schema_version':native.TRAINING_CONTINUATION_RECEIPT_SCHEMA,
        'origin':origin,'origin_cursor':native.ENTRY_LEARNABILITY_ORIGIN_CURSOR,
        'origin_state_sha256':native.ENTRY_LEARNABILITY_ORIGIN_STATE_SHA256,
        'destination_session_contract_sha256':contract['sha256'],'inherited_weight_ema_history':inherited,
        'ema_internal_steps':25685,'global_optimizer_steps':5777,'state_preserved':True,
        'optimizer_procedure_changed':False,'identical_future_trajectory_claimed':False,
        'preserved_state_fields':['model_state','target_model_state','optimizer_state','weight_ema_state',
            'lr_scheduler_state','rng_state','epoch_order','training_progress'],
        'exit_backup_policy_change':{'previous_backup_steps':1,'backup_steps':5,
            'teacher_preserved':True,'sampler_preserved':True,'holding_time_cap_introduced':False}}
    path=directory/native.TRAINING_CONTINUATION_RECEIPT_NAME
    _write(path,receipt)
    kwargs={'session_contract_path':Path(contract['path']),'session_contract_sha256':contract['sha256']}
    assert ema.bind_candidate_weight_ema_history_v1(**kwargs)['optimizer_step_offset']==19908
    for field,value in [('previous_backup_steps',2),('backup_steps',1),('teacher_preserved',False),
                        ('sampler_preserved',False),('holding_time_cap_introduced',True)]:
        changed=copy.deepcopy(receipt);changed['exit_backup_policy_change'][field]=value;_write(path,changed)
        with pytest.raises(RuntimeError,match='TRACE_BACKUP_RECEIPT_INVALID'):
            ema.bind_candidate_weight_ema_history_v1(**kwargs)
    _write(path,receipt)
    changed=copy.deepcopy(receipt);changed['preserved_state_fields'].remove('target_model_state');_write(path,changed)
    with pytest.raises(RuntimeError,match='EMA_HISTORY_INVALID'):
        ema.bind_candidate_weight_ema_history_v1(**kwargs)
