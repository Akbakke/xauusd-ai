from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from gx1.contracts import unified_exit_native_candidate_campaign_v1 as native
from gx1.contracts.unified_exit_reference_policy_v1 import reference_policy_contract, build_reference_policy_hold_targets
from gx1.contracts.unified_exit_random_access_training_v1 import run_random_access_training_step
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from tests.test_native_frozen_policy_trace import (
    real_train_scope, learnability_scope, learnability_origin, fqi_scope, fqi_origin,
    continuation_scope, continuation_origin, optimizer_scope, optimizer_origin,
    optimizer_history, scope,
)
from tests.test_candidate_economics_transition import _optimizer_procedure_fixture, _identical
from tests.test_unified_exit_reference_data_flow_v1 import _batch
from tests.test_liquidation_relative_learning_v1 import relative_fixture, _TwoRowHead
from tests.test_unified_exit_random_access_training_v1 import _collate


@pytest.fixture
def reference_scope(real_train_scope):
    policy, recipe, write, save = real_train_scope
    reference = reference_policy_contract()
    recipe['candidate_resume_origin']['exit_reference_policy'] = reference
    recipe['exit_reference_policy'] = policy['exit_reference_policy'] = reference
    recipe['source_bindings_sha256'] = 'b' * 64
    recipe['native_calibration'] = {'schema_version':'gx1_native_learning_calibration_run_v1',
                                    'arm':'reference','report_only_val':False}
    policy['native_learning_calibration'] = {'schema_version':'gx1_native_reference_policy_scope_v1',
        'additional_optimizer_step_ceilings':[32],'reference_additional_optimizer_step_ceiling':32,
        'full_epoch_training_allowed':False,'test_data_used':False}
    plan = {
        'schema_version':'gx1_reference_critic_learning_plan_v1','decision':'FROZEN_COMPARISON_READY',
        'reference_policy_sha256':reference['policy_sha256'],
        'origin_state_sha256':native.ENTRY_LEARNABILITY_ORIGIN_STATE_SHA256,
        'teacher_model_state_sha256':native.ENTRY_LEARNABILITY_TARGET_MODEL_SHA256,
        'source_bindings_sha256':recipe['source_bindings_sha256'], 'split':'train','test_data_used':False,
        'teacher_refresh_allowed':False,
        'entry_bridge':'unchanged_initial_teacher_greedy_bridge_no_reference_refresh',
        'comparison':'same_frozen_targets_before_after_and_constant_baselines_by_side_and_month',
        'trained_entry_count':512,'separate_train_entry_count':128,
        'targets':{role:write(role+'.json',{'scope_fixture':role}) for role in ('trained','separate_train')},
    }
    def save_reference():
        policy['reference_learning_plan'] = write('reference-plan.json',plan)
        save()
    save_reference()
    return policy, recipe, plan, save_reference


def test_reference_scope_is_one_bound32_step_window_without_val(reference_scope):
    _, recipe, _, _ = reference_scope
    assert native.native_completed_val_ceiling(recipe) is None
    assert native.require_native_run_scope(recipe, invocation_number=1) == 5809
    assert native.require_native_run_scope(recipe, execution_budget={
        'stop_after_optimizer_steps':5809,'stop_after_completed_val_epochs':None,
        'max_invocation_seconds':12000}) == 5809
    with pytest.raises(RuntimeError,match='INVOCATION_INVALID'):
        native.require_native_run_scope(recipe,invocation_number=2)


@pytest.mark.parametrize('fault', ['policy_missing','recipe_missing','origin_missing','mix_greedy','scope',
    'full_training','val','split','teacher','source','test','refresh','target_bytes','same_cohort','plan_missing','plan_decision','entry_bridge'])
def test_reference_cannot_run_without_matching_frozen_evidence(reference_scope, fault):
    policy, recipe, plan, save = reference_scope
    if fault == 'policy_missing': policy.pop('exit_reference_policy')
    elif fault == 'recipe_missing': recipe.pop('exit_reference_policy')
    elif fault == 'origin_missing': recipe['candidate_resume_origin'].pop('exit_reference_policy')
    elif fault == 'mix_greedy': recipe['candidate_resume_origin']['exit_backup_steps']=5
    elif fault == 'scope': policy['native_learning_calibration']['additional_optimizer_step_ceilings']=[32,256]
    elif fault == 'full_training': policy['training_enabled']=True
    elif fault == 'val': recipe['native_calibration']['report_only_val']=True
    elif fault == 'split': recipe['native_calibration']['arm']='split'
    elif fault == 'teacher': plan['teacher_model_state_sha256']='0'*64
    elif fault == 'source': plan['source_bindings_sha256']='0'*64
    elif fault == 'test': plan['test_data_used']=True
    elif fault == 'refresh': plan['teacher_refresh_allowed']=True
    elif fault == 'target_bytes': Path(plan['targets']['trained']['path']).write_text('corrupted')
    elif fault == 'same_cohort': plan['targets']['separate_train']=plan['targets']['trained']
    elif fault == 'plan_decision': plan['decision']='PENDING'
    elif fault == 'entry_bridge': plan['entry_bridge']='silently_use_Q_mu_as_Q_star'
    save()
    if fault == 'plan_missing': Path(policy['reference_learning_plan']['path']).unlink()
    with pytest.raises((RuntimeError,FileNotFoundError)):
        native.require_native_run_scope(recipe,invocation_number=1)


def test_reference_transition_preserves_checkpoint_and_adds_only_target_owner(tmp_path,monkeypatch):
    (old,new),origin=_optimizer_procedure_fixture(tmp_path,monkeypatch,reference_policy=True)
    before=old.load_checkpoint()
    hashes={p.name:trainer._sha256_file(p) for p in old.directory.iterdir() if p.is_file()}
    state=trainer._load_candidate_optimizer_procedure_successor_state(session=new,origin=origin)
    for key in before.keys()-{'session_contract_sha256'}: _identical(before[key],state[key])
    assert state['session_contract_sha256']==new.contract_sha256
    receipt=json.loads((new.directory/native.TRAINING_CONTINUATION_RECEIPT_NAME).read_text())
    change=receipt['exit_reference_policy_change']
    assert change['policy']==reference_policy_contract()
    assert change['teacher_preserved'] and change['sampler_preserved']
    assert change['teacher_role']=='initial_Q_mu_guess_from_preserved_legacy_weights'
    assert receipt['state_preserved'] and not receipt['optimizer_procedure_changed']
    assert 'gx1/contracts/unified_exit_reference_policy_v1.py' in receipt['changed_source_paths']
    _identical(state,trainer._load_candidate_optimizer_procedure_successor_state(session=new,origin=origin))
    new.save_checkpoint(state);_identical(state,new.load_checkpoint())
    assert hashes=={p.name:trainer._sha256_file(p) for p in old.directory.iterdir() if p.is_file()}


@pytest.mark.parametrize('fault,error', [('model_source','UNAPPROVED_SOURCE_CHANGED'),('extra_owner','SOURCE_CLOSURE_CHANGED'),
    ('contract_reference','REFERENCE_POLICY_INVALID'),('recipe_reference','REFERENCE_POLICY_INVALID'),
    ('data','MODEL_DATA_OR_TRAINING_CHANGED'),('batch','MODEL_DATA_OR_TRAINING_CHANGED'),
    ('cursor','STOPPED_STATE_INVALID'),('ema_steps','EMA_HISTORY_INVALID')])
def test_reference_transition_rejects_unrelated_changes(tmp_path,monkeypatch,fault,error):
    (_,new),origin=_optimizer_procedure_fixture(tmp_path,monkeypatch,fault,reference_policy=True)
    with pytest.raises(RuntimeError,match=error):
        trainer._load_candidate_optimizer_procedure_successor_state(session=new,origin=origin)
    assert not new._active_path.exists()
    assert not (new.directory/native.TRAINING_CONTINUATION_RECEIPT_NAME).exists()


@pytest.mark.parametrize('counts,terminals',[((600,600),(False,False)),((3,3),(False,False)),((3,4),(True,False))])
def test_one_native_exit_update_uses_Q_mu_and_keeps_initial_entry_bridge(relative_fixture, counts, terminals):
    reference,_=_batch(counts=counts,terminals=terminals)
    legacy=_collate()
    outputs=[]
    for batch in (legacy,reference):
        model=_TwoRowHead()
        target=copy.deepcopy(model).requires_grad_(False).eval()
        entry=torch.tensor([[2.],[3.],[4.]],requires_grad=True)
        result=run_random_access_training_step(model=model,target_model=target,
            entry_decision_representations=entry,target_entry_decision_representations=entry.detach(),
            batch=batch,grad_accum_steps=1,
            reference_policy=reference_policy_contract() if batch is reference else None)
        assert model.calls==target.calls==1
        assert result['backward_calls']==1
        assert all(p.grad is None for p in target.parameters())
        assert result['entry_gradients'][1].abs().sum()>0
        outputs.append(result)
    torch.testing.assert_close(outputs[0]['entry_targets'],outputs[1]['entry_targets'],rtol=0,atol=0)
    assert outputs[0]['entry_bridge_binding']==outputs[1]['entry_bridge_binding']
    trace=reference['reference_policy_trace']
    q=torch.tensor([[[.21,0.],[.21,0.]]])
    expected=build_reference_policy_hold_targets(policy=trace['policy'],boundary_action_q_bps=q,
        **{k:v for k,v in trace.items() if k not in ('policy','state_view_sha256')})
    torch.testing.assert_close(outputs[1]['targets'][...,0],expected['hold_target_bps'])
    assert outputs[1]['reference_target_evidence']['value_semantics']==reference_policy_contract()['value_semantics']
    assert outputs[1]['entry_bridge_semantics']=='unchanged_initial_teacher_greedy_bridge_no_reference_refresh'
    assert torch.equal(outputs[1]['targets'][...,1],torch.zeros((1,2)))


def test_reference_ema_history_cannot_lose_its_semantic_transition(reference_scope,tmp_path):
    from gx1.contracts import unified_exit_random_access_val_checkpoint_v1 as ema
    from tests.test_native_optimizer_procedure_scope import _write
    _,recipe,_,_=reference_scope
    origin=recipe['candidate_resume_origin']
    directory=tmp_path/'reference-history';directory.mkdir()
    bound_recipe=_write(tmp_path/'reference-history-recipe.json',recipe)
    contract=_write(directory/'CANDIDATE_TRAINING_SESSION_CONTRACT.json',{
        'recipe_source_provenance':{'recipe_audit_path':bound_recipe['path'],'recipe_audit_sha256':bound_recipe['sha256']},
        'training':{'exit_reference_policy':reference_policy_contract()}})
    inherited=ema.bind_candidate_weight_ema_history_v1(session_contract_path=Path(origin['contract']['path']),
        session_contract_sha256=origin['contract']['sha256'])
    receipt={'schema_version':native.TRAINING_CONTINUATION_RECEIPT_SCHEMA,
        'origin':origin,'origin_cursor':native.ENTRY_LEARNABILITY_ORIGIN_CURSOR,
        'origin_state_sha256':native.ENTRY_LEARNABILITY_ORIGIN_STATE_SHA256,
        'destination_session_contract_sha256':contract['sha256'],'inherited_weight_ema_history':inherited,
        'ema_internal_steps':25685,'global_optimizer_steps':5777,'state_preserved':True,
        'optimizer_procedure_changed':False,'identical_future_trajectory_claimed':False,
        'preserved_state_fields':['model_state','target_model_state','optimizer_state','weight_ema_state',
            'lr_scheduler_state','rng_state','epoch_order','training_progress'],
        'exit_reference_policy_change':{'policy':reference_policy_contract(),'teacher_preserved':True,
            'teacher_role':'initial_Q_mu_guess_from_preserved_legacy_weights',
            'entry_bridge':'unchanged_initial_teacher_greedy_bridge_no_reference_refresh',
            'sampler_preserved':True,'holding_time_cap_introduced':False}}
    path=directory/native.TRAINING_CONTINUATION_RECEIPT_NAME
    _write(path,receipt)
    kwargs={'session_contract_path':Path(contract['path']),'session_contract_sha256':contract['sha256']}
    assert ema.bind_candidate_weight_ema_history_v1(**kwargs)['optimizer_step_offset']==19908
    for field,value in [('teacher_preserved',False),('teacher_role','already_learned_Q_mu'),
                         ('entry_bridge','Q_star'),('sampler_preserved',False),('holding_time_cap_introduced',True)]:
        changed=copy.deepcopy(receipt);changed['exit_reference_policy_change'][field]=value;_write(path,changed)
        with pytest.raises(RuntimeError,match='REFERENCE_POLICY_RECEIPT_INVALID'):
            ema.bind_candidate_weight_ema_history_v1(**kwargs)
