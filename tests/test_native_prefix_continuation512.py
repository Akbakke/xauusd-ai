"""Bounded continuation admission and serialized native learning-state preservation."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_native_prefix_learning_measurement import learning_scope, initial_scope, prefix_scope, scope
from tests.test_native_prefix_recipe import native, _write, _bind
from tests.test_native_prefix_coordinator import PrefixHarness, _prepared, equal_tree, digest, trainer


@pytest.fixture
def continuation_scope(learning_scope, tmp_path):
    policy, recipe, _, seal = learning_scope
    recipe['chronological_train_only_measurement']=True
    previous=copy.deepcopy(recipe)
    def rows(name, value):
        path=tmp_path/(name+'.npy');np.save(path,np.asarray(value,dtype='int64'));return _bind(path)
    order=rows('order',range(9000));parents=rows('parents',range(9000));control=rows('control',range(9000,9256))
    contract={'out_bundle_dir':previous['out_bundle_dir'],'chronological_prefix':{'artifacts':recipe['chronological_prefix'],
        'epoch0_parent_order':order,'train_parent_rows':parents,'control_parent_rows':control}}
    cb=_write(tmp_path/'origin-contract.json',contract)
    state=tmp_path/'state.pt';state.write_bytes(b'synthetic-state-only-contract-test')
    pointer={'global_optimizer_steps':256,'next_batch_offset':256,'epoch_index':0,'phase':'train','complete':False,
        'session_contract_sha256':cb['sha256'],'state_sha256':digest(state)}
    pb=_write(tmp_path/'pointer.json',pointer)
    review={'schema_version':'gx1_entry_fuse_normalization_fixed256_train_review_v1','optimizer_steps':256,
        'same_corrected_targets_masks_cohorts_verified':True,'original_target_model_preserved':True,
        'training_pointer':pb,'training_state':_bind(state)}
    cause={'schema_version':'gx1_joint_update_review_v2','decision':'REJECT_AUXILIARY_CONFLICT_AS_CAUSE_ON_MEASURED_TRAIN16',
        'training_state':review['training_state'],'training_pointer':pb,'scope_consumed':True,'resume_authorized':False}
    audit={'schema_version':'gx1_prefix_continuation_order_audit_v1','parent_order':order,'train_rows':parents,
        'control_rows':control,'next_rows':rows('selected',range(4096,8192)),
        'origin_optimizer_steps':256,'stop_after_optimizer_steps':512,'batch_size':16,'previous_entries':4096,
        'additional_entries':4096,'total_entries':8192,'all_unique':True,'all_within_existing_prefix':True,
        'control_overlap':0,'new_model_forwards':0,'new_targets':0,'new_fits':0,'test_data_used':False}
    plan={'schema_version':'gx1_prefix_learning_continuation_plan_v1','from_optimizer_steps':256,
        'stop_after_optimizer_steps':512,'additional_optimizer_steps':256,'maximum_trained_entry_rows':8192,
        'maximum_additional_entry_rows':4096,'max_invocations':1,'teacher_refresh_allowed':False,
        'control_forwards':0,'full_epoch_allowed':False,'full_val_allowed':False,'test_data_used':False,
        'automatic_extension_allowed':False,'origin_review':_write(tmp_path/'review.json',review),
        'origin_contract':cb,'origin_recipe':_write(tmp_path/'origin-recipe.json',previous),
        'cause_review':_write(tmp_path/'cause.json',cause),'order_audit':_write(tmp_path/'order-audit.json',audit)}
    recipe['run_id']+='512';recipe['out_bundle_dir']+='512'
    recipe['chronological_learning_continuation']=_write(tmp_path/'continuation.json',plan)
    policy['chronological_learning_run'].update(run_id=recipe['run_id'],out_bundle_dir=recipe['out_bundle_dir'],
        optimizer_steps=512,maximum_trained_entry_rows=8192,max_invocations=1,
        chronological_train_only_measurement=True,chronological_learning_continuation=recipe['chronological_learning_continuation'])
    seal()
    return policy,recipe,plan,audit,seal


def test_only_bound512_and_one_window_are_admitted(continuation_scope):
    _,r,_,_,_=continuation_scope
    budget={'stop_after_optimizer_steps':512,'stop_after_completed_val_epochs':None,'max_invocation_seconds':12000}
    assert native.require_native_run_scope(r,invocation_number=1,execution_budget=budget)==512
    for value in [256,511,513]:
        with pytest.raises(RuntimeError,match='BUDGET'):
            native.require_native_run_scope(r,execution_budget={**budget,'stop_after_optimizer_steps':value})
    with pytest.raises(RuntimeError,match='INVOCATION'):native.require_native_run_scope(r,invocation_number=2)


@pytest.mark.parametrize('fault',['teacher','source','order','origin','unbound'])
def test_continuation_rejects_changed_learning_state_data_or_authority(continuation_scope,fault):
    p,r,plan,audit,seal=continuation_scope
    if fault=='teacher':r['exit_reference_policy']={'changed':True}
    if fault=='source':r['source_bindings']={**r['source_bindings'],'changed-model':'a'*64}
    if fault=='order':
        path=Path(audit['next_rows']['path']);np.save(path,np.arange(4095,8191,dtype='int64'))
        audit['next_rows']=_bind(path);plan['order_audit']=_write(Path(plan['order_audit']['path']),audit)
    if fault=='origin':
        review=json.loads(Path(plan['origin_review']['path']).read_text());review['training_state']['sha256']='a'*64
        plan['origin_review']=_write(Path(plan['origin_review']['path']),review)
    if fault=='unbound':p['chronological_learning_run'].pop('chronological_learning_continuation')
    if fault in ('order','origin'):
        r['chronological_learning_continuation']=_write(Path(r['chronological_learning_continuation']['path']),plan)
        p['chronological_learning_run']['chronological_learning_continuation']=r['chronological_learning_continuation']
    seal()
    with pytest.raises(RuntimeError):native.require_native_run_scope(r,invocation_number=1)


def test_serialized_fork_preserves_all_learning_state_and_resume_matches(tmp_path):
    artifacts=tmp_path/'artifacts';artifacts.mkdir();h=PrefixHarness(tmp_path/'run',_prepared(artifacts))
    origin=h.root/'ORIGIN';h.run_prefix(origin,256)
    pointer=h.pointer(origin);original=pointer.read_bytes();state=h.state(origin);original_state=copy.deepcopy(state)
    meta=json.loads(original);state_path=pointer.parent/f"candidate_training_state_slot_{meta['slot']}.pt"
    contract=json.loads((pointer.parent/trainer._CANDIDATE_TRAINING_CONTRACT_FILENAME).read_text())
    continuation={'plan_binding':{'path':'synthetic-plan.json','sha256':'a'*64},
        'plan':{'from_optimizer_steps':256,'stop_after_optimizer_steps':512},'origin_contract':contract,
        'origin_pointer':_bind(pointer),'origin_review':{'training_state':_bind(state_path)}}
    direct=h.root/'DIRECT';split=h.root/'SPLIT'
    h.batches.clear();_,d=h.run_prefix(direct,258,chronological_continuation=continuation)
    expected=h.state(direct);batches=list(h.batches);h.batches.clear()
    h.run_prefix(split,257,chronological_continuation=continuation)
    _,s=h.run_prefix(split,258,expected_pointer=digest(h.pointer(split)),chronological_continuation=continuation)
    actual=h.state(split)
    for key in ['model_state','target_model_state','optimizer_state','weight_ema_state','lr_scheduler_state',
                'rng_state','epoch_order','training_progress']:equal_tree(expected[key],actual[key])
    assert h.batches==batches and [i for batch in batches for i in batch]==list(range(4500))[::-1][4096:4128]
    assert d['global_optimizer_steps']==s['global_optimizer_steps']==258 and h.validation_batches==0
    assert len(set(h.teacher_hashes))==1 and pointer.read_bytes()==original
    equal_tree(h.state(origin),original_state)
    assert _bind(state_path)==continuation['origin_review']['training_state']
    receipt=json.loads((h.pointer(direct).parent/'PREFIX_CONTINUATION_RECEIPT.json').read_text())
    assert receipt['changed_state_fields']==['session_contract_sha256'] and not receipt['optimizer_reset']
    with pytest.raises(RuntimeError,match='CONTRACT_CHANGED'):
        h.run_prefix(h.root/'BAD',258,chronological_continuation=continuation,dropout=.1)


def test_native_dispatch_measures_fixed512_without_resetting_origin(continuation_scope,monkeypatch,tmp_path):
    from tests.test_native_prefix_recipe import runner
    import torch
    _,recipe,_,_,seal=continuation_scope;rb=seal();seen=[]
    budget={'stop_after_optimizer_steps':512,'stop_after_completed_val_epochs':None,'max_invocation_seconds':12000}
    monkeypatch.setattr(runner.trainer,'_require_cuda_trainer_guard_execution',lambda **kw:None)
    monkeypatch.setattr(runner.trainer,'_resolve_device',lambda _:torch.device('cpu'))
    monkeypatch.setattr(runner,'_require_native_full_train_recipe',lambda *a:(recipe,{},{}))
    monkeypatch.setattr(runner.launch_owner,'require_candidate_execution_budget',lambda *a,**kw:budget)
    monkeypatch.setattr(runner,'_build_bound_full_train_components',lambda **kw:{'chronological_prefix':kw['chronological_prefix']})
    monkeypatch.setattr(runner,'_restore_prefix_initial_measurement_state',lambda **kw:seen.append('initial_contract_state'))
    def train(**kw):
        assert kw['components']['chronological_continuation']['plan']['from_optimizer_steps']==256
        assert kw['execution_budget']['stop_after_optimizer_steps']==512
        seen.append('native_continuation')
        raise runner.trainer._CandidateExecutionPaused({'reason':'optimizer_step_ceiling','global_optimizer_steps':512})
    monkeypatch.setattr(runner,'_run_bound_full_train_candidate',train)
    def measure(**kw):
        assert kw['optimizer_steps']==512 and kw['scope']['continuation']['plan']['stop_after_optimizer_steps']==512
        seen.append('final_online512');return {'path':'synthetic','sha256':'a'*64}
    monkeypatch.setattr(runner,'_run_prefix_initial_measurement',measure)
    receipt=tmp_path/'pause.json';receipt.write_text('{}')
    monkeypatch.setattr(runner.trainer,'_write_candidate_execution_pause_receipt',lambda *a,**kw:receipt)
    monkeypatch.setattr(runner,'_native_resume_state',lambda **kw:{'global_optimizer_steps':512})
    result=runner.run_guarded_native_candidate_invocation(recipe_path=Path(rb['path']),recipe_file_sha256=rb['sha256'],
        execution_budget_path=tmp_path/'budget.json',execution_budget_file_sha256='b'*64)
    assert seen==['initial_contract_state','native_continuation','final_online512']
    assert result['resume_state']['global_optimizer_steps']==512 and 'chronological_final_measurement' in result['pause']
