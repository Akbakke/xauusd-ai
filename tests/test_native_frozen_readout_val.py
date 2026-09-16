from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.contracts import unified_exit_native_candidate_campaign_v1 as native
from gx1.contracts.unified_exit_reference_policy_v1 import build_reference_policy_hold_targets, reference_policy_contract
from gx1.scripts import run_unified_exit_random_access_val_v1 as val
from tests.test_liquidation_relative_learning_v1 import relative_fixture
from tests.test_native_learning_calibration_scope import scope
from tests.test_unified_exit_bounded_val import _cohort
from tests import test_unified_exit_random_access_state_view_v1 as states


@pytest.mark.parametrize('count', [2, 3, 121, 122, 600])
def test_val_reference_targets_match_train_owner_including_weekend_and_censor(relative_fixture, count):
    view = states._materialize(0, counts=(count,count), reference_policy=reference_policy_contract())
    clock = states._clock(); authority = states._authority(clock,known=True)
    factory = SimpleNamespace(entries=[{'available_state_count':count,'entry_m1_start_row':479}],
        times=clock,closure=authority,economics_objective_contract=states._objective(),
        economic_step_provider=states._EconomicProvider(authority['artifact_sha256']),
        economic_step_manifest={'manifest_sha256':'d'*64,'economic_step_model_sha256':'b'*64,
                                'economic_step_source_manifest_sha256':'c'*64})
    lengths,policy,_,trace = val._bounded_reference_trace(state_factory=factory,child_rows=[0],device=torch.device('cpu'))
    source = view['reference_policy_trace'];steps=source['steps'];n=len(steps)
    assert lengths == [n] and n == min(120,count-1)
    assert torch.equal(trace['hold_reward_bps'],torch.tensor(np.array([s['liquidation_relative_reward_bps'][:,0] for s in steps]))[None])
    assert torch.equal(trace['elapsed_wall_clock_gamma'],torch.tensor([[s['elapsed_wall_clock_gamma'] for s in steps]],dtype=torch.float32))
    q = torch.tensor([[[-17.,0.],[23.,0.]]])
    result = build_reference_policy_hold_targets(policy=policy,boundary_action_q_bps=q,**trace)
    expected=q[...,0].clone()
    for step in reversed(steps):
        expected=torch.tensor(step['liquidation_relative_reward_bps'][:,0])[None]+torch.tensor(step['elapsed_wall_clock_gamma'],dtype=torch.float32)*(119/120)*expected
    torch.testing.assert_close(result['hold_target_bps'],expected,rtol=2e-6,atol=2e-5)
    assert trace['boundary_right_censored_mask'].tolist()==[[n==count-1]*2]
    assert bool((result['boundary_bootstrap_weight']>0).all())
    factory.closure=states._authority(clock,known=False)
    if n>4:
        with pytest.raises(RuntimeError):val._bounded_reference_trace(state_factory=factory,child_rows=[0],device=torch.device('cpu'))


@pytest.fixture
def frozen_scope(scope,tmp_path):
    policy,recipe,write,save=scope
    cohort=_cohort(tmp_path,ids=tuple(range(256)))
    pp=Path(cohort['plan']['path']);plan=json.loads(pp.read_text())
    checkpoint=tmp_path/'candidate_training_state_slot_1.pt';checkpoint.write_bytes(b'preserved-state-fixture')
    state_binding={'path':str(checkpoint),'sha256':native.file_sha256(checkpoint)}
    raw={'schema_version':'gx1_candidate_training_session_v1','slot':1,'state_sha256':state_binding['sha256'],
         'session_contract_sha256':'a'*64,'phase':'train','epoch_index':1,'next_batch_offset':1728,
         'global_optimizer_steps':5809,'complete':False}
    pointer=write('CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json',raw)
    state={k:raw[k] for k in ('session_contract_sha256','phase','epoch_index','next_batch_offset','global_optimizer_steps','complete')}
    state.update(training_pointer=pointer,training_state=state_binding,active_val_cursor=None,
                 active_val_model_forwards=0,epoch_schedule_sha256='b'*64)
    rb=write('origin_recipe.json',{'immutable_origin':True})
    cursor=write('origin_cursor.json',native.build_native_cursor(recipe=rb,resume_state=state,outcome='RESUMABLE'))
    plan.update(base_online_checkpoint=state_binding,val_root=write('val_root.json',{'role':'fixture-only'}))
    for row in plan['selection']['rows']:row['parent_entry_row_index']=row['entry_row_index']+10000
    pb=write(pp.name,plan)
    recipe.pop('candidate_resume_origin');policy.pop('native_learning_calibration')
    recipe['frozen_readout_evaluation']={'plan':pb,'origin_cursor':cursor,'arm':'baseline'}
    recipe['files']['random_access_root']=plan['val_root']
    policy['frozen_readout_evaluation']={'plan':pb,'origin_cursor':cursor,'allowed_arms':['baseline','candidate'],
        'optimizer_steps':0,'max_invocations_per_arm':1,'full_val_allowed':False,'test_data_used':False}
    save()
    return policy,recipe,write,save,state


def test_readonly_scope_admits_only_one_window_and_zero_new_steps(frozen_scope):
    _,recipe,_,_,state=frozen_scope
    for arm in ('baseline','candidate'):
        recipe['frozen_readout_evaluation']['arm']=arm
        assert native.require_native_run_scope(recipe,invocation_number=1)==5809
    for invocation in (0,2,True):
        with pytest.raises(RuntimeError,match='INVOCATION_INVALID'):native.require_native_run_scope(recipe,invocation_number=invocation)
    budget={'stop_after_optimizer_steps':5809,'expected_active_pointer_sha256':None,
            'stop_after_completed_val_epochs':None,'max_invocation_seconds':12000}
    assert native.require_native_run_scope(recipe,execution_budget=budget)==5809
    for key,bad in [('stop_after_optimizer_steps',5810),('stop_after_optimizer_steps',5809.0),
                    ('stop_after_completed_val_epochs',2),('max_invocation_seconds',12001),
                    ('expected_active_pointer_sha256','a'*64),('resume_probe_val_rows',32)]:
        with pytest.raises(RuntimeError,match='BUDGET_INVALID'):
            native.require_native_run_scope(recipe,execution_budget={**budget,key:bad})


@pytest.mark.parametrize('fault',['training','missing_scope','extra_steps','full_val','old_origin','wrong_arm','old_checkpoint'])
def test_readonly_scope_cannot_open_training_or_change_frozen_origin(frozen_scope,fault):
    policy,recipe,write,save,state=frozen_scope
    if fault=='training':policy['training_enabled']=True
    elif fault=='missing_scope':policy.pop('frozen_readout_evaluation')
    elif fault=='extra_steps':policy['frozen_readout_evaluation']['optimizer_steps']=1
    elif fault=='full_val':policy['frozen_readout_evaluation']['full_val_allowed']=True
    elif fault=='old_origin':recipe['candidate_resume_origin']={}
    elif fault=='wrong_arm':recipe['frozen_readout_evaluation']['arm']='retune'
    else:Path(state['training_state']['path']).write_bytes(b'changed-origin')
    save()
    with pytest.raises(RuntimeError):native.require_native_run_scope(recipe,invocation_number=1)


def test_guarded_native_dispatch_evaluates_without_calling_trainer(frozen_scope,tmp_path,monkeypatch):
    from gx1.scripts import run_unified_exit_random_access_full_train_v1 as runner
    from gx1.contracts import unified_exit_random_access_val_checkpoint_v1 as checkpoints
    _,recipe,write,_,state=frozen_scope
    recipe.update(out_bundle_dir=str(tmp_path/'evaluation'/'CANDIDATE_BUNDLE'),dataset_run_id='fixture',
                  seed_launch={'path':'fixture'},seed_authority={'path':'fixture','sha256':'a'*64},
                  smoke_full_val={'path':'fixture'})
    recipe['trainer_cli'].update(epochs=30,seed=1,learning_rate=.001,weight_decay=0.)
    recipe['val_limits'].update(max_model_forwards=10000,max_state_views=100000)
    context={'frame':pd.DataFrame({'entry_row_index':range(5508),'parent_entry_row_index':np.arange(5508)+10000}),
             'state_factory':object(),'parent_coordinate_evidence':{},'val_sequence_audit':tmp_path/'audit',**recipe['val_limits']}
    components={'model':torch.nn.Linear(1,1),'val_ds':object(),'native_val_context':context,
                'seed_binding':{'model_state_sha256':'seed'}}
    budget={'stop_after_optimizer_steps':5809,'expected_active_pointer_sha256':None,
            'stop_after_completed_val_epochs':None,'max_invocation_seconds':12000}
    guard=[]
    monkeypatch.setattr(runner.trainer,'_require_cuda_trainer_guard_execution',lambda **kw:guard.append(kw))
    monkeypatch.setattr(runner.trainer,'_resolve_device',lambda _:torch.device('cuda'))
    monkeypatch.setattr(runner,'_require_native_full_train_recipe',lambda *_:(recipe,{},{}))
    monkeypatch.setattr(runner.launch_owner,'require_candidate_execution_budget',lambda *a,**kw:budget)
    monkeypatch.setattr(runner,'_build_bound_full_train_components',lambda **kw:components)
    monkeypatch.setattr(runner.val,'_read',lambda _: {'checkpoint_binding':{'model_state_sha256':'seed'}})
    def forbidden(**kw):raise AssertionError('read-only evaluation reached trainer')
    monkeypatch.setattr(runner,'_run_bound_full_train_candidate',forbidden)
    def binding(**kw):
        assert kw['arm']=='baseline' and kw['target_model'] is not kw['boundary_model']
        return {'model_variant':'frozen_online_readout'}
    monkeypatch.setattr(checkpoints,'bind_frozen_readout_checkpoint_v1',binding)
    calls=[]
    def evaluate(**kw):
        assert len(kw['frame'])==256 and kw['frame']['parent_entry_row_index'].tolist()==list(range(10000,10256))
        assert kw['exit_policy_batch_size']==256 and kw['cpu_pipeline_workers']==8
        assert kw['compute_guard_max_wall_seconds']==10800
        assert kw['exit_boundary_model'] is not kw['candidate_target_model']
        for key in ('rollout_progress_path','result_path'):
            kw[key].parent.mkdir(parents=True,exist_ok=True);kw[key].write_text('{}')
        calls.append(kw)
        return {'decision':'COMPLETE_WITH_RIGHT_CENSORING'}
    monkeypatch.setattr(runner.val,'evaluate_bound_full_val_v1',evaluate)
    result=runner.run_guarded_native_candidate_invocation(recipe_path=tmp_path/'recipe.json',recipe_file_sha256='a'*64,
        execution_budget_path=tmp_path/'budget.json',execution_budget_file_sha256='b'*64)
    assert guard==[{'execution_tier':'canonical'}] and len(calls)==1
    assert result['decision']=='PAUSED_RESUMABLE' and result['resume_state']==state
    observation=json.loads(Path(result['observation']['path']).read_text())
    assert observation['optimizer_steps']==0 and observation['training_enabled'] is False
    assert native.require_frozen_readout_evaluation(recipe)['origin_resume_state']==state


def test_campaign_preparer_uses_preserved_epoch_for_readonly_counter(frozen_scope,tmp_path,monkeypatch):
    from gx1.scripts import materialize_local_random_access_campaign_v2 as materializer
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    _,recipe,write,_,_=frozen_scope
    monkeypatch.setattr(materializer,'_source_commit',lambda repo:'c'*40)
    monkeypatch.setattr(native,'require_native_recipe_metadata',lambda *a,**kw:(recipe,65295))
    prior=tmp_path/'prior_campaign.json'
    def stop_at_existing_prior_validation(path,sha):
        assert path==prior
        raise RuntimeError('reached_original_campaign_checks')
    monkeypatch.setattr(campaign,'read_bound_json',stop_at_existing_prior_validation)
    # Preserved5809 is below its real epoch2 boundary8162, not genesis4081.
    with pytest.raises(RuntimeError,match='reached_original_campaign_checks'):
        materializer.materialize_native_candidate_campaign(repo=tmp_path,output=tmp_path/'campaign',runtime=tmp_path/'runtime',
            gpu_uuid='fixture',prepared_boot_path=tmp_path/'boot',prepared_boot_file_sha256='a'*64,
            certificate_path=tmp_path/'certificate',prior_campaign_path=prior,prior_campaign_file_sha256='a'*64,
            selection_path=tmp_path/'selection',selection_file_sha256='a'*64,
            recipe_path=tmp_path/'recipe',recipe_file_sha256='a'*64,window_count=1)
