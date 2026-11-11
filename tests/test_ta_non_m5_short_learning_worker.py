"""Synthetic full-period observation and scoped budgets; no native training."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gx1.audit import trace_market_state_ta_m5_original_cuda_prefix_v1 as worker
from gx1.contracts import market_state_ta_non_m5_short_learning_v1 as short
from gx1.contracts.market_state_time_query_native_F_objective_v1 import contract
from tests.test_ta_m5_c171_short_learning_worker import panel, recipe as old_recipe, tree_sha


def recipe():
    value=old_recipe()
    value.update(run_scope=short.SCOPE,max_forwards=768,max_evaluation_forwards=512,
        max_model_constructions=2,max_initial_rng_imports=1,dedicated_non_m5_side=True,
        panel_batch_indices=list(range(256)))
    return value


def test_old_and_new_worker_bounds_cannot_be_substituted():
    worker._require_execution_bounds(old_recipe(),short_learning=True)
    worker._require_execution_bounds(recipe(),short_learning=True,non_m5_learning=True)
    for new_recipe,flag in ((recipe(),False),(old_recipe(),True)):
        with pytest.raises(RuntimeError):worker._require_execution_bounds(new_recipe,short_learning=True,non_m5_learning=flag)
    with pytest.raises(RuntimeError):worker._require_execution_bounds(recipe(),short_learning=False,non_m5_learning=True)


@pytest.mark.parametrize('key,value',[
    ('max_forwards',769),('max_private_updates',257),('max_evaluation_forwards',513),
    ('max_model_constructions',3),('max_initial_rng_imports',2),('dedicated_non_m5_side',False),
    ('panel_batch_indices',list(range(16))),('max_checkpoint_writes',3),('max_model_seconds',1801),
])
def test_new_bounds_fail_before_work(key,value):
    changed=recipe();changed[key]=value
    with pytest.raises(RuntimeError):worker._require_execution_bounds(changed,short_learning=True,non_m5_learning=True)


def test_entire4096_period_is_measured_with_original_loss_and_passivity(panel):
    panel.args['recipe'].update(recipe())
    loss_sha=contract(ta_only=True,dedicated_m5_side=True,m5_equal_observation=True,dedicated_non_m5_side=True)['contract_sha256']
    panel.args['recipe']['expected_loss_contract_sha256']=loss_sha
    real=panel.args['evaluate']
    def evaluate(left):
        report,parents,issues=real(left);report['loss_contract_sha256']=loss_sha
        return report,parents,issues
    panel.args['evaluate']=evaluate
    measured,arrays,components,digest=worker._short_train_panel(panel.model,**panel.args,non_m5_learning=True)
    assert panel.seen==list(range(0,4096,16))
    assert measured['rows']==4096 and measured['batches']==256 and len(components)==256
    assert np.array_equal(arrays['parents'],np.arange(4096))
    assert measured['loss_contract_sha256']==loss_sha and tree_sha(panel.capture())==panel.before
    assert panel.counters==dict(forwards=256,evaluation_forwards=256,forward_returns=256,
        extra_target_gathers=512,loss_component_owner_calls=256)


@pytest.mark.parametrize('failure',['none','outside_old_panel_axis','outside_old_panel_m5','outside_old_panel_side','loss'])
def test_full_initial_reference_checks_rows_outside_old_control_panel(monkeypatch,failure):
    previous=short.initial.previous
    reference={k:np.zeros(shape,dtype) for k,(shape,dtype) in previous._layout(8192,np=np,prefix=True).items()}
    reference['phase'][4096:]=1
    reference['parents'][:4096]=np.arange(4096);reference['parents'][4096:]=np.arange(4096)
    actual={k:v[:4096].copy() for k,v in reference.items() if k!='phase'}
    losses=[1.]*256
    class Archive:
        files=list(reference)
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def __getitem__(self,key):return reference[key]
    monkeypatch.setattr(np,'load',lambda *a,**kw:Archive())
    monkeypatch.setattr(short,'_read',lambda ref:dict(batch_records=[dict(phase='INITIAL_TRAIN',batch_index=i,loss=1.) for i in range(256)]))
    row=16*100+3
    if failure=='outside_old_panel_axis':actual['parents'][row]+=1
    if failure=='outside_old_panel_m5':actual['future_raw'][row,0,1]=1.
    if failure=='outside_old_panel_side':actual['state_raw'][row,4,1]=1.
    if failure=='loss':losses[100]=2.
    counters={'archive_decodes':0}
    if failure!='none':
        with pytest.raises(RuntimeError):short.require_full_initial_reference(actual,losses,counters,np=np)
    else:
        proof=short.require_full_initial_reference(actual,losses,counters,np=np)
        assert proof['rows']==4096 and proof['batches']==256 and proof['max_abs_loss']==0.
        assert proof['axes_targets_m5_activity_bitexact'] is proof['cloned_sides_within_declared_tolerance'] is True
    assert counters['archive_decodes']==1


def test_new_pre_model_failure_preserves_failed_scope_without_training(monkeypatch,tmp_path):
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    from gx1.contracts.market_state_time_query_fold1_fit_v1 import ta_sf_numerical_recipe
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    value=recipe();value['numerical_recipe']=ta_sf_numerical_recipe(dedicated_m5_side=True,m5_equal_observation=True,dedicated_non_m5_side=True)
    value['candidate_inputs']={'short_initial_checkpoint':{'path':'/fixture/INITIAL.pt','sha256':'a'*64}}
    original=ta_sf_numerical_recipe(dedicated_m5_side=True,m5_equal_observation=True)
    recipe_ref={'path':str(short.RECIPE_PATH),'sha256':'a'*64}
    window=dict(recipe=recipe_ref,output_directory=str(tmp_path/'output'),campaign_cursor_path=str(tmp_path/'cursor.json'),max_invocation_seconds=3600)
    plan=dict(source_commit='b'*40,plan_sha256='a'*64)
    monkeypatch.setattr(cap,'require_guarded_cuda_trainer_execution',lambda:{})
    monkeypatch.setattr(authority,'require_native_direction_research_context',lambda *a:(window,plan,{},{},value))
    monkeypatch.setattr(short,'read_origin',lambda r:dict(original_recipe={'numerical_recipe':original}))
    monkeypatch.setattr(short,'ROOT',tmp_path)
    def stop(*args):raise RuntimeError('synthetic source failure')
    monkeypatch.setattr(worker,'_bookend',stop)
    published={}
    def publish(path,payload,*args):
        published[path.name]=deepcopy(payload);return {'path':str(path),'sha256':'a'*64}
    monkeypatch.setattr(common,'_json',publish)
    monkeypatch.setattr(authority,'build_research_cursor',lambda **kw:kw)
    monkeypatch.setattr(authority,'build_research_progress',lambda **kw:kw)
    monkeypatch.setattr(authority,'require_research_progress',lambda *a,**kw:None)
    with pytest.raises(RuntimeError,match='synthetic source failure'):
        worker.run_original_cuda_prefix(policy_path=tmp_path/'window.json',policy_file_sha256='a'*64,
            progress_path=tmp_path/'progress.json',short_learning=True,non_m5_learning=True)
    result=published['RESULT.json']
    assert result['schema_version']==short.RESULT_SCHEMA and result['outcome']=='FAILED'
    assert result['dedicated_non_m5_side'] is True and result['final_state'] is None
    assert result['full_initial_reference'] is None and result['original_training_rng_endpoints_matched'] is False
    assert result['counts']['model_constructions']==result['counts']['optimizer_updates']==0
    assert published['TERMINAL.json']['exit_code']==1
