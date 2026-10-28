"""Synthetic scope checks: no model construction, data load or optimizer step."""
from pathlib import Path

import pytest

from gx1.contracts import gx1_capped_execution_v1 as capped
from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
from gx1.scripts import run_entry_native_direction_research_window_v1 as dispatcher
from gx1.scripts import run_entry_native_direction_order_qualification_v1 as qualifier


def test_order_private_worker_requires_gpu_guard_before_campaign_or_data(monkeypatch,tmp_path):
    calls=[]
    def rejected():
        calls.append('guard')
        raise RuntimeError('synthetic guard rejection')
    monkeypatch.setattr(capped,'require_guarded_cuda_trainer_execution',rejected)
    monkeypatch.setattr(authority,'require_native_direction_research_context',lambda *args:calls.append('context'))
    with pytest.raises(RuntimeError,match='guard rejection'):
        qualifier.run_order_qualification(policy_path=tmp_path/'missing',policy_file_sha256='0'*64,progress_path=tmp_path/'progress')
    assert calls==['guard'] and list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('scope',[authority.ORDER_PRIVATE_SCOPE,authority.ORDER_SUPPORTED_SCOPE])
def test_existing_guarded_dispatcher_selects_only_explicit_order_scope(monkeypatch,tmp_path,scope):
    calls=[]
    monkeypatch.setattr(capped,'require_guarded_cuda_trainer_execution',lambda:calls.append('guard') or {})
    monkeypatch.setattr(authority,'require_native_direction_research_context',lambda *args:
        (None,None,None,None,{'run_scope':scope}))
    def run(**kwargs):
        calls.append('order_worker')
        return {'scope':'synthetic'}
    monkeypatch.setattr(qualifier,'run_order_qualification',run)
    result=dispatcher.run_native_direction_research_window(policy_path=tmp_path/'window',
        policy_file_sha256='0'*64,progress_path=tmp_path/'progress')
    assert result=={'scope':'synthetic'} and calls==['guard','order_worker']
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('supported',[False,True])
def test_failed_worker_publishes_terminal_failure_then_exits_nonzero(monkeypatch,tmp_path,supported):
    from gx1.contracts import entry_model_native_train_recipe_v1 as recipes
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix_authority
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as prefix
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    numeric=recipes.native_direction_research_recipe(batch_size=16,max_updates=2)
    original=recipes.native_direction_research_recipe(batch_size=16,max_updates=2976)
    def ref(name):return {'path':str(tmp_path/name),'sha256':'a'*64}
    launch={'run_scope':authority.ORDER_SUPPORTED_SCOPE if supported else authority.ORDER_PRIVATE_SCOPE,
        'numerical_recipe':numeric,'execution_binding_path':str(tmp_path/'binding'),
        'qualification_design':ref('design'),'gc_layout_metadata_parity':ref('gc'),'input_manifest':ref('input'),
        'source_owners':{prefix_authority.ORDER_COMPONENT_SOURCE:ref('component')},'fold':authority.BPS_FOLD,
        'runtime_root':str(tmp_path/'runtime'),'max_published_json_bytes':33554432}
    window={'recipe':ref('recipe'),'output_directory':str(tmp_path/'output'),'campaign_cursor_path':str(tmp_path/'cursor')}
    plan={'plan_sha256':'b'*64};invocation={'batch_size':16,'invocation_sha256':'c'*64}
    monkeypatch.setattr(capped,'require_guarded_cuda_trainer_execution',lambda:{})
    monkeypatch.setattr(authority,'require_native_direction_research_context',lambda *args:(window,plan,invocation,None,launch))
    monkeypatch.setattr(common,'_hash_file',lambda path:ref('binding'))
    def read(item,metadata):
        if item==ref('binding'):return {'plan':ref('plan')}
        if item==common.INPUT_DESIGN:return {'budget':{'max_rchar_bytes':137438953472}}
        if item==prefix_authority.ORDER_ORIGIN['launch_recipe']:return {'numerical_recipe':original}
        if item==prefix_authority.ORDER_ORIGIN['initial_measurement']:return {}
        if item==launch['gc_layout_metadata_parity']:return {'status':'COMPLETE_CONSUMED','metadata_exact_parity':True,'test_cases_passed':6,'counters':{}}
        if item==launch['qualification_design'] and supported:return {'schema_version':'gx1_order_supported_qualification_design_v1',
            'status':'DESIGN_ONLY_NO_EXECUTION_AUTHORITY','seed':prefix.SEED,'prefix_rows':authority.BPS_FOLD['prefix_rows'],
            'batch_size':16,'test_data_used':False,**{key:ref(key) for key in ('prior_failure_review','prior_support_report','prior_support_run_result','prior_support_root_review')}}
        if item==launch['input_manifest']:return {'view_source_bindings':{'direction_component_source':ref('component')}}
        return {}
    monkeypatch.setattr(common,'_read',read)
    monkeypatch.setattr(prefix,'_metadata',lambda *args,**kwargs:{'info':{'metadata':{}},'ancestor_plan':{'payloads':{}}})
    monkeypatch.setattr(common,'_freeze',lambda *args:None)
    monkeypatch.setattr(common,'_install_read_guard',lambda *args,**kwargs:[])
    monkeypatch.setattr(common,'_io',lambda:{'rchar':0})
    monkeypatch.setattr(common,'_deny_legacy_work',lambda *args:None)
    def fail(*args,**kwargs):raise RuntimeError('synthetic blocked batch')
    monkeypatch.setattr(prefix,'_prepare',fail)
    published={}
    def publish(path,value,counters,limit):
        published[Path(path).name]=value
        return {'path':str(path),'sha256':'d'*64}
    monkeypatch.setattr(common,'_json',publish)
    monkeypatch.setattr(authority,'build_research_cursor',lambda **kwargs:{'outcome':kwargs['outcome']})
    monkeypatch.setattr(authority,'build_research_progress',lambda **kwargs:{'outcome':kwargs['outcome'],'terminal':True})
    monkeypatch.setattr(authority,'require_research_progress',lambda *args,**kwargs:None)
    with pytest.raises(RuntimeError,match='NATIVE_ORDER_QUALIFICATION_FAILED:RuntimeError:synthetic blocked batch'):
        qualifier.run_order_qualification(policy_path=tmp_path/'policy',policy_file_sha256='0'*64,progress_path=tmp_path/'progress.json')
    assert published['RESULT.json']['decision']=='FAILED'
    assert published['progress.json']=={'outcome':'FAILED','terminal':True}
    assert published['RESULT.json']['counts']['optimizer_updates']==0


def test_supported_mechanics_selection_is_exact_original_set_in_time_order():
    import numpy as np
    ordered=np.arange(300,dtype=np.int64)
    diagnostic_ids=np.random.RandomState(9).permutation(300).astype(np.int64)[:256]
    expected=np.sort(diagnostic_ids[:32]).tolist();design={'parents_sorted':expected}
    chosen=qualifier._select_qualification_parents(ordered,diagnostic_ids,design,supported=True,np=np)
    assert chosen.tolist()==expected and np.all(chosen[1:]>chosen[:-1])
    assert qualifier._select_qualification_parents(ordered,diagnostic_ids,None,supported=False,np=np).tolist()==list(range(32))
    wrong={'parents_sorted':expected.copy()};wrong['parents_sorted'][0]+=1
    with pytest.raises(RuntimeError,match='PREDECLARED_SUPPORTED_PARENTS'):
        qualifier._select_qualification_parents(ordered,diagnostic_ids,wrong,supported=True,np=np)
    wrong={'parents_sorted':expected.copy()};wrong['parents_sorted'][0]=True
    with pytest.raises(RuntimeError,match='SUPPORTED_PARENT_ID_TYPES'):
        qualifier._select_qualification_parents(ordered,diagnostic_ids,wrong,supported=True,np=np)
    with pytest.raises(RuntimeError,match='PREDECLARED_SUPPORTED_PARENTS'):
        qualifier._select_qualification_parents(ordered[ordered!=chosen[0]],diagnostic_ids,design,supported=True,np=np)
