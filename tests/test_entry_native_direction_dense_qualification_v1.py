"""Synthetic selection and fail-closed checks; no real data or CUDA."""
import numpy as np
import pytest

from gx1.contracts import gx1_capped_execution_v1 as capped
from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
from gx1.scripts import run_entry_native_direction_dense_qualification_v1 as worker


def test_cuda_guard_rejects_before_admission_or_output(monkeypatch,tmp_path):
    calls=[]
    def rejected(): calls.append('guard'); raise RuntimeError('synthetic guard')
    monkeypatch.setattr(capped,'require_guarded_cuda_trainer_execution',rejected)
    monkeypatch.setattr(authority,'require_native_direction_research_context',lambda *args:calls.append('context'))
    with pytest.raises(RuntimeError,match='synthetic guard'):
        worker.run_dense_qualification(policy_path=tmp_path/'missing',policy_file_sha256='0'*64,progress_path=tmp_path/'progress')
    assert calls==['guard'] and list(tmp_path.iterdir())==[]


def test_fixed_supported_parents_are_not_reselected():
    ordered=np.arange(256,dtype=np.int64)[::-1].copy()
    diagnostics=ordered.copy(); expected=sorted(diagnostics[:32].tolist())
    design={'parents_sorted':expected}
    assert worker.select_qualification_parents(ordered,diagnostics,design,np=np).tolist()==expected
    for wrong in ({'parents_sorted':expected[:-1]}, {'parents_sorted':expected[:-1]+[999]}, {'parents_sorted':[True]*32}):
        with pytest.raises(RuntimeError):
            worker.select_qualification_parents(ordered,diagnostics,wrong,np=np)


def test_guarded_dispatcher_selects_only_explicit_dense_scope(monkeypatch,tmp_path):
    from gx1.scripts import run_entry_native_direction_research_window_v1 as dispatcher
    calls=[]
    monkeypatch.setattr(capped,'require_guarded_cuda_trainer_execution',lambda:calls.append('guard') or {})
    monkeypatch.setattr(authority,'require_native_direction_research_context',lambda *args:
        (None,None,None,None,{'run_scope':authority.DENSE_PRIVATE_SCOPE}))
    monkeypatch.setattr(worker,'run_dense_qualification',lambda **kwargs:calls.append('dense') or {'scope':'synthetic'})
    result=dispatcher.run_native_direction_research_window(policy_path=tmp_path/'policy',
        policy_file_sha256='0'*64,progress_path=tmp_path/'progress')
    assert result=={'scope':'synthetic'} and calls==['guard','dense']
    assert list(tmp_path.iterdir())==[]


def test_failed_dense_worker_publishes_failure_before_nonzero_exit(monkeypatch,tmp_path):
    from pathlib import Path
    from gx1.contracts import entry_model_native_train_recipe_v1 as recipes
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix_authority
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as prefix
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    def ref(name):return {'path':str(tmp_path/name),'sha256':'a'*64}
    launch={'run_scope':authority.DENSE_PRIVATE_SCOPE,
        'numerical_recipe':recipes.native_direction_research_recipe(batch_size=16,max_updates=2,dense_minutes=True),
        'execution_binding_path':str(tmp_path/'binding'),'qualification_design':ref('design'),
        'gc_layout_metadata_parity':ref('gc'),'dense_source_transition_proof':ref('proof'),
        'dense_target':ref('targets'),'fold':authority.BPS_FOLD,'runtime_root':str(tmp_path/'runtime'),
        'max_published_json_bytes':33554432}
    window={'recipe':ref('recipe'),'output_directory':str(tmp_path/'output'),'campaign_cursor_path':str(tmp_path/'cursor')}
    plan={'plan_sha256':'b'*64};invocation={'batch_size':16,'invocation_sha256':'c'*64}
    monkeypatch.setattr(capped,'require_guarded_cuda_trainer_execution',lambda:{})
    monkeypatch.setattr(authority,'require_native_direction_research_context',lambda *args:(window,plan,invocation,None,launch))
    monkeypatch.setattr(common,'_hash_file',lambda path:ref('binding'))
    def read(item,metadata):
        if item==ref('binding'):return {'plan':ref('plan')}
        if item==common.INPUT_DESIGN:return {'budget':{'max_rchar_bytes':137438953472}}
        if item==prefix_authority.ORDER_ORIGIN['initial_measurement']:return {}
        if item==launch['gc_layout_metadata_parity']:return {'status':'COMPLETE_CONSUMED','metadata_exact_parity':True,
            'test_cases_passed':6,'counters':dict.fromkeys(('cuda_initializations','fit_calls','native_model_calls','optimizer_steps','payload_loads'),0)}
        if item==launch['qualification_design']:return {'schema_version':'gx1_order_supported_qualification_design_v1',
            'status':'DESIGN_ONLY_NO_EXECUTION_AUTHORITY','seed':prefix.SEED,'prefix_rows':authority.BPS_FOLD['prefix_rows'],
            'batch_size':16,'test_data_used':False,**{key:ref(key) for key in ('prior_failure_review','prior_support_report','prior_support_run_result','prior_support_root_review')}}
        return {}
    monkeypatch.setattr(common,'_read',read)
    monkeypatch.setattr(prefix,'_launch_metadata',lambda *args:{'info':{'metadata':{}},
        'ancestor_plan':{'payloads':{}},'dense_target':launch['dense_target']})
    monkeypatch.setattr(common,'_freeze',lambda *args:None)
    monkeypatch.setattr(common,'_install_read_guard',lambda *args,**kwargs:[])
    monkeypatch.setattr(common,'_io',lambda:{'rchar':0})
    monkeypatch.setattr(common,'_deny_legacy_work',lambda *args:None)
    def fail(*args,**kwargs):raise RuntimeError('synthetic blocked batch')
    monkeypatch.setattr(prefix,'_prepare',fail)
    published={}
    def publish(path,value,counters,limit):
        published[Path(path).name]=value;return {'path':str(path),'sha256':'d'*64}
    monkeypatch.setattr(common,'_json',publish)
    monkeypatch.setattr(authority,'build_research_cursor',lambda **kwargs:{'outcome':kwargs['outcome']})
    monkeypatch.setattr(authority,'build_research_progress',lambda **kwargs:{'outcome':kwargs['outcome'],'terminal':True})
    monkeypatch.setattr(authority,'require_research_progress',lambda *args,**kwargs:None)
    with pytest.raises(RuntimeError,match='NATIVE_DENSE_QUALIFICATION_FAILED:RuntimeError:synthetic blocked batch'):
        worker.run_dense_qualification(policy_path=tmp_path/'policy',policy_file_sha256='0'*64,progress_path=tmp_path/'progress.json')
    assert published['RESULT.json']['decision']=='FAILED' and published['RESULT.json']['counts']['optimizer_updates']==0
    assert published['progress.json']=={'outcome':'FAILED','terminal':True}


def test_synthetic_private_control_flow_completes_fixed_resume(monkeypatch,tmp_path):
    from copy import deepcopy
    from pathlib import Path
    import torch
    from gx1.contracts import entry_model_native_train_recipe_v1 as recipes
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix_authority
    from gx1.contracts import entry_native_direction_research_state_v1 as state_owner
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as prefix
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    def ref(name):return {'path':str(tmp_path/name),'sha256':'a'*64}
    recipe=recipes.native_direction_research_recipe(batch_size=16,max_updates=2,dense_minutes=True)
    launch={'run_scope':authority.DENSE_PRIVATE_SCOPE,'numerical_recipe':recipe,'execution_binding_path':str(tmp_path/'binding'),
        'qualification_design':ref('design'),'gc_layout_metadata_parity':ref('gc'),'dense_source_transition_proof':ref('proof'),
        'dense_target':ref('targets'),'fold':authority.BPS_FOLD,'runtime_root':str(tmp_path/'runtime'),
        'max_published_json_bytes':33554432,'max_checkpoint_bytes':1073741824}
    window={'recipe':ref('recipe'),'output_directory':str(tmp_path/'output'),'campaign_cursor_path':str(tmp_path/'cursor')}
    plan={'plan_sha256':'b'*64};invocation={'batch_size':16,'invocation_sha256':'c'*64}
    prior={key:[1]*60 for key in ('eligible_up','eligible_down')}
    prior.update({key:[0.5]*60 for key in ('probability_up','logit_up')})
    old_prior={key:[value[i] for i in (0,4,14)] for key,value in prior.items()}
    monkeypatch.setattr(capped,'require_guarded_cuda_trainer_execution',lambda:{})
    monkeypatch.setattr(authority,'require_native_direction_research_context',lambda *args:(window,plan,invocation,None,launch))
    monkeypatch.setattr(common,'_hash_file',lambda path:ref('binding'))
    def read(item,metadata):
        if item==ref('binding'):return {'plan':ref('plan')}
        if item==common.INPUT_DESIGN:return {'budget':{'max_rchar_bytes':137438953472}}
        if item==prefix_authority.ORDER_ORIGIN['initial_measurement']:return {'own_prefix_prior':old_prior}
        if item==launch['gc_layout_metadata_parity']:return {'status':'COMPLETE_CONSUMED','metadata_exact_parity':True,
            'test_cases_passed':6,'counters':dict.fromkeys(('cuda_initializations','fit_calls','native_model_calls','optimizer_steps','payload_loads'),0)}
        if item==launch['qualification_design']:return {'schema_version':'gx1_order_supported_qualification_design_v1',
            'status':'DESIGN_ONLY_NO_EXECUTION_AUTHORITY','seed':prefix.SEED,'prefix_rows':authority.BPS_FOLD['prefix_rows'],
            'batch_size':16,'test_data_used':False,'parents_sorted':list(range(32)),
            **{key:ref(key) for key in ('prior_failure_review','prior_support_report','prior_support_run_result','prior_support_root_review')}}
        return {}
    monkeypatch.setattr(common,'_read',read)
    context={'info':{'metadata':{}},'ancestor_plan':{'payloads':{}},'dense_target':launch['dense_target']}
    monkeypatch.setattr(prefix,'_launch_metadata',lambda *args:context)
    monkeypatch.setattr(common,'_freeze',lambda *args:None)
    monkeypatch.setattr(common,'_install_read_guard',lambda *args,**kwargs:[])
    monkeypatch.setattr(common,'_io',lambda:{'rchar':0})
    monkeypatch.setattr(common,'_deny_legacy_work',lambda *args:None)
    parents=np.arange(300,dtype=np.int64);prepared={'arrays':{'parent_rows':parents},'gc':{'layout':{}}}
    monkeypatch.setattr(prefix,'_prepare',lambda *args,**kwargs:(prepared,{}, {'synthetic':True},{},parents))
    monkeypatch.setattr(prefix,'_prefix_prior',lambda *args:prior)
    monkeypatch.setattr(prefix,'_fixed_order',lambda *args:parents[:256].copy())
    monkeypatch.setattr(prefix,'_batch',lambda *args:({'features':{}},{'synthetic':1}))
    monkeypatch.setattr(common,'_gc_support',lambda *args,**kwargs:{'M5':{'family':{'raw_nonzero_values_per_parent':[1]}}})
    monkeypatch.setattr(prefix,'_device_batch',lambda *args:({'x':1},{'target':1}))
    class FakeModel:
        def __init__(self):self.number=0;self.feature_view_identity={'synthetic':True}
        def named_parameters(self):return []
    def fresh(_context,_view,_recipe,_device,counters,**kwargs):
        for k in ('model_constructions','model_construction_returns','optimizer_constructions'):counters[k]+=1
        return FakeModel(),object(),{}
    monkeypatch.setattr(common,'_fresh',fresh)
    def capture(model,optimizer,**kwargs):
        return {'binding':{},'model_state':model.number,'buffer_state':{},'optimizer_state':model.number,
            'rng_state':0,'module_training':{},'contract_sha256':str(model.number)}
    monkeypatch.setattr(state_owner,'capture_native_direction_research_state',capture)
    def evaluate(*args,**kwargs):
        counters=kwargs['counters'];counters['forward_calls']=counters['forward_returns']=17
        return {'initial_parity_passed':True,'arrays':{'logit_up':np.zeros((256,60),np.float32)}},256
    monkeypatch.setattr(prefix,'_evaluate',evaluate)
    monkeypatch.setattr(prefix,'_save_predictions',lambda *args,**kwargs:ref('prediction'))
    saved={}
    def save(path,state,*,counters,**kwargs):
        item=ref(Path(path).name);saved[item['path']]=deepcopy(state)
        counters['staged_state_load_calls']+=1;counters['staged_state_load_returns']+=1
        return item
    monkeypatch.setattr(common,'_save_state',save)
    def load(item,*,counters,**kwargs):
        counters['resume_state_load_calls']+=1;counters['resume_state_load_returns']+=1
        return deepcopy(saved[item['path']])
    monkeypatch.setattr(common,'_load_state',load)
    def restore(saved,model,optimizer,**kwargs):
        model.number=saved['model_state'];return {'cursor':16,'optimizer_updates':1}
    monkeypatch.setattr(state_owner,'restore_native_direction_research_state',restore)
    def step(model,optimizer,features,target,*,counters,**kwargs):
        model.number+=1
        for key in ('forwards','forward_returns','backwards','backward_returns','optimizer_step_calls','optimizer_updates'):counters[key]+=1
        return {'gradient_nonzero_counts':{'chart_local_weight_nonzero':1,'chart_mtf_weight_nonzero':1}}
    monkeypatch.setattr(common,'_step',step);monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    published={}
    def publish(path,value,counters,limit):published[Path(path).name]=value;return ref(Path(path).name)
    monkeypatch.setattr(common,'_json',publish)
    monkeypatch.setattr(authority,'build_research_cursor',lambda **kwargs:{'outcome':kwargs['outcome']})
    monkeypatch.setattr(authority,'build_research_progress',lambda **kwargs:{'outcome':kwargs['outcome'],'terminal':True})
    monkeypatch.setattr(authority,'require_research_progress',lambda *args,**kwargs:None)
    result=worker.run_dense_qualification(policy_path=tmp_path/'policy',policy_file_sha256='0'*64,progress_path=tmp_path/'progress.json')
    assert result=={'outcome':'COMPLETE','terminal':True}
    assert published['RESULT.json']['decision']=='PASS_DENSE_NATIVE_MECHANICS'
    assert published['RESULT.json']['bit_exact_resume'] and published['RESULT.json']['physical_optimizer_updates']==4
