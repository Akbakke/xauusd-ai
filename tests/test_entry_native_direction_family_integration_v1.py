"""Synthetic family input/scheduling checks. No genuine data or native training."""
from copy import deepcopy
import numpy as np
import pytest
from gx1.contracts import entry_direction_family_intervention_v1 as intervention
from gx1.contracts import entry_native_direction_feature_view_v1 as view_owner
from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as authority
from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as runner
from tests.test_entry_direction_family_intervention_v1 import case
from tests.test_entry_native_direction_feature_view_v1 import bound_case, _ref


def test_shared_batch_hook_preserves_reporting_and_exact_parent_targets(case, monkeypatch):
    _, contract, features = case
    original = deepcopy(features)
    monkeypatch.setattr(view_owner, 'gather_native_direction_feature_batch', lambda *a, **k: {'features':features,'witness':{'parent_rows':[5,12]}})
    prepared={'arrays':{'parent_rows':np.array([5,12]),'bar_start_ns':np.array([10,20]),'decision_ns':np.array([11,21])},
        'entry_sequence_source':{},'current':{'snap':None,'ctx_cont':None,'ctx_cat':None},'cache':{},
        'gc':{'surfaces':{},'layout':{}},'family_intervention_contract':contract}
    targets={'sign':np.array([[1,-1],[-1,1]]),'observed':np.ones((2,2),dtype=bool)}
    for arm in contract['arms']:
        ref={'arm':arm,'contract_sha256':contract['contract_sha256']}
        raw,target=runner._batch(prepared,targets,{'family_intervention':ref},np.array([5,12]),np)
        expected=intervention.apply_family_intervention_numpy(features,contract,arm=arm,expected_sha256=ref['contract_sha256'])
        assert raw['report_features'] is features
        assert np.array_equal(target['decision_ns'],[11,21])
        for key in targets: assert np.array_equal(target[key],targets[key])
        for key in features:
            assert np.array_equal(raw['features'][key],expected[key])
            assert np.array_equal(features[key],original[key])
    with pytest.raises(RuntimeError,match='FAMILY_VIEW_MISSING'):
        runner._batch(prepared,targets,{},np.array([5,12]),np)


def test_family_seal_and_baseline_reference_share_architecture(bound_case):
    args=deepcopy(bound_case['arguments'])
    prefix=args['prefix_normalization']; gc=prefix['gc_mtf_history']
    profile=view_owner.price_momentum_owner.compile_price_momentum_profile(gc)
    contract=intervention.compile_family_intervention(profile_reference={k:profile[k] for k in ('name','contract_sha256')},
        gc_mtf_history=gc,base_normalization=prefix['base_normalization'],gc_normalization=gc['normalization'])
    args['source_bindings'].update({k:_ref(k+'.py') for k in view_owner.DENSE_SOURCE_REF_KEYS|view_owner.PRICE_MOMENTUM_SOURCE_REF_KEYS|view_owner.FAMILY_SOURCE_REF_KEYS})
    args.update(dense_minutes=True,price_momentum=True)
    baseline=view_owner.bind_native_direction_feature_view(**args,family_intervention={'arm':'baseline','contract_sha256':contract['contract_sha256']})
    for arm in contract['arms']:
        ref={'arm':arm,'contract_sha256':contract['contract_sha256']}
        view=view_owner.bind_native_direction_feature_view(**args,family_intervention=ref)
        assert view_owner.family_reference_view(view)==baseline
        assert view['entry_execution_profile']==baseline['entry_execution_profile']
        assert view['research_objective']==baseline['research_objective']
        assert (view['contract_sha256']==baseline['contract_sha256']) == (arm=='baseline')
    with pytest.raises(RuntimeError):
        view_owner.bind_native_direction_feature_view(**args,family_intervention={'arm':'baseline','contract_sha256':'0'*64})
    with pytest.raises(RuntimeError):
        view_owner.bind_native_direction_feature_view(**args,family_intervention={'arm':'unknown','contract_sha256':contract['contract_sha256']})
    del args['source_bindings']['family_intervention_source']
    with pytest.raises(RuntimeError):
        view_owner.bind_native_direction_feature_view(**args,family_intervention=baseline['family_intervention'])


def test_full_train_selection_is_complete_and_deferred_after_updates():
    parents=np.arange(64890,dtype=np.int64)
    prepared={'arrays':{'parent_rows':parents}}; diagnostic=parents[:256]
    launch={'run_scope':authority.FAMILY_SIGN_SCOPE,'fold':{'prefix_rows':47602},'final_train_measurement_rows':47602}
    assert runner._final_train_selection(prepared,diagnostic,launch,1) is None
    actual=runner._final_train_selection(prepared,diagnostic,launch,0)
    assert np.array_equal(actual,parents[:47602]) and len(actual)==47602
    with pytest.raises(RuntimeError,match='FAMILY_FULL_TRAIN_SELECTION'):
        runner._final_train_selection({'arrays':{'parent_rows':parents[:256]}},diagnostic,launch,0)
    old={**launch,'run_scope':authority.M5_PRICE_FIT_SCOPE}
    assert runner._final_train_selection(prepared,diagnostic,old,1) is diagnostic


def test_family_adapter_and_power_scope_are_distinct():
    launch={'run_scope':authority.FAMILY_SIGN_SCOPE,'candidate_mode':'price_only','candidate_signed_bps':False}
    adapted=runner._fit_input_launch(launch,authority)
    assert adapted['run_scope']==authority.FAMILY_PRIVATE_ADAPTER_SCOPE
    assert launch['run_scope']==authority.FAMILY_SIGN_SCOPE
    assert authority.guarded_prefix_power_limit(launch)==250


@pytest.mark.parametrize('updates_this_window,complete_rows,expected_phase', [
    (1,None,'SCORING'), (0,None,'SCORING'), (0,47602,'COMPLETE')])
def test_actual_scoring_tail_never_repeats_optimizer_or_completes_partial_train(updates_this_window,complete_rows,expected_phase):
    # Execute the production main-loop tail with synthetic evaluator/publication
    # spies. The restored full cursor must skip the actual training while-body.
    import ast
    import inspect
    from types import SimpleNamespace
    source=ast.parse(inspect.getsource(runner.run_native_direction_prefix_fit_window))
    nodes=list(ast.walk(source))
    loop=next(n for n in nodes if isinstance(n,ast.While)
        and isinstance(n.test,ast.BoolOp) and isinstance(n.test.values[0],ast.Compare)
        and isinstance(n.test.values[0].left,ast.Name) and n.test.values[0].left.id=='parent_cursor')
    parent=next(n for n in nodes if isinstance(getattr(n,'body',None),list) and loop in n.body)
    start=parent.body.index(loop)
    stop=next(i for i in range(start+1,len(parent.body))
        if isinstance(parent.body[i],ast.Assign) and any(isinstance(t,ast.Tuple) for t in parent.body[i].targets))
    code=compile(ast.fix_missing_locations(ast.Module(body=parent.body[start:stop-1],type_ignores=[])),'<production-scoring-tail>','exec')
    parents=np.arange(64890); calls=[]; published=[]
    def evaluate(*a,**kw):
        calls.append((kw['kind'],len(a[5])))
        if kw['kind']=='FINAL_TRAIN':
            return (({'synthetic':True},complete_rows) if complete_rows else (None,16))
        return {'synthetic':True},17288
    def publish(kind,*a,**kw):
        published.append(kind); return {'path':kind,'sha256':'1'*64}
    def forbidden(*a,**kw): pytest.fail('Completed restored cursor repeated training')
    launch={'run_scope':authority.FAMILY_SIGN_SCOPE,'fold':{'prefix_rows':47602},'final_train_measurement_rows':47602}
    env={**runner.__dict__,'parent_cursor':47602,'train_passes':1,'ordered':parents[:47602],
        'phase':'SCORING','final_measurement':None,'deadline':100,'time':SimpleNamespace(monotonic=lambda:0),
        'counters':{'optimizer_step_returns':updates_this_window},'family_fit':True,
        'prepared':{'arrays':{'parent_rows':parents}},'diagnostic_ids':parents[:256],
        'launch':launch,'fold':{'prefix_rows':47602,'train_cutoff_ns':1,'oof_start_row':47602,'oof_end_row':64890,'oof_cutoff_ns':2},
        'model':None,'targets':None,'view':None,'context_map':None,'ceiling':9999,'device':None,
        'torch':None,'np':np,'tree_sha':None,'work':None,'view_identity':None,'prior':None,
        'initial_ref':None,'fresh_identity':None,'common':None,'_evaluate':evaluate,
        '_publish_measurement':publish,'_train_batch':forbidden}
    exec(code,env)
    assert env['phase']==expected_phase
    if updates_this_window:
        assert calls==[] and published==[]
    elif complete_rows is None:
        assert calls==[('FINAL_TRAIN',47602)] and published==[]
    else:
        assert calls==[('FINAL_TRAIN',47602),('OOF',17288)]
        assert published==['FINAL_TRAIN','OOF']

def test_family_dense_source_transition_reuses_only_exact_three_input_owners():
    import ast
    import inspect
    names={'owner_profile_source':'gx1/features/entry_specialist_feature_groups_v1.py',
        'native_constructor_source':'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py',
        'direction_component_source':'gx1/contracts/entry_native_direction_research_v1.py'}
    original={k:{'path':'/synthetic/'+v,'sha256':'1'*64} for k,v in names.items()}
    current={v:{'path':'/synthetic/'+v,'sha256':'2'*64} for v in names.values()}
    proof={k:{'original':original[k],'current':current[v]} for k,v in names.items()}
    tree=ast.parse(inspect.getsource(runner._metadata))
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
        and isinstance(n.test.left,ast.Name) and n.test.left.id=='dense_source_transition')
    code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<production-source-transition>','exec')
    env={**runner.__dict__,'launch':{'run_scope':authority.FAMILY_PRIVATE_ADAPTER_SCOPE,'source_owners':current},
        'manifest':{'view_source_bindings':original},'dense_source_transition':proof,'dense_transitions':{},
        'independent_design':None,'weight_design':None,'order_transition':None}
    exec(code,env)
    assert len(env['dense_transitions'])==3
    bad=deepcopy(proof);bad['owner_profile_source']['current']['sha256']='0'*64
    with pytest.raises(RuntimeError,match='DENSE_SOURCE_TRANSITION'):
        exec(code,{**env,'dense_source_transition':bad,'dense_transitions':{}})


def test_family_fullfit_initial_read_guard_binds_scope_arm_review_and_exact_ref(monkeypatch,tmp_path):
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    ref={'path':'/home/andre2/GX1_RUNS/synthetic_family/INITIAL_TRAIN.npz','sha256':'1'*64}
    report_ref={'path':'/home/andre2/GX1_RUNS/synthetic_family/INITIAL_TRAIN.json','sha256':'2'*64}
    shared={'family_arm':'momentum_flow_encoder','family_intervention_sha256':'3'*64,'test_data_used':False}
    launch={**shared,'run_scope':authority.FAMILY_SIGN_SCOPE,'decimal_initial_predictions':ref,'family_initial_report':report_ref}
    review={**shared,'schema_version':'gx1_m5_family_private_root_review_v1',
        'status':'PASS_CONSUMED_FAMILY_PRIVATE_NATIVE_MECHANICS','initial_predictions':ref,'initial_report':report_ref}
    initial={**shared,'schema_version':'gx1_native_m5_family_initial_measurement_v1','role':'INITIAL_TRAIN','predictions':ref}
    context={'ancestor_plan':{'payloads':{},'native_cache_dir':'/synthetic/cache','gc_surfaces':{'path':'/synthetic/gc/x'}},
        'info':{'metadata':{},'audit':{'manifest_path':'/synthetic/forbidden'}},
        'candidate_inputs':{'mode':'price_only'},'decimal_initial_predictions':ref,
        'family_initial_read_authority':{'launch':launch,'review':review,'initial_report':initial}}
    hooks=[];monkeypatch.setattr(common.sys,'addaudithook',hooks.append)
    common._install_read_guard(context,tmp_path/'output',tmp_path/'runtime',candidate_initial_predictions=ref)
    assert len(hooks)==1
    hooks[0]('open',(ref['path'],'r',0))
    with pytest.raises(RuntimeError,match='ORIGINAL_WRITE'):
        hooks[0]('open',(ref['path'],'w',common.os.O_WRONLY))
    wrong={**ref,'path':ref['path'].replace('synthetic_family','other_arm')}
    with pytest.raises(RuntimeError,match='UNBOUND_OPEN'):
        hooks[0]('open',(wrong['path'],'r',0))
    with pytest.raises(RuntimeError,match='CANDIDATE_INITIAL_BINDING'):
        common._install_read_guard(context,tmp_path/'output',tmp_path/'runtime',candidate_initial_predictions=wrong)
    for member,key,value in (('launch','run_scope',authority.M5_PRICE_FIT_SCOPE),
            ('review','family_arm','baseline'),('initial_report','family_intervention_sha256','4'*64)):
        broken=deepcopy(context);broken['family_initial_read_authority'][member][key]=value
        with pytest.raises(RuntimeError,match='FAMILY_INITIAL_READ_SCOPE'):
            common._install_read_guard(broken,tmp_path/'output',tmp_path/'runtime',candidate_initial_predictions=ref)
    unbound=deepcopy(context);del unbound['family_initial_read_authority']
    with pytest.raises(RuntimeError,match='CANDIDATE_INITIAL_BINDING'):
        common._install_read_guard(unbound,tmp_path/'output',tmp_path/'runtime',candidate_initial_predictions=ref)
