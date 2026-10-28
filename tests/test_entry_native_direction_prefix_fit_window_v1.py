"""Synthetic275 boundaries only: no native constructor/F/B/update or data load.

The evaluation spy uses two tiny CPU Linear modules (three toy forwards total,
zero Torch backwards/steps). The train-branch spy has one Python fake backward
and optimizer return. Actual270 state mathematics and273 CUDA resume are reused,
not requalified by these unit tests. All arrays below are synthetic.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gx1.contracts import gx1_capped_execution_v1 as capped
from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as authority
from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as runner
from gx1.scripts import run_entry_native_direction_research_window_v1 as common


def test_direct_prefix_entry_requires_cuda_guard_before_context(monkeypatch,tmp_path):
    calls=[]
    def rejected():calls.append('guard');raise RuntimeError('synthetic guard rejection')
    monkeypatch.setattr(capped,'require_guarded_cuda_trainer_execution',rejected)
    monkeypatch.setattr(authority,'require_native_direction_prefix_fit_context',lambda *args:calls.append('context'))
    with pytest.raises(RuntimeError,match='guard rejection'):
        runner.run_native_direction_prefix_fit_window(policy_path=tmp_path/'missing',policy_file_sha256='0'*64,progress_path=tmp_path/'progress')
    assert calls==['guard'] and list(tmp_path.iterdir())==[]


def test_prefix_context_rejection_precedes_numerical_or_output_work(monkeypatch,tmp_path):
    calls=[];monkeypatch.setattr(capped,'require_guarded_cuda_trainer_execution',lambda:calls.append('guard') or {})
    def rejected(*args):calls.append('context');raise RuntimeError('synthetic context rejection')
    monkeypatch.setattr(authority,'require_native_direction_prefix_fit_context',rejected)
    monkeypatch.setattr(runner,'_metadata',lambda *args:calls.append('metadata'))
    with pytest.raises(RuntimeError,match='context rejection'):
        runner.run_native_direction_prefix_fit_window(policy_path=tmp_path/'missing',policy_file_sha256='0'*64,progress_path=tmp_path/'progress')
    assert calls==['guard','context'] and list(tmp_path.iterdir())==[]


def test_fixed_mt19937_order_consumes_all_parents_and_preserves_global_rng():
    parents=np.arange(10000,10300,dtype=np.int64);before=np.random.get_state()
    order=runner._fixed_order(parents,256,runner.SEED,np)
    assert np.array_equal(order,parents[:256][np.random.RandomState(runner.SEED).permutation(256)])
    assert np.array_equal(np.sort(order),parents[:256]) and np.array_equal(order,runner._fixed_order(parents,256,runner.SEED,np))
    after=np.random.get_state();assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]


def _axes_targets():
    n=118249;D=1_700_000_100_000_000_000+np.arange(n,dtype=np.int64)*300_000_000_000
    axes={'parent_rows':np.arange(n,dtype=np.int64),'bar_start_ns':D-300_000_000_000,'decision_ns':D}
    target={'terminal_time_ns':D[:,None]+np.asarray([1,5,15],np.int64)[None,:]*runner.MINUTE_NS,
        'observed':np.ones((n,3),np.bool_),'sign':np.ones((n,3),np.int8)}
    fold={'prefix_rows':20,'oof_start_row':20,'oof_end_row':30,'train_cutoff_ns':int(D[20]),'oof_cutoff_ns':int(D[30])}
    return axes,target,fold


def test_full_axes_guard_keeps_ties_missing_and_immature_parents():
    axes,target,fold=_axes_targets();target['observed'][1,0]=False;target['sign'][1,0]=0;target['sign'][2,1]=0
    runner._qualify_axes_targets(axes,target,fold,np)
    assert len(axes['parent_rows'])==118249 and target['terminal_time_ns'][19,2]>fold['train_cutoff_ns']


@pytest.mark.parametrize('fault',['clock','fold','terminal','sign','missing_storage','dtype'])
def test_axes_target_preflight_rejects_clock_domain_and_fold_corruption(fault):
    axes,target,fold=_axes_targets()
    if fault=='clock':axes['decision_ns'][0]+=1
    elif fault=='fold':fold['prefix_rows']+=1
    elif fault=='terminal':target['terminal_time_ns'][0,2]+=1
    elif fault=='sign':target['sign'][0,0]=2
    elif fault=='missing_storage':target['observed'][0,0]=False
    else:target['sign']=target['sign'].astype(np.int64)
    with pytest.raises(RuntimeError):runner._qualify_axes_targets(axes,target,fold,np)


@pytest.mark.parametrize('eligible',[False,True])
def test_masked_train_branch_empty_means_no_backward_or_optimizer(monkeypatch,eligible):
    from gx1.contracts import entry_native_direction_research_v1 as objective
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    calls=[]
    class Loss:
        def backward(self):calls.append('backward')
        def detach(self):return .25
    model=SimpleNamespace(training=True,modules=lambda:[SimpleNamespace(training=True)],parameters=lambda:[])
    monkeypatch.setattr(runner,'_forward',lambda *args,**kwargs:calls.append('forward') or object())
    monkeypatch.setattr(objective,'native_direction_binary_loss',lambda *args,**kwargs:{'counts':{'eligible_cells':int(eligible)},
        'per_horizon_counts':[],'update_allowed':eligible,'loss':Loss() if eligible else None})
    monkeypatch.setattr(trainer,'_optimizer_step_with_finite_gradients',lambda **kwargs:calls.append('optimizer') or 1.)
    counters={k:0 for k in ('empty_batches','backward_calls','backward_returns','optimizer_step_calls','optimizer_step_returns')}
    record,updated=runner._train_batch(model,object(),{}, {},cutoff=1,counters=counters,ceiling=1,torch=torch,diagnostics=False)
    assert updated is eligible and record['eligible_cells']==int(eligible)
    assert calls==(['forward','backward','optimizer'] if eligible else ['forward'])
    assert counters['empty_batches']==int(not eligible) and counters['optimizer_step_returns']==int(eligible)


def test_prediction_archive_roundtrip_and_wrong_staged_content_rejected(tmp_path,monkeypatch):
    from gx1.scripts.run_unified_exit_random_access_full_train_v1 import _dynamic_m5_technical_tree_sha256 as digest
    arrays={'parent_rows':np.asarray([4,1],np.int64),'logit_up':np.ones((2,3),np.float32),'observed':np.ones((2,3),np.bool_)}
    path=tmp_path/'PREDICTIONS.npz';ref=runner._save_predictions(path,arrays,np=np,common=common,tree_sha=digest)
    assert ref==common._hash_file(path) and path.stat().st_mode&0o222==0
    original=np.savez
    monkeypatch.setattr(np,'savez',lambda handle,**values:original(handle,**{**values,'logit_up':values['logit_up']+1}))
    with pytest.raises(RuntimeError,match='STAGED_ROUNDTRIP'):
        runner._save_predictions(tmp_path/'BAD.npz',arrays,np=np,common=common,tree_sha=digest)
    assert not (tmp_path/'BAD.npz').exists() and not list(tmp_path.glob('.BAD.npz.*'))


@pytest.mark.parametrize('used',[100,200])
def test_prefix_checkpoint_total_budget_precedes_publication(tmp_path,used):
    (tmp_path/'old.pt').write_bytes(b'x'*used);calls=[]
    state_owner=SimpleNamespace(capture_native_direction_research_state=lambda *args,**kwargs:{'contract_sha256':'a'*64})
    def save(path,state,**kwargs):calls.append(kwargs['maximum']);return {'path':str(path),'sha256':'b'*64}
    stub=SimpleNamespace(_stat=common._stat,_save_state=save)
    args=dict(context={},recipe={},order=None,parent_cursor=0,updates=0,state_directory=tmp_path,name='new',
        launch={'max_checkpoint_total_bytes':200,'max_checkpoint_bytes':150},counters={},torch=None,device=None,common=stub,state_owner=state_owner)
    model=SimpleNamespace(feature_view_identity={})
    if used==200:
        with pytest.raises(RuntimeError,match='TOTAL_BUDGET'):runner._save_snapshot(model,object(),**args)
        assert calls==[]
    else:runner._save_snapshot(model,object(),**args);assert calls==[100]


def test_latest_cursor_update_preserves_immutable_original(tmp_path):
    output=tmp_path/'snapshots';output.mkdir();pointer=tmp_path/'cursor'/'RESUME_CURSOR.json';counters={'json_bytes':0}
    old={'schema_version':'synthetic','parent_cursor':16};new={**old,'parent_cursor':32}
    old_ref=common._json(output/'ORIGINAL.json',old,counters,10000)
    runner._latest_cursor(pointer,old,common=common,counters=counters,limit=10000)
    ref=runner._latest_cursor(pointer,new,common=common,counters=counters,limit=10000)
    assert ref==common._hash_file(pointer) and __import__('json').loads(pointer.read_text())==new
    assert common._hash_file(output/'ORIGINAL.json')==old_ref


@pytest.mark.parametrize('expired',[False,True])
def test_fixed_eval_preserves_order_state_and_parity_without_updates(monkeypatch,expired):
    from gx1.contracts import entry_native_direction_context_report_v1 as reporter
    from gx1.scripts.run_unified_exit_random_access_full_train_v1 import _dynamic_m5_technical_tree_sha256 as digest
    model=torch.nn.Linear(3,3);model.train();calls=[];selected=np.asarray([8,1,9,2],np.int64)
    def batch(prepared,targets,view,ids,np):
        D=np.asarray(ids*300_000_000_000+1_700_000_000_000_000_000,np.int64)
        raw={'features':{'x':np.ones((len(ids),3),np.float32)},'witness':{'parent_rows':ids.tolist(),'bar_start_ns':(D-300_000_000_000).tolist()}}
        target={'decision_ns':D,'terminal_time_ns':D[:,None]+np.asarray([1,5,15],np.int64)[None,:]*runner.MINUTE_NS,
            'observed':np.ones((len(ids),3),np.bool_),'sign':np.ones((len(ids),3),np.int8)}
        return raw,target
    def tag(features,*,context_map,witness):
        ids=np.asarray(witness['parent_rows'],np.int64);D=np.asarray(witness['bar_start_ns'],np.int64)+300_000_000_000
        return {'schema_version':'synthetic_tags','context_map_sha256':'a'*64,'feature_view_sha256':'b'*64,
            'parent_rows':ids,'decision_ns':D,'tags':{'synthetic':np.zeros(len(ids),np.int8)}}
    def forward(m,features,*,grad=False,**kwargs):
        assert not m.training;calls.append(grad)
        with torch.set_grad_enabled(grad):return m(features['x'])
    monkeypatch.setattr(runner,'BATCH_SIZE',2);monkeypatch.setattr(runner,'_batch',batch);monkeypatch.setattr(runner,'_forward',forward)
    monkeypatch.setattr(reporter,'tag_native_direction_contexts',tag)
    monkeypatch.setattr(reporter,'score_native_direction_logits',lambda logits,**kwargs:{'rows':len(logits)})
    measured,count=runner._evaluate(model,{}, {},{}, {'contract_sha256':'a'*64},selected,cutoff=2_000_000_000_000_000_000,
        kind='INITIAL_TRAIN',counters={},ceiling=10,device=torch.device('cpu'),torch=torch,np=np,tree_sha=digest,
        deadline=0 if expired else None,parity=True)
    assert model.training and all(p.grad is None for p in model.parameters())
    if expired:assert measured is None and count==0 and calls==[]
    else:
        assert count==4 and calls==[False,True,False] and measured['initial_parity_passed']
        assert np.array_equal(measured['arrays']['parent_rows'],selected)
        assert measured['context_tags']['feature_view_sha256']=='b'*64


@pytest.mark.parametrize('signed',[False,True])
def test_launch_metadata_reuses_bound_gc_proof_and_keeps_scale_metadata(signed,monkeypatch):
    launch={};proof_ref={'path':'proof.json','sha256':'1'*64};gc_ref={'path':'gc.json','sha256':'2'*64}
    scale_ref={'path':'scale.json','sha256':'3'*64};scale={'synthetic_own_prefix_scale':True};seen=[]
    if signed:launch.update(native_bps_qualification=proof_ref,gc_layout_metadata_parity=gc_ref)
    context={'info':{'metadata':{}}}
    def metadata(actual,common,*,gc_layout_metadata_parity):
        assert actual is launch;seen.append(gc_layout_metadata_parity);return context
    def read(ref,bookends):
        bookends[ref['path']]=ref
        return {'own_prefix_bps_scale':scale_ref} if ref==proof_ref else (scale if ref==scale_ref else {})
    monkeypatch.setattr(runner,'_metadata',metadata)
    assert runner._launch_metadata(launch,SimpleNamespace(_read=read)) is context
    assert seen==[gc_ref if signed else None]
    if signed:
        assert context['bps_qualified_scale']==scale and context['bps_qualified_scale_ref']==scale_ref
        assert set(context['info']['metadata'])=={'proof.json','gc.json','scale.json'}
    else:assert context=={'info':{'metadata':{}}}


@pytest.mark.parametrize('case',['default','same','different','missing_view_scale','unqualified'])
def test_recomputed_bps_scale_must_match_qualified_prefix_before_training(case):
    scale={'synthetic_own_prefix_scale':True};ref={'path':'scale.json','sha256':'3'*64}
    context={} if case in ('default','unqualified') else {'bps_qualified_scale':scale,'bps_qualified_scale_ref':ref}
    view={} if case in ('default','missing_view_scale') else {'bps_scale':scale if case!='different' else {}}
    if case=='default':assert runner._qualified_bps_scale(context,view) is None
    elif case=='same':assert runner._qualified_bps_scale(context,view)==ref
    else:
        with pytest.raises(RuntimeError):runner._qualified_bps_scale(context,view)


@pytest.mark.parametrize('signed', [False, True])
def test_independent_extraction_optin_does_not_change_training_metadata_route(monkeypatch, signed):
    # Only F's guarded metadata call may supply the new optional argument.
    launch = {'independent_completion': {'synthetic': 'not a training opt-in'}}
    if signed: launch['gc_layout_metadata_parity'] = {'path': 'gc.json', 'sha256': '2' * 64}
    seen = []
    def metadata(actual, common, *, gc_layout_metadata_parity=None, independent_completion=None):
        seen.append((actual, gc_layout_metadata_parity, independent_completion))
        return {'info': {'metadata': {}}}
    monkeypatch.setattr(runner, '_metadata', metadata)
    runner._launch_metadata(launch, object())
    assert seen == [(launch, launch.get('gc_layout_metadata_parity'), None)]


def test_continuation_repeats_exact_order_and_preserves_partial_pass_boundaries():
    n=34;parent_order=list(range(n-1,-1,-1));cursor=n;presentations=[];lengths=[]
    while cursor<3*n:
        left,right,cursor=runner._next_training_parent_range(cursor,n,3)
        presentations.extend(parent_order[left:right]);lengths.append(right-left)
    assert presentations==parent_order*2
    assert lengths==[16,16,2,16,16,2]
    assert cursor==3*n
    with pytest.raises(RuntimeError,match='TRAIN_PARENT_BOUNDARY'):
        runner._next_training_parent_range(n+1,n,3)
    with pytest.raises(RuntimeError,match='TRAIN_PARENT_RANGE'):
        runner._next_training_parent_range(3*n,n,3)


def test_read_guard_allows_only_the_two_bound_original_checkpoints_as_readonly(monkeypatch):
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as authority
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    hooks=[];monkeypatch.setattr(common.sys,'addaudithook',hooks.append)
    context={'ancestor_plan':{'payloads':{},'native_cache_dir':'/synthetic/cache',
        'gc_surfaces':{'path':'/synthetic/gc/surface'}},
        'info':{'metadata':{},'audit':{'manifest_path':'/synthetic/audit'}}}
    terminal=authority.CONTINUATION_ORIGIN['terminal_state']['path']
    initial=authority.CONTINUATION_ORIGIN['initial_state']['path']
    denied=common._install_read_guard(context,'/synthetic/output','/synthetic/runtime',extra_read_paths=(terminal,initial))
    hooks[0]('open',(terminal,'rb',0));hooks[0]('open',(initial,'rb',0))
    with pytest.raises(RuntimeError,match='ORIGINAL_WRITE'):
        hooks[0]('open',(initial,'wb',__import__('os').O_WRONLY))
    with pytest.raises(RuntimeError,match='UNBOUND_OPEN'):
        hooks[0]('open',('/home/andre2/GX1_RUNS/unbound.pt','r',0))
    assert denied==[initial,'/home/andre2/GX1_RUNS/unbound.pt']
    for invalid in ((terminal,), (initial,terminal), (terminal,'/home/andre2/GX1_RUNS/unbound.pt')):
        with pytest.raises(RuntimeError,match='EXTRA_READ_PATHS'):
            common._install_read_guard(context,'/synthetic/output','/synthetic/runtime',extra_read_paths=invalid)


def test_continuation_binds_qualification_and_original_measurement_to_freeze(monkeypatch):
    from gx1.contracts import entry_native_direction_frozen_extraction_v1 as frozen
    proof={'path':'/synthetic/proof.json','sha256':'1'*64}
    gc={'path':'/synthetic/gc.json','sha256':'2'*64}
    terminal={'path':'/synthetic/terminal.json','sha256':'3'*64}
    original_ref={'path':'/synthetic/original.json','sha256':'4'*64}
    old_numerical={'max_updates':2976};new_numerical={'max_updates':8928}
    completion={'design':{'path':'/synthetic/design','sha256':'5'*64},'initial_parity':None}
    launch={'native_continuation_qualification':proof,'gc_layout_metadata_parity':gc,'numerical_recipe':new_numerical,
        'continuation_origin':{'terminal_train_measurement':terminal,'launch_recipe':original_ref}}
    context={'info':{'metadata':{}}};seen=[]
    def check_metadata(actual,common,**kwargs):
        assert actual is not launch and actual['numerical_recipe']==old_numerical
        assert kwargs=={'gc_layout_metadata_parity':gc,'independent_completion':completion}
        return context
    monkeypatch.setattr(runner,'_metadata',check_metadata)
    def read(ref,metadata):
        seen.append(ref);metadata[ref['path']]=ref
        if ref==frozen.CONTINUATION_PRIOR_RECIPE:return {'independent_completion':completion}
        if ref==original_ref:return {'numerical_recipe':old_numerical}
        return {}
    runner._launch_metadata(launch,SimpleNamespace(_read=read))
    assert launch['numerical_recipe']==new_numerical
    assert seen==[frozen.CONTINUATION_PRIOR_RECIPE,original_ref,gc,proof,terminal]
    assert set(context['info']['metadata'])=={item['path'] for item in seen}


def test_weighted_historical_source_transition_is_exact_and_one_component_only():
    old = {'path':'/synthetic/component.py', 'sha256':'a'*64}
    new = {'path':'/synthetic/component.py', 'sha256':'b'*64}
    other = {'path':'/synthetic/other.py', 'sha256':'c'*64}
    manifest = {'view_source_bindings':{'direction_component_source':old}}
    design = {'historical_component_transition':{'original':old, 'current':new}}
    assert runner._weighted_historical_source(old, manifest, design) == new
    assert runner._weighted_historical_source(other, manifest, design) == other
    with pytest.raises(RuntimeError, match='WEIGHT_HISTORICAL_COMPONENT'):
        runner._weighted_historical_source(old, manifest, {'historical_component_transition':{'original':other, 'current':new}})
    with pytest.raises(RuntimeError, match='WEIGHT_HISTORICAL_COMPONENT'):
        runner._weighted_historical_source(old, manifest, {'historical_component_transition':{'original':old, 'current':other}})


def test_weighted_initial_read_guard_allows_one_readonly_bound_prediction(monkeypatch):
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    hooks=[]; monkeypatch.setattr(common.sys,'addaudithook',hooks.append)
    context={'ancestor_plan':{'payloads':{},'native_cache_dir':'/synthetic/cache',
        'gc_surfaces':{'path':'/synthetic/gc/surface'}},
        'info':{'metadata':{},'audit':{'manifest_path':'/synthetic/audit'}}}
    target=authority.WEIGHT_INITIAL_PREDICTIONS['path']
    denied=common._install_read_guard(context,'/synthetic/output','/synthetic/runtime',original_initial_prediction=target)
    hooks[0]('open',(target,'rb',0))
    with pytest.raises(RuntimeError,match='ORIGINAL_WRITE'):
        hooks[0]('open',(target,'wb',__import__('os').O_WRONLY))
    with pytest.raises(RuntimeError,match='UNBOUND_OPEN'):
        hooks[0]('open',('/home/andre2/GX1_RUNS/unbound.npz','rb',0))
    with pytest.raises(RuntimeError,match='INITIAL_PREDICTION_PATH'):
        common._install_read_guard(context,'/synthetic/output','/synthetic/runtime',original_initial_prediction='/home/andre2/GX1_RUNS/unbound.npz')
    assert denied==[target,'/home/andre2/GX1_RUNS/unbound.npz']


def test_chronological_order_scope_preserves_original_shuffle_and_global_rng():
    parents=np.arange(10000,10300,dtype=np.int64);before=np.random.get_state()
    launch={'run_scope':authority.ORDER_SCOPE,'parent_order_algorithm':authority.ORDER_ALGORITHM}
    actual=runner._presentation_order(parents,256,runner.SEED,np,launch=launch)
    assert np.array_equal(actual,parents[:256])
    legacy=runner._presentation_order(parents,256,runner.SEED,np,launch={'run_scope':'single_prefix_one_pass_direction_fit'})
    assert np.array_equal(legacy,runner._fixed_order(parents,256,runner.SEED,np))
    after=np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    for bad in ({**launch,'parent_order_algorithm':'permutation'},launch):
        rows=parents.copy()
        if bad is launch:rows[5]=rows[4]
        with pytest.raises(RuntimeError,match='CHRONOLOGICAL_ORDER'):
            runner._presentation_order(rows,256,runner.SEED,np,launch=bad)


def test_chronological_initial_predictions_must_match_bound_original(tmp_path):
    arrays={'parent_rows':np.array([4,1],np.int64),'logit_up':np.array([[.1,.2,.3],[.4,.5,.6]],np.float32)}
    path=tmp_path/'original.npz';np.savez(path,**arrays);ref=common._hash_file(path)
    runner._require_original_initial_match({'arrays':arrays},ref,common=common,np=np)
    changed={**arrays,'logit_up':arrays['logit_up'].copy()};changed['logit_up'][0,1]+=1
    with pytest.raises(RuntimeError,match='ORDER_INITIAL_PREDICTION:logit_up'):
        runner._require_original_initial_match({'arrays':changed},ref,common=common,np=np)
    changed={**arrays,'parent_rows':arrays['parent_rows'][::-1].copy()}
    with pytest.raises(RuntimeError,match='ORDER_INITIAL_PREDICTION:parent_rows'):
        runner._require_original_initial_match({'arrays':changed},ref,common=common,np=np)
    zero={'parent_rows':np.array([0],np.int64),'logit_up':np.array([[0.]],np.float32)}
    zero_path=tmp_path/'zero.npz';np.savez(zero_path,**zero);zero_ref=common._hash_file(zero_path)
    opposite_zero={**zero,'logit_up':np.array([[-0.]],np.float32)}
    with pytest.raises(RuntimeError,match='ORDER_INITIAL_PREDICTION:logit_up'):
        runner._require_original_initial_match({'arrays':opposite_zero},zero_ref,common=common,np=np)
    np.savez(path,**arrays)
    with pytest.raises(RuntimeError,match='ORDER_INITIAL_PREDICTION_SHA'):
        runner._require_original_initial_match({'arrays':arrays},{**ref,'sha256':'0'*64},common=common,np=np)


def test_dense_axes_validate_all_sixty_maturities_and_exact_clocks():
    axes, _old, fold = _axes_targets()
    n = len(axes['parent_rows']); decision = axes['decision_ns']
    target = {'terminal_time_ns': decision[:, None] + np.arange(1, 61, dtype=np.int64)[None, :] * runner.MINUTE_NS,
        'observed': np.ones((n, 60), np.bool_), 'sign': np.ones((n, 60), np.int8)}
    runner._qualify_axes_targets(axes, target, fold, np, dense_minutes=True)
    assert target['terminal_time_ns'][19, 59] > fold['train_cutoff_ns']
    target['terminal_time_ns'][0, 59] += 1
    with pytest.raises(RuntimeError, match='TARGET_CLOCK_AND_DOMAIN'):
        runner._qualify_axes_targets(axes, target, fold, np, dense_minutes=True)


def test_dense_target_guard_allows_only_bound_archive_read(monkeypatch):
    import os
    hooks = []; monkeypatch.setattr(common.sys, 'addaudithook', hooks.append)
    target = {'path':'/home/andre2/GX1_RUNS/synthetic/DENSE_TARGETS.npz','sha256':'a'*64}
    context = {'ancestor_plan':{'payloads':{},'native_cache_dir':'/synthetic/cache',
        'gc_surfaces':{'path':'/synthetic/gc/surface'}},
        'info':{'metadata':{},'audit':{'manifest_path':'/synthetic/audit'}},'dense_target':target}
    denied = common._install_read_guard(context,'/synthetic/output','/synthetic/runtime',dense_target=target)
    hooks[0]('open',(target['path'],'rb',0))
    with pytest.raises(RuntimeError,match='ORIGINAL_WRITE'):
        hooks[0]('open',(target['path'],'wb',os.O_WRONLY))
    with pytest.raises(RuntimeError,match='UNBOUND_OPEN'):
        hooks[0]('open',('/home/andre2/GX1_RUNS/synthetic/unbound.npz','rb',0))
    with pytest.raises(RuntimeError,match='DENSE_TARGET_BINDING'):
        common._install_read_guard(context,'/synthetic/output','/synthetic/runtime',dense_target={**target,'sha256':'b'*64})
    assert denied == [target['path'],'/home/andre2/GX1_RUNS/synthetic/unbound.npz']


def test_dense_prepare_joins_only_bound_synthetic_targets_and_current_sources(monkeypatch,tmp_path):
    from gx1.contracts import entry_native_direction_dense_data_v1 as bind
    from gx1.contracts import entry_native_direction_feature_view_v1 as view_owner
    from gx1.contracts import entry_native_direction_context_report_v1 as report_owner
    from gx1.scripts import materialize_entry_direction_dense_v1 as materialize
    from tests.test_materialize_entry_direction_dense_v1 import fixture as synthetic
    source, legacy, cutoff, _ = synthetic()
    dense = materialize.build_dense_targets(source, legacy, cutoff_ns=cutoff)
    old_path = tmp_path/'old.npz'; dense_path = tmp_path/'dense.npz'
    legacy['elapsed_wall_seconds'] = np.tile(np.array([60,300,900],np.int64),(2,1))
    np.savez(old_path, **{key:legacy[key] for key in runner.PRICE_MEMBERS})
    np.savez(dense_path, **dense)
    ref = common._hash_file(str(dense_path))
    arrays = {key:legacy[key].copy() for key in ('parent_rows','bar_start_ns','decision_ns')}
    prepared = {'arrays':arrays,'gc':{'layout':{},'transition':{}},
        'entry_sequence_source':{'times_ns':np.arange(3,dtype=np.int64),'sample_positions':np.arange(2,dtype=np.int64)}}
    physical = SimpleNamespace(prepare_inputs=lambda *_:prepared, release_pages=lambda *_:None)
    source_refs = {key:{'path':'/synthetic/'+key,'sha256':'a'*64} for key in view_owner.SOURCE_REF_KEYS}
    manifest = {'targets':{'path':str(old_path)}, 'view_source_bindings':source_refs,
        'prefix_normalization':{'output':{'path':'/synthetic/prefix','sha256':'a'*64}},
        'constructor_metadata':{'path':'/synthetic/constructor','sha256':'a'*64}}
    context = {'physical':physical,'ancestor':object(),'ancestor_plan':{},'info':{'metadata':{}},
        'manifest':manifest,'fitted':{'base_normalization':{'surfaces':{'mtf_m5':{'field_names':[]}}}},
        'constructor':{},'dense_target':ref}
    owners = {p:{'path':'/synthetic/'+p,'sha256':'b'*64} for p in (
        'gx1/contracts/entry_native_direction_dense_v1.py','gx1/features/entry_specialist_feature_groups_v1.py',
        'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py')}
    launch = {'run_scope':authority.DENSE_SCOPE,'dense_target':ref,'fold':{'prefix_rows':2},'source_owners':owners}
    monkeypatch.setattr(common,'_read',lambda *_:{})
    calls = []; monkeypatch.setattr(runner,'_qualify_axes_targets',lambda *a,**k:calls.append(k))
    monkeypatch.setattr(runner,'_presentation_order',lambda parents,*a,**k:parents.copy())
    monkeypatch.setattr(view_owner,'bind_native_direction_feature_view',lambda **kwargs:kwargs)
    monkeypatch.setattr(report_owner,'native_direction_context_map',lambda **kwargs:{})
    got, target, view, _map, order = runner._prepare(context,launch,np,dense_minutes=True)
    assert got is prepared and target['sign'].shape == (2,60) and all(not v.flags.writeable for v in target.values())
    assert calls[0]['dense_minutes'] is True and view['dense_minutes'] is True
    assert view['source_bindings']['dense_profile_source'] == owners['gx1/features/entry_specialist_feature_groups_v1.py']
    assert np.array_equal(order,arrays['parent_rows'])
    with pytest.raises(RuntimeError,match='DENSE_EXACT_SCOPE'):
        runner._prepare(context,{**launch,'run_scope':'not_admitted'},np,dense_minutes=True)


def test_dense_launch_metadata_requires_qualified_anchors_and_exact_target_binding(monkeypatch):
    proof_ref={'path':'/synthetic/transition.json','sha256':'1'*64}
    result_ref={'path':'/synthetic/result.json','sha256':'2'*64}
    target_ref={'path':'/synthetic/targets.npz','sha256':'3'*64}
    gc_ref={'path':'/synthetic/gc.json','sha256':'4'*64}
    transitions={'synthetic':'source-transition'}
    proof={'schema_version':'gx1_dense_source_transition_proof_v1',
        'status':'SOURCE_ONLY_VERIFIED_NO_DATA_OR_TRAINING','test_data_used':False,
        'genuine_training_authorized':False,'source_transitions':transitions}
    report={'status':'COMPLETE_REAL_TARGETS_ONLY','test_data_used':False,'parents':118249,
        'horizons_minutes':list(range(1,61)), 'legacy_anchor_byte_parity':{'1':True,'5':True,'15':True},
        'target':{**target_ref,'bytes':10}}
    launch={'run_scope':authority.DENSE_SCOPE,'dense_source_transition_proof':proof_ref,
        'dense_target_result':result_ref,'dense_target':target_ref,'gc_layout_metadata_parity':gc_ref}
    values={proof_ref['path']:proof,result_ref['path']:report}
    monkeypatch.setattr(common,'_read',lambda ref,metadata:values[ref['path']])
    monkeypatch.setattr(common,'_stat',lambda _path:{'st_size':10})
    monkeypatch.setattr(common,'_hash_file',lambda _path:target_ref)
    calls=[]
    def metadata(actual, _common, **kwargs):
        calls.append((actual,kwargs));return {'info':{'metadata':{}}}
    monkeypatch.setattr(runner,'_metadata',metadata)
    context=runner._launch_metadata(launch,common)
    assert context['dense_target']==target_ref and context['dense_target_stat']=={'st_size':10}
    assert calls[0][1]['dense_source_transition']==transitions
    report['legacy_anchor_byte_parity']['5']=False
    with pytest.raises(RuntimeError,match='DENSE_TARGET_RESULT'):
        runner._launch_metadata(launch,common)
    assert len(calls)==1
