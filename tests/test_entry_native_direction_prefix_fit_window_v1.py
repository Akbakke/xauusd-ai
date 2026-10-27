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
