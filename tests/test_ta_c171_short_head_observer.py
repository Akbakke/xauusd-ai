"""Synthetic boundaries only; no GX1 model, data, checkpoint or fit."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from gx1.audit import observe_market_state_ta_c171_short_heads_v1 as worker


SHA='a'*64
COMMIT='b'*40


def arrays(rows=16, heads=True):
    result={k:np.zeros(shape,dtype) for k,(shape,dtype) in worker._layout(rows,np=np).items()}
    result['parents']=np.arange(rows,dtype=np.int64)
    result['issues']=np.arange(rows,dtype=np.int64)*300_000_000_000
    result['state_query'][:]=[1,2,3,4,5]
    result['future_query'][:]=[1,3,12,48,288]
    for task in ('state','future'):
        result[task+'_raw'][:]=np.arange(rows*10,dtype=np.float32).reshape(rows,5,2)/100
        result[task+'_representation'][:]=np.arange(rows*128,dtype=np.float32).reshape(rows,128)/100
        result[task+'_allowed'][:]=4
        result[task+'_eligible'][:]=True
        result[task+'_mature'][:]=True
    return result if heads else {k:v for k,v in result.items() if k in worker.BASE_KEYS}


def plan():
    return dict(schema_version=worker.SCHEMA,stage=worker.STAGE,source_repo=str(worker.REPO),
        source_base_commit=COMMIT,worker={'path':str(worker.WORKER),'sha256':SHA},
        dispatch={'path':str(worker.REPO/worker.cache.DISPATCH),'sha256':SHA},
        cap_runner={'path':str(worker.REPO/worker.cache.CAP_RUNNER),'sha256':SHA},
        source_owners={},inputs=deepcopy(worker.INPUTS),**deepcopy(worker.BOUNDS))


def test_layout_is_only_two_heads_plus_original_panel():
    value=arrays(256)
    assert set(value)==set(worker.BASE_KEYS)|set(worker.HEAD_KEYS)
    assert value['state_representation'].shape==(256,128)
    assert value['future_representation'].shape==(256,128)
    assert sum(x.nbytes for x in value.values())<2*1024**2
    worker._require_arrays(value,256,np=np)


@pytest.mark.parametrize('rows',[0,1,15,17,257,272,True])
def test_only_intact_bounded_groups(rows):
    with pytest.raises(RuntimeError,match='ROWS'):worker._layout(rows,np=np)


def test_exact_parity_preserves_full_axes_not_only_logits():
    actual=arrays();baseline={k:actual[k].copy() for k in worker.BASE_KEYS}
    worker._require_parity(actual,baseline,1.25,1.25,np=np)
    actual['state_representation']+=17  # No old representation is claimed as a baseline.
    worker._require_parity(actual,baseline,1.25,1.25,np=np)
    actual['parents']=actual['parents'][::-1].copy()
    with pytest.raises(RuntimeError,match='INITIAL_ARRAY_PARITY'):
        worker._require_parity(actual,baseline,1.25,1.25,np=np)


@pytest.mark.parametrize('key',['state_query','future_query','state_raw','future_allowed','state_eligible','future_mature'])
def test_each_original_kind_is_compared_bitexact(key):
    actual=arrays();baseline={k:actual[k].copy() for k in worker.BASE_KEYS}
    value=actual[key].reshape(-1)
    value[0]=not value[0] if value.dtype==np.bool_ else value[0]+1
    with pytest.raises(RuntimeError,match='INITIAL_ARRAY_PARITY'):
        worker._require_parity(actual,baseline,1.25,1.25,np=np)


def test_signed_zero_and_loss_are_not_approximately_equal():
    actual=arrays();baseline={k:actual[k].copy() for k in worker.BASE_KEYS}
    actual['state_raw'][0,0,0]=-0.
    with pytest.raises(RuntimeError,match='INITIAL_ARRAY_PARITY'):
        worker._require_parity(actual,baseline,1.25,1.25,np=np)
    actual['state_raw'][0,0,0]=0.
    with pytest.raises(RuntimeError,match='INITIAL_LOSS_PARITY'):
        worker._require_parity(actual,baseline,1.25,np.nextafter(1.25,2.).item(),np=np)


@pytest.mark.parametrize('change', ['shape','dtype','nan','field'])
def test_invalid_capture_layout_rejected(change):
    value=arrays()
    if change=='shape':value['state_representation']=value['state_representation'][:,:127]
    if change=='dtype':value['future_representation']=value['future_representation'].astype(np.float64)
    if change=='nan':value['future_representation'][0,0]=np.nan
    if change=='field':value['unexpected']=np.zeros(16)
    with pytest.raises(RuntimeError,match='ARRAY'):
        worker._require_arrays(value,16,np=np)


def test_serialized_bound_is_enforced_before_immutable_publish(monkeypatch):
    value=arrays()
    monkeypatch.setitem(worker.BOUNDS,'max_cache_bytes',sum(x.nbytes for x in value.values()))
    called=[]
    monkeypatch.setattr(worker.cache,'_save_fusion_arrays',lambda *a,**k:called.append(a))
    with pytest.raises(RuntimeError,match='OUTPUT_BYTES'):
        worker._save([value],{'published_npz_bytes':0},partial=False,np=np)
    assert called==[]


def test_partial_capture_keeps_actual_batch_order_and_existing_writer(monkeypatch):
    one,two=arrays(),arrays();two['parents']+=1000
    captured={}
    def save(name,values,counts,**kwargs):
        captured.update(name=name,arrays=values,kwargs=kwargs)
        return {'path':str(worker.ROOT/name),'sha256':SHA}
    monkeypatch.setattr(worker.cache,'_save_fusion_arrays',save)
    result=worker._save([one,two],{'published_npz_bytes':0},partial=True,np=np)
    assert captured['name']=='TRAIN_FUSION_PARTIAL.npz'
    assert np.array_equal(captured['arrays']['parents'],np.r_[one['parents'],two['parents']])
    assert captured['kwargs']['root']==worker.ROOT
    assert result['sha256']==SHA


def test_exact_plan_and_paths():
    value=plan()
    assert worker._require_plan(value,worker.ROOT/'PLAN.json',worker.ROOT)==value
    for key,changed in [('max_backbone_encodings',17),('max_cache_bytes',16*1024**2),
                        ('training_enabled',True),('panel_batch_indices',list(range(16)))]:
        invalid=deepcopy(value);invalid[key]=changed
        with pytest.raises(RuntimeError,match='EXACT_PLAN'):
            worker._require_plan(invalid,worker.ROOT/'PLAN.json',worker.ROOT)
    with pytest.raises(RuntimeError,match='EXACT_PLAN'):
        worker._require_plan(value,worker.SHORT_ROOT/'PLAN.json',worker.ROOT)


def test_build_uses_readonly_current_short_qualification(monkeypatch):
    seen=[]
    monkeypatch.setattr(worker.short,'build_recipe',lambda **kw:seen.append(kw) or {'source_owners':{'x':{'path':'x','sha256':SHA}}})
    monkeypatch.setattr(worker.cache,'_ref',lambda p:{'path':str(p),'sha256':SHA})
    value=worker.build_plan(COMMIT)
    assert seen==[{'source_commit':COMMIT}]
    assert value['source_owners']=={'x':{'path':'x','sha256':SHA}}
    assert value['inputs']['short_trace']==worker.INPUTS['short_trace']
    worker._require_plan(value,worker.ROOT/'PLAN.json',worker.ROOT)


def test_preflight_reports_zero_payload_decodes_and_claims(monkeypatch):
    seen=[]
    monkeypatch.setattr(worker,'_inspect',lambda *args,**kw:seen.append((args,kw)) or {'plan_ref':{'path':'p','sha256':SHA}})
    result=worker.preflight('p',SHA,'out')
    assert seen==[(('p',SHA,'out'),{'execute':False})]
    assert all(result[k]==0 for k in ('array_deserializations','checkpoint_loads','model_constructions','claims','optimizer_updates'))


def test_qualification_failure_precedes_cap_import_constructor_and_claim(monkeypatch):
    events=[]
    def reject(*args,**kwargs):
        events.append(('inspect',kwargs));raise RuntimeError('synthetic qualification failure')
    monkeypatch.setattr(worker,'_inspect',reject)
    monkeypatch.setattr(worker,'_runtime_cap',lambda:pytest.fail('must not reach runtime'))
    monkeypatch.setattr(worker.cache,'_publish',lambda *a,**k:pytest.fail('must not claim'))
    with pytest.raises(RuntimeError,match='synthetic qualification failure'):
        worker.run(worker.ROOT/'PLAN.json',SHA,worker.ROOT)
    assert events==[('inspect',{'execute':True})]


def test_wrong_device_precedes_all_qualification(monkeypatch):
    monkeypatch.setattr(worker,'_inspect',lambda *a,**k:pytest.fail('must fail before inspect'))
    with pytest.raises(RuntimeError,match='CUDA_ONLY'):
        worker.run(worker.ROOT/'PLAN.json',SHA,worker.ROOT,device='cpu')
