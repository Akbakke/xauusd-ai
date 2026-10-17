"""Synthetic contract tests only; no market-learning/edge evidence."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from gx1.contracts import gc_sequence_research_v1 as c
from gx1.models.gc_sequence_research_v1 import GCSequenceTCN,GCSequenceTransformer


def test_no_missing_time_compression_or_roll_bridge():
    t=np.array([0,1,2,4,5,6,7,8],np.int64);contract=np.array([1,1,1,1,1,1,2,2])
    rows=c.sequence_rows(t,contract,np.ones(8,bool),length=3,step_ns=1)
    assert rows.tolist()==[[0,1,2],[3,4,5]]


def test_missing_observation_poisons_every_window_that_uses_it():
    observed=np.ones(7,bool);observed[2]=False
    rows=c.sequence_rows(np.arange(7),np.ones(7,int),observed,length=3,step_ns=1)
    assert rows.tolist()==[[3,4,5],[4,5,6]]


@pytest.mark.parametrize('times',[np.array([0,0,1]),np.array([2,1,0])])
def test_nonascending_clock_rejected(times):
    with pytest.raises(ValueError,match='CLOCK'):c.sequence_rows(times,np.ones(3,int),np.ones(3,bool),length=2,step_ns=1)


def test_short_prefix_cannot_wrap_to_end():
    rows=c.sequence_rows(np.arange(3),np.ones(3,int),np.ones(3,bool),length=6,step_ns=1)
    assert rows.shape==(0,6)


def test_normalization_unique_train_rows_no_eval_refit():
    x=np.array([[0.,4],[2.,4],[4.,4],[999.,4]])
    windows=np.array([[0,1],[1,2],[2,3]])
    mean,scale,rows=c.train_normalization(x,windows,np.array([0,1]))
    np.testing.assert_array_equal(mean,[2.,4.]);assert rows.tolist()==[0,1,2]
    assert scale[1]==1 and np.all(c.standardized(x,windows,mean,scale)[:,:,1]==0)
    x[3,0]=np.nan
    with pytest.raises(ValueError,match='NO_IMPUTATION'):c.standardized(x,windows,mean,scale)


def test_purge_and_embargo_do_not_filter_evaluation_by_future_labels():
    ns=10**9;start=pd.Timestamp('2025-05-01T00:00:00Z').value
    d=start+np.array([-90000,-89900,-86399,0,300])*ns
    e=np.array([start-1,start+1,start-1,start+600*ns,start+900*ns])
    v=np.array([True,True,True,False,True])
    f={'train_start_utc_inclusive':'2024-05-01T00:00:00Z','validation_start_utc_inclusive':'2025-05-01T00:00:00Z','validation_end_utc_exclusive':'2025-08-01T00:00:00Z'}
    tr,ev=c.purged_fold(d,e,v,f,embargo_seconds=86400)
    assert tr.tolist()==[0] and ev.tolist()==[3,4]


def test_pilot_remains_closed_and_requires_exact_qualification():
    gate={'pilot_training_authorized':False,'pilot_status':'CLOSED','pilot_claimed_invocations':0,'pilot_completed_invocations':0,
        'plan_sha256':'plan','qualification_sha256':'qualification','native_training_authorized':False,'test_authorized':False}
    p={'current_work':{'gc_sequence_research_20261017':gate}}
    with pytest.raises(RuntimeError,match='CLOSED_BY_POLICY'):c.require_pilot_authority(p,'plan','qualification')
    gate.update(pilot_training_authorized=True,pilot_status='AUTHORIZED_SINGLE_USE')
    c.require_pilot_authority(p,'plan','qualification')
    for key,value in [('qualification_sha256','wrong'),('pilot_claimed_invocations',1),('test_authorized',True)]:
        changed=deepcopy(p);changed['current_work']['gc_sequence_research_20261017'][key]=value
        with pytest.raises(RuntimeError):c.require_pilot_authority(changed,'plan','qualification')


def test_tcn_convolution_is_causal_without_history_padding():
    torch.set_num_threads(1);torch.manual_seed(1)
    model=GCSequenceTCN(2,6,{'width':4,'kernel':2,'dilations':[1,2]}).eval()
    x=torch.arange(12,dtype=torch.float32).reshape(1,6,2);changed=x.clone();changed[:,4:]+=100
    with torch.no_grad():a=model.tokens(x);b=model.tokens(changed)
    assert a.shape==(1,3,4)
    torch.testing.assert_close(a[:,0],b[:,0],rtol=0,atol=0)


@pytest.mark.parametrize('name,cls',[('transformer',GCSequenceTransformer),('tcn',GCSequenceTCN)])
def test_real_recipe_shapes_finite_gradients_and_missing_rejection(name,cls):
    spec=json.loads((Path(__file__).parents[1]/'configs/research/GC_SEQUENCE_ALL_FEATURES_20261017_027.json').read_text())
    width=len(spec['encoded_field_order']);torch.set_num_threads(1);torch.manual_seed(1)
    model=cls(width,spec['sequence_length'],spec[name]);x=torch.arange(2*6*width,dtype=torch.float32).reshape(2,6,width)/100
    y=model(x);assert y.shape==(2,) and torch.isfinite(y).all()
    y.square().mean().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    x[0,0,0]=float('nan')
    with pytest.raises(ValueError):model(x)


@pytest.mark.parametrize('name',['transformer','tcn'])
def test_synthetic_optimizer_and_saved_weight_roundtrip(name,tmp_path):
    from gx1.scripts.research_gc_sequence_v1 import fit_network
    spec=json.loads((Path(__file__).parents[1]/'configs/research/GC_SEQUENCE_ALL_FEATURES_20261017_027.json').read_text())
    spec.update(optimizer_steps=2,batch_size=2)
    x=np.arange(8*6*3,dtype=np.float32).reshape(8,6,3)/100;y=np.arange(8,dtype=float)
    pred,receipt=fit_network(name,x,y,np.arange(6),np.arange(6,8),spec,tmp_path)
    assert pred.shape==(2,) and np.isfinite(pred).all()
    assert receipt['optimizer_steps']==2 and receipt['model_readback_bit_exact']


def test_nullable_contract_ids_are_checked_on_actual_admitted_rows():
    c.require_contract_alignment(np.array([1,2,2]),np.array([1,np.nan,2]),np.array([True,False,True]))
    for basis,mask in [(np.array([1,np.nan,2]),np.array([True,True,True])),(np.array([1,np.nan,3]),np.array([True,False,True]))]:
        with pytest.raises(RuntimeError,match='CONTRACT_ALIGNMENT'):c.require_contract_alignment(np.array([1,2,2]),basis,mask)


def test_categorical_states_are_not_treated_as_ordered_numbers_or_missing_fills():
    raw=np.array([[2.,0.,3.],[5.,2.,1.]])
    x,names=c.encode_inputs(raw,['real','state','session'],{'state':[0,1,2],'session':[0,1,2,3]})
    assert names==['real','state=0','state=1','state=2','session=0','session=1','session=2','session=3']
    np.testing.assert_array_equal(x[:,1:4],[[1,0,0],[0,0,1]])
    raw[0,1]=np.nan
    with pytest.raises(ValueError,match='CATEGORY_DOMAIN'):c.encode_inputs(raw,['real','state','session'],{'state':[0,1,2],'session':[0,1,2,3]})


def test_full_scope_preserves_native_fields_and_covers_each_train_population():
    spec=json.loads((Path(__file__).parents[1]/'configs/research/GC_SEQUENCE_ALL_FEATURES_20261017_027.json').read_text())
    assert len(spec['fields']['native'])==254 and len(spec['excluded_macro_fields'])==61
    assert not set(spec['fields']['native'])&set(spec['excluded_macro_fields'])
    assert len(spec['raw_field_order'])==276 and len(spec['encoded_field_order'])==283
    assert len(spec['folds'])==5 and spec['max_fits']==20 and spec['max_neural_fits']==10
    assert spec['optimizer_steps']*spec['batch_size']>=10988
