"""Causal and missingness tests for the separate five-minute M1 input."""
import numpy as np
import pytest
from gx1.contracts import entry_closed_m1_ofi_window_v1 as owner


def _sample():
    t=np.arange(20,dtype=np.int64)*owner.MINUTE_NS
    return {'bar_start_ns':t,'available_ns':t+owner.MINUTE_NS,
        'contract_index':np.ones(20,np.int16),'assigned':np.ones(20,bool),
        'ofi_contracts':np.arange(20,dtype=np.float64),'ofi_valid':np.ones(20,bool),
        'receive_clock_excluded':np.zeros(20,bool),'book_event_state_excluded':np.zeros(20,bool)}


def _freeze(values):
    for value in values.values():value.setflags(write=False)
    return values


def _clock():
    a={'parents':np.array([3,8,9,20],np.int64),
       'decisions':np.array([5,10,15,20],np.int64)*owner.MINUTE_NS,
       'contracts':np.array([1,1,1,-1],np.int16)}
    _freeze(a)
    return owner.qualify_clock(a['parents'],a['decisions'],a['contracts'])


def test_five_exact_closed_minutes_survive_shuffled_parent_batches_and_masking():
    data=_sample();data['ofi_valid'][7]=False;data['receive_clock_excluded'][11]=True;data['contract_index'][14]=2
    source=owner.qualify_source(_freeze(data));clock=_clock()
    out=owner.gather(source,clock,batch_parent_rows=np.array([9,3,20,8],np.int64))
    assert out['parent_rows'].tolist()==[9,3,20,8]
    assert out['ofi_contracts'][1].tolist()==[0.,1.,2.,3.,4.]
    assert out['ofi_observed'][1].tolist()==[True]*5  # observed zero is not missing
    assert out['ofi_contracts'][3].tolist()==[5.,6.,0.,8.,9.]
    assert out['ofi_observed'][3].tolist()==[True,True,False,True,True]
    assert out['ofi_observed'][0].tolist()==[True,False,True,True,False]
    assert not out['ofi_observed'][2].any()
    assert np.all(out['minute_start_ns']+owner.MINUTE_NS<=out['decision_ns'][:,None])
    assert not any(value.flags.writeable for value in out.values())


def test_missing_minute_is_not_replaced_and_future_changes_cannot_affect_past():
    clock=_clock();before=owner.gather(owner.qualify_source(_freeze(_sample())),clock,batch_parent_rows=np.array([3],np.int64))
    changed=_sample();changed['ofi_contracts'][5:]=1e12
    after=owner.gather(owner.qualify_source(_freeze(changed)),clock,batch_parent_rows=np.array([3],np.int64))
    assert np.array_equal(before['ofi_contracts'],after['ofi_contracts'])
    missing={k:np.delete(v,4) for k,v in _sample().items()}
    out=owner.gather(owner.qualify_source(_freeze(missing)),clock,batch_parent_rows=np.array([3],np.int64))
    assert out['ofi_contracts'][0].tolist()==[0.,1.,2.,3.,0.]
    assert out['ofi_observed'][0].tolist()==[True,True,True,True,False]


@pytest.mark.parametrize('fault',['writable','bad_available','duplicate_clock','nonfinite_observed','duplicate_parent','unknown_parent'])
def test_bad_input_fails_closed(fault):
    data=_sample()
    if fault=='bad_available':data['available_ns'][4]+=owner.MINUTE_NS
    if fault=='duplicate_clock':data['bar_start_ns'][4]=data['bar_start_ns'][3]
    if fault=='nonfinite_observed':data['ofi_contracts'][4]=np.nan
    if fault!='writable':_freeze(data)
    if fault in ('writable','bad_available','duplicate_clock'):
        with pytest.raises(RuntimeError):owner.qualify_source(data)
    else:
        source=owner.qualify_source(data);batch=np.array([3,3] if fault=='duplicate_parent' else [4] if fault=='unknown_parent' else [3],np.int64)
        with pytest.raises(RuntimeError):owner.gather(source,_clock(),batch_parent_rows=batch)
