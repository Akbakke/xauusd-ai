"""Causal and missingness tests for the separate five-minute M1 input."""
import json
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


def _model_clock_case():
    from gx1.scripts import audit_raw_gc_m1_reference_v1 as roll_owner
    origin=roll_owner._MASK134_ORIGIN;minute=owner.MINUTE_NS;day=roll_owner._MASK134_DAY
    starts=np.r_[origin+np.arange(5,10,dtype=np.int64)*minute,
        origin+day+np.arange(0,5,dtype=np.int64)*minute]
    contracts=np.array([1]*5+[1]+[2]*4,dtype=np.int16)
    data={'bar_start_ns':starts,'available_ns':starts+minute,'contract_index':contracts,
        'assigned':np.ones(10,bool),'ofi_contracts':np.arange(10,dtype=np.float64),
        'ofi_valid':np.ones(10,bool),'receive_clock_excluded':np.zeros(10,bool),
        'book_event_state_excluded':np.zeros(10,bool)}
    parents=np.array([3,8],np.int64);decisions=np.array([origin+10*minute,origin+day+5*minute],np.int64)
    model_available=decisions.copy();model_native=np.array([0,1],np.int64);cache_start=decisions-5*minute
    model=np.array([[1.,1.],[0.,1.]],dtype=np.float32)
    roll={'origin_ns':origin,'availability_offset_ns':minute,'selected_daily':[1,2],
        'contract_order':['unused','GCM4','GCQ4']}
    for x in (parents,decisions,model_available,model_native,cache_start,model):x.setflags(write=False)
    source=owner.qualify_source(_freeze(data))
    fields=['gc_m5_observed_book_contract_valid','gc_m5_observed_book_contract_valid_observed']
    return {'source':source,'parent_rows':parents,'decision_ns':decisions,'roll_bytes':json.dumps(roll,sort_keys=True).encode(),
        'm5_available_ns':model_available,'m5_native_row_indices':model_native,'m5_features':model,
        'm5_field_names':fields,'m5_cache_start_ns':cache_start}


def test_model_bound_clock_masks_entire_mixed_roll_bucket():
    case=_model_clock_case();clock=owner.qualify_model_clock(**case)
    assert clock.current_m5_contract_index.tolist()==[1,-1] and clock.roll_sha256 is not None
    out=owner.gather(case['source'],clock,batch_parent_rows=np.array([8,3],np.int64))
    assert not out['ofi_observed'][0].any()
    assert out['ofi_observed'][1].tolist()==[True]*5


def test_roll_file_bytes_are_part_of_the_qualified_clock_identity():
    case=_model_clock_case();first=owner.qualify_model_clock(**case)
    case['roll_bytes']+=b' ';second=owner.qualify_model_clock(**case)
    assert np.array_equal(first.current_m5_contract_index,second.current_m5_contract_index)
    assert first.roll_sha256!=second.roll_sha256 and first.clock_sha256!=second.clock_sha256


def test_unobserved_model_contract_mask_never_creates_contract():
    case=_model_clock_case();features=case['m5_features'].copy();features[0]=[0.,0.];features.setflags(write=False)
    case['m5_features']=features;clock=owner.qualify_model_clock(**case)
    assert clock.current_m5_contract_index.tolist()==[-1,-1]
    assert not owner.gather(case['source'],clock,batch_parent_rows=case['parent_rows'])['ofi_observed'].any()


@pytest.mark.parametrize('fault',['model_flag','masked_storage','clock','native','source_contract','roll','writable'])
def test_model_clock_rejects_unbound_mismatched_or_untrusted_inputs(fault):
    case=_model_clock_case()
    if fault in ('model_flag','masked_storage'):
        v=case['m5_features'].copy();v[1]=[1.,1.] if fault=='model_flag' else [1.,0.];v.setflags(write=False);case['m5_features']=v
    if fault=='clock':
        v=case['m5_available_ns'].copy();v[0]+=owner.MINUTE_NS;v.setflags(write=False);case['m5_available_ns']=v
    if fault=='native':
        v=case['m5_cache_start_ns'].copy();v[0]-=5*owner.MINUTE_NS;v.setflags(write=False);case['m5_cache_start_ns']=v
    if fault=='source_contract':
        d={k:v.copy() for k,v in case['source'].arrays.items()};d['contract_index'][3]=2
        case['source']=owner.qualify_source(_freeze(d))
    if fault=='roll':
        roll=json.loads(case['roll_bytes']);roll['availability_offset_ns']=0;case['roll_bytes']=json.dumps(roll).encode()
    if fault=='writable':case['parent_rows']=case['parent_rows'].copy()
    with pytest.raises(RuntimeError):owner.qualify_model_clock(**case)
