"""Causal, missingness-preserving five-minute OFI gather; no I/O or model."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType
import hashlib
import json
import numpy as np

MINUTE_NS = 60_000_000_000
_FIELDS = {'bar_start_ns':np.dtype('int64'), 'available_ns':np.dtype('int64'),
    'contract_index':np.dtype('int16'), 'assigned':np.dtype('bool'),
    'ofi_contracts':np.dtype('float64'), 'ofi_valid':np.dtype('bool'),
    'receive_clock_excluded':np.dtype('bool'), 'book_event_state_excluded':np.dtype('bool')}


def _require(condition, reason):
    if not bool(condition):
        raise RuntimeError('CLOSED_M1_OFI_WINDOW_' + reason)


def _array_hash(named_arrays):
    """Fingerprint qualified arrays once, so fitted statistics cannot change source."""
    digest=hashlib.sha256()
    for name, array in named_arrays:
        digest.update(name.encode('ascii')+b'\0')
        digest.update(array.dtype.str.encode('ascii')+b'\0')
        digest.update(np.asarray(array.shape,dtype='<i8').tobytes())
        digest.update(memoryview(np.ascontiguousarray(array)).cast('B'))
    return digest.hexdigest()


@dataclass(frozen=True)
class QualifiedSource:
    arrays: Mapping[str, np.ndarray]
    rows: int
    source_sha256: str


def qualify_source(arrays):
    """Validate immutable M1 clock once, then reuse it for bounded batches."""
    _require(isinstance(arrays, Mapping) and set(_FIELDS).issubset(arrays), 'FIELDS')
    source = {key:arrays[key] for key in _FIELDS}
    t = source['bar_start_ns']
    _require(isinstance(t,np.ndarray) and t.ndim==1 and len(t)>0, 'CLOCK')
    for name,dtype in _FIELDS.items():
        a=source[name]
        _require(isinstance(a,np.ndarray) and a.dtype==dtype and a.shape==t.shape
            and not a.flags.writeable, 'IMMUTABLE:' + name)
    _require(np.all(t>=0) and np.all(t%MINUTE_NS==0) and np.all(np.diff(t)>0)
        and np.all(t<=np.iinfo(np.int64).max-MINUTE_NS)
        and np.array_equal(source['available_ns'],t+MINUTE_NS), 'SOURCE_CLOCK')
    return QualifiedSource(MappingProxyType(source),len(t),_array_hash(sorted(source.items())))


@dataclass(frozen=True)
class QualifiedClock:
    parent_rows: np.ndarray
    decision_ns: np.ndarray
    current_m5_contract_index: np.ndarray
    clock_sha256: str
    roll_sha256: str | None = None


def qualify_clock(parent_rows, decision_ns, current_m5_contract_index):
    """Bind the immutable authoritative parent, decision and M5-contract axes once."""
    for name,a in (('parent',parent_rows),('decision',decision_ns),('contract',current_m5_contract_index)):
        _require(isinstance(a,np.ndarray) and a.ndim==1 and not a.flags.writeable, 'CLOCK_IMMUTABLE:'+name)
    _require(parent_rows.dtype==decision_ns.dtype==np.dtype('int64') and current_m5_contract_index.dtype.kind=='i'
        and len(parent_rows)>0 and parent_rows.shape==decision_ns.shape==current_m5_contract_index.shape
        and np.all(parent_rows>=0) and np.all(np.diff(parent_rows)>0)
        and np.all(decision_ns>=5*MINUTE_NS) and np.all(np.diff(decision_ns)>0)
        and np.all(decision_ns%(5*MINUTE_NS)==0) and np.all(current_m5_contract_index>=-1), 'PARENT_CLOCK')
    return QualifiedClock(parent_rows,decision_ns,current_m5_contract_index,
        _array_hash((('parent_rows',parent_rows),('decision_ns',decision_ns),('current_m5_contract_index',current_m5_contract_index))))



def qualify_model_clock(source, *, parent_rows, decision_ns, roll_bytes, m5_available_ns,
    m5_native_row_indices, m5_features, m5_field_names, m5_cache_start_ns):
    """Join the model M5 contract mask and canonical roll to each decision.

    Callers verify input bytes, the roll reference and the model layout before
    invoking this pure owner. A missing/invalid M5 contract masks the whole M1
    bucket; a changed contract, native row or clock fails closed. No prices,
    labels, fitted statistics or model executions are used here.
    """
    from gx1.scripts import audit_raw_gc_m1_reference_v1 as roll_owner
    _require(isinstance(source,QualifiedSource) and type(roll_bytes) is bytes, 'MODEL_SOURCE_OR_ROLL')
    try:
        roll=json.loads(roll_bytes)
    except (UnicodeDecodeError,ValueError) as exc:
        raise RuntimeError('CLOSED_M1_OFI_WINDOW_MODEL_ROLL_DECODE') from exc
    _require(isinstance(roll,Mapping), 'MODEL_ROLL_MAPPING')
    roll_sha=hashlib.sha256(roll_bytes).hexdigest()
    for name,a in (('parent',parent_rows),('decision',decision_ns),('m5_available',m5_available_ns),
        ('m5_native',m5_native_row_indices),('m5_cache',m5_cache_start_ns)):
        _require(isinstance(a,np.ndarray) and a.dtype==np.int64 and a.ndim==1
            and not a.flags.writeable and len(a)>0, 'MODEL_AXIS:'+name)
    _require(parent_rows.shape==decision_ns.shape and np.all(np.diff(parent_rows)>0)
        and np.all(np.diff(decision_ns)>0) and np.all(decision_ns%(5*MINUTE_NS)==0)
        and np.all(decision_ns>=5*MINUTE_NS) and np.all(np.diff(m5_available_ns)>0)
        and m5_available_ns.shape==m5_native_row_indices.shape
        and np.all(np.diff(m5_native_row_indices)>0) and np.all(np.diff(m5_cache_start_ns)>0), 'MODEL_CLOCKS')
    _require(isinstance(m5_features,np.ndarray) and m5_features.dtype==np.float32
        and m5_features.ndim==2 and m5_features.shape[0]==len(m5_available_ns)
        and not m5_features.flags.writeable and isinstance(m5_field_names,(tuple,list))
        and len(m5_field_names)==m5_features.shape[1] and len(set(m5_field_names))==len(m5_field_names), 'MODEL_FEATURES')
    value_name='gc_m5_observed_book_contract_valid';mask_name=value_name+'_observed'
    _require(value_name in m5_field_names and mask_name in m5_field_names, 'MODEL_CONTRACT_FIELDS')
    at=np.searchsorted(m5_available_ns,decision_ns);inside=at<len(m5_available_ns)
    _require(np.all(inside) and np.array_equal(m5_available_ns[at],decision_ns), 'MODEL_DECISION_JOIN')
    native=m5_native_row_indices[at]
    _require(np.all(native>=0) and np.all(native<len(m5_cache_start_ns))
        and np.array_equal(m5_cache_start_ns[native],decision_ns-5*MINUTE_NS), 'MODEL_NATIVE_M5_JOIN')
    marker=m5_features[at,m5_field_names.index(value_name)]
    marker_observed=m5_features[at,m5_field_names.index(mask_name)]
    _require(np.isin(marker,(0.,1.)).all() and np.isin(marker_observed,(0.,1.)).all()
        and np.all(marker[marker_observed==0]==0), 'MODEL_CONTRACT_MASK')
    order=roll.get('contract_order');daily=roll.get('selected_daily')
    _require(roll.get('origin_ns')==roll_owner._MASK134_ORIGIN and roll.get('availability_offset_ns')==MINUTE_NS
        and isinstance(order,list) and 0<len(order)<=32768 and len(set(order))==len(order)
        and all(type(x) is str and x for x in order) and isinstance(daily,list) and len(daily)>0
        and all(type(x) is int and -1<=x<len(order) for x in daily), 'MODEL_ROLL')
    wanted=decision_ns[:,None]-np.arange(5,0,-1,dtype=np.int64)[None,:]*MINUTE_NS
    scheduled=roll_owner.reference_roll_minutes134(wanted.ravel(),np.asarray(daily,np.int16)).reshape(wanted.shape)
    a=source.arrays;positions=np.searchsorted(a['bar_start_ns'],wanted);found=positions<len(a['bar_start_ns'])
    clip=np.minimum(positions,len(a['bar_start_ns'])-1);found &= a['bar_start_ns'][clip]==wanted
    _require(np.all(a['contract_index'][clip][found]==scheduled[found]), 'MODEL_ROLL_M1_CONTRACT')
    same=(scheduled[:,0]>=0)&np.all(scheduled==scheduled[:,0,None],axis=1)
    is_observed=marker_observed==1
    _require(np.all((marker[is_observed]==1)==same[is_observed]), 'MODEL_ROLL_M5_CONTRACT')
    current=np.where(is_observed & same,scheduled[:,0],-1).astype(np.int16)
    current.setflags(write=False)
    base=qualify_clock(parent_rows,decision_ns,current)
    identity=hashlib.sha256(bytes.fromhex(base.clock_sha256)+bytes.fromhex(source.source_sha256)
        +bytes.fromhex(roll_sha)).hexdigest()
    return QualifiedClock(base.parent_rows,base.decision_ns,base.current_m5_contract_index,identity,roll_sha)


def gather(source, clock, *, batch_parent_rows):
    """Return [D-5,D-4,...,D-1] OFI and masks, never substitute older values.

    The caller provides authoritative current-M5 contract indices. An unknown
    index is -1; a minute on another contract is masked. Observed zero differs
    from missing zero by the separate boolean mask.
    """
    _require(isinstance(source,QualifiedSource) and isinstance(clock,QualifiedClock), 'SOURCE_OR_CLOCK')
    parent_rows=np.asarray(batch_parent_rows)
    _require(parent_rows.dtype==np.dtype('int64') and parent_rows.ndim==1 and len(parent_rows)>0
        and np.unique(parent_rows).size==len(parent_rows), 'BATCH_PARENTS')
    index=np.searchsorted(clock.parent_rows,parent_rows)
    _require(np.all(index<len(clock.parent_rows))
        and np.array_equal(clock.parent_rows[index],parent_rows), 'EXACT_PARENT_JOIN')
    decision_ns=clock.decision_ns[index]
    current_m5_contract_index=clock.current_m5_contract_index[index]
    a=source.arrays;t=a['bar_start_ns']
    wanted=decision_ns[:,None]-np.arange(5,0,-1,dtype=np.int64)[None,:]*MINUTE_NS
    at=np.searchsorted(t,wanted,side='left'); found=at<len(t); clip=np.minimum(at,len(t)-1)
    found &= t[clip]==wanted
    selected=clip[found]
    _require(np.all(a['available_ns'][selected]==wanted[found]+MINUTE_NS), 'EXACT_AVAILABILITY')
    admitted=found & a['assigned'][clip] & a['ofi_valid'][clip] & ~a['receive_clock_excluded'][clip] & ~a['book_event_state_excluded'][clip]
    admitted &= (current_m5_contract_index[:,None]>=0) & (a['contract_index'][clip]==current_m5_contract_index[:,None])
    raw=a['ofi_contracts'][clip]
    _require(np.isfinite(raw[admitted]).all(), 'NONFINITE_OBSERVED')
    values=np.where(admitted,raw,0.0).astype(np.float64,copy=False)
    result={'parent_rows':parent_rows.copy(),'decision_ns':decision_ns.copy(),
        'minute_start_ns':wanted.copy(),'ofi_contracts':values,'ofi_observed':admitted.copy()}
    for v in result.values():v.setflags(write=False)
    return result
