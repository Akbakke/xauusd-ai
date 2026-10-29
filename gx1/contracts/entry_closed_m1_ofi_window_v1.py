"""Causal, missingness-preserving five-minute OFI gather; no I/O or model."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType
import hashlib
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
