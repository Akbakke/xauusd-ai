"""Strict offline GC sequences; no native target or serving authority."""
from __future__ import annotations
import numpy as np


def sequence_rows(times, contracts, observed, *, length, step_ns):
    t=np.asarray(times);c=np.asarray(contracts);ok=np.asarray(observed)
    if length<2 or step_ns<=0 or t.ndim!=1 or c.shape!=t.shape or ok.shape!=t.shape:
        raise ValueError('GC_SEQUENCE_SHAPE')
    if t.dtype.kind not in 'iu' or c.dtype.kind not in 'iu' or ok.dtype.kind!='b':
        raise ValueError('GC_SEQUENCE_DTYPE')
    if np.any(np.diff(t)<=0):raise ValueError('GC_SEQUENCE_CLOCK')
    if len(t)<length:return np.empty((0,length),np.int64)
    rows=np.lib.stride_tricks.sliding_window_view(np.arange(len(t)),length)
    complete=ok[rows].all(axis=1)&(c[rows]>=0).all(axis=1)
    complete &= (np.diff(t[rows],axis=1)==step_ns).all(axis=1)
    complete &= (np.diff(c[rows],axis=1)==0).all(axis=1)
    return rows[complete].copy()


def purged_fold(decisions, outcomes, observed_targets, declaration, *, embargo_seconds):
    import pandas as pd
    d=np.asarray(decisions,np.int64);e=np.asarray(outcomes,np.int64);v=np.asarray(observed_targets,bool)
    start=pd.Timestamp(declaration['validation_start_utc_inclusive']).value
    end=pd.Timestamp(declaration['validation_end_utc_exclusive']).value
    train_start=pd.Timestamp(declaration['train_start_utc_inclusive']).value
    if not (d.shape==e.shape==v.shape) or embargo_seconds<0 or start>=end:
        raise ValueError('GC_SEQUENCE_FOLD')
    train=np.flatnonzero(v&(d>=train_start)&(d<start-embargo_seconds*10**9)&(e<start))
    # Future target availability never selects decision/economic rows.
    evaluation=np.flatnonzero((d>=start)&(d<end))
    return train,evaluation


def train_normalization(features, windows, train):
    """One fit on unique physical input rows, before repeated-window sampling."""
    unique=np.unique(np.asarray(windows)[train])
    if not len(unique):raise ValueError('GC_SEQUENCE_NO_TRAIN')
    x=np.asarray(features)[unique]
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('GC_SEQUENCE_NONFINITE')
    mean=x.mean(axis=0);scale=x.std(axis=0)
    scale=np.where(scale==0,1.,scale)  # Constant TRAIN axes center to zero.
    return mean,scale,unique


def standardized(features, windows, mean, scale):
    x=(np.asarray(features)[windows]-mean)/scale
    if not np.isfinite(x).all():raise ValueError('GC_SEQUENCE_NO_IMPUTATION')
    return x.astype(np.float32)


def require_pilot_authority(policy, plan_sha256, qualification_sha256):
    gate=policy['current_work']['gc_sequence_research_20261017']
    if (gate['pilot_training_authorized'] is not True or gate['pilot_status']!='AUTHORIZED_SINGLE_USE'
        or gate['pilot_claimed_invocations']!=0 or gate['pilot_completed_invocations']!=0
        or gate['plan_sha256']!=plan_sha256 or gate['qualification_sha256']!=qualification_sha256
        or gate['native_training_authorized'] is not False or gate['test_authorized'] is not False):
        raise RuntimeError('GC_SEQUENCE_PILOT_CLOSED_BY_POLICY')


def require_contract_alignment(selected, basis_contract, observed):
    selected=np.asarray(selected);basis=np.asarray(basis_contract);observed=np.asarray(observed)
    if selected.shape!=basis.shape or observed.shape!=selected.shape or observed.dtype.kind!='b':raise ValueError('GC_SEQUENCE_CONTRACT_SHAPE')
    if np.any(selected[observed]<0) or not np.isfinite(basis[observed]).all() or not np.array_equal(selected[observed],basis[observed]):
        raise RuntimeError('GC_SEQUENCE_CONTRACT_ALIGNMENT')


def encode_inputs(raw,names,domains):
    """Explicit native categorical domains; never numeric missing-value fill."""
    if len(names)!=len(set(names)) or raw.shape[1]!=len(names):raise ValueError('GC_SEQUENCE_FIELD_ORDER')
    columns=[];encoded=[]
    for j,name in enumerate(names):
        if name in domains:
            values=raw[:,j];domain=domains[name]
            if not np.isfinite(values).all() or not np.isin(values,domain).all():raise ValueError('GC_SEQUENCE_CATEGORY_DOMAIN:'+name)
            for value in domain:
                columns.append((values==value).astype(float));encoded.append(name+'='+str(value))
        else:columns.append(raw[:,j]);encoded.append(name)
    if set(domains)-set(names):raise ValueError('GC_SEQUENCE_UNUSED_CATEGORY')
    return np.column_stack(columns),encoded
