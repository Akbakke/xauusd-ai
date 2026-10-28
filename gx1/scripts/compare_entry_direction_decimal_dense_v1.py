"""Compare fixed 60-head scores on the same corrected targets; no I/O or fit."""
from __future__ import annotations

import numpy as np


def _require(ok, reason):
    if not bool(ok):
        raise RuntimeError('DECIMAL_DENSE_COMPARISON_' + reason)


def corrected_comparison_inputs(candidate, baseline, original, corrected):
    """Join exact parent/clock/target identities and re-score old logits on exact ties.

    Caller owns source hashes, admitted data reads and cutoff. This function does
    not select rows, fit a model or infer missing prices.
    """
    axes=('parent_rows','bar_start_ns','decision_ns')
    targets=('terminal_time_ns','observed','sign')
    n=len(candidate['parent_rows'])
    _require(n>0, 'EMPTY')
    for key in axes:
        x,y=candidate[key],baseline[key]
        _require(x.dtype==y.dtype==np.int64 and x.shape==y.shape==(n,)
            and np.array_equal(x,y), 'PREDICTION_AXIS:'+key)
    _require(np.all(np.diff(candidate['parent_rows'])>0)
        and np.all(np.diff(candidate['decision_ns'])>0), 'PREDICTION_ORDER')
    for target in (original,corrected):
        parents=target['parent_rows']
        _require(parents.dtype==np.int64 and parents.ndim==1 and len(parents)>0
            and np.all(np.diff(parents)>0), 'TARGET_PARENT_ORDER')
        _require(all(target[key].dtype==np.int64 and target[key].shape==parents.shape for key in axes), 'TARGET_AXES')
        for key,dtype in (('terminal_time_ns',np.int64),('observed',np.bool_),('sign',np.int8)):
            _require(target[key].dtype==dtype and target[key].shape==(len(parents),60), 'TARGET_SHAPE:'+key)
    _require(np.array_equal(original['parent_rows'],corrected['parent_rows']), 'TARGET_POPULATION')
    at=np.searchsorted(corrected['parent_rows'],candidate['parent_rows'])
    _require(np.all(at<len(corrected['parent_rows']))
        and np.array_equal(corrected['parent_rows'][at],candidate['parent_rows']), 'PARENT_JOIN')
    for key in axes:
        _require(np.array_equal(original[key][at],candidate[key])
            and np.array_equal(corrected[key][at],candidate[key]), 'TARGET_AXIS:'+key)
    for source,target in ((baseline,original),(candidate,corrected)):
        for key,dtype in (('terminal_time_ns',np.int64),('observed',np.bool_),('sign',np.int8)):
            _require(source[key].dtype==dtype and source[key].shape==(n,60)
                and np.array_equal(source[key],target[key][at]), 'TARGET_VALUE:'+key)
        _require(source['logit_up'].dtype in (np.float32,np.float64) and source['logit_up'].shape==(n,60)
            and np.isfinite(source['logit_up']).all(), 'LOGIT_DOMAIN')
    for key in ('terminal_time_ns','observed'):
        _require(np.array_equal(original[key][at],corrected[key][at]), 'CORRECTION_MASK_OR_CLOCK:'+key)
    before,after=original['sign'][at],corrected['sign'][at]
    _require(np.isin(before,(-1,0,1)).all() and np.isin(after,(-1,0,1)).all()
        and np.all(before[~original['observed'][at]]==0)
        and np.all(after[~corrected['observed'][at]]==0), 'SIGN_DOMAIN')
    changed=before!=after
    _require(np.all((after[changed]==0)&(before[changed]!=0)), 'NONZERO_SIGN_FLIP')
    anchors=np.array([0,4,14],np.int64)
    reference={key:np.ascontiguousarray(baseline[key]) for key in axes}
    for key in targets:
        reference[key]=np.ascontiguousarray(corrected[key][at][:,anchors])
    reference['logit_up']=np.ascontiguousarray(baseline['logit_up'][:,anchors])
    return reference, {'changed_cells_all_60':int(changed.sum()),'changed_cells_5min':int(changed[:,4].sum())}


def compare_decimal_dense_predictions(candidate, baseline, original, corrected, *, cutoff,
                                      replicates=10000, minimum_blocks=8, candidate_report=None):
    """Run the existing fixed 60-horizon uncertainty owner with corrected anchors."""
    reference,changes=corrected_comparison_inputs(candidate,baseline,original,corrected)
    from gx1.scripts.score_entry_direction_dense_onepass_v1 import compare_saved_predictions
    result=compare_saved_predictions(candidate,reference,cutoff=cutoff,replicates=replicates,
        minimum_blocks=minimum_blocks,candidate_report=candidate_report)
    result['correction_counts_on_common_parents']=changes
    result['scope']='One reused development fold; both fixed score sets use the same exact-decimal target and eligibility masks.'
    result['limits']=('The new fit and saved baseline are compared on reused development data; matching initial rows does not prove '
        'all training inputs equal. Calendar block exchangeability, executable fills and economic edge are unproved.')
    return result
