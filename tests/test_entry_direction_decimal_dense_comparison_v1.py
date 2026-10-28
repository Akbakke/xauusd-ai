import copy

import numpy as np
import pytest

from gx1.scripts.compare_entry_direction_decimal_dense_v1 import corrected_comparison_inputs


def case():
    n=4;parents=np.arange(10,10+n,dtype=np.int64);start=np.arange(n,dtype=np.int64)*300
    decision=start+300;terminal=decision[:,None]+np.arange(1,61,dtype=np.int64)[None,:]
    observed=np.ones((n,60),dtype=np.bool_);sign=np.ones((n,60),dtype=np.int8)
    original={'parent_rows':parents,'bar_start_ns':start,'decision_ns':decision,
        'terminal_time_ns':terminal,'observed':observed,'sign':sign}
    corrected=copy.deepcopy(original);corrected['sign'][1,4]=0;corrected['sign'][2,8]=0
    at=np.array([1,2],np.int64)
    def scores(target):
        return {**{k:target[k][at].copy() for k in target},'logit_up':np.zeros((2,60),np.float32)}
    return scores(corrected),scores(original),original,corrected


def test_corrected_common_inputs_rescore_old_logits_without_mutation():
    new,old,original,corrected=case();old['logit_up'][:,4]=[0.2,-0.3]
    before=old['sign'].copy()
    ref,counts=corrected_comparison_inputs(new,old,original,corrected)
    assert counts=={'changed_cells_all_60':2,'changed_cells_5min':1}
    assert ref['sign'][:,1].tolist()==[0,1]
    assert np.array_equal(ref['logit_up'][:,1],old['logit_up'][:,4])
    assert np.array_equal(old['sign'],before) and ref['logit_up'].shape==(2,3)


@pytest.mark.parametrize('fault',['parent','candidate_sign','old_sign','observed','opposite','clock','logit_nan'])
def test_unmatched_or_invalid_predictions_fail_closed(fault):
    new,old,original,corrected=case()
    if fault=='parent':old['parent_rows'][0]+=1
    elif fault=='candidate_sign':new['sign'][0,4]=1
    elif fault=='old_sign':old['sign'][0,4]=-1
    elif fault=='observed':corrected['observed'][1,4]=False;new['observed'][0,4]=False
    elif fault=='opposite':corrected['sign'][1,4]=-1;new['sign'][0,4]=-1
    elif fault=='clock':new['decision_ns'][0]+=1
    else:new['logit_up'][0,0]=np.nan
    with pytest.raises(RuntimeError):corrected_comparison_inputs(new,old,original,corrected)


def test_end_to_end_fixed_comparison_uses_corrected_five_minute_cohort():
    from gx1.scripts.compare_entry_direction_decimal_dense_v1 import compare_decimal_dense_predictions
    minute=60_000_000_000;week=7*24*60*minute;n=40
    decision=np.repeat(np.arange(10,dtype=np.int64)*week,4)+np.tile(np.arange(4,dtype=np.int64)*5*minute,10)+5*minute
    parents=np.arange(n,dtype=np.int64);start=decision-5*minute
    labels=np.tile(np.array([1,-1,1,-1],np.int8),10)
    sign=np.repeat(labels[:,None],60,axis=1);observed=np.ones((n,60),np.bool_)
    target={'parent_rows':parents,'bar_start_ns':start,'decision_ns':decision,
        'terminal_time_ns':decision[:,None]+np.arange(1,61,dtype=np.int64)[None,:]*minute,
        'observed':observed,'sign':sign}
    corrected=copy.deepcopy(target);corrected['sign'][0,4]=0
    candidate={**copy.deepcopy(corrected),'logit_up':np.repeat(labels[:,None],60,axis=1).astype(np.float32)}
    baseline={**copy.deepcopy(target),'logit_up':-candidate['logit_up']}
    result=compare_decimal_dense_predictions(candidate,baseline,target,corrected,
        cutoff=int(decision[-1]+61*minute),replicates=64,minimum_blocks=8)
    primary=result['primary_five_minute']
    assert primary['eligible']==39 and primary['candidate']['auc']==1.0 and primary['baseline']['auc']==0.0
    assert result['correction_counts_on_common_parents']=={'changed_cells_all_60':1,'changed_cells_5min':1}
    assert len(result['panels'])==60 and result['test_data_used'] is False
