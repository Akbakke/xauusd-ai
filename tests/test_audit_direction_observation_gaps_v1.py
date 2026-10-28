import numpy as np
import pytest

from gx1.scripts.audit_direction_observation_gaps_v1 import MINUTE, summarize


def case():
    times=np.delete(np.arange(80,dtype=np.int64),np.arange(10,15))*MINUTE
    decision=np.array([5,6],dtype=np.int64)*MINUTE
    terminal=decision[:,None]+np.arange(1,61,dtype=np.int64)[None,:]*MINUTE
    observed=np.isin(terminal,times);sign=np.where(observed,1,0).astype(np.int8)
    return decision,terminal,observed,sign,times


def test_missing_endpoints_and_present_endpoint_with_incomplete_m1_clock():
    d,t,o,s,times=case();r=summarize(d,t,o,s,times,80*MINUTE)
    assert r['per_horizon'][4]['missing_before_cutoff']==2
    assert r['per_horizon'][9]['observed_before_cutoff']==2
    assert r['per_horizon'][9]['observed_with_missing_intermediate_m1']==2
    assert r['distinct_resolved_gaps_by_minutes']=={'5':1}


def test_cutoff_censors_next_source_time_and_rejects_wrong_observed_mask():
    d,t,o,s,times=case();r=summarize(d,t,o,s,times,13*MINUTE)
    assert r['per_horizon'][4]['missing_right_censored_at_cutoff']==2
    bad=o.copy();bad[0,0]=False
    with pytest.raises(RuntimeError,match='ENDPOINT:1'):summarize(d,t,bad,s,times,80*MINUTE)
