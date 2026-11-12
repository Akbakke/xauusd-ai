"""New original-position joins and retained-control parity; synthetic only."""
from copy import deepcopy
import numpy as np
import pytest

from gx1.audit.run_market_state_ta_m5_objective_benchmark_v1 import (
    _require_coverage_observations, _require_control_arrays, _require_common_pass1)
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TICK_NS


def batch():
    rows=np.arange(1000,1064,dtype=np.int64)
    issues=np.arange(64,dtype=np.int64)*TICK_NS
    start=32;parents=rows[start:start+16].copy();clocks=issues[start:start+16].copy()
    query=np.tile(np.asarray(NATIVE_K,np.int64),(16,1))
    truths={task:dict(parent_rows=parents.copy(),issue_ns=clocks.copy(),allowed=np.full((16,5),4,np.uint8),
        mature=np.ones((16,5),bool)) for task in ('state','future')}
    truths['state']['query_k']=query.copy()
    truths['future']['validity_ns']=clocks[:,None]+query*TICK_NS
    report=dict(state_query_k=query.copy(),future_native_k=np.asarray(NATIVE_K,np.int64))
    report.update({task+'_'+k:truths[task][k].copy() for task in truths for k in ('allowed','mature')})
    return dict(report=report,parents=parents,clocks=clocks,rows=rows,issues=issues,start=start,truths=truths,np=np)


def test_new_batch_is_joined_at_original_position_instead_of_old_panel_slot():
    value=batch();_require_coverage_observations(**value)
    value['start']=0
    with pytest.raises(RuntimeError,match='ORIGINAL_TRAIN_AXES'):_require_coverage_observations(**value)


@pytest.mark.parametrize('kind',['parent','issue','query','native_query','future_time','allowed','maturity'])
def test_new_batch_rejects_changed_axes_queries_or_target_masks(kind):
    value=batch()
    if kind=='parent':value['truths']['future']['parent_rows'][0]+=1
    elif kind=='issue':value['truths']['state']['issue_ns'][0]+=1
    elif kind=='query':value['report']['state_query_k'][0,0]+=1
    elif kind=='native_query':value['report']['future_native_k'][0]+=1
    elif kind=='future_time':value['truths']['future']['validity_ns'][0,0]+=1
    elif kind=='allowed':value['report']['future_allowed'][0,0]=7
    else:value['report']['state_mature'][0,0]=False
    with pytest.raises(RuntimeError):_require_coverage_observations(**value)


def test_control_requires_exact_arrays_and_model_not_close_average_scores():
    reference=dict(parents=np.arange(4096,dtype=np.int64),raw=np.zeros((4096,5,2),np.float32))
    arrays={k:v.copy() for k,v in reference.items()}
    assert _require_control_arrays(arrays,reference,model_digest='a',expected_digest='a',np=np)==dict(
        arrays_exact=True,model_state_exact=True,rows=4096)
    arrays['raw'][0,0,0]=np.nextafter(np.float32(0),np.float32(1))
    with pytest.raises(RuntimeError,match='CAPTURE_EXACT'):
        _require_control_arrays(arrays,reference,model_digest='a',expected_digest='a',np=np)
    arrays={k:v.copy() for k,v in reference.items()};arrays['raw'][0,0,0]=np.float32(-0.)
    with pytest.raises(RuntimeError,match='CAPTURE_EXACT'):
        _require_control_arrays(arrays,reference,model_digest='a',expected_digest='a',np=np)
    arrays={k:v.copy() for k,v in reference.items()};arrays['raw']=arrays['raw'].astype(np.float64)
    with pytest.raises(RuntimeError,match='CAPTURE_EXACT'):
        _require_control_arrays(arrays,reference,model_digest='a',expected_digest='a',np=np)
    with pytest.raises(RuntimeError,match='MODEL_DIGEST'):
        _require_control_arrays(reference,reference,model_digest='b',expected_digest='a',np=np)


def test_common_prefix_requires_model_rng_and_every_applied_step_record():
    records=[dict(pass_number=1,original_batch_index=i,loss=.25,gradient_norm=.1,skip_update=False) for i in range(256)]
    expected=dict(model_state_sha256='a',training_rng_sha256='b',records=records)
    assert _require_common_pass1(model_digest='a',rng_digest='b',records=records,expected=expected)==dict(
        model_state_exact=True,rng_exact=True,training_records_exact=True)
    for model,rng,changed in [('c','b',records),('a','c',records),('a','b',records[:-1])]:
        with pytest.raises(RuntimeError):_require_common_pass1(model_digest=model,rng_digest=rng,records=changed,expected=expected)
    changed=deepcopy(records);changed[77]['gradient_norm']=.2
    with pytest.raises(RuntimeError,match='TRAIN_RECORDS'):
        _require_common_pass1(model_digest='a',rng_digest='b',records=changed,expected=expected)
