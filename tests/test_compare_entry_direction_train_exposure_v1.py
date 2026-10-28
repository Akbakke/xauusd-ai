import hashlib
import subprocess
import sys

import numpy as np
import pytest

from gx1.scripts import compare_entry_direction_train_exposure_v1 as audit


def fixture():
    n = 12
    minute = 60_000_000_000
    start = np.int64(1714521600000000000) + np.arange(n,dtype=np.int64)*5*minute
    decision = start + 5*minute
    sign = np.tile(np.array([-1,1],np.int8),n//2)[:,None]*np.ones((1,3),np.int8)
    observed = np.ones((n,3),np.bool_)
    sign[2,1] = 0
    observed[3,1] = False
    sign[3,1] = 0
    permutation = np.array([5,2,8,0,9,3,10,1,4,11,6,7],np.int64)
    presentation = np.argsort(permutation).astype(np.int64)
    one = {'parent_rows':np.arange(n,dtype=np.int64),'bar_start_ns':start,'decision_ns':decision,
        'terminal_time_ns':decision[:,None]+np.array([1,5,15],np.int64)*minute,
        'observed':observed,'sign':sign,
        'logit_up':np.repeat(np.linspace(-1,1,n,dtype=np.float32)[:,None],3,axis=1),
        'presentation_index':presentation}
    for name,labels in audit.CONTEXT_SPEC:
        one['context__'+name] = np.full(n,labels[0][0],np.int8)
    three = {k:v.copy() for k,v in one.items()}
    three['logit_up'] = np.ascontiguousarray(one['logit_up'] + np.float32(0.25))
    parts = []
    for lo,hi in ((0,4),(4,8),(8,12)):
        use = (presentation >= lo) & (presentation < hi)
        parts.append({k:np.ascontiguousarray(v[use]) for k,v in one.items()})
    cutoff = decision[-1] + 5*minute
    prior = {'eligible_up':[],'eligible_down':[],'probability_up':[],'logit_up':[]}
    for j in range(3):
        mask = observed[:,j] & (sign[:,j] != 0) & (one['terminal_time_ns'][:,j] < cutoff)
        up = int((sign[mask,j] == 1).sum());down = int((sign[mask,j] == -1).sum())
        prior['eligible_up'].append(up);prior['eligible_down'].append(down)
        prior['probability_up'].append(up/(up+down));prior['logit_up'].append(float(np.log(up/down)))
    return parts,three,prior,cutoff


def test_fixed_partition_join_and_scores_use_exact_same_observations():
    parts,three,prior,cutoff=fixture();legacy=audit.load_legacy()
    one,three=audit.pair_arrays(parts,three,legacy=legacy,ranges=((0,4),(4,8),(8,12)),rows=12)
    out=audit.score_train(one,three,prior,cutoff=cutoff)
    five=out['panels'][1]
    assert five['counts']=={'parents':12,'eligible':9,'observed_mature_zero':1,'mature_missing':1,'immature':1}
    mask=one['observed'][:,1] & (one['sign'][:,1]!=0) & (one['terminal_time_ns'][:,1]<cutoff)
    y=(one['sign'][mask,1]==1).astype(np.int8);z=three['logit_up'][mask,1].astype(np.float64)
    pos,neg=z[y==1],z[y==0]
    expected_auc=np.mean((pos[:,None]>neg)+0.5*(pos[:,None]==neg))
    assert five['arms']['three_pass']['auc']==pytest.approx(expected_auc)
    assert five['arms']['three_pass']['logloss']==pytest.approx(np.mean(np.logaddexp(0,z)-y*z))
    assert five['arms']['fixed_prior']['auc']==.5
    assert five['three_minus_one']['auc']==0
    assert out['economic_edge_qualified'] is False and out['model_forwards']==out['optimizer_steps']==0


@pytest.mark.parametrize('key',['parent_rows','bar_start_ns','decision_ns','terminal_time_ns','observed','sign','presentation_index','context__m5_sweep_pair'])
def test_join_rejects_changed_parent_clock_label_mask_order_or_context(key):
    parts,three,_,_=fixture()
    if key=='presentation_index': three[key][0],three[key][1]=three[key][1],three[key][0]
    elif three[key].dtype==np.bool_: three[key].flat[0]=not three[key].flat[0]
    else: three[key].flat[0]+=1
    with pytest.raises(RuntimeError):
        audit.pair_arrays(parts,three,legacy=audit.load_legacy(),ranges=((0,4),(4,8),(8,12)),rows=12)


def test_partition_gaps_duplicates_and_wrong_prior_fail_closed():
    parts,three,prior,cutoff=fixture();legacy=audit.load_legacy()
    parts[0]['presentation_index'][0]=parts[0]['presentation_index'][1]
    with pytest.raises(RuntimeError,match='PRESENTATION_DOMAIN'):
        audit.pair_arrays(parts,three,legacy=legacy,ranges=((0,4),(4,8),(8,12)),rows=12)
    parts,three,prior,cutoff=fixture();one,three=audit.pair_arrays(parts,three,legacy=legacy,ranges=((0,4),(4,8),(8,12)),rows=12)
    prior['eligible_up'][1]+=1
    with pytest.raises(RuntimeError,match='FIXED_PRIOR_COUNTS'):
        audit.score_train(one,three,prior,cutoff=cutoff)


def test_only_bound_synthetic_archive_is_decoded_and_hash_checked(tmp_path):
    _,three,_,_=fixture();data={k:v[:2] for k,v in three.items()}
    data.update(hidden=np.zeros((2,128),np.float32),head_weight=np.zeros((3,128),np.float32),head_bias=np.zeros(3,np.float32))
    path=tmp_path/'synthetic.npz';np.savez(path,**data)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    out=audit._read_archive({'path':str(path),'sha256':digest})
    assert set(out)==set(three) and np.array_equal(out['parent_rows'],three['parent_rows'][:2])
    assert 'hidden' not in out
    with pytest.raises(RuntimeError,match='ARCHIVE_SHA_AND_FREEZE'):
        audit._read_archive({'path':str(path),'sha256':'0'*64})


def test_runtime_guard_allows_git_but_rejects_unbound_payloads():
    program="""\
import sys
from pathlib import Path
from gx1.scripts import compare_entry_direction_train_exposure_v1 as a
assert 'torch' not in sys.modules
reads, denied=a._install_guard([],set(),Path('/tmp/synthetic-output'))
head, branch, status=a._git_state()
assert len(head)==40 and branch=='work/gx1-current' and not reads and not denied
try:
    open('/tmp/unbound.npz','rb')
except RuntimeError as exc:
    assert 'UNBOUND_PAYLOAD' in str(exc)
else:
    raise AssertionError('unbound payload was not rejected')
"""
    run=subprocess.run([sys.executable,'-B','-c',program],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
