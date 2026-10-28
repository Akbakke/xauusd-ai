import numpy as np
import pytest

from gx1.scripts import audit_entry_direction_slope_response_v1 as audit


def fixture():
    days=14
    decision=(np.int64(audit.TRAIN_CUTOFF)+np.repeat(np.arange(days,dtype=np.int64),4)*audit.DAY_NS
        +np.tile(np.arange(4,dtype=np.int64),days)*5*audit.MINUTE_NS)
    n=len(decision); slope=np.tile(np.array([-1,-1,1,1],np.int8),days)
    sign=np.tile(np.array([-1,1,1,1],np.int8),days)
    candidate={'parent_rows':np.arange(n,dtype=np.int64),'bar_start_ns':decision-5*audit.MINUTE_NS,
        'decision_ns':decision,'terminal_time_ns':decision[:,None]+np.arange(1,61,dtype=np.int64)[None,:]*audit.MINUTE_NS,
        'observed':np.ones((n,60),np.bool_),'sign':np.tile(sign[:,None],(1,60)),
        'logit_up':np.zeros((n,60),np.float32),'context__h1_ema20_slope_atr_sign':slope}
    old={k:v.copy() for k,v in candidate.items() if k in ('parent_rows','bar_start_ns','decision_ns','context__h1_ema20_slope_atr_sign')}
    old.update({k:candidate[k][:,[0,4,14]].copy() for k in ('terminal_time_ns','observed','sign')})
    old['logit_up']=np.zeros((n,3),np.float32)
    return candidate,old,int(decision[-1]+61*audit.MINUTE_NS)


def test_fixed_slope_groups_separate_observed_and_predicted_gaps():
    c,b,cutoff=fixture();r=audit.analyze(c,b,cutoff=cutoff,replicates=32,minimum_blocks=1)
    assert r['groups'][0]['actual_up_fraction']==.5 and r['groups'][2]['actual_up_fraction']==1
    assert r['observed_positive_minus_negative']==.5 and r['candidate_positive_minus_negative']==0
    assert r['candidate_minus_observed_contrast']==-.5
    assert all(x['contrasts']['predicted_minus_observed_contrast']['low']==-.5 for x in r['intervals'])


def test_context_anchor_and_clock_must_match():
    c,b,cutoff=fixture();b['context__h1_ema20_slope_atr_sign'][0]=1
    with pytest.raises(RuntimeError,match='SLOPE_TAG'):
        audit.analyze(c,b,cutoff=cutoff,replicates=8,minimum_blocks=1)
    c,b,cutoff=fixture();c['terminal_time_ns'][0,20]+=1
    with pytest.raises(RuntimeError,match='TARGET_CLOCK_DOMAIN'):
        audit.analyze(c,b,cutoff=cutoff,replicates=8,minimum_blocks=1)


def test_paired_bootstrap_matches_physical_block_duplicates():
    from gx1.scripts.compare_entry_direction_saved_bps_v1 import block_weights
    c,b,cutoff=fixture();rng=np.random.default_rng(14)
    c['logit_up'][:]=rng.normal(size=(len(c['parent_rows']),1)).astype(np.float32)
    out=audit.analyze(c,b,cutoff=cutoff,replicates=32,minimum_blocks=1)
    day=c['decision_ns']//audit.DAY_NS;y=(c['sign'][:,4]==1).astype(np.int8)
    slopes=c['context__h1_ema20_slope_atr_sign'];p=audit.sigmoid(c['logit_up'][:,4])
    for item,grouping,seed in zip(out['intervals'],(day,(day+3)//7),(20261027,20261028)):
        blocks,ids=np.unique(grouping,return_inverse=True);weights=block_weights(len(blocks),seed=seed,np=np,replicates=32)
        diffs=[]
        for w in weights:
            idx=np.repeat(np.arange(len(day)),w[ids]);plus=slopes[idx]==1;minus=slopes[idx]==-1
            diffs.append((p[idx][plus].mean()-p[idx][minus].mean())-(y[idx][plus].mean()-y[idx][minus].mean()))
        lo,hi=np.quantile(diffs,[.05/12,1-.05/12],method='linear')
        actual=item['contrasts']['predicted_minus_observed_contrast']
        assert actual['low']==pytest.approx(lo,abs=1e-15) and actual['high']==pytest.approx(hi,abs=1e-15)


def test_slope_analysis_runs_under_read_and_model_guards(tmp_path):
    import subprocess
    import sys
    code=r'''
import runpy,sys
from pathlib import Path
from gx1.contracts import gx1_capped_execution_v1,immutable_event_authority_v1,local_random_access_campaign_v2
from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
from gx1.contracts.entry_native_direction_context_report_v1 import CONTEXT_SPEC,_panel
from gx1.contracts.entry_direction_probability_reference_v1 import direction_rank_diagnostics
from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
from gx1.scripts import compare_entry_direction_matched_exposure_v1 as matched
from gx1.scripts.compare_entry_direction_saved_bps_v1 import block_weights
from gx1.scripts.audit_entry_direction_slope_response_v1 import analyze
c,b,cutoff=runpy.run_path('tests/test_audit_entry_direction_slope_response_v1.py')['fixture']()
sys.dont_write_bytecode=True
sys.addaudithook(candle.make_guard(Path(sys.argv[1]).resolve(),[],[],synthetic=True))
sys.setprofile(candle.no_queries_or_models)
r=analyze(c,b,cutoff=cutoff,replicates=16,minimum_blocks=1)
assert r['observed_positive_minus_negative']==.5 and r['candidate_positive_minus_negative']==0
sys.setprofile(None)
'''
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
