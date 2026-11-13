"""Synthetic gates, scalar geometry and reporting; no actual model/data reads."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from gx1.audit import observe_market_state_ta_current_updates_v1 as observer


def fixture_piece():
    return {name:np.zeros((16,5,2),np.float32) if name.endswith('_raw')
            else np.zeros((16,5),np.int64) for name in observer.previous.BASE_KEYS}


def test_parity_requires_exact_targets_and_bounds_raw_rounding():
    expected=fixture_piece(); actual={k:v.copy() for k,v in expected.items()}
    actual['future_raw'][0,0,1]=1e-6
    result=observer._parity(actual,expected,1.+1e-6,1.,np=np)
    assert result['axes_targets_bitexact'] and not result['raw']['future_raw']['bitexact']
    actual['future_allowed'][0,0]=4
    with pytest.raises(RuntimeError,match='EXACT_AXES_TARGETS'):
        observer._parity(actual,expected,1.,1.,np=np)
    actual['future_allowed'][0,0]=0; actual['future_raw'][0,0,1]=3e-6
    with pytest.raises(RuntimeError,match='RAW_PARITY'):
        observer._parity(actual,expected,1.,1.,np=np)
    actual['future_raw'][0,0,1]=0
    with pytest.raises(RuntimeError,match='LOSS_PARITY'):
        observer._parity(actual,expected,1.001,1.,np=np)


def test_geometry_direction_sign_and_complement_close_across_disjoint_groups():
    vectors={g:{arm:torch.tensor([value],dtype=torch.float64) for arm,value in
        [('total',6.),('future_all_side',4.),('future_m5_side',1.),
         ('future_m5_mean_logit',2.),('future_non_m5_mean_logit',-3.)]} for g in ('a','b')}
    adam={g:torch.tensor([.5],dtype=torch.float64) for g in vectors}
    delta={g:torch.tensor([.1],dtype=torch.float64) for g in vectors}
    result=observer.geometry(vectors,adam,delta,torch=torch)
    m5=result['all_active_parameters']['future_m5_mean_logit']
    assert m5['sgd_total']==-24. and m5['nominal_last_adam']==2.
    assert m5['sgd_total']==m5['sgd_future_side']+m5['sgd_other_than_future_side']
    assert m5['sgd_future_side']==m5['sgd_future_m5_side']+m5['sgd_future_non_m5_side']
    assert result['all_active_parameters']['future_non_m5_mean_logit']['nominal_last_adam']==-3.
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('complete',[True,False])
def test_small_success_and_failure_reports_preserve_consumed_receipt(tmp_path,complete):
    plan=dict(path=str(tmp_path/'PLAN.json'),sha256='0'*64)
    observer.cache._publish('CLAIM.json',dict(plan=plan),root=tmp_path)
    refs=observer.cache._publish_fusion_outcome(plan,dict(source_unchanged=True,
        counts=dict(forwards=2,optimizer_updates=0),error=None if complete else 'synthetic stop'),
        complete=complete,root=tmp_path,json_bytes_bound=16384)
    result=observer.cache._read(refs['result']); terminal=observer.cache._read(refs['terminal'])
    assert terminal['status']==result['status']==('COMPLETE' if complete else 'FAILED_CONSUMED')
    assert terminal['exit_code']==(0 if complete else 1)
    assert terminal['no_relaunch_under_this_plan'] is True
    assert result['counts']['optimizer_updates']==0 and result['training_enabled'] is False
    with pytest.raises(Exception):
        observer.cache._publish('RESULT.json',{},root=tmp_path)


def test_exact_cap_matcher_permits_only_same_command_and_optional_final_preflight(tmp_path):
    source=(observer.REPO/'scripts/gx1_capped_run.sh').read_text()
    start=source.index('matches_bound_market_state_ta_current_updates() {')
    function=source[start:source.index('\n}\n',start)+3]
    plan=tmp_path/'PLAN.json'; plan.write_text('{}\n'); sha=hashlib.sha256(plan.read_bytes()).hexdigest()
    python=str(observer.REPO/'.venv/bin/python'); module='gx1.audit.observe_market_state_ta_current_updates_v1'
    flags=[python,'-m',module,'--market-state-ta-current-updates-plan',str(plan),
        '--market-state-ta-current-updates-plan-sha256',sha,'--device','cuda','--out-dir',str(tmp_path)]
    script='CANONICAL_TRAINER_PYTHON="$1"\nTA_CURRENT_UPDATES_PRODUCER_MODULE="$2"\nshift 2\n'+function+'\nmatches_bound_market_state_ta_current_updates "$@"\n'
    def accepted(args):
        return subprocess.run(['bash','-c',script,'bash',python,module,str(plan),str(tmp_path),*args],capture_output=True).returncode==0
    assert accepted(flags) and accepted(flags+['--preflight'])
    assert not accepted(flags+['--preflight','--preflight'])
    assert not accepted(flags[:-1]+[str(tmp_path/'other')])
    assert not accepted(flags+['--other'])
    assert not accepted(flags[:6]+['f'*64]+flags[7:])
