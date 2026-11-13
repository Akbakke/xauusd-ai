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


def test_profile_comparison_preserves_cross_route_difference_and_requires_own_reference():
    expected=fixture_piece(); native={k:v.copy() for k,v in expected.items()}
    dynamic={k:v.copy() for k,v in expected.items()}; dynamic['state_raw'][0,0,1]=4e-6
    grad={k:v.copy() for k,v in dynamic.items()}
    native_result=observer.compare_profile('NATIVE_NO_GRAD',native,expected,1.,1.,np=np)
    dynamic_result=observer.compare_profile('DYNAMIC_NO_GRAD',dynamic,expected,1.,1.,np=np)
    grad_result=observer.compare_profile('DYNAMIC_GRAD',grad,expected,1.,1.,
        dynamic_reference=dynamic,dynamic_loss=1.,np=np)
    assert native_result['required_reference_qualified'] is True
    assert native_result['retained_comparison']['within_existing_tolerance'] is True
    assert dynamic_result['required_reference_qualified'] is None
    assert dynamic_result['retained_comparison']['within_existing_tolerance'] is False
    assert grad_result['retained_comparison']['within_existing_tolerance'] is False
    assert grad_result['required_reference_qualified'] is True
    assert grad_result['dynamic_comparison']['within_existing_tolerance'] is True
    assert grad_result['dynamic_comparison']['raw']['state_raw']['bitexact'] is True
    assert grad_result['retained_comparison']['raw']['state_raw']['max_abs']==float(grad['state_raw'][0,0,1])
    bad_native=observer.compare_profile('NATIVE_NO_GRAD',dynamic,expected,1.,1.,np=np)
    assert bad_native['required_reference_qualified'] is False
    grad['state_raw'][0,0,1]=8e-6
    bad_grad=observer.compare_profile('DYNAMIC_GRAD',grad,expected,1.,1.,
        dynamic_reference=dynamic,dynamic_loss=1.,np=np)
    assert bad_grad['required_reference_qualified'] is False
    assert bad_grad['dynamic_comparison']['within_existing_tolerance'] is False
    bad_loss=observer.compare_profile('NATIVE_NO_GRAD',native,expected,1.001,1.,np=np)
    assert bad_loss['required_reference_qualified'] is False
    with pytest.raises(RuntimeError):
        observer.compare_profile('DYNAMIC_GRAD',grad,expected,1.,1.,np=np)
    native['future_allowed'][0,0]=4
    with pytest.raises(RuntimeError,match='AXES_TARGETS'):
        observer.compare_profile('NATIVE_NO_GRAD',native,expected,1.,1.,np=np)
    json.dumps([native_result,dynamic_result,grad_result,bad_native,bad_grad,bad_loss],allow_nan=False)


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


@pytest.mark.parametrize('phase,field',[('INITIAL','parent_order'),('FINAL','selected_parent_order')])
def test_parent_order_uses_both_actual_saved_orders_without_recipe_key(phase,field):
    order=torch.tensor([11,12,14],dtype=torch.int64)
    states={'INITIAL':{'parent_order':order.clone()},'FINAL':{'selected_parent_order':order.clone()}}
    observer._require_parent_order(order,states,torch=torch)
    states[phase][field]=order[[0,2,1]]
    with pytest.raises(RuntimeError,match='ACTUAL_PAIRED_PARENT_ORDER'):
        observer._require_parent_order(order,states,torch=torch)
    states[phase][field]=order.float()
    with pytest.raises(RuntimeError,match='ACTUAL_PAIRED_PARENT_ORDER'):
        observer._require_parent_order(order,states,torch=torch)
