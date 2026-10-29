"""Synthetic source, clock and authorization checks; no real fit or model."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gx1.contracts import entry_closed_m1_ofi_window_v1 as windows
from gx1.contracts import entry_native_direction_m1_ofi_normalization_v1 as normalization
from gx1.scripts import materialize_entry_direction_m1_normalization_v1 as owner
from tests.test_entry_closed_m1_ofi_window_v1 import _model_clock_case


def test_one_shot_authority_requires_own_approval_and_globally_closed_training():
    ref={'path':'/synthetic/plan.json','sha256':'1'*64}
    gate={'status':'ADMITTED','plan':ref,'execution_authorized':True,'normalization_fit_authorized':True,
        'max_invocations':1,'claimed_invocations':0,'completed_invocations':0,'relaunch_allowed':False}
    policy={owner.STAGE:gate,**{k:False for k in ('training_enabled','native_training_enabled',
        'full_epoch_training_allowed','full_val_allowed','native_launch_authorized')}}
    owner.require_authority(policy,ref)
    for key,value in (('execution_authorized',False),('normalization_fit_authorized',False),
            ('claimed_invocations',1),('completed_invocations',1),('max_invocations',2),('relaunch_allowed',True)):
        with pytest.raises(RuntimeError,match='ONE_SHOT_AUTHORITY'):
            owner.require_authority({**policy,owner.STAGE:{**gate,key:value}},ref)
    for key in ('training_enabled','native_training_enabled','full_epoch_training_allowed','full_val_allowed','native_launch_authorized'):
        with pytest.raises(RuntimeError,match='GLOBAL_AUTHORITY'):
            owner.require_authority({**policy,key:True},ref)
    with pytest.raises(RuntimeError,match='ONE_SHOT_AUTHORITY'):
        owner.require_authority(policy,{**ref,'sha256':'2'*64})


def test_fit_uses_observed_train_minutes_and_never_later_values():
    case=_model_clock_case();clock=windows.qualify_model_clock(**case);cutoff=int(case['decision_ns'][1])
    fitted=owner.fit_prefix(case['source'],clock,prefix_rows=1,cutoff_ns=cutoff)
    assert normalization.require(fitted)==fitted and fitted['observed_minute_count']==5
    assert fitted['surface']['center'][0]==2.0 and fitted['clock_sha256']==clock.clock_sha256
    values={key:value.copy() for key,value in case['source'].arrays.items()}
    values['ofi_contracts'][5:]+=1_000_000
    for value in values.values():value.setflags(write=False)
    later=windows.qualify_source(values);other=windows.qualify_model_clock(**{**case,'source':later})
    new=owner.fit_prefix(later,other,prefix_rows=1,cutoff_ns=cutoff)
    assert new['surface']==fitted['surface'] and new['source_sha256']!=fitted['source_sha256']
    with pytest.raises(RuntimeError,match='STRICT_PREFIX_CUTOFF'):
        owner.fit_prefix(case['source'],clock,prefix_rows=1,cutoff_ns=int(case['decision_ns'][0]))


def test_missing_observation_is_not_replaced_or_fitted_as_zero():
    case=_model_clock_case();values={key:value.copy() for key,value in case['source'].arrays.items()}
    values['ofi_valid'][1]=False
    for value in values.values():value.setflags(write=False)
    source=windows.qualify_source(values);clock=windows.qualify_model_clock(**{**case,'source':source})
    fitted=owner.fit_prefix(source,clock,prefix_rows=1,cutoff_ns=int(case['decision_ns'][1]))
    assert fitted['observed_minute_count']==4 and fitted['surface']['center'][0]==2.5
    unbound=windows.qualify_clock(clock.parent_rows,clock.decision_ns,clock.current_m5_contract_index)
    with pytest.raises(RuntimeError,match='QUALIFIED_CLOCK_REQUIRED'):
        owner.fit_prefix(source,unbound,prefix_rows=1,cutoff_ns=int(case['decision_ns'][1]))


def test_plan_is_exact_and_never_authorizes_models_or_test():
    root=Path(__file__).resolve().parents[1]
    plan=json.loads((root/'docs/DIRECTION_M1_TRAIN_NORMALIZATION_20261029.json').read_text())
    assert owner.require_plan(plan)==plan
    for change in ('train_cutoff','test','model','later','target','relaunch'):
        bad=copy.deepcopy(plan)
        if change=='train_cutoff':bad['fold']['train_cutoff_ns']+=1
        if change=='test':bad['test_data_used']=True
        if change=='model':bad['model_forward_authorized']=True
        if change=='later':bad['method']['later_observations_in_fit']=True
        if change=='target':bad['method']['target_values_used']=True
        if change=='relaunch':bad['relaunch_allowed']=True
        with pytest.raises(RuntimeError,match='PLAN_SCOPE'):owner.require_plan(bad)


def test_plan_binds_all_executed_source_owners():
    root=Path(__file__).resolve().parents[1]
    plan=json.loads((root/'docs/DIRECTION_M1_TRAIN_NORMALIZATION_20261029.json').read_text())
    assert set(plan['sources'])==set(owner.SOURCE_PATHS)
    for name,ref in plan['sources'].items():
        path=root/name
        assert ref['path']==str(path) and hashlib.sha256(path.read_bytes()).hexdigest()==ref['sha256']
