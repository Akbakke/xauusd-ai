"""Deterministic math checks; these never establish market predictiveness."""
import numpy as np
import pytest
from gx1.scripts import audit_direction_m1_conditional_v1 as owner


def _data():
    return (np.array([10,10,10,10,5,5,20],np.float32),
        np.array([1,2,3,4,7,7,0],np.float64),
        np.array([-1,1,-1,1,1,-1,1],np.int8),np.ones(7,bool),
        np.array([0,1,0,1,0,1,1],np.int64))


def _brute(total,late,sign,eligible,block,weights):
    wins=pairs=0
    for i in range(len(total)):
        for j in range(len(total)):
            if eligible[i] and eligible[j] and sign[i]==1 and sign[j]==-1 and total[i]==total[j]:
                weight=weights[block[i]]*weights[block[j]];pairs+=weight
                wins+=weight*(2 if late[i]>late[j] else 1 if late[i]==late[j] else 0)
    return wins,pairs


def test_only_identical_m5_total_pairs_are_compared_and_score_ties_get_half():
    data=_data();summary=owner.conditional_block_pairs(*data)
    w,p=_brute(*data,np.ones(2,dtype=np.int64))
    assert summary['matched_rows']==6 and summary['matched_groups']==2 and summary['pair_count']==5
    assert summary['auc']==w/(2*p)==0.7


def test_block_bootstrap_matches_explicit_resampled_pair_count_and_week_coarsening():
    data=_data();summary=owner.conditional_block_pairs(*data)
    weights=np.array([[1,1],[2,0],[0,2],[1,3]],np.int64)
    actual=owner.bootstrap_conditional_pairs(summary,weights)
    expected=[]
    for w in weights:
        wins,pairs=_brute(*data,w);expected.append(np.nan if pairs==0 else wins/(2*pairs))
    np.testing.assert_allclose(actual,expected,equal_nan=True)
    week=owner.coarsen_conditional_pairs(summary,np.array([0,0],np.int64))
    np.testing.assert_allclose(owner.bootstrap_conditional_pairs(week,np.array([[1],[2]],np.int64)),
        [summary['auc'],summary['auc']])


def test_missing_rows_never_contribute_and_no_pairs_stays_unknown():
    total,late,sign,eligible,blocks=_data();eligible[:6]=False
    result=owner.conditional_block_pairs(total,late,sign,eligible,blocks)
    assert result['pair_count']==0 and result['auc'] is None and result['matched_rows']==0
    assert np.isnan(owner.bootstrap_conditional_pairs(result,np.array([[1,1]],np.int64))[0])
    assert owner.conditional_simultaneous_bands(np.array([np.nan]),np.array([[np.nan]])) is None


@pytest.mark.parametrize('fault',['float64_total','nan_score','bad_sign','noncontiguous_blocks','negative_weight'])
def test_invalid_inputs_fail_closed(fault):
    total,late,sign,eligible,blocks=_data()
    if fault=='float64_total':total=total.astype(np.float64)
    if fault=='nan_score':late[0]=np.nan
    if fault=='bad_sign':sign[0]=2
    if fault=='noncontiguous_blocks':blocks[blocks==1]=2
    if fault=='negative_weight':
        with pytest.raises(RuntimeError):owner.bootstrap_conditional_pairs(owner.conditional_block_pairs(total,late,sign,eligible,blocks),np.array([[-1,1]]))
    else:
        with pytest.raises(RuntimeError):owner.conditional_block_pairs(total,late,sign,eligible,blocks)


def test_simultaneous_horizon_audit_respects_cutoff_missing_and_fixed_blocks():
    day=86_400_000_000_000;minute=60_000_000_000
    decision=np.array([10*day,10*day+5*minute,17*day,17*day+5*minute],np.int64)
    terminal=decision[:,None]+np.array([1,2],np.int64)[None,:]*minute
    total=np.full(4,10.,np.float32);late=np.array([1.,2.,1.,2.],np.float64)
    sign=np.array([[-1,1],[1,-1],[-1,1],[1,-1]],np.int8)
    observed=np.ones((4,2),bool);complete=np.array([True,True,True,False])
    output=owner.conditional_multi_horizon(total,late,complete,decision,terminal,observed,sign,
        cutoff_ns=18*day,replicates=20,minimum_blocks=2,day_seed=1,week_seed=2)
    assert output['parents']==4 and output['input_complete']==3
    assert [p['eligible'] for p in output['panels']]==[3,3]
    assert output['panels'][0]['conditional_auc']==1.0 and output['panels'][1]['conditional_auc']==0.0
    assert output['panels'][0]['within_m5_total_up_down_pairs']==2
    bad=terminal.copy();bad[0,0]+=minute
    with pytest.raises(RuntimeError,match='TARGET_CLOCK'):
        owner.conditional_multi_horizon(total,late,complete,decision,bad,observed,sign,
            cutoff_ns=18*day,replicates=20,minimum_blocks=2,day_seed=1,week_seed=2)


def test_separate_one_shot_authority_and_all_global_guards_are_mandatory():
    ref={'path':'/synthetic/plan.json','sha256':'1'*64}
    gate={'status':'ADMITTED','plan':ref,'execution_authorized':True,'max_invocations':1,
        'claimed_invocations':0,'completed_invocations':0,'relaunch_allowed':False}
    policy={owner.STAGE:gate,**{k:False for k in ('training_enabled','native_training_enabled',
        'full_epoch_training_allowed','full_val_allowed','native_launch_authorized')}}
    owner.require_audit_authority(policy,ref)
    for key,value in (('execution_authorized',False),('claimed_invocations',1),('relaunch_allowed',True)):
        bad={**policy,owner.STAGE:{**gate,key:value}}
        with pytest.raises(RuntimeError):owner.require_audit_authority(bad,ref)
    with pytest.raises(RuntimeError):owner.require_audit_authority({**policy,'training_enabled':True},ref)
    with pytest.raises(RuntimeError):owner.require_audit_authority(policy,{**ref,'sha256':'2'*64})


def test_all_sixty_fixed_horizons_share_simultaneous_block_bands():
    day=86_400_000_000_000;minute=60_000_000_000
    decision=np.array([100*day+week*7*day+offset*5*minute for week in range(8) for offset in (0,1)],np.int64)
    total=np.full(len(decision),8.,np.float32);late=np.tile(np.array([1.,2.],np.float64),8)
    sign=np.tile(np.array([[-1],[1]],np.int8),(8,60));observed=np.ones_like(sign,bool)
    terminal=decision[:,None]+np.arange(1,61,dtype=np.int64)[None,:]*minute
    result=owner.conditional_multi_horizon(total,late,np.ones(len(decision),bool),decision,terminal,observed,sign,
        cutoff_ns=int(decision[-1]+61*minute),replicates=40,minimum_blocks=8,day_seed=20261027,week_seed=20261028)
    assert len(result['panels'])==60 and result['day_blocks']==result['week_blocks']==8
    assert all(p['conditional_auc']==1.0 and p['both_lower_bounds_above_half'] for p in result['panels'])
    assert result['day_simultaneous_half_width']==result['week_simultaneous_half_width']==0.0


def test_target_equal_to_cutoff_is_excluded():
    total,late,sign,eligible,blocks=_data()
    day=86_400_000_000_000;minute=60_000_000_000
    decision=np.array([10*day,10*day+5*minute,10*day+10*minute,10*day+15*minute,
        17*day,17*day+5*minute,17*day+10*minute],np.int64)
    terminal=decision[:,None]+minute;sign=sign[:,None];observed=np.ones(sign.shape,bool)
    result=owner.conditional_multi_horizon(total,late,eligible,decision,terminal,observed,sign,
        cutoff_ns=int(terminal[-1,0]),replicates=10,minimum_blocks=2,day_seed=1,week_seed=2)
    assert result['panels'][0]['eligible']==6


def test_preregistered_plan_does_not_authorize_fit_and_rejects_scope_changes():
    import copy,json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    plan=json.loads((root/'docs/DIRECTION_M1_TEMPORAL_CONDITIONAL_AUDIT_20261029.json').read_text())
    assert owner.require_audit_plan(plan)==plan
    for change in ('extra_minute','fit','future_cutoff','fewer_draws'):
        bad=copy.deepcopy(plan)
        if change=='extra_minute':bad['method']['horizons_minutes'].append(61)
        if change=='fit':bad['fit_authorized']=True
        if change=='future_cutoff':bad['fold']['oof_cutoff_ns']+=60_000_000_000
        if change=='fewer_draws':bad['method']['replicates']-=1
        with pytest.raises(RuntimeError,match='PLAN_SCOPE'):owner.require_audit_plan(bad)


def test_plan_source_hashes_cover_the_exact_executed_owners():
    import hashlib,json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    plan=json.loads((root/'docs/DIRECTION_M1_TEMPORAL_CONDITIONAL_AUDIT_20261029.json').read_text())
    assert set(plan['sources'])==set(owner.SOURCE_PATHS)
    for name,ref in plan['sources'].items():
        path=root/name
        assert ref['path']==str(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==ref['sha256']
