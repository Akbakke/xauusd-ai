"""Synthetic native clock reconciliation and new one-use audit authority tests."""
from __future__ import annotations
import copy,json,sys
import numpy as np
import pytest
from gx1.audit import audit_native_future_window_coverage_v1 as audit
T=audit.TICK


def example(closed_missing=False):
    starts=np.array([0,T,3*T,4*T],np.int64);closed=np.array([False,False,closed_missing,False,False,True])
    grid=audit.clock_grid(starts,begin=0,upper=6*T,closed=closed)
    a=np.array([T,T,3*T,6*T]);b=np.array([4*T,4*T,5*T,7*T]);present=np.ones(4,np.bool_);mature=np.array([True,True,True,False])
    expected=np.array([2,2,2,0]if closed_missing else[3,3,2,0]);missing=np.array([0,0,0,0]if closed_missing else[1,1,0,0])
    complete=np.array([closed_missing,closed_missing,True,False])
    return (a,b,present,mature,expected,missing,complete,grid)


def test_exact_prefix_reproduction_counts_repeated_issues_and_unique_windows_separately():
    args=example();r,w=audit.reconcile_windows(*args)
    assert r['compared_issues']==3 and r['unique_compared_windows']==2
    assert r['unique_present_windows']==3 and r['unique_nonmature_present_windows']==1
    assert r['reproduced_complete_compared_issues']==r['reproduced_complete_unique_windows']==1
    assert r['missing_open_slots_sum_unique_windows']==1 and r['missing_open_slots_sum_issued_cells']==2
    assert r['expected_open_slots_sum_unique_windows']==5 and r['expected_open_slots_sum_issued_cells']==8
    assert r['all_mature_exactly_reconciled'] is True
    # Direct membership on each authoritative [start,end) M5 grid, independent of prefix implementation.
    for i,(a,b)in enumerate(w['pairs']):
        slots=np.arange(a,b,T);indices=(slots//T).astype(np.int64);closed=args[-1]['closed'][indices];present=args[-1]['present'][indices]
        assert w['expected'][i]==int((~closed).sum()) and w['missing'][i]==int((~closed&~present).sum())


def test_only_manifest_weekly_mask_exempts_missing_slots():
    r,_=audit.reconcile_windows(*example(closed_missing=True))
    assert r['reproduced_complete_compared_issues']==3 and r['reproduced_complete_unique_windows']==2
    assert r['missing_open_slots_sum_unique_windows']==0


@pytest.mark.parametrize('at,name',[(4,'EXPECTED'),(5,'MISSING'),(6,'COMPLETE')])
def test_any_stored_count_or_completeness_mismatch_fails_exactly(at,name):
    args=list(example());args[at]=args[at].copy();args[at][0]=not bool(args[at][0])if at==6 else args[at][0]+1
    with pytest.raises(ValueError,match=name+'_EXACT_MISMATCH'):audit.reconcile_windows(*args)


def test_observed_clock_inside_weekly_closed_slot_is_rejected():
    with pytest.raises(ValueError,match='OBSERVED_INSIDE_DECLARED_WEEKLY_CLOSURE'):
        audit.clock_grid(np.array([0,T]),begin=0,upper=2*T,closed=np.array([False,True]))


def test_mature_window_outside_train_projection_is_reported_not_compared_or_filled():
    args=list(example());args[3]=np.ones(4,np.bool_);r,_=audit.reconcile_windows(*args)
    assert r['mature_out_of_clock_domain_issues']==1 and r['unique_mature_out_of_clock_domain_windows']==1
    assert r['all_mature_exactly_reconciled'] is False and r['compared_issues']==3


def test_missing_clock_timing_accounts_for_new_york_dst_without_inferred_closure():
    clocks=np.array([np.datetime64(s,'ns').astype(np.int64)for s in ('2021-01-06T22:00','2021-07-07T21:00')])
    r=audit.missing_patterns(clocks);utc=r['weekday_hour_counts']['UTC'];ny=r['weekday_hour_counts']['America/New_York']
    assert utc['2']['22']==utc['2']['21']==1 and ny['2']['17']==2
    assert sum(sum(day.values())for day in ny.values())==len(r['all_missing_clocks_utc'])==2
    assert 'no daily/holiday closure' in r['meaning']


def test_contiguous_missing_run_duration_does_not_bridge_nonmissing_or_excluded_slots():
    r=audit.missing_patterns(np.array([0,T,3*T],np.int64))
    assert r['contiguous_gap_duration_minutes_histogram']=={'10':1,'5':1}
    assert sum(g['missing_slots']for g in r['gaps'])==3


def test_d1_weekdays_use_both_start_and_end_in_both_timezones():
    pair=np.array([[np.datetime64('2021-01-04T00:00','ns').astype(np.int64),np.datetime64('2021-01-05T00:00','ns').astype(np.int64)]])
    r=audit.window_weekdays({'pairs':pair,'weights':np.array([3]),'expected':np.array([288]),'missing':np.array([0]),'complete':np.array([True])})
    assert r['UTC']['target_start']['0']['complete_unique']==1 and r['UTC']['target_end']['1']['complete_issues']==3
    assert r['America/New_York']['target_start']['6']['unique_windows']==1
    assert r['America/New_York']['target_end']['0']['unique_windows']==1


def test_calculate_routes_only_one_capture_clock_and_eight_target_fields(monkeypatch):
    issue=np.array([T,2*T,3*T],np.int64);shape=(3,5)
    t={'issue_ns':issue,'target_start_ns':np.tile(issue[:,None],(1,5)),'validity_ns':np.tile((issue+T)[:,None],(1,5)),
        'target_present':np.ones(shape,np.bool_),'mature':np.ones(shape,np.bool_),
        'state_window_complete':np.tile(np.array([True,True,False])[:,None],(1,5)),
        'state_expected_m5_count':np.ones(shape,np.int64),'state_missing_open_m5_count':np.tile(np.array([0,0,1],np.int64)[:,None],(1,5))}
    contract={'tf_duration_ns':{tf:T for tf in audit.TF},'tf_origin_ns':{tf:0 for tf in audit.TF}}
    plan={'inputs':{'capture':{'path':'capture'},'target_arrays':{'path':'targets'}},'capture_array_headers':{},'target_array_headers':{},'target_contract':contract,'TRAIN_issue_ns_sha256':audit.saved.array_sha(issue)}
    monkeypatch.setattr(audit,'ROWS',3);monkeypatch.setattr(audit,'START',0);monkeypatch.setattr(audit,'UPPER',6*T)
    seen=[]
    def load(binding,fields,headers):
        seen.append(tuple(fields));return {'issue_ns':issue}if binding['path']=='capture'else{key:t[key]for key in fields}
    monkeypatch.setattr(audit.saved,'load_fields',load)
    monkeypatch.setattr(audit,'scheduled_weekly_closure_mask',lambda ticks:np.zeros(len(ticks),np.bool_))
    r=audit.calculate(plan)
    assert seen==[('issue_ns',),audit.FIELDS] and len(audit.FIELDS)==8
    assert set(r['all_five_timeframes'])==set(audit.TF) and r['side_known']=='NOT_READ_NOT_IN_BUDGET'
    assert all(v['all_mature_exactly_reconciled']for v in r['all_five_timeframes'].values())


def lifecycle(monkeypatch,tmp_path):
    monkeypatch.setattr(audit,'ROOT',tmp_path/'OUT');binding={'path':str(tmp_path/'PLAN.json'),'sha256':'1'*64};pref={'path':str(tmp_path/'POLICY.json'),'sha256':'2'*64}
    monkeypatch.setattr(audit,'require_capped_cpu_audit_execution',lambda:{'memory_max':4*1024**3,'memory_high':4*1024**3,'swap':512*1024**2,'numerical_threads':1,'cgroup_path':'/sys/fs/cgroup/synthetic'})
    monkeypatch.setattr(audit,'require_capped_lock_ancestry',lambda:{'scope':'/synthetic'});monkeypatch.setattr(audit,'inspect',lambda *a,**kw:({},binding,pref))
    monkeypatch.setattr(audit,'checked_inputs',lambda p:None);monkeypatch.setattr(audit,'source_clean',lambda p:True)
    original=audit.ref;monkeypatch.setattr(audit,'ref',lambda p:pref if p==audit.POLICY else binding if p==audit.PLAN else original(p))


def test_new_one_use_audit_success_durable_receipts_and_no_relaunch(monkeypatch,tmp_path):
    lifecycle(monkeypatch,tmp_path);monkeypatch.setattr(audit,'calculate',lambda p:{'synthetic':True});r=audit.run('1'*64)
    terminal=json.loads((audit.ROOT/'TERMINAL.json').read_text());result=json.loads((audit.ROOT/'RESULT.json').read_text())
    assert terminal['status']=='COMPLETE' and terminal['result']==r['result'] and terminal['remaining_invocations']==0
    assert result['models']==result['fits']==result['forwards']==result['prices']==result['logits']==0
    with pytest.raises(FileExistsError):audit.run('1'*64)


def test_failed_clock_audit_is_consumed_and_cannot_publish_success(monkeypatch,tmp_path):
    lifecycle(monkeypatch,tmp_path)
    def fail(p):raise ValueError('exact-mismatch')
    monkeypatch.setattr(audit,'calculate',fail)
    with pytest.raises(ValueError,match='exact-mismatch'):audit.run('1'*64)
    terminal=json.loads((audit.ROOT/'TERMINAL.json').read_text())
    assert terminal['status']=='FAILED_CONSUMED' and terminal['result']is None and terminal['remaining_invocations']==0


def test_preflight_reads_no_payload_and_creates_no_claim(monkeypatch,tmp_path,capsys):
    lifecycle(monkeypatch,tmp_path);monkeypatch.setattr(audit,'calculate',lambda p:pytest.fail('preflight decoded clock'))
    monkeypatch.setattr(sys,'argv',['audit','--plan-sha256','1'*64,'--preflight']);audit.main();r=json.loads(capsys.readouterr().out)
    assert r['claim_created']is False and r['arrays_decoded']==r['models']==r['prices']==r['logits']==0
    assert not audit.ROOT.exists()
