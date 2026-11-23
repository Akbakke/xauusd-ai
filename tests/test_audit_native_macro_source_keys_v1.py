"""Synthetic source-key reachability checks; no archived source or model is read."""
from __future__ import annotations
import copy,json,sys
import numpy as np
import pytest
from gx1.audit import audit_native_macro_source_keys_v1 as audit


def test_source_keys_asof_ties_use_strict_left_or_inclusive_right():
    available=np.array([100,100,200],np.int64);issues=np.array([99,100,101,200,201],np.int64)
    np.testing.assert_array_equal(audit.asof_positions(issues,available,inclusive=False),[-1,-1,1,1,2])
    np.testing.assert_array_equal(audit.asof_positions(issues,available,inclusive=True),[-1,1,1,2,2])


def test_source_keys_dependency_rows_preserve_unknown_and_never_wrap_negative_lag():
    join=np.array([-1,0,1,4],np.int64);before=join.copy()
    result=audit.dependency_rows(join,3)
    assert result.dtype==np.int64
    np.testing.assert_array_equal(result,[-1,-1,-1,1]);np.testing.assert_array_equal(join,before)


def test_source_keys_unsorted_availability_or_negative_lag_fails_closed():
    with pytest.raises(ValueError):audit.asof_positions(np.array([100],np.int64),np.array([200,100],np.int64),inclusive=False)
    with pytest.raises(ValueError):audit.dependency_rows(np.array([0,1],np.int64),-1)


def source_key_primitive_table():
    leaves=('nominal2y','nominal10y','real10y','vix_close','gld_tonnes')
    rows={date:{leaf:np.nan for leaf in leaves}for date in ('2020-01-01','2020-01-02','2020-01-05','2020-01-08','2020-01-09')}
    for date,values in (('2020-01-01',(1.,2.,.5,10.,800.)),('2020-01-05',(1.1,2.1,.6,11.,810.)),('2020-01-09',(1.2,2.2,np.inf,12.,820.))):
        rows[date].update(zip(leaves,values))
    return [('synthetic.csv',rows)]


def test_source_keys_allmissing_dates_remain_on_merged_grid_and_nan_keeps_prior_origin():
    result=audit.primitive_origins(source_key_primitive_table())
    assert result['dates']==['2020-01-01','2020-01-02','2020-01-05','2020-01-08','2020-01-09']
    np.testing.assert_array_equal(result['origins']['vix_close'],[0,0,2,2,4])
    assert result['key_axes']['vix_close'][1]is None and result['key_axes']['vix_close'][3]is None
    assert result['key_axes']['vix_close'][2]=='synthetic.csv|vix_close|2020-01-05'
    assert {k:result['presence']['vix_close'][k]for k in ('parsed_present','missing','nonfinite_invalid')}=={'parsed_present':3,'missing':2,'nonfinite_invalid':0}


def test_source_keys_inf_is_canonical_notna_present_but_explicitly_invalid():
    result=audit.primitive_origins(source_key_primitive_table())
    assert {k:result['presence']['real10y'][k]for k in ('parsed_present','missing','nonfinite_invalid')}=={'parsed_present':3,'missing':2,'nonfinite_invalid':1}
    assert result['origins']['real10y'][-1]==4
    assert result['key_axes']['real10y'][-1]=='synthetic.csv|real10y|2020-01-09'
    table=[('synthetic.csv',{'2020-01-01':{leaf:np.nan for leaf in result['origins']},
        '2020-01-02':{leaf:1. for leaf in result['origins']}})]
    first=audit.primitive_origins(table)
    for origins in first['origins'].values():np.testing.assert_array_equal(origins,[-1,1])


def test_source_keys_lag_uses_merged_rows_then_ffill_origin_not_vendor_or_calendar_lag():
    primitives=audit.primitive_origins(source_key_primitive_table())
    available=np.arange(1,6,dtype=np.int64)*100;issues={'signal':np.array([501],np.int64),'context':np.array([501],np.int64)}
    result=audit.make_field_ledger(issues,available,[('vix_close',3)],primitives['key_axes'],primitives['origins'])
    assert result['surfaces']['signal']['raw_keys']==['synthetic.csv|vix_close|2020-01-01']
    assert result['surfaces']['signal']['supported_issue_rows']==1


def test_source_keys_rv20_reaches_all_21_close_observations_not_just_endpoints():
    keys=[f'panel|close|{i}'for i in range(21)];available=np.arange(1,22,dtype=np.int64)*100
    issues={'signal':np.array([2101],np.int64),'context':np.array([2101],np.int64)}
    result=audit.make_field_ledger(issues,available,[('close',i)for i in range(21)],{'close':keys})
    endpoints=audit.make_field_ledger(issues,available,[('close',0),('close',20)],{'close':keys})
    assert result['surfaces']['signal']['raw_key_count']==21 and set(result['surfaces']['signal']['raw_keys'])==set(keys)
    assert endpoints['surfaces']['signal']['raw_key_count']==2


def test_source_keys_signal_history_reach_and_context_reach_are_reported_separately():
    issues={'signal':np.array([101,201],np.int64),'context':np.array([201],np.int64)}
    result=audit.make_field_ledger(issues,np.array([100,200],np.int64),[('leaf',0)],{'leaf':['K0','K1']})
    assert result['surfaces']['signal']['supported_issue_rows']==2
    assert result['surfaces']['context']['supported_issue_rows']==1
    assert result['context_keys_subset_of_signal']is True
    assert result['signal_only_raw_keys']==['K0']and result['context_only_raw_keys']==[]


def test_source_keys_zero_support_does_not_wrap_unknown_join_to_last_source_key():
    issues={surface:np.array([99],np.int64)for surface in ('signal','context')}
    result=audit.make_field_ledger(issues,np.array([100],np.int64),[('leaf',0)],{'leaf':['MUST_NOT_WRAP']})
    for report in result['surfaces'].values():
        assert report['supported_issue_rows']==0 and report['unsupported_issue_rows']==1
        assert report['raw_key_count']==0 and report['raw_keys']==[]


def test_source_keys_local_history_is_observed_row_union_across_clock_gap():
    from gx1.contracts.market_state_native_m5_feature_inputs_v1 import _window_union
    tick=300_000_000_000;clock=np.arange(97,dtype=np.int64)*tick;clock[95:]+=86_400_000_000_000
    rows=_window_union(np.array([96],np.int64),96,len(clock))
    np.testing.assert_array_equal(rows,np.arange(1,97,dtype=np.int64))
    assert clock[rows[0]]<clock[-1]-95*tick
    assert len(rows)>int((clock>=clock[-1]-95*tick).sum())


def test_source_keys_panel_uses_actual_last_h1_timestamp_and_fixed_utc22_availability():
    times=np.array([np.datetime64(t,'ns').astype(np.int64)for t in ('2020-01-01T23:00','2020-01-02T20:00','2020-01-02T23:00')])
    result=audit.panel_close_keys(times,'panel.parquet')
    available=np.array([np.datetime64(t,'ns').astype(np.int64)for t in ('2020-01-02T22:00','2020-01-03T22:00')])
    np.testing.assert_array_equal(result['available'],available)
    assert result['keys']==[f'panel.parquet|D1label={int(a-audit.DAY)}|closeH1={int(t)}'for a,t in zip(available,times[[1,2]])]
    assert result['observed_H1_rows']==3


def test_source_keys_panel_excludes_noneligible_d1_close_before_keying_and_rejects_duplicates():
    times=np.array([np.datetime64(t,'ns').astype(np.int64)for t in ('2025-12-30T23:00','2025-12-31T23:00')])
    result=audit.panel_close_keys(times,'panel.parquet')
    assert result['observed_H1_rows']==1 and len(result['keys'])==1
    assert result['available'][0]<audit.END and result['keys'][0].endswith(f'closeH1={int(times[0])}')
    with pytest.raises(ValueError,match='PANEL_TIME_ORDER'):audit.panel_close_keys(np.array([times[0],times[0]]),'panel.parquet')


def test_source_keys_csv_numeric_shim_matches_canonical_coercion_and_rejects_boolean_ambiguity(tmp_path):
    worker=audit.load_primitive_worker();original=worker.numeric_text
    path=tmp_path/'synthetic.csv';path.write_text('Date,2 Yr,10 Yr\n2025-01-01,1_000,1e400\n2025-01-02,NaN,inf\n2025-01-03,2.5,-inf\n')
    record={'raw_path':str(path),'item':'TREASURY_NOMINAL_SYNTHETIC'}
    table,counts=audit.canonical_csv_primitives(worker,record,'treasury')
    assert np.isnan(table['2025-01-01']['nominal2y'])and np.isnan(table['2025-01-01']['nominal10y'])
    assert np.isnan(table['2025-01-02']['nominal2y'])and np.isposinf(table['2025-01-02']['nominal10y'])
    assert table['2025-01-03']['nominal2y']==2.5 and np.isneginf(table['2025-01-03']['nominal10y'])
    assert counts['eligible_numeric_cells_converted']==6 and worker.numeric_text is original
    path.write_text('Date,2 Yr,10 Yr\n2025-01-01,True,False\n')
    with pytest.raises(ValueError,match='AMBIGUOUS_CSV_BOOLEAN'):audit.canonical_csv_primitives(worker,record,'treasury')
    assert worker.numeric_text is original


def test_source_keys_real_csv_owner_checks_train_end_before_excluded_value_access(monkeypatch,tmp_path):
    worker=audit.load_primitive_worker();assert worker.END==audit.END
    touched=[]
    class ExcludedRow(dict):
        def __getitem__(self,key):
            if key=='Date':return '2026-01-02'
            touched.append(key);raise AssertionError('excluded primitive value accessed')
    class Reader:
        fieldnames=['Date','2 Yr','10 Yr']
        def __iter__(self):return iter([ExcludedRow()])
    monkeypatch.setattr(worker.csv,'DictReader',lambda stream:Reader())
    path=tmp_path/'synthetic.csv';path.write_text('Date,2 Yr,10 Yr\n')
    table,counts=audit.canonical_csv_primitives(worker,{'raw_path':str(path),'item':'TREASURY_NOMINAL_SYNTHETIC'},'treasury')
    assert table=={}and counts['eligible_numeric_cells_converted']==0 and touched==[]


def source_key_lifecycle(monkeypatch,tmp_path):
    monkeypatch.setattr(audit,'ROOT',tmp_path/'OUT');binding={'path':str(tmp_path/'PLAN.json'),'sha256':'1'*64};pref={'path':str(tmp_path/'POLICY.json'),'sha256':'2'*64}
    monkeypatch.setattr(audit.guard,'require_capped_cpu_audit_execution',lambda:{'memory_max':4*1024**3,'memory_high':4*1024**3,'swap':512*1024**2,'numerical_threads':1,'cgroup_path':'/sys/fs/cgroup/synthetic'})
    monkeypatch.setattr(audit.guard,'require_capped_lock_ancestry',lambda:{'scope':'/synthetic'})
    monkeypatch.setattr(audit,'require_timeout_ancestry',lambda *args:{'seconds':600,'scope':'/synthetic','exact_cli':True})
    monkeypatch.setattr(audit,'inspect',lambda *args,**kwargs:({},binding,pref));monkeypatch.setattr(audit,'checked_inputs',lambda plan:None)
    monkeypatch.setattr(audit,'source_clean',lambda plan:True);original=audit.ref
    monkeypatch.setattr(audit,'ref',lambda path:pref if path==audit.POLICY else binding if path==audit.PLAN else original(path))
    return binding,pref


def test_source_keys_preflight_never_loads_worker_payload_or_claim(monkeypatch,tmp_path,capsys):
    source_key_lifecycle(monkeypatch,tmp_path);seen=[]
    monkeypatch.setattr(audit,'checked_inputs',lambda plan:seen.append('metadata'))
    def forbidden(*args,**kwargs):raise AssertionError('preflight payload/worker/claim')
    monkeypatch.setattr(audit.np,'load',forbidden);monkeypatch.setattr(audit,'load_primitive_worker',forbidden)
    monkeypatch.setattr(audit,'calculate',forbidden);monkeypatch.setattr(audit,'publish',forbidden)
    monkeypatch.setattr(sys,'argv',['audit','--plan-sha256','1'*64,'--preflight']);audit.main()
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='PREFLIGHT_MACRO_SOURCE_KEYS_PASS'and result['claim_created']is False
    assert result['payloads_decoded']==result['models']==0 and seen==['metadata']and not audit.ROOT.exists()
    gate=audit.gate_expected({'path':'synthetic','sha256':'1'*64},True)
    assert gate['training_authorized']is gate['test_authorized']is False and gate['relaunch_allowed']is False


def test_source_keys_failure_consumes_claim_and_retains_progress(monkeypatch,tmp_path):
    source_key_lifecycle(monkeypatch,tmp_path)
    def fail(plan,*,progress):
        progress['phase']='synthetic stopped';raise ValueError('synthetic key failure')
    monkeypatch.setattr(audit,'calculate',fail)
    with pytest.raises(ValueError,match='synthetic key failure'):audit.run('1'*64)
    terminal=json.loads((audit.ROOT/'TERMINAL.json').read_text())
    assert terminal['status']=='FAILED_CONSUMED'and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False
    assert terminal['result']is None and terminal['progress']['phase']=='synthetic stopped'
    assert (audit.ROOT/'CLAIM.json').exists()and not (audit.ROOT/'RESULT.json').exists()


def test_source_keys_success_ledger_is_durable_and_bookend_failure_blocks_success(monkeypatch,tmp_path):
    source_key_lifecycle(monkeypatch,tmp_path/'good');monkeypatch.setattr(audit,'calculate',lambda plan,**kwargs:{'field_ledger':{'synthetic':True}})
    result=audit.run('1'*64);terminal=json.loads((audit.ROOT/'TERMINAL.json').read_text())
    assert terminal['status']=='COMPLETE'and terminal['result']==result['result']and terminal['relaunch_allowed']is False
    assert json.loads((audit.ROOT/'SOURCE_KEYS.json').read_text())['fields']=={'synthetic':True}
    source_key_lifecycle(monkeypatch,tmp_path/'drift');monkeypatch.setattr(audit,'source_clean',lambda plan:False)
    with pytest.raises(ValueError,match='SOURCE_INPUT_PLAN_POLICY_BOOT_BOOKEND'):audit.run('1'*64)
    terminal=json.loads((audit.ROOT/'TERMINAL.json').read_text())
    assert terminal['status']=='FAILED_CONSUMED'and not (audit.ROOT/'RESULT.json').exists()and not (audit.ROOT/'SOURCE_KEYS.json').exists()


def test_source_keys_outer_timeout_rejects_wrong_duration_before_admission(monkeypatch):
    from types import SimpleNamespace
    identity=SimpleNamespace(parent_pid=0,start_ticks=7,cgroup='/synthetic')
    argv=['/usr/bin/timeout','--signal=TERM','--kill-after=10s','599s',str(audit.REPO/'.venv/bin/python'),'-m','gx1.audit.audit_native_macro_source_keys_v1','--plan-sha256','1'*64]
    monkeypatch.setattr(audit.os,'getpid',lambda:12345)
    monkeypatch.setattr(audit.guard,'_lock_process',lambda *args:identity)
    monkeypatch.setattr(audit.guard,'_read_lock_proc',lambda path:b'\0'.join(x.encode()for x in argv)+b'\0')
    monkeypatch.setattr(audit.os,'readlink',lambda path:'/usr/bin/timeout'if str(path).endswith('/exe')else str(audit.REPO))
    with pytest.raises(ValueError,match='CANONICAL_OUTER_TIMEOUT'):audit.require_timeout_ancestry('1'*64,'/synthetic')
