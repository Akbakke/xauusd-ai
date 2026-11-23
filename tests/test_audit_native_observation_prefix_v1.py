"""Generated lifecycle and full-serving-axis prefix witnesses; no archived payloads."""
from __future__ import annotations
import copy,json,sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from gx1.audit import audit_native_observation_prefix_v1 as audit


def prefix_lifecycle_fixture():
    high,low='high_pivot','low_pivot'
    events=[('create',5,3,high,41,110.),('create',6,4,low,19,90.),
        ('create',7,5,high,0,111.),('create',8,6,low,1,89.)]
    state={'levels':[{'level_id':41,'birth_side':high,'status':'expired','retest_state':'expired'},
        {'level_id':19,'birth_side':low,'status':'broken','retest_state':'failed'},
        {'level_id':0,'birth_side':high,'status':'active','retest_state':'none'},
        {'level_id':1,'birth_side':low,'status':'active','retest_state':'none'}]}
    return state,events


def test_prefix_witness_chooses_first_create_per_actual_birth_side_not_ids_zero_one():
    state,events=prefix_lifecycle_fixture();result=audit.first_anchor_witness(state,events)
    first=result['first_create_by_birth_side']
    assert first['high_pivot']['level_id']==41 and first['low_pivot']['level_id']==19
    assert first['high_pivot']['create_bar']==5 and first['high_pivot']['anchor_bar']==3
    assert result['both_terminal']is True and result['missing_birth_sides']==[]
    # Final canonical state proves expiry/failure; no expiry callback is fabricated.
    assert {e[0]for e in events}=={'create'}


def test_prefix_witness_requires_first_anchors_on_both_birth_sides():
    state,events=prefix_lifecycle_fixture();events=[e for e in events if e[3]=='high_pivot']
    result=audit.first_anchor_witness(state,events)
    assert result['both_terminal']is False and result['missing_birth_sides']==['low_pivot']


def test_prefix_witness_active_held_flip_does_not_become_terminal():
    state,events=prefix_lifecycle_fixture();state['levels'][0].update(status='active',retest_state='held',side_of_origin='low_pivot')
    result=audit.first_anchor_witness(state,events)
    assert result['both_terminal']is False and result['first_create_by_birth_side']['high_pivot']['terminal']is False


def test_prefix_witness_broken_pending_retest_is_not_a_terminal_anchor():
    state,events=prefix_lifecycle_fixture();state['levels'][1]['retest_state']='pending'
    result=audit.first_anchor_witness(state,events)
    assert result['both_terminal']is False and result['first_create_by_birth_side']['low_pivot']['terminal']is False


def test_prefix_witness_requires_real_positive_release_duration_not_warm_age_zero():
    for duration in (None,0):
        assert audit.squeeze_release_witness(SimpleNamespace(last_release_duration=duration,release_age_bars=0))is False
    assert audit.squeeze_release_witness(SimpleNamespace(last_release_duration=3,release_age_bars=9))is True


def test_prefix_witness_pass_requires_strict_end_before_history_start():
    state,events=prefix_lifecycle_fixture();carry=SimpleNamespace(last_release_duration=3,release_age_bars=9)
    good=audit.evaluate_witness(state,events,carry,prefix_end_ns=10,history_start_ns=11)
    assert good['status']=='PASS'and good['reasons']==[]and good['actual_squeeze_release']is True
    for end in (11,12):
        result=audit.evaluate_witness(state,events,carry,prefix_end_ns=end,history_start_ns=11)
        assert result['status']=='UNKNOWN'and result['reasons']


def test_prefix_witness_missing_release_or_nonterminal_anchor_stays_unknown():
    state,events=prefix_lifecycle_fixture();carry=SimpleNamespace(last_release_duration=None,release_age_bars=0)
    result=audit.evaluate_witness(state,events,carry,prefix_end_ns=10,history_start_ns=11)
    assert result['status']=='UNKNOWN'and result['reasons']
    state['levels'][0].update(status='active',retest_state='none');carry.last_release_duration=3
    result=audit.evaluate_witness(state,events,carry,prefix_end_ns=10,history_start_ns=11)
    assert result['status']=='UNKNOWN'and result['reasons']


def test_prefix_owner_axis_reconciles_retained_times_with_actual_full_label_positions():
    full=np.array([100,200,300,400],np.int64);rows=np.array([1,3],np.int64);retained=np.array([200,400],np.int64)
    assert isinstance(audit.verify_owner_axis(full,retained,rows),dict)


def test_prefix_owner_axis_rejects_wrong_mapping_negative_wrap_and_nonmonotone_rows():
    full=np.array([100,200,300,400],np.int64)
    for retained,rows in (([100,400],[1,3]),([400],[-1]),([200,200],[1,1])):
        with pytest.raises(ValueError):audit.verify_owner_axis(full,np.array(retained,np.int64),np.array(rows,np.int64))


def prefix_quote_frame():
    left=pd.date_range('2009-12-29T00:05:00Z','2009-12-31T22:05:00Z',freq='5min')
    right=pd.date_range('2010-01-01T00:00:00Z','2010-01-02T00:05:00Z',freq='5min');index=left.append(right).as_unit('ns')
    price=100.+np.arange(len(index),dtype=np.float64)*.001
    return pd.DataFrame({'open':price,'high':price+.5,'low':price-.5,'close':price+.1,'volume':np.ones(len(index)),
        'bid_close':price+.09,'ask_close':price+.11,'bid_high':price+.49,'bid_low':price-.51,'ask_high':price+.51,'ask_low':price-.49},index=index)


def test_prefix_resampling_keeps_full_serving_first_partial_exclusion():
    from gx1.features import htf_features as htf
    frame=prefix_quote_frame();labels=htf.build_multi_tf_v4_closed_timestamp_indices(frame.index);cut=int(pd.Timestamp('2010-01-01T00:00:00Z').value)
    prefix=frame.loc[frame.index.asi8<cut];partial=pd.Timestamp('2009-12-28T22:00:00Z')
    assert partial in htf._resample_ohlcv(prefix,'D1',require_quotes=True).index
    assert partial not in labels['D1']
    with pytest.raises(ValueError,match='EXACT_FULL_SERVING_OWNER_AXIS'):
        audit.verify_owner_axis(labels['D1'].asi8,np.array([partial.value],np.int64),np.array([0],np.int64))
    result=audit.resample_prefix(prefix,'D1',labels['D1'],cut)
    assert partial not in result.index


def test_prefix_resampling_keeps_full_axis_h1_tail_after_last_quote_2205():
    from gx1.features import htf_features as htf
    frame=prefix_quote_frame();labels=htf.build_multi_tf_v4_closed_timestamp_indices(frame.index);cut=int(pd.Timestamp('2010-01-01T00:00:00Z').value)
    prefix=frame.loc[frame.index.asi8<cut];tail=pd.Timestamp('2009-12-31T22:00:00Z')
    assert prefix.index[-1]==pd.Timestamp('2009-12-31T22:05:00Z')
    assert tail not in htf.build_multi_tf_v4_closed_timestamp_indices(prefix.index)['H1']
    assert tail in labels['H1']and tail+pd.Timedelta(hours=1)<pd.Timestamp(cut,unit='ns',tz='UTC')
    result=audit.resample_prefix(prefix,'H1',labels['H1'],cut)
    assert tail in result.index and result.loc[tail,'volume']==2


def test_prefix_real_registry_retains_atr_nan_prefix_and_can_anchor_before_first_finite_atr():
    from gx1.features import level_registry_v1 as level,technical_indicators_v1 as technical
    first_atr=13;anchor=first_atr-level.SWING_LOOKBACK;assert anchor>=level.SWING_LOOKBACK
    index=pd.date_range('2009-01-01T00:00:00Z',periods=40,freq='5min').as_unit('ns')
    frame=pd.DataFrame({'high':np.full(40,101.),'low':np.full(40,99.),'close':np.full(40,100.)},index=index)
    frame.iloc[anchor,frame.columns.get_loc('high')]=110.
    frame['atr']=technical.wilder_atr(frame['high'],frame['low'],frame['close'],14);before=frame['atr'].to_numpy().copy()
    assert np.isnan(before[:first_atr]).all()and np.isfinite(before[first_atr:]).all()
    events=[];matrix,names,state=level.compute_level_registry_m5_block_v1(frame,recurrence_threshold_atr=1.,max_evidence_age_bars=4,
        return_registry_state=True,runtime_lifecycle_log=events)
    np.testing.assert_array_equal(frame['atr'].to_numpy(),before)
    assert np.isnan(matrix[:first_atr]).all()
    assert any(e[0]=='create'and e[1]==first_atr and e[2]==anchor for e in events)
    assert anchor<first_atr and state['bars_processed']==len(frame)


def prefix_witness_lifecycle(monkeypatch,tmp_path):
    monkeypatch.setattr(audit,'ROOT',tmp_path/'OUT');binding={'path':str(tmp_path/'PLAN.json'),'sha256':'1'*64};pref={'path':str(tmp_path/'POLICY.json'),'sha256':'2'*64}
    monkeypatch.setattr(audit.guard,'require_capped_cpu_audit_execution',lambda:{'memory_max':4*1024**3,'memory_high':4*1024**3,'swap':512*1024**2,'numerical_threads':1,'cgroup_path':'/sys/fs/cgroup/synthetic'})
    monkeypatch.setattr(audit.guard,'require_capped_lock_ancestry',lambda:{'scope':'/synthetic'})
    monkeypatch.setattr(audit,'require_timeout_ancestry',lambda *args:{'seconds':600,'scope':'/synthetic','exact_cli':True})
    monkeypatch.setattr(audit,'inspect',lambda *args,**kwargs:({},binding,pref));monkeypatch.setattr(audit,'checked_inputs',lambda plan:None)
    monkeypatch.setattr(audit,'source_clean',lambda plan:True);original=audit.ref
    monkeypatch.setattr(audit,'ref',lambda path:pref if path==audit.POLICY else binding if path==audit.PLAN else original(path))
    return binding,pref


def test_prefix_witness_preflight_never_decodes_axes_raw_or_creates_claim(monkeypatch,tmp_path,capsys):
    prefix_witness_lifecycle(monkeypatch,tmp_path);seen=[]
    monkeypatch.setattr(audit,'checked_inputs',lambda plan:seen.append('metadata'))
    def forbidden(*args,**kwargs):raise AssertionError('preflight axes/raw/claim')
    monkeypatch.setattr(audit.np,'load',forbidden);monkeypatch.setattr(audit.pipeline,'read_feature_source',forbidden)
    monkeypatch.setattr(audit.keys_owner.prior.raw.saved,'load_fields',forbidden)
    monkeypatch.setattr(audit,'calculate',forbidden);monkeypatch.setattr(audit,'publish',forbidden)
    monkeypatch.setattr(sys,'argv',['audit','--plan-sha256','1'*64,'--preflight']);audit.main()
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='PREFLIGHT_OBSERVATION_PREFIX_PASS'and result['claim_created']is False
    assert result['payloads_decoded']==result['models']==0 and seen==['metadata']and not audit.ROOT.exists()
    gate=audit.gate_expected({'path':'synthetic','sha256':'1'*64},True)
    assert gate['training_authorized']is gate['test_authorized']is False and gate['relaunch_allowed']is False


def test_prefix_witness_failure_consumes_claim_and_retains_partial_progress(monkeypatch,tmp_path):
    prefix_witness_lifecycle(monkeypatch,tmp_path)
    def fail(plan,*,progress):
        progress['phase']='synthetic stopped';raise ValueError('synthetic prefix failure')
    monkeypatch.setattr(audit,'calculate',fail)
    with pytest.raises(ValueError,match='synthetic prefix failure'):audit.run('1'*64)
    terminal=json.loads((audit.ROOT/'TERMINAL.json').read_text())
    assert terminal['status']=='FAILED_CONSUMED'and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False
    assert terminal['result']is None and terminal['progress']['phase']=='synthetic stopped'
    assert (audit.ROOT/'CLAIM.json').exists()and not (audit.ROOT/'RESULT.json').exists()


def test_prefix_witness_success_is_durable_and_source_bookend_prevents_later_result(monkeypatch,tmp_path):
    prefix_witness_lifecycle(monkeypatch,tmp_path/'good');monkeypatch.setattr(audit,'calculate',lambda plan,**kwargs:{'synthetic':True})
    result=audit.run('1'*64);terminal=json.loads((audit.ROOT/'TERMINAL.json').read_text())
    assert terminal['status']=='COMPLETE'and terminal['result']==result['result']and terminal['relaunch_allowed']is False
    assert json.loads((audit.ROOT/'RESULT.json').read_text())['source_unchanged']is True
    prefix_witness_lifecycle(monkeypatch,tmp_path/'drift');monkeypatch.setattr(audit,'source_clean',lambda plan:False)
    with pytest.raises(ValueError,match='SOURCE_INPUT_PLAN_POLICY_BOOT_BOOKEND'):audit.run('1'*64)
    terminal=json.loads((audit.ROOT/'TERMINAL.json').read_text())
    assert terminal['status']=='FAILED_CONSUMED'and not (audit.ROOT/'RESULT.json').exists()


def test_prefix_real_builder_rejects_unbound_design_before_payload_or_input_refs(monkeypatch):
    bad={**audit.DESIGN,'sha256':'0'*64};seen=[]
    design={'schema_version':'gx1_native_observation_prefix_witness_design_v1','execution_authorized':False,
        'execution':audit.EXECUTION,'capture_fields':list(audit.CAPTURE_FIELDS),'feature_fields':list(audit.FEATURE_FIELDS)}
    monkeypatch.setattr(audit,'read',lambda binding:(seen.append(binding)or design))
    def forbidden(*args,**kwargs):raise AssertionError('unbound design reached input or payload')
    monkeypatch.setattr(audit,'ref',forbidden);monkeypatch.setattr(audit.np,'load',forbidden)
    monkeypatch.setattr(audit.pipeline,'read_feature_source',forbidden)
    with pytest.raises(ValueError,match='ACTIVE_DESIGN002'):audit.build_plan(bad)
    assert seen==[bad]
