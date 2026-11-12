"""Synthetic clock/receipt controls; no real arrays, models or fitting."""
from copy import deepcopy
import inspect

import numpy as np
import pytest

from gx1.audit import qualify_market_state_ta_benchmark_later_inputs_v1 as worker
from gx1.audit import qualify_market_state_ta_anchor_span_inputs_v1 as replay


def clock_case(monkeypatch):
    # Timestamp grid semantics are owned by unchanged HTF tests. This fixture
    # isolates selection/envelope algebra on small uniform synthetic clocks.
    monkeypatch.setattr(worker.htf,'require_multi_tf_timestamp_grid',lambda *a,**kw:None)
    start=worker.overlay.LATER_SCORE_START_NS
    step=300*10**9
    decision=start+(np.arange(118249,dtype=np.int64)-29849)*step
    axes=dict(parent_rows=3*np.arange(118249,dtype=np.int64)+2,
              bar_start_ns=decision-step,decision_ns=decision)
    for a in axes.values():a.setflags(write=False)
    plan=dict(full_axes=dict(rows=118249,**{k+'_sha256':replay.axis_sha(a) for k,a in axes.items()}),
              local_sequence_rows=worker.MODEL_NATIVE_SEQ_LEN,window_rows=dict(worker.WINDOW_ROWS))
    local=np.arange(start-200*step,start+18000*step,step,dtype=np.int64)
    clocks={}
    for tf,n in worker.WINDOW_ROWS.items():
        shift=int(worker.htf.MULTI_TF_SHIFT[tf].value)
        clocks[tf]=np.arange(start-(n+3)*shift,worker.overlay.LATER_SCORE_END_NS,shift,dtype=np.int64)
    return plan,axes,local,clocks


def test_fixed_score_envelopes_preserve_intervening_rows_and_actual_window_lengths(monkeypatch):
    plan,axes,local,clocks=clock_case(monkeypatch)
    before={k:a.tobytes() for k,a in axes.items()}
    bounds,pops,chosen,positions=worker.envelopes_from_clocks(plan,axes,local,clocks)
    assert len(chosen)==4096 and chosen[0]==29849 and chosen[-1]==47592
    assert bounds['local']==[int(positions[0])-worker.MODEL_NATIVE_SEQ_LEN+1,int(positions[-1])+1]
    for tf,n in worker.WINDOW_ROWS.items():
        shift=int(worker.htf.MULTI_TF_SHIFT[tf].value)
        first=np.flatnonzero(clocks[tf]+shift<=axes['decision_ns'][chosen[0]])[-n:]
        last=np.flatnonzero(clocks[tf]+shift<=axes['decision_ns'][chosen[-1]])[-n:]
        assert bounds[tf]==[int(first[0]),int(last[-1])+1]
        rows=np.arange(*bounds[tf],dtype=np.int64)
        assert pops[tf]['selected_unique_row_count']==len(rows)
        assert pops[tf]['selected_row_indices_sha256']==replay.row_sha(rows,tf)
        assert pops[tf]['available_ns_sha256']==replay.axis_sha(clocks[tf][rows]+shift)
    assert pops['M5']['selected_unique_row_count']>4096  # retain gaps between selected B16 groups
    assert {k:a.tobytes() for k,a in axes.items()}==before


@pytest.mark.parametrize('fault,error',[
    ('axis','FULL_AXIS'),('length','CANONICAL_WINDOW_LENGTHS'),
    ('local_clock','EXACT_SOURCE_CLOCK'),('short_history','MTF_HISTORY:D1')])
def test_envelope_derivation_rejects_unbound_axes_or_history(monkeypatch,fault,error):
    plan,axes,local,clocks=clock_case(monkeypatch)
    if fault=='axis':plan['full_axes']['parent_rows_sha256']='0'*64
    elif fault=='length':plan['window_rows']['D1']-=1
    elif fault=='local_clock':local=local+1
    else:clocks['D1']=clocks['D1'][-20:]
    with pytest.raises(RuntimeError,match=error):
        worker.envelopes_from_clocks(plan,axes,local,clocks)


def test_coverage_matches_adapter_schema_and_binds_result_not_itself(monkeypatch):
    plan,axes,local,clocks=clock_case(monkeypatch)
    _,pops,_,_=worker.envelopes_from_clocks(plan,axes,local,clocks)
    plan.update(view_sha256='a'*64,normalization_prefix=dict(rows=29574,cutoff_ns=1727654400000000000))
    view=dict(contract_sha256=plan['view_sha256'],full_axes=plan['full_axes'],prefix=plan['normalization_prefix'])
    result=dict(path=str(worker.ROOT/'RESULT.json'),sha256='b'*64)
    c=worker.coverage_record(plan,view,axes,pops,result)
    assert c['qualification']==result and c['normalization_prefix']==view['prefix']
    assert c['batch_indices']==[i*1108//255 for i in range(256)]
    assert c['physical_populations']==pops and 'coverage' not in c


def test_shared_replay_rejects_unbound_override_before_payload_io(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('payload I/O before scope binding')
    monkeypatch.setattr(replay,'sha',forbidden)
    p=dict(scope=dict(ranges=deepcopy(replay.RANGES),history_end_exclusive_ns=replay.CUTOFF),
           output_root=str(replay.ROOT))
    assert inspect.signature(replay.measure).parameters['ranges'].default is None
    with pytest.raises(RuntimeError,match='EXPLICIT_REPLAY_SCOPE'):
        replay.measure(p,cutoff=worker.overlay.LATER_SCORE_END_NS)


def mocked_run(monkeypatch,tmp_path,fault):
    from gx1.contracts import gx1_capped_execution_v1 as cap
    published={};order=[];digest='a'*64
    monkeypatch.setattr(worker,'ROOT',tmp_path/'run')
    monkeypatch.setattr(cap,'require_capped_cpu_audit_execution',lambda:{'synthetic':True})
    monkeypatch.setattr(worker,'inspect',lambda *a,**kw:(
        dict(scope=deepcopy(worker.SCOPE),inputs={'feature_view':{}},normalization_prefix={}),
        {'path':'plan','sha256':digest},{'path':'policy','sha256':digest},'b'*40))
    monkeypatch.setattr(worker.signal,'getitimer',lambda *a:(0.,0.))
    monkeypatch.setattr(worker.signal,'getsignal',lambda *a:'original')
    monkeypatch.setattr(worker.signal,'signal',lambda *a:None)
    monkeypatch.setattr(worker.signal,'setitimer',lambda *a:None)
    monkeypatch.setattr(worker,'read_axes_and_clocks',lambda p:(
        {'parent_rows':np.arange(4096,dtype=np.int64),'decision_ns':np.arange(4096,dtype=np.int64)},None,{}))
    monkeypatch.setattr(worker,'envelopes_from_clocks',lambda *a:({}, {},np.arange(4096),None))
    def publish(name,value):
        assert name not in published
        order.append(name);published[name]=deepcopy(value)
        return dict(path=name,sha256=digest)
    monkeypatch.setattr(worker,'publish',publish)
    def measure(p,**kw):
        assert kw==dict(ranges={},cutoff=worker.overlay.LATER_SCORE_END_NS,output_root=worker.ROOT)
        publish('local_QUALIFICATION.json',{'synthetic':True})
        if fault=='replay':raise RuntimeError('synthetic_replay_failure')
        return {lane:dict(parity=dict(fields=31,float32_bit_mismatched_cells=0))
                for lane in ('local',*replay.TFS)}
    monkeypatch.setattr(replay,'measure',measure)
    def verify(*a):
        if fault=='windows':raise RuntimeError('synthetic_window_failure')
        return dict(local_windows=4096,native_window_calls=4096,mtf_windows=20480)
    monkeypatch.setattr(worker,'verify_native_windows',verify)
    monkeypatch.setattr(replay,'read',lambda *a:{})
    monkeypatch.setattr(worker,'coverage_record',lambda *a:dict(status='COMPLETE',qualification=a[-1]))
    calls=[]
    def bookend(*a):
        calls.append(None)
        if fault=='bookend' and len(calls)==2:raise RuntimeError('synthetic_source_changed')
    monkeypatch.setattr(worker,'bookend',bookend)
    return published,order,digest


@pytest.mark.parametrize('fault',['none','replay','windows','bookend'])
def test_one_use_terminal_preserves_failures_and_requires_windows_before_complete(monkeypatch,tmp_path,fault):
    published,order,digest=mocked_run(monkeypatch,tmp_path,fault)
    if fault=='none':worker.run(digest)
    else:
        with pytest.raises(RuntimeError,match='synthetic_|BOOKEND'):worker.run(digest)
    assert order[:2]==['CLAIM.json','AXIS_QUALIFICATION.json']
    assert 'local_QUALIFICATION.json' in published
    terminal=published['TERMINAL.json']
    assert terminal['status']==('COMPLETE' if fault=='none' else 'FAILED_CONSUMED')
    assert terminal['remaining_invocations']==0 and terminal['relaunch_allowed'] is False
    assert terminal['source_unchanged']==(fault!='bookend')
    if fault in ('none','bookend'):
        assert order.index('RESULT.json')<order.index('COVERAGE.json')<order.index('TERMINAL.json')
        assert published['COVERAGE.json']['qualification']==terminal['result']
    else:assert 'RESULT.json' not in published and 'COVERAGE.json' not in published


def test_cap_guard_runs_before_inspection_claim_or_decode(monkeypatch):
    from gx1.contracts import gx1_capped_execution_v1 as cap
    def fail():raise RuntimeError('synthetic_cap_required')
    def forbidden(*a,**kw):raise AssertionError('must not inspect/claim/decode')
    monkeypatch.setattr(cap,'require_capped_cpu_audit_execution',fail)
    for name in ('inspect','publish','read_axes_and_clocks'):monkeypatch.setattr(worker,name,forbidden)
    with pytest.raises(RuntimeError,match='synthetic_cap_required'):worker.run('a'*64)


def test_preflight_has_no_claim_or_array_decode(monkeypatch,capsys):
    monkeypatch.setattr(worker,'inspect',lambda *a,**kw:({'scope':worker.SCOPE},{'path':'plan'},{},'a'*40))
    def forbidden(*a,**kw):raise AssertionError('preflight must not decode/claim')
    for name in ('run','publish','read_axes_and_clocks'):monkeypatch.setattr(worker,name,forbidden)
    worker.main(['--preflight','--plan-sha256','a'*64])
    assert 'READY_NO_CLAIM_NO_ARRAY_DECODE' in capsys.readouterr().out


@pytest.mark.parametrize('argv',[
    ['--preflight'],['--plan-sha256','g'*64],
    ['--preflight','--preflight','--plan-sha256','a'*64]])
def test_cli_rejects_missing_hash_and_duplicate_modes(argv):
    with pytest.raises(RuntimeError):worker.main(argv)
