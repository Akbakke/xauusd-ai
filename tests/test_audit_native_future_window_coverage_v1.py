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


# RAW001 owns a separate one-use authority; fixtures contain synthetic JSON only.
def raw_example():
    times=['2021-01-04T00:00:00Z','2021-01-04T00:05:00Z','2021-01-04T00:10:00Z']
    metadata={'sequence':282,'request_from_utc':times[0],'request_to_utc_exclusive':'2021-01-04T00:15:00Z',
        'response_candles':3,'complete_candles':3,'incomplete_candles':0,'first_complete_utc':times[0],'last_complete_utc':times[-1]}
    payload={'schema_version':'xau_oanda_native_source_chunk_v2',
        'request':{'instrument':'XAU_USD','granularity':'M5','price':'MBA','from':metadata['request_from_utc'],'to':metadata['request_to_utc_exclusive']},
        'response':{'instrument':'XAU_USD','granularity':'M5','candles':[{'time':t,'complete':True,'mid':{'o':'not-used-as-price'}}for t in times]}}
    return payload,metadata


def test_raw_current_decoder_and_clock_parser_never_consume_price_fields():
    import gzip
    from gx1.contracts.xau_tape_provenance_v1 import _decode_source_chunk
    payload,metadata=raw_example();raw=gzip.compress(json.dumps(payload).encode())
    decoded=_decode_source_chunk(raw,timeframe='M5',path=audit.REPO/'synthetic-raw.json.gz')
    times=audit.raw_parse_chunk(decoded,metadata)
    assert times.dtype==np.dtype('int64') and len(times)==3 and np.all(np.diff(times)==T)
    assert payload['response']['candles'][0]['mid']['o']=='not-used-as-price'


@pytest.mark.parametrize('case,match',[
 ('schema','RAW_CHUNK_SCHEMA'),('request_instrument','RAW_REQUEST_IDENTITY'),('request_granularity','RAW_REQUEST_IDENTITY'),
 ('request_price','RAW_REQUEST_IDENTITY'),('request_extra','RAW_REQUEST_IDENTITY'),('request_from','RAW_REQUEST_WINDOW'),
 ('response_instrument','RAW_RESPONSE_IDENTITY'),('response_granularity','RAW_RESPONSE_IDENTITY'),('response_candles','RAW_RESPONSE_IDENTITY'),
 ('count','RAW_RESPONSE_COUNT'),('complete_false','RAW_COMPLETE'),('complete_integer','RAW_COMPLETE'),
 ('naive_time','RAW_EXPLICIT_UTC'),('off_grid','NOT_M5_ALIGNED'),('outside_request','RAW_CANDLE_REQUEST_DOMAIN'),
 ('duplicate','RAW_CHUNK_ORDER'),('backwards','RAW_CHUNK_ORDER'),('first_boundary','RAW_RESPONSE_BOUNDARIES'),('last_boundary','RAW_RESPONSE_BOUNDARIES')])
def test_raw_payload_rejects_identity_domain_order_and_completion_defects(case,match):
    payload,metadata=raw_example();candles=payload['response']['candles']
    if case=='schema':payload['schema_version']='other'
    elif case.startswith('request_')and case.split('_',1)[1]in('instrument','granularity','price'):payload['request'][case.split('_',1)[1]]='wrong'
    elif case=='request_extra':payload['request']['count']=500
    elif case=='request_from':payload['request']['from']='2021-01-03T23:55:00Z'
    elif case.startswith('response_'):payload['response'][case.split('_',1)[1]]='wrong'
    elif case=='count':metadata['response_candles']=4
    elif case=='complete_false':candles[0]['complete']=False
    elif case=='complete_integer':candles[0]['complete']=1
    elif case=='naive_time':candles[0]['time']='2021-01-04T00:00:00'
    elif case=='off_grid':candles[0]['time']='2021-01-04T00:01:00Z'
    elif case=='outside_request':candles[-1]['time']=metadata['request_to_utc_exclusive']
    elif case=='duplicate':candles[1]['time']=candles[0]['time']
    elif case=='backwards':candles[0],candles[1]=candles[1],candles[0]
    elif case=='first_boundary':metadata['first_complete_utc']='2021-01-04T00:05:00Z'
    elif case=='last_boundary':metadata['last_complete_utc']='2021-01-04T00:05:00Z'
    with pytest.raises((ValueError,RuntimeError),match=match):audit.raw_parse_chunk(payload,metadata)


def raw_compare_example(monkeypatch):
    begin=int(np.datetime64('2021-01-04T00:00','ns').astype(np.int64));starts=begin+np.array([0,1,3])*T
    monkeypatch.setattr(audit,'ROWS',3);monkeypatch.setattr(audit,'START',begin);monkeypatch.setattr(audit,'UPPER',begin+6*T)
    return starts,starts+T,np.array([begin+2*T],np.int64)


def test_raw_raw_absence_and_declared_weekly_filter_are_distinguished(monkeypatch):
    starts,issue,missing=raw_compare_example(monkeypatch);extra=starts[0]+4*T;train=np.r_[starts,extra]
    monkeypatch.setattr(audit,'scheduled_weekly_closure_mask',lambda values:np.asarray(values.asi8==extra))
    r=audit.raw_compare_clocks(train,issue,missing)
    assert r['raw_TRAIN_rows']==4 and r['weekly_filter_removed_rows']==1 and r['canonical_TRAIN_rows']==3
    assert r['previous_missing_slots_absent_in_raw']==1 and r['previous_missing_slots_present_in_raw']==0
    assert r['previous_missing_slots_removed_by_weekly_filter']==0


@pytest.mark.parametrize('case',['lost','extra','shifted','issue_dtype','issue_shape','issue_domain','stale_missing'])
def test_raw_no_dropped_shifted_extra_or_invalid_clock_is_silently_accepted(monkeypatch,case):
    starts,issue,missing=raw_compare_example(monkeypatch)
    monkeypatch.setattr(audit,'scheduled_weekly_closure_mask',lambda values:np.zeros(len(values),np.bool_))
    if case=='lost':starts=starts[:-1]
    elif case=='extra':starts=np.sort(np.r_[starts,missing])
    elif case=='shifted':starts=starts+T
    elif case=='issue_dtype':issue=issue.astype(np.float64)
    elif case=='issue_shape':issue=issue[:-1]
    elif case=='issue_domain':issue=issue+10*T
    elif case=='stale_missing':missing=starts[:1]
    with pytest.raises(ValueError,match='RAW_CAPTURE_CLOCK|RAW_CANONICAL_PARITY|RAW_PRIOR_MISSING_SCOPE'):
        audit.raw_compare_clocks(starts,issue,missing)


def raw_lifecycle(monkeypatch,tmp_path):
    lifecycle(monkeypatch,tmp_path);monkeypatch.setattr(audit,'RAW_ROOT',tmp_path/'RAW_OUT');monkeypatch.setattr(audit,'RAW_PLAN',tmp_path/'RAW_PLAN.json')
    binding={'path':str(audit.RAW_PLAN),'sha256':'1'*64};pref={'path':str(tmp_path/'POLICY.json'),'sha256':'2'*64}
    monkeypatch.setattr(audit,'raw_inspect',lambda *a,**kw:({},binding,pref));monkeypatch.setattr(audit,'raw_checked_inputs',lambda p,**kw:None)
    original=audit.saved.ref;monkeypatch.setattr(audit,'ref',lambda p:pref if p==audit.POLICY else binding if p==audit.RAW_PLAN else original(p))
    return binding,pref


def test_raw_success_has_own_durable_claim_result_terminal_and_no_relaunch(monkeypatch,tmp_path):
    raw_lifecycle(monkeypatch,tmp_path);monkeypatch.setattr(audit,'raw_calculate',lambda p:{'synthetic':True});audit.raw_run('1'*64)
    terminal=json.loads((audit.RAW_ROOT/'TERMINAL.json').read_text());result=json.loads((audit.RAW_ROOT/'RESULT.json').read_text())
    assert terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False and terminal['status']=='COMPLETE'
    assert result['stage']==audit.RAW_STAGE and result['HTTP_calls']==result['price_computations']==result['models']==0
    assert not audit.ROOT.exists()
    with pytest.raises(FileExistsError):audit.raw_run('1'*64)


def test_raw_failure_consumes_claim_without_success_result(monkeypatch,tmp_path):
    raw_lifecycle(monkeypatch,tmp_path)
    def fail(p):raise ValueError('raw-hash-mismatch')
    monkeypatch.setattr(audit,'raw_calculate',fail)
    with pytest.raises(ValueError,match='raw-hash-mismatch'):audit.raw_run('1'*64)
    t=json.loads((audit.RAW_ROOT/'TERMINAL.json').read_text())
    assert t['status']=='FAILED_CONSUMED' and t['remaining_invocations']==0 and t['result']is None
    assert not(audit.RAW_ROOT/'RESULT.json').exists()


@pytest.mark.parametrize('changed',['source','policy','plan'])
def test_raw_source_policy_or_plan_bookend_change_cannot_publish_success(monkeypatch,tmp_path,changed):
    binding,pref=raw_lifecycle(monkeypatch,tmp_path);monkeypatch.setattr(audit,'raw_calculate',lambda p:{})
    if changed=='source':monkeypatch.setattr(audit,'source_clean',lambda p:False)
    else:
        original=audit.ref
        monkeypatch.setattr(audit,'ref',lambda p:{'path':str(p),'sha256':'9'*64}if p==(audit.POLICY if changed=='policy'else audit.RAW_PLAN)else original(p))
    with pytest.raises(ValueError,match='RAW_SOURCE_INPUT_PLAN_POLICY_BOOKEND'):audit.raw_run('1'*64)
    assert json.loads((audit.RAW_ROOT/'TERMINAL.json').read_text())['status']=='FAILED_CONSUMED'


def test_raw_preflight_decodes_no_gzip_or_array_and_creates_no_claim(monkeypatch,tmp_path,capsys):
    raw_lifecycle(monkeypatch,tmp_path);monkeypatch.setattr(audit,'raw_calculate',lambda p:pytest.fail('payload decoded'))
    monkeypatch.setattr(sys,'argv',['audit','--raw-feed-plan-sha256','1'*64,'--preflight']);audit.main();r=json.loads(capsys.readouterr().out)
    assert r['claim_created']is False and r['raw_gzip_bytes_decoded']==r['raw_JSON_objects_decoded']==r['capture_arrays_decoded']==0
    assert not audit.RAW_ROOT.exists()


@pytest.mark.parametrize('args',[
 ['--raw-feed-plan-sha256','1'*64,'--plan-sha256','2'*64],
 ['--preflight','--raw-feed-plan-sha256','1'*64],
 ['--raw-feed-plan-sha256','1'*64,'--preflight','--preflight']])
def test_raw_exact_cli_refuses_mixed_modes_reordered_or_duplicate_switches(monkeypatch,args):
    monkeypatch.setattr(sys,'argv',['audit']+args)
    with pytest.raises((SystemExit,ValueError)):audit.main()


def test_raw_gzip_identity_is_checked_before_current_decoder(monkeypatch,tmp_path):
    p=tmp_path/'chunk.json.gz';p.write_bytes(b'corrupt synthetic gzip');monkeypatch.setattr(audit.saved,'load_fields',lambda *a:{'issue_ns':np.array([T],np.int64)})
    from gx1.contracts import xau_tape_provenance_v1 as owner
    monkeypatch.setattr(owner,'_decode_source_chunk',lambda *a,**kw:pytest.fail('unverified bytes decoded'))
    plan={'inputs':{'capture':{}},'capture_array_headers':{},'TRAIN_issue_ns_sha256':audit.saved.array_sha(np.array([T],np.int64)),
        'raw_chunks':[{'file':{'path':str(p),'sha256':'0'*64},'metadata':{'size_bytes':p.stat().st_size}}]}
    with pytest.raises(ValueError,match='RAW_GZIP_SHA'):audit.raw_calculate(plan)


@pytest.mark.parametrize('bad_flag',list(audit.saved.FLAGS))
def test_raw_admission_requires_every_global_training_flag_false(monkeypatch,tmp_path,bad_flag):
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(audit,'REPO',tmp_path);monkeypatch.setattr(audit,'RAW_PLAN',tmp_path/'RAW_PLAN.json');monkeypatch.setattr(audit,'RAW_ROOT',tmp_path/'RAW_OUT')
    plan={'source_base_commit':'synthetic'};binding={'path':str(audit.RAW_PLAN),'sha256':'1'*64};pref={'path':str(tmp_path/'POLICY.json'),'sha256':'2'*64}
    policy={k:False for k in audit.saved.FLAGS};policy[bad_flag]=True;policy[audit.RAW_STAGE]=audit.raw_gate_expected(binding,True)
    monkeypatch.setattr(audit,'read',lambda b:plan if b==binding else policy);monkeypatch.setattr(audit,'ref',lambda p:pref)
    monkeypatch.setattr(audit,'raw_build_plan',lambda **kw:plan);monkeypatch.setattr(audit,'source_clean',lambda p:True)
    with pytest.raises(ValueError,match='RAW_EXACT_ONE_USE_ALL_TRAINING_CLOSED'):audit.raw_inspect('1'*64,admitted=True)



def raw_manifest_example(monkeypatch,tmp_path):
    import hashlib
    from gx1.contracts import xau_tape_provenance_v1 as owner
    monkeypatch.setattr(owner,'require_native_producer_source_inventory_metadata',lambda _:[])
    monkeypatch.setattr(audit,'RAW_FIRST',0);monkeypatch.setattr(audit,'RAW_LAST',1);monkeypatch.setattr(audit,'RAW_CHUNKS',2)
    monkeypatch.setattr(audit,'RAW_BEGIN','2021-01-04T00:00:00Z');monkeypatch.setattr(audit,'RAW_END','2021-01-04T00:30:00Z')
    chunks=[];(tmp_path/'source_chunks').mkdir()
    for i,(begin,end)in enumerate((('00:00','00:15'),('00:15','00:30'))):
        rel=f'source_chunks/chunk-{i:06d}.json.gz';raw=b'not-decoded-in-preflight';(tmp_path/rel).write_bytes(raw)
        chunks.append({'sequence':i,'request_from_utc':f'2021-01-04T{begin}:00Z','request_to_utc_exclusive':f'2021-01-04T{end}:00Z',
            'incomplete_candles':0,'response_candles':3,'complete_candles':3,'relative_path':rel,'size_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    monkeypatch.setattr(audit,'RAW_BYTES',sum(x['size_bytes']for x in chunks));monkeypatch.setattr(audit,'RAW_CANDLES',6)
    manifest={'instrument':'XAU_USD','timeframe':'M5','source_granularity':'M5','source_environment':'practice',
        'source_base_url':'https://api-fxpractice.oanda.com/v3','source_endpoint':'/instruments/XAU_USD/candles','prices':'MBA',
        'timestamp_semantics':'bar_start_utc','bar_duration_seconds':300,'decision_available_offset_seconds':300,
        'completion_field':'complete','completion_value':True,'market_closure_contract':audit.CLOSURE,'out_root':str(tmp_path),
        'source_chunks':chunks,'source_chunks_sha256':owner.canonical_json_sha256(chunks)}
    return manifest,{'path':str(tmp_path/'MANIFEST.json'),'sha256':'1'*64}


def test_raw_metadata_preflight_reuses_inventory_owner_without_decompressing_gzip(monkeypatch,tmp_path):
    manifest,binding=raw_manifest_example(monkeypatch,tmp_path);items,snapshots=audit.raw_manifest_inputs(manifest,binding)
    assert len(items)==2 and snapshots==[] and items[0]['metadata']['sequence']==0
    # The synthetic bytes are intentionally not gzip; metadata mode never opens them.
    assert audit.Path(items[0]['file']['path']).stat().st_size==items[0]['metadata']['size_bytes']


@pytest.mark.parametrize('case,match',[
 ('wrong_environment','RAW_MANIFEST_IDENTITY'),('wrong_M1','RAW_MANIFEST_IDENTITY'),('inventory_hash','RAW_CHUNK_INVENTORY_SHA'),
 ('sequence','RAW_REQUEST_CONTINUITY'),('gap','RAW_REQUEST_CONTINUITY'),('overlap','RAW_REQUEST_CONTINUITY'),
 ('TEST_end','RAW_REQUEST_DOMAIN_COMPLETE'),('incomplete','RAW_REQUEST_DOMAIN_COMPLETE'),('traversal','RAW_CHUNK_PATH'),
 ('size','RAW_FILE_PATH_SIZE'),('symlink','RAW_FILE_PATH_SIZE')])
def test_raw_request_metadata_refuses_gap_overlap_TEST_scope_and_unsafe_files(monkeypatch,tmp_path,case,match):
    from gx1.contracts import xau_tape_provenance_v1 as owner
    manifest,binding=raw_manifest_example(monkeypatch,tmp_path);item=manifest['source_chunks'][1]
    if case=='wrong_environment':manifest['source_environment']='live'
    elif case=='wrong_M1':manifest['source_granularity']='M1'
    elif case=='sequence':item['sequence']=2
    elif case=='gap':item['request_from_utc']='2021-01-04T00:20:00Z'
    elif case=='overlap':item['request_from_utc']='2021-01-04T00:10:00Z'
    elif case=='TEST_end':item['request_to_utc_exclusive']='2026-07-01T00:05:00Z'
    elif case=='incomplete':item['incomplete_candles']=1
    elif case=='traversal':item['relative_path']='../other.json.gz'
    elif case=='size':item['size_bytes']+=1
    elif case=='symlink':
        path=tmp_path/item['relative_path'];path.unlink();path.symlink_to(tmp_path/manifest['source_chunks'][0]['relative_path'])
    manifest['source_chunks_sha256']='0'*64 if case=='inventory_hash'else owner.canonical_json_sha256(manifest['source_chunks'])
    with pytest.raises(ValueError,match=match):audit.raw_manifest_inputs(manifest,binding)
