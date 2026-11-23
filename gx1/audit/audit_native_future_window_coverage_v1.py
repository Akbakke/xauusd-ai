"""One-use TRAIN clock/window and saved-response timestamp audits; no inferred closures."""
from __future__ import annotations
import argparse,json,signal,subprocess,time
from pathlib import Path
if __name__=='__main__':
    from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
    install_native_m5_direction_io_firewall()
import numpy as np
import pandas as pd
from gx1.audit import audit_native_selective_moves_v1 as saved
from gx1.contracts.xau_tape_provenance_v1 import scheduled_weekly_closure_mask
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution,require_capped_lock_ancestry

REPO,POLICY=saved.REPO,saved.POLICY
STAGE='native_future_window_coverage_20261123_001';BASENAME='NATIVE_FUTURE_WINDOW_COVERAGE_20261123_001'
PLAN=REPO/'configs/research'/(BASENAME+'.json');ROOT=Path('/home/andre2/GX1_RUNS')/BASENAME
SELF='gx1/audit/audit_native_future_window_coverage_v1.py';TEST='tests/test_audit_native_future_window_coverage_v1.py'
DESIGN={'path':'/home/andre2/GX1_RUNS/GX1_NATIVE_FUTURE_WINDOW_COVERAGE_PREPARATION_20261123_001/DESIGN001.json','sha256':'e37aefa0be7190b455510b2872015365c5d2492df0b08d14eaea7c6fe846abfe'}
CALENDAR={'path':'/home/andre2/GX1_RUNS/GX1_NATIVE_FUTURE_WINDOW_COVERAGE_PREPARATION_20261123_001/CALENDAR_SOURCE_LIMITS001.json','sha256':'d179d36365c9915305b3d4249ee5764a1202e8ba5f4048bc600d9883b046aeb0'}
FIELDS=('issue_ns','target_start_ns','validity_ns','target_present','mature','state_window_complete','state_expected_m5_count','state_missing_open_m5_count')
TF=('M5','M15','H1','H4','D1');ROWS=354550;START=1609459200000000000;UPPER=1767225600000000000;TICK=300000000000
CLOSURE='oanda_complete_true_scheduled_weekly_closure_excluded_v2'
OWNERS=(SELF,TEST,saved.SELF,saved.TEST,'gx1/audit/build_native_m5_direction_targets_v1.py','gx1/contracts/market_state_native_m5_targets_v1.py',
 'gx1/contracts/xau_tape_provenance_v1.py','gx1/contracts/native_m5_direction_io_firewall_v1.py','gx1/contracts/immutable_event_authority_v1.py',
 'gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh')
EXECUTION={'class':'audit','memory':'4G','swap':'512M','numerical_threads':1,'outer_seconds':600,'internal_seconds':570,'kill_after_seconds':10}
ref,read=saved.ref,saved.read


def need(ok,why):
    if not bool(ok):raise ValueError('NATIVE_FUTURE_WINDOW_COVERAGE_'+why)


def build_plan(*,source_base_commit=None):
    d=read(DESIGN);p=d['population'];budget=d['budget']
    need(d['schema_version']=='gx1_native_future_window_coverage_design_v1' and d['execution_authorized'] is False
         and d['capture_fields']==['issue_ns'] and d['target_fields']==list(FIELDS) and p['TRAIN_issue_rows']==ROWS
         and p['grid_start_ns']==START and p['cutoff_ns']==UPPER and p['tick_ns']==TICK and p['timeframes']==list(TF)
         and p['sampling'] is False and all(budget[k]==0 for k in ('prices','features','logits','models','checkpoint_payloads','fits','forwards','VAL_TEST_rows')),'LOCKED_DESIGN')
    join=read(d['clock_join_authority']);join_plan=read(join['plan']);j=join['report']
    need(join['status']=='COMPLETE_NATIVE_SELECTIVE_MOVE_AUDIT' and join['source_unchanged'] is True and join['test_data_used'] is False
         and j['full_clock_bytes_match'] is True and j['rows']==j['native_price_rows_decoded']==ROWS
         and j['native_price_columns']==['time','open','high','low','close'],'PRIOR_EXACT_NATIVE_CLOCK_JOIN')
    ip=join_plan['inputs'];capture_result=read(ip['capture_result']);capture=capture_result['report']['capture'];dataset=read(ip['dataset']);manifest=read(ip['native_manifest'])
    need(capture_result['status']=='COMPLETE_NATIVE_FULL_TRAIN_CAPTURE' and capture_result['source_unchanged'] is True
         and capture_result['test_data_used'] is False and capture==ip['capture']
         and dataset['native_manifest']==ip['native_manifest'] and dataset['cutoff_ns']==UPPER
         and manifest['market_closure_contract']==dataset['source_closure_contract']==dataset['contract']['source_closure_contract']==CLOSURE,'CANONICAL_SOURCE_TARGET_BINDING')
    ch=capture_result['report']['capture_array_headers'];th={k:dataset['array_headers'][k]for k in FIELDS}
    need(ch['issue_ns']==th['issue_ns'] and th['issue_ns']['shape']==[ROWS],'CLOCK_HEADERS')
    inputs={'design':DESIGN,'calendar_source_limits':CALENDAR,**{k:d[k]for k in ('alignment_result','clock_join_authority','coverage_requirements','observability_correction','source_coverage_review','original_goal')},
        'clock_join_plan':join['plan'],'capture_result':ip['capture_result'],'capture':capture,'dataset':ip['dataset'],'target_arrays':dataset['arrays'],'native_manifest':ip['native_manifest']}
    for key in ('alignment_result','coverage_requirements','observability_correction','source_coverage_review','calendar_source_limits'):read(inputs[key])
    return {'schema_version':'gx1_native_future_window_coverage_plan_v1','stage':STAGE,'status':'PREPARED_NOT_ADMITTED','plan_path':str(PLAN),'output_root':str(ROOT),
        'source_base_commit':source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'source_owners':{n:saved.native.sha(REPO/n)for n in OWNERS},'inputs':inputs,'target_contract':dataset['contract'],
        'capture_array_headers':ch,'target_array_headers':th,'TRAIN_issue_ns_sha256':dataset['TRAIN_issue_ns_sha256'],
        'execution':EXECUTION,'budget':budget,'question':d['question'],'runtime_versions':{'numpy':np.__version__,'pandas':pd.__version__}}


def source_clean(plan):
    return (subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='work/gx1-current'
        and subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_base_commit']
        and not subprocess.check_output(['git','status','--porcelain','--','gx1','scripts','tests'],cwd=REPO,text=True)
        and set(plan['source_owners'])==set(OWNERS) and all(saved.native.sha(REPO/k)==v for k,v in plan['source_owners'].items()))


def gate_expected(binding,admitted):
    return {'status':'ADMITTED_EXACT_CPU_ONLY_FUTURE_WINDOW_COVERAGE'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,
        'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}


def inspect(plan_sha,*,admitted):
    need(Path.cwd().resolve()==REPO,'CWD');binding={'path':str(PLAN),'sha256':plan_sha};plan=read(binding)
    need(plan==build_plan(source_base_commit=plan['source_base_commit']) and source_clean(plan),'EXACT_PLAN_SOURCE')
    pref=ref(POLICY);policy=read(pref)
    need(all(policy.get(k)is False for k in saved.FLAGS) and policy[STAGE]==gate_expected(binding,admitted),'EXACT_ONE_USE_ALL_TRAINING_CLOSED')
    need(not ROOT.exists() and not ROOT.is_symlink(),'FRESH_ROOT');return plan,binding,pref


def checked_inputs(plan):
    for b in plan['inputs'].values():need(ref(b['path'])==b,'INPUT_SHA')
    saved.archive_headers(plan['inputs']['capture'],plan['capture_array_headers'],exact_inventory=saved.CAPTURE_KEYS)
    saved.archive_headers(plan['inputs']['target_arrays'],plan['target_array_headers'])


def clock_grid(starts,*,begin,upper,closed):
    grid=np.arange(begin,upper,TICK,dtype=np.int64)
    need(closed.dtype==np.bool_ and closed.shape==grid.shape and np.all(np.diff(starts)>0)
         and np.all((starts>=begin)&(starts<upper)) and np.all((starts-begin)%TICK==0),'GRID_CLOCK')
    present=np.zeros(len(grid),np.bool_);present[((starts-begin)//TICK).astype(np.int64)]=True
    need(not np.any(present&closed),'OBSERVED_INSIDE_DECLARED_WEEKLY_CLOSURE')
    missing=~closed&~present
    return {'begin':begin,'upper':upper,'grid':grid,'present':present,'missing':missing,'closed':closed,
        'expected_prefix':np.r_[0,np.cumsum(~closed)],'missing_prefix':np.r_[0,np.cumsum(missing)]}


def validate_axes(t,contract):
    issue=t['issue_ns'];shape=(len(issue),5)
    need(issue.dtype==np.int64 and np.all(np.diff(issue)>0) and np.all((issue>=START)&(issue<UPPER)) and np.all(issue%TICK==0),'ISSUE_AXIS')
    for k in FIELDS[1:]:need(t[k].shape==shape and t[k].dtype==(np.bool_ if k in ('target_present','mature','state_window_complete')else np.int64),'TARGET_'+k)
    for j,tf in enumerate(TF):
        present=t['target_present'][:,j];mature=t['mature'][:,j];a=t['target_start_ns'][:,j];b=t['validity_ns'][:,j];duration=contract['tf_duration_ns'][tf];origin=contract['tf_origin_ns'][tf]
        need(np.all(a[~present]==-1) and np.all(b[~present]==-1) and np.all(a[present]>=issue[present])
             and np.all((a[present]-origin)%duration==0) and np.all(b[present]-a[present]==duration)
             and np.array_equal(mature,present&(b<UPPER)),'CANONICAL_WINDOW_AXIS_'+tf)
        need(np.all(t['state_expected_m5_count'][~mature,j]==0) and np.all(t['state_missing_open_m5_count'][~mature,j]==0)
             and not t['state_window_complete'][~mature,j].any(),'NONMATURE_STORAGE_'+tf)


def reconcile_windows(a,b,present,mature,stored_expected,stored_missing,stored_complete,grid):
    domain=(a>=grid['begin'])&(b<=grid['upper']);take=mature&domain
    pairs,inverse,weights=np.unique(np.column_stack((a[take],b[take])),axis=0,return_inverse=True,return_counts=True)
    need(np.all((pairs-grid['begin'])%TICK==0) and np.all(pairs[:,1]>pairs[:,0]),'WINDOW_GRID')
    x=((pairs-grid['begin'])//TICK).astype(np.int64)
    expected=grid['expected_prefix'][x[:,1]]-grid['expected_prefix'][x[:,0]];missing=grid['missing_prefix'][x[:,1]]-grid['missing_prefix'][x[:,0]]
    complete=(expected>0)&(missing==0)
    for name,stored,actual in (('EXPECTED',stored_expected,expected),('MISSING',stored_missing,missing),('COMPLETE',stored_complete,complete)):
        bad=np.flatnonzero(stored[take]!=actual[inverse]);need(not len(bad),name+'_EXACT_MISMATCH_'+str(len(bad)))
    all_pairs=np.unique(np.column_stack((a[present],b[present])),axis=0)
    report={'all_issue_cells':len(a),'target_present_issues':int(present.sum()),'mature_issues':int(mature.sum()),'nonmature_issues':int((~mature).sum()),
        'unique_present_windows':len(all_pairs),
        'unique_nonmature_present_windows':len(np.unique(np.column_stack((a[present&~mature],b[present&~mature])),axis=0)),
        'unique_mature_out_of_clock_domain_windows':len(np.unique(np.column_stack((a[mature&~domain],b[mature&~domain])),axis=0)),
        'compared_issues':int(take.sum()),'unique_compared_windows':len(pairs),
        'mature_out_of_clock_domain_issues':int((mature&~domain).sum()),'all_mature_exactly_reconciled':bool(not np.any(mature&~domain)),
        'stored_complete_issues':int(stored_complete.sum()),'reproduced_complete_compared_issues':int(weights[complete].sum()),'reproduced_complete_unique_windows':int(complete.sum()),
        'expected_open_slots_sum_unique_windows':int(expected.sum()),'missing_open_slots_sum_unique_windows':int(missing.sum()),
        'expected_open_slots_sum_issued_cells':int(np.dot(weights,expected)),'missing_open_slots_sum_issued_cells':int(np.dot(weights,missing)),
        'zero_expected_unique_windows':int((expected==0).sum()),'missing_unique_windows':int((missing>0).sum()),'exact_compared_values':True,
        'unique_complete_rate':float(complete.mean())if len(pairs)else None,'issue_complete_rate':float(weights[complete].sum()/weights.sum())if weights.sum()else None}
    return report,{'pairs':pairs,'weights':weights,'expected':expected,'missing':missing,'complete':complete}


def timing_histogram(clocks,zone):
    local=pd.to_datetime(clocks,utc=True).tz_convert(zone);weekday=np.asarray(local.weekday);hour=np.asarray(local.hour)
    return {str(d):{str(h):int(((weekday==d)&(hour==h)).sum())for h in range(24)}for d in range(7)}


def missing_patterns(clocks):
    begins=np.r_[0,np.flatnonzero(np.diff(clocks)!=TICK)+1]if len(clocks)else np.empty(0,np.int64);ends=np.r_[begins[1:],len(clocks)]if len(clocks)else begins
    gaps=[];hist={}
    for a,b in zip(begins,ends):
        first,last=int(clocks[a]),int(clocks[b-1]+TICK);minutes=int((last-first)//60000000000);key=str(minutes);hist[key]=hist.get(key,0)+1
        gaps.append({'start_utc':pd.Timestamp(first,tz='UTC').isoformat(),'end_utc_exclusive':pd.Timestamp(last,tz='UTC').isoformat(),
            'start_New_York':pd.Timestamp(first,tz='UTC').tz_convert('America/New_York').isoformat(),
            'end_New_York_exclusive':pd.Timestamp(last,tz='UTC').tz_convert('America/New_York').isoformat(),'missing_slots':int(b-a),'duration_minutes':minutes})
    return {'missing_expected_open_slots':len(clocks),'all_missing_clocks_utc':[x.isoformat()for x in pd.to_datetime(clocks,utc=True)],
        'weekday_hour_counts':{z:timing_histogram(clocks,z)for z in ('UTC','America/New_York')},'contiguous_gap_duration_minutes_histogram':hist,'gaps':gaps,
        'meaning':'Absent from the previously verified native TRAIN clock while not excluded by the bound weekly mask; no daily/holiday closure or source-failure cause inferred.'}


def window_weekdays(windows):
    report={}
    for zone in ('UTC','America/New_York'):
        report[zone]={}
        for name,j in (('target_start',0),('target_end',1)):
            weekday=np.asarray(pd.to_datetime(windows['pairs'][:,j],utc=True).tz_convert(zone).weekday);rows={}
            for day in range(7):
                take=weekday==day;good=take&windows['complete'];missing=take&(windows['missing']>0);weights=windows['weights']
                rows[str(day)]={'unique_windows':int(take.sum()),'issued_cells':int(weights[take].sum()),'complete_unique':int(good.sum()),'complete_issues':int(weights[good].sum()),
                    'missing_unique':int(missing.sum()),'missing_issues':int(weights[missing].sum()),'missing_slots_sum_unique':int(windows['missing'][take].sum())}
            report[zone][name]=rows
    return report


def calculate(plan):
    capture=saved.load_fields(plan['inputs']['capture'],('issue_ns',),plan['capture_array_headers']);t=saved.load_fields(plan['inputs']['target_arrays'],FIELDS,plan['target_array_headers']);issue=capture['issue_ns']
    need(issue.shape==(ROWS,) and np.array_equal(issue,t['issue_ns']) and saved.array_sha(issue)==plan['TRAIN_issue_ns_sha256'],'FULL_NATIVE_CLOCK_JOIN');validate_axes(t,plan['target_contract'])
    starts=issue-TICK;grid_ns=np.arange(START,UPPER,TICK,dtype=np.int64);closed=np.asarray(scheduled_weekly_closure_mask(pd.to_datetime(grid_ns,utc=True)),np.bool_)
    grid=clock_grid(starts,begin=START,upper=UPPER,closed=closed);report={}
    for j,tf in enumerate(TF):
        summary,windows=reconcile_windows(t['target_start_ns'][:,j],t['validity_ns'][:,j],t['target_present'][:,j],t['mature'][:,j],
            t['state_expected_m5_count'][:,j],t['state_missing_open_m5_count'][:,j],t['state_window_complete'][:,j],grid)
        if tf=='D1':summary['mature_window_start_and_end_weekdays']=window_weekdays(windows)
        report[tf]=summary
    return {'all_five_timeframes':report,'TRAIN_issue_rows':ROWS,'grid_counts':{'calendar_slots':len(grid_ns),'declared_weekly_closed':int(closed.sum()),
        'expected_open':int((~closed).sum()),'observed_native':len(starts),'missing_expected_open':int(grid['missing'].sum())},
        'weekday_numbering':{'0':'Monday','1':'Tuesday','2':'Wednesday','3':'Thursday','4':'Friday','5':'Saturday','6':'Sunday'},
        'missing_clock_patterns':missing_patterns(grid_ns[grid['missing']]),'side_known':'NOT_READ_NOT_IN_BUDGET',
        'payload_counts':{'capture_archives':1,'capture_fields':['issue_ns'],'target_archives':1,'target_fields':list(FIELDS)},
        'limits':['No inferred calendar, label rewrite, source repair, price/feature/logit/model/checkpoint read or fit.',
            'Immature/out-of-clock-domain windows are reported separately and never filled from a TRAIN projection.',
            'Repeated issue labels and unique target windows are reported separately, not independent events; temporal pattern alone does not prove its cause.']}


def publish(name,value):
    path=ROOT/name;saved._publish_bytes(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());return ref(path)


def run(plan_sha):
    began=time.monotonic();cap=require_capped_cpu_audit_execution();lock=require_capped_lock_ancestry()
    need(cap['memory_max']==cap['memory_high']==4*1024**3 and cap['swap']==512*1024**2 and cap['numerical_threads']==1 and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):],'CAP_LIMITS')
    plan,binding,pref=inspect(plan_sha,admitted=True);checked_inputs(plan);ROOT.mkdir(parents=True,exist_ok=False)
    claim=publish('CLAIM.json',{'stage':STAGE,'plan':binding,'policy_at_claim':pref,'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False});old=signal.getsignal(signal.SIGALRM);result=None
    try:
        def expired(*_):raise TimeoutError('CLOCK_WINDOW_WALL')
        signal.signal(signal.SIGALRM,expired);remaining=EXECUTION['internal_seconds']-(time.monotonic()-began);need(remaining>0,'TIME_BEFORE_CALCULATE');signal.setitimer(signal.ITIMER_REAL,remaining)
        report=calculate(plan);checked_inputs(plan);need(source_clean(plan) and ref(POLICY)==pref and ref(PLAN)==binding,'SOURCE_INPUT_PLAN_POLICY_BOOKEND')
        result=publish('RESULT.json',{'status':'COMPLETE_NATIVE_FUTURE_WINDOW_COVERAGE','stage':STAGE,'plan':binding,'claim':claim,'policy_at_claim':pref,'design':DESIGN,
            'report':report,'source_unchanged':True,'test_data_used':False,'models':0,'fits':0,'forwards':0,'prices':0,'logits':0,'elapsed_seconds':time.monotonic()-began})
        terminal=publish('TERMINAL.json',{'status':'COMPLETE','exit_code':0,'stage':STAGE,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'source_unchanged':True,'test_data_used':False});return {'result':result,'terminal':terminal}
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0);publish('TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'stage':STAGE,'plan':binding,'claim':claim,'result':result,
            'remaining_invocations':0,'relaunch_allowed':False,'error':type(error).__name__+':'+str(error)[:2000],'test_data_used':False});raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


# Separate read-only raw-response qualification; the consumed clock001 path stays unchanged.
RAW_STAGE='native_raw_feed_clock_qualification_20261123_001'
RAW_BASENAME='NATIVE_RAW_FEED_CLOCK_QUALIFICATION_20261123_001'
RAW_PLAN=REPO/'configs/research'/(RAW_BASENAME+'.json');RAW_ROOT=Path('/home/andre2/GX1_RUNS')/RAW_BASENAME
RAW_DESIGN={'path':'/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001/RAW_FEED_CLOCK_DESIGN001.json','sha256':'896205aafbeca80264ea37af039407bbb80beaff9d4e988adfdf8fce7fc64006'}
RAW_FIRST=282;RAW_LAST=403;RAW_CHUNKS=122;RAW_BYTES=15680098;RAW_CANDLES=355366
RAW_BEGIN='2020-12-30T00:00:00Z';RAW_END='2026-01-03T00:00:00Z';RAW_TEST='2026-07-01T00:00:00Z'


def raw_time(value,label):
    from gx1.contracts.xau_tape_provenance_v1 import _utc_native_bar
    need(type(value)is str and (value.endswith('Z')or value.endswith('+00:00')),'RAW_EXPLICIT_UTC:'+label)
    return int(_utc_native_bar(value,timeframe='M5',label=label).value)


def raw_file(path,size):
    path=Path(path)
    need(path.is_absolute() and not path.is_symlink() and path.resolve()==path and path.is_file()
         and type(size)is int and size>0 and path.stat().st_size==size,'RAW_FILE_PATH_SIZE')
    return path


def raw_manifest_inputs(manifest,binding):
    from gx1.contracts.xau_tape_provenance_v1 import canonical_json_sha256,require_native_producer_source_inventory_metadata
    need(manifest['instrument']=='XAU_USD' and manifest['timeframe']==manifest['source_granularity']=='M5'
         and manifest['source_environment']=='practice' and manifest['source_base_url']=='https://api-fxpractice.oanda.com/v3'
         and manifest['source_endpoint']=='/instruments/XAU_USD/candles' and manifest['prices']=='MBA'
         and manifest['timestamp_semantics']=='bar_start_utc' and manifest['bar_duration_seconds']==300
         and manifest['decision_available_offset_seconds']==300 and manifest['completion_field']=='complete'
         and manifest['completion_value']is True and manifest['market_closure_contract']==CLOSURE,'RAW_MANIFEST_IDENTITY')
    root=Path(binding['path']).parent;need(str(root)==manifest['out_root'] and root.resolve()==root,'RAW_MANIFEST_ROOT')
    chunks=manifest['source_chunks'];need(canonical_json_sha256(chunks)==manifest['source_chunks_sha256'],'RAW_CHUNK_INVENTORY_SHA')
    selected=chunks[RAW_FIRST:RAW_LAST+1];need(len(selected)==RAW_CHUNKS,'RAW_CHUNK_COUNT');descriptors=[]
    previous=None
    for seq,item in enumerate(selected,RAW_FIRST):
        begin,end=(raw_time(item[k],k)for k in ('request_from_utc','request_to_utc_exclusive'))
        need(item['sequence']==seq and begin<end and (previous is None or begin==previous),'RAW_REQUEST_CONTINUITY')
        need(end<=raw_time(RAW_TEST,'TEST_BOUNDARY') and item['incomplete_candles']==0
             and type(item['response_candles'])is int and item['response_candles']==item['complete_candles']>0,'RAW_REQUEST_DOMAIN_COMPLETE')
        relative=f'source_chunks/chunk-{seq:06d}.json.gz';need(item['relative_path']==relative,'RAW_CHUNK_PATH')
        path=raw_file(root/relative,item['size_bytes']);descriptors.append({'file':{'path':str(path),'sha256':item['sha256']},'metadata':item})
        previous=end
    need(raw_time(selected[0]['request_from_utc'],'FIRST')==raw_time(RAW_BEGIN,'FIRST')
         and previous==raw_time(RAW_END,'LAST') and sum(x['size_bytes']for x in selected)==RAW_BYTES
         and sum(x['response_candles']for x in selected)==RAW_CANDLES,'RAW_REQUEST_LOCKED_POPULATION')
    producer=[]
    for item in require_native_producer_source_inventory_metadata(manifest):
        path=raw_file(root/item['snapshot_relative_path'],item['size_bytes']);proof=ref(path)
        need(proof['sha256']==item['sha256'],'RAW_PRODUCER_SNAPSHOT_SHA');producer.append(proof)
        if item['repo_relative_path']=='gx1/contracts/xau_tape_provenance_v1.py':
            need(saved.native.sha(REPO/item['repo_relative_path'])==item['sha256'],'RAW_CURRENT_CANONICAL_DECODER_IDENTITY')
    return descriptors,producer


def raw_build_plan(*,source_base_commit=None):
    d=read(RAW_DESIGN);scope=d['source_scope'];budget=d['budget']
    need(d['schema_version']=='gx1_native_raw_feed_clock_qualification_design_v1' and d['execution_authorized']is False
         and d['execution']=={**EXECUTION,'all5_training_flags_false':True,'invocations':1}
         and scope['TRAIN_issue_start_ns']==START and scope['TRAIN_issue_end_exclusive_ns']==UPPER
         and scope['capture_fields']==['issue_ns'] and scope['first_sequence']==RAW_FIRST and scope['last_sequence']==RAW_LAST
         and scope['saved_raw_responses']==budget['saved_gzip_responses']==RAW_CHUNKS
         and scope['compressed_bytes']==RAW_BYTES and scope['response_candles_from_metadata']==RAW_CANDLES
         and scope['incomplete_from_metadata']==0 and scope['instrument']=='XAU_USD' and scope['granularity']=='M5'
         and scope['price']=='MBA' and scope['source_environment']=='practice'
         and scope['request_start_utc']==RAW_BEGIN and scope['request_end_utc_exclusive']==RAW_END
         and scope['sealed_TEST_start_utc']==RAW_TEST and scope['capture_archive_loads']==budget['capture_archive_loads']==1
         and budget['capture_fields_decoded']==1 and budget['full_raw_JSON_candle_objects_decoded']is True
         and budget['boundary_nonTRAIN_objects_decoded']is True
         and all(budget[k]==0 for k in ('HTTP_calls','price_computations','models','fits','forwards','backward','features','checkpoint_loads','VAL_TEST_prediction_or_score_rows')),'RAW_LOCKED_DESIGN')
    manifest=read(d['native_manifest']);descriptors,producer=raw_manifest_inputs(manifest,d['native_manifest'])
    capture_result=read(d['capture_result']);clock=read(d['clock_result'])
    need(capture_result['status']=='COMPLETE_NATIVE_FULL_TRAIN_CAPTURE' and capture_result['source_unchanged']is True
         and capture_result['test_data_used']is False and capture_result['report']['rows']==ROWS
         and clock['status']=='COMPLETE_NATIVE_FUTURE_WINDOW_COVERAGE' and clock['source_unchanged']is True
         and clock['test_data_used']is False and clock['report']['TRAIN_issue_rows']==ROWS
         and clock['report']['grid_counts']['missing_expected_open']==17870,'RAW_AUTHENTIC_CAPTURE_CLOCK')
    clock_plan=read(clock['plan']);capture=capture_result['report']['capture'];headers=capture_result['report']['capture_array_headers']
    need(clock_plan['inputs']['capture_result']==d['capture_result'] and clock_plan['inputs']['capture']==capture
         and clock_plan['inputs']['native_manifest']==d['native_manifest']
         and headers['issue_ns']==clock_plan['capture_array_headers']['issue_ns']
         and headers['issue_ns']['dtype']=='<i8' and headers['issue_ns']['shape']==[ROWS],'RAW_PRIOR_CLOCK_CHAIN')
    read(d['qualification_requirements'])
    inputs={'design':RAW_DESIGN,**{k:d[k]for k in ('native_manifest','capture_result','clock_result','qualification_requirements')},'clock_plan':clock['plan'],'capture':capture}
    return {'schema_version':'gx1_native_raw_feed_clock_qualification_plan_v1','stage':RAW_STAGE,'status':'PREPARED_NOT_ADMITTED',
        'plan_path':str(RAW_PLAN),'output_root':str(RAW_ROOT),'source_base_commit':source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'source_owners':{n:saved.native.sha(REPO/n)for n in OWNERS},'inputs':inputs,'raw_chunks':descriptors,'producer_source_snapshots':producer,
        'capture_array_headers':headers,'TRAIN_issue_ns_sha256':headers['issue_ns']['sha256'],'execution':EXECUTION,'budget':budget,
        'question':d['question'],'runtime_versions':{'numpy':np.__version__,'pandas':pd.__version__}}


def raw_gate_expected(binding,admitted):
    return {'status':'ADMITTED_EXACT_CPU_ONLY_RAW_FEED_CLOCK_QUALIFICATION'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,
        'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}


def raw_inspect(plan_sha,*,admitted):
    need(Path.cwd().resolve()==REPO,'RAW_CWD');need(type(plan_sha)is str and len(plan_sha)==64 and all(x in '0123456789abcdef'for x in plan_sha),'RAW_PLAN_SHA')
    binding={'path':str(RAW_PLAN),'sha256':plan_sha};plan=read(binding)
    need(plan==raw_build_plan(source_base_commit=plan['source_base_commit']) and source_clean(plan),'RAW_EXACT_PLAN_SOURCE')
    pref=ref(POLICY);policy=read(pref)
    need(all(policy.get(k)is False for k in saved.FLAGS) and policy[RAW_STAGE]==raw_gate_expected(binding,admitted),'RAW_EXACT_ONE_USE_ALL_TRAINING_CLOSED')
    need(not RAW_ROOT.exists() and not RAW_ROOT.is_symlink(),'RAW_FRESH_ROOT');return plan,binding,pref


def raw_checked_inputs(plan,*,payload_hashes=False):
    for binding in plan['inputs'].values():need(ref(binding['path'])==binding,'RAW_INPUT_SHA')
    for binding in plan['producer_source_snapshots']:need(ref(binding['path'])==binding,'RAW_PRODUCER_SNAPSHOT_BOOKEND')
    for item in plan['raw_chunks']:
        path=raw_file(item['file']['path'],item['metadata']['size_bytes'])
        if payload_hashes:need(ref(path)==item['file'],'RAW_GZIP_BOOKEND_SHA')
    saved.archive_headers(plan['inputs']['capture'],{'issue_ns':plan['capture_array_headers']['issue_ns']},exact_inventory=saved.CAPTURE_KEYS)


def raw_parse_chunk(payload,metadata):
    need(type(payload)is dict and set(payload)=={'schema_version','request','response'}
         and payload['schema_version']=='xau_oanda_native_source_chunk_v2','RAW_CHUNK_SCHEMA')
    request,response=payload['request'],payload['response'];seq=metadata['sequence']
    need(type(request)is dict and set(request)=={'instrument','from','to','granularity','price'}
         and request['instrument']=='XAU_USD' and request['granularity']=='M5' and request['price']=='MBA','RAW_REQUEST_IDENTITY')
    begin,end=(raw_time(metadata[k],k)for k in ('request_from_utc','request_to_utc_exclusive'))
    need(raw_time(request['from'],'REQUEST_FROM')==begin and raw_time(request['to'],'REQUEST_TO')==end,'RAW_REQUEST_WINDOW')
    need(type(response)is dict and response.get('instrument')=='XAU_USD' and response.get('granularity')=='M5'
         and type(response.get('candles'))is list,'RAW_RESPONSE_IDENTITY')
    candles=response['candles'];need(len(candles)==metadata['response_candles']==metadata['complete_candles'] and metadata['incomplete_candles']==0,'RAW_RESPONSE_COUNT')
    times=[]
    for i,candle in enumerate(candles):
        need(type(candle)is dict and type(candle.get('complete'))is bool and candle['complete']is True,f'RAW_COMPLETE:chunk={seq},row={i}')
        tick=raw_time(candle.get('time'),f'CHUNK_{seq}_ROW_{i}')
        need(begin<=tick<end,f'RAW_CANDLE_REQUEST_DOMAIN:chunk={seq},row={i}');times.append(tick)
    times=np.asarray(times,dtype=np.int64);diff=np.diff(times)
    need(not np.any(diff<=0),f'RAW_CHUNK_ORDER:chunk={seq},duplicates={int((diff==0).sum())},backwards={int((diff<0).sum())}')
    need(len(times)>0 and times[0]==raw_time(metadata['first_complete_utc'],'FIRST_COMPLETE')
         and times[-1]==raw_time(metadata['last_complete_utc'],'LAST_COMPLETE'),'RAW_RESPONSE_BOUNDARIES')
    return times


def raw_compare_clocks(train,issue,missing):
    need(issue.dtype==np.dtype('int64') and issue.shape==(ROWS,) and np.all(np.diff(issue)>0)
         and np.all((issue>=START)&(issue<UPPER)),'RAW_CAPTURE_CLOCK')
    starts=issue-TICK;closed=np.asarray(scheduled_weekly_closure_mask(pd.to_datetime(train,utc=True)),np.bool_);canonical=train[~closed]
    missing_native=np.setdiff1d(starts,canonical);extra=np.setdiff1d(canonical,starts)
    need(np.array_equal(canonical,starts),f'RAW_CANONICAL_PARITY:missing={len(missing_native)},extra={len(extra)}')
    need(not np.isin(missing,starts).any() and not np.asarray(scheduled_weekly_closure_mask(pd.to_datetime(missing,utc=True))).any(),'RAW_PRIOR_MISSING_SCOPE')
    in_raw=np.isin(missing,train);in_canonical=np.isin(missing,canonical)
    return {'raw_TRAIN_rows':len(train),'weekly_filter_removed_rows':int(closed.sum()),'canonical_TRAIN_rows':len(canonical),
        'canonical_equals_capture_bar_starts':True,'canonical_bar_start_sha256':saved.array_sha(canonical),'capture_issue_sha256':saved.array_sha(issue),
        'missing_from_canonical':len(missing_native),'extra_in_canonical':len(extra),'previous_missing_expected_open_slots':len(missing),
        'previous_missing_slots_present_in_raw':int(in_raw.sum()),'previous_missing_slots_absent_in_raw':int((~in_raw).sum()),
        'previous_missing_slots_present_after_weekly_filter':int(in_canonical.sum()),'previous_missing_slots_removed_by_weekly_filter':int((in_raw&~in_canonical).sum())}


def raw_calculate(plan):
    import hashlib
    from gx1.contracts.xau_tape_provenance_v1 import _decode_source_chunk
    issue=saved.load_fields(plan['inputs']['capture'],('issue_ns',),plan['capture_array_headers'])['issue_ns']
    need(saved.array_sha(issue)==plan['TRAIN_issue_ns_sha256'],'RAW_CAPTURE_CLOCK_SHA');all_times=[];reports=[];previous=None;decoded_bytes=0
    for item in plan['raw_chunks']:
        binding,metadata=item['file'],item['metadata'];path=raw_file(binding['path'],metadata['size_bytes']);raw=path.read_bytes()
        need(hashlib.sha256(raw).hexdigest()==binding['sha256'],'RAW_GZIP_SHA')
        payload=_decode_source_chunk(raw,timeframe='M5',path=path);times=raw_parse_chunk(payload,metadata)
        need(previous is None or times[0]>previous,f'RAW_CROSS_CHUNK_ORDER:sequence={metadata["sequence"]},previous={previous},first={int(times[0])}')
        previous=int(times[-1]);all_times.append(times);decoded_bytes+=len(raw)
        reports.append({'sequence':metadata['sequence'],'candle_objects':len(times),'before_TRAIN':int((times<START).sum()),'after_TRAIN':int((times>=UPPER).sum()),'duplicates':0,'backwards':0})
        del payload,raw
    times=np.concatenate(all_times);need(len(times)==RAW_CANDLES and decoded_bytes==RAW_BYTES and np.all(times<raw_time(RAW_TEST,'SEALED_TEST')),'RAW_DECODE_BUDGET_TEST')
    train=times[(times>=START)&(times<UPPER)];clock=read(plan['inputs']['clock_result'])['report'];old=clock['missing_clock_patterns']
    missing=pd.DatetimeIndex(pd.to_datetime(old['all_missing_clocks_utc'],utc=True)).asi8
    comparison=raw_compare_clocks(train,issue,missing);groups={'Mon_Thu_17_18_NY':[],'Sunday_2021_11_21_18_19_NY':[],'other':[]}
    for gap in old['gaps']:
        start=pd.Timestamp(gap['start_New_York']);end=pd.Timestamp(gap['end_New_York_exclusive'])
        key=('Mon_Thu_17_18_NY'if gap['duration_minutes']==60 and start.dayofweek<4 and (start.hour,start.minute,end.hour,end.minute)==(17,0,18,0)
             else'Sunday_2021_11_21_18_19_NY'if gap['start_utc']=='2021-11-21T23:00:00+00:00'and gap['duration_minutes']==60 else'other')
        ticks=np.arange(pd.Timestamp(gap['start_utc']).value,pd.Timestamp(gap['end_utc_exclusive']).value,TICK,dtype=np.int64)
        positions=np.searchsorted(train,ticks);inside=positions<len(train);inside[inside]&=train[positions[inside]]==ticks[inside]
        groups[key].append({'slots':len(ticks),'present_in_raw':int(inside.sum())})
    group_summary={k:{'gaps':len(v),'missing_slots':sum(x['slots']for x in v),'slots_present_in_saved_raw':sum(x['present_in_raw']for x in v)}for k,v in groups.items()}
    need(len(missing)==17870 and group_summary['Mon_Thu_17_18_NY']['gaps']==994 and group_summary['Sunday_2021_11_21_18_19_NY']['gaps']==1,'RAW_PRIOR_GAP_POPULATION')
    return {'comparison':comparison,'gap_groups':group_summary,'raw_chunks':reports,'raw_response_request_windows_contiguous':True,'cross_chunk_duplicates':0,'cross_chunk_overlaps':0,
        'payload_counts':{'saved_gzip_responses_decoded':len(reports),'compressed_gzip_bytes_decoded':decoded_bytes,'full_raw_JSON_candle_objects_decoded':len(times),
            'raw_OHLC_objects_deserialized':True,'before_TRAIN_boundary_objects':int((times<START).sum()),'after_TRAIN_boundary_objects':int((times>=UPPER).sum()),
            'sealed_TEST_objects':0,'capture_archives':1,'capture_fields':['issue_ns'],'price_computations':0,'VAL_TEST_prediction_or_score_rows':0},
        'limits':['This proves the content of the archived successful practice responses, not historical quote availability or market closure.',
            'Full gzip/JSON decoding reads OHLC objects including admitted2020/2026 boundary objects; price fields are not used for computation.',
            'Only the existing manifest-bound weekly filter is applied; no daily closure is inferred and no original data or masks are changed.']}


def raw_publish(name,value):
    path=RAW_ROOT/name;saved._publish_bytes(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());return ref(path)


def raw_run(plan_sha):
    began=time.monotonic();cap=require_capped_cpu_audit_execution();lock=require_capped_lock_ancestry()
    need(cap['memory_max']==cap['memory_high']==4*1024**3 and cap['swap']==512*1024**2 and cap['numerical_threads']==1 and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):],'RAW_CAP_LIMITS')
    plan,binding,pref=raw_inspect(plan_sha,admitted=True);raw_checked_inputs(plan);RAW_ROOT.mkdir(parents=True,exist_ok=False)
    claim=raw_publish('CLAIM.json',{'stage':RAW_STAGE,'plan':binding,'policy_at_claim':pref,'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False});old=signal.getsignal(signal.SIGALRM);result=None
    try:
        def expired(*_):raise TimeoutError('RAW_FEED_CLOCK_WALL')
        signal.signal(signal.SIGALRM,expired);remaining=EXECUTION['internal_seconds']-(time.monotonic()-began);need(remaining>0,'RAW_TIME_BEFORE_CALCULATE');signal.setitimer(signal.ITIMER_REAL,remaining)
        report=raw_calculate(plan);raw_checked_inputs(plan,payload_hashes=True)
        need(source_clean(plan) and ref(POLICY)==pref and ref(RAW_PLAN)==binding,'RAW_SOURCE_INPUT_PLAN_POLICY_BOOKEND')
        result=raw_publish('RESULT.json',{'status':'COMPLETE_NATIVE_RAW_FEED_CLOCK_QUALIFICATION','stage':RAW_STAGE,'plan':binding,'claim':claim,'policy_at_claim':pref,'design':RAW_DESIGN,
            'report':report,'source_unchanged':True,'test_data_used':False,'models':0,'fits':0,'forwards':0,'price_computations':0,'HTTP_calls':0,'elapsed_seconds':time.monotonic()-began})
        terminal=raw_publish('TERMINAL.json',{'status':'COMPLETE','exit_code':0,'stage':RAW_STAGE,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'source_unchanged':True,'test_data_used':False});return {'result':result,'terminal':terminal}
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0);raw_publish('TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'stage':RAW_STAGE,'plan':binding,'claim':claim,'result':result,
            'remaining_invocations':0,'relaunch_allowed':False,'error':type(error).__name__+':'+str(error)[:2000],'test_data_used':False});raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan-sha256');mode.add_argument('--raw-feed-plan-sha256');parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.raw_feed_plan_sha256 is not None:
        import sys
        need(sys.argv[1:]==['--raw-feed-plan-sha256',args.raw_feed_plan_sha256]+(['--preflight']if args.preflight else[]),'RAW_EXACT_CLI')
        if args.preflight:
            plan,binding,pref=raw_inspect(args.raw_feed_plan_sha256,admitted=False);raw_checked_inputs(plan)
            print(json.dumps({'status':'PREFLIGHT_RAW_FEED_PASS','plan':binding,'policy':pref,'claim_created':False,'raw_gzip_bytes_decoded':0,'raw_JSON_objects_decoded':0,'capture_arrays_decoded':0,'models':0,'price_computations':0}))
        else:print(json.dumps(raw_run(args.raw_feed_plan_sha256)))
    elif args.preflight:
        plan,binding,pref=inspect(args.plan_sha256,admitted=False);checked_inputs(plan);print(json.dumps({'status':'PREFLIGHT_PASS','plan':binding,'policy':pref,'claim_created':False,'arrays_decoded':0,'models':0,'prices':0,'logits':0}))
    else:print(json.dumps(run(args.plan_sha256)))

if __name__=='__main__':main()
