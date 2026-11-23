"""Bounded serving-bootstrap exclusion witness; no wholeTRAIN replay or source-known claim."""
from __future__ import annotations
import argparse,hashlib,json,os,signal,subprocess,sys,time
from dataclasses import asdict
from pathlib import Path
if __name__=='__main__':
    from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
    install_native_m5_direction_io_firewall()
import numpy as np
import pandas as pd
from gx1.audit import audit_native_macro_source_keys_v1 as keys_owner
from gx1.audit import qualify_market_state_native_m5_pipeline_v1 as pipeline
from gx1.features import htf_features as htf,level_registry_v1 as levels,volatility_squeeze_state_v1 as squeeze
from gx1.features.technical_indicators_v1 import ema50_200_spread_atr_block
from gx1.contracts import gx1_capped_execution_v1 as guard
REPO,POLICY=keys_owner.REPO,keys_owner.POLICY
STAGE='native_observation_prefix_witness_20261123_001';BASENAME='NATIVE_OBSERVATION_PREFIX_WITNESS_20261123_001'
PLAN=REPO/'configs/research'/(BASENAME+'.json');ROOT=Path('/home/andre2/GX1_RUNS')/BASENAME
SELF='gx1/audit/audit_native_observation_prefix_v1.py';TEST='tests/test_audit_native_observation_prefix_v1.py'
DESIGN={'path':str(keys_owner.prior.EROOT/'OBSERVATION_PREFIX_WITNESS_DESIGN002.json'),'sha256':'3c17acf72811011bb49ff9775b86914b8c5fc8d8878d2352905b6efb8a7e5964'}
TF=('M5','M15','H1','H4','D1');CUTOFF=1262304000000000000;RAW_ROWS=42471;CHUNK=128
CAPTURE_FIELDS=('source_rows','issue_ns');FEATURE_FIELDS=('bar_start_ns','retained_source_rows',*[f'tf_{tf}_{key}'for tf in TF for key in ('ts_int64','owner_row_indices')])
EXECUTION={'class':'audit','memory':'4G','swap':'512M','numerical_threads':1,'outer_seconds':600,'internal_seconds':570,'kill_after_seconds':10}
OWNERS=tuple(dict.fromkeys((*keys_owner.OWNERS,SELF,TEST,'gx1/features/level_registry_v1.py','gx1/features/technical_indicators_v1.py','gx1/features/volatility_squeeze_state_v1.py','gx1/features/htf_features.py')))
ref,read,array_sha=keys_owner.ref,keys_owner.read,keys_owner.array_sha
SIDES=(levels._SIDE_HIGH,levels._SIDE_LOW)


def need(ok,why):
    if not bool(ok):raise ValueError('NATIVE_OBSERVATION_PREFIX_'+why)


def first_anchor_witness(state,lifecycle):
    first={};stored={v['level_id']:v for v in state['levels']}
    for event in lifecycle:
        if event[0]!='create':continue
        side=event[3];need(side in SIDES and type(event[4])is int,'CREATE_CALLBACK')
        first.setdefault(side,{'level_id':event[4],'create_bar':int(event[1]),'anchor_bar':int(event[2])})
    out={}
    for side,created in first.items():
        need(created['level_id']in stored and stored[created['level_id']]['birth_side']==side,'IMMUTABLE_BIRTH_SIDE')
        value=stored[created['level_id']];pair=(value['status'],value['retest_state']);terminal=pair in (('expired','expired'),('broken','failed'))
        out[side]={**created,'status':pair[0],'retest_state':pair[1],'terminal':terminal}
    missing=[side for side in SIDES if side not in out]
    return {'first_create_by_birth_side':out,'missing_birth_sides':missing,'both_terminal':not missing and all(v['terminal']for v in out.values())}


def squeeze_release_witness(carry):
    return carry is not None and type(carry.last_release_duration)is int and carry.last_release_duration>0


def evaluate_witness(state,lifecycle,carry,*,prefix_end_ns,history_start_ns):
    anchors=first_anchor_witness(state,lifecycle);release=squeeze_release_witness(carry);reasons=[]
    reasons.extend('FIRST_BIRTH_MISSING:'+s for s in anchors['missing_birth_sides'])
    reasons.extend('FIRST_ANCHOR_NOT_TERMINAL:'+s for s,v in anchors['first_create_by_birth_side'].items()if not v['terminal'])
    if not release:reasons.append('NO_ACTUAL_SQUEEZE_RELEASE')
    if not prefix_end_ns<history_start_ns:reasons.append('PREFIX_NOT_STRICTLY_BEFORE_HISTORY')
    return {'status':'PASS'if not reasons else'UNKNOWN','reasons':reasons,'first_anchors':anchors,'actual_squeeze_release':release,
        'prefix_end_ns':int(prefix_end_ns),'history_start_ns':int(history_start_ns)}


def verify_owner_axis(full_labels_ns,retained_times,owner_rows):
    need(all(v.dtype==np.int64 and v.ndim==1 and len(v)>0 for v in (full_labels_ns,retained_times,owner_rows))
        and len(retained_times)==len(owner_rows)and np.all(np.diff(full_labels_ns)>0)and np.all(np.diff(retained_times)>0)
        and np.all(np.diff(owner_rows)>0)and owner_rows[0]>=0 and owner_rows[-1]<len(full_labels_ns)
        and np.array_equal(retained_times,full_labels_ns[owner_rows]),'EXACT_FULL_SERVING_OWNER_AXIS')
    return {'full_serving_rows':len(full_labels_ns),'full_serving_axis_sha256':array_sha(full_labels_ns),'retained_axis_sha256':array_sha(retained_times),'owner_row_indices_sha256':array_sha(owner_rows),'exact':True}


def resample_prefix(frame,tf,full_labels,cutoff_ns):
    expected=full_labels[full_labels.asi8+int(htf.MULTI_TF_SHIFT[tf].value)<cutoff_ns]
    result=htf._resample_ohlcv(frame,tf,require_quotes=True).dropna(subset=['open','high','low','close'])
    need(len(expected)>0 and expected.isin(result.index).all(),'PREFIX_FULL_SERVING_LABELS_PRESENT')
    result=result.loc[expected];need(result.index.equals(expected),'PREFIX_EXACT_INITIAL_FULL_SERVING_AXIS');return result


def clock_and_history(cap,axes,main,normalization):
    features=keys_owner.prior.features;rows=cap['source_rows'];issue=cap['issue_ns'];starts=axes['bar_start_ns'];retained=axes['retained_source_rows']
    need(all(a.dtype==np.int64 and a.ndim==1 and np.all(np.diff(a)>0)for a in (rows,issue,starts,retained))and len(rows)==len(issue)==354550
        and rows[0]>=0 and rows[-1]<len(starts)and np.array_equal(starts[rows]+keys_owner.M5,issue)
        and array_sha(rows)==main['schedule']['source_rows_sha256']and array_sha(issue)==main['schedule']['issue_ns_sha256'],'ORIGINAL_TRAIN_CLOCK')
    whole=np.flatnonzero((starts+keys_owner.M5>=keys_owner.START)&(starts+keys_owner.M5<keys_owner.END));need(np.array_equal(rows,whole),'WHOLE_TRAIN_CLOCK')
    local=features._window_union(rows,features.MODEL_NATIVE_SEQ_LEN,len(starts));positions=np.searchsorted(retained,local)
    need(np.all(positions<len(retained))and np.array_equal(retained[positions],local)and len(local)==354645,'LOCAL_OBSERVED96_UNION')
    full=htf.build_multi_tf_v4_closed_timestamp_indices(pd.to_datetime(starts,utc=True));proof={};per_tf={};deadlines={};unions={}
    for tf in TF:
        times=axes[f'tf_{tf}_ts_int64'];owners=axes[f'tf_{tf}_owner_row_indices'];proof[tf]=verify_owner_axis(full[tf].asi8,times,owners)
        shift=int(htf.MULTI_TF_SHIFT[tf].value);tails=np.searchsorted(times,issue-shift,side='right')-1;length=features.WINDOWS[tf]
        need(np.all(tails>=length-1)and np.all(owners[tails]-owners[tails-length+1]==length-1),'ACTUAL_HISTORY_CONTIGUITY:'+tf)
        ids=features._window_union(tails,length,len(times));available=times[ids]+shift
        need(np.all(available<keys_owner.END),'TF_TRAIN_AVAILABLE_CUTOFF:'+tf)
        per_tf[tf]={'rows':len(ids),'source_rows_sha256':array_sha(owners[ids]),'available_ns_sha256':array_sha(available)}
        deadlines[tf]=int(times[ids[0]]);unions[tf]={'sequence_length':length,'retained_rows':len(ids),'first_retained_index':int(ids[0]),'first_owner_row':int(owners[ids[0]]),'first_bar_start_ns':int(times[ids[0]]),'row_positions_sha256':array_sha(ids)}
    population={'cutoff_ns':keys_owner.END,'first_issue_ns':int(issue[0]),'last_issue_ns':int(issue[-1]),'issue_rows':len(rows),
        'selected_issue_ns_sha256':array_sha(issue),'selected_source_rows_sha256':array_sha(rows),'local_rows':len(local),'context_rows':len(rows),'per_tf':per_tf}
    need(population==normalization['fit_population'],'EXACT_FIT_POPULATION_UNIONS');deadlines['M5']=min(deadlines['M5'],int(starts[local[0]]))
    return full,deadlines,{'full_serving_owner_axes':proof,'fit_population_reproduced':population,'TF_history_unions':unions,
        'local96':{'rows':len(local),'source_row_union_sha256':array_sha(local),'first_bar_start_ns':int(starts[local[0]])},
        'earliest_history_bar_start_ns':deadlines,'M5_deadline_rule':'min actual local96 history and M5TF16 history; observedindices, not elapsedtime'}


def replay_prefix(frame,tf,registry_params,squeeze_params,*,history_start_ns,parameter_available_ns,chunk_rows=CHUNK):
    need(type(chunk_rows)is int and chunk_rows>0 and chunk_rows<=CHUNK,'CHUNK_BOUND')
    atr=ema50_200_spread_atr_block(frame['high'],frame['low'],frame['close'])['atr14'];source=frame[['high','low','close']].copy();source['atr']=atr
    state=None;carry=None;lifecycle=[];processed=0;output_hash=hashlib.sha256();witness=None
    for start in range(0,len(frame),chunk_rows):
        stop=min(start+chunk_rows,len(frame));matrix,names,state=levels.compute_level_registry_mtf_block_v1(source.iloc[start:stop],tf=tf.lower(),
            recurrence_threshold_atr=registry_params['level_recurrence_threshold_atr'][tf],max_evidence_age_bars=int(registry_params['level_expiry_bars'][tf]),
            registry_state=state,return_registry_state=True,runtime_lifecycle_log=lifecycle)
        sq,carry=squeeze.compute_volatility_squeeze_state(frame.iloc[start:stop][['open','high','low','close','volume']],timeframe=tf,params=squeeze_params,carry=carry)
        output_hash.update(array_sha(matrix).encode());output_hash.update(array_sha(sq.to_numpy()).encode());processed=stop
        end=int(frame.index[stop-1].value)+int(htf.MULTI_TF_SHIFT[tf].value)
        witness=evaluate_witness(state,lifecycle,carry,prefix_end_ns=end,history_start_ns=history_start_ns)
        if parameter_available_ns>=history_start_ns:
            witness['reasons'].append('PARAMETERS_NOT_AVAILABLE_BEFORE_HISTORY');witness['status']='UNKNOWN'
        if witness['status']=='PASS':break
    need(witness is not None,'NONEMPTY_PREFIX_REPLAY')
    return {**witness,'timeframe':tf,'processed_rows':processed,'available_prefix_rows':len(frame),'stopped_at_witness':witness['status']=='PASS',
        'first_bar_ns':int(frame.index[0].value),'processed_bar_axis_sha256':array_sha(frame.index.asi8[:processed]),'ATR_nan_prefix_rows':int(atr.isna().sum()),
        'lifecycle_log_sha256':keys_owner.json_sha(lifecycle),'registry_final_state_sha256':keys_owner.json_sha(state),'squeeze_carry_sha256':keys_owner.json_sha(asdict(carry)),
        'actual_release_duration':carry.last_release_duration,'registry_levels_retained':len(state['levels']),'serving_apply_selected_eligibility':True,
        'cold_initial_registry_and_squeeze':None,'parameter_available_ns':int(parameter_available_ns),'parameters_not_claimed_known_in2009':True,
        'bounded_output_hash':output_hash.hexdigest(),'bound_rule':'No automatic extension if witness absent; terminalfinalstate irreversible eligibility, not a TTLfrombirth argument.'}


def build_plan(design_ref,*,source_base_commit=None):
    d=read(design_ref);need(design_ref==DESIGN and d['schema_version']=='gx1_native_observation_prefix_witness_design_v1'and d['execution_authorized']is False
        and d['execution']==EXECUTION and d['capture_fields']==list(CAPTURE_FIELDS)and d['feature_fields']==list(FEATURE_FIELDS),'ACTIVE_DESIGN002')
    scope=read(d['scope']);read(d['semantic_partition']);cache=read(d['source_cache_plan']);part=d['source_part'];header=cache['metadata']['feature_source_metadata'][part['path']]
    need(header==d['source_part_metadata']and part==cache['metadata']['feature_source_spec']['parts'][0]and part['row_groups']==list(range(15))
        and header['total_rows']==RAW_ROWS and header['total_row_groups']==15 and header['stat']['size']==1667359
        and d['source_prefix']['cutoff_ns']==CUTOFF and d['source_prefix']['logical_time_and13numeric_bytes']==4756752,'EXACT_SOURCE_BOUND')
    need(pipeline.stamp(part['path'])==header['stat'],'RAW_PREFIX_STAT');manifest=cache['metadata']['feature_source_spec']['manifest'];read(manifest)
    main_ref=ref(keys_owner.prior.MAIN);need(main_ref['sha256']==keys_owner.prior.MAIN_SHA,'ORIGINAL_MAIN');main=read(main_ref);meta=main['metadata'];snapshot=read(meta['source_snapshot']);normalization=read(meta['normalization'])
    need(manifest==meta['native_manifest']and snapshot['surfaces_metadata']['parameter_available_ns']>d['source_prefix']['first_bar_ns'],'EXPLICIT_REPLAY_NOT2009_PARAMETER_KNOWLEDGE')
    registry=htf.require_native_m5_registry_constants(read(d['parameters']['registry']));manifest_sq=read(d['parameters']['squeeze_manifest']);carrier=squeeze.load_volatility_squeeze_artifact_manifest(Path(d['parameters']['squeeze_manifest']['path']),expected_sha256=d['parameters']['squeeze_manifest']['sha256'],clocks=TF)
    need(registry['contract_sha256']==snapshot['surfaces_metadata']['registry_contract_sha256']and carrier.binding()==snapshot['surfaces_metadata']['squeeze_binding'],'SAME_FROZEN_PARAMETERS')
    capture_ref=ref(keys_owner.CAPTURE_ROOT/'RESULT.json');capture=read(capture_ref);ctref=ref(keys_owner.CAPTURE_ROOT/'TERMINAL.json');ct=read(ctref);cp=read(capture['plan']);ch=capture['report']['capture_array_headers']
    need(ct['status']=='COMPLETE'and ct['exit_code']==0 and ct['result']==capture_ref and capture['status']=='COMPLETE_NATIVE_FULL_TRAIN_CAPTURE'and capture['source_unchanged']is True
        and capture['test_data_used']is False and cp['inputs']['main_plan']==main_ref and ch['issue_ns']['sha256']==main['schedule']['issue_ns_sha256']and ch['source_rows']['sha256']==main['schedule']['source_rows_sha256'],'CAPTURE_IDENTITY')
    inputs={'design':design_ref,'scope':d['scope'],'semantic_partition':d['semantic_partition'],'source_cache_plan':d['source_cache_plan'],'native_manifest':manifest,
        'raw_part':{k:part[k]for k in ('path','sha256')},'main_plan':main_ref,'snapshot':meta['source_snapshot'],'feature_archive':snapshot['archive'],'normalization':meta['normalization'],
        'registry':d['parameters']['registry'],'squeeze_manifest':d['parameters']['squeeze_manifest'],'capture_result':capture_ref,'capture_terminal':ctref,'capture_plan':capture['plan'],'capture':capture['report']['capture']}
    for tf in TF:entry=manifest_sq['artifacts'][tf];inputs['squeeze_params_'+tf]={'path':entry['params_artifact'],'sha256':entry['params_file_sha256']};read(inputs['squeeze_params_'+tf])
    npbytes=sum(int(np.prod(h['shape']))*np.dtype(h['dtype']).itemsize for h in [*(ch[k]for k in CAPTURE_FIELDS),*(snapshot['arrays'][k]for k in FEATURE_FIELDS)])
    need(npbytes==d['axis_ndarray_bytes']==27510800 and d['budget']['axis_arrays']==14 and set(registry['level_expiry_bars'])==set(TF),'EXACT_AXIS_BUDGET')
    return {'schema_version':'gx1_native_observation_prefix_witness_plan_v1','stage':STAGE,'status':'PREPARED_NOT_ADMITTED','plan_path':str(PLAN),'output_root':str(ROOT),
        'source_base_commit':source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'source_owners':{n:keys_owner.prior.raw.saved.native.sha(REPO/n)for n in OWNERS},
        'inputs':inputs,'execution':EXECUTION,'budget':d['budget'],'raw_read_plan':{'feature_source_spec':{'manifest':manifest,'parts':[part],'cutoff_ns':CUTOFF},'feature_source_metadata':{part['path']:header}},
        'capture_array_headers':ch,'feature_array_headers':snapshot['arrays'],'parameter_available_ns':snapshot['surfaces_metadata']['parameter_available_ns'],
        'question':d['question'],'serving_bootstrap_owner':d['bootstrap_source_evidence'],'chunk_rows':CHUNK,'runtime_versions':{'numpy':np.__version__,'pandas':pd.__version__}}


def checked_inputs(plan):
    for b in plan['inputs'].values():need(ref(b['path'])==b,'INPUT_SHA')
    part=plan['inputs']['raw_part'];need(pipeline.stamp(part['path'])==plan['raw_read_plan']['feature_source_metadata'][part['path']]['stat'],'RAW_STAT_BOOKEND')
    keys_owner.prior.raw.saved.archive_headers(plan['inputs']['capture'],{k:plan['capture_array_headers'][k]for k in CAPTURE_FIELDS},exact_inventory=keys_owner.prior.raw.saved.CAPTURE_KEYS)
    keys_owner.prior.raw.saved.archive_headers(plan['inputs']['feature_archive'],{k:plan['feature_array_headers'][k]for k in FEATURE_FIELDS},exact_inventory=plan['feature_array_headers'])


def source_clean(plan):
    return (subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='work/gx1-current'
        and subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_base_commit']and not subprocess.check_output(['git','status','--porcelain','--','gx1','scripts','tests'],cwd=REPO,text=True)
        and set(plan['source_owners'])==set(OWNERS)and all(keys_owner.prior.raw.saved.native.sha(REPO/n)==v for n,v in plan['source_owners'].items()))


def gate_expected(binding,admitted):
    return {'status':'ADMITTED_EXACT_CPU_ONLY_OBSERVATION_PREFIX_WITNESS'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,
        'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}


def inspect(plan_sha,*,admitted):
    need(Path.cwd().resolve()==REPO and type(plan_sha)is str and len(plan_sha)==64 and all(c in '0123456789abcdef'for c in plan_sha),'CWD_SHA')
    binding={'path':str(PLAN),'sha256':plan_sha};plan=read(binding);need(plan==build_plan(plan['inputs']['design'],source_base_commit=plan['source_base_commit'])and source_clean(plan),'EXACT_PLAN_SOURCE')
    pref=ref(POLICY);policy=read(pref);need(all(policy.get(k)is False for k in keys_owner.prior.raw.saved.FLAGS)and policy[STAGE]==gate_expected(binding,admitted),'ALL_TRAINING_CLOSED_GATE')
    need(not ROOT.exists()and not ROOT.is_symlink(),'FRESH_ROOT');return plan,binding,pref


def require_timeout_ancestry(plan_sha,scope):
    pid=os.getpid();runner=REPO/'scripts/gx1_capped_run.sh';seen=set()
    while pid and pid not in seen and len(seen)<64:
        seen.add(pid);p=Path('/proc')/str(pid);identity=guard._lock_process(pid,runner);args=os.fsdecode(guard._read_lock_proc(p/'cmdline')).rstrip('\0').split('\0')
        if Path(os.readlink(p/'exe'))==Path('/usr/bin/timeout').resolve():
            operand=Path(args[4])if len(args)>4 else Path('/invalid');operand=operand if operand.is_absolute()else Path(os.readlink(p/'cwd'))/operand
            need(args[1:4]==['--signal=TERM','--kill-after=10s','600s']and operand==REPO/'.venv/bin/python'and args[5:]==['-m','gx1.audit.audit_native_observation_prefix_v1','--plan-sha256',plan_sha]
                and identity.cgroup==scope and guard._lock_process(pid,runner)==identity,'CANONICAL_OUTER_TIMEOUT')
            return {'pid':pid,'start_ticks':identity.start_ticks,'seconds':600,'scope':scope,'exact_cli':True}
        pid=identity.parent_pid
    need(False,'MISSING_CANONICAL_OUTER_TIMEOUT')


def calculate(plan,*,progress=None):
    progress={}if progress is None else progress;progress['phase']='AXES14_ONLY';reader=keys_owner.prior.raw.saved.load_fields
    cap=reader(plan['inputs']['capture'],CAPTURE_FIELDS,plan['capture_array_headers']);axes=reader(plan['inputs']['feature_archive'],FEATURE_FIELDS,plan['feature_array_headers'])
    full,deadlines,clock=clock_and_history(cap,axes,read(plan['inputs']['main_plan']),read(plan['inputs']['normalization']));starts=axes['bar_start_ns']
    need(np.array_equal(np.flatnonzero(starts<CUTOFF),np.arange(RAW_ROWS))and starts[RAW_ROWS-1]<CUTOFF-keys_owner.M5,'ALL2009_SOURCE_CLOCK_PREFIX')
    progress['phase']='ONE_2009_RAW_PART';raw=pipeline.read_feature_source(plan['raw_read_plan'])
    need(len(raw)==RAW_ROWS and np.array_equal(raw.index.asi8,starts[:RAW_ROWS]),'EXACT_RAW_BOOTSTRAP_PREFIX')
    registry=htf.require_native_m5_registry_constants(read(plan['inputs']['registry']));carrier=squeeze.load_volatility_squeeze_artifact_manifest(Path(plan['inputs']['squeeze_manifest']['path']),expected_sha256=plan['inputs']['squeeze_manifest']['sha256'],clocks=TF);reports={}
    for tf in TF:
        progress['phase']='SERVING_PREFIX_'+tf;frame=resample_prefix(raw,tf,full[tf],CUTOFF)
        if int(frame.index[-1].value)+int(htf.MULTI_TF_SHIFT[tf].value)>=deadlines[tf]:
            reports[tf]={'status':'UNKNOWN','reasons':['PREFIX_BOUND_NOT_STRICTLY_BEFORE_HISTORY'],'history_start_ns':deadlines[tf]};continue
        reports[tf]=replay_prefix(frame,tf,registry,carrier.require_params(tf),history_start_ns=deadlines[tf],parameter_available_ns=plan['parameter_available_ns'],chunk_rows=plan['chunk_rows'])
    progress['phase']='COMPLETE_PREFIX_WITNESS'
    return {'timeframes':reports,'internal_encoding_exclusion_status':'PASS'if all(v['status']=='PASS'for v in reports.values())else'UNKNOWN','clock_and_history':clock,
        'payload_counts':{'axis_arrays':14,'axis_ndarray_bytes':27510800,'raw_parts':1,'raw_row_groups':15,'raw_rows':len(raw),'raw_logical_bytes':4756752,'TF_streams':5,'local_M5_alias_sixth_stream':False,'FULL_TRAIN_feature_replays':0,'models':0,'fits':0,'forwards':0,'checkpoint_loads':0},
        'limits':['Internalfirst-anchor/pre-release censor exclusion only. No global source-observability, source-calendar, firstvintage, teachercoverage, learning, OOS or profit claim.',
            'Fullclockmetadata defines actual serving bootstrap and closedprefix tail; no future prices enter the2009 replay. Wrongowneraxis is rejected, never shifted to forcePASS.',
            'Frozen parameters are not claimed known in2009; actual availability must precede earliestconsumedhistory. UNKNOWN never expands the prefix automatically.']}


def publish(name,value):
    path=ROOT/name;keys_owner.prior.raw.saved._publish_bytes(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());return ref(path)


def run(plan_sha):
    began=time.monotonic();cap=guard.require_capped_cpu_audit_execution();lock=guard.require_capped_lock_ancestry()
    need(cap['memory_max']==cap['memory_high']==4*1024**3 and cap['swap']==512*1024**2 and cap['numerical_threads']==1 and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):],'CAP_LIMITS')
    timeout=require_timeout_ancestry(plan_sha,lock['scope']);plan,binding,pref=inspect(plan_sha,admitted=True);checked_inputs(plan);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();ROOT.mkdir(parents=True,exist_ok=False)
    handlers={s:signal.getsignal(s)for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};claim=result=None;progress={'phase':'CLAIM'}
    try:
        def expired(*_):raise TimeoutError('OBSERVATION_PREFIX_WALL')
        def stopped(s,*_):raise InterruptedError('OBSERVATION_PREFIX_EXTERNAL_SIGNAL_'+str(s))
        signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
        claim=publish('CLAIM.json',{'stage':STAGE,'plan':binding,'policy_at_claim':pref,'boot_id':boot,'outer_timeout':timeout,'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
        remaining=EXECUTION['internal_seconds']-(time.monotonic()-began);need(remaining>0,'TIME_BEFORE_READS');signal.setitimer(signal.ITIMER_REAL,remaining)
        report=calculate(plan,progress=progress);checked_inputs(plan)
        need(source_clean(plan)and ref(POLICY)==pref and ref(PLAN)==binding and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot,'SOURCE_INPUT_PLAN_POLICY_BOOT_BOOKEND')
        result=publish('RESULT.json',{'status':'COMPLETE_NATIVE_OBSERVATION_PREFIX_WITNESS','stage':STAGE,'plan':binding,'claim':claim,'report':report,'source_unchanged':True,'test_data_used':False,'elapsed_seconds':time.monotonic()-began})
        terminal=publish('TERMINAL.json',{'status':'COMPLETE','exit_code':0,'stage':STAGE,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'source_unchanged':True,'test_data_used':False});return {'result':result,'terminal':terminal}
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0);publish('TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'stage':STAGE,'plan':binding,'claim':claim,'result':result,
            'remaining_invocations':0,'relaunch_allowed':False,'progress':progress,'error':type(error).__name__+':'+str(error)[:2000],'test_data_used':False});raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items():signal.signal(s,h)


def main():
    if '--d1-followup-plan-sha256' in sys.argv[1:]:return d1_main()
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    need(sys.argv[1:]==['--plan-sha256',args.plan_sha256]+(['--preflight']if args.preflight else[]),'EXACT_CLI')
    if args.preflight:
        plan,binding,pref=inspect(args.plan_sha256,admitted=False);checked_inputs(plan);print(json.dumps({'status':'PREFLIGHT_OBSERVATION_PREFIX_PASS','plan':binding,'policy':pref,'claim_created':False,'payloads_decoded':0,'models':0}))
    else:print(json.dumps(run(args.plan_sha256)))

# Explicitly separate one-use D1 follow-up; original prefix authority stays sealed.
D1_STAGE='native_observation_d1_followup_20261123_001';D1_BASENAME='NATIVE_OBSERVATION_D1_FOLLOWUP_20261123_001'
D1_PLAN=REPO/'configs/research'/(D1_BASENAME+'.json');D1_ROOT=Path('/home/andre2/GX1_RUNS')/D1_BASENAME
D1_DESIGN={'path':str(keys_owner.prior.EROOT/'OBSERVATION_D1_FOLLOWUP_DESIGN001.json'),'sha256':'4a54a73883c70d0437fbef98f334b893abceb679bd514f80e833c468d647c457'}
D1_FIELDS=('bar_start_ns','tf_D1_ts_int64','tf_D1_owner_row_indices');D1_CUTOFF=1293840000000000000;D1_RAW_ROWS=114696;D1_HISTORY=1578866400000000000


def d1_validate_parent(result,terminal,parent_plan,claim):
    need(result['status']=='COMPLETE_NATIVE_OBSERVATION_PREFIX_WITNESS'and result['stage']==STAGE and result['source_unchanged']is True and result['test_data_used']is False
        and terminal['stage']==STAGE and terminal['status']=='COMPLETE'and terminal['exit_code']==0 and terminal['source_unchanged']is True and terminal['test_data_used']is False
        and terminal['plan']==result['plan']==claim['plan']and terminal['claim']==result['claim']and claim['stage']==STAGE
        and parent_plan['schema_version']=='gx1_native_observation_prefix_witness_plan_v1'and parent_plan['stage']==STAGE and parent_plan['output_root']==str(ROOT)
        and terminal['remaining_invocations']==claim['remaining_invocations']==0 and terminal['relaunch_allowed']is claim['relaunch_allowed']is False,'D1_AUTHENTIC_COMPLETE_PARENT')
    frames=result['report']['timeframes'];old=frames['D1'];first=old['first_anchors']['first_create_by_birth_side']
    need(set(frames)==set(TF)and all(frames[t]['status']=='PASS'for t in TF if t!='D1')and all(frames[t]['actual_squeeze_release']is True and frames[t]['actual_release_duration']>0 for t in TF)
        and old['status']=='UNKNOWN'and old['processed_rows']==old['available_prefix_rows']==153 and old['history_start_ns']==D1_HISTORY
        and old['reasons']==['FIRST_ANCHOR_NOT_TERMINAL:low_pivot']and old['serving_apply_selected_eligibility']is True
        and first['high_pivot']['terminal']is True and first['low_pivot']['level_id']==1 and first['low_pivot']['status']=='broken'and first['low_pivot']['retest_state']=='pending'
        and result['report']['clock_and_history']['earliest_history_bar_start_ns']['D1']==D1_HISTORY,'D1_EXACT_UNRESOLVED_PARENT')
    return old


def d1_source_bridge(parent_plan):
    import ast
    base=parent_plan['source_base_commit'];owners=parent_plan['source_owners'];need(set(owners)==set(OWNERS),'D1_PARENT_SOURCE_INVENTORY')
    dispatch=ast.parse("if '--d1-followup-plan-sha256' in sys.argv[1:]:return d1_main()").body[0]
    for name in (SELF,TEST):
        original=subprocess.check_output(['git','show',base+':'+name],cwd=REPO);need(hashlib.sha256(original).hexdigest()==owners[name],'D1_AUTHENTIC_PARENT_SOURCE')
        old=ast.parse(original);current=ast.parse((REPO/name).read_bytes());kept=[]
        for node in current.body:
            names=[n.id for n in node.targets if isinstance(n,ast.Name)]if isinstance(node,ast.Assign)else[]
            if isinstance(node,(ast.FunctionDef,ast.ClassDef))and node.name.startswith(('d1_','test_d1_')):continue
            if names and all(n.startswith('D1_')for n in names):continue
            if name==SELF and isinstance(node,ast.FunctionDef)and node.name=='main':
                need(ast.dump(node.body[0],include_attributes=False)==ast.dump(dispatch,include_attributes=False),'D1_EXACT_NEW_DISPATCH');node.body=node.body[1:]
            kept.append(node)
        current.body=kept;need(ast.dump(current,include_attributes=False)==ast.dump(old,include_attributes=False),'D1_ORIGINAL_AST_UNCHANGED:'+name)
    other={n:v for n,v in owners.items()if n not in (SELF,TEST)};need(all(keys_owner.prior.raw.saved.native.sha(REPO/n)==v for n,v in other.items()),'D1_OTHER_PARENT_OWNERS_UNCHANGED')
    return {'parent_producer_commit':base,'other_parent_owners_unchanged':len(other),'original_owner_and_test_AST_preserved':True,'allowed_delta':'D1_/d1_ additions and exact new CLI dispatch only'}


def d1_replay_registry(frame,registry_params,parent_d1,*,history_start_ns,parameter_available_ns,progress=None):
    progress={}if progress is None else progress;n=parent_d1['processed_rows'];shift=int(htf.MULTI_TF_SHIFT['D1'].value)
    need(n==153 and len(frame)>n and array_sha(frame.index.asi8[:n])==parent_d1['processed_bar_axis_sha256'],'D1_PARENT_AXIS_BEFORE_REPLAY')
    need(parameter_available_ns<history_start_ns and int(frame.index[-1].value)+shift<history_start_ns,'D1_PARAMETER_AND_PREFIX_DEADLINE')
    source=frame[['high','low','close']].copy();source['atr']=ema50_200_spread_atr_block(source['high'],source['low'],source['close'])['atr14'];state=None;lifecycle=[]
    for start,stop in ((0,128),(128,n)):
        _,_,state=levels.compute_level_registry_mtf_block_v1(source.iloc[start:stop],tf='d1',recurrence_threshold_atr=registry_params['level_recurrence_threshold_atr']['D1'],
            max_evidence_age_bars=int(registry_params['level_expiry_bars']['D1']),registry_state=state,return_registry_state=True,runtime_lifecycle_log=lifecycle)
    proof={'processed_bar_axis_sha256':array_sha(frame.index.asi8[:n]),'registry_final_state_sha256':keys_owner.json_sha(state),'lifecycle_log_sha256':keys_owner.json_sha(lifecycle)}
    need(all(v==parent_d1[k]for k,v in proof.items()),'D1_PARENT_REPLAY_THREE_SHA_EXACT');need(first_anchor_witness(state,lifecycle)==parent_d1['first_anchors'],'D1_PARENT_ANCHORS_EXACT')
    progress.update(phase='D1_PARENT153_QUALIFIED',parent_replay_proof=proof);processed=n;anchors=first_anchor_witness(state,lifecycle)
    for row in range(n,len(source)):
        _,_,state=levels.compute_level_registry_mtf_block_v1(source.iloc[row:row+1],tf='d1',recurrence_threshold_atr=registry_params['level_recurrence_threshold_atr']['D1'],
            max_evidence_age_bars=int(registry_params['level_expiry_bars']['D1']),registry_state=state,return_registry_state=True,runtime_lifecycle_log=lifecycle)
        processed=row+1;anchors=first_anchor_witness(state,lifecycle);progress.update(phase='D1_ONE_BAR_CONTINUATION',processed_D1_rows=processed)
        if anchors['both_terminal']:break
    reasons=[]if anchors['both_terminal']else['FIRST_ANCHOR_NOT_TERMINAL:'+s for s,v in anchors['first_create_by_birth_side'].items()if not v['terminal']]
    reasons.extend('FIRST_BIRTH_MISSING:'+s for s in anchors['missing_birth_sides']);status='PASS'if not reasons else'UNKNOWN'
    return {'status':status,'reasons':reasons,'timeframe':'D1','first_anchors':anchors,'parent153_replay_exact':True,'parent_replay_proof':proof,'parent_replay_chunk_rows':[128,25],
        'processed_rows':processed,'continuation_rows':processed-n,'available_prefix_rows':len(frame),'stopped_at_first_terminal_pair':status=='PASS','continuation_chunk_rows':1,
        'prefix_end_ns':int(frame.index[processed-1].value)+shift,'history_start_ns':history_start_ns,'parameter_available_ns':parameter_available_ns,
        'processed_bar_axis_sha256':array_sha(frame.index.asi8[:processed]),'registry_final_state_sha256':keys_owner.json_sha(state),'lifecycle_log_sha256':keys_owner.json_sha(lifecycle),
        'serving_apply_selected_eligibility':True,'new_squeeze_computations':0,'actual_squeeze_release_reused_from_parent':True,'parameters_not_claimed_known_in2009':True}


def d1_build_plan(design_ref,*,source_base_commit=None):
    d=read(design_ref);need(design_ref==D1_DESIGN and d['execution_authorized']is False and d['execution']==EXECUTION,'D1_EXACT_DESIGN')
    inputs=dict(d['inputs']);inputs['design']=design_ref;result=read(inputs['parent_result']);terminal=read(inputs['parent_terminal']);parent=read(inputs['parent_plan']);claim=read(inputs['parent_claim']);qualification=read(inputs['parent_source_qualification'])
    need(terminal['result']==inputs['parent_result']and result['plan']==inputs['parent_plan']and result['claim']==inputs['parent_claim'],'D1_PARENT_RECEIPT_BINDINGS');d1_validate_parent(result,terminal,parent,claim)
    need(qualification['status']=='SOURCE_SYNTHETIC_AND_ACTUAL_METADATA_QUALIFIED_NOT_ADMITTED'and qualification['source']['sha256']==parent['source_owners'][SELF]and qualification['test']['sha256']==parent['source_owners'][TEST]and qualification['design']==parent['inputs']['design'],'D1_PARENT_SOURCE_QUALIFICATION')
    bridge=d1_source_bridge(parent);cache=read(inputs['source_cache_plan']);parts=cache['metadata']['feature_source_spec']['parts'][:2];headers=cache['metadata']['feature_source_metadata']
    need(inputs['source_cache_plan']==parent['inputs']['source_cache_plan']and len(parts)==2 and parts[0]['sha256']==parent['inputs']['raw_part']['sha256']and parts[1]['sha256']=='20a4b0f2e7d6005dbf4652b65d89744b94eed293eaa908c36b3f0af835474ee8'
        and parts[0]['row_groups']==list(range(15))and parts[1]['row_groups']==list(range(25))and sum(headers[p['path']]['total_rows']for p in parts)==D1_RAW_ROWS
        and sum(headers[p['path']]['stat']['size']for p in parts)==4486929,'D1_EXACT_TWO_SOURCE_PARTS')
    for name in ('native_manifest','feature_archive','snapshot','normalization','registry','squeeze_manifest',*('squeeze_params_'+t for t in TF)):
        need(inputs[name]==parent['inputs'][name],'D1_INHERITED_INPUT:'+name);read(inputs[name])if name!='feature_archive'else None
    registry=htf.require_native_m5_registry_constants(read(inputs['registry']));snapshot=read(inputs['snapshot']);need(registry['contract_sha256']==snapshot['surfaces_metadata']['registry_contract_sha256'],'D1_SAME_REGISTRY')
    for i,p in enumerate(parts):inputs['raw_part_'+str(i)]={k:p[k]for k in ('path','sha256')};need(pipeline.stamp(p['path'])==headers[p['path']]['stat'],'D1_SOURCE_STAT')
    array_headers={k:parent['feature_array_headers'][k]for k in D1_FIELDS};nbytes=sum(int(np.prod(h['shape']))*np.dtype(h['dtype']).itemsize for h in array_headers.values());need(nbytes==9750808,'D1_THREE_CLOCK_BUDGET')
    return {'schema_version':'gx1_native_observation_d1_followup_plan_v1','stage':D1_STAGE,'status':'PREPARED_NOT_ADMITTED','plan_path':str(D1_PLAN),'output_root':str(D1_ROOT),
        'source_base_commit':source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'source_owners':{n:keys_owner.prior.raw.saved.native.sha(REPO/n)for n in OWNERS},'source_bridge':bridge,
        'inputs':inputs,'execution':EXECUTION,'budget':d['budget'],'feature_array_headers':parent['feature_array_headers'],'clock_fields':list(D1_FIELDS),'clock_ndarray_bytes':nbytes,
        'raw_read_plan':{'feature_source_spec':{'manifest':inputs['native_manifest'],'parts':parts,'cutoff_ns':D1_CUTOFF},'feature_source_metadata':{p['path']:headers[p['path']]for p in parts}},
        'history_start_ns':D1_HISTORY,'parameter_available_ns':parent['parameter_available_ns'],'parent_clock_proof':result['report']['clock_and_history'],'parent_D1':result['report']['timeframes']['D1'],
        'reuse':'Four TF PASS and allfive actual squeeze-release witnesses reused unchanged. D1 registry-only cold replay then completecarry, no HMM.'}


def d1_checked_inputs(plan):
    for b in plan['inputs'].values():need(ref(b['path'])==b,'D1_INPUT_SHA')
    for p in plan['raw_read_plan']['feature_source_spec']['parts']:need(pipeline.stamp(p['path'])==plan['raw_read_plan']['feature_source_metadata'][p['path']]['stat'],'D1_RAW_STAT_BOOKEND')
    keys_owner.prior.raw.saved.archive_headers(plan['inputs']['feature_archive'],{k:plan['feature_array_headers'][k]for k in D1_FIELDS},exact_inventory=plan['feature_array_headers'])


def d1_gate_expected(binding,admitted):
    return {'status':'ADMITTED_EXACT_CPU_ONLY_D1_FOLLOWUP'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,
        'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}


def d1_inspect(plan_sha,*,admitted):
    need(Path.cwd().resolve()==REPO and type(plan_sha)is str and len(plan_sha)==64 and all(c in '0123456789abcdef'for c in plan_sha),'D1_CWD_SHA')
    binding={'path':str(D1_PLAN),'sha256':plan_sha};plan=read(binding);need(plan==d1_build_plan(plan['inputs']['design'],source_base_commit=plan['source_base_commit'])and source_clean(plan),'D1_EXACT_PLAN_SOURCE')
    pref=ref(POLICY);policy=read(pref);need(all(policy.get(k)is False for k in keys_owner.prior.raw.saved.FLAGS)and policy[D1_STAGE]==d1_gate_expected(binding,admitted),'D1_ALL_TRAINING_CLOSED_GATE')
    need(not D1_ROOT.exists()and not D1_ROOT.is_symlink(),'D1_FRESH_ROOT');return plan,binding,pref


def d1_require_timeout_ancestry(plan_sha,scope):
    pid=os.getpid();runner=REPO/'scripts/gx1_capped_run.sh';seen=set()
    while pid and pid not in seen and len(seen)<64:
        seen.add(pid);p=Path('/proc')/str(pid);identity=guard._lock_process(pid,runner);args=os.fsdecode(guard._read_lock_proc(p/'cmdline')).rstrip('\0').split('\0')
        if Path(os.readlink(p/'exe'))==Path('/usr/bin/timeout').resolve():
            operand=Path(args[4])if len(args)>4 else Path('/invalid');operand=operand if operand.is_absolute()else Path(os.readlink(p/'cwd'))/operand
            need(args[1:4]==['--signal=TERM','--kill-after=10s','600s']and operand==REPO/'.venv/bin/python'and args[5:]==['-m','gx1.audit.audit_native_observation_prefix_v1','--d1-followup-plan-sha256',plan_sha]
                and identity.cgroup==scope and guard._lock_process(pid,runner)==identity,'D1_CANONICAL_OUTER_TIMEOUT');return {'pid':pid,'start_ticks':identity.start_ticks,'seconds':600,'scope':scope,'exact_cli':True}
        pid=identity.parent_pid
    need(False,'D1_MISSING_CANONICAL_OUTER_TIMEOUT')


def d1_calculate(plan,*,progress=None):
    progress={}if progress is None else progress;progress['phase']='D1_THREE_CLOCK_AXES';axes=keys_owner.prior.raw.saved.load_fields(plan['inputs']['feature_archive'],D1_FIELDS,plan['feature_array_headers']);starts=axes['bar_start_ns']
    full=htf.build_multi_tf_v4_closed_timestamp_indices(pd.to_datetime(starts,utc=True))['D1'];axis_proof=verify_owner_axis(full.asi8,axes['tf_D1_ts_int64'],axes['tf_D1_owner_row_indices'])
    need(axis_proof==plan['parent_clock_proof']['full_serving_owner_axes']['D1']and np.array_equal(np.flatnonzero(starts<D1_CUTOFF),np.arange(D1_RAW_ROWS))and starts[D1_RAW_ROWS-1]<D1_CUTOFF-keys_owner.M5,'D1_ORIGINAL_FULL_AXIS_AND_RAW_PREFIX')
    progress['phase']='D1_TWO_RAW_PARTS';raw=pipeline.read_feature_source(plan['raw_read_plan']);need(len(raw)==D1_RAW_ROWS and np.array_equal(raw.index.asi8,starts[:D1_RAW_ROWS]),'D1_EXACT_COMBINED_RAW_PREFIX')
    frame=resample_prefix(raw,'D1',full,D1_CUTOFF);registry=htf.require_native_m5_registry_constants(read(plan['inputs']['registry']));report=d1_replay_registry(frame,registry,plan['parent_D1'],history_start_ns=plan['history_start_ns'],parameter_available_ns=plan['parameter_available_ns'],progress=progress)
    return {'D1':report,'full_D1_owner_axis':axis_proof,'allfive_internal_encoding_exclusion_status':report['status'],'reused_parent_result':plan['inputs']['parent_result'],
        'reused_four_TF_PASS':['M5','M15','H1','H4'],'reused_allfive_squeeze_releases':True,'payload_counts':{'axis_arrays':3,'axis_ndarray_bytes':9750808,'raw_parts':2,'raw_row_groups':40,'raw_rows':len(raw),'raw_logical_bytes':12845952,'D1_registry_streams':1,'HMM_computations':0,'models':0,'forwards':0,'fits':0,'checkpoint_loads':0,'network_calls':0},
        'limits':['Internalfirst-anchor/pre-release domain exclusion only; no global07/sourcecalendar/vintage/learning/OOS/profit claim.','Parent2009 replay must match exactly before any newcarry; fixed2010bound, no automatic2011 or newobservability bits.','Frozenparameters not claimed known in2009; availability precedes actual consumedTRAINhistory.']}


def d1_publish(name,value):
    path=D1_ROOT/name;keys_owner.prior.raw.saved._publish_bytes(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());return ref(path)


def d1_run(plan_sha):
    began=time.monotonic();cap=guard.require_capped_cpu_audit_execution();lock=guard.require_capped_lock_ancestry()
    need(cap['memory_max']==cap['memory_high']==4*1024**3 and cap['swap']==512*1024**2 and cap['numerical_threads']==1 and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):],'D1_CAP_LIMITS')
    timeout=d1_require_timeout_ancestry(plan_sha,lock['scope']);plan,binding,pref=d1_inspect(plan_sha,admitted=True);d1_checked_inputs(plan);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();D1_ROOT.mkdir(parents=True,exist_ok=False)
    handlers={s:signal.getsignal(s)for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};claim=result=None;progress={'phase':'D1_CLAIM'}
    try:
        def expired(*_):raise TimeoutError('D1_FOLLOWUP_WALL')
        def stopped(s,*_):raise InterruptedError('D1_FOLLOWUP_EXTERNAL_SIGNAL_'+str(s))
        signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
        claim=d1_publish('CLAIM.json',{'stage':D1_STAGE,'plan':binding,'policy_at_claim':pref,'boot_id':boot,'outer_timeout':timeout,'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
        remaining=EXECUTION['internal_seconds']-(time.monotonic()-began);need(remaining>0,'D1_TIME_BEFORE_READS');signal.setitimer(signal.ITIMER_REAL,remaining)
        report=d1_calculate(plan,progress=progress);d1_checked_inputs(plan)
        need(source_clean(plan)and ref(POLICY)==pref and ref(D1_PLAN)==binding and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot,'D1_SOURCE_INPUT_PLAN_POLICY_BOOT_BOOKEND')
        result=d1_publish('RESULT.json',{'status':'COMPLETE_NATIVE_OBSERVATION_D1_FOLLOWUP','stage':D1_STAGE,'plan':binding,'claim':claim,'report':report,'source_unchanged':True,'test_data_used':False,'elapsed_seconds':time.monotonic()-began})
        terminal=d1_publish('TERMINAL.json',{'status':'COMPLETE','exit_code':0,'stage':D1_STAGE,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'source_unchanged':True,'test_data_used':False});return {'result':result,'terminal':terminal}
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0);d1_publish('TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'stage':D1_STAGE,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'progress':progress,'error':type(error).__name__+':'+str(error)[:2000],'test_data_used':False});raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items():signal.signal(s,h)


def d1_main():
    parser=argparse.ArgumentParser(description='Distinct bounded D1registry follow-up; originaloneuseprefixsealed.');parser.add_argument('--d1-followup-plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    need(sys.argv[1:]==['--d1-followup-plan-sha256',args.d1_followup_plan_sha256]+(['--preflight']if args.preflight else[]),'D1_EXACT_CLI')
    if args.preflight:
        plan,binding,pref=d1_inspect(args.d1_followup_plan_sha256,admitted=False);d1_checked_inputs(plan);print(json.dumps({'status':'PREFLIGHT_OBSERVATION_D1_FOLLOWUP_PASS','plan':binding,'policy':pref,'claim_created':False,'payloads_decoded':0,'models':0}))
    else:print(json.dumps(d1_run(args.d1_followup_plan_sha256)))

if __name__=='__main__':main()
