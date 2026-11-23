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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    need(sys.argv[1:]==['--plan-sha256',args.plan_sha256]+(['--preflight']if args.preflight else[]),'EXACT_CLI')
    if args.preflight:
        plan,binding,pref=inspect(args.plan_sha256,admitted=False);checked_inputs(plan);print(json.dumps({'status':'PREFLIGHT_OBSERVATION_PREFIX_PASS','plan':binding,'policy':pref,'claim_created':False,'payloads_decoded':0,'models':0}))
    else:print(json.dumps(run(args.plan_sha256)))

if __name__=='__main__':main()
