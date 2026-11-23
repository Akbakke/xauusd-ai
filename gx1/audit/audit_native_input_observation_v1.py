"""One-use TRAIN input-representation census; finite values are not source-known flags."""
from __future__ import annotations
import argparse,copy,json,resource,signal,subprocess,sys,time
from pathlib import Path
def _reject_torch(event,args):
    if event=='import'and (args[0]=='torch'or args[0].startswith('torch.')):raise RuntimeError('NATIVE_INPUT_OBSERVATION_TORCH_FORBIDDEN')
if __name__=='__main__':
    sys.addaudithook(_reject_torch)
    from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
    install_native_m5_direction_io_firewall()
import numpy as np
from gx1.audit import audit_native_future_window_coverage_v1 as raw
from gx1.audit import build_native_m5_ta_feature_cache_v1 as cache
from gx1.contracts import market_state_native_m5_feature_inputs_v1 as features
from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.features import micro_structure_v1 as micro
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution,require_capped_lock_ancestry

REPO,POLICY=raw.REPO,raw.POLICY
STAGE='native_input_observation_census_20261123_001';BASENAME='NATIVE_INPUT_OBSERVATION_CENSUS_20261123_001'
PLAN=REPO/'configs/research'/(BASENAME+'.json');ROOT=Path('/home/andre2/GX1_RUNS')/BASENAME
SELF='gx1/audit/audit_native_input_observation_v1.py';TEST='tests/test_audit_native_input_observation_v1.py'
EROOT=Path('/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001')
CENSUS={'path':str(EROOT/'FIELD_CENSUS001.json'),'sha256':'0db995158e1a22a16371d68f71ef183bcf9236926d7c2d07b02ae031b7cb42d7'}
ERRATA={'path':str(EROOT/'FIELD_CENSUS_ERRATA001.json'),'sha256':'b362293b6c48720d1d1e6d62c91da49791d3df079ae04337add01010c93f7d0a'}
MAIN=REPO/'configs/research/NATIVE_FUTURE_STATE_ONE_EPOCH_20261121_001.json'
MAIN_SHA='c3e769fc5468739b576144f2cb2e5bf572d6b662a3c490ee56d8641fde1dd561'
ROWS=354550;BLOCK=4096;TF=('M5','M15','H1','H4','D1')
UNION_ROWS={'signal':354645,'ctx_cont':354550,'mtf_m5':354565,'mtf_m15':118271,'mtf_h1':29663,'mtf_h4':7829,'mtf_d1':1542}
EXECUTION={'class':'audit','memory':'4G','swap':'512M','numerical_threads':1,'outer_seconds':900,'internal_seconds':870,'kill_after_seconds':10}
BUDGET={'feature_archive_loads':1,'feature_arrays_decoded':27,'capture_archive_loads':1,'capture_fields':['source_rows'],'relations':86,'block_rows':BLOCK,
 'prices':0,'targets':0,'logits':0,'checkpoint_loads':0,'models':0,'forwards':0,'backward':0,'fits':0,'HTTP_calls':0,'new_caches':0,'VAL_TEST_rows_analyzed':0}
OWNERS=tuple(dict.fromkeys((*raw.OWNERS,*cache.SOURCE_PATHS,SELF,TEST,'gx1/features/micro_structure_v1.py','gx1/contracts/entry_model_native_input_normalization_v1.py')))
ref,read,array_sha=raw.saved.ref,raw.saved.read,raw.saved.array_sha
IDS={'micro_range','swing_impulse','smc_envelope','level_below','level_above2','level_below2','level_above','geomline_above','geomline_below','geomchan',
 'swing_impulse_mtf_alias','smc_envelope_mtf_alias','smc_sweep_up','smc_sweep_down','level_above_pending_retest','level_below_pending_retest'}


def need(ok,why):
    if not bool(ok):raise ValueError('NATIVE_INPUT_OBSERVATION_'+why)


def observation_rules(census,errata):
    need(census['counts']['field_records']==1731 and errata['original_census']==CENSUS
         and errata['affected_rule_ids']==['level_above_pending_retest','level_below_pending_retest'],'CENSUS_ERRATA')
    rules=copy.deepcopy(census['cache_measurable_rules'])
    need(set(rules)==set(UNION_ROWS)|{'ctx_cat'} and len(rules['ctx_cat'])==0 and len(rules['ctx_cont'])==2
         and all(len(rules[s])==14 for s in UNION_ROWS if s!='ctx_cont') and sum(map(len,rules.values()))==86,'EXACT_86_RELATIONS')
    for surface,items in rules.items():
        need(len({r['id']for r in items})==len(items) and all(r['id']in IDS for r in items),'NAMED_RELATIONS:'+surface)
        for r in items:
            if r['id'].startswith('level_')and r['id'].endswith('_pending_retest'):
                r['absent_storage']='age0 implies distance0; canonical eligible occupied pending slot has integer age>=1 and distance>0'
                r['preconditions']=errata['corrected_rule']['preconditions']
    return rules


def count_relation(raw_values,names,rule):
    a=np.asarray(raw_values);need(a.ndim==2 and a.shape[1]==len(names),'RELATION_MATRIX')
    need(rule['id']in IDS and len(set(names))==len(names) and rule['value_fields'],'RELATION_ID')
    positions={n:i for i,n in enumerate(names)};m=a[:,positions[rule['availability_field']]];v=a[:,[positions[n]for n in rule['value_fields']]]
    rid=rule['id'];finite=np.isfinite(m)
    if rid.startswith('smc_envelope'):
        valid=finite&(m>=0);present=valid&(m>0)
    elif rid.startswith('geomline_')or rid=='level_above' or rid.endswith('_pending_retest'):
        valid=finite&(m>=0)&(m==np.floor(m));present=valid&(m>0)
    else:valid=finite&np.isin(m,(0.,1.));present=valid&(m==1)
    absent=valid&~present
    bad=absent&np.any(v!=0,axis=1)
    bad|=present&~np.isfinite(v).all(axis=1)
    if rid.endswith('_pending_retest')or rid.startswith('smc_sweep_'):bad|=present&np.any(v<=0,axis=1)
    return {'support_true':int(present.sum()),'support_false':int(absent.sum()),'invalid_predicate':int((~valid).sum()),'raw_storage_violations':int(bad.sum()),
        'present_raw_zero_by_value':{n:int(np.count_nonzero(v[present,j]==0))for j,n in enumerate(rule['value_fields'])},
        'absent_raw_zero_by_value':{n:int(np.count_nonzero(v[absent,j]==0))for j,n in enumerate(rule['value_fields'])}}


def summarize_relation(total):
    failed=any(total.get(k,0)>0 for k in ('invalid_predicate','raw_storage_violations','normalized_range_violations'))
    return {**total,'qualification_status':'FAIL'if failed else'NO_SUPPORT'if total['support_true']==0 else'COUNTS_COMPLETE',
        'absent_storage_status':'NOT_EXERCISED'if total['support_false']==0 else'EXERCISED',
        'meaning':'Named representation predicate only; not independent vendor/source-observation or learned utility.'}


def census_surface(values,row_indices,names,rules,norm_contract,*,block_rows=BLOCK):
    rows=np.asarray(row_indices);need(type(block_rows)is int and 0<block_rows<=BLOCK and rows.dtype==np.int64 and rows.ndim==1
        and len(rows)>0 and np.all(np.diff(rows)>0) and rows[0]>=0 and rows[-1]<len(values),'SURFACE_ROWS_BLOCK')
    need(list(names)==norm_contract['field_names'] and values.shape[1]==len(names),'SURFACE_FIELDS')
    counts=np.zeros((len(names),3),np.int64);totals=[{**{k:0 for k in ('support_true','support_false','invalid_predicate','raw_storage_violations')},
        **{k:{n:0 for n in r['value_fields']}for k in ('present_raw_zero_by_value','absent_raw_zero_by_value')}}for r in rules]
    pairs=micro.range_observation_masking_pairs(names);range_totals=[{'value_field':names[v],'mask_field':names[m],'masked_rows':0,'normalized_range_violations':0}for v,m in pairs]
    errors=[];calls=0
    for start in range(0,len(rows),block_rows):
        block=values[rows[start:start+block_rows]];finite=np.isfinite(block);counts[:,0]+=finite.sum(axis=0);counts[:,1]+=(~finite).sum(axis=0);counts[:,2]+=(block==0).sum(axis=0)
        for total,r in zip(totals,rules):
            measured=count_relation(block,names,r)
            for k in ('support_true','support_false','invalid_predicate','raw_storage_violations'):total[k]+=measured[k]
            for k in ('present_raw_zero_by_value','absent_raw_zero_by_value'):
                for n,v in measured[k].items():total[k][n]+=v
        for total,(v,m)in zip(range_totals,pairs):total['masked_rows']+=int((block[:,m]==0).sum())
        try:
            normalized=norm.apply_surface_normalization(block,norm_contract);calls+=1
            for total,(v,m)in zip(range_totals,pairs):total['normalized_range_violations']+=int(np.count_nonzero(normalized[block[:,m]==0,v]))
            del normalized
        except (ValueError,RuntimeError)as error:errors.append({'block_start':start,'rows':len(block),'error':type(error).__name__+':'+str(error)[:500]})
        del block,finite
    reports=[{'id':r['id'],'availability_field':r['availability_field'],'value_fields':r['value_fields'],'kind':r['kind'],
        **summarize_relation(t)}for r,t in zip(rules,totals)]
    failed=bool(errors or counts[:,1].sum() or any(x['qualification_status']=='FAIL'for x in reports)or any(x['normalized_range_violations']for x in range_totals))
    return {'rows':len(rows),'field_counts':[{'canonical_name':n,'finite':int(v[0]),'nonfinite':int(v[1]),'raw_zero':int(v[2]),
        'source_known_status':'NOT_ESTABLISHED_FROM_FINITE_OR_ZERO'}for n,v in zip(names,counts)],'relations':reports,'normalized_range':range_totals,
        'normalization_apply_calls':calls,'normalization_block_errors':errors,'qualification_status':'FAIL'if failed else'COUNTS_COMPLETE_WITH_UNCLASSIFIED_FIELDS'}


def build_unions(provider,rows,normalization,main):
    need(rows.dtype==np.int64 and rows.shape==(ROWS,) and np.all(np.diff(rows)>0) and rows[0]>=0 and rows[-1]<len(provider.issue_ns),'GLOBAL_SOURCE_ROWS')
    whole=np.flatnonzero((provider.issue_ns>=raw.START)&(provider.issue_ns<raw.UPPER)).astype(np.int64)
    need(np.array_equal(rows,whole) and np.array_equal(provider.source_row[whole],rows) and provider.input_ready[rows].all()
         and array_sha(rows)==main['schedule']['source_rows_sha256'] and array_sha(provider.issue_ns[rows])==main['schedule']['issue_ns_sha256']
         and provider.input_binding(normalization)==main['metadata']['input_binding'],'WHOLE_TRAIN_AXIS_INPUT_BINDING')
    local=provider._feature_positions(features._window_union(rows,features.MODEL_NATIVE_SEQ_LEN,len(provider.issue_ns)))
    unions={'signal':local,'ctx_cont':provider._feature_positions(rows)};per_tf={};ledger={}
    for tf in TF:
        ids=features._window_union(provider.tf_tails[tf][rows],features.WINDOWS[tf],len(provider.tf_times[tf]));unions['mtf_'+tf.lower()]=ids
        available=provider.tf_times[tf][ids]+int(features.htf.MULTI_TF_SHIFT[tf].value)
        need(np.all(available<raw.UPPER),'TF_TRAIN_CUTOFF:'+tf)
        per_tf[tf]={'rows':len(ids),'source_rows_sha256':array_sha(provider.tf_source_rows[tf][ids]),'available_ns_sha256':array_sha(available)}
    population={'cutoff_ns':raw.UPPER,'first_issue_ns':int(provider.issue_ns[rows[0]]),'last_issue_ns':int(provider.issue_ns[rows[-1]]),'issue_rows':len(rows),
        'selected_issue_ns_sha256':array_sha(provider.issue_ns[rows]),'selected_source_rows_sha256':array_sha(rows),'local_rows':len(local),'context_rows':len(rows),'per_tf':per_tf}
    need(population==normalization['fit_population'] and {s:len(x)for s,x in unions.items()}==UNION_ROWS
         and all(normalization['surfaces'][s]['fit_row_count']==len(x)for s,x in unions.items()),'EXACT_FIT_POPULATION_UNIONS')
    for s,ids in unions.items():ledger[s]={'rows':len(ids),'provider_row_positions_sha256':array_sha(ids),'source_rows_sha256':array_sha(provider.retained_source_rows[ids]if s in ('signal','ctx_cont')else provider.tf_source_rows[s[4:].upper()][ids])}
    return unions,{'fit_population_reproduced_exactly':True,'fit_population':population,'unions':ledger,
        'local_union_hash_scope':'Derived deterministically from bound global source_rows with unchanged canonical window union; old fit metadata stores local count, not a separate local-union SHA.'}


def build_plan(*,design,source_base_commit=None):
    d=read(design);need(Path(design['path'])==EROOT/'INPUT_OBSERVATION_CENSUS_DESIGN001.json'
        and d['schema_version']=='gx1_native_input_observation_census_design_v1'and d['execution_authorized']is False
        and d['execution']==EXECUTION and d['budget']==BUDGET and d['field_census']==CENSUS and d['field_census_errata']==ERRATA,'LOCKED_DESIGN')
    main_ref=ref(MAIN);need(main_ref['sha256']==MAIN_SHA,'ORIGINAL_MAIN');main=read(main_ref);metadata=main['metadata']
    capture_result=read(d['capture_result']);capture_plan=read(capture_result['plan']);report=capture_result['report'];headers=report['capture_array_headers']
    need(capture_result['status']=='COMPLETE_NATIVE_FULL_TRAIN_CAPTURE'and capture_result['source_unchanged']is True and capture_result['test_data_used']is False
        and report['rows']==ROWS and set(headers)==raw.saved.CAPTURE_KEYS and headers['source_rows']=={'shape':[ROWS],'dtype':'<i8','sha256':main['schedule']['source_rows_sha256']}
        and capture_plan['inputs']['main_plan']==main_ref,'AUTHENTIC_FULL_CAPTURE_MAIN_AXIS')
    clock=read(d['raw_clock_result']);clock_plan=read(clock['plan'])
    need(clock['status']=='COMPLETE_NATIVE_RAW_FEED_CLOCK_QUALIFICATION'and clock['source_unchanged']is True and clock['test_data_used']is False
        and clock_plan['inputs']['capture_result']==d['capture_result'],'RAW_PROVENANCE_CHAIN')
    census=read(CENSUS);errata=read(ERRATA);observation_rules(census,errata)
    need(ref(errata['canonical_source']['path'])=={k:errata['canonical_source'][k]for k in ('path','sha256')},'ERRATA_CANONICAL_SOURCE')
    snapshot=read(metadata['source_snapshot']);normalization=read(metadata['normalization']);norm.require_native_m5_input_normalization_contract(normalization)
    need(census['inputs']['snapshot']==metadata['source_snapshot'] and census['inputs']['normalization']==metadata['normalization']
        and {s:v['fit_row_count']for s,v in normalization['surfaces'].items()}==UNION_ROWS,'EXACT_EXISTING_FEATURE_NORMALIZATION')
    for key in ('field_contracts','qualification_requirements'):read(d[key])
    inputs={'design':design,'main_plan':main_ref,'capture_result':d['capture_result'],'capture_plan':capture_result['plan'],'capture':report['capture'],
        'raw_clock_result':d['raw_clock_result'],'raw_clock_plan':clock['plan'],'field_census':CENSUS,'field_census_errata':ERRATA,
        'field_contracts':d['field_contracts'],'qualification_requirements':d['qualification_requirements'],
        'snapshot':metadata['source_snapshot'],'feature_archive':snapshot['archive'],'normalization':metadata['normalization']}
    if 'field_unit_authority'in d:read(d['field_unit_authority']);inputs['field_unit_authority']=d['field_unit_authority']
    npbytes=sum(int(np.prod(h['shape']))*np.dtype(h['dtype']).itemsize for h in snapshot['arrays'].values())
    need(len(snapshot['arrays'])==27 and npbytes==1433216758 and Path(snapshot['archive']['path']).stat().st_size==563845063,'PHYSICAL_FEATURE_BUDGET')
    return {'schema_version':'gx1_native_input_observation_census_plan_v1','stage':STAGE,'status':'PREPARED_NOT_ADMITTED','plan_path':str(PLAN),'output_root':str(ROOT),
        'source_base_commit':source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'source_owners':{p:raw.saved.native.sha(REPO/p)for p in OWNERS},'inputs':inputs,'execution':EXECUTION,'budget':BUDGET,
        'feature_array_headers':snapshot['arrays'],'capture_array_headers':headers,'expected_union_rows':UNION_ROWS,
        'memory_accounting':{'feature_archive_bytes':563845063,'feature_ndarray_bytes':npbytes,'largest_array_bytes':495544488,
            'capture_source_rows_bytes':ROWS*8,'block_rows':BLOCK,'peak_status':'NOT_MEASURED; 4GiB hard cap; canonical decoder/provider temporary copies included in scope'},
        'question':d['question'],'runtime_versions':{'numpy':np.__version__,'pandas':raw.pd.__version__}}


def source_clean(plan):
    return (subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='work/gx1-current'
        and subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_base_commit']
        and not subprocess.check_output(['git','status','--porcelain','--','gx1','scripts','tests'],cwd=REPO,text=True)
        and set(plan['source_owners'])==set(OWNERS)and all(raw.saved.native.sha(REPO/p)==h for p,h in plan['source_owners'].items()))


def gate_expected(binding,admitted):
    return {'status':'ADMITTED_EXACT_CPU_ONLY_INPUT_OBSERVATION_CENSUS'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,
        'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}


def inspect(plan_sha,*,admitted):
    need(Path.cwd().resolve()==REPO and type(plan_sha)is str and len(plan_sha)==64 and all(c in '0123456789abcdef'for c in plan_sha),'CWD_PLAN_SHA')
    binding={'path':str(PLAN),'sha256':plan_sha};plan=read(binding)
    need(plan==build_plan(design=plan['inputs']['design'],source_base_commit=plan['source_base_commit'])and source_clean(plan),'EXACT_PLAN_SOURCE')
    pref=ref(POLICY);policy=read(pref);need(all(policy.get(k)is False for k in raw.saved.FLAGS)and policy[STAGE]==gate_expected(binding,admitted),'EXACT_ONE_USE_ALL_TRAINING_CLOSED')
    need(not ROOT.exists()and not ROOT.is_symlink(),'FRESH_ROOT');return plan,binding,pref


def checked_inputs(plan):
    for b in plan['inputs'].values():need(ref(b['path'])==b,'INPUT_SHA')
    raw.saved.archive_headers(plan['inputs']['feature_archive'],plan['feature_array_headers'],exact_inventory=plan['feature_array_headers'])
    raw.saved.archive_headers(plan['inputs']['capture'],{'source_rows':plan['capture_array_headers']['source_rows']},exact_inventory=raw.saved.CAPTURE_KEYS)


def calculate(plan,*,progress=None):
    progress={}if progress is None else progress;progress['phase']='CAPTURE_SOURCE_ROWS'
    rows=raw.saved.load_fields(plan['inputs']['capture'],('source_rows',),plan['capture_array_headers'])['source_rows'];progress['capture_fields_decoded']=['source_rows']
    progress['phase']='FEATURE_CACHE';progress['feature_archive_loads_started']=1;provider=cache.load_snapshot(plan['inputs']['snapshot']);progress['feature_archive_loads_completed']=1
    normalization=read(plan['inputs']['normalization']);main=read(plan['inputs']['main_plan']);unions,ledger=build_unions(provider,rows,normalization,main)
    rules=observation_rules(read(CENSUS),read(ERRATA));results={};progress['surfaces_complete']=[]
    for surface,ids in unions.items():
        progress['phase']=surface;values=provider.surfaces[surface]if surface in ('signal','ctx_cont')else provider.tf_values[surface[4:].upper()]
        contract=normalization['surfaces'][surface];results[surface]=census_surface(values,ids,contract['field_names'],rules[surface],contract)
        progress['surfaces_complete'].append(surface)
    need(sum(len(v['relations'])for v in results.values())==86,'COUNTED_EXACT_86')
    cat=provider.surfaces['ctx_cat'][unions['ctx_cont']];cat_fields=provider.profile['physical_field_names']['ctx_cat'];cats={}
    for j,name in enumerate(cat_fields):
        domain=normalization['ctx_cat']['domains'][name];need(np.isin(cat[:,j],domain).all(),'CTX_CAT_DOMAIN');cats[name]={str(v):int((cat[:,j]==v).sum())for v in domain}
    failed=any(x['qualification_status']=='FAIL'for x in results.values());progress['phase']='COMPLETE_COUNTS'
    return {'rows':len(rows),'union_ledger':ledger,'surfaces':results,'ctx_cat_counts':cats,'relations_counted':86,
        'qualification_status':'FAIL'if failed else'NO_OBSERVED_CONTRACT_VIOLATIONS_REMAINING_UNCLASSIFIED',
        'physical_decode':{'feature_archives':1,'feature_arrays':27,'feature_ndarray_bytes':1433216758,'includes_retained_DEV_feature_values':True,
            'capture_archives':1,'capture_fields':['source_rows'],'analyzed_rows':'Only hash-qualified wholeTRAIN consumed row-unions; no DEV/TEST analyzed'},
        'memory_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'limits':['Finite/zero and event/state absence are not independent vendor/source-known masks; unclassified undefined conditions remain UNKNOWN.',
            'NO_SUPPORT and NOT_EXERCISED are not numerical PASS by vacuity. Only named pairs and source-specific preconditions are tested.',
            'Normalization hard-zero applies only where the unchanged canonical normalizer enforces it; other centered structural zeros may be nonzero.',
            'Archived practice-feed clock parity does not prove market calendar closure, historical quote availability, revision-free macro vintages or learned utility.',
            'WholeTRAIN normalization is reused without fit; this is not prefix-OOF normalization, feature utility, generalization, edge or economic evidence.']}


def publish(name,value):
    path=ROOT/name;raw.saved._publish_bytes(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());return ref(path)


def run(plan_sha):
    began=time.monotonic();cap=require_capped_cpu_audit_execution();lock=require_capped_lock_ancestry()
    need(cap['memory_max']==cap['memory_high']==4*1024**3 and cap['swap']==512*1024**2 and cap['numerical_threads']==1
        and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):],'CAP_LIMITS')
    plan,binding,pref=inspect(plan_sha,admitted=True);checked_inputs(plan);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();ROOT.mkdir(parents=True,exist_ok=False)
    previous={s:signal.getsignal(s)for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};claim=result=None;progress={'phase':'CLAIM'}
    try:
        def expired(*_):raise TimeoutError('INPUT_OBSERVATION_WALL')
        def stopped(s,*_):raise InterruptedError('INPUT_OBSERVATION_EXTERNAL_SIGNAL_'+str(s))
        signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
        claim=publish('CLAIM.json',{'stage':STAGE,'plan':binding,'policy_at_claim':pref,'boot_id':boot,'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
        remaining=EXECUTION['internal_seconds']-(time.monotonic()-began);need(remaining>0,'TIME_BEFORE_CALCULATE');signal.setitimer(signal.ITIMER_REAL,remaining)
        report=calculate(plan,progress=progress);checked_inputs(plan)
        report['memory_cgroup_peak_bytes']=int((Path(cap['cgroup_path'])/'memory.peak').read_text())
        need(source_clean(plan)and ref(POLICY)==pref and ref(PLAN)==binding and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot,'SOURCE_INPUT_PLAN_POLICY_BOOT_BOOKEND')
        result=publish('RESULT.json',{'status':'COMPLETE_NATIVE_INPUT_OBSERVATION_CENSUS','stage':STAGE,'plan':binding,'claim':claim,'policy_at_claim':pref,'boot_id':boot,
            'report':report,'source_unchanged':True,'test_data_used':False,'budget':BUDGET,'elapsed_seconds':time.monotonic()-began})
        terminal=publish('TERMINAL.json',{'status':'COMPLETE','exit_code':0,'stage':STAGE,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,
            'source_unchanged':True,'test_data_used':False});return {'result':result,'terminal':terminal}
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0);publish('TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'stage':STAGE,'plan':binding,'claim':claim,'result':result,
            'remaining_invocations':0,'relaunch_allowed':False,'progress':progress,'error':type(error).__name__+':'+str(error)[:2000],'test_data_used':False});raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in previous.items():signal.signal(s,h)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    need(sys.argv[1:]==['--plan-sha256',args.plan_sha256]+(['--preflight']if args.preflight else[]),'EXACT_CLI')
    if args.preflight:
        plan,binding,pref=inspect(args.plan_sha256,admitted=False);checked_inputs(plan)
        print(json.dumps({'status':'PREFLIGHT_INPUT_OBSERVATION_PASS','plan':binding,'policy':pref,'claim_created':False,'arrays_decoded':0,'models':0,'fits':0,'forwards':0}))
    else:print(json.dumps(run(args.plan_sha256)))

if __name__=='__main__':main()
