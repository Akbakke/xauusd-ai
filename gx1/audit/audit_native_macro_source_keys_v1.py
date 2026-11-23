"""One-use consumed macro-source keys; no feature reconstruction or vintage claim."""
from __future__ import annotations
import argparse,hashlib,importlib.util,io,json,os,resource,signal,subprocess,sys,time,zipfile
from datetime import datetime,timedelta,timezone
from pathlib import Path
from zoneinfo import ZoneInfo
if __name__=='__main__':
    from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
    install_native_m5_direction_io_firewall()
import numpy as np
import pandas as pd
from gx1.audit import audit_native_input_observation_v1 as prior
from gx1.features import d1_macro_context_v1 as macro
from gx1.contracts import gx1_capped_execution_v1 as guard
REPO,POLICY=prior.REPO,prior.POLICY
STAGE='native_macro_source_keys_20261123_001';BASENAME='NATIVE_MACRO_SOURCE_KEYS_20261123_001'
PLAN=REPO/'configs/research'/(BASENAME+'.json');ROOT=Path('/home/andre2/GX1_RUNS')/BASENAME
SELF='gx1/audit/audit_native_macro_source_keys_v1.py';TEST='tests/test_audit_native_macro_source_keys_v1.py'
DESIGN={'path':str(prior.EROOT/'MACRO_SOURCE_KEY_DESIGN001.json'),'sha256':'0ccf3fb82e32671b177015334e57319798dbf7265c47ced3c0b5d97859144bb2'}
WORKER={'path':'/home/andre2/GX1_RUNS/GX1_FULL_PROJECT_AUDIT_20261120_001/MACRO_NUMERIC_WORKER_001.py','sha256':'d04c24982b6941f98ad3888388c1269c75454839312105eae2a9ab944734d635'}
SPEC={'path':str(REPO/'configs/research/D1_MACRO_CONTEXT_SOURCES_20261011_001.json'),'sha256':'0372508910a88993540327b17cf651d519fc4ef7d9b960f245eb969ccdf1ec04'}
NUMERIC_ROOT=Path('/home/andre2/GX1_RUNS/GX1_FULL_PROJECT_AUDIT_20261120_001/FULL_MACRO_NUMERIC001')
CAPTURE_ROOT=Path('/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_20261122_001')
EXECUTION={'class':'audit','memory':'4G','swap':'512M','numerical_threads':1,'outer_seconds':600,'internal_seconds':570,'kill_after_seconds':10}
START,END,DAY,M5=1609459200000000000,1767225600000000000,86400000000000,300000000000
CAPTURE_FIELDS=('source_rows','issue_ns');FEATURE_FIELDS=('bar_start_ns','retained_source_rows')
LEAVES=('nominal2y','nominal10y','real10y','vix_close','gld_tonnes')
DRIVER_DEPS={'real10y_level':[('real10y',0)],**{f'real10y_change{k}D':[('real10y',0),('real10y',k)]for k in (1,5,21)},
 'nominal10y_level':[('nominal10y',0)],'slope10y2y_level':[('nominal10y',0),('nominal2y',0)],
 'slope10y2y_change21D':[(p,k)for k in (0,21)for p in ('nominal10y','nominal2y')],
 'breakeven10y_level':[('nominal10y',0),('real10y',0)],'breakeven10y_change21D':[(p,k)for k in (0,21)for p in ('nominal10y','real10y')],
 'vix_log_level':[('vix_close',0)],'vix_log_change5D':[('vix_close',0),('vix_close',5)],
 'gld_log_level':[('gld_tonnes',0)],'gld_log_change21D':[('gld_tonnes',0),('gld_tonnes',21)]}
OWNERS=tuple(dict.fromkeys((*prior.OWNERS,SELF,TEST,'gx1/features/d1_macro_context_v1.py','gx1/scripts/research_ta_campaign_v1.py')))
ref,read,array_sha=prior.ref,prior.read,prior.array_sha


def need(ok,why):
    if not bool(ok):raise ValueError('NATIVE_MACRO_SOURCE_KEYS_'+why)


def json_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def asof_positions(issues,available,*,inclusive):
    need(issues.dtype==available.dtype==np.int64 and issues.ndim==available.ndim==1 and len(available)>0
        and np.all(np.diff(available)>=0)and np.all(np.diff(issues)>0)and type(inclusive)is bool,'ASOF_AXES')
    return np.searchsorted(available,issues,side='right'if inclusive else'left').astype(np.int64)-1


def dependency_rows(join,lag):
    need(join.dtype==np.int64 and join.ndim==1 and type(lag)is int and lag>=0,'DEPENDENCY_INDEX')
    rows=join-lag;rows[(join<0)|(rows<0)]=-1;return rows


def primitive_origins(tables):
    merged={};sources={}
    for source_id,table in tables:
        for date,values in table.items():
            row=merged.setdefault(date,{})
            for leaf,value in values.items():
                need(leaf in LEAVES and leaf not in row,'DUPLICATE_PRIMITIVE_DATE');row[leaf]=value;sources[(date,leaf)]=source_id
    dates=sorted(merged);origins={};key_axes={};presence={};invalid_keys=[]
    for leaf in LEAVES:
        values=np.asarray([merged[d].get(leaf,np.nan)for d in dates],np.float64);present=np.asarray(pd.notna(values),np.bool_)
        origin=np.maximum.accumulate(np.where(present,np.arange(len(dates),dtype=np.int64),-1));keys=[f'{sources[(d,leaf)]}|{leaf}|{d}'if present[j]else None for j,d in enumerate(dates)]
        invalid=present&(~np.isfinite(values)|((values<=0)if leaf in ('vix_close','gld_tonnes')else False))
        invalid_keys.extend(keys[j]for j in np.flatnonzero(invalid));origins[leaf]=origin;key_axes[leaf]=keys
        presence[leaf]={'parsed_present':int(present.sum()),'missing':int((~present).sum()),'nonfinite_invalid':int((present&~np.isfinite(values)).sum()),
            'nonpositive_log_primitive':int((present&(values<=0)).sum())if leaf in ('vix_close','gld_tonnes')else 0,
            'present_axis_sha256':array_sha(present),'ffill_origin_axis_sha256':array_sha(origin),'raw_key_axis_sha256':json_sha(keys)}
    return {'dates':dates,'origins':origins,'key_axes':key_axes,'presence':presence,'invalid_keys':sorted(invalid_keys)}


def panel_close_keys(times_ns,source_id):
    need(times_ns.dtype==np.int64 and times_ns.ndim==1 and np.all(np.diff(times_ns)>0),'PANEL_TIME_ORDER')
    labels=((times_ns-22*3600000000000)//DAY)*DAY+22*3600000000000;keep=labels+DAY<END;times=times_ns[keep];labels=labels[keep]
    starts,first,counts=np.unique(labels,return_index=True,return_counts=True);last=first+counts-1
    return {'available':starts+DAY,'keys':[f'{source_id}|D1label={int(d)}|closeH1={int(t)}'for d,t in zip(starts,times[last])],
        'time_axis_sha256':array_sha(times),'last_H1_axis_sha256':array_sha(times[last]),'observed_H1_rows':len(times)}


def make_field_ledger(issues_by_surface,available,deps,key_axes,origins=None,*,inclusive=False):
    result={}
    for surface,issues in issues_by_surface.items():
        join=asof_positions(issues,available,inclusive=inclusive);supported=np.ones(len(join),np.bool_);keys=set();hashes={};mapped_rows=[]
        for axis,lag in deps:
            rows=dependency_rows(join,lag);valid=rows>=0
            if origins is not None:
                mapped=np.full(len(rows),-1,np.int64);mapped[valid]=origins[axis][rows[valid]];rows=mapped
            valid=rows>=0;supported&=valid;hashes[f'{axis}@-{lag}']=array_sha(rows)
            mapped_rows.append((axis,rows))
        for axis,rows in mapped_rows:keys.update(key_axes[axis][int(j)]for j in np.unique(rows[supported]))
        need(None not in keys,'MISSING_ORIGIN_KEY')
        result[surface]={'issue_rows':len(issues),'issue_axis_sha256':array_sha(issues),'join_index_sha256':array_sha(join),'availability_axis_sha256':array_sha(available),
            'dependency_index_sha256':hashes,'supported_issue_rows':int(supported.sum()),'unsupported_issue_rows':int((~supported).sum()),'raw_keys':sorted(keys),'raw_key_count':len(keys),'key_scope':'All dependencies of supported issued-field rows only',
            'support_status':'SUPPORTED'if supported.all()else'NO_SUPPORT'if not supported.any()else'PARTIAL_SUPPORT'}
    signal=set(result['signal']['raw_keys']);context=set(result['context']['raw_keys'])
    return {'dependencies':[{'axis':a,'lag_observed_rows':k}for a,k in deps],'raw_key_axis_sha256':{a:json_sha(key_axes[a])for a in sorted({x for x,_ in deps})},'surfaces':result,'context_keys_subset_of_signal':context<=signal,
        'signal_only_raw_keys':sorted(signal-context),'context_only_raw_keys':sorted(context-signal)}


def load_primitive_worker():
    need(ref(WORKER['path'])==WORKER,'IMMUTABLE_WORKER');spec=importlib.util.spec_from_file_location('gx1_macro_key_primitive_owner',WORKER['path']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.END=END;return module


def canonical_csv_primitives(worker,record,kind):
    need(worker.END==END,'WORKER_TRAIN_BOUND');old=worker.numeric_text;worker.numeric_text=lambda value:str(value).strip()
    try:table=worker.csv_primitives([record],kind)
    finally:worker.numeric_text=old
    dates=sorted(table);fields=sorted({k for row in table.values()for k in row});count=0
    for leaf in fields:
        tokens=[table[d].get(leaf,np.nan)for d in dates]
        need(not any(isinstance(x,str)and x.lower()in ('true','false')for x in tokens),'AMBIGUOUS_CSV_BOOLEAN_'+leaf)
        converted=pd.to_numeric(pd.Series(tokens,dtype=object),errors='coerce');count+=len(tokens)
        for d,value in zip(dates,converted):table[d][leaf]=value
    return table,{'eligible_numeric_cells_converted':count,'ambiguous_boolean_tokens_rejected':True,'parser':'immutable date/header reader + eligible rawtext + canonical pd.to_numeric(errors=coerce), then pd.notna; not generic read_csv dtype equivalence'}


def source_metadata(spec):
    bindings={'cot':spec['cot']['result'],'panel':spec['oanda_panel']['result'],**{k:spec['drivers'][k]for k in ('treasury','vix','gld')}};records=[]
    for group,binding in bindings.items():
        result=read(binding)
        for item in result['records']:
            need(item['status']=='RETRIEVED_NOT_ADMITTED','SOURCE_STATUS');path=item['parquet_path']if group=='panel'else item['raw_path'];digest=item['parquet_sha256']if group=='panel'else item['raw_sha256']
            p=Path(path);need(p.is_absolute()and p.resolve()==p and p.is_file()and not p.is_symlink(),'SOURCE_PATH')
            records.append({'group':group,'source_id':group+'|'+str(item['item']),'file':{'path':path,'sha256':digest},'bytes':p.stat().st_size,'record':item})
    need(len({r['source_id']for r in records})==len(records)==67,'EXACT67_SOURCES')
    need({g:sum(r['group']==g for r in records)for g in bindings}=={'cot':18,'panel':11,'treasury':36,'vix':1,'gld':1},'SOURCE_GROUP_COUNTS')
    return bindings,records


def build_plan(design_ref,*,source_base_commit=None):
    d=read(design_ref);need(design_ref==DESIGN and d['schema_version']=='gx1_native_macro_source_key_design_v1'and d['execution_authorized']is False
        and d['old_qualified_primitive_worker']==WORKER and d['capture_archive_members']==list(CAPTURE_FIELDS)and d['feature_archive_members']==list(FEATURE_FIELDS),'LOCKED_DESIGN')
    b=d['budget'];need(all(b[k]==EXECUTION[k]for k in ('memory','swap','numerical_threads','outer_seconds','internal_seconds'))
        and all(b[k]==0 for k in ('checkpoint_loads','feature_formula_reconstructions','model_constructions','model_forwards','new_network_calls','normalization_or_prior_fits','optimizer_updates','VAL_TEST_rows_analyzed','native_TEST_split_files')),'DESIGN_BUDGET')
    main_ref=ref(prior.MAIN);need(main_ref['sha256']==prior.MAIN_SHA,'ORIGINAL_MAIN');main=read(main_ref);meta=main['metadata'];snapshot=read(meta['source_snapshot']);normalization=read(meta['normalization'])
    capture_result_ref=ref(CAPTURE_ROOT/'RESULT.json');capture=read(capture_result_ref);terminal_ref=ref(CAPTURE_ROOT/'TERMINAL.json');terminal=read(terminal_ref);capture_plan=read(capture['plan'])
    need(terminal['status']=='COMPLETE'and terminal['exit_code']==0 and terminal['result']==capture_result_ref and capture['status']=='COMPLETE_NATIVE_FULL_TRAIN_CAPTURE'
        and capture['source_unchanged']is True and capture['test_data_used']is False and capture_plan['inputs']['main_plan']==main_ref,'AUTHENTIC_CAPTURE')
    ch=capture['report']['capture_array_headers'];need(ch['issue_ns']['sha256']==main['schedule']['issue_ns_sha256']and ch['source_rows']['sha256']==main['schedule']['source_rows_sha256'],'CAPTURE_TRAIN_HEADERS')
    need(snapshot['surfaces_metadata']['axes']['contract_sha256']==normalization['source_axes_sha256']and normalization['fit_population']['local_rows']==354645
        and normalization['fit_population']['context_rows']==354550,'ACTUAL_NORMALIZATION_AXES')
    spec=read(SPEC);need(spec['cot']['contract_market_code']=='088691'and spec['cot']['availability_lag_days']==7 and set(DRIVER_DEPS)==set(macro.D1_MACRO_DRIVER_FIELDS),'CURRENT_SOURCE_CONTRACT')
    source_results,records=source_metadata(spec);nr_ref=ref(NUMERIC_ROOT/'RESULT.json');nr=read(nr_ref);nt_ref=ref(NUMERIC_ROOT/'TERMINAL.json');nt=read(nt_ref);contract=read(nr['contract'])
    need(nt['status']=='COMPLETE'and nt['exit_code']==0 and nt['result']==nr_ref and nr['status']=='COMPLETE_CURRENT61_MACRO_FORMULAS_AND_DECLARED_ASOF'
        and nr['source_unchanged']is True and contract['source_owners'][WORKER['path']]==WORKER['sha256'],'PRIOR_PANEL_FINITE_POSITIVE_PROOF')
    old={(x['path'],x['sha256'])for x in contract['public_macro_source_bindings']};need(all((r['file']['path'],r['file']['sha256'])in old for r in records if r['group']=='panel'),'UNCHANGED11_PANEL_SOURCES')
    clock=spec['cot']['publication_clock'];clock_ref={'path':str(REPO/clock['path']),'sha256':clock['sha256']};read(clock_ref);need(ref(WORKER['path'])==WORKER,'WORKER_SOURCE_SHA')
    inputs={'design':design_ref,'main_plan':main_ref,'snapshot':meta['source_snapshot'],'feature_archive':snapshot['archive'],'normalization':meta['normalization'],
        'capture_result':capture_result_ref,'capture_terminal':terminal_ref,'capture_plan':capture['plan'],'capture':capture['report']['capture'],
        'source_spec':SPEC,'publication_clock':clock_ref,'primitive_worker':WORKER,'numeric_result':nr_ref,'numeric_terminal':nt_ref,'numeric_contract':nr['contract'],
        **{'source_result_'+k:v for k,v in source_results.items()}}
    npbytes=sum(int(np.prod(h['shape']))*np.dtype(h['dtype']).itemsize for h in [*(ch[k]for k in CAPTURE_FIELDS),*(snapshot['arrays'][k]for k in FEATURE_FIELDS)])
    need(npbytes==18513536 and sum(r['bytes']for r in records)==116879913,'PHYSICAL_HEADER_BUDGET')
    return {'schema_version':'gx1_native_macro_source_key_plan_v1','stage':STAGE,'status':'PREPARED_NOT_ADMITTED','plan_path':str(PLAN),'output_root':str(ROOT),
        'source_base_commit':source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'source_owners':{n:prior.raw.saved.native.sha(REPO/n)for n in OWNERS},'inputs':inputs,'source_records':records,'execution':EXECUTION,
        'capture_array_headers':ch,'feature_array_headers':snapshot['arrays'],'question':d['question'],
        'budget':{'capture_archive_loads':1,'capture_fields':list(CAPTURE_FIELDS),'feature_archive_loads':1,'feature_fields':list(FEATURE_FIELDS),'ndarray_bytes_decoded':npbytes,
            'raw_containers':67,'raw_container_bytes_hashed':116879913,'COT_date_code_only':18,'panel_time_only':11,'treasury_csv':36,'vix_csv':1,'gld_xlsx':1,
            'raw_driver_primitive_columns':5,'driver_feature_reconstructions':0,'COT_position_value_conversions':0,'panel_close_value_conversions':0,'numeric_source_cutoff_ns':END,
            'model_forwards':0,'models':0,'checkpoint_loads':0,'fits':0,'HTTP_calls':0,'VAL_TEST_rows_analyzed':0},
        'parser_shim':'Unmodified immutable csv_primitives date/header/END filter; temporary numeric_text rawtext shim then per-column pd.to_numeric; reject ambiguous boolean lexical cases. GLD unchanged plain-decimal regex. pd.notna precedes ffill; Inf never silently treated missing.',
        'runtime_versions':{'numpy':np.__version__,'pandas':pd.__version__}}


def checked_inputs(plan):
    for b in plan['inputs'].values():need(ref(b['path'])==b,'INPUT_SHA')
    for r in plan['source_records']:need(ref(r['file']['path'])==r['file']and Path(r['file']['path']).stat().st_size==r['bytes'],'RAW_CONTAINER_SHA')
    prior.raw.saved.archive_headers(plan['inputs']['capture'],{k:plan['capture_array_headers'][k]for k in CAPTURE_FIELDS},exact_inventory=prior.raw.saved.CAPTURE_KEYS)
    prior.raw.saved.archive_headers(plan['inputs']['feature_archive'],{k:plan['feature_array_headers'][k]for k in FEATURE_FIELDS},exact_inventory=plan['feature_array_headers'])


def source_clean(plan):
    return (subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='work/gx1-current'
        and subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_base_commit']
        and not subprocess.check_output(['git','status','--porcelain','--','gx1','scripts','tests'],cwd=REPO,text=True)
        and set(plan['source_owners'])==set(OWNERS)and all(prior.raw.saved.native.sha(REPO/n)==v for n,v in plan['source_owners'].items()))


def gate_expected(binding,admitted):
    return {'status':'ADMITTED_EXACT_CPU_ONLY_MACRO_SOURCE_KEYS'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,
        'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}


def inspect(plan_sha,*,admitted):
    need(Path.cwd().resolve()==REPO and type(plan_sha)is str and len(plan_sha)==64 and all(c in '0123456789abcdef'for c in plan_sha),'CWD_SHA')
    binding={'path':str(PLAN),'sha256':plan_sha};plan=read(binding);need(plan==build_plan(plan['inputs']['design'],source_base_commit=plan['source_base_commit'])and source_clean(plan),'EXACT_PLAN_SOURCE')
    pref=ref(POLICY);policy=read(pref);need(all(policy.get(k)is False for k in prior.raw.saved.FLAGS)and policy[STAGE]==gate_expected(binding,admitted),'ALL_TRAINING_CLOSED_GATE')
    need(not ROOT.exists()and not ROOT.is_symlink(),'FRESH_ROOT');return plan,binding,pref


def require_timeout_ancestry(plan_sha,scope):
    pid=os.getpid();runner=REPO/'scripts/gx1_capped_run.sh';expected=['--signal=TERM','--kill-after=10s','600s'];seen=set()
    while pid and pid not in seen and len(seen)<64:
        seen.add(pid);p=Path('/proc')/str(pid);identity=guard._lock_process(pid,runner);args=os.fsdecode(guard._read_lock_proc(p/'cmdline')).rstrip('\0').split('\0')
        if Path(os.readlink(p/'exe'))==Path('/usr/bin/timeout').resolve():
            operand=Path(args[4])if len(args)>4 else Path('/invalid');operand=operand if operand.is_absolute()else Path(os.readlink(p/'cwd'))/operand
            need(args[1:4]==expected and operand==REPO/'.venv/bin/python'and args[5:]==['-m','gx1.audit.audit_native_macro_source_keys_v1','--plan-sha256',plan_sha]
                and identity.cgroup==scope and guard._lock_process(pid,runner)==identity,'CANONICAL_OUTER_TIMEOUT')
            return {'pid':pid,'start_ticks':identity.start_ticks,'seconds':600,'scope':scope,'exact_cli':True}
        pid=identity.parent_pid
    need(False,'MISSING_CANONICAL_OUTER_TIMEOUT')


def _cot_keys(records,spec,calendar):
    delayed={r['as_of']:int(pd.Timestamp(datetime.fromisoformat(r['publication_date_ny']).date()+timedelta(days=1),tz='America/New_York').tz_convert('UTC').value)for r in calendar['rows']};rows=[]
    wanted=('As of Date in Form YYYY-MM-DD','CFTC Contract Market Code')
    for record in records:
        with zipfile.ZipFile(record['file']['path'])as z:raw=z.read('annual.txt')
        need(hashlib.sha256(raw).hexdigest()==record['record']['members']['annual.txt']['sha256'],'COT_MEMBER_SHA')
        frame=pd.read_csv(io.BytesIO(raw),usecols=lambda n:n.strip()in wanted,dtype=str,keep_default_na=False);frame.columns=[c.strip()for c in frame.columns];need(set(frame.columns)==set(wanted),'COT_TWO_HEADERS')
        for date,code in frame.loc[frame[wanted[1]].str.strip()==spec['cot']['contract_market_code'],list(wanted)].itertuples(index=False,name=None):
            date=date.strip()
            if date>='2026-01-01':continue
            available=max(int(pd.Timestamp(date,tz='UTC').value)+spec['cot']['availability_lag_days']*DAY,delayed.get(date,0))
            if available<END:rows.append((date,available,f"{record['source_id']}|code={code.strip()}|as_of={date}"))
    rows.sort();need(len(rows)>4 and len({r[0]for r in rows})==len(rows),'COT_UNIQUE_DATES')
    return np.asarray([r[1]for r in rows],np.int64),[r[2]for r in rows]


def calculate(plan,*,progress=None):
    progress={}if progress is None else progress;progress['phase']='AXES_ONLY'
    cap=prior.raw.saved.load_fields(plan['inputs']['capture'],CAPTURE_FIELDS,plan['capture_array_headers']);axes=prior.raw.saved.load_fields(plan['inputs']['feature_archive'],FEATURE_FIELDS,plan['feature_array_headers'])
    rows,issue,starts,retained=cap['source_rows'],cap['issue_ns'],axes['bar_start_ns'],axes['retained_source_rows'];main=read(plan['inputs']['main_plan']);norm=read(plan['inputs']['normalization'])
    need(rows.shape==issue.shape==(354550,)and rows.dtype==issue.dtype==starts.dtype==retained.dtype==np.int64 and np.all(np.diff(starts)>0)
        and np.all(np.diff(rows)>0)and np.all(np.diff(retained)>0)and np.array_equal(starts[rows]+M5,issue)
        and array_sha(rows)==main['schedule']['source_rows_sha256']and array_sha(issue)==main['schedule']['issue_ns_sha256'],'EXACT_TRAIN_AXES')
    whole=np.flatnonzero((starts+M5>=START)&(starts+M5<END));need(np.array_equal(rows,whole),'WHOLE_TRAIN_SOURCE_ROWS')
    local=prior.features._window_union(rows,prior.features.MODEL_NATIVE_SEQ_LEN,len(starts));at=np.searchsorted(retained,local)
    need(np.all(at<len(retained))and np.array_equal(retained[at],local)and len(local)==norm['fit_population']['local_rows']==354645
        and len(rows)==norm['fit_population']['context_rows']==354550 and array_sha(issue)==norm['fit_population']['selected_issue_ns_sha256']
        and array_sha(rows)==norm['fit_population']['selected_source_rows_sha256'],'ACTUAL_OBSERVED_HISTORY_UNION')
    issues={'signal':starts[local]+M5,'context':issue};ledger={};spec=read(plan['inputs']['source_spec']);calendar=read(plan['inputs']['publication_clock']);records=plan['source_records']
    progress['phase']='COT_DATE_CODE';available,keys=_cot_keys([r for r in records if r['group']=='cot'],spec,calendar)
    for field in macro.D1_MACRO_COT_FIELDS:
        deps=[('report',0),('report',4)]if field in ('cot_change4reports','cot_open_interest_log_change4reports')else[('report',0)]
        ledger[field]=make_field_ledger(issues,available,deps,{'report':keys})
    panel_report={};progress['phase']='PANEL_TIME_ONLY'
    for record in (r for r in records if r['group']=='panel'):
        name=record['record']['item'];read_end=((END-1-22*3600000000000)//DAY)*DAY+22*3600000000000
        frame=pd.read_parquet(record['file']['path'],columns=['time'],filters=[('time','<',pd.Timestamp(read_end,tz='UTC'))]);times=pd.DatetimeIndex(pd.to_datetime(frame['time'],utc=True)).as_unit('ns').asi8.copy();times.sort()
        need(np.all(times<read_end),'PANEL_TIME_PREDICATE');panel=panel_close_keys(times,record['source_id']);panel_report[name]={k:v for k,v in panel.items()if k not in ('keys','available')}
        panel_report[name].update(availability_axis_sha256=array_sha(panel['available']),raw_key_axis_sha256=json_sha(panel['keys']),close_value_proof='Reused unchangedSHA+oldqualifiedfinite-positive H1close witness; no close column decoded here')
        for lag in (1,5,20):ledger[f'{name}_logret_{lag}D']=make_field_ledger(issues,panel['available'],[('close',0),('close',lag)],{'close':panel['keys']},inclusive=True)
        ledger[name+'_rv_20D']=make_field_ledger(issues,panel['available'],[('close',k)for k in range(21)],{'close':panel['keys']},inclusive=True)
    progress['phase']='FIVE_PRIMITIVE_PRESENCE';worker=load_primitive_worker();tables=[];parser_counts=[]
    for record in (r for r in records if r['group']in ('treasury','vix','gld')):
        if record['group']=='gld':table=worker.xlsx_gld(record['record']);counts={'eligible_numeric_cells_converted':len(table),'parser':'unchangedGLDplain-decimalregex'}
        else:table,counts=canonical_csv_primitives(worker,record['record'],record['group'])
        tables.append((record['source_id'],table));parser_counts.append({'source_id':record['source_id'],**counts})
    primitives=primitive_origins(tables);available=np.asarray([worker.driver_availability(datetime.fromisoformat(d).date())for d in primitives['dates']],np.int64)
    need(np.all(available<END),'DRIVER_TRAIN_VALUE_BOUND')
    for field,deps in DRIVER_DEPS.items():ledger[field]=make_field_ledger(issues,available,deps,primitives['key_axes'],primitives['origins'])
    need(set(ledger)==set(macro.D1_MACRO_CONTEXT_FIELDS)and len(ledger)==61,'ALL61_FIELDS');invalid=set(primitives['invalid_keys']);consumed=set().union(*(set(v['surfaces']['signal']['raw_keys'])for v in ledger.values()))
    need(all(v['context_keys_subset_of_signal']for v in ledger.values()),'SAME_FIELD_CONTEXT_SIGNAL_REACH');bad=sorted(invalid&consumed);unsupported=any(x['unsupported_issue_rows']for v in ledger.values()for x in v['surfaces'].values());progress['phase']='COMPLETE_KEYS'
    return {'field_ledger':ledger,'field_count':61,'issue_populations':{s:{'rows':len(v),'issue_axis_sha256':array_sha(v)}for s,v in issues.items()},
        'local_source_row_union_sha256':array_sha(local),'context_source_rows_sha256':array_sha(rows),'panel_time_proofs':panel_report,
        'driver_date_axis_sha256':json_sha(primitives['dates']),'driver_presence':primitives['presence'],'driver_parser_counts':parser_counts,
        'raw_invalid_keys_consumed':bad,'qualification_status':'FAIL_INVALID_CONSUMED_PRIMITIVE'if bad else'UNSUPPORTED_SOURCE_DEPENDENCY'if unsupported else'EXACT_KEYS_NOT_VINTAGE_QUALIFICATION',
        'counts':{'raw_containers':67,'COT_position_values_converted':0,'panel_close_values_converted':0,'driver_primitive_numeric_cells_converted':sum(x['eligible_numeric_cells_converted']for x in parser_counts),
            'feature_formula_reconstructions':0,'NPZ_arrays_decoded':4,'models':0,'forwards':0,'fits':0,'native_TEST_split_files':0},
        'memory_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'limits':['Exact current-function source keys and parsed presence only; not firstvintage, original release timing, source calendar, learned utility, OOS or economic edge.',
            'All bound public containers are hash-read, including public afterseal bytes; COTdate/code, paneltime and eligible5drivercolumns only are decoded. No at/after2026-01-01 macro value numeric conversion.',
            'CSV default boolean inference is ambiguous and rejected; accepted ordinary numeric/NA presence uses pd.to_numeric then pd.notna. Inf remains present and is separately invalid, never silently skipped.',
            'History uses observed source indices; RV20 uses all21 observedD1closes; driver lags use outermerged dates with perleaf source predecessors, notcalendar or vendor-own days.']}


def publish(name,value):
    path=ROOT/name;prior.raw.saved._publish_bytes(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());return ref(path)


def run(plan_sha):
    began=time.monotonic();cap=guard.require_capped_cpu_audit_execution();lock=guard.require_capped_lock_ancestry()
    need(cap['memory_max']==cap['memory_high']==4*1024**3 and cap['swap']==512*1024**2 and cap['numerical_threads']==1 and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):],'CAP_LIMITS')
    timeout=require_timeout_ancestry(plan_sha,lock['scope']);plan,binding,pref=inspect(plan_sha,admitted=True);checked_inputs(plan);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();ROOT.mkdir(parents=True,exist_ok=False)
    handlers={s:signal.getsignal(s)for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};claim=result=None;progress={'phase':'CLAIM'}
    try:
        def expired(*_):raise TimeoutError('MACRO_KEYS_WALL')
        def stopped(s,*_):raise InterruptedError('MACRO_KEYS_EXTERNAL_SIGNAL_'+str(s))
        signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
        claim=publish('CLAIM.json',{'stage':STAGE,'plan':binding,'policy_at_claim':pref,'boot_id':boot,'outer_timeout':timeout,'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
        remaining=EXECUTION['internal_seconds']-(time.monotonic()-began);need(remaining>0,'TIME_BEFORE_READS');signal.setitimer(signal.ITIMER_REAL,remaining)
        report=calculate(plan,progress=progress);checked_inputs(plan)
        need(source_clean(plan)and ref(POLICY)==pref and ref(PLAN)==binding and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot,'SOURCE_INPUT_PLAN_POLICY_BOOT_BOOKEND')
        ledger=publish('SOURCE_KEYS.json',{'schema_version':'gx1_native_macro_raw_key_ledger_v1','plan':binding,'fields':report.pop('field_ledger')});report['source_key_ledger']=ledger
        result=publish('RESULT.json',{'status':'COMPLETE_NATIVE_MACRO_SOURCE_KEYS','stage':STAGE,'plan':binding,'claim':claim,'report':report,'source_unchanged':True,'test_data_used':False,'elapsed_seconds':time.monotonic()-began})
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
        plan,binding,pref=inspect(args.plan_sha256,admitted=False);checked_inputs(plan);print(json.dumps({'status':'PREFLIGHT_MACRO_SOURCE_KEYS_PASS','plan':binding,'policy':pref,'claim_created':False,'payloads_decoded':0,'models':0}))
    else:print(json.dumps(run(args.plan_sha256)))

if __name__=='__main__':main()
