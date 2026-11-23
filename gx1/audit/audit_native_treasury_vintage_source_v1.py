"""One-use quarantined ALFRED initial-release responses; no observation decoding."""
from __future__ import annotations
import argparse,csv,hashlib,io,json,os,re,signal,ssl,subprocess,sys,time,zipfile
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request,ProxyHandler,HTTPRedirectHandler,HTTPSHandler,build_opener
_HTTP=None

def audit_event(event,args):
    if event=='import'and(args[0]=='torch'or args[0].startswith('torch.')):raise RuntimeError('TREASURY_SOURCE_TORCH_FORBIDDEN')
    if event=='urllib.Request':
        need(_HTTP is not None and not _HTTP['seen']and args[0]==_HTTP['url']and args[1]==_HTTP['body']and args[3]=='POST','HTTP_OUTSIDE_EXACT_ADMITTED_REQUEST')
        need(not any(k.lower()in ('authorization','proxy-authorization','cookie')for k in args[2]),'HTTP_CREDENTIALS');_HTTP['seen']=True

if __name__=='__main__':
    sys.addaudithook(audit_event)
    from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
    install_native_m5_direction_io_firewall()
from gx1.audit import audit_native_macro_source_keys_v1 as base
from gx1.contracts import gx1_capped_execution_v1 as guard
REPO,POLICY=base.REPO,base.POLICY
STAGE='native_treasury_initial_release_source_20261123_001';BASENAME='NATIVE_TREASURY_INITIAL_RELEASE_SOURCE_20261123_001'
PLAN=REPO/'configs/research'/(BASENAME+'.json');ROOT=Path('/home/andre2/GX1_RUNS')/BASENAME
SELF='gx1/audit/audit_native_treasury_vintage_source_v1.py';TEST='tests/test_audit_native_treasury_vintage_source_v1.py'
EROOT=base.prior.EROOT
MANIFEST={'path':str(EROOT/'TREASURY_INITIAL_RELEASE_REFERENCE_MANIFEST001.json'),'sha256':'bec6075acba9c62c2c3ef2f6c70935493b3c94d1cdf97e75439796a987dc4413'}
DESIGN={'path':str(EROOT/'TREASURY_INITIAL_RELEASE_SOURCE_DESIGN001.json'),'sha256':'f69d9292f5d0e3f724be69b8d150a399ec3bf69c1d1d36bef27f0ad1443bbc70'}
INVENTORY={'path':str(EROOT/'ALFRED_DOWNLOAD_FORM_INVENTORY003.json'),'sha256':'a58593113ea9603aba1338f135c16a2e112aa84f2c95e6b309e5b928baede993'}
SERIES=(('DGS2','nominal2y'),('DGS10','nominal10y'),('DFII10','real10y'))
FIELDS=[['form[units]','lin'],['form[obs_start_date]','2020-11-30'],['form[obs_end_date]','2025-12-29'],['form[file_type]','4'],['form[file_format]','csv'],['form[download_data]','']]
EXECUTION={'class':'audit','memory':'4G','swap':'512M','numerical_threads':1,'outer_seconds':600,'internal_seconds':570,'kill_after_seconds':10}
OWNERS=tuple(dict.fromkeys((*base.OWNERS,SELF,TEST,'gx1/contracts/native_m5_direction_io_firewall_v1.py')))
ref,read=base.ref,base.read
BODY_MAX,HEADER_MAX,ZIP_MAX,MEMBERS_MAX=1048576,65536,16777216,16


def need(ok,why):
    if not bool(ok):raise ValueError('NATIVE_TREASURY_SOURCE_'+why)


def body_bytes(item):
    return urlencode([tuple(pair)for pair in item['post_fields']]).encode('ascii')


def locked_manifest(manifest):
    need(manifest['schema_version']=='gx1_treasury_initial_release_reference_manifest_v1'and manifest['execution_authorized']is False
        and manifest['training_authorized']is False and manifest['form_inventory']==INVENTORY,'MANIFEST_IDENTITY')
    expected={'series_requests_max':3,'request_timeout_seconds':20,'max_body_bytes':BODY_MAX,'overflow_probe_bytes':1,
        'max_stored_oversize_prefix_bytes':BODY_MAX,'max_requests_per_series':1,'redirects_allowed':0,'retries_allowed':0,
        'credentials_cookies_APIkeys_supplied':False,'proxies_allowed':False,'TLS_certificate_and_hostname_verification_required':True}
    need(manifest['request_limits']==expected,'EXACT_REQUEST_LIMITS');limits=manifest['container_limits']
    need(limits['format']=='ZIP_ONLY'and limits['max_header_bytes']==HEADER_MAX and limits['max_members']==MEMBERS_MAX
        and limits['max_uncompressed_total_bytes']==ZIP_MAX and all(limits[k]is True for k in ('no_disk_extraction','no_encryption','no_path_traversal_or_absolute_members')),'EXACT_ZIP_LIMITS')
    need(len(manifest['series'])==3,'EXACT_THREE_SERIES')
    for item,(name,primitive) in zip(manifest['series'],SERIES):
        need(item['id']==name and item['primitive']==primitive and item['method']=='POST'and item['url']=='https://alfred.stlouisfed.org/series/downloaddata?seid='+name
            and item['post_fields']==FIELDS and hashlib.sha256(body_bytes(item)).hexdigest()==item['post_body_sha256'],'EXACT_POST')
    return manifest


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):return None


def make_opener():
    context=ssl.create_default_context();need(context.check_hostname and context.verify_mode==ssl.CERT_REQUIRED,'VERIFIED_TLS')
    return build_opener(ProxyHandler({}),NoRedirect(),HTTPSHandler(context=context))


def fetch_response(item,*,opener=None):
    """Exactly one request; HTTP errors still carry a bounded response body."""
    need(_HTTP is not None and _HTTP['url']==item['url']and _HTTP['body']==body_bytes(item),'HTTP_NOT_ADMITTED')
    request=Request(item['url'],data=body_bytes(item),method='POST',headers={'Content-Type':'application/x-www-form-urlencoded','User-Agent':'GX1 offline reference qualification'})
    try:response=(make_opener()if opener is None else opener).open(request,timeout=20)
    except HTTPError as error:response=error
    with response:
        metadata={'HTTP_status':response.getcode(),'content_type':response.headers.get('Content-Type'),'content_encoding':response.headers.get('Content-Encoding'),
            'content_length':response.headers.get('Content-Length'),'final_url':response.geturl()}
        raw=bytearray();complete=False;read_error=None
        try:
            while len(raw)<=BODY_MAX:
                part=response.read(min(65536,BODY_MAX+1-len(raw)))
                if not part:
                    complete=getattr(response,'length',0)in (None,0)
                    if not complete:read_error='TRUNCATED_DECLARED_CONTENT_LENGTH'
                    break
                raw.extend(part)
        except Exception as error:
            partial=getattr(error,'partial',b'');raw.extend(partial[:max(0,BODY_MAX+1-len(raw))])
            read_error=type(error).__name__
        overflow=len(raw)>BODY_MAX;complete=complete and not overflow
        metadata.update(body_bytes_observed=len(raw),oversize=overflow,body_complete=complete,read_error=read_error,
            full_response_sha256=hashlib.sha256(raw).hexdigest()if complete else'UNKNOWN')
    return bytes(raw[:BODY_MAX]),metadata


def safe_member(name):
    need(type(name)is str and name and '\\'not in name and ':'not in name and not any(ord(c)<32 for c in name),'ZIP_MEMBER_PATH')
    parts=name.rstrip('/').split('/');p=PurePosixPath(name)
    need(not p.is_absolute()and all(x not in ('','.','..')for x in parts)and str(p)==name.rstrip('/'),'ZIP_MEMBER_PATH')


def csv_header_candidate(firstline):
    """Only identifier-like CSV headers are decoded; never log a possible value row."""
    need(len(firstline)<=HEADER_MAX,'CSV_FIRSTLINE_LIMIT')
    line=firstline.removeprefix(b'\xef\xbb\xbf').rstrip(b'\r\n')
    token=rb'(?:"[A-Za-z_][A-Za-z0-9_ ./()%:+-]*"|[A-Za-z_][A-Za-z0-9_ ./()%:+-]*)'
    need(re.fullmatch(token+rb'(?:,'+token+rb')+',line)is not None,'UNKNOWN_NONLEXICAL_CSV_FIRSTLINE')
    fields=next(csv.reader([line.decode('ascii')],strict=True))
    return {'status':'LEXICAL_HEADER_CANDIDATE_ONLY','schema_qualified':False,'fields':fields,'firstline_bytes':len(firstline),'firstline_sha256':hashlib.sha256(firstline).hexdigest()}


def inspect_zip(raw,*,progress=None):
    progress={}if progress is None else progress
    need(raw.startswith(b'PK\x03\x04'),'ZIP_ONLY_NO_FALLBACK');members=[];headers=[];progress['members']=members
    with zipfile.ZipFile(io.BytesIO(raw))as z:
        infos=z.infolist();need(0<len(infos)<=MEMBERS_MAX and sum(i.file_size for i in infos)<=ZIP_MAX,'ZIP_INVENTORY_LIMIT')
        need(len({i.filename.casefold()for i in infos})==len(infos),'ZIP_DUPLICATE_MEMBER')
        for info in infos:
            safe_member(info.filename);need(not info.flag_bits&1,'ZIP_ENCRYPTED');need((info.external_attr>>16)&0o170000!=0o120000,'ZIP_SYMLINK')
            members.append({'name':info.filename,'uncompressed_bytes':info.file_size,'compressed_bytes':info.compress_size,'compression':info.compress_type,'CRC32':f'{info.CRC:08x}'})
            if not info.is_dir()and info.filename.lower().endswith('.csv'):
                with z.open(info)as stream:first=stream.readline(HEADER_MAX+1)
                progress['current_firstline']={'member':info.filename,'bytes':len(first),'sha256':hashlib.sha256(first).hexdigest(),'status':'UNKNOWN_UNTIL_LEXICALLY_CHECKED'}
                headers.append({'member':info.filename,**csv_header_candidate(first)});progress.pop('current_firstline')
    need(headers,'ZIP_NO_CSV_HEADER_CANDIDATE')
    return {'members':members,'declared_uncompressed_total_bytes':sum(x['uncompressed_bytes']for x in members),'CSV_header_candidates':headers,
        'full_member_CRC_verified':False,'disk_extractions':0,'observation_rows_parsed':0,'observation_values_converted':0,'first_vintage_qualified':False}


def build_plan(manifest_ref=MANIFEST,design_ref=DESIGN,form_inventory_ref=INVENTORY,*,source_base_commit=None):
    need((manifest_ref,design_ref,form_inventory_ref)==(MANIFEST,DESIGN,INVENTORY),'LOCKED_REFS')
    manifest=locked_manifest(read(manifest_ref));design=read(design_ref);inventory=read(form_inventory_ref)
    need(design['schema_version']=='gx1_treasury_initial_release_source_design_v1'and design['execution_authorized']is False
        and design['manifest']==MANIFEST and design['form_inventory']==INVENTORY and design['execution']==EXECUTION,'LOCKED_DESIGN')
    budget=design['budget'];need(budget['actual_invocations_max']==1 and budget['series_POST_attempts_max']==3 and budget['response_body_bytes_max']==3*BODY_MAX
        and budget['overflow_probe_bytes_total_max']==3 and budget['ZIP_uncompressed_bytes_per_response_max']==ZIP_MAX
        and budget['header_bytes_per_member_max']==HEADER_MAX and all(budget[k]==0 for k in ('observation_rows_parsed','observation_values_converted','feature_or_target_reconstructions',
            'native_source_arrays_or_CP_loads','model_constructions','forwards','fits','optimizer_updates','native_TEST_split_files')),'LOCKED_BUDGET')
    need(inventory['schema_version']=='gx1_alfred_public_download_metadata_inventory_v1'and inventory['dataset_downloads_or_POSTs']==0
        and inventory['financial_observation_values_extracted']==0 and len(inventory['records'])==3,'FORM_INVENTORY')
    inputs={'manifest':MANIFEST,'design':DESIGN,'form_inventory':INVENTORY,'source_key_acceptance':manifest['source_key_acceptance']}
    for item,record in zip(manifest['series'],inventory['records']):
        need(record['series']==item['id']and record['requested_url']==item['url']and record['HTML']==item['source_HTML']
            and record['form']['method']=='post'and record['form']['action']is None,'BOUND_FORM_IDENTITY')
        controls={c['name']:c for c in record['form']['controls']}
        for name,value,label in (('form[file_type]','4','Observations, Initial Release Only'),('form[file_format]','csv','Zipped CSV'),('form[units]','lin','Percent')):
            need({'value':value,'label':label}in controls[name]['options'],'BOUND_FORM_OPTION')
        need(all(name in controls for name,_ in FIELDS),'BOUND_FORM_FIELDS');inputs['HTML_'+item['id']]=item['source_HTML']
    accepted=read(inputs['source_key_acceptance']);need(accepted['status']=='EXACT_ALL61_CURRENT_TRAIN_SOURCE_KEYS_QUALIFIED_VINTAGE_UNPROVEN','CURRENT_KEY_ACCEPTANCE')
    for _,primitive in SERIES:need(accepted['source_domains'][primitive]['first']=='2020-11-30'and accepted['source_domains'][primitive]['last']=='2025-12-29','EXACT_OBSERVATION_RANGE')
    return {'schema_version':'gx1_treasury_initial_release_source_plan_v1','stage':STAGE,'status':'PREPARED_NOT_ADMITTED','plan_path':str(PLAN),'output_root':str(ROOT),
        'source_base_commit':source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'source_owners':{n:base.prior.raw.saved.native.sha(REPO/n)for n in OWNERS},'inputs':inputs,'requests':manifest['series'],
        'request_limits':manifest['request_limits'],'container_limits':manifest['container_limits'],'execution':EXECUTION,'budget':budget,
        'runtime':{'python':sys.version,'OpenSSL':ssl.OPENSSL_VERSION},'qualification_scope':'Physical quarantine/inventory only; no observation decode, nativeinput change or firstvintage PASS.'}


def checked_inputs(plan):
    for binding in plan['inputs'].values():need(ref(binding['path'])==binding,'INPUT_SHA')


def source_clean(plan):
    return (subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='work/gx1-current'
        and subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_base_commit']
        and not subprocess.check_output(['git','status','--porcelain','--','gx1','scripts','tests'],cwd=REPO,text=True)
        and set(plan['source_owners'])==set(OWNERS)and all(base.prior.raw.saved.native.sha(REPO/n)==v for n,v in plan['source_owners'].items()))


def gate_expected(binding,admitted):
    return {'status':'ADMITTED_EXACT_CPU_ONLY_TREASURY_INITIAL_RELEASE_SOURCE'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,
        'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}


def inspect(plan_sha,*,admitted):
    need(Path.cwd().resolve()==REPO and type(plan_sha)is str and len(plan_sha)==64 and all(c in '0123456789abcdef'for c in plan_sha),'CWD_SHA')
    binding={'path':str(PLAN),'sha256':plan_sha};plan=read(binding)
    need(plan==build_plan(source_base_commit=plan['source_base_commit'])and source_clean(plan),'EXACT_PLAN_SOURCE')
    pref=ref(POLICY);policy=read(pref);need(all(policy.get(k)is False for k in base.prior.raw.saved.FLAGS)and policy[STAGE]==gate_expected(binding,admitted),'ALL_TRAINING_CLOSED_GATE')
    need(not ROOT.exists()and not ROOT.is_symlink(),'FRESH_ROOT');return plan,binding,pref


def require_timeout_ancestry(plan_sha,scope):
    pid=os.getpid();runner=REPO/'scripts/gx1_capped_run.sh';seen=set()
    while pid and pid not in seen and len(seen)<64:
        seen.add(pid);p=Path('/proc')/str(pid);identity=guard._lock_process(pid,runner);args=os.fsdecode(guard._read_lock_proc(p/'cmdline')).rstrip('\0').split('\0')
        if Path(os.readlink(p/'exe'))==Path('/usr/bin/timeout').resolve():
            operand=Path(args[4])if len(args)>4 else Path('/invalid');operand=operand if operand.is_absolute()else Path(os.readlink(p/'cwd'))/operand
            need(args[1:4]==['--signal=TERM','--kill-after=10s','600s']and operand==REPO/'.venv/bin/python'
                and args[5:]==['-m','gx1.audit.audit_native_treasury_vintage_source_v1','--plan-sha256',plan_sha]
                and identity.cgroup==scope and guard._lock_process(pid,runner)==identity,'CANONICAL_OUTER_TIMEOUT')
            return {'pid':pid,'start_ticks':identity.start_ticks,'seconds':600,'scope':scope,'exact_cli':True}
        pid=identity.parent_pid
    need(False,'MISSING_CANONICAL_OUTER_TIMEOUT')


def publish(name,value):
    path=ROOT/name;raw=value if isinstance(value,bytes)else(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    base.prior.raw.saved._publish_bytes(path,raw);return ref(path)


def collect_one(item,progress,*,opener=None):
    """Record every attempted response before interpreting any container/header."""
    entry={'series':item['id'],'request_URL':item['url'],'method':'POST','post_body_sha256':item['post_body_sha256'],'status':'ATTEMPTING',
        'started_utc':datetime.now(timezone.utc).isoformat(),'HTTP_status':None,'content_type':None};progress.append(entry)
    entry['attempt']=publish(item['id']+'_ATTEMPT.json',dict(entry))
    try:
        raw,metadata=fetch_response(item,opener=opener);entry.update(metadata);entry['raw_response']=publish(item['id']+'_RESPONSE.bin',raw)
        entry['saved_bytes']=len(raw);entry['status']='RESPONSE_CAPTURED_UNINSPECTED';entry['response_receipt']=publish(item['id']+'_RESPONSE.json',dict(entry))
        need(not metadata['oversize']and metadata['body_complete'],'RESPONSE_BODY_INCOMPLETE')
        need(metadata['HTTP_status']==200 and metadata['final_url']==item['url'],'UNSUPPORTED_HTTP_OR_REDIRECT')
        entry['inspection_progress']={};inspection=inspect_zip(raw,progress=entry['inspection_progress']);entry['inspection']=publish(item['id']+'_ZIP_HEADER_CANDIDATES.json',inspection);entry['status']='PHYSICAL_ZIP_AND_HEADER_CANDIDATES_ONLY'
    except Exception as error:
        entry['status']='FAILED_CONSUMED';entry['error_type']=type(error).__name__;entry['error']=str(error)[:1000]
        entry['failure_receipt']=publish(item['id']+'_FAILED.json',dict(entry))
    return entry


def bookends(plan,binding,pref,boot):
    try:
        checked_inputs(plan)
        return {'source_inputs_unchanged':source_clean(plan),'plan_unchanged':ref(PLAN)==binding,'policy_unchanged':ref(POLICY)==pref,
            'boot_unchanged':Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot}
    except Exception as error:return {'bookend_error':type(error).__name__+':'+str(error)[:1000]}


def run(plan_sha):
    global _HTTP
    began=time.monotonic();deadline=began+EXECUTION['internal_seconds'];cap=guard.require_capped_cpu_audit_execution();lock=guard.require_capped_lock_ancestry()
    need(cap['memory_max']==cap['memory_high']==4*1024**3 and cap['swap']==512*1024**2 and cap['numerical_threads']==1
        and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):]and not any(n=='torch'or n.startswith('torch.')for n in sys.modules),'CAP_LIMITS_NO_TORCH')
    timeout=require_timeout_ancestry(plan_sha,lock['scope']);plan,binding,pref=inspect(plan_sha,admitted=True);checked_inputs(plan)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();ROOT.mkdir(parents=True,exist_ok=False)
    handlers={s:signal.getsignal(s)for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};claim=result=None;attempts=[]
    try:
        def expired(*_):raise TimeoutError('TREASURY_REQUEST_20S_WALL'if _HTTP is not None else'TREASURY_GLOBAL_WALL')
        def stopped(s,*_):raise InterruptedError('TREASURY_EXTERNAL_SIGNAL_'+str(s))
        signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
        claim=publish('CLAIM.json',{'stage':STAGE,'plan':binding,'policy_at_claim':pref,'boot_id':boot,'outer_timeout':timeout,'remaining_invocations':0,
            'relaunch_allowed':False,'test_data_used':False});need(deadline>time.monotonic(),'TIME_BEFORE_REQUESTS')
        for item in plan['requests']:
            need(_HTTP is None and len(attempts)<3,'ONE_REQUEST_PER_SERIES');_HTTP={'url':item['url'],'body':body_bytes(item),'seen':False}
            signal.setitimer(signal.ITIMER_REAL,min(20,max(.001,deadline-time.monotonic())))
            try:entry=collect_one(item,attempts)
            finally:
                if attempts:attempts[-1]['urllib_request_event_seen']=_HTTP['seen']
                _HTTP=None;signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
            need(entry['status']=='PHYSICAL_ZIP_AND_HEADER_CANDIDATES_ONLY'and entry['urllib_request_event_seen'],'RESPONSE_COLLECTION_FAILED_CONSUMED')
        ending=bookends(plan,binding,pref,boot);need(len(attempts)==3 and len(ending)==4 and all(ending.values()),'SOURCE_INPUT_PLAN_POLICY_BOOT_BOOKEND')
        result=publish('RESULT.json',{'status':'COMPLETE_PHYSICAL_TREASURY_RESPONSE_COLLECTION','stage':STAGE,'plan':binding,'claim':claim,'attempts':attempts,
            'bookends':ending,'source_unchanged':True,'elapsed_seconds':time.monotonic()-began,'HTTP_POST_attempts':sum(x.get('urllib_request_event_seen',False)for x in attempts),
            'series_attempts_begun':len(attempts),'stored_response_bytes':sum(x['saved_bytes']for x in attempts),'observation_rows_parsed':0,'observation_values_converted':0,'model_forwards':0,'fits':0,
            'zero_work':{'native_source_arrays_or_CP_loads':0,'models':0,'optimizer_updates':0,'feature_or_target_reconstructions':0,'normalization_or_prior_fits':0},'schema_status':'LEXICAL_HEADER_CANDIDATES_ONLY_UNKNOWN_SCHEMA','first_vintage_status':'UNKNOWN_NOT_QUALIFIED','native_inputs_changed':False,'test_data_used':False})
        terminal=publish('TERMINAL.json',{'status':'COMPLETE','exit_code':0,'stage':STAGE,'plan':binding,'claim':claim,'result':result,
            'remaining_invocations':0,'relaunch_allowed':False,'source_unchanged':True,'test_data_used':False});return {'result':result,'terminal':terminal}
    except BaseException as error:
        _HTTP=None;signal.setitimer(signal.ITIMER_REAL,0);ending=bookends(plan,binding,pref,boot)
        if result is None:result=publish('RESULT.json',{'status':'FAILED_CONSUMED_PHYSICAL_REFERENCE_COLLECTION','stage':STAGE,'plan':binding,'claim':claim,
            'attempts':attempts,'bookends':ending,'error_type':type(error).__name__,'error':str(error)[:1000],'observation_rows_parsed':0,'observation_values_converted':0,
            'schema_status':'UNKNOWN','first_vintage_status':'UNKNOWN_NOT_QUALIFIED','elapsed_seconds':time.monotonic()-began,'test_data_used':False})
        publish('TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'stage':STAGE,'plan':binding,'claim':claim,'result':result,'bookends':ending,
            'remaining_invocations':0,'relaunch_allowed':False,'HTTP_POST_attempts':sum(x.get('urllib_request_event_seen',False)for x in attempts),'error_type':type(error).__name__,'error':str(error)[:1000],'test_data_used':False});raise
    finally:
        _HTTP=None;signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items():signal.signal(s,h)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    need(sys.argv[1:]==['--plan-sha256',args.plan_sha256]+(['--preflight']if args.preflight else[]),'EXACT_CLI')
    if args.preflight:
        plan,binding,pref=inspect(args.plan_sha256,admitted=False);checked_inputs(plan)
        print(json.dumps({'status':'PREFLIGHT_TREASURY_PHYSICAL_SOURCE_PASS','plan':binding,'policy':pref,'claim_created':False,'HTTP_POST_attempts':0,'observation_values_converted':0}))
    else:print(json.dumps(run(args.plan_sha256)))

if __name__=='__main__':main()
