"""One-use Mac physical Treasury reference relay; no observation decoding."""
from __future__ import annotations
import argparse,ast,csv,hashlib,io,json,os,re,signal,ssl,sys,time,zipfile
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request,ProxyHandler,HTTPRedirectHandler,HTTPSHandler,build_opener
BASE=Path('/Users/andrekildalbakke/GX1_BACKUP');NAME='NATIVE_TREASURY_MAC_RELAY_20261123_002'
ROOT=BASE/NAME;PLAN=BASE/(NAME+'.plan.json');POLICY=BASE/(NAME+'.policy.json')
REMOTE=Path('/home/andre2/src/GX1_CURRENT');SELF='scripts/gx1_treasury_mac_relay_v1.py'
REMOTE_PLAN=REMOTE/'configs/research'/(NAME+'.json');STAGE='native_treasury_mac_relay_20261123_002'
OWNER=REMOTE/'gx1/audit/audit_native_treasury_vintage_source_v1.py';OWNER_SHA='7fe77d7f1e630ba291f8240181457e6b9a773d93053620639605a1be4adf7324'
MANIFEST=Path('/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001/TREASURY_INITIAL_RELEASE_MAC_RELAY_MANIFEST002.json')
MANIFEST_SHA='ef52c18697143c8db0c4c1d1aa67b0083828de0f17bb62eb8c0e2c02eb0cda5e'
FUNCTIONS=('need','body_bytes','audit_event','NoRedirect','make_opener','fetch_response','safe_member','csv_header_candidate','inspect_zip','collect_one')
FLAGS=('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed')
EXECUTION={'host':'Mac_Darwin','global_wall_seconds':90,'request_timeout_seconds':20,'requests_max':3}


def need(ok,why):
    if not bool(ok):raise ValueError('TREASURY_MAC_RELAY_'+why)


def ref(path):
    return {'path':str(path),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def checked_bytes(binding):
    raw=Path(binding['path']).read_bytes();need(hashlib.sha256(raw).hexdigest()==binding['sha256'],'INPUT_SHA');return raw


def build_plan(source_base_commit):
    need(type(source_base_commit)is str and re.fullmatch('[0-9a-f]{40}',source_base_commit)is not None,'SOURCE_COMMIT')
    shas={'adapter':ref(__file__)['sha256'],'owner':OWNER_SHA,'manifest':MANIFEST_SHA}
    return {'schema_version':'gx1_treasury_mac_relay_plan_v1','stage':STAGE,'status':'PREPARED_NOT_ADMITTED','execution_authorized':False,
        'plan_path':str(REMOTE_PLAN),'local_plan_path':str(PLAN),'output_root':str(ROOT),'policy_path':str(POLICY),'source_base_commit':source_base_commit,
        'source_copies':{k:{'path':str(BASE/(NAME+'.'+{'adapter':'adapter.py','owner':'owner.py','manifest':'manifest.json'}[k])),'sha256':v}for k,v in shas.items()},
        'canonical_sources':{k:{'path':str(v),'sha256':shas[k]}for k,v in {'adapter':REMOTE/SELF,'owner':OWNER,'manifest':MANIFEST}.items()},
        'execution':EXECUTION,'qualification_scope':'Physical quarantine only; no data-row decode, native input changes or first-vintage PASS.'}


def extract_owner(binding):
    need(binding['sha256']==OWNER_SHA,'PURE_OWNER_IDENTITY');tree=ast.parse(checked_bytes(binding));nodes=[]
    for name in FUNCTIONS:
        found=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name]
        need(len(found)==1,'EXACT_NAMED_OWNER_FUNCTION');nodes+=found
    namespace={name:globals()[name]for name in ('csv','hashlib','io','json','re','ssl','zipfile','datetime','timezone','PurePosixPath','HTTPError','urlencode',
        'Request','ProxyHandler','HTTPRedirectHandler','HTTPSHandler','build_opener')}
    namespace.update(_HTTP=None,BODY_MAX=1048576,HEADER_MAX=65536,ZIP_MAX=16777216,MEMBERS_MAX=16,publish=publish)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(OWNER),'exec'),namespace)
    return namespace


def validate_manifest(manifest):
    need(manifest['schema_version']=='gx1_treasury_initial_release_mac_relay_manifest_v1'and manifest['execution_authorized']is False
        and manifest['training_authorized']is False and manifest['test_authorized']is False and manifest['route']['retrieval_host']=='Mac_Darwin'
        and manifest['route']['global_wall_seconds']==90 and manifest['route']['output_root']==str(ROOT)
        and manifest['route']['pure_HTTP_ZIP_owner']=={'path':str(OWNER),'sha256':OWNER_SHA},'MANIFEST_SCOPE')
    limits=manifest['request_limits'];need(limits['series_requests_max']==3 and limits['request_timeout_seconds']==20 and limits['max_body_bytes']==1048576
        and limits['max_stored_oversize_prefix_bytes']==1048576 and limits['overflow_probe_bytes']==1 and limits['max_requests_per_series']==1
        and limits['redirects_allowed']==limits['retries_allowed']==0 and limits['proxies_allowed']is False
        and limits['credentials_cookies_APIkeys_supplied']is False and limits['TLS_certificate_and_hostname_verification_required']is True,'REQUEST_LIMITS')
    series=manifest['series'];need(len(series)==3 and [x['id']for x in series]==['DGS2','DGS10','DFII10'],'EXACT_SERIES')
    for item in series:
        need(item['method']=='POST'and item['url']=='https://alfred.stlouisfed.org/series/downloaddata?seid='+item['id']
            and item['post_fields']==[['form[units]','lin'],['form[obs_start_date]','2020-11-30'],['form[obs_end_date]','2025-12-29'],['form[file_type]','4'],['form[file_format]','csv'],['form[download_data]','']]
            and item['post_body_sha256']==hashlib.sha256(urlencode([tuple(p)for p in item['post_fields']]).encode('ascii')).hexdigest(),'EXACT_POST')
    return manifest


def inspect(plan_sha,*,admitted):
    need(sys.platform=='darwin'and Path(__file__).resolve()==BASE/(NAME+'.adapter.py'),'EXACT_MAC_ADAPTER')
    need(type(plan_sha)is str and re.fullmatch('[0-9a-f]{64}',plan_sha)is not None,'PLAN_SHA')
    binding={'path':str(REMOTE_PLAN),'sha256':plan_sha};plan=json.loads(checked_bytes({'path':str(PLAN),'sha256':plan_sha}))
    need(plan==build_plan(plan['source_base_commit']),'EXACT_PLAN');copies=plan['source_copies']
    for b in copies.values():checked_bytes(b)
    manifest=validate_manifest(json.loads(checked_bytes(copies['manifest'])));policy_ref=ref(POLICY);policy=json.loads(checked_bytes(policy_ref))
    gate={'status':'ADMITTED_EXACT_MAC_PHYSICAL_REFERENCE_ONLY'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,
        'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}
    need(all(policy.get(k)is False for k in FLAGS)and policy[STAGE]==gate,'POLICY_GATE_ALL_TRAINING_CLOSED')
    need(not ROOT.exists()and not ROOT.is_symlink(),'FRESH_ROOT');need(not any(n=='torch'or n.startswith('torch.')for n in sys.modules),'TORCH_FORBIDDEN')
    return plan,binding,policy_ref,manifest


def publish(name,value):
    need(type(name)is str and Path(name).name==name,'OUTPUT_NAME')
    if name.endswith('_ATTEMPT.json'):
        value={**value,'source_origin':'DIRECT_HTTPS','retrieval_host':'Mac_Darwin','TLS_context':'ssl.create_default_context; CERT_REQUIRED; check_hostname=True','relay_verification_status':'NOT_PERFORMED_HERE'}
    raw=value if isinstance(value,bytes)else(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    path=ROOT/name;fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    directory=os.open(ROOT,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(directory)
    finally:os.close(directory)
    return ref(path)


def bookends(plan,binding,policy_ref):
    try:
        return {'plan_unchanged':ref(PLAN)['sha256']==binding['sha256'],'policy_unchanged':ref(POLICY)==policy_ref,
            'all_source_copies_unchanged':all(ref(b['path'])==b for b in plan['source_copies'].values())}
    except Exception as error:return {'bookend_error':type(error).__name__+':'+str(error)[:500]}


def run(plan_sha):
    began=time.monotonic();deadline=began+90;handlers={s:signal.getsignal(s)for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    pure=None;plan=binding=policy_ref=None;claim=result=None;attempts=[];root_created=False
    def remaining():
        seconds=deadline-time.monotonic();need(seconds>0,'GLOBAL_DEADLINE_EXPIRED');return seconds
    def expired(*_):raise TimeoutError('TREASURY_MAC_REQUEST_20S_WALL'if pure is not None and pure['_HTTP']is not None else'TREASURY_MAC_GLOBAL_90S_WALL')
    def stopped(s,*_):raise InterruptedError('TREASURY_MAC_EXTERNAL_SIGNAL_'+str(s))
    try:
        signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
        signal.setitimer(signal.ITIMER_REAL,remaining())
        plan,binding,policy_ref,manifest=inspect(plan_sha,admitted=True);remaining()
        pure=extract_owner(plan['source_copies']['owner']);remaining();sys.addaudithook(pure['audit_event'])
        ROOT.mkdir(mode=0o700,exist_ok=False);root_created=True
        claim=publish('CLAIM.json',{'stage':STAGE,'plan':binding,'policy_at_claim':policy_ref,'pid':os.getpid(),'host':'Mac_Darwin',
            'source_origin':'DIRECT_HTTPS','WSL_cgroup_claim':False,'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
        remaining()
        for item in manifest['series']:
            need(pure['_HTTP']is None and len(attempts)<3,'ONE_REQUEST_PER_SERIES');seconds=remaining()
            pure['_HTTP']={'url':item['url'],'body':pure['body_bytes'](item),'seen':False}
            signal.setitimer(signal.ITIMER_REAL,min(20,seconds))
            try:entry=pure['collect_one'](item,attempts)
            finally:
                if attempts:
                    attempts[-1].update(urllib_request_event_seen=pure['_HTTP']['seen'],source_origin='DIRECT_HTTPS',retrieval_host='Mac_Darwin',
                        TLS_context='ssl.create_default_context; CERT_REQUIRED; check_hostname=True',relay_verification_status='NOT_PERFORMED_HERE')
                pure['_HTTP']=None;seconds=deadline-time.monotonic();signal.setitimer(signal.ITIMER_REAL,seconds if seconds>0 else 0)
            remaining();need(entry['status']=='PHYSICAL_ZIP_AND_HEADER_CANDIDATES_ONLY'and entry['urllib_request_event_seen'],'RESPONSE_FAILED_CONSUMED')
        ending=bookends(plan,binding,policy_ref);need(len(attempts)==3 and len(ending)==3 and all(ending.values()),'SOURCE_PLAN_POLICY_BOOKEND')
        result=publish('RESULT.json',{'status':'COMPLETE_MAC_PHYSICAL_TREASURY_REFERENCE','stage':STAGE,'plan':binding,'claim':claim,'attempts':attempts,'bookends':ending,
            'source_origin':'DIRECT_HTTPS','retrieval_host':'Mac_Darwin','WSL_cgroup_claim':False,'relay_verification_status':'NOT_PERFORMED_HERE',
            'HTTP_POST_attempts':sum(x.get('urllib_request_event_seen',False)for x in attempts),'elapsed_seconds':time.monotonic()-began,
            'schema_status':'LEXICAL_HEADER_CANDIDATES_ONLY_UNKNOWN_SCHEMA','first_vintage_status':'UNKNOWN_NOT_QUALIFIED',
            'observation_rows_parsed':0,'observation_values_converted':0,'native_inputs_changed':False,'models':0,'forwards':0,'fits':0,'test_data_used':False})
        terminal=publish('TERMINAL.json',{'status':'COMPLETE','exit_code':0,'stage':STAGE,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
        return {'result':result,'terminal':terminal}
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0)
        if pure is not None:pure['_HTTP']=None
        if not root_created:raise
        ending=bookends(plan,binding,policy_ref)
        if result is None:result=publish('RESULT.json',{'status':'FAILED_CONSUMED_MAC_PHYSICAL_REFERENCE','stage':STAGE,'plan':binding,'claim':claim,'attempts':attempts,'bookends':ending,
            'source_origin':'DIRECT_HTTPS','retrieval_host':'Mac_Darwin','HTTP_POST_attempts':sum(x.get('urllib_request_event_seen',False)for x in attempts),
            'error_type':type(error).__name__,'error':str(error)[:1000],'schema_status':'UNKNOWN','first_vintage_status':'UNKNOWN_NOT_QUALIFIED',
            'observation_rows_parsed':0,'observation_values_converted':0,'native_inputs_changed':False,'models':0,'forwards':0,'fits':0,'test_data_used':False})
        publish('TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'stage':STAGE,'plan':binding,'claim':claim,'result':result,'bookends':ending,
            'remaining_invocations':0,'relaunch_allowed':False,'error_type':type(error).__name__,'error':str(error)[:1000],'test_data_used':False});raise
    finally:
        if pure is not None:pure['_HTTP']=None
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items():signal.signal(s,h)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    need(sys.argv[1:]==['--plan-sha256',args.plan_sha256]+(['--preflight']if args.preflight else[]),'EXACT_CLI')
    if args.preflight:
        plan,binding,policy_ref,_=inspect(args.plan_sha256,admitted=True);extract_owner(plan['source_copies']['owner'])
        print(json.dumps({'status':'PREFLIGHT_MAC_PHYSICAL_REFERENCE_PASS','plan':binding,'policy':policy_ref,'claim_created':False,'HTTP_POST_attempts':0,'observation_values_converted':0}))
    else:print(json.dumps(run(args.plan_sha256)))

if __name__=='__main__':main()
