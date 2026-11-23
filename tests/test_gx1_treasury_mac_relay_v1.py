"""New Mac one-use authority only; generated HTTP/ZIP fixtures, no network."""
import hashlib,importlib.util,io,json,sys,zipfile
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
import pytest
REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('treasury_mac_adapter',REPO/'scripts/gx1_treasury_mac_relay_v1.py')
mac=importlib.util.module_from_spec(spec);spec.loader.exec_module(mac)
OWNER=REPO/'gx1/audit/audit_native_treasury_vintage_source_v1.py'


def raw_json(value):return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()


def generated_manifest():
    fields=[['form[units]','lin'],['form[obs_start_date]','2020-11-30'],['form[obs_end_date]','2025-12-29'],['form[file_type]','4'],['form[file_format]','csv'],['form[download_data]','']]
    return {'schema_version':'gx1_treasury_initial_release_mac_relay_manifest_v1','execution_authorized':False,'training_authorized':False,'test_authorized':False,
        'route':{'retrieval_host':'Mac_Darwin','global_wall_seconds':90,'output_root':str(mac.ROOT),'pure_HTTP_ZIP_owner':{'path':str(mac.OWNER),'sha256':mac.OWNER_SHA}},
        'request_limits':{'series_requests_max':3,'request_timeout_seconds':20,'max_body_bytes':1048576,'max_stored_oversize_prefix_bytes':1048576,
            'overflow_probe_bytes':1,'max_requests_per_series':1,'redirects_allowed':0,'retries_allowed':0,'proxies_allowed':False,'credentials_cookies_APIkeys_supplied':False,'TLS_certificate_and_hostname_verification_required':True},
        'series':[{'id':name,'method':'POST','url':'https://alfred.stlouisfed.org/series/downloaddata?seid='+name,'post_fields':fields,
            'post_body_sha256':hashlib.sha256(mac.urlencode([tuple(x)for x in fields]).encode()).hexdigest()}for name in ('DGS2','DGS10','DFII10')]}


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    original=Path(mac.__file__).read_bytes();monkeypatch.setattr(mac,'BASE',tmp_path);monkeypatch.setattr(mac,'ROOT',tmp_path/mac.NAME)
    monkeypatch.setattr(mac,'PLAN',tmp_path/(mac.NAME+'.plan.json'));monkeypatch.setattr(mac,'POLICY',tmp_path/(mac.NAME+'.policy.json'))
    adapter=tmp_path/(mac.NAME+'.adapter.py');adapter.write_bytes(original);monkeypatch.setattr(mac,'__file__',str(adapter));monkeypatch.setattr(mac,'sys',SimpleNamespace(platform='darwin',modules={},argv=sys.argv,addaudithook=sys.addaudithook))
    (tmp_path/(mac.NAME+'.owner.py')).write_bytes(OWNER.read_bytes())
    manifest_raw=raw_json(generated_manifest());(tmp_path/(mac.NAME+'.manifest.json')).write_bytes(manifest_raw)
    monkeypatch.setattr(mac,'MANIFEST_SHA',hashlib.sha256(manifest_raw).hexdigest())
    plan=mac.build_plan('a'*40);raw=raw_json(plan);mac.PLAN.write_bytes(raw);sha=hashlib.sha256(raw).hexdigest()
    binding={'path':str(mac.REMOTE_PLAN),'sha256':sha};policy={k:False for k in mac.FLAGS}
    policy[mac.STAGE]={'status':'ADMITTED_EXACT_MAC_PHYSICAL_REFERENCE_ONLY','plan':binding,'execution_authorized':True,'remaining_invocations':1,'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}
    mac.POLICY.write_bytes(raw_json(policy));return plan,sha,policy


def fake_transport(monkeypatch,plan,status=200):
    pure=mac.extract_owner(plan['source_copies']['owner']);calls=[];timers=[]
    class Response(io.BytesIO):
        def __init__(self,url):
            out=io.BytesIO()
            with zipfile.ZipFile(out,'w')as z:z.writestr('data.csv',b'DATE,VALUE\nGENERATED_UNREAD_DATA\n')
            super().__init__(out.getvalue());self.url=url;self.headers=Message();self.headers['Content-Type']='application/zip'
        def geturl(self):return self.url
        def getcode(self):return status
    class Opener:
        def open(self,request,timeout):
            calls.append((request.full_url,timeout));pure['audit_event']('urllib.Request',(request.full_url,request.data,request.headers,request.get_method()))
            return Response(request.full_url)
    pure['make_opener']=lambda:Opener();monkeypatch.setattr(mac,'extract_owner',lambda b:pure)
    monkeypatch.setattr(mac.sys,'addaudithook',lambda h:None);monkeypatch.setattr(mac.signal,'setitimer',lambda *a:timers.append(a))
    return calls,timers


def test_mac_exact_owner_extract_has_no_project_imports_or_free_source(monkeypatch):
    pure=mac.extract_owner({'path':str(OWNER),'sha256':mac.OWNER_SHA})
    assert all(name in pure for name in mac.FUNCTIONS)and 'base'not in pure and pure['_HTTP']is None
    with pytest.raises(ValueError,match='PURE_OWNER_IDENTITY'):mac.extract_owner({'path':str(OWNER),'sha256':'f'*64})


def test_mac_admission_requires_Darwin_closed_flags_and_fresh_root(prepared,monkeypatch):
    _,sha,policy=prepared;assert mac.inspect(sha,admitted=True)[1]['sha256']==sha
    monkeypatch.setattr(mac.sys,'platform','linux')
    with pytest.raises(ValueError,match='EXACT_MAC'):mac.inspect(sha,admitted=True)
    monkeypatch.setattr(mac.sys,'platform','darwin');mac.sys.modules={'torch':object()}
    with pytest.raises(ValueError,match='TORCH_FORBIDDEN'):mac.inspect(sha,admitted=True)
    mac.sys.modules={};policy[mac.FLAGS[0]]=True;mac.POLICY.write_bytes(raw_json(policy))
    with pytest.raises(ValueError,match='POLICY_GATE'):mac.inspect(sha,admitted=True)
    policy[mac.FLAGS[0]]=False;mac.POLICY.write_bytes(raw_json(policy));mac.ROOT.mkdir()
    with pytest.raises(ValueError,match='FRESH_ROOT'):mac.inspect(sha,admitted=True)


def test_mac_exact_plan_source_copy_and_oneuse_counter(prepared):
    plan,sha,policy=prepared;policy[mac.STAGE]['remaining_invocations']=0;mac.POLICY.write_bytes(raw_json(policy))
    with pytest.raises(ValueError,match='POLICY_GATE'):mac.inspect(sha,admitted=True)
    policy[mac.STAGE]['remaining_invocations']=1;mac.POLICY.write_bytes(raw_json(policy));Path(plan['source_copies']['owner']['path']).write_bytes(b'forbidden changed owner')
    with pytest.raises(ValueError,match='INPUT_SHA'):mac.inspect(sha,admitted=True)


def test_mac_preflight_checks_admitted_policy_no_HTTP_claim_or_root(prepared,monkeypatch,capsys):
    _,sha,_=prepared;monkeypatch.setattr(sys,'argv',['adapter','--plan-sha256',sha,'--preflight']);mac.sys.argv=sys.argv
    def forbidden(*a,**k):raise AssertionError('preflight side effect')
    monkeypatch.setattr(mac,'run',forbidden);monkeypatch.setattr(mac,'publish',forbidden);monkeypatch.setattr(mac,'build_opener',forbidden)
    mac.main();report=json.loads(capsys.readouterr().out)
    assert report['claim_created']is False and report['HTTP_POST_attempts']==0 and not mac.ROOT.exists()


def test_mac_first_failure_preserves_attempt_response_terminal_no_second_request(prepared,monkeypatch):
    plan,sha,_=prepared;calls,timers=fake_transport(monkeypatch,plan,status=503)
    with pytest.raises(ValueError,match='FAILED_CONSUMED'):mac.run(sha)
    assert len(calls)==1 and calls[0][1]==20 and not (mac.ROOT/'DGS10_ATTEMPT.json').exists()
    result=json.loads((mac.ROOT/'RESULT.json').read_text());terminal=json.loads((mac.ROOT/'TERMINAL.json').read_text())
    assert result['HTTP_POST_attempts']==1 and result['attempts'][0]['HTTP_status']==503
    assert terminal['status']=='FAILED_CONSUMED'and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False
    attempt=json.loads((mac.ROOT/'DGS2_ATTEMPT.json').read_text());assert attempt['source_origin']=='DIRECT_HTTPS'and attempt['retrieval_host']=='Mac_Darwin'
    assert (mac.ROOT/'DGS2_RESPONSE.bin').exists()and any(0<t[1]<=20 for t in timers)


def test_mac_success_three_requests_bookended_still_no_schema_or_vintage_PASS(prepared,monkeypatch):
    plan,sha,_=prepared;calls,timers=fake_transport(monkeypatch,plan);mac.run(sha)
    assert len(calls)==3 and len({x[0]for x in calls})==3
    result=json.loads((mac.ROOT/'RESULT.json').read_text())
    assert all(result['bookends'].values())and result['first_vintage_status']=='UNKNOWN_NOT_QUALIFIED'and result['WSL_cgroup_claim']is False
    assert result['HTTP_POST_attempts']==3 and result['observation_values_converted']==result['models']==result['forwards']==result['fits']==0
    assert any(20<t[1]<=90 for t in timers)and timers[-1][1]==0


def test_mac_publish_serializes_before_exclusive_open_and_never_overwrites(tmp_path,monkeypatch):
    monkeypatch.setattr(mac,'ROOT',tmp_path)
    with pytest.raises(ValueError):mac.publish('invalid.json',{'value':float('nan')})
    assert not (tmp_path/'invalid.json').exists()
    mac.publish('immutable.bin',b'first')
    with pytest.raises(FileExistsError):mac.publish('immutable.bin',b'second')
    assert (tmp_path/'immutable.bin').read_bytes()==b'first'


def test_mac_global_deadline_before_requests_is_consumed_without_HTTP(prepared,monkeypatch):
    plan,sha,_=prepared;calls,_=fake_transport(monkeypatch,plan);ticks=iter((0.,100.))
    monkeypatch.setattr(mac,'time',SimpleNamespace(monotonic=lambda:next(ticks)))
    with pytest.raises(ValueError,match='TIME_BEFORE_REQUESTS'):mac.run(sha)
    assert calls==[] and json.loads((mac.ROOT/'TERMINAL.json').read_text())['remaining_invocations']==0
