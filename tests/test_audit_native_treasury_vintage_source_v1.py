"""Generated HTTP/ZIP fixtures only: no network, source arrays or model payloads."""
import copy,hashlib,io,json,struct,sys,zipfile
from email.message import Message
from urllib.error import HTTPError
import pytest
from gx1.audit import audit_native_treasury_vintage_source_v1 as audit


def request_fixture():
    return {'id':'DGS2','primitive':'nominal2y','method':'POST','url':'https://alfred.stlouisfed.org/series/downloaddata?seid=DGS2',
        'post_fields':copy.deepcopy(audit.FIELDS),'post_body_sha256':hashlib.sha256(audit.body_bytes({'post_fields':audit.FIELDS})).hexdigest()}


def manifest_fixture():
    requests=[]
    for name,primitive in audit.SERIES:
        r=request_fixture();r.update(id=name,primitive=primitive,url='https://alfred.stlouisfed.org/series/downloaddata?seid='+name,
            source_HTML={'path':'/synthetic/'+name+'.html','sha256':'f'*64});requests.append(r)
    return {'schema_version':'gx1_treasury_initial_release_reference_manifest_v1','execution_authorized':False,'training_authorized':False,
        'form_inventory':audit.INVENTORY,'series':requests,'source_key_acceptance':{'path':'/synthetic/keys.json','sha256':'e'*64},
        'request_limits':{'series_requests_max':3,'request_timeout_seconds':20,'max_body_bytes':1048576,'overflow_probe_bytes':1,'max_stored_oversize_prefix_bytes':1048576,
            'max_requests_per_series':1,'redirects_allowed':0,'retries_allowed':0,'credentials_cookies_APIkeys_supplied':False,'proxies_allowed':False,'TLS_certificate_and_hostname_verification_required':True},
        'container_limits':{'format':'ZIP_ONLY','max_header_bytes':65536,'max_members':16,'max_uncompressed_total_bytes':16777216,
            'no_disk_extraction':True,'no_encryption':True,'no_path_traversal_or_absolute_members':True}}


def zip_fixture(name='data.csv',header=b'observation_date,DGS2,realtime_start\n'):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED)as z:
        z.writestr(name,header+b'GENERATED_DATA_MUST_NOT_BE_DECODED\n');z.writestr('README.txt',b'UNREAD_GENERATED_METADATA')
    return stream.getvalue()


class Response(io.BytesIO):
    def __init__(self,data,url,status=200,content_type='application/zip'):
        super().__init__(data);self.url=url;self.status=status;self.headers=Message();self.headers['Content-Type']=content_type
    def getcode(self):return self.status
    def geturl(self):return self.url


class Opener:
    def __init__(self,response=None,error=None):self.response=response;self.error=error;self.calls=0
    def open(self,request,timeout):
        self.calls+=1;assert timeout==20
        audit.audit_event('urllib.Request',(request.full_url,request.data,request.headers,request.get_method()))
        assert not any(k.lower()in ('authorization','cookie','proxy-authorization')for k in request.headers)
        if self.error:raise self.error
        return self.response


def authority(monkeypatch,item):
    monkeypatch.setattr(audit,'_HTTP',{'url':item['url'],'body':audit.body_bytes(item),'seen':False})


def test_treasury_exact_manifest_and_list_pair_urlencoding():
    m=manifest_fixture();assert audit.locked_manifest(m)==m
    assert audit.body_bytes(m['series'][0]).endswith(b'form%5Bdownload_data%5D=')
    assert m['series'][0]['post_body_sha256']=='94fc94452b8e1d87b72b7e497d36ce162b0db0c7cf4afff9492c8ca78cc2e1c9'


@pytest.mark.parametrize('change',['URL','body','series','redirect'])
def test_treasury_manifest_has_no_free_request_route(change):
    m=manifest_fixture()
    if change=='URL':m['series'][0]['url']='https://example.invalid/'
    elif change=='body':m['series'][0]['post_fields'][2][1]='2026-01-01'
    elif change=='series':m['series'][1]=copy.deepcopy(m['series'][0])
    else:m['request_limits']['redirects_allowed']=1
    with pytest.raises(ValueError):audit.locked_manifest(m)


def test_treasury_network_requires_authority_and_exact_single_request(monkeypatch):
    item=request_fixture();monkeypatch.setattr(audit,'_HTTP',None)
    with pytest.raises(ValueError,match='NOT_ADMITTED'):audit.fetch_response(item,opener=Opener())
    authority(monkeypatch,item);wrong=(item['url'],audit.body_bytes(item),{},'GET')
    with pytest.raises(ValueError):audit.audit_event('urllib.Request',wrong)
    good=(item['url'],audit.body_bytes(item),{},'POST');audit.audit_event('urllib.Request',good)
    with pytest.raises(ValueError):audit.audit_event('urllib.Request',good)
    with pytest.raises(RuntimeError,match='TORCH_FORBIDDEN'):audit.audit_event('import',('torch.nn',))


def test_treasury_HTTP_error_body_preserved_no_redirect_retry_or_cookie(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path);item=request_fixture();authority(monkeypatch,item);headers=Message();headers['Content-Type']='text/html'
    opener=Opener(error=HTTPError(item['url'],302,'Found',headers,io.BytesIO(b'GENERATED_REDIRECT_BODY')));attempts=[]
    entry=audit.collect_one(item,attempts,opener=opener)
    assert opener.calls==1 and entry['status']=='FAILED_CONSUMED'and entry['HTTP_status']==302
    assert (tmp_path/'DGS2_RESPONSE.bin').read_bytes()==b'GENERATED_REDIRECT_BODY'
    assert json.loads((tmp_path/'DGS2_RESPONSE.json').read_text())['body_complete']is True
    assert audit.NoRedirect().redirect_request(None,None,302,'Found',headers,'https://example.invalid')is None


def test_treasury_oversize_prefix_has_no_false_full_response_hash(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path);monkeypatch.setattr(audit,'BODY_MAX',16);item=request_fixture();authority(monkeypatch,item)
    opener=Opener(Response(b'x'*30,item['url']));entry=audit.collect_one(item,[],opener=opener)
    assert opener.calls==1 and entry['status']=='FAILED_CONSUMED'and entry['oversize']
    assert entry['saved_bytes']==16 and entry['body_bytes_observed']==17 and entry['body_complete']is False
    assert entry['full_response_sha256']=='UNKNOWN'and len((tmp_path/'DGS2_RESPONSE.bin').read_bytes())==16


def test_treasury_partial_transport_failure_keeps_captured_status_and_prefix(tmp_path,monkeypatch):
    class Partial(Response):
        reads=0
        def read(self,n=-1):
            self.reads+=1
            if self.reads==1:return b'GENERATED_PREFIX'
            raise TimeoutError('fixture timeout')
    monkeypatch.setattr(audit,'ROOT',tmp_path);item=request_fixture();authority(monkeypatch,item)
    entry=audit.collect_one(item,[],opener=Opener(Partial(b'',item['url'])))
    assert entry['HTTP_status']==200 and entry['read_error']=='TimeoutError'and entry['body_complete']is False
    assert entry['full_response_sha256']=='UNKNOWN'and (tmp_path/'DGS2_RESPONSE.bin').read_bytes()==b'GENERATED_PREFIX'


def test_treasury_quarantined_zip_decodes_only_lexical_header_candidate(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path);item=request_fixture();authority(monkeypatch,item)
    result=audit.collect_one(item,[],opener=Opener(Response(zip_fixture(),item['url'])))
    assert result['status']=='PHYSICAL_ZIP_AND_HEADER_CANDIDATES_ONLY'
    details=json.loads((tmp_path/'DGS2_ZIP_HEADER_CANDIDATES.json').read_text())
    assert details['CSV_header_candidates'][0]['fields']==['observation_date','DGS2','realtime_start']
    assert details['CSV_header_candidates'][0]['schema_qualified']is False
    assert details['observation_rows_parsed']==details['observation_values_converted']==0
    assert details['first_vintage_qualified']is False and details['disk_extractions']==0
    assert 'GENERATED_DATA'not in json.dumps(details)


@pytest.mark.parametrize('line',[b'2020-11-30,1.23,2020-12-01\n',b'<html>Forbidden</html>\n',b'\xff,VALUE\n'])
def test_treasury_nonlexical_firstline_is_never_decoded_or_echoed(line,tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path);item=request_fixture();authority(monkeypatch,item)
    def forbidden(*a,**k):raise AssertionError('CSV reader must not see possible data')
    monkeypatch.setattr(audit.csv,'reader',forbidden)
    result=audit.collect_one(item,[],opener=Opener(Response(zip_fixture(header=line),item['url'])))
    assert result['status']=='FAILED_CONSUMED'and result['error_type']=='ValueError'
    progress=result['inspection_progress']['current_firstline']
    assert progress['sha256']==hashlib.sha256(line).hexdigest()and progress['bytes']==len(line)
    assert set(progress)=={'member','bytes','sha256','status'}
    assert 'fields'not in progress and '1.23'not in json.dumps(result)
    assert (tmp_path/'DGS2_RESPONSE.bin').exists()


@pytest.mark.parametrize('name',['../data.csv','/data.csv','a\\data.csv'])
def test_treasury_zip_unsafe_paths_fail_closed(name):
    with pytest.raises(ValueError,match='ZIP_MEMBER_PATH'):audit.inspect_zip(zip_fixture(name=name))


def test_treasury_zip_limits_encryption_duplicates_symlinks_and_header_bound():
    def archive(names):
        s=io.BytesIO()
        with zipfile.ZipFile(s,'w')as z:
            for name in names:z.writestr(name,b'DATE,VALUE\n')
        return s.getvalue()
    with pytest.raises(ValueError,match='DUPLICATE'):audit.inspect_zip(archive(['A.csv','a.csv']))
    with pytest.raises(ValueError,match='INVENTORY_LIMIT'):audit.inspect_zip(archive([str(i)+'.csv'for i in range(17)]))
    with pytest.raises(ValueError,match='CSV_FIRSTLINE_LIMIT'):audit.inspect_zip(zip_fixture(header=b'A'*65537+b',B\n'))
    raw=bytearray(archive(['data.csv']));central=raw.index(b'PK\x01\x02')
    encrypted=bytearray(raw);struct.pack_into('<H',encrypted,central+8,1)
    with pytest.raises(ValueError,match='ENCRYPTED'):audit.inspect_zip(bytes(encrypted))
    huge=bytearray(raw);struct.pack_into('<I',huge,central+24,16777217)
    with pytest.raises(ValueError,match='INVENTORY_LIMIT'):audit.inspect_zip(bytes(huge))
    symlink=bytearray(raw);struct.pack_into('<I',symlink,central+38,0o120777<<16)
    with pytest.raises(ValueError,match='SYMLINK'):audit.inspect_zip(bytes(symlink))


def test_treasury_no_uncompressed_or_HTML_fallback():
    for raw in (b'DATE,VALUE\n2020-11-30,1\n',b'<html>blocked</html>'):
        with pytest.raises(ValueError,match='ZIP_ONLY_NO_FALLBACK'):audit.inspect_zip(raw)


def test_treasury_TLS_verified_no_proxy_cookie_or_auth(monkeypatch):
    def inspect_handlers(*handlers):return handlers
    monkeypatch.setattr(audit,'build_opener',inspect_handlers)
    handlers=audit.make_opener()
    assert handlers[0].proxies=={} and isinstance(handlers[1],audit.NoRedirect)
    context=handlers[2]._context
    assert context.check_hostname and context.verify_mode==audit.ssl.CERT_REQUIRED
    item=request_fixture();authority(monkeypatch,item)
    with pytest.raises(ValueError,match='CREDENTIALS'):
        audit.audit_event('urllib.Request',(item['url'],audit.body_bytes(item),{'Cookie':'fixture'},'POST'))
    assert audit._HTTP['seen']is False


def test_treasury_preflight_no_network_claim_or_output(monkeypatch,capsys):
    plan={'inputs':{}};binding={'path':'/synthetic/plan','sha256':'a'*64};seen=[]
    monkeypatch.setattr(sys,'argv',['audit','--plan-sha256','a'*64,'--preflight'])
    monkeypatch.setattr(audit,'inspect',lambda sha,admitted:(seen.append((sha,admitted))or(plan,binding,{})))
    monkeypatch.setattr(audit,'checked_inputs',lambda p:seen.append(p))
    def forbidden(*a,**k):raise AssertionError('preflight side effect')
    monkeypatch.setattr(audit,'publish',forbidden);monkeypatch.setattr(audit,'fetch_response',forbidden);monkeypatch.setattr(audit,'run',forbidden)
    audit.main();out=json.loads(capsys.readouterr().out)
    assert seen==[('a'*64,False),plan]and out['claim_created']is False and out['HTTP_POST_attempts']==0


def test_treasury_oneuse_and_training_flags_fail_before_network(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'REPO',tmp_path);monkeypatch.chdir(tmp_path);monkeypatch.setattr(audit,'ROOT',tmp_path/'output')
    binding={'path':str(audit.PLAN),'sha256':'a'*64};plan={'source_base_commit':'fixture'}
    policy={k:False for k in audit.base.prior.raw.saved.FLAGS};policy[audit.STAGE]=audit.gate_expected(binding,True)
    monkeypatch.setattr(audit,'build_plan',lambda **k:plan);monkeypatch.setattr(audit,'source_clean',lambda p:True)
    monkeypatch.setattr(audit,'ref',lambda p:{'path':str(p),'sha256':'b'*64})
    monkeypatch.setattr(audit,'read',lambda r:plan if r==binding else policy)
    (tmp_path/'output').mkdir()
    with pytest.raises(ValueError,match='FRESH_ROOT'):audit.inspect('a'*64,admitted=True)
    (tmp_path/'output').rmdir();policy[audit.base.prior.raw.saved.FLAGS[0]]=True
    with pytest.raises(ValueError,match='TRAINING_CLOSED'):audit.inspect('a'*64,admitted=True)
    policy[audit.base.prior.raw.saved.FLAGS[0]]=False;policy[audit.STAGE]['remaining_invocations']=0
    with pytest.raises(ValueError,match='TRAINING_CLOSED'):audit.inspect('a'*64,admitted=True)
