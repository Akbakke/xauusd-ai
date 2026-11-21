"""Small NumPy-only cache/lifecycle fixtures, never a production materialization."""
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from gx1.audit import build_native_m5_ta_feature_cache_v1 as cache
from gx1.contracts import market_state_native_m5_feature_inputs_v1 as features
from gx1.contracts.immutable_event_authority_v1 import ImmutableEventAuthorityError


def _provider(*,missing_macro=False):
    profile=features.owners.compile_full_ta_entry_execution_profile();names=profile['physical_field_names']
    n=400;starts=pd.date_range('2024-05-01',periods=n,freq='5min',tz='UTC').asi8
    rows=np.arange(n,dtype=np.int64);issues=rows[95:];signal=np.ones((n,len(names['signal'])),np.float32)
    if missing_macro:signal[150,names['signal'].index('ctx_cont.'+profile['macro_context_fields'][0])]=np.nan
    context=np.ones((n,len(names['ctx_cont'])),np.float32)
    context[:,profile['temporal_alias_ctx_cont_indices']]=signal[:,profile['temporal_alias_signal_indices']]
    source=features._seal({'rows':n,'native_m5_manifest':{'path':'/unit/MANIFEST.json','sha256':'1'*64}})
    squeeze={'native_five_only':True};frames={};retained={}
    for tf in features.TIMEFRAMES:
        width=int(features.htf.MULTI_TF_SHIFT[tf].value)
        times=starts if tf=='M5'else (starts[-1]+features.BAR_NS)//width*width-width-np.arange(n-1,-1,-1,dtype=np.int64)*width
        matrix=features._owned(np.ones((n,len(names['mtf'])),np.float32),np.float32)
        owner_rows=features._owned(rows+(0 if tf=='M5'else 10000),np.int64)
        fields=features.htf.MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4[tf]
        scalar=features._owned(np.ones((n,len(fields)),np.float32),np.float32)
        frame=pd.DataFrame(matrix,index=pd.to_datetime(times,utc=True),columns=names['mtf'],copy=False)
        frame.attrs={'ts_int64':features._owned(times,np.int64),'feats_np':matrix,'owner_row_indices':owner_rows,
            'causal_warmup_rows':0,'htf_feature_contract':features.htf.HTF_V4_MATRIX_CONTRACT,
            'model_native_mtf_scalar_contract_v4':features.htf.MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4,
            'model_native_mtf_scalar_fields_v4':list(fields),'model_native_mtf_scalar_warmup_rows_v4':0,
            'model_native_mtf_scalars_np_v4':scalar}
        frames[tf]=frame
        retained[tf]={'rows':n,'owner_row_indices_sha256':features._array_sha(owner_rows,'<i8'),
            'bar_start_ns_sha256':features._array_sha(times,'<i8'),'scalar_fields':list(fields),
            'scalar_values_sha256':features._array_sha(scalar,'<f4')}
    macro={'binding_sha256':'2'*64};registry='3'*64;available=1357077600000000000
    axes=features._seal({'source_binding_sha256':source['contract_sha256'],'registry_contract_sha256':registry,
        'squeeze_binding_sha256':features._sha(squeeze),'macro_binding_sha256':macro['binding_sha256'],
        'parameter_available_ns':available,'issue_start_ns':int(starts[95]+features.BAR_NS),
        'issue_end_ns':int(starts[-1]+2*features.BAR_NS),'retained_source_rows_sha256':features._array_sha(rows,'<i8'),
        'retained_issue_rows_sha256':features._array_sha(issues,'<i8'),'retained_local_rows':n,'retained_timeframes':retained,
        'rows':n,'source_row_sha256':features._array_sha(rows,'<i8'),'bar_start_ns_sha256':features._array_sha(starts,'<i8'),
        'issue_ns_sha256':features._array_sha(starts+features.BAR_NS,'<i8')})
    surfaces={'schema_version':features.SURFACES_SCHEMA,'provider_contract':features.native_m5_feature_provider_contract(),
        'source_binding':source,'axes':axes,'bar_start_ns':features._owned(starts,np.int64),
        'retained_source_rows':features._owned(rows,np.int64),'retained_issue_rows':features._owned(issues,np.int64),
        'signal':features._owned(signal,np.float32),'ctx_cont':features._owned(context,np.float32),
        'ctx_cat':features._owned(np.zeros((n,len(names['ctx_cat'])),np.int64),np.int64),
        'observed':features._owned(np.isfinite(signal),np.bool_),'multi_tf_sources':frames,
        'field_lineage':{name:'unit_declared_owner'for name in names['signal']},
        'registry_contract_sha256':registry,'squeeze_binding':squeeze,'macro_binding':macro,'parameter_available_ns':available}
    return features.NativeM5FeatureProvider(surfaces)


def _publisher(root):
    root.mkdir(exist_ok=True)
    def publish(name,value):
        raw=value if isinstance(value,bytes)else(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
        path=root/name;path.write_bytes(raw)
        return {'path':str(path),'sha256':cache.sha(path)}
    return publish


def test_writer_default_and_callback_keep_same_27_array_bytes_and_reload(tmp_path,monkeypatch):
    provider=_provider();old_publish=_publisher(tmp_path/'default');new_publish=_publisher(tmp_path/'new');calls=[]
    def original(name,value):calls.append(name);return old_publish(name,value)
    monkeypatch.setattr(cache.owner,'publish',original)
    old=cache.owner.publish_feature_snapshot(provider)
    assert calls==['FEATURE_INPUTS.npz','FEATURE_INPUTS.json']
    new=cache.owner.publish_feature_snapshot(provider,publish_artifact=new_publish)
    assert calls==['FEATURE_INPUTS.npz','FEATURE_INPUTS.json']
    a,b=cache.read(old),cache.read(new)
    assert len(a['arrays'])==len(b['arrays'])==27 and a['arrays']==b['arrays']
    loaded=cache.load_snapshot(new)
    np.testing.assert_array_equal(loaded.input_ready,provider.input_ready)
    report=cache.overlap_parity(loaded,old,a['archive'])
    assert report['array_count']==27 and report['exact_dtype_shape_bytes_and_global_owner_clocks']
    for tf in features.TIMEFRAMES:
        np.testing.assert_array_equal(loaded.tf_source_rows[tf],provider.tf_source_rows[tf])
    assert 'torch'not in cache.sys.modules


def test_snapshot_rejects_bad_byte_identity_and_overlap_rejects_clock_domain_shift(tmp_path):
    provider=_provider();publish=_publisher(tmp_path)
    snapshot=cache.owner.publish_feature_snapshot(provider,publish_artifact=publish);saved=cache.read(snapshot)
    broken=copy.deepcopy(saved);broken['arrays']['signal']['sha256']='0'*64
    bad=publish('bad.json',broken)
    with pytest.raises(ValueError,match='ARRAY_BYTES:signal'):cache.load_snapshot(bad)
    with np.load(saved['archive']['path'],allow_pickle=False)as archive:arrays={key:archive[key]for key in archive.files}
    arrays['tf_D1_owner_row_indices']=arrays['tf_D1_owner_row_indices']+100000
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);archive_ref=publish('shifted.npz',stream.getvalue())
    changed=copy.deepcopy(saved);changed['archive']=archive_ref
    changed['arrays']['tf_D1_owner_row_indices']['sha256']=cache.digest(arrays['tf_D1_owner_row_indices'])
    shifted=publish('shifted.json',changed)
    with pytest.raises(ValueError,match='OVERLAP_GLOBAL_TF_IDS:D1'):cache.overlap_parity(provider,shifted,archive_ref)


def test_readiness_preserves_real_missingness_and_macro_earliest_complete():
    provider=_provider(missing_macro=True);report=cache.yearly_readiness(provider)['2024']
    assert report['rows']==305 and report['macro_current_all61_observed_rows']==304
    first=provider.profile['macro_context_fields'][0]
    assert report['macro_missing_current_rows'][first]==1
    assert report['readiness']['input_not_ready']>0
    assert report['first_current_all61_macro_observed_ns']==int(provider.issue_ns[95])
    assert report['first_all_inputs_ready_ns']==int(provider.issue_ns[95])


def test_exact_parent_and_cli_gates_precede_io_and_torch_is_forbidden(monkeypatch):
    monkeypatch.setattr(cache,'read',lambda *_:pytest.fail('unbound receipt IO'))
    with pytest.raises(ValueError,match='EXACT_QUALIFIED_PARENT'):cache.metadata({})
    args=['cache','--plan','/wrong','--plan-sha256','0'*64,'--out-dir',str(cache.ROOT),'--preflight']
    monkeypatch.setattr(cache.sys,'argv',args)
    with pytest.raises(ValueError,match='CLI_SCOPE'):cache.main()
    with pytest.raises(RuntimeError,match='TORCH_FORBIDDEN'):cache._reject_torch('import',('torch',))
    assert cache.START_NS==int(pd.Timestamp('2021-01-01',tz='UTC').value)
    assert cache.END_NS==int(pd.Timestamp('2026-07-01',tz='UTC').value)
    assert cache.EXECUTION['internal_seconds']==3570 and cache.EXECUTION['external_wall_timeout_enforced']is False
    assert all(value==0 for value in cache.COUNTS.values())


def test_preflight_keeps_training_false_and_never_creates_root(tmp_path,monkeypatch):
    root=tmp_path/'absent';monkeypatch.setattr(cache,'ROOT',root)
    plan={'inputs':{},'metadata':{'dependencies':{},'feature_source_spec':{'parts':[]}},'source_base_commit':'unit'}
    plan_ref={'path':str(cache.PLAN),'sha256':'1'*64};policy_ref={'path':str(cache.POLICY),'sha256':'2'*64}
    policy={key:False for key in cache.owner.data.CLOSED_FLAGS}
    policy[cache.STAGE]={'plan':plan_ref,'output_root':str(root),'execution':cache.EXECUTION,
        'max_invocations':1,'claimed_invocations':0,'completed_invocations':0,'remaining_invocations':0,
        'relaunch_allowed':False,'training_authorized':False,'test_authorized':False,
        'status':'PREPARED_NOT_ADMITTED','execution_authorized':False,**cache.COUNTS}
    monkeypatch.setattr(cache.Path,'cwd',lambda:cache.REPO)
    monkeypatch.setattr(cache,'read',lambda binding:plan if binding==plan_ref else policy)
    monkeypatch.setattr(cache,'ref',lambda path:policy_ref)
    monkeypatch.setattr(cache,'build_plan',lambda **kwargs:plan);monkeypatch.setattr(cache,'source_clean',lambda p:True)
    assert cache.inspect('1'*64,admitted=False)[0]is plan and not root.exists()
    policy['training_enabled']=True
    with pytest.raises(ValueError,match='CLOSED_POLICY'):cache.inspect('1'*64,admitted=False)
    assert not root.exists()


def test_controller_no_claim_on_gate_failure_and_consumes_failure_then_separate_success(tmp_path,monkeypatch):
    root=tmp_path/'failed';monkeypatch.setattr(cache,'ROOT',root)
    cap={'memory_max':12*1024**3,'memory_high':12*1024**3,'swap':512*1024**2,'pids':64,
        'numerical_threads':1,'cgroup_path':'/sys/fs/cgroup/unit'}
    monkeypatch.setattr(cache.owner,'require_capped_cpu_research_producer_execution',lambda:cap)
    monkeypatch.setattr(cache.owner,'require_capped_lock_ancestry',lambda:{'scope':'/unit'})
    def blocked(*a,**k):raise ValueError('preclaim gate')
    monkeypatch.setattr(cache,'inspect',blocked)
    with pytest.raises(ValueError,match='preclaim gate'):cache.run('0'*64)
    assert not root.exists()
    plan_ref={'path':'unitplan','sha256':'1'*64};policy_ref={'path':'unitpolicy','sha256':'2'*64}
    monkeypatch.setattr(cache,'inspect',lambda *a,**k:({},plan_ref,policy_ref))
    def failed(*args):raise ValueError('overlap mismatch')
    monkeypatch.setattr(cache,'produce',failed)
    with pytest.raises(ValueError,match='overlap mismatch'):cache.run('0'*64)
    terminal=json.loads((root/'TERMINAL.json').read_text())
    assert terminal['status']=='FAILED_CONSUMED'and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False
    claim=json.loads((root/'CLAIM.json').read_text())
    assert claim['capped_execution']==cap and claim['cgroup_path']==cap['cgroup_path']
    assert claim['lock_ancestry']=={'scope':'/unit'} and claim['pid']==cache.os.getpid()
    assert claim['timestamp_utc']
    assert cache.signal.getitimer(cache.signal.ITIMER_REAL)==(0.,0.)
    complete=tmp_path/'complete';monkeypatch.setattr(cache,'ROOT',complete)
    monkeypatch.setattr(cache,'produce',lambda p:{'actual_counts':cache.COUNTS})
    monkeypatch.setattr(cache,'checked_inputs',lambda p:None);monkeypatch.setattr(cache,'source_clean',lambda p:True)
    real_ref=cache.ref
    monkeypatch.setattr(cache,'ref',lambda path:policy_ref if path==cache.POLICY else plan_ref if path==cache.PLAN else real_ref(path))
    refs=cache.run('0'*64);done=json.loads((complete/'RESULT.json').read_text())
    assert done['actual_counts']==cache.COUNTS and done['training_authorized']is False
    assert refs['terminal']['path']==str(complete/'TERMINAL.json')


def test_native_alias_vectors_keep_all_bytes_and_never_copy_dataframe_attrs():
    class CannotDeepcopy:
        def __deepcopy__(self,memo):raise AssertionError('DataFrame attrs were copied')
    names=features.owners.compile_full_ta_entry_execution_profile()['physical_field_names']['mtf']
    matrix=np.arange(8*len(names),dtype=np.float32).reshape(8,-1)
    matrix[1,:]=np.nan;matrix[2,:]=-0.0
    frame=pd.DataFrame(matrix,columns=names,copy=False)
    frame.attrs={'feats_np':matrix,'deepcopy_sentinel':CannotDeepcopy()}
    aliases=features.layers.model_native_m5_local_per_bar_exact_aliases()
    assert aliases and not frame._item_cache
    for _,name in aliases:
        actual=features._native_m5_alias_vector(frame,name)
        expected=matrix[:,names.index(name)]
        assert actual.dtype==np.float32 and actual.tobytes()==expected.tobytes()
        np.testing.assert_array_equal(actual,expected)
        assert np.shares_memory(actual,matrix)and not frame._item_cache
    with pytest.raises(KeyError):features._native_m5_alias_vector(frame,'not_a_genuine_column')
    assert not frame._item_cache


def test_numeric_bridge_accepts_only_exact_alias_layout_and_preserves_other_owners(tmp_path,monkeypatch):
    relative='gx1/contracts/market_state_native_m5_feature_inputs_v1.py'
    current=cache.Path(features.__file__).read_text()
    old=current.replace(cache._ALIAS_HELPER_SOURCE+'\n\n','').replace(
        'values[local]='+cache._ALIAS_NEW_EXPRESSION,'values[local]='+cache._ALIAS_OLD_EXPRESSION)
    assert old!=current and '_native_m5_alias_vector'not in old
    assert cache.alias_layout_numeric_identity(current)==cache.owner.numeric_ast_identity(
        old,excluded_functions=cache._ALIAS_EXCLUDED)
    with pytest.raises(ValueError,match='EXACT_ALIAS_HELPER_AST'):
        cache.alias_layout_numeric_identity(current.replace('m5.columns.get_loc(per_bar)',"m5.columns.get_loc('open')"))
    with pytest.raises(ValueError,match='EXACT_SINGLE_ALIAS_CALL'):
        cache.alias_layout_numeric_identity(current+"\nx=_native_m5_alias_vector(m5,per_bar)\n")
    other='gx1/features/unit_formula.py';originals={relative:old.encode(),other:b'def formula(x): return x+1\n'}
    for name,value in {relative:current,other:originals[other].decode()}.items():
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
    monkeypatch.setattr(cache,'REPO',tmp_path)
    monkeypatch.setattr(cache.owner,'NUMERIC_OWNER_PATHS',(relative,other))
    monkeypatch.setattr(cache.subprocess,'check_output',lambda args,**kwargs:originals[args[-1].split(':',1)[1]])
    old_plan={'source_base_commit':'unit','source_owners':{
        name:cache.hashlib.sha256(value).hexdigest()for name,value in originals.items()}}
    identities,bridge=cache.numeric_owner_identities(old_plan)
    assert len(identities)==2 and bridge['exact_replaced_alias_calls']==1
    assert bridge['original_sha256']==old_plan['source_owners'][relative]
    assert bridge['current_sha256']==cache.sha(tmp_path/relative)
    (tmp_path/other).write_text('def formula(x): return x+2\n')
    with pytest.raises(ValueError,match='NUMERIC_OWNER_CHANGED:'):
        cache.numeric_owner_identities(old_plan)
    (tmp_path/other).write_bytes(originals[other])
    (tmp_path/relative).write_text(current.replace("lineage[local]='same_native_M5_owner:'+per_bar",
        "lineage[local]='changed_owner:'+per_bar"))
    with pytest.raises(ValueError,match='NUMERIC_OWNER_CHANGED:'):
        cache.numeric_owner_identities(old_plan)




def test_phase_receipt_is_immutable_small_and_records_process_peak_rss(tmp_path,monkeypatch):
    monkeypatch.setattr(cache,'ROOT',tmp_path)
    monkeypatch.setattr(cache.resource,'getrusage',lambda _:SimpleNamespace(ru_maxrss=12345))
    binding=cache.phase('01_BEFORE_RAW_READ');saved=cache.read(binding)
    assert saved['phase']=='01_BEFORE_RAW_READ'and saved['stage']==cache.STAGE
    assert saved['ru_maxrss_kib']==12345 and saved['pid']==cache.os.getpid()and saved['timestamp_utc']
    before=(tmp_path/'PHASE_01_BEFORE_RAW_READ.json').read_bytes()
    assert len(before)<1024
    with pytest.raises(ImmutableEventAuthorityError,match='already exists'):cache.phase('01_BEFORE_RAW_READ')
    assert (tmp_path/'PHASE_01_BEFORE_RAW_READ.json').read_bytes()==before


def test_produce_phase_witnesses_bracket_only_existing_operations(monkeypatch):
    provider=_provider();events=[]
    provider.surfaces['axes']={**provider.surfaces['axes'],'issue_start_ns':cache.START_NS,'issue_end_ns':cache.END_NS}
    parameters={'squeeze_manifest':{'path':'/unit/params.json','sha256':'1'*64}}
    meta={'parameter_inputs':{},'geometry':{},'parameter_reuse':parameters,
        'feature_source_spec':{'manifest':{}},'expected_source_rows':400,'expected_issue_rows_by_year':{},
        'feature_contract':provider.surfaces['provider_contract'],'reference_snapshot':{},'reference_archive':{}}
    def event(name,value):
        events.append(name);return value
    monkeypatch.setattr(cache.owner,'parameter_artifact_metadata',lambda *args:(parameters,{}))
    monkeypatch.setattr(cache,'load_volatility_squeeze_artifact_manifest',lambda *a,**k:None)
    monkeypatch.setattr(cache,'phase',lambda name:event(name,{'phase':name}))
    monkeypatch.setattr(cache.owner,'read_feature_source',lambda *a:event('RAW_READ',object()))
    monkeypatch.setattr(features,'bind_native_m5_feature_source',lambda *a,**k:{'source_spec':meta['feature_source_spec'],'native_m5_manifest':{},'rows':400})
    monkeypatch.setattr(features,'build_native_m5_feature_surfaces',lambda *a,**k:event('FEATURE_BUILD',provider.surfaces))
    monkeypatch.setattr(features,'NativeM5FeatureProvider',lambda *a:event('PROVIDER_INIT',provider))
    monkeypatch.setattr(cache.owner,'publish_feature_snapshot',lambda *a,**k:event('SNAPSHOT_PUBLISH',{}))
    monkeypatch.setattr(cache,'load_snapshot',lambda *a:event('SNAPSHOT_RELOAD',provider))
    monkeypatch.setattr(cache,'yearly_readiness',lambda *a:{})
    monkeypatch.setattr(cache,'overlap_parity',lambda *a:{})
    monkeypatch.setattr(cache,'read',lambda *a:{'archive':{}})
    report=cache.produce({'metadata':meta})
    assert events==['01_BEFORE_RAW_READ','RAW_READ','02_AFTER_RAW_READ',
        '03_BEFORE_FEATURE_BUILD','FEATURE_BUILD','04_AFTER_FEATURE_BUILD',
        '05_BEFORE_PROVIDER_INIT','PROVIDER_INIT','06_AFTER_PROVIDER_INIT',
        '07_BEFORE_SNAPSHOT_PUBLISH','SNAPSHOT_PUBLISH','08_AFTER_SNAPSHOT_PUBLISH',
        '09_BEFORE_SNAPSHOT_RELOAD','SNAPSHOT_RELOAD','10_AFTER_SNAPSHOT_RELOAD']
    assert len(report['phases'])==10 and report['actual_counts']==cache.COUNTS


def test_macro_research_import_and_pure_asof_helper_are_torch_and_data_free_in_fresh_process():
    code='''import json,sys
def deny(event,args):
    if event=='import'and(args[0]=='torch'or args[0].startswith('torch.')):
        raise RuntimeError('Torch import forbidden')
    if event=='open'and isinstance(args[0],str)and(
        '/GX1_DATA/'in args[0]or '/GX1_RUNS/'in args[0]or args[0].endswith(('.parquet','.npz'))):
        raise RuntimeError('Data open forbidden')
    if event.startswith('socket.'):raise RuntimeError('Network forbidden')
sys.addaudithook(deny)
from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
firewall=install_native_m5_direction_io_firewall()
from gx1.features import d1_macro_context_v1 as macro
research=macro._research()
import numpy as np,pandas as pd
at=pd.DatetimeIndex(['2024-05-01T00:00Z'])
out=research.align_asof(pd.DatetimeIndex(['2024-05-01T00:00Z','2024-05-01T00:01Z']),
    pd.Series(at),pd.DataFrame({'value':[17.0]}))
assert np.isnan(out.iloc[0,0])and out.iloc[1,0]==17.0
assert not any(name=='torch'or name.startswith('torch.')for name in sys.modules)
assert firewall['denied_attempts']==0
print(json.dumps({'Torch_imported':False,'data_values_decoded':0,'model_constructions':0,'pure_asof_helper':True}))
'''
    result=cache.subprocess.run([cache.sys.executable,'-c',code],cwd=cache.REPO,
        capture_output=True,text=True,check=True,timeout=30)
    assert json.loads(result.stdout)=={'Torch_imported':False,'data_values_decoded':0,
        'model_constructions':0,'pure_asof_helper':True}


def test_research_bridge_strips_only_two_import_sources_and_rejects_body_or_macro_change(tmp_path,monkeypatch):
    first,second=cache.RESEARCH_LAZY_IMPORTS
    old=('from '+first+' import RidgeGram, Tape, fit_hgb\n'
        'from '+second+' import ResearchFinancingCurve, portfolio_path\n'
        'import math\n'
        'def pure_macro(x):\n    return x+1\n'
        "def load_oanda_panel_d1(spec):\n    return pd.read_parquet(path, columns=['time','open','high','low','close'])\n"
        'def fit(x):\n    return fit_hgb(x)\n'
        'def portfolio(x):\n    return portfolio_path(x)\n')
    new=('import math\n'
        'def pure_macro(x):\n    return x+1\n'
        "def load_oanda_panel_d1(spec):\n    return pd.read_parquet(path, columns=['time','open','high','low','close'], filters=[('time','<',sealed_period_read_end(spec))])\n"
        'def fit(x):\n    from '+first+' import fit_hgb\n    return fit_hgb(x)\n'
        'def portfolio(x):\n    from '+second+' import portfolio_path\n    return portfolio_path(x)\n')
    assert cache.research_import_numeric_identity(old)==cache.research_read_end_numeric_identity(new)
    path=tmp_path/cache.RESEARCH_SOURCE_PATH;path.parent.mkdir(parents=True);path.write_text(new)
    monkeypatch.setattr(cache,'REPO',tmp_path)
    monkeypatch.setattr(cache.subprocess,'check_output',lambda *a,**k:old.encode())
    plan={'source_base_commit':'unit','source_owners':{cache.RESEARCH_SOURCE_PATH:cache.hashlib.sha256(old.encode()).hexdigest()}}
    proof=cache.research_import_bridge(plan)
    assert proof['all_other_module_statements_and_macro_function_bodies_unchanged']is True
    assert proof['original_sha256']==plan['source_owners'][cache.RESEARCH_SOURCE_PATH]
    for changed in (new.replace('x+1','x+2'),new.replace('return fit_hgb(x)','return fit_hgb(x+1)'),
                    new.replace('import math','import cmath')):
        path.write_text(changed)
        with pytest.raises(ValueError,match='RESEARCH_NON_IMPORT_AST_CHANGED'):cache.research_import_bridge(plan)
    plan['source_owners'][cache.RESEARCH_SOURCE_PATH]='0'*64
    with pytest.raises(ValueError,match='ORIGINAL_RESEARCH_SOURCE'):cache.research_import_bridge(plan)




def test_research_read_end_bridge_rejects_other_predicate_columns_or_reader_changes():
    valid=("def load_oanda_panel_d1(spec):\n"
        "    return pd.read_parquet(path, columns=['time','open','high','low','close'], "
        "filters=[('time','<',sealed_period_read_end(spec))])\n")
    cache.research_read_end_numeric_identity(valid)
    for changed in (valid.replace("'<'","'<='"),
                    valid.replace('sealed_period_read_end(spec)','spec["read_end"]'),
                    valid.replace("'high','low'","'high','volume'"),
                    valid.replace('load_oanda_panel_d1','different_reader')):
        with pytest.raises(ValueError,match='EXACT_PANEL_READ_END_PREDICATE'):
            cache.research_read_end_numeric_identity(changed)


def test_new_cache_parent_is_qualified_success_and_scope_never_relabels_old_prefix():
    assert set(cache.INPUT_REFS)=={'cache_qualification','cache_result','cache_terminal'}
    assert all('/NATIVE_M5_TA_HISTORY_EXPANSION_20261120_003/'in v['path']for v in cache.INPUT_REFS.values())
    assert cache.START_NS==int(pd.Timestamp('2021-01-01',tz='UTC').value)
    assert cache.END_NS==int(pd.Timestamp('2026-07-01',tz='UTC').value)
    assert cache.EXECUTION['memory']=='12G'and cache.EXECUTION['internal_seconds']==3570
    assert all(value==0 for value in cache.COUNTS.values())


def test_overlap_uses_shared_global_rows_and_checks_old_clock_prefix_after_append(tmp_path):
    original=_provider()
    snapshot=cache.owner.publish_feature_snapshot(original,publish_artifact=_publisher(tmp_path))
    saved=cache.read(snapshot)
    candidate=copy.copy(original);candidate.surfaces=dict(original.surfaces)
    candidate.bar_start_ns=np.r_[original.bar_start_ns,
        original.bar_start_ns[-1]+features.BAR_NS*np.arange(1,11,dtype=np.int64)]
    candidate.issue_ns=candidate.bar_start_ns+features.BAR_NS
    candidate.retained_source_rows=original.retained_source_rows[120:]
    candidate.surfaces['retained_source_rows']=candidate.retained_source_rows
    candidate.surfaces['retained_issue_rows']=original.surfaces['retained_issue_rows'][120:]
    for key in ('signal','ctx_cont','ctx_cat','observed'):
        candidate.surfaces[key]=original.surfaces[key][120:].copy()
    frames={}
    for tf,frame in original.surfaces['multi_tf_sources'].items():
        values=frame.attrs['feats_np'][20:]
        kept=pd.DataFrame(values,index=frame.index[20:],columns=frame.columns,copy=False)
        kept.attrs={key:value[20:]if isinstance(value,np.ndarray)else value for key,value in frame.attrs.items()}
        frames[tf]=kept
    candidate.surfaces['multi_tf_sources']=frames
    proof=cache.overlap_parity(candidate,snapshot,saved['archive'])
    assert proof['array_count']==27 and proof['reference_issues']==185 and proof['shared_local_rows']==280
    assert proof['original_full_bar_axis_is_exact_prefix']is True
    candidate.surfaces['signal'][0,0]+=1
    with pytest.raises(ValueError,match='OVERLAP_EXACT:signal'):
        cache.overlap_parity(candidate,snapshot,saved['archive'])
    candidate.surfaces['signal'][0,0]-=1
    candidate.bar_start_ns=candidate.bar_start_ns.copy();candidate.bar_start_ns[0]+=features.BAR_NS
    with pytest.raises(ValueError,match='OVERLAP_EXACT:bar_start_ns'):
        cache.overlap_parity(candidate,snapshot,saved['archive'])


# Only synthetic arrays/JSON below; the correction never invokes the old TA build.
def _cot_fixture():
    from gx1.features import d1_macro_context_v1 as macro
    provider=_provider();surfaces=dict(provider.surfaces)
    times=pd.to_datetime(provider.bar_start_ns[provider.retained_source_rows]+features.BAR_NS,utc=True)
    frame=pd.DataFrame({name:np.ones(len(times),np.float64)for name in macro.D1_MACRO_CONTEXT_FIELDS})
    frame.insert(0,'decision_time',times)
    spec={'cot':{'publication_clock':dict(cache.COT_LEGACY_CLOCK)},'oanda_panel':{},'drivers':{},
        'test_seal_utc':'2026-07-01T00:00:00Z'}
    current=features.native_m5_feature_provider_contract();legacy=cache.cot_legacy_provider(current)
    binding=macro.d1_macro_context_binding(spec,frame);binding['contract']=legacy['macro_contract']
    binding['binding_sha256']=features._sha({k:v for k,v in binding.items()if k!='binding_sha256'})
    axes={k:v for k,v in surfaces['axes'].items()if k!='contract_sha256'}
    axes['macro_binding_sha256']=binding['binding_sha256']
    surfaces.update(provider_contract=legacy,macro_binding=binding,axes=features._seal(axes))
    return surfaces,frame,spec,{'legacy_provider_contract':legacy,'feature_contract':current}


def _cot_specs():
    old={'cot':{'publication_clock':dict(cache.COT_LEGACY_CLOCK),'result':{'same':'source'},'availability_lag_days':7},
        'oanda_panel':{'same':1},'drivers':{'same':2}}
    new=copy.deepcopy(old);new['cot']['publication_clock']['sha256']='1'*64
    clock={'schema_version':'gx1_cftc_cot_delayed_publication_clock_v1',
        'rows':[{'as_of':'2025-10-07','publication_date_ny':'2025-11-19'}]}
    corrected=copy.deepcopy(clock);corrected['rows'].append({'as_of':'2023-01-31','publication_date_ny':'2023-02-24'})
    return old,new,clock,corrected


def test_cot_publication_correction_accepts_only_calendar_and_derived_contract_change():
    old,new,clock,corrected=_cot_specs()
    cache.cot_spec_transition(old,new,clock,corrected)
    for branch in ('cot','oanda_panel','drivers'):
        bad=copy.deepcopy(new);bad[branch]['unrelated']='changed'
        with pytest.raises(ValueError,match='ONLY_CLOCK_SOURCE_CHANGE'):
            cache.cot_spec_transition(old,bad,clock,corrected)
    bad=copy.deepcopy(corrected);bad['rows'].append(bad['rows'][0])
    with pytest.raises(ValueError,match='CLOCK_DUPLICATE'):cache.cot_spec_transition(old,new,clock,bad)
    bad=copy.deepcopy(corrected);bad['rows'][0]['publication_date_ny']='2025-10-01'
    with pytest.raises(ValueError,match='CLOCK_EXTENSION'):cache.cot_spec_transition(old,new,clock,bad)
    current=features.native_m5_feature_provider_contract();legacy=cache.cot_legacy_provider(current)
    assert legacy['macro_contract']['source_spec']==cache.COT_LEGACY_SPEC
    assert {k:v for k,v in legacy.items()if k not in ('macro_contract','contract_sha256')}=={
        k:v for k,v in current.items()if k not in ('macro_contract','contract_sha256')}


@pytest.mark.parametrize('suffix',['.pt','.npz','.parquet'])
def test_cot_publication_correction_json_preflight_never_hashes_payload(suffix,monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('payload hash/read before admission')
    monkeypatch.setattr(cache,'read',forbidden);monkeypatch.setattr(cache,'ref',forbidden)
    binding={'path':'/unit/input'+suffix,'sha256':'1'*64}
    with pytest.raises(ValueError,match='JSON_ONLY'):cache.cot_read(binding)
    with pytest.raises(ValueError,match='JSON_ONLY'):cache.cot_json_ref(binding['path'])


def test_cot_publication_correction_wrong_legacy_hash_fails_before_decode(monkeypatch):
    monkeypatch.setattr(cache.subprocess,'check_output',lambda *a,**k:b'{}')
    with pytest.raises(ValueError,match='LEGACY_JSON_HASH'):cache.cot_git_json(cache.COT_LEGACY_SPEC)


def test_cot_publication_correction_invalid_inventory_never_hashes_named_payload(tmp_path,monkeypatch):
    monkeypatch.setattr(cache,'COT_ROOT',tmp_path/'absent')
    plan={'source_owners':{**{name:'1'*64 for name in cache.COT_SOURCE_PATHS},'/unit/weights.pt':'2'*64}}
    monkeypatch.setattr(cache,'cot_read',lambda *a:plan)
    monkeypatch.setattr(cache,'sha',lambda *a:pytest.fail('unvalidated inventory was hashed'))
    with pytest.raises(ValueError,match='SOURCE_SET'):cache.cot_inspect('1'*64,admitted=False)


def test_cot_publication_correction_alias_mask_axis_and_four_column_scope():
    surfaces,frame,spec,metadata=_cot_fixture()
    original_signal=surfaces['signal'].copy();original_ctx=surfaces['ctx_cont'].copy()
    frame.loc[100,cache.COT_FIELDS[0]]=np.nan
    frame.loc[110,cache.COT_FIELDS[1]]=np.float64(3.141592653589793)
    provider,report=cache.cot_replace_surfaces(surfaces,frame,spec,metadata)
    fields=provider.profile['physical_field_names']
    for name in cache.COT_FIELDS:
        si=fields['signal'].index('ctx_cont.'+name);ci=fields['ctx_cont'].index(name)
        np.testing.assert_array_equal(provider.surfaces['signal'][:,si],provider.surfaces['ctx_cont'][:,ci])
    np.testing.assert_array_equal(provider.surfaces['observed'],np.isfinite(provider.surfaces['signal']))
    np.testing.assert_array_equal(original_signal,surfaces['signal']);np.testing.assert_array_equal(original_ctx,surfaces['ctx_cont'])
    assert report['canonical_F64_macro_fields']==61 and report['unchanged_F32_macro_fields']==57
    assert report['changed_fields'][cache.COT_FIELDS[0]]['changed_rows']==1
    assert provider.source_axes_sha256!=surfaces['axes']['contract_sha256']
    assert provider.surfaces['axes']['retained_timeframes']==surfaces['axes']['retained_timeframes']
    assert provider.surfaces['field_lineage']==surfaces['field_lineage']
    assert provider.surfaces['macro_binding']['decision_clock_sha256']==surfaces['macro_binding']['decision_clock_sha256']
    assert not provider.input_ready[100] and provider.input_ready[99]


@pytest.mark.parametrize('fault',['noncot','axis','alias','observed','provider'])
def test_cot_publication_correction_rejects_scope_and_legacy_corruption(fault):
    surfaces,frame,spec,metadata=_cot_fixture()
    expected={'noncot':'NONCOT57_PARITY','axis':'MACRO_AXIS','alias':'LEGACY_ALIASES_MASK',
        'observed':'LEGACY_ALIASES_MASK','provider':'LEGACY_PROVIDER_TRANSITION'}[fault]
    if fault=='noncot':frame.loc[0,'real10y_level']=123.
    if fault=='axis':frame.loc[0,'decision_time']+=pd.Timedelta(minutes=5)
    if fault=='alias':
        value=surfaces['ctx_cont'].copy();value[0,0]=999;surfaces['ctx_cont']=value
    if fault=='observed':
        value=surfaces['observed'].copy();value[0,0]=False;surfaces['observed']=value
    if fault=='provider':
        surfaces['provider_contract']=copy.deepcopy(surfaces['provider_contract']);surfaces['provider_contract']['window_rows']['M5']+=1
    with pytest.raises(ValueError,match=expected):cache.cot_replace_surfaces(surfaces,frame,spec,metadata)


def test_cot_publication_correction_private_legacy_decode_then_current_reload(tmp_path):
    surfaces,frame,spec,metadata=_cot_fixture();old=_publisher(tmp_path/'legacy')
    snapshot=cache.owner.publish_feature_snapshot(SimpleNamespace(surfaces=surfaces),publish_artifact=old)
    with pytest.raises(RuntimeError,match='PROVIDER_IDENTITY'):cache.load_snapshot(snapshot)
    decoded=cache._decode_snapshot_surfaces(snapshot)
    frame.loc[100,cache.COT_FIELDS[0]]=2.
    provider,_=cache.cot_replace_surfaces(decoded,frame,spec,metadata)
    preserved=cache.cot_array_parity(provider.surfaces,cache.read(snapshot));assert len(preserved)==24
    new=cache.owner.publish_feature_snapshot(provider,publish_artifact=_publisher(tmp_path/'current'))
    reloaded=cache.load_snapshot(new)
    assert cache.cot_array_parity(reloaded.surfaces,cache.read(snapshot))==preserved
    corrupted=dict(reloaded.surfaces);corrupted['ctx_cat']=reloaded.surfaces['ctx_cat'].copy()
    corrupted['ctx_cat'][0,0]+=1
    with pytest.raises(ValueError,match='PRESERVED_ARRAY:ctx_cat'):cache.cot_array_parity(corrupted,cache.read(snapshot))


def test_cot_publication_correction_preflight_closed_gate_is_json_only(tmp_path,monkeypatch):
    monkeypatch.setattr(cache,'COT_ROOT',tmp_path/'absent')
    plan={'source_owners':{name:'1'*64 for name in cache.COT_SOURCE_PATHS},'source_base_commit':'unit'}
    pr={'path':str(cache.COT_PLAN),'sha256':'1'*64};policy_ref={'path':str(cache.POLICY),'sha256':'2'*64}
    gate={'plan':pr,'output_root':str(cache.COT_ROOT),'execution':cache.COT_EXECUTION,'limits':cache.COT_COUNTS,
        'max_invocations':1,'claimed_invocations':0,'completed_invocations':0,'remaining_invocations':0,
        'relaunch_allowed':False,'training_authorized':False,'test_authorized':False,
        'status':'PREPARED_NOT_ADMITTED','execution_authorized':False}
    policy={key:False for key in cache.owner.data.CLOSED_FLAGS}
    policy[cache.COT_PARENT]={'goal_status':'ACTIVE','large_training_qualified':False,cache.COT_GATE:gate}
    monkeypatch.setattr(cache,'cot_read',lambda binding:plan if binding==pr else policy)
    monkeypatch.setattr(cache,'cot_json_ref',lambda path:policy_ref)
    monkeypatch.setattr(cache,'cot_build_plan',lambda **kwargs:plan);monkeypatch.setattr(cache,'cot_source_clean',lambda p:True)
    calls=[];monkeypatch.setattr(cache,'cot_checked_inputs',lambda p,**kw:calls.append(kw))
    assert cache.cot_inspect('1'*64,admitted=False)[0]is plan and calls==[{'payloads':False}]
    policy['training_enabled']=True
    with pytest.raises(ValueError,match='CLOSED_POLICY'):cache.cot_inspect('1'*64,admitted=False)
    assert not cache.COT_ROOT.exists()


def test_cot_publication_correction_consumed_failure_and_success_receipts(tmp_path,monkeypatch):
    root=tmp_path/'failure';monkeypatch.setattr(cache,'COT_ROOT',root)
    monkeypatch.setattr(cache.owner,'require_capped_cpu_research_producer_execution',lambda:{'unit':'exact_guard'})
    monkeypatch.setattr(cache,'cot_timeout_ancestry',lambda sha:{'outer_seconds':900,'kill_after_seconds':10})
    def blocked(*a,**k):raise ValueError('preclaim gate')
    monkeypatch.setattr(cache,'cot_inspect',blocked)
    with pytest.raises(ValueError,match='preclaim gate'):cache.cot_run('1'*64)
    assert not root.exists()
    pr={'path':'/unit/plan.json','sha256':'1'*64};policy={'path':'/unit/policy.json','sha256':'2'*64}
    monkeypatch.setattr(cache,'cot_inspect',lambda *a,**k:({},pr,policy))
    monkeypatch.setattr(cache,'cot_bookend',lambda *a:True)
    def fail(*a,**k):raise ValueError('synthetic parity failure')
    monkeypatch.setattr(cache,'cot_produce',fail)
    with pytest.raises(ValueError,match='synthetic parity failure'):cache.cot_run('1'*64)
    terminal=json.loads((root/'TERMINAL.json').read_text())
    assert terminal['status']=='FAILED_CONSUMED'and terminal['result']is None
    assert terminal['source_unchanged']is True and terminal['test_data_used']is False
    assert terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False
    assert terminal['claim']['path']==str(root/'CLAIM.json')
    assert cache.signal.getitimer(cache.signal.ITIMER_REAL)==(0.,0.)
    monkeypatch.setattr(cache,'COT_ROOT',tmp_path/'success')
    monkeypatch.setattr(cache,'cot_produce',lambda p:{'actual_counts':dict(cache.COT_COUNTS)})
    monkeypatch.setattr(cache,'cot_checked_inputs',lambda *a,**k:None)
    refs=cache.cot_run('1'*64);result=json.loads(Path(refs['result']['path']).read_text())
    terminal=json.loads(Path(refs['terminal']['path']).read_text())
    assert result['actual_counts']==cache.COT_COUNTS and result['training_authorized']is False
    assert terminal['result']==refs['result']and terminal['source_unchanged']is True
    assert terminal['status']=='COMPLETE'and terminal['exit_code']==0


def test_cot_publication_correction_cli_is_fixed_and_does_not_reopen_old_plan(monkeypatch):
    calls=[];monkeypatch.setattr(cache,'cot_inspect',lambda *a,**kw:(calls.append((a,kw))or({}, {}, {})))
    cache.cot_main(['--cot-publication-correction','--plan-sha256','1'*64,'--preflight'])
    assert calls==[(('1'*64,),{'admitted':False})]
    with pytest.raises(ValueError,match='COT_EXACT_CLI'):cache.cot_main(['--cot-publication-correction','--plan','/wrong','--plan-sha256','1'*64])
    assert cache.COT_EXECUTION['internal_seconds']==870 and cache.COT_EXECUTION['outer_seconds']==900
    assert cache.COT_COUNTS['normalization_fits']==cache.COT_COUNTS['TA_rebuilds']==cache.COT_COUNTS['target_values_decoded']==0


@pytest.mark.parametrize('fault',[None,'kill','sha','python','missing'])
def test_cot_publication_correction_requires_exact_outer_timeout(fault,monkeypatch):
    argv=['timeout','--kill-after=10s','900s','.venv/bin/python','-m',cache.COT_MODULE,
        '--cot-publication-correction','--plan-sha256','1'*64]
    if fault=='kill':argv[1]='--kill-after=60s'
    if fault=='sha':argv[-1]='2'*64
    if fault=='python':argv[3]='/unbound/python'
    if fault=='missing':argv=['bash']
    monkeypatch.setattr(cache.os,'getppid',lambda:123)
    monkeypatch.setattr(cache.Path,'read_bytes',lambda self:('\0'.join(argv)+'\0').encode())
    monkeypatch.setattr(cache.Path,'read_text',lambda self,*a,**k:'PPid:\t0\n')
    if fault:
        with pytest.raises(ValueError,match='COT_.*TIMEOUT'):cache.cot_timeout_ancestry('1'*64)
    else:
        assert cache.cot_timeout_ancestry('1'*64)=={'argv':argv,'pid':123,'outer_seconds':900,'kill_after_seconds':10}


@pytest.mark.parametrize('fault',[None,'missing','early','duplicate'])
def test_cot_publication_correction_actual_clock_witness(fault):
    calendar=[{'as_of':'2020-12-21','publication_date_ny':'2020-12-28'},
        {'as_of':'2025-11-10','publication_date_ny':'2025-11-19'}]
    cot=pd.DataFrame({'as_of':pd.to_datetime([r['as_of']for r in calendar],utc=True),
        'available_at':pd.to_datetime(['2020-12-29T05:00:00Z','2025-11-20T05:00:00Z'],utc=True)})
    if fault=='missing':cot=cot.iloc[:1]
    if fault=='early':cot.loc[1,'available_at']=pd.Timestamp('2025-11-17T00:00:00Z')
    if fault=='duplicate':cot=pd.concat([cot,cot.iloc[:1]],ignore_index=True)
    args={'calendar_rows':calendar,'lag_days':7,'read_end':pd.Timestamp('2026-07-01T00:00:00Z')}
    if fault:
        with pytest.raises(ValueError,match='COT_WITNESS_'):cache.cot_clock_witness(cot,**args)
    else:
        result=cache.cot_clock_witness(cot,**args)
        assert result['matched_rows']==result['calendar_rows']==2
        assert result['excluded_at_or_after_seal']==0
        assert result['all_declared_preseal_rows_matched_exactly_once']
        assert result['all_available_at_exact_max_lag_official_day_end']
        assert result['rows'][0]['actual_available_ns']==pd.Timestamp('2020-12-29T05:00:00Z').value
        assert result['rows'][1]['actual_available_ns']==pd.Timestamp('2025-11-20T05:00:00Z').value
        assert result['canonical_rows_sha256']==features._sha(result['rows'])
        assert len(result['actual_as_of_available_at_sha256'])==64



def _cot_norm_statistics_fixture():
    names=['ctx_cont.'+f for f in cache.COT_FIELDS]+['ctx_cont.unmodified']
    ctx=[*cache.COT_FIELDS,'unmodified']
    def surface(name,fields):
        return {'surface':name,'field_names':fields,'field_count':len(fields),'stats_sha256':'old',
            'center':[0.]*len(fields),'scale':[1.]*len(fields),'binary_mask':[0]*len(fields),
            'categorical_mask':[0]*len(fields),'scale_source':['raw_iqr']*len(fields),
            'train_transformed_min':[-1.]*len(fields),'train_transformed_max':[1.]*len(fields)}
    old={'contract_sha256':'old','source_axes_sha256':'old','feature_provider_contract_sha256':'old',
        'fit_population':{'same':'wholeTRAIN'},'ctx_cat':{'same':'categorical'},'temporal_aliases':['same'],
        'surfaces':{'signal':surface('signal',names),'ctx_cont':surface('ctx_cont',ctx),
            **{'mtf_'+tf.lower():surface('mtf_'+tf.lower(),['native'])for tf in features.TIMEFRAMES}}}
    new=copy.deepcopy(old);new.update(contract_sha256='new',source_axes_sha256='new',feature_provider_contract_sha256='new')
    for name in ('signal','ctx_cont'):
        new['surfaces'][name]['center'][0]=2.;new['surfaces'][name]['stats_sha256']='new'
    return old,new


@pytest.mark.parametrize('fault',[None,'noncot','population','mtf','alias','category'])
def test_cot_normalization_all_unchanged_stats_and_population_are_exact(fault):
    old,new=_cot_norm_statistics_fixture()
    if fault=='noncot':new['surfaces']['signal']['scale'][-1]=2.
    if fault=='population':new['fit_population']['same']='filtered'
    if fault=='mtf':new['surfaces']['mtf_d1']['center'][0]=2.
    if fault=='alias':new['temporal_aliases']=['changed']
    if fault=='category':new['ctx_cat']['same']='changed'
    if fault:
        with pytest.raises(ValueError,match='COT_NORM_'):cache.cot_norm_stats_comparison(old,new)
    else:
        result=cache.cot_norm_stats_comparison(old,new)
        assert result['all_nonCOT_perfield_statistics_exact']
        assert result['all_five_MTF_surface_contracts_exact']
        assert result['unchanged_field_counts']['signal']==1


def test_cot_normalization_independent_window_union_merges_overlap_and_preserves_gaps():
    rows=np.asarray([95,96,200,202],np.int64)
    expected=np.r_[np.arange(97,dtype=np.int64),np.arange(105,203,dtype=np.int64)]
    np.testing.assert_array_equal(cache.cot_norm_local_union(rows,96),expected)
    with pytest.raises(ValueError,match='LOCAL_UNION'):cache.cot_norm_local_union(np.asarray([94],np.int64),96)


def _cot_norm_witness_fixture():
    provider=_provider();surfaces=dict(provider.surfaces);fields=provider.profile['physical_field_names']
    signal=surfaces['signal'].copy();ctx=surfaces['ctx_cont'].copy()
    stats={}
    for name in ('signal','ctx_cont'):
        width=len(fields[name]);stats[name]={key:[0.]*width for key in cache.COT_NORM_VECTOR_STATS}
    for j,field in enumerate(cache.COT_FIELDS):
        si=fields['signal'].index('ctx_cont.'+field);ci=fields['ctx_cont'].index(field)
        raw=np.arange(400,dtype=np.float32)+j;signal[:,si]=raw;ctx[:,ci]=raw
        for name,index,minvalue in (('signal',si,-1.),('ctx_cont',ci,(95.-199.5)/199.5)):
            expected={'center':199.5+j,'scale':199.5,'scale_source':'raw_iqr','binary_mask':0,'categorical_mask':0,
                'train_transformed_min':float(np.arcsinh(minvalue)),'train_transformed_max':float(np.arcsinh(1.))}
            for key,value in expected.items():stats[name][key][index]=value
    surfaces.update(signal=features._owned(signal,np.float32),ctx_cont=features._owned(ctx,np.float32),
        observed=features._owned(np.isfinite(signal),np.bool_))
    provider=features.NativeM5FeatureProvider(surfaces)
    rows=np.arange(95,400,dtype=np.int64)
    return provider,rows,{'fit_population':{'local_rows':400},'surfaces':stats}


@pytest.mark.parametrize('fault',[None,'signal_stat','ctx_stat','population'])
def test_cot_normalization_four_actual_raw_columns_have_independent_stats_proof(fault):
    provider,rows,normalization=_cot_norm_witness_fixture()
    fields=provider.profile['physical_field_names']
    if fault=='signal_stat':normalization['surfaces']['signal']['center'][fields['signal'].index('ctx_cont.'+cache.COT_FIELDS[0])]+=1
    if fault=='ctx_stat':normalization['surfaces']['ctx_cont']['train_transformed_min'][fields['ctx_cont'].index(cache.COT_FIELDS[0])]+=1
    if fault=='population':normalization['fit_population']['local_rows']=399
    if fault:
        with pytest.raises(ValueError,match='COT_NORM_'):cache.cot_norm_four_stats_witness(provider,rows,normalization)
    else:
        result=cache.cot_norm_four_stats_witness(provider,rows,normalization)
        assert result['independent_columns']==4 and result['local_union_rows']==400
        assert result['per_field'][cache.COT_FIELDS[0]]['signal']['center']==199.5
        assert result['per_field'][cache.COT_FIELDS[0]]['signal']['scale']==199.5
        assert result['per_field'][cache.COT_FIELDS[0]]['ctx_cont']['population_rows']==305


def test_cot_normalization_source_bridge_preserves_parent_functions_and_numeric_owners(tmp_path,monkeypatch):
    driver='gx1/audit/build_native_m5_ta_feature_cache_v1.py';numeric='gx1/features/pure.py';guard='scripts/gx1_capped_run.sh'
    before="def main():\n    return None\n\ndef cot_run():\n    return 1\n"
    after="def main():\n    if '--cot-normalization' in sys.argv[1:]:\n        return cot_norm_main(sys.argv[1:])\n    return None\n\ndef cot_run():\n    return 1\n"
    for name,text in ((driver,after),(numeric,'same'),(guard,'new guard')):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    monkeypatch.setattr(cache,'REPO',tmp_path);monkeypatch.setattr(cache,'COT_SOURCE_PATHS',(driver,numeric,guard))
    monkeypatch.setattr(cache.subprocess,'check_output',lambda *a,**k:before.encode())
    parent={'source_owners':{driver:cache.hashlib.sha256(before.encode()).hexdigest(),numeric:cache.sha(tmp_path/numeric),guard:'0'*64}}
    proof=cache.cot_norm_parent_source_bridge(parent)
    assert proof['preserved_parent_source_hashes']=={numeric:cache.sha(tmp_path/numeric)}
    assert proof['old_functions_AST_preserved_except_exact_norm_dispatch']==2
    (tmp_path/driver).write_text(after.replace('return 1','return 2'))
    with pytest.raises(ValueError,match='OLD_FUNCTION_CHANGED:cot_run'):cache.cot_norm_parent_source_bridge(parent)
    (tmp_path/driver).write_text(after);(tmp_path/numeric).write_text('changed')
    with pytest.raises(ValueError,match='NUMERIC_SOURCE_CHANGED'):cache.cot_norm_parent_source_bridge(parent)


def test_cot_normalization_preflight_validates_inventory_before_hash_and_closed_policy(tmp_path,monkeypatch):
    monkeypatch.setattr(cache,'COT_NORM_ROOT',tmp_path/'absent')
    pr={'path':str(cache.COT_NORM_PLAN),'sha256':'1'*64};policy_ref={'path':str(cache.POLICY),'sha256':'2'*64}
    plan={'source_owners':{name:'1'*64 for name in cache.COT_SOURCE_PATHS},'source_base_commit':'unit'}
    gate={'plan':pr,'output_root':str(cache.COT_NORM_ROOT),'execution':cache.COT_NORM_EXECUTION,'limits':cache.COT_NORM_COUNTS,
        'max_invocations':1,'claimed_invocations':0,'completed_invocations':0,'remaining_invocations':0,
        'relaunch_allowed':False,'training_authorized':False,'test_authorized':False,
        'status':'PREPARED_NOT_ADMITTED','execution_authorized':False}
    policy={key:False for key in cache.owner.data.CLOSED_FLAGS}
    policy[cache.COT_NORM_PARENT]={'goal_status':'ACTIVE','large_training_qualified':False,cache.COT_NORM_GATE:gate}
    monkeypatch.setattr(cache,'cot_read',lambda binding:plan if binding==pr else policy)
    monkeypatch.setattr(cache,'cot_json_ref',lambda path:policy_ref)
    monkeypatch.setattr(cache,'cot_norm_build_plan',lambda **kw:plan);monkeypatch.setattr(cache,'cot_source_clean',lambda p:True)
    checks=[];monkeypatch.setattr(cache,'cot_norm_checked_inputs',lambda p,**kw:checks.append(kw))
    assert cache.cot_norm_inspect('1'*64,admitted=False)[0]is plan and checks==[{'payloads':False}]
    policy['training_enabled']=True
    with pytest.raises(ValueError,match='CLOSED_POLICY'):cache.cot_norm_inspect('1'*64,admitted=False)
    plan['source_owners']['/unit/weights.pt']='2'*64
    monkeypatch.setattr(cache,'sha',lambda *a:pytest.fail('invalid inventory hash'))
    with pytest.raises(ValueError,match='SOURCE_SET'):cache.cot_norm_inspect('1'*64,admitted=False)
    assert not cache.COT_NORM_ROOT.exists()


def test_cot_normalization_consumed_failure_and_success_are_separate_one_use_roots(tmp_path,monkeypatch):
    root=tmp_path/'failed';monkeypatch.setattr(cache,'COT_NORM_ROOT',root)
    monkeypatch.setattr(cache.owner,'require_capped_cpu_research_producer_execution',lambda:{'unit':'exact_guard'})
    monkeypatch.setattr(cache,'cot_norm_timeout_ancestry',lambda sha:{'outer_seconds':900})
    def blocked(*a,**k):raise ValueError('preclaim gate')
    monkeypatch.setattr(cache,'cot_norm_inspect',blocked)
    with pytest.raises(ValueError,match='preclaim gate'):cache.cot_norm_run('1'*64)
    assert not root.exists()
    pr={'path':'/unit/plan.json','sha256':'1'*64};policy={'path':'/unit/policy.json','sha256':'2'*64}
    monkeypatch.setattr(cache,'cot_norm_inspect',lambda *a,**k:({},pr,policy));monkeypatch.setattr(cache,'cot_norm_bookend',lambda *a:True)
    def failed(*a,**k):raise ValueError('synthetic statistics failure')
    monkeypatch.setattr(cache,'cot_norm_produce',failed)
    with pytest.raises(ValueError,match='statistics failure'):cache.cot_norm_run('1'*64)
    terminal=json.loads((root/'TERMINAL.json').read_text())
    assert terminal['status']=='FAILED_CONSUMED'and terminal['result']is None
    assert terminal['source_unchanged']is True and terminal['test_data_used']is False
    assert terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False
    assert cache.signal.getitimer(cache.signal.ITIMER_REAL)==(0.,0.)
    monkeypatch.setattr(cache,'COT_NORM_ROOT',tmp_path/'success')
    monkeypatch.setattr(cache,'cot_norm_produce',lambda p:{'actual_counts':dict(cache.COT_NORM_COUNTS)})
    monkeypatch.setattr(cache,'cot_norm_checked_inputs',lambda *a,**k:None)
    refs=cache.cot_norm_run('1'*64);terminal=json.loads(Path(refs['terminal']['path']).read_text())
    assert terminal['status']=='COMPLETE'and terminal['result']==refs['result']
    assert terminal['training_authorized']is terminal['test_data_used']is False


def test_cot_normalization_fixed_cli_and_immutable_readback(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(cache,'cot_norm_inspect',lambda *a,**kw:(calls.append((a,kw))or({}, {}, {})))
    cache.cot_norm_main(['--cot-normalization','--plan-sha256','1'*64,'--preflight'])
    assert calls==[(('1'*64,),{'admitted':False})]
    with pytest.raises(ValueError,match='EXACT_CLI'):
        cache.cot_norm_main(['--cot-normalization','--plan','/wrong','--plan-sha256','1'*64])
    monkeypatch.setattr(cache,'COT_NORM_ROOT',tmp_path)
    binding=cache.cot_norm_publish('NORMALIZATION.json',{'qualified':'once'})
    assert cache.cot_json_ref(binding['path'])==binding and cache.cot_read(binding)=={'qualified':'once'}
    with pytest.raises(ImmutableEventAuthorityError):cache.cot_norm_publish('NORMALIZATION.json',{'qualified':'twice'})
    assert cache.COT_NORM_COUNTS['normalization_fits']==cache.COT_NORM_COUNTS['snapshot_loads']==1
    assert cache.COT_NORM_COUNTS['prior_fits']==cache.COT_NORM_COUNTS['target_values_decoded']==0


@pytest.mark.parametrize('fault',[None,'wrong_mode','wrong_sha','missing'])
def test_cot_normalization_requires_its_exact_outer_timeout(fault,monkeypatch):
    argv=['timeout','--kill-after=10s','900s','.venv/bin/python','-m',cache.COT_MODULE,
        '--cot-normalization','--plan-sha256','1'*64]
    if fault=='wrong_mode':argv[6]='--cot-publication-correction'
    if fault=='wrong_sha':argv[-1]='2'*64
    if fault=='missing':argv=['bash']
    monkeypatch.setattr(cache.os,'getppid',lambda:123)
    monkeypatch.setattr(cache.Path,'read_bytes',lambda self:('\0'.join(argv)+'\0').encode())
    monkeypatch.setattr(cache.Path,'read_text',lambda self,*a,**k:'PPid:\t0\n')
    if fault:
        with pytest.raises(ValueError,match='COT_NORM_.*TIMEOUT'):cache.cot_norm_timeout_ancestry('1'*64)
    else:assert cache.cot_norm_timeout_ancestry('1'*64)['outer_seconds']==900
