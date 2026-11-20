"""Small NumPy-only cache/lifecycle fixtures, never a production materialization."""
import copy
import io
import json
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
    assert cache.START_NS==int(pd.Timestamp('2018-10-01',tz='UTC').value)
    assert cache.END_NS==int(pd.Timestamp('2024-10-01',tz='UTC').value)
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


def test_consumed_001_observation_is_not_a_native_terminal_or_relaunch(monkeypatch):
    observation={'schema_version':'gx1_native_ta_cache_external_exit_observation_v1',
        'stage':'native_m5_ta_history_expansion_20261120_001','outer_exit_code':137,
        'native_RESULT_present':False,'native_TERMINAL_present':False,'cache_artifacts_present':False,
        'durable_artifacts_before_observation':['CLAIM.json'],'source_policy_plan_unchanged_at_external_bookend':True,
        'scope_directly_recorded_in_claim':False,'independent_cache_qualification_passed':False,
        'training_started':False,'test_data_used':False,'relaunch_allowed':False,
        'plan':{'path':'oldplan','sha256':'1'*64},'claim':{'path':'oldclaim','sha256':'2'*64},
        'failure_phase_unknown':'no progress receipt','attribution_limit':'original claim lacks cgroup/PID'}
    monkeypatch.setattr(cache,'read',lambda binding:copy.deepcopy(observation))
    proof=cache.previous_exit_metadata(cache.INPUT_REFS)
    assert proof['outer_exit_code']==137 and proof['native_TERMINAL_present']is False
    assert cache.STAGE.endswith('_002')and cache.ROOT.name.endswith('_002')
    observation['native_TERMINAL_present']=True
    with pytest.raises(ValueError,match='CONSUMED_001_EXIT_OBSERVATION'):cache.previous_exit_metadata(cache.INPUT_REFS)
    observation['native_TERMINAL_present']=False;observation['source_policy_plan_unchanged_at_external_bookend']=False
    with pytest.raises(ValueError,match='CONSUMED_001_EXIT_OBSERVATION'):cache.previous_exit_metadata(cache.INPUT_REFS)


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
        'feature_source_spec':{'manifest':{}},'expected_full_source_binding':{},
        'feature_contract':provider.surfaces['provider_contract'],'reference_snapshot':{},'reference_archive':{}}
    def event(name,value):
        events.append(name);return value
    monkeypatch.setattr(cache.owner,'parameter_artifact_metadata',lambda *args:(parameters,{}))
    monkeypatch.setattr(cache,'load_volatility_squeeze_artifact_manifest',lambda *a,**k:None)
    monkeypatch.setattr(cache,'phase',lambda name:event(name,{'phase':name}))
    monkeypatch.setattr(cache.owner,'read_feature_source',lambda *a:event('RAW_READ',object()))
    monkeypatch.setattr(features,'bind_native_m5_feature_source',lambda *a,**k:{})
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
