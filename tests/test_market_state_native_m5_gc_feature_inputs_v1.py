"""Synthetic real-provider joins, immutable source admission and joint fitting."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts import market_state_native_m5_gc_feature_inputs_v1 as owner
from gx1.contracts import market_state_native_m5_feature_inputs_v1 as ta
from gx1.features import gc_native_m5_features_v1 as gc
from tests.test_market_state_native_m5_feature_inputs_v1 import _fixture,_build
from tests.test_gc_native_m5_features_v1 import _synthetic_native_packet,ORIGIN,INTERVAL


def _write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False))
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def _source_binding(packet,tmp_path,xau_binding):
    """Tiny synthetic receipts exercise the actual public admission API."""
    raw_root=tmp_path/'GC_NATIVE_M5_RAW_COMPONENTS_UNIT';clock=packet['source_clock']
    raw={'schema_version':'gx1_gc_native_m5_raw_source_binding_v1','source_clock':clock,
        'raw_files':[{'path':'/synthetic/original_raw_'+str(i)+'.gz','sha256':str(i%10)*64,
                      'start_ns':clock['start_ns'],'end_ns':clock['end_ns']} for i in range(66)],
        'm1_data_or_metadata_used':False,'derived_feature_parents':[]}
    raw['source_identity_sha256']=owner._sha(raw)
    plan=_write(tmp_path/'GC_NATIVE_M5_RAW_COMPONENTS_UNIT.json',{
        'schema_version':'gx1_gc_native_m5_raw_components_plan_v1','source_binding':raw,'test_data_used':False})
    claim=_write(raw_root/'CLAIM.json',{'status':'CLAIMED_ONE_USE','plan':plan,'test_data_used':False})
    receipts=[]
    for i,item in enumerate(raw['raw_files']):
        receipts.append(_write(raw_root/'FILES'/('FILE_'+str(i).zfill(3))/'RECEIPT.json',{
            'schema_version':'gx1_gc_native_m5_raw_file_components_v1',
            'status':'COMPLETE_RAW_COMPONENTS_NOT_FEATURE_ADMISSION','ordinal':i,'source':item,
            'source_identity_sha256':raw['source_identity_sha256'],'source_clock':clock,'global_bucket_offset':0,
            'producer_plan':plan,'producer_claim':claim,'raw_hash_bookends_unchanged':True,
            'ladder_and_trade_size_integer_conservation':True}))
    result=_write(raw_root/'RESULT.json',{'status':'COMPLETE_RAW_COMPONENTS_ONLY','source_unchanged':True,
        'all_raw_payload_hash_bookends_unchanged':True,'native_input_interval_ns':INTERVAL,
        'test_data_used':False,'model_forwards':0,'optimizer_steps':0,'plan':plan,'claim':claim,'files':receipts})
    terminal=_write(raw_root/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'result':result,
        'plan':plan,'claim':claim,'test_data_used':False})
    arrays,layout,_=owner._packet_layout(packet)
    archive=tmp_path/'FEATURES.npz';np.savez_compressed(archive,**arrays)
    refs={'npz':{'path':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},
        'json':_write(tmp_path/'FEATURES.json',{'schema_version':owner.PACKET_SCHEMA,
            'packet_metadata':{k:v for k,v in packet.items() if k not in ('timeframes','m5_book_features')},
            'array_layout':layout,'nullable_timestamp_columns':list(gc.NATIVE_NULLABLE_TIMESTAMP_COLUMNS)})}
    catalog=gc.native_m5_field_catalog()
    proof=_write(tmp_path/'ASSEMBLY_PROOF.json',{'schema_version':owner.PROOF_SCHEMA,'status':'COMPLETE',
        'source_identity_sha256':raw['source_identity_sha256'],'source_clock':clock,
        'native_xau_m5_source_binding_sha256':xau_binding['contract_sha256'],
        'raw_components_plan':plan,'raw_components_claim':claim,'raw_components_result':result,
        'raw_components_terminal':terminal,'raw_components':receipts,'m1_data_or_metadata_used':False,
        'raw_replay':False,'packet_artifacts':refs,'catalog_contract_sha256':catalog['contract_sha256'],
        'unit_policy_sha256':catalog['unit_policy_sha256'],'source_unchanged':True,'test_data_used':False})
    return owner.bind_native_m5_gc_feature_packet(packet,raw_source_binding=raw,
        native_xau_m5_source_binding=xau_binding,assembly_proof_ref=proof,artifact_refs=refs)


def _joint(tmp_path,monkeypatch,*,missing_hour=False):
    fixture=_fixture(tmp_path,monkeypatch);raw,source,_,_,frames=fixture
    for i,(tf,frame) in enumerate(frames.items()):
        if tf!='M5':
            width=int(ta.htf.MULTI_TF_SHIFT[tf].value)
            last=ta.htf.multi_tf_bar_label(pd.DatetimeIndex([pd.Timestamp(source['source_spec']['cutoff_ns']-width-1,tz='UTC')]),tf).asi8[0]
            labels=last-np.arange(len(frame)-1,-1,-1,dtype=np.int64)*width
            frame.index=pd.to_datetime(labels,utc=True);frame.attrs['ts_int64']=labels.copy()
        frame.attrs['owner_row_indices']=10_000*(i+1)+np.arange(len(frame),dtype=np.int64)
    if missing_hour:
        frame=frames['H1'];missing=pd.Timestamp('2024-05-01T12:00Z').value
        keep=np.flatnonzero(frame.index.asi8!=missing)
        arrays={key:(value[keep].copy() if isinstance(value,np.ndarray) else value) for key,value in frame.attrs.items()}
        # This synthetic source assigns observed-bar IDs after its clock is
        # defined, as the actual owner does; D1 IDs are untouched.
        arrays['owner_row_indices']=30_000+np.arange(len(keep),dtype=np.int64)
        replacement=pd.DataFrame(arrays['feats_np'],index=frame.index[keep],columns=frame.columns,copy=False)
        replacement.attrs=arrays;frames['H1']=replacement
    base=ta.NativeM5FeatureProvider(_build(fixture))
    (tmp_path/'raw_unit').mkdir()
    packet,_,_=_synthetic_native_packet(tmp_path/'raw_unit',monkeypatch)
    if missing_hour:
        # Distinct synthetic source-column markers prove which closed GC bar
        # is read; this is a wiring fixture, not a raw-market truth claim.
        table=packet['timeframes']['H1']
        for hour,value in ((12,44.),(13,99.),(14,777.)):
            at=np.flatnonzero(table['available_ns']==pd.Timestamp('2024-05-01T'+str(hour)+':00Z').value)
            table['observed_volume'][at]=value
    # Same original native manifest, independent larger causal prefix binding.
    rows=576;index=pd.date_range('2024-05-01',periods=rows,freq='5min',tz='UTC')
    xau=pd.DataFrame(index=index)
    for prefix,value in (('',101.),('bid_',100.),('ask_',102.)):
        for name in ('open','high','low','close'):xau[prefix+name]=value
    xau['volume']=1.;xau=xau.loc[:,list(ta.RAW_COLUMNS)]
    spec=copy.deepcopy(source['source_spec']);spec['cutoff_ns']=int(index[-1].value)+2*INTERVAL
    xau_binding=ta.bind_native_m5_feature_source(xau,native_m5_manifest=source['native_m5_manifest'],source_spec=spec)
    binding=_source_binding(packet,tmp_path/'source_unit',xau_binding)
    return owner.NativeM5GCFeatureProvider(base,packet,binding),packet,binding,xau_binding


def test_actual_source_binding_roundtrip_and_joint_gather_preserve_all_global_axes(tmp_path,monkeypatch):
    provider,packet,binding,_=_joint(tmp_path,monkeypatch)
    loaded=owner.load_native_m5_gc_feature_packet(binding)
    assert owner._packet_layout(loaded)[1]==owner._packet_layout(packet)[1]
    rows=np.array([150,200,399],np.int64);batch=provider.gather(rows,max_rows=3)
    np.testing.assert_array_equal(provider.source_row,provider.ta.source_row)
    np.testing.assert_array_equal(provider.input_ready,provider.ta.input_ready)
    assert batch['seq_x'].shape==(3,96,2748) and batch['ctx_cont'].shape==(3,2562)
    assert len(batch)==14
    np.testing.assert_array_equal(batch['seq_x'][:,-1],batch['snap_x'])
    np.testing.assert_array_equal(batch['snap_x'][:,-2430:],batch['ctx_cont'][:,-2430:])
    assert [provider.gc_surfaces[tf]['features'].shape[1] for tf in owner.TIMEFRAMES]==[643,497,457,435,403]
    for tf in owner.TIMEFRAMES:
        np.testing.assert_array_equal(provider.gc_surfaces[tf]['native_row_indices'],provider.ta.tf_source_rows[tf])
    d1=provider.gc_surfaces['D1'];assert d1['native_row_indices'][0]>10_000
    before=d1['available_ns']<packet['source_clock']['start_ns']
    assert before.any() and np.count_nonzero(d1['features'][before])==0
    assert provider.support_report(rows)['unknown_gc_rows_removed']==0


def test_one_packet_column_diagnostic_regathers_views_without_new_genuine_binding(tmp_path,monkeypatch):
    provider,packet,binding,_=_joint(tmp_path,monkeypatch)
    fit_rows=np.arange(150,390,dtype=np.int64)
    normalization=owner.fit_native_m5_gc_feature_normalization(provider,fit_source_rows=fit_rows,
        cutoff_ns=int(provider.issue_ns[399])+INTERVAL)
    row=200;rows=np.array([row],np.int64)
    original_layout=owner._packet_layout(packet)[1]
    original_proof=Path(binding['assembly_proof']['path']).read_bytes()
    original_norm=owner._sha(normalization)
    files=sorted(str(p) for p in tmp_path.rglob('*') if p.is_file())
    result=owner.diagnostic_regather_native_m5_gc_primitive(provider,source_packet=packet,
        input_normalization=normalization,field_name='gc_m5_observed_volume',
        source_available_ns=int(packet['timeframes']['M5']['available_ns'][row]),delta=1.,
        source_rows=rows,max_rows=1)
    before,after,receipt=result['before'],result['after'],result['diagnostic']
    field='gc_m5_observed_volume';local=provider.gc_context_names.index(field)
    side=provider.layout['field_names']['M5'].index(field)
    signal=318+local;context=132+local
    assert after['seq_x'][0,-1,signal]-before['seq_x'][0,-1,signal]==1
    assert after['snap_x'][0,signal]-before['snap_x'][0,signal]==1
    assert after['ctx_cont'][0,context]-before['ctx_cont'][0,context]==1
    assert after['gc_seq_m5'][0,-1,side]-before['gc_seq_m5'][0,-1,side]==1
    for tf in ('M15','H1','H4','D1'):
        np.testing.assert_array_equal(after['gc_seq_'+tf.lower()],before['gc_seq_'+tf.lower()])
    assert set(result)=={'before','after','diagnostic'}
    assert receipt['purpose']=='DIAGNOSTIC_INPUT_PRIMITIVE_COUNTERFACTUAL'
    assert receipt['source_admission'] is receipt['production_binding_created'] is False
    assert receipt['market_path_reconstructed'] is receipt['sibling_indicator_formulas_recomputed'] is False
    assert receipt['original_source_binding_sha256']==binding['contract_sha256']
    assert receipt['original_source_arrays_unchanged'] and receipt['original_proof_bytes_unchanged']
    assert receipt['original_normalization_bytes_unchanged']
    assert receipt['changed_coordinate_counts']['seq_x']==1
    assert receipt['changed_coordinate_counts']['gc_seq_m5']==1
    assert owner._packet_layout(packet)[1]==original_layout
    assert Path(binding['assembly_proof']['path']).read_bytes()==original_proof
    assert owner._sha(normalization)==original_norm
    assert sorted(str(p) for p in tmp_path.rglob('*') if p.is_file())==files


def test_current_gc_hour_and_ta_history_clock_remain_distinct_across_missing_ta_hour(tmp_path,monkeypatch):
    provider,_,_,_=_joint(tmp_path,monkeypatch,missing_hour=True)
    issue=pd.Timestamp('2024-05-01T13:05Z').value
    row=int(np.flatnonzero(provider.issue_ns==issue)[0])
    provider.gather(np.array([row],np.int64),max_rows=1)
    at=provider.ta._feature_positions(np.array([row],np.int64))[0]
    local_clock=provider.local_gc_bucket_available_ns['H1'][at]
    own=provider.gc_surfaces['H1']['available_ns'][provider.ta.tf_tails['H1'][row]]
    assert local_clock==pd.Timestamp('2024-05-01T13:00Z').value
    assert own==pd.Timestamp('2024-05-01T12:00Z').value
    assert local_clock<=issue and own<=issue and provider._local_tf_join_positions['H1'][at]==-1
    batch=provider.gather(np.array([row],np.int64),max_rows=1)
    name='gc_h1_observed_volume'
    assert batch['ctx_cont'][0,132+provider.gc_context_names.index(name)]==99.
    assert batch['gc_seq_h1'][0,-1,provider.layout['field_names']['H1'].index(name)]==44.


def test_joint_real_statistics_validate_source_units_masks_aliases_and_fold_union(tmp_path,monkeypatch):
    provider,_,_,_=_joint(tmp_path,monkeypatch)
    rows=np.arange(150,390,dtype=np.int64)
    normalization=owner.fit_native_m5_gc_feature_normalization(provider,fit_source_rows=rows,
        cutoff_ns=int(provider.issue_ns[399])+INTERVAL)
    binding=provider.input_binding(normalization)
    assert binding['source_axes_sha256']==provider.source_axes_sha256
    assert binding['native_gc_source_binding_sha256']==provider.source_binding['contract_sha256']
    for tf in owner.TIMEFRAMES:
        actual=normalization['gc_mtf_history']['normalization']['provenance']['row_union'][tf]
        expected=normalization['fit_population']['per_tf'][tf]
        assert actual['native_row_indices_sha256']==expected['source_rows_sha256']
        assert actual['available_ns_sha256']==expected['available_ns_sha256']
    from gx1.contracts import entry_model_native_input_normalization_v1 as norm
    raw=provider.gather(rows[-2:],max_rows=2)
    normalized=norm.apply_surface_normalization(raw['snap_x'],normalization['surfaces']['signal'])
    names=normalization['surfaces']['signal']['field_names']
    for value,mask in norm.native_gc_masking_pairs(names):assert np.all(normalized[:,value][raw['snap_x'][:,mask]==0]==0)


def test_legacy_source_binding_rejected_before_nested_artifact_reads(monkeypatch):
    monkeypatch.setattr(Path,'open',lambda *a,**k:pytest.fail('opened legacy metadata'))
    with pytest.raises(RuntimeError,match='SOURCE_SCHEMA'):
        owner.require_native_m5_gc_source_binding({'schema_version':'legacy','packet_artifacts':{'path':'/XAU_M1/old.json'}})


@pytest.mark.parametrize('field,index,delta,error', [
    ('gc_book_ofi_valid',200,1.,'DIAGNOSTIC_NONMARKET_FIELD'),
    ('gc_m5_observed_sell_max_executed_trade_contracts',0,1.,'DIAGNOSTIC_UNOBSERVED_SOURCE'),
    ('gc_m5_observed_known_aggressor_fraction',200,1.,'DIAGNOSTIC_SOURCE_BOUNDS'),
    ('gc_m5_observed_volume',500,1.,'DIAGNOSTIC_FUTURE_SOURCE'),
    ('gc_m5_observed_volume',200,2.,'DIAGNOSTIC_ONE_SOURCE_UNIT'),
])
def test_diagnostic_rejects_nonmarket_unknown_bounds_future_and_wrong_unit_before_proof_reads(
        tmp_path,monkeypatch,field,index,delta,error):
    provider,packet,_,_=_joint(tmp_path,monkeypatch)
    monkeypatch.setattr(owner,'_file_ref',lambda *a:pytest.fail('read a proof for inadmissible diagnostic'))
    with pytest.raises(RuntimeError,match=error):
        owner.diagnostic_regather_native_m5_gc_primitive(provider,source_packet=packet,
            input_normalization={},field_name=field,
            source_available_ns=int(packet['timeframes']['M5']['available_ns'][index]),delta=delta,
            source_rows=np.array([200],np.int64),max_rows=1)


def test_diagnostic_rejects_legacy_packet_before_any_nested_metadata_read(tmp_path,monkeypatch):
    provider,_,_,_=_joint(tmp_path,monkeypatch)
    monkeypatch.setattr(owner,'_file_ref',lambda *a:pytest.fail('followed legacy metadata'))
    with pytest.raises(RuntimeError,match='PACKET_IDENTITY'):
        owner.diagnostic_regather_native_m5_gc_primitive(provider,
            source_packet={'schema_version':'legacy','nested':{'path':'/XAU_M1/archive.json'}},
            input_normalization={},field_name='gc_m5_observed_volume',source_available_ns=ORIGIN+INTERVAL,
            delta=1.,source_rows=np.array([200],np.int64),max_rows=1)
