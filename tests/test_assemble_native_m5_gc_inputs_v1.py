"""Synthetic native component assembly; never replay genuine raw records."""
import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.audit import assemble_native_m5_gc_inputs_v1 as owner


def fixture(tmp_path):
    from gx1.contracts import gc_continuous_source_v1 as source
    from gx1.features import gc_order_flow_v1 as formulas
    begin=1714521600000000000;day=86400000000000;interval=300000000000
    clock={'schema_version':source.NATIVE_M5_REDUCTION_CLOCK_SCHEMA,
        'start_ns':begin,'end_ns':begin+4*day,'bar_duration_ns':interval}
    binding={'source_clock':clock,'contract_metadata':{'GCM4':[2024,6,100000000],'GCQ4':[2024,8,100000000]}}
    receipts=[]
    for ci in range(2):
        for first_day in (0,2):
            n=576;start=begin+first_day*day
            col=formulas.L1FeatureCollectorV2(np.arange(n),100000000,origin_ns=start,
                include_order_counts=True,include_event_diagnostics=True,
                include_trusted_trade_stats=True,input_interval_ns=interval)
            # Day1 selects GCM4; day2 rolls to GCQ4 and cannot roll backwards.
            daily=((10,5,100,100)if ci==0 else(9,20,15,15))
            volume=np.repeat(np.asarray(daily[first_day:first_day+2],np.uint64),288)
            metrics={name:np.zeros(n,np.uint64)for name in source.METRICS}
            metrics['records'][:]=metrics['trades'][:]=1;metrics['volume'][:]=volume
            metrics['buy'if ci==0 else'sell'][:]=volume
            if ci:
                unknown=np.arange(n)%10==0
                metrics['sell'][unknown]=0;metrics['unknown'][unknown]=volume[unknown]
            metrics['duplicate_trade'][3]=1
            quality={name:np.zeros(n,np.uint64)for name in source.L1_QUALITY_METRICS}
            quality['records'][:]=quality['clean_complete_bbo'][:]=1
            quality['file_seam'][0]=1;quality['duplicate_trade_signature'][3]=1
            col.covered[:]=col.order_covered[:]=interval
            col.sums[:,:6]=1;col.sums[:,6:11]=interval
            col.order_sums[:]=2*interval;col.first[:]=1000;col.last[:]=1001
            col.first_time[:]=start+np.arange(n)*interval
            col.last_time[:]=col.first_time+interval-1
            for name,value in col.event_stats.items():
                value[:]=1
            stats={name:np.zeros(n,np.uint64)for name in formulas.TRUSTED_TRADE_SIZE_FIELDS_V1}
            for side in ('buy','sell','unknown'):
                stats[side+'_trade_records'][:]=metrics[side]>0
            for name in ('size_sum_contracts','size_min_contracts','size_max_contracts'):
                stats[name][:]=volume
            stats['size_sum_squared_contracts2'][:]=volume*volume
            stats['buy_size_max_contracts'][:]=np.where(metrics['buy']>0,volume,0)
            stats['sell_size_max_contracts'][:]=np.where(metrics['sell']>0,volume,0)
            starts=start+np.arange(n,dtype=np.int64)*interval
            arrays={'bar_start_ns':starts,'ohlc_tick':np.tile(np.array([1000,1001,999,1000]),(n,1)),
                'first_trade_recv_ns':starts+1,'last_trade_recv_ns':starts+1,
                'book_sums':col.sums,'book_covered_ns':col.covered,'book_first_mid_tick':col.first,
                'book_last_mid_tick':col.last,'book_first_recv_ns':col.first_time,'book_last_recv_ns':col.last_time,
                'order_sums':col.order_sums,'order_covered_ns':col.order_covered}
            for prefix,group in (('metric',metrics),('quality',quality),('event',col.event_stats),('trade_size',stats)):
                arrays.update({prefix+'_'+key:value for key,value in group.items()})
            path=tmp_path/('COMPONENTS_'+str(ci)+'_'+str(first_day)+'.npz');np.savez(path,**arrays)
            ladder=path.with_suffix('.parquet')
            pq.write_table(pa.table({'bucket_index':np.arange(n,dtype=np.int32),
                'price_tick':np.full(n,1000,np.int64),**{name:metrics[name]for name in('buy','sell','unknown')}}),ladder)
            receipts.append({'contract_index':ci,'global_bucket_offset':first_day*288,
                'source_clock':{**clock,'start_ns':start,'end_ns':start+2*day},
                'components':owner.ref(path),'ladder':owner.ref(ladder),
                'arrays':{key:{'dtype':value.dtype.str,'shape':list(value.shape),
                    'sha256':hashlib.sha256(value.tobytes()).hexdigest()}for key,value in arrays.items()}})
    incidents={'schema_version':'gx1_gc_vendor_source_incidents_v1','dataset':'GLBX.MDP3',
        'nontrade_action_semantics_admitted':False,'incidents':[{'incident_id':'synthetic',
        'source_url':'synthetic://source','start_inclusive_utc':'2024-05-03T00:04:00Z',
        'end_exclusive_utc':'2024-05-03T00:06:00Z'}]}
    return binding,receipts,incidents


def test_previous_complete_day_roll_and_native_quality_incidents_keep_unknowns(tmp_path):
    binding,receipts,incidents=fixture(tmp_path)
    selected,daily,covered,chosen,basis,evidence=owner.daily_roll(binding,receipts,incidents)
    assert daily.dtype==np.uint64 and covered.all()
    assert chosen.tolist()==[-1,0,1,1]
    assert basis.tolist()==[-1,0,1,2]
    state=owner.selected_components(binding,receipts,selected,incidents)
    assert not state['filled'][:288].any()
    assert not state['masks']['volume'][:288].any()
    assert state['masks']['volume'][291]  # identical T signature is not bad-source proof
    assert state['metrics']['unknown'].sum()>0
    assert not state['masks']['delta'][586]
    assert not state['masks']['volume'][576:578].any()
    assert state['masks']['volume'][578]
    assert not state['book']['ofi_valid'][576:579].any()
    assert state['book']['ofi_valid'][579]
    assert state['masks']['volume'].sum()==861
    assert state['book']['bbo_valid'][579]
    assert state['book']['order_count_valid'][579]
    from gx1.features.gc_order_flow_v1 import L1_ORDER_COUNT_FIELDS_V1
    assert state['book'][L1_ORDER_COUNT_FIELDS_V1[0]][579]==2.


def test_selected_ladder_offsets_order_and_three_side_integer_conservation(tmp_path):
    binding,receipts,incidents=fixture(tmp_path)
    selected,*_=owner.daily_roll(binding,receipts,incidents)
    state=owner.selected_components(binding,receipts,selected,incidents)
    path=tmp_path/'SELECTED.parquet'
    proof=owner.selected_ladder(binding,list(reversed(receipts)),state,path)
    table=pq.read_table(path)
    ids=table['bucket_index'].to_numpy()
    assert len(ids)==861 and np.all(np.diff(ids)>=0) and ids[0]==288
    assert proof['per_side_integer_conservation']is True
    assert proof['side_volume_contracts']['unknown']>0
    for name in ('buy','sell','unknown'):
        assert table[name].to_numpy().sum()==state['metrics'][name][state['masks']['volume']].sum()


@pytest.mark.parametrize('fault',('duplicate','bytes','trade_clock'))
def test_component_coverage_bytes_and_real_trade_clock_cannot_be_relabelled(tmp_path,fault):
    binding,receipts,incidents=fixture(tmp_path)
    selected,*_=owner.daily_roll(binding,receipts,incidents)
    if fault=='duplicate':
        receipts.append(receipts[0])
    else:
        receipt=receipts[0]
        with np.load(receipt['components']['path'],allow_pickle=False)as archive:
            arrays={key:archive[key]for key in archive.files}
        key='metric_volume'if fault=='bytes'else'first_trade_recv_ns'
        arrays[key][300]+=300000000000
        if fault=='trade_clock':
            receipt['arrays'][key]['sha256']=hashlib.sha256(arrays[key].tobytes()).hexdigest()
        np.savez(receipt['components']['path'],**arrays)
    with pytest.raises(ValueError,match='(DUPLICATE_SELECTED_COVERAGE|COMPONENT_ARRAY_BYTES|TRUSTED_T_RECV_CLOCK)'):
        owner.selected_components(binding,receipts,selected,incidents)


def test_numeric_packet_roundtrip_keeps_null_clock_storage_and_metadata_only(tmp_path,monkeypatch):
    monkeypatch.setattr(owner,'ROOT',tmp_path)
    packet={'schema_version':'synthetic','source_clock':{'bar_duration_ns':300000000000},
        'timeframes':{'M5':{'available_ns':np.array([10,20],np.int64),
            'source_available_ns':np.array([-1,10],np.int64),'known':np.array([False,True]),
            'value':np.array([np.nan,1.],np.float32)}},
        'm5_book_features':{'available_ns':np.array([10,20],np.int64)}}
    refs=owner.publish_arrays('PACKET',packet,packet=True)
    metadata=owner.read(refs['json'])
    assert 'timeframes'not in metadata['packet_metadata']
    assert 'm5_book_features'not in metadata['packet_metadata']
    with np.load(refs['npz']['path'],allow_pickle=False)as archive:
        assert archive['timeframes__M5__source_available_ns'].tolist()==[-1,10]
        assert archive['timeframes__M5__known'].tolist()==[False,True]
    assert metadata['all_array_dtype_shape_equal_nan_sha256_verified']is True


def test_preparer_refuses_unfinished_raw_stage_before_any_component_decode(monkeypatch):
    inputs={key:{'path':'/synthetic/'+key,'sha256':'a'*64}for key in owner.INPUT_KEYS}
    plan_ref={'path':'/synthetic/PLAN','sha256':'b'*64}
    claim_ref={'path':'/synthetic/CLAIM','sha256':'c'*64}
    records={inputs['raw_components_result']['path']:{'stage':owner.raw.STAGE,
        'status':'FAILED_CONSUMED','plan':plan_ref,'claim':claim_ref},
        inputs['raw_components_terminal']['path']:{'stage':owner.raw.STAGE,'status':'FAILED_CONSUMED'},
        plan_ref['path']:{'stage':owner.raw.STAGE},claim_ref['path']:{'stage':owner.raw.STAGE}}
    monkeypatch.setattr(owner,'read',lambda ref:records[ref['path']])
    monkeypatch.setattr(owner,'component_arrays',lambda *args:pytest.fail('component arrays read before genuine raw completion'))
    with pytest.raises(ValueError,match='RAW_COMPONENTS_COMPLETE_CHAIN'):
        owner.build_plan(inputs=inputs)


def test_full_native_packet_consumes_selected_components_without_raw_replay(tmp_path,monkeypatch):
    import pandas as pd
    from gx1.contracts import gc_continuous_source_v1 as source
    from gx1.features import gc_native_m5_features_v1 as features
    binding,receipts,incidents=fixture(tmp_path)
    selected,*_=owner.daily_roll(binding,receipts,incidents)
    state=owner.selected_components(binding,receipts,selected,incidents)
    path=tmp_path/'SELECTED.parquet'
    owner.selected_ladder(binding,receipts,state,path)
    monkeypatch.setattr(source,'scan_symbol',lambda *a,**k:pytest.fail('raw replay reached'))
    native_xau=pd.DataFrame({'bid_close':np.full(len(selected),99.99),
                            'ask_close':np.full(len(selected),100.01)},
                           index=pd.to_datetime(state['starts'],utc=True))
    packet=features.build_native_m5_gc_surfaces(scope=binding['source_clock'],selected=selected,
        metrics=state['metrics'],masks=state['masks'],ohlc=state['ohlc'],ladder_path=path,
        book=state['book'],events=state['events'],trade_stats=state['trade_stats'],
        tick_size=state['tick'],native_xau_m5=native_xau,
        first_trusted_trade_recv_ns=state['first_recv'],last_trusted_trade_recv_ns=state['last_recv'])
    coverage=features.source_coverage_report(packet)
    from gx1.contracts.market_state_native_m5_gc_feature_inputs_v1 import _packet_layout
    _,expected_layout,_=_packet_layout(packet)
    monkeypatch.setattr(owner,'ROOT',tmp_path)
    refs=owner.publish_arrays('ACTUAL_NUMERIC_PACKET',packet,packet=True)
    metadata=owner.read(refs['json'])
    assert metadata['array_layout']==expected_layout
    assert metadata['packet_metadata']=={key:value for key,value in packet.items()
        if key not in ('timeframes','m5_book_features')}
    assert coverage['all_declared_source_columns_present']is True
    assert len(coverage['fields'])==len(features.native_m5_field_catalog()['fields'])
    assert not coverage['source_boolean_fields_without_binary_declaration']
    assert not packet['timeframes']['M5']['volume_valid'][:288].any()
    assert packet['native_training_admitted']is False
    for tf in ('H4','D1'):
        table=packet['timeframes'][tf]
        at=np.flatnonzero(table['bar_start_ns']==pd.Timestamp('2024-05-02T22:00:00Z').value)
        assert len(at)==1 and not table['observed_contract_valid'][at[0]]
        assert not table['volume_valid'][at[0]]


def lifecycle(monkeypatch,tmp_path):
    monkeypatch.setattr(owner,'ROOT',tmp_path/'OUT')
    plan={'execution':dict(owner.EXECUTION)}
    plan_ref={'path':'/synthetic/PLAN','sha256':'a'*64}
    policy_ref={'path':'/synthetic/POLICY','sha256':'b'*64}
    monkeypatch.setattr(owner,'inspect',lambda *a,**k:(plan,plan_ref,policy_ref))
    monkeypatch.setattr(owner,'checked_inputs',lambda *a,**k:None)
    monkeypatch.setattr(owner,'source_clean',lambda plan:True)
    monkeypatch.setattr(owner,'ref',lambda path:policy_ref if path==owner.POLICY else plan_ref)
    monkeypatch.setattr(owner.raw,'firewall_state',lambda:{'denied_attempts':0})
    monkeypatch.setattr(owner.raw,'require_capped_cpu_research_producer_execution',lambda:{
        'memory_max':12*1024**3,'memory_high':12*1024**3,'swap':512*1024**2,
        'numerical_threads':1,'cgroup_path':'/sys/fs/cgroup/synthetic'})
    monkeypatch.setattr(owner.raw,'require_capped_lock_ancestry',lambda:{'scope':'/synthetic'})


def test_one_use_assembly_success_keeps_training_closed(monkeypatch,tmp_path):
    lifecycle(monkeypatch,tmp_path)
    monkeypatch.setattr(owner,'assemble',lambda *args:{'numeric_packet_only':True})
    result=owner.run('a'*64)
    saved=json.loads((owner.ROOT/'RESULT.json').read_text())
    terminal=json.loads((owner.ROOT/'TERMINAL.json').read_text())
    assert terminal['result']==result['result']and terminal['status']=='COMPLETE'
    assert saved['raw_replay']is False and saved['model_forwards']==saved['optimizer_steps']==0
    assert saved['training_authorized']is saved['test_data_used']is False
    assert terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False
    with pytest.raises(FileExistsError):owner.run('a'*64)


def test_failed_assembly_keeps_durable_intermediate_and_consumes_claim(monkeypatch,tmp_path):
    lifecycle(monkeypatch,tmp_path)
    def failed(*args):
        owner.publish(owner.ROOT/'ASSEMBLY_STATE.json',{'complete_selected_components':True})
        raise RuntimeError('synthetic-downstream-packet-failure')
    monkeypatch.setattr(owner,'assemble',failed)
    with pytest.raises(RuntimeError,match='synthetic-downstream'):
        owner.run('a'*64)
    assert (owner.ROOT/'ASSEMBLY_STATE.json').exists()
    terminal=json.loads((owner.ROOT/'TERMINAL.json').read_text())
    assert terminal['status']=='FAILED_CONSUMED'and terminal['result']is None
    assert terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False


def test_incident_clock_volume_cannot_choose_next_day_contract(tmp_path):
    from gx1.contracts.gc_native_m5_source_inputs_v1 import selected_native_m5_contracts
    binding,receipts,_=fixture(tmp_path)
    receipt=receipts[0]
    with np.load(receipt['components']['path'],allow_pickle=False)as archive:
        arrays={key:archive[key]for key in archive.files}
    volume=arrays['metric_volume']
    volume[:288]=1;volume[0]=2000000
    arrays['metric_buy'][:]=volume
    for name in ('size_sum_contracts','size_min_contracts','size_max_contracts','buy_size_max_contracts'):
        arrays['trade_size_'+name][:]=volume
    arrays['trade_size_size_sum_squared_contracts2'][:]=volume*volume
    np.savez(receipt['components']['path'],**arrays)
    receipt['components']=owner.ref(receipt['components']['path'])
    for key,value in arrays.items():
        receipt['arrays'][key]['sha256']=hashlib.sha256(value.tobytes()).hexdigest()
    incidents={'schema_version':'gx1_gc_vendor_source_incidents_v1','dataset':'GLBX.MDP3',
        'nontrade_action_semantics_admitted':False,'incidents':[{'incident_id':'synthetic-capture-clock',
        'source_url':'synthetic://source','start_inclusive_utc':'2024-05-01T00:00:00Z',
        'end_exclusive_utc':'2024-05-01T00:05:00Z'}]}
    _,daily,_,chosen,_,proof=owner.daily_roll(binding,receipts,incidents)
    maturities=[tuple(value[:2])for value in binding['contract_metadata'].values()]
    _,wrong_unfiltered,_=selected_native_m5_contracts(proof['daily_volume_before_incident'],
                                                    maturities,source_clock=binding['source_clock'])
    assert wrong_unfiltered[1]==0 and chosen[1]==1
    assert proof['daily_excluded_volume'][0,0]==2000000
    assert proof['daily_excluded_trusted_T_records'][:,0].tolist()==[1,1]
    assert proof['partial_observation_days_due_to_incident'][:,0].all()
    assert daily[0,0]==287 and daily[1,0]==9*287
    assert proof['full_feed_or_288_observed_bars_per_day_claimed']is False


def test_actual_synthetic_raw_checkpoint_metadata_matches_assembly_domain(monkeypatch,tmp_path):
    import copy
    from tests.test_build_native_m5_gc_raw_components_v1 import raw_plan
    from gx1.audit import build_native_m5_gc_raw_components_v1 as producer
    plan=raw_plan(monkeypatch,tmp_path)
    authority={'path':'/synthetic/PRODUCER','sha256':'d'*64}
    receipt=producer.read(producer.reduce_file(plan,0,authority,authority))
    assert owner.file_metadata(receipt,0,plan['source_binding'],authority,authority)is receipt
    wrong=copy.deepcopy(receipt);wrong['global_bucket_offset']+=1
    with pytest.raises(ValueError,match='FILE_RECEIPT_IDENTITY'):
        owner.file_metadata(wrong,0,plan['source_binding'],authority,authority)
    wrong=copy.deepcopy(receipt);wrong['arrays']['book_sums']['shape'][1]+=1
    with pytest.raises(ValueError,match='COMPONENT_METADATA_HEADERS'):
        owner.file_metadata(wrong,0,plan['source_binding'],authority,authority)


def test_other_bad_record_does_not_erase_trusted_T_from_daily_roll(tmp_path):
    binding,receipts,incidents=fixture(tmp_path)
    receipt=receipts[0]
    with np.load(receipt['components']['path'],allow_pickle=False)as archive:
        arrays={key:archive[key]for key in archive.files}
    at=288
    arrays['metric_volume'][at]=arrays['metric_buy'][at]=2000000
    arrays['metric_invalid_trade'][at]=1
    arrays['metric_records'][at]=arrays['quality_records'][at]=2
    for name in ('size_sum_contracts','size_min_contracts','size_max_contracts','buy_size_max_contracts'):
        arrays['trade_size_'+name][at]=2000000
    arrays['trade_size_size_sum_squared_contracts2'][at]=2000000**2
    np.savez(receipt['components']['path'],**arrays)
    receipt['components']=owner.ref(receipt['components']['path'])
    for key,value in arrays.items():
        receipt['arrays'][key]['sha256']=hashlib.sha256(value.tobytes()).hexdigest()
    selected,daily,_,chosen,_,proof=owner.daily_roll(binding,receipts,incidents)
    state=owner.selected_components(binding,receipts,selected,incidents)
    assert chosen[2]==0 and daily[0,1]==2000000+287*5
    assert state['metrics']['volume'][at]==2000000
    assert not state['masks']['volume'][at]
    assert proof['bad_T_or_book_gap_does_not_erase_other_trusted_T_from_roll']is True
    assert proof['receive_clock_incident_excludes_entire_native_bucket_from_roll']is True
