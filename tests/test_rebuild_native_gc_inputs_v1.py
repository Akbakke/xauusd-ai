"""Mechanics/algebra checks, not evidence of historical feature quality."""
import numpy as np

from gx1.contracts import gc_source_disposition_v1 as timing
from gx1.features.gc_order_flow_v1 import L1FeatureCollectorV2, L1_FIELDS_V2
from gx1.scripts.rebuild_native_gc_inputs_v1 import l1_masks, selected_minutes


def test_order_counts_do_not_change_existing_book_features():
    origin = timing.STUDY_START_NS
    events = np.array([[origin, 100, 102, 20, 30, 2, 3],
                       [origin + timing.MINUTE_NS // 2, 100, 102, 20, 30, 4, 6]], np.int64)
    original = L1FeatureCollectorV2(np.array([0]), 1)
    extended = L1FeatureCollectorV2(np.array([0]), 1, include_order_counts=True)
    original.consume(events[:, :5])
    extended.consume(events)
    original.finish(origin + timing.MINUTE_NS)
    extended.finish(origin + timing.MINUTE_NS)
    a, am = original.values([True])
    b, bm = extended.values([True])
    assert np.array_equal(am, bm)
    for key in L1_FIELDS_V2:
        np.testing.assert_array_equal(a[key], b[key])
    values, mask = extended.order_count_values([True])
    assert mask.tolist() == [True]
    assert values['tw_bid_orders'][0] == 3
    assert values['tw_ask_orders'][0] == 4.5
    assert values['tw_bid_contracts_per_order'][0] == 7.5
    assert values['tw_ask_contracts_per_order'][0] == 7.5
    assert values['tw_order_count_imbalance'][0] == -0.2


def test_unavailable_counts_preserve_valid_book_and_mask_counts():
    origin = timing.STUDY_START_NS
    collector = L1FeatureCollectorV2(np.array([0]), 1, include_order_counts=True)
    collector.consume(np.array([[origin, 100, 102, 20, 30, 0, 3]], np.int64))
    collector.finish(origin + timing.MINUTE_NS)
    _, mask = collector.values([True])
    counts, count_mask = collector.order_count_values([True])
    assert mask.tolist() == [True]
    assert count_mask.tolist() == [False]
    assert all(np.isnan(v[0]) for v in counts.values())


def test_roll_is_available_one_minute_after_prior_day():
    daily = np.array([[10, 0, 0], [0, 20, 0]], np.uint64)
    selected, chosen, _ = selected_minutes(daily, [(2024, 6), (2024, 8)])
    day = 24 * 60
    assert chosen.tolist() == [-1, 0, 1]
    assert selected[day] == -1
    assert selected[day + 1] == 0
    assert selected[2 * day] == 0
    assert selected[2 * day + 1] == 1


def test_book_mask_requires_same_contract_clean_prior_and_complete_integral():
    keys = ('bad_bbo', 'gap', 'bad_recv', 'snapshot', 'reset',
            'duplicate_trade_signature', 'file_seam')
    quality = {k: np.zeros(5, np.uint64) for k in keys}
    quality.update(records=np.ones(5, np.uint64), clean_complete_bbo=np.ones(5, np.uint64))
    ofi, bbo = l1_masks(quality, np.array([0, 0, 1, 1, 1]),
                         np.array([timing.MINUTE_NS] * 4 + [0]))
    assert ofi.tolist() == [False, True, False, True, True]
    assert bbo.tolist() == [False, True, False, True, False]
    quality['gap'][3] = 1
    ofi, _ = l1_masks(quality, np.array([0, 0, 1, 1, 1]), np.full(5, timing.MINUTE_NS))
    assert not ofi[3] and not ofi[4]


def test_footprint_sums_fragmented_price_rows_and_keeps_unknown(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from gx1.scripts.rebuild_native_gc_inputs_v1 import write_footprint
    path = tmp_path / 'selected.parquet'
    schema = pa.schema([('minute_index', pa.int32()), ('price_tick', pa.int64()),
                        ('buy', pa.uint64()), ('sell', pa.uint64()), ('unknown', pa.uint64())])
    with pq.ParquetWriter(path, schema) as writer:
        for buy, unknown in [(2, 1), (3, 0)]:
            writer.write_table(pa.table({'minute_index': [0], 'price_tick': [100],
                                         'buy': [buy], 'sell': [0], 'unknown': [unknown]}, schema=schema))
    target = tmp_path / 'footprint.parquet'
    count = write_footprint([path], np.array([timing.STUDY_START_NS]), np.array([0]),
        {'buy': np.array([5]), 'sell': np.array([0]), 'unknown': np.array([1])},
        {'volume': np.array([True]), 'delta': np.array([False])}, target)
    assert count == 1
    values = pq.read_table(target).to_pydict()
    assert values['buy'] == [5] and values['unknown'] == [1]
    assert values['signed_valid'] == [False]



def _derived_test_plan():
    from gx1.scripts.rebuild_native_gc_inputs_v1 import DERIVED_TIMEFRAME_SCHEMA, DERIVED_TIMEFRAME_MODE
    return {'schema_version': DERIVED_TIMEFRAME_SCHEMA, 'mode': DERIVED_TIMEFRAME_MODE,
        'timeframes': ['M5', 'M15', 'H1', 'H4', 'D1'],
        'study_start_inclusive': '2024-05-01T00:00:00Z', 'study_end_exclusive': '2026-07-01T00:00:00Z'}


def test_derived_mode_requires_new_explicit_schema_mode_and_original_scope():
    import pytest
    from gx1.scripts.rebuild_native_gc_inputs_v1 import derived_timeframe_mode
    assert not derived_timeframe_mode({'schema_version': 'gx1_native_raw_gc_input_rebuild_plan_v1'})
    assert derived_timeframe_mode(_derived_test_plan())
    for change in ({'mode': 'raw'}, {'timeframes': ['M5']},
            {'study_end_exclusive': '2026-07-02T00:00:00Z'}):
        plan = {**_derived_test_plan(), **change}
        with pytest.raises(RuntimeError, match='DERIVED_GC_PLAN_SCOPE'):
            derived_timeframe_mode(plan)
    with pytest.raises(RuntimeError, match='DERIVED_GC_PLAN_MODE'):
        derived_timeframe_mode({'schema_version': 'unknown', 'mode': 'derive_timeframes_from_completed_m1'})


def _write_completed_footprint(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([('bar_start_ns', pa.int64()), ('contract_index', pa.int16()),
        ('price_tick', pa.int64()), ('buy', pa.uint64()), ('sell', pa.uint64()),
        ('unknown', pa.uint64()), ('signed_valid', pa.bool_())])
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, row_group_size=1)


def test_completed_footprint_maps_actual_m1_clock_and_retains_all_side_quantities(tmp_path):
    import pyarrow.parquet as pq
    from gx1.scripts.rebuild_native_gc_inputs_v1 import stream_completed_footprint
    origin = timing.STUDY_START_NS
    minutes = origin + np.arange(3, dtype=np.int64) * timing.MINUTE_NS
    path, output = tmp_path / 'footprint.parquet', tmp_path / 'ladder.parquet'
    # Published raw footprint order is by contract, not globally by M1 clock.
    _write_completed_footprint(path, [
        {'bar_start_ns': int(minutes[2]), 'contract_index': 0, 'price_tick': 100,
            'buy': 3, 'sell': 2, 'unknown': 1, 'signed_valid': False},
        {'bar_start_ns': int(minutes[0]), 'contract_index': 1, 'price_tick': 101,
            'buy': 4, 'sell': 0, 'unknown': 0, 'signed_valid': True},
        {'bar_start_ns': int(minutes[0]), 'contract_index': 1, 'price_tick': 102,
            'buy': 1, 'sell': 2, 'unknown': 0, 'signed_valid': True}])
    metrics = {'buy': np.array([5, 0, 3], np.uint64), 'sell': np.array([2, 0, 2], np.uint64),
        'unknown': np.array([0, 0, 1], np.uint64)}
    masks = {'volume': np.array([True, False, True]), 'delta': np.array([True, False, False])}
    assert stream_completed_footprint(path, minutes, np.array([1, 1, 0]), metrics, masks, output) == 3
    result = pq.read_table(output).to_pydict()
    assert result['minute_index'] == [2, 0, 0]
    assert result['unknown'] == [1, 0, 0]
    assert result['buy'] == [3, 4, 1]


def test_completed_footprint_rejects_wrong_side_sum_and_missing_admitted_minute(tmp_path):
    import pytest
    from gx1.scripts.rebuild_native_gc_inputs_v1 import stream_completed_footprint
    origin = timing.STUDY_START_NS
    minutes = origin + np.arange(2, dtype=np.int64) * timing.MINUTE_NS
    path = tmp_path / 'footprint.parquet'
    _write_completed_footprint(path, [{'bar_start_ns': int(minutes[0]), 'contract_index': 0,
        'price_tick': 100, 'buy': 3, 'sell': 2, 'unknown': 0, 'signed_valid': True}])
    masks = {'volume': np.array([True, True]), 'delta': np.array([True, True])}
    metrics = {'buy': np.array([4, 1]), 'sell': np.array([2, 1]), 'unknown': np.array([0, 0])}
    with pytest.raises(RuntimeError, match='DERIVED_GC_FOOTPRINT_SIDE_CONSERVATION'):
        stream_completed_footprint(path, minutes, np.array([0, 0]), metrics, masks, tmp_path / 'bad-side.parquet')
    metrics['buy'][0] = 3
    with pytest.raises(RuntimeError, match='DERIVED_GC_FOOTPRINT_MINUTE_CONSERVATION'):
        stream_completed_footprint(path, minutes, np.array([0, 0]), metrics, masks, tmp_path / 'missing-minute.parquet')


def test_completed_footprint_rejects_contract_and_exact_clock_mismatch(tmp_path):
    import pytest
    from gx1.scripts.rebuild_native_gc_inputs_v1 import stream_completed_footprint
    minutes = np.array([timing.STUDY_START_NS], np.int64)
    row = {'bar_start_ns': int(minutes[0]), 'contract_index': 1, 'price_tick': 100,
        'buy': 3, 'sell': 2, 'unknown': 0, 'signed_valid': True}
    metrics = {'buy': np.array([3]), 'sell': np.array([2]), 'unknown': np.array([0])}
    masks = {'volume': np.array([True]), 'delta': np.array([True])}
    for index, change in enumerate(({}, {'contract_index': 0, 'bar_start_ns': int(minutes[0]) + 1})):
        path = tmp_path / f'bad-{index}.parquet'
        _write_completed_footprint(path, [{**row, **change}])
        with pytest.raises(RuntimeError, match='DERIVED_GC_FOOTPRINT_M1_MAPPING'):
            stream_completed_footprint(path, minutes, np.array([0]), metrics, masks, tmp_path / f'out-{index}.parquet')


def test_derived_inputs_require_completed_original_inventory_and_exact_hashes(tmp_path):
    import hashlib
    import json
    import pytest
    from gx1.scripts.rebuild_native_gc_inputs_v1 import DERIVED_INPUT_NAMES, bound_derived_inputs
    inputs, artifacts = {}, {}
    for name, filename in DERIVED_INPUT_NAMES.items():
        path = tmp_path / filename
        path.write_bytes(name.encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        inputs[name] = {'path': str(path), 'sha256': digest}
        artifacts[filename] = {'sha256': digest, 'bytes': path.stat().st_size}
    terminal = tmp_path / 'TERMINAL.json'
    value = {'status': 'COMPLETE', 'exit_code': 0, 'source_unchanged': True,
             'test_data_used': False, 'artifacts': artifacts}
    terminal.write_text(json.dumps(value))
    inputs['source_terminal'] = {'path': str(terminal), 'sha256': hashlib.sha256(terminal.read_bytes()).hexdigest()}
    assert set(bound_derived_inputs({'inputs': inputs})) == set(inputs)
    value['exit_code'] = 1
    terminal.write_text(json.dumps(value))
    inputs['source_terminal']['sha256'] = hashlib.sha256(terminal.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match='DERIVED_GC_SOURCE_NOT_COMPLETE'):
        bound_derived_inputs({'inputs': inputs})


def test_completion_mode_is_new_scope_and_requires_explicit_resource_inputs():
    import pytest
    from gx1.scripts.rebuild_native_gc_inputs_v1 import completion_mode,COMPLETION_SCHEMA,COMPLETION_MODE
    plan={**_derived_test_plan(),'schema_version':COMPLETION_SCHEMA,'mode':COMPLETION_MODE,
        'worker_count':4,'profile_state_max_bytes':64*1024*1024,'expected_source_records':892100712,'expected_source_files':66,
        'source_plan':{'path':'configs/research/GC_M5_FEATURE_BUILD_20261017_006.json','sha256':'a'*64}}
    assert completion_mode(plan)
    assert not completion_mode(_derived_test_plan())
    for change in ({'worker_count':0},{'profile_state_max_bytes':0},{'timeframes':['M5']},{'mode':'raw'}):
        with pytest.raises(RuntimeError,match='COMPLETION_PLAN_SCOPE'):completion_mode({**plan,**change})


def test_completion_event_worker_reuses_original_metrics_and_book_without_new_trade_ladder(tmp_path, monkeypatch):
    import csv,gzip
    import pyarrow.parquet as pq
    import pytest
    from gx1.contracts import gc_continuous_source_v1 as source
    from gx1.scripts.rebuild_native_gc_inputs_v1 import completion_event_worker
    origin=timing.STUDY_START_NS
    monkeypatch.setattr(timing,'WINDOW_END_NS',origin+timing.MINUTE_NS)
    path=tmp_path/'source.csv.gz'
    rows=[]
    for i,offset in enumerate((0,10_000_000_000,40_000_000_000)):
        rows.append({'ts_recv':origin+offset,'ts_event':origin+offset-100,'instrument_id':42,'publisher_id':1,
            'symbol':'GCM4','action':'T','side':'B' if i!=1 else 'A','size':2,'price':100,
            'flags':source.L1_F_LAST,'sequence':i,'rtype':1,'depth':0,'ts_in_delta':20,
            'bid_px_00':99,'ask_px_00':101,'bid_sz_00':10,'ask_sz_00':20,'bid_ct_00':1,'ask_ct_00':2})
    with gzip.open(path,'wt',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames={**source.COLUMNS,**source.L1_AUDIT_COLUMNS})
        writer.writeheader();writer.writerows(rows)
    group={'raw_symbol':'GCM4','items':[{'path':str(path),'raw_symbol':'GCM4','instrument_id':42,'publisher_id':1,
        'start_ns':origin,'end_ns':origin+timing.MINUTE_NS,'max_records':3}]}
    baseline=L1FeatureCollectorV2([0],1)
    expected=source.scan_symbol(group['items'],1,on_batch=baseline,extra_columns=source.L1_AUDIT_COLUMNS,on_file_start=baseline.file_start)
    baseline.finish()
    metrics={name:expected[name] for name in source.METRICS}
    book=baseline.values([True])[0]
    stage=tmp_path/'completion';stage.mkdir()
    result=completion_event_worker(group,0,1,np.array([0]),np.array([True]),metrics,book,stage)
    assert result['selected_m1_rows']==1
    table=pq.read_table(result['path'])
    assert table['qualified_final_event_count'].to_pylist()==[3.]
    assert not list(stage.glob('*LADDER*'))
    with_sizes=tmp_path/'completion_with_sizes';with_sizes.mkdir()
    extended=completion_event_worker(group,0,1,np.array([0]),np.array([True]),metrics,book,with_sizes,include_trade_statistics=True)
    size_table=pq.read_table(extended['trade_statistics_path'])
    assert size_table['buy_trade_records'].to_pylist()==[2]
    assert size_table['sell_trade_records'].to_pylist()==[1]
    assert size_table['unknown_trade_records'].to_pylist()==[0]
    assert size_table['size_sum_contracts'].to_pylist()==[6]
    assert size_table['size_sum_squared_contracts2'].to_pylist()==[12]
    assert size_table['size_min_contracts'].to_pylist()==[2]
    assert size_table['size_max_contracts'].to_pylist()==[2]
    assert pq.read_table(extended['path']).equals(table)
    wrong={name:values.copy() for name,values in metrics.items()};wrong['volume'][0]+=1
    with pytest.raises(RuntimeError,match='ORIGINAL_METRIC_PARITY:volume'):
        completion_event_worker(group,0,1,np.array([0]),np.array([True]),wrong,book,stage)


def test_completion_basis_binds_original_source_plan_and_existing_quote_bytes(tmp_path,monkeypatch):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pytest
    from gx1.scripts import rebuild_native_gc_inputs_v1 as rebuild
    from gx1.scripts.gc_order_flow_features_v1 import previous
    origin=timing.STUDY_START_NS
    minutes=origin+np.arange(5,dtype=np.int64)*timing.MINUTE_NS
    part=tmp_path/'spot.parquet';manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    pq.write_table(pa.table({'time':pa.array(minutes,type=pa.timestamp('ns',tz='UTC')),
        'bid_close':np.full(5,99.),'ask_close':np.full(5,101.)}),part)
    source_plan=tmp_path/'source_plan.json'
    source_plan.write_text(json.dumps({'inputs':{'oanda_manifest':{
        'path':str(manifest),'sha256':previous.sha256_file(manifest),
        'partitions':[{'path':str(part),'sha256':previous.sha256_file(part)}]}}}))
    plan={'source_plan':{'path':str(source_plan),'sha256':previous.sha256_file(source_plan)}}
    surface={'bar_start_ns':np.array([origin]),'available_ns':np.array([origin+5*timing.MINUTE_NS]),
        'volume_valid':np.array([False]),'close_usd':np.array([np.nan]),
        'observed_valid':np.array([True]),'observed_close_usd':np.array([101.]),
        'observed_last_minute_ns':np.array([minutes[2]])}
    sources=rebuild.add_completion_basis(plan,{'M5':surface},minutes)
    assert {item['path'] for item in sources}=={str(source_plan),str(manifest),str(part)}
    assert surface['observed_spot_basis_valid'].tolist()==[True]
    assert surface['observed_spot_basis_bps'][0]==pytest.approx(100.)
    assert 'gx1/scripts/gc_order_flow_features_v1.py' in rebuild.CODE_PATHS
    manifest.write_text('{"changed":true}')
    with pytest.raises(RuntimeError,match='OANDA_MANIFEST'):
        rebuild.add_completion_basis(plan,{'M5':surface},minutes)


def _incident_manifest():
    return {'schema_version':'gx1_gc_vendor_source_incidents_v1','dataset':'GLBX.MDP3',
        'nontrade_action_semantics_admitted':False,'incidents':[{'incident_id':'PTP_20250917',
        'source_url':'https://issues.databento.com/',
        'start_inclusive_utc':'2025-09-17T13:34:28Z','end_exclusive_utc':'2025-09-17T13:50:40Z'}]}


def test_documented_ptp_interval_censors17_minutes_and_first_clean_book_event_anchor():
    import pandas as pd
    from gx1.contracts import gc_continuous_source_v1 as source
    minutes=pd.date_range('2025-09-17T13:20Z',periods=60,freq='min').asi8
    quality={name:np.zeros(60,np.uint64) for name in source.L1_QUALITY_METRICS}
    quality['records'][:]=1;quality['clean_complete_bbo'][:]=1
    selected=np.zeros(60,np.int16)
    result=source.source_incident_masks_v1(minutes,selected,quality,_incident_manifest())
    assert np.flatnonzero(result['receive_clock_excluded']).tolist()==list(range(14,31))
    assert np.flatnonzero(result['book_event_state_excluded']).tolist()==list(range(14,32))
    assert result['incidents'][0]['recovery_anchor_minute_ns']==minutes[31]
    quality['clean_complete_bbo'][31]=0
    result=source.source_incident_masks_v1(minutes,selected,quality,_incident_manifest())
    assert np.flatnonzero(result['book_event_state_excluded']).tolist()==list(range(14,33))
    assert not result['receive_clock_excluded'][31]


def test_completed_footprint_checks_excluded_side_volume_before_source_censor(tmp_path):
    import pyarrow.parquet as pq
    import pytest
    from gx1.scripts.rebuild_native_gc_inputs_v1 import stream_completed_footprint
    minutes=timing.STUDY_START_NS+np.arange(2,dtype=np.int64)*timing.MINUTE_NS
    path=tmp_path/'grid.parquet';out=tmp_path/'filtered.parquet'
    rows=[{'bar_start_ns':int(stamp),'contract_index':0,'price_tick':100,
        'buy':2,'sell':1,'unknown':0,'signed_valid':True} for stamp in minutes]
    _write_completed_footprint(path,rows)
    metrics={'buy':np.array([2,2]),'sell':np.array([1,1]),'unknown':np.array([0,0])}
    masks={'volume':np.ones(2,bool),'delta':np.ones(2,bool)}
    assert stream_completed_footprint(path,minutes,np.zeros(2,int),metrics,masks,out,
        admitted_minutes=np.array([True,False]))==2
    assert pq.read_table(out)['minute_index'].to_pylist()==[0]
    rows[1]['buy']=3;_write_completed_footprint(path,rows)
    with pytest.raises(RuntimeError,match='SIDE_CONSERVATION'):
        stream_completed_footprint(path,minutes,np.zeros(2,int),metrics,masks,out,
            admitted_minutes=np.array([True,False]))
    masks['volume'][1]=False
    with pytest.raises(RuntimeError,match='ADMISSION_NOT_SUBSET'):
        stream_completed_footprint(path,minutes,np.zeros(2,int),metrics,masks,out,
            admitted_minutes=np.ones(2,bool))


def test_quality_event_reuse_rejects_shifted_clock_size_mask_and_covered_time(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pytest
    from gx1.features import gc_order_flow_v1 as features
    from gx1.scripts.rebuild_native_gc_inputs_v1 import load_quality_event_reductions
    minutes=timing.STUDY_START_NS+np.arange(3,dtype=np.int64)*timing.MINUTE_NS
    candidate=np.array([True,False,True]);selected=np.zeros(3,np.int16)
    fields=set(features.EVENT_DIAGNOSTIC_REDUCERS_V1)|{'qualified_covered_ns','qualified_events_per_observed_second'}
    fields|={prefix+'_mean_ns' for prefix in ('receive_interval',)+features.EVENT_LATENCY_DOMAINS_V1}
    values={name:np.where(candidate,1.,np.nan) for name in fields}
    values['qualified_covered_ns']=np.where(candidate,timing.MINUTE_NS,np.nan)
    baseline={'bar_start_ns':minutes,'available_ns':minutes+timing.MINUTE_NS,'contract_index':selected,**values}
    path=tmp_path/'events.parquet';pq.write_table(pa.table(baseline),path)
    actual=load_quality_event_reductions(path,minutes,selected,candidate,np.full(3,timing.MINUTE_NS))
    assert set(actual)==fields
    for column,changed in (
        ('bar_start_ns',minutes+1),('contract_index',np.ones(3,np.int16)),
        ('qualified_final_event_count',np.ones(3)),
        ('qualified_covered_ns',np.where(candidate,timing.MINUTE_NS-1,np.nan))):
        pq.write_table(pa.table({**baseline,column:changed}),path)
        with pytest.raises(RuntimeError,match='GC_QUALITY_EVENT'):
            load_quality_event_reductions(path,minutes,selected,candidate,np.full(3,timing.MINUTE_NS))
    pq.write_table(pa.table(baseline).slice(0,2),path)
    with pytest.raises(RuntimeError,match='M1_CLOCK_OR_CONTRACT'):
        load_quality_event_reductions(path,minutes,selected,candidate,np.full(3,timing.MINUTE_NS))


def test_quality_legacy_strict_m5_keeps_original_sum_mean_response_and_admission():
    from gx1.features import gc_order_flow_v1 as features
    from gx1.scripts.rebuild_native_gc_inputs_v1 import legacy_strict_m5_surface
    from gx1.contracts import gc_continuous_source_v1 as source
    minutes=timing.STUDY_START_NS+np.arange(15,dtype=np.int64)*timing.MINUTE_NS
    selected=np.zeros(15,np.int16);metrics={name:np.zeros(15,np.uint64) for name in source.METRICS}
    metrics.update(volume=np.full(15,3,np.uint64),trades=np.ones(15,np.uint64),
        buy=np.full(15,2,np.uint64),sell=np.ones(15,np.uint64))
    masks={'volume':np.ones(15,bool),'delta':np.ones(15,bool)}
    masks['volume'][6]=False;masks['delta'][6]=False
    book={name:np.ones(15) for name in features.L1_FIELDS_V2+features.L1_ORDER_COUNT_FIELDS_V1}
    for name in ('ofi_valid','bbo_valid','order_count_valid'):
        book[name]=np.ones(15,bool);book[name][11]=False
    book['opening_mid_tick']=np.arange(15)+100.;book['closing_mid_tick']=np.arange(15)+101.
    ohlc=np.full((15,4),100,np.int64)
    out=legacy_strict_m5_surface(minutes,selected,metrics,masks,ohlc,book,100000000,
        features.dynamic_profile_arrays_v2(minutes[::5]))
    assert out['volume'][0]==15 and np.isnan(out['volume'][1])
    assert out['ofi_contracts'][0]==5 and np.isnan(out['ofi_contracts'][2])
    assert out['tw_imbalance'][0]==1 and np.isnan(out['tw_imbalance'][2])
    assert out['mid_response_ticks'][0]==5
    assert out['tw_bid_orders'][0]==1 and np.isnan(out['tw_bid_orders'][2])
    assert out['close_usd'][0]==10 and np.isnan(out['close_usd'][1])


def test_quality_mode_is_derived_only_new_explicit_scope_and_bindings():
    import pytest
    from gx1.scripts import rebuild_native_gc_inputs_v1 as r
    binding={'path':'declared','sha256':'a'*64}
    plan={**_derived_test_plan(),'schema_version':r.QUALITY_REDERIVE_SCHEMA,'mode':r.QUALITY_REDERIVE_MODE,
        'profile_state_max_bytes':67108864,'source_plan':binding,'source_incidents':binding,
        'retained_registry_source':binding,'event_reductions':{**binding,'source_terminal':binding}}
    assert r.quality_rederive_mode(plan)
    assert not r.quality_rederive_mode(_derived_test_plan())
    for field in ('source_plan','source_incidents','retained_registry_source','event_reductions'):
        changed={**plan,field:{}}
        with pytest.raises(RuntimeError,match='QUALITY_REDERIVE_SCOPE'):r.quality_rederive_mode(changed)


def test_trusted_record_mode_requires_new_plan_and_measured_restoration_counts():
    import pytest
    from gx1.scripts import rebuild_native_gc_inputs_v1 as r
    plan = dict(_derived_test_plan(),schema_version=r.TRUSTED_RESTORATION_SCHEMA,mode=r.TRUSTED_RESTORATION_MODE,
        profile_state_max_bytes=1024,source_plan={'path':'base.json','sha256':'x'},
        source_incidents={'path':'incidents.json','sha256':'x'},retained_registry_source={'path':'registry.json','sha256':'x'},
        event_reductions={'path':'events.parquet','sha256':'x','source_terminal':{'path':'TERMINAL.json','sha256':'x'}},
        worker_count=4,expected_source_records=100,expected_source_files=2,
        expected_duplicate_only_m1=3,expected_duplicate_only_volume_contracts=7)
    assert r.trusted_restoration_mode(plan)
    for name in ('expected_duplicate_only_m1','expected_duplicate_only_volume_contracts','worker_count'):
        with pytest.raises(RuntimeError,match='RESTORATION_SCOPE'):r.trusted_restoration_mode({**plan,name:0})
    assert not r.quality_rederive_mode(plan)
    with pytest.raises(RuntimeError):r.derived_timeframe_mode(plan)


def test_signature_identity_changes_only_fresh_explicit_trade_and_book_admission():
    from gx1.contracts import gc_continuous_source_v1 as source
    from gx1.scripts import rebuild_native_gc_inputs_v1 as r
    n=5;selected=np.zeros(n,np.int16)
    metrics={name:np.zeros(n,np.uint64) for name in source.METRICS}
    for name in ('volume','buy','trades'):metrics[name][:]=1
    metrics['duplicate_trade'][1]=1;metrics['invalid_trade'][3]=1
    old=source.row_masks(selected,metrics,np.ones(n,bool))['volume']
    new=source.row_masks(selected,metrics,np.ones(n,bool),reject_duplicate_signature=False)['volume']
    assert old.tolist()==[True,False,True,False,True]
    assert new.tolist()==[True,True,True,False,True]
    quality={name:np.zeros(n,np.uint64) for name in source.L1_QUALITY_METRICS}
    quality['records'][:]=1;quality['clean_complete_bbo'][:]=1
    quality['duplicate_trade_signature'][1]=1;quality['gap'][3]=1
    covered=np.full(n,timing.MINUTE_NS)
    oldofi,_=r.l1_masks(quality,selected,covered)
    newofi,_=r.l1_masks(quality,selected,covered,reject_duplicate_signature=False)
    assert oldofi.tolist()==[False,False,False,False,False]
    assert newofi.tolist()==[False,True,True,False,False]
    assert metrics['duplicate_trade'][1]==1 and quality['duplicate_trade_signature'][1]==1


def _trusted_cache_fixture(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from gx1.contracts import gc_continuous_source_v1 as source
    from gx1.scripts import rebuild_native_gc_inputs_v1 as r
    n=3*source.MINUTES_PER_DAY;minutes=timing.STUDY_START_NS+np.arange(n,dtype=np.int64)*timing.MINUTE_NS
    metrics={name:np.zeros(n,np.uint64) for name in source.METRICS}
    for name in ('volume','buy','trades'):metrics[name][:]=1
    metrics['duplicate_trade'][1500]=1;metrics['invalid_trade'][1600]=1
    quality={name:np.zeros(n,np.uint64) for name in source.L1_QUALITY_METRICS}
    quality['records'][:]=1;quality['clean_complete_bbo'][:]=1
    quality['duplicate_trade_signature'][1500]=1;quality['bad_bbo'][1600]=1
    selected,chosen,basis=r.selected_minutes(np.full((1,3),source.MINUTES_PER_DAY,np.uint64),[(2025,6)])
    raw={**metrics,**{'quality_'+k:v for k,v in quality.items()},
        **{k:np.full(n,100,np.int64) for k in ('open_tick','high_tick','low_tick','close_tick')},
        'covered_ns':np.full(n,timing.MINUTE_NS,np.int64),'order_covered_ns':np.full(n,timing.MINUTE_NS,np.int64),
        'opening_mid_tick':np.full(n,101.),'closing_mid_tick':np.full(n,102.),
        **{k:np.ones(n) for k in r.features.L1_FIELDS_V2+r.features.L1_ORDER_COUNT_FIELDS_V1}}
    path=tmp_path/'RAW_M1.parquet';pq.write_table(pa.table(raw),path)
    values={k:np.where(selected>=0,v,np.nan if v.dtype.kind=='f' else 0).astype(v.dtype) for k,v in raw.items()}
    old=source.row_masks(selected,{k:values[k] for k in metrics},np.ones(n,bool))
    ofi,bbo=r.l1_masks({k:values['quality_'+k] for k in quality},selected,values['covered_ns'])
    order=ofi & (values['order_covered_ns']==timing.MINUTE_NS)
    for j,k in enumerate(r.features.L1_FIELDS_V2+r.features.L1_ORDER_COUNT_FIELDS_V1):
        values[k]=np.where(ofi if j<6 else bbo if j<len(r.features.L1_FIELDS_V2) else order,values[k],np.nan)
    original=pa.table({'bar_start_ns':minutes,'contract_index':selected,**values,
        'volume_valid':old['volume'],'signed_valid':old['delta'],'ofi_valid':ofi,'bbo_valid':bbo,'order_count_valid':order})
    roll={'contract_order':['GCJ5'],'metadata_identity':{'GCJ5':[2025,6,100000000]},
        'selected_daily':chosen.tolist(),'basis_daily':basis.tolist()}
    return original,[{'symbol':'GCJ5','contract_index':0,'m1':path}],roll


def test_trusted_cache_restoration_preserves_raw_metrics_roll_and_nonduplicate_guards(tmp_path):
    import pytest
    from gx1.scripts import rebuild_native_gc_inputs_v1 as r
    original,caches,roll=_trusted_cache_fixture(tmp_path)
    restored,old,new,delta=r.restore_selected_m1_from_raw_caches(original,caches,roll)
    assert delta['volume_m1']==1 and delta['volume_contracts']==1 and delta['ofi_m1']==2
    assert not old['volume'][1500] and new['volume'][1500] and not new['volume'][1600]
    assert restored['duplicate_trade'][1500]==1
    assert np.isfinite(restored['tw_spread_ticks'][1500])
    for name in r.source.METRICS:np.testing.assert_array_equal(restored[name],original[name].to_numpy())
    bad=dict(roll,selected_daily=[-1,-1,-1])
    with pytest.raises(RuntimeError,match='ROLL_PARITY'):r.restore_selected_m1_from_raw_caches(original,caches,bad)
    import pyarrow as pa
    broken=original.set_column(original.schema.get_field_index('volume'), 'volume', pa.array(original['volume'].to_numpy()+1))
    with pytest.raises(RuntimeError,match='REDUCTION_PARITY'):r.restore_selected_m1_from_raw_caches(broken,caches,roll)


def test_unfiltered_cache_inventory_requires_exact_receipt_entry_hash_and_bytes(tmp_path):
    import json
    import pytest
    from gx1.scripts import rebuild_native_gc_inputs_v1 as r
    root=tmp_path/'RESULTS';(root/'RAW_REDUCTIONS').mkdir(parents=True)
    artifacts={}
    for suffix in ('_M1.parquet','_LADDER.parquet'):
        relative='RAW_REDUCTIONS/GCJ5'+suffix;path=root/relative;path.write_bytes(b'bound cache')
        artifacts[relative]={'sha256':r.io.sha256_file(path),'bytes':path.stat().st_size}
    terminal=root/'TERMINAL.json';terminal.write_text(json.dumps({'artifacts':artifacts}))
    paths={'source_terminal':terminal};roll={'contract_order':['GCJ5']}
    caches,bindings=r.bound_raw_reduction_inputs(paths,roll)
    assert len(caches)==1 and len(bindings)==2
    del artifacts['RAW_REDUCTIONS/GCJ5_LADDER.parquet'];terminal.write_text(json.dumps({'artifacts':artifacts}))
    with pytest.raises(RuntimeError,match='CACHE_INVENTORY'):r.bound_raw_reduction_inputs(paths,roll)


def test_retained_rebinding_uses_exact_declared_columns_masks_and_source_clock(tmp_path,monkeypatch):
    import json
    import pytest
    import pyarrow as pa
    import pyarrow.parquet as pq
    from gx1.scripts import rebuild_native_gc_inputs_v1 as r
    monkeypatch.setattr(r.io,'REPO',tmp_path)
    old=tmp_path/'OLD.parquet';stage=tmp_path/'stage';stage.mkdir()
    data=pa.table({'bar_start_ns':np.array([0],np.int64),'available_ns':np.array([5*timing.MINUTE_NS],np.int64),
        'contract_index':np.array([0],np.int16),'volume_valid':[True],'close_usd':[2000.]})
    pq.write_table(data,old);pq.write_table(data,stage/'M5_GC_FEATURES.parquet')
    registry={'schema_version':'gx1_native_gc_context_registry_v1','outcomes_accessed':False,
        'tables':{'M5_GC_FEATURES.parquet':{'path':str(old),'sha256':r.io.sha256_file(old)}},
        'fields':[{'name':'original_close','table':'M5_GC_FEATURES.parquet','column':'close_usd',
            'owner':'price_structure','validity_column':'volume_valid','source_clock_column':'available_ns'}]}
    path=tmp_path/'registry.json';path.write_text(json.dumps(registry))
    plan={'retained_registry_source':{'path':'registry.json','sha256':r.io.sha256_file(path)},'run_root':str(tmp_path/'run')}
    result=r.retained_quality_rebindings(plan,stage)
    assert result['retained_field_count']==1 and not result['native_registry_admitted']
    assert result['table_rebindings'][0]['columns'][0]['column']=='close_usd'
    changed=data.set_column(1,'available_ns',pa.array([6*timing.MINUTE_NS],type=pa.int64()))
    pq.write_table(changed,stage/'M5_GC_FEATURES.parquet')
    with pytest.raises(RuntimeError,match='CLOCK_OR_CONTRACT'):r.retained_quality_rebindings(plan,stage)


def test_raw_unmasked_event_subset_parity_keeps_new_observations_but_rejects_old_change():
    import pytest
    from gx1.scripts import rebuild_native_gc_inputs_v1 as r
    prior=np.array([True,False,True])
    original={'count':np.array([2.,np.nan,3.]),'latency':np.array([100.,np.nan,np.nan])}
    raw={'count':np.array([2.,4.,3.]),'latency':np.array([100.,200.,np.nan])}
    r.assert_original_event_subset(raw,original,prior)
    bad={**raw,'count':np.array([2.,4.,4.])}
    with pytest.raises(RuntimeError,match='SUBSET_PARITY'):r.assert_original_event_subset(bad,original,prior)
    with pytest.raises(RuntimeError,match='SUBSET_SCHEMA'):r.assert_original_event_subset({'count':raw['count']},original,prior)


def test_restored_price_grid_requires_exact_price_side_rows_on_prior_population(tmp_path):
    import pytest
    from gx1.scripts import rebuild_native_gc_inputs_v1 as r
    origin=timing.STUDY_START_NS;minutes=origin+np.arange(3)*timing.MINUTE_NS
    old=tmp_path/'old.parquet';restored=tmp_path/'restored.parquet'
    def row(start,price,buy,sell):
        return {'bar_start_ns':start,'contract_index':0,'price_tick':price,
            'buy':buy,'sell':sell,'unknown':0,'signed_valid':True}
    rows=[row(origin,100,2,1),row(origin+2*timing.MINUTE_NS,102,3,1)]
    _write_completed_footprint(old,rows)
    _write_completed_footprint(restored,[rows[0],row(origin+timing.MINUTE_NS,101,8,2),rows[1]])
    prior=np.array([True,False,True])
    assert r.assert_restored_price_grid_original_subset(old,restored,minutes,prior)==2
    _write_completed_footprint(restored,[rows[0],row(origin+2*timing.MINUTE_NS,103,3,1)])
    with pytest.raises(RuntimeError,match='PRICE_GRID_PARITY'):r.assert_restored_price_grid_original_subset(old,restored,minutes,prior)
