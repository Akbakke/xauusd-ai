"""Direct 300-second reduction and all formula families on synthetic MBP events."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.features import gc_native_m5_features_v1 as native
from gx1.features import gc_order_flow_v1 as f

INTERVAL = 300_000_000_000
ORIGIN = pd.Timestamp('2024-05-01T00:00Z').value


def scope(rows):
    return {'schema_version': 'gx1_gc_native_m5_reduction_clock_v1',
            'start_ns': ORIGIN, 'end_ns': ORIGIN + rows * INTERVAL,
            'bar_duration_ns': INTERVAL}


def test_native_l1_means_integrate_300_seconds_and_never_fill_missing_interior_buckets():
    collector = f.L1FeatureCollectorV2(np.arange(5), 1, origin_ns=ORIGIN,
        include_order_counts=True, input_interval_ns=INTERVAL)
    # A genuine state at the left edge plus a later raw event proves the first
    # interval; a long event gap does not manufacture interior source buckets.
    events = np.array([[ORIGIN, 100, 102, 9, 3, 3, 1],
        [ORIGIN + INTERVAL, 101, 103, 4, 4, 2, 2],
        [ORIGIN + 4*INTERVAL, 102, 104, 5, 5, 1, 1]], np.int64)
    collector.consume(events)
    collector.finish(ORIGIN + 5*INTERVAL)
    values, valid = collector.values(np.ones(5, bool))
    orders, order_valid = collector.order_count_values(np.ones(5, bool))
    np.testing.assert_array_equal(collector.covered, [INTERVAL, INTERVAL, 0, 0, INTERVAL])
    np.testing.assert_array_equal(valid, [True, True, False, False, True])
    np.testing.assert_array_equal(order_valid, valid)
    assert values['tw_imbalance'][0] == .5
    assert orders['tw_bid_contracts_per_order'][0] == 3
    assert np.isnan(values['tw_imbalance'][2:4]).all()
    with pytest.raises(ValueError, match='REDUCTION_INTERVAL'):
        f.L1FeatureCollectorV2(np.arange(1), 1, input_interval_ns=120_000_000_000)


def test_native_skeleton_counts_physical_minutes_and_censors_only_real_gap_duration():
    starts = ORIGIN + np.arange(6, dtype=np.int64)*INTERVAL
    selected = np.zeros(6, np.int64)
    metrics = {name: np.ones(6, np.uint64) for name in ('volume','buy','sell','unknown','trades')}
    metrics.update(volume=np.array([2]*6,np.uint64), buy=np.array([2]*6,np.uint64),
                   sell=np.zeros(6,np.uint64), unknown=np.zeros(6,np.uint64))
    masks = {name: np.array([True,False,True,True,True,True]) for name in ('volume','delta','cvd')}
    prices = np.array([[100+i,102+i,99+i,101+i] for i in range(6)],np.int64)
    surface,_ = f.timeframe_skeleton(starts,selected,metrics,masks,prices,'M15',10**9,
                                      input_interval_ns=INTERVAL)
    np.testing.assert_array_equal(surface['observed_accepted_minutes'], [10,15])
    np.testing.assert_array_equal(surface['observed_expected_clock_minutes'], [15,15])
    np.testing.assert_allclose(surface['observed_coverage_fraction'],[2/3,1])
    assert 'observed_first_minute_ns' not in surface
    np.testing.assert_array_equal(surface['observed_last_bucket_start_ns'],starts[[2,5]])
    f.add_observed_structure(surface,input_interval_ns=INTERVAL)
    assert surface['observed_proxy_boundary_gap_minutes'][1] == 0
    # A complete missing source bucket between observed M5 candles is five
    # minutes, rather than the old one-minute subtraction from an M1 endpoint.
    m5,_=f.timeframe_skeleton(starts,selected,metrics,masks,prices,'M5',10**9,
                              input_interval_ns=INTERVAL)
    f.add_observed_structure(m5,input_interval_ns=INTERVAL)
    assert m5['observed_proxy_boundary_gap_minutes'][2] == 5


def test_native_basis_requires_exact_last_gc_bucket_pair_and_never_backsearches():
    surface={'available_ns':np.array([ORIGIN+3*INTERVAL]),
        'observed_valid':np.array([True]),'volume_valid':np.array([True]),
        'observed_last_bucket_start_ns':np.array([ORIGIN+2*INTERVAL]),
        'observed_close_usd':np.array([103.]),'close_usd':np.array([103.])}
    quotes=pd.DataFrame({'bid_close':[100.,100.],'ask_close':[102.,102.]},
                        index=pd.to_datetime([ORIGIN,ORIGIN+INTERVAL],utc=True))
    starts=ORIGIN+np.arange(3,dtype=np.int64)*INTERVAL
    trade_metadata={'bar_start_ns':starts,'first_trusted_trade_recv_ns':starts+INTERVAL//2,
                    'last_trusted_trade_recv_ns':starts+INTERVAL//2}
    native.add_native_m5_basis(surface,quotes,native_m5_trade_metadata=trade_metadata)
    assert not surface['observed_spot_basis_valid'][0]
    assert np.isnan(surface['observed_spot_basis_bps'][0])
    assert surface['observed_spot_basis_source_available_ns'][0] == -1
    quotes.loc[pd.Timestamp(ORIGIN+2*INTERVAL,tz='UTC')]=[100.,102.]
    native.add_native_m5_basis(surface,quotes,native_m5_trade_metadata=trade_metadata)
    assert surface['observed_spot_basis_valid'][0]
    assert surface['observed_spot_basis_bps'][0] == pytest.approx(10_000*(103/101-1))
    assert surface['observed_spot_basis_age_minutes'][0] == 0
    assert surface['observed_spot_basis_gc_trade_age_minutes'][0] == 2.5
    assert surface['observed_spot_basis_gc_last_trade_recv_ns'][0] == starts[2]+INTERVAL//2
    assert surface['observed_spot_basis_xau_bar_available_ns'][0] == starts[2]+INTERVAL
    surface['observed_last_bucket_start_ns'][0] -= INTERVAL
    native.add_native_m5_basis(surface,quotes,native_m5_trade_metadata=trade_metadata)
    assert surface['observed_spot_basis_age_minutes'][0] == 5
    quotes.loc[pd.Timestamp(ORIGIN+INTERVAL,tz='UTC'),'bid_close']=np.nan
    native.add_native_m5_basis(surface,quotes,native_m5_trade_metadata=trade_metadata)
    assert not surface['observed_spot_basis_valid'][0]
    assert surface['spot_basis_valid'][0]  # Exact final bucket has its own domain.


def _synthetic_native_packet(tmp_path,monkeypatch,*,rows=576,origin=ORIGIN):
    # Two full UTC days exercise a complete canonical D1 bar (22Z anchor).
    clock={**scope(rows),'start_ns':origin,'end_ns':origin+rows*INTERVAL}
    starts=origin+np.arange(rows,dtype=np.int64)*INTERVAL
    selected=np.zeros(rows,np.int64)
    unknown=np.where(np.arange(rows)%4==0,2,0).astype(np.uint64)
    metrics={'trades':np.full(rows,2,np.uint64),'volume':np.full(rows,3,np.uint64),
        'buy':np.ones(rows,np.uint64),'sell':2-unknown,'unknown':unknown}
    masks={'volume':np.ones(rows,bool),'delta':unknown==0,'cvd':unknown==0}
    ladder=tmp_path/'NATIVE_M5_LADDER.parquet'
    callbacks=native.native_m5_collectors(scope=clock,ladder_path=ladder,selected=selected,
        masks=masks,symbol_index=0,tick_size=10**9)
    idx=np.repeat(np.arange(rows),2)
    recv=np.repeat(starts,2)+np.tile(np.array([0,INTERVAL//2],np.int64),rows)
    ticks=100+idx%11
    sides=np.tile(np.array(['B','A']),rows);sides[1::8]='N'
    data=pa.table({'ts_recv':recv,'ts_event':recv-10,'ts_in_delta':np.full(len(idx),5,np.int32),
        'flags':np.full(len(idx),128,np.uint8),'action':['T']*len(idx),'side':sides,
        'price':ticks*10**9,'size':np.tile(np.array([1,2],np.uint32),rows),
        'bid_px_00':(ticks-1)*10**9,'ask_px_00':(ticks+1)*10**9,
        'bid_sz_00':np.full(len(idx),9,np.uint32),'ask_sz_00':np.full(len(idx),3,np.uint32),
        'bid_ct_00':np.full(len(idx),3,np.uint32),'ask_ct_00':np.ones(len(idx),np.uint32)})
    callbacks['on_file_start']({'end_ns':clock['end_ns']})
    callbacks['on_batch'](data,idx,np.ones(len(idx),bool))
    callbacks['book'].finish(clock['end_ns']);callbacks['trades'].close()
    assert pq.ParquetFile(ladder).schema_arrow.names == ['bucket_index','price_tick','buy','sell','unknown']
    candidate=np.ones(rows,bool);candidate[0]=False
    collector=callbacks['book']
    values,bbo=collector.values(candidate);orders,order_valid=collector.order_count_values(candidate)
    book={**values,**orders,'ofi_valid':candidate,'bbo_valid':bbo,'order_count_valid':order_valid,
          'covered_ns':collector.covered,'order_covered_ns':collector.order_covered,
          'opening_mid_tick':collector.first,'closing_mid_tick':collector.last}
    events=collector.event_diagnostics(candidate);stats=collector.trusted_trade_statistics()
    def forbidden(*args,**kwargs):raise AssertionError('legacy registry was read')
    from gx1.features import gc_native_context_v1 as legacy
    monkeypatch.setattr(legacy,'_read_registry',forbidden)
    xau=pd.DataFrame({'bid_close':np.full(rows,100.),'ask_close':np.full(rows,102.)},
                     index=pd.to_datetime(starts,utc=True))
    packet=native.build_native_m5_gc_surfaces(scope=clock,selected=selected,metrics=metrics,masks=masks,
        ohlc=callbacks['trades'].ohlc,ladder_path=ladder,book=book,events=events,trade_stats=stats,
        tick_size=10**9,native_xau_m5=xau,
        first_trusted_trade_recv_ns=callbacks['trades'].first_trusted_trade_recv_ns,
        last_trusted_trade_recv_ns=callbacks['trades'].last_trusted_trade_recv_ns,
        profile_state_max_bytes=1_000_000)
    return packet,callbacks,unknown


def test_raw_native_callbacks_to_all_five_tf_formula_families_without_legacy_reads(tmp_path,monkeypatch):
    packet,callbacks,unknown=_synthetic_native_packet(tmp_path,monkeypatch)
    rows=576;clock=scope(rows);starts=ORIGIN+np.arange(rows,dtype=np.int64)*INTERVAL
    assert tuple(packet['timeframes']) == native.TIMEFRAMES
    for tf,surface in packet['timeframes'].items():
        assert len(surface['available_ns'])>0
        assert np.all(surface['available_ns']<=clock['end_ns'])
        assert not any(name.endswith('_minute_ns') for name in surface)
        assert all(surface[name].dtype==np.int64 for name in native.NATIVE_NULLABLE_TIMESTAMP_COLUMNS)
        assert {'observed_volume','observed_known_cumulative_volume_contracts','observed_proxy_bos_up',
            'observed_book_tw_imbalance','observed_order_tw_bid_orders','observed_event_matching_clock_mean_ns',
            'observed_executed_trade_size_std_contracts','observed_profile_rolling15_vwap_usd',
            'spot_basis_bps','observed_spot_basis_bps'}.issubset(surface)
    m5=packet['timeframes']['M5']
    np.testing.assert_array_equal(m5['observed_volume'],np.full(rows,3))
    np.testing.assert_array_equal(m5['observed_accepted_minutes'],np.full(rows,5))
    np.testing.assert_array_equal(m5['observed_unknown'],unknown)
    assert not m5['signed_valid'][0]
    assert np.isnan(m5['cvd_segment_delta_contracts'][0])
    assert m5['observed_profile_rolling15_valid'][2]
    assert m5['observed_profile_rolling60_valid'][11]
    assert packet['candidate_map']['schema_version']=='gc_native_m5_completion_candidates_v1'
    assert not any('_minute_ns' in name for name in packet['candidate_map']['metadata_columns'])
    assert 'same_closed_m1_cross_venue_basis' not in repr(packet['candidate_map'])
    np.testing.assert_array_equal(callbacks['trades'].last_trade_recv_ns,starts+INTERVAL//2)
    coverage=native.source_coverage_report(packet)
    assert len(coverage['fields'])==1220
    assert coverage['all_declared_source_columns_present']
    assert coverage['source_boolean_fields_without_binary_declaration']==[]
    assert coverage['fields']['gc_m5_observed_profile_rolling60_vwap_usd']['unknown']>0
    assert coverage['fields']['gc_m5_observed_volume']['finite_observed']==rows


def test_native_source_catalog_preserves_all_genuine_names_owners_and_boolean_domains():
    from gx1.features import gc_native_field_catalog_v1 as pure
    catalog=native.native_m5_field_catalog()
    fields=catalog['fields'][:1205]
    assert tuple(field['name'] for field in fields)==pure.VALUE_FIELDS
    assert tuple(field['availability_field'] for field in fields)==pure.OBSERVED_FIELDS
    assert {field['name']:field['owner'] for field in fields}=={
        name:pure.SPECIALIST_OWNERS[name] for name in pure.VALUE_FIELDS}
    historical=pure.STRUCTURAL_VALUE_FIELDS|pure.SOURCE_BINARY_VALUE_FIELDS
    binary={field['name'] for field in fields if field['binary_value']}
    assert historical.issubset(binary)
    assert len(binary-historical)==59
    assert all(field['binary_value'] for field in fields
        if field['column'] in native.NATIVE_BOOLEAN_SOURCE_COLUMNS)
    assert catalog['field_names'][:2410]==list(pure.CONTEXT_FIELDS)
    assert len(catalog['fields'])==1220 and len(catalog['field_names'])==2435
    assert catalog['base_value_fields']==1215
    fresh=catalog['fields'][1205:1215]
    assert len(fresh)==10 and all(f['unit']=='contract' for f in fresh)
    expected={x['name'] for x in f.completion_native_candidates_v1(input_interval_ns=INTERVAL)['fields']}
    actual={x['name'] for x in catalog['fields'] if x['column'].startswith('observed_') and x['packet_group']=='timeframes'
            and x['transform']!='native_m5_trailing_support_gap_minutes'}
    assert actual==expected
    assert sum(x['binary_value'] for x in catalog['fields'])==210
    assert sum(x['binary_value'] and not x['structural_mask_value'] for x in catalog['fields'])==135
    assert not any(value in repr(catalog) for value in ('NATIVE_GC_CONTEXT_FIELDS','retained::','same_closed_m1','_minute_ns','GX1_RUNS'))

    assert native.require_native_m5_field_catalog(catalog)==catalog
    changed={**catalog,'source_clock_contract':'legacy'}
    with pytest.raises(ValueError,match='FIELD_CATALOG_CONTRACT'):
        native.require_native_m5_field_catalog(changed)


def test_native_feature_glue_never_publishes_source_available_at_or_after_test_seal(tmp_path,monkeypatch):
    seal=pd.Timestamp('2026-07-01T00:00Z').value
    packet,callbacks,_=_synthetic_native_packet(tmp_path,monkeypatch,rows=24,origin=seal-24*INTERVAL)
    assert packet['source_clock']['end_ns']==seal
    assert len(packet['timeframes']['M5']['available_ns'])==23
    for table in (*packet['timeframes'].values(),packet['m5_book_features']):
        assert np.all(table['available_ns']<seal)
    assert callbacks['trades'].last_trusted_trade_recv_ns[-1]<seal


@pytest.mark.parametrize('owner', ['completion_native_candidates_v1','strict_native_m5_candidates_v1'])
def test_new_genuine_formula_cannot_disappear_from_native_catalog(monkeypatch,owner):
    original=getattr(f,owner)
    def changed(*args,**kwargs):
        result=original(*args,**kwargs);field=dict(result['fields'][0])
        field.update(name='gc_m5_new_genuine_formula',column='new_genuine_formula')
        return {**result,'fields':[*result['fields'],field]}
    monkeypatch.setattr(f,owner,changed)
    with pytest.raises(RuntimeError,match='FORMULA_CATALOG_UNMAPPED:gc_m5_new_genuine_formula'):
        native.native_m5_field_catalog()


@pytest.mark.parametrize('key,value', [('column','different_column'),('validity_column','different_gate'),
    ('source_clock_column','different_clock'),('role','different_role')])
def test_declared_formula_column_gate_clock_or_role_change_requires_catalog_change(monkeypatch,key,value):
    original=f.completion_native_candidates_v1
    def changed(*args,**kwargs):
        result=original(*args,**kwargs);fields=[dict(x) for x in result['fields']]
        fields[0][key]=value
        return {**result,'fields':fields}
    monkeypatch.setattr(f,'completion_native_candidates_v1',changed)
    with pytest.raises(RuntimeError,match='FORMULA_CATALOG_MISMATCH:.*:'+key):native.native_m5_field_catalog()


def test_side_specific_trade_maxima_are_distinct_genuine_source_columns():
    starts=ORIGIN+np.arange(2,dtype=np.int64)*INTERVAL;selected=np.zeros(2,np.int64)
    metrics={'trades':np.array([2,2],np.uint64),'volume':np.array([7,7],np.uint64),
        'buy':np.array([2,5],np.uint64),'sell':np.array([5,2],np.uint64),'unknown':np.zeros(2,np.uint64)}
    stats={name:np.zeros(2,np.uint64) for name in f.TRUSTED_TRADE_SIZE_FIELDS_V1}
    for name,values in {'buy_trade_records':[1,1],'sell_trade_records':[1,1],
        'size_sum_contracts':[7,7],'size_sum_squared_contracts2':[29,29],
        'size_min_contracts':[2,2],'size_max_contracts':[5,5],
        'buy_size_max_contracts':[2,5],'sell_size_max_contracts':[5,2]}.items():stats[name][:]=values
    out=f.aggregate_observed_trade_size_v1(starts,selected,metrics,{'volume':np.ones(2,bool)},stats,'M5',input_interval_ns=INTERVAL)
    np.testing.assert_array_equal(out['observed_buy_max_executed_trade_contracts'],[2,5])
    np.testing.assert_array_equal(out['observed_sell_max_executed_trade_contracts'],[5,2])
    np.testing.assert_array_equal(out['observed_executed_trade_size_max_contracts'],[5,5])


def test_all_source_units_are_explicit_and_owner_declared_not_fitted_from_values():
    catalog=native.native_m5_field_catalog();policy=native.native_m5_source_unit_policy()
    assert catalog['unit_policy_sha256']==policy['contract_sha256']
    assert len(policy['source_columns'])==321
    for field in catalog['fields']:
        assert field['unit'] and field['unit_scale']==1.
        assert field['unit_owner'].startswith('gx1.features.')
        assert field['unit']=='Boolean' if field['binary_value'] else field['unit']!='Boolean'
    by_name={x['name']:x for x in catalog['fields']}
    assert by_name['gc_m5_observed_spot_basis_bps']['unit']=='basis_point'
    assert by_name['gc_m5_observed_book_tw_bid_contracts']['unit']=='contract'
    assert by_name['gc_d1_observed_event_matching_clock_mean_ns']['unit']=='nanosecond'
    assert by_name['gc_h4_observed_profile_rolling60_vwap_usd']['unit']=='USD'
    assert by_name['gc_m15_observed_proxy_boundary_gap_minutes']['unit']=='minute'
