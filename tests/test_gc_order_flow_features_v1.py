"""Mechanics only; plan 005 must separately prove actual source/row parity."""
from pathlib import Path
import csv
import gzip
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from gx1.features import gc_order_flow_v1 as f
from gx1.features.smc_v1 import replay_volume_order_block_proxies_v1, replay_smc_causal_structure_v2
from gx1.contracts import gc_continuous_source_v1 as c
from gx1.contracts import gc_source_disposition_v1 as t


def minute_inputs(n=30):
    minutes = t.STUDY_START_NS+np.arange(n)*t.MINUTE_NS
    selected = np.zeros(n,dtype=np.int16)
    metrics = {k:np.zeros(n,dtype=np.uint64) for k in c.METRICS}
    metrics['buy'][:]=3; metrics['sell'][:]=1
    metrics['volume'][:]=4; metrics['trades'][:]=2
    masks = {name:np.ones(n,dtype=bool) for name in ('volume','delta','cvd')}
    ohlc = np.tile([100,101,99,100],(n,1)).astype(np.int64)
    return minutes,selected,metrics,masks,ohlc


def test_profile_preserves_integer_volume_quantiles_and_poc_ties():
    huge=2**53+1
    out=f.histogram_features([100,101,102],[huge,0,huge],[0,0,0],[0,0,0],100000000)
    assert out['volume_contracts']==2*huge
    assert out['poc_usd']==10
    assert out['median_usd']==10
    assert out['q75_usd']==pytest.approx(10.2)
    assert np.isnan(out['sell_poc_usd'])
    assert out['max_level_delta_contracts']==huge
    with pytest.raises(ValueError,match='NOT_UNIQUE'):
        f.histogram_features([1,1],[1,1],[0,0],[0,0],100000000)


def test_timeframe_masks_do_not_bridge_unknowns_gaps_rolls_or_partial_bars():
    a=list(minute_inputs(22))
    a[3]['delta'][1]=False; a[3]['cvd'][1]=False
    a[3]['volume'][7]=False
    a[1][13:]=1
    s,_=f.timeframe_skeleton(*a,'M5',100000000)
    assert s['volume_valid'].tolist()==[True,False,False,True,False]
    assert s['signed_valid'].tolist()==[False,False,False,True,False]
    cvd=f.relative_cvd(*a[:2],a[3],a[2])
    assert cvd['cvd_segment_delta_contracts'][0]==2
    assert np.isnan(cvd['cvd_segment_delta_contracts'][1])
    assert cvd['cvd_segment_delta_contracts'][2]==2
    assert cvd['cvd_segment_delta_contracts'][13]==2


def test_canonical_h4_phase_and_m5_join_reject_future_or_stale_bar():
    from gx1.scripts.gc_order_flow_features_v1 import bind_m5_rows
    a=minute_inputs(300)
    surface,_=f.timeframe_skeleton(*a,'H4',100000000)
    assert int(surface['bar_start_ns'][0]//t.HOUR_NS%24)==22
    # First H4 bar is partial (study begins at midnight), not valid.
    assert not surface['volume_valid'][0]
    table=bind_m5_rows(a[0][np.array([0,115,120,235,240])],np.arange(5),np.array(['train']*5),{'H4':surface})
    labels=table['h4_bar_start_ns'].to_pylist()
    assert labels[0] is None
    assert labels[1]==surface['bar_start_ns'][0]
    assert labels[2]==labels[1]
    assert labels[3]==labels[1]
    assert labels[4]==labels[1]


def test_validated_callback_and_sparse_ladder_agree_on_partial_batches(tmp_path,monkeypatch):
    a=minute_inputs(10)
    minutes,selected,metrics,masks,_=a
    monkeypatch.setattr(t,'WINDOW_END_NS',t.STUDY_START_NS+10*t.MINUTE_NS)
    monkeypatch.setattr(c,'CSV_BLOCK_BYTES',600)
    source_path=tmp_path/'input.csv.gz'
    rows=[]
    for i in range(10):
        for j,side,size in ((1,'B',3),(2,'A',1),(3,'N',2)):
            rows.append(dict(ts_recv=int(minutes[i]+j),ts_event=int(minutes[i]+j),
                instrument_id=42,publisher_id=1,symbol='GCM4',action='T',side=side,
                size=size,price=(100+i%2)*100000000,flags=0,sequence=i*10+j))
        # F must not enter the collector or volume a second time.
        rows.append(dict(rows[-1],action='F',size=999,sequence=i*10+4))
    with gzip.open(source_path,'wt',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=c.COLUMNS)
        w.writeheader(); w.writerows(rows)
    path=tmp_path/'ladder.parquet'
    collector=f.TradeLadderCollector(path,selected,masks['volume'],0,100000000)
    result=c.scan_symbol([dict(path=str(source_path),raw_symbol='GCM4',instrument_id=42,publisher_id=1,
        start_ns=t.STUDY_START_NS,end_ns=t.WINDOW_END_NS,max_records=len(rows))],100000000,on_batch=collector)
    collector.close()
    assert collector.rows==30
    assert result['volume'].tolist()==[6]*10
    masks['delta'][:]=False; masks['cvd'][:]=False
    surface,inv=f.timeframe_skeleton(minutes,selected,result,masks,collector.ohlc,'M5',100000000)
    f.consume_ladder(path,{'M5':surface},{'M5':inv},100000000)
    assert surface['poc_usd'].tolist()==[10,pytest.approx(10.1)]
    assert np.isnan(surface['buy_poc_usd']).all()
    assert surface['volume'].tolist()==[30,30]
    assert surface['open_usd'][0]==10 and surface['close_usd'][1]==pytest.approx(10.1)


def test_proxy_uses_existing_events_and_is_prefix_and_future_invariant():
    h=np.array([9,10,11,14,12,11,10,11,15,14,13,12,11,10,9,8.],dtype=float)
    l=h-4; close=h-1; open_=close+0.5; v=np.arange(len(h))+10
    # A separately confirmed low precedes the break and enables structure.
    l[4]=3; close[8]=14.5; open_[8]=14.75
    out=replay_volume_order_block_proxies_v1(open_,h,l,close,v)
    event,_=replay_smc_causal_structure_v2(h,l,close)
    mask=event['structure_available'].astype(bool)
    np.testing.assert_array_equal(out['bos_up'][mask],event['bos_up'][mask])
    for cut in (8,10,13):
        prefix=replay_volume_order_block_proxies_v1(open_[:cut],h[:cut],l[:cut],close[:cut],v[:cut])
        for name in out:
            np.testing.assert_array_equal(prefix[name],out[name][:cut])
    mutated=replay_volume_order_block_proxies_v1(open_+np.r_[np.zeros(10),np.ones(6)],
        h+np.r_[np.zeros(10),np.ones(6)],l+np.r_[np.zeros(10),np.ones(6)],
        close+np.r_[np.zeros(10),np.ones(6)],v)
    for name in out:
        np.testing.assert_array_equal(out[name][:10],mutated[name][:10])
    assert np.isnan(out['ob_bull_active'][:6]).all()
    # Creating candle cannot be its own origin or retrospective zone.
    firing=np.flatnonzero(out['ob_bull_created']==1)
    assert len(firing)>0
    i=firing[0]
    assert out['ob_bull_origin_volume_contracts'][i]==v[i-1]
    assert out['ob_bull_low'][i]==l[i-1]


def test_invalid_windows_keep_feature_nulls_and_reset_cvd_and_structure():
    a=list(minute_inputs(20))
    a[3]['volume'][5]=False; a[3]['delta'][5]=False; a[3]['cvd'][5]=False
    s,inv=f.timeframe_skeleton(*a,'M5',100000000)
    cvd=f.relative_cvd(a[0],a[1],a[3],a[2])
    f.add_structure_and_cvd(s,cvd,inv,a[0])
    assert np.isnan(s['volume'][1])
    assert np.isnan(s['delta_contracts'][1])
    assert np.isnan(s['bos_up'][1])
    assert s['cvd_segment_minutes'][0]==5
    assert s['cvd_segment_minutes'][2]==9
    assert np.isnan(s['ob_bull_active'][2])


def test_basis_and_native_row_join_use_only_the_same_closed_minute(tmp_path):
    from gx1.scripts.gc_order_flow_features_v1 import add_basis,bind_m5_rows,previous
    a=minute_inputs(10)
    surface,inv=f.timeframe_skeleton(*a,'M5',100000000)
    cvd=f.relative_cvd(a[0],a[1],a[3],a[2])
    f.add_structure_and_cvd(surface,cvd,inv,a[0])
    part=tmp_path/'spot.parquet'; manifest=tmp_path/'manifest.json'
    manifest.write_text('{}')
    pq.write_table(pa.table({'time':pa.array(a[0],type=pa.timestamp('ns',tz='UTC')),
        'bid_close':np.arange(10)+100.,'ask_close':np.arange(10)+102.}),part)
    spec={'inputs':{'oanda_manifest':{'path':str(manifest),'sha256':previous.sha256_file(manifest),
        'partitions':[{'path':str(part),'sha256':previous.sha256_file(part)}]}}}
    add_basis(spec,{'M5':surface},a[0])
    table=bind_m5_rows(a[0][[0,5]],np.array([13,19]),np.array(['train','val']),{'M5':surface})
    assert table['native_row_index'].to_pylist()==[13,19]
    assert table['m5_spot_basis_bps'][0].as_py()==1e4*(10./105.-1)
    assert table['m5_spot_basis_bps'][1].as_py()==1e4*(10./110.-1)
    assert table['gc_m5_abc_input_valid'].to_pylist()==[True,True]


def test_raw_reference_reducer_detects_wrong_profile_and_no_references():
    from gx1.scripts.gc_order_flow_features_v1 import verify_real_references
    raw={0:[(100,3,'B'),(101,1,'A'),(100,2,'B')]}
    stats=f.histogram_features([100,101],[5,0],[0,1],[0,0],100000000)
    surface={k:np.array([v]) for k,v in stats.items()}
    surface.update(bar_start_ns=np.array([t.STUDY_START_NS]),signed_valid=np.array([True]),
        volume=np.array([6]),open_usd=np.array([10.]),close_usd=np.array([10.]),
        high_usd=np.array([101*0.1]),low_usd=np.array([10.]),delta_contracts=np.array([4]))
    assert verify_real_references(raw,surface,100000000)['raw_reference_bars']==1
    surface['poc_usd'][0]=10.1
    with pytest.raises(RuntimeError,match='RAW_REFERENCE_PARITY_FAILED:poc_usd'):
        verify_real_references(raw,surface,100000000)
    with pytest.raises(RuntimeError,match='NO_REAL_REFERENCE_BARS'):
        verify_real_references({},surface,100000000)


def test_real_prefix_evidence_is_strict_json_serializable_and_discloses_suffix():
    import json
    from gx1.scripts.gc_order_flow_features_v1 import verify_real_prefixes
    a=minute_inputs(30)
    surface,_=f.timeframe_skeleton(*a,'M5',100000000)
    checks=verify_real_prefixes({'M5':surface},a[0],a[1],a[3],a[2],100000000)
    encoded=json.dumps(checks,allow_nan=False)
    assert json.loads(encoded)[0]['future_suffix_bars']==3


def test_ohlc_recovery_callback_does_not_create_a_second_ladder():
    a=minute_inputs(2)
    collector=f.TradeLadderCollector(None,a[1],a[3]['volume'],0,100000000,reference_bars=[0])
    data={
        'price':pa.array([10000000000,10100000000,9900000000]),
        'size':pa.array([3,1,2],type=pa.uint32()),'side':pa.array(['B','A','B'])}
    collector(data,np.array([0,0,1]),np.array([True,True,True]))
    collector.close()
    assert collector.ohlc.tolist()==[[100,101,100,101],[99,99,99,99]]
    assert collector.rows==3
    assert collector.reference_trades[0]==[(100,3,'B'),(101,1,'A'),(99,2,'B')]


def test_observed_d1_keeps_fixed_clock_gap_and_exact_last_accepted_minute():
    a = list(minute_inputs(2880))
    # The only full D1 bucket is [1320, 2760). Missing its final hour must
    # remain incomplete, while its 1380 admitted minutes retain their value.
    rejected = slice(2700, 2760)
    for key in ('volume', 'delta', 'cvd'):
        a[3][key][rejected] = False
    a[2]['unknown'][1325] = 2
    a[2]['volume'][1325] += 2
    a[3]['delta'][1325] = a[3]['cvd'][1325] = False
    a[4][rejected] = [999, 9999, 1, 999]
    surface, _ = f.timeframe_skeleton(*a, 'D1', 100000000)
    assert surface['volume_valid'].tolist() == [False, False, False]
    assert surface['signed_valid'].tolist() == [False, False, False]
    assert surface['observed_valid'].tolist() == [False, True, False]
    assert surface['observed_expected_clock_minutes'][1] == 1440
    assert surface['observed_clock_minutes'][1] == 1440
    assert surface['observed_accepted_minutes'][1] == 1380
    assert surface['observed_signed_minutes'][1] == 1379
    assert surface['observed_coverage_fraction'][1] == 23 / 24
    assert surface['observed_volume'][1] == 5522
    assert surface['observed_buy_known'][1] == 4140
    assert surface['observed_sell_known'][1] == 1380
    assert surface['observed_unknown'][1] == 2
    assert surface['observed_known_delta_contracts'][1] == 2760
    assert not surface['observed_signed_complete'][1]
    assert surface['observed_high_usd'][1] == pytest.approx(10.1)
    assert surface['observed_low_usd'][1] == pytest.approx(9.9)
    assert int(surface['observed_first_minute_ns'][1]) == int(a[0][1320])
    assert int(surface['observed_last_minute_ns'][1]) == int(a[0][2699])
    assert np.isnan(surface['observed_close_usd'][0])
    assert np.isnan(surface['observed_last_minute_ns'][2])


def _write_admitted_ladder(path, a, *, corrupt_sides=False):
    take = np.flatnonzero(a[3]['volume'])
    buy = a[2]['buy'][take].copy()
    sell = a[2]['sell'][take].copy()
    if corrupt_sides:
        buy[0] += 1
        sell[0] -= 1
    schema = pa.schema([('minute_index', pa.int32()), ('price_tick', pa.int64()),
        ('buy', pa.uint64()), ('sell', pa.uint64()), ('unknown', pa.uint64())])
    # Multiple row groups exercise state carried across physical batch boundaries.
    with pq.ParquetWriter(path, schema) as writer:
        for indices in np.array_split(np.arange(len(take)), 2):
            writer.write_table(pa.table({'minute_index': take[indices].astype(np.int32),
                'price_tick': np.full(len(indices), 100, np.int64),
                'buy': buy[indices], 'sell': sell[indices],
                'unknown': a[2]['unknown'][take[indices]]}, schema=schema))


def test_observed_profile_retains_unknown_bins_without_fabricating_signed_flow(tmp_path):
    a = list(minute_inputs(10))
    for key in ('volume', 'delta', 'cvd'):
        a[3][key][1] = False
    a[2]['unknown'][2] = 2
    a[2]['volume'][2] += 2
    a[3]['delta'][2] = a[3]['cvd'][2] = False
    path = tmp_path / 'admitted.parquet'
    _write_admitted_ladder(path, a)
    surface, inverse = f.timeframe_skeleton(*a, 'M5', 100000000)
    f.consume_ladder(path, {'M5': surface}, {'M5': inverse}, 100000000)
    assert surface['observed_valid'].tolist() == [True, True]
    assert surface['observed_volume'].tolist() == [18, 20]
    assert surface['observed_buy_known'].tolist() == [12, 15]
    assert surface['observed_sell_known'].tolist() == [4, 5]
    assert surface['observed_unknown'].tolist() == [2, 0]
    assert surface['observed_signed_complete'].tolist() == [False, True]
    assert surface['observed_known_max_level_delta_contracts'].tolist() == [8, 10]
    assert surface['observed_buy_known_poc_usd'].tolist() == [10, 10]
    assert surface['observed_sell_known_poc_usd'].tolist() == [10, 10]
    assert surface['observed_poc_usd'].tolist() == [10, 10]
    assert np.isnan(surface['poc_usd'][0])
    # Fully supported strict values keep the exact existing formula/bytes.
    for name in f.PROFILE_FIELDS:
        np.testing.assert_array_equal(surface[name][1:], surface['observed_' + name][1:])
    for name, target in f.OBSERVED_FOOTPRINT_FIELDS.items():
        np.testing.assert_array_equal(surface[name][1:], surface[target][1:])
    cvd = f.relative_cvd(a[0], a[1], a[3], a[2])
    f.add_structure_and_cvd(surface, cvd, {'M5': inverse}, a[0])
    assert np.isnan(surface['volume'][0]) and np.isnan(surface['delta_contracts'][0])
    assert surface['observed_volume'][0] == 18
    assert surface['observed_known_delta_contracts'][0] == 8


def test_observed_profile_rejects_side_corruption_even_when_total_volume_matches(tmp_path):
    a = minute_inputs(5)
    path = tmp_path / 'corrupt.parquet'
    _write_admitted_ladder(path, a, corrupt_sides=True)
    surface, inverse = f.timeframe_skeleton(*a, 'M5', 100000000)
    with pytest.raises(RuntimeError, match='OBSERVED_SIDE_VOLUME_PARITY_FAILED:buy_known'):
        f.consume_ladder(path, {'M5': surface}, {'M5': inverse}, 100000000)


def test_observed_domain_rejects_roll_unselected_empty_and_partial_buckets():
    a = list(minute_inputs(22))
    a[1][3:5] = 1
    a[1][5] = -1
    for key in ('volume', 'delta', 'cvd'):
        a[3][key][10:15] = False
    surface, _ = f.timeframe_skeleton(*a, 'M5', 100000000)
    assert surface['observed_contract_valid'].tolist() == [False, False, True, True, True]
    assert surface['observed_valid'].tolist() == [False, False, False, True, False]
    assert surface['observed_accepted_minutes'].tolist() == [5, 5, 0, 5, 2]
    assert np.isnan(surface['observed_volume'][[0, 1, 2, 4]]).all()
    assert np.isnan(surface['observed_signed_minute_fraction'][2])


def _observed_structure_surface():
    h = np.array([9, 10, 11, 14, 12, 11, 10, 11, 15, 14, 13, 12, 11, 10, 9, 8.])
    low = h - 4
    close = h - 1
    opening = close + 0.5
    low[4] = 3
    close[8] = 14.5
    opening[8] = 14.75
    start = t.STUDY_START_NS + np.arange(len(h)) * 5 * t.MINUTE_NS
    return {'bar_start_ns': start, 'available_ns': start + 5 * t.MINUTE_NS,
        'contract_index': np.zeros(len(h), np.int16), 'observed_valid': np.ones(len(h), bool),
        'observed_contract_valid': np.ones(len(h), bool),
        'observed_open_usd': opening, 'observed_high_usd': h, 'observed_low_usd': low,
        'observed_close_usd': close, 'observed_volume': np.arange(len(h)) + 10.,
        'observed_coverage_fraction': np.full(len(h), 0.8),
        'observed_first_minute_ns': start.astype(float),
        'observed_last_minute_ns': (start + 4 * t.MINUTE_NS).astype(float),
        'bos_up': np.full(len(h), 17.)}


def test_observed_structure_uses_existing_owner_and_preserves_strict_and_prefix():
    surface = _observed_structure_surface()
    expected = replay_volume_order_block_proxies_v1(*[surface['observed_' + name] for name in
        ('open_usd', 'high_usd', 'low_usd', 'close_usd', 'volume')])
    f.add_observed_structure(surface)
    for name, value in expected.items():
        np.testing.assert_array_equal(surface['observed_proxy_' + name], value)
    np.testing.assert_array_equal(surface['bos_up'], np.full(16, 17.))
    np.testing.assert_array_equal(surface['observed_coverage_fraction'], np.full(16, 0.8))
    for cut in (8, 10, 13):
        prefix = {name: value[:cut].copy() for name, value in _observed_structure_surface().items()}
        f.add_observed_structure(prefix)
        for name in expected:
            np.testing.assert_array_equal(prefix['observed_proxy_' + name], surface['observed_proxy_' + name][:cut])


def test_observed_structure_keeps_partial_gap_pivots_but_censors_missing_candle_ob_state():
    surface = _observed_structure_surface()
    surface['observed_last_minute_ns'][5] -= t.MINUTE_NS
    surface['observed_valid'][9] = False
    surface['contract_index'][12:] = 1
    f.add_observed_structure(surface)
    assert surface['observed_proxy_boundary_gap_minutes'][6] == 1
    assert surface['observed_proxy_boundary_gap_minutes'][10] == 5
    assert surface['observed_proxy_elapsed_minutes'][10] == 10
    assert surface['observed_proxy_skipped_candles'][10] == 1
    # The missing candle9 does not erase the confirmed observed pivot history.
    for rows in (np.r_[np.arange(9),10,11], np.arange(12,16)):
        skipped = np.r_[False,np.diff(surface['bar_start_ns'][rows]) != 5*t.MINUTE_NS]
        expected = replay_volume_order_block_proxies_v1(*[surface['observed_' + name][rows] for name in
            ('open_usd', 'high_usd', 'low_usd', 'close_usd', 'volume')], zone_reset=skipped)
        for name, value in expected.items():
            np.testing.assert_array_equal(surface['observed_proxy_' + name][rows], value)
    assert np.isnan(surface['observed_proxy_bos_up'][9])
    assert np.isnan(surface['observed_proxy_ob_bull_active'][10:12]).all()
    assert not surface['observed_proxy_ob_bull_state_valid'][10:12].any()
    assert np.isnan(surface['observed_proxy_boundary_gap_minutes'][12])


@pytest.mark.parametrize('duration_minutes',[240,1440])
def test_h4_d1_observed_pivots_survive_measured_partial_boundary_gaps(duration_minutes):
    surface = _observed_structure_surface()
    labels=t.STUDY_START_NS+np.arange(16)*duration_minutes*t.MINUTE_NS
    surface['bar_start_ns']=labels
    surface['available_ns']=labels+duration_minutes*t.MINUTE_NS
    surface['observed_first_minute_ns']=(labels+60*t.MINUTE_NS).astype(float)
    surface['observed_last_minute_ns']=(labels+(duration_minutes-1)*t.MINUTE_NS).astype(float)
    expected=replay_volume_order_block_proxies_v1(*[surface['observed_'+name] for name in
        ('open_usd','high_usd','low_usd','close_usd','volume')])
    f.add_observed_structure(surface)
    np.testing.assert_array_equal(surface['observed_proxy_bos_up'],expected['bos_up'])
    assert np.isfinite(surface['observed_proxy_bos_up']).any()
    assert (surface['observed_proxy_boundary_gap_minutes'][1:]==60).all()
    assert (surface['observed_proxy_elapsed_minutes'][1:]==duration_minutes).all()
    assert (surface['observed_proxy_skipped_candles'][1:]==0).all()
    np.testing.assert_array_equal(surface['bos_up'],np.full(16,17.))


def test_observed_structure_does_not_hide_contract_excursion_in_a_missing_candle():
    surface=_observed_structure_surface()
    surface['observed_valid'][9]=False
    surface['contract_index'][9]=1  # observed endpoints8/10 both contract0
    f.add_observed_structure(surface)
    expected=replay_volume_order_block_proxies_v1(*[surface['observed_'+name][10:] for name in
        ('open_usd','high_usd','low_usd','close_usd','volume')])
    for name,value in expected.items():
        np.testing.assert_array_equal(surface['observed_proxy_'+name][10:],value)
    assert np.isnan(surface['observed_proxy_boundary_gap_minutes'][10])


def test_order_block_zone_reset_preserves_pivots_and_censors_until_new_observed_origin_and_creation():
    surface=_observed_structure_surface()
    arrays=[surface['observed_'+name] for name in ('open_usd','high_usd','low_usd','close_usd','volume')]
    arrays=[np.r_[a,a+20] for a in arrays]
    original=replay_volume_order_block_proxies_v1(*arrays)
    reset=np.zeros(32,bool);reset[9]=True
    censored=replay_volume_order_block_proxies_v1(*arrays,zone_reset=reset)
    for name in ('bos_up','bos_down','sweep_up_event','sweep_down_event','structure_available'):
        np.testing.assert_array_equal(censored[name],original[name])
    creations=np.flatnonzero(censored['ob_bull_created'][10:]==1)+10
    assert len(creations)
    born=int(creations[0])
    assert np.isnan(censored['ob_bull_active'][9:born]).all()
    assert np.isnan(censored['ob_bull_first_retouch'][9:born]).all()
    assert np.isnan(censored['ob_bull_low'][9:born]).all()
    assert censored['ob_bull_active'][born]==1
    assert censored['ob_bull_first_retouch'][born]==0
    assert censored['ob_bull_age_bars'][born]==0
    assert censored['ob_bull_origin_volume_contracts'][born]==arrays[4][born-1]
    for cut in (8,10,17,25,31):
        prefix=replay_volume_order_block_proxies_v1(*[a[:cut] for a in arrays],zone_reset=reset[:cut])
        for name,value in censored.items():
            np.testing.assert_array_equal(prefix[name],value[:cut])
    # An origin from before the gap cannot create a zone on the gap row.
    reset=np.zeros(32,bool);reset[8]=True
    no_origin=replay_volume_order_block_proxies_v1(*arrays,zone_reset=reset)
    assert no_origin['bos_up'][8]==1
    assert np.isnan(no_origin['ob_bull_active'][8])
    zeros=replay_volume_order_block_proxies_v1(*arrays,zone_reset=np.zeros(32,bool))
    for name,value in original.items():
        np.testing.assert_array_equal(zeros[name],value)
    for invalid in (np.zeros(31,bool),np.full(32,np.nan),np.full(32,2)):
        with pytest.raises(ValueError,match='ZONE_RESET_INVALID'):
            replay_volume_order_block_proxies_v1(*arrays,zone_reset=invalid)


def test_observed_structure_candidate_state_masks_and_gap_clocks_are_explicit():
    declared=f.completion_native_candidates_v1()
    for side in ('bull','bear'):
        rows=[row for row in declared['fields'] if row['column']=='observed_proxy_ob_'+side+'_first_retouch']
        assert len(rows)==5
        assert all(row['validity_column']=='observed_proxy_ob_'+side+'_state_valid' for row in rows)
    for column in ('observed_proxy_boundary_gap_minutes','observed_proxy_elapsed_minutes','observed_proxy_skipped_candles'):
        rows=[row for row in declared['fields'] if row['column']==column]
        assert len(rows)==5
        assert all(row['validity_column']=='observed_valid' and row['source_clock_column']=='available_ns' for row in rows)
    assert 'first OBSERVED' in declared['observed_structure_semantics']


@pytest.mark.parametrize('observed_only', [False, True])
def test_contract_major_ladder_keeps_late_bucket_continuation_before_earlier_contract(tmp_path, monkeypatch, observed_only):
    # raw001 writes contract-major rows. The later contract's open TF bucket
    # continues across a physical batch before the next contract goes back in
    # calendar time. Sorting unique bars would flush that open bucket too soon.
    a = list(minute_inputs(4320))
    a[1][2760:4200] = 1
    admitted = np.zeros(4320, bool)
    admitted[1320:4200] = True
    if observed_only:
        admitted[np.arange(4320) % 5 == 1] = False
    for key in ('volume', 'delta', 'cvd'):
        a[3][key][:] = admitted
    a[2]['unknown'][1320] = 2
    a[2]['volume'][1320] += 2
    a[3]['delta'][1320] = a[3]['cvd'][1320] = False
    later = np.flatnonzero(admitted & (a[1] == 1))
    earlier = np.flatnonzero(admitted & (a[1] == 0))
    source_minutes = np.r_[later, earlier]
    rows = np.repeat(source_minutes, 3)
    buy = np.ones(len(rows), np.uint64)
    sell = np.tile(np.array([0, 0, 1], np.uint64), len(source_minutes))
    unknown = np.zeros(len(rows), np.uint64)
    unknown[::3] = a[2]['unknown'][source_minutes]
    schema = pa.schema([('minute_index', pa.int32()), ('price_tick', pa.int64()),
        ('buy', pa.uint64()), ('sell', pa.uint64()), ('unknown', pa.uint64())])
    path = tmp_path / 'contract_major.parquet'
    # Repeated price100 fragments are intentionally split across row groups.
    with pq.ParquetWriter(path, schema) as writer:
        for part in np.array_split(np.arange(len(rows)), 3):
            writer.write_table(pa.table({'minute_index': rows[part].astype(np.int32),
                'price_tick': np.tile(np.array([100, 100, 101], np.int64), len(source_minutes))[part],
                'buy': buy[part], 'sell': sell[part], 'unknown': unknown[part]}, schema=schema))
    original_reader = pq.ParquetFile
    class FragmentedReader:
        def __init__(self, source):
            self.source = original_reader(source)
        def iter_batches(self, **kwargs):
            kwargs['batch_size'] = 1001
            yield from self.source.iter_batches(**kwargs)
    monkeypatch.setattr(f.pq, 'ParquetFile', FragmentedReader)
    surfaces, maps = {}, {}
    for timeframe in ('M5', 'M15', 'H1', 'H4', 'D1'):
        surfaces[timeframe], maps[timeframe] = f.timeframe_skeleton(*a, timeframe, 100000000)
    f.consume_ladder(path, surfaces, maps, 100000000)
    for timeframe, surface in surfaces.items():
        observed = surface['observed_valid']
        assert observed.any(), timeframe
        assert np.isfinite(surface['observed_vwap_usd'][observed]).all(), timeframe
        assert (surface['observed_poc_usd'][observed] == 10).all(), timeframe
        accepted = surface['observed_accepted_minutes'][observed]
        assert np.array_equal(surface['observed_buy_known'][observed], 3 * accepted), timeframe
        assert np.array_equal(surface['observed_sell_known'][observed], accepted), timeframe
        assert np.array_equal(surface['observed_known_delta_contracts'][observed], 2 * accepted), timeframe
        if observed_only:
            assert not surface['volume_valid'].any(), timeframe
            assert np.isnan(surface['poc_usd']).all(), timeframe
        else:
            strict = surface['volume_valid']
            assert strict.any(), timeframe
            for field in f.PROFILE_FIELDS:
                np.testing.assert_array_equal(surface[field][strict], surface['observed_' + field][strict])


def test_executed_trade_average_is_ratio_of_aggregate_quantity_not_displayed_orders():
    a = list(minute_inputs(5))
    a[2]['volume'][:] = [4, 4, 4, 4, 24]
    a[2]['buy'][:] = [3, 3, 3, 3, 23]
    a[2]['trades'][:] = [1, 1, 1, 1, 12]
    surface, _ = f.timeframe_skeleton(*a, 'M5', 100000000)
    assert surface['observed_avg_executed_trade_contracts'][0] == 2.5
    assert surface['observed_volume'][0] == 40
    assert surface['observed_trades'][0] == 16


def test_all_timeframe_value_area_uses_contiguous70percent_not_q25_q75(tmp_path):
    a = list(minute_inputs(2880))
    a[2]['buy'][:] = 50; a[2]['sell'][:] = 50
    a[2]['volume'][:] = 100; a[2]['trades'][:] = 3
    a[4][:] = [100, 102, 100, 102]
    minutes = np.repeat(np.arange(2880, dtype=np.int32), 3)
    path = tmp_path / 'value_area.parquet'
    pq.write_table(pa.table({'minute_index': minutes,
        'price_tick': np.tile(np.array([100, 101, 102], np.int64), 2880),
        'buy': np.tile(np.array([15, 20, 15], np.uint64), 2880),
        'sell': np.tile(np.array([15, 20, 15], np.uint64), 2880),
        'unknown': np.zeros(len(minutes), np.uint64)}), path)
    surfaces, maps = {}, {}
    for tf in ('M5','M15','H1','H4','D1'):surfaces[tf], maps[tf] = f.timeframe_skeleton(*a, tf, 100000000)
    f.consume_ladder(path, surfaces, maps, 100000000)
    for surface in surfaces.values():
        observed = surface['observed_valid']
        assert observed.any()
        np.testing.assert_allclose(surface['observed_val_usd'][observed], 10)
        np.testing.assert_allclose(surface['observed_vah_usd'][observed], 10.1)
        np.testing.assert_allclose(surface['observed_q75_usd'][observed], 10.2)


def test_known_cumulative_preserves_unknown_and_resets_clock_gap_date_and_roll():
    a = list(minute_inputs(10))
    a[2]['unknown'][1] = 3; a[2]['volume'][1] += 3
    a[3]['delta'][1] = a[3]['cvd'][1] = False
    a[3]['volume'][3] = False
    a[1][7:] = 1
    known = f.known_cumulative_contributions_v1(a[0], a[1], a[3], a[2])
    assert known['known_delta_contracts'][1] == 4
    assert known['unknown_contracts'][1] == 3
    assert known['unknown_fraction'][1] == 3 / 11
    assert np.isnan(known['volume_contracts'][3])
    assert known['observed_minutes'][4] == 1
    assert known['observed_minutes'][7] == 1
    surface, _ = f.timeframe_skeleton(*a, 'M5', 100000000)
    f.add_known_cumulative_contributions_v1(surface, known, a[0])
    assert surface['observed_known_cumulative_observed_minutes'][0] == 1
    for name, values in known.items():
        prefix = f.known_cumulative_contributions_v1(a[0][:6], a[1][:6], {k:v[:6] for k,v in a[3].items()}, {k:v[:6] for k,v in a[2].items()})
        np.testing.assert_array_equal(prefix[name], values[:6])


def test_completion_native_candidates_have_one_explicit_owner_mask_clock_and_metadata_exclusion():
    declared = f.completion_native_candidates_v1()
    fields = declared['fields']
    assert len({(row['table'], row['column']) for row in fields}) == len(fields)
    assert not set(row['column'] for row in fields).intersection(declared['metadata_columns'])
    assert set(row['owner'] for row in fields) <= {'momentum_flow_encoder','smc_liquidity_encoder',
        'vol_compression_encoder','session_regime_encoder','price_action_candle_encoder'}
    row = next(row for row in fields if row['column'] == 'observed_event_matching_clock_mean_ns')
    assert row['validity_column'] == 'observed_event_matching_clock_valid'
    assert row['owner'] == 'session_regime_encoder'
    profile = next(row for row in fields if row['column'] == 'observed_profile_rolling15_poc_usd')
    assert profile['source_clock_column'] == 'observed_profile_snapshot_available_ns'


@pytest.mark.parametrize('tf,duration_minutes', [('M5',5),('M15',15),('H1',60),('H4',240),('D1',1440)])
def test_observed_basis_matches_exact_last_gc_m1_and_preserves_strict_default(tmp_path,tf,duration_minutes):
    from gx1.scripts.gc_order_flow_features_v1 import add_basis,previous
    origin=t.STUDY_START_NS
    minutes=origin+np.arange(10,dtype=np.int64)*t.MINUTE_NS
    part=tmp_path/'spot.parquet';manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    # Minute4 is absent: its GC close must not use minute3 or the TF ending quote.
    quoted=np.array([0,1,2,3,5,6,7,8,9])
    pq.write_table(pa.table({'time':pa.array(minutes[quoted],type=pa.timestamp('ns',tz='UTC')),
        'bid_close':100.+quoted,'ask_close':102.+quoted}),part)
    plan={'inputs':{'oanda_manifest':{'path':str(manifest),'sha256':previous.sha256_file(manifest),
        'partitions':[{'path':str(part),'sha256':previous.sha256_file(part)}]}}}
    surface={'bar_start_ns':np.full(3,origin,np.int64),
        'available_ns':np.full(3,origin+duration_minutes*t.MINUTE_NS,np.int64),
        'volume_valid':np.array([duration_minutes==5,False,False]),'close_usd':np.array([10.,np.nan,np.nan]),
        'observed_valid':np.array([True,True,False]),'observed_close_usd':np.array([10.,11.,np.nan]),
        'observed_last_minute_ns':np.array([minutes[2],minutes[4],np.nan])}
    strict={key:value.copy() for key,value in surface.items()}
    add_basis(plan,{tf:strict},minutes)
    hashes=add_basis(plan,{tf:surface},minutes,include_observed=True)
    np.testing.assert_array_equal(strict['spot_basis_bps'],surface['spot_basis_bps'])
    assert 'observed_spot_basis_bps' not in strict
    assert surface['observed_spot_basis_valid'].tolist()==[True,False,False]
    assert surface['observed_spot_basis_bps'][0]==pytest.approx(1e4*(10./103.-1))
    assert np.isnan(surface['observed_spot_basis_bps'][1:]).all()
    assert surface['observed_spot_basis_source_available_ns'][0]==minutes[2]+t.MINUTE_NS
    assert surface['observed_spot_basis_age_minutes'][0]==duration_minutes-3
    assert {str(path) for path,_ in hashes}=={str(manifest),str(part)}
    surface['observed_last_minute_ns'][0]=surface['available_ns'][0]
    with pytest.raises(RuntimeError,match='OBSERVED_BASIS_SOURCE_CLOCK_INVALID'):
        add_basis(plan,{tf:surface},minutes,include_observed=True)


def test_observed_basis_candidates_have_explicit_mask_clock_owner_without_duplicate_mask_value():
    declared=f.completion_native_candidates_v1()
    records=[row for row in declared['fields'] if row['column'].startswith('observed_spot_basis')]
    assert len(records)==10
    assert {row['column'] for row in records}=={'observed_spot_basis_bps','observed_spot_basis_age_minutes'}
    assert all(row['owner']=='momentum_flow_encoder' and row['validity_column']=='observed_spot_basis_valid'
        and row['source_clock_column']=='observed_spot_basis_source_available_ns' for row in records)
    assert 'observed_spot_basis_source_available_ns' in declared['metadata_columns']


def test_observed_structure_requires_explicit_contract_domain_and_valid_source_clock():
    surface=_observed_structure_surface()
    del surface['observed_contract_valid']
    with pytest.raises(ValueError,match='CONTRACT_METADATA_REQUIRED'):
        f.add_observed_structure(surface)
    surface=_observed_structure_surface();surface['observed_contract_valid'][2]=False
    with pytest.raises(ValueError,match='CONTRACT_DOMAIN'):
        f.add_observed_structure(surface)
    for field,value in (
        ('bar_start_ns',np.zeros((1,16),np.int64)),
        ('observed_contract_valid',np.ones(15,bool)),
        ('observed_contract_valid',np.ones(16,np.int64)),
        ('bar_start_ns',_observed_structure_surface()['bar_start_ns'].astype(float)),
        ('available_ns',_observed_structure_surface()['available_ns']+np.arange(16)),
        ('observed_last_minute_ns',_observed_structure_surface()['available_ns'].astype(float))):
        surface=_observed_structure_surface();surface[field]=value
        with pytest.raises(ValueError,match='STRUCTURE_CLOCK'):
            f.add_observed_structure(surface)


def test_streamed_strict_profiles_match_original_recipe_through_gap_and_roll(tmp_path):
    a=minute_inputs(120)
    # One partially observed M5 is rejected by the original strict domain.
    a[3]['volume'][16]=False;a[3]['delta'][16]=False;a[3]['cvd'][16]=False
    a[1][65:]=1
    surface,inverse=f.timeframe_skeleton(*a,'M5',100000000)
    path=tmp_path/'ladder.parquet';_write_admitted_ladder(path,a)
    strict=f.dynamic_profile_arrays_v2(surface['bar_start_ns'])
    observed=f.dynamic_profile_arrays_v2(surface['bar_start_ns'])
    evidence=f.consume_ladder(path,{'M5':surface},{'M5':inverse},100000000,
        strict_profile_output=strict,profile_output=observed,profile_state_max_bytes=8*1024**2)
    hist=[{100:20} if valid else {} for valid in surface['volume_valid']]
    original=f.dynamic_profiles_v2(surface['bar_start_ns'],hist,surface['volume_valid'],surface['contract_index'],surface['close_usd'],0.1)
    for name,value in strict.items():np.testing.assert_array_equal(value,original[name],err_msg=name)
    assert evidence['strict_profile_stream_peak_state_bytes']>0
    assert evidence['combined_profile_peak_upper_bound_bytes']<=8*1024**2
    # The partial bar retains an honest observed profile but is not a strict alias.
    assert not strict['session_observed_valid'][3] and observed['session_observed_valid'][3]


@pytest.mark.parametrize('tf,duration',[('M5',5),('M15',15),('H1',60),('H4',240),('D1',1440)])
def test_trusted_trade_sizes_keep_exact_side_counts_population_dispersion_and_volume_mask(tf,duration):
    a=minute_inputs(3*duration)
    a[3]['volume'][duration+2]=False
    stats={name:np.zeros(len(a[0]),np.uint64) for name in f.TRUSTED_TRADE_SIZE_FIELDS_V1}
    stats['buy_trade_records'][:]=1;stats['sell_trade_records'][:]=1
    stats['size_sum_contracts'][:]=4;stats['size_sum_squared_contracts2'][:]=10
    stats['size_min_contracts'][:]=1;stats['size_max_contracts'][:]=3
    # Replace a real known-side print with an unknown-side print, no imputation.
    stats['buy_trade_records'][5]=0;stats['unknown_trade_records'][5]=1
    a[2]['buy'][5]=0;a[2]['unknown'][5]=3
    a[3]['delta'][5]=False;a[3]['cvd'][5]=False
    surface,_=f.timeframe_skeleton(*a,tf,100000000)
    out=f.aggregate_observed_trade_size_v1(a[0],a[1],a[2],a[3],stats,tf)
    valid=surface['observed_valid']
    total=sum(out['observed_'+side+'_trade_records'] for side in ('buy','sell','unknown'))
    np.testing.assert_array_equal(total[valid],surface['observed_trades'][valid])
    np.testing.assert_array_equal(out['observed_executed_trade_size_std_contracts'][valid],np.ones(valid.sum()))
    np.testing.assert_array_equal(out['observed_executed_trade_size_min_contracts'][valid],np.ones(valid.sum()))
    np.testing.assert_array_equal(out['observed_executed_trade_size_max_contracts'][valid],np.full(valid.sum(),3.))
    for side,expected in (('buy',3.),('sell',1.),('unknown',3.)):
        side_valid=out['observed_'+side+'_trade_size_valid']
        assert np.asarray(side_valid).dtype==np.dtype(bool)
        np.testing.assert_array_equal(out['observed_'+side+'_avg_executed_trade_contracts'][side_valid],np.full(side_valid.sum(),expected))
        assert np.isnan(out['observed_'+side+'_avg_executed_trade_contracts'][~side_valid]).all()
    bad={**stats,'size_sum_contracts':stats['size_sum_contracts']+1}
    with pytest.raises(RuntimeError,match='M1_VOLUME_OR_RECORD_PARITY'):
        f.aggregate_observed_trade_size_v1(a[0],a[1],a[2],a[3],bad,tf)


def test_trade_size_uint64_guard_and_native_declaration_are_explicit():
    maximum=np.iinfo(np.uint64).max
    with pytest.raises(RuntimeError,match='UINT64_OVERFLOW'):
        f.checked_trade_uint64_sums_v1(np.array([maximum,1],np.uint64),np.array([0]),np.array([2]))
    with pytest.raises(RuntimeError,match='UINT64_OVERFLOW'):
        f.trusted_trade_record_count_v1({'buy_trade_records':np.array([maximum],np.uint64),
            'sell_trade_records':np.array([1],np.uint64),'unknown_trade_records':np.array([0],np.uint64)})
    old=f.completion_native_candidates_v1(include_trade_size_stats=False)['fields']
    new=f.completion_native_candidates_v1()['fields']
    assert len(new)-len(old)==5*(len(f.TRADE_SIZE_NATIVE_FIELDS_V1)+3)
    row=next(v for v in new if v['column']=='observed_buy_avg_executed_trade_contracts')
    assert row['owner']=='momentum_flow_encoder' and row['validity_column']=='observed_buy_trade_size_valid'
    mask=next(v for v in new if v['column']=='observed_buy_trade_size_valid')
    assert mask['role']=='structural_observation_mask'


def test_trusted_T_statistics_keep_identical_prints_and_batch_carry_without_nonT_size():
    collector=f.L1FeatureCollectorV2(np.array([0]),1,include_trusted_trade_stats=True)
    data={'size':pa.array([2,2,3,99],type=pa.uint32()),'side':pa.array(['B','B','N','A'])}
    collector._consume_trusted_trade_statistics(data,np.zeros(4,np.int64),np.array([True,True,True,False]))
    data2={'size':pa.array([4],type=pa.uint32()),'side':pa.array(['A'])}
    collector._consume_trusted_trade_statistics(data2,np.array([0]),np.array([True]))
    stats=collector.trusted_trade_statistics()
    assert [int(stats[side+'_trade_records'][0]) for side in ('buy','sell','unknown')]==[2,1,1]
    assert stats['size_sum_contracts'][0]==11 and stats['size_sum_squared_contracts2'][0]==33
    assert stats['size_min_contracts'][0]==2 and stats['size_max_contracts'][0]==4


@pytest.mark.parametrize('tf,duration',[('M5',5),('M15',15),('H1',60),('H4',240),('D1',1440)])
def test_known_cumulative_seven_field_reference_on_each_tf(tf,duration):
    """Independent integer ledger; UTC-day/gap/roll are distinct resets."""
    import pandas as pd
    from gx1.time.session_detector import TRADING_SESSION_BOUNDARY_OFFSET
    prefix=pd.DatetimeIndex(['2024-05-01T23:58Z','2024-05-01T23:59Z',
        '2024-05-02T00:00Z','2024-05-02T00:02Z','2024-05-02T00:03Z']).as_unit('ns').asi8
    # This source-owned session boundary is also on every smaller TF grid.
    start=(pd.Timestamp('2024-05-02',tz='UTC')+TRADING_SESSION_BOUNDARY_OFFSET).value
    minutes=np.r_[prefix,start+np.arange(duration)*t.MINUTE_NS]
    selected=np.r_[np.array([0,0,0,0,1],np.int16),np.ones(duration,np.int16)]
    metrics={name:np.zeros(len(minutes),np.uint64) for name in c.METRICS}
    for name,values in {'buy':[3,1,2,4,5],'sell':[1,2,3,1,1],
                        'unknown':[0,2,1,0,3],'volume':[4,5,6,5,9],
                        'trades':[2,3,3,2,3]}.items():
        metrics[name][:5]=values
    # Only these two tail minutes are observed in the complete TF clock.
    # The final clock minute has a deliberately large unadmitted print.
    left,right,last=5+duration-3,5+duration-2,5+duration-1
    for name,values in {'buy':[2,7,500],'sell':[3,1,0],'unknown':[1,2,0],
                        'volume':[6,10,500],'trades':[3,3,1]}.items():
        metrics[name][[left,right,last]]=values
    volume=np.zeros(len(minutes),bool);volume[:5]=True;volume[[left,right]]=True
    masks={'volume':volume,'delta':volume&(metrics['unknown']==0),'cvd':volume&(metrics['unknown']==0)}
    actual=f.known_cumulative_contributions_v1(minutes,selected,masks,metrics)
    names=('buy_contracts','sell_contracts','unknown_contracts','volume_contracts',
           'known_delta_contracts','unknown_fraction','observed_minutes')
    # Second row retains unknowns; row2 resets at midnight, row3 after missing
    # 00:01, and row4 on the contract change, despite consecutive minute clocks.
    expected=np.array([[3,1,0,4,2,0,1],[4,3,2,9,1,2/9,2],
        [2,3,1,6,-1,1/6,1],[4,1,0,5,3,0,1],[5,1,3,9,4,1/3,1]],float)
    np.testing.assert_allclose(np.column_stack([actual[name][:5] for name in names]),expected,rtol=0,atol=1e-15)
    tail=np.array([9,4,3,16,5,3/16,2],float)
    np.testing.assert_allclose([actual[name][right] for name in names],tail,rtol=0,atol=1e-15)
    assert all(np.isnan(actual[name][last]) for name in names)
    ohlc=np.tile([100,101,99,100],(len(minutes),1)).astype(np.int64)
    surface,_=f.timeframe_skeleton(minutes,selected,metrics,masks,ohlc,tf,100000000)
    f.add_known_cumulative_contributions_v1(surface,actual,minutes)
    row=np.flatnonzero(surface['bar_start_ns']==start).item()
    assert surface['observed_valid'][row]
    assert surface['observed_accepted_minutes'][row]==2
    assert int(surface['observed_last_minute_ns'][row])==int(minutes[right])
    np.testing.assert_allclose([surface['observed_known_cumulative_'+name][row] for name in names],tail,rtol=0,atol=1e-15)


@pytest.mark.parametrize('tf,duration',[('M5',5),('M15',15),('H1',60),('H4',240),('D1',1440)])
def test_unequal_trade_record_counts_have_exact_all_tf_quantity_weighted_means(tf,duration):
    """Seven actual print sizes: B1,B3,A2 / B10,A4,A4,N5."""
    import pandas as pd
    from gx1.time.session_detector import TRADING_SESSION_BOUNDARY_OFFSET
    a=list(minute_inputs(duration))
    start=(pd.Timestamp('2024-05-02',tz='UTC')+TRADING_SESSION_BOUNDARY_OFFSET).value
    a[0]=start+np.arange(duration)*t.MINUTE_NS
    for values in a[2].values():values[:]=0
    for name,values in {'buy':[4,10],'sell':[2,8],'unknown':[0,5],
                        'volume':[6,23],'trades':[3,4]}.items():a[2][name][:2]=values
    for values in a[3].values():values[:]=False
    a[3]['volume'][:2]=True;a[3]['delta'][0]=a[3]['cvd'][0]=True
    stats={name:np.zeros(duration,np.uint64) for name in f.TRUSTED_TRADE_SIZE_FIELDS_V1}
    for name,values in {'buy_trade_records':[2,1],'sell_trade_records':[1,2],
        'unknown_trade_records':[0,1],'size_sum_contracts':[6,23],
        'size_sum_squared_contracts2':[14,157],'size_min_contracts':[1,4],
        'size_max_contracts':[3,10]}.items():stats[name][:2]=values
    surface,_=f.timeframe_skeleton(*a,tf,100000000)
    sized=f.aggregate_observed_trade_size_v1(a[0],a[1],a[2],a[3],stats,tf)
    assert surface['observed_valid'].tolist()==[True]
    assert surface['observed_volume'][0]==29 and surface['observed_trades'][0]==7
    assert surface['observed_avg_executed_trade_contracts'][0]==29/7
    assert surface['observed_avg_executed_trade_contracts'][0]!=(2+23/4)/2
    assert surface['observed_known_delta_contracts'][0]==4
    assert surface['observed_unknown'][0]==5 and not surface['observed_signed_complete'][0]
    for side,records,mean in (('buy',3,14/3),('sell',3,10/3),('unknown',1,5.)):
        assert sized['observed_'+side+'_trade_records'][0]==records
        assert sized['observed_'+side+'_trade_size_valid'][0]
        assert sized['observed_'+side+'_avg_executed_trade_contracts'][0]==mean
    # n=7, sum=29, sumsq=171: population variance numerator=7*171-29^2=356.
    np.testing.assert_allclose(sized['observed_executed_trade_size_std_contracts'],[np.sqrt(356)/7],rtol=0,atol=1e-15)
    assert sized['observed_executed_trade_size_min_contracts'][0]==1
    assert sized['observed_executed_trade_size_max_contracts'][0]==10
