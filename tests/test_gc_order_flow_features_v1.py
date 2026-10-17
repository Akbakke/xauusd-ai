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
