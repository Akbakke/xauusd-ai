"""Synthetic mechanics only. Real-data admission is produced by plan 004."""
import csv
import gzip
import numpy as np
import pytest
from gx1.contracts import gc_continuous_source_v1 as c
from gx1.contracts import gc_source_disposition_v1 as t


def test_roll_is_past_only_prefix_invariant_and_does_not_roll_back():
    volumes=np.array([[10,20,1,100],[2,3,30,1]],dtype=np.uint64)
    selected,basis=c.choose_daily_contract(volumes,[(2024,6),(2024,8)])
    assert selected.tolist()==[-1,0,0,1]
    assert basis.tolist()==[-1,0,1,2]
    changed=volumes.copy()
    changed[:,2:]=[[0,0],[999,999]]
    assert np.array_equal(c.choose_daily_contract(changed,[(2024,6),(2024,8)])[0][:3],selected[:3])
    extended=np.column_stack((volumes,np.array([1000,1],dtype=np.uint64)))
    assert c.choose_daily_contract(extended,[(2024,6),(2024,8)])[0][-1]==1


def test_roll_ties_zero_days_and_large_integer_volume_are_exact():
    big=2**53
    v=np.array([[0,5,big+2,0],[0,5,big+3,0]],dtype=np.uint64)
    selected,_=c.choose_daily_contract(v,[(2024,6),(2024,8)])
    assert selected.tolist()==[-1,-1,-1,1]
    with pytest.raises(ValueError,match='MATURITY'):
        c.choose_daily_contract(v,[(2024,6),(2024,6)])


def test_masks_distinguish_unsigned_volume_from_unknown_delta():
    values={name:np.zeros(6,dtype=np.uint64) for name in c.METRICS}
    values['trades'][:]=1
    values['unknown'][1]=7
    values['gap'][2]=1
    values['bad_recv_trade'][3]=1
    values['snapshot_trade'][4]=1
    m=c.row_masks(np.array([0,0,0,0,0,-1]),values,np.ones(6,dtype=bool))
    assert m['volume_profile'].tolist()==[True,True,False,False,False,False]
    assert m['footprint'].tolist()==[True,False,False,False,False,False]
    assert np.array_equal(m['delta'],m['cvd'])


def test_segments_reset_on_date_roll_or_invalid_minute():
    ts=np.arange(8,dtype=np.int64)*t.MINUTE_NS+c.DAY_NS-2*t.MINUTE_NS
    selected=np.array([0,0,0,0,1,1,1,1])
    valid=np.array([True,True,True,True,True,False,True,True])
    assert c.reset_segments(selected,valid,ts).tolist()==[0,0,1,1,2,-1,3,3]


def test_admission_requires_all_26_months_and_trustworthy_clock():
    months=np.arange(np.datetime64('2024-05'),np.datetime64('2026-07'),dtype='datetime64[M]')
    masks={name:np.ones(26,dtype=bool) for name in ('volume_profile','footprint','delta','cvd')}
    assert all(c.admission_decisions(months,masks,0)[name]['decision']=='ADMIT' for name in masks)
    assert c.admission_decisions(months,masks,1)['delta']['decision']=='REJECT'
    masks['delta'][12]=False
    assert c.admission_decisions(months,masks,0)['delta']['decision']=='REJECT'
    assert c.admission_decisions(months,masks,0)['sweeps']['decision']=='REJECT'
    assert c.admission_decisions(months,masks,0)['order_blocks']['decision']=='REJECT'


@pytest.mark.parametrize('repeat',[False,True])
def test_arrow_scan_counts_trades_only_and_keeps_unknown_volume(tmp_path,monkeypatch,repeat):
    start=t.STUDY_START_NS
    monkeypatch.setattr(t,'WINDOW_END_NS',start+3*t.MINUTE_NS)
    path=tmp_path/'source.csv.gz'
    header=list(c.COLUMNS)
    def row(offset,action='T',side='B',size=5,flags=0,price=100000000):
        return dict(ts_recv=start+offset,instrument_id=42,publisher_id=1,symbol='GCM4',
                    action=action,side=side,size=size,price=price,flags=flags,sequence=offset%1000,ts_event=start+offset)
    rows=[row(1),row(2,action='F',size=999),row(3,side='N',size=3),
          row(t.MINUTE_NS+1,side='A',flags=c.F_BAD_TS_RECV),
          row(2*t.MINUTE_NS+1,flags=c.F_SNAPSHOT)]
    if repeat:
        rows.insert(1,dict(rows[0]))
    with gzip.open(path,'wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    item=dict(path=str(path),raw_symbol='GCM4',start_ns=start,end_ns=start+3*t.MINUTE_NS,
              instrument_id=42,publisher_id=1,max_records=len(rows))
    out=c.scan_symbol([item],100000000)
    assert out['volume'].tolist()==[13 if repeat else 8,0,0]
    assert out['duplicate_trade'].tolist()==[1 if repeat else 0,0,0]
    assert out['buy'].tolist()==[10 if repeat else 5,0,0]
    assert out['unknown'].tolist()==[3,0,0]
    assert out['bad_recv_trade'].tolist()==[0,1,0]
    assert out['files'][0]['bad_recv_trades']==1
    assert out['snapshot_trade'].tolist()==[0,0,1]
    item['max_records']=len(rows)-1
    with pytest.raises(ValueError,match='BUDGET'):
        c.scan_symbol([item],100000000)


def test_series_roll_uses_60s_availability_and_strict_test_boundary(monkeypatch):
    from gx1.scripts.gc_continuous_source_v1 import build_series
    start=t.STUDY_START_NS
    width=2*c.MINUTES_PER_DAY
    monkeypatch.setattr(t,'WINDOW_END_NS',start+width*t.MINUTE_NS)
    result={name:np.zeros(width,dtype=np.uint64) for name in c.METRICS}
    result.update(symbol='GCM4',files=[])
    result['volume'][:]=1
    result['buy'][:]=1
    result['trades'][:]=1
    result['records'][:]=1
    lookup={start+i*t.MINUTE_NS:i for i in range(width)}
    table,schedule,summary=build_series([result],{'GCM4':(2024,6,100000000)},lookup)
    assert table.num_rows==width-1
    choices=table['contract_index'].to_pylist()
    assert choices[c.MINUTES_PER_DAY] is None
    assert choices[c.MINUTES_PER_DAY+1]==0
    assert int(table['available_ns'][-1].as_py())<t.WINDOW_END_NS
    assert summary['rows_with_contract']==c.MINUTES_PER_DAY-2
