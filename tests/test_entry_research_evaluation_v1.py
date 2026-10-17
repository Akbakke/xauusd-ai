"""New variant mechanics only. Historical calendar/edge need real evidence."""
import numpy as np
import pandas as pd
import pytest
from gx1.contracts import entry_research_evaluation_v1 as ev
from gx1.scripts import research_m5_intraday_recheck_v1 as m5
from gx1.scripts.audit_entry_gc_foundation_v1 import reference_horizon


def test_strict_terminal_censors_even_one_minute_delay_and_keeps_baseline_value():
    t=np.array([0,ev.DAY+ev.MINUTE]);r=reference_horizon(t,np.array([100.,101.]),np.array([101.,102.]),np.array([0]),1,0,2*ev.DAY)
    before=r['gross'].copy();mask,delay=ev.strict_terminal_mask(r,t)
    assert r['valid'].tolist()==[True] and delay.tolist()==[60] and mask.tolist()==[False]
    assert np.array_equal(before,r['gross']) and ev.STRICT_TERMINAL_MAX_DELAY_SECONDS==0


def test_exact_terminal_retains_original_label():
    t=np.array([0,ev.DAY]);r=reference_horizon(t,np.array([100.,101.]),np.array([101.,102.]),np.array([0]),1,0,2*ev.DAY)
    assert ev.strict_terminal_mask(r,t)[0].tolist()==[True]


def test_calendar_is_DST_aware_and_does_not_skip_missing_weekday_session():
    a=pd.Timestamp('2024-03-08 12:00',tz=ev.NY).tz_convert('UTC').value
    b=pd.Timestamp('2024-03-11 12:00',tz=ev.NY).tz_convert('UTC').value
    c=pd.Timestamp('2024-03-12 12:00',tz=ev.NY).tz_convert('UTC').value
    cal=ev.regular_calendar(a,c);t=np.array([a,b,c]);prices=np.array([100.,110.,120.])
    r=ev.calendar_terminal(t,prices,prices+1,np.array([a]),1,c,cal)
    assert r['valid'].tolist()==[True] and r['desired_ns'].tolist()==[b] and b-a==71*3600*ev.NS
    missing=ev.calendar_terminal(t[[0,2]],prices[[0,2]],prices[[0,2]]+1,np.array([a]),1,c,cal)
    assert not missing['valid'][0] and missing['desired_ns'][0]==b
    assert cal['historical_calendar_qualified'] is False


def test_calendar_regular_closure_is_not_called_missing_open_quote():
    a=pd.Timestamp('2024-03-05 16:58',tz=ev.NY).tz_convert('UTC').value
    b=pd.Timestamp('2024-03-05 18:05',tz=ev.NY).tz_convert('UTC').value
    gaps=ev.gap_inventory(np.array([a,b]),ev.regular_calendar(a,b))
    assert gaps.missing_minutes.tolist()==[66] and gaps.missing_during_current_regular_template.tolist()==[0]


@pytest.mark.parametrize('maximum',[True,False])
def test_range_extrema_exact_for_varied_overlapping_windows(maximum):
    x=np.array([2.,-1.,4.,4.,0.,9.,1.]);start=np.array([0,1,2,0,6]);end=np.array([7,3,5,1,7])
    out=ev.range_extrema(x,start,end,maximum=maximum)
    op=np.max if maximum else np.min
    assert np.array_equal(out,np.array([op(x[a:b]) for a,b in zip(start,end)]))


def test_v2_entry_notional_symmetry_with_equal_entry_prices_and_unchanged_mid_labels():
    t=np.arange(65,dtype=np.int64)*ev.MINUTE
    p=np.ones(65)*100;p[31:]=110
    ex=m5.execution_for_horizon_v2(np.array([0]),m5.Tape(t,p,p,p,p),horizon_minutes=30,latency_seconds=60)
    assert ex.long_gross_bps[0]==1000 and ex.short_gross_bps[0]==-1000
    old=m5.execution_for_horizon(np.array([0]),m5.Tape(t,p,p,p,p),horizon_minutes=30,latency_seconds=60)
    assert old.short_gross_bps[0]!=ex.short_gross_bps[0]
    assert np.array_equal(old.y_mid_bps,ex.y_mid_bps) and np.array_equal(old.exit_ns,ex.exit_ns)


def test_v2_spread_once_slippage_twice_financing_once_for_each_side():
    t=np.arange(65,dtype=np.int64)*ev.MINUTE;b=np.full(65,100.);a=np.full(65,101.)
    ex=m5.execution_for_horizon_v2(np.array([0]),m5.Tape(t,b,a,b,a),horizon_minutes=30,latency_seconds=60)
    assert ex.long_gross_bps[0]==pytest.approx(-10000/101) and ex.short_gross_bps[0]==-100
    fin={'long_bps_per_rollover':-1.5589,'short_bps_per_rollover':.8849}
    assert m5.net_trade_bps_v2(1,10.,-12.,1,slippage_bps=2,financing=fin)==pytest.approx(10-4-1.5589)
    assert m5.net_trade_bps_v2(-1,10.,-12.,1,slippage_bps=2,financing=fin)==pytest.approx(-12-4+.8849)
    with pytest.raises(ValueError):m5.net_trade_bps_v2(0,10.,-12.,1,slippage_bps=2,financing=fin)


def test_L1_full_record_duplicate_is_distinguished_from_signature_collision():
    import pyarrow as pa
    from gx1.contracts import gc_continuous_source_v1 as gc
    def data(bid,duplicate):
        d={name:pa.array([0],type=kind) for name,kind in {**gc.COLUMNS,**gc.L1_AUDIT_COLUMNS}.items() if kind!=pa.string()}
        d.update({name:pa.array([value]) for name,value in {'symbol':'GCQ6','action':'T','side':'B'}.items()})
        for k,v in {'bid_px_00':bid,'ask_px_00':bid+10,'bid_sz_00':2,'ask_sz_00':3,'flags':128}.items():d[k]=pa.array([v])
        d['_duplicate_trade_signature']=pa.array([duplicate]);return d
    c=gc.L1QualityCollector(np.array([0]),10);c.file_start({})
    c(data(100,False),np.array([0]),np.array([True]))
    c(data(110,True),np.array([0]),np.array([True]))  # same old T signature, BBO differs
    c(data(110,True),np.array([0]),np.array([True]))
    assert c.values['duplicate_trade_signature'].tolist()==[2]
    assert c.values['signature_collision'].tolist()==[1]
    assert c.values['full_record_duplicate_trade'].tolist()==[1]


def test_L1_masks_require_clean_prior_minute_and_never_unknown_aggressor_filter():
    import pyarrow as pa
    from gx1.contracts import gc_continuous_source_v1 as gc
    from gx1.scripts.audit_entry_gc_foundation_v1 import build_l1_quality_masks
    t=np.arange(11)*ev.MINUTE
    table=pa.table({'bar_start_ns':t,'contract_index':[0]*11})
    metrics={k:np.zeros(11,np.uint64) for k in gc.L1_QUALITY_METRICS}
    metrics['records'][:]=10;metrics['clean_complete_bbo'][:]=8;metrics['reset'][3]=1
    native=pa.table({'m5_bar_start_ns':t[[0,5]]})
    masks,coverage=build_l1_quality_masks(table,metrics,native)
    valid=masks['ofi_event_candidate_valid'].to_numpy()
    assert not valid[0] and not valid[3] and not valid[4] and valid[5:].all()
    assert coverage['ofi_event_candidate']['valid_native_M5']==1
    assert coverage['ofi_event_candidate']['signed_trade_mask_required'] is False


def test_preparation_has_exact_existing_fields_eight_unique_arms_and_no_fit_authority():
    import json
    from pathlib import Path
    import pyarrow.parquet as pq
    root=Path(__file__).resolve().parents[1]
    spec=json.loads((root/'configs/research/P0_FACTORIAL_PREPARATION_20261017.json').read_text())
    assert {tuple(sorted(v['layers'])) for v in spec['arms'].values()}=={(),('F',),('L',),('V',),('F','L'),('F','V'),('L','V'),('F','L','V')}
    assert spec['training_authorized'] is False and spec['execution_authorized'] is False
    baseline=json.loads((root/'configs/research/XAU_M5_INFORMATION_BASELINE_20261017.json').read_text())
    all_fields=set(baseline['unchanged_full_baseline']['native_signal_fields'])
    assert set(spec['core_native_fields']).isdisjoint(spec['layers']['V']['existing_native_fields'])
    assert set(spec['core_native_fields'])|set(spec['layers']['V']['existing_native_fields'])|set(spec['pattern_pause_fields'])==all_fields
    path=root/'configs/research/ENTRY_EVALUATION_CORRECTION_20261017_011.json'
    plan=json.loads(path.read_text());gc=set(pq.ParquetFile(plan['GC_predictions']['path']).schema_arrow.names)
    # Candidate predictors are read from the existing feature sidecar, never outcomes.
    path=json.loads((root/'configs/research/ENTRY_TARGET_AND_GC_L1_AUDIT_20261017_010.json').read_text())['GC_features']['path']
    gc=set(pq.ParquetFile(path).schema_arrow.names)
    for layer in ['F','L']:assert set(spec['layers'][layer]['existing_fields'])<=gc
    assert spec['targets']['future_MFE_MAE_as_features'] is False


def test_weekly_review_excludes_closed_endpoints_without_censoring_normal_closed_path(tmp_path):
    source=tmp_path/'source';dest=tmp_path/'out';source.mkdir();dest.mkdir()
    stamp=lambda s:pd.Timestamp(s,tz=ev.NY).tz_convert('UTC').value
    a=stamp('2024-03-01 17:00');b=stamp('2024-03-03 18:00')
    pd.DataFrame({'start_ns':[a],'end_ns':[b],'missing_minutes':[(b-a)//ev.MINUTE]}).to_parquet(source/'GAP_CLASSIFICATION.parquet')
    for split in ['TRAIN','VAL']:
        for h in [1,5,20]:
            pd.DataFrame({'native_row':[0,1],'decision_ns':[stamp('2024-03-01 16:00')]*2,
              'baseline_exit_ns':[stamp('2024-03-03 17:00'),stamp('2024-03-04 12:00')],
              'strict_terminal_valid':[True,True]}).to_parquet(source/f'{split}_H{h}_RESEARCH_DIAGNOSTICS.parquet')
    r=ev.weekly_closure_review(source,dest)
    assert r['counts']['normal_weekly_closed_missing_minutes_under_existing_contract']==2940
    assert r['counts']['remaining_missing_minutes_not_classified_by_weekly_owner']==0
    assert r['horizons']['train_h20']['strict_additionally_censored_by_weekly_endpoint']==1
    assert r['horizons']['train_h20']['strict_and_weekly_open_valid']==1
    assert r['training_authorized'] is False
