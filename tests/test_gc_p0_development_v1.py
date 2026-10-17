"""P0 mathematical contracts only; toy prices do not establish trading quality."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from gx1.scripts.research_m5_intraday_recheck_v1 import (
    Tape,MINUTE_NS,NS,p0_field_contract,p0_actions,p0_position_ledger,
    p0_rule_predictions,execution_for_horizon_v2,net_cash_values_v3)
from gx1.features.d1_macro_context_v1 import D1_MACRO_CONTEXT_FIELDS
from gx1.scripts.research_openbb_source_review_v1 import select_verified_cot_vintage_v2
ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/'configs/research/GC_P0_FIRST_DEVELOPMENT_20261017_021.json').read_text())


def tape(prices,spread=0.):
    t=pd.Timestamp('2025-05-01',tz='UTC').value+np.arange(len(prices))*MINUTE_NS
    p=np.asarray(prices,float)
    return Tape(t,p-spread/2,p+spread/2,p-spread/2,p+spread/2)


def run(t,sides,scenario,decisions=None):
    d=t.time_ns[:-1]-MINUTE_NS if decisions is None else decisions
    return p0_position_ledger(d,sides,t,t.time_ns[0],t.time_ns[-1]+MINUTE_NS,{'s':scenario},31557600.)


def test_exact_macro_free_partition_all_eight_arms():
    prep=json.loads((ROOT/'configs/research/P0_FACTORIAL_NEW_FL_PREPARATION_20261017.json').read_text())
    fields=prep['core_native_fields']+prep['layers']['V']['existing_native_fields']+prep['pattern_pause_fields']
    names,macro=p0_field_contract(prep,fields,D1_MACRO_CONTEXT_FIELDS)
    assert len(macro)==61 and len(names['CORE'])==211 and len(names['FLV'])==247
    assert names==PLAN['feature_order']
    for values in names.values():assert not set(values)&set(macro)


def test_signed_entry_notional_spread_and_terminal_open():
    t=tape([100,101,102,103],.2)
    z={'round_trip_execution_cost_bps':4.,'annual_financing_cost_bps':[0.,0.]}
    sums,_,trades,account=run(t,[1,1,1],z)
    expected=(102.9-100.1)/100.1*10000-4
    assert sums['s']['net_total_bps']==pytest.approx(expected)
    assert sums['s']['trades_opened']==1 and sums['s']['trades_closed']==0
    assert trades[-1]['terminal_open'] and account['final_open_position']==1
    assert sums['s']['execution_cost_and_exit_reserve_bps']==4


def test_short_matches_v3_cash_owner_with_no_forced_exit():
    t=tape([100,99,98,97],.2)
    e=PLAN['economics'];decision=np.array([t.time_ns[0]-MINUTE_NS])
    ex=execution_for_horizon_v2(decision,t,horizon_minutes=3,latency_seconds=60)
    expected=net_cash_values_v3(ex,economics=e)[0,1]
    scenario={k:e[k] for k in ['round_trip_execution_cost_bps','annual_financing_cost_bps']}
    result,_,_,_=run(t,[-1,-1,-1],scenario)
    assert result['s']['net_total_bps']==pytest.approx(expected,abs=1e-10)


def test_long_short_symmetric_zero_spread_opposite_move():
    cfg={'round_trip_execution_cost_bps':2.,'annual_financing_cost_bps':[0.,0.]}
    a=run(tape([100,101,102,103]),[1,1,1],cfg)[0]['s']['net_total_bps']
    b=run(tape([100,99,98,97]),[-1,-1,-1],cfg)[0]['s']['net_total_bps']
    assert a==pytest.approx(b)


def test_flip_counts_both_legs_and_preserves_open_loss():
    cfg={'round_trip_execution_cost_bps':4.,'annual_financing_cost_bps':[0.,0.]}
    r,_,trades,_=run(tape([100,101,102,103]),[1,-1,-1],cfg)
    assert r['s']['trades_opened']==2 and r['s']['trades_closed']==1
    assert r['s']['net_total_bps']==pytest.approx(100-2/101*10000-8)
    assert len(trades)==2 and trades[-1]['gross_bps']<0


def test_funding_accrues_through_gaps_no_new_signal_no_time_exit():
    t=tape([100]*4);t.time_ns[2:]+=2*86400*NS
    cfg={'round_trip_execution_cost_bps':4.,'annual_financing_cost_bps':[569.,0.]}
    r,_,trades,_=run(t,[1],cfg,np.array([t.time_ns[0]-MINUTE_NS]))
    duration=(t.time_ns[-1]-t.time_ns[0])/NS
    assert r['s']['net_total_bps']==pytest.approx(-4-569*duration/31557600)
    assert trades[-1]['terminal_open']


def test_missing_execution_quote_does_not_create_fill_or_close():
    t=tape([100,101,102,103]);t.time_ns[2:]+=MINUTE_NS
    cfg={'round_trip_execution_cost_bps':0.,'annual_financing_cost_bps':[0.,0.]}
    d=np.array([t.time_ns[0]-MINUTE_NS,t.time_ns[1]])
    r,_,trades,a=run(t,[1,0],cfg,d)
    assert a['rejected_missing_exact_execution_quote']==1
    assert r['s']['trades_closed']==0 and trades[-1]['terminal_open']


def test_actions_cost_known_and_symmetric_no_future_inputs():
    econ=dict(PLAN['economics']);econ['annual_financing_cost_bps']=[0.,0.]
    assert p0_actions(np.array([10.,-10.,0.,np.nan]),np.array([1.]*4),econ,30).tolist()==[1,-1,0,0]
    assert p0_actions(np.array([5.,-5.]),np.array([1.,1.]),econ,30).tolist()==[0,0]


def test_rule_future_labels_cannot_change_current_forecast():
    snap=np.array([[-1.,1.],[1.,2.],[-1.,3.],[1.,4.]])
    cfg={'core_direction_field':'ema','F_direction_field':'ofi','L_direction_field':'dist','V_state_field':'rv','vol_quantiles':[1/3,2/3]}
    y=np.array([-10.,10.,1.,-1.]);train=np.array([0,1]);ev=np.array([2,3])
    a,_=p0_rule_predictions(snap,['ema','rv'],{}, {},train,ev,'CORE',y,cfg)
    y[ev]=1e9
    b,_=p0_rule_predictions(snap,['ema','rv'],{}, {},train,ev,'CORE',y,cfg)
    np.testing.assert_array_equal(a,b)


def test_cot_december_revision_and_unqualified_schedule_never_admitted():
    p=json.loads((ROOT/'configs/research/CFTC_RESEARCH_RELEASE_HISTORY_V2_20261017.json').read_text())
    rows={r['report_date']:r for r in p['revised_catchup']}
    assert rows['2025-11-10']['scheduled_release_utc']=='2025-12-10T20:30:00Z'
    assert rows['2025-12-23']['scheduled_release_utc']=='2025-12-29T20:30:00Z'
    assert '2026-01-06' not in rows
    assert select_verified_cot_vintage_v2(pd.Timestamp('2026-06-01',tz='UTC').value,list(rows.values())) is None


def test_pit_strict_release_and_revision_vintage():
    event={'report_date':'2025-11-10','actual_release_utc':'2025-12-10T20:30:00Z','vintage_sha256':'a'*64,'evidence_class':'VERIFIED_PUBLICATION_AND_VINTAGE'}
    now=pd.Timestamp(event['actual_release_utc']).value
    assert select_verified_cot_vintage_v2(now,[event]) is None
    assert select_verified_cot_vintage_v2(now+1,[event])==event
    revision=dict(event,actual_release_utc='2025-12-12T20:30:00Z',vintage_sha256='b'*64)
    assert select_verified_cot_vintage_v2(now+1,[event,revision])==event
