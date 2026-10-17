"""Algebra/clock contracts only; synthetic cases do not establish edge."""
from pathlib import Path
import ast,json
import numpy as np
import pandas as pd
import pytest
from gx1.scripts.audit_entry_gc_foundation_v1 import p0_forecast_diagnostics,p0_fixed_horizon_orders,p0_group_forecasts
from gx1.scripts.research_m5_intraday_recheck_v1 import Execution,NS,MINUTE_NS,Tape,execution_for_horizon_v2,p0_position_ledger,net_cash_values_v3


def test_MSE_decomposition_explains_anticorrelation_and_bias():
    y=np.array([-2.,-1.,1.,2.]);p=-y;c=np.zeros(4)
    d=p0_forecast_diagnostics(y,p,c,[0,.5,1])
    assert d['calibration_slope']==pytest.approx(-1)
    assert d['mse_bps2']==pytest.approx(10)
    assert d['MSE_excess_bps2']==pytest.approx(d['reference_relative_prediction_energy_bps2']-d['twice_reference_relative_alignment_bps2'])
    assert d['mse_bps2']==pytest.approx(d['MSE_variance_component_bps2']+d['MSE_squared_bias_bps2'])


def test_constant_forecast_does_not_invent_calibration_or_IC():
    d=p0_forecast_diagnostics([1.,-1.],[0.,0.],[0.,0.],[0,1])
    assert d['rank_ic'] is None and 'calibration_slope' not in d
    assert d['prediction_std_bps']==0


def ex(decisions,valid=True):
    n=len(decisions);entry=decisions+60*NS;end=entry+30*MINUTE_NS
    return Execution(np.full(n,valid),entry,end,np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n))


def test_fixed30_ignores_intermediate_flat_and_flip_until_scheduled_end():
    d=np.arange(0,46,5,dtype=np.int64)*MINUTE_NS;side=np.array([1,0,-1,1,1,0,1,-1,0,1])
    times,actions,chosen=p0_fixed_horizon_orders(d,side,ex(d))
    assert chosen.tolist()==[0,7]
    assert times.tolist()==[0,30*MINUTE_NS,35*MINUTE_NS,65*MINUTE_NS]
    assert actions.tolist()==[1,0,-1,0]


def test_censored_fixed30_source_fails_before_any_entry_choice():
    d=np.array([0],np.int64)
    with pytest.raises(RuntimeError,match='CENSORED'):p0_fixed_horizon_orders(d,[1],ex(d,False))


def test_no_signals_gives_no_orders():
    d=np.arange(3,dtype=np.int64)*300*NS
    t,a,chosen=p0_fixed_horizon_orders(d,np.zeros(3,int),ex(d))
    assert not len(t) and not len(a) and not len(chosen)


def test_fixed30_cash_ledger_matches_canonical_v3_for_LONG_and_SHORT():
    stamp=pd.Timestamp('2026-05-01',tz='UTC').value;clock=stamp+np.arange(100)*MINUTE_NS
    mid=100+np.arange(100)*.01;tp=Tape(clock,mid-.01,mid+.01,mid-.01,mid+.01)
    d=clock[np.arange(0,61,5)];sides=np.array([1]*7+[-1]*6)
    e=execution_for_horizon_v2(d,tp,horizon_minutes=30,latency_seconds=60)
    tt,aa,chosen=p0_fixed_horizon_orders(d,sides,e)
    recipe=json.loads((Path(__file__).resolve().parents[1]/'configs/research/GC_P0_FIRST_DEVELOPMENT_20261017_021.json').read_text())
    scenario={'round_trip_execution_cost_bps':4.,'annual_financing_cost_bps':[569.,0.]}
    result,_,trades,_=p0_position_ledger(tt,aa,tp,clock[0],clock[-1]+MINUTE_NS,{'c':scenario},31557600.)
    cash=net_cash_values_v3(e,economics=recipe['economics'])
    expected=cash[chosen,np.where(sides[chosen]>0,0,1)].sum()
    assert result['c']['net_total_bps']==pytest.approx(expected)
    assert all(t['elapsed_seconds']==1800 and not t['terminal_open'] for t in trades)


def test_group_statistics_keep_fold_and_session_RV_coordinates():
    f=pd.DataFrame({'y30_mid_bps':[-1.,1.],'train_mean_bps':[0.,0.],'tree_CORE':[-.1,.1],
                    'fold':['a','b'],'session':['EU','ASIA'],'RV_tercile':[0,2],'EMA_trend_sign':[-1,1]})
    r=p0_group_forecasts(f,['tree_CORE'],[0,1])
    assert r['pooled']['tree_CORE']['rows']==2
    assert r['fold_session_RV=a/EU/0']['tree_CORE']['rows']==1


def test_diagnosis_has_no_fit_predict_or_threshold_search_calls():
    path=Path(__file__).resolve().parents[1]/'gx1/scripts/audit_entry_gc_foundation_v1.py'
    tree=ast.parse(path.read_text());names={'p0_forecast_diagnostics','p0_fixed_horizon_orders','p0_group_forecasts','run_p0_diagnosis'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==4
    forbidden={'fit','train','predict','ridge_fit','gbdt_select_and_fit','ridge_select_and_fit','run_p0_development'}
    for node in nodes:
        for call in ast.walk(node):
            if isinstance(call,ast.Call):
                name=call.func.attr if isinstance(call.func,ast.Attribute) else call.func.id if isinstance(call.func,ast.Name) else ''
                assert name not in forbidden
