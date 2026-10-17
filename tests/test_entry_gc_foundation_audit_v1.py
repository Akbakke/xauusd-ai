"""Adversarial mechanics of the audit, not evidence of trading value."""
import numpy as np
import pandas as pd
import pyarrow as pa
from gx1.scripts.audit_entry_gc_foundation_v1 import reference_horizon,mask_attribution,NS,MINUTE,DAY

def test_independent_markout_has_entry_notional_and_no_exit_policy(monkeypatch):
    from gx1.contracts import entry_observed_market_horizons_v1 as owner
    monkeypatch.setattr(owner,'causal_m1_terminal_outcomes_at_trading_days',lambda **k:(_ for _ in ()).throw(AssertionError('must not delegate reference')))
    t=np.array([0,DAY,DAY+MINUTE],np.int64)
    out=reference_horizon(t,np.array([100.,110.,109.]),np.array([101.,111.,110.]),np.array([0]),1,0,DAY+MINUTE)
    assert out['valid'].tolist()==[True]
    assert out['gross'][0,0]==(110-101)/101*10000
    assert out['gross'][0,1]==(100-111)/100*10000
    assert out['elapsed'].tolist()==[86400]

def test_missing_target_minute_delay_and_split_censor_are_exposed():
    t=np.array([0,MINUTE,DAY+MINUTE],np.int64)
    out=reference_horizon(t,np.ones(3)*100,np.ones(3)*101,np.array([0]),1,0,2*DAY)
    assert out['exit'].tolist()==[2] and out['target_ns'].tolist()==[DAY]
    assert out['valid'].tolist()==[True]
    cut=reference_horizon(t,np.ones(3)*100,np.ones(3)*101,np.array([0]),1,0,DAY)
    assert cut['valid'].tolist()==[False] and np.isnan(cut['gross']).all()

def test_attribution_distinguishes_unknown_volume_from_roll_and_gap():
    minutes=np.arange(15,dtype=np.int64)*MINUTE
    cols={'bar_start_ns':minutes,'contract_index':np.array([0]*10+[0,1,1,1,1]),'trades':np.ones(15,dtype=np.int64),'volume':np.ones(15,dtype=np.int64)*100,'unknown':np.array([1]+[0]*14), 'invalid_trade':np.zeros(15,dtype=np.int64),'bad_recv_trade':np.zeros(15,dtype=np.int64),'gap':np.array([0]*5+[1]+[0]*9),'snapshot_trade':np.zeros(15,dtype=np.int64),'duplicate_trade':np.zeros(15,dtype=np.int64)}
    native=pa.table({'m5_bar_start_ns':minutes[[0,5,10]],'m5_volume_valid':[True,False,False],'m5_signed_valid':[False,False,False]})
    out=mask_attribution(pa.table(cols),native)
    assert out['unsigned_valid']==1 and out['signed_valid']==0
    assert out['unknown_only_after_unsigned_quality']==1
    assert out['marginal_rejections_overlap']['gap']==1
    assert out['marginal_rejections_overlap']['contract_change_inside_M5']==1

def test_active_audit_config_inputs_have_canonical_existing_bindings():
    import json
    from pathlib import Path
    from gx1.scripts.audit_entry_gc_foundation_v1 import REPO,PLAN_REL
    plan=json.loads((REPO/PLAN_REL).read_text())
    for key in ['native_design','GC_probe_plan','GC_predictions','GC_masks','GC_features']:
        path=Path(plan[key]['path'])
        assert path.is_absolute() and path==path.resolve() and path.is_file()

def test_VAL_reference_does_not_call_partial_source_session_a_censored_label():
    from gx1.scripts.audit_entry_gc_foundation_v1 import safe_reference_prefix
    r={'session_supported':np.array([True,True,False]),'target_ns':np.array([DAY,2*DAY,3*DAY])}
    assert safe_reference_prefix(r,2*DAY+MINUTE,0).tolist()==[True,False,False]
