from types import SimpleNamespace

import numpy as np
import pytest

from gx1.scripts import audit_entry_direction_decimal_sign_v1 as original
from gx1.scripts import materialize_entry_direction_decimal_targets_v1 as owner


def test_grid_recovers_exact_sides_only_with_verified_vendor_midpoints():
    source={'bid_open':np.array([2300.000,2300.001]),'ask_open':np.array([2300.120,2300.119])}
    numerators=np.array([4600120,4600120],np.int64)
    grid=owner.reconstruct_grid(source,numerators,3)
    np.testing.assert_array_equal(grid['bid_ticks'],[2300000,2300001])
    np.testing.assert_array_equal(grid['ask_ticks'],[2300120,2300119])
    assert not grid['bid_ticks'].flags.writeable and not grid['ask_ticks'].flags.writeable
    with pytest.raises(RuntimeError,match='GRID_DECIMAL_SOURCE_PARITY'):
        owner.reconstruct_grid(source,numerators+1,3)
    bad=dict(source,bid_open=np.array([2300.0001,2300.001]))
    with pytest.raises(RuntimeError,match='GRID_FLOAT_PARITY'):
        owner.reconstruct_grid(bad,numerators,3)


def test_independent_audit_confusions_and_target_clocks_must_match():
    stub=SimpleNamespace(confusion_by_horizon=original.confusion_by_horizon,FOLD_TRAIN=1,FOLD_LATER_END=2,
        CUTOFF_NS=100,TRAIN_CUTOFF=100,LATER_CUTOFF=100)
    common={'parent_rows':np.array([0,1],np.int64),'bar_start_ns':np.array([1,2],np.int64),
        'decision_ns':np.array([3,4],np.int64),'dynamic_entry_m1_row':np.array([10,11],np.int64),
        'terminal_time_ns':np.array([[5,6],[6,7]],np.int64),'terminal_source_row':np.array([[11,12],[12,13]],np.int64),
        'observed':np.ones((2,2),np.bool_)}
    old=dict(common,sign=np.array([[1,-1],[1,-1]],np.int8))
    new=dict(common,sign=np.array([[0,-1],[1,-1]],np.int8),mid_move=np.array([[0.,-1.],[1.,-1.]]),
        mid_return_bps=np.array([[0.,-1.],[1.,-1.]]))
    old_s,new_s,obs,t=old['sign'],new['sign'],old['observed'],old['terminal_time_ns']
    audit={'status':'COMPLETE','test_data_used':False,'confusions':{
        'all_118249':original.confusion_by_horizon(old_s,new_s,obs,t<100),
        'fold1_train':original.confusion_by_horizon(old_s[:1],new_s[:1],obs[:1],t[:1]<100),
        'fold1_later':original.confusion_by_horizon(old_s[1:],new_s[1:],obs[1:],t[1:]<100)}}
    assert owner.require_corrected_target(old,new,audit,decimal=stub)=={'all_false_moves_removed':1,'nonzero_sign_flips':0}
    changed=dict(new,terminal_source_row=np.array([[11,99],[12,13]],np.int64))
    with pytest.raises(RuntimeError,match='PRESERVE:terminal_source_row'):
        owner.require_corrected_target(old,changed,audit,decimal=stub)
    changed=dict(new,sign=np.array([[-1,-1],[1,-1]],np.int8))
    with pytest.raises(RuntimeError,match='AUDIT_CONFUSION_PARITY'):
        owner.require_corrected_target(old,changed,audit,decimal=stub)


def test_decimal_target_roundtrip_runs_under_existing_guards(tmp_path):
    import subprocess
    import sys
    output=tmp_path.parent/'decimal_grid_actual_guard_case'
    output.mkdir()
    code=r'''
import runpy,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as arrow
import pyarrow.parquet as parquet
import pyarrow.dataset
from gx1.contracts import gx1_capped_execution_v1,immutable_event_authority_v1,local_random_access_campaign_v2
from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
from gx1.contracts.xau_tape_provenance_v1 import scheduled_weekly_closure_mask
from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
from gx1.scripts import audit_entry_direction_decimal_sign_v1 as decimal
from gx1.scripts import materialize_entry_direction_dense_v1 as dense
from gx1.scripts.materialize_entry_direction_decimal_targets_v1 import reconstruct_grid
source,saved,cutoff,original=runpy.run_path('tests/test_materialize_entry_direction_dense_v1.py')['decimal_grid_fixture']()
nums=original['bid_ticks']+original['ask_ticks']
out=Path(sys.argv[1]).resolve(); target=out/'synthetic_targets.npz'
sys.dont_write_bytecode=True
sys.addaudithook(candle.make_guard(out,[],[],output_payload_paths=[str(target)]))
sys.setprofile(candle.no_queries_or_models)
grid=reconstruct_grid(source,nums,3)
corrected=dense.build_decimal_grid_dense_targets(source,saved,decimal_grid=grid,cutoff_ns=cutoff,block_rows=2)
with target.open('xb') as f: np.savez(f,**corrected)
with np.load(target,allow_pickle=False) as archive:
 assert archive['sign'][0,0]==0 and archive['sign'][0,2]==1 and not archive['observed'][0,1]
sys.setprofile(None)
'''
    result=subprocess.run([sys.executable,'-c',code,str(output)],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
