"""Synthetic fit/causality mechanics, never market quality or native evidence."""
import numpy as np
import pytest
from gx1.scripts import research_m5_intraday_recheck_v1 as p


def test_input_lock_keeps_native_aliases_once_and_uses_initial_train_only():
    snap=np.array([[1,2],[2,3],[3,4],[4,5.]],dtype=float)
    ctx=snap[:,1:]
    gc={'mask':np.ones(4),'price':np.array([10,11,12,13.]),
        'flow':np.array([2,np.nan,5,6.]),'future_only':np.array([np.nan,np.nan,10,11.])}
    with pytest.raises(RuntimeError,match='EMPTY_ARM_INCREMENT:C'):
        p.gc_input_matrices(snap,ctx,np.array([1,2,1,2]),
            ['native','ctx_cont.ctx'],['ctx'],gc,{'B':['price'],'C':['flow','future_only']},['mask'],np.array([0,1]))
    gc['flow'][1]=3
    matrices,names,excluded=p.gc_input_matrices(snap,ctx,np.array([1,2,1,2]),
        ['native','ctx_cont.ctx'],['ctx'],gc,{'B':['price'],'C':['flow','future_only']},['mask'],np.array([0,1]))
    assert matrices['A'].shape[1]==5
    assert 'future_only' in excluded
    assert names['C'][-2:]==['flow','flow__missing']
    np.testing.assert_array_equal(matrices['B'][:,:5],matrices['A'])
    with pytest.raises(RuntimeError,match='ALIAS_MISMATCH'):
        p.gc_input_matrices(snap,ctx+1,np.array([1,2,1,2]),
            ['native','ctx_cont.ctx'],['ctx'],gc,{'B':['price'],'C':['flow']},['mask'],np.array([0,1]))


def test_returned_ridge_state_replays_and_future_mutation_cannot_change_fit():
    x=np.column_stack((np.arange(40.),np.sin(np.arange(40.))))
    y=np.cos(np.arange(40.))
    fold=p.Fold('fold',25,40,np.arange(25),np.arange(18),np.arange(18,25),np.arange(25,40))
    result=p.ridge_select_and_fit(x,y,fold,[1.,10.],return_state=True)
    replay=p.standardize(x,fold.eval,result['mean'],result['std']) @ result['beta']+result['intercept']
    np.testing.assert_array_equal(replay,result['pred'])
    changed=x.copy();changed[25:]+=10000
    changed_y=y.copy();changed_y[25:]+=10000
    future=p.ridge_select_and_fit(changed,changed_y,fold,[1.,10.],return_state=True)
    for key in ('beta','intercept','mean','std','alpha'):
        np.testing.assert_array_equal(result[key],future[key])


def test_purge_inner_split_and_evaluation_rows_are_chronological():
    import pandas as pd
    decisions=pd.date_range('2024-05-01','2025-06-01',freq='30min',tz='UTC').asi8
    exits=decisions+360*p.MINUTE_NS
    valid=np.ones(len(decisions),dtype=bool)
    fold=p.build_fold('dev','2025-05-01','2025-06-01',decisions,exits,valid,train_years=3,inner_fraction=.2)
    assert np.max(exits[fold.train])<fold.eval_start_ns
    assert np.max(exits[fold.inner_train])<np.min(decisions[fold.inner_val])
    assert np.max(decisions[fold.train])<fold.eval_start_ns-86400*p.NS
