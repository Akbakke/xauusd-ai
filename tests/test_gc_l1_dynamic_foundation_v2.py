import numpy as np
import pandas as pd
from gx1.features.gc_order_flow_v1 import (L1FeatureCollectorV2,bbo_transition_v2,
    value_area_v2,profile_values_v2,dynamic_profiles_v2)

NS=10**9

def events():
    return np.array([[0,100,102,10,20],[10*NS,100,102,15,20],
        [30*NS,101,103,7,9]],np.int64)

def reduce(e,cut=60*NS):
    c=L1FeatureCollectorV2([0],1,origin_ns=0);c.consume(e);c.finish(cut)
    return c,c.values([True])[0]

def test_scalar_ofi_integral_reference():
    c,v=reduce(events())
    assert c.covered[0]==60*NS
    assert v['ofi_contracts'][0]==32
    assert v['bid_refill_contracts'][0]==5
    assert v['ask_depletion_contracts'][0]==0
    assert v['quote_updates'][0]==2
    expected=(10*(-10/30)+20*(-5/35)+30*(-2/16))/60
    np.testing.assert_allclose(v['tw_imbalance'],[expected],atol=1e-15)
    assert v['tw_spread_ticks'][0]==2
    assert v['mid_response_ticks'][0]==1

def test_prefix_and_future_mutation():
    _,p=reduce(events())
    c=L1FeatureCollectorV2([0],1,origin_ns=0)
    c.consume(np.vstack((events(),[70*NS,900,902,99,1])))
    f,_=c.values([True])
    for key in p:np.testing.assert_allclose(p[key],f[key],atol=1e-12)

def test_batch_boundary_and_identical_snapshot_invariance():
    c=L1FeatureCollectorV2([0],1,origin_ns=0)
    e=events();c.consume(e[:2]);c.consume(np.vstack((e[1],e[2])));c.finish(60*NS)
    _,expected=reduce(e)
    actual,_=c.values([True])
    for key in expected:np.testing.assert_array_equal(actual[key],expected[key])

def test_partial_integral_and_family_mask_not_neutral():
    c=L1FeatureCollectorV2([0,1,2],1,origin_ns=0)
    c.consume(np.array([[10*NS,100,102,10,20],[170*NS,101,103,7,9]]));c.finish(180*NS)
    v,m=c.values([True,True,False]);assert not m.any()
    for key in ['tw_imbalance','tw_spread_ticks','mid_response_ticks']:assert np.isnan(v[key]).all()
    assert c.covered[1]==0

def test_ofi_side_symmetry_and_same_price_depletion():
    p=np.array([[0,100,102,10,20]]);c=np.array([[1,100,102,6,25]])
    out=bbo_transition_v2(p,c)[0];assert tuple(out[:5])==(-9,0,5,4,0)
    # Reflect price and swap bid/ask; signed OFI changes sign.
    reflect=lambda a:np.column_stack((a[:,0],-a[:,2],-a[:,1],a[:,4],a[:,3]))
    assert bbo_transition_v2(reflect(p),reflect(c))[0,0]==9

def test_value_area_is_contiguous_not_quantiles_and_lower_tie():
    assert value_area_v2({100:40,101:10,102:35,103:15})==(100,100,102)
    assert value_area_v2({100:30,101:40,102:30})==(101,100,101)
    np.testing.assert_allclose(profile_values_v2({100:1,102:1},.1)['vwap_usd'],10.1)

def test_profiles_gap_roll_reset_and_prefix():
    start=pd.Timestamp('2024-05-01T22:00Z').value+np.arange(15)*300*NS
    hist=[{100+i:10} for i in range(15)];valid=np.ones(15,bool);valid[6]=False
    contracts=np.zeros(15,int);contracts[12:]=1;close=np.arange(15)+100.
    out=dynamic_profiles_v2(start,hist,valid,contracts,close,1)
    assert not out['rolling15_valid'][1] and out['rolling15_valid'][2]
    assert not out['rolling15_valid'][7] and out['rolling15_valid'][9]
    assert not out['rolling15_valid'][12] and out['rolling15_valid'][14]
    assert out['session_complete_valid'][5] and not out['session_complete_valid'][7]
    p=dynamic_profiles_v2(start[:8],hist[:8],valid[:8],contracts[:8],close[:8],1)
    for key in p:np.testing.assert_array_equal(out[key][:8],p[key])

def test_dst_session_anchor_is_local_wall_clock():
    t=pd.DatetimeIndex(['2024-03-10T22:00Z','2024-11-03T23:00Z']).asi8
    p=dynamic_profiles_v2(t,[{100:1},{100:1}],np.ones(2,bool),np.zeros(2,int),np.ones(2)*100,1)
    np.testing.assert_array_equal(p['session_expected_minutes'],[5,5])
    assert p['session_complete_valid'].all()
