"""Adversarial mechanics of the audit, not evidence of trading value."""
import numpy as np
import pytest
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


def test_actual_h20_train_epoch_ignores_changed_exit_teacher_and_uses_h20(monkeypatch):
    """Exercise the actual h20 trainer branch, not only the older 95-minute branch."""
    import copy
    import torch
    from tests.test_entry_observed_market_native import SmallEntry, training_fixture
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.contracts import entry_observed_market_horizons_v1 as horizons
    original = horizons.observed_entry_batch_targets
    seen = []
    def record_target(batch, binding, *, device):
        out = original(batch, binding, device=device)
        seen.append(out[0].clone())
        return out
    monkeypatch.setattr(horizons, 'observed_entry_batch_targets', record_target)
    class ForbiddenExit(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.exit_policy = torch.nn.Parameter(torch.tensor(value), requires_grad=False)
        def forward(self, *args, **kwargs):
            raise AssertionError('h20 must not query changed Exit teacher')
    states, losses = [], []
    expected = torch.tensor([[4., -6., 0.], [-8., 6., 0.]])
    for exit_value in [-10000., 10000.]:
        torch.manual_seed(910)
        model = SmallEntry()
        base = training_fixture(monkeypatch)
        base.dataset._entry_observed_market_binding = {'identity': {
            'training_phase': 'entry_only_observed_horizon_20d',
            'loss_target_batch_key': horizons.DECISION_BATCH_TARGET_KEY}}
        class HorizonLoader:
            dataset = base.dataset
            def __len__(self): return len(base)
            def __iter__(self):
                for old in base:
                    batch = {k:v for k,v in old.items() if k != 'entry_observed_market_target_bps'}
                    for h,key in horizons.BATCH_TARGET_KEY_BY_HORIZON.items():
                        batch[key] = expected.clone() if h == 20 else expected.clone() * h
                    yield batch
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.5)
        loss, stats, complete = trainer.train_epoch(
            model, ForbiddenExit(exit_value), HorizonLoader(), optimizer,
            torch.device('cpu'), 1, {}, {})
        assert complete and stats['exit_training_active'] is False
        losses.append(loss); states.append(copy.deepcopy(model.state_dict()))
    assert len(seen) == 4 and all(torch.equal(y, expected) for y in seen)
    assert losses[0] == losses[1]
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])


_CLOCK_MINUTES = {'M5':5, 'M15':15, 'H1':60, 'H4':240, 'D1':1440}


def _event_clock_fixture(timeframe):
    # Independent explicit grids: 22Z is an opening boundary on every TF.
    start = pd.Timestamp('2024-05-01T22:00:00Z').value
    size = _CLOCK_MINUTES[timeframe]
    minutes = start + np.arange(size, dtype=np.int64)*MINUTE
    selected = np.zeros(size, np.int16)
    events = {'qualified_final_event_count': np.full(size, np.nan)}
    events['qualified_final_event_count'][:3] = [5,6,1]
    for domain, counts, sums in (
            ('send_to_capture',[1,3,0],[-10,90,0]),
            ('receive_minus_event',[2,1,0],[-80,20,0]),
            ('matching_clock',[1,2,0],[-60,30,0])):
        events[domain+'_count'] = np.full(size, np.nan)
        events[domain+'_sum_ns'] = np.full(size, np.nan)
        events[domain+'_count'][:3] = counts
        events[domain+'_sum_ns'][:3] = sums
    return minutes, selected, events


@pytest.mark.parametrize('timeframe', tuple(_CLOCK_MINUTES))
def test_event_clock_independent_unequal_count_signed_reference_all_five_tf(monkeypatch,timeframe):
    from gx1.scripts.audit_entry_gc_foundation_v1 import reference_event_clock_means,compare_event_clock_reference
    from gx1.features import gc_order_flow_v1 as owner
    production=owner.aggregate_observed_event_diagnostics_v1
    monkeypatch.setattr(owner,'aggregate_observed_event_diagnostics_v1',
        lambda *a,**k: (_ for _ in ()).throw(AssertionError('independent reference must not call producer')))
    minutes,selected,events = _event_clock_fixture(timeframe)
    reference = reference_event_clock_means(minutes,selected,events,timeframe)
    assert reference['bar_start_ns'].tolist() == [minutes[0]]
    assert reference['available_ns'].tolist() == [minutes[0]+_CLOCK_MINUTES[timeframe]*MINUTE]
    assert reference['observed_event_valid'].tolist() == [True]
    assert reference['observed_event_accepted_minutes'].tolist() == [3]
    for domain,count,total,mean,wrong in (
            ('send_to_capture',4,80,20,10),
            ('receive_minus_event',3,-60,-20,-10),
            ('matching_clock',3,-30,-10,-22.5)):
        prefix='observed_event_'+domain
        assert reference[prefix+'_count'].tolist() == [count]
        assert reference[prefix+'_sum_ns'].tolist() == [total]
        assert reference[prefix+'_mean_ns'].tolist() == [mean]
        assert reference[prefix+'_mean_of_means'].tolist() == [wrong]
        assert reference[prefix+'_unequal_positive_counts'].tolist() == [True]
    # The separate mechanical comparison also pins the actual production
    # aggregator to the hand answer; it is never used to build the oracle.
    accepted=np.isfinite(events['qualified_final_event_count'])
    full={name:events.get(name,np.where(accepted,0.,np.nan))
        for name in owner.EVENT_DIAGNOSTIC_REDUCERS_V1}
    full['qualified_covered_ns']=np.where(accepted,MINUTE,np.nan)
    actual={'bar_start_ns':minutes[[0]],
        'available_ns':minutes[[0]]+_CLOCK_MINUTES[timeframe]*MINUTE,
        **production(minutes,selected,full,timeframe)}
    report=compare_event_clock_reference(reference,actual)
    assert all(item['discriminating_buckets']==1 for item in report['domains'].values())
    actual['observed_event_send_to_capture_mean_ns'][0]=10
    with pytest.raises(RuntimeError,match='GC_EVENT_NUMERIC_VALUE'):
        compare_event_clock_reference(reference,actual)


@pytest.mark.parametrize('timeframe', tuple(_CLOCK_MINUTES))
def test_event_clock_zero_domain_missing_grid_and_unaccepted_contract_roll(timeframe):
    from gx1.scripts.audit_entry_gc_foundation_v1 import reference_event_clock_means
    minutes,selected,events=_event_clock_fixture(timeframe)
    events['matching_clock_count'][:3]=0;events['matching_clock_sum_ns'][:3]=0
    reference=reference_event_clock_means(minutes,selected,events,timeframe)
    assert reference['observed_event_valid'].tolist()==[True]
    assert reference['observed_event_matching_clock_valid'].tolist()==[False]
    assert np.isnan(reference['observed_event_matching_clock_mean_ns']).all()
    assert reference['observed_event_send_to_capture_mean_ns'].tolist()==[20]
    # All physical minutes participate in full-grid/contract validation,
    # including an unaccepted final minute; partial accepted coverage is legal.
    selected[-1]=1
    rolled=reference_event_clock_means(minutes,selected,events,timeframe)
    assert rolled['observed_event_valid'].tolist()==[False]
    selected[-1]=-1
    unknown=reference_event_clock_means(minutes,selected,events,timeframe)
    assert unknown['observed_event_valid'].tolist()==[False]
    selected[-1]=0
    take=np.arange(len(minutes))!=3
    missing=reference_event_clock_means(minutes[take],selected[take],
        {name:value[take] for name,value in events.items()},timeframe)
    assert missing['observed_event_valid'].tolist()==[False]


def test_event_clock_no_discriminating_witness_is_not_a_numeric_pass():
    from gx1.scripts.audit_entry_gc_foundation_v1 import reference_event_clock_means,compare_event_clock_reference
    minutes,selected,events=_event_clock_fixture('M5')
    events['send_to_capture_count'][:3]=[1,1,0]
    reference=reference_event_clock_means(minutes,selected,events,'M5')
    with pytest.raises(RuntimeError,match='GC_EVENT_NUMERIC_NO_UNEQUAL_COUNT_WITNESS'):
        compare_event_clock_reference(reference,reference)


def test_event_clock_footer_rejects_test_intersection_before_any_payload(monkeypatch):
    from types import SimpleNamespace
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    seal=pd.Timestamp('2026-07-01T00:00:00Z').value
    rows=[SimpleNamespace(min=seal-3*MINUTE,max=seal-2*MINUTE,has_min_max=True),
        SimpleNamespace(min=seal-MINUTE,max=seal,has_min_max=True)]
    class Parquet:
        schema_arrow=SimpleNamespace(names=['bar_start_ns'])
        metadata=SimpleNamespace(num_row_groups=2,row_group=lambda i:
            SimpleNamespace(column=lambda _:SimpleNamespace(statistics=rows[i])))
        def read_row_groups(self,*args,**kwargs):
            raise AssertionError('unsafe footer must fail before any group decode')
    monkeypatch.setattr(audit.pq,'ParquetFile',lambda _:Parquet())
    with pytest.raises(RuntimeError,match='GC_EVENT_NUMERIC_ROW_GROUP_INTERSECTS_TEST'):
        audit.read_event_numeric_columns('never-read',['bar_start_ns'],seal-4*MINUTE,seal,seal)


@pytest.mark.parametrize('infinity', [np.inf,-np.inf])
def test_event_clock_unobserved_mean_requires_nan_not_infinity(infinity):
    from gx1.scripts.audit_entry_gc_foundation_v1 import reference_event_clock_means,compare_event_clock_reference
    minutes,selected,events=_event_clock_fixture('M5')
    events['matching_clock_count'][:3]=0;events['matching_clock_sum_ns'][:3]=0
    reference=reference_event_clock_means(minutes,selected,events,'M5')
    actual={name:value.copy() for name,value in reference.items()}
    actual['observed_event_matching_clock_mean_ns'][0]=infinity
    with pytest.raises(RuntimeError,match='GC_EVENT_NUMERIC_FINITE_MASK'):
        compare_event_clock_reference(reference,actual)



def _event_book_fixture(timeframe):
    from gx1.scripts.audit_entry_gc_foundation_v1 import GC_EVENT_REFERENCE_REDUCERS,GC_BOOK_REFERENCE_ADDITIVE,GC_BOOK_REFERENCE_MEANS,GC_ORDER_REFERENCE_MEANS
    minutes,selected,events=_event_clock_fixture(timeframe);size=len(minutes)
    accepted=np.isfinite(events['qualified_final_event_count'])
    for name in GC_EVENT_REFERENCE_REDUCERS:
        if name not in events:events[name]=np.where(accepted,0.,np.nan)
    pinned={'spread_state_min_ticks':[2,1,3],'spread_state_max_ticks':[5,9,4],
        'spread_change_count':[2,4,0],'spread_expansion_count':[1,2,0],'spread_contraction_count':[1,2,0],
        'spread_expansion_sum_ticks':[4,3,0],'spread_contraction_sum_ticks':[2,5,0],
        'spread_max_expansion_ticks':[4,2,0],'spread_max_contraction_ticks':[2,3,0],
        'receive_interval_count':[2,1,0],'receive_interval_sum_ns':[8,12,0],
        'receive_interval_min_ns':[3,12,np.nan],'receive_interval_max_ns':[5,12,np.nan],
        'ts_in_delta_clamped_low_count':[0,1,0],'ts_in_delta_clamped_high_count':[1,0,0],
        'undefined_event_clock_count':[0,2,1],'negative_send_to_capture_count':[1,0,0],
        'negative_matching_clock_count':[1,0,0]}
    for name,values in pinned.items():events[name][:3]=values
    for domain in ('send_to_capture','receive_minus_event','matching_clock'):
        for suffix in ('_min_ns','_max_ns'):
            events[domain+suffix][:3]=np.divide(events[domain+'_sum_ns'][:3],events[domain+'_count'][:3],
                out=np.full(3,np.nan),where=events[domain+'_count'][:3]>0)
    events['qualified_covered_ns']=np.where(accepted,0.,np.nan);events['qualified_covered_ns'][:3]=[MINUTE,MINUTE/2,0]
    book={name:np.zeros(size,bool) for name in ('ofi_valid','bbo_valid','order_count_valid')}
    book['ofi_valid'][:4]=True;book['bbo_valid'][[0,2,3]]=True;book['order_count_valid'][[0,3]]=True
    book['covered_ns']=np.zeros(size);book['covered_ns'][:4]=[MINUTE,MINUTE/2,MINUTE,MINUTE]
    book['order_covered_ns']=np.zeros(size);book['order_covered_ns'][[0,3]]=MINUTE
    for name,values in zip(GC_BOOK_REFERENCE_ADDITIVE,([7,-3,5,-2],[5,0,4,1],[0,2,1,0],[0,4,0,2],[3,1,0,5],[1,2,1,3])):
        book[name]=np.full(size,np.nan);book[name][:4]=values
    for name,values in zip(GC_BOOK_REFERENCE_MEANS,([-.5,.25,1],[2,4,9],[10,20,90],[30,20,10],[-1,.5,2])):
        book[name]=np.full(size,np.nan);book[name][[0,2,3]]=values
    for name,values in zip(GC_ORDER_REFERENCE_MEANS,([2,6],[3,9],[-.2,-.2],[5,15],[10,10/9])):
        book[name]=np.full(size,np.nan);book[name][[0,3]]=values
    book['opening_mid_tick']=np.full(size,np.nan);book['closing_mid_tick']=np.full(size,np.nan)
    book['opening_mid_tick'][[0,2,3]]=[100,102,105];book['closing_mid_tick'][[0,2,3]]=[101,104,107]
    book['mid_response_ticks']=book['closing_mid_tick']-book['opening_mid_tick']
    return minutes,selected,events,book


@pytest.mark.parametrize('timeframe', tuple(_CLOCK_MINUTES))
def test_event_book_asymmetric_all_five_tf_hand_reference_and_real_aggregate(monkeypatch,timeframe):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    from gx1.features import gc_order_flow_v1 as owner
    minutes,selected,events,book=_event_book_fixture(timeframe)
    event_producer=owner.aggregate_observed_event_diagnostics_v1;book_producer=owner.aggregate_observed_book_v1
    for name in ('aggregate_observed_event_diagnostics_v1','aggregate_observed_book_v1'):
        monkeypatch.setattr(owner,name,lambda *a,**k:(_ for _ in ()).throw(AssertionError('oracle delegates to producer')))
    reference,bounds,units,clock,strict=audit.reference_event_book_aggregates(minutes,selected,events,book,timeframe)
    assert len(reference)==84 and len(units)==84
    expected={'observed_event_qualified_final_event_count':12,'observed_event_spread_state_min_ticks':1,
        'observed_event_spread_state_max_ticks':9,'observed_event_spread_change_count':6,
        'observed_event_spread_expansion_count':3,'observed_event_spread_contraction_count':3,
        'observed_event_spread_expansion_sum_ticks':7,'observed_event_spread_contraction_sum_ticks':7,
        'observed_event_spread_max_expansion_ticks':4,'observed_event_spread_max_contraction_ticks':3,
        'observed_event_receive_interval_count':3,'observed_event_receive_interval_sum_ns':20,
        'observed_event_receive_interval_min_ns':3,'observed_event_receive_interval_max_ns':12,
        'observed_event_receive_interval_mean_ns':20/3,'observed_event_qualified_covered_ns':90*NS,
        'observed_event_qualified_events_per_observed_second':2/15,
        'observed_event_coverage_fraction':1.5/_CLOCK_MINUTES[timeframe],
        'observed_event_ts_in_delta_clamped_low_count':1,'observed_event_ts_in_delta_clamped_high_count':1,
        'observed_event_undefined_event_clock_count':3,'observed_event_negative_send_to_capture_count':1,
        'observed_event_negative_matching_clock_count':1,
        'observed_book_flow_accepted_minutes':4,'observed_book_accepted_minutes':3,'observed_order_accepted_minutes':2,
        'observed_book_covered_ns':180*NS,'observed_order_covered_ns':120*NS,
        'observed_book_ofi_contracts':7,'observed_book_bid_refill_contracts':10,'observed_book_ask_refill_contracts':3,
        'observed_book_bid_depletion_contracts':6,'observed_book_ask_depletion_contracts':9,'observed_book_quote_updates':7,
        'observed_book_tw_imbalance':.25,'observed_book_tw_spread_ticks':5,'observed_book_tw_bid_contracts':40,
        'observed_book_tw_ask_contracts':20,'observed_book_tw_microprice_offset_ticks':.5,
        'observed_order_tw_bid_orders':4,'observed_order_tw_ask_orders':6,
        'observed_order_tw_order_count_imbalance':-.2,'observed_order_tw_bid_contracts_per_order':10,
        'observed_order_tw_ask_contracts_per_order':50/9,'observed_book_opening_mid_tick':100,
        'observed_book_closing_mid_tick':107,'observed_book_mid_response_ticks':7,'observed_book_response_span_minutes':4,
        'observed_book_first_minute_ns':float(minutes[0]),'observed_book_last_minute_ns':float(minutes[3])}
    for name,value in expected.items():
        assert abs(float(reference[name][0])-value)<=float(bounds[name][0]),name
    assert reference['observed_book_response_contiguous'].tolist()==[False]
    actual={**book_producer(minutes,selected,book,timeframe),**event_producer(minutes,selected,events,timeframe)}
    report=audit.compare_event_book_reference(reference,actual,bounds,units)
    assert all(item['numeric_reference_qualified'] for item in report.values())
    # A shared BBO denominator would corrupt the order mean: own masks matter.
    actual['observed_order_tw_bid_orders'][0]=8/3
    with pytest.raises(RuntimeError,match='GC_EVENT_BOOK_NUMERIC_VALUE:observed_order_tw_bid_orders'):
        audit.compare_event_book_reference(reference,actual,bounds,units)


@pytest.mark.parametrize('timeframe', tuple(_CLOCK_MINUTES))
def test_event_book_roll_missing_nan_zero_and_partial_own_coverage(timeframe):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    minutes,selected,events,book=_event_book_fixture(timeframe)
    events['receive_interval_count'][:3]=0;events['receive_interval_sum_ns'][:3]=0
    events['receive_interval_min_ns'][:3]=np.nan;events['receive_interval_max_ns'][:3]=np.nan
    events['qualified_covered_ns'][:3]=0
    ref,_,_,_,_=audit.reference_event_book_aggregates(minutes,selected,events,book,timeframe)
    assert ref['observed_event_valid'].tolist()==[True]
    assert not ref['observed_event_receive_interval_valid'][0]
    assert np.isnan(ref['observed_event_receive_interval_mean_ns'][0])
    assert np.isnan(ref['observed_event_receive_interval_min_ns'][0])
    assert np.isnan(ref['observed_event_qualified_events_per_observed_second'][0])
    assert ref['observed_event_coverage_fraction'][0]==0
    selected[-1]=1
    rolled=audit.reference_event_book_aggregates(minutes,selected,events,book,timeframe)[0]
    assert not rolled['observed_book_valid'][0] and not rolled['observed_event_valid'][0]
    assert rolled['observed_book_accepted_minutes'][0]==3
    assert np.isnan(rolled['observed_book_tw_bid_contracts'][0])
    selected[-1]=0;take=np.arange(len(minutes))!=4
    missing=audit.reference_event_book_aggregates(minutes[take],selected[take],
        {k:v[take] for k,v in events.items()},{k:v[take] for k,v in book.items()},timeframe)[0]
    assert not missing['observed_book_valid'][0] and not missing['observed_event_valid'][0]
    book['covered_ns'][0]=MINUTE/2
    with pytest.raises(RuntimeError,match='GC_EVENT_BOOK_M1_COVERAGE_OR_MASK_DEPENDENCY'):
        audit.reference_event_book_aggregates(minutes,selected,events,book,timeframe)


def test_event_book_strict_m5_all_five_domain_and_endpoint_response():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    from gx1.scripts.rebuild_native_gc_inputs_v1 import legacy_strict_m5_surface
    minutes,selected,events,book=_event_book_fixture('M5')
    for name in ('ofi_valid','bbo_valid','order_count_valid'):book[name][:]=True
    book['covered_ns'][:]=MINUTE;book['order_covered_ns'][:]=MINUTE
    for i,name in enumerate(audit.GC_BOOK_REFERENCE_ADDITIVE):book[name]=np.arange(1,6,dtype=float)*(i+1)
    for names in (audit.GC_BOOK_REFERENCE_MEANS,audit.GC_ORDER_REFERENCE_MEANS):
        for i,name in enumerate(names):book[name]=np.arange(2,11,2,dtype=float)*(i+1)
    book['opening_mid_tick']=np.arange(100,105,dtype=float);book['closing_mid_tick']=np.arange(101,110,2,dtype=float)
    book['mid_response_ticks']=book['closing_mid_tick']-book['opening_mid_tick']
    strict,limits,units=audit.reference_event_book_aggregates(minutes,selected,events,book,'M5')[4]
    assert len(strict)==20 and strict['ofi_valid'].tolist()==[True]
    for i,name in enumerate(audit.GC_BOOK_REFERENCE_ADDITIVE):assert strict[name][0]==15*(i+1)
    for names in (audit.GC_BOOK_REFERENCE_MEANS,audit.GC_ORDER_REFERENCE_MEANS):
        for i,name in enumerate(names):assert strict[name][0]==6*(i+1)
    assert strict['mid_response_ticks'][0]==9
    metrics={name:np.ones(5,np.uint64) for name in ('volume','trades','buy','sell','unknown')}
    actual=legacy_strict_m5_surface(minutes,selected,metrics,{'volume':np.ones(5,bool),'delta':np.ones(5,bool)},
        np.zeros((5,4)),book,1,{})
    audit.compare_event_book_reference(strict,{name:actual[name] for name in strict},limits,units)
    for name in ('ofi_valid','bbo_valid','order_count_valid'):book[name][1]=False
    ref=audit.reference_event_book_aggregates(minutes,selected,events,book,'M5')
    assert ref[0]['observed_book_valid'][0]
    strict=ref[4][0]
    assert not strict['ofi_valid'][0] and not strict['bbo_valid'][0] and not strict['order_count_valid'][0]
    assert np.isnan(strict['ofi_contracts'][0]) and np.isnan(strict['mid_response_ticks'][0])


def test_event_book_active_370_fields_match_executable_registry_and_formula_owners():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    from gx1.features import gc_native_context_v1 as registry,gc_order_flow_v1 as owner
    declarations=audit.event_book_numeric_active_fields()
    assert [len(v) for v in declarations.values()]==[70,70,70,70,70,20]
    expected={name for columns in declarations.values() for name in columns.values()}
    actual=set()
    for tf,columns in declarations.items():
        table='M5_GC_FEATURES.parquet' if tf=='strict_m5' else 'TRADE_FEATURES_'+tf+'.parquet'
        actual.update(row['name'] for row in registry.REGISTRY['fields'] if row['table']==table and row['column'] in columns)
    assert len(expected)==370 and actual==expected
    assert expected<=set(registry.contract()['value_fields'])
    assert {name:operation for name,(operation,_) in audit.GC_EVENT_REFERENCE_REDUCERS.items()}==owner.EVENT_DIAGNOSTIC_REDUCERS_V1
    assert tuple(audit.GC_BOOK_REFERENCE_ADDITIVE)==owner.L1_FIELDS_V2[:6]
    assert tuple(audit.GC_BOOK_REFERENCE_MEANS)==owner.L1_FIELDS_V2[6:11]
    assert tuple(audit.GC_ORDER_REFERENCE_MEANS)==owner.L1_ORDER_COUNT_FIELDS_V1


def test_event_book_unobserved_field_is_not_numeric_qualification_and_infinity_rejected():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    expected={'unobserved':np.array([np.nan])};bounds={'unobserved':np.array([0.])};units={'unobserved':'nanoseconds'}
    result=audit.compare_event_book_reference(expected,expected,bounds,units)
    assert result['unobserved']['numeric_reference_qualified'] is False
    assert result['unobserved']['finite_published_values_compared']==0
    with pytest.raises(RuntimeError,match='GC_EVENT_BOOK_NUMERIC_MASK'):
        audit.compare_event_book_reference(expected,{'unobserved':np.array([np.inf])},bounds,units)
