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



def test_event_book_published_projection_without_availability_preserves_actual_grid(monkeypatch):
    import pyarrow.parquet as pq
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    minutes,selected,_,source=_event_book_fixture('M5')
    # Same M1 publication dictionary as the writer: no available_ns column.
    publication=pa.table({'bar_start_ns':minutes,'contract_index':selected,**source})
    buffer=pa.BufferOutputStream();pq.write_table(publication,buffer)
    parquet=pq.ParquetFile(pa.BufferReader(buffer.getvalue()))
    monkeypatch.setattr(audit.pq,'ParquetFile',lambda _:parquet)
    events={'bar_start_ns':minutes.copy(),'contract_index':selected.copy(),'available_ns':minutes+MINUTE}
    book,groups=audit.read_event_numeric_book('in-memory-book',int(minutes[0]),int(minutes[-1]+MINUTE),
        pd.Timestamp('2026-07-01T00:00:00Z').value,events)
    assert groups==[0] and 'available_ns' not in book
    assert np.array_equal(book['bar_start_ns'],minutes)
    assert np.array_equal(book['contract_index'],selected)
    assert set(book)-{'bar_start_ns','contract_index'} <= set(source)
    for name in ('bar_start_ns','contract_index'):
        shifted={key:value.copy() for key,value in events.items()};shifted[name][0]+=1
        with pytest.raises(RuntimeError,match='GC_EVENT_BOOK_SHARED_M1_CLOCK_OR_CONTRACT'):
            audit.read_event_numeric_book('in-memory-book',int(minutes[0]),int(minutes[-1]+MINUTE),
                pd.Timestamp('2026-07-01T00:00:00Z').value,shifted)



def _trade_numeric_fixture(timeframe, record_minutes=None, valid_minutes=None):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    size=_CLOCK_MINUTES[timeframe];start=pd.Timestamp('2024-05-01T22:00:00Z').value
    minutes=start+np.arange(size,dtype=np.int64)*MINUTE;selected=np.zeros(size,np.int16)
    records={0:[('B',1),('B',3),('A',2)],1:[('B',10),('A',4),('A',4),('N',5)],2:[('B',1000)]} if record_minutes is None else record_minutes
    valid=(0,1) if valid_minutes is None else valid_minutes
    trade={name:np.zeros(size,np.uint64) for name in audit.GC_TRADE_QUANTITIES}
    stats={name:np.zeros(size,np.uint64) for name in audit.GC_TRADE_SIZE_STATISTICS}
    trade['volume_valid']=np.zeros(size,bool);trade['signed_valid']=np.zeros(size,bool)
    for row,items in records.items():
        quantities=[v for _,v in items]
        trade['volume'][row]=sum(quantities);trade['trades'][row]=len(items)
        for code,side in (('B','buy'),('A','sell'),('N','unknown')):
            trade[side][row]=sum(v for label,v in items if label==code)
            stats[side+'_trade_records'][row]=sum(label==code for label,_ in items)
        stats['size_sum_contracts'][row]=sum(quantities);stats['size_sum_squared_contracts2'][row]=sum(v*v for v in quantities)
        stats['size_min_contracts'][row]=min(quantities);stats['size_max_contracts'][row]=max(quantities)
    trade['volume_valid'][list(valid)]=True
    trade['signed_valid']=trade['volume_valid'] & (trade['unknown']==0)
    return minutes,selected,trade,stats


@pytest.mark.parametrize('timeframe', tuple(_CLOCK_MINUTES))
def test_trade_numeric_pooled_hand_values_and_actual_producers_all_five_tf(monkeypatch,timeframe):
    import math
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    from gx1.features import gc_order_flow_v1 as owner
    from gx1.contracts import gc_continuous_source_v1 as source
    minutes,selected,trade,stats=_trade_numeric_fixture(timeframe)
    with monkeypatch.context() as guard:
        for name in ('timeframe_skeleton','relative_cvd','known_cumulative_contributions_v1','aggregate_observed_trade_size_v1'):
            guard.setattr(owner,name,lambda *a,**k:(_ for _ in ()).throw(AssertionError('oracle calls producer')))
        guard.setattr(source,'reset_segments',lambda *a,**k:(_ for _ in ()).throw(AssertionError('oracle delegates resets')))
        known,cvd=audit.reference_trade_cumulatives(minutes,selected,trade,stats)
        reference,bounds,units=audit.reference_trade_aggregates(minutes,selected,trade,stats,known,cvd,timeframe)
    assert len(reference)==53
    expected={'observed_volume':29,'observed_trades':7,'observed_buy_known':14,'observed_sell_known':10,
        'observed_unknown':5,'observed_known_delta_contracts':4,'observed_known_aggressor_fraction':24/29,
        'observed_avg_executed_trade_contracts':29/7,'observed_accepted_minutes':2,'observed_signed_minutes':1,
        'observed_coverage_fraction':2/_CLOCK_MINUTES[timeframe],'observed_signed_minute_fraction':.5,
        'observed_buy_trade_records':3,'observed_sell_trade_records':3,'observed_unknown_trade_records':1,
        'observed_buy_avg_executed_trade_contracts':14/3,'observed_sell_avg_executed_trade_contracts':10/3,
        'observed_unknown_avg_executed_trade_contracts':5,'observed_executed_trade_size_std_contracts':math.sqrt(356)/7,
        'observed_executed_trade_size_min_contracts':1,'observed_executed_trade_size_max_contracts':10,
        'observed_known_cumulative_buy_contracts':14,'observed_known_cumulative_sell_contracts':10,
        'observed_known_cumulative_unknown_contracts':5,'observed_known_cumulative_volume_contracts':29,
        'observed_known_cumulative_known_delta_contracts':4,'observed_known_cumulative_unknown_fraction':5/29,
        'observed_known_cumulative_observed_minutes':2,'observed_last_minute_ns':float(minutes[1])}
    for name,value in expected.items():assert abs(float(reference[name][0])-value)<=bounds[name][0],name
    assert not reference['volume_valid'][0] and not reference['signed_valid'][0]
    assert np.isnan(reference['volume'][0]) and np.isnan(reference['cvd_segment_delta_contracts'][0])
    masks={'volume':trade['volume_valid'],'delta':trade['signed_valid'],'cvd':trade['signed_valid']}
    actual,mapping=owner.timeframe_skeleton(minutes,selected,trade,masks,np.zeros((len(minutes),4),np.int64),timeframe,1)
    owner.add_structure_and_cvd(actual,owner.relative_cvd(minutes,selected,masks,trade),mapping,minutes)
    owner.add_known_cumulative_contributions_v1(actual,owner.known_cumulative_contributions_v1(minutes,selected,masks,trade),minutes)
    actual.update(owner.aggregate_observed_trade_size_v1(minutes,selected,trade,masks,stats,timeframe))
    actual={name:actual[name] for name in reference}
    audit.compare_trade_reference(reference,actual,bounds,units)
    actual['observed_buy_avg_executed_trade_contracts'][0]=6 # Incorrect mean of the two minute side means.
    with pytest.raises(RuntimeError,match='GC_EVENT_BOOK_NUMERIC_VALUE'):
        audit.compare_trade_reference(reference,actual,bounds,units)


@pytest.mark.parametrize('timeframe', tuple(_CLOCK_MINUTES))
def test_trade_numeric_unadmitted_roll_missing_clock_and_all_m1_parity(timeframe):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    minutes,selected,trade,stats=_trade_numeric_fixture(timeframe)
    selected[-1]=1
    known,cvd=audit.reference_trade_cumulatives(minutes,selected,trade,stats)
    out=audit.reference_trade_aggregates(minutes,selected,trade,stats,known,cvd,timeframe)[0]
    assert not out['observed_valid'][0] and out['observed_accepted_minutes'][0]==2
    assert np.isnan(out['observed_volume'][0])
    selected[-1]=0;take=np.arange(len(minutes))!=len(minutes)-1
    mt,st=minutes[take],selected[take];tr={k:v[take] for k,v in trade.items()};ss={k:v[take] for k,v in stats.items()}
    known,cvd=audit.reference_trade_cumulatives(mt,st,tr,ss)
    assert not audit.reference_trade_aggregates(mt,st,tr,ss,known,cvd,timeframe)[0]['observed_valid'][0]
    # The last raw trade is outside the accepted domain but must still reconcile.
    stats['size_sum_contracts'][2]+=1
    with pytest.raises(RuntimeError,match='GC_TRADE_REFERENCE_ALL_M1_RECORD_OR_QUANTITY_PARITY'):
        audit.reference_trade_cumulatives(minutes,selected,trade,stats)


def test_trade_numeric_large_almost_constant_sizes_exact_variance_and_absent_sides():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    minutes,selected,trade,stats=_trade_numeric_fixture('M5', {0:[('B',100000001),('B',100000002)]},(0,))
    assert int(stats['size_sum_squared_contracts2'][0])==20000000600000005
    known,cvd=audit.reference_trade_cumulatives(minutes,selected,trade,stats)
    out=audit.reference_trade_aggregates(minutes,selected,trade,stats,known,cvd,'M5')[0]
    assert out['observed_executed_trade_size_std_contracts'].tolist()==[.5]
    assert out['observed_buy_avg_executed_trade_contracts'].tolist()==[100000001.5]
    for side in ('sell','unknown'):
        assert out['observed_'+side+'_trade_records'].tolist()==[0]
        assert out['observed_'+side+'_trade_size_valid'].tolist()==[False]
        assert np.isnan(out['observed_'+side+'_avg_executed_trade_contracts']).all()


def test_trade_numeric_independent_ledgers_reset_utc_gap_invalid_and_roll():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    minutes,selected,trade,stats=_trade_numeric_fixture('M15',
        {0:[('B',2)],1:[('N',3)],2:[('A',5)],3:[('B',1000)],4:[('B',7)],5:[('A',11)],6:[('B',13)],7:[('A',17)]},
        (0,1,2,4,5,6,7))
    take=np.arange(len(minutes))<8;trade={k:v[take] for k,v in trade.items()};stats={k:v[take] for k,v in stats.items()}
    start=pd.Timestamp('2024-05-01T23:58:00Z').value
    minutes=start+np.array([0,1,2,3,4,6,7,8],np.int64)*MINUTE;selected=np.array([0,0,0,0,0,0,1,1],np.int16)
    known,cvd=audit.reference_trade_cumulatives(minutes,selected,trade,stats)
    np.testing.assert_array_equal(known['volume_contracts'],[2,5,5,np.nan,7,11,13,30])
    np.testing.assert_array_equal(known['known_delta_contracts'],[2,2,-5,np.nan,7,-11,13,-4])
    np.testing.assert_array_equal(known['observed_minutes'],[1,2,1,np.nan,1,1,1,2])
    np.testing.assert_array_equal(cvd['cvd_segment_delta_contracts'],[2,np.nan,-5,np.nan,7,-11,13,-4])
    np.testing.assert_array_equal(cvd['cvd_segment_minutes'],[1,np.nan,1,np.nan,1,1,1,2])


def test_trade_numeric_d1_cumulative_endpoint_resets_at_utc_not_tf_open():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    size=_CLOCK_MINUTES['D1']
    minutes,selected,trade,stats=_trade_numeric_fixture('D1',{i:[('B',1)] for i in range(size)},tuple(range(size)))
    known,cvd=audit.reference_trade_cumulatives(minutes,selected,trade,stats)
    out=audit.reference_trade_aggregates(minutes,selected,trade,stats,known,cvd,'D1')[0]
    assert out['volume_valid'][0] and out['signed_valid'][0] and out['observed_valid'][0]
    assert out['volume'][0]==out['observed_volume'][0]==1440
    assert out['cvd_segment_volume_contracts'][0]==out['cvd_segment_delta_contracts'][0]==1320
    assert out['observed_known_cumulative_volume_contracts'][0]==1320
    assert out['observed_known_cumulative_observed_minutes'][0]==1320


def test_trade_numeric_actual_registry_211_and_all_tf_auxiliary_scope():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    from gx1.features import gc_native_context_v1 as owner,gc_order_flow_v1 as formulas
    declared=audit.trade_numeric_active_fields()
    assert [len(v) for v in declared.values()]==[46,46,46,39,34]
    expected={name for fields in declared.values() for name in fields.values()};actual=set()
    for tf,columns in declared.items():
        tables={'TRADE_FEATURES_'+tf+'.parquet','retained::TRADE_FEATURES_'+tf+'.parquet'}
        actual.update(field['name'] for field in owner.REGISTRY['fields'] if field['table'] in tables and field['column'] in columns)
    assert len(expected)==211 and actual==expected and expected<=set(owner.contract()['value_fields'])
    assert audit.GC_TRADE_SIZE_STATISTICS==formulas.TRUSTED_TRADE_SIZE_FIELDS_V1
    assert audit.GC_TRADE_CVD_FIELDS==formulas.CVD_FIELDS
    assert len(audit.trade_reference_units())==53


def test_trade_numeric_real_tiny_projection_nine_and_ten_columns_without_core_availability(monkeypatch):
    import pyarrow.parquet as pq
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    minutes,selected,trade,stats=_trade_numeric_fixture('M5')
    core={'bar_start_ns':minutes,'contract_index':selected,**trade}
    size={'bar_start_ns':minutes,'available_ns':minutes+MINUTE,'contract_index':selected,**stats}
    readers={};open_parquet=pq.ParquetFile
    for key,columns in [('core',core),('sizes',size)]:
        stream=pa.BufferOutputStream();pq.write_table(pa.table(columns),stream)
        readers[key]=open_parquet(pa.BufferReader(stream.getvalue()))
    monkeypatch.setattr(audit.pq,'ParquetFile',lambda path:readers[str(path)])
    args=({'m1_trade':{'path':'core'},'m1_trade_sizes':{'path':'sizes'}},int(minutes[0]),int(minutes[-1]+MINUTE),
        pd.Timestamp('2026-07-01T00:00:00Z').value)
    projected,projected_sizes,groups,size_groups=audit.read_trade_numeric_inputs(*args)
    assert len(projected)==9 and 'available_ns' not in projected and len(projected_sizes)==7
    assert groups==size_groups==[0]
    # Each actual shared key and the size-only availability is checked independently.
    for name in ('bar_start_ns','contract_index','available_ns'):
        changed={key:value.copy() for key,value in size.items()};changed[name][0]+=1
        stream=pa.BufferOutputStream();pq.write_table(pa.table(changed),stream)
        readers['sizes']=open_parquet(pa.BufferReader(stream.getvalue()))
        with pytest.raises(RuntimeError,match='GC_TRADE_REFERENCE_PUBLISHED_M1_JOIN_OR_AVAILABILITY'):
            audit.read_trade_numeric_inputs(*args)


@pytest.mark.parametrize('name', ['bar_start_ns','available_ns','contract_index'])
def test_trade_numeric_exact_grid_rejects_one_ns_or_contract_change(name):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    minutes,selected,trade,stats=_trade_numeric_fixture('M5')
    known,cvd=audit.reference_trade_cumulatives(minutes,selected,trade,stats)
    reference,bounds,units=audit.reference_trade_aggregates(minutes,selected,trade,stats,known,cvd,'M5')
    actual={key:value.copy() for key,value in reference.items()}
    actual['contract_index']=actual['contract_index'].astype(np.int16)
    actual[name][0]+=1
    with pytest.raises(RuntimeError,match='GC_TRADE_REFERENCE_EXACT_TF_GRID'):
        audit.compare_trade_reference(reference,actual,bounds,units)


def test_trade_numeric_unadmitted_side_quantity_requires_own_record_before_mask():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    minutes,selected,trade,stats=_trade_numeric_fixture('M5')
    stats['buy_trade_records'][2]=0;stats['sell_trade_records'][2]=1
    with pytest.raises(RuntimeError,match='GC_TRADE_REFERENCE_M1_SIDE_VOLUME_WITHOUT_RECORDS'):
        audit.reference_trade_cumulatives(minutes,selected,trade,stats)


@pytest.mark.parametrize('timeframe', tuple(_CLOCK_MINUTES))
def test_trade_numeric_full_unsigned_unknown_keeps_known_contributions_all_tf(timeframe):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    size=_CLOCK_MINUTES[timeframe]
    records={i:[('B',1)] for i in range(size)};records[1]=[('N',3)]
    minutes,selected,trade,stats=_trade_numeric_fixture(timeframe,records,tuple(range(size)))
    known,cvd=audit.reference_trade_cumulatives(minutes,selected,trade,stats)
    out,bounds,_=audit.reference_trade_aggregates(minutes,selected,trade,stats,known,cvd,timeframe)
    assert out['volume_valid'][0] and not out['signed_valid'][0]
    assert out['volume'][0]==size+2 and out['trades'][0]==size and out['unknown'][0]==3
    assert abs(out['known_aggressor_fraction'][0]-(size-1)/(size+2))<=bounds['known_aggressor_fraction'][0]
    for name in ('buy','sell','delta_contracts')+audit.GC_TRADE_CVD_FIELDS:assert np.isnan(out[name][0]),name
    assert out['observed_valid'][0] and not out['observed_signed_complete'][0]
    assert out['observed_buy_known'][0]==size-1 and out['observed_sell_known'][0]==0
    assert out['observed_unknown'][0]==3 and out['observed_known_delta_contracts'][0]==size-1
    # H4/D1 cross UTC midnight: cumulative contributions retain their own
    # reset clock rather than being replaced by the complete TF quantities.
    buy,unknown,volume,count={'M5':(4,3,7,5),'M15':(14,3,17,15),'H1':(59,3,62,60),
        'H4':(120,0,120,120),'D1':(1320,0,1320,1320)}[timeframe]
    for name,value in {'buy_contracts':buy,'sell_contracts':0,'unknown_contracts':unknown,
            'volume_contracts':volume,'known_delta_contracts':buy,'observed_minutes':count}.items():
        assert out['observed_known_cumulative_'+name][0]==value,name


# Literal integer-grid SMC paths: no mocked producer pivots or future labels.
def _smc_ob_candles(mirror=False):
    high=np.array([9,10,11,14,12,11,10,11,15,14,13,12,11,10,9,8])*4
    low=high-16;low[4]=12
    close=high-4;close[8]=58
    open_=close+2;open_[8]=59
    candles=np.column_stack((open_,high,low,close)).astype(np.int64)
    if mirror:candles=np.column_stack((200-open_,200-low,200-high,200-close)).astype(np.int64)
    return candles,np.arange(10,26,dtype=np.uint64)


def _smc_m1_candles(timeframe, candles):
    size=_CLOCK_MINUTES[timeframe];duration=size*MINUTE
    origin=pd.Timestamp('2024-05-02T00:00:00Z').value
    origin=((origin-(22*60*MINUTE)%duration)//duration)*duration+(22*60*MINUTE)%duration
    n=len(candles)*size;minutes=origin+np.arange(n,dtype=np.int64)*MINUTE
    repeated=np.repeat(candles,size,axis=0)
    return {'bar_start_ns':minutes,'contract_index':np.zeros(n,np.int16),
        'volume_valid':np.ones(n,bool),'volume':np.repeat(np.arange(10,10+len(candles),dtype=np.uint64),size),
        **{name:repeated[:,column].copy() for column,name in enumerate(('open_tick','high_tick','low_tick','close_tick'))}}


@pytest.mark.parametrize('mirror',[False,True])
def test_smc_numeric_literal_pivot_ties_right_three_and_one_shot_events(monkeypatch,mirror):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    from gx1.features import smc_v1 as owner
    high=np.array([9,10,11,14,12,11,10,15,15,14,13,12,16,17,16,18])*4
    low=np.zeros(len(high),np.int64)
    close=np.array([8,9,10,13,11,10,9,13,14,13,12,11,14,16,14,16])*4
    open_=close+1;candles=np.column_stack((open_,high,low,close))
    if mirror:candles=np.column_stack((200-open_,200-low,200-high,200-close))
    producer=owner.replay_volume_order_block_proxies_v1
    def forbidden(*args,**kwargs):raise AssertionError('independent oracle delegated producer')
    with monkeypatch.context() as blocked:
        for name in ('replay_volume_order_block_proxies_v1','replay_smc_causal_structure_v2','_detect_swing_pivots'):
            blocked.setattr(owner,name,forbidden)
        out,pivots=audit.reference_smc_sequence(candles,np.arange(10,26,dtype=np.uint64))
        for end in range(1,len(candles)+1):
            prefix=audit.reference_smc_sequence(candles[:end],np.arange(10,10+end,dtype=np.uint64))[0]
            for name in out:np.testing.assert_equal(prefix[name],out[name][:end])
    direction='down' if mirror else 'up';level='low' if mirror else 'high'
    assert pivots['last_'+level+'_idx'][:6].tolist()==[-1]*6
    assert pivots['last_'+level+'_idx'][6:12].tolist()==[3,3,3,3,7,8]
    assert np.flatnonzero(out['sweep_'+direction+'_event']==1).tolist()==[7,12]
    assert np.flatnonzero(out['bos_'+direction]==1).tolist()==[13]
    actual=producer(*[candles[:,j]*.25 for j in range(4)],np.arange(10,26,dtype=np.uint64))
    for name,values in out.items():
        expected=values*.25 if name.endswith(('_low','_high')) else values
        np.testing.assert_equal(actual[name],expected)


@pytest.mark.parametrize('mirror',[False,True])
def test_smc_numeric_literal_origin_zone_first_retouch_equality_and_invalidation(mirror):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    candles,volume=_smc_ob_candles(mirror);out,_=audit.reference_smc_sequence(candles,volume)
    side='bear' if mirror else 'bull';p='ob_'+side+'_'
    assert out['structure_available'].tolist()==[0]*16
    assert out[p+'input_valid'][:6].tolist()==[0]*6 and out[p+'input_valid'][6]==1
    assert np.isnan(out[p+'active'][:6]).all() and out[p+'active'][6:8].tolist()==[0,0]
    assert out[p+'created'][8]==out[p+'active'][8]==1 and out[p+'first_retouch'][8]==0
    assert out[p+'origin_volume_contracts'][8:16].tolist()==[17]*8
    assert out[p+'age_bars'][8:16].tolist()==list(range(8))
    lower,upper=(156,172) if mirror else (28,44)
    assert out[p+'low'][8]==lower and out[p+'high'][8]==upper
    assert out[p+'first_retouch'][9:11].tolist()==[1,0] and out[p+'active'][15]==1
    changed=candles.copy()
    if mirror:changed[9,1]=176;changed[9,3]=174
    else:changed[9,2]=24;changed[9,3]=26
    invalid=audit.reference_smc_sequence(changed,volume)[0]
    assert invalid[p+'active'][9]==invalid[p+'first_retouch'][9]==0
    for field in ('low','high','age_bars','origin_volume_contracts'):assert np.isnan(invalid[p+field][9])


@pytest.mark.parametrize('mirror',[False,True])
def test_smc_numeric_created_can_invalidate_immediately_without_current_origin(mirror):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    high=np.array([30,10,11,12,14,12,11,10,16])*4;low=high-12;low[0]=80
    close=np.array([21,9,10,11,13,11,10,9,15])*4;open_=close-1;open_[0]=116
    candles=np.column_stack((open_,high,low,close))
    if mirror:candles=np.column_stack((200-open_,200-low,200-high,200-close))
    out,_=audit.reference_smc_sequence(candles,np.arange(10,19,dtype=np.uint64))
    p='ob_'+('bear' if mirror else 'bull')+'_'
    assert out[p+'created'][8]==1 and out[p+'active'][8]==out[p+'first_retouch'][8]==0
    assert np.isnan(out[p+'origin_volume_contracts'][8]) and np.isnan(out[p+'age_bars'][8])


@pytest.mark.parametrize('timeframe',tuple(_CLOCK_MINUTES))
def test_smc_numeric_all_tf_partial_price_ignores_unadmitted_extremes(timeframe):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    size=_CLOCK_MINUTES[timeframe]
    m1=_smc_m1_candles(timeframe,np.array([[100,1000,1,100]],np.int64))
    m1['volume_valid'][:]=False;m1['volume_valid'][[1,size-2]]=True
    m1['volume'][:]=999999;m1['volume'][[1,size-2]]=[2,3]
    for name,values in zip(audit.GC_SMC_M1_COLUMNS[4:],([100,120],[110,130],[90,115],[105,125])):
        m1[name][[1,size-2]]=values
    out,bounds,units=audit.reference_smc_aggregates(m1,timeframe,100000000,10**9)
    assert out['observed_volume'].tolist()==[5] and out['observed_valid'].tolist()==[True]
    assert out['volume_valid'].tolist()==[False]
    for name,value in zip(audit.GC_SMC_PRICE_FIELDS,(10,13,9,12.5)):
        assert out['observed_'+name][0]==value and np.isnan(out[name][0])
    assert out['observed_first_minute_ns'][0]==m1['bar_start_ns'][1]
    assert out['observed_last_minute_ns'][0]==m1['bar_start_ns'][size-2]
    report=audit.compare_trade_reference(out,{k:v.copy() for k,v in out.items()},bounds,units)
    assert report['observed_proxy_ob_bull_low']['numeric_reference_qualified'] is False
    # Physical missing M1 is distinct from a complete clock with false admission.
    missing={name:np.delete(value,0) for name,value in m1.items()}
    assert not audit.reference_smc_aggregates(missing,timeframe,100000000,10**9)[0]['observed_valid'][0]


@pytest.mark.parametrize('timeframe',tuple(_CLOCK_MINUTES))
def test_smc_numeric_all_tf_full_partial_missing_and_contract_reset_domains(timeframe):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    candles,_=_smc_ob_candles();size=_CLOCK_MINUTES[timeframe]
    m1=_smc_m1_candles(timeframe,candles)
    full=audit.reference_smc_aggregates(m1,timeframe,100000000,10**9)[0]
    assert full['ob_bull_created'][8]==full['observed_proxy_ob_bull_created'][8]==1
    assert full['ob_bull_origin_volume_contracts'][8]==17*size
    assert full['observed_proxy_ob_bull_first_retouch'][9]==1
    # A partial candle preserves the observed zone; strict replay starts over.
    m1['volume_valid'][9*size:10*size]=False;m1['volume_valid'][[9*size+1,10*size-2]]=True
    partial=audit.reference_smc_aggregates(m1,timeframe,100000000,10**9)[0]
    assert partial['observed_proxy_ob_bull_active'][9]==1 and partial['observed_proxy_ob_bull_first_retouch'][9]==1
    assert partial['observed_proxy_skipped_candles'][9]==0 and np.isnan(partial['ob_bull_active'][9])
    # A whole missing same-contract candle keeps pivots but censors zones/origins.
    m1['volume_valid'][9*size:10*size]=False
    missing=audit.reference_smc_aggregates(m1,timeframe,100000000,10**9)[0]
    assert missing['observed_proxy_ob_bull_input_valid'][10]==1
    assert missing['observed_proxy_skipped_candles'][10]==1
    assert np.isnan(missing['observed_proxy_ob_bull_active'][10])
    assert not missing['observed_proxy_ob_bull_state_valid'][10] and missing['ob_bull_input_valid'][10]==0
    # Even a contract change in a filtered-out bucket resets observed pivots.
    m1['contract_index'][9*size]=1
    rolled=audit.reference_smc_aggregates(m1,timeframe,100000000,10**9)[0]
    assert not rolled['observed_contract_valid'][9]
    assert rolled['observed_proxy_ob_bull_input_valid'][10]==0
    assert np.isnan(rolled['observed_proxy_elapsed_minutes'][10])


def test_smc_numeric_rational_price_bounds_cover_both_owner_orders_and_tick_separation():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    for tick in (23141,26003,100000001):
        expected,bound=audit.smc_price_reference(tick,100000000,10**9)
        assert abs(float(tick)*(100000000/10**9)-expected)<=bound
        assert abs(float(tick)*100000000/10**9-expected)<=bound
    assert audit.validate_smc_tick_order(100000001,100000000,10**9,1e-12)['ordering_separation_proven']
    with pytest.raises(RuntimeError,match='TICK_TOLERANCE_NOT_SEPARATED'):
        audit.validate_smc_tick_order(100000001,1,10**20,1e-12)


def test_smc_numeric_real_eight_column_projection_and_admission_schema(monkeypatch):
    import pyarrow.parquet as pq
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    m1=_smc_m1_candles('M5',np.array([[100,110,90,105]],np.int64))
    stream=pa.BufferOutputStream();pq.write_table(pa.table(m1),stream)
    reader=pq.ParquetFile(pa.BufferReader(stream.getvalue()))
    monkeypatch.setattr(audit.pq,'ParquetFile',lambda path:reader)
    values,groups=audit.read_smc_numeric_inputs({'m1_smc':{'path':'core'}},int(m1['bar_start_ns'][0]),
        int(m1['bar_start_ns'][-1]+MINUTE),pd.Timestamp('2026-07-01T00:00:00Z').value)
    assert set(values)==set(audit.GC_SMC_M1_COLUMNS) and len(values)==8 and 'available_ns' not in values and groups==[0]
    for bad in (np.array([2]*5),np.array([np.nan]*5)):
        changed={k:v.copy() for k,v in m1.items()};changed['volume_valid']=bad
        with pytest.raises(RuntimeError,match='M1_SCHEMA'):audit.validate_smc_m1_schema(changed)
    changed={k:v.copy() for k,v in m1.items()};changed['volume']=changed['volume'][:-1]
    with pytest.raises(RuntimeError,match='M1_SCHEMA'):audit.validate_smc_m1_schema(changed)


def test_smc_numeric_actual_registry_198_and_auxiliary_outputs():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    from gx1.features import gc_native_context_v1 as owner,smc_v1 as formulas
    declared,book=audit.smc_numeric_active_fields()
    assert [len(fields) for fields in declared.values()]==[53,52,32,32,28]
    expected={name for fields in declared.values() for name in fields.values()}|set(book.values());actual=set()
    for tf,columns in declared.items():
        tables={'TRADE_FEATURES_'+tf+'.parquet','retained::TRADE_FEATURES_'+tf+'.parquet'}
        actual.update(field['name'] for field in owner.REGISTRY['fields'] if field['table'] in tables and field['column'] in columns)
    actual.update(field['name'] for field in owner.REGISTRY['fields'] if field['table']=='M5_GC_FEATURES.parquet' and field['name'] in book.values())
    assert len(expected)==198 and actual==expected and expected<=set(owner.contract()['value_fields'])
    assert audit.GC_SMC_FIELDS==formulas.VOLUME_ORDER_BLOCK_PROXY_FIELDS_V1 and len(audit.smc_reference_units())==65
    trade={name for fields in audit.trade_numeric_active_fields().values() for name in fields.values()}
    event={name for fields in audit.event_book_numeric_active_fields().values() for name in fields.values()}
    assert not expected.intersection(trade|event)


# Independent profile/footprint hand references. Existing 77 cases above stay fixed.
@pytest.mark.parametrize('hist,expected', [
    ({104:(0,0,6),100:(3,1,0),102:(4,0,0),101:(0,2,0)},
     {'volume_contracts':16,'vwap_usd':10.2125,'poc_usd':10.4,'q25_usd':10.,
      'median_usd':10.2,'q75_usd':10.4,'price_levels':4,
      'max_level_delta_contracts':4,'min_level_delta_contracts':-2,
      'buy_poc_usd':10.2,'sell_poc_usd':10.1}),
    ({100:(0,4,0),101:(0,2,0),104:(0,0,7)},
     {'volume_contracts':13,'vwap_usd':133/13,'poc_usd':10.4,'q25_usd':10.,
      'median_usd':10.4,'q75_usd':10.4,'price_levels':3,
      'max_level_delta_contracts':0,'min_level_delta_contracts':-4,
      'buy_poc_usd':np.nan,'sell_poc_usd':10.}),
    ({100:(0,0,2),102:(0,0,3)},
     {'volume_contracts':5,'vwap_usd':10.12,'poc_usd':10.2,'q25_usd':10.,
      'median_usd':10.2,'q75_usd':10.2,'price_levels':2,
      'max_level_delta_contracts':0,'min_level_delta_contracts':0,
      'buy_poc_usd':np.nan,'sell_poc_usd':np.nan}),
])
def test_profile_numeric_literal_histogram_quantiles_and_known_footprint(monkeypatch,hist,expected):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    from gx1.features import gc_order_flow_v1 as owner
    with monkeypatch.context() as no_delegate:
        for name in ('histogram_features','value_area_v2','profile_values_v2'):
            no_delegate.setattr(owner,name,lambda *a,**k:(_ for _ in ()).throw(AssertionError('profile oracle delegates')))
        values,bounds=audit.reference_profile_histogram(hist,100000000,10**9)
    for name,value in expected.items():
        if np.isnan(value):assert np.isnan(values[name]),name
        else:assert values[name]==pytest.approx(value,rel=0,abs=1e-13),name
    ticks=list(hist);quantities=np.array(list(hist.values()),np.uint64)
    actual=owner.histogram_features(ticks,*quantities.T,100000000)
    for name,value in actual.items():
        if np.isnan(value):assert np.isnan(values[name]),name
        else:assert abs(values[name]-value)<=bounds[name],name


@pytest.mark.parametrize('hist,poc,low,high', [
    ({100:(4,0,0),102:(8,0,0),104:(4,0,0)},102,100,102),
    ({102:(4,0,0),100:(4,0,0)},100,100,102),
    ({100:(3,0,0),101:(4,0,0),102:(3,0,0)},101,100,101),
])
def test_profile_numeric_literal_contiguous_70pct_value_area_gaps_and_ties(hist,poc,low,high):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    from gx1.features import gc_order_flow_v1 as owner
    values,bounds=audit.reference_profile_histogram(hist,100000000,10**9)
    assert values['poc_usd']==pytest.approx(poc/10,rel=0,abs=1e-13)
    assert values['val_usd']==pytest.approx(low/10,rel=0,abs=1e-13)
    assert values['vah_usd']==pytest.approx(high/10,rel=0,abs=1e-13)
    assert owner.value_area_v2({price:sum(qty) for price,qty in hist.items()})==(poc,low,high)
    # The empty ticks are interval space, not extra volume or price levels.
    assert values['price_levels']==len(hist)


def test_profile_numeric_independent_ny18_dst_and_partial_initial_prefix(monkeypatch):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    from gx1.features import gc_order_flow_v1 as owner
    stamps=pd.DatetimeIndex(['2024-03-10T21:55Z','2024-03-10T22:00Z',
        '2024-11-03T22:55Z','2024-11-03T23:00Z','2024-05-01T00:00Z']).asi8
    expected=pd.DatetimeIndex(['2024-03-09T23:00Z','2024-03-10T22:00Z',
        '2024-11-02T22:00Z','2024-11-03T23:00Z','2024-04-30T22:00Z']).asi8
    with monkeypatch.context() as no_delegate:
        no_delegate.setattr(owner,'profile_anchors_v2',lambda *a,**k:(_ for _ in ()).throw(AssertionError('session oracle delegates')))
        actual=audit.reference_profile_anchors(stamps)
    assert actual.dtype==np.int64
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(owner.profile_anchors_v2(stamps),expected)
    assert ((stamps[-1]+5*MINUTE-actual[-1])//MINUTE)==125


def test_profile_numeric_literal_m5_three_twelve_windows_eviction_and_owner_independence(monkeypatch):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    from gx1.features import gc_order_flow_v1 as owner
    start=pd.Timestamp('2024-05-01T22:00Z').value
    starts=start+np.arange(13,dtype=np.int64)*5*MINUTE
    state={};reference=[];bounds=[]
    with monkeypatch.context() as no_delegate:
        for name in ('DynamicProfileStateV2','profile_values_v2','value_area_v2','dynamic_profiles_v2'):
            no_delegate.setattr(owner,name,lambda *a,**k:(_ for _ in ()).throw(AssertionError('dynamic oracle delegates')))
        for i,stamp in enumerate(starts):
            row,bound=audit.reference_profile_step(state,int(stamp),{100+i:i+1},0,103+i,start,
                tick_size=10**9,price_scale=10**9)
            reference.append(row);bounds.append(bound)
    assert len(reference[0])==len(bounds[0])==51
    assert [r['rolling15_valid'] for r in reference]==[False,False]+[True]*11
    assert [r['rolling60_valid'] for r in reference]==[False]*11+[True,True]
    assert all(r['session_complete_valid'] for r in reference)
    assert reference[2]['rolling15_volume_contracts']==6
    assert reference[2]['rolling15_vwap_usd']==pytest.approx(304/3,rel=0,abs=1e-12)
    assert (reference[2]['rolling15_poc_usd'],reference[2]['rolling15_val_usd'],reference[2]['rolling15_vah_usd'])==(102,101,102)
    assert reference[11]['rolling60_volume_contracts']==78
    assert reference[11]['rolling60_vwap_usd']==pytest.approx(322/3,rel=0,abs=1e-12)
    assert np.isnan(reference[11]['rolling60_poc_prior_level_response_ticks'])
    assert reference[12]['rolling60_volume_contracts']==90
    assert reference[12]['rolling60_vwap_usd']==pytest.approx(4864/45,rel=0,abs=1e-12)
    assert reference[12]['rolling60_val_usd']==107
    assert reference[12]['anchored_segment_volume_contracts']==91
    assert reference[12]['anchored_segment_vwap_usd']==108
    assert reference[12]['anchored_segment_val_usd']==106
    assert reference[12]['rolling15_volume_contracts']==36
    assert reference[12]['rolling15_vwap_usd']==pytest.approx(1999/18,rel=0,abs=1e-12)
    assert reference[12]['rolling60_vwap_prior_level_response_ticks']==pytest.approx(23/3,rel=0,abs=1e-12)
    assert reference[12]['session_accepted_minutes']==reference[12]['session_expected_minutes']==65
    actual=owner.DynamicProfileStateV2(1.)
    for i,stamp in enumerate(starts):
        published=actual.step(int(stamp),{100+i:i+1},True,0,float(103+i),start)
        for name,value in published.items():
            if np.isnan(value):assert np.isnan(reference[i][name]),name
            else:assert abs(reference[i][name]-value)<=bounds[i][name],name


def test_profile_numeric_anchor_prior_response_partial_minutes_invalid_gap_and_roll():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    anchor=pd.Timestamp('2024-05-01T22:00Z').value
    starts=anchor+np.array([-15,-10,-5,0,5,10,15,25],np.int64)*MINUTE
    histograms=[{100:1},{101:2},{102:3},{110:4},{120:5},{},{130:7},{140:11}]
    contracts=[0,0,0,0,1,1,1,1];accepted=[5,5,5,5,2,0,3,5]
    close=[100,101,102,111,121,125,131,141];state={};rows=[]
    for i,stamp in enumerate(starts):
        row,_=audit.reference_profile_step(state,int(stamp),histograms[i],contracts[i],close[i],
            anchor-DAY if i<3 else anchor,accepted[i],tick_size=10**9,price_scale=10**9,valid=i!=5)
        rows.append(row)
    assert rows[2]['rolling15_valid'] and not rows[2]['session_complete_valid']
    assert not rows[3]['rolling15_valid'] and np.isnan(rows[3]['rolling15_poc_usd'])
    # Prior levels can remain observable across the new NY18 anchor.
    assert rows[3]['rolling15_poc_prior_level_response_ticks']==9
    assert rows[3]['anchored_segment_poc_prior_level_response_ticks']==9
    assert rows[3]['session_complete_valid'] and rows[3]['session_accepted_minutes']==5
    assert rows[3]['anchored_segment_volume_contracts']==4
    assert rows[4]['session_observed_volume_contracts']==5 and rows[4]['session_accepted_minutes']==2
    assert not rows[4]['session_complete_valid']
    assert np.isnan(rows[4]['anchored_segment_poc_prior_level_response_ticks'])
    assert rows[5]['session_accepted_minutes']==0 and not rows[5]['session_observed_valid']
    assert np.isnan(rows[5]['session_observed_volume_contracts'])
    assert rows[6]['anchored_segment_volume_contracts']==7
    assert rows[6]['session_observed_volume_contracts']==12 and rows[6]['session_accepted_minutes']==5
    assert np.isnan(rows[6]['anchored_segment_poc_prior_level_response_ticks'])
    assert rows[7]['anchored_segment_volume_contracts']==11
    assert rows[7]['session_observed_volume_contracts']==23 and rows[7]['session_accepted_minutes']==10
    assert rows[7]['session_expected_minutes']==30 and not rows[7]['rolling15_valid']
    assert np.isnan(rows[7]['anchored_segment_poc_prior_level_response_ticks'])
    initial,_=audit.reference_profile_step({},pd.Timestamp('2024-05-01T00:00Z').value,{100:2},0,100,
        pd.Timestamp('2024-04-30T22:00Z').value,tick_size=10**9,price_scale=10**9)
    assert initial['session_accepted_minutes']==5 and initial['session_expected_minutes']==125
    assert not initial['session_complete_valid']


def _profile_numeric_fixture(timeframe,partial=True):
    n=_CLOCK_MINUTES[timeframe];start=pd.Timestamp('2024-05-01T22:00Z').value
    clock=start+np.arange(n,dtype=np.int64)*MINUTE
    m1={'bar_start_ns':clock,'contract_index':np.zeros(n,np.int16),
        'volume_valid':np.ones(n,bool),'signed_valid':np.ones(n,bool),
        'volume':np.ones(n,np.uint64),'buy':np.ones(n,np.uint64),
        'sell':np.zeros(n,np.uint64),'unknown':np.zeros(n,np.uint64),'close_tick':np.full(n,100,np.int64)}
    m1['volume'][:2]=[4,12];m1['buy'][:2]=[3,4];m1['sell'][:2]=[1,2];m1['unknown'][1]=6
    m1['signed_valid'][1]=False;m1['close_tick'][1]=104
    if partial:m1['volume_valid'][2]=m1['signed_valid'][2]=False
    rows=[]
    for i in range(n):
        if not m1['volume_valid'][i]:continue
        levels=[(100,3,1,0)] if i==0 else ([(101,0,2,0),(102,4,0,0),(104,0,0,6)] if i==1 else [(100,1,0,0)])
        rows.extend((int(clock[i]),0,price,buy,sell,unknown,bool(m1['signed_valid'][i]))
            for price,buy,sell,unknown in levels)
    names=('bar_start_ns','contract_index','price_tick','buy','sell','unknown','signed_valid')
    dtypes=(np.int64,np.int16,np.int64,np.uint64,np.uint64,np.uint64,bool)
    ladder={name:np.array([row[j] for row in rows],dtype) for j,(name,dtype) in enumerate(zip(names,dtypes))}
    return m1,ladder


@pytest.mark.parametrize('timeframe',tuple(_CLOCK_MINUTES))
def test_profile_numeric_all_tf_partial_histogram_exact_m5_snapshot_and_domains(timeframe):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    m1,ladder=_profile_numeric_fixture(timeframe);n=len(m1['bar_start_ns'])
    # Continue the same minute over a real batch boundary without replay/sort.
    batches=[{k:v[:2] for k,v in ladder.items()},{k:v[2:] for k,v in ladder.items()}]
    refs,bounds,report=audit.reference_profile_ladder(m1,batches,100000000,10**9)
    out=refs[timeframe];grid=audit.reference_profile_grid(m1,timeframe)
    assert len(out)==79 and len(refs['strict_m5'])==54
    assert grid['observed_valid'].tolist()==[True] and grid['volume_valid'].tolist()==[False]
    assert grid['signed_valid'].tolist()==[False] and grid['accepted_minutes'].tolist()==[n-1]
    assert grid['observed_volume'][0]==n+13
    assert out['observed_vwap_usd'][0]==pytest.approx((100*n+1334)/(n+13)/10,rel=0,abs=1e-12)
    assert out['observed_price_levels'][0]==4 and out['observed_known_min_level_delta_contracts'][0]==-2
    assert out['observed_buy_known_poc_usd'][0]==10. and out['observed_sell_known_poc_usd'][0]==10.1
    assert np.isnan(out['vwap_usd']).all() and np.isnan(out['buy_poc_usd']).all()
    assert out['observed_profile_source_coverage_fraction'][0]==(.8 if timeframe=='M5' else 1.)
    assert out['observed_profile_snapshot_age_minutes'].tolist()==[0.]
    np.testing.assert_array_equal(out['observed_profile_snapshot_available_ns'],out['available_ns'])
    assert report['ladder_passes']==1 and report['seen_minutes']==report['admitted_minutes']==n-1
    # A full clock with rejected input is distinct from missing clock and roll.
    missing={name:np.delete(value,2) for name,value in m1.items()}
    with pytest.raises(RuntimeError,match='GC_PROFILE_REFERENCE_M1_SCHEMA'):
        audit.reference_profile_grid(missing,timeframe)
    rolled={name:value.copy() for name,value in m1.items()};rolled['contract_index'][2]=1
    assert not audit.reference_profile_grid(rolled,timeframe)['observed_valid'][0]
    # Validity at TF end uses its exact final M5, even after earlier valid state.
    rejected={name:value.copy() for name,value in m1.items()}
    rejected['volume_valid'][-5:]=False;rejected['signed_valid'][-5:]=False
    keep=ladder['bar_start_ns']<m1['bar_start_ns'][-5]
    cut={name:value[keep] for name,value in ladder.items()}
    rejected_out=audit.reference_profile_ladder(rejected,[cut],100000000,10**9)[0][timeframe]
    assert not rejected_out['observed_profile_anchored_segment_valid'][0]
    assert np.isnan(rejected_out['observed_profile_anchored_segment_poc_usd'][0])
    assert rejected_out['observed_profile_source_coverage_fraction'].tolist()==[0.]
    assert rejected_out['observed_profile_snapshot_age_minutes'].tolist()==[0.]


@pytest.mark.parametrize('timeframe',tuple(_CLOCK_MINUTES))
def test_profile_numeric_all_tf_unsigned_strict_histogram_with_unknown_known_footprint(timeframe):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    m1,ladder=_profile_numeric_fixture(timeframe,partial=False);n=len(m1['bar_start_ns'])
    refs,_,report=audit.reference_profile_ladder(m1,[ladder],100000000,10**9)
    out=refs[timeframe];grid=audit.reference_profile_grid(m1,timeframe)
    assert grid['volume_valid'].tolist()==[True] and grid['signed_valid'].tolist()==[False]
    assert grid['observed_volume'].tolist()==[n+14]
    assert out['poc_usd'][0]==out['observed_poc_usd'][0]==10.
    assert out['vwap_usd'][0]==pytest.approx((100*n+1434)/(n+14)/10,rel=0,abs=1e-12)
    assert out['observed_known_max_level_delta_contracts'].tolist()==[n]
    assert out['observed_known_min_level_delta_contracts'].tolist()==[-2]
    for name in ('max_level_delta_contracts','min_level_delta_contracts','buy_poc_usd','sell_poc_usd'):
        assert np.isnan(out[name]).all(),name
    assert np.isfinite(out['observed_buy_known_poc_usd']).all()
    assert report['seen_minutes']==report['admitted_minutes']==n


@pytest.mark.parametrize('failure',('missing','float_clock','float_contract','mask_dtype','quantity_dtype','signed_unknown','signed_unadmitted'))
def test_profile_numeric_m1_schema_rejects_malformed_clock_dtype_and_signed_domains(failure):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    m1,_=_profile_numeric_fixture('M5')
    if failure=='missing':del m1['close_tick']
    elif failure=='float_clock':m1['bar_start_ns']=m1['bar_start_ns'].astype(np.float64)
    elif failure=='float_contract':m1['contract_index']=m1['contract_index'].astype(np.float64)
    elif failure=='mask_dtype':m1['volume_valid']=m1['volume_valid'].astype(np.int8)
    elif failure=='quantity_dtype':m1['buy']=m1['buy'].astype(np.int64)
    elif failure=='signed_unknown':m1['signed_valid'][1]=True
    elif failure=='signed_unadmitted':m1['signed_valid'][2]=True
    with pytest.raises(RuntimeError,match='GC_PROFILE_REFERENCE_M1_SCHEMA'):
        audit.reference_profile_grid(m1,'M5')


@pytest.mark.parametrize('failure,error', [
    ('clock','LADDER_M1_MAPPING'),('contract','LADDER_M1_MAPPING'),
    ('signed','LADDER_M1_MAPPING'),('dtype','LADDER_SCHEMA'),
    ('duplicate','LADDER_DUPLICATE_LEVEL'),('revisit','LADDER_MINUTE_REVISIT'),
    ('quantity','LADDER_SIDE_CONSERVATION'),('source_volume','LADDER_SIDE_CONSERVATION'),
    ('coverage','LADDER_MINUTE_COVERAGE'),
])
def test_profile_numeric_ladder_mapping_duplicates_conservation_and_coverage(failure,error):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    m1,ladder=_profile_numeric_fixture('M5');batches=[ladder]
    if failure=='clock':ladder['bar_start_ns'][0]+=1
    elif failure=='contract':ladder['contract_index'][0]=1
    elif failure=='signed':ladder['signed_valid'][0]=False
    elif failure=='dtype':ladder['buy']=ladder['buy'].astype(np.float64)
    elif failure=='duplicate':batches=[{k:v[:1] for k,v in ladder.items()},ladder]
    elif failure=='revisit':batches=[ladder,{k:v[:1] for k,v in ladder.items()}]
    elif failure=='quantity':ladder['buy'][0]+=np.uint64(1)
    elif failure=='source_volume':m1['volume'][0]+=np.uint64(1)
    elif failure=='coverage':
        # Even when a mixed-contract TF cannot be published, each admitted M1
        # must still be represented: an all-NaN output cannot hide missing input.
        m1['contract_index'][2]=1
        batches=[{k:v[:-1] for k,v in ladder.items()}]
    with pytest.raises(RuntimeError,match='GC_PROFILE_REFERENCE_'+error):
        audit.reference_profile_ladder(m1,batches,100000000,10**9)


def test_profile_numeric_physical_stream_skips_invalid_contract_without_resetting_session():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    m1,ladder=_profile_numeric_fixture('M15',partial=False)
    m1['volume_valid'][5:10]=False;m1['signed_valid'][5:10]=False;m1['contract_index'][5:10]=1
    take=(ladder['bar_start_ns']<m1['bar_start_ns'][5])|(ladder['bar_start_ns']>=m1['bar_start_ns'][10])
    ladder={name:values[take] for name,values in ladder.items()}
    refs,bounds,_=audit.reference_profile_ladder(m1,[ladder],100000000,10**9)
    observed=refs['M5'];strict=refs['strict_m5']
    for prefix,out in (('observed_profile_',observed),('',strict)):
        assert out[prefix+'session_observed_volume_contracts'][0]==19
        assert not out[prefix+'session_observed_valid'][1]
        assert np.isnan(out[prefix+'session_observed_volume_contracts'][1])
        assert out[prefix+'session_accepted_minutes'][1]==0
        assert out[prefix+'session_expected_minutes'][1]==10
        assert out[prefix+'anchored_segment_volume_contracts'][2]==5
        assert out[prefix+'session_observed_volume_contracts'][2]==24
        assert out[prefix+'session_accepted_minutes'][2]==10
        assert out[prefix+'session_expected_minutes'][2]==15
        assert not out[prefix+'rolling15_valid'][2]
        assert np.isnan(out[prefix+'anchored_segment_poc_prior_level_response_ticks'][2])
    # No finite observations is not numerical qualification, despite exact masks.
    key='observed_profile_rolling60_vwap_usd'
    columns=('bar_start_ns','available_ns','contract_index',key)
    expected={name:observed[name] for name in columns}
    report=audit.compare_trade_reference(expected,{name:value.copy() for name,value in expected.items()},
        {name:bounds['M5'][name] for name in columns},{name:audit.profile_reference_units()[name] for name in columns})
    assert not report[key]['numeric_reference_qualified'] and report[key]['finite_published_values_compared']==0


def test_profile_numeric_real_nine_and_seven_column_readers_without_availability(monkeypatch):
    import pyarrow.parquet as pq
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    m1,ladder=_profile_numeric_fixture('M5');readers={}
    for name,values in (('m1',m1),('ladder',ladder)):
        stream=pa.BufferOutputStream();pq.write_table(pa.table(values),stream)
        readers[name]=pq.ParquetFile(pa.BufferReader(stream.getvalue()))
    monkeypatch.setattr(audit.pq,'ParquetFile',lambda path:readers[str(path)])
    start=int(m1['bar_start_ns'][0]);end=int(m1['bar_start_ns'][-1])+MINUTE;seal=end+DAY
    source,groups=audit.read_profile_numeric_inputs({'m1_profile':{'path':'m1'}},start,end,seal)
    batches,ladder_groups=audit.iter_profile_numeric_ladder('ladder',start,end,seal)
    actual=list(batches)
    assert groups==ladder_groups==[0] and set(source)==set(audit.GC_PROFILE_M1_COLUMNS) and len(source)==9
    assert len(actual)==1 and set(actual[0])==set(audit.GC_PROFILE_LADDER_COLUMNS) and len(actual[0])==7
    assert 'available_ns' not in source and 'available_ns' not in actual[0]
    for name,value in m1.items():np.testing.assert_array_equal(source[name],value)
    for name,value in ladder.items():np.testing.assert_array_equal(actual[0][name],value)
    assert source['volume_valid'].dtype.kind==actual[0]['signed_valid'].dtype.kind=='b'
    assert audit.reference_profile_ladder(source,actual,100000000,10**9)[2]['seen_minutes']==4


def test_profile_numeric_actual_registry_412_no_overlap_previous_779_and_auxiliary_scope():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    from gx1.features import gc_native_context_v1 as registry
    declared=audit.profile_numeric_active_fields()
    assert [len(fields) for fields in declared.values()]==[75,75,75,71,65,51]
    expected={name for columns in declared.values() for name in columns.values()};actual=set()
    for tf,columns in declared.items():
        tables={'M5_GC_FEATURES.parquet'} if tf=='strict_m5' else {'TRADE_FEATURES_'+tf+'.parquet','retained::TRADE_FEATURES_'+tf+'.parquet'}
        actual.update(field['name'] for field in registry.REGISTRY['fields'] if field['table'] in tables and field['column'] in columns)
    assert len(expected)==412 and actual==expected and expected<=set(registry.contract()['value_fields'])
    smc,book=audit.smc_numeric_active_fields()
    previous={name for scope in (audit.event_book_numeric_active_fields(),audit.trade_numeric_active_fields(),smc)
        for columns in scope.values() for name in columns.values()}|set(book.values())
    assert len(previous)==779 and not previous.intersection(expected)
    assert len(audit.profile_reference_units())==79 and len(audit.profile_dynamic_units())==51
    assert all('observed_profile_snapshot_available_ns' not in fields for fields in declared.values())


def test_profile_numeric_large_integer_lower_crossings_and_exact_quantity_bound():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    from gx1.features import gc_order_flow_v1 as owner
    a=2**51
    hist={100:(a-1,0,0),101:(1,0,0),102:(a-1,0,0),103:(1,0,0),104:(a-1,0,0),105:(a,0,0)}
    values,bounds=audit.reference_profile_histogram(hist,100000000,10**9)
    assert values['volume_contracts']==2**53-1
    for name,expected in (('q25_usd',10.1),('median_usd',10.3),('q75_usd',10.5)):
        assert values[name]==pytest.approx(expected,rel=0,abs=1e-13)
    actual=owner.histogram_features(list(hist),np.array([q[0] for q in hist.values()],np.uint64),
        np.zeros(6,np.uint64),np.zeros(6,np.uint64),100000000)
    for name in ('q25_usd','median_usd','q75_usd'):
        assert abs(values[name]-actual[name])<=bounds[name],name
    with pytest.raises(RuntimeError,match='GC_PROFILE_REFERENCE_FLOAT_QUANTITY_BOUND'):
        audit.reference_profile_histogram({100:(2**53+1,0,0)},100000000,10**9)
