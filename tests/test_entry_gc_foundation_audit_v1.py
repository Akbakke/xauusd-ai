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
