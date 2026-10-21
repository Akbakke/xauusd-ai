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


def _basis_numeric_fixture(timeframe):
    n=_CLOCK_MINUTES[timeframe];start=pd.Timestamp('2024-05-01T22:00Z').value
    clock=start+np.arange(3*n,dtype=np.int64)*MINUTE
    admitted=np.ones(3*n,bool);admitted[2*n:]=False;admitted[[2*n,2*n+2]]=True
    close=np.full(3*n,1000,np.int64);close[n-1]=1020;close[2*n:]=0
    close[2*n]=1050;close[2*n+2]=1080
    m1={'bar_start_ns':clock,'contract_index':np.zeros(3*n,np.int16),
        'volume_valid':admitted,'close_tick':close}
    mid=np.full(3*n,100.);mid[2*n]=90.;mid[2*n+1]=80.;mid[2*n+2]=120.;mid[-1]=200.
    quotes={'time':clock.copy(),'bid_close':mid-1,'ask_close':mid+1}
    return m1,quotes


@pytest.mark.parametrize('timeframe',tuple(_CLOCK_MINUTES))
def test_basis_numeric_literal_positive_zero_negative_same_m1_all_five_tf(monkeypatch,tmp_path,timeframe):
    import pyarrow.parquet as pq
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit,gc_order_flow_features_v1 as owner
    from gx1.features import gc_order_flow_v1 as formulas
    m1,quotes=_basis_numeric_fixture(timeframe);n=_CLOCK_MINUTES[timeframe]
    with monkeypatch.context() as no_delegate:
        for module,name in ((owner,'add_basis'),(formulas,'timeframe_skeleton')):
            no_delegate.setattr(module,name,lambda *a,**k:(_ for _ in ()).throw(AssertionError('basis oracle delegates')))
        spot,report=audit.reference_basis_spot(m1['bar_start_ns'],[quotes],pd.Timestamp('2026-07-01T00:00Z').value)
        observed,strict,bounds,units=audit.reference_basis_aggregates(m1,spot,timeframe,100000000,10**9)
    assert len(observed)==7 and len(strict)==6
    np.testing.assert_array_equal(observed['observed_spot_basis_valid'],[True,True,True])
    np.testing.assert_allclose(observed['observed_spot_basis_bps'],[200.,0.,-1000.],rtol=0,atol=1e-10)
    np.testing.assert_allclose(strict['spot_basis_bps'],[200.,0.,np.nan],rtol=0,atol=1e-10,equal_nan=True)
    np.testing.assert_array_equal(strict['volume_valid'],[True,True,False])
    np.testing.assert_array_equal(strict['close_usd'],[102.,100.,np.nan])
    source_clock=m1['bar_start_ns'][[n-1,2*n-1,2*n+2]]+MINUTE
    np.testing.assert_array_equal(observed['observed_spot_basis_source_available_ns'],source_clock)
    np.testing.assert_array_equal(observed['observed_spot_basis_age_minutes'],[0.,0.,n-3.])
    # The unchanged producer is checked separately against hand-built surfaces.
    surface={'bar_start_ns':m1['bar_start_ns'][[0,n,2*n]].copy(),
        'available_ns':m1['bar_start_ns'][[0,n,2*n]]+n*MINUTE,
        'contract_index':np.zeros(3,np.int16),'volume_valid':np.array([True,True,False]),
        'close_usd':np.array([102.,100.,np.nan]),'observed_valid':np.ones(3,bool),
        'observed_close_usd':np.array([102.,100.,108.]),
        'observed_last_minute_ns':m1['bar_start_ns'][[n-1,2*n-1,2*n+2]].astype(np.float64)}
    part=tmp_path/'spot.parquet';manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    pq.write_table(pa.table({'time':pa.array(quotes['time'],type=pa.timestamp('ns',tz='UTC')),
        'bid_close':quotes['bid_close'],'ask_close':quotes['ask_close']}),part)
    plan={'inputs':{'oanda_manifest':{'path':str(manifest),'sha256':owner.previous.sha256_file(manifest),
        'partitions':[{'path':str(part),'sha256':owner.previous.sha256_file(part)}]}}}
    monkeypatch.setattr(owner.timing,'STUDY_START_NS',int(m1['bar_start_ns'][0]))
    owner.add_basis(plan,{timeframe:surface},m1['bar_start_ns'],include_observed=True)
    for key,reference in (('observed',observed),('strict',strict)):
        compared=audit.compare_trade_reference(reference,{name:surface[name] for name in reference},bounds[key],units[key])
        assert all(item['numeric_reference_qualified'] for item in compared.values())


@pytest.mark.parametrize('timeframe',tuple(_CLOCK_MINUTES))
def test_basis_numeric_missing_exact_spot_never_falls_back_to_old_or_tf_end(timeframe):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    m1,quotes=_basis_numeric_fixture(timeframe);n=_CLOCK_MINUTES[timeframe]
    take=np.ones(3*n,bool);take[[n-1,2*n+2]]=False
    quotes={name:value[take] for name,value in quotes.items()}
    spot,_=audit.reference_basis_spot(m1['bar_start_ns'],[quotes],pd.Timestamp('2026-07-01T00:00Z').value)
    observed,strict,_,_=audit.reference_basis_aggregates(m1,spot,timeframe,100000000,10**9)
    np.testing.assert_array_equal(strict['volume_valid'],[True,True,False])
    np.testing.assert_array_equal(observed['observed_spot_basis_valid'],[False,True,False])
    assert np.isnan(strict['spot_basis_bps'][[0,2]]).all() and strict['spot_basis_bps'][1]==0
    assert observed['observed_spot_basis_bps'][1]==0
    for name in ('observed_spot_basis_bps','observed_spot_basis_source_available_ns','observed_spot_basis_age_minutes'):
        assert np.isnan(observed[name][[0,2]]).all(),name


@pytest.mark.parametrize('timeframe',tuple(_CLOCK_MINUTES))
def test_basis_numeric_roll_including_unadmitted_minute_censors_whole_tf(timeframe):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    m1,quotes=_basis_numeric_fixture(timeframe);n=_CLOCK_MINUTES[timeframe]
    m1['contract_index'][n//2]=1;m1['contract_index'][2*n+1]=1
    spot,_=audit.reference_basis_spot(m1['bar_start_ns'],[quotes],pd.Timestamp('2026-07-01T00:00Z').value)
    observed,strict,_,_=audit.reference_basis_aggregates(m1,spot,timeframe,100000000,10**9)
    np.testing.assert_array_equal(strict['volume_valid'],[False,True,False])
    np.testing.assert_array_equal(observed['observed_spot_basis_valid'],[False,True,False])
    assert observed['observed_spot_basis_bps'][1]==strict['spot_basis_bps'][1]==0
    for name in ('observed_spot_basis_bps','observed_spot_basis_source_available_ns','observed_spot_basis_age_minutes'):
        assert np.isnan(observed[name][[0,2]]).all(),name
    assert np.isnan(strict['close_usd'][[0,2]]).all()


@pytest.mark.parametrize('failure',('missing','float_clock','float_contract','mask_dtype','float_close','invalid_close','gap'))
def test_basis_numeric_m1_schema_rejects_missing_clock_dtype_price_and_gap(failure):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    m1,quotes=_basis_numeric_fixture('M5')
    spot,_=audit.reference_basis_spot(m1['bar_start_ns'],[quotes],pd.Timestamp('2026-07-01T00:00Z').value)
    if failure=='missing':del m1['close_tick']
    elif failure=='float_clock':m1['bar_start_ns']=m1['bar_start_ns'].astype(np.float64)
    elif failure=='float_contract':m1['contract_index']=m1['contract_index'].astype(np.float64)
    elif failure=='mask_dtype':m1['volume_valid']=m1['volume_valid'].astype(np.int8)
    elif failure=='float_close':m1['close_tick']=m1['close_tick'].astype(np.float64)
    elif failure=='invalid_close':m1['close_tick'][0]=0
    elif failure=='gap':m1['bar_start_ns'][2]+=MINUTE
    with pytest.raises(RuntimeError,match='GC_BASIS_REFERENCE_M1_SCHEMA'):
        audit.reference_basis_aggregates(m1,spot,'M5',100000000,10**9)


@pytest.mark.parametrize('failure,error', [
    ('missing','SPOT_SCHEMA'),('float_clock','SPOT_SCHEMA'),('quote_dtype','SPOT_SCHEMA'),
    ('zero_bid','SPOT_QUOTES'),('crossed','SPOT_QUOTES'),('nan_bid','SPOT_QUOTES'),
    ('inf_ask','SPOT_QUOTES'),('offgrid','SPOT_TIME_GRID'),
])
def test_basis_numeric_spot_schema_quote_and_minute_grid_rejects(failure,error):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    m1,quotes=_basis_numeric_fixture('M5')
    if failure=='missing':del quotes['ask_close']
    elif failure=='float_clock':quotes['time']=quotes['time'].astype(np.float64)
    elif failure=='quote_dtype':quotes['bid_close']=quotes['bid_close'].astype(np.float32)
    elif failure=='zero_bid':quotes['bid_close'][0]=0
    elif failure=='crossed':quotes['ask_close'][0]=quotes['bid_close'][0]-1
    elif failure=='nan_bid':quotes['bid_close'][0]=np.nan
    elif failure=='inf_ask':quotes['ask_close'][0]=np.inf
    elif failure=='offgrid':quotes['time'][0]+=1
    with pytest.raises(RuntimeError,match='GC_BASIS_REFERENCE_'+error):
        audit.reference_basis_spot(m1['bar_start_ns'],[quotes],pd.Timestamp('2026-07-01T00:00Z').value)


@pytest.mark.parametrize('across_batches',(False,True))
def test_basis_numeric_duplicate_spot_minute_is_rejected_within_and_across_batches(across_batches):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    m1,quotes=_basis_numeric_fixture('M5')
    duplicate={name:value[:1] for name,value in quotes.items()}
    batches=[quotes,duplicate] if across_batches else [{name:np.concatenate((value,value[:1])) for name,value in quotes.items()}]
    with pytest.raises(RuntimeError,match='GC_BASIS_REFERENCE_SPOT_DUPLICATE'):
        audit.reference_basis_spot(m1['bar_start_ns'],batches,pd.Timestamp('2026-07-01T00:00Z').value)


def test_basis_numeric_last_source_minute_is_excluded_without_backfill_or_test_access():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    seal=pd.Timestamp('2026-07-01T00:00Z').value;clock=seal-np.arange(5,0,-1,dtype=np.int64)*MINUTE
    m1={'bar_start_ns':clock,'contract_index':np.zeros(5,np.int16),
        'volume_valid':np.array([True,True,True,True,False]),'close_tick':np.full(5,1000,np.int64)}
    quotes={'time':clock.copy(),'bid_close':np.full(5,99.),'ask_close':np.full(5,101.)}
    quotes['bid_close'][-1]=199.;quotes['ask_close'][-1]=201.
    spot,report=audit.reference_basis_spot(clock,[quotes],seal)
    assert np.isnan(spot['bid'][-1]) and np.isnan(spot['ask'][-1])
    assert report['decoded_spot_rows']==5 and report['matched_spot_minutes']==4
    assert report['missing_spot_minutes']==report['terminal_m1_rows_excluded']==1
    assert report['spot_passes']==1 and report['older_spot_fallbacks']==0
    observed,strict,_,_=audit.reference_basis_aggregates(m1,spot,'M5',100000000,10**9)
    assert observed['observed_spot_basis_valid'].tolist()==[True]
    assert observed['observed_spot_basis_bps'].tolist()==[0.]
    assert observed['observed_spot_basis_source_available_ns'].tolist()==[seal-MINUTE]
    assert observed['observed_spot_basis_age_minutes'].tolist()==[1.]
    assert not strict['volume_valid'][0] and np.isnan(strict['spot_basis_bps'][0])
    forbidden={name:value[:1].copy() for name,value in quotes.items()};forbidden['time'][0]=seal
    with pytest.raises(RuntimeError,match='GC_BASIS_REFERENCE_SPOT_TEST_CLOCK'):
        audit.reference_basis_spot(clock,[forbidden],seal)


def test_basis_numeric_real_four_column_gc_and_three_column_timestamp_spot_readers(monkeypatch):
    import pyarrow.parquet as pq
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    m1,quotes=_basis_numeric_fixture('M5');readers={}
    # The physical spot schema can contain other fields; only the declared three
    # columns are decoded. The physical clock must be timestamp[ns,UTC].
    spot_table=pa.table({'time':pa.array(quotes['time'],type=pa.timestamp('ns',tz='UTC')),
        'bid_close':quotes['bid_close'],'ask_close':quotes['ask_close'],'unused':np.ones(len(quotes['time']))})
    for name,table in (('m1',pa.table(m1)),('spot',spot_table)):
        stream=pa.BufferOutputStream();pq.write_table(table,stream)
        readers[name]=pq.ParquetFile(pa.BufferReader(stream.getvalue()))
    monkeypatch.setattr(audit.pq,'ParquetFile',lambda path:readers[str(path)])
    start=int(m1['bar_start_ns'][0]);end=int(m1['bar_start_ns'][-1])+MINUTE;seal=end+DAY
    source,groups=audit.read_basis_numeric_inputs({'m1_basis':{'path':'m1'}},start,end,seal)
    batches,spot_groups=audit.iter_basis_spot('spot',start,end,seal);decoded=list(batches)
    assert groups==spot_groups==[0] and len(source)==4 and set(source)==set(audit.GC_BASIS_M1_COLUMNS)
    assert len(decoded)==1 and set(decoded[0])==set(audit.GC_BASIS_SPOT_COLUMNS)
    assert decoded[0]['time'].dtype==np.int64 and source['volume_valid'].dtype.kind=='b'
    assert 'available_ns' not in source and 'unused' not in decoded[0]
    for name,value in m1.items():np.testing.assert_array_equal(source[name],value)
    for name,value in quotes.items():np.testing.assert_array_equal(decoded[0][name],value)
    _,report=audit.reference_basis_spot(source['bar_start_ns'],decoded,seal)
    assert report['matched_spot_minutes']==report['decoded_spot_rows']==len(quotes['time'])
    assert report['spot_passes']==1 and report['older_spot_fallbacks']==0


def test_basis_numeric_spot_footer_rejects_test_before_decoding_earlier_safe_group(monkeypatch):
    import pyarrow.parquet as pq
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    seal=pd.Timestamp('2026-07-01T00:00Z').value
    table=pa.table({'time':pa.array([seal-2*MINUTE,seal],type=pa.timestamp('ns',tz='UTC')),
        'bid_close':[99.,99.],'ask_close':[101.,101.]})
    stream=pa.BufferOutputStream();pq.write_table(table,stream,row_group_size=1)
    reader=pq.ParquetFile(pa.BufferReader(stream.getvalue()))
    class NoDecode:
        metadata=reader.metadata
        schema_arrow=reader.schema_arrow
        def iter_batches(self,**kwargs):raise AssertionError('decoded payload before all footers admitted')
    monkeypatch.setattr(audit.pq,'ParquetFile',lambda path:NoDecode())
    with pytest.raises(RuntimeError,match='GC_EVENT_NUMERIC_ROW_GROUP_INTERSECTS_TEST'):
        audit.iter_basis_spot('memory',seal-2*MINUTE,seal+MINUTE,seal)


def test_basis_numeric_actual_registry_14_completes_1205_and_d1_strict_stays_auxiliary():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    from gx1.features import gc_native_context_v1 as registry
    declared=audit.basis_numeric_active_fields()
    assert [len(fields) for fields in declared.values()]==[2,2,2,2,2,1,1,1,1]
    expected={name for columns in declared.values() for name in columns.values()};actual=set()
    for scope,columns in declared.items():
        table='BASIS_'+scope.removeprefix('strict_')+'.parquet' if scope.startswith('strict_') else 'TRADE_FEATURES_'+scope+'.parquet'
        actual.update(field['name'] for field in registry.REGISTRY['fields'] if field['table']==table and field['column'] in columns)
    assert len(expected)==14 and actual==expected and 'strict_D1' not in declared
    smc,book=audit.smc_numeric_active_fields()
    previous={name for scope in (audit.event_book_numeric_active_fields(),audit.trade_numeric_active_fields(),smc,audit.profile_numeric_active_fields())
        for columns in scope.values() for name in columns.values()}|set(book.values())
    assert len(previous)==1191 and not previous.intersection(expected)
    assert previous|expected==set(registry.contract()['value_fields'])
    # Physical strict D1 is still compared, but an all-NaN value is no proof.
    m1,quotes=_basis_numeric_fixture('D1');m1['volume_valid'][:]=False
    spot,_=audit.reference_basis_spot(m1['bar_start_ns'],[quotes],pd.Timestamp('2026-07-01T00:00Z').value)
    _,strict,bounds,units=audit.reference_basis_aggregates(m1,spot,'D1',100000000,10**9)
    assert len(strict)==6 and np.isnan(strict['spot_basis_bps']).all()
    report=audit.compare_trade_reference(strict,{name:value.copy() for name,value in strict.items()},bounds['strict'],units['strict'])
    assert not report['spot_basis_bps']['numeric_reference_qualified']
    assert report['spot_basis_bps']['finite_published_values_compared']==0


@pytest.mark.parametrize('failure',('naive','milliseconds','timezone'))
def test_basis_numeric_spot_requires_ns_utc_timestamp_before_any_payload(monkeypatch,failure):
    import pyarrow.parquet as pq
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    stamp=pd.Timestamp('2024-05-01T22:00Z')
    kind={'naive':pa.timestamp('ns'),'milliseconds':pa.timestamp('ms',tz='UTC'),
        'timezone':pa.timestamp('ns',tz='Europe/Oslo')}[failure]
    value=stamp.tz_localize(None) if failure=='naive' else stamp
    table=pa.table({'time':pa.array([value],type=kind),'bid_close':[99.],'ask_close':[101.]})
    stream=pa.BufferOutputStream();pq.write_table(table,stream)
    reader=pq.ParquetFile(pa.BufferReader(stream.getvalue()))
    class NoDecode:
        metadata=reader.metadata
        schema_arrow=reader.schema_arrow
        def iter_batches(self,**kwargs):raise AssertionError('decoded non-ns-UTC spot payload')
    monkeypatch.setattr(audit.pq,'ParquetFile',lambda path:NoDecode())
    with pytest.raises(RuntimeError,match='GC_BASIS_REFERENCE_SPOT_SCHEMA'):
        audit.iter_basis_spot('memory',stamp.value-MINUTE,stamp.value+MINUTE,stamp.value+DAY)


@pytest.mark.parametrize('domain',('overflow_mid','subnormal_mid'))
def test_basis_numeric_analytic_bound_rejects_unsupported_float_arithmetic(domain):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    m1,quotes=_basis_numeric_fixture('M5')
    value=np.finfo(np.float64).max if domain=='overflow_mid' else float.fromhex('0x0.0000000000001p-1022')
    quotes['bid_close'][:]=value;quotes['ask_close'][:]=value
    spot,_=audit.reference_basis_spot(m1['bar_start_ns'],[quotes],pd.Timestamp('2026-07-01T00:00Z').value)
    with pytest.raises(RuntimeError,match='GC_BASIS_REFERENCE_UNSUPPORTED_FLOAT_ARITHMETIC'):
        audit.reference_basis_aggregates(m1,spot,'M5',100000000,10**9)


# 126: independent published-TF -> consumed M5 context projection.
def _projection_field(name="x", *, table="T", column=None, validity="own_valid",
                      source_clock="source_ns", role=None, transform=None):
    value = {"name": name, "table": table, "column": name if column is None else column,
             "availability_field": name + "_observed", "owner": "momentum_flow_encoder"}
    if validity is not None:
        value["validity_column"] = validity
    if source_clock is not None:
        value["source_clock_column"] = source_clock
    if role is not None:
        value["role"] = role
    if transform is not None:
        value["transform"] = transform
    return value


def test_native_projection_barstart_plus_300s_observes_exact_closed_boundary():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    base = pd.Timestamp("2024-05-01T00:00:00Z").value
    barstarts = base + np.array([0, 5, 10], np.int64) * MINUTE
    tables = {"T": {
        "available_ns": base + np.array([5, 10, 15], np.int64) * MINUTE,
        "x": np.array([10., 20., 30.]), "own_valid": np.ones(3),
        "source_ns": base + np.array([5, 10, 15], np.int64) * MINUTE,
    }}
    decisions = audit.gc_native_decision_stamps(barstarts)
    np.testing.assert_array_equal(decisions, base + np.array([5, 10, 15], np.int64) * MINUTE)
    actual, _ = audit.reference_gc_native_projection([_projection_field()], tables, decisions)
    np.testing.assert_array_equal(actual, np.array([[10, 1], [20, 1], [30, 1]], np.float32))
    wrong_clock, _ = audit.reference_gc_native_projection([_projection_field()], tables, barstarts)
    np.testing.assert_array_equal(wrong_clock, np.array([[0, 0], [10, 1], [20, 1]], np.float32))


def test_native_projection_literal_inclusive_latest_invalid_and_real_zero(monkeypatch):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    from gx1.features import gc_native_context_v1 as producer
    def forbidden(*args, **kwargs):
        raise AssertionError("independent reference delegated to production context")
    monkeypatch.setattr(producer, "build_context", forbidden)
    monkeypatch.setattr(producer, "_field_values", forbidden)
    clock = np.array([5, 10, 15], np.int64) * MINUTE
    table = {"available_ns": clock, "source_ns": clock.copy(),
             "x": np.array([10., 99., 0.]), "own_valid": np.array([1., 0., 1.])}
    decisions = np.array([5*MINUTE-1, 5*MINUTE, 7*MINUTE, 10*MINUTE, 12*MINUTE, 15*MINUTE], np.int64)
    actual, report = audit.reference_gc_native_projection([_projection_field()], {"T": table}, decisions)
    expected = np.array([[0, 0], [10, 1], [10, 1], [0, 0], [0, 0], [0, 1]], np.float32)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    assert report["x"]["observed"] == 3 and report["x"]["missing"] == 3
    assert report["x"]["observed_zero"] == 1 and report["x"]["nonzero"] == 2


@pytest.mark.parametrize("source_ns", [6*MINUTE, 7*MINUTE+1])
def test_native_projection_future_source_rejected_even_with_invalid_latest_field(source_ns):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    table = {"available_ns": np.array([5*MINUTE], np.int64),
             "source_ns": np.array([source_ns], np.int64),
             "x": np.array([99.]), "own_valid": np.array([0.])}
    with pytest.raises(RuntimeError, match="GC_NATIVE_PROJECTION_FUTURE_SOURCE_CLOCK"):
        audit.reference_gc_native_projection([_projection_field()], {"T": table},
                                            np.array([7*MINUTE], np.int64))


def test_native_projection_future_poison_is_not_read_as_current_information():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    table = {"available_ns": np.array([5, 10], np.int64)*MINUTE,
             "source_ns": np.array([5, 11], np.int64)*MINUTE,
             "x": np.array([10., np.nan]), "own_valid": np.array([1., 0.])}
    earlier, _ = audit.reference_gc_native_projection([_projection_field()], {"T": table},
                                                    np.array([7*MINUTE], np.int64))
    np.testing.assert_array_equal(earlier, np.array([[10, 1]], np.float32))
    with pytest.raises(RuntimeError, match="GC_NATIVE_PROJECTION_FUTURE_SOURCE_CLOCK"):
        audit.reference_gc_native_projection([_projection_field()], {"T": table},
                                            np.array([10*MINUTE], np.int64))


@pytest.mark.parametrize("missing", ["value", "validity", "source_clock"])
def test_native_projection_nan_components_make_missing_without_backsearch(missing):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    table = {"available_ns": np.array([5, 10], np.int64)*MINUTE,
             "source_ns": np.array([5*MINUTE, 10*MINUTE], np.float64),
             "x": np.array([7., 8.]), "own_valid": np.ones(2)}
    key = {"value": "x", "validity": "own_valid", "source_clock": "source_ns"}[missing]
    table[key][1] = np.nan
    actual, _ = audit.reference_gc_native_projection([_projection_field()], {"T": table},
                                                   np.array([5, 10, 12], np.int64)*MINUTE)
    np.testing.assert_array_equal(actual, np.array([[7, 1], [0, 0], [0, 0]], np.float32))


def test_native_projection_structural_false_and_separate_own_validity():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    clock = np.array([5, 10, 15], np.int64)*MINUTE
    table = {"available_ns": clock, "source_ns": clock.copy(),
             "flag": np.array([0., 1., 0.]), "own_valid": np.array([1., 0., 1.])}
    field = _projection_field("flag", role="structural_observation_mask")
    actual, _ = audit.reference_gc_native_projection([field], {"T": table}, clock)
    np.testing.assert_array_equal(actual, np.array([[0, 1], [0, 0], [0, 1]], np.float32))
    table["flag"][0] = 2
    with pytest.raises(RuntimeError, match="GC_NATIVE_PROJECTION_STRUCTURAL_VALUE_DOMAIN"):
        audit.reference_gc_native_projection([field], {"T": table}, clock)


@pytest.mark.parametrize("fault,marker", [
    ("validity", "VALIDITY_DOMAIN"), ("fractional_clock", "SOURCE_CLOCK_PRECISION"),
])
def test_native_projection_rejects_invalid_selected_domain_without_value_repair(fault, marker):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    table = {"available_ns": np.array([5*MINUTE], np.int64),
             "source_ns": np.array([0.]), "x": np.array([3.]), "own_valid": np.array([1.])}
    if fault == "validity":
        table["own_valid"][0] = 2
    else:
        table["source_ns"][0] = .5
    with pytest.raises(RuntimeError, match="GC_NATIVE_PROJECTION_" + marker):
        audit.reference_gc_native_projection([_projection_field()], {"T": table},
                                            np.array([7*MINUTE], np.int64))


@pytest.mark.parametrize("column", ["observed_profile_snapshot_age_minutes",
                                   "observed_spot_basis_age_minutes"])
@pytest.mark.parametrize("explicit_transform", [False, True])
def test_native_projection_age_grows_from_own_source_clock_not_stored_snapshot(column, explicit_transform):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    field = _projection_field("age", column=column,
        transform="decision_age_minutes" if explicit_transform else None)
    table = {"available_ns": np.array([5, 10], np.int64)*MINUTE,
             "source_ns": np.array([3, 8], np.int64)*MINUTE,
             column: np.array([123., 456.]), "own_valid": np.ones(2)}
    actual, report = audit.reference_gc_native_projection([field], {"T": table},
                                                        np.array([5, 7, 10, 12], np.int64)*MINUTE)
    np.testing.assert_array_equal(actual, np.array([[2, 1], [4, 1], [2, 1], [4, 1]], np.float32))
    assert report["age"]["age_transformed"] is True


def test_native_projection_float32_cast_happens_after_own_mask_and_preserves_observed_underflow():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    clock = np.array([5, 10, 15], np.int64)*MINUTE
    table = {"available_ns": clock, "source_ns": clock.copy(),
             "x": np.array([1e300, -1e-300, 0.]), "own_valid": np.array([0., 1., 1.])}
    actual, _ = audit.reference_gc_native_projection([_projection_field()], {"T": table}, clock)
    expected = np.array([[0., 0.], [-0., 1.], [0., 1.]], np.float32)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    table["own_valid"][0] = 1
    with pytest.raises(RuntimeError, match="GC_NATIVE_PROJECTION_FLOAT32_OVERFLOW"):
        audit.reference_gc_native_projection([_projection_field()], {"T": table}, clock)


def test_native_projection_multiple_tables_aliases_and_opaque_batch_cursor_have_literal_lanes():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    a = {"available_ns": np.array([5, 10], np.int64)*MINUTE,
         "source_ns": np.array([5, 10], np.int64)*MINUTE,
         "a": np.array([1., 2.]), "copy": np.array([7., 9.]), "own_valid": np.ones(2)}
    b = {"available_ns": np.array([6, 12], np.int64)*MINUTE,
         "source_ns": np.array([6, 12], np.int64)*MINUTE,
         "b": np.array([4., 8.]), "own_valid": np.ones(2)}
    fields = [_projection_field("b", table="B"), _projection_field("a", table="A"),
              _projection_field("alias", table="A_ALIAS", column="copy")]
    tables = {"A": a, "A_ALIAS": a, "B": b}
    decisions = np.array([5, 6, 10, 12], np.int64)*MINUTE
    expected = np.array([[0, 0, 1, 1, 7, 1], [4, 1, 1, 1, 7, 1],
                         [4, 1, 2, 1, 9, 1], [8, 1, 2, 1, 9, 1]], np.float32)
    whole, _ = audit.reference_gc_native_projection(fields, tables, decisions)
    cursor = {}
    first, _ = audit.reference_gc_native_projection(fields, tables, decisions[:2], cursors=cursor)
    second, _ = audit.reference_gc_native_projection(fields, tables, decisions[2:], cursors=cursor)
    np.testing.assert_array_equal(whole, expected)
    np.testing.assert_array_equal(np.concatenate((first, second)), expected)
    for stale in (decisions[-1:], decisions[:1]):
        with pytest.raises(RuntimeError, match="GC_NATIVE_PROJECTION_"):
            audit.reference_gc_native_projection(fields, tables, stale, cursors=cursor)


def test_native_projection_layout_uses_registry_order_and_exact_context_signal_aliases():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    fields = [_projection_field("a"), _projection_field("b")]
    ctx = ["dense", "b", "b_observed", "a", "a_observed"]
    signal = ["dense_signal", "ctx_cont.b", "ctx_cont.b_observed", "ctx_cont.a", "ctx_cont.a_observed"]
    result = audit.native_projection_layout(fields, ctx, signal, signal.copy())
    assert list(result["names"]) == ["a", "a_observed", "b", "b_observed"]
    assert list(result["ctx_indices"]) == [3, 4, 1, 2]
    assert list(result["signal_indices"]) == [3, 4, 1, 2]


@pytest.mark.parametrize("fault", ["missing_mask", "duplicate_name", "wrong_alias", "surface_order"])
def test_native_projection_layout_rejects_missing_duplicate_or_split_brain_lanes(fault):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    fields = [_projection_field("a")]
    ctx, signal = ["a", "a_observed"], ["ctx_cont.a", "ctx_cont.a_observed"]
    surface = signal.copy()
    if fault == "missing_mask":
        ctx.pop()
    elif fault == "duplicate_name":
        ctx.append("a")
    elif fault == "wrong_alias":
        signal[0] = surface[0] = "ctx_cont.other"
    else:
        surface.reverse()
    with pytest.raises(RuntimeError, match="GC_NATIVE_PROJECTION_FIELD_LAYOUT"):
        audit.native_projection_layout(fields, ctx, signal, surface)


def test_native_projection_96_history_union_is_exact_for_both_physical_splits():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    base = pd.Timestamp("2024-05-01T00:00:00Z").value
    source = base + np.arange(240, dtype=np.int64)*5*MINUTE
    split = {"train": source[[95, 100]], "val": source[[220]]}
    masks, report = audit.reference_consumed_surface_rows(source, split, sequence_length=96)
    expected_train, expected_val = np.zeros(240, bool), np.zeros(240, bool)
    expected_train[:101], expected_val[125:221] = True, True
    np.testing.assert_array_equal(masks["train"], expected_train)
    np.testing.assert_array_equal(masks["val"], expected_val)
    np.testing.assert_array_equal(masks["union"], expected_train | expected_val)
    assert masks["union"].sum() == report["union_rows"] == 197
    assert not masks["union"][101:125].any()
    assert report["sequence_length"] == 96


@pytest.mark.parametrize("fault", ["missing_exact_ns", "insufficient_history", "source_duplicate",
                                 "split_duplicate", "source_reversed", "source_dtype"])
def test_native_projection_96_history_union_rejects_wrong_clocks_and_unavailable_history(fault):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    source = np.arange(240, dtype=np.int64)*5*MINUTE
    split = {"train": source[[95, 100]], "val": source[[220]]}
    if fault == "missing_exact_ns":
        split["train"] = split["train"].copy(); split["train"][1] += 1
    elif fault == "insufficient_history":
        split["train"] = source[[94]]
    elif fault == "source_duplicate":
        source[100] = source[99]
    elif fault == "split_duplicate":
        split["val"] = source[[220, 220]]
    elif fault == "source_reversed":
        source = source[::-1].copy()
    else:
        source = source.astype(np.float64)
    with pytest.raises(RuntimeError, match="GC_NATIVE_PROJECTION_ROW_UNION_"):
        audit.reference_consumed_surface_rows(source, split, sequence_length=96)



def _projection_footer_no_payload(monkeypatch, audit):
    original = audit.pq.ParquetFile
    def forbidden(*args, **kwargs):
        raise AssertionError("projection footer decoded payload before all clock/schema barriers")
    class FooterOnly:
        def __init__(self, wrapped):
            self.wrapped = wrapped
        def __getattr__(self, name):
            if name in ("read", "read_row_group", "read_row_groups", "iter_batches"):
                return forbidden
            return getattr(self.wrapped, name)
    monkeypatch.setattr(audit.pq, "ParquetFile",
                        lambda *args, **kwargs: FooterOnly(original(*args, **kwargs)))
    monkeypatch.setattr(audit.pq, "read_table", forbidden)


@pytest.mark.parametrize("kind", ["train", "val", "gc"])
def test_native_projection_footer_all_groups_are_checked_without_payload(tmp_path, monkeypatch, kind):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    import pyarrow.parquet as pq
    seal = pd.Timestamp("2026-07-01T00:00:00Z").value
    stamps = np.array([seal-15*MINUTE, seal-10*MINUTE], np.int64)
    name = "available_ns" if kind == "gc" else "time"
    values = pa.array(stamps) if kind == "gc" else pa.array(pd.to_datetime(stamps, utc=True))
    path = tmp_path / (kind + ".parquet")
    columns = {name: values}
    if kind == "gc": columns["bar_start_ns"] = pa.array(stamps-5*MINUTE)
    pq.write_table(pa.table(columns), path, row_group_size=1)
    _projection_footer_no_payload(monkeypatch, audit)
    result = audit.projection_numeric_footer(path, seal, kind=kind)
    assert result["groups"] == [0, 1]
    assert result["rows"] == 2 and result["row_group_rows"] == [1, 1]


@pytest.mark.parametrize("kind", ["train", "val", "gc"])
def test_native_projection_footer_rejects_later_sealed_group_before_decoding_safe_group(tmp_path, monkeypatch, kind):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    import pyarrow.parquet as pq
    seal = pd.Timestamp("2026-07-01T00:00:00Z").value
    # GC admission follows bar-start provenance; a bucket starting at TEST is forbidden.
    stamps = np.array([seal-15*MINUTE, seal + (5*MINUTE if kind == "gc" else 0)], np.int64)
    name = "available_ns" if kind == "gc" else "time"
    values = pa.array(stamps) if kind == "gc" else pa.array(pd.to_datetime(stamps, utc=True))
    path = tmp_path / (kind + "-sealed.parquet")
    columns = {name: values}
    if kind == "gc": columns["bar_start_ns"] = pa.array(stamps-5*MINUTE)
    pq.write_table(pa.table(columns), path, row_group_size=1)
    _projection_footer_no_payload(monkeypatch, audit)
    with pytest.raises(RuntimeError, match="GC_NATIVE_PROJECTION_TEST_ROW_GROUP"):
        audit.projection_numeric_footer(path, seal, kind=kind)


@pytest.mark.parametrize("fault", ["naive", "milliseconds", "integer_time", "float_gc_clock"])
def test_native_projection_footer_rejects_wrong_clock_schema_before_payload(tmp_path, monkeypatch, fault):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    import pyarrow.parquet as pq
    seal = pd.Timestamp("2026-07-01T00:00:00Z").value
    stamps = np.array([seal-15*MINUTE, seal-10*MINUTE], np.int64)
    kind, name = ("gc", "available_ns") if fault == "float_gc_clock" else ("train", "time")
    if fault == "naive":
        values = pa.array(pd.to_datetime(stamps))
    elif fault == "milliseconds":
        values = pa.array(pd.to_datetime(stamps, utc=True).as_unit("ms"))
    elif fault == "integer_time":
        values = pa.array(stamps)
    else:
        values = pa.array(stamps.astype(np.float64))
    path = tmp_path / (fault + ".parquet")
    columns = {name: values}
    if kind == "gc": columns["bar_start_ns"] = pa.array(stamps-5*MINUTE)
    pq.write_table(pa.table(columns), path, row_group_size=1)
    _projection_footer_no_payload(monkeypatch, audit)
    with pytest.raises(RuntimeError, match="GC_NATIVE_PROJECTION_SCHEMA"):
        audit.projection_numeric_footer(path, seal, kind=kind)


def test_native_projection_closed_decision_reaching_seal_is_rejected():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    seal = pd.Timestamp("2026-07-01T00:00:00Z").value
    safe = audit.gc_native_decision_stamps(np.array([seal-10*MINUTE], np.int64), seal=seal)
    np.testing.assert_array_equal(safe, np.array([seal-5*MINUTE], np.int64))
    with pytest.raises(RuntimeError, match="GC_NATIVE_PROJECTION_"):
        audit.gc_native_decision_stamps(np.array([seal-5*MINUTE], np.int64), seal=seal)



def test_native_projection_each_field_uses_its_own_validity_on_same_table():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    clock = np.array([5, 10, 15], np.int64)*MINUTE
    table = {"available_ns": clock, "source_ns": clock.copy(),
             "x": np.array([10., 20., 30.]), "valid_x": np.array([1., 0., 1.]),
             "y": np.array([100., 200., 300.]), "valid_y": np.array([0., 1., 1.])}
    fields = [_projection_field("x", validity="valid_x"), _projection_field("y", validity="valid_y")]
    actual, _ = audit.reference_gc_native_projection(fields, {"T": table}, clock)
    np.testing.assert_array_equal(actual, np.array([[10, 1, 0, 0], [0, 0, 200, 1],
                                                  [30, 1, 300, 1]], np.float32))



def test_native_projection_surface_footer_accepts_exact_f32_context_width_without_payload(tmp_path, monkeypatch):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    import pyarrow.parquet as pq
    seal = pd.Timestamp("2026-07-01T00:00:00Z").value
    stamps = np.array([seal-15*MINUTE, seal-10*MINUTE], np.int64)
    context = pa.FixedSizeListArray.from_arrays(pa.array(np.zeros(2*2542, np.float32)), 2542)
    path = tmp_path / "surface.parquet"
    pq.write_table(pa.table({"time": pa.array(pd.to_datetime(stamps, utc=True)),
                            "ctx_cont": context}), path, row_group_size=1)
    _projection_footer_no_payload(monkeypatch, audit)
    result = audit.projection_numeric_footer(path, seal, kind="surface")
    assert result["groups"] == [0, 1] and result["rows"] == 2


@pytest.mark.parametrize("fault", ["wrong_width", "float64_values", "variable_list", "missing_context"])
def test_native_projection_surface_footer_rejects_wrong_context_schema_before_payload(tmp_path, monkeypatch, fault):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    import pyarrow.parquet as pq
    seal = pd.Timestamp("2026-07-01T00:00:00Z").value
    columns = {"time": pa.array(pd.to_datetime([seal-10*MINUTE], utc=True))}
    if fault != "missing_context":
        width = 2541 if fault == "wrong_width" else 2542
        dtype = np.float64 if fault == "float64_values" else np.float32
        values = pa.array(np.zeros(width, dtype))
        columns["ctx_cont"] = (pa.ListArray.from_arrays(pa.array([0, width], pa.int32()), values)
                               if fault == "variable_list" else
                               pa.FixedSizeListArray.from_arrays(values, width))
    path = tmp_path / (fault + "-surface.parquet")
    pq.write_table(pa.table(columns), path)
    _projection_footer_no_payload(monkeypatch, audit)
    with pytest.raises(RuntimeError, match="GC_NATIVE_PROJECTION_SCHEMA"):
        audit.projection_numeric_footer(path, seal, kind="surface")


def test_native_projection_gc_footer_allows_bucket_closed_exactly_at_seal_without_payload(tmp_path, monkeypatch):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    import pyarrow.parquet as pq
    seal = pd.Timestamp("2026-07-01T00:00:00Z").value
    path = tmp_path / "closed-at-seal.parquet"
    pq.write_table(pa.table({"available_ns": pa.array([seal], pa.int64()),
                            "bar_start_ns": pa.array([seal-5*MINUTE], pa.int64())}), path)
    _projection_footer_no_payload(monkeypatch, audit)
    result = audit.projection_numeric_footer(path, seal, kind="gc")
    assert result["groups"] == [0] and result["rows"] == 1



def test_native_projection_plain_field_uses_bucket_clock_and_finite_value_gate():
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    clock = np.array([5, 10], np.int64)*MINUTE
    table = {"available_ns": clock, "x": np.array([0., np.nan])}
    field = _projection_field(validity=None, source_clock=None)
    actual, _ = audit.reference_gc_native_projection([field], {"T": table}, clock)
    np.testing.assert_array_equal(actual, np.array([[0, 1], [0, 0]], np.float32))



def test_native_projection_published_partial_day_tail_is_admitted_but_never_selected_before_seal(tmp_path, monkeypatch):
    from gx1.scripts import audit_entry_gc_foundation_v1 as audit
    import pyarrow.parquet as pq
    seal = pd.Timestamp("2026-07-01T00:00:00Z").value
    # A genuine D1 bucket starts June30 22Z and nominally closes July1 22Z.
    # It is pre-TEST publication provenance, with no usable observation yet.
    table = {
        "bar_start_ns": np.array([seal-26*60*MINUTE, seal-2*60*MINUTE], np.int64),
        "available_ns": np.array([seal-2*60*MINUTE, seal+22*60*MINUTE], np.int64),
        "source_ns": np.array([seal-2*60*MINUTE, seal+22*60*MINUTE], np.int64),
        "x": np.array([4., 999.]), "own_valid": np.array([1., 0.]),
    }
    path = tmp_path / "published-partial-D1.parquet"
    pq.write_table(pa.table(table), path, row_group_size=1)
    _projection_footer_no_payload(monkeypatch, audit)
    footer = audit.projection_numeric_footer(path, seal, kind="gc")
    assert footer["groups"] == [0, 1] and footer["rows"] == 2
    before_seal, _ = audit.reference_gc_native_projection(
        [_projection_field()], {"T": table}, np.array([seal-60*MINUTE], np.int64))
    np.testing.assert_array_equal(before_seal, np.array([[4., 1.]], np.float32))



class TestTechnicalReference127:
    """Small mathematical witnesses; no cache/data/model payload is needed."""

    @staticmethod
    def _owner():
        from gx1.scripts import audit_entry_gc_foundation_v1 as audit
        return audit

    @staticmethod
    def _frame(close, *, high=None, low=None, volume=None, stamps=None):
        close = np.asarray(close, dtype=np.float64)
        n = len(close)
        if stamps is None:
            origin = pd.Timestamp("2009-06-01T00:00:00Z").value
            stamps = origin + np.arange(n, dtype=np.int64)*5*MINUTE
        return {
            "bar_start_ns": np.asarray(stamps, dtype=np.int64),
            "open": close.copy(),
            "high": close + .5 if high is None else np.asarray(high, dtype=np.float64),
            "low": close - .5 if low is None else np.asarray(low, dtype=np.float64),
            "close": close.copy(),
            "volume": np.ones(n, dtype=np.int64) if volume is None else np.asarray(volume, dtype=np.int64),
        }

    @staticmethod
    def _encloses(values, radii, index, numerator, denominator=1):
        # Fractions compare the stated mathematical value with the returned
        # binary64 centre/radius without introducing an arbitrary test epsilon.
        from fractions import Fraction
        centre = Fraction.from_float(float(values[index]))
        radius = Fraction.from_float(float(radii[index]))
        target = Fraction(numerator, denominator)
        assert radius >= 0
        assert centre-radius <= target <= centre+radius

    def test_exact_twenty_three_field_contract_and_nonnegative_bounds(self):
        audit = self._owner()
        expected = (
            "atr_bps_14", "rsi14_centered", "mom_5_atr", "mom_20_atr", "close_open_atr",
            "ema20_dist_atr", "ema50_dist_atr", "ema100_dist_atr", "ema200_dist_atr",
            "ema20_slope_atr", "ema50_slope_atr", "ema100_slope_atr", "ema200_slope_atr",
            "ema_stack_aligned_v2", "vwap_rolling5_dist_atr", "vwap20_dist_atr",
            "vwap96_dist_atr", "vwap_rolling5_slope_atr", "bb_position", "bb_width_atr",
            "adx14", "di_spread_signed", "trend_state_age_bars",
        )
        assert tuple(audit.TECHNICAL_NUMERIC_FIELDS) == expected
        close = 100 + np.arange(240, dtype=np.float64)
        frame = self._frame(close, high=close+2, low=close-2)
        frame["open"] -= 1
        values, bounds, report = audit.reference_technical_block(frame)
        assert tuple(values) == expected and tuple(bounds) == expected
        assert report["source_rows"] == 240
        for name in expected:
            assert values[name].dtype == np.float64 and bounds[name].dtype == np.float64
            observed = np.isfinite(values[name])
            assert np.array_equal(np.isfinite(bounds[name]), observed)
            assert np.all(bounds[name][observed] >= 0)
        # On an arithmetic price progression, the classic EMA's seeded lag
        # is (span-1)/2 forever; all returns/distances below are in ATR=4.
        rational = {
            "atr_bps_14": (40000, 339), "rsi14_centered": (1, 1),
            "mom_5_atr": (5, 4), "mom_20_atr": (5, 1), "close_open_atr": (1, 4),
            "ema20_dist_atr": (19, 8), "ema50_dist_atr": (49, 8),
            "ema100_dist_atr": (99, 8), "ema200_dist_atr": (199, 8),
            "ema20_slope_atr": (5, 4), "ema50_slope_atr": (5, 4),
            "ema100_slope_atr": (5, 1), "ema200_slope_atr": (5, 1),
            "ema_stack_aligned_v2": (1, 1), "vwap_rolling5_dist_atr": (1, 2),
            "vwap20_dist_atr": (19, 8), "vwap96_dist_atr": (95, 8),
            "vwap_rolling5_slope_atr": (5, 4), "adx14": (100, 1),
            "di_spread_signed": (1, 1), "trend_state_age_bars": (40, 1),
        }
        for name, (num, den) in rational.items():
            self._encloses(values[name], bounds[name], 239, num, den)
        # BB20 population variance is133/4. Squaring the positive exact
        # interval avoids a guessed epsilon for the irrational square root.
        from fractions import Fraction
        for name, offset, target in [("bb_width_atr", 0, Fraction(133, 4)),
                                     ("bb_position", Fraction(1, 2), Fraction(361, 2128))]:
            centre = Fraction.from_float(float(values[name][239]))-offset
            radius = Fraction.from_float(float(bounds[name][239]))
            assert centre-radius > 0
            assert (centre-radius)**2 <= target <= (centre+radius)**2

    def test_ema_impulse_has_sma_seed_then_half_weight_recursion(self):
        values, bounds = self._owner().reference_technical_ema(
            np.array([1., 1., 4., 1., 1.], np.float64), 3)
        np.testing.assert_array_equal(values, np.array([np.nan, np.nan, 2., 1.5, 1.25]))
        for row, numerator, denominator in [(2, 2, 1), (3, 3, 2), (4, 5, 4)]:
            self._encloses(values, bounds, row, numerator, denominator)

    def test_wilder_primitive_has_mean_seed_and_rational_recurrence(self):
        values, bounds = self._owner().reference_technical_wilder(
            np.array([2., 2., 8., 2., 2.], np.float64), 3)
        assert np.isnan(values[:2]).all()
        for row, numerator, denominator in [(2, 4, 1), (3, 10, 3), (4, 26, 9)]:
            self._encloses(values, bounds, row, numerator, denominator)

    def test_atr_gap_uses_previous_close_and_keeps_raw_bps_units(self):
        close = np.array([100.]*14 + [110., 110.], np.float64)
        frame = self._frame(close, high=np.array([101.]*14 + [111., 111.]),
                            low=np.array([99.]*14 + [109., 109.]))
        values, bounds, _ = self._owner().reference_technical_block(frame)
        assert np.isnan(values["atr_bps_14"][:13]).all()
        self._encloses(values["atr_bps_14"], bounds["atr_bps_14"], 13, 200)
        # ATR14: 2 -> (13*2+11)/14 -> (13*(37/14)+2)/14.
        self._encloses(values["atr_bps_14"], bounds["atr_bps_14"], 14, 37000, 154)
        self._encloses(values["atr_bps_14"], bounds["atr_bps_14"], 15, 127250, 539)
        self._encloses(values["mom_5_atr"], bounds["mom_5_atr"], 14, 140, 37)

    def test_rsi_fourteen_changes_and_centered_scale_have_rational_answer(self):
        # Seven +1 and seven -1 changes seed equal mean gain/loss.  The
        # following +1 gives gain15/28, loss13/28 and centred RSI1/14.
        close = np.array([100.]+[101. if i % 2 else 100. for i in range(1, 15)]+[101.])
        values, bounds, _ = self._owner().reference_technical_block(self._frame(close))
        assert np.isnan(values["rsi14_centered"][:14]).all()
        self._encloses(values["rsi14_centered"], bounds["rsi14_centered"], 14, 0)
        self._encloses(values["rsi14_centered"], bounds["rsi14_centered"], 15, 1, 14)

    @pytest.mark.parametrize("direction", [1, -1])
    def test_rsi_and_adx_directional_extremes_have_distinct_warmups(self, direction):
        close = 200 + direction*np.arange(40, dtype=np.float64)
        values, bounds, _ = self._owner().reference_technical_block(self._frame(close))
        assert np.isnan(values["rsi14_centered"][:14]).all()
        assert np.isnan(values["di_spread_signed"][:14]).all()
        assert np.isnan(values["adx14"][:27]).all()
        for row in (14, 39):
            self._encloses(values["rsi14_centered"], bounds["rsi14_centered"], row, direction)
            self._encloses(values["di_spread_signed"], bounds["di_spread_signed"], row, direction)
        for row in (27, 39):
            self._encloses(values["adx14"], bounds["adx14"], row, 100)

    def test_equal_outside_bar_moves_do_not_create_directional_movement(self):
        n = 40
        frame = self._frame(np.full(n, 100.), high=101+np.arange(n), low=99-np.arange(n))
        values, _, _ = self._owner().reference_technical_block(frame)
        # UpMove == DownMove: both DM values are zero. DI-sum is zero, so
        # neither signedDI nor ADX is an observed zero or a synthetic trend.
        assert np.isnan(values["di_spread_signed"]).all()
        assert np.isnan(values["adx14"]).all()

    def test_flat_close_preserves_legitimate_zero_and_unavailable_divisors(self):
        frame = self._frame(np.full(240, 100.), high=np.full(240, 101.), low=np.full(240, 99.))
        values, _, _ = self._owner().reference_technical_block(frame)
        assert values["atr_bps_14"][13] == 200
        assert values["rsi14_centered"][14] == 0
        assert values["bb_width_atr"][19] == 0
        assert np.isnan(values["bb_position"]).all()
        assert np.isnan(values["di_spread_signed"]).all()
        assert np.isnan(values["adx14"]).all()
        assert values["ema_stack_aligned_v2"][199] == 0
        assert values["trend_state_age_bars"][199] == 0
        assert values["trend_state_age_bars"][239] == 40

    def test_zero_atr_is_valid_bps_zero_but_never_a_division_floor(self):
        frame = self._frame(np.full(240, 100.), high=np.full(240, 100.), low=np.full(240, 100.))
        values, _, _ = self._owner().reference_technical_block(frame)
        assert values["atr_bps_14"][13] == 0
        for name in ("mom_5_atr", "mom_20_atr", "close_open_atr", "ema20_dist_atr",
                     "ema200_slope_atr", "vwap96_dist_atr", "bb_width_atr"):
            assert np.isnan(values[name]).all(), name

    def test_rolling_quote_vwap_has_unequal_weights_and_full_window(self):
        values, bounds = self._owner().reference_technical_rolling_vwap(
            np.arange(1, 7, dtype=np.float64), np.array([1, 1, 1, 1, 2, 1], np.int64), 5)
        assert np.isnan(values[:4]).all()
        self._encloses(values, bounds, 4, 10, 3)
        self._encloses(values, bounds, 5, 25, 6)

    def test_vwap_distance_and_lag_five_slope_use_the_same_rolling_operand(self):
        close = np.arange(100, 120, dtype=np.float64)
        frame = self._frame(close, high=close+2, low=close-2)
        values, bounds, _ = self._owner().reference_technical_block(frame)
        # Unit quote weights: VWAP5[t]=C[t]-2, ATR=4 throughout.  Its
        # five-observation change is5, irrespective of the clock gaps.
        self._encloses(values["vwap_rolling5_dist_atr"], bounds["vwap_rolling5_dist_atr"], 19, 1, 2)
        self._encloses(values["vwap_rolling5_slope_atr"], bounds["vwap_rolling5_slope_atr"], 19, 5, 4)
        self._encloses(values["vwap20_dist_atr"], bounds["vwap20_dist_atr"], 19, 19, 8)

    def test_bollinger_population_variance_and_atr_units_are_hand_computable(self):
        close = np.array([9.]*10 + [11.]*10, np.float64)
        stats, radii, _ = self._owner().reference_technical_bb20(close)
        for name, expected in [("mean", 10), ("variance", 1), ("std", 1)]:
            assert np.isnan(stats[name][:19]).all()
            self._encloses(stats[name], radii[name], 19, expected)
        frame = self._frame(close, high=np.full(20, 15.), low=np.full(20, 5.))
        values, bounds, _ = self._owner().reference_technical_block(frame)
        self._encloses(values["bb_position"], bounds["bb_position"], 19, 3, 4)
        self._encloses(values["bb_width_atr"], bounds["bb_width_atr"], 19, 2, 5)

    def test_all_twenty_three_fields_have_their_own_honest_warmup(self):
        values, _, report = self._owner().reference_technical_block(
            self._frame(100+np.arange(240, dtype=np.float64)))
        first = {
            "atr_bps_14": 13, "rsi14_centered": 14, "mom_5_atr": 13, "mom_20_atr": 20,
            "close_open_atr": 13, "ema20_dist_atr": 19, "ema50_dist_atr": 49,
            "ema100_dist_atr": 99, "ema200_dist_atr": 199, "ema20_slope_atr": 24,
            "ema50_slope_atr": 54, "ema100_slope_atr": 119, "ema200_slope_atr": 219,
            "ema_stack_aligned_v2": 199, "vwap_rolling5_dist_atr": 13,
            "vwap20_dist_atr": 19, "vwap96_dist_atr": 95, "vwap_rolling5_slope_atr": 13,
            "bb_position": 19, "bb_width_atr": 19, "adx14": 27,
            "di_spread_signed": 14, "trend_state_age_bars": 199,
        }
        assert set(first) == set(values)
        assert report["first_valid_rows"] == first
        for name, row in first.items():
            assert np.isnan(values[name][:row]).all(), name
            assert np.isfinite(values[name][row]), name

    def test_stack_ties_direction_changes_and_raw_state_age_are_not_saturated(self):
        close = np.array([100.]*200 + [101., 99., 100.], np.float64)
        values, _, _ = self._owner().reference_technical_block(self._frame(close))
        np.testing.assert_array_equal(values["ema_stack_aligned_v2"][199:], [0, 1, -1, -1])
        np.testing.assert_array_equal(values["trend_state_age_bars"][199:], [0, 0, 0, 1])
        flat, _, _ = self._owner().reference_technical_block(self._frame(np.full(520, 100.)))
        assert flat["trend_state_age_bars"][519] == 320

    def test_calendar_gaps_do_not_reset_any_recurrence_or_rolling_window(self):
        frame = self._frame(100+np.arange(240, dtype=np.float64))
        gapped = {name: array.copy() for name, array in frame.items()}
        gapped["bar_start_ns"][100:] += 17*DAY
        a, ab, _ = self._owner().reference_technical_block(frame)
        b, bb, _ = self._owner().reference_technical_block(gapped)
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
            np.testing.assert_array_equal(ab[name], bb[name])

    def test_full_native_prefix_is_seeded_before_the_cache_is_trimmed(self):
        close = np.full(400, 100., np.float64)
        close[20] = 200.
        frame = self._frame(close)
        full, _, report = self._owner().reference_technical_block(frame)
        cut = {name: array[-96:].copy() for name, array in frame.items()}
        reseeded, _, _ = self._owner().reference_technical_block(cut)
        assert report["source_rows"] == 400
        assert np.isfinite(full["ema200_dist_atr"][-96:]).all()
        assert full["ema200_dist_atr"][-1] < 0
        assert np.isnan(reseeded["ema200_dist_atr"]).all()
        assert full["trend_state_age_bars"][-1] >= 1

    @pytest.mark.parametrize("timeframe,minutes,origin", [
        ("M5", 5, "2009-06-01T00:00:00Z"), ("M15", 15, "2009-06-01T00:00:00Z"),
        ("H1", 60, "2009-06-01T00:00:00Z"), ("H4", 240, "2009-06-01T02:00:00Z"),
        ("D1", 1440, "2009-06-01T22:00:00Z"),
    ])
    def test_all_native_calendar_phases_and_first_max_min_last_sum(self, timeframe, minutes, origin):
        n = minutes//5
        stamp = pd.Timestamp(origin).value
        close = 100+np.arange(n, dtype=np.float64)
        frame = self._frame(close, volume=1+np.arange(n),
                            stamps=stamp+np.arange(n, dtype=np.int64)*5*MINUTE)
        frame["open"] -= .25
        result = self._owner().native_resample_calendar_frame(frame, timeframe)
        assert result["bar_start_ns"].tolist() == [stamp]
        assert result["open"].tolist() == [99.75]
        assert result["high"].tolist() == [99.5+n]
        assert result["low"].tolist() == [99.5]
        assert result["close"].tolist() == [99.+n]
        assert result["volume"].tolist() == [n*(n+1)//2]

    def test_resample_drops_only_leading_and_unclosed_trailing_partials(self):
        origin = pd.Timestamp("2009-06-01T00:00:00Z").value
        frame = self._frame(np.arange(100, 106, dtype=np.float64),
                            stamps=origin+np.array([5, 10, 15, 20, 25, 30], np.int64)*MINUTE)
        result = self._owner().native_resample_calendar_frame(frame, "M15")
        assert result["bar_start_ns"].tolist() == [origin+15*MINUTE]
        assert result["open"].tolist() == [102.]
        assert result["close"].tolist() == [104.]
        assert result["volume"].tolist() == [3]

    def test_resample_keeps_an_interior_partial_and_skips_empty_buckets(self):
        origin = pd.Timestamp("2009-06-01T00:00:00Z").value
        frame = self._frame(np.arange(100, 108, dtype=np.float64),
                            stamps=origin+np.array([0, 5, 10, 20, 25, 60, 65, 70], np.int64)*MINUTE)
        result = self._owner().native_resample_calendar_frame(frame, "M15")
        assert result["bar_start_ns"].tolist() == [origin, origin+15*MINUTE, origin+60*MINUTE]
        assert result["volume"].tolist() == [3, 2, 3]
        assert result["close"].tolist() == [102., 104., 107.]

    def test_float64_native_prices_are_not_precast_before_indicators(self):
        precise = self._frame(10000+np.arange(240, dtype=np.float64)/10000)
        precise["high"] = precise["close"]+.00005
        precise["low"] = precise["close"]-.00005
        rounded = {name: array.copy() for name, array in precise.items()}
        for name in ("open", "high", "low", "close"):
            rounded[name] = rounded[name].astype(np.float32).astype(np.float64)
        native, _, _ = self._owner().reference_technical_block(precise)
        wrong, _, _ = self._owner().reference_technical_block(rounded)
        assert np.isfinite(native["atr_bps_14"][13:]).all()
        assert np.any(native["atr_bps_14"][13:] != wrong["atr_bps_14"][13:])

    def test_shadow_bound_accepts_final_float32_rounding_but_catches_wrong_value(self):
        audit = self._owner()
        reference = {"ema20_dist_atr": np.array([1+2.**-25], np.float64)}
        bounds = {"ema20_dist_atr": np.array([0.], np.float64)}
        actual = {"ema20_dist_atr": np.array([1.], np.float32)}
        result = audit.compare_technical_reference(reference, actual, bounds)
        assert result["ema20_dist_atr"]["numeric_reference_qualified"] is True
        wrong = {"ema20_dist_atr": np.nextafter(actual["ema20_dist_atr"], np.float32(np.inf))}
        with pytest.raises(RuntimeError, match="TECHNICAL_NUMERIC_REFERENCE_"):
            audit.compare_technical_reference(reference, wrong, bounds)

    @pytest.mark.parametrize("fault", ["wrong_value", "missing_observation", "synthetic_observation", "discrete"])
    def test_comparison_rejects_value_mask_and_discrete_faults(self, fault):
        audit = self._owner()
        reference = {"rsi14_centered": np.array([np.nan, 0.], np.float64),
                     "ema_stack_aligned_v2": np.array([np.nan, 0.], np.float64)}
        bounds = {name: np.array([np.nan, 0.], np.float64) for name in reference}
        actual = {name: array.astype(np.float32) for name, array in reference.items()}
        if fault == "wrong_value": actual["rsi14_centered"][1] = .25
        elif fault == "missing_observation": actual["rsi14_centered"][1] = np.nan
        elif fault == "synthetic_observation": actual["rsi14_centered"][0] = 0
        else:
            bounds["ema_stack_aligned_v2"][1] = 1000
            actual["ema_stack_aligned_v2"][1] = 1
        with pytest.raises(RuntimeError, match="TECHNICAL_NUMERIC_REFERENCE_"):
            audit.compare_technical_reference(reference, actual, bounds)

    @pytest.mark.parametrize("fault", ["nan_price", "nonpositive_price", "bad_geometry", "zero_volume", "fractional_volume"])
    def test_invalid_native_input_is_rejected_without_reset_or_synthetic_values(self, fault):
        frame = self._frame(100+np.arange(240, dtype=np.float64))
        if fault == "nan_price": frame["close"][100] = np.nan
        elif fault == "nonpositive_price": frame["close"][100] = 0
        elif fault == "bad_geometry": frame["high"][100] = frame["low"][100]-1
        elif fault == "zero_volume": frame["volume"][100] = 0
        else:
            frame["volume"] = frame["volume"].astype(np.float64)
            frame["volume"][100] = 1.5
        with pytest.raises(RuntimeError, match="TECHNICAL_NUMERIC_REFERENCE_"):
            self._owner().reference_technical_block(frame)


    @pytest.mark.parametrize("case", ["remove_add_41", "uniform_then_variation", "large_binary_offset"])
    def test_required_full_prefix_bollinger_handchecks(self, case):
        from fractions import Fraction
        if case == "remove_add_41":
            close = np.arange(1, 42, dtype=np.float64)
            expected = [(row, Fraction(2*row-17, 2), Fraction(133, 4))
                        for row in range(19, 41)]
        elif case == "uniform_then_variation":
            close = np.concatenate([np.arange(1, 21, dtype=np.float64),
                                    np.full(20, 10.), np.array([9., 11.])])
            expected = [(39, Fraction(10), Fraction(0)),
                        (41, Fraction(10), Fraction(1, 10))]
        else:
            close = np.array([2.**40+(i % 4)/4 for i in range(80)], np.float64)
            expected = [(row, Fraction(8*2**40+3, 8), Fraction(5, 64))
                        for row in range(19, 80)]
        stats, radii, diagnostics = self._owner().reference_technical_bb20(close)
        for name in ("mean", "variance", "std"):
            assert np.isnan(stats[name][:19]).all()
        for row, mean, variance in expected:
            self._encloses(stats["mean"], radii["mean"], row, mean.numerator, mean.denominator)
            self._encloses(stats["variance"], radii["variance"], row,
                           variance.numerator, variance.denominator)
            centre = Fraction.from_float(float(stats["std"][row]))
            radius = Fraction.from_float(float(radii["std"][row]))
            assert radius >= 0 and centre+radius >= 0
            assert max(Fraction(0), centre-radius)**2 <= variance <= (centre+radius)**2
        if case == "uniform_then_variation":
            assert diagnostics["uniform_shortcuts"] >= 1
            assert diagnostics["full_prefix_shadow_not_reset_by_uniform_output"] is True
            block, _, _ = self._owner().reference_technical_block(self._frame(close))
            assert block["bb_width_atr"][39] == 0
            assert np.isnan(block["bb_position"][39])
            assert block["bb_position"][41] > 1  # No clipping at the upper band.
        elif case == "large_binary_offset":
            block, errors, _ = self._owner().reference_technical_block(self._frame(close))
            # Last close is B+3/4, mean B+3/8 and std sqrt(5)/8:
            # (position-1/2)^2 = (3/(4*sqrt(5)))^2 = 9/80.
            centre = Fraction.from_float(float(block["bb_position"][79]))-Fraction(1, 2)
            radius = Fraction.from_float(float(errors["bb_position"][79]))
            assert centre-radius > 0
            assert (centre-radius)**2 <= Fraction(9, 80) <= (centre+radius)**2



class TestTrendEventReference128:
    """Exact event/state witnesses, independent of production event owners."""

    @staticmethod
    def _owner():
        return TestTechnicalReference127._owner()

    @staticmethod
    def _frame(close):
        return TestTechnicalReference127._frame(close)

    @staticmethod
    def _fields():
        return (
            "ema20_50_bull_state", "ema20_50_cross_up", "ema20_50_cross_down", "ema20_50_state_age_bars",
            "ema50_200_bull_state", "ema50_200_cross_up", "ema50_200_cross_down", "ema50_200_state_age_bars",
            "price_x_ema20_cross_up", "price_x_ema20_cross_down",
            "price_x_ema50_cross_up", "price_x_ema50_cross_down",
            "price_x_ema200_cross_up", "price_x_ema200_cross_down",
            "price_vs_ema20_state_age_bars", "price_vs_ema50_state_age_bars", "price_vs_ema200_state_age_bars",
        )

    @staticmethod
    def _first_rows():
        return {
            "ema20_50_bull_state": 49, "ema20_50_cross_up": 50,
            "ema20_50_cross_down": 50, "ema20_50_state_age_bars": 49,
            "ema50_200_bull_state": 199, "ema50_200_cross_up": 200,
            "ema50_200_cross_down": 200, "ema50_200_state_age_bars": 199,
            "price_x_ema20_cross_up": 20, "price_x_ema20_cross_down": 20,
            "price_x_ema50_cross_up": 50, "price_x_ema50_cross_down": 50,
            "price_x_ema200_cross_up": 200, "price_x_ema200_cross_down": 200,
            "price_vs_ema20_state_age_bars": 19, "price_vs_ema50_state_age_bars": 49,
            "price_vs_ema200_state_age_bars": 199,
        }

    def test_literal_sign_changes_use_strict_crosses_and_boolean_age(self):
        spread = np.array([-1., 0., 1., 0., -1.], np.float64)
        state, up, down, age = self._owner().reference_trend_events_from_spread(spread, np.zeros(5))
        np.testing.assert_array_equal(state, [0., 0., 1., 0., 0.])
        np.testing.assert_array_equal(up, [np.nan, 0., 1., 0., 0.])
        np.testing.assert_array_equal(down, [np.nan, 0., 0., 0., 1.])
        np.testing.assert_array_equal(age, [0., 1., 0., 0., 1.])
        # +1 ->0 changes the bullish state without a strict down-cross;
        # 0 ->-1 triggers down-cross while the Boolean false-state ages.
        assert state[2] != state[3] and down[3] == 0
        assert state[3] == state[4] and age[4] == 1 and down[4] == 1

    def test_cross_warmup_requires_current_and_previous_observation(self):
        spread = np.array([np.nan, np.nan, -1., 1., 1.], np.float64)
        radii = np.array([np.nan, np.nan, 0., 0., 0.], np.float64)
        state, up, down, age = self._owner().reference_trend_events_from_spread(spread, radii)
        np.testing.assert_array_equal(state, [np.nan, np.nan, 0., 1., 1.])
        np.testing.assert_array_equal(up, [np.nan, np.nan, np.nan, 1., 0.])
        np.testing.assert_array_equal(down, [np.nan, np.nan, np.nan, 0., 0.])
        np.testing.assert_array_equal(age, [np.nan, np.nan, 0., 0., 1.])

    def test_signed_zero_and_two_hundred_constant_bars_are_exact_ties(self):
        audit = self._owner()
        state, up, down, age = audit.reference_trend_events_from_spread(
            np.array([0., -0., 0., -0.], np.float64), np.zeros(4))
        np.testing.assert_array_equal(state, [0., 0., 0., 0.])
        np.testing.assert_array_equal(up, [np.nan, 0., 0., 0.])
        np.testing.assert_array_equal(down, [np.nan, 0., 0., 0.])
        np.testing.assert_array_equal(age, [0., 1., 2., 3.])
        values, bounds, _ = audit.reference_trend_event_block(self._frame(np.full(200, 100.)))
        for name, first in self._first_rows().items():
            assert np.isnan(values[name][:first]).all(), name
            observed = np.isfinite(values[name])
            if name.endswith("age_bars"):
                np.testing.assert_array_equal(values[name][first:], np.arange(200-first))
            else:
                assert np.all(values[name][observed] == 0), name
            assert np.all(bounds[name][observed] == 0), name
        assert values["ema50_200_bull_state"][199] == 0
        assert values["ema50_200_state_age_bars"][199] == 0
        assert np.isnan(values["ema50_200_cross_up"]).all()

    @pytest.mark.parametrize("direction", [1, -1])
    def test_linear_prices_have_hand_known_all_seventeen_states_masks_and_raw_ages(self, direction):
        audit = self._owner()
        values, bounds, report = audit.reference_trend_event_block(
            self._frame(500+direction*np.arange(320, dtype=np.float64)))
        assert tuple(audit.TREND_EVENT_NUMERIC_FIELDS) == self._fields()
        assert tuple(values) == self._fields() and tuple(bounds) == self._fields()
        assert report["first_valid_rows"] == self._first_rows()
        for name, first in self._first_rows().items():
            assert values[name].dtype == np.float64 and bounds[name].dtype == np.float64
            assert np.isnan(values[name][:first]).all(), name
            assert np.isnan(bounds[name][:first]).all(), name
            assert np.all(bounds[name][first:] == 0), name
            if name.endswith("age_bars"):
                np.testing.assert_array_equal(values[name][first:], np.arange(320-first))
            elif name.endswith("bull_state"):
                assert np.all(values[name][first:] == int(direction > 0)), name
            else:
                assert np.all(values[name][first:] == 0), name

    def test_calendar_holes_do_not_reset_state_events_or_raw_observed_bar_ages(self):
        frame = self._frame(100+np.arange(240, dtype=np.float64))
        gapped = {name: values.copy() for name, values in frame.items()}
        gapped["bar_start_ns"][120:] += 13*DAY
        regular, rb, _ = self._owner().reference_trend_event_block(frame)
        actual, ab, _ = self._owner().reference_trend_event_block(gapped)
        for name in self._fields():
            np.testing.assert_array_equal(actual[name], regular[name])
            np.testing.assert_array_equal(ab[name], rb[name])
        assert actual["price_vs_ema20_state_age_bars"][-1] == 220

    def test_appending_future_price_changes_leaves_all_seventeen_prefixes_unchanged(self):
        close = np.concatenate([100+np.arange(240, dtype=np.float64), np.tile([60., 400.], 10)])
        whole_frame = self._frame(close)
        prefix_frame = {name: values[:240].copy() for name, values in whole_frame.items()}
        before, bb, _ = self._owner().reference_trend_event_block(prefix_frame)
        after, ab, _ = self._owner().reference_trend_event_block(whole_frame)
        for name in self._fields():
            np.testing.assert_array_equal(before[name], after[name][:240])
            np.testing.assert_array_equal(bb[name], ab[name][:240])

    @pytest.mark.parametrize("centre,radius", [(0., 1.), (1., 1.), (-1., 1.), (1., 2.)])
    def test_uncertified_three_way_sign_branch_is_rejected(self, centre, radius):
        # Touching zero is insufficient: a positive/negative/tie branch
        # must each be certified before any cache comparison can occur.
        with pytest.raises(RuntimeError, match="TREND_EVENT_BRANCH_AMBIGUOUS"):
            self._owner().reference_trend_events_from_spread(
                np.array([centre], np.float64), np.array([radius], np.float64))

    def test_nonzero_radius_is_allowed_when_the_sign_is_certified(self):
        state, up, down, age = self._owner().reference_trend_events_from_spread(
            np.array([-2., 0., 2.], np.float64), np.array([.25, 0., .25], np.float64))
        np.testing.assert_array_equal(state, [0., 0., 1.])
        np.testing.assert_array_equal(up, [np.nan, 0., 1.])
        np.testing.assert_array_equal(down, [np.nan, 0., 0.])
        np.testing.assert_array_equal(age, [0., 1., 0.])

    def test_missing_observation_inside_suffix_cannot_be_reset_to_a_fresh_state(self):
        with pytest.raises(RuntimeError, match="TREND_EVENT_NUMERIC_NAN_SUFFIX"):
            self._owner().reference_trend_events_from_spread(
                np.array([-1., np.nan, 1.], np.float64), np.array([0., np.nan, 0.], np.float64))

    @pytest.mark.parametrize("fault", ["missing_observation", "synthetic_observation", "wrong_bit", "wrong_category", "fractional_age"])
    def test_all_discrete_outputs_reject_mask_bit_category_and_age_changes(self, fault):
        audit = self._owner()
        values, bounds, _ = audit.reference_trend_event_block(self._frame(100+np.arange(240, dtype=np.float64)))
        actual = {name: array.astype(np.float32) for name, array in values.items()}
        valid = audit.compare_technical_reference(values, actual, bounds, discrete_fields=self._fields())
        assert all(valid[name]["discrete_exact"] is True for name in self._fields())
        if fault == "missing_observation": actual["ema20_50_bull_state"][-1] = np.nan
        elif fault == "synthetic_observation": actual["ema20_50_bull_state"][0] = 0
        elif fault == "wrong_bit": actual["ema20_50_bull_state"][-1] = 0
        elif fault == "wrong_category": actual["ema20_50_cross_up"][-1] = 2
        else: actual["price_vs_ema20_state_age_bars"][-1] += .25
        with pytest.raises(RuntimeError, match="TECHNICAL_NUMERIC_REFERENCE_(NAN_MASK|DISCRETE_MISMATCH)"):
            audit.compare_technical_reference(values, actual, bounds, discrete_fields=self._fields())



class TestNativeSMCReference129:
    """Literal native-spot replay and geometry witnesses; no producer oracle."""

    @staticmethod
    def _owner():
        return TestTechnicalReference127._owner()

    @staticmethod
    def _frame(high, low, close):
        return TestTechnicalReference127._frame(close, high=high, low=low)

    @classmethod
    def _literal(cls):
        high, low, close = np.full(15, 105.), np.full(15, 95.), np.full(15, 100.)
        high[[3, 10]] = [110., 112.]
        low[[4, 11]] = [90., 92.]
        return cls._frame(high, low, close)

    @classmethod
    def _extend(cls, frame, high, low, close):
        return cls._frame(np.concatenate([frame["high"], high]),
                          np.concatenate([frame["low"], low]),
                          np.concatenate([frame["close"], close]))

    @classmethod
    def _raw(cls, frame, *, lookback=3):
        return cls._owner().reference_native_smc_replay(
            frame["high"], frame["low"], frame["close"], swing_lookback=lookback)

    @staticmethod
    def _encloses(values, radii, row, numerator, denominator=1):
        TestTechnicalReference127._encloses(values, radii, row, numerator, denominator)

    @staticmethod
    def _fields():
        return (
            "mtf_smc_structure_bias", "mtf_smc_bos_up", "mtf_smc_bos_down",
            "mtf_smc_choch_up", "mtf_smc_choch_down", "mtf_smc_sweep_up_state", "mtf_smc_sweep_down_state",
            "mtf_smc_sweep_up_depth_atr", "mtf_smc_sweep_down_depth_atr",
            "mtf_smc_pivot_envelope_position", "mtf_smc_range_width_atr", "mtf_smc_bos_displacement_atr",
            "mtf_smc_sweep_up_event", "mtf_smc_sweep_down_event", "mtf_smc_sweep_event_age_bars",
            "mtf_smc_swing_state", "mtf_smc_sweep_last_event_side", "mtf_geometry_support_dist_atr",
            "mtf_geometry_resistance_dist_atr", "mtf_geometry_support_age_bars", "mtf_geometry_resistance_age_bars",
            "mtf_geometry_support_rail_slope_atr_per_bar", "mtf_geometry_resistance_rail_slope_atr_per_bar",
            "mtf_geometry_range_mid_dist_atr",
        )

    def test_fifteen_bar_literal_covers_all_twenty_four_fields_and_pre_warmup_event(self):
        audit = self._owner()
        frame = self._literal()
        values, radii, report = audit.reference_native_smc_block(frame)
        assert tuple(audit.NATIVE_SMC_NUMERIC_FIELDS) == self._fields()
        assert tuple(values) == self._fields() and tuple(radii) == self._fields()
        continuous = {"mtf_smc_sweep_up_depth_atr", "mtf_smc_sweep_down_depth_atr",
                      "mtf_smc_pivot_envelope_position", "mtf_smc_range_width_atr", "mtf_smc_bos_displacement_atr",
                      "mtf_geometry_support_dist_atr", "mtf_geometry_resistance_dist_atr",
                      "mtf_geometry_support_rail_slope_atr_per_bar", "mtf_geometry_resistance_rail_slope_atr_per_bar",
                      "mtf_geometry_range_mid_dist_atr"}
        assert set(audit.NATIVE_SMC_DISCRETE_FIELDS) == set(self._fields())-continuous
        assert report["source_rows"] == 15 and report["common_first_valid_row"] == 14
        # Four pivots are3/10(high),4/11(low), visible6/13 and7/14.
        # TR0..13 sums160: ATR13=80/7, ATR14=555/49.
        expected = {
            "mtf_smc_structure_bias": (1, 1), "mtf_smc_bos_up": (0, 1), "mtf_smc_bos_down": (0, 1),
            "mtf_smc_choch_up": (0, 1), "mtf_smc_choch_down": (0, 1),
            "mtf_smc_sweep_up_state": (0, 1), "mtf_smc_sweep_down_state": (0, 1),
            "mtf_smc_sweep_up_depth_atr": (0, 1), "mtf_smc_sweep_down_depth_atr": (0, 1),
            "mtf_smc_pivot_envelope_position": (5, 11), "mtf_smc_range_width_atr": (1078, 555),
            "mtf_smc_bos_displacement_atr": (0, 1), "mtf_smc_sweep_up_event": (0, 1),
            "mtf_smc_sweep_down_event": (0, 1), "mtf_smc_sweep_event_age_bars": (4, 1),
            "mtf_smc_swing_state": (0, 1), "mtf_smc_sweep_last_event_side": (1, 1),
            "mtf_geometry_support_dist_atr": (392, 555), "mtf_geometry_resistance_dist_atr": (588, 555),
            "mtf_geometry_support_age_bars": (3, 1), "mtf_geometry_resistance_age_bars": (4, 1),
            "mtf_geometry_support_rail_slope_atr_per_bar": (14, 555),
            "mtf_geometry_resistance_rail_slope_atr_per_bar": (14, 555),
            "mtf_geometry_range_mid_dist_atr": (-49, 555),
        }
        assert set(expected) == set(values)
        for name, (num, den) in expected.items():
            assert values[name].dtype == np.float64 and radii[name].dtype == np.float64
            assert np.isnan(values[name][:14]).all() and np.isnan(radii[name][:14]).all(), name
            self._encloses(values[name], radii[name], 14, num, den)
            if name in audit.NATIVE_SMC_DISCRETE_FIELDS:
                assert values[name][14] == num/den and radii[name][14] == 0
        raw = self._raw(frame)
        assert raw["sweep_up_event"][10] == 1
        assert raw["last_high_idx"][13] == 10 and raw["last_low_idx"][14] == 11
        assert values["mtf_smc_sweep_event_age_bars"][14] == 4  # Replay ran before output warmup.

    def test_right_three_confirmation_accepts_all_equal_neighbours_causally(self):
        frame = self._frame(np.full(10, 101.), np.full(10, 99.), np.full(10, 100.))
        raw = self._raw(frame)
        for side in ("high", "low"):
            assert raw["last_"+side+"_idx"][5] == -1
            assert raw["last_"+side+"_idx"][6] == 3
            assert raw["prev_"+side+"_idx"][6] == -1
            assert raw["last_"+side+"_idx"][7] == 4
            assert raw["prev_"+side+"_idx"][7] == 3
        assert raw["structure_available"][6] == 0 and raw["structure_available"][7] == 1

    @pytest.mark.parametrize("side,accepted", [("high", True), ("high", False), ("low", True), ("low", False)])
    def test_pivot_boundary_uses_the_existing_represented_tolerance(self, side, accepted):
        high, low, close = np.full(7, 100.), np.full(7, 99.), np.full(7, 99.5)
        if side == "high":
            threshold = np.float64(100.)-np.float64(1e-12)
            high[3] = threshold if accepted else np.nextafter(threshold, -np.inf)
        else:
            threshold = np.float64(99.)+np.float64(1e-12)
            low[3] = threshold if accepted else np.nextafter(threshold, np.inf)
        raw = self._raw(self._frame(high, low, close))
        assert raw["last_"+side+"_idx"][6] == (3 if accepted else -1)

    @pytest.mark.parametrize("mirror", [False, True])
    def test_unmocked_opposing_close_break_and_mirror_have_literal_choch(self, mirror):
        bars = np.array([
            [105.,104.,104.5], [110.,106.,108.], [106.5,105.8,106.], [102.,100.,101.],
            [107.,102.5,105.], [115.,107.,112.], [109.,106.,107.], [108.,105.,106.],
            [110.,105.5,107.], [112.,106.,109.], [108.,103.,105.], [107.,103.5,105.], [104.,98.,99.],
        ], np.float64)
        high, low, close = bars.T.copy()
        if mirror: high, low, close = 220-low, 220-high, 220-close
        raw = self._raw(self._frame(high, low, close), lookback=1)
        side, other = ("up", "down") if mirror else ("down", "up")
        assert raw["choch_"+side][11] == 0
        assert raw["bos_"+side][12] == 1 and raw["choch_"+side][12] == 1
        assert raw["choch_"+side].sum() == 1 and raw["choch_"+other].sum() == 0
        equal_close = close.copy();equal_close[12] = 117. if mirror else 103.
        equal = self._raw(self._frame(high, low, equal_close), lookback=1)
        assert equal["bos_"+side][12] == 0 and equal["choch_"+side][12] == 0

    def test_equal_price_new_pivot_identity_rearms_the_sweep_once(self):
        high = np.array([101.,105.,103.,106.,104.,105.,104.,106.,107.,107.])
        close = np.array([100.,104.,102.,104.,103.,104.,103.,104.,106.,104.])
        raw = self._raw(self._frame(high, np.full(10, 99.), close), lookback=1)
        np.testing.assert_array_equal(raw["sweep_up_event"], [0,0,0,1,0,0,0,1,0,0])
        assert raw["last_high_idx"][2] == 1 and raw["last_high_idx"][6] == 5
        assert raw["last_high_price"][2] == raw["last_high_price"][6] == 105
        assert raw["structure_bias"][4] == .5  # Higher high, equal low.
        assert raw["bos_up"][8] == 1 and raw["sweep_up"][9] == 0

    def test_repeated_wick_condition_after_false_gap_does_not_rearm_same_pivot(self):
        frame = self._literal();frame["high"][12] = 111.
        raw = self._raw(frame)
        np.testing.assert_array_equal(raw["last_high_idx"][10:13], [3,3,3])
        np.testing.assert_array_equal(raw["sweep_up"][10:13], [1,0,1])
        np.testing.assert_array_equal(raw["sweep_up_event"][10:13], [1,0,0])
        np.testing.assert_array_equal(raw["sweep_event_age_bars"][10:13], [0,1,2])

    @pytest.mark.parametrize("boundary", ["wick_touch", "close_equal", "close_through"])
    def test_wick_is_strict_closeback_inclusive_and_bos_consumes_first(self, boundary):
        frame = self._literal()
        if boundary == "wick_touch": frame["high"][10] = 110.
        elif boundary == "close_equal": frame["close"][10] = frame["open"][10] = 110.
        else: frame["close"][10] = frame["open"][10] = 111.
        raw = self._raw(frame)
        assert raw["bos_up"][10] == int(boundary == "close_through")
        assert raw["sweep_up"][10] == int(boundary == "close_equal")
        assert raw["sweep_up_event"][10] == int(boundary == "close_equal")

    def test_double_sweep_has_two_depths_held_side_zero_and_raw_age(self):
        frame = self._extend(self._literal(), [114.,105.,105.], [88.,95.,95.], [100.,100.,100.])
        values, radii, _ = self._owner().reference_native_smc_block(frame)
        # ATR15=(13*(555/49)+26)/14=8489/686. Both levels are new
        # confirmed identities: high112 and low92, with wick114/88.
        for side in ("up", "down"):
            assert values["mtf_smc_sweep_"+side+"_state"][15] == 1
            assert values["mtf_smc_sweep_"+side+"_event"][15] == 1
        self._encloses(values["mtf_smc_sweep_up_depth_atr"], radii["mtf_smc_sweep_up_depth_atr"], 15, 1372, 8489)
        self._encloses(values["mtf_smc_sweep_down_depth_atr"], radii["mtf_smc_sweep_down_depth_atr"], 15, 2744, 8489)
        np.testing.assert_array_equal(values["mtf_smc_sweep_event_age_bars"][15:], [0,1,2])
        np.testing.assert_array_equal(values["mtf_smc_sweep_last_event_side"][15:], [0,0,0])

    def test_collapsed_envelope_keeps_true_zero_geometry_and_own_event_nan_prefix(self):
        high, low, close = np.full(30, 100.), np.full(30, 100.), np.full(30, 100.)
        high[:7] = 105.;low[:7] = 95.
        values, _, _ = self._owner().reference_native_smc_block(self._frame(high, low, close))
        for name in ("mtf_smc_pivot_envelope_position", "mtf_smc_range_width_atr",
                     "mtf_geometry_support_dist_atr", "mtf_geometry_resistance_dist_atr",
                     "mtf_geometry_support_rail_slope_atr_per_bar", "mtf_geometry_resistance_rail_slope_atr_per_bar",
                     "mtf_geometry_range_mid_dist_atr"):
            np.testing.assert_array_equal(values[name][14:], np.zeros(16))
        np.testing.assert_array_equal(values["mtf_geometry_support_age_bars"][14:], np.full(16, 3.))
        np.testing.assert_array_equal(values["mtf_geometry_resistance_age_bars"][14:], np.full(16, 3.))
        assert np.isnan(values["mtf_smc_sweep_event_age_bars"]).all()
        assert np.isnan(values["mtf_smc_sweep_last_event_side"]).all()

    @pytest.mark.parametrize("side", ["above", "below"])
    def test_outside_envelope_positions_remain_unclipped(self, side):
        high, low, close = ([122.],[95.],[120.]) if side == "above" else ([105.],[78.],[80.])
        frame = self._extend(self._literal(), high, low, close)
        values, radii, _ = self._owner().reference_native_smc_block(frame)
        target = 15 if side == "above" else -5
        self._encloses(values["mtf_smc_pivot_envelope_position"], radii["mtf_smc_pivot_envelope_position"], 15, target, 11)
        assert (values["mtf_smc_pivot_envelope_position"][15] > 1 if side == "above"
                else values["mtf_smc_pivot_envelope_position"][15] < 0)

    @pytest.mark.parametrize("mirror", [False, True])
    def test_simultaneous_bos_selects_the_newest_pivot_with_real_tolerance_edge(self, mirror):
        from fractions import Fraction
        delta = 2.**-42
        high, low, close = np.full(8, 100.+delta), np.full(8, 100.+delta), np.full(8, 100.+delta)
        high[4] = low[4] = close[4] = 100.
        high[7] = close[7] = 100.+delta/2
        low[7] = 100.-2.**-38
        frame = self._extend(self._literal(), high, low, close)
        if mirror:
            original_high = frame["high"].copy()
            frame["high"] = 200-frame["low"]
            frame["low"] = 200-original_high
            frame["close"] = 200-frame["close"]
            frame["open"] = frame["close"].copy()
        raw = self._raw(frame)
        assert raw["bos_up"][22] == raw["bos_down"][22] == 1
        newest, older = ("low", "high") if mirror else ("high", "low")
        assert raw["last_"+newest+"_idx"][22] == 19
        assert raw["last_"+older+"_idx"][22] == 18
        # Equal-index dual BOS cannot occur in valid OHLC: C>H and C<L
        # would contradict H>=L. The valid unequal-index edge covers both
        # reachable priority branches without inventing invalid market data.
        atr = Fraction(555,49)
        d = Fraction(1,2**42)
        for true_range in (d,0,0,0,d,d,0,17*d): atr = (13*atr+true_range)/14
        expected = (-d/2 if mirror else d/2)/atr
        values, radii, _ = self._owner().reference_native_smc_block(frame)
        self._encloses(values["mtf_smc_bos_displacement_atr"], radii["mtf_smc_bos_displacement_atr"], 22,
                       expected.numerator, expected.denominator)
        assert (values["mtf_smc_bos_displacement_atr"][22] < 0 if mirror
                else values["mtf_smc_bos_displacement_atr"][22] > 0)

    def test_calendar_holes_do_not_reset_and_future_cannot_rewrite_the_prefix(self):
        frame = self._extend(self._literal(), [114.,105.,105.], [88.,95.,95.], [100.,100.,100.])
        gapped = {name: a.copy() for name, a in frame.items()}
        gapped["bar_start_ns"][10:] += 17*DAY
        original, ob, _ = self._owner().reference_native_smc_block(frame)
        gap, gb, _ = self._owner().reference_native_smc_block(gapped)
        future = self._extend(frame, [121.,81.,121.,81.,121.], [119.,79.,119.,79.,119.], [120.,80.,120.,80.,120.])
        extended, eb, _ = self._owner().reference_native_smc_block(future)
        for name in self._fields():
            np.testing.assert_array_equal(gap[name], original[name])
            np.testing.assert_array_equal(gb[name], ob[name])
            np.testing.assert_array_equal(extended[name][:18], original[name])
            np.testing.assert_array_equal(eb[name][:18], ob[name])

    @pytest.mark.parametrize("fault", ["wrong_value", "wrong_mask", "wrong_category"])
    def test_comparison_rejects_wrong_continuous_value_mask_and_discrete_category(self, fault):
        audit = self._owner()
        values, radii, _ = audit.reference_native_smc_block(self._literal())
        actual = {name: a.astype(np.float32) for name, a in values.items()}
        valid = audit.compare_technical_reference(values, actual, radii, discrete_fields=audit.NATIVE_SMC_DISCRETE_FIELDS)
        assert all(valid[name]["numeric_reference_qualified"] for name in self._fields())
        if fault == "wrong_value": actual["mtf_geometry_support_dist_atr"][14] += 1
        elif fault == "wrong_mask": actual["mtf_smc_structure_bias"][0] = 0
        else: actual["mtf_smc_swing_state"][14] = 7
        with pytest.raises(RuntimeError, match="TECHNICAL_NUMERIC_REFERENCE_(VALUE_MISMATCH|NAN_MASK|DISCRETE_MISMATCH)"):
            audit.compare_technical_reference(values, actual, radii, discrete_fields=audit.NATIVE_SMC_DISCRETE_FIELDS)


class TestNativeCoreReader130:
    @staticmethod
    def _owner():
        from gx1.scripts import audit_entry_gc_foundation_v1
        return audit_entry_gc_foundation_v1

    def test_internal_float32_wire_has_exact_zero_radius_for_exact_binary64_input(self):
        a=self._owner();v=np.array([0.,1.,-1.,1.+2.**-24],np.float64)
        out,r=a._core_round32_interval(v,np.zeros(4))
        np.testing.assert_array_equal(out,np.array([0.,1.,-1.,1.]))
        np.testing.assert_array_equal(r,np.zeros(4))

    def test_interval_crossing_float32_midpoint_covers_both_tie_neighbours(self):
        a=self._owner();mid=1.+2.**-24
        out,r=a._core_round32_interval(np.array([mid,-mid]),np.full(2,2.**-52))
        assert out.tolist()==[1.,-1.]
        assert np.all(r>=2.**-23) and np.all(r<2.**-22)
        for sign in (-1.,1.):
            actual=np.asarray([mid+sign*2.**-52,-mid+sign*2.**-52],np.float32).astype(np.float64)
            assert np.all(np.abs(actual-out)<=r)

    def test_internal_float32_wire_preserves_nan_prefix_and_subnormal_interval(self):
        a=self._owner();out,r=a._core_round32_interval(np.array([np.nan,0.]),np.array([np.nan,2.**-149]))
        assert np.isnan(out[0]) and np.isnan(r[0]) and out[1]==0. and r[1]>=2.**-149

    @pytest.mark.parametrize('values,radii',[(np.array([np.inf]),np.array([0.])),
        (np.array([1.]),np.array([-1.])),(np.array([1.]),np.array([np.nan]))])
    def test_invalid_internal_float32_wire_is_rejected(self,values,radii):
        with pytest.raises(RuntimeError,match='F32_INTERVAL_DOMAIN'):
            self._owner()._core_round32_interval(values,radii)

    def test_internal_float32_overflow_is_rejected(self):
        with pytest.raises(RuntimeError,match='F32_INTERVAL_OVERFLOW'):
            self._owner()._core_round32_interval(np.array([1.e40]),np.array([0.]))

    @staticmethod
    def _source():
        minute=60*10**9
        t=np.array([0,5,10,15,20,25,35,40,45],np.int64)*minute
        c=np.array([100.,101.,102.,103.,104.,105.,107.,108.,109.])
        out={'bar_start_ns':t,'open':c.copy(),'high':c+2.,'low':c-2.,'close':c.copy(),'volume':np.arange(1,10,dtype=np.int64)}
        for side,offset in [('bid',-0.25),('ask',0.25)]:
            for part in ['close','high','low']:out[side+'_'+part]=out[part]+offset
        return out

    def test_quotes_share_native_axis_with_observed_extrema_and_trailing_trim(self):
        source=self._source();out=self._owner().native_core_resample_frame(source,'M15')
        np.testing.assert_array_equal(out['bar_start_ns'],np.array([0,15,30],np.int64)*60*10**9)
        np.testing.assert_array_equal(out['bid_close'],[101.75,104.75,107.75])
        np.testing.assert_array_equal(out['ask_high'],[104.25,107.25,110.25])
        np.testing.assert_array_equal(out['bid_low'],[97.75,100.75,104.75])
        np.testing.assert_array_equal(out['volume'],[6,15,15])

    def test_quote_resampling_preserves_leading_partial_and_empty_bucket_rules(self):
        source=self._source();source={k:v[1:6] for k,v in source.items()}
        out=self._owner().native_core_resample_frame(source,'M15')
        np.testing.assert_array_equal(out['bar_start_ns'],np.array([15],np.int64)*60*10**9)
        np.testing.assert_array_equal(out['ask_close'],[105.25])

    def test_quote_resampling_rejects_crossed_genuine_quotes(self):
        source=self._source();source['ask_close']=source['bid_close']-1.
        with pytest.raises(RuntimeError,match='CROSSED_QUOTE'):
            self._owner().native_core_resample_frame(source,'M15')

    def test_prior_reference_modes_remain_mutually_exclusive_with_core(self):
        with pytest.raises(RuntimeError,match='MUTUALLY_EXCLUSIVE_MODES'):
            self._owner().run_technical_numeric(None,native_smc=True,remaining_core=True)


class TestNativeCoreSwing130:
    """Literal strict-pivot witnesses; no production swing/ATR/age oracle."""

    @staticmethod
    def _owner():
        from gx1.scripts import audit_entry_gc_foundation_v1
        return audit_entry_gc_foundation_v1

    @staticmethod
    def _frame(close):
        c=np.asarray(close,dtype=np.float64)
        return {'bar_start_ns':np.arange(len(c),dtype=np.int64)*5*MINUTE,
            'high':c+.5,'low':c-.5,'close':c}

    @staticmethod
    def _atr(frame):
        from fractions import Fraction
        h,l,c=(frame[name] for name in ('high','low','close'));tr=[]
        for i in range(len(c)):
            candidates=[Fraction.from_float(float(h[i]))-Fraction.from_float(float(l[i]))]
            if i:
                candidates.extend((abs(Fraction.from_float(float(h[i]))-Fraction.from_float(float(c[i-1]))),
                    abs(Fraction.from_float(float(l[i]))-Fraction.from_float(float(c[i-1])))))
            tr.append(max(candidates))
        values=np.full(len(c),np.nan);radii=values.copy();exact=None
        for i in range(13,len(c)):
            exact=sum(tr[:14],Fraction(0))/14 if i==13 else (13*exact+tr[i])/14
            value=float(exact);error=abs(Fraction.from_float(value)-exact)
            values[i]=value;radii[i]=0. if not error else np.nextafter(float(error),np.inf)
        return values,radii

    @staticmethod
    def _close():
        return [100.,101.,103.,101.,100.,100.5,102.,102.5,104.,104.5,105.,104.,103.,103.5,106.,106.5,107.]

    def _calculate(self,frame):
        atr,radius=self._atr(frame)
        return self._owner().reference_native_core_swing(frame,atr=atr,atr_radius=radius)

    @pytest.mark.parametrize('mirror',[False,True])
    def test_literal_twenty_fields_have_exact_rational_formula_witness(self,mirror):
        from fractions import Fraction as F
        frame=self._frame(self._close())
        if mirror:frame.update(high=200.-frame['low'],low=200.-frame['high'],close=200.-frame['close'])
        values,bounds,report=self._calculate(frame);assert report['source_rows']==17
        expected={
            'swing_dist_last_swing_high_atr':-F(98,45) if mirror else F(14,45),
            'swing_dist_last_swing_low_atr':-F(14,45) if mirror else F(98,45),
            'swing_bars_since_swing_high':2 if mirror else 4,
            'swing_bars_since_swing_low':4 if mirror else 2,
            'swing_retracement_from_last_impulse':F(7,6),'swing_impulse_present':1,
            'swing_high_break_event':0 if mirror else 1,'swing_low_break_event':1 if mirror else 0,
            'swing_high_break_displacement_atr':0 if mirror else F(14,45),
            'swing_low_break_displacement_atr':F(14,45) if mirror else 0,
            'bars_since_swing_high_break':None if mirror else 0,'bars_since_swing_low_break':0 if mirror else None,
            'swing_high_sequence_delta_atr':-F(28,15) if mirror else F(56,45),
            'swing_low_sequence_delta_atr':-F(56,45) if mirror else F(28,15),
            'consecutive_higher_lows_count':0 if mirror else 1,'consecutive_lower_highs_count':1 if mirror else 0,
            'consecutive_higher_highs_count':0 if mirror else 1,'consecutive_lower_lows_count':1 if mirror else 0,
            'swing_high_level_intact':1 if mirror else 0,'swing_low_level_intact':0 if mirror else 1}
        assert tuple(values)==tuple(expected)==self._owner().NATIVE_CORE_SWING_FIELDS
        assert len(self._owner().NATIVE_CORE_SWING_DISCRETE_FIELDS)==13
        for name,truth in expected.items():
            if truth is None:assert np.isnan(values[name][14]) and np.isnan(bounds[name][14])
            else:
                error=abs(F.from_float(float(values[name][14]))-F(truth))
                assert error<=F.from_float(float(bounds[name][14])),name
                if name in self._owner().NATIVE_CORE_SWING_DISCRETE_FIELDS:assert values[name][14]==float(truth) and bounds[name][14]==0.
        assert report['first_valid_rows']['swing_dist_last_swing_high_atr']==13
        assert report['first_valid_rows']['swing_dist_last_swing_low_atr']==13
        assert values['swing_retracement_from_last_impulse'][14]>1.

    def test_pre_atr_break_consumes_level_but_has_undefined_displacement(self):
        values,bounds,_=self._calculate(self._frame(self._close()))
        assert np.flatnonzero(values['swing_high_break_event']).tolist()==[8,14]
        assert values['swing_high_level_intact'][8]==0.
        assert np.isnan(values['swing_high_break_displacement_atr'][:13]).all()
        assert np.isnan(bounds['swing_high_break_displacement_atr'][:13]).all()
        np.testing.assert_array_equal(values['bars_since_swing_high_break'][8:],[0.,1.,2.,3.,4.,5.,0.,1.,2.])
        assert np.isnan(values['bars_since_swing_low_break']).all()

    def test_equal_neighbour_highs_are_rejected_without_tolerance(self):
        values,_,report=self._calculate(self._frame([100.,101.,103.,103.,101.,100.]))
        assert report['pivot_counts']['high']==0
        assert np.isnan(values['swing_bars_since_swing_high']).all()
        assert np.isnan(values['swing_high_level_intact']).all()
        assert not values['swing_high_break_event'].any()

    def test_right_two_confirmation_has_no_pseudopivot_or_age_cap(self):
        close=np.r_[100.,101.,103.,101.,100.,99.95-.05*np.arange(615,dtype=np.float64)]
        values,_,_=self._calculate(self._frame(close))
        assert np.isnan(values['swing_bars_since_swing_high'][:4]).all()
        assert values['swing_bars_since_swing_high'][4]==2.
        assert values['swing_bars_since_swing_high'][-1]==len(close)-3
        assert values['swing_bars_since_swing_high'][-1]>500.
        assert np.isnan(values['swing_bars_since_swing_low']).all()

    def test_same_price_new_pivot_rearms_and_equal_sequence_resets_both_runs(self):
        close=[100.,101.,103.,101.,100.,101.,104.,104.,104.,101.,100.,103.,101.,100.,101.,104.]
        values,_,_=self._calculate(self._frame(close))
        assert np.flatnonzero(values['swing_high_break_event']).tolist()==[6,15]
        assert values['swing_high_level_intact'][12]==0.
        assert values['swing_high_level_intact'][13]==1.
        assert values['swing_high_level_intact'][15]==0.
        assert values['swing_high_sequence_delta_atr'][13]==0.
        assert values['consecutive_higher_highs_count'][13]==values['consecutive_lower_highs_count'][13]==0.

    def test_raw_pivot_run_counts_can_exceed_the_retired_ninety_six_cap(self):
        close=50.+np.tile([0.,1.,3.,1.,0.],130)+np.repeat(np.arange(130,dtype=np.float64),5)*4.
        values,_,_=self._calculate(self._frame(close))
        assert values['consecutive_higher_highs_count'][-1]==129.
        assert values['consecutive_higher_lows_count'][-1]==128.
        assert values['consecutive_lower_highs_count'][-1]==values['consecutive_lower_lows_count'][-1]==0.

    def test_same_row_opposite_pivots_have_no_defined_impulse(self):
        frame=self._frame(np.full(16,100.));frame['high'][2]=110.;frame['low'][2]=90.
        values,_,_=self._calculate(frame)
        assert values['swing_bars_since_swing_high'][4]==values['swing_bars_since_swing_low'][4]==2.
        assert not values['swing_impulse_present'].any()
        assert not values['swing_retracement_from_last_impulse'].any()
        assert np.isfinite(values['swing_dist_last_swing_high_atr'][13:]).all()

    def test_retracement_can_be_negative_without_atr_or_clip_floor(self):
        frame=self._frame(self._close());frame['close'][13]=106.;frame['high'][13]=106.5;frame['low'][13]=105.5
        values,bounds,_=self._calculate(frame)
        from fractions import Fraction as F
        assert values['swing_retracement_from_last_impulse'][13]<0.
        assert abs(F.from_float(float(values['swing_retracement_from_last_impulse'][13]))+F(1,12))<=F.from_float(float(bounds['swing_retracement_from_last_impulse'][13]))

    def test_zero_atr_is_unavailable_without_hiding_observed_zero_flags(self):
        frame=self._frame(np.full(20,100.));frame['high'][:]=100.;frame['low'][:]=100.
        values,bounds,_=self._calculate(frame)
        for name in ('swing_dist_last_swing_high_atr','swing_dist_last_swing_low_atr','swing_high_break_displacement_atr','swing_low_break_displacement_atr'):
            assert np.isnan(values[name]).all() and np.isnan(bounds[name]).all()
        for name in ('swing_impulse_present','swing_high_break_event','consecutive_higher_highs_count'):
            assert np.all(values[name]==0.) and np.all(bounds[name]==0.)

    def test_holes_do_not_reset_and_future_append_cannot_change_prefix(self):
        frame=self._frame(self._close()+[106.,105.,106.,108.]);prefix={k:v[:15].copy() for k,v in frame.items()}
        first,fb,_=self._calculate(prefix);whole,wb,_=self._calculate(frame)
        gap={k:v.copy() for k,v in frame.items()};gap['bar_start_ns'][9:]+=13*DAY
        gapped,gb,_=self._calculate(gap)
        for name in first:
            np.testing.assert_array_equal(first[name],whole[name][:15]);np.testing.assert_array_equal(fb[name],wb[name][:15])
            np.testing.assert_array_equal(gapped[name],whole[name]);np.testing.assert_array_equal(gb[name],wb[name])

    def test_reference_does_not_delegate_to_production_swing_or_age(self,monkeypatch):
        from gx1.features import swing_structure_v1,event_age_v1,technical_indicators_v1
        def forbidden(*args,**kwargs):raise AssertionError('production oracle called')
        monkeypatch.setattr(swing_structure_v1,'compute_swing_structure_features',forbidden)
        monkeypatch.setattr(event_age_v1,'raw_event_age_bars',forbidden)
        monkeypatch.setattr(technical_indicators_v1,'wilder_atr',forbidden)
        values,_,_=self._calculate(self._frame(self._close()))
        assert values['swing_high_break_event'][14]==1.

    @pytest.mark.parametrize('fault',['mask','category','continuous'])
    def test_comparison_rejects_wrong_mask_discrete_and_continuous_values(self,fault):
        audit=self._owner();values,bounds,_=self._calculate(self._frame(self._close()))
        actual={name:a.astype(np.float32) for name,a in values.items()}
        audit.compare_technical_reference(values,actual,bounds,discrete_fields=audit.NATIVE_CORE_SWING_DISCRETE_FIELDS)
        if fault=='mask':actual['swing_high_break_event'][0]=np.nan
        elif fault=='category':actual['consecutive_higher_highs_count'][14]=.5
        else:actual['swing_dist_last_swing_high_atr'][14]=1e6
        with pytest.raises(RuntimeError,match='TECHNICAL_NUMERIC_REFERENCE_(NAN_MASK|DISCRETE_MISMATCH|VALUE_MISMATCH)'):
            audit.compare_technical_reference(values,actual,bounds,discrete_fields=audit.NATIVE_CORE_SWING_DISCRETE_FIELDS)

    @pytest.mark.parametrize('fault',['shape','nan_mask','negative_radius'])
    def test_shared_atr_schema_is_checked_before_interpreting_its_domain(self,fault):
        frame=self._frame(self._close());a,b=self._atr(frame)
        if fault=='shape':a=a[:-1]
        elif fault=='nan_mask':b[0]=0.
        else:b[14]=-1.
        with pytest.raises(RuntimeError,match='NATIVE_CORE_SWING_ATR_SCHEMA'):
            self._owner().reference_native_core_swing(frame,atr=a,atr_radius=b)


class TestNativeCoreLocal130:
    @staticmethod
    def _owner():
        return TestTechnicalReference127._owner()

    @staticmethod
    def _frame(close, *, high=None, low=None, volume=None, stamps=None):
        c = np.asarray(close,dtype=np.float64)
        h = c+2 if high is None else np.asarray(high,dtype=np.float64)
        l = c-2 if low is None else np.asarray(low,dtype=np.float64)
        f = TestTechnicalReference127._frame(c,high=h,low=l,volume=volume,stamps=stamps)
        f.update({'bid_close':c-.5,'ask_close':c+.5,'bid_high':h-.5,'bid_low':l-.5,
                  'ask_high':h+.5,'ask_low':l+.5})
        return f

    @staticmethod
    def _raw(n):
        out = {name:np.zeros(n,np.float64) for name in
               ('bos_up','bos_down','choch_up','choch_down','sweep_up_event','sweep_down_event')}
        out['last_high_price'] = np.full(n,np.nan)
        out['last_low_price'] = np.full(n,np.nan)
        return out

    @staticmethod
    def _atr(frame):
        h,l,c = (frame[k] for k in ('high','low','close'));n = len(c)
        tr = [float(h[0]-l[0])]
        tr += [max(float(h[i]-l[i]),abs(float(h[i]-c[i-1])),abs(float(l[i]-c[i-1]))) for i in range(1,n)]
        values = np.full(n,np.nan)
        if n >= 14:
            values[13] = sum(tr[:14])/14
            for i in range(14,n):values[i] = (13*values[i-1]+tr[i])/14
        return values,np.where(np.isfinite(values),0.,np.nan)

    @classmethod
    def _run(cls,frame,*,raw=None,timeframe='M5',atr=None):
        a,e = cls._atr(frame) if atr is None else (np.asarray(atr,np.float64),np.where(np.isfinite(atr),0.,np.nan))
        return cls._owner().reference_native_core_local(frame,atr=a,atr_radius=e,
            raw_smc=cls._raw(len(a)) if raw is None else raw,timeframe=timeframe)

    @staticmethod
    def _rounded32_fasit(target, *, squared=False):
        # Prove the expected RN32 cell against exact rational midpoints. This
        # does not call the owner sqrt/wire helpers or choose an epsilon.
        import math
        from fractions import Fraction
        seed = math.sqrt(float(target)) if squared else float(target)
        candidate = np.float32(seed)
        left = Fraction.from_float(float(np.nextafter(candidate,np.float32(-np.inf))))
        center = Fraction.from_float(float(candidate))
        right = Fraction.from_float(float(np.nextafter(candidate,np.float32(np.inf))))
        lower,upper = (left+center)/2,(center+right)/2
        if squared:
            assert lower > 0 and lower*lower < target < upper*upper
        else:
            assert lower < target < upper  # These declared fasits are not ties.
        return center

    @classmethod
    def _rational(cls,v,e,row,num,den=1,*,wire32=False):
        from fractions import Fraction
        target = Fraction(num,den)
        if wire32:target = cls._rounded32_fasit(target)
        center = Fraction.from_float(float(v[row]));radius = Fraction.from_float(float(e[row]))
        assert radius >= 0 and center-radius <= target <= center+radius

    @classmethod
    def _square(cls,v,e,row,target):
        from fractions import Fraction
        expected = cls._rounded32_fasit(target,squared=True)
        center = Fraction.from_float(float(v[row]));radius = Fraction.from_float(float(e[row]))
        assert radius >= 0 and center-radius <= expected <= center+radius

    def test_active_twenty_seven_contract_warmups_and_price_quote_units(self):
        audit = self._owner();frame = self._frame(100+np.arange(240,dtype=np.float64))
        v,e,report = self._run(frame)
        assert tuple(v) == tuple(audit.NATIVE_CORE_LOCAL_FIELDS) and len(v) == 27
        assert len(audit.NATIVE_CORE_LOCAL_DISCRETE_FIELDS) == 7
        first = {'_v1_ema3_ema6_spread_atr':13,'close_distance_from_ema5_bps':4,
                 'spread_bps':0,'spread_bps_delta_1':1,'spread_extremes_sum_bps':0,
                 'quote_range_asymmetry_bps':0,'session_change_flag':0,
                 'mid_ema50_dist_atr':79,'ema5_20_spread_atr':19,'mid_ema200_dist_atr':219}
        for name,row in first.items():
            assert report['first_valid_rows'][name] == row
            assert np.isnan(v[name][:row]).all()
        expected = {'_v1_ema3_ema6_spread_atr':(3,8),'close_distance_from_ema5_bps':(20000,339),
                    'spread_bps':(20000,677),'spread_bps_delta_1':(-1600,18279),
                    'spread_extremes_sum_bps':(20000,339),'quote_range_asymmetry_bps':(0,1),
                    'mid_ema50_dist_atr':(49,8),'ema5_20_spread_atr':(15,8),'mid_ema200_dist_atr':(199,8)}
        for name,(num,den) in expected.items():
            self._rational(v[name],e[name],239,num,den,wire32=name in
                ('close_distance_from_ema5_bps','spread_bps_delta_1','spread_extremes_sum_bps','quote_range_asymmetry_bps'))
        assert np.isnan(v['atr_z']).all()  # ConstantTR => unavailable standard deviation.
        assert all(np.isnan(v[name]).all() for name in v if name.startswith('smc_sweep_'))
        assert np.isnan(v['chart.foundation_bos_up_event_age_bars']).all()
        assert np.all(v['smc_choch'] == 0)

    def test_atr_z_double_fifty_sample_variance_has_exact_squared_answer(self):
        from fractions import Fraction
        n = 140;close = np.full(n,100.);ranges = 2+np.arange(n,dtype=np.float64)
        frame = self._frame(close,high=close+ranges/2,low=close-ranges/2)
        v,e,report = self._run(frame)
        assert np.isnan(v['atr_z'][:98]).all() and report['first_valid_rows']['atr_z'] == 98
        # FirstSMA50 grows by1; second samplevariance=425/2, numerator49/2.
        for row in (98,139):self._square(v['atr_z'],e['atr_z'],row,Fraction(2401,850))
        assert report['ATR_z']['ddof'] == 1 and report['ATR_z']['full_prefix'] is True

    def test_atr_z_uniform_shortcut_retains_full_prefix_state_and_no_clip(self):
        from fractions import Fraction
        ranges = np.array([2.]*60+[4.]*100+[6.]*20);c = np.full(len(ranges),100.)
        v,e,_ = self._run(self._frame(c,high=c+ranges/2,low=c-ranges/2))
        assert np.isnan(v['atr_z'][158:160]).all()
        # At160:49oldATR50=4 andone4.04; samplevar1/31250.
        self._square(v['atr_z'],e['atr_z'],160,Fraction(2401,50))
        assert v['atr_z'][160] > 6

    def test_foundation_raw_ages_and_choch_or_have_no_common_pivot_warmup_mask(self):
        frame = self._frame(np.full(8,100.));raw = self._raw(8)
        raw['bos_up'][2] = 1;raw['bos_down'][5] = 1;raw['choch_up'][4] = 1
        v,e,_ = self._run(frame,raw=raw)
        expected = {'chart.foundation_bos_up_event_age_bars':[np.nan,np.nan,0,1,2,3,4,5],
                    'chart.foundation_bos_down_event_age_bars':[np.nan,np.nan,np.nan,np.nan,np.nan,0,1,2],
                    'chart.foundation_choch_event_age_bars':[np.nan,np.nan,np.nan,np.nan,0,1,2,3]}
        for name,wanted in expected.items():
            np.testing.assert_array_equal(v[name],wanted)
            assert np.all(e[name][np.isfinite(e[name])] == 0)
        np.testing.assert_array_equal(v['smc_choch'],[0,0,0,0,1,0,0,0])

    def test_both_sweep_anchors_have_independent_weighted_moment_fasits(self):
        from fractions import Fraction
        c = np.array([100.,102.,101.,103.,98.,105.,104.,103.,99.])
        weights = np.array([2,7,3,11,5,13,2,17,19],np.int64)
        frame = self._frame(c,volume=weights);raw = self._raw(9)
        raw['sweep_up_event'][[1,5]] = 1;raw['sweep_down_event'][[3,5]] = 1
        raw['last_high_price'][[1,5]] = c[[1,5]]+1
        raw['last_low_price'][[3,5]] = c[[3,5]]-1
        atr = np.array([np.nan,np.nan,2.,3.,4.,2.,1.,2.,3.])
        v,e,_ = self._run(frame,raw=raw,atr=atr)
        up = 'smc_sweep_up_avwap_';down = 'smc_sweep_down_avwap_'
        assert v[up+'age_bars'][1] == 0 and v[up+'mean_activity_ratio'][1] == 0
        assert np.isnan(v[up+'dist_atr'][1])
        assert all(np.isnan(v[down+suffix][:3]).all() for suffix in
                   ('age_bars','dist_atr','dispersion_atr','anchor_close_dist_atr','level_dist_atr','mean_activity_ratio'))
        for suffix,num,den in [('age_bars',2,1),('dist_atr',13,63),('anchor_close_dist_atr',1,3),
                               ('level_dist_atr',0,1),('mean_activity_ratio',0,1)]:
            self._rational(v[up+suffix],e[up+suffix],3,num,den,wire32=True)
        self._square(v[up+'dispersion_atr'],e[up+'dispersion_atr'],3,Fraction(230,3969))
        assert v[up+'age_bars'][3] == 2 and v[down+'age_bars'][3] == 0
        for prefix in (up,down):
            assert v[prefix+'age_bars'][5] == 0
            self._rational(v[prefix+'dist_atr'],e[prefix+'dist_atr'],6,-13,15,wire32=True)
            self._square(v[prefix+'dispersion_atr'],e[prefix+'dispersion_atr'],6,Fraction(26,225))
            self._rational(v[prefix+'mean_activity_ratio'],e[prefix+'mean_activity_ratio'],6,-11,26,wire32=True)
        self._rational(v[up+'level_dist_atr'],e[up+'level_dist_atr'],5,-1,2,wire32=True)
        self._rational(v[down+'level_dist_atr'],e[down+'level_dist_atr'],5,1,2,wire32=True)

    @pytest.mark.parametrize('timeframe,labels,expected',[
        ('M5',['2026-01-01T06:50:00Z','2026-01-01T06:55:00Z','2026-01-01T07:00:00Z'],[0,1,0]),
        ('H1',['2026-01-01T05:00:00Z','2026-01-01T06:00:00Z','2026-01-01T07:00:00Z'],[0,1,0]),
        ('D1',['2026-01-01T22:00:00Z','2026-01-02T22:00:00Z','2026-01-03T22:00:00Z'],[0,0,0]),
    ])
    def test_session_flag_uses_actual_closed_lane_time(self,timeframe,labels,expected):
        stamps = np.array([pd.Timestamp(t).value for t in labels],np.int64)
        v,e,_ = self._run(self._frame(np.full(3,100.),stamps=stamps),timeframe=timeframe)
        np.testing.assert_array_equal(v['session_change_flag'],expected)
        np.testing.assert_array_equal(e['session_change_flag'],[0,0,0])

    def test_calendar_gaps_and_appended_future_do_not_rewrite_local_prefix(self):
        frame = self._frame(100+np.arange(240,dtype=np.float64))
        gapped = {k:a.copy() for k,a in frame.items()};gapped['bar_start_ns'][120:] += 17*DAY
        v,e,_ = self._run(frame);g,ge,_ = self._run(gapped)
        future = self._frame(np.concatenate([frame['close'],[120.,80.,120.,80.,120.]]))
        later,le,_ = self._run(future)
        for name in v:
            np.testing.assert_array_equal(v[name],g[name]);np.testing.assert_array_equal(e[name],ge[name])
            np.testing.assert_array_equal(v[name],later[name][:240]);np.testing.assert_array_equal(e[name],le[name][:240])

    @pytest.mark.parametrize('fault',['value','mask','category'])
    def test_local_comparison_rejects_wrong_value_mask_and_discrete_category(self,fault):
        audit = self._owner();v,e,_ = self._run(self._frame(100+np.arange(240,dtype=np.float64)))
        actual = {k:a.astype(np.float32) for k,a in v.items()}
        audit.compare_technical_reference(v,actual,e,discrete_fields=audit.NATIVE_CORE_LOCAL_DISCRETE_FIELDS)
        if fault == 'value':actual['_v1_ema3_ema6_spread_atr'][-1] += 1
        elif fault == 'mask':actual['chart.foundation_bos_up_event_age_bars'][0] = 0
        else:actual['session_change_flag'][-1] = 2
        with pytest.raises(RuntimeError,match='TECHNICAL_NUMERIC_REFERENCE_'):
            audit.compare_technical_reference(v,actual,e,discrete_fields=audit.NATIVE_CORE_LOCAL_DISCRETE_FIELDS)


class TestNativeCoreLevelCross130:
    """Component-boundary hand witnesses; ATR wires are explicit test inputs."""
    @staticmethod
    def _owner():
        return TestTechnicalReference127._owner()
    @staticmethod
    def _frame(high,low,close,stamps=None):
        return TestTechnicalReference127._frame(close,high=high,low=low,stamps=stamps)
    @classmethod
    def _literal(cls):
        high,low,close=np.full(15,105.),np.full(15,95.),np.full(15,100.)
        high[[3,10]]=[110.,112.];low[[4,11]]=[90.,92.]
        return cls._frame(high,low,close)
    @classmethod
    def _extend(cls,frame,high,low,close):
        return cls._frame(np.r_[frame['high'],high],np.r_[frame['low'],low],np.r_[frame['close'],close])
    @classmethod
    def _level(cls,frame,expiry=100,first_atr=13):
        audit=cls._owner();a=np.ones(len(frame['close']),np.float64);a[:first_atr]=np.nan;e=np.where(np.isfinite(a),0.,np.nan)
        raw=audit.reference_native_smc_replay(frame['high'],frame['low'],frame['close'])
        return audit.reference_native_core_level(frame,atr=a,atr_radius=e,raw_smc=raw,timeframe='M5',expiry_bars=expiry)
    @staticmethod
    def _encloses(values,radii,row,numerator,denominator=1):
        TestTechnicalReference127._encloses(values,radii,row,numerator,denominator)
    @classmethod
    def _days(cls,days=60):
        origin=pd.Timestamp('2009-06-01T00:00:00Z').value
        stamps=np.array([origin+day*DAY+hour*60*MINUTE for day in range(days) for hour in (0,4,8,12)]
                        +[origin+(days-1)*DAY+22*60*MINUTE],np.int64)
        return cls._frame(np.full(len(stamps),110.),np.full(len(stamps),90.),np.full(len(stamps),100.),stamps)
    @classmethod
    def _cross(cls,raw,frame=None,timeframe='M5',a=None):
        if frame is None:frame={k:v[-1:] for k,v in raw.items()}
        if a is None:a=np.full(len(frame['close']),2.,np.float64)
        e=np.where(np.isfinite(a),0.,np.nan)
        return cls._owner().reference_native_core_cross(frame,atr=a,atr_radius=e,raw_m5=raw,timeframe=timeframe)

    def test_level_admission_waits_for_ATR_and_has_own_break_nan(self):
        v,e,proof=self._level(self._literal())
        assert len(v)==33 and len(self._owner().NATIVE_CORE_LEVEL_DISCRETE_FIELDS)==17
        assert proof['first_valid_row']==13 and proof['new_immutable_anchors']==2
        for name in v:assert np.isnan(v[name][:13]).all() and np.isnan(e[name][:13]).all()
        self._encloses(v['mtf_level_above_dist_atr'],e['mtf_level_above_dist_atr'],13,12)
        self._encloses(v['mtf_level_below_dist_atr'],e['mtf_level_below_dist_atr'],14,8)
        assert v['mtf_level_above_age_bars'][13]==0 and v['mtf_level_above_bars_since_touch'][13]==3
        assert v['mtf_level_below_present'][13]==0 and v['mtf_level_below_present'][14]==1
        assert np.isnan(v['mtf_level_bars_since_break_signed']).all()
        # SMC replay retained its earlier pivots; this registry deliberately did not.
        assert v['mtf_level_above_recurrence_dist_atr'][13]==0

    def test_recurrence_uses_all_immutable_birth_anchors_not_merged_levels(self):
        frame=self._literal();v,e,proof=self._level(frame,first_atr=0)
        assert proof['new_immutable_anchors']==4
        self._encloses(v['mtf_level_below_recurrence_dist_atr'],e['mtf_level_below_recurrence_dist_atr'],14,2)
        self._encloses(v['mtf_level_above_dist_atr'],e['mtf_level_above_dist_atr'],14,10)
        self._encloses(v['mtf_level_above2_dist_atr'],e['mtf_level_above2_dist_atr'],14,12)
        assert v['mtf_level_above2_present'][14]==1

    def test_fixed_expiry_is_inclusive_and_precedes_same_bar_touch(self):
        frame=self._extend(self._literal(),[105.,115.,105.],[95.,99.,95.],[100.,100.,100.])
        v,_,proof=self._level(frame,expiry=2)
        assert v['mtf_level_above_dist_atr'][15]==12  # birth13, age2 still eligible.
        assert v['mtf_level_above_dist_atr'][16]==0   # age3 expires before its touching range.
        assert v['mtf_level_below_present'][16]==1 and v['mtf_level_below_present'][17]==0
        assert proof['eligibility_expiries']==2
        assert np.isfinite(v['mtf_level_above_age_bars'][17]) and v['mtf_level_above_age_bars'][17]==0

    def test_reactions_complete_at_events_and_preserve_signed_last_mean_max(self):
        frame=self._extend(self._literal(),[120.,116.,111.,113.],[99.,111.,109.,108.],[115.,114.,109.,110.])
        v,e,proof=self._level(frame)
        assert v['mtf_level_break_up_event'][15]==1 and v['mtf_level_retest_hold_signed'][16]==1
        assert v['mtf_level_break_down_event'][17]==1 and v['mtf_level_retest_hold_signed'][18]==-1
        # High112: first reaction min(low14:15)=95 =>17; after role flip,
        # support112 breaks at17 with high111 =>-1. Mean8,max17,last-1.
        for key,expected in [('mean',8),('max',17),('last',-1)]:
            name='mtf_level_above_'+key+'_reaction_atr';self._encloses(v[name],e[name],18,expected)
        assert v['mtf_level_above_touch_count'][18]==3
        assert v['mtf_level_bars_since_break_signed'][18]==-1 and proof['held_retests']==2

    def test_equal_close_retest_fails_and_pending_excludes_its_break_bar(self):
        frame=self._extend(self._literal(),[120.,113.],[99.,111.],[115.,112.])
        v,_,proof=self._level(frame)
        assert v['mtf_level_above_pending_retest_dist_atr'][15]==0
        assert v['mtf_level_retest_fail_signed'][16]==1 and v['mtf_level_retest_hold_signed'][16]==0
        assert proof['failed_retests']==1

    @pytest.mark.parametrize('price,fifty,hundred',[(25.,-25.,25.),(50.,0.,-50.),(75.,-25.,-25.),(125.,-25.,25.)])
    def test_signed_round_grid_half_ties_choose_upper_grid(self,price,fifty,hundred):
        frame=self._frame(np.full(20,price+2),np.full(20,price-2),np.full(20,price))
        v,e,_=self._level(frame)
        for grid,target in [(50,fifty),(100,hundred)]:
            key='mtf_level_round_number_dist_'+str(grid)+'_atr';self._encloses(v[key],e[key],19,int(target))

    def test_newborn_same_bar_break_keeps_pending_episode_without_broken_middle_path(self):
        delta=2.**-42;h=np.full(10,100.);l=np.full(10,99.);c=np.full(10,99.5)
        h[6]=c[6]=100.+delta
        h[7:9]=99.;l[7:9]=[98.,97.];c[7:9]=[98.5,98.]
        h[9]=c[9]=100.+delta/2;l[9]=99.5
        v,e,_=self._level(self._frame(h,l,c),first_atr=0)
        assert v['mtf_level_break_up_event'][6]==1 and v['mtf_level_retest_hold_signed'][9]==1
        key='mtf_level_below_mean_reaction_atr'
        self._encloses(v[key],e[key],9,1,2)  # low7/8 are broken-state observations, excluded.

    def test_level_calendar_holes_and_future_append_preserve_prefix(self):
        frame=self._extend(self._literal(),[120.,116.,111.,113.],[99.,111.,109.,108.],[115.,114.,109.,110.])
        future=self._extend(frame,[150.,151.],[70.,71.],[120.,121.])
        before,eb,_=self._level(frame);after,ea,_=self._level(future)
        gapped={k:v.copy() for k,v in frame.items()};gapped['bar_start_ns'][10:]+=7*DAY
        holes,eh,_=self._level(gapped)
        for key in before:
            np.testing.assert_array_equal(before[key],after[key][:len(frame['close'])])
            np.testing.assert_array_equal(eb[key],ea[key][:len(frame['close'])])
            np.testing.assert_array_equal(before[key],holes[key]);np.testing.assert_array_equal(eb[key],eh[key])

    def test_thirteen_declared_constants_and_discrete_contract(self):
        audit=self._owner()
        assert audit.NATIVE_CORE_LEVEL_EXPIRY=={'M5':2721,'M15':1742,'H1':964,'H4':507,'D1':132}
        assert audit.NATIVE_CORE_CROSS_WINDOWS=={'M5':240,'M15':192,'H1':168,'H4':168,'D1':60}
        assert audit.NATIVE_CORE_ROUND_GRIDS==(50.,100.) and audit.NATIVE_CORE_SESSION_OFFSET_NS==22*60*MINUTE
        assert len(audit.NATIVE_CORE_LEVEL_FIELDS)==33 and len(audit.NATIVE_CORE_CROSS_FIELDS)==14
        assert len(set(audit.NATIVE_CORE_LEVEL_FIELDS)&set(audit.NATIVE_CORE_CROSS_FIELDS))==0

    def test_cross_four_prior_pivots_and_ten_extrema_have_hand_known_values(self):
        v,e,proof=self._cross(self._days())
        expected={'R1':-5,'R2':-10,'S1':5,'S2':10}
        for level,target in expected.items():
            name='dist_to_'+level+'_atr';self._encloses(v[name],e[name],0,target)
        for tf in ('m5','m15','h1','h4','d1'):
            for side in ('hi','lo'):
                name='dist_to_'+tf+'_'+side+'_atr';self._encloses(v[name],e[name],0,5)
        origin=pd.Timestamp('2009-06-01T00:00:00Z').value
        assert proof['reference_first_labels_ns']['D1']==origin-2*60*MINUTE
        assert proof['reference_rows']['D1']==61 and proof['leading_partial_reference_bins_retained'] is True

    def test_cross_all_ten_extrema_wait_for_full_D1_population_but_pivots_do_not(self):
        v,_,proof=self._cross(self._days(59))
        for name in self._owner().NATIVE_CORE_CROSS_FIELDS:
            if name.startswith(('dist_to_R','dist_to_S')):assert np.isfinite(v[name][0])
            else:assert np.isnan(v[name][0]) and proof['first_valid_rows'][name] is None

    @pytest.mark.parametrize('timeframe',['M5','M15','H1','H4','D1'])
    def test_cross_uses_each_lanes_actual_close_and_fixed_UTC_phase(self,timeframe):
        raw=self._days(61);audit=self._owner();full=audit.native_resample_calendar_frame(raw,timeframe)
        frame={k:v[-1:] for k,v in full.items()};v,e,proof=self._cross(raw,frame,timeframe)
        assert proof['timeframe']==timeframe and proof['session_offset_hours_UTC']==22
        for tf in ('m5','m15','h1','h4','d1'):
            self._encloses(v['dist_to_'+tf+'_hi_atr'],e['dist_to_'+tf+'_hi_atr'],0,5)

    def test_pivot_reference_changes_at_UTC22_close_not_midnight(self):
        raw=self._days();origin=raw['bar_start_ns'][0];day58=origin+58*DAY;day59=origin+59*DAY
        selected=(raw['bar_start_ns']>=day58)&(raw['bar_start_ns']<day59)
        raw['high'][selected]=130.;raw['low'][selected]=70.
        stamps=np.r_[raw['bar_start_ns'][:-1],day59+(21*60+50)*MINUTE,day59+(21*60+55)*MINUTE,raw['bar_start_ns'][-1]]
        raw=self._frame(np.r_[raw['high'][:-1],110.,110.,110.],np.r_[raw['low'][:-1],90.,90.,90.],np.full(len(stamps),100.),stamps)
        frame={k:v[-3:] for k,v in raw.items()};v,e,_=self._cross(raw,frame)
        for row,target in enumerate([-15,-5,-5]):self._encloses(v['dist_to_R1_atr'],e['dist_to_R1_atr'],row,target)

    def test_extrema_fallback_is_signed_and_excludes_forming_D1_bar(self):
        raw=self._days();raw['high'][:]=99.;raw['low'][:]=90.;raw['close'][:]=raw['open'][:]=95.
        raw['high'][-1]=raw['close'][-1]=raw['open'][-1]=100.;raw['low'][-1]=99.
        v,e,_=self._cross(raw)
        self._encloses(v['dist_to_d1_hi_atr'],e['dist_to_d1_hi_atr'],0,-1,2)
        assert v['dist_to_m5_hi_atr'][0]==0 and v['dist_to_d1_hi_atr'][0]<0

    def test_cross_internal_float32_bps_wire_is_not_silently_replaced_by_raw_ATR(self):
        import struct
        raw=self._days();a=np.array([1.00000003],np.float64);v,_,proof=self._cross(raw,a=a)
        represented=struct.unpack('f',struct.pack('f',float(a[0])/100.*1e4))[0]/1e4*100.
        assert v['dist_to_d1_hi_atr'][0]==10./represented
        assert v['dist_to_d1_hi_atr'][0]!=10./a[0] and proof['hidden_cache_inputs'] is False

    @pytest.mark.parametrize('fault',['age','continuous','mask'])
    def test_frozen_comparator_rejects_level_value_or_observation_changes(self,fault):
        frame=self._literal();v,e,_=self._level(frame);actual={k:x.astype(np.float32) for k,x in v.items()}
        if fault=='age':actual['mtf_level_above_age_bars'][-1]+=.25
        elif fault=='continuous':actual['mtf_level_above_dist_atr'][-1]+=1.
        else:actual['mtf_level_above_dist_atr'][0]=0.
        with pytest.raises(RuntimeError,match='TECHNICAL_NUMERIC_REFERENCE_'):
            self._owner().compare_technical_reference(v,actual,e,discrete_fields=self._owner().NATIVE_CORE_LEVEL_DISCRETE_FIELDS)
