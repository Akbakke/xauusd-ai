"""Synthetic causal M1 input, view binding and model learning-contact checks."""
from copy import deepcopy
import json
import numpy as np
import pytest
import torch

from gx1.contracts import entry_closed_m1_ofi_window_v1 as windows
from gx1.contracts import entry_native_direction_m1_ofi_normalization_v1 as normalizer
from gx1.contracts import entry_native_direction_dense_v1 as dense
from gx1.contracts import entry_native_direction_feature_view_v1 as view_owner
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
from tests.test_entry_closed_m1_ofi_window_v1 import _sample, _freeze, _clock
from tests.test_entry_native_direction_feature_view_v1 import bound_case, _inputs, _ref, _constructor_spy
from tests import test_entry_native_direction_research_v1 as native_fixture


def _stats():
    source=windows.qualify_source(_freeze(_sample()))
    clock=_clock()
    return normalizer.fit(source,clock,prefix_rows=2,cutoff_ns=15*windows.MINUTE_NS)


def test_only_observed_strict_train_minutes_fit_not_later_or_masked_zeros():
    base=_sample();base['ofi_valid'][7]=False
    fitted=normalizer.fit(windows.qualify_source(_freeze(base)),_clock(),prefix_rows=2,cutoff_ns=15*windows.MINUTE_NS)
    assert fitted['observed_minute_count']==9
    assert fitted['surface']['center'][0]==4.0  # sorted observed 0,1,2,3,4,5,6,8,9
    later=_sample();later['ofi_valid'][7]=False;later['ofi_contracts'][10:]+=1e6
    other=normalizer.fit(windows.qualify_source(_freeze(later)),_clock(),prefix_rows=2,cutoff_ns=15*windows.MINUTE_NS)
    assert other['surface']==fitted['surface'] and other['source_sha256']!=fitted['source_sha256']
    with pytest.raises(RuntimeError,match='STRICT_PREFIX_CUTOFF'):
        normalizer.fit(windows.qualify_source(_freeze(_sample())),_clock(),prefix_rows=3,cutoff_ns=15*windows.MINUTE_NS)
    damaged=deepcopy(fitted);damaged['surface']['center'][0]+=1
    with pytest.raises(RuntimeError,match='CONTRACT_SEAL'):normalizer.require(damaged)


def test_m1_branch_preserves_initial_rng_and_logits_then_gets_gradients(monkeypatch):
    monkeypatch.setattr(EntryV10CtxHybridTransformer,'forward',native_fixture._native_forward_spy)
    torch.manual_seed(812);old=dense.DenseNativeEntryDirectionResearch(native_fixture._native_shell());rng=torch.get_rng_state().clone()
    torch.manual_seed(812);new=dense.DenseNativeEntryDirectionResearch(native_fixture._native_shell(),m1_ofi_normalization=_stats())
    assert torch.equal(rng,torch.get_rng_state())
    x=torch.tensor([[0.,1.,2.,3.,4.],[5.,6.,7.,8.,9.]],dtype=torch.float64)
    mask=torch.tensor([[1,1,1,1,1],[1,1,1,1,1]],dtype=torch.bool)
    original=old(**native_fixture._inputs())['logit_up']
    initial=new(**native_fixture._inputs(),m1_ofi_contracts=x,m1_ofi_observed=mask)['logit_up']
    assert torch.equal(original,initial)
    initial.square().mean().backward()
    assert new.m1_ofi_out.weight.grad is not None and torch.count_nonzero(new.m1_ofi_out.weight.grad)>0
    with torch.no_grad():new.m1_ofi_out.weight.add_(new.m1_ofi_out.weight.grad,alpha=-0.5)
    changed=x.clone();changed[0,0]+=10
    after=new(**native_fixture._inputs(),m1_ofi_contracts=x,m1_ofi_observed=mask)['logit_up']
    perturbed=new(**native_fixture._inputs(),m1_ofi_contracts=changed,m1_ofi_observed=mask)['logit_up']
    assert not torch.equal(after[0],perturbed[0]) and torch.equal(after[1],perturbed[1])
    # A completely missing window must have no effect even after an update.
    missing=torch.zeros_like(mask);zeros=torch.zeros_like(x)
    assert torch.equal(original,new(**native_fixture._inputs(),m1_ofi_contracts=zeros,m1_ofi_observed=missing)['logit_up'])
    for broken in (x.float(), torch.full_like(x,float('nan'))):
        with pytest.raises(RuntimeError,match='M1_INPUT'):new(**native_fixture._inputs(),m1_ofi_contracts=broken,m1_ofi_observed=mask)
    with pytest.raises(RuntimeError,match='UNBOUND_M1_INPUT'):
        old(**native_fixture._inputs(),m1_ofi_contracts=x,m1_ofi_observed=mask)
    with pytest.raises(RuntimeError,match='M1_INPUT'):
        new(**native_fixture._inputs(),m1_ofi_contracts=x,m1_ofi_observed=missing)


def _bound_minutes(case):
    parents=case['parent_rows'].copy();decisions=case['decision_ns'].copy();contracts=np.ones(len(parents),np.int16)
    t=np.unique((decisions[:,None]-np.arange(5,0,-1,dtype=np.int64)[None,:]*windows.MINUTE_NS).ravel())
    data={'bar_start_ns':t,'available_ns':t+windows.MINUTE_NS,'contract_index':np.ones(len(t),np.int16),
        'assigned':np.ones(len(t),bool),'ofi_contracts':np.arange(len(t),dtype=np.float64),
        'ofi_valid':np.ones(len(t),bool),'receive_clock_excluded':np.zeros(len(t),bool),
        'book_event_state_excluded':np.zeros(len(t),bool)}
    data['ofi_valid'][1]=False
    from gx1.scripts import audit_raw_gc_m1_reference_v1 as roll_owner
    roll={'origin_ns':roll_owner._MASK134_ORIGIN,'availability_offset_ns':windows.MINUTE_NS,
        'contract_order':['unused','GCM4'],'selected_daily':[1]*791}
    model_available=decisions.copy();native=np.arange(len(parents),dtype=np.int64)
    cache=decisions-5*windows.MINUTE_NS;model=np.tile(np.array([[1.,1.]],dtype=np.float32),(len(parents),1))
    for v in (parents,decisions,contracts,model_available,native,cache,model):v.setflags(write=False)
    source=windows.qualify_source(_freeze(data))
    clock=windows.qualify_model_clock(source,parent_rows=parents,decision_ns=decisions,
        roll_bytes=json.dumps(roll,sort_keys=True).encode(),m5_available_ns=model_available,
        m5_native_row_indices=native,m5_features=model,m5_field_names=['gc_m5_observed_book_contract_valid',
        'gc_m5_observed_book_contract_valid_observed'],m5_cache_start_ns=cache)
    return source,clock


def test_combined_price_and_m1_view_source_and_decision_bound(bound_case,monkeypatch):
    bound=deepcopy(bound_case);args=bound['arguments'];case=bound['case'];source,clock=_bound_minutes(case)
    fitted=normalizer.fit(source,clock,prefix_rows=4,cutoff_ns=case['cutoff_ns'])
    for names in (view_owner.DENSE_SOURCE_REF_KEYS,view_owner.PRICE_MOMENTUM_SOURCE_REF_KEYS,view_owner.M1_OFI_SOURCE_REF_KEYS):
        args['source_bindings'].update({key:_ref(key+'.py') for key in names})
    args['source_bindings']['m1_ofi_roll_schedule']={'path':'/synthetic/roll.json','sha256':clock.roll_sha256}
    view=view_owner.bind_native_direction_feature_view(**args,dense_minutes=True,price_momentum=True,m1_ofi_normalization=fitted)
    assert view['m1_ofi_normalization']['contract_sha256']==fitted['contract_sha256']
    unbound=windows.qualify_clock(clock.parent_rows,clock.decision_ns,clock.current_m5_contract_index)
    unbound_stats=normalizer.fit(source,unbound,prefix_rows=4,cutoff_ns=case['cutoff_ns'])
    with pytest.raises(RuntimeError,match='M1_OFI_ROLL_JOIN'):
        view_owner.bind_native_direction_feature_view(**args,dense_minutes=True,price_momentum=True,m1_ofi_normalization=unbound_stats)
    foreign_parents=clock.parent_rows.copy();foreign_parents[0]+=1;foreign_parents.setflags(write=False)
    foreign_source,foreign=_bound_minutes({**case,'parent_rows':foreign_parents})
    foreign_stats=normalizer.fit(foreign_source,foreign,prefix_rows=4,cutoff_ns=case['cutoff_ns'])
    bad_roll=deepcopy(args);bad_roll['source_bindings']['m1_ofi_roll_schedule']['sha256']='0'*64
    with pytest.raises(RuntimeError,match='M1_OFI_ROLL_JOIN'):
        view_owner.bind_native_direction_feature_view(**bad_roll,dense_minutes=True,price_momentum=True,m1_ofi_normalization=fitted)
    with pytest.raises(RuntimeError,match='M1_OFI_AXIS_JOIN'):
        view_owner.bind_native_direction_feature_view(**args,dense_minutes=True,price_momentum=True,m1_ofi_normalization=foreign_stats)
    inputs=_inputs(bound)
    gathered=view_owner.gather_native_direction_feature_batch(view,**inputs,m1_ofi_source=source,m1_ofi_clock=clock)
    expected=windows.gather(source,clock,batch_parent_rows=inputs['batch_parent_rows'])
    assert np.array_equal(gathered['features']['m1_ofi_contracts'],expected['ofi_contracts'])
    assert np.array_equal(gathered['features']['m1_ofi_observed'],expected['ofi_observed'])
    calls,_=_constructor_spy(monkeypatch)
    model=view_owner.build_native_direction_research_model(view,prefix_normalization=args['prefix_normalization'],
        constructor_metadata=args['constructor_metadata'],device=torch.device('cpu'))
    assert len(calls)==1 and model._m1_normalization_json is not None
    changed=_sample();changed['ofi_contracts'][0]=999
    with pytest.raises(RuntimeError,match='M1_OFI_SOURCE_IDENTITY'):
        view_owner.gather_native_direction_feature_batch(view,**inputs,m1_ofi_source=windows.qualify_source(_freeze(changed)),m1_ofi_clock=clock)
    with pytest.raises(RuntimeError,match='M1_OFI_SOURCE_IDENTITY'):
        view_owner.gather_native_direction_feature_batch(view,**inputs)


def test_actual_native_backbone_with_compact_price_and_m1_together_gets_learning_contact(bound_history,candidate_case):
    from gx1.features import entry_specialist_feature_groups_v1 as groups
    from gx1.features.entry_direction_momentum_profile_v1 import compile_price_momentum_profile, MOMENTUM_FAMILY
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EXACT_SPECIALIST_NAMES
    from tests.test_entry_gc_mtf_model_v1 import _reference
    from tests.test_entry_v10_ctx_model_shapes import _make_dynamic_m5_model
    history=bound_history[0];_profile,_model,inputs,normalization=candidate_case
    profile=compile_price_momentum_profile(history);physical=profile['physical_field_names']
    signal=list(physical['signal']);mtf=list(physical['mtf'])
    aliases=groups.model_native_context_temporal_alias_policy(signal)['aliases']
    grouped=groups.group_features_by_specialist(signal)
    backbone=_make_dynamic_m5_model(seq_input_dim=len(signal),snap_input_dim=len(signal),
        specialist_input_indices={f:[i for i,n in enumerate(signal) if n in set(grouped[f])] for f in EXACT_SPECIALIST_NAMES},
        **{tf.lower()+'_seq_dim':len(mtf) for tf in history['layout']['timeframes']},
        **{tf.lower()+'_seq_len':history['layout']['window_rows'][tf] for tf in history['layout']['timeframes']},
        multi_tf_specialist_input_indices={f:list(v) for f,v in groups.require_multi_tf_specialist_routing_v4(mtf).items()},
        temporal_alias_signal_indices=[int(x['signal_index']) for x in aliases],
        temporal_alias_ctx_cont_indices=[int(x['ctx_cont_index']) for x in aliases],
        input_normalization=normalization,gc_mtf_history=history,dropout=0.0,entry_execution_profile=_reference(profile)).eval()
    model=dense.DenseNativeEntryDirectionResearch(backbone,m1_ofi_normalization=_stats()).eval()
    source=windows.qualify_source(_freeze(_sample()));minutes=windows.gather(source,_clock(),batch_parent_rows=np.array([3],np.int64))
    kwargs={'m1_ofi_contracts':torch.from_numpy(minutes['ofi_contracts'].copy()),
        'm1_ofi_observed':torch.from_numpy(minutes['ofi_observed'].copy())}
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.scripts.run_entry_native_direction_prefix_fit_window_v1 import _forward
    gates, feature_gates=trainer._dynamic_m5_gate_execution_kwargs(backbone)
    assert feature_gates['gate_shape']==(5,182)
    assert gates['gate_widths']['family_tf_cooperation_gate']==5*len(backbone._mtf_specialist_names)
    counters={'forward_calls':0,'forward_kind_calls':{},'forward_returns':0}
    output=_forward(model,{**inputs,**kwargs},kind='SYNTHETIC',counters=counters,ceiling=1,torch=torch,grad=True)
    assert output.shape==(1,60) and torch.isfinite(output).all() and counters['forward_returns']==1
    output.square().mean().backward()
    for gradient in (model.m1_ofi_out.weight.grad,backbone.mtf_family_proj[MOMENTUM_FAMILY].weight.grad):
        assert gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient)>0


from tests.test_entry_gc_mtf_model_v1 import bound_history, candidate_case


def test_exact_native_research_constructor_does_not_promote_profile(bound_case):
    from gx1.contracts.entry_exit_production_architecture_v1 import entry_exit_production_architecture_contract
    bound=deepcopy(bound_case);args=bound['arguments'];source,clock=_bound_minutes(bound['case'])
    fitted=normalizer.fit(source,clock,prefix_rows=4,cutoff_ns=bound['case']['cutoff_ns'])
    for names in (view_owner.DENSE_SOURCE_REF_KEYS,view_owner.PRICE_MOMENTUM_SOURCE_REF_KEYS,view_owner.M1_OFI_SOURCE_REF_KEYS):
        args['source_bindings'].update({key:_ref(key+'.py') for key in names})
    args['source_bindings']['m1_ofi_roll_schedule']={'path':'/synthetic/roll.json','sha256':clock.roll_sha256}
    view=view_owner.bind_native_direction_feature_view(**args,dense_minutes=True,price_momentum=True,m1_ofi_normalization=fitted)
    model=view_owner.build_native_direction_research_model(view,prefix_normalization=args['prefix_normalization'],
        constructor_metadata=args['constructor_metadata'],device=torch.device('cpu'))
    assert type(model.backbone) is EntryV10CtxHybridTransformer
    assert model.backbone.entry_execution_profile==view['entry_execution_profile']
    with pytest.raises(RuntimeError):
        entry_exit_production_architecture_contract(entry_execution_profile=view['entry_execution_profile'],
            gc_mtf_history=args['prefix_normalization']['gc_mtf_history'])
