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


def test_coarsened_m1_control_keeps_capacity_masks_and_sum_but_removes_order(monkeypatch):
    monkeypatch.setattr(EntryV10CtxHybridTransformer,'forward',native_fixture._native_forward_spy)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(812)
        ordered=dense.DenseNativeEntryDirectionResearch(native_fixture._native_shell(),m1_ofi_normalization=_stats()).eval()
        torch.manual_seed(812)
        coarse=dense.DenseNativeEntryDirectionResearch(native_fixture._native_shell(),m1_ofi_normalization=_stats(),m1_ofi_coarsened=True).eval()
    left,right=ordered.state_dict(),coarse.state_dict()
    assert left.keys()==right.keys()
    assert all(torch.equal(left[k],right[k]) for k in left if k!='m1_ofi_coarsened')
    assert left['m1_ofi_coarsened'].item()==0 and right['m1_ofi_coarsened'].item()==1
    first=torch.tensor([[0.,1.,2.,3.,4.],[0.,3.,0.,7.,0.]],dtype=torch.float64)
    second=torch.tensor([[4.,3.,2.,1.,0.],[0.,7.,0.,3.,0.]],dtype=torch.float64)
    mask=torch.tensor([[True]*5,[False,True,False,True,False]])
    assert torch.equal(first.sum(dim=1),second.sum(dim=1))
    for model in (ordered,coarse):
        with torch.no_grad():
            model.m1_ofi_in.weight.zero_();model.m1_ofi_in.bias.zero_()
            model.m1_ofi_in.weight[0,model._bound_hidden_width]=1.
            model.m1_ofi_out.weight.zero_();model.m1_ofi_out.bias.zero_();model.m1_ofi_out.weight[0,0]=1.
    def predict(model,values,observed=mask):
        return model(**native_fixture._inputs(),m1_ofi_contracts=values,m1_ofi_observed=observed)['logit_up']
    assert torch.equal(predict(coarse,first),predict(coarse,second))
    assert not torch.equal(predict(ordered,first)[0],predict(ordered,second)[0])
    missing=torch.zeros_like(mask);zeros=torch.zeros_like(first)
    assert torch.isfinite(predict(coarse,zeros,missing)).all()
    with pytest.raises(RuntimeError,match='M1_REPRESENTATION'):
        dense.DenseNativeEntryDirectionResearch(native_fixture._native_shell(),m1_ofi_coarsened=True)
    with torch.no_grad():coarse.m1_ofi_coarsened.zero_()
    with pytest.raises(RuntimeError,match='M1_REPRESENTATION'):
        predict(coarse,first)


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
    coarse=view_owner.bind_native_direction_feature_view(**args,dense_minutes=True,price_momentum=True,
        m1_ofi_normalization=fitted,m1_ofi_coarsened=True)
    assert coarse['m1_ofi_coarsened'] is True and coarse['contract_sha256']!=view['contract_sha256']
    with pytest.raises(RuntimeError,match='OBJECTIVE_MODE'):
        view_owner.bind_native_direction_feature_view(**args,dense_minutes=True,price_momentum=True,m1_ofi_coarsened=True)
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
    coarse_raw=view_owner.gather_native_direction_feature_batch(coarse,**inputs,m1_ofi_source=source,m1_ofi_clock=clock)
    for name in gathered['features']:assert np.array_equal(gathered['features'][name],coarse_raw['features'][name])
    calls,_=_constructor_spy(monkeypatch)
    model=view_owner.build_native_direction_research_model(view,prefix_normalization=args['prefix_normalization'],
        constructor_metadata=args['constructor_metadata'],device=torch.device('cpu'))
    assert len(calls)==1 and model._m1_normalization_json is not None
    changed=_sample();changed['ofi_contracts'][0]=999
    with pytest.raises(RuntimeError,match='M1_OFI_SOURCE_IDENTITY'):
        view_owner.gather_native_direction_feature_batch(view,**inputs,m1_ofi_source=windows.qualify_source(_freeze(changed)),m1_ofi_clock=clock)
    with pytest.raises(RuntimeError,match='M1_OFI_SOURCE_IDENTITY'):
        view_owner.gather_native_direction_feature_batch(view,**inputs)


def test_actual_native_backbone_with_compact_price_and_m1_together_gets_learning_contact(bound_history,candidate_case,monkeypatch):
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
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260911)
        price_only=dense.DenseNativeEntryDirectionResearch(deepcopy(backbone)).eval()
        torch.manual_seed(20260911)
        coarse_model=dense.DenseNativeEntryDirectionResearch(deepcopy(backbone),m1_ofi_normalization=_stats(),m1_ofi_coarsened=True).eval()
        torch.manual_seed(20260911)
        model=dense.DenseNativeEntryDirectionResearch(backbone,m1_ofi_normalization=_stats()).eval()
    only_state=price_only.state_dict();m1_state=model.state_dict();coarse_state=coarse_model.state_dict()
    assert set(only_state).issubset(m1_state) and all(torch.equal(value,m1_state[name]) for name,value in only_state.items())
    assert coarse_state.keys()==m1_state.keys()
    assert all(torch.equal(value,m1_state[name]) for name,value in coarse_state.items() if name!='m1_ofi_coarsened')
    source=windows.qualify_source(_freeze(_sample()));minutes=windows.gather(source,_clock(),batch_parent_rows=np.array([3],np.int64))
    kwargs={'m1_ofi_contracts':torch.from_numpy(minutes['ofi_contracts'].copy()),
        'm1_ofi_observed':torch.from_numpy(minutes['ofi_observed'].copy())}
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.scripts.run_entry_native_direction_prefix_fit_window_v1 import _forward
    gates, feature_gates=trainer._dynamic_m5_gate_execution_kwargs(backbone)
    assert feature_gates['gate_shape']==(5,182)
    assert gates['gate_widths']['family_tf_cooperation_gate']==5*len(backbone._mtf_specialist_names)
    # Join the real strict context reporter, device conversion, native forward
    # and dense scorer on the same synthetic batch. The qualification worker's
    # earlier control-flow test mocks these boundaries independently.
    from gx1.contracts import entry_native_direction_context_report_v1 as reporter
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as runner
    from gx1.scripts.run_unified_exit_random_access_full_train_v1 import _dynamic_m5_technical_tree_sha256 as digest
    from tests.test_entry_native_direction_context_report_v1 import _case
    _,_,witness=_case(1)
    context_map=reporter.native_direction_context_map(gc_layout=history['layout'],mtf_field_names=physical['mtf'])
    raw={'features':{name:value.detach().cpu().numpy().copy() for name,value in {**inputs,**kwargs}.items()},'witness':witness}
    selected=np.asarray(witness['parent_rows'],np.int64);decisions=np.asarray(witness['decision_ns'],np.int64)
    target={'decision_ns':decisions,'terminal_time_ns':decisions[:,None]+np.arange(1,61,dtype=np.int64)[None,:]*runner.MINUTE_NS,
        'observed':np.ones((1,60),np.bool_),'sign':np.ones((1,60),np.int8)}
    monkeypatch.setattr(runner,'_batch',lambda *args:(raw,target))
    evaluation={'forward_calls':0,'forward_kind_calls':{},'forward_returns':0}
    measured,count=runner._evaluate(model,{}, {},{'dense_minute_supervision':True,'m1_ofi_normalization':_stats()},
        context_map,selected,cutoff=int(decisions[-1]+120*runner.MINUTE_NS),kind='SYNTHETIC_INITIAL',counters=evaluation,
        ceiling=2,device=torch.device('cpu'),torch=torch,np=np,tree_sha=digest,parity=True)
    assert count==1 and measured['initial_parity_passed'] and evaluation['forward_returns']==2
    assert measured['arrays']['logit_up'].shape==(1,60) and np.isfinite(measured['arrays']['logit_up']).all()
    reference_counts={'forward_calls':0,'forward_kind_calls':{},'forward_returns':0}
    reference=_forward(price_only,inputs,kind='SYNTHETIC_MATCHED_INITIAL',counters=reference_counts,ceiling=1,torch=torch)
    assert np.array_equal(reference.detach().cpu().numpy(),measured['arrays']['logit_up'])
    coarse_counts={'forward_calls':0,'forward_kind_calls':{},'forward_returns':0}
    coarse_initial=_forward(coarse_model,{**inputs,**kwargs},kind='SYNTHETIC_COARSE_INITIAL',counters=coarse_counts,ceiling=1,torch=torch)
    assert np.array_equal(coarse_initial.detach().cpu().numpy(),measured['arrays']['logit_up'])
    assert all(p.grad is None for p in model.parameters())
    counters={'forward_calls':0,'forward_kind_calls':{},'forward_returns':0}
    output=_forward(model,{**inputs,**kwargs},kind='SYNTHETIC',counters=counters,ceiling=1,torch=torch,grad=True)
    assert output.shape==(1,60) and torch.isfinite(output).all() and counters['forward_returns']==1
    output.square().mean().backward()
    for gradient in (model.m1_ofi_out.weight.grad,backbone.mtf_family_proj[MOMENTUM_FAMILY].weight.grad):
        assert gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient)>0
    from gx1.scripts.run_entry_native_direction_research_window_v1 import _candidate_gradient_counts
    evidence=_candidate_gradient_counts(model,torch)
    for field in ('rsi14_centered','macd_line_atr','macd_hist_atr','ret_1'):
        assert 'price_momentum_'+field+'_projection_nonzero' in evidence
        for tf in ('m5','m15','h1','h4','d1'):
            assert 'price_momentum_'+tf+'_'+field+'_gate_nonzero' in evidence
    for minute in range(1,6):
        assert evidence['m1_ofi_value_minute_'+str(minute)+'_weight_nonzero']==0
        assert evidence['m1_ofi_mask_minute_'+str(minute)+'_weight_nonzero']==0
    with torch.no_grad():
        model.m1_ofi_out.weight.add_(model.m1_ofi_out.weight.grad,alpha=-.01)
        backbone.mtf_family_proj[MOMENTUM_FAMILY].weight.add_(
            backbone.mtf_family_proj[MOMENTUM_FAMILY].weight.grad,alpha=-.01)
    model.zero_grad(set_to_none=True)
    again=_forward(model,{**inputs,**kwargs},kind='SYNTHETIC',counters=counters,ceiling=2,torch=torch,grad=True)
    again.square().mean().backward();after=_candidate_gradient_counts(model,torch)
    width=model._bound_hidden_width;grad=model.m1_ofi_in.weight.grad
    assert after['m1_ofi_in_weight_nonzero']>0
    for minute in range(5):
        assert after['m1_ofi_value_minute_'+str(minute+1)+'_weight_nonzero']==int(torch.count_nonzero(grad[:,width+minute]))
        assert after['m1_ofi_mask_minute_'+str(minute+1)+'_weight_nonzero']==int(torch.count_nonzero(grad[:,width+5+minute]))
    routed=backbone._multi_tf_specialist_index_tuples[MOMENTUM_FAMILY]
    names=backbone._input_normalization_field_names['mtf_m5']
    projection=backbone.mtf_family_proj[MOMENTUM_FAMILY].weight.grad
    for position,index in enumerate(routed):
        field=names[index]
        assert after['price_momentum_'+field+'_projection_nonzero']==int(torch.count_nonzero(projection[:,position]))
        for tf in ('m5','m15','h1','h4','d1'):
            gate=backbone.mtf_feature_context_gate[tf+'__'+MOMENTUM_FAMILY]
            expected=(int(torch.count_nonzero(gate.weight.grad[position])) if gate.weight.grad is not None else 0)
            expected+=(int(gate.bias.grad[position]!=0) if gate.bias.grad is not None else 0)
            assert after['price_momentum_'+tf+'_'+field+'_gate_nonzero']==expected


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


def test_private_qualification_reports_m1_and_price_gradients_without_inventing_liveness():
    from types import SimpleNamespace
    from torch import nn
    from torch.nn import functional as F
    from gx1.features.entry_direction_momentum_profile_v1 import PROFILE_NAME, MOMENTUM_FAMILY
    from gx1.scripts.run_entry_native_direction_research_window_v1 import _candidate_gradient_counts
    torch.manual_seed(142)
    m1in=nn.Linear(12,2);m1out=nn.Linear(2,2);price=nn.Linear(4,2)
    with torch.no_grad():
        m1out.weight.zero_();m1out.bias.zero_();price.weight.zero_();price.bias.zero_()
    fields=['macd_hist_atr','other','ret_1','rsi14_centered','macd_line_atr']
    indices=(2,0,4,3)  # Deliberately different from the canonical feature order.
    gates=nn.ModuleDict({tf+'__'+MOMENTUM_FAMILY:nn.Linear(2,4) for tf in ('m5','m15','h1','h4','d1')})
    model=SimpleNamespace(m1_ofi_in=m1in,m1_ofi_out=m1out,_bound_hidden_width=2,
        backbone=SimpleNamespace(entry_execution_profile={'name':PROFILE_NAME},
            mtf_family_proj=nn.ModuleDict({MOMENTUM_FAMILY:price}),mtf_feature_context_gate=gates,
            _multi_tf_specialist_index_tuples={MOMENTUM_FAMILY:indices},
            _input_normalization_field_names={'mtf_'+tf:fields for tf in ('m5','m15','h1','h4','d1')}))
    x=torch.tensor([[0.,1.,1.,2.,3.,4.,5.,1.,1.,1.,1.,1.],[1.,2.,6.,7.,8.,9.,10.,1.,1.,1.,1.,1.]])
    price_x=torch.tensor([[1.,2.,3.,4.],[3.,-1.,0.,2.]])
    loss=(m1out(F.gelu(m1in(x)))+price(price_x)-1).square().mean();loss.backward()
    first=_candidate_gradient_counts(model,torch)
    assert first['m1_ofi_out_weight_nonzero']>0 and first['m1_ofi_in_weight_nonzero']==0
    assert first['price_momentum_weight_nonzero']>0
    assert all(first['m1_ofi_value_minute_'+str(i)+'_weight_nonzero']==0 for i in range(1,6))
    price.weight.grad=torch.tensor([[0.,1.,1.,0.],[0.,1.,0.,1.]])
    m15=gates['m15__'+MOMENTUM_FAMILY];m15.weight.grad=torch.zeros_like(m15.weight)
    m15.bias.grad=torch.zeros_like(m15.bias);m15.weight.grad[0,:]=1
    mapped=_candidate_gradient_counts(model,torch)
    assert mapped['price_momentum_ret_1_projection_nonzero']==0
    assert mapped['price_momentum_macd_hist_atr_projection_nonzero']==2
    assert mapped['price_momentum_macd_line_atr_projection_nonzero']==1
    assert mapped['price_momentum_rsi14_centered_projection_nonzero']==1
    assert mapped['price_momentum_m15_ret_1_gate_nonzero']==2
    assert mapped['price_momentum_h1_ret_1_gate_nonzero']==0
    with torch.no_grad():m1out.weight.add_(m1out.weight.grad,alpha=-.1)
    for layer in (m1in,m1out,price):layer.zero_grad(set_to_none=True)
    (m1out(F.gelu(m1in(x)))+price(price_x)-1).square().mean().backward()
    second=_candidate_gradient_counts(model,torch)
    assert second['m1_ofi_in_weight_nonzero']>0
    for minute in range(1,6):
        assert second['m1_ofi_value_minute_'+str(minute)+'_weight_nonzero']==int(torch.count_nonzero(m1in.weight.grad[:,1+minute]))
        assert second['m1_ofi_mask_minute_'+str(minute)+'_weight_nonzero']==int(torch.count_nonzero(m1in.weight.grad[:,6+minute]))
    bad=SimpleNamespace(**vars(model.backbone));bad._multi_tf_specialist_index_tuples={MOMENTUM_FAMILY:(0,0,2,3)}
    with pytest.raises(RuntimeError,match='PRICE_MOMENTUM_FIELD_INDICES'):
        _candidate_gradient_counts(SimpleNamespace(**{**vars(model),'backbone':bad}),torch)
    assert _candidate_gradient_counts(SimpleNamespace(backbone=SimpleNamespace(entry_execution_profile=None)),torch)=={}


def test_private_input_adapter_binds_matched_price_and_m1_without_granting_launch():
    import hashlib
    from types import SimpleNamespace
    from gx1.scripts.run_entry_native_direction_prefix_fit_window_v1 import _candidate_view_inputs
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as authority
    from tests.test_entry_closed_m1_ofi_window_v1 import _model_clock_case
    case=_model_clock_case();source=case['source'];clock=windows.qualify_model_clock(**case)
    fitted=normalizer.fit(source,clock,prefix_rows=1,cutoff_ns=int(case['decision_ns'][1]))
    m5={'available_ns':case['m5_available_ns'],'native_row_indices':case['m5_native_row_indices'],
        'features':case['m5_features'],'field_names':case['m5_field_names']}
    prepared={'arrays':{'parent_rows':case['parent_rows'],'decision_ns':case['decision_ns']},
        'gc':{'surfaces':{'M5':m5},'layout':{'field_names':{'M5':case['m5_field_names']}}},
        'cache':{'M5':SimpleNamespace(attrs={'ts_int64':case['m5_cache_start_ns']})}}
    owners={path:_ref(path) for path in ('gx1/features/entry_direction_momentum_profile_v1.py',
        'gx1/contracts/entry_closed_m1_ofi_window_v1.py','gx1/contracts/entry_native_direction_m1_ofi_normalization_v1.py',
        'gx1/scripts/audit_raw_gc_m1_reference_v1.py')}
    launch={'run_scope':authority.M1_PRICE_PRIVATE_ADAPTER_SCOPE,'fold':{'prefix_rows':1,
        'train_cutoff_ns':int(case['decision_ns'][1])},'source_owners':owners}
    roll_ref={'path':'/synthetic/roll.json','sha256':hashlib.sha256(case['roll_bytes']).hexdigest()}
    candidate={'mode':'price_and_m1','source':source,'roll_bytes':case['roll_bytes'],'roll_ref':roll_ref,'normalization':fitted}
    sources={};kwargs=_candidate_view_inputs({'candidate_inputs':candidate},launch,prepared,sources)
    assert kwargs['price_momentum'] is True and kwargs['m1_ofi_normalization']==fitted
    assert prepared['m1_ofi_clock'].clock_sha256==clock.clock_sha256
    assert sources['m1_ofi_roll_schedule']==roll_ref and sources['m1_ofi_roll_owner_source']==owners['gx1/scripts/audit_raw_gc_m1_reference_v1.py']
    coarse_sources={};coarse=_candidate_view_inputs({'candidate_inputs':{**candidate,'mode':authority.M1_PRICE_COARSENED_MODE}},
        launch,prepared,coarse_sources)
    assert coarse['m1_ofi_coarsened'] is True and coarse['m1_ofi_normalization']==fitted and coarse_sources==sources
    price_sources={};price=_candidate_view_inputs({'candidate_inputs':{'mode':'price_only'}},launch,prepared,price_sources)
    assert price=={'price_momentum':True} and set(price_sources)=={'price_momentum_profile_source'}
    bad={**candidate,'roll_bytes':candidate['roll_bytes']+b' '}
    with pytest.raises(RuntimeError,match='CANDIDATE_ROLL_BINDING'):
        _candidate_view_inputs({'candidate_inputs':bad},launch,prepared,{})
    assert authority.M1_PRICE_PRIVATE_ADAPTER_SCOPE not in (authority.DENSE_SCOPE,authority.DENSE_DECIMAL_FIT_SCOPE)


def test_candidate_m1_guard_allows_only_exact_readonly_bound_inputs(monkeypatch):
    import os
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    hooks=[];monkeypatch.setattr(common.sys,'addaudithook',hooks.append)
    refs={name:{'path':'/home/andre2/GX1_RUNS/synthetic/'+filename,'sha256':char*64}
        for name,filename,char in (('source','m1.npz','a'),('roll','roll.json','b'),('normalization','norm.json','c'))}
    context={'ancestor_plan':{'payloads':{},'native_cache_dir':'/synthetic/cache',
        'gc_surfaces':{'path':'/synthetic/gc/surface'}},
        'info':{'metadata':{},'audit':{'manifest_path':'/synthetic/audit'}},'candidate_m1_refs':refs}
    denied=common._install_read_guard(context,'/synthetic/output','/synthetic/runtime',candidate_m1_refs=refs)
    assert context['candidate_m1_guard_installed'] is True
    for ref in refs.values():
        hooks[0]('open',(ref['path'],'rb',0))
        with pytest.raises(RuntimeError,match='ORIGINAL_WRITE'):
            hooks[0]('open',(ref['path'],'wb',os.O_WRONLY))
    with pytest.raises(RuntimeError,match='UNBOUND_OPEN'):
        hooks[0]('open',('/home/andre2/GX1_RUNS/synthetic/unbound.npz','rb',0))
    assert len(denied)==4
    bad=deepcopy(refs);bad['source']['sha256']='0'*64
    with pytest.raises(RuntimeError,match='CANDIDATE_M1_GUARD_BINDING'):
        common._install_read_guard(context,'/synthetic/output','/synthetic/runtime',candidate_m1_refs=bad)
    bad=deepcopy(refs);bad['source']['path']='/home/andre2/GX1_RUNS/TEST/m1.npz'
    with pytest.raises(RuntimeError,match='CANDIDATE_M1_PATH'):
        common._candidate_m1_refs(bad)
    bad=deepcopy(refs);bad['normalization']['path']=bad['roll']['path']
    with pytest.raises(RuntimeError,match='CANDIDATE_M1_DISTINCT'):
        common._candidate_m1_refs(bad)


def test_guarded_m1_source_loader_binds_real_bytes_and_detects_mutation(tmp_path,monkeypatch):
    import os
    from types import SimpleNamespace
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as authority
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as runner
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from tests.test_entry_closed_m1_ofi_window_v1 import _model_clock_case
    case=_model_clock_case();clock=windows.qualify_model_clock(**case);cutoff=int(case['decision_ns'][1])
    fitted=normalizer.fit(case['source'],clock,prefix_rows=1,cutoff_ns=cutoff)
    src=tmp_path/'m1.npz';roll=tmp_path/'roll.json';norm=tmp_path/'norm.json'
    np.savez(src,**case['source'].arrays);roll.write_bytes(case['roll_bytes']);norm.write_text(json.dumps(fitted))
    refs={key:common._hash_file(str(path)) for key,path in (('source',src),('roll',roll),('normalization',norm))}
    launch={'run_scope':authority.M1_PRICE_PRIVATE_ADAPTER_SCOPE,'candidate_mode':'price_and_m1',
        'candidate_m1_refs':refs,'fold':{'prefix_rows':1,'train_cutoff_ns':cutoff},'source_owners':{}}
    context={'ancestor_plan':{'payloads':{},'native_cache_dir':'/synthetic/cache',
        'gc_surfaces':{'path':'/synthetic/gc/surface'}},
        'info':{'metadata':{},'audit':{'manifest_path':'/synthetic/audit'}},'candidate_m1_refs':refs}
    with pytest.raises(RuntimeError,match='CANDIDATE_M1_NOT_GUARDED'):
        runner._load_bound_candidate_m1(context,launch,np,common=common)
    hooks=[];monkeypatch.setattr(common.sys,'addaudithook',hooks.append)
    common._install_read_guard(context,'/synthetic/output','/synthetic/runtime',candidate_m1_refs=refs)
    with pytest.raises(RuntimeError,match='ORIGINAL_WRITE'):
        hooks[0]('open',(str(src),'wb',os.O_WRONLY))
    candidate=runner._load_bound_candidate_m1(context,launch,np,common=common)
    assert candidate['source'].source_sha256==case['source'].source_sha256
    assert candidate['roll_bytes']==case['roll_bytes'] and candidate['normalization']==fitted
    assert all(not array.flags.writeable for array in candidate['source'].arrays.values())
    coarse_candidate=runner._load_bound_candidate_m1(context,{**launch,'candidate_mode':authority.M1_PRICE_COARSENED_MODE},np,common=common)
    assert coarse_candidate['mode']==authority.M1_PRICE_COARSENED_MODE and coarse_candidate['normalization']==candidate['normalization']
    m5={'available_ns':case['m5_available_ns'],'native_row_indices':case['m5_native_row_indices'],
        'features':case['m5_features'],'field_names':case['m5_field_names']}
    prepared={'arrays':{'parent_rows':case['parent_rows'],'decision_ns':case['decision_ns']},
        'gc':{'surfaces':{'M5':m5},'layout':{'field_names':{'M5':case['m5_field_names']}}},
        'cache':{'M5':SimpleNamespace(attrs={'ts_int64':case['m5_cache_start_ns']})}}
    paths=('gx1/features/entry_direction_momentum_profile_v1.py','gx1/contracts/entry_closed_m1_ofi_window_v1.py',
        'gx1/contracts/entry_native_direction_m1_ofi_normalization_v1.py','gx1/scripts/audit_raw_gc_m1_reference_v1.py')
    launch['source_owners']={path:_ref(path) for path in paths}
    view=runner._candidate_view_inputs({'candidate_inputs':candidate},launch,prepared,{})
    assert view['m1_ofi_normalization']==fitted and prepared['m1_ofi_clock'].clock_sha256==clock.clock_sha256
    wrong={**launch,'fold':{'prefix_rows':1,'train_cutoff_ns':cutoff+1}}
    with pytest.raises(RuntimeError,match='CANDIDATE_M1_FIT_BINDING'):
        runner._load_bound_candidate_m1(context,wrong,np,common=common)
    policy=tmp_path/'policy.json';policy.write_text('{}');policy_ref=common._hash_file(str(policy))
    binding={'source_head':'synthetic','policy':policy_ref}
    monkeypatch.setattr(common.subprocess,'check_output',lambda args,**kwargs:'synthetic\n' if 'rev-parse' in args else '')
    # Source owners are handled by their own preexisting freeze path; this test
    # isolates the newly added payload bookend with no physical market data.
    launch['source_owners']={}
    common._freeze(launch,binding,context)
    src.write_bytes(src.read_bytes()+b'altered')
    with pytest.raises(RuntimeError,match='CANDIDATE_M1_FREEZE:source'):
        common._freeze(launch,binding,context)
    with pytest.raises(RuntimeError,match='CANDIDATE_M1_SOURCE_SHA:source'):
        runner._load_bound_candidate_m1(context,launch,np,common=common)
