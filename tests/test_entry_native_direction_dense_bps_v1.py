"""Synthetic dense-bps contracts only; no native, market, training or trade evidence."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from gx1.contracts import entry_native_direction_bps_v1 as bps
from gx1.contracts import entry_native_direction_dense_v1 as dense
from gx1.contracts import entry_model_native_train_recipe_v1 as recipes
from gx1.contracts import entry_native_direction_research_state_v1 as state
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from tests import test_entry_native_direction_research_v1 as component
from tests import test_entry_native_direction_research_state_v1 as existing

M = 60 * 10**9


def _arrays():
    decisions = np.asarray([90, 100, 101], np.int64) * M
    base = np.linspace(0.1, 6.0, 60, dtype=np.float64)
    values = np.stack([base, -2 * base, np.zeros(60, dtype=np.float64)])
    observed = np.ones((3, 60), dtype=np.bool_)
    observed[0, 0] = False
    values[0, 0] = 0
    return {'parent_rows': np.arange(3, dtype=np.int64), 'decision_ns': decisions,
        'terminal_time_ns': decisions[:, None] + np.arange(1, 61, dtype=np.int64)[None, :] * M,
        'observed': observed, 'sign': np.sign(values).astype(np.int8), 'mid_return_bps': values}


def _scale(values=None):
    return bps.fit_dense_direction_bps_scale(**(_arrays() if values is None else values),
        cutoff_ns=160 * M, target_source={'path': '/synthetic/dense.npz', 'sha256': 'b' * 64},
        fit_partition='train')


def _loss(values, *, cutoff=200 * M):
    values = np.asarray(values, np.float64)
    n = len(values); decisions = torch.arange(100, 100 + n, dtype=torch.int64) * M
    return {'logit_up': torch.zeros((n, 60), requires_grad=True),
        'mean_return_bps': torch.zeros((n, 60), requires_grad=True), 'rms_bps': torch.ones(60),
        'mid_return_bps': torch.from_numpy(values),
        'sign': torch.from_numpy(np.sign(values).astype(np.int8)),
        'observed': torch.ones((n, 60), dtype=torch.bool), 'decision_ns': decisions,
        'terminal_time_ns': decisions[:, None] + torch.arange(1, 61)[None, :] * M, 'cutoff_ns': cutoff}


def test_dense_scale_is_per_horizon_train_only_including_ties_and_strict_cutoff():
    a = _arrays(); result = _scale(a)
    assert result['horizons_minutes'] == list(range(1, 61))
    assert result['counts'][0] == 2
    assert result['counts'][59] == 1
    assert result['rms_bps'][59] == float(np.float32(a['mid_return_bps'][0, 59]))
    changed = deepcopy(a)
    changed['mid_return_bps'][1, 59] = -9000.0
    changed['mid_return_bps'][2, 59] = 8000.0
    changed['sign'][:, 59] = np.sign(changed['mid_return_bps'][:, 59]).astype(np.int8)
    assert _scale(changed) == result
    corrupt = deepcopy(result); corrupt['rms_bps'][0] *= 2
    with pytest.raises(RuntimeError): bps.require_dense_direction_bps_scale(corrupt)
    with pytest.raises(RuntimeError): bps.require_direction_bps_scale(result)


@pytest.mark.parametrize('fault', ['scope', 'sign', 'clock', 'no_scale', 'dtype'])
def test_dense_scale_rejects_invalid_scope_and_targets(fault):
    a = _arrays(); kw = {'cutoff_ns': 160*M, 'target_source': {'path':'/synthetic/dense.npz','sha256':'b'*64}, 'fit_partition':'train'}
    if fault == 'scope': kw['fit_partition'] = 'control'
    elif fault == 'sign': a['sign'][0, 3] = -1
    elif fault == 'clock': a['terminal_time_ns'][0, 3] += 1
    elif fault == 'no_scale': a['mid_return_bps'][:, 2] = 0; a['sign'][:, 2] = 0
    else: a['mid_return_bps'] = a['mid_return_bps'].astype(np.float32)
    with pytest.raises(RuntimeError): bps.fit_dense_direction_bps_scale(**a, **kw)


def test_dense_bps_magnitude_produces_distinct_gradients_for_same_direction():
    args = _loss(np.stack([np.full(60, .1), np.full(60, 50.)]))
    out = bps.dense_direction_bps_loss(**args); out['loss'].backward()
    assert out['counts']['eligible_cells'] == out['bps_eligible_cells'] == 120
    assert out['research_objective_sha256'] == dense.dense_direction_bps_research_contract()['contract_sha256']
    assert torch.equal(args['logit_up'].grad[0], args['logit_up'].grad[1])
    torch.testing.assert_close(args['mean_return_bps'].grad[1] / args['mean_return_bps'].grad[0], torch.full((60,), 500.))


def test_dense_bps_ties_train_magnitude_but_do_not_create_direction_labels():
    args = _loss(np.zeros((1, 60)))
    args['mean_return_bps'] = torch.ones((1, 60), requires_grad=True)
    out = bps.dense_direction_bps_loss(**args)
    assert out['direction_loss'] is None and out['bps_eligible_cells'] == 60
    out['loss'].backward()
    assert args['logit_up'].grad is None and torch.all(args['mean_return_bps'].grad > 0)
    args = _loss(np.zeros((1, 60))); args['observed'].zero_()
    out = bps.dense_direction_bps_loss(**args)
    assert out['loss'] is None and not out['update_allowed']


def test_dense_bps_strict_cutoff_excludes_immature_targets_and_gradients():
    args = _loss(np.ones((1, 60)), cutoff=103*M)
    out = bps.dense_direction_bps_loss(**args)
    assert out['bps_eligible_cells'] == 2
    out['loss'].backward()
    assert torch.count_nonzero(args['mean_return_bps'].grad) == 2
    assert torch.all(args['mean_return_bps'].grad[0, 2:] == 0)


def test_dense_bps_and_original_three_head_loss_share_the_same_math():
    values = np.tile(np.asarray([[1.], [-2.]], np.float64), (1, 60)); args = _loss(values)
    wide = bps.dense_direction_bps_loss(**args)
    ix = torch.tensor([0, 4, 14]); small = {}
    for key, value in args.items():
        small[key] = value[:, ix] if key in ('logit_up', 'mean_return_bps', 'mid_return_bps', 'sign', 'observed', 'terminal_time_ns') else value[ix] if key == 'rms_bps' else value
    narrow = bps.native_direction_bps_loss(**small)
    torch.testing.assert_close(wide['direction_loss'], narrow['direction_loss'])
    torch.testing.assert_close(wide['bps_loss'], narrow['bps_loss'])
    torch.testing.assert_close(wide['loss'], narrow['loss'])


def test_dense_bps_initial_direction_rng_and_all_shared_parameters_are_identical(monkeypatch):
    monkeypatch.setattr(component.EntryV10CtxHybridTransformer, 'forward', component._native_forward_spy)
    torch.manual_seed(420); baseline = dense.DenseNativeEntryDirectionResearch(component._native_shell())
    saved_rng = torch.get_rng_state().clone()
    torch.manual_seed(420); candidate = dense.DenseNativeEntryDirectionResearch(component._native_shell(), dense_bps_scale=_scale())
    assert torch.equal(saved_rng, torch.get_rng_state())
    for name, value in baseline.state_dict().items(): assert torch.equal(value, candidate.state_dict()[name])
    out = candidate(**component._inputs()); before = baseline(**component._inputs())
    assert torch.equal(out['logit_up'], before['logit_up'])
    assert out['logit_up'].shape == out['mean_return_bps'].shape == (2, 60)
    assert torch.equal(out['mean_return_bps'], torch.zeros((2, 60)))
    assert len(candidate.backbone._spy_calls) == 1
    assert candidate.research_objective != baseline.research_objective
    with pytest.raises(TypeError): candidate(**component._inputs(), mid_return_bps=torch.zeros((2, 60)))
    candidate.dense_bps_rms[0] *= 2
    with pytest.raises(RuntimeError, match='BPS_SCALE'): candidate(**component._inputs())
    candidate._dense_bps_scale_json = None
    candidate._dense_objective_json = dense._OBJECTIVE_JSON
    with pytest.raises(RuntimeError, match='UNBOUND_BPS_STATE'): candidate(**component._inputs())


def _state_case(monkeypatch):
    case = existing._make(monkeypatch)
    model = dense.DenseNativeEntryDirectionResearch(case.model.backbone, dense_bps_scale=_scale())
    identity = existing._identity(); identity['research_objective_sha256'] = dense.dense_direction_bps_research_contract()['contract_sha256']
    model.feature_view_identity = identity
    recipe = recipes.native_direction_research_recipe(batch_size=2, max_updates=4, dense_minutes=True, dense_bps=True)
    optimizer, binding = state.build_native_direction_research_optimizer(model, recipe=recipe)
    model.train(); existing._seed(recipe['seed'])
    return model, optimizer, recipe, identity, case.order, binding


def _update(parts, step):
    model, optimizer, *_ = parts; out = model(**existing._features(step))
    decisions = torch.tensor([100 + 2*step, 101 + 2*step], dtype=torch.int64)*M
    sign = torch.where(torch.arange(60)[None, :] % 2 == 0, 1, -1).to(torch.int8).repeat(2, 1)
    target = sign.to(torch.float64) * torch.arange(1, 61, dtype=torch.float64)[None, :]
    result = bps.dense_direction_bps_loss(out['logit_up'], out['mean_return_bps'], rms_bps=model.dense_bps_rms,
        mid_return_bps=target, sign=sign, observed=torch.ones((2, 60), dtype=torch.bool),
        decision_ns=decisions, terminal_time_ns=decisions[:, None]+torch.arange(1, 61)[None, :]*M, cutoff_ns=200*M)
    result['loss'].backward()
    assert torch.count_nonzero(model.dense_bps_head.weight.grad) > 0
    trainer._optimizer_step_with_finite_gradients(model=model, optimizer=optimizer)


def _capture(parts, cursor, updates):
    model, optimizer, recipe, identity, order, _ = parts
    return state.capture_native_direction_research_state(model, optimizer, recipe=recipe, feature_view_identity=identity,
        parent_order=order, cursor=cursor, optimizer_updates=updates, device=torch.device('cpu'))


def test_dense_bps_resume_preserves_optimizer_rng_and_all_parameters(monkeypatch):
    full = _state_case(monkeypatch); _update(full, 0); _update(full, 1); expected = _capture(full, 4, 2)
    part = _state_case(monkeypatch); _update(part, 0); saved = _capture(part, 2, 1)
    resumed = _state_case(monkeypatch); model, optimizer, recipe, identity, order, binding = resumed
    assert {'dense_bps_head.weight','dense_bps_head.bias'}.issubset(binding['active_parameter_names'])
    assert binding['dense_bps_scale_contract_sha256'] == model.dense_bps_scale['contract_sha256']
    state.restore_native_direction_research_state(saved, model, optimizer, recipe=recipe,
        feature_view_identity=identity, parent_order=order, device=torch.device('cpu'))
    _update(resumed, 1); actual = _capture(resumed, 4, 2)
    assert actual['contract_sha256'] == expected['contract_sha256']
    assert actual['optimizer_state_sha256'] == expected['optimizer_state_sha256']
    assert actual['rng_state_sha256'] == expected['rng_state_sha256']


def test_recipe_keeps_previous_modes_and_requires_explicit_dense_scope():
    normal = recipes.native_direction_research_recipe(batch_size=2,max_updates=4,dense_minutes=True)
    b = recipes.native_direction_research_recipe(batch_size=2,max_updates=4,dense_minutes=True,dense_bps=True)
    assert 'dense_signed_bps_supervision' not in normal and b['dense_signed_bps_supervision'] is True
    assert b['research_objective_sha256'] == dense.dense_direction_bps_research_contract()['contract_sha256']
    assert recipes.require_native_direction_research_recipe(b) == b
    for kw in ({'dense_bps':True}, {'dense_bps':True,'dense_minutes':True,'five_minute_priority':True},
               {'dense_bps':True,'dense_minutes':True,'signed_bps':True}, {'dense_bps':1,'dense_minutes':True}):
        with pytest.raises(RuntimeError): recipes.native_direction_research_recipe(batch_size=2,max_updates=4,**kw)

from tests.test_entry_native_direction_feature_view_v1 import bound_case, _constructor_spy, _ref
from gx1.contracts import entry_native_direction_feature_view_v1 as view_owner
from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as window


def _scale_for_view(view):
    scale = _scale(); prefix = view['prefix']
    scale.update(prefix_rows=prefix['rows'], cutoff_ns=prefix['cutoff_ns'],
        parent_rows_sha256=prefix['parent_rows_sha256'], decision_ns_sha256=prefix['decision_ns_sha256'])
    scale.pop('contract_sha256'); scale['contract_sha256'] = bps._sha(scale)
    return bps.require_dense_direction_bps_scale(scale)


def test_dense_bps_view_is_m5_bound_and_rejects_wrong_train_prefix(bound_case, monkeypatch):
    args = deepcopy(bound_case['arguments'])
    args['source_bindings'].update({key:_ref(key+'.py') for key in
        view_owner.DENSE_SOURCE_REF_KEYS | view_owner.PRICE_MOMENTUM_SOURCE_REF_KEYS | view_owner.DENSE_BPS_SOURCE_REF_KEYS})
    scale = _scale_for_view(bound_case['view'])
    view = view_owner.bind_native_direction_feature_view(**args, dense_minutes=True, price_momentum=True, dense_bps_scale=scale)
    assert view['research_objective'] == dense.dense_direction_bps_research_contract()
    assert view['dense_bps_scale'] == scale and 'm1_ofi_normalization' not in view
    calls, norm_calls = _constructor_spy(monkeypatch)
    model = view_owner.build_native_direction_research_model(view, prefix_normalization=args['prefix_normalization'],
        constructor_metadata=args['constructor_metadata'], device=torch.device('cpu'))
    assert model.dense_signed_bps and model.dense_bps_head.out_features == 60
    assert model.feature_view_identity['research_objective_sha256'] == view['research_objective']['contract_sha256']
    assert len(calls) == len(norm_calls) == 1
    wrong = deepcopy(scale); wrong['parent_rows_sha256'] = '0'*64
    wrong.pop('contract_sha256'); wrong['contract_sha256'] = bps._sha(wrong)
    with pytest.raises(RuntimeError, match='DENSE_BPS_OWN_PREFIX'):
        view_owner.bind_native_direction_feature_view(**args, dense_minutes=True, price_momentum=True, dense_bps_scale=wrong)
    with pytest.raises(RuntimeError):
        view_owner.bind_native_direction_feature_view(**args, dense_minutes=True, dense_bps_scale=scale)


def test_dense_bps_real_runner_dispatches_both_losses_and_updates_both_heads(monkeypatch):
    model, optimizer, *_ = _state_case(monkeypatch)
    monkeypatch.setattr(window, '_forward', lambda model, features, **kwargs: model(**features) if kwargs.get('return_outputs') else model(**features)['logit_up'])
    args = _loss(np.stack([np.full(60, .1), np.full(60, -2.)]))
    target = {key:args[key] for key in ('mid_return_bps','sign','observed','decision_ns','terminal_time_ns')}
    counters = {key:0 for key in ('backward_calls','backward_returns','optimizer_step_calls','optimizer_step_returns','empty_batches')}
    before_dir = model.dense_aux_head.weight.detach().clone(); before_bps = model.dense_bps_head.weight.detach().clone()
    record, updated = window._train_batch(model, optimizer, existing._features(0), target,
        cutoff=200*M, counters=counters, ceiling=1, torch=torch, diagnostics=False)
    assert updated and record['eligible_cells'] == record['bps_eligible_cells'] == 120
    assert len(record['per_horizon_counts']) == len(record['bps_per_horizon_counts']) == 60
    assert record['direction_loss'] > 0 and record['bps_loss'] > 0
    assert not torch.equal(before_dir, model.dense_aux_head.weight)
    assert not torch.equal(before_bps, model.dense_bps_head.weight)
    assert counters['optimizer_step_returns'] == 1


def test_dense_bps_scale_is_fail_closed_without_own_qualification():
    scale = _scale(); view = {'dense_bps_scale':scale}; reference = {'path':'/synthetic/scale.json','sha256':'b'*64}
    with pytest.raises(RuntimeError,match='QUALIFIED_DENSE_BPS_SCALE'):
        window._qualified_bps_scale({},view)
    assert window._qualified_bps_scale({'dense_bps_qualified_scale':scale,'dense_bps_qualified_scale_ref':reference},view) == reference


def test_dense_bps_scoring_reports_all_sixty_with_strict_maturity():
    values = _arrays(); scale = _scale(); prediction = np.zeros((3,60),np.float32)
    result = bps.score_dense_direction_bps(prediction, mid_return_bps=values['mid_return_bps'],
        observed=values['observed'], terminal_time_ns=values['terminal_time_ns'], cutoff_ns=160*M, scale=scale)
    assert len(result['per_horizon']) == 60
    assert result['per_horizon'][0]['count'] == 2 and result['per_horizon'][59]['count'] == 1
    assert result['per_horizon'][59]['sse_bps2'] == values['mid_return_bps'][0,59]**2
    assert result['scale_contract_sha256'] == scale['contract_sha256']


def test_preparation_routes_corrected_60_bps_to_loss_only(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as authority
    from gx1.contracts import entry_native_direction_context_report_v1 as report
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from tests.test_entry_native_direction_decimal_input_v1 import make_case
    refs, axes, legacy, corrected = make_case(tmp_path)
    class Archive:
        files=list(window.PRICE_MEMBERS)
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def __getitem__(self,key): return np.tile(np.array([60,300,900],np.int64),(2,1)) if key=='elapsed_wall_seconds' else legacy[key]
    np_proxy=SimpleNamespace(load=lambda path,allow_pickle=False:Archive())
    arrays={k:axes[k].copy() for k in ('parent_rows','bar_start_ns','decision_ns')}
    prepared={'arrays':arrays,'gc':{'layout':{},'transition':{}},
        'entry_sequence_source':{'times_ns':np.arange(3,dtype=np.int64),'sample_positions':np.arange(2,dtype=np.int64)}}
    physical=SimpleNamespace(prepare_inputs=lambda *_:prepared,release_pages=lambda *_:None)
    sources={k:{'path':'/synthetic/'+k,'sha256':'a'*64} for k in view_owner.SOURCE_REF_KEYS}
    manifest={'targets':refs['legacy'],'view_source_bindings':sources,
        'prefix_normalization':{'output':{'path':'/synthetic/prefix','sha256':'a'*64}},
        'constructor_metadata':{'path':'/synthetic/constructor','sha256':'a'*64}}
    context={'physical':physical,'ancestor':object(),'ancestor_plan':{},'info':{'metadata':{}},'manifest':manifest,
        'fitted':{'base_normalization':{'surfaces':{'mtf_m5':{'field_names':[]}}}},'constructor':{},
        'dense_target':refs['corrected'],'decimal_original':refs['original'],'decimal_target_refs':refs,'candidate_inputs':{}}
    paths=('gx1/contracts/entry_native_direction_dense_v1.py','gx1/contracts/entry_native_direction_bps_v1.py',
        'gx1/features/entry_specialist_feature_groups_v1.py','gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py')
    owners={p:{'path':'/synthetic/'+p,'sha256':'b'*64} for p in paths}
    launch={'run_scope':authority.M1_PRICE_PRIVATE_ADAPTER_SCOPE,'dense_target':refs['corrected'],
        'decimal_target_refs':refs,'fold':{'prefix_rows':2,'train_cutoff_ns':int(axes['decision_ns'][-1]+100*M)},'source_owners':owners}
    captures=[]; monkeypatch.setattr(common,'_read',lambda *_:{})
    monkeypatch.setattr(window,'_qualify_axes_targets',lambda *a,**k:None)
    monkeypatch.setattr(window,'_presentation_order',lambda parents,*a,**k:parents.copy())
    monkeypatch.setattr(window,'_candidate_view_inputs',lambda *a,**k:{'price_momentum':True})
    monkeypatch.setattr(view_owner,'bind_native_direction_feature_view',lambda **kwargs:kwargs)
    monkeypatch.setattr(report,'native_direction_context_map',lambda **kwargs:{})
    synthetic_values=np.ones((2,60),np.float64)
    expected_scale=bps.fit_dense_direction_bps_scale(parent_rows=arrays['parent_rows'],decision_ns=arrays['decision_ns'],
        terminal_time_ns=arrays['decision_ns'][:,None]+np.arange(1,61,dtype=np.int64)[None,:]*M,
        observed=np.ones((2,60),np.bool_),sign=np.ones((2,60),np.int8),mid_return_bps=synthetic_values,
        cutoff_ns=launch['fold']['train_cutoff_ns'],target_source=refs['corrected'],fit_partition='train')
    context['dense_bps_qualified_scale']=expected_scale
    monkeypatch.setattr(bps,'fit_dense_direction_bps_scale',lambda **kwargs:pytest.fail('qualified scale must be reused'))
    _, targets, view, _, _ = window._prepare(context,launch,np_proxy,dense_minutes=True,dense_bps=True)
    assert np.array_equal(targets['mid_return_bps'],corrected['mid_return_bps'])
    assert view['dense_bps_scale']==expected_scale and view['price_momentum'] is True
    assert all(not value.flags.writeable for value in targets.values())
    wrong=deepcopy(expected_scale);wrong['target_source']['sha256']='0'*64
    wrong.pop('contract_sha256');wrong['contract_sha256']=bps._sha(wrong)
    context['dense_bps_qualified_scale']=wrong
    with pytest.raises(RuntimeError,match='DENSE_BPS_SCALE_SCOPE'):
        window._prepare(context,launch,np_proxy,dense_minutes=True,dense_bps=True)


def test_dense_bps_evaluation_saves_all_sixty_scores_without_state_change(monkeypatch):
    from gx1.contracts import entry_native_direction_context_report_v1 as reporter
    from gx1.contracts import entry_native_direction_dense_report_v1 as dense_report
    from gx1.scripts.run_unified_exit_random_access_full_train_v1 import _dynamic_m5_technical_tree_sha256 as digest
    model, _, _, _, _, _ = _state_case(monkeypatch); selected=np.asarray([0,1,2,3],np.int64)
    def deterministic_native(self, seq_x, snap_x, **features):
        x=self.entry_q_joint_in(self.entry_q_joint_norm(torch.cat([seq_x[:,0,:]]*4,dim=-1)))
        return {'entry_q_joint_hidden':x+self.persistent_probe+self.ephemeral_probe}
    monkeypatch.setattr(component.EntryV10CtxHybridTransformer,'forward',deterministic_native)
    def batch(prepared, targets, view, ids, np):
        d=(100+ids)*M; val=np.where(ids[:,None]%2==0,1.,-1.)*np.ones((len(ids),60),np.float64)
        raw={'features':existing._features(0),'witness':{'parent_rows':ids.tolist(),'bar_start_ns':(d-5*M).tolist()}}
        target={'decision_ns':d,'terminal_time_ns':d[:,None]+np.arange(1,61,dtype=np.int64)[None,:]*M,
            'observed':np.ones((len(ids),60),np.bool_),'sign':np.sign(val).astype(np.int8),'mid_return_bps':val}
        return raw,target
    def tag(features,*,context_map,witness):
        ids=np.asarray(witness['parent_rows'],np.int64); d=np.asarray(witness['bar_start_ns'],np.int64)+5*M
        return {'schema_version':'synthetic_tags','context_map_sha256':'a'*64,'feature_view_sha256':'b'*64,
            'parent_rows':ids,'decision_ns':d,'tags':{'synthetic':np.zeros(len(ids),np.int8)}}
    def forward(m,features,*,grad=False,return_outputs=False,**kwargs):
        with torch.set_grad_enabled(grad): out=m(**features)
        return out if return_outputs else out['logit_up']
    monkeypatch.setattr(window,'BATCH_SIZE',2);monkeypatch.setattr(window,'_batch',batch)
    monkeypatch.setattr(window,'_device_batch',lambda raw,target,torch,device:(raw['features'],{}))
    monkeypatch.setattr(window,'_forward',forward);monkeypatch.setattr(reporter,'tag_native_direction_contexts',tag)
    monkeypatch.setattr(dense_report,'score_dense_direction_logits',lambda logits,**kwargs:{'rows':len(logits)})
    measured,count=window._evaluate(model,{}, {},{'dense_minute_supervision':True},{'contract_sha256':'a'*64},selected,
        cutoff=200*M,kind='INITIAL_TRAIN',counters={},ceiling=10,device=torch.device('cpu'),torch=torch,np=np,tree_sha=digest,parity=True)
    assert count==4 and measured['initial_parity_passed']
    assert measured['arrays']['logit_up'].shape==measured['arrays']['mean_return_bps'].shape==(4,60)
    assert measured['arrays']['mid_return_bps'].shape==(4,60)
    assert len(measured['report']['signed_bps']['per_horizon'])==60
    assert all(p.grad is None for p in model.parameters())


def test_dense_bps_physical_shape_validation_rejects_wrong_sign_and_mode():
    from tests.test_entry_native_direction_prefix_fit_window_v1 import _axes_targets
    axes, _, fold = _axes_targets(); decisions=axes['decision_ns']; n=len(decisions)
    targets={'terminal_time_ns':decisions[:,None]+np.arange(1,61,dtype=np.int64)[None,:]*M,
        'observed':np.ones((n,60),np.bool_),'sign':np.ones((n,60),np.int8),
        'mid_return_bps':np.ones((n,60),np.float64)}
    window._qualify_axes_targets(axes,targets,fold,np,dense_minutes=True,dense_bps=True)
    targets['mid_return_bps'][0,4]=-1
    with pytest.raises(RuntimeError,match='BPS_TARGET_DOMAIN'):
        window._qualify_axes_targets(axes,targets,fold,np,dense_minutes=True,dense_bps=True)
    with pytest.raises(RuntimeError,match='OBJECTIVE_MODE'):
        window._qualify_axes_targets(axes,targets,fold,np,dense_bps=True)


def test_private_step_dispatches_dense_bps_and_preserves_all_horizons(monkeypatch):
    from contextlib import contextmanager
    from gx1.scripts import run_entry_native_direction_research_window_v1 as private
    parts = _state_case(monkeypatch);model, optimizer, *_ = parts
    @contextmanager
    def no_fastpath(backbone):
        before=torch.backends.mha.get_fastpath_enabled();torch.backends.mha.set_fastpath_enabled(False)
        try: yield
        finally: torch.backends.mha.set_fastpath_enabled(before)
    monkeypatch.setattr(trainer,'_training_mha_fastpath_context',no_fastpath)
    monkeypatch.setattr(trainer,'_model_forward_fp32',lambda model,**features:model(**features))
    monkeypatch.setattr(private,'_gradient_evidence',lambda model,torch:{'synthetic':1})
    args=_loss(np.stack([np.full(60,.1),np.full(60,-2.)]))
    target={key:args[key] for key in ('mid_return_bps','sign','observed','decision_ns','terminal_time_ns')}
    counters={k:0 for k in ('forwards','forward_returns','backwards','backward_returns','optimizer_step_calls','optimizer_updates')}
    before=model.dense_bps_head.weight.detach().clone()
    result=private._step(model,optimizer,existing._features(0),target,cutoff=200*M,counters=counters,torch=torch)
    assert result['counts']['eligible_cells']==120 and len(result['per_horizon_counts'])==60
    assert counters['optimizer_updates']==1 and not torch.equal(before,model.dense_bps_head.weight)
    saved=_capture(parts,2,1)
    probed=private._step(model,optimizer,existing._features(1),target,cutoff=200*M,counters=counters,torch=torch,bps_only_probe=True)
    assert counters['optimizer_updates']==2 and counters['backwards']==3
    assert probed['bps_only_probe']['joint_projection_nonzero']>0
    expected=_capture(parts,4,2)
    resumed=_state_case(monkeypatch); rm,ro,recipe,identity,order,_=resumed
    state.restore_native_direction_research_state(saved,rm,ro,recipe=recipe,feature_view_identity=identity,
        parent_order=order,device=torch.device('cpu'))
    other={k:0 for k in ('forwards','forward_returns','backwards','backward_returns','optimizer_step_calls','optimizer_updates')}
    private._step(rm,ro,existing._features(1),target,cutoff=200*M,counters=other,torch=torch)
    assert _capture(resumed,4,2)['contract_sha256']==expected['contract_sha256']


def test_private_gradient_evidence_checks_all_sixty_bps_rows(monkeypatch):
    from gx1.scripts import run_entry_native_direction_research_window_v1 as private
    model, _, _, _, _, _ = _state_case(monkeypatch)
    nn=torch.nn
    model.backbone.gc_mtf_family_proj=nn.ModuleDict({f'{tf}__{i}':nn.Linear(1,1) for tf in ('m5','m15','h1','h4','d1') for i in range(5)})
    model.backbone.gc_mtf_feature_context_gate=nn.ModuleDict({f'{tf}__{i}':nn.Linear(1,1) for tf in ('m5','m15','h1','h4','d1') for i in range(5)})
    model.backbone.specialist_proj=nn.ModuleDict({'chart_geometry_encoder':nn.Linear(1,1)})
    model.backbone.mtf_family_proj=nn.ModuleDict({'chart_geometry_encoder':nn.Linear(1,1)})
    monkeypatch.setattr(private,'_candidate_gradient_counts',lambda *args:{})
    for p in model.parameters():
        if p.requires_grad:p.grad=torch.ones_like(p)
    evidence=private._gradient_evidence(model,torch)
    assert len(evidence['dense_bps_head_weight_rows'])==60
    assert all(evidence['dense_bps_head_weight_rows'])
    model.dense_bps_head.weight.grad[9].zero_();model.dense_bps_head.bias.grad[9]=0
    with pytest.raises(RuntimeError,match='ALL_SIXTY_BPS_HEAD_ROWS'):
        private._gradient_evidence(model,torch)


def test_two_pass_resume_preserves_full_state_with_partial_tail(monkeypatch):
    from gx1.contracts import entry_native_direction_research_state_v1 as owner
    continuation=recipes.native_direction_research_recipe(batch_size=2,max_updates=4)
    def capture(case,cursor,updates):
        return owner.capture_native_direction_research_state(case.model,case.optimizer,recipe=continuation,
            feature_view_identity=existing._identity(),parent_order=case.order,cursor=cursor,
            optimizer_updates=updates,device=torch.device('cpu'),train_passes=2)
    full=existing._multipass_case(monkeypatch)
    for _ in range(2):
        assert existing._multipass_batch(full,[0,1]);assert existing._multipass_batch(full,[2])
    expected=capture(full,6,4)
    first=existing._multipass_case(monkeypatch)
    assert existing._multipass_batch(first,[0,1]);assert existing._multipass_batch(first,[2])
    terminal=existing._capture(first,3,2)
    migrated=owner.migrate_one_pass_terminal_to_two_pass(terminal,first.model,first.optimizer,original_recipe=first.recipe,
        continuation_recipe=continuation,feature_view_identity=existing._identity(),parent_order=first.order,device=torch.device('cpu'))
    assert migrated['cursor']==3 and migrated['optimizer_updates']==2 and migrated['train_passes']==2
    for key in ('model_state_sha256','buffer_state_sha256','optimizer_state_sha256','rng_state_sha256'):
        assert migrated[key]==terminal[key]
    assert existing._multipass_batch(first,[0,1]);interrupted=capture(first,5,3)
    resumed=existing._multipass_case(monkeypatch)
    owner.restore_native_direction_research_state(interrupted,resumed.model,resumed.optimizer,recipe=continuation,
        feature_view_identity=existing._identity(),parent_order=resumed.order,device=torch.device('cpu'),train_passes=2)
    assert existing._multipass_batch(resumed,[2]);actual=capture(resumed,6,4)
    assert actual['contract_sha256']==expected['contract_sha256']
    with pytest.raises(RuntimeError): capture(resumed,7,4)


def test_two_pass_parent_ranges_keep_partial_tails_and_do_not_admit_launch():
    assert [window._next_training_parent_range(c,18,2) for c in (0,16,18,34)]==[(0,16,16),(16,18,18),(0,16,34),(16,18,36)]
    with pytest.raises(RuntimeError,match='TRAIN_PARENT_RANGE'):window._next_training_parent_range(36,18,2)
    with pytest.raises(RuntimeError,match='TRAIN_PARENT_RANGE'):window._next_training_parent_range(0,18,4)
