"""Synthetic270 state checks: no real native constructor/feature/training proof.

Exact260 wrappers use native __new__ shells with a stochastic artificial forward.
The tests exercise real AdamW/state/RNG mechanics only. No market data, fitted
normalizer, EMA, scheduler, production session or actual native forward is used.
"""
from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from gx1.contracts import entry_model_native_train_recipe_v1 as recipes
from gx1.contracts import entry_native_direction_research_state_v1 as state_owner
from gx1.contracts import entry_native_direction_research_v1 as direction
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import CtxModelConfig, EntryV10CtxHybridTransformer

DEVICE = torch.device('cpu')
MINUTE = 60_000_000_000


def _identity():
    return {'contract_sha256':'1'*64,'research_objective_sha256':direction.native_direction_research_contract()['contract_sha256'],
        'prefix_normalization':{'binding':{'path':'/synthetic/PREFIX_NORMALIZATION.json','sha256':'2'*64},
            'contract_sha256':'3'*64,'base_sha256':'4'*64,'gc_layout_sha256':'5'*64,
            'gc_normalization_sha256':'6'*64,'qualification_sha256':'7'*64}}


def _forward_spy(self, seq_x, snap_x, **features):
    value = self.entry_q_joint_in(self.entry_q_joint_norm(torch.cat([seq_x[:,0,:]]*4,dim=-1)))
    # All three global RNGs affect actual gradients; restore must recover each.
    value = self._shared_drop(value + (random.random()+float(np.random.random()))*torch.rand_like(value))
    value = value + self.persistent_probe + self.ephemeral_probe
    return {'entry_q_joint_hidden':value,'dynamic_mfe_bps':self.head_dynamic_mfe(value),
        'forecast_pred':self.head_forecast(value)}


def _seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def _make(monkeypatch):
    monkeypatch.setattr(EntryV10CtxHybridTransformer,'forward',_forward_spy)
    monkeypatch.setattr(EntryV10CtxHybridTransformer,'require_input_normalization_state',lambda self:None)
    recipe = recipes.native_direction_research_recipe(batch_size=2,max_updates=4)
    _seed(recipe['seed'])
    native = EntryV10CtxHybridTransformer.__new__(EntryV10CtxHybridTransformer)
    nn.Module.__init__(native)
    cfg = CtxModelConfig.__new__(CtxModelConfig)
    object.__setattr__(cfg,'d_model',4)
    object.__setattr__(cfg,'entry_training_phase',direction.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    native.cfg=cfg;native.entry_training_phase=cfg.entry_training_phase
    native.entry_q_joint_norm=nn.LayerNorm(16);native.entry_q_joint_in=nn.Linear(16,4)
    native._shared_drop=nn.Dropout(.25)
    native.head_dynamic_mfe=nn.Linear(4,2);native.head_forecast=nn.Linear(4,3)
    native.dynamic_m5_query_projection=nn.Linear(4,3)
    native.task_log_variances=nn.ParameterDict({'unused':nn.Parameter(torch.zeros(()))})
    native.entry_decision_token=nn.Sequential(nn.Linear(4,4));native.exit_probe=nn.Linear(4,1)
    native.register_buffer('persistent_probe',torch.tensor([.1,.2,.3,.4]))
    native.register_buffer('ephemeral_probe',torch.tensor([.4,.3,.2,.1]),persistent=False)
    native.register_buffer('input_norm_signal_center',torch.zeros(4))
    model=direction.NativeEntryDirectionResearch(native);model.feature_view_identity=_identity()
    optimizer,binding=state_owner.build_native_direction_research_optimizer(model,recipe=recipe)
    model.train()
    # Remove any cold optimizer import effects from this synthetic stochastic probe.
    _seed(recipe['seed'])
    return SimpleNamespace(model=model,optimizer=optimizer,binding=binding,recipe=recipe,order=torch.arange(4,dtype=torch.int64))


def _features(step):
    x=torch.tensor([[1.,2.,3.,4.],[4.,2.,1.,3.]]) + float(step)
    features={'seq_x':x[:,None,:],'snap_x':x,'ctx_cont':torch.zeros(2,1),'ctx_cat':torch.zeros(2,1,dtype=torch.int64)}
    for tf in ('m5','m15','h1','h4','d1'):
        features['seq_'+tf]=torch.zeros(2,1,1);features['gc_seq_'+tf]=torch.zeros(2,1,1)
    return features


def _loss(case, step, *, empty=False):
    logits=case.model(**_features(step))['logit_up']
    D=torch.tensor([100+2*step,101+2*step],dtype=torch.int64)*MINUTE
    return direction.native_direction_binary_loss(logits,
        sign=torch.zeros(2,3,dtype=torch.int8) if empty else torch.tensor([[1,-1,1],[-1,1,-1]],dtype=torch.int8),
        observed=torch.ones(2,3,dtype=torch.bool),decision_ns=D,
        terminal_time_ns=D[:,None]+torch.tensor([1,5,15],dtype=torch.int64)[None,:]*MINUTE,cutoff_ns=200*MINUTE)


def _update(case, step):
    result=_loss(case,step);assert result['update_allowed']
    result['loss'].backward()
    trainer._optimizer_step_with_finite_gradients(model=case.model,optimizer=case.optimizer)


def _capture(case, cursor, updates):
    return state_owner.capture_native_direction_research_state(case.model,case.optimizer,recipe=case.recipe,
        feature_view_identity=_identity(),parent_order=case.order,cursor=cursor,optimizer_updates=updates,device=DEVICE)


def _restore(case, saved):
    return state_owner.restore_native_direction_research_state(saved,case.model,case.optimizer,recipe=case.recipe,
        feature_view_identity=_identity(),parent_order=case.order,device=DEVICE)


def _live_hash(case):
    return state_owner._tree_sha({'model':case.model.state_dict(),'buffers':dict(case.model.named_buffers()),
        'optimizer':case.optimizer.state_dict(),'rng':trainer._attended_session_rng_state(device=DEVICE),
        'modes':{name:module.training for name,module in case.model.named_modules()}})


def _reseal(saved):
    saved['contract_sha256']=state_owner._tree_sha({key:value for key,value in saved.items() if key!='contract_sha256'})


def test_research_recipe_off_controls_copies_and_native_recipe_unchanged():
    old=dict(recipes.MODEL_NATIVE_RECIPE_ENV)
    value=recipes.native_direction_research_recipe(batch_size=2,max_updates=4)
    assert value['weight_ema']=='0.0' and value['lr_cosine_decay']=='0' and value['predictions']=='ONLINE'
    assert value['optimizer']['kwargs']['lr']==value['optimizer']['kwargs']['weight_decay']==.0001
    assert recipes.require_native_direction_research_recipe(value)==value
    value['optimizer']['kwargs']['betas'][0]=.5
    assert recipes.native_direction_research_recipe(batch_size=2,max_updates=4)['optimizer']['kwargs']['betas']==[.9,.999]
    assert recipes.MODEL_NATIVE_RECIPE_ENV==old and recipes.require_model_native_recipe_env(old)==old
    forged=recipes.native_direction_research_recipe(batch_size=2,max_updates=4);forged['grad_accum_steps']=True
    with pytest.raises(RuntimeError,match='RECIPE_MISMATCH'):recipes.require_native_direction_research_recipe(forged)


@pytest.mark.parametrize('batch,updates',[(True,2),(0,2),(-1,2),(2.,2),(2,True),(2,0),(2,-1),(2,2.)])
def test_recipe_requires_explicit_positive_integer_geometry(batch,updates):
    with pytest.raises(RuntimeError,match='POSITIVE_BUDGET'):
        recipes.native_direction_research_recipe(batch_size=batch,max_updates=updates)


def test_optimizer_partition_defaults_and_no_old_phase_aliases(monkeypatch):
    case=_make(monkeypatch);parameters=dict(case.model.named_parameters())
    active=case.binding['active_parameter_names'];inactive=case.binding['inactive_parameter_names']
    assert set(active).isdisjoint(inactive) and set(active)|set(inactive)==set(parameters)
    assert all(parameters[name].requires_grad for name in active)
    assert all(not parameters[name].requires_grad for name in inactive)
    assert [id(p) for p in case.optimizer.param_groups[0]['params']]==[id(parameters[name]) for name in active]
    assert type(case.optimizer) is torch.optim.AdamW and case.optimizer.defaults['weight_decay']==.0001
    assert not hasattr(case.model,'entry_training_phase') and not hasattr(case.model,'task_log_variances')
    assert case.binding['ema_created'] is case.binding['scheduler_created'] is case.binding['target_model_created'] is False
    case.optimizer.defaults['capturable']=0
    with pytest.raises(RuntimeError,match='DEFAULTS'):_capture(case,0,0)
    case.optimizer.defaults['capturable']=False;case.optimizer.param_groups[0]['capturable']=0
    with pytest.raises(RuntimeError,match='GROUP_ORDER'):_capture(case,0,0)
    case.optimizer.param_groups[0]['capturable']=False
    case.model.direction_head_alias=case.model.direction_head
    with pytest.raises(RuntimeError,match='PARAMETER_ALIASES'):
        state_owner.build_native_direction_research_optimizer(case.model,recipe=case.recipe)


def test_snapshot_is_independent_and_restores_full_buffers_modes_rng_and_cursor(monkeypatch):
    case=_make(monkeypatch);case.model.backbone.head_forecast.eval()
    inactive={name:p.detach().clone() for name,p in case.model.named_parameters() if name in case.binding['inactive_parameter_names']}
    active={name:p.detach().clone() for name,p in case.model.named_parameters() if name in case.binding['active_parameter_names']}
    _update(case,0);saved=_capture(case,2,1);expected=_live_hash(case)
    assert any(not torch.equal(p,active[name]) for name,p in case.model.named_parameters() if name in active)
    assert all(torch.equal(p,inactive[name]) and p.grad is None for name,p in case.model.named_parameters() if name in inactive)
    assert 'backbone.ephemeral_probe' in saved['buffer_state'] and 'backbone.ephemeral_probe' not in saved['model_state']
    original=saved['buffer_state']['backbone.ephemeral_probe'].clone()
    with torch.no_grad():
        case.model.backbone.ephemeral_probe.add_(5)
        next(case.model.parameters()).add_(1)
        next(iter(case.optimizer.state.values()))['exp_avg'].add_(2)
    case.model.eval();random.random();np.random.random();torch.rand(1)
    assert torch.equal(saved['buffer_state']['backbone.ephemeral_probe'],original)
    progress=_restore(case,saved)
    assert progress['cursor']==2 and progress['optimizer_updates']==1 and torch.equal(progress['parent_order'],case.order)
    assert _live_hash(case)==expected


def test_stochastic_two_updates_equals_one_plus_fresh_restore_one(monkeypatch):
    reference=_make(monkeypatch);_update(reference,0);_update(reference,1)
    expected=_capture(reference,4,2)
    interrupted=_make(monkeypatch);_update(interrupted,0);saved=_capture(interrupted,2,1)
    resumed=_make(monkeypatch);resumed.model.eval();random.random();np.random.random();torch.rand(3)
    assert _restore(resumed,saved)['cursor']==2
    _update(resumed,1);actual=_capture(resumed,4,2)
    assert actual['contract_sha256']==expected['contract_sha256']
    assert actual['optimizer_state_sha256']==expected['optimizer_state_sha256']
    assert actual['rng_state_sha256']==expected['rng_state_sha256']


def test_empty_loss_advances_cursor_without_any_optimizer_update(monkeypatch):
    case=_make(monkeypatch);initial=_capture(case,0,0)
    result=_loss(case,0,empty=True)
    assert result['loss'] is None and result['update_allowed'] is False
    after=_capture(case,2,0)
    assert after['model_state_sha256']==initial['model_state_sha256']
    assert after['optimizer_state_sha256']==initial['optimizer_state_sha256'] and not case.optimizer.state
    assert after['cursor']==2 and after['optimizer_updates']==0


def test_capture_rejects_actual_partial_backward(monkeypatch):
    case=_make(monkeypatch)
    original=case.model.backbone.ephemeral_probe.clone();case.model.backbone.ephemeral_probe[0]=float('nan')
    with pytest.raises(RuntimeError,match='BUFFER_TENSOR'):_capture(case,0,0)
    case.model.backbone.ephemeral_probe.copy_(original)
    _loss(case,0)['loss'].backward()
    with pytest.raises(RuntimeError,match='PARTIAL_GRADIENTS'):_capture(case,0,0)


@pytest.mark.parametrize('fault',['view','recipe','group_order','order','cursor','updates','buffer','buffer_dtype','model_nonfinite','semantic_buffer','rng_position','mode','schema','seal'])
def test_restore_rejects_corrupt_state_before_mutation(monkeypatch,fault):
    case=_make(monkeypatch);saved=_capture(case,0,0)
    if fault=='view':saved['binding']['feature_view_identity']['contract_sha256']='f'*64
    elif fault=='recipe':saved['binding']['recipe']['batch_size']=4
    elif fault=='group_order':saved['binding']['optimizer_group_parameter_names'][0].reverse()
    elif fault=='order':saved['parent_order']=saved['parent_order'].flip(0)
    elif fault=='cursor':saved['cursor']=1
    elif fault=='updates':saved['optimizer_updates']=5
    elif fault=='buffer':saved['buffer_state'].pop('backbone.ephemeral_probe')
    elif fault=='buffer_dtype':saved['buffer_state']['backbone.ephemeral_probe']=saved['buffer_state']['backbone.ephemeral_probe'].double()
    elif fault=='model_nonfinite':saved['model_state']['direction_head.weight'].fill_(float('nan'))
    elif fault=='semantic_buffer':
        name='backbone.input_norm_signal_center'
        saved['model_state'][name].add_(1);saved['buffer_state'][name].add_(1)
        saved['model_state_sha256']=state_owner.canonical_model_state_sha256(saved['model_state'])
        saved['buffer_state_sha256']=state_owner._tree_sha(saved['buffer_state'])
    elif fault=='rng_position':
        saved['rng_state']['numpy_position']=saved['rng_state']['numpy_keys'].numel()+1
        saved['rng_state_sha256']=state_owner._tree_sha(saved['rng_state'])
    elif fault=='mode':saved['module_training'].pop('')
    elif fault=='schema':saved['extra']=True
    elif fault=='seal':saved['contract_sha256']='0'*64
    if fault!='seal':_reseal(saved)
    before=_live_hash(case)
    with pytest.raises(RuntimeError,match='IMMUTABLE_BUFFER' if fault=='semantic_buffer' else None):_restore(case,saved)
    assert _live_hash(case)==before


@pytest.mark.parametrize('fault',['step','shape','nonfinite'])
def test_restore_rejects_corrupt_adamw_history_before_mutation(monkeypatch,fault):
    case=_make(monkeypatch);_update(case,0);saved=_capture(case,2,1)
    item=next(iter(saved['optimizer_state']['state'].values()))
    if fault=='step':item['step'].fill_(2)
    elif fault=='shape':item['exp_avg']=torch.zeros(1)
    elif fault=='nonfinite':item['exp_avg'].fill_(float('nan'))
    saved['optimizer_state_sha256']=state_owner._tree_sha(saved['optimizer_state']);_reseal(saved)
    before=_live_hash(case)
    with pytest.raises(RuntimeError):_restore(case,saved)
    assert _live_hash(case)==before


def test_restore_rejects_live_partial_backward(monkeypatch):
    case=_make(monkeypatch);saved=_capture(case,0,0);_loss(case,0)['loss'].backward()
    before=_live_hash(case)
    with pytest.raises(RuntimeError,match='PARTIAL_GRADIENTS'):_restore(case,saved)
    assert _live_hash(case)==before and any(p.grad is not None for p in case.model.parameters())


def _multipass_batch(case, ids, *, empty=False):
    """One synthetic batch whose actual length also covers the last partial batch."""
    count=len(ids)
    x=torch.tensor([[float(i+1), 2., 3., 4.] for i in ids])
    features={'seq_x':x[:,None,:], 'snap_x':x, 'ctx_cont':torch.zeros(count,1),
        'ctx_cat':torch.zeros(count,1,dtype=torch.int64)}
    for tf in ('m5','m15','h1','h4','d1'):
        features['seq_'+tf]=torch.zeros(count,1,1);features['gc_seq_'+tf]=torch.zeros(count,1,1)
    D=torch.tensor([100+i for i in ids],dtype=torch.int64)*MINUTE
    signs=torch.zeros(count,3,dtype=torch.int8) if empty else torch.tensor(
        [[1,-1,1] if i%2==0 else [-1,1,-1] for i in ids],dtype=torch.int8)
    output=case.model(**features)['logit_up']
    loss=direction.native_direction_binary_loss(output,sign=signs,observed=torch.ones(count,3,dtype=torch.bool),
        decision_ns=D,terminal_time_ns=D[:,None]+torch.tensor([1,5,15],dtype=torch.int64)[None,:]*MINUTE,cutoff_ns=200*MINUTE)
    if loss['update_allowed']:
        loss['loss'].backward();trainer._optimizer_step_with_finite_gradients(model=case.model,optimizer=case.optimizer)
    return loss['update_allowed']


def _multipass_case(monkeypatch):
    case=_make(monkeypatch)
    case.order=torch.arange(3,dtype=torch.int64)
    case.recipe=recipes.native_direction_research_recipe(batch_size=2,max_updates=2)
    return case


def _capture_multipass(case, cursor, updates, recipe):
    return state_owner.capture_native_direction_research_state(case.model,case.optimizer,recipe=recipe,
        feature_view_identity=_identity(),parent_order=case.order,cursor=cursor,optimizer_updates=updates,
        device=DEVICE,train_passes=3)


def test_three_pass_migration_partial_batch_and_stochastic_resume_are_exact(monkeypatch):
    continuation=recipes.native_direction_research_recipe(batch_size=2,max_updates=6)
    uninterrupted=_multipass_case(monkeypatch)
    for _ in range(3):
        assert _multipass_batch(uninterrupted,[0,1]);assert _multipass_batch(uninterrupted,[2])
    expected=_capture_multipass(uninterrupted,9,6,continuation)

    first=_multipass_case(monkeypatch)
    assert _multipass_batch(first,[0,1]);assert _multipass_batch(first,[2])
    legacy=_capture(first,3,2)
    migrated=state_owner.migrate_one_pass_terminal_to_three_pass(legacy,first.model,first.optimizer,
        original_recipe=first.recipe,continuation_recipe=continuation,feature_view_identity=_identity(),
        parent_order=first.order,device=DEVICE)
    assert migrated['cursor']==3 and migrated['optimizer_updates']==2 and migrated['train_passes']==3
    for key in ('model_state_sha256','buffer_state_sha256','optimizer_state_sha256','rng_state_sha256','module_training'):
        assert migrated[key]==legacy[key]
    assert _multipass_batch(first,[0,1])
    interrupted=_capture_multipass(first,5,3,continuation)

    resumed=_multipass_case(monkeypatch);random.random();np.random.random();torch.rand(5)
    progress=state_owner.restore_native_direction_research_state(interrupted,resumed.model,resumed.optimizer,
        recipe=continuation,feature_view_identity=_identity(),parent_order=resumed.order,device=DEVICE,train_passes=3)
    assert progress['cursor']==5 and progress['optimizer_updates']==3
    assert _multipass_batch(resumed,[2]);assert _multipass_batch(resumed,[0,1]);assert _multipass_batch(resumed,[2])
    actual=_capture_multipass(resumed,9,6,continuation)
    assert actual['contract_sha256']==expected['contract_sha256']


def test_three_pass_counts_empty_batches_and_rejects_cross_pass_partial_cursor(monkeypatch):
    case=_multipass_case(monkeypatch);continuation=recipes.native_direction_research_recipe(batch_size=2,max_updates=6)
    assert _multipass_batch(case,[0,1]);assert _multipass_batch(case,[2])
    legacy=_capture(case,3,2)
    state_owner.migrate_one_pass_terminal_to_three_pass(legacy,case.model,case.optimizer,original_recipe=case.recipe,
        continuation_recipe=continuation,feature_view_identity=_identity(),parent_order=case.order,device=DEVICE)
    assert not _multipass_batch(case,[0,1],empty=True)
    saved=_capture_multipass(case,5,2,continuation)
    assert saved['cursor']==5 and saved['optimizer_updates']==2
    with pytest.raises(RuntimeError,match='CURSOR'):_capture_multipass(case,4,2,continuation)
    with pytest.raises(RuntimeError,match='CURSOR'):_capture_multipass(case,10,2,continuation)


def test_three_pass_migration_rejects_changed_recipe_and_incomplete_legacy_before_mutation(monkeypatch):
    case=_multipass_case(monkeypatch);assert _multipass_batch(case,[0,1])
    incomplete=_capture(case,2,1);before=_live_hash(case)
    good=recipes.native_direction_research_recipe(batch_size=2,max_updates=6)
    with pytest.raises(RuntimeError,match='MIGRATION_NOT_FULL_ONE_PASS'):
        state_owner.migrate_one_pass_terminal_to_three_pass(incomplete,case.model,case.optimizer,original_recipe=case.recipe,
            continuation_recipe=good,feature_view_identity=_identity(),parent_order=case.order,device=DEVICE)
    assert _live_hash(case)==before
    wrong=recipes.native_direction_research_recipe(batch_size=2,max_updates=5)
    with pytest.raises(RuntimeError,match='MIGRATION_RECIPE'):
        state_owner.migrate_one_pass_terminal_to_three_pass(incomplete,case.model,case.optimizer,original_recipe=case.recipe,
            continuation_recipe=wrong,feature_view_identity=_identity(),parent_order=case.order,device=DEVICE)
    assert _live_hash(case)==before
