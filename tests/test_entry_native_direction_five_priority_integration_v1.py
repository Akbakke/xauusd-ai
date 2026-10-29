"""Synthetic objective identity, unchanged initialization, and exact resume."""
import copy
import pytest
import torch
from gx1.contracts import entry_native_direction_dense_v1 as dense
from gx1.contracts import entry_native_direction_five_priority_v1 as priority
from gx1.contracts import entry_model_native_train_recipe_v1 as recipes
from gx1.contracts import entry_native_direction_research_state_v1 as state
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
from tests import test_entry_native_direction_research_v1 as fixture
from tests import test_entry_native_direction_research_state_v1 as state_fixture

MINUTE=60*10**9


def test_priority_preserves_initial_parameters_predictions_and_rng(monkeypatch):
    monkeypatch.setattr(EntryV10CtxHybridTransformer,'forward',fixture._native_forward_spy)
    torch.manual_seed(503); old=dense.DenseNativeEntryDirectionResearch(fixture._native_shell()); old_rng=torch.get_rng_state().clone()
    torch.manual_seed(503); new=dense.DenseNativeEntryDirectionResearch(fixture._native_shell(),five_minute_priority=True)
    assert torch.equal(old_rng,torch.get_rng_state())
    assert set(old.state_dict())==set(new.state_dict())
    assert all(torch.equal(value,new.state_dict()[key]) for key,value in old.state_dict().items())
    assert torch.equal(old(**fixture._inputs())['logit_up'],new(**fixture._inputs())['logit_up'])
    assert old.research_objective==dense.dense_direction_research_contract()
    assert new.research_objective==priority.five_priority_direction_contract()
    assert not old.five_minute_priority and new.five_minute_priority
    with pytest.raises(RuntimeError):dense.DenseNativeEntryDirectionResearch(fixture._native_shell(),five_minute_priority=1)
    new._five_minute_priority=False
    with pytest.raises(RuntimeError):new(**fixture._inputs())


def test_priority_recipe_is_separate_and_old_recipe_unchanged():
    old=recipes.native_direction_research_recipe(batch_size=2,max_updates=4,dense_minutes=True)
    candidate=recipes.native_direction_research_recipe(batch_size=2,max_updates=4,dense_minutes=True,five_minute_priority=True)
    assert candidate['research_objective_sha256']==priority.five_priority_direction_contract()['contract_sha256']
    assert candidate['five_minute_priority'] is True and recipes.require_native_direction_research_recipe(candidate)==candidate
    assert recipes.native_direction_research_recipe(batch_size=2,max_updates=4,dense_minutes=True)==old
    for kwargs in ({'five_minute_priority':True}, {'five_minute_priority':1,'dense_minutes':True}, {'five_minute_priority':True,'dense_minutes':True,'signed_bps':True}):
        with pytest.raises(RuntimeError):recipes.native_direction_research_recipe(batch_size=2,max_updates=4,**kwargs)


def _priority_case(monkeypatch):
    base=state_fixture._make(monkeypatch)
    model=dense.DenseNativeEntryDirectionResearch(base.model.backbone,five_minute_priority=True)
    identity=state_fixture._identity();identity['research_objective_sha256']=priority.five_priority_direction_contract()['contract_sha256'];model.feature_view_identity=identity
    recipe=recipes.native_direction_research_recipe(batch_size=2,max_updates=4,dense_minutes=True,five_minute_priority=True)
    optimizer,binding=state.build_native_direction_research_optimizer(model,recipe=recipe)
    model.train();state_fixture._seed(recipe['seed'])
    return model,optimizer,recipe,identity,base.order,binding


def _update(parts,step):
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    model,optimizer,*_=parts
    z=model(**state_fixture._features(step))['logit_up']
    decisions=torch.tensor([100+2*step,101+2*step],dtype=torch.int64)*MINUTE
    signs=torch.where(torch.arange(60)[None,:]%2==0,1,-1).to(torch.int8).repeat(2,1)
    loss=priority.five_priority_direction_binary_loss(z,sign=signs,observed=torch.ones((2,60),dtype=torch.bool),decision_ns=decisions,terminal_time_ns=decisions[:,None]+torch.arange(1,61)[None,:]*MINUTE,cutoff_ns=200*MINUTE)
    loss['loss'].backward();trainer._optimizer_step_with_finite_gradients(model=model,optimizer=optimizer)


def _capture(parts,cursor,updates):
    model,optimizer,recipe,identity,order,_=parts
    return state.capture_native_direction_research_state(model,optimizer,recipe=recipe,feature_view_identity=identity,parent_order=order,cursor=cursor,optimizer_updates=updates,device=torch.device('cpu'))


def test_priority_stochastic_optimizer_resume_matches_uninterrupted(monkeypatch):
    ref=_priority_case(monkeypatch);_update(ref,0);_update(ref,1);expected=_capture(ref,4,2)
    stopped=_priority_case(monkeypatch);_update(stopped,0);saved=_capture(stopped,2,1)
    resumed=_priority_case(monkeypatch);model,opt,recipe,identity,order,_=resumed
    state.restore_native_direction_research_state(saved,model,opt,recipe=recipe,feature_view_identity=identity,parent_order=order,device=torch.device('cpu'))
    _update(resumed,1);actual=_capture(resumed,4,2)
    assert actual['contract_sha256']==expected['contract_sha256']
    assert actual['optimizer_state_sha256']==expected['optimizer_state_sha256']
    assert actual['rng_state_sha256']==expected['rng_state_sha256']


def test_prefix_trainer_dispatches_only_priority_objective(monkeypatch):
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as window
    model,opt,*_=_priority_case(monkeypatch)
    monkeypatch.setattr(window,'_forward',lambda model,features,**kwargs:model(**features)['logit_up'])
    calls=[]; original=priority.five_priority_direction_binary_loss
    def observed(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(priority,'five_priority_direction_binary_loss',observed)
    d=torch.tensor([100,101],dtype=torch.int64)*MINUTE
    target={'sign':torch.ones((2,60),dtype=torch.int8),'observed':torch.ones((2,60),dtype=torch.bool),'decision_ns':d,'terminal_time_ns':d[:,None]+torch.arange(1,61)[None,:]*MINUTE}
    counters={key:0 for key in ('backward_calls','backward_returns','optimizer_step_calls','optimizer_step_returns','empty_batches')}
    record,updated=window._train_batch(model,opt,state_fixture._features(0),target,cutoff=200*MINUTE,counters=counters,ceiling=1,torch=torch,diagnostics=False)
    assert calls==[1] and updated and record['eligible_cells']==120 and counters['optimizer_step_returns']==1


def test_private_step_dispatches_priority_objective(monkeypatch):
    from contextlib import nullcontext
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    model,opt,*_=_priority_case(monkeypatch)
    monkeypatch.setattr(trainer,'_training_mha_fastpath_context',lambda *_:nullcontext())
    monkeypatch.setattr(trainer,'_model_forward_fp32',lambda model,**features:model(**features))
    monkeypatch.setattr(torch.backends.mha,'get_fastpath_enabled',lambda:False)
    monkeypatch.setattr(common,'_gradient_evidence',lambda *_:{'synthetic':True})
    calls=[];original=priority.five_priority_direction_binary_loss
    def observed(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(priority,'five_priority_direction_binary_loss',observed)
    d=torch.tensor([100,101],dtype=torch.int64)*MINUTE
    target={'sign':torch.ones((2,60),dtype=torch.int8),'observed':torch.ones((2,60),dtype=torch.bool),'decision_ns':d,'terminal_time_ns':d[:,None]+torch.arange(1,61)[None,:]*MINUTE}
    counters={key:0 for key in ('forwards','forward_returns','backwards','backward_returns','optimizer_step_calls','optimizer_updates')}
    report=common._step(model,opt,state_fixture._features(0),target,cutoff=200*MINUTE,counters=counters,torch=torch)
    assert calls==[1] and report['counts']['eligible_cells']==120 and counters['optimizer_updates']==1


def test_unweighted_checkpoint_cannot_restore_into_priority_objective(monkeypatch):
    from tests import test_entry_native_direction_dense_v1 as old
    baseline=old._state_case(monkeypatch);old._dense_update(baseline,0);saved=old._dense_capture(baseline,2,1)
    candidate=_priority_case(monkeypatch);model,opt,recipe,identity,order,_=candidate
    with pytest.raises(RuntimeError):
        state.restore_native_direction_research_state(saved,model,opt,recipe=recipe,feature_view_identity=identity,parent_order=order,device=torch.device('cpu'))
