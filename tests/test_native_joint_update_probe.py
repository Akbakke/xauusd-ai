"""No-step diagnostic checked against real native clipping/AdamW and shared gradients."""
import copy
import pytest
import torch
from gx1.scripts import run_unified_exit_random_access_full_train_v1 as runner
from tests.test_native_entry_gradient_diagnostic import Small, pair


def optimizer_for(model):
    return torch.optim.AdamW([
        {'params':[p for n,p in model.named_parameters() if not n.startswith('task_log_variances.')], 'weight_decay':0.07},
        {'params':list(model.task_log_variances.parameters()), 'weight_decay':0.0}], lr=.001)


@pytest.mark.parametrize('scale', [.001, 100.])
def test_readonly_adam_estimate_matches_actual_native_step_with_history(scale):
    torch.manual_seed(13)
    model=Small().double();optimizer=optimizer_for(model)
    for _ in range(3):
        for p in model.parameters():p.grad=torch.randn_like(p)
        runner.trainer._optimizer_step_with_finite_gradients(model=model,optimizer=optimizer)
    before=copy.deepcopy(model.state_dict());saved=copy.deepcopy(optimizer.state_dict())
    gradients={n:scale*torch.randn_like(p) for n,p in model.named_parameters()}
    gradients['family_tf_context_gate.bias']=None
    delta,_=runner._joint_probe_adam_delta(model=model,optimizer=optimizer,saved_optimizer=saved,gradients=gradients)
    for n,p in model.named_parameters():
        torch.testing.assert_close(p,before[n],atol=0,rtol=0)
        p.grad=gradients[n]
    runner.trainer._optimizer_step_with_finite_gradients(model=model,optimizer=optimizer)
    for n,p in model.named_parameters():torch.testing.assert_close(p-before[n],delta[n],atol=1e-14,rtol=1e-9)
    assert all(int(s['step'])==3 for s in saved['state'].values())
    assert torch.equal(model.family_tf_context_gate.bias,before['family_tf_context_gate.bias'])


@pytest.mark.parametrize('fault', [None, 'target', 'parity', 'forward'])
def test_joint_probe_covers_shared_exit_and_aux_preserves_state_and_cleans_errors(pair,monkeypatch,fault):
    class JointSmall(Small):
        def __init__(self):
            super().__init__();self.fuse=torch.nn.Linear(4,4);self.exit_gru=torch.nn.GRU(4,4,batch_first=True);self.exit_dropout=torch.nn.Dropout(.5);self.head_exit_action=torch.nn.Linear(4,2)
        def forward(self,seq_x,snap_x,**kw):
            h=torch.tanh(self.fuse(seq_x)+self.family_tf_context_gate(seq_x)+self.family_tf_token_gate(snap_x))
            return {'entry_action_q_bps':self.head_entry_action_q(self.entry_q_joint_in(h)), 'aux':h,
                    runner.trainer.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY:h}
        def forward_exit_random_access_batch(self,token):
            assert not self.training and not self.exit_dropout.training
            if torch.is_grad_enabled():
                assert self.exit_gru.training  # Simulate the cuDNN backward requirement on CPU.
                if fault == 'forward': raise RuntimeError('SIMULATED_FORWARD_FAILURE')
            else: assert not self.exit_gru.training
            h=self.exit_gru(token.unsqueeze(1))[0][:,0]
            q=self.head_exit_action(self.exit_dropout(h))
            if fault == 'parity' and torch.is_grad_enabled(): q=q+1
            return {'exit_action_q_bps':q}
    model=JointSmall().eval();optimizer=optimizer_for(model);batch=pair['batch'];batch['entry_row_index']=torch.arange(16)
    before=copy.deepcopy(model.state_dict());saved={'model_state':before,'target_model_state':before,'optimizer_state':copy.deepcopy(optimizer.state_dict())}
    monkeypatch.setattr(runner.trainer,'_copy_frozen_prefix_reference_model',lambda m:copy.deepcopy(m).eval().requires_grad_(False))
    def exit_loss(**kw):
        token=kw['entry_decision_representations'].detach().clone().requires_grad_(True)
        q=model.forward_exit_random_access_batch(token=token)['exit_action_q_bps'];raw=(q-.5).square().mean()
        (torch.exp(-model.task_log_variances['unified_exit_action'])*raw).backward()
        stats={'raw_loss':float(raw.detach()),'random_access_online_forward_calls':1,'random_access_target_forward_calls':1,
               'random_access_backward_calls':1,'random_access_transition_count':16}
        return token.grad,stats,pair['target']+(1 if fault=='target' else 0),pair['valid']
    monkeypatch.setattr(runner.trainer,'_train_unified_exit_full_population',exit_loss)
    with torch.inference_mode():expected=model(batch['seq_x'],batch['snap_x'])['entry_action_q_bps']
    args=dict(model=model,optimizer=optimizer,saved_state=saved,batch=batch,target=pair['target'],valid=pair['valid'],expected=expected,dataset=None,device=torch.device('cpu'))
    if fault:
        error={'target':'TARGET_PARITY_INVALID','parity':'EXIT_INFERENCE_PARITY_INVALID','forward':'SIMULATED_FORWARD_FAILURE'}[fault]
        with pytest.raises(RuntimeError,match=error):runner._joint_update_probe(**args)
    else:
        result=runner._joint_update_probe(**args)
        assert result['optimizer_steps']==0 and result['native_accumulation_all_parameter_gradients_matched']
        assert result['model_forwards']==6 and result['exit_inference_parity']['actions_identical']
        assert result['exit_inference_parity']['max_abs_difference_bps']==0
        assert result['gradients']['fuse']['norms']['exit']>0
        assert result['gradients']['fuse']['norms']['auxiliary']>0
        assert result['gradients']['exit']['norms']['exit']>0
        assert result['gradients']['exit']['norms']['entry']==0
        assert result['adam_directions']['joint']['preclip_norms']['prediction']>0
    assert all(p.grad is None for p in model.parameters()) and not optimizer.state
    assert not any(m.training for m in model.modules())
    assert 'forward_exit_random_access_batch' not in model.__dict__
    for n,v in before.items():torch.testing.assert_close(v,model.state_dict()[n],atol=0,rtol=0)
