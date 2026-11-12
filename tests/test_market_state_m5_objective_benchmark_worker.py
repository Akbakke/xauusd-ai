"""Prove the repeated-pass worker never applies AdamW to empty M5 support."""
from types import SimpleNamespace
import pytest

from gx1.audit.run_market_state_ta_m5_objective_benchmark_v1 import _apply_training_update


class Loss:
    def __init__(self, calls):
        self.calls = calls
    def detach(self):
        return 0.25
    def backward(self):
        self.calls.append('backward')


@pytest.mark.parametrize('skip', [True, False])
def test_empty_support_visits_without_adamw_or_weight_decay(skip):
    calls = []
    model = object()
    optimizer = SimpleNamespace(zero_grad=lambda **kw: calls.append(('zero', kw)))
    def step(**kwargs):
        assert kwargs == {'model':model, 'optimizer':optimizer}
        calls.append('adamw')
        return 0.75
    trainer = SimpleNamespace(_optimizer_step_with_finite_gradients=step)
    counters = dict.fromkeys(('backwards','backward_returns','optimizer_step_calls',
        'optimizer_step_returns','optimizer_updates','skipped_batches','batch_visits'), 0)
    progress = dict(batch_visits=0, optimizer_updates=0, skipped_batches=0)
    result = _apply_training_update({'skip_update':skip,'loss':Loss(calls)},
        model=model, optimizer=optimizer, trainer=trainer, counters=counters, progress=progress)
    assert calls == ([] if skip else ['backward','adamw']) + [('zero',{'set_to_none':True})]
    assert progress == dict(batch_visits=1,optimizer_updates=int(not skip),skipped_batches=int(skip))
    assert counters['batch_visits'] == 1
    assert counters['optimizer_step_calls'] == counters['optimizer_step_returns'] == int(not skip)
    assert counters['backwards'] == counters['backward_returns'] == int(not skip)
    assert result == dict(loss=.25,gradient_norm=None if skip else .75,skip_update=skip)


def test_failed_backward_does_not_claim_optimizer_progress():
    class FailedLoss(Loss):
        def backward(self):
            raise RuntimeError('observed backward failure')
    calls=[]
    counters=dict.fromkeys(('backwards','backward_returns','optimizer_step_calls',
        'optimizer_step_returns','optimizer_updates','skipped_batches','batch_visits'),0)
    progress=dict(batch_visits=0,optimizer_updates=0,skipped_batches=0)
    with pytest.raises(RuntimeError,match='observed backward failure'):
        _apply_training_update({'skip_update':False,'loss':FailedLoss(calls)},model=None,
            optimizer=SimpleNamespace(zero_grad=lambda **kw:calls.append('zero')),
            trainer=SimpleNamespace(_optimizer_step_with_finite_gradients=lambda **kw:calls.append('adamw')),
            counters=counters,progress=progress)
    assert calls == []
    assert counters['backwards'] == 1 and counters['backward_returns'] == 0
    assert progress == dict(batch_visits=0,optimizer_updates=0,skipped_batches=0)
