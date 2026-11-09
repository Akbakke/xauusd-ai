"""Strict score-only restore, using one synthetic fitted step as the terminal fixture."""
import hashlib
from types import SimpleNamespace

import pytest
import torch

from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
from tests.test_market_state_ta_dedicated_m5_binding_v1 import (
    COMMIT, DEVICE, ORDER, _capture, _model, _seed, _update)


def _case(monkeypatch, tmp_path):
    _seed()
    model = _model(monkeypatch, True).train()
    numerical = fit.ta_sf_numerical_recipe(dedicated_m5_side=True, m5_equal_observation=True)
    optimizer, _ = state.build_fold1_fit_optimizer(model, numerical_recipe=numerical)
    _update(model, optimizer, 0)
    optimizer.zero_grad(set_to_none=True)
    saved = _capture(model, optimizer, numerical, 1)
    path = tmp_path / 'FINAL_FOLD1_FIT_STATE.pt'
    torch.save(saved, path)
    ref = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    target = _model(monkeypatch, True).train()
    target_opt, _ = state.build_fold1_fit_optimizer(target, numerical_recipe=numerical)
    # Only the terminal fixture's progress is shortened; real origin routing
    # separately requires the immutable 29574-row,1849-update source receipt.
    monkeypatch.setattr(runner, 'scope', SimpleNamespace(FIT_ROWS=16, MAX_UPDATES=1))
    args = dict(launch={'scoring_recovery_origin': {
        'final_checkpoint': ref, 'source_commit': COMMIT},
        'max_checkpoint_bytes': 8 * 1024**2, 'numerical_recipe': numerical},
        model=target, optimizer=target_opt,
        view_identity=target.state_model.feature_view_identity,
        order=ORDER, device=DEVICE,
        counters={'resume_state_load_calls': 0, 'resume_state_load_returns': 0})
    return saved, args, path


def test_fitted_checkpoint_restored_without_initial_fit_or_state_change(monkeypatch, tmp_path):
    saved, args, path = _case(monkeypatch, tmp_path)
    original = path.read_bytes()
    identities = [id(p) for p in args['model'].parameters()]
    monkeypatch.setattr(args['optimizer'], 'step',
        lambda *a, **k: pytest.fail('scoring restore must never update'))
    proof = runner._ta_m5_restore_scoring_checkpoint(**args)
    actual = state.capture_fold1_fit_state(args['model'], args['optimizer'],
        numerical_recipe=args['launch']['numerical_recipe'],
        feature_view_identity=args['view_identity'], parent_order=ORDER,
        cursor=16, optimizer_updates=1, skipped_batches=0,
        source_commit=COMMIT, device=DEVICE)
    assert actual['contract_sha256'] == saved['contract_sha256'] == proof['contract_sha256']
    assert proof['optimizer_steps_performed'] == proof['initial_forward_calls'] == 0
    assert args['optimizer'].state
    assert identities == [id(p) for p in args['model'].parameters()]
    assert path.read_bytes() == original
    assert args['counters'] == {'resume_state_load_calls': 1, 'resume_state_load_returns': 1}


@pytest.mark.parametrize('fault,reason', [
    ('progress', 'FULL_PROGRESS'), ('source', 'EXACT_BINDING'),
    ('order', 'PARENT_ORDER'), ('corrupt', 'CHECKPOINT_LOAD_SHA_SIZE'),
    ('numerical', 'EXACT_BINDING')])
def test_invalid_scoring_checkpoint_is_rejected(monkeypatch, tmp_path, fault, reason):
    saved, args, path = _case(monkeypatch, tmp_path)
    if fault == 'progress':
        monkeypatch.setattr(runner.scope, 'MAX_UPDATES', 1849)
    elif fault == 'source':
        args['launch']['scoring_recovery_origin']['source_commit'] = 'b' * 40
    elif fault == 'order':
        args['order'] = ORDER.flip(0)
    elif fault == 'corrupt':
        path.write_bytes(path.read_bytes() + b'changed')
    elif fault == 'numerical':
        args['launch']['numerical_recipe'] = fit.ta_sf_numerical_recipe(dedicated_m5_side=True)
    before = {k: v.clone() for k, v in args['model'].state_dict().items()}
    with pytest.raises((ValueError, RuntimeError), match=reason):
        runner._ta_m5_restore_scoring_checkpoint(**args)
    assert all(torch.equal(v, args['model'].state_dict()[k]) for k, v in before.items())
    assert not args['optimizer'].state
