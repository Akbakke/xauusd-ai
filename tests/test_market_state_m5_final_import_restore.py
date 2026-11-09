"""Exact same-loss FINAL import restore; synthetic decoder, no data or fit."""
import hashlib

import pytest
import torch

from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
from gx1.contracts import entry_native_direction_research_state_v1 as digests
from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
from tests.test_market_state_ta_dedicated_m5_binding_v1 import (
    COMMIT, DEVICE, ORDER, _capture, _model, _seed)


def _case(monkeypatch, tmp_path, fault=None):
    _seed()
    source = _model(monkeypatch, True).train()
    numerical = fit.ta_sf_numerical_recipe(dedicated_m5_side=True, m5_equal_observation=True)
    old = (fit.ta_sf_numerical_recipe(dedicated_m5_side=True)
           if fault == 'old_numerical' else numerical)
    optimizer, _ = state.build_fold1_fit_optimizer(source, numerical_recipe=old)
    saved = _capture(source, optimizer, old, 0)
    if fault == 'progress':
        saved['cursor'] = 16
    if fault == 'nonempty':
        saved['optimizer_state']['state'][0] = {'step': torch.tensor(1.)}
    path = tmp_path / 'FINAL_IMPORT.pt'
    torch.save(saved, path)
    reference = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    target = _model(monkeypatch, True).train()
    opt, _ = state.build_fold1_fit_optimizer(target, numerical_recipe=numerical)
    origin = dict(imported_checkpoint=reference, source_commit=COMMIT,
        model_state_sha256=saved['model_state_sha256'],
        rng_state_sha256=saved['rng_state_sha256'], parent_order_sha256=digests._tree_sha(ORDER))
    launch = dict(imported_origin=origin, max_checkpoint_bytes=8 * 1024**2,
        numerical_recipe=numerical, expected_initial_model_state_sha256=saved['model_state_sha256'])
    counters = dict(resume_state_load_calls=0, resume_state_load_returns=0)
    args = dict(launch=launch, model=target, optimizer=opt,
        view_identity=target.state_model.feature_view_identity, order=ORDER,
        source_commit='b' * 40, device=DEVICE, counters=counters)
    return saved, args, path


def test_same_loss_import_changes_only_source_and_seal(monkeypatch, tmp_path):
    saved, args, path = _case(monkeypatch, tmp_path)
    original = path.read_bytes()
    monkeypatch.setattr(state, 'rebind_m5_equal_observation_initial_state',
        lambda *a, **k: pytest.fail('already equal-loss checkpoint must not rebind loss'))
    proof = runner._ta_m5_restore_final_import(**args)
    actual = state.capture_fold1_fit_state(args['model'], args['optimizer'],
        numerical_recipe=args['launch']['numerical_recipe'], feature_view_identity=args['view_identity'],
        parent_order=ORDER, cursor=0, optimizer_updates=0, skipped_batches=0,
        source_commit=args['source_commit'], device=DEVICE)
    changed = {key for key in saved if digests._tree_sha(saved[key]) != digests._tree_sha(actual[key])}
    assert changed == {'source_commit', 'contract_sha256'}
    assert path.read_bytes() == original
    assert proof['all_state_except_source_and_seal_unchanged']
    assert proof['numerical_recipe_unchanged'] and proof['optimizer_steps_performed'] == 0
    assert args['counters'] == dict(resume_state_load_calls=1, resume_state_load_returns=1)
    assert not args['optimizer'].state


@pytest.mark.parametrize('fault,reason', [
    ('old_numerical', 'EXACT_BINDING'), ('progress', 'ZERO_PROGRESS'),
    ('nonempty', 'EMPTY_OPTIMIZER'), ('rng', 'MODEL_AND_RNG'),
    ('model', 'MODEL_AND_RNG'), ('source', 'EXACT_BINDING'),
    ('order', 'PARENT_ORDER')])
def test_wrong_import_is_rejected(monkeypatch, tmp_path, fault, reason):
    saved, args, path = _case(monkeypatch, tmp_path, fault)
    if fault == 'rng':
        args['launch']['imported_origin']['rng_state_sha256'] = '0' * 64
    if fault == 'model':
        args['launch']['expected_initial_model_state_sha256'] = '0' * 64
    if fault == 'source':
        args['launch']['imported_origin']['source_commit'] = 'c' * 40
    if fault == 'order':
        args['order'] = ORDER.flip(0)
    before = {name: value.clone() for name, value in args['model'].state_dict().items()}
    with pytest.raises(RuntimeError, match=reason):
        runner._ta_m5_restore_final_import(**args)
    assert all(torch.equal(value, args['model'].state_dict()[key]) for key, value in before.items())
    assert not args['optimizer'].state
