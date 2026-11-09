"""Fullfit imports the original zero-step state; no market input or forward."""
import hashlib

import pytest
import torch

from gx1.contracts import entry_native_direction_research_state_v1 as digests
from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
from tests.test_market_state_ta_dedicated_m5_binding_v1 import (
    COMMIT, DEVICE, ORDER, _capture, _model, _seed,
)


def _case(monkeypatch, tmp_path, fault=None):
    _seed()
    source = _model(monkeypatch, True)
    source.train()
    old = fit.ta_sf_numerical_recipe(dedicated_m5_side=True)
    optimizer, _ = state.build_fold1_fit_optimizer(source, numerical_recipe=old)
    if fault == 'mode':
        source.future_m5_side_head.eval()
    saved = _capture(source, optimizer, old, 0)
    path = tmp_path / 'ORIGINAL_IMPORT.pt'
    torch.save(saved, path)
    ref = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    target = _model(monkeypatch, True)
    target.train()
    numerical = fit.ta_sf_numerical_recipe(
        dedicated_m5_side=True, m5_equal_observation=True)
    opt, _ = state.build_fold1_fit_optimizer(target, numerical_recipe=numerical)
    origin = dict(imported_checkpoint=ref, source_commit=COMMIT,
        model_state_sha256=saved['model_state_sha256'],
        rng_state_sha256=saved['rng_state_sha256'],
        parent_order_sha256=digests._tree_sha(ORDER))
    launch = dict(imported_origin=origin, max_checkpoint_bytes=8 * 1024 * 1024,
        numerical_recipe=numerical,
        expected_initial_model_state_sha256=saved['model_state_sha256'])
    counters = dict(resume_state_load_calls=0, resume_state_load_returns=0)
    kwargs = dict(launch=launch, model=target, optimizer=opt,
        view_identity=target.state_model.feature_view_identity, order=ORDER,
        source_commit='b' * 40, device=DEVICE, counters=counters)
    return saved, kwargs, path


def test_fullfit_import_preserves_original_state_without_reset_or_migration(monkeypatch, tmp_path):
    saved, kwargs, path = _case(monkeypatch, tmp_path)
    before_file = path.read_bytes()
    parameter_ids = [id(p) for p in kwargs['model'].parameters()]
    monkeypatch.setattr(state, 'copy_legacy_weights_to_dedicated_m5',
        lambda *a, **k: pytest.fail('legacy head migration would erase imported heads'))
    proof = runner._ta_m5_rebind_import(**kwargs)
    actual = state.capture_fold1_fit_state(
        kwargs['model'], kwargs['optimizer'],
        numerical_recipe=kwargs['launch']['numerical_recipe'],
        feature_view_identity=kwargs['view_identity'], parent_order=ORDER,
        cursor=0, optimizer_updates=0, skipped_batches=0,
        source_commit=kwargs['source_commit'], device=DEVICE)
    assert {key for key in saved
            if digests._tree_sha(saved[key]) != digests._tree_sha(actual[key])} == {
        'binding', 'source_commit', 'contract_sha256'}
    assert path.read_bytes() == before_file
    assert parameter_ids == [id(p) for p in kwargs['model'].parameters()]
    assert kwargs['counters'] == dict(resume_state_load_calls=1, resume_state_load_returns=1)
    assert proof['optimizer_steps_performed'] == 0
    assert not kwargs['optimizer'].state


@pytest.mark.parametrize('fault, reason, loads', [
    ('order', 'IMPORTED_PARENT_ORDER', 0),
    ('file', 'CHECKPOINT_LOAD_SHA_SIZE', 0),
    ('rng', 'IMPORTED_MODEL_AND_RNG', 1),
    ('model', 'IMPORTED_MODEL_AND_RNG', 1),
    ('mode', 'IMPORTED_EMPTY_OPTIMIZER_AND_TRAIN_MODES', 1),
])
def test_fullfit_import_rejects_changed_origin_or_modes(monkeypatch, tmp_path, fault, reason, loads):
    saved, kwargs, path = _case(monkeypatch, tmp_path, fault)
    launch = kwargs['launch']
    if fault == 'order':
        kwargs['order'] = ORDER.flip(0)
    elif fault == 'file':
        with path.open('ab') as handle:
            handle.write(b'changed')
    elif fault == 'rng':
        launch['imported_origin']['rng_state_sha256'] = '0' * 64
    elif fault == 'model':
        launch['expected_initial_model_state_sha256'] = '0' * 64
    with pytest.raises(RuntimeError, match=reason):
        runner._ta_m5_rebind_import(**kwargs)
    assert kwargs['counters']['resume_state_load_calls'] == loads
    assert not kwargs['optimizer'].state
