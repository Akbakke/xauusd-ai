"""Actual small state restore and decoder-hook checks; no genuine market input."""
import copy
import hashlib

import pytest
import torch

from gx1.scripts import run_market_state_native_F_private_window_v1 as runner
from gx1.scripts import run_entry_native_direction_research_window_v1 as common
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
from gx1.contracts import entry_native_direction_research_state_v1 as old
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from tests.test_market_state_time_query_dedicated_m5_side_v1 import _model, _inputs
from tests.test_market_state_ta_dedicated_m5_binding_v1 import (
    _model as state_model, _seed, _update, _capture, ORDER, DEVICE, COMMIT,
)


def _forward(model):
    hidden, query = _inputs()
    return {'state_activity_side_logits': model.state_model.decode_raw_query_grid(
        hidden, query[:, :1])[:, 0], 'future_activity_side_logits': model.decode_native_F_raw(hidden)}


def test_m5_clone_parity_reuses_existing_calls_and_removes_hooks():
    model = _model(dedicated_m5_side=True)
    calls = []
    def forward():
        calls.append(1)
        return _forward(model)
    outcome, report = runner._m5_baseline_parity(model, forward, atol=2e-6, rtol=1e-6)
    assert calls == [1] and report['extra_native_forwards'] == 0
    assert all(v['unchanged_coordinates_bit_equal'] and v['m5_side_within_fp32_tolerance']
               for v in report['tasks'].values())
    assert not model.state_model.query_head._forward_hooks
    assert not model.future_head._forward_hooks
    assert set(outcome) == {'state_activity_side_logits', 'future_activity_side_logits'}


@pytest.mark.parametrize('failure', ['side_changed', 'activity_changed', 'forward_raises'])
def test_parity_fails_closed_and_always_removes_hooks(failure):
    model = _model(dedicated_m5_side=True)
    if failure == 'side_changed':
        with torch.no_grad():
            model.future_m5_side_head.bias.add_(0.01)
    def forward():
        if failure == 'forward_raises':
            raise RuntimeError('sentinel')
        out = _forward(model)
        if failure == 'activity_changed':
            out['state_activity_side_logits'] = out['state_activity_side_logits'].clone()
            out['state_activity_side_logits'][:, 0, 0] += 0.01
        return out
    with pytest.raises(RuntimeError, match='sentinel|M5_CLONED_OUTPUT_PARITY'):
        runner._m5_baseline_parity(model, forward, atol=2e-6, rtol=1e-6)
    assert not model.state_model.query_head._forward_hooks
    assert not model.future_head._forward_hooks


def _origin_case(tmp_path, monkeypatch):
    _seed()
    model = state_model(monkeypatch)
    numerical = fit.ta_sf_numerical_recipe()
    optimizer, _ = state.build_fold1_fit_optimizer(model, numerical_recipe=numerical)
    _update(model, optimizer, 0)
    saved = _capture(model, optimizer, numerical, 1)
    path = tmp_path / 'synthetic_legacy.pt'
    torch.save(saved, path)
    original_view = copy.deepcopy(model.state_model.feature_view_identity)
    refs = {'recipe': {'numerical_recipe': numerical,
                       'expected_view_sha256': original_view['contract_sha256']},
            'view': original_view}
    monkeypatch.setattr(runner.scope, '_read', lambda ref, label: copy.deepcopy(refs[ref['path']]))
    monkeypatch.setattr(views, 'build_native_direction_research_model',
                        lambda *args, **kwargs: state_model(monkeypatch).state_model)
    monkeypatch.setattr(trainer, '_set_deterministic', lambda *args: _seed())
    origin = {'original_recipe': {'path': 'recipe'},
              'original_candidate_view': {'path': 'view'},
              'original_checkpoint': {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
              'source_commit': COMMIT, 'parent_order_sha256': old._tree_sha(ORDER),
              'model_state_sha256': saved['model_state_sha256'],
              'cursor': 16, 'optimizer_updates': 1, 'skipped_batches': 0}
    counters = {k: 0 for k in ('model_constructions','model_construction_returns',
        'optimizer_constructions','resume_state_load_calls','resume_state_load_returns')}
    receipts = {'fold1': {'normal': {}, 'profile': {}}, 'constructor': {}}
    return origin, receipts, counters, saved


def test_origin_restores_full_legacy_state_before_returning_weights(tmp_path, monkeypatch):
    origin, receipts, counts, expected = _origin_case(tmp_path, monkeypatch)
    model, rng, evidence = runner._restore_m5_legacy_origin(
        origin=origin, receipts=receipts, order=ORDER, device=DEVICE,
        counters=counts, maximum=1024**2)
    assert all(v == 1 for v in counts.values())
    assert evidence['strict_full_restore']
    assert evidence['full_state_contract_sha256'] == expected['contract_sha256']
    assert evidence['optimizer_state_sha256'] == expected['optimizer_state_sha256']
    assert old._tree_sha(rng) == expected['rng_state_sha256']
    assert evidence['optimizer_history_transferred_to_candidate'] is False
    assert model.state_model._dedicated_m5_side is False


def test_origin_rejects_checkpoint_with_invalid_optimizer_binding(tmp_path, monkeypatch):
    origin, receipts, counts, expected = _origin_case(tmp_path, monkeypatch)
    # Hash the tampered artifact so that the full-state validator, not IO hashing,
    # must reject its invalid optimizer/model binding before transfer.
    expected['binding']['active_parameter_names'] = []
    path = tmp_path / 'synthetic_legacy.pt'
    torch.save(expected, path)
    origin['original_checkpoint']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match='EXACT_BINDING'):
        runner._restore_m5_legacy_origin(
            origin=origin, receipts=receipts, order=ORDER, device=DEVICE,
            counters=counts, maximum=1024**2)
