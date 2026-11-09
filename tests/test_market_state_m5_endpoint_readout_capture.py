"""Passive capture parity and complete-only publication; no encoder or fitting."""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gx1.contracts import market_state_time_query_native_F_batch_v1 as batch
from gx1.contracts.market_state_time_query_model_v2 import (
    FactorizedTimeQueryMarketStateModel, factorized_log_probabilities)
from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
from gx1.features.entry_specialist_feature_groups_v1 import (
    FULL_TA_ENTRY_EXECUTION_PROFILE_NAME, FULL_TA_ENTRY_FEATURE_KEYS)
from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
from tests.test_market_state_time_query_native_F_objective import _endpoint_fixture
from tests.test_market_state_ta_sf_fold1_evaluation_v1 import _fake_evaluation


class CaptureModel(torch.nn.Module):
    def __init__(self, fault=None):
        super().__init__()
        self.state_model = torch.nn.Module()
        self.state_model.register_buffer('query_native_k', torch.tensor(NATIVE_K))
        self.state_model.feature_view_identity = {'contract_sha256': 'synthetic-view'}
        self.state_model._dedicated_m5_side = True
        self.state_model.m5_side_head = torch.nn.Linear(128, 1)
        self.future_m5_side_head = torch.nn.Linear(128, 1)
        self.backbone = SimpleNamespace(entry_execution_profile={
            'name': FULL_TA_ENTRY_EXECUTION_PROFILE_NAME})
        self.research_objective = {'contract_sha256': 'synthetic-model'}
        self.fault = fault
        self.calls = []

    def forward(self, **features):
        self.calls.append(set(features))
        x = features['seq_x']
        if self.fault == 'nan':
            x = x.clone()
            x[0, 0] = float('nan')
        state = torch.zeros((len(x), 5, 2), dtype=torch.float32)
        future = state.clone()
        state[:, 0, 1] = self.state_model.m5_side_head(x[:, None, :])[:, 0, 0]
        if self.fault == 'duplicate':
            self.state_model.m5_side_head(x[:, None, :])
        if self.fault == 'raise':
            raise RuntimeError('synthetic forward failure')
        if self.fault != 'missing':
            future[:, 0, 1] = self.future_m5_side_head(x + .25).squeeze(-1)
        return {'state_activity_side_logits': state,
                'state_log_probs': factorized_log_probabilities(state),
                'future_activity_side_logits': future,
                'future_log_probs': factorized_log_probabilities(future)}


def _endpoint_case(monkeypatch, fault=None, model_type=CaptureModel):
    monkeypatch.setattr(batch, 'NativeFutureDualModel', model_type)
    s, f = _endpoint_fixture()
    axes = s.mature_oof_axes(fold_id=1)
    model = model_type(fault).eval()
    n = len(axes['fit_parent_rows'])
    features = {key: torch.zeros((n, 1), dtype=torch.float32)
                for key in FULL_TA_ENTRY_FEATURE_KEYS}
    features['seq_x'] = torch.arange(n * 128, dtype=torch.float32).reshape(n, 128) / 128
    args = (model, features, s, f, axes['fit_parent_rows'], axes['fit_decision_ns'])
    return model, args, dict(fold_id=1, partition='fit', future_weight=1.0)


def test_passive_capture_preserves_forecast_state_rng_and_feature_only_boundary(monkeypatch):
    model, args, kw = _endpoint_case(monkeypatch)
    before = copy.deepcopy(model.state_dict())
    rng = torch.random.get_rng_state().clone()
    plain = batch.evaluate_endpoint_batch(*args, **kw)
    captured = batch.evaluate_endpoint_batch(*args, **kw, capture_m5_readout=True)
    assert len(model.calls) == 2
    assert all(keys == FULL_TA_ENTRY_FEATURE_KEYS | {'query_k'} for keys in model.calls)
    assert plain['measurement'] == captured['measurement']
    for task in ('state', 'future'):
        np.testing.assert_array_equal(plain[task + '_activity_side_logits'],
                                      captured[task + '_activity_side_logits'])
    out = captured['m5_readout']
    np.testing.assert_array_equal(out['state_representation'], args[1]['seq_x'].numpy())
    np.testing.assert_array_equal(out['future_representation'], (args[1]['seq_x'] + .25).numpy())
    for task, bound in (('state', args[2]), ('future', args[3])):
        truth = bound.gather(args[4], args[5], fold_id=1, partition='fit')
        for field in ('allowed', 'eligible', 'mature'):
            np.testing.assert_array_equal(out[task + '_' + field], truth[field])
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert all(p.grad is None for p in model.parameters())
    assert not model.state_model.m5_side_head._forward_pre_hooks
    assert not model.future_m5_side_head._forward_pre_hooks


@pytest.mark.parametrize('fault,reason', [
    ('duplicate', 'CAPTURE_INPUT'), ('nan', 'CAPTURE_INPUT'),
    ('missing', 'MISSING_HEAD'), ('raise', 'synthetic forward failure')])
def test_failed_capture_removes_every_hook(monkeypatch, fault, reason):
    model, args, kw = _endpoint_case(monkeypatch, fault)
    with pytest.raises(RuntimeError, match=reason):
        batch.evaluate_endpoint_batch(*args, **kw, capture_m5_readout=True)
    assert not model.state_model.m5_side_head._forward_pre_hooks
    assert not model.future_m5_side_head._forward_pre_hooks
    assert all(p.grad is None for p in model.parameters())



class RealDecoderCaptureModel(CaptureModel):
    """Actual production decoders; only encoder and binding IO are replaced."""
    def __init__(self, fault=None):
        super().__init__(fault)
        self.state_model._hidden_width = 128
        self.state_model.register_buffer('query_tf_coordinates', torch.eye(5))
        self.state_model.query_projection = torch.nn.Linear(134, 128)
        self.state_model.query_head = torch.nn.Linear(128, 2)
        self.state_model._require_binding = lambda: None
        self.future_head = torch.nn.Linear(128, 2)

    def _require_binding(self):
        pass

    def forward(self, **features):
        self.calls.append(set(features))
        hidden, query = features['seq_x'], features['query_k']
        state = FactorizedTimeQueryMarketStateModel.decode_raw_query_grid(
            self.state_model, hidden, query[:, None, :])[:, 0]
        future = NativeFutureDualModel.decode_native_F_raw(self, hidden)
        return {'state_activity_side_logits': state,
                'state_log_probs': factorized_log_probabilities(state),
                'future_activity_side_logits': future,
                'future_log_probs': factorized_log_probabilities(future)}


def test_real_decoder_query_axis_preserves_logits_and_captures_exact_head_inputs(monkeypatch):
    model, args, kw = _endpoint_case(monkeypatch, model_type=RealDecoderCaptureModel)
    before = copy.deepcopy(model.state_dict())
    rng = torch.random.get_rng_state().clone()
    plain = batch.evaluate_endpoint_batch(*args, **kw)
    captured = batch.evaluate_endpoint_batch(*args, **kw, capture_m5_readout=True)
    out = captured['m5_readout']
    with torch.no_grad():
        hidden = args[1]['seq_x']
        n = len(hidden)
        projected = model.state_model.query_projection(torch.cat((
            hidden[:, None, :].expand(-1, 5, -1),
            torch.eye(5)[None].expand(n, -1, -1),
            torch.ones(n, 5, 1)), dim=-1))[:, 0]
    assert plain['measurement'] == captured['measurement']
    for task, head in (('state', model.state_model.m5_side_head),
                       ('future', model.future_m5_side_head)):
        np.testing.assert_array_equal(out[task + '_representation'], projected.numpy())
        np.testing.assert_array_equal(plain[task + '_activity_side_logits'],
                                      captured[task + '_activity_side_logits'])
        with torch.no_grad():
            expected_side = head(projected).squeeze(-1).numpy()
        np.testing.assert_array_equal(out[task + '_activity_side_logits'][:, 0, 1], expected_side)
    assert len(model.calls) == 2
    assert all(keys == FULL_TA_ENTRY_FEATURE_KEYS | {'query_k'} for keys in model.calls)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('task,shape', [
    ('state', (3, 128)), ('state', (3, 2, 128)),
    ('future', (3, 1, 128))])
def test_capture_rejects_wrong_task_query_axes(task, shape):
    model = CaptureModel()
    _, handles = batch._m5_readout_hooks(model, 3)
    head = model.state_model.m5_side_head if task == 'state' else model.future_m5_side_head
    try:
        with pytest.raises(RuntimeError, match='NATIVE_F_M5_CAPTURE_INPUT'):
            head(torch.zeros(shape))
    finally:
        for handle in handles:
            handle.remove()


def _capture_evaluation(monkeypatch, partition, fault=None):
    args, model, calls, counters = _fake_evaluation(
        monkeypatch, partition=partition, fail='deadline' if fault == 'deadline' else None)
    original = batch.evaluate_endpoint_batch

    def evaluate(*pos, **kw):
        assert kw.pop('capture_m5_readout') is True
        report = original(*pos, **kw)
        parents, issues = pos[4:6]
        n = len(parents)
        arrays = {key: np.zeros(shape, dtype)
                  for key, (shape, dtype) in runner._ta_m5_capture_layout(n).items()}
        arrays.update(parents=parents.copy(), issues=issues.copy(),
                      query_native=np.broadcast_to(np.asarray(NATIVE_K, np.int64), (n, 5)).copy())
        for task in ('state', 'future'):
            arrays[task + '_representation'][:] = parents[:, None] / 65536
            arrays[task + '_allowed'][:] = 4
            arrays[task + '_eligible'][:] = True
            arrays[task + '_mature'][:] = True
        if fault == 'axis':
            arrays['parents'][0] += 1
        if fault == 'dtype':
            arrays['state_representation'] = arrays['state_representation'].astype(np.float64)
        report['m5_readout'] = arrays
        return report

    monkeypatch.setattr(batch, 'evaluate_endpoint_batch', evaluate)
    return args, model, calls, counters


@pytest.mark.parametrize('partition', ['fit', 'score'])
def test_complete_population_capture_reuses_each_measurement_forward(monkeypatch, tmp_path, partition):
    args, model, calls, counters = _capture_evaluation(monkeypatch, partition)
    path = tmp_path / 'M5.npz'
    measured = runner._ta_endpoint_measurement(**args, capture_path=path)
    sealed_body = {key: value for key, value in measured.items() if key != 'contract_sha256'}
    assert measured['contract_sha256'] == hashlib.sha256(json.dumps(
        sealed_body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    capture = measured.pop('m5_readout_capture')
    n = len(args['rows'])
    assert counters['forward_calls'] == len(calls) == (n + 15) // 16
    assert capture['array_bytes'] == 1190 * n <= 40 * 1024**2
    assert capture['additional_forward_calls'] == 0
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == set(runner._ta_m5_capture_layout(n))
        np.testing.assert_array_equal(saved['parents'], args['rows'])
        np.testing.assert_array_equal(saved['issues'], args['issues'])
        np.testing.assert_array_equal(saved['state_representation'][:, 0],
                                      (args['rows'] / 65536).astype(np.float32))
        assert saved['future_mature'].all()
    assert model.training and not model.child.training


@pytest.mark.parametrize('fault,reason', [
    ('axis', 'M5_CAPTURE_AXES'), ('dtype', 'M5_CAPTURE_ARRAY'),
    ('deadline', None)])
def test_partial_or_misaligned_measurement_never_publishes(monkeypatch, tmp_path, fault, reason):
    args, model, calls, counters = _capture_evaluation(monkeypatch, 'fit', fault)
    path = tmp_path / 'M5.npz'
    error = runner._ScoreWindowExhausted if fault == 'deadline' else RuntimeError
    with pytest.raises(error, match=reason):
        runner._ta_endpoint_measurement(**args, capture_path=path)
    assert not path.exists()
    assert model.training and not model.child.training
