"""M5 TRAIN-weighting bindings; synthetic batches/states, no market payload."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gx1.contracts import market_state_time_query_native_F_batch_v1 as batch
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
from gx1.contracts.market_state_time_query_model_v2 import factorized_log_probabilities
from gx1.contracts.market_state_time_query_binding_v1 import sampled_query_k
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TICK_NS
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as digest
from gx1.features.entry_specialist_feature_groups_v1 import FULL_TA_ENTRY_EXECUTION_PROFILE_NAME
from tests.test_market_state_ta_dedicated_m5_binding_v1 import (
    _model, _capture, _seed, ORDER, DEVICE, COMMIT,
)

# Recorded before this change in FIT_DOMAIN_AND_OLD_CONTRACTS.json, under
# MARKET_STATE_TA_M5_EQUAL_OBSERVATION_LOSS_20261109_001.
OLD_OBJECTIVE_SHA = 'f5a79eacd89aefd878c6cc823e616456fead2e7748c5e589b2a2d8f7a73b30cb'
OLD_NUMERICAL_SHA = 'dc133ed1aabfa357d8fbf18237f6fb6a1816c3e12d55fc24020b2bba532740ca'


def test_default_numerical_is_exact_and_opt_in_changes_only_training_binding():
    old = fit.ta_sf_numerical_recipe(dedicated_m5_side=True)
    assert old == fit.ta_sf_numerical_recipe(dedicated_m5_side=True, m5_equal_observation=False)
    assert old['contract_sha256'] == OLD_NUMERICAL_SHA
    assert old['native_F_loss_contract_sha256'] == OLD_OBJECTIVE_SHA
    assert 'm5_equal_observation' not in old and 'm5_side_weighting_profile' not in old
    new = fit.ta_sf_numerical_recipe(dedicated_m5_side=True, m5_equal_observation=True)
    assert new['m5_equal_observation'] is True
    assert new['m5_side_weighting_profile'] == objective.m5_equal_observation_profile()
    assert new['native_F_loss_contract_sha256'] == objective.contract(
        ta_only=True, dedicated_m5_side=True, m5_equal_observation=True)['contract_sha256']
    changing = {'contract_sha256', 'native_F_loss_contract_sha256',
                'm5_equal_observation', 'm5_side_weighting_profile'}
    assert {k: v for k, v in new.items() if k not in changing} == {
        k: v for k, v in old.items() if k not in changing}
    assert objective.contract(ta_only=True, dedicated_m5_side=True)['contract_sha256'] == OLD_OBJECTIVE_SHA
    with pytest.raises(ValueError):
        fit.ta_sf_numerical_recipe(m5_equal_observation=True)


class _RecordingTA:
    def __init__(self, events):
        self.events = events
        self.state_model = SimpleNamespace(query_native_k=torch.tensor(NATIVE_K),
                                           _dedicated_m5_side=True)
        self.backbone = SimpleNamespace(entry_execution_profile={
            'name': FULL_TA_ENTRY_EXECUTION_PROFILE_NAME})
        self.research_objective = {'contract_sha256': 'synthetic'}

    def __call__(self, **kwargs):
        self.events.append('forward')
        assert set(kwargs) == batch._feature_keys(self) | {'query_k'}
        raw = torch.zeros((len(kwargs['seq_x']), 5, 2), dtype=torch.float32, requires_grad=True)
        return {'state_activity_side_logits': raw,
                'state_log_probs': factorized_log_probabilities(raw),
                'future_activity_side_logits': raw.clone(),
                'future_log_probs': factorized_log_probabilities(raw)}


class _Targets:
    def __init__(self, parents, issues, events):
        self.parent_rows, self.decision_ns = parents.copy(), issues.copy()
        self.events = events

    def _truth(self, parents, issues):
        shape = (len(parents), 5)
        return {'parent_rows': parents.copy(), 'issue_ns': issues.copy(),
                'allowed': np.full(shape, 4, np.uint8),
                'eligible': np.ones(shape, bool), 'mature': np.ones(shape, bool)}

    def gather_fit_sampled(self, parents, issues, *, fold_id, pass_index, seed):
        self.events.append('S-targets')
        return {**self._truth(parents, issues),
                'query_k': sampled_query_k(issues, pass_index=pass_index, seed=seed)}

    def gather(self, parents, issues, *, fold_id, partition):
        self.events.append('F-targets')
        assert partition == 'fit'
        return self._truth(parents, issues)


def _batch_case(monkeypatch, start, size):
    events = []
    model = _RecordingTA(events)
    # Deliberate parent-ID gaps ensure alignment uses axis positions.
    all_parents = np.arange(29574, dtype=np.int64) * 3 + 7
    all_issues = (np.arange(29574, dtype=np.int64) + 1) * TICK_NS
    s = _Targets(all_parents, all_issues, events)
    f = _Targets(all_parents, all_issues, events)
    rows, issues = all_parents[start:start+size].copy(), all_issues[start:start+size].copy()
    features = {key: torch.zeros((len(rows), 1)) for key in batch._feature_keys(model)}
    features['seq_x'] = torch.zeros((len(rows), 1, 4))
    monkeypatch.setattr(batch, 'NativeFutureDualModel', _RecordingTA)
    monkeypatch.setattr(batch, 'BoundTimeQueryTargets', _Targets)
    monkeypatch.setattr(batch, 'BoundFutureDisplacement', _Targets)
    args = [model, features, s, f, rows, issues]
    kwargs = dict(fold_id=1, pass_index=0, seed=20260911, future_weight=1.0,
                  m5_equal_observation=True, fit_batch_start=start)
    return events, args, kwargs


@pytest.mark.parametrize('start,size', [(16, 16), (29568, 6)])
def test_opt_in_binds_axis_positions_before_feature_only_forward(monkeypatch, start, size):
    events, args, kwargs = _batch_case(monkeypatch, start, size)
    result = batch.evaluate_fit_batch(*args, **kwargs)
    assert events == ['forward', 'S-targets', 'F-targets']
    assert result['loss_contract_sha256'] == objective.contract(
        ta_only=True, dedicated_m5_side=True, m5_equal_observation=True)['contract_sha256']
    assert np.all(result['state_query_k'][:, 0] == 1)
    assert torch.isfinite(result['loss'])


@pytest.mark.parametrize('fault', [
    'fold', 'pass', 'seed', 'no_start', 'unaligned_start', 'wrong_start',
    'bool_start', 'short_batch', 'early_tail', 'future_axis', 'issues',
    'no_dedicated', 'gc_profile', 'negative_start', 'past_end',
    'flag_type', 'start_without_opt_in',
])
def test_bad_opt_in_batch_is_rejected_before_forward_or_target_read(monkeypatch, fault):
    size = 15 if fault == 'short_batch' else 6 if fault == 'early_tail' else 16
    events, args, kwargs = _batch_case(monkeypatch, 0, size)
    if fault == 'fold':
        kwargs['fold_id'] = 2
    elif fault == 'pass':
        kwargs['pass_index'] = 1
    elif fault == 'seed':
        kwargs['seed'] += 1
    elif fault == 'no_start':
        kwargs['fit_batch_start'] = None
    elif fault == 'unaligned_start':
        kwargs['fit_batch_start'] = 1
    elif fault == 'wrong_start':
        kwargs['fit_batch_start'] = 16
    elif fault == 'bool_start':
        kwargs['fit_batch_start'] = False
    elif fault == 'future_axis':
        args[3].parent_rows[5] += 1
    elif fault == 'issues':
        args[5] = args[5] + TICK_NS
    elif fault == 'no_dedicated':
        args[0].state_model._dedicated_m5_side = False
    elif fault == 'gc_profile':
        args[0].backbone.entry_execution_profile = {'name': 'legacy-GC'}
        args[1].update({key: torch.zeros((size, 1)) for key in
                       batch._feature_keys(args[0]) - set(args[1])})
    elif fault == 'negative_start':
        kwargs['fit_batch_start'] = -16
    elif fault == 'past_end':
        kwargs['fit_batch_start'] = 29574
    elif fault == 'flag_type':
        kwargs['m5_equal_observation'] = 1
    elif fault == 'start_without_opt_in':
        kwargs['m5_equal_observation'] = False
    with pytest.raises(ValueError, match='NATIVE_F_M5_EQUAL_OBSERVATION'):
        batch.evaluate_fit_batch(*args, **kwargs)
    assert not events


class _RowRecordingTA(_RecordingTA):
    def __call__(self, **kwargs):
        self.events.append('forward')
        assert set(kwargs) == batch._feature_keys(self) | {'query_k'}
        row = kwargs['seq_x'][:, 0, 0, None]
        tf = torch.arange(5, dtype=torch.float32)[None, :]
        query = kwargs['query_k'].to(torch.float32) / torch.tensor(NATIVE_K)
        state_raw = torch.stack((row + tf / 10, .75 * row - tf / 7 + query), -1)
        future_raw = torch.stack((-.5 * row + tf / 11, .25 - row + tf / 13), -1)
        state_raw.requires_grad_()
        future_raw.requires_grad_()
        return {'state_activity_side_logits': state_raw,
                'state_log_probs': factorized_log_probabilities(state_raw),
                'future_activity_side_logits': future_raw,
                'future_log_probs': factorized_log_probabilities(future_raw)}


class _RowTargets(_Targets):
    def _truth(self, parents, issues):
        code = parents[:, None] // 3 + np.arange(5, dtype=np.int64)[None, :]
        return {'parent_rows': parents.copy(), 'issue_ns': issues.copy(),
                'allowed': np.array([1, 2, 4], np.uint8)[code % 3],
                'eligible': code % 7 != 0, 'mature': code % 11 != 0}


def _positions_case(monkeypatch, positions):
    events, args, kwargs = _batch_case(monkeypatch, 0, len(positions))
    args[0] = _RowRecordingTA(events)
    for index in (2, 3):
        args[index] = _RowTargets(args[index].parent_rows, args[index].decision_ns, events)
    args[4] = args[2].parent_rows[positions].copy()
    args[5] = args[2].decision_ns[positions].copy()
    args[1]['seq_x'][:, 0, 0] = torch.from_numpy(args[4]).to(torch.float32) / 100000 - .5
    monkeypatch.setattr(batch, 'NativeFutureDualModel', _RowRecordingTA)
    monkeypatch.setattr(batch, 'BoundTimeQueryTargets', _RowTargets)
    monkeypatch.setattr(batch, 'BoundFutureDisplacement', _RowTargets)
    kwargs.pop('fit_batch_start')
    kwargs['fit_batch_positions'] = positions.copy()
    return events, args, kwargs


def test_explicit_noncontiguous_positions_preserve_row_query_target_pairing(monkeypatch):
    positions = np.array([29573, 0, 14272, 17, 4096, 83, 12288, 31,
                          256, 20001, 9, 28672, 55, 16384, 1, 8192], np.int64)
    events, args, kwargs = _positions_case(monkeypatch, positions)
    axes_before = [(bound.parent_rows.copy(), bound.decision_ns.copy()) for bound in args[2:4]]
    result = batch.evaluate_fit_batch(*args, **kwargs)
    assert events == ['forward', 'S-targets', 'F-targets']
    expected_query = sampled_query_k(args[5], pass_index=0, seed=20260911)
    np.testing.assert_array_equal(result['state_query_k'], expected_query)
    code = args[4][:, None] // 3 + np.arange(5, dtype=np.int64)[None, :]
    for prefix in ('state', 'future'):
        np.testing.assert_array_equal(result[prefix + '_allowed'],
                                      np.array([1, 2, 4], np.uint8)[code % 3])
        np.testing.assert_array_equal(result[prefix + '_mature'], code % 11 != 0)
    torch.testing.assert_close(result['state_activity_side_logits'][:, 0, 0],
                               args[1]['seq_x'][:, 0, 0], rtol=0, atol=0)
    torch.testing.assert_close(result['future_activity_side_logits'][:, 0, 1],
                               .25 - args[1]['seq_x'][:, 0, 0], rtol=0, atol=0)
    assert torch.unique(result['state_activity_side_logits'][:, 0, 0]).numel() == 16
    assert set(result['state_allowed'][:, 0]) == {1, 2, 4}
    assert torch.isfinite(result['loss'])
    for bound, (parents, issues) in zip(args[2:4], axes_before):
        np.testing.assert_array_equal(bound.parent_rows, parents)
        np.testing.assert_array_equal(bound.decision_ns, issues)
    np.testing.assert_array_equal(kwargs['fit_batch_positions'], positions)


def test_explicit_contiguous_positions_equal_original_start_outputs_and_gradients(monkeypatch):
    events, args, kwargs = _positions_case(monkeypatch, np.arange(16, 32, dtype=np.int64))
    original_kwargs = {k: v for k, v in kwargs.items() if k != 'fit_batch_positions'}
    original_kwargs['fit_batch_start'] = 16
    old = batch.evaluate_fit_batch(*args, **original_kwargs)
    explicit = batch.evaluate_fit_batch(*args, **kwargs)
    assert events == ['forward', 'S-targets', 'F-targets'] * 2
    assert old.keys() == explicit.keys()
    for key, value in old.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, explicit[key]), key
        elif isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, explicit[key], err_msg=key)
        else:
            assert value == explicit[key], key
    heads = ('state_activity_side_logits', 'future_activity_side_logits')
    old_grad = torch.autograd.grad(old['loss'], tuple(old[key] for key in heads))
    explicit_grad = torch.autograd.grad(explicit['loss'], tuple(explicit[key] for key in heads))
    for before, after in zip(old_grad, explicit_grad):
        assert torch.equal(before, after)
        assert torch.count_nonzero(before) > 0


@pytest.mark.parametrize('fault', [
    'float_dtype', 'int32_dtype', 'bool_dtype', 'list', 'rank2', 'duplicate',
    'negative', 'past_end', 'short', 'long', 'empty', 'six_row_tail',
    'both_selectors', 'without_opt_in', 'fold', 'pass', 'seed',
    'no_dedicated', 'gc_profile', 'state_parent', 'future_parent',
    'state_issue', 'future_issue', 'row_order', 'issue_order',
])
def test_bad_explicit_positions_reject_before_forward_or_targets(monkeypatch, fault):
    positions = np.arange(0, 32, 2, dtype=np.int64)
    if fault == 'six_row_tail':
        positions = np.arange(29568, 29574, dtype=np.int64)
    events, args, kwargs = _positions_case(monkeypatch, positions)
    if fault in ('float_dtype', 'int32_dtype', 'bool_dtype'):
        dtype = {'float_dtype': np.float64, 'int32_dtype': np.int32,
                 'bool_dtype': bool}[fault]
        kwargs['fit_batch_positions'] = positions.astype(dtype)
    elif fault == 'list':
        kwargs['fit_batch_positions'] = positions.tolist()
    elif fault == 'rank2':
        kwargs['fit_batch_positions'] = positions.reshape(4, 4)
    elif fault == 'duplicate':
        kwargs['fit_batch_positions'][1] = positions[0]
    elif fault == 'negative':
        kwargs['fit_batch_positions'][0] = -1
    elif fault == 'past_end':
        kwargs['fit_batch_positions'][-1] = 29574
    elif fault == 'short':
        kwargs['fit_batch_positions'] = positions[:-1]
    elif fault == 'long':
        kwargs['fit_batch_positions'] = np.append(positions, np.int64(32))
    elif fault == 'empty':
        kwargs['fit_batch_positions'] = np.array([], np.int64)
    elif fault == 'both_selectors':
        kwargs['fit_batch_start'] = 0
    elif fault == 'without_opt_in':
        kwargs['m5_equal_observation'] = False
    elif fault in ('fold', 'pass', 'seed'):
        kwargs[{'fold': 'fold_id', 'pass': 'pass_index', 'seed': 'seed'}[fault]] += 1
    elif fault == 'no_dedicated':
        args[0].state_model._dedicated_m5_side = False
    elif fault == 'gc_profile':
        args[0].backbone.entry_execution_profile = {'name': 'legacy-GC'}
        args[1].update({key: torch.zeros((16, 1)) for key in
                       batch._feature_keys(args[0]) - set(args[1])})
    elif fault in ('state_parent', 'future_parent', 'state_issue', 'future_issue'):
        bound = args[2 if fault.startswith('state_') else 3]
        axis = bound.parent_rows if fault.endswith('_parent') else bound.decision_ns
        axis[positions[2]] += 1
    elif fault == 'row_order':
        args[4] = args[4][::-1].copy()
    elif fault == 'issue_order':
        args[5] = args[5][::-1].copy()
    with pytest.raises(ValueError, match='NATIVE_F_M5_EQUAL_OBSERVATION'):
        batch.evaluate_fit_batch(*args, **kwargs)
    assert not events


@pytest.mark.parametrize('equal', [False, True])
def test_state_roundtrip_preserves_each_exact_recipe_and_rejects_cross_restore(monkeypatch, equal):
    model = _model(monkeypatch, True)
    recipe = fit.ta_sf_numerical_recipe(dedicated_m5_side=True, m5_equal_observation=equal)
    optimizer, _ = state.build_fold1_fit_optimizer(model, numerical_recipe=recipe)
    _seed()
    saved = _capture(model, optimizer, recipe, 0)
    resumed = _model(monkeypatch, True)
    resumed_optimizer, _ = state.build_fold1_fit_optimizer(resumed, numerical_recipe=recipe)
    assert state.restore_fold1_fit_state(saved, resumed, resumed_optimizer,
        numerical_recipe=recipe, feature_view_identity=resumed.state_model.feature_view_identity,
        parent_order=ORDER, source_commit=COMMIT, device=DEVICE) == {
            'cursor': 0, 'optimizer_updates': 0, 'skipped_batches': 0}
    actual = _capture(resumed, resumed_optimizer, recipe, 0)
    assert actual['contract_sha256'] == saved['contract_sha256']
    assert actual['binding']['numerical_recipe'] == recipe
    other = fit.ta_sf_numerical_recipe(dedicated_m5_side=True, m5_equal_observation=not equal)
    destination = _model(monkeypatch, True)
    other_optimizer, _ = state.build_fold1_fit_optimizer(destination, numerical_recipe=other)
    before = digest(destination.state_dict())
    with pytest.raises(RuntimeError, match='EXACT_BINDING'):
        state.restore_fold1_fit_state(saved, destination, other_optimizer,
            numerical_recipe=other, feature_view_identity=destination.state_model.feature_view_identity,
            parent_order=ORDER, source_commit=COMMIT, device=DEVICE)
    assert digest(destination.state_dict()) == before and not other_optimizer.state


def test_resealed_but_changed_weighting_profile_is_rejected(monkeypatch):
    model = _model(monkeypatch, True)
    recipe = fit.ta_sf_numerical_recipe(dedicated_m5_side=True, m5_equal_observation=True)
    broken = copy.deepcopy(recipe)
    broken['m5_side_weighting_profile']['fit_rows'] -= 1
    broken['contract_sha256'] = fit.campaign.canonical_sha256({
        k: v for k, v in broken.items() if k != 'contract_sha256'})
    before = digest(model.state_dict())
    with pytest.raises(RuntimeError, match='TA_SF_NUMERICAL_RECIPE'):
        state.build_fold1_fit_optimizer(model, numerical_recipe=broken)
    assert digest(model.state_dict()) == before
