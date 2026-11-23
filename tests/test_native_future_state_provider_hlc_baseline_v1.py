"""Fake-only V2 baseline adapters: no native data, backbone or checkpoint IO."""
import numpy as np
import pytest
import torch

from gx1.audit import native_future_state_provider_hlc_baseline_v1 as owner
from tests.test_native_future_state_provider_hlc_v1 import _projected_case


def _setup(monkeypatch):
    _, truth, certificate, upper = _projected_case(monkeypatch)
    target = truth['future_state_contract']
    prior = owner.objective.fit_native_future_state_train_prior(truth=truth,
        fold_upper_ns=upper, target_contract=target)
    issues = truth['issue_ns'].copy()
    source = np.arange(100, 108, dtype=np.int64)
    binding = {'schema_version': 'synthetic_full_TA_input_binding'}
    norm = {'fit_population': {'selected_source_rows_sha256': owner.core.digest(source)}}
    keys = owner.model_owner._native_bar_feature_keys(binding)

    class Provider:
        def __init__(self):
            self.issue_ns = issues.copy()
            self.source_row = source.copy()
            self.input_ready = np.ones(8, dtype=np.bool_)
            self.calls = []
            self.extra_metadata = False
        def input_binding(self, normalization):
            assert normalization is norm
            return binding
        def gather(self, rows, *, max_rows):
            assert len(rows) == max_rows
            self.calls.append(rows.copy())
            raw = {key: np.repeat(rows[:, None], 2, axis=1).astype(
                np.int64 if key == 'ctx_cat' else np.float32) for key in keys}
            if self.extra_metadata:
                raw['target_observation_certificate_ref'] = certificate['ref']
            return raw

    provider = Provider()
    meta = {'source_snapshot': {'path': '/synthetic/snapshot.json', 'sha256': 'a' * 64},
        'normalization': {'path': '/synthetic/norm.json', 'sha256': 'b' * 64},
        'prior': {'path': '/synthetic/prior.json', 'sha256': 'c' * 64},
        'dataset': {'path': '/synthetic/v2_dataset.json', 'sha256': 'd' * 64},
        'input_binding': binding, 'target_contract': target}
    objects = {meta['normalization']['path']: norm, meta['prior']['path']: prior}
    loaded = []
    for name in owner.training.provider_hlc_truth_fields(target):
        truth[name].flags.writeable = False
    def loader(ref, *, expected_issue_ns, target_contract, fold_upper_ns):
        assert ref == meta['dataset'] and target_contract == target and fold_upper_ns == upper
        np.testing.assert_array_equal(expected_issue_ns, issues)
        loaded.append(ref)
        return truth
    monkeypatch.setattr(owner, 'ROWS', 8)
    monkeypatch.setattr(owner, 'TRAIN_SHA', owner.core.digest(issues))
    monkeypatch.setattr(owner, 'load_snapshot', lambda ref: provider)
    monkeypatch.setattr(owner, 'read', lambda ref: objects[ref['path']])
    monkeypatch.setattr(owner, 'load_provider_hlc_target_truth', loader)
    # Exercise score restoration without touching CUDA RNG in this CPU-only fixture.
    monkeypatch.setattr(owner.core, 'training_rng',
        lambda: {'torch_cpu': torch.get_rng_state().clone()})
    monkeypatch.setattr(owner.core, 'restore_training_rng',
        lambda state: torch.set_rng_state(state['torch_cpu']))
    return provider, meta, objects, truth, prior, loaded


class TinyModel(torch.nn.Module):
    def __init__(self, context, bad=None):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.context = context
        self.bad = bad
        self.calls = 0
    def forward(self, **features):
        assert 'target_observation_certificate_ref' not in features
        assert not self.training and not torch.is_grad_enabled()
        self.calls += 1
        torch.rand(1)  # The scoring helper must restore caller RNG after evaluation.
        output = owner.objective.native_future_state_train_prior_logits(self.context['prior'],
            rows=len(features['seq_x']), target_contract=self.context['target_contract'])
        output = {key: value + self.anchor * 0 for key, value in output.items()}
        if self.bad == 'wrong_prior':
            output['side_logits'][0, 0] += 1
        elif self.bad == 'nonfinite':
            output['activity_logits'][0, 0] = float('nan')
        return output


def _mock_forward(monkeypatch):
    calls = []
    def forward(model, features, *, grouped, equal_tf, capture):
        assert grouped is True and equal_tf is True and capture is False
        assert len(features) == 9 and all(isinstance(x, torch.Tensor)
            and x.device.type == 'cpu' for x in features.values())
        calls.append(len(features['seq_x']))
        return model(**features), None, {'batch_eval_active': True}
    monkeypatch.setattr(owner.core, 'future_cuda_forward', forward)
    monkeypatch.setattr(owner.core, 'future_cuda_score',
        lambda *a, **k: pytest.fail('legacy truth-concatenating scorer reached'))
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('checkpoint IO reached'))
    return calls


def test_packet_preserves_physical_features_and_readonly_truth_metadata(monkeypatch):
    provider, meta, _, truth, _, loaded = _setup(monkeypatch)
    context = owner.load_context(meta)
    before = {key: value.tobytes() for key, value in truth.items()
              if isinstance(value, np.ndarray)}
    take = np.array([0, 2, 7], dtype=np.int64)
    result = owner.packet(context, take)
    assert len(loaded) == 1 and len(provider.calls) == 1
    np.testing.assert_array_equal(provider.calls[0], context['source_rows'][take])
    assert set(result['features']) == set(owner.model_owner._native_bar_feature_keys(meta['input_binding']))
    assert all(x.device.type == 'cpu' for x in result['features'].values())
    assert 'target_observation_certificate_ref' not in result['features']
    assert result['truth']['target_observation_certificate_ref'] == truth['target_observation_certificate_ref']
    assert result['truth']['target_observation_certificate_ref'] is not truth['target_observation_certificate_ref']
    for key in owner.training.provider_hlc_truth_fields(context['target_contract']):
        np.testing.assert_array_equal(result['truth'][key], truth[key][take])
        assert not result['truth'][key].flags.writeable
        assert not np.shares_memory(result['truth'][key], truth[key])
    assert all(truth[key].tobytes() == value for key, value in before.items())


@pytest.mark.parametrize('bad', ['input_binding', 'issue_clock', 'source_rows', 'prior_truth'])
def test_context_rejects_changed_inputs_clock_or_matched_prior_truth(monkeypatch, bad):
    provider, meta, objects, truth, _, _ = _setup(monkeypatch)
    if bad == 'input_binding':
        meta['input_binding'] = {'schema_version': 'wrong_binding'}
    elif bad == 'issue_clock':
        provider.issue_ns[0] += 300_000_000_000
    elif bad == 'source_rows':
        provider.source_row[0] += 1
    else:
        changed = dict(truth)
        changed['side_up'] = truth['side_up'].copy()
        changed['side_up'][1, 4] = False
        objects[meta['prior']['path']] = owner.objective.fit_native_future_state_train_prior(
            truth=changed, fold_upper_ns=owner.UPPER, target_contract=meta['target_contract'])
    with pytest.raises(ValueError, match='NATIVE_PROVIDER_HLC_BASELINE_'):
        owner.load_context(meta)
    assert not provider.calls


def test_actual_score_utility_covers_clock_once_and_reuses_full_truth(monkeypatch):
    provider, meta, _, truth, prior, loaded = _setup(monkeypatch)
    context = owner.load_context(meta)
    calls = _mock_forward(monkeypatch)
    original_score = owner.objective.native_future_state_score_sums
    seen_truth = []
    def score(**kwargs):
        assert kwargs['truth'] is truth
        seen_truth.append(kwargs['truth'])
        return original_score(**kwargs)
    monkeypatch.setattr(owner.objective, 'native_future_state_score_sums', score)
    model = TinyModel(context)
    before_rng = torch.get_rng_state().clone()
    before_state = owner.core.state_sha(model.state_dict())
    progress = []
    report = owner.score_initial(model, context, batch_rows=3, device='cpu',
                                 progress_callback=progress.append)
    assert report['rows'] == report['actual_NN_forward_rows'] == 8
    assert report['whole_TRAIN_initial'] is True and report['initial_prior_exact'] is True
    assert report['batch_count'] == model.calls == 3 and calls == [3, 3, 2]
    assert [x['rows_done'] for x in progress] == [3, 6, 8]
    assert len(loaded) == 1 and len(provider.calls) == 3
    np.testing.assert_array_equal(np.concatenate(provider.calls), context['source_rows'])
    assert len(seen_truth) == 2
    assert model.training is True and torch.equal(torch.get_rng_state(), before_rng)
    assert owner.core.state_sha(model.state_dict()) == before_state
    assert all(p.grad is None for p in model.parameters())
    reference = owner.objective.native_future_state_train_prior_logits(prior,
        rows=8, target_contract=context['target_contract'])
    assert report['scores'] == original_score(model_output=reference, truth=truth,
        fold_upper_ns=owner.UPPER, target_contract=context['target_contract'])
    assert report['output_sha256'] == {k: owner.core.digest(v.numpy()) for k, v in reference.items()}
    assert report['output_arrays_collected'] == 2 and report['output_ndarray_bytes'] == 8 * 5 * 4 * 2
    assert report['loss_score_abs_delta'] < 1e-6
    assert report['score_utility_counters'] == {'model_forwards': 3, 'optimizer_updates': 0,
        'backward_calls': 0, 'fits': 0, 'checkpoint_loads': 0, 'checkpoint_writes': 0}


@pytest.mark.parametrize('bad', ['wrong_prior', 'nonfinite'])
def test_bad_initial_output_stops_at_first_batch_and_restores_state(monkeypatch, bad):
    provider, meta, _, _, _, _ = _setup(monkeypatch)
    context = owner.load_context(meta)
    calls = _mock_forward(monkeypatch)
    model = TinyModel(context, bad=bad)
    before_rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        owner.score_initial(model, context, batch_rows=3, device='cpu')
    assert model.calls == len(provider.calls) == 1 and calls == [3]
    assert model.training is True and torch.equal(torch.get_rng_state(), before_rng)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('bad', ['short_clock', 'wrong_clock_sha'])
def test_full_initial_rejects_partial_or_changed_clock_before_forward(monkeypatch, bad):
    provider, meta, _, _, _, _ = _setup(monkeypatch)
    context = owner.load_context(meta)
    calls = _mock_forward(monkeypatch)
    if bad == 'short_clock':
        context['source_rows'] = context['source_rows'][:-1]
    else:
        context['truth'] = dict(context['truth'])
        context['truth']['issue_ns'] = context['truth']['issue_ns'].copy()
        context['truth']['issue_ns'][0] += 300_000_000_000
    with pytest.raises(ValueError, match='FULL_INITIAL_CLOCK'):
        owner.score_initial(TinyModel(context), context, batch_rows=3, device='cpu')
    assert calls == [] and provider.calls == []


def test_packet_rejects_certificate_metadata_in_provider_features(monkeypatch):
    provider, meta, _, _, _, _ = _setup(monkeypatch)
    context = owner.load_context(meta)
    provider.extra_metadata = True
    with pytest.raises(ValueError, match='FEATURE_KEYS'):
        owner.packet(context, np.array([0], dtype=np.int64))
