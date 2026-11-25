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


# Durable INITIAL scoring: synthetic outputs only, never a native payload.
class SegmentedTinyModel(TinyModel):
    def forward(self, **features):
        assert not self.training and not torch.is_grad_enabled()
        assert 'target_observation_certificate_ref' not in features
        self.calls += 1
        output = owner.objective.native_future_state_train_prior_logits(self.context['prior'],
            rows=len(features['seq_x']), target_contract=self.context['target_contract'])
        output = {key: value + self.anchor * 0 for key, value in output.items()}
        if self.bad == 'RNG':
            torch.rand(1)
        elif self.bad == 'parameter':
            self.anchor.add_(1)
        elif self.bad == 'gradient':
            self.anchor.grad = torch.zeros_like(self.anchor)
        elif self.bad == 'wrong_prior':
            output['side_logits'][0, 0] += 1
        elif self.bad == 'nonfinite':
            output['activity_logits'][0, 0] = float('nan')
        return output


def _segmented_case(monkeypatch):
    provider, meta, _, _, _, _ = _setup(monkeypatch)
    context = owner.load_context(meta)
    calls = _mock_forward(monkeypatch)
    return provider, context, calls


def _segmented_capture(context, *, batch_rows=3):
    import copy
    committed = []
    def commit(event, values):
        arrays = {key: (value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor)
                       else value.copy()) for key, value in values.items()}
        committed.append((copy.deepcopy(event), arrays))
    report = owner.score_initial_segmented(SegmentedTinyModel(context), context,
        batch_rows=batch_rows, device='cpu', commit_batch=commit)
    return report, committed


def _segmented_prefix(committed, batches):
    import copy
    chosen = committed[:batches]
    return {'rows_done': sum(len(values['activity_logits']) for _, values in chosen),
        'batches': [copy.deepcopy(event['batch']) for event, _ in chosen],
        'outputs': {key: np.concatenate([values[key] for _, values in chosen], axis=0)
                    if chosen else np.empty((0, 5), dtype=np.float32)
                    for key in owner.OUTPUT_KEYS}}


def test_segmented_initial_addition_preserves_every_old_adapter_byte():
    import pathlib
    import subprocess
    path = 'gx1/audit/native_future_state_provider_hlc_baseline_v1.py'
    original = subprocess.check_output(['git', 'show',
        '9d7b39da9214f74dcf6a7b8bf1a0fd3a719da29f:'+path], cwd=owner.core.REPO)
    current = (pathlib.Path(owner.core.REPO)/path).read_bytes()
    import ast
    text = current.decode(); tree = ast.parse(text)
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'require_provider_hlc_scoring_contract')
    expected = ast.parse('''def require_provider_hlc_scoring_contract(value):
    """Only authenticated provider generations V2/V3 enter this scoring adapter."""
    if training._provider_hlc_all_tf_v3(value):
        return hlc.require_all_tf_provider_hlc_state_contract(value)
    return hlc.require_provider_hlc_state_contract(value)
''').body[0]
    assert ast.dump(helper) == ast.dump(expected)
    addition = '\n' + ast.get_source_segment(text, helper) + '\n\n'
    assert text.count(addition) == 1
    normalized = text.replace(addition, '', 1)
    assert normalized.count('require_provider_hlc_scoring_contract(') == 4
    normalized = normalized.replace('require_provider_hlc_scoring_contract(',
                                    'require_provider_hlc_state_contract(')
    current_import = 'from gx1.contracts import native_future_state_provider_hlc_v1 as hlc'
    assert normalized.count(current_import) == 1
    normalized = normalized.replace(current_import,
        'from gx1.contracts.native_future_state_provider_hlc_v1 import require_provider_hlc_state_contract')
    assert normalized.encode().startswith(original)
    assert current.count(b'def score_initial_segmented(') == 1


@pytest.mark.parametrize('prefix_batches,expected_calls', [(0, [3, 3, 2]), (1, [3, 2]), (3, [])])
def test_segmented_initial_authenticated_prefix_matches_uninterrupted_full_scores(monkeypatch, prefix_batches, expected_calls):
    provider, context, calls = _segmented_case(monkeypatch)
    full, committed = _segmented_capture(context)
    prefix = _segmented_prefix(committed, prefix_batches)
    calls.clear(); provider.calls.clear()
    model = SegmentedTinyModel(context)
    before_rng = torch.get_rng_state().clone()
    before_state = owner.core.state_sha(model.state_dict())
    progress, new_commits = [], []
    report = owner.score_initial_segmented(model, context, batch_rows=3, device='cpu',
        completed_prefix=prefix, commit_batch=lambda event, values: new_commits.append(event),
        progress_callback=progress.append)
    assert calls == expected_calls and model.calls == len(expected_calls)
    assert len(provider.calls) == len(expected_calls)
    assert report['actual_NN_forward_rows'] == 8 and report['new_NN_forward_rows'] == 8-prefix['rows_done']
    assert report['new_model_forwards'] == len(expected_calls)
    assert report['completed_prefix_rows'] == prefix['rows_done'] and report['completed_prefix_batches'] == prefix_batches
    assert report['unique_complete_batches'] == 3
    assert report['score_utility_counters'] == {'model_forwards': len(expected_calls), 'optimizer_updates': 0,
        'backward_calls': 0, 'fits': 0, 'checkpoint_loads': 0, 'checkpoint_writes': 0}
    for key in ('rows', 'batch_count', 'batches', 'scores', 'output_sha256',
                'known_activity_cells', 'proper_partial_NLL', 'batch_weighted_partial_NLL'):
        assert report[key] == full[key]
    assert [event['batch_cursor'] for event in new_commits] == list(range(prefix_batches+1, 4))
    assert [event['rows_done'] for event in progress] == [event['rows_done'] for event in new_commits]
    assert model.training and torch.equal(torch.get_rng_state(), before_rng)
    assert owner.core.state_sha(model.state_dict()) == before_state
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize('bad', ['rows', 'gap', 'duplicate', 'shape', 'dtype', 'head', 'nonfinite', 'hash', 'prior'])
def test_segmented_initial_rejects_bad_prefix_before_any_new_forward(monkeypatch, bad):
    provider, context, calls = _segmented_case(monkeypatch)
    _, committed = _segmented_capture(context)
    prefix = _segmented_prefix(committed, 1)
    if bad == 'rows': prefix['rows_done'] += 1
    elif bad == 'gap': prefix['batches'][0]['cursor'] = 2
    elif bad == 'duplicate': prefix['batches'].append(dict(prefix['batches'][0]))
    elif bad == 'shape': prefix['outputs']['activity_logits'] = np.zeros((2, 5), np.float32)
    elif bad == 'dtype': prefix['outputs']['side_logits'] = prefix['outputs']['side_logits'].astype(np.float64)
    elif bad == 'head': prefix['outputs']['unexpected'] = np.zeros((3, 5), np.float32)
    elif bad == 'nonfinite': prefix['outputs']['side_logits'][0, 0] = np.nan
    elif bad == 'hash': prefix['batches'][0]['output_sha256']['side_logits'] = '0'*64
    else:
        prefix['outputs']['side_logits'][0, 0] += 1
        prefix['batches'][0]['output_sha256']['side_logits'] = owner.core.digest(prefix['outputs']['side_logits'])
    calls.clear(); provider.calls.clear()
    with pytest.raises(ValueError):
        owner.score_initial_segmented(SegmentedTinyModel(context), context,
            batch_rows=3, device='cpu', completed_prefix=prefix)
    assert calls == [] and provider.calls == []


@pytest.mark.parametrize('bad', ['RNG', 'parameter', 'gradient', 'wrong_prior', 'nonfinite'])
def test_segmented_initial_proves_every_batch_state_RNG_gradients_and_outputs_before_commit(monkeypatch, bad):
    provider, context, calls = _segmented_case(monkeypatch)
    model = SegmentedTinyModel(context, bad=bad)
    before_rng = torch.get_rng_state().clone()
    commits = []
    with pytest.raises(ValueError):
        owner.score_initial_segmented(model, context, batch_rows=3, device='cpu',
            commit_batch=lambda *args: commits.append(args))
    assert commits == [] and calls == [3] and len(provider.calls) == 1
    assert model.training and torch.equal(torch.get_rng_state(), before_rng)


def test_segmented_initial_commit_exposes_exact_cpu_batch_proof_before_progress(monkeypatch):
    _, context, _ = _segmented_case(monkeypatch)
    model = SegmentedTinyModel(context)
    events = []
    def commit(event, values):
        left, right = event['left'], event['right']
        assert set(values) == set(owner.OUTPUT_KEYS)
        assert event['model_state_sha256'] == owner.core.state_sha(model.state_dict())
        assert event['rng_sha256'] == owner.core.state_sha(owner.core.training_rng())
        assert event['gradients_none'] is True
        assert event['issue_ns_sha256'] == owner.core.digest(context['truth']['issue_ns'][left:right])
        assert event['source_rows_sha256'] == owner.core.digest(context['source_rows'][left:right])
        for key, value in values.items():
            array = value.numpy() if isinstance(value, torch.Tensor) else value
            assert array.dtype == np.float32 and array.shape == (right-left, 5)
            assert owner.core.digest(array) == event['batch']['output_sha256'][key]
        events.append(('commit', event['batch_cursor']))
    owner.score_initial_segmented(model, context, batch_rows=3, device='cpu', commit_batch=commit,
        progress_callback=lambda event: events.append(('progress', event['batch_cursor'])))
    assert events == [(kind, cursor) for cursor in range(1, 4) for kind in ('commit', 'progress')]


def test_segmented_initial_commit_failure_does_not_report_progress_and_restores_caller_state(monkeypatch):
    _, context, calls = _segmented_case(monkeypatch)
    model = SegmentedTinyModel(context)
    before_rng = torch.get_rng_state().clone()
    before_state = owner.core.state_sha(model.state_dict())
    progress = []
    def fail(event, values): raise OSError('SYNTHETIC_DURABILITY_FAILURE')
    with pytest.raises(OSError, match='SYNTHETIC_DURABILITY_FAILURE'):
        owner.score_initial_segmented(model, context, batch_rows=3, device='cpu',
            commit_batch=fail, progress_callback=progress.append)
    assert calls == [3] and progress == []
    assert model.training and torch.equal(torch.get_rng_state(), before_rng)
    assert owner.core.state_sha(model.state_dict()) == before_state
    assert all(parameter.grad is None for parameter in model.parameters())
