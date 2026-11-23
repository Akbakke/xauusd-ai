"""Synthetic CPU-only final scoring; no native backbone, payload or checkpoint IO."""
import numpy as np
import pytest
import torch

from gx1.audit import native_future_state_provider_hlc_baseline_v1 as owner
from tests.test_native_future_state_provider_hlc_baseline_v1 import _setup, _mock_forward, TinyModel


class TrainedModel(TinyModel):
    def forward(self, **features):
        output = super().forward(**features)
        delta = (features['seq_x'][:, :1].float() - 100) / 10
        return {key: value + delta for key, value in output.items()}


def test_final_scores_full_axis_once_with_unknown_rows_and_returns_owned_outputs(monkeypatch):
    provider, meta, _, truth, prior, loaded = _setup(monkeypatch)
    context = owner.load_context(meta); calls = _mock_forward(monkeypatch)
    model = TrainedModel(context); original_score = owner.objective.native_future_state_score_sums
    seen_truth = []
    def score(**kwargs):
        assert kwargs['truth'] is truth
        seen_truth.append(kwargs['truth'])
        return original_score(**kwargs)
    monkeypatch.setattr(owner.objective, 'native_future_state_score_sums', score)
    before = owner.core.state_sha(model.state_dict()); rng = torch.get_rng_state().clone(); progress = []
    report, outputs = owner.score_final(model, context, batch_rows=1, device='cpu',
                                        progress_callback=progress.append)
    assert report['rows'] == report['actual_NN_forward_rows'] == 8
    assert report['whole_TRAIN_final'] is True and 'initial_prior_exact' not in report
    assert report['batch_count'] == model.calls == 8 and calls == [1] * 8
    assert [p['rows_done'] for p in progress] == list(range(1, 9))
    assert report['batches'][5]['known_activity_cells'] == 0
    assert report['batches'][5]['partial_NLL'] == 0
    np.testing.assert_array_equal(np.concatenate(provider.calls), context['source_rows'])
    assert len(loaded) == 1 and len(seen_truth) == 2
    assert set(outputs) == {'activity_logits', 'side_logits'}
    assert all(value.dtype == np.float32 and value.shape == (8, 5) and value.flags.owndata
               and value.flags.c_contiguous and not value.flags.writeable for value in outputs.values())
    assert report['output_sha256'] == {key: owner.core.digest(value) for key, value in outputs.items()}
    assert report['output_ndarray_bytes'] == 8 * 5 * 4 * 2
    as_tensors = {key: torch.from_numpy(value.copy()) for key, value in outputs.items()}
    assert report['scores'] == original_score(model_output=as_tensors, truth=truth,
        fold_upper_ns=owner.UPPER, target_contract=context['target_contract'])
    constant = owner.objective.native_future_state_train_prior_logits(prior, rows=8,
        target_contract=context['target_contract'])
    assert report['constant_references']['whole_TRAIN_factory'] == original_score(
        model_output=constant, truth=truth, fold_upper_ns=owner.UPPER,
        target_contract=context['target_contract'])
    assert report['scores'] != report['constant_references']['whole_TRAIN_factory']
    assert report['rank_and_variation'] == owner.training.rank_metrics(outputs, truth, context['target_contract'])
    assert report['loss_score_abs_delta'] < 1e-6
    assert report['target_observation_certificate_ref'] == truth['target_observation_certificate_ref']
    assert model.training is True and owner.core.state_sha(model.state_dict()) == before
    assert torch.equal(torch.get_rng_state(), rng) and all(p.grad is None for p in model.parameters())


def test_final_accepts_outputs_that_differ_from_fresh_prior(monkeypatch):
    _, meta, _, _, _, _ = _setup(monkeypatch); context = owner.load_context(meta)
    _mock_forward(monkeypatch)
    report, _ = owner.score_final(TinyModel(context, bad='wrong_prior'), context,
                                 batch_rows=3, device='cpu')
    assert report['batch_count'] == 3 and report['whole_TRAIN_final'] is True


@pytest.mark.parametrize('bad', ['short_clock', 'wrong_clock_sha', 'changed_prior_truth'])
def test_final_rejects_changed_full_clock_or_truth_before_any_forward(monkeypatch, bad):
    provider, meta, _, _, _, _ = _setup(monkeypatch); context = owner.load_context(meta)
    calls = _mock_forward(monkeypatch)
    if bad == 'short_clock':
        context['source_rows'] = context['source_rows'][:-1]
    elif bad == 'wrong_clock_sha':
        monkeypatch.setattr(owner, 'TRAIN_SHA', 'f' * 64)
    else:
        context['truth_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='FINAL_'):
        owner.score_final(TinyModel(context), context, batch_rows=3, device='cpu')
    assert calls == [] and provider.calls == []


def test_final_rejects_preexisting_gradients_without_clearing_them(monkeypatch):
    provider, meta, _, _, _, _ = _setup(monkeypatch); context = owner.load_context(meta)
    calls = _mock_forward(monkeypatch); model = TinyModel(context)
    model.anchor.grad = torch.ones_like(model.anchor); before = model.anchor.grad.clone()
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='FINAL_PREEXISTING_GRADIENTS'):
        owner.score_final(model, context, batch_rows=3, device='cpu')
    assert torch.equal(model.anchor.grad, before) and torch.equal(torch.get_rng_state(), rng)
    assert model.training is True and calls == [] and provider.calls == []


@pytest.mark.parametrize('failure', ['nonfinite', 'callback'])
def test_final_failure_restores_mode_rng_and_unchanged_model(monkeypatch, failure):
    provider, meta, _, _, _, _ = _setup(monkeypatch); context = owner.load_context(meta)
    calls = _mock_forward(monkeypatch); model = TinyModel(context, bad=failure)
    model.eval(); rng = torch.get_rng_state().clone(); before = owner.core.state_sha(model.state_dict())
    def callback(_):
        raise RuntimeError('synthetic callback failure')
    with pytest.raises((ValueError, RuntimeError)):
        owner.score_final(model, context, batch_rows=3, device='cpu',
            progress_callback=callback if failure == 'callback' else None)
    assert model.calls == len(provider.calls) == 1 and calls == [3]
    assert model.training is False and torch.equal(torch.get_rng_state(), rng)
    assert owner.core.state_sha(model.state_dict()) == before and all(p.grad is None for p in model.parameters())


def test_final_detects_parameter_mutation_after_restoring_mode_and_rng(monkeypatch):
    _, meta, _, _, _, _ = _setup(monkeypatch); context = owner.load_context(meta)
    _mock_forward(monkeypatch); model = TinyModel(context)
    rng = torch.get_rng_state().clone()
    def callback(_):
        with torch.no_grad():
            model.anchor.add_(1)
    with pytest.raises(ValueError, match='FINAL_SCORE_STATE_GRAD_MUTATION'):
        owner.score_final(model, context, batch_rows=3, device='cpu', progress_callback=callback)
    assert model.training is True and torch.equal(torch.get_rng_state(), rng)


def test_final_rejects_certificate_metadata_in_feature_payload(monkeypatch):
    provider, meta, _, _, _, _ = _setup(monkeypatch); context = owner.load_context(meta)
    calls = _mock_forward(monkeypatch); provider.extra_metadata = True
    rng = torch.get_rng_state().clone(); model = TinyModel(context)
    with pytest.raises(ValueError, match='FEATURE_KEYS'):
        owner.score_final(model, context, batch_rows=3, device='cpu')
    assert calls == [] and model.training is True and torch.equal(torch.get_rng_state(), rng)
