"""Synthetic opt-in v2 view and batch routing; no physical fit or TEST."""
from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest
import torch

from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts import market_state_time_query_batch_v1 as old_batch
from gx1.contracts import market_state_time_query_batch_v2 as batch
from gx1.contracts.market_state_time_query_model_v2 import (
    FactorizedTimeQueryMarketStateModel, factorized_log_probabilities,
    factorized_time_query_model_contract,
)
from tests.test_market_state_time_query_batch_v1 import _bound, _features, _model
from gx1.contracts.market_state_time_query_binding_v1 import sampled_query_k

pytest_plugins = (
    "tests.test_entry_native_direction_feature_view_v1",
    "tests.test_market_state_time_query_view_mode",
)


def _v2_arguments(time_query_case):
    args = deepcopy(time_query_case)
    args.pop('market_state_time_query')
    args['market_state_time_query_v2'] = True
    refs = args['source_bindings']
    refs.pop('time_query_model_source')
    refs.pop('time_query_loss_source')
    for key, rel in (
        ('time_query_v2_model_source', 'gx1/contracts/market_state_time_query_model_v2.py'),
        ('time_query_v2_loss_source', 'gx1/contracts/market_state_time_query_loss_v2.py'),
    ):
        path = views.time_query_targets.ROOT / rel
        refs[key] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return args


def test_v2_view_is_opt_in_and_rebinds(time_query_case):
    old = views.bind_native_direction_feature_view(**time_query_case)
    args = _v2_arguments(time_query_case)
    current = views.bind_native_direction_feature_view(**args)
    assert current['market_state_time_query_v2'] is True
    assert 'market_state_time_query' not in current
    assert current['research_objective'] == factorized_time_query_model_contract()
    assert current['contract_sha256'] != old['contract_sha256']
    assert views.bind_native_direction_feature_view(**args, expected=current) == current
    assert views.bind_native_direction_feature_view(**time_query_case, expected=old) == old


def test_v2_view_rejects_mixed_mode_and_foreign_source(time_query_case):
    args = _v2_arguments(time_query_case)
    with pytest.raises(RuntimeError, match='OBJECTIVE_MODE'):
        views.bind_native_direction_feature_view(**args, market_state_time_query=True)
    bad = deepcopy(args)
    bad['source_bindings']['time_query_v2_model_source']['path'] = '/tmp/other-model.py'
    with pytest.raises(RuntimeError, match='TIME_QUERY_V2_SOURCE_PATHS'):
        views.bind_native_direction_feature_view(**bad)
    bad = deepcopy(args)
    bad['source_bindings'].pop('time_query_v2_loss_source')
    with pytest.raises(RuntimeError, match='SOURCE_REF_KEYS'):
        views.bind_native_direction_feature_view(**bad)


def test_v2_builder_selects_factorized_model(time_query_case, monkeypatch):
    args = _v2_arguments(time_query_case)
    bound = views.bind_native_direction_feature_view(**args)
    calls = []

    class BackboneShell:
        def __init__(self, **kwargs):
            calls.append('backbone')
            self.entry_execution_profile = kwargs['entry_execution_profile']
            self.gc_mtf_history = kwargs['gc_mtf_history']

        def to(self, device):
            return self

        def require_input_normalization_state(self):
            calls.append('normalization')

    class QueryShell:
        def __init__(self, backbone):
            self.backbone = backbone
            calls.append('factorized_head')

    def forbidden(*args, **kwargs):
        raise AssertionError('legacy wrapper constructed for v2')

    monkeypatch.setattr(views, 'EntryV10CtxHybridTransformer', BackboneShell)
    monkeypatch.setattr(views, 'FactorizedTimeQueryMarketStateModel', QueryShell)
    monkeypatch.setattr(views, 'TimeQueryMarketStateModel', forbidden)
    monkeypatch.setattr(views, 'FiveEndpointMarketStateModel', forbidden)
    model = views.build_native_direction_research_model(
        bound, prefix_normalization=args['prefix_normalization'],
        constructor_metadata=args['constructor_metadata'], device='synthetic')
    assert calls == ['backbone', 'normalization', 'factorized_head']
    assert model.feature_view_identity['research_objective_sha256'] == (
        factorized_time_query_model_contract()['contract_sha256'])


def test_v2_batch_uses_raw_head_and_rejects_v1(monkeypatch):
    forwards = []
    model = FactorizedTimeQueryMarketStateModel(_model(monkeypatch, forwards).backbone)
    bound, parents, issues = _bound()
    report = batch.evaluate_fit_batch(model, _features(2), bound, parents, issues,
                                      fold_id=1, pass_index=0, seed=19)
    assert forwards == [1]
    assert report['count'] == 5 and report['skip_update'] is False
    assert report['allowed'][0].tolist() == [1, 2, 4, 5, 5]
    assert report['query_k'].shape == (2, 5)
    assert report['validity_ns'].shape == (2, 5)
    assert torch.isfinite(report['loss'])
    report['loss'].backward()
    assert model.query_head.weight.grad is not None
    assert torch.count_nonzero(model.query_head.weight.grad[0]) > 0
    assert torch.count_nonzero(model.query_head.weight.grad[1]) > 0
    with pytest.raises(TypeError, match='TIME_QUERY_BATCH_MODEL_OWNER'):
        old_batch.evaluate_fit_batch(model, _features(2), bound, parents, issues,
                                     fold_id=1, pass_index=0, seed=19)


def test_v2_model_output_matches_factorization(monkeypatch):
    model = FactorizedTimeQueryMarketStateModel(_model(monkeypatch, []).backbone)
    _, _, issues = _bound()
    query = torch.from_numpy(sampled_query_k(issues, pass_index=0, seed=19))
    output = model(**_features(2), query_k=query)
    raw = output['activity_side_logits']
    assert torch.equal(output['logits'], factorized_log_probabilities(raw))
    assert torch.allclose(output['logits'].exp().sum(dim=-1),
                          torch.ones_like(raw[..., 0]), rtol=0, atol=1e-6)
