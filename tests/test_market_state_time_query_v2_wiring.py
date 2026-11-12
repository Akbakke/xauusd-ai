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
        def __init__(self, backbone, *, dedicated_m5_side, dedicated_non_m5_side,
                     dedicated_tf_activity):
            assert (dedicated_m5_side, dedicated_non_m5_side, dedicated_tf_activity) == (False, False, False)
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
    assert report['activity_side_logits'].shape == (2, 5, 2)
    assert report['activity_side_logits'].requires_grad
    assert torch.equal(report['logits'], factorized_log_probabilities(
        report['activity_side_logits']))
    report['activity_side_logits'].retain_grad()
    report['loss'].backward()
    assert report['activity_side_logits'].grad[0, 3, 1] == 0
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


def test_ta_view_gathers_only_canonical_columns_and_never_calls_gc(time_query_case, bound_case, monkeypatch):
    import numpy as np
    from gx1.features.entry_specialist_feature_groups_v1 import (
        compile_full_ta_entry_execution_profile, FULL_TA_ENTRY_FEATURE_KEYS)
    from tests.test_entry_native_direction_feature_view_v1 import _inputs
    args = _v2_arguments(time_query_case)
    old = views.bind_native_direction_feature_view(**args)
    args['full_ta_feature_profile'] = True
    args['source_bindings']['owner_profile_source']['sha256'] = '2' * 64
    ta = views.bind_native_direction_feature_view(**args)
    assert ta['prefix_normalization']['source_owner_profile_sha256'] == '1' * 64
    assert ta['source_bindings']['owner_profile_source']['sha256'] == '2' * 64
    assert ta['research_objective'] == factorized_time_query_model_contract(ta_only=True)
    assert ta['research_objective'] != old['research_objective']
    assert views.bind_native_direction_feature_view(**args, expected=ta) == ta
    inputs = _inputs(deepcopy(bound_case))
    before = views.gather_native_direction_feature_batch(old, **inputs)
    inputs.update(gc_layout=None, gc_surfaces=None)
    def forbidden(*args, **kwargs):
        raise AssertionError('TA candidate reached GC gather or surface validator')
    monkeypatch.setattr(views.history, 'require_gc_mtf_history_layout', forbidden)
    monkeypatch.setattr(views.history, 'require_gc_mtf_history_surfaces', forbidden)
    monkeypatch.setattr(views.history, 'gather_gc_mtf_history_rows', forbidden)
    got = views.gather_native_direction_feature_batch(ta, **inputs)
    profile = compile_full_ta_entry_execution_profile()
    assert set(got['features']) == FULL_TA_ENTRY_FEATURE_KEYS
    for key, indices in (('seq_x', profile['source_signal_indices']),
                         ('snap_x', profile['source_signal_indices']),
                         ('ctx_cont', profile['source_ctx_cont_indices'])):
        assert np.array_equal(got['features'][key], np.take(before['features'][key], indices, axis=-1))
    for key in ('ctx_cat', 'seq_m5', 'seq_m15', 'seq_h1', 'seq_h4', 'seq_d1'):
        assert got['features'][key].tobytes() == before['features'][key].tobytes()
    assert {k: v for k, v in got['witness'].items() if k != 'view_sha256'} == {
        k: v for k, v in before['witness'].items() if k != 'view_sha256'}
    with pytest.raises(RuntimeError, match='TA_GC_INPUT_FORBIDDEN'):
        views.gather_native_direction_feature_batch(ta, **{**inputs, 'gc_surfaces': {}})


def test_ta_builder_passes_reduced_statistics_and_no_gc_modules(time_query_case, monkeypatch):
    args = _v2_arguments(time_query_case)
    args['full_ta_feature_profile'] = True
    bound = views.bind_native_direction_feature_view(**args)
    calls = []
    class BackboneShell:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.entry_execution_profile = kwargs['entry_execution_profile']
            self.gc_mtf_history = kwargs['gc_mtf_history']
        def to(self, device):
            return self
        def require_input_normalization_state(self):
            return None
    class QueryShell:
        def __init__(self, backbone, *, dedicated_m5_side, dedicated_non_m5_side,
                     dedicated_tf_activity):
            assert (dedicated_m5_side, dedicated_non_m5_side, dedicated_tf_activity) == (False, False, False)
            self.backbone = backbone
    monkeypatch.setattr(views, 'EntryV10CtxHybridTransformer', BackboneShell)
    monkeypatch.setattr(views, 'FactorizedTimeQueryMarketStateModel', QueryShell)
    model = views.build_native_direction_research_model(
        bound, prefix_normalization=args['prefix_normalization'],
        constructor_metadata=args['constructor_metadata'], device='synthetic')
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs['seq_input_dim'] == kwargs['snap_input_dim'] == 317
    assert kwargs['ctx_cont_dim'] == 132
    assert kwargs['gc_mtf_history'] is None
    assert 'research_unobserved_units_profile' not in kwargs
    assert kwargs['input_normalization']['contract_sha256'] == (
        bound['prefix_normalization']['ta_base_normalization_sha256'])
    assert kwargs['input_normalization']['ta_projection']['source_contract'] == (
        args['prefix_normalization']['base_normalization'])
    assert model.feature_view_identity['prefix_normalization'] == bound['prefix_normalization']
