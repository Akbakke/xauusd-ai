"""Synthetic OOF bridge regression: raw side survives underflow, no data IO."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gx1.audit.market_state_time_query_oof_batch_v2 import score_oof_issue_batch
from gx1.contracts.market_state_five_endpoint_target_binding_v1 import FOLD_BOUNDARIES_NS
from gx1.contracts.market_state_time_query_scorer_v2 import FactorizedTimeQueryMatchedScorer
from gx1.contracts.market_state_time_query_target_v1 import TICK_NS


class _RawModel:
    def __init__(self):
        self.encodes = 0
        self.decoded = []

    def encode_features(self, **features):
        self.encodes += 1
        return features['x']

    def decode_raw_query_grid(self, hidden, query):
        self.decoded.append(query.detach().cpu().numpy().copy())
        raw = torch.zeros((*query.shape, 2), dtype=torch.float32)
        raw[..., 0] = -1000.0
        return raw


class _Bound:
    def gather(self, parent_rows, issue_ns, query_k, *, fold_id, partition):
        assert fold_id == 1 and partition == 'score'
        n = len(parent_rows)
        allowed = np.full((n, 5), 7, dtype=np.uint8)
        allowed[:, 0] = 4
        eligible = np.zeros((n, 5), dtype=np.bool_)
        eligible[:, 0] = True
        return {'parent_rows': parent_rows.copy(), 'issue_ns': issue_ns.copy(),
                'query_k': query_k.copy(),
                'validity_ns': issue_ns[:, None] + query_k * TICK_NS,
                'allowed': allowed, 'eligible': eligible,
                'mature': np.ones((n, 5), dtype=np.bool_)}


def _priors():
    return (np.full((288, 5), .5, dtype=np.float64),
            np.full((288, 5), .5, dtype=np.float64),
            np.broadcast_to(np.array([.25, .5, .25], dtype=np.float64),
                            (288, 5, 3)).copy())


def _axes():
    return (np.array([0], dtype=np.int64),
            np.array([FOLD_BOUNDARIES_NS[0] + 24 * 60 * 60 * 1_000_000_000],
                     dtype=np.int64))


def test_underflowed_activity_keeps_conditional_side_in_full_oof_batch():
    rows, issues = _axes()
    model = _RawModel()
    scorer = FactorizedTimeQueryMatchedScorer()
    deadlines = []
    score_oof_issue_batch(model, {'x': torch.zeros((1, 2))}, rows, issues,
                          bound=_Bound(), scorer=scorer, priors=_priors(),
                          device=torch.device('cpu'), fold_id=1,
                          check_deadline=lambda: deadlines.append(True))
    result = scorer.result()
    assert model.encodes == 1
    assert len(model.decoded) == len(deadlines) == 9
    assert model.decoded[0].shape == (1, 32, 5)
    assert model.decoded[-1].shape == (1, 32, 5)
    assert result['primary_M5_k1']['certified_side']['count'] == 1
    assert result['primary_M5_k1']['certified_side']['model_brier_sum'] == pytest.approx(.25)
    assert result['primary_M5_k1']['activity']['model_brier_sum'] == pytest.approx(1.0)
    assert result['per_tf_k']['M5']['1']['three_class_model_brier_sum'] == pytest.approx(2.0)
    assert result['issue_axis']['1']['count'] == 1


def test_v1_only_decoder_and_invalid_priors_fail_before_scoring():
    rows, issues = _axes()
    scorer = FactorizedTimeQueryMatchedScorer()

    class _V1Only:
        def encode_features(self, **features):
            return features['x']

        def decode_query_grid(self, hidden, query):
            return torch.zeros((*query.shape, 3))

    with pytest.raises(ValueError, match='OWNERS'):
        score_oof_issue_batch(
            _V1Only(), {'x': torch.zeros((1, 2))}, rows, issues,
            bound=_Bound(), scorer=scorer, priors=_priors(),
            device=torch.device('cpu'), fold_id=1, check_deadline=lambda: None)
    assert scorer._active is None

    bad = list(_priors())
    bad[1] = np.full((288, 5), .5, dtype=np.float32)
    with pytest.raises(ValueError, match='PRIOR_DOMAIN'):
        score_oof_issue_batch(
            _RawModel(), {'x': torch.zeros((1, 2))}, rows, issues,
            bound=_Bound(), scorer=scorer, priors=tuple(bad),
            device=torch.device('cpu'), fold_id=1, check_deadline=lambda: None)
    assert scorer._active is None
