"""Check the reused scorer's batch tails and native direction diagnostics."""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gx1.audit import run_market_state_ta_apr_jun_paired_fit_v1 as paired
from gx1.audit import observe_market_state_ta_fold2_fresh_initial_v1 as initial
from gx1.audit import run_market_state_ta_integrated_fold2_fit_v1 as integrated
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as five
from gx1.contracts import market_state_future_displacement_candidate_v1 as future
from gx1.contracts import market_state_time_query_binding_v1 as sampled
from gx1.contracts.market_state_time_query_target_v1 import TICK_NS


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.batch_sizes = []

    def forward(self, features, *, query_k):
        n = len(query_k)
        self.batch_sizes.append(n)
        z = torch.zeros((n, 5, 2))
        return {'state_activity_side_logits': z,
                'state_native_activity_side_logits': z,
                'future_activity_side_logits': z,
                'm5_structure_hit_logits': torch.zeros((n, 2)),
                'm5_cost_path_bps': torch.zeros((n, 6, 2, 3))}


@pytest.mark.parametrize(('n', 'scope', 'sizes'), [
    (16, (3, 'score'), [16]), (17, (1, 'score'), [16, 1])])
def test_score_full_batch_and_one_row_tail(monkeypatch, n, scope, sizes):
    def truth(rows, *args, **kwargs):
        assert (kwargs['fold_id'], kwargs['partition']) == scope
        return {'allowed': np.ones((len(rows), 5), dtype=np.uint8),
                'eligible': np.ones((len(rows), 5), dtype=np.bool_),
                'mature': np.ones((len(rows), 5), dtype=np.bool_)}

    for module, loader in ((sampled, 'load_bound_time_query_targets'),
                           (five, 'load_bound_five_endpoint_targets'),
                           (future, 'load_bound_future_displacement')):
        monkeypatch.setattr(module, loader, lambda: SimpleNamespace(gather=truth))
    monkeypatch.setattr(sampled, 'sampled_query_k',
                        lambda issues, **kw: np.ones((len(issues), 5), dtype=np.int64))
    monkeypatch.setattr(paired, '_proper', lambda *a, **kw: {})
    monkeypatch.setattr(paired, '_turn_and_disagreement', lambda *a: {})
    monkeypatch.setattr(initial, 'direction_readout', lambda *a: {})
    monkeypatch.setattr(integrated, 'structure_readout', lambda *a: {})
    tape = {'structure_hit': np.zeros((n, 2), dtype=np.bool_),
            'structure_observed': np.ones((n, 2), dtype=np.bool_)}
    fit_tape = {'structure_hit': np.array([[True, False], [False, True]]),
                'structure_observed': np.ones((2, 2), dtype=np.bool_)}
    context = {'axes': {'parent_rows': np.arange(n, dtype=np.int64),
                        'decision_ns': np.arange(n, dtype=np.int64) * 300_000_000_000},
               'selected': np.arange(n, dtype=np.int64), 'score_tape': tape,
               'fit_tape': fit_tape, 'features': lambda label, rows, issues: {},
               'priors': {'tasks': {'state': {'raw_logits': [[0, 0]] * 5},
                                    'future': {'raw_logits': [[0, 0]] * 5}}}}
    model = _Model()
    result, capture = paired.score(model, context, deadline=time.monotonic() + 60,
                                   fold_id=scope[0], partition=scope[1],
                                   collect_decision=True)
    assert model.batch_sizes == sizes
    assert result['rows'] == n and result['model_forwards'] == len(sizes)
    assert capture['m5_cost_path_bps'].shape == (n, 6, 2, 3)


def test_score_rejects_fold2_fit_before_reading_context_or_truth(monkeypatch):
    def unexpected_target_read():
        pytest.fail('unexpected target read')

    monkeypatch.setattr(sampled, 'load_bound_time_query_targets', unexpected_target_read)
    with pytest.raises(RuntimeError, match='SCORE_TRUTH_SCOPE'):
        paired.score(None, None, deadline=time.monotonic(), fold_id=2, partition='fit')


def test_turns_require_correct_destination_and_disagreement_reports_native_queries():
    issues = np.array([0, 1, 2, 3, 4, 6, 7], dtype=np.int64) * TICK_NS
    allowed = np.full((len(issues), 5), 4, dtype=np.uint8)
    allowed[:, 0] = [4, 1, 1, 4, 4, 1, 7]
    known = np.ones(allowed.shape, dtype=np.bool_)
    raw = np.zeros((*allowed.shape, 2), dtype=np.float32)
    raw[:, 0, 1] = [-1, 1, -1, 1, -1, 1, -1]

    result = paired._turn_and_disagreement(
        raw, {'allowed': allowed, 'eligible': known, 'mature': known}, issues)

    assert result['M5_turns'] == {
        'known_contiguous_pairs': 4,
        'actual_turns': 2,
        'predicted_turns': 4,
        'captured_actual_turns': 1,
        'reverse_direction_turns': 1,
        'false_predicted_turns': 2,
    }
    h4 = result['TF_disagreement']['H4']
    assert (h4['alignment'], h4['m5_query_k'], h4['other_query_k']) == ('same_issue', 1, 48)
    assert (h4['joint_known'], h4['opposite_truth']) == (6, 3)
