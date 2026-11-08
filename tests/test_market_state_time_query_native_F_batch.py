from __future__ import annotations

import torch

from gx1.audit import verify_market_state_time_query_v2_zero_step_v1 as zero
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
from gx1.contracts import market_state_future_displacement_candidate_v1 as future
from gx1.contracts import market_state_time_query_binding_v1 as query
from gx1.contracts import market_state_time_query_native_F_batch_v1 as batch
from gx1.contracts.market_state_time_query_model_v2 import factorized_log_probabilities
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K


class RecordingModel:
    """Only the adapter protocol matters; targets must not enter forward."""
    def __init__(self):
        self.state_model = type('State', (), {'query_native_k': torch.tensor(NATIVE_K)})()
        self.research_objective = {'contract_sha256': 'synthetic'}
        self.keys = None

    def __call__(self, **kwargs):
        self.keys = set(kwargs)
        n = kwargs['seq_x'].shape[0]
        raw = torch.zeros((n, 5, 2), dtype=torch.float32, requires_grad=True)
        future = raw.clone()
        return {'state_activity_side_logits': raw,
                'state_log_probs': factorized_log_probabilities(raw),
                'future_activity_side_logits': future,
                'future_log_probs': factorized_log_probabilities(future)}


def test_feature_only_forward_on_actual_bound_train_rows(monkeypatch):
    # Synthetic forward keeps the expensive backbone out; target data are real.
    bound_state = query.load_bound_time_query_targets()
    bound_future = future.load_bound_future_displacement()
    axes = state.load_bound_five_endpoint_targets().mature_oof_axes(fold_id=1)
    rows = axes['fit_parent_rows'][:8]
    issues = axes['fit_decision_ns'][:8]
    # The batch owner explicitly requires its dual model class; patch only the
    # class identity here so this test can inspect the exact forward boundary.
    monkeypatch.setattr(batch, 'NativeFutureDualModel', RecordingModel)
    model = RecordingModel()
    features = {key: torch.zeros((8, 1), dtype=torch.float32)
                for key in batch.FEATURE_KEYS}
    features['seq_x'] = torch.zeros((8, 1, 4), dtype=torch.float32)
    out = batch.evaluate_fit_batch(
        model, features, bound_state, bound_future, rows, issues,
        fold_id=1, pass_index=zero.PASS_INDEX, seed=zero.SEED,
        future_weight=0.25)
    assert model.keys == set(batch.FEATURE_KEYS) | {'query_k'}
    assert out['state_count'] > 0 and out['future_count'] == 40
    assert out['future_native_k'].tolist() == list(NATIVE_K)
    assert out['state_query_k'][0].tolist() != list(NATIVE_K)
    assert bool(torch.isfinite(out['loss']))
