"""Fixed integrated S/F plus masked structure objective arithmetic."""
import numpy as np
import pytest
import torch

from gx1.contracts.market_state_ta_integrated_model_v1 import masked_structure_hit_sums
from gx1.contracts.market_state_ta_integrated_objective_v1 import (
    STRUCTURE_WEIGHT, contract, integrated_loss,
)
from gx1.contracts.market_state_time_query_model_v2 import factorized_log_probabilities
from gx1.contracts.market_state_time_query_native_F_objective_v1 import combined_loss
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K


def _inputs():
    state = torch.randn(3, 5, 2, dtype=torch.float32, requires_grad=True)
    future = torch.randn(3, 5, 2, dtype=torch.float32, requires_grad=True)
    structure = torch.randn(3, 2, dtype=torch.float32, requires_grad=True)
    output = {'state_activity_side_logits': state,
              'state_log_probs': factorized_log_probabilities(state),
              'future_activity_side_logits': future,
              'future_log_probs': factorized_log_probabilities(future),
              'm5_structure_hit_logits': structure}
    allowed = np.array([[1, 4, 1, 4, 1], [4, 1, 4, 1, 4],
                        [2, 5, 7, 1, 4]], dtype=np.uint8)
    truth = {'allowed': allowed, 'eligible': np.ones((3, 5), dtype=np.bool_),
             'mature': np.ones((3, 5), dtype=np.bool_)}
    hits = np.array([[True, False], [False, True], [False, False]], dtype=np.bool_)
    observed = np.array([[True, True], [False, True], [False, False]], dtype=np.bool_)
    return output, truth, hits, observed, torch.tensor([NATIVE_K] * 3, dtype=torch.int64)


def test_integrated_loss_equals_bound_components_and_masks_unknown_gradients():
    output, truth, hits, observed, query = _inputs()
    report = integrated_loss(model_output=output, state_truth=truth,
        future_truth=truth, state_query_k=query,
        structure_hit=hits, structure_observed=observed)
    sf = combined_loss(state_raw=output['state_activity_side_logits'],
        future_raw=output['future_activity_side_logits'],
        state_truth=truth, future_truth=truth, state_query_k=query,
        future_weight=1.0, ta_only=True, dedicated_m5_side=True,
        dedicated_non_m5_side=True, dedicated_tf_activity=True,
        direct_tf_hidden=True, m5_equal_observation=True, fold_id=2)
    structural = masked_structure_hit_sums(output['m5_structure_hit_logits'],
        torch.from_numpy(hits), torch.from_numpy(observed))
    expected_structure = (structural['loss_sums'] /
                          structural['observed_counts'].clamp(min=1)).mean()
    assert torch.equal(report['S_F_loss'], sf['loss'])
    assert torch.equal(report['structure_loss'], expected_structure)
    assert torch.equal(report['loss'], sf['loss'] + STRUCTURE_WEIGHT * expected_structure)
    assert report['structure_observed_counts'].tolist() == [1, 2]
    assert report['contract_sha256'] == contract()['contract_sha256']
    report['loss'].backward()
    gradient = output['m5_structure_hit_logits'].grad
    assert gradient[1, 0] == gradient[2, 0] == gradient[2, 1] == 0
    assert gradient[0, 0] != 0 and gradient[0, 1] != 0 and gradient[1, 1] != 0
    assert torch.count_nonzero(output['state_activity_side_logits'].grad)
    assert torch.count_nonzero(output['future_activity_side_logits'].grad)


def test_integrated_loss_rejects_unobserved_positive_and_broken_factorization():
    output, truth, hits, observed, query = _inputs()
    hits[1, 0] = True
    with pytest.raises(ValueError, match='STRUCTURE'):
        integrated_loss(model_output=output, state_truth=truth, future_truth=truth,
                        state_query_k=query, structure_hit=hits,
                        structure_observed=observed)
    hits[1, 0] = False
    output['state_log_probs'] = output['state_log_probs'] + 1
    with pytest.raises(ValueError, match='FACTOR_PROBABILITIES'):
        integrated_loss(model_output=output, state_truth=truth, future_truth=truth,
                        state_query_k=query, structure_hit=hits,
                        structure_observed=observed)
