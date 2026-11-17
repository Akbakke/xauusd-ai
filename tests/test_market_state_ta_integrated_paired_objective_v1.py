"""Paired S weighting and unknown-mask controls for the integrated campaign."""
import numpy as np
import pytest
import torch

from gx1.contracts.market_state_ta_integrated_paired_objective_v1 import paired_integrated_loss
from gx1.contracts.market_state_time_query_model_v2 import factorized_log_probabilities
from gx1.contracts.market_state_time_query_native_F_objective_v1 import combined_loss
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K


def _truth(bits=None):
    allowed = np.array(bits if bits is not None else [7] * 5, dtype=np.uint8)
    return {'allowed': allowed[None, :],
            'eligible': np.ones((1, 5), dtype=np.bool_),
            'mature': np.ones((1, 5), dtype=np.bool_)}


def _case(sampled_truth=None, native_truth=None, future_truth=None,
          sampled_k=None, shared_raw=None):
    sampled = (shared_raw if shared_raw is not None else
               torch.zeros((1, 5, 2), dtype=torch.float32, requires_grad=True))
    native = (shared_raw if shared_raw is not None else
              torch.zeros((1, 5, 2), dtype=torch.float32, requires_grad=True))
    future = torch.zeros((1, 5, 2), dtype=torch.float32, requires_grad=True)
    structure = torch.zeros((1, 2), dtype=torch.float32, requires_grad=True)
    output = {'state_activity_side_logits': sampled,
              'state_log_probs': factorized_log_probabilities(sampled),
              'state_native_activity_side_logits': native,
              'state_native_log_probs': factorized_log_probabilities(native),
              'future_activity_side_logits': future,
              'future_log_probs': factorized_log_probabilities(future),
              'm5_structure_hit_logits': structure}
    query = torch.tensor([[sampled_k or NATIVE_K, NATIVE_K]], dtype=torch.int64)
    args = dict(model_output=output,
                sampled_truth=sampled_truth or _truth(),
                native_truth=native_truth or _truth(),
                future_truth=future_truth or _truth(),
                query_grid=query,
                structure_hit=np.zeros((1, 2), dtype=np.bool_),
                structure_observed=np.zeros((1, 2), dtype=np.bool_))
    return args


def test_native_h4_d1_known_side_updates_when_sampled_and_f_unknown():
    args = _case(native_truth=_truth([7, 7, 7, 4, 1]),
                 sampled_k=[1, 1, 2, 4, 24])
    report = paired_integrated_loss(**args)
    assert report['sampled_S_counts'] == report['native_F_counts'] == {
        'M5': 0, 'M15': 0, 'H1': 0, 'H4': 0, 'D1': 0}
    assert report['native_S_counts']['H4'] == report['native_S_counts']['D1'] == 1
    assert report['skip_update'] is False
    report['loss'].backward()
    native_grad = args['model_output']['state_native_activity_side_logits'].grad
    assert torch.count_nonzero(native_grad[0, 3:]) == 4
    assert torch.count_nonzero(native_grad[0, :3]) == 0
    assert torch.count_nonzero(args['model_output']['state_activity_side_logits'].grad) == 0
    assert torch.count_nonzero(args['model_output']['future_activity_side_logits'].grad) == 0
    assert torch.count_nonzero(args['model_output']['m5_structure_hit_logits'].grad) == 0


def test_duplicate_native_m5_does_not_double_s_mass():
    raw = torch.zeros((1, 5, 2), dtype=torch.float32, requires_grad=True)
    state = _truth([4, 1, 2, 5, 7])
    future = _truth([1, 4, 2, 7, 5])
    args = _case(sampled_truth=state, native_truth=state,
                 future_truth=future, shared_raw=raw)
    report = paired_integrated_loss(**args)
    old = combined_loss(state_raw=raw,
                        future_raw=args['model_output']['future_activity_side_logits'],
                        state_truth=state, future_truth=future,
                        state_query_k=args['query_grid'][:, 0], future_weight=1.0,
                        ta_only=True, dedicated_m5_side=True,
                        dedicated_non_m5_side=True, dedicated_tf_activity=True,
                        direct_tf_hidden=True, m5_equal_observation=True, fold_id=2)
    torch.testing.assert_close(report['loss'], old['loss'], atol=0, rtol=0)
    assert report['structure_loss'].item() == 0


def test_m5_target_parity_is_required():
    args = _case(native_truth=_truth([4, 7, 7, 7, 7]))
    with pytest.raises(ValueError, match='TA_PAIRED_M5_TARGET_PARITY'):
        paired_integrated_loss(**args)
