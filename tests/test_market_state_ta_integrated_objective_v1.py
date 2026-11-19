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


def test_structure_readout_uses_bool_known_labels():
    from gx1.audit.run_market_state_ta_integrated_fold2_fit_v1 import structure_readout

    logits = np.array([[2.0, -2.0], [-2.0, 2.0], [100.0, -100.0]], dtype=np.float32)
    hit = np.array([[True, False], [False, True], [False, False]], dtype=bool)
    observed = np.array([[True, True], [True, True], [False, False]], dtype=bool)
    result = structure_readout(logits, hit, observed, np.array([.5, .5]))
    assert result['high']['observed'] == result['low']['observed'] == 2
    assert result['high']['auc'] == result['low']['auc'] == 1.0
    assert result['high']['hit'] == result['low']['hit'] == 1
    assert result['high']['brier'] < result['high']['prior_brier']


def _native_bar_truth():
    issues = np.array([1, 2, 3], dtype=np.int64) * 300_000_000_000
    validity = np.broadcast_to(issues[:, None] + 300_000_000_000, (3, 5)).copy()
    mature = validity < 1_200_000_000_000
    target_rows = np.full((3, 5), -1, dtype=np.int64)
    target_rows[:2] = [[2] * 5, [3] * 5]
    target_rows[:2, 1:] = 2
    previous = mature.copy()
    open_known = np.zeros((3, 5), dtype=np.bool_)
    open_known[:2, 1:] = True
    signs = np.array([[-1, 0, 1, -1, 0], [0, 1, -1, 0, 1], [0] * 5], dtype=np.int8)
    return {'issue_ns': issues, 'validity_ns': validity,
            'target_row': target_rows, 'target_present': np.ones((3, 5), dtype=np.bool_),
            'mature': mature, 'observed_previous': previous, 'eligible': mature.copy(),
            'open_known_at_issue': open_known, 'open_sign': signs, 'close_sign': signs.copy()}


def test_native_bar_loss_keeps_equality_masks_known_open_and_matches_proper_scores():
    from gx1.contracts.market_state_native_m5_targets_v1 import contract as target_contract
    from gx1.contracts.market_state_ta_integrated_objective_v1 import (
        native_bar_direction_loss, native_bar_direction_score_sums,
    )
    output = {key: torch.zeros(3, 5, 3, requires_grad=True)
              for key in ('open_logits', 'close_logits')}
    kwargs = {'model_output': output, 'truth': _native_bar_truth(), 'fold_upper_ns': 1_200_000_000_000,
              'target_contract_sha256': target_contract()['contract_sha256']}
    result = native_bar_direction_loss(**kwargs)
    assert result['counts']['open'].tolist() == [2, 0, 0, 0, 0]
    assert result['counts']['close'].tolist() == [2] * 5
    assert float(result['loss']) == pytest.approx(np.log(3))
    result['loss'].backward()
    assert torch.count_nonzero(output['open_logits'].grad[:, 1:]) == 0
    assert torch.count_nonzero(output['close_logits'].grad[-1]) == 0
    assert output['close_logits'].grad[0, 1, 1] < 0  # equality is class 1
    score = native_bar_direction_score_sums(**kwargs)['per_task']
    assert score['open']['M5']['count'] == 2
    assert score['open_known']['H4']['count'] == 2
    assert score['close']['H4']['count'] == 2
    assert score['close']['H4']['unique_target_rows'] == 1
    assert score['close']['H4']['max_logit_ties'] == 2
    assert score['close']['H4']['log_loss_sum'] == pytest.approx(2 * np.log(3))
    assert score['close']['H4']['brier_sum'] == pytest.approx(4 / 3)
    assert np.asarray(score['close']['H4']['confusion_truth_by_prediction']).sum() == 2


def test_native_bar_objective_rejects_old_target_identity_and_false_maturity():
    from gx1.contracts.market_state_native_m5_targets_v1 import contract as target_contract
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_direction_loss
    output = {key: torch.zeros(3, 5, 3) for key in ('open_logits', 'close_logits')}
    kwargs = {'model_output': output, 'truth': _native_bar_truth(), 'fold_upper_ns': 1_200_000_000_000,
              'target_contract_sha256': '0' * 64}
    with pytest.raises(ValueError, match='TARGET_CONTRACT'):
        native_bar_direction_loss(**kwargs)
    kwargs['target_contract_sha256'] = target_contract()['contract_sha256']
    kwargs['truth']['mature'][0, 0] = False
    with pytest.raises(ValueError, match='OBSERVABILITY'):
        native_bar_direction_loss(**kwargs)
