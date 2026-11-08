import numpy as np
import torch

from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts import market_state_time_query_loss_v2 as legacy_loss
from gx1.contracts import market_state_time_query_side_normalized_loss_v1 as state_loss
from gx1.contracts.market_state_future_displacement_candidate_v1 import BoundFutureDisplacement
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K


def test_two_losses_keep_F_flat_side_gradient_zero_and_S_independent():
    s = torch.zeros((1, 5, 2), dtype=torch.float32, requires_grad=True)
    f = torch.zeros((1, 5, 2), dtype=torch.float32, requires_grad=True)
    state_truth = {'allowed': np.array([[5, 1, 4, 2, 7]], dtype=np.uint8),
                   'eligible': np.array([[True, True, True, True, False]]),
                   'mature': np.ones((1, 5), bool)}
    future_truth = {'allowed': np.array([[4, 2, 1, 7, 2]], dtype=np.uint8),
                    'eligible': np.array([[True, True, True, False, True]]),
                    'mature': np.ones((1, 5), bool)}
    q = torch.tensor([NATIVE_K], dtype=torch.int64)
    result = objective.combined_loss(state_raw=s, future_raw=f,
                                     state_truth=state_truth, future_truth=future_truth,
                                     state_query_k=q, future_weight=1.0)
    result['loss'].backward()
    assert result['state']['count'] == 4 and result['future']['count'] == 4
    assert s.grad[0, 0, 1] == 0 and s.grad[0, 1, 1] != 0
    assert f.grad[0, 0, 1] != 0 and f.grad[0, 2, 1] != 0
    assert torch.equal(f.grad[0, [1, 3, 4], 1], torch.zeros(3))
    assert result['skip_update'] is False


def test_native_F_score_uses_earlier_prefix_and_partitions_UTC_week():
    fit = np.array([state.TRAIN_START_NS + 2*state.M5_NS,
                    state.TRAIN_START_NS + 3*state.M5_NS], np.int64)
    score = np.array([state.FOLD_BOUNDARIES_NS[0] + state.M5_NS,
                      state.FOLD_BOUNDARIES_NS[0] + 2*state.M5_NS], np.int64)
    issues = np.r_[fit, score]
    parents = np.arange(4, dtype=np.int64)
    sign = np.array([[1]*5,[-1]*5,[1]*5,[-1]*5],dtype=np.int8)
    mag = np.ones((4,5),dtype=np.float64)
    valid = np.ones((4,5),dtype=bool)
    bound = BoundFutureDisplacement(parents,issues,valid,sign,mag,
                                    np.zeros((4,5),dtype=bool))
    axes = {'fit_parent_rows':parents[:2], 'fit_decision_ns':fit,
            'score_parent_rows':parents[2:], 'score_decision_ns':score,
            'fit_parent_rows_sha256':'fit','score_parent_rows_sha256':'score'}
    raw = np.zeros((2,5,2), dtype=np.float32)
    report=objective.score_native_F_fold(bound=bound,axes=axes,fold_id=1,
                                         score_raw=raw)
    assert len(report['by_utc_week']) == 1
    for tf in ('M5','M15','H1','H4','D1'):
        assert report['overall'][tf]['activity']['count']==2
        assert report['overall'][tf]['conditional_side']['count']==2
        assert report['overall'][tf]['conditional_side']['prior_minus_model_brier_sum']==0


def test_side_normalization_uses_only_known_side_rows_and_preserves_old_loss():
    allowed = torch.tensor([[1, 7, 7, 7, 7],
                            [5, 7, 7, 7, 7],
                            [2, 7, 7, 7, 7],
                            [4, 7, 7, 7, 7]], dtype=torch.uint8)
    eligible = torch.ones((4, 5), dtype=torch.bool)
    mature = torch.ones((4, 5), dtype=torch.bool)
    query = torch.tensor([NATIVE_K] * 4, dtype=torch.int64)
    raw = torch.zeros((4, 5, 2), dtype=torch.float32, requires_grad=True)
    old = legacy_loss.allowed_set_loss(raw, allowed, eligible, mature, query)
    balanced = state_loss.allowed_set_loss(
        raw, allowed, eligible, mature, query)
    assert torch.isclose(old['loss'], torch.tensor(1.5 * np.log(2) / 5, dtype=torch.float32))
    assert torch.isclose(balanced['loss'], torch.tensor(2 * np.log(2) / 5, dtype=torch.float32))
    assert balanced['contract_sha256'] == state_loss.contract()['contract_sha256']
    assert old['contract_sha256'] == legacy_loss.factorized_time_query_loss_contract()['contract_sha256']
    balanced['loss'].backward()
    assert raw.grad[1, 0, 1] == raw.grad[2, 0, 1] == 0
    assert torch.isclose(raw.grad[0, 0, 1], torch.tensor(0.05))
    assert torch.isclose(raw.grad[3, 0, 1], torch.tensor(-0.05))
    assert torch.isclose(raw.grad[0, 0, 0], torch.tensor(-0.025))
    assert raw.grad[0, 1:, :].abs().sum() == 0
