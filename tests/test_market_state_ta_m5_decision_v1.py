"""Focused five-TF decision connection and unknown-target gradient checks."""
from types import SimpleNamespace

import numpy as np
from pytest import MonkeyPatch
import torch

from gx1.contracts.market_state_ta_m5_decision_v1 import (
    IntegratedTAM5DecisionModel, choose_actions, contract, decision_loss, fit_statistics,
)
from tests.test_market_state_ta_integrated_model_v1 import _inputs, _model
from tests.test_market_state_ta_integrated_paired_objective_v1 import _case


def _stats():
    return {'mean_bps': np.zeros((6, 2, 3), dtype=np.float32),
            'std_bps': np.ones((6, 2, 3), dtype=np.float32),
            'quality_std_bps': np.ones((6, 2), dtype=np.float32),
            'paired_quality_std_bps': np.ones(6, dtype=np.float32),
            'sha256': 'synthetic'}


def test_decision_uses_five_tf_hidden_without_changing_existing_outputs():
    patch = MonkeyPatch()
    calls = []
    try:
        core = _model(patch, calls)
        args = _inputs()
        original = core(**args)
        model = IntegratedTAM5DecisionModel(core, _stats())
        new = model(**args)
        assert calls == [1, 1]
        assert all(torch.equal(new[key], value) for key, value in original.items())
        assert new['m5_cost_path_bps'].shape == (2, 6, 2, 3)
        assert torch.count_nonzero(new['m5_cost_path_bps']) == 0
        with torch.no_grad():
            model.decision_head.weight[0, 2 * core.dual.state_model._hidden_width] = 1
        changed = model(**args)['m5_cost_path_bps']
        assert not torch.equal(changed[..., 0], new['m5_cost_path_bps'][..., 0])
        assert calls == [1, 1, 1]
    finally:
        patch.undo()


def test_cost_and_path_masks_block_unknown_gradients_and_flat_tie():
    args = _case()
    prediction = torch.zeros((1, 6, 2, 3), dtype=torch.float32, requires_grad=True)
    args['model_output']['m5_cost_path_bps'] = prediction
    terminal = np.zeros((1, 6), dtype=np.bool_)
    path = terminal.copy()
    terminal[0, :2] = True
    path[0, 1] = True
    net = np.full((1, 6, 2), np.nan)
    mfe = net.copy()
    mae = net.copy()
    net[0, :2] = 2
    mfe[0, 1] = 1
    mae[0, 1] = 1
    targets = {'terminal_observed': terminal, 'path_observed': path,
               'net_bps': net, 'mfe_bps': mfe,
               'pre_mfe_mae_upper_bps': mae}
    model = SimpleNamespace(target_std_bps=torch.ones((6, 2, 3)),
        quality_std_bps=torch.ones((6, 2)),
        paired_quality_std_bps=torch.ones(6),
        research_objective={'contract_sha256': 'synthetic'}, aux_fold_id=2)
    result = decision_loss(**args, targets=targets, model=model)
    result['loss'].backward()
    grad = prediction.grad.detach().numpy()
    assert np.all(grad[0, 0, :, 1:] == 0)
    assert np.any(grad[0, 0, :, 0] != 0)
    assert np.any(grad[0, 1] != 0)
    assert np.all(grad[0, 2:] == 0)
    actions, horizons = choose_actions(np.zeros((1, 6, 2, 3), dtype=np.float32))
    assert actions.tolist() == [2] and horizons.tolist() == [-1]


def test_side_pair_and_adverse_path_gradients():
    args = _case()
    prediction = torch.zeros((1, 6, 2, 3), dtype=torch.float32, requires_grad=True)
    args['model_output']['m5_cost_path_bps'] = prediction
    observed = np.zeros((1, 6), dtype=np.bool_)
    observed[0, 0] = True
    net = np.full((1, 6, 2), np.nan)
    mfe = net.copy()
    mae = net.copy()
    net[0, 0] = [6, -6]
    mfe[0, 0] = [1, 1]
    mae[0, 0] = [2, 4]
    targets = {'terminal_observed': observed, 'path_observed': observed.copy(),
               'net_bps': net, 'mfe_bps': mfe,
               'pre_mfe_mae_upper_bps': mae}
    model = SimpleNamespace(target_std_bps=torch.ones((6, 2, 3)),
        quality_std_bps=torch.ones((6, 2)),
        paired_quality_std_bps=torch.full((6,), 4.0),
        research_objective={'contract_sha256': 'synthetic'}, aux_fold_id=2)
    losses = decision_loss(**args, targets=targets, model=model)
    assert torch.isclose(losses['side_pair_loss'], torch.tensor(3.0))
    side_grad = torch.autograd.grad(losses['side_pair_loss'], prediction,
                                    retain_graph=True)[0]
    mae_grad = torch.autograd.grad(losses['mae_under_loss'], prediction)[0]
    assert side_grad[0, 0, 0, 0] < 0 and side_grad[0, 0, 1, 0] > 0
    assert torch.all(mae_grad[0, 0, :, 2] < 0)
    assert torch.count_nonzero(side_grad[:, 1:]) == 0
    assert torch.count_nonzero(mae_grad[:, 1:]) == 0


def test_cold_aux_fold_contract_is_distinct_from_consumed_fold2():
    from gx1.contracts.market_state_ta_integrated_paired_objective_v1 import (
        contract as paired_contract,
    )
    warm = contract('core', 'stats')
    assert warm == contract('core', 'stats', aux_fold_id=2)
    cold = contract('core', 'stats', aux_fold_id=1)
    assert cold['contract_sha256'] != warm['contract_sha256']
    assert cold['cold_aux_fold_id'] == 1
    assert cold['cold_aux_contract_sha256'] == paired_contract(fold_id=1)['contract_sha256']
    assert 'cold_aux_fold_id' not in warm

def test_paired_quality_scale_uses_joint_fit_paths_and_changes_statistics_hash():
    n = 4
    x = np.arange(n, dtype=np.float64)
    net = np.empty((n, 6, 2), dtype=np.float64)
    mfe = np.empty_like(net)
    mae = np.empty_like(net)
    for j in range(6):
        net[:, j, 0], net[:, j, 1] = 2 * x + j, -x + j
        mfe[:, j, 0], mfe[:, j, 1] = x + 1, 2 * x + 2
        mae[:, j, 0], mae[:, j, 1] = 0.25 * x + 1, 0.5 * x + 2
    observed = np.ones((n, 6), dtype=np.bool_)
    targets = {'terminal_observed': observed, 'path_observed': observed.copy(),
               'horizon_minutes': np.array((1, 5, 15, 60, 240, 480)),
               'net_bps': net, 'mfe_bps': mfe,
               'pre_mfe_mae_upper_bps': mae}
    fitted = fit_statistics(targets)
    np.testing.assert_allclose(fitted['paired_quality_std_bps'],
                               3.25 * x.std(), rtol=1e-6)
    assert not np.allclose(fitted['paired_quality_std_bps'],
                           np.sqrt((fitted['quality_std_bps'] ** 2).sum(axis=1)))
    flipped = {key: value.copy() for key, value in targets.items()}
    for key in ('net_bps', 'mfe_bps', 'pre_mfe_mae_upper_bps'):
        flipped[key][:, :, 1] = targets[key][::-1, :, 1]
    refitted = fit_statistics(flipped)
    np.testing.assert_array_equal(refitted['quality_std_bps'],
                                  fitted['quality_std_bps'])
    assert refitted['sha256'] != fitted['sha256']
    assert contract('core', fitted['sha256'])['schema_version'] == (
        'gx1_ta_m5_cost_path_decision_v2')



def test_paired_scale_is_bound_to_model_and_contract_without_forward():
    patch = MonkeyPatch()
    try:
        calls = []
        core = _model(patch, calls)
        model = IntegratedTAM5DecisionModel(core, _stats())
        assert calls == []
        torch.testing.assert_close(model.paired_quality_std_bps, torch.ones(6))
        assert 'paired_quality_std_bps' in model.state_dict()
        assert model.research_objective['schema_version'] == (
            'gx1_ta_m5_cost_path_decision_v2')
    finally:
        patch.undo()
