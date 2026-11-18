"""Focused guards for frozen Fold1 FIT scoring; no model or data replay."""
from pathlib import Path

import numpy as np
import pytest

from gx1.audit import score_ta_ema_final_fit_v1 as replay


def test_h480_ranking_uses_only_complete_paths_and_fixed_side_order(monkeypatch):
    monkeypatch.setattr(replay, 'FIT_ROWS', 4)
    predicted = np.zeros((4, 6, 2, 3), dtype=np.float32)
    predicted[:, 5, 0, 0] = [4, 0, 2, 999]
    predicted[:, 5, 1, 0] = [0, 4, 2, -999]
    targets = {'path_observed': np.zeros((4, 6), dtype=np.bool_),
               'net_bps': np.zeros((4, 6, 2), dtype=np.float32),
               'pre_mfe_mae_upper_bps': np.zeros((4, 6, 2), dtype=np.float32)}
    targets['path_observed'][:3, 5] = True
    targets['net_bps'][:3, 5, :] = [[4, 0], [0, 4], [2, 2]]
    result = replay.h480_side_readout(predicted, targets)
    assert result['complete_path_rows'] == 3
    assert result['predicted_long_fraction'] == pytest.approx(2 / 3)
    assert result['side_accuracy'] == 1
    assert result['side_ranking_auc_for_actual_long_better'] == 1
    assert result['side_difference_pearson'] == pytest.approx(1)
    assert result['predicted_long_minus_short_quality_bps_mean'] == 0
    assert result['actual_long_minus_short_quality_bps_mean'] == 0
    assert result['side_only_net_bps_mean'] == pytest.approx(10 / 3)
    assert result['always_long_net_bps_mean'] == 2


def test_inspect_requires_closed_training_and_exact_one_use_gate(monkeypatch, tmp_path):
    plan_path = tmp_path / 'PLAN.json'
    plan_path.write_text('{}')
    output = tmp_path / 'OUTPUT'
    plan = {'source_base_commit': 'fixed', 'consumed_ema': {
        'plan': {'path': 'p', 'sha256': 'p'},
        'claim': {'path': 'cl', 'sha256': 'cl'},
        'result': {'path': 'r', 'sha256': 'r'},
        'terminal': {'path': 't', 'sha256': 't'},
        'checkpoint': {'path': 'c', 'sha256': 'c'},
        'final_capture': {'path': 'f', 'sha256': 'f'}}}
    plan_ref = {'path': str(plan_path), 'sha256': 'plan'}
    policy_ref = {'path': str(replay.REPO / 'NEXT_RUN_POLICY.json'), 'sha256': 'policy'}
    gate = {'status': 'PREPARED_NOT_ADMITTED', 'plan': plan_ref,
            'output_root': str(output), 'fit_rows': replay.FIT_ROWS,
            'model_forwards': replay.FORWARDS, 'max_optimizer_updates': 0,
            'training_authorized': False, 'test_authorized': False,
            'online_authorized': False, 'broker_access_allowed': False,
            'test_data_used': False, 'relaunch_allowed': False,
            'execution_authorized': False, 'remaining_invocations': 0}
    policy = {'training_enabled': False, 'native_training_enabled': False,
              'full_epoch_training_allowed': False, 'native_launch_authorized': False,
              'full_val_allowed': False,
              'exit_training_and_evaluation_this_wave': False,
              replay.STAGE: gate,
              replay.prior_fit.EMA_STAGE: {
                  **plan['consumed_ema'],
                  'status': 'REVIEWED_CONSUMED_EMA_SIDE_COLLAPSE_EDGE_UNPROVEN',
                  'remaining_invocations': 0, 'completed_invocations': 1,
                  'execution_authorized': False}}
    monkeypatch.setattr(replay, 'PLAN', plan_path)
    monkeypatch.setattr(replay, 'ROOT', output)
    monkeypatch.setattr(replay, 'build_plan', lambda **_: plan)
    monkeypatch.setattr(replay, 'source_clean', lambda _: True)
    monkeypatch.setattr(replay, 'ref', lambda path: plan_ref if Path(path) == plan_path else policy_ref)
    monkeypatch.setattr(replay, 'read', lambda binding: plan if binding == plan_ref else policy)
    assert replay.inspect('plan', admitted=False) == (plan, policy_ref)
    assert not output.exists()
    with pytest.raises(RuntimeError, match='ONE_USE_ADMISSION'):
        replay.inspect('plan', admitted=True)
    gate.update(status='ADMITTED_ONE_USE', execution_authorized=True,
                remaining_invocations=1, max_invocations=1,
                claimed_invocations=0, completed_invocations=0)
    assert replay.inspect('plan', admitted=True) == (plan, policy_ref)
    policy['training_enabled'] = True
    with pytest.raises(RuntimeError, match='CLOSED_POLICY'):
        replay.inspect('plan', admitted=True)
    policy['training_enabled'] = False
    gate['claimed_invocations'] = 1
    with pytest.raises(RuntimeError, match='ONE_USE_ADMISSION'):
        replay.inspect('plan', admitted=True)
    assert not output.exists()
