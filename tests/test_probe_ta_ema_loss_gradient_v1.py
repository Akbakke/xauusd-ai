"""Focused zero-update EMA fused-gradient probe math and policy gates."""
from pathlib import Path

import numpy as np
import pytest
import torch

from gx1.audit import probe_ta_ema_loss_gradient_v1 as probe


def test_predeclared_batches_are_distinct_full_b16():
    assert probe.FIT_ROWS == 29574
    assert probe.FULL_BATCHES == 1848
    assert len(probe.BATCH_INDICES) == len(set(probe.BATCH_INDICES)) == 20
    assert probe.BATCH_INDICES[0] == 0 and probe.BATCH_INDICES[-1] == 1847
    assert all(0 <= i < probe.FULL_BATCHES for i in probe.BATCH_INDICES)


def test_isolated_native_F_M5_side_matches_exact_q_over_N_gradient():
    raw = torch.zeros((4, 5, 2), dtype=torch.float32, requires_grad=True)
    truth = {'allowed': np.array([[4], [1], [2], [7]], dtype=np.uint8),
             'eligible': np.ones((4, 1), dtype=np.bool_),
             'mature': np.ones((4, 1), dtype=np.bool_)}
    loss, count = probe.isolated_native_f_m5_side(raw, truth, torch=torch)
    grad = torch.autograd.grad(loss, raw)[0]
    coefficient = .10 * 1844 / 11909 / 5
    assert count == 2
    assert grad[:, 0, 1].tolist() == pytest.approx(
        [-coefficient / 2, coefficient / 2, 0, 0])
    mask = torch.ones_like(grad, dtype=torch.bool)
    mask[:, 0, 1] = False
    assert torch.count_nonzero(grad[mask]) == 0


def test_summary_requires_pooled_and_majority_negative(monkeypatch):
    monkeypatch.setattr(probe, 'BATCH_INDICES', (0, 1, 2))
    def record(dot):
        geometry = {'dot': dot, 'a_norm': 2., 'b_norm': 1., 'cosine': dot / 2}
        return {'native_F_M5_known_side': 2,
                **{key: geometry for key in
                   ('decision_vs_F_side', 'auxiliary_vs_F_side',
                    'total_vs_F_side', 'decision_vs_auxiliary')}}
    measured = probe.summarize_records([record(-1), record(-2), record(1)])
    assert measured['interpretation'] == 'MEASURED_OPPOSITION_AT_FUSED_BOUNDARY'
    assert measured['pooled']['decision_vs_F_side']['negative_supported_batches'] == 2
    assert measured['pooled']['decision_vs_F_side']['a_to_b_norm_ratio'] == 2
    assert probe.summarize_records([record(-1), record(2), record(2)])['interpretation'] == \
        'OPPOSITION_NOT_ESTABLISHED'
    with pytest.raises(RuntimeError, match='SUMMARY_BATCH_COUNT'):
        probe.summarize_records([record(-1)])


def test_inspect_requires_consumed_score_closed_training_and_exact_gate(monkeypatch, tmp_path):
    plan_path = tmp_path / 'PLAN.json'
    plan_path.write_text('{}')
    root = tmp_path / 'RESULT'
    plan_ref = {'path': str(plan_path), 'sha256': 'plan'}
    policy_ref = {'path': str(probe.REPO / 'NEXT_RUN_POLICY.json'), 'sha256': 'policy'}
    old = {'result': {'path': 'old-result', 'sha256': 'r'},
           'terminal': {'path': 'old-terminal', 'sha256': 't'},
           'capture': {'path': 'old-capture', 'sha256': 'c'}}
    plan = {'source_base_commit': 'fixed', 'frozen_fit_score': old}
    gate = {'status': 'PREPARED_NOT_ADMITTED', 'plan': plan_ref,
            'output_root': str(root), 'fit_rows': probe.FIT_ROWS,
            'model_forwards': len(probe.BATCH_INDICES) + 1,
            'autograd_calls': len(probe.BATCH_INDICES) * 4,
            'max_optimizer_updates': 0, 'training_authorized': False,
            'test_authorized': False, 'online_authorized': False,
            'broker_access_allowed': False, 'test_data_used': False,
            'relaunch_allowed': False, 'execution_authorized': False,
            'remaining_invocations': 0}
    policy = {'training_enabled': False, 'native_training_enabled': False,
              'full_epoch_training_allowed': False, 'native_launch_authorized': False,
              'full_val_allowed': False, 'exit_training_and_evaluation_this_wave': False,
              probe.STAGE: gate,
              probe.replay.STAGE: {
                  'status': 'REVIEWED_CONSUMED_FIT_SF_WEAK_EDGE_UNPROVEN',
                  'result': old['result'], 'terminal': old['terminal'],
                  'fit_capture': old['capture'], 'remaining_invocations': 0,
                  'execution_authorized': False}}
    monkeypatch.setattr(probe, 'PLAN', plan_path)
    monkeypatch.setattr(probe, 'ROOT', root)
    monkeypatch.setattr(probe, 'build_plan', lambda **_: plan)
    monkeypatch.setattr(probe, 'source_clean', lambda _: True)
    monkeypatch.setattr(probe, 'ref', lambda path: plan_ref if Path(path) == plan_path else policy_ref)
    monkeypatch.setattr(probe, 'read', lambda binding: plan if binding == plan_ref else policy)
    assert probe.inspect('plan', admitted=False) == (plan, policy_ref)
    with pytest.raises(RuntimeError, match='ONE_USE_ADMISSION'):
        probe.inspect('plan', admitted=True)
    gate.update(status='ADMITTED_ONE_USE', execution_authorized=True,
                remaining_invocations=1, max_invocations=1,
                claimed_invocations=0, completed_invocations=0)
    assert probe.inspect('plan', admitted=True) == (plan, policy_ref)
    policy['training_enabled'] = True
    with pytest.raises(RuntimeError, match='CLOSED_POLICY'):
        probe.inspect('plan', admitted=True)
    assert not root.exists()
