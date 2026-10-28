import pytest
import torch

from gx1.audit.direction_gc_stale_context_v1 import replace_current_gc_context


def test_replace_current_gc_uses_prior_values_and_masks_only_at_last_step():
    seq = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    snap = seq[:, -1].clone(); ctx = torch.stack((snap[:, 1], torch.tensor([20., 21.]), snap[:, 3]), dim=1)
    prior_snap = snap.clone(); prior_snap[:, 1] = torch.tensor([111., 222.]); prior_snap[:, 3] = 0
    prior_ctx = ctx.clone(); prior_ctx[:, 0] = prior_snap[:, 1]; prior_ctx[:, 2] = 0
    features = {'seq_x': seq, 'snap_x': snap, 'ctx_cont': ctx, 'gc_seq_m5': torch.tensor([5.])}
    original = {k: v.clone() for k, v in features.items()}
    got = replace_current_gc_context(features, prior_snap, prior_ctx, {'signal_indices': (1, 3), 'ctx_cont_indices': (0, 2)})
    assert all(torch.equal(features[k], v) for k, v in original.items())
    assert torch.equal(got['seq_x'][:, :-1], seq[:, :-1])
    assert torch.equal(got['seq_x'][:, -1], got['snap_x'])
    assert torch.equal(got['snap_x'][:, (1, 3)], prior_snap[:, (1, 3)])
    assert torch.equal(got['ctx_cont'][:, (0, 2)], prior_ctx[:, (0, 2)])
    assert torch.equal(got['snap_x'][:, (0, 2)], snap[:, (0, 2)])
    assert torch.equal(got['ctx_cont'][:, 1], ctx[:, 1])
    assert got['gc_seq_m5'] is features['gc_seq_m5']


def test_replace_rejects_broken_prior_alias_or_duplicate_layout():
    seq = torch.zeros(1, 2, 3); snap = seq[:, -1].clone(); ctx = torch.zeros(1, 2)
    prior = snap.clone(); prior[:, 1] = 1
    with pytest.raises(RuntimeError, match='STALE_ORIGINAL_ALIAS_PARITY'):
        replace_current_gc_context({'seq_x': seq, 'snap_x': snap, 'ctx_cont': ctx}, prior, ctx,
            {'signal_indices': (1,), 'ctx_cont_indices': (0,)})
    with pytest.raises(RuntimeError, match='STALE_LAYOUT'):
        replace_current_gc_context({'seq_x': seq, 'snap_x': snap, 'ctx_cont': ctx}, snap, ctx,
            {'signal_indices': (1, 1), 'ctx_cont_indices': (0, 1)})


def test_stale_context_dispatch_requires_cuda_and_exact_flags(monkeypatch, tmp_path):
    from gx1.audit import direction_gc_mtf_followup_v1 as followup
    from gx1.scripts import validate_entry_model_native_technical_checkpoint_v1 as validator
    calls = []
    monkeypatch.setattr(followup, 'run', lambda **kwargs: calls.append(kwargs) or {'status': 'SYNTHETIC'})
    args = ['--direction-gc-stale-context-plan', str(tmp_path / 'plan.json'),
        '--direction-gc-stale-context-plan-sha256', 'a' * 64, '--device', 'cuda', '--out-dir', str(tmp_path)]
    assert validator.main(args) == 0 and len(calls) == 1 and calls[0]['mode'] == 'stale_context'
    with pytest.raises(SystemExit): validator.main(args + ['--device', 'cuda'])
    with pytest.raises(SystemExit): validator.main(args[:4] + ['--device', 'cpu', '--out-dir', str(tmp_path)])
