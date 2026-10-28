import pytest
import torch

from gx1.audit.direction_gc_context_intervention_v1 import drop_gc_context


def test_drop_gc_context_preserves_aliases_and_unchanged_input():
    seq = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    snap = seq[:, -1, :].clone()
    ctx = torch.tensor([[snap[0, 1], 17, snap[0, 3]], [snap[1, 1], 19, snap[1, 3]]])
    untouched = torch.tensor([[8.], [9.]])
    features = {'seq_x': seq, 'snap_x': snap, 'ctx_cont': ctx, 'seq_m5': untouched}
    before = {name: value.clone() for name, value in features.items()}
    layout = {'signal_indices': (1, 3), 'ctx_cont_indices': (0, 2)}
    result = drop_gc_context(features, layout)
    assert all(torch.equal(features[name], value) for name, value in before.items())
    assert torch.count_nonzero(result['seq_x'][..., (1, 3)]) == 0
    assert torch.count_nonzero(result['snap_x'][:, (1, 3)]) == 0
    assert torch.count_nonzero(result['ctx_cont'][:, (0, 2)]) == 0
    assert torch.equal(result['seq_x'][..., (0, 2)], seq[..., (0, 2)])
    assert torch.equal(result['snap_x'][:, (0, 2)], snap[:, (0, 2)])
    assert torch.equal(result['ctx_cont'][:, 1], ctx[:, 1])
    assert result['seq_m5'] is untouched
    assert torch.equal(result['seq_x'][:, -1, :], result['snap_x'])


def test_drop_gc_context_rejects_broken_alias_or_invalid_layout():
    seq = torch.zeros((1, 2, 3)); snap = seq[:, -1, :].clone(); ctx = torch.ones((1, 2))
    with pytest.raises(RuntimeError, match='ORIGINAL_ALIAS_PARITY'):
        drop_gc_context({'seq_x': seq, 'snap_x': snap, 'ctx_cont': ctx},
            {'signal_indices': (1,), 'ctx_cont_indices': (0,)})
    with pytest.raises(RuntimeError, match='LAYOUT'):
        drop_gc_context({'seq_x': seq, 'snap_x': snap, 'ctx_cont': ctx},
            {'signal_indices': (1, 1), 'ctx_cont_indices': (0, 1)})


def test_real_gc_owner_layout_rejects_inactive_or_cross_owner_fields(monkeypatch):
    from gx1.audit.direction_gc_context_intervention_v1 import compile_gc_context_layout
    from gx1.features import gc_native_context_v1 as gc
    from gx1.features.entry_specialist_feature_groups_v1 import MODEL_NATIVE_CTX_CONT_FIELDS
    names = tuple(MODEL_NATIVE_CTX_CONT_FIELDS)
    signal = tuple('ctx_cont.' + name for name in names)
    profile = {'physical_field_names': {'signal': signal, 'ctx_cont': names, 'ctx_cat': (), 'mtf': ()},
        'gc_context_fields': tuple(name for name in names if name.startswith('gc_')),
        'signal_indices': tuple(range(len(names))), 'ctx_cont_indices': tuple(range(len(names)))}
    layout = compile_gc_context_layout(profile)
    assert (layout['gc_fields'], layout['value_mask_pairs']) == (2410, 1205)
    inactive = dict(profile, signal_indices=tuple(i for i in profile['signal_indices'] if i != layout['signal_indices'][0]))
    with pytest.raises(RuntimeError, match='ACTIVE_GC_COORDINATES'):
        compile_gc_context_layout(inactive)
    original = gc.masking_pairs
    monkeypatch.setattr(gc, 'masking_pairs', lambda fields: tuple(original(fields)) + ((0, layout['signal_indices'][0]),))
    with pytest.raises(RuntimeError, match='CROSS_OWNER_MASK'):
        compile_gc_context_layout(profile)


def test_context_dispatch_requires_cuda_and_sets_new_mode(monkeypatch, tmp_path):
    from gx1.audit import direction_gc_mtf_followup_v1 as followup
    from gx1.scripts import validate_entry_model_native_technical_checkpoint_v1 as validator
    calls = []
    monkeypatch.setattr(followup, 'run', lambda **kwargs: calls.append(kwargs) or {'status': 'SYNTHETIC'})
    args = ['--direction-gc-context-followup-plan', str(tmp_path / 'plan.json'),
        '--direction-gc-context-followup-plan-sha256', 'a' * 64, '--device', 'cuda', '--out-dir', str(tmp_path)]
    assert validator.main(args) == 0 and len(calls) == 1 and calls[0]['mode'] == 'context_and_history'
    with pytest.raises(SystemExit):
        validator.main(args + ['--device', 'cuda'])
    with pytest.raises(SystemExit):
        validator.main(args[:4] + ['--device', 'cpu', '--out-dir', str(tmp_path)])
