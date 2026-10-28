import numpy as np
import pytest
import torch

from gx1.audit import direction_unseen_gc_projection_v1 as audit


def test_input_only_selection_keeps_original_whole_batches():
    active = np.zeros(45, dtype=np.bool_); active[[1, 17, 44]] = True
    batches = audit.select_batches(active)
    assert [len(b) for b in batches] == [16, 16, 13]
    assert np.array_equal(np.concatenate(batches), np.arange(45))
    active[:] = False; active[[1, 44]] = True
    assert [b[0] for b in audit.select_batches(active)] == [0, 32]
    with pytest.raises(RuntimeError, match='SELECTION_INPUT'):
        audit.select_batches(active.astype(np.int8))


def test_hook_changes_only_named_projection_columns_and_is_removable():
    module = torch.nn.Linear(8, 2, bias=False)
    with torch.no_grad(): module.weight.copy_(torch.arange(16).reshape(2, 8))
    inputs = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    positions = [0, 2, 4, 5, 7]; expected = inputs[:, positions].numpy().copy()
    baseline = module(inputs).detach().clone(); before = inputs.clone()
    handle, calls = audit.install_projection_ablation(module, positions, expected)
    changed = module(inputs); handle.remove()
    manual = before.clone(); manual[:, positions] = 0
    assert torch.equal(changed, module(manual)) and torch.equal(inputs, before)
    assert torch.equal(module(inputs), baseline) and calls['count'] == 1
    with pytest.raises(RuntimeError, match='HOOK_SPEC'):
        audit.install_projection_ablation(module, [0, 0, 4, 5, 7], expected)
    handle, _ = audit.install_projection_ablation(module, positions, expected + 1)
    try:
        with pytest.raises(RuntimeError, match='MODEL_NORMALIZATION_PARITY'): module(inputs)
    finally: handle.remove()


def test_summarize_requires_unchanged_inactive_rows_and_counts_flips():
    original = np.zeros((2, 60), dtype=np.float32); original[:, 4] = [1, -1]
    changed = original.copy(); changed[0, 4] = -1
    active = np.array([True, False])
    result = audit.summarize(original, changed, active)
    assert result['active_inputs']['hard_direction_flips_5m'] == 1
    assert result['active_inputs']['mean_absolute_logit_change_5m'] == 2
    changed[1, 0] = 1
    with pytest.raises(RuntimeError, match='INACTIVE_ROWS_IDENTICAL'):
        audit.summarize(original, changed, active)


def test_guarded_dispatch_requires_cuda_and_single_flags(monkeypatch, tmp_path):
    from gx1.scripts import validate_entry_model_native_technical_checkpoint_v1 as validator
    calls = []
    monkeypatch.setattr(audit, 'run', lambda **kwargs: calls.append(kwargs) or {'status': 'SYNTHETIC'})
    args = ['--direction-unseen-gc-projection-plan', str(tmp_path / 'plan.json'),
        '--direction-unseen-gc-projection-plan-sha256', 'a' * 64, '--device', 'cuda', '--out-dir', str(tmp_path)]
    assert validator.main(args) == 0 and len(calls) == 1 and calls[0]['device'] == 'cuda'
    with pytest.raises(SystemExit): validator.main(args + ['--device', 'cuda'])
    with pytest.raises(SystemExit): validator.main(args[:4] + ['--device', 'cpu', '--out-dir', str(tmp_path)])
