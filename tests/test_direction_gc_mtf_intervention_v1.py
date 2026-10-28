import numpy as np
import pytest
import torch

from gx1.audit import direction_gc_mtf_intervention_v1 as audit


def test_original_batch_selection_is_stable_and_preserves_final_partial_batch():
    chunks = audit.selected_batches(329)
    assert [chunk.tolist() for chunk in chunks] == [list(range(16)), list(range(160, 176)), list(range(320, 329))]
    with pytest.raises(RuntimeError, match='ROWS'):
        audit.selected_batches(0)


def test_gc_projection_hook_changes_only_requested_route_and_can_be_removed():
    modules = torch.nn.ModuleDict({'m5__smc': torch.nn.Linear(2, 2), 'h1__smc': torch.nn.Linear(2, 2)})
    sample = torch.tensor([[1., 2.]])
    before = {key: modules[key](sample).detach().clone() for key in modules}
    handles, calls = audit.install_gc_projection_ablation(modules, tuple(modules.keys()))
    try:
        for key in modules:
            assert torch.count_nonzero(modules[key](sample)) == 0
        assert calls == {'m5__smc': 1, 'h1__smc': 1}
    finally:
        for handle in handles:
            handle.remove()
    for key in modules:
        assert torch.equal(modules[key](sample), before[key])
    with pytest.raises(RuntimeError, match='GC_MODULE_KEYS'):
        audit.install_gc_projection_ablation(modules, ('h1__smc', 'm5__smc'))


def test_paired_scores_compare_same_labels_across_all_minutes(monkeypatch):
    monkeypatch.setattr(audit, 'REPLICATES', 50)
    n = 16
    decisions = np.arange(n, dtype=np.int64) * 7 * audit.DAY_NS + audit.DAY_NS
    terminal = decisions[:, None] + np.arange(1, 61, dtype=np.int64)[None, :] * audit.MINUTE_NS
    labels = np.where(np.arange(n) % 2 == 0, -1, 1).astype(np.int8)
    signs = np.tile(labels[:, None], (1, 60))
    observed = np.ones((n, 60), dtype=np.bool_)
    original = signs.astype(np.float32); ablated = -original
    result = audit.paired_metrics(decisions, observed, signs, terminal, original, ablated, cutoff=200 * audit.DAY_NS)
    assert len(result['per_horizon']) == 60
    assert result['per_horizon'][4]['arms']['original']['auc'] == 1.0
    assert result['per_horizon'][4]['arms']['ablated']['auc'] == 0.0
    assert len(result['paired_intervals']) == 4


def test_exact_saved_replay_rejects_float_and_shape_drift():
    expected = np.array([[0., -0., 1.]], dtype=np.float32)
    audit.require_exact_replay(expected.copy(), expected)
    changed = expected.copy(); changed[0, 1] = 0.
    with pytest.raises(RuntimeError, match='EXACT_OOF_REPLAY'):
        audit.require_exact_replay(changed, expected)
    with pytest.raises(RuntimeError, match='EXACT_OOF_REPLAY'):
        audit.require_exact_replay(expected.astype(np.float64), expected)


def test_guarded_validator_dispatch_requires_cuda_and_rejects_duplicate_flags(monkeypatch, tmp_path):
    from gx1.scripts import validate_entry_model_native_technical_checkpoint_v1 as validator
    calls = []
    monkeypatch.setattr(audit, 'run', lambda **kwargs: calls.append(kwargs) or {'status': 'SYNTHETIC'})
    args = ['--direction-gc-mtf-intervention-plan', str(tmp_path / 'plan.json'),
        '--direction-gc-mtf-intervention-plan-sha256', 'a' * 64, '--device', 'cuda', '--out-dir', str(tmp_path)]
    assert validator.main(args) == 0 and len(calls) == 1
    assert calls[0]['device'] == 'cuda'
    with pytest.raises(SystemExit):
        validator.main(args + ['--device', 'cuda'])
    with pytest.raises(SystemExit):
        validator.main(args[:4] + ['--device', 'cpu', '--out-dir', str(tmp_path)])


def test_partial_hook_installation_is_rolled_back(monkeypatch):
    modules = torch.nn.ModuleDict({'a': torch.nn.Linear(2, 2), 'b': torch.nn.Linear(2, 2)})
    sample = torch.tensor([[1., 2.]])
    before = modules['a'](sample).detach().clone()
    def fail(*args, **kwargs):
        raise RuntimeError('registration failed')
    monkeypatch.setattr(modules['b'], 'register_forward_hook', fail)
    with pytest.raises(RuntimeError, match='registration failed'):
        audit.install_gc_projection_ablation(modules, ('a', 'b'))
    assert torch.equal(modules['a'](sample), before) and not modules['a']._forward_hooks
