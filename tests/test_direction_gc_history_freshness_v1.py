import numpy as np
import pytest
import torch

from gx1.audit.direction_gc_history_freshness_v1 import TIMEFRAMES, M5_NS, hide_fresh_gc_history


def inputs():
    decision = np.array([20, 21], dtype=np.int64) * M5_NS
    witness = {}; features = {'seq_m5': torch.tensor([37.])}
    for tf in TIMEFRAMES:
        features['gc_seq_' + tf.lower()] = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) + 1
        witness[tf] = [{'available_ns': [int(t-2*M5_NS), int(t-M5_NS), int(t)]} for t in decision]
    return decision, witness, features


def test_hides_only_new_gc_rows_and_preserves_all_other_inputs():
    decision, witness, features = inputs(); before = {key: value.clone() for key, value in features.items()}
    result, counts = hide_fresh_gc_history(features, witness=witness, decision_ns=decision,
        prior_available=np.array([True, False]))
    assert all(torch.equal(features[k], v) for k, v in before.items())
    assert result['seq_m5'] is features['seq_m5']
    for tf in TIMEFRAMES:
        key = 'gc_seq_' + tf.lower()
        assert torch.equal(result[key][0, :2], features[key][0, :2])
        assert torch.count_nonzero(result[key][0, 2]) == 0
        assert torch.equal(result[key][1], features[key][1])
        assert counts[tf] == {'hidden_rows': 1, 'nonzero_rows_hidden': 1}


def test_rejects_future_clock_and_more_than_one_new_bucket():
    decision, witness, features = inputs(); active = np.array([True, True])
    witness['M5'][0]['available_ns'][-1] += M5_NS
    with pytest.raises(RuntimeError, match='FRESHNESS_CLOCK:M5'):
        hide_fresh_gc_history(features, witness=witness, decision_ns=decision, prior_available=active)
    decision, witness, features = inputs()
    witness['M5'][0]['available_ns'][-2] = int(decision[0] - M5_NS // 2)
    with pytest.raises(RuntimeError, match='FRESHNESS_MULTIPLE_BUCKETS:M5'):
        hide_fresh_gc_history(features, witness=witness, decision_ns=decision, prior_available=active)


def test_full_stale_dispatch_requires_cuda_and_exact_flags(monkeypatch, tmp_path):
    from gx1.audit import direction_gc_mtf_followup_v1 as followup
    from gx1.scripts import validate_entry_model_native_technical_checkpoint_v1 as validator
    calls = []
    monkeypatch.setattr(followup, 'run', lambda **kwargs: calls.append(kwargs) or {'status': 'SYNTHETIC'})
    args = ['--direction-gc-full-stale-plan', str(tmp_path / 'plan.json'),
        '--direction-gc-full-stale-plan-sha256', 'a' * 64, '--device', 'cuda', '--out-dir', str(tmp_path)]
    assert validator.main(args) == 0 and len(calls) == 1 and calls[0]['mode'] == 'full_stale_gc'
    with pytest.raises(SystemExit): validator.main(args + ['--device', 'cuda'])
    with pytest.raises(SystemExit): validator.main(args[:4] + ['--device', 'cpu', '--out-dir', str(tmp_path)])
