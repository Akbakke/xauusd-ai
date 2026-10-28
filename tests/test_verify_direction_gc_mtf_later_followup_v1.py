import copy

import numpy as np
import pytest

from gx1.audit import direction_gc_mtf_intervention_v1 as original
from gx1.scripts import verify_direction_gc_mtf_later_followup_v1 as verify


def test_independent_block_auc_handles_duplicate_blocks_and_ties():
    labels = np.array([0, 1, 0, 1], dtype=np.int8)
    scores = np.array([0., 0., 1., 2.], dtype=np.float64)
    ids = np.array([0, 0, 1, 1], dtype=np.int64)
    weights = np.array([[1, 1], [2, 0], [0, 2]], dtype=np.int64)
    got = verify.independent_auc_draws(labels, scores, ids, weights)
    assert got.tolist() == [0.625, 0.5, 1.0]


def test_independent_all_horizons_and_bootstrap_detects_metric_or_target_drift(monkeypatch):
    monkeypatch.setattr(original, 'REPLICATES', 25)
    n = 1600; parent = np.arange(n, dtype=np.int64)
    decision = ((parent // 160) * 7 * verify.DAY_NS + (parent % 160) * 300_000_000_000).astype(np.int64)
    terminal = decision[:, None] + np.arange(1, 61, dtype=np.int64)[None, :] * 60_000_000_000
    sign = np.tile(np.where(parent % 2 == 0, -1, 1).astype(np.int8)[:, None], (1, 60))
    observed = np.ones((n, 60), dtype=np.bool_)
    corrected = {'parent_rows': parent, 'decision_ns': decision, 'terminal_time_ns': terminal, 'observed': observed, 'sign': sign}
    indices = np.concatenate([np.arange(start, min(start + 16, n)) for start in range(0, n, 160)])
    base = np.tile(np.where(indices % 2 == 0, -0.5, 0.5).astype(np.float32)[:, None], (1, 60))
    alternative = -base
    pair = {key: value[indices].copy() for key, value in corrected.items()}
    pair.update(original_logit_up=base, ablated_logit_up=alternative)
    cutoff = int(decision[-1] + verify.DAY_NS)
    metrics = original.paired_metrics(pair['decision_ns'], pair['observed'], pair['sign'], pair['terminal_time_ns'], base, alternative, cutoff=cutoff)
    result = {'later_fold': {'oof_start_row': 0, 'oof_end_row': n, 'oof_cutoff_ns': cutoff},
        'selection': {'policy': 'every tenth contiguous 16-row later batch', 'batch_size': 16,
            'batch_stride': 10, 'selected_parents': len(indices), 'selected_batches': 10}, 'metrics': metrics}
    proof = verify.verify_metrics(pair, corrected, result, replicates=25)
    assert proof['horizons'] == 60 and proof['arms'] == 2 and proof['paired_interval_families'] == 4
    wrong = copy.deepcopy(result); wrong['metrics']['per_horizon'][4]['arms']['ablated']['auc'] += .01
    with pytest.raises(RuntimeError, match='POINT_REPLAY:5:ablated:auc'):
        verify.verify_metrics(pair, corrected, wrong, replicates=25)
    wrong_pair = dict(pair, sign=pair['sign'].copy()); wrong_pair['sign'][0, 0] *= -1
    with pytest.raises(RuntimeError, match='TARGET_JOIN:sign'):
        verify.verify_metrics(wrong_pair, corrected, result, replicates=25)
