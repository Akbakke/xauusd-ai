"""Synthetic scope and failure checks for the one-use cold raw gather audit."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gx1.audit import qualify_market_state_ta_cold_model_input_gather_v1 as audit


def _features(n):
    return {key: torch.zeros((n, 2), dtype=torch.int64 if key == 'ctx_cat'
                               else torch.float32) for key in audit.FEATURE_KEYS}


def test_plan_binds_qualified_cold_receipts_and_no_model():
    plan = audit.build_plan()
    assert plan['inputs']['coverage']['sha256'] == (
        'ca030948f3f450d19e7af6244d881db7c5cc7cb10157a53aba368d5bfa55ea2c')
    assert plan['inputs']['prior_result']['sha256'] == (
        '224f6f3f9565e7f1ad22eea59dcfdda8ddc1f19a3cec115d8da803b620663d66')
    assert plan['inputs']['prior_terminal']['sha256'] == (
        '6e95acc26c2d0b7eddefa00f3515f0c20321cf8a8d98b06e6f5b2719fdf40914')
    assert plan['scope']['fit_rows'] == 29574
    assert plan['scope']['score_rows'] == 17489
    assert plan['scope']['fit_tail_rows'] == 6
    assert plan['scope']['score_tail_rows'] == 1
    assert plan['scope']['raw_values_not_normalized'] is True
    assert plan['scope']['model_constructions'] == plan['scope']['model_forwards'] == 0
    assert plan['execution']['training_authorized'] is False


def test_batch_summary_requires_finite_cpu_model_input_schema():
    rows = np.array([1], dtype=np.int64)
    issues = rows * 300_000_000_000
    report = audit.summarize_batch('SCORE_TAIL1', _features(1), rows, issues)
    assert report['rows'] == 1 and report['raw_values_not_normalized'] is True
    bad = _features(1)
    bad['seq_x'][0, 0] = float('nan')
    with pytest.raises(RuntimeError, match='FEATURE_TENSOR_seq_x'):
        audit.summarize_batch('SCORE_TAIL1', bad, rows, issues)
    bad = _features(1)
    bad['ctx_cat'] = bad['ctx_cat'].float()
    with pytest.raises(RuntimeError, match='FEATURE_TENSOR_ctx_cat'):
        audit.summarize_batch('SCORE_TAIL1', bad, rows, issues)


def test_gather_uses_exact_first_and_tail_axes():
    parent = np.arange(47338, dtype=np.int64)
    issue = parent * 300_000_000_000
    calls = []
    def features(label, rows, issues):
        calls.append((label, int(rows[0]), len(rows)))
        return _features(len(rows))
    context = {'axes': {'parent_rows': parent, 'decision_ns': issue},
               'selected': np.arange(29849, 47338, dtype=np.int64),
               'features': features}
    reports = audit.gather_four(context)
    assert calls == [('TRAIN', 0, 16), ('TRAIN', 29568, 6),
                     ('SCORE', 29849, 16), ('SCORE', 47337, 1)]
    assert [item['rows'] for item in reports] == [16, 6, 16, 1]
