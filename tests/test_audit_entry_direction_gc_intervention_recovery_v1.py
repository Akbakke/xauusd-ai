import copy

import numpy as np
import pytest

from gx1.audit.direction_gc_mtf_intervention_v1 import selected_batches
from gx1.contracts import entry_direction_probability_reference_v1 as probability
from gx1.scripts import audit_entry_direction_gc_intervention_recovery_v1 as recovery


def test_recovery_recomputes_all_original_scores_and_rejects_drift():
    n = 32
    parent = np.arange(n, dtype=np.int64)
    decision = 1000 + parent
    terminal = decision[:, None] + np.arange(1, 61, dtype=np.int64)[None, :]
    sign = np.tile(np.where(parent % 2 == 0, -1, 1).astype(np.int8)[:, None], (1, 60))
    observed = np.ones((n, 60), dtype=np.bool_)
    saved = {'parent_rows': parent, 'bar_start_ns': decision - 1, 'decision_ns': decision,
        'terminal_time_ns': terminal, 'observed': observed, 'sign': sign, 'logit_up': sign.astype(np.float32)}
    corrected = {key: value.copy() for key, value in saved.items() if key != 'logit_up'}
    labels = np.ascontiguousarray((sign[:16, 0] == 1).astype(np.int8)); logits = np.ascontiguousarray(saved['logit_up'][:16, 0], dtype=np.float64)
    m = probability.direction_probability_metrics(labels, logits)
    arm = {'auc': probability.direction_rank_diagnostics(labels, logits, score_kind='logit_up')['auc'],
        'logloss': float(np.mean(m['logloss_by_row'])), 'brier': float(np.mean(m['brier_by_row']))}
    result = {'selection': {'policy': 'every tenth original OOF batch', 'batch_size': 16, 'batch_stride': 10,
        'selected_parents': 16, 'selected_batches': 1},
        'metrics': {'per_horizon': [{'horizon_minutes': j + 1, 'eligible': 16, 'arms': {'original': dict(arm)}} for j in range(60)]}}
    proof = recovery.verify_original_scores(saved, corrected, result, first=0, last=n, cutoff=99999,
        select=selected_batches, probability=probability)
    assert proof['horizons_verified'] == 60 and proof['max_absolute_score_difference'] == 0
    wrong = copy.deepcopy(result); wrong['metrics']['per_horizon'][4]['arms']['original']['auc'] += .01
    with pytest.raises(RuntimeError, match='ORIGINAL_SCORE:5:auc'):
        recovery.verify_original_scores(saved, corrected, wrong, first=0, last=n, cutoff=99999,
            select=selected_batches, probability=probability)


def test_failed_run_cannot_be_misrepresented_as_successful_original_run():
    ref = {'path': '/original-plan', 'sha256': 'a' * 64}
    interval = {'low': -1., 'high': 1., 'valid_replicates': 10000, 'undefined_replicates': 0}
    result = {'schema_version': 'gx1_direction_gc_mtf_intervention_result_v1', 'status': 'COMPLETE',
        'plan': ref, 'source_head': 'abc', 'exact_saved_oof_logit_bytes_reproduced': True,
        'forward_calls': 218, 'optimizer_steps': 0, 'model_unchanged': True, 'test_data_used': False,
        'corrected_target_identity': 'exact_original_decimal_midpoint_v1',
        'metrics': {'per_interval_alpha': .05 / 12, 'paired_intervals': [
            {'horizon_minutes': h, 'scheme': scheme, 'intervals': {'auc': interval}}
            for h in (5, 60) for scheme in ('UTC_DAY', 'UTC_WEEK_MONDAY')]}}
    inner = {'status': 'COMPLETE', 'exit_code': 0, 'inputs_unchanged': True, 'source_unchanged': True, 'test_data_used': False}
    failed = {'status': 'FAILED', 'exit_code': 1, 'relaunch_allowed': False, 'error': 'NATIVE_DIRECTION_RESEARCH_WINDOW_UNBOUND_OPEN:/x'}
    claim = {'plan': ref, 'source_head': 'abc'}
    stdio = 'publish_bundle_directory_noreplace NATIVE_DIRECTION_RESEARCH_WINDOW_UNBOUND_OPEN:/x'
    guard = 'event=exit child_status=1 peak_core_temp_c=53 peak_memory_temp_c=56 peak_power_draw_w=132.17'
    recovery.validate_failed_run(result, inner, failed, claim, ref, stdio, guard)
    result['forward_calls'] = 217
    with pytest.raises(RuntimeError, match='INNER_RESULT'):
        recovery.validate_failed_run(result, inner, failed, claim, ref, stdio, guard)
