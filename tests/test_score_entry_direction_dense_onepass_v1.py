import numpy as np
import pytest

from gx1.scripts import score_entry_direction_dense_onepass_v1 as score


def fixture():
    days = 14
    decision = (np.int64(1735689600000000000)
        + np.repeat(np.arange(days, dtype=np.int64), 2) * score.DAY_NS
        + np.tile(np.array([0, 5], np.int64), days) * score.MINUTE_NS)
    n = len(decision)
    labels = np.tile(np.array([1, -1], np.int8), days)
    candidate = {'parent_rows': np.arange(n, dtype=np.int64), 'bar_start_ns': decision - 5 * score.MINUTE_NS,
        'decision_ns': decision, 'terminal_time_ns': decision[:, None] + np.arange(1, 61, dtype=np.int64)[None, :] * score.MINUTE_NS,
        'observed': np.ones((n, 60), np.bool_), 'sign': np.tile(labels[:, None], (1, 60)),
        'logit_up': np.tile(labels[:, None].astype(np.float32), (1, 60))}
    baseline = {k: v.copy() for k, v in candidate.items() if k in ('parent_rows', 'bar_start_ns', 'decision_ns')}
    baseline.update({k: candidate[k][:, [0, 4, 14]].copy() for k in ('terminal_time_ns', 'observed', 'sign')})
    baseline['logit_up'] = -candidate['logit_up'][:, [0, 4, 14]].copy()
    return candidate, baseline, int(decision[-1] + 61 * score.MINUTE_NS)


def test_sixty_fixed_outputs_and_paired_five_minute_intervals():
    candidate, baseline, cutoff = fixture()
    result = score.compare_saved_predictions(candidate, baseline, cutoff=cutoff, replicates=64, minimum_blocks=1)
    assert len(result['panels']) == 60
    assert all(x['auc'] == 1 and x['both_lower_bounds_above_half'] for x in result['panels'])
    assert result['simultaneous_day_half_width'] == result['simultaneous_week_half_width'] == 0
    primary = result['primary_five_minute']
    assert primary['candidate']['auc'] == 1 and primary['baseline']['auc'] == 0
    assert primary['auc_improvement_supported'] and primary['candidate_above_half_supported']
    assert all(x['candidate_minus_baseline']['auc']['low'] == 1 for x in primary['intervals'])


def test_wrong_anchor_or_clock_fails_before_any_comparison():
    candidate, baseline, cutoff = fixture()
    baseline['sign'][0, 1] *= -1
    with pytest.raises(RuntimeError, match='ANCHOR:sign'):
        score.compare_saved_predictions(candidate, baseline, cutoff=cutoff, replicates=8, minimum_blocks=1)
    candidate, baseline, cutoff = fixture()
    candidate['terminal_time_ns'][0, 7] += 1
    with pytest.raises(RuntimeError, match='TARGET_CLOCK_DOMAIN'):
        score.compare_saved_predictions(candidate, baseline, cutoff=cutoff, replicates=8, minimum_blocks=1)


def test_constant_scores_count_tied_rank_pairs_as_half():
    candidate, baseline, cutoff = fixture()
    candidate['logit_up'][:] = 0
    baseline['logit_up'][:] = 0
    result = score.compare_saved_predictions(candidate, baseline, cutoff=cutoff, replicates=32, minimum_blocks=1)
    assert all(x['auc'] == .5 and not x['both_lower_bounds_above_half'] for x in result['panels'])
    assert not result['primary_five_minute']['auc_improvement_supported']
    assert not result['primary_five_minute']['candidate_above_half_supported']


def test_paired_intervals_match_physically_duplicated_calendar_blocks():
    from gx1.contracts.entry_direction_probability_reference_v1 import direction_rank_diagnostics
    from gx1.scripts.compare_entry_direction_saved_bps_v1 import block_weights

    candidate, baseline, cutoff = fixture()
    rng = np.random.default_rng(17)
    candidate['logit_up'][:] = rng.normal(size=(len(candidate['parent_rows']), 1)).astype(np.float32)
    baseline['logit_up'][:] = rng.normal(size=(len(candidate['parent_rows']), 1)).astype(np.float32)
    count = 32
    result = score.compare_saved_predictions(candidate, baseline, cutoff=cutoff, replicates=count, minimum_blocks=1)
    day = candidate['decision_ns'] // score.DAY_NS
    labels = (candidate['sign'][:, 4] == 1).astype(np.int8)
    for interval, grouping, seed in zip(result['primary_five_minute']['intervals'], (day, (day + 3) // 7), (20261027, 20261028)):
        blocks, ids = np.unique(grouping, return_inverse=True)
        weights = block_weights(len(blocks), seed=seed, np=np, replicates=count)
        differences = []
        for weights_row in weights:
            ix = np.repeat(np.arange(len(day)), weights_row[ids])
            y = np.ascontiguousarray(labels[ix]); old = np.ascontiguousarray(baseline['logit_up'][ix, 1], dtype=np.float64)
            new = np.ascontiguousarray(candidate['logit_up'][ix, 4], dtype=np.float64)
            differences.append(direction_rank_diagnostics(y, new, score_kind='logit_up')['auc']
                - direction_rank_diagnostics(y, old, score_kind='logit_up')['auc'])
        low, high = np.quantile(differences, [.00625, .99375], method='linear')
        actual = interval['candidate_minus_baseline']['auc']
        assert actual['low'] == pytest.approx(low, abs=1e-15)
        assert actual['high'] == pytest.approx(high, abs=1e-15)


def test_synthetic_sixty_head_scoring_runs_under_read_and_model_guards(tmp_path):
    import subprocess
    import sys
    code = r'''
import runpy, sys
from pathlib import Path
import numpy as np
from gx1.contracts import gx1_capped_execution_v1, immutable_event_authority_v1, local_random_access_campaign_v2
from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
from gx1.contracts.entry_native_direction_context_report_v1 import CONTEXT_SPEC, _panel
from gx1.contracts.entry_direction_probability_reference_v1 import direction_rank_diagnostics
from gx1.contracts.entry_direction_block_rank_v1 import block_rank_pairs, weighted_rank_pairs
from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
from gx1.scripts import compare_entry_direction_matched_exposure_v1 as matched
from gx1.scripts import compare_entry_direction_saved_bps_v1 as saved
from gx1.scripts.score_entry_direction_dense_minutes_v1 import coarsen_block_pairs, simultaneous_intervals
from gx1.scripts.score_entry_direction_dense_onepass_v1 import compare_saved_predictions
case = runpy.run_path('tests/test_score_entry_direction_dense_onepass_v1.py')
candidate, baseline, cutoff = case['fixture']()
sys.dont_write_bytecode = True
sys.addaudithook(candle.make_guard(Path(sys.argv[1]).resolve(), [], [], synthetic=True))
sys.setprofile(candle.no_queries_or_models)
r = compare_saved_predictions(candidate, baseline, cutoff=cutoff, replicates=16, minimum_blocks=1)
assert len(r['panels']) == 60 and r['primary_five_minute']['auc_improvement_supported']
sys.setprofile(None)
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
