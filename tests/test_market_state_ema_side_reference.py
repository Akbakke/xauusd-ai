"""Synthetic EMA-reference checks; no full cache payload or physical run."""
import json
import signal
from types import SimpleNamespace

import numpy as np
import pytest

from gx1.audit import measure_market_state_ema_side_reference_v1 as audit
from gx1.contracts import gx1_capped_execution_v1 as cap_contract


HOUR = 3_600_000_000_000


def _cache(labels, states):
    features = np.zeros((len(labels), 253), dtype=np.float32)
    features[:, audit.FEATURE_COLUMN] = states
    return np.asarray(labels, dtype=np.int64), features


def test_h4_and_shifted_d1_use_last_closed_bar():
    h4_ts, h4 = _cache(np.array([2, 6, 10, 14]) * HOUR, [0, 1, 0, 1])
    values, rows = audit._asof_binary(
        np.array([10, 12, 14], np.int64) * HOUR, h4_ts, h4,
        tf='H4', train_end_ns=16 * HOUR)
    assert values.tolist() == [1, 1, 0]
    assert rows.tolist() == [1, 1, 2]
    d1_ts, d1 = _cache(np.array([-2, 22, 46]) * HOUR, [0, 1, 0])
    values, rows = audit._asof_binary(
        np.array([24, 46, 60], np.int64) * HOUR, d1_ts, d1,
        tf='D1', train_end_ns=70 * HOUR)
    assert values.tolist() == [0, 1, 1]
    assert rows.tolist() == [0, 1, 1]


def test_later_development_cache_values_are_not_selected_or_scanned():
    duration = audit.DURATION_NS['M5']
    ts, cache = _cache(np.array([0, 1, 2, 3, 100, 99]) * duration,
                       [0, 1, 0, 1, np.nan, np.nan])
    values, rows = audit._asof_binary(
        np.array([2, 3], np.int64) * duration, ts, cache,
        tf='M5', train_end_ns=10 * duration)
    assert values.tolist() == [1, 0]
    assert rows.tolist() == [1, 2]


@pytest.mark.parametrize('invalid', [np.nan, 0.5])
def test_nonbinary_ema_fails_closed(invalid):
    ts, cache = _cache(np.array([0, 5, 10, 15]) * audit.DURATION_NS['M5'],
                       [0, invalid, 1, 0])
    issue = np.array([6, 7], np.int64) * audit.DURATION_NS['M5']
    with pytest.raises(RuntimeError, match='EMA_DOMAIN'):
        audit._asof_binary(issue, ts, cache, tf='M5',
                           train_end_ns=20 * audit.DURATION_NS['M5'])


def test_certified_side_only_fits_ema_bins_and_activity_is_fixed():
    bits = np.array([1, 1, 1, 4, 5, 7], dtype=np.uint8)
    allowed = np.tile(bits[:, None], (1, 5))
    eligible = np.ones((6, 5), dtype=np.bool_)
    mature = eligible.copy()
    feature = np.tile(np.array([0, 0, 1, 1, 1, 0], np.uint8)[:, None], (1, 5))
    prior, p_up, counts = audit._fit_side_bins(
        {'allowed': allowed, 'eligible': eligible, 'mature': mature}, feature, feature)
    assert counts['M5'] == {
        '0': {'fit_certified_side': 2, 'fit_up': 0, 'score_rows': 3},
        '1': {'fit_certified_side': 2, 'fit_up': 1, 'score_rows': 3}}
    assert p_up[0].tolist() == [0.25, 0.5]
    predicted = audit._predictions(feature, prior, p_up)
    assert np.allclose(predicted[:, :, 0] + predicted[:, :, 2],
                       (6 / 7), rtol=0, atol=1e-15)
    assert predicted[0, 0, 2] < predicted[2, 0, 2]
    assert prior['counts']['M5']['certified_side'] == 4


def test_missing_certified_fit_bin_fails_without_fallback():
    allowed = np.full((3, 5), 1, dtype=np.uint8)
    eligible = mature = np.ones((3, 5), dtype=np.bool_)
    feature = np.zeros((3, 5), dtype=np.uint8)
    with pytest.raises(RuntimeError, match='EMPTY_CERTIFIED_FIT_BIN:M5:1'):
        audit._fit_side_bins(
            {'allowed': allowed, 'eligible': eligible, 'mature': mature}, feature,
            np.ones((2, 5), dtype=np.uint8))


def test_unused_unfitted_bin_is_unestimated_without_fallback():
    allowed = np.full((3, 5), 1, dtype=np.uint8)
    eligible = mature = np.ones((3, 5), dtype=np.bool_)
    fit_feature = np.zeros((3, 5), dtype=np.uint8)
    score_feature = np.zeros((2, 5), dtype=np.uint8)
    prior, p_up, counts = audit._fit_side_bins(
        {'allowed': allowed, 'eligible': eligible, 'mature': mature},
        fit_feature, score_feature)
    assert np.isnan(p_up[:, 1]).all()
    assert counts['M5']['1'] == {
        'fit_certified_side': 0, 'fit_up': 0, 'score_rows': 0}
    assert np.isfinite(audit._predictions(score_feature, prior, p_up)).all()


def test_fold_pairs_scores_on_identical_mature_rows():
    day = 24 * 60 * audit.target.MINUTE_NS
    fit_times = audit.target.TRAIN_START_NS + np.arange(61, 66, dtype=np.int64) * day
    score_times = audit.target.FOLD_BOUNDARIES_NS[0] + np.arange(1, 6, dtype=np.int64) * day
    issues = np.concatenate((fit_times, score_times))
    fit_bits = [1, 1, 4, 4, 5]
    score_bits = [1, 4, 2, 5, 7]
    allowed = np.tile(np.asarray(fit_bits + score_bits, dtype=np.uint8)[:, None],
                      (1, 5))
    bound = audit.target.BoundFiveEndpointTargets(
        parent_rows=np.arange(10, dtype=np.int64), decision_ns=issues,
        allowed_set=allowed,
        m1_source_row=np.zeros((10, 5), dtype=np.int64),
        m5_source_row=np.arange(10, dtype=np.int64),
        unjoined_target_rows=0)
    feature = np.tile(np.asarray([0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
                                 dtype=np.uint8)[:, None], (1, 5))
    frozen = audit.base._fold_report(bound, 1)
    report = audit._fold(bound, feature, 1, frozen)
    assert report['fit_rows'] == report['score_rows'] == 5
    for tf in audit.TIMEFRAMES:
        assert report['paired_score'][tf]['activity']['brier_gain_sum'] == 0
        assert report['ema_score_sums'][tf]['activity'] == frozen['score_sums'][tf]['activity']
        assert report['paired_score'][tf]['conditional_side']['count'] == 2
        assert report['paired_score'][tf]['three_class']['count'] == 3
        assert report['fit_bin_counts'][tf]['0']['score_rows'] == 2
        assert report['fit_bin_counts'][tf]['1']['score_rows'] == 3
    corrupt = dict(frozen)
    corrupt['score_sums'] = {tf: {name: dict(value) for name, value in scores.items()}
                             for tf, scores in frozen['score_sums'].items()}
    corrupt['score_sums']['M5']['conditional_side']['count'] += 1
    with pytest.raises(RuntimeError, match='FROZEN_PRIOR_SCORE'):
        audit._fold(bound, feature, 1, corrupt)


def test_preclaim_receipts_are_bound_without_cache_array_read(monkeypatch):
    original = audit._sha
    def no_cache_hash(path):
        if str(path).endswith('.npy'):
            raise AssertionError('cache payload hash before claim')
        return original(path)
    monkeypatch.setattr(audit, '_sha', no_cache_hash)
    prior, cache_refs = audit._small_inputs()
    assert prior['target_rows_joined'] == 118249
    assert set(cache_refs) == set(audit.TIMEFRAMES)
    assert len(cache_refs) == 5


@pytest.mark.parametrize('failure', [None, 'error', 'timeout'])
def test_one_use_terminal_success_and_failure(tmp_path, monkeypatch, failure):
    out = tmp_path / 'out'
    plan_file = tmp_path / 'plan.json'
    plan_file.write_text('{}')
    plan_ref = {'path': str(plan_file), 'sha256': audit._sha(plan_file)}
    policy_file = tmp_path / 'policy.json'
    policy_file.write_text(json.dumps({
        'training_enabled': False, 'native_training_enabled': False,
        'native_launch_authorized': False, 'full_val_allowed': False,
        audit.STAGE: {'status': 'ADMITTED', 'decision': 'ADMITTED',
                      'plan': plan_ref, 'execution_authorized': True,
                      'data_reads_authorized': True, 'claimed_invocations': 0,
                      'completed_invocations': 0, 'max_invocations': 1,
                      'relaunch_allowed': False, 'memory_max': '4G',
                      'swap_max': '512M', 'wall_seconds': 1800,
                      'test_data_used': False,
                      'checkpoint_reads_authorized': False,
                      'broker_access_allowed': False,
                      'model_forwards_authorized': 0,
                      'optimizer_steps_authorized': 0,
                      'training_authorized': False, 'test_authorized': False,
                      'output_root': str(out)}}))
    monkeypatch.setattr(audit, 'POLICY', policy_file)
    monkeypatch.setattr(audit, 'OUT_ROOT', out)
    monkeypatch.setattr(audit, '_clean_head', lambda: 'a' * 40)
    monkeypatch.setattr(audit, '_inspect_plan', lambda *args: (
        {}, plan_ref, 'a' * 40, {}, {}))
    monkeypatch.setattr(audit, 'sys', SimpleNamespace(modules={}))
    monkeypatch.setattr(cap_contract, 'require_capped_cpu_audit_execution',
                        lambda: {**audit.LIMITS, 'pids': 64,
                                 'numerical_threads': 1,
                                 'cgroup_path': '/sys/fs/cgroup/test'})
    monkeypatch.setattr(cap_contract, 'require_capped_lock_ancestry',
                        lambda: {'scope': '/test'})
    def measure(*_args):
        if failure == 'error':
            raise RuntimeError('synthetic failure')
        if failure == 'timeout':
            signal.raise_signal(signal.SIGALRM)
        return {'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False}
    monkeypatch.setattr(audit, '_measure', measure)
    old_handler = signal.getsignal(signal.SIGALRM)
    if failure:
        with pytest.raises(TimeoutError if failure == 'timeout' else RuntimeError):
            audit.run(plan_file, plan_ref['sha256'], out)
        assert {p.name for p in out.iterdir()} == {
            'CLAIM.json', 'FAILED.json', 'TERMINAL.json'}
        assert json.loads((out / 'TERMINAL.json').read_text())['status'] == 'FAILED'
    else:
        audit.run(plan_file, plan_ref['sha256'], out)
        assert {p.name for p in out.iterdir()} == {
            'CLAIM.json', 'RESULT.json', 'TERMINAL.json'}
    assert signal.getsignal(signal.SIGALRM) == old_handler
    assert json.loads((out / 'TERMINAL.json').read_text())[
        'no_relaunch_under_this_plan'] is True
    with pytest.raises(FileExistsError):
        audit.run(plan_file, plan_ref['sha256'], out)
