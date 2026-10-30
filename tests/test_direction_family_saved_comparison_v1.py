"""Synthetic paired-cohort and sign-convention tests, no market payloads."""
import copy
import hashlib
import numpy as np
import pytest
from gx1.scripts.compare_entry_direction_matched_exposure_v1 import compare_family_value_predictions

DAY = 86_400_000_000_000
MINUTE = 60_000_000_000

def arrays(start_day, first_parent):
    decisions = np.repeat(np.arange(start_day, start_day + 56, dtype=np.int64) * DAY, 2)
    decisions[1::2] += 5 * MINUTE
    y = np.tile(np.array([1, -1], np.int8), 56)
    sign = np.repeat(y[:, None], 60, axis=1)
    return {'parent_rows': np.arange(first_parent, first_parent + 112, dtype=np.int64),
        'bar_start_ns': decisions - 5 * MINUTE, 'decision_ns': decisions,
        'terminal_time_ns': decisions[:, None] + np.arange(1, 61, dtype=np.int64)[None, :] * MINUTE,
        'observed': np.ones((112, 60), np.bool_), 'sign': sign,
        'logit_up': sign.astype(np.float32) * .25,
        'context__session': np.zeros(112, np.int8)}

def inputs():
    train, later = arrays(10, 0), arrays(100, 112)
    excluded_train, excluded_later = copy.deepcopy(train), copy.deepcopy(later)
    excluded_train['logit_up'][:] = 0
    excluded_later['logit_up'][:] = 0
    return {'baseline': train, 'family': excluded_train}, {'baseline': later, 'family': excluded_later}

def compare(train, later):
    canonical_train, canonical_later = inputs()
    hashes = {label: {k: hashlib.sha256(data['baseline'][k].tobytes()).hexdigest()
        for k in ('parent_rows', 'bar_start_ns', 'decision_ns')}
        for label, data in (('train', canonical_train), ('later', canonical_later))}
    return compare_family_value_predictions(train, later, arm_order=['baseline', 'family'],
        train_cutoff=80 * DAY, later_cutoff=170 * DAY,
        expected_train_rows=112, expected_later_rows=112, expected_axis_hashes=hashes,
        replicates=32, seed=20260911, familywise_alpha=.05)

def test_full_train_all_horizons_and_difference_orientation():
    train, later = inputs()
    result = compare(train, later)
    assert len(result['panels']['train']) == len(result['panels']['later']) == 60
    for partition in result['panels'].values():
        assert all(p['counts']['parents'] == p['counts']['eligible'] == 112 for p in partition)
        assert all(p['arms']['baseline']['auc'] == 1 for p in partition)
        assert all(p['arms']['family']['auc'] == .5 for p in partition)
        assert all(p['arms']['fixed_train_prior']['auc'] == .5 for p in partition)
    primary = result['primary_five_minute'][0]
    assert primary['full_minus_excluded']['auc'] == .5
    assert primary['full_minus_excluded']['logloss'] < 0
    assert primary['full_minus_excluded']['brier'] < 0
    assert all(primary['support_by_metric'].values())
    assert result['per_interval_alpha'] == .05 / 6
    assert result['economic_edge_qualified'] is False

def test_identical_predictions_have_zero_contrasts():
    train, later = inputs()
    for partition in (train, later):
        partition['family']['logit_up'][:] = partition['baseline']['logit_up']
    primary = compare(train, later)['primary_five_minute'][0]
    assert set(primary['full_minus_excluded'].values()) == {0}
    for group in primary['intervals']:
        assert all(i['low'] == i['high'] == 0 for i in group['full_minus_excluded'].values())

@pytest.mark.parametrize('field', ['parent_rows', 'decision_ns', 'sign', 'observed', 'context__session'])
def test_mismatch_rejected(field):
    train, later = inputs()
    x = later['family'][field]
    x.flat[0] = (not x.flat[0]) if x.dtype == np.bool_ else x.flat[0] + 1
    with pytest.raises(RuntimeError):
        compare(train, later)

def test_missing_arm_and_partial_train_rejected():
    train, later = inputs()
    with pytest.raises(RuntimeError, match='ARM_SET'):
        compare(train, {'baseline': later['baseline']})
    for arm in train:
        train[arm] = {k: v[:16] for k, v in train[arm].items()}
    with pytest.raises(RuntimeError):
        compare(train, later)

def test_null_and_missing_partition_without_complete_case_filter():
    train, later = inputs()
    for partition in (train, later):
        for data in partition.values():
            data['sign'][0, 4] = 0
            data['sign'][1, 4] = 0
            data['observed'][1, 4] = False
    result = compare(train, later)
    count = result['panels']['later'][4]['counts']
    assert count == {'parents':112, 'eligible':110, 'mature_zero':1, 'mature_missing':1, 'immature':0}
    assert result['own_train_prior'][4]['eligible'] == 110


def test_same_length_wrong_cohort_all_arms_rejected():
    train, later = inputs()
    for data in later.values():
        data['parent_rows'] += 1
    with pytest.raises(RuntimeError, match='DECLARED_COHORT'):
        compare(train, later)
