"""Pure native-M5 target clocks, masks, direction proofs and prefix stability."""
from dataclasses import replace

import numpy as np
import pytest

from gx1.contracts import market_state_native_m5_targets_v1 as target

SOURCE = {'path': '/synthetic/native_m5_manifest.json', 'sha256': 'a' * 64}


def source(n=5000):
    starts = np.arange(n, dtype=np.int64) * target.TICK_NS
    ohlc = np.tile(np.array([100., 100.2, 99.8, 100.], np.float64), (n, 1))
    return starts, ohlc


def build(starts, ohlc):
    return target.build_native_m5_targets(
        starts, ohlc, source_ref=SOURCE, cutoff_ns=int(starts[-1]) + 2 * target.TICK_NS)


@pytest.fixture(scope='module')
def tape():
    return build(*source())


def test_prefix_append_invariance_includes_all_five_timeframes(tape):
    longer = build(*source(5400))
    assert tape.atr.shape == (5000, 5)
    for name in ('issue_ns', 'source_row', 'close', 'atr', 'atr_available_ns',
                 'tf_bar_m5_counts', 'source_gap', 's_allowed',
                 's_span_complete', 's_local_range'):
        np.testing.assert_equal(getattr(tape, name), getattr(longer, name)[:5000])
    assert np.isfinite(tape.atr[-1]).all()
    assert (tape.atr[-1] > 0).all()
    assert (tape.atr_available_ns <= tape.issue_ns[:, None]).all()
    assert ((tape.tf_bar_m5_counts[:, -1] > 0)
            & (tape.tf_bar_m5_counts[:, -1] < target.NATIVE_K[-1])).any()
    query = np.broadcast_to(np.asarray(target.NATIVE_K, np.int64), (1, 5)).copy()
    issues = tape.issue_ns[[4500]]
    for key, values in tape.gather_state(issues, query, fold_upper_ns=tape.cutoff_ns).items():
        np.testing.assert_equal(values, longer.gather_state(
            issues, query, fold_upper_ns=tape.cutoff_ns)[key])
    for key, values in tape.gather_displacement(issues, fold_upper_ns=tape.cutoff_ns).items():
        np.testing.assert_equal(values, longer.gather_displacement(
            issues, fold_upper_ns=tape.cutoff_ns)[key])


def test_wide_bar_proves_activity_and_preserves_unknown_side(tape):
    assert tape.s_allowed[-1].tolist() == [target.ACTIVE_UNKNOWN] * 5
    assert target._ending_direction(target.UNKNOWN, 100., 100., 101., 99.,
                                    100., .5)[0] == target.UNKNOWN
    assert target._ending_direction(target.UNKNOWN, 100., 100., 101., 99.,
                                    101., .5)[0] == target.UP
    assert target._ending_direction(target.UNKNOWN, 100., 100., 101., 99.,
                                    99., .5)[0] == target.DOWN


def test_displacement_is_independent_of_state_masks_and_uses_own_maturity(tape):
    issues = tape.issue_ns[[4500]]
    report = tape.gather_displacement(issues, fold_upper_ns=tape.cutoff_ns)
    assert report['eligible'].all()
    assert report['allowed'].tolist() == [[target.FLAT] * 5]
    assert np.isfinite(report['net_change'][report['eligible']]).all()
    assert np.isfinite(report['abs_change_over_atr'][report['eligible']]).all()
    hidden_state = replace(tape, s_allowed=np.full_like(tape.s_allowed, target.UNKNOWN))
    np.testing.assert_equal(report['allowed'], hidden_state.gather_displacement(
        issues, fold_upper_ns=tape.cutoff_ns)['allowed'])
    fold_upper = int(tape.issue_ns[4788])
    immature = tape.gather_displacement(issues, fold_upper_ns=fold_upper)
    assert not immature['mature'][0, -1]
    assert not immature['eligible'][0, -1]
    assert immature['allowed'][0, -1] == target.UNKNOWN
    assert immature['source_row'][0, -1] == -1
    assert np.isnan(immature['net_change'][0, -1])


def test_unknown_source_gap_resets_state_span_but_exact_F_endpoint_survives():
    starts, ohlc = source()
    starts, ohlc = np.delete(starts, 4510), np.delete(ohlc, 4510, axis=0)
    tape = build(starts, ohlc)
    assert tape.source_gap[4510]
    assert tape.s_span_complete[4510].tolist() == [True, False, False, False, False]
    assert tape.s_allowed[4510, 1:].tolist() == [target.UNKNOWN] * 4
    report = tape.gather_displacement(tape.issue_ns[[4499]], fold_upper_ns=tape.cutoff_ns)
    assert report['endpoint_present'][0, 2] and report['eligible'][0, 2]
    assert report['gap_between'][0, 2]
    missing = tape.gather_displacement(tape.issue_ns[[4509]], fold_upper_ns=tape.cutoff_ns)
    assert not missing['endpoint_present'][0, 0]
    assert missing['allowed'][0, 0] == target.UNKNOWN


def test_input_arrays_are_owned_and_immutable(tape):
    with pytest.raises(ValueError):
        tape.issue_ns.setflags(write=True)
    with pytest.raises(ValueError):
        tape.s_allowed[0, 0] = target.UP
    with pytest.raises(TypeError):
        tape.source_ref['sha256'] = 'b' * 64


@pytest.mark.parametrize('change', ('clock', 'dtype', 'ohlc', 'cutoff'))
def test_rejects_invalid_source_before_target_build(change):
    starts, ohlc = source(20)
    cutoff = int(starts[-1]) + 2 * target.TICK_NS
    if change == 'clock':
        starts[3] += 1
    elif change == 'dtype':
        ohlc = ohlc.astype(np.float32)
    elif change == 'ohlc':
        ohlc[1, 1] = 98.
    else:
        cutoff = int(starts[-1]) + target.TICK_NS
    with pytest.raises(ValueError, match='NATIVE_M5_TARGET_(SOURCE_CLOCK|OHLC)'):
        target.build_native_m5_targets(starts, ohlc, source_ref=SOURCE, cutoff_ns=cutoff)


def test_state_query_masks_maturity_and_rejects_non_source_issue(tape):
    issue = tape.issue_ns[[4500]]
    query = np.broadcast_to(np.asarray(target.NATIVE_K, np.int64), (1, 5)).copy()
    result = tape.gather_state(issue, query, fold_upper_ns=int(tape.issue_ns[4788]))
    assert result['allowed'][0, -1] == target.UNKNOWN
    assert not result['eligible'][0, -1] and not result['mature'][0, -1]
    with pytest.raises(ValueError, match='ISSUE_AXIS'):
        tape.gather_state(issue + 1, query, fold_upper_ns=tape.cutoff_ns)
