"""Observed native bar signs, independent components and next-bar target clocks."""
import numpy as np
import pytest

from gx1.contracts import market_state_native_m5_targets_v1 as target

SOURCE = {'path': '/synthetic/native_m5_manifest.json', 'sha256': 'a' * 64}


def source(n=1000):
    starts = np.arange(n, dtype=np.int64) * target.TICK_NS
    opening = 100. + np.arange(n) * .001
    closing = 100. + np.sin(np.arange(n) / 10.)
    ohlc = np.column_stack((opening, np.maximum(opening, closing) + 2.,
                           np.minimum(opening, closing) - 2., closing))
    return starts, ohlc


def build(starts, ohlc):
    return target.build_native_m5_targets(
        starts, ohlc, source_ref=SOURCE, cutoff_ns=int(starts[-1]) + 2 * target.TICK_NS)


@pytest.fixture(scope='module')
def tape():
    return build(*source())


def test_all_observed_tf_bar_comparisons_known_including_equal_and_mixed():
    starts, ohlc = source()
    ohlc[:, 3] = 100.
    tape = build(starts, ohlc)
    for tf in target.TIMEFRAMES:
        bars = tape.bars_by_timeframe[tf]
        assert len(bars.ohlc) > 1
        assert not bars.observed_previous[0] and bars.observed_previous[1:].all()
        np.testing.assert_equal(bars.open_sign[1:],
                                np.sign(bars.ohlc[1:, 0] - bars.ohlc[:-1, 0]))
        np.testing.assert_equal(bars.close_sign[1:], 0)
        assert (bars.open_sign[1:] == 1).all()
        assert (bars.available_ns < tape.cutoff_ns).all()
        assert (bars.open_available_ns <= bars.available_ns).all()
    starts, ohlc = source(4)
    ohlc[:, 0] = [100., 101., 101., 99.]
    ohlc[:, 3] = [101., 100., 100., 102.]
    ohlc[:, 1] = 110.
    ohlc[:, 2] = 90.
    bars = build(starts, ohlc).bars_by_timeframe['M5']
    assert bars.open_sign.tolist() == [0, 1, 0, -1]
    assert bars.close_sign.tolist() == [0, -1, 0, 1]
    assert bars.observed_previous.tolist() == [False, True, True, True]


def test_all_tf_append_invariance_and_owned_arrays(tape):
    longer = build(*source(1400))
    for name in ('issue_ns', 'source_row', 'ohlc', 'source_gap'):
        np.testing.assert_equal(getattr(tape, name), getattr(longer, name)[:1000])
    for tf in target.TIMEFRAMES:
        left, right = tape.bars_by_timeframe[tf], longer.bars_by_timeframe[tf]
        for name in left.__dataclass_fields__:
            np.testing.assert_equal(getattr(left, name), getattr(right, name)[:len(left.ohlc)])
        with pytest.raises(ValueError):
            left.open_sign.setflags(write=True)
    with pytest.raises(TypeError):
        tape.bars_by_timeframe['M5'] = None
    with pytest.raises(TypeError):
        tape.source_ref['sha256'] = 'b' * 64


def test_previous_observed_comparisons_survive_source_gaps_on_all_tf():
    starts, ohlc = source(1800)
    keep = np.r_[np.arange(400), np.arange(1000, 1800)]
    tape = build(starts[keep], ohlc[keep])
    assert tape.source_gap.sum() == 1
    for tf in target.TIMEFRAMES:
        bars = tape.bars_by_timeframe[tf]
        assert bars.source_gap.any()
        assert bars.observed_previous[bars.source_gap].all()
        np.testing.assert_equal(bars.previous_open[1:], bars.ohlc[:-1, 0])
        np.testing.assert_equal(bars.previous_close[1:], bars.ohlc[:-1, 3])
        np.testing.assert_equal(bars.open_sign[1:],
                                np.sign(bars.ohlc[1:, 0] - bars.ohlc[:-1, 0]))
        np.testing.assert_equal(bars.close_sign[1:],
                                np.sign(bars.ohlc[1:, 3] - bars.ohlc[:-1, 3]))


def test_future_targets_use_next_actual_native_close_and_separate_open_observability(tape):
    issues = tape.issue_ns[[500, 503]]
    out = tape.gather_next_bar(issues, fold_upper_ns=tape.cutoff_ns)
    assert out['eligible'].all()
    assert (out['validity_ns'] > issues[:, None]).all()
    assert not out['open_known_at_issue'][:, 0].any()
    assert out['open_known_at_issue'][0, -1]
    for j, tf in enumerate(target.TIMEFRAMES):
        bars = tape.bars_by_timeframe[tf]
        next_row = np.searchsorted(bars.available_ns, issues, side='right')
        np.testing.assert_equal(out['target_row'][:, j], next_row)
        np.testing.assert_equal(out['open_sign'][:, j], bars.open_sign[next_row])
        np.testing.assert_equal(out['close_sign'][:, j], bars.close_sign[next_row])
        np.testing.assert_equal(out['open_known_at_issue'][:, j],
                                bars.open_available_ns[next_row] <= issues)
    d1 = target.TIMEFRAMES.index('D1')
    assert out['validity_ns'][0, d1] != issues[0] + 288 * target.TICK_NS


def test_native_tf_target_does_not_require_m5_issue_at_its_close():
    starts, ohlc = source()
    starts, ohlc = np.delete(starts, 551), np.delete(ohlc, 551, axis=0)
    tape = build(starts, ohlc)
    issue = np.array([501 * target.TICK_NS], np.int64)
    out = tape.gather_next_bar(issue, fold_upper_ns=tape.cutoff_ns)
    assert out['validity_ns'][0, -1] == 552 * target.TICK_NS
    assert out['validity_ns'][0, -1] not in tape.issue_ns
    assert out['eligible'][0, -1] and out['observed_previous'][0, -1]
    assert out['m5_count'][0, -1] == 287


def test_fold_maturity_censors_labels_before_reading_next_bar(tape):
    issue = tape.issue_ns[[500]]
    upper = int(tape.bars_by_timeframe['D1'].available_ns[1])
    out = tape.gather_next_bar(issue, fold_upper_ns=upper)
    assert out['target_present'][0, -1]
    assert not out['mature'][0, -1] and not out['eligible'][0, -1]
    assert out['target_row'][0, -1] == -1
    assert not out['observed_previous'][0, -1]
    assert out['open_sign'][0, -1] == out['close_sign'][0, -1] == 0
    assert out['open_known_at_issue'][0, -1]
    with pytest.raises(ValueError, match='ISSUE_AXIS'):
        tape.gather_next_bar(issue + 1, fold_upper_ns=tape.cutoff_ns)


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
