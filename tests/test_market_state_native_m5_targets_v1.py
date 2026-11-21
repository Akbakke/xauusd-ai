"""Observed native bar signs, independent components and next-bar target clocks."""
import copy

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


def test_noise_margin_preserves_all_raw_keys_bytes_and_default_build_does_not_compute_atr(monkeypatch):
    calls=[];canonical=target.wilder_atr
    def record(high,low,close,period):
        calls.append((len(high),period));return canonical(high,low,close,period)
    monkeypatch.setattr(target,'wilder_atr',record)
    tape=build(*source())
    assert calls==[]
    issues=tape.issue_ns[[500,503]]
    raw=tape.gather_next_bar(issues,fold_upper_ns=tape.cutoff_ns)
    augmented=tape.gather_next_bar_noise_margin(issues,fold_upper_ns=tape.cutoff_ns,margin_atr=.125)
    assert calls==[(len(tape.bars_by_timeframe[tf].ohlc),14)for tf in target.TIMEFRAMES]
    for name,value in raw.items():
        assert augmented[name].dtype==value.dtype and augmented[name].shape==value.shape
        assert augmented[name].tobytes()==value.tobytes()
    spec=augmented['noise_margin_contract']
    assert spec['raw_target_contract_sha256']==target.contract()['contract_sha256']
    assert spec['metadata_as_model_input']is False
    assert 'not_quiet_path_or_whole_market_FLAT'in spec['meaning']


@pytest.mark.parametrize('delta,expected',[(0.,0),(.125,0),(-.125,0),(.25,1),(-.25,-1),(.5,1),(-.5,-1)])
def test_noise_margin_exact_zero_inside_and_signed_boundary(delta,expected):
    starts=np.arange(30,dtype=np.int64)*target.TICK_NS
    ohlc=np.tile(np.array([100.,101.,99.,100.]),(30,1))
    ohlc[14,[0,3]]+=delta
    tape=build(starts,ohlc);issue=tape.issue_ns[[13]]
    out=tape.gather_next_bar_noise_margin(issue,fold_upper_ns=tape.cutoff_ns,margin_atr=.125)
    assert out['noise_atr14_asof'][0,0]==2. and out['noise_margin_price'][0,0]==.25
    for task in ('open','close'):
        assert out['noise_'+task+'_known'][0,0]and out['noise_'+task+'_eligible'][0,0]
        assert out['noise_'+task+'_delta'][0,0]==delta
        assert out['noise_'+task+'_sign'][0,0]==expected
        assert out[task+'_sign'][0,0]==np.sign(delta)


def test_noise_margin_warmup_keeps_equal_known_and_nonzero_unknown_and_censors_immature():
    starts=np.arange(30,dtype=np.int64)*target.TICK_NS
    ohlc=np.tile(np.array([100.,101.,99.,100.]),(30,1))
    for delta in (0.,.125):
        changed=ohlc.copy();changed[3,[0,3]]+=delta
        tape=build(starts,changed);issue=tape.issue_ns[[2]]
        out=tape.gather_next_bar_noise_margin(issue,fold_upper_ns=tape.cutoff_ns,margin_atr=.125)
        assert np.isnan(out['noise_atr14_asof'][0,0])and np.isnan(out['noise_margin_price'][0,0])
        for task in ('open','close'):
            assert out['noise_'+task+'_sign'][0,0]==0
            assert bool(out['noise_'+task+'_known'][0,0])==(delta==0)
            assert bool(out['noise_'+task+'_eligible'][0,0])==(delta==0)
        upper=int(tape.bars_by_timeframe['M5'].available_ns[3])
        censored=tape.gather_next_bar_noise_margin(issue,fold_upper_ns=upper,margin_atr=.125)
        assert censored['target_present'][0,0]and not censored['mature'][0,0]
        for task in ('open','close'):
            assert not censored['noise_'+task+'_known'][0,0]
            assert not censored['noise_'+task+'_eligible'][0,0]
            assert np.isnan(censored['noise_'+task+'_delta'][0,0])


def test_noise_margin_all_tf_use_last_closed_prefix_atr_and_ignore_future_ranges():
    starts,ohlc=source(5000);tape=build(starts,ohlc);issues=tape.issue_ns[[4500]]
    out=tape.gather_next_bar_noise_margin(issues,fold_upper_ns=tape.cutoff_ns,margin_atr=.125)
    for j,tf in enumerate(target.TIMEFRAMES):
        bars=tape.bars_by_timeframe[tf];at=np.flatnonzero(bars.available_ns<=issues[0])[-1]
        assert out['noise_atr_asof_row'][0,j]==at
        assert out['noise_atr_asof_available_ns'][0,j]==bars.available_ns[at]<=issues[0]
        high,low,close=bars.ohlc[:,1],bars.ohlc[:,2],bars.ohlc[:,3]
        true_range=high-low
        true_range[1:]=np.maximum.reduce((true_range[1:],abs(high[1:]-close[:-1]),abs(low[1:]-close[:-1])))
        reference=np.full(len(bars.ohlc),np.nan)
        reference[13]=np.mean(true_range[:14])
        for row in range(14,len(reference)):reference[row]=(13*reference[row-1]+true_range[row])/14
        assert out['noise_atr14_asof'][0,j]==reference[at]
        assert out['noise_margin_price'][0,j]==.125*reference[at]
    changed=ohlc.copy();changed[starts>=issues[0],1]+=100.
    future=build(starts,changed).gather_next_bar_noise_margin(issues,fold_upper_ns=tape.cutoff_ns,margin_atr=.125)
    for name,value in out.items():
        if isinstance(value,np.ndarray):
            assert future[name].dtype==value.dtype and future[name].tobytes()==value.tobytes()
        else:assert future[name]==value
    known_open=out['open_known_at_issue']
    assert known_open.any()and not out['noise_open_eligible'][known_open].any()
    assert out['noise_close_eligible'].all()


def test_noise_margin_keeps_observed_gap_comparisons():
    starts,ohlc=source(40);keep=np.r_[np.arange(21),np.arange(22,40)]
    tape=build(starts[keep],ohlc[keep]);issue=np.array([21*target.TICK_NS],np.int64)
    out=tape.gather_next_bar_noise_margin(issue,fold_upper_ns=tape.cutoff_ns,margin_atr=.125)
    assert out['source_gap'][0,0]and out['observed_previous'][0,0]
    assert out['noise_close_known'][0,0]and out['noise_close_eligible'][0,0]


@pytest.mark.parametrize('margin',(None,True,False,0.,-1.,np.nan,np.inf,-np.inf))
def test_noise_margin_requires_explicit_finite_positive_recipe_value(tape,margin):
    with pytest.raises(ValueError,match='NOISE_MARGIN_ATR'):target.noise_margin_contract(margin)
    with pytest.raises(ValueError,match='NOISE_MARGIN_ATR'):
        tape.gather_next_bar_noise_margin(tape.issue_ns[[500]],fold_upper_ns=tape.cutoff_ns,margin_atr=margin)
    with pytest.raises(TypeError):
        tape.gather_next_bar_noise_margin(tape.issue_ns[[500]],fold_upper_ns=tape.cutoff_ns)


def test_noise_dataset_contract_binds_explicit_margin_and_all_persisted_projection_fields(tape):
    spec=target.noise_margin_dataset_contract(.3)
    other=target.noise_margin_dataset_contract(.4)
    assert spec['schema_version']=='gx1_native_m5_bar_noise_margin_dataset_v1'
    assert spec['raw_target_contract_sha256']==other['raw_target_contract_sha256']==target.contract()['contract_sha256']
    assert spec['contract_sha256']!=other['contract_sha256']
    assert spec['noise_margin_contract']==target.noise_margin_contract(.3)
    assert spec['class_order']==[-1,0,1]and spec['class_names']==['DOWN','NEUTRAL_NET','UP']
    assert spec['raw_fact_fields_preserved']and not spec['path_or_range_flat_semantics_changed']
    assert spec['target_arrays_and_metadata_as_model_input']is False
    out=tape.gather_next_bar_noise_margin(tape.issue_ns[[500]],fold_upper_ns=tape.cutoff_ns,margin_atr=.3)
    arrays={key for key,value in out.items()if isinstance(value,np.ndarray)}
    assert len(arrays)==len(spec['required_array_fields'])==24
    assert arrays==set(spec['required_array_fields'])
    assert set(out)-arrays==set(spec['required_projection_metadata_fields'])
    assert out['noise_margin_contract']==spec['noise_margin_contract']
    checked=target.require_noise_margin_dataset_contract(copy.deepcopy(spec))
    assert checked==spec and checked is not spec
    with pytest.raises(TypeError):target.noise_margin_dataset_contract()


def test_noise_dataset_task_fields_preserve_independent_unknown_masks_and_raw_signs():
    starts=np.arange(30,dtype=np.int64)*target.TICK_NS
    ohlc=np.tile(np.array([100.,101.,99.,100.]),(30,1));ohlc[3,3]+=.125
    tape=build(starts,ohlc)
    out=tape.gather_next_bar_noise_margin(tape.issue_ns[[2]],fold_upper_ns=tape.cutoff_ns,margin_atr=.3)
    tasks=target.noise_margin_dataset_contract(.3)['tasks']
    assert tasks=={task:{kind:'noise_'+task+'_'+kind for kind in ('sign','known','eligible')}
        for task in ('open','close')}
    assert out[tasks['open']['known']][0,0]and out[tasks['open']['eligible']][0,0]
    assert not out[tasks['close']['known']][0,0]and not out[tasks['close']['eligible']][0,0]
    assert out[tasks['open']['sign']][0,0]==out[tasks['close']['sign']][0,0]==0
    assert out['eligible'][0,0]and out['open_sign'][0,0]==0 and out['close_sign'][0,0]==1


def test_noise_dataset_validator_rejects_relabeling_mask_or_metadata_contract_changes():
    spec=target.noise_margin_dataset_contract(.3)
    def altered(change):
        result=copy.deepcopy(spec);change(result);return result
    changes=[
        lambda v:v['tasks']['close'].update(sign='close_sign'),
        lambda v:v['tasks']['open'].update(known='eligible'),
        lambda v:v['tasks']['close'].update(eligible='noise_open_eligible'),
        lambda v:v['class_names'].__setitem__(1,'FLAT'),
        lambda v:v['noise_margin_contract'].update(margin_atr=.4),
        lambda v:v['required_array_fields'].remove('noise_atr_asof_available_ns'),
        lambda v:v.update(raw_target_contract_sha256='0'*64),
        lambda v:v.update(raw_fact_fields_preserved=1),
        lambda v:v['class_order'].__setitem__(2,True),
        lambda v:v.update(target_arrays_and_metadata_as_model_input=True),
        lambda v:v.update(contract_sha256='0'*64)]
    for change in changes:
        with pytest.raises(ValueError,match='NOISE_DATASET_CONTRACT'):
            target.require_noise_margin_dataset_contract(altered(change))
    with pytest.raises(ValueError,match='NOISE_DATASET_CONTRACT'):
        target.require_noise_margin_dataset_contract({})


def test_fully_future_bar_selects_start_ge_issue_on_every_tf(tape):
    issues = tape.issue_ns[[0, 2, 11, 47, 287, 499]]
    out = tape.gather_fully_future_bar(issues, fold_upper_ns=tape.cutoff_ns)
    for j, tf in enumerate(target.TIMEFRAMES):
        bars = tape.bars_by_timeframe[tf]
        selected = np.searchsorted(bars.bar_start_ns, issues, side='left')
        present = selected < len(bars.bar_start_ns)
        np.testing.assert_equal(out['target_present'][:, j], present)
        np.testing.assert_equal(out['target_start_ns'][present, j], bars.bar_start_ns[selected[present]])
        assert np.all(out['target_start_ns'][present, j] >= issues[present])
        assert np.all(out['open_available_ns'][present, j] > issues[present])
        assert not out['open_known_at_issue'][:, j].any()
        mature = present & (out['validity_ns'][:, j] < tape.cutoff_ns)
        np.testing.assert_equal(out['target_ohlc'][mature, j], bars.ohlc[selected[mature]])
    # At M15 start select that bar; inside it skip to the following start.
    assert out['target_start_ns'][0, 1] == 3 * target.TICK_NS
    assert out['target_start_ns'][1, 1] == issues[1]
    old = tape.gather_next_bar(issues, fold_upper_ns=tape.cutoff_ns)
    assert old['validity_ns'][0, 1] < out['validity_ns'][0, 1]


def test_fully_future_bar_maturity_before_price_and_coverage_labels(tape):
    issue = tape.issue_ns[[0]]
    upper = 6 * target.TICK_NS
    out = tape.gather_fully_future_bar(issue, fold_upper_ns=upper)
    j = target.TIMEFRAMES.index('M15')
    assert out['target_present'][0, j] and out['validity_ns'][0, j] == upper
    assert not out['mature'][0, j] and out['target_row'][0, j] == -1
    assert np.isnan(out['target_ohlc'][0, j]).all()
    assert np.isnan(out['target_range'][0, j])
    for field in ('nominal_complete', 'observed_previous', 'eligible', 'source_gap'):
        assert not out[field][0, j]
    assert out['open_sign'][0, j] == out['close_sign'][0, j] == out['m5_count'][0, j] == 0


def test_fully_future_bar_does_not_invent_empty_bins_or_conflate_gap_with_coverage():
    starts, values = source(15)
    keep = np.array([0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    tape = build(starts[keep], values[keep])
    out = tape.gather_fully_future_bar(np.array([3 * target.TICK_NS], np.int64),
                                      fold_upper_ns=tape.cutoff_ns)
    j = target.TIMEFRAMES.index('M15')
    assert out['target_start_ns'][0, j] == 6 * target.TICK_NS
    assert out['source_gap'][0, j] and out['nominal_complete'][0, j]
    assert out['observed_previous'][0, j] and out['eligible'][0, j]
    # A missing interior member is incomplete; raw comparison remains observed.
    keep = keep[keep != 7]
    tape = build(starts[keep], values[keep])
    out = tape.gather_fully_future_bar(np.array([3 * target.TICK_NS], np.int64),
                                      fold_upper_ns=tape.cutoff_ns)
    assert not out['nominal_complete'][0, j]
    assert out['m5_count'][0, j] == 2 and out['eligible'][0, j]


def test_fully_future_bar_source_coverage_includes_boundary_members():
    starts, values = source(16)
    for missing in (3, 4, 5):
        keep = np.arange(16) != missing
        tape = build(starts[keep], values[keep])
        # Issue 5min is before target start 15min regardless of the omitted member.
        out = tape.gather_fully_future_bar(tape.issue_ns[[0]], fold_upper_ns=tape.cutoff_ns)
        assert not out['nominal_complete'][0, target.TIMEFRAMES.index('M15')]
    tape = build(starts, values)
    out = tape.gather_fully_future_bar(tape.issue_ns[[0]], fold_upper_ns=tape.cutoff_ns)
    assert out['nominal_complete'][0, target.TIMEFRAMES.index('M15')]


def test_fully_future_bar_absent_target_has_only_unknown_storage(tape):
    out = tape.gather_fully_future_bar(tape.issue_ns[[-1]], fold_upper_ns=tape.cutoff_ns)
    assert not out['target_present'].any()
    for field in ('target_row', 'target_start_ns', 'validity_ns', 'open_available_ns'):
        assert (out[field] == -1).all()
    assert np.isnan(out['target_ohlc']).all()
    assert not out['mature'].any() and not out['nominal_complete'].any()


def test_fully_future_bar_asof_atr_is_unchanged_by_future_prices():
    starts, values = source(5000)
    tape = build(starts, values)
    issue = tape.issue_ns[[4200]]
    left = tape.gather_fully_future_bar(issue, fold_upper_ns=tape.cutoff_ns)
    changed = values.copy()
    changed[4201:, 1] += 10000.
    right_tape = build(starts, changed)
    right = right_tape.gather_fully_future_bar(issue, fold_upper_ns=tape.cutoff_ns)
    assert np.isfinite(left['atr14_asof']).all()
    np.testing.assert_equal(left['atr14_asof'], right['atr14_asof'])
    np.testing.assert_equal(left['atr_asof_row'], right['atr_asof_row'])
    assert (left['atr_asof_available_ns'] <= issue[:, None]).all()
    assert (left['target_start_ns'] >= issue[:, None]).all()
    assert (left['target_range'] != right['target_range']).all()


def test_fully_future_bar_projection_append_invariance_and_independent_arrays(tape):
    issues = tape.issue_ns[[1, 287, 499]]
    longer = build(*source(1400))
    left = tape.gather_fully_future_bar(issues, fold_upper_ns=tape.cutoff_ns)
    right = longer.gather_fully_future_bar(issues, fold_upper_ns=tape.cutoff_ns)
    for name in left:
        if isinstance(left[name], np.ndarray):
            np.testing.assert_equal(left[name], right[name])
        else:
            assert left[name] == right[name]
    left['target_ohlc'][:] = 0
    fresh = tape.gather_fully_future_bar(issues, fold_upper_ns=tape.cutoff_ns)
    assert np.nanmax(fresh['target_ohlc']) > 0


def test_fully_future_state_identity_is_algebra_only_and_not_old_noise():
    spec = target.fully_future_state_contract()
    assert target.require_fully_future_state_contract(spec) == spec
    assert spec['production_training_admitted'] is False
    assert spec['local_window'].startswith('unbound_')
    changed = copy.deepcopy(spec)
    changed['side_certificate'] = 'close_sign'
    with pytest.raises(ValueError, match='FULLY_FUTURE_STATE_CONTRACT'):
        target.require_fully_future_state_contract(changed)
    with pytest.raises(ValueError, match='FULLY_FUTURE_STATE_CONTRACT'):
        target.require_fully_future_state_contract(target.noise_margin_dataset_contract(.3))


def test_fully_future_bar_rejects_outside_native_issue_axis(tape):
    with pytest.raises(ValueError, match='ISSUE_AXIS'):
        tape.gather_fully_future_bar(np.array([1], np.int64), fold_upper_ns=tape.cutoff_ns)
    with pytest.raises(ValueError, match='QUERY'):
        tape.gather_fully_future_bar(tape.issue_ns[:1], fold_upper_ns=tape.cutoff_ns + target.TICK_NS)
