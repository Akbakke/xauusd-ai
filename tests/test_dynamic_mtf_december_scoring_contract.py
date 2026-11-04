import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from gx1.contracts import entry_dynamic_mtf_state_report_v1 as report
from gx1.contracts.entry_gc_mtf_history_v1 import WINDOW_ROWS
from gx1.features import htf_features as htf


def _clock_case():
    minute = report.MINUTE_NS
    hour = 60 * minute
    start = pd.Timestamp('2024-07-01', tz='UTC').value
    # A missing H1 opening on the theoretical grid does not erase the last physical close.
    opens = start + np.delete(np.arange(103, dtype=np.int64), 100) * hour
    issues = start + np.array([101 * 60, 101 * 60 + 5, 102 * 60, 102 * 60 + 5, 103 * 60]) * minute
    columns = htf.MULTI_TF_PER_BAR_FEATURES_V4
    values = np.zeros((len(opens), len(columns)), np.float32)
    candle = columns.index('close_open_atr')
    values[-3:, candle] = (3., 1., 8.)
    frame = pd.DataFrame(values, columns=columns, index=pd.DatetimeIndex(opens, tz='UTC'), copy=False)
    frame.attrs.update(htf_feature_contract=htf.HTF_V4_MATRIX_CONTRACT, ts_int64=opens, feats_np=values, causal_warmup_rows=2)
    indices, scores = [], []
    for issue in issues:
        window, ids = htf.slice_multi_tf_v4_window(frame, pd.Timestamp(issue, tz='UTC'), WINDOW_ROWS['H1'],
                                                   htf.MULTI_TF_SHIFT['H1'], return_row_indices=True)
        indices.append(ids[-1])
        scores.append(window[-1, candle])
    np.testing.assert_array_equal(opens[indices], start + np.array([99, 99, 101, 101, 102]) * hour)
    return opens, issues, np.asarray(indices), np.asarray(scores, dtype=np.float64)


def test_oracle_on_physical_native_windows_and_exact_pair_ties():
    data = _clock_case()
    for labels, expected_orientation in (([0, 1, 1, 0, -1], 1), ([0, 0, 1, 1, -1], -1), ([0, 0, 0, 0, -1], None)):
        lab = np.array(labels, np.int8)
        result = report.verify_prefix_candle_rank_and_clock(*data, lab, lab != -1, timeframe='H1', warmup_rows=2)
        scores = data[-1]
        wins = sum(2 * int(a > b) + int(a == b) for a in scores[lab == 1] for b in scores[lab == 0])
        assert result['twice_wins'] == wins and result['orientation'] == expected_orientation
        assert result['eligible_rows'] == 4
        assert result['pairs'] == result['up'] * result['down']
        if expected_orientation is None:
            assert result['raw_auc'] is result['oriented_auc'] is None
        else:
            raw = wins / (2 * result['pairs'])
            assert result['raw_auc'] == raw and result['oriented_auc'] == max(raw, 1 - raw)


def test_oracle_rejects_stale_tail_and_strictly_immature_fit_target():
    opens, issues, ids, scores = _clock_case()
    labels = np.array([0, 1, 1, 0, 0], np.int8)
    with pytest.raises(RuntimeError, match='CANDLE_ORACLE_WINDOW_CLOCK'):
        report.verify_prefix_candle_rank_and_clock(opens, issues, ids - 1, scores, labels, labels >= 0, timeframe='H1', warmup_rows=2)
    shift = pd.Timestamp('2024-12-01', tz='UTC').value - 60 * report.MINUTE_NS - issues[-1]
    with pytest.raises(RuntimeError, match='CANDLE_ORACLE_STRICT_FIT'):
        report.verify_prefix_candle_rank_and_clock(opens + shift, issues + shift, ids, scores, labels,
                                                  labels >= 0, timeframe='H1', warmup_rows=2)


def test_candidate_measurement_specification_is_sealed_and_data_free():
    contract = report.december_oof_scoring_contract()
    digest = contract.pop('contract_sha256')
    assert digest == hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert contract['fit_parents_half_open'] == [0, 41866] and contract['oof_parents_half_open'] == [41866, 47602]
    assert list(contract['native_lead_minutes'].values()) == [5, 15, 60, 240, 1440]
    assert contract['paired_uncertainty']['draws'] == 10000 and contract['paired_uncertainty']['seed'] == 20260911
    assert contract['paired_uncertainty']['quantiles'] == [.025, .975]
    assert contract['origins'] == contract['prediction_sets'] == 1 and contract['refits_in_oof'] == 0
    contract['native_lead_minutes'].clear()
    assert report.december_oof_scoring_contract()['contract_sha256'] == digest


def test_registered_exit_and_return_phase_count_three_step_paths_from_issue():
    from itertools import product

    rule = report.december_oof_scoring_contract()['event_forecasts_at_issue']
    # FLAT start. DOWN and UP have different probabilities of returning to FLAT.
    first = np.array([.60, .25, .15])
    transition = np.array([[.2, .3, .5], [.1, .8, .1], [.5, .4, .1]])
    totals = [[1., 0., 0.]]
    for length in (1, 2, 3):
        phases = np.zeros(3)
        for path in product(range(3), repeat=length):
            prob = first[path[0]]
            phase = 0
            for k, state in enumerate(path):
                table = ('next_phase_if_next_state_is_start' if state == 1
                         else 'next_phase_if_next_state_differs')
                phase = rule[table][phase]
                if k:
                    prob *= transition[path[k - 1], state]
            phases[phase] += prob
        totals.append(phases)
    np.testing.assert_allclose(totals, [[1, 0, 0], [.25, .75, 0], [.20, .56, .24], [.16, .398, .442]])
    a, _, r = np.asarray(totals).T
    assert (a[2] - a[3]) / a[2] == pytest.approx(.2)
    assert (r[3] - r[2]) / (1 - r[2]) == pytest.approx(.202 / .76)
