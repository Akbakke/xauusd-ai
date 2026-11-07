"""Synthetic fixed GC-flow rank metrics; no market data or model call."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from gx1.audit.market_state_m5_gc_flow_reference_core_v1 import (
    gc_flow_reference_contract, score_fold,
)
from gx1.contracts.market_state_five_endpoint_target_binding_v1 import FOLD_BOUNDARIES_NS
from gx1.contracts.market_state_time_query_target_v1 import TICK_NS


def _case():
    start = FOLD_BOUNDARIES_NS[0]
    issue = np.array([start + 24 * 60 * 60 * 1_000_000_000 + i * TICK_NS
                      for i in range(4)] +
                     [start + 8 * 24 * 60 * 60 * 1_000_000_000 + i * TICK_NS
                      for i in range(4)], dtype=np.int64)
    allowed = np.array([4, 1, 4, 1, 5, 2, 7, 4], dtype=np.uint8)
    return dict(fold_id=1, issue_ns=issue, allowed=allowed,
                eligible=np.ones(8, dtype=np.bool_), mature=np.ones(8, dtype=np.bool_),
                buy=np.array([9, 1, 5, 5, 10, 0, 0, 0], dtype=np.float32),
                sell=np.array([1, 9, 5, 5, 0, 0, 0, 0], dtype=np.float32),
                unknown=np.zeros(8, dtype=np.float32),
                buy_observed=np.ones(8, dtype=np.bool_),
                sell_observed=np.ones(8, dtype=np.bool_),
                unknown_observed=np.ones(8, dtype=np.bool_),
                ofi=np.array([2, -2, 0, 0, 5, 0, 0, np.nan], dtype=np.float32),
                ofi_observed=np.array([True] * 7 + [False], dtype=np.bool_))


def test_tie_aware_auc_sign_bins_unknown_side_and_week_counts():
    case = _case()
    report = score_fold(**case)
    assert report['contract_sha256'] == gc_flow_reference_contract()['contract_sha256']
    assert report['issue_axis'] == {'count': 8,
        'sha256': hashlib.sha256(case['issue_ns'].tobytes()).hexdigest()}
    overall = report['overall']
    assert overall['issues'] == 8 and overall['certified_side'] == 5
    assert overall['active_side_unknown'] == 1
    assert (overall['up'], overall['down']) == (3, 2)
    for name in ('trade_pressure', 'book_ofi'):
        got = overall['signals'][name]
        assert got['side_rows'] == 5 and got['observed_rows'] == 4
        assert got['missing_rows'] == 1 and got['missing_up'] == 1 and got['missing_down'] == 0
        assert got['zero_score_rows'] == 2
        assert got['sign_bins'] == {
            'negative': {'count': 1, 'up': 0, 'down': 1, 'up_rate': 0.0},
            'zero': {'count': 2, 'up': 1, 'down': 1, 'up_rate': .5},
            'positive': {'count': 1, 'up': 1, 'down': 0, 'up_rate': 1.0},
        }
        assert got['auc'] == {'value': .875, 'favorable_pairs': 3,
                              'opposite_tie_pairs': 1, 'opposite_pairs': 4,
                              'distinct_scores': 3}
    assert overall['signals']['trade_pressure']['zero_volume_rows'] == 1
    assert 'zero_volume_rows' not in overall['signals']['book_ofi']
    weeks = report['by_utc_week']
    assert len(weeks) == 2
    assert sum(w['issues'] for w in weeks.values()) == 8
    assert sum(w['certified_side'] for w in weeks.values()) == 5
    later = next(w for w in weeks.values() if w['issues'] == 4 and w['certified_side'] == 1)
    assert later['signals']['trade_pressure']['auc']['value'] is None
    assert later['signals']['trade_pressure']['observed_rows'] == 0


def test_unknown_aggressor_attenuates_pressure_without_filling_direction():
    case = _case()
    case['allowed'][:] = 7
    case['allowed'][:2] = [4, 1]
    case['buy'][:2] = [9, 6]
    case['sell'][:2] = [1, 4]
    case['unknown'][:2] = [100, 0]
    report = score_fold(**case)['overall']['signals']['trade_pressure']
    # UP delta=8 exceeds DOWN delta=2, but unknown volume makes its fraction lower.
    assert report['observed_rows'] == 2
    assert report['auc']['value'] == 0.0
    assert report['sign_bins']['positive']['count'] == 2


def test_missing_and_zero_are_distinct_and_single_class_auc_is_unknown():
    case = _case()
    case['eligible'][:] = False
    case['eligible'][0] = True
    case['sell_observed'][0] = False
    case['ofi'][0] = 0
    report = score_fold(**case)['overall']
    assert report['certified_side'] == 1
    assert report['signals']['trade_pressure']['missing_rows'] == 1
    assert report['signals']['trade_pressure']['sign_bins']['zero']['count'] == 0
    assert report['signals']['book_ofi']['observed_rows'] == 1
    assert report['signals']['book_ofi']['zero_score_rows'] == 1
    assert report['signals']['book_ofi']['auc']['value'] is None


@pytest.mark.parametrize('change,reason', [
    (lambda c: c.update(issue_ns=c['issue_ns'][::-1]), 'ISSUE_AXIS'),
    (lambda c: c.update(allowed=c['allowed'].astype(np.int64)), 'TARGET_BITS'),
    (lambda c: c.update(eligible=c['eligible'].astype(np.uint8)), 'MASK_ELIGIBLE'),
    (lambda c: c['buy'].__setitem__(0, -1), 'CONTRACTS_BUY'),
    (lambda c: c['sell'].__setitem__(0, 1.5), 'CONTRACTS_SELL'),
    (lambda c: c['unknown'].__setitem__(0, np.inf), 'CONTRACTS_UNKNOWN'),
    (lambda c: c['ofi'].__setitem__(0, np.nan), 'BOOK_OFI'),
])
def test_invalid_axes_masks_or_source_values_fail_closed(change, reason):
    case = _case()
    change(case)
    with pytest.raises(ValueError, match=reason):
        score_fold(**case)
