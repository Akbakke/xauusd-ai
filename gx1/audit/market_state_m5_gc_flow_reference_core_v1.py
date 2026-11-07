"""Pure no-fit M5 GC flow reference on certified later-TRAIN market state.

The caller owns the causal M5 source join, physical target binding and provenance.
This module reads no files and never maps a raw flow score to a probability.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from gx1.contracts.market_state_five_endpoint_target_binding_v1 import FOLD_BOUNDARIES_NS
from gx1.contracts.market_state_time_query_target_v1 import TICK_NS, VALID_BITS

SIGN_NAMES = ('negative', 'zero', 'positive')
FLOAT_DTYPES = (np.dtype('float32'), np.dtype('float64'))


def gc_flow_reference_contract() -> dict:
    value = {
        'schema_version': 'gx1_market_state_m5_gc_flow_reference_core_v1',
        'role': 'TRAIN_only_no_fit_no_entry_exit_test_or_trade',
        'target': 'M5_k1_certified_DOWN_bit1_or_UP_bit4',
        'target_masks': 'eligible_and_mature;_bit5_activity_unknown_side_not_scored',
        'primary_score': '(observed_buy_known-observed_sell_known)/(observed_buy_known+observed_sell_known+observed_unknown)',
        'secondary_score': 'observed_book_ofi_contracts',
        'orientation': 'positive_score_predicts_UP_for_both_fixed_signals',
        'pressure_valid': 'all_three_trade_fields_observed_finite_nonnegative_and_positive_total_volume',
        'ofi_valid': 'own_book_flow_mask_and_finite_value',
        'zero': 'observed_zero_is_separate_sign_bin',
        'missing': 'never_zero_filled_or_assigned_a_side;_report_missing_UP_DOWN',
        'auc': 'Mann_Whitney_tie_pairs_half_credit;_null_if_one_side_absent',
        'aggregation': 'overall_fold_and_UTC_issue_week;_one_M5_k1_query_per_issue',
        'probability_mapping_or_fitted_threshold': False,
        'full_grid_or_other_TF_claim': False,
        'test_data_used': False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('MARKET_STATE_M5_GC_FLOW_' + reason)


def _weeks(issues: np.ndarray) -> np.ndarray:
    days = issues // (24 * 60 * 60 * 1_000_000_000)
    monday = days - (days + 3) % 7  # Epoch day zero was Thursday.
    return (np.datetime64('1970-01-01', 'D')
            + monday.astype('timedelta64[D]')).astype(str)


def _auc(scores: np.ndarray, up: np.ndarray) -> dict:
    n_up = int(up.sum())
    n_down = len(up) - n_up
    if not n_up or not n_down:
        return {'value': None, 'favorable_pairs': 0, 'opposite_tie_pairs': 0,
                'opposite_pairs': n_up * n_down, 'distinct_scores': int(len(np.unique(scores)))}
    order = np.argsort(scores, kind='mergesort')
    ordered = scores[order]
    labels = up[order].astype(np.int64)
    starts = np.r_[0, 1 + np.flatnonzero(ordered[1:] != ordered[:-1])]
    counts = np.diff(np.r_[starts, len(ordered)])
    up_per_value = np.add.reduceat(labels, starts)
    down_per_value = counts - up_per_value
    down_below = np.cumsum(down_per_value) - down_per_value
    favorable = int(np.sum(up_per_value * down_below, dtype=np.int64))
    ties = int(np.sum(up_per_value * down_per_value, dtype=np.int64))
    return {'value': (2 * favorable + ties) / (2 * n_up * n_down),
            'favorable_pairs': favorable, 'opposite_tie_pairs': ties,
            'opposite_pairs': n_up * n_down, 'distinct_scores': int(len(starts))}


def _signal(scores: np.ndarray, present: np.ndarray, side: np.ndarray,
            up: np.ndarray, *, zero_volume: np.ndarray | None = None) -> dict:
    selected = side & present
    values = scores[selected]
    labels = up[selected]
    bins = {}
    for name, test in zip(SIGN_NAMES, (values < 0, values == 0, values > 0)):
        count = int(test.sum())
        ups = int(labels[test].sum())
        bins[name] = {'count': count, 'up': ups, 'down': count - ups,
                      'up_rate': ups / count if count else None}
    result = {'side_rows': int(side.sum()), 'observed_rows': int(selected.sum()),
              'missing_rows': int((side & ~present).sum()),
              'missing_up': int((side & ~present & up).sum()),
              'missing_down': int((side & ~present & ~up).sum()),
              'up': int(labels.sum()), 'down': int(len(labels) - labels.sum()),
              'zero_score_rows': bins['zero']['count'], 'sign_bins': bins,
              'auc': _auc(values, labels)}
    if zero_volume is not None:
        result['zero_volume_rows'] = int((side & zero_volume).sum())
    return result


def score_fold(*, fold_id: int, issue_ns: np.ndarray, allowed: np.ndarray,
               eligible: np.ndarray, mature: np.ndarray,
               buy: np.ndarray, sell: np.ndarray, unknown: np.ndarray,
               buy_observed: np.ndarray, sell_observed: np.ndarray,
               unknown_observed: np.ndarray, ofi: np.ndarray,
               ofi_observed: np.ndarray) -> dict:
    """Score two fixed causal M5 proxies on one later-TRAIN fold axis."""
    _need(type(fold_id) is int and 1 <= fold_id <= 5, 'FOLD_ID')
    _need(isinstance(issue_ns, np.ndarray) and issue_ns.dtype == np.int64
          and issue_ns.ndim == 1 and len(issue_ns) > 0
          and bool(np.all(np.diff(issue_ns) > 0))
          and bool(np.all(issue_ns % TICK_NS == 0))
          and bool(np.all((issue_ns >= FOLD_BOUNDARIES_NS[fold_id - 1])
                          & (issue_ns < FOLD_BOUNDARIES_NS[fold_id]))), 'ISSUE_AXIS')
    n = len(issue_ns)
    _need(isinstance(allowed, np.ndarray) and allowed.dtype == np.uint8
          and allowed.shape == (n,) and bool(np.isin(allowed, VALID_BITS).all()),
          'TARGET_BITS')
    for name, mask in (('ELIGIBLE', eligible), ('MATURE', mature),
                       ('BUY', buy_observed), ('SELL', sell_observed),
                       ('UNKNOWN', unknown_observed), ('OFI', ofi_observed)):
        _need(isinstance(mask, np.ndarray) and mask.dtype == np.bool_
              and mask.shape == (n,), 'MASK_' + name)
    for name, value, mask in (('BUY', buy, buy_observed),
                              ('SELL', sell, sell_observed),
                              ('UNKNOWN', unknown, unknown_observed)):
        _need(isinstance(value, np.ndarray) and value.dtype in FLOAT_DTYPES
              and value.shape == (n,) and bool(np.isfinite(value[mask]).all())
              and bool(np.all(value[mask] >= 0))
              and bool(np.all(value[mask] == np.floor(value[mask]))),
              'CONTRACTS_' + name)
    _need(isinstance(ofi, np.ndarray) and ofi.dtype in FLOAT_DTYPES
          and ofi.shape == (n,) and bool(np.isfinite(ofi[ofi_observed]).all()),
          'BOOK_OFI')
    all_trade = buy_observed & sell_observed & unknown_observed
    total = np.zeros(n, dtype=np.float64)
    total[all_trade] = (buy[all_trade].astype(np.float64)
                        + sell[all_trade].astype(np.float64)
                        + unknown[all_trade].astype(np.float64))
    _need(bool(np.isfinite(total[all_trade]).all()), 'TRADE_TOTAL')
    zero_volume = all_trade & (total == 0)
    pressure_valid = all_trade & (total > 0)
    pressure = np.full(n, np.nan, dtype=np.float64)
    pressure[pressure_valid] = ((buy[pressure_valid].astype(np.float64)
                                - sell[pressure_valid].astype(np.float64))
                               / total[pressure_valid])
    _need(bool(np.isfinite(pressure[pressure_valid]).all())
          and bool(np.all(np.abs(pressure[pressure_valid]) <= 1)), 'PRESSURE_DOMAIN')
    book = np.full(n, np.nan, dtype=np.float64)
    book[ofi_observed] = ofi[ofi_observed].astype(np.float64)
    side = eligible & mature & np.isin(allowed, (1, 4))
    up = allowed == 4
    weeks = _weeks(issue_ns)

    def report(selected: np.ndarray) -> dict:
        scoped_side = side & selected
        return {'issues': int(selected.sum()), 'certified_side': int(scoped_side.sum()),
                'active_side_unknown': int((selected & eligible & mature & (allowed == 5)).sum()),
                'up': int((scoped_side & up).sum()),
                'down': int((scoped_side & ~up).sum()),
                'signals': {
                    'trade_pressure': _signal(pressure, pressure_valid, scoped_side, up,
                                              zero_volume=zero_volume),
                    'book_ofi': _signal(book, ofi_observed, scoped_side, up),
                }}

    return {'schema_version': gc_flow_reference_contract()['schema_version'],
            'contract_sha256': gc_flow_reference_contract()['contract_sha256'],
            'scope': 'later_TRAIN_M5_k1_no_fit_no_TEST', 'fold_id': fold_id,
            'issue_axis': {'count': n, 'sha256': hashlib.sha256(issue_ns.tobytes()).hexdigest()},
            'overall': report(np.ones(n, dtype=np.bool_)),
            'by_utc_week': {str(week): report(weeks == week) for week in np.unique(weeks)}}
