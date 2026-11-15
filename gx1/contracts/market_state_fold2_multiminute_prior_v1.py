"""Source-only fixed five-point Fold2 S prior; no fit authority.

Reuses the endpoint activity/conditional-side +1 prior exactly once for each
of the 25 predeclared query/timeframe cells. Fitting physical TRAIN labels
requires a separately admitted run.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from gx1.contracts.market_state_five_endpoint_prior_v1 import (
    fit_activity_side_prior,
)
from gx1.contracts.market_state_time_query_target_v1 import TIMEFRAMES
from gx1.contracts.market_state_fold2_multiminute_path_v1 import (
    QUERY_MINUTES, QUERY_SHA256,
)
from gx1.contracts.market_state_fold2_multiminute_side_loss_v1 import (
    FIT_INFORMATIVE, FIT_KNOWN_SIDE, FIT_ROWS, contract as loss_contract,
)

SCHEMA = 'gx1_fold2_fixed_25_cell_S_prior_v1'
FIT_CUTOFF_NS = 1735689600000000000


def _need(ok: object, code: str) -> None:
    if not bool(ok):
        raise ValueError('FOLD2_FIXED_25_CELL_PRIOR_' + code)


def _axis_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype='<i8').tobytes()).hexdigest()


def prior_grid(allowed: np.ndarray, eligible: np.ndarray,
               mature: np.ndarray, query: np.ndarray) -> dict:
    """Pure algebra for five fixed query rows; no physical data read."""
    shape = getattr(allowed, 'shape', None)
    _need(isinstance(allowed, np.ndarray) and allowed.dtype == np.uint8
          and allowed.ndim == 3
          and shape[0] > 0 and shape[1:] == QUERY_MINUTES.shape
          and all(isinstance(mask, np.ndarray) and mask.dtype == np.bool_
                  and mask.shape == shape for mask in (eligible, mature))
          and isinstance(query, np.ndarray) and query.dtype == np.int64
          and query.shape == shape
          and np.array_equal(query, np.broadcast_to(
              QUERY_MINUTES[None], shape))
          and np.all(~eligible | mature)
          and np.all(allowed[~mature] == 7), 'INPUT')
    raw = np.empty(QUERY_MINUTES.shape + (2,), dtype=np.float64)
    activity = np.empty(QUERY_MINUTES.shape, dtype=np.int64)
    known_side = np.empty_like(activity)
    counts = []
    for q in range(len(QUERY_MINUTES)):
        fitted = fit_activity_side_prior(
            allowed[:, q, :], eligible[:, q, :], mature[:, q, :])
        p = fitted['probabilities']
        active = p[:, 0] + p[:, 2]
        up = p[:, 2] / active
        raw[q, :, 0] = np.log(active / (1 - active))
        raw[q, :, 1] = np.log(up / (1 - up))
        counts.append(fitted['counts'])
        activity[q] = [fitted['counts'][tf]['activity']
                       for tf in TIMEFRAMES]
        known_side[q] = [fitted['counts'][tf]['certified_side']
                         for tf in TIMEFRAMES]
    _need(raw.shape == (5, 5, 2) and np.isfinite(raw).all(),
          'NUMERICS')
    return {'raw_logits': raw, 'activity_support': activity,
            'known_side_support': known_side, 'counts': counts,
            'smoothing': 'laplace_plus_one_separate_binary_activity_and_conditional_side'}


def fit_exact_fold2_prior(path: dict, parent_rows: np.ndarray,
                          *, expected_parent_rows_sha256: str,
                          expected_issue_ns_sha256: str) -> dict:
    """Bind actual Fold2 earlier-fit population and all 25 supports.

    This function is source-only until a separate exact fit admission calls it.
    """
    _need(isinstance(path, dict) and path.get('query_sha256') == QUERY_SHA256
          and isinstance(parent_rows, np.ndarray)
          and parent_rows.dtype == np.int64
          and parent_rows.shape == (FIT_ROWS,)
          and len(np.unique(parent_rows)) == FIT_ROWS
          and _axis_sha(parent_rows) == expected_parent_rows_sha256
          and isinstance(path.get('issue_ns'), np.ndarray)
          and path['issue_ns'].dtype == np.int64
          and path['issue_ns'].shape == (FIT_ROWS,)
          and bool(np.all(np.diff(path['issue_ns']) > 0))
          and bool(np.all(path['issue_ns'] < FIT_CUTOFF_NS))
          and _axis_sha(path['issue_ns']) == expected_issue_ns_sha256,
          'FIT_AXIS')
    fitted = prior_grid(
        path['allowed'], path['eligible'], path['mature'],
        path['query_minutes'])
    _need(np.array_equal(fitted['activity_support'], FIT_INFORMATIVE)
          and np.array_equal(fitted['known_side_support'], FIT_KNOWN_SIDE),
          'EXACT_SUPPORT')
    value = {
        'schema_version': SCHEMA, 'fold_id': 2, 'partition': 'fit',
        'fit_rows': FIT_ROWS, 'query_sha256': QUERY_SHA256,
        'parent_rows_sha256': expected_parent_rows_sha256,
        'issue_ns_sha256': expected_issue_ns_sha256,
        'loss_contract_sha256': loss_contract()['contract_sha256'],
        'raw_logits': fitted['raw_logits'].tolist(),
        'activity_support': fitted['activity_support'].tolist(),
        'known_side_support': fitted['known_side_support'].tolist(),
        'counts': fitted['counts'], 'smoothing': fitted['smoothing'],
        'test_data_used': False,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(encoded).hexdigest()}


def require_exact_fold2_prior(value: dict, *, parent_rows_sha256: str,
                              issue_ns_sha256: str) -> np.ndarray:
    _need(isinstance(value, dict) and value.get('schema_version') == SCHEMA
          and value.get('fold_id') == 2 and value.get('partition') == 'fit'
          and value.get('fit_rows') == FIT_ROWS
          and value.get('query_sha256') == QUERY_SHA256
          and value.get('parent_rows_sha256') == parent_rows_sha256
          and value.get('issue_ns_sha256') == issue_ns_sha256
          and value.get('loss_contract_sha256')
              == loss_contract()['contract_sha256']
          and value.get('activity_support') == [list(x) for x in FIT_INFORMATIVE]
          and value.get('known_side_support') == [list(x) for x in FIT_KNOWN_SIDE]
          and value.get('test_data_used') is False,
          'BINDING')
    body = {k: v for k, v in value.items() if k != 'contract_sha256'}
    encoded = json.dumps(body, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode()
    _need(hashlib.sha256(encoded).hexdigest() == value.get('contract_sha256'),
          'SEAL')
    raw = np.asarray(value['raw_logits'], dtype=np.float64)
    _need(raw.shape == (5, 5, 2) and np.isfinite(raw).all()
          and value.get('smoothing')
              == 'laplace_plus_one_separate_binary_activity_and_conditional_side'
          and isinstance(value.get('counts'), list)
          and len(value['counts']) == len(QUERY_MINUTES), 'LOGITS')
    for q, row in enumerate(value['counts']):
        _need(isinstance(row, dict) and set(row) == set(TIMEFRAMES),
              'COUNTS')
        for j, tf in enumerate(TIMEFRAMES):
            count = row[tf]
            _need(isinstance(count, dict)
                  and set(count) == {'activity', 'active', 'certified_side', 'up'}
                  and all(type(v) is int and v >= 0 for v in count.values())
                  and 0 <= count['up'] <= count['certified_side']
                  <= count['active'] <= count['activity']
                  and count['activity'] == FIT_INFORMATIVE[q][j]
                  and count['certified_side'] == FIT_KNOWN_SIDE[q][j],
                  'COUNT_SUPPORT')
            p_active = (count['active'] + 1) / (count['activity'] + 2)
            p_up = (count['up'] + 1) / (count['certified_side'] + 2)
            expected = (np.log(p_active / (1 - p_active)),
                        np.log(p_up / (1 - p_up)))
            _need(np.allclose(raw[q, j], expected, rtol=0, atol=1e-12),
                  'COUNT_LOGITS')
    return raw
