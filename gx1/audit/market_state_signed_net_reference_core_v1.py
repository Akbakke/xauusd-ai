"""Pure chronological reference for signed local XAU displacement.

A fixed sign-bin table tests whether causal GC pressure adds information to
EMA distance and confirmed swing order. It is a TRAIN diagnostic, not a state
relabeling, model fit, or trading rule. Source joins and publication are owned
by the caller.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from gx1.audit.market_state_m5_gc_flow_reference_core_v1 import _auc

BINS = (-1, 0, 1)


def contract() -> dict:
    value = {
        'schema_version': 'gx1_market_state_signed_net_reference_core_v1',
        'target': 'sign(last_M1_close-minus-first_M1_open)_at_native_TF_endpoint',
        'target_zero': 'excluded_from_binary_direction_scoring_reported_separately',
        'inputs': ['causal_XAU_EMA5_distance_sign',
                   'confirmed_XAU_swing_pivot_order_sign',
                   'GC_M5_observed_aggressor_pressure_sign'],
        'fit': 'earlier_mature_TRAIN_only_plus_one_bin_rates;unseen_bin_uses_prefix_prior',
        'score': 'later_TRAIN_same_common_rows_prior_vs_price_vs_price_plus_GC_Brier',
        'raw_rank': 'GC_pressure_and_price_EMA_swing_on_same_common_rows',
        'magnitude_bins_abs_D_over_ATR': [0.0, 0.5, 1.0, 'infinity'],
        'weekly_group': 'UTC_Monday_of_issue_time',
        'threshold_selection_from_score': False,
        'TEST_or_Entry_or_trade_authority': False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def _need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise ValueError('MARKET_STATE_SIGNED_NET_' + reason)


def _check_sign(name: str, sign: np.ndarray, valid: np.ndarray, n: int) -> None:
    _need(isinstance(sign, np.ndarray) and sign.dtype == np.int8
          and sign.shape == (n,) and bool(np.isin(sign, BINS).all())
          and isinstance(valid, np.ndarray) and valid.dtype == np.bool_
          and valid.shape == (n,), name)


def _weeks(issues: np.ndarray) -> np.ndarray:
    days = issues // (24 * 60 * 60 * 1_000_000_000)
    monday = days - (days + 3) % 7
    return (np.datetime64('1970-01-01', 'D')
            + monday.astype('timedelta64[D]')).astype(str)


def _fit_table(keys: np.ndarray, up: np.ndarray) -> dict[tuple[int, ...], float]:
    counts: dict[tuple[int, ...], list[int]] = {}
    for key, label in zip(keys, up):
        cell = counts.setdefault(tuple(map(int, key)), [0, 0])
        cell[int(label)] += 1
    return {key: (cell[1] + 1) / (sum(cell) + 2) for key, cell in counts.items()}


def _predict(table: dict[tuple[int, ...], float], keys: np.ndarray,
             prior: float) -> tuple[np.ndarray, int]:
    p = np.empty(len(keys), dtype=np.float64)
    unseen = 0
    for i, key in enumerate(keys):
        cell = tuple(map(int, key))
        if cell not in table:
            unseen += 1
        p[i] = table.get(cell, prior)
    return p, unseen


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.square(p - y).sum(dtype=np.float64))


def score_fold(*, fit_target: np.ndarray, fit_target_valid: np.ndarray,
               fit_ema: np.ndarray, fit_ema_valid: np.ndarray,
               fit_swing: np.ndarray, fit_swing_valid: np.ndarray,
               fit_gc: np.ndarray, fit_gc_valid: np.ndarray,
               score_target: np.ndarray, score_target_valid: np.ndarray,
               score_magnitude_atr: np.ndarray, score_issues: np.ndarray,
               score_ema: np.ndarray, score_ema_valid: np.ndarray,
               score_swing: np.ndarray, score_swing_valid: np.ndarray,
               score_gc: np.ndarray, score_gc_valid: np.ndarray,
               score_gc_raw: np.ndarray) -> dict:
    """One prebound TF/fold; all comparisons use the exact same score rows."""
    nf, ns = len(fit_target), len(score_target)
    _need(nf > 0 and ns > 0, 'EMPTY_AXIS')
    for prefix, n, target, target_valid, ema, ema_valid, swing, swing_valid, gc, gc_valid in (
        ('FIT', nf, fit_target, fit_target_valid, fit_ema, fit_ema_valid,
         fit_swing, fit_swing_valid, fit_gc, fit_gc_valid),
        ('SCORE', ns, score_target, score_target_valid, score_ema,
         score_ema_valid, score_swing, score_swing_valid, score_gc, score_gc_valid),
    ):
        for name, sign, valid in (('TARGET', target, target_valid),
                                  ('EMA', ema, ema_valid),
                                  ('SWING', swing, swing_valid),
                                  ('GC', gc, gc_valid)):
            _check_sign(prefix + '_' + name, sign, valid, n)
    _need(isinstance(score_issues, np.ndarray) and score_issues.dtype == np.int64
          and score_issues.shape == (ns,) and bool(np.all(np.diff(score_issues) > 0))
          and bool(np.all(score_issues % 300_000_000_000 == 0))
          and isinstance(score_magnitude_atr, np.ndarray)
          and score_magnitude_atr.dtype == np.float64
          and score_magnitude_atr.shape == (ns,)
          and bool(np.isfinite(score_magnitude_atr[score_target_valid]).all())
          and bool(np.all(score_magnitude_atr[score_target_valid] >= 0))
          and isinstance(score_gc_raw, np.ndarray)
          and score_gc_raw.dtype == np.float64 and score_gc_raw.shape == (ns,)
          and bool(np.isfinite(score_gc_raw[score_gc_valid]).all())
          and bool(np.all(np.abs(score_gc_raw[score_gc_valid]) <= 1))
          and bool(np.array_equal(
              np.sign(score_gc_raw[score_gc_valid]).astype(np.int8),
              score_gc[score_gc_valid]))
          and bool(np.array_equal(score_target[score_target_valid] == 0,
                                  score_magnitude_atr[score_target_valid] == 0)),
          'SCORE_NUMERICS')
    fdir = fit_target_valid & (fit_target != 0)
    sdir = score_target_valid & (score_target != 0)
    _need(bool(fdir.any()) and bool(sdir.any()), 'DIRECTION_SUPPORT')
    prior = (int((fdir & (fit_target == 1)).sum()) + 1) / (int(fdir.sum()) + 2)
    fcommon = fdir & fit_ema_valid & fit_swing_valid & fit_gc_valid
    scommon = sdir & score_ema_valid & score_swing_valid & score_gc_valid
    _need(bool(fcommon.any()) and bool(scommon.any()), 'COMMON_SUPPORT')
    fit_price = np.column_stack((fit_ema[fcommon], fit_swing[fcommon]))
    score_price = np.column_stack((score_ema[scommon], score_swing[scommon]))
    fit_joint = np.column_stack((fit_price, fit_gc[fcommon]))
    score_joint = np.column_stack((score_price, score_gc[scommon]))
    table_price = _fit_table(fit_price, fit_target[fcommon] == 1)
    table_joint = _fit_table(fit_joint, fit_target[fcommon] == 1)
    p_price, unseen_price = _predict(table_price, score_price, prior)
    p_joint, unseen_joint = _predict(table_joint, score_joint, prior)
    y = (score_target[scommon] == 1).astype(np.float64)
    p_prior = np.full(len(y), prior)
    raw_gc = score_gc_raw[scommon]
    raw_ema = score_ema[scommon].astype(np.float64)
    raw_swing = score_swing[scommon].astype(np.float64)
    weeks = _weeks(score_issues[scommon])
    mag = score_magnitude_atr[scommon]

    def report(choose: np.ndarray) -> dict:
        label = y[choose]
        return {
            'n': int(choose.sum()), 'up': int(label.sum()),
            'prior_brier_sum': _brier(label, p_prior[choose]),
            'price_brier_sum': _brier(label, p_price[choose]),
            'price_plus_gc_brier_sum': _brier(label, p_joint[choose]),
            'gc_auc': _auc(raw_gc[choose], label.astype(np.bool_))['value'],
            'ema_auc': _auc(raw_ema[choose], label.astype(np.bool_))['value'],
            'swing_auc': _auc(raw_swing[choose], label.astype(np.bool_))['value'],
        }

    result = {
        'schema_version': contract()['schema_version'],
        'contract_sha256': contract()['contract_sha256'],
        'fit_rows': nf, 'score_rows': ns,
        'fit_target_valid': int(fit_target_valid.sum()),
        'fit_target_nonzero': int(fdir.sum()),
        'fit_common': int(fcommon.sum()),
        'score_target_valid': int(score_target_valid.sum()),
        'score_target_zero': int((score_target_valid & (score_target == 0)).sum()),
        'score_target_nonzero': int(sdir.sum()),
        'score_common': int(scommon.sum()),
        'score_missing_gc': int((sdir & ~score_gc_valid).sum()),
        'score_missing_ema': int((sdir & ~score_ema_valid).sum()),
        'score_missing_swing': int((sdir & ~score_swing_valid).sum()),
        'prefix_prior_up': prior,
        'unseen_price_bins': unseen_price,
        'unseen_joint_bins': unseen_joint,
        'overall': report(np.ones(len(y), dtype=np.bool_)),
        'by_utc_week': {str(week): report(weeks == week) for week in np.unique(weeks)},
        'by_abs_D_over_ATR': {
            'lt_0_5': report(mag < 0.5),
            'gte_0_5_lt_1_0': report((mag >= 0.5) & (mag < 1.0)),
            'gte_1_0': report(mag >= 1.0),
        },
    }
    _need(sum(x['n'] for x in result['by_utc_week'].values()) == len(y)
          and sum(x['n'] for x in result['by_abs_D_over_ATR'].values()) == len(y),
          'PARTITIONS')
    return result
