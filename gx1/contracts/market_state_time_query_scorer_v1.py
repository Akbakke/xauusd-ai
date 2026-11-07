"""Streaming matched TRAIN scoring for future M5-tick market-state queries.

This consumes already bound target masks and frozen priors. Query cells sharing an
issue are reduced within issue and timeframe before any week/fold comparison.
No target load, model fit, TEST access or trading authority lives here.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from gx1.contracts.market_state_five_endpoint_contract_v1 import CLASS_NAMES
from gx1.contracts.market_state_five_endpoint_target_binding_v1 import FOLD_BOUNDARIES_NS
from gx1.contracts.market_state_time_query_loss_v1 import time_query_loss_contract
from gx1.contracts.market_state_time_query_target_v1 import (
    NATIVE_K, TICK_NS, TIMEFRAMES, VALID_BITS,
)

_METRICS = ('activity', 'conditional_side', 'three_class')
_K = tuple(int(value) for value in NATIVE_K)


def time_query_scorer_contract() -> dict:
    value = {
        'schema_version': 'gx1_market_state_time_query_scorer_v1',
        'role': 'research_only_no_entry_exit_or_trade_authority',
        'loss_contract_sha256': time_query_loss_contract()['contract_sha256'],
        'timeframes': list(TIMEFRAMES), 'native_k': list(_K),
        'class_order': list(CLASS_NAMES),
        'matched_reference': 'separate_frozen_activity_conditional_side_three_class_priors',
        'population': 'same_mature_eligible_allowed_cell_for_model_and_prior',
        'reduction': 'query_Brier_sums_to_issue_TF_mean_then_UTC_issue_week_and_fold',
        'primary': 'M5_k1_certified_side_Brier_prior_minus_model',
        'unknown': 'bit5_activity_only_bit7_unscored',
        'full_grid': 'all_k_1_through_native_k_submitted_once_per_issue_TF',
        'clock': 'query_k_equals_min_step_native_k_and_validity_equals_issue_plus_5k',
        'inactive_TF': 'capped_query_is_decoded_but_never_counted',
        'max_query_block': 32,
        'test_data_used': False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def _require(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError('TIME_QUERY_SCORER_' + code)


def _weeks(issues: np.ndarray) -> np.ndarray:
    days = issues // (24 * 60 * 60 * 1_000_000_000)
    monday = days - (days + 3) % 7  # Epoch day zero was Thursday.
    return (np.datetime64('1970-01-01', 'D')
            + monday.astype('timedelta64[D]')).astype(str)


def _bucket() -> dict:
    return {metric: {'issue_count': 0, 'issue_delta_sum': 0.0,
                     'cell_count': 0, 'model_brier_sum': 0.0,
                     'prior_brier_sum': 0.0} for metric in _METRICS}


def _coverage() -> dict:
    return {'issues': 0, 'mature': 0, 'activity_known': 0,
            'active_side_unknown': 0, 'side_known': 0, 'singleton': 0,
            **{metric + '_model_brier_sum': 0.0 for metric in _METRICS},
            **{metric + '_prior_brier_sum': 0.0 for metric in _METRICS}}


class TimeQueryMatchedScorer:
    """Consume one bounded issue batch at a time, in contiguous k blocks.

    Each add_block array has shape B,Q,5 (probabilities have final class axis 3).
    An explicit conditional-side probability is accepted for a factorized
    successor; the original probability-ratio path remains the default.
    The scorer ignores inactive TF slots after their native K even if a capped
    tape gather returns cached labels. The tape owner binds source provenance.
    """

    def __init__(self):
        self._active = None
        self._last_issue_by_fold: dict[int, int] = {}
        self._axis_hash = {fold: hashlib.sha256() for fold in range(1, 6)}
        self._axis_count = {fold: 0 for fold in range(1, 6)}
        self._coverage = {tf: {str(k): _coverage() for k in range(1, _K[j] + 1)}
                          for j, tf in enumerate(TIMEFRAMES)}
        self._weekly: dict[tuple[int, str, str], dict] = {}
        self._folds: dict[tuple[int, str], dict] = {}

    def begin_batch(self, issue_ns: np.ndarray, *, fold_id: int) -> None:
        _require(self._active is None, 'BATCH_ALREADY_OPEN')
        _require(type(fold_id) is int and 1 <= fold_id <= 5, 'FOLD_ID')
        lower, upper = FOLD_BOUNDARIES_NS[fold_id - 1:fold_id + 1]
        _require(isinstance(issue_ns, np.ndarray) and issue_ns.dtype == np.int64
                 and issue_ns.ndim == 1 and len(issue_ns) > 0
                 and bool(np.all(np.diff(issue_ns) > 0))
                 and bool(np.all(issue_ns % TICK_NS == 0))
                 and bool(np.all((issue_ns >= lower) & (issue_ns < upper)))
                 and int(issue_ns[0]) > self._last_issue_by_fold.get(fold_id, -1),
                 'ISSUE_AXIS')
        batch = len(issue_ns)
        self._active = {'issues': issue_ns.copy(), 'fold': fold_id,
                        'weeks': _weeks(issue_ns), 'next_step': 1,
                        'count': np.zeros((batch, 5, 3), dtype=np.int64),
                        'model_sum': np.zeros((batch, 5, 3), dtype=np.float64),
                        'prior_sum': np.zeros((batch, 5, 3), dtype=np.float64),
                        'coverage': {tf: {str(k): _coverage()
                                          for k in range(1, _K[j] + 1)}
                                     for j, tf in enumerate(TIMEFRAMES)}}

    def add_block(self, step_start: int, *, query_k: np.ndarray,
                  validity_ns: np.ndarray, model_p: np.ndarray,
                  prior_activity: np.ndarray, prior_side_up: np.ndarray,
                  prior_three: np.ndarray, allowed: np.ndarray,
                  eligible: np.ndarray, mature: np.ndarray,
                  model_side_up: np.ndarray | None = None) -> None:
        batch = self._active
        _require(batch is not None, 'NO_OPEN_BATCH')
        _require(type(step_start) is int and step_start == batch['next_step'],
                 'STEP_ORDER')
        b = len(batch['issues'])
        _require(isinstance(allowed, np.ndarray) and allowed.dtype == np.uint8
                 and allowed.ndim == 3 and allowed.shape[0] == b
                 and allowed.shape[2] == 5 and 1 <= allowed.shape[1] <= 32
                 and step_start + allowed.shape[1] - 1 <= 288
                 and bool(np.isin(allowed, VALID_BITS).all()), 'ALLOWED')
        q = allowed.shape[1]
        shape = (b, q, 5)
        steps = np.arange(step_start, step_start + q, dtype=np.int64)
        expected_k = np.minimum(steps[:, None], np.asarray(_K, dtype=np.int64)[None, :])
        _require(isinstance(query_k, np.ndarray) and query_k.dtype == np.int64
                 and query_k.shape == shape
                 and bool(np.array_equal(query_k,
                       np.broadcast_to(expected_k[None, :, :], shape))), 'QUERY_K')
        expected_validity = batch['issues'][:, None, None] + query_k * TICK_NS
        _require(isinstance(validity_ns, np.ndarray) and validity_ns.dtype == np.int64
                 and validity_ns.shape == shape
                 and bool(np.array_equal(validity_ns, expected_validity)), 'VALIDITY_CLOCK')
        upper = FOLD_BOUNDARIES_NS[batch['fold']]
        _require(all(isinstance(mask, np.ndarray) and mask.dtype == np.bool_
                     and mask.shape == shape for mask in (eligible, mature))
                 and bool(np.array_equal(mature, validity_ns < upper))
                 and bool(np.array_equal(eligible, mature & (allowed != 7))),
                 'MASKS')
        for name, value, expected in (
            ('MODEL', model_p, shape + (3,)),
            ('PRIOR_ACTIVITY', prior_activity, shape),
            ('PRIOR_SIDE', prior_side_up, shape),
            ('PRIOR_THREE', prior_three, shape + (3,)),
        ):
            _require(isinstance(value, np.ndarray)
                     and value.dtype in (np.dtype('float32'), np.dtype('float64'))
                     and value.shape == expected and bool(np.isfinite(value).all())
                     and bool(np.all((value >= 0) & (value <= 1))), name)
        if model_side_up is not None:
            _require(isinstance(model_side_up, np.ndarray)
                     and model_side_up.dtype in (np.dtype('float32'), np.dtype('float64'))
                     and model_side_up.shape == shape
                     and bool(np.isfinite(model_side_up).all())
                     and bool(np.all((model_side_up >= 0) & (model_side_up <= 1))),
                     'MODEL_SIDE')
        _require(bool(np.allclose(model_p.sum(axis=-1, dtype=np.float64), 1,
                                  rtol=0, atol=1e-6))
                 and bool(np.allclose(prior_three.sum(axis=-1, dtype=np.float64), 1,
                                      rtol=0, atol=1e-6)), 'PROBABILITY_SUM')
        # Validate the whole block before any score accumulator is mutated.
        included = np.broadcast_to((steps[:, None] <= np.asarray(_K)[None, :])
                                   [None, :, :], shape)
        side_cells = eligible & mature & included & np.isin(allowed, (1, 4))
        active_mass = model_p[..., 0] + model_p[..., 2]
        if model_side_up is None:
            _require(bool(np.all(active_mass[side_cells] > 0)),
                     'UNDEFINED_CONDITIONAL_SIDE')
        else:
            representable = side_cells & (active_mass > 0)
            if representable.any():
                ratio = (model_p[..., 2][representable].astype(np.float64)
                         / active_mass[representable].astype(np.float64))
                supplied = model_side_up[representable].astype(np.float64)
                eps = max(np.finfo(model_p.dtype).eps,
                          np.finfo(model_side_up.dtype).eps)
                # Near the subnormal limit the class masses lose side precision.
                underflow = (2 * np.finfo(model_p.dtype).smallest_subnormal
                             / active_mass[representable].astype(np.float64))
                _require(bool(np.all(np.abs(ratio - supplied) <= 8 * eps + underflow)),
                         'MODEL_SIDE_PARITY')
        for offset in range(q):
            step = step_start + offset
            for j, tf in enumerate(TIMEFRAMES):
                if step > _K[j]:
                    continue
                bits = allowed[:, offset, j]
                ready = eligible[:, offset, j] & mature[:, offset, j]
                side = ready & np.isin(bits, (1, 4))
                singleton = ready & np.isin(bits, (1, 2, 4))
                cov = batch['coverage'][tf][str(step)]
                cov['issues'] += b
                cov['mature'] += int(mature[:, offset, j].sum())
                cov['activity_known'] += int(ready.sum())
                cov['active_side_unknown'] += int((ready & (bits == 5)).sum())
                cov['side_known'] += int(side.sum())
                cov['singleton'] += int(singleton.sum())
                if ready.any():
                    y = (bits[ready] != 2).astype(np.float64)
                    mp = model_p[ready, offset, j, 0] + model_p[ready, offset, j, 2]
                    pp = prior_activity[ready, offset, j]
                    self._add_scores(batch, ready, j, 0, np.square(mp - y),
                                     np.square(pp - y), cov)
                if side.any():
                    y = (bits[side] == 4).astype(np.float64)
                    if model_side_up is None:
                        denom = model_p[side, offset, j, 0] + model_p[side, offset, j, 2]
                        mp = model_p[side, offset, j, 2] / denom
                    else:
                        mp = model_side_up[side, offset, j]
                    pp = prior_side_up[side, offset, j]
                    self._add_scores(batch, side, j, 1, np.square(mp - y),
                                     np.square(pp - y), cov)
                if singleton.any():
                    labels = np.where(bits[singleton] == 1, 0,
                                      np.where(bits[singleton] == 2, 1, 2))
                    truth = np.eye(3, dtype=np.float64)[labels]
                    mp = model_p[singleton, offset, j, :]
                    pp = prior_three[singleton, offset, j, :]
                    self._add_scores(batch, singleton, j, 2,
                                     np.square(mp - truth).sum(axis=-1),
                                     np.square(pp - truth).sum(axis=-1), cov)
        batch['next_step'] += q

    @staticmethod
    def _add_scores(batch: dict, selected: np.ndarray, j: int, metric: int,
                    model_scores: np.ndarray, prior_scores: np.ndarray,
                    cov: dict) -> None:
        name = _METRICS[metric]
        batch['count'][selected, j, metric] += 1
        batch['model_sum'][selected, j, metric] += model_scores
        batch['prior_sum'][selected, j, metric] += prior_scores
        cov[name + '_model_brier_sum'] += float(model_scores.sum(dtype=np.float64))
        cov[name + '_prior_brier_sum'] += float(prior_scores.sum(dtype=np.float64))

    def end_batch(self) -> None:
        batch = self._active
        _require(batch is not None and batch['next_step'] == 289,
                 'INCOMPLETE_QUERY_GRID')
        fold = batch['fold']
        issues = batch['issues']
        self._axis_hash[fold].update(issues.tobytes())
        self._axis_count[fold] += len(issues)
        self._last_issue_by_fold[fold] = int(issues[-1])
        for tf, table in batch['coverage'].items():
            for k, row in table.items():
                total = self._coverage[tf][k]
                for key, value in row.items():
                    total[key] += value
        for j, tf in enumerate(TIMEFRAMES):
            fold_bucket = self._folds.setdefault((fold, tf), _bucket())
            for metric, name in enumerate(_METRICS):
                counts = batch['count'][:, j, metric]
                valid = counts > 0
                model = batch['model_sum'][:, j, metric]
                prior = batch['prior_sum'][:, j, metric]
                delta = np.zeros(len(issues), dtype=np.float64)
                delta[valid] = (prior[valid] - model[valid]) / counts[valid]
                target = fold_bucket[name]
                target['issue_count'] += int(valid.sum())
                target['issue_delta_sum'] += float(delta[valid].sum(dtype=np.float64))
                target['cell_count'] += int(counts.sum())
                target['model_brier_sum'] += float(model.sum(dtype=np.float64))
                target['prior_brier_sum'] += float(prior.sum(dtype=np.float64))
                for week in np.unique(batch['weeks']):
                    where = batch['weeks'] == week
                    week_target = self._weekly.setdefault((fold, str(week), tf),
                                                          _bucket())[name]
                    week_target['issue_count'] += int((valid & where).sum())
                    week_target['issue_delta_sum'] += float(delta[valid & where].sum(dtype=np.float64))
                    week_target['cell_count'] += int(counts[where].sum())
                    week_target['model_brier_sum'] += float(model[where].sum(dtype=np.float64))
                    week_target['prior_brier_sum'] += float(prior[where].sum(dtype=np.float64))
        self._active = None

    def result(self) -> dict:
        _require(self._active is None and any(self._axis_count.values()), 'UNFINISHED')
        def present(bucket: dict) -> dict:
            return {name: {**values,
                           'issue_mean_prior_minus_model': (
                               values['issue_delta_sum'] / values['issue_count']
                               if values['issue_count'] else None),
                           'cell_mean_prior_minus_model': (
                               (values['prior_brier_sum'] - values['model_brier_sum'])
                               / values['cell_count'] if values['cell_count'] else None)}
                    for name, values in bucket.items()}
        m5 = self._coverage['M5']['1']
        side_count = m5['side_known']
        activity_count = m5['activity_known']
        primary = {
            'certified_side': {'count': side_count,
                'model_brier_sum': m5['conditional_side_model_brier_sum'],
                'prior_brier_sum': m5['conditional_side_prior_brier_sum'],
                'prior_minus_model': (
                    (m5['conditional_side_prior_brier_sum']
                     - m5['conditional_side_model_brier_sum']) / side_count
                    if side_count else None)},
            'activity': {'count': activity_count,
                'model_brier_sum': m5['activity_model_brier_sum'],
                'prior_brier_sum': m5['activity_prior_brier_sum'],
                'prior_minus_model': (
                    (m5['activity_prior_brier_sum']
                     - m5['activity_model_brier_sum']) / activity_count
                    if activity_count else None)},
        }
        return {
            'schema_version': 'gx1_market_state_time_query_matched_score_v1',
            'contract_sha256': time_query_scorer_contract()['contract_sha256'],
            'scope': 'TRAIN_only_no_fit_no_TEST_no_trade',
            'issue_axis': {str(fold): {'count': self._axis_count[fold],
                          'sha256': self._axis_hash[fold].hexdigest()}
                           for fold in range(1, 6) if self._axis_count[fold]},
            'primary_M5_k1': primary,
            'per_tf_k': self._coverage,
            'fold_after_issue_TF_reduction': {
                str(fold): {tf: present(values)
                            for (f, tf), values in sorted(self._folds.items()) if f == fold}
                for fold in range(1, 6) if self._axis_count[fold]},
            'week_after_issue_TF_reduction': {
                str(fold): {week: {tf: present(values)
                                  for (f, w, tf), values in sorted(self._weekly.items())
                                  if f == fold and w == week}
                            for week in sorted({w for f, w, _ in self._weekly if f == fold})}
                for fold in range(1, 6) if self._axis_count[fold]},
            'test_data_used': False,
        }
