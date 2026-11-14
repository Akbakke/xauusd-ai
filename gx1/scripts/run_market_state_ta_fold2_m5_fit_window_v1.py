"""Fold 2 M5 direction fit core. No CLI, campaign route, or admission yet."""
from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np
import torch

from gx1.contracts.market_state_ta_fold2_m5_fit_v1 import (
    ADAMW_DEFAULTS, BATCH_VISITS, INITIAL_MODEL_SHA256, LOSS_SHA256,
    MODEL_CONTRACT_SHA256, TRAIN_ROWS,
    batch_bounds,
)
from gx1.contracts.market_state_time_query_native_F_batch_v1 import evaluate_fit_batch
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.contracts.market_state_time_query_native_F_objective_v1 import (
    require_m5_equal_observation_axes,
)
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError('FOLD2_M5_FIT_' + reason)


def fit_one_pass(*, model, optimizer, parent_rows: np.ndarray,
                 issue_ns: np.ndarray, state_bound, future_bound,
                 gather_features: Callable, deadline_monotonic: float) -> dict:
    """Consume every fixed TRAIN batch once after the caller proves admission.

    The caller owns guarded launch, exact source/start binding, feature inputs,
    receipts, and failure consumption. This function cannot grant execution.
    """
    _need(canonical_model_state_sha256(model.state_dict())
          == INITIAL_MODEL_SHA256, 'FRESH_INITIAL_STATE')
    parameters = list(model.parameters())
    _need(type(optimizer) is torch.optim.AdamW and not optimizer.state
          and optimizer.defaults == ADAMW_DEFAULTS
          and len(optimizer.param_groups) == 1
          and {id(p) for p in optimizer.param_groups[0]['params']}
              == {id(p) for p in parameters}
          and model.training and all(parameter.grad is None for parameter in parameters),
          'EMPTY_OPTIMIZER_AND_TRAIN_MODE')
    _need(isinstance(parent_rows, np.ndarray) and isinstance(issue_ns, np.ndarray)
          and parent_rows.dtype == issue_ns.dtype == np.dtype('int64')
          and parent_rows.shape == issue_ns.shape == (TRAIN_ROWS,), 'TRAIN_AXES')
    require_m5_equal_observation_axes(parent_rows, issue_ns, fold_id=2)
    for bound in (state_bound, future_bound):
        _need(np.array_equal(bound.parent_rows, parent_rows)
              and np.array_equal(bound.decision_ns, issue_ns), 'TARGET_AXES')
    _need(type(deadline_monotonic) is float
          and time.monotonic() < deadline_monotonic, 'DEADLINE')
    optimizer.zero_grad(set_to_none=True)
    records, updates, skipped = [], 0, 0
    for index in range(BATCH_VISITS):
        _need(time.monotonic() < deadline_monotonic, 'TIME_BUDGET')
        left, right = batch_bounds(index)
        rows, issues = parent_rows[left:right], issue_ns[left:right]
        features = gather_features(left, right, rows, issues)
        report = evaluate_fit_batch(
            model, features, state_bound, future_bound, rows, issues,
            fold_id=2, pass_index=0, seed=20260911, future_weight=1.0,
            m5_equal_observation=True, fit_batch_start=left,
            m5_direction_only=True)
        _need(report['loss_contract_sha256'] == LOSS_SHA256
              and report['model_contract_sha256'] == MODEL_CONTRACT_SHA256
              and bool(torch.isfinite(report['loss']).all())
              and report['skip_update'] is (report['count'] == 0), 'LOSS_AND_SUPPORT')
        record = {'batch_index': index, 'start': left, 'stop': right,
                  'known_side_count': int(report['count']),
                  'skip_update': report['skip_update'],
                  'loss': float(report['loss'].detach())}
        if report['skip_update']:
            skipped += 1
        else:
            report['loss'].backward()
            norm = float(trainer._optimizer_step_with_finite_gradients(
                model=model, optimizer=optimizer))
            _need(math.isfinite(norm), 'GRADIENT_NORM')
            record['gradient_norm'] = norm
            updates += 1
        optimizer.zero_grad(set_to_none=True)
        records.append(record)
        del features, report
    _need(len(records) == BATCH_VISITS and records[-1]['stop'] == TRAIN_ROWS
          and updates + skipped == BATCH_VISITS
          and all(parameter.grad is None for parameter in model.parameters()),
          'COMPLETE_ONE_PASS')
    return {'batch_visits': BATCH_VISITS, 'optimizer_updates': updates,
            'skipped_batches': skipped, 'training_records': records,
            'parent_cursor': TRAIN_ROWS, 'checkpoint_writes': 0,
            'test_data_used': False}
