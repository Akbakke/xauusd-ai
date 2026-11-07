"""One OOF issue batch for the factorized time-query model.

The caller binds the exact v2 model, fold, feature view, checkpoint, and frozen
priors. This bridge encodes each issue once and sends primitive logits directly
to the v2 matched scorer. It has no IO, fit, TEST, or run authority.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import torch

from gx1.contracts.market_state_time_query_scorer_v2 import FactorizedTimeQueryMatchedScorer
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TIMEFRAMES

BATCH_SIZE = 16
QUERY_BLOCK = 32
MAX_STEP = max(int(value) for value in NATIVE_K)


def _need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise ValueError('TIME_QUERY_V2_OOF_BATCH_' + reason)


def score_oof_issue_batch(
    model: object, features: Mapping[str, torch.Tensor],
    parent_rows: np.ndarray, issue_ns: np.ndarray, *,
    bound: object, scorer: FactorizedTimeQueryMatchedScorer,
    priors: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device, fold_id: int,
    check_deadline: Callable[[], None],
) -> None:
    """Score all native future queries without float32 class softmax."""
    _need(isinstance(parent_rows, np.ndarray) and parent_rows.dtype == np.int64
          and parent_rows.ndim == 1 and 0 < len(parent_rows) <= BATCH_SIZE
          and isinstance(issue_ns, np.ndarray) and issue_ns.dtype == np.int64
          and issue_ns.shape == parent_rows.shape
          and bool(np.all(np.diff(issue_ns) > 0))
          and type(fold_id) is int and 1 <= fold_id <= 5,
          'AXES')
    _need(isinstance(features, Mapping)
          and all(isinstance(value, torch.Tensor) for value in features.values())
          and callable(getattr(model, 'encode_features', None))
          and callable(getattr(model, 'decode_raw_query_grid', None))
          and callable(getattr(bound, 'gather', None))
          and type(scorer) is FactorizedTimeQueryMatchedScorer
          and isinstance(device, torch.device)
          and callable(check_deadline), 'OWNERS')
    _need(isinstance(priors, tuple) and len(priors) == 3, 'PRIORS')
    activity, side, three = priors
    _need(all(isinstance(a, np.ndarray) and a.dtype == np.float64
              and bool(np.isfinite(a).all()) for a in priors)
          and activity.shape == side.shape == (MAX_STEP, len(TIMEFRAMES))
          and three.shape == (MAX_STEP, len(TIMEFRAMES), 3)
          and bool(np.all((activity >= 0) & (activity <= 1)))
          and bool(np.all((side >= 0) & (side <= 1)))
          and bool(np.all((three >= 0) & (three <= 1)))
          and bool(np.allclose(three.sum(axis=-1), 1.0, rtol=0, atol=1e-10)),
          'PRIOR_DOMAIN')
    batch = len(parent_rows)
    scorer.begin_batch(issue_ns, fold_id=fold_id)
    with torch.inference_mode():
        hidden = model.encode_features(**features)
        for step_start in range(1, MAX_STEP + 1, QUERY_BLOCK):
            check_deadline()
            steps = np.arange(step_start, min(step_start + QUERY_BLOCK, MAX_STEP + 1),
                              dtype=np.int64)
            q = len(steps)
            k = np.minimum(steps[:, None], np.asarray(NATIVE_K, dtype=np.int64)[None, :])
            query = np.broadcast_to(k[None, :, :],
                                    (batch, q, len(TIMEFRAMES))).copy()
            raw = model.decode_raw_query_grid(
                hidden, torch.from_numpy(query).to(device))
            _need(isinstance(raw, torch.Tensor)
                  and raw.dtype == torch.float32
                  and raw.shape == (batch, q, len(TIMEFRAMES), 2)
                  and bool(torch.isfinite(raw).all()), 'RAW_LOGITS')
            raw_np = raw.detach().cpu().numpy()
            flat_parent = np.repeat(parent_rows, q)
            flat_issue = np.repeat(issue_ns, q)
            truth = bound.gather(flat_parent, flat_issue,
                                 query.reshape(batch * q, len(TIMEFRAMES)),
                                 fold_id=fold_id, partition='score')
            _need(np.array_equal(truth['parent_rows'], flat_parent)
                  and np.array_equal(truth['issue_ns'], flat_issue)
                  and np.array_equal(truth['query_k'],
                                     query.reshape(batch * q, len(TIMEFRAMES))),
                  'TARGET_AXIS')
            shape = (batch, q, len(TIMEFRAMES))
            scorer.add_block(
                step_start, query_k=query,
                validity_ns=truth['validity_ns'].reshape(shape),
                raw_logits=raw_np,
                prior_activity=np.broadcast_to(activity[steps - 1][None, :, :],
                                               shape),
                prior_side_up=np.broadcast_to(side[steps - 1][None, :, :], shape),
                prior_three=np.broadcast_to(three[steps - 1][None, :, :, :],
                                            shape + (3,)),
                allowed=truth['allowed'].reshape(shape),
                eligible=truth['eligible'].reshape(shape),
                mature=truth['mature'].reshape(shape))
    scorer.end_batch()
