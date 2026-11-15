"""Fixed, label-blind S path queries over the five native timeframes.

Five semantic checkpoints per timeframe retain intermediate states. A changed
pair proves at least one transition in its interval; equal endpoints do not
prove that the intervening path was constant. No joint-path probability is
claimed from the existing marginal head. This is source-only, not fit authority.
"""
from __future__ import annotations

import hashlib

import numpy as np

from gx1.contracts import market_state_time_query_scorer_v2 as scorer
from gx1.contracts.market_state_minute_query_candidate_v1 import (
    MinuteQueryTargetTape, NATIVE_MINUTES, minute_proper_score_sums, need,
)
from gx1.contracts.market_state_time_query_target_v1 import MINUTE_NS, TIMEFRAMES

# Earliest observable minute, then quarter/half/three-quarter/native landmarks.
# The rule is fixed by clock geometry, not by later score or target content.
QUERY_MINUTES = np.asarray([
    [1, 1, 1, 1, 1],
    [2, 4, 15, 60, 360],
    [3, 8, 30, 120, 720],
    [4, 12, 45, 180, 1080],
    [5, 15, 60, 240, 1440],
], dtype=np.int64)
QUERY_SHA256 = hashlib.sha256(QUERY_MINUTES.tobytes()).hexdigest()
QUERY_MINUTES.setflags(write=False)


def fixed_query_grid(issue_ns: np.ndarray) -> np.ndarray:
    """B x 5 checkpoints x 5 TF; independent of labels and batch order."""
    need(isinstance(issue_ns, np.ndarray) and issue_ns.dtype == np.int64
         and issue_ns.ndim == 1 and len(issue_ns) > 0
         and bool(np.all(issue_ns >= 0))
         and bool(np.all(issue_ns % (5 * MINUTE_NS) == 0)), 'PATH_ISSUES')
    return np.broadcast_to(QUERY_MINUTES[None],
                           (len(issue_ns),) + QUERY_MINUTES.shape).copy()


def gather_fixed_path(tape: MinuteQueryTargetTape, issue_ns: np.ndarray,
                      *, fold_upper_ns: int) -> dict:
    """Reuse the exact minute tape and its maturity/unknown masks."""
    need(isinstance(tape, MinuteQueryTargetTape), 'PATH_TAPE')
    query = fixed_query_grid(issue_ns)
    points = [tape.gather_minutes(issue_ns, query[:, q, :],
                                  fold_upper_ns=fold_upper_ns)
              for q in range(len(QUERY_MINUTES))]
    keys = ('validity_ns', 'allowed', 'm1_source_row', 'mature', 'eligible',
            'path_observed', 'source_gap_kind')
    result = {key: np.stack([point[key] for point in points], axis=1)
              for key in keys}
    result['issue_ns'] = issue_ns.copy()
    result['query_minutes'] = query
    result['query_sha256'] = QUERY_SHA256
    need(bool(np.array_equal(result['validity_ns'],
                             issue_ns[:, None, None] + query * MINUTE_NS))
         and bool(np.all(~result['eligible'] | result['mature']))
         and bool(np.all(result['allowed'][~result['mature']] == 7))
         and bool(np.all(~result['eligible'][~result['mature']])),
         'PATH_MASK_PARITY')
    return result


def observed_transition_intervals(path: dict) -> dict:
    """Count certified endpoint changes; never infer an unseen constant path."""
    allowed = path['allowed']
    mature = path['mature']
    eligible = path['eligible']
    query = path['query_minutes']
    need(isinstance(allowed, np.ndarray) and allowed.dtype == np.uint8
         and allowed.ndim == 3 and allowed.shape[1:] == QUERY_MINUTES.shape
         and all(isinstance(x, np.ndarray) and x.dtype == np.bool_
                 and x.shape == allowed.shape for x in (mature, eligible))
         and isinstance(query, np.ndarray) and query.dtype == np.int64
         and query.shape == allowed.shape
         and bool(np.array_equal(query, np.broadcast_to(
             QUERY_MINUTES[None], query.shape))), 'TRANSITION_SHAPE')
    singleton = eligible & mature & np.isin(allowed, (1, 2, 4))
    intervals = {}
    for j, tf in enumerate(TIMEFRAMES):
        intervals[tf] = []
        for q in range(1, len(QUERY_MINUTES)):
            paired = singleton[:, q - 1, j] & singleton[:, q, j]
            changed = paired & (allowed[:, q - 1, j] != allowed[:, q, j])
            intervals[tf].append({
                'from_minute': int(QUERY_MINUTES[q - 1, j]),
                'through_minute': int(QUERY_MINUTES[q, j]),
                'certified_endpoint_pairs': int(paired.sum()),
                'proven_at_least_one_change': int(changed.sum()),
                'equal_endpoints_path_unresolved': int(paired.sum() - changed.sum()),
                'unresolved_pairs': int(len(allowed) - paired.sum()),
            })
    return {'intervals_by_tf': intervals,
            'semantics': 'changed_endpoints_prove_one_or_more_transitions_within_interval;'
                         'equal_endpoints_do_not_prove_no_intermediate_transition',
            'query_sha256': QUERY_SHA256}


def checkpoint_transition_diagnostic(raw: np.ndarray, path: dict) -> dict:
    """Argmax change diagnostics on certified adjacent checkpoint pairs only."""
    classes, _, _ = scorer.factorized_probabilities(raw)
    predicted = np.asarray((1, 2, 4), np.uint8)[np.argmax(classes, axis=-1)]
    allowed = path['allowed']
    singleton = (path['eligible'] & path['mature']
                 & np.isin(allowed, (1, 2, 4)))
    result = {}
    for j, tf in enumerate(TIMEFRAMES):
        result[tf] = []
        for q in range(1, len(QUERY_MINUTES)):
            paired = singleton[:, q - 1, j] & singleton[:, q, j]
            truth_change = allowed[:, q - 1, j] != allowed[:, q, j]
            predicted_change = predicted[:, q - 1, j] != predicted[:, q, j]
            result[tf].append({
                'from_minute': int(QUERY_MINUTES[q - 1, j]),
                'through_minute': int(QUERY_MINUTES[q, j]),
                'certified_endpoint_pairs': int(paired.sum()),
                'detected_change': int(np.count_nonzero(
                    paired & truth_change & predicted_change)),
                'missed_change': int(np.count_nonzero(
                    paired & truth_change & ~predicted_change)),
                'false_change': int(np.count_nonzero(
                    paired & ~truth_change & predicted_change)),
                'equal_endpoints_no_predicted_change': int(np.count_nonzero(
                    paired & ~truth_change & ~predicted_change)),
                'unresolved_pairs': int(len(allowed) - paired.sum()),
            })
    return {'per_tf': result,
            'interpretation': 'checkpoint_argmax_change_only_not_joint_transition_probability',
            'query_sha256': QUERY_SHA256}


def score_fixed_path(raw: np.ndarray, path: dict) -> dict:
    """Proper marginal S scores at exactly the declared five checkpoints."""
    allowed = path['allowed']
    need(isinstance(raw, np.ndarray) and raw.dtype in (np.float32, np.float64)
         and raw.shape == allowed.shape + (2,)
         and bool(np.isfinite(raw).all()), 'PATH_SCORE_RAW')
    return {
        'query_sha256': QUERY_SHA256,
        'per_checkpoint': [minute_proper_score_sums(
            raw[:, q], allowed[:, q], path['eligible'][:, q],
            path['mature'][:, q], path['query_minutes'][:, q])
            for q in range(len(QUERY_MINUTES))],
        'transition_observations': observed_transition_intervals(path),
        'transition_prediction_diagnostic': checkpoint_transition_diagnostic(raw, path),
        'joint_path_probability_claimed': False,
    }


def _three_sign_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Exact tied-rank AUC for the reused current-sign {-1,0,+1} rule."""
    pos = int(labels.sum())
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    lower_neg = 0
    wins = 0.0
    for score in (-1, 0, 1):
        at = scores == score
        up = int(np.count_nonzero(at & (labels == 1)))
        down = int(np.count_nonzero(at & (labels == 0)))
        wins += up * (lower_neg + 0.5 * down)
        lower_neg += down
    return wins / (pos * neg)


def same_tf_persistence_reference(path: dict, current_allowed: np.ndarray) -> dict:
    """Current S sign ranks future known side at each fixed time on same rows.

    The caller must supply exact causal current bits from the predecessor M1
    anchor check. Unknown/flat current states have score zero and stay in scope.
    """
    allowed = path['allowed']
    need(isinstance(current_allowed, np.ndarray)
         and current_allowed.dtype == np.uint8
         and current_allowed.shape == (len(allowed), len(TIMEFRAMES))
         and bool(np.isin(current_allowed, (0, 1, 2, 4, 5, 7)).all()),
         'PERSISTENCE_CURRENT')
    result = {}
    for j, tf in enumerate(TIMEFRAMES):
        now = current_allowed[:, j]
        sign = np.zeros(len(now), dtype=np.int8)
        sign[now == 1] = -1
        sign[now == 4] = 1
        result[tf] = []
        for q in range(len(QUERY_MINUTES)):
            future = allowed[:, q, j]
            mask = (path['eligible'][:, q, j] & path['mature'][:, q, j]
                    & np.isin(future, (1, 4)))
            labels = (future[mask] == 4).astype(np.uint8)
            result[tf].append({
                'query_minute': int(QUERY_MINUTES[q, j]),
                'mature_rows': int(path['mature'][:, q, j].sum()),
                'eligible_rows': int(path['eligible'][:, q, j].sum()),
                'known_future_side_rows': int(mask.sum()),
                'future_up_rows': int(labels.sum()),
                'current_certified_direction_on_known_future':
                    int(np.count_nonzero(sign[mask])),
                'same_tf_current_sign_auc': _three_sign_auc(labels, sign[mask]),
                'constant_rank_auc': 0.5 if 0 < labels.sum() < len(labels) else None,
            })
    return {'per_tf': result, 'query_sha256': QUERY_SHA256,
            'current_rule': 'exact_causal_same_TF_current_state_supplied_by_caller',
            'no_fit': True}
