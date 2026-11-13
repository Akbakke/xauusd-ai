"""Fixed-threshold descriptive accuracy on retained S/F logits.

Pure array arithmetic: no files, model calls, fitting or threshold selection.
Five is the timeframe axis; the joint class order is DOWN, FLAT, UP. Callers
own the endpoint/task binding, in particular supplying F to the M5 bin report.
"""
from __future__ import annotations

import numpy as np

from gx1.contracts.market_state_time_query_scorer_v2 import factorized_probabilities
from gx1.contracts.market_state_time_query_target_v1 import TIMEFRAMES, VALID_BITS

FIXED_P_UP_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _need(ok, reason):
    if not bool(ok):
        raise ValueError('MARKET_STATE_RETAINED_ACCURACY_' + reason)


def _validated(raw, allowed, eligible, mature):
    _need(isinstance(raw, np.ndarray) and raw.ndim == 3, 'RAW_LAYOUT')
    classes, active, side_up = factorized_probabilities(raw)
    shape = raw.shape[:2]
    _need(isinstance(allowed, np.ndarray) and allowed.dtype == np.uint8
          and allowed.shape == shape and np.isin(allowed, VALID_BITS).all(), 'ALLOWED')
    _need(all(isinstance(mask, np.ndarray) and mask.dtype == np.bool_ and mask.shape == shape
              for mask in (eligible, mature)), 'MASKS')
    return classes, active, side_up, eligible & mature


def _classification(truth, prediction, names):
    """Rows are truth, columns prediction; absent-class recalls stay undefined."""
    n, width = len(truth), len(names)
    matrix = np.bincount(truth * width + prediction, minlength=width * width).reshape(width, width)
    actual, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    correct = int(np.trace(matrix))
    recalls = {name: float(matrix[i, i] / actual[i]) if actual[i] else None
               for i, name in enumerate(names)}
    return dict(count=n, correct=correct, true_class_order=list(names), predicted_class_order=list(names),
        confusion_matrix=matrix.tolist(), accuracy=correct / n if n else None,
        balanced_accuracy=sum(recalls.values()) / width if all(actual > 0) else None,
        recalls=recalls, predicted_fractions={name: float(predicted[i] / n) if n else None
                                            for i, name in enumerate(names)})


def _allowed_counts(bits):
    return {str(bit): int(np.count_nonzero(bits == bit)) for bit in VALID_BITS}


def score_accuracy(raw, allowed, eligible, mature):
    """Return per-TF side/activity/three-class accuracy on their separate masks.

    Side predicts UP at logit >= 0 (ties UP), activity predicts ACTIVE at >= 0.
    Three-class argmax uses the original factorized probabilities and NumPy's
    first-maximum rule: DOWN before FLAT before UP. No threshold is estimated.
    Balanced accuracy is None unless every class of that metric is represented.
    """
    classes, _, _, common = _validated(raw, allowed, eligible, mature)
    report = {}
    for j, tf in enumerate(TIMEFRAMES):
        bits, ready = allowed[:, j], common[:, j]
        side_mask = ready & np.isin(bits, (1, 4))
        activity_mask = ready & (bits != 7)
        single_mask = ready & np.isin(bits, (1, 2, 4))
        side_truth = (bits[side_mask] == 4).astype(np.int64)
        side_logits = raw[side_mask, j, 1]
        side = _classification(side_truth, (side_logits >= 0).astype(np.int64), ('DOWN', 'UP'))
        side.update(threshold_logit=0.0, tie_rule='UP', exact_ties=int(np.count_nonzero(side_logits == 0)),
            always_up_accuracy=float(side_truth.mean()) if len(side_truth) else None,
            always_down_accuracy=float((side_truth == 0).mean()) if len(side_truth) else None)
        activity_logits = raw[activity_mask, j, 0]
        activity = _classification((bits[activity_mask] != 2).astype(np.int64),
            (activity_logits >= 0).astype(np.int64), ('FLAT', 'ACTIVE'))
        activity.update(threshold_logit=0.0, tie_rule='ACTIVE',
            exact_ties=int(np.count_nonzero(activity_logits == 0)))
        selected_bits = bits[single_mask]
        truth = np.where(selected_bits == 1, 0, np.where(selected_bits == 2, 1, 2))
        probabilities = classes[single_mask, j]
        prediction = np.argmax(probabilities, axis=1)
        three = _classification(truth, prediction, ('DOWN', 'FLAT', 'UP'))
        ties = np.count_nonzero((probabilities == probabilities.max(axis=1, keepdims=True)).sum(axis=1) > 1)
        three.update(tie_rule='FIRST_MAX_DOWN_FLAT_UP', exact_ties=int(ties),
            always_flat_accuracy=float((selected_bits == 2).mean()) if len(selected_bits) else None)
        support = dict(rows=len(bits), eligible=int(eligible[:, j].sum()), mature=int(mature[:, j].sum()),
            eligible_mature=int(ready.sum()), ineligible_or_immature=int((~ready).sum()),
            allowed_counts_on_common=_allowed_counts(bits[ready]), conditional_side_rows=side['count'],
            activity_rows=activity['count'], three_class_rows=three['count'])
        report[tf] = dict(support=support, conditional_side=side, activity=activity, three_class=three)
    return report


def _fixed_bin_indices(p_up):
    # Interior edges belong to the right bin; both 0 and 1 are retained.
    return np.searchsorted(np.asarray(FIXED_P_UP_EDGES[1:-1]), p_up, side='right')


def fixed_probability_bins(raw, allowed, eligible, mature):
    """M5 conditional-UP bins on ALL eligible/mature rows, including FLAT/UNKNOWN.

    Bins are [0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1]. Their row fractions
    share the all-common-row denominator. Only known sides enter each bin's
    side accuracy/UP fraction; no direction is imputed for bits 2, 5 or 7.
    """
    _, _, side_up, common = _validated(raw, allowed, eligible, mature)
    ready = common[:, 0]
    bits, p_up, logits = allowed[ready, 0], side_up[ready, 0], raw[ready, 0, 1]
    indices = _fixed_bin_indices(p_up)
    total = len(bits)
    bins = []
    for index, (lower, upper) in enumerate(zip(FIXED_P_UP_EDGES[:-1], FIXED_P_UP_EDGES[1:])):
        selected = indices == index
        known = selected & np.isin(bits, (1, 4))
        count, n_known = int(selected.sum()), int(known.sum())
        correct = int(np.count_nonzero((logits[known] >= 0) == (bits[known] == 4)))
        bins.append(dict(index=index, lower=lower, upper=upper, upper_inclusive=index == 4,
            count=count, all_common_row_fraction=count / total if total else None,
            allowed_counts=_allowed_counts(bits[selected]), mean_p_up=float(p_up[selected].mean()) if count else None,
            known_side_count=n_known, known_side_correct=correct,
            known_side_accuracy=correct / n_known if n_known else None,
            known_side_up_fraction=float((bits[known] == 4).mean()) if n_known else None))
    return dict(timeframe='M5', probability='UP_GIVEN_ACTIVE', edges=list(FIXED_P_UP_EDGES),
        rows=len(allowed), eligible_mature_rows=total, excluded_rows=int((~ready).sum()),
        side_threshold_logit=0.0, side_tie_rule='UP', bins=bins)
