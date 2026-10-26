"""Fixed239 binary probability reference; no I/O, cohort selection or authority."""
from collections.abc import Mapping
import warnings

import numpy as np
from scipy.special import expit

SCHEMA_VERSION = "gx1_direction_probability_reference_v1"
SKLEARN_VERSION = "1.7.2"
PROBABILITY_CONDITION = "observed_nonzero_price_direction"
_CONFIG = {
    "penalty": "l2", "C": 1.0, "solver": "lbfgs", "tol": 1e-4,
    "max_iter": 500, "fit_intercept": True, "class_weight": None,
    "warm_start": False, "random_state": 20260911,
}


def _require(condition, reason):
    if not condition:
        raise RuntimeError("DIRECTION_PROBABILITY_REFERENCE_" + reason)


def fixed_direction_probability_config():
    return dict(_CONFIG)


def _matrix(value, *, width=None):
    _require(isinstance(value, np.ndarray) and value.dtype == np.float64
        and value.ndim == 2 and min(value.shape) > 0
        and value.flags.c_contiguous and np.isfinite(value).all(), "MATRIX")
    _require(width is None or value.shape[1] == width, "FEATURE_WIDTH")
    return value


def _labels(value, *, rows):
    _require(isinstance(value, np.ndarray) and value.ndim == 1
        and value.shape == (rows,) and np.issubdtype(value.dtype, np.integer)
        and np.isin(value, (0, 1)).all(), "BINARY_LABELS")
    return value.astype(np.int8, copy=False)


def _logits(value):
    _require(isinstance(value, np.ndarray) and value.dtype == np.float64
        and value.ndim == 1 and len(value) > 0 and np.isfinite(value).all(), "LOGITS")
    return value


def _classifier_type():
    import sklearn
    from sklearn.linear_model import LogisticRegression
    _require(sklearn.__version__ == SKLEARN_VERSION, "SKLEARN_VERSION")
    return LogisticRegression


def _require_classifier(model, *, width):
    _require(type(model) is _classifier_type(), "CLASSIFIER_TYPE")
    params = model.get_params(deep=False)
    _require(all(params.get(k) == v for k, v in _CONFIG.items())
        and params.get("dual") is False and params.get("intercept_scaling") == 1
        and params.get("n_jobs") is None and params.get("l1_ratio") is None
        and params.get("multi_class") == "deprecated", "CLASSIFIER_CONFIG")
    classes = getattr(model, "classes_", None)
    _require(isinstance(classes, np.ndarray) and np.array_equal(classes, [0, 1]), "CLASS_ORDER")
    coef, intercept = getattr(model, "coef_", None), getattr(model, "intercept_", None)
    iterations = getattr(model, "n_iter_", None)
    _require(isinstance(coef, np.ndarray) and coef.dtype == np.float64
        and coef.shape == (1, width) and np.isfinite(coef).all()
        and isinstance(intercept, np.ndarray) and intercept.dtype == np.float64
        and intercept.shape == (1,) and np.isfinite(intercept).all()
        and getattr(model, "n_features_in_", None) == width, "CLASSIFIER_STATE")
    _require(isinstance(iterations, np.ndarray) and iterations.shape == (1,)
        and np.issubdtype(iterations.dtype, np.integer)
        and 0 <= int(iterations[0]) <= _CONFIG["max_iter"], "ITERATIONS")
    return int(iterations[0])


def fit_direction_probability_reference(matrix, labels, *, split):
    """One fixed, unweighted fit on caller-selected own mature prefix rows."""
    _require(split == "train", "TRAIN_ONLY_FIT")
    x = _matrix(matrix)
    y = _labels(labels, rows=len(x))
    counts = np.bincount(y, minlength=2)
    _require(bool((counts > 0).all()), "BOTH_CLASSES_REQUIRED")
    model = _classifier_type()(**fixed_direction_probability_config())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x, y)
    _require(not caught, "FIT_WARNING")
    n_iter = _require_classifier(model, width=x.shape[1])
    return {
        "schema_version": SCHEMA_VERSION + "_fit", "classifier": model,
        "library_version": SKLEARN_VERSION, "config": fixed_direction_probability_config(),
        "feature_width": int(x.shape[1]), "fit_rows": int(len(x)),
        "class_counts": {"down": int(counts[0]), "up": int(counts[1])},
        "class_order": (0, 1), "fit_partition": "train", "n_iter": n_iter,
        "probability_condition": PROBABILITY_CONDITION,
    }


def predict_direction_probability_reference(matrix, state):
    """One decision_function call; probabilities are conditional, not trade scores."""
    keys = {"schema_version", "classifier", "library_version", "config", "feature_width",
        "fit_rows", "class_counts", "class_order", "fit_partition", "n_iter",
        "probability_condition"}
    _require(isinstance(state, Mapping) and set(state) == keys
        and state["schema_version"] == SCHEMA_VERSION + "_fit"
        and state["library_version"] == SKLEARN_VERSION
        and state["config"] == fixed_direction_probability_config()
        and state["fit_partition"] == "train" and state["class_order"] == (0, 1)
        and state["probability_condition"] == PROBABILITY_CONDITION
        and type(state["feature_width"]) is int and state["feature_width"] > 0
        and type(state["fit_rows"]) is int and state["fit_rows"] > 1, "FIT_STATE")
    counts = state["class_counts"]
    _require(isinstance(counts, Mapping) and set(counts) == {"down", "up"}
        and all(type(v) is int and v > 0 for v in counts.values())
        and sum(counts.values()) == state["fit_rows"], "FIT_COUNTS")
    x = _matrix(matrix, width=state["feature_width"])
    model = state["classifier"]
    n_iter = _require_classifier(model, width=x.shape[1])
    _require(type(state["n_iter"]) is int and state["n_iter"] == n_iter, "ITERATION_STATE")
    z = _logits(model.decision_function(x))
    _require(z.shape == (len(x),), "LOGIT_ROWS")
    return {
        "logit_up": z, "probability_up": expit(z),
        "direction_class": (z > 0).astype(np.int8),
        "nullscore_rows": int(np.count_nonzero(z == 0)),
        "probability_condition": PROBABILITY_CONDITION,
    }


def direction_probability_metrics(labels, logit_up):
    """Unweighted row scores and fixed10-bin reliability; no baseline or fitting."""
    z = _logits(logit_up)
    y = _labels(labels, rows=len(z))
    p = expit(z)
    loss = np.logaddexp(0.0, z) - y * z
    brier = (p - y) ** 2
    _require(np.isfinite(loss).all() and (loss >= 0).all(), "SCORE_NONFINITE")
    predicted = z > 0
    up = y == 1
    tn, fp = int(np.sum(~up & ~predicted)), int(np.sum(~up & predicted))
    fn, tp = int(np.sum(up & ~predicted)), int(np.sum(up & predicted))
    down_n, up_n = tn + fp, tp + fn
    down_recall = tn / down_n if down_n else None
    up_recall = tp / up_n if up_n else None
    bins = np.minimum((p * 10).astype(np.int64), 9)
    reliability = []
    ece = 0.0
    for i in range(10):
        selected = bins == i
        count = int(np.sum(selected))
        mean_p = float(np.mean(p[selected])) if count else None
        fraction_up = float(np.mean(y[selected])) if count else None
        if count:
            ece += count / len(y) * abs(mean_p - fraction_up)
        reliability.append({"bin": i, "lower": i / 10, "upper": (i + 1) / 10,
            "right_inclusive": i == 9, "rows": count,
            "mean_probability_up": mean_p, "observed_up_fraction": fraction_up})
    return {
        "rows": len(y), "up_rows": up_n, "down_rows": down_n,
        "logloss_by_row": loss, "brier_by_row": brier,
        "logloss": float(np.sum(loss / len(y))), "brier": float(np.mean(brier)),
        "confusion": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "accuracy": (tn + tp) / len(y), "down_recall": down_recall, "up_recall": up_recall,
        "balanced_accuracy": (down_recall + up_recall) / 2 if down_n and up_n else None,
        "up_predictions": tp + fp, "down_predictions": tn + fn,
        "nullscore_rows": int(np.count_nonzero(z == 0)),
        "reliability": reliability, "ece_fixed10": ece,
        "probability_condition": PROBABILITY_CONDITION,
    }


def direction_rank_diagnostics(labels, scores, *, score_kind):
    """Exact opposite-label rank pairs on raw scores, without fitting.

    Equal scores contribute half a win. Empty/single-class inputs have no
    AUC. Sorting needs O(n log n) time and O(n) memory. Inputs are unchanged.
    """
    _require(type(score_kind) is str
             and score_kind in ("logit_up", "probability_up",
                                "raw_signed_flow_contracts"), "RANK_SCORE_KIND")
    _require(isinstance(scores, np.ndarray) and scores.dtype == np.float64
             and scores.ndim == 1 and scores.flags.c_contiguous,
             "RANK_SCORE_HEADER")
    _require(bool(np.isfinite(scores).all()), "RANK_SCORE_NONFINITE")
    if score_kind == "probability_up":
        _require(bool(((scores >= 0) & (scores <= 1)).all()),
                 "RANK_PROBABILITY_DOMAIN")
    n = len(scores)
    _require(isinstance(labels, np.ndarray) and labels.dtype == np.int8
             and labels.shape == (n,) and labels.flags.c_contiguous,
             "RANK_LABEL_HEADER")
    y = _labels(labels, rows=n) if n else labels
    up = int(np.count_nonzero(y))
    down = n - up
    pairs = up * down
    concordant = tied = distinct = 0
    if n:
        order = np.argsort(scores, kind="mergesort")
        ordered_scores = scores[order]
        starts = np.r_[0, np.flatnonzero(
            ordered_scores[1:] != ordered_scores[:-1]) + 1]
        counts = np.diff(np.r_[starts, n])
        positives = np.add.reduceat(y[order], starts, dtype=np.int64)
        negatives = counts - positives
        lower_negatives = np.cumsum(negatives, dtype=np.int64) - negatives
        concordant = sum(int(a) * int(b)
                         for a, b in zip(positives, lower_negatives))
        tied = sum(int(a) * int(b)
                   for a, b in zip(positives, negatives))
        distinct = len(starts)
    discordant = pairs - concordant - tied
    return {
        "schema_version": "gx1_direction_rank_diagnostics_v1",
        "score_kind": score_kind,
        "rows": n, "up_rows": up, "down_rows": down,
        "distinct_scores": distinct,
        "up_down_pairs": pairs,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "tied_pairs": tied,
        "untied_pairs": concordant + discordant,
        "auc": (2 * concordant + tied) / (2 * pairs) if pairs else None,
    }


def direction_within_fold_rank_diagnostics(
        labels, scores, fold_ids, *, score_kind):
    """Per-fold rank plus pair-weighted within-fold AUC.

    Pooled cross-fold AUC is mixture diagnostics: changing fold priors can
    rank observations even without discrimination inside any single fold.
    """
    pooled = direction_rank_diagnostics(labels, scores, score_kind=score_kind)
    _require(isinstance(fold_ids, np.ndarray)
             and fold_ids.dtype.kind == "i"
             and fold_ids.shape == (len(scores),)
             and fold_ids.flags.c_contiguous
             and bool((fold_ids >= 0).all()), "RANK_FOLD_HEADER")
    per_fold = []
    for fold in np.unique(fold_ids):
        take = fold_ids == fold
        result = direction_rank_diagnostics(
            labels[take], scores[take], score_kind=score_kind)
        per_fold.append({"fold_id": int(fold), "diagnostics": result})
    totals = {
        key: sum(item["diagnostics"][key] for item in per_fold)
        for key in ("up_down_pairs", "concordant_pairs",
                    "discordant_pairs", "tied_pairs")
    }
    pairs = totals["up_down_pairs"]
    within_auc = (
        (2 * totals["concordant_pairs"] + totals["tied_pairs"]) / (2 * pairs)
        if pairs else None
    )
    return {
        "schema_version": "gx1_direction_within_fold_rank_diagnostics_v1",
        "score_kind": score_kind,
        "primary_scope": "per_fold",
        "per_fold": per_fold,
        "within_fold_pair_counts": totals,
        "pair_weighted_within_fold_auc": within_auc,
        "pooled_crossfold_diagnostic": pooled,
        "pooled_crossfold_is_within_fold_evidence": False,
    }
