"""Tests for materialize_learn_trade_skip_meta_classifier_v2_logistic_balanced_v1."""
from __future__ import annotations

import numpy as np
import pytest

from gx1.scripts import (
    materialize_learn_trade_skip_meta_classifier_v2_logistic_balanced_v1 as gate,
)


def test_threshold_grid_covers_balanced_range() -> None:
    """A balanced logistic classifier produces probs centered near 0.5; the
    grid must cover that range."""
    assert 0.50 in gate.THRESHOLD_GRID
    assert min(gate.THRESHOLD_GRID) >= 0.30
    assert max(gate.THRESHOLD_GRID) <= 0.80
    assert len(gate.THRESHOLD_GRID) == 10


def test_logreg_hyperparameters_use_balanced_class_weight() -> None:
    assert gate.LOGREG_CLASS_WEIGHT == "balanced"
    assert gate.LOGREG_PENALTY == "l2"
    assert gate.LOGREG_C == 1.0
    assert gate.LOGREG_MAX_ITER >= 100


def test_train_logistic_drops_intercept_column() -> None:
    rng = np.random.default_rng(7)
    n, k = 200, 4
    # Build a design matrix with a leading intercept column then k features.
    X = np.column_stack([np.ones(n), rng.normal(size=(n, k))])
    # Synthetic decision boundary on feature 0 only.
    y = (X[:, 1] > 0.0).astype(int)
    model = gate._train_logistic(X, y)
    # sklearn's intercept is learned internally -> coef_ has shape (1, k)
    assert model.coef_.shape == (1, k)
    assert hasattr(model, "intercept_")


def test_train_logistic_learns_decision_boundary() -> None:
    rng = np.random.default_rng(7)
    n, k = 1000, 3
    X = np.column_stack([np.ones(n), rng.normal(size=(n, k))])
    # Strong signal on feature 1 (column index 2 in design matrix).
    y = (X[:, 2] > 0.5).astype(int)
    model = gate._train_logistic(X, y)
    p = gate._predict_p_skip(model, X)
    # Predictions should be strictly inside (0, 1) and correlated with X[:, 2]
    assert (p > 0.0).all() and (p < 1.0).all()
    corr = float(np.corrcoef(p, X[:, 2])[0, 1])
    assert corr > 0.5


def test_predict_p_skip_returns_probability_for_class_one() -> None:
    rng = np.random.default_rng(7)
    n = 100
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    y = (X[:, 1] + X[:, 2] > 0).astype(int)
    model = gate._train_logistic(X, y)
    p = gate._predict_p_skip(model, X)
    assert p.shape == (n,)
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_predict_p_skip_rejects_model_without_positive_class() -> None:
    """If the training data had only class 0 the logistic regression cannot
    predict class 1; we should raise rather than silently misalign columns."""

    class StubModel:
        classes_ = np.array([0])

        def predict_proba(self, X):
            return np.full((len(X), 1), 1.0)

    with pytest.raises(RuntimeError, match="LOGREG_DID_NOT_LEARN_POSITIVE_CLASS"):
        gate._predict_p_skip(StubModel(), np.zeros((3, 2)))


def test_validate_final_status_rejects_unknown_status() -> None:
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status(
            "MADE_UP", "COMBINE_SKIP_CLASSIFIER_V2_WITH_EXIT_IQL_V3_V1"
        )


def test_validate_final_status_rejects_unknown_next_action() -> None:
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(
            "LEARN_TRADE_SKIP_META_CLASSIFIER_V2_PASS_LIFTS_V1_BASELINE", "TRAIN_NOW"
        )


def test_validate_no_deprecated_revival_passes_on_self() -> None:
    from pathlib import Path

    gate.validate_no_deprecated_revival(Path(gate.__file__))


def test_v2_reuses_v1_label_formula() -> None:
    """V2 must use the same audit_should_have_skipped_v2 label formula as V1
    to keep results comparable."""
    from gx1.scripts import (
        materialize_learn_trade_skip_meta_classifier_at_trade_open_v1 as v1_gate,
    )

    assert gate.v1_gate is v1_gate
    # The constants must match.
    assert v1_gate.SHOULD_SKIP_PNL_THRESHOLD == 0.0
    assert v1_gate.SHOULD_SKIP_MAE_THRESHOLD == -50.0
    assert v1_gate.SHOULD_SKIP_MFE_THRESHOLD == 25.0
