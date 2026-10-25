"""Synthetic239 probability mechanics; one shared fit, no market evidence."""
from copy import deepcopy
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from gx1.contracts import entry_direction_probability_reference_v1 as reference


def _case():
    x = np.array([[-3., 1.], [-2., 1.], [-1., 1.], [-.5, 1.],
        [.5, 1.], [1., 1.], [2., 1.], [3., 1.]], dtype=np.float64)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    return x, y


@pytest.fixture(scope="module")
def fitted_case():
    x, y = _case()
    before_x, before_y = x.copy(), y.copy()
    state = reference.fit_direction_probability_reference(x, y, split="train")
    np.testing.assert_array_equal(x, before_x)
    np.testing.assert_array_equal(y, before_y)
    return x, y, state


def test_fixed_config_orientation_and_single_logit_api(fitted_case, monkeypatch):
    x, y, state = fitted_case
    assert state["class_order"] == (0, 1)
    assert state["class_counts"] == {"down": 4, "up": 4}
    assert state["library_version"] == "1.7.2"
    assert state["config"] == {"penalty": "l2", "C": 1., "solver": "lbfgs", "tol": 1e-4,
        "max_iter": 500, "fit_intercept": True, "class_weight": None,
        "warm_start": False, "random_state": 20260911}
    model = state["classifier"]
    original = model.decision_function
    calls = []
    def counted(value):
        calls.append(len(value))
        return original(value)
    monkeypatch.setattr(model, "decision_function", counted)
    prediction = reference.predict_direction_probability_reference(x, state)
    assert calls == [len(x)]
    np.testing.assert_array_equal(prediction["logit_up"], original(x))
    np.testing.assert_allclose(prediction["probability_up"], model.predict_proba(x)[:, 1], rtol=1e-15, atol=0)
    # This only checks orientation of an informative synthetic sign, not acceptance performance.
    assert model.coef_[0, 0] > 0
    assert prediction["logit_up"][0] < 0 < prediction["logit_up"][-1]
    assert prediction["probability_condition"] == "observed_nonzero_price_direction"
    changed = reference.fixed_direction_probability_config()
    changed["C"] = 2.
    assert reference.fixed_direction_probability_config()["C"] == 1.


def test_predictor_exact_zero_logit_is_half_probability_and_down(fitted_case, monkeypatch):
    x, _, original = fitted_case
    state = deepcopy(original)
    z = np.array([-0., 0., -1e-12, 1e-12], dtype=np.float64)
    monkeypatch.setattr(state["classifier"], "decision_function", lambda value: z.copy())
    result = reference.predict_direction_probability_reference(x[:4], state)
    np.testing.assert_array_equal(result["probability_up"][:2], [.5, .5])
    np.testing.assert_array_equal(result["direction_class"], [0, 0, 0, 1])
    assert result["nullscore_rows"] == 2


def test_stable_extreme_logits_exact_ties_confusion_and_bins():
    z = np.array([-1000., 1000., 0., -0., 1000., -1000.], dtype=np.float64)
    y = np.array([0, 1, 1, 0, 0, 1], dtype=np.int8)
    result = reference.direction_probability_metrics(y, z)
    np.testing.assert_allclose(result["logloss_by_row"], [0., 0., np.log(2), np.log(2), 1000., 1000.])
    np.testing.assert_array_equal(result["brier_by_row"], [0., 0., .25, .25, 1., 1.])
    assert np.isfinite(result["logloss"]) and result["nullscore_rows"] == 2
    assert result["confusion"] == {"TN": 2, "FP": 1, "FN": 2, "TP": 1}
    assert result["accuracy"] == result["balanced_accuracy"] == .5
    bins = result["reliability"]
    assert len(bins) == 10 and sum(row["rows"] for row in bins) == 6
    assert [bins[i]["rows"] for i in (0, 5, 9)] == [2, 2, 2]
    assert bins[9]["right_inclusive"] and bins[9]["mean_probability_up"] == 1.
    assert bins[1]["mean_probability_up"] is None
    assert result["ece_fixed10"] == pytest.approx(1 / 3)


def test_known_prior_logit_and_single_class_metric_without_fit():
    y = np.array([0, 1, 1, 1], dtype=np.int8)
    z = np.full(4, np.log(3.), dtype=np.float64)
    result = reference.direction_probability_metrics(y, z)
    assert result["brier"] == pytest.approx(3 / 16)
    assert result["logloss"] == pytest.approx(-(.25 * np.log(.25) + .75 * np.log(.75)))
    only_down = reference.direction_probability_metrics(np.zeros(2, dtype=np.int8), np.array([-2., 2.]))
    assert only_down["up_recall"] is None and only_down["balanced_accuracy"] is None
    assert only_down["down_recall"] == .5


@pytest.mark.parametrize("fault", ["split", "float32", "nonfinite", "noncontiguous",
    "matrix_shape", "labels_float", "labels_invalid", "labels_shape", "one_class"])
def test_invalid_fit_domains_reject_before_library_fit(monkeypatch, fault):
    x, y = _case()
    split = "train"
    if fault == "split": split = "control"
    elif fault == "float32": x = x.astype(np.float32)
    elif fault == "nonfinite": x[0, 0] = np.nan
    elif fault == "noncontiguous": x = x[:, ::-1]
    elif fault == "matrix_shape": x = x[:, 0]
    elif fault == "labels_float": y = y.astype(np.float64)
    elif fault == "labels_invalid": y[0] = -1
    elif fault == "labels_shape": y = y[:, None]
    elif fault == "one_class": y[:] = 0
    def forbidden(*args, **kwargs):
        pytest.fail("invalid input reached sklearn.fit")
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    with pytest.raises(RuntimeError, match="DIRECTION_PROBABILITY_REFERENCE_"):
        reference.fit_direction_probability_reference(x, y, split=split)


@pytest.mark.parametrize("warning_type", [ConvergenceWarning, RuntimeWarning])
def test_any_fit_warning_is_fatal_without_real_fit(monkeypatch, warning_type):
    x, y = _case()
    def warning_only(self, *args, **kwargs):
        warnings.warn("synthetic warning", warning_type)
        return self
    monkeypatch.setattr(LogisticRegression, "fit", warning_only)
    with pytest.raises(RuntimeError, match="FIT_WARNING"):
        reference.fit_direction_probability_reference(x, y, split="train")


@pytest.mark.parametrize("fault", ["config", "classes", "coefficient", "intercept", "counts"])
def test_mutated_state_rejects_without_refit(fitted_case, fault):
    x, _, original = fitted_case
    state = deepcopy(original)
    if fault == "config": state["classifier"].set_params(C=2.)
    elif fault == "classes": state["classifier"].classes_ = np.array([1, 0], dtype=np.int8)
    elif fault == "coefficient": state["classifier"].coef_[0, 0] = np.nan
    elif fault == "intercept": state["classifier"].intercept_[0] = np.inf
    elif fault == "counts": state["class_counts"]["up"] += 1
    with pytest.raises(RuntimeError, match="DIRECTION_PROBABILITY_REFERENCE_"):
        reference.predict_direction_probability_reference(x, state)


def test_nonfinite_prediction_logit_rejects(fitted_case, monkeypatch):
    x, _, original = fitted_case
    state = deepcopy(original)
    monkeypatch.setattr(state["classifier"], "decision_function", lambda value: np.full(len(value), np.nan))
    with pytest.raises(RuntimeError, match="LOGITS"):
        reference.predict_direction_probability_reference(x, state)


@pytest.mark.parametrize("fault", ["labels", "nan", "inf", "float32", "shape"])
def test_invalid_metrics_reject(fault):
    y = np.array([0, 1], dtype=np.int8)
    z = np.array([-1., 1.], dtype=np.float64)
    if fault == "labels": y[0] = -1
    elif fault == "nan": z[0] = np.nan
    elif fault == "inf": z[0] = np.inf
    elif fault == "float32": z = z.astype(np.float32)
    elif fault == "shape": y = y[:, None]
    with pytest.raises(RuntimeError, match="DIRECTION_PROBABILITY_REFERENCE_"):
        reference.direction_probability_metrics(y, z)


@pytest.mark.parametrize("scores,expected", [
    ([-2., -1., 1., 2.], (4, 0, 0, 1.)),
    ([2., 1., -1., -2.], (0, 4, 0, 0.)),
    ([0., 0., 0., 0.], (0, 0, 4, .5)),
])
def test_rank_exact_pairs(scores, expected):
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    result = reference.direction_rank_diagnostics(
        y, np.array(scores, dtype=np.float64), score_kind="logit_up")
    assert (result["concordant_pairs"], result["discordant_pairs"],
            result["tied_pairs"], result["auc"]) == expected
    assert result["up_down_pairs"] == 4
    assert result["untied_pairs"] + result["tied_pairs"] == 4


def test_rank_mixed_ties_and_signed_zero():
    result = reference.direction_rank_diagnostics(
        np.array([0, 1, 0, 1], dtype=np.int8),
        np.array([0., 0., 1., 2.]), score_kind="logit_up")
    assert result["distinct_scores"] == 3
    assert (result["concordant_pairs"], result["discordant_pairs"],
            result["tied_pairs"], result["auc"]) == (2, 1, 1, .625)
    zero = reference.direction_rank_diagnostics(
        np.array([0, 1], dtype=np.int8),
        np.array([-0., 0.]), score_kind="logit_up")
    assert zero["distinct_scores"] == zero["tied_pairs"] == 1
    assert zero["auc"] == .5


@pytest.mark.parametrize("labels", [[], [0, 0], [1, 1]])
def test_rank_empty_or_single_class_has_no_auc(labels):
    y = np.array(labels, dtype=np.int8)
    result = reference.direction_rank_diagnostics(
        y, np.zeros(len(y)), score_kind="logit_up")
    assert result["auc"] is None
    assert result["up_down_pairs"] == result["concordant_pairs"] == 0
    assert result["discordant_pairs"] == result["tied_pairs"] == 0


def test_rank_extreme_logits_and_readonly_inputs_remain_unchanged():
    limit = np.finfo(np.float64).max
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    scores = np.array([-limit, -1000., 1000., limit])
    before_y, before_scores = y.copy(), scores.copy()
    y.setflags(write=False)
    scores.setflags(write=False)
    result = reference.direction_rank_diagnostics(
        y, scores, score_kind="logit_up")
    assert result["auc"] == 1.
    np.testing.assert_array_equal(y, before_y)
    np.testing.assert_array_equal(scores, before_scores)
    assert not y.flags.writeable and not scores.flags.writeable


@pytest.mark.parametrize("fault", [
    "float32", "nan", "inf", "score_shape", "labels_dtype",
    "labels_domain", "labels_shape", "score_kind", "probability_domain",
    "scores_noncontiguous", "labels_noncontiguous",
])
def test_rank_invalid_inputs_reject(fault):
    y = np.array([0, 1], dtype=np.int8)
    scores = np.array([0., 1.])
    kind = "probability_up"
    if fault == "float32": scores = scores.astype(np.float32)
    elif fault == "nan": scores[0] = np.nan
    elif fault == "inf": scores[0] = np.inf
    elif fault == "score_shape": scores = scores[:, None]
    elif fault == "labels_dtype": y = y.astype(np.float64)
    elif fault == "labels_domain": y[0] = 2
    elif fault == "labels_shape": y = y[:, None]
    elif fault == "score_kind": kind = "unknown"
    elif fault == "probability_domain": scores[0] = -1.
    elif fault == "scores_noncontiguous": scores = np.array([0., 0., 1., 1.])[::2]
    elif fault == "labels_noncontiguous": y = np.array([0, 0, 1, 1], dtype=np.int8)[::2]
    with pytest.raises(RuntimeError, match="DIRECTION_PROBABILITY_REFERENCE_"):
        reference.direction_rank_diagnostics(y, scores, score_kind=kind)


def test_crossfold_prior_auc_does_not_imply_within_fold_ranking():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=np.int8)
    scores = np.array([.25] * 4 + [.75] * 4)
    folds = np.array([1] * 4 + [2] * 4, dtype=np.int8)
    before = (y.copy(), scores.copy(), folds.copy())
    result = reference.direction_within_fold_rank_diagnostics(
        y, scores, folds, score_kind="probability_up")
    assert result["pooled_crossfold_diagnostic"]["auc"] == .75
    assert result["pair_weighted_within_fold_auc"] == .5
    assert [x["diagnostics"]["auc"] for x in result["per_fold"]] == [.5, .5]
    assert result["within_fold_pair_counts"] == {
        "up_down_pairs": 6, "concordant_pairs": 0,
        "discordant_pairs": 0, "tied_pairs": 6}
    assert result["primary_scope"] == "per_fold"
    assert not result["pooled_crossfold_is_within_fold_evidence"]
    for actual, original in zip((y, scores, folds), before):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("folds", [
    np.array([1., 2.]), np.array([-1, 2]), np.array([1]),
    np.array([1, 0, 2, 0], dtype=np.int8)[::2],
])
def test_rank_invalid_fold_headers_reject(folds):
    with pytest.raises(RuntimeError, match="RANK_FOLD_HEADER"):
        reference.direction_within_fold_rank_diagnostics(
            np.array([0, 1], dtype=np.int8), np.array([0., 1.]),
            folds, score_kind="logit_up")


def test_rank_empty_within_fold_aggregation():
    result = reference.direction_within_fold_rank_diagnostics(
        np.array([], dtype=np.int8), np.array([], dtype=np.float64),
        np.array([], dtype=np.int8), score_kind="logit_up")
    assert result["per_fold"] == []
    assert result["pair_weighted_within_fold_auc"] is None
    assert result["pooled_crossfold_diagnostic"]["auc"] is None


def test_rank_single_class_folds_do_not_create_within_fold_evidence():
    result = reference.direction_within_fold_rank_diagnostics(
        np.array([0, 0, 1, 1], dtype=np.int8),
        np.array([.25, .25, .75, .75]),
        np.array([1, 1, 2, 2], dtype=np.int8), score_kind="probability_up")
    assert result["pooled_crossfold_diagnostic"]["auc"] == 1.
    assert result["pair_weighted_within_fold_auc"] is None
    assert all(x["diagnostics"]["auc"] is None for x in result["per_fold"])


def test_rank_unsorted_tie_pairs_are_permutation_invariant():
    y = np.array([0, 1, 0, 1], dtype=np.int8)
    scores = np.array([0., 0., 1., 2.])
    before = reference.direction_rank_diagnostics(y, scores, score_kind="logit_up")
    order = np.array([3, 0, 2, 1])
    after = reference.direction_rank_diagnostics(
        y[order], scores[order], score_kind="logit_up")
    assert before == after



def test_rank_within_fold_uses_opposite_pair_weights_not_fold_mean():
    result = reference.direction_within_fold_rank_diagnostics(
        np.array([0, 1, 0, 0, 1, 1], dtype=np.int8),
        np.array([0., 1., 1., 1., 0., 0.]),
        np.array([1, 1, 2, 2, 2, 2], dtype=np.int8), score_kind="logit_up")
    assert [x["diagnostics"]["auc"] for x in result["per_fold"]] == [1., 0.]
    assert result["within_fold_pair_counts"] == {
        "up_down_pairs": 5, "concordant_pairs": 1,
        "discordant_pairs": 4, "tied_pairs": 0}
    assert result["pair_weighted_within_fold_auc"] == .2


def test_rank_opposite_pair_counts_exceed_signed_int32_without_overflow():
    labels = np.repeat(np.array([0, 1], dtype=np.int8), 50_000)
    result = reference.direction_rank_diagnostics(
        labels, np.zeros(len(labels)), score_kind="logit_up")
    assert result["up_down_pairs"] == result["tied_pairs"] == 2_500_000_000
    assert type(result["up_down_pairs"]) is type(result["tied_pairs"]) is int
    assert result["concordant_pairs"] == result["discordant_pairs"] == 0
    assert result["auc"] == .5


def test_rank_within_fold_preserves_readonly_labels_scores_and_folds():
    labels = np.array([0, 1, 0, 1], dtype=np.int8)
    scores = np.array([0., 1., 1., 0.])
    folds = np.array([1, 1, 2, 2], dtype=np.int8)
    originals = [a.copy() for a in (labels, scores, folds)]
    for array in (labels, scores, folds):
        array.setflags(write=False)
    result = reference.direction_within_fold_rank_diagnostics(
        labels, scores, folds, score_kind="logit_up")
    assert result["pair_weighted_within_fold_auc"] == .5
    for array, original in zip((labels, scores, folds), originals):
        np.testing.assert_array_equal(array, original)
        assert not array.flags.writeable


def test_rank_single_class_fold_does_not_change_valid_fold_pair_denominator():
    result = reference.direction_within_fold_rank_diagnostics(
        np.array([0, 1, 1, 1], dtype=np.int8),
        np.array([0., 1., -10., 10.]),
        np.array([1, 1, 2, 2], dtype=np.int8), score_kind="logit_up")
    assert [x["diagnostics"]["auc"] for x in result["per_fold"]] == [1., None]
    assert result["within_fold_pair_counts"]["up_down_pairs"] == 1
    assert result["pair_weighted_within_fold_auc"] == 1.
