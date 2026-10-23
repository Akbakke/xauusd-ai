"""Analytical213 capacity-reference checks; synthetic values, no market inputs."""
from copy import deepcopy

import numpy as np
import pytest

from gx1.contracts import entry_direction_snapshot_capacity_reference_v1 as capacity
from gx1.contracts import entry_direction_snapshot_reference_v1 as legacy


@pytest.fixture
def small_chunks(monkeypatch):
    from gx1.scripts import research_entry_direction_walkforward_v1 as owners

    monkeypatch.setattr(owners, "GRAM_CHUNK_ROWS", 2)


def _analytic_case():
    # Own horizon rows are all four, the first three, and rows1/3.
    x = np.array([[0, 1], [1, 1], [2, 1], [3, 1]], dtype=np.float32)
    y = np.array(
        [[7, 4, np.nan], [9, 3, -2], [11, 2, np.nan], [13, np.nan, 4]],
        dtype=np.float64,
    )
    observed = np.array(
        [[True, True, False], [True, True, True],
         [True, True, False], [True, False, True]],
        dtype=bool,
    )
    return x, y, observed


def _fixed_state():
    # Literal frozen prediction fixture; no fitted or genuine-data claim.
    betas = ([1., -2.], [0., 0.], [-1., 1.])
    means, constants, intercepts, counts = [1., 2.], (3., 4., 5.), (6., 4., 4.), (4, 3, 2)
    return {
        "schema_version": capacity.SCHEMA_VERSION + "_ridge",
        "horizons_minutes": (1, 5, 15),
        "mean_sse_lambda": 1.0,
        "objective": "mean_SSE_plus_lambda_beta_squared_free_intercept",
        "feature_width": 2,
        "feature_scale": "ones_in_supplied_snapshot_coordinates",
        "fit_partition": "train",
        "fits": {
            h: {
                "beta": np.array(beta, dtype=np.float64),
                "feature_mean": np.array(means, dtype=np.float64),
                "target_mean_bps": constant,
                "intercept_bps": intercept,
                "observed_train_rows": count,
                "sum_sse_alpha": float(count),
                "solver_residual_l2": 0.,
                "rhs_l2": 1.,
                "solver_relative_residual": 0.,
            }
            for h, beta, constant, intercept, count in zip(
                (1, 5, 15), betas, constants, intercepts, counts
            )
        },
    }


def test_hand_derived_own_horizon_solution_and_honest_capacity_state(small_chunks):
    x, y, observed = _analytic_case()
    before = x.copy(), y.copy(), observed.copy()
    state = capacity.fit_direction_snapshot_capacity_ridge(x, y, observed, split="train")

    # Centered Sxx=(5,2,2), Sxy=(10,-2,6), and alpha=N=(4,3,2).
    # Thus beta_x=(10/9,-2/5,3/2); the constant-column coefficient is zero.
    slopes, intercepts = (10/9, -2/5, 3/2), (25/3, 17/5, -2)
    x_means, y_means, counts = (1.5, 1., 2.), (10., 3., 1.), (4, 3, 2)
    assert capacity.SCHEMA_VERSION == "gx1_direction_snapshot_capacity_reference_v1"
    assert state["schema_version"] == capacity.SCHEMA_VERSION + "_ridge"
    assert state["schema_version"] != legacy.SCHEMA_VERSION + "_ridge"
    assert state["mean_sse_lambda"] == 1.0 and "alpha" not in state
    assert state["objective"] == "mean_SSE_plus_lambda_beta_squared_free_intercept"
    assert state["feature_scale"] == "ones_in_supplied_snapshot_coordinates"
    assert state["feature_width"] == 2 and state["fit_partition"] == "train"
    assert tuple(state["horizons_minutes"]) == (1, 5, 15)
    assert set(state["fits"]) == {1, 5, 15}
    for h, slope, intercept, xm, ym, count in zip(
        (1, 5, 15), slopes, intercepts, x_means, y_means, counts
    ):
        fit = state["fits"][h]
        assert fit["beta"].dtype == fit["feature_mean"].dtype == np.float64
        np.testing.assert_allclose(fit["beta"], [slope, 0.], rtol=1e-13, atol=1e-13)
        np.testing.assert_array_equal(fit["feature_mean"], [xm, 1.])
        assert fit["target_mean_bps"] == ym
        assert fit["intercept_bps"] == pytest.approx(intercept, rel=1e-13, abs=1e-13)
        assert type(fit["observed_train_rows"]) is int
        assert fit["observed_train_rows"] == count
        assert type(fit["sum_sse_alpha"]) is float and fit["sum_sse_alpha"] == float(count)

    prediction = capacity.predict_direction_snapshot_capacity_ridge(x, state)
    expected = np.array(
        [[25/3, 17/5, -2.], [85/9, 3., -.5],
         [95/9, 13/5, 1.], [35/3, 11/5, 2.5]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(prediction, expected, rtol=1e-13, atol=1e-13)
    for actual, original in zip((x, y, observed), before):
        np.testing.assert_array_equal(actual.view(np.uint8), original.view(np.uint8))


@pytest.mark.parametrize("copies", (2, 3))
def test_uniform_row_duplication_preserves_mean_sse_solution_and_own_masks(small_chunks, copies):
    x, y, observed = _analytic_case()
    state = capacity.fit_direction_snapshot_capacity_ridge(x, y, observed, split="train")
    repeated = capacity.fit_direction_snapshot_capacity_ridge(
        np.repeat(x, copies, axis=0), np.repeat(y, copies, axis=0),
        np.repeat(observed, copies, axis=0), split="train",
    )
    for h in (1, 5, 15):
        original, duplicate = state["fits"][h], repeated["fits"][h]
        for key in ("beta", "feature_mean", "target_mean_bps", "intercept_bps"):
            np.testing.assert_allclose(duplicate[key], original[key], rtol=1e-13, atol=1e-13)
        assert duplicate["observed_train_rows"] == copies * original["observed_train_rows"]
        assert duplicate["sum_sse_alpha"] == copies * original["sum_sse_alpha"]
    np.testing.assert_allclose(
        capacity.predict_direction_snapshot_capacity_ridge(x, repeated),
        capacity.predict_direction_snapshot_capacity_ridge(x, state),
        rtol=1e-13, atol=1e-13,
    )


def test_free_intercept_is_unpenalized_and_constant_columns_carry_no_slope(small_chunks):
    x = np.tile(np.array([[1., 7.]], dtype=np.float32), (4, 1))
    y = np.tile(np.array([[7., -3., 2.]], dtype=np.float64), (4, 1))
    observed = np.array(
        [[True, True, False], [True, True, True],
         [True, True, False], [True, False, True]], dtype=bool,
    )
    y[~observed] = np.nan
    state = capacity.fit_direction_snapshot_capacity_ridge(x, y, observed, split="train")
    for h, constant in zip((1, 5, 15), (7., -3., 2.)):
        fit = state["fits"][h]
        np.testing.assert_array_equal(fit["beta"], [0., 0.])
        assert fit["target_mean_bps"] == fit["intercept_bps"] == constant
    evaluation = np.array([[99., -99.], [0., 1000.]], dtype=np.float32)
    np.testing.assert_array_equal(
        capacity.predict_direction_snapshot_capacity_ridge(evaluation, state),
        np.array([[7., -3., 2.], [7., -3., 2.]], dtype=np.float64),
    )


def test_unobserved_nan_storage_is_preserved_and_never_becomes_fit_evidence(small_chunks):
    x, y, observed = _analytic_case()
    before_y, before_observed = y.copy(), observed.copy()
    missing = capacity.fit_direction_snapshot_capacity_ridge(x, y, observed, split="train")
    alternative_storage = y.copy()
    alternative_storage[~observed] = 1e200
    alternative = capacity.fit_direction_snapshot_capacity_ridge(
        x, alternative_storage, observed, split="train",
    )
    assert np.isnan(y[~observed]).all()
    np.testing.assert_array_equal(y.view(np.uint64), before_y.view(np.uint64))
    np.testing.assert_array_equal(observed, before_observed)
    for h in (1, 5, 15):
        for key in ("beta", "feature_mean", "target_mean_bps", "intercept_bps"):
            np.testing.assert_array_equal(missing["fits"][h][key], alternative["fits"][h][key])


@pytest.mark.parametrize(
    "fault",
    ("split", "empty_horizon", "observed_nan", "observed_inf", "target_float32",
     "mask_integer", "matrix_float64", "feature_nan", "target_shape"),
)
def test_invalid_fit_domains_fail_closed(small_chunks, fault):
    x, y, observed = _analytic_case()
    split = "train"
    if fault == "split":
        split = "control"
    elif fault == "empty_horizon":
        observed[:, 0] = False
    elif fault == "observed_nan":
        y[0, 0] = np.nan
    elif fault == "observed_inf":
        y[0, 0] = np.inf
    elif fault == "target_float32":
        y = y.astype(np.float32)
    elif fault == "mask_integer":
        observed = observed.astype(np.int64)
    elif fault == "matrix_float64":
        x = x.astype(np.float64)
    elif fault == "feature_nan":
        x[0, 0] = np.nan
    else:
        y = y[:, :2]
    with pytest.raises(RuntimeError):
        capacity.fit_direction_snapshot_capacity_ridge(x, y, observed, split=split)


@pytest.mark.parametrize(
    "fault",
    ("old_schema", "lambda", "count_zero", "count_float", "sum_sse_alpha",
     "objective", "feature_scale", "feature_width", "horizon_order",
     "beta_float32", "mean_float32", "intercept"),
)
def test_predictor_rejects_incompatible_or_malformed_saved_state(fault):
    state = deepcopy(_fixed_state())
    if fault == "old_schema":
        state["schema_version"] = legacy.SCHEMA_VERSION + "_ridge"
    elif fault == "lambda":
        state["mean_sse_lambda"] = 2.0
    elif fault == "count_zero":
        state["fits"][1]["observed_train_rows"] = 0
    elif fault == "count_float":
        state["fits"][1]["observed_train_rows"] = 4.0
    elif fault == "sum_sse_alpha":
        state["fits"][1]["sum_sse_alpha"] = 1.0
    elif fault == "objective":
        state["objective"] = "sum_SSE_plus_alpha_beta_squared_free_intercept"
    elif fault == "feature_scale":
        state["feature_scale"] = "ones_after_frozen055"
    elif fault == "feature_width":
        state["feature_width"] = 3
    elif fault == "horizon_order":
        state["horizons_minutes"] = (5, 1, 15)
    elif fault == "beta_float32":
        state["fits"][1]["beta"] = state["fits"][1]["beta"].astype(np.float32)
    elif fault == "mean_float32":
        state["fits"][1]["feature_mean"] = state["fits"][1]["feature_mean"].astype(np.float32)
    else:
        state["fits"][1]["intercept_bps"] += 1.0
    with pytest.raises(RuntimeError):
        capacity.predict_direction_snapshot_capacity_ridge(
            np.array([[1., 2.]], dtype=np.float32), state,
        )


def test_frozen_literal_prediction_is_exact_whole_or_chunked(small_chunks):
    state = _fixed_state()
    x = np.array([[1., 2.], [3., 4.], [-1., 2.],
                  [1., -2.], [5., 6.], [0., 0.]], dtype=np.float32)
    expected = np.array(
        [[3., 4., 5.], [1., 4., 5.], [1., 4., 7.],
         [11., 4., 1.], [-1., 4., 5.], [6., 4., 4.]], dtype=np.float64,
    )
    whole = capacity.predict_direction_snapshot_capacity_ridge(x, state)
    joined = np.vstack([
        capacity.predict_direction_snapshot_capacity_ridge(x[:1], state),
        capacity.predict_direction_snapshot_capacity_ridge(x[1:4], state),
        capacity.predict_direction_snapshot_capacity_ridge(x[4:], state),
    ])
    assert whole.dtype == joined.dtype == np.float64
    np.testing.assert_array_equal(whole.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(joined.view(np.uint64), expected.view(np.uint64))
