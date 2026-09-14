from __future__ import annotations

import numpy as np
import pytest
import torch

from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


def _forecast_arrays():
    target = np.tile(np.asarray([-2.0, -1.0, 0.0, 2.0])[:, None], (1, 4))
    prediction = np.column_stack(([-1.0, 1.0, 0.0, 1.0], [1.0, 2.0, -1.0, -2.0], target[:, 2], target[:, 3]))
    mask = np.ones_like(target, dtype=bool)
    mask[0, 3] = False
    prediction[0, 3] = np.nan  # An unsupervised cell cannot affect any metric.
    return prediction, target, mask


def _metrics(prediction, target, mask):
    return trainer._active_head_component_validation_metrics(
        component_name="forecast_pred",
        prediction=prediction,
        target=target,
        element_mask=mask,
    )


def test_forecast_horizons_keep_sign_errors_masks_and_gross_return_semantics():
    prediction, target, mask = _forecast_arrays()
    metrics = _metrics(prediction, target, mask)
    # The pre-existing flattened aggregate remains unchanged.
    assert metrics["mean_squared_error"] == pytest.approx(np.mean((prediction[mask] - target[mask]) ** 2))
    evidence = metrics["forecast_horizon_diagnostics"]
    assert evidence["target_semantics"] == "observed_future_m5_close_return_bps_before_execution_costs"
    assert evidence["horizon_semantics"] == "observed_m5_bars_nominal_minutes_not_elapsed_wall_clock"
    assert evidence["exit_teacher_used"] is False
    assert evidence["used_for_entry_selection"] is False
    assert evidence["sign_order"] == ["down", "unchanged", "up"]
    horizons = evidence["horizons"]
    assert [item["target_column"] for item in horizons] == ["y_forecast_ret_K1", "y_forecast_ret_K5", "y_forecast_ret_K12", "y_forecast_ret_K24"]
    assert [item["nominal_horizon_minutes"] for item in horizons] == [5, 25, 60, 120]
    assert [item["supervised_rows"] for item in horizons] == [4, 4, 4, 3]
    short = horizons[0]
    assert short["mean_prediction_minus_target_bps"] == pytest.approx(0.5)
    assert short["regression_metrics"]["mean_absolute_error"] == pytest.approx(1.0)
    assert short["regression_metrics"]["mean_squared_error"] == pytest.approx(1.5)
    assert short["zero_return_baseline_mean_absolute_error_bps"] == pytest.approx(1.25)
    assert short["zero_return_baseline_mean_squared_error_bps2"] == pytest.approx(2.25)
    assert short["sign_confusion_actual_rows_predicted_columns"] == [[1, 0, 1], [0, 1, 0], [0, 0, 1]]
    assert short["actual_nonzero_rows"] == short["predicted_nonzero_rows"] == 3
    assert short["direction_accuracy_on_nonzero_targets"] == pytest.approx(2.0 / 3.0)
    assert horizons[1]["direction_accuracy_on_nonzero_targets"] == 0.0
    assert horizons[2]["regression_metrics"]["pearson"] == pytest.approx(1.0)
    assert horizons[2]["regression_metrics"]["mean_squared_error"] == 0.0


def test_forecast_zero_returns_do_not_fabricate_direction_quality():
    prediction = np.tile(np.asarray([-1.0, 0.0, 1.0])[:, None], (1, 4))
    target = np.zeros_like(prediction)
    evidence = _metrics(prediction, target, np.ones_like(target, dtype=bool))["forecast_horizon_diagnostics"]
    for horizon in evidence["horizons"]:
        assert horizon["direction_accuracy_on_nonzero_targets"] is None
        assert horizon["actual_nonzero_rows"] == 0
        assert horizon["predicted_nonzero_rows"] == 2
        assert horizon["sign_confusion_actual_rows_predicted_columns"] == [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        assert horizon["regression_metrics"]["pearson"] is None


def test_forecast_rejects_supervised_nonfinite_prediction():
    prediction, target, mask = _forecast_arrays()
    mask[0, 3] = True
    with pytest.raises(RuntimeError, match="ENTRY_ACTIVE_HEAD_DIAGNOSTIC_NONFINITE"):
        _metrics(prediction, target, mask)


def test_forecast_horizon_evidence_is_identical_after_segmented_snapshot_resume(monkeypatch):
    prediction, target, mask = _forecast_arrays()

    def target_surfaces(out, batch, device):
        indices = batch["entry_row_index"].numpy()
        surfaces = {}
        for head, components in trainer._ACTIVE_HEAD_TARGET_COMPONENTS.items():
            surfaces[head] = {}
            for component in components:
                width = trainer._ACTIVE_HEAD_COMPONENT_WIDTHS[component]
                if component == "forecast_pred":
                    values = prediction[indices], target[indices], mask[indices]
                else:
                    p = np.repeat(indices.astype(np.float32)[:, None], width, axis=1)
                    t = p + 1.0
                    for column in trainer._ACTIVE_HEAD_STRUCTURAL_CONSTANT_COLUMNS.get(component, ()):
                        t[:, column] = 0.0
                    values = p, t, np.ones_like(p, dtype=bool)
                surfaces[head][component] = tuple(torch.as_tensor(value) for value in values)
        return surfaces

    monkeypatch.setattr(trainer, "_active_head_target_surfaces", target_surfaces)

    def accumulate(accumulator, indices):
        trainer._accumulate_active_head_epoch(
            accumulator, None, {}, {"entry_row_index": torch.tensor(indices)}, torch.device("cpu")
        )

    uninterrupted = trainer._new_active_head_epoch_accumulator()
    accumulate(uninterrupted, [0, 1, 2, 3])
    resumed = trainer._new_active_head_epoch_accumulator()
    accumulate(resumed, [0, 1])
    resumed = trainer._candidate_snapshot_restore(trainer._candidate_snapshot_safe(resumed))
    accumulate(resumed, [2, 3])
    full_metrics, full_failures = trainer._active_head_epoch_diagnostics(uninterrupted, minimum_supervised_rows=2)
    resumed_metrics, resumed_failures = trainer._active_head_epoch_diagnostics(resumed, minimum_supervised_rows=2)
    assert full_failures == resumed_failures == []
    assert full_metrics == resumed_metrics
    assert resumed_metrics["active_head_diagnostic_schema"] == "entry_model_native_active_head_epoch_diagnostics_v6"
    horizons = resumed_metrics["active_head_diagnostics"]["forecast"]["components"]["forecast_pred"]["validation_metrics"]["forecast_horizon_diagnostics"]["horizons"]
    assert [row["supervised_rows"] for row in horizons] == [4, 4, 4, 3]


def test_forecast_empty_horizon_reports_unavailable_without_nan():
    import json
    import warnings

    prediction, target, mask = _forecast_arrays()
    mask[:, 3] = False
    prediction[:, 3] = np.nan
    target[:, 3] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        metrics = _metrics(prediction, target, mask)
    # No NaN/Infinity may enter the hashed native VAL result.
    json.dumps(metrics, allow_nan=False)
    empty = metrics["forecast_horizon_diagnostics"]["horizons"][3]
    assert empty["supervised_rows"] == 0
    assert empty["regression_metrics"] is None
    assert empty["mean_prediction_minus_target_bps"] is None
    assert empty["zero_return_baseline_mean_absolute_error_bps"] is None
    assert empty["zero_return_baseline_mean_squared_error_bps2"] is None
    assert empty["direction_accuracy_on_nonzero_targets"] is None
    assert empty["sign_confusion_actual_rows_predicted_columns"] == [[0, 0, 0]] * 3
    assert empty["actual_nonzero_rows"] == empty["predicted_nonzero_rows"] == 0
    assert metrics["forecast_horizon_diagnostics"]["horizons"][0]["supervised_rows"] == 4
