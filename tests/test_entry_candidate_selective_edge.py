from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    PREDICTION_EVIDENCE_STAGE_SPLITS,
)
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_DIP_OUTPUT_DIM,
    MODEL_NATIVE_FORECAST_TARGET_COLUMNS,
    MODEL_NATIVE_TAIL_RISK_TARGET_COLUMNS,
    MODEL_NATIVE_TIMING_OUTPUT_DIM,
    MODEL_NATIVE_VOL_FORECAST_TARGET_COLUMNS,
)
import torch

from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
)
from gx1.scripts.evaluate_entry_candidate_selective_edge_v1 import (
    EVALUATION_COVERAGES,
    _EXTRA_VECTOR_HEADS,
    _append_extra_vector_head_evidence,
    _canonical_live_decision_evidence,
    _concatenate_evidence_chunks,
    _preregistered_hypothesis,
    _research_policy_pnl,
    _require_entry_q_ssot,
    _require_selective_edge_stage_split,
    _selection_sort_column,
    build_metric_rows,
)


def test_vector_evidence_widths_match_model_output_owners() -> None:
    """A producer must not substitute physical target count for head width."""

    assert _EXTRA_VECTOR_HEADS == {
        "dip_pred": MODEL_NATIVE_DIP_OUTPUT_DIM,
        "forecast_pred": len(MODEL_NATIVE_FORECAST_TARGET_COLUMNS),
        "timing_pred": MODEL_NATIVE_TIMING_OUTPUT_DIM,
        "tail_risk_pred": len(MODEL_NATIVE_TAIL_RISK_TARGET_COLUMNS),
        "vol_forecast_pred": len(MODEL_NATIVE_VOL_FORECAST_TARGET_COLUMNS),
    }


def test_vector_head_evidence_is_persisted_as_exact_dense_vectors() -> None:
    """Prediction evidence must retain head names, not split them into scalars."""

    outputs = {
        name: torch.full((2, width), float(index + 1), dtype=torch.float32)
        for index, (name, width) in enumerate(_EXTRA_VECTOR_HEADS.items())
    }
    chunks: dict[str, list[np.ndarray]] = {}
    _append_extra_vector_head_evidence(chunks, outputs)
    combined = _concatenate_evidence_chunks(chunks, expected_rows=2)

    assert set(combined) == set(_EXTRA_VECTOR_HEADS)
    for name, width in _EXTRA_VECTOR_HEADS.items():
        assert combined[name].shape == (2, width)
    assert not any(name.rsplit("_", 1)[0] in _EXTRA_VECTOR_HEADS for name in combined)


def test_entry_q_is_the_only_decision_surface_and_ties_fail_closed() -> None:
    q = torch.tensor(
        [[3.0, -1.0, 0.0], [-2.0, 4.0, 0.0], [-1.0, -2.0, 0.0]],
        dtype=torch.float32,
    )
    assert torch.equal(_require_entry_q_ssot({"entry_action_q_bps": q}), q)
    with pytest.raises(RuntimeError, match="forbidden legacy"):
        _require_entry_q_ssot(
            {"entry_action_q_bps": q, "anchor_logits": torch.zeros_like(q)}
        )
    with pytest.raises(RuntimeError, match="no unique top action"):
        _require_entry_q_ssot(
            {"entry_action_q_bps": torch.tensor([[1.0, 1.0, 0.0]])}
        )


def test_live_decision_evidence_contains_raw_q_argmax_and_no_probabilities() -> None:
    q = torch.tensor(
        [[3.0, -1.0, 0.0], [-2.0, 4.0, 0.0], [-1.0, -2.0, 0.0]],
        dtype=torch.float32,
    )
    evidence = _canonical_live_decision_evidence(
        {"entry_action_q_bps": q}
    )
    assert set(evidence) == {
        "entry_action_q_bps",
        "entry_action_q_margin_bps",
        "model_direction_index",
        "edge_score",
        "selection_score",
    }
    assert evidence["model_direction_index"].tolist() == [0, 1, 2]
    assert np.array_equal(evidence["entry_action_q_bps"], q.numpy())
    assert np.all(evidence["entry_action_q_margin_bps"] > 0.0)


def test_research_policy_pnl_uses_action_side_and_flat_zero_without_claiming_net() -> None:
    frame = pd.DataFrame(
        {
            "pred_direction": [0, 1, 2],
            "y_long_final_pnl_at_direction_horizon_bps": [5.0, 99.0, 99.0],
            "y_short_final_pnl_at_direction_horizon_bps": [99.0, -3.0, 99.0],
        }
    )
    assert _research_policy_pnl(frame).tolist() == [5.0, -3.0, 0.0]
    with pytest.raises(RuntimeError, match="gross spread-inclusive research outcomes"):
        _research_policy_pnl(
            frame.drop(columns=["y_long_final_pnl_at_direction_horizon_bps"])
        )


def test_selection_sort_is_raw_q_action_value_and_mode_bound() -> None:
    frame = pd.DataFrame(
        {
            "selection_score": [3.0, 2.0],
            "selection_score_mode": [MODEL_DIRECTION_SELECTION_MODE] * 2,
        }
    )
    assert _selection_sort_column(frame) == "selection_score"
    frame.loc[0, "selection_score_mode"] = "probability"
    with pytest.raises(RuntimeError, match="direction mode mismatch"):
        _selection_sort_column(frame)


def test_evidence_chunks_require_exact_row_count_and_shape() -> None:
    combined = _concatenate_evidence_chunks(
        {
            "entry_action_q_bps": [
                np.ones((2, 3), dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32),
            ]
        },
        expected_rows=3,
    )
    assert combined["entry_action_q_bps"].shape == (3, 3)
    with pytest.raises(RuntimeError, match="row mismatch"):
        _concatenate_evidence_chunks(
            {"entry_action_q_bps": [np.ones((2, 3), dtype=np.float32)]},
            expected_rows=3,
        )


def test_preregistered_metrics_use_fixed_grid_and_autocorrelation_null() -> None:
    """A synthetic time-linked signal clears both fixed primary nulls.

    Direction is deliberately attached to the better side on each row; a
    non-zero circular label shift breaks that time alignment, while an iid
    coin-flip has expectation zero.  This exercises the actual two-part gate,
    not only a convenience mean-PnL calculation.
    """

    rng = np.random.default_rng(71)
    rows = 2_048
    state = rng.choice(np.array([-1.0, 1.0]), size=rows)
    long_outcome = state * 5.0 + rng.normal(0.0, 0.1, size=rows)
    short_outcome = -state * 5.0 + rng.normal(0.0, 0.1, size=rows)
    direction = np.where(state > 0.0, 0, 1)
    frame = pd.DataFrame(
        {
            "split": "val",
            "model": "candidate",
            "time": pd.date_range("2025-06-01", periods=rows, freq="5min", tz="UTC"),
            "pred_direction": direction,
            "selection_score": np.linspace(float(rows), 1.0, rows),
            "selection_score_mode": [MODEL_DIRECTION_SELECTION_MODE] * rows,
            "edge_score": np.ones(rows),
            "y_long_final_pnl_at_direction_horizon_bps": long_outcome,
            "y_short_final_pnl_at_direction_horizon_bps": short_outcome,
        }
    )
    metrics = pd.DataFrame(
        build_metric_rows(frame, top_fracs=list(EVALUATION_COVERAGES))
    )
    assert metrics["coverage_fraction"].tolist() == list(EVALUATION_COVERAGES)
    assert metrics.loc[metrics["coverage_fraction"] == 0.25, "primary_pass"].item()
    assert set(metrics["coin_flip_null_method"]) == {
        "exact_uniform_long_short_expectation"
    }
    hypothesis = _preregistered_hypothesis(
        metrics,
        evidence_stage="pre_calibration",
        val_reference=None,
    )
    assert hypothesis["decision"] == "PASS"
    assert 0.25 in hypothesis["qualifying_coverages"]


@pytest.mark.parametrize(
    ("stage", "split"),
    tuple(
        (stage, splits[0])
        for stage, splits in PREDICTION_EVIDENCE_STAGE_SPLITS.items()
    ),
)
def test_stage_split_contract_is_exact(stage: str, split: str) -> None:
    _require_selective_edge_stage_split(
        evidence_stage=stage, split_spec=split
    )
    wrong = "test" if split == "val" else "val"
    with pytest.raises(RuntimeError, match="SELECTIVE_EDGE_STAGE_SPLIT_INVALID"):
        _require_selective_edge_stage_split(
            evidence_stage=stage, split_spec=wrong
        )
