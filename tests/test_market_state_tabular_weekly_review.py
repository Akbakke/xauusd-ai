"""Synthetic paired week review; no project data, fit or launch."""
import numpy as np
import pytest

from gx1.contracts.market_state_tabular_weekly_review_v1 import paired_m5_weekly_review

SUNDAYS = ("2024-10-06", "2025-01-05", "2025-04-06", "2025-07-06", "2025-10-05")


def _fold(fold_id: int, *, correct: bool = True):
    sunday = np.datetime64(SUNDAYS[fold_id - 1] + "T23:55:00", "ns").astype(np.int64)
    decisions = sunday + np.arange(6, dtype=np.int64) * 300_000_000_000
    allowed = np.full((6, 5), 7, dtype=np.uint8)
    allowed[:, 0] = (1, 4, 5, 7, 1, 4)
    eligible = np.ones((6, 5), dtype=np.bool_)
    mature = np.ones_like(eligible)
    eligible[4, 0] = False
    mature[5, 0] = False
    prior = np.empty((6, 5, 3), dtype=np.float64)
    prior[:] = (0.4, 0.2, 0.4)
    model = prior.copy()
    if correct:
        model[0, 0, :] = (0.64, 0.2, 0.16)  # DOWN, conditional UP 0.2
        model[1, 0, :] = (0.16, 0.2, 0.64)  # UP, conditional UP 0.8
    else:
        model[0, 0, :] = (0.16, 0.2, 0.64)
        model[1, 0, :] = (0.64, 0.2, 0.16)
    # These unscored cells may look terrible; the masks must exclude them.
    model[2:, 0, :] = (0.79, 0.2, 0.01)
    return {"fold_id": fold_id, "decision_ns": decisions,
            "allowed": allowed, "eligible": eligible, "mature": mature,
            "model_probabilities": model, "prior_probabilities": prior}


def test_positive_gate_uses_only_paired_certified_side_and_utc_week_blocks():
    result = paired_m5_weekly_review([_fold(i) for i in range(1, 6)])
    assert result["side_count"] == 10
    assert result["positive_fold_count"] == 5
    assert result["positive_train_diagnostic"] is True
    assert result["decision"] == "POSITIVE_DEVELOPMENT_TRAIN_DIAGNOSTIC"
    assert result["pooled_delta"] == pytest.approx(0.21)
    assert result["bootstrap"]["lower"] == pytest.approx(0.21)
    assert result["bootstrap"]["upper"] == pytest.approx(0.21)
    assert all(fold["week_count"] == 2 for fold in result["folds"])
    assert all(fold["week_ids"][1] == fold["week_ids"][0] + 1
               for fold in result["folds"])


def test_negative_and_four_positive_fold_gate_are_distinct():
    negative = paired_m5_weekly_review([_fold(i, correct=False) for i in range(1, 6)])
    assert negative["positive_train_diagnostic"] is False
    assert negative["pooled_delta"] == pytest.approx(-0.39)
    folds = [_fold(i) for i in range(1, 6)]
    folds[-1] = _fold(5, correct=False)
    mixed = paired_m5_weekly_review(folds)
    assert mixed["positive_fold_count"] == 4
    assert mixed["pooled_delta"] > 0
    assert mixed["positive_train_diagnostic"] is True
    folds[-2] = _fold(4, correct=False)
    assert paired_m5_weekly_review(folds)["positive_train_diagnostic"] is False


@pytest.mark.parametrize("fault", ["bit", "axis", "probability", "eligibility", "chronology"])
def test_malformed_or_unpaired_inputs_fail_closed(fault):
    folds = [_fold(i) for i in range(1, 6)]
    if fault == "bit":
        folds[0]["allowed"][0, 0] = 3
    elif fault == "axis":
        folds[0]["decision_ns"][1] = folds[0]["decision_ns"][0]
    elif fault == "probability":
        folds[0]["model_probabilities"][0, 0, :] = (0.1, 0.2, 0.8)
    elif fault == "eligibility":
        folds[0]["eligible"] = folds[0]["eligible"].astype(np.uint8)
    else:
        folds[1]["decision_ns"] = folds[0]["decision_ns"].copy()
    with pytest.raises(ValueError, match="MARKET_STATE_TABULAR_WEEKLY_"):
        paired_m5_weekly_review(folds)
