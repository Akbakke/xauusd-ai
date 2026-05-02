"""Tests for materialize_audit_feature_stability_across_folds_v1."""
from __future__ import annotations

import pytest

from gx1.scripts import materialize_audit_feature_stability_across_folds_v1 as gate


def test_classify_stable_when_same_sign_and_low_variance() -> None:
    out = gate.classify_feature_stability("X", [1.0, 1.1, 0.9])
    assert out["classification_v1"] == "STABLE"
    assert out["recommendation_v1"] == "KEEP"
    assert out["same_sign_across_folds_v1"] is True


def test_classify_directional_when_same_sign_high_variance() -> None:
    # mean 1.0, std ~ 0.82, 0.5*|mean| = 0.5; std > 0.5 -> directional
    out = gate.classify_feature_stability("X", [0.1, 0.5, 2.4])
    assert out["classification_v1"] == "DIRECTIONAL"
    assert out["recommendation_v1"] == "KEEP_BUT_NOTE_HIGH_VARIANCE"


def test_classify_flips_sign_when_signs_disagree() -> None:
    out = gate.classify_feature_stability("X", [1.0, -0.5, 0.8])
    assert out["classification_v1"] == "FLIPS_SIGN"
    assert out["recommendation_v1"] == "REGIME_CONDITION_OR_DROP"


def test_classify_dead_when_all_near_zero() -> None:
    out = gate.classify_feature_stability("X", [1e-9, -1e-9, 0.0])
    assert out["classification_v1"] == "DEAD"
    assert out["recommendation_v1"] == "DROP"


def test_aggregate_skip_v2_stability_groups_by_feature() -> None:
    per_fold = [
        {"fold_id_v1": "F1", "feature_names_v1": ["a", "b"], "coefs_v1": [1.0, -1.0]},
        {"fold_id_v1": "F2", "feature_names_v1": ["a", "b"], "coefs_v1": [1.1, -0.9]},
        {"fold_id_v1": "F3", "feature_names_v1": ["a", "b"], "coefs_v1": [1.2, -1.1]},
    ]
    rows = gate._aggregate_skip_v2_stability(per_fold)
    assert len(rows) == 2
    by_feat = {r["feature_v1"]: r for r in rows}
    assert by_feat["a"]["classification_v1"] == "STABLE"
    assert by_feat["b"]["classification_v1"] == "STABLE"


def test_aggregate_v2_iql_stability_groups_by_reward_head_feature() -> None:
    per_fold = [
        {"fold_id_v1": "F1", "reward_id_v1": "R1", "head_v1": "HOLD", "feature_names_v1": ["x"], "coefs_v1": [0.5]},
        {"fold_id_v1": "F2", "reward_id_v1": "R1", "head_v1": "HOLD", "feature_names_v1": ["x"], "coefs_v1": [0.6]},
        {"fold_id_v1": "F3", "reward_id_v1": "R1", "head_v1": "HOLD", "feature_names_v1": ["x"], "coefs_v1": [0.4]},
        {"fold_id_v1": "F1", "reward_id_v1": "R1", "head_v1": "EXIT_NOW", "feature_names_v1": ["x"], "coefs_v1": [-0.3]},
        {"fold_id_v1": "F2", "reward_id_v1": "R1", "head_v1": "EXIT_NOW", "feature_names_v1": ["x"], "coefs_v1": [-0.2]},
        {"fold_id_v1": "F3", "reward_id_v1": "R1", "head_v1": "EXIT_NOW", "feature_names_v1": ["x"], "coefs_v1": [-0.4]},
    ]
    rows = gate._aggregate_v2_iql_stability(per_fold)
    assert len(rows) == 2  # one per (reward, head) for the single feature
    by_head = {r["head_v1"]: r for r in rows}
    assert by_head["HOLD"]["classification_v1"] == "STABLE"
    assert by_head["EXIT_NOW"]["classification_v1"] == "STABLE"


def test_summarize_classifications_counts_correctly() -> None:
    rows = [
        {"classification_v1": "STABLE"},
        {"classification_v1": "STABLE"},
        {"classification_v1": "FLIPS_SIGN"},
        {"classification_v1": "DEAD"},
    ]
    out = gate._summarize_classifications(rows)
    assert out["STABLE"] == 2
    assert out["FLIPS_SIGN"] == 1
    assert out["DEAD"] == 1


def test_validate_final_status_rejects_unknown_status() -> None:
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("MADE_UP", "INVESTIGATE_TRAIL_STOP_DEEP_DIVE_V1")


def test_validate_final_status_rejects_unknown_next_action() -> None:
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status("AUDIT_FEATURE_STABILITY_LOCKED_V1", "TRAIN_NOW")


def test_validate_no_deprecated_revival_passes_on_self() -> None:
    from pathlib import Path

    gate.validate_no_deprecated_revival(Path(gate.__file__))
