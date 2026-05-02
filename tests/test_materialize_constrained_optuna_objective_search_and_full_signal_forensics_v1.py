import json
from pathlib import Path

import pandas as pd
import pytest

import gx1.scripts.materialize_constrained_optuna_objective_search_and_full_signal_forensics_v1 as optuna_materializer


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _minimal_ledger(rows: int = 8) -> pd.DataFrame:
    ledger = pd.DataFrame(
        {
            "candidate_uid": [f"c{i}" for i in range(rows)],
            "bad_label_v1": [i % 2 == 0 for i in range(rows)],
            "tail_label_v1": [i % 3 == 0 for i in range(rows)],
            "split_loso_group_v1": ["g1" if i < rows // 2 else "g2" for i in range(rows)],
            "batch_v1": ["b1", "b2"] * (rows // 2),
            "repaired_flag_v1": [False] * rows,
            "fifty_plus_mfe_v1": [False] * rows,
            "hundred_plus_mfe_v1": [False] * rows,
            "two_hundred_plus_mfe_v1": [False] * rows,
            "strongest_winner_path_v1": [False] * rows,
            "high_mfe_flag_v1": [False] * rows,
            "ambiguous_high_mfe_flag_v1": [False] * rows,
            "dangerous_or_protected_v1": [False] * rows,
            "runner_flag_v1": [False] * rows,
        }
    )
    for column in [
        "r5_2_v3_oof_bad_score_v1",
        "r5_2_v2_bad_score_v1",
        "r5_bad_score_v1",
        "r5_1_bad_score_v1",
        "r5_2_v3_oof_tail_score_v1",
        "r5_2_v2_tail_score_v1",
        "r5_tail_score_v1",
        "r5_2_v3_oof_runner_protect_score_v1",
        "r5_2_v2_runner_protect_score_v1",
        "r5_runner_score_v1",
        "r5_1_runner_score_v1",
    ]:
        ledger[column] = 0.9
    return ledger


def _passing_params() -> dict:
    return {
        "w_v3_bad": 1.0,
        "w_v2_bad": 0.0,
        "w_r5_bad": 0.0,
        "w_r51_bad": 0.0,
        "w_v3_tail": 1.0,
        "w_v2_tail": 0.0,
        "w_r5_tail": 0.0,
        "w_v3_runner": 0.0,
        "w_v2_runner": 0.0,
        "w_r5_runner": 0.0,
        "w_r51_runner": 0.0,
        "calibration_temperature": 1.0,
        "bad_threshold": 0.1,
        "tail_threshold": 0.1,
        "risky_threshold": 0.1,
        "confirm_threshold": 0.1,
        "protection_threshold": 10.0,
        "min_final_base_count": 0,
        "fifty_plus_cap": 1,
        "exclude_all_50_plus": False,
    }


def test_implicit_latest_glob_active_selection_is_forbidden(tmp_path: Path) -> None:
    selected_root = tmp_path / "selected"
    selected_root.mkdir()
    selection = tmp_path / "active_score_artifact_selection_v1.json"
    _write_json(
        selection,
        {
            "contract": optuna_materializer.ACTIVE_SELECTION_CONTRACT,
            "decisioning_stage": "PRE_OPTUNA",
            "selection_policy": "LATEST_GLOB",
            "selected_artifacts": {"v3_oof_scores": str(selected_root)},
        },
    )

    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB_SELECTION_FORBIDDEN"):
        optuna_materializer._resolve_active_v3_selection(
            selected_v3_oof_artifact_root=selected_root,
            active_score_artifact_selection=selection,
        )


def test_foundation_audit_must_be_green_before_optuna(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    _write_json(audit_root / "summary_v1.json", {"decision_v1": "FOUNDATION_BLOCKED"})

    with pytest.raises(RuntimeError, match="FOUNDATION_AUDIT_NOT_GREEN"):
        optuna_materializer._assert_foundation_audit_green(audit_root)


def test_active_selection_manifest_required_when_requested(tmp_path: Path) -> None:
    selected_root = tmp_path / "selected"
    selected_root.mkdir()

    with pytest.raises(RuntimeError, match="ACTIVE_SCORE_ARTIFACT_SELECTION_MANIFEST_REQUIRED"):
        optuna_materializer.materialize(
            reports_root=tmp_path,
            selected_v3_oof_artifact_root=selected_root,
            foundation_audit_dir=tmp_path / "audit",
            require_explicit_artifact_selection=True,
        )


def test_fifty_plus_overlap_obeys_explicit_cap() -> None:
    ledger = _minimal_ledger()
    ledger.loc[0, "fifty_plus_mfe_v1"] = True
    ledger.loc[0, "high_mfe_flag_v1"] = True
    params = _passing_params()
    params["fifty_plus_cap"] = 0

    metrics, selected = optuna_materializer._candidate_rule_metrics(ledger, params)

    assert selected.iloc[0]
    assert metrics["fifty_plus_overlap_v1"] == 1
    assert metrics["safety_pass_v1"] is False
    assert "HIGH_MFE_WINNER_DAMAGE" in metrics["fail_reason_v1"]


def test_optuna_trial_rows_carry_artifact_and_denominator_status(tmp_path: Path) -> None:
    trials, _forensics, _best = optuna_materializer._run_optuna_search_if_available(
        _minimal_ledger(),
        {"available_v1": True, "status_v1": "OPTUNA_AVAILABLE", "version_v1": "test"},
        n_trials=1,
        selected_score_artifact_root=tmp_path / "selected",
        provenance_status="PASS",
        denominator_status="PASS",
    )

    assert len(trials) == 1
    assert trials[0]["selected_score_artifact_root_v1"] == str(tmp_path / "selected")
    assert trials[0]["oof_provenance_status_v1"] == "PASS"
    assert trials[0]["metric_denominator_status_v1"] == "PASS"
    assert "precision_denominator_v1" in trials[0]
