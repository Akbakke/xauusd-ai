from __future__ import annotations

import pandas as pd
import pytest

from gx1.scripts import materialize_repair_loso_grouping_or_denominator_contract_v1 as loso


def _toy_scores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_uid": "c1",
                "run_id": "week_a",
                "fold_id_v1": "fold_0",
                "trade_id": "t1",
                "decision_timestamp": "2026-04-01T00:00:00Z",
                "r5_2_v2_final_base_membership": True,
                "label_should_not_take_v1": True,
                "tail_10_50_mfe_v1": True,
                "v2_bucket": "BAD_RECALL_POSITIVE",
                "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE",
            },
            {
                "candidate_uid": "c2",
                "run_id": "week_a",
                "fold_id_v1": "fold_0",
                "trade_id": "t2",
                "decision_timestamp": "2026-04-01T00:01:00Z",
                "r5_2_v2_final_base_membership": True,
                "label_should_not_take_v1": True,
                "tail_10_50_mfe_v1": False,
                "v2_bucket": "BAD_RECALL_POSITIVE",
                "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE",
            },
            {
                "candidate_uid": "c3",
                "run_id": "week_b",
                "fold_id_v1": "fold_1",
                "trade_id": "t3",
                "decision_timestamp": "2026-04-02T00:00:00Z",
                "r5_2_v2_final_base_membership": True,
                "label_should_not_take_v1": True,
                "tail_10_50_mfe_v1": False,
                "v2_bucket": "BAD_RECALL_POSITIVE",
                "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE",
            },
            {
                "candidate_uid": "c4",
                "run_id": "week_b",
                "fold_id_v1": "fold_1",
                "trade_id": "t4",
                "decision_timestamp": "2026-04-02T00:01:00Z",
                "r5_2_v2_final_base_membership": False,
                "label_should_not_take_v1": False,
                "tail_10_50_mfe_v1": False,
                "v2_bucket": "MONITOR_ONLY",
                "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE",
            },
        ]
    )


def test_denominator_2_cannot_be_made_valid_by_silently_lowering_guard() -> None:
    with pytest.raises(RuntimeError, match="SILENT_DENOMINATOR_GUARD_LOWERING_FORBIDDEN"):
        loso.validate_min_denominator_contract(requested_min_denominator=2)


def test_small_loso_groups_cannot_be_silently_dropped() -> None:
    with pytest.raises(RuntimeError, match="SMALL_LOSO_GROUPS_CANNOT_BE_SILENTLY_DROPPED"):
        loso.validate_low_support_policy(exclude_low_support=True, explicit_contract=False)


def test_loso_group_key_must_be_explicit() -> None:
    with pytest.raises(RuntimeError, match="EXPLICIT_LOSO_GROUP_KEY_REQUIRED"):
        loso.validate_explicit_group_key(None)


def test_implicit_latest_or_glob_group_selection_is_forbidden() -> None:
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB_GROUP_SELECTION_FORBIDDEN"):
        loso.validate_explicit_group_key("latest_*_run_id")


def test_metric_repair_must_not_alter_v2_oof_scores() -> None:
    result = loso.validate_metric_repair_integrity(
        scores_hash_before="same",
        scores_hash_after="same",
        provenance_hash_before="prov",
        provenance_hash_after="prov",
        selected_count_before=2,
        selected_count_after=2,
    )

    assert result["status_v1"] == "PASS"
    assert result["scores_unchanged_v1"] is True


def test_metric_repair_must_not_alter_v2_provenance() -> None:
    result = loso.validate_metric_repair_integrity(
        scores_hash_before="same",
        scores_hash_after="same",
        provenance_hash_before="before",
        provenance_hash_after="after",
        selected_count_before=2,
        selected_count_after=2,
    )

    assert result["status_v1"] == "FAIL"
    assert "V2_OOF_PROVENANCE_CHANGED" in result["changed_v1"]


def test_metric_repair_must_not_alter_v2_selected_rows() -> None:
    result = loso.validate_metric_repair_integrity(
        scores_hash_before="same",
        scores_hash_after="same",
        provenance_hash_before="prov",
        provenance_hash_after="prov",
        selected_count_before=2,
        selected_count_after=3,
    )

    assert result["status_v1"] == "FAIL"
    assert "V2_SELECTED_ROWS_CHANGED" in result["changed_v1"]


def test_wrong_group_key_can_be_detected() -> None:
    result = loso.detect_wrong_group_key(current_group_key="run_id", contract_group_key="source_batch_v1")

    assert result["status_v1"] == "WRONG_LOSO_GROUP_KEY_USED"
    assert result["wrong_group_key_detected_v1"] is True


def test_denominator_formula_bug_can_be_detected() -> None:
    result = loso.detect_denominator_formula_bug(observed_denominator=2, recomputed_denominator=5)

    assert result["status_v1"] == "DENOMINATOR_FORMULA_BUG"
    assert result["denominator_formula_bug_detected_v1"] is True


def test_low_support_group_is_reported_even_if_excluded_by_explicit_contract() -> None:
    status = loso.validate_low_support_policy(exclude_low_support=True, explicit_contract=True)
    rows, _ = loso.group_distribution(_toy_scores(), group_key="run_id")

    assert status == "EXCLUDED_LOW_SUPPORT_EXPLICITLY_REPORTED"
    assert any(row["denominator_v1"] == 2 for row in rows)


def test_true_low_support_generalization_weakness_keeps_final_invalid() -> None:
    result = loso.classify_root_cause(
        wrong_group_key=False,
        formula_bug=False,
        threshold_misconfigured=False,
        current_group_explicit=True,
        current_group_legitimate=True,
        worst_denominator=2,
        wednesday_contract_missing=True,
    )

    assert result["root_cause_v1"] == "TRUE_LOW_SUPPORT_GENERALIZATION_WEAKNESS"
    assert result["metric_repair_allowed_v1"] is False


def test_unknown_requires_artifact_allows_no_repair_pass() -> None:
    result = loso.classify_root_cause(
        wrong_group_key=False,
        formula_bug=False,
        threshold_misconfigured=False,
        current_group_explicit=False,
        current_group_legitimate=False,
        worst_denominator=2,
        wednesday_contract_missing=True,
    )

    assert result["root_cause_v1"] == "UNKNOWN_REQUIRES_ARTIFACT"
    assert result["metric_repair_allowed_v1"] is False


def test_wednesday_loso_contract_reconstruction_does_not_invent_missing_artifacts() -> None:
    contract = loso._wednesday_contract()

    assert contract["wednesday_loso_group_key_v1"] == "UNKNOWN_REQUIRES_ARTIFACT"
    assert contract["do_not_invent_group_key_v1"] is True


def test_v2_oof_decision_valid_requires_both_provenance_and_denominator_pass() -> None:
    assert loso.decision_valid_requires_provenance_and_denominator(provenance_pass=True, denominator_pass=False) is False
    assert loso.decision_valid_requires_provenance_and_denominator(provenance_pass=True, denominator_pass=True) is True


def test_optuna_and_v3_metrics_cannot_justify_weakening_v2_guard() -> None:
    assert loso.optuna_v3_metrics_can_override_v2_guard() is False


def test_candidate_grouping_comparison_keeps_run_id_invalid_on_low_support() -> None:
    rows = loso.grouping_candidate_comparison(
        _toy_scores(),
        pd.DataFrame({"candidate_uid": ["c1", "c2", "c3", "c4"], "group_key_v1": ["week_a", "week_a", "week_b", "week_b"]}),
    )
    run_id = next(row for row in rows if row["group_key_name_v1"] == "run_id")

    assert run_id["worst_denominator_v1"] < loso.MIN_LOSO_DENOMINATOR
    assert run_id["denominator_valid_v1"] is False
