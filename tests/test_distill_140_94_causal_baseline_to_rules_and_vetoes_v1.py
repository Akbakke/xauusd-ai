from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_distill_140_94_causal_baseline_to_rules_and_vetoes_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [
            Path("/tmp/DISTILL_140_94_CAUSAL_BASELINE_TO_RULES_AND_VETOES_V1_20260428T000000Z_LOCK"),
            Path("/tmp/RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1_20260428T065344Z_LOCK"),
        ]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])


def test_denylist_blocks_leakage_features() -> None:
    blocked = [
        "bad_label_v1",
        "tail_label_v1",
        "post_outcome_mfe_v1",
        "safe_recoverable_v1",
        "coverage_proxy_member_v1",
        "plus45_flag_v1",
        "lane_selected_v1",
        "selected_by_artifact_v1",
        "candidate_uid_v1",
    ]
    with pytest.raises(RuntimeError, match="FORBIDDEN_140_94_RULE_FEATURE"):
        gate.validate_no_forbidden_feature_names(blocked)


def test_allowed_feature_names_are_clean() -> None:
    assert gate.validate_no_forbidden_feature_names(
        [feature for feature in gate.AS_OF_ALLOWED_FEATURES if feature != "tail_repaired_r5_2_oof_candidate_score_v1"]
    )


def test_no_forbidden_actions_default_passes_and_flags_block() -> None:
    assert gate.validate_no_forbidden_actions()["status_v1"] == "PASS"
    blocked = gate.validate_no_forbidden_actions(
        r6=True,
        adapter=True,
        package=True,
        freeze=True,
        promo=True,
        live=True,
        optuna=True,
    )
    assert blocked["status_v1"] == "FAIL"
    assert "R6_FORBIDDEN" in blocked["failures_v1"]
    assert "ADAPTER_BUILD_FORBIDDEN" in blocked["failures_v1"]
    assert "OPTUNA_FORBIDDEN" in blocked["failures_v1"]


def test_reproducibility_requires_exact_140_94_and_clean_safety() -> None:
    payload = {
        "selected_rows_v1": 140,
        "bad_count_v1": 140,
        "tail_count_v1": 94,
        "safety_status_v1": "CLEAN",
    }
    assert gate.validate_reproducibility(payload)
    payload["tail_count_v1"] = 93
    with pytest.raises(RuntimeError, match="DISTILL_140_94_REPRODUCIBILITY_FAILED"):
        gate.validate_reproducibility(payload)


def test_rule_definition_blocks_selected_flag_and_plus45() -> None:
    rule = {
        "required_positive_signals_v1": [
            "tail_repaired_r5_2_oof_candidate_score_v1",
            "asof_signal__r5_1_bad_score_v1",
        ],
        "uses_selected_flag_as_adapter_feature_v1": False,
        "uses_plus45_as_target_feature_filter_or_threshold_v1": False,
    }
    assert gate.validate_rule_definition(rule)
    rule["uses_selected_flag_as_adapter_feature_v1"] = True
    with pytest.raises(RuntimeError, match="SELECTED_FLAG_CANNOT_BE_ADAPTER_FEATURE"):
        gate.validate_rule_definition(rule)


def test_final_status_and_next_action_are_allowlisted() -> None:
    assert gate.validate_final_status(
        "140_94_RULE_VETO_DISTILLATION_PARTIAL_NEEDS_SIMPLIFICATION",
        "SIMPLIFY_140_94_RULES_AND_VETOES_V1",
    )
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("PROMOTE_NOW", "SIMPLIFY_140_94_RULES_AND_VETOES_V1")
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status("140_94_RULE_VETO_DISTILLATION_PARTIAL_NEEDS_SIMPLIFICATION", "RUN_R6_NOW")


def test_materializer_writes_outputs_and_go_no_go(tmp_path: Path) -> None:
    artifact_root = tmp_path / "DISTILL_140_94_CAUSAL_BASELINE_TO_RULES_AND_VETOES_V1_20260428T000000Z_LOCK"
    summary = gate.materialize(artifact_root)
    assert summary["selected_rows_v1"] == 140
    assert summary["bad_tail_v1"] == [140, 94]
    assert summary["r6_run_v1"] is False
    assert summary["adapter_built_v1"] is False
    assert summary["package_built_v1"] is False
    for name in gate.REQUIRED_OUTPUTS:
        assert (artifact_root / name).exists(), name
    go = json.loads((artifact_root / "distill_140_94_causal_baseline_to_rules_and_vetoes_go_no_go_v1.json").read_text())
    assert go["status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert go["next_recommended_action_v1"] in gate.ALLOWED_NEXT_ACTIONS
    assert go["adapter_can_be_built_next_v1"] is False
    assert go["r6_run_v1"] is False
    coverage = json.loads((artifact_root / "distill_140_94_rule_coverage_audit_v1.json").read_text())
    assert coverage["summary_v1"]["full_cover_rule_recovers_all_140_v1"] is True
    assert coverage["summary_v1"]["full_cover_rule_adapter_ready_v1"] is False
