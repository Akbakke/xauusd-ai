from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_harden_140_94_safe_core_and_expand_later_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [
            Path("/tmp/HARDEN_140_94_SAFE_CORE_AND_EXPAND_LATER_V1_20260428T000000Z_LOCK"),
            Path("/tmp/SIMPLIFY_140_94_RULES_AND_VETOES_V1_20260428T083415Z_LOCK"),
        ]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])


def test_feature_denylist_blocks_shortcuts() -> None:
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
    with pytest.raises(RuntimeError, match="FORBIDDEN_HARDEN_140_94_FEATURE"):
        gate.validate_no_forbidden_feature_names(blocked)


def test_allowed_adapter_fields_are_clean() -> None:
    assert gate.validate_no_forbidden_feature_names(
        [
            "tail_repaired_r5_2_oof_candidate_score_v1",
            "asof_signal__r5_1_bad_score_v1",
            "asof_signal__v2_like_bad_tail_v1",
            "asof_low_support_missing_artifact_veto_v1",
            "asof_hard_safety_veto_set_v1",
        ]
    )


def test_reproducibility_requires_exact_simplify_result() -> None:
    payload = {
        "simplified_selected_rows_v1": 91,
        "simplified_recovered_original_140_rows_v1": 86,
        "simplified_extra_rows_v1": 5,
        "simplified_bad_count_audit_only_v1": 86,
        "simplified_tail_count_audit_only_v1": 55,
        "simplified_precision_audit_only_v1": 0.945054945054945,
        "simplified_safety_status_v1": "CLEAN",
        "simplified_unsafe_hits_v1": 0,
    }
    assert gate.validate_reproducibility(payload)
    payload["simplified_extra_rows_v1"] = 4
    with pytest.raises(RuntimeError, match="HARDEN_140_94_REPRODUCIBILITY_FAILED"):
        gate.validate_reproducibility(payload)


def test_hardened_safe_core_validator_blocks_unsafe_or_broad() -> None:
    payload = {
        "recipe_id_v1": gate.HARDENED_RECIPE_ID,
        "safety_status_v1": "CLEAN",
        "extra_rows_v1": 3,
        "recovered_original_140_rows_v1": 86,
    }
    assert gate.validate_hardened_safe_core(payload)
    payload["extra_rows_v1"] = 4
    with pytest.raises(RuntimeError, match="HARDENED_SAFE_CORE_OVERSELECTS"):
        gate.validate_hardened_safe_core(payload)


def test_no_forbidden_actions_guard_flags_side_effects() -> None:
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


def test_final_status_and_next_action_are_allowlisted() -> None:
    assert gate.validate_final_status(
        "140_94_SAFE_CORE_HARDENED_NEEDS_INPUT_MAPPING_EXPAND_LATER",
        "BUILD_140_94_SAFE_CORE_ADAPTER_INPUT_MAPPING_V1",
    )
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("PROMOTE_NOW", "BUILD_140_94_SAFE_CORE_ADAPTER_INPUT_MAPPING_V1")
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status("140_94_SAFE_CORE_HARDENED_NEEDS_INPUT_MAPPING_EXPAND_LATER", "RUN_R6_NOW")


def test_materializer_writes_required_outputs_and_keeps_expansion_separate(tmp_path: Path) -> None:
    artifact_root = tmp_path / "HARDEN_140_94_SAFE_CORE_AND_EXPAND_LATER_V1_20260428T000000Z_LOCK"
    summary = gate.materialize(artifact_root)
    assert summary["safe_core_rule_id_v1"] == gate.HARDENED_RECIPE_ID
    assert summary["selected_rows_v1"] == 89
    assert summary["recovered_original_140_rows_v1"] == 86
    assert summary["extra_rows_v1"] == 3
    assert summary["bad_tail_audit_only_v1"] == [86, 55]
    assert summary["safety_status_v1"] == "CLEAN"
    assert summary["r6_run_v1"] is False
    assert summary["adapter_built_v1"] is False
    assert summary["package_built_v1"] is False
    for name in gate.REQUIRED_OUTPUTS:
        assert (artifact_root / name).exists(), name
    go = json.loads((artifact_root / "harden_140_94_safe_core_and_expand_later_go_no_go_v1.json").read_text())
    assert go["status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert go["next_recommended_action_v1"] in gate.ALLOWED_NEXT_ACTIONS
    assert go["expansion_merged_v1"] is False
    missing = json.loads((artifact_root / "harden_140_94_missing_54_audit_v1.json").read_text())
    assert missing["row_count_v1"] == 54
    extra = json.loads((artifact_root / "harden_140_94_extra_5_audit_v1.json").read_text())
    assert extra["summary_v1"]["extra_rows_v1"] == 5
    assert extra["summary_v1"]["blocked_by_hardened_rule_v1"] == 2
