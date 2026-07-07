from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gx1.scripts import materialize_return_to_140_94_causal_baseline_and_precheck_adapter_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [
            Path("/tmp/RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1_20260428T000000Z_LOCK"),
            Path("/tmp/REJECT_OR_REBUILD_BEST_LANE_FROM_CAUSAL_SIGNALS_V1_20260428T063714Z_LOCK"),
        ]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])


def test_reproduce_140_94_requires_exact_counts() -> None:
    frame = pd.DataFrame(
        {
            "r5_2_package_selected_v1": [True] * 140 + [False],
            "bad_label_v1": [True] * 140 + [False],
            "tail_label_v1": [True] * 94 + [False] * 47,
        }
    )
    assert gate.validate_reproduce_140_94(frame)
    frame.loc[139, "bad_label_v1"] = False
    with pytest.raises(RuntimeError, match="BASELINE_140_94_REPRODUCTION_FAILED"):
        gate.validate_reproduce_140_94(frame)


def test_denylist_blocks_leakage_fields() -> None:
    blocked = [
        "bad_label_v1",
        "post_outcome_mfe_v1",
        "safe_recoverable_v1",
        "coverage_proxy_member_v1",
        "lane_selected_v1",
        "rows_added_vs_140_94_v1",
        "selected_by_v2_v1",
        "candidate_uid_v1",
    ]
    with pytest.raises(RuntimeError, match="FORBIDDEN_140_94_ADAPTER_FEATURE"):
        gate.validate_no_forbidden_feature_names(blocked)


def test_allowlist_is_clean() -> None:
    assert gate.validate_no_forbidden_feature_names(gate.AS_OF_ALLOWED_FEATURES)


def test_forbidden_side_effect_flags_block_r6_adapter_package_and_live_actions() -> None:
    assert gate.validate_no_forbidden_actions()["status_v1"] == "PASS"
    blocked = gate.validate_no_forbidden_actions(r6=True, adapter=True, package=True, freeze=True, promo=True, live=True)
    assert blocked["status_v1"] == "FAIL"
    assert "R6_FORBIDDEN" in blocked["failures_v1"]
    assert "ADAPTER_BUILD_FORBIDDEN" in blocked["failures_v1"]
    assert "LIVE_FORBIDDEN" in blocked["failures_v1"]


def test_comparator_roles_require_185_and_plus45_to_stay_non_deployable() -> None:
    assert gate.validate_comparator_roles(
        {
            "best_lane_185_139_role_v1": "COMPARATOR_DIAGNOSTIC_ONLY",
            "plus45_role_v1": "DIAGNOSTIC_ONLY_NOT_TARGET",
        }
    )
    with pytest.raises(RuntimeError, match="185_139_MUST_REMAIN_COMPARATOR_ONLY"):
        gate.validate_comparator_roles(
            {
                "best_lane_185_139_role_v1": "ADAPTER_TARGET",
                "plus45_role_v1": "DIAGNOSTIC_ONLY_NOT_TARGET",
            }
        )


def test_final_status_and_next_action_are_allowlisted() -> None:
    assert gate.validate_final_status(
        "140_94_CAUSAL_BASELINE_NEEDS_RULE_DISTILLATION_BEFORE_ADAPTER",
        "DISTILL_140_94_CAUSAL_BASELINE_TO_RULES_AND_VETOES_V1",
    )
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("PROMOTE_NOW", "DISTILL_140_94_CAUSAL_BASELINE_TO_RULES_AND_VETOES_V1")
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status("140_94_CAUSAL_BASELINE_NEEDS_RULE_DISTILLATION_BEFORE_ADAPTER", "RUN_R6_NOW")


def test_materializer_writes_required_outputs_and_valid_go_no_go(tmp_path: Path) -> None:
    artifact_root = tmp_path / "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1_20260428T000000Z_LOCK"
    summary = gate.materialize(artifact_root)
    assert summary["reproduced_140_94_exactly_v1"] is True
    assert summary["r6_run_v1"] is False
    assert summary["adapter_built_v1"] is False
    assert summary["package_built_v1"] is False
    required = [
        "return_to_140_94_input_manifest_v1.json",
        "return_to_140_94_reproducibility_audit_v1.json",
        "return_to_140_94_reproducibility_audit_v1.md",
        "baseline_140_94_selection_lineage_v1.csv",
        "baseline_140_94_selection_lineage_v1.json",
        "baseline_140_94_selection_lineage_v1.md",
        "baseline_140_94_as_of_feature_allowlist_v1.json",
        "baseline_140_94_as_of_feature_denylist_v1.json",
        "baseline_140_94_feature_lineage_audit_v1.csv",
        "baseline_140_94_feature_lineage_audit_v1.json",
        "baseline_140_94_feature_lineage_audit_v1.md",
        "baseline_140_94_adapter_precheck_v1.json",
        "baseline_140_94_adapter_precheck_v1.md",
        "baseline_140_94_stress_boundary_audit_v1.json",
        "baseline_140_94_stress_boundary_audit_v1.md",
        "baseline_140_94_near_miss_and_near_fail_rows_v1.csv",
        "baseline_140_94_near_miss_and_near_fail_rows_v1.json",
        "baseline_140_94_group_stability_audit_v1.csv",
        "baseline_140_94_group_stability_audit_v1.json",
        "baseline_140_94_group_stability_audit_v1.md",
        "baseline_140_94_comparison_against_known_candidates_v1.json",
        "baseline_140_94_comparison_against_known_candidates_v1.md",
        "baseline_140_94_anti_overfit_no_shortcut_audit_v1.json",
        "baseline_140_94_anti_overfit_no_shortcut_audit_v1.md",
        "return_to_140_94_recommendation_v1.json",
        "return_to_140_94_recommendation_v1.md",
        "return_to_140_94_causal_baseline_and_precheck_adapter_go_no_go_v1.json",
    ]
    for name in required:
        assert (artifact_root / name).exists(), name
    go = json.loads((artifact_root / "return_to_140_94_causal_baseline_and_precheck_adapter_go_no_go_v1.json").read_text())
    assert go["status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert go["next_recommended_action_v1"] in gate.ALLOWED_NEXT_ACTIONS
    adapter = json.loads((artifact_root / "baseline_140_94_adapter_precheck_v1.json").read_text())
    assert adapter["direct_r6_compatible_v1"] is False
