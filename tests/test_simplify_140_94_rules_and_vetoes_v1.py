from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_simplify_140_94_rules_and_vetoes_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [
            Path("/tmp/SIMPLIFY_140_94_RULES_AND_VETOES_V1_20260428T000000Z_LOCK"),
            Path("/tmp/DISTILL_140_94_CAUSAL_BASELINE_TO_RULES_AND_VETOES_V1_20260428T081017Z_LOCK"),
        ]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])


def test_denylist_blocks_leakage_fields() -> None:
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
    with pytest.raises(RuntimeError, match="FORBIDDEN_SIMPLIFY_140_94_FEATURE"):
        gate.validate_no_forbidden_feature_names(blocked)


def test_adapter_safe_features_are_clean() -> None:
    assert gate.validate_no_forbidden_feature_names(gate.ADAPTER_SAFE_FEATURES)


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
    assert "PACKAGE_BUILD_FORBIDDEN" in blocked["failures_v1"]
    assert "OPTUNA_FORBIDDEN" in blocked["failures_v1"]


def test_reproducibility_requires_original_and_full_cover_exact() -> None:
    payload = {
        "original_selected_rows_v1": 140,
        "original_bad_count_v1": 140,
        "original_tail_count_v1": 94,
        "full_cover_selected_rows_v1": 250,
        "full_cover_recovered_original_140_rows_v1": 140,
        "full_cover_extra_rows_v1": 110,
        "full_cover_safety_status_v1": "CLEAN",
    }
    assert gate.validate_reproducibility(payload)
    payload["full_cover_extra_rows_v1"] = 109
    with pytest.raises(RuntimeError, match="SIMPLIFY_140_94_REPRODUCIBILITY_FAILED"):
        gate.validate_reproducibility(payload)


def test_final_status_and_next_action_are_allowlisted() -> None:
    assert gate.validate_final_status(
        "140_94_SIMPLIFIED_RULES_FOUND_SAFE_CORE_NEEDS_EXPANSION_LATER",
        "HARDEN_140_94_SAFE_CORE_AND_EXPAND_LATER_V1",
    )
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("PROMOTE_NOW", "HARDEN_140_94_SAFE_CORE_AND_EXPAND_LATER_V1")
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status("140_94_SIMPLIFIED_RULES_FOUND_SAFE_CORE_NEEDS_EXPANSION_LATER", "RUN_R6_NOW")


def test_candidate_metrics_require_all_recipes_and_clean_selected() -> None:
    rows = []
    for recipe_id in gate.RECIPE_IDS:
        rows.append(
            {
                "recipe_id_v1": recipe_id,
                "safety_status_v1": "CLEAN",
                "extra_rows_v1": 5 if recipe_id == gate.SELECTED_RECIPE_ID else 0,
            }
        )
    assert gate.validate_candidate_metrics(rows)
    broken = [row.copy() for row in rows if row["recipe_id_v1"] != "FULL_COVER_TIGHTENED_RULE_V1"]
    with pytest.raises(RuntimeError, match="SIMPLIFY_140_94_RECIPE_SET_INCOMPLETE"):
        gate.validate_candidate_metrics(broken)
    unsafe = [row.copy() for row in rows]
    for row in unsafe:
        if row["recipe_id_v1"] == gate.SELECTED_RECIPE_ID:
            row["safety_status_v1"] = "FAIL"
    with pytest.raises(RuntimeError, match="SELECTED_SIMPLIFIED_RECIPE_NOT_SAFETY_CLEAN"):
        gate.validate_candidate_metrics(unsafe)


def test_materializer_writes_required_outputs_and_no_side_effects(tmp_path: Path) -> None:
    artifact_root = tmp_path / "SIMPLIFY_140_94_RULES_AND_VETOES_V1_20260428T000000Z_LOCK"
    summary = gate.materialize(artifact_root)
    assert summary["selected_recipe_v1"] == gate.SELECTED_RECIPE_ID
    assert summary["original_140_94_reproduced_v1"] is True
    assert summary["full_cover_extra_rows_v1"] == 110
    assert summary["extra_selected_rows_v1"] <= 10
    assert summary["safety_status_v1"] == "CLEAN"
    assert summary["r6_run_v1"] is False
    assert summary["adapter_built_v1"] is False
    assert summary["package_built_v1"] is False
    for name in gate.REQUIRED_OUTPUTS:
        assert (artifact_root / name).exists(), name
    go = json.loads((artifact_root / "simplify_140_94_rules_and_vetoes_go_no_go_v1.json").read_text())
    assert go["status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert go["next_recommended_action_v1"] in gate.ALLOWED_NEXT_ACTIONS
    assert go["selected_recipe_v1"] == gate.SELECTED_RECIPE_ID
    assert go["r6_run_v1"] is False
    assert go["adapter_built_v1"] is False
    metrics = json.loads((artifact_root / "simplify_140_94_candidate_recipe_metrics_v1.json").read_text())
    selected = [row for row in metrics["rows_v1"] if row["recipe_id_v1"] == gate.SELECTED_RECIPE_ID][0]
    assert selected["safety_status_v1"] == "CLEAN"
    assert selected["extra_rows_v1"] <= 10
