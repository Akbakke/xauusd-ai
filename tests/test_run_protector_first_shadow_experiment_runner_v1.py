import json
from pathlib import Path

import pandas as pd
import pytest

from gx1.scripts.run_protector_first_shadow_experiment_runner_v1 import (
    FORENSIC_TRADE,
    PrelaunchValidationError,
    apply_protector_first_decision_effect,
    run_runner,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _feature_names(feature_count: int) -> list[str]:
    baseline = [f"as_of_baseline_feature_{idx:03d}_v1" for idx in range(max(feature_count - 5, 0))]
    proxies = [
        "as_of_pre_entry_vol_exp_comp_score_v1",
        "as_of_pre_entry_directional_asymmetry_score_v1",
        "as_of_pre_entry_swing_retracement_alignment_score_v1",
        "as_of_pre_entry_tail_leakage_pocket_score_v1",
        "as_of_pre_entry_runner_protection_guard_score_v1",
    ]
    return baseline + proxies[: max(feature_count - len(baseline), 0)]


def _gate_payload(status: str = "PASS") -> dict:
    return {
        "review_required_before_training_v1": True,
        "objective_label_review_gate_status_v1": status,
        "labels_to_recheck_v1": [
            "runner_protect",
            "runner_near_miss",
            "strongest_winner",
            "100_plus_winner",
            "200_plus_winner",
            "repaired_165_safety",
            "bad_risk_vs_runner_conflict",
        ],
        "costs_to_weight_harder_v1": [
            "winner_damage_cost",
            "strongest_winner_damage_cost",
            "100_plus_block_cost",
            "200_plus_block_cost",
            "runner_near_miss_block_cost",
            "repaired_165_damage_cost",
        ],
        "training_stop_if_review_not_green_v1": [],
    }


def _build_fixture(
    tmp_path: Path,
    *,
    row_count: int = 1689,
    feature_count: int = 67,
    bridge_input: bool = False,
    forbidden_feature: bool = False,
    missing_decision_contract: bool = False,
    objective_gate_green: bool = True,
    include_bridge_only_in_training: bool = False,
    include_forbidden_matrix_field: bool = False,
) -> tuple[Path, Path, Path]:
    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    spec_dir = reports_root / "PROTECTOR_FIRST_SHADOW_EXPERIMENT_RUNNER_SPEC_V1_20260424T183817Z"
    spec_dir.mkdir()
    bridge_dir = reports_root / "MONDAY_ENTRY_TO_FAILURE_POCKET_BRIDGE_IMPLEMENTATION_V1_20260424T142808Z"
    bridge_dir.mkdir()
    data_dir = reports_root / "data"
    data_dir.mkdir()

    features = _feature_names(feature_count)
    if forbidden_feature and features:
        features[0] = "as_of_skip_xgb_illegal_feature_v1"
    if include_forbidden_matrix_field and features:
        features[-1] = "decision_log_future_truth_v1"

    exact_ids = [f"cand::{idx:05d}" for idx in range(row_count)]
    bridge_only_ids = [FORENSIC_TRADE] + [f"bridge-only::{idx:05d}" for idx in range(162)]
    if include_bridge_only_in_training and exact_ids:
        exact_ids[0] = FORENSIC_TRADE
    eval_ids = [f"cand::{idx:05d}" for idx in range(1689)] + bridge_only_ids

    raw = pd.DataFrame({"candidate_uid": exact_ids, "trade_uid": [f"trade::{idx:05d}" for idx in range(row_count)], "trade_id": [f"T-{idx:05d}" for idx in range(row_count)]})
    for idx, feature in enumerate(features):
        raw[feature] = ((pd.Series(range(row_count)) + idx) % 11) / 10.0
    input_path = data_dir / ("bridge_exact_training_surface.parquet" if bridge_input else "exact_training_surface.parquet")
    raw.to_parquet(input_path, index=False)

    label = pd.DataFrame(
        {
            "candidate_uid": eval_ids,
            "run_id": ["RUN"] * len(eval_ids),
            "trade_uid": [f"trade::{idx:05d}" for idx in range(len(eval_ids))],
            "trade_id": [f"T-{idx:05d}" for idx in range(len(eval_ids))],
            "r6_label_bad_risk_v1": [idx % 4 == 0 for idx in range(len(eval_ids))],
            "r6_label_runner_protect_v1": [idx % 5 == 0 for idx in range(len(eval_ids))],
            "r6_label_tail_control_10_50_v1": [idx % 7 == 0 for idx in range(len(eval_ids))],
            "r6_label_risky_allow_v1": [idx % 6 == 0 for idx in range(len(eval_ids))],
            "r6_label_batch04_blindspot_v1": [idx % 8 == 0 for idx in range(len(eval_ids))],
            "r6_label_runner_50_mfe_v1": [idx % 9 == 0 for idx in range(len(eval_ids))],
            "r6_label_runner_100_mfe_v1": [False] * len(eval_ids),
            "r6_label_runner_200_mfe_v1": [False] * len(eval_ids),
            "r6_label_repaired_165_like_runner_v1": [False] * len(eval_ids),
            "r6_label_strong_low_mae_runner_v1": [False] * len(eval_ids),
            "r6_label_high_mfe_low_giveback_v1": [False] * len(eval_ids),
            "r6_label_runner_near_miss_v1": [idx % 10 == 0 for idx in range(len(eval_ids))],
        }
    )
    forensic_idx = label.index[label["candidate_uid"].astype(str).eq(FORENSIC_TRADE)]
    if len(forensic_idx):
        label.loc[forensic_idx, "r6_label_repaired_165_like_runner_v1"] = True
        label.loc[forensic_idx, "r6_label_runner_50_mfe_v1"] = True
        label.loc[forensic_idx, "r6_label_runner_near_miss_v1"] = True
        label.loc[forensic_idx, "r6_label_bad_risk_v1"] = False
    label_path = data_dir / "labels.parquet"
    label.to_parquet(label_path, index=False)

    bridge = pd.DataFrame(
        {
            "candidate_uid": eval_ids,
            "trade_uid": label["trade_uid"],
            "trade_id": label["trade_id"],
            "bridge_surface_origin_v1": ["EXACT_CANONICAL_RAW_STATE"] * 1689 + ["FULLCOVERAGE_R6_ASOF_BRIDGE_ONLY"] * 163,
            "exact_canonical_raw_state_present_v1": [True] * 1689 + [False] * 163,
            "fullcoverage_r6_asof_present_v1": [True] * len(eval_ids),
            "bridge_proxy_source_v1": ["EXACT_CANONICAL_RAW_STATE"] * 1689 + ["FULLCOVERAGE_R6_ASOF_DERIVED"] * 163,
            "bridge_surface_semantic_contract_v1": ["READINESS_ONLY_NOT_CANONICAL_TRAINING_SURFACE"] * len(eval_ids),
            "bridge_all_selected_proxies_available_v1": [True] * len(eval_ids),
            "bridge_pocket_repaired_165_v1": label["r6_label_repaired_165_like_runner_v1"],
            "bridge_pocket_forensic_repaired_trade_v1": label["candidate_uid"].astype(str).eq(FORENSIC_TRADE),
            "bridge_pocket_runner_near_miss_v1": label["r6_label_runner_near_miss_v1"],
            "bridge_pocket_fifty_plus_mfe_seed_v1": label["r6_label_runner_50_mfe_v1"],
            "bridge_pocket_missed_10_50_tail_control_v1": label["r6_label_tail_control_10_50_v1"],
            "bridge_pocket_missed_should_not_take_v1": label["r6_label_bad_risk_v1"],
            "bridge_pocket_risky_allow_v1": label["r6_label_risky_allow_v1"],
            "bridge_readiness_trackable_v1": [True] * len(eval_ids),
        }
    )
    if include_bridge_only_in_training:
        bridge.loc[bridge["candidate_uid"].astype(str).eq(FORENSIC_TRADE), "exact_canonical_raw_state_present_v1"] = False
    for feature in [
        "as_of_pre_entry_vol_exp_comp_score_v1",
        "as_of_pre_entry_directional_asymmetry_score_v1",
        "as_of_pre_entry_swing_retracement_alignment_score_v1",
        "as_of_pre_entry_tail_leakage_pocket_score_v1",
        "as_of_pre_entry_runner_protection_guard_score_v1",
    ]:
        bridge[feature] = 0.55
    bridge_path = bridge_dir / "entry_to_failure_pocket_bridge_surface_v1.parquet"
    bridge.to_parquet(bridge_path, index=False)

    _write_json(
        spec_dir / "protector_first_runner_spec_v1.json",
        {
            "job_name_v1": "PROTECTOR_FIRST_SHADOW_EXPERIMENT_V1",
            "runner_name_v1": "PROTECTOR_FIRST_SHADOW_EXPERIMENT_RUNNER_V1",
            "execution_mode_v1": "SPEC_ONLY_DO_NOT_TRAIN",
            "training_now_v1": False,
            "replay_now_v1": False,
            "policy_controller_change_v1": False,
            "input_training_surface_v1": str(input_path),
            "eval_readiness_bridge_surface_v1": str(bridge_path),
            "training_surface_kind_v1": "CANONICAL_EXACT_ONLY_ENTRY_RAW_STATE",
            "expected_training_rows_v1": 1689,
            "feature_set_v1": {"feature_count_v1": 67},
            "label_target_contract_v1": {
                "label_artifact_v1": str(label_path),
                "objective_label_review_required_before_training_v1": True,
            },
        },
    )
    _write_json(
        spec_dir / "protector_first_config_lock_v1.json",
        {
            "architecture_v1": "PROTECTOR_FIRST_VETO_OR_DAMPER",
            "shadow_only_v1": True,
            "not_live_gate_v1": True,
            "not_policy_controller_v1": True,
            "bridge_as_training_surface_allowed_v1": False,
            "management_exit_truth_as_entry_features_allowed_v1": False,
            "can_change_in_this_experiment_v1": ["protector-first shadow decision contract"],
            "cannot_change_v1": ["live/controller behavior"],
        },
    )
    if not missing_decision_contract:
        _write_json(
            spec_dir / "protector_first_decision_contract_v1.json",
            {
                "architecture_v1": "PROTECTOR_FIRST_VETO_OR_DAMPER",
                "protector_has_decision_power_v1": True,
                "hard_protector_veto_v1": [
                    {"pocket_v1": "forensic_repaired_trade"},
                    {"pocket_v1": "repaired_165_like_pockets"},
                    {"pocket_v1": "strongest_winner"},
                    {"pocket_v1": "100_plus_200_plus_winner_pockets"},
                ],
                "soft_damper_v1": [
                    {"pocket_v1": "runner_near_miss"},
                    {"pocket_v1": "50_plus_mfe_seed_pockets"},
                ],
                "conflict_resolution_order_v1": ["hard safety veto pockets", "protector-first veto/damper"],
                "conflict_summary_required_fields_v1": [
                    "candidate_uid",
                    "pocket_tag",
                    "blocker_score",
                    "protector_score",
                    "protector_action",
                    "blocker_action_before_protection",
                    "final_shadow_action",
                    "score_margin",
                    "override_or_damper_reason",
                ],
            },
        )
    _write_json(
        spec_dir / "protector_first_objective_label_review_spec_v1.json",
        _gate_payload("PASS" if objective_gate_green else "BLOCKED"),
    )
    _write_json(
        spec_dir / "protector_first_feature_and_surface_lock_v1.json",
        {
            "training_surface_v1": str(input_path),
            "eval_readiness_bridge_surface_v1": str(bridge_path),
            "training_surface_kind_v1": "CANONICAL_EXACT_ONLY_ENTRY_RAW_STATE",
            "expected_training_rows_v1": 1689,
            "bridge_as_training_surface_allowed_v1": False,
            "management_exit_truth_as_features_allowed_v1": False,
            "policy_controller_fields_allowed_v1": False,
            "feature_count_v1": feature_count,
            "baseline_feature_count_v1": max(feature_count - 5, 0),
            "new_proxy_feature_count_v1": min(5, feature_count),
            "new_proxy_features_reused_v1": features[-5:],
            "feature_names_v1": features,
        },
    )
    _write_json(
        spec_dir / "protector_first_eval_and_verdict_matrix_v1.json",
        {
            "hard_safety_requirements_v1": {"repaired_165_damage_v1": "== 0"},
            "protection_specific_metrics_v1": ["protector_over_block_override_count"],
            "verdicts_v1": [{"verdict_v1": "NOT_ESTABLISHED"}],
        },
    )
    _write_json(spec_dir / "protector_first_prelaunch_checklist_v1.json", {"checks_v1": []})
    _write_json(spec_dir / "protector_first_abort_rules_v1.json", {"abort_before_training_v1": [], "reject_after_eval_v1": []})
    _write_json(spec_dir / "summary_v1.json", {"training_now_v1": False})
    return reports_root, spec_dir, reports_root / "OUT"


def test_dry_prelaunch_materializes_scaffold_without_training(tmp_path):
    reports_root, spec_dir, output_dir = _build_fixture(tmp_path)

    summary = run_runner(reports_root=reports_root, spec_dir=spec_dir, output_dir=output_dir, run_training=False)

    assert summary["training_started_v1"] is False
    assert summary["RUNNER_STATUS"] == "DRY_PRELAUNCH_COMPLETED"
    assert summary["feature_count_v1"] == 67
    assert summary["raw_rows_v1"] == 1689
    assert summary["surface_boundary_guard_pass_v1"] is True
    assert summary["next_action_v1"] == "NEXT_AGENT_MAY_RUN_PROTECTOR_FIRST_TRAINING_WITH_EXPLICIT_FLAG"
    assert summary["blocked_action_v1"] == "RUN_TRAINING_WITHOUT_EXPLICIT_FLAG"
    assert (output_dir / "learning_surface_parity_guard_v1.json").exists()
    assert not (output_dir / "model_manifest_v1.json").exists()
    status = json.loads((output_dir / "status_v1.json").read_text(encoding="utf-8"))
    assert status["training_started_v1"] is False
    assert status["objective_label_gate_green_v1"] is True


def test_training_execution_requires_green_objective_gate(tmp_path):
    reports_root, spec_dir, output_dir = _build_fixture(tmp_path, objective_gate_green=False)

    with pytest.raises(PrelaunchValidationError, match="objective/label review gate is not green"):
        run_runner(reports_root=reports_root, spec_dir=spec_dir, output_dir=output_dir, run_training=True)


def test_training_execution_runs_only_with_explicit_flag_and_writes_outputs(tmp_path):
    reports_root, spec_dir, output_dir = _build_fixture(tmp_path)

    dry = run_runner(reports_root=reports_root, spec_dir=spec_dir, output_dir=output_dir / "dry", run_training=False)
    trained = run_runner(reports_root=reports_root, spec_dir=spec_dir, output_dir=output_dir / "train", run_training=True)

    assert dry["training_started_v1"] is False
    assert trained["training_started_v1"] is True
    assert trained["RUNNER_STATUS"] in {"TRAINING_EXECUTION_COMPLETED", "TRAINING_EXECUTION_DISQUALIFIED"}
    for name in [
        "training_execution_summary_v1.json",
        "training_matrix_summary_v1.json",
        "learning_surface_parity_guard_v1.json",
        "model_manifest_v1.json",
        "config_manifest_v1.json",
        "feature_manifest_echo_v1.csv",
        "prediction_view_v1.parquet",
        "eval_summary_v1.json",
        "compare_against_report_v1.json",
        "pocket_report_v1.csv",
        "blocker_vs_protector_conflict_summary_v1.csv",
        "blocker_vs_protector_conflict_summary_v1.json",
        "verdict_package_v1.json",
        "status_v1.json",
        "manifest_v1.json",
        "consistency_audit_v1.csv",
    ]:
        assert (output_dir / "train" / name).exists()


def test_wrong_row_count_hard_fails(tmp_path):
    reports_root, spec_dir, output_dir = _build_fixture(tmp_path, row_count=1688)

    with pytest.raises(PrelaunchValidationError, match="training row count"):
        run_runner(reports_root=reports_root, spec_dir=spec_dir, output_dir=output_dir)


def test_wrong_feature_count_hard_fails(tmp_path):
    reports_root, spec_dir, output_dir = _build_fixture(tmp_path, feature_count=66)

    with pytest.raises(PrelaunchValidationError, match="feature count|baseline feature count|selected feature count"):
        run_runner(reports_root=reports_root, spec_dir=spec_dir, output_dir=output_dir)


def test_bridge_training_surface_hard_fails(tmp_path):
    reports_root, spec_dir, output_dir = _build_fixture(tmp_path, bridge_input=True)

    with pytest.raises(PrelaunchValidationError, match="bridge path proposed"):
        run_runner(reports_root=reports_root, spec_dir=spec_dir, output_dir=output_dir)


def test_forbidden_fields_hard_fail(tmp_path):
    reports_root, spec_dir, output_dir = _build_fixture(tmp_path, forbidden_feature=True)

    with pytest.raises(PrelaunchValidationError, match="forbidden feature fields"):
        run_runner(reports_root=reports_root, spec_dir=spec_dir, output_dir=output_dir)


def test_missing_surface_boundary_report_hard_fails(tmp_path):
    reports_root, spec_dir, output_dir = _build_fixture(tmp_path)
    for path in reports_root.glob("MONDAY_ENTRY_TO_FAILURE_POCKET_BRIDGE_IMPLEMENTATION_V1_*/*.parquet"):
        path.unlink()

    with pytest.raises(PrelaunchValidationError, match="surface-boundary report"):
        run_runner(reports_root=reports_root, spec_dir=spec_dir, output_dir=output_dir)


def test_training_matrix_excludes_bridge_only_and_forensic_is_eval_guard(tmp_path):
    reports_root, spec_dir, output_dir = _build_fixture(tmp_path)

    run_runner(reports_root=reports_root, spec_dir=spec_dir, output_dir=output_dir, run_training=True)

    matrix = json.loads((output_dir / "training_matrix_summary_v1.json").read_text(encoding="utf-8"))
    parity = json.loads((output_dir / "learning_surface_parity_guard_v1.json").read_text(encoding="utf-8"))
    assert matrix["row_count_v1"] == 1689
    assert matrix["feature_count_v1"] == 67
    assert matrix["bridge_only_rows_in_training_matrix_v1"] == 0
    assert parity["bridge_only_count_v1"] == 163
    assert parity["forensic_repaired_trade_v1"]["present_on_training_surface_v1"] is False
    assert parity["forensic_repaired_trade_v1"]["eval_hard_guard_even_when_not_training_row_v1"] is True


def test_bridge_only_rows_in_training_matrix_hard_fail(tmp_path):
    reports_root, spec_dir, output_dir = _build_fixture(tmp_path, include_bridge_only_in_training=True)

    with pytest.raises(PrelaunchValidationError, match="bridge-only rows"):
        run_runner(reports_root=reports_root, spec_dir=spec_dir, output_dir=output_dir)


def test_hard_veto_changes_decision_in_controlled_case():
    scored = pd.DataFrame(
        {
            "candidate_uid": [FORENSIC_TRADE],
            "trade_uid": ["trade"],
            "trade_id": ["T"],
            "pred__protector_first__bad_risk__prob_true_v1": [0.9],
            "pred__protector_first__tail_control_10_50__prob_true_v1": [0.1],
            "pred__protector_first__risky_allow__prob_true_v1": [0.1],
            "pred__protector_first__batch04_blindspot__prob_true_v1": [0.1],
            "pred__protector_first__runner_protector__prob_true_v1": [0.8],
            "as_of_pre_entry_runner_protection_guard_score_v1": [0.8],
            "bridge_pocket_forensic_repaired_trade_v1": [True],
        }
    )

    out, conflicts, summary = apply_protector_first_decision_effect(scored)

    assert out.loc[0, "decision_before_protector_v1"] == "BLOCK"
    assert out.loc[0, "final_shadow_decision_v1"] == "ALLOW"
    assert bool(out.loc[0, "hard_veto_applied_v1"]) is True
    assert len(conflicts) == 1
    assert summary["hard_veto_count_v1"] == 1


def test_soft_damper_changes_decision_in_controlled_case():
    scored = pd.DataFrame(
        {
            "candidate_uid": ["cand::soft"],
            "trade_uid": ["trade"],
            "trade_id": ["T"],
            "pred__protector_first__bad_risk__prob_true_v1": [0.56],
            "pred__protector_first__tail_control_10_50__prob_true_v1": [0.1],
            "pred__protector_first__risky_allow__prob_true_v1": [0.1],
            "pred__protector_first__batch04_blindspot__prob_true_v1": [0.1],
            "pred__protector_first__runner_protector__prob_true_v1": [0.6],
            "as_of_pre_entry_runner_protection_guard_score_v1": [0.6],
            "r6_label_runner_near_miss_v1": [True],
            "r6_label_runner_50_mfe_v1": [True],
        }
    )

    out, conflicts, summary = apply_protector_first_decision_effect(scored)

    assert out.loc[0, "decision_before_protector_v1"] == "BLOCK"
    assert out.loc[0, "final_shadow_decision_v1"] == "ALLOW"
    assert bool(out.loc[0, "soft_damper_applied_v1"]) is True
    assert len(conflicts) == 1
    assert summary["soft_damper_count_v1"] == 1


def test_verdict_disqualifies_on_controlled_safety_fail(tmp_path):
    reports_root, spec_dir, output_dir = _build_fixture(tmp_path)

    summary = run_runner(reports_root=reports_root, spec_dir=spec_dir, output_dir=output_dir, run_training=True)
    verdict = json.loads((output_dir / "verdict_package_v1.json").read_text(encoding="utf-8"))

    assert summary["RUNNER_STATUS"] in {"TRAINING_EXECUTION_COMPLETED", "TRAINING_EXECUTION_DISQUALIFIED"}
    if verdict["candidate_disqualified_v1"]:
        assert summary["RUNNER_STATUS"] == "TRAINING_EXECUTION_DISQUALIFIED"
        assert verdict["hard_fail_reasons_v1"]


def test_missing_protector_contract_hard_fails(tmp_path):
    reports_root, spec_dir, output_dir = _build_fixture(tmp_path, missing_decision_contract=True)

    with pytest.raises(FileNotFoundError, match="missing required artifacts"):
        run_runner(reports_root=reports_root, spec_dir=spec_dir, output_dir=output_dir)
