import json
import sys
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_protector_first_shadow_experiment_runner_spec_v1 import main


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_materialize_protector_first_shadow_experiment_runner_spec_v1(tmp_path, monkeypatch):
    reports_root = tmp_path / "reports"
    reports_root.mkdir()

    protector_dir = reports_root / "PROTECTOR_FIRST_SHADOW_EXPERIMENT_SPEC_V1_20260424T175620Z"
    narrow_dir = reports_root / "MONDAY_NARROW_RETRAIN_RUNNER_SPEC_V1_20260424T155748Z"
    protector_dir.mkdir()
    narrow_dir.mkdir()

    _write_json(
        protector_dir / "summary_v1.json",
        {
            "chosen_architecture_v1": "PROTECTOR_FIRST_VETO_OR_DAMPER",
            "decision_v1": "DESIGN_PROTECTOR_FIRST_RUNNER_SPEC_NEXT",
            "do_not_train_yet_v1": True,
        },
    )
    _write_json(
        protector_dir / "protector_first_experiment_scope_v1.json",
        {"scope_locks_v1": {"shadow_only_v1": True, "not_training_now_v1": True}},
    )
    _write_json(
        protector_dir / "protector_architecture_choice_lock_v1.json",
        {
            "chosen_primary_design_v1": "PROTECTOR_FIRST_VETO_OR_DAMPER",
            "model_vs_decision_contract_v1": {
                "model_parts_v1": ["protector score remains model/scoring surface"],
                "decision_contract_parts_v1": ["protector evaluated before block action"],
            },
        },
    )
    _write_json(
        protector_dir / "protector_signal_translation_contract_v1.json",
        {
            "raw_signals_allowed_v1": ["as_of_pre_entry_runner_protection_guard_score_v1"],
            "decision_power_v1": {"blocker_no_longer_uncontrolled_v1": True},
        },
    )
    _write_json(
        protector_dir / "objective_and_label_review_lock_v1.json",
        {"review_required_before_training_v1": True},
    )
    _write_json(
        protector_dir / "protector_first_eval_matrix_v1.json",
        {"hard_fail_metrics_v1": {"two_hundred_plus_mfe_blocked_v1": "== 0"}},
    )
    _write_json(
        protector_dir / "no_go_constraints_v1.json",
        {"forbidden_v1": ["Do not use bridge as training surface."]},
    )
    _write_json(
        protector_dir / "experiment_implementation_shape_v1.json",
        {"next_job_type_v1": "RUNNER_JOB_SPEC_JOB"},
    )
    _write_json(
        protector_dir / "go_or_no_go_next_step_v1.json",
        {"decision_v1": "DESIGN_PROTECTOR_FIRST_RUNNER_SPEC_NEXT"},
    )
    _write_json(
        protector_dir / "status_v1.json",
        {"failed_check_count_v1": 0},
    )

    feature_names = [f"as_of_baseline_feature_{idx:02d}_v1" for idx in range(62)] + [
        "as_of_pre_entry_vol_exp_comp_score_v1",
        "as_of_pre_entry_directional_asymmetry_score_v1",
        "as_of_pre_entry_swing_retracement_alignment_score_v1",
        "as_of_pre_entry_tail_leakage_pocket_score_v1",
        "as_of_pre_entry_runner_protection_guard_score_v1",
    ]
    pd.DataFrame({"feature_name_v1": feature_names}).to_csv(
        narrow_dir / "monday_narrow_retrain_feature_manifest_v1.csv",
        index=False,
    )
    _write_json(
        narrow_dir / "monday_narrow_retrain_feature_manifest_v1.json",
        {
            "baseline_feature_count_v1": 62,
            "new_proxy_feature_count_v1": 5,
            "total_feature_count_v1": 67,
            "new_proxy_features_v1": feature_names[-5:],
        },
    )
    _write_json(
        narrow_dir / "monday_narrow_retrain_runner_spec_v1.json",
        {
            "input_artifact_v1": str(reports_root / "exact_raw_state.parquet"),
            "input_surface_kind_v1": "CANONICAL_EXACT_ONLY_ENTRY_RAW_STATE",
            "expected_training_rows_v1": 1689,
            "label_artifact_v1": str(reports_root / "labels.parquet"),
            "label_contract_v1": "LOCKED_R6_HINDSIGHT_LABEL_SURFACE_FILTERED_TO_EXACT_ONLY_CANDIDATES",
        },
    )
    _write_json(
        narrow_dir / "monday_narrow_retrain_config_lock_v1.json",
        {"model_family_v1": "R6_STYLE_FIVE_HEAD_SHADOW_FAMILY"},
    )

    extension_dir = reports_root / "OUT"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_protector_first_shadow_experiment_runner_spec_v1.py",
            "--reports-root",
            str(reports_root),
            "--protector-spec-dir",
            str(protector_dir),
            "--narrow-runner-spec-dir",
            str(narrow_dir),
            "--extension-dir",
            str(extension_dir),
        ],
    )
    main()

    summary = json.loads((extension_dir / "summary_v1.json").read_text(encoding="utf-8"))
    runner = json.loads((extension_dir / "protector_first_runner_spec_v1.json").read_text(encoding="utf-8"))
    config = json.loads((extension_dir / "protector_first_config_lock_v1.json").read_text(encoding="utf-8"))
    decision = json.loads((extension_dir / "protector_first_decision_contract_v1.json").read_text(encoding="utf-8"))
    objective = json.loads((extension_dir / "protector_first_objective_label_review_spec_v1.json").read_text(encoding="utf-8"))
    surface = json.loads((extension_dir / "protector_first_feature_and_surface_lock_v1.json").read_text(encoding="utf-8"))
    eval_matrix = json.loads((extension_dir / "protector_first_eval_and_verdict_matrix_v1.json").read_text(encoding="utf-8"))
    prelaunch = json.loads((extension_dir / "protector_first_prelaunch_checklist_v1.json").read_text(encoding="utf-8"))
    abort = json.loads((extension_dir / "protector_first_abort_rules_v1.json").read_text(encoding="utf-8"))
    action = json.loads((extension_dir / "next_agent_action_lock_v1.json").read_text(encoding="utf-8"))
    status = json.loads((extension_dir / "status_v1.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(extension_dir / "consistency_audit_v1.csv")

    assert summary["architecture_v1"] == "PROTECTOR_FIRST_VETO_OR_DAMPER"
    assert summary["training_now_v1"] is False
    assert runner["execution_mode_v1"] == "SPEC_ONLY_DO_NOT_TRAIN"
    assert runner["feature_set_v1"]["feature_count_v1"] == 67
    assert config["architecture_v1"] == "PROTECTOR_FIRST_VETO_OR_DAMPER"
    assert config["bridge_as_training_surface_allowed_v1"] is False
    assert decision["protector_has_decision_power_v1"] is True
    assert len(decision["hard_protector_veto_v1"]) == 4
    assert objective["review_required_before_training_v1"] is True
    assert surface["bridge_as_training_surface_allowed_v1"] is False
    assert surface["feature_count_v1"] == 67
    assert "protector_over_block_override_count" in eval_matrix["protection_specific_metrics_v1"]
    assert any(check["check_id_v1"] == "OBJECTIVE_LABEL_REVIEW_GREEN" for check in prelaunch["checks_v1"])
    assert "objective/label review not green" in abort["abort_before_training_v1"]
    assert action["primary_action_v1"] == "NEXT_AGENT_MAY_IMPLEMENT_PROTECTOR_FIRST_RUNNER"
    assert action["blocked_action_v1"] == "RUN_TRAINING_NOW"
    assert status["failed_check_count_v1"] == 0
    assert audit["status_v1"].astype("string").eq("PASS").all()
