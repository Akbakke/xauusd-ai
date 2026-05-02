import json
import sys
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_protector_first_shadow_experiment_spec_v1 import main


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_materialize_protector_first_shadow_experiment_spec_v1(tmp_path, monkeypatch):
    reports_root = tmp_path / "reports"
    reports_root.mkdir()
    design_lock_dir = reports_root / "MONDAY_PROTECTION_FIRST_DESIGN_LOCK_V1_20260424T175003Z"
    design_lock_dir.mkdir()

    _write_json(
        design_lock_dir / "summary_v1.json",
        {
            "main_failure_v1": "BLOCKER_OVERPOWERED_RUNNER_PROTECTOR_AND_COLLATERAL_DAMAGED_WINNERS",
            "recommended_next_step_v1": "DESIGN_PROTECTOR_FIRST_EXPERIMENT_NEXT",
        },
    )
    _write_json(
        design_lock_dir / "protection_first_design_lock_v1.json",
        {
            "explicit_locks_v1": {
                "next_phase_must_be_protection_first_v1": True,
                "runner_protection_cannot_be_only_another_feature_v1": True,
            }
        },
    )
    _write_json(
        design_lock_dir / "runner_protector_architecture_options_v1.json",
        {
            "options_v1": [
                {
                    "option_id_v1": "PROTECTOR_FIRST_VETO_OR_DAMPER",
                    "definition_v1": "protector first",
                    "how_protection_strengthens_v1": "decision order",
                    "difference_from_current_v1": "not just feature",
                    "benefits_v1": "winner protection",
                    "risk_v1": "lower recall",
                    "r6_family_compatible_v1": True,
                    "requires_new_label_or_objective_v1": "decision logic first",
                    "priority_v1": "HIGH",
                }
            ]
        },
    )
    _write_json(
        design_lock_dir / "runner_protection_signal_translation_v1.json",
        {
            "starting_point_v1": {
                "raw_runner_guard_mean_on_blocked_runners_v1": 0.51,
                "model_runner_protector_mean_on_blocked_runners_v1": 0.04,
                "bad_score_mean_on_blocked_runners_v1": 0.54,
            },
            "why_raw_signal_failed_v1": "Raw guard was only a feature.",
        },
    )
    _write_json(
        design_lock_dir / "objective_label_and_head_balance_review_v1.json",
        {
            "threshold_only_solution_v1": "INSUFFICIENT",
            "winner_damage_penalized_too_weakly_v1": True,
        },
    )
    _write_json(
        design_lock_dir / "protection_first_eval_contract_v1.json",
        {"hard_fail_guardrails_v1": {"two_hundred_plus_mfe_blocked_v1": "== 0"}},
    )
    _write_json(
        design_lock_dir / "no_go_for_same_setup_retrain_v1.json",
        {"no_go_locks_v1": ["Do not retrain same narrow setup again."]},
    )
    _write_json(
        design_lock_dir / "next_experiment_shape_options_v1.json",
        {"options_v1": [{"experiment_shape_v1": "PROTECTOR_FIRST_SHADOW_EXPERIMENT"}]},
    )
    _write_json(
        design_lock_dir / "go_or_no_go_next_step_v1.json",
        {"decision_v1": "DESIGN_PROTECTOR_FIRST_EXPERIMENT_NEXT"},
    )

    extension_dir = reports_root / "OUT"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_protector_first_shadow_experiment_spec_v1.py",
            "--reports-root",
            str(reports_root),
            "--design-lock-dir",
            str(design_lock_dir),
            "--extension-dir",
            str(extension_dir),
        ],
    )
    main()

    summary = json.loads((extension_dir / "summary_v1.json").read_text(encoding="utf-8"))
    scope = json.loads((extension_dir / "protector_first_experiment_scope_v1.json").read_text(encoding="utf-8"))
    architecture = json.loads((extension_dir / "protector_architecture_choice_lock_v1.json").read_text(encoding="utf-8"))
    signal = json.loads((extension_dir / "protector_signal_translation_contract_v1.json").read_text(encoding="utf-8"))
    eval_matrix = json.loads((extension_dir / "protector_first_eval_matrix_v1.json").read_text(encoding="utf-8"))
    no_go = json.loads((extension_dir / "no_go_constraints_v1.json").read_text(encoding="utf-8"))
    action = json.loads((extension_dir / "go_or_no_go_next_step_v1.json").read_text(encoding="utf-8"))
    status = json.loads((extension_dir / "status_v1.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(extension_dir / "consistency_audit_v1.csv")

    assert summary["chosen_architecture_v1"] == "PROTECTOR_FIRST_VETO_OR_DAMPER"
    assert summary["do_not_train_yet_v1"] is True
    assert scope["scope_locks_v1"]["shadow_only_v1"] is True
    assert architecture["chosen_primary_design_v1"] == "PROTECTOR_FIRST_VETO_OR_DAMPER"
    assert signal["decision_power_v1"]["blocker_no_longer_uncontrolled_v1"] is True
    assert eval_matrix["hard_fail_metrics_v1"]["two_hundred_plus_mfe_blocked_v1"] == "== 0"
    assert "Do not use bridge as training surface." in no_go["forbidden_v1"]
    assert action["decision_v1"] == "DESIGN_PROTECTOR_FIRST_RUNNER_SPEC_NEXT"
    assert status["not_training_v1"] is True
    assert audit["status_v1"].astype("string").eq("PASS").all()
