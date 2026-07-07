import json
import sys
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_protection_first_design_lock_v1 import main


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_materialize_monday_protection_first_design_lock_v1(tmp_path, monkeypatch):
    reports_root = tmp_path / "reports"
    reports_root.mkdir()
    forensics_dir = reports_root / "MONDAY_NARROW_RETRAIN_FAILURE_FORENSICS_V1_20260424T174149Z"
    forensics_dir.mkdir()

    _write_json(
        forensics_dir / "summary_v1.json",
        {
            "main_failure_v1": "BLOCKER_OVERPOWERED_RUNNER_PROTECTOR_AND_COLLATERAL_DAMAGED_WINNERS",
            "global_precision_v1": 0.422,
            "strongest_winner_damage_v1": 23,
            "runner_near_miss_blocked_v1": 8,
        },
    )
    _write_json(
        forensics_dir / "go_or_no_go_next_step_v1.json",
        {
            "decision_v1": "STRENGTHEN_RUNNER_PROTECTION_BEFORE_ANY_NEW_RETRAIN",
            "supporting_decisions_v1": ["DO_NOT_RETRAIN_SAME_SETUP_AGAIN"],
        },
    )
    _write_json(
        forensics_dir / "runner_protection_failure_analysis_v1.json",
        {
            "main_failure_mode_v1": "Runner-protection was undervalued/miscalibrated relative to blocker heads, not absent from the input table.",
            "guard_score_diagnosis_v1": {
                "raw_runner_guard_mean_on_blocked_runners_v1": 0.51,
                "model_runner_protector_mean_on_blocked_runners_v1": 0.04,
                "bad_score_mean_on_blocked_runners_v1": 0.54,
            },
        },
    )
    _write_json(
        forensics_dir / "feature_proxy_behavior_review_v1.json",
        {
            "informative_candidates_v1": ["as_of_pre_entry_vol_exp_comp_score_v1"],
            "weak_or_misaligned_candidates_v1": ["as_of_pre_entry_runner_protection_guard_score_v1"],
        },
    )
    _write_json(
        forensics_dir / "strongest_winner_damage_forensics_v1.json",
        {"strongest_winner_damage_count_v1": 23},
    )
    _write_json(
        forensics_dir / "tail_help_vs_bad_block_decomposition_v1.json",
        {"bad_blocks_v1": 111, "tail_help_v1": 32},
    )

    extension_dir = reports_root / "OUT"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_monday_protection_first_design_lock_v1.py",
            "--reports-root",
            str(reports_root),
            "--forensics-dir",
            str(forensics_dir),
            "--extension-dir",
            str(extension_dir),
        ],
    )
    main()

    summary = json.loads((extension_dir / "summary_v1.json").read_text(encoding="utf-8"))
    design = json.loads((extension_dir / "protection_first_design_lock_v1.json").read_text(encoding="utf-8"))
    options = pd.read_csv(extension_dir / "runner_protector_architecture_options_v1.csv")
    eval_contract = json.loads((extension_dir / "protection_first_eval_contract_v1.json").read_text(encoding="utf-8"))
    no_go = json.loads((extension_dir / "no_go_for_same_setup_retrain_v1.json").read_text(encoding="utf-8"))
    action = json.loads((extension_dir / "go_or_no_go_next_step_v1.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(extension_dir / "consistency_audit_v1.csv")

    assert summary["recommended_next_step_v1"] == "DESIGN_PROTECTOR_FIRST_EXPERIMENT_NEXT"
    assert summary["do_not_retrain_same_setup_again_v1"] is True
    assert design["explicit_locks_v1"]["runner_protection_cannot_be_only_another_feature_v1"] is True
    assert "PROTECTOR_FIRST_VETO_OR_DAMPER" in set(options["option_id_v1"])
    assert eval_contract["hard_fail_guardrails_v1"]["two_hundred_plus_mfe_blocked_v1"] == "== 0"
    assert any("Do not retrain same narrow setup" in row for row in no_go["no_go_locks_v1"])
    assert action["decision_v1"] == "DESIGN_PROTECTOR_FIRST_EXPERIMENT_NEXT"
    assert audit["status_v1"].astype("string").eq("PASS").all()
