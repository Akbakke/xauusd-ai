import json
import sys
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_protector_first_objective_label_review_v1 import main


FORENSIC_TRADE = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_materialize_protector_first_objective_label_review_v1(tmp_path, monkeypatch):
    reports_root = tmp_path / "reports"
    reports_root.mkdir()
    spec_dir = reports_root / "PROTECTOR_FIRST_SHADOW_EXPERIMENT_RUNNER_SPEC_V1_20260424T183817Z"
    spec_dir.mkdir()
    data_dir = reports_root / "data"
    data_dir.mkdir()

    raw = pd.DataFrame({"candidate_uid": [f"cand::{idx:05d}" for idx in range(1689)]})
    raw.to_parquet(data_dir / "exact_raw.parquet", index=False)
    labels = pd.DataFrame({"candidate_uid": [f"cand::{idx:05d}" for idx in range(1851)] + [FORENSIC_TRADE]})
    for col in [
        "r6_label_runner_protect_v1",
        "r6_label_runner_near_miss_v1",
        "r6_label_strong_low_mae_runner_v1",
        "r6_label_runner_100_mfe_v1",
        "r6_label_runner_200_mfe_v1",
        "r6_label_repaired_165_like_runner_v1",
        "r6_label_bad_risk_v1",
        "r6_label_runner_50_mfe_v1",
        "r6_label_tail_control_10_50_v1",
    ]:
        labels[col] = False
    labels.to_parquet(data_dir / "labels.parquet", index=False)

    _write_json(
        spec_dir / "protector_first_runner_spec_v1.json",
        {
            "label_target_contract_v1": {"label_artifact_v1": str(data_dir / "labels.parquet")},
        },
    )
    _write_json(
        spec_dir / "protector_first_feature_and_surface_lock_v1.json",
        {"training_surface_v1": str(data_dir / "exact_raw.parquet")},
    )
    _write_json(
        spec_dir / "protector_first_decision_contract_v1.json",
        {
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
        },
    )
    _write_json(
        spec_dir / "protector_first_eval_and_verdict_matrix_v1.json",
        {
            "hard_safety_requirements_v1": {
                "repaired_165_damage_v1": "== 0",
                "two_hundred_plus_mfe_blocked_v1": "== 0",
            }
        },
    )
    _write_json(
        spec_dir / "protector_first_objective_label_review_spec_v1.json",
        {"review_required_before_training_v1": True},
    )
    _write_json(spec_dir / "protector_first_abort_rules_v1.json", {"abort_before_training_v1": []})
    _write_json(spec_dir / "summary_v1.json", {"runner_config_spec_complete_v1": True})

    extension_dir = reports_root / "OUT"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_protector_first_objective_label_review_v1.py",
            "--reports-root",
            str(reports_root),
            "--runner-spec-dir",
            str(spec_dir),
            "--extension-dir",
            str(extension_dir),
        ],
    )
    main()

    summary = json.loads((extension_dir / "summary_v1.json").read_text(encoding="utf-8"))
    review = json.loads((extension_dir / "protector_first_objective_label_review_v1.json").read_text(encoding="utf-8"))
    cost = json.loads((extension_dir / "winner_damage_cost_lock_v1.json").read_text(encoding="utf-8"))
    treatment = json.loads((extension_dir / "protector_label_treatment_lock_v1.json").read_text(encoding="utf-8"))
    balance = json.loads((extension_dir / "blocker_vs_protector_objective_balance_v1.json").read_text(encoding="utf-8"))
    gate = json.loads((extension_dir / "objective_label_gate_decision_v1.json").read_text(encoding="utf-8"))
    runner_export = json.loads((extension_dir / "runner_gate_artifact_export_v1.json").read_text(encoding="utf-8"))
    compat = json.loads((extension_dir / "protector_first_objective_label_review_spec_v1.json").read_text(encoding="utf-8"))
    action = json.loads((extension_dir / "next_agent_action_lock_v1.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(extension_dir / "consistency_audit_v1.csv")

    assert summary["gate_decision_v1"] == "OBJECTIVE_LABEL_GATE_PASS_WITH_STRICT_GUARDS"
    assert summary["allowed_to_train_v1"] is True
    assert summary["training_authorized_now_v1"] is False
    assert review["blocker_reward_can_still_overwrite_protector_safety_v1"] is False
    assert cost["winner_damage_cost_hierarchy_v1"][0]["cost_treatment_v1"] == "HARD_FAIL"
    assert any(row["pocket_v1"] == "runner_near_miss" for row in treatment["label_treatment_rows_v1"])
    assert "hard-veto pocket blocked" in balance["hard_fail_conflicts_v1"]
    assert gate["decision_v1"] == "OBJECTIVE_LABEL_GATE_PASS_WITH_STRICT_GUARDS"
    assert runner_export["objective_label_review_gate_status_v1"] == "PASS"
    assert runner_export["allowed_to_train_v1"] is True
    assert compat["gate_status_v1"] == "PASS"
    assert action["primary_action_v1"] == "NEXT_AGENT_MAY_IMPLEMENT_PROTECTOR_FIRST_TRAINING_EXECUTION"
    assert action["blocked_action_v1"] == "RUN_TRAINING_NOW"
    assert audit["status_v1"].astype("string").eq("PASS").all()
