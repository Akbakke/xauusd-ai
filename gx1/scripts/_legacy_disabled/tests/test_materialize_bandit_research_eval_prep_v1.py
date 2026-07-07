from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_bandit_research_eval_prep_v1 import (
    REWARD_VERSION_ID,
    build_bandit_research_eval_prep,
    write_bandit_research_eval_prep_artifacts,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_fixture(root: Path) -> tuple[Path, Path]:
    dataset_dir = root / "IQL_INTEGRATION" / "BUILD_MANAGEMENT_BANDIT_DATASET_V1_FIXTURE"
    contract_dir = root / "IQL_INTEGRATION" / "IQL_REWARD_COMPARATOR_AND_BANDIT_CONTRACT_LOCK_V1_FIXTURE"
    dataset_dir.mkdir(parents=True)
    contract_dir.mkdir(parents=True)

    _write_json(
        dataset_dir / "build_management_bandit_dataset_summary_v1.json",
        {
            "dataset_built_v1": True,
            "dataset_verdict_v1": "BANDIT_RESEARCH_DATASET_BUILT_WITH_LIMITATIONS",
            "included_rows_v1": 1796,
            "excluded_rows_v1": 0,
            "reward_version_v1": REWARD_VERSION_ID,
            "support_ood_verdict_v1": "SUPPORT_TOO_THIN",
            "sequence_iql_still_blocked_v1": True,
            "action_distribution_v1": [
                {"action_v1": "HOLD", "row_count_v1": 1751, "row_share_v1": 0.9749},
                {"action_v1": "EXIT_NOW", "row_count_v1": 45, "row_share_v1": 0.0251},
            ],
        },
    )
    _write_json(
        dataset_dir / "management_bandit_dataset_profile_v1.json",
        {
            "verdict_v1": "BANDIT_RESEARCH_DATASET_BUILT_WITH_LIMITATIONS",
            "support_ood_verdict_from_foundation_v1": "SUPPORT_TOO_THIN",
            "not_iql_ready_v1": True,
        },
    )

    comparators = [
        ("no-RL/current locked ledger", "READY"),
        ("R6 frozen shadow candidate", "PENDING_CALIBRATION"),
        ("R5.2 frozen historical reference", "PENDING_CALIBRATION"),
        ("management harvest comparator", "PENDING_CALIBRATION"),
        ("supervised EXIT_LOCAL/tree baseline", "PENDING_CALIBRATION"),
        ("dummy/random sanity comparator", "READY"),
    ]
    pd.DataFrame(
        [
            {
                "comparator_v1": name,
                "foundation_registry_key_v1": name,
                "role_v1": "fixture",
                "status_v1": status,
                "source_registry_status_v1": "REFERENCE_REGISTERED",
                "strengths_v1": "fixture",
                "weaknesses_v1": "fixture",
                "future_rl_comparator_readiness_v1": "fixture",
                "performance_analysis_done_now_v1": False,
            }
            for name, status in comparators
        ]
    ).to_csv(contract_dir / "iql_baseline_comparator_lock_v1.csv", index=False)

    failchecks = [
        ("realized pnl", "SOFT_REVIEW_UNTIL_BASELINE_CALIBRATED", "HIGHER", False, False),
        ("bad-trade reduction", "SOFT_REVIEW", "HIGHER", False, False),
        ("MFE capture", "SOFT_REVIEW", "HIGHER", False, True),
        ("MAE burden", "HARD_GATE_IF_TAIL_WORSENS", "LOWER", True, False),
        ("giveback", "SOFT_REVIEW", "LOWER", False, False),
        ("tail-control help", "HARD_GATE", "HIGHER", True, False),
        ("runner damage", "SOFT_REVIEW", "LOWER", False, True),
        ("50+/100+/200+ MFE damage", "HARD_GATE_FOR_100_200_PLUS", "LOWER", True, False),
        ("strongest-winner path damage", "HARD_GATE", "LOWER", True, True),
        ("action agreement", "SOFT_REVIEW", "EXPLAINED_STABLE", False, False),
        ("OOD action rate", "HARD_GATE", "LOWER", True, True),
        ("worst-slice performance", "HARD_GATE", "HIGHER", True, True),
        ("rolling-window stability", "HARD_GATE_IF_UNSTABLE", "MORE_STABLE", True, False),
        ("BATCH_04 stress", "SOFT_REVIEW_UNTIL_CALIBRATED", "HIGHER_OR_NOT_WORSE", False, False),
        ("BATCH_05 stress", "SOFT_REVIEW_UNTIL_CALIBRATED", "HIGHER_OR_NOT_WORSE", False, False),
        ("harvest candidate capture", "SOFT_REVIEW", "HIGHER_OR_NOT_WORSE", False, False),
        ("failed checks", "HARD_GATE", "LOWER_ZERO_REQUIRED", True, False),
    ]
    pd.DataFrame(
        [
            {
                "metric_or_failcheck_v1": name,
                "why_exists_v1": "fixture",
                "gate_type_v1": gate,
                "better_direction_v1": direction,
                "unacceptable_damage_v1": "fixture damage",
                "protected_pockets_or_slices_v1": "fixture slice",
                "auto_stop_promotion_v1": auto,
                "requires_extra_audit_even_with_good_headline_pnl_v1": audit,
            }
            for name, gate, direction, auto, audit in failchecks
        ]
    ).to_csv(contract_dir / "iql_failcheck_policy_lock_v1.csv", index=False)
    _write_json(contract_dir / "iql_baseline_comparator_and_failcheck_lock_v1.json", {"lock_id_v1": "BASELINE_COMPARATOR_AND_FAILCHECK_LOCK_V1"})
    _write_json(root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json", {"freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"})
    return dataset_dir, contract_dir


def test_bandit_research_eval_prep_locks_boundaries_and_gates(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    dataset_dir, contract_dir = _write_fixture(reports_root)
    output_dir = reports_root / "IQL_INTEGRATION" / "BANDIT_RESEARCH_EVAL_PREP_V1_FIXTURE"
    built_at = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)

    payload = build_bandit_research_eval_prep(
        reports_root,
        dataset_dir=dataset_dir,
        contract_lock_dir=contract_dir,
        output_dir=output_dir,
        built_at=built_at,
        exit_manager_sha_before="same",
        exit_manager_sha_after="same",
        r6_sha_before="same",
        r6_sha_after="same",
    )

    assert payload["summary"]["eval_prep_ready_v1"] is True
    assert "NOT_IQL_EVAL" in payload["scope"]["scope_verdicts_v1"]
    assert "NOT_R7_READINESS" in payload["scope"]["scope_verdicts_v1"]
    assert "SEVERE_ACTION_IMBALANCE" in payload["risk_lock"]["verdicts_v1"]
    assert "OOD action rate" in payload["failcheck_plan"]["policy_v1"]["auto_stop_positive_interpretation_v1"]
    assert len(payload["comparator_plan_df"]) == 6
    assert payload["output_contract"]["verdict_v1"] == "EVAL_OUTPUT_CONTRACT_LOCKED"
    assert payload["consistency"]["failed_check_count_v1"] == 0
    assert payload["non_interference"]["failed_check_count_v1"] == 0

    result = write_bandit_research_eval_prep_artifacts(
        reports_root,
        dataset_dir=dataset_dir,
        contract_lock_dir=contract_dir,
        output_dir=output_dir,
        built_at=built_at,
    )

    assert Path(result["artifact_paths"]["summary"]).exists()
    assert result["summary"]["recommended_next_steps_v1"] == [
        "RUN_FIRST_BANDIT_RESEARCH_EVAL_V1",
        "WAIT_FOR_REPLAY_THEN_REBUILD_CHAIN",
    ]
    assert result["status"]["training_executed_v1"] is False
    assert result["status"]["iql_training_started_v1"] is False
