from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_run_first_bandit_research_eval_v1 import (
    REWARD_VERSION_ID,
    build_first_bandit_research_eval,
    write_first_bandit_research_eval_artifacts,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_fixture(root: Path) -> tuple[Path, Path]:
    dataset_dir = root / "IQL_INTEGRATION" / "BUILD_MANAGEMENT_BANDIT_DATASET_V1_FIXTURE"
    eval_prep_dir = root / "IQL_INTEGRATION" / "BANDIT_RESEARCH_EVAL_PREP_V1_FIXTURE"
    source_dir = root / "SOURCES"
    dataset_dir.mkdir(parents=True)
    eval_prep_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)

    dataset = pd.DataFrame(
        [
            {"row_id": "r1", "episode_id": "t1", "candidate_uid_exact": "c1", "decision_ts": "2026-04-22T10:00:00Z", "action": "HOLD", "reward": 10.0, "reward_version": REWARD_VERSION_ID, "support_status": "MIXED_FEATURE_SUPPORT"},
            {"row_id": "r2", "episode_id": "t2", "candidate_uid_exact": "c2", "decision_ts": "2026-04-22T11:00:00Z", "action": "HOLD", "reward": -5.0, "reward_version": REWARD_VERSION_ID, "support_status": "EDGE_FEATURE_SUPPORT"},
            {"row_id": "r3", "episode_id": "t3", "candidate_uid_exact": "c3", "decision_ts": "2026-04-22T12:00:00Z", "action": "EXIT_NOW", "reward": 20.0, "reward_version": REWARD_VERSION_ID, "support_status": "STRONG_FEATURE_SUPPORT"},
        ]
    )
    dataset_path = dataset_dir / "management_bandit_research_dataset_v1.parquet"
    dataset.to_parquet(dataset_path, index=False)
    dm = pd.DataFrame(
        [
            {"management_row_key_v1": "r1", "action_label_v1": "HOLD", "as_of_session_v1": "NY", "as_of_side_v1": "LONG", "as_of_vol_regime_v1": "MID", "as_of_trend_regime_v1": "UP", "sequence_dataset_membership_v1": "BANDIT_SAFE_ONLY", "hindsight_reward_bad_trade_v1": 0},
            {"management_row_key_v1": "r2", "action_label_v1": "HOLD", "as_of_session_v1": "NY", "as_of_side_v1": "SHORT", "as_of_vol_regime_v1": "HIGH", "as_of_trend_regime_v1": "DOWN", "sequence_dataset_membership_v1": "BANDIT_SAFE_ONLY", "hindsight_reward_bad_trade_v1": 1},
            {"management_row_key_v1": "r3", "action_label_v1": "EXIT_NOW", "as_of_session_v1": "LONDON", "as_of_side_v1": "LONG", "as_of_vol_regime_v1": "LOW", "as_of_trend_regime_v1": "UP", "sequence_dataset_membership_v1": "STRICT_SEQUENCE_SUBSTRATE", "hindsight_reward_bad_trade_v1": 0},
        ]
    )
    dm_path = source_dir / "dm.parquet"
    dm.to_parquet(dm_path, index=False)
    ledger = pd.DataFrame(
        [
            {"trade_uid": "t1", "candidate_uid": "c1", "realized_pnl_bps": 10.0, "mfe_bps": 30.0, "mae_bps": -5.0, "hindsight_peak_mfe_bps_v1": 30.0, "hindsight_peak_to_exit_giveback_bps_v1": 20.0, "bad_trade": 0, "good_trade": 1, "good_exit": 1, "premature_exit": 0, "late_exit": 0, "cata_loser": 0},
            {"trade_uid": "t2", "candidate_uid": "c2", "realized_pnl_bps": -5.0, "mfe_bps": 5.0, "mae_bps": -15.0, "hindsight_peak_mfe_bps_v1": 5.0, "hindsight_peak_to_exit_giveback_bps_v1": 10.0, "bad_trade": 1, "good_trade": 0, "good_exit": 0, "premature_exit": 0, "late_exit": 1, "cata_loser": 0},
            {"trade_uid": "t3", "candidate_uid": "c3", "realized_pnl_bps": 20.0, "mfe_bps": 25.0, "mae_bps": -3.0, "hindsight_peak_mfe_bps_v1": 25.0, "hindsight_peak_to_exit_giveback_bps_v1": 5.0, "bad_trade": 0, "good_trade": 1, "good_exit": 1, "premature_exit": 0, "late_exit": 0, "cata_loser": 0},
        ]
    )
    ledger_path = source_dir / "ledger.parquet"
    ledger.to_parquet(ledger_path, index=False)

    _write_json(
        dataset_dir / "build_management_bandit_dataset_summary_v1.json",
        {
            "dataset_built_v1": True,
            "dataset_verdict_v1": "BANDIT_RESEARCH_DATASET_BUILT_WITH_LIMITATIONS",
            "dataset_parquet_v1": str(dataset_path),
            "included_rows_v1": 3,
            "excluded_rows_v1": 0,
            "reward_version_v1": REWARD_VERSION_ID,
            "support_ood_verdict_v1": "SUPPORT_TOO_THIN",
            "sequence_iql_still_blocked_v1": True,
            "failed_consistency_check_count_v1": 0,
            "failed_non_interference_check_count_v1": 0,
            "action_distribution_v1": [
                {"action_v1": "HOLD", "row_count_v1": 2, "row_share_v1": 2 / 3},
                {"action_v1": "EXIT_NOW", "row_count_v1": 1, "row_share_v1": 1 / 3},
            ],
        },
    )
    _write_json(dataset_dir / "management_bandit_dataset_profile_v1.json", {"support_ood_verdict_from_foundation_v1": "SUPPORT_TOO_THIN"})
    _write_json(
        dataset_dir / "build_management_bandit_dataset_contract_v1.json",
        {
            "source_paths_v1": {
                "locked_ledger_source_v1": str(ledger_path),
                "management_bandit_dm_view_v1": str(dm_path),
            }
        },
    )

    _write_json(
        eval_prep_dir / "bandit_research_eval_prep_summary_v1.json",
        {"eval_prep_ready_v1": True, "eval_prep_verdict_v1": "BANDIT_EVAL_PREP_READY_WITH_LIMITATIONS"},
    )
    comparators = [
        ("no-RL/current locked ledger", "DIRECT_EVAL_COMPARATOR"),
        ("R6 frozen shadow candidate", "PENDING_CALIBRATION"),
        ("R5.2 frozen historical reference", "PENDING_CALIBRATION"),
        ("management harvest comparator", "PENDING_CALIBRATION"),
        ("supervised EXIT_LOCAL/tree baseline", "PENDING_CALIBRATION"),
        ("dummy/random sanity comparator", "DIRECT_EVAL_COMPARATOR"),
    ]
    pd.DataFrame([{"comparator_v1": name, "application_status_v1": status} for name, status in comparators]).to_csv(
        eval_prep_dir / "bandit_comparator_application_plan_v1.csv", index=False
    )
    failchecks = [
        ("realized pnl", "SOFT_REVIEW"),
        ("bad-trade reduction", "SOFT_REVIEW"),
        ("MFE capture", "SOFT_REVIEW"),
        ("MAE burden", "HARD_GATE"),
        ("giveback", "SOFT_REVIEW"),
        ("tail-control help", "HARD_GATE"),
        ("runner damage", "SOFT_REVIEW"),
        ("50+/100+/200+ MFE damage", "HARD_GATE"),
        ("strongest-winner path damage", "HARD_GATE"),
        ("action agreement", "SOFT_REVIEW"),
        ("OOD action rate", "HARD_GATE"),
        ("worst-slice performance", "HARD_GATE"),
        ("rolling-window stability", "HARD_GATE"),
        ("BATCH_04 stress", "SOFT_REVIEW"),
        ("BATCH_05 stress", "SOFT_REVIEW"),
        ("harvest candidate capture", "SOFT_REVIEW"),
        ("failed checks", "HARD_GATE"),
    ]
    pd.DataFrame(
        [
            {
                "metric_or_failcheck_v1": name,
                "enforcement_type_v1": gate,
                "directionality_v1": "LOWER" if "damage" in name or name in {"MAE burden", "OOD action rate"} else "HIGHER",
                "auto_stops_positive_interpretation_v1": gate == "HARD_GATE",
            }
            for name, gate in failchecks
        ]
    ).to_csv(eval_prep_dir / "bandit_failcheck_enforcement_plan_v1.csv", index=False)
    _write_json(root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json", {"freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"})
    return dataset_dir, eval_prep_dir


def test_first_bandit_research_eval_is_limited_and_non_interfering(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    dataset_dir, eval_prep_dir = _write_fixture(reports_root)
    output_dir = reports_root / "IQL_INTEGRATION" / "RUN_FIRST_BANDIT_RESEARCH_EVAL_V1_FIXTURE"
    built_at = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)

    payload = build_first_bandit_research_eval(
        reports_root,
        dataset_dir=dataset_dir,
        eval_prep_dir=eval_prep_dir,
        output_dir=output_dir,
        built_at=built_at,
        exit_manager_sha_before="same",
        exit_manager_sha_after="same",
        r6_sha_before="same",
        r6_sha_after="same",
    )

    assert payload["summary"]["eval_ran_v1"] is True
    assert payload["final_verdict"]["final_verdict_v1"] == "WEAK_OR_INCONCLUSIVE_SIGNAL"
    assert payload["failcheck_review"]["safety_verdict_v1"] == "NO_POSITIVE_CLAIM_ALLOWED"
    assert "OOD action rate" in payload["summary"]["hard_gate_metrics_v1"]
    assert payload["consistency"]["failed_check_count_v1"] == 0
    assert payload["non_interference"]["failed_check_count_v1"] == 0

    result = write_first_bandit_research_eval_artifacts(
        reports_root,
        dataset_dir=dataset_dir,
        eval_prep_dir=eval_prep_dir,
        output_dir=output_dir,
        built_at=built_at,
    )
    assert Path(result["artifact_paths"]["summary"]).exists()
    assert result["summary"]["signal_polarity_v1"] == "INCONCLUSIVE"
    assert result["status"]["training_executed_v1"] is False
    assert result["status"]["iql_training_started_v1"] is False
