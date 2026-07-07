from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_entry_pocket_coverage_alignment_lock_v1 import (
    CONSISTENCY_AUDIT,
    NEXT_ACTION,
    POCKET_COVERAGE_GAP_FORENSICS,
    RUNNER_NEAR_MISS_ALIGNMENT,
    STATUS,
    SUMMARY,
    materialize,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_fixture(tmp_path: Path) -> Path:
    reports_root = tmp_path / "reports"
    reports_root.mkdir()

    selected_pack = reports_root / "MONDAY_SELECTED_PRE_ENTRY_PROXIES_AND_READINESS_PACK_V1_20260424T135846Z"
    selected_pack.mkdir()
    _write_json(selected_pack / "summary_v1.json", {"readiness_decision_v1": "READY_FOR_MORE_NARROW_FEATURE_HARDENING"})

    ledger_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260411"
    ledger_dir.mkdir()
    raw_state_df = pd.DataFrame(
        [
            {
                "candidate_uid": "A",
                "run_id": "RUN_A",
                "trade_uid": "TRADE_A",
                "trade_id": "TID_A",
            },
            {
                "candidate_uid": "B",
                "run_id": "RUN_B",
                "trade_uid": "TRADE_B",
                "trade_id": "TID_B",
            },
        ]
    )
    raw_state_df.to_parquet(ledger_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet", index=False)

    r6_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1"
    r6_dir.mkdir()
    asof_df = pd.DataFrame(
        [
            {
                "candidate_uid": "A",
                "run_id": "RUN_A",
                "trade_uid": "TRADE_A",
                "trade_id": "TID_A",
                "entry_coverage_original_entry_observation_present_v1": True,
                "entry_coverage_original_entry_raw_state_present_v1": True,
                "entry_coverage_repair_applied_v1": False,
                "entry_coverage_repair_source_v1": "ORIGINAL_R2_ENTRY_OBSERVABILITY",
                "as_of_skip_replay_spread_bps_v1": 5.0,
            },
            {
                "candidate_uid": "B",
                "run_id": "RUN_B",
                "trade_uid": "TRADE_B",
                "trade_id": "TID_B",
                "entry_coverage_original_entry_observation_present_v1": True,
                "entry_coverage_original_entry_raw_state_present_v1": True,
                "entry_coverage_repair_applied_v1": False,
                "entry_coverage_repair_source_v1": "ORIGINAL_R2_ENTRY_OBSERVABILITY",
                "as_of_skip_replay_spread_bps_v1": 6.0,
            },
            {
                "candidate_uid": "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03",
                "run_id": "TRUTH_MONFRI_WEEK_20260330_20260406",
                "trade_uid": "TRADE_C",
                "trade_id": "TID_C",
                "entry_coverage_original_entry_observation_present_v1": False,
                "entry_coverage_original_entry_raw_state_present_v1": False,
                "entry_coverage_repair_applied_v1": True,
                "entry_coverage_repair_source_v1": "RUN_SHADOW_META_CANDIDATES_PLUS_REPLAY_CHUNK_PLUS_XGB_EXACT",
                "as_of_skip_replay_spread_bps_v1": 7.0,
            },
            {
                "candidate_uid": "D",
                "run_id": "RUN_D",
                "trade_uid": "TRADE_D",
                "trade_id": "TID_D",
                "entry_coverage_original_entry_observation_present_v1": False,
                "entry_coverage_original_entry_raw_state_present_v1": False,
                "entry_coverage_repair_applied_v1": True,
                "entry_coverage_repair_source_v1": "RUN_SHADOW_META_CANDIDATES_PLUS_REPLAY_CHUNK_PLUS_XGB_EXACT",
                "as_of_skip_replay_spread_bps_v1": 8.0,
            },
            {
                "candidate_uid": "E",
                "run_id": "RUN_A",
                "trade_uid": "TRADE_A",
                "trade_id": "TID_A",
                "entry_coverage_original_entry_observation_present_v1": False,
                "entry_coverage_original_entry_raw_state_present_v1": False,
                "entry_coverage_repair_applied_v1": True,
                "entry_coverage_repair_source_v1": "RUN_SHADOW_META_CANDIDATES_PLUS_REPLAY_CHUNK_PLUS_XGB_EXACT",
                "as_of_skip_replay_spread_bps_v1": 9.0,
            },
        ]
    )
    asof_df.to_parquet(r6_dir / "shadow_meta_all_trade_review_r6_entry_runner_first_as_of_feature_table_v1.parquet", index=False)

    policy_df = pd.DataFrame(
        [
            {
                "candidate_uid": "A",
                "run_id": "RUN_A",
                "trade_uid": "TRADE_A",
                "trade_id": "TID_A",
                "is_repaired_165_v1": False,
                "fifty_plus_mfe_v1": True,
            },
            {
                "candidate_uid": "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03",
                "run_id": "TRUTH_MONFRI_WEEK_20260330_20260406",
                "trade_uid": "TRADE_C",
                "trade_id": "TID_C",
                "is_repaired_165_v1": True,
                "fifty_plus_mfe_v1": True,
            },
            {
                "candidate_uid": "E",
                "run_id": "RUN_A",
                "trade_uid": "TRADE_A",
                "trade_id": "TID_A",
                "is_repaired_165_v1": False,
                "fifty_plus_mfe_v1": True,
            },
        ]
    )
    policy_df.to_parquet(r6_dir / "shadow_meta_all_trade_review_r6_policy_prediction_view_v1.parquet", index=False)

    hindsight_df = pd.DataFrame(
        [
            {
                "candidate_uid": "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03",
                "run_id": "TRUTH_MONFRI_WEEK_20260330_20260406",
                "trade_uid": "TRADE_C",
                "trade_id": "TID_C",
                "r6_label_runner_near_miss_v1": True,
                "r6_label_tail_control_10_50_v1": False,
                "r6_label_missed_should_not_take_v1": False,
                "r6_label_risky_allow_v1": False,
                "r6_label_repaired_165_like_runner_v1": True,
            },
            {
                "candidate_uid": "D",
                "run_id": "RUN_D",
                "trade_uid": "TRADE_D",
                "trade_id": "TID_D",
                "r6_label_runner_near_miss_v1": True,
                "r6_label_tail_control_10_50_v1": False,
                "r6_label_missed_should_not_take_v1": False,
                "r6_label_risky_allow_v1": False,
                "r6_label_repaired_165_like_runner_v1": False,
            },
        ]
    )
    hindsight_df.to_parquet(r6_dir / "shadow_meta_all_trade_review_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet", index=False)

    return reports_root


def test_materialize_monday_entry_pocket_coverage_alignment_lock(tmp_path: Path) -> None:
    reports_root = _build_fixture(tmp_path)
    extension_dir = reports_root / "pack"

    result = materialize(reports_root, extension_dir=extension_dir)
    assert result["status"]["SPEC_STATUS"] == "READ_ONLY_DIAGNOSIS_COMPLETE"
    for artifact in [
        POCKET_COVERAGE_GAP_FORENSICS,
        RUNNER_NEAR_MISS_ALIGNMENT,
        NEXT_ACTION,
        SUMMARY,
        STATUS,
        CONSISTENCY_AUDIT,
    ]:
        assert (extension_dir / artifact).exists()

    pocket_df = pd.read_csv(extension_dir / POCKET_COVERAGE_GAP_FORENSICS)
    repaired = pocket_df.loc[pocket_df["pocket_id_v1"].astype("string").eq("repaired_165")].iloc[0]
    assert int(repaired["total_pocket_size_v1"]) == 1
    assert int(repaired["canonical_entry_raw_state_exact_match_count_v1"]) == 0
    assert int(repaired["r6_fullcoverage_asof_exact_match_count_v1"]) == 1
    assert repaired["alignment_status_v1"] == "MISALIGNED_SURFACE"
    assert repaired["dominant_root_cause_v1"] == "POCKET_LIVES_ON_REPAIRED_FULLCOVERAGE_ENTRY_SURFACE"

    fifty_plus = pocket_df.loc[pocket_df["pocket_id_v1"].astype("string").eq("fifty_plus_mfe_seed")].iloc[0]
    assert int(fifty_plus["candidate_lineage_variant_count_v1"]) == 1

    runner_alignment = json.loads((extension_dir / RUNNER_NEAR_MISS_ALIGNMENT).read_text(encoding="utf-8"))
    assert runner_alignment["canonical_raw_state_exact_matches_v1"] == 0
    assert runner_alignment["r6_fullcoverage_asof_matches_v1"] == 2

    next_action = json.loads((extension_dir / NEXT_ACTION).read_text(encoding="utf-8"))
    assert next_action["primary_action_v1"] == "BUILD_ENTRY_TO_FAILURE_POCKET_BRIDGE_FIRST"
    assert "DO_NOT_RETRAIN_YET" in next_action["supporting_actions_v1"]

    consistency = pd.read_csv(extension_dir / CONSISTENCY_AUDIT)
    assert not consistency["status_v1"].astype("string").eq("FAIL").any()
