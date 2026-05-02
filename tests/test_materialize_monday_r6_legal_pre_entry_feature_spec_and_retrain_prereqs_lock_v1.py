from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_r6_legal_pre_entry_feature_spec_and_retrain_prereqs_lock_v1 import (
    CONSISTENCY_AUDIT,
    CONTRACT_DELTA,
    ENTRY_LEGALITY,
    FAMILY_MATRIX,
    LEGAL_CANDIDATES,
    NEXT_ACTION,
    PROTECTION_LOCK,
    RETRAIN_PREREQS,
    RUNNER_GAP,
    SUMMARY,
    materialize,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_fixture(tmp_path: Path) -> Path:
    reports_root = tmp_path / "reports"
    reports_root.mkdir()

    diag_dir = reports_root / "MONDAY_R6_READONLY_DIAGNOSIS_AND_NEXT_STEP_LOCK_V1_20260424T120208Z"
    diag_dir.mkdir()
    _write_json(
        diag_dir / "monday_r6_result_recheck_v1.json",
        {
            "verdict_v1": "R6_FEATURES_INSUFFICIENT",
            "candidate_family_v1": "R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
            "candidate_policy_v1": "R6_CANDIDATE_04789_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
            "metrics_v1": {
                "bad_blocks_v1": 84,
                "tail_help_v1": 84,
                "precision_v1": 0.9545,
                "worst_loso_precision_v1": 0.8889,
                "repaired_165_damage_v1": 1,
            },
        },
    )
    _write_json(
        diag_dir / "repaired_165_damage_forensic_v1.json",
        {
            "deterministic_trade_key_v1": "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03",
            "take_was_ok_v1": True,
            "label_should_not_take_v1": False,
        },
    )
    _write_json(diag_dir / "summary_v1.json", {"monday_r6_rechecked_v1": True})
    pd.DataFrame(
        [
            {"reference_id_v1": "FROZEN_R6_BENCHMARK"},
            {"reference_id_v1": "MONDAY_R5_1_SAFETY_REFERENCE"},
            {"reference_id_v1": "MONDAY_R6_FAILURE_MINER"},
        ]
    ).to_csv(diag_dir / "comparator_hierarchy_reference_lock_v1.csv", index=False)
    pd.DataFrame(
        [
            {"bucket_id_v1": "MISSED_SHOULD_NOT_TAKE", "count_v1": 462},
            {"bucket_id_v1": "MISSED_10_50_TAIL_CONTROL", "count_v1": 198},
            {"bucket_id_v1": "RISKY_ALLOW", "count_v1": 347},
            {"bucket_id_v1": "RUNNER_NEAR_MISS", "count_v1": 83},
        ]
    ).to_csv(diag_dir / "failure_backlog_gap_map_v1.csv", index=False)
    pd.DataFrame(
        [
            {"field_name_v1": "last_peak_ts", "classification_v1": "NOT_CANONICAL_YET"},
            {"field_name_v1": "last_mfe_ts", "classification_v1": "NOT_CANONICAL_YET"},
            {"field_name_v1": "last_peak_mfe", "classification_v1": "NOT_CANONICAL_YET"},
            {"field_name_v1": "max_mfe_without_mae", "classification_v1": "NOT_CANONICAL_YET"},
            {"field_name_v1": "mfe_mae_sequence_order", "classification_v1": "NOT_CANONICAL_YET"},
        ]
    ).to_csv(diag_dir / "path_dynamics_bottleneck_lock_v1.csv", index=False)

    snapshot_dir = reports_root / "MONDAY_WEDNESDAY_BENCHMARK_SNAPSHOT_V1_20260424T0900Z"
    snapshot_dir.mkdir()
    _write_json(
        snapshot_dir
        / "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1/shadow_meta_path_dynamics_instrumentation_spec_v2.json",
        {
            "fields_v1": [
                {
                    "field_name_v1": "as_of_last_peak_ts_utc_v1",
                    "as_of_semantics_v1": "LEGAL_FOR_MANAGEMENT_EXIT_ANCHOR; NOT_LEGAL_FOR_PRE_ENTRY_IF_COMPUTED_FROM_THIS_TRADE_FUTURE_PATH",
                }
            ]
        },
    )

    r6_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1"
    r6_dir.mkdir()
    pd.DataFrame(
        [
            {
                "contrast_name_v1": "MISSED_SHOULD_NOT_TAKE_VS_CLEAN_TAKE",
                "feature_family_v1": "volatility_range",
                "mean_top5_effect_score_v1": 0.54,
                "path_dynamics_status_v1": "AVAILABLE_EXISTING_AS_OF",
                "top_features_json_v1": "[]",
            }
        ]
    ).to_csv(r6_dir / "shadow_meta_all_trade_review_r6_feature_path_dynamics_audit_v1.csv", index=False)

    freeze_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_2_SHADOW_FREEZE_AND_R6_FAILURE_BACKLOG_V1"
    freeze_dir.mkdir()
    pd.DataFrame(
        [
            {
                "r6_direction_v1": "BETTER_SHOULD_NOT_TAKE_AND_RISKY_ALLOW_LABELS",
                "addressed_failure_types_v1": "MISSED_SHOULD_NOT_TAKE,RISKY_ALLOW",
                "evidence_v1": "462 missed should-not-take remain.",
            }
        ]
    ).to_csv(freeze_dir / "shadow_meta_all_trade_review_r6_label_feature_opportunity_audit_v1.csv", index=False)
    return reports_root


def test_monday_r6_legal_pre_entry_feature_spec_materializes(tmp_path: Path) -> None:
    reports_root = _build_fixture(tmp_path)
    extension_dir = reports_root / "spec"
    result = materialize(reports_root, extension_dir=extension_dir)
    assert result["status"]["SPEC_STATUS"] == "MATERIALIZED_READ_ONLY"
    for artifact in [
        ENTRY_LEGALITY,
        RUNNER_GAP,
        LEGAL_CANDIDATES,
        FAMILY_MATRIX,
        PROTECTION_LOCK,
        RETRAIN_PREREQS,
        CONTRACT_DELTA,
        NEXT_ACTION,
        SUMMARY,
        CONSISTENCY_AUDIT,
    ]:
        assert (extension_dir / artifact).exists()
    summary = json.loads((extension_dir / SUMMARY).read_text(encoding="utf-8"))
    assert summary["benchmark_lock_v1"] == "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"
    prereqs = json.loads((extension_dir / RETRAIN_PREREQS).read_text(encoding="utf-8"))
    assert prereqs["decision_v1"] == "READY_FOR_NARROW_IMPLEMENTATION_PHASE"
    legality = pd.read_csv(extension_dir / ENTRY_LEGALITY)
    assert legality["classification_v1"].astype("string").eq("NOT_LEGAL_FOR_ENTRY").any()
    consistency = pd.read_csv(extension_dir / CONSISTENCY_AUDIT)
    assert not consistency["status_v1"].eq("FAIL").any()
