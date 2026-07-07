from __future__ import annotations

import json
from pathlib import Path

from gx1.scripts.audit_truth_management_next_step_priority import (
    build_management_next_step_priority_summary,
    write_management_next_step_priority_artifacts,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def test_build_management_next_step_priority_summary_orders_real_tracks(tmp_path: Path) -> None:
    reports_root = tmp_path / "truth_root"
    reports_root.mkdir()
    extension_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260420T133345Z_MANAGEMENT_AUDIT_EXTENSION_V1"
    extension_dir.mkdir()

    _write_json(
        extension_dir / "shadow_meta_all_trade_review_management_audit_extension_build_summary_v1.json",
        {
            "source_invariants_v1": {"exact_rows": 4036, "fallback_rows": 0, "unjoinable_rows": 1},
            "policy_logging_summary_v1": {
                "instrumentation_status_v1": "BEVIST",
                "behavior_policy_readiness_v1": "IKKE_ETABLERT",
                "propensity_readiness_v1": "IKKE_ETABLERT",
                "observed_sample_rows_v1": 1888,
                "decision_log_rows_v1": 1796,
                "manual_review_rows_attached_v1": 314,
            },
            "regime_overlay_summary_v1": {
                "regime_consistency_status_v1": "IKKE_ETABLERT",
                "outcome_advantage_status_v1": "INDIKERT",
            },
            "path_dynamics_source_binding_summary_v1": {
                "derivation_rows_v1": [
                    {
                        "field_name_v1": "as_of_management_core_last_peak_ts_utc_v1",
                        "derivation_status_v1": "NOT_ESTABLISHED",
                    },
                    {
                        "field_name_v1": "as_of_management_core_last_mfe_ts_utc_v1",
                        "derivation_status_v1": "NOT_ESTABLISHED",
                    },
                ]
            },
            "reference_compare_summary_v1": {"local_tight_reference_pairs_v1": 0},
            "local_reference_diff_summary_v1": {"status_v1": "NO_LOCAL_TIGHT_REFERENCE_PRESENT"},
            "manual_deep_dive_summary_v1": {
                "overall_conclusion_status_counts_v1": {"IKKE_ETABLERT": 5}
            },
        },
    )
    _write_json(
        extension_dir / "shadow_meta_all_trade_review_management_root_cause_parallel_audit_summary_v1.json",
        {
            "root_cause_primary_v1": "BUCKET_TAXONOMY",
            "root_cause_secondary_v1": "REGIME_TAXONOMY",
            "recommended_next_step_v1": "COARSEN_BUCKET_OR_USE_RAW_SCORE",
        },
    )
    _write_json(
        extension_dir / "shadow_meta_all_trade_review_management_parallel_next_step_triage_summary_v1.json",
        {
            "triage_summary_v1": {
                "winner_v1": {"track_v1": "COARSE_CONDITIONING_GRID_V1"}
            }
        },
    )
    _write_json(
        extension_dir / "shadow_meta_all_trade_review_management_walkforward_regime_scoreboard_summary_v1.json",
        {
            "thin_groups_lt_10_v1": 82,
            "stability_status_counts_v1": {"IKKE_ETABLERT": 126, "INDIKERT": 3},
        },
    )
    _write_json(
        extension_dir / "shadow_meta_all_trade_review_management_regime_incremental_signal_audit_summary_v1.json",
        {
            "verdict_counts_v1": {"THIN": 126, "NO_CLEAR_LIFT": 34, "MIXED": 18}
        },
    )
    _write_json(
        reports_root / "truth_trade_foundation_quality_v1.json",
        {
            "trade_count": 1971,
            "outlook_v1": "POSITIVE_EDGE_HIGH_REGRET",
            "profitability": {
                "avg_pnl_bps": 2.48,
                "profit_factor": 1.14,
                "max_drawdown_bps": -9805.3,
            },
            "hold_longer_pressure": {
                "extra_value_bps": {"mean": 18.1},
                "meaningful_extra_value_10bps_rate": 0.5388,
            },
        },
    )
    _write_json(
        reports_root / "truth_entry_skipability_pressure_v1.json",
        {
            "completed_zero_trade_runs": 10,
            "candidate_rich_zero_trade_runs": 10,
        },
    )
    _write_json(
        reports_root / "truth_continuous_market_opportunity_v1.json",
        {
            "opportunity_rich_zero_trade_runs_anchor": [
                "E2E_SANITY_ORDERFIX_20260318_20260325"
            ]
        },
    )

    summary = build_management_next_step_priority_summary(
        reports_root=reports_root,
        extension_dir=extension_dir,
    )

    assert summary["overall_verdict_v1"] == "REAL_AUDIT_SURFACE_CLEAR_PRIORITIES_NOT_RL_READY"
    assert summary["priority_tracks_v1"][0]["track_v1"] == "COARSE_CONDITIONING_GRID_AND_RAW_SCORE"
    assert summary["priority_tracks_v1"][1]["track_v1"] == "BEHAVIOR_POLICY_AND_PROPENSITY_LOGGING"
    assert summary["priority_tracks_v1"][2]["track_v1"] == "COARSE_REGIME_AND_SESSION_CONDITIONING"
    assert summary["gates_v1"]["rl_training_gate_v1"] == "DO_NOT_START_RL_TRAINING_YET"
    assert summary["do_not_optimize_toward_v1"][0]["topic_v1"] == "LOCAL_TIGHT_REFERENCE_EXIT_EXPLANATIONS"

    written = write_management_next_step_priority_artifacts(
        reports_root=reports_root,
        extension_dir=extension_dir,
    )
    assert Path(written["json_path"]).exists()
    assert Path(written["md_path"]).exists()
