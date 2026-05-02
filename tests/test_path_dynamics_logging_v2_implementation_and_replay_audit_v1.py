from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from gx1.execution.exit_manager import EXIT_EVAL_TRACE_V2_FIELDS, ExitManager
from gx1.scripts.materialize_path_dynamics_logging_v2_implementation_and_replay_audit_v1 import (
    AS_OF_RAW_STATE_TABLE,
    CONTRACT,
    COVERAGE_AUDIT,
    DERIVED_READINESS,
    RUN_DIR_COVERAGE_AUDIT,
    POLICY_LOG_TABLE,
    R6_FREEZE_EXTENSION_NAME,
    R6_SUMMARY,
    SUMMARY,
    TRACE_SCHEMA_AUDIT,
    materialize,
)


def _r6_summary() -> dict:
    return {
        "freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
        "policy_logging_v1": {"row_count_v1": 1971},
        "batch05_v1": {"precision_v1": 0.9285714285714286, "near_boundary_count_v1": 53, "monitor_required_v1": True},
        "status_v1": {"not_live_gate": True, "failed_check_count_v1": 0},
        "selected_candidate_v1": {
            "row_count_v1": 1971,
            "should_not_take_block_count_v1": 180,
            "tail_10_50_help_count_v1": 149,
            "repaired_165_block_count_v1": 0,
            "strong_trade_false_block_count_v1": 0,
            "hundred_plus_mfe_block_count_v1": 0,
            "two_hundred_plus_mfe_block_count_v1": 0,
            "fifty_plus_mfe_block_count_v1": 1,
            "batch04_loso_pass_v1": True,
            "batch05_loso_pass_v1": True,
        },
    }


def test_exit_eval_trace_v2_header_and_mismatch_guard(tmp_path: Path) -> None:
    manager = ExitManager.__new__(ExitManager)
    object.__setattr__(manager, "_runner", None)
    object.__setattr__(manager, "replay_mode", True)
    object.__setattr__(manager, "explicit_output_dir", tmp_path)
    object.__setattr__(manager, "output_dir", None)
    object.__setattr__(manager, "_exit_eval_trace_path", None)
    object.__setattr__(manager, "_exit_eval_trace_header_written", False)

    row = {field: "" for field in EXIT_EVAL_TRACE_V2_FIELDS}
    row.update(
        {
            "trade_id": "T1",
            "timestamp": "2026-01-01 00:00:00+00:00",
            "last_peak_mfe_bps": 12.5,
            "max_mfe_without_mae_bps": 8.0,
            "mfe_mae_sequence_order": "MFE_BEFORE_MAE",
        }
    )
    manager._append_exit_eval_trace(row)
    with (tmp_path / "EXIT_EVAL_TRACE.csv").open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert header == EXIT_EVAL_TRACE_V2_FIELDS

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "EXIT_EVAL_TRACE.csv").write_text("trade_id,timestamp\nT0,2026-01-01\n", encoding="utf-8")
    legacy = ExitManager.__new__(ExitManager)
    object.__setattr__(legacy, "_runner", None)
    object.__setattr__(legacy, "replay_mode", True)
    object.__setattr__(legacy, "explicit_output_dir", legacy_dir)
    object.__setattr__(legacy, "output_dir", None)
    object.__setattr__(legacy, "_exit_eval_trace_path", legacy_dir / "EXIT_EVAL_TRACE.csv")
    object.__setattr__(legacy, "_exit_eval_trace_header_written", False)
    try:
        legacy._append_exit_eval_trace(row)
    except RuntimeError as exc:
        assert "EXIT_EVAL_TRACE_HEADER_MISMATCH" in str(exc)
    else:
        raise AssertionError("legacy trace append should hard-fail on header mismatch")


def test_path_dynamics_logging_v2_materializer_full_chain_fixture(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    reports_root.mkdir()
    freeze_dir = reports_root / R6_FREEZE_EXTENSION_NAME
    freeze_dir.mkdir()
    (freeze_dir / R6_SUMMARY).write_text(json.dumps(_r6_summary()), encoding="utf-8")

    run_dir = reports_root / "runs" / "E2E_SANITY_ORDERFIX_20260101_20260108" / "replay" / "chunk_0"
    run_dir.mkdir(parents=True)
    trace_row = {field: "" for field in EXIT_EVAL_TRACE_V2_FIELDS}
    trace_row.update(
        {
            "trade_id": "T1",
            "timestamp": "2026-01-01 00:05:00+00:00",
            "last_peak_ts_utc": "2026-01-01T00:04:00+00:00",
            "last_mfe_ts_utc": "2026-01-01T00:03:00+00:00",
            "last_peak_mfe_bps": 15.0,
            "max_mfe_without_mae_bps": 10.0,
            "mfe_mae_sequence_order": "MFE_BEFORE_MAE",
        }
    )
    with (run_dir / "EXIT_EVAL_TRACE.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXIT_EVAL_TRACE_V2_FIELDS)
        writer.writeheader()
        writer.writerow(trace_row)

    source_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260421T_SOURCE"
    source_dir.mkdir()
    raw = pd.DataFrame(
        {
            "run_id": ["E2E_SANITY_ORDERFIX_20260101_20260108"],
            "as_of_row_uid_v1": ["A1"],
            "decision_timestamp": ["2026-01-01T00:05:00+00:00"],
            "anchor_timestamp_utc": ["2026-01-01T00:05:00+00:00"],
            "as_of_mgmt_trace_last_peak_ts_utc_v1": ["2026-01-01T00:04:00+00:00"],
            "as_of_mgmt_trace_last_mfe_ts_utc_v1": ["2026-01-01T00:03:00+00:00"],
            "as_of_mgmt_trace_last_peak_mfe_bps_v1": [15.0],
            "as_of_mgmt_trace_max_mfe_without_mae_bps_v1": [10.0],
            "as_of_mgmt_trace_mfe_mae_sequence_order_v1": ["MFE_BEFORE_MAE"],
        }
    )
    raw.to_parquet(source_dir / "shadow_meta_all_trade_review_management_anchor_raw_state_v1.parquet", index=False)
    policy = pd.DataFrame(
        {
            "run_id": ["E2E_SANITY_ORDERFIX_20260101_20260108"],
            "candidate_uid_exact_v1": ["C1"],
            "decision_timestamp": ["2026-01-01T00:05:00+00:00"],
            "decision_anchor_timestamp_utc_v1": ["2026-01-01T00:05:00+00:00"],
            "as_of_management_core_last_peak_ts_utc_v1": ["2026-01-01T00:04:00+00:00"],
            "as_of_management_core_last_mfe_ts_utc_v1": ["2026-01-01T00:03:00+00:00"],
            "as_of_management_core_mfe_bps_at_anchor_v1": [20.0],
            "as_of_management_core_last_peak_mfe_bps_v1": [15.0],
            "as_of_management_core_max_mfe_without_mae_bps_v1": [10.0],
            "as_of_management_core_mfe_mae_sequence_order_v1": ["MFE_BEFORE_MAE"],
        }
    )
    policy.to_parquet(
        source_dir / "shadow_meta_all_trade_review_management_policy_logging_decision_log_harness_v1.parquet",
        index=False,
    )

    extension_dir = reports_root / "v2_audit"
    result = materialize(reports_root, freeze_dir=freeze_dir, extension_dir=extension_dir, test_status="PYTEST")
    assert result["status"]["not_live_gate_v1"] is True
    for artifact in [CONTRACT, TRACE_SCHEMA_AUDIT, AS_OF_RAW_STATE_TABLE, POLICY_LOG_TABLE, COVERAGE_AUDIT, DERIVED_READINESS, SUMMARY]:
        assert (extension_dir / artifact).exists()
    summary = json.loads((extension_dir / SUMMARY).read_text(encoding="utf-8"))
    assert summary["decision_v1"] == "PATH_DYNAMICS_V2_READY_FOR_R7_RETRAIN"
    coverage = pd.read_csv(extension_dir / COVERAGE_AUDIT)
    assert coverage["non_null_count"].gt(0).all()
    derived = pd.read_csv(extension_dir / DERIVED_READINESS)
    assert derived["readiness_verdict"].eq("READY_FROM_LOGGING").sum() >= 4


def test_path_dynamics_logging_v2_materializer_supports_monday_runs_without_policy_log(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    reports_root.mkdir()
    freeze_dir = reports_root / R6_FREEZE_EXTENSION_NAME
    freeze_dir.mkdir()
    (freeze_dir / R6_SUMMARY).write_text(json.dumps(_r6_summary()), encoding="utf-8")

    run_root = reports_root / "TRUTH_MONFRI_WEEK_20260106_20260113"
    trace_dir = run_root / "replay" / "chunk_0"
    trace_dir.mkdir(parents=True)
    (run_root / "RUN_COMPLETED.json").write_text("{}", encoding="utf-8")
    pd.DataFrame({"candidate_uid": ["C1"]}).to_parquet(
        run_root / "trade_outcomes_TRUTH_MONFRI_WEEK_20260106_20260113_MERGED.parquet",
        index=False,
    )
    trace_row = {field: "" for field in EXIT_EVAL_TRACE_V2_FIELDS}
    trace_row.update(
        {
            "trade_id": "T1",
            "timestamp": "2026-01-06 00:05:00+00:00",
            "last_peak_ts_utc": "2026-01-06T00:04:00+00:00",
            "last_mfe_ts_utc": "2026-01-06T00:03:00+00:00",
            "last_peak_mfe_bps": 15.0,
            "max_mfe_without_mae_bps": 10.0,
            "mfe_mae_sequence_order": "MFE_BEFORE_MAE",
        }
    )
    with (trace_dir / "EXIT_EVAL_TRACE.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXIT_EVAL_TRACE_V2_FIELDS)
        writer.writeheader()
        writer.writerow(trace_row)

    source_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260411"
    source_dir.mkdir()
    raw = pd.DataFrame(
        {
            "run_id": ["TRUTH_MONFRI_WEEK_20260106_20260113"],
            "as_of_row_uid_v1": ["A1"],
            "decision_timestamp": ["2026-01-06T00:05:00+00:00"],
            "anchor_timestamp_utc": ["2026-01-06T00:05:00+00:00"],
            "as_of_mgmt_trace_last_peak_ts_utc_v1": ["2026-01-06T00:04:00+00:00"],
            "as_of_mgmt_trace_last_mfe_ts_utc_v1": ["2026-01-06T00:03:00+00:00"],
            "as_of_mgmt_trace_last_peak_mfe_bps_v1": [15.0],
            "as_of_mgmt_trace_max_mfe_without_mae_bps_v1": [10.0],
            "as_of_mgmt_trace_mfe_mae_sequence_order_v1": ["MFE_BEFORE_MAE"],
        }
    )
    raw.to_parquet(source_dir / "shadow_meta_all_trade_review_management_anchor_raw_state_v1.parquet", index=False)
    observation_contract = {
        "derived_observation_fill_counts_v1": {
            "as_of_management_core_minutes_since_last_peak_v1": 1,
            "as_of_management_core_minutes_since_last_mfe_v1": 1,
        }
    }
    readiness_summary = {
        "management_rows_v1": 1,
        "raw_exact_observation_alias_fill_counts_v1": {
            "as_of_management_core_last_peak_ts_utc_v1": 1,
            "as_of_management_core_last_mfe_ts_utc_v1": 1,
            "as_of_management_core_last_peak_mfe_bps_v1": 1,
            "as_of_management_core_max_mfe_without_mae_bps_v1": 1,
            "as_of_management_core_mfe_mae_sequence_order_v1": 1,
        },
    }
    (source_dir / "shadow_meta_all_trade_review_management_rl_observation_contract_v1.json").write_text(
        json.dumps(observation_contract),
        encoding="utf-8",
    )
    (source_dir / "shadow_meta_all_trade_review_management_rl_readiness_summary_v1.json").write_text(
        json.dumps(readiness_summary),
        encoding="utf-8",
    )

    extension_dir = reports_root / "monday_v2_audit"
    result = materialize(reports_root, freeze_dir=freeze_dir, extension_dir=extension_dir, test_status="PYTEST")
    assert result["status"]["not_live_gate_v1"] is True
    summary = json.loads((extension_dir / SUMMARY).read_text(encoding="utf-8"))
    assert summary["decision_v1"] == "FIX_PATH_DYNAMICS_CHAIN_FIRST"
    assert summary["chain_proof_v1"]["raw_state_to_policy_log_v1"] == "POLICY_LOG_ARTIFACT_MISSING_FROM_MONDAY_ROOT"
    coverage = pd.read_csv(extension_dir / COVERAGE_AUDIT)
    policy = coverage[coverage["layer_name"].eq("POLICY_LOG")]
    assert policy["schema_present"].eq(False).all()
    derived = pd.read_csv(extension_dir / DERIVED_READINESS)
    assert "RL_OBSERVATION_CONTRACT" in set(derived["evidence_source_v1"])
    run_coverage = pd.read_csv(extension_dir / RUN_DIR_COVERAGE_AUDIT)
    assert run_coverage.loc[0, "coverage_status"] == "NONZERO_COMPLETED_WITH_V2_TRACE"
