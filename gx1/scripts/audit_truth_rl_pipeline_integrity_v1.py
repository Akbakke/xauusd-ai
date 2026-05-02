#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
LEDGER_NAMESPACE_PREFIX = "ALL_TRADE_REVIEW_LEDGER_"
AUDIT_EXTENSION_SUFFIX = "RL_PIPELINE_INTEGRITY_AUDIT_V1"
TOP_LEVEL_JSON = "truth_rl_pipeline_integrity_audit_v1.json"
TOP_LEVEL_CSV = "truth_rl_pipeline_integrity_audit_v1.csv"

BOOL_REWARD_CHANNELS = [
    "good_trade",
    "good_trade_mfe20_mae5",
    "bad_trade",
    "good_exit",
    "premature_exit",
    "late_exit",
]
TERMINAL_BOOL_REWARD_CHANNELS = [
    "terminal_good_trade_v1",
    "terminal_good_trade_mfe20_mae5_v1",
    "terminal_bad_trade_v1",
    "terminal_good_exit_v1",
    "terminal_premature_exit_v1",
    "terminal_late_exit_v1",
]
BANDIT_BOOL_REWARD_CHANNELS = [
    "hindsight_reward_good_trade_v1",
    "hindsight_reward_good_trade_mfe20_mae5_v1",
    "hindsight_reward_bad_trade_v1",
    "hindsight_reward_good_exit_v1",
    "hindsight_reward_premature_exit_v1",
    "hindsight_reward_late_exit_v1",
]
ENTRY_XGB_REQUIRED = {
    "as_of_skip_xgb_p_flat_v1",
    "as_of_skip_xgb_p_hat_v1",
    "as_of_skip_xgb_p_long_v1",
    "as_of_skip_xgb_p_short_v1",
    "as_of_skip_xgb_pred_side_v1",
    "as_of_skip_xgb_has_ctx_v1",
}
MANAGEMENT_XGB_REQUIRED = {
    "as_of_management_xgb_p_long_v1",
    "as_of_management_xgb_p_short_v1",
    "as_of_management_xgb_p_flat_v1",
    "as_of_management_xgb_p_hat_v1",
    "as_of_management_xgb_pred_side_v1",
    "as_of_management_xgb_has_ctx_v1",
}
MANAGEMENT_EXIT_REQUIRED = {
    "as_of_management_exit_model_evaluated_v1",
    "as_of_management_exit_prob_v1",
    "as_of_management_exit_prob_available_v1",
    "as_of_management_exit_threshold_v1",
}
FORBIDDEN_OBSERVATION_TOKENS = [
    "hindsight",
    "terminal",
    "reward",
    "good_trade",
    "bad_trade",
    "realized_pnl",
    "exit_reason",
]


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    return Path(ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()).expanduser().resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected object JSON in {path}")
    return payload


def _resolve_dir_from_summary(
    reports_root: Path,
    *,
    summary_name: str,
    key: str,
    fallback_suffix: str | None = None,
) -> Path:
    summary_path = reports_root / summary_name
    if summary_path.exists():
        value = _load_json(summary_path).get(key)
        if isinstance(value, str) and value.strip():
            candidate = Path(value).expanduser().resolve()
            if candidate.exists():
                return candidate
    if fallback_suffix is None:
        raise FileNotFoundError(f"Could not resolve {key} from {summary_path}")
    candidates = sorted(
        [
            path
            for path in reports_root.iterdir()
            if path.is_dir() and path.name.startswith(LEDGER_NAMESPACE_PREFIX) and path.name.endswith(fallback_suffix)
        ],
        key=lambda path: path.name,
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()
    raise FileNotFoundError(f"Could not resolve {key}; no {fallback_suffix} extension found under {reports_root}")


def _bool_text_count(series: pd.Series) -> int:
    return int(series.astype("string").isin(["TRUE", "FALSE"]).sum())


def _missing_count(series: pd.Series) -> int:
    if pd.api.types.is_numeric_dtype(series):
        return int(series.isna().sum())
    text = series.astype("string").str.strip()
    return int(
        (
            text.isna()
            | text.eq("")
            | text.str.upper().isin(["NOT_AVAILABLE", "IKKE_ETABLERT", "NAN", "NAT", "<NA>", "NONE"])
        ).sum()
    )


def _all_present(frame: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    return [column for column in columns if column not in frame.columns]


def _observation_leak_fields(fields: Iterable[str]) -> List[str]:
    leaks: List[str] = []
    for field in fields:
        lowered = str(field).lower()
        if any(token in lowered for token in FORBIDDEN_OBSERVATION_TOKENS):
            leaks.append(str(field))
    return leaks


def _append_check(
    rows: List[Dict[str, Any]],
    *,
    check_name: str,
    status: str,
    observed: Any,
    expected: Any,
    severity: str = "FAIL",
    note: str,
) -> None:
    rows.append(
        {
            "check_name_v1": check_name,
            "status_v1": status,
            "severity_if_not_pass_v1": severity,
            "observed_value_v1": json.dumps(observed, ensure_ascii=True, sort_keys=True)
            if isinstance(observed, (dict, list))
            else observed,
            "expected_value_v1": json.dumps(expected, ensure_ascii=True, sort_keys=True)
            if isinstance(expected, (dict, list))
            else expected,
            "note_v1": note,
        }
    )


def build_rl_pipeline_integrity_audit(
    reports_root: Path,
    *,
    review_dir: Path | None = None,
    unified_dir: Path | None = None,
    recommendation_dir: Path | None = None,
) -> Dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    review_dir = review_dir or _resolve_dir_from_summary(
        reports_root,
        summary_name="truth_downstream_canonical_rebuild_v1.json",
        key="ledger_dir",
    )
    unified_dir = unified_dir or _resolve_dir_from_summary(
        reports_root,
        summary_name="truth_rl_unified_observability_v1.json",
        key="extension_dir_v1",
        fallback_suffix="RL_UNIFIED_OBSERVABILITY_V1",
    )
    recommendation_dir = recommendation_dir or _resolve_dir_from_summary(
        reports_root,
        summary_name="truth_rl_recommendation_candidate_v1.json",
        key="extension_dir_v1",
        fallback_suffix="RL_RECOMMENDATION_CANDIDATE_V1",
    )

    rebuild_summary = _load_json(reports_root / "truth_downstream_canonical_rebuild_v1.json")
    unified_summary = _load_json(reports_root / "truth_rl_unified_observability_v1.json")
    recommendation_summary = _load_json(reports_root / "truth_rl_recommendation_candidate_v1.json")
    entry_contract = _load_json(review_dir / "shadow_meta_all_trade_review_entry_rl_observability_contract_v1.json")
    entry_status = _load_json(review_dir / "shadow_meta_all_trade_review_entry_rl_observability_status_v1.json")
    entry_snapshot_contract = _load_json(review_dir / "shadow_meta_all_trade_review_entry_policy_snapshot_contract_v1.json")
    management_contract = _load_json(review_dir / "shadow_meta_all_trade_review_management_rl_observation_contract_v1.json")
    management_summary = _load_json(review_dir / "shadow_meta_all_trade_review_management_rl_readiness_summary_v1.json")
    bandit_summary = _load_json(review_dir / "shadow_meta_all_trade_review_management_bandit_reward_candidate_summary_v1.json")

    ledger_df = pd.read_parquet(review_dir / "shadow_meta_all_trade_review_ledger_closed_trades.parquet")
    entry_df = pd.read_parquet(review_dir / "shadow_meta_all_trade_review_entry_rl_observability_view_v1.parquet")
    entry_terminal_df = pd.read_parquet(
        review_dir / "shadow_meta_all_trade_review_entry_actual_take_terminal_outcome_view_v1.parquet"
    )
    management_df = pd.read_parquet(review_dir / "shadow_meta_all_trade_review_management_rl_row_semantics_view_v1.parquet")
    transition_df = pd.read_parquet(review_dir / "shadow_meta_all_trade_review_management_rl_transition_eligible_view_v1.parquet")
    terminal_df = pd.read_parquet(
        review_dir / "shadow_meta_all_trade_review_management_rl_terminal_outcome_channel_view_v1.parquet"
    )
    bandit_df = pd.read_parquet(review_dir / "shadow_meta_all_trade_review_management_bandit_observed_sample_view_v1.parquet")
    unified_episode_df = pd.read_parquet(unified_dir / "shadow_meta_all_trade_review_rl_unified_episode_view_v1.parquet")
    unified_event_df = pd.read_parquet(unified_dir / "shadow_meta_all_trade_review_rl_unified_decision_event_view_v1.parquet")
    recommendation_trade_df = pd.read_parquet(
        recommendation_dir / "shadow_meta_all_trade_review_rl_recommendation_candidate_trade_view_v1.parquet"
    )
    recommendation_batch_df = pd.read_csv(
        recommendation_dir / "shadow_meta_all_trade_review_rl_recommendation_shadow_replay_15week_v1.csv"
    )

    expected_trades = int(rebuild_summary.get("headline", {}).get("trade_count") or len(ledger_df))
    rows: List[Dict[str, Any]] = []

    _append_check(
        rows,
        check_name="DOWNSTREAM_REBUILD_HAS_NO_BLOCKED_STEPS",
        status="PASS" if int(rebuild_summary.get("headline", {}).get("blocked_step_count", -1)) == 0 else "FAIL",
        observed=rebuild_summary.get("headline", {}).get("blocked_step_count"),
        expected=0,
        note="Canonical downstream rebuild must finish every step before RL artifacts can be trusted.",
    )
    _append_check(
        rows,
        check_name="LEDGER_ROW_COUNT_MATCHES_FOUNDATION",
        status="PASS" if len(ledger_df) == expected_trades else "FAIL",
        observed=len(ledger_df),
        expected=expected_trades,
        note="Closed trade ledger is the episode universe for RL observability.",
    )
    _append_check(
        rows,
        check_name="LEDGER_CANDIDATE_UID_IS_UNIQUE",
        status="PASS" if int(ledger_df["candidate_uid"].duplicated().sum()) == 0 else "FAIL",
        observed=int(ledger_df["candidate_uid"].duplicated().sum()),
        expected=0,
        note="Candidate UID must identify one closed-trade episode.",
    )
    ledger_bool_missing = {
        column: len(ledger_df) - _bool_text_count(ledger_df[column]) if column in ledger_df.columns else len(ledger_df)
        for column in BOOL_REWARD_CHANNELS
    }
    _append_check(
        rows,
        check_name="LEDGER_BOOL_REWARD_CHANNELS_FULLY_CANONICAL",
        status="PASS" if sum(ledger_bool_missing.values()) == 0 else "FAIL",
        observed=ledger_bool_missing,
        expected={column: 0 for column in BOOL_REWARD_CHANNELS},
        note="Ledger reward ontology channels must be TRUE/FALSE, never placeholders.",
    )
    entry_terminal_missing = {
        column: len(entry_terminal_df) - _bool_text_count(entry_terminal_df[column])
        if column in entry_terminal_df.columns
        else len(entry_terminal_df)
        for column in BOOL_REWARD_CHANNELS
    }
    _append_check(
        rows,
        check_name="ENTRY_TERMINAL_READ_MODEL_HAS_REWARD_LABELS",
        status="PASS" if sum(entry_terminal_missing.values()) == 0 else "FAIL",
        observed=entry_terminal_missing,
        expected={column: 0 for column in BOOL_REWARD_CHANNELS},
        note="Entry actual-take terminal read model must receive the same closed-trade truth labels as the ledger.",
    )

    entry_fields = set(entry_contract.get("observation_feature_names_v1", []))
    entry_missing_xgb = sorted(ENTRY_XGB_REQUIRED - entry_fields)
    _append_check(
        rows,
        check_name="ENTRY_OBSERVATION_CONTAINS_XGB_MODEL_SNAPSHOT",
        status="PASS" if not entry_missing_xgb else "FAIL",
        observed=entry_missing_xgb,
        expected=[],
        note="Entry RL must see the same XGB snapshot fields as the entry/read-model lane.",
    )
    entry_missing_columns = _all_present(entry_df, entry_fields)
    _append_check(
        rows,
        check_name="ENTRY_OBSERVATION_COLUMNS_EXIST_ON_VIEW",
        status="PASS" if not entry_missing_columns else "FAIL",
        observed=entry_missing_columns,
        expected=[],
        note="Every declared entry observation feature must exist physically in the entry RL view.",
    )
    entry_required_missing_counts = {
        column: _missing_count(entry_df[column]) if column in entry_df.columns else len(entry_df)
        for column in ENTRY_XGB_REQUIRED
    }
    _append_check(
        rows,
        check_name="ENTRY_XGB_FIELDS_DENSE_ON_DIRECT_ENTRY_VIEW",
        status="PASS" if sum(entry_required_missing_counts.values()) == 0 else "FAIL",
        observed=entry_required_missing_counts,
        expected={column: 0 for column in ENTRY_XGB_REQUIRED},
        note="Direct-entry RL rows must have dense XGB snapshot fields.",
    )
    _append_check(
        rows,
        check_name="ENTRY_PROPENSITY_STATUS_IS_EXPLICITLY_NOT_ESTABLISHED",
        status="PASS"
        if entry_status.get("ENTRY_PROPENSITY_STATUS") == "NOT_ESTABLISHED"
        and bool(entry_contract.get("propensity_not_established_v1"))
        else "FAIL",
        observed={
            "status": entry_status.get("ENTRY_PROPENSITY_STATUS"),
            "contract_flag": entry_contract.get("propensity_not_established_v1"),
        },
        expected={"status": "NOT_ESTABLISHED", "contract_flag": True},
        note="Entry can be observed and analyzed, but must not be treated as off-policy logged propensity truth yet.",
    )
    entry_leaks = _observation_leak_fields(entry_fields)
    snapshot_leaks = _observation_leak_fields(entry_snapshot_contract.get("model_snapshot_fields_v1", []))
    _append_check(
        rows,
        check_name="ENTRY_OBSERVATION_HAS_NO_HINDSIGHT_OR_REWARD_LEAKAGE",
        status="PASS" if not entry_leaks and not snapshot_leaks else "FAIL",
        observed={"entry_contract": entry_leaks, "snapshot_contract": snapshot_leaks},
        expected={"entry_contract": [], "snapshot_contract": []},
        note="Entry observation vectors must stay AS_OF/model-snapshot only.",
    )

    management_fields = set(management_contract.get("observation_vector_feature_names_v1", []))
    management_missing_xgb = sorted(MANAGEMENT_XGB_REQUIRED - management_fields)
    management_missing_exit = sorted(MANAGEMENT_EXIT_REQUIRED - management_fields)
    _append_check(
        rows,
        check_name="MANAGEMENT_OBSERVATION_CONTAINS_XGB_AND_EXIT_MODEL_FIELDS",
        status="PASS" if not management_missing_xgb and not management_missing_exit else "FAIL",
        observed={"missing_xgb": management_missing_xgb, "missing_exit": management_missing_exit},
        expected={"missing_xgb": [], "missing_exit": []},
        note="Management RL must see both entry/XGB context and exit-model/M1 management context.",
    )
    management_missing_columns = _all_present(management_df, management_fields)
    _append_check(
        rows,
        check_name="MANAGEMENT_OBSERVATION_COLUMNS_EXIST_ON_VIEW",
        status="PASS" if not management_missing_columns else "FAIL",
        observed=management_missing_columns,
        expected=[],
        note="Every declared management observation feature must exist physically in the row semantics view.",
    )
    required_coverage = management_summary.get("required_observation_feature_coverage_v1", {})
    required_not_dense = {
        field: payload
        for field, payload in required_coverage.items()
        if float(payload.get("available_share_v1", 0.0)) < 1.0
    }
    _append_check(
        rows,
        check_name="MANAGEMENT_REQUIRED_OBSERVATION_FEATURES_ARE_DENSE",
        status="PASS" if not required_not_dense else "FAIL",
        observed=required_not_dense,
        expected={},
        note="Required management observations must be dense; sparse signals need explicit availability masks.",
    )
    management_leaks = _observation_leak_fields(management_fields)
    _append_check(
        rows,
        check_name="MANAGEMENT_OBSERVATION_HAS_NO_HINDSIGHT_OR_REWARD_LEAKAGE",
        status="PASS" if not management_leaks else "FAIL",
        observed=management_leaks,
        expected=[],
        note="Management observation vector must not contain terminal/reward/hindsight truth.",
    )
    terminal_missing = {
        column: len(terminal_df) - _bool_text_count(terminal_df[column]) if column in terminal_df.columns else len(terminal_df)
        for column in TERMINAL_BOOL_REWARD_CHANNELS
    }
    _append_check(
        rows,
        check_name="MANAGEMENT_TERMINAL_BOOL_REWARD_CHANNELS_FULLY_CANONICAL",
        status="PASS" if sum(terminal_missing.values()) == 0 else "FAIL",
        observed=terminal_missing,
        expected={column: 0 for column in TERMINAL_BOOL_REWARD_CHANNELS},
        note="Management terminal outcome channel must carry TRUE/FALSE reward ontology labels.",
    )
    bandit_missing = {
        column: len(bandit_df) - _bool_text_count(bandit_df[column]) if column in bandit_df.columns else len(bandit_df)
        for column in BANDIT_BOOL_REWARD_CHANNELS
    }
    _append_check(
        rows,
        check_name="BANDIT_BOOL_REWARD_CHANNELS_FULLY_CANONICAL",
        status="PASS" if sum(bandit_missing.values()) == 0 else "FAIL",
        observed=bandit_missing,
        expected={column: 0 for column in BANDIT_BOOL_REWARD_CHANNELS},
        note="Bandit observed sample must receive the repaired reward labels.",
    )
    dm_count = int(bandit_summary.get("dm_eligible_row_count_v1", 0))
    reward_spec_coverage = {
        item.get("reward_spec_name_v1"): item.get("coverage_count_v1")
        for item in bandit_summary.get("candidate_reward_specs_built_v1", [])
    }
    reward_spec_gaps = {
        name: coverage
        for name, coverage in reward_spec_coverage.items()
        if int(coverage or 0) != dm_count
    }
    _append_check(
        rows,
        check_name="BANDIT_REWARD_SPECS_COVER_ALL_DM_ELIGIBLE_ROWS",
        status="PASS" if dm_count > 0 and not reward_spec_gaps else "FAIL",
        observed={"dm_eligible_rows": dm_count, "spec_coverage": reward_spec_coverage},
        expected={"all_specs": dm_count},
        note="Candidate scalar reward specs must cover every DM-eligible management row.",
    )

    _append_check(
        rows,
        check_name="UNIFIED_RL_COVERS_EVERY_CLOSED_TRADE",
        status="PASS"
        if int(unified_summary.get("closed_trade_ledger_episode_covered_count_v1", -1)) == expected_trades
        and int(unified_summary.get("failed_check_count_v1", -1)) == 0
        and len(unified_episode_df) == expected_trades
        else "FAIL",
        observed={
            "covered": unified_summary.get("closed_trade_ledger_episode_covered_count_v1"),
            "failed_checks": unified_summary.get("failed_check_count_v1"),
            "episode_rows": len(unified_episode_df),
        },
        expected={"covered": expected_trades, "failed_checks": 0, "episode_rows": expected_trades},
        note="Unified RL episode view must cover every closed ledger trade exactly once.",
    )
    _append_check(
        rows,
        check_name="UNIFIED_RL_EVENT_COUNTS_MATCH_ENTRY_AND_MANAGEMENT",
        status="PASS"
        if len(unified_event_df) == int(unified_summary.get("decision_event_rows_v1", -1))
        and int(unified_summary.get("entry_event_rows_v1", -1)) == len(entry_df)
        and int(unified_summary.get("management_event_rows_v1", -1)) == len(management_df)
        else "FAIL",
        observed={
            "event_rows": len(unified_event_df),
            "summary_event_rows": unified_summary.get("decision_event_rows_v1"),
            "entry_rows": len(entry_df),
            "summary_entry_events": unified_summary.get("entry_event_rows_v1"),
            "management_rows": len(management_df),
            "summary_management_events": unified_summary.get("management_event_rows_v1"),
        },
        expected="physical row counts equal summary row counts",
        note="Unified decision event view must be a faithful union of entry, management, and diagnostic ledger-only events.",
    )
    _append_check(
        rows,
        check_name="MANAGEMENT_POLICY_LOGGING_PROPENSITY_READY",
        status="PASS"
        if unified_summary.get("management_propensity_readiness_v1") == "READY_DETERMINISTIC_LOGGED_ACTION_PROPENSITY"
        and int(unified_summary.get("management_policy_logged_rows_v1", -1)) == len(transition_df)
        else "FAIL",
        observed={
            "readiness": unified_summary.get("management_propensity_readiness_v1"),
            "logged_rows": unified_summary.get("management_policy_logged_rows_v1"),
            "transition_rows": len(transition_df),
        },
        expected={
            "readiness": "READY_DETERMINISTIC_LOGGED_ACTION_PROPENSITY",
            "logged_rows": len(transition_df),
        },
        note="Management layer has deterministic logged action propensity for eligible rows.",
    )

    _append_check(
        rows,
        check_name="RECOMMENDATION_REPLAY_COVERS_FULL_LEDGER",
        status="PASS"
        if int(recommendation_summary.get("failed_check_count_v1", -1)) == 0
        and int(recommendation_summary.get("unified_episode_covered_trade_count_v1", -1)) == expected_trades
        and len(recommendation_trade_df) == expected_trades
        else "FAIL",
        observed={
            "failed_checks": recommendation_summary.get("failed_check_count_v1"),
            "covered": recommendation_summary.get("unified_episode_covered_trade_count_v1"),
            "trade_rows": len(recommendation_trade_df),
        },
        expected={"failed_checks": 0, "covered": expected_trades, "trade_rows": expected_trades},
        note="Recommendation replay must compare RL recommendations against every realized baseline trade.",
    )
    batch_coverage_gaps = recommendation_batch_df.loc[
        pd.to_numeric(recommendation_batch_df.get("unified_episode_coverage_rate_v1"), errors="coerce").fillna(0.0) < 1.0
    ]
    _append_check(
        rows,
        check_name="RECOMMENDATION_15WEEK_BATCHES_HAVE_FULL_EPISODE_COVERAGE",
        status="PASS" if batch_coverage_gaps.empty else "FAIL",
        observed=batch_coverage_gaps[["batch_index_v1", "unified_episode_coverage_rate_v1"]].to_dict(orient="records")
        if not batch_coverage_gaps.empty
        else [],
        expected=[],
        note="Every 15-week shadow batch must have full unified episode coverage.",
    )
    warning_count = int(recommendation_summary.get("warning_check_count_v1", 0))
    _append_check(
        rows,
        check_name="ENTRY_RETRAIN_SCOPE_WARNING_IS_EXPLICIT",
        status="PASS" if warning_count == 0 else "WARN",
        severity="WARN",
        observed={
            "warning_check_count_v1": warning_count,
            "entry_direct_feature_coverage_status_v1": recommendation_summary.get(
                "entry_direct_feature_coverage_status_v1"
            ),
        },
        expected={"warning_check_count_v1": 0},
        note="Full episode coverage is ready, but pure entry-direct retrain scope remains partial because management-only and ledger-only episodes exist.",
    )

    audit_df = pd.DataFrame.from_records(rows)
    fail_count = int((audit_df["status_v1"].astype("string") == "FAIL").sum())
    warn_count = int((audit_df["status_v1"].astype("string") == "WARN").sum())
    pass_count = int((audit_df["status_v1"].astype("string") == "PASS").sum())
    payload = {
        "layer_name": "RL_PIPELINE_INTEGRITY_AUDIT_V1",
        "reports_root_v1": str(reports_root),
        "review_dir_v1": str(review_dir),
        "unified_dir_v1": str(unified_dir),
        "recommendation_dir_v1": str(recommendation_dir),
        "overall_status_v1": "FAIL" if fail_count else ("PASS_WITH_WARNINGS" if warn_count else "PASS"),
        "check_count_v1": int(len(audit_df)),
        "pass_count_v1": pass_count,
        "warn_count_v1": warn_count,
        "fail_count_v1": fail_count,
        "headline_v1": {
            "closed_trades_v1": int(len(ledger_df)),
            "entry_direct_rows_v1": int(len(entry_df)),
            "management_rows_v1": int(len(management_df)),
            "management_transition_eligible_rows_v1": int(len(transition_df)),
            "bandit_rows_v1": int(len(bandit_df)),
            "unified_episode_rows_v1": int(len(unified_episode_df)),
            "recommendation_trade_rows_v1": int(len(recommendation_trade_df)),
            "recommendation_status_v1": recommendation_summary.get("status_v1", {}).get(
                "RL_RECOMMENDATION_CANDIDATE_STATUS"
            ),
        },
        "checks_v1": rows,
    }
    return {"summary": payload, "audit_df": audit_df}


def materialize_rl_pipeline_integrity_audit(
    reports_root: Path,
    *,
    review_dir: Path | None = None,
    unified_dir: Path | None = None,
    recommendation_dir: Path | None = None,
    extension_dir: Path | None = None,
) -> Dict[str, Any]:
    result = build_rl_pipeline_integrity_audit(
        reports_root,
        review_dir=review_dir,
        unified_dir=unified_dir,
        recommendation_dir=recommendation_dir,
    )
    summary = result["summary"]
    audit_df = result["audit_df"]
    reports_root = Path(summary["reports_root_v1"]).expanduser().resolve()
    if extension_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        extension_dir = reports_root / f"{LEDGER_NAMESPACE_PREFIX}{stamp}_{AUDIT_EXTENSION_SUFFIX}"
    extension_dir = Path(extension_dir).expanduser().resolve()
    extension_dir.mkdir(parents=True, exist_ok=False)
    extension_json = extension_dir / TOP_LEVEL_JSON
    extension_csv = extension_dir / TOP_LEVEL_CSV
    audit_df.to_csv(extension_csv, index=False)
    extension_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    top_summary = dict(summary)
    top_summary["extension_dir_v1"] = str(extension_dir)
    top_summary["audit_csv_v1"] = str(extension_csv)
    (reports_root / TOP_LEVEL_JSON).write_text(json.dumps(top_summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    audit_df.to_csv(reports_root / TOP_LEVEL_CSV, index=False)
    return {
        "extension_dir": extension_dir,
        "summary_path": reports_root / TOP_LEVEL_JSON,
        "audit_csv": reports_root / TOP_LEVEL_CSV,
        "summary": top_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit end-to-end RL pipeline wiring and reward integrity.")
    parser.add_argument("--reports-root", type=str, default=None)
    parser.add_argument("--review-dir", type=str, default=None)
    parser.add_argument("--unified-dir", type=str, default=None)
    parser.add_argument("--recommendation-dir", type=str, default=None)
    parser.add_argument("--extension-dir", type=str, default=None)
    args = parser.parse_args()

    result = materialize_rl_pipeline_integrity_audit(
        _resolve_reports_root(args.reports_root),
        review_dir=Path(args.review_dir).expanduser().resolve() if args.review_dir else None,
        unified_dir=Path(args.unified_dir).expanduser().resolve() if args.unified_dir else None,
        recommendation_dir=Path(args.recommendation_dir).expanduser().resolve() if args.recommendation_dir else None,
        extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None,
    )
    print(
        json.dumps(
            {
                "extension_dir": str(result["extension_dir"]),
                "summary_path": str(result["summary_path"]),
                "audit_csv": str(result["audit_csv"]),
                "overall_status_v1": result["summary"]["overall_status_v1"],
                "pass_count_v1": result["summary"]["pass_count_v1"],
                "warn_count_v1": result["summary"]["warn_count_v1"],
                "fail_count_v1": result["summary"]["fail_count_v1"],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
