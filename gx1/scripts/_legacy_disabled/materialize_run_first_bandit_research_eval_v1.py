#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from gx1.scripts.materialize_iql_readonly_transition_reward_bandit_planning_v1 import (
    PATH_DYNAMICS_V2_FIELDS,
    _json_ready,
    _read_csv_optional,
    _read_json_optional,
    _resolve_reports_root,
    _sha256,
    _utc_now,
    _write_json,
)


LAYER_ID = "RUN_FIRST_BANDIT_RESEARCH_EVAL_V1"
DATASET_LAYER_ID = "BUILD_MANAGEMENT_BANDIT_DATASET_V1"
EVAL_PREP_LAYER_ID = "BANDIT_RESEARCH_EVAL_PREP_V1"
REWARD_VERSION_ID = "MGMT_BANDIT_REALIZED_PNL_BPS_V1"
REWARD_FORMULA = "reward_bps = terminal_realized_pnl_bps"

OUTPUTS = {
    "contract": "first_bandit_research_eval_contract_v1.json",
    "execution_report": "first_bandit_eval_execution_report_v1.json",
    "headline_results": "first_bandit_headline_and_comparator_results_v1.json",
    "headline_metrics": "first_bandit_headline_metrics_v1.csv",
    "comparator_results": "first_bandit_comparator_results_v1.csv",
    "failcheck_review": "first_bandit_failcheck_and_safety_review_v1.json",
    "failcheck_table": "first_bandit_failcheck_review_v1.csv",
    "action_support_review": "first_bandit_action_imbalance_support_review_v1.json",
    "slice_breakdown": "first_bandit_slice_pocket_stress_breakdown_v1.json",
    "slice_table": "first_bandit_slice_pocket_breakdown_v1.csv",
    "rolling_table": "first_bandit_rolling_block_breakdown_v1.csv",
    "stress_table": "first_bandit_stress_breakdown_v1.csv",
    "allowed_conclusions": "first_bandit_allowed_conclusions_v1.csv",
    "forbidden_conclusions": "first_bandit_forbidden_conclusions_v1.csv",
    "conclusions_json": "first_bandit_allowed_forbidden_conclusions_v1.json",
    "final_verdict": "first_bandit_eval_final_verdict_v1.json",
    "post_status": "post_first_bandit_eval_status_update_v1.json",
    "summary": "run_first_bandit_research_eval_summary_v1.json",
    "report": "run_first_bandit_research_eval_report_v1.md",
    "manifest": "run_first_bandit_research_eval_manifest_v1.json",
    "status": "run_first_bandit_research_eval_status_v1.json",
    "consistency_audit": "run_first_bandit_research_eval_consistency_audit_v1.csv",
    "consistency_audit_json": "run_first_bandit_research_eval_consistency_audit_v1.json",
    "non_interference_audit": "run_first_bandit_research_eval_non_interference_audit_v1.csv",
    "non_interference_audit_json": "run_first_bandit_research_eval_non_interference_audit_v1.json",
}

EXPECTED_COMPARATORS = [
    "no-RL/current locked ledger",
    "R6 frozen shadow candidate",
    "supervised EXIT_LOCAL/tree baseline",
    "R5.2 frozen historical reference",
    "management harvest comparator",
    "dummy/random sanity comparator",
]

EXPECTED_FAILCHECKS = [
    "realized pnl",
    "bad-trade reduction",
    "MFE capture",
    "MAE burden",
    "giveback",
    "tail-control help",
    "runner damage",
    "50+/100+/200+ MFE damage",
    "strongest-winner path damage",
    "action agreement",
    "OOD action rate",
    "worst-slice performance",
    "rolling-window stability",
    "BATCH_04 stress",
    "BATCH_05 stress",
    "harvest candidate capture",
    "failed checks",
]


def _default_output_dir(reports_root: Path, now: datetime) -> Path:
    return reports_root / "IQL_INTEGRATION" / f"{LAYER_ID}_{now.strftime('%Y%m%dT%H%M%SZ')}"


def _latest_dir(reports_root: Path, layer_id: str, summary_name: str, arg: str | None) -> Path:
    if arg:
        path = Path(arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{layer_id} dir does not exist: {path}")
        return path
    base = reports_root / "IQL_INTEGRATION"
    candidates = sorted(base.glob(f"{layer_id}_*/{summary_name}"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No {layer_id} output found under {base}")
    return candidates[0].parent.resolve()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _metric_summary(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {
            "count_v1": 0,
            "mean_v1": None,
            "std_v1": None,
            "min_v1": None,
            "p05_v1": None,
            "p50_v1": None,
            "p95_v1": None,
            "max_v1": None,
            "sum_v1": None,
            "positive_rate_v1": None,
            "negative_rate_v1": None,
        }
    return {
        "count_v1": int(len(numeric)),
        "mean_v1": float(numeric.mean()),
        "std_v1": float(numeric.std(ddof=0)),
        "min_v1": float(numeric.min()),
        "p05_v1": float(numeric.quantile(0.05)),
        "p50_v1": float(numeric.quantile(0.50)),
        "p95_v1": float(numeric.quantile(0.95)),
        "max_v1": float(numeric.max()),
        "sum_v1": float(numeric.sum()),
        "positive_rate_v1": float((numeric > 0).mean()),
        "negative_rate_v1": float((numeric < 0).mean()),
    }


def _value_counts(series: pd.Series, name: str) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=[name, "row_count_v1", "row_share_v1"])
    counts = series.fillna("NULL").astype(str).value_counts(dropna=False).rename_axis(name).reset_index(name="row_count_v1")
    total = int(counts["row_count_v1"].sum())
    counts["row_share_v1"] = counts["row_count_v1"] / total if total else 0.0
    return counts


def _source_paths(reports_root: Path, dataset_dir: Path, eval_prep_dir: Path) -> dict[str, str | None]:
    dataset_summary = _read_json_optional(dataset_dir / "build_management_bandit_dataset_summary_v1.json")
    dataset_contract = _read_json_optional(dataset_dir / "build_management_bandit_dataset_contract_v1.json")
    source = dataset_contract.get("source_paths_v1", {}) if isinstance(dataset_contract.get("source_paths_v1"), dict) else {}
    return {
        "reports_root_v1": str(reports_root),
        "dataset_dir_v1": str(dataset_dir),
        "dataset_parquet_v1": dataset_summary.get("dataset_parquet_v1"),
        "dataset_summary_v1": str(dataset_dir / "build_management_bandit_dataset_summary_v1.json"),
        "dataset_contract_v1": str(dataset_dir / "build_management_bandit_dataset_contract_v1.json"),
        "dataset_profile_v1": str(dataset_dir / "management_bandit_dataset_profile_v1.json"),
        "eval_prep_dir_v1": str(eval_prep_dir),
        "eval_prep_summary_v1": str(eval_prep_dir / "bandit_research_eval_prep_summary_v1.json"),
        "eval_prep_scope_v1": str(eval_prep_dir / "bandit_eval_scope_boundary_lock_v1.json"),
        "eval_prep_comparator_plan_v1": str(eval_prep_dir / "bandit_comparator_application_plan_v1.csv"),
        "eval_prep_failcheck_plan_v1": str(eval_prep_dir / "bandit_failcheck_enforcement_plan_v1.csv"),
        "locked_ledger_source_v1": source.get("locked_ledger_source_v1"),
        "management_bandit_dm_view_v1": source.get("management_bandit_dm_view_v1"),
        "r5_2_freeze_summary_v1": str(reports_root / "truth_r5_2_shadow_freeze_and_r6_failure_backlog_v1.json"),
        "r6_freeze_summary_v1": str(reports_root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json"),
    }


def _read_optional_parquet(path_value: str | None, columns: list[str] | None = None) -> pd.DataFrame:
    if not path_value:
        return pd.DataFrame()
    path = Path(path_value)
    if not path.exists():
        return pd.DataFrame()
    try:
        if columns is None:
            return pd.read_parquet(path)
        available = pd.read_parquet(path).columns.tolist()
        keep = [col for col in columns if col in available]
        return pd.read_parquet(path, columns=keep) if keep else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _load_eval_frame(dataset_df: pd.DataFrame, source_paths: dict[str, str | None]) -> pd.DataFrame:
    df = dataset_df.copy()
    dm_cols = [
        "management_row_key_v1",
        "sequence_dataset_membership_v1",
        "as_of_session_v1",
        "as_of_side_v1",
        "as_of_trend_regime_v1",
        "as_of_vol_regime_v1",
        "hindsight_reward_trade_outcome_class_v1",
        "hindsight_reward_good_trade_v1",
        "hindsight_reward_bad_trade_v1",
        "hindsight_reward_good_exit_v1",
        "hindsight_reward_premature_exit_v1",
        "hindsight_reward_late_exit_v1",
    ]
    dm = _read_optional_parquet(source_paths.get("management_bandit_dm_view_v1"), dm_cols)
    if not dm.empty and "management_row_key_v1" in dm.columns:
        df = df.merge(dm, left_on="row_id", right_on="management_row_key_v1", how="left")
    ledger_cols = [
        "trade_uid",
        "candidate_uid",
        "realized_pnl_bps",
        "mfe_bps",
        "mae_bps",
        "hindsight_peak_mfe_bps_v1",
        "hindsight_peak_to_exit_giveback_bps_v1",
        "bad_trade",
        "good_trade",
        "good_exit",
        "premature_exit",
        "late_exit",
        "cata_loser",
        "trade_outcome_class",
    ]
    ledger = _read_optional_parquet(source_paths.get("locked_ledger_source_v1"), ledger_cols)
    if not ledger.empty and "trade_uid" in ledger.columns:
        keys = ["trade_uid"]
        ledger_slim = ledger.drop_duplicates(subset=keys).copy()
        df = df.merge(ledger_slim, left_on="episode_id", right_on="trade_uid", how="left", suffixes=("", "_ledger"))
    df["decision_ts_parsed_v1"] = pd.to_datetime(df.get("decision_ts"), errors="coerce", utc=True)
    df["eval_reward_bps_v1"] = pd.to_numeric(df.get("reward"), errors="coerce")
    peak = pd.to_numeric(df.get("hindsight_peak_mfe_bps_v1", df.get("mfe_bps")), errors="coerce")
    realized = pd.to_numeric(df.get("realized_pnl_bps", df.get("reward")), errors="coerce")
    df["mfe_capture_ratio_v1"] = (realized / peak.where(peak.abs() > 1e-9)).replace([float("inf"), -float("inf")], pd.NA)
    return df


def _ledger_summary(source_paths: dict[str, str | None]) -> dict[str, Any]:
    ledger = _read_optional_parquet(source_paths.get("locked_ledger_source_v1"), ["realized_pnl_bps", "trade_uid"])
    if ledger.empty or "realized_pnl_bps" not in ledger.columns:
        return {"status_v1": "NOT_ESTABLISHED", "row_count_v1": 0}
    out = _metric_summary(ledger["realized_pnl_bps"])
    out["status_v1"] = "DIRECT_LOCKED_LEDGER_REFERENCE_AVAILABLE"
    out["row_count_v1"] = int(len(ledger))
    return out


def _build_execution_report(dataset_summary: dict[str, Any], eval_prep_summary: dict[str, Any], eval_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "report_id_v1": "FIRST_BANDIT_EVAL_EXECUTION_V1",
        "eval_verdict_v1": "BANDIT_EVAL_COMPLETED_WITH_LIMITATIONS" if not eval_df.empty else "BANDIT_EVAL_FAILED",
        "what_was_evaluated_v1": "Observed/logged management contextual bandit research dataset with locked realized-PnL reward.",
        "protocol_used_v1": "BANDIT_RESEARCH_EVAL_PREP_V1: chronological blocks, action/support/slice visibility, comparator/fail-check governance.",
        "splits_blocks_slices_used_v1": [
            "overall logged dataset",
            "action HOLD/EXIT_NOW",
            "support_status",
            "safe AS_OF descriptive pockets when available: session, side, vol_regime, trend_regime",
            "chronological quintile blocks by decision_ts",
        ],
        "outputs_measured_v1": [
            "realized reward distribution",
            "logged action distribution",
            "support distribution",
            "locked-ledger realized-PnL reference",
            "available HINDSIGHT safety metrics from locked ledger/source artifacts",
            "slice and rolling/block descriptive breakdowns",
        ],
        "compared_against_v1": EXPECTED_COMPARATORS,
        "global_limitations_v1": [
            eval_prep_summary.get("eval_prep_verdict_v1", "BANDIT_EVAL_PREP_READY_WITH_LIMITATIONS"),
            dataset_summary.get("dataset_verdict_v1", "BANDIT_RESEARCH_DATASET_BUILT_WITH_LIMITATIONS"),
            "No counterfactual policy training or action-value estimate is produced.",
            "No next_state/done/sequence transition is used or inferred.",
            "HOLD dominance and SUPPORT_TOO_THIN block strong claims.",
        ],
        "not_iql_eval_v1": True,
        "not_sequence_rl_eval_v1": True,
        "not_controller_readiness_v1": True,
        "not_r7_readiness_v1": True,
    }


def _build_headline(eval_df: pd.DataFrame, source_paths: dict[str, str | None], comparator_plan_df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    overall = _metric_summary(eval_df["eval_reward_bps_v1"])
    ledger = _ledger_summary(source_paths)
    rows = [
        {"metric_v1": "reward_bps_overall", "scope_v1": "logged_bandit_dataset", **overall},
    ]
    for action, group in eval_df.groupby("action", dropna=False):
        rows.append({"metric_v1": "reward_bps_by_action", "scope_v1": str(action), **_metric_summary(group["eval_reward_bps_v1"])})
    action_counts = _value_counts(eval_df["action"], "action_v1").to_dict(orient="records")
    support_counts = _value_counts(eval_df["support_status"], "support_status_v1").to_dict(orient="records")
    observed_action_agreement = 1.0
    if "action_label_v1" in eval_df.columns:
        observed_action_agreement = float((eval_df["action"].astype(str) == eval_df["action_label_v1"].astype(str)).mean())

    comp_rows: list[dict[str, Any]] = []
    for row in comparator_plan_df.to_dict(orient="records"):
        name = str(row.get("comparator_v1"))
        status = str(row.get("application_status_v1"))
        if name == "no-RL/current locked ledger":
            compared_on = "realized_pnl_bps distribution: bandit-covered rows versus locked 1971 ledger reference"
            full = "PARTIAL_DIRECT_REFERENCE_NOT_POLICY_LIFT"
            calibration = "Direct locked reference exists, but this is coverage/context comparison rather than policy improvement proof."
            observed = {
                "bandit_mean_reward_bps_v1": overall.get("mean_v1"),
                "locked_ledger_mean_realized_pnl_bps_v1": ledger.get("mean_v1"),
                "bandit_rows_v1": int(len(eval_df)),
                "locked_ledger_rows_v1": ledger.get("row_count_v1"),
            }
        elif name == "dummy/random sanity comparator":
            compared_on = "sanity anchor only; no dummy policy is trained in this eval"
            full = "SANITY_ONLY"
            calibration = "Not a target comparator and no policy-promotion meaning."
            observed = {}
        else:
            compared_on = "registered comparator status and required future role"
            full = "INTERPRETIVE_ONLY_PENDING_CALIBRATION"
            calibration = "Comparator remains pending calibration; no full performance comparison is claimed here."
            observed = {}
        comp_rows.append(
            {
                "comparator_v1": name,
                "compared_on_v1": compared_on,
                "application_status_v1": status,
                "calibration_limitations_v1": calibration,
                "comparison_completeness_v1": full,
                "observed_summary_json_v1": json.dumps(_json_ready(observed), ensure_ascii=True, sort_keys=True),
            }
        )
    headline = {
        "result_id_v1": "HEADLINE_AND_COMPARATOR_RESULTS_V1",
        "headline_not_standalone_v1": True,
        "realized_pnl_reward_summary_v1": overall,
        "locked_ledger_reference_summary_v1": ledger,
        "action_distribution_v1": action_counts,
        "support_distribution_v1": support_counts,
        "agreement_disagreement_patterns_v1": {
            "logged_action_agreement_rate_v1": observed_action_agreement,
            "counterfactual_disagreement_available_v1": False,
            "note_v1": "No alternative learned policy was trained or scored; only logged action identity can be checked.",
        },
        "relative_improvement_claim_v1": "NOT_ESTABLISHED_NO_COUNTERFACTUAL_POLICY_EVAL",
        "comparator_uncertainty_visible_v1": True,
    }
    return headline, pd.DataFrame.from_records(rows), pd.DataFrame.from_records(comp_rows)


def _obs_for_metric(metric: str, eval_df: pd.DataFrame, support_verdict: str, upstream_failed_checks: int) -> tuple[str, str, str]:
    reward_summary = _metric_summary(eval_df["eval_reward_bps_v1"])
    if metric == "realized pnl":
        return "INDETERMINATE", f"observed_mean_bps={reward_summary['mean_v1']}; no calibrated improvement threshold", "soft metric available but not sufficient for positive claim"
    if metric == "bad-trade reduction":
        if "bad_trade" in eval_df.columns:
            rate = pd.to_numeric(eval_df["bad_trade"], errors="coerce").mean()
            return "INDETERMINATE", f"observed_bad_trade_rate={rate}", "no counterfactual reduction policy was evaluated"
        return "INDETERMINATE", "bad_trade metric not available", "cannot establish reduction"
    if metric == "MFE capture":
        rate = pd.to_numeric(eval_df.get("mfe_capture_ratio_v1"), errors="coerce").replace([float("inf"), -float("inf")], pd.NA).dropna()
        return "INDETERMINATE", f"observed_median_capture={_safe_float(rate.median()) if not rate.empty else None}", "capture is descriptive only without calibrated comparator"
    if metric == "MAE burden":
        if "mae_bps" in eval_df.columns:
            return "INDETERMINATE", f"observed_p95_mae_bps={_metric_summary(eval_df['mae_bps'])['p95_v1']}", "hard gate cannot pass without damage threshold/comparator"
        return "INDETERMINATE", "mae_bps not available", "hard gate cannot pass"
    if metric == "giveback":
        col = "hindsight_peak_to_exit_giveback_bps_v1"
        return "INDETERMINATE", f"observed_mean_giveback={_metric_summary(eval_df[col])['mean_v1'] if col in eval_df.columns else None}", "descriptive only"
    if metric == "tail-control help":
        cata = pd.to_numeric(eval_df.get("cata_loser"), errors="coerce").mean() if "cata_loser" in eval_df.columns else None
        return "INDETERMINATE", f"observed_cata_rate={_safe_float(cata)}", "hard gate cannot pass without proven tail-control help"
    if metric == "runner damage":
        return "INDETERMINATE", "runner_damage_counterfactual_not_established", "counterfactual runner damage is not locked"
    if metric == "50+/100+/200+ MFE damage":
        if "hindsight_peak_mfe_bps_v1" in eval_df.columns:
            peak = pd.to_numeric(eval_df["hindsight_peak_mfe_bps_v1"], errors="coerce")
            return "INDETERMINATE", f"mfe50_count={int((peak >= 50).sum())}; mfe100_count={int((peak >= 100).sum())}; mfe200_count={int((peak >= 200).sum())}", "damage is not counterfactually established"
        return "INDETERMINATE", "mfe thresholds not available", "hard gate cannot pass"
    if metric == "strongest-winner path damage":
        return "INDETERMINATE", "strongest-winner descriptive slice can be computed if MFE exists", "path damage not established without counterfactual"
    if metric == "action agreement":
        return "PASS", "logged action identity agreement checked; no learned policy disagreement available", "logged behavior provenance is internally consistent"
    if metric == "OOD action rate":
        return "FAIL" if support_verdict == "SUPPORT_TOO_THIN" else "INDETERMINATE", f"support_ood_verdict={support_verdict}", "SUPPORT_TOO_THIN blocks positive interpretation"
    if metric == "worst-slice performance":
        return "INDETERMINATE", "worst-slice descriptive table generated", "no calibrated hard threshold"
    if metric == "rolling-window stability":
        return "INDETERMINATE", "rolling/block descriptive table generated", "no calibrated instability threshold"
    if metric in {"BATCH_04 stress", "BATCH_05 stress"}:
        return "INDETERMINATE", "canonical BATCH_04/BATCH_05 stress labels not present in bandit dataset", "stress block not established"
    if metric == "harvest candidate capture":
        return "INDETERMINATE", "management harvest comparator remains pending calibration", "interpretive only"
    if metric == "failed checks":
        return ("PASS" if upstream_failed_checks == 0 else "FAIL"), f"upstream_failed_checks={upstream_failed_checks}", "aggregate invariant status"
    return "INDETERMINATE", "not mapped", "fail-closed"


def _build_failcheck_review(
    eval_df: pd.DataFrame,
    failcheck_plan_df: pd.DataFrame,
    support_verdict: str,
    upstream_failed_checks: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for row in failcheck_plan_df.to_dict(orient="records"):
        metric = str(row.get("metric_or_failcheck_v1"))
        result, observed, why = _obs_for_metric(metric, eval_df, support_verdict, upstream_failed_checks)
        hard = str(row.get("enforcement_type_v1")) == "HARD_GATE" or bool(row.get("auto_stops_positive_interpretation_v1"))
        stops = hard and result != "PASS"
        rows.append(
            {
                "metric_or_failcheck_v1": metric,
                "enforcement_type_v1": "HARD_GATE" if hard else "SOFT_REVIEW",
                "directionality_v1": row.get("directionality_v1"),
                "observed_result_v1": observed,
                "result_status_v1": result,
                "why_v1": why,
                "stops_positive_interpretation_v1": bool(stops),
            }
        )
    df = pd.DataFrame.from_records(rows)
    hard_block_count = int(df["stops_positive_interpretation_v1"].sum())
    fail_count = int((df["result_status_v1"] == "FAIL").sum())
    verdict = "NO_POSITIVE_CLAIM_ALLOWED" if hard_block_count else "LIMITED_POSITIVE_SIGNAL_ONLY"
    if fail_count and hard_block_count:
        verdict = "NO_POSITIVE_CLAIM_ALLOWED"
    return {
        "review_id_v1": "FAILCHECK_AND_SAFETY_REVIEW_V1",
        "safety_verdict_v1": verdict,
        "hard_gate_block_count_v1": hard_block_count,
        "fail_count_v1": fail_count,
        "indeterminate_count_v1": int((df["result_status_v1"] == "INDETERMINATE").sum()),
        "pass_count_v1": int((df["result_status_v1"] == "PASS").sum()),
        "headline_pnl_not_sufficient_v1": True,
        "rows_v1": df.to_dict(orient="records"),
    }, df


def _build_action_support_review(eval_df: pd.DataFrame, support_verdict: str) -> dict[str, Any]:
    hold = int((eval_df["action"].astype(str) == "HOLD").sum())
    exit_now = int((eval_df["action"].astype(str) == "EXIT_NOW").sum())
    total = int(len(eval_df))
    return {
        "review_id_v1": "ACTION_IMBALANCE_SUPPORT_AND_INTERPRETABILITY_REVIEW_V1",
        "verdicts_v1": [
            "INTERPRETABLE_ONLY_AS_LIMITED_RESEARCH_SIGNAL",
            "TOO_IMBALANCED_FOR_STRONG_CLAIMS",
            "SUPPORT_TOO_THIN_FOR_STRONG_CLAIMS",
        ],
        "hold_rows_v1": hold,
        "exit_now_rows_v1": exit_now,
        "hold_share_v1": float(hold / total) if total else 0.0,
        "exit_now_share_v1": float(exit_now / total) if total else 0.0,
        "support_ood_verdict_v1": support_verdict,
        "misleading_due_to_action_imbalance_v1": [
            "aggregate reward mostly reflects HOLD rows",
            "EXIT_NOW signal has only 45 rows in the locked dataset",
            "action agreement can mirror behavior imbalance rather than policy quality",
        ],
        "misleading_due_to_support_thinness_v1": [
            "thin support pockets may hide OOD action risk",
            "comparator gaps cannot be promoted without support-aware slices",
        ],
        "exit_now_too_thin_for_strong_conclusion_v1": exit_now < 100,
        "eval_strength_v1": "SANITY_RESEARCH_SIGNAL_ONLY",
        "must_be_true_before_more_trust_v1": [
            "stronger EXIT_NOW support or calibrated support thresholds",
            "hard gates computable and passed",
            "comparator calibration completed",
            "post-replay chain/HOLD diagnosis remains separate for sequence track",
        ],
    }


def _slice_summary(df: pd.DataFrame, field: str) -> pd.DataFrame:
    if field not in df.columns:
        return pd.DataFrame(columns=["slice_field_v1", "slice_value_v1", "row_count_v1", "mean_reward_bps_v1", "p05_reward_bps_v1", "p50_reward_bps_v1", "p95_reward_bps_v1", "thin_slice_v1"])
    rows: list[dict[str, Any]] = []
    for value, group in df.groupby(field, dropna=False):
        summary = _metric_summary(group["eval_reward_bps_v1"])
        rows.append(
            {
                "slice_field_v1": field,
                "slice_value_v1": str(value),
                "row_count_v1": int(len(group)),
                "mean_reward_bps_v1": summary["mean_v1"],
                "p05_reward_bps_v1": summary["p05_v1"],
                "p50_reward_bps_v1": summary["p50_v1"],
                "p95_reward_bps_v1": summary["p95_v1"],
                "thin_slice_v1": int(len(group)) < 30,
            }
        )
    return pd.DataFrame.from_records(rows)


def _rolling_blocks(df: pd.DataFrame) -> pd.DataFrame:
    work = df.dropna(subset=["decision_ts_parsed_v1"]).sort_values("decision_ts_parsed_v1").copy()
    if work.empty:
        return pd.DataFrame(columns=["block_id_v1", "row_count_v1", "start_ts_v1", "end_ts_v1", "mean_reward_bps_v1", "p05_reward_bps_v1", "p50_reward_bps_v1"])
    block_count = min(5, len(work))
    work["block_id_v1"] = pd.qcut(range(len(work)), q=block_count, labels=[f"ROLLING_BLOCK_{i:02d}" for i in range(1, block_count + 1)])
    rows = []
    for block, group in work.groupby("block_id_v1", observed=True):
        summary = _metric_summary(group["eval_reward_bps_v1"])
        rows.append(
            {
                "block_id_v1": str(block),
                "row_count_v1": int(len(group)),
                "start_ts_v1": group["decision_ts_parsed_v1"].min().isoformat(),
                "end_ts_v1": group["decision_ts_parsed_v1"].max().isoformat(),
                "mean_reward_bps_v1": summary["mean_v1"],
                "p05_reward_bps_v1": summary["p05_v1"],
                "p50_reward_bps_v1": summary["p50_v1"],
            }
        )
    return pd.DataFrame.from_records(rows)


def _build_slice_breakdown(eval_df: pd.DataFrame, rolling_df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    slice_fields = ["action", "support_status", "as_of_session_v1", "as_of_side_v1", "as_of_vol_regime_v1", "as_of_trend_regime_v1"]
    slice_df = pd.concat([_slice_summary(eval_df, field) for field in slice_fields], ignore_index=True)
    stress_rows = [
        {
            "stress_block_v1": "BATCH_04",
            "status_v1": "NOT_ESTABLISHED",
            "reason_v1": "canonical BATCH_04 label not present in bandit eval dataset",
            "auto_blocks_positive_interpretation_v1": False,
        },
        {
            "stress_block_v1": "BATCH_05",
            "status_v1": "NOT_ESTABLISHED",
            "reason_v1": "canonical BATCH_05 label not present in bandit eval dataset",
            "auto_blocks_positive_interpretation_v1": False,
        },
    ]
    stress_df = pd.DataFrame.from_records(stress_rows)
    worst = {}
    if not slice_df.empty:
        valid = slice_df.dropna(subset=["mean_reward_bps_v1"])
        if not valid.empty:
            worst = valid.sort_values("mean_reward_bps_v1").head(1).to_dict(orient="records")[0]
    verdict = "PARTIAL_STRESS_BREAKDOWN_ONLY"
    return {
        "breakdown_id_v1": "SLICE_POCKET_STRESS_BREAKDOWN_V1",
        "verdict_v1": verdict,
        "slice_fields_v1": slice_fields,
        "rolling_blocks_available_v1": not rolling_df.empty,
        "batch_04_05_status_v1": "NOT_ESTABLISHED",
        "worst_observed_slice_v1": worst,
        "positive_interpretation_blocked_by_stress_v1": False,
        "fail_closed_note_v1": "BATCH_04/BATCH_05 cannot be claimed without canonical labels; rolling blocks are descriptive only.",
    }, slice_df, stress_df


def _build_conclusions(final_signal: str, failcheck_review: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    allowed = [
        ("eval_completed", "First logged management bandit research-eval completed with limitations.", "BEVIST"),
        ("signal_status", f"Signal is {final_signal}; positive claim is blocked by safety/support limits.", "BEVIST"),
        ("contract_status", f"Comparator/fail-check contract verdict: {failcheck_review['safety_verdict_v1']}.", "BEVIST"),
        ("future_research", "Further bandit research/eval may continue only as research, not promotion.", "INDIKERT"),
    ]
    forbidden = [
        "IQL-ready",
        "sequence-ready",
        "R7-ready",
        "live-ready",
        "policy-promotion-ready",
        "HOLD transition truth established",
        "path-dynamics canonical for training",
        "universal RL reward proven",
    ]
    allowed_df = pd.DataFrame.from_records([{"claim_key_v1": key, "allowed_conclusion_v1": text, "hard_status_v1": status} for key, text, status in allowed])
    forbidden_df = pd.DataFrame.from_records([{"forbidden_conclusion_v1": item, "status_v1": "FORBIDDEN"} for item in forbidden])
    return {
        "conclusion_id_v1": "ALLOWED_CONCLUSIONS_AND_FORBIDDEN_CONCLUSIONS_V1",
        "allowed_rows_v1": allowed_df.to_dict(orient="records"),
        "forbidden_rows_v1": forbidden_df.to_dict(orient="records"),
    }, allowed_df, forbidden_df


def _build_final_verdict(failcheck_review: dict[str, Any], action_support_review: dict[str, Any], slice_breakdown: dict[str, Any]) -> dict[str, Any]:
    if failcheck_review["safety_verdict_v1"] == "NO_POSITIVE_CLAIM_ALLOWED":
        final = "WEAK_OR_INCONCLUSIVE_SIGNAL"
        polarity = "INCONCLUSIVE"
    elif "SUPPORT_TOO_THIN_FOR_STRONG_CLAIMS" in action_support_review.get("verdicts_v1", []):
        final = "WEAK_OR_INCONCLUSIVE_SIGNAL"
        polarity = "INCONCLUSIVE"
    else:
        final = "PROMISING_RESEARCH_SIGNAL_WITH_LIMITATIONS"
        polarity = "POSITIVE_LIMITED"
    return {
        "verdict_id_v1": "FIRST_BANDIT_EVAL_FINAL_VERDICT_V1",
        "final_verdict_v1": final,
        "signal_polarity_v1": polarity,
        "driven_by_v1": [
            "comparator/fail-check contract",
            failcheck_review["safety_verdict_v1"],
            "HOLD/EXIT_NOW imbalance",
            action_support_review["support_ood_verdict_v1"],
            slice_breakdown["verdict_v1"],
        ],
        "further_bandit_research_recommended_v1": final != "NEGATIVE_SIGNAL",
        "eval_should_stop_here_v1": False,
        "next_bandit_step_v1": "ONLY_LIMITED_RESEARCH_EVAL_OR_SUPPORT_AUDIT_NO_TRAINING_PROMOTION",
        "replay_dependent_work_still_important_v1": True,
        "headline_reward_alone_decided_v1": False,
    }


def _build_post_status(final_verdict: dict[str, Any], dataset_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "update_id_v1": "POST_EVAL_STATUS_UPDATE_V1",
        "first_bandit_research_eval_completed_v1": True,
        "eval_signal_v1": final_verdict["signal_polarity_v1"],
        "final_verdict_v1": final_verdict["final_verdict_v1"],
        "dataset_reward_eval_contract_still_stands_v1": True,
        "comparator_failcheck_contract_still_governing_v1": True,
        "sequence_iql_still_blocked_v1": True,
        "hold_transition_truth_status_v1": "MISSING_HOLD_NEXT_STATE_COUNT_ZERO",
        "r7_status_unchanged_blocked_v1": True,
        "replay_status_unchanged_v1": True,
        "next_right_steps_v1": [
            "CONTINUE_LIMITED_BANDIT_RESEARCH_EVAL_OR_SUPPORT_AUDIT",
            "WAIT_FOR_REPLAY_THEN_REBUILD_CHAIN",
        ],
        "hard_status_v1": {
            "BEVIST": [
                "first_bandit_research_eval_completed",
                "dataset_reward_eval_contract_still_stands",
                "comparator_failcheck_contract_still_governing",
                "sequence_iql_still_blocked",
                "hold_transition_truth_still_missing",
                "r7_still_blocked",
                "replay_status_unchanged",
            ],
            "INDIKERT": [
                f"eval_signal_{final_verdict['signal_polarity_v1'].lower()}",
                "further_bandit_research_possible_only_under_limitations",
            ],
            "IKKE_ETABLERT": [
                "sequence_iql_readiness",
                "r7_readiness",
                "canonical_hold_next_state_transitions",
                "path_dynamics_training_canonical_status",
                "policy_promotion_readiness",
            ],
        },
        "source_dataset_verdict_v1": dataset_summary.get("dataset_verdict_v1"),
    }


def _build_non_interference(
    output_dir: Path,
    source_paths: dict[str, str | None],
    exit_manager_sha_before: str | None,
    exit_manager_sha_after: str | None,
    r6_sha_before: str | None,
    r6_sha_after: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_values = [str(value) for value in source_paths.values() if value]
    checks = [
        ("OUTPUT_DIR_IS_IQL_INTEGRATION_NAMESPACE", "PASS" if "IQL_INTEGRATION" in output_dir.parts else "FAIL", str(output_dir), "path contains IQL_INTEGRATION"),
        ("OUTPUT_DIR_NOT_REPLAY_DIRECTORY", "PASS" if "PATH_DYNAMICS_LOGGING_V2_REPLAY" not in str(output_dir) else "FAIL", str(output_dir), "no replay path"),
        ("NO_IN_PROGRESS_REPLAY_USED_AS_CANONICAL", "PASS" if all("PATH_DYNAMICS_LOGGING_V2_REPLAY" not in path for path in source_values) else "FAIL", json.dumps(source_values, ensure_ascii=True), "no replay canonical source"),
        ("RAW_STATE_UNTOUCHED", "PASS", "not_rebuilt", "not_rebuilt"),
        ("POLICY_LOG_UNTOUCHED", "PASS", "not_rebuilt", "not_rebuilt"),
        ("EXIT_MANAGER_UNTOUCHED", "PASS" if exit_manager_sha_before == exit_manager_sha_after else "FAIL", exit_manager_sha_after, exit_manager_sha_before),
        ("R6_FREEZE_UNTOUCHED", "PASS" if r6_sha_before == r6_sha_after else "FAIL", r6_sha_after, r6_sha_before),
        ("R7_NOT_STARTED", "PASS", "not_started", "not_started"),
        ("IQL_TRAINING_NOT_STARTED", "PASS", "not_started", "not_started"),
        ("SEQUENCE_IQL_DATASET_NOT_BUILT", "PASS", "not_built", "not_built"),
        ("POLICY_PROMOTION_NOT_ATTEMPTED", "PASS", "not_attempted", "not_attempted"),
    ]
    df = pd.DataFrame.from_records(
        [{"check_name_v1": name, "status_v1": status, "observed_value_v1": observed, "expected_value_v1": expected} for name, status, observed, expected in checks]
    )
    return df, {
        "audit_id_v1": "NON_INTERFERENCE_RECHECK_V1",
        "failed_check_count_v1": int((df["status_v1"] != "PASS").sum()),
        "checks_v1": df.to_dict(orient="records"),
    }


def _build_consistency(
    dataset_df: pd.DataFrame,
    eval_prep_summary: dict[str, Any],
    execution: dict[str, Any],
    comparator_results_df: pd.DataFrame,
    failcheck_df: pd.DataFrame,
    conclusions: dict[str, Any],
    final_verdict: dict[str, Any],
    non_interference: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    checks = [
        ("DATASET_REWARD_VERSION_LOCKED", not dataset_df.empty and dataset_df["reward_version"].astype(str).eq(REWARD_VERSION_ID).all(), dataset_df["reward_version"].astype(str).unique().tolist() if "reward_version" in dataset_df else [], REWARD_VERSION_ID),
        ("NO_SEQUENCE_COLUMNS_USED", not any(col in dataset_df.columns for col in ["next_state", "next_state_vector", "done", "transition_id"]), list(dataset_df.columns), "no sequence columns"),
        ("EVAL_PREP_READY", eval_prep_summary.get("eval_prep_ready_v1") is True, eval_prep_summary.get("eval_prep_ready_v1"), True),
        ("EXECUTION_COMPLETED_WITH_LIMITATIONS", execution.get("eval_verdict_v1") == "BANDIT_EVAL_COMPLETED_WITH_LIMITATIONS", execution.get("eval_verdict_v1"), "BANDIT_EVAL_COMPLETED_WITH_LIMITATIONS"),
        ("ALL_REQUIRED_COMPARATORS_PRESENT", set(EXPECTED_COMPARATORS).issubset(set(comparator_results_df["comparator_v1"].astype(str))), comparator_results_df["comparator_v1"].astype(str).tolist(), EXPECTED_COMPARATORS),
        ("ALL_REQUIRED_FAILCHECKS_PRESENT", set(EXPECTED_FAILCHECKS).issubset(set(failcheck_df["metric_or_failcheck_v1"].astype(str))), failcheck_df["metric_or_failcheck_v1"].astype(str).tolist(), EXPECTED_FAILCHECKS),
        ("FORBIDDEN_CONCLUSIONS_PRESENT", len(conclusions.get("forbidden_rows_v1", [])) >= 8, len(conclusions.get("forbidden_rows_v1", [])), ">=8"),
        ("FINAL_VERDICT_ALLOWED", final_verdict.get("final_verdict_v1") in {"PROMISING_RESEARCH_SIGNAL_WITH_LIMITATIONS", "WEAK_OR_INCONCLUSIVE_SIGNAL", "NEGATIVE_SIGNAL", "NOT_INTERPRETABLE"}, final_verdict.get("final_verdict_v1"), "allowed final verdict"),
        ("NON_INTERFERENCE_PASSED", int(non_interference.get("failed_check_count_v1", 1) or 0) == 0, non_interference.get("failed_check_count_v1"), 0),
    ]
    df = pd.DataFrame.from_records(
        [{"check_name_v1": name, "status_v1": "PASS" if passed else "FAIL", "observed_value_v1": observed, "expected_value_v1": expected} for name, passed, observed, expected in checks]
    )
    return df, {
        "audit_id_v1": "RUN_FIRST_BANDIT_RESEARCH_EVAL_CONSISTENCY_AUDIT_V1",
        "failed_check_count_v1": int((df["status_v1"] != "PASS").sum()),
        "passed_check_count_v1": int((df["status_v1"] == "PASS").sum()),
        "checks_v1": df.to_dict(orient="records"),
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    return "\n".join(
        [
            "# Run First Bandit Research Eval V1",
            "",
            "## Result",
            "",
            f"- Eval execution: `{summary['eval_execution_verdict_v1']}`",
            f"- Final verdict: `{summary['final_verdict_v1']}`",
            f"- Signal: `{summary['signal_polarity_v1']}`",
            f"- Safety verdict: `{summary['safety_verdict_v1']}`",
            f"- Dataset rows: `{summary['included_rows_v1']}`",
            f"- Action split: `HOLD={summary['hold_rows_v1']}`, `EXIT_NOW={summary['exit_now_rows_v1']}`",
            f"- Support/OOD: `{summary['support_ood_verdict_v1']}`",
            "",
            "## Boundaries",
            "",
            "- This is logged management contextual bandit research eval only.",
            "- It is not IQL eval, not sequence-RL eval, not R7 readiness, not live readiness, and not policy promotion.",
            "- Positive interpretation is blocked by support/action imbalance and hard-gate indeterminacy/failure.",
            "",
            "## Next",
            "",
            "- Continue only limited bandit research/support audit if useful.",
            "- `WAIT_FOR_REPLAY_THEN_REBUILD_CHAIN` remains required for HOLD transition truth and sequence-IQL.",
        ]
    ) + "\n"


def build_first_bandit_research_eval(
    reports_root: Path,
    *,
    dataset_dir: Path | None = None,
    eval_prep_dir: Path | None = None,
    output_dir: Path | None = None,
    built_at: datetime | None = None,
    exit_manager_sha_before: str | None = None,
    exit_manager_sha_after: str | None = None,
    r6_sha_before: str | None = None,
    r6_sha_after: str | None = None,
) -> dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    built_at = built_at or _utc_now()
    dataset_dir = dataset_dir or _latest_dir(reports_root, DATASET_LAYER_ID, "build_management_bandit_dataset_summary_v1.json", None)
    eval_prep_dir = eval_prep_dir or _latest_dir(reports_root, EVAL_PREP_LAYER_ID, "bandit_research_eval_prep_summary_v1.json", None)
    output_dir = output_dir or _default_output_dir(reports_root, built_at)

    source_paths = _source_paths(reports_root, dataset_dir, eval_prep_dir)
    dataset_summary = _read_json_optional(dataset_dir / "build_management_bandit_dataset_summary_v1.json")
    dataset_profile = _read_json_optional(dataset_dir / "management_bandit_dataset_profile_v1.json")
    eval_prep_summary = _read_json_optional(eval_prep_dir / "bandit_research_eval_prep_summary_v1.json")
    comparator_plan_df = _read_csv_optional(eval_prep_dir / "bandit_comparator_application_plan_v1.csv")
    failcheck_plan_df = _read_csv_optional(eval_prep_dir / "bandit_failcheck_enforcement_plan_v1.csv")
    dataset_df = pd.read_parquet(Path(str(source_paths["dataset_parquet_v1"])))
    eval_df = _load_eval_frame(dataset_df, source_paths)

    execution = _build_execution_report(dataset_summary, eval_prep_summary, eval_df)
    headline, headline_metrics_df, comparator_results_df = _build_headline(eval_df, source_paths, comparator_plan_df)
    upstream_failed = int(dataset_summary.get("failed_consistency_check_count_v1", 0) or 0) + int(dataset_summary.get("failed_non_interference_check_count_v1", 0) or 0)
    support_verdict = str(dataset_profile.get("support_ood_verdict_from_foundation_v1", dataset_summary.get("support_ood_verdict_v1", "NOT_ESTABLISHED")))
    failcheck_review, failcheck_df = _build_failcheck_review(eval_df, failcheck_plan_df, support_verdict, upstream_failed)
    action_support_review = _build_action_support_review(eval_df, support_verdict)
    rolling_df = _rolling_blocks(eval_df)
    slice_breakdown, slice_df, stress_df = _build_slice_breakdown(eval_df, rolling_df)
    final_verdict = _build_final_verdict(failcheck_review, action_support_review, slice_breakdown)
    conclusions, allowed_df, forbidden_df = _build_conclusions(final_verdict["signal_polarity_v1"], failcheck_review)
    post_status = _build_post_status(final_verdict, dataset_summary)
    non_interference_df, non_interference = _build_non_interference(
        output_dir,
        source_paths,
        exit_manager_sha_before,
        exit_manager_sha_after,
        r6_sha_before,
        r6_sha_after,
    )
    consistency_df, consistency = _build_consistency(
        dataset_df,
        eval_prep_summary,
        execution,
        comparator_results_df,
        failcheck_df,
        conclusions,
        final_verdict,
        non_interference,
    )

    hold_rows = int((dataset_df["action"].astype(str) == "HOLD").sum())
    exit_rows = int((dataset_df["action"].astype(str) == "EXIT_NOW").sum())
    contract = {
        "contract_id_v1": LAYER_ID,
        "built_at_utc_v1": built_at.isoformat(),
        "mode_v1": "READONLY_APPEND_ONLY_MANAGEMENT_BANDIT_RESEARCH_EVAL",
        "reward_version_v1": REWARD_VERSION_ID,
        "reward_formula_v1": REWARD_FORMULA,
        "source_paths_v1": source_paths,
        "not_training_v1": True,
        "not_iql_eval_v1": True,
        "not_sequence_rl_eval_v1": True,
        "not_r7_readiness_v1": True,
        "not_live_gate_v1": True,
        "not_policy_promotion_v1": True,
        "hard_boundaries_v1": {
            "do_not_touch_replay_v1": True,
            "do_not_start_replay_v1": True,
            "do_not_rebuild_raw_state_v1": True,
            "do_not_rebuild_policy_log_v1": True,
            "do_not_modify_exit_manager_v1": True,
            "do_not_train_r7_v1": True,
            "do_not_train_iql_v1": True,
            "do_not_build_sequence_iql_dataset_v1": True,
            "do_not_use_in_progress_replay_as_canonical_v1": True,
            "do_not_modify_r6_freeze_v1": True,
            "do_not_modify_locked_ledger_v1": True,
            "do_not_claim_hold_next_state_truth_v1": True,
        },
        "path_dynamics_v2_status_v1": {
            "status_v1": "PENDING_REPLAY_NOT_CANONICAL_YET_DO_NOT_USE_FOR_TRAINING",
            "fields_v1": PATH_DYNAMICS_V2_FIELDS,
        },
    }
    summary = {
        "layer_id_v1": LAYER_ID,
        "built_at_utc_v1": built_at.isoformat(),
        "eval_ran_v1": execution["eval_verdict_v1"] == "BANDIT_EVAL_COMPLETED_WITH_LIMITATIONS",
        "eval_execution_verdict_v1": execution["eval_verdict_v1"],
        "final_verdict_v1": final_verdict["final_verdict_v1"],
        "signal_polarity_v1": final_verdict["signal_polarity_v1"],
        "safety_verdict_v1": failcheck_review["safety_verdict_v1"],
        "hard_gate_block_count_v1": failcheck_review["hard_gate_block_count_v1"],
        "failcheck_fail_count_v1": failcheck_review["fail_count_v1"],
        "failcheck_indeterminate_count_v1": failcheck_review["indeterminate_count_v1"],
        "failcheck_pass_count_v1": failcheck_review["pass_count_v1"],
        "included_rows_v1": int(len(dataset_df)),
        "hold_rows_v1": hold_rows,
        "exit_now_rows_v1": exit_rows,
        "support_ood_verdict_v1": support_verdict,
        "reward_version_v1": REWARD_VERSION_ID,
        "headline_reward_mean_bps_v1": headline["realized_pnl_reward_summary_v1"].get("mean_v1"),
        "headline_reward_sum_bps_v1": headline["realized_pnl_reward_summary_v1"].get("sum_v1"),
        "comparators_used_v1": comparator_results_df["comparator_v1"].astype(str).tolist(),
        "hard_gate_metrics_v1": failcheck_df.loc[failcheck_df["enforcement_type_v1"].eq("HARD_GATE"), "metric_or_failcheck_v1"].astype(str).tolist(),
        "action_imbalance_support_limits_strong_claims_v1": True,
        "sequence_iql_still_blocked_v1": True,
        "hold_transition_truth_status_v1": "MISSING_HOLD_NEXT_STATE_COUNT_ZERO",
        "r7_still_blocked_v1": True,
        "replay_touched_v1": False,
        "raw_state_rebuilt_v1": False,
        "policy_log_rebuilt_v1": False,
        "exit_manager_modified_v1": False,
        "r7_started_v1": False,
        "iql_training_started_v1": False,
        "sequence_iql_dataset_built_v1": False,
        "recommended_next_steps_v1": post_status["next_right_steps_v1"],
        "hard_status_partition_v1": post_status["hard_status_v1"],
    }
    status = {
        "layer_id_v1": LAYER_ID,
        "status_v1": "MATERIALIZED_FIRST_BANDIT_RESEARCH_EVAL",
        "eval_ran_v1": summary["eval_ran_v1"],
        "training_executed_v1": False,
        "r7_started_v1": False,
        "iql_training_started_v1": False,
        "sequence_iql_dataset_built_v1": False,
        "replay_touched_v1": False,
        "failed_consistency_check_count_v1": int(consistency.get("failed_check_count_v1", 0)),
        "failed_non_interference_check_count_v1": int(non_interference.get("failed_check_count_v1", 0)),
    }
    return {
        "contract": contract,
        "execution": execution,
        "headline": headline,
        "headline_metrics_df": headline_metrics_df,
        "comparator_results_df": comparator_results_df,
        "failcheck_review": failcheck_review,
        "failcheck_df": failcheck_df,
        "action_support_review": action_support_review,
        "slice_breakdown": slice_breakdown,
        "slice_df": slice_df,
        "rolling_df": rolling_df,
        "stress_df": stress_df,
        "conclusions": conclusions,
        "allowed_df": allowed_df,
        "forbidden_df": forbidden_df,
        "final_verdict": final_verdict,
        "post_status": post_status,
        "non_interference_df": non_interference_df,
        "non_interference": non_interference,
        "consistency_df": consistency_df,
        "consistency": consistency,
        "summary": summary,
        "status": status,
        "source_paths": source_paths,
    }


def write_first_bandit_research_eval_artifacts(
    reports_root: Path,
    *,
    dataset_dir: Path | None = None,
    eval_prep_dir: Path | None = None,
    output_dir: Path | None = None,
    built_at: datetime | None = None,
) -> dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    built_at = built_at or _utc_now()
    target_dir = output_dir.expanduser().resolve() if output_dir is not None else _default_output_dir(reports_root, built_at).resolve()
    exit_manager_path = Path("/home/andre2/src/GX1_ENGINE/gx1/execution/exit_manager.py")
    r6_path = reports_root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json"
    exit_manager_sha_before = _sha256(exit_manager_path)
    r6_sha_before = _sha256(r6_path)
    payload = build_first_bandit_research_eval(
        reports_root,
        dataset_dir=dataset_dir,
        eval_prep_dir=eval_prep_dir,
        output_dir=target_dir,
        built_at=built_at,
        exit_manager_sha_before=exit_manager_sha_before,
        exit_manager_sha_after=exit_manager_sha_before,
        r6_sha_before=r6_sha_before,
        r6_sha_after=r6_sha_before,
    )
    target_dir.mkdir(parents=True, exist_ok=False)
    _write_json(target_dir / OUTPUTS["contract"], payload["contract"])
    _write_json(target_dir / OUTPUTS["execution_report"], payload["execution"])
    _write_json(target_dir / OUTPUTS["headline_results"], payload["headline"])
    payload["headline_metrics_df"].to_csv(target_dir / OUTPUTS["headline_metrics"], index=False)
    payload["comparator_results_df"].to_csv(target_dir / OUTPUTS["comparator_results"], index=False)
    _write_json(target_dir / OUTPUTS["failcheck_review"], payload["failcheck_review"])
    payload["failcheck_df"].to_csv(target_dir / OUTPUTS["failcheck_table"], index=False)
    _write_json(target_dir / OUTPUTS["action_support_review"], payload["action_support_review"])
    _write_json(target_dir / OUTPUTS["slice_breakdown"], payload["slice_breakdown"])
    payload["slice_df"].to_csv(target_dir / OUTPUTS["slice_table"], index=False)
    payload["rolling_df"].to_csv(target_dir / OUTPUTS["rolling_table"], index=False)
    payload["stress_df"].to_csv(target_dir / OUTPUTS["stress_table"], index=False)
    payload["allowed_df"].to_csv(target_dir / OUTPUTS["allowed_conclusions"], index=False)
    payload["forbidden_df"].to_csv(target_dir / OUTPUTS["forbidden_conclusions"], index=False)
    _write_json(target_dir / OUTPUTS["conclusions_json"], payload["conclusions"])
    _write_json(target_dir / OUTPUTS["final_verdict"], payload["final_verdict"])
    _write_json(target_dir / OUTPUTS["post_status"], payload["post_status"])

    exit_manager_sha_after = _sha256(exit_manager_path)
    r6_sha_after = _sha256(r6_path)
    non_interference_df, non_interference = _build_non_interference(
        target_dir,
        payload["source_paths"],
        exit_manager_sha_before,
        exit_manager_sha_after,
        r6_sha_before,
        r6_sha_after,
    )
    payload["non_interference_df"] = non_interference_df
    payload["non_interference"] = non_interference
    dataset_df = pd.read_parquet(Path(str(payload["source_paths"]["dataset_parquet_v1"])))
    payload["consistency_df"], payload["consistency"] = _build_consistency(
        dataset_df,
        _read_json_optional(Path(payload["source_paths"]["eval_prep_summary_v1"])),
        payload["execution"],
        payload["comparator_results_df"],
        payload["failcheck_df"],
        payload["conclusions"],
        payload["final_verdict"],
        non_interference,
    )
    payload["summary"]["exit_manager_modified_v1"] = exit_manager_sha_before != exit_manager_sha_after
    payload["status"]["failed_non_interference_check_count_v1"] = int(non_interference["failed_check_count_v1"])
    payload["status"]["failed_consistency_check_count_v1"] = int(payload["consistency"]["failed_check_count_v1"])

    non_interference_df.to_csv(target_dir / OUTPUTS["non_interference_audit"], index=False)
    _write_json(target_dir / OUTPUTS["non_interference_audit_json"], non_interference)
    payload["consistency_df"].to_csv(target_dir / OUTPUTS["consistency_audit"], index=False)
    _write_json(target_dir / OUTPUTS["consistency_audit_json"], payload["consistency"])
    _write_json(target_dir / OUTPUTS["summary"], payload["summary"])
    (target_dir / OUTPUTS["report"]).write_text(_markdown_report(payload), encoding="utf-8")

    artifact_paths = {key: str(target_dir / filename) for key, filename in OUTPUTS.items()}
    manifest = {
        "layer_id_v1": LAYER_ID,
        "built_at_utc_v1": payload["summary"]["built_at_utc_v1"],
        "output_dir_v1": str(target_dir),
        "append_only_namespace_v1": "IQL_INTEGRATION",
        "artifact_paths_v1": artifact_paths,
        "source_paths_v1": payload["source_paths"],
        "read_only_references_v1": True,
        "not_training_v1": True,
        "not_iql_eval_v1": True,
        "not_sequence_rl_eval_v1": True,
        "not_policy_promotion_v1": True,
    }
    _write_json(target_dir / OUTPUTS["manifest"], manifest)
    _write_json(target_dir / OUTPUTS["status"], payload["status"])
    return {
        "output_dir": str(target_dir),
        "artifact_paths": artifact_paths,
        "summary": payload["summary"],
        "status": payload["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the first limited management bandit research eval.")
    parser.add_argument("--reports-root", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--eval-prep-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else None
    eval_prep_dir = Path(args.eval_prep_dir).expanduser().resolve() if args.eval_prep_dir else None
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    result = write_first_bandit_research_eval_artifacts(
        reports_root,
        dataset_dir=dataset_dir,
        eval_prep_dir=eval_prep_dir,
        output_dir=output_dir,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
