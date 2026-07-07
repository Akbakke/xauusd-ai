#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
LEDGER_NAMESPACE_PREFIX = "ALL_TRADE_REVIEW_LEDGER_"
HARVEST_EXTENSION_SUFFIX = "EXIT_HARVEST_POLICY_CANDIDATE_V1"

HARVEST_CONTRACT = "shadow_meta_all_trade_review_exit_harvest_policy_candidate_contract_v1.json"
HARVEST_QUALITY_VIEW = "shadow_meta_all_trade_review_harvest_quality_trade_view_v1.parquet"
HARVEST_POLICY_VIEW = "shadow_meta_all_trade_review_exit_harvest_policy_candidate_trade_view_v1.parquet"
HARVEST_MODEL_TARGET_VIEW = "shadow_meta_all_trade_review_harvest_model_adjustment_target_view_v1.parquet"
HARVEST_BATCH_REPLAY = "shadow_meta_all_trade_review_exit_harvest_shadow_replay_15week_v1.csv"
HARVEST_SUMMARY = "shadow_meta_all_trade_review_exit_harvest_policy_candidate_summary_v1.json"
HARVEST_STATUS = "shadow_meta_all_trade_review_exit_harvest_policy_candidate_status_v1.json"
HARVEST_AUDIT = "shadow_meta_all_trade_review_exit_harvest_policy_candidate_consistency_audit_v1.csv"
HARVEST_MANIFEST = "shadow_meta_all_trade_review_exit_harvest_policy_candidate_manifest_v1.json"
HARVEST_MD = "shadow_meta_all_trade_review_exit_harvest_policy_candidate_v1.md"
TOP_LEVEL_SUMMARY = "truth_exit_harvest_policy_candidate_v1.json"

RUN_RE = re.compile(r"^E2E_SANITY_ORDERFIX_(\d{8})_(\d{8})$")
MIN_RUNNER_MFE_BPS_V1 = 50.0
HOME_RUN_MFE_BPS_V1 = 200.0
LOW_CAPTURE_RATIO_V1 = 0.35
GOOD_CAPTURE_RATIO_V1 = 0.65
REWARD_CLIP_BPS_V1 = 200.0


LEDGER_REQUIRED_COLUMNS = [
    "run_id",
    "candidate_uid",
    "trade_uid",
    "trade_id",
    "decision_timestamp",
    "entry_timestamp",
    "exit_timestamp",
    "realized_pnl_bps",
    "mfe_bps",
    "mae_bps",
    "hindsight_peak_mfe_bps_v1",
    "hindsight_peak_to_exit_giveback_bps_v1",
    "hindsight_hold_longer_extra_value_bps_v1",
    "hindsight_exit_earlier_saved_bps_v1",
    "hindsight_skip_trade_avoided_loss_bps_v1",
    "hindsight_should_skip_trade_v1",
    "hindsight_should_hold_longer_v1",
    "hindsight_should_exit_earlier_v1",
    "good_trade",
    "good_trade_mfe20_mae5",
    "bad_trade",
    "good_exit",
    "premature_exit",
    "late_exit",
    "exit_reason",
    "trade_outcome_class",
    "session",
    "vol_regime",
    "trend_regime",
    "used_for_training",
    "used_for_validation",
    "used_for_holdout",
]

RECOMMENDATION_REQUIRED_COLUMNS = [
    "candidate_uid",
    "rl_priority_recommendation_v1",
    "rl_priority_counterfactual_delta_bps_v1",
    "rl_priority_entry_skip_delta_bps_v1",
    "rl_priority_exit_earlier_delta_bps_v1",
    "rl_priority_hold_longer_delta_bps_v1",
    "rl_recommendation_semantics_v1",
    "unified_episode_coverage_status_v1",
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


def _resolve_review_dir(reports_root: Path, review_dir_arg: str | None) -> Path:
    if review_dir_arg:
        review_dir = Path(review_dir_arg).expanduser().resolve()
        if not review_dir.exists():
            raise FileNotFoundError(f"Review dir does not exist: {review_dir}")
        return review_dir
    summary_path = reports_root / "truth_downstream_canonical_rebuild_v1.json"
    if summary_path.exists():
        raw_dir = _load_json(summary_path).get("ledger_dir")
        if isinstance(raw_dir, str) and raw_dir.strip():
            candidate = Path(raw_dir).expanduser().resolve()
            if (candidate / "shadow_meta_all_trade_review_ledger_closed_trades.parquet").exists():
                return candidate
    raise FileNotFoundError("Could not resolve canonical review dir from truth_downstream_canonical_rebuild_v1.json.")


def _resolve_recommendation_dir(reports_root: Path, recommendation_dir_arg: str | None) -> Path:
    if recommendation_dir_arg:
        recommendation_dir = Path(recommendation_dir_arg).expanduser().resolve()
        if not recommendation_dir.exists():
            raise FileNotFoundError(f"Recommendation dir does not exist: {recommendation_dir}")
        return recommendation_dir
    top_summary_path = reports_root / "truth_rl_recommendation_candidate_v1.json"
    if top_summary_path.exists():
        raw_dir = _load_json(top_summary_path).get("extension_dir_v1")
        if isinstance(raw_dir, str) and raw_dir.strip():
            candidate = Path(raw_dir).expanduser().resolve()
            if (candidate / "shadow_meta_all_trade_review_rl_recommendation_candidate_trade_view_v1.parquet").exists():
                return candidate
    namespace_dirs = sorted(
        [
            path
            for path in reports_root.iterdir()
            if path.is_dir()
            and path.name.startswith(LEDGER_NAMESPACE_PREFIX)
            and path.name.endswith("RL_RECOMMENDATION_CANDIDATE_V1")
        ],
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate in namespace_dirs:
        if (candidate / "shadow_meta_all_trade_review_rl_recommendation_candidate_trade_view_v1.parquet").exists():
            return candidate
    raise FileNotFoundError("Could not resolve RL recommendation candidate dir.")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, artifact_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{artifact_name} is missing required columns: {missing}")


def _num_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f"Missing required numeric column: {column}")
    series = pd.to_numeric(frame[column], errors="coerce")
    if bool(series.isna().any()):
        raise ValueError(f"Column {column} contains null/non-numeric values; refusing to synthesize harvest metrics.")
    return series.astype(float)


def _optional_num_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f"Missing required numeric column: {column}")
    return pd.to_numeric(frame[column], errors="coerce")


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f"Missing required boolean column: {column}")
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    valid = normalized.isin(["true", "false"]) | normalized.isna()
    if not bool(valid.all()):
        bad_values = sorted(normalized.loc[~valid].dropna().unique().tolist())
        raise ValueError(f"Column {column} contains non-boolean values: {bad_values[:10]}")
    return normalized.eq("true").fillna(False).astype(bool)


def _truth_text_bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f"Missing required TRUE/FALSE truth column: {column}")
    normalized = frame[column].astype("string").str.strip().str.upper()
    valid = normalized.isin(["TRUE", "FALSE"])
    if not bool(valid.all()):
        bad_values = sorted(normalized.loc[~valid].dropna().unique().tolist())
        raise ValueError(f"Column {column} contains non TRUE/FALSE values: {bad_values[:10]}")
    return normalized.eq("TRUE")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = numerator.astype(float) / denominator.where(denominator.gt(0.0))
    return out.replace([float("inf"), float("-inf")], pd.NA)


def _counts(frame: pd.DataFrame, column: str) -> Dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].astype("string").value_counts(dropna=False).to_dict().items()
    }


def _sum_numeric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def _mean_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def _quantiles(series: pd.Series) -> Dict[str, float | None]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None, "p99": None}
    return {
        "p10": float(numeric.quantile(0.10)),
        "p25": float(numeric.quantile(0.25)),
        "p50": float(numeric.quantile(0.50)),
        "p75": float(numeric.quantile(0.75)),
        "p90": float(numeric.quantile(0.90)),
        "p99": float(numeric.quantile(0.99)),
    }


def _run_sort_key(run_id: str) -> str:
    match = RUN_RE.match(str(run_id))
    return match.group(1) if match else str(run_id)


def _all_run_ids(reports_root: Path, ledger_df: pd.DataFrame) -> List[str]:
    runs_root = reports_root / "runs"
    if runs_root.exists():
        run_ids = sorted(
            [path.name for path in runs_root.iterdir() if path.is_dir() and RUN_RE.match(path.name)],
            key=_run_sort_key,
        )
        if run_ids:
            return run_ids
    return sorted(ledger_df["run_id"].astype("string").dropna().unique().tolist(), key=_run_sort_key)


def _mfe_band(peak_mfe: float) -> str:
    if peak_mfe < 0.0:
        return "NEGATIVE_MFE"
    if peak_mfe < 20.0:
        return "MFE_LT_20"
    if peak_mfe < 50.0:
        return "MFE_20_50"
    if peak_mfe < 100.0:
        return "MFE_50_100"
    if peak_mfe < HOME_RUN_MFE_BPS_V1:
        return "MFE_100_200"
    if peak_mfe < 500.0:
        return "MFE_200_500"
    return "MFE_500_PLUS"


def _render_markdown(summary: Dict[str, Any], batch_df: pd.DataFrame) -> str:
    lines = [
        "# Exit Harvest Policy Candidate V1",
        "",
        "Dette er et shadow/retrain-target lag. Det er ikke live fill, ikke trainer, og ikke controller.",
        "",
        "## Headline",
        "",
        f"- Status: `{summary['status_v1']['EXIT_HARVEST_POLICY_CANDIDATE_STATUS']}`",
        f"- Trades: `{summary['trade_count_v1']}`",
        f"- Realized PnL bps: `{summary['baseline_total_pnl_bps_v1']:.2f}`",
        f"- Peak MFE bps: `{summary['peak_mfe_total_bps_v1']:.2f}`",
        f"- Giveback bps: `{summary['giveback_total_bps_v1']:.2f}`",
        f"- Portfolio capture ratio: `{summary['portfolio_capture_ratio_v1']:.4f}`",
        f"- Harvest delta target bps: `{summary['harvest_priority_delta_bps_v1']:.2f}`",
        f"- Home-run 200bps opportunities: `{summary['home_run_200bps_opportunity_count_v1']}`",
        f"- 15-week batches: `{summary['batch_count_v1']}`",
        "",
        "## Batch Replay",
        "",
        "| batch | runs | trades | pnl | peak mfe | giveback | capture | harvest delta | hold | exit earlier | skip | home-run opp |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in batch_df.to_dict(orient="records"):
        capture = row.get("portfolio_capture_ratio_v1")
        capture_text = "NA" if capture is None or pd.isna(capture) else f"{float(capture):.4f}"
        lines.append(
            "| {batch_index_v1} | {run_count_v1} | {trade_count_v1} | {baseline_total_pnl_bps_v1:.2f} | "
            "{peak_mfe_total_bps_v1:.2f} | {giveback_total_bps_v1:.2f} | "
            f"{capture_text} | "
            "{harvest_priority_delta_bps_v1:.2f} | {hold_longer_count_v1} | {exit_earlier_count_v1} | "
            "{skip_trade_count_v1} | {home_run_200bps_opportunity_count_v1} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Model Update Contract",
            "",
            "- Entry/XGB/entry-transformer bruker target-viewen som label/weight-kilde, ikke som live feature.",
            "- Exit-transformer bruker management labels som HOLD_LONGER, EXIT_EARLIER, RUNNER_TRAIL_REVIEW og KEEP_BASELINE.",
            "- RL bruker raw reward og en clipped reward-kanal for stabilitet; begge er deterministisk avledet fra truth, ikke fabricert fill.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_exit_harvest_policy_candidate_payload(
    *,
    reports_root: Path,
    review_dir: Path,
    recommendation_dir: Path,
    batch_weeks: int = 15,
) -> Dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    ledger_df = pd.read_parquet(review_dir / "shadow_meta_all_trade_review_ledger_closed_trades.parquet")
    recommendation_df = pd.read_parquet(
        recommendation_dir / "shadow_meta_all_trade_review_rl_recommendation_candidate_trade_view_v1.parquet"
    )
    recommendation_summary = _load_json(
        recommendation_dir / "shadow_meta_all_trade_review_rl_recommendation_candidate_summary_v1.json"
    )
    recommendation_status = _load_json(
        recommendation_dir / "shadow_meta_all_trade_review_rl_recommendation_candidate_status_v1.json"
    )
    if int(batch_weeks) <= 0:
        raise ValueError("batch_weeks must be positive")
    if ledger_df.empty:
        raise RuntimeError("EXIT_HARVEST_POLICY_CANDIDATE_V1 requires non-empty closed-trade ledger.")
    if int(recommendation_summary.get("failed_check_count_v1", -1)) != 0:
        raise RuntimeError("EXIT_HARVEST_POLICY_CANDIDATE_V1 requires recommendation failed_check_count_v1 == 0.")
    if recommendation_status.get("RL_RECOMMENDATION_CANDIDATE_STATUS") != "READY_SHADOW_REPLAY_15WEEK":
        raise RuntimeError("EXIT_HARVEST_POLICY_CANDIDATE_V1 requires ready RL recommendation candidate.")
    _require_columns(ledger_df, LEDGER_REQUIRED_COLUMNS, artifact_name="closed trade ledger")
    _require_columns(recommendation_df, RECOMMENDATION_REQUIRED_COLUMNS, artifact_name="RL recommendation trade view")

    if int(ledger_df["candidate_uid"].astype("string").duplicated().sum()) != 0:
        raise RuntimeError("EXIT_HARVEST_POLICY_CANDIDATE_V1 requires unique candidate_uid in closed trade ledger.")
    if int(recommendation_df["candidate_uid"].astype("string").duplicated().sum()) != 0:
        raise RuntimeError("EXIT_HARVEST_POLICY_CANDIDATE_V1 requires unique candidate_uid in recommendation view.")

    rec_cols = ["candidate_uid"] + RECOMMENDATION_REQUIRED_COLUMNS[1:]
    work = ledger_df.copy().merge(
        recommendation_df[rec_cols],
        on="candidate_uid",
        how="left",
        validate="one_to_one",
    )
    if bool(work["rl_priority_recommendation_v1"].isna().any()):
        missing_count = int(work["rl_priority_recommendation_v1"].isna().sum())
        raise RuntimeError(
            f"EXIT_HARVEST_POLICY_CANDIDATE_V1 recommendation coverage gap: {missing_count} ledger trades missing RL recommendation rows"
        )

    work["baseline_realized_pnl_bps_v1"] = _num_series(work, "realized_pnl_bps")
    work["peak_mfe_bps_v1"] = _num_series(work, "hindsight_peak_mfe_bps_v1")
    work["raw_mfe_bps_v1"] = _num_series(work, "mfe_bps")
    work["mae_abs_bps_v1"] = _num_series(work, "mae_bps").abs()
    work["giveback_bps_v1"] = _num_series(work, "hindsight_peak_to_exit_giveback_bps_v1")
    work["hold_longer_extra_value_bps_v1"] = _optional_num_series(
        work,
        "hindsight_hold_longer_extra_value_bps_v1",
    ).fillna(0.0)
    work["exit_earlier_saved_bps_v1"] = _optional_num_series(
        work,
        "hindsight_exit_earlier_saved_bps_v1",
    ).fillna(0.0)
    work["skip_trade_avoided_loss_bps_v1"] = _optional_num_series(
        work,
        "hindsight_skip_trade_avoided_loss_bps_v1",
    ).fillna(0.0)
    work["harvest_capture_ratio_v1"] = _safe_divide(work["baseline_realized_pnl_bps_v1"], work["peak_mfe_bps_v1"])
    work["harvest_capture_ratio_clipped_v1"] = pd.to_numeric(
        work["harvest_capture_ratio_v1"],
        errors="coerce",
    ).clip(lower=0.0, upper=1.0)
    work["harvest_giveback_ratio_v1"] = _safe_divide(work["giveback_bps_v1"], work["peak_mfe_bps_v1"])
    work["harvest_mfe_band_v1"] = work["peak_mfe_bps_v1"].map(lambda value: _mfe_band(float(value)))
    work["home_run_200bps_opportunity_v1"] = work["peak_mfe_bps_v1"].ge(HOME_RUN_MFE_BPS_V1)
    work["runner_100bps_opportunity_v1"] = work["peak_mfe_bps_v1"].ge(100.0)
    work["runner_50bps_opportunity_v1"] = work["peak_mfe_bps_v1"].ge(MIN_RUNNER_MFE_BPS_V1)
    work["captured_at_least_50pct_mfe_v1"] = work["harvest_capture_ratio_clipped_v1"].ge(0.50).fillna(False)

    should_skip = _bool_series(work, "hindsight_should_skip_trade_v1")
    should_hold = _bool_series(work, "hindsight_should_hold_longer_v1")
    should_exit_earlier = _bool_series(work, "hindsight_should_exit_earlier_v1")
    good_clean = _truth_text_bool(work, "good_trade_mfe20_mae5")
    bad_trade = _truth_text_bool(work, "bad_trade")
    good_exit = _truth_text_bool(work, "good_exit")
    premature_exit = _truth_text_bool(work, "premature_exit")
    late_exit = _truth_text_bool(work, "late_exit")
    cata_or_never_mfe = work["trade_outcome_class"].astype("string").isin(["cata", "never_mfe"])
    capture_ratio_numeric = pd.to_numeric(work["harvest_capture_ratio_clipped_v1"], errors="coerce")
    low_capture_with_runner = (
        work["runner_50bps_opportunity_v1"]
        & capture_ratio_numeric.lt(LOW_CAPTURE_RATIO_V1).fillna(False)
    )

    work["harvest_quality_bucket_v1"] = "KEEP_BASELINE_HARVEST"
    work.loc[good_exit & work["harvest_capture_ratio_clipped_v1"].ge(GOOD_CAPTURE_RATIO_V1).fillna(False), "harvest_quality_bucket_v1"] = "GOOD_EXIT_GOOD_CAPTURE"
    work.loc[low_capture_with_runner, "harvest_quality_bucket_v1"] = "UNDERHARVESTED_RUNNER"
    work.loc[premature_exit & should_hold, "harvest_quality_bucket_v1"] = "EXIT_TOO_EARLY_UNDERHARVEST"
    work.loc[late_exit | should_exit_earlier, "harvest_quality_bucket_v1"] = "EXIT_TOO_LATE_DAMAGE"
    work.loc[bad_trade | cata_or_never_mfe | should_skip, "harvest_quality_bucket_v1"] = "ENTRY_OR_RISK_FILTER_FAILURE"

    work["exit_harvest_policy_action_v1"] = "KEEP_BASELINE"
    work.loc[low_capture_with_runner & ~should_skip & ~should_exit_earlier, "exit_harvest_policy_action_v1"] = (
        "PARTIAL_PROFIT_AND_RUNNER_TRAIL_REVIEW"
    )
    work.loc[should_hold & ~should_skip & ~should_exit_earlier, "exit_harvest_policy_action_v1"] = "HOLD_LONGER_RUNNER_TRAIL"
    work.loc[
        should_hold
        & ~should_skip
        & ~should_exit_earlier
        & work["exit_reason"].astype("string").eq("BE_PLUS_FLOOR"),
        "exit_harvest_policy_action_v1",
    ] = "DELAY_BE_PLUS_FLOOR_AND_RUNNER_TRAIL"
    work.loc[
        should_hold & ~should_skip & ~should_exit_earlier & work["home_run_200bps_opportunity_v1"],
        "exit_harvest_policy_action_v1",
    ] = "HOLD_LONGER_HOME_RUN_RUNNER"
    work.loc[should_exit_earlier & ~should_skip, "exit_harvest_policy_action_v1"] = "EXIT_EARLIER_DAMAGE_CONTROL"
    work.loc[should_skip, "exit_harvest_policy_action_v1"] = "ENTRY_SUPPRESS_OR_DOWNSIZE"

    work["harvest_model_update_family_v1"] = "BASELINE_KEEP"
    work.loc[work["exit_harvest_policy_action_v1"].eq("ENTRY_SUPPRESS_OR_DOWNSIZE"), "harvest_model_update_family_v1"] = "ENTRY_FILTER"
    work.loc[work["exit_harvest_policy_action_v1"].eq("EXIT_EARLIER_DAMAGE_CONTROL"), "harvest_model_update_family_v1"] = "DAMAGE_CONTROL_EXIT"
    work.loc[
        work["exit_harvest_policy_action_v1"].isin(
            [
                "HOLD_LONGER_RUNNER_TRAIL",
                "DELAY_BE_PLUS_FLOOR_AND_RUNNER_TRAIL",
                "HOLD_LONGER_HOME_RUN_RUNNER",
                "PARTIAL_PROFIT_AND_RUNNER_TRAIL_REVIEW",
            ]
        ),
        "harvest_model_update_family_v1",
    ] = "RUNNER_HARVEST"

    work["entry_xgb_harvest_label_v1"] = "ALLOW_BASELINE"
    work.loc[should_skip | bad_trade | cata_or_never_mfe, "entry_xgb_harvest_label_v1"] = "REJECT_OR_LOW_SIZE"
    work.loc[
        ~should_skip & ~bad_trade & ~cata_or_never_mfe & (good_clean | work["home_run_200bps_opportunity_v1"]),
        "entry_xgb_harvest_label_v1",
    ] = "PRIORITIZE_CLEAN_RUNNER"
    work["entry_xgb_binary_take_target_v1"] = ~work["entry_xgb_harvest_label_v1"].eq("REJECT_OR_LOW_SIZE")
    work["entry_xgb_sample_weight_proposed_v1"] = 1.0
    work.loc[work["entry_xgb_harvest_label_v1"].eq("REJECT_OR_LOW_SIZE"), "entry_xgb_sample_weight_proposed_v1"] = 3.0
    work.loc[work["entry_xgb_harvest_label_v1"].eq("PRIORITIZE_CLEAN_RUNNER"), "entry_xgb_sample_weight_proposed_v1"] = 2.0

    work["exit_transformer_supervision_label_v1"] = "KEEP_BASELINE"
    work.loc[work["harvest_model_update_family_v1"].eq("RUNNER_HARVEST"), "exit_transformer_supervision_label_v1"] = "HOLD_LONGER_OR_RUNNER_TRAIL"
    work.loc[work["harvest_model_update_family_v1"].eq("DAMAGE_CONTROL_EXIT"), "exit_transformer_supervision_label_v1"] = "EXIT_EARLIER_DAMAGE_CONTROL"
    work.loc[work["harvest_model_update_family_v1"].eq("ENTRY_FILTER"), "exit_transformer_supervision_label_v1"] = "NO_EXIT_TRAINING_ENTRY_FILTER"
    work["exit_transformer_target_extra_value_bps_v1"] = (
        work["hold_longer_extra_value_bps_v1"]
        .where(work["harvest_model_update_family_v1"].eq("RUNNER_HARVEST"), 0.0)
        .astype(float)
    )
    work["exit_transformer_target_saved_bps_v1"] = (
        work["exit_earlier_saved_bps_v1"]
        .where(work["harvest_model_update_family_v1"].eq("DAMAGE_CONTROL_EXIT"), 0.0)
        .astype(float)
    )
    work["exit_transformer_sample_weight_proposed_v1"] = 1.0
    work.loc[work["harvest_model_update_family_v1"].isin(["RUNNER_HARVEST", "DAMAGE_CONTROL_EXIT"]), "exit_transformer_sample_weight_proposed_v1"] = 2.0
    work.loc[
        work["rl_priority_counterfactual_delta_bps_v1"].astype(float).ge(50.0)
        & work["harvest_model_update_family_v1"].isin(["RUNNER_HARVEST", "DAMAGE_CONTROL_EXIT"]),
        "exit_transformer_sample_weight_proposed_v1",
    ] = 3.0

    work["management_rl_harvest_action_label_v1"] = work["exit_harvest_policy_action_v1"]
    work["management_rl_harvest_reward_bps_raw_v1"] = pd.to_numeric(
        work["rl_priority_counterfactual_delta_bps_v1"],
        errors="raise",
    ).astype(float)
    work["management_rl_harvest_reward_bps_clipped_200_v1"] = work[
        "management_rl_harvest_reward_bps_raw_v1"
    ].clip(lower=0.0, upper=REWARD_CLIP_BPS_V1)
    work["management_rl_harvest_reward_contract_v1"] = "TRUTH_DERIVED_SHADOW_TARGET_NOT_LIVE_FILL"
    work["model_adjustment_contract_v1"] = "TARGETS_ONLY_NOT_OBSERVATION_FEATURES_NO_AUTO_RETRAIN"

    quality_columns = [
        "run_id",
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "entry_timestamp",
        "exit_timestamp",
        "baseline_realized_pnl_bps_v1",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "giveback_bps_v1",
        "harvest_capture_ratio_v1",
        "harvest_capture_ratio_clipped_v1",
        "harvest_giveback_ratio_v1",
        "harvest_mfe_band_v1",
        "home_run_200bps_opportunity_v1",
        "runner_100bps_opportunity_v1",
        "runner_50bps_opportunity_v1",
        "captured_at_least_50pct_mfe_v1",
        "harvest_quality_bucket_v1",
        "exit_reason",
        "trade_outcome_class",
        "good_trade",
        "good_trade_mfe20_mae5",
        "bad_trade",
        "good_exit",
        "premature_exit",
        "late_exit",
        "session",
        "vol_regime",
        "trend_regime",
    ]
    policy_columns = quality_columns + [
        "rl_priority_recommendation_v1",
        "rl_priority_counterfactual_delta_bps_v1",
        "rl_priority_entry_skip_delta_bps_v1",
        "rl_priority_exit_earlier_delta_bps_v1",
        "rl_priority_hold_longer_delta_bps_v1",
        "exit_harvest_policy_action_v1",
        "harvest_model_update_family_v1",
        "hold_longer_extra_value_bps_v1",
        "exit_earlier_saved_bps_v1",
        "skip_trade_avoided_loss_bps_v1",
        "management_rl_harvest_reward_bps_raw_v1",
        "management_rl_harvest_reward_bps_clipped_200_v1",
        "management_rl_harvest_reward_contract_v1",
    ]
    model_columns = [
        "run_id",
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "used_for_training",
        "used_for_validation",
        "used_for_holdout",
        "entry_xgb_harvest_label_v1",
        "entry_xgb_binary_take_target_v1",
        "entry_xgb_sample_weight_proposed_v1",
        "exit_transformer_supervision_label_v1",
        "exit_transformer_target_extra_value_bps_v1",
        "exit_transformer_target_saved_bps_v1",
        "exit_transformer_sample_weight_proposed_v1",
        "management_rl_harvest_action_label_v1",
        "management_rl_harvest_reward_bps_raw_v1",
        "management_rl_harvest_reward_bps_clipped_200_v1",
        "harvest_model_update_family_v1",
        "harvest_quality_bucket_v1",
        "model_adjustment_contract_v1",
    ]
    quality_df = work[quality_columns].copy()
    policy_df = work[policy_columns].copy()
    model_target_df = work[model_columns].copy()

    run_ids = _all_run_ids(reports_root, work)
    batch_rows: List[Dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(run_ids), int(batch_weeks)), start=1):
        batch_run_ids = run_ids[start : start + int(batch_weeks)]
        batch = work.loc[work["run_id"].astype("string").isin(batch_run_ids)].copy()
        baseline_total = _sum_numeric(batch, "baseline_realized_pnl_bps_v1")
        peak_total = _sum_numeric(batch, "peak_mfe_bps_v1")
        giveback_total = _sum_numeric(batch, "giveback_bps_v1")
        batch_rows.append(
            {
                "batch_index_v1": int(batch_index),
                "batch_weeks_v1": int(batch_weeks),
                "run_count_v1": int(len(batch_run_ids)),
                "first_run_id_v1": batch_run_ids[0] if batch_run_ids else None,
                "last_run_id_v1": batch_run_ids[-1] if batch_run_ids else None,
                "trade_count_v1": int(len(batch)),
                "baseline_total_pnl_bps_v1": baseline_total,
                "peak_mfe_total_bps_v1": peak_total,
                "giveback_total_bps_v1": giveback_total,
                "portfolio_capture_ratio_v1": float(baseline_total / peak_total) if peak_total > 0 else None,
                "mean_capture_ratio_v1": _mean_numeric(batch, "harvest_capture_ratio_clipped_v1"),
                "harvest_priority_delta_bps_v1": _sum_numeric(batch, "management_rl_harvest_reward_bps_raw_v1"),
                "hold_longer_count_v1": int(batch["rl_priority_recommendation_v1"].astype("string").eq("HOLD_LONGER").sum()),
                "exit_earlier_count_v1": int(batch["rl_priority_recommendation_v1"].astype("string").eq("EXIT_EARLIER").sum()),
                "skip_trade_count_v1": int(batch["rl_priority_recommendation_v1"].astype("string").eq("SKIP_TRADE").sum()),
                "home_run_200bps_opportunity_count_v1": int(batch["home_run_200bps_opportunity_v1"].sum()),
                "home_run_200bps_captured_50pct_count_v1": int(
                    (batch["home_run_200bps_opportunity_v1"] & batch["captured_at_least_50pct_mfe_v1"]).sum()
                ),
                "be_plus_floor_runner_review_count_v1": int(
                    batch["exit_harvest_policy_action_v1"].astype("string").eq("DELAY_BE_PLUS_FLOOR_AND_RUNNER_TRAIL").sum()
                ),
                "zero_trade_run_count_v1": int(
                    len([run_id for run_id in batch_run_ids if run_id not in set(batch["run_id"].astype("string"))])
                ),
            }
        )
    batch_df = pd.DataFrame.from_records(batch_rows)

    total_baseline = _sum_numeric(work, "baseline_realized_pnl_bps_v1")
    total_peak = _sum_numeric(work, "peak_mfe_bps_v1")
    total_giveback = _sum_numeric(work, "giveback_bps_v1")
    portfolio_capture = float(total_baseline / total_peak) if total_peak > 0 else None

    consistency_rows = [
        {
            "check_name_v1": "HARVEST_VIEW_COVERS_CLOSED_TRADE_LEDGER_EXACTLY",
            "status_v1": "PASS" if len(policy_df) == len(ledger_df) else "FAIL",
            "observed_value_v1": int(len(policy_df)),
            "expected_value_v1": int(len(ledger_df)),
            "note_v1": "One harvest policy row per closed truth trade.",
        },
        {
            "check_name_v1": "RL_RECOMMENDATION_COVERAGE_EXACT",
            "status_v1": "PASS"
            if int(work["rl_priority_recommendation_v1"].isna().sum()) == 0
            and int(recommendation_summary.get("baseline_trade_count_v1", -1)) == int(len(ledger_df))
            else "FAIL",
            "observed_value_v1": json.dumps(
                {
                    "missing_recommendations": int(work["rl_priority_recommendation_v1"].isna().sum()),
                    "recommendation_baseline_trade_count": int(recommendation_summary.get("baseline_trade_count_v1", -1)),
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            "expected_value_v1": json.dumps(
                {"missing_recommendations": 0, "recommendation_baseline_trade_count": int(len(ledger_df))},
                ensure_ascii=True,
                sort_keys=True,
            ),
            "note_v1": "Harvest policy candidate must be downstream of full RL recommendation coverage.",
        },
        {
            "check_name_v1": "MODEL_TARGETS_ARE_COMPLETE",
            "status_v1": "PASS"
            if int(model_target_df[model_columns].isna().sum().sum()) == 0
            else "FAIL",
            "observed_value_v1": int(model_target_df[model_columns].isna().sum().sum()),
            "expected_value_v1": 0,
            "note_v1": "All model adjustment targets must be populated; no synthetic NA shortcuts.",
        },
        {
            "check_name_v1": "BATCH_REPLAY_COVERS_ALL_RUNS_AND_TRADES",
            "status_v1": "PASS"
            if int(batch_df["run_count_v1"].sum()) == int(len(run_ids))
            and int(batch_df["trade_count_v1"].sum()) == int(len(policy_df))
            else "FAIL",
            "observed_value_v1": json.dumps(
                {"run_count": int(batch_df["run_count_v1"].sum()), "trade_count": int(batch_df["trade_count_v1"].sum())},
                ensure_ascii=True,
                sort_keys=True,
            ),
            "expected_value_v1": json.dumps(
                {"run_count": int(len(run_ids)), "trade_count": int(len(policy_df))},
                ensure_ascii=True,
                sort_keys=True,
            ),
            "note_v1": "15-week harvest replay must cover the whole truth universe.",
        },
        {
            "check_name_v1": "HARVEST_REWARD_DELTA_MATCHES_RL_RECOMMENDATION_PRIORITY_DELTA",
            "status_v1": "PASS"
            if abs(
                _sum_numeric(work, "management_rl_harvest_reward_bps_raw_v1")
                - float(recommendation_summary.get("priority_counterfactual_delta_bps_v1", 0.0))
            )
            < 1e-6
            else "FAIL",
            "observed_value_v1": _sum_numeric(work, "management_rl_harvest_reward_bps_raw_v1"),
            "expected_value_v1": float(recommendation_summary.get("priority_counterfactual_delta_bps_v1", 0.0)),
            "note_v1": "Harvest target reward must remain the same non-overlapping RL priority delta.",
        },
        {
            "check_name_v1": "CONTRACT_STAYS_SHADOW_TARGET_NOT_LIVE_FILL",
            "status_v1": "PASS"
            if work["management_rl_harvest_reward_contract_v1"].astype("string").eq(
                "TRUTH_DERIVED_SHADOW_TARGET_NOT_LIVE_FILL"
            ).all()
            else "FAIL",
            "observed_value_v1": _counts(work, "management_rl_harvest_reward_contract_v1"),
            "expected_value_v1": {"TRUTH_DERIVED_SHADOW_TARGET_NOT_LIVE_FILL": int(len(work))},
            "note_v1": "This layer creates training/review targets only and never claims live counterfactual fills.",
        },
    ]
    audit_df = pd.DataFrame.from_records(consistency_rows)
    failed_checks = int(audit_df["status_v1"].astype("string").eq("FAIL").sum())

    summary = {
        "layer_name": "EXIT_HARVEST_POLICY_CANDIDATE_SUMMARY_V1",
        "review_dir_v1": str(review_dir),
        "recommendation_dir_v1": str(recommendation_dir),
        "batch_weeks_v1": int(batch_weeks),
        "batch_count_v1": int(len(batch_df)),
        "run_count_v1": int(len(run_ids)),
        "trade_count_v1": int(len(work)),
        "baseline_total_pnl_bps_v1": total_baseline,
        "baseline_mean_pnl_bps_v1": _mean_numeric(work, "baseline_realized_pnl_bps_v1"),
        "peak_mfe_total_bps_v1": total_peak,
        "peak_mfe_mean_bps_v1": _mean_numeric(work, "peak_mfe_bps_v1"),
        "giveback_total_bps_v1": total_giveback,
        "giveback_mean_bps_v1": _mean_numeric(work, "giveback_bps_v1"),
        "portfolio_capture_ratio_v1": portfolio_capture,
        "capture_ratio_quantiles_v1": _quantiles(work["harvest_capture_ratio_clipped_v1"]),
        "peak_mfe_quantiles_bps_v1": _quantiles(work["peak_mfe_bps_v1"]),
        "giveback_quantiles_bps_v1": _quantiles(work["giveback_bps_v1"]),
        "harvest_quality_bucket_counts_v1": _counts(work, "harvest_quality_bucket_v1"),
        "exit_harvest_policy_action_counts_v1": _counts(work, "exit_harvest_policy_action_v1"),
        "model_update_family_counts_v1": _counts(work, "harvest_model_update_family_v1"),
        "entry_xgb_harvest_label_counts_v1": _counts(work, "entry_xgb_harvest_label_v1"),
        "exit_transformer_supervision_label_counts_v1": _counts(work, "exit_transformer_supervision_label_v1"),
        "harvest_priority_delta_bps_v1": _sum_numeric(work, "management_rl_harvest_reward_bps_raw_v1"),
        "harvest_priority_delta_clipped_200_bps_v1": _sum_numeric(
            work,
            "management_rl_harvest_reward_bps_clipped_200_v1",
        ),
        "home_run_200bps_opportunity_count_v1": int(work["home_run_200bps_opportunity_v1"].sum()),
        "home_run_200bps_captured_50pct_count_v1": int(
            (work["home_run_200bps_opportunity_v1"] & work["captured_at_least_50pct_mfe_v1"]).sum()
        ),
        "runner_100bps_opportunity_count_v1": int(work["runner_100bps_opportunity_v1"].sum()),
        "runner_50bps_opportunity_count_v1": int(work["runner_50bps_opportunity_v1"].sum()),
        "be_plus_floor_runner_review_count_v1": int(
            work["exit_harvest_policy_action_v1"].astype("string").eq("DELAY_BE_PLUS_FLOOR_AND_RUNNER_TRAIL").sum()
        ),
        "failed_check_count_v1": failed_checks,
    }
    status = {
        "layer_name": "EXIT_HARVEST_POLICY_CANDIDATE_STATUS_V1",
        "EXIT_HARVEST_POLICY_CANDIDATE_STATUS": "READY_FOR_RETRAIN_TARGET_REVIEW"
        if failed_checks == 0
        else "ISSUES_FOUND",
        "HARVEST_MODE_STATUS": "SHADOW_TARGETS_NOT_LIVE_FILL",
        "MODEL_UPDATE_TARGET_STATUS": "ENTRY_XGB_ENTRY_TRANSFORMER_EXIT_TRANSFORMER_AND_RL_TARGETS_READY"
        if failed_checks == 0
        else "TARGETS_BLOCKED",
        "RETRAIN_AUTOMATION_STATUS": "NOT_AUTO_RETRAIN",
        "not_trainer": True,
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    contract = {
        "layer_name": "EXIT_HARVEST_POLICY_CANDIDATE_CONTRACT_V1",
        "mode_v1": "HARVEST_QUALITY_AUDIT_AND_MODEL_TARGETS",
        "truth_sources_v1": [
            "closed_trade_ledger.realized_pnl_bps",
            "closed_trade_ledger.hindsight_peak_mfe_bps_v1",
            "closed_trade_ledger.hindsight_peak_to_exit_giveback_bps_v1",
            "closed_trade_ledger.hindsight_should_skip_trade_v1",
            "closed_trade_ledger.hindsight_should_hold_longer_v1",
            "closed_trade_ledger.hindsight_should_exit_earlier_v1",
            "rl_recommendation_candidate.rl_priority_recommendation_v1",
            "rl_recommendation_candidate.rl_priority_counterfactual_delta_bps_v1",
        ],
        "generic_thresholds_v1": {
            "runner_mfe_min_bps": MIN_RUNNER_MFE_BPS_V1,
            "home_run_mfe_min_bps": HOME_RUN_MFE_BPS_V1,
            "low_capture_ratio": LOW_CAPTURE_RATIO_V1,
            "good_capture_ratio": GOOD_CAPTURE_RATIO_V1,
            "reward_clip_bps": REWARD_CLIP_BPS_V1,
        },
        "model_target_outputs_v1": {
            "entry_xgb_and_entry_transformer": [
                "entry_xgb_harvest_label_v1",
                "entry_xgb_binary_take_target_v1",
                "entry_xgb_sample_weight_proposed_v1",
            ],
            "exit_transformer": [
                "exit_transformer_supervision_label_v1",
                "exit_transformer_target_extra_value_bps_v1",
                "exit_transformer_target_saved_bps_v1",
                "exit_transformer_sample_weight_proposed_v1",
            ],
            "management_rl": [
                "management_rl_harvest_action_label_v1",
                "management_rl_harvest_reward_bps_raw_v1",
                "management_rl_harvest_reward_bps_clipped_200_v1",
            ],
        },
        "prohibitions_v1": [
            "Do not use target columns as live observation features.",
            "Do not treat harvest delta as realized counterfactual fill.",
            "Do not auto-retrain models without separate walk-forward and leakage gate.",
            "Do not replace risk engine with RL outputs.",
        ],
    }
    manifest = {
        "layer_name": "EXIT_HARVEST_POLICY_CANDIDATE_MANIFEST_V1",
        "mode_v1": "APPEND_ONLY_EXTENSION",
        "review_dir_v1": str(review_dir),
        "recommendation_dir_v1": str(recommendation_dir),
        "artifacts_v1": {
            "contract_v1": HARVEST_CONTRACT,
            "harvest_quality_view_v1": HARVEST_QUALITY_VIEW,
            "harvest_policy_view_v1": HARVEST_POLICY_VIEW,
            "model_adjustment_target_view_v1": HARVEST_MODEL_TARGET_VIEW,
            "batch_replay_v1": HARVEST_BATCH_REPLAY,
            "summary_v1": HARVEST_SUMMARY,
            "status_v1": HARVEST_STATUS,
            "audit_v1": HARVEST_AUDIT,
            "markdown_v1": HARVEST_MD,
        },
    }
    return {
        "contract_v1": contract,
        "harvest_quality_view_v1_df": quality_df,
        "harvest_policy_view_v1_df": policy_df,
        "model_adjustment_target_view_v1_df": model_target_df,
        "batch_replay_v1_df": batch_df,
        "summary_v1": summary,
        "status_v1": status,
        "audit_v1_df": audit_df,
        "manifest_v1": manifest,
        "markdown_v1": _render_markdown({**summary, "status_v1": status}, batch_df),
    }


def materialize_truth_exit_harvest_policy_candidate(
    reports_root: Path,
    *,
    review_dir: Path | None = None,
    recommendation_dir: Path | None = None,
    extension_dir: Path | None = None,
    batch_weeks: int = 15,
) -> Dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    resolved_review_dir = _resolve_review_dir(reports_root, str(review_dir) if review_dir else None)
    resolved_recommendation_dir = _resolve_recommendation_dir(
        reports_root,
        str(recommendation_dir) if recommendation_dir else None,
    )
    payload = build_exit_harvest_policy_candidate_payload(
        reports_root=reports_root,
        review_dir=resolved_review_dir,
        recommendation_dir=resolved_recommendation_dir,
        batch_weeks=batch_weeks,
    )
    if int(payload["summary_v1"].get("failed_check_count_v1", -1)) != 0:
        raise RuntimeError("EXIT_HARVEST_POLICY_CANDIDATE_V1 consistency checks failed; refusing to materialize.")

    if extension_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        extension_dir = reports_root / f"{LEDGER_NAMESPACE_PREFIX}{stamp}_{HARVEST_EXTENSION_SUFFIX}"
    extension_dir = Path(extension_dir).expanduser().resolve()
    extension_dir.mkdir(parents=True, exist_ok=False)

    payload["harvest_quality_view_v1_df"].to_parquet(extension_dir / HARVEST_QUALITY_VIEW, index=False)
    payload["harvest_policy_view_v1_df"].to_parquet(extension_dir / HARVEST_POLICY_VIEW, index=False)
    payload["model_adjustment_target_view_v1_df"].to_parquet(extension_dir / HARVEST_MODEL_TARGET_VIEW, index=False)
    payload["batch_replay_v1_df"].to_csv(extension_dir / HARVEST_BATCH_REPLAY, index=False)
    payload["audit_v1_df"].to_csv(extension_dir / HARVEST_AUDIT, index=False)
    (extension_dir / HARVEST_CONTRACT).write_text(
        json.dumps(payload["contract_v1"], ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (extension_dir / HARVEST_SUMMARY).write_text(
        json.dumps(payload["summary_v1"], ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (extension_dir / HARVEST_STATUS).write_text(
        json.dumps(payload["status_v1"], ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (extension_dir / HARVEST_MANIFEST).write_text(
        json.dumps(payload["manifest_v1"], ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (extension_dir / HARVEST_MD).write_text(payload["markdown_v1"], encoding="utf-8")

    top_level_summary = dict(payload["summary_v1"])
    top_level_summary["extension_dir_v1"] = str(extension_dir)
    top_level_summary["review_dir_v1"] = str(resolved_review_dir)
    top_level_summary["recommendation_dir_v1"] = str(resolved_recommendation_dir)
    top_level_summary["status_v1"] = payload["status_v1"]
    (reports_root / TOP_LEVEL_SUMMARY).write_text(
        json.dumps(top_level_summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "extension_dir": extension_dir,
        "top_level_summary_path": reports_root / TOP_LEVEL_SUMMARY,
        "summary": payload["summary_v1"],
        "status": payload["status_v1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize exit harvest quality and model-adjustment targets.")
    parser.add_argument("--reports-root", type=str, default=None)
    parser.add_argument("--review-dir", type=str, default=None)
    parser.add_argument("--recommendation-dir", type=str, default=None)
    parser.add_argument("--extension-dir", type=str, default=None)
    parser.add_argument("--batch-weeks", type=int, default=15)
    args = parser.parse_args()

    result = materialize_truth_exit_harvest_policy_candidate(
        _resolve_reports_root(args.reports_root),
        review_dir=Path(args.review_dir).expanduser().resolve() if args.review_dir else None,
        recommendation_dir=Path(args.recommendation_dir).expanduser().resolve() if args.recommendation_dir else None,
        extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None,
        batch_weeks=max(1, int(args.batch_weeks)),
    )
    print(
        json.dumps(
            {
                "extension_dir": str(result["extension_dir"]),
                "top_level_summary_path": str(result["top_level_summary_path"]),
                "status": result["status"],
                "summary": result["summary"],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
