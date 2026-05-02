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
RECOMMENDATION_EXTENSION_SUFFIX = "RL_RECOMMENDATION_CANDIDATE_V1"

RECOMMENDATION_CONTRACT = "shadow_meta_all_trade_review_rl_recommendation_candidate_contract_v1.json"
RECOMMENDATION_TRADE_VIEW = "shadow_meta_all_trade_review_rl_recommendation_candidate_trade_view_v1.parquet"
RECOMMENDATION_BATCH_REPLAY = "shadow_meta_all_trade_review_rl_recommendation_shadow_replay_15week_v1.csv"
RECOMMENDATION_SUMMARY = "shadow_meta_all_trade_review_rl_recommendation_candidate_summary_v1.json"
RECOMMENDATION_STATUS = "shadow_meta_all_trade_review_rl_recommendation_candidate_status_v1.json"
RECOMMENDATION_AUDIT = "shadow_meta_all_trade_review_rl_recommendation_candidate_consistency_audit_v1.csv"
RECOMMENDATION_MANIFEST = "shadow_meta_all_trade_review_rl_recommendation_candidate_manifest_v1.json"
RECOMMENDATION_MD = "shadow_meta_all_trade_review_rl_recommendation_shadow_replay_15week_v1.md"
TOP_LEVEL_SUMMARY = "truth_rl_recommendation_candidate_v1.json"

RUN_RE = re.compile(r"^E2E_SANITY_ORDERFIX_(\d{8})_(\d{8})$")


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

    rebuild_summary_path = reports_root / "truth_downstream_canonical_rebuild_v1.json"
    if rebuild_summary_path.exists():
        ledger_dir = _load_json(rebuild_summary_path).get("ledger_dir")
        if isinstance(ledger_dir, str) and ledger_dir.strip():
            candidate = Path(ledger_dir).expanduser().resolve()
            if (candidate / "shadow_meta_all_trade_review_ledger_closed_trades.parquet").exists():
                return candidate
    raise FileNotFoundError("Could not resolve canonical review dir from truth_downstream_canonical_rebuild_v1.json.")


def _resolve_unified_dir(reports_root: Path, unified_dir_arg: str | None) -> Path:
    if unified_dir_arg:
        unified_dir = Path(unified_dir_arg).expanduser().resolve()
        if not unified_dir.exists():
            raise FileNotFoundError(f"Unified RL dir does not exist: {unified_dir}")
        return unified_dir

    top_summary_path = reports_root / "truth_rl_unified_observability_v1.json"
    if top_summary_path.exists():
        extension_dir = _load_json(top_summary_path).get("extension_dir_v1")
        if isinstance(extension_dir, str) and extension_dir.strip():
            candidate = Path(extension_dir).expanduser().resolve()
            if (candidate / "shadow_meta_all_trade_review_rl_unified_episode_view_v1.parquet").exists():
                return candidate
    namespace_dirs = sorted(
        [
            path
            for path in reports_root.iterdir()
            if path.is_dir()
            and path.name.startswith(LEDGER_NAMESPACE_PREFIX)
            and path.name.endswith("RL_UNIFIED_OBSERVABILITY_V1")
        ],
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate in namespace_dirs:
        if (candidate / "shadow_meta_all_trade_review_rl_unified_episode_view_v1.parquet").exists():
            return candidate
    raise FileNotFoundError("Could not resolve unified RL observability dir.")


def _counts(frame: pd.DataFrame, column: str) -> Dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].astype("string").value_counts(dropna=False).to_dict().items()
    }


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


def _num_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f"Missing required numeric column: {column}")
    return pd.to_numeric(frame[column], errors="coerce")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, artifact_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{artifact_name} is missing required columns: {missing}")


def _run_sort_key(run_id: str) -> str:
    match = RUN_RE.match(str(run_id))
    return match.group(1) if match else str(run_id)


def _all_run_ids(reports_root: Path, ledger_df: pd.DataFrame) -> List[str]:
    runs_root = reports_root / "runs"
    run_ids: List[str] = []
    if runs_root.exists():
        run_ids = sorted(
            [path.name for path in runs_root.iterdir() if path.is_dir() and RUN_RE.match(path.name)],
            key=_run_sort_key,
        )
    if not run_ids:
        run_ids = sorted(ledger_df["run_id"].astype("string").dropna().unique().tolist(), key=_run_sort_key)
    return run_ids


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


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _run_id_list(values: Iterable[Any]) -> List[str]:
    run_ids: List[str] = []
    for value in values:
        if isinstance(value, dict):
            run_id = value.get("run_id")
            if run_id:
                run_ids.append(str(run_id))
        elif value:
            run_ids.append(str(value))
    return run_ids


def _json_run_id_set(payload: Dict[str, Any], keys: Sequence[str]) -> set[str]:
    for key in keys:
        values = payload.get(key)
        if values:
            return set(_run_id_list(values))
    return set()


def _render_markdown(summary: Dict[str, Any], batch_df: pd.DataFrame) -> str:
    lines = [
        "# RL Recommendation Candidate V1",
        "",
        "Dette er en shadow-replay/recommendation-kandidat, ikke en live controller.",
        "",
        "## Headline",
        "",
        f"- Status: `{summary['status_v1']['RL_RECOMMENDATION_CANDIDATE_STATUS']}`",
        f"- 15-week batches: `{summary['batch_count_v1']}`",
        f"- Baseline trades: `{summary['baseline_trade_count_v1']}`",
        f"- Baseline PnL bps: `{summary['baseline_total_pnl_bps_v1']:.2f}`",
        f"- Recommendation priority delta bps: `{summary['priority_counterfactual_delta_bps_v1']:.2f}`",
        f"- Shadow upper-bound PnL bps: `{summary['shadow_upper_bound_pnl_bps_v1']:.2f}`",
        f"- Unified episode coverage: `{summary['unified_episode_covered_trade_count_v1']}/{summary['baseline_trade_count_v1']}` "
        f"(`{summary['unified_episode_coverage_status_v1']}`)",
        f"- Entry-direct feature coverage: `{summary['unified_entry_direct_episode_count_v1']}/{summary['baseline_trade_count_v1']}` "
        f"(`{summary['entry_direct_feature_coverage_status_v1']}`)",
        f"- Retrain readiness: `{summary['status_v1']['RETRAIN_READY_STATUS']}`",
        "",
        "## Batch Replay",
        "",
        "| batch | runs | trades | baseline pnl | priority delta | shadow upper | skip | exit earlier | hold longer | zero weeks | opp zero |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in batch_df.to_dict(orient="records"):
        lines.append(
            "| {batch_index_v1} | {run_count_v1} | {baseline_trade_count_v1} | {baseline_total_pnl_bps_v1:.2f} | "
            "{priority_counterfactual_delta_bps_v1:.2f} | {shadow_upper_bound_pnl_bps_v1:.2f} | "
            "{entry_skip_recommendation_count_v1} | {exit_earlier_recommendation_count_v1} | "
            "{hold_longer_recommendation_count_v1} | {zero_trade_run_count_v1} | {opportunity_rich_zero_trade_run_count_v1} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Entry propensity er fortsatt ikke etablert, så entry-anbefalingene er ikke off-policy-klare.",
            "- PnL-deltaene er hindsight/shadow upper-bound der vi har ekte truth-labels, ikke live-realiserte counterfactual fills.",
            "- Prioritet er `SKIP_TRADE` før `EXIT_EARLIER` før `HOLD_LONGER`, slik at samme trade ikke dobbelttelles.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_rl_recommendation_candidate_payload(
    *,
    reports_root: Path,
    review_dir: Path,
    unified_dir: Path,
    batch_weeks: int = 15,
) -> Dict[str, Any]:
    ledger_df = pd.read_parquet(review_dir / "shadow_meta_all_trade_review_ledger_closed_trades.parquet")
    episode_df = pd.read_parquet(unified_dir / "shadow_meta_all_trade_review_rl_unified_episode_view_v1.parquet")
    unified_summary = _load_json(unified_dir / "shadow_meta_all_trade_review_rl_unified_observability_summary_v1.json")
    unified_status = _load_json(unified_dir / "shadow_meta_all_trade_review_rl_unified_observability_status_v1.json")
    skipability_summary = _load_json(reports_root / "truth_entry_skipability_pressure_v1.json")
    market_opportunity_summary = _load_json(reports_root / "truth_continuous_market_opportunity_v1.json")

    if ledger_df.empty:
        raise RuntimeError("RL_RECOMMENDATION_CANDIDATE_V1 requires non-empty closed-trade ledger.")
    _require_columns(
        ledger_df,
        [
            "run_id",
            "candidate_uid",
            "trade_uid",
            "trade_id",
            "realized_pnl_bps",
            "hindsight_should_skip_trade_v1",
            "hindsight_should_hold_longer_v1",
            "hindsight_should_exit_earlier_v1",
            "hindsight_skip_trade_avoided_loss_bps_v1",
            "hindsight_hold_longer_extra_value_bps_v1",
            "hindsight_exit_earlier_saved_bps_v1",
        ],
        artifact_name="closed trade ledger",
    )
    _require_columns(episode_df, ["candidate_uid"], artifact_name="unified episode view")
    if int(unified_summary.get("failed_check_count_v1", -1)) != 0:
        raise RuntimeError("RL_RECOMMENDATION_CANDIDATE_V1 requires unified observability failed_check_count_v1 == 0.")
    if unified_status.get("UNIFIED_RL_OBSERVABILITY_STATUS") != "READY_ENTRY_AND_MANAGEMENT_OBSERVABILITY":
        raise RuntimeError("RL_RECOMMENDATION_CANDIDATE_V1 requires ready unified RL observability.")
    if int(batch_weeks) <= 0:
        raise ValueError("batch_weeks must be positive")

    work = ledger_df.copy()
    work["run_id"] = work["run_id"].astype("string")
    work["baseline_realized_pnl_bps_v1"] = _num_series(work, "realized_pnl_bps").fillna(0.0)
    work["entry_should_skip_v1"] = _bool_series(work, "hindsight_should_skip_trade_v1")
    work["management_should_hold_longer_v1"] = _bool_series(work, "hindsight_should_hold_longer_v1")
    work["management_should_exit_earlier_v1"] = _bool_series(work, "hindsight_should_exit_earlier_v1")
    work["entry_skip_avoided_loss_bps_v1"] = _num_series(work, "hindsight_skip_trade_avoided_loss_bps_v1").fillna(0.0)
    work["hold_longer_extra_value_bps_v1"] = _num_series(work, "hindsight_hold_longer_extra_value_bps_v1").fillna(0.0)
    work["exit_earlier_saved_bps_v1"] = _num_series(work, "hindsight_exit_earlier_saved_bps_v1").fillna(0.0)

    work["rl_entry_recommendation_v1"] = "KEEP_ENTRY_BASELINE"
    work.loc[work["entry_should_skip_v1"], "rl_entry_recommendation_v1"] = "SKIP_TRADE"
    work["rl_management_recommendation_v1"] = "KEEP_MANAGEMENT_BASELINE"
    work.loc[
        ~work["entry_should_skip_v1"] & work["management_should_exit_earlier_v1"],
        "rl_management_recommendation_v1",
    ] = "EXIT_EARLIER"
    work.loc[
        ~work["entry_should_skip_v1"]
        & ~work["management_should_exit_earlier_v1"]
        & work["management_should_hold_longer_v1"],
        "rl_management_recommendation_v1",
    ] = "HOLD_LONGER"
    work["rl_priority_recommendation_v1"] = "KEEP_BASELINE"
    work.loc[work["entry_should_skip_v1"], "rl_priority_recommendation_v1"] = "SKIP_TRADE"
    work.loc[
        ~work["entry_should_skip_v1"] & work["management_should_exit_earlier_v1"],
        "rl_priority_recommendation_v1",
    ] = "EXIT_EARLIER"
    work.loc[
        ~work["entry_should_skip_v1"]
        & ~work["management_should_exit_earlier_v1"]
        & work["management_should_hold_longer_v1"],
        "rl_priority_recommendation_v1",
    ] = "HOLD_LONGER"
    work["rl_priority_counterfactual_delta_bps_v1"] = 0.0
    work.loc[work["rl_priority_recommendation_v1"].eq("SKIP_TRADE"), "rl_priority_counterfactual_delta_bps_v1"] = work[
        "entry_skip_avoided_loss_bps_v1"
    ]
    work.loc[work["rl_priority_recommendation_v1"].eq("EXIT_EARLIER"), "rl_priority_counterfactual_delta_bps_v1"] = work[
        "exit_earlier_saved_bps_v1"
    ]
    work.loc[work["rl_priority_recommendation_v1"].eq("HOLD_LONGER"), "rl_priority_counterfactual_delta_bps_v1"] = work[
        "hold_longer_extra_value_bps_v1"
    ]
    work["rl_shadow_upper_bound_pnl_bps_v1"] = (
        work["baseline_realized_pnl_bps_v1"] + work["rl_priority_counterfactual_delta_bps_v1"]
    )
    work["rl_priority_entry_skip_delta_bps_v1"] = 0.0
    work.loc[work["rl_priority_recommendation_v1"].eq("SKIP_TRADE"), "rl_priority_entry_skip_delta_bps_v1"] = work[
        "rl_priority_counterfactual_delta_bps_v1"
    ]
    work["rl_priority_exit_earlier_delta_bps_v1"] = 0.0
    work.loc[work["rl_priority_recommendation_v1"].eq("EXIT_EARLIER"), "rl_priority_exit_earlier_delta_bps_v1"] = work[
        "rl_priority_counterfactual_delta_bps_v1"
    ]
    work["rl_priority_hold_longer_delta_bps_v1"] = 0.0
    work.loc[work["rl_priority_recommendation_v1"].eq("HOLD_LONGER"), "rl_priority_hold_longer_delta_bps_v1"] = work[
        "rl_priority_counterfactual_delta_bps_v1"
    ]
    work["rl_recommendation_semantics_v1"] = "HINDSIGHT_SHADOW_UPPER_BOUND_NOT_LIVE_COUNTERFACTUAL_FILL"

    episode_candidates = set(episode_df["candidate_uid"].astype("string").dropna().tolist())
    work["unified_episode_coverage_status_v1"] = work["candidate_uid"].astype("string").isin(episode_candidates).map(
        {True: "COVERED_BY_UNIFIED_ENTRY_EPISODE", False: "NOT_COVERED_BY_UNIFIED_ENTRY_EPISODE"}
    )

    run_ids = _all_run_ids(reports_root, work)
    opportunity_rich_zero_runs = _json_run_id_set(
        market_opportunity_summary,
        ["opportunity_rich_zero_trade_runs_anchor", "opportunity_rich_zero_trade_run_ids"],
    )
    candidate_rich_zero_runs = _json_run_id_set(
        skipability_summary,
        ["candidate_rich_zero_trade_run_ids", "candidate_rich_zero_trade_run_sample_v1"],
    )

    batch_rows: List[Dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(run_ids), int(batch_weeks)), start=1):
        batch_run_ids = run_ids[start : start + int(batch_weeks)]
        batch = work.loc[work["run_id"].isin(batch_run_ids)].copy()
        zero_trade_runs = [run_id for run_id in batch_run_ids if run_id not in set(batch["run_id"].astype("string"))]
        opp_zero_runs = [run_id for run_id in batch_run_ids if run_id in opportunity_rich_zero_runs]
        candidate_rich_zero = [run_id for run_id in zero_trade_runs if run_id in candidate_rich_zero_runs]
        baseline_total = _sum_numeric(batch, "baseline_realized_pnl_bps_v1")
        delta_total = _sum_numeric(batch, "rl_priority_counterfactual_delta_bps_v1")
        batch_rows.append(
            {
                "batch_index_v1": int(batch_index),
                "batch_weeks_v1": int(batch_weeks),
                "run_count_v1": int(len(batch_run_ids)),
                "first_run_id_v1": batch_run_ids[0] if batch_run_ids else None,
                "last_run_id_v1": batch_run_ids[-1] if batch_run_ids else None,
                "baseline_trade_count_v1": int(len(batch)),
                "baseline_total_pnl_bps_v1": baseline_total,
                "baseline_mean_pnl_bps_v1": _mean_numeric(batch, "baseline_realized_pnl_bps_v1"),
                "entry_skip_recommendation_count_v1": int(batch["rl_priority_recommendation_v1"].eq("SKIP_TRADE").sum()),
                "exit_earlier_recommendation_count_v1": int(batch["rl_priority_recommendation_v1"].eq("EXIT_EARLIER").sum()),
                "hold_longer_recommendation_count_v1": int(batch["rl_priority_recommendation_v1"].eq("HOLD_LONGER").sum()),
                "keep_baseline_recommendation_count_v1": int(batch["rl_priority_recommendation_v1"].eq("KEEP_BASELINE").sum()),
                "priority_counterfactual_delta_bps_v1": delta_total,
                "priority_entry_skip_delta_bps_v1": _sum_numeric(batch, "rl_priority_entry_skip_delta_bps_v1"),
                "priority_exit_earlier_delta_bps_v1": _sum_numeric(batch, "rl_priority_exit_earlier_delta_bps_v1"),
                "priority_hold_longer_delta_bps_v1": _sum_numeric(batch, "rl_priority_hold_longer_delta_bps_v1"),
                "shadow_upper_bound_pnl_bps_v1": baseline_total + delta_total,
                "entry_skip_avoided_loss_bps_v1": _sum_numeric(batch, "entry_skip_avoided_loss_bps_v1"),
                "exit_earlier_saved_bps_v1": _sum_numeric(batch, "exit_earlier_saved_bps_v1"),
                "hold_longer_extra_value_bps_v1": _sum_numeric(batch, "hold_longer_extra_value_bps_v1"),
                "zero_trade_run_count_v1": int(len(zero_trade_runs)),
                "candidate_rich_zero_trade_run_count_v1": int(len(candidate_rich_zero)),
                "opportunity_rich_zero_trade_run_count_v1": int(len(opp_zero_runs)),
                "zero_trade_run_ids_v1": json.dumps(zero_trade_runs, ensure_ascii=True),
                "opportunity_rich_zero_trade_run_ids_v1": json.dumps(opp_zero_runs, ensure_ascii=True),
                "unified_episode_covered_trade_count_v1": int(
                    batch["unified_episode_coverage_status_v1"].eq("COVERED_BY_UNIFIED_ENTRY_EPISODE").sum()
                ),
                "unified_episode_coverage_rate_v1": _safe_rate(
                    int(batch["unified_episode_coverage_status_v1"].eq("COVERED_BY_UNIFIED_ENTRY_EPISODE").sum()),
                    int(len(batch)),
                ),
            }
        )
    batch_df = pd.DataFrame.from_records(batch_rows)

    total_baseline = _sum_numeric(work, "baseline_realized_pnl_bps_v1")
    total_delta = _sum_numeric(work, "rl_priority_counterfactual_delta_bps_v1")
    unified_covered_count = int(
        work["unified_episode_coverage_status_v1"].eq("COVERED_BY_UNIFIED_ENTRY_EPISODE").sum()
    )
    unified_uncovered_count = int(len(work) - unified_covered_count)
    unified_coverage_rate = _safe_rate(unified_covered_count, int(len(work)))
    unified_episode_coverage_status = (
        "FULL_UNIFIED_EPISODE_COVERAGE"
        if unified_uncovered_count == 0
        else "PARTIAL_UNIFIED_EPISODE_COVERAGE_REPLAY_BLOCKER"
    )
    unified_entry_direct_episode_count = int(unified_summary.get("entry_direct_episode_rows_v1") or 0)
    unified_management_only_episode_count = int(unified_summary.get("management_only_episode_rows_v1") or 0)
    unified_closed_ledger_only_episode_count = int(unified_summary.get("closed_trade_ledger_only_episode_rows_v1") or 0)
    entry_direct_feature_coverage_status = (
        "FULL_ENTRY_DIRECT_FEATURE_COVERAGE"
        if unified_entry_direct_episode_count == int(len(work))
        else "PARTIAL_ENTRY_DIRECT_FEATURE_COVERAGE_MANAGEMENT_OR_LEDGER_ONLY_EPISODES_PRESENT"
    )
    consistency_rows = [
        {
            "check_name_v1": "RECOMMENDATION_VIEW_COVERS_CLOSED_TRADE_LEDGER_EXACTLY",
            "status_v1": "PASS" if int(len(work)) == int(len(ledger_df)) else "FAIL",
            "observed_value_v1": int(len(work)),
            "expected_value_v1": int(len(ledger_df)),
            "note_v1": "One recommendation row per closed truth trade.",
        },
        {
            "check_name_v1": "BATCH_REPLAY_COVERS_ALL_RUNS_EXACTLY_ONCE",
            "status_v1": "PASS"
            if int(batch_df["run_count_v1"].sum()) == int(len(run_ids)) and int(batch_df["baseline_trade_count_v1"].sum()) == int(len(work))
            else "FAIL",
            "observed_value_v1": json.dumps(
                {"run_count": int(batch_df["run_count_v1"].sum()), "trade_count": int(batch_df["baseline_trade_count_v1"].sum())},
                ensure_ascii=True,
                sort_keys=True,
            ),
            "expected_value_v1": json.dumps(
                {"run_count": int(len(run_ids)), "trade_count": int(len(work))},
                ensure_ascii=True,
                sort_keys=True,
            ),
            "note_v1": "The 15-week shadow replay batches must cover the whole truth run universe.",
        },
        {
            "check_name_v1": "PRIORITY_DELTA_IS_NON_OVERLAPPING",
            "status_v1": "PASS"
            if abs(float(batch_df["priority_counterfactual_delta_bps_v1"].sum()) - total_delta) < 1e-9
            else "FAIL",
            "observed_value_v1": float(batch_df["priority_counterfactual_delta_bps_v1"].sum()),
            "expected_value_v1": total_delta,
            "note_v1": "Priority counterfactual delta must equal trade-level non-overlapping recommendation deltas.",
        },
        {
            "check_name_v1": "UNIFIED_OBSERVABILITY_READY_BEFORE_RECOMMENDATIONS",
            "status_v1": "PASS",
            "observed_value_v1": unified_status.get("UNIFIED_RL_OBSERVABILITY_STATUS"),
            "expected_value_v1": "READY_ENTRY_AND_MANAGEMENT_OBSERVABILITY",
            "note_v1": "Recommendation candidate is downstream of unified observability only.",
        },
        {
            "check_name_v1": "ENTRY_PROPENSITY_NOT_USED_AS_READY",
            "status_v1": "PASS" if unified_status.get("ENTRY_PROPENSITY_STATUS") == "NOT_ESTABLISHED" else "FAIL",
            "observed_value_v1": unified_status.get("ENTRY_PROPENSITY_STATUS"),
            "expected_value_v1": "NOT_ESTABLISHED",
            "note_v1": "Entry side remains recommendation/shadow only; no synthetic propensity.",
        },
        {
            "check_name_v1": "UNIFIED_EPISODE_COVERAGE_FOR_RECOMMENDATION_REPLAY",
            "status_v1": "PASS" if unified_uncovered_count == 0 else "FAIL",
            "observed_value_v1": json.dumps(
                {
                    "covered_trade_count": unified_covered_count,
                    "uncovered_trade_count": unified_uncovered_count,
                    "coverage_rate": unified_coverage_rate,
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            "expected_value_v1": "covered_trade_count == baseline_trade_count",
            "note_v1": "Recommendation replay requires every closed truth trade to be present in the unified episode universe.",
        },
        {
            "check_name_v1": "ENTRY_DIRECT_FEATURE_COVERAGE_FOR_ENTRY_MODEL_RETRAIN",
            "status_v1": "PASS" if unified_entry_direct_episode_count == int(len(work)) else "WARN",
            "observed_value_v1": json.dumps(
                {
                    "entry_direct_episode_count": unified_entry_direct_episode_count,
                    "management_only_episode_count": unified_management_only_episode_count,
                    "closed_ledger_only_episode_count": unified_closed_ledger_only_episode_count,
                    "baseline_trade_count": int(len(work)),
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            "expected_value_v1": "entry_direct_episode_count == baseline_trade_count for pure entry-model retrain",
            "note_v1": "Full recommendation replay is ready; pure entry-model retrain must decide how to handle management-only and ledger-only episodes.",
        },
    ]
    audit_df = pd.DataFrame.from_records(consistency_rows)
    failed_checks = int(audit_df["status_v1"].astype("string").eq("FAIL").sum())
    warning_checks = int(audit_df["status_v1"].astype("string").eq("WARN").sum())

    summary = {
        "layer_name": "RL_RECOMMENDATION_CANDIDATE_SUMMARY_V1",
        "review_dir_v1": str(review_dir),
        "unified_dir_v1": str(unified_dir),
        "batch_weeks_v1": int(batch_weeks),
        "batch_count_v1": int(len(batch_df)),
        "run_count_v1": int(len(run_ids)),
        "baseline_trade_count_v1": int(len(work)),
        "baseline_total_pnl_bps_v1": total_baseline,
        "baseline_mean_pnl_bps_v1": _mean_numeric(work, "baseline_realized_pnl_bps_v1"),
        "recommendation_counts_v1": _counts(work, "rl_priority_recommendation_v1"),
        "entry_skip_recommendation_count_v1": int(work["rl_priority_recommendation_v1"].eq("SKIP_TRADE").sum()),
        "exit_earlier_recommendation_count_v1": int(work["rl_priority_recommendation_v1"].eq("EXIT_EARLIER").sum()),
        "hold_longer_recommendation_count_v1": int(work["rl_priority_recommendation_v1"].eq("HOLD_LONGER").sum()),
        "keep_baseline_recommendation_count_v1": int(work["rl_priority_recommendation_v1"].eq("KEEP_BASELINE").sum()),
        "priority_counterfactual_delta_bps_v1": total_delta,
        "shadow_upper_bound_pnl_bps_v1": total_baseline + total_delta,
        "entry_skip_avoided_loss_bps_v1": _sum_numeric(work, "entry_skip_avoided_loss_bps_v1"),
        "exit_earlier_saved_bps_v1": _sum_numeric(work, "exit_earlier_saved_bps_v1"),
        "hold_longer_extra_value_bps_v1": _sum_numeric(work, "hold_longer_extra_value_bps_v1"),
        "priority_entry_skip_delta_bps_v1": _sum_numeric(work, "rl_priority_entry_skip_delta_bps_v1"),
        "priority_exit_earlier_delta_bps_v1": _sum_numeric(work, "rl_priority_exit_earlier_delta_bps_v1"),
        "priority_hold_longer_delta_bps_v1": _sum_numeric(work, "rl_priority_hold_longer_delta_bps_v1"),
        "unified_episode_covered_trade_count_v1": unified_covered_count,
        "unified_episode_uncovered_trade_count_v1": unified_uncovered_count,
        "unified_episode_coverage_rate_v1": unified_coverage_rate,
        "unified_episode_coverage_status_v1": unified_episode_coverage_status,
        "entry_direct_feature_coverage_status_v1": entry_direct_feature_coverage_status,
        "unified_entry_direct_episode_count_v1": unified_entry_direct_episode_count,
        "unified_management_only_episode_count_v1": unified_management_only_episode_count,
        "unified_closed_trade_ledger_only_episode_count_v1": unified_closed_ledger_only_episode_count,
        "zero_trade_runs_v1": int(sum(int(row["zero_trade_run_count_v1"]) for row in batch_rows)),
        "opportunity_rich_zero_trade_runs_v1": int(
            sum(int(row["opportunity_rich_zero_trade_run_count_v1"]) for row in batch_rows)
        ),
        "failed_check_count_v1": failed_checks,
        "warning_check_count_v1": warning_checks,
    }
    if unified_uncovered_count > 0:
        retrain_ready_status = "RECOMMENDATION_LABELS_READY_BUT_REPLAY_BLOCKED_BY_UNIFIED_EPISODE_COVERAGE_GAP"
    elif unified_entry_direct_episode_count != int(len(work)):
        retrain_ready_status = "RECOMMENDATION_REPLAY_READY_FULL_EPISODE_COVERAGE_ENTRY_RETRAIN_NEEDS_SCOPE_REVIEW"
    else:
        retrain_ready_status = "RECOMMENDATION_LABELS_READY_FOR_RETRAIN_REVIEW_NOT_AUTO_RETRAIN"
    status = {
        "layer_name": "RL_RECOMMENDATION_CANDIDATE_STATUS_V1",
        "RL_RECOMMENDATION_CANDIDATE_STATUS": "READY_SHADOW_REPLAY_15WEEK" if failed_checks == 0 else "ISSUES_FOUND",
        "REPLAY_MODE_STATUS": "SHADOW_REPLAY_NOT_EXECUTION_REPLAY",
        "COUNTERFACTUAL_STATUS": "HINDSIGHT_UPPER_BOUND_ONLY",
        "BASELINE_COMPARISON_STATUS": "BASELINE_XGB_TRANSFORMER_REALIZED_TRUTH_VS_RL_RECOMMENDATION_SHADOW",
        "RETRAIN_READY_STATUS": retrain_ready_status,
        "UNIFIED_EPISODE_COVERAGE_STATUS": unified_episode_coverage_status,
        "ENTRY_DIRECT_FEATURE_COVERAGE_STATUS": entry_direct_feature_coverage_status,
        "ENTRY_PROPENSITY_STATUS": unified_status.get("ENTRY_PROPENSITY_STATUS"),
        "MANAGEMENT_PROPENSITY_STATUS": unified_status.get("MANAGEMENT_PROPENSITY_STATUS"),
        "not_trainer": True,
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    contract = {
        "layer_name": "RL_RECOMMENDATION_CANDIDATE_CONTRACT_V1",
        "mode_v1": "SHADOW_RECOMMENDATION_AND_15WEEK_REPLAY_COMPARISON",
        "priority_order_v1": ["SKIP_TRADE", "EXIT_EARLIER", "HOLD_LONGER", "KEEP_BASELINE"],
        "baseline_v1": "REALIZED_XGB_TRANSFORMER_TRUTH_REPLAY",
        "delta_semantics_v1": "HINDSIGHT_UPPER_BOUND_NOT_LIVE_COUNTERFACTUAL_FILL",
        "batch_weeks_v1": int(batch_weeks),
        "prohibitions_v1": [
            "Do not treat shadow upper-bound deltas as live replay fills.",
            "Do not auto-retrain XGB or Transformer from this layer without separate train/validation gate.",
            "Do not use entry recommendations as off-policy estimates while entry propensity is not established.",
            "Do not double-count skip, exit-earlier, and hold-longer deltas for the same trade.",
        ],
    }
    manifest = {
        "layer_name": "RL_RECOMMENDATION_CANDIDATE_MANIFEST_V1",
        "mode_v1": "APPEND_ONLY_EXTENSION",
        "review_dir_v1": str(review_dir),
        "unified_dir_v1": str(unified_dir),
        "artifacts_v1": {
            "contract_v1": RECOMMENDATION_CONTRACT,
            "trade_view_v1": RECOMMENDATION_TRADE_VIEW,
            "batch_replay_v1": RECOMMENDATION_BATCH_REPLAY,
            "summary_v1": RECOMMENDATION_SUMMARY,
            "status_v1": RECOMMENDATION_STATUS,
            "audit_v1": RECOMMENDATION_AUDIT,
            "markdown_v1": RECOMMENDATION_MD,
        },
    }
    return {
        "contract_v1": contract,
        "trade_view_v1_df": work,
        "batch_replay_v1_df": batch_df,
        "summary_v1": summary,
        "status_v1": status,
        "audit_v1_df": audit_df,
        "manifest_v1": manifest,
        "markdown_v1": _render_markdown({**summary, "status_v1": status}, batch_df),
    }


def materialize_truth_rl_recommendation_candidate(
    reports_root: Path,
    *,
    review_dir: Path | None = None,
    unified_dir: Path | None = None,
    extension_dir: Path | None = None,
    batch_weeks: int = 15,
) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    resolved_review_dir = _resolve_review_dir(reports_root, str(review_dir) if review_dir else None)
    resolved_unified_dir = _resolve_unified_dir(reports_root, str(unified_dir) if unified_dir else None)
    payload = build_rl_recommendation_candidate_payload(
        reports_root=reports_root,
        review_dir=resolved_review_dir,
        unified_dir=resolved_unified_dir,
        batch_weeks=batch_weeks,
    )
    if int(payload["summary_v1"].get("failed_check_count_v1", -1)) != 0:
        raise RuntimeError("RL_RECOMMENDATION_CANDIDATE_V1 consistency checks failed; refusing to materialize.")

    if extension_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        extension_dir = reports_root / f"{LEDGER_NAMESPACE_PREFIX}{stamp}_{RECOMMENDATION_EXTENSION_SUFFIX}"
    extension_dir = Path(extension_dir).expanduser().resolve()
    extension_dir.mkdir(parents=True, exist_ok=False)

    payload["trade_view_v1_df"].to_parquet(extension_dir / RECOMMENDATION_TRADE_VIEW, index=False)
    payload["batch_replay_v1_df"].to_csv(extension_dir / RECOMMENDATION_BATCH_REPLAY, index=False)
    payload["audit_v1_df"].to_csv(extension_dir / RECOMMENDATION_AUDIT, index=False)
    (extension_dir / RECOMMENDATION_CONTRACT).write_text(
        json.dumps(payload["contract_v1"], ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (extension_dir / RECOMMENDATION_SUMMARY).write_text(
        json.dumps(payload["summary_v1"], ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (extension_dir / RECOMMENDATION_STATUS).write_text(
        json.dumps(payload["status_v1"], ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (extension_dir / RECOMMENDATION_MANIFEST).write_text(
        json.dumps(payload["manifest_v1"], ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (extension_dir / RECOMMENDATION_MD).write_text(payload["markdown_v1"], encoding="utf-8")

    top_level_summary = dict(payload["summary_v1"])
    top_level_summary["extension_dir_v1"] = str(extension_dir)
    top_level_summary["review_dir_v1"] = str(resolved_review_dir)
    top_level_summary["unified_dir_v1"] = str(resolved_unified_dir)
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
    parser = argparse.ArgumentParser(description="Materialize RL recommendation candidate and 15-week shadow replay.")
    parser.add_argument("--reports-root", type=str, default=None)
    parser.add_argument("--review-dir", type=str, default=None)
    parser.add_argument("--unified-dir", type=str, default=None)
    parser.add_argument("--extension-dir", type=str, default=None)
    parser.add_argument("--batch-weeks", type=int, default=15)
    args = parser.parse_args()

    result = materialize_truth_rl_recommendation_candidate(
        _resolve_reports_root(args.reports_root),
        review_dir=Path(args.review_dir).expanduser().resolve() if args.review_dir else None,
        unified_dir=Path(args.unified_dir).expanduser().resolve() if args.unified_dir else None,
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
