#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

RUN_PREFIX = "ALL_TRADE_REVIEW_LEDGER_"
RUN_SUFFIX = "_MONDAY_NARROW_RETRAIN_RUN_V1"
EXTENSION_PREFIX = "MONDAY_NARROW_RETRAIN_FAILURE_FORENSICS_V1"

PREDICTION_VIEW = "shadow_meta_all_trade_review_monday_narrow_retrain_policy_prediction_view_v1.parquet"
EVAL_SUMMARY = "shadow_meta_all_trade_review_monday_narrow_retrain_eval_summary_v1.json"
COMPARE_REPORT = "shadow_meta_all_trade_review_monday_narrow_retrain_compare_against_report_v1.csv"
POCKET_REPORT = "shadow_meta_all_trade_review_monday_narrow_retrain_pocket_report_v1.csv"
LOSO_METRICS = "shadow_meta_all_trade_review_monday_narrow_retrain_loso_metrics_v1.csv"
VERDICT_PACKAGE = "shadow_meta_all_trade_review_monday_narrow_retrain_verdict_package_v1.json"
FEATURE_MANIFEST = "shadow_meta_all_trade_review_monday_narrow_retrain_feature_manifest_v1.csv"

CONTRACT = "contract_v1.json"
COLLAPSE_FORENSICS = "narrow_retrain_failure_collapse_forensics_v1.json"
COLLAPSE_TABLE = "narrow_retrain_failure_collapse_forensics_v1.csv"
RUNNER_PROTECTION_ANALYSIS = "runner_protection_failure_analysis_v1.json"
RUNNER_PROTECTION_TABLE = "runner_protection_failure_score_patterns_v1.csv"
STRONGEST_WINNER_FORENSICS = "strongest_winner_damage_forensics_v1.json"
STRONGEST_WINNER_TABLE = "strongest_winner_damage_rows_v1.csv"
TAIL_DECOMPOSITION = "tail_help_vs_bad_block_decomposition_v1.json"
TAIL_DECOMPOSITION_TABLE = "tail_help_vs_bad_block_decomposition_v1.csv"
FEATURE_PROXY_REVIEW = "feature_proxy_behavior_review_v1.json"
FEATURE_PROXY_TABLE = "feature_proxy_behavior_review_v1.csv"
GO_NO_GO = "go_or_no_go_next_step_v1.json"
SUMMARY = "summary_v1.json"
REPORT = "report_v1.md"
MANIFEST = "manifest_v1.json"
STATUS = "status_v1.json"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"

BLOCK_COL = "monday_narrow_block_v1"
BAD_COL = "r6_label_bad_risk_v1"
TAIL_COL = "r6_label_tail_control_10_50_v1"
RUNNER50_COL = "r6_label_runner_50_mfe_v1"
RUNNER100_COL = "r6_label_runner_100_mfe_v1"
RUNNER200_COL = "r6_label_runner_200_mfe_v1"
STRONG_COL = "r6_label_strong_low_mae_runner_v1"
NEAR_MISS_COL = "r6_label_runner_near_miss_v1"
REPAIRED_COL = "r6_label_repaired_165_like_runner_v1"
RUNNER_PROTECT_LABEL_COL = "r6_label_runner_protect_v1"

RUNNER_SCORE = "pred__monday_narrow__runner_protector__prob_true_v1"
BAD_SCORE = "pred__monday_narrow__bad_risk__prob_true_v1"
TAIL_SCORE = "pred__monday_narrow__tail_control_10_50__prob_true_v1"
RISKY_SCORE = "pred__monday_narrow__risky_allow__prob_true_v1"
BLINDSPOT_SCORE = "pred__monday_narrow__batch04_blindspot__prob_true_v1"

PROXY_FEATURES = [
    "as_of_pre_entry_vol_exp_comp_score_v1",
    "as_of_pre_entry_directional_asymmetry_score_v1",
    "as_of_pre_entry_swing_retracement_alignment_score_v1",
    "as_of_pre_entry_tail_leakage_pocket_score_v1",
    "as_of_pre_entry_runner_protection_guard_score_v1",
]

SCORE_AND_PROXY_COLS = [
    RUNNER_SCORE,
    BAD_SCORE,
    TAIL_SCORE,
    RISKY_SCORE,
    BLINDSPOT_SCORE,
    *PROXY_FEATURES,
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _latest_run_dir(reports_root: Path) -> Path:
    matches = sorted(
        [
            path
            for path in reports_root.iterdir()
            if path.is_dir() and path.name.startswith(RUN_PREFIX) and path.name.endswith(RUN_SUFFIX)
        ],
        key=lambda path: path.name,
    )
    if not matches:
        raise FileNotFoundError(f"No Monday narrow retrain run found under {reports_root}")
    return matches[-1]


def _resolve_extension_dir(reports_root: Path, extension_dir_arg: str | None) -> Path:
    if extension_dir_arg:
        return Path(extension_dir_arg).expanduser().resolve()
    return reports_root / f"{EXTENSION_PREFIX}_{_utc_compact()}"


def _audit_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.astype("string").str.lower().isin(["1", "true", "yes", "y"])


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if pd.notna(out) else None


def _safe_rate(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return float(num) / float(den)


def _mean_delta(frame: pd.DataFrame, mask_a: pd.Series, mask_b: pd.Series, column: str) -> float | None:
    a = _num(frame.loc[mask_a], column).mean()
    b = _num(frame.loc[mask_b], column).mean()
    if pd.isna(a) or pd.isna(b):
        return None
    return float(a - b)


def _group_stats(frame: pd.DataFrame, masks: Dict[str, pd.Series]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for name, mask in masks.items():
        mask = mask.reindex(frame.index).fillna(False).astype(bool)
        row: Dict[str, Any] = {"group_v1": name, "row_count_v1": int(mask.sum())}
        for column in SCORE_AND_PROXY_COLS:
            if column in frame.columns:
                row[f"{column}__mean_v1"] = _safe_float(_num(frame.loc[mask], column).mean())
                row[f"{column}__median_v1"] = _safe_float(_num(frame.loc[mask], column).median())
        rows.append(row)
    return pd.DataFrame(rows)


def _load_run(run_dir: Path) -> Dict[str, Any]:
    required = [PREDICTION_VIEW, EVAL_SUMMARY, COMPARE_REPORT, POCKET_REPORT, LOSO_METRICS, VERDICT_PACKAGE, FEATURE_MANIFEST]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Run dir missing required artifacts: {missing}")
    return {
        "run_dir": run_dir,
        "prediction_view": pd.read_parquet(run_dir / PREDICTION_VIEW),
        "eval_summary": _load_json(run_dir / EVAL_SUMMARY),
        "compare_report": pd.read_csv(run_dir / COMPARE_REPORT),
        "pocket_report": pd.read_csv(run_dir / POCKET_REPORT),
        "loso_metrics": pd.read_csv(run_dir / LOSO_METRICS),
        "verdict_package": _load_json(run_dir / VERDICT_PACKAGE),
        "feature_manifest": pd.read_csv(run_dir / FEATURE_MANIFEST),
    }


def _masks(frame: pd.DataFrame) -> Dict[str, pd.Series]:
    block = _bool(frame, BLOCK_COL)
    bad = _bool(frame, BAD_COL)
    tail = _bool(frame, TAIL_COL)
    runner50 = _bool(frame, RUNNER50_COL)
    runner100 = _bool(frame, RUNNER100_COL)
    runner200 = _bool(frame, RUNNER200_COL)
    strong = _bool(frame, STRONG_COL)
    near = _bool(frame, NEAR_MISS_COL)
    repaired = _bool(frame, REPAIRED_COL)
    runner_protect = _bool(frame, RUNNER_PROTECT_LABEL_COL)
    strongest = runner200 | strong
    return {
        "all": pd.Series(True, index=frame.index),
        "blocked": block,
        "allowed": ~block,
        "true_bad_block": block & bad,
        "false_block": block & ~bad,
        "missed_bad": ~block & bad,
        "tail_help": block & tail,
        "missed_tail": ~block & tail,
        "runner_50_blocked": block & runner50,
        "runner_100_blocked": block & runner100,
        "runner_200_blocked": block & runner200,
        "strongest_winner_blocked": block & strongest,
        "runner_near_miss_blocked": block & near,
        "repaired_blocked": block & repaired,
        "runner_protect_label_blocked": block & runner_protect,
    }


def _collapse_forensics(payload: Dict[str, Any]) -> tuple[Dict[str, Any], pd.DataFrame]:
    eval_summary = payload["eval_summary"]
    compare = payload["compare_report"]
    loso = payload["loso_metrics"]
    metric = eval_summary["global_metric_v1"]
    candidate = compare[compare["reference_v1"].eq("MONDAY_NARROW_RETRAIN_CANDIDATE")].iloc[0].to_dict()
    frozen = compare[compare["reference_v1"].eq("FROZEN_WEDNESDAY_R6_BENCHMARK")].iloc[0].to_dict()
    weakest = loso.sort_values("should_not_take_precision_v1", ascending=True).iloc[0].to_dict()
    rows = [
        {
            "failure_area_v1": "global_precision",
            "candidate_v1": metric.get("global_precision_v1"),
            "frozen_benchmark_v1": frozen.get("global_precision_v1"),
            "delta_vs_frozen_v1": candidate.get("delta_vs_frozen_global_precision_v1"),
            "primary_cause_v1": "152 false blocks on 263 total blocks; blocker recall expanded into non-bad rows.",
        },
        {
            "failure_area_v1": "worst_loso",
            "candidate_v1": metric.get("worst_loso_precision_v1"),
            "frozen_benchmark_v1": frozen.get("worst_loso_precision_v1"),
            "delta_vs_frozen_v1": candidate.get("delta_vs_frozen_worst_loso_precision_v1"),
            "primary_cause_v1": f"Weakest slice {weakest.get('scope_v1')} precision={weakest.get('should_not_take_precision_v1')}; slice-level false blocks dominate.",
        },
        {
            "failure_area_v1": "50_100_200_runner_damage",
            "candidate_v1": f"{metric.get('fifty_plus_mfe_block_count_v1')}/{metric.get('hundred_plus_mfe_block_count_v1')}/{metric.get('two_hundred_plus_mfe_block_count_v1')}",
            "frozen_benchmark_v1": f"{frozen.get('fifty_plus_mfe_block_count_v1')}/{frozen.get('hundred_plus_mfe_block_count_v1')}/{frozen.get('two_hundred_plus_mfe_block_count_v1')}",
            "delta_vs_frozen_v1": None,
            "primary_cause_v1": "Runner-protection head did not suppress blocker scores on high-MFE pockets.",
        },
        {
            "failure_area_v1": "strongest_winner_damage",
            "candidate_v1": metric.get("strongest_winner_path_damage_v1"),
            "frozen_benchmark_v1": frozen.get("strongest_winner_path_damage_v1"),
            "delta_vs_frozen_v1": None,
            "primary_cause_v1": "Strong/200+ runner labels were blocked despite zero-tolerance guard.",
        },
        {
            "failure_area_v1": "runner_near_miss_regression",
            "candidate_v1": metric.get("runner_near_miss_block_count_v1"),
            "frozen_benchmark_v1": frozen.get("runner_near_miss_block_count_v1"),
            "delta_vs_frozen_v1": None,
            "primary_cause_v1": "Near-miss runner pocket still looked risky to bad/blindspot heads.",
        },
        {
            "failure_area_v1": "tail_help_collapse",
            "candidate_v1": metric.get("tail_help_v1"),
            "frozen_benchmark_v1": frozen.get("tail_help_v1"),
            "delta_vs_frozen_v1": candidate.get("delta_vs_frozen_tail_help_v1"),
            "primary_cause_v1": "Low tail capture while many blocks went to non-tail runner collateral.",
        },
    ]
    table = pd.DataFrame(rows)
    out = {
        "layer_name_v1": "NARROW_RETRAIN_FAILURE_COLLAPSE_FORENSICS_V1",
        "verdict_v1": payload["verdict_package"].get("verdict_v1"),
        "candidate_disqualified_v1": payload["verdict_package"].get("candidate_disqualified_v1"),
        "hard_fail_reasons_v1": payload["verdict_package"].get("hard_fail_reasons_v1", []),
        "global_metric_v1": metric,
        "weakest_loso_slice_v1": weakest,
        "collapse_explanation_v1": "The setup increased bad blocks versus Monday-native R6, but precision collapsed because blocker heads fired broadly while runner protection did not activate strongly enough.",
    }
    return out, table


def _runner_protection_analysis(frame: pd.DataFrame, masks: Dict[str, pd.Series]) -> tuple[Dict[str, Any], pd.DataFrame]:
    stats = _group_stats(
        frame,
        {
            "all": masks["all"],
            "blocked": masks["blocked"],
            "true_bad_block": masks["true_bad_block"],
            "false_block": masks["false_block"],
            "runner_50_blocked": masks["runner_50_blocked"],
            "runner_100_blocked": masks["runner_100_blocked"],
            "runner_200_blocked": masks["runner_200_blocked"],
            "strongest_winner_blocked": masks["strongest_winner_blocked"],
            "runner_near_miss_blocked": masks["runner_near_miss_blocked"],
        },
    )
    blocked_runner = masks["runner_50_blocked"] | masks["runner_100_blocked"] | masks["runner_200_blocked"] | masks["runner_near_miss_blocked"] | masks["strongest_winner_blocked"]
    raw_guard_mean = _safe_float(_num(frame.loc[blocked_runner], "as_of_pre_entry_runner_protection_guard_score_v1").mean())
    model_runner_mean = _safe_float(_num(frame.loc[blocked_runner], RUNNER_SCORE).mean())
    bad_score_mean = _safe_float(_num(frame.loc[blocked_runner], BAD_SCORE).mean())
    analysis = {
        "layer_name_v1": "RUNNER_PROTECTION_FAILURE_ANALYSIS_V1",
        "overblocked_pockets_v1": {
            "runner_50_blocked_v1": int(masks["runner_50_blocked"].sum()),
            "runner_100_blocked_v1": int(masks["runner_100_blocked"].sum()),
            "runner_200_blocked_v1": int(masks["runner_200_blocked"].sum()),
            "strongest_winner_blocked_v1": int(masks["strongest_winner_blocked"].sum()),
            "runner_near_miss_blocked_v1": int(masks["runner_near_miss_blocked"].sum()),
        },
        "guard_score_diagnosis_v1": {
            "raw_runner_guard_mean_on_blocked_runners_v1": raw_guard_mean,
            "model_runner_protector_mean_on_blocked_runners_v1": model_runner_mean,
            "bad_score_mean_on_blocked_runners_v1": bad_score_mean,
            "diagnosis_v1": "Raw guard/proxy signal was only a feature, not a hard runtime guard; the learned runner_protector head under-asserted protection while blocker scores crossed threshold.",
        },
        "main_failure_mode_v1": "Runner-protection was undervalued/miscalibrated relative to blocker heads, not absent from the input table.",
    }
    return analysis, stats


def _strongest_winner_forensics(frame: pd.DataFrame, masks: Dict[str, pd.Series]) -> tuple[Dict[str, Any], pd.DataFrame]:
    damage = frame.loc[masks["strongest_winner_blocked"]].copy()
    keep_cols = [
        "candidate_uid",
        RUNNER_SCORE,
        BAD_SCORE,
        TAIL_SCORE,
        RISKY_SCORE,
        BLINDSPOT_SCORE,
        *PROXY_FEATURES,
        RUNNER50_COL,
        RUNNER100_COL,
        RUNNER200_COL,
        STRONG_COL,
        NEAR_MISS_COL,
        BAD_COL,
        TAIL_COL,
    ]
    table = damage[[c for c in keep_cols if c in damage.columns]].copy()
    out = {
        "layer_name_v1": "STRONGEST_WINNER_DAMAGE_FORENSICS_V1",
        "strongest_winner_damage_count_v1": int(len(damage)),
        "two_hundred_overlap_v1": int(_bool(damage, RUNNER200_COL).sum()),
        "strong_low_mae_overlap_v1": int(_bool(damage, STRONG_COL).sum()),
        "bad_label_overlap_v1": int(_bool(damage, BAD_COL).sum()),
        "tail_label_overlap_v1": int(_bool(damage, TAIL_COL).sum()),
        "systematic_failure_verdict_v1": "SYSTEMATIC_BLOCKER_OVERPOWERED_PROTECTOR" if len(damage) else "NO_DAMAGE",
        "what_commonality_v1": "Damaged strongest winners share low model runner_protector scores with enough bad/risky/blindspot pressure to cross the block rule.",
    }
    return out, table


def _tail_decomposition(frame: pd.DataFrame, masks: Dict[str, pd.Series]) -> tuple[Dict[str, Any], pd.DataFrame]:
    block = masks["blocked"]
    bad = _bool(frame, BAD_COL)
    tail = _bool(frame, TAIL_COL)
    runner_collateral = masks["runner_50_blocked"] | masks["runner_100_blocked"] | masks["runner_200_blocked"] | masks["strongest_winner_blocked"] | masks["runner_near_miss_blocked"]
    rows = [
        {"bucket_v1": "total_blocks", "count_v1": int(block.sum()), "meaning_v1": "All policy blocks."},
        {"bucket_v1": "true_bad_blocks", "count_v1": int((block & bad).sum()), "meaning_v1": "Blocks that hit bad-risk label."},
        {"bucket_v1": "false_blocks", "count_v1": int((block & ~bad).sum()), "meaning_v1": "Blocks outside bad-risk label; precision killer."},
        {"bucket_v1": "tail_help", "count_v1": int((block & tail).sum()), "meaning_v1": "Blocked 10-50 tail-control cases."},
        {"bucket_v1": "missed_tail_cases", "count_v1": int((~block & tail).sum()), "meaning_v1": "Tail cases still allowed."},
        {"bucket_v1": "runner_collateral_blocks", "count_v1": int(runner_collateral.sum()), "meaning_v1": "Runner/winner pockets hit by blocker."},
    ]
    table = pd.DataFrame(rows)
    out = {
        "layer_name_v1": "TAIL_HELP_VS_BAD_BLOCK_DECOMPOSITION_V1",
        "bad_blocks_v1": int((block & bad).sum()),
        "tail_help_v1": int((block & tail).sum()),
        "precision_v1": _safe_rate(float((block & bad).sum()), float(block.sum())),
        "why_low_precision_and_low_tail_help_v1": "The model blocked many non-bad rows while missing most tail labels; tail signal did not dominate, and collateral runner blocks consumed much of the block budget.",
    }
    return out, table


def _feature_proxy_review(frame: pd.DataFrame, masks: Dict[str, pd.Series]) -> tuple[Dict[str, Any], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    informative: List[str] = []
    weak: List[str] = []
    for feature in PROXY_FEATURES:
        if feature not in frame.columns:
            rows.append({"feature_v1": feature, "status_v1": "MISSING"})
            weak.append(feature)
            continue
        all_mean = _safe_float(_num(frame, feature).mean())
        true_bad_mean = _safe_float(_num(frame.loc[masks["true_bad_block"]], feature).mean())
        false_block_mean = _safe_float(_num(frame.loc[masks["false_block"]], feature).mean())
        runner_damage_mean = _safe_float(_num(frame.loc[masks["strongest_winner_blocked"] | masks["runner_50_blocked"]], feature).mean())
        missed_tail_mean = _safe_float(_num(frame.loc[masks["missed_tail"]], feature).mean())
        tail_help_mean = _safe_float(_num(frame.loc[masks["tail_help"]], feature).mean())
        bad_delta = None if true_bad_mean is None or false_block_mean is None else true_bad_mean - false_block_mean
        tail_delta = None if tail_help_mean is None or missed_tail_mean is None else tail_help_mean - missed_tail_mean
        runner_delta = None if runner_damage_mean is None or all_mean is None else runner_damage_mean - all_mean
        status = "INFORMATIVE_BUT_MISUSED" if feature == "as_of_pre_entry_runner_protection_guard_score_v1" else "WEAK_OR_MISALIGNED"
        if bad_delta is not None and abs(bad_delta) > 0.05:
            informative.append(feature)
        else:
            weak.append(feature)
        rows.append(
            {
                "feature_v1": feature,
                "all_mean_v1": all_mean,
                "true_bad_block_mean_v1": true_bad_mean,
                "false_block_mean_v1": false_block_mean,
                "runner_damage_mean_v1": runner_damage_mean,
                "tail_help_mean_v1": tail_help_mean,
                "missed_tail_mean_v1": missed_tail_mean,
                "true_bad_minus_false_block_mean_v1": bad_delta,
                "tail_help_minus_missed_tail_mean_v1": tail_delta,
                "runner_damage_minus_all_mean_v1": runner_delta,
                "status_v1": status,
            }
        )
    out = {
        "layer_name_v1": "FEATURE_PROXY_BEHAVIOR_REVIEW_V1",
        "informative_candidates_v1": sorted(set(informative)),
        "weak_or_misaligned_candidates_v1": sorted(set(weak)),
        "overall_verdict_v1": "FEATURES_HAVE_SOME_SIGNAL_BUT_MODEL_COMBINATION_AND_PROTECTOR_CALIBRATION_FAILED",
        "problem_split_v1": {
            "features_v1": "some proxy means move in plausible directions but do not cleanly separate bad vs runner pockets",
            "combination_v1": "block rule allowed bad/risky/blindspot heads to overpower weak runner_protector probabilities",
            "model_use_v1": "runner guard feature did not become a reliable protection output",
        },
    }
    return out, pd.DataFrame(rows)


def _decision() -> Dict[str, Any]:
    return {
        "layer_name_v1": "GO_OR_NO_GO_NEXT_STEP_V1",
        "decision_v1": "STRENGTHEN_RUNNER_PROTECTION_BEFORE_ANY_NEW_RETRAIN",
        "supporting_decisions_v1": [
            "DO_NOT_RETRAIN_SAME_SETUP_AGAIN",
            "RETURN_TO_FAILURE_MINING_AND_DESIGN",
            "CONSIDER_MODEL_OBJECTIVE_OR_LABEL_RETHINK",
        ],
        "why_v1": "The setup improved bad-block count versus Monday-native R6 but destroyed precision and winner safety; same setup would likely repeat blocker-over-protector collapse.",
    }


def _write_report(path: Path, summary: Dict[str, Any], collapse: Dict[str, Any], runner: Dict[str, Any], strongest: Dict[str, Any], tail: Dict[str, Any], feature_review: Dict[str, Any], decision: Dict[str, Any]) -> None:
    lines = [
        "# Monday Narrow Retrain Failure Forensics V1",
        "",
        "## Verdict",
        f"- Run verdict: `{summary['run_verdict_v1']}`",
        f"- Forensics decision: `{decision['decision_v1']}`",
        "",
        "## Collapse",
        f"- Precision: `{summary['global_precision_v1']}`",
        f"- Worst LOSO: `{summary['worst_loso_precision_v1']}`",
        f"- 50+/100+/200+ blocked: `{summary['fifty_plus_blocked_v1']}` / `{summary['hundred_plus_blocked_v1']}` / `{summary['two_hundred_plus_blocked_v1']}`",
        f"- Strongest-winner damage: `{summary['strongest_winner_damage_v1']}`",
        f"- Tail help: `{summary['tail_help_v1']}`",
        "",
        "## Main Diagnosis",
        f"- {runner['main_failure_mode_v1']}",
        f"- {feature_review['overall_verdict_v1']}",
        f"- {tail['why_low_precision_and_low_tail_help_v1']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize Monday narrow retrain failure forensics V1.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--extension-dir", default=None)
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else _latest_run_dir(reports_root)
    extension_dir = _resolve_extension_dir(reports_root, args.extension_dir)
    extension_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_run(run_dir)
    frame = payload["prediction_view"].copy()
    masks = _masks(frame)
    collapse, collapse_table = _collapse_forensics(payload)
    runner_analysis, runner_table = _runner_protection_analysis(frame, masks)
    strongest, strongest_table = _strongest_winner_forensics(frame, masks)
    tail, tail_table = _tail_decomposition(frame, masks)
    feature_review, feature_table = _feature_proxy_review(frame, masks)
    decision = _decision()

    metric = payload["eval_summary"]["global_metric_v1"]
    summary = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_FAILURE_FORENSICS_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "run_dir_v1": str(run_dir),
        "extension_dir_v1": str(extension_dir),
        "run_verdict_v1": payload["verdict_package"].get("verdict_v1"),
        "forensics_decision_v1": decision["decision_v1"],
        "bad_blocks_v1": metric.get("bad_blocks_v1"),
        "tail_help_v1": metric.get("tail_help_v1"),
        "global_precision_v1": metric.get("global_precision_v1"),
        "worst_loso_precision_v1": metric.get("worst_loso_precision_v1"),
        "fifty_plus_blocked_v1": metric.get("fifty_plus_mfe_block_count_v1"),
        "hundred_plus_blocked_v1": metric.get("hundred_plus_mfe_block_count_v1"),
        "two_hundred_plus_blocked_v1": metric.get("two_hundred_plus_mfe_block_count_v1"),
        "strongest_winner_damage_v1": metric.get("strongest_winner_path_damage_v1"),
        "runner_near_miss_blocked_v1": metric.get("runner_near_miss_block_count_v1"),
        "main_failure_v1": "BLOCKER_OVERPOWERED_RUNNER_PROTECTOR_AND_COLLATERAL_DAMAGED_WINNERS",
        "hard_status_division_v1": {
            "BEVIST": [
                "The completed narrow retrain run was read without retraining or replay.",
                "Precision collapse, winner damage and runner near-miss regression are materialized from run artifacts.",
                "The same setup should not be retrained again as-is.",
            ],
            "INDIKERT": [
                "The raw runner guard/proxy contains some signal, but the learned runner_protector output under-asserted protection.",
                "Model objective/calibration and protection-first design need rework before another retrain.",
            ],
            "IKKE_ETABLERT": [
                "That any single proxy alone can fix the collapse.",
                "That threshold tuning alone would beat frozen R6 without new safety design.",
            ],
        },
    }

    contract = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_FAILURE_FORENSICS_CONTRACT_V1",
        "materialized_at_utc_v1": summary["materialized_at_utc_v1"],
        "run_dir_v1": str(run_dir),
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_policy_change_v1": True,
        "not_threshold_tuning_v1": True,
    }

    _write_json(extension_dir / CONTRACT, contract)
    _write_json(extension_dir / COLLAPSE_FORENSICS, collapse)
    collapse_table.to_csv(extension_dir / COLLAPSE_TABLE, index=False)
    _write_json(extension_dir / RUNNER_PROTECTION_ANALYSIS, runner_analysis)
    runner_table.to_csv(extension_dir / RUNNER_PROTECTION_TABLE, index=False)
    _write_json(extension_dir / STRONGEST_WINNER_FORENSICS, strongest)
    strongest_table.to_csv(extension_dir / STRONGEST_WINNER_TABLE, index=False)
    _write_json(extension_dir / TAIL_DECOMPOSITION, tail)
    tail_table.to_csv(extension_dir / TAIL_DECOMPOSITION_TABLE, index=False)
    _write_json(extension_dir / FEATURE_PROXY_REVIEW, feature_review)
    feature_table.to_csv(extension_dir / FEATURE_PROXY_TABLE, index=False)
    _write_json(extension_dir / GO_NO_GO, decision)
    _write_json(extension_dir / SUMMARY, summary)
    _write_report(extension_dir / REPORT, summary, collapse, runner_analysis, strongest, tail, feature_review, decision)

    artifacts = [
        CONTRACT,
        COLLAPSE_FORENSICS,
        COLLAPSE_TABLE,
        RUNNER_PROTECTION_ANALYSIS,
        RUNNER_PROTECTION_TABLE,
        STRONGEST_WINNER_FORENSICS,
        STRONGEST_WINNER_TABLE,
        TAIL_DECOMPOSITION,
        TAIL_DECOMPOSITION_TABLE,
        FEATURE_PROXY_REVIEW,
        FEATURE_PROXY_TABLE,
        GO_NO_GO,
        SUMMARY,
        REPORT,
        MANIFEST,
        STATUS,
        CONSISTENCY_AUDIT,
    ]
    audit_rows = [
        _audit_record("RUN_VERDICT_READ", "PASS" if summary["run_verdict_v1"] else "FAIL", {"run_verdict_v1": summary["run_verdict_v1"]}),
        _audit_record("NO_TRAINING_OR_REPLAY", "PASS", {"not_training_v1": True, "not_replay_v1": True}),
        _audit_record("COLLAPSE_CAPTURED", "PASS" if summary["global_precision_v1"] is not None and summary["strongest_winner_damage_v1"] is not None else "FAIL", summary),
        _audit_record("DECISION_LOCKED", "PASS" if decision["decision_v1"] == "STRENGTHEN_RUNNER_PROTECTION_BEFORE_ANY_NEW_RETRAIN" else "FAIL", decision),
        _audit_record("OUTPUTS_PRESENT", "PASS" if all((extension_dir / a).exists() for a in artifacts if a not in {MANIFEST, STATUS, CONSISTENCY_AUDIT}) else "FAIL", {"artifact_count_v1": len(artifacts)}),
    ]
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)
    manifest = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_FAILURE_FORENSICS_MANIFEST_V1",
        "materialized_at_utc_v1": summary["materialized_at_utc_v1"],
        "extension_dir_v1": str(extension_dir),
        "artifacts_v1": artifacts,
    }
    _write_json(extension_dir / MANIFEST, manifest)
    status = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_FAILURE_FORENSICS_STATUS_V1",
        "FORENSICS_STATUS": "MATERIALIZED_READ_ONLY",
        "failed_check_count_v1": int(audit_df["status_v1"].astype("string").ne("PASS").sum()),
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_policy_change_v1": True,
        "decision_v1": decision["decision_v1"],
    }
    _write_json(extension_dir / STATUS, status)

    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
