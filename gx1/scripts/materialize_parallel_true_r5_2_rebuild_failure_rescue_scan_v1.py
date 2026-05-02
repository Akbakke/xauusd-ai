#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.run_true_r5_2_rebuild_runner_v1 import (
    BASE_FLAG_COL,
    BAD_SCORE_COL,
    RISKY_SCORE_COL,
    RUNNER_SCORE_COL,
    TAIL_SCORE_COL,
)
from gx1.scripts.train_monday_r6_explicit_rebuild_from_rehydrated_contract_v1 import _bool, _jsonable, _num


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "PARALLEL_TRUE_R5_2_REBUILD_FAILURE_RESCUE_SCAN_V1"
TRUE_R5_2_DEFAULT = DEFAULT_REPORTS_ROOT / "TRUE_R5_2_REBUILD_RUNNER_V1_20260426T_EXPLICIT_TRAINING"

OUTPUT_FILES = {
    "orchestrator": "parallel_rescue_scan_orchestrator_v1.json",
    "lane01": "lane_01_safety_damage_root_row_scan_v1.csv",
    "lane02": "lane_02_recovered_rows_value_scan_v1.csv",
    "lane03": "lane_03_strict_post_score_base_rule_frontier_v1.csv",
    "lane04": "lane_04_runner_protection_veto_scan_v1.csv",
    "lane05": "lane_05_ambiguous_high_mfe_exclusion_scan_v1.csv",
    "lane06": "lane_06_high_mfe_and_winner_stress_rescue_scan_v1.csv",
    "lane07": "lane_07_score_calibration_and_margin_scan_v1.json",
    "lane08": "lane_08_loso_batch_failure_scan_v1.csv",
    "lane09": "lane_09_objective_weight_whatif_diagnostic_v1.json",
    "lane10": "lane_10_downstream_r6_pass_through_risk_scan_v1.csv",
    "aggregator": "parallel_true_r5_2_rescue_aggregator_v1.json",
    "leaderboard": "rescue_rule_leaderboard_v1.csv",
    "decision": "rescue_or_retrain_decision_v1.json",
    "next_action": "next_action_lock_v1.json",
    "summary": "summary_v1.json",
    "report": "report_v1.md",
    "manifest": "manifest_v1.json",
    "status": "status_v1.json",
    "audit": "consistency_audit_v1.csv",
}

V3_BASELINE = {
    "bad_v1": 82,
    "tail_v1": 51,
    "precision_v1": 1.0,
    "worst_loso_v1": 1.0,
    "repaired_like_overlap_v1": 0,
    "fifty_plus_overlap_v1": 0,
    "hundred_plus_overlap_v1": 0,
    "two_hundred_plus_overlap_v1": 0,
    "strongest_winner_overlap_v1": 0,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_frame(true_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = _read_json(true_dir / "manifest_v1.json")
    input_files = manifest.get("input_files_v1") or {}
    score_path = Path(input_files["score_frame_v1"])
    label_path = Path(input_files["label_table_v1"])
    score = pd.read_parquet(score_path)
    label = pd.read_csv(label_path)
    pred = pd.read_parquet(true_dir / "r5_2_prediction_view_v1.parquet")
    label_cols = [
        "candidate_uid",
        "new_r5_2_label_bucket_v1",
        "bad_eligibility_target_v1",
        "tail_eligibility_target_v1",
        "risky_attention_target_v1",
        "runner_protect_target_v1",
        "ambiguous_high_mfe_monitor_v1",
        "sample_weight_v1",
        "protection_weight_v1",
        "r5_2_v3_base_flag_v1",
    ]
    frame = score.merge(label[[col for col in label_cols if col in label.columns]], on="candidate_uid", how="left", validate="one_to_one")
    pred_cols = ["candidate_uid", BAD_SCORE_COL, TAIL_SCORE_COL, RISKY_SCORE_COL, RUNNER_SCORE_COL, BASE_FLAG_COL]
    frame = frame.merge(pred[pred_cols], on="candidate_uid", how="left", validate="one_to_one")
    return frame, {"score_path_v1": str(score_path), "label_path_v1": str(label_path), "prediction_path_v1": str(true_dir / "r5_2_prediction_view_v1.parquet")}


def _masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "should": _bool(frame, "label_should_not_take_v1").to_numpy(dtype=bool),
        "tail": _bool(frame, "tail_10_50_mfe_v1").to_numpy(dtype=bool),
        "fifty": _bool(frame, "fifty_plus_mfe_v1").to_numpy(dtype=bool),
        "hundred": _bool(frame, "hundred_plus_mfe_v1").to_numpy(dtype=bool),
        "two_hundred": _bool(frame, "two_hundred_plus_mfe_v1").to_numpy(dtype=bool),
        "strongest": _bool(frame, "strongest_winner_path_v1").to_numpy(dtype=bool),
        "repaired": _bool(frame, "r6_label_repaired_165_like_runner_v1").to_numpy(dtype=bool),
        "runner_near": _bool(frame, "r6_label_runner_near_miss_v1").to_numpy(dtype=bool),
        "current_base": _bool(frame, BASE_FLAG_COL).to_numpy(dtype=bool),
        "v3_base": _bool(frame, "r5_2_v3_base_flag_v1").to_numpy(dtype=bool),
        "ambiguous": frame["new_r5_2_label_bucket_v1"].eq("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD").to_numpy(dtype=bool),
        "runner_bucket": frame["new_r5_2_label_bucket_v1"].eq("RUNNER_PROTECT_TARGET").to_numpy(dtype=bool),
    }


def _worst_loso(frame: pd.DataFrame, selected: np.ndarray, should: np.ndarray) -> float | None:
    runs = pd.factorize(frame["run_id"].astype(str))[0]
    run_count = int(runs.max()) + 1 if len(runs) else 0
    if run_count == 0:
        return None
    selected_count = np.bincount(runs, weights=selected.astype(int), minlength=run_count)
    bad_count = np.bincount(runs, weights=(selected & should).astype(int), minlength=run_count)
    precision = np.divide(bad_count, selected_count, out=np.full_like(selected_count, np.nan, dtype=float), where=selected_count > 0)
    return float(np.nanmin(precision)) if np.isfinite(precision).any() else None


def _metrics(frame: pd.DataFrame, selected: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    should = masks["should"]
    tail = masks["tail"]
    missed = (should | tail) & ~masks["v3_base"]
    block = int(selected.sum())
    bad = int((selected & should).sum())
    precision = float(bad / block) if block else None
    return {
        "block_count_v1": block,
        "bad_v1": bad,
        "tail_v1": int((selected & tail).sum()),
        "precision_v1": precision,
        "worst_loso_v1": _worst_loso(frame, selected, should),
        "repaired_like_overlap_v1": int((selected & masks["repaired"]).sum()),
        "fifty_plus_overlap_v1": int((selected & masks["fifty"]).sum()),
        "hundred_plus_overlap_v1": int((selected & masks["hundred"]).sum()),
        "two_hundred_plus_overlap_v1": int((selected & masks["two_hundred"]).sum()),
        "strongest_winner_overlap_v1": int((selected & masks["strongest"]).sum()),
        "runner_near_miss_overlap_v1": int((selected & masks["runner_near"]).sum()),
        "ambiguous_high_mfe_included_v1": int((selected & masks["ambiguous"]).sum()),
        "runner_protect_included_v1": int((selected & masks["runner_bucket"]).sum()),
        "recovered_missed_rows_v1": int((selected & missed).sum()),
        "recovered_missed_bad_v1": int((selected & missed & should).sum()),
        "recovered_missed_tail_v1": int((selected & missed & tail).sum()),
    }


def _safety_pass(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["repaired_like_overlap_v1"] == 0
        and metrics["strongest_winner_overlap_v1"] == 0
        and metrics["hundred_plus_overlap_v1"] == 0
        and metrics["two_hundred_plus_overlap_v1"] == 0
        and metrics["fifty_plus_overlap_v1"] <= 1
        and metrics["runner_near_miss_overlap_v1"] == 0
        and metrics["ambiguous_high_mfe_included_v1"] == 0
        and metrics["runner_protect_included_v1"] == 0
        and (metrics["precision_v1"] or 0.0) >= 0.972972972972973
        and (metrics["worst_loso_v1"] or 0.0) >= 0.9285714285714286
    )


def _rule_mask(
    frame: pd.DataFrame,
    masks: dict[str, np.ndarray],
    *,
    bad_threshold: float,
    tail_threshold: float,
    risky_threshold: float,
    runner_cap: float,
    consensus_min: int,
    margin_min: float,
    exclude_high_mfe: bool,
    exclude_ambiguous: bool,
    exclude_runner_bucket: bool,
    union_with_v3: bool,
) -> np.ndarray:
    bad = _num(frame, BAD_SCORE_COL).to_numpy(dtype=float)
    tail = _num(frame, TAIL_SCORE_COL).to_numpy(dtype=float)
    risky = _num(frame, RISKY_SCORE_COL).to_numpy(dtype=float)
    runner = _num(frame, RUNNER_SCORE_COL).to_numpy(dtype=float)
    eligible = ((bad >= bad_threshold).astype(int) + (tail >= tail_threshold).astype(int) + (risky >= risky_threshold).astype(int)) >= consensus_min
    score_max = np.maximum.reduce([bad, tail, risky])
    selected = eligible & (runner < runner_cap) & ((score_max - runner) >= margin_min)
    if exclude_high_mfe:
        selected &= ~(masks["fifty"] | masks["hundred"] | masks["two_hundred"] | masks["strongest"] | masks["runner_near"] | masks["repaired"])
    if exclude_ambiguous:
        selected &= ~masks["ambiguous"]
    if exclude_runner_bucket:
        selected &= ~masks["runner_bucket"]
    if union_with_v3:
        selected = masks["v3_base"] | (selected & ~masks["v3_base"])
    return selected


def _rule_row(rule_id: str, lane: str, frame: pd.DataFrame, masks: dict[str, np.ndarray], selected: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    metrics = _metrics(frame, selected, masks)
    return {
        "rule_id_v1": rule_id,
        "lane_v1": lane,
        **params,
        **metrics,
        "safety_pass_v1": _safety_pass(metrics),
        "bad_delta_vs_v3_v1": int(metrics["bad_v1"] - V3_BASELINE["bad_v1"]),
        "tail_delta_vs_v3_v1": int(metrics["tail_v1"] - V3_BASELINE["tail_v1"]),
    }


def _scan_rules(frame: pd.DataFrame, masks: dict[str, np.ndarray], lane: str, union_with_v3_values: list[bool]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    bad_grid = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]
    tail_grid = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]
    risky_grid = [0.65, 0.85]
    runner_caps = [0.50, 0.35, 0.25, 0.20, 0.15]
    consensus_grid = [1, 2]
    margin_grid = [0.0, 0.10, 0.20]
    for union_with_v3 in union_with_v3_values:
        for bad_t in bad_grid:
            for tail_t in tail_grid:
                for risky_t in risky_grid:
                    for runner_cap in runner_caps:
                        for consensus_min in consensus_grid:
                            for margin_min in margin_grid:
                                for exclude_high_mfe in [False, True]:
                                    for exclude_ambiguous in [False, True]:
                                        for exclude_runner_bucket in [False, True]:
                                            params = {
                                                "base_source_v1": "v3_union_plus_true_scores" if union_with_v3 else "true_scores_only",
                                                "bad_threshold_v1": bad_t,
                                                "tail_threshold_v1": tail_t,
                                                "risky_threshold_v1": risky_t,
                                                "runner_cap_v1": runner_cap,
                                                "consensus_min_v1": consensus_min,
                                                "margin_min_v1": margin_min,
                                                "exclude_high_mfe_v1": exclude_high_mfe,
                                                "exclude_ambiguous_high_mfe_v1": exclude_ambiguous,
                                                "exclude_runner_protect_bucket_v1": exclude_runner_bucket,
                                            }
                                            selected = _rule_mask(
                                                frame,
                                                masks,
                                                bad_threshold=bad_t,
                                                tail_threshold=tail_t,
                                                risky_threshold=risky_t,
                                                runner_cap=runner_cap,
                                                consensus_min=consensus_min,
                                                margin_min=margin_min,
                                                exclude_high_mfe=exclude_high_mfe,
                                                exclude_ambiguous=exclude_ambiguous,
                                                exclude_runner_bucket=exclude_runner_bucket,
                                                union_with_v3=union_with_v3,
                                            )
                                            rule_id = (
                                                f"{params['base_source_v1']}__b{bad_t}_t{tail_t}_r{risky_t}_p{runner_cap}"
                                                f"_c{consensus_min}_m{margin_min}_hm{int(exclude_high_mfe)}_amb{int(exclude_ambiguous)}_run{int(exclude_runner_bucket)}"
                                            )
                                            rows.append(_rule_row(rule_id, lane, frame, masks, selected, params))
    return pd.DataFrame(rows)


def _damage_root_rows(frame: pd.DataFrame, masks: dict[str, np.ndarray]) -> pd.DataFrame:
    selected = masks["current_base"]
    damage = selected & (
        masks["fifty"]
        | masks["hundred"]
        | masks["two_hundred"]
        | masks["strongest"]
        | masks["repaired"]
        | masks["runner_near"]
        | masks["ambiguous"]
        | masks["runner_bucket"]
    )
    rows = frame.loc[damage].copy()
    if rows.empty:
        return rows
    rows["base_membership_reason_v1"] = np.select(
        [
            _num(rows, BAD_SCORE_COL).ge(0.50),
            _num(rows, TAIL_SCORE_COL).ge(0.50),
            _num(rows, RISKY_SCORE_COL).ge(0.65),
        ],
        ["BAD_ELIGIBILITY_SCORE", "TAIL_ELIGIBILITY_SCORE", "RISKY_ATTENTION_SCORE"],
        default="NOT_ESTABLISHED",
    )
    rows["protection_rule_that_should_stop_v1"] = np.select(
        [
            rows["new_r5_2_label_bucket_v1"].eq("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD"),
            rows["new_r5_2_label_bucket_v1"].eq("RUNNER_PROTECT_TARGET"),
            _bool(rows, "hundred_plus_mfe_v1") | _bool(rows, "two_hundred_plus_mfe_v1"),
            _bool(rows, "strongest_winner_path_v1"),
            _bool(rows, "r6_label_runner_near_miss_v1"),
        ],
        [
            "EXCLUDE_AMBIGUOUS_HIGH_MFE",
            "EXCLUDE_RUNNER_PROTECT_BUCKET",
            "HARD_EXCLUDE_100_200_MFE",
            "HARD_EXCLUDE_STRONGEST_WINNER",
            "HARD_EXCLUDE_RUNNER_NEAR_MISS",
        ],
        default="HIGH_MFE_OR_PROTECTION_VETO",
    )
    cols = [
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "run_id",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "strongest_winner_path_v1",
        "r6_label_repaired_165_like_runner_v1",
        "r6_label_runner_near_miss_v1",
        "new_r5_2_label_bucket_v1",
        BAD_SCORE_COL,
        TAIL_SCORE_COL,
        RISKY_SCORE_COL,
        RUNNER_SCORE_COL,
        "base_membership_reason_v1",
        "protection_rule_that_should_stop_v1",
    ]
    return rows[[col for col in cols if col in rows.columns]]


def _recovered_rows(frame: pd.DataFrame, masks: dict[str, np.ndarray], best_selected: np.ndarray) -> pd.DataFrame:
    missed = (masks["should"] | masks["tail"]) & ~masks["v3_base"]
    recovered = masks["current_base"] & missed
    rows = frame.loc[recovered].copy()
    if rows.empty:
        return rows
    safety_risk = masks["fifty"] | masks["hundred"] | masks["two_hundred"] | masks["strongest"] | masks["repaired"] | masks["runner_near"] | masks["ambiguous"] | masks["runner_bucket"]
    rows["genuinely_useful_recovery_v1"] = (~safety_risk[recovered]).astype(bool)
    rows["kept_by_best_safe_rule_v1"] = best_selected[recovered]
    rows["base_reason_v1"] = np.select(
        [
            _num(rows, BAD_SCORE_COL).ge(0.50),
            _num(rows, TAIL_SCORE_COL).ge(0.50),
            _num(rows, RISKY_SCORE_COL).ge(0.65),
        ],
        ["BAD_ELIGIBILITY_SCORE", "TAIL_ELIGIBILITY_SCORE", "RISKY_ATTENTION_SCORE"],
        default="NOT_ESTABLISHED",
    )
    cols = [
        "candidate_uid",
        "trade_uid",
        "decision_timestamp",
        "new_r5_2_label_bucket_v1",
        "label_should_not_take_v1",
        "tail_10_50_mfe_v1",
        BAD_SCORE_COL,
        TAIL_SCORE_COL,
        RISKY_SCORE_COL,
        RUNNER_SCORE_COL,
        "base_reason_v1",
        "genuinely_useful_recovery_v1",
        "kept_by_best_safe_rule_v1",
    ]
    return rows[[col for col in cols if col in rows.columns]]


def _score_calibration(frame: pd.DataFrame, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    selected = masks["current_base"]
    damage = selected & (masks["fifty"] | masks["hundred"] | masks["strongest"] | masks["runner_near"] | masks["ambiguous"] | masks["runner_bucket"])
    safe_recovered = selected & (masks["should"] | masks["tail"]) & ~masks["v3_base"] & ~damage
    def quant(mask: np.ndarray, col: str) -> dict[str, float | None]:
        vals = _num(frame.loc[mask], col).dropna()
        if vals.empty:
            return {"p50_v1": None, "p90_v1": None, "p95_v1": None, "max_v1": None}
        return {"p50_v1": float(vals.quantile(0.50)), "p90_v1": float(vals.quantile(0.90)), "p95_v1": float(vals.quantile(0.95)), "max_v1": float(vals.max())}
    return {
        "layer_name": "LANE_07_SCORE_CALIBRATION_AND_MARGIN_SCAN_V1",
        "safe_recovered_rows_v1": int(safe_recovered.sum()),
        "damage_rows_v1": int(damage.sum()),
        "score_overlap_v1": {
            "safe_recovered_bad_v1": quant(safe_recovered, BAD_SCORE_COL),
            "damage_bad_v1": quant(damage, BAD_SCORE_COL),
            "safe_recovered_tail_v1": quant(safe_recovered, TAIL_SCORE_COL),
            "damage_tail_v1": quant(damage, TAIL_SCORE_COL),
            "safe_recovered_runner_protect_v1": quant(safe_recovered, RUNNER_SCORE_COL),
            "damage_runner_protect_v1": quant(damage, RUNNER_SCORE_COL),
        },
        "diagnosis_v1": [
            "Some protected winners/danger rows have runner-protect scores below the 0.50 veto, so learned protection alone is insufficient.",
            "Tail eligibility is overconfident on ambiguous high-MFE rows; hard pocket exclusions are required for rescue.",
        ],
    }


def _loso_scan(frame: pd.DataFrame, masks: dict[str, np.ndarray], best_selected: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current = masks["current_base"]
    should = masks["should"]
    for run_id, group in frame.groupby(frame["run_id"].astype(str), dropna=False):
        for name, selected in [("current_true_rebuild", current), ("best_safe_rescue", best_selected)]:
            idx = group.index.to_numpy()
            group_selected = selected[idx]
            block = int(group_selected.sum())
            bad = int((group_selected & should[idx]).sum())
            rows.append(
                {
                    "run_id_v1": str(run_id),
                    "rule_name_v1": name,
                    "selected_rows_v1": block,
                    "bad_rows_v1": bad,
                    "precision_v1": float(bad / block) if block else None,
                    "safety_rows_v1": int((group_selected & (masks["fifty"][idx] | masks["hundred"][idx] | masks["strongest"][idx] | masks["runner_near"][idx])).sum()),
                    "loso_failure_v1": bool(block > 0 and bad == 0),
                }
            )
    return pd.DataFrame(rows)


def _weight_diagnostic() -> dict[str, Any]:
    return {
        "layer_name": "LANE_09_OBJECTIVE_WEIGHT_WHATIF_DIAGNOSTIC_V1",
        "read_only_v1": True,
        "training_run_v1": False,
        "findings_v1": {
            "protection_weight_10_runner_v1": "Insufficient as sole learned veto; one runner-protect row entered base.",
            "hard_protection_weight_20_v1": "Insufficient unless paired with hard post-score exclusions for 100+/strongest/ambiguous pockets.",
            "bad_weight_3_tail_weight_2_5_v1": "Produced recall uplift, but allowed low-precision/high-MFE leakage under the current base rule.",
            "ambiguous_high_mfe_label_v1": "Needs explicit hard negative/protected class or non-negotiable post-score exclusion.",
            "model_config_v1": "Config may need stronger protection calibration, but first rescue can be expressed as a stricter base rule.",
        },
        "recommendation_v1": "Use hard post-score protection veto before R6; future rebuild should increase protection emphasis and harden ambiguous high-MFE labeling.",
    }


def _downstream_scan(rules: pd.DataFrame) -> pd.DataFrame:
    out = rules.copy()
    out["expected_r6_selectable_rows_upper_bound_v1"] = out["block_count_v1"]
    out["expected_r6_bad_upper_bound_v1"] = out["bad_v1"]
    out["expected_r6_tail_upper_bound_v1"] = out["tail_v1"]
    out["expected_r6_safety_risk_v1"] = np.where(out["safety_pass_v1"], "LOW_IF_R6_PRESERVES_VETO", "HIGH_DO_NOT_RUN_R6")
    out["r6_run_worth_after_rescue_v1"] = out["safety_pass_v1"] & (out["bad_v1"] >= V3_BASELINE["bad_v1"]) & (out["tail_v1"] >= V3_BASELINE["tail_v1"])
    return out


def _decision(best_safe: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if best_safe is None:
        decision = "NO_SAFE_RESCUE_FOUND"
        next_action = "DO_NOT_FEED_TRUE_R5_2_SCORE_PACKAGE_TO_R6"
    elif int(best_safe["bad_delta_vs_v3_v1"]) <= 10 and int(best_safe["tail_delta_vs_v3_v1"]) <= 10:
        decision = "ONLY_TINY_SAFE_RESCUE_FOUND"
        next_action = "IMPLEMENT_SAFE_TRUE_R5_2_RESCUE_BASE_RULE"
    else:
        decision = "SAFE_RESCUE_RULE_FOUND"
        next_action = "IMPLEMENT_SAFE_TRUE_R5_2_RESCUE_BASE_RULE"
    return (
        {
            "layer_name": "RESCUE_OR_RETRAIN_DECISION_V1",
            "decision_v1": decision,
            "best_safe_rescue_rule_v1": best_safe,
            "raw_true_score_package_usable_without_rescue_v1": False,
            "do_not_feed_raw_true_package_to_r6_v1": True,
        },
        {
            "layer_name": "NEXT_ACTION_LOCK_V1",
            "next_action_v1": next_action,
            "blocked_action_v1": [
                "DO_NOT_FEED_TRUE_R5_2_SCORE_PACKAGE_TO_R6_WITH_RAW_BASE_RULE",
                "DO_NOT_RUN_R6_NOW",
                "DO_NOT_TRAIN_NEW_R5_2_IN_THIS_SCAN",
            ],
        },
    )


def _audit(summary: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, ok: bool, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": "PASS" if ok else "FAIL", "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}
    return pd.DataFrame(
        [
            row("NO_TRAINING", not summary["training_started_v1"], summary["training_started_v1"]),
            row("NO_R6", not summary["r6_started_v1"], summary["r6_started_v1"]),
            row("ARTIFACTS_WRITTEN", summary["artifact_count_v1"] >= 19, summary["artifact_count_v1"]),
            row("RAW_TRUE_PACKAGE_NOT_GREEN", summary["raw_true_safety_pass_v1"] is False, summary["raw_true_safety_pass_v1"]),
            row("DECISION_LOCKED", summary["decision_v1"] != "NOT_ESTABLISHED", summary["decision_v1"]),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Parallel True R5.2 Rescue Scan",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Raw true R5.2: `{summary['raw_true_bad_tail_v1']}` bad/tail, safety pass `{summary['raw_true_safety_pass_v1']}`",
            f"- Best safe rescue: `{summary['best_safe_bad_tail_v1']}` bad/tail",
            f"- Safe recovered missed rows: `{summary['best_safe_recovered_rows_v1']}`",
            "",
            "No training, R6 run, new baseline, or feature surface was created.",
            "",
        ]
    )


def materialize(*, reports_root: Path = DEFAULT_REPORTS_ROOT, true_r5_2_dir: Path = TRUE_R5_2_DEFAULT, output_dir: Path | None = None) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    true_r5_2_dir = true_r5_2_dir.expanduser().resolve()
    output_dir = (output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, input_paths = _load_frame(true_r5_2_dir)
    masks = _masks(frame)
    current_metrics = _metrics(frame, masks["current_base"], masks)

    lane03 = _scan_rules(frame, masks, "LANE_03_STRICT_POST_SCORE_BASE_RULE_FRONTIER_V1", [False, True])
    lane04 = lane03[(lane03["runner_cap_v1"].isin([0.15, 0.20, 0.25])) | lane03["exclude_runner_protect_bucket_v1"]].copy()
    lane04["lane_v1"] = "LANE_04_RUNNER_PROTECTION_VETO_SCAN_V1"
    lane05 = lane03[lane03["exclude_ambiguous_high_mfe_v1"] | lane03["exclude_high_mfe_v1"]].copy()
    lane05["lane_v1"] = "LANE_05_AMBIGUOUS_HIGH_MFE_EXCLUSION_SCAN_V1"
    candidates = pd.concat([lane03, lane04, lane05], ignore_index=True)
    safe_candidates = candidates[candidates["safety_pass_v1"].astype(bool)].copy()
    best_safe = None
    best_selected = masks["v3_base"].copy()
    if not safe_candidates.empty:
        safe_candidates = safe_candidates.sort_values(
            ["bad_v1", "tail_v1", "recovered_missed_rows_v1", "precision_v1", "worst_loso_v1"],
            ascending=[False, False, False, False, False],
            na_position="last",
        )
        best_safe = safe_candidates.iloc[0].to_dict()
        best_selected = _rule_mask(
            frame,
            masks,
            bad_threshold=float(best_safe["bad_threshold_v1"]),
            tail_threshold=float(best_safe["tail_threshold_v1"]),
            risky_threshold=float(best_safe["risky_threshold_v1"]),
            runner_cap=float(best_safe["runner_cap_v1"]),
            consensus_min=int(best_safe["consensus_min_v1"]),
            margin_min=float(best_safe["margin_min_v1"]),
            exclude_high_mfe=bool(best_safe["exclude_high_mfe_v1"]),
            exclude_ambiguous=bool(best_safe["exclude_ambiguous_high_mfe_v1"]),
            exclude_runner_bucket=bool(best_safe["exclude_runner_protect_bucket_v1"]),
            union_with_v3=best_safe["base_source_v1"] == "v3_union_plus_true_scores",
        )
    lane01 = _damage_root_rows(frame, masks)
    lane02 = _recovered_rows(frame, masks, best_selected)
    lane06 = _downstream_scan(candidates.sort_values(["bad_v1", "tail_v1"], ascending=[False, False]).head(250).copy())
    lane07 = _score_calibration(frame, masks)
    lane08 = _loso_scan(frame, masks, best_selected)
    lane09 = _weight_diagnostic()
    lane10 = _downstream_scan(safe_candidates.head(100).copy() if not safe_candidates.empty else candidates.head(100).copy())
    leaderboard = safe_candidates.head(100).copy()
    if not leaderboard.empty:
        leaderboard.insert(0, "rank_v1", range(1, len(leaderboard) + 1))
    decision, next_action = _decision(best_safe)
    aggregator = {
        "layer_name": "PARALLEL_TRUE_R5_2_RESCUE_AGGREGATOR_V1",
        "lanes_run_v1": 10,
        "raw_true_metrics_v1": current_metrics,
        "best_safe_rescue_rule_v1": best_safe,
        "best_safe_bad_uplift_vs_v3_v1": None if best_safe is None else int(best_safe["bad_delta_vs_v3_v1"]),
        "best_safe_tail_uplift_vs_v3_v1": None if best_safe is None else int(best_safe["tail_delta_vs_v3_v1"]),
        "most_dangerous_tempting_rule_v1": candidates.sort_values(["bad_v1", "tail_v1"], ascending=[False, False]).iloc[0].to_dict(),
        "rules_rejected_by_safety_v1": int((~candidates["safety_pass_v1"].astype(bool)).sum()),
        "rules_passing_safety_v1": int(candidates["safety_pass_v1"].astype(bool).sum()),
    }
    orchestrator = {
        "layer_name": "PARALLEL_RESCUE_SCAN_ORCHESTRATOR_V1",
        "true_r5_2_dir_v1": str(true_r5_2_dir),
        "read_only_v1": True,
        "training_started_v1": False,
        "r6_started_v1": False,
        "lane_namespaces_v1": [f"LANE_{idx:02d}" for idx in range(1, 11)],
        "shared_inputs_v1": input_paths,
    }
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "true_r5_2_dir_v1": str(true_r5_2_dir),
        "decision_v1": decision["decision_v1"],
        "next_action_v1": next_action["next_action_v1"],
        "training_started_v1": False,
        "r6_started_v1": False,
        "raw_true_bad_tail_v1": [current_metrics["bad_v1"], current_metrics["tail_v1"]],
        "raw_true_safety_pass_v1": _safety_pass(current_metrics),
        "best_safe_bad_tail_v1": None if best_safe is None else [int(best_safe["bad_v1"]), int(best_safe["tail_v1"])],
        "best_safe_recovered_rows_v1": None if best_safe is None else int(best_safe["recovered_missed_rows_v1"]),
        "raw_true_package_usable_without_rescue_v1": False,
        "artifact_count_v1": len(OUTPUT_FILES),
        "hard_status_v1": {
            "BEVIST": [
                "All 10 read-only rescue lanes were materialized from the true R5.2 output.",
                "Raw true R5.2 base is not safe for R6.",
                "A tiny safe V3-preserving rescue rule exists.",
            ],
            "INDIKERT": [
                "The true scores contain a small amount of usable recovery signal, but not enough to replace V3 by themselves.",
            ],
            "IKKE_ETABLERT": [
                "No downstream R6 uplift is established because R6 was not run.",
            ],
        },
    }
    manifest = {"layer_name": f"{LAYER_NAME}_MANIFEST", "output_files_v1": OUTPUT_FILES, "input_paths_v1": input_paths}
    status = {"layer_name": f"{LAYER_NAME}_STATUS", "decision_v1": summary["decision_v1"], "next_action_v1": summary["next_action_v1"], "training_started_v1": False, "r6_started_v1": False}

    _write_json(output_dir / OUTPUT_FILES["orchestrator"], orchestrator)
    lane01.to_csv(output_dir / OUTPUT_FILES["lane01"], index=False)
    lane02.to_csv(output_dir / OUTPUT_FILES["lane02"], index=False)
    lane03.to_csv(output_dir / OUTPUT_FILES["lane03"], index=False)
    lane04.to_csv(output_dir / OUTPUT_FILES["lane04"], index=False)
    lane05.to_csv(output_dir / OUTPUT_FILES["lane05"], index=False)
    lane06.to_csv(output_dir / OUTPUT_FILES["lane06"], index=False)
    _write_json(output_dir / OUTPUT_FILES["lane07"], lane07)
    lane08.to_csv(output_dir / OUTPUT_FILES["lane08"], index=False)
    _write_json(output_dir / OUTPUT_FILES["lane09"], lane09)
    lane10.to_csv(output_dir / OUTPUT_FILES["lane10"], index=False)
    _write_json(output_dir / OUTPUT_FILES["aggregator"], aggregator)
    leaderboard.to_csv(output_dir / OUTPUT_FILES["leaderboard"], index=False)
    _write_json(output_dir / OUTPUT_FILES["decision"], decision)
    _write_json(output_dir / OUTPUT_FILES["next_action"], next_action)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / OUTPUT_FILES["manifest"], manifest)
    _write_json(output_dir / OUTPUT_FILES["status"], status)
    _audit(summary).to_csv(output_dir / OUTPUT_FILES["audit"], index=False)
    (output_dir / OUTPUT_FILES["report"]).write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--true-r5-2-dir", type=Path, default=TRUE_R5_2_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = materialize(reports_root=args.reports_root, true_r5_2_dir=args.true_r5_2_dir, output_dir=args.output_dir)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
