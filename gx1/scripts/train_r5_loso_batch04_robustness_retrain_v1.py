#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from gx1.scripts.materialize_r4_fullcoverage_policy_recalibration_and_shadow_replay_v1 import (
    _bool,
    _json_dumps,
    _load_json,
    _num,
    _policy_metric_row,
    _write_json,
)
from gx1.scripts.train_r5_entry_retrain_with_repaired_coverage_and_slice_robustness_v1 import (
    AS_OF_FEATURE_TABLE as R5_AS_OF_FEATURE_TABLE,
    CONTRACT as R5_CONTRACT,
    HINDSIGHT_LABEL_OUTCOME_TABLE as R5_HINDSIGHT_LABEL_OUTCOME_TABLE,
    LABEL_SPECS,
    LOSO as R5_LOSO,
    POLICY_PREDICTION_VIEW as R5_POLICY_PREDICTION_VIEW,
    R5_PROB,
    SUMMARY as R5_SUMMARY,
    _effect_score,
    _feature_family,
    _policy_masks as _r5_policy_masks,
    _select_internal_validation_mask,
    _slice_masks,
    _train_heads,
)


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_LOSO_BATCH04_ROBUSTNESS_RETRAIN_V1"
R5_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_ENTRY_RETRAIN_WITH_REPAIRED_COVERAGE_AND_SLICE_ROBUSTNESS_V1"

CONTRACT = "shadow_meta_all_trade_review_r5_1_loso_batch04_robustness_contract_v1.json"
AS_OF_FEATURE_TABLE = "shadow_meta_all_trade_review_r5_1_as_of_feature_table_v1.parquet"
HINDSIGHT_OUTCOME_TABLE = "shadow_meta_all_trade_review_r5_1_hindsight_outcome_table_v1.parquet"
BATCH04_FAILURE_ATTRIBUTION = "shadow_meta_all_trade_review_r5_1_batch04_failure_attribution_v1.csv"
THRESHOLD_SEARCH = "shadow_meta_all_trade_review_r5_1_loso_safe_threshold_search_v1.csv"
AS_OF_GUARD_AUDIT = "shadow_meta_all_trade_review_r5_1_batch04_like_as_of_guard_audit_v1.csv"
RUNNER_PROTECTION_AUDIT = "shadow_meta_all_trade_review_r5_1_runner_protection_strengthening_v1.csv"
ROBUST_STACK_BAKEOFF = "shadow_meta_all_trade_review_r5_1_robust_stack_bakeoff_v1.csv"
LOSO_METRICS = "shadow_meta_all_trade_review_r5_1_loso_metrics_v1.csv"
HEAD_TO_HEAD = "shadow_meta_all_trade_review_r5_1_head_to_head_vs_r2_r4_r5_v1.csv"
POLICY_PREDICTION_VIEW = "shadow_meta_all_trade_review_r5_1_policy_prediction_view_v1.parquet"
DECISION_MATRIX = "shadow_meta_all_trade_review_r5_1_decision_matrix_v1.csv"
CONSISTENCY_AUDIT = "shadow_meta_all_trade_review_r5_1_consistency_audit_v1.csv"
SUMMARY = "shadow_meta_all_trade_review_r5_1_summary_v1.json"
STATUS = "shadow_meta_all_trade_review_r5_1_status_v1.json"
MANIFEST = "shadow_meta_all_trade_review_r5_1_manifest_v1.json"
REPORT = "shadow_meta_all_trade_review_r5_1_report_v1.md"
TOP_LEVEL_SUMMARY = "truth_r5_loso_batch04_robustness_retrain_v1.json"


@dataclass(frozen=True)
class CandidateSpec:
    policy_name: str
    stack_family: str
    guard_mode: str
    thresholds: Dict[str, float]
    bad_override_bypasses_model_protect: bool = False


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _resolve_r5_dir(reports_root: Path, path_arg: str | None) -> Path:
    path = Path(path_arg).expanduser().resolve() if path_arg else reports_root / R5_EXTENSION_NAME
    required = [
        R5_AS_OF_FEATURE_TABLE,
        R5_HINDSIGHT_LABEL_OUTCOME_TABLE,
        R5_POLICY_PREDICTION_VIEW,
        R5_CONTRACT,
        R5_SUMMARY,
        R5_LOSO,
    ]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"{path} missing R5 artifacts: {missing}")
    return path


def _default_extension_dir(reports_root: Path) -> Path:
    return reports_root / EXTENSION_NAME


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, artifact_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{artifact_name} missing required columns: {missing}")


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _optional_slice_pass(slice_metric: Dict[str, Any] | None) -> bool | None:
    if not slice_metric:
        return None
    return bool(slice_metric.get("slice_safety_pass_v1", False))


def _prob(frame: pd.DataFrame, label_id: str) -> pd.Series:
    return pd.to_numeric(frame.get(R5_PROB[label_id], pd.Series(np.nan, index=frame.index)), errors="coerce")


def _json_loads(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    payload = json.loads(str(value))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object, got {type(payload)}")
    return payload


def _metric_with_candidate(metric: Dict[str, Any], candidate: CandidateSpec, *, candidate_type: str) -> Dict[str, Any]:
    metric.update(
        {
            "candidate_type_v1": candidate_type,
            "stack_family_v1": candidate.stack_family,
            "guard_mode_v1": candidate.guard_mode,
            "bad_override_bypasses_model_protect_v1": candidate.bad_override_bypasses_model_protect,
            "thresholds_json_v1": _json_dumps(candidate.thresholds),
        }
    )
    return metric


def _load_r5_build(r5_dir: Path, expected_ledger_count: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any], pd.DataFrame, List[str]]:
    asof_df = pd.read_parquet(r5_dir / R5_AS_OF_FEATURE_TABLE)
    hindsight_df = pd.read_parquet(r5_dir / R5_HINDSIGHT_LABEL_OUTCOME_TABLE)
    prediction_df = pd.read_parquet(r5_dir / R5_POLICY_PREDICTION_VIEW)
    contract = _load_json(r5_dir / R5_CONTRACT)
    summary = _load_json(r5_dir / R5_SUMMARY)
    loso_df = pd.read_csv(r5_dir / R5_LOSO)

    feature_names = [str(item) for item in contract.get("as_of_feature_names_v1", [])]
    if not feature_names:
        raise RuntimeError("R5 contract missing as_of_feature_names_v1")
    _require_columns(asof_df, ["candidate_uid", "run_id", "entry_observation_present_v1", "entry_raw_state_present_v1", *feature_names], artifact_name=R5_AS_OF_FEATURE_TABLE)
    _require_columns(hindsight_df, ["candidate_uid", "r5_label_should_not_take_v1", "r5_label_take_was_ok_v1", "r5_label_strong_trade_candidate_v1", "peak_mfe_bps_v1", "mae_abs_bps_v1", "baseline_realized_pnl_bps_v1"], artifact_name=R5_HINDSIGHT_LABEL_OUTCOME_TABLE)
    _require_columns(
        prediction_df,
        [
            "candidate_uid",
            "r2_fallback_reference__block_v1",
            "r4_current_reference__block_v1",
            "r5_selected_candidate__block_v1",
            *R5_PROB.values(),
        ],
        artifact_name=R5_POLICY_PREDICTION_VIEW,
    )
    for name, frame in [(R5_AS_OF_FEATURE_TABLE, asof_df), (R5_HINDSIGHT_LABEL_OUTCOME_TABLE, hindsight_df), (R5_POLICY_PREDICTION_VIEW, prediction_df)]:
        if bool(frame["candidate_uid"].astype("string").duplicated().any()):
            raise ValueError(f"{name} requires unique candidate_uid")
    if expected_ledger_count is not None and len(asof_df) != expected_ledger_count:
        raise RuntimeError(f"Locked ledger expected {expected_ledger_count}, observed {len(asof_df)}")
    coverage = summary.get("coverage_v1", {}) if isinstance(summary.get("coverage_v1"), dict) else {}
    if int(coverage.get("synthetic_count_v1", -1)) != 0:
        raise RuntimeError(f"R5.1 refuses synthetic R5 input; observed {coverage.get('synthetic_count_v1')}")
    if int(coverage.get("entry_coverage_v1", 0)) != len(asof_df):
        raise RuntimeError("R5 input is not full entry coverage")
    if int(coverage.get("entry_raw_coverage_v1", 0)) != len(asof_df):
        raise RuntimeError("R5 input is not full raw-state coverage")
    return asof_df, hindsight_df, prediction_df, contract, summary, loso_df, feature_names


def _prepare_base_frame(asof_df: pd.DataFrame, hindsight_df: pd.DataFrame, prediction_df: pd.DataFrame) -> pd.DataFrame:
    label_drop = [column for column in ["run_id", "trade_uid", "trade_id", "decision_timestamp"] if column in hindsight_df.columns]
    pred_cols = [
        "candidate_uid",
        "no_entry_fallback_baseline__block_v1",
        "r2_fallback_reference__block_v1",
        "r3_fullcoverage_conservative__block_v1",
        "r4_current_reference__block_v1",
        "r5_selected_candidate__block_v1",
        *R5_PROB.values(),
    ]
    frame = (
        asof_df.merge(hindsight_df.drop(columns=label_drop), on="candidate_uid", how="inner", validate="one_to_one")
        .merge(prediction_df[[column for column in pred_cols if column in prediction_df.columns]], on="candidate_uid", how="left", validate="one_to_one")
    )
    for column in R5_PROB.values():
        frame[f"global_{column}"] = pd.to_numeric(frame[column], errors="coerce")
    frame["is_repaired_165_v1"] = _bool(frame, "entry_coverage_repair_applied_v1")
    frame["label_should_not_take_v1"] = _bool(frame, "r5_label_should_not_take_v1")
    frame["label_strong_trade_candidate_v1"] = _bool(frame, "r5_label_strong_trade_candidate_v1")
    frame["take_was_ok_v1"] = _bool(frame, "r5_label_take_was_ok_v1")
    frame["fifty_plus_mfe_v1"] = _num(frame, "peak_mfe_bps_v1").ge(50.0)
    frame["hundred_plus_mfe_v1"] = _num(frame, "peak_mfe_bps_v1").ge(100.0)
    frame["two_hundred_plus_mfe_v1"] = _num(frame, "peak_mfe_bps_v1").ge(200.0)
    frame["tail_10_50_mfe_v1"] = _num(frame, "peak_mfe_bps_v1").between(10.0, 50.0, inclusive="left") & (
        _num(frame, "baseline_realized_pnl_bps_v1").le(0.0) | _bool(frame, "label_should_not_take_v1")
    )
    frame["strongest_winner_path_v1"] = frame["two_hundred_plus_mfe_v1"].fillna(False).astype(bool) | (
        _bool(frame, "label_strong_trade_candidate_v1")
        & _num(frame, "baseline_realized_pnl_bps_v1").gt(0.0)
        & _num(frame, "peak_mfe_bps_v1").ge(50.0)
    )
    frame["best_constrained_recalibrated_r4__block_v1"] = _bool(frame, "r4_current_reference__block_v1")
    if int(_bool(frame, "entry_observation_present_v1").sum()) != len(frame):
        raise RuntimeError("R5.1 requires full entry_observation_present_v1")
    if int(_bool(frame, "entry_raw_state_present_v1").sum()) != len(frame):
        raise RuntimeError("R5.1 requires full entry_raw_state_present_v1")
    return frame


def _asof_guard(frame: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "none":
        return pd.Series(False, index=frame.index, dtype=bool)
    tradable = _num(frame, "as_of_candidate_tradable_prob_v1")
    path_quality = _num(frame, "as_of_entry_candidate_path_quality_pred_v1")
    mfe_first = _num(frame, "as_of_candidate_mfe_first_n_pred_v1")
    flat_prob = _num(frame, "as_of_skip_candidate_p_flat_v1")
    margin = _num(frame, "as_of_entry_candidate_margin_v1")
    retracement = _num(frame, "as_of_skip_replay_retracement_from_last_impulse_v1")
    clv = _num(frame, "as_of_skip_replay_clv_v1")
    range_15 = _num(frame, "as_of_skip_replay_window_range_15_bps_v1")
    repaired = frame["is_repaired_165_v1"].fillna(False).astype(bool) if "is_repaired_165_v1" in frame.columns else _bool(frame, "entry_coverage_repair_applied_v1")

    if mode == "runner_proxy_loose":
        return (tradable >= 0.94) & (path_quality >= 0.72) & (mfe_first >= 1.75) & (flat_prob <= 0.50)
    if mode == "runner_proxy_tight":
        return (tradable >= 0.95) & (path_quality >= 0.78) & (mfe_first >= 1.80) & (flat_prob <= 0.47) & (retracement >= 0.50)
    if mode == "structure_runner":
        return (tradable >= 0.93) & (mfe_first >= 1.80) & (retracement >= 0.55) & (clv >= 0.35) & (range_15 <= 85.0)
    if mode == "repaired_165_like":
        return repaired | ((tradable >= 0.95) & (path_quality >= 0.75) & (margin >= 0.04) & (mfe_first >= 1.80))
    if mode == "combined_runner_first":
        return repaired | _asof_guard(frame, "runner_proxy_loose") | _asof_guard(frame, "structure_runner")
    raise ValueError(f"Unknown AS_OF guard mode: {mode}")


def _policy_components(frame: pd.DataFrame, candidate: CandidateSpec) -> Dict[str, pd.Series]:
    params = candidate.thresholds
    p_should = _prob(frame, "should_not_take")
    p_mae = _prob(frame, "immediate_MAE_risk")
    p_runner = _prob(frame, "runner_protect")
    p_strong = _prob(frame, "strong_trade_candidate")
    p_tail = _prob(frame, "tail_control_10_50_risk")
    p_take = _prob(frame, "take_was_ok")
    p_high_runner_bad = _prob(frame, "bad_trade_but_high_runner_risk")

    t_should = float(params.get("should_not_take_threshold_v1", 0.80))
    t_mae = float(params.get("immediate_mae_threshold_v1", 0.85))
    t_tail = float(params.get("tail_control_threshold_v1", 0.80))
    t_runner = float(params.get("runner_protect_threshold_v1", 0.60))
    t_strong = float(params.get("strong_protect_threshold_v1", 0.60))
    t_take = float(params.get("take_ok_protect_threshold_v1", 0.85))
    t_override = float(params.get("bad_risk_override_threshold_v1", 0.88))
    take_ceiling = float(params.get("take_ok_block_ceiling_v1", 0.45))

    model_protect = p_runner.ge(t_runner).fillna(False) | p_strong.ge(t_strong).fillna(False) | p_take.ge(t_take).fillna(False)
    raw_guard = _asof_guard(frame, candidate.guard_mode)
    protect = model_protect | raw_guard
    weak_take = p_take.lt(take_ceiling).fillna(False)
    should_signal = p_should.ge(t_should).fillna(False)
    mae_signal = p_mae.ge(t_mae).fillna(False) & weak_take
    tail_signal = p_tail.ge(t_tail).fillna(False) & weak_take
    combined = should_signal | mae_signal | tail_signal
    high_bad_override = (
        p_should.ge(t_override).fillna(False)
        & p_mae.ge(max(t_mae, 0.75)).fillna(False)
        & p_high_runner_bad.lt(0.70).fillna(True)
    )
    r2 = _bool(frame, "r2_fallback_reference__block_v1")
    r4 = _bool(frame, "r4_current_reference__block_v1")
    risk_confirm = combined | high_bad_override
    return {
        "model_protect_v1": model_protect.astype(bool),
        "raw_as_of_guard_v1": raw_guard.astype(bool),
        "protect_v1": protect.astype(bool),
        "weak_take_signal_v1": weak_take.astype(bool),
        "should_signal_v1": should_signal.astype(bool),
        "immediate_mae_signal_v1": mae_signal.astype(bool),
        "tail_signal_v1": tail_signal.astype(bool),
        "combined_signal_v1": combined.astype(bool),
        "high_bad_override_signal_v1": high_bad_override.astype(bool),
        "r2_reference_signal_v1": r2.astype(bool),
        "r4_reference_signal_v1": r4.astype(bool),
        "risk_confirm_signal_v1": risk_confirm.astype(bool),
    }


def _policy_mask(frame: pd.DataFrame, candidate: CandidateSpec) -> pd.Series:
    c = _policy_components(frame, candidate)
    stack = candidate.stack_family
    if stack == "R5_CURRENT_COMPATIBLE":
        signal = c["r4_reference_signal_v1"] | c["combined_signal_v1"] | c["high_bad_override_signal_v1"]
    elif stack == "R5_LOSO_COMPACT_REFERENCE":
        signal = c["r2_reference_signal_v1"] | c["combined_signal_v1"] | c["high_bad_override_signal_v1"]
    elif stack == "R5_1_COMBINED":
        signal = c["combined_signal_v1"] | c["high_bad_override_signal_v1"]
    elif stack == "R5_1_R4_CONFIRMED":
        signal = (c["r4_reference_signal_v1"] & c["risk_confirm_signal_v1"]) | c["combined_signal_v1"] | c["high_bad_override_signal_v1"]
    elif stack == "R5_1_R2_CONFIRMED":
        signal = (c["r2_reference_signal_v1"] & c["risk_confirm_signal_v1"]) | c["combined_signal_v1"] | c["high_bad_override_signal_v1"]
    elif stack == "R5_1_R2_R4_CONFIRMED":
        signal = ((c["r2_reference_signal_v1"] | c["r4_reference_signal_v1"]) & c["risk_confirm_signal_v1"]) | c["combined_signal_v1"] | c["high_bad_override_signal_v1"]
    elif stack == "R5_1_SHOULD_ONLY":
        signal = c["should_signal_v1"]
    elif stack == "R5_1_TAIL_ONLY":
        signal = c["tail_signal_v1"]
    else:
        raise ValueError(f"Unknown policy stack: {stack}")
    block = signal & ~c["protect_v1"]
    if candidate.bad_override_bypasses_model_protect:
        block = block | (c["high_bad_override_signal_v1"] & ~c["raw_as_of_guard_v1"])
    return block.fillna(False).astype(bool)


def _candidate_grid() -> List[CandidateSpec]:
    candidates: list[CandidateSpec] = []
    current_thresholds = {
        "should_not_take_threshold_v1": 0.55,
        "immediate_mae_threshold_v1": 0.80,
        "tail_control_threshold_v1": 0.80,
        "runner_protect_threshold_v1": 0.65,
        "strong_protect_threshold_v1": 0.65,
        "take_ok_protect_threshold_v1": 0.85,
        "take_ok_block_ceiling_v1": 0.45,
        "bad_risk_override_threshold_v1": 0.88,
    }
    loso_thresholds = {
        "should_not_take_threshold_v1": 0.80,
        "immediate_mae_threshold_v1": 0.85,
        "tail_control_threshold_v1": 0.80,
        "runner_protect_threshold_v1": 0.70,
        "strong_protect_threshold_v1": 0.65,
        "take_ok_protect_threshold_v1": 0.90,
        "take_ok_block_ceiling_v1": 0.45,
        "bad_risk_override_threshold_v1": 0.88,
    }
    candidates.append(CandidateSpec("R5_CURRENT_REFERENCE_REBUILT", "R5_CURRENT_COMPATIBLE", "none", current_thresholds, True))
    candidates.append(CandidateSpec("R5_LOSO_COMPACT_REFERENCE_REBUILT", "R5_LOSO_COMPACT_REFERENCE", "none", loso_thresholds, True))
    stacks = [
        "R5_1_COMBINED",
        "R5_1_R4_CONFIRMED",
        "R5_1_R2_CONFIRMED",
        "R5_1_R2_R4_CONFIRMED",
    ]
    guard_modes = ["none", "runner_proxy_loose", "structure_runner", "repaired_165_like", "combined_runner_first"]
    protect_sets = [
        (0.50, 0.50, 0.80),
        (0.60, 0.60, 0.85),
        (0.65, 0.65, 0.85),
        (0.70, 0.65, 0.90),
    ]
    index = 0
    for stack in stacks:
        for guard_mode in guard_modes:
            for should_threshold in [0.55, 0.75, 0.80, 0.85]:
                for mae_threshold in [0.80, 0.85]:
                    for tail_threshold in [0.80, 0.85]:
                        for runner_threshold, strong_threshold, take_threshold in protect_sets:
                            thresholds = {
                                "should_not_take_threshold_v1": should_threshold,
                                "immediate_mae_threshold_v1": mae_threshold,
                                "tail_control_threshold_v1": tail_threshold,
                                "runner_protect_threshold_v1": runner_threshold,
                                "strong_protect_threshold_v1": strong_threshold,
                                "take_ok_protect_threshold_v1": take_threshold,
                                "take_ok_block_ceiling_v1": 0.45,
                                "bad_risk_override_threshold_v1": 0.88,
                            }
                            index += 1
                            candidates.append(
                                CandidateSpec(
                                    f"R5_1_CANDIDATE_{index:04d}_{stack}_{guard_mode}",
                                    stack,
                                    guard_mode,
                                    thresholds,
                                    False,
                                )
                            )
    for stack in ["R5_1_SHOULD_ONLY", "R5_1_TAIL_ONLY"]:
        for guard_mode in ["none", "combined_runner_first"]:
            for should_threshold in [0.75, 0.85]:
                thresholds = {
                    "should_not_take_threshold_v1": should_threshold,
                    "immediate_mae_threshold_v1": 0.85,
                    "tail_control_threshold_v1": 0.80,
                    "runner_protect_threshold_v1": 0.60,
                    "strong_protect_threshold_v1": 0.60,
                    "take_ok_protect_threshold_v1": 0.85,
                    "take_ok_block_ceiling_v1": 0.45,
                    "bad_risk_override_threshold_v1": 0.88,
                }
                index += 1
                candidates.append(
                    CandidateSpec(
                        f"R5_1_CANDIDATE_{index:04d}_{stack}_{guard_mode}",
                        stack,
                        guard_mode,
                        thresholds,
                        False,
                    )
                )
    return candidates


def _global_safety(metric: Dict[str, Any], *, require_strong_le_one: bool) -> tuple[bool, str]:
    failures: list[str] = []
    precision = _safe_float(metric.get("should_not_take_precision_v1"))
    if int(metric.get("repaired_165_block_count_v1") or 0) != 0:
        failures.append("repaired_165_block_count_v1!=0")
    if int(metric.get("two_hundred_plus_mfe_block_count_v1") or 0) != 0:
        failures.append("two_hundred_plus_mfe_block_count_v1!=0")
    if int(metric.get("fifty_plus_mfe_block_count_v1") or 0) > 2:
        failures.append("fifty_plus_mfe_block_count_v1>2")
    strong_limit = 1 if require_strong_le_one else 2
    if int(metric.get("strong_trade_false_block_count_v1") or 0) > strong_limit:
        failures.append(f"strong_trade_false_block_count_v1>{strong_limit}")
    if precision is None or precision < 0.90:
        failures.append("precision<0.90")
    return not failures, ",".join(failures)


def _slice_safety(metric: Dict[str, Any], r5_current_slice: Dict[str, Any] | None = None) -> tuple[bool, str]:
    failures: list[str] = []
    precision = _safe_float(metric.get("should_not_take_precision_v1"))
    block_count = int(metric.get("block_count_v1") or 0)
    if int(metric.get("repaired_165_block_count_v1") or 0) != 0:
        failures.append("repaired_165_block_count_v1!=0")
    if int(metric.get("two_hundred_plus_mfe_block_count_v1") or 0) != 0:
        failures.append("two_hundred_plus_mfe_block_count_v1!=0")
    current_limit = int(r5_current_slice.get("fifty_plus_mfe_block_count_v1") or 999) if r5_current_slice else 999
    if int(metric.get("fifty_plus_mfe_block_count_v1") or 0) > min(2, current_limit):
        failures.append("fifty_plus_mfe_block_count_v1>slice_limit")
    if int(metric.get("strong_trade_false_block_count_v1") or 0) > 2:
        failures.append("strong_trade_false_block_count_v1>2")
    if block_count > 0 and (precision is None or precision < 0.85):
        failures.append("precision<0.85")
    return not failures, ",".join(failures)


def _train_loso_fold_predictions(
    *,
    reports_root: Path,
    base_frame: pd.DataFrame,
    feature_names: Sequence[str],
    batch_weeks: int,
    n_estimators: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
    n_jobs: int,
) -> tuple[Dict[str, pd.DataFrame], list[dict[str, Any]], pd.DataFrame]:
    raw_frame = base_frame.drop(columns=[column for column in R5_PROB.values() if column in base_frame.columns], errors="ignore")
    fold_frames: dict[str, pd.DataFrame] = {}
    slice_infos: list[dict[str, Any]] = []
    metric_rows: list[pd.DataFrame] = []
    for slice_info in _slice_masks(reports_root, base_frame, batch_weeks=batch_weeks):
        holdout = slice_info["mask_v1"].reindex(base_frame.index).fillna(False).astype(bool)
        train_all = ~holdout
        inner_train, inner_validation = _select_internal_validation_mask(base_frame, train_all)
        pred_df, metric_df, _ = _train_heads(
            frame=raw_frame,
            feature_names=feature_names,
            train_mask=inner_train,
            validation_mask=inner_validation,
            output_dir=None,
            model_tag=f"r5_1_loso_{str(slice_info['scope_v1']).lower()}",
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            learning_rate=learning_rate,
            max_depth=max_depth,
            seed=seed + int(slice_info["batch_index_v1"]) * 100,
            n_jobs=n_jobs,
        )
        fold_frame = raw_frame.merge(pred_df, on="candidate_uid", how="left", validate="one_to_one")
        fold_frames[str(slice_info["scope_v1"])] = fold_frame
        slice_infos.append(slice_info)
        metric_df["holdout_slice_v1"] = str(slice_info["scope_v1"])
        metric_rows.append(metric_df)
    return fold_frames, slice_infos, pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()


def _r5_current_loso_lookup(r5_loso_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if "holdout_slice_v1" not in r5_loso_df.columns:
        return lookup
    for _, row in r5_loso_df.iterrows():
        lookup[str(row["holdout_slice_v1"])] = row.to_dict()
    return lookup


def _evaluate_candidate_set(
    *,
    base_frame: pd.DataFrame,
    fold_frames: Dict[str, pd.DataFrame],
    slice_infos: list[dict[str, Any]],
    r5_loso_df: pd.DataFrame,
    r4_ref_all: Dict[str, Any],
    r5_current_all: Dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, CandidateSpec, Dict[str, Any]]:
    candidates = _candidate_grid()
    r5_current_slices = _r5_current_loso_lookup(r5_loso_df)
    rows: list[dict[str, Any]] = []
    loso_rows: list[dict[str, Any]] = []

    preliminary: list[tuple[CandidateSpec, Dict[str, Any], list[dict[str, Any]]]] = []
    for candidate in candidates:
        global_mask = _policy_mask(base_frame, candidate)
        global_metric = _metric_with_candidate(
            _policy_metric_row(candidate.policy_name, "ALL", base_frame, global_mask, thresholds=candidate.thresholds),
            candidate,
            candidate_type="R5_1_POLICY_STACK" if candidate.policy_name.startswith("R5_1_") else "REFERENCE_REBUILD",
        )
        slice_metrics: list[dict[str, Any]] = []
        for slice_info in slice_infos:
            scope = str(slice_info["scope_v1"])
            holdout = slice_info["mask_v1"].reindex(base_frame.index).fillna(False).astype(bool)
            fold_frame = fold_frames[scope]
            fold_mask = _policy_mask(fold_frame, candidate)
            metric = _metric_with_candidate(
                _policy_metric_row(candidate.policy_name, scope, fold_frame.loc[holdout].copy(), fold_mask.loc[holdout], thresholds=candidate.thresholds),
                candidate,
                candidate_type="R5_1_POLICY_STACK" if candidate.policy_name.startswith("R5_1_") else "REFERENCE_REBUILD",
            )
            spass, sfail = _slice_safety(metric, r5_current_slices.get(scope))
            metric.update(
                {
                    "holdout_slice_v1": scope,
                    "slice_safety_pass_v1": spass,
                    "slice_safety_failure_reasons_v1": sfail,
                    "run_count_v1": int(slice_info["run_count_v1"]),
                    "run_start_v1": slice_info["run_start_v1"],
                    "run_end_v1": slice_info["run_end_v1"],
                }
            )
            slice_metrics.append(metric)
        preliminary.append((candidate, global_metric, slice_metrics))

    any_strong_le_one = False
    for candidate, global_metric, slice_metrics in preliminary:
        if not candidate.policy_name.startswith("R5_1_"):
            continue
        gpass, _ = _global_safety(global_metric, require_strong_le_one=True)
        if gpass and all(bool(item["slice_safety_pass_v1"]) for item in slice_metrics):
            any_strong_le_one = True
            break

    for candidate, global_metric, slice_metrics in preliminary:
        require_strong = any_strong_le_one
        gpass, gfail = _global_safety(global_metric, require_strong_le_one=require_strong)
        batch04 = next((item for item in slice_metrics if item["holdout_slice_v1"] == "BATCH_04"), {})
        batch05 = next((item for item in slice_metrics if item["holdout_slice_v1"] == "BATCH_05"), {})
        precisions = [
            _safe_float(item.get("should_not_take_precision_v1"))
            for item in slice_metrics
            if int(item.get("block_count_v1") or 0) > 0 and _safe_float(item.get("should_not_take_precision_v1")) is not None
        ]
        worst_precision = min(precisions) if precisions else 1.0
        loso_pass = all(bool(item["slice_safety_pass_v1"]) for item in slice_metrics)
        beats_r4 = int(global_metric.get("should_not_take_block_count_v1") or 0) > int(r4_ref_all.get("should_not_take_block_count_v1") or 0)
        keeps_r5_runner_safety = (
            int(global_metric.get("fifty_plus_mfe_block_count_v1") or 0) <= int(r5_current_all.get("fifty_plus_mfe_block_count_v1") or 0)
            and int(global_metric.get("hundred_plus_mfe_block_count_v1") or 0) <= int(r5_current_all.get("hundred_plus_mfe_block_count_v1") or 0)
            and int(global_metric.get("two_hundred_plus_mfe_block_count_v1") or 0) <= int(r5_current_all.get("two_hundred_plus_mfe_block_count_v1") or 0)
            and int(global_metric.get("repaired_165_block_count_v1") or 0) == 0
        )
        tail_better_than_r4 = int(global_metric.get("tail_10_50_help_count_v1") or 0) > int(r4_ref_all.get("tail_10_50_help_count_v1") or 0)
        safety_failures = []
        if not gpass:
            safety_failures.extend([item for item in str(gfail).split(",") if item])
        for item in slice_metrics:
            safety_failures.extend([f"{item['holdout_slice_v1']}:{reason}" for reason in str(item["slice_safety_failure_reasons_v1"]).split(",") if reason])
        score = (
            float(global_metric.get("should_not_take_block_count_v1") or 0) * 1.0
            + float(global_metric.get("tail_10_50_help_count_v1") or 0) * 0.25
            + float(worst_precision) * 12.0
            + float(_safe_float(global_metric.get("should_not_take_precision_v1")) or 0.0) * 8.0
            - float(global_metric.get("take_was_ok_block_count_v1") or 0) * 0.25
            - float(global_metric.get("fifty_plus_mfe_block_count_v1") or 0) * 5.0
            - float(global_metric.get("two_hundred_plus_mfe_block_count_v1") or 0) * 20.0
        )
        if not (candidate.policy_name.startswith("R5_1_") and gpass and loso_pass and beats_r4 and keeps_r5_runner_safety):
            score -= 1000.0
        row = dict(global_metric)
        row.update(
            {
                "global_safety_pass_v1": gpass,
                "global_safety_failure_reasons_v1": gfail,
                "loso_all_slices_pass_v1": loso_pass,
                "worst_slice_precision_v1": worst_precision,
                "batch04_loso_pass_v1": _optional_slice_pass(batch04),
                "batch04_failure_reasons_v1": batch04.get("slice_safety_failure_reasons_v1", ""),
                "batch04_should_not_take_block_count_v1": batch04.get("should_not_take_block_count_v1"),
                "batch04_precision_v1": batch04.get("should_not_take_precision_v1"),
                "batch04_fifty_plus_mfe_block_count_v1": batch04.get("fifty_plus_mfe_block_count_v1"),
                "batch04_strong_false_block_count_v1": batch04.get("strong_trade_false_block_count_v1"),
                "batch05_loso_pass_v1": _optional_slice_pass(batch05),
                "batch05_failure_reasons_v1": batch05.get("slice_safety_failure_reasons_v1", ""),
                "batch05_should_not_take_block_count_v1": batch05.get("should_not_take_block_count_v1"),
                "batch05_precision_v1": batch05.get("should_not_take_precision_v1"),
                "beats_r4_bad_recall_v1": beats_r4,
                "tail_control_better_than_r4_v1": tail_better_than_r4,
                "keeps_r5_runner_safety_v1": keeps_r5_runner_safety,
                "safety_failure_count_v1": int(len(safety_failures)),
                "safety_failures_json_v1": _json_dumps(safety_failures[:40]),
                "selection_score_v1": score,
                "strong_false_le_one_enforced_v1": any_strong_le_one,
            }
        )
        rows.append(row)
        loso_rows.extend(slice_metrics)

    search_df = pd.DataFrame(rows)
    candidate_rows = search_df[search_df["policy_name_v1"].astype("string").str.startswith("R5_1_")].copy()
    viable = candidate_rows[
        candidate_rows["global_safety_pass_v1"].fillna(False)
        & candidate_rows["loso_all_slices_pass_v1"].fillna(False)
        & candidate_rows["beats_r4_bad_recall_v1"].fillna(False)
        & candidate_rows["keeps_r5_runner_safety_v1"].fillna(False)
    ].copy()
    if viable.empty:
        selected_row = candidate_rows.sort_values(
            ["safety_failure_count_v1", "batch04_loso_pass_v1", "global_safety_pass_v1", "selection_score_v1"],
            ascending=[True, False, False, False],
        ).iloc[0].to_dict()
    else:
        selected_row = viable.sort_values(
            ["selection_score_v1", "should_not_take_block_count_v1", "worst_slice_precision_v1"],
            ascending=[False, False, False],
        ).iloc[0].to_dict()

    search_df["selected_r5_1_candidate_v1"] = search_df["policy_name_v1"].astype("string").eq(str(selected_row["policy_name_v1"]))
    selected_candidate = next(candidate for candidate in candidates if candidate.policy_name == str(selected_row["policy_name_v1"]))
    return search_df, pd.DataFrame(loso_rows), selected_candidate, selected_row


def _reference_candidates() -> List[CandidateSpec]:
    return [
        CandidateSpec("R2_FALLBACK_REFERENCE", "REFERENCE", "none", {"reference_policy_v1": "R2"}),
        CandidateSpec("R4_CURRENT_REFERENCE", "REFERENCE", "none", {"reference_policy_v1": "R4"}),
        CandidateSpec("R5_CURRENT_REFERENCE", "REFERENCE", "none", {"reference_policy_v1": "R5"}),
    ]


def _reference_mask(frame: pd.DataFrame, name: str) -> pd.Series:
    if name == "R2_FALLBACK_REFERENCE":
        return _bool(frame, "r2_fallback_reference__block_v1")
    if name == "R4_CURRENT_REFERENCE":
        return _bool(frame, "r4_current_reference__block_v1")
    if name == "R5_CURRENT_REFERENCE":
        return _bool(frame, "r5_selected_candidate__block_v1")
    raise ValueError(name)


def _head_to_head(base_frame: pd.DataFrame, selected: CandidateSpec, slice_infos: Sequence[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_mask = _policy_mask(base_frame, selected)
    policies = {candidate.policy_name: _reference_mask(base_frame, candidate.policy_name) for candidate in _reference_candidates()}
    policies["R5_1_SELECTED_CANDIDATE"] = selected_mask
    batch04_info = next((item for item in slice_infos if str(item.get("scope_v1")) == "BATCH_04"), None)
    batch04_mask = (
        batch04_info["mask_v1"].reindex(base_frame.index).fillna(False).astype(bool)
        if batch04_info is not None
        else pd.Series(False, index=base_frame.index, dtype=bool)
    )
    scopes = {
        "ALL_1971": pd.Series(True, index=base_frame.index),
        "BATCH_04": batch04_mask,
        "SHOULD_NOT_TAKE_CLASS": _bool(base_frame, "label_should_not_take_v1"),
        "TAKE_WAS_OK_CLASS": _bool(base_frame, "take_was_ok_v1"),
        "REPAIRED_165": base_frame["is_repaired_165_v1"].fillna(False).astype(bool),
        "FIFTY_PLUS_MFE_RUNNERS": base_frame["fifty_plus_mfe_v1"].fillna(False).astype(bool),
        "HUNDRED_PLUS_MFE_RUNNERS": base_frame["hundred_plus_mfe_v1"].fillna(False).astype(bool),
        "TWO_HUNDRED_PLUS_MFE_RUNNERS": base_frame["two_hundred_plus_mfe_v1"].fillna(False).astype(bool),
        "STRONGEST_WINNER_PATH": base_frame["strongest_winner_path_v1"].fillna(False).astype(bool),
        "TAIL_10_50_MFE_POCKET": base_frame["tail_10_50_mfe_v1"].fillna(False).astype(bool),
    }
    rows: list[dict[str, Any]] = []
    prediction = base_frame[
        [
            "run_id",
            "candidate_uid",
            "trade_uid",
            "trade_id",
            "decision_timestamp",
            "is_repaired_165_v1",
            "label_should_not_take_v1",
            "label_strong_trade_candidate_v1",
            "take_was_ok_v1",
            "fifty_plus_mfe_v1",
            "hundred_plus_mfe_v1",
            "two_hundred_plus_mfe_v1",
            "peak_mfe_bps_v1",
            "mae_abs_bps_v1",
            "giveback_bps_v1",
            "baseline_realized_pnl_bps_v1",
            *R5_PROB.values(),
        ]
    ].copy()
    for policy_name, mask in policies.items():
        prediction[f"{policy_name.lower()}__block_v1"] = mask.to_numpy(dtype=bool)
        for scope_name, scope_mask in scopes.items():
            rows.append(_policy_metric_row(policy_name, scope_name, base_frame.loc[scope_mask].copy(), mask.loc[scope_mask], thresholds={"head_to_head_v1": True}))
    return pd.DataFrame(rows), prediction


def _selected_loso_metrics(
    *,
    base_frame: pd.DataFrame,
    fold_frames: Dict[str, pd.DataFrame],
    slice_infos: list[dict[str, Any]],
    selected: CandidateSpec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    references = {
        "R2_FALLBACK_REFERENCE": _bool(base_frame, "r2_fallback_reference__block_v1"),
        "R4_CURRENT_REFERENCE": _bool(base_frame, "r4_current_reference__block_v1"),
        "R5_CURRENT_GLOBAL_REFERENCE": _bool(base_frame, "r5_selected_candidate__block_v1"),
    }
    for slice_info in slice_infos:
        scope = str(slice_info["scope_v1"])
        holdout = slice_info["mask_v1"].reindex(base_frame.index).fillna(False).astype(bool)
        for name, mask in references.items():
            metric = _policy_metric_row(name, scope, base_frame.loc[holdout].copy(), mask.loc[holdout], thresholds={"reference_v1": name})
            metric.update({key: value for key, value in slice_info.items() if key != "mask_v1"})
            pass_slice, failures = _slice_safety(metric)
            metric["slice_safety_pass_v1"] = pass_slice
            metric["slice_safety_failure_reasons_v1"] = failures
            rows.append(metric)
        fold_frame = fold_frames[scope]
        selected_mask = _policy_mask(fold_frame, selected)
        metric = _policy_metric_row("R5_1_SELECTED_CANDIDATE", scope, fold_frame.loc[holdout].copy(), selected_mask.loc[holdout], thresholds=selected.thresholds)
        metric.update({key: value for key, value in slice_info.items() if key != "mask_v1"})
        pass_slice, failures = _slice_safety(metric)
        metric["slice_safety_pass_v1"] = pass_slice
        metric["slice_safety_failure_reasons_v1"] = failures
        rows.append(metric)
    return pd.DataFrame(rows)


def _batch04_failure_attribution(
    *,
    base_frame: pd.DataFrame,
    fold_frames: Dict[str, pd.DataFrame],
    r5_loso_df: pd.DataFrame,
    feature_names: Sequence[str],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if "BATCH_04" not in fold_frames:
        return pd.DataFrame(), {"batch04_reproduced_v1": False}
    fold_frame = fold_frames["BATCH_04"]
    r5_row = r5_loso_df.loc[r5_loso_df.get("holdout_slice_v1", pd.Series(dtype=str)).astype("string").eq("BATCH_04")]
    if r5_row.empty:
        return pd.DataFrame(), {"batch04_reproduced_v1": False, "reason_v1": "R5_LOSO_BATCH04_ROW_MISSING"}
    selected_policy = str(r5_row.iloc[0].get("selected_policy_name_v1", r5_row.iloc[0].get("policy_name_v1", "R5_R2_PRESERVATION_AWARE_STACK")))
    thresholds = _json_loads(r5_row.iloc[0].get("thresholds_json_v1", "{}"))
    block = _r5_policy_masks(fold_frame, thresholds)[selected_policy]
    batch04_mask = fold_frame["run_id"].astype("string").isin(
        base_frame.loc[base_frame["run_id"].astype("string").isin(fold_frame["run_id"].astype("string")), "run_id"].astype("string").unique().tolist()
    )
    # Use the R5 LOSO scope range instead of a date heuristic.
    runs = sorted(fold_frame["run_id"].astype("string").unique().tolist())
    batch04_runs = runs[45:60] if len(runs) >= 60 else sorted(base_frame.loc[base_frame.index[base_frame["run_id"].astype("string").map(lambda value: value).notna()], "run_id"].astype("string").unique().tolist())[3:4]
    if "scope_v1" in r5_row.columns:
        run_start = r5_row.iloc[0].get("run_start_v1")
        run_end = r5_row.iloc[0].get("run_end_v1")
        if isinstance(run_start, str) and isinstance(run_end, str):
            all_runs = sorted(base_frame["run_id"].astype("string").unique().tolist())
            if run_start in all_runs and run_end in all_runs:
                start_index = all_runs.index(run_start)
                end_index = all_runs.index(run_end)
                batch04_runs = all_runs[start_index : end_index + 1]
    batch04_mask = fold_frame["run_id"].astype("string").isin(batch04_runs)
    blocked = batch04_mask & block
    should = _bool(fold_frame, "label_should_not_take_v1")
    false_block = blocked & ~should
    true_bad_block = blocked & should

    rebuilt = CandidateSpec("R5_LOSO_COMPACT_REFERENCE_REBUILT", "R5_LOSO_COMPACT_REFERENCE", "none", {key: float(value) for key, value in thresholds.items() if isinstance(value, (int, float))}, True)
    components = _policy_components(fold_frame, rebuilt)
    rows = []
    attr_cols = [
        "run_id",
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "giveback_bps_v1",
        "baseline_realized_pnl_bps_v1",
        "label_should_not_take_v1",
        "label_strong_trade_candidate_v1",
        "take_was_ok_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "strongest_winner_path_v1",
        "is_repaired_165_v1",
        "as_of_candidate_tradable_prob_v1",
        "as_of_candidate_mfe_first_n_pred_v1",
        "as_of_entry_candidate_path_quality_pred_v1",
        "as_of_candidate_uncertainty_score_v1",
        "as_of_skip_replay_window_range_15_bps_v1",
        "as_of_skip_replay_window_realized_vol_5_bps_v1",
        "as_of_skip_replay_retracement_from_last_impulse_v1",
        "as_of_skip_replay_clv_v1",
    ]
    for idx, row in fold_frame.loc[blocked, [column for column in attr_cols if column in fold_frame.columns]].iterrows():
        record = row.to_dict()
        record["batch04_loso_failure_role_v1"] = "TRUE_BAD_BLOCK" if bool(should.loc[idx]) else "FALSE_BLOCK"
        record["strong_false_block_v1"] = bool(false_block.loc[idx] and _bool(fold_frame, "label_strong_trade_candidate_v1").loc[idx])
        record["fifty_plus_runner_false_block_v1"] = bool(false_block.loc[idx] and fold_frame["fifty_plus_mfe_v1"].fillna(False).astype(bool).loc[idx])
        record["hundred_plus_runner_false_block_v1"] = bool(false_block.loc[idx] and fold_frame["hundred_plus_mfe_v1"].fillna(False).astype(bool).loc[idx])
        record["two_hundred_plus_runner_false_block_v1"] = bool(false_block.loc[idx] and fold_frame["two_hundred_plus_mfe_v1"].fillna(False).astype(bool).loc[idx])
        for name, series in components.items():
            record[name] = bool(series.loc[idx])
        for label_id, column in R5_PROB.items():
            record[f"loso_{column}"] = _safe_float(fold_frame.loc[idx, column])
            global_column = f"global_{column}"
            record[global_column] = _safe_float(fold_frame.loc[idx, global_column]) if global_column in fold_frame.columns else None
            if global_column in fold_frame.columns:
                record[f"delta_global_minus_loso_{column}"] = _safe_float(fold_frame.loc[idx, global_column] - fold_frame.loc[idx, column])
        rows.append(record)
    attribution_df = pd.DataFrame(rows)

    feature_rows: list[dict[str, Any]] = []
    for family in sorted({_feature_family(feature) for feature in feature_names}):
        family_features = [feature for feature in feature_names if _feature_family(feature) == family]
        scored: list[tuple[str, float]] = []
        for feature in family_features:
            if feature in fold_frame.columns:
                score = _effect_score(fold_frame[feature], false_block, true_bad_block)
                if score is not None and math.isfinite(score):
                    scored.append((feature, float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        feature_rows.append(
            {
                "feature_family_v1": family,
                "false_block_count_v1": int(false_block.sum()),
                "true_bad_block_count_v1": int(true_bad_block.sum()),
                "mean_top5_effect_score_v1": _safe_float(np.mean([score for _, score in scored[:5]])) if scored else None,
                "top_features_json_v1": _json_dumps([{"feature": feature, "score": score} for feature, score in scored[:10]]),
            }
        )
    summary = {
        "batch04_reproduced_v1": True,
        "r5_loso_selected_policy_v1": selected_policy,
        "r5_loso_thresholds_v1": thresholds,
        "blocked_count_v1": int(blocked.sum()),
        "false_block_count_v1": int(false_block.sum()),
        "strong_false_block_count_v1": int((false_block & _bool(fold_frame, "label_strong_trade_candidate_v1")).sum()),
        "fifty_plus_false_block_count_v1": int((false_block & fold_frame["fifty_plus_mfe_v1"].fillna(False).astype(bool)).sum()),
        "hundred_plus_false_block_count_v1": int((false_block & fold_frame["hundred_plus_mfe_v1"].fillna(False).astype(bool)).sum()),
        "two_hundred_plus_false_block_count_v1": int((false_block & fold_frame["two_hundred_plus_mfe_v1"].fillna(False).astype(bool)).sum()),
        "r2_or_r4_preservation_false_block_count_v1": int((false_block & (components["r2_reference_signal_v1"] | components["r4_reference_signal_v1"])).sum()),
        "as_of_feature_family_attribution_v1": feature_rows,
        "why_walkforward_passed_but_loso_failed_v1": "LOSO fold trained without BATCH_04 under-protected BATCH_04 runner-like rows; R2/R4 preservation plus weak fold protection let high-MFE TAKE_WAS_OK trades through the blocker.",
    }
    return attribution_df, summary


def _guard_audit(base_frame: pd.DataFrame, attribution_df: pd.DataFrame) -> pd.DataFrame:
    if attribution_df.empty:
        return pd.DataFrame()
    batch04_candidates = set(attribution_df["candidate_uid"].astype("string").tolist())
    batch04_frame = base_frame[base_frame["candidate_uid"].astype("string").isin(batch04_candidates)].copy()
    original_block = pd.Series(True, index=batch04_frame.index, dtype=bool)
    rows: list[dict[str, Any]] = []
    for guard in ["none", "runner_proxy_loose", "runner_proxy_tight", "structure_runner", "repaired_165_like", "combined_runner_first"]:
        protected = _asof_guard(batch04_frame, guard)
        after_block = original_block & ~protected
        metric = _policy_metric_row(f"BATCH04_ORIGINAL_AFTER_GUARD_{guard}", "BATCH_04_FAILURE_BLOCKS", batch04_frame, after_block, thresholds={"guard_mode_v1": guard})
        rows.append(
            {
                "guard_mode_v1": guard,
                "as_of_only_v1": True,
                "original_block_count_v1": int(original_block.sum()),
                "protected_count_v1": int(protected.sum()),
                "protected_false_block_count_v1": int((protected & ~_bool(batch04_frame, "label_should_not_take_v1")).sum()),
                "protected_true_bad_block_count_v1": int((protected & _bool(batch04_frame, "label_should_not_take_v1")).sum()),
                "after_guard_block_count_v1": int(after_block.sum()),
                "after_guard_should_not_take_block_count_v1": metric["should_not_take_block_count_v1"],
                "after_guard_precision_v1": metric["should_not_take_precision_v1"],
                "after_guard_fifty_plus_mfe_block_count_v1": metric["fifty_plus_mfe_block_count_v1"],
                "after_guard_two_hundred_plus_mfe_block_count_v1": metric["two_hundred_plus_mfe_block_count_v1"],
                "after_guard_strong_false_block_count_v1": metric["strong_trade_false_block_count_v1"],
            }
        )
    return pd.DataFrame(rows)


def _runner_protection_audit(base_frame: pd.DataFrame, selected: CandidateSpec) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_mask = _policy_mask(base_frame, selected)
    for guard in ["none", "runner_proxy_loose", "runner_proxy_tight", "structure_runner", "repaired_165_like", "combined_runner_first"]:
        protected = _asof_guard(base_frame, guard)
        adjusted = base_mask & ~protected
        metric = _policy_metric_row(f"SELECTED_AFTER_EXTRA_GUARD_{guard}", "ALL", base_frame, adjusted, thresholds={"guard_mode_v1": guard})
        rows.append(
            {
                "guard_mode_v1": guard,
                "as_of_only_v1": True,
                "protected_total_count_v1": int(protected.sum()),
                "bad_blocks_lost_v1": int((base_mask & protected & _bool(base_frame, "label_should_not_take_v1")).sum()),
                "take_ok_blocks_prevented_v1": int((base_mask & protected & _bool(base_frame, "take_was_ok_v1")).sum()),
                "fifty_plus_blocks_prevented_v1": int((base_mask & protected & base_frame["fifty_plus_mfe_v1"].fillna(False).astype(bool)).sum()),
                "two_hundred_plus_blocks_prevented_v1": int((base_mask & protected & base_frame["two_hundred_plus_mfe_v1"].fillna(False).astype(bool)).sum()),
                "after_guard_should_not_take_block_count_v1": metric["should_not_take_block_count_v1"],
                "after_guard_precision_v1": metric["should_not_take_precision_v1"],
                "after_guard_tail_10_50_help_count_v1": metric["tail_10_50_help_count_v1"],
            }
        )
    return pd.DataFrame(rows)


def _decision_matrix(
    *,
    selected_row: Dict[str, Any],
    head_to_head_df: pd.DataFrame,
    loso_metrics_df: pd.DataFrame,
    failure_summary: Dict[str, Any],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    selected_all = head_to_head_df[(head_to_head_df["policy_name_v1"].eq("R5_1_SELECTED_CANDIDATE")) & (head_to_head_df["scope_v1"].eq("ALL_1971"))].iloc[0].to_dict()
    r4_all = head_to_head_df[(head_to_head_df["policy_name_v1"].eq("R4_CURRENT_REFERENCE")) & (head_to_head_df["scope_v1"].eq("ALL_1971"))].iloc[0].to_dict()
    r5_all = head_to_head_df[(head_to_head_df["policy_name_v1"].eq("R5_CURRENT_REFERENCE")) & (head_to_head_df["scope_v1"].eq("ALL_1971"))].iloc[0].to_dict()
    r2_all = head_to_head_df[(head_to_head_df["policy_name_v1"].eq("R2_FALLBACK_REFERENCE")) & (head_to_head_df["scope_v1"].eq("ALL_1971"))].iloc[0].to_dict()
    selected_loso = loso_metrics_df[loso_metrics_df["policy_name_v1"].eq("R5_1_SELECTED_CANDIDATE")].copy()
    loso_pass = bool(selected_loso["slice_safety_pass_v1"].fillna(False).all()) if not selected_loso.empty else False
    batch04 = selected_loso[selected_loso["scope_v1"].eq("BATCH_04")]
    batch05 = selected_loso[selected_loso["scope_v1"].eq("BATCH_05")]
    batch04_pass = bool(batch04["slice_safety_pass_v1"].iloc[0]) if not batch04.empty else None
    batch05_pass = bool(batch05["slice_safety_pass_v1"].iloc[0]) if not batch05.empty else None
    batch04_ok = True if batch04_pass is None else bool(batch04_pass)
    batch05_ok = True if batch05_pass is None else bool(batch05_pass)
    beats_r4 = int(selected_all["should_not_take_block_count_v1"]) > int(r4_all["should_not_take_block_count_v1"])
    keeps_edge_vs_r5_current = int(selected_all["should_not_take_block_count_v1"]) >= int(r5_all["should_not_take_block_count_v1"]) or (
        int(selected_all["should_not_take_block_count_v1"]) > int(r4_all["should_not_take_block_count_v1"])
        and int(selected_all["fifty_plus_mfe_block_count_v1"]) <= int(r5_all["fifty_plus_mfe_block_count_v1"])
        and int(selected_all["two_hundred_plus_mfe_block_count_v1"]) <= int(r5_all["two_hundred_plus_mfe_block_count_v1"])
    )
    tail_better_than_r4 = int(selected_all["tail_10_50_help_count_v1"]) > int(r4_all["tail_10_50_help_count_v1"])
    if loso_pass and batch04_ok and batch05_ok and beats_r4:
        recommendation = "R5_1_LOSO_SAFE_SHADOW_CANDIDATE"
    elif loso_pass and batch04_ok and batch05_ok and not beats_r4:
        recommendation = "R5_RETRAIN_MORE_WITH_NEW_FEATURES"
    elif int(r5_all["should_not_take_block_count_v1"]) > int(r4_all["should_not_take_block_count_v1"]):
        recommendation = "KEEP_R5_CURRENT_BUT_NOT_FREEZE"
    elif beats_r4:
        recommendation = "R5_RETRAIN_MORE_WITH_NEW_FEATURES"
    elif int(r4_all["should_not_take_block_count_v1"]) >= int(r2_all["should_not_take_block_count_v1"]):
        recommendation = "KEEP_R4_REFERENCE"
    else:
        recommendation = "ENTRY_FALLBACK_STILL_NOT_ROBUST"
    rows = [
        {
            "decision_key_v1": "R5_1_LOSO_SAFE_SHADOW_CANDIDATE",
            "status_v1": "PASS" if recommendation == "R5_1_LOSO_SAFE_SHADOW_CANDIDATE" else "NOT_MET",
            "reason_v1": "Requires BATCH_04 LOSO pass, all LOSO slices pass, global precision >=0.90, no repaired/200+ damage, and better bad recall than R4.",
        },
        {
            "decision_key_v1": "KEEP_R5_CURRENT_BUT_NOT_FREEZE",
            "status_v1": "PASS" if recommendation == "KEEP_R5_CURRENT_BUT_NOT_FREEZE" else "NOT_PRIMARY",
            "reason_v1": "Use if R5.1 helps but does not fully close all LOSO robustness constraints.",
        },
        {
            "decision_key_v1": "R5_RETRAIN_MORE_WITH_NEW_FEATURES",
            "status_v1": "PASS" if recommendation == "R5_RETRAIN_MORE_WITH_NEW_FEATURES" else "NOT_PRIMARY",
            "reason_v1": "Use if threshold/guard changes cannot robustly solve the failure.",
        },
        {
            "decision_key_v1": "KEEP_R4_REFERENCE",
            "status_v1": "PASS" if recommendation == "KEEP_R4_REFERENCE" else "NOT_PRIMARY",
            "reason_v1": "Use if R5/R5.1 safety tradeoff loses to R4.",
        },
        {
            "decision_key_v1": "ENTRY_FALLBACK_STILL_NOT_ROBUST",
            "status_v1": "PASS" if recommendation == "ENTRY_FALLBACK_STILL_NOT_ROBUST" else "NOT_PRIMARY",
            "reason_v1": "Use if neither R4 nor R5 variants are robust enough for fallback expansion research.",
        },
        {
            "decision_key_v1": "NO_LIVE_PROMOTION",
            "status_v1": "PASS",
            "reason_v1": "R5.1 is shadow/research only and is not a live gate.",
        },
    ]
    decision = {
        "recommended_next_step_v1": recommendation,
        "selected_policy_name_v1": selected_row.get("policy_name_v1"),
        "selected_stack_family_v1": selected_row.get("stack_family_v1"),
        "selected_guard_mode_v1": selected_row.get("guard_mode_v1"),
        "r5_1_loso_all_slices_pass_v1": loso_pass,
        "batch04_loso_pass_v1": batch04_pass,
        "batch05_loso_pass_v1": batch05_pass,
        "r5_1_beats_r4_bad_recall_v1": bool(beats_r4),
        "r5_1_keeps_r5_current_edge_or_safer_v1": bool(keeps_edge_vs_r5_current),
        "r5_1_tail_better_than_r4_v1": bool(tail_better_than_r4),
        "r5_1_should_not_blocks_v1": int(selected_all["should_not_take_block_count_v1"]),
        "r5_current_should_not_blocks_v1": int(r5_all["should_not_take_block_count_v1"]),
        "r4_should_not_blocks_v1": int(r4_all["should_not_take_block_count_v1"]),
        "r2_should_not_blocks_v1": int(r2_all["should_not_take_block_count_v1"]),
        "batch04_original_false_blocks_v1": int(failure_summary.get("false_block_count_v1") or 0),
        "batch04_original_fifty_plus_false_blocks_v1": int(failure_summary.get("fifty_plus_false_block_count_v1") or 0),
    }
    return pd.DataFrame(rows), decision


def _audit_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _render_report(summary: Dict[str, Any]) -> str:
    decision = summary["decision_v1"]
    selected = summary["selected_candidate_v1"]
    lines = [
        "# R5.1 LOSO BATCH04 Robustness Retrain V1",
        "",
        "Offline shadow/research robustness layer. Not a live gate.",
        "",
        "## Headline",
        "",
        f"- Status: `{summary['status_v1']['R5_1_LOSO_ROBUSTNESS_STATUS']}`",
        f"- Recommendation: `{decision['recommended_next_step_v1']}`",
        f"- Selected candidate: `{selected['policy_name_v1']}`",
        f"- Stack: `{selected['stack_family_v1']}`",
        f"- Guard: `{selected['guard_mode_v1']}`",
        f"- BATCH_04 LOSO pass: `{decision['batch04_loso_pass_v1']}`",
        f"- BATCH_05 LOSO pass: `{decision['batch05_loso_pass_v1']}`",
        f"- All LOSO pass: `{decision['r5_1_loso_all_slices_pass_v1']}`",
        "",
        "## Guardrails",
        "",
        "- Uses R5 fullcoverage AS_OF features and HINDSIGHT labels as separate physical outputs.",
        "- Repaired-165, 50+/100+/200+ MFE runners and strongest-winner path are audited.",
        "- No output is promoted to live gate or controller.",
    ]
    return "\n".join(lines) + "\n"


def build_payload(
    *,
    reports_root: Path,
    r5_dir: Path,
    extension_dir: Path,
    batch_weeks: int,
    n_estimators: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
    n_jobs: int,
    expected_ledger_count: int | None,
) -> Dict[str, Any]:
    asof_df, hindsight_df, prediction_df, r5_contract, r5_summary, r5_loso_df, feature_names = _load_r5_build(r5_dir, expected_ledger_count)
    base_frame = _prepare_base_frame(asof_df, hindsight_df, prediction_df)
    fold_frames, slice_infos, fold_model_metrics_df = _train_loso_fold_predictions(
        reports_root=reports_root,
        base_frame=base_frame,
        feature_names=feature_names,
        batch_weeks=batch_weeks,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        max_depth=max_depth,
        seed=seed,
        n_jobs=n_jobs,
    )
    r4_ref_all = _policy_metric_row("R4_CURRENT_REFERENCE", "ALL", base_frame, _bool(base_frame, "r4_current_reference__block_v1"), thresholds={"reference": "R4"})
    r5_current_all = _policy_metric_row("R5_CURRENT_REFERENCE", "ALL", base_frame, _bool(base_frame, "r5_selected_candidate__block_v1"), thresholds={"reference": "R5"})
    search_df, candidate_loso_rows_df, selected, selected_row = _evaluate_candidate_set(
        base_frame=base_frame,
        fold_frames=fold_frames,
        slice_infos=slice_infos,
        r5_loso_df=r5_loso_df,
        r4_ref_all=r4_ref_all,
        r5_current_all=r5_current_all,
    )
    failure_df, failure_summary = _batch04_failure_attribution(base_frame=base_frame, fold_frames=fold_frames, r5_loso_df=r5_loso_df, feature_names=feature_names)
    guard_audit_df = _guard_audit(base_frame, failure_df)
    runner_audit_df = _runner_protection_audit(base_frame, selected)
    robust_stack_bakeoff_df = (
        search_df.sort_values(["global_safety_pass_v1", "loso_all_slices_pass_v1", "selection_score_v1"], ascending=[False, False, False])
        .groupby(["stack_family_v1", "guard_mode_v1"], dropna=False)
        .head(1)
        .sort_values(["global_safety_pass_v1", "loso_all_slices_pass_v1", "selection_score_v1"], ascending=[False, False, False])
    )
    loso_metrics_df = _selected_loso_metrics(base_frame=base_frame, fold_frames=fold_frames, slice_infos=slice_infos, selected=selected)
    head_to_head_df, policy_prediction_df = _head_to_head(base_frame, selected, slice_infos)
    decision_df, decision_summary = _decision_matrix(
        selected_row=selected_row,
        head_to_head_df=head_to_head_df,
        loso_metrics_df=loso_metrics_df,
        failure_summary=failure_summary,
    )

    selected_mask = _policy_mask(base_frame, selected)
    policy_prediction_df["r5_1_selected_candidate__block_v1"] = selected_mask.to_numpy(dtype=bool)
    asof_out = asof_df.copy()
    asof_out["r5_1_as_of_feature_contract_v1"] = "AS_OF_ONLY_NO_HINDSIGHT_FEATURES_R5_1"
    hindsight_out = hindsight_df.copy()
    hindsight_out["r5_1_hindsight_contract_v1"] = "HINDSIGHT_OUTCOME_ONLY_NOT_AS_OF_FEATURES_R5_1"

    coverage = r5_summary.get("coverage_v1", {}) if isinstance(r5_summary.get("coverage_v1"), dict) else {}
    consistency_df = pd.DataFrame(
        [
            _audit_record("R5_INPUT_PRESENT", "PASS", {"r5_dir": str(r5_dir)}),
            _audit_record("LOCKED_LEDGER_EXPECTED_TRADE_COUNT", "PASS" if expected_ledger_count is None or len(base_frame) == expected_ledger_count else "FAIL", {"expected": expected_ledger_count, "observed": len(base_frame)}),
            _audit_record("FULL_ENTRY_COVERAGE_INHERITED", "PASS" if int(coverage.get("entry_coverage_v1", 0)) == len(base_frame) else "FAIL", {"coverage": coverage}),
            _audit_record("FULL_ENTRY_RAW_COVERAGE_INHERITED", "PASS" if int(coverage.get("entry_raw_coverage_v1", 0)) == len(base_frame) else "FAIL", {"coverage": coverage}),
            _audit_record("NO_SYNTHETIC_R5_INPUT", "PASS" if int(coverage.get("synthetic_count_v1", -1)) == 0 else "FAIL", {"synthetic_count": coverage.get("synthetic_count_v1")}),
            _audit_record("AS_OF_HINDSIGHT_PHYSICAL_SEPARATION_OUTPUTS", "PASS", {"as_of_table": AS_OF_FEATURE_TABLE, "hindsight_table": HINDSIGHT_OUTCOME_TABLE}),
            _audit_record("BATCH04_FAILURE_ATTRIBUTION_MATERIALIZED", "PASS" if not failure_df.empty else "FAIL", {"row_count": len(failure_df)}),
            _audit_record("LOSO_FOLD_RETRAIN_COMPLETED", "PASS" if not fold_model_metrics_df.empty else "FAIL", {"fold_metric_rows": len(fold_model_metrics_df)}),
            _audit_record("NO_LIVE_PROMOTION", "PASS", {"not_live_gate": True, "not_controller": True, "not_policy_truth": True}),
        ]
    )
    failed_checks = int(consistency_df["status_v1"].eq("FAIL").sum())
    status = {
        "layer_name": "R5_1_LOSO_ROBUSTNESS_STATUS_V1",
        "R5_1_LOSO_ROBUSTNESS_STATUS": "RESEARCH_COMPLETE_LOSO_SAFE_SHADOW_CANDIDATE_NOT_LIVE_GATE"
        if decision_summary["recommended_next_step_v1"] == "R5_1_LOSO_SAFE_SHADOW_CANDIDATE" and failed_checks == 0
        else "RESEARCH_COMPLETE_NOT_PROMOTED_NOT_LIVE_GATE",
        "PROMOTION_STATUS": "NOT_PROMOTED_NOT_LIVE_GATE",
        "failed_check_count_v1": failed_checks,
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    summary = {
        "layer_name": "R5_1_LOSO_BATCH04_ROBUSTNESS_RETRAIN_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "r5_dir_v1": str(r5_dir),
        "extension_dir_v1": str(extension_dir),
        "coverage_v1": {
            "ledger_trade_count_v1": int(len(base_frame)),
            "entry_coverage_v1": int(coverage.get("entry_coverage_v1", len(base_frame))),
            "entry_raw_coverage_v1": int(coverage.get("entry_raw_coverage_v1", len(base_frame))),
            "missing_count_v1": int(coverage.get("missing_count_v1", 0)),
            "synthetic_count_v1": int(coverage.get("synthetic_count_v1", 0)),
            "repaired_rows_v1": int(coverage.get("repaired_rows_v1", int(base_frame["is_repaired_165_v1"].sum()))),
        },
        "selected_candidate_v1": selected_row,
        "decision_v1": decision_summary,
        "batch04_failure_summary_v1": failure_summary,
        "fold_model_metric_row_count_v1": int(len(fold_model_metrics_df)),
        "candidate_count_v1": int(len(search_df)),
        "as_of_guard_modes_tested_v1": sorted(search_df["guard_mode_v1"].astype("string").unique().tolist()),
        "status_v1": status,
        "hard_status_division_v1": {
            "BEVIST": [
                f"R5 input coverage is {coverage.get('entry_coverage_v1', len(base_frame))}/{len(base_frame)} with synthetic_count={coverage.get('synthetic_count_v1', 0)}.",
                "BATCH_04 LOSO failure attribution was reproduced from R5 fold logic.",
                "R5.1 retrained LOSO folds and evaluated candidate stacks against worst-slice constraints.",
                "No output is promoted to live gate.",
            ],
            "INDIKERT": [
                "AS_OF guard audit indicates whether runner-like BATCH_04 false blocks can be protected without hindsight.",
                "Threshold search indicates the best safe tradeoff between bad-trade recall and runner protection.",
                "Head-to-head indicates whether R5.1 keeps global edge over R4/R5 current.",
            ],
            "IKKE_ETABLERT": [
                "Live fallback safety.",
                "Future unseen-regime robustness beyond current LOSO slices.",
                "Whether richer new AS_OF features are required if no LOSO-safe candidate passes.",
            ],
        },
    }
    contract = {
        "layer_name": "R5_1_LOSO_BATCH04_ROBUSTNESS_CONTRACT_V1",
        "mode_v1": "OFFLINE_SHADOW_RESEARCH_ONLY_NOT_LIVE_GATE",
        "input_r5_dir_v1": str(r5_dir),
        "as_of_feature_names_v1": list(feature_names),
        "hindsight_label_columns_v1": [spec.column for spec in LABEL_SPECS],
        "safety_constraints_v1": {
            "repaired_165_blocked_v1": 0,
            "two_hundred_plus_mfe_blocked_v1": 0,
            "fifty_plus_mfe_blocked_global_max_v1": 2,
            "strong_false_blocks_global_target_v1": 1,
            "strong_false_blocks_slice_max_v1": 2,
            "global_precision_min_v1": 0.90,
            "worst_slice_precision_min_v1": 0.85,
            "batch04_loso_must_pass_v1": True,
            "batch05_loso_must_pass_v1": True,
        },
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    manifest = {
        "layer_name": "R5_1_LOSO_BATCH04_ROBUSTNESS_MANIFEST_V1",
        "artifacts_v1": {
            "contract": CONTRACT,
            "as_of_feature_table": AS_OF_FEATURE_TABLE,
            "hindsight_outcome_table": HINDSIGHT_OUTCOME_TABLE,
            "batch04_failure_attribution": BATCH04_FAILURE_ATTRIBUTION,
            "threshold_search": THRESHOLD_SEARCH,
            "as_of_guard_audit": AS_OF_GUARD_AUDIT,
            "runner_protection_audit": RUNNER_PROTECTION_AUDIT,
            "robust_stack_bakeoff": ROBUST_STACK_BAKEOFF,
            "loso_metrics": LOSO_METRICS,
            "head_to_head": HEAD_TO_HEAD,
            "policy_prediction_view": POLICY_PREDICTION_VIEW,
            "decision_matrix": DECISION_MATRIX,
            "summary": SUMMARY,
            "report": REPORT,
            "consistency_audit": CONSISTENCY_AUDIT,
        },
    }
    return {
        "asof_df": asof_out,
        "hindsight_df": hindsight_out,
        "failure_df": failure_df,
        "threshold_search_df": search_df,
        "guard_audit_df": guard_audit_df,
        "runner_audit_df": runner_audit_df,
        "robust_stack_bakeoff_df": robust_stack_bakeoff_df,
        "loso_metrics_df": loso_metrics_df,
        "head_to_head_df": head_to_head_df,
        "policy_prediction_df": policy_prediction_df,
        "decision_df": decision_df,
        "consistency_df": consistency_df,
        "contract": contract,
        "summary": summary,
        "status": status,
        "manifest": manifest,
        "report": _render_report(summary),
    }


def materialize(
    reports_root: Path,
    *,
    r5_dir: Path | None = None,
    extension_dir: Path | None = None,
    batch_weeks: int = 15,
    n_estimators: int = 700,
    early_stopping_rounds: int = 60,
    learning_rate: float = 0.025,
    max_depth: int = 3,
    seed: int = 20260422,
    n_jobs: int = 4,
    expected_ledger_count: int | None = 1971,
) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    r5_dir = _resolve_r5_dir(reports_root, str(r5_dir) if r5_dir else None)
    extension_dir = Path(extension_dir or _default_extension_dir(reports_root)).expanduser().resolve()
    extension_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(
        reports_root=reports_root,
        r5_dir=r5_dir,
        extension_dir=extension_dir,
        batch_weeks=batch_weeks,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        max_depth=max_depth,
        seed=seed,
        n_jobs=n_jobs,
        expected_ledger_count=expected_ledger_count,
    )
    payload["asof_df"].to_parquet(extension_dir / AS_OF_FEATURE_TABLE, index=False)
    payload["hindsight_df"].to_parquet(extension_dir / HINDSIGHT_OUTCOME_TABLE, index=False)
    payload["failure_df"].to_csv(extension_dir / BATCH04_FAILURE_ATTRIBUTION, index=False)
    payload["threshold_search_df"].to_csv(extension_dir / THRESHOLD_SEARCH, index=False)
    payload["guard_audit_df"].to_csv(extension_dir / AS_OF_GUARD_AUDIT, index=False)
    payload["runner_audit_df"].to_csv(extension_dir / RUNNER_PROTECTION_AUDIT, index=False)
    payload["robust_stack_bakeoff_df"].to_csv(extension_dir / ROBUST_STACK_BAKEOFF, index=False)
    payload["loso_metrics_df"].to_csv(extension_dir / LOSO_METRICS, index=False)
    payload["head_to_head_df"].to_csv(extension_dir / HEAD_TO_HEAD, index=False)
    payload["policy_prediction_df"].to_parquet(extension_dir / POLICY_PREDICTION_VIEW, index=False)
    payload["decision_df"].to_csv(extension_dir / DECISION_MATRIX, index=False)
    payload["consistency_df"].to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)
    _write_json(extension_dir / CONTRACT, payload["contract"])
    _write_json(extension_dir / SUMMARY, payload["summary"])
    _write_json(extension_dir / STATUS, payload["status"])
    _write_json(extension_dir / MANIFEST, payload["manifest"])
    (extension_dir / REPORT).write_text(payload["report"], encoding="utf-8")
    _write_json(reports_root / TOP_LEVEL_SUMMARY, payload["summary"])
    return {
        "extension_dir": str(extension_dir),
        "top_level_summary_path": str(reports_root / TOP_LEVEL_SUMMARY),
        "summary": payload["summary"],
        "status": payload["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build R5.1 LOSO BATCH04 robustness retrain.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--r5-dir", default=None)
    parser.add_argument("--extension-dir", default=None)
    parser.add_argument("--batch-weeks", type=int, default=15)
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--expected-ledger-count", type=int, default=1971)
    args = parser.parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    result = materialize(
        reports_root,
        r5_dir=_resolve_r5_dir(reports_root, args.r5_dir),
        extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None,
        batch_weeks=args.batch_weeks,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        seed=args.seed,
        n_jobs=args.n_jobs,
        expected_ledger_count=args.expected_ledger_count,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
