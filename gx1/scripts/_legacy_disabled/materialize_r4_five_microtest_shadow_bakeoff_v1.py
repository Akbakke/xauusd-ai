#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from gx1.scripts.materialize_r4_fullcoverage_policy_recalibration_and_shadow_replay_v1 import (
    ACTIVE_TRUTH_POINTER,
    AS_OF_TABLE as FULLCOVERAGE_AS_OF_TABLE,
    CONTRACT as FULLCOVERAGE_CONTRACT,
    EXTENSION_NAME as FULLCOVERAGE_EXTENSION_NAME,
    HINDSIGHT_TABLE as FULLCOVERAGE_HINDSIGHT_TABLE,
    SUMMARY as FULLCOVERAGE_SUMMARY,
    TASK_PROB_COLUMNS,
    _all_run_ids,
    _bool,
    _build_joined,
    _json_dumps,
    _load_json,
    _mask_for_policy_record,
    _num,
    _policy_metric_row,
    _prob,
    _resolve_reports_root,
    _safe_rate,
    _threshold_mask,
    _write_json,
)


EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R4_FIVE_MICROTEST_SHADOW_BAKEOFF_V1"

CONTRACT = "shadow_meta_all_trade_review_r4_five_microtest_bakeoff_contract_v1.json"
AS_OF_TABLE = "shadow_meta_all_trade_review_r4_five_microtest_as_of_table_v1.parquet"
HINDSIGHT_TABLE = "shadow_meta_all_trade_review_r4_five_microtest_hindsight_outcome_table_v1.parquet"
MICROTEST_RESULTS = "shadow_meta_all_trade_review_r4_five_microtest_results_v1.csv"
COMPONENT_ABLATION = "shadow_meta_all_trade_review_r4_five_microtest_component_ablation_v1.csv"
WINNER_GUARD_STRESS = "shadow_meta_all_trade_review_r4_five_microtest_winner_guard_stress_v1.csv"
TAIL_CONTROL = "shadow_meta_all_trade_review_r4_five_microtest_tail_control_v1.csv"
WALKFORWARD_ROBUSTNESS = "shadow_meta_all_trade_review_r4_five_microtest_walkforward_robustness_v1.csv"
CONSISTENCY_AUDIT = "shadow_meta_all_trade_review_r4_five_microtest_consistency_audit_v1.csv"
SUMMARY = "shadow_meta_all_trade_review_r4_five_microtest_summary_v1.json"
STATUS = "shadow_meta_all_trade_review_r4_five_microtest_status_v1.json"
MANIFEST = "shadow_meta_all_trade_review_r4_five_microtest_manifest_v1.json"
REPORT = "shadow_meta_all_trade_review_r4_five_microtest_report_v1.md"
TOP_LEVEL_SUMMARY = "truth_r4_five_microtest_shadow_bakeoff_v1.json"

CURRENT_BEST_POLICY = "R2_PRESERVED_PLUS_IMMEDIATE_MAE_DIRECT_WEAK_STRONG_PROTECTED"
CURRENT_BEST_PARAMS = {
    "should_not_take_threshold_v1": 0.60,
    "immediate_mae_risk_threshold_v1": 0.80,
    "wait_advisory_threshold_v1": 0.85,
    "direct_take_protection_ceiling_v1": 0.45,
    "strong_winner_protection_threshold_v1": 0.50,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_extension_dir(reports_root: Path) -> Path:
    return reports_root / EXTENSION_NAME


def _resolve_fullcoverage_dir(reports_root: Path, path_arg: str | None) -> Path:
    path = Path(path_arg).expanduser().resolve() if path_arg else reports_root / FULLCOVERAGE_EXTENSION_NAME
    required = [FULLCOVERAGE_CONTRACT, FULLCOVERAGE_SUMMARY, FULLCOVERAGE_AS_OF_TABLE, FULLCOVERAGE_HINDSIGHT_TABLE]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Fullcoverage build is missing required artifacts in {path}: {missing}")
    return path


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _parse_thresholds(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _current_best_record(fullcoverage_summary: Dict[str, Any]) -> Dict[str, Any]:
    record = fullcoverage_summary.get("best_constrained_policy_v1")
    if isinstance(record, dict) and record.get("policy_name_v1"):
        return record
    frontier = fullcoverage_summary.get("frontier_v1", {})
    record = frontier.get("best_constrained_policy_v1") if isinstance(frontier, dict) else None
    if isinstance(record, dict) and record.get("policy_name_v1"):
        return record
    raise RuntimeError("Fullcoverage summary missing best_constrained_policy_v1")


def _load_source_frame(
    reports_root: Path,
    fullcoverage_dir: Path,
    *,
    expected_ledger_count: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    contract = _load_json(fullcoverage_dir / FULLCOVERAGE_CONTRACT)
    fullcoverage_summary = _load_json(fullcoverage_dir / FULLCOVERAGE_SUMMARY)
    readiness_dir = Path(str(contract.get("input_readiness_dir_v1", ""))).expanduser().resolve()
    r3_dir = Path(str(contract.get("input_r3_dir_v1", ""))).expanduser().resolve()
    r4_dir = Path(str(contract.get("input_r4_dir_v1", ""))).expanduser().resolve()
    if not readiness_dir.exists() or not r3_dir.exists() or not r4_dir.exists():
        raise FileNotFoundError(
            "Fullcoverage contract points to missing input dirs; refusing fallback/dummy reconstruction: "
            f"readiness={readiness_dir}, r3={r3_dir}, r4={r4_dir}"
        )

    joined, _, _, _, _, _, _ = _build_joined(readiness_dir=readiness_dir, r3_dir=r3_dir, r4_dir=r4_dir)
    asof_df = pd.read_parquet(fullcoverage_dir / FULLCOVERAGE_AS_OF_TABLE)
    hindsight_df = pd.read_parquet(fullcoverage_dir / FULLCOVERAGE_HINDSIGHT_TABLE)
    if expected_ledger_count is not None and len(joined) != expected_ledger_count:
        raise RuntimeError(f"Locked ledger expected {expected_ledger_count}, observed {len(joined)}")
    for name, frame in [("AS_OF", asof_df), ("HINDSIGHT", hindsight_df)]:
        if len(frame) != len(joined):
            raise RuntimeError(f"{name} table row count mismatch: expected {len(joined)}, observed {len(frame)}")
        if set(frame["candidate_uid"].astype("string")) != set(joined["candidate_uid"].astype("string")):
            raise RuntimeError(f"{name} candidate_uid set does not match rebuilt joined fullcoverage source")
    return joined, asof_df, hindsight_df, contract, fullcoverage_summary


def _run_slices(reports_root: Path, frame: pd.DataFrame, *, batch_weeks: int) -> list[dict[str, Any]]:
    run_ids = _all_run_ids(reports_root, frame)
    slices: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(run_ids), batch_weeks), start=1):
        batch_run_ids = run_ids[start : start + batch_weeks]
        if not batch_run_ids:
            continue
        mask = frame["run_id"].astype("string").isin(batch_run_ids)
        slices.append(
            {
                "scope_v1": f"BATCH_{batch_index:02d}",
                "batch_index_v1": int(batch_index),
                "run_count_v1": int(len(batch_run_ids)),
                "run_start_v1": batch_run_ids[0],
                "run_end_v1": batch_run_ids[-1],
                "mask_v1": mask,
            }
        )
    return slices


def _safety_constraints(current_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "max_repaired_165_block_count_v1": 0,
        "max_two_hundred_plus_mfe_block_count_v1": 1,
        "max_fifty_plus_mfe_block_count_v1": 3,
        "max_strong_trade_false_block_count_v1": 2,
        "max_strongest_winner_path_block_count_v1": int(current_row.get("strongest_winner_path_block_count_v1", 0)),
        "min_should_not_take_precision_v1": 0.85,
    }


def _annotate_safety(row: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
    precision = _safe_float(row.get("should_not_take_precision_v1"))
    failures: list[str] = []
    checks = {
        "repaired_165_block_count_v1": int(constraints["max_repaired_165_block_count_v1"]),
        "two_hundred_plus_mfe_block_count_v1": int(constraints["max_two_hundred_plus_mfe_block_count_v1"]),
        "fifty_plus_mfe_block_count_v1": int(constraints["max_fifty_plus_mfe_block_count_v1"]),
        "strong_trade_false_block_count_v1": int(constraints["max_strong_trade_false_block_count_v1"]),
        "strongest_winner_path_block_count_v1": int(constraints["max_strongest_winner_path_block_count_v1"]),
    }
    for column, max_allowed in checks.items():
        observed = int(row.get(column) or 0)
        if observed > max_allowed:
            failures.append(f"{column}>{max_allowed}")
    if precision is None or precision < float(constraints["min_should_not_take_precision_v1"]):
        failures.append(f"precision<{constraints['min_should_not_take_precision_v1']}")
    row["strict_no_strongest_winner_path_damage_pass_v1"] = int(row.get("strongest_winner_path_block_count_v1") or 0) == 0
    row["no_additional_strongest_winner_path_damage_pass_v1"] = (
        int(row.get("strongest_winner_path_block_count_v1") or 0) <= int(constraints["max_strongest_winner_path_block_count_v1"])
    )
    row["precision_constraint_pass_v1"] = precision is not None and precision >= float(constraints["min_should_not_take_precision_v1"])
    row["safety_constraint_pass_v1"] = not failures
    row["constraint_failure_reasons_v1"] = ",".join(failures)
    row["microtest_score_v1"] = (
        float(row.get("should_not_take_block_count_v1") or 0) * 1.0
        + float(row.get("tail_10_50_help_count_v1") or 0) * 0.20
        + float(precision or 0.0) * 10.0
        - float(row.get("block_count_v1") or 0) * 0.03
        - float(row.get("take_was_ok_block_count_v1") or 0) * 0.10
    )
    if failures:
        row["microtest_score_v1"] -= 1000.0
    return row


def _weakest_slice_fields(
    reports_root: Path,
    frame: pd.DataFrame,
    block: pd.Series,
    constraints: Dict[str, Any],
    *,
    batch_weeks: int,
) -> Dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for slice_info in _run_slices(reports_root, frame, batch_weeks=batch_weeks):
        mask = slice_info["mask_v1"]
        metric = _policy_metric_row("SLICE", str(slice_info["scope_v1"]), frame.loc[mask].copy(), block.loc[mask], thresholds={"slice_v1": True})
        metric = _annotate_safety(metric, constraints)
        metric.update({key: value for key, value in slice_info.items() if key != "mask_v1"})
        rows.append(metric)
    if not rows:
        return {
            "weakest_slice_v1": None,
            "weakest_slice_precision_v1": None,
            "weakest_slice_should_not_take_block_count_v1": None,
            "weakest_slice_safety_pass_v1": False,
        }
    ranked = sorted(
        rows,
        key=lambda item: (
            bool(item.get("safety_constraint_pass_v1")),
            _safe_float(item.get("should_not_take_precision_v1")) if _safe_float(item.get("should_not_take_precision_v1")) is not None else -1.0,
            _safe_float(item.get("should_not_take_recall_v1")) or 0.0,
        ),
    )
    weakest = ranked[0]
    return {
        "weakest_slice_v1": weakest["scope_v1"],
        "weakest_slice_precision_v1": weakest.get("should_not_take_precision_v1"),
        "weakest_slice_should_not_take_block_count_v1": weakest.get("should_not_take_block_count_v1"),
        "weakest_slice_safety_pass_v1": bool(weakest.get("safety_constraint_pass_v1")),
        "weakest_slice_failure_reasons_v1": weakest.get("constraint_failure_reasons_v1", ""),
    }


def _metric_row(
    *,
    reports_root: Path,
    frame: pd.DataFrame,
    block: pd.Series,
    microtest_name: str,
    candidate_name: str,
    direction_name: str,
    constraints: Dict[str, Any],
    batch_weeks: int,
    thresholds: Dict[str, Any] | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    row = _policy_metric_row(candidate_name, "ALL", frame, block, thresholds=thresholds or {})
    row.update(
        {
            "microtest_name_v1": microtest_name,
            "candidate_name_v1": candidate_name,
            "direction_name_v1": direction_name,
        }
    )
    row.update(_weakest_slice_fields(reports_root, frame, block, constraints, batch_weeks=batch_weeks))
    if extra:
        row.update(extra)
    return _annotate_safety(row, constraints)


def _feature_available(frame: pd.DataFrame) -> pd.Series:
    return _bool(frame, "entry_r3_feature_available_v1")


def _current_best_mask(frame: pd.DataFrame) -> pd.Series:
    return _threshold_mask(frame, "IMMEDIATE_MAE_DIRECT_WEAK_STRONG_PROTECTED", CURRENT_BEST_PARAMS, preserve_r2=True)


def _threshold_nudge_rows(
    reports_root: Path,
    frame: pd.DataFrame,
    constraints: Dict[str, Any],
    *,
    batch_weeks: int,
) -> tuple[pd.DataFrame, Dict[str, pd.Series]]:
    variants = [
        (
            "A_STRICTER",
            {
                **CURRENT_BEST_PARAMS,
                "immediate_mae_risk_threshold_v1": 0.85,
                "direct_take_protection_ceiling_v1": 0.40,
            },
            "litt_strengere",
            False,
        ),
        ("A_CURRENT", dict(CURRENT_BEST_PARAMS), "current", False),
        (
            "A_SLIGHTLY_AGGRESSIVE",
            {
                **CURRENT_BEST_PARAMS,
                "immediate_mae_risk_threshold_v1": 0.75,
                "direct_take_protection_ceiling_v1": 0.50,
            },
            "litt_mer_aggressiv",
            False,
        ),
        (
            "A_AGGRESSIVE_HARD_WINNER_PROTECTION",
            {
                **CURRENT_BEST_PARAMS,
                "immediate_mae_risk_threshold_v1": 0.70,
                "direct_take_protection_ceiling_v1": 0.55,
                "strong_winner_protection_threshold_v1": 0.40,
            },
            "aggressiv_med_hard_asof_winner_guard",
            True,
        ),
    ]
    rows: list[dict[str, Any]] = []
    masks: dict[str, pd.Series] = {}
    for candidate_name, params, variant, hard_guard in variants:
        mask = _threshold_mask(frame, "IMMEDIATE_MAE_DIRECT_WEAK_STRONG_PROTECTED", params, preserve_r2=True)
        if hard_guard:
            hard_protect = (
                _prob(frame, "strong_trade_candidate").ge(0.40).fillna(False)
                | _prob(frame, "direct_take_ok").ge(0.70).fillna(False)
                | frame["is_repaired_165_v1"].fillna(False).astype(bool)
            )
            mask = (mask & ~hard_protect).fillna(False).astype(bool)
        masks[candidate_name] = mask
        rows.append(
            _metric_row(
                reports_root=reports_root,
                frame=frame,
                block=mask,
                microtest_name="R4_THRESHOLD_NUDGE_TEST_V1",
                candidate_name=candidate_name,
                direction_name="A_THRESHOLD_NUDGE",
                constraints=constraints,
                batch_weeks=batch_weeks,
                thresholds={**params, "preserve_r2_fallback_v1": True, "hard_asof_winner_guard_v1": hard_guard},
                extra={"threshold_variant_v1": variant, "policy_usable_v1": True},
            )
        )
    return pd.DataFrame(rows), masks


def _component_ablation_rows(
    reports_root: Path,
    frame: pd.DataFrame,
    constraints: Dict[str, Any],
    *,
    batch_weeks: int,
) -> tuple[pd.DataFrame, Dict[str, pd.Series]]:
    feature_available = _feature_available(frame)
    r2 = _bool(frame, "r2_entry_fallback_row_v1")
    p_mae = _prob(frame, "immediate_mae_risk")
    p_direct = _prob(frame, "direct_take_ok")
    p_wait = _prob(frame, "wait_advisory")
    p_strong = _prob(frame, "strong_trade_candidate")
    masks = {
        "B_R2_PRESERVED_ONLY": r2,
        "B_R2_PLUS_IMMEDIATE_MAE": (r2 | (feature_available & p_mae.ge(0.80).fillna(False))).astype(bool),
        "B_R2_PLUS_DIRECT_WEAK": (r2 | (feature_available & p_direct.lt(0.45).fillna(False))).astype(bool),
        "B_R2_PLUS_WAIT_ADVISORY": (r2 | (feature_available & p_wait.ge(0.85).fillna(False))).astype(bool),
        "B_R2_WITH_GLOBAL_STRONG_PROTECTOR": (r2 & ~p_strong.ge(0.50).fillna(False)).astype(bool),
        "B_FULL_CURRENT_STACK": _current_best_mask(frame),
    }
    component_roles = {
        "B_R2_PRESERVED_ONLY": "baseline_preservation",
        "B_R2_PLUS_IMMEDIATE_MAE": "adds_immediate_mae_without_direct_or_strong_guard",
        "B_R2_PLUS_DIRECT_WEAK": "adds_direct_weak_without_mae_or_strong_guard",
        "B_R2_PLUS_WAIT_ADVISORY": "adds_wait_advisory_without_direct_or_strong_guard",
        "B_R2_WITH_GLOBAL_STRONG_PROTECTOR": "tests_strong_guard_against_preserved_r2",
        "B_FULL_CURRENT_STACK": "full_current_stack",
    }
    rows = [
        _metric_row(
            reports_root=reports_root,
            frame=frame,
            block=mask,
            microtest_name="R4_STACK_ABLATION_TEST_V1",
            candidate_name=name,
            direction_name="B_STACK_ABLATION",
            constraints=constraints,
            batch_weeks=batch_weeks,
            thresholds={"ablation_v1": component_roles[name]},
            extra={"component_role_v1": component_roles[name], "policy_usable_v1": True},
        )
        for name, mask in masks.items()
    ]
    return pd.DataFrame(rows), masks


def _winner_guard_rows(
    reports_root: Path,
    frame: pd.DataFrame,
    constraints: Dict[str, Any],
    *,
    batch_weeks: int,
) -> tuple[pd.DataFrame, Dict[str, pd.Series]]:
    feature_available = _feature_available(frame)
    base = (
        _bool(frame, "r2_entry_fallback_row_v1")
        | (feature_available & _prob(frame, "immediate_mae_risk").ge(0.75).fillna(False) & _prob(frame, "direct_take_ok").lt(0.55).fillna(False))
    ).astype(bool)
    guards = {
        "C_UNGUARDED_AGGRESSIVE_BASE": (pd.Series(False, index=frame.index), "NO_GUARD", "AS_OF_POLICY_TEST"),
        "C_PROTECT_ALL_50_PLUS_MFE": (frame["fifty_plus_mfe_v1"].fillna(False).astype(bool), "protect_all_50_plus_mfe_candidates", "HINDSIGHT_STRESS_BOUND_ONLY"),
        "C_PROTECT_ALL_100_PLUS_MFE": (frame["hundred_plus_mfe_v1"].fillna(False).astype(bool), "protect_all_100_plus_mfe_candidates", "HINDSIGHT_STRESS_BOUND_ONLY"),
        "C_PROTECT_ALL_200_PLUS_MFE": (frame["two_hundred_plus_mfe_v1"].fillna(False).astype(bool), "protect_all_200_plus_mfe_candidates", "HINDSIGHT_STRESS_BOUND_ONLY"),
        "C_PROTECT_REPAIRED_165": (frame["is_repaired_165_v1"].fillna(False).astype(bool), "protect_repaired_165_pocket", "HISTORICAL_LINEAGE_GUARD_ONLY"),
        "C_PROTECT_STRONG_TRADE_CANDIDATE": (_bool(frame, "label_strong_trade_candidate_v1"), "protect_hindsight_strong_trade_candidate", "HINDSIGHT_STRESS_BOUND_ONLY"),
        "C_COMBINED_CONSERVATIVE_GUARD": (
            frame["fifty_plus_mfe_v1"].fillna(False).astype(bool)
            | _bool(frame, "label_strong_trade_candidate_v1")
            | frame["is_repaired_165_v1"].fillna(False).astype(bool),
            "protect_50_plus_or_strong_or_repaired",
            "HINDSIGHT_STRESS_BOUND_ONLY",
        ),
    }
    base_row = _policy_metric_row("C_UNGUARDED_AGGRESSIVE_BASE", "ALL", frame, base, thresholds={"aggressive_base_v1": True})
    rows: list[dict[str, Any]] = []
    masks: dict[str, pd.Series] = {}
    for name, (guard_mask, guard_name, legality) in guards.items():
        mask = (base & ~guard_mask).fillna(False).astype(bool)
        masks[name] = mask
        row = _metric_row(
            reports_root=reports_root,
            frame=frame,
            block=mask,
            microtest_name="R4_WINNER_GUARD_STRESS_TEST_V1",
            candidate_name=name,
            direction_name="C_WINNER_GUARD_STRESS",
            constraints=constraints,
            batch_weeks=batch_weeks,
            thresholds={"aggressive_base_v1": True, "guard_name_v1": guard_name},
            extra={
                "winner_guard_name_v1": guard_name,
                "guard_legality_v1": legality,
                "policy_usable_v1": legality == "AS_OF_POLICY_TEST",
                "bad_blocks_lost_vs_unguarded_v1": int(base_row["should_not_take_block_count_v1"]) - int(_policy_metric_row(name, "ALL", frame, mask)["should_not_take_block_count_v1"]),
            },
        )
        rows.append(row)
    return pd.DataFrame(rows), masks


def _tail_control_rows(
    reports_root: Path,
    frame: pd.DataFrame,
    constraints: Dict[str, Any],
    *,
    batch_weeks: int,
) -> tuple[pd.DataFrame, Dict[str, pd.Series]]:
    feature_available = _feature_available(frame)
    r2 = _bool(frame, "r2_entry_fallback_row_v1")
    p_should = _prob(frame, "should_not_take")
    p_mae = _prob(frame, "immediate_mae_risk")
    p_wait = _prob(frame, "wait_advisory")
    p_direct = _prob(frame, "direct_take_ok")
    p_strong = _prob(frame, "strong_trade_candidate")
    asof_guard = p_strong.ge(0.50).fillna(False) | frame["is_repaired_165_v1"].fillna(False).astype(bool)
    masks = {
        "D_CURRENT_STACK": _current_best_mask(frame),
        "D_TAIL_MAE_DIRECT_CONSERVATIVE": (r2 | (feature_available & p_mae.ge(0.80).fillna(False) & p_direct.lt(0.55).fillna(False) & ~asof_guard)).astype(bool),
        "D_TAIL_SHOULD_OR_MAE_DIRECT": (
            r2
            | (
                feature_available
                & (p_should.ge(0.55).fillna(False) | p_mae.ge(0.80).fillna(False))
                & p_direct.lt(0.55).fillna(False)
                & ~asof_guard
            )
        ).astype(bool),
        "D_TAIL_WAIT_DIRECT": (r2 | (feature_available & p_wait.ge(0.75).fillna(False) & p_direct.lt(0.55).fillna(False) & ~asof_guard)).astype(bool),
        "D_TAIL_COMBINED_HINDSIGHT_RUNNER_GUARD": (
            (
                r2
                | (
                    feature_available
                    & (p_should.ge(0.50).fillna(False) | p_mae.ge(0.75).fillna(False) | p_wait.ge(0.75).fillna(False))
                    & p_direct.lt(0.60).fillna(False)
                )
            )
            & ~frame["fifty_plus_mfe_v1"].fillna(False).astype(bool)
            & ~frame["is_repaired_165_v1"].fillna(False).astype(bool)
        ).astype(bool),
    }
    tail_scope = frame["tail_10_50_mfe_v1"].fillna(False).astype(bool)
    rows: list[dict[str, Any]] = []
    for name, mask in masks.items():
        tail_block_count = int((mask & tail_scope).sum())
        tail_scope_count = int(tail_scope.sum())
        row = _metric_row(
            reports_root=reports_root,
            frame=frame,
            block=mask,
            microtest_name="R4_TAIL_CONTROL_TARGET_TEST_V1",
            candidate_name=name,
            direction_name="D_TAIL_CONTROL_TARGET",
            constraints=constraints,
            batch_weeks=batch_weeks,
            thresholds={"tail_control_candidate_v1": name},
            extra={
                "tail_scope_count_v1": tail_scope_count,
                "tail_scope_block_count_v1": tail_block_count,
                "tail_scope_block_rate_v1": _safe_rate(float(tail_block_count), float(tail_scope_count)),
                "guard_legality_v1": "HINDSIGHT_STRESS_BOUND_ONLY" if "HINDSIGHT" in name else "AS_OF_POLICY_TEST",
                "policy_usable_v1": "HINDSIGHT" not in name,
            },
        )
        rows.append(row)
    return pd.DataFrame(rows), masks


def _candidate_pool_for_loo(
    candidate_maps: Sequence[Dict[str, pd.Series]],
) -> Dict[str, pd.Series]:
    pool: Dict[str, pd.Series] = {}
    for candidate_map in candidate_maps:
        for name, mask in candidate_map.items():
            if "HINDSIGHT" in name or name.startswith("C_PROTECT_ALL_") or name == "C_COMBINED_CONSERVATIVE_GUARD":
                continue
            pool[name] = mask
    return pool


def _select_best_on_scope(
    frame: pd.DataFrame,
    masks: Dict[str, pd.Series],
    scope_mask: pd.Series,
    constraints: Dict[str, Any],
) -> tuple[str, Dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, mask in masks.items():
        row = _policy_metric_row(name, "TRAIN_4_SLICES", frame.loc[scope_mask].copy(), mask.loc[scope_mask], thresholds={"loo_train_v1": True})
        rows.append(_annotate_safety(row, constraints))
    ranked = pd.DataFrame(rows)
    passed = ranked[ranked["safety_constraint_pass_v1"].fillna(False)].copy()
    if passed.empty:
        ranked = ranked.sort_values(["microtest_score_v1", "should_not_take_block_count_v1"], ascending=[False, False])
        best = ranked.iloc[0].to_dict()
    else:
        passed = passed.sort_values(["microtest_score_v1", "should_not_take_block_count_v1", "should_not_take_precision_v1"], ascending=[False, False, False])
        best = passed.iloc[0].to_dict()
    return str(best["policy_name_v1"]), best


def _walkforward_robustness_rows(
    reports_root: Path,
    frame: pd.DataFrame,
    candidate_pool: Dict[str, pd.Series],
    constraints: Dict[str, Any],
    *,
    batch_weeks: int,
) -> pd.DataFrame:
    slices = _run_slices(reports_root, frame, batch_weeks=batch_weeks)
    rows: list[dict[str, Any]] = []
    for slice_info in slices:
        holdout = slice_info["mask_v1"]
        train = ~holdout
        selected_name, train_row = _select_best_on_scope(frame, candidate_pool, train, constraints)
        selected_mask = candidate_pool[selected_name]
        holdout_row = _policy_metric_row(selected_name, str(slice_info["scope_v1"]), frame.loc[holdout].copy(), selected_mask.loc[holdout], thresholds={"loo_holdout_v1": True})
        holdout_row = _annotate_safety(holdout_row, constraints)
        rows.append(
            {
                "microtest_name_v1": "R4_WALKFORWARD_ROBUSTNESS_TEST_V1",
                "direction_name_v1": "E_WALKFORWARD_ROBUSTNESS",
                "holdout_slice_v1": slice_info["scope_v1"],
                "run_count_v1": slice_info["run_count_v1"],
                "run_start_v1": slice_info["run_start_v1"],
                "run_end_v1": slice_info["run_end_v1"],
                "selected_candidate_name_v1": selected_name,
                "selected_is_current_best_v1": selected_name in {"A_CURRENT", "B_FULL_CURRENT_STACK", "D_CURRENT_STACK"},
                "train_block_count_v1": train_row["block_count_v1"],
                "train_should_not_take_block_count_v1": train_row["should_not_take_block_count_v1"],
                "train_should_not_take_precision_v1": train_row["should_not_take_precision_v1"],
                "train_should_not_take_recall_v1": train_row["should_not_take_recall_v1"],
                "train_safety_constraint_pass_v1": train_row["safety_constraint_pass_v1"],
                "holdout_block_count_v1": holdout_row["block_count_v1"],
                "holdout_should_not_take_block_count_v1": holdout_row["should_not_take_block_count_v1"],
                "holdout_should_not_take_precision_v1": holdout_row["should_not_take_precision_v1"],
                "holdout_should_not_take_recall_v1": holdout_row["should_not_take_recall_v1"],
                "holdout_take_was_ok_block_count_v1": holdout_row["take_was_ok_block_count_v1"],
                "holdout_strong_trade_false_block_count_v1": holdout_row["strong_trade_false_block_count_v1"],
                "holdout_fifty_plus_mfe_block_count_v1": holdout_row["fifty_plus_mfe_block_count_v1"],
                "holdout_hundred_plus_mfe_block_count_v1": holdout_row["hundred_plus_mfe_block_count_v1"],
                "holdout_two_hundred_plus_mfe_block_count_v1": holdout_row["two_hundred_plus_mfe_block_count_v1"],
                "holdout_repaired_165_block_count_v1": holdout_row["repaired_165_block_count_v1"],
                "holdout_tail_10_50_help_count_v1": holdout_row["tail_10_50_help_count_v1"],
                "holdout_hindsight_skip_delta_bps_v1": holdout_row["hindsight_skip_delta_bps_v1"],
                "holdout_safety_constraint_pass_v1": holdout_row["safety_constraint_pass_v1"],
                "holdout_constraint_failure_reasons_v1": holdout_row["constraint_failure_reasons_v1"],
            }
        )
    return pd.DataFrame(rows)


def _aggregate_loo_as_microtest_row(walkforward_df: pd.DataFrame, constraints: Dict[str, Any]) -> Dict[str, Any]:
    precision_num = pd.to_numeric(walkforward_df["holdout_should_not_take_block_count_v1"], errors="coerce").sum()
    precision_den = pd.to_numeric(walkforward_df["holdout_block_count_v1"], errors="coerce").sum()
    should_count = None
    row = {
        "policy_name_v1": "E_LEAVE_ONE_SLICE_OUT_SELECTED",
        "scope_v1": "ALL_HOLDOUTS",
        "microtest_name_v1": "R4_WALKFORWARD_ROBUSTNESS_TEST_V1",
        "candidate_name_v1": "E_LEAVE_ONE_SLICE_OUT_SELECTED",
        "direction_name_v1": "E_WALKFORWARD_ROBUSTNESS",
        "row_count_v1": None,
        "block_count_v1": int(pd.to_numeric(walkforward_df["holdout_block_count_v1"], errors="coerce").sum()),
        "should_not_take_count_v1": should_count,
        "should_not_take_block_count_v1": int(precision_num),
        "should_not_take_precision_v1": _safe_rate(float(precision_num), float(precision_den)),
        "should_not_take_recall_v1": None,
        "take_was_ok_block_count_v1": int(pd.to_numeric(walkforward_df["holdout_take_was_ok_block_count_v1"], errors="coerce").sum()),
        "strong_trade_false_block_count_v1": int(pd.to_numeric(walkforward_df["holdout_strong_trade_false_block_count_v1"], errors="coerce").sum()),
        "fifty_plus_mfe_block_count_v1": int(pd.to_numeric(walkforward_df["holdout_fifty_plus_mfe_block_count_v1"], errors="coerce").sum()),
        "hundred_plus_mfe_block_count_v1": int(pd.to_numeric(walkforward_df["holdout_hundred_plus_mfe_block_count_v1"], errors="coerce").sum()),
        "two_hundred_plus_mfe_block_count_v1": int(pd.to_numeric(walkforward_df["holdout_two_hundred_plus_mfe_block_count_v1"], errors="coerce").sum()),
        "strongest_winner_path_block_count_v1": None,
        "repaired_165_block_count_v1": int(pd.to_numeric(walkforward_df["holdout_repaired_165_block_count_v1"], errors="coerce").sum()),
        "tail_10_50_help_count_v1": int(pd.to_numeric(walkforward_df["holdout_tail_10_50_help_count_v1"], errors="coerce").sum()),
        "hindsight_skip_delta_bps_v1": float(pd.to_numeric(walkforward_df["holdout_hindsight_skip_delta_bps_v1"], errors="coerce").sum()),
        "weakest_slice_v1": walkforward_df.sort_values(
            ["holdout_safety_constraint_pass_v1", "holdout_should_not_take_precision_v1", "holdout_should_not_take_block_count_v1"],
            ascending=[True, True, True],
        ).iloc[0]["holdout_slice_v1"]
        if not walkforward_df.empty
        else None,
        "weakest_slice_safety_pass_v1": bool(walkforward_df["holdout_safety_constraint_pass_v1"].fillna(False).all()) if not walkforward_df.empty else False,
        "policy_usable_v1": True,
    }
    row = _annotate_safety(row, constraints)
    row["safety_constraint_pass_v1"] = bool(walkforward_df["holdout_safety_constraint_pass_v1"].fillna(False).all()) if not walkforward_df.empty else False
    if not row["safety_constraint_pass_v1"]:
        row["constraint_failure_reasons_v1"] = "one_or_more_holdout_slices_failed"
        row["microtest_score_v1"] -= 1000.0
    return row


def _build_direction_leaderboard(microtest_df: pd.DataFrame) -> pd.DataFrame:
    eligible = microtest_df.copy()
    eligible["policy_usable_v1"] = eligible.get("policy_usable_v1", True)
    rows: list[dict[str, Any]] = []
    for direction, group in eligible.groupby("direction_name_v1", dropna=False):
        usable = group[group["policy_usable_v1"].fillna(False).astype(bool)].copy()
        source = usable if not usable.empty else group
        best = source.sort_values(
            ["safety_constraint_pass_v1", "microtest_score_v1", "should_not_take_block_count_v1", "should_not_take_precision_v1"],
            ascending=[False, False, False, False],
        ).iloc[0].to_dict()
        rows.append(
            {
                "direction_name_v1": direction,
                "best_candidate_name_v1": best["candidate_name_v1"],
                "best_microtest_name_v1": best["microtest_name_v1"],
                "best_score_v1": best["microtest_score_v1"],
                "best_safety_pass_v1": best["safety_constraint_pass_v1"],
                "best_should_not_take_block_count_v1": best.get("should_not_take_block_count_v1"),
                "best_should_not_take_precision_v1": best.get("should_not_take_precision_v1"),
                "best_fifty_plus_mfe_block_count_v1": best.get("fifty_plus_mfe_block_count_v1"),
                "best_two_hundred_plus_mfe_block_count_v1": best.get("two_hundred_plus_mfe_block_count_v1"),
                "best_repaired_165_block_count_v1": best.get("repaired_165_block_count_v1"),
                "best_tail_10_50_help_count_v1": best.get("tail_10_50_help_count_v1"),
                "policy_usable_v1": best.get("policy_usable_v1", True),
            }
        )
    leaderboard = pd.DataFrame(rows)
    return leaderboard.sort_values(["best_safety_pass_v1", "best_score_v1"], ascending=[False, False])


def _consistency_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _render_report(summary: Dict[str, Any]) -> str:
    winner = summary["direction_leaderboard_v1"][0] if summary.get("direction_leaderboard_v1") else {}
    runner_up = summary["direction_leaderboard_v1"][1] if len(summary.get("direction_leaderboard_v1", [])) > 1 else {}
    lines = [
        "# R4 Five Microtest Shadow Bakeoff V1",
        "",
        "Research/shadow only. Not a live gate.",
        "",
        "## Headline",
        "",
        f"- Status: `{summary['status_v1']['R4_FIVE_MICROTEST_SHADOW_BAKEOFF_STATUS']}`",
        f"- Coverage: `{summary['coverage_v1']['entry_coverage_v1']}/{summary['coverage_v1']['ledger_trade_count_v1']}`",
        f"- Winner direction: `{winner.get('direction_name_v1')}` / `{winner.get('best_candidate_name_v1')}`",
        f"- Runner-up direction: `{runner_up.get('direction_name_v1')}` / `{runner_up.get('best_candidate_name_v1')}`",
        f"- Recommendation: `{summary['decision_v1']['recommended_next_step_v1']}`",
        "",
        "## Guardrail",
        "",
        "All outputs are offline hindsight audits or AS_OF shadow candidates. No candidate is promoted to live gate.",
    ]
    return "\n".join(lines) + "\n"


def build_payload(
    *,
    reports_root: Path,
    fullcoverage_dir: Path,
    extension_dir: Path,
    batch_weeks: int,
    expected_ledger_count: int | None,
) -> Dict[str, Any]:
    frame, asof_df, hindsight_df, fullcoverage_contract, fullcoverage_summary = _load_source_frame(
        reports_root,
        fullcoverage_dir,
        expected_ledger_count=expected_ledger_count,
    )
    current_record = _current_best_record(fullcoverage_summary)
    current_mask = _mask_for_policy_record(frame, current_record)
    current_row = _policy_metric_row(CURRENT_BEST_POLICY, "ALL", frame, current_mask, thresholds=_parse_thresholds(current_record.get("thresholds_json_v1")))
    constraints = _safety_constraints(current_row)

    threshold_df, threshold_masks = _threshold_nudge_rows(reports_root, frame, constraints, batch_weeks=batch_weeks)
    ablation_df, ablation_masks = _component_ablation_rows(reports_root, frame, constraints, batch_weeks=batch_weeks)
    guard_df, guard_masks = _winner_guard_rows(reports_root, frame, constraints, batch_weeks=batch_weeks)
    tail_df, tail_masks = _tail_control_rows(reports_root, frame, constraints, batch_weeks=batch_weeks)
    candidate_pool = _candidate_pool_for_loo([threshold_masks, ablation_masks, guard_masks, tail_masks])
    robustness_df = _walkforward_robustness_rows(reports_root, frame, candidate_pool, constraints, batch_weeks=batch_weeks)
    robustness_summary_row = _aggregate_loo_as_microtest_row(robustness_df, constraints)

    microtest_parts = [
        threshold_df,
        ablation_df,
        guard_df,
        tail_df,
        pd.DataFrame([robustness_summary_row]),
    ]
    microtest_df = pd.concat(
        [part.dropna(axis=1, how="all") for part in microtest_parts if not part.empty],
        ignore_index=True,
        sort=False,
    )
    current_best = microtest_df[microtest_df["candidate_name_v1"].isin(["A_CURRENT", "B_FULL_CURRENT_STACK", "D_CURRENT_STACK"])].head(1)
    current_should_blocks = int(current_row["should_not_take_block_count_v1"])
    current_strong = int(current_row["strong_trade_false_block_count_v1"])
    current_fifty = int(current_row["fifty_plus_mfe_block_count_v1"])
    current_two_hundred = int(current_row["two_hundred_plus_mfe_block_count_v1"])
    current_repaired = int(current_row["repaired_165_block_count_v1"])
    current_strongest = int(current_row["strongest_winner_path_block_count_v1"])
    microtest_df["beats_current_without_more_winner_damage_v1"] = (
        pd.to_numeric(microtest_df["should_not_take_block_count_v1"], errors="coerce").gt(current_should_blocks)
        & pd.to_numeric(microtest_df["strong_trade_false_block_count_v1"], errors="coerce").le(current_strong)
        & pd.to_numeric(microtest_df["fifty_plus_mfe_block_count_v1"], errors="coerce").le(current_fifty)
        & pd.to_numeric(microtest_df["two_hundred_plus_mfe_block_count_v1"], errors="coerce").le(current_two_hundred)
        & pd.to_numeric(microtest_df["repaired_165_block_count_v1"], errors="coerce").le(current_repaired)
        & pd.to_numeric(microtest_df["strongest_winner_path_block_count_v1"], errors="coerce").fillna(current_strongest).le(current_strongest)
        & microtest_df["precision_constraint_pass_v1"].fillna(False)
    )
    direction_leaderboard_df = _build_direction_leaderboard(microtest_df)
    winner = direction_leaderboard_df.iloc[0].to_dict() if not direction_leaderboard_df.empty else {}
    runner_up = direction_leaderboard_df.iloc[1].to_dict() if len(direction_leaderboard_df) > 1 else {}
    beating = microtest_df[microtest_df["beats_current_without_more_winner_damage_v1"].fillna(False)].copy()
    usable_beating = beating[beating.get("policy_usable_v1", True).fillna(False).astype(bool)] if not beating.empty and "policy_usable_v1" in beating.columns else beating
    usable_static_beating = usable_beating[~usable_beating["direction_name_v1"].astype("string").eq("E_WALKFORWARD_ROBUSTNESS")] if not usable_beating.empty else usable_beating

    ablation_best = ablation_df.sort_values(["safety_constraint_pass_v1", "microtest_score_v1"], ascending=[False, False]).iloc[0].to_dict()
    guard_best = guard_df.sort_values(["safety_constraint_pass_v1", "microtest_score_v1"], ascending=[False, False]).iloc[0].to_dict()
    tail_best = tail_df.sort_values(["safety_constraint_pass_v1", "microtest_score_v1"], ascending=[False, False]).iloc[0].to_dict()
    loo_pass = bool(robustness_df["holdout_safety_constraint_pass_v1"].fillna(False).all()) if not robustness_df.empty else False
    threshold_has_better = bool(
        microtest_df[
            microtest_df["direction_name_v1"].astype("string").eq("A_THRESHOLD_NUDGE")
            & microtest_df["beats_current_without_more_winner_damage_v1"].fillna(False)
            & microtest_df.get("policy_usable_v1", True).fillna(False).astype(bool)
        ].shape[0]
    )
    if not usable_static_beating.empty and loo_pass:
        recommendation = "R4_THRESHOLD_FINALIZE"
    elif loo_pass and current_best is not None:
        recommendation = "FREEZE_R4_SHADOW_FALLBACK"
    elif int(current_row["should_not_take_block_count_v1"]) >= 56:
        recommendation = "R5_ENTRY_RETRAIN_WITH_REPAIRED_COVERAGE"
    else:
        recommendation = "KEEP_R2_FALLBACK"

    ledger_count = int(len(frame))
    entry_coverage = int(_bool(frame, "entry_observation_present_v1").sum())
    raw_coverage = int(_bool(frame, "entry_raw_state_present_v1").sum())
    synthetic_count = int(
        fullcoverage_summary.get("coverage_v1", {}).get("synthetic_count_v1", 0)
        if isinstance(fullcoverage_summary.get("coverage_v1"), dict)
        else 0
    )
    repaired_count = int(frame["is_repaired_165_v1"].fillna(False).sum())
    consistency_df = pd.DataFrame(
        [
            _consistency_record("LOCKED_LEDGER_EXPECTED_TRADE_COUNT", "PASS" if expected_ledger_count is None or ledger_count == expected_ledger_count else "FAIL", {"expected": expected_ledger_count, "observed": ledger_count}),
            _consistency_record("ENTRY_COVERAGE_FULL", "PASS" if entry_coverage == ledger_count else "FAIL", {"expected": ledger_count, "observed": entry_coverage}),
            _consistency_record("ENTRY_RAW_COVERAGE_FULL", "PASS" if raw_coverage == ledger_count else "FAIL", {"expected": ledger_count, "observed": raw_coverage}),
            _consistency_record("NO_SYNTHETIC_REPAIR_VALUES", "PASS" if synthetic_count == 0 else "FAIL", {"observed": synthetic_count}),
            _consistency_record("REPAIRED_165_POCKET_PRESENT", "PASS" if repaired_count > 0 else "FAIL", {"observed": repaired_count}),
            _consistency_record("R3_PROBABILITY_COLUMNS_PRESENT", "PASS" if all(column in frame.columns for column in TASK_PROB_COLUMNS.values()) else "FAIL", {"required": list(TASK_PROB_COLUMNS.values())}),
            _consistency_record("AS_OF_AND_HINDSIGHT_TABLES_SEPARATE", "PASS", {"as_of_rows": len(asof_df), "hindsight_rows": len(hindsight_df)}),
            _consistency_record("NO_LIVE_PROMOTION", "PASS", {"not_live_gate": True, "not_controller": True, "not_policy_truth": True}),
        ]
    )
    failed_checks = int(consistency_df["status_v1"].eq("FAIL").sum())
    status = {
        "layer_name": "R4_FIVE_MICROTEST_SHADOW_BAKEOFF_STATUS_V1",
        "R4_FIVE_MICROTEST_SHADOW_BAKEOFF_STATUS": "MICROTEST_BAKEOFF_READY_NOT_LIVE_GATE" if failed_checks == 0 else "ISSUES_FOUND",
        "PROMOTION_STATUS": "NOT_PROMOTED_NOT_LIVE_GATE",
        "failed_check_count_v1": failed_checks,
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    summary = {
        "layer_name": "R4_FIVE_MICROTEST_SHADOW_BAKEOFF_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "input_fullcoverage_dir_v1": str(fullcoverage_dir),
        "extension_dir_v1": str(extension_dir),
        "coverage_v1": {
            "ledger_trade_count_v1": ledger_count,
            "entry_coverage_v1": entry_coverage,
            "entry_raw_coverage_v1": raw_coverage,
            "missing_count_v1": int(ledger_count - entry_coverage),
            "synthetic_count_v1": synthetic_count,
            "repaired_rows_v1": repaired_count,
        },
        "current_best_reference_v1": current_row,
        "safety_constraints_v1": constraints,
        "direction_leaderboard_v1": direction_leaderboard_df.replace({np.nan: None}).to_dict(orient="records"),
        "winner_direction_v1": winner,
        "runner_up_direction_v1": runner_up,
        "candidate_beats_current_without_more_winner_damage_v1": bool(not usable_static_beating.empty),
        "beating_candidates_v1": usable_static_beating.replace({np.nan: None}).head(10).to_dict(orient="records") if not usable_static_beating.empty else [],
        "component_ablation_best_v1": {k: (None if pd.isna(v) else v) for k, v in ablation_best.items()},
        "winner_guard_best_v1": {k: (None if pd.isna(v) else v) for k, v in guard_best.items()},
        "tail_control_best_v1": {k: (None if pd.isna(v) else v) for k, v in tail_best.items()},
        "walkforward_robustness_v1": {
            "leave_one_slice_out_pass_v1": loo_pass,
            "holdout_slice_count_v1": int(len(robustness_df)),
            "failed_holdout_slices_v1": robustness_df.loc[~robustness_df["holdout_safety_constraint_pass_v1"].fillna(False), "holdout_slice_v1"].astype("string").tolist()
            if not robustness_df.empty
            else [],
            "selected_candidates_v1": robustness_df["selected_candidate_name_v1"].astype("string").value_counts().to_dict() if not robustness_df.empty else {},
        },
        "decision_v1": {
            "recommended_next_step_v1": recommendation,
            "threshold_nudge_found_better_than_current_v1": threshold_has_better,
            "any_policy_usable_candidate_beats_current_v1": bool(not usable_static_beating.empty),
            "leave_one_slice_out_holds_v1": loo_pass,
        },
        "hard_status_division_v1": {
            "BEVIST": [
                f"Fullcoverage source remains {entry_coverage}/{ledger_count} with {ledger_count - entry_coverage} missing.",
                f"Synthetic repair value count is {synthetic_count}.",
                "Five microtest families were materialized as shadow/research artifacts only.",
                "AS_OF table and HINDSIGHT outcome table are physically separate outputs.",
            ],
            "INDIKERT": [
                "The leaderboard indicates which R4 direction has the best offline safety/reward tradeoff.",
                "Leave-one-slice-out indicates whether threshold selection is slice-stable.",
                "Winner-guard stress tests indicate theoretical protection bounds, not deployable truth when hindsight guards are used.",
            ],
            "IKKE_ETABLERT": [
                "Live policy safety.",
                "Future causal performance improvement.",
                "Whether hindsight-only winner guards can be replaced by equally strong AS_OF predictors.",
            ],
        },
        "status_v1": status,
    }
    contract = {
        "layer_name": "R4_FIVE_MICROTEST_SHADOW_BAKEOFF_CONTRACT_V1",
        "mode_v1": "OFFLINE_SHADOW_RESEARCH_ONLY_NOT_LIVE_GATE",
        "input_fullcoverage_dir_v1": str(fullcoverage_dir),
        "source_fullcoverage_contract_v1": fullcoverage_contract,
        "microtests_v1": [
            "R4_THRESHOLD_NUDGE_TEST_V1",
            "R4_STACK_ABLATION_TEST_V1",
            "R4_WINNER_GUARD_STRESS_TEST_V1",
            "R4_TAIL_CONTROL_TARGET_TEST_V1",
            "R4_WALKFORWARD_ROBUSTNESS_TEST_V1",
        ],
        "safety_constraints_v1": constraints,
        "hindsight_labels_physically_separate_v1": True,
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    manifest = {
        "layer_name": "R4_FIVE_MICROTEST_SHADOW_BAKEOFF_MANIFEST_V1",
        "artifacts_v1": {
            "contract": CONTRACT,
            "as_of_table": AS_OF_TABLE,
            "hindsight_outcome_table": HINDSIGHT_TABLE,
            "microtest_results": MICROTEST_RESULTS,
            "component_ablation": COMPONENT_ABLATION,
            "winner_guard_stress": WINNER_GUARD_STRESS,
            "tail_control": TAIL_CONTROL,
            "walkforward_robustness": WALKFORWARD_ROBUSTNESS,
            "consistency_audit": CONSISTENCY_AUDIT,
            "summary": SUMMARY,
            "report": REPORT,
        },
    }
    return {
        "asof_df": asof_df,
        "hindsight_df": hindsight_df,
        "microtest_df": microtest_df,
        "ablation_df": ablation_df,
        "guard_df": guard_df,
        "tail_df": tail_df,
        "robustness_df": robustness_df,
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
    fullcoverage_dir: Path | None = None,
    extension_dir: Path | None = None,
    batch_weeks: int = 15,
    expected_ledger_count: int | None = 1971,
) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    resolved_fullcoverage_dir = Path(fullcoverage_dir).expanduser().resolve() if fullcoverage_dir else _resolve_fullcoverage_dir(reports_root, None)
    resolved_extension_dir = Path(extension_dir or _default_extension_dir(reports_root)).expanduser().resolve()
    payload = build_payload(
        reports_root=reports_root,
        fullcoverage_dir=resolved_fullcoverage_dir,
        extension_dir=resolved_extension_dir,
        batch_weeks=batch_weeks,
        expected_ledger_count=expected_ledger_count,
    )
    resolved_extension_dir.mkdir(parents=True, exist_ok=True)
    payload["asof_df"].to_parquet(resolved_extension_dir / AS_OF_TABLE, index=False)
    payload["hindsight_df"].to_parquet(resolved_extension_dir / HINDSIGHT_TABLE, index=False)
    payload["microtest_df"].to_csv(resolved_extension_dir / MICROTEST_RESULTS, index=False)
    payload["ablation_df"].to_csv(resolved_extension_dir / COMPONENT_ABLATION, index=False)
    payload["guard_df"].to_csv(resolved_extension_dir / WINNER_GUARD_STRESS, index=False)
    payload["tail_df"].to_csv(resolved_extension_dir / TAIL_CONTROL, index=False)
    payload["robustness_df"].to_csv(resolved_extension_dir / WALKFORWARD_ROBUSTNESS, index=False)
    payload["consistency_df"].to_csv(resolved_extension_dir / CONSISTENCY_AUDIT, index=False)
    _write_json(resolved_extension_dir / CONTRACT, payload["contract"])
    _write_json(resolved_extension_dir / SUMMARY, payload["summary"])
    _write_json(resolved_extension_dir / STATUS, payload["status"])
    _write_json(resolved_extension_dir / MANIFEST, payload["manifest"])
    (resolved_extension_dir / REPORT).write_text(payload["report"], encoding="utf-8")
    top_level = dict(payload["summary"])
    top_level["extension_dir_v1"] = str(resolved_extension_dir)
    _write_json(reports_root / TOP_LEVEL_SUMMARY, top_level)
    return {
        "extension_dir": resolved_extension_dir,
        "top_level_summary_path": reports_root / TOP_LEVEL_SUMMARY,
        "summary": payload["summary"],
        "status": payload["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize R4 five microtest shadow bakeoff.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--fullcoverage-dir", default=None)
    parser.add_argument("--extension-dir", default=None)
    parser.add_argument("--batch-weeks", type=int, default=15)
    parser.add_argument("--expected-ledger-count", type=int, default=1971)
    args = parser.parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    result = materialize(
        reports_root,
        fullcoverage_dir=_resolve_fullcoverage_dir(reports_root, args.fullcoverage_dir),
        extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None,
        batch_weeks=args.batch_weeks,
        expected_ledger_count=args.expected_ledger_count,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
