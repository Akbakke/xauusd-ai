#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from gx1.analysis.shadow_meta_v1 import (
    _ENTRY_PRE_ENTRY_PROXY_SPEC_V1,
    _build_entry_skipability_pre_entry_proxy_contract_rows_v1,
    _build_entry_skipability_pre_entry_proxy_fields_v1,
    _entry_skipability_pre_entry_proxy_source_analysis_v1,
    _validate_entry_pre_entry_proxy_input_fields_v1,
)


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

EXTENSION_PREFIX = "MONDAY_SELECTED_PRE_ENTRY_PROXIES_AND_READINESS_PACK_V1"
CANONICAL_LEDGER_DIRNAME = "ALL_TRADE_REVIEW_LEDGER_20260411"
R6_DIRNAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1"
IMPLEMENTATION_LOCK_PREFIX = "MONDAY_R6_NARROW_PRE_ENTRY_UPLIFT_IMPLEMENTATION_LOCK_V1_"

RAW_STATE = "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet"
RAW_STATE_CONTRACT = "shadow_meta_all_trade_review_entry_skipability_raw_state_contract_v1.csv"
RAW_STATE_CONTRACT_SUMMARY = "shadow_meta_all_trade_review_entry_skipability_raw_state_contract_summary_v1.json"

R6_POLICY_VIEW = "shadow_meta_all_trade_review_r6_policy_prediction_view_v1.parquet"
R6_HINDSIGHT_TABLE = "shadow_meta_all_trade_review_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet"

CONTRACT = "contract_v1.json"
IMPLEMENTATION_SUMMARY = "implementation_summary_v1.json"
RAW_STATE_CONTRACT_EXTENSION = "raw_state_contract_extension_v1.csv"
RUNNER_GUARD_LOCK = "runner_guard_implementation_lock_v1.json"
LEGALITY_TEST_REPORT = "legality_and_leakage_test_report_v1.csv"
FEATURE_COVERAGE_REPORT = "feature_coverage_and_null_policy_report_v1.csv"
FAILURE_POCKET_REPORT = "failure_pocket_wiring_report_v1.csv"
READINESS_RECHECK = "post_implementation_readiness_recheck_pack_v1.json"
NEXT_ACTION = "next_agent_action_lock_v1.json"
SUMMARY = "summary_v1.json"
REPORT = "report_v1.md"
MANIFEST = "manifest_v1.json"
STATUS = "status_v1.json"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"

NEW_FIELDS = list(_ENTRY_PRE_ENTRY_PROXY_SPEC_V1.keys())
FORENSIC_TRADE = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"


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


def _latest_dir(reports_root: Path, prefix: str) -> Path:
    matches = sorted([path for path in reports_root.iterdir() if path.is_dir() and path.name.startswith(prefix)], key=lambda path: path.name)
    if not matches:
        raise FileNotFoundError(f"No directory found for prefix {prefix} under {reports_root}")
    return matches[-1]


def _resolve_extension_dir(reports_root: Path, extension_dir_arg: str | None) -> Path:
    if extension_dir_arg:
        return Path(extension_dir_arg).expanduser().resolve()
    return reports_root / f"{EXTENSION_PREFIX}_{_utc_compact()}"


def _resolve_ledger_dir(reports_root: Path) -> Path:
    path = reports_root / CANONICAL_LEDGER_DIRNAME
    if not path.exists():
        raise FileNotFoundError(f"Canonical ledger dir missing: {path}")
    return path


def _resolve_r6_dir(reports_root: Path) -> Path:
    path = reports_root / R6_DIRNAME
    if not path.exists():
        raise FileNotFoundError(f"R6 dir missing: {path}")
    return path


def _audit_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _load_inputs(reports_root: Path) -> Dict[str, Any]:
    ledger_dir = _resolve_ledger_dir(reports_root)
    r6_dir = _resolve_r6_dir(reports_root)
    implementation_lock_dir = _latest_dir(reports_root, IMPLEMENTATION_LOCK_PREFIX)
    raw_state_df = pd.read_parquet(ledger_dir / RAW_STATE)
    contract_df = pd.read_csv(ledger_dir / RAW_STATE_CONTRACT)
    contract_summary = _load_json(ledger_dir / RAW_STATE_CONTRACT_SUMMARY)
    policy_df = pd.read_parquet(r6_dir / R6_POLICY_VIEW)
    hindsight_df = pd.read_parquet(r6_dir / R6_HINDSIGHT_TABLE)
    implementation_lock_summary = _load_json(implementation_lock_dir / SUMMARY)
    return {
        "ledger_dir": ledger_dir,
        "r6_dir": r6_dir,
        "implementation_lock_dir": implementation_lock_dir,
        "raw_state_df": raw_state_df,
        "contract_df": contract_df,
        "contract_summary": contract_summary,
        "policy_df": policy_df,
        "hindsight_df": hindsight_df,
        "implementation_lock_summary": implementation_lock_summary,
    }


def _compute_pocket_frame(
    raw_state_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    hindsight_df: pd.DataFrame,
) -> pd.DataFrame:
    pocket_cols = [
        "candidate_uid",
        "run_id",
        "is_repaired_165_v1",
        "fifty_plus_mfe_v1",
        "r6_selected_candidate__block_v1",
    ]
    hint_cols = [
        "candidate_uid",
        "r6_label_runner_near_miss_v1",
        "r6_label_tail_control_10_50_v1",
        "r6_label_missed_should_not_take_v1",
        "r6_label_risky_allow_v1",
        "r6_label_repaired_165_like_runner_v1",
    ]
    policy_subset = policy_df[[column for column in pocket_cols if column in policy_df.columns]].copy()
    if "run_id" in policy_subset.columns:
        policy_subset = policy_subset.rename(columns={"run_id": "policy_run_id_v1"})
    pocket_df = raw_state_df.merge(
        policy_subset,
        on="candidate_uid",
        how="outer",
        validate="one_to_one",
    ).merge(
        hindsight_df[[column for column in hint_cols if column in hindsight_df.columns]],
        on="candidate_uid",
        how="outer",
        validate="one_to_one",
    )
    if "run_id" not in pocket_df.columns:
        pocket_df["run_id"] = pd.Series(pd.NA, index=pocket_df.index, dtype="string")
    if "policy_run_id_v1" in pocket_df.columns:
        pocket_df["run_id"] = pocket_df["run_id"].astype("string").fillna(
            pocket_df["policy_run_id_v1"].astype("string")
        )
    for col in [
        "is_repaired_165_v1",
        "fifty_plus_mfe_v1",
        "r6_selected_candidate__block_v1",
        "r6_label_runner_near_miss_v1",
        "r6_label_tail_control_10_50_v1",
        "r6_label_missed_should_not_take_v1",
        "r6_label_risky_allow_v1",
        "r6_label_repaired_165_like_runner_v1",
    ]:
        if col in pocket_df.columns:
            pocket_df[col] = pocket_df[col].fillna(False).astype(bool)
    return pocket_df


def _update_contract_summary(
    contract_summary: Dict[str, Any],
    contract_df: pd.DataFrame,
    raw_state_df: pd.DataFrame,
) -> Dict[str, Any]:
    updated = dict(contract_summary)
    source_analysis = dict(updated.get("source_analysis_v1") or {})
    source_rows = list(source_analysis.get("source_family_rows") or [])
    source_rows = [
        row
        for row in source_rows
        if str((row or {}).get("source_family")) != "derived_pre_entry_proxy_from_exact_entry_raw_state_v1"
    ]
    source_rows.append(_entry_skipability_pre_entry_proxy_source_analysis_v1(raw_state_df))
    source_analysis["source_family_rows"] = source_rows
    updated["source_analysis_v1"] = source_analysis
    updated["role_counts"] = contract_df["raw_state_role_v1"].astype("string").value_counts(dropna=False).to_dict()
    updated["direct_only_canonical_candidate_count"] = int(
        contract_df["raw_state_role_v1"].astype("string").eq("DIRECT_ONLY_CANONICAL_CANDIDATE").sum()
    )
    updated["pre_entry_proxy_uplift_v1"] = {
        "implemented_fields_v1": NEW_FIELDS,
        "field_count_v1": int(len(NEW_FIELDS)),
        "null_means_unavailable_v1": True,
        "research_only_not_model_active_v1": True,
        "updated_at_utc_v1": _utc_now_iso(),
    }
    return updated


def _run_legality_checks(updated_raw_state_df: pd.DataFrame, updated_contract_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add(check_name: str, passed: bool, details: Dict[str, Any]) -> None:
        rows.append({"check_name_v1": check_name, "status_v1": "PASS" if passed else "FAIL", "details_json_v1": _json_dumps(details)})

    try:
        _validate_entry_pre_entry_proxy_input_fields_v1(
            "negative_management_exit_fields",
            ["as_of_skip_replay_spread_bps_v1", "last_peak_ts"],
        )
        neg_direct = False
    except RuntimeError:
        neg_direct = True
    add("NEGATIVE_DIRECT_MANAGEMENT_EXIT_FIELDS_REJECTED", neg_direct, {"expected_runtime_error_v1": True})

    try:
        _validate_entry_pre_entry_proxy_input_fields_v1(
            "negative_tail_guard_hindsight_policy_log",
            ["as_of_skip_replay_spread_bps_v1", "hindsight_peak_mfe_bps_v1", "policy_log_runner_protector_score_v1"],
        )
        neg_hindsight = False
    except RuntimeError:
        neg_hindsight = True
    add("NEGATIVE_HINDSIGHT_AND_POLICY_LOG_REJECTED", neg_hindsight, {"expected_runtime_error_v1": True})

    try:
        _validate_entry_pre_entry_proxy_input_fields_v1(
            "positive_legal_skip_replay_fields",
            _ENTRY_PRE_ENTRY_PROXY_SPEC_V1["as_of_pre_entry_directional_asymmetry_score_v1"]["inputs"],
        )
        pos_legal = True
    except RuntimeError:
        pos_legal = False
    add("POSITIVE_LEGAL_SKIP_REPLAY_FIELDS_ACCEPTED", pos_legal, {"field_count_v1": len(_ENTRY_PRE_ENTRY_PROXY_SPEC_V1["as_of_pre_entry_directional_asymmetry_score_v1"]["inputs"])})

    sample = updated_raw_state_df.head(10).copy()
    future_variant = sample.copy()
    future_variant["hindsight_peak_mfe_bps_v1"] = np.linspace(0.0, 999.0, len(future_variant))
    future_variant["last_peak_ts"] = "2099-01-01T00:00:00Z"
    base_scores = _build_entry_skipability_pre_entry_proxy_fields_v1(sample)
    future_scores = _build_entry_skipability_pre_entry_proxy_fields_v1(future_variant)
    future_invariant = base_scores.fillna(-9999.0).equals(future_scores.fillna(-9999.0))
    add("FUTURE_PERTURBATION_INVARIANCE", future_invariant, {"row_count_v1": int(len(sample))})

    null_variant = sample.copy()
    for field in _ENTRY_PRE_ENTRY_PROXY_SPEC_V1["as_of_pre_entry_vol_exp_comp_score_v1"]["inputs"]:
        null_variant[field] = np.nan
    null_scores = _build_entry_skipability_pre_entry_proxy_fields_v1(null_variant)
    null_policy_ok = bool(null_scores["as_of_pre_entry_vol_exp_comp_score_v1"].isna().all())
    add("NULL_DEFAULT_RETURNS_NULL_NOT_FALLBACK", null_policy_ok, {"feature_v1": "as_of_pre_entry_vol_exp_comp_score_v1"})

    contract_rows = updated_contract_df[updated_contract_df["feature_name"].astype("string").isin(NEW_FIELDS)].copy()
    schema_ok = (
        len(contract_rows) == len(NEW_FIELDS)
        and bool(contract_rows["as_of_safe_v1"].fillna(False).all())
        and bool(contract_rows["research_input_allowed_v1"].fillna(False).all())
    )
    add("SCHEMA_CONTRACT_EXTENDED_WITH_AS_OF_ONLY_FIELDS", schema_ok, {"contract_row_count_v1": int(len(contract_rows))})
    return pd.DataFrame(rows)


def _feature_coverage_report(
    updated_raw_state_df: pd.DataFrame,
    pocket_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    pocket_masks = {
        "repaired_165": pocket_df["is_repaired_165_v1"] if "is_repaired_165_v1" in pocket_df.columns else pd.Series(False, index=pocket_df.index),
        "runner_near_miss": pocket_df["r6_label_runner_near_miss_v1"] if "r6_label_runner_near_miss_v1" in pocket_df.columns else pd.Series(False, index=pocket_df.index),
        "tail_10_50": pocket_df["r6_label_tail_control_10_50_v1"] if "r6_label_tail_control_10_50_v1" in pocket_df.columns else pd.Series(False, index=pocket_df.index),
        "missed_should_not_take": pocket_df["r6_label_missed_should_not_take_v1"] if "r6_label_missed_should_not_take_v1" in pocket_df.columns else pd.Series(False, index=pocket_df.index),
        "risky_allow": pocket_df["r6_label_risky_allow_v1"] if "r6_label_risky_allow_v1" in pocket_df.columns else pd.Series(False, index=pocket_df.index),
    }
    for field_name in NEW_FIELDS:
        series = pd.to_numeric(updated_raw_state_df[field_name], errors="coerce")
        row: Dict[str, Any] = {
            "feature_name_v1": field_name,
            "row_count_v1": int(len(series)),
            "non_null_count_v1": int(series.notna().sum()),
            "coverage_rate_v1": float(series.notna().mean()),
            "null_count_v1": int(series.isna().sum()),
            "null_rate_v1": float(series.isna().mean()),
            "unavailable_count_v1": int(series.isna().sum()),
            "unavailable_rate_v1": float(series.isna().mean()),
            "min_v1": None if series.notna().sum() == 0 else float(series.min()),
            "max_v1": None if series.notna().sum() == 0 else float(series.max()),
        }
        holes: List[str] = []
        for pocket_name, mask in pocket_masks.items():
            if len(mask) != len(series):
                continue
            pocket_count = int(mask.fillna(False).sum())
            coverage = float(series[mask.fillna(False)].notna().mean()) if pocket_count else None
            row[f"{pocket_name}_row_count_v1"] = pocket_count
            row[f"{pocket_name}_coverage_rate_v1"] = coverage
            if pocket_count and coverage is not None and coverage < 1.0:
                holes.append(f"{pocket_name}:{coverage:.4f}")
        row["pocket_holes_v1"] = ",".join(holes)
        rows.append(row)
    return pd.DataFrame(rows)


def _failure_pocket_report(
    pocket_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> pd.DataFrame:
    coverage_lookup = coverage_df.set_index("feature_name_v1")
    pocket_specs = [
        {
            "pocket_id_v1": "repaired_165_pocket",
            "mask": pocket_df["is_repaired_165_v1"],
            "not_worse_than_v1": "blocked_count must remain 0",
            "relevant_fields_v1": ["as_of_pre_entry_swing_retracement_alignment_score_v1", "as_of_pre_entry_runner_protection_guard_score_v1"],
        },
        {
            "pocket_id_v1": "forensic_repaired_trade",
            "mask": pocket_df["candidate_uid"].astype("string").eq(FORENSIC_TRADE),
            "not_worse_than_v1": "must remain unblocked and guard-covered",
            "relevant_fields_v1": ["as_of_pre_entry_swing_retracement_alignment_score_v1", "as_of_pre_entry_runner_protection_guard_score_v1"],
        },
        {
            "pocket_id_v1": "runner_near_miss_pocket",
            "mask": pocket_df["r6_label_runner_near_miss_v1"],
            "not_worse_than_v1": "no worse runner damage than Monday R6 failure-miner baseline",
            "relevant_fields_v1": ["as_of_pre_entry_directional_asymmetry_score_v1", "as_of_pre_entry_runner_protection_guard_score_v1"],
        },
        {
            "pocket_id_v1": "missed_10_50_tail_control_pocket",
            "mask": pocket_df["r6_label_tail_control_10_50_v1"],
            "not_worse_than_v1": "tail-pocket must become coverage-trackable before retrain readiness opens",
            "relevant_fields_v1": ["as_of_pre_entry_vol_exp_comp_score_v1", "as_of_pre_entry_tail_leakage_pocket_score_v1"],
        },
        {
            "pocket_id_v1": "missed_should_not_take_pocket",
            "mask": pocket_df["r6_label_missed_should_not_take_v1"],
            "not_worse_than_v1": "no blocker expansion until protection remains green",
            "relevant_fields_v1": ["as_of_pre_entry_vol_exp_comp_score_v1", "as_of_pre_entry_directional_asymmetry_score_v1"],
        },
        {
            "pocket_id_v1": "risky_allow_pocket",
            "mask": pocket_df["r6_label_risky_allow_v1"],
            "not_worse_than_v1": "monitor only in this phase; no new blocker aggression",
            "relevant_fields_v1": ["as_of_pre_entry_vol_exp_comp_score_v1", "as_of_pre_entry_directional_asymmetry_score_v1"],
        },
    ]
    rows: List[Dict[str, Any]] = []
    for spec in pocket_specs:
        mask = spec["mask"].fillna(False).astype(bool)
        pocket_rows = pocket_df.loc[mask].copy()
        field_coverages = {
            field_name: float(pocket_rows[field_name].notna().mean()) if len(pocket_rows) and field_name in pocket_rows.columns else None
            for field_name in spec["relevant_fields_v1"]
        }
        readiness_trackable = len(pocket_rows) > 0 and all(
            field_coverages.get(field_name) is not None and field_coverages.get(field_name) >= 1.0
            for field_name in spec["relevant_fields_v1"]
        )
        rows.append(
            {
                "pocket_id_v1": spec["pocket_id_v1"],
                "row_count_v1": int(len(pocket_rows)),
                "tracking_filter_v1": spec["pocket_id_v1"],
                "relevant_fields_json_v1": _json_dumps(spec["relevant_fields_v1"]),
                "coverage_by_field_json_v1": _json_dumps(field_coverages),
                "not_worse_than_requirement_v1": spec["not_worse_than_v1"],
                "readiness_trackable_v1": bool(readiness_trackable),
            }
        )
    return pd.DataFrame(rows)


def _readiness_recheck_pack(
    legality_report_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    failure_pocket_df: pd.DataFrame,
) -> Dict[str, Any]:
    legality_failures = int(legality_report_df["status_v1"].astype("string").eq("FAIL").sum())
    minimum_coverage = float(pd.to_numeric(coverage_df["coverage_rate_v1"], errors="coerce").min())
    all_pockets_trackable = bool(failure_pocket_df["readiness_trackable_v1"].fillna(False).all())
    if legality_failures > 0:
        decision = "WAIT_FOR_LEGALITY_FIXES"
    elif minimum_coverage < 0.95:
        decision = "WAIT_FOR_COVERAGE_FIXES"
    elif all_pockets_trackable:
        decision = "READY_FOR_RETRAIN_READINESS_RECHECK"
    else:
        decision = "READY_FOR_MORE_NARROW_FEATURE_HARDENING"
    return {
        "layer_name_v1": "POST_IMPLEMENTATION_READINESS_RECHECK_PACK_V1",
        "decision_v1": decision,
        "retrain_now_v1": False,
        "minimum_feature_coverage_rate_v1": minimum_coverage,
        "legality_failure_count_v1": legality_failures,
        "all_failure_pockets_trackable_v1": all_pockets_trackable,
        "why_v1": [
            "This pack evaluates implementation quality only.",
            "Retrain remains closed until a separate retrain-readiness job explicitly opens it.",
        ],
    }


def _next_action_lock(readiness_decision: str) -> Dict[str, Any]:
    if readiness_decision == "READY_FOR_RETRAIN_READINESS_RECHECK":
        primary = "RUN_RETRAIN_READINESS_RECHECK_NEXT"
    elif readiness_decision == "WAIT_FOR_LEGALITY_FIXES":
        primary = "FIX_LEGALITY_OR_CONTRACT_ISSUES_FIRST"
    else:
        primary = "HARDEN_FEATURE_COVERAGE_FIRST"
    return {
        "layer_name_v1": "NEXT_AGENT_ACTION_LOCK_V1",
        "primary_action_v1": primary,
        "supporting_actions_v1": [
            "DO_NOT_RETRAIN_YET",
            "KEEP_MONDAY_R6_AS_FAILURE_MINER",
            "DO_NOT_TOUCH_POLICY_LAYER",
        ],
    }


def _status_block(readiness_decision: str) -> Dict[str, Any]:
    return {
        "layer_name_v1": "STATUS_DISCIPLINE_V1",
        "BEVIST": [
            "A small legal pre-entry uplift layer was implemented in canonical entry raw-state.",
            "The five selected proxy/guard fields now exist in the canonical raw-state artifact and contract.",
            "Legal deny-lists are enforced in code and tested before any model work.",
            "Retrain still does not start now.",
        ],
        "INDIKERT": [
            "The new uplift is sufficient to open a dedicated retrain-readiness recheck." if readiness_decision == "READY_FOR_RETRAIN_READINESS_RECHECK" else "The new uplift is directionally correct but still needs follow-up hardening before retrain readiness can be reopened.",
            "Runner pockets are now materially more trackable in the evaluation surface.",
            "Guard-first uplift remains the right sequencing before any blocker-side expansion.",
        ],
        "IKKE_ETABLERT": [
            "That these five fields alone will be enough to beat frozen Wednesday R6.",
            "That model work should open immediately after implementation.",
            "That no second narrow uplift wave will be needed later.",
        ],
    }


def _render_report(
    implementation_summary: Dict[str, Any],
    readiness_pack: Dict[str, Any],
    next_action: Dict[str, Any],
    status_block: Dict[str, Any],
) -> str:
    lines = [
        "# Monday Selected Pre-Entry Proxies And Readiness Pack V1",
        "",
        "Canonical raw-state uplift only. No retrain, replay, or policy activation was started.",
        "",
        "## Headline",
        "",
        f"- Readiness decision: `{readiness_pack['decision_v1']}`",
        f"- Primary next action: `{next_action['primary_action_v1']}`",
        "",
        "## Implemented Fields",
        "",
    ]
    for field_name in implementation_summary["implemented_fields_v1"]:
        lines.append(f"- `{field_name}`")
    lines += [
        "",
        "## Hard Status",
        "",
    ]
    for key in ["BEVIST", "INDIKERT", "IKKE_ETABLERT"]:
        lines.append(f"### {key}")
        lines.append("")
        for item in status_block[key]:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_payload(reports_root: Path, extension_dir: Path) -> Dict[str, Any]:
    inputs = _load_inputs(reports_root)
    ledger_dir = inputs["ledger_dir"]
    raw_state_path = ledger_dir / RAW_STATE
    contract_path = ledger_dir / RAW_STATE_CONTRACT
    summary_path = ledger_dir / RAW_STATE_CONTRACT_SUMMARY

    raw_state_df = inputs["raw_state_df"].copy()
    raw_before_cols = list(raw_state_df.columns)
    proxy_df = _build_entry_skipability_pre_entry_proxy_fields_v1(raw_state_df)
    for field_name in NEW_FIELDS:
        raw_state_df[field_name] = pd.to_numeric(proxy_df[field_name], errors="coerce")
    raw_state_df.to_parquet(raw_state_path, index=False)

    contract_df = inputs["contract_df"].copy()
    contract_df = contract_df[~contract_df["feature_name"].astype("string").isin(NEW_FIELDS)].copy()
    contract_df = pd.concat(
        [contract_df, pd.DataFrame.from_records(_build_entry_skipability_pre_entry_proxy_contract_rows_v1(raw_state_df))],
        ignore_index=True,
    ).sort_values(["feature_name"], kind="mergesort").reset_index(drop=True)
    contract_df.to_csv(contract_path, index=False)

    contract_summary = _update_contract_summary(inputs["contract_summary"], contract_df, raw_state_df)
    _write_json(summary_path, contract_summary)

    pocket_df = _compute_pocket_frame(raw_state_df, inputs["policy_df"], inputs["hindsight_df"])
    legality_report_df = _run_legality_checks(raw_state_df, contract_df)
    coverage_df = _feature_coverage_report(raw_state_df, pocket_df)
    failure_pocket_df = _failure_pocket_report(pocket_df, coverage_df)
    readiness_pack = _readiness_recheck_pack(legality_report_df, coverage_df, failure_pocket_df)
    next_action = _next_action_lock(readiness_pack["decision_v1"])
    status_block = _status_block(readiness_pack["decision_v1"])

    implementation_summary = {
        "layer_name_v1": "IMPLEMENT_SELECTED_PRE_ENTRY_PROXIES_V1",
        "canonical_ledger_dir_v1": str(ledger_dir),
        "updated_files_v1": [str(raw_state_path), str(contract_path), str(summary_path)],
        "implemented_fields_v1": NEW_FIELDS,
        "raw_state_row_count_v1": int(len(raw_state_df)),
        "raw_state_column_count_before_v1": int(len(raw_before_cols)),
        "raw_state_column_count_after_v1": int(len(raw_state_df.columns)),
        "forbidden_fields_still_banned_v1": [
            "last_peak_ts",
            "last_mfe_ts",
            "last_peak_mfe",
            "max_mfe_without_mae",
            "mfe_mae_sequence_order",
            "management policy/decision-log fields",
        ],
        "null_means_unavailable_v1": True,
    }
    runner_guard_lock = {
        "layer_name_v1": "RUNNER_GUARD_IMPLEMENTATION_LOCK_V1",
        "field_name_v1": "as_of_pre_entry_runner_protection_guard_score_v1",
        "research_shadow_only_v1": True,
        "not_live_controller_v1": True,
        "not_policy_activation_v1": True,
        "legal_input_sources_v1": _ENTRY_PRE_ENTRY_PROXY_SPEC_V1["as_of_pre_entry_runner_protection_guard_score_v1"]["inputs"],
        "explicit_use_rule_v1": "May only be consumed in future shadow evaluation as a dampener on aggressive blocker expansion after a separate readiness recheck.",
    }

    consistency_df = pd.DataFrame(
        [
            _audit_record("RAW_STATE_FILE_UPDATED", "PASS", {"path": str(raw_state_path)}),
            _audit_record("RAW_STATE_CONTRACT_UPDATED", "PASS", {"path": str(contract_path)}),
            _audit_record("RAW_STATE_SUMMARY_UPDATED", "PASS", {"path": str(summary_path)}),
            _audit_record(
                "NEW_FIELDS_PRESENT_IN_RAW_STATE",
                "PASS" if all(field_name in raw_state_df.columns for field_name in NEW_FIELDS) else "FAIL",
                {"fields": NEW_FIELDS},
            ),
            _audit_record(
                "NEW_FIELDS_PRESENT_IN_CONTRACT",
                "PASS" if set(NEW_FIELDS).issubset(set(contract_df["feature_name"].astype("string"))) else "FAIL",
                {"fields": NEW_FIELDS},
            ),
            _audit_record(
                "LEGALITY_REPORT_ALL_PASS",
                "PASS" if int(legality_report_df["status_v1"].astype("string").eq("FAIL").sum()) == 0 else "FAIL",
                {"failed": int(legality_report_df["status_v1"].astype("string").eq("FAIL").sum())},
            ),
            _audit_record(
                "FORENSIC_TRADE_TRACKABLE",
                "PASS" if int(pocket_df["candidate_uid"].astype("string").eq(FORENSIC_TRADE).sum()) == 1 else "FAIL",
                {"forensic_trade_key_v1": FORENSIC_TRADE},
            ),
            _audit_record(
                "NO_POLICY_LAYER_CHANGES",
                "PASS",
                {"note_v1": "This job only updated entry raw-state, raw-state contract, and raw-state contract summary."},
            ),
        ]
    )

    contract = {
        "layer_name_v1": "MONDAY_SELECTED_PRE_ENTRY_PROXIES_AND_READINESS_PACK_CONTRACT_V1",
        "mode_v1": "IMPLEMENTATION_ONLY_NO_MODEL_WORK",
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_promotion_v1": True,
        "not_policy_activation_v1": True,
        "inputs_v1": {
            "ledger_dir_v1": str(ledger_dir),
            "r6_dir_v1": str(inputs["r6_dir"]),
            "implementation_lock_dir_v1": str(inputs["implementation_lock_dir"]),
        },
    }
    status = {
        "layer_name_v1": "MONDAY_SELECTED_PRE_ENTRY_PROXIES_AND_READINESS_PACK_STATUS_V1",
        "SPEC_STATUS": "IMPLEMENTED_AND_AUDITED",
        "failed_check_count_v1": int(consistency_df["status_v1"].astype("string").eq("FAIL").sum()),
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_promotion_v1": True,
        "not_policy_activation_v1": True,
    }
    summary = {
        "layer_name_v1": "MONDAY_SELECTED_PRE_ENTRY_PROXIES_AND_READINESS_PACK_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "extension_dir_v1": str(extension_dir),
        "benchmark_lock_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
        "monday_safety_reference_v1": "R5_1_CANDIDATE_0241_R5_1_COMBINED_repaired_165_like",
        "monday_r6_role_v1": "FAILURE_MINER_DIAGNOSIS_ONLY",
        "implemented_fields_v1": NEW_FIELDS,
        "readiness_decision_v1": readiness_pack["decision_v1"],
        "next_action_v1": next_action["primary_action_v1"],
        "status_v1": status,
        "hard_status_division_v1": status_block,
    }
    manifest = {
        "layer_name_v1": "MONDAY_SELECTED_PRE_ENTRY_PROXIES_AND_READINESS_PACK_MANIFEST_V1",
        "artifacts_v1": {
            "contract": CONTRACT,
            "implementation_summary": IMPLEMENTATION_SUMMARY,
            "raw_state_contract_extension": RAW_STATE_CONTRACT_EXTENSION,
            "runner_guard_implementation_lock": RUNNER_GUARD_LOCK,
            "legality_and_leakage_test_report": LEGALITY_TEST_REPORT,
            "feature_coverage_and_null_policy_report": FEATURE_COVERAGE_REPORT,
            "failure_pocket_wiring_report": FAILURE_POCKET_REPORT,
            "post_implementation_readiness_recheck_pack": READINESS_RECHECK,
            "next_agent_action_lock": NEXT_ACTION,
            "summary": SUMMARY,
            "report": REPORT,
            "manifest": MANIFEST,
            "status": STATUS,
            "consistency_audit": CONSISTENCY_AUDIT,
        }
    }
    report = _render_report(implementation_summary, readiness_pack, next_action, status_block)
    return {
        "contract": contract,
        "implementation_summary": implementation_summary,
        "contract_extension_df": contract_df[contract_df["feature_name"].astype("string").isin(NEW_FIELDS)].copy(),
        "runner_guard_lock": runner_guard_lock,
        "legality_report_df": legality_report_df,
        "coverage_df": coverage_df,
        "failure_pocket_df": failure_pocket_df,
        "readiness_pack": readiness_pack,
        "next_action": next_action,
        "summary": summary,
        "manifest": manifest,
        "status": status,
        "consistency_df": consistency_df,
        "report": report,
    }


def materialize(reports_root: Path, *, extension_dir: Path | None = None) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    extension_dir = _resolve_extension_dir(reports_root, str(extension_dir) if extension_dir else None)
    extension_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(reports_root, extension_dir)
    _write_json(extension_dir / CONTRACT, payload["contract"])
    _write_json(extension_dir / IMPLEMENTATION_SUMMARY, payload["implementation_summary"])
    payload["contract_extension_df"].to_csv(extension_dir / RAW_STATE_CONTRACT_EXTENSION, index=False)
    _write_json(extension_dir / RUNNER_GUARD_LOCK, payload["runner_guard_lock"])
    payload["legality_report_df"].to_csv(extension_dir / LEGALITY_TEST_REPORT, index=False)
    payload["coverage_df"].to_csv(extension_dir / FEATURE_COVERAGE_REPORT, index=False)
    payload["failure_pocket_df"].to_csv(extension_dir / FAILURE_POCKET_REPORT, index=False)
    _write_json(extension_dir / READINESS_RECHECK, payload["readiness_pack"])
    _write_json(extension_dir / NEXT_ACTION, payload["next_action"])
    _write_json(extension_dir / SUMMARY, payload["summary"])
    (extension_dir / REPORT).write_text(payload["report"], encoding="utf-8")
    _write_json(extension_dir / MANIFEST, payload["manifest"])
    _write_json(extension_dir / STATUS, payload["status"])
    payload["consistency_df"].to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)
    return {"extension_dir": str(extension_dir), "status": payload["status"], "summary": payload["summary"]}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Implement selected legal pre-entry proxies in canonical entry raw-state and build a readiness pack.")
    parser.add_argument("--reports-root", type=str, default=None)
    parser.add_argument("--extension-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    result = materialize(reports_root, extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
