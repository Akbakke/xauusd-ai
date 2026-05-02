#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

EXTENSION_PREFIX = "MONDAY_R6_LEGAL_PRE_ENTRY_FEATURE_SPEC_AND_RETRAIN_PREREQS_LOCK_V1"
PRIOR_DIAG_PREFIX = "MONDAY_R6_READONLY_DIAGNOSIS_AND_NEXT_STEP_LOCK_V1_"
BENCHMARK_SNAPSHOT_PREFIX = "MONDAY_WEDNESDAY_BENCHMARK_SNAPSHOT_V1_"

CONTRACT = "contract_v1.json"
ENTRY_LEGALITY = "entry_feature_legality_boundary_lock_v1.csv"
RUNNER_GAP = "runner_protection_signal_gap_spec_v1.csv"
LEGAL_CANDIDATES = "legal_pre_entry_path_context_candidates_v1.csv"
FAMILY_MATRIX = "feature_family_priority_matrix_v1.csv"
PROTECTION_LOCK = "repaired_165_and_runner_pocket_protection_lock_v1.json"
RETRAIN_PREREQS = "retrain_prerequisites_lock_v1.json"
CONTRACT_DELTA = "next_retrain_contract_delta_v1.json"
NEXT_ACTION = "next_agent_action_lock_v1.json"
SUMMARY = "summary_v1.json"
REPORT = "report_v1.md"
MANIFEST = "manifest_v1.json"
STATUS = "status_v1.json"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"

PRIOR_RESULT_RECHECK = "monday_r6_result_recheck_v1.json"
PRIOR_COMPARATOR = "comparator_hierarchy_reference_lock_v1.csv"
PRIOR_REPAIRED = "repaired_165_damage_forensic_v1.json"
PRIOR_GAP_MAP = "failure_backlog_gap_map_v1.csv"
PRIOR_PATH_LOCK = "path_dynamics_bottleneck_lock_v1.csv"

R6_FEATURE_AUDIT = "shadow_meta_all_trade_review_r6_feature_path_dynamics_audit_v1.csv"
FREEZE_OPPORTUNITY_AUDIT = "shadow_meta_all_trade_review_r6_label_feature_opportunity_audit_v1.csv"

BENCHMARK_PATH_SPEC = (
    "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1/"
    "shadow_meta_path_dynamics_instrumentation_spec_v2.json"
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if pd.notna(out) else None


def _load_inputs(reports_root: Path) -> Dict[str, Any]:
    diag_dir = _latest_dir(reports_root, PRIOR_DIAG_PREFIX)
    snapshot_dir = _latest_dir(reports_root, BENCHMARK_SNAPSHOT_PREFIX)
    result_recheck = _load_json(diag_dir / PRIOR_RESULT_RECHECK)
    repaired = _load_json(diag_dir / PRIOR_REPAIRED)
    summary = _load_json(diag_dir / SUMMARY)
    comparator_df = pd.read_csv(diag_dir / PRIOR_COMPARATOR)
    gap_df = pd.read_csv(diag_dir / PRIOR_GAP_MAP)
    path_lock_df = pd.read_csv(diag_dir / PRIOR_PATH_LOCK)
    r6_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1"
    freeze_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_2_SHADOW_FREEZE_AND_R6_FAILURE_BACKLOG_V1"
    feature_audit_df = pd.read_csv(r6_dir / R6_FEATURE_AUDIT)
    opportunity_df = pd.read_csv(freeze_dir / FREEZE_OPPORTUNITY_AUDIT)
    benchmark_path_spec = _load_json(snapshot_dir / BENCHMARK_PATH_SPEC)
    return {
        "diag_dir": diag_dir,
        "snapshot_dir": snapshot_dir,
        "result_recheck": result_recheck,
        "repaired": repaired,
        "summary": summary,
        "comparator_df": comparator_df,
        "gap_df": gap_df,
        "path_lock_df": path_lock_df,
        "feature_audit_df": feature_audit_df,
        "opportunity_df": opportunity_df,
        "benchmark_path_spec": benchmark_path_spec,
    }


def _entry_legality_boundary(path_lock_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in path_lock_df.iterrows():
        rows.append(
            {
                "feature_or_family_v1": str(row["field_name_v1"]),
                "classification_v1": "NOT_LEGAL_FOR_ENTRY",
                "same_trade_leakage_risk_v1": "HIGH",
                "comes_from_management_or_exit_truth_v1": True,
                "requires_derivation_v1": True,
                "why_v1": "Logged from management/exit anchor truth; direct same-trade pre-entry use would leak future path information.",
                "allowed_future_use_v1": "Use only as inspiration for prior-window legal proxies, not as direct entry features.",
            }
        )
    rows.extend(
        [
            {
                "feature_or_family_v1": "prior_window_volatility_expansion_compression_context",
                "classification_v1": "PRE_ENTRY_LEGAL",
                "same_trade_leakage_risk_v1": "LOW",
                "comes_from_management_or_exit_truth_v1": False,
                "requires_derivation_v1": False,
                "why_v1": "Built only from data available before entry from current/prior bars.",
                "allowed_future_use_v1": "Safe direct entry feature family.",
            },
            {
                "feature_or_family_v1": "prior_window_directional_asymmetry_and_impulse_context",
                "classification_v1": "PRE_ENTRY_LEGAL",
                "same_trade_leakage_risk_v1": "LOW",
                "comes_from_management_or_exit_truth_v1": False,
                "requires_derivation_v1": False,
                "why_v1": "Uses prior 15/60/240 move asymmetry and directional imbalance only.",
                "allowed_future_use_v1": "Safe direct entry feature family.",
            },
            {
                "feature_or_family_v1": "swing_retracement_and_structure_context",
                "classification_v1": "PRE_ENTRY_LEGAL",
                "same_trade_leakage_risk_v1": "LOW",
                "comes_from_management_or_exit_truth_v1": False,
                "requires_derivation_v1": False,
                "why_v1": "Uses last swing distances/retracement known at decision time.",
                "allowed_future_use_v1": "Safe direct entry feature family.",
            },
            {
                "feature_or_family_v1": "session_time_and_market_pocket_context",
                "classification_v1": "PRE_ENTRY_LEGAL",
                "same_trade_leakage_risk_v1": "LOW",
                "comes_from_management_or_exit_truth_v1": False,
                "requires_derivation_v1": False,
                "why_v1": "Hour/session/boundary context is available before entry.",
                "allowed_future_use_v1": "Safe direct entry feature family.",
            },
            {
                "feature_or_family_v1": "spread_cost_and_liquidity_pressure_context",
                "classification_v1": "PRE_ENTRY_LEGAL",
                "same_trade_leakage_risk_v1": "LOW",
                "comes_from_management_or_exit_truth_v1": False,
                "requires_derivation_v1": False,
                "why_v1": "Spread/cost observations are known at entry.",
                "allowed_future_use_v1": "Safe direct entry hardening family.",
            },
            {
                "feature_or_family_v1": "entry_model_context_probs_and_uncertainty",
                "classification_v1": "PRE_ENTRY_LEGAL",
                "same_trade_leakage_risk_v1": "LOW",
                "comes_from_management_or_exit_truth_v1": False,
                "requires_derivation_v1": False,
                "why_v1": "These are generated from pre-entry inputs only when computed correctly.",
                "allowed_future_use_v1": "Safe as meta-features if sourced strictly from entry-time models.",
            },
            {
                "feature_or_family_v1": "runner_expectancy_proxy_from_prior_context",
                "classification_v1": "PRE_ENTRY_LEGAL_IF_DERIVED",
                "same_trade_leakage_risk_v1": "MEDIUM_IF_BADLY_DERIVED",
                "comes_from_management_or_exit_truth_v1": False,
                "requires_derivation_v1": True,
                "why_v1": "Must be translated from historical/past-bar context, not from same-trade future MFE path.",
                "allowed_future_use_v1": "Allowed only as a proxy derived from pre-entry regime/path context.",
            },
            {
                "feature_or_family_v1": "adverse_first_risk_proxy_from_pre_entry_context",
                "classification_v1": "PRE_ENTRY_LEGAL_IF_DERIVED",
                "same_trade_leakage_risk_v1": "MEDIUM_IF_BADLY_DERIVED",
                "comes_from_management_or_exit_truth_v1": False,
                "requires_derivation_v1": True,
                "why_v1": "Can be derived from prior volatility/asymmetry/close-in-bar patterns, but not from same-trade MAE order truth.",
                "allowed_future_use_v1": "Allowed only as pre-entry proxy, never by reading management/exit anchor sequence fields directly.",
            },
            {
                "feature_or_family_v1": "tail_leakage_10_50_proxy_from_pre_entry_context",
                "classification_v1": "PRE_ENTRY_LEGAL_IF_DERIVED",
                "same_trade_leakage_risk_v1": "MEDIUM_IF_BADLY_DERIVED",
                "comes_from_management_or_exit_truth_v1": False,
                "requires_derivation_v1": True,
                "why_v1": "Must use pre-entry compression/expansion and path-style context instead of realized same-trade 10–50 MFE outcome truth.",
                "allowed_future_use_v1": "Allowed as a derived pocket-risk proxy only.",
            },
            {
                "feature_or_family_v1": "repaired_165_lineage_flags",
                "classification_v1": "NOT_ESTABLISHED",
                "same_trade_leakage_risk_v1": "LOW",
                "comes_from_management_or_exit_truth_v1": False,
                "requires_derivation_v1": False,
                "why_v1": "Useful for audit/pocket evaluation, but not a robust live market signal to lean on as a core entry feature.",
                "allowed_future_use_v1": "Use as evaluation pocket tag and guardrail, not as main model signal unless proven stable.",
            },
            {
                "feature_or_family_v1": "management_policy_scores_or_decision_log_fields",
                "classification_v1": "NOT_LEGAL_FOR_ENTRY",
                "same_trade_leakage_risk_v1": "HIGH",
                "comes_from_management_or_exit_truth_v1": True,
                "requires_derivation_v1": True,
                "why_v1": "These depend on downstream management-state availability and must not be fed back into entry.",
                "allowed_future_use_v1": "Diagnosis only.",
            },
        ]
    )
    return pd.DataFrame(rows)


def _runner_gap_spec(result_recheck: Dict[str, Any], repaired: Dict[str, Any], gap_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_gap_id_v1": "REPAIRED_165_MODERATE_MAE_RUNNER_SEED",
                "problem_v1": "A profitable repaired-pocket trade with ~96 MFE and +62.95 bps was blocked because bad/risky/tail scores overwhelmed runner protection.",
                "what_must_be_protected_v1": "Repaired-165 profitable seeds that can tolerate moderate MAE before expansion.",
                "missing_signal_v1": "Pre-entry runner expectancy conditioned on moderate-MAE tolerance and continuation setup quality.",
                "likely_solution_v1": "RUNNER_EXPECTANCY_PROXY + explicit repaired-pocket guard evaluation.",
                "priority_v1": "HIGH",
            },
            {
                "signal_gap_id_v1": "RUNNER_NEAR_MISS_POCKET",
                "problem_v1": f"{int(gap_df.loc[gap_df['bucket_id_v1'].eq('RUNNER_NEAR_MISS'), 'count_v1'].iloc[0])} runner near-misses remain.",
                "what_must_be_protected_v1": "50+ MFE seeds and late runners that look noisy early but are legitimate winners.",
                "missing_signal_v1": "Stronger separation between risky setup and late runner seed using legal prior-path and volatility context.",
                "likely_solution_v1": "RUNNER_PROTECTOR_UPLIFT + pocket-aware calibration.",
                "priority_v1": "HIGH",
            },
            {
                "signal_gap_id_v1": "TAIL_10_50_VS_TRUE_RUNNER_SEPARATION",
                "problem_v1": f"{int(gap_df.loc[gap_df['bucket_id_v1'].eq('MISSED_10_50_TAIL_CONTROL'), 'count_v1'].iloc[0])} tail-control misses remain.",
                "what_must_be_protected_v1": "True runners must not be mistaken for low-value 10–50 leakage trades.",
                "missing_signal_v1": "Better pre-entry distinction between compressed runner seed and noisy low-value pocket.",
                "likely_solution_v1": "TAIL_PROXY + runner-first guard preserved.",
                "priority_v1": "HIGH",
            },
            {
                "signal_gap_id_v1": "SHOULD_NOT_TAKE_WITHOUT_RUNNER_COLLISION",
                "problem_v1": f"{int(gap_df.loc[gap_df['bucket_id_v1'].eq('MISSED_SHOULD_NOT_TAKE'), 'count_v1'].iloc[0])} bad trades still slip through.",
                "what_must_be_protected_v1": "Bad-trade recall must go up without damaging runner pockets.",
                "missing_signal_v1": "Harder pre-entry bad-risk separation that respects runner expectancy proxies.",
                "likely_solution_v1": "BAD_RISK_HARDENING + protector-first blocker logic.",
                "priority_v1": "HIGH",
            },
            {
                "signal_gap_id_v1": "RISKY_ALLOW_DISCRIMINATION",
                "problem_v1": f"{int(gap_df.loc[gap_df['bucket_id_v1'].eq('RISKY_ALLOW'), 'count_v1'].iloc[0])} risky allows remain.",
                "what_must_be_protected_v1": "Risky setups must be caught without overblocking legitimate continuation seeds.",
                "missing_signal_v1": "Better pre-entry adverse-first proxy and regime-conditioned risk discrimination.",
                "likely_solution_v1": "ADVERSE_FIRST_PROXY + regime-conditioned thresholding in evaluation.",
                "priority_v1": "MEDIUM_HIGH",
            },
        ]
    )


def _legal_candidates() -> pd.DataFrame:
    rows = [
        {
            "candidate_name_v1": "pre_entry_volatility_expansion_compression_stack_v1",
            "definition_v1": "Combine ATR percentile, range compression, squeeze, and recent range expansion into one pre-entry context score.",
            "legality_v1": "PRE_ENTRY_LEGAL",
            "solves_v1": "missed should-not-take, missed 10-50 tail-control, risky allows",
            "expected_value_v1": "HIGH",
            "leakage_risk_v1": "LOW",
            "implementation_complexity_v1": "MEDIUM",
            "priority_v1": "HIGH",
        },
        {
            "candidate_name_v1": "pre_entry_directional_asymmetry_proxy_v1",
            "definition_v1": "Use 15/60/240 up/down move asymmetry and directional imbalance before entry to estimate whether the setup is clean continuation or noisy trap.",
            "legality_v1": "PRE_ENTRY_LEGAL",
            "solves_v1": "missed should-not-take, risky allows, runner near-misses",
            "expected_value_v1": "HIGH",
            "leakage_risk_v1": "LOW",
            "implementation_complexity_v1": "MEDIUM",
            "priority_v1": "HIGH",
        },
        {
            "candidate_name_v1": "pre_entry_swing_retracement_alignment_v1",
            "definition_v1": "Distance to recent swings plus retracement-from-last-impulse and bars-since-swing as a continuation-vs-exhaustion proxy.",
            "legality_v1": "PRE_ENTRY_LEGAL",
            "solves_v1": "runner near-misses, repaired-165 protection, missed should-not-take",
            "expected_value_v1": "HIGH",
            "leakage_risk_v1": "LOW",
            "implementation_complexity_v1": "MEDIUM",
            "priority_v1": "HIGH",
        },
        {
            "candidate_name_v1": "pre_entry_session_pocket_runner_expectancy_v1",
            "definition_v1": "Session/hour/boundary-aware expectancy proxy that estimates whether this is a late runner seed or a low-value pocket.",
            "legality_v1": "PRE_ENTRY_LEGAL_IF_DERIVED",
            "solves_v1": "runner near-misses, repaired-165 protection, tail-control assistance",
            "expected_value_v1": "MEDIUM_HIGH",
            "leakage_risk_v1": "LOW_MEDIUM",
            "implementation_complexity_v1": "MEDIUM",
            "priority_v1": "HIGH",
        },
        {
            "candidate_name_v1": "pre_entry_adverse_first_risk_proxy_v1",
            "definition_v1": "A derived pre-entry proxy for immediate adverse risk using only prior volatility, wick/clv, spread, and directional asymmetry.",
            "legality_v1": "PRE_ENTRY_LEGAL_IF_DERIVED",
            "solves_v1": "risky allows, missed should-not-take",
            "expected_value_v1": "HIGH",
            "leakage_risk_v1": "MEDIUM_IF_BADLY_DERIVED",
            "implementation_complexity_v1": "MEDIUM_HIGH",
            "priority_v1": "HIGH",
        },
        {
            "candidate_name_v1": "pre_entry_tail_leakage_pocket_proxy_v1",
            "definition_v1": "A legal proxy for the 10–50 MFE leakage pocket using pre-entry range/volatility/path-style context instead of realized same-trade tail labels.",
            "legality_v1": "PRE_ENTRY_LEGAL_IF_DERIVED",
            "solves_v1": "missed 10-50 tail-control",
            "expected_value_v1": "MEDIUM_HIGH",
            "leakage_risk_v1": "MEDIUM_IF_BADLY_DERIVED",
            "implementation_complexity_v1": "MEDIUM_HIGH",
            "priority_v1": "HIGH",
        },
        {
            "candidate_name_v1": "runner_protection_guard_score_v1",
            "definition_v1": "A guard score built from legal pre-entry proxies that can override aggressive blocking when runner expectancy is high.",
            "legality_v1": "PRE_ENTRY_LEGAL_IF_DERIVED",
            "solves_v1": "runner near-misses, repaired-165 protection, 50+ MFE protection",
            "expected_value_v1": "HIGH",
            "leakage_risk_v1": "LOW_MEDIUM",
            "implementation_complexity_v1": "MEDIUM_HIGH",
            "priority_v1": "HIGH",
        },
        {
            "candidate_name_v1": "spread_cost_pressure_hardening_v1",
            "definition_v1": "More explicit spread/cost pressure features to separate weak risky allows from clean runners.",
            "legality_v1": "PRE_ENTRY_LEGAL",
            "solves_v1": "risky allows, should-not-take hardening",
            "expected_value_v1": "MEDIUM",
            "leakage_risk_v1": "LOW",
            "implementation_complexity_v1": "LOW",
            "priority_v1": "MEDIUM",
        },
        {
            "candidate_name_v1": "direct_use_of_management_exit_path_fields_v1",
            "definition_v1": "Directly feed last_peak_ts / last_mfe_ts / max_mfe_without_mae / sequence-order into entry.",
            "legality_v1": "NOT_LEGAL_FOR_ENTRY",
            "solves_v1": "none_safely",
            "expected_value_v1": "UNACCEPTABLE",
            "leakage_risk_v1": "HIGH",
            "implementation_complexity_v1": "LOW",
            "priority_v1": "LOW",
        },
    ]
    return pd.DataFrame(rows)


def _family_priority_matrix() -> pd.DataFrame:
    rows = [
        {
            "feature_family_v1": "regime_context_features",
            "can_help_v1": "Contextualize whether a setup is naturally trend-following, mean-reverting, or expansion-prone.",
            "cannot_help_v1": "Cannot alone distinguish repaired-runner seeds from all bad trades.",
            "legality_status_v1": "PRE_ENTRY_LEGAL",
            "expected_impact_v1": "MEDIUM_HIGH",
            "overfit_risk_v1": "LOW_MEDIUM",
            "prioritize_before_retrain_v1": True,
        },
        {
            "feature_family_v1": "volatility_expansion_compression_context",
            "can_help_v1": "Strongest legal family for should-not-take, risky allow, and tail pocket separation.",
            "cannot_help_v1": "Needs runner-protection pairing; otherwise can still overblock volatile winners.",
            "legality_status_v1": "PRE_ENTRY_LEGAL",
            "expected_impact_v1": "HIGH",
            "overfit_risk_v1": "LOW_MEDIUM",
            "prioritize_before_retrain_v1": True,
        },
        {
            "feature_family_v1": "session_time_pocket_context",
            "can_help_v1": "Adds pocket awareness for late-session seeds and low-value windows.",
            "cannot_help_v1": "Not enough alone to classify runner expectancy.",
            "legality_status_v1": "PRE_ENTRY_LEGAL",
            "expected_impact_v1": "MEDIUM",
            "overfit_risk_v1": "LOW",
            "prioritize_before_retrain_v1": True,
        },
        {
            "feature_family_v1": "historical_path_style_proxies",
            "can_help_v1": "Best legal translation of path-dynamics intuition into entry-time context.",
            "cannot_help_v1": "Cannot use same-trade realized path truth directly.",
            "legality_status_v1": "PRE_ENTRY_LEGAL_IF_DERIVED",
            "expected_impact_v1": "HIGH",
            "overfit_risk_v1": "MEDIUM",
            "prioritize_before_retrain_v1": True,
        },
        {
            "feature_family_v1": "runner_expectancy_proxies",
            "can_help_v1": "Protect repaired pocket, runner near-miss pocket, and 50+ seeds before recall expands.",
            "cannot_help_v1": "Must not become a hidden same-trade runner label.",
            "legality_status_v1": "PRE_ENTRY_LEGAL_IF_DERIVED",
            "expected_impact_v1": "HIGH",
            "overfit_risk_v1": "MEDIUM_HIGH",
            "prioritize_before_retrain_v1": True,
        },
        {
            "feature_family_v1": "risky_allow_discrimination_features",
            "can_help_v1": "Separate truly dangerous setups from late seeds with some heat.",
            "cannot_help_v1": "Without protector-first logic, more aggressive risk features can still hurt runners.",
            "legality_status_v1": "PRE_ENTRY_LEGAL_IF_DERIVED",
            "expected_impact_v1": "HIGH",
            "overfit_risk_v1": "MEDIUM",
            "prioritize_before_retrain_v1": True,
        },
        {
            "feature_family_v1": "should_not_take_hardening_features",
            "can_help_v1": "Raise bad-trade recall using legal volatility/path/regime context.",
            "cannot_help_v1": "Should not override runner-protection pockets blindly.",
            "legality_status_v1": "PRE_ENTRY_LEGAL",
            "expected_impact_v1": "HIGH",
            "overfit_risk_v1": "MEDIUM",
            "prioritize_before_retrain_v1": True,
        },
        {
            "feature_family_v1": "tail_control_assistance_features",
            "can_help_v1": "Improve 10–50 pocket handling while preserving true runners.",
            "cannot_help_v1": "Cannot be derived from realized tail outcome fields directly.",
            "legality_status_v1": "PRE_ENTRY_LEGAL_IF_DERIVED",
            "expected_impact_v1": "MEDIUM_HIGH",
            "overfit_risk_v1": "MEDIUM",
            "prioritize_before_retrain_v1": True,
        },
        {
            "feature_family_v1": "direct_management_exit_anchor_fields",
            "can_help_v1": "Strong diagnosis only.",
            "cannot_help_v1": "Cannot be fed directly into entry.",
            "legality_status_v1": "NOT_LEGAL_FOR_ENTRY",
            "expected_impact_v1": "N_A_FOR_ENTRY",
            "overfit_risk_v1": "LEAKAGE_NOT_OVERFIT",
            "prioritize_before_retrain_v1": False,
        },
    ]
    return pd.DataFrame(rows)


def _protection_lock(repaired: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "layer_name_v1": "REPAIRED_165_AND_RUNNER_POCKET_PROTECTION_LOCK_V1",
        "repaired_165_zero_tolerance_v1": True,
        "runner_damage_hard_limits_v1": {
            "repaired_165_blocked_v1": 0,
            "fifty_plus_mfe_blocked_max_v1": 1,
            "hundred_plus_mfe_blocked_v1": 0,
            "two_hundred_plus_mfe_blocked_v1": 0,
            "strongest_winner_path_damage_v1": 0,
        },
        "why_current_r6_failed_v1": "The current runner guard stayed too weak on a profitable repaired-pocket trade and allowed blocker/risky scores to dominate.",
        "required_design_truths_v1": [
            "Runner-protection must be evaluated before blocker expansion.",
            "Repaired pocket must be re-audited explicitly after every candidate evaluation.",
            "50+/100+/200+ pockets must be audited separately, not only in global metrics.",
            "A candidate that improves recall but breaks repaired-165 is automatically rejected.",
        ],
        "solution_type_v1": "COMBINATION_OF_FEATURE_PROBLEM_POCKET_PROBLEM_AND_GUARD_CALIBRATION",
        "explicit_next_eval_gates_v1": [
            "repaired_165_damage == 0",
            "50+ MFE blocked <= 1",
            "100+ MFE blocked == 0",
            "200+ MFE blocked == 0",
            "strongest_winner_path_damage == 0",
            f"forensic trade {repaired['deterministic_trade_key_v1']} must remain unblocked",
        ],
    }


def _retrain_prereqs() -> Dict[str, Any]:
    return {
        "layer_name_v1": "RETRAIN_PREREQUISITES_LOCK_V1",
        "decision_v1": "READY_FOR_NARROW_IMPLEMENTATION_PHASE",
        "retrain_now_v1": False,
        "why_not_now_v1": [
            "No new legal pre-entry feature uplift has been implemented yet.",
            "Runner-protection uplift is only specified, not built.",
            "Path-dynamics intuition has not yet been translated into canonical pre-entry proxies.",
        ],
        "must_exist_before_next_retrain_v1": [
            "At least one new legal pre-entry feature family implemented.",
            "Runner-protection uplift implemented on canonical pre-entry signals.",
            "Repaired-pocket protection gates wired into evaluation.",
            "Leakage boundary locked and respected in code/tests.",
            "Monday failure pockets translated into feature requirements, not only labels.",
            "Next retrain evaluation contract locked against frozen R6, Monday R5.1, and Monday R6 failure-miner.",
        ],
    }


def _next_retrain_contract_delta(result_recheck: Dict[str, Any], repaired: Dict[str, Any]) -> Dict[str, Any]:
    current = result_recheck["metrics_v1"]
    return {
        "layer_name_v1": "NEXT_RETRAIN_CONTRACT_DELTA_V1",
        "compare_against_v1": [
            "FROZEN_WEDNESDAY_R6_BENCHMARK",
            "MONDAY_R5_1_SAFETY_REFERENCE",
            "MONDAY_R6_FAILURE_MINER",
        ],
        "must_improve_over_monday_r6_v1": {
            "bad_blocks_gt_v1": int(current["bad_blocks_v1"]),
            "tail_help_gt_v1": int(current["tail_help_v1"]),
            "precision_gte_v1": _safe_float(current["precision_v1"]),
            "worst_loso_precision_gte_v1": _safe_float(current["worst_loso_precision_v1"]),
        },
        "must_keep_safe_v1": {
            "repaired_165_damage_v1": 0,
            "fifty_plus_mfe_blocked_max_v1": 1,
            "hundred_plus_mfe_blocked_v1": 0,
            "two_hundred_plus_mfe_blocked_v1": 0,
            "strongest_winner_path_damage_v1": 0,
            "forensic_trade_must_stay_unblocked_v1": repaired["deterministic_trade_key_v1"],
        },
        "explicit_pockets_to_monitor_v1": [
            "repaired_165_pocket",
            "runner_near_miss_pocket",
            "10_50_tail_control_pocket",
            "BATCH_04",
            "BATCH_05_if_present",
        ],
        "benchmark_direction_v1": {
            "bad_blocks_target_v1": 180,
            "tail_help_target_v1": 149,
            "precision_target_v1": 0.972972972972973,
            "worst_loso_target_v1": 0.9285714285714286,
        },
        "note_v1": "Beating Monday-native R6 alone is insufficient; the candidate must show a credible approach toward frozen R6 without safety loss.",
    }


def _next_action_lock() -> Dict[str, Any]:
    return {
        "layer_name_v1": "NEXT_AGENT_ACTION_LOCK_V1",
        "primary_action_v1": "TRANSLATE_PATH_DYNAMICS_TO_LEGAL_PRE_ENTRY_PROXIES",
        "supporting_actions_v1": [
            "DO_NOT_RETRAIN_YET",
            "IMPLEMENT_RUNNER_PROTECTION_UPLIFT_FIRST",
            "KEEP_MONDAY_R6_AS_FAILURE_MINER",
            "DO_NOT_USE_MANAGEMENT_EXIT_FIELDS_DIRECTLY",
            "ONLY_AFTER_THIS_START_NARROW_FEATURE_IMPLEMENTATION",
        ],
    }


def _status_block() -> Dict[str, Any]:
    return {
        "layer_name_v1": "STATUS_DISCIPLINE_V1",
        "BEVIST": [
            "Monday-native R6 does not hold the benchmark contract.",
            "Runner-protection/path-context is a real bottleneck.",
            "Management/exit-anchor fields are not automatically legal entry features.",
            "A new retrain must not start now.",
        ],
        "INDIKERT": [
            "Legal pre-entry proxies are the right translation path forward.",
            "Feature uplift is more promising than blind retrain.",
            "Repaired-165 and runner near-miss pockets require explicit protection design.",
        ],
        "IKKE_ETABLERT": [
            "That the proposed feature candidates will beat frozen R6.",
            "That the current Monday lane alone has enough signal for a new freeze.",
            "That path-dynamics intuition can be translated into stable entry-features without further iteration.",
        ],
    }


def _render_report(
    legality_df: pd.DataFrame,
    runner_gap_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    family_df: pd.DataFrame,
    prereqs: Dict[str, Any],
    next_action: Dict[str, Any],
    status_block: Dict[str, Any],
) -> str:
    lines = [
        "# Monday R6 Legal Pre-Entry Feature Spec And Retrain Prereqs Lock V1",
        "",
        "Read-only feature/legality spec. No retrain or replay was started.",
        "",
        "## Headline",
        "",
        f"- Retrain readiness: `{prereqs['decision_v1']}`",
        f"- Primary next action: `{next_action['primary_action_v1']}`",
        "",
        "## Entry Legality",
        "",
    ]
    for row in legality_df.to_dict(orient="records"):
        lines.append(f"- `{row['feature_or_family_v1']}` -> `{row['classification_v1']}`")
    lines += [
        "",
        "## Runner Gap",
        "",
    ]
    for row in runner_gap_df.to_dict(orient="records"):
        lines.append(f"- `{row['signal_gap_id_v1']}`: {row['missing_signal_v1']}")
    lines += [
        "",
        "## Highest-Priority Legal Candidates",
        "",
    ]
    for row in candidates_df[candidates_df["priority_v1"].astype("string").eq("HIGH")].to_dict(orient="records"):
        lines.append(f"- `{row['candidate_name_v1']}`: {row['definition_v1']}")
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


def _audit_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def build_payload(reports_root: Path, extension_dir: Path) -> Dict[str, Any]:
    inputs = _load_inputs(reports_root)
    legality_df = _entry_legality_boundary(inputs["path_lock_df"])
    runner_gap_df = _runner_gap_spec(inputs["result_recheck"], inputs["repaired"], inputs["gap_df"])
    candidates_df = _legal_candidates()
    family_df = _family_priority_matrix()
    protection_lock = _protection_lock(inputs["repaired"])
    prereqs = _retrain_prereqs()
    contract_delta = _next_retrain_contract_delta(inputs["result_recheck"], inputs["repaired"])
    next_action = _next_action_lock()
    status_block = _status_block()
    contract = {
        "layer_name_v1": "MONDAY_R6_LEGAL_PRE_ENTRY_FEATURE_SPEC_CONTRACT_V1",
        "mode_v1": "READ_ONLY_SPEC_ONLY",
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_promotion_v1": True,
        "inputs_v1": {
            "prior_diagnosis_dir_v1": str(inputs["diag_dir"]),
            "benchmark_snapshot_dir_v1": str(inputs["snapshot_dir"]),
            "r6_feature_audit_rows_v1": int(len(inputs["feature_audit_df"])),
            "freeze_opportunity_rows_v1": int(len(inputs["opportunity_df"])),
        },
    }
    consistency_df = pd.DataFrame(
        [
            _audit_record("PRIOR_DIAGNOSIS_PRESENT", "PASS", {"dir": str(inputs["diag_dir"])}),
            _audit_record("BENCHMARK_SNAPSHOT_PRESENT", "PASS", {"dir": str(inputs["snapshot_dir"])}),
            _audit_record("RESULT_RECHECK_LOCKED", "PASS" if inputs["result_recheck"]["verdict_v1"] == "R6_FEATURES_INSUFFICIENT" else "FAIL", {"verdict": inputs["result_recheck"]["verdict_v1"]}),
            _audit_record("REPAIRED_FORENSIC_LOCKED", "PASS" if inputs["repaired"]["take_was_ok_v1"] and not inputs["repaired"]["label_should_not_take_v1"] else "FAIL", {"forensic": inputs["repaired"]}),
            _audit_record("PATH_FIELDS_NOT_ENTRY_LEGAL", "PASS" if legality_df["classification_v1"].astype("string").eq("NOT_LEGAL_FOR_ENTRY").sum() >= 5 else "FAIL", {"not_legal_count": int(legality_df["classification_v1"].astype("string").eq("NOT_LEGAL_FOR_ENTRY").sum())}),
            _audit_record("RETRAIN_NOT_STARTED", "PASS", {"decision": prereqs["decision_v1"], "retrain_now": prereqs["retrain_now_v1"]}),
        ]
    )
    status = {
        "layer_name_v1": "MONDAY_R6_LEGAL_PRE_ENTRY_FEATURE_SPEC_STATUS_V1",
        "SPEC_STATUS": "MATERIALIZED_READ_ONLY",
        "failed_check_count_v1": int(consistency_df["status_v1"].eq("FAIL").sum()),
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_promotion_v1": True,
    }
    summary = {
        "layer_name_v1": "MONDAY_R6_LEGAL_PRE_ENTRY_FEATURE_SPEC_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "extension_dir_v1": str(extension_dir),
        "benchmark_lock_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
        "safety_reference_v1": "R5_1_CANDIDATE_0241_R5_1_COMBINED_repaired_165_like",
        "monday_r6_role_v1": "FAILURE_MINER_DIAGNOSIS_ONLY",
        "retrain_readiness_v1": prereqs["decision_v1"],
        "next_action_v1": next_action["primary_action_v1"],
        "status_v1": status,
        "hard_status_division_v1": status_block,
    }
    manifest = {
        "layer_name_v1": "MONDAY_R6_LEGAL_PRE_ENTRY_FEATURE_SPEC_MANIFEST_V1",
        "artifacts_v1": {
            "contract": CONTRACT,
            "entry_feature_legality_boundary_lock": ENTRY_LEGALITY,
            "runner_protection_signal_gap_spec": RUNNER_GAP,
            "legal_pre_entry_path_context_candidates": LEGAL_CANDIDATES,
            "feature_family_priority_matrix": FAMILY_MATRIX,
            "repaired_165_and_runner_pocket_protection_lock": PROTECTION_LOCK,
            "retrain_prerequisites_lock": RETRAIN_PREREQS,
            "next_retrain_contract_delta": CONTRACT_DELTA,
            "next_agent_action_lock": NEXT_ACTION,
            "summary": SUMMARY,
            "report": REPORT,
            "manifest": MANIFEST,
            "status": STATUS,
            "consistency_audit": CONSISTENCY_AUDIT,
        }
    }
    return {
        "contract": contract,
        "legality_df": legality_df,
        "runner_gap_df": runner_gap_df,
        "candidates_df": candidates_df,
        "family_df": family_df,
        "protection_lock": protection_lock,
        "prereqs": prereqs,
        "contract_delta": contract_delta,
        "next_action": next_action,
        "summary": summary,
        "status": status,
        "manifest": manifest,
        "consistency_df": consistency_df,
        "report": _render_report(legality_df, runner_gap_df, candidates_df, family_df, prereqs, next_action, status_block),
    }


def materialize(reports_root: Path, *, extension_dir: Path | None = None) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    extension_dir = _resolve_extension_dir(reports_root, str(extension_dir) if extension_dir else None)
    extension_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(reports_root, extension_dir)
    _write_json(extension_dir / CONTRACT, payload["contract"])
    payload["legality_df"].to_csv(extension_dir / ENTRY_LEGALITY, index=False)
    payload["runner_gap_df"].to_csv(extension_dir / RUNNER_GAP, index=False)
    payload["candidates_df"].to_csv(extension_dir / LEGAL_CANDIDATES, index=False)
    payload["family_df"].to_csv(extension_dir / FAMILY_MATRIX, index=False)
    _write_json(extension_dir / PROTECTION_LOCK, payload["protection_lock"])
    _write_json(extension_dir / RETRAIN_PREREQS, payload["prereqs"])
    _write_json(extension_dir / CONTRACT_DELTA, payload["contract_delta"])
    _write_json(extension_dir / NEXT_ACTION, payload["next_action"])
    _write_json(extension_dir / SUMMARY, payload["summary"])
    (extension_dir / REPORT).write_text(payload["report"], encoding="utf-8")
    _write_json(extension_dir / MANIFEST, payload["manifest"])
    _write_json(extension_dir / STATUS, payload["status"])
    payload["consistency_df"].to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)
    return {"extension_dir": str(extension_dir), "status": payload["status"], "summary": payload["summary"]}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize legal pre-entry feature spec and retrain prerequisites after Monday-native R6 diagnosis.")
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
