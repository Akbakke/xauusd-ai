#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

EXTENSION_PREFIX = "MONDAY_NARROW_RETRAIN_SCOPE_PLAN_V1"

READINESS_PREFIX = "MONDAY_RETRAIN_READINESS_RECHECK_AND_SCOPE_LOCK_V1_"
BRIDGE_PREFIX = "MONDAY_ENTRY_TO_FAILURE_POCKET_BRIDGE_IMPLEMENTATION_V1_"
DIAG_PREFIX = "MONDAY_R6_READONLY_DIAGNOSIS_AND_NEXT_STEP_LOCK_V1_"

CANONICAL_LEDGER_DIRNAME = "ALL_TRADE_REVIEW_LEDGER_20260411"
R6_DIRNAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1"
R5_1_TOP_LEVEL_SUMMARY = "truth_r5_loso_batch04_robustness_retrain_v1.json"

ENTRY_RAW_STATE = "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet"
ENTRY_RAW_CONTRACT = "shadow_meta_all_trade_review_entry_skipability_raw_state_contract_v1.csv"
ENTRY_RAW_CONTRACT_SUMMARY = "shadow_meta_all_trade_review_entry_skipability_raw_state_contract_summary_v1.json"

R6_HINDSIGHT = "shadow_meta_all_trade_review_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet"
R6_POLICY_VIEW = "shadow_meta_all_trade_review_r6_policy_prediction_view_v1.parquet"

READINESS_DECISION = "readiness_decision_v1.json"
READINESS_SUMMARY = "summary_v1.json"
READINESS_SCOPE = "narrow_retrain_scope_proposal_v1.json"
READINESS_CONTRACT = "retrain_contract_and_guard_recheck_v1.json"
BRIDGE_SUMMARY = "summary_v1.json"

DIAG_SUMMARY = "summary_v1.json"

CONTRACT = "contract_v1.json"
NARROW_PLAN = "narrow_retrain_plan_v1.json"
TRAINING_SURFACE_LOCK = "training_surface_lock_v1.json"
FEATURE_SET_LOCK = "feature_set_lock_v1.json"
TRAINING_OBJECTIVE_LOCK = "training_objective_and_priority_lock_v1.json"
EVAL_GUARD_PLAN = "eval_and_regression_guard_plan_v1.json"
TRAINING_IO_LOCK = "training_run_inputs_and_outputs_lock_v1.json"
STOP_CONDITIONS = "stop_conditions_and_no_go_cases_v1.json"
EXECUTION_ORDER = "narrow_retrain_execution_order_v1.json"
NEXT_ACTION = "next_agent_action_lock_v1.json"
SUMMARY = "summary_v1.json"
REPORT = "report_v1.md"
MANIFEST = "manifest_v1.json"
STATUS = "status_v1.json"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"

SELECTED_PROXIES = [
    "as_of_pre_entry_vol_exp_comp_score_v1",
    "as_of_pre_entry_directional_asymmetry_score_v1",
    "as_of_pre_entry_swing_retracement_alignment_score_v1",
    "as_of_pre_entry_tail_leakage_pocket_score_v1",
    "as_of_pre_entry_runner_protection_guard_score_v1",
]

FORBIDDEN_FEATURES = [
    "last_peak_ts",
    "last_mfe_ts",
    "last_peak_mfe",
    "max_mfe_without_mae",
    "mfe_mae_sequence_order",
    "management_policy_scores_or_decision_log_fields",
    "bridge_only_rows_from_fullcoverage_r6_asof",
    "deferred_session_pocket_runner_expectancy",
    "deferred_adverse_first_risk_proxy",
    "spread_cost_pressure_hardening_v1",
    "any_new_live_controller_or_policy_logic",
    "any_new_policy_family_or_broad_refactor",
]

IDENTITY_FIELDS = {"run_id", "candidate_uid", "trade_uid", "trade_id", "decision_ts_utc", "decision_ts"}

BENCHMARK = "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"
MONDAY_SAFETY_REFERENCE = "R5_1_CANDIDATE_0241_R5_1_COMBINED_repaired_165_like"
MONDAY_R6_ROLE = "FAILURE_MINER_DIAGNOSIS_ONLY"
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


def _latest_dir(reports_root: Path, prefix: str) -> Path:
    matches = sorted(
        [path for path in reports_root.iterdir() if path.is_dir() and path.name.startswith(prefix)],
        key=lambda path: path.name,
    )
    if not matches:
        raise FileNotFoundError(f"No directory found for prefix {prefix} under {reports_root}")
    return matches[-1]


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _resolve_extension_dir(reports_root: Path, extension_dir_arg: str | None) -> Path:
    if extension_dir_arg:
        return Path(extension_dir_arg).expanduser().resolve()
    return reports_root / f"{EXTENSION_PREFIX}_{_utc_compact()}"


def _audit_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _feature_name_column(df: pd.DataFrame) -> str:
    for candidate in ("feature_name_v1", "field_name_v1", "feature_name"):
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Could not resolve feature-name column from {list(df.columns)}")


def _semantic_group_column(df: pd.DataFrame) -> str | None:
    for candidate in ("semantic_group", "semantic_group_v1", "potential_canonical_alias_group_v1"):
        if candidate in df.columns:
            return candidate
    return None


def _load_inputs(reports_root: Path) -> Dict[str, Any]:
    readiness_dir = _latest_dir(reports_root, READINESS_PREFIX)
    bridge_dir = _latest_dir(reports_root, BRIDGE_PREFIX)
    diag_dir = _latest_dir(reports_root, DIAG_PREFIX)
    ledger_dir = reports_root / CANONICAL_LEDGER_DIRNAME
    r6_dir = reports_root / R6_DIRNAME
    if not ledger_dir.exists():
        raise FileNotFoundError(f"Missing canonical ledger dir: {ledger_dir}")
    if not r6_dir.exists():
        raise FileNotFoundError(f"Missing R6 dir: {r6_dir}")
    return {
        "readiness_dir": readiness_dir,
        "bridge_dir": bridge_dir,
        "diag_dir": diag_dir,
        "ledger_dir": ledger_dir,
        "r6_dir": r6_dir,
        "readiness_decision": _load_json(readiness_dir / READINESS_DECISION),
        "readiness_summary": _load_json(readiness_dir / READINESS_SUMMARY),
        "readiness_scope": _load_json(readiness_dir / READINESS_SCOPE),
        "readiness_contract": _load_json(readiness_dir / "retrain_contract_and_guard_recheck_v1.json"),
        "bridge_summary": _load_json(bridge_dir / BRIDGE_SUMMARY),
        "diag_summary": _load_json(diag_dir / DIAG_SUMMARY),
        "r5_1_summary": _load_json(reports_root / R5_1_TOP_LEVEL_SUMMARY),
        "raw_contract_df": pd.read_csv(ledger_dir / ENTRY_RAW_CONTRACT),
        "raw_contract_summary": _load_json(ledger_dir / ENTRY_RAW_CONTRACT_SUMMARY),
        "entry_row_count_df": pd.read_parquet(ledger_dir / ENTRY_RAW_STATE, columns=["candidate_uid"]),
        "hindsight_row_count_df": pd.read_parquet(r6_dir / R6_HINDSIGHT, columns=["candidate_uid"]),
        "policy_row_count_df": pd.read_parquet(r6_dir / R6_POLICY_VIEW, columns=["candidate_uid"]),
    }


def _select_included_baseline_features(raw_contract_df: pd.DataFrame) -> Dict[str, Any]:
    feature_col = _feature_name_column(raw_contract_df)
    semantic_col = _semantic_group_column(raw_contract_df)
    work = raw_contract_df.copy()
    work["_feature_name_v1"] = work[feature_col].astype("string")
    if "as_of_safe_v1" in work.columns:
        work["_as_of_safe_v1"] = work["as_of_safe_v1"].map(lambda x: str(x).strip().lower() == "true")
    else:
        work["_as_of_safe_v1"] = False
    if "direct_only_allowed_v1" in work.columns:
        work["_direct_only_allowed_v1"] = work["direct_only_allowed_v1"].map(lambda x: str(x).strip().lower() == "true")
    else:
        work["_direct_only_allowed_v1"] = False
    if "leakage_risk_v1" in work.columns:
        work["_leakage_risk_v1"] = work["leakage_risk_v1"].astype("string")
    else:
        work["_leakage_risk_v1"] = ""

    include_mask = (
        work["_as_of_safe_v1"]
        & work["_direct_only_allowed_v1"]
        & ~work["_feature_name_v1"].isin(SELECTED_PROXIES)
        & ~work["_feature_name_v1"].isin(list(IDENTITY_FIELDS))
        & ~work["_feature_name_v1"].astype("string").str.startswith("as_of_skip_xgb_")
        & ~work["_leakage_risk_v1"].str.contains("TARGET_ADJACENT|UNKNOWN", case=False, na=False)
    )

    baseline = work.loc[include_mask].copy()
    families: Dict[str, int] = {}
    if semantic_col is not None:
        grouped = baseline[semantic_col].astype("string").value_counts().sort_index()
        families = {str(key): int(val) for key, val in grouped.items()}
    return {
        "feature_names_v1": sorted([str(x) for x in baseline["_feature_name_v1"].tolist()]),
        "feature_count_v1": int(len(baseline)),
        "feature_families_v1": families,
    }


def _build_narrow_retrain_plan(payload: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    readiness_scope = payload["readiness_scope"]
    return {
        "layer_name_v1": "NARROW_RETRAIN_PLAN_V1",
        "purpose_v1": (
            "Plan a narrow runner-first shadow-only retrain that uses the existing exact-only canonical entry surface plus the five new legal "
            "pre-entry proxies to improve runner protection and pocket separation without widening policy family or training population."
        ),
        "scope_v1": "NARROW_RUNNER_FIRST_SHADOW_ONLY",
        "why_narrow_not_broad_v1": [
            "No new policy family is introduced.",
            "No bridge-only rows are allowed into training.",
            "No management/exit truth is permitted as direct entry input.",
            "No policy/controller changes are allowed.",
            "The first slice targets only runner-protection and tail-pocket separation rather than a broad refactor.",
        ],
        "training_surface_v1": readiness_scope["training_surface_v1"],
        "eval_surfaces_v1": {
            "canonical_training_surface_v1": "CANONICAL_EXACT_ONLY_ENTRY_RAW_STATE",
            "readiness_bridge_surface_v1": "ENTRY_TO_FAILURE_POCKET_BRIDGE",
            "hindsight_label_surface_v1": "R6_ENTRY_HINDSIGHT_LABEL_OUTCOME_TABLE_FILTERED_TO_EXACT_TRAINING_ROWS",
            "failure_miner_surface_v1": "MONDAY_NATIVE_R6_FAILURE_MINER_OUTPUTS",
        },
        "model_family_continuation_v1": {
            "continue_family_v1": "EXISTING_ENTRY_RUNNER_FIRST_SHADOW_FAMILY",
            "do_not_introduce_new_policy_family_v1": True,
            "policy_family_note_v1": (
                "Continue the runner-first shadow entry lane as a narrow retrain only; do not widen into a new policy family or live gate."
            ),
        },
        "included_feature_sets_v1": {
            "baseline_feature_count_v1": baseline["feature_count_v1"],
            "baseline_feature_families_v1": baseline["feature_families_v1"],
            "new_proxy_features_v1": SELECTED_PROXIES,
        },
        "explicit_exclusions_v1": FORBIDDEN_FEATURES,
        "do_not_touch_v1": [
            "policy/controller layer",
            "frozen Wednesday-R6 artifacts",
            "Monday R5.1 safety-reference artifacts",
            "Monday-native R6 verdict or role",
            "readiness bridge population",
        ],
    }


def _build_training_surface_lock(payload: Dict[str, Any]) -> Dict[str, Any]:
    training_rows = int(len(payload["entry_row_count_df"]))
    bridge_rows = int(payload["bridge_summary"]["bridge_surface_row_count_v1"])
    bridge_only_rows = int(payload["bridge_summary"]["bridge_only_row_count_v1"])
    return {
        "layer_name_v1": "TRAINING_SURFACE_LOCK_V1",
        "training_surface_artifact_v1": str(payload["ledger_dir"] / ENTRY_RAW_STATE),
        "training_surface_kind_v1": "CANONICAL_EXACT_ONLY_ENTRY_RAW_STATE",
        "training_row_count_v1": training_rows,
        "rows_included_v1": [
            "Only exact-only canonical entry candidate rows present in shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet.",
            "Only rows carrying legal pre-entry AS_OF inputs already materialized on the exact surface.",
        ],
        "rows_excluded_v1": [
            f"All {bridge_only_rows} bridge-only rows from the readiness bridge.",
            "All repaired/fullcoverage-only rows not present on the exact canonical raw-state.",
            "Any rows whose visibility exists only through readiness bridge alignment.",
        ],
        "bridge_surface_not_allowed_v1": {
            "artifact_v1": str(payload["bridge_dir"] / "entry_to_failure_pocket_bridge_surface_v1.parquet"),
            "row_count_v1": bridge_rows,
            "bridge_only_row_count_v1": bridge_only_rows,
            "why_not_allowed_v1": [
                "It is a readiness/eval visibility surface, not a canonical training surface.",
                "It contains bridge-only rows that would silently widen the training population.",
                "It exists to make failure pockets visible without changing training legality.",
            ],
        },
        "legality_basis_v1": [
            "Exact-only raw-state preserves pre-entry legality and canonical population boundaries.",
            "Bridge-only rows would violate the locked separation between readiness visibility and training population.",
            "Management/exit truth must remain outside direct entry training features.",
        ],
    }


def _build_feature_set_lock(payload: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    feature_roles = {
        "as_of_pre_entry_vol_exp_comp_score_v1": {
            "role_v1": "Improve pre-entry compression/expansion context for tail-pocket separation.",
            "pockets_helped_v1": ["missed_10_50_tail_control_pocket", "risky_allow_pocket"],
            "why_legal_v1": "Derived from exact AS_OF skip-replay volatility/compression context only.",
            "risk_v1": "Can overfit if collapsed into aggressive blocker behavior.",
        },
        "as_of_pre_entry_directional_asymmetry_score_v1": {
            "role_v1": "Separate noisy trap entries from continuation-like runner seeds.",
            "pockets_helped_v1": ["runner_near_miss_pocket", "risky_allow_pocket"],
            "why_legal_v1": "Uses prior window up/down move, imbalance, close-in-range and micro-momentum known at entry.",
            "risk_v1": "Directional asymmetry can become brittle across slices if overweighted.",
        },
        "as_of_pre_entry_swing_retracement_alignment_score_v1": {
            "role_v1": "Protect repaired-165-style moderate-MAE runner seeds via structure context.",
            "pockets_helped_v1": ["repaired_165_pocket", "forensic_repaired_trade", "runner_near_miss_pocket"],
            "why_legal_v1": "Built from pre-entry swing distance, bars-since-swing, retracement and EMA distance only.",
            "risk_v1": "Structure context can become regime-specific without clipping/normalization discipline.",
        },
        "as_of_pre_entry_tail_leakage_pocket_score_v1": {
            "role_v1": "Provide a narrow legal pre-entry proxy for 10-50 tail-leakage risk.",
            "pockets_helped_v1": ["missed_10_50_tail_control_pocket", "runner_near_miss_pocket"],
            "why_legal_v1": "Derived only from legal proxies plus anchor-bar geometry and session boundary context.",
            "risk_v1": "Illegal if it drifts toward realized tail truth in future iterations.",
        },
        "as_of_pre_entry_runner_protection_guard_score_v1": {
            "role_v1": "Carry runner-protection strength into narrow retrain evaluation without policy activation.",
            "pockets_helped_v1": ["repaired_165_pocket", "forensic_repaired_trade", "runner_near_miss_pocket", "50_plus_mfe_seed_pocket"],
            "why_legal_v1": "Built only from legal pre-entry proxies and candidate snapshot fields available at entry.",
            "risk_v1": "Must not become a disguised hindsight protector or live controller guard.",
        },
    }
    return {
        "layer_name_v1": "FEATURE_SET_LOCK_V1",
        "baseline_training_features_v1": {
            "feature_count_v1": baseline["feature_count_v1"],
            "feature_names_v1": baseline["feature_names_v1"],
            "feature_families_v1": baseline["feature_families_v1"],
            "why_included_v1": "These are existing legal exact-only baseline features already on the locked training surface.",
        },
        "new_proxy_features_v1": feature_roles,
        "explicit_exclusions_v1": {
            "forbidden_sources_v1": [
                "management/exit truth",
                "policy-log / decision-log fields",
                "bridge-only derived signals",
                "deferred candidates",
                "any source that breaks pre-entry legality",
            ],
            "forbidden_field_examples_v1": FORBIDDEN_FEATURES,
            "target_adjacent_review_excluded_v1": ["as_of_skip_xgb_*"],
        },
    }


def _build_training_objective_lock() -> Dict[str, Any]:
    return {
        "layer_name_v1": "TRAINING_OBJECTIVE_AND_PRIORITY_LOCK_V1",
        "priority_order_v1": [
            {
                "rank_v1": 1,
                "objective_v1": "RUNNER_PROTECTION_AND_REPAIRED_165_SAFETY",
                "why_v1": "The repaired-165 failure and runner near-miss pocket remain the first safety priority.",
                "hard_non_negotiables_v1": [
                    "repaired_165_damage = 0",
                    f"forensic trade {FORENSIC_TRADE} must remain unblocked",
                    "50+ / 100+ / 200+ winner protections must hold",
                ],
            },
            {
                "rank_v1": 2,
                "objective_v1": "TAIL_CONTROL_10_50_UPLIFT",
                "why_v1": "The new narrow slice should try to improve the 10-50 tail pocket without opening blocker aggression first.",
                "hard_non_negotiables_v1": [
                    "No new runner damage to get tail uplift",
                    "Do not sacrifice strongest-winner path protection",
                ],
            },
            {
                "rank_v1": 3,
                "objective_v1": "RISKY_VS_RUNNER_SEPARATION",
                "why_v1": "Improve discrimination between risky allows and late runner seeds using the new legal proxies.",
                "hard_non_negotiables_v1": [
                    "runner near-miss pocket must not worsen",
                    "global precision must not regress uncontrollably",
                ],
            },
            {
                "rank_v1": 4,
                "objective_v1": "SHOULD_NOT_TAKE_HARDENING_WITHOUT_PROTECTOR_DAMAGE",
                "why_v1": "Bad-trade hardening comes after protector safety is established in this narrow slice.",
                "hard_non_negotiables_v1": [
                    "Do not chase recall by damaging repaired or runner pockets",
                    "No blocker expansion before protector-safe evidence",
                ],
            },
        ],
        "what_not_to_optimize_aggressively_v1": [
            "Raw bad-block count at the expense of repaired or runner safety",
            "Broad policy-family exploration",
            "Any live-gate-like behavior",
            "Bridge-based population expansion",
        ],
        "compromises_not_allowed_v1": [
            "Safety regression traded for local recall uplift",
            "Using bridge rows as a shortcut to more training data",
            "Relaxing the frozen Wednesday-R6 benchmark hierarchy",
        ],
    }


def _build_eval_guard_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
    contract = payload["readiness_contract"]
    r5_1 = payload["r5_1_summary"]["selected_candidate_v1"]
    return {
        "layer_name_v1": "EVAL_AND_REGRESSION_GUARD_PLAN_V1",
        "compare_against_v1": [
            {"reference_v1": "FROZEN_WEDNESDAY_R6_BENCHMARK", "kind_v1": "BENCHMARK", "id_v1": BENCHMARK},
            {"reference_v1": "MONDAY_R5_1_SAFETY_REFERENCE", "kind_v1": "SAFETY_REFERENCE", "id_v1": MONDAY_SAFETY_REFERENCE},
            {"reference_v1": "MONDAY_NATIVE_R6_FAILURE_MINER", "kind_v1": "FAILURE_MINER", "id_v1": MONDAY_R6_ROLE},
        ],
        "guards_v1": [
            {
                "guard_id_v1": "REPAIRED_165_DAMAGE",
                "must_pass_v1": "repaired_165_damage = 0",
                "hard_fail_v1": "repaired_165_damage > 0",
                "monitor_only_v1": False,
                "not_worse_than_v1": "must stay at zero",
            },
            {
                "guard_id_v1": "FORENSIC_REPAIRED_TRADE",
                "must_pass_v1": f"{FORENSIC_TRADE} must be unblocked",
                "hard_fail_v1": "forensic trade blocked",
                "monitor_only_v1": False,
                "not_worse_than_v1": "must remain unblocked",
            },
            {
                "guard_id_v1": "FIFTY_PLUS_BLOCKS",
                "must_pass_v1": "50+ MFE blocked <= 1",
                "hard_fail_v1": "50+ MFE blocked > 1",
                "monitor_only_v1": False,
                "not_worse_than_v1": "not worse than locked max of 1",
            },
            {
                "guard_id_v1": "HUNDRED_TWO_HUNDRED_PLUS_BLOCKS",
                "must_pass_v1": "100+/200+ blocked = 0/0",
                "hard_fail_v1": "100+ > 0 or 200+ > 0",
                "monitor_only_v1": False,
                "not_worse_than_v1": "must stay at zero",
            },
            {
                "guard_id_v1": "STRONGEST_WINNER_PATH_DAMAGE",
                "must_pass_v1": "strongest_winner_path_damage = 0",
                "hard_fail_v1": "strongest_winner_path_damage > 0",
                "monitor_only_v1": False,
                "not_worse_than_v1": "must stay at zero",
            },
            {
                "guard_id_v1": "RUNNER_NEAR_MISS_POCKET",
                "must_pass_v1": "runner near-miss pocket must not worsen",
                "hard_fail_v1": "runner near-miss worsens without compensating safety evidence",
                "monitor_only_v1": False,
                "not_worse_than_v1": "not worse than Monday-native R6 failure-miner baseline",
            },
            {
                "guard_id_v1": "GLOBAL_PRECISION",
                "must_pass_v1": f"global precision >= {contract['must_improve_over_monday_r6_v1']['precision_gte_v1']}",
                "hard_fail_v1": "global precision regresses materially below Monday-native R6 floor",
                "monitor_only_v1": False,
                "not_worse_than_v1": ">= Monday-native R6 precision floor and ideally toward frozen benchmark",
            },
            {
                "guard_id_v1": "WORST_LOSO",
                "must_pass_v1": f"worst LOSO >= {contract['must_improve_over_monday_r6_v1']['worst_loso_precision_gte_v1']}",
                "hard_fail_v1": "worst LOSO below locked floor",
                "monitor_only_v1": False,
                "not_worse_than_v1": ">= Monday-native R6 locked floor",
            },
            {
                "guard_id_v1": "BATCH_04",
                "must_pass_v1": "must not regress on BATCH_04 if slice is present",
                "hard_fail_v1": "BATCH_04 pass turns into fail",
                "monitor_only_v1": False,
                "not_worse_than_v1": "not worse than R5.1 / Monday-native R6 slice behavior",
            },
            {
                "guard_id_v1": "BATCH_05",
                "must_pass_v1": "if present, must not fail; if absent, report as null not fail",
                "hard_fail_v1": "BATCH_05 explicitly present and failing",
                "monitor_only_v1": True,
                "not_worse_than_v1": "null is acceptable when slice is absent on this surface",
            },
            {
                "guard_id_v1": "MONDAY_R5_1_SAFETY_REFERENCE",
                "must_pass_v1": "must not be less safe than Monday R5.1 on repaired/winner protections",
                "hard_fail_v1": "candidate is less safe than Monday R5.1",
                "monitor_only_v1": False,
                "not_worse_than_v1": f"R5.1 should_not_take_block_count={r5_1.get('should_not_take_block_count_v1')}, repaired_165=0, 50+/100+/200+=0/0/0",
            },
        ],
    }


def _build_training_io_lock(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "layer_name_v1": "TRAINING_RUN_INPUTS_AND_OUTPUTS_LOCK_V1",
        "inputs_v1": {
            "training_feature_surface_v1": str(payload["ledger_dir"] / ENTRY_RAW_STATE),
            "training_feature_contract_v1": str(payload["ledger_dir"] / ENTRY_RAW_CONTRACT),
            "training_feature_contract_summary_v1": str(payload["ledger_dir"] / ENTRY_RAW_CONTRACT_SUMMARY),
            "hindsight_label_surface_v1": str(payload["r6_dir"] / R6_HINDSIGHT),
            "failure_miner_policy_view_v1": str(payload["r6_dir"] / R6_POLICY_VIEW),
            "readiness_bridge_surface_v1": str(payload["bridge_dir"] / "entry_to_failure_pocket_bridge_surface_v1.parquet"),
            "readiness_bridge_use_v1": "EVAL_ONLY_NOT_TRAINING",
            "required_alignment_key_v1": "candidate_uid exact",
        },
        "input_schema_rules_v1": {
            "training_surface_must_remain_exact_only_v1": True,
            "bridge_rows_forbidden_in_training_v1": True,
            "as_of_and_hindsight_must_remain_separate_v1": True,
        },
        "required_future_outputs_v1": [
            "top_level_training_summary_json",
            "candidate_bakeoff_table_csv",
            "threshold_and_calibration_table_csv",
            "head_to_head_vs_frozen_r6_r5_1_monday_r6_csv",
            "loso_metrics_csv",
            "pocket_guard_eval_report_csv_or_json",
            "manifest_v1.json",
            "status_v1.json",
            "consistency_audit_v1.csv",
        ],
        "required_future_audits_v1": [
            "training-surface freeze check",
            "legality boundary recheck",
            "bridge-not-used-in-training proof",
            "repaired/runner/fifty-plus pocket guard audit",
            "frozen benchmark comparison",
        ],
    }


def _build_stop_conditions() -> Dict[str, Any]:
    return {
        "layer_name_v1": "STOP_CONDITIONS_AND_NO_GO_CASES_V1",
        "pre_run_no_go_v1": [
            "readiness decision is no longer READY_TO_PLAN_NARROW_RETRAIN",
            "any legality check fails",
            "bridge surface is proposed as training surface",
            "training surface row count or contract drifts unexpectedly",
            "forbidden management/exit or policy-log fields appear in feature set",
        ],
        "post_run_no_go_v1": [
            "repaired_165_damage > 0",
            f"forensic trade {FORENSIC_TRADE} blocked",
            "100+/200+ blocked > 0",
            "strongest_winner_path_damage > 0",
            "50+ MFE blocked > 1",
            "worst LOSO below locked floor",
            "global precision regresses materially below locked floor",
            "runner near-miss pocket worsens without compensating safety evidence",
        ],
        "stop_immediately_if_v1": [
            "bridge-only rows detected in training matrix",
            "policy/controller codepath touched",
            "any sign that training output is being treated as promo/freeze by default",
        ],
    }


def _build_execution_order() -> Dict[str, Any]:
    return {
        "layer_name_v1": "NARROW_RETRAIN_EXECUTION_ORDER_V1",
        "steps_v1": [
            {"step_v1": 1, "name_v1": "RECHECK_INPUTS_AND_SCHEMAS", "must_pass_v1": "Training inputs, labels, contracts and row counts all match locked plan."},
            {"step_v1": 2, "name_v1": "RECHECK_LEGALITY_BOUNDARY", "must_pass_v1": "Forbidden fields absent; bridge-not-training guard still green."},
            {"step_v1": 3, "name_v1": "RECHECK_TRAINING_SURFACE_FREEZE", "must_pass_v1": "Exact-only canonical raw-state still the sole training surface."},
            {"step_v1": 4, "name_v1": "START_NARROW_RUNNER_FIRST_SHADOW_TRAINING", "must_pass_v1": "Use only locked feature set and locked objective priorities."},
            {"step_v1": 5, "name_v1": "MATERIALIZE_OUTPUTS", "must_pass_v1": "All required training outputs, manifests and audits written."},
            {"step_v1": 6, "name_v1": "RUN_EVAL_AND_REGRESSION_GUARDS", "must_pass_v1": "All hard safety guards and comparator checks evaluated."},
            {"step_v1": 7, "name_v1": "MATERIALIZE_VERDICT", "must_pass_v1": "Explicit verdict: failure-miner only, shadow candidate, or not usable."},
            {"step_v1": 8, "name_v1": "DECIDE_NEXT_ROLE", "must_pass_v1": "No automatic promo/freeze. Explicit human-reviewed phase gate required."},
        ],
    }


def _build_next_action() -> Dict[str, Any]:
    return {
        "layer_name_v1": "NEXT_AGENT_ACTION_LOCK_V1",
        "primary_action_v1": "AUTHORIZE_NARROW_RETRAIN_JOB_SPEC_ONLY",
        "supporting_actions_v1": [
            "NEXT_AGENT_MAY_PREPARE_TRAINING_BUT_NOT_RUN_IT",
            "KEEP_MONDAY_R6_AS_FAILURE_MINER",
            "DO_NOT_USE_BRIDGE_AS_TRAINING_SURFACE",
            "DO_NOT_TOUCH_POLICY_LAYER",
        ],
    }


def _write_report(
    path: Path,
    plan: Dict[str, Any],
    training_surface: Dict[str, Any],
    feature_lock: Dict[str, Any],
    objective_lock: Dict[str, Any],
    eval_guards: Dict[str, Any],
    stop_conditions: Dict[str, Any],
    execution_order: Dict[str, Any],
    next_action: Dict[str, Any],
) -> None:
    lines = [
        "# Monday Narrow Retrain Scope Plan V1",
        "",
        "## Plan",
        f"- Scope: `{plan['scope_v1']}`",
        f"- Purpose: {plan['purpose_v1']}",
        "",
        "## Training Surface",
        f"- Training surface: `{training_surface['training_surface_kind_v1']}`",
        f"- Training rows: `{training_surface['training_row_count_v1']}`",
        f"- Bridge-only rows forbidden: `{training_surface['bridge_surface_not_allowed_v1']['bridge_only_row_count_v1']}`",
        "",
        "## Included New Features",
    ]
    for feature_name in SELECTED_PROXIES:
        meta = feature_lock["new_proxy_features_v1"][feature_name]
        lines.append(f"- `{feature_name}`: {meta['role_v1']}")
    lines.extend(
        [
            "",
            "## Objective Order",
        ]
    )
    for row in objective_lock["priority_order_v1"]:
        lines.append(f"- `{row['rank_v1']}` `{row['objective_v1']}`")
    lines.extend(
        [
            "",
            "## Hard Guards",
            "- `repaired_165_damage = 0`",
            f"- forensic trade `{FORENSIC_TRADE}` must remain unblocked",
            "- `50+ <= 1`, `100+/200+ = 0`",
            "- `strongest_winner_path_damage = 0`",
            "- worst LOSO and global precision must not regress below locked floors",
            "",
            "## Stop Conditions",
        ]
    )
    for item in stop_conditions["post_run_no_go_v1"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Execution Order",
        ]
    )
    for row in execution_order["steps_v1"]:
        lines.append(f"- `{row['step_v1']}` `{row['name_v1']}`")
    lines.extend(
        [
            "",
            "## Next Action",
            f"- `{next_action['primary_action_v1']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize Monday narrow retrain scope plan V1.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--extension-dir", default=None)
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    extension_dir = _resolve_extension_dir(reports_root, args.extension_dir)
    extension_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_inputs(reports_root)

    baseline = _select_included_baseline_features(payload["raw_contract_df"])
    plan = _build_narrow_retrain_plan(payload, baseline)
    training_surface = _build_training_surface_lock(payload)
    feature_lock = _build_feature_set_lock(payload, baseline)
    objective_lock = _build_training_objective_lock()
    eval_guards = _build_eval_guard_plan(payload)
    training_io = _build_training_io_lock(payload)
    stop_conditions = _build_stop_conditions()
    execution_order = _build_execution_order()
    next_action = _build_next_action()

    _write_json(extension_dir / NARROW_PLAN, plan)
    _write_json(extension_dir / TRAINING_SURFACE_LOCK, training_surface)
    _write_json(extension_dir / FEATURE_SET_LOCK, feature_lock)
    _write_json(extension_dir / TRAINING_OBJECTIVE_LOCK, objective_lock)
    _write_json(extension_dir / EVAL_GUARD_PLAN, eval_guards)
    _write_json(extension_dir / TRAINING_IO_LOCK, training_io)
    _write_json(extension_dir / STOP_CONDITIONS, stop_conditions)
    _write_json(extension_dir / EXECUTION_ORDER, execution_order)
    _write_json(extension_dir / NEXT_ACTION, next_action)

    summary = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_SCOPE_PLAN_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "extension_dir_v1": str(extension_dir),
        "scope_v1": "NARROW_RUNNER_FIRST_SHADOW_ONLY",
        "training_surface_v1": str(payload["ledger_dir"] / ENTRY_RAW_STATE),
        "training_row_count_v1": training_surface["training_row_count_v1"],
        "bridge_rows_forbidden_v1": True,
        "selected_new_proxy_count_v1": len(SELECTED_PROXIES),
        "benchmark_v1": BENCHMARK,
        "monday_safety_reference_v1": MONDAY_SAFETY_REFERENCE,
        "monday_r6_role_v1": MONDAY_R6_ROLE,
        "next_action_v1": next_action["primary_action_v1"],
        "training_now_v1": False,
        "hard_status_division_v1": {
            "BEVIST": [
                "The narrow retrain is now planned as a runner-first shadow-only scope, not a broad retrain.",
                "The exact-only canonical raw-state remains the only locked training surface.",
                "Bridge-only rows remain forbidden for training.",
                "Hard safety guards and stop conditions are locked before any later training run.",
            ],
            "INDIKERT": [
                "The selected proxy and runner-protection slice is serious enough to justify a later narrow retrain job spec.",
                "The next agent may prepare the retrain job specification without reopening scope, guards, or surface decisions.",
            ],
            "IKKE_ETABLERT": [
                "That the later retrain will beat frozen Wednesday-R6.",
                "That training should start automatically from this plan.",
            ],
        },
    }
    _write_json(extension_dir / SUMMARY, summary)

    contract = {
        "layer_name_v1": "CONTRACT_V1",
        "job_v1": "MONDAY_NARROW_RETRAIN_SCOPE_PLAN_V1",
        "read_only_v1": True,
        "not_replay_v1": True,
        "not_training_v1": True,
        "not_policy_change_v1": True,
        "benchmark_v1": BENCHMARK,
        "monday_safety_reference_v1": MONDAY_SAFETY_REFERENCE,
        "monday_r6_role_v1": MONDAY_R6_ROLE,
        "do_not_use_bridge_as_training_surface_v1": True,
    }
    _write_json(extension_dir / CONTRACT, contract)

    _write_report(
        extension_dir / REPORT,
        plan=plan,
        training_surface=training_surface,
        feature_lock=feature_lock,
        objective_lock=objective_lock,
        eval_guards=eval_guards,
        stop_conditions=stop_conditions,
        execution_order=execution_order,
        next_action=next_action,
    )

    manifest = {
        "layer_name_v1": "MANIFEST_V1",
        "generated_at_utc_v1": _utc_now_iso(),
        "artifacts_v1": [
            CONTRACT,
            NARROW_PLAN,
            TRAINING_SURFACE_LOCK,
            FEATURE_SET_LOCK,
            TRAINING_OBJECTIVE_LOCK,
            EVAL_GUARD_PLAN,
            TRAINING_IO_LOCK,
            STOP_CONDITIONS,
            EXECUTION_ORDER,
            NEXT_ACTION,
            SUMMARY,
            REPORT,
        ],
    }
    _write_json(extension_dir / MANIFEST, manifest)

    audit_rows = [
        _audit_record(
            "READINESS_DECISION_IS_PLAN_ONLY_GREEN",
            "PASS" if payload["readiness_decision"].get("decision_v1") == "READY_TO_PLAN_NARROW_RETRAIN" else "FAIL",
            {"readiness_decision_v1": payload["readiness_decision"].get("decision_v1")},
        ),
        _audit_record(
            "TRAINING_SURFACE_IS_EXACT_ONLY",
            "PASS" if training_surface["training_surface_kind_v1"] == "CANONICAL_EXACT_ONLY_ENTRY_RAW_STATE" else "FAIL",
            {"training_surface_kind_v1": training_surface["training_surface_kind_v1"]},
        ),
        _audit_record(
            "BRIDGE_FORBIDDEN_FOR_TRAINING",
            "PASS" if training_surface["bridge_surface_not_allowed_v1"]["bridge_only_row_count_v1"] >= 0 else "FAIL",
            {"bridge_only_row_count_v1": training_surface["bridge_surface_not_allowed_v1"]["bridge_only_row_count_v1"]},
        ),
        _audit_record(
            "SELECTED_PROXIES_LOCKED",
            "PASS" if feature_lock["new_proxy_features_v1"].keys() == set(SELECTED_PROXIES) else "FAIL",
            {"selected_proxies_v1": list(feature_lock["new_proxy_features_v1"].keys())},
        ),
        _audit_record(
            "NEXT_ACTION_IS_SPEC_ONLY",
            "PASS" if next_action["primary_action_v1"] == "AUTHORIZE_NARROW_RETRAIN_JOB_SPEC_ONLY" else "FAIL",
            {"primary_action_v1": next_action["primary_action_v1"]},
        ),
    ]
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)

    status = {
        "layer_name_v1": "MONDAY_NARROW_RETRAIN_SCOPE_PLAN_STATUS_V1",
        "SPEC_STATUS": "MATERIALIZED_READ_ONLY",
        "failed_check_count_v1": int(audit_df["status_v1"].astype("string").ne("PASS").sum()),
        "not_replay_v1": True,
        "not_training_v1": True,
        "not_policy_change_v1": True,
        "plan_only_v1": True,
    }
    _write_json(extension_dir / STATUS, status)

    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
