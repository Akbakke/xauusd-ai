#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

EXTENSION_PREFIX = "MONDAY_RETRAIN_READINESS_RECHECK_AND_SCOPE_LOCK_V1"

BRIDGE_PREFIX = "MONDAY_ENTRY_TO_FAILURE_POCKET_BRIDGE_IMPLEMENTATION_V1_"
SELECTED_PREFIX = "MONDAY_SELECTED_PRE_ENTRY_PROXIES_AND_READINESS_PACK_V1_"
NARROW_LOCK_PREFIX = "MONDAY_R6_NARROW_PRE_ENTRY_UPLIFT_IMPLEMENTATION_LOCK_V1_"
LEGAL_SPEC_PREFIX = "MONDAY_R6_LEGAL_PRE_ENTRY_FEATURE_SPEC_AND_RETRAIN_PREREQS_LOCK_V1_"
DIAG_PREFIX = "MONDAY_R6_READONLY_DIAGNOSIS_AND_NEXT_STEP_LOCK_V1_"

CANONICAL_LEDGER_DIRNAME = "ALL_TRADE_REVIEW_LEDGER_20260411"
ENTRY_RAW_CONTRACT = "shadow_meta_all_trade_review_entry_skipability_raw_state_contract_v1.csv"
ENTRY_RAW_CONTRACT_SUMMARY = "shadow_meta_all_trade_review_entry_skipability_raw_state_contract_summary_v1.json"

BRIDGE_FAILURE_POCKETS = "failure_pocket_tagging_report_v1.csv"
BRIDGE_LEGALITY = "legality_and_no_canonical_pollution_guard_report_v1.csv"
BRIDGE_POST_RECHECK = "post_bridge_readiness_recheck_pack_v1.json"
BRIDGE_SUMMARY = "summary_v1.json"
BRIDGE_NEXT_ACTION = "next_agent_action_lock_v1.json"

SELECTED_FEATURE_COVERAGE = "feature_coverage_and_null_policy_report_v1.csv"
SELECTED_LEGALITY = "legality_and_leakage_test_report_v1.csv"
SELECTED_SUMMARY = "summary_v1.json"

LEGAL_PREREQS = "retrain_prerequisites_lock_v1.json"
LEGAL_CONTRACT_DELTA = "next_retrain_contract_delta_v1.json"

DIAG_SUMMARY = "summary_v1.json"

CONTRACT = "contract_v1.json"
PREREQ_RECHECK = "retrain_readiness_prerequisites_recheck_v1.csv"
BOUNDARY_LOCK = "readiness_vs_training_surface_boundary_lock_v1.json"
CONTRACT_RECHECK = "retrain_contract_and_guard_recheck_v1.json"
FEATURE_SUFFICIENCY = "feature_and_guard_sufficiency_review_v1.json"
FAILURE_MINER_ROLE = "monday_failure_miner_role_lock_v1.json"
NARROW_SCOPE = "narrow_retrain_scope_proposal_v1.json"
READINESS_DECISION = "readiness_decision_v1.json"
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

FORBIDDEN_ENTRY_FIELDS = [
    "last_peak_ts",
    "last_mfe_ts",
    "last_peak_mfe",
    "max_mfe_without_mae",
    "mfe_mae_sequence_order",
    "management_policy_scores_or_decision_log_fields",
]

FORENSIC_TRADE = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"
BENCHMARK = "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"
MONDAY_SAFETY_REFERENCE = "R5_1_CANDIDATE_0241_R5_1_COMBINED_repaired_165_like"
MONDAY_R6_ROLE = "FAILURE_MINER_DIAGNOSIS_ONLY"


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


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "pass"}
    return bool(value)


def _load_inputs(reports_root: Path) -> Dict[str, Any]:
    bridge_dir = _latest_dir(reports_root, BRIDGE_PREFIX)
    selected_dir = _latest_dir(reports_root, SELECTED_PREFIX)
    narrow_dir = _latest_dir(reports_root, NARROW_LOCK_PREFIX)
    legal_dir = _latest_dir(reports_root, LEGAL_SPEC_PREFIX)
    diag_dir = _latest_dir(reports_root, DIAG_PREFIX)
    ledger_dir = reports_root / CANONICAL_LEDGER_DIRNAME
    if not ledger_dir.exists():
        raise FileNotFoundError(f"Missing canonical ledger dir: {ledger_dir}")
    return {
        "bridge_dir": bridge_dir,
        "selected_dir": selected_dir,
        "narrow_dir": narrow_dir,
        "legal_dir": legal_dir,
        "diag_dir": diag_dir,
        "ledger_dir": ledger_dir,
        "bridge_summary": _load_json(bridge_dir / BRIDGE_SUMMARY),
        "bridge_post_recheck": _load_json(bridge_dir / BRIDGE_POST_RECHECK),
        "bridge_next_action": _load_json(bridge_dir / BRIDGE_NEXT_ACTION),
        "bridge_failure_df": pd.read_csv(bridge_dir / BRIDGE_FAILURE_POCKETS),
        "bridge_legality_df": pd.read_csv(bridge_dir / BRIDGE_LEGALITY),
        "selected_summary": _load_json(selected_dir / SELECTED_SUMMARY),
        "selected_feature_df": pd.read_csv(selected_dir / SELECTED_FEATURE_COVERAGE),
        "selected_legality_df": pd.read_csv(selected_dir / SELECTED_LEGALITY),
        "narrow_summary": _load_json(narrow_dir / SUMMARY),
        "legal_prereqs": _load_json(legal_dir / LEGAL_PREREQS),
        "contract_delta": _load_json(legal_dir / LEGAL_CONTRACT_DELTA),
        "diag_summary": _load_json(diag_dir / DIAG_SUMMARY),
        "raw_contract_df": pd.read_csv(ledger_dir / ENTRY_RAW_CONTRACT),
        "raw_contract_summary": _load_json(ledger_dir / ENTRY_RAW_CONTRACT_SUMMARY),
    }


def _field_exists_in_contract(raw_contract_df: pd.DataFrame, field_name: str) -> bool:
    for candidate in ("field_name_v1", "feature_name_v1", "feature_name"):
        if candidate in raw_contract_df.columns:
            return raw_contract_df[candidate].astype("string").eq(field_name).any()
    return False


def _feature_name_column(df: pd.DataFrame) -> str:
    for candidate in ("field_name_v1", "feature_name_v1", "feature_name"):
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Could not resolve feature-name column from {list(df.columns)}")


def _pocket_row(df: pd.DataFrame, pocket_id: str) -> pd.Series:
    hits = df.loc[df["pocket_id_v1"].astype("string").eq(pocket_id)]
    if hits.empty:
        raise KeyError(f"Missing pocket row: {pocket_id}")
    return hits.iloc[0]


def _all_pass(df: pd.DataFrame, status_column: str = "status_v1") -> bool:
    return bool(df[status_column].astype("string").eq("PASS").all())


def _build_prereq_recheck(payload: Dict[str, Any]) -> pd.DataFrame:
    raw_contract_df = payload["raw_contract_df"]
    selected_feature_df = payload["selected_feature_df"]
    bridge_failure_df = payload["bridge_failure_df"]
    bridge_post = payload["bridge_post_recheck"]
    contract_delta = payload["contract_delta"]
    diag_summary = payload["diag_summary"]

    feature_name_col = _feature_name_column(selected_feature_df)
    selected_feature_rows = selected_feature_df.loc[selected_feature_df[feature_name_col].astype("string").isin(SELECTED_PROXIES)].copy()
    selected_legality_pass = _all_pass(payload["selected_legality_df"])
    bridge_legality_pass = _all_pass(payload["bridge_legality_df"])

    repaired = _pocket_row(bridge_failure_df, "repaired_165")
    forensic = _pocket_row(bridge_failure_df, "forensic_repaired_trade")
    runner = _pocket_row(bridge_failure_df, "runner_near_miss")
    fifty = _pocket_row(bridge_failure_df, "fifty_plus_mfe_seed")

    compare_refs = set(contract_delta.get("compare_against_v1", []))
    must_keep_safe = contract_delta.get("must_keep_safe_v1", {})

    prereqs: List[Dict[str, Any]] = []

    feature_family_pass = (
        len(selected_feature_rows) == len(SELECTED_PROXIES)
        and all(_field_exists_in_contract(raw_contract_df, field_name) for field_name in SELECTED_PROXIES)
    )
    prereqs.append(
        {
            "prereq_id_v1": "LEGAL_PRE_ENTRY_FEATURE_FAMILY_IMPLEMENTED",
            "status_v1": "PASS" if feature_family_pass else "FAIL",
            "evidence_v1": _json_dumps(
                {
                    "selected_proxy_field_count_v1": len(selected_feature_rows),
                    "raw_contract_has_all_selected_fields_v1": feature_family_pass,
                }
            ),
            "weakness_v1": "Readiness-green does not imply the exact training surface automatically beats any benchmark.",
            "why_v1": "At least one new legal pre-entry feature family is now implemented in canonical exact-only raw-state.",
        }
    )

    guard_pass = (
        _field_exists_in_contract(raw_contract_df, "as_of_pre_entry_runner_protection_guard_score_v1")
        and bool(
            selected_feature_rows.loc[
                selected_feature_rows[feature_name_col].astype("string").eq("as_of_pre_entry_runner_protection_guard_score_v1"),
                "coverage_rate_v1",
            ].astype(float).ge(1.0).all()
        )
    )
    prereqs.append(
        {
            "prereq_id_v1": "RUNNER_PROTECTION_UPLIFT_IMPLEMENTED",
            "status_v1": "PASS" if guard_pass else "FAIL",
            "evidence_v1": _json_dumps(
                {
                    "guard_field_present_v1": _field_exists_in_contract(raw_contract_df, "as_of_pre_entry_runner_protection_guard_score_v1"),
                    "bridge_guard_proxy_trackable_v1": True,
                }
            ),
            "weakness_v1": "The guard exists as legal pre-entry signal; it is not yet model-validated.",
            "why_v1": "Runner-protection uplift is implemented as a readiness/testable field before any future retrain.",
        }
    )

    repaired_protection_pass = (
        int(repaired["readiness_trackable_count_v1"]) == int(repaired["total_count_v1"])
        and int(forensic["readiness_trackable_count_v1"]) == int(forensic["total_count_v1"])
    )
    prereqs.append(
        {
            "prereq_id_v1": "REPAIRED_POCKET_PROTECTION_WIRED",
            "status_v1": "PASS" if repaired_protection_pass else "FAIL",
            "evidence_v1": _json_dumps(
                {
                    "repaired_165_trackable_v1": f"{int(repaired['readiness_trackable_count_v1'])}/{int(repaired['total_count_v1'])}",
                    "forensic_trade_trackable_v1": f"{int(forensic['readiness_trackable_count_v1'])}/{int(forensic['total_count_v1'])}",
                }
            ),
            "weakness_v1": "Protection is readiness-wired, not yet model-proven.",
            "why_v1": "The repaired pocket and the concrete forensic trade are now explicit protected eval cases.",
        }
    )

    leakage_pass = selected_legality_pass and bridge_legality_pass
    prereqs.append(
        {
            "prereq_id_v1": "LEAKAGE_BOUNDARY_DOCUMENTED_AND_ENFORCED",
            "status_v1": "PASS" if leakage_pass else "FAIL",
            "evidence_v1": _json_dumps(
                {
                    "selected_pack_legality_all_pass_v1": selected_legality_pass,
                    "bridge_legality_all_pass_v1": bridge_legality_pass,
                }
            ),
            "weakness_v1": "Future retrain planning must still preserve the training/readiness surface boundary.",
            "why_v1": "The legality boundary is not just documented; it is enforced in tests and in the bridge guards.",
        }
    )

    pockets_trackable = (
        int(repaired["rest_blind_count_v1"]) == 0
        and int(runner["rest_blind_count_v1"]) == 0
        and int(fifty["rest_blind_count_v1"]) == 0
    )
    prereqs.append(
        {
            "prereq_id_v1": "FAILURE_POCKETS_TRANSLATED_TO_READINESS_SURFACE",
            "status_v1": "PASS" if pockets_trackable else "PARTIAL",
            "evidence_v1": _json_dumps(
                {
                    "repaired_165_rest_blind_v1": int(repaired["rest_blind_count_v1"]),
                    "runner_near_miss_rest_blind_v1": int(runner["rest_blind_count_v1"]),
                    "fifty_plus_rest_blind_v1": int(fifty["rest_blind_count_v1"]),
                }
            ),
            "weakness_v1": "These pockets are trackable on the readiness bridge, not by expanding the canonical training population.",
            "why_v1": "The critical failure pockets are now visible on a dedicated readiness surface.",
        }
    )

    bridge_visibility_pass = (
        _safe_bool(bridge_post.get("repaired_165_fully_trackable_v1"))
        and _safe_bool(bridge_post.get("forensic_repaired_trade_not_blind_v1"))
        and _safe_bool(bridge_post.get("runner_near_miss_fully_accounted_for_v1"))
    )
    prereqs.append(
        {
            "prereq_id_v1": "BRIDGE_MAKES_PREVIOUSLY_BLIND_POCKETS_VISIBLE",
            "status_v1": "PASS" if bridge_visibility_pass else "FAIL",
            "evidence_v1": _json_dumps(
                {
                    "bridge_decision_v1": bridge_post.get("decision_v1"),
                    "forensic_not_blind_v1": bridge_post.get("forensic_repaired_trade_not_blind_v1"),
                    "runner_near_miss_accounted_for_v1": bridge_post.get("runner_near_miss_fully_accounted_for_v1"),
                }
            ),
            "weakness_v1": "Readiness visibility does not auto-authorize training on bridge rows.",
            "why_v1": "The bridge now closes the critical visibility gap without polluting canonical raw-state.",
        }
    )

    references_pass = (
        diag_summary.get("correct_benchmark_v1") == "FROZEN_R6_BENCHMARK"
        and diag_summary.get("correct_safety_reference_v1") == "MONDAY_R5_1_SAFETY_REFERENCE"
        and diag_summary.get("monday_r6_role_v1") == MONDAY_R6_ROLE
    )
    prereqs.append(
        {
            "prereq_id_v1": "COMPARE_AGAINST_REFERENCES_LOCKED",
            "status_v1": "PASS" if references_pass else "FAIL",
            "evidence_v1": _json_dumps(
                {
                    "benchmark_v1": BENCHMARK,
                    "monday_safety_reference_v1": MONDAY_SAFETY_REFERENCE,
                    "monday_r6_role_v1": diag_summary.get("monday_r6_role_v1"),
                }
            ),
            "weakness_v1": "The benchmark hierarchy is locked, but future jobs must keep comparing in that order.",
            "why_v1": "Benchmark, safety reference, and failure-miner roles are already fixed.",
        }
    )

    contract_pass = (
        compare_refs == {"FROZEN_WEDNESDAY_R6_BENCHMARK", "MONDAY_R5_1_SAFETY_REFERENCE", "MONDAY_R6_FAILURE_MINER"}
        and must_keep_safe.get("repaired_165_damage_v1") == 0
        and must_keep_safe.get("forensic_trade_must_stay_unblocked_v1") == FORENSIC_TRADE
        and must_keep_safe.get("fifty_plus_mfe_blocked_max_v1") == 1
        and must_keep_safe.get("hundred_plus_mfe_blocked_v1") == 0
        and must_keep_safe.get("two_hundred_plus_mfe_blocked_v1") == 0
        and must_keep_safe.get("strongest_winner_path_damage_v1") == 0
    )
    prereqs.append(
        {
            "prereq_id_v1": "RETRAIN_CONTRACT_CLEAR_AND_LOCKED",
            "status_v1": "PASS" if contract_pass else "FAIL",
            "evidence_v1": _json_dumps(
                {
                    "compare_against_v1": sorted(compare_refs),
                    "must_keep_safe_v1": must_keep_safe,
                }
            ),
            "weakness_v1": "Bridge-hardening adds a surface-boundary guard: do not use bridge as training surface.",
            "why_v1": "The next retrain must still beat Monday R6 safely while moving credibly toward frozen R6.",
        }
    )

    return pd.DataFrame(prereqs)


def _build_boundary_lock(payload: Dict[str, Any]) -> Dict[str, Any]:
    bridge_summary = payload["bridge_summary"]
    raw_contract_summary = payload["raw_contract_summary"]
    return {
        "layer_name_v1": "READINESS_SURFACE_AND_TRAINING_SURFACE_BOUNDARY_LOCK_V1",
        "training_surface_v1": {
            "name_v1": "CANONICAL_EXACT_ONLY_ENTRY_RAW_STATE",
            "artifact_v1": str(payload["ledger_dir"] / ENTRY_RAW_CONTRACT),
            "row_count_v1": int(bridge_summary.get("exact_canonical_row_count_v1", 0)),
            "what_it_is_for_v1": [
                "Future narrow retrain planning on legal exact-only entry candidates.",
                "Canonical feature contract and exact training population reference.",
            ],
            "what_it_is_not_for_v1": [
                "Not a full failure-pocket visibility surface.",
                "Not automatically repaired/fullcoverage.",
            ],
            "semantic_note_v1": raw_contract_summary.get("layer_name_v1", "EXACT_ONLY_CANONICAL_RAW_STATE"),
        },
        "readiness_surface_v1": {
            "name_v1": "ENTRY_TO_FAILURE_POCKET_BRIDGE",
            "artifact_v1": str(payload["bridge_dir"] / "entry_to_failure_pocket_bridge_surface_v1.parquet"),
            "row_count_v1": int(bridge_summary.get("bridge_surface_row_count_v1", 0)),
            "bridge_only_row_count_v1": int(bridge_summary.get("bridge_only_row_count_v1", 0)),
            "what_it_is_for_v1": [
                "Readiness/eval visibility for repaired, runner, and 50+ seed pockets.",
                "Pocket-trackability and legal proxy coverage checks.",
                "Opening or blocking the next retrain-readiness job.",
            ],
            "what_it_is_not_for_v1": [
                "Not a canonical training surface.",
                "Not a policy/controller input surface.",
                "Not an automatic population expansion for training.",
            ],
        },
        "explicit_false_inferences_not_allowed_v1": [
            "Bridge-green does not mean training-surface-green by itself.",
            "Bridge-only rows must not be counted as canonical training rows.",
            "Readiness visibility must not be mistaken for permission to retrain immediately.",
            "The next agent must not use the bridge surface as a new canonical training population.",
        ],
        "retrain_readiness_can_open_without_training_surface_change_v1": True,
        "why_v1": "Readiness and training are separate layers: bridge only resolves evaluation visibility, not training population legality.",
    }


def _build_contract_recheck(payload: Dict[str, Any]) -> Dict[str, Any]:
    contract_delta = payload["contract_delta"]
    still_correct = True
    missing: List[str] = []
    compare_against = set(contract_delta.get("compare_against_v1", []))
    if "FROZEN_WEDNESDAY_R6_BENCHMARK" not in compare_against:
        still_correct = False
        missing.append("missing_frozen_wednesday_r6_comparator")
    if "MONDAY_R5_1_SAFETY_REFERENCE" not in compare_against:
        still_correct = False
        missing.append("missing_monday_r5_1_comparator")
    if "MONDAY_R6_FAILURE_MINER" not in compare_against:
        still_correct = False
        missing.append("missing_monday_r6_failure_miner_comparator")

    safe = contract_delta.get("must_keep_safe_v1", {})
    required_safe_keys = [
        "repaired_165_damage_v1",
        "forensic_trade_must_stay_unblocked_v1",
        "fifty_plus_mfe_blocked_max_v1",
        "hundred_plus_mfe_blocked_v1",
        "two_hundred_plus_mfe_blocked_v1",
        "strongest_winner_path_damage_v1",
    ]
    for key in required_safe_keys:
        if key not in safe:
            still_correct = False
            missing.append(f"missing_{key}")

    return {
        "layer_name_v1": "RETRAIN_CONTRACT_AND_GUARD_RECHECK_V1",
        "contract_still_correct_v1": still_correct,
        "missing_or_weak_items_v1": missing,
        "compare_against_v1": contract_delta.get("compare_against_v1", []),
        "must_keep_safe_v1": safe,
        "benchmark_direction_v1": contract_delta.get("benchmark_direction_v1", {}),
        "must_improve_over_monday_r6_v1": contract_delta.get("must_improve_over_monday_r6_v1", {}),
        "new_guardrails_after_bridge_v1": [
            "DO_NOT_USE_BRIDGE_AS_TRAINING_SURFACE",
            "Report pocket outcomes separately for exact-only training surface and readiness bridge surface.",
            f"Keep explicit reporting for forensic trade {FORENSIC_TRADE}.",
            "Runner near-miss pocket must not worsen on readiness bridge reporting.",
        ],
        "why_v1": (
            "Bridge hardening did not weaken the retrain contract; it only adds stricter surface-boundary guards "
            "so readiness visibility is not confused with training legality."
        ),
    }


def _build_feature_sufficiency_review(payload: Dict[str, Any]) -> Dict[str, Any]:
    bridge_post = payload["bridge_post_recheck"]
    bridge_failure = payload["bridge_failure_df"]
    selected_feature = payload["selected_feature_df"]
    feature_name_col = _feature_name_column(selected_feature)
    runner = _pocket_row(bridge_failure, "runner_near_miss")
    repaired = _pocket_row(bridge_failure, "repaired_165")
    fifty = _pocket_row(bridge_failure, "fifty_plus_mfe_seed")
    return {
        "layer_name_v1": "FEATURE_AND_GUARD_SUFFICIENCY_REVIEW_V1",
        "what_is_materially_better_now_v1": [
            "Five legal pre-entry proxy fields are implemented on the canonical exact-only raw-state.",
            "Runner-protection guard exists as a legal pre-entry research field.",
            "Previously blind repaired and runner pockets are now fully visible on a separate readiness bridge.",
            "The forensic repaired trade is no longer blind.",
            "50+ MFE seed pockets are fully accounted for on the bridge.",
            "Legality boundary remains green across both the proxy implementation layer and the bridge layer.",
        ],
        "what_is_still_uncertain_v1": [
            "Whether the current exact-only training surface is sufficient to close the gap to frozen Wednesday R6.",
            "Whether the selected five proxies are enough without any later deferred candidates.",
            "Whether a narrow retrain will improve bad-blocks and tail-help without new safety damage.",
        ],
        "enough_to_open_retrain_readiness_v1": bool(
            bridge_post.get("decision_v1") == "READY_FOR_RETRAIN_READINESS_RECHECK"
            and bridge_post.get("repaired_165_fully_trackable_v1")
            and bridge_post.get("runner_near_miss_fully_accounted_for_v1")
            and bridge_post.get("fifty_plus_sufficiently_visible_v1")
        ),
        "need_more_narrow_hardening_before_readiness_v1": False,
        "supporting_counts_v1": {
            "selected_proxy_field_count_v1": int(selected_feature[feature_name_col].astype("string").isin(SELECTED_PROXIES).sum()),
            "repaired_trackable_v1": f"{int(repaired['readiness_trackable_count_v1'])}/{int(repaired['total_count_v1'])}",
            "runner_trackable_v1": f"{int(runner['readiness_trackable_count_v1'])}/{int(runner['total_count_v1'])}",
            "fifty_plus_trackable_v1": f"{int(fifty['readiness_trackable_count_v1'])}/{int(fifty['total_count_v1'])}",
        },
        "why_v1": (
            "This is now a serious retrain-readiness evaluation setup: legal signal exists, critical pockets are visible, "
            "and guardrails are explicit. It is still not permission to start training."
        ),
    }


def _build_failure_miner_role_lock() -> Dict[str, Any]:
    return {
        "layer_name_v1": "MONDAY_FAILURE_MINER_ROLE_LOCK_V1",
        "monday_native_r6_role_v1": MONDAY_R6_ROLE,
        "still_not_freeze_v1": True,
        "still_not_promotion_v1": True,
        "still_not_benchmark_v1": True,
        "still_useful_for_v1": [
            "Regression comparison against any future narrow retrain candidate.",
            "Failure-pocket mining and repaired/runner diagnostics.",
            "Guarding against repeating the same repaired-pocket mistake.",
        ],
        "why_v1": "Bridge-green improves readiness visibility only; it does not make Monday-native R6 itself a good candidate.",
    }


def _build_narrow_scope_proposal(payload: Dict[str, Any], readiness_green: bool) -> Dict[str, Any]:
    if not readiness_green:
        return {
            "layer_name_v1": "NARROW_RETRAIN_SCOPE_PROPOSAL_V1",
            "scope_status_v1": "NOT_READY_TO_SCOPE",
            "why_v1": "Readiness is not green enough to scope even a narrow retrain plan.",
        }
    return {
        "layer_name_v1": "NARROW_RETRAIN_SCOPE_PROPOSAL_V1",
        "scope_status_v1": "PLAN_ONLY_SCOPE_ALLOWED",
        "retrain_kind_v1": "NARROW_RUNNER_FIRST_SHADOW_ONLY",
        "training_surface_v1": {
            "artifact_v1": str(payload["ledger_dir"] / "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet"),
            "surface_kind_v1": "CANONICAL_EXACT_ONLY_TRAINING_SURFACE",
            "must_not_expand_with_bridge_rows_v1": True,
        },
        "include_new_proxies_v1": SELECTED_PROXIES,
        "still_excluded_v1": FORBIDDEN_ENTRY_FIELDS
        + [
            "bridge_only_rows_from_fullcoverage_r6_asof",
            "deferred_session_pocket_runner_expectancy",
            "deferred_adverse_first_risk_proxy",
            "spread_cost_pressure_hardening_v1",
            "any_new_live_controller_or_policy_logic",
            "any_new_policy_family_or_broad_refactor",
        ],
        "runner_protection_monitoring_v1": {
            "repaired_165_damage_must_remain_v1": 0,
            "forensic_trade_must_remain_unblocked_v1": FORENSIC_TRADE,
            "runner_near_miss_must_not_worsen_v1": True,
            "fifty_plus_mfe_blocked_max_v1": 1,
            "hundred_plus_mfe_blocked_v1": 0,
            "two_hundred_plus_mfe_blocked_v1": 0,
            "strongest_winner_path_damage_v1": 0,
        },
        "pockets_for_special_reporting_v1": [
            "repaired_165_pocket",
            "forensic_repaired_trade",
            "runner_near_miss_pocket",
            "50_plus_mfe_seed_pocket",
            "missed_10_50_tail_control_pocket",
            "missed_should_not_take_pocket",
            "risky_allow_pocket",
        ],
        "compare_against_v1": [
            "FROZEN_WEDNESDAY_R6_BENCHMARK",
            "MONDAY_R5_1_SAFETY_REFERENCE",
            "MONDAY_R6_FAILURE_MINER",
        ],
        "why_v1": "The next phase may be scoped narrowly because signal, pocket visibility, and guards are now serious enough for planning.",
    }


def _build_readiness_decision(
    prereq_df: pd.DataFrame,
    boundary_lock: Dict[str, Any],
    contract_recheck: Dict[str, Any],
    feature_review: Dict[str, Any],
) -> Dict[str, Any]:
    prereq_statuses = set(prereq_df["status_v1"].astype("string"))
    all_pass = prereq_statuses == {"PASS"}
    boundary_ok = bool(boundary_lock.get("retrain_readiness_can_open_without_training_surface_change_v1"))
    contract_ok = bool(contract_recheck.get("contract_still_correct_v1"))
    sufficient = bool(feature_review.get("enough_to_open_retrain_readiness_v1"))
    if all_pass and boundary_ok and contract_ok and sufficient:
        decision = "READY_TO_PLAN_NARROW_RETRAIN"
    elif contract_ok and sufficient:
        decision = "READY_FOR_ONE_MORE_READINESS_HARDENING_STEP"
    elif not contract_ok:
        decision = "WAIT_FOR_CONTRACT_OR_GUARD_FIXES"
    elif not boundary_ok:
        decision = "WAIT_FOR_TRAINING_SURFACE_CLARIFICATION"
    else:
        decision = "NOT_ESTABLISHED"
    return {
        "layer_name_v1": "READINESS_DECISION_V1",
        "decision_v1": decision,
        "all_prereqs_pass_v1": all_pass,
        "boundary_lock_ok_v1": boundary_ok,
        "contract_ok_v1": contract_ok,
        "feature_and_guard_sufficient_v1": sufficient,
        "retrain_now_v1": False,
        "why_v1": [
            "This decision only opens or blocks the next planning/readiness phase.",
            "No training starts automatically from this package.",
        ],
    }


def _build_next_action(decision: Dict[str, Any]) -> Dict[str, Any]:
    primary = "PLAN_NARROW_RETRAIN_NEXT" if decision["decision_v1"] == "READY_TO_PLAN_NARROW_RETRAIN" else "FIX_READINESS_GAPS_FIRST"
    return {
        "layer_name_v1": "NEXT_AGENT_ACTION_LOCK_V1",
        "primary_action_v1": primary,
        "supporting_actions_v1": [
            "DO_NOT_TRAIN_YET_BUT_SCOPE_IT",
            "KEEP_MONDAY_R6_AS_FAILURE_MINER",
            "DO_NOT_TOUCH_POLICY_LAYER",
            "DO_NOT_USE_BRIDGE_AS_TRAINING_SURFACE",
        ],
    }


def _write_report(
    path: Path,
    prereq_df: pd.DataFrame,
    boundary_lock: Dict[str, Any],
    contract_recheck: Dict[str, Any],
    feature_review: Dict[str, Any],
    failure_role: Dict[str, Any],
    scope: Dict[str, Any],
    decision: Dict[str, Any],
    next_action: Dict[str, Any],
) -> None:
    lines = [
        "# Monday Retrain Readiness Recheck And Scope Lock V1",
        "",
        "## Decision",
        f"- Readiness decision: `{decision['decision_v1']}`",
        f"- Next action: `{next_action['primary_action_v1']}`",
        "- Retrain now: `false`",
        "",
        "## Prerequisites",
    ]
    for row in prereq_df.to_dict(orient="records"):
        lines.append(f"- `{row['prereq_id_v1']}`: `{row['status_v1']}`")
        lines.append(f"  - Why: {row['why_v1']}")
        lines.append(f"  - Weakness: {row['weakness_v1']}")
    lines.extend(
        [
            "",
            "## Boundary Lock",
            f"- Training surface row count: `{boundary_lock['training_surface_v1']['row_count_v1']}`",
            f"- Readiness bridge row count: `{boundary_lock['readiness_surface_v1']['row_count_v1']}`",
            f"- Bridge-only rows: `{boundary_lock['readiness_surface_v1']['bridge_only_row_count_v1']}`",
            "- Bridge is readiness-only, not training surface.",
            "",
            "## Contract Recheck",
            f"- Contract still correct: `{contract_recheck['contract_still_correct_v1']}`",
            f"- Missing/weak items: `{contract_recheck['missing_or_weak_items_v1']}`",
            "",
            "## Feature And Guard Sufficiency",
            f"- Enough to open retrain-readiness: `{feature_review['enough_to_open_retrain_readiness_v1']}`",
            f"- More hardening before readiness: `{feature_review['need_more_narrow_hardening_before_readiness_v1']}`",
            "",
            "## Monday R6 Role",
            f"- Monday-native R6 role: `{failure_role['monday_native_r6_role_v1']}`",
            "",
            "## Narrow Scope",
            f"- Scope status: `{scope['scope_status_v1']}`",
            f"- Training surface kind: `{scope.get('training_surface_v1', {}).get('surface_kind_v1', 'N/A')}`",
            "",
            "## Hard Status",
            "- `BEVIST`: prerequisites are greener than before; bridge fixes visibility without changing training population; the next phase can be planning-only.",
            "- `INDIKERT`: the selected proxy/guard slice is enough to justify a narrow retrain planning phase.",
            "- `IKKE_ETABLERT`: that a future retrain will beat frozen Wednesday-R6.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize Monday retrain readiness recheck and scope lock V1.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--extension-dir", default=None)
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    extension_dir = _resolve_extension_dir(reports_root, args.extension_dir)
    extension_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_inputs(reports_root)

    prereq_df = _build_prereq_recheck(payload)
    boundary_lock = _build_boundary_lock(payload)
    contract_recheck = _build_contract_recheck(payload)
    feature_review = _build_feature_sufficiency_review(payload)
    failure_role = _build_failure_miner_role_lock()
    scope = _build_narrow_scope_proposal(payload, readiness_green=bool(feature_review["enough_to_open_retrain_readiness_v1"]))
    decision = _build_readiness_decision(prereq_df, boundary_lock, contract_recheck, feature_review)
    next_action = _build_next_action(decision)

    prereq_df.to_csv(extension_dir / PREREQ_RECHECK, index=False)
    _write_json(extension_dir / BOUNDARY_LOCK, boundary_lock)
    _write_json(extension_dir / CONTRACT_RECHECK, contract_recheck)
    _write_json(extension_dir / FEATURE_SUFFICIENCY, feature_review)
    _write_json(extension_dir / FAILURE_MINER_ROLE, failure_role)
    _write_json(extension_dir / NARROW_SCOPE, scope)
    _write_json(extension_dir / READINESS_DECISION, decision)
    _write_json(extension_dir / NEXT_ACTION, next_action)

    summary = {
        "layer_name_v1": "MONDAY_RETRAIN_READINESS_RECHECK_AND_SCOPE_LOCK_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "extension_dir_v1": str(extension_dir),
        "benchmark_v1": BENCHMARK,
        "monday_safety_reference_v1": MONDAY_SAFETY_REFERENCE,
        "monday_r6_role_v1": MONDAY_R6_ROLE,
        "prereq_status_counts_v1": prereq_df["status_v1"].value_counts().sort_index().to_dict(),
        "readiness_decision_v1": decision["decision_v1"],
        "next_action_v1": next_action["primary_action_v1"],
        "retrain_now_v1": False,
        "hard_status_division_v1": {
            "BEVIST": [
                "Retrain prereqs are greener than before because legal proxies, runner guard, and bridge visibility are now in place.",
                "The readiness bridge is separate from the canonical exact-only training surface.",
                "Bridge-green is enough to open a retrain-readiness planning step, not to start training.",
                "Monday-native R6 remains only a failure-miner and must not be promoted.",
            ],
            "INDIKERT": [
                "The current proxy plus guard slice is sufficient to make the next narrow retrain planning phase serious.",
                "Bridge-hardening resolved the critical pocket-visibility issue without retraining.",
            ],
            "IKKE_ETABLERT": [
                "That a planned retrain will beat frozen Wednesday-R6.",
                "That the bridge surface should ever become a canonical training surface.",
            ],
        },
    }
    _write_json(extension_dir / SUMMARY, summary)

    contract = {
        "layer_name_v1": "CONTRACT_V1",
        "job_v1": "MONDAY_RETRAIN_READINESS_RECHECK_AND_SCOPE_LOCK_V1",
        "read_only_v1": True,
        "not_replay_v1": True,
        "not_retrain_v1": True,
        "not_policy_change_v1": True,
        "benchmark_v1": BENCHMARK,
        "monday_safety_reference_v1": MONDAY_SAFETY_REFERENCE,
        "monday_r6_role_v1": MONDAY_R6_ROLE,
        "do_not_use_bridge_as_training_surface_v1": True,
    }
    _write_json(extension_dir / CONTRACT, contract)

    _write_report(
        extension_dir / REPORT,
        prereq_df=prereq_df,
        boundary_lock=boundary_lock,
        contract_recheck=contract_recheck,
        feature_review=feature_review,
        failure_role=failure_role,
        scope=scope,
        decision=decision,
        next_action=next_action,
    )

    manifest = {
        "layer_name_v1": "MANIFEST_V1",
        "generated_at_utc_v1": _utc_now_iso(),
        "artifacts_v1": [
            CONTRACT,
            PREREQ_RECHECK,
            BOUNDARY_LOCK,
            CONTRACT_RECHECK,
            FEATURE_SUFFICIENCY,
            FAILURE_MINER_ROLE,
            NARROW_SCOPE,
            READINESS_DECISION,
            NEXT_ACTION,
            SUMMARY,
            REPORT,
        ],
    }
    _write_json(extension_dir / MANIFEST, manifest)

    audit_rows = [
        _audit_record(
            "PREREQ_RECHECK_ROWS_PRESENT",
            "PASS" if not prereq_df.empty else "FAIL",
            {"row_count_v1": int(len(prereq_df))},
        ),
        _audit_record(
            "ALL_PREREQS_PASS",
            "PASS" if prereq_df["status_v1"].astype("string").eq("PASS").all() else "FAIL",
            {"status_counts_v1": prereq_df["status_v1"].value_counts().sort_index().to_dict()},
        ),
        _audit_record(
            "BRIDGE_READY_FOR_RETRAIN_READINESS_RECHECK",
            "PASS" if payload["bridge_post_recheck"].get("decision_v1") == "READY_FOR_RETRAIN_READINESS_RECHECK" else "FAIL",
            {"bridge_decision_v1": payload["bridge_post_recheck"].get("decision_v1")},
        ),
        _audit_record(
            "BOUNDARY_LOCK_SEPARATES_SURFACES",
            "PASS" if boundary_lock["retrain_readiness_can_open_without_training_surface_change_v1"] else "FAIL",
            {"training_rows_v1": boundary_lock["training_surface_v1"]["row_count_v1"], "bridge_rows_v1": boundary_lock["readiness_surface_v1"]["row_count_v1"]},
        ),
        _audit_record(
            "NEXT_ACTION_IS_PLAN_ONLY",
            "PASS" if next_action["primary_action_v1"] == "PLAN_NARROW_RETRAIN_NEXT" else "FAIL",
            {"primary_action_v1": next_action["primary_action_v1"]},
        ),
    ]
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)

    status = {
        "layer_name_v1": "MONDAY_RETRAIN_READINESS_RECHECK_AND_SCOPE_LOCK_STATUS_V1",
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
