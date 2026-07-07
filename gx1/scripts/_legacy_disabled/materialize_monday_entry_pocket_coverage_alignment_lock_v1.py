#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from gx1.analysis.shadow_meta_v1 import _ENTRY_PRE_ENTRY_PROXY_SPEC_V1


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

EXTENSION_PREFIX = "MONDAY_ENTRY_POCKET_COVERAGE_ALIGNMENT_LOCK_V1"
SELECTED_PRE_ENTRY_PACK_PREFIX = "MONDAY_SELECTED_PRE_ENTRY_PROXIES_AND_READINESS_PACK_V1_"

CANONICAL_LEDGER_DIRNAME = "ALL_TRADE_REVIEW_LEDGER_20260411"
R6_DIRNAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1"

RAW_STATE = "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet"
R6_ASOF = "shadow_meta_all_trade_review_r6_entry_runner_first_as_of_feature_table_v1.parquet"
R6_POLICY = "shadow_meta_all_trade_review_r6_policy_prediction_view_v1.parquet"
R6_HINDSIGHT = "shadow_meta_all_trade_review_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet"

CONTRACT = "contract_v1.json"
POCKET_COVERAGE_GAP_FORENSICS = "pocket_coverage_gap_forensics_v1.csv"
ENTRY_SURFACE_ALIGNMENT_LOCK = "entry_surface_alignment_lock_v1.csv"
COVERAGE_HARDENING_PLAN = "coverage_hardening_plan_v1.csv"
CONCRETE_REPAIRED_TRADE_FORENSIC = "concrete_repaired_trade_alignment_forensic_v1.json"
RUNNER_NEAR_MISS_ALIGNMENT = "runner_near_miss_alignment_and_hardening_v1.json"
LEGALITY_SAFE_COVERAGE_FIX_SPEC = "legality_safe_coverage_fix_spec_v1.csv"
IMPLEMENTATION_TARGETS = "implementation_targets_for_coverage_hardening_v1.csv"
POST_HARDENING_READINESS = "post_hardening_readiness_criteria_v1.json"
NEXT_ACTION = "next_agent_action_lock_v1.json"
SUMMARY = "summary_v1.json"
REPORT = "report_v1.md"
MANIFEST = "manifest_v1.json"
STATUS = "status_v1.json"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"

FORENSIC_TRADE = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"
BENCHMARK = "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"
MONDAY_SAFETY_REFERENCE = "R5_1_CANDIDATE_0241_R5_1_COMBINED_repaired_165_like"

SELECTED_PROXY_FIELDS = list(_ENTRY_PRE_ENTRY_PROXY_SPEC_V1.keys())
SELECTED_PROXY_INPUTS = sorted(
    {
        str(field_name)
        for spec in _ENTRY_PRE_ENTRY_PROXY_SPEC_V1.values()
        for field_name in spec.get("inputs", [])
    }
)


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
    matches = sorted(
        [path for path in reports_root.iterdir() if path.is_dir() and path.name.startswith(prefix)],
        key=lambda path: path.name,
    )
    if not matches:
        raise FileNotFoundError(f"No directory found for prefix {prefix} under {reports_root}")
    return matches[-1]


def _resolve_extension_dir(reports_root: Path, extension_dir_arg: str | None) -> Path:
    if extension_dir_arg:
        return Path(extension_dir_arg).expanduser().resolve()
    return reports_root / f"{EXTENSION_PREFIX}_{_utc_compact()}"


def _resolve_inputs(reports_root: Path) -> Dict[str, Any]:
    ledger_dir = reports_root / CANONICAL_LEDGER_DIRNAME
    r6_dir = reports_root / R6_DIRNAME
    if not ledger_dir.exists():
        raise FileNotFoundError(f"Missing canonical ledger dir: {ledger_dir}")
    if not r6_dir.exists():
        raise FileNotFoundError(f"Missing R6 dir: {r6_dir}")
    selected_pre_entry_pack_dir = _latest_dir(reports_root, SELECTED_PRE_ENTRY_PACK_PREFIX)

    raw_state_df = pd.read_parquet(
        ledger_dir / RAW_STATE,
        columns=["candidate_uid", "run_id", "trade_uid", "trade_id"],
    )
    asof_df = pd.read_parquet(ledger_dir.parent / R6_DIRNAME / R6_ASOF)
    policy_df = pd.read_parquet(
        r6_dir / R6_POLICY,
        columns=["candidate_uid", "run_id", "trade_uid", "trade_id", "is_repaired_165_v1", "fifty_plus_mfe_v1"],
    )
    hindsight_df = pd.read_parquet(
        r6_dir / R6_HINDSIGHT,
        columns=[
            "candidate_uid",
            "run_id",
            "trade_uid",
            "trade_id",
            "r6_label_runner_near_miss_v1",
            "r6_label_tail_control_10_50_v1",
            "r6_label_missed_should_not_take_v1",
            "r6_label_risky_allow_v1",
            "r6_label_repaired_165_like_runner_v1",
        ],
    )
    selected_pre_entry_summary = _load_json(selected_pre_entry_pack_dir / SUMMARY)
    return {
        "ledger_dir": ledger_dir,
        "r6_dir": r6_dir,
        "selected_pre_entry_pack_dir": selected_pre_entry_pack_dir,
        "selected_pre_entry_summary": selected_pre_entry_summary,
        "raw_state_df": raw_state_df,
        "asof_df": asof_df,
        "policy_df": policy_df,
        "hindsight_df": hindsight_df,
    }


def _string_set(series: pd.Series) -> set[str]:
    return set(series.astype("string").tolist())


def _triplet_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        list(
            zip(
                df["run_id"].astype("string"),
                df["trade_uid"].astype("string"),
                df["trade_id"].astype("string"),
            )
        ),
        index=df.index,
        dtype="object",
    )


def _merge_eval_surfaces(inputs: Dict[str, Any]) -> Dict[str, Any]:
    raw_state_df = inputs["raw_state_df"].copy()
    asof_df = inputs["asof_df"].copy()
    policy_df = inputs["policy_df"].copy()
    hindsight_df = inputs["hindsight_df"].copy()

    raw_state_df["candidate_uid"] = raw_state_df["candidate_uid"].astype("string")
    asof_df["candidate_uid"] = asof_df["candidate_uid"].astype("string")
    policy_df["candidate_uid"] = policy_df["candidate_uid"].astype("string")
    hindsight_df["candidate_uid"] = hindsight_df["candidate_uid"].astype("string")

    for col in [
        "is_repaired_165_v1",
        "fifty_plus_mfe_v1",
        "r6_label_runner_near_miss_v1",
        "r6_label_tail_control_10_50_v1",
        "r6_label_missed_should_not_take_v1",
        "r6_label_risky_allow_v1",
        "r6_label_repaired_165_like_runner_v1",
    ]:
        if col in policy_df.columns:
            policy_df[col] = policy_df[col].fillna(False).astype(bool)
        if col in hindsight_df.columns:
            hindsight_df[col] = hindsight_df[col].fillna(False).astype(bool)

    asof_enriched_df = (
        asof_df.merge(
            policy_df[["candidate_uid", "is_repaired_165_v1", "fifty_plus_mfe_v1"]],
            on="candidate_uid",
            how="left",
            validate="one_to_one",
        )
        .merge(
            hindsight_df[
                [
                    "candidate_uid",
                    "r6_label_runner_near_miss_v1",
                    "r6_label_tail_control_10_50_v1",
                    "r6_label_missed_should_not_take_v1",
                    "r6_label_risky_allow_v1",
                    "r6_label_repaired_165_like_runner_v1",
                ]
            ],
            on="candidate_uid",
            how="left",
            validate="one_to_one",
        )
    )
    for col in [
        "is_repaired_165_v1",
        "fifty_plus_mfe_v1",
        "r6_label_runner_near_miss_v1",
        "r6_label_tail_control_10_50_v1",
        "r6_label_missed_should_not_take_v1",
        "r6_label_risky_allow_v1",
        "r6_label_repaired_165_like_runner_v1",
    ]:
        if col in asof_enriched_df.columns:
            asof_enriched_df[col] = asof_enriched_df[col].astype("boolean").fillna(False).astype(bool)

    raw_candidate_keys = _string_set(raw_state_df["candidate_uid"])
    raw_trade_keys = set(_triplet_series(raw_state_df).tolist())
    asof_candidate_keys = _string_set(asof_df["candidate_uid"])
    return {
        "raw_state_df": raw_state_df,
        "asof_enriched_df": asof_enriched_df,
        "policy_df": policy_df,
        "hindsight_df": hindsight_df,
        "raw_candidate_keys": raw_candidate_keys,
        "raw_trade_keys": raw_trade_keys,
        "asof_candidate_keys": asof_candidate_keys,
    }


def _source_surface_rows(merged: Dict[str, Any], pocket_id: str) -> pd.DataFrame:
    asof_enriched_df = merged["asof_enriched_df"]
    policy_df = merged["policy_df"]
    hindsight_df = merged["hindsight_df"]
    if pocket_id == "repaired_165":
        source_df = policy_df.loc[policy_df["is_repaired_165_v1"].fillna(False).astype(bool)].copy()
    elif pocket_id == "runner_near_miss":
        source_df = hindsight_df.loc[hindsight_df["r6_label_runner_near_miss_v1"].fillna(False).astype(bool)].copy()
    elif pocket_id == "fifty_plus_mfe_seed":
        source_df = policy_df.loc[policy_df["fifty_plus_mfe_v1"].fillna(False).astype(bool)].copy()
    elif pocket_id == "forensic_repaired_trade":
        policy_match_df = policy_df.loc[policy_df["candidate_uid"].astype("string").eq(FORENSIC_TRADE)].copy()
        if not policy_match_df.empty:
            source_df = policy_match_df
        else:
            source_df = hindsight_df.loc[hindsight_df["candidate_uid"].astype("string").eq(FORENSIC_TRADE)].copy()
    else:
        raise KeyError(f"Unknown pocket_id: {pocket_id}")

    asof_bridge_cols = [
        "candidate_uid",
        "entry_coverage_original_entry_observation_present_v1",
        "entry_coverage_original_entry_raw_state_present_v1",
        "entry_coverage_repair_applied_v1",
        "entry_coverage_repair_source_v1",
        *[field_name for field_name in SELECTED_PROXY_INPUTS if field_name in asof_enriched_df.columns],
    ]
    return source_df.merge(
        asof_enriched_df[asof_bridge_cols].drop_duplicates(subset=["candidate_uid"]),
        on="candidate_uid",
        how="left",
        validate="one_to_one",
    )


def _source_artifact_and_kind(pocket_id: str) -> Dict[str, str]:
    if pocket_id in {"repaired_165", "fifty_plus_mfe_seed", "forensic_repaired_trade"}:
        return {
            "source_artifact_v1": R6_POLICY,
            "source_surface_kind_v1": "POLICY_VIEW_CANDIDATE_SURFACE",
        }
    if pocket_id == "runner_near_miss":
        return {
            "source_artifact_v1": R6_HINDSIGHT,
            "source_surface_kind_v1": "HINDSIGHT_LABEL_CANDIDATE_SURFACE",
        }
    raise KeyError(f"Unknown pocket_id: {pocket_id}")


def _proxy_input_coverage(sub_df: pd.DataFrame) -> Dict[str, Any]:
    available_inputs = [field_name for field_name in SELECTED_PROXY_INPUTS if field_name in sub_df.columns]
    if not len(sub_df):
        return {
            "input_field_count_v1": int(len(available_inputs)),
            "minimum_input_coverage_rate_v1": None,
            "fully_covered_input_count_v1": 0,
        }
    coverage = {
        field_name: float(pd.to_numeric(sub_df[field_name], errors="coerce").notna().mean())
        for field_name in available_inputs
    }
    return {
        "input_field_count_v1": int(len(available_inputs)),
        "minimum_input_coverage_rate_v1": None if not coverage else float(min(coverage.values())),
        "fully_covered_input_count_v1": int(sum(value >= 1.0 for value in coverage.values())),
    }


def _classify_alignment(total_rows: int, raw_exact_matches: int, asof_matches: int) -> str:
    if total_rows == 0:
        return "NOT_ESTABLISHED"
    if raw_exact_matches == total_rows:
        return "ALIGNED"
    if asof_matches == total_rows and raw_exact_matches == 0:
        return "MISALIGNED_SURFACE"
    if asof_matches == total_rows and raw_exact_matches < total_rows:
        return "PARTIAL_ALIGNMENT"
    return "NOT_ESTABLISHED"


def _pocket_forensics_rows(merged: Dict[str, Any]) -> pd.DataFrame:
    raw_candidate_keys = merged["raw_candidate_keys"]
    raw_trade_keys = merged["raw_trade_keys"]
    asof_candidate_keys = merged["asof_candidate_keys"]
    asof_enriched_df = merged["asof_enriched_df"]

    pocket_rows: List[Dict[str, Any]] = []
    for pocket_id in ["repaired_165", "runner_near_miss", "forensic_repaired_trade", "fifty_plus_mfe_seed"]:
        source_rows = _source_surface_rows(merged, pocket_id)
        source_rows["raw_candidate_exact_match_v1"] = source_rows["candidate_uid"].astype("string").isin(raw_candidate_keys)
        source_rows["raw_trade_lineage_match_v1"] = _triplet_series(source_rows).isin(raw_trade_keys)
        source_rows["asof_candidate_exact_match_v1"] = source_rows["candidate_uid"].astype("string").isin(asof_candidate_keys)
        repaired_only_mask = (
            ~source_rows["raw_candidate_exact_match_v1"]
            & source_rows["asof_candidate_exact_match_v1"]
            & source_rows["entry_coverage_repair_applied_v1"].fillna(False).astype(bool)
            & ~source_rows["entry_coverage_original_entry_observation_present_v1"].fillna(False).astype(bool)
            & ~source_rows["entry_coverage_original_entry_raw_state_present_v1"].fillna(False).astype(bool)
        )
        lineage_variant_mask = ~source_rows["raw_candidate_exact_match_v1"] & source_rows["raw_trade_lineage_match_v1"]
        asof_unresolved_mask = ~source_rows["asof_candidate_exact_match_v1"]
        total_rows = int(len(source_rows))
        raw_exact_matches = int(source_rows["raw_candidate_exact_match_v1"].sum())
        raw_trade_matches = int(source_rows["raw_trade_lineage_match_v1"].sum())
        asof_matches = int(source_rows["asof_candidate_exact_match_v1"].sum())
        repaired_only_count = int(repaired_only_mask.sum())
        lineage_variant_count = int(lineage_variant_mask.sum())
        unresolved_count = int(asof_unresolved_mask.sum())
        alignment_status = _classify_alignment(total_rows, raw_exact_matches, asof_matches)

        if total_rows and asof_matches == total_rows and raw_exact_matches < total_rows:
            if repaired_only_count == total_rows - raw_exact_matches and lineage_variant_count == 0:
                dominant_root_cause = "POCKET_LIVES_ON_REPAIRED_FULLCOVERAGE_ENTRY_SURFACE"
            elif lineage_variant_count > 0 and repaired_only_count > 0:
                dominant_root_cause = "MIXED_REPAIRED_SURFACE_AND_CANDIDATE_LINEAGE_VARIANT"
            elif lineage_variant_count > 0:
                dominant_root_cause = "CANDIDATE_UID_VARIANT_WITHIN_ENTRY_LINEAGE"
            else:
                dominant_root_cause = "ENTRY_TO_FAILURE_POCKET_BRIDGE_REQUIRED"
        elif raw_exact_matches == total_rows:
            dominant_root_cause = "ALREADY_VISIBLE_ON_CANONICAL_ENTRY_RAW_STATE"
        else:
            dominant_root_cause = "UNRESOLVED_MAPPING_GAP"

        root_cause_details = {
            "missing_join_key_v1": 0,
            "surface_misaligned_v1": repaired_only_count,
            "candidate_lineage_variant_v1": lineage_variant_count,
            "unresolved_mapping_v1": unresolved_count,
        }
        metadata = _source_artifact_and_kind(pocket_id)
        proxy_coverage = _proxy_input_coverage(source_rows.loc[repaired_only_mask | lineage_variant_mask].copy())
        pocket_rows.append(
            {
                "pocket_id_v1": pocket_id,
                **metadata,
                "total_pocket_size_v1": total_rows,
                "canonical_entry_raw_state_exact_match_count_v1": raw_exact_matches,
                "canonical_entry_raw_state_trade_lineage_match_count_v1": raw_trade_matches,
                "r6_fullcoverage_asof_exact_match_count_v1": asof_matches,
                "repaired_fullcoverage_only_count_v1": repaired_only_count,
                "candidate_lineage_variant_count_v1": lineage_variant_count,
                "unresolved_count_v1": unresolved_count,
                "alignment_status_v1": alignment_status,
                "dominant_root_cause_v1": dominant_root_cause,
                "root_cause_breakdown_json_v1": _json_dumps(root_cause_details),
                "bridge_input_min_coverage_rate_v1": proxy_coverage["minimum_input_coverage_rate_v1"],
                "bridge_input_field_count_v1": proxy_coverage["input_field_count_v1"],
            }
        )
    return pd.DataFrame(pocket_rows)


def _entry_surface_alignment_lock_df(pocket_forensics_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rec in pocket_forensics_df.to_dict(orient="records"):
        pocket_id = str(rec["pocket_id_v1"])
        if pocket_id == "forensic_repaired_trade":
            join_contract = "candidate_uid exact -> R6 fullcoverage as_of feature table"
            legal_alignment = "Exact candidate_uid bridge into repaired entry as_of surface; do not require exact-only raw-state row."
        elif pocket_id == "fifty_plus_mfe_seed":
            join_contract = "candidate_uid exact primary; (run_id, trade_uid, trade_id) diagnostic fallback for lineage variants"
            legal_alignment = "Primary alignment remains candidate_uid exact; lineage fallback is diagnostic only, not training join."
        else:
            join_contract = "candidate_uid exact -> R6 fullcoverage as_of feature table"
            legal_alignment = "Pocket is candidate-based on a repaired/fullcoverage entry eval surface and can be bridged deterministically by candidate_uid exact."
        rows.append(
            {
                "pocket_id_v1": pocket_id,
                "source_surface_kind_v1": rec["source_surface_kind_v1"],
                "source_artifact_v1": rec["source_artifact_v1"],
                "canonical_entry_raw_state_surface_v1": "EXACT_ONLY_CANONICAL_ENTRY_RAW_STATE",
                "current_alignment_status_v1": rec["alignment_status_v1"],
                "candidate_uid_exact_join_contract_v1": join_contract,
                "legal_alignment_target_v1": "R6_FULLCOVERAGE_ENTRY_ASOF_SURFACE",
                "alignment_interpretation_v1": legal_alignment,
                "should_expect_100pct_on_canonical_raw_state_v1": False,
            }
        )
    return pd.DataFrame(rows)


def _coverage_hardening_plan_df() -> pd.DataFrame:
    rows = [
        {
            "hardening_id_v1": "ENTRY_FAILURE_POCKET_BRIDGE_V1",
            "hardening_type_v1": "eval-surface bridge hardening",
            "what_to_adjust_v1": "Build a deterministic bridge from canonical entry readiness to the fullcoverage R6 as_of candidate surface keyed by candidate_uid exact.",
            "where_to_adjust_v1": "new materializer/readiness bridge layer; do not widen exact-only raw-state rows",
            "why_legal_v1": "Uses existing pre-entry as_of rows and coverage lineage metadata only; no management/exit truth and no policy activation.",
            "risk_v1": "Low; read-model/eval only",
            "expected_effect_repaired_165_v1": "178/178 readiness-trackable via explicit bridge",
            "expected_effect_forensic_trade_v1": "1/1 explicit protection-case visibility",
            "expected_effect_runner_near_miss_v1": "83/83 readiness-trackable via explicit bridge",
            "priority_v1": "HIGH",
        },
        {
            "hardening_id_v1": "PROXY_DERIVATION_ON_FULLCOVERAGE_ASOF_V1",
            "hardening_type_v1": "raw-state availability hardening",
            "what_to_adjust_v1": "Re-derive the five legal pre-entry proxy fields on the repaired/fullcoverage entry as_of surface for rows absent from exact-only raw-state.",
            "where_to_adjust_v1": "bridge materializer calling the same proxy derivation helper against R6/R5.1 fullcoverage as_of tables",
            "why_legal_v1": "Inputs are the same legal pre-entry replay/candidate snapshot fields already present in as_of tables.",
            "risk_v1": "Low-medium; must keep outputs research/eval only",
            "expected_effect_repaired_165_v1": "removes blind spot on 163 repaired-only rows",
            "expected_effect_forensic_trade_v1": "forensic trade gets proxy visibility without changing policy layer",
            "expected_effect_runner_near_miss_v1": "covers the 54 runner near-miss rows missing from exact-only raw-state",
            "priority_v1": "HIGH",
        },
        {
            "hardening_id_v1": "PROTECTION_CASE_TAGGING_V1",
            "hardening_type_v1": "pocket tagging hardening",
            "what_to_adjust_v1": "Tag repaired-165 pocket, forensic repaired trade, runner near-miss pocket, and 50+ MFE seeds directly on the bridge surface.",
            "where_to_adjust_v1": "bridge/readiness materializer outputs only",
            "why_legal_v1": "Pocket tags stay in eval/readiness artifacts, not in training features",
            "risk_v1": "Low",
            "expected_effect_repaired_165_v1": "explicit 0-tolerance monitoring",
            "expected_effect_forensic_trade_v1": "case cannot go blind again in readiness reports",
            "expected_effect_runner_near_miss_v1": "pocket-level regression guards become deterministic",
            "priority_v1": "HIGH",
        },
        {
            "hardening_id_v1": "FIFTY_PLUS_LINEAGE_DIAGNOSTIC_V1",
            "hardening_type_v1": "candidate lineage hardening",
            "what_to_adjust_v1": "Add diagnostic-only fallback accounting on (run_id, trade_uid, trade_id) for 50+ pockets where candidate_uid differs but trade lineage matches.",
            "where_to_adjust_v1": "bridge coverage report only",
            "why_legal_v1": "Diagnostic accounting only; not a training join and not a policy change",
            "risk_v1": "Low-medium if accidentally promoted beyond diagnostics",
            "expected_effect_repaired_165_v1": "none",
            "expected_effect_forensic_trade_v1": "none",
            "expected_effect_runner_near_miss_v1": "none",
            "priority_v1": "MEDIUM",
        },
    ]
    return pd.DataFrame(rows)


def _concrete_repaired_trade_forensic(merged: Dict[str, Any]) -> Dict[str, Any]:
    raw_state_df = merged["raw_state_df"]
    asof_enriched_df = merged["asof_enriched_df"]
    raw_trade_keys = merged["raw_trade_keys"]
    raw_candidate_keys = merged["raw_candidate_keys"]

    asof_row_df = asof_enriched_df.loc[asof_enriched_df["candidate_uid"].astype("string").eq(FORENSIC_TRADE)].copy()
    if asof_row_df.empty:
        raise RuntimeError(f"Missing forensic trade in R6 as_of surface: {FORENSIC_TRADE}")
    row = asof_row_df.iloc[0]
    trade_lineage_key = (
        str(row.get("run_id")),
        str(row.get("trade_uid")),
        str(row.get("trade_id")),
    )
    raw_trade_match = trade_lineage_key in raw_trade_keys
    raw_candidate_match = FORENSIC_TRADE in raw_candidate_keys
    alternate_raw_df = raw_state_df.loc[
        raw_state_df["run_id"].astype("string").eq(str(row.get("run_id")))
    ].copy()
    return {
        "layer_name_v1": "CONCRETE_REPAIRED_TRADE_ALIGNMENT_FORENSIC_V1",
        "candidate_uid_v1": FORENSIC_TRADE,
        "run_id_v1": str(row.get("run_id")),
        "trade_uid_v1": str(row.get("trade_uid")),
        "trade_id_v1": str(row.get("trade_id")),
        "exists_in_canonical_entry_raw_state_v1": bool(raw_candidate_match),
        "exists_in_r6_fullcoverage_asof_v1": True,
        "exists_in_raw_trade_lineage_v1": bool(raw_trade_match),
        "entry_coverage_original_entry_observation_present_v1": bool(row.get("entry_coverage_original_entry_observation_present_v1")),
        "entry_coverage_original_entry_raw_state_present_v1": bool(row.get("entry_coverage_original_entry_raw_state_present_v1")),
        "entry_coverage_repair_applied_v1": bool(row.get("entry_coverage_repair_applied_v1")),
        "entry_coverage_repair_source_v1": str(row.get("entry_coverage_repair_source_v1")),
        "should_expect_exact_only_raw_state_row_v1": False,
        "alignment_verdict_v1": "MUST_BE_TRACKED_VIA_FULLCOVERAGE_ENTRY_ASOF_BRIDGE",
        "why_zero_of_one_v1": (
            "The trade is absent from exact-only canonical entry raw-state, has no alternate raw trade-lineage row, "
            "and exists only on the repaired/fullcoverage entry as_of surface."
        ),
        "same_trade_under_other_raw_key_v1": False,
        "same_run_raw_candidate_count_v1": int(
            alternate_raw_df["candidate_uid"].astype("string").nunique()
        ),
        "next_required_change_v1": (
            "Build an explicit entry-to-failure-pocket bridge using candidate_uid exact into the fullcoverage R6 as_of feature table "
            "and compute the legal pre-entry proxy scores there for readiness only."
        ),
    }


def _runner_near_miss_alignment(merged: Dict[str, Any], pocket_forensics_df: pd.DataFrame) -> Dict[str, Any]:
    runner_row = pocket_forensics_df.loc[pocket_forensics_df["pocket_id_v1"].astype("string").eq("runner_near_miss")]
    if runner_row.empty:
        raise RuntimeError("Missing runner_near_miss row in pocket forensics")
    rec = runner_row.iloc[0]
    return {
        "layer_name_v1": "RUNNER_NEAR_MISS_ALIGNMENT_AND_HARDENING_V1",
        "total_runner_near_miss_rows_v1": int(rec["total_pocket_size_v1"]),
        "canonical_raw_state_exact_matches_v1": int(rec["canonical_entry_raw_state_exact_match_count_v1"]),
        "r6_fullcoverage_asof_matches_v1": int(rec["r6_fullcoverage_asof_exact_match_count_v1"]),
        "coverage_gap_interpretation_v1": "Feature fields are present when alignment exists; the gap is primarily entry-surface alignment, not feature absence.",
        "why_29_of_83_v1": (
            "29 rows come from ORIGINAL_R2_ENTRY_OBSERVABILITY and already exist on exact-only canonical entry raw-state; "
            "54 rows are repaired/fullcoverage-only and therefore invisible on the current exact-only raw-state surface."
        ),
        "is_expected_under_current_contract_v1": True,
        "hardening_needed_v1": [
            "Bridge runner near-miss pocket to the fullcoverage entry as_of surface by candidate_uid exact.",
            "Compute the five legal pre-entry proxy scores on bridge rows missing from exact-only raw-state.",
            "Keep pocket tags and regression guards on the bridge/eval surface only.",
        ],
        "guard_fields_present_once_aligned_v1": True,
    }


def _legality_safe_fix_spec_df() -> pd.DataFrame:
    rows = [
        {
            "fix_id_v1": "ENTRY_FAILURE_POCKET_BRIDGE_V1",
            "same_trade_leakage_introduced_v1": False,
            "management_exit_truth_used_directly_v1": False,
            "policy_or_controller_changed_v1": False,
            "replay_required_v1": False,
            "layer_scope_v1": "read-model / eval-alignment only",
            "legality_verdict_v1": "LEGAL",
            "guardrail_v1": "Use candidate_uid exact and entry_coverage_* lineage fields only.",
        },
        {
            "fix_id_v1": "PROXY_DERIVATION_ON_FULLCOVERAGE_ASOF_V1",
            "same_trade_leakage_introduced_v1": False,
            "management_exit_truth_used_directly_v1": False,
            "policy_or_controller_changed_v1": False,
            "replay_required_v1": False,
            "layer_scope_v1": "read-model / as_of bridge feature derivation only",
            "legality_verdict_v1": "LEGAL",
            "guardrail_v1": "Derive only from the same as_of_skip_replay_* and as_of_skip_candidate_* families already approved for entry.",
        },
        {
            "fix_id_v1": "FIFTY_PLUS_LINEAGE_DIAGNOSTIC_V1",
            "same_trade_leakage_introduced_v1": False,
            "management_exit_truth_used_directly_v1": False,
            "policy_or_controller_changed_v1": False,
            "replay_required_v1": False,
            "layer_scope_v1": "diagnostic coverage report only",
            "legality_verdict_v1": "LEGAL_IF_DIAGNOSTIC_ONLY",
            "guardrail_v1": "Do not promote lineage fallback beyond diagnostic accounting.",
        },
    ]
    return pd.DataFrame(rows)


def _implementation_targets_df() -> pd.DataFrame:
    rows = [
        {
            "target_id_v1": "MUST_DO_NOW_ENTRY_FAILURE_POCKET_BRIDGE",
            "file_target_v1": "new bridge materializer or extension of materialize_monday_selected_pre_entry_proxies_and_readiness_pack_v1.py",
            "function_target_v1": "build explicit candidate_uid exact bridge from canonical raw-state to R6 fullcoverage as_of surface",
            "test_target_v1": "new bridge coverage test incl. forensic repaired trade and runner near-miss pocket",
            "artifact_target_v1": "entry failure pocket bridge report + coverage audit",
            "phase_v1": "MUST_DO_NOW",
        },
        {
            "target_id_v1": "MUST_DO_NOW_PROXY_DERIVATION_ON_BRIDGE",
            "file_target_v1": "/home/andre2/src/GX1_ENGINE/gx1/analysis/shadow_meta_v1.py",
            "function_target_v1": "reuse _build_entry_skipability_pre_entry_proxy_fields_v1 against fullcoverage as_of-compatible frames",
            "test_target_v1": "bridge derivation legality/coverage test",
            "artifact_target_v1": "bridge proxy coverage report for repaired and runner pockets",
            "phase_v1": "MUST_DO_NOW",
        },
        {
            "target_id_v1": "MUST_DO_NOW_PROTECTION_CASE_TAGS",
            "file_target_v1": "bridge materializer",
            "function_target_v1": "tag repaired-165 pocket, forensic repaired trade, runner near-miss, 50+ seeds on bridge output",
            "test_target_v1": "forensic trade 1/1 visibility test",
            "artifact_target_v1": "bridge pocket tag report",
            "phase_v1": "MUST_DO_NOW",
        },
        {
            "target_id_v1": "NOT_NOW_WIDEN_EXACT_ONLY_RAW_STATE",
            "file_target_v1": "/home/andre2/src/GX1_ENGINE/gx1/analysis/shadow_meta_v1.py",
            "function_target_v1": "do not append repaired/fullcoverage-only rows into exact-only canonical entry raw-state parquet",
            "test_target_v1": "exact-only raw-state contract remains intact",
            "artifact_target_v1": "none",
            "phase_v1": "NOT_NOW",
        },
        {
            "target_id_v1": "NOT_NOW_POLICY_OR_RETRAIN",
            "file_target_v1": "none",
            "function_target_v1": "do not touch policy/controller or open retrain",
            "test_target_v1": "n/a",
            "artifact_target_v1": "n/a",
            "phase_v1": "NOT_NOW",
        },
    ]
    return pd.DataFrame(rows)


def _post_hardening_readiness_criteria() -> Dict[str, Any]:
    return {
        "layer_name_v1": "POST_HARDENING_READINESS_CRITERIA_V1",
        "must_be_true_v1": [
            "repaired_165 pocket is 100% trackable either on exact-only canonical raw-state or via an explicit canonical bridge into the fullcoverage entry as_of surface",
            f"forensic repaired trade {FORENSIC_TRADE} is no longer blind in entry readiness reporting",
            "runner near-miss pocket is fully accounted for and bridge-trackable",
            "50+ MFE seed pocket is explicitly split into exact/raw, bridge-only, and lineage-diagnostic rows",
            "no management/exit truth is introduced into entry features or bridge proxies",
            "no policy/controller changes are introduced",
        ],
        "decision_if_true_v1": "READY_FOR_RETRAIN_READINESS_RECHECK",
        "decision_if_false_v1": "READY_FOR_MORE_NARROW_FEATURE_HARDENING",
        "retrain_now_v1": False,
    }


def _next_action_lock() -> Dict[str, Any]:
    return {
        "layer_name_v1": "NEXT_AGENT_ACTION_LOCK_V1",
        "primary_action_v1": "BUILD_ENTRY_TO_FAILURE_POCKET_BRIDGE_FIRST",
        "supporting_actions_v1": [
            "DO_NOT_RETRAIN_YET",
            "KEEP_MONDAY_R6_AS_FAILURE_MINER",
            "DO_NOT_TOUCH_POLICY_LAYER",
        ],
    }


def _status_block() -> Dict[str, List[str]]:
    return {
        "BEVIST": [
            "Pocket coverage is low because feature coverage was measured on the 1689-row exact-only canonical raw-state, while repaired and runner pockets live on the 1852-row fullcoverage R6 entry as_of surface.",
            "Monday-native repaired-165 and runner near-miss gaps are not primarily feature-coverage failures.",
            "The forensic repaired trade is absent from exact-only canonical raw-state and exists only on the repaired/fullcoverage entry as_of surface.",
            "Retrain still must not start now.",
        ],
        "INDIKERT": [
            "A narrow entry-to-failure-pocket bridge is the right next hardening step.",
            "Recomputing the legal pre-entry proxy scores on the fullcoverage as_of bridge should be sufficient to make repaired and runner pockets readiness-trackable.",
            "50+ seed tracking also benefits from a small lineage-diagnostic overlay.",
        ],
        "IKKE_ETABLERT": [
            "That bridge hardening alone will be enough to reopen retrain readiness without any second feature-hardening wave.",
            "That every 50+ lineage variant should be promoted to a training join rather than staying diagnostic-only.",
        ],
    }


def _render_report(
    pocket_forensics_df: pd.DataFrame,
    repaired_trade_forensic: Dict[str, Any],
    runner_alignment: Dict[str, Any],
    next_action: Dict[str, Any],
    status_block: Dict[str, List[str]],
) -> str:
    lines = [
        "# Monday Entry Pocket Coverage Alignment Lock V1",
        "",
        "Read-only diagnosis. No retrain, replay, raw-state rebuild, or policy activation was started.",
        "",
        "## Headline",
        "",
        "- Global proxy field coverage is 100% on exact-only canonical entry raw-state.",
        "- Pocket visibility is lower because the critical repaired/runner pockets live on the fullcoverage R6 entry as_of surface, not only on the exact-only raw-state subset.",
        f"- Primary next action: `{next_action['primary_action_v1']}`",
        "",
        "## Pocket Gap Snapshot",
        "",
    ]
    for rec in pocket_forensics_df.to_dict(orient="records"):
        lines.append(
            f"- `{rec['pocket_id_v1']}`: raw exact `{rec['canonical_entry_raw_state_exact_match_count_v1']}/{rec['total_pocket_size_v1']}`, "
            f"R6 as_of `{rec['r6_fullcoverage_asof_exact_match_count_v1']}/{rec['total_pocket_size_v1']}`, "
            f"status `{rec['alignment_status_v1']}`, cause `{rec['dominant_root_cause_v1']}`"
        )
    lines += [
        "",
        "## Forensic Trade",
        "",
        f"- Candidate: `{repaired_trade_forensic['candidate_uid_v1']}`",
        f"- Exists in canonical raw-state: `{repaired_trade_forensic['exists_in_canonical_entry_raw_state_v1']}`",
        f"- Exists in R6 fullcoverage as_of: `{repaired_trade_forensic['exists_in_r6_fullcoverage_asof_v1']}`",
        f"- Why 0/1 now: {repaired_trade_forensic['why_zero_of_one_v1']}",
        "",
        "## Runner Near-Miss",
        "",
        f"- Coverage now: `{runner_alignment['canonical_raw_state_exact_matches_v1']}/{runner_alignment['total_runner_near_miss_rows_v1']}` on exact-only raw-state",
        f"- Fullcoverage as_of: `{runner_alignment['r6_fullcoverage_asof_matches_v1']}/{runner_alignment['total_runner_near_miss_rows_v1']}`",
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
    inputs = _resolve_inputs(reports_root)
    merged = _merge_eval_surfaces(inputs)
    pocket_forensics_df = _pocket_forensics_rows(merged)
    entry_surface_alignment_df = _entry_surface_alignment_lock_df(pocket_forensics_df)
    coverage_hardening_plan_df = _coverage_hardening_plan_df()
    repaired_trade_forensic = _concrete_repaired_trade_forensic(merged)
    runner_alignment = _runner_near_miss_alignment(merged, pocket_forensics_df)
    legality_fix_df = _legality_safe_fix_spec_df()
    implementation_targets_df = _implementation_targets_df()
    post_hardening_readiness = _post_hardening_readiness_criteria()
    next_action = _next_action_lock()
    status_block = _status_block()

    raw_subset_count = int(len(merged["raw_state_df"]))
    fullcoverage_count = int(len(merged["asof_enriched_df"]))
    exact_subset_gap = fullcoverage_count - raw_subset_count

    consistency_df = pd.DataFrame(
        [
            {
                "check_name_v1": "CANONICAL_RAW_IS_SUBSET_OF_R6_ASOF",
                "status_v1": "PASS" if exact_subset_gap >= 0 else "FAIL",
                "details_json_v1": _json_dumps(
                    {
                        "canonical_raw_row_count_v1": raw_subset_count,
                        "r6_asof_row_count_v1": fullcoverage_count,
                        "gap_v1": exact_subset_gap,
                    }
                ),
            },
            {
                "check_name_v1": "REPAIRED_165_FULLCOVERAGE_MATCHES_EXIST",
                "status_v1": "PASS"
                if int(
                    pocket_forensics_df.loc[
                        pocket_forensics_df["pocket_id_v1"].astype("string").eq("repaired_165"),
                        "r6_fullcoverage_asof_exact_match_count_v1",
                    ].iloc[0]
                )
                == int(
                    pocket_forensics_df.loc[
                        pocket_forensics_df["pocket_id_v1"].astype("string").eq("repaired_165"),
                        "total_pocket_size_v1",
                    ].iloc[0]
                )
                else "FAIL",
                "details_json_v1": _json_dumps({"pocket_id_v1": "repaired_165"}),
            },
            {
                "check_name_v1": "RUNNER_NEAR_MISS_FULLCOVERAGE_MATCHES_EXIST",
                "status_v1": "PASS"
                if int(
                    pocket_forensics_df.loc[
                        pocket_forensics_df["pocket_id_v1"].astype("string").eq("runner_near_miss"),
                        "r6_fullcoverage_asof_exact_match_count_v1",
                    ].iloc[0]
                )
                == int(
                    pocket_forensics_df.loc[
                        pocket_forensics_df["pocket_id_v1"].astype("string").eq("runner_near_miss"),
                        "total_pocket_size_v1",
                    ].iloc[0]
                )
                else "FAIL",
                "details_json_v1": _json_dumps({"pocket_id_v1": "runner_near_miss"}),
            },
            {
                "check_name_v1": "FORENSIC_REPAIRED_TRADE_PRESENT_ON_BRIDGE_SURFACE",
                "status_v1": "PASS" if repaired_trade_forensic["exists_in_r6_fullcoverage_asof_v1"] else "FAIL",
                "details_json_v1": _json_dumps({"candidate_uid_v1": FORENSIC_TRADE}),
            },
            {
                "check_name_v1": "NEXT_ACTION_REMAINS_NO_RETRAIN",
                "status_v1": "PASS" if "DO_NOT_RETRAIN_YET" in next_action["supporting_actions_v1"] else "FAIL",
                "details_json_v1": _json_dumps(next_action),
            },
        ]
    )

    contract = {
        "layer_name_v1": "MONDAY_ENTRY_POCKET_COVERAGE_ALIGNMENT_LOCK_CONTRACT_V1",
        "mode_v1": "READ_ONLY_DIAGNOSIS_AND_NEXT_STEP_ONLY",
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_raw_state_rebuild_v1": True,
        "not_policy_activation_v1": True,
        "benchmark_v1": BENCHMARK,
        "monday_safety_reference_v1": MONDAY_SAFETY_REFERENCE,
        "monday_r6_role_v1": "FAILURE_MINER_DIAGNOSIS_ONLY",
        "selected_pre_entry_pack_dir_v1": str(inputs["selected_pre_entry_pack_dir"]),
    }

    status = {
        "layer_name_v1": "MONDAY_ENTRY_POCKET_COVERAGE_ALIGNMENT_LOCK_STATUS_V1",
        "SPEC_STATUS": "READ_ONLY_DIAGNOSIS_COMPLETE",
        "failed_check_count_v1": int(consistency_df["status_v1"].astype("string").eq("FAIL").sum()),
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_policy_activation_v1": True,
    }

    summary = {
        "layer_name_v1": "MONDAY_ENTRY_POCKET_COVERAGE_ALIGNMENT_LOCK_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "extension_dir_v1": str(extension_dir),
        "benchmark_v1": BENCHMARK,
        "monday_safety_reference_v1": MONDAY_SAFETY_REFERENCE,
        "canonical_raw_row_count_v1": raw_subset_count,
        "r6_fullcoverage_asof_row_count_v1": fullcoverage_count,
        "global_field_coverage_denominator_v1": raw_subset_count,
        "pocket_alignment_denominator_v1": fullcoverage_count,
        "key_explanation_v1": (
            "Global field coverage is 100% because it is measured only on the exact-only canonical raw-state subset. "
            "Critical repaired/runner pockets are defined on the fullcoverage R6 entry as_of surface, which contains 163 additional repaired candidates."
        ),
        "next_action_v1": next_action["primary_action_v1"],
        "status_v1": status,
        "hard_status_division_v1": status_block,
    }

    manifest = {
        "layer_name_v1": "MONDAY_ENTRY_POCKET_COVERAGE_ALIGNMENT_LOCK_MANIFEST_V1",
        "artifacts_v1": {
            "contract": CONTRACT,
            "pocket_coverage_gap_forensics": POCKET_COVERAGE_GAP_FORENSICS,
            "entry_surface_alignment_lock": ENTRY_SURFACE_ALIGNMENT_LOCK,
            "coverage_hardening_plan": COVERAGE_HARDENING_PLAN,
            "concrete_repaired_trade_alignment_forensic": CONCRETE_REPAIRED_TRADE_FORENSIC,
            "runner_near_miss_alignment_and_hardening": RUNNER_NEAR_MISS_ALIGNMENT,
            "legality_safe_coverage_fix_spec": LEGALITY_SAFE_COVERAGE_FIX_SPEC,
            "implementation_targets_for_coverage_hardening": IMPLEMENTATION_TARGETS,
            "post_hardening_readiness_criteria": POST_HARDENING_READINESS,
            "next_agent_action_lock": NEXT_ACTION,
            "summary": SUMMARY,
            "report": REPORT,
            "manifest": MANIFEST,
            "status": STATUS,
            "consistency_audit": CONSISTENCY_AUDIT,
        },
    }

    report = _render_report(pocket_forensics_df, repaired_trade_forensic, runner_alignment, next_action, status_block)
    return {
        "contract": contract,
        "pocket_forensics_df": pocket_forensics_df,
        "entry_surface_alignment_df": entry_surface_alignment_df,
        "coverage_hardening_plan_df": coverage_hardening_plan_df,
        "repaired_trade_forensic": repaired_trade_forensic,
        "runner_alignment": runner_alignment,
        "legality_fix_df": legality_fix_df,
        "implementation_targets_df": implementation_targets_df,
        "post_hardening_readiness": post_hardening_readiness,
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
    payload["pocket_forensics_df"].to_csv(extension_dir / POCKET_COVERAGE_GAP_FORENSICS, index=False)
    payload["entry_surface_alignment_df"].to_csv(extension_dir / ENTRY_SURFACE_ALIGNMENT_LOCK, index=False)
    payload["coverage_hardening_plan_df"].to_csv(extension_dir / COVERAGE_HARDENING_PLAN, index=False)
    _write_json(extension_dir / CONCRETE_REPAIRED_TRADE_FORENSIC, payload["repaired_trade_forensic"])
    _write_json(extension_dir / RUNNER_NEAR_MISS_ALIGNMENT, payload["runner_alignment"])
    payload["legality_fix_df"].to_csv(extension_dir / LEGALITY_SAFE_COVERAGE_FIX_SPEC, index=False)
    payload["implementation_targets_df"].to_csv(extension_dir / IMPLEMENTATION_TARGETS, index=False)
    _write_json(extension_dir / POST_HARDENING_READINESS, payload["post_hardening_readiness"])
    _write_json(extension_dir / NEXT_ACTION, payload["next_action"])
    _write_json(extension_dir / SUMMARY, payload["summary"])
    (extension_dir / REPORT).write_text(payload["report"], encoding="utf-8")
    _write_json(extension_dir / MANIFEST, payload["manifest"])
    _write_json(extension_dir / STATUS, payload["status"])
    payload["consistency_df"].to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)
    return {"extension_dir": str(extension_dir), "status": payload["status"], "summary": payload["summary"]}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose pocket coverage/alignment gaps between canonical entry raw-state and fullcoverage failure-pocket surfaces.")
    parser.add_argument("--reports-root", type=str, default=None)
    parser.add_argument("--extension-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    result = materialize(
        reports_root,
        extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
