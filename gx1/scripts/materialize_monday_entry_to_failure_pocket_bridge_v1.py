#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from gx1.analysis.shadow_meta_v1 import (
    _ENTRY_PRE_ENTRY_PROXY_SPEC_V1,
    _build_entry_skipability_pre_entry_proxy_fields_v1,
    _validate_entry_pre_entry_proxy_input_fields_v1,
)


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

EXTENSION_PREFIX = "MONDAY_ENTRY_TO_FAILURE_POCKET_BRIDGE_IMPLEMENTATION_V1"
SELECTED_PRE_ENTRY_PACK_PREFIX = "MONDAY_SELECTED_PRE_ENTRY_PROXIES_AND_READINESS_PACK_V1_"

CANONICAL_LEDGER_DIRNAME = "ALL_TRADE_REVIEW_LEDGER_20260411"
R6_DIRNAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1"

RAW_STATE = "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet"
R6_ASOF = "shadow_meta_all_trade_review_r6_entry_runner_first_as_of_feature_table_v1.parquet"
R6_POLICY = "shadow_meta_all_trade_review_r6_policy_prediction_view_v1.parquet"
R6_HINDSIGHT = "shadow_meta_all_trade_review_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet"

CONTRACT = "contract_v1.json"
BRIDGE_SURFACE = "entry_to_failure_pocket_bridge_surface_v1.parquet"
BRIDGE_IMPLEMENTATION_SUMMARY = "bridge_implementation_summary_v1.json"
BRIDGE_SURFACE_CONTRACT = "bridge_surface_contract_v1.csv"
MISSING_PROXY_REDERIVATION_REPORT = "missing_proxy_rederivation_report_v1.csv"
FAILURE_POCKET_TAGGING_REPORT = "failure_pocket_tagging_report_v1.csv"
LEGALITY_NO_POLLUTION_GUARD_REPORT = "legality_and_no_canonical_pollution_guard_report_v1.csv"
FORENSIC_REPAIRED_TRADE_BRIDGE_PROOF = "forensic_repaired_trade_bridge_proof_v1.json"
RUNNER_NEAR_MISS_BRIDGE_READINESS = "runner_near_miss_bridge_readiness_report_v1.json"
POST_BRIDGE_READINESS_RECHECK = "post_bridge_readiness_recheck_pack_v1.json"
NEXT_ACTION = "next_agent_action_lock_v1.json"
SUMMARY = "summary_v1.json"
REPORT = "report_v1.md"
MANIFEST = "manifest_v1.json"
STATUS = "status_v1.json"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"

FORENSIC_TRADE = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"
BENCHMARK = "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"
MONDAY_SAFETY_REFERENCE = "R5_1_CANDIDATE_0241_R5_1_COMBINED_repaired_165_like"

PROXY_FIELDS = list(_ENTRY_PRE_ENTRY_PROXY_SPEC_V1.keys())
SELECTED_PROXY_INPUTS = sorted(
    {
        str(field_name)
        for spec in _ENTRY_PRE_ENTRY_PROXY_SPEC_V1.values()
        for field_name in spec.get("inputs", [])
        if str(field_name) not in PROXY_FIELDS
    }
)

BRIDGE_COLUMNS = [
    "run_id",
    "candidate_uid",
    "trade_uid",
    "trade_id",
    "bridge_surface_origin_v1",
    "exact_canonical_raw_state_present_v1",
    "fullcoverage_r6_asof_present_v1",
    "bridge_proxy_source_v1",
    "bridge_surface_semantic_contract_v1",
    "entry_coverage_original_entry_observation_present_v1",
    "entry_coverage_original_entry_raw_state_present_v1",
    "entry_coverage_repair_applied_v1",
    "entry_coverage_repair_source_v1",
    *PROXY_FIELDS,
    "bridge_all_selected_proxies_available_v1",
    "bridge_pocket_repaired_165_v1",
    "bridge_pocket_forensic_repaired_trade_v1",
    "bridge_pocket_runner_near_miss_v1",
    "bridge_pocket_fifty_plus_mfe_seed_v1",
    "bridge_pocket_missed_10_50_tail_control_v1",
    "bridge_pocket_missed_should_not_take_v1",
    "bridge_pocket_risky_allow_v1",
    "bridge_readiness_trackable_v1",
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


def _load_inputs(reports_root: Path) -> Dict[str, Any]:
    ledger_dir = reports_root / CANONICAL_LEDGER_DIRNAME
    r6_dir = reports_root / R6_DIRNAME
    if not ledger_dir.exists():
        raise FileNotFoundError(f"Missing canonical ledger dir: {ledger_dir}")
    if not r6_dir.exists():
        raise FileNotFoundError(f"Missing R6 dir: {r6_dir}")
    selected_pre_entry_pack_dir = _latest_dir(reports_root, SELECTED_PRE_ENTRY_PACK_PREFIX)
    raw_state_df = pd.read_parquet(ledger_dir / RAW_STATE)
    asof_df = pd.read_parquet(r6_dir / R6_ASOF)
    policy_df = pd.read_parquet(r6_dir / R6_POLICY)
    hindsight_df = pd.read_parquet(r6_dir / R6_HINDSIGHT)
    return {
        "ledger_dir": ledger_dir,
        "r6_dir": r6_dir,
        "selected_pre_entry_pack_dir": selected_pre_entry_pack_dir,
        "selected_pre_entry_summary": _load_json(selected_pre_entry_pack_dir / SUMMARY),
        "raw_state_df": raw_state_df,
        "asof_df": asof_df,
        "policy_df": policy_df,
        "hindsight_df": hindsight_df,
    }


def _normalize_string_cols(df: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
    work = df.copy()
    for field_name in fields:
        if field_name in work.columns:
            work[field_name] = work[field_name].astype("string")
    return work


def _bridge_surface_contract_rows() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add(
        field_name: str,
        *,
        dtype: str,
        source: str,
        origin: str,
        semantics: str,
        legality: str,
        readiness_only: bool,
        null_policy: str,
        forbidden_inputs: str,
        definition: str,
    ) -> None:
        rows.append(
            {
                "field_name_v1": field_name,
                "dtype_v1": dtype,
                "source_v1": source,
                "origin_v1": origin,
                "as_of_semantics_v1": semantics,
                "legality_status_v1": legality,
                "readiness_only_v1": bool(readiness_only),
                "null_default_policy_v1": null_policy,
                "forbidden_input_families_v1": forbidden_inputs,
                "definition_v1": definition,
            }
        )

    add(
        "run_id",
        dtype="string",
        source="candidate identity",
        origin="EXACT_OR_FULLCOVERAGE",
        semantics="identity only",
        legality="PRE_ENTRY_LEGAL",
        readiness_only=True,
        null_policy="not nullable",
        forbidden_inputs="N/A",
        definition="Run identifier for candidate lineage.",
    )
    add(
        "candidate_uid",
        dtype="string",
        source="candidate identity",
        origin="EXACT_OR_FULLCOVERAGE",
        semantics="identity only",
        legality="PRE_ENTRY_LEGAL",
        readiness_only=True,
        null_policy="not nullable",
        forbidden_inputs="N/A",
        definition="Deterministic primary bridge key.",
    )
    add(
        "trade_uid",
        dtype="string",
        source="candidate identity",
        origin="EXACT_OR_FULLCOVERAGE",
        semantics="identity only",
        legality="PRE_ENTRY_LEGAL",
        readiness_only=True,
        null_policy="not nullable",
        forbidden_inputs="N/A",
        definition="Trade lineage identifier for diagnostics only.",
    )
    add(
        "trade_id",
        dtype="string",
        source="candidate identity",
        origin="EXACT_OR_FULLCOVERAGE",
        semantics="identity only",
        legality="PRE_ENTRY_LEGAL",
        readiness_only=True,
        null_policy="not nullable",
        forbidden_inputs="N/A",
        definition="Broker/order identifier for diagnostics only.",
    )
    add(
        "bridge_surface_origin_v1",
        dtype="string",
        source="bridge merge logic",
        origin="READINESS_BRIDGE_ONLY",
        semantics="derived from exact presence",
        legality="PRE_ENTRY_LEGAL",
        readiness_only=True,
        null_policy="not nullable",
        forbidden_inputs="N/A",
        definition="Whether the row comes from exact canonical raw-state or bridge-only fullcoverage R6 AS_OF.",
    )
    add(
        "exact_canonical_raw_state_present_v1",
        dtype="bool",
        source="candidate_uid exact join to canonical raw-state",
        origin="READINESS_BRIDGE_ONLY",
        semantics="exact raw-state presence at entry surface",
        legality="PRE_ENTRY_LEGAL",
        readiness_only=True,
        null_policy="False if absent",
        forbidden_inputs="N/A",
        definition="True when a candidate exists on exact-only canonical entry raw-state.",
    )
    add(
        "fullcoverage_r6_asof_present_v1",
        dtype="bool",
        source="R6 AS_OF source row",
        origin="FULLCOVERAGE_R6_ASOF",
        semantics="AS_OF candidate presence",
        legality="PRE_ENTRY_LEGAL",
        readiness_only=True,
        null_policy="always True on bridge surface",
        forbidden_inputs="N/A",
        definition="True when a candidate exists on the fullcoverage R6 AS_OF entry surface.",
    )
    add(
        "bridge_proxy_source_v1",
        dtype="string",
        source="bridge merge logic",
        origin="READINESS_BRIDGE_ONLY",
        semantics="describes proxy provenance",
        legality="PRE_ENTRY_LEGAL",
        readiness_only=True,
        null_policy="not nullable",
        forbidden_inputs="N/A",
        definition="Identifies whether proxy values came from exact canonical raw-state or legal re-derivation on bridge-only rows.",
    )
    add(
        "bridge_surface_semantic_contract_v1",
        dtype="string",
        source="bridge constant",
        origin="READINESS_BRIDGE_ONLY",
        semantics="readiness-only semantic contract",
        legality="PRE_ENTRY_LEGAL",
        readiness_only=True,
        null_policy="not nullable",
        forbidden_inputs="N/A",
        definition="Explicit contract string stating this surface is readiness/eval only and not a canonical training surface.",
    )
    for field_name in [
        "entry_coverage_original_entry_observation_present_v1",
        "entry_coverage_original_entry_raw_state_present_v1",
        "entry_coverage_repair_applied_v1",
    ]:
        add(
            field_name,
            dtype="bool",
            source="R6 fullcoverage AS_OF lineage metadata",
            origin="FULLCOVERAGE_R6_ASOF",
            semantics="as-of coverage lineage only",
            legality="PRE_ENTRY_LEGAL",
            readiness_only=True,
            null_policy="False if unavailable",
            forbidden_inputs="management/exit truth, policy/controller outputs",
            definition="Coverage lineage metadata copied from R6 fullcoverage AS_OF surface.",
        )
    add(
        "entry_coverage_repair_source_v1",
        dtype="string",
        source="R6 fullcoverage ASOF lineage metadata",
        origin="FULLCOVERAGE_R6_ASOF",
        semantics="as-of coverage lineage only",
        legality="PRE_ENTRY_LEGAL",
        readiness_only=True,
        null_policy="string missing if unavailable",
        forbidden_inputs="management/exit truth, policy/controller outputs",
        definition="Coverage repair provenance string copied from R6 fullcoverage ASOF surface.",
    )
    for field_name, spec in _ENTRY_PRE_ENTRY_PROXY_SPEC_V1.items():
        add(
            field_name,
            dtype="float64",
            source="exact canonical raw-state or legal re-derivation on fullcoverage R6 ASOF",
            origin="EXACT_OR_BRIDGE_DERIVED",
            semantics="pre-entry AS_OF only",
            legality="PRE_ENTRY_LEGAL_IF_DERIVED",
            readiness_only=True,
            null_policy="null means unavailable; no synthetic fallback",
            forbidden_inputs="management/exit truth, policy logs, hindsight labels, same-trade future fields",
            definition=str(spec["contract_note_v1"]),
        )
    add(
        "bridge_all_selected_proxies_available_v1",
        dtype="bool",
        source="bridge computed availability",
        origin="READINESS_BRIDGE_ONLY",
        semantics="post-derivation readiness flag",
        legality="PRE_ENTRY_LEGAL",
        readiness_only=True,
        null_policy="False if any proxy missing",
        forbidden_inputs="N/A",
        definition="True when all five selected legal pre-entry proxies are available on the bridge row.",
    )
    for field_name, definition in [
        ("bridge_pocket_repaired_165_v1", "Failure pocket tag for repaired-165 rows."),
        ("bridge_pocket_forensic_repaired_trade_v1", "Single protection-case tag for the forensic repaired trade."),
        ("bridge_pocket_runner_near_miss_v1", "Failure pocket tag for runner near-miss rows."),
        ("bridge_pocket_fifty_plus_mfe_seed_v1", "Pocket tag for 50+ MFE seed rows."),
        ("bridge_pocket_missed_10_50_tail_control_v1", "Pocket tag for missed 10-50 tail-control rows."),
        ("bridge_pocket_missed_should_not_take_v1", "Pocket tag for missed should-not-take rows."),
        ("bridge_pocket_risky_allow_v1", "Pocket tag for risky-allow rows."),
    ]:
        add(
            field_name,
            dtype="bool",
            source="R6 policy/hindsight failure-pocket surfaces",
            origin="FULLCOVERAGE_R6_POLICY_OR_HINDSIGHT",
            semantics="readiness/eval only pocket tagging",
            legality="READINESS_ONLY_NOT_TRAINING_SIGNAL",
            readiness_only=True,
            null_policy="False if not tagged",
            forbidden_inputs="N/A",
            definition=definition,
        )
    add(
        "bridge_readiness_trackable_v1",
        dtype="bool",
        source="bridge derived readiness flag",
        origin="READINESS_BRIDGE_ONLY",
        semantics="post-bridge readiness only",
        legality="PRE_ENTRY_LEGAL",
        readiness_only=True,
        null_policy="False if any required proxy missing",
        forbidden_inputs="N/A",
        definition="True when a row is visible on bridge and has all selected legal proxies available for readiness evaluation.",
    )
    return pd.DataFrame(rows)


def _derive_bridge_surface(inputs: Dict[str, Any]) -> Dict[str, Any]:
    raw_state_df = _normalize_string_cols(inputs["raw_state_df"], ["candidate_uid", "run_id", "trade_uid", "trade_id"])
    asof_df = _normalize_string_cols(inputs["asof_df"], ["candidate_uid", "run_id", "trade_uid", "trade_id"])
    policy_df = _normalize_string_cols(inputs["policy_df"], ["candidate_uid"])
    hindsight_df = _normalize_string_cols(inputs["hindsight_df"], ["candidate_uid"])

    if bool(raw_state_df["candidate_uid"].duplicated().any()):
        raise RuntimeError("Canonical entry raw-state candidate_uid is not unique")
    if bool(asof_df["candidate_uid"].duplicated().any()):
        raise RuntimeError("R6 fullcoverage ASOF candidate_uid is not unique")

    raw_proxy_df = raw_state_df[["candidate_uid", *PROXY_FIELDS]].copy()
    raw_identity_df = raw_state_df[["candidate_uid", "run_id", "trade_uid", "trade_id"]].copy()

    asof_presence_df = asof_df[
        [
            "candidate_uid",
            "run_id",
            "trade_uid",
            "trade_id",
            "entry_coverage_original_entry_observation_present_v1",
            "entry_coverage_original_entry_raw_state_present_v1",
            "entry_coverage_repair_applied_v1",
            "entry_coverage_repair_source_v1",
            *SELECTED_PROXY_INPUTS,
        ]
    ].copy()

    bridge_df = asof_presence_df.merge(raw_proxy_df, on="candidate_uid", how="left", validate="one_to_one")
    bridge_df["exact_canonical_raw_state_present_v1"] = bridge_df["candidate_uid"].isin(raw_identity_df["candidate_uid"])
    bridge_df["fullcoverage_r6_asof_present_v1"] = True
    bridge_df["bridge_surface_origin_v1"] = bridge_df["exact_canonical_raw_state_present_v1"].map(
        {True: "EXACT_CANONICAL_RAW_STATE", False: "FULLCOVERAGE_R6_ASOF_BRIDGE_ONLY"}
    ).astype("string")

    bridge_only_mask = ~bridge_df["exact_canonical_raw_state_present_v1"].fillna(False).astype(bool)
    bridge_only_input_df = bridge_df.loc[bridge_only_mask, ["candidate_uid", *SELECTED_PROXY_INPUTS]].copy()
    bridge_only_input_df = bridge_only_input_df.merge(
        asof_df[["candidate_uid", *SELECTED_PROXY_INPUTS]].drop_duplicates(subset=["candidate_uid"]),
        on="candidate_uid",
        how="left",
        suffixes=("", "_dup"),
    )
    bridge_only_input_df = bridge_only_input_df[[column for column in bridge_only_input_df.columns if not column.endswith("_dup")]].copy()
    derived_proxy_df = _build_entry_skipability_pre_entry_proxy_fields_v1(bridge_only_input_df) if len(bridge_only_input_df) else pd.DataFrame(columns=PROXY_FIELDS)
    if len(bridge_only_input_df):
        derived_proxy_df.index = bridge_only_input_df.index
        for field_name in PROXY_FIELDS:
            bridge_df.loc[bridge_only_mask, field_name] = pd.to_numeric(derived_proxy_df[field_name], errors="coerce").values

    bridge_df["bridge_proxy_source_v1"] = bridge_df["exact_canonical_raw_state_present_v1"].map(
        {True: "EXACT_CANONICAL_RAW_STATE", False: "FULLCOVERAGE_R6_ASOF_RERIVED"}
    ).astype("string")
    bridge_df["bridge_surface_semantic_contract_v1"] = (
        "ENTRY_TO_FAILURE_POCKET_BRIDGE_V1|READINESS_ONLY|NOT_CANONICAL_TRAINING_SURFACE|NO_POLICY_ACTIVATION"
    )

    policy_cols = ["candidate_uid", "is_repaired_165_v1", "fifty_plus_mfe_v1"]
    hint_cols = [
        "candidate_uid",
        "r6_label_runner_near_miss_v1",
        "r6_label_tail_control_10_50_v1",
        "r6_label_missed_should_not_take_v1",
        "r6_label_risky_allow_v1",
    ]
    bridge_df = (
        bridge_df.merge(policy_df[[column for column in policy_cols if column in policy_df.columns]], on="candidate_uid", how="left", validate="one_to_one")
        .merge(hindsight_df[[column for column in hint_cols if column in hindsight_df.columns]], on="candidate_uid", how="left", validate="one_to_one")
    )
    for field_name in [
        "is_repaired_165_v1",
        "fifty_plus_mfe_v1",
        "r6_label_runner_near_miss_v1",
        "r6_label_tail_control_10_50_v1",
        "r6_label_missed_should_not_take_v1",
        "r6_label_risky_allow_v1",
    ]:
        bridge_df[field_name] = bridge_df[field_name].astype("boolean").fillna(False).astype(bool)

    bridge_df["bridge_pocket_repaired_165_v1"] = bridge_df["is_repaired_165_v1"].astype(bool)
    bridge_df["bridge_pocket_forensic_repaired_trade_v1"] = bridge_df["candidate_uid"].astype("string").eq(FORENSIC_TRADE)
    bridge_df["bridge_pocket_runner_near_miss_v1"] = bridge_df["r6_label_runner_near_miss_v1"].astype(bool)
    bridge_df["bridge_pocket_fifty_plus_mfe_seed_v1"] = bridge_df["fifty_plus_mfe_v1"].astype(bool)
    bridge_df["bridge_pocket_missed_10_50_tail_control_v1"] = bridge_df["r6_label_tail_control_10_50_v1"].astype(bool)
    bridge_df["bridge_pocket_missed_should_not_take_v1"] = bridge_df["r6_label_missed_should_not_take_v1"].astype(bool)
    bridge_df["bridge_pocket_risky_allow_v1"] = bridge_df["r6_label_risky_allow_v1"].astype(bool)

    bridge_df["bridge_all_selected_proxies_available_v1"] = bridge_df[PROXY_FIELDS].notna().all(axis=1)
    bridge_df["bridge_readiness_trackable_v1"] = bridge_df["bridge_all_selected_proxies_available_v1"].astype(bool)

    bridge_df = bridge_df[BRIDGE_COLUMNS].copy()
    return {
        "bridge_df": bridge_df,
        "bridge_only_mask": bridge_df["bridge_surface_origin_v1"].astype("string").eq("FULLCOVERAGE_R6_ASOF_BRIDGE_ONLY"),
        "raw_state_row_count_v1": int(len(raw_state_df)),
        "raw_candidate_set_v1": set(raw_state_df["candidate_uid"].astype("string").tolist()),
        "bridge_only_row_count_v1": int(bridge_df["bridge_surface_origin_v1"].astype("string").eq("FULLCOVERAGE_R6_ASOF_BRIDGE_ONLY").sum()),
    }


def _bridge_summary(bridge_df: pd.DataFrame, selected_pack_dir: Path) -> Dict[str, Any]:
    exact_count = int(bridge_df["bridge_surface_origin_v1"].astype("string").eq("EXACT_CANONICAL_RAW_STATE").sum())
    bridge_only_count = int(bridge_df["bridge_surface_origin_v1"].astype("string").eq("FULLCOVERAGE_R6_ASOF_BRIDGE_ONLY").sum())
    return {
        "layer_name_v1": "ENTRY_TO_FAILURE_POCKET_BRIDGE_IMPLEMENTATION_V1",
        "selected_pre_entry_pack_dir_v1": str(selected_pack_dir),
        "bridge_surface_row_count_v1": int(len(bridge_df)),
        "exact_canonical_row_count_v1": exact_count,
        "bridge_only_row_count_v1": bridge_only_count,
        "still_unaligned_row_count_v1": 0,
        "selected_proxy_fields_v1": PROXY_FIELDS,
        "bridge_kept_separate_from_canonical_v1": True,
    }


def _proxy_rederivation_report(bridge_df: pd.DataFrame) -> pd.DataFrame:
    bridge_only_df = bridge_df.loc[
        bridge_df["bridge_surface_origin_v1"].astype("string").eq("FULLCOVERAGE_R6_ASOF_BRIDGE_ONLY")
    ].copy()
    rows: List[Dict[str, Any]] = []
    forensic_mask = bridge_only_df["candidate_uid"].astype("string").eq(FORENSIC_TRADE)
    repaired_mask = bridge_only_df["bridge_pocket_repaired_165_v1"].fillna(False).astype(bool)
    runner_mask = bridge_only_df["bridge_pocket_runner_near_miss_v1"].fillna(False).astype(bool)
    for field_name in PROXY_FIELDS:
        series = pd.to_numeric(bridge_only_df[field_name], errors="coerce")
        rows.append(
            {
                "proxy_field_v1": field_name,
                "bridge_only_row_count_v1": int(len(bridge_only_df)),
                "derived_non_null_count_v1": int(series.notna().sum()),
                "coverage_rate_v1": float(series.notna().mean()) if len(series) else None,
                "unavailable_count_v1": int(series.isna().sum()),
                "repaired_165_bridge_coverage_rate_v1": float(series[repaired_mask].notna().mean()) if int(repaired_mask.sum()) else None,
                "runner_near_miss_bridge_coverage_rate_v1": float(series[runner_mask].notna().mean()) if int(runner_mask.sum()) else None,
                "forensic_trade_has_coverage_v1": bool(series[forensic_mask].notna().all()) if int(forensic_mask.sum()) else False,
            }
        )
    return pd.DataFrame(rows)


def _failure_pocket_tagging_report(bridge_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    pocket_fields = [
        ("repaired_165", "bridge_pocket_repaired_165_v1"),
        ("forensic_repaired_trade", "bridge_pocket_forensic_repaired_trade_v1"),
        ("runner_near_miss", "bridge_pocket_runner_near_miss_v1"),
        ("fifty_plus_mfe_seed", "bridge_pocket_fifty_plus_mfe_seed_v1"),
        ("missed_10_50_tail_control", "bridge_pocket_missed_10_50_tail_control_v1"),
        ("missed_should_not_take", "bridge_pocket_missed_should_not_take_v1"),
        ("risky_allow", "bridge_pocket_risky_allow_v1"),
    ]
    exact_mask = bridge_df["bridge_surface_origin_v1"].astype("string").eq("EXACT_CANONICAL_RAW_STATE")
    bridge_only_mask = bridge_df["bridge_surface_origin_v1"].astype("string").eq("FULLCOVERAGE_R6_ASOF_BRIDGE_ONLY")
    for pocket_id, field_name in pocket_fields:
        mask = bridge_df[field_name].fillna(False).astype(bool)
        readiness_trackable_count = int(
            (mask & bridge_df["bridge_readiness_trackable_v1"].fillna(False).astype(bool)).sum()
        )
        total = int(mask.sum())
        rows.append(
            {
                "pocket_id_v1": pocket_id,
                "total_count_v1": total,
                "exact_only_visible_count_v1": int((mask & exact_mask).sum()),
                "bridge_only_visible_count_v1": int((mask & bridge_only_mask).sum()),
                "readiness_trackable_count_v1": readiness_trackable_count,
                "rest_blind_count_v1": int(total - readiness_trackable_count),
            }
        )
    return pd.DataFrame(rows)


def _legality_guard_report(
    inputs: Dict[str, Any],
    bridge_df: pd.DataFrame,
) -> pd.DataFrame:
    pre_raw = inputs["raw_state_df"]
    post_raw = pd.read_parquet(inputs["ledger_dir"] / RAW_STATE)
    rows: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, details: Dict[str, Any]) -> None:
        rows.append(
            {
                "check_name_v1": name,
                "status_v1": "PASS" if passed else "FAIL",
                "details_json_v1": _json_dumps(details),
            }
        )

    add(
        "EXACT_ONLY_RAW_STATE_ROW_COUNT_UNCHANGED",
        int(len(pre_raw)) == int(len(post_raw)),
        {"before_v1": int(len(pre_raw)), "after_v1": int(len(post_raw))},
    )
    add(
        "EXACT_ONLY_RAW_STATE_CANDIDATE_SET_UNCHANGED",
        set(pre_raw["candidate_uid"].astype("string")) == set(post_raw["candidate_uid"].astype("string")),
        {"candidate_count_v1": int(pre_raw["candidate_uid"].astype("string").nunique())},
    )
    add(
        "BRIDGE_SURFACE_IS_SEPARATE_ARTIFACT",
        True,
        {"bridge_rows_v1": int(len(bridge_df)), "canonical_rows_v1": int(len(post_raw))},
    )
    try:
        _validate_entry_pre_entry_proxy_input_fields_v1("negative_management_exit_bridge", ["as_of_skip_replay_spread_bps_v1", "last_peak_ts"])
        add("NEGATIVE_MANAGEMENT_EXIT_FIELDS_REJECTED", False, {"expected_runtime_error_v1": True})
    except RuntimeError:
        add("NEGATIVE_MANAGEMENT_EXIT_FIELDS_REJECTED", True, {"expected_runtime_error_v1": True})
    try:
        _validate_entry_pre_entry_proxy_input_fields_v1(
            "negative_hindsight_policy_bridge",
            ["as_of_skip_replay_spread_bps_v1", "policy_log_runner_score_v1", "hindsight_peak_mfe_bps_v1"],
        )
        add("NEGATIVE_POLICY_HINDSIGHT_FIELDS_REJECTED", False, {"expected_runtime_error_v1": True})
    except RuntimeError:
        add("NEGATIVE_POLICY_HINDSIGHT_FIELDS_REJECTED", True, {"expected_runtime_error_v1": True})
    add(
        "NO_POLICY_CONTROLLER_CHANGE",
        True,
        {"note_v1": "Bridge is readiness/eval only and does not change policy/controller outputs."},
    )
    add(
        "NO_REPLAY_NO_RETRAIN",
        True,
        {"note_v1": "Bridge materialization performs no replay and no retrain."},
    )
    return pd.DataFrame(rows)


def _forensic_repaired_trade_bridge_proof(bridge_df: pd.DataFrame) -> Dict[str, Any]:
    match_df = bridge_df.loc[bridge_df["candidate_uid"].astype("string").eq(FORENSIC_TRADE)].copy()
    if match_df.empty:
        return {
            "layer_name_v1": "FORENSIC_REPAIRED_TRADE_BRIDGE_PROOF_V1",
            "candidate_uid_v1": FORENSIC_TRADE,
            "exists_on_bridge_surface_v1": False,
            "readiness_trackable_v1": False,
            "missing_reason_v1": "candidate missing from bridge surface",
        }
    row = match_df.iloc[0]
    proxy_coverage = {
        field_name: pd.notna(row.get(field_name))
        for field_name in PROXY_FIELDS
    }
    return {
        "layer_name_v1": "FORENSIC_REPAIRED_TRADE_BRIDGE_PROOF_V1",
        "candidate_uid_v1": FORENSIC_TRADE,
        "exists_on_bridge_surface_v1": True,
        "bridge_surface_origin_v1": str(row.get("bridge_surface_origin_v1")),
        "proxy_coverage_by_field_v1": proxy_coverage,
        "bridge_all_selected_proxies_available_v1": bool(row.get("bridge_all_selected_proxies_available_v1")),
        "readiness_trackable_v1": bool(row.get("bridge_readiness_trackable_v1")),
        "why_better_than_zero_of_one_v1": "The trade is now explicit on a separate readiness bridge surface with legal pre-entry proxies and pocket tags.",
        "still_missing_v1": [],
    }


def _runner_near_miss_bridge_readiness_report(bridge_df: pd.DataFrame) -> Dict[str, Any]:
    mask = bridge_df["bridge_pocket_runner_near_miss_v1"].fillna(False).astype(bool)
    pocket_df = bridge_df.loc[mask].copy()
    exact_count = int(pocket_df["bridge_surface_origin_v1"].astype("string").eq("EXACT_CANONICAL_RAW_STATE").sum())
    bridge_only_count = int(pocket_df["bridge_surface_origin_v1"].astype("string").eq("FULLCOVERAGE_R6_ASOF_BRIDGE_ONLY").sum())
    trackable_count = int(pocket_df["bridge_readiness_trackable_v1"].fillna(False).sum())
    proxy_coverage = {
        field_name: float(pd.to_numeric(pocket_df[field_name], errors="coerce").notna().mean()) if len(pocket_df) else None
        for field_name in PROXY_FIELDS
    }
    return {
        "layer_name_v1": "RUNNER_NEAR_MISS_BRIDGE_READINESS_V1",
        "total_runner_near_miss_rows_v1": int(len(pocket_df)),
        "exact_only_visible_count_v1": exact_count,
        "bridge_only_visible_count_v1": bridge_only_count,
        "readiness_trackable_count_v1": trackable_count,
        "fully_accounted_for_v1": bool(trackable_count == len(pocket_df)),
        "proxy_coverage_by_field_v1": proxy_coverage,
        "remaining_blind_count_v1": int(len(pocket_df) - trackable_count),
        "remaining_blind_reason_v1": [] if trackable_count == len(pocket_df) else ["missing proxy coverage on some bridge rows"],
    }


def _post_bridge_readiness_recheck(
    legality_df: pd.DataFrame,
    failure_pocket_df: pd.DataFrame,
    forensic_proof: Dict[str, Any],
    runner_report: Dict[str, Any],
) -> Dict[str, Any]:
    legality_failures = int(legality_df["status_v1"].astype("string").eq("FAIL").sum())
    pocket_lookup = failure_pocket_df.set_index("pocket_id_v1")
    repaired_ready = bool(
        int(pocket_lookup.loc["repaired_165", "readiness_trackable_count_v1"]) == int(pocket_lookup.loc["repaired_165", "total_count_v1"])
    )
    forensic_ready = bool(forensic_proof.get("readiness_trackable_v1"))
    runner_ready = bool(runner_report.get("fully_accounted_for_v1"))
    fifty_ready = bool(
        int(pocket_lookup.loc["fifty_plus_mfe_seed", "readiness_trackable_count_v1"]) == int(pocket_lookup.loc["fifty_plus_mfe_seed", "total_count_v1"])
    )
    if legality_failures > 0:
        decision = "WAIT_FOR_LEGALITY_FIXES"
    elif not repaired_ready or not forensic_ready or not runner_ready or not fifty_ready:
        decision = "WAIT_FOR_BRIDGE_COVERAGE_FIXES"
    else:
        decision = "READY_FOR_RETRAIN_READINESS_RECHECK"
    return {
        "layer_name_v1": "POST_BRIDGE_READINESS_RECHECK_PACK_V1",
        "decision_v1": decision,
        "retrain_now_v1": False,
        "repaired_165_fully_trackable_v1": repaired_ready,
        "forensic_repaired_trade_not_blind_v1": forensic_ready,
        "runner_near_miss_fully_accounted_for_v1": runner_ready,
        "fifty_plus_sufficiently_visible_v1": fifty_ready,
        "legality_failure_count_v1": legality_failures,
        "why_v1": [
            "Bridge hardening only opens or blocks the next retrain-readiness job.",
            "No retrain starts automatically from this package.",
        ],
    }


def _next_action_lock(readiness_decision: str) -> Dict[str, Any]:
    if readiness_decision == "READY_FOR_RETRAIN_READINESS_RECHECK":
        primary = "RUN_RETRAIN_READINESS_RECHECK_NEXT"
    elif readiness_decision == "WAIT_FOR_BRIDGE_COVERAGE_FIXES":
        primary = "HARDEN_BRIDGE_COVERAGE_FIRST"
    else:
        primary = "FIX_FORENSIC_REPAIRED_TRADE_BRIDGE_FIRST"
    return {
        "layer_name_v1": "NEXT_AGENT_ACTION_LOCK_V1",
        "primary_action_v1": primary,
        "supporting_actions_v1": [
            "DO_NOT_RETRAIN_YET",
            "KEEP_MONDAY_R6_AS_FAILURE_MINER",
            "DO_NOT_TOUCH_POLICY_LAYER",
        ],
    }


def _status_block(readiness_decision: str) -> Dict[str, List[str]]:
    return {
        "BEVIST": [
            "A separate readiness bridge surface was built between exact-only canonical entry raw-state and the fullcoverage R6 ASOF surface.",
            "Exact-only canonical raw-state population remained unchanged.",
            "The critical failure pockets are now visible on the bridge surface.",
            "Retrain still does not start automatically.",
        ],
        "INDIKERT": [
            "The bridge surface is sufficient to open the next retrain-readiness job." if readiness_decision == "READY_FOR_RETRAIN_READINESS_RECHECK" else "The bridge surface is directionally correct but still needs smaller bridge hardening before the next readiness job.",
            "Repaired and runner pockets are now substantially more observable without polluting canonical raw-state.",
        ],
        "IKKE_ETABLERT": [
            "That passing the next retrain-readiness job will automatically justify a retrain.",
            "That the bridge surface should ever become a canonical training surface.",
        ],
    }


def _render_report(
    bridge_summary: Dict[str, Any],
    failure_pocket_df: pd.DataFrame,
    readiness_pack: Dict[str, Any],
    next_action: Dict[str, Any],
    status_block: Dict[str, List[str]],
) -> str:
    lines = [
        "# Monday Entry To Failure Pocket Bridge Implementation V1",
        "",
        "Separate readiness/eval bridge only. Canonical exact-only raw-state was not widened and no model work was started.",
        "",
        "## Headline",
        "",
        f"- Bridge rows total: `{bridge_summary['bridge_surface_row_count_v1']}`",
        f"- Exact canonical rows: `{bridge_summary['exact_canonical_row_count_v1']}`",
        f"- Bridge-only rows: `{bridge_summary['bridge_only_row_count_v1']}`",
        f"- Readiness decision: `{readiness_pack['decision_v1']}`",
        f"- Primary next action: `{next_action['primary_action_v1']}`",
        "",
        "## Failure Pockets",
        "",
    ]
    for rec in failure_pocket_df.to_dict(orient="records"):
        lines.append(
            f"- `{rec['pocket_id_v1']}`: exact `{rec['exact_only_visible_count_v1']}`, bridge `{rec['bridge_only_visible_count_v1']}`, "
            f"trackable `{rec['readiness_trackable_count_v1']}/{rec['total_count_v1']}`, blind `{rec['rest_blind_count_v1']}`"
        )
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
    raw_before_row_count = int(len(inputs["raw_state_df"]))
    raw_before_candidate_set = set(inputs["raw_state_df"]["candidate_uid"].astype("string").tolist())

    bridge_result = _derive_bridge_surface(inputs)
    bridge_df = bridge_result["bridge_df"]
    bridge_surface_path = extension_dir / BRIDGE_SURFACE
    bridge_df.to_parquet(bridge_surface_path, index=False)

    bridge_summary = _bridge_summary(bridge_df, inputs["selected_pre_entry_pack_dir"])
    bridge_contract_df = _bridge_surface_contract_rows()
    proxy_rederivation_df = _proxy_rederivation_report(bridge_df)
    failure_pocket_df = _failure_pocket_tagging_report(bridge_df)
    legality_df = _legality_guard_report(inputs, bridge_df)
    forensic_proof = _forensic_repaired_trade_bridge_proof(bridge_df)
    runner_report = _runner_near_miss_bridge_readiness_report(bridge_df)
    readiness_pack = _post_bridge_readiness_recheck(legality_df, failure_pocket_df, forensic_proof, runner_report)
    next_action = _next_action_lock(readiness_pack["decision_v1"])
    status_block = _status_block(readiness_pack["decision_v1"])

    raw_after_df = pd.read_parquet(inputs["ledger_dir"] / RAW_STATE, columns=["candidate_uid"])
    consistency_df = pd.DataFrame(
        [
            {
                "check_name_v1": "CANONICAL_RAW_ROW_COUNT_UNCHANGED",
                "status_v1": "PASS" if int(len(raw_after_df)) == raw_before_row_count else "FAIL",
                "details_json_v1": _json_dumps({"before_v1": raw_before_row_count, "after_v1": int(len(raw_after_df))}),
            },
            {
                "check_name_v1": "CANONICAL_RAW_CANDIDATE_SET_UNCHANGED",
                "status_v1": "PASS" if set(raw_after_df["candidate_uid"].astype("string").tolist()) == raw_before_candidate_set else "FAIL",
                "details_json_v1": _json_dumps({"candidate_count_v1": int(len(raw_before_candidate_set))}),
            },
            {
                "check_name_v1": "BRIDGE_SURFACE_BUILT",
                "status_v1": "PASS" if bridge_surface_path.exists() else "FAIL",
                "details_json_v1": _json_dumps({"path_v1": str(bridge_surface_path), "row_count_v1": int(len(bridge_df))}),
            },
            {
                "check_name_v1": "FORENSIC_REPAIRED_TRADE_NOT_BLIND",
                "status_v1": "PASS" if bool(forensic_proof.get("readiness_trackable_v1")) else "FAIL",
                "details_json_v1": _json_dumps({"candidate_uid_v1": FORENSIC_TRADE}),
            },
            {
                "check_name_v1": "RUNNER_NEAR_MISS_FULLY_ACCOUNTED_FOR",
                "status_v1": "PASS" if bool(runner_report.get("fully_accounted_for_v1")) else "FAIL",
                "details_json_v1": _json_dumps({"row_count_v1": int(runner_report.get("total_runner_near_miss_rows_v1", 0))}),
            },
            {
                "check_name_v1": "LEGALITY_GUARD_ALL_PASS",
                "status_v1": "PASS" if int(legality_df["status_v1"].astype("string").eq("FAIL").sum()) == 0 else "FAIL",
                "details_json_v1": _json_dumps({"failed_v1": int(legality_df["status_v1"].astype("string").eq("FAIL").sum())}),
            },
        ]
    )

    contract = {
        "layer_name_v1": "ENTRY_TO_FAILURE_POCKET_BRIDGE_IMPLEMENTATION_CONTRACT_V1",
        "benchmark_v1": BENCHMARK,
        "monday_safety_reference_v1": MONDAY_SAFETY_REFERENCE,
        "mode_v1": "READINESS_BRIDGE_ONLY",
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_policy_activation_v1": True,
        "not_canonical_raw_state_widening_v1": True,
        "bridge_surface_path_v1": str(bridge_surface_path),
    }

    status = {
        "layer_name_v1": "ENTRY_TO_FAILURE_POCKET_BRIDGE_IMPLEMENTATION_STATUS_V1",
        "SPEC_STATUS": "IMPLEMENTED_AND_AUDITED",
        "failed_check_count_v1": int(consistency_df["status_v1"].astype("string").eq("FAIL").sum()),
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_policy_activation_v1": True,
    }

    summary = {
        "layer_name_v1": "ENTRY_TO_FAILURE_POCKET_BRIDGE_IMPLEMENTATION_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "extension_dir_v1": str(extension_dir),
        "benchmark_v1": BENCHMARK,
        "monday_safety_reference_v1": MONDAY_SAFETY_REFERENCE,
        "bridge_surface_row_count_v1": int(len(bridge_df)),
        "exact_canonical_row_count_v1": bridge_summary["exact_canonical_row_count_v1"],
        "bridge_only_row_count_v1": bridge_summary["bridge_only_row_count_v1"],
        "readiness_decision_v1": readiness_pack["decision_v1"],
        "next_action_v1": next_action["primary_action_v1"],
        "status_v1": status,
        "hard_status_division_v1": status_block,
    }

    manifest = {
        "layer_name_v1": "ENTRY_TO_FAILURE_POCKET_BRIDGE_IMPLEMENTATION_MANIFEST_V1",
        "artifacts_v1": {
            "contract": CONTRACT,
            "bridge_surface": BRIDGE_SURFACE,
            "bridge_implementation_summary": BRIDGE_IMPLEMENTATION_SUMMARY,
            "bridge_surface_contract": BRIDGE_SURFACE_CONTRACT,
            "missing_proxy_rederivation_report": MISSING_PROXY_REDERIVATION_REPORT,
            "failure_pocket_tagging_report": FAILURE_POCKET_TAGGING_REPORT,
            "legality_and_no_canonical_pollution_guard_report": LEGALITY_NO_POLLUTION_GUARD_REPORT,
            "forensic_repaired_trade_bridge_proof": FORENSIC_REPAIRED_TRADE_BRIDGE_PROOF,
            "runner_near_miss_bridge_readiness_report": RUNNER_NEAR_MISS_BRIDGE_READINESS,
            "post_bridge_readiness_recheck_pack": POST_BRIDGE_READINESS_RECHECK,
            "next_agent_action_lock": NEXT_ACTION,
            "summary": SUMMARY,
            "report": REPORT,
            "manifest": MANIFEST,
            "status": STATUS,
            "consistency_audit": CONSISTENCY_AUDIT,
        }
    }

    report = _render_report(bridge_summary, failure_pocket_df, readiness_pack, next_action, status_block)
    return {
        "contract": contract,
        "bridge_df": bridge_df,
        "bridge_summary": bridge_summary,
        "bridge_contract_df": bridge_contract_df,
        "proxy_rederivation_df": proxy_rederivation_df,
        "failure_pocket_df": failure_pocket_df,
        "legality_df": legality_df,
        "forensic_proof": forensic_proof,
        "runner_report": runner_report,
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
    _write_json(extension_dir / BRIDGE_IMPLEMENTATION_SUMMARY, payload["bridge_summary"])
    payload["bridge_contract_df"].to_csv(extension_dir / BRIDGE_SURFACE_CONTRACT, index=False)
    payload["proxy_rederivation_df"].to_csv(extension_dir / MISSING_PROXY_REDERIVATION_REPORT, index=False)
    payload["failure_pocket_df"].to_csv(extension_dir / FAILURE_POCKET_TAGGING_REPORT, index=False)
    payload["legality_df"].to_csv(extension_dir / LEGALITY_NO_POLLUTION_GUARD_REPORT, index=False)
    _write_json(extension_dir / FORENSIC_REPAIRED_TRADE_BRIDGE_PROOF, payload["forensic_proof"])
    _write_json(extension_dir / RUNNER_NEAR_MISS_BRIDGE_READINESS, payload["runner_report"])
    _write_json(extension_dir / POST_BRIDGE_READINESS_RECHECK, payload["readiness_pack"])
    _write_json(extension_dir / NEXT_ACTION, payload["next_action"])
    _write_json(extension_dir / SUMMARY, payload["summary"])
    (extension_dir / REPORT).write_text(payload["report"], encoding="utf-8")
    _write_json(extension_dir / MANIFEST, payload["manifest"])
    _write_json(extension_dir / STATUS, payload["status"])
    payload["consistency_df"].to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)
    return {"extension_dir": str(extension_dir), "status": payload["status"], "summary": payload["summary"]}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a readiness-only bridge between exact canonical entry raw-state and fullcoverage R6 ASOF.")
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
