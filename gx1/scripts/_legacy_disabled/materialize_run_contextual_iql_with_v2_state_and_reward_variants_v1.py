#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.scripts import materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate
from gx1.scripts import materialize_refine_clean_as_of_safety_layer_to_retain_safe_core_v1 as refine_gate
from gx1.scripts import materialize_run_iql_offline_sanity_training_research_only_v1 as sanity_gate


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "RUN_CONTEXTUAL_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1"

INPUT_CONTRACT_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "BUILD_IQL_OFFLINE_DATA_CONTRACT_RESEARCH_ONLY_V1_20260428T190901Z_LOCK"
)
INPUT_SANITY_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "RUN_IQL_OFFLINE_SANITY_TRAINING_RESEARCH_ONLY_V1_20260428T192801Z_LOCK"
)
INPUT_REBUILD_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "REBUILD_IQL_STATE_CONTRACT_WITH_MORE_AS_OF_FEATURES_V1_20260429T081445Z_LOCK"
)
INPUT_REFINE_CLEAN_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "REFINE_CLEAN_AS_OF_SAFETY_LAYER_TO_RETAIN_SAFE_CORE_V1_20260428T185018Z_LOCK"
)

EXPECTED_FRAME_ROWS = 1914
EXPECTED_HARDENED_ROWS = 89
EXPECTED_SHIELD_ROWS = 78
EXPECTED_REWARD_VARIANT_COUNT = 4

MODEL_STATE_COLUMNS = sanity_gate.MODEL_STATE_COLUMNS

REWARD_FAMILY_V1 = "SAFETY_WEIGHTED_REWARD_V1"
REWARD_FAMILIES_V2 = [
    "ENTRY_REALIZED_PNL_REWARD_V2",
    "ENTRY_MFE_CAPTURE_REWARD_V2",
    "ENTRY_MAE_BURDEN_REWARD_V2",
    "ENTRY_TRANSPARENT_COMBINED_REWARD_V2",
]
ALL_REWARD_FAMILIES = [REWARD_FAMILY_V1] + REWARD_FAMILIES_V2

REWARD_INPUT_FIELDS_FORBIDDEN_AS_STATE = {
    "mfe_bps",
    "mae_bps",
    "pnl_bps",
    "post_exit_mfe_bps",
    "early_exit_regret",
    "duration_bars",
    "exit_reason",
}

ALLOWED_FINAL_STATUSES = {
    "RUN_CONTEXTUAL_IQL_V2_PASS_REWARD_VARIANTS_LIFT_OBSERVED",
    "RUN_CONTEXTUAL_IQL_V2_PASS_REWARD_VARIANTS_NEUTRAL",
    "RUN_CONTEXTUAL_IQL_V2_PARTIAL_SOME_VARIANTS_COLLAPSE",
    "RUN_CONTEXTUAL_IQL_V2_BLOCKED_BY_SAFETY_VIOLATION",
    "RUN_CONTEXTUAL_IQL_V2_BLOCKED_BY_REWARD_LEAKAGE",
    "RUN_CONTEXTUAL_IQL_V2_BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "DEEPEN_IQL_STATE_FAMILY_DISCOVERY_V1",
    "RUN_IQL_REWARD_VARIANT_SENSITIVITY_V1",
    "DEEPEN_IQL_ACTION_SUPPORT_AND_BEHAVIOR_POLICY_AUDIT_V1",
    "HOLD_UNTIL_NEW_AS_OF_FAMILIES_LANDED_V1",
}

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")

SEED_V1 = 20260429


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    return contract_gate._jsonable(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    contract_gate._write_json(path, payload)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    contract_gate._write_rows(path, rows)


def _write_report(path: Path, lines: Sequence[str]) -> None:
    contract_gate._write_report(path, lines)


def _read_json(path: Path) -> dict[str, Any]:
    return contract_gate._read_json(path)


def _file_hash(path: Path) -> str:
    return contract_gate._file_hash(path)


def _python_manifest() -> dict[str, Any]:
    return contract_gate._python_manifest()


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    return refine_gate._bool(frame, column)


def validate_explicit_artifact_roots(paths: Iterable[Path]) -> bool:
    return contract_gate.validate_explicit_artifact_roots(paths)


def validate_no_forbidden_actions(**kwargs: Any) -> dict[str, Any]:
    return contract_gate.validate_no_forbidden_actions(**kwargs)


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def validate_no_deprecated_revival(script_path: Path) -> bool:
    text = script_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.lstrip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        for fragment in QUARANTINE_FORBIDDEN_PATH_FRAGMENTS:
            if fragment in stripped:
                raise RuntimeError("DEPRECATED_QUARANTINE_REVIVAL_FORBIDDEN")
    return True


def _load_inputs() -> dict[str, Any]:
    roots = [INPUT_CONTRACT_ROOT, INPUT_SANITY_ROOT, INPUT_REBUILD_ROOT, INPUT_REFINE_CLEAN_ROOT]
    validate_explicit_artifact_roots(roots)
    required = {
        "v1_state_contract": INPUT_CONTRACT_ROOT / "iql_offline_state_contract_v1.json",
        "v1_reward_contract": INPUT_CONTRACT_ROOT / "iql_offline_reward_contract_v1.json",
        "v1_safety_shield": INPUT_CONTRACT_ROOT / "iql_offline_safety_shield_contract_v1.json",
        "v1_sanity_summary": INPUT_SANITY_ROOT / "summary_v1.json",
        "rebuild_summary": INPUT_REBUILD_ROOT / "summary_v1.json",
        "rebuild_go_no_go": INPUT_REBUILD_ROOT
        / "rebuild_iql_state_contract_with_more_as_of_features_go_no_go_v1.json",
        "rebuild_reward_variants_contract": INPUT_REBUILD_ROOT
        / "REWARD_VARIANTS_V2"
        / "iql_entry_iql_reward_variants_contract_v2.json",
        "rebuild_reward_join_audit": INPUT_REBUILD_ROOT
        / "REWARD_VARIANTS_V2"
        / "entry_iql_post_trade_outcome_join_audit_v1.json",
        "rebuild_reward_join_table": INPUT_REBUILD_ROOT
        / "REWARD_VARIANTS_V2"
        / "entry_iql_post_trade_outcome_join_table_v1.csv",
        "rebuild_reward_class_audit": INPUT_REBUILD_ROOT
        / "REWARD_VARIANTS_V2"
        / "reward_variant_class_audit_v1.json",
        "rebuild_state_contract_v2": INPUT_REBUILD_ROOT
        / "STATE_EXPANSION_V2"
        / "iql_offline_state_contract_v2.json",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_INPUT_ARTIFACTS: {missing}")
    rebuild_join = _read_json(required["rebuild_reward_join_audit"])
    if rebuild_join.get("join_status_v1") != "REWARD_JOIN_LOCKED":
        raise RuntimeError("REBUILD_REWARD_JOIN_NOT_LOCKED")
    if rebuild_join.get("take_trade_match_rate_v1", 0.0) < 0.95:
        raise RuntimeError("REBUILD_REWARD_TAKE_MATCH_RATE_TOO_LOW")
    rebuild_class = _read_json(required["rebuild_reward_class_audit"])
    if rebuild_class.get("leakage_status_v1") != "PASS":
        raise RuntimeError("REBUILD_REWARD_CLASS_AUDIT_NOT_PASS")
    return {
        "required_paths": required,
        "v1_state_contract": _read_json(required["v1_state_contract"]),
        "v1_reward_contract": _read_json(required["v1_reward_contract"]),
        "v1_safety_shield": _read_json(required["v1_safety_shield"]),
        "v1_sanity_summary": _read_json(required["v1_sanity_summary"]),
        "rebuild_summary": _read_json(required["rebuild_summary"]),
        "rebuild_go_no_go": _read_json(required["rebuild_go_no_go"]),
        "rebuild_reward_variants_contract": _read_json(
            required["rebuild_reward_variants_contract"]
        ),
        "rebuild_reward_join_audit": rebuild_join,
        "rebuild_reward_class_audit": rebuild_class,
        "rebuild_state_contract_v2": _read_json(required["rebuild_state_contract_v2"]),
        "frame_inputs": refine_gate._load_inputs(),
    }


def _frame_and_masks(inputs: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    frame, masks = refine_gate._build_frame_and_masks(inputs["frame_inputs"])
    if frame.shape[0] != EXPECTED_FRAME_ROWS:
        raise RuntimeError("FRAME_ROW_COUNT_MISMATCH")
    if int(masks["hardened"].sum()) != EXPECTED_HARDENED_ROWS:
        raise RuntimeError("HARDENED_MASK_MISMATCH")
    shielded = masks["hardened"] & ~masks["source_confluence_repairable_v1"]
    if int(shielded.sum()) != EXPECTED_SHIELD_ROWS:
        raise RuntimeError("SHIELD_MASK_MISMATCH")
    return frame, masks


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in inputs["required_paths"].items()
    ]
    return {
        "layer_name": "RUN_CONTEXTUAL_IQL_V2_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "v1_data_contract_root_v1": str(INPUT_CONTRACT_ROOT),
            "v1_sanity_training_root_v1": str(INPUT_SANITY_ROOT),
            "rebuild_state_v2_root_v1": str(INPUT_REBUILD_ROOT),
            "refine_clean_root_v1": str(INPUT_REFINE_CLEAN_ROOT),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "research_only_contract_v1": True,
        "iql_training_run_v1": True,
        "iql_production_allowed_v1": False,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "deprecated_quarantine_revival_v1": False,
        "seed_v1": SEED_V1,
        "python_manifest_v1": _python_manifest(),
    }


# ---------------------------------------------------------------------------
# Reward arrays (5 variants)
# ---------------------------------------------------------------------------


def _load_join_table(rebuild_root: Path) -> pd.DataFrame:
    path = rebuild_root / "REWARD_VARIANTS_V2" / "entry_iql_post_trade_outcome_join_table_v1.csv"
    df = pd.read_csv(path, dtype={"candidate_uid_v1": str, "trade_uid_v1": str})
    if "candidate_uid_v1" not in df.columns:
        raise RuntimeError("JOIN_TABLE_MISSING_CANDIDATE_UID_V1")
    if "pnl_bps" not in df.columns:
        raise RuntimeError("JOIN_TABLE_MISSING_PNL_BPS")
    return df


def _align_join_to_frame(
    frame: pd.DataFrame, join_table: pd.DataFrame
) -> pd.DataFrame:
    frame_uids = frame["candidate_uid_v1"].astype(str).reset_index(drop=True)
    join_indexed = join_table.set_index("candidate_uid_v1")
    aligned = join_indexed.reindex(frame_uids).reset_index()
    aligned["candidate_uid_v1"] = frame_uids
    if aligned["pnl_bps"].isna().sum() > 0:
        raise RuntimeError(
            "REWARD_JOIN_ALIGNMENT_FAILED: not all frame rows matched join table"
        )
    return aligned


def _compute_reward_arrays(
    frame: pd.DataFrame,
    masks: dict[str, pd.Series],
    join_aligned: pd.DataFrame,
) -> dict[str, np.ndarray]:
    shield = (masks["hardened"] & ~masks["source_confluence_repairable_v1"]).to_numpy()
    pnl = join_aligned["pnl_bps"].astype(float).to_numpy()
    mfe = join_aligned["mfe_bps"].astype(float).to_numpy()
    mae = join_aligned["mae_bps"].astype(float).to_numpy()
    eps = 1e-6

    rewards: dict[str, np.ndarray] = {}
    rewards[REWARD_FAMILY_V1] = sanity_gate._reward(
        frame,
        masks["hardened"] & ~masks["source_confluence_repairable_v1"],
    )
    pnl_reward = np.where(shield, pnl, 0.0)
    rewards["ENTRY_REALIZED_PNL_REWARD_V2"] = pnl_reward.astype(float)
    mfe_capture = np.where(shield, np.clip(pnl / np.maximum(mfe, eps), -2.0, 2.0), 0.0)
    rewards["ENTRY_MFE_CAPTURE_REWARD_V2"] = mfe_capture.astype(float)
    mae_burden = np.where(shield, pnl - 0.5 * np.abs(mae), 0.0)
    rewards["ENTRY_MAE_BURDEN_REWARD_V2"] = mae_burden.astype(float)
    giveback = np.maximum(mfe - pnl, 0.0)
    combined = np.where(shield, pnl - 0.25 * np.abs(mae) - 0.25 * giveback, 0.0)
    rewards["ENTRY_TRANSPARENT_COMBINED_REWARD_V2"] = combined.astype(float)
    return rewards


# ---------------------------------------------------------------------------
# Per-variant training and evaluation
# ---------------------------------------------------------------------------


def _compute_quality_metrics(
    policy_mask: np.ndarray,
    join_aligned: pd.DataFrame,
) -> dict[str, float]:
    if int(policy_mask.sum()) == 0:
        return {
            "mean_pnl_bps_v1": 0.0,
            "mean_mfe_bps_v1": 0.0,
            "mean_mae_bps_v1": 0.0,
            "mean_mfe_capture_v1": 0.0,
            "mean_mae_burden_v1": 0.0,
            "mean_giveback_bps_v1": 0.0,
            "mae_dominated_count_v1": 0,
            "peak_giveback_count_v1": 0,
            "cata_exit_count_v1": 0,
        }
    pnl = join_aligned["pnl_bps"].astype(float).to_numpy()[policy_mask]
    mfe = join_aligned["mfe_bps"].astype(float).to_numpy()[policy_mask]
    mae = join_aligned["mae_bps"].astype(float).to_numpy()[policy_mask]
    eps = 1e-6
    mfe_capture = np.clip(pnl / np.maximum(mfe, eps), -2.0, 2.0)
    mae_burden = pnl - 0.5 * np.abs(mae)
    giveback = np.maximum(mfe - pnl, 0.0)
    mae_dominated = (np.abs(mae) > mfe) & (pnl < 0.0)
    peak_giveback = (mfe > eps) & ((mfe - pnl) > 0.5 * mfe)
    if "exit_reason" in join_aligned.columns:
        exit_reasons = (
            join_aligned["exit_reason"].astype(str).to_numpy()[policy_mask]
        )
        cata = np.array([reason == "CATASTROPHIC_GUARD" for reason in exit_reasons])
    else:
        cata = np.zeros(int(policy_mask.sum()), dtype=bool)
    return {
        "mean_pnl_bps_v1": float(pnl.mean()),
        "mean_mfe_bps_v1": float(mfe.mean()),
        "mean_mae_bps_v1": float(mae.mean()),
        "mean_mfe_capture_v1": float(mfe_capture.mean()),
        "mean_mae_burden_v1": float(mae_burden.mean()),
        "mean_giveback_bps_v1": float(giveback.mean()),
        "mae_dominated_count_v1": int(mae_dominated.sum()),
        "peak_giveback_count_v1": int(peak_giveback.sum()),
        "cata_exit_count_v1": int(cata.sum()),
    }


def _train_and_evaluate(
    frame: pd.DataFrame,
    masks: dict[str, pd.Series],
    state: pd.DataFrame,
    split: pd.Series,
    rewards: dict[str, np.ndarray],
    join_aligned: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    shield = masks["hardened"] & ~masks["source_confluence_repairable_v1"]
    policy_masks: dict[str, np.ndarray] = {}
    configs: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    per_split_rows: list[dict[str, Any]] = []
    for reward_id in ALL_REWARD_FAMILIES:
        reward_array = rewards[reward_id]
        policy_take, config, train_metrics = sanity_gate._train_contextual_iql(
            state, split, shield, reward_array
        )
        policy_masks[reward_id] = policy_take
        configs[reward_id] = config
        policy_metrics = sanity_gate._policy_metrics(
            frame,
            masks,
            policy_take,
            reward_array,
            policy_name=f"IQL_CONTEXTUAL_ONE_STEP_POLICY_{reward_id}",
        )
        quality = _compute_quality_metrics(policy_take, join_aligned)
        rows.append(
            {
                "reward_id_v1": reward_id,
                "policy_name_v1": policy_metrics["policy_name_v1"],
                **{k: v for k, v in policy_metrics.items() if k != "policy_name_v1"},
                **quality,
                "research_only_v1": True,
            }
        )
        for tm in train_metrics:
            per_split_rows.append(
                {
                    "reward_id_v1": reward_id,
                    **tm,
                }
            )
    return rows, per_split_rows, policy_masks, configs


def _baseline_policies(
    frame: pd.DataFrame,
    masks: dict[str, pd.Series],
    join_aligned: pd.DataFrame,
) -> list[dict[str, Any]]:
    shield = (masks["hardened"] & ~masks["source_confluence_repairable_v1"]).to_numpy()
    safe_core = masks["hardened"].to_numpy()
    baseline_140 = _bool(frame, "is_140_94_baseline_v1").to_numpy()
    zero = np.zeros(len(frame), dtype=bool)
    safety_reward = sanity_gate._reward(
        frame, masks["hardened"] & ~masks["source_confluence_repairable_v1"]
    )
    rows: list[dict[str, Any]] = []
    for name, mask in [
        ("ALWAYS_SKIP", zero),
        ("ALWAYS_TAKE_WITHIN_78_SHIELD", shield),
        ("SAFE_CORE_RULE_POLICY_89", safe_core),
        ("BASELINE_140_94_COMPARATOR", baseline_140),
    ]:
        policy_metrics = sanity_gate._policy_metrics(
            frame, masks, mask, safety_reward, policy_name=name
        )
        quality = _compute_quality_metrics(mask, join_aligned)
        rows.append(
            {
                "reward_id_v1": "BASELINE_REFERENCE",
                "policy_name_v1": name,
                **{k: v for k, v in policy_metrics.items() if k != "policy_name_v1"},
                **quality,
                "research_only_v1": True,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


def _no_shortcut_audit_for_run() -> dict[str, Any]:
    return sanity_gate._no_shortcut_audit(
        MODEL_STATE_COLUMNS,
        {"heldout_used_for_fit_v1": False},
    )


def _reward_class_audit() -> dict[str, Any]:
    state_set = {col.lower() for col in MODEL_STATE_COLUMNS}
    leak = sorted(
        f for f in REWARD_INPUT_FIELDS_FORBIDDEN_AS_STATE if f.lower() in state_set
    )
    payload = {
        "layer_name": "RUN_CONTEXTUAL_IQL_V2_REWARD_CLASS_AUDIT_V1",
        "leakage_status_v1": "PASS" if not leak else "FAIL",
        "leaked_input_fields_v1": leak,
        "state_columns_v1": list(MODEL_STATE_COLUMNS),
        "research_only_v1": True,
    }
    if leak:
        raise RuntimeError(f"REWARD_INPUT_LEAK_INTO_STATE: {leak}")
    return payload


def _reproducibility_audit(
    frame: pd.DataFrame,
    masks: dict[str, pd.Series],
    rewards: dict[str, np.ndarray],
    policy_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    shield = masks["hardened"] & ~masks["source_confluence_repairable_v1"]
    audit = {
        "layer_name": "RUN_CONTEXTUAL_IQL_V2_REPRODUCIBILITY_AUDIT_V1",
        "frame_row_count_v1": int(len(frame)),
        "expected_frame_rows_v1": EXPECTED_FRAME_ROWS,
        "row_count_invariant_v1": int(len(frame)) == EXPECTED_FRAME_ROWS,
        "hardened_count_v1": int(masks["hardened"].sum()),
        "shielded_count_v1": int(shield.sum()),
        "seventy_eight_shield_invariant_v1": int(shield.sum()) == EXPECTED_SHIELD_ROWS,
        "reward_family_count_v1": len(ALL_REWARD_FAMILIES),
        "reward_v2_variant_count_v1": EXPECTED_REWARD_VARIANT_COUNT,
        "policy_count_v1": len(policy_rows),
        "research_only_v1": True,
        "deterministic_seed_v1": SEED_V1,
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
    }
    if not audit["row_count_invariant_v1"]:
        raise RuntimeError("ROW_COUNT_INVARIANT_FAILED")
    if not audit["seventy_eight_shield_invariant_v1"]:
        raise RuntimeError("SHIELD_INVARIANT_FAILED")
    return audit


def _go_no_go(
    policy_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> tuple[str, str, str]:
    iql_v1 = next(r for r in policy_rows if r["reward_id_v1"] == REWARD_FAMILY_V1)
    iql_v2_variants = [
        r for r in policy_rows if r["reward_id_v1"] in REWARD_FAMILIES_V2
    ]
    always_take = next(
        r for r in baseline_rows if r["policy_name_v1"] == "ALWAYS_TAKE_WITHIN_78_SHIELD"
    )
    always_skip = next(
        r for r in baseline_rows if r["policy_name_v1"] == "ALWAYS_SKIP"
    )
    safety_violations = sum(
        1 for r in policy_rows if r["safety_violations_v1"] > 0
    )
    if safety_violations > 0:
        return (
            "RUN_CONTEXTUAL_IQL_V2_BLOCKED_BY_SAFETY_VIOLATION",
            "HOLD_UNTIL_NEW_AS_OF_FAMILIES_LANDED_V1",
            "At least one V2 policy violated the safety shield invariant; hold further training until safety lineage is reaffirmed.",
        )
    collapsed = [
        r
        for r in policy_rows
        if r["selected_rows_v1"] == always_take["selected_rows_v1"]
        or r["selected_rows_v1"] == always_skip["selected_rows_v1"]
    ]
    if len(collapsed) >= len(policy_rows):
        return (
            "RUN_CONTEXTUAL_IQL_V2_PARTIAL_SOME_VARIANTS_COLLAPSE",
            "DEEPEN_IQL_STATE_FAMILY_DISCOVERY_V1",
            "All trained policies collapsed to ALWAYS_TAKE or ALWAYS_SKIP, indicating insufficient state discrimination. Deepen state-family discovery.",
        )
    v1_reward_take_count = iql_v1["selected_rows_v1"]
    v2_diff = [
        abs(r["selected_rows_v1"] - v1_reward_take_count) for r in iql_v2_variants
    ]
    has_lift = any(d > 0 for d in v2_diff)
    if has_lift:
        return (
            "RUN_CONTEXTUAL_IQL_V2_PASS_REWARD_VARIANTS_LIFT_OBSERVED",
            "RUN_IQL_REWARD_VARIANT_SENSITIVITY_V1",
            "At least one V2 reward variant produced a different policy from V1 SAFETY_WEIGHTED_REWARD; explore reward-variant sensitivity to localize MAE/MFE-aware behavior.",
        )
    return (
        "RUN_CONTEXTUAL_IQL_V2_PASS_REWARD_VARIANTS_NEUTRAL",
        "DEEPEN_IQL_ACTION_SUPPORT_AND_BEHAVIOR_POLICY_AUDIT_V1",
        "All V2 reward variants produced policies identical in row-count to V1 SAFETY_WEIGHTED_REWARD on the 78-shielded TAKE cohort. Reward shaping does not separate policies under current state. Next research lever is action-support and behavior-policy audit.",
    )


# ---------------------------------------------------------------------------
# Materialize entrypoint
# ---------------------------------------------------------------------------


def write_artifacts(
    out_root: Path | None = None,
    *,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    inputs = _load_inputs()
    frame, masks = _frame_and_masks(inputs)
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (
        DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)

    validate_no_deprecated_revival(Path(__file__))
    forbidden_audit = validate_no_forbidden_actions(
        adapter=False,
        r6=False,
        iql_production=False,
        package=False,
        freeze=False,
        promo=False,
        live=False,
        optuna=False,
        broad_sweep=False,
    )

    manifest = _input_manifest(inputs, artifact_root)
    _write_json(artifact_root / "input_manifest_v1.json", manifest)

    join_table = _load_join_table(INPUT_REBUILD_ROOT)
    join_aligned = _align_join_to_frame(frame, join_table)
    join_aligned.to_csv(artifact_root / "reward_join_aligned_v1.csv", index=False)

    split = sanity_gate._split_series(frame)
    state, normalization, state_audit = sanity_gate._normalization_and_state(frame, split)
    _write_json(artifact_root / "normalization_audit_v1.json", normalization)

    rewards = _compute_reward_arrays(frame, masks, join_aligned)
    reward_summary_rows: list[dict[str, Any]] = []
    for reward_id, arr in rewards.items():
        reward_summary_rows.append(
            {
                "reward_id_v1": reward_id,
                "non_zero_count_v1": int((arr != 0.0).sum()),
                "mean_v1": float(arr.mean()),
                "std_v1": float(arr.std(ddof=0)),
                "min_v1": float(arr.min()),
                "max_v1": float(arr.max()),
            }
        )
    _write_rows(artifact_root / "reward_arrays_summary_v1.csv", reward_summary_rows)

    reward_class_audit = _reward_class_audit()
    _write_json(artifact_root / "reward_class_audit_v1.json", reward_class_audit)

    no_shortcut = _no_shortcut_audit_for_run()
    _write_json(artifact_root / "no_shortcut_audit_v1.json", no_shortcut)

    policy_rows, per_split_rows, policy_masks, configs = _train_and_evaluate(
        frame, masks, state, split, rewards, join_aligned
    )
    baseline_rows = _baseline_policies(frame, masks, join_aligned)

    _write_rows(
        artifact_root / "iql_policy_per_reward_comparator_v1.csv",
        policy_rows,
    )
    _write_json(
        artifact_root / "iql_policy_per_reward_comparator_v1.json",
        {"row_count_v1": len(policy_rows), "rows_v1": policy_rows},
    )
    _write_rows(
        artifact_root / "iql_baseline_policy_comparator_v1.csv",
        baseline_rows,
    )
    _write_json(
        artifact_root / "iql_baseline_policy_comparator_v1.json",
        {"row_count_v1": len(baseline_rows), "rows_v1": baseline_rows},
    )
    _write_rows(
        artifact_root / "iql_per_reward_per_split_metrics_v1.csv",
        per_split_rows,
    )

    config_table = []
    for reward_id, config in configs.items():
        config_table.append(
            {
                "reward_id_v1": reward_id,
                "model_id_v1": config["model_id_v1"],
                "trained_on_rows_v1": config["trained_on_rows_v1"],
                "ridge_lambda_v1": config["ridge_lambda_v1"],
                "seed_v1": config["seed_v1"],
                "discount_v1": config["discount_v1"],
                "expectile_v1": config["expectile_v1"],
            }
        )
    _write_rows(artifact_root / "iql_per_reward_training_configs_v1.csv", config_table)
    _write_json(
        artifact_root / "iql_per_reward_training_configs_v1.json",
        {"row_count_v1": len(config_table), "rows_v1": config_table},
    )

    reproducibility = _reproducibility_audit(frame, masks, rewards, policy_rows)
    _write_json(artifact_root / "reproducibility_audit_v1.json", reproducibility)

    status, next_action, recommendation = _go_no_go(policy_rows, baseline_rows)
    validate_final_status(status, next_action)

    iql_v1 = next(r for r in policy_rows if r["reward_id_v1"] == REWARD_FAMILY_V1)

    summary = {
        "layer_name": "RUN_CONTEXTUAL_IQL_V2_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "reward_family_count_v1": len(ALL_REWARD_FAMILIES),
        "reward_v2_variant_count_v1": EXPECTED_REWARD_VARIANT_COUNT,
        "policy_count_v1": len(policy_rows),
        "iql_v1_baseline_selected_rows_v1": iql_v1["selected_rows_v1"],
        "iql_v1_baseline_precision_v1": iql_v1["precision_audit_only_v1"],
        "iql_v1_baseline_safety_status_v1": iql_v1["safety_status_v1"],
        "policy_summary_per_reward_v1": [
            {
                "reward_id_v1": row["reward_id_v1"],
                "selected_rows_v1": row["selected_rows_v1"],
                "precision_audit_only_v1": row["precision_audit_only_v1"],
                "safety_status_v1": row["safety_status_v1"],
                "mean_pnl_bps_v1": row["mean_pnl_bps_v1"],
                "mean_mfe_capture_v1": row["mean_mfe_capture_v1"],
                "mean_mae_burden_v1": row["mean_mae_burden_v1"],
                "mean_giveback_bps_v1": row["mean_giveback_bps_v1"],
            }
            for row in policy_rows
        ],
        "baseline_summary_v1": [
            {
                "policy_name_v1": row["policy_name_v1"],
                "selected_rows_v1": row["selected_rows_v1"],
                "precision_audit_only_v1": row["precision_audit_only_v1"],
                "mean_pnl_bps_v1": row["mean_pnl_bps_v1"],
                "mean_mfe_capture_v1": row["mean_mfe_capture_v1"],
                "mean_mae_burden_v1": row["mean_mae_burden_v1"],
            }
            for row in baseline_rows
        ],
        "row_count_invariant_v1": reproducibility["row_count_invariant_v1"],
        "seventy_eight_shield_invariant_v1": reproducibility[
            "seventy_eight_shield_invariant_v1"
        ],
        "no_shortcut_status_v1": no_shortcut["status_v1"],
        "reward_class_audit_status_v1": reward_class_audit["leakage_status_v1"],
        "research_only_v1": True,
        "iql_training_run_v1": True,
        "iql_production_allowed_v1": False,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "freeze_promo_live_run_v1": False,
        "deprecated_quarantine_revival_v1": False,
        "forbidden_actions_audit_v1": forbidden_audit,
        "next_gate_hook_v1": next_action,
    }
    _write_json(artifact_root / "summary_v1.json", summary)

    status_payload = {
        "layer_name": "RUN_CONTEXTUAL_IQL_V2_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
        "final_status_v1": status,
        "next_action_v1": next_action,
        "training_executed_v1": True,
        "policy_count_v1": len(policy_rows),
        "reward_family_count_v1": len(ALL_REWARD_FAMILIES),
    }
    _write_json(artifact_root / "status_v1.json", status_payload)

    go_no_go = {
        "layer_name": "RUN_CONTEXTUAL_IQL_V2_GO_NO_GO_V1",
        "status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "research_only_v1": True,
        "iql_production_allowed_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "policy_promotion_allowed_v1": False,
        "downstream_block_v1": (
            "This gate does not open adapter, R6, IQL production/live, full lifecycle "
            "sequential IQL, policy promotion, package, freeze, promo, or live."
        ),
    }
    for blocked in (
        "iql_production_allowed_v1",
        "adapter_build_allowed_v1",
        "r6_allowed_v1",
        "package_freeze_promo_live_allowed_v1",
        "policy_promotion_allowed_v1",
    ):
        if go_no_go.get(blocked):
            raise RuntimeError(f"FORBIDDEN_PATH_OPENED: {blocked}")
    _write_json(
        artifact_root / "run_contextual_iql_with_v2_state_and_reward_variants_go_no_go_v1.json",
        go_no_go,
    )

    report_lines = [
        "# Run Contextual IQL With V2 State And Reward Variants V1",
        "",
        "## Final status",
        "",
        f"- `{status}`",
        f"- Next action: `{next_action}`",
        "",
        "## Per-reward IQL policies (78-shielded TAKE cohort)",
        "",
        "| Reward | Selected | Precision | Safety | Mean PNL | Mean MFE-capture | Mean MAE-burden | Mean giveback |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in policy_rows:
        report_lines.append(
            "| `{rid}` | {sel} | {prec:.4f} | {safe} | {pnl:.2f} | {mfe:.4f} | {mae:.2f} | {gb:.2f} |".format(
                rid=row["reward_id_v1"],
                sel=row["selected_rows_v1"],
                prec=row["precision_audit_only_v1"],
                safe=row["safety_status_v1"],
                pnl=row["mean_pnl_bps_v1"],
                mfe=row["mean_mfe_capture_v1"],
                mae=row["mean_mae_burden_v1"],
                gb=row["mean_giveback_bps_v1"],
            )
        )
    report_lines += [
        "",
        "## Baseline policies",
        "",
        "| Policy | Selected | Precision | Mean PNL | Mean MFE-capture | Mean MAE-burden |",
        "|---|---|---|---|---|---|",
    ]
    for row in baseline_rows:
        report_lines.append(
            "| `{name}` | {sel} | {prec:.4f} | {pnl:.2f} | {mfe:.4f} | {mae:.2f} |".format(
                name=row["policy_name_v1"],
                sel=row["selected_rows_v1"],
                prec=row["precision_audit_only_v1"],
                pnl=row["mean_pnl_bps_v1"],
                mfe=row["mean_mfe_capture_v1"],
                mae=row["mean_mae_burden_v1"],
            )
        )
    report_lines += [
        "",
        "## Recommendation",
        "",
        recommendation,
    ]
    _write_report(artifact_root / "report_v1.md", report_lines)

    artifact_manifest = {
        "layer_id_v1": ACTION,
        "built_at_utc_v1": summary["built_at_utc_v1"],
        "output_dir_v1": str(artifact_root),
        "append_only_namespace_v1": "truth_e2e_sanity",
        "artifact_paths_v1": {
            "summary": str(artifact_root / "summary_v1.json"),
            "status": str(artifact_root / "status_v1.json"),
            "go_no_go": str(
                artifact_root
                / "run_contextual_iql_with_v2_state_and_reward_variants_go_no_go_v1.json"
            ),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "reproducibility_audit": str(artifact_root / "reproducibility_audit_v1.json"),
            "no_shortcut_audit": str(artifact_root / "no_shortcut_audit_v1.json"),
            "reward_class_audit": str(artifact_root / "reward_class_audit_v1.json"),
            "policy_per_reward_comparator": str(
                artifact_root / "iql_policy_per_reward_comparator_v1.json"
            ),
            "baseline_policy_comparator": str(
                artifact_root / "iql_baseline_policy_comparator_v1.json"
            ),
            "training_configs": str(
                artifact_root / "iql_per_reward_training_configs_v1.json"
            ),
            "reward_arrays_summary": str(
                artifact_root / "reward_arrays_summary_v1.csv"
            ),
            "normalization_audit": str(artifact_root / "normalization_audit_v1.json"),
        },
        "read_only_references_v1": True,
        "not_trainer_for_production_v1": True,
        "not_controller_v1": True,
        "not_live_gate_v1": True,
    }
    _write_json(artifact_root / "manifest_v1.json", artifact_manifest)

    return {
        "artifact_root": str(artifact_root),
        "summary": summary,
        "status": status_payload,
        "go_no_go": go_no_go,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize RUN_CONTEXTUAL_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1 gate."
    )
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    result = write_artifacts(out_root=out_root, built_at_utc=args.built_at_utc)
    print(json.dumps(_jsonable(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
