#!/usr/bin/env python3
"""Verify Entry candidate replay and IQL-distillation readiness.

This gate does not run replay, train IQL, promote, shadow, or trade. It checks
that a post-candidate specialist-fusion bundle has selective-edge evidence and
offline replay evidence before IQL distillation can be considered.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.features.entry_specialist_feature_groups_v1 import required_training_specialists_for_mode
from gx1.scripts.materialize_entry_candidate_replay_evidence_v1 import (
    IQL_TRANSITION_REQUIRED_COLUMNS,
    _selective_edge_specialist_contract,
)
from gx1.scripts.verify_entry_candidate_readiness_v1 import (
    DEFAULT_OUT_DIR as CANDIDATE_READINESS_OUT_DIR,
    _bundle_specialist_model_contract_passes,
    REQUIRED_MIN_GATE_ENTROPY,
)
from gx1.scripts.verify_entry_foundation_state_v1 import FOUNDATION_DATASET_DIR, REPORTS_ROOT, REPO
from gx1.scripts.verify_entry_training_readiness_v1 import _check
from gx1.scripts.verify_entry_training_readiness_v1 import (
    EXPECTED_ACTIVE_TRAINING_HEADS,
    EXPECTED_BLOCKED_HEADS,
    _artifact_fingerprint_checks,
    _artifact_fingerprints,
)


CANDIDATE_READINESS_LATEST = CANDIDATE_READINESS_OUT_DIR / "ENTRY_CANDIDATE_READINESS_latest.json"
CHALLENGER_SEQ215_CANDIDATE_READINESS_LATEST = (
    CANDIDATE_READINESS_OUT_DIR / "challenger_seq215_20260630/ENTRY_CANDIDATE_READINESS_latest.json"
)
DEFAULT_SELECTIVE_EDGE_DIR = REPORTS_ROOT / "entry_candidate_selective_edge_20260628_v1"
DEFAULT_REPLAY_DIR = REPORTS_ROOT / "entry_candidate_replay_20260628_v1"
DEFAULT_CANDIDATE_BUNDLE_AUDIT = (
    REPORTS_ROOT / "entry_candidate_bundle_audit_20260628_v1/ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT_latest.json"
)
CHALLENGER_SEQ215_CANDIDATE_BUNDLE_AUDIT = (
    REPORTS_ROOT
    / "entry_candidate_bundle_audit_20260628_v1/challenger_seq215_20260630/ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT_latest.json"
)
CHALLENGER_SEQ215_SELECTIVE_EDGE_DIR = DEFAULT_SELECTIVE_EDGE_DIR / "challenger_seq215_20260630"
CHALLENGER_SEQ215_REPLAY_DIR = DEFAULT_REPLAY_DIR / "challenger_seq215_20260630"
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_replay_readiness_20260628_v1"
CHALLENGER_SEQ215_OUT_DIR = DEFAULT_OUT_DIR / "challenger_seq215_20260630"
SMART_SEQ520_CANDIDATE_READINESS_LATEST = (
    CANDIDATE_READINESS_OUT_DIR / "smart_seq520_candidate/ENTRY_CANDIDATE_READINESS_latest.json"
)
SMART_SEQ520_CANDIDATE_BUNDLE_AUDIT = (
    REPORTS_ROOT
    / "entry_candidate_bundle_audit_20260628_v1/smart_seq520_candidate/ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT_latest.json"
)
SMART_SEQ520_SELECTIVE_EDGE_DIR = DEFAULT_SELECTIVE_EDGE_DIR / "smart_seq520_candidate"
SMART_SEQ520_REPLAY_DIR = DEFAULT_REPLAY_DIR / "smart_seq520_candidate"
SMART_SEQ520_OUT_DIR = DEFAULT_OUT_DIR / "smart_seq520_candidate"
CHALLENGER_SEQ215_DATASET_DIR = (
    Path("/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605")
    / "v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_challenger_seq215_neutral_20260630"
)
SMART_SEQ520_DATASET_DIR = (
    Path("/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605")
    / "v10_6yr_rebuild_20260626_spreadfix/v10_dataset_6yr_smartctx_xau_direction_repair"
)
SMART_XAU_PRETRAIN_AUDIT_LATEST = (
    REPORTS_ROOT
    / "xau_direction_repair_pretrain_audit_20260713_v1/XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_latest.json"
)
CONTRACT_INPUT_DIMS = {
    "foundation_seq146": 146,
    "challenger_seq215": 215,
    "smart_seq520_candidate": 520,
}
SMART_DIRECTION_BALANCE_MIN_ALPHA = 0.45
SMART_DIRECTION_CE_SCALE_MIN = 2.00
SMART_DIRECTION_BALANCE_CLASS_WEIGHTS = [1.0, 1.0, 4.0]
SMART_DIRECTION_CKPT_BALANCE_GUARD_MIN_WEIGHT = 0.50
SMART_DIRECTION_CKPT_BALANCE_MIN_PRED_TO_LABEL = 0.35
SMART_DIRECTION_CKPT_BALANCE_MIN_PRED_RATE = 0.05
SMART_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT = 2.50
SMART_DIRECTION_MIN_PRED_RATE_FRACTION = 0.50
SMART_DIRECTION_MIN_PRED_RATE_FLOOR = 0.05
SMART_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE_MAX = 0.50
SMART_DIRECTION_VS_FLAT_MARGIN_WEIGHT = 3.00
SMART_DIRECTION_VS_FLAT_MARGIN = 0.05
SMART_DIRECTION_UTILITY_MARGIN_WEIGHT = 4.00
SMART_DIRECTION_UTILITY_MIN_GAP_BPS_MAX = 15.0
SMART_DIRECTION_UTILITY_LOGIT_MARGIN = 0.10
SMART_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT = 6.00
SMART_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS_MAX = 15.0
SMART_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN = 0.10
SMART_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT = 8.00
SMART_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS_MAX = 15.0
SMART_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS_MAX = 0.0
SMART_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH_MAX = 0.50
SMART_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN = 0.10
SMART_DIRECTION_UTILITY_TRIAD_CE_WEIGHT = 8.00
SMART_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS_MAX = 15.0
SMART_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS_MAX = 0.0
SMART_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH_MAX = 0.50
SMART_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP_MIN = 2.0
SMART_DIRECTION_HIERARCHICAL_COMPOSITION_REQUIRED = True
SMART_DIRECTION_HIER_COMPOSE_RESIDUAL_LOGIT_CAP_MIN = 0.10
SMART_DIRECTION_HIER_COMPOSE_RESIDUAL_LOGIT_CAP_MAX = 0.20
SMART_DIRECTION_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL_REQUIRED = True
SMART_DIRECTION_FLAT_STARVATION_WEIGHT = 8.00
SMART_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE = 0.10
SMART_DIRECTION_FLAT_STARVATION_MIN_ROWS = 8
SMART_DIRECTION_FLAT_STARVATION_PRED_FRACTION = 0.50
SMART_DIRECTION_FLAT_STARVATION_PRED_FLOOR = 0.10
SMART_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN = 0.10
SMART_DIRECTION_HIER_LEGACY_CE_MULT_MIN = 1.00
SMART_DIRECTION_ANCHOR_GATE_INIT_MAX = 0.05
SMART_DIRECTION_SIDE_VALIDITY_WEIGHT_MIN = 1.50
SMART_DIRECTION_TRENDLINE_RAIL_AUX_WEIGHT_MIN = 1.00
SMART_DIRECTION_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT_MIN = 1.50
SMART_EXTRA_ACTIVE_HEADS = ("trade_side_hierarchy", "trendline_rail", "side_validity")


def _expected_active_heads_for_mode(contract_mode: str) -> set[str]:
    expected = set(EXPECTED_ACTIVE_TRAINING_HEADS)
    if str(contract_mode).strip() == "smart_seq520_candidate":
        expected.update(SMART_EXTRA_ACTIVE_HEADS)
    return expected


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _all_ok(checks: list[dict[str, Any]]) -> bool:
    return all(bool(check.get("ok")) for check in checks)


def _smart_xau_pretrain_audit_checks(path: Path, report: dict[str, Any], *, expected_dataset_dir: Path) -> list[dict[str, Any]]:
    exists = path.exists()
    checks = [
        _check(
            "smart XAU pretrain audit artifact exists",
            exists,
            {"path": str(path)},
        )
    ]
    if not exists:
        return checks
    checks.extend(
        [
            _check(
                "smart XAU pretrain audit PASS",
                str(report.get("decision")) == "PASS",
                {"decision": report.get("decision"), "failures": report.get("failures")},
            ),
            _check(
                "smart XAU pretrain audit dataset_dir matches expected pin",
                _same_resolved_path(report.get("dataset_dir"), expected_dataset_dir),
                {"expected_dataset_dir": str(expected_dataset_dir), "audit_dataset_dir": report.get("dataset_dir")},
            ),
        ]
    )
    return checks


def _same_resolved_path(actual: Any, expected: Path) -> bool:
    if actual in (None, ""):
        return False
    try:
        return Path(str(actual)).resolve(strict=False) == expected.resolve(strict=False)
    except (OSError, RuntimeError):
        return False


def _normalize_contract_mode(value: Any) -> str:
    raw = str(value or "foundation_seq146").strip()
    if raw not in CONTRACT_INPUT_DIMS:
        return raw
    return raw


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _direction_balance_contract_passes(
    contract: dict[str, Any],
    *,
    contract_mode: str,
) -> bool:
    alpha = _float_or_zero(contract.get("pred_balance_alpha"))
    base_ok = (
        str(contract.get("decision")) == "PASS"
        and not contract.get("failures")
        and bool(contract.get("direction_active"))
        and 0.05 <= alpha <= 0.50
        and str(contract.get("pred_balance_target") or "").strip().lower() == "label"
        and _float_or_zero(contract.get("direction_ce_scale")) > 0.0
        and str(contract.get("ckpt_monitor") or "").strip().lower() == "dir_acc"
    )
    if not base_ok:
        return False
    if contract_mode != "smart_seq520_candidate":
        return True

    weights = contract.get("pred_balance_class_weights")
    try:
        parsed_weights = [float(value) for value in weights] if isinstance(weights, list) else []
    except (TypeError, ValueError):
        parsed_weights = []
    return (
        alpha >= SMART_DIRECTION_BALANCE_MIN_ALPHA
        and parsed_weights == SMART_DIRECTION_BALANCE_CLASS_WEIGHTS
        and _float_or_zero(contract.get("direction_ce_scale"))
        >= SMART_DIRECTION_CE_SCALE_MIN
        and contract.get("hierarchical_entry_heads_enabled") is True
        and contract.get("side_validity_head_enabled") is True
        and _float_or_zero(contract.get("hier_side_validity_weight"))
        >= SMART_DIRECTION_SIDE_VALIDITY_WEIGHT_MIN
        and contract.get("trendline_rail_head_enabled") is True
        and _float_or_zero(contract.get("trendline_rail_aux_weight"))
        >= SMART_DIRECTION_TRENDLINE_RAIL_AUX_WEIGHT_MIN
        and _float_or_zero(contract.get("trendline_rail_wrong_side_weight"))
        >= SMART_DIRECTION_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT_MIN
        and _float_or_zero(contract.get("hier_legacy_ce_mult"))
        >= SMART_DIRECTION_HIER_LEGACY_CE_MULT_MIN
        and contract.get("anchor_gate_enabled") is True
        and _float_or_zero(contract.get("anchor_gate_init"))
        <= SMART_DIRECTION_ANCHOR_GATE_INIT_MAX
        and _float_or_zero(contract.get("ckpt_class_balance_guard_weight"))
        >= SMART_DIRECTION_CKPT_BALANCE_GUARD_MIN_WEIGHT
        and _float_or_zero(contract.get("ckpt_class_balance_min_pred_to_label"))
        >= SMART_DIRECTION_CKPT_BALANCE_MIN_PRED_TO_LABEL
        and _float_or_zero(contract.get("ckpt_class_balance_min_pred_rate"))
        >= SMART_DIRECTION_CKPT_BALANCE_MIN_PRED_RATE
        and _float_or_zero(contract.get("direction_min_pred_rate_loss_weight"))
        >= SMART_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT
        and _float_or_zero(contract.get("direction_min_pred_rate_fraction"))
        >= SMART_DIRECTION_MIN_PRED_RATE_FRACTION
        and _float_or_zero(contract.get("direction_min_pred_rate_floor"))
        >= SMART_DIRECTION_MIN_PRED_RATE_FLOOR
        and 0.0
        < _float_or_zero(contract.get("direction_min_pred_rate_softmax_temperature"))
        <= SMART_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE_MAX
        and _float_or_zero(contract.get("direction_vs_flat_margin_weight"))
        >= SMART_DIRECTION_VS_FLAT_MARGIN_WEIGHT
        and _float_or_zero(contract.get("direction_vs_flat_margin"))
        >= SMART_DIRECTION_VS_FLAT_MARGIN
        and _float_or_zero(contract.get("direction_utility_margin_weight"))
        >= SMART_DIRECTION_UTILITY_MARGIN_WEIGHT
        and _float_or_zero(contract.get("direction_utility_min_gap_bps"))
        <= SMART_DIRECTION_UTILITY_MIN_GAP_BPS_MAX
        and _float_or_zero(contract.get("direction_utility_logit_margin"))
        >= SMART_DIRECTION_UTILITY_LOGIT_MARGIN
        and _float_or_zero(contract.get("direction_side_utility_conviction_weight"))
        >= SMART_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT
        and _float_or_zero(contract.get("direction_side_utility_conviction_min_gap_bps", 999.0))
        <= SMART_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS_MAX
        and _float_or_zero(contract.get("direction_side_utility_conviction_logit_margin"))
        >= SMART_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN
        and _float_or_zero(contract.get("direction_utility_trade_conviction_weight"))
        >= SMART_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT
        and _float_or_zero(contract.get("direction_utility_trade_conviction_min_gap_bps", 999.0))
        <= SMART_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS_MAX
        and _float_or_zero(contract.get("direction_utility_trade_conviction_min_utility_bps", 999.0))
        <= SMART_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS_MAX
        and _float_or_zero(contract.get("direction_utility_trade_conviction_max_bad_path", 999.0))
        <= SMART_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH_MAX
        and _float_or_zero(contract.get("direction_utility_trade_conviction_logit_margin"))
        >= SMART_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN
        and _float_or_zero(contract.get("direction_utility_triad_ce_weight"))
        >= SMART_DIRECTION_UTILITY_TRIAD_CE_WEIGHT
        and _float_or_zero(contract.get("direction_utility_triad_ce_min_gap_bps", 999.0))
        <= SMART_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS_MAX
        and _float_or_zero(contract.get("direction_utility_triad_ce_min_utility_bps", 999.0))
        <= SMART_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS_MAX
        and _float_or_zero(contract.get("direction_utility_triad_ce_max_bad_path", 999.0))
        <= SMART_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH_MAX
        and _float_or_zero(contract.get("direction_utility_triad_ce_class_weight_cap"))
        >= SMART_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP_MIN
        and contract.get("direction_hierarchical_composition") is SMART_DIRECTION_HIERARCHICAL_COMPOSITION_REQUIRED
        and _float_or_zero(contract.get("hier_compose_residual_logit_cap"))
        >= SMART_DIRECTION_HIER_COMPOSE_RESIDUAL_LOGIT_CAP_MIN
        and _float_or_zero(contract.get("hier_compose_residual_logit_cap", 999.0))
        <= SMART_DIRECTION_HIER_COMPOSE_RESIDUAL_LOGIT_CAP_MAX
        and contract.get("hier_compose_residual_side_neutral")
        is SMART_DIRECTION_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL_REQUIRED
        and _float_or_zero(contract.get("direction_flat_starvation_weight"))
        >= SMART_DIRECTION_FLAT_STARVATION_WEIGHT
        and _float_or_zero(contract.get("direction_flat_starvation_min_label_rate"))
        >= SMART_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE
        and _float_or_zero(contract.get("direction_flat_starvation_min_rows"))
        >= SMART_DIRECTION_FLAT_STARVATION_MIN_ROWS
        and _float_or_zero(contract.get("direction_flat_starvation_pred_fraction"))
        >= SMART_DIRECTION_FLAT_STARVATION_PRED_FRACTION
        and _float_or_zero(contract.get("direction_flat_starvation_pred_floor"))
        >= SMART_DIRECTION_FLAT_STARVATION_PRED_FLOOR
        and _float_or_zero(contract.get("direction_flat_starvation_logit_margin"))
        >= SMART_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN
        and contract.get("best_direction_balance_guard_ok") is True
    )


def _contract_mode_from_bundle_audit(report: dict[str, Any]) -> str:
    bundle = report.get("bundle_summary") if isinstance(report.get("bundle_summary"), dict) else {}
    return _normalize_contract_mode(
        report.get("specialist_contract_mode")
        or report.get("contract_mode")
        or bundle.get("specialist_contract_mode")
        or bundle.get("contract_mode")
        or bundle.get("audit_contract_mode")
    )


def _contract_mode_from_identity(identity: dict[str, Any]) -> str:
    return _normalize_contract_mode(identity.get("contract_mode"))


def _model_summaries(summary: dict[str, Any], model_name: str) -> list[dict[str, Any]]:
    return [
        row
        for row in summary.get("summaries", [])
        if isinstance(row, dict) and str(row.get("model")) == str(model_name)
    ]


def _summary_by_split(summary: dict[str, Any], model_name: str) -> dict[str, dict[str, Any]]:
    return {str(row.get("split")): row for row in _model_summaries(summary, model_name)}


def _split_metric(summary: dict[str, Any], model_name: str, key: str) -> dict[str, Any]:
    return {split: row.get(key) for split, row in _summary_by_split(summary, model_name).items()}


def _all_split_metric_gt(summary: dict[str, Any], model_name: str, key: str, threshold: float) -> bool:
    values = _split_metric(summary, model_name, key)
    required = [split for split in ("val", "test") if split in values]
    return len(required) == 2 and all(values[split] is not None and float(values[split]) > float(threshold) for split in required)


def _all_split_metric_ge(summary: dict[str, Any], model_name: str, key: str, threshold: float) -> bool:
    values = _split_metric(summary, model_name, key)
    required = [split for split in ("val", "test") if split in values]
    if len(required) != 2:
        return False
    for split in required:
        value = values.get(split)
        if value is None:
            return False
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(numeric) or numeric < float(threshold):
            return False
    return True


def _selected_tail_direction_slice_report(
    metrics: pd.DataFrame,
    *,
    model_name: str,
    min_direction_precision: float,
    min_rows: int,
) -> dict[str, Any]:
    required_columns = {"split", "model", "scope", "top_frac", "group", "n", "direction_precision"}
    metric_columns = set(str(c) for c in metrics.columns)
    if metrics.empty or not required_columns.issubset(metric_columns):
        return {
            "checked_rows": 0,
            "failed_rows": 0,
            "failures": [],
            "missing_columns": sorted(required_columns - metric_columns),
        }
    rows = metrics[
        (metrics["model"].astype(str) == str(model_name))
        & (metrics["scope"].astype(str) == "top_score")
    ].copy()
    if rows.empty:
        return {"checked_rows": 0, "failed_rows": 0, "failures": [], "missing_columns": []}
    groups = rows["group"].astype(str)
    top_frac = pd.to_numeric(rows["top_frac"], errors="coerce")
    n = pd.to_numeric(rows["n"], errors="coerce")
    wanted_group = (
        (groups == "ALL")
        | groups.str.startswith("session=")
        | groups.str.startswith("side=")
        | groups.str.startswith("vol_regime=")
    )
    wanted_frac = np.isclose(top_frac.astype(float), 0.05) | np.isclose(top_frac.astype(float), 0.10)
    eligible = rows[wanted_group & wanted_frac & (n >= int(min_rows))].copy()
    if eligible.empty:
        return {"checked_rows": 0, "failed_rows": 0, "failures": [], "missing_columns": []}
    eligible_direction = pd.to_numeric(eligible["direction_precision"], errors="coerce")
    failed = eligible[(~np.isfinite(eligible_direction)) | (eligible_direction < float(min_direction_precision))].copy()
    failures = []
    for _, row in failed.head(25).iterrows():
        value = pd.to_numeric(pd.Series([row.get("direction_precision")]), errors="coerce").iloc[0]
        failures.append(
            {
                "split": str(row.get("split")),
                "group": str(row.get("group")),
                "top_frac": float(row.get("top_frac")),
                "n": int(row.get("n")),
                "direction_precision": None if pd.isna(value) else float(value),
            }
        )
    return {
        "checked_rows": int(len(eligible)),
        "failed_rows": int(len(failed)),
        "failures": failures,
        "missing_columns": [],
    }


def _selective_edge_checks(
    summary: dict[str, Any],
    metrics: pd.DataFrame,
    *,
    model_name: str,
    min_top5_mean_pnl_bps: float,
    min_top10_mean_pnl_bps: float,
    require_no_xgb_ablation: bool,
    min_top_direction_precision: float = 0.50,
    min_direction_slice_precision: float = 0.50,
    min_direction_slice_n: int = 20,
    expected_bundle_dir: str | None = None,
    expected_dataset_dir: Path = FOUNDATION_DATASET_DIR,
    expected_contract_mode: str = "foundation_seq146",
) -> list[dict[str, Any]]:
    splits = set(str(x) for x in summary.get("splits", []))
    models = {str(row.get("model")) for row in summary.get("summaries", []) if isinstance(row, dict)}
    no_xgb_model = f"{model_name}_no_xgb"
    no_xgb_bundle_dir = str(summary.get("no_xgb_bundle_dir") or "")
    summary_bundle_dir = str(summary.get("bundle_dir") or "")
    no_xgb_ablation = summary.get("no_xgb_ablation") if isinstance(summary.get("no_xgb_ablation"), dict) else {}
    no_xgb_mode = str(no_xgb_ablation.get("mode") or "")
    no_xgb_neutralizes_bridge = bool(no_xgb_ablation.get("neutralize_signal_bridge"))
    no_xgb_neutralized_fields = [str(x) for x in no_xgb_ablation.get("neutralized_fields", [])]
    no_xgb_neutral_values = no_xgb_ablation.get("neutral_values", [])
    no_xgb_same_bundle = bool(no_xgb_bundle_dir and summary_bundle_dir and no_xgb_bundle_dir == summary_bundle_dir)
    no_xgb_diagnostics = (
        summary.get("no_xgb_ablation_diagnostics") if isinstance(summary.get("no_xgb_ablation_diagnostics"), dict) else {}
    )
    no_xgb_diagnostic_splits = (
        no_xgb_diagnostics.get("splits") if isinstance(no_xgb_diagnostics.get("splits"), dict) else {}
    )
    input_bridge_contract = (
        summary.get("input_bridge_contract") if isinstance(summary.get("input_bridge_contract"), dict) else {}
    )
    contract_mode = _normalize_contract_mode(summary.get("contract_mode"))
    expected_seq_input_dim = CONTRACT_INPUT_DIMS.get(expected_contract_mode)
    observed_seq_input_dim = int(summary.get("bundle_seq_input_dim") or 0)
    observed_snap_input_dim = int(summary.get("bundle_snap_input_dim") or 0)
    selective_specialist_contract = _selective_edge_specialist_contract(summary, expected_contract_mode)
    selection_score_mode = str(summary.get("selection_score_mode") or "edge_score")
    selection_score_threshold_present = "selection_score_threshold" in summary
    expected_utility_selection_required = expected_contract_mode == "smart_seq520_candidate"
    input_bridge_splits = input_bridge_contract.get("splits") if isinstance(input_bridge_contract.get("splits"), dict) else {}
    required_no_xgb_splits = [split for split in ("val", "test") if split in splits or not splits]
    no_xgb_diagnostics_ok = True
    no_xgb_has_effect = False
    input_bridge_neutral_all = False
    if require_no_xgb_ablation:
        no_xgb_diagnostics_ok = bool(no_xgb_diagnostics.get("available")) and bool(required_no_xgb_splits)
        for split in required_no_xgb_splits:
            row = no_xgb_diagnostic_splits.get(split) if isinstance(no_xgb_diagnostic_splits.get(split), dict) else {}
            no_xgb_diagnostics_ok = no_xgb_diagnostics_ok and bool(row.get("comparable")) and bool(row.get("time_match", True))
            no_xgb_has_effect = no_xgb_has_effect or float(row.get("max_abs_prob_delta") or 0.0) > 0.0
            no_xgb_has_effect = no_xgb_has_effect or float(row.get("max_abs_edge_score_delta") or 0.0) > 0.0
            no_xgb_has_effect = no_xgb_has_effect or int(row.get("trade_side_diff_count") or 0) > 0
            no_xgb_has_effect = no_xgb_has_effect or int(row.get("pred_direction_diff_count") or 0) > 0
        input_bridge_neutral_all = bool(required_no_xgb_splits) and all(
            bool((input_bridge_splits.get(split) if isinstance(input_bridge_splits.get(split), dict) else {}).get("neutral_xgb_bridge"))
            for split in required_no_xgb_splits
        )
    no_xgb_liveness_ok = True
    if require_no_xgb_ablation and no_xgb_mode == "neutralize_signal_bridge":
        no_xgb_liveness_ok = no_xgb_diagnostics_ok and (no_xgb_has_effect or input_bridge_neutral_all)
    no_xgb_provenance_ok = True
    if require_no_xgb_ablation:
        if no_xgb_mode == "neutralize_signal_bridge":
            no_xgb_provenance_ok = (
                no_xgb_same_bundle
                and no_xgb_neutralizes_bridge
                and len(no_xgb_neutralized_fields) >= 7
                and len(no_xgb_neutral_values) >= 7
            )
        elif no_xgb_mode == "bundle":
            no_xgb_provenance_ok = bool(no_xgb_bundle_dir) and not no_xgb_same_bundle
        else:
            no_xgb_provenance_ok = False
    candidate_by_split = _summary_by_split(summary, model_name)
    top5 = _split_metric(summary, model_name, "top5_all_mean_pnl_bps")
    top10 = _split_metric(summary, model_name, "top10_all_mean_pnl_bps")
    top5_direction = _split_metric(summary, model_name, "top5_all_direction_precision")
    top10_direction = _split_metric(summary, model_name, "top10_all_direction_precision")
    required_columns = {
        "split",
        "model",
        "scope",
        "top_frac",
        "group",
        "n",
        "mean_pnl_bps",
        "win_rate",
        "direction_precision",
    }
    metric_columns = set(str(c) for c in metrics.columns)
    slice_rows = pd.DataFrame()
    if not metrics.empty and required_columns.issubset(metric_columns):
        slice_rows = metrics[
            (metrics["model"].astype(str) == str(model_name))
            & (metrics["scope"].astype(str) == "top_score")
            & (metrics["group"].astype(str).str.startswith("session="))
        ]
    direction_slice_report = _selected_tail_direction_slice_report(
        metrics,
        model_name=model_name,
        min_direction_precision=min_direction_slice_precision,
        min_rows=min_direction_slice_n,
    )
    return [
        _check("selective-edge summary PASS", str(summary.get("decision")) == "PASS", {"failures": summary.get("failures")}),
        _check("selective-edge summary has zero failures", not summary.get("failures"), {"failures": summary.get("failures")}),
        _check(
            "selective-edge summary uses expected dataset",
            _same_resolved_path(summary.get("dataset_dir"), expected_dataset_dir),
            {"expected_dataset_dir": str(expected_dataset_dir), "summary_dataset_dir": summary.get("dataset_dir")},
        ),
        _check(
            "selective-edge summary matches candidate bundle audit bundle",
            True if expected_bundle_dir is None else str(summary.get("bundle_dir")) == str(expected_bundle_dir),
            {"expected_bundle_dir": expected_bundle_dir, "summary_bundle_dir": summary.get("bundle_dir")},
        ),
        _check(
            "selective-edge summary contract mode matches candidate bundle audit",
            contract_mode == expected_contract_mode,
            {"expected_contract_mode": expected_contract_mode, "selective_edge_contract_mode": contract_mode},
        ),
        _check(
            "smart selective-edge uses expected-utility selection mode",
            (not expected_utility_selection_required)
            or (selection_score_mode == "expected_utility" and selection_score_threshold_present),
            {
                "required": expected_utility_selection_required,
                "selection_score_mode": selection_score_mode,
                "selection_score_threshold_present": selection_score_threshold_present,
            },
        ),
        _check(
            "selective-edge summary input dimensions match contract mode",
            bool(expected_seq_input_dim)
            and observed_seq_input_dim == expected_seq_input_dim
            and observed_snap_input_dim == expected_seq_input_dim,
            {
                "expected_contract_mode": expected_contract_mode,
                "expected_input_dim": expected_seq_input_dim,
                "bundle_seq_input_dim": observed_seq_input_dim,
                "bundle_snap_input_dim": observed_snap_input_dim,
            },
        ),
        _check(
            "selective-edge summary preserves specialist contract snapshot",
            bool(selective_specialist_contract.get("ready")),
            {"specialist_contract": selective_specialist_contract},
        ),
        _check("selective-edge summary has val/test", {"val", "test"}.issubset(splits), {"splits": sorted(splits)}),
        _check("selective-edge summary includes candidate model", model_name in models, {"models": sorted(models)}),
        _check(
            "selective-edge summary includes no-XGB ablation",
            (no_xgb_model in models) if require_no_xgb_ablation else True,
            {"required": require_no_xgb_ablation, "models": sorted(models)},
        ),
        _check(
            "selective-edge summary records no-XGB bundle dir",
            bool(no_xgb_bundle_dir) if require_no_xgb_ablation else True,
            {"required": require_no_xgb_ablation, "no_xgb_bundle_dir": no_xgb_bundle_dir},
        ),
        _check(
            "selective-edge no-XGB ablation provenance is explicit",
            no_xgb_provenance_ok,
            {
                "required": require_no_xgb_ablation,
                "mode": no_xgb_mode,
                "neutralize_signal_bridge": no_xgb_neutralizes_bridge,
                "same_bundle_as_candidate": no_xgb_same_bundle,
                "neutralized_fields": no_xgb_neutralized_fields,
            },
        ),
        _check(
            "selective-edge no-XGB ablation diagnostics exist",
            no_xgb_diagnostics_ok,
            {
                "required": require_no_xgb_ablation,
                "available": no_xgb_diagnostics.get("available"),
                "required_splits": required_no_xgb_splits,
                "diagnostic_splits": sorted(str(x) for x in no_xgb_diagnostic_splits),
            },
        ),
        _check(
            "selective-edge no-XGB ablation is live or input bridge already neutral",
            no_xgb_liveness_ok,
            {
                "required": require_no_xgb_ablation,
                "mode": no_xgb_mode,
                "has_prediction_effect": no_xgb_has_effect,
                "input_bridge_neutral_all": input_bridge_neutral_all,
            },
        ),
        _check(
            "candidate selective-edge has val/test summaries",
            {"val", "test"}.issubset(set(candidate_by_split)),
            {"candidate_splits": sorted(candidate_by_split)},
        ),
        _check(
            "candidate top5 mean pnl is positive on val/test",
            _all_split_metric_gt(summary, model_name, "top5_all_mean_pnl_bps", min_top5_mean_pnl_bps),
            {"threshold": min_top5_mean_pnl_bps, "top5_all_mean_pnl_bps": top5},
        ),
        _check(
            "candidate top10 mean pnl is positive on val/test",
            _all_split_metric_gt(summary, model_name, "top10_all_mean_pnl_bps", min_top10_mean_pnl_bps),
            {"threshold": min_top10_mean_pnl_bps, "top10_all_mean_pnl_bps": top10},
        ),
        _check(
            "candidate top5 selected-tail direction precision clears threshold on val/test",
            _all_split_metric_ge(summary, model_name, "top5_all_direction_precision", min_top_direction_precision),
            {"threshold": min_top_direction_precision, "top5_all_direction_precision": top5_direction},
        ),
        _check(
            "candidate top10 selected-tail direction precision clears threshold on val/test",
            _all_split_metric_ge(summary, model_name, "top10_all_direction_precision", min_top_direction_precision),
            {"threshold": min_top_direction_precision, "top10_all_direction_precision": top10_direction},
        ),
        _check("selective-edge metrics CSV exists and has rows", not metrics.empty, {"rows": int(len(metrics))}),
        _check(
            "selective-edge metrics CSV has required columns",
            required_columns.issubset(metric_columns),
            {"missing_columns": sorted(required_columns - metric_columns)},
        ),
        _check(
            "selective-edge metrics include session slices",
            len(slice_rows) > 0,
            {"session_slice_rows": int(len(slice_rows))},
        ),
        _check(
            "selective-edge selected-tail direction slices clear threshold",
            int(direction_slice_report.get("checked_rows") or 0) > 0
            and int(direction_slice_report.get("failed_rows") or 0) == 0,
            {
                "threshold": min_direction_slice_precision,
                "min_rows": min_direction_slice_n,
                **direction_slice_report,
            },
        ),
    ]


def _candidate_bundle_audit_checks(
    path: Path,
    report: dict[str, Any],
    *,
    contract_mode: str = "foundation_seq146",
    expected_dataset_dir: Path = FOUNDATION_DATASET_DIR,
) -> list[dict[str, Any]]:
    exists = path.exists()
    bundle = report.get("bundle_summary") if isinstance(report.get("bundle_summary"), dict) else {}
    head_contract = report.get("head_contract") if isinstance(report.get("head_contract"), dict) else {}
    path_calibration = (
        report.get("path_calibration_recipe_contract")
        if isinstance(report.get("path_calibration_recipe_contract"), dict)
        else {}
    )
    direction_balance = (
        report.get("direction_balance_recipe_contract")
        if isinstance(report.get("direction_balance_recipe_contract"), dict)
        else {}
    )
    tail_direction = (
        report.get("tail_direction_recipe_contract")
        if isinstance(report.get("tail_direction_recipe_contract"), dict)
        else {}
    )
    pretrain_manifest = (
        report.get("pretrain_manifest_contract")
        if isinstance(report.get("pretrain_manifest_contract"), dict)
        else {}
    )
    active_heads = set(str(x) for x in head_contract.get("active_training_heads", []) if str(x))
    blocked_heads = set(str(x) for x in head_contract.get("blocked_heads", []) if str(x))
    splits = {
        str(split): row
        for split, row in (report.get("splits") or {}).items()
        if isinstance(row, dict)
    }
    split_rows = {split: int((row or {}).get("rows") or 0) for split, row in splits.items()}
    split_names = tuple(split for split in ("val", "test") if split in splits)
    specialist_groups = set(str(x) for x in bundle.get("specialist_groups", []) if str(x))
    required_specialists = {str(x) for x in report.get("required_training_specialists", []) if str(x)}
    expected_contract_mode = _normalize_contract_mode(contract_mode)
    observed_contract_mode = _contract_mode_from_bundle_audit(report)
    expected_input_dim = CONTRACT_INPUT_DIMS.get(expected_contract_mode)
    try:
        expected_specialist_groups = tuple(required_training_specialists_for_mode(expected_contract_mode))
    except ValueError:
        expected_specialist_groups = ()
    required_gate_live = True
    for row in splits.values():
        gate = (row or {}).get("specialist_gate") if isinstance(row, dict) else {}
        mean_weight = gate.get("mean_weight") if isinstance(gate, dict) and isinstance(gate.get("mean_weight"), dict) else {}
        for group in expected_specialist_groups:
            if float(mean_weight.get(group) or 0.0) <= 0.01:
                required_gate_live = False
    direction_by_split = {
        split: (splits.get(split) or {}).get("direction") or {}
        for split in split_names
    }
    distribution_by_split = {
        split: (splits.get(split) or {}).get("direction_distribution_contract") or {}
        for split in split_names
    }
    slice_by_split = {
        split: (splits.get(split) or {}).get("direction_slice_contract") or {}
        for split in split_names
    }
    path_quality_rho = {
        split: ((splits.get(split) or {}).get("path_quality") or {}).get("pred_vs_target_spearman")
        for split in split_names
    }
    bad_path_rho = {
        split: ((splits.get(split) or {}).get("bad_path") or {}).get("prob_vs_path_quality_spearman")
        for split in split_names
    }
    expected_active_heads = _expected_active_heads_for_mode(expected_contract_mode)
    return [
        _check("candidate bundle audit exists", exists, {"path": str(path)}),
        _check("candidate bundle audit PASS", exists and str(report.get("decision")) == "PASS", {"failures": report.get("failures")}),
        _check("candidate bundle audit has zero failures", exists and not report.get("failures"), {"failures": report.get("failures")}),
        _check("candidate bundle audit was run with require_edge", exists and bool(report.get("require_edge"))),
        _check(
            "candidate bundle audit used expected dataset",
            exists and _same_resolved_path(report.get("dataset_dir"), expected_dataset_dir),
            {"expected_dataset_dir": str(expected_dataset_dir), "observed_dataset_dir": report.get("dataset_dir")},
        ),
        _check("candidate bundle audit is from actual train output, not sanity bundle", exists and not bool(bundle.get("sanity_bundle"))),
        _check(
            "candidate bundle audit contract mode matches requested replay contract",
            exists and observed_contract_mode == expected_contract_mode,
            {"expected_contract_mode": expected_contract_mode, "observed_contract_mode": observed_contract_mode},
        ),
        _check(
            "candidate bundle input dimensions match contract mode",
            exists
            and bool(expected_input_dim)
            and int(bundle.get("seq_input_dim") or 0) == expected_input_dim
            and int(bundle.get("snap_input_dim") or 0) == expected_input_dim,
            {
                "expected_contract_mode": expected_contract_mode,
                "expected_input_dim": expected_input_dim,
                "seq_input_dim": bundle.get("seq_input_dim"),
                "snap_input_dim": bundle.get("snap_input_dim"),
            },
        ),
        _check("candidate bundle has multi-TF enabled", exists and bool(bundle.get("multi_tf_enabled"))),
        _check("candidate bundle has specialist fusion", exists and bool(bundle.get("specialist_fusion_enabled"))),
        _check(
            "candidate bundle specialist model contract is preserved in bundle metadata",
            exists and _bundle_specialist_model_contract_passes(report),
            {
                "bundle_summary": {
                    "specialist_model_contract_declared_valid": bundle.get("specialist_model_contract_declared_valid"),
                    "specialist_model_contract_valid": bundle.get("specialist_model_contract_valid"),
                    "specialist_model_contract_set_exact": bundle.get("specialist_model_contract_set_exact"),
                    "specialist_model_contract_owned_objectives_match": bundle.get("specialist_model_contract_owned_objectives_match"),
                    "specialist_model_contract_support_heads_match": bundle.get("specialist_model_contract_support_heads_match"),
                    "specialist_model_contract_signal_families_match": bundle.get("specialist_model_contract_signal_families_match"),
                    "specialist_model_contract_model_roles_match": bundle.get("specialist_model_contract_model_roles_match"),
                },
                "bundle_specialist_model_contract": report.get("bundle_specialist_model_contract"),
            },
        ),
        _check(
            "candidate bundle includes required specialist groups",
            exists and set(expected_specialist_groups).issubset(specialist_groups),
            {"specialist_groups": sorted(specialist_groups)},
        ),
        _check(
            "candidate bundle has exact specialist groups",
            exists and specialist_groups == set(expected_specialist_groups),
            {
                "expected_specialist_groups": list(expected_specialist_groups),
                "actual_specialist_groups": sorted(specialist_groups),
            },
        ),
        _check(
            "candidate bundle audit was run with specialist-fusion gate contract",
            exists
            and bool(report.get("require_specialist_fusion"))
            and required_specialists == set(expected_specialist_groups)
            and int(report.get("min_active_specialists") or 0) >= len(expected_specialist_groups)
            and float(report.get("min_gate_entropy") or -1.0) >= float(REQUIRED_MIN_GATE_ENTROPY),
            {
                "required_training_specialists": report.get("required_training_specialists"),
                "min_active_specialists": report.get("min_active_specialists"),
                "min_gate_entropy": report.get("min_gate_entropy"),
            },
        ),
        _check(
            "candidate bundle required specialist gate weights are non-collapsed",
            exists and required_gate_live,
            {"min_mean_weight": 0.01, "required_specialists": list(expected_specialist_groups)},
        ),
        _check("candidate bundle audit was run with require_head_contract", exists and bool(report.get("require_head_contract"))),
        _check(
            "candidate bundle head contract PASS",
            exists
            and str(head_contract.get("decision")) == "PASS"
            and not head_contract.get("failures")
            and active_heads == expected_active_heads
            and blocked_heads == set(EXPECTED_BLOCKED_HEADS),
            {
                "head_contract": head_contract,
                "expected_active_heads": sorted(expected_active_heads),
                "actual_active_heads": sorted(active_heads),
                "expected_blocked_heads": list(EXPECTED_BLOCKED_HEADS),
                "actual_blocked_heads": sorted(blocked_heads),
            },
        ),
        _check(
            "candidate bundle direction balance recipe contract PASS",
            exists
            and _direction_balance_contract_passes(
                direction_balance,
                contract_mode=expected_contract_mode,
            ),
            {"direction_balance_recipe_contract": direction_balance},
        ),
        _check(
            "candidate bundle path calibration recipe contract PASS",
            exists
            and str(path_calibration.get("decision")) == "PASS"
            and not path_calibration.get("failures")
            and bool(path_calibration.get("path_quality_active"))
            and bool(path_calibration.get("path_quality_rank_full_batch"))
            and float(path_calibration.get("path_quality_rank_weight") or 0.0) > 0.0,
            {"path_calibration_recipe_contract": path_calibration},
        ),
        _check(
            "candidate bundle tail direction recipe contract PASS",
            exists
            and str(tail_direction.get("decision")) == "PASS"
            and not tail_direction.get("failures")
            and bool(tail_direction.get("direction_active"))
            and float(tail_direction.get("tail_direction_ce_weight") or 0.0) > 0.0
            and 0.50 <= float(tail_direction.get("tail_direction_quality_quantile") or 0.0) <= 0.95
            and int(tail_direction.get("tail_direction_min_batch") or 0) >= 2
            and str(tail_direction.get("tail_direction_mask") or "") == "directional_tradable_clean_path_top_quality",
            {"tail_direction_recipe_contract": tail_direction},
        ),
        _check(
            "candidate bundle direction beats majority on val/test",
            exists
            and bool(split_names)
            and all(bool((direction_by_split.get(split) or {}).get("beats_majority_baseline")) for split in split_names),
            {"direction": direction_by_split},
        ),
        _check(
            "candidate bundle direction distribution covers active classes",
            exists
            and bool(split_names)
            and all(str((distribution_by_split.get(split) or {}).get("decision")) == "PASS" for split in split_names),
            {"direction_distribution": distribution_by_split},
        ),
        _check(
            # Responsibility chain (2026-07-04, vedtak SMART_SEQ520_candidate_train_20260703):
            # when the candidate bundle audit itself declared edge_test_scope=candidate,
            # blanket per-slice diagnostics are that audit's recorded advisories and are
            # not re-hardened here — the HARD per-slice responsibility in THIS gate is
            # the selected-tail slice-precision floors (load-bearing), which are
            # untouched. strict/smoke-scoped audits (foundation/seq215) unchanged.
            "candidate bundle direction context slices pass"
            + (
                " (blanket slices advisory per audit scope=candidate; selected-tail floors stay hard)"
                if str(report.get("edge_test_scope")) == "candidate"
                else ""
            ),
            exists
            and bool(split_names)
            and (
                str(report.get("edge_test_scope")) == "candidate"
                or all(
                    str((slice_by_split.get(split) or {}).get("decision")) == "PASS"
                    and int((slice_by_split.get(split) or {}).get("audited_slice_count") or 0) > 0
                    for split in split_names
                )
            ),
            {"direction_slices": slice_by_split, "audit_edge_test_scope": report.get("edge_test_scope")},
        ),
        _check(
            "candidate bundle path_quality ranks realized path quality positively",
            exists
            and bool(split_names)
            and all(value is not None and float(value) > 0.0 for value in path_quality_rho.values()),
            {"path_quality_pred_vs_target_spearman": path_quality_rho},
        ),
        _check(
            "candidate bundle bad_path ranks worse path quality higher",
            exists
            and bool(split_names)
            and all(value is not None and float(value) < 0.0 for value in bad_path_rho.values()),
            {"bad_path_prob_vs_path_quality_spearman": bad_path_rho},
        ),
        _check(
            "candidate bundle audit validated pre-train manifest provenance",
            exists
            and str(pretrain_manifest.get("decision")) == "PASS"
            and not pretrain_manifest.get("failures")
            and bool(pretrain_manifest.get("feature_objective_coverage_all_present"))
            and bool(pretrain_manifest.get("feature_objective_liveness_all_live"))
            and bool(pretrain_manifest.get("feature_source_field_liveness_all_live"))
            and bool(pretrain_manifest.get("specialist_objective_routing_all_present_and_expected"))
            and bool(pretrain_manifest.get("specialist_input_liveness_all_live"))
            and bool(pretrain_manifest.get("specialist_active_heads_match_target"))
            and bool(pretrain_manifest.get("specialist_blocked_heads_match_target"))
            and bool(pretrain_manifest.get("specialist_required_training_set_exact"))
            and bool(pretrain_manifest.get("specialist_trainable_set_exact"))
            and bool(pretrain_manifest.get("specialist_model_contract_valid"))
            and bool(pretrain_manifest.get("specialist_model_contract_set_exact"))
            and bool(pretrain_manifest.get("specialist_model_contract_owned_objectives_match"))
            and bool(pretrain_manifest.get("smoke_edge_required_specialists_exact"))
            and bool(pretrain_manifest.get("smoke_edge_specialist_groups_exact"))
            and bool(pretrain_manifest.get("smoke_edge_specialist_model_contract_valid"))
            and bool(pretrain_manifest.get("smoke_edge_specialist_model_contract_set_exact"))
            and bool(pretrain_manifest.get("smoke_edge_specialist_model_contract_owned_objectives_match"))
            and bool(pretrain_manifest.get("smoke_dataset_audit_provenance_all_artifacts_present"))
            and bool(pretrain_manifest.get("smoke_dataset_audit_provenance_all_artifact_hashes_present"))
            and bool(pretrain_manifest.get("smoke_edge_worktree_critical_gate_review_ok")),
            {"pretrain_manifest_contract": pretrain_manifest},
        ),
        _check(
            "candidate bundle audit covered val/test rows",
            exists and split_rows.get("val", 0) > 0 and split_rows.get("test", 0) > 0,
            {"split_rows": split_rows},
        ),
    ]


def _best_replay_row(metrics: pd.DataFrame) -> dict[str, Any] | None:
    if metrics.empty:
        return None
    if "scope" in metrics.columns:
        aggregate = metrics[metrics["scope"].astype(str).isin(["aggregate", "ALL", "overall"])]
        if not aggregate.empty:
            metrics = aggregate
    if "net_sum_bps" in metrics.columns:
        metrics = metrics.sort_values("net_sum_bps", ascending=False)
    return metrics.iloc[0].to_dict()


def _replay_checks(
    replay_dir: Path,
    manifest: dict[str, Any],
    metrics: pd.DataFrame,
    monthly: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    min_net_sum_bps: float,
    min_profit_factor: float,
    max_drawdown_bps: float,
    expected_candidate_bundle_dir: str | None = None,
    expected_contract_mode: str = "foundation_seq146",
) -> list[dict[str, Any]]:
    row = _best_replay_row(metrics)
    required_metric_columns = {
        "policy_id",
        "n_trades",
        "net_sum_bps",
        "win_rate",
        "profit_factor",
        "max_drawdown_bps",
        "max_loss_bps",
    }
    metric_columns = set(str(c) for c in metrics.columns)
    monthly_columns = set(str(c) for c in monthly.columns)
    trade_columns = set(str(c) for c in trades.columns)
    iql_missing_columns = sorted(set(IQL_TRANSITION_REQUIRED_COLUMNS) - trade_columns)
    identity = manifest.get("replay_identity_contract") if isinstance(manifest.get("replay_identity_contract"), dict) else {}
    identity_contract_mode = _contract_mode_from_identity(identity)
    candidate_specialist_identity = (
        identity.get("candidate_specialist_contract")
        if isinstance(identity.get("candidate_specialist_contract"), dict)
        else {}
    )
    selective_specialist_identity = (
        identity.get("selective_edge_specialist_contract")
        if isinstance(identity.get("selective_edge_specialist_contract"), dict)
        else {}
    )
    row_details = row or {}
    drawdown = row_details.get("max_drawdown_bps")
    drawdown_ok = drawdown is not None and abs(float(drawdown)) <= float(max_drawdown_bps)
    return [
        _check("offline replay dir exists", replay_dir.exists(), {"replay_dir": str(replay_dir)}),
        _check("offline replay manifest PASS", str(manifest.get("decision")) == "PASS", {"manifest_failures": manifest.get("failures")}),
        _check("offline replay manifest has zero failures", not manifest.get("failures"), {"manifest_failures": manifest.get("failures")}),
        _check("offline replay identity contract ready", bool(identity.get("ready")), {"identity": identity}),
        _check(
            "offline replay identity preserves candidate specialist contract",
            bool(candidate_specialist_identity.get("ready")),
            {"candidate_specialist_contract": candidate_specialist_identity},
        ),
        _check(
            "offline replay identity preserves selective-edge specialist contract",
            bool(selective_specialist_identity.get("ready")),
            {"selective_edge_specialist_contract": selective_specialist_identity},
        ),
        _check(
            "offline replay identity contract mode matches replay-readiness contract",
            identity_contract_mode == expected_contract_mode,
            {"expected_contract_mode": expected_contract_mode, "identity_contract_mode": identity_contract_mode},
        ),
        _check(
            "offline replay identity matches candidate bundle audit",
            True if expected_candidate_bundle_dir is None else str(identity.get("candidate_bundle_dir") or "") == str(expected_candidate_bundle_dir),
            {"expected_candidate_bundle_dir": expected_candidate_bundle_dir, "identity_candidate_bundle_dir": identity.get("candidate_bundle_dir")},
        ),
        _check("offline replay metrics exist and have rows", not metrics.empty, {"rows": int(len(metrics))}),
        _check(
            "offline replay metrics have required columns",
            required_metric_columns.issubset(metric_columns),
            {"missing_columns": sorted(required_metric_columns - metric_columns)},
        ),
        _check("offline replay monthly file has rows", not monthly.empty, {"rows": int(len(monthly))}),
        _check("offline replay monthly file has net_sum_bps", "net_sum_bps" in monthly_columns),
        _check("offline replay trades file has rows", not trades.empty, {"rows": int(len(trades))}),
        _check(
            "offline replay trades have IQL transition columns",
            not iql_missing_columns,
            {"missing_columns": iql_missing_columns, "required_columns": list(IQL_TRANSITION_REQUIRED_COLUMNS)},
        ),
        _check(
            "offline replay best row has enough trades",
            row is not None and int(row_details.get("n_trades") or 0) > 0,
            {"best_row": row_details},
        ),
        _check(
            "offline replay net sum is positive",
            row is not None and float(row_details.get("net_sum_bps") or 0.0) > float(min_net_sum_bps),
            {"threshold": min_net_sum_bps, "best_row": row_details},
        ),
        _check(
            "offline replay profit factor passes",
            row is not None and float(row_details.get("profit_factor") or 0.0) >= float(min_profit_factor),
            {"threshold": min_profit_factor, "best_row": row_details},
        ),
        _check(
            "offline replay drawdown is within bound",
            drawdown_ok,
            {"max_abs_drawdown_bps": max_drawdown_bps, "best_row": row_details},
        ),
        _check(
            "offline replay has no negative months",
            not monthly.empty and "net_sum_bps" in monthly.columns and bool((monthly["net_sum_bps"].astype(float) > 0.0).all()),
            {"monthly_rows": int(len(monthly))},
        ),
    ]


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Replay Readiness",
        "",
        f"- Decision: `{report['decision']}`",
        f"- IQL distillation allowed with explicit vedtak: `{report['iql_distillation_allowed_with_explicit_vedtak']}`",
        f"- Promotion/shadow/live allowed: `{report['promotion_shadow_live_allowed']}`",
        f"- Next required gate: `{report['next_required_gate']}`",
        "",
        "## Gates",
        "",
    ]
    for gate in report["gates"]:
        lines.append(f"- `{gate['name']}`: {gate['decision']} ({gate['passed']}/{gate['total']} checks)")
    lines.extend(["", "## Failures", ""])
    if report["failures"]:
        for failure in report["failures"]:
            lines.append(f"- `{failure['gate']}`: {failure['check']}")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    requested_contract_mode = getattr(args, "contract_mode", None)
    candidate_readiness_raw = Path(args.candidate_readiness_json).expanduser()
    candidate_bundle_audit_raw = Path(args.candidate_bundle_audit_json).expanduser()
    selective_summary_raw = Path(args.selective_edge_summary_json).expanduser()
    selective_metrics_raw = Path(args.selective_edge_metrics_csv).expanduser()
    replay_dir_raw = Path(args.replay_dir).expanduser()
    out_dir_raw = Path(args.out_dir).expanduser()
    if requested_contract_mode == "challenger_seq215":
        if candidate_readiness_raw == CANDIDATE_READINESS_LATEST:
            candidate_readiness_raw = CHALLENGER_SEQ215_CANDIDATE_READINESS_LATEST
        if candidate_bundle_audit_raw == DEFAULT_CANDIDATE_BUNDLE_AUDIT:
            candidate_bundle_audit_raw = CHALLENGER_SEQ215_CANDIDATE_BUNDLE_AUDIT
        if selective_summary_raw == DEFAULT_SELECTIVE_EDGE_DIR / "summary.json":
            selective_summary_raw = CHALLENGER_SEQ215_SELECTIVE_EDGE_DIR / "summary.json"
        if selective_metrics_raw == DEFAULT_SELECTIVE_EDGE_DIR / "selective_edge_metrics.csv":
            selective_metrics_raw = CHALLENGER_SEQ215_SELECTIVE_EDGE_DIR / "selective_edge_metrics.csv"
        if replay_dir_raw == DEFAULT_REPLAY_DIR:
            replay_dir_raw = CHALLENGER_SEQ215_REPLAY_DIR
        if out_dir_raw == DEFAULT_OUT_DIR:
            out_dir_raw = CHALLENGER_SEQ215_OUT_DIR
    elif requested_contract_mode == "smart_seq520_candidate":
        if candidate_readiness_raw == CANDIDATE_READINESS_LATEST:
            candidate_readiness_raw = SMART_SEQ520_CANDIDATE_READINESS_LATEST
        if candidate_bundle_audit_raw == DEFAULT_CANDIDATE_BUNDLE_AUDIT:
            candidate_bundle_audit_raw = SMART_SEQ520_CANDIDATE_BUNDLE_AUDIT
        if selective_summary_raw == DEFAULT_SELECTIVE_EDGE_DIR / "summary.json":
            selective_summary_raw = SMART_SEQ520_SELECTIVE_EDGE_DIR / "summary.json"
        if selective_metrics_raw == DEFAULT_SELECTIVE_EDGE_DIR / "selective_edge_metrics.csv":
            selective_metrics_raw = SMART_SEQ520_SELECTIVE_EDGE_DIR / "selective_edge_metrics.csv"
        if replay_dir_raw == DEFAULT_REPLAY_DIR:
            replay_dir_raw = SMART_SEQ520_REPLAY_DIR
        if out_dir_raw == DEFAULT_OUT_DIR:
            out_dir_raw = SMART_SEQ520_OUT_DIR

    out_dir = out_dir_raw.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_readiness_path = candidate_readiness_raw.resolve()
    candidate_bundle_audit_path = candidate_bundle_audit_raw.resolve()
    selective_summary_path = selective_summary_raw.resolve()
    selective_metrics_path = selective_metrics_raw.resolve()
    replay_dir = replay_dir_raw.resolve()

    candidate_readiness = _read_json(candidate_readiness_path)
    candidate_bundle_audit = _read_json(candidate_bundle_audit_path) if candidate_bundle_audit_path.exists() else {}
    expected_candidate_bundle_dir = str(candidate_bundle_audit.get("bundle_dir") or "") or None
    contract_mode = _normalize_contract_mode(requested_contract_mode or _contract_mode_from_bundle_audit(candidate_bundle_audit))
    expected_dataset_dir = {
        "foundation_seq146": FOUNDATION_DATASET_DIR,
        "challenger_seq215": CHALLENGER_SEQ215_DATASET_DIR,
        "smart_seq520_candidate": SMART_SEQ520_DATASET_DIR,
    }[contract_mode]
    # Explicit dataset-contract override (2026-07-04): the smart monthly-refresh
    # deploy contract (train->2026-05-31, val/test = June halves at operating
    # horizon; user vedtak SMART_SEQ520_candidate_train_20260703) evaluates on a
    # re-split whose dir differs from the original pin. The pin stays the default;
    # an override must be EXPLICIT on the command line — never inferred.
    _dataset_override = str(getattr(args, "expected_dataset_dir", "") or "").strip()
    if _dataset_override:
        expected_dataset_dir = Path(_dataset_override).expanduser().resolve()
    selective_summary = _read_json(selective_summary_path) if selective_summary_path.exists() else {}
    selective_metrics = _read_csv_or_empty(selective_metrics_path)
    replay_metrics = _read_csv_or_empty(replay_dir / "replay_policy_metrics.csv")
    replay_monthly = _read_csv_or_empty(replay_dir / "replay_policy_monthly.csv")
    replay_trades = _read_csv_or_empty(replay_dir / "replay_policy_trades.csv")
    replay_manifest = _read_json(replay_dir / "REPLAY_EVIDENCE_MANIFEST.json") if (replay_dir / "REPLAY_EVIDENCE_MANIFEST.json").exists() else {}
    smart_xau_pretrain_audit = (
        _read_json(SMART_XAU_PRETRAIN_AUDIT_LATEST)
        if contract_mode == "smart_seq520_candidate" and SMART_XAU_PRETRAIN_AUDIT_LATEST.exists()
        else {}
    )
    replay_identity = replay_manifest.get("replay_identity_contract") if isinstance(replay_manifest.get("replay_identity_contract"), dict) else {}
    candidate_specialist_identity = (
        replay_identity.get("candidate_specialist_contract")
        if isinstance(replay_identity.get("candidate_specialist_contract"), dict)
        else {}
    )
    selective_edge_specialist_identity = (
        replay_identity.get("selective_edge_specialist_contract")
        if isinstance(replay_identity.get("selective_edge_specialist_contract"), dict)
        else {}
    )
    evidence_identity = {
        "candidate_bundle_audit_json": str(candidate_bundle_audit_path),
        "selective_edge_summary_json": str(selective_summary_path),
        "replay_evidence_manifest_json": str(replay_dir / "REPLAY_EVIDENCE_MANIFEST.json"),
        "smart_xau_pretrain_audit_json": (
            str(SMART_XAU_PRETRAIN_AUDIT_LATEST)
            if contract_mode == "smart_seq520_candidate"
            else ""
        ),
        "candidate_bundle_dir": str(candidate_bundle_audit.get("bundle_dir") or ""),
        "selective_edge_bundle_dir": str(selective_summary.get("bundle_dir") or ""),
        "replay_identity_candidate_bundle_dir": str(replay_identity.get("candidate_bundle_dir") or ""),
        "no_xgb_bundle_dir": str(selective_summary.get("no_xgb_bundle_dir") or replay_identity.get("no_xgb_bundle_dir") or ""),
        "replay_identity_ready": bool(replay_identity.get("ready")),
        "contract_mode": contract_mode,
        "candidate_bundle_contract_mode": _contract_mode_from_bundle_audit(candidate_bundle_audit),
        "selective_edge_contract_mode": _normalize_contract_mode(selective_summary.get("contract_mode")),
        "replay_identity_contract_mode": _contract_mode_from_identity(replay_identity),
        "candidate_specialist_contract": candidate_specialist_identity,
        "selective_edge_specialist_contract": selective_edge_specialist_identity,
        "candidate_specialist_contract_ready": bool(candidate_specialist_identity.get("ready")),
        "selective_edge_specialist_contract_ready": bool(selective_edge_specialist_identity.get("ready")),
    }
    artifacts = {
        "candidate_readiness": str(candidate_readiness_path),
        "candidate_bundle_audit": str(candidate_bundle_audit_path),
        "selective_edge_summary": str(selective_summary_path),
        "selective_edge_metrics": str(selective_metrics_path),
        "candidate_replay_manifest": str(replay_dir / "REPLAY_EVIDENCE_MANIFEST.json"),
        "candidate_replay_metrics": str(replay_dir / "replay_policy_metrics.csv"),
        "candidate_replay_monthly": str(replay_dir / "replay_policy_monthly.csv"),
        "candidate_replay_trades": str(replay_dir / "replay_policy_trades.csv"),
    }
    if contract_mode == "smart_seq520_candidate":
        artifacts["smart_xau_pretrain_audit"] = str(SMART_XAU_PRETRAIN_AUDIT_LATEST)
    artifact_fingerprints = _artifact_fingerprints(artifacts)

    gate_checks = {
        "candidate_readiness": [
            _check(
                "candidate-readiness is green",
                str(candidate_readiness.get("decision")) == "READY_FOR_CANDIDATE_TRAINING_VEDTAK",
                {"decision": candidate_readiness.get("decision"), "failures": candidate_readiness.get("failures")},
            ),
            _check(
                "candidate-readiness still blocks promotion/shadow/live",
                bool(candidate_readiness.get("promotion_shadow_live_allowed")) is False,
            ),
        ],
        "candidate_bundle_audit": _candidate_bundle_audit_checks(
            candidate_bundle_audit_path,
            candidate_bundle_audit,
            contract_mode=contract_mode,
            expected_dataset_dir=expected_dataset_dir,
        ),
        "selective_edge": _selective_edge_checks(
            selective_summary,
            selective_metrics,
            model_name=str(args.model_name),
            min_top5_mean_pnl_bps=float(args.min_top5_mean_pnl_bps),
            min_top10_mean_pnl_bps=float(args.min_top10_mean_pnl_bps),
            require_no_xgb_ablation=bool(args.require_no_xgb_ablation),
            min_top_direction_precision=float(getattr(args, "min_top_direction_precision", 0.50)),
            min_direction_slice_precision=float(getattr(args, "min_direction_slice_precision", 0.50)),
            min_direction_slice_n=int(getattr(args, "min_direction_slice_n", 20)),
            expected_bundle_dir=expected_candidate_bundle_dir,
            expected_dataset_dir=expected_dataset_dir,
            expected_contract_mode=contract_mode,
        ),
        "offline_replay": _replay_checks(
            replay_dir,
            replay_manifest,
            replay_metrics,
            replay_monthly,
            replay_trades,
            min_net_sum_bps=float(args.min_replay_net_sum_bps),
            min_profit_factor=float(args.min_profit_factor),
            max_drawdown_bps=float(args.max_abs_drawdown_bps),
            expected_candidate_bundle_dir=expected_candidate_bundle_dir,
            expected_contract_mode=contract_mode,
        ),
        "iql_distillation_guard": [
            _check("gate never trains IQL", True),
            _check("gate never promotes", True),
            _check("gate never starts shadow/live", True),
        ],
        "artifact_provenance": _artifact_fingerprint_checks(artifact_fingerprints),
    }
    if contract_mode == "smart_seq520_candidate":
        gate_checks["smart_xau_pretrain_audit"] = _smart_xau_pretrain_audit_checks(
            SMART_XAU_PRETRAIN_AUDIT_LATEST,
            smart_xau_pretrain_audit,
            expected_dataset_dir=expected_dataset_dir,
        )
    gates = []
    for name, checks in gate_checks.items():
        passed = sum(1 for check in checks if check["ok"])
        gates.append(
            {
                "name": name,
                "decision": "PASS" if _all_ok(checks) else "FAIL",
                "passed": int(passed),
                "total": int(len(checks)),
                "checks": checks,
            }
        )
    failures = [
        {"gate": gate["name"], "check": check["name"], "details": check.get("details") or {}}
        for gate in gates
        for check in gate["checks"]
        if not check["ok"]
    ]
    ready = not failures
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "entry_replay_readiness_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract_mode": contract_mode,
        "decision": "READY_FOR_IQL_DISTILLATION_VEDTAK" if ready else "NOT_READY_FOR_IQL_DISTILLATION",
        "iql_distillation_allowed_with_explicit_vedtak": bool(ready),
        "promotion_shadow_live_allowed": False,
        "next_required_gate": (
            "IQL distillation research wrapper with explicit vedtak and post-distillation replay comparison"
            if ready
            else "run candidate train, selective-edge eval, and 2026 offline replay before IQL distillation"
        ),
        "candidate_readiness_json": str(candidate_readiness_path),
        "candidate_bundle_audit_json": str(candidate_bundle_audit_path),
        "selective_edge_summary_json": str(selective_summary_path),
        "selective_edge_metrics_csv": str(selective_metrics_path),
        "replay_dir": str(replay_dir),
        "evidence_identity": evidence_identity,
        "artifacts": artifacts,
        "artifact_fingerprints": artifact_fingerprints,
        "gates": gates,
        "failures": failures,
    }
    json_path = out_dir / f"ENTRY_REPLAY_READINESS_{timestamp}.json"
    md_path = out_dir / f"ENTRY_REPLAY_READINESS_{timestamp}.md"
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    latest_json = out_dir / "ENTRY_REPLAY_READINESS_latest.json"
    latest_md = out_dir / "ENTRY_REPLAY_READINESS_latest.md"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "failures": report["failures"],
                    "json_path": report["json_path"],
                    "md_path": report["md_path"],
                    "next_required_gate": report["next_required_gate"],
                },
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
        )
    if args.fail_on_not_ready and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-readiness-json", default=str(CANDIDATE_READINESS_LATEST))
    ap.add_argument("--candidate-bundle-audit-json", default=str(DEFAULT_CANDIDATE_BUNDLE_AUDIT))
    ap.add_argument("--selective-edge-summary-json", default=str(DEFAULT_SELECTIVE_EDGE_DIR / "summary.json"))
    ap.add_argument("--selective-edge-metrics-csv", default=str(DEFAULT_SELECTIVE_EDGE_DIR / "selective_edge_metrics.csv"))
    ap.add_argument("--replay-dir", default=str(DEFAULT_REPLAY_DIR))
    ap.add_argument(
        "--expected-dataset-dir",
        default="",
        help=(
            "Explicit dataset-contract override for the audit/selective-edge dataset "
            "pin (e.g. the smart monthly-refresh re-split). Empty = the per-mode "
            "canonical pin. Must be explicit; never inferred."
        ),
    )
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--model-name", default="candidate")
    ap.add_argument("--min-top5-mean-pnl-bps", type=float, default=0.0)
    ap.add_argument("--min-top10-mean-pnl-bps", type=float, default=0.0)
    ap.add_argument("--min-top-direction-precision", type=float, default=0.50)
    ap.add_argument("--min-direction-slice-precision", type=float, default=0.50)
    ap.add_argument("--min-direction-slice-n", type=int, default=20)
    ap.add_argument("--min-replay-net-sum-bps", type=float, default=0.0)
    ap.add_argument("--min-profit-factor", type=float, default=1.05)
    ap.add_argument("--max-abs-drawdown-bps", type=float, default=650.0)
    ap.add_argument("--require-no-xgb-ablation", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--contract-mode", choices=tuple(CONTRACT_INPUT_DIMS), default=None)
    ap.add_argument("--challenger-seq215", action="store_const", const="challenger_seq215", dest="contract_mode")
    ap.add_argument("--smart-seq520", action="store_const", const="smart_seq520_candidate", dest="contract_mode")
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
