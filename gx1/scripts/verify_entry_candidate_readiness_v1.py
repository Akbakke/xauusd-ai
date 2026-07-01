#!/usr/bin/env python3
"""Verify Entry readiness for full candidate training after smoke edge evidence.

This gate is intentionally stricter than train-readiness. It does not train. It
requires a real smoke-train bundle audit with edge diagnostics before any full
candidate-training vedtak should be considered.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gx1.features.entry_specialist_feature_groups_v1 import (
    SPECIALIST_CONTRACT_MODES,
    required_training_specialists_for_mode,
    specialist_contract_training_allowed_for_mode,
)
from gx1.scripts.verify_entry_foundation_state_v1 import (
    FOUNDATION_SMOKE_DATASET_DIR,
    REPORTS_ROOT,
    REPO,
    SPECIALIST_AUDIT_LATEST,
)
from gx1.scripts.verify_entry_training_readiness_v1 import (
    DEFAULT_OUT_DIR as TRAIN_READINESS_OUT_DIR,
    EXPECTED_ACTIVE_TRAINING_HEADS,
    EXPECTED_BLOCKED_HEADS,
    SMOKE_BUNDLE_AUDIT_LATEST,
    _artifact_fingerprint_checks,
    _artifact_fingerprints,
    _check,
)
from gx1.scripts.verify_entry_training_readiness_v1 import run as run_train_readiness


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_candidate_readiness_20260628_v1"
CHALLENGER_SEQ215_OUT_DIR = DEFAULT_OUT_DIR / "challenger_seq215_20260630"
CHALLENGER_SEQ215_SMOKE_BUNDLE_AUDIT_LATEST = (
    REPORTS_ROOT
    / "entry_foundation_smoke_bundle_audit_20260628_v1"
    / "challenger_seq215_20260630"
    / "ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT_latest.json"
)
CHALLENGER_SEQ215_SMOKE_DATASET_DIR = (
    Path("/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605")
    / "v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_challenger_seq215_smoke_20260630"
)
CHALLENGER_SEQ215_SPECIALIST_AUDIT_LATEST = (
    REPORTS_ROOT
    / "entry_specialist_feature_group_audit_20260628_v1"
    / "challenger_seq215_20260630_contract8"
    / "ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.json"
)
SMART_SEQ520_OUT_DIR = DEFAULT_OUT_DIR / "smart_seq520_candidate"
SMART_SEQ520_SMOKE_BUNDLE_AUDIT_LATEST = (
    REPORTS_ROOT
    / "entry_foundation_smoke_bundle_audit_20260628_v1"
    / "smart_seq520_candidate"
    / "ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT_latest.json"
)
SMART_SEQ520_SMOKE_DATASET_DIR = (
    Path("/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605")
    / "v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_smart_seq520_smoke_20260630"
)
SMART_SEQ520_SPECIALIST_AUDIT_LATEST = (
    REPORTS_ROOT
    / "entry_specialist_feature_group_audit_20260628_v1"
    / "smart_seq520_candidate_20260630"
    / "ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.json"
)
SMART_SEQ520_TRAINABILITY_READINESS_LATEST = (
    REPORTS_ROOT
    / "entry_smart_seq520_trainability_readiness_20260630_v1"
    / "ENTRY_SMART_SEQ520_TRAINABILITY_READINESS_latest.json"
)
SMART_SEQ520_TRAINABILITY_READY_DECISION = "READY_FOR_SMART_SEQ520_TRAINABILITY_REVIEW"
REQUIRED_SPECIALIST_GROUPS = tuple(required_training_specialists_for_mode("foundation_seq146"))
REQUIRED_MIN_GATE_ENTROPY = 0.05
EXPECTED_SIGNAL_DIMS_BY_MODE = {
    "foundation_seq146": 146,
    "challenger_seq215": 215,
    "smart_seq520_candidate": 520,
}
CLASS_NAMES = ("LONG", "SHORT", "FLAT")
MIN_DIRECTION_CLASS_LABEL_RATE_FOR_COVERAGE = 0.10
MIN_DIRECTION_CLASS_PRED_RATE = 0.05
MIN_DIRECTION_CLASS_PRED_TO_LABEL_RATIO = 0.35


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


def _all_ok(checks: list[dict[str, Any]]) -> bool:
    return all(bool(check.get("ok")) for check in checks)


def _splits(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(split): row
        for split, row in (report.get("splits") or {}).items()
        if isinstance(row, dict)
    }


def _split_names(report: dict[str, Any]) -> list[str]:
    raw = report.get("data_splits")
    if isinstance(raw, list) and raw:
        return [str(x) for x in raw]
    return sorted(_splits(report))


def _all_split_direction_beats_majority(report: dict[str, Any]) -> bool:
    rows = _splits(report)
    names = _split_names(report)
    return bool(names) and all(bool(((rows.get(split) or {}).get("direction") or {}).get("beats_majority_baseline")) for split in names)


def _direction_distribution_contract(direction: dict[str, Any]) -> dict[str, Any]:
    rows = int(direction.get("rows") or 0)
    label_counts = direction.get("label_counts") if isinstance(direction.get("label_counts"), dict) else {}
    prediction_counts = (
        direction.get("prediction_counts") if isinstance(direction.get("prediction_counts"), dict) else {}
    )
    failures: list[str] = []
    class_rows: dict[str, dict[str, float | int]] = {}
    for name in CLASS_NAMES:
        label_count = int(label_counts.get(name) or 0)
        prediction_count = int(prediction_counts.get(name) or 0)
        label_rate = float(label_count) / float(rows) if rows > 0 else 0.0
        prediction_rate = float(prediction_count) / float(rows) if rows > 0 else 0.0
        required_prediction_rate = 0.0
        if label_rate >= MIN_DIRECTION_CLASS_LABEL_RATE_FOR_COVERAGE:
            required_prediction_rate = max(
                MIN_DIRECTION_CLASS_PRED_RATE,
                label_rate * MIN_DIRECTION_CLASS_PRED_TO_LABEL_RATIO,
            )
            if prediction_rate < required_prediction_rate:
                failures.append(
                    f"{name} prediction_rate={prediction_rate:.6f} below required "
                    f"{required_prediction_rate:.6f} for label_rate={label_rate:.6f}"
                )
        class_rows[name] = {
            "label_count": label_count,
            "prediction_count": prediction_count,
            "label_rate": label_rate,
            "prediction_rate": prediction_rate,
            "prediction_to_label_rate": prediction_rate / max(label_rate, 1e-12),
            "required_prediction_rate": required_prediction_rate,
        }
    if rows <= 0:
        failures.append("direction distribution has zero rows")
    if rows > 0 and not label_counts:
        failures.append("direction distribution missing label_counts")
    if rows > 0 and not prediction_counts:
        failures.append("direction distribution missing prediction_counts")
    return {
        "decision": "PASS" if not failures else "FAIL",
        "classes": class_rows,
        "failures": failures,
    }


def _split_direction_distribution_contract(split_row: dict[str, Any]) -> dict[str, Any]:
    contract = split_row.get("direction_distribution_contract")
    if isinstance(contract, dict):
        return contract
    direction = split_row.get("direction") if isinstance(split_row.get("direction"), dict) else {}
    return _direction_distribution_contract(direction)


def _direction_split_summary(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = _splits(report)
    out: dict[str, dict[str, Any]] = {}
    for split in ("val", "test"):
        split_row = rows.get(split) or {}
        direction = split_row.get("direction") if isinstance(split_row.get("direction"), dict) else {}
        out[split] = {
            "accuracy": direction.get("accuracy"),
            "label_counts": direction.get("label_counts") if isinstance(direction.get("label_counts"), dict) else {},
            "prediction_counts": (
                direction.get("prediction_counts") if isinstance(direction.get("prediction_counts"), dict) else {}
            ),
        }
    return out


def _class_rate(counts: dict[str, Any], class_name: str) -> float | None:
    total = sum(int(counts.get(name) or 0) for name in CLASS_NAMES)
    if total <= 0:
        return None
    return float(int(counts.get(class_name) or 0)) / float(total)


def _class_balance_drift(direction: dict[str, Any], class_name: str) -> float | None:
    label_counts = direction.get("label_counts") if isinstance(direction.get("label_counts"), dict) else {}
    prediction_counts = (
        direction.get("prediction_counts") if isinstance(direction.get("prediction_counts"), dict) else {}
    )
    label_rate = _class_rate(label_counts, class_name)
    prediction_rate = _class_rate(prediction_counts, class_name)
    if label_rate is None or prediction_rate is None:
        return None
    return prediction_rate - label_rate


def _smart_smoke_benchmark_checks(
    smart_report: dict[str, Any],
    *,
    foundation_report: dict[str, Any],
    seq215_report: dict[str, Any],
) -> list[dict[str, Any]]:
    smart = _direction_split_summary(smart_report)
    baselines = {
        "foundation_seq146": _direction_split_summary(foundation_report),
        "challenger_seq215": _direction_split_summary(seq215_report),
    }
    missing_metrics: list[dict[str, str]] = []
    accuracy_regressions: list[dict[str, Any]] = []
    balance_regressions: list[dict[str, Any]] = []
    for split in ("val", "test"):
        smart_accuracy = smart.get(split, {}).get("accuracy")
        if smart_accuracy is None:
            missing_metrics.append({"model": "smart_seq520_candidate", "split": split, "metric": "accuracy"})
        for baseline_name, baseline in baselines.items():
            baseline_accuracy = baseline.get(split, {}).get("accuracy")
            if baseline_accuracy is None:
                missing_metrics.append({"model": baseline_name, "split": split, "metric": "accuracy"})
                continue
            if smart_accuracy is not None and float(smart_accuracy) < float(baseline_accuracy):
                accuracy_regressions.append(
                    {
                        "baseline": baseline_name,
                        "split": split,
                        "smart_accuracy": float(smart_accuracy),
                        "baseline_accuracy": float(baseline_accuracy),
                        "delta": float(smart_accuracy) - float(baseline_accuracy),
                    }
                )
            for class_name in CLASS_NAMES:
                smart_drift = _class_balance_drift(smart.get(split, {}), class_name)
                baseline_drift = _class_balance_drift(baseline.get(split, {}), class_name)
                if smart_drift is None:
                    missing_metrics.append(
                        {"model": "smart_seq520_candidate", "split": split, "metric": f"{class_name}_drift"}
                    )
                    continue
                if baseline_drift is None:
                    missing_metrics.append(
                        {"model": baseline_name, "split": split, "metric": f"{class_name}_drift"}
                    )
                    continue
                worse_by = abs(float(smart_drift)) - abs(float(baseline_drift))
                if worse_by > 1e-12:
                    balance_regressions.append(
                        {
                            "baseline": baseline_name,
                            "split": split,
                            "class": class_name,
                            "smart_prediction_minus_label_rate": float(smart_drift),
                            "baseline_prediction_minus_label_rate": float(baseline_drift),
                            "worse_by": float(worse_by),
                        }
                    )
    return [
        _check(
            "smart smoke direction benchmark has foundation/seq215 metrics",
            not missing_metrics,
            {"missing_metrics": missing_metrics},
        ),
        _check(
            "smart smoke direction accuracy does not regress versus foundation/seq215",
            not accuracy_regressions,
            {"accuracy_regressions": accuracy_regressions},
        ),
        _check(
            "smart smoke class-balance drift does not regress versus foundation/seq215",
            not balance_regressions,
            {"class_balance_regressions": balance_regressions[:12]},
        ),
    ]


def _all_split_direction_distribution_live(report: dict[str, Any]) -> bool:
    rows = _splits(report)
    names = _split_names(report)
    return bool(names) and all(
        str(_split_direction_distribution_contract(rows.get(split) or {}).get("decision")) == "PASS"
        for split in names
    )


def _split_direction_slice_contract(split_row: dict[str, Any]) -> dict[str, Any]:
    contract = split_row.get("direction_slice_contract")
    return contract if isinstance(contract, dict) else {}


def _all_split_direction_slices_live(report: dict[str, Any]) -> bool:
    rows = _splits(report)
    names = _split_names(report)
    if not names:
        return False
    for split in names:
        contract = _split_direction_slice_contract(rows.get(split) or {})
        if str(contract.get("decision")) != "PASS":
            return False
        if int(contract.get("audited_slice_count") or 0) <= 0:
            return False
    return True


def _all_split_bad_path_negative(report: dict[str, Any]) -> bool:
    rows = _splits(report)
    names = _split_names(report)
    values = [
        (((rows.get(split) or {}).get("bad_path") or {}).get("prob_vs_path_quality_spearman"))
        for split in names
    ]
    return bool(values) and all(value is not None and float(value) < 0.0 for value in values)


def _all_split_gate_live(report: dict[str, Any], *, min_active_specialists: int, min_gate_entropy: float) -> bool:
    rows = _splits(report)
    names = _split_names(report)
    if not names:
        return False
    for split in names:
        gate = ((rows.get(split) or {}).get("specialist_gate") or {})
        if not bool(gate.get("finite")):
            return False
        if float(gate.get("row_sum_max_abs_error") or 999.0) > 1e-4:
            return False
        if int(gate.get("active_specialist_count_gt_1pct") or 0) < int(min_active_specialists):
            return False
        entropy_mean = gate.get("entropy_mean")
        if entropy_mean is None or float(entropy_mean) < float(min_gate_entropy):
            return False
    return True


def _all_required_split_gate_weights_live(
    report: dict[str, Any],
    *,
    required_specialist_groups: tuple[str, ...],
    min_mean_weight: float = 0.01,
) -> bool:
    rows = _splits(report)
    names = _split_names(report)
    if not names:
        return False
    for split in names:
        gate = ((rows.get(split) or {}).get("specialist_gate") or {})
        mean_weight = gate.get("mean_weight") if isinstance(gate.get("mean_weight"), dict) else {}
        for group in required_specialist_groups:
            if float(mean_weight.get(group) or 0.0) <= float(min_mean_weight):
                return False
    return True


def _specialist_audit_contract_passes(
    report: dict[str, Any],
    *,
    required_specialist_groups: tuple[str, ...],
    min_active_specialists: int,
    min_gate_entropy: float,
) -> bool:
    required = {str(x) for x in report.get("required_training_specialists", []) if str(x)}
    return (
        bool(report.get("require_specialist_fusion"))
        and required == set(required_specialist_groups)
        and int(report.get("min_active_specialists") or 0) >= int(min_active_specialists)
        and float(report.get("min_gate_entropy") or -1.0) >= float(min_gate_entropy)
    )


def _head_contract_passes(report: dict[str, Any]) -> bool:
    contract = report.get("head_contract") if isinstance(report.get("head_contract"), dict) else {}
    active = set(str(x) for x in contract.get("active_training_heads", []) if str(x))
    blocked = set(str(x) for x in contract.get("blocked_heads", []) if str(x))
    return (
        bool(report.get("require_head_contract"))
        and str(contract.get("decision")) == "PASS"
        and not contract.get("failures")
        and active == set(EXPECTED_ACTIVE_TRAINING_HEADS)
        and blocked == set(EXPECTED_BLOCKED_HEADS)
    )


def _bundle_specialist_model_contract_passes(report: dict[str, Any]) -> bool:
    bundle = report.get("bundle_summary") if isinstance(report.get("bundle_summary"), dict) else {}
    contract = (
        report.get("bundle_specialist_model_contract")
        if isinstance(report.get("bundle_specialist_model_contract"), dict)
        else {}
    )
    return (
        bool(bundle.get("specialist_model_contract_declared_valid"))
        and bool(bundle.get("specialist_model_contract_valid"))
        and bool(bundle.get("specialist_model_contract_set_exact"))
        and bool(bundle.get("specialist_model_contract_owned_objectives_match"))
        and bool(bundle.get("specialist_model_contract_support_heads_match"))
        and bool(bundle.get("specialist_model_contract_signal_families_match"))
        and bool(bundle.get("specialist_model_contract_model_roles_match"))
        and str(contract.get("decision")) == "PASS"
        and bool(contract.get("valid"))
        and bool(contract.get("set_exact"))
        and bool(contract.get("owned_objectives_match"))
        and bool(contract.get("support_heads_match"))
        and bool(contract.get("signal_families_match"))
        and bool(contract.get("model_roles_match"))
        and not contract.get("failures")
    )


def _mode_smoke_dataset_dir(contract_mode: str) -> Path:
    if contract_mode == "foundation_seq146":
        return FOUNDATION_SMOKE_DATASET_DIR
    if contract_mode == "challenger_seq215":
        return CHALLENGER_SEQ215_SMOKE_DATASET_DIR
    if contract_mode == "smart_seq520_candidate":
        return SMART_SEQ520_SMOKE_DATASET_DIR
    raise ValueError(f"unknown specialist contract mode: {contract_mode}")


def _mode_specialist_audit_path(contract_mode: str) -> Path:
    if contract_mode == "foundation_seq146":
        return SPECIALIST_AUDIT_LATEST
    if contract_mode == "challenger_seq215":
        return CHALLENGER_SEQ215_SPECIALIST_AUDIT_LATEST
    if contract_mode == "smart_seq520_candidate":
        return SMART_SEQ520_SPECIALIST_AUDIT_LATEST
    raise ValueError(f"unknown specialist contract mode: {contract_mode}")


def _mode_smoke_bundle_audit_path(contract_mode: str) -> Path:
    if contract_mode == "foundation_seq146":
        return SMOKE_BUNDLE_AUDIT_LATEST
    if contract_mode == "challenger_seq215":
        return CHALLENGER_SEQ215_SMOKE_BUNDLE_AUDIT_LATEST
    if contract_mode == "smart_seq520_candidate":
        return SMART_SEQ520_SMOKE_BUNDLE_AUDIT_LATEST
    raise ValueError(f"unknown specialist contract mode: {contract_mode}")


def _mode_out_dir(contract_mode: str) -> Path:
    if contract_mode == "foundation_seq146":
        return DEFAULT_OUT_DIR
    if contract_mode == "challenger_seq215":
        return CHALLENGER_SEQ215_OUT_DIR
    if contract_mode == "smart_seq520_candidate":
        return SMART_SEQ520_OUT_DIR
    raise ValueError(f"unknown specialist contract mode: {contract_mode}")


def _mode_smoke_train_command(contract_mode: str) -> str:
    if contract_mode == "foundation_seq146":
        return "scripts/entry_next_edge_control.sh smoke-train --vedtak <id> --require-edge-audit"
    if contract_mode == "challenger_seq215":
        return "scripts/entry_next_edge_control.sh smoke-train-seq215 --vedtak <id> --require-edge-audit"
    if contract_mode == "smart_seq520_candidate":
        return "scripts/entry_next_edge_control.sh smart-smoke-train --vedtak <id> --require-edge-audit"
    raise ValueError(f"unknown specialist contract mode: {contract_mode}")


def _mode_candidate_train_command(contract_mode: str) -> str:
    if contract_mode == "foundation_seq146":
        return "scripts/entry_next_edge_control.sh candidate-train --vedtak <id>"
    if contract_mode == "challenger_seq215":
        return "scripts/entry_next_edge_control.sh candidate-train-seq215 --vedtak <id>"
    if contract_mode == "smart_seq520_candidate":
        return "scripts/entry_next_edge_control.sh candidate-train-smart --vedtak <id>"
    raise ValueError(f"unknown specialist contract mode: {contract_mode}")


def _observed_contract_mode(report: dict[str, Any]) -> str:
    bundle = report.get("bundle_summary") if isinstance(report.get("bundle_summary"), dict) else {}
    pretrain = (
        report.get("pretrain_manifest_contract")
        if isinstance(report.get("pretrain_manifest_contract"), dict)
        else {}
    )
    for raw in (
        report.get("specialist_contract_mode"),
        report.get("contract_mode"),
        bundle.get("specialist_contract_mode"),
        bundle.get("contract_mode"),
        pretrain.get("specialist_contract_mode"),
        pretrain.get("contract_mode"),
    ):
        normalized = str(raw or "").strip()
        if normalized:
            return normalized
    return ""


def _smoke_audit_contract_mode_passes(report: dict[str, Any], *, contract_mode: str) -> bool:
    observed = _observed_contract_mode(report)
    if contract_mode == "foundation_seq146":
        return observed in {"", "foundation_seq146"}
    return observed == contract_mode


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _path_calibration_contract_passes(report: dict[str, Any]) -> bool:
    contract = report.get("path_calibration_recipe_contract")
    if not isinstance(contract, dict):
        return False
    path_quantile = _float_or_zero(contract.get("path_quality_rank_quantile"))
    bad_quantile = _float_or_zero(contract.get("bad_path_quality_rank_quantile"))
    return (
        str(contract.get("decision")) == "PASS"
        and not contract.get("failures")
        and bool(contract.get("path_quality_active"))
        and bool(contract.get("bad_path_active"))
        and bool(contract.get("path_quality_rank_full_batch"))
        and _float_or_zero(contract.get("path_quality_rank_weight")) > 0.0
        and _float_or_zero(contract.get("path_quality_rank_margin")) > 0.0
        and 0.05 <= path_quantile <= 0.45
        and _float_or_zero(contract.get("bad_path_quality_rank_weight")) > 0.0
        and _float_or_zero(contract.get("bad_path_quality_rank_margin")) > 0.0
        and 0.05 <= bad_quantile <= 0.45
    )


SMART_DIRECTION_BALANCE_MIN_ALPHA = 0.20
SMART_DIRECTION_BALANCE_CLASS_WEIGHTS = [1.0, 1.0, 4.0]
SMART_DIRECTION_CKPT_BALANCE_GUARD_MIN_WEIGHT = 0.50
SMART_DIRECTION_CKPT_BALANCE_MIN_PRED_TO_LABEL = 0.35
SMART_DIRECTION_CKPT_BALANCE_MIN_PRED_RATE = 0.05


def _direction_balance_contract_passes(
    report: dict[str, Any],
    *,
    contract_mode: str = "foundation_seq146",
) -> bool:
    contract = report.get("direction_balance_recipe_contract")
    if not isinstance(contract, dict):
        return False
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
        and _float_or_zero(contract.get("ckpt_class_balance_guard_weight"))
        >= SMART_DIRECTION_CKPT_BALANCE_GUARD_MIN_WEIGHT
        and _float_or_zero(contract.get("ckpt_class_balance_min_pred_to_label"))
        >= SMART_DIRECTION_CKPT_BALANCE_MIN_PRED_TO_LABEL
        and _float_or_zero(contract.get("ckpt_class_balance_min_pred_rate"))
        >= SMART_DIRECTION_CKPT_BALANCE_MIN_PRED_RATE
    )


def _tail_direction_contract_passes(report: dict[str, Any]) -> bool:
    contract = report.get("tail_direction_recipe_contract")
    if not isinstance(contract, dict):
        return False
    weight = _float_or_zero(contract.get("tail_direction_ce_weight"))
    quantile = _float_or_zero(contract.get("tail_direction_quality_quantile"))
    min_batch = int(_float_or_zero(contract.get("tail_direction_min_batch")))
    return (
        str(contract.get("decision")) == "PASS"
        and not contract.get("failures")
        and bool(contract.get("direction_active"))
        and weight > 0.0
        and 0.50 <= quantile <= 0.95
        and min_batch >= 2
        and str(contract.get("tail_direction_mask") or "") == "directional_tradable_clean_path_top_quality"
    )


def _symmetric_validation_contract_passes(
    report: dict[str, Any],
    *,
    contract_mode: str = "foundation_seq146",
) -> bool:
    if contract_mode != "smart_seq520_candidate":
        return True
    contract = report.get("symmetric_validation_recipe_contract")
    if not isinstance(contract, dict):
        return False
    active_heads = {str(head) for head in (contract.get("active_heads") or []) if str(head)}
    base_ok = (
        str(contract.get("decision")) == "PASS"
        and not contract.get("failures")
        and bool(contract.get("symmetric_negatives"))
        and bool(contract.get("selector_masked_aux"))
        and bool(contract.get("validation_objective_matches_train"))
        and str(contract.get("aux_selector_mode") or "") == "long_short_union"
        and str(contract.get("clean_edge_target_mode") or "") == "bidir"
        and str(contract.get("survival_target_mode") or "") == "bidir"
        and bool(contract.get("symmetric_short_prob_penalties"))
    )
    if not base_ok:
        return False
    if "bad_path" in active_heads and not (
        bool(contract.get("bad_path_ce_in_direction_loss"))
        and bool(contract.get("bad_path_prob_penalty_in_validation"))
    ):
        return False
    if "clean_edge" in active_heads and not bool(contract.get("symmetric_clean_edge_rank")):
        return False
    return True


def _smoke_edge_checks(
    report: dict[str, Any],
    *,
    contract_mode: str = "foundation_seq146",
    expected_smoke_dataset_dir: str | Path | None = None,
    min_active_specialists: int = 3,
) -> list[dict[str, Any]]:
    normalized_contract_mode = str(contract_mode or "foundation_seq146").strip()
    required_specialist_groups = tuple(required_training_specialists_for_mode(normalized_contract_mode))
    expected_signal_dim = int(EXPECTED_SIGNAL_DIMS_BY_MODE[normalized_contract_mode])
    expected_dataset_dir = Path(expected_smoke_dataset_dir or _mode_smoke_dataset_dir(normalized_contract_mode))
    effective_min_active_specialists = max(int(min_active_specialists), len(required_specialist_groups))
    bundle = report.get("bundle_summary") if isinstance(report.get("bundle_summary"), dict) else {}
    split_rows = {split: int((row or {}).get("rows") or 0) for split, row in _splits(report).items()}
    specialist_groups = set(str(x) for x in bundle.get("specialist_groups", []) if str(x))
    bad_path_rho = {
        split: (((row or {}).get("bad_path") or {}).get("prob_vs_path_quality_spearman"))
        for split, row in _splits(report).items()
    }
    direction = {
        split: {
            "accuracy": ((row or {}).get("direction") or {}).get("accuracy"),
            "majority_baseline_accuracy": ((row or {}).get("direction") or {}).get("majority_baseline_accuracy"),
            "beats_majority_baseline": ((row or {}).get("direction") or {}).get("beats_majority_baseline"),
        }
        for split, row in _splits(report).items()
    }
    direction_distribution = {
        split: _split_direction_distribution_contract(row or {})
        for split, row in _splits(report).items()
    }
    direction_slices = {
        split: _split_direction_slice_contract(row or {})
        for split, row in _splits(report).items()
    }
    gate = {
        split: {
            "row_sum_max_abs_error": (((row or {}).get("specialist_gate") or {}).get("row_sum_max_abs_error")),
            "active_specialist_count_gt_1pct": (((row or {}).get("specialist_gate") or {}).get("active_specialist_count_gt_1pct")),
            "entropy_mean": (((row or {}).get("specialist_gate") or {}).get("entropy_mean")),
            "finite": (((row or {}).get("specialist_gate") or {}).get("finite")),
            "mean_weight": (((row or {}).get("specialist_gate") or {}).get("mean_weight")),
        }
        for split, row in _splits(report).items()
    }
    head_contract = report.get("head_contract") if isinstance(report.get("head_contract"), dict) else {}
    pretrain_manifest = (
        report.get("pretrain_manifest_contract")
        if isinstance(report.get("pretrain_manifest_contract"), dict)
        else {}
    )
    active_heads = set(str(x) for x in head_contract.get("active_training_heads", []) if str(x))
    blocked_heads = set(str(x) for x in head_contract.get("blocked_heads", []) if str(x))
    specialist_contract = {
        "require_specialist_fusion": report.get("require_specialist_fusion"),
        "required_training_specialists": report.get("required_training_specialists"),
        "min_active_specialists": report.get("min_active_specialists"),
        "min_gate_entropy": report.get("min_gate_entropy"),
    }
    return [
        _check("smoke bundle audit PASS", str(report.get("decision")) == "PASS", {"failures": report.get("failures")}),
        _check("smoke bundle audit has zero failures", not report.get("failures"), {"failures": report.get("failures")}),
        _check("smoke bundle audit used smoke dataset", str(report.get("dataset_dir")) == str(expected_dataset_dir)),
        _check(
            f"smoke bundle audit specialist contract mode is {normalized_contract_mode}",
            _smoke_audit_contract_mode_passes(report, contract_mode=normalized_contract_mode),
            {"observed_contract_mode": _observed_contract_mode(report) or None},
        ),
        _check("smoke bundle audit is from actual train output, not sanity bundle", not bool(bundle.get("sanity_bundle"))),
        _check("smoke bundle audit was run with require_edge", bool(report.get("require_edge"))),
        _check("smoke bundle audit was run with require_head_contract", bool(report.get("require_head_contract"))),
        _check(
            "smoke bundle audit path calibration recipe contract PASS",
            _path_calibration_contract_passes(report),
            {"path_calibration_recipe_contract": report.get("path_calibration_recipe_contract")},
        ),
        _check(
            "smoke bundle audit direction balance recipe contract PASS",
            _direction_balance_contract_passes(report, contract_mode=normalized_contract_mode),
            {"direction_balance_recipe_contract": report.get("direction_balance_recipe_contract")},
        ),
        _check(
            "smoke bundle audit tail direction recipe contract PASS",
            _tail_direction_contract_passes(report),
            {"tail_direction_recipe_contract": report.get("tail_direction_recipe_contract")},
        ),
        _check(
            "smoke bundle audit symmetric validation recipe contract PASS",
            _symmetric_validation_contract_passes(report, contract_mode=normalized_contract_mode),
            {"symmetric_validation_recipe_contract": report.get("symmetric_validation_recipe_contract")},
        ),
        _check(
            "smoke bundle audit validated pre-train manifest provenance",
            str(pretrain_manifest.get("decision")) == "PASS"
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
            and bool(pretrain_manifest.get("smoke_dataset_audit_provenance_all_artifacts_present"))
            and bool(pretrain_manifest.get("smoke_dataset_audit_provenance_all_artifact_hashes_present"))
            and bool(pretrain_manifest.get("worktree_critical_gate_review_ok")),
            {"pretrain_manifest_contract": pretrain_manifest},
        ),
        _check(
            "smoke bundle head contract PASS",
            _head_contract_passes(report),
            {
                "head_contract": head_contract,
                "expected_active_heads": list(EXPECTED_ACTIVE_TRAINING_HEADS),
                "actual_active_heads": sorted(active_heads),
                "expected_blocked_heads": list(EXPECTED_BLOCKED_HEADS),
                "actual_blocked_heads": sorted(blocked_heads),
            },
        ),
        _check(
            "smoke bundle signal dims match contract",
            int(bundle.get("seq_input_dim") or 0) == expected_signal_dim
            and int(bundle.get("snap_input_dim") or 0) == expected_signal_dim,
            {
                "expected_signal_dim": expected_signal_dim,
                "seq_input_dim": bundle.get("seq_input_dim"),
                "snap_input_dim": bundle.get("snap_input_dim"),
            },
        ),
        _check("smoke bundle has multi-TF enabled", bool(bundle.get("multi_tf_enabled"))),
        _check("smoke bundle has specialist fusion", bool(bundle.get("specialist_fusion_enabled"))),
        _check(
            "smoke bundle specialist model contract is preserved in bundle metadata",
            _bundle_specialist_model_contract_passes(report),
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
            "smoke bundle audit was run with specialist-fusion gate contract",
            _specialist_audit_contract_passes(
                report,
                required_specialist_groups=required_specialist_groups,
                min_active_specialists=effective_min_active_specialists,
                min_gate_entropy=REQUIRED_MIN_GATE_ENTROPY,
            ),
            {
                "required_min_active_specialists": effective_min_active_specialists,
                "required_min_gate_entropy": REQUIRED_MIN_GATE_ENTROPY,
                "specialist_contract": specialist_contract,
            },
        ),
        _check(
            "smoke bundle includes required specialist groups",
            all(group in specialist_groups for group in required_specialist_groups),
            {"specialist_groups": sorted(specialist_groups)},
        ),
        _check(
            "smoke bundle has exact specialist groups",
            specialist_groups == set(required_specialist_groups),
            {
                "expected_specialist_groups": list(required_specialist_groups),
                "actual_specialist_groups": sorted(specialist_groups),
            },
        ),
        _check(
            "smoke audit covered val/test rows",
            split_rows.get("val", 0) > 0 and split_rows.get("test", 0) > 0,
            {"split_rows": split_rows},
        ),
        _check("direction beats majority on all audited splits", _all_split_direction_beats_majority(report), direction),
        _check(
            "direction distribution covers active LONG/SHORT/FLAT classes",
            _all_split_direction_distribution_live(report),
            {"direction_distribution": direction_distribution},
        ),
        _check(
            "direction context slices pass session/regime bucket diagnostics",
            _all_split_direction_slices_live(report),
            {"direction_slices": direction_slices},
        ),
        _check("bad_path probability ranks worse path quality higher", _all_split_bad_path_negative(report), {"bad_path_rho": bad_path_rho}),
        _check(
            "specialist gate is finite, normalized, non-collapsed, and entropic",
            _all_split_gate_live(
                report,
                min_active_specialists=effective_min_active_specialists,
                min_gate_entropy=REQUIRED_MIN_GATE_ENTROPY,
            ),
            {
                "min_active_specialists": effective_min_active_specialists,
                "min_gate_entropy": REQUIRED_MIN_GATE_ENTROPY,
                "gate": gate,
            },
        ),
        _check(
            "each required specialist has non-collapsed gate weight",
            _all_required_split_gate_weights_live(
                report,
                required_specialist_groups=required_specialist_groups,
                min_mean_weight=0.01,
            ),
            {"min_mean_weight": 0.01, "gate": gate},
        ),
    ]


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Candidate Readiness",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Candidate training allowed with explicit vedtak: `{report['candidate_training_allowed_with_explicit_vedtak']}`",
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
    contract_mode = str(getattr(args, "contract_mode", "foundation_seq146") or "foundation_seq146").strip()
    expected_smoke_dataset_dir = _mode_smoke_dataset_dir(contract_mode)
    expected_signal_dim = int(EXPECTED_SIGNAL_DIMS_BY_MODE[contract_mode])
    required_specialist_groups = tuple(required_training_specialists_for_mode(contract_mode))
    out_dir = Path(getattr(args, "out_dir", None) or _mode_out_dir(contract_mode)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke_audit_path = Path(
        getattr(args, "smoke_bundle_audit_json", None) or _mode_smoke_bundle_audit_path(contract_mode)
    ).expanduser().resolve()
    specialist_audit_path = Path(
        getattr(args, "specialist_audit_json", None) or _mode_specialist_audit_path(contract_mode)
    ).expanduser().resolve()

    upstream_gate_name = "train_readiness"
    upstream_artifact_path = TRAIN_READINESS_OUT_DIR / "ENTRY_TRAINING_READINESS_latest.json"
    if contract_mode == "smart_seq520_candidate":
        upstream_gate_name = "smart_trainability_readiness"
        upstream_artifact_path = SMART_SEQ520_TRAINABILITY_READINESS_LATEST
        try:
            upstream_readiness = _read_json(upstream_artifact_path)
            upstream_load_error = None
        except Exception as exc:
            upstream_readiness = {}
            upstream_load_error = str(exc)
        upstream_checks = [
            _check(
                "smart trainability readiness exists and is readable",
                upstream_load_error is None,
                {"path": str(upstream_artifact_path), "error": upstream_load_error},
            ),
            _check(
                "smart trainability readiness is green",
                str(upstream_readiness.get("decision")) == SMART_SEQ520_TRAINABILITY_READY_DECISION,
                {"decision": upstream_readiness.get("decision"), "failures": upstream_readiness.get("failures")},
            ),
            _check("smart trainability gate still blocks direct execution", bool(upstream_readiness.get("execution_allowed_now")) is False),
            _check("smart trainability gate still blocks training", bool(upstream_readiness.get("training_allowed")) is False),
            _check(
                "smart trainability gate still blocks replay IQL shadow live",
                not any(
                    bool(upstream_readiness.get(key))
                    for key in ("replay_allowed", "iql_allowed", "shadow_live_allowed")
                ),
            ),
        ]
    else:
        upstream_readiness = run_train_readiness(
            argparse.Namespace(
                audit_doc=str(args.audit_doc),
                out_dir=str(TRAIN_READINESS_OUT_DIR),
                fail_on_not_ready=False,
                quiet=True,
            )
        )
        upstream_artifact_path = Path(
            str(upstream_readiness.get("json_path") or (TRAIN_READINESS_OUT_DIR / "ENTRY_TRAINING_READINESS_latest.json"))
        )
        upstream_checks = [
            _check(
                "foundation train-readiness is green",
                str(upstream_readiness.get("decision")) == "READY_FOR_VEDTAK_SMOKE_TRAIN",
                {"decision": upstream_readiness.get("decision"), "failures": upstream_readiness.get("failures")},
            ),
            _check("train-readiness still blocks candidate training", bool(upstream_readiness.get("candidate_training_allowed")) is False),
            _check("train-readiness still blocks promotion/shadow/live", bool(upstream_readiness.get("promotion_shadow_live_allowed")) is False),
        ]
    smoke_audit_load_error = None
    try:
        smoke_audit = _read_json(smoke_audit_path)
    except Exception as exc:  # Artifact absence/corruption is a gate failure, not a traceback.
        smoke_audit = {}
        smoke_audit_load_error = str(exc)
    smart_benchmark_checks: list[dict[str, Any]] = []
    if contract_mode == "smart_seq520_candidate":
        try:
            foundation_smoke_audit = _read_json(SMOKE_BUNDLE_AUDIT_LATEST)
            foundation_smoke_load_error = None
        except Exception as exc:
            foundation_smoke_audit = {}
            foundation_smoke_load_error = str(exc)
        try:
            seq215_smoke_audit = _read_json(CHALLENGER_SEQ215_SMOKE_BUNDLE_AUDIT_LATEST)
            seq215_smoke_load_error = None
        except Exception as exc:
            seq215_smoke_audit = {}
            seq215_smoke_load_error = str(exc)
        smart_benchmark_checks = [
            _check(
                "foundation smoke benchmark audit exists and is readable",
                foundation_smoke_load_error is None,
                {"path": str(SMOKE_BUNDLE_AUDIT_LATEST), "error": foundation_smoke_load_error},
            ),
            _check(
                "seq215 smoke benchmark audit exists and is readable",
                seq215_smoke_load_error is None,
                {"path": str(CHALLENGER_SEQ215_SMOKE_BUNDLE_AUDIT_LATEST), "error": seq215_smoke_load_error},
            ),
            *_smart_smoke_benchmark_checks(
                smoke_audit,
                foundation_report=foundation_smoke_audit,
                seq215_report=seq215_smoke_audit,
            ),
        ]
    artifacts = {
        upstream_gate_name: str(upstream_artifact_path),
        "smoke_bundle_audit": str(smoke_audit_path),
        "specialist_audit": str(specialist_audit_path),
    }
    artifact_fingerprints = _artifact_fingerprints(artifacts)
    gate_checks = {
        upstream_gate_name: upstream_checks,
        "smoke_edge_audit": [
            _check(
                "smoke bundle audit JSON exists and is readable",
                smoke_audit_load_error is None,
                {"path": str(smoke_audit_path), "error": smoke_audit_load_error},
            ),
            *_smoke_edge_checks(
                smoke_audit,
                contract_mode=contract_mode,
                expected_smoke_dataset_dir=expected_smoke_dataset_dir,
                min_active_specialists=int(args.min_active_specialists),
            ),
        ],
        "promotion_guard": [
            _check("candidate gate never promotes", True),
            _check("candidate gate never starts shadow/live", True),
        ],
        "artifact_provenance": _artifact_fingerprint_checks(artifact_fingerprints),
    }
    if smart_benchmark_checks:
        gate_checks["smart_smoke_benchmark"] = smart_benchmark_checks
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
    contract_training_enabled = bool(specialist_contract_training_allowed_for_mode(contract_mode))
    candidate_training_allowed = bool(ready and contract_training_enabled)
    if ready and not contract_training_enabled:
        failures.append(
            {
                "gate": "training_enablement",
                "check": "specialist contract training enabled",
                "details": {
                    "contract_mode": contract_mode,
                    "training_allowed_by_contract": False,
                    "reason": "audited evidence contract is not enabled for candidate trainer start",
                },
            }
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "entry_candidate_readiness_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract_mode": contract_mode,
        "expected_signal_dim": expected_signal_dim,
        "expected_smoke_dataset_dir": str(expected_smoke_dataset_dir),
        "required_specialist_groups": list(required_specialist_groups),
        "readiness_checks_pass": bool(ready),
        "training_allowed_by_contract": bool(contract_training_enabled),
        "decision": (
            "READY_FOR_CANDIDATE_TRAINING_VEDTAK"
            if candidate_training_allowed
            else "NOT_READY_FOR_CANDIDATE_TRAINING"
        ),
        "candidate_training_allowed_with_explicit_vedtak": bool(candidate_training_allowed),
        "promotion_shadow_live_allowed": False,
        "next_required_gate": (
            f"{_mode_candidate_train_command(contract_mode)} then post-train replay gates"
            if candidate_training_allowed
            else "enable specialist contract training for this mode under explicit review"
            if ready and not contract_training_enabled
            else f"run {_mode_smoke_train_command(contract_mode)}"
        ),
        "artifacts": artifacts,
        "artifact_fingerprints": artifact_fingerprints,
        "smoke_bundle_audit_json": str(smoke_audit_path),
        "smoke_bundle_audit_load_error": smoke_audit_load_error,
        "upstream_readiness_gate": upstream_gate_name,
        "upstream_readiness_json": str(upstream_artifact_path),
        "train_readiness_json": str(upstream_artifact_path) if upstream_gate_name == "train_readiness" else None,
        "gates": gates,
        "failures": failures,
    }
    json_path = out_dir / f"ENTRY_CANDIDATE_READINESS_{timestamp}.json"
    md_path = out_dir / f"ENTRY_CANDIDATE_READINESS_{timestamp}.md"
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    latest_json = out_dir / "ENTRY_CANDIDATE_READINESS_latest.json"
    latest_md = out_dir / "ENTRY_CANDIDATE_READINESS_latest.md"
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
    ap.add_argument("--audit-doc", default=str(REPO / "docs/ENTRY_FOUNDATION_AUDIT_20260628.md"))
    ap.add_argument("--smoke-bundle-audit-json", default=None)
    ap.add_argument("--specialist-audit-json", default=None)
    ap.add_argument("--contract-mode", choices=SPECIALIST_CONTRACT_MODES, default="foundation_seq146")
    ap.add_argument("--challenger-seq215", action="store_const", const="challenger_seq215", dest="contract_mode")
    ap.add_argument("--smart-seq520", action="store_const", const="smart_seq520_candidate", dest="contract_mode")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--min-active-specialists", type=int, default=len(REQUIRED_SPECIALIST_GROUPS))
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
