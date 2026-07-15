#!/usr/bin/env python3
"""Audit an Entry foundation smoke bundle after strict load and forward pass.

This is a post-train plumbing/diagnostic gate. It never trains, promotes, pins,
starts shadow, or touches live/practice order paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gx1.features.entry_specialist_feature_groups_v1 import (
    SPECIALIST_FUSION_ACTIVE_HEADS,
    SPECIALIST_FUSION_BLOCKED_HEADS,
    required_training_specialists_for_mode,
    specialist_model_contract_for_mode,
)
from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import EntryV10CtxDataset, _multi_tf_kwargs_from_batch
from gx1.scripts.verify_entry_foundation_state_v1 import (
    FOUNDATION_SMOKE_DATASET_DIR,
    FOUNDATION_SPECIALIST_SANITY_BUNDLE_DIR,
    REPORTS_ROOT,
    RUN_ROOT,
    TARGET_AUDIT_LATEST,
)


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_foundation_smoke_bundle_audit_20260628_v1"
DEFAULT_M5_PREBUILT = (
    RUN_ROOT
    / "v10_6yr_rebuild_20260626_spreadfix/cv3/xauusd_m5_CANONICAL_V3_2020_2026.parquet"
)
DEFAULT_MTF_CACHE_DIR = RUN_ROOT / "MULTI_TF_V2_CACHE"
CLASS_NAMES = {0: "LONG", 1: "SHORT", 2: "FLAT"}
MIN_DIRECTION_CLASS_LABEL_RATE_FOR_COVERAGE = 0.10
MIN_DIRECTION_CLASS_PRED_RATE = 0.05
MIN_DIRECTION_CLASS_PRED_TO_LABEL_RATIO = 0.35
MIN_DIRECTION_SLICE_ROWS = 64
SMART_DIRECTION_BALANCE_MIN_ALPHA = 0.45
SMART_DIRECTION_CE_SCALE_MIN = 2.00
SMART_DIRECTION_BALANCE_CLASS_WEIGHTS = [1.0, 1.0, 4.0]
SMART_DIRECTION_CKPT_BALANCE_GUARD_MIN_WEIGHT = 0.50
SMART_DIRECTION_CKPT_BALANCE_MIN_PRED_TO_LABEL = 0.35
SMART_DIRECTION_CKPT_BALANCE_MIN_PRED_RATE = 0.05
SMART_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT = 2.50
SMART_DIRECTION_MIN_PRED_RATE_FRACTION = 0.50
SMART_DIRECTION_MIN_PRED_RATE_FLOOR = 0.05
SMART_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE_MAX = 0.05
SMART_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT = 8.00
SMART_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE = 0.02
SMART_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE = 0.10
SMART_DIRECTION_SLICE_BALANCED_CE_WEIGHT = 2.00
SMART_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE = 0.10
SMART_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS = 8
SMART_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT = 2.00
SMART_DIRECTION_SLICE_TRUE_MARGIN = 0.10
SMART_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE = 0.10
SMART_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS = 8
SMART_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT = 4.00
SMART_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN = 0.02
SMART_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE = 0.10
SMART_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS = 8
SMART_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT = 3.00
SMART_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE = 0.02
SMART_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE = 0.10
SMART_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS = 8
SMART_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS = 8
SMART_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE = 3
SMART_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS = 6
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
SMART_DIRECTION_HIER_SLICE_SIDE_CE_WEIGHT = 4.00
SMART_DIRECTION_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT = 3.00
SMART_DIRECTION_HIER_SLICE_SIDE_TRUE_MARGIN = 0.10
SMART_DIRECTION_HIER_SLICE_SIDE_MIN_LABEL_RATE = 0.10
SMART_DIRECTION_HIER_SLICE_SIDE_MIN_ROWS = 8
SMART_DIRECTION_FLAT_STARVATION_WEIGHT = 8.00
SMART_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE = 0.10
SMART_DIRECTION_FLAT_STARVATION_MIN_ROWS = 8
SMART_DIRECTION_FLAT_STARVATION_PRED_FRACTION = 0.50
SMART_DIRECTION_FLAT_STARVATION_PRED_FLOOR = 0.10
SMART_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN = 0.10
SMART_DIRECTION_HIER_LEGACY_CE_MULT_MIN = 1.00
SMART_DIRECTION_RESIDUAL_SCALE = 0.35
SMART_DIRECTION_ANCHOR_EPS = 1e-6
SMART_DIRECTION_ANCHOR_GATE_INIT_MAX = 0.05
SMART_DIRECTION_SIDE_VALIDITY_WEIGHT_MIN = 1.50
SMART_DIRECTION_TRENDLINE_RAIL_AUX_WEIGHT_MIN = 1.00
SMART_DIRECTION_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT_MIN = 1.50
HEAD_OUTPUT_KEYS = {
    "direction": "direction_logits",
    "tradable": "tradable_logit",
    "path_quality": "path_quality",
    "mfe_first_n": "mfe_first_n",
    "bad_path": "bad_path_logit",
    "clean_edge": "clean_edge_logit",
    "survival": "survival_logit",
    "tf_agreement": "tf_agreement_logit",
    "path_quality_log_var": "path_quality_log_var",
    "position_size": "position_size_logit",
    "dip": "dip_pred",
    "forecast": "forecast_pred",
    "timing": "timing_pred",
    "tail_risk": "tail_risk_pred",
    "vol_forecast": "vol_forecast_pred",
    "mtf_direction": "mtf_dir_logits",
    "hold_horizon": "hold_horizon_logit",
    "trendline_rail": "trendline_rail_logits",
    "side_validity": "side_validity_logit",
    "trade_side_hierarchy": (
        "trade_logit",
        "side_logits",
        "side_utility",
        "side_bad_path_logit",
        "side_mae",
    ),
}
HEAD_OUTPUT_TRAILING_SHAPES = {
    "direction": (3,),
    "tradable": (1,),
    "path_quality": (1,),
    "mfe_first_n": (1,),
    "bad_path": (1,),
    "clean_edge": (1,),
    "survival": (1,),
    "tf_agreement": (1,),
    "path_quality_log_var": (1,),
    "position_size": (1,),
    "dip": (18,),
    "forecast": (4,),
    "timing": (12,),
    "tail_risk": (6,),
    "vol_forecast": (3,),
    "mtf_direction": (3,),
    "hold_horizon": (1,),
    "trendline_rail": (6,),
    "side_validity": (2,),
    "trade_side_hierarchy": {
        "trade_logit": (1,),
        "side_logits": (2,),
        "side_utility": (2,),
        "side_bad_path_logit": (2,),
        "side_mae": (2,),
    },
}
SMART_EXTRA_ACTIVE_HEADS = ("trade_side_hierarchy", "trendline_rail", "side_validity")
SMART_ALLOWED_INTERNAL_HEADS = ("anchor_gate",)


def _head_output_specs(head: str) -> list[tuple[str, list[int]]]:
    keys = HEAD_OUTPUT_KEYS[head]
    shapes = HEAD_OUTPUT_TRAILING_SHAPES[head]
    if isinstance(keys, str):
        return [(keys, list(shapes))]
    if not isinstance(shapes, dict):
        raise RuntimeError(f"head {head} has grouped output keys but no grouped shape contract")
    return [(str(key), list(shapes[str(key)])) for key in keys]


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


def _parse_csv(raw: str) -> list[str]:
    values = [part.strip() for part in str(raw or "").split(",") if part.strip()]
    if not values:
        raise SystemExit("expected at least one split")
    return values


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_hash_check(path: Path, expected: str) -> dict[str, Any]:
    observed = _sha256_file(path) if path.exists() else ""
    check: dict[str, Any] = {
        "path": str(path),
        "expected": expected,
        "observed": observed,
        "ok": bool(expected and observed and expected == observed),
    }
    if check["ok"] or not expected or "_latest." not in path.name:
        return check

    matches: list[Path] = []
    for sibling in sorted(path.parent.glob("*.json")):
        sibling = sibling.resolve()
        if sibling == path or "_latest." in sibling.name:
            continue
        try:
            if _sha256_file(sibling) == expected:
                matches.append(sibling)
        except OSError:
            continue

    check["latest_resolution_candidate_count"] = len(matches)
    check["latest_resolution_candidates"] = [str(match) for match in matches]
    if len(matches) == 1:
        check["ok"] = True
        check["mutable_latest_observed"] = observed
        check["observed"] = expected
        check["resolved_path"] = str(matches[0])
        check["resolution"] = "timestamped_sibling_by_expected_sha256"
    elif len(matches) > 1:
        check["latest_resolution_error"] = "ambiguous timestamped artifacts with expected sha256"
    else:
        check["latest_resolution_error"] = "no timestamped sibling artifact with expected sha256"
    return check


def _device_arg(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if raw not in {"cpu", "cuda"}:
        raise SystemExit(f"--device must be auto, cpu, or cuda; got {raw!r}")
    if raw == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    return raw


def _split_file(dataset_dir: Path, split: str) -> Path:
    matches = sorted(dataset_dir.glob(f"*_{split}.parquet"))
    if not matches:
        matches = sorted(dataset_dir.glob(f"*{split}*.parquet"))
    if not matches:
        raise FileNotFoundError(f"no parquet found for split={split!r} under {dataset_dir}")
    if len(matches) > 1:
        exact = [path for path in matches if path.name.endswith(f"_{split}.parquet")]
        matches = exact or matches
    if len(matches) != 1:
        raise RuntimeError(f"ambiguous parquet files for split={split}: {[str(path) for path in matches]}")
    return matches[0]


def _majority_baseline_accuracy(labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return 0.0
    counts = np.bincount(labels, minlength=3)
    return float(counts.max() / labels.size)


def _class_count_dict(labels: np.ndarray) -> dict[str, int]:
    counts = Counter(int(x) for x in np.asarray(labels, dtype=np.int64).ravel())
    return {CLASS_NAMES.get(i, str(i)): int(counts.get(i, 0)) for i in range(3)}


def _safe_mean(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.nanmean(arr))


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 3:
        return None
    if float(np.nanstd(x_arr)) == 0.0 or float(np.nanstd(y_arr)) == 0.0:
        return None
    rx = pd.Series(x_arr).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y_arr).rank(method="average").to_numpy(dtype=np.float64)
    corr = float(np.corrcoef(rx, ry)[0, 1])
    return corr if np.isfinite(corr) else None


def _direction_metrics(direction_logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    logits = np.asarray(direction_logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).ravel()
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise RuntimeError(f"direction logits must have shape (N,3), got {logits.shape}")
    if logits.shape[0] != y.shape[0]:
        raise RuntimeError(f"direction logits/labels row mismatch: {logits.shape[0]} vs {y.shape[0]}")
    shifted = logits - np.nanmax(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
    pred = probs.argmax(axis=1).astype(np.int64)
    accuracy = float((pred == y).mean()) if y.size else 0.0
    majority = _majority_baseline_accuracy(y)
    label_counts = _class_count_dict(y)
    prediction_counts = _class_count_dict(pred)
    rows = int(y.size)
    label_rate = {
        name: (float(count) / float(rows) if rows else 0.0)
        for name, count in label_counts.items()
    }
    prediction_rate = {
        name: (float(count) / float(rows) if rows else 0.0)
        for name, count in prediction_counts.items()
    }
    return {
        "rows": rows,
        "accuracy": accuracy,
        "majority_baseline_accuracy": majority,
        "beats_majority_baseline": bool(accuracy > majority),
        "label_counts": label_counts,
        "prediction_counts": prediction_counts,
        "label_rate": label_rate,
        "prediction_rate": prediction_rate,
        "prediction_to_label_rate": {
            name: (
                float(prediction_rate.get(name, 0.0)) / max(float(label_rate.get(name, 0.0)), 1e-12)
            )
            for name in CLASS_NAMES.values()
        },
        "mean_probability": {
            CLASS_NAMES[i]: float(probs[:, i].mean()) if probs.size else 0.0
            for i in range(3)
        },
        "max_abs_logit": float(np.nanmax(np.abs(logits))) if logits.size else 0.0,
    }


def _direction_distribution_contract(direction: dict[str, Any]) -> dict[str, Any]:
    rows = int(direction.get("rows") or 0)
    label_counts = direction.get("label_counts") if isinstance(direction.get("label_counts"), dict) else {}
    prediction_counts = (
        direction.get("prediction_counts") if isinstance(direction.get("prediction_counts"), dict) else {}
    )
    failures: list[str] = []
    class_rows: dict[str, dict[str, float | int]] = {}
    for name in CLASS_NAMES.values():
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
        "rows": rows,
        "min_label_rate_for_coverage": MIN_DIRECTION_CLASS_LABEL_RATE_FOR_COVERAGE,
        "min_prediction_rate": MIN_DIRECTION_CLASS_PRED_RATE,
        "min_prediction_to_label_rate": MIN_DIRECTION_CLASS_PRED_TO_LABEL_RATIO,
        "classes": class_rows,
        "failures": failures,
    }


def _direction_slice_contract(
    direction_logits: np.ndarray,
    labels: np.ndarray,
    ctx_cat: np.ndarray,
    ctx_cat_names: list[str],
    *,
    min_rows: int = MIN_DIRECTION_SLICE_ROWS,
) -> dict[str, Any]:
    logits = np.asarray(direction_logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).ravel()
    cat = np.asarray(ctx_cat)
    if cat.ndim == 1:
        cat = cat.reshape(-1, 1)
    if logits.shape[0] != y.shape[0] or cat.shape[0] != y.shape[0]:
        raise RuntimeError(
            "direction slice row mismatch: "
            f"logits={logits.shape[0]} labels={y.shape[0]} ctx_cat={cat.shape[0]}"
        )
    names = list(ctx_cat_names or [])
    if len(names) < cat.shape[1]:
        names.extend(f"ctx_cat_{idx}" for idx in range(len(names), cat.shape[1]))

    failures: list[str] = []
    fields: dict[str, Any] = {}
    audited_slice_count = 0
    skipped_slice_count = 0
    for col in range(cat.shape[1]):
        field_name = str(names[col])
        raw_values = cat[:, col]
        finite = np.isfinite(raw_values.astype(np.float64, copy=False))
        field_slices: dict[str, Any] = {}
        for raw_value in sorted(set(int(v) for v in raw_values[finite].ravel())):
            mask = finite & (raw_values.astype(np.int64, copy=False) == int(raw_value))
            rows = int(mask.sum())
            if rows < int(min_rows):
                skipped_slice_count += 1
                field_slices[str(raw_value)] = {
                    "decision": "SKIP_INSUFFICIENT_ROWS",
                    "rows": rows,
                    "min_rows": int(min_rows),
                }
                continue
            metrics = _direction_metrics(logits[mask], y[mask])
            active_label_class_count = sum(
                1
                for rate in (metrics.get("label_rate") or {}).values()
                if float(rate) >= MIN_DIRECTION_CLASS_LABEL_RATE_FOR_COVERAGE
            )
            if active_label_class_count < 2:
                skipped_slice_count += 1
                field_slices[str(raw_value)] = {
                    "decision": "SKIP_LOW_LABEL_DIVERSITY",
                    "rows": rows,
                    "min_rows": int(min_rows),
                    "active_label_class_count": int(active_label_class_count),
                    "label_counts": metrics["label_counts"],
                }
                continue
            audited_slice_count += 1
            distribution = _direction_distribution_contract(metrics)
            slice_failures: list[str] = []
            if not bool(metrics.get("beats_majority_baseline")):
                slice_failures.append(
                    f"accuracy={float(metrics.get('accuracy') or 0.0):.6f} does not beat "
                    f"majority={float(metrics.get('majority_baseline_accuracy') or 0.0):.6f}"
                )
            if str(distribution.get("decision")) != "PASS":
                slice_failures.extend(str(item) for item in distribution.get("failures", []))
            if slice_failures:
                failures.extend(
                    f"{field_name}={raw_value}: {failure}"
                    for failure in slice_failures
                )
            field_slices[str(raw_value)] = {
                "decision": "PASS" if not slice_failures else "FAIL",
                "rows": rows,
                "accuracy": metrics["accuracy"],
                "majority_baseline_accuracy": metrics["majority_baseline_accuracy"],
                "beats_majority_baseline": metrics["beats_majority_baseline"],
                "label_counts": metrics["label_counts"],
                "prediction_counts": metrics["prediction_counts"],
                "direction_distribution_contract": distribution,
                "failures": slice_failures,
            }
        fields[field_name] = {
            "finite": bool(finite.all()),
            "slice_count": len(field_slices),
            "slices": field_slices,
        }
        if not bool(finite.all()):
            failures.append(f"{field_name}: non-finite categorical slice values")

    if cat.shape[1] <= 0:
        failures.append("direction slice contract has no categorical context columns")
    if audited_slice_count <= 0:
        failures.append("direction slice contract has no audited slices with sufficient rows")

    return {
        "decision": "PASS" if not failures else "FAIL",
        "min_rows": int(min_rows),
        "ctx_cat_names": names[: cat.shape[1]],
        "audited_slice_count": int(audited_slice_count),
        "skipped_slice_count": int(skipped_slice_count),
        "fields": fields,
        "failures": failures,
    }


def _gate_stats(gate: np.ndarray, names: list[str]) -> dict[str, Any]:
    arr = np.asarray(gate, dtype=np.float64)
    if arr.ndim != 2:
        raise RuntimeError(f"specialist gate must have shape (N,S), got {arr.shape}")
    row_sum = arr.sum(axis=1)
    entropy = -(arr * np.log(np.maximum(arr, 1e-12))).sum(axis=1)
    mean_weight = arr.mean(axis=0) if arr.size else np.zeros((len(names),), dtype=np.float64)
    if len(names) != arr.shape[1]:
        names = [f"specialist_{i}" for i in range(arr.shape[1])]
    return {
        "rows": int(arr.shape[0]),
        "specialist_count": int(arr.shape[1]),
        "finite": bool(np.isfinite(arr).all()),
        "row_sum_max_abs_error": float(np.max(np.abs(row_sum - 1.0))) if row_sum.size else 0.0,
        "entropy_mean": _safe_mean(entropy),
        "entropy_min": float(np.nanmin(entropy)) if entropy.size else None,
        "mean_weight": {name: float(mean_weight[i]) for i, name in enumerate(names)},
        "mean_weight_min": float(np.nanmin(mean_weight)) if mean_weight.size else None,
        "mean_weight_max": float(np.nanmax(mean_weight)) if mean_weight.size else None,
        "active_specialist_count_gt_1pct": int(np.sum(mean_weight > 0.01)) if mean_weight.size else 0,
    }


def _specialist_gate_failures(
    *,
    split: str,
    gate_report: dict[str, Any] | None,
    specialist_names: list[str],
    required_specialists: tuple[str, ...],
    min_active_specialists: int,
    min_gate_entropy: float,
) -> list[str]:
    if gate_report is None:
        return [f"{split}: specialist_fusion enabled/required but model emitted no specialist_gate"]

    failures: list[str] = []
    names = set(str(name) for name in specialist_names)
    expected = set(str(name) for name in required_specialists)
    missing = [name for name in required_specialists if name not in names]
    if missing:
        failures.append(f"{split}: specialist_fusion metadata missing required specialists: {missing}")
    extra = sorted(names - expected)
    if extra:
        failures.append(f"{split}: specialist_fusion metadata has non-required specialists: {extra}")

    if int(gate_report.get("specialist_count") or 0) != len(specialist_names):
        failures.append(
            f"{split}: specialist_gate width={gate_report.get('specialist_count')} "
            f"does not match metadata groups={len(specialist_names)}"
        )

    active_count = int(gate_report.get("active_specialist_count_gt_1pct") or 0)
    if active_count < int(min_active_specialists):
        failures.append(
            f"{split}: specialist_gate collapsed active_specialist_count_gt_1pct={active_count} "
            f"below minimum {int(min_active_specialists)}"
        )

    mean_weight = gate_report.get("mean_weight") if isinstance(gate_report.get("mean_weight"), dict) else {}
    collapsed_required = [
        name
        for name in required_specialists
        if float(mean_weight.get(name) or 0.0) <= 0.01
    ]
    if collapsed_required:
        failures.append(
            f"{split}: required specialist gate weights collapsed below 1pct: {collapsed_required}"
        )

    entropy_mean = gate_report.get("entropy_mean")
    if entropy_mean is None or float(entropy_mean) < float(min_gate_entropy):
        failures.append(
            f"{split}: specialist_gate entropy_mean={entropy_mean} below minimum {float(min_gate_entropy)}"
        )
    return failures


def _bundle_dataset_kwargs(meta: dict[str, Any], m5_prebuilt_path: Path) -> dict[str, Any]:
    mtf = meta.get("multi_tf") if isinstance(meta.get("multi_tf"), dict) else {}
    if not bool(mtf.get("enabled", False)):
        return {}
    if not m5_prebuilt_path.is_file():
        raise FileNotFoundError(f"multi-TF bundle requires M5 prebuilt: {m5_prebuilt_path}")
    seq_len = int(mtf.get("m15_seq_len", 96))
    return {
        "enable_multi_tf": True,
        "m5_prebuilt_path": m5_prebuilt_path,
        "multi_tf_seq_len": seq_len,
        "multi_tf_closed_bar": bool(float(mtf.get("target_availability_shift_minutes", 0.0) or 0.0) > 0.0),
        "per_tf_seq_lens": {
            "M5": int(mtf.get("m5_seq_len", seq_len) or seq_len),
            "M15": int(mtf.get("m15_seq_len", seq_len) or seq_len),
            "H1": int(mtf.get("h1_seq_len", seq_len) or seq_len),
            "H4": int(mtf.get("h4_seq_len", seq_len) or seq_len),
            "D1": int(mtf.get("d1_seq_len", seq_len) or seq_len),
        },
    }


def _specialist_names(meta: dict[str, Any]) -> list[str]:
    cfg = meta.get("specialist_fusion") if isinstance(meta.get("specialist_fusion"), dict) else {}
    indices = cfg.get("input_indices") if isinstance(cfg.get("input_indices"), dict) else {}
    return sorted(str(name) for name in indices.keys())


def _specialist_model_contract_report(
    contract: dict[str, Any] | None,
    *,
    contract_mode: str = "foundation_seq146",
) -> dict[str, Any]:
    observed = contract if isinstance(contract, dict) else {}
    failures: list[str] = []
    expected_contract = specialist_model_contract_for_mode(contract_mode)
    observed_keys = {str(name) for name in observed}
    expected_keys = {str(name) for name in expected_contract}
    if observed_keys != expected_keys:
        failures.append(
            "specialist model contract set mismatch: "
            f"observed={sorted(observed_keys)} expected={sorted(expected_keys)}"
        )

    owned_objectives_match = True
    support_heads_match = True
    signal_families_match = True
    model_roles_match = True
    for name, expected_spec in expected_contract.items():
        observed_spec = observed.get(name)
        if not isinstance(observed_spec, dict):
            failures.append(f"specialist model contract missing spec for {name}")
            owned_objectives_match = False
            support_heads_match = False
            signal_families_match = False
            model_roles_match = False
            continue
        if str(observed_spec.get("model_role") or "") != str(expected_spec.get("model_role") or ""):
            failures.append(f"specialist model contract model_role mismatch: {name}")
            model_roles_match = False
        checks = (
            ("owned_objectives", "owned objectives", "owned_objectives_match"),
            ("supports_heads", "support heads", "support_heads_match"),
            ("primary_signal_families", "signal families", "signal_families_match"),
        )
        for field, label, flag in checks:
            observed_values = tuple(str(x) for x in observed_spec.get(field) or ())
            expected_values = tuple(str(x) for x in expected_spec.get(field) or ())
            if observed_values != expected_values:
                failures.append(f"specialist model contract {label} mismatch: {name}")
                if flag == "owned_objectives_match":
                    owned_objectives_match = False
                elif flag == "support_heads_match":
                    support_heads_match = False
                elif flag == "signal_families_match":
                    signal_families_match = False

    return {
        "decision": "PASS" if not failures else "FAIL",
        "contract_mode": str(contract_mode),
        "valid": not failures,
        "set_exact": observed_keys == expected_keys,
        "owned_objectives_match": owned_objectives_match,
        "support_heads_match": support_heads_match,
        "signal_families_match": signal_families_match,
        "model_roles_match": model_roles_match,
        "observed_specialists": sorted(observed_keys),
        "expected_specialists": sorted(expected_keys),
        "failures": failures,
    }


def _head_contract_report(
    *,
    target_audit: dict[str, Any],
    capabilities: dict[str, Any],
    split_reports: dict[str, Any],
    contract_mode: str = "foundation_seq146",
) -> dict[str, Any]:
    contract = target_audit.get("target_head_contract") if isinstance(target_audit.get("target_head_contract"), dict) else {}
    base_active_heads = [str(head) for head in contract.get("active_training_heads", []) if str(head)]
    blocked_heads = [str(head) for head in contract.get("blocked_heads", []) if str(head)]
    supported_heads = set(str(head) for head in capabilities.get("supported_heads", []) if str(head))
    declared_heads = set(str(head) for head in capabilities.get("declared_active_heads", []) if str(head))
    state_dict_heads = set(str(head) for head in capabilities.get("state_dict_heads", []) if str(head))
    smart_extra_heads = [
        head
        for head in SMART_EXTRA_ACTIVE_HEADS
        if str(contract_mode) == "smart_seq520_candidate"
        and (head in declared_heads or head in supported_heads or head in state_dict_heads)
    ]
    active_heads = list(dict.fromkeys([*base_active_heads, *smart_extra_heads]))
    output_keys_by_split = {
        str(split): [str(key) for key in (row.get("output_keys_seen") or [])]
        for split, row in split_reports.items()
        if isinstance(row, dict)
    }
    output_shapes_by_split = {
        str(split): {
            str(key): [list(shape) for shape in (shapes or [])]
            for key, shapes in ((row.get("output_shapes_seen") or {}).items())
        }
        for split, row in split_reports.items()
        if isinstance(row, dict)
    }
    expected_output_specs = {
        head: _head_output_specs(head)
        for head in active_heads
        if head in HEAD_OUTPUT_KEYS and head in HEAD_OUTPUT_TRAILING_SHAPES
    }
    expected_output_keys = {
        head: [key for key, _ in specs]
        for head, specs in expected_output_specs.items()
    }
    expected_output_shapes = {
        head: {key: shape for key, shape in specs}
        for head, specs in expected_output_specs.items()
    }

    failures: list[str] = []
    if str(target_audit.get("decision")) != "PASS":
        failures.append(f"target audit decision is not PASS: {target_audit.get('decision')}")
    if not contract:
        failures.append("target audit does not contain target_head_contract")
    if not active_heads:
        failures.append("target_head_contract active_training_heads is empty")
    missing_output_contract = [head for head in active_heads if head not in HEAD_OUTPUT_KEYS]
    if missing_output_contract:
        failures.append(f"no smoke-audit output-key contract for active heads: {missing_output_contract}")
    missing_shape_contract = [head for head in active_heads if head not in HEAD_OUTPUT_TRAILING_SHAPES]
    if missing_shape_contract:
        failures.append(f"no smoke-audit output-shape contract for active heads: {missing_shape_contract}")

    missing_supported = [head for head in active_heads if head not in supported_heads]
    if missing_supported:
        failures.append(f"bundle capabilities missing active heads: {missing_supported}")
    missing_declared = [head for head in active_heads if head not in declared_heads]
    if missing_declared:
        failures.append(f"bundle train_recipe.active_heads missing active heads: {missing_declared}")
    missing_state = [head for head in active_heads if head not in state_dict_heads]
    if missing_state:
        failures.append(f"bundle state_dict missing active heads: {missing_state}")

    approved_heads = set(active_heads) | set(blocked_heads) | set(SMART_ALLOWED_INTERNAL_HEADS)
    extra_supported = sorted(supported_heads - approved_heads)
    if extra_supported:
        failures.append(f"bundle capabilities contain unapproved heads outside target contract: {extra_supported}")
    extra_declared = sorted(declared_heads - approved_heads)
    if extra_declared:
        failures.append(f"bundle train_recipe.active_heads contains unapproved heads: {extra_declared}")
    extra_state = sorted(state_dict_heads - approved_heads)
    if extra_state:
        failures.append(f"bundle state_dict contains unapproved heads outside target contract: {extra_state}")

    blocked_present = sorted(
        set(blocked_heads)
        & (supported_heads | declared_heads | state_dict_heads)
    )
    if blocked_present:
        failures.append(f"bundle contains blocked heads: {blocked_present}")

    for split, keys in output_keys_by_split.items():
        seen = set(keys)
        missing_keys = [
            f"{head}->{key}"
            for head, specs in expected_output_specs.items()
            for key, _ in specs
            if key not in seen
        ]
        if missing_keys:
            failures.append(f"{split}: missing forward outputs for active heads: {missing_keys}")
        split_shapes = output_shapes_by_split.get(split, {})
        shape_failures = []
        for head, specs in expected_output_specs.items():
            for key, expected_shape in specs:
                if key not in seen:
                    continue
                actual_shapes = split_shapes.get(key) or []
                if expected_shape not in actual_shapes:
                    shape_failures.append(f"{head}->{key} expected_tail={expected_shape} actual_tails={actual_shapes}")
        if shape_failures:
            failures.append(f"{split}: wrong forward output shapes for active heads: {shape_failures}")
        blocked_keys = []
        for head in blocked_heads:
            if head not in HEAD_OUTPUT_KEYS or head not in HEAD_OUTPUT_TRAILING_SHAPES:
                continue
            for key, _ in _head_output_specs(head):
                if key in seen:
                    blocked_keys.append(f"{head}->{key}")
        if blocked_keys:
            failures.append(f"{split}: blocked heads emitted forward outputs: {blocked_keys}")

    return {
        "decision": "PASS" if not failures else "FAIL",
        "target_audit_decision": str(target_audit.get("decision")),
        "contract_mode": str(contract_mode),
        "target_active_training_heads": base_active_heads,
        "smart_extra_active_heads": smart_extra_heads,
        "active_training_heads": active_heads,
        "blocked_heads": blocked_heads,
        "supported_heads": sorted(supported_heads),
        "declared_active_heads": sorted(declared_heads),
        "state_dict_heads": sorted(state_dict_heads),
        "expected_output_keys": expected_output_keys,
        "expected_output_shapes": expected_output_shapes,
        "output_keys_by_split": output_keys_by_split,
        "output_shapes_by_split": output_shapes_by_split,
        "failures": failures,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _three_float_list(value: Any, default: list[float]) -> list[float]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        return list(default)
    if len(parts) != 3:
        return list(default)
    out = [_safe_float(part, np.nan) for part in parts]
    if any(not np.isfinite(part) for part in out):
        return list(default)
    return [float(part) for part in out]


def _bool_value(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if value is None:
        return bool(default)
    return bool(value)


def _path_calibration_recipe_contract(meta: dict[str, Any], capabilities: dict[str, Any]) -> dict[str, Any]:
    recipe = meta.get("train_recipe") if isinstance(meta.get("train_recipe"), dict) else {}
    active_heads = {
        str(head)
        for head in (
            recipe.get("active_heads")
            or capabilities.get("declared_active_heads")
            or capabilities.get("supported_heads")
            or []
        )
        if str(head)
    }
    path_weight = _safe_float(recipe.get("path_quality_rank_weight", meta.get("path_quality_rank_weight", 0.0)))
    path_margin = _safe_float(recipe.get("path_quality_rank_margin", meta.get("path_quality_rank_margin", 0.0)))
    path_quantile = _safe_float(recipe.get("path_quality_rank_quantile", meta.get("path_quality_rank_quantile", 0.0)))
    bad_weight = _safe_float(
        recipe.get("bad_path_quality_rank_weight", meta.get("bad_path_quality_rank_weight", 0.0))
    )
    bad_margin = _safe_float(
        recipe.get("bad_path_quality_rank_margin", meta.get("bad_path_quality_rank_margin", 0.0))
    )
    bad_quantile = _safe_float(
        recipe.get("bad_path_quality_rank_quantile", meta.get("bad_path_quality_rank_quantile", 0.0))
    )
    failures: list[str] = []
    if "path_quality" in active_heads:
        if path_weight <= 0.0:
            failures.append("path_quality active head requires positive path_quality_rank_weight")
        if not bool(recipe.get("path_quality_rank_full_batch")):
            failures.append("path_quality active head requires path_quality_rank_full_batch=true")
        if path_margin <= 0.0:
            failures.append("path_quality active head requires positive path_quality_rank_margin")
        if path_quantile < 0.05 or path_quantile > 0.45:
            failures.append("path_quality active head requires path_quality_rank_quantile in [0.05, 0.45]")
    if "bad_path" in active_heads:
        if bad_weight <= 0.0:
            failures.append("bad_path active head requires positive bad_path_quality_rank_weight")
        if bad_margin <= 0.0:
            failures.append("bad_path active head requires positive bad_path_quality_rank_margin")
        if bad_quantile < 0.05 or bad_quantile > 0.45:
            failures.append("bad_path active head requires bad_path_quality_rank_quantile in [0.05, 0.45]")
    return {
        "decision": "PASS" if not failures else "FAIL",
        "active_heads": sorted(active_heads),
        "path_quality_active": "path_quality" in active_heads,
        "bad_path_active": "bad_path" in active_heads,
        "path_quality_rank_full_batch": bool(recipe.get("path_quality_rank_full_batch")),
        "path_quality_rank_weight": path_weight,
        "path_quality_rank_margin": path_margin,
        "path_quality_rank_quantile": path_quantile,
        "bad_path_quality_rank_weight": bad_weight,
        "bad_path_quality_rank_margin": bad_margin,
        "bad_path_quality_rank_quantile": bad_quantile,
        "failures": failures,
    }


def _direction_balance_recipe_contract(
    meta: dict[str, Any],
    capabilities: dict[str, Any],
    *,
    contract_mode: str = "foundation_seq146",
) -> dict[str, Any]:
    recipe = meta.get("train_recipe") if isinstance(meta.get("train_recipe"), dict) else {}
    active_heads = {
        str(head)
        for head in (
            recipe.get("active_heads")
            or capabilities.get("declared_active_heads")
            or capabilities.get("supported_heads")
            or []
        )
        if str(head)
    }
    pred_balance_alpha = _safe_float(recipe.get("pred_balance_alpha", meta.get("pred_balance_alpha", 0.0)))
    pred_balance_target = str(recipe.get("pred_balance_target", meta.get("pred_balance_target", ""))).strip().lower()
    pred_balance_class_weights = _three_float_list(
        recipe.get("pred_balance_class_weights", meta.get("pred_balance_class_weights")),
        [1.0, 1.0, 1.0],
    )
    enable_mtf_direction_head = bool(meta.get("enable_mtf_direction_head", False))
    mtf_dir_aux_weight_present = ("mtf_dir_aux_weight" in recipe) or ("mtf_dir_aux_weight" in meta)
    mtf_dir_aux_weight = _safe_float(recipe.get("mtf_dir_aux_weight", meta.get("mtf_dir_aux_weight", 0.0)))
    mtf_dir_aux_uses_direction_balance_repair = bool(
        recipe.get(
            "mtf_dir_aux_uses_direction_balance_repair",
            meta.get("mtf_dir_aux_uses_direction_balance_repair", False),
        )
    )
    mtf_dir_aux_balance_repair_required = bool(
        contract_mode == "smart_seq520_candidate"
        and enable_mtf_direction_head
        and mtf_dir_aux_weight > 0.0
    )
    direction_ce_scale = _safe_float(recipe.get("direction_ce_scale", meta.get("direction_ce_scale", 0.0)))
    hierarchical_entry_heads_enabled = bool(
        recipe.get(
            "hierarchical_entry_heads_enabled",
            meta.get("hierarchical_entry_heads_enabled", False),
        )
    )
    side_validity_head_enabled = bool(
        recipe.get(
            "side_validity_head_enabled",
            (meta.get("hierarchical_entry_heads") or {}).get("side_validity", {}).get(
                "enabled",
                meta.get("side_validity_head_enabled", False),
            )
            if isinstance(meta.get("hierarchical_entry_heads"), dict)
            else meta.get("side_validity_head_enabled", False),
        )
    )
    hier_side_validity_weight = _safe_float(
        recipe.get(
            "hier_side_validity_weight",
            (meta.get("hierarchical_entry_heads") or {}).get("side_validity", {}).get(
                "loss_weight",
                meta.get("hier_side_validity_weight", 0.0),
            )
            if isinstance(meta.get("hierarchical_entry_heads"), dict)
            else meta.get("hier_side_validity_weight", 0.0),
        )
    )
    hier_side_validity_min_utility_bps = _safe_float(
        recipe.get(
            "hier_side_validity_min_utility_bps",
            (meta.get("hierarchical_entry_heads") or {}).get("side_validity", {}).get(
                "min_utility_bps",
                meta.get("hier_side_validity_min_utility_bps", 0.0),
            )
            if isinstance(meta.get("hierarchical_entry_heads"), dict)
            else meta.get("hier_side_validity_min_utility_bps", 0.0),
        )
    )
    hier_side_validity_pos_weight_cap = _safe_float(
        recipe.get(
            "hier_side_validity_pos_weight_cap",
            (meta.get("hierarchical_entry_heads") or {}).get("side_validity", {}).get(
                "pos_weight_cap",
                meta.get("hier_side_validity_pos_weight_cap", 0.0),
            )
            if isinstance(meta.get("hierarchical_entry_heads"), dict)
            else meta.get("hier_side_validity_pos_weight_cap", 0.0),
        )
    )
    trendline_rail_head_enabled = bool(
        recipe.get(
            "trendline_rail_head_enabled",
            (meta.get("trendline_rail_head") or {}).get("enabled", meta.get("trendline_rail_head_enabled", False))
            if isinstance(meta.get("trendline_rail_head"), dict)
            else meta.get("trendline_rail_head_enabled", False),
        )
    )
    trendline_rail_aux_weight = _safe_float(
        recipe.get(
            "trendline_rail_aux_weight",
            (meta.get("trendline_rail_head") or {}).get("aux_weight", meta.get("trendline_rail_aux_weight", 0.0))
            if isinstance(meta.get("trendline_rail_head"), dict)
            else meta.get("trendline_rail_aux_weight", 0.0),
        )
    )
    trendline_rail_wrong_side_weight = _safe_float(
        recipe.get(
            "trendline_rail_wrong_side_weight",
            (meta.get("trendline_rail_head") or {}).get(
                "wrong_side_weight",
                meta.get("trendline_rail_wrong_side_weight", 0.0),
            )
            if isinstance(meta.get("trendline_rail_head"), dict)
            else meta.get("trendline_rail_wrong_side_weight", 0.0),
        )
    )
    hier_legacy_ce_mult = _safe_float(
        recipe.get(
            "hier_legacy_ce_mult",
            meta.get("hier_legacy_ce_mult", 0.0),
        )
    )
    residual_scale = _safe_float(
        recipe.get(
            "residual_scale",
            meta.get("residual_scale", -1.0),
        )
    )
    anchor_eps = _safe_float(
        recipe.get(
            "anchor_eps",
            meta.get("anchor_eps", -1.0),
        )
    )
    anchor_gate_enabled = bool(
        recipe.get(
            "anchor_gate_enabled",
            meta.get("anchor_gate_enabled", False),
        )
    )
    anchor_gate_init = _safe_float(
        recipe.get(
            "anchor_gate_init",
            (meta.get("anchor_gate") or {}).get("init", meta.get("anchor_gate_init", 1.0))
            if isinstance(meta.get("anchor_gate"), dict)
            else meta.get("anchor_gate_init", 1.0),
        )
    )
    ckpt_monitor = str(recipe.get("ckpt_monitor", meta.get("ckpt_monitor", ""))).strip().lower()
    ckpt_class_balance_guard_weight = _safe_float(
        recipe.get(
            "ckpt_class_balance_guard_weight",
            meta.get("ckpt_class_balance_guard_weight", 0.0),
        )
    )
    ckpt_class_balance_min_pred_to_label = _safe_float(
        recipe.get(
            "ckpt_class_balance_min_pred_to_label",
            meta.get("ckpt_class_balance_min_pred_to_label", 0.0),
        )
    )
    ckpt_class_balance_min_pred_rate = _safe_float(
        recipe.get(
            "ckpt_class_balance_min_pred_rate",
            meta.get("ckpt_class_balance_min_pred_rate", 0.0),
        )
    )
    ckpt_direction_slice_guard = _bool_value(
        recipe.get(
            "ckpt_direction_slice_guard",
            meta.get("ckpt_direction_slice_guard", False),
        )
    )
    direction_min_pred_rate_loss_weight = _safe_float(
        recipe.get(
            "direction_min_pred_rate_loss_weight",
            meta.get("direction_min_pred_rate_loss_weight", 0.0),
        )
    )
    direction_min_pred_rate_fraction = _safe_float(
        recipe.get(
            "direction_min_pred_rate_fraction",
            meta.get("direction_min_pred_rate_fraction", 0.0),
        )
    )
    direction_min_pred_rate_floor = _safe_float(
        recipe.get(
            "direction_min_pred_rate_floor",
            meta.get("direction_min_pred_rate_floor", 0.0),
        )
    )
    direction_min_pred_rate_softmax_temperature = _safe_float(
        recipe.get(
            "direction_min_pred_rate_softmax_temperature",
            meta.get("direction_min_pred_rate_softmax_temperature", 0.0),
        )
    )
    direction_global_prior_match_weight = _safe_float(
        recipe.get(
            "direction_global_prior_match_weight",
            meta.get("direction_global_prior_match_weight", -1.0),
        )
    )
    direction_global_prior_match_tolerance = _safe_float(
        recipe.get(
            "direction_global_prior_match_tolerance",
            meta.get("direction_global_prior_match_tolerance", -1.0),
        )
    )
    direction_global_prior_match_min_label_rate = _safe_float(
        recipe.get(
            "direction_global_prior_match_min_label_rate",
            meta.get("direction_global_prior_match_min_label_rate", -1.0),
        )
    )
    direction_slice_balanced_ce_weight = _safe_float(
        recipe.get(
            "direction_slice_balanced_ce_weight",
            meta.get("direction_slice_balanced_ce_weight", -1.0),
        )
    )
    direction_slice_balanced_ce_min_label_rate = _safe_float(
        recipe.get(
            "direction_slice_balanced_ce_min_label_rate",
            meta.get("direction_slice_balanced_ce_min_label_rate", -1.0),
        )
    )
    direction_slice_balanced_ce_min_rows = int(
        _safe_float(
            recipe.get(
                "direction_slice_balanced_ce_min_rows",
                meta.get("direction_slice_balanced_ce_min_rows", -1.0),
            )
        )
    )
    direction_slice_true_margin_weight = _safe_float(
        recipe.get(
            "direction_slice_true_margin_weight",
            meta.get("direction_slice_true_margin_weight", -1.0),
        )
    )
    direction_slice_true_margin = _safe_float(
        recipe.get(
            "direction_slice_true_margin",
            meta.get("direction_slice_true_margin", -1.0),
        )
    )
    direction_slice_true_margin_min_label_rate = _safe_float(
        recipe.get(
            "direction_slice_true_margin_min_label_rate",
            meta.get("direction_slice_true_margin_min_label_rate", -1.0),
        )
    )
    direction_slice_true_margin_min_rows = int(
        _safe_float(
            recipe.get(
                "direction_slice_true_margin_min_rows",
                meta.get("direction_slice_true_margin_min_rows", -1.0),
            )
        )
    )
    direction_slice_accuracy_edge_weight = _safe_float(
        recipe.get(
            "direction_slice_accuracy_edge_weight",
            meta.get("direction_slice_accuracy_edge_weight", -1.0),
        )
    )
    direction_slice_accuracy_edge_margin = _safe_float(
        recipe.get(
            "direction_slice_accuracy_edge_margin",
            meta.get("direction_slice_accuracy_edge_margin", -1.0),
        )
    )
    direction_slice_accuracy_edge_min_label_rate = _safe_float(
        recipe.get(
            "direction_slice_accuracy_edge_min_label_rate",
            meta.get("direction_slice_accuracy_edge_min_label_rate", -1.0),
        )
    )
    direction_slice_accuracy_edge_min_rows = int(
        _safe_float(
            recipe.get(
                "direction_slice_accuracy_edge_min_rows",
                meta.get("direction_slice_accuracy_edge_min_rows", -1.0),
            )
        )
    )
    direction_slice_prior_match_weight = _safe_float(
        recipe.get(
            "direction_slice_prior_match_weight",
            meta.get("direction_slice_prior_match_weight", -1.0),
        )
    )
    direction_slice_prior_match_tolerance = _safe_float(
        recipe.get(
            "direction_slice_prior_match_tolerance",
            meta.get("direction_slice_prior_match_tolerance", -1.0),
        )
    )
    direction_slice_prior_match_min_label_rate = _safe_float(
        recipe.get(
            "direction_slice_prior_match_min_label_rate",
            meta.get("direction_slice_prior_match_min_label_rate", -1.0),
        )
    )
    direction_slice_prior_match_min_rows = int(
        _safe_float(
            recipe.get(
                "direction_slice_prior_match_min_rows",
                meta.get("direction_slice_prior_match_min_rows", -1.0),
            )
        )
    )
    direction_slice_balanced_sampler = _bool_value(
        recipe.get(
            "direction_slice_balanced_sampler",
            meta.get("direction_slice_balanced_sampler", False),
        )
    )
    direction_slice_balanced_sampler_min_rows = int(
        _safe_float(
            recipe.get(
                "direction_slice_balanced_sampler_min_rows",
                meta.get("direction_slice_balanced_sampler_min_rows", -1.0),
            )
        )
    )
    direction_slice_hard_red_stop_patience = int(
        _safe_float(
            recipe.get(
                "direction_slice_hard_red_stop_patience",
                meta.get("direction_slice_hard_red_stop_patience", -1.0),
            )
        )
    )
    direction_slice_hard_red_stop_min_epochs = int(
        _safe_float(
            recipe.get(
                "direction_slice_hard_red_stop_min_epochs",
                meta.get("direction_slice_hard_red_stop_min_epochs", -1.0),
            )
        )
    )
    direction_vs_flat_margin_weight = _safe_float(
        recipe.get(
            "direction_vs_flat_margin_weight",
            meta.get("direction_vs_flat_margin_weight", 0.0),
        )
    )
    direction_vs_flat_margin = _safe_float(
        recipe.get(
            "direction_vs_flat_margin",
            meta.get("direction_vs_flat_margin", 0.0),
        )
    )
    direction_utility_margin_weight = _safe_float(
        recipe.get(
            "direction_utility_margin_weight",
            meta.get("direction_utility_margin_weight", 0.0),
        )
    )
    direction_utility_min_gap_bps = _safe_float(
        recipe.get(
            "direction_utility_min_gap_bps",
            meta.get("direction_utility_min_gap_bps", 999.0),
        )
    )
    direction_utility_logit_margin = _safe_float(
        recipe.get(
            "direction_utility_logit_margin",
            meta.get("direction_utility_logit_margin", 0.0),
        )
    )
    direction_side_utility_conviction_weight = _safe_float(
        recipe.get(
            "direction_side_utility_conviction_weight",
            meta.get("direction_side_utility_conviction_weight", 0.0),
        )
    )
    direction_side_utility_conviction_min_gap_bps = _safe_float(
        recipe.get(
            "direction_side_utility_conviction_min_gap_bps",
            meta.get("direction_side_utility_conviction_min_gap_bps", 999.0),
        )
    )
    direction_side_utility_conviction_logit_margin = _safe_float(
        recipe.get(
            "direction_side_utility_conviction_logit_margin",
            meta.get("direction_side_utility_conviction_logit_margin", 0.0),
        )
    )
    direction_utility_trade_conviction_weight = _safe_float(
        recipe.get(
            "direction_utility_trade_conviction_weight",
            meta.get("direction_utility_trade_conviction_weight", 0.0),
        )
    )
    direction_utility_trade_conviction_min_gap_bps = _safe_float(
        recipe.get(
            "direction_utility_trade_conviction_min_gap_bps",
            meta.get("direction_utility_trade_conviction_min_gap_bps", 999.0),
        )
    )
    direction_utility_trade_conviction_min_utility_bps = _safe_float(
        recipe.get(
            "direction_utility_trade_conviction_min_utility_bps",
            meta.get("direction_utility_trade_conviction_min_utility_bps", 999.0),
        )
    )
    direction_utility_trade_conviction_max_bad_path = _safe_float(
        recipe.get(
            "direction_utility_trade_conviction_max_bad_path",
            meta.get("direction_utility_trade_conviction_max_bad_path", 999.0),
        )
    )
    direction_utility_trade_conviction_logit_margin = _safe_float(
        recipe.get(
            "direction_utility_trade_conviction_logit_margin",
            meta.get("direction_utility_trade_conviction_logit_margin", 0.0),
        )
    )
    direction_utility_triad_ce_weight = _safe_float(
        recipe.get(
            "direction_utility_triad_ce_weight",
            meta.get("direction_utility_triad_ce_weight", 0.0),
        )
    )
    direction_utility_triad_ce_min_gap_bps = _safe_float(
        recipe.get(
            "direction_utility_triad_ce_min_gap_bps",
            meta.get("direction_utility_triad_ce_min_gap_bps", 999.0),
        )
    )
    direction_utility_triad_ce_min_utility_bps = _safe_float(
        recipe.get(
            "direction_utility_triad_ce_min_utility_bps",
            meta.get("direction_utility_triad_ce_min_utility_bps", 999.0),
        )
    )
    direction_utility_triad_ce_max_bad_path = _safe_float(
        recipe.get(
            "direction_utility_triad_ce_max_bad_path",
            meta.get("direction_utility_triad_ce_max_bad_path", 999.0),
        )
    )
    direction_utility_triad_ce_class_weight_cap = _safe_float(
        recipe.get(
            "direction_utility_triad_ce_class_weight_cap",
            meta.get("direction_utility_triad_ce_class_weight_cap", 0.0),
        )
    )
    _hierarchical_direction_meta = (
        meta.get("hierarchical_direction_composition")
        if isinstance(meta.get("hierarchical_direction_composition"), dict)
        else {}
    )
    direction_hierarchical_composition = bool(
        recipe.get(
            "direction_hierarchical_composition",
            _hierarchical_direction_meta.get("enabled", False),
        )
    )
    _hier_entry_meta = meta.get("hierarchical_entry_heads") if isinstance(meta.get("hierarchical_entry_heads"), dict) else {}
    _hier_slice_side_meta = (
        _hier_entry_meta.get("slice_side_supervision")
        if isinstance(_hier_entry_meta.get("slice_side_supervision"), dict)
        else {}
    )
    hier_slice_side_ce_weight = _safe_float(
        recipe.get(
            "hier_slice_side_ce_weight",
            _hier_slice_side_meta.get(
                "balanced_ce_weight",
                meta.get("hier_slice_side_ce_weight", 0.0),
            ),
        )
    )
    hier_slice_side_true_margin_weight = _safe_float(
        recipe.get(
            "hier_slice_side_true_margin_weight",
            _hier_slice_side_meta.get(
                "true_margin_weight",
                meta.get("hier_slice_side_true_margin_weight", 0.0),
            ),
        )
    )
    hier_slice_side_true_margin = _safe_float(
        recipe.get(
            "hier_slice_side_true_margin",
            _hier_slice_side_meta.get(
                "true_margin",
                meta.get("hier_slice_side_true_margin", 0.0),
            ),
        )
    )
    hier_slice_side_min_label_rate = _safe_float(
        recipe.get(
            "hier_slice_side_min_label_rate",
            _hier_slice_side_meta.get(
                "min_label_rate",
                meta.get("hier_slice_side_min_label_rate", 0.0),
            ),
        )
    )
    hier_slice_side_min_rows = int(
        _safe_float(
            recipe.get(
                "hier_slice_side_min_rows",
                _hier_slice_side_meta.get(
                    "min_rows",
                    meta.get("hier_slice_side_min_rows", -1.0),
                ),
            )
        )
    )
    direction_flat_starvation_weight = _safe_float(
        recipe.get(
            "direction_flat_starvation_weight",
            meta.get("direction_flat_starvation_weight", 0.0),
        )
    )
    direction_flat_starvation_min_label_rate = _safe_float(
        recipe.get(
            "direction_flat_starvation_min_label_rate",
            meta.get("direction_flat_starvation_min_label_rate", 0.0),
        )
    )
    direction_flat_starvation_min_rows = int(
        _safe_float(
            recipe.get(
                "direction_flat_starvation_min_rows",
                meta.get("direction_flat_starvation_min_rows", -1.0),
            )
        )
    )
    direction_flat_starvation_pred_fraction = _safe_float(
        recipe.get(
            "direction_flat_starvation_pred_fraction",
            meta.get("direction_flat_starvation_pred_fraction", 0.0),
        )
    )
    direction_flat_starvation_pred_floor = _safe_float(
        recipe.get(
            "direction_flat_starvation_pred_floor",
            meta.get("direction_flat_starvation_pred_floor", 0.0),
        )
    )
    direction_flat_starvation_logit_margin = _safe_float(
        recipe.get(
            "direction_flat_starvation_logit_margin",
            meta.get("direction_flat_starvation_logit_margin", 0.0),
        )
    )
    best_direction_balance_guard_ok = meta.get("best_direction_balance_guard_ok")
    best_direction_slice_contract_ok = meta.get("best_direction_slice_contract_ok")
    ckpt_balance_guard_required = (
        ckpt_class_balance_guard_weight > 0.0
        and (
            ckpt_class_balance_min_pred_to_label > 0.0
            or ckpt_class_balance_min_pred_rate > 0.0
        )
    )
    failures: list[str] = []
    if "direction" in active_heads:
        if pred_balance_alpha < 0.05:
            failures.append("direction active head requires pred_balance_alpha >= 0.05")
        if pred_balance_alpha > 0.50:
            failures.append("direction active head requires pred_balance_alpha <= 0.50")
        if pred_balance_target != "label":
            failures.append("direction active head requires pred_balance_target=label")
        if direction_ce_scale <= 0.0:
            failures.append("direction active head requires positive direction_ce_scale")
        if ckpt_monitor != "dir_acc":
            failures.append("direction active head requires ckpt_monitor=dir_acc")
        if contract_mode == "smart_seq520_candidate":
            if pred_balance_alpha < SMART_DIRECTION_BALANCE_MIN_ALPHA:
                failures.append(
                    "smart direction active head requires pred_balance_alpha >= "
                    f"{SMART_DIRECTION_BALANCE_MIN_ALPHA:.2f}"
                )
            if direction_ce_scale < SMART_DIRECTION_CE_SCALE_MIN:
                failures.append(
                    "smart direction active head requires direction_ce_scale >= "
                    f"{SMART_DIRECTION_CE_SCALE_MIN:.2f}"
                )
            if pred_balance_class_weights != SMART_DIRECTION_BALANCE_CLASS_WEIGHTS:
                failures.append(
                    "smart direction active head requires pred_balance_class_weights="
                    + ",".join(str(value) for value in SMART_DIRECTION_BALANCE_CLASS_WEIGHTS)
            )
            if not hierarchical_entry_heads_enabled:
                failures.append("smart direction active head requires hierarchical_entry_heads_enabled=true")
            if not side_validity_head_enabled:
                failures.append("smart direction active head requires side_validity_head_enabled=true")
            if hier_side_validity_weight < SMART_DIRECTION_SIDE_VALIDITY_WEIGHT_MIN:
                failures.append(
                    "smart direction active head requires hier_side_validity_weight >= "
                    f"{SMART_DIRECTION_SIDE_VALIDITY_WEIGHT_MIN:.2f}"
                )
            if not trendline_rail_head_enabled:
                failures.append("smart direction active head requires trendline_rail_head_enabled=true")
            if trendline_rail_aux_weight < SMART_DIRECTION_TRENDLINE_RAIL_AUX_WEIGHT_MIN:
                failures.append(
                    "smart direction active head requires trendline_rail_aux_weight >= "
                    f"{SMART_DIRECTION_TRENDLINE_RAIL_AUX_WEIGHT_MIN:.2f}"
                )
            if trendline_rail_wrong_side_weight < SMART_DIRECTION_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT_MIN:
                failures.append(
                    "smart direction active head requires trendline_rail_wrong_side_weight >= "
                    f"{SMART_DIRECTION_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT_MIN:.2f}"
                )
            if hier_legacy_ce_mult < SMART_DIRECTION_HIER_LEGACY_CE_MULT_MIN:
                failures.append(
                    "smart direction active head requires hier_legacy_ce_mult >= "
                    f"{SMART_DIRECTION_HIER_LEGACY_CE_MULT_MIN:.2f}"
                )
            if abs(residual_scale - SMART_DIRECTION_RESIDUAL_SCALE) > 1e-12:
                failures.append(
                    "smart direction active head requires residual_scale="
                    f"{SMART_DIRECTION_RESIDUAL_SCALE:.2f}"
                )
            if abs(anchor_eps - SMART_DIRECTION_ANCHOR_EPS) > 1e-18:
                failures.append(
                    "smart direction active head requires anchor_eps="
                    f"{SMART_DIRECTION_ANCHOR_EPS:.0e}"
                )
            if not anchor_gate_enabled:
                failures.append("smart direction active head requires anchor_gate_enabled=true")
            if anchor_gate_init > SMART_DIRECTION_ANCHOR_GATE_INIT_MAX:
                failures.append(
                    "smart direction active head requires anchor_gate_init <= "
                    f"{SMART_DIRECTION_ANCHOR_GATE_INIT_MAX:.2f}"
                )
            if ckpt_class_balance_guard_weight < SMART_DIRECTION_CKPT_BALANCE_GUARD_MIN_WEIGHT:
                failures.append(
                    "smart direction active head requires ckpt_class_balance_guard_weight >= "
                    f"{SMART_DIRECTION_CKPT_BALANCE_GUARD_MIN_WEIGHT:.2f}"
                )
            if ckpt_class_balance_min_pred_to_label < SMART_DIRECTION_CKPT_BALANCE_MIN_PRED_TO_LABEL:
                failures.append(
                    "smart direction active head requires ckpt_class_balance_min_pred_to_label >= "
                    f"{SMART_DIRECTION_CKPT_BALANCE_MIN_PRED_TO_LABEL:.2f}"
                )
            if ckpt_class_balance_min_pred_rate < SMART_DIRECTION_CKPT_BALANCE_MIN_PRED_RATE:
                failures.append(
                    "smart direction active head requires ckpt_class_balance_min_pred_rate >= "
                    f"{SMART_DIRECTION_CKPT_BALANCE_MIN_PRED_RATE:.2f}"
                )
            if not ckpt_direction_slice_guard:
                failures.append("smart direction active head requires ckpt_direction_slice_guard=true")
            if direction_min_pred_rate_loss_weight < SMART_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT:
                failures.append(
                    "smart direction active head requires direction_min_pred_rate_loss_weight >= "
                    f"{SMART_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT:.2f}"
                )
            if direction_min_pred_rate_fraction < SMART_DIRECTION_MIN_PRED_RATE_FRACTION:
                failures.append(
                    "smart direction active head requires direction_min_pred_rate_fraction >= "
                    f"{SMART_DIRECTION_MIN_PRED_RATE_FRACTION:.2f}"
                )
            if direction_min_pred_rate_floor < SMART_DIRECTION_MIN_PRED_RATE_FLOOR:
                failures.append(
                    "smart direction active head requires direction_min_pred_rate_floor >= "
                    f"{SMART_DIRECTION_MIN_PRED_RATE_FLOOR:.2f}"
                )
            if not (
                0.0
                < direction_min_pred_rate_softmax_temperature
                <= SMART_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE_MAX
            ):
                failures.append(
                    "smart direction active head requires direction_min_pred_rate_softmax_temperature "
                    f"in (0.0, {SMART_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE_MAX:.2f}]"
                )
            if direction_global_prior_match_weight < SMART_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT:
                failures.append(
                    "smart direction active head requires direction_global_prior_match_weight >= "
                    f"{SMART_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT:.2f}"
                )
            if direction_global_prior_match_tolerance > SMART_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE:
                failures.append(
                    "smart direction active head requires direction_global_prior_match_tolerance <= "
                    f"{SMART_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE:.2f}"
                )
            if (
                abs(
                    direction_global_prior_match_min_label_rate
                    - SMART_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE
                )
                > 1e-12
            ):
                failures.append(
                    "smart direction active head requires direction_global_prior_match_min_label_rate="
                    f"{SMART_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE:.2f}"
                )
            if direction_slice_balanced_ce_weight < SMART_DIRECTION_SLICE_BALANCED_CE_WEIGHT:
                failures.append(
                    "smart direction active head requires direction_slice_balanced_ce_weight >= "
                    f"{SMART_DIRECTION_SLICE_BALANCED_CE_WEIGHT:.2f}"
                )
            if (
                abs(
                    direction_slice_balanced_ce_min_label_rate
                    - SMART_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE
                )
                > 1e-12
            ):
                failures.append(
                    "smart direction active head requires direction_slice_balanced_ce_min_label_rate="
                    f"{SMART_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE:.2f}"
                )
            if direction_slice_balanced_ce_min_rows != SMART_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS:
                failures.append(
                    "smart direction active head requires direction_slice_balanced_ce_min_rows="
                    f"{SMART_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS}"
                )
            if direction_slice_true_margin_weight < SMART_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT:
                failures.append(
                    "smart direction active head requires direction_slice_true_margin_weight >= "
                    f"{SMART_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT:.2f}"
                )
            if direction_slice_true_margin < SMART_DIRECTION_SLICE_TRUE_MARGIN:
                failures.append(
                    "smart direction active head requires direction_slice_true_margin >= "
                    f"{SMART_DIRECTION_SLICE_TRUE_MARGIN:.2f}"
                )
            if (
                abs(
                    direction_slice_true_margin_min_label_rate
                    - SMART_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE
                )
                > 1e-12
            ):
                failures.append(
                    "smart direction active head requires direction_slice_true_margin_min_label_rate="
                    f"{SMART_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE:.2f}"
                )
            if direction_slice_true_margin_min_rows != SMART_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS:
                failures.append(
                    "smart direction active head requires direction_slice_true_margin_min_rows="
                    f"{SMART_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS}"
                )
            if direction_slice_accuracy_edge_weight < SMART_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT:
                failures.append(
                    "smart direction active head requires direction_slice_accuracy_edge_weight >= "
                    f"{SMART_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT:.2f}"
                )
            if direction_slice_accuracy_edge_margin < SMART_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN:
                failures.append(
                    "smart direction active head requires direction_slice_accuracy_edge_margin >= "
                    f"{SMART_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN:.2f}"
                )
            if (
                abs(
                    direction_slice_accuracy_edge_min_label_rate
                    - SMART_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE
                )
                > 1e-12
            ):
                failures.append(
                    "smart direction active head requires direction_slice_accuracy_edge_min_label_rate="
                    f"{SMART_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE:.2f}"
                )
            if direction_slice_accuracy_edge_min_rows != SMART_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS:
                failures.append(
                    "smart direction active head requires direction_slice_accuracy_edge_min_rows="
                    f"{SMART_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS}"
                )
            if direction_slice_prior_match_weight < SMART_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT:
                failures.append(
                    "smart direction active head requires direction_slice_prior_match_weight >= "
                    f"{SMART_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT:.2f}"
                )
            if direction_slice_prior_match_tolerance > SMART_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE:
                failures.append(
                    "smart direction active head requires direction_slice_prior_match_tolerance <= "
                    f"{SMART_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE:.2f}"
                )
            if (
                abs(
                    direction_slice_prior_match_min_label_rate
                    - SMART_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE
                )
                > 1e-12
            ):
                failures.append(
                    "smart direction active head requires direction_slice_prior_match_min_label_rate="
                    f"{SMART_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE:.2f}"
                )
            if direction_slice_prior_match_min_rows != SMART_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS:
                failures.append(
                    "smart direction active head requires direction_slice_prior_match_min_rows="
                    f"{SMART_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS}"
                )
            if not bool(direction_slice_balanced_sampler):
                failures.append(
                    "smart direction active head requires direction_slice_balanced_sampler=true"
                )
            if direction_slice_balanced_sampler_min_rows != SMART_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS:
                failures.append(
                    "smart direction active head requires direction_slice_balanced_sampler_min_rows="
                    f"{SMART_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS}"
                )
            if direction_slice_hard_red_stop_patience != SMART_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE:
                failures.append(
                    "smart direction active head requires direction_slice_hard_red_stop_patience="
                    f"{SMART_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE}"
                )
            if direction_slice_hard_red_stop_min_epochs != SMART_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS:
                failures.append(
                    "smart direction active head requires direction_slice_hard_red_stop_min_epochs="
                    f"{SMART_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS}"
                )
            if direction_vs_flat_margin_weight < SMART_DIRECTION_VS_FLAT_MARGIN_WEIGHT:
                failures.append(
                    "smart direction active head requires direction_vs_flat_margin_weight >= "
                    f"{SMART_DIRECTION_VS_FLAT_MARGIN_WEIGHT:.2f}"
                )
            if direction_vs_flat_margin < SMART_DIRECTION_VS_FLAT_MARGIN:
                failures.append(
                    "smart direction active head requires direction_vs_flat_margin >= "
                    f"{SMART_DIRECTION_VS_FLAT_MARGIN:.2f}"
                )
            if direction_utility_margin_weight < SMART_DIRECTION_UTILITY_MARGIN_WEIGHT:
                failures.append(
                    "smart direction active head requires direction_utility_margin_weight >= "
                    f"{SMART_DIRECTION_UTILITY_MARGIN_WEIGHT:.2f}"
                )
            if direction_utility_min_gap_bps > SMART_DIRECTION_UTILITY_MIN_GAP_BPS_MAX:
                failures.append(
                    "smart direction active head requires direction_utility_min_gap_bps <= "
                    f"{SMART_DIRECTION_UTILITY_MIN_GAP_BPS_MAX:.1f}"
                )
            if direction_utility_logit_margin < SMART_DIRECTION_UTILITY_LOGIT_MARGIN:
                failures.append(
                    "smart direction active head requires direction_utility_logit_margin >= "
                    f"{SMART_DIRECTION_UTILITY_LOGIT_MARGIN:.2f}"
                )
            if direction_side_utility_conviction_weight < SMART_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT:
                failures.append(
                    "smart direction active head requires direction_side_utility_conviction_weight >= "
                    f"{SMART_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT:.2f}"
                )
            if (
                direction_side_utility_conviction_min_gap_bps
                > SMART_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS_MAX
            ):
                failures.append(
                    "smart direction active head requires direction_side_utility_conviction_min_gap_bps <= "
                    f"{SMART_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS_MAX:.1f}"
                )
            if (
                direction_side_utility_conviction_logit_margin
                < SMART_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN
            ):
                failures.append(
                    "smart direction active head requires direction_side_utility_conviction_logit_margin >= "
                    f"{SMART_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN:.2f}"
                )
            if direction_utility_trade_conviction_weight < SMART_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT:
                failures.append(
                    "smart direction active head requires direction_utility_trade_conviction_weight >= "
                    f"{SMART_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT:.2f}"
                )
            if (
                direction_utility_trade_conviction_min_gap_bps
                > SMART_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS_MAX
            ):
                failures.append(
                    "smart direction active head requires direction_utility_trade_conviction_min_gap_bps <= "
                    f"{SMART_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS_MAX:.1f}"
                )
            if (
                direction_utility_trade_conviction_min_utility_bps
                > SMART_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS_MAX
            ):
                failures.append(
                    "smart direction active head requires direction_utility_trade_conviction_min_utility_bps <= "
                    f"{SMART_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS_MAX:.1f}"
                )
            if (
                direction_utility_trade_conviction_max_bad_path
                > SMART_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH_MAX
            ):
                failures.append(
                    "smart direction active head requires direction_utility_trade_conviction_max_bad_path <= "
                    f"{SMART_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH_MAX:.2f}"
                )
            if (
                direction_utility_trade_conviction_logit_margin
                < SMART_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN
            ):
                failures.append(
                    "smart direction active head requires direction_utility_trade_conviction_logit_margin >= "
                    f"{SMART_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN:.2f}"
                )
            if direction_utility_triad_ce_weight < SMART_DIRECTION_UTILITY_TRIAD_CE_WEIGHT:
                failures.append(
                    "smart direction active head requires direction_utility_triad_ce_weight >= "
                    f"{SMART_DIRECTION_UTILITY_TRIAD_CE_WEIGHT:.2f}"
                )
            if direction_utility_triad_ce_min_gap_bps > SMART_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS_MAX:
                failures.append(
                    "smart direction active head requires direction_utility_triad_ce_min_gap_bps <= "
                    f"{SMART_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS_MAX:.1f}"
                )
            if (
                direction_utility_triad_ce_min_utility_bps
                > SMART_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS_MAX
            ):
                failures.append(
                    "smart direction active head requires direction_utility_triad_ce_min_utility_bps <= "
                    f"{SMART_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS_MAX:.1f}"
                )
            if direction_utility_triad_ce_max_bad_path > SMART_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH_MAX:
                failures.append(
                    "smart direction active head requires direction_utility_triad_ce_max_bad_path <= "
                    f"{SMART_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH_MAX:.2f}"
                )
            if direction_utility_triad_ce_class_weight_cap < SMART_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP_MIN:
                failures.append(
                    "smart direction active head requires direction_utility_triad_ce_class_weight_cap >= "
                    f"{SMART_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP_MIN:.1f}"
                )
            if direction_hierarchical_composition is not SMART_DIRECTION_HIERARCHICAL_COMPOSITION_REQUIRED:
                failures.append("smart direction active head requires direction_hierarchical_composition=true")
            if hier_slice_side_ce_weight < SMART_DIRECTION_HIER_SLICE_SIDE_CE_WEIGHT:
                failures.append(
                    "smart direction active head requires hier_slice_side_ce_weight >= "
                    f"{SMART_DIRECTION_HIER_SLICE_SIDE_CE_WEIGHT:.2f}"
                )
            if hier_slice_side_true_margin_weight < SMART_DIRECTION_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT:
                failures.append(
                    "smart direction active head requires hier_slice_side_true_margin_weight >= "
                    f"{SMART_DIRECTION_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT:.2f}"
                )
            if hier_slice_side_true_margin < SMART_DIRECTION_HIER_SLICE_SIDE_TRUE_MARGIN:
                failures.append(
                    "smart direction active head requires hier_slice_side_true_margin >= "
                    f"{SMART_DIRECTION_HIER_SLICE_SIDE_TRUE_MARGIN:.2f}"
                )
            if hier_slice_side_min_label_rate < SMART_DIRECTION_HIER_SLICE_SIDE_MIN_LABEL_RATE:
                failures.append(
                    "smart direction active head requires hier_slice_side_min_label_rate >= "
                    f"{SMART_DIRECTION_HIER_SLICE_SIDE_MIN_LABEL_RATE:.2f}"
                )
            if hier_slice_side_min_rows < SMART_DIRECTION_HIER_SLICE_SIDE_MIN_ROWS:
                failures.append(
                    "smart direction active head requires hier_slice_side_min_rows >= "
                    f"{SMART_DIRECTION_HIER_SLICE_SIDE_MIN_ROWS}"
                )
            if direction_flat_starvation_weight < SMART_DIRECTION_FLAT_STARVATION_WEIGHT:
                failures.append(
                    "smart direction active head requires direction_flat_starvation_weight >= "
                    f"{SMART_DIRECTION_FLAT_STARVATION_WEIGHT:.2f}"
                )
            if direction_flat_starvation_min_label_rate < SMART_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE:
                failures.append(
                    "smart direction active head requires direction_flat_starvation_min_label_rate >= "
                    f"{SMART_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE:.2f}"
                )
            if direction_flat_starvation_min_rows < SMART_DIRECTION_FLAT_STARVATION_MIN_ROWS:
                failures.append(
                    "smart direction active head requires direction_flat_starvation_min_rows >= "
                    f"{SMART_DIRECTION_FLAT_STARVATION_MIN_ROWS}"
                )
            if direction_flat_starvation_pred_fraction < SMART_DIRECTION_FLAT_STARVATION_PRED_FRACTION:
                failures.append(
                    "smart direction active head requires direction_flat_starvation_pred_fraction >= "
                    f"{SMART_DIRECTION_FLAT_STARVATION_PRED_FRACTION:.2f}"
                )
            if direction_flat_starvation_pred_floor < SMART_DIRECTION_FLAT_STARVATION_PRED_FLOOR:
                failures.append(
                    "smart direction active head requires direction_flat_starvation_pred_floor >= "
                    f"{SMART_DIRECTION_FLAT_STARVATION_PRED_FLOOR:.2f}"
                )
            if direction_flat_starvation_logit_margin < SMART_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN:
                failures.append(
                    "smart direction active head requires direction_flat_starvation_logit_margin >= "
                    f"{SMART_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN:.2f}"
                )
            if enable_mtf_direction_head and not mtf_dir_aux_weight_present:
                failures.append(
                    "smart direction active head with MTF aux head enabled requires "
                    "mtf_dir_aux_weight present in bundle metadata (missing key would "
                    "silently default to 0.0 and skip the balance-repair requirement)"
                )
            if mtf_dir_aux_balance_repair_required and not mtf_dir_aux_uses_direction_balance_repair:
                failures.append(
                    "smart direction active head with MTF aux requires "
                    "mtf_dir_aux_uses_direction_balance_repair=true"
                )
        if ckpt_balance_guard_required and best_direction_balance_guard_ok is not True:
            failures.append("direction active head requires best_direction_balance_guard_ok=true")
        if ckpt_direction_slice_guard and best_direction_slice_contract_ok is not True:
            failures.append("direction active head requires best_direction_slice_contract_ok=true")
    return {
        "decision": "PASS" if not failures else "FAIL",
        "active_heads": sorted(active_heads),
        "direction_active": "direction" in active_heads,
        "contract_mode": contract_mode,
        "pred_balance_alpha": pred_balance_alpha,
        "pred_balance_target": pred_balance_target,
        "pred_balance_class_weights": pred_balance_class_weights,
        "enable_mtf_direction_head": enable_mtf_direction_head,
        "mtf_dir_aux_weight": mtf_dir_aux_weight,
        "mtf_dir_aux_balance_repair_required": mtf_dir_aux_balance_repair_required,
        "mtf_dir_aux_uses_direction_balance_repair": mtf_dir_aux_uses_direction_balance_repair,
        "direction_ce_scale": direction_ce_scale,
        "hierarchical_entry_heads_enabled": hierarchical_entry_heads_enabled,
        "side_validity_head_enabled": side_validity_head_enabled,
        "hier_side_validity_weight": hier_side_validity_weight,
        "hier_side_validity_min_utility_bps": hier_side_validity_min_utility_bps,
        "hier_side_validity_pos_weight_cap": hier_side_validity_pos_weight_cap,
        "trendline_rail_head_enabled": trendline_rail_head_enabled,
        "trendline_rail_aux_weight": trendline_rail_aux_weight,
        "trendline_rail_wrong_side_weight": trendline_rail_wrong_side_weight,
        "hier_legacy_ce_mult": hier_legacy_ce_mult,
        "residual_scale": residual_scale,
        "anchor_eps": anchor_eps,
        "anchor_gate_enabled": anchor_gate_enabled,
        "anchor_gate_init": anchor_gate_init,
        "ckpt_monitor": ckpt_monitor,
        "ckpt_class_balance_guard_weight": ckpt_class_balance_guard_weight,
        "ckpt_class_balance_min_pred_to_label": ckpt_class_balance_min_pred_to_label,
        "ckpt_class_balance_min_pred_rate": ckpt_class_balance_min_pred_rate,
        "ckpt_direction_slice_guard": ckpt_direction_slice_guard,
        "direction_min_pred_rate_loss_weight": direction_min_pred_rate_loss_weight,
        "direction_min_pred_rate_fraction": direction_min_pred_rate_fraction,
        "direction_min_pred_rate_floor": direction_min_pred_rate_floor,
        "direction_min_pred_rate_softmax_temperature": direction_min_pred_rate_softmax_temperature,
        "direction_global_prior_match_weight": direction_global_prior_match_weight,
        "direction_global_prior_match_tolerance": direction_global_prior_match_tolerance,
        "direction_global_prior_match_min_label_rate": direction_global_prior_match_min_label_rate,
        "direction_slice_balanced_ce_weight": direction_slice_balanced_ce_weight,
        "direction_slice_balanced_ce_min_label_rate": direction_slice_balanced_ce_min_label_rate,
        "direction_slice_balanced_ce_min_rows": direction_slice_balanced_ce_min_rows,
        "direction_slice_true_margin_weight": direction_slice_true_margin_weight,
        "direction_slice_true_margin": direction_slice_true_margin,
        "direction_slice_true_margin_min_label_rate": direction_slice_true_margin_min_label_rate,
        "direction_slice_true_margin_min_rows": direction_slice_true_margin_min_rows,
        "direction_slice_accuracy_edge_weight": direction_slice_accuracy_edge_weight,
        "direction_slice_accuracy_edge_margin": direction_slice_accuracy_edge_margin,
        "direction_slice_accuracy_edge_min_label_rate": direction_slice_accuracy_edge_min_label_rate,
        "direction_slice_accuracy_edge_min_rows": direction_slice_accuracy_edge_min_rows,
        "direction_slice_prior_match_weight": direction_slice_prior_match_weight,
        "direction_slice_prior_match_tolerance": direction_slice_prior_match_tolerance,
        "direction_slice_prior_match_min_label_rate": direction_slice_prior_match_min_label_rate,
        "direction_slice_prior_match_min_rows": direction_slice_prior_match_min_rows,
        "direction_slice_balanced_sampler": direction_slice_balanced_sampler,
        "direction_slice_balanced_sampler_min_rows": direction_slice_balanced_sampler_min_rows,
        "direction_slice_hard_red_stop_patience": direction_slice_hard_red_stop_patience,
        "direction_slice_hard_red_stop_min_epochs": direction_slice_hard_red_stop_min_epochs,
        "direction_vs_flat_margin_weight": direction_vs_flat_margin_weight,
        "direction_vs_flat_margin": direction_vs_flat_margin,
        "direction_utility_margin_weight": direction_utility_margin_weight,
        "direction_utility_min_gap_bps": direction_utility_min_gap_bps,
        "direction_utility_logit_margin": direction_utility_logit_margin,
        "direction_side_utility_conviction_weight": direction_side_utility_conviction_weight,
        "direction_side_utility_conviction_min_gap_bps": direction_side_utility_conviction_min_gap_bps,
        "direction_side_utility_conviction_logit_margin": direction_side_utility_conviction_logit_margin,
        "direction_utility_trade_conviction_weight": direction_utility_trade_conviction_weight,
        "direction_utility_trade_conviction_min_gap_bps": direction_utility_trade_conviction_min_gap_bps,
        "direction_utility_trade_conviction_min_utility_bps": direction_utility_trade_conviction_min_utility_bps,
        "direction_utility_trade_conviction_max_bad_path": direction_utility_trade_conviction_max_bad_path,
        "direction_utility_trade_conviction_logit_margin": direction_utility_trade_conviction_logit_margin,
        "direction_utility_triad_ce_weight": direction_utility_triad_ce_weight,
        "direction_utility_triad_ce_min_gap_bps": direction_utility_triad_ce_min_gap_bps,
        "direction_utility_triad_ce_min_utility_bps": direction_utility_triad_ce_min_utility_bps,
        "direction_utility_triad_ce_max_bad_path": direction_utility_triad_ce_max_bad_path,
        "direction_utility_triad_ce_class_weight_cap": direction_utility_triad_ce_class_weight_cap,
        "direction_hierarchical_composition": direction_hierarchical_composition,
        "hier_slice_side_ce_weight": hier_slice_side_ce_weight,
        "hier_slice_side_true_margin_weight": hier_slice_side_true_margin_weight,
        "hier_slice_side_true_margin": hier_slice_side_true_margin,
        "hier_slice_side_min_label_rate": hier_slice_side_min_label_rate,
        "hier_slice_side_min_rows": hier_slice_side_min_rows,
        "direction_flat_starvation_weight": direction_flat_starvation_weight,
        "direction_flat_starvation_min_label_rate": direction_flat_starvation_min_label_rate,
        "direction_flat_starvation_min_rows": direction_flat_starvation_min_rows,
        "direction_flat_starvation_pred_fraction": direction_flat_starvation_pred_fraction,
        "direction_flat_starvation_pred_floor": direction_flat_starvation_pred_floor,
        "direction_flat_starvation_logit_margin": direction_flat_starvation_logit_margin,
        "ckpt_balance_guard_required": ckpt_balance_guard_required,
        "best_direction_balance_guard_ok": best_direction_balance_guard_ok,
        "best_direction_slice_contract_ok": best_direction_slice_contract_ok,
        "failures": failures,
    }


def _tail_direction_recipe_contract(meta: dict[str, Any], capabilities: dict[str, Any]) -> dict[str, Any]:
    recipe = meta.get("train_recipe") if isinstance(meta.get("train_recipe"), dict) else {}
    active_heads = {
        str(head)
        for head in (
            recipe.get("active_heads")
            or capabilities.get("declared_active_heads")
            or capabilities.get("supported_heads")
            or []
        )
        if str(head)
    }
    weight = _safe_float(recipe.get("tail_direction_ce_weight", meta.get("tail_direction_ce_weight", 0.0)))
    quantile = _safe_float(
        recipe.get("tail_direction_quality_quantile", meta.get("tail_direction_quality_quantile", 0.0))
    )
    min_batch = int(_safe_float(recipe.get("tail_direction_min_batch", meta.get("tail_direction_min_batch", 0))))
    failures: list[str] = []
    if "direction" in active_heads:
        if weight <= 0.0:
            failures.append("direction active head requires positive tail_direction_ce_weight")
        if quantile < 0.50 or quantile > 0.95:
            failures.append("direction active head requires tail_direction_quality_quantile in [0.50, 0.95]")
        if min_batch < 2:
            failures.append("direction active head requires tail_direction_min_batch >= 2")
    return {
        "decision": "PASS" if not failures else "FAIL",
        "active_heads": sorted(active_heads),
        "direction_active": "direction" in active_heads,
        "tail_direction_ce_weight": weight,
        "tail_direction_quality_quantile": quantile,
        "tail_direction_min_batch": min_batch,
        "tail_direction_mask": "directional_tradable_clean_path_top_quality",
        "failures": failures,
    }


def _symmetric_validation_recipe_contract(
    meta: dict[str, Any],
    capabilities: dict[str, Any],
    *,
    contract_mode: str = "foundation_seq146",
) -> dict[str, Any]:
    recipe = meta.get("train_recipe") if isinstance(meta.get("train_recipe"), dict) else {}
    active_heads = {
        str(head)
        for head in (
            recipe.get("active_heads")
            or capabilities.get("declared_active_heads")
            or capabilities.get("supported_heads")
            or []
        )
        if str(head)
    }
    symmetric_negatives = _bool_value(recipe.get("symmetric_negatives", meta.get("symmetric_negatives", False)))
    selector_masked_aux = _bool_value(recipe.get("selector_masked_aux", meta.get("selector_masked_aux", False)))
    validation_objective_matches_train = _bool_value(
        recipe.get(
            "validation_objective_matches_train",
            meta.get("validation_objective_matches_train", False),
        )
    )
    aux_selector_mode = str(recipe.get("aux_selector_mode", meta.get("aux_selector_mode", ""))).strip()
    clean_edge_target_mode = str(recipe.get("clean_edge_target_mode", meta.get("clean_edge_target_mode", ""))).strip()
    survival_target_mode = str(recipe.get("survival_target_mode", meta.get("survival_target_mode", ""))).strip()
    bad_path_ce_in_direction_loss = _bool_value(
        recipe.get("bad_path_ce_in_direction_loss", meta.get("bad_path_ce_in_direction_loss", False))
    )
    bad_path_prob_penalty_in_validation = _bool_value(
        recipe.get(
            "bad_path_prob_penalty_in_validation",
            meta.get("bad_path_prob_penalty_in_validation", False),
        )
    )
    symmetric_short_prob_penalties = _bool_value(
        recipe.get("symmetric_short_prob_penalties", meta.get("symmetric_short_prob_penalties", False))
    )
    symmetric_clean_edge_rank = _bool_value(
        recipe.get("symmetric_clean_edge_rank", meta.get("symmetric_clean_edge_rank", False))
    )
    hierarchical_entry_heads_enabled = _bool_value(
        recipe.get("hierarchical_entry_heads_enabled", meta.get("hierarchical_entry_heads_enabled", False))
    )
    side_validity_head_enabled = _bool_value(
        recipe.get(
            "side_validity_head_enabled",
            (meta.get("hierarchical_entry_heads") or {}).get("side_validity", {}).get(
                "enabled",
                meta.get("side_validity_head_enabled", False),
            )
            if isinstance(meta.get("hierarchical_entry_heads"), dict)
            else meta.get("side_validity_head_enabled", False),
        )
    )
    trendline_rail_head_enabled = _bool_value(
        recipe.get(
            "trendline_rail_head_enabled",
            (meta.get("trendline_rail_head") or {}).get("enabled", meta.get("trendline_rail_head_enabled", False))
            if isinstance(meta.get("trendline_rail_head"), dict)
            else meta.get("trendline_rail_head_enabled", False),
        )
    )
    xau_side_specific_repair = bool(
        contract_mode == "smart_seq520_candidate"
        and (
            hierarchical_entry_heads_enabled
            or side_validity_head_enabled
            or trendline_rail_head_enabled
            or "side_validity" in active_heads
            or "trendline_rail" in active_heads
            or "trade_side_hierarchy" in active_heads
        )
    )
    expected_bad_path_prob_penalty_in_validation = not xau_side_specific_repair
    failures: list[str] = []
    if contract_mode == "smart_seq520_candidate":
        if not symmetric_negatives:
            failures.append("smart symmetric validation requires symmetric_negatives=true")
        if not selector_masked_aux:
            failures.append("smart symmetric validation requires selector_masked_aux=true")
        if not validation_objective_matches_train:
            failures.append("smart symmetric validation requires validation_objective_matches_train=true")
        if aux_selector_mode != "long_short_union":
            failures.append("smart symmetric validation requires aux_selector_mode=long_short_union")
        if clean_edge_target_mode != "bidir":
            failures.append("smart symmetric validation requires clean_edge_target_mode=bidir")
        if survival_target_mode != "bidir":
            failures.append("smart symmetric validation requires survival_target_mode=bidir")
        if "bad_path" in active_heads:
            if not bad_path_ce_in_direction_loss:
                failures.append("smart bad_path active head requires bad_path_ce_in_direction_loss=true")
            if xau_side_specific_repair and bad_path_prob_penalty_in_validation:
                failures.append(
                    "smart XAU side-specific bad_path repair requires "
                    "bad_path_prob_penalty_in_validation=false"
                )
            if (not xau_side_specific_repair) and (not bad_path_prob_penalty_in_validation):
                failures.append("smart bad_path active head requires bad_path_prob_penalty_in_validation=true")
        if not symmetric_short_prob_penalties:
            failures.append("smart symmetric validation requires symmetric_short_prob_penalties=true")
        if "clean_edge" in active_heads and not symmetric_clean_edge_rank:
            failures.append("smart clean_edge active head requires symmetric_clean_edge_rank=true")
    return {
        "decision": "PASS" if not failures else "FAIL",
        "active_heads": sorted(active_heads),
        "contract_mode": contract_mode,
        "symmetric_negatives": symmetric_negatives,
        "selector_masked_aux": selector_masked_aux,
        "validation_objective_matches_train": validation_objective_matches_train,
        "aux_selector_mode": aux_selector_mode,
        "clean_edge_target_mode": clean_edge_target_mode,
        "survival_target_mode": survival_target_mode,
        "bad_path_ce_in_direction_loss": bad_path_ce_in_direction_loss,
        "bad_path_prob_penalty_in_validation": bad_path_prob_penalty_in_validation,
        "expected_bad_path_prob_penalty_in_validation": expected_bad_path_prob_penalty_in_validation,
        "xau_side_specific_repair": xau_side_specific_repair,
        "hierarchical_entry_heads_enabled": hierarchical_entry_heads_enabled,
        "side_validity_head_enabled": side_validity_head_enabled,
        "trendline_rail_head_enabled": trendline_rail_head_enabled,
        "symmetric_short_prob_penalties": symmetric_short_prob_penalties,
        "symmetric_clean_edge_rank": symmetric_clean_edge_rank,
        "failures": failures,
    }


def _pretrain_manifest_contract_report(
    manifest_path: Path | None,
    *,
    expected_bundle_dir: Path,
    expected_dataset_dir: Path,
) -> dict[str, Any] | None:
    if manifest_path is None:
        return None
    manifest = _read_json(manifest_path)
    failures: list[str] = []
    inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
    hashes = manifest.get("artifact_sha256") if isinstance(manifest.get("artifact_sha256"), dict) else {}
    contracts = manifest.get("preflight_contracts") if isinstance(manifest.get("preflight_contracts"), dict) else {}

    schema_version = str(manifest.get("schema_version") or "")
    allowed_schema_versions = {
        "entry_foundation_smoke_train_run_manifest_v1",
        "entry_foundation_candidate_train_run_manifest_v1",
    }
    if schema_version not in allowed_schema_versions:
        failures.append(f"unexpected pretrain manifest schema_version={manifest.get('schema_version')}")
    if str(Path(str(manifest.get("out_bundle_dir") or "")).expanduser().resolve()) != str(expected_bundle_dir):
        failures.append("pretrain manifest out_bundle_dir does not match audited bundle_dir")
    dataset_input = (
        inputs.get("smoke_dataset_dir")
        or inputs.get("candidate_dataset_dir")
        or inputs.get("foundation_dataset_dir")
        or ""
    )
    if str(Path(str(dataset_input)).expanduser().resolve()) != str(expected_dataset_dir):
        failures.append("pretrain manifest smoke_dataset_dir does not match audited dataset_dir")
    if bool(manifest.get("promotion_shadow_live_allowed")) is not False:
        failures.append("pretrain manifest must keep promotion_shadow_live_allowed=false")
    if bool(manifest.get("trainer_started_by_manifest_writer")) is not False:
        failures.append("pretrain manifest writer must not mark trainer_started_by_manifest_writer=true")

    if schema_version == "entry_foundation_candidate_train_run_manifest_v1":
        expected_hash_keys = (
            "candidate_readiness",
            "smoke_bundle_audit",
            "specialist_audit",
        )
    else:
        expected_hash_keys = (
            "training_readiness",
            "feature_audit",
            "target_audit",
            "specialist_audit",
            "foundation_guardrails",
            "smoke_dataset_manifest",
            "worktree_hygiene",
        )
    missing_hashes = [key for key in expected_hash_keys if not hashes.get(key)]
    if missing_hashes:
        failures.append(f"pretrain manifest missing artifact hashes: {missing_hashes}")
    path_by_hash_key = {
        "training_readiness": inputs.get("training_readiness_json"),
        "feature_audit": inputs.get("feature_audit_json"),
        "target_audit": inputs.get("target_audit_json"),
        "specialist_audit": inputs.get("specialist_audit_json"),
        "foundation_guardrails": inputs.get("foundation_guardrails_json"),
        "smoke_dataset_manifest": inputs.get("smoke_dataset_manifest"),
        "worktree_hygiene": inputs.get("worktree_hygiene_json"),
        "candidate_readiness": inputs.get("candidate_readiness_json"),
        "smoke_bundle_audit": inputs.get("smoke_bundle_audit_json"),
    }
    hash_checks: dict[str, dict[str, Any]] = {}
    for key in expected_hash_keys:
        raw_path = path_by_hash_key.get(key)
        if not raw_path:
            failures.append(f"pretrain manifest missing input path for hash key: {key}")
            continue
        path = Path(str(raw_path)).expanduser().resolve()
        expected = str(hashes.get(key) or "")
        hash_checks[key] = _artifact_hash_check(path, expected)
        if not hash_checks[key]["ok"]:
            failures.append(f"pretrain manifest artifact hash mismatch: {key}")

    readiness = contracts.get("training_readiness") if isinstance(contracts.get("training_readiness"), dict) else {}
    candidate_readiness = (
        contracts.get("candidate_readiness")
        if isinstance(contracts.get("candidate_readiness"), dict)
        else {}
    )
    smoke_edge = contracts.get("smoke_edge_audit") if isinstance(contracts.get("smoke_edge_audit"), dict) else {}
    feature = contracts.get("feature_foundation") if isinstance(contracts.get("feature_foundation"), dict) else {}
    target = contracts.get("target_foundation") if isinstance(contracts.get("target_foundation"), dict) else {}
    specialist = contracts.get("specialist_contract") if isinstance(contracts.get("specialist_contract"), dict) else {}
    smoke_dataset = contracts.get("smoke_dataset") if isinstance(contracts.get("smoke_dataset"), dict) else {}
    worktree = contracts.get("worktree_hygiene") if isinstance(contracts.get("worktree_hygiene"), dict) else {}
    contract_mode = str(
        manifest.get("specialist_contract_mode")
        or specialist.get("contract_mode")
        or specialist.get("audit_contract_mode")
        or "foundation_seq146"
    )
    expected_required_specialists = set(required_training_specialists_for_mode(contract_mode))
    expected_model_contract = specialist_model_contract_for_mode(contract_mode)

    if schema_version == "entry_foundation_candidate_train_run_manifest_v1":
        if str(candidate_readiness.get("decision")) != "READY_FOR_CANDIDATE_TRAINING_VEDTAK":
            failures.append(
                "candidate pretrain manifest candidate_readiness decision is not READY_FOR_CANDIDATE_TRAINING_VEDTAK"
            )
        if str(candidate_readiness.get("artifact_provenance_decision")) != "PASS":
            failures.append("candidate pretrain manifest candidate_readiness artifact_provenance decision is not PASS")
        candidate_fingerprints = (
            candidate_readiness.get("artifact_fingerprints")
            if isinstance(candidate_readiness.get("artifact_fingerprints"), dict)
            else {}
        )
        if not candidate_fingerprints:
            failures.append("candidate pretrain manifest candidate_readiness artifact fingerprints are missing")
        for key in ("smoke_bundle_audit", "specialist_audit"):
            fingerprint = candidate_fingerprints.get(key)
            if not isinstance(fingerprint, dict):
                failures.append(f"candidate pretrain manifest candidate_readiness missing artifact fingerprint: {key}")
                continue
            fingerprint_path = fingerprint.get("path")
            manifest_path_for_key = path_by_hash_key.get(key)
            fingerprint_sha = str(fingerprint.get("sha256") or "")
            manifest_sha = str(hashes.get(key) or "")
            if (
                fingerprint_path
                and manifest_path_for_key
                and str(Path(str(fingerprint_path)).expanduser().resolve())
                != str(Path(str(manifest_path_for_key)).expanduser().resolve())
                and not (
                    fingerprint_sha
                    and manifest_sha
                    and fingerprint_sha == manifest_sha
                    and bool(hash_checks.get(key, {}).get("ok"))
                )
            ):
                failures.append(
                    f"candidate pretrain manifest candidate_readiness artifact fingerprint path mismatch: {key}"
                )
            if fingerprint_sha != manifest_sha:
                failures.append(
                    f"candidate pretrain manifest candidate_readiness artifact fingerprint hash mismatch: {key}"
                )
        if str(smoke_edge.get("decision")) != "PASS":
            failures.append(f"candidate pretrain smoke edge audit decision is not PASS: {smoke_edge.get('decision')}")
        smoke_edge_required_specialists = {
            str(name) for name in smoke_edge.get("required_training_specialists", []) if str(name)
        }
        smoke_edge_specialist_groups = {
            str(name) for name in smoke_edge.get("specialist_groups", []) if str(name)
        }
        if smoke_edge_required_specialists != expected_required_specialists:
            failures.append(
                "candidate pretrain smoke edge required specialists were not exact: "
                f"{sorted(smoke_edge_required_specialists)}"
            )
        if smoke_edge_specialist_groups != expected_required_specialists:
            failures.append(
                "candidate pretrain smoke edge specialist groups were not exact: "
                f"{sorted(smoke_edge_specialist_groups)}"
            )
        if str(smoke_edge.get("pretrain_manifest_contract_decision")) != "PASS":
            failures.append("candidate pretrain smoke edge audit did not validate its own pretrain manifest")
        if not bool(smoke_edge.get("feature_objective_coverage_all_present")):
            failures.append("candidate pretrain smoke edge feature objective coverage is not all-present")
        if not bool(smoke_edge.get("feature_objective_liveness_all_live")):
            failures.append("candidate pretrain smoke edge feature objective liveness is not all-live")
        if not bool(smoke_edge.get("feature_source_field_liveness_all_live")):
            failures.append("candidate pretrain smoke edge feature source-field liveness is not all-live")
        if not bool(smoke_edge.get("specialist_active_heads_match_target")):
            failures.append("candidate pretrain smoke edge specialist active heads did not match target")
        if not bool(smoke_edge.get("specialist_blocked_heads_match_target")):
            failures.append("candidate pretrain smoke edge specialist blocked heads did not match target")
        if not bool(smoke_edge.get("specialist_objective_routing_all_present_and_expected")):
            failures.append("candidate pretrain smoke edge specialist routing is not all-present")
        if not bool(smoke_edge.get("specialist_model_contract_valid")):
            failures.append("candidate pretrain smoke edge specialist model contract is not valid")
        if not bool(smoke_edge.get("specialist_model_contract_set_exact")):
            failures.append("candidate pretrain smoke edge specialist model contract set is not exact")
        if not bool(smoke_edge.get("specialist_model_contract_owned_objectives_match")):
            failures.append("candidate pretrain smoke edge specialist model contract owned objectives do not match")
        if not bool(smoke_edge.get("smoke_dataset_audit_provenance_all_artifacts_present")):
            failures.append("candidate pretrain smoke edge did not preserve smoke-dataset audit artifacts")
        if not bool(smoke_edge.get("smoke_dataset_audit_provenance_all_artifact_hashes_present")):
            failures.append("candidate pretrain smoke edge did not preserve smoke-dataset audit artifact hashes")
        if not bool(smoke_edge.get("worktree_critical_gate_review_ok")):
            failures.append("candidate pretrain smoke edge did not preserve worktree critical gate review")
    else:
        if not bool(readiness.get("foundation_contract_ready_for_smoke")):
            failures.append("pretrain manifest readiness foundation_contract_ready_for_smoke is not true")
        if str(readiness.get("artifact_provenance_decision")) != "PASS":
            failures.append(
                "pretrain manifest readiness artifact_provenance decision is not PASS"
            )
        readiness_fingerprints = (
            readiness.get("artifact_fingerprints")
            if isinstance(readiness.get("artifact_fingerprints"), dict)
            else {}
        )
        if not readiness_fingerprints:
            failures.append("pretrain manifest readiness artifact fingerprints are missing")
        for key in expected_hash_keys:
            if key == "training_readiness":
                continue
            fingerprint = readiness_fingerprints.get(key) if isinstance(readiness_fingerprints, dict) else None
            if not isinstance(fingerprint, dict):
                failures.append(f"pretrain manifest readiness missing artifact fingerprint: {key}")
                continue
            fingerprint_path = fingerprint.get("path")
            manifest_path_for_key = path_by_hash_key.get(key)
            if (
                fingerprint_path
                and manifest_path_for_key
                and str(Path(str(fingerprint_path)).expanduser().resolve())
                != str(Path(str(manifest_path_for_key)).expanduser().resolve())
            ):
                failures.append(f"pretrain manifest readiness artifact fingerprint path mismatch: {key}")
            if str(fingerprint.get("sha256") or "") != str(hashes.get(key) or ""):
                failures.append(f"pretrain manifest readiness artifact fingerprint hash mismatch: {key}")
        if str(feature.get("decision")) != "PASS":
            failures.append(f"pretrain feature contract decision is not PASS: {feature.get('decision')}")
        if not bool(feature.get("foundation_objective_coverage_all_present")):
            failures.append("pretrain feature contract objective coverage is not all-present")
        if not bool(feature.get("foundation_objective_liveness_all_live")):
            failures.append("pretrain feature contract objective liveness is not all-live")
        if not bool(feature.get("foundation_source_field_liveness_all_live")):
            failures.append("pretrain feature contract source-field liveness is not all-live")
        for row in feature.get("foundation_objective_coverage") or []:
            if int((row or {}).get("missing_count") or 0) != 0:
                failures.append(f"pretrain feature objective has missing features: {(row or {}).get('objective')}")
        for row in feature.get("foundation_objective_liveness") or []:
            if (
                int((row or {}).get("missing_count") or 0) != 0
                or int((row or {}).get("nonfinite_count") or 0) != 0
                or int((row or {}).get("near_constant_count") or 0) != 0
            ):
                failures.append(
                    f"pretrain feature objective liveness failed: "
                    f"{(row or {}).get('split')}:{(row or {}).get('objective')}"
                )
        source_liveness_rows = feature.get("foundation_source_field_liveness") or []
        if not source_liveness_rows:
            failures.append("pretrain feature source-field liveness rows are missing")
        min_source_active_rate = float(feature.get("min_required_source_active_rate") or 0.0)
        min_source_active_count = int(feature.get("min_required_source_active_count") or 1)
        for row in source_liveness_rows:
            if (
                not bool((row or {}).get("observed"))
                or int((row or {}).get("nonfinite_count") or 0) != 0
                or bool((row or {}).get("near_constant"))
                or int((row or {}).get("active_count") or 0) < min_source_active_count
                or float((row or {}).get("active_rate") or 0.0) < min_source_active_rate
            ):
                failures.append(
                    f"pretrain feature source-field liveness failed: "
                    f"{(row or {}).get('split')}:{(row or {}).get('source_field')}"
                )
        for split, row in (feature.get("foundation_source_fields_by_split") or {}).items():
            if int((row or {}).get("source_missing_count") or 0) != 0:
                failures.append(f"pretrain foundation source fields missing for split={split}")
        allowed_smoke_dataset_schemas = {
            "entry_foundation_seq146_smoke_dataset_v1",
            "entry_foundation_seq215_smoke_dataset_v1",
            "entry_smart_seq520_smoke_dataset_v1",
        }
        if str(smoke_dataset.get("schema_version")) not in allowed_smoke_dataset_schemas:
            failures.append(
                "pretrain smoke dataset contract schema is not an approved foundation smoke schema: "
                f"{smoke_dataset.get('schema_version')}"
            )
        if (
            str(smoke_dataset.get("audit_provenance_schema_version"))
            != "entry_foundation_smoke_dataset_audit_provenance_v1"
        ):
            failures.append("pretrain smoke dataset contract audit provenance schema is invalid")
        if not bool(smoke_dataset.get("audit_provenance_all_artifacts_present")):
            failures.append("pretrain smoke dataset contract audit artifacts are not all present")
        if not bool(smoke_dataset.get("audit_provenance_all_artifact_hashes_present")):
            failures.append("pretrain smoke dataset contract audit artifact hashes are missing")
        smoke_dataset_artifacts = (
            smoke_dataset.get("audit_provenance_artifacts")
            if isinstance(smoke_dataset.get("audit_provenance_artifacts"), dict)
            else {}
        )
        for key in ("feature_audit", "target_audit", "specialist_audit"):
            row = smoke_dataset_artifacts.get(key) if isinstance(smoke_dataset_artifacts.get(key), dict) else {}
            if not row:
                failures.append(f"pretrain smoke dataset contract missing audit artifact: {key}")
                continue
            manifest_path_for_key = path_by_hash_key.get(key)
            if (
                row.get("path")
                and manifest_path_for_key
                and str(Path(str(row.get("path"))).expanduser().resolve())
                != str(Path(str(manifest_path_for_key)).expanduser().resolve())
            ):
                failures.append(f"pretrain smoke dataset contract audit path mismatch: {key}")
            if str(row.get("sha256") or "") != str(hashes.get(key) or ""):
                failures.append(f"pretrain smoke dataset contract audit hash mismatch: {key}")
        split_hashes = smoke_dataset.get("split_hashes") if isinstance(smoke_dataset.get("split_hashes"), dict) else {}
        for split in ("train", "val", "test"):
            split_row = split_hashes.get(split) if isinstance(split_hashes.get(split), dict) else {}
            if not split_row:
                failures.append(f"pretrain smoke dataset contract missing split hash row: {split}")
                continue
            for field in ("source_manifest_sha256", "out_parquet_sha256", "out_manifest_sha256"):
                value = str(split_row.get(field) or "")
                if len(value) != 64:
                    failures.append(f"pretrain smoke dataset contract missing {field}: {split}")
    if str(target.get("decision")) != "PASS":
        failures.append(f"pretrain target contract decision is not PASS: {target.get('decision')}")
    if not target.get("active_training_heads"):
        failures.append("pretrain target contract has no active_training_heads")
    target_active_heads = set(str(head) for head in target.get("active_training_heads") or [] if str(head))
    target_blocked_heads = set(str(head) for head in target.get("blocked_heads") or [] if str(head))
    specialist_active_heads = set(str(head) for head in specialist.get("architecture_active_heads") or [] if str(head))
    specialist_blocked_heads = set(str(head) for head in specialist.get("architecture_blocked_heads") or [] if str(head))
    specialist_required_training = {
        str(name) for name in specialist.get("required_training_specialists", []) if str(name)
    }
    specialist_trainable = {str(name) for name in specialist.get("trainable_specialists", []) if str(name)}
    specialist_model_contract = (
        specialist.get("specialist_model_contract")
        if isinstance(specialist.get("specialist_model_contract"), dict)
        else {}
    )
    specialist_model_contract_keys = {str(name) for name in specialist_model_contract}
    expected_model_contract_keys = {str(name) for name in expected_model_contract}
    specialist_model_owned = {
        str(name): tuple(str(x) for x in (spec or {}).get("owned_objectives") or ())
        for name, spec in specialist_model_contract.items()
        if isinstance(spec, dict)
    }
    expected_model_owned = {
        str(name): tuple(str(x) for x in spec.get("owned_objectives") or ())
        for name, spec in expected_model_contract.items()
    }
    specialist_model_support_heads = {
        str(name): tuple(str(x) for x in (spec or {}).get("supports_heads") or ())
        for name, spec in specialist_model_contract.items()
        if isinstance(spec, dict)
    }
    specialist_model_signal_families = {
        str(name): tuple(str(x) for x in (spec or {}).get("primary_signal_families") or ())
        for name, spec in specialist_model_contract.items()
        if isinstance(spec, dict)
    }
    train_manifest_with_loader_contract = schema_version in {
        "entry_foundation_smoke_train_run_manifest_v1",
        "entry_foundation_candidate_train_run_manifest_v1",
    }
    if train_manifest_with_loader_contract:
        if specialist_required_training != expected_required_specialists:
            failures.append(
                "pretrain specialist required training set is not exact: "
                f"{sorted(specialist_required_training)}"
            )
        if specialist_trainable != expected_required_specialists:
            failures.append(
                "pretrain trainer specialist-fusion loader trainable set is not exact: "
                f"{sorted(specialist_trainable)}"
            )
        if not bool(specialist.get("specialist_model_contract_valid")):
            failures.append("pretrain specialist model contract is not valid")
        if specialist.get("specialist_model_contract_failures"):
            failures.append(
                "pretrain specialist model contract has failures: "
                f"{specialist.get('specialist_model_contract_failures')}"
            )
        if specialist_model_contract_keys != expected_model_contract_keys:
            failures.append(
                "pretrain specialist model contract trainable set is not exact: "
                f"{sorted(specialist_model_contract_keys)}"
            )
        if specialist_model_owned != expected_model_owned:
            failures.append(
                "pretrain specialist model contract owned objectives mismatch: "
                f"{specialist_model_owned}"
            )
        for specialist_name in sorted(expected_required_specialists):
            support_heads = set(specialist_model_support_heads.get(specialist_name) or ())
            if not support_heads:
                failures.append(f"pretrain specialist model contract has no support heads: {specialist_name}")
            if support_heads - set(SPECIALIST_FUSION_ACTIVE_HEADS):
                failures.append(
                    f"pretrain specialist model contract references inactive heads: "
                    f"{specialist_name} {sorted(support_heads - set(SPECIALIST_FUSION_ACTIVE_HEADS))}"
                )
            if not specialist_model_signal_families.get(specialist_name):
                failures.append(f"pretrain specialist model contract has no signal families: {specialist_name}")
    if specialist_active_heads != target_active_heads or specialist_active_heads != set(SPECIALIST_FUSION_ACTIVE_HEADS):
        failures.append(
            "pretrain specialist architecture active heads do not match target contract: "
            f"specialist={sorted(specialist_active_heads)} target={sorted(target_active_heads)}"
        )
    if specialist_blocked_heads != target_blocked_heads or specialist_blocked_heads != set(SPECIALIST_FUSION_BLOCKED_HEADS):
        failures.append(
            "pretrain specialist architecture blocked heads do not match target contract: "
            f"specialist={sorted(specialist_blocked_heads)} target={sorted(target_blocked_heads)}"
        )
    active_blocked_overlap = sorted(specialist_active_heads & specialist_blocked_heads)
    if active_blocked_overlap:
        failures.append(f"pretrain specialist architecture has active/blocked head overlap: {active_blocked_overlap}")
    if not bool(specialist.get("foundation_objective_routing_all_present_and_expected")):
        failures.append("pretrain specialist contract exact objective routing is not all-present")
    if not bool(specialist.get("specialist_input_liveness_all_live")):
        failures.append("pretrain specialist input liveness is not all-live")
    for row in specialist.get("foundation_objective_routing") or []:
        if int((row or {}).get("missing_count") or 0) != 0 or int((row or {}).get("misrouted_count") or 0) != 0:
            failures.append(f"pretrain specialist objective not exactly routed: {(row or {}).get('objective')}")
    for row in specialist.get("specialist_input_liveness") or []:
        if (
            int((row or {}).get("nonfinite_count") or 0) != 0
            or int((row or {}).get("live_feature_count") or 0)
            < int((row or {}).get("min_required_live_feature_count") or 1)
        ):
            failures.append(
                f"pretrain specialist input liveness failed: "
                f"{(row or {}).get('split')}:{(row or {}).get('specialist')}"
            )
    smoke_manifest_with_worktree_contract = schema_version == "entry_foundation_smoke_train_run_manifest_v1"
    worktree_critical_gate_review_ok: bool | None = None
    if smoke_manifest_with_worktree_contract:
        if not worktree:
            failures.append("pretrain manifest missing worktree hygiene contract")
            worktree_critical_gate_review_ok = False
        else:
            critical_gate_review = (
                worktree.get("foundation_cleanup_critical_gate_review")
                if isinstance(worktree.get("foundation_cleanup_critical_gate_review"), dict)
                else {}
            )
            critical_count = int(critical_gate_review.get("critical_gate_path_count") or 0)
            critical_ok_count = int(critical_gate_review.get("ok_count") or 0)
            critical_missing = critical_gate_review.get("missing_from_repo") or []
            critical_dirty_missing = critical_gate_review.get("dirty_missing_from_stage") or []
            worktree_critical_gate_review_ok = (
                critical_count > 0
                and critical_ok_count == critical_count
                and not critical_missing
                and not critical_dirty_missing
            )
            if not worktree_critical_gate_review_ok:
                failures.append(
                    "pretrain worktree hygiene critical gate review is not complete: "
                    f"ok={critical_ok_count}/{critical_count} "
                    f"missing={len(critical_missing)} dirty_missing_stage={len(critical_dirty_missing)}"
                )

    return {
        "decision": "PASS" if not failures else "FAIL",
        "manifest_path": str(manifest_path),
        "schema_version": manifest.get("schema_version"),
        "manifest_variant": str(
            manifest.get("manifest_variant")
            or manifest.get("candidate_variant")
            or contract_mode
        ),
        "candidate_variant": str(
            manifest.get("candidate_variant")
            or manifest.get("manifest_variant")
            or contract_mode
        ),
        "run_mode": manifest.get("run_mode"),
        "specialist_contract_mode": contract_mode,
        "expected_required_training_specialists": sorted(expected_required_specialists),
        "artifact_hash_checks": hash_checks,
        "feature_objective_coverage_all_present": (
            smoke_edge.get("feature_objective_coverage_all_present")
            if schema_version == "entry_foundation_candidate_train_run_manifest_v1"
            else feature.get("foundation_objective_coverage_all_present")
        ),
        "feature_objective_liveness_all_live": (
            smoke_edge.get("feature_objective_liveness_all_live")
            if schema_version == "entry_foundation_candidate_train_run_manifest_v1"
            else feature.get("foundation_objective_liveness_all_live")
        ),
        "feature_source_field_liveness_all_live": (
            smoke_edge.get("feature_source_field_liveness_all_live")
            if schema_version == "entry_foundation_candidate_train_run_manifest_v1"
            else feature.get("foundation_source_field_liveness_all_live")
        ),
        "smoke_edge_required_specialists_exact": (
            {
                str(name) for name in smoke_edge.get("required_training_specialists", []) if str(name)
            }
            == expected_required_specialists
            if schema_version == "entry_foundation_candidate_train_run_manifest_v1"
            else None
        ),
        "smoke_edge_specialist_groups_exact": (
            {
                str(name) for name in smoke_edge.get("specialist_groups", []) if str(name)
            }
            == expected_required_specialists
            if schema_version == "entry_foundation_candidate_train_run_manifest_v1"
            else None
        ),
        "smoke_edge_specialist_model_contract_valid": (
            bool(smoke_edge.get("specialist_model_contract_valid"))
            if schema_version == "entry_foundation_candidate_train_run_manifest_v1"
            else None
        ),
        "smoke_edge_specialist_model_contract_set_exact": (
            bool(smoke_edge.get("specialist_model_contract_set_exact"))
            if schema_version == "entry_foundation_candidate_train_run_manifest_v1"
            else None
        ),
        "smoke_edge_specialist_model_contract_owned_objectives_match": (
            bool(smoke_edge.get("specialist_model_contract_owned_objectives_match"))
            if schema_version == "entry_foundation_candidate_train_run_manifest_v1"
            else None
        ),
        "specialist_objective_routing_all_present_and_expected": specialist.get(
            "foundation_objective_routing_all_present_and_expected"
        ),
        "specialist_input_liveness_all_live": specialist.get("specialist_input_liveness_all_live"),
        "specialist_required_training_set_exact": (
            {
                str(name) for name in specialist.get("required_training_specialists", []) if str(name)
            }
            == expected_required_specialists
            if train_manifest_with_loader_contract
            else None
        ),
        "specialist_trainable_set_exact": (
            {
                str(name) for name in specialist.get("trainable_specialists", []) if str(name)
            }
            == expected_required_specialists
            if train_manifest_with_loader_contract
            else None
        ),
        "specialist_model_contract_valid": (
            bool(specialist.get("specialist_model_contract_valid"))
            and not specialist.get("specialist_model_contract_failures")
            if train_manifest_with_loader_contract
            else None
        ),
        "specialist_model_contract_set_exact": (
            {
                str(name) for name in (
                    specialist.get("specialist_model_contract")
                    if isinstance(specialist.get("specialist_model_contract"), dict)
                    else {}
                )
            }
            == expected_required_specialists
            if train_manifest_with_loader_contract
            else None
        ),
        "specialist_model_contract_owned_objectives_match": (
            {
                str(name): tuple(str(x) for x in (spec or {}).get("owned_objectives") or ())
                for name, spec in (
                    specialist.get("specialist_model_contract")
                    if isinstance(specialist.get("specialist_model_contract"), dict)
                    else {}
                ).items()
                if isinstance(spec, dict)
            }
            == {
                str(name): tuple(str(x) for x in spec.get("owned_objectives") or ())
                for name, spec in expected_model_contract.items()
            }
            if train_manifest_with_loader_contract
            else None
        ),
        "smoke_dataset_audit_provenance_all_artifacts_present": smoke_dataset.get(
            "audit_provenance_all_artifacts_present"
        )
        if schema_version == "entry_foundation_smoke_train_run_manifest_v1"
        else smoke_edge.get("smoke_dataset_audit_provenance_all_artifacts_present"),
        "smoke_dataset_audit_provenance_all_artifact_hashes_present": smoke_dataset.get(
            "audit_provenance_all_artifact_hashes_present"
        )
        if schema_version == "entry_foundation_smoke_train_run_manifest_v1"
        else smoke_edge.get("smoke_dataset_audit_provenance_all_artifact_hashes_present"),
        "smoke_edge_worktree_critical_gate_review_ok": (
            smoke_edge.get("worktree_critical_gate_review_ok")
            if schema_version == "entry_foundation_candidate_train_run_manifest_v1"
            else None
        ),
        "specialist_active_heads_match_target": (
            target_active_heads == specialist_active_heads == set(SPECIALIST_FUSION_ACTIVE_HEADS)
        ),
        "specialist_blocked_heads_match_target": (
            target_blocked_heads == specialist_blocked_heads == set(SPECIALIST_FUSION_BLOCKED_HEADS)
            and not active_blocked_overlap
        ),
        "target_active_training_heads": target.get("active_training_heads") or [],
        "target_blocked_heads": target.get("blocked_heads") or [],
        "worktree_hygiene_decision": worktree.get("decision"),
        "worktree_critical_gate_review_ok": worktree_critical_gate_review_ok,
        "failures": failures,
    }


def _audit_split(
    *,
    model: torch.nn.Module,
    device: torch.device,
    parquet_path: Path,
    seq_len: int,
    batch_size: int,
    dataset_kwargs: dict[str, Any],
    specialist_names: list[str],
    ctx_cat_names: list[str],
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    dataset = EntryV10CtxDataset(
        parquet_path=parquet_path,
        seq_len=seq_len,
        allow_constant_labels=True,
        **dataset_kwargs,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks: dict[str, list[np.ndarray]] = {
        "direction_logits": [],
        "path_quality_pred": [],
        "bad_path_prob": [],
        "labels": [],
        "ctx_cat": [],
        "path_quality_target": [],
        "bad_path_target": [],
        "specialist_gate": [],
    }
    output_keys_seen: set[str] = set()
    output_shapes_seen: dict[str, set[tuple[int, ...]]] = {}

    with torch.no_grad():
        for batch in loader:
            seq_x = batch["seq_x"].to(device)
            snap_x = batch["snap_x"].to(device)
            ctx_cat = batch["ctx_cat"].to(device)
            ctx_cont = batch["ctx_cont"].to(device)
            out = model(
                seq_x,
                snap_x,
                ctx_cat=ctx_cat,
                ctx_cont=ctx_cont,
                **_multi_tf_kwargs_from_batch(batch, device),
            )
            output_keys_seen.update(str(key) for key in out.keys())
            for key, value in out.items():
                if hasattr(value, "detach"):
                    output_shapes_seen.setdefault(str(key), set()).add(tuple(int(dim) for dim in value.shape[1:]))
                    if not bool(torch.isfinite(value).all().item()):
                        failures.append(f"{parquet_path.name}: non-finite model output {key}")
            direction_logits = out["direction_logits"]
            if tuple(direction_logits.shape[-1:]) != (3,):
                failures.append(f"{parquet_path.name}: direction_logits shape={tuple(direction_logits.shape)}")
            chunks["direction_logits"].append(direction_logits.detach().cpu().float().numpy())
            chunks["labels"].append(batch["y"].detach().cpu().numpy().astype(np.int64))
            chunks["ctx_cat"].append(batch["ctx_cat"].detach().cpu().numpy().astype(np.int64))
            chunks["path_quality_target"].append(batch["path_quality_bps"].detach().cpu().float().numpy())
            chunks["bad_path_target"].append(batch["y_bad_path"].detach().cpu().float().numpy())
            chunks["path_quality_pred"].append(out["path_quality"].detach().cpu().float().numpy().reshape(-1))
            chunks["bad_path_prob"].append(torch.sigmoid(out["bad_path_logit"]).detach().cpu().float().numpy().reshape(-1))
            if "specialist_gate" in out:
                chunks["specialist_gate"].append(out["specialist_gate"].detach().cpu().float().numpy())

    arr = {
        key: np.concatenate(value, axis=0) if value else np.zeros((0,), dtype=np.float32)
        for key, value in chunks.items()
    }
    direction = _direction_metrics(arr["direction_logits"], arr["labels"])
    direction_distribution = _direction_distribution_contract(direction)
    direction_slice = _direction_slice_contract(
        arr["direction_logits"],
        arr["labels"],
        arr["ctx_cat"],
        ctx_cat_names,
    )
    path_quality_spearman = _spearman(arr["path_quality_pred"], arr["path_quality_target"])
    bad_path_spearman = _spearman(arr["bad_path_prob"], arr["path_quality_target"])
    gate_report = None
    if len(chunks["specialist_gate"]) > 0:
        gate = np.concatenate(chunks["specialist_gate"], axis=0)
        gate_report = _gate_stats(gate, specialist_names)
        if not bool(gate_report["finite"]):
            failures.append(f"{parquet_path.name}: specialist gate has non-finite values")
        if float(gate_report["row_sum_max_abs_error"]) > 1e-4:
            failures.append(
                f"{parquet_path.name}: specialist gate row sums drift by {gate_report['row_sum_max_abs_error']}"
            )

    report = {
        "parquet_path": str(parquet_path),
        "rows": int(direction["rows"]),
        "direction": direction,
        "direction_distribution_contract": direction_distribution,
        "direction_slice_contract": direction_slice,
        "path_quality": {
            "target_mean_bps": _safe_mean(arr["path_quality_target"]),
            "pred_mean_bps": _safe_mean(arr["path_quality_pred"]),
            "pred_vs_target_spearman": path_quality_spearman,
        },
        "bad_path": {
            "target_rate": _safe_mean(arr["bad_path_target"]),
            "prob_mean": _safe_mean(arr["bad_path_prob"]),
            "prob_vs_path_quality_spearman": bad_path_spearman,
        },
        "specialist_gate": gate_report,
        "output_keys_seen": sorted(output_keys_seen),
        "output_shapes_seen": {
            key: [list(shape) for shape in sorted(shapes)]
            for key, shapes in sorted(output_shapes_seen.items())
        },
    }
    return report, failures


def _require_edge_failures(
    split: str, split_report: dict[str, Any], *, edge_test_scope: str = "strict"
) -> list[str]:
    """Split-level edge checks. Returns hard failures.

    edge_test_scope="strict" (default): every edge check is a hard failure on
    every split. Non-strict scopes are forbidden for XAU direction repair; a
    slice/path failure must block the artifact instead of becoming a logged-only
    diagnostic.
    """
    if edge_test_scope != "strict":
        raise ValueError(
            "[ENTRY_SMOKE_AUDIT_NO_EDGE_FALLBACK] "
            f"edge_test_scope={edge_test_scope!r} is forbidden; use strict"
        )
    failures: list[str] = []
    if not bool(split_report["direction"]["beats_majority_baseline"]):
        failures.append(f"{split}: direction accuracy does not beat majority baseline")
    distribution_contract = split_report.get("direction_distribution_contract") or {}
    if str(distribution_contract.get("decision")) != "PASS":
        failures.append(
            f"{split}: direction distribution collapsed: "
            f"{distribution_contract.get('failures')}"
        )
    slice_contract = split_report.get("direction_slice_contract") or {}
    if str(slice_contract.get("decision")) != "PASS":
        failures.append(
            f"{split}: direction slice diagnostics failed: "
            f"{slice_contract.get('failures')}"
        )
    path_rho = split_report["path_quality"]["pred_vs_target_spearman"]
    if path_rho is None or float(path_rho) <= 0.0:
        failures.append(f"{split}: path_quality_pred is not positively related to path_quality_bps")
    bad_path_rho = split_report["bad_path"]["prob_vs_path_quality_spearman"]
    if bad_path_rho is None or float(bad_path_rho) >= 0.0:
        failures.append(f"{split}: bad_path_prob is not negatively related to path_quality_bps")
    return failures


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Foundation Smoke Bundle Audit",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Bundle: `{report['bundle_dir']}`",
        f"- Dataset: `{report['dataset_dir']}`",
        f"- Require edge: `{report['require_edge']}`",
        f"- Edge test scope: `{report.get('edge_test_scope', 'strict')}`",
        f"- Require head contract: `{report['require_head_contract']}`",
        f"- Failure count: `{len(report['failures'])}`",
        "",
        "## Failures",
        "",
    ]
    if report["failures"]:
        lines.extend([f"- {failure}" for failure in report["failures"]])
    else:
        lines.append("- None")
    bundle_specialist_contract = report.get("bundle_specialist_model_contract")
    if isinstance(bundle_specialist_contract, dict):
        lines.extend(
            [
                "",
                "## Bundle Specialist Model Contract",
                "",
                f"- Decision: `{bundle_specialist_contract.get('decision')}`",
                f"- Set exact: `{bundle_specialist_contract.get('set_exact')}`",
                f"- Owned objectives match: `{bundle_specialist_contract.get('owned_objectives_match')}`",
                f"- Support heads match: `{bundle_specialist_contract.get('support_heads_match')}`",
                f"- Signal families match: `{bundle_specialist_contract.get('signal_families_match')}`",
                f"- Model roles match: `{bundle_specialist_contract.get('model_roles_match')}`",
            ]
        )
    pretrain = report.get("pretrain_manifest_contract")
    if isinstance(pretrain, dict):
        lines.extend(
            [
                "",
                "## Pretrain Manifest",
                "",
                f"- Decision: `{pretrain.get('decision')}`",
                f"- Manifest: `{pretrain.get('manifest_path')}`",
                f"- Feature objective coverage: `{pretrain.get('feature_objective_coverage_all_present')}`",
                f"- Feature objective liveness: `{pretrain.get('feature_objective_liveness_all_live')}`",
                f"- Feature source-field liveness: `{pretrain.get('feature_source_field_liveness_all_live')}`",
                f"- Specialist objective routing: `{pretrain.get('specialist_objective_routing_all_present_and_expected')}`",
                f"- Specialist input liveness: `{pretrain.get('specialist_input_liveness_all_live')}`",
                f"- Specialist active heads match target: `{pretrain.get('specialist_active_heads_match_target')}`",
                f"- Specialist blocked heads match target: `{pretrain.get('specialist_blocked_heads_match_target')}`",
                f"- Specialist model contract valid: `{pretrain.get('specialist_model_contract_valid')}`",
                f"- Specialist model contract set exact: `{pretrain.get('specialist_model_contract_set_exact')}`",
                f"- Specialist model contract owned objectives match: `{pretrain.get('specialist_model_contract_owned_objectives_match')}`",
            ]
        )
    lines.extend(["", "## Split Metrics", ""])
    for split, row in report["splits"].items():
        direction = row["direction"]
        bad_path = row["bad_path"]
        direction_slice = row.get("direction_slice_contract") or {}
        lines.append(
            f"- `{split}` rows={row['rows']} acc={direction['accuracy']:.4f} "
            f"majority={direction['majority_baseline_accuracy']:.4f} "
            f"beats_majority={direction['beats_majority_baseline']} "
            f"direction_slices={direction_slice.get('decision')} "
            f"bad_path_rho={bad_path['prob_vs_path_quality_spearman']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if str(getattr(args, "edge_test_scope", "strict") or "strict").strip() != "strict":
        raise ValueError(
            "[ENTRY_SMOKE_AUDIT_NO_EDGE_FALLBACK] "
            f"edge_test_scope={getattr(args, 'edge_test_scope', None)!r} is forbidden; use strict"
        )
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    m5_prebuilt = Path(args.m5_prebuilt_path).expanduser().resolve()
    mtf_cache_dir = Path(args.multi_tf_cache_dir).expanduser().resolve() if args.multi_tf_cache_dir else None
    splits = _parse_csv(args.splits)
    device = torch.device(_device_arg(args.device))
    out_dir.mkdir(parents=True, exist_ok=True)

    if mtf_cache_dir and mtf_cache_dir.exists():
        os.environ.setdefault("GX1_V10_MULTI_TF_V2_CACHE_DIR", str(mtf_cache_dir))

    failures: list[str] = []
    bundle = load_entry_v10_ctx_bundle(bundle_dir=bundle_dir, device=str(device), xgb_models=None)
    model = bundle.transformer_model
    model.eval()
    meta = dict(bundle.metadata)
    seq_len = int(meta.get("seq_len") or 96)
    specialist_cfg = meta.get("specialist_fusion") if isinstance(meta.get("specialist_fusion"), dict) else {}
    specialist_enabled = bool(specialist_cfg.get("enabled", False))
    specialist_contract_mode = str(specialist_cfg.get("contract_mode") or "foundation_seq146")
    required_specialists = tuple(required_training_specialists_for_mode(specialist_contract_mode))
    min_active_specialists = (
        len(required_specialists)
        if int(args.min_active_specialists) <= 0
        else int(args.min_active_specialists)
    )
    specialist_names = _specialist_names(meta)
    ctx_cat_names = [str(name) for name in (meta.get("ordered_ctx_cat_names") or []) if str(name)]
    if args.require_specialist_fusion and not specialist_enabled:
        failures.append("bundle metadata does not enable specialist_fusion")
    bundle_specialist_model_contract = _specialist_model_contract_report(
        specialist_cfg.get("specialist_model_contract")
        if isinstance(specialist_cfg.get("specialist_model_contract"), dict)
        else None,
        contract_mode=specialist_contract_mode,
    )
    if args.require_specialist_fusion:
        if not bool(specialist_cfg.get("specialist_model_contract_valid")):
            failures.append("bundle metadata specialist model contract is not declared valid")
        if specialist_cfg.get("specialist_model_contract_failures"):
            failures.append(
                "bundle metadata specialist model contract carries failures: "
                f"{specialist_cfg.get('specialist_model_contract_failures')}"
            )
        if not bool(bundle_specialist_model_contract["valid"]):
            failures.extend(
                f"bundle metadata specialist model contract: {failure}"
                for failure in bundle_specialist_model_contract["failures"]
            )
    path_calibration_recipe_contract = _path_calibration_recipe_contract(meta, bundle.capabilities)
    failures.extend(
        f"path_calibration_recipe: {failure}"
        for failure in path_calibration_recipe_contract.get("failures", [])
    )
    direction_balance_recipe_contract = _direction_balance_recipe_contract(
        meta,
        bundle.capabilities,
        contract_mode=specialist_contract_mode,
    )
    failures.extend(
        f"direction_balance_recipe: {failure}"
        for failure in direction_balance_recipe_contract.get("failures", [])
    )
    tail_direction_recipe_contract = _tail_direction_recipe_contract(meta, bundle.capabilities)
    failures.extend(
        f"tail_direction_recipe: {failure}"
        for failure in tail_direction_recipe_contract.get("failures", [])
    )
    symmetric_validation_recipe_contract = _symmetric_validation_recipe_contract(
        meta,
        bundle.capabilities,
        contract_mode=specialist_contract_mode,
    )
    failures.extend(
        f"symmetric_validation_recipe: {failure}"
        for failure in symmetric_validation_recipe_contract.get("failures", [])
    )

    dataset_kwargs = _bundle_dataset_kwargs(meta, m5_prebuilt)
    split_reports: dict[str, Any] = {}
    for split in splits:
        split_path = _split_file(dataset_dir, split)
        split_report, split_failures = _audit_split(
            model=model,
            device=device,
            parquet_path=split_path,
            seq_len=seq_len,
            batch_size=int(args.batch_size),
            dataset_kwargs=dataset_kwargs,
            specialist_names=specialist_names,
            ctx_cat_names=ctx_cat_names,
        )
        split_reports[split] = split_report
        failures.extend(split_failures)
        if args.require_specialist_fusion:
            failures.extend(
                _specialist_gate_failures(
                    split=split,
                    gate_report=split_report["specialist_gate"],
                    specialist_names=specialist_names,
                    required_specialists=required_specialists,
                    min_active_specialists=int(min_active_specialists),
                    min_gate_entropy=float(args.min_gate_entropy),
                )
        )
        if args.require_edge:
            edge_failures = _require_edge_failures(
                split, split_report, edge_test_scope=str(args.edge_test_scope)
            )
            failures.extend(edge_failures)

    head_contract = None
    if args.require_head_contract:
        target_audit = _read_json(Path(args.target_audit_json).expanduser().resolve())
        head_contract = _head_contract_report(
            target_audit=target_audit,
            capabilities=bundle.capabilities,
            split_reports=split_reports,
            contract_mode=specialist_contract_mode,
        )
        failures.extend([f"head_contract: {failure}" for failure in head_contract["failures"]])

    pretrain_manifest_path = (
        Path(args.pretrain_manifest_json).expanduser().resolve()
        if str(args.pretrain_manifest_json or "").strip()
        else None
    )
    pretrain_manifest_contract = _pretrain_manifest_contract_report(
        pretrain_manifest_path,
        expected_bundle_dir=bundle_dir,
        expected_dataset_dir=dataset_dir,
    )
    if pretrain_manifest_contract is not None:
        failures.extend(
            f"pretrain_manifest: {failure}"
            for failure in pretrain_manifest_contract.get("failures", [])
        )
    manifest_variant = str(
        (pretrain_manifest_contract or {}).get("manifest_variant")
        or (pretrain_manifest_contract or {}).get("candidate_variant")
        or specialist_contract_mode
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "entry_foundation_smoke_bundle_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS" if not failures else "FAIL",
        "manifest_variant": manifest_variant,
        "candidate_variant": manifest_variant,
        "bundle_dir": str(bundle_dir),
        "dataset_dir": str(dataset_dir),
        "data_splits": splits,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "require_edge": bool(args.require_edge),
        "edge_test_scope": str(args.edge_test_scope),
        "require_specialist_fusion": bool(args.require_specialist_fusion),
        "specialist_contract_mode": specialist_contract_mode,
        "required_training_specialists": list(required_specialists),
        "min_active_specialists": int(min_active_specialists),
        "min_gate_entropy": float(args.min_gate_entropy),
        "require_head_contract": bool(args.require_head_contract),
        "target_audit_json": str(Path(args.target_audit_json).expanduser().resolve()),
        "head_contract": head_contract,
        "pretrain_manifest_json": str(pretrain_manifest_path) if pretrain_manifest_path else "",
        "pretrain_manifest_contract": pretrain_manifest_contract,
        "path_calibration_recipe_contract": path_calibration_recipe_contract,
        "direction_balance_recipe_contract": direction_balance_recipe_contract,
        "tail_direction_recipe_contract": tail_direction_recipe_contract,
        "symmetric_validation_recipe_contract": symmetric_validation_recipe_contract,
        "bundle_specialist_model_contract": bundle_specialist_model_contract,
        "bundle_summary": {
            "manifest_variant": manifest_variant,
            "candidate_variant": manifest_variant,
            "specialist_contract_mode": specialist_contract_mode,
            "seq_len": seq_len,
            "seq_input_dim": int(meta.get("seq_input_dim") or 0),
            "snap_input_dim": int(meta.get("snap_input_dim") or 0),
            "ctx_cont_dim": int(meta.get("ctx_cont_dim") or 0),
            "ctx_cat_dim": int(meta.get("ctx_cat_dim") or 0),
            "sanity_bundle": bool(meta.get("sanity_bundle")),
            "multi_tf_enabled": bool((meta.get("multi_tf") or {}).get("enabled", False)),
            "specialist_fusion_enabled": specialist_enabled,
            "specialist_groups": specialist_names,
            "specialist_model_contract_declared_valid": bool(
                specialist_cfg.get("specialist_model_contract_valid")
            ),
            "specialist_model_contract_valid": bool(bundle_specialist_model_contract["valid"]),
            "specialist_model_contract_set_exact": bool(bundle_specialist_model_contract["set_exact"]),
            "specialist_model_contract_owned_objectives_match": bool(
                bundle_specialist_model_contract["owned_objectives_match"]
            ),
            "specialist_model_contract_support_heads_match": bool(
                bundle_specialist_model_contract["support_heads_match"]
            ),
            "specialist_model_contract_signal_families_match": bool(
                bundle_specialist_model_contract["signal_families_match"]
            ),
            "specialist_model_contract_model_roles_match": bool(
                bundle_specialist_model_contract["model_roles_match"]
            ),
            "capabilities": bundle.capabilities,
        },
        "splits": split_reports,
        "failures": failures,
    }
    json_path = out_dir / f"ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT_{timestamp}.json"
    md_path = out_dir / f"ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT_{timestamp}.md"
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    latest_json = out_dir / "ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT_latest.json"
    latest_md = out_dir / "ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT_latest.md"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "bundle_dir": report["bundle_dir"],
                    "dataset_dir": report["dataset_dir"],
                    "failures": report["failures"],
                    "json_path": report["json_path"],
                    "md_path": report["md_path"],
                },
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
        )
    if args.fail_on_audit_fail and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", default=str(FOUNDATION_SPECIALIST_SANITY_BUNDLE_DIR))
    ap.add_argument("--dataset-dir", default=str(FOUNDATION_SMOKE_DATASET_DIR))
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--m5-prebuilt-path", default=str(DEFAULT_M5_PREBUILT))
    ap.add_argument("--multi-tf-cache-dir", default=str(DEFAULT_MTF_CACHE_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--require-edge", action="store_true")
    ap.add_argument(
        "--edge-test-scope",
        choices=("strict",),
        default="strict",
        help=(
            "strict only: all edge checks hard-fail on every split. Non-strict scopes are "
            "forbidden for XAU direction repair."
        ),
    )
    ap.add_argument("--require-specialist-fusion", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--require-head-contract", action="store_true")
    ap.add_argument("--target-audit-json", default=str(TARGET_AUDIT_LATEST))
    ap.add_argument("--pretrain-manifest-json", default="")
    ap.add_argument("--min-active-specialists", type=int, default=0)
    ap.add_argument("--min-gate-entropy", type=float, default=0.05)
    ap.add_argument("--fail-on-audit-fail", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
