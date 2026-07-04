#!/usr/bin/env python3
"""Evaluate selective edge for an Entry foundation candidate bundle.

This is a post-candidate evidence writer. It strict-loads the runtime bundle,
runs val/test forward passes, ranks model trade candidates by directional edge
score, and writes the selective-edge artifacts consumed by replay-readiness.

It never trains, promotes, shadows, starts live, or writes adapter artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gx1.contracts.signal_bridge_v3 import ORDERED_SEQ_FIELDS_V3
from gx1.features.entry_specialist_feature_groups_v1 import (
    required_training_specialists_for_mode,
    specialist_model_contract_for_mode,
)
from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import EntryV10CtxDataset, _multi_tf_kwargs_from_batch
from gx1.scripts.audit_entry_foundation_smoke_bundle_v1 import (
    DEFAULT_M5_PREBUILT,
    DEFAULT_MTF_CACHE_DIR,
    _bundle_dataset_kwargs,
    _device_arg,
    _parse_csv,
    _split_file,
)
from gx1.scripts.verify_entry_foundation_state_v1 import FOUNDATION_DATASET_DIR, REPORTS_ROOT


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_candidate_selective_edge_20260628_v1"
SMART_SEQ520_DATASET_DIR = (
    Path("/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605")
    / "v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_smart_candidate_20260630"
)
SMART_SEQ520_OUT_DIR = DEFAULT_OUT_DIR / "smart_seq520_candidate"
SESSION_NAMES = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}
SIDE_NAMES = {0: "LONG", 1: "SHORT"}
CONTRACT_SIGNAL_DIMS = {
    "foundation_seq146": 146,
    "challenger_seq215": 215,
    "smart_seq520_candidate": 520,
}
FEATURE_MASK_SCHEMA_VERSION = "entry_selective_edge_feature_mask_v1"
SPECIALIST_MODEL_CONTRACT_FLAGS = (
    "specialist_model_contract_valid",
    "specialist_model_contract_set_exact",
    "specialist_model_contract_owned_objectives_match",
    "specialist_model_contract_signal_families_match",
    "specialist_model_contract_support_heads_match",
    "specialist_model_contract_model_roles_match",
)
SIGNAL_BRIDGE_NEUTRAL_VALUES = np.array(
    [
        1.0 / 3.0,  # p_long
        1.0 / 3.0,  # p_short
        1.0 / 3.0,  # p_flat
        1.0 / 3.0,  # p_hat
        2.0 / 3.0,  # uncertainty_score = 1 - p_hat
        0.0,  # margin_top1_top2
        math.log(3.0),  # entropy for a uniform 3-class distribution
    ],
    dtype=np.float32,
)


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


def _safe_mean(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _normalize_contract_mode(value: Any) -> str:
    mode = str(value or "foundation_seq146").strip()
    if mode not in CONTRACT_SIGNAL_DIMS:
        raise RuntimeError(f"unknown specialist contract mode: {mode}")
    return mode


def _specialist_cfg_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    cfg = meta.get("specialist_fusion") if isinstance(meta.get("specialist_fusion"), dict) else {}
    return dict(cfg)


def _observed_contract_mode(meta: dict[str, Any]) -> str:
    cfg = _specialist_cfg_from_meta(meta)
    for source in (cfg, meta):
        for key in ("contract_mode", "specialist_contract_mode"):
            value = str(source.get(key) or "").strip()
            if value:
                return value
    return ""


def _string_tuple(value: Any) -> tuple[str, ...]:
    return tuple(str(x) for x in (value or ()) if str(x))


def _specialist_contract_snapshot(meta: dict[str, Any], requested_mode: str) -> dict[str, Any]:
    contract_mode = _normalize_contract_mode(requested_mode)
    expected_dim = int(CONTRACT_SIGNAL_DIMS[contract_mode])
    expected_specialists = sorted(required_training_specialists_for_mode(contract_mode))
    expected_contract = specialist_model_contract_for_mode(contract_mode)
    cfg = _specialist_cfg_from_meta(meta)
    input_indices = cfg.get("input_indices") if isinstance(cfg.get("input_indices"), dict) else {}
    observed_specialists = sorted(
        str(name)
        for name, values in input_indices.items()
        if str(name) and list(values or [])
    )
    observed_mode = _observed_contract_mode(meta)
    observed_contract = cfg.get("specialist_model_contract") if isinstance(cfg.get("specialist_model_contract"), dict) else {}
    seq_dim = int(meta.get("seq_input_dim") or meta.get("snap_input_dim") or 0)
    snap_dim = int(meta.get("snap_input_dim") or meta.get("seq_input_dim") or 0)
    failures: list[str] = []

    if seq_dim != expected_dim:
        failures.append(f"bundle seq_input_dim mismatch: observed={seq_dim} expected={expected_dim}")
    if snap_dim != expected_dim:
        failures.append(f"bundle snap_input_dim mismatch: observed={snap_dim} expected={expected_dim}")
    if not bool(cfg.get("enabled")):
        failures.append("bundle specialist_fusion.enabled is not true")
    if observed_mode and observed_mode != contract_mode:
        failures.append(f"bundle specialist contract mode mismatch: observed={observed_mode} expected={contract_mode}")
    if contract_mode != "foundation_seq146" and not observed_mode:
        failures.append(f"{contract_mode} bundle must declare specialist_fusion.contract_mode")

    missing = sorted(set(expected_specialists) - set(observed_specialists))
    extra = sorted(set(observed_specialists) - set(expected_specialists))
    if missing:
        failures.append(f"bundle specialist_fusion missing required specialists: {missing}")
    if extra:
        failures.append(f"bundle specialist_fusion has non-required specialists: {extra}")
    if contract_mode != "foundation_seq146":
        for required in ("chart_geometry_encoder", "price_action_candle_encoder"):
            if required not in observed_specialists:
                failures.append(f"{contract_mode} missing required specialist: {required}")

    expected_contract_keys = sorted(str(name) for name in expected_contract)
    observed_contract_keys = sorted(str(name) for name in observed_contract)
    if observed_contract_keys != expected_contract_keys:
        failures.append(
            "bundle specialist_model_contract set mismatch: "
            f"observed={observed_contract_keys} expected={expected_contract_keys}"
        )
    for flag in SPECIALIST_MODEL_CONTRACT_FLAGS:
        if not bool(cfg.get(flag)):
            failures.append(f"bundle specialist_fusion.{flag} is not true")
    for name, expected_spec in expected_contract.items():
        observed_spec = observed_contract.get(name)
        if not isinstance(observed_spec, dict):
            failures.append(f"bundle specialist_model_contract missing spec for {name}")
            continue
        if str(observed_spec.get("model_role") or "") != str(expected_spec.get("model_role") or ""):
            failures.append(f"bundle specialist_model_contract model_role mismatch: {name}")
        for field in ("owned_objectives", "primary_signal_families", "supports_heads"):
            if _string_tuple(observed_spec.get(field)) != _string_tuple(expected_spec.get(field)):
                failures.append(f"bundle specialist_model_contract {field} mismatch: {name}")

    return {
        "requested_contract_mode": contract_mode,
        "observed_contract_mode": observed_mode,
        "contract_mode_declared": bool(observed_mode),
        "expected_signal_dim": expected_dim,
        "bundle_seq_input_dim": seq_dim,
        "bundle_snap_input_dim": snap_dim,
        "specialist_fusion_enabled": bool(cfg.get("enabled")),
        "expected_specialists": expected_specialists,
        "observed_specialists": observed_specialists,
        "required_specialists_exact": not missing and not extra,
        "chart_geometry_present": "chart_geometry_encoder" in observed_specialists,
        "price_action_candle_present": "price_action_candle_encoder" in observed_specialists,
        "specialist_model_contract_keys": observed_contract_keys,
        "specialist_model_contract_expected_keys": expected_contract_keys,
        "specialist_model_contract_valid": bool(cfg.get("specialist_model_contract_valid")),
        "specialist_model_contract_set_exact": bool(cfg.get("specialist_model_contract_set_exact")),
        "specialist_model_contract_owned_objectives_match": bool(
            cfg.get("specialist_model_contract_owned_objectives_match")
        ),
        "specialist_model_contract_signal_families_match": bool(
            cfg.get("specialist_model_contract_signal_families_match")
        ),
        "specialist_model_contract_support_heads_match": bool(
            cfg.get("specialist_model_contract_support_heads_match")
        ),
        "specialist_model_contract_model_roles_match": bool(
            cfg.get("specialist_model_contract_model_roles_match")
        ),
        "failures": failures,
    }


def _pnl_proxy_for_side(frame: pd.DataFrame) -> np.ndarray:
    """Return a trade-side PnL proxy in bps for LONG/SHORT model decisions."""
    side = frame["trade_side"].astype(int).to_numpy()
    if "y_forecast_ret_K24" in frame.columns:
        ret = pd.to_numeric(frame["y_forecast_ret_K24"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        return np.where(side == 0, ret, -ret)

    label = frame["y_direction"].astype(int).to_numpy()
    if "path_quality_bps" in frame.columns:
        quality = pd.to_numeric(frame["path_quality_bps"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    else:
        quality = np.zeros((len(frame),), dtype=np.float64)
    abs_quality = np.abs(quality)
    return np.where(label == side, abs_quality, np.where(label == 2, 0.0, -abs_quality))


def _direction_precision(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    return float((frame["trade_side"].astype(int) == frame["y_direction"].astype(int)).mean())


def _metrics_for_group(
    frame: pd.DataFrame,
    *,
    split: str,
    model: str,
    scope: str,
    top_frac: float,
    group: str,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "split": split,
            "model": model,
            "scope": scope,
            "top_frac": float(top_frac),
            "group": group,
            "n": 0,
            "mean_pnl_bps": None,
            "win_rate": None,
            "direction_precision": None,
            "mean_edge_score": None,
            "mean_bad_path_prob": None,
            "mean_path_quality_pred": None,
        }
    pnl = pd.to_numeric(frame["pnl_proxy_bps"], errors="coerce")
    return {
        "split": split,
        "model": model,
        "scope": scope,
        "top_frac": float(top_frac),
        "group": group,
        "n": int(len(frame)),
        "mean_pnl_bps": _safe_mean(pnl),
        "win_rate": _safe_mean((pnl > 0.0).astype(float)),
        "direction_precision": _direction_precision(frame),
        "mean_edge_score": _safe_mean(frame["edge_score"]),
        "mean_bad_path_prob": _safe_mean(frame["bad_path_prob"]) if "bad_path_prob" in frame.columns else None,
        "mean_path_quality_pred": _safe_mean(frame["path_quality_pred"]) if "path_quality_pred" in frame.columns else None,
    }


def _top_frame(frame: pd.DataFrame, top_frac: float) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    n = max(1, int(math.ceil(len(frame) * float(top_frac))))
    return frame.sort_values("edge_score", ascending=False, kind="mergesort").head(n).copy()


def _neutralize_signal_bridge(seq_x: torch.Tensor, snap_x: torch.Tensor) -> None:
    values = torch.as_tensor(SIGNAL_BRIDGE_NEUTRAL_VALUES, dtype=seq_x.dtype, device=seq_x.device)
    seq_x[..., : len(SIGNAL_BRIDGE_NEUTRAL_VALUES)] = values
    snap_x[..., : len(SIGNAL_BRIDGE_NEUTRAL_VALUES)] = values.to(dtype=snap_x.dtype, device=snap_x.device)


def _sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_signal_fields(dataset_dir: Path, splits: list[str]) -> tuple[list[str], str, str]:
    candidates: list[Path] = []
    for split in splits:
        candidates.extend(sorted(dataset_dir.glob(f"*_{split}.manifest.json")))
    candidates.extend(sorted(dataset_dir.glob("*.manifest.json")))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        extra = data.get("extra") if isinstance(data.get("extra"), dict) else {}
        signal_bridge = extra.get("signal_bridge") if isinstance(extra.get("signal_bridge"), dict) else {}
        fields = [str(value) for value in signal_bridge.get("fields", []) if str(value)]
        if fields:
            return fields, str(path), _sha256_file(path)
    return [], "", ""


def _load_feature_mask_spec(path_text: str, *, dataset_dir: Path, splits: list[str]) -> tuple[dict[str, Any], list[str]]:
    path_text = str(path_text or "").strip()
    if not path_text:
        return {"enabled": False}, []
    path = Path(path_text).expanduser().resolve()
    failures: list[str] = []
    if not path.exists() or not path.is_file():
        return {"enabled": True, "path": str(path)}, [f"feature mask json missing: {path}"]
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"enabled": True, "path": str(path)}, [f"feature mask json parse failed: {path}: {exc}"]
    if not isinstance(spec, dict):
        return {"enabled": True, "path": str(path)}, [f"feature mask json must be an object: {path}"]

    signal_fields, manifest_path, manifest_sha = _read_signal_fields(dataset_dir, splits)
    if not signal_fields:
        failures.append(f"could not resolve signal fields from dataset manifests in {dataset_dir}")
    field_to_idx = {name: idx for idx, name in enumerate(signal_fields)}
    names = [str(value) for value in spec.get("zero_feature_names", []) if str(value)]
    raw_indices = [int(value) for value in spec.get("zero_indices", []) if str(value).strip()]
    missing_names = [name for name in names if name not in field_to_idx]
    if missing_names:
        failures.append(f"feature mask names not found in signal fields: {missing_names[:20]}")
    name_indices = [field_to_idx[name] for name in names if name in field_to_idx]
    all_indices = sorted(set(name_indices + raw_indices))
    invalid_indices = [idx for idx in all_indices if idx < 0 or idx >= len(signal_fields)]
    if invalid_indices:
        failures.append(f"feature mask indices out of range: {invalid_indices[:20]}")
    all_indices = [idx for idx in all_indices if 0 <= idx < len(signal_fields)]
    if not all_indices:
        failures.append("feature mask selects zero valid signal indices")
    schema_version = str(spec.get("schema_version") or "")
    if schema_version and schema_version != FEATURE_MASK_SCHEMA_VERSION:
        failures.append(
            f"feature mask schema_version mismatch: observed={schema_version} expected={FEATURE_MASK_SCHEMA_VERSION}"
        )

    normalized = {
        "enabled": True,
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema_version": schema_version or FEATURE_MASK_SCHEMA_VERSION,
        "mask_mode": str(spec.get("mask_mode") or "zero_seq_snap_features"),
        "ablation_id": str(spec.get("ablation_id") or ""),
        "model_name": str(spec.get("model_name") or ""),
        "zero_value": float(spec.get("zero_value", 0.0)),
        "zero_indices": all_indices,
        "zero_feature_names": [signal_fields[idx] for idx in all_indices],
        "requested_zero_feature_names": names,
        "requested_zero_indices": raw_indices,
        "signal_field_count": int(len(signal_fields)),
        "signal_fields_manifest": manifest_path,
        "signal_fields_manifest_sha256": manifest_sha,
        "missing_feature_names": missing_names,
        "invalid_indices": invalid_indices,
    }
    return normalized, failures


def _apply_feature_mask(seq_x: torch.Tensor, snap_x: torch.Tensor, feature_mask: dict[str, Any] | None) -> None:
    if not feature_mask or not bool(feature_mask.get("enabled")):
        return
    indices = [int(value) for value in feature_mask.get("zero_indices", [])]
    if not indices:
        return
    index = torch.as_tensor(indices, dtype=torch.long, device=seq_x.device)
    value = float(feature_mask.get("zero_value", 0.0))
    seq_x.index_fill_(-1, index, value)
    snap_x.index_fill_(-1, index, value)


def build_metric_rows(
    predictions: pd.DataFrame,
    *,
    top_fracs: list[float],
    exclude_sessions: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Selection-policy note (2026-07-04, vedtak SMART_SEQ520_candidate_train_20260703):
    exclude_sessions removes those sessions from the SELECTION POOL before the
    top-frac ranking (the selection budget stays a fraction of the FULL split, so
    excluded-session capacity redirects to sessions with proven tail precision).
    SKIP_ASIA-precedent policy class; grounds: quiet-session memorized-confidence
    (EU morning: 97% SHORT calls vs 61% FLAT truth, tail precision 0.08-0.26).
    The policy is recorded in the summary and must be mirrored by the deploy
    operating point."""
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return rows
    for (split, model), sm in predictions.groupby(["split", "model"], sort=True):
        pool = sm[~sm["session"].astype(str).isin(exclude_sessions)] if exclude_sessions else sm
        for top_frac in top_fracs:
            n_budget = max(1, int(math.ceil(len(sm) * float(top_frac))))
            top = pool.sort_values("edge_score", ascending=False, kind="mergesort").head(n_budget).copy()
            rows.append(_metrics_for_group(top, split=str(split), model=str(model), scope="top_score", top_frac=top_frac, group="ALL"))
            for session, group in top.groupby("session", sort=True):
                rows.append(
                    _metrics_for_group(
                        group,
                        split=str(split),
                        model=str(model),
                        scope="top_score",
                        top_frac=top_frac,
                        group=f"session={session}",
                    )
                )
            for side, group in top.groupby("side", sort=True):
                rows.append(
                    _metrics_for_group(
                        group,
                        split=str(split),
                        model=str(model),
                        scope="top_score",
                        top_frac=top_frac,
                        group=f"side={side}",
                    )
                )
            for vol, group in top.groupby("vol_regime", sort=True):
                rows.append(
                    _metrics_for_group(
                        group,
                        split=str(split),
                        model=str(model),
                        scope="top_score",
                        top_frac=top_frac,
                        group=f"vol_regime={vol}",
                    )
                )
    return rows


def build_summary(predictions: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    splits = sorted(str(x) for x in predictions["split"].dropna().unique()) if not predictions.empty else []
    models = sorted(str(x) for x in predictions["model"].dropna().unique()) if not predictions.empty else []
    for split in splits:
        for model in models:
            scoped = metrics[
                (metrics["split"].astype(str) == split)
                & (metrics["model"].astype(str) == model)
                & (metrics["scope"].astype(str) == "top_score")
                & (metrics["group"].astype(str) == "ALL")
            ]
            def _value(top_frac: float, key: str) -> Any:
                row = scoped[np.isclose(scoped["top_frac"].astype(float), float(top_frac))]
                if row.empty:
                    return None
                value = row.iloc[0].get(key)
                if pd.isna(value):
                    return None
                return float(value)

            summaries.append(
                {
                    "split": split,
                    "model": model,
                    "rows": int(len(predictions[(predictions["split"] == split) & (predictions["model"] == model)])),
                    "top5_all_mean_pnl_bps": _value(0.05, "mean_pnl_bps"),
                    "top10_all_mean_pnl_bps": _value(0.10, "mean_pnl_bps"),
                    "top5_all_direction_precision": _value(0.05, "direction_precision"),
                    "top10_all_direction_precision": _value(0.10, "direction_precision"),
                }
            )
    return {
        "splits": splits,
        "models": models,
        "summaries": summaries,
    }


def build_no_xgb_ablation_diagnostics(predictions: pd.DataFrame, *, model_name: str) -> dict[str, Any]:
    no_xgb_model = f"{model_name}_no_xgb"
    if predictions.empty or model_name not in set(predictions["model"].astype(str)) or no_xgb_model not in set(predictions["model"].astype(str)):
        return {"available": False, "reason": "candidate or no-XGB predictions missing", "splits": {}}

    split_rows: dict[str, Any] = {}
    for split in sorted(str(x) for x in predictions["split"].dropna().unique()):
        base = predictions[(predictions["split"].astype(str) == split) & (predictions["model"].astype(str) == model_name)].reset_index(drop=True)
        ablation = predictions[
            (predictions["split"].astype(str) == split) & (predictions["model"].astype(str) == no_xgb_model)
        ].reset_index(drop=True)
        if base.empty or ablation.empty or len(base) != len(ablation):
            split_rows[split] = {
                "rows_candidate": int(len(base)),
                "rows_no_xgb": int(len(ablation)),
                "comparable": False,
            }
            continue
        prob_deltas = [(base[col].astype(float) - ablation[col].astype(float)).abs() for col in ("p_long", "p_short", "p_flat")]
        edge_delta = (base["edge_score"].astype(float) - ablation["edge_score"].astype(float)).abs()
        side_diff = base["trade_side"].astype(int) != ablation["trade_side"].astype(int)
        pred_diff = base["pred_direction"].astype(int) != ablation["pred_direction"].astype(int)
        time_match = True
        if "time" in base.columns and "time" in ablation.columns:
            time_match = bool(base["time"].equals(ablation["time"]))
        split_rows[split] = {
            "rows": int(len(base)),
            "comparable": True,
            "time_match": bool(time_match),
            "max_abs_prob_delta": float(max(delta.max() for delta in prob_deltas)),
            "mean_abs_prob_delta": float(np.mean([delta.mean() for delta in prob_deltas])),
            "max_abs_edge_score_delta": float(edge_delta.max()),
            "mean_abs_edge_score_delta": float(edge_delta.mean()),
            "trade_side_diff_count": int(side_diff.sum()),
            "pred_direction_diff_count": int(pred_diff.sum()),
            "identical_predictions": bool(
                max(delta.max() for delta in prob_deltas) == 0.0
                and float(edge_delta.max()) == 0.0
                and int(side_diff.sum()) == 0
                and int(pred_diff.sum()) == 0
            ),
        }

    return {
        "available": True,
        "candidate_model": model_name,
        "no_xgb_model": no_xgb_model,
        "splits": split_rows,
    }


def _dataset_signal_bridge_contract(dataset_dir: Path, splits: list[str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for split in splits:
        matches = sorted(dataset_dir.glob(f"*_{split}.manifest.json"))
        if len(matches) != 1:
            rows[split] = {
                "manifest_path": "",
                "manifest_count": int(len(matches)),
                "neutral_xgb_bridge": None,
                "bridge_source": "",
                "fields": [],
            }
            continue
        path = matches[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        extra = data.get("extra") if isinstance(data.get("extra"), dict) else {}
        inputs = data.get("inputs") if isinstance(data.get("inputs"), dict) else {}
        signal_bridge = extra.get("signal_bridge") if isinstance(extra.get("signal_bridge"), dict) else {}
        fields = [str(x) for x in signal_bridge.get("fields", [])]
        rows[split] = {
            "manifest_path": str(path),
            "manifest_count": 1,
            "neutral_xgb_bridge": bool(
                extra.get("neutral_xgb_bridge", False)
                or inputs.get("neutral_xgb_bridge", False)
                or signal_bridge.get("neutral_xgb_bridge", False)
            ),
            "bridge_source": str(
                extra.get("xgb_bridge_source") or inputs.get("xgb_bridge_source") or signal_bridge.get("bridge_source") or ""
            ),
            "seq_input_dim": int(signal_bridge.get("seq_input_dim") or 0),
            "snap_input_dim": int(signal_bridge.get("snap_input_dim") or 0),
            "fields": fields,
            "bridge_fields": fields[: len(SIGNAL_BRIDGE_NEUTRAL_VALUES)],
        }
    return {"splits": rows}


def _predict_bundle(
    *,
    bundle_dir: Path,
    dataset_dir: Path,
    splits: list[str],
    model_name: str,
    device: torch.device,
    batch_size: int,
    m5_prebuilt_path: Path,
    neutralize_signal_bridge: bool = False,
    feature_mask: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    failures: list[str] = []
    bundle = load_entry_v10_ctx_bundle(bundle_dir=bundle_dir, device=str(device), xgb_models=None)
    model = bundle.transformer_model
    model.eval()
    meta = dict(bundle.metadata)
    seq_len = int(meta.get("seq_len") or 96)
    dataset_kwargs = _bundle_dataset_kwargs(meta, m5_prebuilt_path)
    rows: list[pd.DataFrame] = []

    for split in splits:
        parquet_path = _split_file(dataset_dir, split)
        dataset = EntryV10CtxDataset(
            parquet_path=parquet_path,
            seq_len=seq_len,
            allow_constant_labels=True,
            **dataset_kwargs,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        chunks: dict[str, list[np.ndarray]] = {
            "p_long": [],
            "p_short": [],
            "p_flat": [],
            "edge_score": [],
            "trade_side": [],
            "pred_direction": [],
            "y_direction": [],
            "ctx_cat": [],
            "path_quality_pred": [],
            "bad_path_prob": [],
        }
        with torch.no_grad():
            for batch in loader:
                seq_x = batch["seq_x"].to(device)
                snap_x = batch["snap_x"].to(device)
                ctx_cat = batch["ctx_cat"].to(device)
                ctx_cont = batch["ctx_cont"].to(device)
                if neutralize_signal_bridge:
                    _neutralize_signal_bridge(seq_x, snap_x)
                _apply_feature_mask(seq_x, snap_x, feature_mask)
                out = model(
                    seq_x,
                    snap_x,
                    ctx_cat=ctx_cat,
                    ctx_cont=ctx_cont,
                    **_multi_tf_kwargs_from_batch(batch, device),
                )
                for key, value in out.items():
                    if hasattr(value, "detach") and not bool(torch.isfinite(value).all().item()):
                        failures.append(f"{model_name}/{split}: non-finite model output {key}")
                probs = torch.softmax(out["direction_logits"], dim=-1).detach().cpu().float().numpy()
                trade_side = np.where(probs[:, 0] >= probs[:, 1], 0, 1).astype(np.int64)
                edge_score = np.maximum(probs[:, 0], probs[:, 1]) - probs[:, 2]
                chunks["p_long"].append(probs[:, 0])
                chunks["p_short"].append(probs[:, 1])
                chunks["p_flat"].append(probs[:, 2])
                chunks["edge_score"].append(edge_score.astype(np.float32))
                chunks["trade_side"].append(trade_side)
                chunks["pred_direction"].append(probs.argmax(axis=1).astype(np.int64))
                chunks["y_direction"].append(batch["y"].detach().cpu().numpy().astype(np.int64))
                chunks["ctx_cat"].append(batch["ctx_cat"].detach().cpu().numpy().astype(np.int64))
                chunks["path_quality_pred"].append(out["path_quality"].detach().cpu().float().numpy().reshape(-1))
                chunks["bad_path_prob"].append(torch.sigmoid(out["bad_path_logit"]).detach().cpu().float().numpy().reshape(-1))

        arrays = {key: np.concatenate(value, axis=0) if value else np.zeros((0,), dtype=np.float32) for key, value in chunks.items()}
        frame = dataset.df.iloc[dataset.indices].reset_index(drop=True).copy()
        n = int(len(arrays["y_direction"]))
        frame = frame.iloc[:n].copy()
        ctx_cat = np.asarray(arrays["ctx_cat"], dtype=np.int64)
        frame["split"] = split
        frame["model"] = model_name
        frame["p_long"] = arrays["p_long"]
        frame["p_short"] = arrays["p_short"]
        frame["p_flat"] = arrays["p_flat"]
        frame["edge_score"] = arrays["edge_score"]
        frame["trade_side"] = arrays["trade_side"].astype(np.int64)
        frame["pred_direction"] = arrays["pred_direction"].astype(np.int64)
        frame["y_direction"] = arrays["y_direction"].astype(np.int64)
        frame["path_quality_pred"] = arrays["path_quality_pred"]
        frame["bad_path_prob"] = arrays["bad_path_prob"]
        frame["session"] = [SESSION_NAMES.get(int(x), f"UNKNOWN_{int(x)}") for x in ctx_cat[:, 0]]
        frame["vol_regime"] = [str(int(x)) for x in ctx_cat[:, 1]] if ctx_cat.shape[1] > 1 else "UNKNOWN"
        frame["side"] = [SIDE_NAMES.get(int(x), f"UNKNOWN_{int(x)}") for x in frame["trade_side"].to_numpy()]
        frame["pnl_proxy_bps"] = _pnl_proxy_for_side(frame)
        keep_cols = [
            "split",
            "model",
            "time",
            "y_direction",
            "pred_direction",
            "trade_side",
            "side",
            "session",
            "vol_regime",
            "edge_score",
            "p_long",
            "p_short",
            "p_flat",
            "path_quality_pred",
            "bad_path_prob",
            "path_quality_bps",
            "y_bad_path",
            "pnl_proxy_bps",
        ]
        if "y_forecast_ret_K24" in frame.columns:
            keep_cols.append("y_forecast_ret_K24")
        rows.append(frame[[c for c in keep_cols if c in frame.columns]])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), failures, meta


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Candidate Selective Edge",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Dataset: `{report['dataset_dir']}`",
        f"- Failure count: `{len(report['failures'])}`",
        f"- Promotion/shadow/live allowed: `{report['promotion_shadow_live_allowed']}`",
        "",
        "## Failures",
        "",
    ]
    if report["failures"]:
        lines.extend([f"- {failure}" for failure in report["failures"]])
    else:
        lines.append("- None")
    lines.extend(["", "## Summaries", ""])
    for row in report["summaries"]:
        lines.append(
            f"- `{row['model']}` `{row['split']}` rows={row['rows']} "
            f"top5={row['top5_all_mean_pnl_bps']} top10={row['top10_all_mean_pnl_bps']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    contract_mode = _normalize_contract_mode(getattr(args, "contract_mode", "foundation_seq146"))
    dataset_dir_raw = Path(args.dataset_dir).expanduser()
    out_dir_raw = Path(args.out_dir).expanduser()
    if contract_mode == "smart_seq520_candidate":
        if dataset_dir_raw.resolve() == FOUNDATION_DATASET_DIR.expanduser().resolve():
            dataset_dir_raw = SMART_SEQ520_DATASET_DIR
        if out_dir_raw.resolve() == DEFAULT_OUT_DIR.expanduser().resolve():
            out_dir_raw = SMART_SEQ520_OUT_DIR
    dataset_dir = dataset_dir_raw.resolve()
    out_dir = out_dir_raw.resolve()
    m5_prebuilt = Path(args.m5_prebuilt_path).expanduser().resolve()
    mtf_cache_dir = Path(args.multi_tf_cache_dir).expanduser().resolve() if args.multi_tf_cache_dir else None
    splits = _parse_csv(args.splits)
    top_fracs = [float(x) for x in _parse_csv(args.top_fracs)]
    device = torch.device(_device_arg(args.device))
    no_xgb_ablation_mode = str(getattr(args, "no_xgb_ablation_mode", "bundle")).strip()
    if no_xgb_ablation_mode not in {"bundle", "neutralize_signal_bridge"}:
        raise RuntimeError(f"invalid no-XGB ablation mode: {no_xgb_ablation_mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if mtf_cache_dir and mtf_cache_dir.exists():
        os.environ.setdefault("GX1_V10_MULTI_TF_V2_CACHE_DIR", str(mtf_cache_dir))

    failures: list[str] = []
    feature_mask, feature_mask_failures = _load_feature_mask_spec(
        str(getattr(args, "feature_mask_json", "") or ""),
        dataset_dir=dataset_dir,
        splits=splits,
    )
    failures.extend(feature_mask_failures)
    all_predictions: list[pd.DataFrame] = []
    bundle_meta: dict[str, Any] = {}
    no_xgb_bundle_meta: dict[str, Any] = {}
    candidate, candidate_failures, bundle_meta = _predict_bundle(
        bundle_dir=bundle_dir,
        dataset_dir=dataset_dir,
        splits=splits,
        model_name=str(args.model_name),
        device=device,
        batch_size=int(args.batch_size),
        m5_prebuilt_path=m5_prebuilt,
        feature_mask=feature_mask,
    )
    all_predictions.append(candidate)
    failures.extend(candidate_failures)
    bundle_specialist_contract = _specialist_contract_snapshot(bundle_meta, contract_mode)
    failures.extend([f"candidate bundle: {failure}" for failure in bundle_specialist_contract["failures"]])

    no_xgb_bundle = str(args.no_xgb_bundle_dir or "").strip()
    no_xgb_neutralize = False
    no_xgb_bundle_specialist_contract: dict[str, Any] | None = None
    if no_xgb_bundle:
        no_xgb_bundle_path = Path(no_xgb_bundle).expanduser().resolve()
        no_xgb_neutralize = no_xgb_ablation_mode == "neutralize_signal_bridge"
        if no_xgb_bundle_path == bundle_dir and not no_xgb_neutralize:
            failures.append(
                "no-XGB bundle dir equals candidate bundle but mode=bundle; use "
                "--no-xgb-ablation-mode neutralize_signal_bridge or provide a distinct ablation bundle"
            )
        ablation, ablation_failures, no_xgb_bundle_meta = _predict_bundle(
            bundle_dir=no_xgb_bundle_path,
            dataset_dir=dataset_dir,
            splits=splits,
            model_name=f"{args.model_name}_no_xgb",
            device=device,
            batch_size=int(args.batch_size),
            m5_prebuilt_path=m5_prebuilt,
            neutralize_signal_bridge=no_xgb_neutralize,
            feature_mask=feature_mask,
        )
        all_predictions.append(ablation)
        failures.extend(ablation_failures)
        no_xgb_bundle_specialist_contract = _specialist_contract_snapshot(no_xgb_bundle_meta, contract_mode)
        failures.extend(
            [f"no-XGB bundle: {failure}" for failure in no_xgb_bundle_specialist_contract["failures"]]
        )
    elif args.require_no_xgb_ablation:
        failures.append("--no-xgb-bundle-dir is required to produce candidate_no_xgb ablation evidence")

    predictions = pd.concat([df for df in all_predictions if not df.empty], ignore_index=True) if all_predictions else pd.DataFrame()
    if predictions.empty:
        failures.append("no selective-edge predictions were produced")
    exclude_sessions = tuple(
        s.strip() for s in str(getattr(args, "exclude_sessions", "") or "").split(",") if s.strip()
    )
    metric_rows = build_metric_rows(predictions, top_fracs=top_fracs, exclude_sessions=exclude_sessions)
    metrics = pd.DataFrame(metric_rows)
    summary_payload = build_summary(predictions, metrics)
    no_xgb_ablation_diagnostics = build_no_xgb_ablation_diagnostics(predictions, model_name=str(args.model_name))
    input_bridge_contract = _dataset_signal_bridge_contract(dataset_dir, splits)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ready = not failures

    predictions_path = out_dir / "selective_edge_predictions.parquet"
    metrics_path = out_dir / "selective_edge_metrics.csv"
    summary_path = out_dir / "summary.json"
    report_json_path = out_dir / f"ENTRY_CANDIDATE_SELECTIVE_EDGE_{timestamp}.json"
    report_md_path = out_dir / f"ENTRY_CANDIDATE_SELECTIVE_EDGE_{timestamp}.md"
    if not predictions.empty:
        predictions.to_parquet(predictions_path, index=False)
        if exclude_sessions:
            # Policy artifact for the downstream trade-log/replay chain: the
            # SAME selection policy (sessions excluded from the pool) applied to
            # the predictions, so thresholds and trades are computed under the
            # policy the deploy operating point will mirror.
            policy_pred = predictions[~predictions["session"].astype(str).isin(exclude_sessions)]
            policy_path = out_dir / "selective_edge_predictions_policy.parquet"
            policy_pred.to_parquet(policy_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    report = {
        "schema_version": "entry_candidate_selective_edge_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS" if ready else "FAIL",
        "contract_mode": contract_mode,
        "bundle_dir": str(bundle_dir),
        "no_xgb_bundle_dir": no_xgb_bundle or "",
        "no_xgb_ablation": {
            "required": bool(args.require_no_xgb_ablation),
            "mode": no_xgb_ablation_mode if no_xgb_bundle else "",
            "neutralize_signal_bridge": bool(no_xgb_neutralize),
            "neutralized_fields": ORDERED_SEQ_FIELDS_V3[: len(SIGNAL_BRIDGE_NEUTRAL_VALUES)] if no_xgb_neutralize else [],
            "neutral_values": SIGNAL_BRIDGE_NEUTRAL_VALUES.tolist() if no_xgb_neutralize else [],
        },
        "feature_mask_ablation": feature_mask,
        "dataset_dir": str(dataset_dir),
        "input_bridge_contract": input_bridge_contract,
        "splits": summary_payload["splits"],
        "models": summary_payload["models"],
        "summaries": summary_payload["summaries"],
        "no_xgb_ablation_diagnostics": no_xgb_ablation_diagnostics,
        "top_fracs": top_fracs,
        "selection_policy": {
            "exclude_sessions": list(exclude_sessions),
            "note": (
                "sessions removed from the selection pool before top-frac ranking; "
                "budget = frac of full split (capacity redirects to proven sessions); "
                "deploy operating point must mirror this policy"
            ) if exclude_sessions else "",
        },
        "bundle_seq_len": int(bundle_meta.get("seq_len") or 0),
        "bundle_seq_input_dim": int(bundle_meta.get("seq_input_dim") or bundle_meta.get("snap_input_dim") or 0),
        "bundle_snap_input_dim": int(bundle_meta.get("snap_input_dim") or bundle_meta.get("seq_input_dim") or 0),
        "bundle_specialist_fusion_enabled": bool((bundle_meta.get("specialist_fusion") or {}).get("enabled")) if isinstance(bundle_meta.get("specialist_fusion"), dict) else False,
        "bundle_specialist_contract": bundle_specialist_contract,
        "no_xgb_bundle_specialist_contract": no_xgb_bundle_specialist_contract,
        "trainer_started": False,
        "promotion_shadow_live_allowed": False,
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "json_path": str(report_json_path),
        "md_path": str(report_md_path),
        "failures": failures,
    }
    summary_payload.update(
        {
            "schema_version": report["schema_version"],
            "created_utc": report["created_utc"],
            "decision": report["decision"],
            "contract_mode": report["contract_mode"],
            "bundle_dir": report["bundle_dir"],
            "no_xgb_bundle_dir": report["no_xgb_bundle_dir"],
            "no_xgb_ablation": report["no_xgb_ablation"],
            "no_xgb_ablation_diagnostics": report["no_xgb_ablation_diagnostics"],
            "feature_mask_ablation": report["feature_mask_ablation"],
            "dataset_dir": report["dataset_dir"],
            "input_bridge_contract": report["input_bridge_contract"],
            "bundle_seq_len": report["bundle_seq_len"],
            "bundle_seq_input_dim": report["bundle_seq_input_dim"],
            "bundle_snap_input_dim": report["bundle_snap_input_dim"],
            "bundle_specialist_fusion_enabled": report["bundle_specialist_fusion_enabled"],
            "bundle_specialist_contract": report["bundle_specialist_contract"],
            "no_xgb_bundle_specialist_contract": report["no_xgb_bundle_specialist_contract"],
            "failures": failures,
        }
    )
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(report_md_path, report)
    latest_json = out_dir / "ENTRY_CANDIDATE_SELECTIVE_EDGE_latest.json"
    latest_md = out_dir / "ENTRY_CANDIDATE_SELECTIVE_EDGE_latest.md"
    latest_json.write_text(report_json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(report_md_path.read_text(encoding="utf-8"), encoding="utf-8")

    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "failures": failures,
                    "summary_path": str(summary_path),
                    "metrics_path": str(metrics_path),
                    "json_path": str(report_json_path),
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
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--no-xgb-bundle-dir", default="")
    ap.add_argument("--dataset-dir", default=str(FOUNDATION_DATASET_DIR))
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--top-fracs", default="0.05,0.10")
    ap.add_argument(
        "--exclude-sessions",
        default="",
        help=(
            "Comma-separated sessions removed from the SELECTION pool before top-frac "
            "ranking (selection-policy, SKIP_ASIA precedent class; recorded in the "
            "summary as selection_policy and must be mirrored by the deploy operating "
            "point). Empty = no policy."
        ),
    )
    ap.add_argument("--model-name", default="candidate")
    ap.add_argument(
        "--contract-mode",
        choices=("foundation_seq146", "challenger_seq215", "smart_seq520_candidate"),
        default="foundation_seq146",
    )
    ap.add_argument("--challenger-seq215", action="store_const", dest="contract_mode", const="challenger_seq215")
    ap.add_argument("--smart-seq520", action="store_const", dest="contract_mode", const="smart_seq520_candidate")
    ap.add_argument("--no-xgb-ablation-mode", choices=("bundle", "neutralize_signal_bridge"), default="bundle")
    ap.add_argument("--feature-mask-json", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--m5-prebuilt-path", default=str(DEFAULT_M5_PREBUILT))
    ap.add_argument("--multi-tf-cache-dir", default=str(DEFAULT_MTF_CACHE_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--require-no-xgb-ablation", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fail-on-audit-fail", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
