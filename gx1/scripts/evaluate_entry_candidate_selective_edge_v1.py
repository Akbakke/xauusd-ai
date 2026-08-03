#!/usr/bin/env python3
"""Evaluate selective edge for an exact model-native seq513 candidate bundle.

This is a post-candidate evidence writer. It strict-loads the runtime bundle,
runs val/test forward passes, ranks the model's own LONG/SHORT/FLAT choices by
their selected-class probability, and writes evidence for replay-readiness.

It never trains, promotes, shadows, starts live, or writes adapter artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gx1.contracts.entry_model_native_signal_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_FIELDS,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DOMAINS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CAT_INDEX_BY_NAME,
    MODEL_NATIVE_SIGNAL_DIM,
    require_model_native_signal_contract,
)
from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_ENTRY_TREND_REGIME_NAMES,
    MODEL_NATIVE_ENTRY_VOL_REGIME_NAMES,
    MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
    MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION,
    MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION,
    MODEL_NATIVE_RUNTIME_POLICY,
    encode_model_native_runtime_head_evidence,
    project_model_native_path_calibration,
)
from gx1.contracts.entry_model_native_direction_evidence_fusion_v1 import (
    INPUTS as DIRECTION_EVIDENCE_INPUTS,
)
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_TIMING_OUTPUT_DIM,
    MODEL_NATIVE_TIMING_TARGET_COLUMNS,
)
from gx1.contracts.entry_model_native_offline_rl_v1 import (
    ACTION_VALUE_TARGET_COLUMNS,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    required_training_specialists_for_mode,
    specialist_model_contract_for_mode,
)
from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_FLAT_INDEX,
    MODEL_DIRECTION_LONG_INDEX,
    MODEL_DIRECTION_NAME_BY_INDEX,
    MODEL_DIRECTION_SELECTION_MODE,
    MODEL_DIRECTION_SHORT_INDEX,
    MODEL_DIRECTION_TRADE_INDICES,
    PUBLIC_FLAT_INDEX,
    PUBLIC_TRADE_INDEX,
    require_model_direction_decision_contract,
)
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import EntryV10CtxDataset, _multi_tf_kwargs_from_batch
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    atomic_write_parquet_immutable,
    atomic_write_text,
    build_prediction_evidence_declaration,
)
from gx1.scripts.audit_entry_foundation_smoke_bundle_v1 import (
    _bundle_dataset_kwargs,
    _device_arg,
)
from gx1.time.session_detector import SESSION_NAME_BY_ID

SESSION_NAMES = SESSION_NAME_BY_ID
SIDE_NAMES = MODEL_DIRECTION_NAME_BY_INDEX
CONTRACT_SIGNAL_DIMS = {MODEL_NATIVE_CONTRACT_MODE: MODEL_NATIVE_SIGNAL_DIM}
EVALUATION_SPLITS = ("train", "val", "test")
DEFAULT_EVALUATION_SPLITS = ("val", "test")
EVIDENCE_STAGES = ("pre_calibration", "runtime_authoritative")
EVALUATION_TOP_FRACS = (0.05, 0.10)
EVALUATION_MODEL_NAME = "candidate"
SPECIALIST_MODEL_CONTRACT_FLAGS = (
    "specialist_model_contract_valid",
    "specialist_model_contract_set_exact",
    "specialist_model_contract_owned_objectives_match",
    "specialist_model_contract_signal_families_match",
    "specialist_model_contract_support_heads_match",
    "specialist_model_contract_model_roles_match",
)
def _reject_retired_selection_environment() -> None:
    retired = sorted(
        name
        for name in (
            "GX1_SMART_SELECTION_SCORE",
            "GX1_SMART_SELECTION_SCORE_THRESHOLD",
            "GX1_ENTRY_EXPECTED_UTILITY_THRESHOLD_BPS",
        )
        if name in os.environ
    )
    if retired:
        raise RuntimeError(
            "retired Entry selection environment is forbidden; "
            f"present={retired}"
        )


def _require_model_direction_ssot(
    out: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the sole public decision tensors or fail closed.

    ``public_trade_flat_decision_logits`` must be derived by the model from the
    final, calibrated three-class logits.  Rechecking the equality here keeps
    candidate evidence on exactly the same surface as training gates and live.
    """

    forbidden = sorted({"anchor_logits", "delta_logits", "anchor_gate"}.intersection(out))
    if forbidden:
        raise RuntimeError(
            f"model-native direction output contains forbidden legacy keys: {forbidden}"
        )
    missing = [
        key
        for key in ("direction_logits", "public_trade_flat_decision_logits")
        if key not in out
    ]
    if missing:
        raise RuntimeError(
            "model-native direction contract missing outputs: " + ",".join(missing)
        )
    direction_logits = out["direction_logits"]
    public_pair_logits = out["public_trade_flat_decision_logits"]
    if direction_logits.ndim != 2 or direction_logits.shape[1] != 3:
        raise RuntimeError(
            "direction_logits must have shape (B,3); "
            f"got {tuple(direction_logits.shape)}"
        )
    if public_pair_logits.ndim != 2 or public_pair_logits.shape != (
        direction_logits.shape[0],
        2,
    ):
        raise RuntimeError(
            "public_trade_flat_decision_logits must have shape (B,2); "
            f"got {tuple(public_pair_logits.shape)}"
        )
    if not bool(torch.isfinite(direction_logits).all().item()) or not bool(
        torch.isfinite(public_pair_logits).all().item()
    ):
        raise RuntimeError("model-native direction contract contains non-finite logits")
    expected_pair = torch.stack(
        (
            direction_logits[
                :, list(MODEL_DIRECTION_TRADE_INDICES)
            ].amax(dim=1),
            direction_logits[:, MODEL_DIRECTION_FLAT_INDEX],
        ),
        dim=1,
    )
    if not torch.equal(public_pair_logits, expected_pair):
        max_delta = float((public_pair_logits - expected_pair).abs().max().item())
        raise RuntimeError(
            "public trade/FLAT logits do not match final direction logits; "
            f"max_abs_delta={max_delta:.9g}"
        )
    winner_counts = direction_logits.eq(
        direction_logits.amax(dim=1, keepdim=True)
    ).sum(dim=1)
    tied_rows = int((winner_counts != 1).sum().item())
    if tied_rows:
        raise RuntimeError(
            "final direction logits have no unique top class; "
            f"rows={tied_rows}"
        )
    direction_decision = torch.argmax(direction_logits, dim=1)
    public_pair_decision = torch.argmax(public_pair_logits, dim=1)
    expected_pair_decision = torch.where(
        direction_decision == MODEL_DIRECTION_FLAT_INDEX,
        PUBLIC_FLAT_INDEX,
        PUBLIC_TRADE_INDEX,
    ).to(dtype=torch.long)
    if not torch.equal(public_pair_decision, expected_pair_decision):
        raise RuntimeError(
            "public trade/FLAT argmax does not match final LONG/SHORT/FLAT argmax"
        )
    return direction_logits, public_pair_logits


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


def _sigmoid_np(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.clip(arr, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-arr))).astype(np.float32)


def _softmax_np(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr - np.nanmax(arr, axis=1, keepdims=True)
    exp = np.exp(arr)
    denom = np.maximum(np.nansum(exp, axis=1, keepdims=True), 1e-12)
    return (exp / denom).astype(np.float32)


def _canonical_live_decision_evidence(
    out: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    """Materialize the exact probability/argmax surface consumed by live."""
    direction_logits_t, public_pair_logits_t = _require_model_direction_ssot(out)
    direction_logits = direction_logits_t.detach().cpu().float().numpy()
    public_pair_logits = public_pair_logits_t.detach().cpu().float().numpy()
    direction_probs = _softmax_np(direction_logits)
    public_pair_probs = _softmax_np(public_pair_logits)
    direction_index = np.argmax(direction_probs, axis=1).astype(np.int64)
    public_pair_index = np.argmax(public_pair_probs, axis=1).astype(np.int64)
    expected_public_index = np.where(
        direction_index == MODEL_DIRECTION_FLAT_INDEX,
        PUBLIC_FLAT_INDEX,
        PUBLIC_TRADE_INDEX,
    ).astype(np.int64)
    if not np.array_equal(public_pair_index, expected_public_index):
        raise RuntimeError(
            "canonical public TRADE/FLAT probability argmax does not match "
            "final LONG/SHORT/FLAT probability argmax"
        )
    row_index = np.arange(direction_index.shape[0], dtype=np.int64)
    selection_score = direction_probs[row_index, direction_index].astype(np.float32)
    edge_score = (
        np.maximum(
            direction_probs[:, MODEL_DIRECTION_LONG_INDEX],
            direction_probs[:, MODEL_DIRECTION_SHORT_INDEX],
        )
        - direction_probs[:, MODEL_DIRECTION_FLAT_INDEX]
    ).astype(np.float32)
    return {
        "direction_logits": direction_logits,
        "direction_probs": direction_probs,
        "model_direction_index": direction_index,
        "public_trade_flat_decision_logits": public_pair_logits,
        "public_trade_flat_decision_probs": public_pair_probs,
        "public_trade_flat_decision_index": public_pair_index,
        "p_long": direction_probs[:, MODEL_DIRECTION_LONG_INDEX],
        "p_short": direction_probs[:, MODEL_DIRECTION_SHORT_INDEX],
        "p_flat": direction_probs[:, MODEL_DIRECTION_FLAT_INDEX],
        "p_trade": public_pair_probs[:, PUBLIC_TRADE_INDEX],
        "p_flat_hier": public_pair_probs[:, PUBLIC_FLAT_INDEX],
        "edge_score": edge_score,
        "selection_score": selection_score,
    }


def _tensor_np(
    out: dict[str, torch.Tensor],
    key: str,
    *,
    width: int,
) -> np.ndarray:
    value = out.get(key)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"exact model-native bundle did not emit tensor {key}")
    arr = value.detach().cpu().float().numpy()
    if arr.ndim != 2 or arr.shape[1] != int(width) or not np.isfinite(arr).all():
        raise RuntimeError(
            f"exact model-native output {key} invalid: observed={arr.shape} "
            f"expected=(*,{int(width)}) finite={np.isfinite(arr).all()}"
        )
    return arr


def _normalize_contract_mode(value: Any) -> str:
    mode = str(value or MODEL_NATIVE_CONTRACT_MODE).strip()
    if mode != MODEL_NATIVE_CONTRACT_MODE:
        raise RuntimeError(
            "retired candidate contract mode is forbidden: "
            f"observed={mode!r} required={MODEL_NATIVE_CONTRACT_MODE!r}"
        )
    return MODEL_NATIVE_CONTRACT_MODE


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
    if observed_mode != contract_mode:
        failures.append(f"bundle specialist contract mode mismatch: observed={observed_mode} expected={contract_mode}")

    missing = sorted(set(expected_specialists) - set(observed_specialists))
    extra = sorted(set(observed_specialists) - set(expected_specialists))
    if missing:
        failures.append(f"bundle specialist_fusion missing required specialists: {missing}")
    if extra:
        failures.append(f"bundle specialist_fusion has non-required specialists: {extra}")
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
    """Score the final model argmax against the canonical two-sided utility target."""

    required = {"pred_direction", "y_long_path_utility_bps", "y_short_path_utility_bps"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"selective-edge frame lacks canonical utility evidence: {missing}"
        )
    side = pd.to_numeric(frame["pred_direction"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    long_score = pd.to_numeric(
        frame["y_long_path_utility_bps"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    short_score = pd.to_numeric(
        frame["y_short_path_utility_bps"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(side).all() or not np.isfinite(long_score).all() or not np.isfinite(
        short_score
    ).all():
        raise RuntimeError("canonical utility evidence contains non-finite values")
    if not set(side.astype(np.int64)).issubset(
        set(MODEL_DIRECTION_NAME_BY_INDEX)
    ) or not np.array_equal(side, side.astype(np.int64)):
        raise RuntimeError("pred_direction contains values outside LONG/SHORT/FLAT")
    side_int = side.astype(np.int64)
    return np.where(
        side_int == MODEL_DIRECTION_FLAT_INDEX,
        0.0,
        np.where(
            side_int == MODEL_DIRECTION_LONG_INDEX,
            long_score,
            short_score,
        ),
    )


def _direction_precision(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    if "pred_direction" not in frame.columns:
        raise RuntimeError("direction precision lacks final pred_direction")
    return float((frame["pred_direction"].astype(int) == frame["y_direction"].astype(int)).mean())


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


def _selection_sort_column(frame: pd.DataFrame) -> str:
    if "selection_score" not in frame.columns:
        raise RuntimeError("selective-edge frame lacks model-native selection_score")
    modes = (
        sorted({str(value) for value in frame["selection_score_mode"].dropna()})
        if "selection_score_mode" in frame.columns
        else []
    )
    if modes != [MODEL_DIRECTION_SELECTION_MODE]:
        raise RuntimeError(
            f"selective-edge direction mode mismatch: observed={modes}"
        )
    return "selection_score"


def _top_frame(frame: pd.DataFrame, top_frac: float) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    n = max(1, int(math.ceil(len(frame) * float(top_frac))))
    return frame.sort_values(_selection_sort_column(frame), ascending=False, kind="mergesort").head(n).copy()


def _sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_metric_rows(
    predictions: pd.DataFrame,
    *,
    top_fracs: list[float],
    exclude_sessions: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Build diagnostic tail metrics without changing model direction.

    Session exclusion is retained in the helper signature only to make stale
    direct callers fail explicitly.  Session evidence belongs inside seq513.
    """
    if exclude_sessions:
        raise RuntimeError(
            "external session exclusion is forbidden; session evidence must be "
            "fused into the model-native LONG/SHORT/FLAT decision"
        )
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return rows
    for (split, model), sm in predictions.groupby(["split", "model"], sort=True):
        pool = sm[~sm["session"].astype(str).isin(exclude_sessions)] if exclude_sessions else sm
        for top_frac in top_fracs:
            n_budget = max(1, int(math.ceil(len(sm) * float(top_frac))))
            top = pool.sort_values(_selection_sort_column(pool), ascending=False, kind="mergesort").head(n_budget).copy()
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


def _explicit_dataset_artifact(
    path_value: str,
    sha256_value: str,
    *,
    dataset_dir: Path,
    label: str,
    suffix: str,
) -> tuple[Path, str]:
    path = Path(path_value).expanduser()
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve() != path
        or path.parent != dataset_dir
        or not path.name.endswith(suffix)
        or any("latest" in part.lower() for part in path.parts)
    ):
        raise RuntimeError(
            f"selective-edge explicit dataset artifact is invalid: {label}={path}"
        )
    expected_sha = str(sha256_value or "").strip().lower()
    if len(expected_sha) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha
    ):
        raise RuntimeError(
            f"selective-edge explicit dataset artifact lacks SHA-256: {label}"
        )
    observed_sha = _sha256_file(path)
    if observed_sha != expected_sha:
        raise RuntimeError(
            f"selective-edge dataset artifact hash mismatch: {label} "
            f"expected={expected_sha} observed={observed_sha}"
        )
    return path, expected_sha


def _dataset_model_native_contract(
    dataset_dir: Path,
    splits: list[str],
    split_bindings: dict[str, dict[str, str]],
) -> dict[str, Any]:
    if set(split_bindings) != set(splits):
        raise RuntimeError(
            "selective-edge explicit dataset split bindings are incomplete"
        )
    rows: dict[str, Any] = {}
    reference_contract: dict[str, Any] | None = None
    for split in splits:
        binding = split_bindings[split]
        if not isinstance(binding, dict) or set(binding) != {
            "manifest_path",
            "manifest_sha256",
            "parquet_path",
            "parquet_sha256",
        }:
            raise RuntimeError(
                f"selective-edge split binding is invalid: split={split}"
            )
        manifest_path, manifest_sha = _explicit_dataset_artifact(
            binding["manifest_path"],
            binding["manifest_sha256"],
            dataset_dir=dataset_dir,
            label=f"{split}_manifest",
            suffix=f"_{split}.manifest.json",
        )
        parquet_path, parquet_sha = _explicit_dataset_artifact(
            binding["parquet_path"],
            binding["parquet_sha256"],
            dataset_dir=dataset_dir,
            label=f"{split}_parquet",
            suffix=f"_{split}.parquet",
        )
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared_parquet = Path(str(data.get("output_data_path") or "")).expanduser()
        if not declared_parquet.is_absolute() or declared_parquet != parquet_path:
            raise RuntimeError(
                f"dataset split {split!r} manifest output_data_path mismatch"
            )
        extra = data.get("extra") if isinstance(data.get("extra"), dict) else {}
        contract = extra.get("model_native_signal_contract")
        if not isinstance(contract, dict):
            raise RuntimeError(f"dataset split {split!r} lacks model_native_signal_contract")
        require_model_native_signal_contract(
            contract,
            context=f"SELECTIVE_EDGE_DATASET_{split.upper()}",
        )
        if reference_contract is None:
            reference_contract = dict(contract)
        elif dict(contract) != reference_contract:
            raise RuntimeError("dataset split model-native signal contracts differ")
        rows[split] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "parquet_path": str(parquet_path),
            "parquet_sha256": parquet_sha,
            "seq_input_dim": int(contract["seq_input_dim"]),
            "snap_input_dim": int(contract["snap_input_dim"]),
            "ordered_fields_sha256": str(contract["ordered_fields_sha256"]),
        }
        if _sha256_file(manifest_path) != manifest_sha:
            raise RuntimeError(
                f"dataset split {split!r} manifest changed during validation"
            )
    if reference_contract is None:
        raise RuntimeError("dataset has no model-native signal contract")
    return {"contract": reference_contract, "splits": rows}


def _iter_split_chunks(
    parquet_path: Path,
    manifest_path: Path,
    stream_chunk_rows: int,
):
    """Yield (chunk_parquet_path, cleanup_fn). Each dataset row carries its own
    nested (seq_len, signal_dim) sequence, so rows are independent and row-range
    chunking is loss-free. stream_chunk_rows<=0 -> the original file, unsliced.
    Memory-bounded evaluation for huge splits (2026-07-04: the 390K-row dense
    forward would need ~78GB if materialized in one piece)."""
    import pyarrow.parquet as _pq
    import tempfile

    if stream_chunk_rows <= 0:
        yield parquet_path, (lambda: None)
        return
    pf = _pq.ParquetFile(parquet_path)
    n_groups = pf.metadata.num_row_groups
    group_rows = [pf.metadata.row_group(i).num_rows for i in range(n_groups)]
    start = 0
    while start < n_groups:
        rows_acc = 0
        end = start
        while end < n_groups and rows_acc + group_rows[end] <= max(stream_chunk_rows, group_rows[end]):
            rows_acc += group_rows[end]
            end += 1
        print(f"[STREAM_CHUNK] reading row-groups {start}..{end - 1} of {n_groups} ({rows_acc:,} rows)", flush=True)
        table = pf.read_row_groups(list(range(start, end)))
        tmp = tempfile.NamedTemporaryFile(
            suffix=f"_{parquet_path.stem}_rg{start}-{end - 1}.parquet", delete=False
        )
        tmp.close()
        _pq.write_table(table, tmp.name)
        del table
        tmp_path = Path(tmp.name)
        tmp_manifest = tmp_path.with_suffix(".manifest.json")
        shutil.copy2(manifest_path, tmp_manifest)
        print(f"[STREAM_CHUNK] chunk ready: {tmp_path.name}", flush=True)
        yield tmp_path, (
            lambda p=tmp_path, m=tmp_manifest: (
                p.unlink(missing_ok=True),
                m.unlink(missing_ok=True),
            )
        )
        start = end


# Persist learned auxiliary-head outputs as parity diagnostics. They may
# explain or audit the model, but none may rewrite the sole serving authority:
# ``argmax(direction_logits)``.
_EXTRA_SIGMOID_HEADS = {
    # forward-output key -> persisted probability column (sigmoid of raw logit)
    "tradable_logit": "tradable_prob",
    "bad_path_logit": "bad_path_prob",
    "clean_edge_logit": "clean_edge_prob",
    "survival_logit": "survival_prob",
    "tf_agreement_logit": "tf_agreement_pred",
    "position_size_logit": "position_size_pred",
}

_RUNTIME_SNAPSHOT_FEATURE_SOURCES = {
    "geometry_channel_edge_pressure": "chart.geometry_channel_edge_pressure",
    "geometry_rising_support_rail_long_pressure": (
        "chart.geometry_rising_support_rail_long_pressure"
    ),
    "geometry_rising_support_rail_short_trap_pressure": (
        "chart.geometry_rising_support_rail_short_trap_pressure"
    ),
    "geometry_falling_resistance_rail_short_pressure": (
        "chart.geometry_falling_resistance_rail_short_pressure"
    ),
    "geometry_falling_resistance_rail_long_trap_pressure": (
        "chart.geometry_falling_resistance_rail_long_trap_pressure"
    ),
    "mtf_trend_evidence": "trend.mtf_confluence_trend_direction_score",
}


def _python_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_python_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_python_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _python_value(item) for key, item in value.items()}
    return value


def _runtime_head_evidence_for_row(
    row: Mapping[str, Any],
    *,
    bundle_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    direction_index = int(row["model_direction_index"])
    if direction_index not in SIDE_NAMES:
        raise RuntimeError(
            f"runtime head evidence has invalid direction index {direction_index}"
        )
    direction_calibration = bundle_metadata.get("direction_calibration")
    path_calibration = bundle_metadata.get("path_calibration")
    if (
        not isinstance(direction_calibration, Mapping)
        or direction_calibration.get("enabled") is not True
    ):
        raise RuntimeError(
            "candidate bundle lacks enabled direction_calibration for runtime evidence"
        )
    if (
        not isinstance(path_calibration, Mapping)
        or path_calibration.get("enabled") is not True
    ):
        raise RuntimeError(
            "candidate bundle lacks enabled path_calibration for runtime evidence"
        )

    evidence: dict[str, Any] = {
        "runtime_head_evidence_schema_version": (
            MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION
        ),
        "runtime_evidence_schema_version": (
            MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION
        ),
        "model_policy": MODEL_NATIVE_RUNTIME_POLICY,
        "decision_ts": str(pd.Timestamp(row["time"])),
        "session_id": int(row["session_id"]),
        "session": str(row["session"]),
        "entry_vol_regime_id": int(row["vol_regime_id"]),
        "entry_vol_regime": str(row["vol_regime"]),
        "entry_atr_bucket": int(row["atr_bucket"]),
        "entry_spread_bucket": int(row["spread_bucket"]),
        "entry_h4_trend_sign_cat": int(row["H4_trend_sign_cat"]),
        "entry_trend_regime_id": int(row["trend_regime_id"]),
        "entry_trend_regime": str(row["trend_regime"]),
        "model_direction_index": direction_index,
        "model_direction": str(row["model_direction"]),
        "selected_side": (
            direction_index
            if direction_index in MODEL_DIRECTION_TRADE_INDICES
            else None
        ),
        "public_trade_flat_decision": str(
            row["public_trade_flat_decision"]
        ),
        "specialist_names": list(MODEL_NATIVE_TRAINING_SPECIALISTS),
        "calibration_version": str(direction_calibration["version"]),
        "direction_calibration_enabled": True,
        "direction_calibration_temperature": float(
            direction_calibration["temperature"]
        ),
        "direction_calibration_bias": _python_value(
            direction_calibration["bias"]
        ),
        "path_calibration_enabled": True,
        "path_calibration": project_model_native_path_calibration(
            path_calibration,
            context="SELECTIVE_EDGE_RUNTIME_PATH_CALIBRATION",
        ),
    }
    same_name_fields = (
        MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS
        - {"sizing_authority_contract"}
        - set(evidence)
    )
    missing = sorted(field for field in same_name_fields if field not in row)
    if missing:
        raise RuntimeError(
            "prediction row cannot form exact runtime head evidence; "
            f"missing={missing}"
        )
    for field in sorted(same_name_fields):
        evidence[field] = _python_value(row[field])
    return evidence
_EXTRA_RAW_HEADS = {
    # forward-output key -> persisted raw-value column
    "mfe_first_n": "mfe_first_n_pred",
    "path_quality_log_var": "path_quality_log_var",
}
# Multi-dim heads -> per-index columns <key>_{i} (widths: dip 18, forecast 4,
# timing 12, tail_risk 6, vol_forecast 3 — taken from the tensor itself).
_EXTRA_VECTOR_HEADS = {
    "dip_pred": 18,
    "forecast_pred": 4,
    "timing_pred": MODEL_NATIVE_TIMING_OUTPUT_DIM,
    "tail_risk_pred": 6,
    "vol_forecast_pred": 3,
}


def _derived_serve_parity_outputs(
    outputs: Mapping[str, torch.Tensor],
    *,
    path_quality_scale: float,
) -> dict[str, np.ndarray]:
    """Return exact derived tensors required by the live forward parity schema."""

    if not np.isfinite(path_quality_scale) or path_quality_scale <= 0.0:
        raise RuntimeError("path_quality_scale must be finite-positive")
    path_log_var = _tensor_np(
        outputs, "path_quality_log_var", width=1
    ).reshape(-1)
    path_std = (
        float(path_quality_scale) * np.exp(0.5 * path_log_var)
    ).astype(np.float32)
    if not np.isfinite(path_std).all():
        raise RuntimeError(
            "path_quality_log_var produced non-finite path_quality_std"
        )
    mtf_logits = _tensor_np(outputs, "mtf_dir_logits", width=3)
    mtf_probs = _softmax_np(mtf_logits).astype(np.float32)
    if not np.isfinite(mtf_probs).all():
        raise RuntimeError("mtf_dir_logits produced non-finite mtf_dir_probs")
    return {
        "path_quality_std": path_std,
        "mtf_dir_probs": mtf_probs,
    }


def _predict_bundle(
    *,
    bundle_dir: Path,
    split_manifests: dict[str, Path],
    split_parquets: dict[str, Path],
    splits: list[str],
    model_name: str,
    device: torch.device,
    batch_size: int,
    m5_prebuilt_path: Path,
    evidence_stage: str,
    stream_chunk_rows: int = 0,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    failures: list[str] = []
    bundle = load_entry_v10_ctx_bundle(bundle_dir=bundle_dir, device=str(device))
    model = bundle.transformer_model
    model.eval()
    meta = dict(bundle.metadata)
    run_lineage = meta.get("run_lineage")
    if (
        model_name == EVALUATION_MODEL_NAME
        and (
            not isinstance(run_lineage, Mapping)
            or run_lineage.get("training_profile") != "candidate"
            or run_lineage.get("requested_subsample_rows") != 0
            or run_lineage.get("physical_train_rows")
            != run_lineage.get("effective_train_rows")
        )
    ):
        raise RuntimeError(
            "candidate evaluation requires a full-population candidate-profile bundle"
        )
    require_model_direction_decision_contract(
        meta,
        context=f"candidate bundle {Path(bundle_dir).expanduser().resolve()}",
    )
    signal_contract = meta.get("model_native_signal_contract")
    if not isinstance(signal_contract, dict):
        raise RuntimeError("candidate bundle lacks model_native_signal_contract")
    require_model_native_signal_contract(
        signal_contract,
        context="SELECTIVE_EDGE_BUNDLE",
    )
    if int(meta.get("seq_input_dim") or -1) != MODEL_NATIVE_SIGNAL_DIM or int(
        meta.get("snap_input_dim") or -1
    ) != MODEL_NATIVE_SIGNAL_DIM:
        raise RuntimeError(
            "candidate bundle must expose exact seq/snap width "
            f"{MODEL_NATIVE_SIGNAL_DIM}"
        )
    if "seq_len" not in meta or int(meta["seq_len"]) <= 0:
        raise RuntimeError("candidate bundle lacks a positive contracted seq_len")
    seq_len = int(meta["seq_len"])
    dataset_kwargs = _bundle_dataset_kwargs(meta, m5_prebuilt_path)
    hier_meta = meta.get("hierarchical_entry_heads") if isinstance(meta.get("hierarchical_entry_heads"), dict) else {}
    utility_scale_bps = (
        float(hier_meta["side_utility_scale_bps"])
        if "side_utility_scale_bps" in hier_meta
        else None
    )
    mae_scale_bps = (
        float(hier_meta["side_mae_scale_bps"])
        if "side_mae_scale_bps" in hier_meta
        else None
    )
    direction_calibration = meta.get("direction_calibration")
    path_calibration = meta.get("path_calibration")
    calibrated_runtime_bundle = bool(
        isinstance(direction_calibration, Mapping)
        and direction_calibration.get("enabled") is True
        and isinstance(path_calibration, Mapping)
        and path_calibration.get("enabled") is True
    )
    if evidence_stage not in EVIDENCE_STAGES:
        raise RuntimeError(
            f"unsupported prediction evidence stage: {evidence_stage!r}"
        )
    if evidence_stage == "runtime_authoritative" and not calibrated_runtime_bundle:
        raise RuntimeError(
            "runtime-authoritative prediction evidence requires enabled "
            "direction and path calibration"
        )
    runtime_head_ready = evidence_stage == "runtime_authoritative"
    path_inference_calibration = (
        project_model_native_path_calibration(
            path_calibration,
            context="SELECTIVE_EDGE_BUNDLE_PATH_CALIBRATION",
        )
        if runtime_head_ready
        else None
    )
    ordered_signal_names = tuple(
        str(name) for name in meta.get("ordered_signal_names") or ()
    )
    signal_positions = {
        name: index for index, name in enumerate(ordered_signal_names)
    }
    if runtime_head_ready:
        if len(ordered_signal_names) != MODEL_NATIVE_SIGNAL_DIM or len(
            set(ordered_signal_names)
        ) != MODEL_NATIVE_SIGNAL_DIM:
            raise RuntimeError(
                "candidate bundle ordered_signal_names is not the exact unique "
                f"{MODEL_NATIVE_SIGNAL_DIM}-signal contract"
            )
        missing_runtime_features = sorted(
            set(_RUNTIME_SNAPSHOT_FEATURE_SOURCES.values())
            - set(signal_positions)
        )
        if missing_runtime_features:
            raise RuntimeError(
                "candidate bundle lacks runtime snapshot signal features: "
                f"{missing_runtime_features}"
            )
    rows: list[pd.DataFrame] = []

    for split in splits:
        manifest_path = split_manifests[split]
        parquet_path = split_parquets[split]
        for _chunk_path, _chunk_cleanup in _iter_split_chunks(
            parquet_path,
            manifest_path,
            int(stream_chunk_rows),
        ):
            try:
                dataset = EntryV10CtxDataset(
                    parquet_path=_chunk_path,
                    seq_len=seq_len,
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
                # Dynamic per-head buffers retain learned auxiliary evidence.
                extra_chunks: dict[str, list[np.ndarray]] = {}
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
                        for key, value in out.items():
                            if hasattr(value, "detach") and not bool(torch.isfinite(value).all().item()):
                                failures.append(f"{model_name}/{split}: non-finite model output {key}")
                        forbidden_outputs = sorted(
                            {"anchor_logits", "delta_logits", "anchor_gate"}.intersection(out)
                        )
                        if forbidden_outputs:
                            raise RuntimeError(
                                "model-native bundle emitted forbidden legacy direction "
                                f"outputs: {forbidden_outputs}"
                            )
                        decision = _canonical_live_decision_evidence(out)
                        if runtime_head_ready:
                            snap_values = (
                                batch["snap_x"].detach().cpu().float().numpy()
                            )
                            if (
                                snap_values.ndim != 2
                                or snap_values.shape[1]
                                != MODEL_NATIVE_SIGNAL_DIM
                                or not np.isfinite(snap_values).all()
                            ):
                                raise RuntimeError(
                                    "candidate snap_x is not the exact finite "
                                    f"{MODEL_NATIVE_SIGNAL_DIM}-signal state"
                                )
                            for (
                                evidence_name,
                                signal_name,
                            ) in _RUNTIME_SNAPSHOT_FEATURE_SOURCES.items():
                                extra_chunks.setdefault(
                                    evidence_name, []
                                ).append(
                                    snap_values[:, signal_positions[signal_name]]
                                )
                        pred_direction = decision["model_direction_index"]
                        public_pair_decision = decision[
                            "public_trade_flat_decision_index"
                        ]
                        chunks["p_long"].append(decision["p_long"])
                        chunks["p_short"].append(decision["p_short"])
                        chunks["p_flat"].append(decision["p_flat"])
                        chunks["edge_score"].append(decision["edge_score"])
                        chunks["trade_side"].append(pred_direction)
                        chunks["pred_direction"].append(pred_direction)
                        chunks["y_direction"].append(batch["y"].detach().cpu().numpy().astype(np.int64))
                        chunks["ctx_cat"].append(batch["ctx_cat"].detach().cpu().numpy().astype(np.int64))
                        chunks["path_quality_pred"].append(out["path_quality"].detach().cpu().float().numpy().reshape(-1))
                        chunks["bad_path_prob"].append(torch.sigmoid(out["bad_path_logit"]).detach().cpu().float().numpy().reshape(-1))
                        extra_chunks.setdefault("direction_logits", []).append(
                            decision["direction_logits"]
                        )
                        extra_chunks.setdefault("direction_probs", []).append(
                            decision["direction_probs"]
                        )
                        extra_chunks.setdefault("model_direction_index", []).append(
                            decision["model_direction_index"]
                        )
                        extra_chunks.setdefault("public_trade_flat_decision_logits", []).append(
                            decision["public_trade_flat_decision_logits"]
                        )
                        extra_chunks.setdefault("public_trade_flat_decision_probs", []).append(
                            decision["public_trade_flat_decision_probs"]
                        )
                        extra_chunks.setdefault("public_trade_flat_decision_index", []).append(
                            decision["public_trade_flat_decision_index"]
                        )
                        extra_chunks.setdefault("public_trade_probability", []).append(
                            decision["p_trade"]
                        )
                        extra_chunks.setdefault("public_flat_probability", []).append(
                            decision["p_flat_hier"]
                        )
                        extra_chunks.setdefault("public_trade_flat_margin", []).append(
                            decision["public_trade_flat_decision_logits"][:, 0]
                            - decision["public_trade_flat_decision_logits"][:, 1]
                        )
                        extra_chunks.setdefault("public_trade_flat_hard_decision", []).append(
                            public_pair_decision
                        )
                        for _out_key, _width in (
                            ("raw_direction_logits", 3),
                            ("path_quality", 1),
                            ("bad_path_logit", 1),
                            *DIRECTION_EVIDENCE_INPUTS,
                        ):
                            extra_chunks.setdefault(_out_key, []).append(
                                _tensor_np(out, _out_key, width=_width)
                            )
                        # Persist auxiliary-head diagnostics without composing
                        # another direction decision.
                        for _out_key, _col in _EXTRA_SIGMOID_HEADS.items():
                            _raw = _tensor_np(out, _out_key, width=1)
                            extra_chunks.setdefault(_col, []).append(
                                _sigmoid_np(_raw.reshape(-1))
                            )
                            if _out_key == "tf_agreement_logit":
                                extra_chunks.setdefault(
                                    "tf_agreement_prob", []
                                ).append(_sigmoid_np(_raw.reshape(-1)))
                            if _out_key == "position_size_logit":
                                extra_chunks.setdefault(
                                    "position_size_logit", []
                                ).append(_raw.reshape(-1))
                        for _out_key, _col in _EXTRA_RAW_HEADS.items():
                            _raw = _tensor_np(out, _out_key, width=1).reshape(-1)
                            if _col != _out_key:
                                extra_chunks.setdefault(_col, []).append(_raw)
                        _derived_parity = _derived_serve_parity_outputs(
                            out,
                            path_quality_scale=(
                                float(
                                    path_inference_calibration[
                                        "path_quality_scale"
                                    ]
                                )
                                if path_inference_calibration is not None
                                else 1.0
                            ),
                        )
                        extra_chunks.setdefault("path_quality_std", []).append(
                            _derived_parity["path_quality_std"]
                        )
                        for _out_key, _width in _EXTRA_VECTOR_HEADS.items():
                            _vec = _tensor_np(out, _out_key, width=_width)
                            for _i in range(_width):
                                extra_chunks.setdefault(f"{_out_key}_{_i}", []).append(_vec[:, _i])
                        _mtf_logits = _tensor_np(out, "mtf_dir_logits", width=3)
                        extra_chunks.setdefault("mtf_long_minus_short", []).append(
                            (_mtf_logits[:, 0] - _mtf_logits[:, 1]).astype(np.float32)
                        )
                        _mtf = _derived_parity["mtf_dir_probs"]
                        extra_chunks.setdefault("mtf_dir_probs", []).append(_mtf)
                        extra_chunks.setdefault("mtf_p_long", []).append(_mtf[:, 0])
                        extra_chunks.setdefault("mtf_p_short", []).append(_mtf[:, 1])
                        extra_chunks.setdefault("mtf_p_flat", []).append(_mtf[:, 2])
                        specialist_gate = _tensor_np(
                            out,
                            "specialist_gate",
                            width=len(MODEL_NATIVE_TRAINING_SPECIALISTS),
                        )
                        extra_chunks.setdefault("specialist_gate", []).append(
                            specialist_gate
                        )
                        extra_chunks.setdefault("tf_gate", []).append(
                            _tensor_np(out, "tf_gate", width=5)
                        )
                        extra_chunks.setdefault(
                            "family_tf_cooperation_gate", []
                        ).append(
                            _tensor_np(
                                out,
                                "family_tf_cooperation_gate",
                                width=(
                                    len(MODEL_NATIVE_TRAINING_SPECIALISTS) * 5
                                ),
                            )
                        )
                        _feature_gate = out.get("family_tf_feature_gate")
                        _expected_feature_width = int(
                            len(meta["multi_tf"]["feature_names"])
                        )
                        if (
                            not isinstance(_feature_gate, torch.Tensor)
                            or _feature_gate.ndim != 3
                            or tuple(_feature_gate.shape[1:])
                            != (5, _expected_feature_width)
                            or not bool(torch.isfinite(_feature_gate).all().item())
                        ):
                            raise RuntimeError(
                                "exact model-native output family_tf_feature_gate "
                                f"invalid: observed={getattr(_feature_gate, 'shape', None)} "
                                f"expected=(*,5,{_expected_feature_width})"
                            )
                        extra_chunks.setdefault(
                            "family_tf_feature_gate", []
                        ).append(
                            _feature_gate.detach()
                            .cpu()
                            .float()
                            .numpy()
                            .reshape(int(_feature_gate.shape[0]), -1)
                        )
                        trendline_rail_logits = _tensor_np(out, "trendline_rail_logits", width=6)
                        trendline_rail_probs = _sigmoid_np(trendline_rail_logits)
                        extra_chunks.setdefault("trendline_rail_probs", []).append(
                            trendline_rail_probs
                        )
                        for _name, _index in (
                            ("trendline_rail_rising_support_prob", 0),
                            ("trendline_rail_falling_resistance_prob", 1),
                            ("trendline_rail_countertrend_short_trap_prob", 2),
                            ("trendline_rail_countertrend_long_trap_prob", 3),
                            ("trendline_rail_short_early_failure_prob", 4),
                            ("trendline_rail_long_early_failure_prob", 5),
                        ):
                            extra_chunks.setdefault(_name, []).append(trendline_rail_probs[:, _index])
                        _tensor_np(out, "trade_logit", width=1)
                        side_logits = _tensor_np(out, "side_logits", width=2)
                        side_utility = _tensor_np(out, "side_utility", width=2)
                        side_bad_path_logit = _tensor_np(out, "side_bad_path_logit", width=2)
                        side_mae = _tensor_np(out, "side_mae", width=2)
                        side_validity_logit = _tensor_np(out, "side_validity_logit", width=2)
                        extra_chunks.setdefault("p_trade", []).append(decision["p_trade"])
                        extra_chunks.setdefault("p_flat_hier", []).append(
                            decision["p_flat_hier"]
                        )
                        side_probs = _softmax_np(side_logits)
                        extra_chunks.setdefault("side_probs", []).append(side_probs)
                        extra_chunks.setdefault("p_long_given_trade", []).append(side_probs[:, 0])
                        extra_chunks.setdefault("p_short_given_trade", []).append(side_probs[:, 1])
                        side_uncertainty = (1.0 - np.maximum(side_probs[:, 0], side_probs[:, 1])).astype(np.float32)
                        extra_chunks.setdefault("side_uncertainty", []).append(side_uncertainty)
                        side_bad = _sigmoid_np(side_bad_path_logit)
                        extra_chunks.setdefault("long_bad_path_prob", []).append(side_bad[:, 0])
                        extra_chunks.setdefault("short_bad_path_prob", []).append(side_bad[:, 1])
                        side_validity = _sigmoid_np(side_validity_logit)
                        extra_chunks.setdefault("long_validity_prob", []).append(side_validity[:, 0])
                        extra_chunks.setdefault("short_validity_prob", []).append(side_validity[:, 1])
                        if utility_scale_bps is None or not np.isfinite(utility_scale_bps) or utility_scale_bps <= 0.0:
                            raise RuntimeError("side_utility output lacks a finite positive contracted scale")
                        long_util = (side_utility[:, 0] * utility_scale_bps).astype(np.float32)
                        short_util = (side_utility[:, 1] * utility_scale_bps).astype(np.float32)
                        extra_chunks.setdefault("long_path_utility_pred_bps", []).append(long_util)
                        extra_chunks.setdefault("short_path_utility_pred_bps", []).append(short_util)
                        if mae_scale_bps is None or not np.isfinite(mae_scale_bps) or mae_scale_bps <= 0.0:
                            raise RuntimeError("side_mae output lacks a finite positive contracted scale")
                        extra_chunks.setdefault("long_expected_mae_bps", []).append(
                            np.maximum(side_mae[:, 0] * mae_scale_bps, 0.0).astype(np.float32)
                        )
                        extra_chunks.setdefault("short_expected_mae_bps", []).append(
                            np.maximum(side_mae[:, 1] * mae_scale_bps, 0.0).astype(np.float32)
                        )

                arrays = {key: np.concatenate(value, axis=0) if value else np.zeros((0,), dtype=np.float32) for key, value in chunks.items()}
                extra_arrays = {col: np.concatenate(vals, axis=0) for col, vals in extra_chunks.items()}
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
                # Auxiliary outputs are evidence columns only.
                for _col, _arr in extra_arrays.items():
                    if _arr.ndim == 2 and _arr.shape[1] == 1:
                        frame[_col] = _arr[:, 0].astype(np.float32)
                    elif _arr.ndim == 2:
                        frame[_col] = [row.astype(np.float32).tolist() for row in _arr]
                    elif (
                        _col.endswith("_side")
                        or _col.endswith("_index")
                        or _col == "public_trade_flat_hard_decision"
                    ):
                        frame[_col] = _arr.astype(np.int64)
                    else:
                        frame[_col] = _arr.astype(np.float32)
                frame["selection_score_mode"] = MODEL_DIRECTION_SELECTION_MODE
                direction_probabilities = frame[
                    ["p_long", "p_short", "p_flat"]
                ].to_numpy(dtype=np.float32)
                direction_indices = frame["pred_direction"].to_numpy(dtype=np.int64)
                if np.any((direction_indices < 0) | (direction_indices > 2)):
                    raise RuntimeError(
                        f"{model_name}/{split}: invalid model direction index"
                    )
                frame["selection_score"] = direction_probabilities[
                    np.arange(n, dtype=np.int64), direction_indices
                ].astype(np.float32)
                frame["model_direction"] = frame["pred_direction"].map(SIDE_NAMES)
                frame["public_trade_flat_decision"] = np.where(
                    frame["public_trade_flat_decision_index"].to_numpy(
                        dtype=np.int64
                    )
                    == 0,
                    "TRADE",
                    "FLAT",
                )
                required_targets = {
                    "y_long_path_utility_bps",
                    "y_short_path_utility_bps",
                    "path_quality_bps",
                    "y_bad_path",
                    *MODEL_NATIVE_TIMING_TARGET_COLUMNS,
                    *ACTION_VALUE_TARGET_COLUMNS,
                }
                missing_targets = sorted(required_targets - set(frame.columns))
                if missing_targets:
                    raise RuntimeError(
                        f"{model_name}/{split}: dataset lacks required target evidence: "
                        f"{missing_targets}"
                    )
                if (
                    ctx_cat.ndim != 2
                    or ctx_cat.shape[1] != len(MODEL_NATIVE_CTX_CAT_FIELDS)
                ):
                    raise RuntimeError(
                        f"{model_name}/{split}: ctx_cat is not the exact five-field contract"
                    )
                session_ids = ctx_cat[
                    :, MODEL_NATIVE_CTX_CAT_INDEX_BY_NAME["session_id"]
                ].astype(np.int64)
                unknown_sessions = sorted(set(session_ids) - set(SESSION_NAMES))
                if unknown_sessions:
                    raise RuntimeError(
                        f"{model_name}/{split}: unknown session ids {unknown_sessions}"
                    )
                frame["session"] = [SESSION_NAMES[int(x)] for x in session_ids]
                frame["session_id"] = session_ids
                vol_regime_ids = ctx_cat[
                    :, MODEL_NATIVE_CTX_CAT_INDEX_BY_NAME["vol_regime_id"]
                ].astype(np.int64)
                h4_trend_ids = ctx_cat[
                    :, MODEL_NATIVE_CTX_CAT_INDEX_BY_NAME["H4_trend_sign_cat"]
                ].astype(np.int64)
                if "trend_regime_id" not in frame.columns:
                    raise RuntimeError(
                        f"{model_name}/{split}: exact entry trend_regime_id is missing"
                    )
                trend_regime_numeric = pd.to_numeric(
                    frame["trend_regime_id"],
                    errors="coerce",
                ).to_numpy(dtype=np.float64)
                if (
                    not np.isfinite(trend_regime_numeric).all()
                    or not np.array_equal(
                        trend_regime_numeric,
                        trend_regime_numeric.astype(np.int64),
                    )
                ):
                    raise RuntimeError(
                        f"{model_name}/{split}: invalid entry trend_regime_id"
                    )
                trend_regime_ids = trend_regime_numeric.astype(np.int64)
                if not np.isin(
                    vol_regime_ids,
                    MODEL_NATIVE_CTX_CAT_DOMAINS["vol_regime_id"],
                ).all():
                    raise RuntimeError(
                        f"{model_name}/{split}: invalid entry vol-regime ids"
                    )
                if not np.isin(
                    h4_trend_ids,
                    MODEL_NATIVE_CTX_CAT_DOMAINS["H4_trend_sign_cat"],
                ).all():
                    raise RuntimeError(
                        f"{model_name}/{split}: invalid entry H4-trend ids"
                    )
                if not np.isin(
                    trend_regime_ids,
                    np.arange(len(MODEL_NATIVE_ENTRY_TREND_REGIME_NAMES)),
                ).all():
                    raise RuntimeError(
                        f"{model_name}/{split}: invalid entry trend-regime ids"
                    )
                frame["vol_regime_id"] = vol_regime_ids
                frame["vol_regime"] = [
                    MODEL_NATIVE_ENTRY_VOL_REGIME_NAMES[int(value)]
                    for value in vol_regime_ids
                ]
                atr_bucket_ids = ctx_cat[
                    :, MODEL_NATIVE_CTX_CAT_INDEX_BY_NAME["atr_bucket"]
                ].astype(np.int64)
                spread_bucket_ids = ctx_cat[
                    :, MODEL_NATIVE_CTX_CAT_INDEX_BY_NAME["spread_bucket"]
                ].astype(np.int64)
                if not np.isin(
                    atr_bucket_ids,
                    MODEL_NATIVE_CTX_CAT_DOMAINS["atr_bucket"],
                ).all() or not np.isin(
                    spread_bucket_ids,
                    MODEL_NATIVE_CTX_CAT_DOMAINS["spread_bucket"],
                ).all():
                    raise RuntimeError(
                        f"{model_name}/{split}: invalid entry rank-bucket ids"
                    )
                frame["atr_bucket"] = atr_bucket_ids
                frame["spread_bucket"] = spread_bucket_ids
                frame["H4_trend_sign_cat"] = h4_trend_ids
                frame["trend_regime_id"] = trend_regime_ids
                frame["trend_regime"] = [
                    MODEL_NATIVE_ENTRY_TREND_REGIME_NAMES[int(value)]
                    for value in trend_regime_ids
                ]
                direction_ids = frame["pred_direction"].to_numpy(dtype=np.int64)
                if not set(direction_ids).issubset(SIDE_NAMES):
                    raise RuntimeError(
                        f"{model_name}/{split}: invalid direction ids "
                        f"{sorted(set(direction_ids) - set(SIDE_NAMES))}"
                    )
                frame["side"] = [SIDE_NAMES[int(x)] for x in direction_ids]
                frame["pnl_proxy_bps"] = _pnl_proxy_for_side(frame)
                if runtime_head_ready:
                    if "atr_bps" not in frame.columns:
                        raise RuntimeError(
                            f"{model_name}/{split}: exact runtime atr_bps is missing"
                        )
                    atr_values = pd.to_numeric(
                        frame["atr_bps"],
                        errors="coerce",
                    ).to_numpy(dtype=np.float64)
                    if not np.isfinite(atr_values).all() or np.any(
                        atr_values <= 0.0
                    ):
                        raise RuntimeError(
                            f"{model_name}/{split}: runtime atr_bps is not "
                            "finite-positive"
                        )
                    head_payloads: list[str] = []
                    head_hashes: list[str] = []
                    for row_index, row in frame.iterrows():
                        payload, payload_sha = (
                            encode_model_native_runtime_head_evidence(
                                _runtime_head_evidence_for_row(
                                    row,
                                    bundle_metadata=meta,
                                ),
                                context=(
                                    f"SELECTIVE_EDGE_{model_name}_{split}_"
                                    f"ROW_{row_index}"
                                ),
                            )
                        )
                        head_payloads.append(payload)
                        head_hashes.append(payload_sha)
                    frame["runtime_head_evidence_schema_version"] = (
                        MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION
                    )
                    frame["runtime_head_evidence_json"] = head_payloads
                    frame["runtime_head_evidence_sha256"] = head_hashes
                keep_cols = [
                    "split",
                    "model",
                    "time",
                    "y_direction",
                    "pred_direction",
                    "trade_side",
                    "side",
                    "session",
                    "session_id",
                    "vol_regime_id",
                    "vol_regime",
                    "atr_bucket",
                    "spread_bucket",
                    "H4_trend_sign_cat",
                    "trend_regime_id",
                    "trend_regime",
                    "edge_score",
                    "p_long",
                    "p_short",
                    "p_flat",
                    "selection_score_mode",
                    "selection_score",
                    "path_quality_pred",
                    "bad_path_prob",
                    "path_quality_bps",
                    "y_bad_path",
                    "pnl_proxy_bps",
                ]
                if runtime_head_ready:
                    keep_cols.extend(
                        [
                            "runtime_head_evidence_schema_version",
                            "runtime_head_evidence_json",
                            "runtime_head_evidence_sha256",
                            "atr_bps",
                        ]
                    )
                # Immutable smoke audit targets.  These remain evidence only;
                # none is allowed to rewrite the model's final direction.
                for target_col in (
                    "mfe_first_n_bps",
                    "y_tradable",
                    "y_selector_long_mask",
                    "y_selector_short_mask",
                    "y_position_size_target",
                    *MODEL_NATIVE_TIMING_TARGET_COLUMNS,
                    *ACTION_VALUE_TARGET_COLUMNS,
                ):
                    if target_col not in frame.columns:
                        raise RuntimeError(
                            f"{model_name}/{split}: dataset lacks required smoke "
                            f"target evidence {target_col}"
                        )
                    keep_cols.append(target_col)
                if "y_forecast_ret_K24" in frame.columns:
                    keep_cols.append("y_forecast_ret_K24")
                for utility_col in (
                    "y_long_path_utility_bps",
                    "y_short_path_utility_bps",
                    "y_direction_long_score_bps",
                    "y_direction_short_score_bps",
                ):
                    if utility_col in frame.columns:
                        keep_cols.append(utility_col)
                # Keep every persisted auxiliary evidence column.
                keep_cols.extend(c for c in extra_arrays.keys() if c not in keep_cols)
                rows.append(frame[[c for c in keep_cols if c in frame.columns]])
            finally:
                _chunk_cleanup()

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
    contract_mode = MODEL_NATIVE_CONTRACT_MODE
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    m5_prebuilt = Path(args.m5_prebuilt_path).expanduser().resolve()
    mtf_cache_dir = Path(args.multi_tf_cache_dir).expanduser().resolve()
    for label, path, expected_kind in (
        ("bundle", bundle_dir, "dir"),
        ("dataset", dataset_dir, "dir"),
        ("M5 prebuilt", m5_prebuilt, "file"),
        ("multi-TF cache", mtf_cache_dir, "dir"),
    ):
        valid = path.is_dir() if expected_kind == "dir" else path.is_file()
        if not valid:
            raise RuntimeError(f"explicit {label} artifact is missing: {path}")
    requested_splits = [
        value.strip()
        for value in str(
            getattr(args, "splits", ",".join(DEFAULT_EVALUATION_SPLITS))
        ).split(",")
        if value.strip()
    ]
    if (
        not requested_splits
        or len(requested_splits) != len(set(requested_splits))
        or not set(requested_splits).issubset(EVALUATION_SPLITS)
    ):
        raise RuntimeError(
            "evaluation splits must be a unique non-empty subset of "
            f"{EVALUATION_SPLITS}"
        )
    splits = [
        split for split in EVALUATION_SPLITS if split in requested_splits
    ]
    evidence_stage = str(
        getattr(args, "evidence_stage", "runtime_authoritative")
    )
    if evidence_stage not in EVIDENCE_STAGES:
        raise RuntimeError(
            f"evidence_stage must be one of {EVIDENCE_STAGES}"
        )
    missing_split_arguments = [
        f"{split}_{suffix}"
        for split in splits
        for suffix in (
            "manifest_json",
            "manifest_sha256",
            "parquet",
            "parquet_sha256",
        )
        if not str(getattr(args, f"{split}_{suffix}", "") or "").strip()
    ]
    if missing_split_arguments:
        raise RuntimeError(
            "selected evaluation splits lack explicit artifact bindings: "
            f"{missing_split_arguments}"
        )
    split_bindings = {
        split: {
            "manifest_path": str(getattr(args, f"{split}_manifest_json")),
            "manifest_sha256": str(getattr(args, f"{split}_manifest_sha256")),
            "parquet_path": str(getattr(args, f"{split}_parquet")),
            "parquet_sha256": str(getattr(args, f"{split}_parquet_sha256")),
        }
        for split in splits
    }
    dataset_contract = _dataset_model_native_contract(
        dataset_dir,
        splits,
        split_bindings,
    )
    split_parquets = {
        split: Path(dataset_contract["splits"][split]["parquet_path"])
        for split in splits
    }
    split_manifests = {
        split: Path(dataset_contract["splits"][split]["manifest_path"])
        for split in splits
    }
    top_fracs = list(EVALUATION_TOP_FRACS)
    device = torch.device(_device_arg(args.device))
    _reject_retired_selection_environment()
    selection_score_mode = MODEL_DIRECTION_SELECTION_MODE
    exclude_sessions = tuple(
        s.strip() for s in str(getattr(args, "exclude_sessions", "") or "").split(",") if s.strip()
    )
    if exclude_sessions:
        raise RuntimeError(
            "external session exclusion is forbidden; session evidence "
            "must be fused into the model-native LONG/SHORT/FLAT decision"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GX1_V10_MULTI_TF_V4_CACHE_DIR"] = str(mtf_cache_dir)

    failures: list[str] = []
    feature_mask = {"enabled": False}
    all_predictions: list[pd.DataFrame] = []
    bundle_meta: dict[str, Any] = {}
    candidate, candidate_failures, bundle_meta = _predict_bundle(
        bundle_dir=bundle_dir,
        split_manifests=split_manifests,
        split_parquets=split_parquets,
        splits=splits,
        model_name=EVALUATION_MODEL_NAME,
        device=device,
        batch_size=int(args.batch_size),
        m5_prebuilt_path=m5_prebuilt,
        evidence_stage=evidence_stage,
        stream_chunk_rows=int(getattr(args, "stream_chunk_rows", 0) or 0),
    )
    all_predictions.append(candidate)
    failures.extend(candidate_failures)
    bundle_specialist_contract = _specialist_contract_snapshot(bundle_meta, contract_mode)
    failures.extend([f"candidate bundle: {failure}" for failure in bundle_specialist_contract["failures"]])
    if bundle_meta.get("model_native_signal_contract") != dataset_contract["contract"]:
        failures.append("bundle and dataset model-native signal contracts are not exact-equal")

    predictions = pd.concat([df for df in all_predictions if not df.empty], ignore_index=True) if all_predictions else pd.DataFrame()
    if predictions.empty:
        failures.append("no selective-edge predictions were produced")
    forbidden_prediction_columns = sorted(
        set(FORBIDDEN_LEGACY_BRIDGE_FIELDS)
        .union({"anchor_logits", "delta_logits", "anchor_gate"})
        .intersection(predictions.columns)
    )
    if forbidden_prediction_columns:
        failures.append(
            "prediction evidence contains forbidden legacy direction columns: "
            f"{forbidden_prediction_columns}"
        )
    metric_rows = build_metric_rows(predictions, top_fracs=top_fracs, exclude_sessions=exclude_sessions)
    metrics = pd.DataFrame(metric_rows)
    summary_payload = build_summary(predictions, metrics)
    event_created_utc = datetime.now(timezone.utc)
    timestamp = event_created_utc.strftime("%Y%m%dT%H%M%S%fZ")
    ready = not failures

    predictions_path = out_dir / f"selective_edge_predictions_{timestamp}.parquet"
    metrics_path = out_dir / f"selective_edge_metrics_{timestamp}.csv"
    summary_path = out_dir / f"ENTRY_CANDIDATE_SELECTIVE_EDGE_SUMMARY_{timestamp}.json"
    report_json_path = out_dir / f"ENTRY_CANDIDATE_SELECTIVE_EDGE_{timestamp}.json"
    report_md_path = out_dir / f"ENTRY_CANDIDATE_SELECTIVE_EDGE_{timestamp}.md"
    prediction_evidence: dict[str, Any] = {}
    if not predictions.empty:
        atomic_write_parquet_immutable(predictions, predictions_path)
        prediction_evidence = build_prediction_evidence_declaration(
            predictions_path=predictions_path,
            bundle_dir=bundle_dir,
            bundle_metadata=bundle_meta,
            requested_splits=splits,
        )
    atomic_write_text(metrics_path, metrics.to_csv(index=False))
    metrics_sha256 = _sha256_file(metrics_path)
    direction_decision_contract = require_model_direction_decision_contract(
        bundle_meta,
        context=f"candidate evaluator report bundle {bundle_dir}",
    )
    report = {
        "schema_version": "entry_candidate_selective_edge_v1",
        "created_utc": event_created_utc.isoformat(),
        "decision": "PASS" if ready else "FAIL",
        "contract_mode": contract_mode,
        "evidence_stage": evidence_stage,
        "bundle_dir": str(bundle_dir),
        "feature_mask_ablation": feature_mask,
        "dataset_dir": str(dataset_dir),
        "model_native_signal_contract": dataset_contract["contract"],
        "dataset_signal_contract": dataset_contract,
        "splits": summary_payload["splits"],
        "models": summary_payload["models"],
        "summaries": summary_payload["summaries"],
        "top_fracs": top_fracs,
        "selection_score_mode": selection_score_mode,
        "direction_decision_contract": direction_decision_contract,
        "bundle_seq_len": int(bundle_meta.get("seq_len") or 0),
        "bundle_seq_input_dim": int(bundle_meta["seq_input_dim"]),
        "bundle_snap_input_dim": int(bundle_meta["snap_input_dim"]),
        "bundle_specialist_fusion_enabled": bool((bundle_meta.get("specialist_fusion") or {}).get("enabled")) if isinstance(bundle_meta.get("specialist_fusion"), dict) else False,
        "bundle_specialist_contract": bundle_specialist_contract,
        "trainer_started": False,
        "promotion_shadow_live_allowed": False,
        "predictions_path": str(predictions_path),
        "prediction_evidence": prediction_evidence,
        "bundle_metadata_path": str(bundle_dir / "bundle_metadata.json"),
        "bundle_metadata_sha256": str(prediction_evidence.get("bundle_metadata_sha256") or ""),
        "model_state_dict_sha256": str(prediction_evidence.get("model_state_dict_sha256") or ""),
        "metrics_path": str(metrics_path),
        "metrics_sha256": metrics_sha256,
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
            "feature_mask_ablation": report["feature_mask_ablation"],
            "dataset_dir": report["dataset_dir"],
            "model_native_signal_contract": report["model_native_signal_contract"],
            "dataset_signal_contract": report["dataset_signal_contract"],
            "selection_score_mode": report["selection_score_mode"],
            "direction_decision_contract": report["direction_decision_contract"],
            "bundle_seq_len": report["bundle_seq_len"],
            "bundle_seq_input_dim": report["bundle_seq_input_dim"],
            "bundle_snap_input_dim": report["bundle_snap_input_dim"],
            "bundle_specialist_fusion_enabled": report["bundle_specialist_fusion_enabled"],
            "bundle_specialist_contract": report["bundle_specialist_contract"],
            "prediction_evidence": report["prediction_evidence"],
            "predictions_path": report["predictions_path"],
            "prediction_report_json": report["json_path"],
            "metrics_path": report["metrics_path"],
            "metrics_sha256": report["metrics_sha256"],
            "bundle_metadata_path": report["bundle_metadata_path"],
            "bundle_metadata_sha256": report["bundle_metadata_sha256"],
            "model_state_dict_sha256": report["model_state_dict_sha256"],
            "authoritative": True,
            "note": "immutable summary bound to the matching prediction report and parquet",
            "failures": failures,
        }
    )
    atomic_write_text(
        summary_path,
        json.dumps(summary_payload, indent=2, sort_keys=True, default=_json_default) + "\n",
    )
    atomic_write_text(
        report_json_path,
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
    )
    markdown_tmp = out_dir / f".{report_md_path.name}.render"
    _write_markdown(markdown_tmp, report)
    try:
        atomic_write_text(
            report_md_path,
            markdown_tmp.read_text(encoding="utf-8"),
        )
    finally:
        markdown_tmp.unlink(missing_ok=True)
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
    if failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument(
        "--splits",
        default=",".join(DEFAULT_EVALUATION_SPLITS),
        help=(
            "Exact comma-separated artifact role: train,val for sizing fit; "
            "test for OOS; val,test for evaluation."
        ),
    )
    ap.add_argument(
        "--evidence-stage",
        choices=EVIDENCE_STAGES,
        default="runtime_authoritative",
        help=(
            "runtime_authoritative requires calibrated runtime-head envelopes; "
            "pre_calibration emits non-authorizing V2 evidence explicitly."
        ),
    )
    for split in EVALUATION_SPLITS:
        ap.add_argument(f"--{split}-manifest-json")
        ap.add_argument(f"--{split}-manifest-sha256")
        ap.add_argument(f"--{split}-parquet")
        ap.add_argument(f"--{split}-parquet-sha256")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument(
        "--stream-chunk-rows",
        type=int,
        default=0,
        help=(
            "Memory-bounded evaluation: forward the split in row-chunks of this size "
            "(each dataset row carries its own nested sequence, so chunking is loss-free). "
            "0 = load the whole split (default, unchanged). Use for huge splits, e.g. the "
            "390K-row dense train-window forward (~78GB unchunked)."
        ),
    )
    ap.add_argument("--m5-prebuilt-path", required=True)
    ap.add_argument("--multi-tf-cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
