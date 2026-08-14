#!/usr/bin/env python3
"""Evaluate selective edge for an exact model-native seq513 candidate bundle.

This is a post-candidate evidence writer with two exact, fail-closed stages:
pre-adoption reads only VAL, while runtime_authoritative reads only TEST after
the frozen bundle has been proven. It ranks the model's own LONG/SHORT/FLAT
choices by selected raw-bps Q
and writes evidence for replay-readiness.

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
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CAT_INDEX_BY_NAME,
    MODEL_NATIVE_SIGNAL_DIM,
    require_model_native_signal_contract,
)
from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_MTF_CONTEXT_COUNT
from gx1.contracts.entry_fitted_q_v1 import (
    require_entry_fitted_q_production_economics_readiness,
)
from gx1.contracts.entry_exit_production_architecture_v1 import (
    current_entry_exit_architecture_observation,
    require_entry_exit_production_architecture,
)
from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
    MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION,
    MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION,
    MODEL_NATIVE_RUNTIME_POLICY,
    encode_model_native_runtime_head_evidence,
)
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_DIP_TARGET_COLUMNS,
    MODEL_NATIVE_FORECAST_TARGET_COLUMNS,
    MODEL_NATIVE_TAIL_RISK_TARGET_COLUMNS,
    MODEL_NATIVE_TIMING_OUTPUT_DIM,
    MODEL_NATIVE_TIMING_TARGET_COLUMNS,
    MODEL_NATIVE_VOL_FORECAST_TARGET_COLUMNS,
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
    UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM,
    UNIFIED_EXIT_ENTRY_REPRESENTATION_KEY,
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
EVIDENCE_STAGES = ("validation_research", "runtime_authoritative")
SELECTIVE_EDGE_STAGE_SPLITS = {
    "validation_research": ("val",),
    "runtime_authoritative": ("test",),
}
EVALUATION_SPLITS = tuple(
    split
    for stage in EVIDENCE_STAGES
    for split in SELECTIVE_EDGE_STAGE_SPLITS[stage]
)
EVALUATION_TOP_FRACS = (0.05, 0.10)
EVALUATION_MODEL_NAME = "candidate"
SELECTIVE_EDGE_MAX_STREAM_CHUNK_ROWS = 4096
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


def _require_entry_q_ssot(
    out: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return the sole unique raw-bps Entry-Q decision tensor."""

    forbidden = sorted({"anchor_logits", "delta_logits", "anchor_gate"}.intersection(out))
    if forbidden:
        raise RuntimeError(
            f"model-native direction output contains forbidden legacy keys: {forbidden}"
        )
    q_values = out.get("entry_action_q_bps")
    if not isinstance(q_values, torch.Tensor):
        raise RuntimeError("model-native Entry-Q output is missing")
    if q_values.ndim != 2 or q_values.shape[1] != 3:
        raise RuntimeError(
            "entry_action_q_bps must have shape (B,3); "
            f"got {tuple(q_values.shape)}"
        )
    if not bool(torch.isfinite(q_values).all().item()):
        raise RuntimeError("model-native Entry-Q contains non-finite values")
    winner_counts = q_values.eq(
        q_values.amax(dim=1, keepdim=True)
    ).sum(dim=1)
    tied_rows = int((winner_counts != 1).sum().item())
    if tied_rows:
        raise RuntimeError(
            "Entry-Q has no unique top action; "
            f"rows={tied_rows}"
        )
    return q_values


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
    """Materialize the exact raw-Q/argmax surface consumed by live."""
    q_tensor = _require_entry_q_ssot(out)
    q_values = q_tensor.detach().cpu().float().numpy()
    direction_index = np.argmax(q_values, axis=1).astype(np.int64)
    row_index = np.arange(direction_index.shape[0], dtype=np.int64)
    selection_score = q_values[row_index, direction_index].astype(np.float32)
    edge_score = (
        np.maximum(
            q_values[:, MODEL_DIRECTION_LONG_INDEX],
            q_values[:, MODEL_DIRECTION_SHORT_INDEX],
        )
        - q_values[:, MODEL_DIRECTION_FLAT_INDEX]
    ).astype(np.float32)
    ordered = np.sort(q_values, axis=1)
    return {
        "entry_action_q_bps": q_values,
        "entry_action_q_margin_bps": (ordered[:, -1] - ordered[:, -2]).astype(np.float32),
        "model_direction_index": direction_index,
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


def _concatenate_evidence_chunks(
    chunks: Mapping[str, list[np.ndarray]],
    *,
    expected_rows: int,
) -> dict[str, np.ndarray]:
    combined: dict[str, np.ndarray] = {}
    for column, values in chunks.items():
        arrays = [np.asarray(value) for value in values]
        trailing_shapes = {array.shape[1:] for array in arrays if array.ndim > 0}
        if not arrays or any(array.ndim == 0 for array in arrays) or len(trailing_shapes) != 1:
            raise RuntimeError(
                f"prediction evidence chunks have incompatible shapes: column={column} "
                f"shapes={[array.shape for array in arrays]}"
            )
        value = np.concatenate(arrays, axis=0)
        if int(value.shape[0]) != int(expected_rows):
            raise RuntimeError(
                f"prediction evidence row mismatch: column={column} "
                f"observed={value.shape[0]} expected={expected_rows}"
            )
        combined[column] = value
    return combined


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


def _realized_net_policy_pnl(frame: pd.DataFrame) -> np.ndarray:
    """Select immutable net executable OOS PnL for the raw-Q action.

    These columns are intentionally not synthesized from old path-utility
    labels. Until the causal quote/cost replay owner emits them, evaluation
    fails closed rather than reporting a gross same-close proxy as PnL.
    """

    required = {
        "pred_direction",
        "realized_net_long_pnl_bps",
        "realized_net_short_pnl_bps",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"selective-edge frame lacks executable net OOS evidence: {missing}"
        )
    side = pd.to_numeric(frame["pred_direction"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    long_score = pd.to_numeric(
        frame["realized_net_long_pnl_bps"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    short_score = pd.to_numeric(
        frame["realized_net_short_pnl_bps"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(side).all() or not np.isfinite(long_score).all() or not np.isfinite(
        short_score
    ).all():
        raise RuntimeError("executable net OOS evidence contains non-finite values")
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
            "mean_edge_score": None,
        }
    pnl = pd.to_numeric(frame["realized_net_policy_pnl_bps"], errors="coerce")
    return {
        "split": split,
        "model": model,
        "scope": scope,
        "top_frac": float(top_frac),
        "group": group,
        "n": int(len(frame)),
        "mean_pnl_bps": _safe_mean(pnl),
        "win_rate": _safe_mean((pnl > 0.0).astype(float)),
        "mean_edge_score": _safe_mean(frame["edge_score"]),
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
    if tuple(splits) not in tuple(SELECTIVE_EDGE_STAGE_SPLITS.values()):
        raise RuntimeError(
            "selective-edge dataset contract requires one exact stage split"
        )
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
    """Yield exact bounded row-group chunks or fail before reading data."""
    import pyarrow.parquet as _pq
    import tempfile

    if (
        isinstance(stream_chunk_rows, bool)
        or not 1 <= int(stream_chunk_rows) <= SELECTIVE_EDGE_MAX_STREAM_CHUNK_ROWS
    ):
        raise RuntimeError(
            "SELECTIVE_EDGE_STREAM_CHUNK_ROWS_INVALID: expected "
            f"1..{SELECTIVE_EDGE_MAX_STREAM_CHUNK_ROWS}"
        )
    pf = _pq.ParquetFile(parquet_path)
    n_groups = pf.metadata.num_row_groups
    group_rows = [pf.metadata.row_group(i).num_rows for i in range(n_groups)]
    oversized = [rows for rows in group_rows if rows > stream_chunk_rows]
    if oversized:
        raise RuntimeError(
            "SELECTIVE_EDGE_PARQUET_ROW_GROUP_EXCEEDS_MEMORY_CONTRACT: "
            f"largest={max(oversized)} limit={stream_chunk_rows}"
        )
    start = 0
    while start < n_groups:
        rows_acc = 0
        end = start
        while end < n_groups and rows_acc + group_rows[end] <= stream_chunk_rows:
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
# ``unique_argmax(entry_action_q_bps)``.
_EXTRA_SIGMOID_HEADS = {
    # forward-output key -> persisted probability column (sigmoid of raw logit)
    "position_size_logit": "position_size_pred",
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
    if any(key in bundle_metadata for key in ("direction_calibration", "path_calibration")):
        raise RuntimeError("candidate Entry-Q bundle carries retired calibration")

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
        "model_direction_index": direction_index,
        "model_direction": str(row["model_direction"]),
        "selected_side": (
            direction_index
            if direction_index in MODEL_DIRECTION_TRADE_INDICES
            else None
        ),
        "specialist_names": list(MODEL_NATIVE_TRAINING_SPECIALISTS),
    }
    same_name_fields = (
        MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS
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
# Multi-dimensional genuine auxiliary heads; widths derive from target owners.
_EXTRA_VECTOR_HEADS = {
    "dip_pred": len(MODEL_NATIVE_DIP_TARGET_COLUMNS),
    "forecast_pred": len(MODEL_NATIVE_FORECAST_TARGET_COLUMNS),
    "timing_pred": MODEL_NATIVE_TIMING_OUTPUT_DIM,
    "tail_risk_pred": len(MODEL_NATIVE_TAIL_RISK_TARGET_COLUMNS),
    "vol_forecast_pred": len(MODEL_NATIVE_VOL_FORECAST_TARGET_COLUMNS),
}


def _require_evaluation_lineage(
    run_lineage: Any,
    *,
    evidence_stage: str,
) -> None:
    if evidence_stage not in EVIDENCE_STAGES:
        raise RuntimeError(
            f"unsupported prediction evidence stage: {evidence_stage!r}"
        )
    if not isinstance(run_lineage, Mapping):
        raise RuntimeError("evaluation bundle lacks exact training run lineage")

    requested = run_lineage.get("requested_subsample_rows")
    physical = run_lineage.get("physical_train_rows")
    effective = run_lineage.get("effective_train_rows")
    exact_counts = all(type(value) is int for value in (requested, physical, effective))
    full_candidate = bool(
        exact_counts
        and run_lineage.get("training_profile") == "candidate"
        and requested == 0
        and physical == effective
        and physical > 0
    )
    bounded_smoke = bool(
        exact_counts
        and run_lineage.get("training_profile") == "smoke"
        and requested > 0
        and physical > effective > 0
        and effective <= requested
    )

    if evidence_stage == "runtime_authoritative" and not full_candidate:
        raise RuntimeError(
            "runtime-authoritative evaluation requires a full-population "
            "candidate-profile bundle"
        )
    if evidence_stage == "validation_research" and not (
        full_candidate or bounded_smoke
    ):
        raise RuntimeError(
            "validation research requires either a bounded smoke-profile "
            "bundle or a full-population candidate-profile bundle"
        )


def _require_selective_edge_stage_split(
    *,
    evidence_stage: Any,
    split_spec: Any,
) -> list[str]:
    """Resolve the only legal split for one evidence stage."""

    stage = str(evidence_stage or "")
    if stage not in SELECTIVE_EDGE_STAGE_SPLITS:
        raise RuntimeError(
            "SELECTIVE_EDGE_STAGE_INVALID: expected validation_research or "
            "runtime_authoritative"
        )
    expected = SELECTIVE_EDGE_STAGE_SPLITS[stage]
    observed = str(split_spec or "")
    if observed != expected[0]:
        raise RuntimeError(
            "SELECTIVE_EDGE_STAGE_SPLIT_INVALID: "
            f"{stage} requires exactly --splits {expected[0]}; "
            f"observed={observed!r}"
        )
    return list(expected)


def _require_stage_split_bindings(
    args: argparse.Namespace,
    *,
    splits: list[str],
) -> dict[str, dict[str, str]]:
    """Require all selected bindings and reject every cross-stage binding."""

    if tuple(splits) not in tuple(SELECTIVE_EDGE_STAGE_SPLITS.values()):
        raise RuntimeError(
            "SELECTIVE_EDGE_STAGE_SPLIT_INVALID: expected one exact stage split"
        )
    suffixes = (
        "manifest_json",
        "manifest_sha256",
        "parquet",
        "parquet_sha256",
    )
    selected = set(splits)
    missing = [
        f"{split}_{suffix}"
        for split in splits
        for suffix in suffixes
        if not str(getattr(args, f"{split}_{suffix}", "") or "").strip()
    ]
    if missing:
        raise RuntimeError(
            "selected evaluation stage lacks explicit artifact bindings: "
            f"{missing}"
        )
    forbidden = [
        f"{split}_{suffix}"
        for split in EVALUATION_SPLITS
        if split not in selected
        for suffix in suffixes
        if str(getattr(args, f"{split}_{suffix}", "") or "").strip()
    ]
    if forbidden:
        raise RuntimeError(
            "selected evaluation stage includes forbidden cross-stage artifact "
            f"bindings: {forbidden}"
        )
    return {
        split: {
            "manifest_path": str(getattr(args, f"{split}_manifest_json")),
            "manifest_sha256": str(
                getattr(args, f"{split}_manifest_sha256")
            ),
            "parquet_path": str(getattr(args, f"{split}_parquet")),
            "parquet_sha256": str(
                getattr(args, f"{split}_parquet_sha256")
            ),
        }
        for split in splits
    }


def _load_selective_edge_stage_bundle(
    *,
    bundle_dir: Path,
    device: torch.device,
    evidence_stage: str,
) -> Any:
    """Strict-load and prove stage authority before any dataset is read."""

    require_entry_exit_production_architecture(
        current_entry_exit_architecture_observation(),
        context="SELECTIVE_EDGE_EVALUATOR_CONSTRUCTION",
    )

    bundle = load_entry_v10_ctx_bundle(
        bundle_dir=bundle_dir,
        device=str(device),
    )
    metadata = bundle.metadata
    if not isinstance(metadata, Mapping):
        raise RuntimeError("selective-edge bundle metadata is not an object")
    _require_evaluation_lineage(
        metadata.get("run_lineage"),
        evidence_stage=evidence_stage,
    )
    if any(key in metadata for key in ("direction_calibration", "path_calibration")):
        raise RuntimeError("SELECTIVE_EDGE_RETIRED_CALIBRATION_FORBIDDEN")
    require_entry_fitted_q_production_economics_readiness(
        metadata.get("entry_fitted_q_production_economics"),
        context="SELECTIVE_EDGE_CANDIDATE",
        require_ready=True,
    )
    return bundle


def _predict_bundle(
    *,
    bundle: Any,
    bundle_dir: Path,
    split_manifests: dict[str, Path],
    split_parquets: dict[str, Path],
    splits: list[str],
    model_name: str,
    device: torch.device,
    batch_size: int,
    m5_prebuilt_path: Path,
    evidence_stage: str,
    stream_chunk_rows: int,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    failures: list[str] = []
    model = bundle.transformer_model
    model.eval()
    meta = dict(bundle.metadata)
    run_lineage = meta.get("run_lineage")
    if model_name == EVALUATION_MODEL_NAME:
        _require_evaluation_lineage(
            run_lineage,
            evidence_stage=evidence_stage,
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
    if "hierarchical_entry_heads" in meta:
        raise RuntimeError("candidate bundle carries retired hierarchical Entry heads")
    if any(key in meta for key in ("direction_calibration", "path_calibration")):
        raise RuntimeError("candidate Entry-Q bundle carries retired calibration")
    runtime_head_ready = evidence_stage == "runtime_authoritative"
    ordered_signal_names = tuple(
        str(name) for name in meta.get("ordered_signal_names") or ()
    )
    if runtime_head_ready:
        if len(ordered_signal_names) != MODEL_NATIVE_SIGNAL_DIM or len(
            set(ordered_signal_names)
        ) != MODEL_NATIVE_SIGNAL_DIM:
            raise RuntimeError(
                "candidate bundle ordered_signal_names is not the exact unique "
                f"{MODEL_NATIVE_SIGNAL_DIM}-signal contract"
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
                    "edge_score": [],
                    "trade_side": [],
                    "pred_direction": [],
                    "ctx_cat": [],
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
                        pred_direction = decision["model_direction_index"]
                        chunks["edge_score"].append(decision["edge_score"])
                        chunks["trade_side"].append(pred_direction)
                        chunks["pred_direction"].append(pred_direction)
                        chunks["ctx_cat"].append(batch["ctx_cat"].detach().cpu().numpy().astype(np.int64))
                        extra_chunks.setdefault("entry_action_q_bps", []).append(
                            decision["entry_action_q_bps"]
                        )
                        extra_chunks.setdefault(
                            "entry_action_q_margin_bps", []
                        ).append(decision["entry_action_q_margin_bps"])
                        extra_chunks.setdefault("model_direction_index", []).append(
                            decision["model_direction_index"]
                        )
                        for _out_key, _width in (
                            ("entry_q_joint_hidden", UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM),
                            (UNIFIED_EXIT_ENTRY_REPRESENTATION_KEY, UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM),
                            ("side_mae_bps", 2),
                            ("trendline_event_logits", 4),
                        ):
                            extra_chunks.setdefault(_out_key, []).append(
                                _tensor_np(out, _out_key, width=_width)
                            )
                        # Persist genuine auxiliary diagnostics without
                        # composing another direction decision.
                        for _out_key, _col in _EXTRA_SIGMOID_HEADS.items():
                            _raw = _tensor_np(out, _out_key, width=1)
                            extra_chunks.setdefault(_col, []).append(
                                _sigmoid_np(_raw.reshape(-1))
                            )
                        extra_chunks.setdefault("position_size_logit", []).append(
                            _tensor_np(out, "position_size_logit", width=1).reshape(-1)
                        )
                        for _out_key, _width in _EXTRA_VECTOR_HEADS.items():
                            _vec = _tensor_np(out, _out_key, width=_width)
                            for _i in range(_width):
                                extra_chunks.setdefault(f"{_out_key}_{_i}", []).append(_vec[:, _i])
                        specialist_gate = _tensor_np(
                            out,
                            "specialist_gate",
                            width=len(MODEL_NATIVE_TRAINING_SPECIALISTS),
                        )
                        extra_chunks.setdefault("specialist_gate", []).append(
                            specialist_gate
                        )
                        extra_chunks.setdefault("tf_gate", []).append(
                            _tensor_np(
                                out,
                                "tf_gate",
                                width=ENTRY_MTF_CONTEXT_COUNT,
                            )
                        )
                        extra_chunks.setdefault(
                            "family_tf_cooperation_gate", []
                        ).append(
                            _tensor_np(
                                out,
                                "family_tf_cooperation_gate",
                                width=(
                                    len(MODEL_NATIVE_TRAINING_SPECIALISTS)
                                    * ENTRY_MTF_CONTEXT_COUNT
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
                            != (ENTRY_MTF_CONTEXT_COUNT, _expected_feature_width)
                            or not bool(torch.isfinite(_feature_gate).all().item())
                        ):
                            raise RuntimeError(
                                "exact model-native output family_tf_feature_gate "
                                f"invalid: observed={getattr(_feature_gate, 'shape', None)} "
                                f"expected=(*,{ENTRY_MTF_CONTEXT_COUNT},"
                                f"{_expected_feature_width})"
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

                arrays = {key: np.concatenate(value, axis=0) if value else np.zeros((0,), dtype=np.float32) for key, value in chunks.items()}
                n = int(len(arrays["pred_direction"]))
                extra_arrays = _concatenate_evidence_chunks(
                    extra_chunks,
                    expected_rows=n,
                )
                frame = dataset.df.iloc[dataset.indices].reset_index(drop=True).copy()
                frame = frame.iloc[:n].copy()
                ctx_cat = np.asarray(arrays["ctx_cat"], dtype=np.int64)
                frame["split"] = split
                frame["model"] = model_name
                frame["edge_score"] = arrays["edge_score"]
                frame["trade_side"] = arrays["trade_side"].astype(np.int64)
                frame["pred_direction"] = arrays["pred_direction"].astype(np.int64)
                # Auxiliary outputs are evidence columns only. Add them in one
                # block to avoid fragmented frames and needless memory churn.
                evidence_columns: dict[str, Any] = {}
                for _col, _arr in extra_arrays.items():
                    if _arr.ndim == 2 and _arr.shape[1] == 1:
                        evidence_columns[_col] = _arr[:, 0].astype(np.float32)
                    elif _arr.ndim == 2:
                        evidence_columns[_col] = [
                            row.astype(np.float32).tolist() for row in _arr
                        ]
                    elif (
                        _col.endswith("_side")
                        or _col.endswith("_index")
                    ):
                        evidence_columns[_col] = _arr.astype(np.int64)
                    else:
                        evidence_columns[_col] = _arr.astype(np.float32)
                overlapping_columns = frame.columns.intersection(evidence_columns)
                if len(overlapping_columns):
                    raise RuntimeError(
                        f"{model_name}/{split}: model evidence collides with "
                        f"dataset columns {overlapping_columns.tolist()}"
                    )
                frame = pd.concat(
                    [
                        frame,
                        pd.DataFrame(evidence_columns, index=frame.index),
                    ],
                    axis=1,
                )
                frame["selection_score_mode"] = MODEL_DIRECTION_SELECTION_MODE
                direction_indices = frame["pred_direction"].to_numpy(dtype=np.int64)
                if np.any((direction_indices < 0) | (direction_indices > 2)):
                    raise RuntimeError(
                        f"{model_name}/{split}: invalid model direction index"
                    )
                entry_q = np.stack(
                    frame["entry_action_q_bps"].map(
                        lambda value: np.asarray(value, dtype=np.float32)
                    )
                )
                frame["selection_score"] = entry_q[
                    np.arange(n, dtype=np.int64), direction_indices
                ].astype(np.float32)
                frame["model_direction"] = frame["pred_direction"].map(SIDE_NAMES)
                required_targets = {
                    "realized_net_long_pnl_bps",
                    "realized_net_short_pnl_bps",
                    *MODEL_NATIVE_TIMING_TARGET_COLUMNS,
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
                direction_ids = frame["pred_direction"].to_numpy(dtype=np.int64)
                if not set(direction_ids).issubset(SIDE_NAMES):
                    raise RuntimeError(
                        f"{model_name}/{split}: invalid direction ids "
                        f"{sorted(set(direction_ids) - set(SIDE_NAMES))}"
                    )
                frame["side"] = [SIDE_NAMES[int(x)] for x in direction_ids]
                frame["realized_net_policy_pnl_bps"] = _realized_net_policy_pnl(frame)
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
                    "pred_direction",
                    "trade_side",
                    "side",
                    "session",
                    "session_id",
                    "edge_score",
                    "entry_action_q_bps",
                    "entry_action_q_margin_bps",
                    "selection_score_mode",
                    "selection_score",
                    "realized_net_policy_pnl_bps",
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
                    "y_position_size_target",
                    *MODEL_NATIVE_TIMING_TARGET_COLUMNS,
                ):
                    if target_col not in frame.columns:
                        raise RuntimeError(
                            f"{model_name}/{split}: dataset lacks required smoke "
                            f"target evidence {target_col}"
                        )
                    keep_cols.append(target_col)
                if "y_forecast_ret_K24" in frame.columns:
                    keep_cols.append("y_forecast_ret_K24")
                keep_cols.extend(
                    ["realized_net_long_pnl_bps", "realized_net_short_pnl_bps"]
                )
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
    evidence_stage = str(args.evidence_stage)
    splits = _require_selective_edge_stage_split(
        evidence_stage=evidence_stage,
        split_spec=args.splits,
    )
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
    split_bindings = _require_stage_split_bindings(args, splits=splits)
    device = torch.device(_device_arg(args.device))
    _reject_retired_selection_environment()
    bundle = _load_selective_edge_stage_bundle(
        bundle_dir=bundle_dir,
        device=device,
        evidence_stage=evidence_stage,
    )
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
        bundle=bundle,
        bundle_dir=bundle_dir,
        split_manifests=split_manifests,
        split_parquets=split_parquets,
        splits=splits,
        model_name=EVALUATION_MODEL_NAME,
        device=device,
        batch_size=int(args.batch_size),
        m5_prebuilt_path=m5_prebuilt,
        evidence_stage=evidence_stage,
        stream_chunk_rows=int(args.stream_chunk_rows),
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
    # FORBIDDEN_LEGACY_BRIDGE_FIELDS names input features the retired IQL bridge
    # model consumed (self-referential leakage), not this writer's canonical
    # decision-evidence output columns. Entry authority is raw Q, so no
    # probability compatibility surface is emitted.
    forbidden_prediction_columns = sorted(
        {"anchor_logits", "delta_logits", "anchor_gate"}.intersection(
            predictions.columns
        )
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
            evidence_stage=evidence_stage,
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
            "evidence_stage": evidence_stage,
            "authoritative": bool(
                prediction_evidence.get("authoritative") is True
            ),
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
        required=True,
        help=(
            "Exactly one stage-owned split: use val with validation_research or "
            "test with runtime_authoritative."
        ),
    )
    ap.add_argument(
        "--evidence-stage",
        choices=EVIDENCE_STAGES,
        required=True,
        help=(
            "validation_research is VAL-only and non-authorizing; "
            "runtime_authoritative is TEST-only and requires a strict-loaded "
            "full candidate bundle with frozen fitted-Q lineage."
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
        required=True,
        help=(
            "Required bounded row-group chunk size in the range "
            f"1..{SELECTIVE_EDGE_MAX_STREAM_CHUNK_ROWS}; oversized source "
            "row groups fail closed before loading."
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
