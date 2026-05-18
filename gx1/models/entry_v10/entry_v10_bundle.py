"""
ENTRY_V10 / ENTRY_V10_CTX bundle loader (STRICT, manifest-only friendly).

- No dummy models.
- No fallback paths.
- No hardcoded directories.
- No network.
- CPU/GPU determined by caller.
- Hard-fails if bundle is incomplete.

This loader assumes:
bundle_dir contains:
    - MASTER_TRANSFORMER_LOCK.json
    - bundle_metadata.json
    - model_state_dict.pt
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set

import torch


@dataclass
class EntryV10Bundle:
    bundle_dir: str
    device: torch.device
    transformer_model: Any
    xgb_model_universal: Any
    xgb_models_by_session: Dict[str, Any]
    metadata: Dict[str, Any]
    transformer_config: Dict[str, Any]
    capabilities: Dict[str, Any]


def _guard_required(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"[ENTRY_V10_BUNDLE_MISSING] {label} not found: {path}")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_device(device: Optional[str]) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return torch.device(device)


_ENTRY_HEAD_STATE_KEYS: Dict[str, Set[str]] = {
    "direction": {"head_direction.weight", "head_direction.bias"},
    "path_quality": {"head_path_quality.weight", "head_path_quality.bias"},
    "mfe_first_n": {"head_mfe_first_n.weight", "head_mfe_first_n.bias"},
    "tradable": {"head_tradable.weight", "head_tradable.bias"},
    "bad_path": {"head_bad_path.weight", "head_bad_path.bias"},
    "clean_edge": {"head_clean_edge.weight", "head_clean_edge.bias"},
    "survival": {"head_survival.weight", "head_survival.bias"},
}


def _infer_entry_bundle_capabilities(meta: Dict[str, Any], state_dict: Dict[str, Any]) -> Dict[str, Any]:
    train_recipe = meta.get("train_recipe") if isinstance(meta.get("train_recipe"), dict) else {}
    declared_heads = {
        str(h).strip()
        for h in (train_recipe.get("active_heads") or [])
        if str(h).strip()
    }
    state_heads = {
        head_name
        for head_name, required_keys in _ENTRY_HEAD_STATE_KEYS.items()
        if required_keys.issubset(set(state_dict.keys()))
    }
    if declared_heads:
        missing_declared = declared_heads - state_heads
        if missing_declared:
            raise RuntimeError(
                f"[ENTRY_BUNDLE_CAPABILITY_MISMATCH] metadata declares unsupported heads: {sorted(missing_declared)}"
            )
        supported_heads = {"direction"} | declared_heads
    else:
        supported_heads = state_heads
    supported_heads.add("direction")
    unsupported_heads = sorted(set(_ENTRY_HEAD_STATE_KEYS) - supported_heads)
    return {
        "supported_heads": sorted(supported_heads),
        "unsupported_heads": unsupported_heads,
        "declared_active_heads": sorted(declared_heads),
        "state_dict_heads": sorted(state_heads),
        "supports_context_features": bool(meta.get("supports_context_features", False)),
    }


def load_entry_v10_ctx_bundle(
    *,
    bundle_dir: str | Path,
    feature_meta_path: Optional[str | Path] = None,
    seq_scaler_path: Optional[str | Path] = None,
    snap_scaler_path: Optional[str | Path] = None,
    device: Optional[str] = None,
    is_replay: bool = True,
    xgb_models: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> EntryV10Bundle:

    bd = Path(bundle_dir).expanduser().resolve()
    _guard_required(bd, "bundle_dir")

    lock_path = bd / "MASTER_TRANSFORMER_LOCK.json"
    meta_path = bd / "bundle_metadata.json"
    state_path = bd / "model_state_dict.pt"

    _guard_required(lock_path, "MASTER_TRANSFORMER_LOCK.json")
    _guard_required(meta_path, "bundle_metadata.json")
    _guard_required(state_path, "model_state_dict.pt")

    lock = _load_json(lock_path)
    meta = _load_json(meta_path)

    seq_input_dim = int(meta.get("seq_input_dim") or lock.get("seq_input_dim") or 0)
    snap_input_dim = int(meta.get("snap_input_dim") or lock.get("snap_input_dim") or 0)
    seq_len = int(meta.get("seq_len") or lock.get("seq_len") or 0)
    ctx_cont_dim = int(meta.get("ctx_cont_dim") or lock.get("ctx_cont_dim") or 0)
    ctx_cat_dim = int(meta.get("ctx_cat_dim") or lock.get("ctx_cat_dim") or 0)

    if seq_input_dim <= 0 or snap_input_dim <= 0 or seq_len <= 0:
        raise RuntimeError(
            f"[ENTRY_V10_CTX_INVALID_META] seq_input_dim={seq_input_dim} "
            f"snap_input_dim={snap_input_dim} seq_len={seq_len}"
        )
    if ctx_cont_dim <= 0 or ctx_cat_dim <= 0:
        raise RuntimeError(
            f"[ENTRY_V10_CTX_INVALID_META] ctx_cont_dim={ctx_cont_dim} ctx_cat_dim={ctx_cat_dim}"
        )

    dev = _resolve_device(device)
    logging.getLogger(__name__).info(
        "[ENTRY_BUNDLE_LOAD_PROOF] ctx_cont_dim=%d ctx_cat_dim=%d signal_dim=7",
        ctx_cont_dim,
        ctx_cat_dim,
    )

    # Import real model
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
        EntryV10CtxHybridTransformer,
    )

    model = EntryV10CtxHybridTransformer(
        seq_input_dim=seq_input_dim,
        snap_input_dim=snap_input_dim,
        seq_len=seq_len,
        ctx_cont_dim=ctx_cont_dim,
        ctx_cat_dim=ctx_cat_dim,
    ).to(dev)

    state_dict = torch.load(state_path, map_location=dev)
    parked_head_keys = {
        "head_early_move.weight",
        "head_early_move.bias",
        "head_quality.weight",
        "head_quality.bias",
        "head_clean_edge.weight",
        "head_clean_edge.bias",
        "head_survival.weight",
        "head_survival.bias",
    }
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = set(getattr(incompatible, "missing_keys", []) or [])
    unexpected = set(getattr(incompatible, "unexpected_keys", []) or [])
    unexpected_active = unexpected - parked_head_keys
    if unexpected_active:
        raise RuntimeError(f"[ENTRY_V10_CTX_UNEXPECTED_KEYS] {sorted(unexpected_active)}")
    allowed_missing = set()
    if missing and missing.issubset(parked_head_keys):
        allowed_missing = parked_head_keys
    if missing - allowed_missing:
        raise RuntimeError(f"[ENTRY_V10_CTX_MISSING_KEYS] {sorted(missing - allowed_missing)}")
    model.eval()
    capabilities = _infer_entry_bundle_capabilities(meta, state_dict)
    logging.getLogger(__name__).info(
        "[ENTRY_BUNDLE_CAPABILITIES] supported_heads=%s declared_active_heads=%s unsupported_heads=%s",
        capabilities["supported_heads"],
        capabilities["declared_active_heads"],
        capabilities["unsupported_heads"],
    )

    # Feature meta path is optional for CTX bundles (metadata carries contract)
    fmeta: Optional[Path] = None
    meta_feat = meta.get("feature_meta_path")
    if feature_meta_path is not None:
        fmeta = Path(feature_meta_path).expanduser().resolve()
        if not fmeta.is_absolute():
            fmeta = (bd / fmeta).resolve()
        _guard_required(fmeta, "feature_meta_path")
    elif meta_feat:
        fmeta = Path(meta_feat).expanduser()
        if not fmeta.is_absolute():
            fmeta = (bd / fmeta).resolve()
        else:
            fmeta = fmeta.resolve()
        _guard_required(fmeta, "feature_meta_path")
    if seq_scaler_path is None and meta.get("seq_scaler_path"):
        seq_scaler_path = meta.get("seq_scaler_path")
    if snap_scaler_path is None and meta.get("snap_scaler_path"):
        snap_scaler_path = meta.get("snap_scaler_path")

    if seq_scaler_path:
        seq_scaler_path = Path(seq_scaler_path).expanduser()
        if not seq_scaler_path.is_absolute():
            seq_scaler_path = (bd / seq_scaler_path).resolve()
        else:
            seq_scaler_path = seq_scaler_path.resolve()
        _guard_required(seq_scaler_path, "seq_scaler_path")
    if snap_scaler_path:
        snap_scaler_path = Path(snap_scaler_path).expanduser()
        if not snap_scaler_path.is_absolute():
            snap_scaler_path = (bd / snap_scaler_path).resolve()
        else:
            snap_scaler_path = snap_scaler_path.resolve()
        _guard_required(snap_scaler_path, "snap_scaler_path")

    bundle = EntryV10Bundle(
        bundle_dir=str(bd),
        device=dev,
        transformer_model=model,
        xgb_model_universal=xgb_models.get("UNIVERSAL") if xgb_models else None,
        xgb_models_by_session=xgb_models or {},
        metadata={
            **meta,
            "model_variant": "v10_ctx",
            "feature_meta_path": str(fmeta) if fmeta else None,
            "seq_scaler_path": str(seq_scaler_path) if seq_scaler_path else None,
            "snap_scaler_path": str(snap_scaler_path) if snap_scaler_path else None,
            "is_replay": bool(is_replay),
            "capabilities": capabilities,
        },
        transformer_config={
            "seq_input_dim": seq_input_dim,
            "snap_input_dim": snap_input_dim,
            "seq_len": seq_len,
        },
        capabilities=capabilities,
    )

    return bundle


def load_entry_v10_bundle(*_args: Any, **_kwargs: Any) -> EntryV10Bundle:
    """
    LEGACY_DISABLED: Non-CTX ENTRY_V10 is removed.
    """
    raise RuntimeError("LEGACY_DISABLED: ENTRY_V10 (non-ctx) is removed. Use load_entry_v10_ctx_bundle().")
