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
    "tf_agreement": {"head_tf_agreement.weight", "head_tf_agreement.bias"},
    "path_quality_log_var": {"head_path_quality_log_var.weight", "head_path_quality_log_var.bias"},
    "position_size": {"head_position_size.weight", "head_position_size.bias"},
    "hold_horizon": {"head_hold_horizon.weight", "head_hold_horizon.bias"},
    "mtf_direction": {"head_mtf_direction.weight", "head_mtf_direction.bias", "mtf_dir_scale"},
    "anchor_gate": {"head_anchor_gate.weight", "head_anchor_gate.bias"},
    "trendline_rail": {"head_trendline_rail.weight", "head_trendline_rail.bias"},
    "q_per_action": {"q_head.weight", "q_head.bias"},
    "trade_side_hierarchy": {
        "head_trade.weight",
        "head_trade.bias",
        "head_side.weight",
        "head_side.bias",
        "head_side_utility.weight",
        "head_side_utility.bias",
        "head_side_bad_path.weight",
        "head_side_bad_path.bias",
        "head_side_mae.weight",
        "head_side_mae.bias",
    },
    "side_validity": {"head_side_validity.weight", "head_side_validity.bias"},
    "dip": {"head_dip.weight", "head_dip.bias"},
    "forecast": {"head_forecast.weight", "head_forecast.bias"},
    "timing": {"head_timing.weight", "head_timing.bias"},
    "tail_risk": {"head_tail_risk.weight", "head_tail_risk.bias"},
    "vol_forecast": {"head_vol_forecast.weight", "head_vol_forecast.bias"},
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
        "[ENTRY_BUNDLE_LOAD_PROOF] ctx_cont_dim=%d ctx_cat_dim=%d seq_input_dim=%d snap_input_dim=%d",
        ctx_cont_dim,
        ctx_cat_dim,
        seq_input_dim,
        snap_input_dim,
    )

    # Import real model
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
        EntryV10CtxHybridTransformer,
    )

    # Detect multi-TF and v3+ aux heads from bundle metadata so model is
    # constructed with matching parameters (otherwise their weights show
    # up as unexpected_keys in strict-load below).
    mtf_meta = (meta.get("multi_tf") or {}) if isinstance(meta, dict) else {}
    enable_mtf = bool(mtf_meta.get("enabled", False))
    _m15_seq_dim = int(mtf_meta.get("m15_seq_dim", 0)) if enable_mtf else 0
    _h1_seq_dim = int(mtf_meta.get("h1_seq_dim", _m15_seq_dim)) if enable_mtf else 0
    _h4_seq_dim = int(mtf_meta.get("h4_seq_dim", _m15_seq_dim)) if enable_mtf else 0
    _d1_seq_dim = int(mtf_meta.get("d1_seq_dim", _m15_seq_dim)) if enable_mtf else 0
    _m15_seq_len = int(mtf_meta.get("m15_seq_len", 96)) if enable_mtf else 96
    _h1_seq_len = int(mtf_meta.get("h1_seq_len", _m15_seq_len)) if enable_mtf else _m15_seq_len
    _h4_seq_len = int(mtf_meta.get("h4_seq_len", _m15_seq_len)) if enable_mtf else _m15_seq_len
    _d1_seq_len = int(mtf_meta.get("d1_seq_len", _m15_seq_len)) if enable_mtf else _m15_seq_len
    mtf_scale = float(mtf_meta.get("multi_tf_scale", 0.5)) if enable_mtf else 0.5
    # V2 manifest fields (default 0/False = V1 bundles work unchanged).
    _v2_mode = bool(mtf_meta.get("v2_mode", False))
    _m5_seq_dim = int(mtf_meta.get("m5_seq_dim", 0)) if (enable_mtf and _v2_mode) else 0
    _m5_seq_len = int(mtf_meta.get("m5_seq_len", _m15_seq_len)) if (enable_mtf and _v2_mode) else _m15_seq_len
    # V10 v3+ aux heads: detect by presence in state_dict keys.
    state_dict_preview = torch.load(state_path, map_location="cpu")
    _has_tf_agreement = "head_tf_agreement.weight" in state_dict_preview
    _has_path_var = "head_path_quality_log_var.weight" in state_dict_preview
    _has_position_size = "head_position_size.weight" in state_dict_preview
    _has_hold_horizon = "head_hold_horizon.weight" in state_dict_preview
    # 2026-05-26 dip-aware heads + cross-TF attention — detect by state_dict
    # presence so strict-load matches (else they read as unexpected_keys).
    _has_dip = "head_dip.weight" in state_dict_preview
    _has_forecast = "head_forecast.weight" in state_dict_preview
    _has_timing = "head_timing.weight" in state_dict_preview
    _has_tail_risk = "head_tail_risk.weight" in state_dict_preview
    _has_vol_forecast = "head_vol_forecast.weight" in state_dict_preview
    _has_cross_tf = any(k.startswith("cross_tf_attn.") or k == "tf_gate_logits" for k in state_dict_preview)
    # Forceful MTF→direction head (2026-06-06): real params (head + scale) in
    # state_dict → detect by presence so the rebuilt model matches strict-load.
    _has_mtf_direction = "head_mtf_direction.weight" in state_dict_preview
    _has_anchor_gate = "head_anchor_gate.weight" in state_dict_preview
    _has_q_head = "q_head.weight" in state_dict_preview
    _has_hierarchical_entry_heads = "head_trade.weight" in state_dict_preview and "head_side.weight" in state_dict_preview
    _hierarchical_direction_cfg = (meta.get("hierarchical_direction_composition") or {}) if isinstance(meta, dict) else {}
    _enable_hierarchical_direction_composition = bool(_hierarchical_direction_cfg.get("enabled", False))
    _hierarchical_composition_residual_logit_cap = float(
        _hierarchical_direction_cfg.get(
            "residual_logit_cap",
            meta.get("hier_compose_residual_logit_cap", 0.0) if isinstance(meta, dict) else 0.0,
        )
    )
    _hierarchical_composition_residual_side_neutral = bool(
        _hierarchical_direction_cfg.get(
            "residual_side_neutral",
            meta.get("hier_compose_residual_side_neutral", False) if isinstance(meta, dict) else False,
        )
    )
    _hierarchical_composition_public_flat_from_trade = bool(
        _hierarchical_direction_cfg.get(
            "public_flat_from_trade",
            meta.get("hier_compose_public_flat_from_trade", False) if isinstance(meta, dict) else False,
        )
    )
    _has_side_validity_head = "head_side_validity.weight" in state_dict_preview
    _has_trendline_rail = "head_trendline_rail.weight" in state_dict_preview
    _trendline_rail_output_dim = 4
    if _has_trendline_rail:
        _trendline_weight = state_dict_preview.get("head_trendline_rail.weight")
        _shape = tuple(getattr(_trendline_weight, "shape", ()) or ())
        if len(_shape) >= 1 and int(_shape[0]) >= 4:
            _trendline_rail_output_dim = int(_shape[0])
    # Positional encoding uses a persistent=False buffer (NOT in state_dict),
    # so it cannot be probed from keys like the aux heads — it MUST be read
    # from bundle metadata. Default False keeps old (pos-enc-free) bundles
    # bit-identical at inference.
    _enable_pos_enc = bool(meta.get("enable_pos_enc", False)) if isinstance(meta, dict) else False
    # BIG-9 (2026-06-03): FiLM regime-conditioning — must rebuild with the module if the
    # bundle was trained with it (regime_film.* weights in state_dict), else strict-load fails.
    _enable_regime_film = bool(meta.get("enable_regime_film", False)) if isinstance(meta, dict) else False
    if not _enable_regime_film:
        _enable_regime_film = any(str(k).startswith("regime_film.") for k in state_dict_preview)

    # 2026-06-02: per-TF learnable input scaling (V10 v5+). Detect via
    # state_dict so bundles without this feature continue to load. Init
    # values come from meta so the Parameter shape matches state_dict
    # (state_dict load overwrites init with learned values).
    _tf_input_scale_cfg = (meta.get("tf_input_scale") or {}) if isinstance(meta, dict) else {}
    _has_tf_input_scale = any(
        f"tf_input_scale_{tf}" in state_dict_preview for tf in ("m5", "m15", "h1", "h4", "d1")
    )
    _tf_inits = (_tf_input_scale_cfg.get("init") or {})
    _tf_scale_kwargs = dict(
        enable_tf_input_scale=_has_tf_input_scale,
        tf_input_scale_init_m5=float(_tf_inits.get("m5", 1.0)),
        tf_input_scale_init_m15=float(_tf_inits.get("m15", 1.0)),
        tf_input_scale_init_h1=float(_tf_inits.get("h1", 0.7)),
        tf_input_scale_init_h4=float(_tf_inits.get("h4", 0.5)),
        tf_input_scale_init_d1=float(_tf_inits.get("d1", 0.3)),
    ) if _has_tf_input_scale else {}

    _specialist_cfg = (meta.get("specialist_fusion") or {}) if isinstance(meta, dict) else {}
    _has_specialist_fusion = bool(_specialist_cfg.get("enabled", False)) or any(
        str(k).startswith("specialist_proj.")
        or str(k).startswith("specialist_encoder.")
        or str(k).startswith("specialist_gate.")
        or str(k).startswith("specialist_out.")
        or str(k) == "specialist_fusion_scale"
        for k in state_dict_preview
    )
    _specialist_kwargs = {}
    if _has_specialist_fusion:
        _indices = _specialist_cfg.get("input_indices")
        if not isinstance(_indices, dict) or not _indices:
            raise RuntimeError("[ENTRY_BUNDLE_SPECIALIST_METADATA_MISSING] specialist_fusion.input_indices required")
        _specialist_kwargs = {
            "enable_specialist_fusion": True,
            "specialist_input_indices": {str(k): [int(i) for i in list(v or [])] for k, v in _indices.items()},
            "specialist_num_layers": int(_specialist_cfg.get("num_layers") or 1),
            "specialist_fusion_scale": float(_specialist_cfg.get("fusion_scale") or 0.25),
        }

    model = EntryV10CtxHybridTransformer(
        seq_input_dim=seq_input_dim,
        snap_input_dim=snap_input_dim,
        seq_len=seq_len,
        ctx_cont_dim=ctx_cont_dim,
        ctx_cat_dim=ctx_cat_dim,
        enable_anchor_gate=_has_anchor_gate,
        anchor_gate_init=float((meta.get("anchor_gate") or {}).get("init", 1.0)) if isinstance(meta, dict) else 1.0,
        enable_hierarchical_entry_heads=_has_hierarchical_entry_heads,
        enable_hierarchical_direction_composition=_enable_hierarchical_direction_composition,
        hierarchical_composition_residual_logit_cap=_hierarchical_composition_residual_logit_cap,
        hierarchical_composition_residual_side_neutral=_hierarchical_composition_residual_side_neutral,
        hierarchical_composition_public_flat_from_trade=_hierarchical_composition_public_flat_from_trade,
        enable_side_validity_head=_has_side_validity_head,
        enable_trendline_rail_head=_has_trendline_rail,
        trendline_rail_output_dim=_trendline_rail_output_dim,
        enable_multi_tf=enable_mtf,
        m15_seq_dim=_m15_seq_dim, h1_seq_dim=_h1_seq_dim, h4_seq_dim=_h4_seq_dim, d1_seq_dim=_d1_seq_dim,
        m15_seq_len=_m15_seq_len, h1_seq_len=_h1_seq_len,
        h4_seq_len=_h4_seq_len, d1_seq_len=_d1_seq_len,
        # V2: enable M5 branch when manifest says so.
        enable_multi_tf_m5=_v2_mode,
        m5_seq_dim=_m5_seq_dim, m5_seq_len=_m5_seq_len,
        multi_tf_scale=mtf_scale,
        enable_tf_agreement_head=_has_tf_agreement,
        enable_path_quality_variance_head=_has_path_var,
        enable_position_size_head=_has_position_size,
        enable_hold_horizon_head=_has_hold_horizon,
        enable_pos_enc=_enable_pos_enc,
        enable_regime_film=_enable_regime_film,
        enable_dip_head=_has_dip,
        enable_forecast_head=_has_forecast,
        enable_timing_head=_has_timing,
        enable_tail_risk_head=_has_tail_risk,
        enable_vol_forecast_head=_has_vol_forecast,
        enable_cross_tf_attn=_has_cross_tf,
        enable_mtf_direction_head=_has_mtf_direction,
        enable_q_head=_has_q_head,
        **_tf_scale_kwargs,
        **_specialist_kwargs,
    ).to(dev)

    state_dict = state_dict_preview
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
    allowed_missing = parked_head_keys | {"hierarchical_composition_residual_logit_cap"}
    if missing - allowed_missing:
        raise RuntimeError(f"[ENTRY_V10_CTX_MISSING_KEYS] {sorted(missing - allowed_missing)}")
    model.eval()
    # Post-hoc direction calibration (vedtak SMART_SEQ520_candidate_train_20260703):
    # installed from bundle metadata into the model's canonical forward, so the
    # bundle audit and live serve see identical calibrated logits by construction.
    # Bundles without the key (all cemented/live bundles) are bit-identical.
    _dir_cal = meta.get("direction_calibration") if isinstance(meta, dict) else None
    if isinstance(_dir_cal, dict) and bool(_dir_cal.get("enabled", True)):
        _cal_bias = torch.tensor([float(x) for x in (_dir_cal.get("bias") or [])], dtype=torch.float32)
        model.set_direction_calibration(float(_dir_cal.get("temperature", 1.0)), _cal_bias)
        logging.getLogger(__name__).info(
            "[ENTRY_DIRECTION_CAL] installed: temperature=%.4f bias=%s fitted_on=%s",
            float(_dir_cal.get("temperature", 1.0)),
            [round(float(x), 4) for x in (_dir_cal.get("bias") or [])],
            _dir_cal.get("fitted_on_split", "?"),
        )
    _path_cal = meta.get("path_calibration") if isinstance(meta, dict) else None
    if isinstance(_path_cal, dict) and bool(_path_cal.get("enabled", True)):
        model.set_path_calibration(
            float(_path_cal.get("path_quality_scale", 1.0)),
            float(_path_cal.get("path_quality_shift", 0.0)),
            float(_path_cal.get("bad_path_temperature", 1.0)),
            float(_path_cal.get("bad_path_bias", 0.0)),
        )
        logging.getLogger(__name__).info(
            "[ENTRY_PATH_CAL] installed: pq=%.4f*x%+.4f bad_path=x/%.4f%+.4f",
            float(_path_cal.get("path_quality_scale", 1.0)),
            float(_path_cal.get("path_quality_shift", 0.0)),
            float(_path_cal.get("bad_path_temperature", 1.0)),
            float(_path_cal.get("bad_path_bias", 0.0)),
        )
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
