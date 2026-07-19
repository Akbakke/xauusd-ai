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
import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set

import torch

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
    require_model_native_signal_contract,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    require_training_objective_contract,
)
from gx1.contracts.entry_model_native_direction_evidence_fusion_v1 import (
    HIDDEN_DIM as EVIDENCE_FUSION_HIDDEN_DIM,
    INPUT_DIM as EVIDENCE_FUSION_INPUT_DIM,
    OUTPUT_DIM as EVIDENCE_FUSION_OUTPUT_DIM,
    require_direction_evidence_fusion_metadata,
)
from gx1.contracts.entry_model_native_learned_component_movement_v1 import (
    require_learned_component_movement_metadata,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
)
from gx1.models.entry_v10.direction_decision_contract import (
    require_model_direction_decision_contract,
)


@dataclass
class EntryV10Bundle:
    bundle_dir: str
    device: torch.device
    transformer_model: Any
    metadata: Dict[str, Any]
    transformer_config: Dict[str, Any]
    capabilities: Dict[str, Any]


def _guard_required(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"[ENTRY_V10_BUNDLE_MISSING] {label} not found: {path}")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    "mtf_direction": {"head_mtf_direction.weight", "head_mtf_direction.bias"},
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
    "model_native_evidence_fusion": {
        "evidence_fusion_norm.weight",
        "evidence_fusion_norm.bias",
        "evidence_fusion_in.weight",
        "evidence_fusion_in.bias",
        "evidence_fusion_out.weight",
        "evidence_fusion_out.bias",
    },
    "dip": {"head_dip.weight", "head_dip.bias"},
    "forecast": {"head_forecast.weight", "head_forecast.bias"},
    "timing": {"head_timing.weight", "head_timing.bias"},
    "tail_risk": {"head_tail_risk.weight", "head_tail_risk.bias"},
    "vol_forecast": {"head_vol_forecast.weight", "head_vol_forecast.bias"},
}

_MODEL_NATIVE_METADATA_ONLY_COMPONENTS: frozenset[str] = frozenset()

_MODEL_NATIVE_REQUIRED_ACTIVE_COMPONENTS = frozenset(
    {
        "direction",
        "tradable",
        "path_quality",
        "mfe_first_n",
        "bad_path",
        "clean_edge",
        "survival",
        "tf_agreement",
        "path_quality_log_var",
        "position_size",
        "mtf_direction",
        "dip",
        "forecast",
        "timing",
        "tail_risk",
        "vol_forecast",
        "trade_side_hierarchy",
        "model_native_evidence_fusion",
        "side_validity",
        "trendline_rail",
    }
)

_MODEL_NATIVE_REQUIRED_SPECIALISTS = (
    "structure_swing_encoder",
    "smc_liquidity_encoder",
    "trend_ema_encoder",
    "vol_compression_encoder",
    "momentum_flow_encoder",
    "session_regime_encoder",
    "chart_geometry_encoder",
    "price_action_candle_encoder",
)

_MODEL_NATIVE_AUX_TARGET_HORIZON_ITEMS = tuple(
    (f"y_dip_mae_{side}_K{horizon}", horizon)
    for side in ("long", "short")
    for horizon in (12, 48, 96)
) + tuple(
    (f"y_dip_mfe_{side}_K{horizon}", horizon)
    for side in ("long", "short")
    for horizon in (12, 48, 96)
) + tuple(
    (f"y_forecast_ret_K{horizon}", horizon)
    for horizon in (1, 5, 12, 24)
) + tuple(
    (f"y_dip_bottom_frac_{side}_K{horizon}", horizon)
    for side in ("long", "short")
    for horizon in (12, 48, 96)
) + tuple(
    (f"y_time_to_mfe_frac_{side}_K{horizon}", horizon)
    for side in ("long", "short")
    for horizon in (12, 48, 96)
) + tuple(
    (f"y_tail_mae_{side}_K{horizon}", horizon)
    for side in ("long", "short")
    for horizon in (12, 48, 96)
) + tuple(
    (f"y_vol_fwd_K{horizon}", horizon)
    for horizon in (12, 48, 96)
)
_MODEL_NATIVE_AUX_TARGET_COLUMNS = tuple(
    name for name, _ in _MODEL_NATIVE_AUX_TARGET_HORIZON_ITEMS
)
_MODEL_NATIVE_AUX_TARGET_HORIZONS = {
    name: int(horizon)
    for name, horizon in _MODEL_NATIVE_AUX_TARGET_HORIZON_ITEMS
}


def _require_mapping_field(parent: Mapping[str, Any], key: str, *, context: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"[ENTRY_BUNDLE_MODEL_NATIVE_METADATA_MISSING] {context}.{key}")
    return value


def _require_exact_model_native_bundle_metadata(
    meta: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> None:
    """Reject every metadata default that could alter a model-native forward.

    Lock and metadata must each carry the complete reconstruction contract;
    the loader never infers a missing value from the other file or a default.
    """

    shared_exact = (
        "contract_mode",
        "direction_logit_mode",
        "seq_input_dim",
        "snap_input_dim",
        "seq_len",
        "ctx_cont_dim",
        "ctx_cat_dim",
        "ordered_signal_names",
        "ordered_ctx_cont_names",
        "ordered_ctx_cat_names",
        "model_native_signal_contract",
        "aux_head_target_contract",
        "model_native_training_objective",
        "model_native_direction_evidence_fusion",
        "model_native_learned_component_movement",
    )
    missing_meta = [key for key in shared_exact if key not in meta]
    missing_lock = [key for key in shared_exact if key not in lock]
    if missing_meta or missing_lock:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_EXACT_METADATA_MISSING] "
            f"meta={missing_meta} lock={missing_lock}"
        )
    split_brain = [key for key in shared_exact if meta[key] != lock[key]]
    if split_brain:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_META_LOCK_SPLIT_BRAIN] "
            f"fields={split_brain}"
        )
    meta_training_objective = require_training_objective_contract(
        meta["model_native_training_objective"],
        context="ENTRY_BUNDLE_META",
    )
    lock_training_objective = require_training_objective_contract(
        lock["model_native_training_objective"],
        context="ENTRY_BUNDLE_LOCK",
    )
    if meta_training_objective != lock_training_objective:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_TRAINING_OBJECTIVE_SPLIT_BRAIN]"
        )
    require_direction_evidence_fusion_metadata(
        meta["model_native_direction_evidence_fusion"],
        context="ENTRY_BUNDLE_META",
    )
    require_direction_evidence_fusion_metadata(
        lock["model_native_direction_evidence_fusion"],
        context="ENTRY_BUNDLE_LOCK",
    )
    require_learned_component_movement_metadata(
        meta["model_native_learned_component_movement"],
        context="ENTRY_BUNDLE_META",
    )
    require_learned_component_movement_metadata(
        lock["model_native_learned_component_movement"],
        context="ENTRY_BUNDLE_LOCK",
    )
    aux_contract = meta["aux_head_target_contract"]
    if not isinstance(aux_contract, Mapping):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_AUX_TARGET_CONTRACT_MISSING]")
    if (
        aux_contract.get("schema_version") != "entry_model_native_aux_targets_v2"
        or aux_contract.get("columns") != list(_MODEL_NATIVE_AUX_TARGET_COLUMNS)
        or aux_contract.get("future_horizon_bars_by_column")
        != _MODEL_NATIVE_AUX_TARGET_HORIZONS
        or aux_contract.get("max_future_horizon_bars") != 96
        or aux_contract.get("spread_aware_risk_magnitudes_required") is not True
        or aux_contract.get("mid_price_timing_reference_only") is not True
        or aux_contract.get("incomplete_value") != "NaN_before_emission_only"
        or aux_contract.get("incomplete_rows_may_be_emitted") is not False
    ):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_AUX_TARGET_CONTRACT_INVALID]")
    if meta["contract_mode"] != MODEL_NATIVE_CONTRACT_MODE:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_MODE_MISSING]")
    if meta["direction_logit_mode"] != MODEL_NATIVE_DIRECTION_LOGIT_MODE:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_DIRECTION_MODE_INVALID]")
    if int(meta["seq_input_dim"]) != MODEL_NATIVE_SIGNAL_DIM or int(meta["snap_input_dim"]) != MODEL_NATIVE_SIGNAL_DIM:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_SIGNAL_DIM_INVALID] "
            f"seq={meta['seq_input_dim']} snap={meta['snap_input_dim']} expected={MODEL_NATIVE_SIGNAL_DIM}"
        )
    if int(meta["seq_len"]) != MODEL_NATIVE_SEQ_LEN:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_SEQ_LEN_INVALID] "
            f"got={meta['seq_len']!r} expected={MODEL_NATIVE_SEQ_LEN}"
        )
    if (
        int(meta["ctx_cont_dim"]) != MODEL_NATIVE_CTX_CONT_DIM
        or int(meta["ctx_cat_dim"]) != MODEL_NATIVE_CTX_CAT_DIM
    ):
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_CTX_DIM_INVALID] "
            f"continuous={meta['ctx_cont_dim']!r} categorical={meta['ctx_cat_dim']!r} "
            f"expected={MODEL_NATIVE_CTX_CONT_DIM}+{MODEL_NATIVE_CTX_CAT_DIM}"
        )
    if len(meta["ordered_ctx_cont_names"]) != int(meta["ctx_cont_dim"]):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_CTX_CONT_ORDER_INVALID]")
    if len(meta["ordered_ctx_cat_names"]) != int(meta["ctx_cat_dim"]):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_CTX_CAT_ORDER_INVALID]")
    if list(meta["ordered_ctx_cont_names"]) != list(MODEL_NATIVE_CTX_CONT_FIELDS):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_CTX_CONT_ORDER_INVALID]")
    if list(meta["ordered_ctx_cat_names"]) != list(MODEL_NATIVE_CTX_CAT_FIELDS):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_CTX_CAT_ORDER_INVALID]")
    meta_signal_contract = meta["model_native_signal_contract"]
    lock_signal_contract = lock["model_native_signal_contract"]
    require_model_native_signal_contract(meta_signal_contract, context="ENTRY_BUNDLE_META")
    require_model_native_signal_contract(lock_signal_contract, context="ENTRY_BUNDLE_LOCK")
    if list(meta["ordered_signal_names"]) != list(meta_signal_contract["fields"]):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_SIGNAL_ORDER_INVALID]")
    if meta.get("supports_context_features") is not True:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_CONTEXT_FEATURES_REQUIRED]")
    if meta.get("neutral_xgb_bridge") is not False or lock.get("neutral_xgb_bridge") is not False:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_NEUTRAL_BRIDGE_FLAG_INVALID]")
    if meta.get("xgb_bridge_source") is not None or lock.get("xgb_bridge_source") is not None:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_BRIDGE_SOURCE_INVALID]")
    if meta.get("anchored_entry_enabled") is not False or meta.get("anchor_source") is not None:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_ANCHOR_METADATA_FORBIDDEN]")
    anchor_gate = _require_mapping_field(meta, "anchor_gate", context="meta")
    if anchor_gate.get("enabled") is not False:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_ANCHOR_GATE_METADATA_FORBIDDEN]")

    require_model_direction_decision_contract(meta, context="ENTRY_BUNDLE_META")
    require_model_direction_decision_contract(lock, context="ENTRY_BUNDLE_LOCK")
    if meta["direction_decision_contract"] != lock["direction_decision_contract"]:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_DIRECTION_CONTRACT_SPLIT_BRAIN]")

    train_recipe = _require_mapping_field(meta, "train_recipe", context="meta")
    active_raw = train_recipe.get("active_heads")
    if not isinstance(active_raw, list) or any(not isinstance(value, str) for value in active_raw):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_ACTIVE_COMPONENTS_MISSING]")
    active = frozenset(active_raw)
    if len(active_raw) != len(active):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_ACTIVE_COMPONENTS_DUPLICATE]")
    if active != _MODEL_NATIVE_REQUIRED_ACTIVE_COMPONENTS:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_ACTIVE_COMPONENTS_MISMATCH] "
            f"missing={sorted(_MODEL_NATIVE_REQUIRED_ACTIVE_COMPONENTS - active)} "
            f"unexpected={sorted(active - _MODEL_NATIVE_REQUIRED_ACTIVE_COMPONENTS)}"
        )

    mtf = _require_mapping_field(meta, "multi_tf", context="meta")
    required_mtf = (
        "enabled",
        "v2_mode",
        "m5_seq_dim",
        "m5_seq_len",
        "m15_seq_dim",
        "m15_seq_len",
        "h1_seq_dim",
        "h1_seq_len",
        "h4_seq_dim",
        "h4_seq_len",
        "d1_seq_dim",
        "d1_seq_len",
        "multi_tf_scale",
        "closed_bar_target_availability",
        "target_availability_shift_minutes",
    )
    missing_mtf = [key for key in required_mtf if key not in mtf]
    if missing_mtf:
        raise RuntimeError(f"[ENTRY_BUNDLE_MODEL_NATIVE_MTF_METADATA_MISSING] {missing_mtf}")
    if mtf["enabled"] is not True or mtf["v2_mode"] is not True:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_MTF_V2_REQUIRED]")
    if not math.isfinite(float(mtf["multi_tf_scale"])) or float(mtf["multi_tf_scale"]) <= 0.0:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_MTF_SCALE_INVALID]")
    if mtf["closed_bar_target_availability"] is not True:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_MTF_CLOSED_BAR_REQUIRED]")
    if abs(float(mtf["target_availability_shift_minutes"]) - 5.0) > 1e-9:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_MTF_SHIFT_INVALID]")
    for key in (
        "m5_seq_dim",
        "m5_seq_len",
        "m15_seq_dim",
        "m15_seq_len",
        "h1_seq_dim",
        "h1_seq_len",
        "h4_seq_dim",
        "h4_seq_len",
        "d1_seq_dim",
        "d1_seq_len",
    ):
        if isinstance(mtf[key], bool) or int(mtf[key]) <= 0:
            raise RuntimeError(f"[ENTRY_BUNDLE_MODEL_NATIVE_MTF_VALUE_INVALID] {key}={mtf[key]!r}")

    for key in ("enable_pos_enc", "enable_regime_film"):
        if meta.get(key) is not True:
            raise RuntimeError(
                f"[ENTRY_BUNDLE_MODEL_NATIVE_FULL_STACK_COMPONENT_REQUIRED] meta.{key}"
            )
    tf_input_scale = _require_mapping_field(meta, "tf_input_scale", context="meta")
    if tf_input_scale.get("enabled") is not True:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_TF_INPUT_SCALE_REQUIRED]")
    tf_init = _require_mapping_field(tf_input_scale, "init", context="meta.tf_input_scale")
    if set(tf_init) != {"m5", "m15", "h1", "h4", "d1"}:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_TF_INPUT_SCALE_INIT_MISMATCH]")
    if any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in tf_init.values()
    ):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_TF_INPUT_SCALE_INVALID]")

    forbidden_direction_artifacts = (
        "hierarchical_direction_composition",
        "residual_scale",
    )
    stale = [
        f"{owner}.{key}"
        for owner, payload in (("meta", meta), ("lock", lock))
        for key in forbidden_direction_artifacts
        if key in payload
    ]
    if stale:
        raise RuntimeError(
            "[ENTRY_BUNDLE_STALE_DIRECTION_ARTIFACT_FORBIDDEN] " + ",".join(stale)
        )

    hierarchy = _require_mapping_field(meta, "hierarchical_entry_heads", context="meta")
    if hierarchy.get("enabled") is not True:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_HIERARCHY_REQUIRED]")
    trendline = _require_mapping_field(meta, "trendline_rail_head", context="meta")
    if trendline.get("enabled") is not True or int(trendline.get("output_dim", 0)) != 6:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_TRENDLINE_RAIL_CONTRACT_INVALID]")
    if trendline.get("hand_written_direction_pressure") is not False:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_TRENDLINE_DIRECTION_PRESSURE_FORBIDDEN]")
    if trendline.get("direction_mapping") != "direct_learned_evidence_fusion":
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_TRENDLINE_FUSION_MAPPING_INVALID]")
    state_contract = _require_mapping_field(meta, "model_native_state_contract", context="meta")
    if not state_contract:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_STATE_CONTRACT_MISSING]")
    specialist = _require_mapping_field(meta, "specialist_fusion", context="meta")
    if specialist.get("enabled") is not True:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_FUSION_REQUIRED]")
    if specialist.get("contract_mode") != MODEL_NATIVE_CONTRACT_MODE:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_MODE_INVALID]")
    input_indices = specialist.get("input_indices")
    if not isinstance(input_indices, Mapping) or not input_indices:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_INDICES_MISSING]")
    if list(input_indices) != list(_MODEL_NATIVE_REQUIRED_SPECIALISTS):
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_ORDER_MISMATCH] "
            f"observed={list(input_indices)} expected={list(_MODEL_NATIVE_REQUIRED_SPECIALISTS)}"
        )
    seen_specialist_indices: set[int] = set()
    for specialist_name in _MODEL_NATIVE_REQUIRED_SPECIALISTS:
        indices = input_indices[specialist_name]
        if not isinstance(indices, list) or not indices:
            raise RuntimeError(
                f"[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_INDICES_INVALID] {specialist_name}"
            )
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise RuntimeError(
                f"[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_INDEX_TYPE_INVALID] {specialist_name}"
            )
        if indices != sorted(set(indices)):
            raise RuntimeError(
                f"[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_INDEX_ORDER_INVALID] {specialist_name}"
            )
        if any(index < 0 or index >= MODEL_NATIVE_SIGNAL_DIM for index in indices):
            raise RuntimeError(
                f"[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_INDEX_RANGE_INVALID] {specialist_name}"
            )
        overlap = seen_specialist_indices.intersection(indices)
        if overlap:
            raise RuntimeError(
                "[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_INDEX_OVERLAP] "
                f"{specialist_name} overlap={sorted(overlap)}"
            )
        seen_specialist_indices.update(indices)
    if list(specialist.get("trainable_specialists") or []) != list(
        _MODEL_NATIVE_REQUIRED_SPECIALISTS
    ):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_TRAINABLE_SPECIALISTS_MISMATCH]")
    for key in ("num_layers", "fusion_scale"):
        if key not in specialist:
            raise RuntimeError(f"[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_METADATA_MISSING] {key}")
    if isinstance(specialist["num_layers"], bool) or int(specialist["num_layers"]) <= 0:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_NUM_LAYERS_INVALID]")
    if float(specialist["fusion_scale"]) <= 0.0:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_FUSION_SCALE_INVALID]")

    direction_calibration = meta.get("direction_calibration")
    if direction_calibration is not None:
        if not isinstance(direction_calibration, Mapping):
            raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_DIRECTION_CALIBRATION_INVALID]")
        missing = [
            key
            for key in ("enabled", "version", "temperature", "bias")
            if key not in direction_calibration
        ]
        if missing or direction_calibration["enabled"] is not True:
            raise RuntimeError(
                "[ENTRY_BUNDLE_MODEL_NATIVE_DIRECTION_CALIBRATION_INCOMPLETE] "
                f"missing={missing}"
            )
        bias = direction_calibration["bias"]
        if not isinstance(bias, list) or len(bias) != 3:
            raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_DIRECTION_CALIBRATION_BIAS_INVALID]")
        values = [float(direction_calibration["temperature"]), *(float(value) for value in bias)]
        if not bool(torch.isfinite(torch.tensor(values, dtype=torch.float64)).all().item()) or values[0] <= 0.0:
            raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_DIRECTION_CALIBRATION_NONFINITE]")
    path_calibration = meta.get("path_calibration")
    if path_calibration is not None:
        if not isinstance(path_calibration, Mapping):
            raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_PATH_CALIBRATION_INVALID]")
        required_path_cal = (
            "enabled",
            "version",
            "path_quality_scale",
            "path_quality_shift",
            "bad_path_temperature",
            "bad_path_bias",
        )
        missing = [key for key in required_path_cal if key not in path_calibration]
        if missing or path_calibration["enabled"] is not True:
            raise RuntimeError(
                "[ENTRY_BUNDLE_MODEL_NATIVE_PATH_CALIBRATION_INCOMPLETE] "
                f"missing={missing}"
            )
        values = [float(path_calibration[key]) for key in required_path_cal[2:]]
        if not bool(torch.isfinite(torch.tensor(values, dtype=torch.float64)).all().item()):
            raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_PATH_CALIBRATION_NONFINITE]")
        if values[0] <= 0.0 or values[2] <= 0.0:
            raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_PATH_CALIBRATION_SCALE_INVALID]")


def _require_model_native_state_head_contract(
    meta: Mapping[str, Any],
    state_dict: Mapping[str, Any],
) -> None:
    active = frozenset(meta["train_recipe"]["active_heads"])
    expected_state_heads = active - _MODEL_NATIVE_METADATA_ONLY_COMPONENTS
    state_heads = {
        name
        for name, keys in _ENTRY_HEAD_STATE_KEYS.items()
        if keys.issubset(set(state_dict))
    }
    if state_heads != expected_state_heads:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_STATE_HEAD_CONTRACT_MISMATCH] "
            f"missing={sorted(expected_state_heads - state_heads)} "
            f"unexpected={sorted(state_heads - expected_state_heads)}"
        )


_MODEL_NATIVE_ZERO_INIT_COMPONENT_GROUPS: Dict[str, tuple[str, ...]] = {
    # These blocks are deliberately zero-initialized.  Merely finding their
    # keys in a state_dict therefore does not prove they ever joined the
    # learned decision path; at least one value must have moved during train.
    "specialist_fusion_output": (
        "specialist_out.weight",
        "specialist_out.bias",
    ),
    "regime_film": (
        "regime_film.2.weight",
        "regime_film.2.bias",
    ),
    "cross_tf_output": (
        "cross_tf_out.weight",
        "cross_tf_out.bias",
    ),
    "cross_tf_gate": ("tf_gate_logits",),
}

_RETIRED_DIRECTION_STATE_PREFIXES = (
    "head_public_trade.",
    "head_public_flat.",
    "head_public_side.",
    "hierarchical_ctx_prior_adapter.",
    "hierarchical_ctx_direction_calibration.",
)
_RETIRED_DIRECTION_STATE_KEYS = frozenset({"mtf_dir_scale"})


def _require_evidence_fusion_state_contract(state_dict: Mapping[str, Any]) -> None:
    expected_shapes = {
        "evidence_fusion_norm.weight": (EVIDENCE_FUSION_INPUT_DIM,),
        "evidence_fusion_norm.bias": (EVIDENCE_FUSION_INPUT_DIM,),
        "evidence_fusion_in.weight": (
            EVIDENCE_FUSION_HIDDEN_DIM,
            EVIDENCE_FUSION_INPUT_DIM,
        ),
        "evidence_fusion_in.bias": (EVIDENCE_FUSION_HIDDEN_DIM,),
        "evidence_fusion_out.weight": (
            EVIDENCE_FUSION_OUTPUT_DIM,
            EVIDENCE_FUSION_HIDDEN_DIM,
        ),
        "evidence_fusion_out.bias": (EVIDENCE_FUSION_OUTPUT_DIM,),
    }
    failures: list[str] = []
    for key, expected_shape in expected_shapes.items():
        value = state_dict.get(key)
        if not isinstance(value, torch.Tensor):
            failures.append(f"{key}:missing_or_non_tensor")
            continue
        if tuple(value.shape) != expected_shape:
            failures.append(
                f"{key}:shape={tuple(value.shape)} expected={expected_shape}"
            )
            continue
        if not bool(torch.isfinite(value).all().item()):
            failures.append(f"{key}:non_finite")
    for key in (
        "evidence_fusion_norm.weight",
        "evidence_fusion_in.weight",
        "evidence_fusion_out.weight",
    ):
        value = state_dict.get(key)
        if isinstance(value, torch.Tensor) and not bool(torch.count_nonzero(value).item()):
            failures.append(f"{key}:all_zero")
    out_weight = state_dict.get("evidence_fusion_out.weight")
    if isinstance(out_weight, torch.Tensor) and tuple(out_weight.shape) == (
        EVIDENCE_FUSION_OUTPUT_DIM,
        EVIDENCE_FUSION_HIDDEN_DIM,
    ):
        if any(
            bool(torch.equal(out_weight[i], out_weight[j]))
            for i in range(EVIDENCE_FUSION_OUTPUT_DIM)
            for j in range(i + 1, EVIDENCE_FUSION_OUTPUT_DIM)
        ):
            failures.append("evidence_fusion_out.weight:identical_class_rows")
    stale = sorted(
        key
        for key in state_dict
        if key in _RETIRED_DIRECTION_STATE_KEYS
        or key.startswith(_RETIRED_DIRECTION_STATE_PREFIXES)
    )
    if stale:
        failures.append(f"retired_direction_state={stale}")
    if failures:
        raise RuntimeError(
            "[ENTRY_BUNDLE_DIRECTION_EVIDENCE_FUSION_STATE_INVALID] "
            + " | ".join(failures)
        )


def _require_model_native_learned_component_liveness(
    state_dict: Mapping[str, Any],
) -> None:
    """Reject structurally present full-stack blocks that remained pass-throughs."""

    failures: list[str] = []
    for component, keys in _MODEL_NATIVE_ZERO_INIT_COMPONENT_GROUPS.items():
        missing = [key for key in keys if key not in state_dict]
        if missing:
            failures.append(f"{component}:missing={missing}")
            continue
        tensors = [state_dict[key] for key in keys]
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            failures.append(f"{component}:non_tensor_state")
            continue
        if any(not bool(torch.isfinite(value).all().item()) for value in tensors):
            failures.append(f"{component}:non_finite_state")
            continue
        if not any(bool(torch.count_nonzero(value).item()) for value in tensors):
            failures.append(f"{component}:zero_init_pass_through")
    _require_evidence_fusion_state_contract(state_dict)
    if failures:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_LEARNED_COMPONENT_LIVENESS_INVALID] "
            + " | ".join(failures)
        )


def _infer_entry_bundle_capabilities(meta: Dict[str, Any], state_dict: Dict[str, Any]) -> Dict[str, Any]:
    declared_heads = {str(h) for h in meta["train_recipe"]["active_heads"]}
    state_heads = {
        head_name
        for head_name, required_keys in _ENTRY_HEAD_STATE_KEYS.items()
        if required_keys.issubset(set(state_dict.keys()))
    }
    missing_declared = declared_heads - state_heads - _MODEL_NATIVE_METADATA_ONLY_COMPONENTS
    if missing_declared:
        raise RuntimeError(
            f"[ENTRY_BUNDLE_CAPABILITY_MISMATCH] metadata declares unsupported heads: {sorted(missing_declared)}"
        )
    supported_heads = {"direction"} | declared_heads
    unsupported_heads = sorted(set(_ENTRY_HEAD_STATE_KEYS) - supported_heads)
    return {
        "supported_heads": sorted(supported_heads),
        "unsupported_heads": unsupported_heads,
        "declared_active_heads": sorted(declared_heads),
        "state_dict_heads": sorted(state_heads),
        "supports_context_features": True,
    }


def load_entry_v10_ctx_bundle(
    *,
    bundle_dir: str | Path,
    feature_meta_path: Optional[str | Path] = None,
    seq_scaler_path: Optional[str | Path] = None,
    snap_scaler_path: Optional[str | Path] = None,
    device: Optional[str] = None,
    is_replay: bool = True,
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
    _require_exact_model_native_bundle_metadata(meta, lock)
    seq_input_dim = int(meta["seq_input_dim"])
    snap_input_dim = int(meta["snap_input_dim"])
    seq_len = int(meta["seq_len"])
    ctx_cont_dim = int(meta["ctx_cont_dim"])
    ctx_cat_dim = int(meta["ctx_cat_dim"])
    if lock.get("model_path_relative") != state_path.name:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_STATE_PATH_BINDING_INVALID]")
    observed_state_sha256 = _sha256_file(state_path)
    lock_state_sha256 = lock.get("model_sha256")
    meta_state_sha256 = meta.get("state_dict_sha256")
    if lock_state_sha256 != observed_state_sha256 or meta_state_sha256 != observed_state_sha256:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_STATE_SHA256_MISMATCH] "
            f"observed={observed_state_sha256} lock={lock_state_sha256!r} "
            f"meta={meta_state_sha256!r}"
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

    # Reconstruct the exact multi-TF contract declared by the validated bundle.
    mtf_meta = meta["multi_tf"]
    _m15_seq_dim = int(mtf_meta["m15_seq_dim"])
    _h1_seq_dim = int(mtf_meta["h1_seq_dim"])
    _h4_seq_dim = int(mtf_meta["h4_seq_dim"])
    _d1_seq_dim = int(mtf_meta["d1_seq_dim"])
    _m15_seq_len = int(mtf_meta["m15_seq_len"])
    _h1_seq_len = int(mtf_meta["h1_seq_len"])
    _h4_seq_len = int(mtf_meta["h4_seq_len"])
    _d1_seq_len = int(mtf_meta["d1_seq_len"])
    mtf_scale = float(mtf_meta["multi_tf_scale"])
    _m5_seq_dim = int(mtf_meta["m5_seq_dim"])
    _m5_seq_len = int(mtf_meta["m5_seq_len"])
    state_dict_preview = torch.load(state_path, map_location="cpu")
    _require_model_native_state_head_contract(meta, state_dict_preview)
    _require_model_native_learned_component_liveness(state_dict_preview)
    _tf_inits = meta["tf_input_scale"]["init"]
    required_tf_scale_keys = {f"tf_input_scale_{tf}" for tf in ("m5", "m15", "h1", "h4", "d1")}
    missing_tf_scale_keys = sorted(required_tf_scale_keys - set(state_dict_preview))
    if missing_tf_scale_keys:
        raise RuntimeError(f"[ENTRY_BUNDLE_MODEL_NATIVE_TF_INPUT_SCALE_STATE_MISSING] {missing_tf_scale_keys}")

    _specialist_cfg = meta["specialist_fusion"]
    _indices = _specialist_cfg["input_indices"]
    model = EntryV10CtxHybridTransformer(
        seq_input_dim=seq_input_dim,
        snap_input_dim=snap_input_dim,
        seq_len=seq_len,
        ctx_cont_dim=ctx_cont_dim,
        ctx_cat_dim=ctx_cat_dim,
        m15_seq_dim=_m15_seq_dim, h1_seq_dim=_h1_seq_dim, h4_seq_dim=_h4_seq_dim, d1_seq_dim=_d1_seq_dim,
        m15_seq_len=_m15_seq_len, h1_seq_len=_h1_seq_len,
        h4_seq_len=_h4_seq_len, d1_seq_len=_d1_seq_len,
        m5_seq_dim=_m5_seq_dim, m5_seq_len=_m5_seq_len,
        multi_tf_scale=mtf_scale,
        tf_input_scale_init_m5=float(_tf_inits["m5"]),
        tf_input_scale_init_m15=float(_tf_inits["m15"]),
        tf_input_scale_init_h1=float(_tf_inits["h1"]),
        tf_input_scale_init_h4=float(_tf_inits["h4"]),
        tf_input_scale_init_d1=float(_tf_inits["d1"]),
        specialist_input_indices={
            str(k): list(v) for k, v in _indices.items()
        },
        specialist_num_layers=int(_specialist_cfg["num_layers"]),
        specialist_fusion_scale=float(_specialist_cfg["fusion_scale"]),
    ).to(dev)

    state_dict = state_dict_preview
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    # Optional immutable calibrations are part of the canonical forward and
    # were fully validated above before any model is returned.
    _dir_cal = meta.get("direction_calibration")
    if _dir_cal is not None:
        _cal_bias = torch.tensor([float(x) for x in _dir_cal["bias"]], dtype=torch.float32)
        _cal_temperature = float(_dir_cal["temperature"])
        _cal_fitted_on = str(_dir_cal.get("fitted_on_split", "unspecified"))
        model.set_direction_calibration(_cal_temperature, _cal_bias)
        logging.getLogger(__name__).info(
            "[ENTRY_DIRECTION_CAL] installed: temperature=%.4f bias=%s fitted_on=%s",
            _cal_temperature,
            [round(float(x), 4) for x in _cal_bias.tolist()],
            _cal_fitted_on,
        )
    _path_cal = meta.get("path_calibration")
    if _path_cal is not None:
        _path_values = (
            float(_path_cal["path_quality_scale"]),
            float(_path_cal["path_quality_shift"]),
            float(_path_cal["bad_path_temperature"]),
            float(_path_cal["bad_path_bias"]),
        )
        model.set_path_calibration(
            *_path_values,
        )
        logging.getLogger(__name__).info(
            "[ENTRY_PATH_CAL] installed: pq=%.4f*x%+.4f bad_path=x/%.4f%+.4f",
            *_path_values,
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
