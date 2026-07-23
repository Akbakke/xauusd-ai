"""Exact sole learned direction-evidence fusion contract."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


SCHEMA_VERSION = "entry_model_native_direction_evidence_fusion_v3"
FUSION_MODE = "sole_learned_acyclic_96x128x3"
CLASS_ORDER = ("LONG", "SHORT", "FLAT")
INPUT_DIM = 96
HIDDEN_DIM = 128
OUTPUT_DIM = 3
INPUTS = (
    ("model_native_logits", 3),
    ("mtf_dir_logits", 3),
    ("path_quality_raw", 1),
    ("path_quality_log_var", 1),
    ("mfe_first_n", 1),
    ("tradable_logit", 1),
    ("bad_path_logit_raw", 1),
    ("clean_edge_logit", 1),
    ("survival_logit", 1),
    ("trade_logit", 1),
    ("side_logits", 2),
    ("side_utility", 2),
    ("side_bad_path_logit", 2),
    ("side_mae", 2),
    ("side_validity_logit", 2),
    ("trendline_rail_logits", 6),
    ("tf_agreement_logit", 1),
    ("position_size_logit", 1),
    ("dip_pred", 18),
    ("forecast_pred", 4),
    ("timing_pred", 12),
    ("tail_risk_pred", 6),
    ("vol_forecast_pred", 3),
    ("action_value", 9),
    ("expectile_value", 3),
    ("action_advantage", 9),
)

PREDICTION_UNIT_CONTRACT = {
    "model_unit": "approximately_one_for_material_evidence",
    "train_target_bps_scale": 20.0,
    "bps_scaled_outputs": [
        "dip_pred",
        "forecast_pred",
        "tail_risk_pred",
        "vol_forecast_pred",
    ],
    "unit_interval_outputs": ["timing_pred"],
    "already_scaled_regression_outputs": [
        "path_quality_raw",
        "mfe_first_n",
        "side_utility",
        "side_mae",
        "action_value",
        "expectile_value",
        "action_advantage",
    ],
    "logit_or_dimensionless_outputs": [
        name
        for name, _width in INPUTS
        if name
        not in {
            "dip_pred",
            "forecast_pred",
            "tail_risk_pred",
            "vol_forecast_pred",
            "timing_pred",
            "path_quality_raw",
            "mfe_first_n",
            "side_utility",
            "side_mae",
            "action_value",
            "expectile_value",
            "action_advantage",
        }
    ],
    "raw_bps_outputs_forbidden": True,
}


def _ordered_input_layout() -> list[dict[str, Any]]:
    layout: list[dict[str, Any]] = []
    start = 0
    for name, width in INPUTS:
        stop = start + width
        layout.append(
            {"name": name, "width": width, "start": start, "stop": stop}
        )
        start = stop
    if start != INPUT_DIM:
        raise RuntimeError(f"direction evidence fusion width={start} expected={INPUT_DIM}")
    return layout


ORDERED_INPUT_LAYOUT = _ordered_input_layout()
_HASH_PAYLOAD = {
    "schema_version": SCHEMA_VERSION,
    "mode": FUSION_MODE,
    "class_order": list(CLASS_ORDER),
    "inputs": ORDERED_INPUT_LAYOUT,
    "prediction_unit_contract": PREDICTION_UNIT_CONTRACT,
}
INPUTS_SHA256 = hashlib.sha256(
    json.dumps(
        _HASH_PAYLOAD,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def direction_evidence_fusion_metadata() -> dict[str, Any]:
    return {
        **_HASH_PAYLOAD,
        "inputs_sha256": INPUTS_SHA256,
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": OUTPUT_DIM,
        "normalization": "LayerNorm",
        "activation": "GELU",
        "raw_pre_aux_calibration": True,
        # Directional sibling heads are valid causal evidence.  What is
        # forbidden is feedback from this block's own final fused output.
        "no_final_fused_direction_feedback": True,
        "directional_sibling_evidence_inputs": [
            "model_native_logits",
            "mtf_dir_logits",
            "side_logits",
            "side_utility",
            "side_bad_path_logit",
            "side_mae",
            "side_validity_logit",
            "action_value",
            "action_advantage",
        ],
        "derived_evidence_relations": {
            "action_advantage": "action_value - expectile_value_by_horizon"
        },
        "no_detach": True,
        "sole_direction_path": True,
        "additive_direction_overrides": False,
        "residual_direction_path": False,
        "manual_direction_cap": False,
        "raw_output": "raw_direction_logits",
        "calibrated_output": "direction_logits",
    }


def require_direction_evidence_fusion_metadata(
    value: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    expected = direction_evidence_fusion_metadata()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise RuntimeError(f"[{context}_DIRECTION_EVIDENCE_FUSION_INVALID]")
    return expected
