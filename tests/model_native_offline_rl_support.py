"""One exact Q/V/advantage fixture for model-native runtime tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_EXTRA_ACTIVE_TARGET_HEADS,
    model_native_aux_target_contract_metadata,
)
from gx1.contracts.entry_model_native_calibration_v1 import (
    DIRECTION_CALIBRATION_TIE_POLICY,
    DIRECTION_CALIBRATION_TRANSFORM,
    DIRECTION_CALIBRATION_VERSION,
)
from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_MTF_CONTEXT_COUNT
from gx1.contracts.entry_model_native_offline_rl_v1 import (
    ACTION_ORDER,
    ACTION_VALUE_TARGET_COLUMNS,
    HORIZON_BARS,
    REWARD_SCALE_BPS,
    offline_rl_contract_metadata,
)
from gx1.contracts.entry_model_native_readiness_v1 import (
    MODEL_NATIVE_BASE_ACTIVE_HEADS,
    MODEL_NATIVE_BLOCKED_HEADS,
)
from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
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
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4


TEST_DIRECTION_CALIBRATION = {
    "enabled": True,
    "version": DIRECTION_CALIBRATION_VERSION,
    "temperature": 1.0,
    "class_order": ["LONG", "SHORT", "FLAT"],
    "transform": DIRECTION_CALIBRATION_TRANSFORM,
    "argmax_preserving": True,
    "tie_policy": DIRECTION_CALIBRATION_TIE_POLICY,
    "fitted_at_utc": "2026-07-17T09:00:00+00:00",
    "fitted_on_split": "val",
    "fitted_rows": 300,
    "model": "candidate",
    "min_fit_rows": 100,
    "run_id": "UNIT_RUNTIME_DIRECTION_CALIBRATION",
    "source_bundle_dir": "/tmp/entry_model_native_bundle_20260717T080000123456Z",
    "source_bundle_metadata_sha256": "1" * 64,
    "predictions_path": "/tmp/selective_edge_predictions_20260717T083000123456Z.parquet",
    "predictions_sha256": "2" * 64,
    "prediction_report_path": "/tmp/ENTRY_CANDIDATE_SELECTIVE_EDGE_20260717T083000123456Z.json",
    "prediction_report_sha256": "3" * 64,
}
TEST_PATH_CALIBRATION = {
    "enabled": True,
    "version": "entry_model_native_path_calibration_v1",
    "path_quality_scale": 1.0,
    "path_quality_shift": 0.0,
    "bad_path_temperature": 1.0,
    "bad_path_bias": 0.0,
    "fitted_at_utc": "2026-07-17T09:30:00+00:00",
    "fitted_on_split": "val",
    "fitted_rows": 300,
    "model": "candidate",
    "min_fit_rows": 100,
    "run_id": "UNIT_RUNTIME_PATH_CALIBRATION",
    "source_bundle_dir": "/tmp/entry_model_native_bundle_20260717T090000123456Z",
    "source_bundle_metadata_sha256": "4" * 64,
    "predictions_path": "/tmp/selective_edge_predictions_20260717T091500123456Z.parquet",
    "predictions_sha256": "5" * 64,
    "prediction_report_path": "/tmp/ENTRY_CANDIDATE_SELECTIVE_EDGE_20260717T091500123456Z.json",
    "prediction_report_sha256": "6" * 64,
}


def offline_rl_evidence() -> dict[str, list[float]]:
    action_value = [0.4, 0.5, 0.6, -0.2, -0.1, 0.0, 0.0, 0.0, 0.0]
    expectile_value = [0.1, 0.2, 0.3]
    return {
        "action_value": action_value,
        "expectile_value": expectile_value,
        "action_advantage": [
            value - expectile_value[index % 3]
            for index, value in enumerate(action_value)
        ],
    }


def model_native_mtf_cooperation_evidence() -> dict[str, list[float]]:
    """Return the exact neutral-shape fixture for learned MTF diagnostics."""

    timeframe_count = ENTRY_MTF_CONTEXT_COUNT
    cooperation_width = timeframe_count * len(MODEL_NATIVE_TRAINING_SPECIALISTS)
    feature_width = timeframe_count * len(MULTI_TF_PER_BAR_FEATURES_V4)
    return {
        "tf_gate": [1.0 / timeframe_count] * timeframe_count,
        "family_tf_cooperation_gate": [
            1.0 / cooperation_width
        ] * cooperation_width,
        "family_tf_feature_gate": [1.0] * feature_width,
    }


def model_native_target_audit_evidence() -> dict[str, object]:
    return {
        "model_native_aux_target_contract": (
            model_native_aux_target_contract_metadata()
        ),
        "offline_rl_target_contract": {
            "decision": "PASS",
            "failures": [],
            "offline_rl_contract": offline_rl_contract_metadata(),
            "action_value_target_columns": list(ACTION_VALUE_TARGET_COLUMNS),
        },
        "target_head_contract": {
            "active_training_heads": list(MODEL_NATIVE_BASE_ACTIVE_HEADS),
            "blocked_heads": list(MODEL_NATIVE_BLOCKED_HEADS),
            "extra_active_target_heads": list(
                MODEL_NATIVE_EXTRA_ACTIVE_TARGET_HEADS
            ),
            "extra_active_target_head_liveness": {
                head: True for head in MODEL_NATIVE_EXTRA_ACTIVE_TARGET_HEADS
            },
        },
    }


def offline_rl_prediction_columns(rows: int) -> dict[str, object]:
    rows = int(rows)
    ordinal = np.arange(rows, dtype=np.float64)
    base = np.mod(ordinal, 97.0) / 96.0
    rewards = np.zeros((rows, len(ACTION_ORDER), len(HORIZON_BARS)))
    columns: dict[str, object] = {}
    for horizon_index, horizon in enumerate(HORIZON_BARS):
        long_reward = 50.0 + 5.0 * base + horizon_index
        short_reward = -25.0 + 5.0 * base - horizon_index
        rewards[:, 0, horizon_index] = long_reward
        rewards[:, 1, horizon_index] = short_reward
        columns[f"y_action_value_long_K{horizon}"] = long_reward
        columns[f"y_action_value_short_K{horizon}"] = short_reward
        columns[f"y_action_value_flat_K{horizon}"] = np.zeros(rows)
    q_values = rewards / float(REWARD_SCALE_BPS)
    value = q_values.max(axis=1) - 0.05
    advantage = q_values - value[:, None, :]
    columns["action_value"] = [row.tolist() for row in q_values.reshape(rows, -1)]
    columns["expectile_value"] = [row.tolist() for row in value]
    columns["action_advantage"] = [
        row.tolist() for row in advantage.reshape(rows, -1)
    ]
    return columns


def offline_rl_prediction_row(index: int = 0) -> dict[str, object]:
    columns = offline_rl_prediction_columns(max(1, int(index) + 1))
    return {name: values[int(index)] for name, values in columns.items()}


def add_test_runtime_calibration_metadata(metadata: dict[str, object]) -> None:
    metadata["direction_calibration"] = dict(TEST_DIRECTION_CALIBRATION)
    metadata["path_calibration"] = dict(TEST_PATH_CALIBRATION)


def runtime_head_prediction_columns(
    frame: pd.DataFrame,
    bundle_metadata: dict[str, object],
) -> dict[str, object]:
    """Build exact immutable runtime-head envelopes for prediction fixtures."""

    direction_calibration = bundle_metadata["direction_calibration"]
    path_calibration = bundle_metadata["path_calibration"]
    payloads: list[str] = []
    hashes: list[str] = []
    for ordinal, row in frame.reset_index(drop=True).iterrows():
        logits = np.asarray(row["direction_logits"], dtype=np.float64)
        shifted = logits - logits.max()
        probs = np.exp(shifted)
        probs /= probs.sum()
        direction_index = int(np.argmax(logits))
        public_logits = np.asarray(
            [max(logits[0], logits[1]), logits[2]],
            dtype=np.float64,
        )
        public_probs = np.exp(public_logits - public_logits.max())
        public_probs /= public_probs.sum()
        side_logits = [0.4, -0.2]
        side_probs = np.exp(np.asarray(side_logits) - max(side_logits))
        side_probs /= side_probs.sum()
        path_log_var = math.log(0.25)
        tf_logit = 0.2
        size_logit = float(row.get("position_size_logit", -0.1))
        side_bad = [-0.5, 0.5]
        side_validity = [0.6, -0.4]
        mtf_logits = [0.3, -0.1, -0.2]
        mtf_probs = np.exp(np.asarray(mtf_logits) - max(mtf_logits))
        mtf_probs /= mtf_probs.sum()
        rail_logits = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]
        action_value = list(row["action_value"])
        expectile_value = list(row["expectile_value"])
        action_advantage = list(row["action_advantage"])
        session_id = int(row.get("session_id", ordinal % 4))
        session_names = ("ASIA", "EU", "OVERLAP", "US")
        session = session_names[session_id]
        vol_id = int(row.get("vol_regime_id", 2))
        vol_names = MODEL_NATIVE_ENTRY_VOL_REGIME_NAMES
        trend_id = int(row.get("trend_regime_id", 1))
        trend_names = ("TREND_DOWN", "TREND_NEUTRAL", "TREND_UP")
        evidence: dict[str, object] = {
            "runtime_head_evidence_schema_version": (
                MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION
            ),
            "runtime_evidence_schema_version": (
                MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION
            ),
            "model_policy": MODEL_NATIVE_RUNTIME_POLICY,
            "decision_ts": str(pd.Timestamp(row["time"])),
            "session_id": session_id,
            "session": session,
            "entry_vol_regime_id": vol_id,
            "entry_vol_regime": vol_names[vol_id],
            "entry_atr_bucket": int(row.get("atr_bucket", 2)),
            "entry_spread_bucket": int(row.get("spread_bucket", 1)),
            "entry_h4_trend_sign_cat": int(row.get("H4_trend_sign_cat", 1)),
            "entry_trend_regime_id": trend_id,
            "entry_trend_regime": trend_names[trend_id],
            "raw_direction_logits": logits.tolist(),
            "direction_logits": logits.tolist(),
            "direction_probs": probs.tolist(),
            "model_direction_index": direction_index,
            "model_direction": ("LONG", "SHORT", "FLAT")[direction_index],
            "entry_shared_representation": [
                float(index - 64) / 64.0 for index in range(128)
            ],
            "selected_side": direction_index if direction_index in (0, 1) else None,
            "public_trade_flat_decision_logits": public_logits.tolist(),
            "public_trade_flat_decision_probs": public_probs.tolist(),
            "public_trade_flat_decision_index": int(np.argmax(public_logits)),
            "public_trade_flat_decision": (
                ("TRADE", "FLAT")[int(np.argmax(public_logits))]
            ),
            "path_quality_raw": 1.0,
            "path_quality": 1.0,
            "path_quality_pred": 1.0,
            "path_quality_log_var": path_log_var,
            "path_quality_std": float(
                path_calibration["path_quality_scale"]
            ) * 0.5,
            "mfe_first_n": 2.0,
            "mfe_first_n_pred": 2.0,
            "bad_path_logit_raw": -1.0,
            "bad_path_logit": -1.0,
            "bad_path_prob": 1.0 / (1.0 + math.exp(1.0)),
            "tradable_logit": 1.0,
            "tradable_prob": 1.0 / (1.0 + math.exp(-1.0)),
            "clean_edge_logit": 0.7,
            "clean_edge_prob": 1.0 / (1.0 + math.exp(-0.7)),
            "survival_logit": 0.5,
            "survival_prob": 1.0 / (1.0 + math.exp(-0.5)),
            "p_trade": float(public_probs[0]),
            "p_flat_hier": float(public_probs[1]),
            "atr_bps": float(row.get("atr_bps", 12.0)),
            "tf_agreement_logit": tf_logit,
            "tf_agreement_pred": 1.0 / (1.0 + math.exp(-tf_logit)),
            "position_size_logit": size_logit,
            "position_size_pred": 1.0 / (1.0 + math.exp(-size_logit)),
            "p_long_given_trade": float(side_probs[0]),
            "p_short_given_trade": float(side_probs[1]),
            "side_logits": side_logits,
            "side_probs": side_probs.tolist(),
            "side_utility": [0.2, -0.1],
            "side_bad_path_logit": side_bad,
            "long_bad_path_prob": 1.0 / (1.0 + math.exp(-side_bad[0])),
            "short_bad_path_prob": 1.0 / (1.0 + math.exp(-side_bad[1])),
            "side_validity_logit": side_validity,
            "long_validity_prob": 1.0 / (1.0 + math.exp(-side_validity[0])),
            "short_validity_prob": 1.0 / (1.0 + math.exp(-side_validity[1])),
            "side_mae": [-0.2, -0.3],
            "mtf_dir_logits": mtf_logits,
            "mtf_dir_probs": mtf_probs.tolist(),
            "mtf_trend_evidence": 0.2,
            "specialist_names": list(MODEL_NATIVE_TRAINING_SPECIALISTS),
            "specialist_gate": [0.125] * len(MODEL_NATIVE_TRAINING_SPECIALISTS),
            **model_native_mtf_cooperation_evidence(),
            "trendline_rail_logits": rail_logits,
            "trendline_rail_probs": [
                1.0 / (1.0 + math.exp(-value)) for value in rail_logits
            ],
            "geometry_channel_edge_pressure": 0.2,
            "geometry_rising_support_rail_long_pressure": 0.3,
            "geometry_rising_support_rail_short_trap_pressure": 0.1,
            "geometry_falling_resistance_rail_short_pressure": 0.2,
            "geometry_falling_resistance_rail_long_trap_pressure": 0.1,
            "calibration_version": direction_calibration["version"],
            "direction_calibration_enabled": True,
            "direction_calibration_temperature": direction_calibration["temperature"],
            "path_calibration_enabled": True,
            "path_calibration": project_model_native_path_calibration(
                path_calibration
            ),
            "dip_pred": [0.0] * 18,
            "forecast_pred": [0.0] * 4,
            "timing_pred": list(row["timing_pred"]),
            "tail_risk_pred": [0.0] * 6,
            "vol_forecast_pred": [0.0] * 3,
            "action_value": action_value,
            "expectile_value": expectile_value,
            "action_advantage": action_advantage,
        }
        for name, width in DIRECTION_EVIDENCE_INPUTS:
            if name not in evidence:
                evidence[name] = 0.0 if width == 1 else [0.0] * width
        missing = (
            MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS
            - {"sizing_authority_contract"}
            - set(evidence)
        )
        if missing:
            raise AssertionError(f"runtime fixture missing fields: {sorted(missing)}")
        payload, payload_sha = encode_model_native_runtime_head_evidence(evidence)
        payloads.append(payload)
        hashes.append(payload_sha)
    return {
        "runtime_head_evidence_schema_version": [
            MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION
        ]
        * len(frame),
        "runtime_head_evidence_json": payloads,
        "runtime_head_evidence_sha256": hashes,
    }
