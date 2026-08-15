from __future__ import annotations

import copy
import math

import pytest

from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_MTF_CONTEXT_COUNT
from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
    MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION,
    MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION,
    MODEL_NATIVE_RUNTIME_POLICY,
    ModelNativeRuntimeEvidenceError,
    decode_model_native_runtime_head_evidence,
    encode_model_native_runtime_head_evidence,
    require_model_native_entry_time,
    require_model_native_fill_time,
    require_model_native_runtime_evidence,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4


def _valid_evidence(*, with_timing: bool = False) -> dict:
    specialist_weight = 1.0 / len(MODEL_NATIVE_TRAINING_SPECIALISTS)
    tf_weight = 1.0 / ENTRY_MTF_CONTEXT_COUNT
    family_tf_weight = 1.0 / (
        ENTRY_MTF_CONTEXT_COUNT * len(MODEL_NATIVE_TRAINING_SPECIALISTS)
    )
    evidence = {
        "decision_ts": "2026-07-08T17:55:00+00:00",
        "runtime_evidence_schema_version": (
            MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION
        ),
        "model_policy": MODEL_NATIVE_RUNTIME_POLICY,
        "session_id": 2,
        "session": "OVERLAP",
        "entry_decision_representation": [0.01] * 128,
        "entry_action_q_bps": [2.0, -1.0, 0.0],
        "entry_action_q_margin_bps": 2.0,
        "entry_q_joint_hidden": [0.02] * 128,
        "model_direction_index": 0,
        "model_direction": "LONG",
        "selected_side": 0,
        "side_mae_bps": [2.0, 3.0],
        "trendline_event_logits": [0.1, -0.1, 0.2, -0.2],
        "dip_pred": [0.1] * 18,
        "forecast_pred": [0.1] * 4,
        "timing_pred": [0.1] * 12,
        "tail_risk_pred": [0.1] * 6,
        "vol_forecast_pred": [0.1] * 3,
        "atr_bps": 12.0,
        "position_size_logit": 0.0,
        "position_size_pred": 0.5,
        "specialist_names": list(MODEL_NATIVE_TRAINING_SPECIALISTS),
        "specialist_gate": [specialist_weight]
        * len(MODEL_NATIVE_TRAINING_SPECIALISTS),
        "tf_gate": [tf_weight] * ENTRY_MTF_CONTEXT_COUNT,
        "family_tf_cooperation_gate": [family_tf_weight]
        * (
            ENTRY_MTF_CONTEXT_COUNT
            * len(MODEL_NATIVE_TRAINING_SPECIALISTS)
        ),
        "family_tf_feature_gate": [1.0]
        * (
            ENTRY_MTF_CONTEXT_COUNT * len(MULTI_TF_PER_BAR_FEATURES_V4)
        ),
    }
    if with_timing:
        evidence.update(
            {
                "decision_available_ts": "2026-07-08T18:00:00+00:00",
                "entry_signal_latency_sec": 30.0,
                "context_cutoff_ts": evidence["decision_ts"],
                "context_age_m5_bars": 0,
            }
        )
    return evidence


def test_runtime_evidence_exact_raw_q_surface_round_trips() -> None:
    evidence = _valid_evidence()
    assert set(evidence) == set(MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS)
    assert require_model_native_runtime_evidence(evidence) == evidence
    head = {
        **evidence,
        "runtime_head_evidence_schema_version": (
            MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION
        ),
    }
    payload, payload_sha = encode_model_native_runtime_head_evidence(head)
    assert decode_model_native_runtime_head_evidence(
        payload, payload_sha
    ) == head


def test_runtime_evidence_rejects_raw_q_tie_or_argmax_alias_drift() -> None:
    tied = _valid_evidence()
    tied["entry_action_q_bps"] = [2.0, 2.0, 0.0]
    with pytest.raises(ModelNativeRuntimeEvidenceError, match="unique raw-Q"):
        require_model_native_runtime_evidence(tied)

    drift = _valid_evidence()
    drift["model_direction_index"] = 1
    drift["model_direction"] = "SHORT"
    drift["selected_side"] = 1
    with pytest.raises(ModelNativeRuntimeEvidenceError, match="argmax mismatch"):
        require_model_native_runtime_evidence(drift)


def test_runtime_evidence_rejects_q_margin_or_size_parity_drift() -> None:
    margin = _valid_evidence()
    margin["entry_action_q_margin_bps"] = 1.0
    with pytest.raises(ModelNativeRuntimeEvidenceError, match="margin mismatch"):
        require_model_native_runtime_evidence(margin)

    sizing = _valid_evidence()
    sizing["position_size_pred"] = 0.6
    with pytest.raises(ModelNativeRuntimeEvidenceError, match="sigmoid parity"):
        require_model_native_runtime_evidence(sizing)


@pytest.mark.parametrize(
    "retired_key",
    (
        "direction_" + "logits",
        "direction_" + "probs",
        "public_trade_" + "flat_decision",
        "bad_" + "path_prob",
        "hier_" + "trade_logit",
    ),
)
def test_runtime_evidence_rejects_retired_authority_fields(
    retired_key: str,
) -> None:
    evidence = _valid_evidence()
    evidence[retired_key] = 0.0
    with pytest.raises(
        ModelNativeRuntimeEvidenceError,
        match="retired fields|exact schema mismatch",
    ):
        require_model_native_runtime_evidence(evidence)


def test_runtime_head_hash_and_schema_tamper_fail_closed() -> None:
    head = {
        **_valid_evidence(),
        "runtime_head_evidence_schema_version": (
            MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION
        ),
    }
    payload, payload_sha = encode_model_native_runtime_head_evidence(head)
    with pytest.raises(ModelNativeRuntimeEvidenceError, match="hash mismatch"):
        decode_model_native_runtime_head_evidence(payload, "0" * 64)
    mutated = copy.deepcopy(head)
    mutated["runtime_head_evidence_schema_version"] = "stale"
    with pytest.raises(ModelNativeRuntimeEvidenceError, match="version mismatch"):
        encode_model_native_runtime_head_evidence(mutated)


def test_complete_execution_timing_is_bound_and_partial_timing_rejected() -> None:
    evidence = _valid_evidence(with_timing=True)
    assert require_model_native_entry_time(
        evidence, "2026-07-08T18:00:00+00:00"
    ).isoformat() == "2026-07-08T18:00:00+00:00"
    assert require_model_native_fill_time(
        evidence, "2026-07-08T18:00:30+00:00"
    ).isoformat() == "2026-07-08T18:00:30+00:00"
    partial = _valid_evidence()
    partial["decision_available_ts"] = "2026-07-08T18:00:00+00:00"
    with pytest.raises(ModelNativeRuntimeEvidenceError, match="absent or complete"):
        require_model_native_runtime_evidence(partial)


def test_runtime_vectors_are_finite_and_gate_contracts_are_exact() -> None:
    evidence = _valid_evidence()
    evidence["entry_q_joint_hidden"][0] = math.nan
    with pytest.raises(ModelNativeRuntimeEvidenceError, match="non-finite"):
        require_model_native_runtime_evidence(evidence)
    evidence = _valid_evidence()
    evidence["tf_gate"][0] = 0.0
    with pytest.raises(ModelNativeRuntimeEvidenceError, match="simplex"):
        require_model_native_runtime_evidence(evidence)
