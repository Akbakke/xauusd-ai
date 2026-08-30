"""Regression coverage for the narrow candidate static-gate decoder."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gx1.contracts import unified_exit_gate_evidence_v1 as evidence


def _static_open_evidence(*, saturated: bool = False) -> dict[str, object]:
    rows = 17
    value: dict[str, object] = {
        "exit_gate_evidence_schema_version": evidence.SCHEMA_VERSION,
        "exit_cooperation_gate_health_ok": False,
        "exit_cooperation_gate_health_failures": [
            "family_tf_feature_gate constant/dead indices=[3]"
        ],
        "exit_family_tf_feature_gate_rows": rows,
        "exit_family_tf_feature_gate_mean_weight": np.full(
            int(np.prod(evidence.FEATURE_TF_GATE_SHAPE)), 0.5
        ).tolist(),
        "exit_family_tf_feature_gate_std_weight": np.zeros(
            int(np.prod(evidence.FEATURE_TF_GATE_SHAPE))
        ).tolist(),
        "exit_family_tf_feature_gate_min_observed": np.full(
            int(np.prod(evidence.FEATURE_TF_GATE_SHAPE)),
            0.0 if saturated else 0.5,
        ).tolist(),
        "exit_family_tf_feature_gate_max_observed": np.full(
            int(np.prod(evidence.FEATURE_TF_GATE_SHAPE)), 0.5
        ).tolist(),
        "exit_family_tf_feature_gate_min_std": 0.0,
    }
    for output_name, width in evidence.COOPERATION_GATE_WIDTHS.items():
        prefix = f"exit_{output_name}"
        weights = np.full(width, 1.0 / float(width))
        value.update(
            {
                f"{prefix}_rows": rows,
                f"{prefix}_mean_weight": weights.tolist(),
                f"{prefix}_entropy_mean": math.log(float(width)),
                f"{prefix}_min_mean": float(weights.min()),
            }
        )
    return value


def test_static_positive_open_gate_is_rejected_by_default_and_allowed_only_explicitly() -> None:
    value = _static_open_evidence()

    with pytest.raises(RuntimeError, match="exit_family_tf_feature_gate.dead"):
        evidence.require_unified_exit_gate_evidence(
            value,
            expected_rows=17,
            context="TEST",
        )

    evidence.require_unified_exit_gate_evidence(
        value,
        expected_rows=17,
        context="TEST",
        allow_static_feature_gate_provisional=True,
    )


def test_static_gate_provisional_rejects_zero_or_saturated_feature_route() -> None:
    value = _static_open_evidence(saturated=True)

    with pytest.raises(RuntimeError, match="exit_family_tf_feature_gate"):
        evidence.require_unified_exit_gate_evidence(
            value,
            expected_rows=17,
            context="TEST",
            allow_static_feature_gate_provisional=True,
        )
