"""Strict evidence contract for learned Exit specialist/MTF cooperation gates."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from gx1.contracts.entry_exit_feature_base_v1 import EXIT_MTF_CONTEXT_COUNT
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)
from gx1.features.htf_features import MULTI_TF_FEATURE_COUNT_V4


SCHEMA_VERSION = "gx1_unified_exit_gate_evidence_v1"
FEATURE_GATE_MIN_STD = 1e-6
COOPERATION_GATE_WIDTHS = {
    "specialist_gate": len(MODEL_NATIVE_TRAINING_SPECIALISTS),
    "tf_gate": EXIT_MTF_CONTEXT_COUNT,
    "family_tf_cooperation_gate": (
        EXIT_MTF_CONTEXT_COUNT * len(MODEL_NATIVE_TRAINING_SPECIALISTS)
    ),
}
FEATURE_TF_GATE_SHAPE = (
    EXIT_MTF_CONTEXT_COUNT,
    MULTI_TF_FEATURE_COUNT_V4,
)


def _finite_vector(value: Any, *, width: int) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape != (width,) or not bool(np.isfinite(array).all()):
        return None
    return array


def require_unified_exit_gate_evidence(
    value: Mapping[str, Any],
    *,
    expected_rows: int,
    context: str,
) -> None:
    """Recompute the five-TF Exit health verdict from persisted raw evidence."""

    if not isinstance(value, Mapping):
        raise RuntimeError(f"[{context}_EXIT_GATE_EVIDENCE_MISSING]")
    if value.get("exit_gate_evidence_schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"[{context}_EXIT_GATE_EVIDENCE_SCHEMA_INVALID]")
    if isinstance(expected_rows, bool) or int(expected_rows) <= 0:
        raise RuntimeError(f"[{context}_EXIT_GATE_EXPECTED_ROWS_INVALID]")
    failures: list[str] = []
    for output_name, width in COOPERATION_GATE_WIDTHS.items():
        prefix = f"exit_{output_name}"
        rows = value.get(f"{prefix}_rows")
        weights = _finite_vector(
            value.get(f"{prefix}_mean_weight"),
            width=width,
        )
        entropy = value.get(f"{prefix}_entropy_mean")
        stored_min = value.get(f"{prefix}_min_mean")
        if isinstance(rows, bool) or rows != int(expected_rows):
            failures.append(f"{prefix}.rows")
        if (
            weights is None
            or not math.isclose(
                float(weights.sum()),
                1.0,
                rel_tol=1e-6,
                abs_tol=1e-7,
            )
        ):
            failures.append(f"{prefix}.mean_weight")
            continue
        observed_min = float(weights.min())
        # Exact positivity and positive empirical entropy are liveness
        # invariants, not a hand-written target distribution or gradient term.
        if observed_min <= 0.0:
            failures.append(f"{prefix}.starved")
        if (
            isinstance(stored_min, bool)
            or not isinstance(stored_min, (int, float))
            or not math.isfinite(float(stored_min))
            or not math.isclose(
                float(stored_min),
                observed_min,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            failures.append(f"{prefix}.min_mean")
        if (
            isinstance(entropy, bool)
            or not isinstance(entropy, (int, float))
            or not math.isfinite(float(entropy))
            or float(entropy) <= 0.0
        ):
            failures.append(f"{prefix}.entropy")

    feature_count = int(np.prod(FEATURE_TF_GATE_SHAPE))
    feature_rows = value.get("exit_family_tf_feature_gate_rows")
    feature_mean = _finite_vector(
        value.get("exit_family_tf_feature_gate_mean_weight"),
        width=feature_count,
    )
    feature_std = _finite_vector(
        value.get("exit_family_tf_feature_gate_std_weight"),
        width=feature_count,
    )
    feature_min = _finite_vector(
        value.get("exit_family_tf_feature_gate_min_observed"),
        width=feature_count,
    )
    feature_max = _finite_vector(
        value.get("exit_family_tf_feature_gate_max_observed"),
        width=feature_count,
    )
    stored_min_std = value.get("exit_family_tf_feature_gate_min_std")
    if isinstance(feature_rows, bool) or feature_rows != int(expected_rows):
        failures.append("exit_family_tf_feature_gate.rows")
    if any(
        item is None
        for item in (feature_mean, feature_std, feature_min, feature_max)
    ):
        failures.append("exit_family_tf_feature_gate.vectors")
    else:
        assert feature_std is not None
        assert feature_min is not None
        assert feature_max is not None
        observed_min_std = float(feature_std.min())
        if observed_min_std <= FEATURE_GATE_MIN_STD:
            failures.append("exit_family_tf_feature_gate.dead")
        if bool(((feature_min <= 0.0) | (feature_max >= 2.0)).any()):
            failures.append("exit_family_tf_feature_gate.saturated")
        if (
            isinstance(stored_min_std, bool)
            or not isinstance(stored_min_std, (int, float))
            or not math.isfinite(float(stored_min_std))
            or not math.isclose(
                float(stored_min_std),
                observed_min_std,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            failures.append("exit_family_tf_feature_gate.min_std")
    if value.get("exit_cooperation_gate_health_ok") is not True:
        failures.append("exit_cooperation_gate_health_ok")
    if value.get("exit_cooperation_gate_health_failures") != []:
        failures.append("exit_cooperation_gate_health_failures")
    if failures:
        raise RuntimeError(
            f"[{context}_EXIT_GATE_EVIDENCE_INVALID] " + "; ".join(failures)
        )


__all__ = [
    "COOPERATION_GATE_WIDTHS",
    "FEATURE_GATE_MIN_STD",
    "FEATURE_TF_GATE_SHAPE",
    "SCHEMA_VERSION",
    "require_unified_exit_gate_evidence",
]
