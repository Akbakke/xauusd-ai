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
    allow_static_feature_gate_provisional: bool = False,
) -> None:
    """Recompute the five-TF Exit health verdict from persisted raw evidence.

    ``allow_static_feature_gate_provisional`` is not a general softening of
    gate liveness.  It permits only a positive, non-saturated feature
    multiplier whose finite-window standard deviation is zero.  The caller
    must separately bind that narrow exception to a selected-checkpoint
    direct-input proof (plus the Entry family-ablation proof) before it can
    publish any bundle.
    """

    if not isinstance(value, Mapping):
        raise RuntimeError(f"[{context}_EXIT_GATE_EVIDENCE_MISSING]")
    if value.get("exit_gate_evidence_schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"[{context}_EXIT_GATE_EVIDENCE_SCHEMA_INVALID]")
    if isinstance(expected_rows, bool) or int(expected_rows) <= 0:
        raise RuntimeError(f"[{context}_EXIT_GATE_EXPECTED_ROWS_INVALID]")
    if not isinstance(allow_static_feature_gate_provisional, bool):
        raise RuntimeError(f"[{context}_EXIT_GATE_PROVISIONAL_ARGUMENT_INVALID]")
    failures: list[str] = []
    static_feature_gate_only = False
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
            static_feature_gate_only = bool(
                allow_static_feature_gate_provisional
                and bool((feature_min > 0.0).all())
                and bool((feature_max < 2.0).all())
            )
            if not static_feature_gate_only:
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
    reported_failures = value.get("exit_cooperation_gate_health_failures")
    expected_static_failure = (
        isinstance(reported_failures, list)
        and len(reported_failures) == 1
        and isinstance(reported_failures[0], str)
        and reported_failures[0].startswith(
            "family_tf_feature_gate constant/dead indices="
        )
    )
    if value.get("exit_cooperation_gate_health_ok") is not True and not (
        static_feature_gate_only and expected_static_failure
    ):
        failures.append("exit_cooperation_gate_health_ok")
    if reported_failures != [] and not (
        static_feature_gate_only and expected_static_failure
    ):
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
