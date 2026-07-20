"""Strict runtime evidence contract for model-native XAU Entry decisions.

The calibrated LONG/SHORT/FLAT logits are the sole direction authority. Every
auxiliary head below is immutable learned evidence: it may be reviewed and
journaled, but it may not become an external direction rule. Missing heads,
probability/logit disagreement, incomplete specialist fusion, disabled
calibration, and retired overlay fields all fail closed.
"""
from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any, NoReturn

from gx1.contracts.entry_model_native_sizing_authority_v1 import (
    MODEL_NATIVE_SIZING_MODE_LEARNED,
    require_model_native_sizing_authority_contract,
)
from gx1.contracts.entry_model_native_direction_evidence_fusion_v1 import (
    INPUTS as DIRECTION_EVIDENCE_FUSION_INPUTS,
)
from gx1.contracts.entry_model_native_offline_rl_v1 import (
    ACTION_VALUE_DIM,
    EXPECTILE_VALUE_DIM,
    HORIZON_COUNT as OFFLINE_RL_HORIZON_COUNT,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)


MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION = (
    "entry_model_native_runtime_evidence_v2"
)
MODEL_NATIVE_RUNTIME_POLICY = "xau_seq513_model_native_direction_argmax_v2"
MODEL_NATIVE_DECISION_AVAILABILITY_LAG_SEC = 300.0
MODEL_NATIVE_MAX_ENTRY_SIGNAL_LATENCY_SEC = 90.0
MODEL_DIRECTION_NAMES = ("LONG", "SHORT", "FLAT")
PUBLIC_TRADE_FLAT_NAMES = ("TRADE", "FLAT")
MODEL_NATIVE_SESSION_NAMES = ("ASIA", "EU", "OVERLAP", "US")
RETIRED_RUNTIME_EVIDENCE_FRAGMENTS = (
    "anchor",
    "sniper",
    "risk_guard",
    "entry_critic",
    "xgb",
    "q_take",
    "advantage_over_skip",
    "selection_score_threshold",
    "edge_score_threshold",
    "expected_utility",
    "hold_horizon",
)
MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS = frozenset(
    {
        "decision_ts",
        "runtime_evidence_schema_version",
        "model_policy",
        "session_id",
        "session",
        "direction_logits",
        "raw_direction_logits",
        "direction_probs",
        "model_direction_index",
        "model_direction",
        "selected_side",
        "public_trade_flat_decision_logits",
        "public_trade_flat_decision_probs",
        "public_trade_flat_decision_index",
        "public_trade_flat_decision",
        "path_quality",
        "path_quality_pred",
        "path_quality_log_var",
        "path_quality_std",
        "mfe_first_n",
        "mfe_first_n_pred",
        "bad_path_logit",
        "bad_path_prob",
        "tradable_logit",
        "tradable_prob",
        "clean_edge_logit",
        "clean_edge_prob",
        "survival_logit",
        "survival_prob",
        "dip_pred",
        "forecast_pred",
        "timing_pred",
        "tail_risk_pred",
        "vol_forecast_pred",
        "p_trade",
        "p_flat_hier",
        "atr_bps",
        "tf_agreement_logit",
        "tf_agreement_pred",
        "position_size_logit",
        "position_size_pred",
        "sizing_authority_contract",
        "p_long_given_trade",
        "p_short_given_trade",
        "side_logits",
        "side_probs",
        "long_bad_path_prob",
        "short_bad_path_prob",
        "side_validity_logit",
        "long_validity_prob",
        "short_validity_prob",
        "mtf_dir_logits",
        "mtf_dir_probs",
        "mtf_trend_evidence",
        "specialist_names",
        "specialist_gate",
        "trendline_rail_logits",
        "trendline_rail_probs",
        "geometry_channel_edge_pressure",
        "geometry_rising_support_rail_long_pressure",
        "geometry_rising_support_rail_short_trap_pressure",
        "geometry_falling_resistance_rail_short_pressure",
        "geometry_falling_resistance_rail_long_trap_pressure",
        "calibration_version",
        "direction_calibration_enabled",
        "direction_calibration_temperature",
        "direction_calibration_bias",
        "path_calibration_enabled",
        "path_calibration",
        *(name for name, _width in DIRECTION_EVIDENCE_FUSION_INPUTS),
    }
)
MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS = frozenset(
    {
        "decision_available_ts",
        "entry_signal_latency_sec",
        "context_cutoff_ts",
        "context_age_m5_bars",
    }
)


class ModelNativeRuntimeEvidenceError(RuntimeError):
    """Runtime evidence cannot prove one exact model-native decision."""


def _context_name(context: str) -> str:
    value = str(context).strip()
    return value if value else "ENTRY_MODEL_NATIVE_RUNTIME"


def _fail(context: str, field: str, detail: str) -> NoReturn:
    raise ModelNativeRuntimeEvidenceError(
        f"[{_context_name(context)}_MODEL_NATIVE_RUNTIME_EVIDENCE_INVALID] "
        f"{field}: {detail}"
    )


def _finite_scalar(
    evidence: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> float:
    if key not in evidence:
        _fail(context, key, "missing")
    value = evidence[key]
    if isinstance(value, bool):
        _fail(context, key, "boolean is not numeric evidence")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelNativeRuntimeEvidenceError(
            f"[{_context_name(context)}_MODEL_NATIVE_RUNTIME_EVIDENCE_INVALID] "
            f"{key}: non-numeric value {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        _fail(context, key, f"non-finite value {value!r}")
    return parsed


def _finite_vector(
    evidence: Mapping[str, Any],
    key: str,
    size: int,
    *,
    context: str,
) -> tuple[float, ...]:
    if key not in evidence or isinstance(evidence[key], (str, bytes, Mapping)):
        _fail(context, key, f"expected finite vector[{size}]")
    try:
        raw = list(evidence[key])
    except TypeError as exc:
        raise ModelNativeRuntimeEvidenceError(
            f"[{_context_name(context)}_MODEL_NATIVE_RUNTIME_EVIDENCE_INVALID] "
            f"{key}: expected finite vector[{size}]"
        ) from exc
    if len(raw) != size:
        _fail(context, key, f"size={len(raw)} expected={size}")
    parsed: list[float] = []
    for value in raw:
        if isinstance(value, bool):
            _fail(context, key, "boolean element")
        try:
            item = float(value)
        except (TypeError, ValueError) as exc:
            raise ModelNativeRuntimeEvidenceError(
                f"[{_context_name(context)}_MODEL_NATIVE_RUNTIME_EVIDENCE_INVALID] "
                f"{key}: non-numeric element"
            ) from exc
        if not math.isfinite(item):
            _fail(context, key, "non-finite element")
        parsed.append(item)
    return tuple(parsed)


def _softmax(values: tuple[float, ...]) -> tuple[float, ...]:
    peak = max(values)
    exp_values = tuple(math.exp(value - peak) for value in values)
    total = sum(exp_values)
    return tuple(value / total for value in exp_values)


def _sigmoid(value: float) -> float:
    clipped = max(-80.0, min(80.0, value))
    return 1.0 / (1.0 + math.exp(-clipped))


def _require_close(
    observed: tuple[float, ...],
    expected: tuple[float, ...],
    field: str,
    *,
    context: str,
) -> None:
    if len(observed) != len(expected) or any(
        not math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-7)
        for left, right in zip(observed, expected, strict=True)
    ):
        _fail(context, field, "parity mismatch")


def _require_probability(value: float, field: str, *, context: str) -> float:
    if not 0.0 <= value <= 1.0:
        _fail(context, field, f"outside [0,1]: {value!r}")
    return value


def _require_utc_timestamp(
    evidence: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> datetime:
    value = evidence.get(key)
    if not isinstance(value, str) or not value.strip():
        _fail(context, key, "missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ModelNativeRuntimeEvidenceError(
            f"[{_context_name(context)}_MODEL_NATIVE_RUNTIME_EVIDENCE_INVALID] "
            f"{key}: invalid ISO timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        _fail(context, key, "must be timezone-aware UTC")
    return parsed


def _exact_integer(
    evidence: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> int:
    """Return an integer field without accepting bool/float lookalikes."""

    if key not in evidence or isinstance(evidence[key], bool):
        _fail(context, key, "must be an exact integer")
    try:
        parsed = operator.index(evidence[key])
    except TypeError as exc:
        raise ModelNativeRuntimeEvidenceError(
            f"[{_context_name(context)}_MODEL_NATIVE_RUNTIME_EVIDENCE_INVALID] "
            f"{key}: must be an exact integer"
        ) from exc
    return int(parsed)


def require_model_native_runtime_evidence(
    evidence: Mapping[str, Any],
    context: str = "ENTRY_MODEL_NATIVE_RUNTIME",
) -> dict[str, Any]:
    """Validate and return an exact model-native runtime evidence snapshot.

    No field is filled, clipped, inferred from a retired surface, or silently
    converted into a direction rule. The returned mapping is a shallow copy so
    downstream persistence cannot mutate the caller's dictionary by accident.
    """

    if not isinstance(evidence, Mapping) or not evidence:
        _fail(context, "evidence", "missing or empty")
    validated = dict(evidence)

    retired = sorted(
        key
        for key in validated
        if any(
            fragment in str(key).lower()
            for fragment in RETIRED_RUNTIME_EVIDENCE_FRAGMENTS
        )
    )
    if retired:
        _fail(context, "evidence", f"retired fields={retired}")
    missing = sorted(MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS - set(validated))
    unexpected = sorted(
        set(validated)
        - MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS
        - MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS
    )
    if missing or unexpected:
        _fail(
            context,
            "evidence",
            f"exact schema mismatch missing={missing} unexpected={unexpected}",
        )

    if validated.get("model_policy") != MODEL_NATIVE_RUNTIME_POLICY:
        _fail(
            context,
            "model_policy",
            f"{validated.get('model_policy')!r} != {MODEL_NATIVE_RUNTIME_POLICY!r}",
        )
    if (
        validated.get("runtime_evidence_schema_version")
        != MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION
    ):
        _fail(
            context,
            "runtime_evidence_schema_version",
            f"{validated.get('runtime_evidence_schema_version')!r} != "
            f"{MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION!r}",
        )
    session_id = _exact_integer(validated, "session_id", context=context)
    if session_id not in range(len(MODEL_NATIVE_SESSION_NAMES)):
        _fail(context, "session_id", "must be one of 0,1,2,3")
    if validated.get("session") != MODEL_NATIVE_SESSION_NAMES[session_id]:
        _fail(context, "session", "session_id/name mismatch")

    # Every exact raw tensor that enters the sole learned 96-wide direction
    # fusion is mandatory runtime evidence with its immutable width.  This is
    # observability only; the final calibrated direction argmax remains the
    # sole executable direction authority.
    for fusion_name, fusion_width in DIRECTION_EVIDENCE_FUSION_INPUTS:
        if fusion_width == 1:
            _finite_scalar(validated, fusion_name, context=context)
        else:
            _finite_vector(
                validated,
                fusion_name,
                fusion_width,
                context=context,
            )

    action_value = _finite_vector(
        validated,
        "action_value",
        ACTION_VALUE_DIM,
        context=context,
    )
    expectile_value = _finite_vector(
        validated,
        "expectile_value",
        EXPECTILE_VALUE_DIM,
        context=context,
    )
    action_advantage = _finite_vector(
        validated,
        "action_advantage",
        ACTION_VALUE_DIM,
        context=context,
    )
    expected_advantage = tuple(
        action_value[index] - expectile_value[index % OFFLINE_RL_HORIZON_COUNT]
        for index in range(ACTION_VALUE_DIM)
    )
    _require_close(
        action_advantage,
        expected_advantage,
        "action_advantage",
        context=context,
    )

    raw_direction_logits = _finite_vector(
        validated, "raw_direction_logits", 3, context=context
    )
    direction_logits = _finite_vector(
        validated, "direction_logits", 3, context=context
    )
    direction_probs = _finite_vector(
        validated, "direction_probs", 3, context=context
    )
    _require_close(
        direction_probs,
        _softmax(direction_logits),
        "direction_probs",
        context=context,
    )
    direction_index = max(range(3), key=direction_probs.__getitem__)
    if _exact_integer(
        validated,
        "model_direction_index",
        context=context,
    ) != direction_index:
        _fail(context, "model_direction_index", "argmax mismatch")
    if validated.get("model_direction") != MODEL_DIRECTION_NAMES[direction_index]:
        _fail(context, "model_direction", "index/name mismatch")
    expected_side = direction_index if direction_index in (0, 1) else None
    if expected_side is None:
        if validated.get("selected_side") is not None:
            _fail(context, "selected_side", "direction parity mismatch")
    elif _exact_integer(validated, "selected_side", context=context) != expected_side:
        _fail(context, "selected_side", "direction parity mismatch")

    public_logits = _finite_vector(
        validated,
        "public_trade_flat_decision_logits",
        2,
        context=context,
    )
    expected_public_logits = (
        max(direction_logits[0], direction_logits[1]),
        direction_logits[2],
    )
    if public_logits != expected_public_logits:
        _fail(context, "public_trade_flat_decision_logits", "direction parity mismatch")
    public_probs = _finite_vector(
        validated,
        "public_trade_flat_decision_probs",
        2,
        context=context,
    )
    _require_close(
        public_probs,
        _softmax(public_logits),
        "public_trade_flat_decision_probs",
        context=context,
    )
    public_index = max(range(2), key=public_probs.__getitem__)
    if _exact_integer(
        validated,
        "public_trade_flat_decision_index",
        context=context,
    ) != public_index:
        _fail(context, "public_trade_flat_decision_index", "argmax mismatch")
    if (
        validated.get("public_trade_flat_decision")
        != PUBLIC_TRADE_FLAT_NAMES[public_index]
    ):
        _fail(context, "public_trade_flat_decision", "index/name mismatch")

    for key in (
        "path_quality",
        "mfe_first_n",
        "tradable_prob",
        "bad_path_prob",
        "p_trade",
        "p_flat_hier",
        "atr_bps",
        "tf_agreement_logit",
        "tf_agreement_pred",
        "path_quality_log_var",
        "path_quality_std",
        "position_size_logit",
        "position_size_pred",
        "path_quality_pred",
        "mfe_first_n_pred",
        "bad_path_logit",
        "tradable_logit",
        "clean_edge_logit",
        "clean_edge_prob",
        "survival_logit",
        "survival_prob",
    ):
        _finite_scalar(validated, key, context=context)
    for key in (
        "tradable_prob",
        "bad_path_prob",
        "tf_agreement_pred",
        "position_size_pred",
        "clean_edge_prob",
        "survival_prob",
    ):
        _require_probability(float(validated[key]), key, context=context)
    if float(validated["atr_bps"]) <= 0.0:
        _fail(context, "atr_bps", "must be positive")
    if float(validated["path_quality_std"]) <= 0.0:
        _fail(context, "path_quality_std", "must be positive")
    _require_close(
        (float(validated["p_trade"]), float(validated["p_flat_hier"])),
        public_probs,
        "public hierarchy probabilities",
        context=context,
    )
    _require_close(
        (float(validated["path_quality_pred"]),),
        (float(validated["path_quality"]),),
        "path_quality alias",
        context=context,
    )
    _require_close(
        (float(validated["mfe_first_n_pred"]),),
        (float(validated["mfe_first_n"]),),
        "mfe_first_n alias",
        context=context,
    )
    for logit_key, probability_key in (
        ("bad_path_logit", "bad_path_prob"),
        ("tradable_logit", "tradable_prob"),
        ("clean_edge_logit", "clean_edge_prob"),
        ("survival_logit", "survival_prob"),
    ):
        _require_close(
            (float(validated[probability_key]),),
            (_sigmoid(float(validated[logit_key])),),
            probability_key,
            context=context,
        )
    _require_close(
        (float(validated["tf_agreement_pred"]),),
        (_sigmoid(float(validated["tf_agreement_logit"])),),
        "tf_agreement_pred",
        context=context,
    )
    _require_close(
        (float(validated["position_size_pred"]),),
        (_sigmoid(float(validated["position_size_logit"])),),
        "position_size_pred",
        context=context,
    )
    try:
        require_model_native_sizing_authority_contract(
            validated["sizing_authority_contract"],
            context=f"{context} runtime evidence",
            required_mode=MODEL_NATIVE_SIZING_MODE_LEARNED,
        )
    except RuntimeError as exc:
        _fail(context, "sizing_authority_contract", str(exc))
    try:
        expected_path_std = math.exp(
            0.5 * float(validated["path_quality_log_var"])
        )
    except OverflowError:
        _fail(context, "path_quality_std", "derived value overflowed")
    if not math.isfinite(expected_path_std):
        _fail(context, "path_quality_std", "derived value is non-finite")
    _require_close(
        (float(validated["path_quality_std"]),),
        (expected_path_std,),
        "path_quality_std",
        context=context,
    )

    side_logits = _finite_vector(validated, "side_logits", 2, context=context)
    side_probs = _finite_vector(validated, "side_probs", 2, context=context)
    _require_close(side_probs, _softmax(side_logits), "side_probs", context=context)
    conditional = (
        _require_probability(
            _finite_scalar(validated, "p_long_given_trade", context=context),
            "p_long_given_trade",
            context=context,
        ),
        _require_probability(
            _finite_scalar(validated, "p_short_given_trade", context=context),
            "p_short_given_trade",
            context=context,
        ),
    )
    _require_close(
        conditional,
        side_probs,
        "conditional side probabilities",
        context=context,
    )
    _finite_vector(validated, "side_utility", 2, context=context)
    for key, size in (
        ("dip_pred", 18),
        ("forecast_pred", 4),
        ("timing_pred", 12),
        ("tail_risk_pred", 6),
        ("vol_forecast_pred", 3),
    ):
        _finite_vector(validated, key, size, context=context)

    side_bad_logits = _finite_vector(
        validated, "side_bad_path_logit", 2, context=context
    )
    side_bad_probs = (
        _require_probability(
            _finite_scalar(validated, "long_bad_path_prob", context=context),
            "long_bad_path_prob",
            context=context,
        ),
        _require_probability(
            _finite_scalar(validated, "short_bad_path_prob", context=context),
            "short_bad_path_prob",
            context=context,
        ),
    )
    _require_close(
        side_bad_probs,
        tuple(_sigmoid(value) for value in side_bad_logits),
        "side bad-path probabilities",
        context=context,
    )
    side_validity_logits = _finite_vector(
        validated, "side_validity_logit", 2, context=context
    )
    side_validity_probs = (
        _require_probability(
            _finite_scalar(validated, "long_validity_prob", context=context),
            "long_validity_prob",
            context=context,
        ),
        _require_probability(
            _finite_scalar(validated, "short_validity_prob", context=context),
            "short_validity_prob",
            context=context,
        ),
    )
    _require_close(
        side_validity_probs,
        tuple(_sigmoid(value) for value in side_validity_logits),
        "side validity probabilities",
        context=context,
    )
    _finite_vector(validated, "side_mae", 2, context=context)

    mtf_logits = _finite_vector(validated, "mtf_dir_logits", 3, context=context)
    mtf_probs = _finite_vector(validated, "mtf_dir_probs", 3, context=context)
    _require_close(
        mtf_probs,
        _softmax(mtf_logits),
        "mtf_dir_probs",
        context=context,
    )
    _finite_scalar(validated, "mtf_trend_evidence", context=context)

    specialist_names = validated.get("specialist_names")
    if not isinstance(specialist_names, (list, tuple)) or tuple(
        specialist_names
    ) != tuple(MODEL_NATIVE_TRAINING_SPECIALISTS):
        _fail(context, "specialist_names", "exact eight-specialist contract mismatch")
    specialist_gate = _finite_vector(
        validated,
        "specialist_gate",
        len(MODEL_NATIVE_TRAINING_SPECIALISTS),
        context=context,
    )
    if any(value < 0.0 for value in specialist_gate) or not math.isclose(
        sum(specialist_gate), 1.0, rel_tol=1e-6, abs_tol=1e-7
    ):
        _fail(context, "specialist_gate", "not a probability simplex")

    for key in (
        "geometry_channel_edge_pressure",
        "geometry_rising_support_rail_long_pressure",
        "geometry_rising_support_rail_short_trap_pressure",
        "geometry_falling_resistance_rail_short_pressure",
        "geometry_falling_resistance_rail_long_trap_pressure",
    ):
        _finite_scalar(validated, key, context=context)
    rail_logits = _finite_vector(
        validated, "trendline_rail_logits", 6, context=context
    )
    rail_probs = _finite_vector(
        validated, "trendline_rail_probs", 6, context=context
    )
    _require_close(
        rail_probs,
        tuple(_sigmoid(value) for value in rail_logits),
        "trendline_rail_probs",
        context=context,
    )

    decision_ts = _require_utc_timestamp(validated, "decision_ts", context=context)
    if (
        decision_ts.second != 0
        or decision_ts.microsecond != 0
        or decision_ts.minute % 5 != 0
    ):
        _fail(context, "decision_ts", "must be an exact closed-M5 bar-start timestamp")
    calibration_version = validated.get("calibration_version")
    if not isinstance(calibration_version, str) or not calibration_version.strip():
        _fail(context, "calibration_version", "missing")
    if validated.get("direction_calibration_enabled") is not True:
        _fail(context, "direction_calibration_enabled", "must be true")
    if (
        _finite_scalar(
            validated, "direction_calibration_temperature", context=context
        )
        <= 0.0
    ):
        _fail(context, "direction_calibration_temperature", "must be positive")
    direction_bias = _finite_vector(
        validated, "direction_calibration_bias", 3, context=context
    )
    direction_temperature = float(validated["direction_calibration_temperature"])
    _require_close(
        direction_logits,
        tuple(
            raw / direction_temperature + bias
            for raw, bias in zip(raw_direction_logits, direction_bias, strict=True)
        ),
        "raw direction calibration equation",
        context=context,
    )
    if validated.get("path_calibration_enabled") is not True:
        _fail(context, "path_calibration_enabled", "must be true")
    path_calibration = validated.get("path_calibration")
    if not isinstance(path_calibration, Mapping) or path_calibration.get("enabled") is not True:
        _fail(context, "path_calibration", "missing or disabled")
    path_calibration_keys = {
        "enabled",
        "version",
        "path_quality_scale",
        "path_quality_shift",
        "bad_path_temperature",
        "bad_path_bias",
    }
    if set(path_calibration) != path_calibration_keys:
        _fail(
            context,
            "path_calibration",
            "exact schema mismatch "
            f"missing={sorted(path_calibration_keys - set(path_calibration))} "
            f"unexpected={sorted(set(path_calibration) - path_calibration_keys)}",
        )
    path_calibration_version = path_calibration.get("version")
    if not isinstance(path_calibration_version, str) or not path_calibration_version.strip():
        _fail(context, "path_calibration.version", "missing")
    path_quality_scale = _finite_scalar(
        path_calibration,
        "path_quality_scale",
        context=f"{context}_PATH_CALIBRATION",
    )
    path_quality_shift = _finite_scalar(
        path_calibration,
        "path_quality_shift",
        context=f"{context}_PATH_CALIBRATION",
    )
    bad_path_temperature = _finite_scalar(
        path_calibration,
        "bad_path_temperature",
        context=f"{context}_PATH_CALIBRATION",
    )
    bad_path_bias = _finite_scalar(
        path_calibration,
        "bad_path_bias",
        context=f"{context}_PATH_CALIBRATION",
    )
    if path_quality_scale <= 0.0 or bad_path_temperature <= 0.0:
        _fail(context, "path_calibration", "scales must be positive")
    _require_close(
        (float(validated["path_quality"]),),
        (
            float(validated["path_quality_raw"]) * path_quality_scale
            + path_quality_shift,
        ),
        "path_quality raw calibration equation",
        context=context,
    )
    _require_close(
        (float(validated["bad_path_logit"]),),
        (
            float(validated["bad_path_logit_raw"]) / bad_path_temperature
            + bad_path_bias,
        ),
        "bad_path raw calibration equation",
        context=context,
    )

    optional_timing = set(validated) & MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS
    if optional_timing and optional_timing != set(
        MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS
    ):
        _fail(
            context,
            "timing evidence",
            "must be absent or complete; "
            f"missing={sorted(MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS - optional_timing)}",
        )
    if optional_timing:
        decision_available_ts = _require_utc_timestamp(
            validated,
            "decision_available_ts",
            context=context,
        )
        context_cutoff_ts = _require_utc_timestamp(
            validated,
            "context_cutoff_ts",
            context=context,
        )
        availability_lag_sec = (decision_available_ts - decision_ts).total_seconds()
        if not math.isclose(
            availability_lag_sec,
            MODEL_NATIVE_DECISION_AVAILABILITY_LAG_SEC,
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            _fail(
                context,
                "decision_available_ts",
                "must equal decision_ts + 300 seconds",
            )
        if context_cutoff_ts != decision_ts:
            _fail(
                context,
                "context_cutoff_ts",
                "must equal decision_ts for the zero-staleness serving contract",
            )
        latency_sec = _finite_scalar(
            validated,
            "entry_signal_latency_sec",
            context=context,
        )
        if not 0.0 <= latency_sec <= MODEL_NATIVE_MAX_ENTRY_SIGNAL_LATENCY_SEC:
            _fail(
                context,
                "entry_signal_latency_sec",
                "must be within the immutable [0,90] second serving window",
            )
        context_age = validated.get("context_age_m5_bars")
        if isinstance(context_age, bool) or not isinstance(context_age, int) or context_age != 0:
            _fail(
                context,
                "context_age_m5_bars",
                "must be exact integer 0 for the zero-staleness serving contract",
            )
    return validated


def require_model_native_entry_time(
    evidence: Mapping[str, Any],
    entry_time: Any,
    *,
    context: str = "ENTRY_MODEL_NATIVE_RUNTIME",
) -> datetime:
    """Bind the executable wrapper's M1 entry time to model timing evidence.

    The runner operates at minute resolution.  Its only authoritative entry
    minute is therefore ``floor(decision_available_ts + latency, 1 minute)``;
    callers may not attach a different wrapper/journal/state timestamp.
    """

    validated = require_model_native_runtime_evidence(evidence, context=context)
    if not MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS.issubset(validated):
        _fail(context, "entry_time", "complete executable timing evidence is required")
    if isinstance(entry_time, bool) or entry_time is None:
        _fail(context, "entry_time", "missing or invalid")
    try:
        observed = datetime.fromisoformat(
            str(entry_time).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ModelNativeRuntimeEvidenceError(
            f"[{_context_name(context)}_MODEL_NATIVE_RUNTIME_EVIDENCE_INVALID] "
            "entry_time: invalid ISO timestamp"
        ) from exc
    if (
        observed.tzinfo is None
        or observed.utcoffset() != timezone.utc.utcoffset(observed)
    ):
        _fail(context, "entry_time", "must be timezone-aware UTC")
    available = _require_utc_timestamp(
        validated,
        "decision_available_ts",
        context=context,
    )
    expected = (
        available
        + timedelta(seconds=float(validated["entry_signal_latency_sec"]))
    ).replace(second=0, microsecond=0)
    if observed != expected:
        _fail(
            context,
            "entry_time",
            f"{observed.isoformat()} != model-derived minute {expected.isoformat()}",
        )
    return observed


__all__ = [
    "MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION",
    "MODEL_NATIVE_RUNTIME_POLICY",
    "MODEL_NATIVE_DECISION_AVAILABILITY_LAG_SEC",
    "MODEL_NATIVE_MAX_ENTRY_SIGNAL_LATENCY_SEC",
    "MODEL_NATIVE_SESSION_NAMES",
    "MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS",
    "MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS",
    "RETIRED_RUNTIME_EVIDENCE_FRAGMENTS",
    "ModelNativeRuntimeEvidenceError",
    "require_model_native_entry_time",
    "require_model_native_runtime_evidence",
]
