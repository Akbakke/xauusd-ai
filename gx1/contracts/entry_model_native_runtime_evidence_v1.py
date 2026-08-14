"""Strict immutable runtime evidence for fitted-Q XAU Entry decisions.

The unique raw-bps LONG/SHORT/FLAT Q argmax is the sole Entry authority.
Genuine auxiliary predictions and learned routing gates may be persisted for
representation diagnostics, but no probability, calibration, threshold, or
hierarchical side alias is admitted.
"""
from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any, NoReturn

from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_MTF_CONTEXT_COUNT
from gx1.contracts.entry_fitted_q_v1 import ENTRY_FITTED_Q_ACTION_ORDER
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_DIP_OUTPUT_DIM,
    MODEL_NATIVE_FORECAST_TARGET_COLUMNS,
    MODEL_NATIVE_TAIL_RISK_TARGET_COLUMNS,
    MODEL_NATIVE_TIMING_TARGET_COLUMNS,
    MODEL_NATIVE_VOL_FORECAST_TARGET_COLUMNS,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DOMAINS,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_TRADE_INDICES,
    UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM,
    UNIFIED_EXIT_ENTRY_REPRESENTATION_KEY,
)
from gx1.time.session_detector import SESSION_ORDER


MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION = (
    "entry_model_native_runtime_evidence_v14"
)
MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION = (
    "entry_model_native_runtime_head_evidence_v12"
)
MODEL_NATIVE_RUNTIME_POLICY = "xau_seq513_entry_fitted_q_unique_argmax_v10"
MODEL_NATIVE_DECISION_AVAILABILITY_LAG_SEC = 300.0
MODEL_NATIVE_MAX_ENTRY_SIGNAL_LATENCY_SEC = 90.0
MODEL_DIRECTION_NAMES = ENTRY_FITTED_Q_ACTION_ORDER
MODEL_NATIVE_SESSION_NAMES = SESSION_ORDER

RETIRED_RUNTIME_EVIDENCE_FRAGMENTS = (
    "anchor",
    "sniper",
    "risk_guard",
    "entry_critic",
    "direction_logit",
    "direction_prob",
    "calibration",
    "tradable",
    "bad_path",
    "path_quality",
    "mfe_first_n",
    "clean_edge",
    "survival",
    "hier",
    "side_validity",
    "side_utility",
    "side_logits",
    "side_probs",
    "trade_logit",
    "trendline_rail",
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
        UNIFIED_EXIT_ENTRY_REPRESENTATION_KEY,
        "entry_action_q_bps",
        "entry_action_q_margin_bps",
        "entry_q_joint_hidden",
        "model_direction_index",
        "model_direction",
        "selected_side",
        "side_mae_bps",
        "trendline_event_logits",
        "dip_pred",
        "forecast_pred",
        "timing_pred",
        "tail_risk_pred",
        "vol_forecast_pred",
        "atr_bps",
        "position_size_logit",
        "position_size_pred",
        "specialist_names",
        "specialist_gate",
        "tf_gate",
        "family_tf_cooperation_gate",
        "family_tf_feature_gate",
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
MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_REQUIRED_FIELDS = frozenset(
    {"runtime_head_evidence_schema_version", *MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS}
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


def _finite_scalar(evidence: Mapping[str, Any], key: str, *, context: str) -> float:
    if key not in evidence or isinstance(evidence[key], bool):
        _fail(context, key, "missing or boolean")
    try:
        value = float(evidence[key])
    except (TypeError, ValueError) as exc:
        raise ModelNativeRuntimeEvidenceError(
            f"[{_context_name(context)}_MODEL_NATIVE_RUNTIME_EVIDENCE_INVALID] "
            f"{key}: non-numeric"
        ) from exc
    if not math.isfinite(value):
        _fail(context, key, "non-finite")
    return value


def _finite_vector(
    evidence: Mapping[str, Any], key: str, size: int, *, context: str
) -> tuple[float, ...]:
    value = evidence.get(key)
    if isinstance(value, (str, bytes, Mapping)):
        _fail(context, key, f"expected finite vector[{size}]")
    try:
        raw = list(value)
    except TypeError as exc:
        raise ModelNativeRuntimeEvidenceError(
            f"[{_context_name(context)}_MODEL_NATIVE_RUNTIME_EVIDENCE_INVALID] "
            f"{key}: expected finite vector[{size}]"
        ) from exc
    if len(raw) != size:
        _fail(context, key, f"size={len(raw)} expected={size}")
    parsed: list[float] = []
    for item in raw:
        if isinstance(item, bool):
            _fail(context, key, "boolean element")
        try:
            scalar = float(item)
        except (TypeError, ValueError) as exc:
            raise ModelNativeRuntimeEvidenceError(
                f"[{_context_name(context)}_MODEL_NATIVE_RUNTIME_EVIDENCE_INVALID] "
                f"{key}: non-numeric element"
            ) from exc
        if not math.isfinite(scalar):
            _fail(context, key, "non-finite element")
        parsed.append(scalar)
    return tuple(parsed)


def _exact_integer(evidence: Mapping[str, Any], key: str, *, context: str) -> int:
    if key not in evidence or isinstance(evidence[key], bool):
        _fail(context, key, "must be an exact integer")
    try:
        return int(operator.index(evidence[key]))
    except TypeError as exc:
        raise ModelNativeRuntimeEvidenceError(
            f"[{_context_name(context)}_MODEL_NATIVE_RUNTIME_EVIDENCE_INVALID] "
            f"{key}: must be an exact integer"
        ) from exc


def _require_utc_timestamp(
    evidence: Mapping[str, Any], key: str, *, context: str
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


def _require_simplex(values: tuple[float, ...], field: str, *, context: str) -> None:
    if any(value < 0.0 for value in values) or not math.isclose(
        sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-7
    ):
        _fail(context, field, "not a probability simplex")


def _require_model_native_evidence(
    evidence: Mapping[str, Any], *, context: str, head_evidence: bool
) -> dict[str, Any]:
    if not isinstance(evidence, Mapping) or not evidence:
        _fail(context, "evidence", "missing or empty")
    validated = dict(evidence)
    retired = sorted(
        key
        for key in validated
        if any(fragment in str(key).lower() for fragment in RETIRED_RUNTIME_EVIDENCE_FRAGMENTS)
    )
    if retired:
        _fail(context, "evidence", f"retired fields={retired}")
    required = (
        MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_REQUIRED_FIELDS
        if head_evidence
        else MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS
    )
    optional = frozenset() if head_evidence else MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS
    missing = sorted(required - set(validated))
    unexpected = sorted(set(validated) - required - optional)
    if missing or unexpected:
        _fail(context, "evidence", f"exact schema mismatch missing={missing} unexpected={unexpected}")
    if validated.get("model_policy") != MODEL_NATIVE_RUNTIME_POLICY:
        _fail(context, "model_policy", "policy mismatch")
    if validated.get("runtime_evidence_schema_version") != MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION:
        _fail(context, "runtime_evidence_schema_version", "version mismatch")
    if head_evidence and validated.get("runtime_head_evidence_schema_version") != MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION:
        _fail(context, "runtime_head_evidence_schema_version", "version mismatch")

    session_id = _exact_integer(validated, "session_id", context=context)
    if session_id not in MODEL_NATIVE_CTX_CAT_DOMAINS["session_id"] or validated.get("session") != MODEL_NATIVE_SESSION_NAMES[session_id]:
        _fail(context, "session", "id/name mismatch")
    _finite_vector(validated, UNIFIED_EXIT_ENTRY_REPRESENTATION_KEY, UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM, context=context)
    _finite_vector(validated, "entry_q_joint_hidden", UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM, context=context)
    q_values = _finite_vector(validated, "entry_action_q_bps", len(ENTRY_FITTED_Q_ACTION_ORDER), context=context)
    winners = tuple(index for index, value in enumerate(q_values) if value == max(q_values))
    if len(winners) != 1:
        _fail(context, "entry_action_q_bps", "no unique raw-Q argmax")
    action_index = winners[0]
    if _exact_integer(validated, "model_direction_index", context=context) != action_index or validated.get("model_direction") != MODEL_DIRECTION_NAMES[action_index]:
        _fail(context, "model_direction", "raw-Q argmax mismatch")
    expected_side = action_index if action_index in MODEL_DIRECTION_TRADE_INDICES else None
    if expected_side is None:
        if validated.get("selected_side") is not None:
            _fail(context, "selected_side", "raw-Q action mismatch")
    elif _exact_integer(validated, "selected_side", context=context) != expected_side:
        _fail(context, "selected_side", "raw-Q action mismatch")
    sorted_q = sorted(q_values)
    expected_margin = sorted_q[-1] - sorted_q[-2]
    if not math.isclose(_finite_scalar(validated, "entry_action_q_margin_bps", context=context), expected_margin, rel_tol=1e-6, abs_tol=1e-7):
        _fail(context, "entry_action_q_margin_bps", "raw-Q margin mismatch")

    for field, width in (
        ("side_mae_bps", 2),
        ("trendline_event_logits", 4),
        ("dip_pred", MODEL_NATIVE_DIP_OUTPUT_DIM),
        ("forecast_pred", len(MODEL_NATIVE_FORECAST_TARGET_COLUMNS)),
        ("timing_pred", len(MODEL_NATIVE_TIMING_TARGET_COLUMNS)),
        ("tail_risk_pred", len(MODEL_NATIVE_TAIL_RISK_TARGET_COLUMNS)),
        ("vol_forecast_pred", len(MODEL_NATIVE_VOL_FORECAST_TARGET_COLUMNS)),
    ):
        _finite_vector(validated, field, width, context=context)
    if _finite_scalar(validated, "atr_bps", context=context) <= 0.0:
        _fail(context, "atr_bps", "must be positive")
    size_logit = _finite_scalar(validated, "position_size_logit", context=context)
    size_pred = _finite_scalar(validated, "position_size_pred", context=context)
    expected_size = 1.0 / (1.0 + math.exp(-max(-80.0, min(80.0, size_logit))))
    if not math.isclose(size_pred, expected_size, rel_tol=1e-6, abs_tol=1e-7):
        _fail(context, "position_size_pred", "sigmoid parity mismatch")

    names = validated.get("specialist_names")
    if not isinstance(names, (list, tuple)) or tuple(names) != tuple(MODEL_NATIVE_TRAINING_SPECIALISTS):
        _fail(context, "specialist_names", "specialist order mismatch")
    _require_simplex(_finite_vector(validated, "specialist_gate", len(MODEL_NATIVE_TRAINING_SPECIALISTS), context=context), "specialist_gate", context=context)
    _require_simplex(_finite_vector(validated, "tf_gate", ENTRY_MTF_CONTEXT_COUNT, context=context), "tf_gate", context=context)
    _require_simplex(
        _finite_vector(validated, "family_tf_cooperation_gate", ENTRY_MTF_CONTEXT_COUNT * len(MODEL_NATIVE_TRAINING_SPECIALISTS), context=context),
        "family_tf_cooperation_gate",
        context=context,
    )
    feature_gate = _finite_vector(validated, "family_tf_feature_gate", ENTRY_MTF_CONTEXT_COUNT * len(MULTI_TF_PER_BAR_FEATURES_V4), context=context)
    if any(value <= 0.0 or value >= 2.0 for value in feature_gate):
        _fail(context, "family_tf_feature_gate", "outside learned (0,2) contract")

    decision_ts = _require_utc_timestamp(validated, "decision_ts", context=context)
    if decision_ts.second != 0 or decision_ts.microsecond != 0 or decision_ts.minute % 5 != 0:
        _fail(context, "decision_ts", "must be an exact closed-M5 bar timestamp")
    observed_timing = set(validated) & MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS
    if observed_timing and observed_timing != set(MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS):
        _fail(context, "timing evidence", "must be absent or complete")
    if observed_timing:
        available = _require_utc_timestamp(validated, "decision_available_ts", context=context)
        cutoff = _require_utc_timestamp(validated, "context_cutoff_ts", context=context)
        if not math.isclose((available - decision_ts).total_seconds(), MODEL_NATIVE_DECISION_AVAILABILITY_LAG_SEC, rel_tol=0.0, abs_tol=1e-7):
            _fail(context, "decision_available_ts", "availability lag mismatch")
        if cutoff != decision_ts:
            _fail(context, "context_cutoff_ts", "must equal decision_ts")
        latency = _finite_scalar(validated, "entry_signal_latency_sec", context=context)
        if latency < 0.0 or latency > MODEL_NATIVE_MAX_ENTRY_SIGNAL_LATENCY_SEC:
            _fail(context, "entry_signal_latency_sec", "outside allowed range")
        if _exact_integer(validated, "context_age_m5_bars", context=context) != 0:
            _fail(context, "context_age_m5_bars", "must be zero")
    return validated


def require_model_native_runtime_evidence(
    evidence: Mapping[str, Any], context: str = "ENTRY_MODEL_NATIVE_RUNTIME"
) -> dict[str, Any]:
    return _require_model_native_evidence(evidence, context=context, head_evidence=False)


def require_model_native_runtime_head_evidence(
    evidence: Mapping[str, Any], context: str = "ENTRY_MODEL_NATIVE_RUNTIME_HEAD"
) -> dict[str, Any]:
    return _require_model_native_evidence(evidence, context=context, head_evidence=True)


def encode_model_native_runtime_head_evidence(
    evidence: Mapping[str, Any], *, context: str = "ENTRY_MODEL_NATIVE_RUNTIME_HEAD"
) -> tuple[str, str]:
    validated = require_model_native_runtime_head_evidence(evidence, context=context)
    payload = json.dumps(validated, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def decode_model_native_runtime_head_evidence(
    payload: Any, sha256: Any, *, context: str = "ENTRY_MODEL_NATIVE_RUNTIME_HEAD"
) -> dict[str, Any]:
    if not isinstance(payload, str) or not payload:
        _fail(context, "runtime_head_evidence_json", "missing")
    observed = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if str(sha256 or "").strip().lower() != observed:
        _fail(context, "runtime_head_evidence_sha256", "hash mismatch")
    try:
        decoded = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ModelNativeRuntimeEvidenceError(
            f"[{_context_name(context)}_MODEL_NATIVE_RUNTIME_EVIDENCE_INVALID] "
            "runtime_head_evidence_json: invalid JSON"
        ) from exc
    return require_model_native_runtime_head_evidence(decoded, context=context)


def _parse_external_utc_timestamp(value: Any, *, field: str, context: str) -> datetime:
    if isinstance(value, bool) or value is None:
        _fail(context, field, "missing or invalid")
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ModelNativeRuntimeEvidenceError(
            f"[{_context_name(context)}_MODEL_NATIVE_RUNTIME_EVIDENCE_INVALID] "
            f"{field}: invalid ISO timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        _fail(context, field, "must be timezone-aware UTC")
    return parsed


def require_model_native_entry_time(
    evidence: Mapping[str, Any], entry_time: Any, *, context: str = "ENTRY_MODEL_NATIVE_RUNTIME"
) -> datetime:
    validated = require_model_native_runtime_evidence(evidence, context=context)
    if not MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS.issubset(validated):
        _fail(context, "entry_time", "complete executable timing evidence is required")
    observed = _parse_external_utc_timestamp(entry_time, field="entry_time", context=context)
    available = _require_utc_timestamp(validated, "decision_available_ts", context=context)
    expected = (available + timedelta(seconds=float(validated["entry_signal_latency_sec"]))).replace(second=0, microsecond=0)
    if observed != expected:
        _fail(context, "entry_time", f"{observed.isoformat()} != model-derived minute {expected.isoformat()}")
    return observed


def require_model_native_fill_time(
    evidence: Mapping[str, Any], fill_time: Any, *, context: str = "ENTRY_MODEL_NATIVE_RUNTIME_FILL"
) -> datetime:
    validated = require_model_native_runtime_evidence(evidence, context=context)
    if not MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS.issubset(validated):
        _fail(context, "fill_time", "complete executable timing evidence is required")
    observed = _parse_external_utc_timestamp(fill_time, field="fill_time", context=context)
    available = _require_utc_timestamp(validated, "decision_available_ts", context=context)
    earliest = available + timedelta(seconds=float(validated["entry_signal_latency_sec"]))
    minute_start = earliest.replace(second=0, microsecond=0)
    minute_end = minute_start + timedelta(minutes=1)
    if observed < earliest or observed >= minute_end:
        _fail(context, "fill_time", f"outside [{earliest.isoformat()}, {minute_end.isoformat()})")
    return observed


def require_model_native_exit_replay_entry_time(
    head_evidence: Mapping[str, Any], entry_time: Any, *, context: str = "ENTRY_MODEL_NATIVE_EXIT_REPLAY"
) -> datetime:
    validated = require_model_native_runtime_head_evidence(head_evidence, context=context)
    observed = _parse_external_utc_timestamp(entry_time, field="entry_time", context=context)
    if observed.second != 0 or observed.microsecond != 0:
        _fail(context, "entry_time", "must be an exact UTC minute")
    decision = _require_utc_timestamp(validated, "decision_ts", context=context)
    expected = decision + timedelta(seconds=MODEL_NATIVE_DECISION_AVAILABILITY_LAG_SEC)
    if observed != expected:
        _fail(context, "entry_time", f"{observed.isoformat()} != exact T+5 replay fill {expected.isoformat()}")
    return observed


__all__ = [
    "MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION",
    "MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION",
    "MODEL_NATIVE_RUNTIME_POLICY",
    "MODEL_NATIVE_DECISION_AVAILABILITY_LAG_SEC",
    "MODEL_NATIVE_MAX_ENTRY_SIGNAL_LATENCY_SEC",
    "MODEL_NATIVE_SESSION_NAMES",
    "MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS",
    "MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS",
    "MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_REQUIRED_FIELDS",
    "RETIRED_RUNTIME_EVIDENCE_FRAGMENTS",
    "ModelNativeRuntimeEvidenceError",
    "require_model_native_entry_time",
    "require_model_native_fill_time",
    "require_model_native_exit_replay_entry_time",
    "require_model_native_runtime_head_evidence",
    "encode_model_native_runtime_head_evidence",
    "decode_model_native_runtime_head_evidence",
    "require_model_native_runtime_evidence",
]
