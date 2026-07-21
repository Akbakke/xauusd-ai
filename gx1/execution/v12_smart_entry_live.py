#!/usr/bin/env python3
"""LIVE model-native seq513 XAU Entry adapter.

Loads a contract-resolved, launch-admitted model-native v10_entry bundle through
the one-truth offline loader (gx1.models.entry_v10.entry_v10_bundle.
load_entry_v10_ctx_bundle — strict load + direction/path calibration installed
into the forward), forwards it per M5 close on the exact 513-signal state
(ModelNativeStateBuilder) + live multi-TF windows, and requires the PINNED
operating point read from PROJECT_STATE_artifacts.json to select
``model_direction_argmax``. The final calibrated model ``direction_logits`` are
the only LONG/SHORT/FLAT decision. No live session, threshold, utility, rail, or
side overlay may change that direction.

Serving architecture: the only live Entry load path is
load_entry_v10_ctx_bundle (calibration plus full active-head reconstruction),
which the offline evaluator
(evaluate_entry_candidate_selective_edge_v1._predict_bundle) also uses. This
adapter mirrors that forward exactly, so serve must equal the admitted evidence
path.

Direction SSOT:
    direction_probs = softmax(direction_logits)  # calibrated inside the model
    direction       = argmax(direction_probs)    # LONG=0, SHORT=1, FLAT=2

The model must also emit ``public_trade_flat_decision_logits`` ordered as
``[TRADE, FLAT]``. Its binary argmax must agree with the three-class direction
argmax or live inference fails closed. Auxiliary utility and rail heads are
journaled only as direct model diagnostics.

Exit-bound snapshot carries the model diagnostics required by the downstream
exit contract:
    direction_probs=[p_long,p_short,p_flat], path_quality=path_quality_pred,
    mfe_first_n=mfe_first_n_pred (raw), tradable_prob, bad_path_prob (carried,
    NOT consumed by the ACTIVE exit state), and the real learned
    tf_agreement_pred/path_quality_std diagnostics. ``position_size_logit`` is
    the sole learned sizing input and changes execution units only through the
    separately adopted, fail-closed sizing calibration. atr_bps is the live cv3 value
    at prediction bar T. hold_horizon_bars_pred is DELIBERATELY ABSENT -> TradeState
    keeps the -1 sentinel -> the HOLD_HORIZON_EXPIRED Strategy-F rule stays INERT.
    No substitute hold horizon is synthesized. A future admitted bundle must
    prove the corresponding exit mapping before that field may be added.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gx1.contracts.immutable_event_authority_v1 import (
    ImmutableEventAuthorityError,
    require_newest_immutable_event,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    require_model_native_signal_contract,
)
from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION,
    MODEL_NATIVE_RUNTIME_POLICY,
    RETIRED_RUNTIME_EVIDENCE_FRAGMENTS,
    require_model_native_runtime_evidence,
)
from gx1.contracts.entry_model_native_sizing_authority_v1 import (
    MODEL_NATIVE_SIZING_MODE_LEARNED,
    prepare_model_native_sizing_authority,
    require_model_native_sizing_authority_contract,
)
from gx1.contracts.entry_model_native_direction_evidence_fusion_v1 import (
    INPUTS as DIRECTION_EVIDENCE_FUSION_INPUTS,
)
from gx1.contracts.entry_model_native_state_v2 import (
    validate_state_contract_metadata_v2,
)
from gx1.contracts.model_native_serve_gate_v1 import (
    DIRECTION_POCKET_MAX_BAD_SIDE_RATE,
    DIRECTION_POCKET_MIN_MEAN_PROXY_PNL_BPS_EXCLUSIVE,
    DIRECTION_POCKET_MIN_SELECTED_ROWS,
    cross_gate_contract_failures,
    serve_gate_event_contract_failures,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
    require_model_direction_decision_contract,
    require_model_direction_operating_point,
)
from gx1.execution.v12_model_native_state_live import (
    SEQ_LEN_MODEL_NATIVE,
    SIGNAL_DIM_MODEL_NATIVE,
    ModelNativeStateContract,
    ModelNativeStateBuilder,
    build_multi_tf_from_cv3,
)

LOG = logging.getLogger("v12_smart_entry_live")

SESSION_NAMES = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}
MODEL_DIRECTION_NAMES = {0: "LONG", 1: "SHORT", 2: "FLAT"}
MODEL_DIRECTION_ACTIONS = {0: "TAKE_LONG_NOW", 1: "TAKE_SHORT_NOW", 2: "SKIP"}
MODEL_NATIVE_REQUIRED_SPECIALISTS = (
    "structure_swing_encoder",
    "smc_liquidity_encoder",
    "trend_ema_encoder",
    "vol_compression_encoder",
    "momentum_flow_encoder",
    "session_regime_encoder",
    "chart_geometry_encoder",
    "price_action_candle_encoder",
)
MODEL_NATIVE_FORWARD_PARITY_EVIDENCE_KEYS = tuple(dict.fromkeys((
    "raw_direction_logits",
    "path_quality",
    *(name for name, _width in DIRECTION_EVIDENCE_FUSION_INPUTS),
)))
MODEL_NATIVE_DECISION_DIAGNOSTIC_KEYS = tuple(dict.fromkeys((
    *MODEL_NATIVE_FORWARD_PARITY_EVIDENCE_KEYS,
    "path_quality_pred",
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
    "specialist_names",
    "specialist_gate",
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
    "geometry_channel_edge_pressure",
    "geometry_rising_support_rail_long_pressure",
    "geometry_rising_support_rail_short_trap_pressure",
    "geometry_falling_resistance_rail_short_pressure",
    "geometry_falling_resistance_rail_long_trap_pressure",
    "trendline_rail_logits",
    "trendline_rail_probs",
    "mtf_trend_evidence",
    "calibration_version",
    "direction_calibration_enabled",
    "direction_calibration_temperature",
    "direction_calibration_bias",
    "path_calibration_enabled",
    "path_calibration",
    "tf_agreement_logit",
    "tf_agreement_pred",
    "path_quality_log_var",
    "path_quality_std",
    "position_size_pred",
    "position_size_logit",
)))
MODEL_NATIVE_DECISION_HEAD_REQUIRED_FIELDS = frozenset(
    {
        "time",
        "direction_logits",
        "direction_probs",
        "model_direction_index",
        "model_direction",
        "public_trade_flat_decision_logits",
        "public_trade_flat_decision_probs",
        "public_trade_flat_decision_index",
        "public_trade_flat_decision",
        "p_long",
        "p_short",
        "p_flat",
        "p_trade",
        "p_flat_hier",
        "edge_score",
        "session_id",
        *MODEL_NATIVE_FORWARD_PARITY_EVIDENCE_KEYS,
        *MODEL_NATIVE_DECISION_DIAGNOSTIC_KEYS,
    }
)
MODEL_NATIVE_DECISION_HEAD_CONTEXT_FIELDS = frozenset(
    {
        "context_age_m5_bars",
        "context_cutoff_ts",
        "context_refresh_in_flight",
        "context_mtf_incremental",
    }
)

SMART_PARITY_GATE_MAX_AGE_HOURS = 18.0
SMART_PARITY_GATE_MAX_CUTOFF_LAG_HOURS = 18.0
SMART_DIRECTION_AUDIT_MAX_AGE_HOURS = 18.0

# LIVE direction requires the completed context snapshot to include the exact
# decision bar. The retired tail splice could not prove bit identity for M5,
# H4, and D1, so even one gap bar fails closed until the background full-history
# refresh lands.
SMART_CTX_MAX_STALENESS_M5 = 0

class SmartContextStaleError(RuntimeError):
    """Raised by predict_live_bar when the context snapshot is older than
    SMART_CTX_MAX_STALENESS_M5 bars behind the decision bar — the pipeline
    journals model-direction unavailability and retries on the next poll."""

    def __init__(self, age: int, cap: int, ctx_cutoff: pd.Timestamp, end_ts: pd.Timestamp):
        super().__init__(
            f"[SMART_ENTRY] context snapshot {age} M5 bars behind decision bar {end_ts} "
            f"(cutoff {ctx_cutoff}, cap {cap}) — refusing to decide on stale context"
        )
        self.age = int(age)
        self.cap = int(cap)
        self.ctx_cutoff = ctx_cutoff
        self.end_ts = end_ts


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_declared_gate_event(
    declaration: object,
    event_prefix: str,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Reload one launch-declared gate by exact path and content identity."""

    if not isinstance(declaration, dict) or set(declaration) != {
        "json_path",
        "sha256",
    }:
        raise RuntimeError(
            f"[SMART_GATE] {label} declaration must contain exact json_path/sha256"
        )
    raw_path = str(declaration.get("json_path") or "").strip()
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"[SMART_GATE] {label} path must be absolute: {raw_path!r}")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"[SMART_GATE] {label} path is not a regular file: {path}")
    resolved = path.resolve()
    if resolved != path or any("latest" in part.lower() for part in path.parts):
        raise RuntimeError(
            f"[SMART_GATE] {label} path is not canonical immutable identity: {path}"
        )
    expected_sha = str(declaration.get("sha256") or "").strip().lower()
    if len(expected_sha) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha
    ):
        raise RuntimeError(f"[SMART_GATE] {label} declaration lacks an exact SHA-256")
    try:
        require_newest_immutable_event(path, event_prefix)
    except ImmutableEventAuthorityError as exc:
        raise RuntimeError(f"[SMART_GATE] invalid {label} event authority: {exc}") from exc
    raw = path.read_bytes()
    observed_sha = hashlib.sha256(raw).hexdigest()
    if observed_sha != expected_sha:
        raise RuntimeError(
            f"[SMART_GATE] {label} sha256 mismatch: "
            f"declared={expected_sha} observed={observed_sha}"
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"[SMART_GATE] unreadable {label} event {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"[SMART_GATE] {label} event root is not an object: {path}")
    declared_self = Path(str(payload.get("json_path") or "")).expanduser()
    if not declared_self.is_absolute() or declared_self.resolve() != path:
        raise RuntimeError(f"[SMART_GATE] {label} event json_path is not an exact self-reference")
    if _sha256_file(path) != expected_sha:
        raise RuntimeError(f"[SMART_GATE] {label} changed while being validated")
    return payload, {"json_path": str(path), "sha256": expected_sha}


def _np1d(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().float().numpy().reshape(-1)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _softmax_np(values: np.ndarray | None) -> np.ndarray | None:
    if values is None or len(values) == 0:
        return None
    arr = values.astype(np.float64, copy=False)
    arr = arr - np.nanmax(arr)
    exp = np.exp(arr)
    denom = float(np.nansum(exp))
    if denom <= 0.0 or not np.isfinite(denom):
        return None
    return (exp / denom).astype(np.float32)


def _optional_finite_vector(
    value: Any,
    *,
    name: str,
    size: int | None,
    context: str,
) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = _np1d(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"[SMART_ENTRY] {context} diagnostic '{name}' is not a numeric vector"
        ) from exc
    if arr is None:
        return None
    if size is not None and arr.size != size:
        raise RuntimeError(
            f"[SMART_ENTRY] {context} diagnostic '{name}' must have exactly {size} values; "
            f"got shape={arr.shape} size={arr.size}"
        )
    if not bool(np.isfinite(arr).all()):
        raise RuntimeError(f"[SMART_ENTRY] {context} diagnostic '{name}' contains non-finite values")
    return arr.astype(np.float64, copy=False)


def _require_finite_vector(value: Any, *, name: str, size: int, context: str) -> np.ndarray:
    arr = _optional_finite_vector(
        value,
        name=name,
        size=size,
        context=context,
    )
    if arr is None:
        raise RuntimeError(f"[SMART_ENTRY] {context} missing required SSOT '{name}'")
    return arr


def _strict_softmax(logits: np.ndarray, *, name: str, context: str) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    exp = np.exp(shifted)
    denom = float(np.sum(exp))
    if not np.isfinite(denom) or denom <= 0.0:
        raise RuntimeError(f"[SMART_ENTRY] {context} SSOT '{name}' softmax is invalid")
    probs = exp / denom
    if not bool(np.isfinite(probs).all()):
        raise RuntimeError(f"[SMART_ENTRY] {context} SSOT '{name}' probabilities are non-finite")
    return probs


def _direction_ssot_from_logits(
    direction_logits_value: Any,
    public_trade_flat_logits_value: Any,
    *,
    context: str,
) -> dict[str, Any]:
    direction_logits = _require_finite_vector(
        direction_logits_value,
        name="direction_logits",
        size=3,
        context=context,
    )
    public_logits = _require_finite_vector(
        public_trade_flat_logits_value,
        name="public_trade_flat_decision_logits",
        size=2,
        context=context,
    )
    expected_public_logits = np.asarray(
        [max(float(direction_logits[0]), float(direction_logits[1])), float(direction_logits[2])],
        dtype=np.float64,
    )
    if not np.array_equal(public_logits, expected_public_logits):
        max_delta = float(np.max(np.abs(public_logits - expected_public_logits)))
        raise RuntimeError(
            "[SMART_ENTRY] public_trade_flat_decision_logits are not the canonical "
            "[max(final LONG/SHORT), final FLAT] surface; "
            f"max_abs_delta={max_delta:.9g}"
        )
    direction_probs = _strict_softmax(
        direction_logits,
        name="direction_logits",
        context=context,
    )
    public_probs = _strict_softmax(
        public_logits,
        name="public_trade_flat_decision_logits",
        context=context,
    )
    direction_index = int(np.argmax(direction_probs))
    public_index = int(np.argmax(public_probs))  # TRADE=0, FLAT=1
    expected_public_index = 1 if direction_index == 2 else 0
    if public_index != expected_public_index:
        raise RuntimeError(
            "[SMART_ENTRY] model direction SSOT mismatch: "
            f"direction={MODEL_DIRECTION_NAMES[direction_index]}({direction_index}) "
            f"public_trade_flat={'FLAT' if public_index == 1 else 'TRADE'}({public_index})"
        )
    return {
        "direction_logits": direction_logits,
        "direction_probs": direction_probs,
        "model_direction_index": direction_index,
        "model_direction": MODEL_DIRECTION_NAMES[direction_index],
        "public_trade_flat_decision_logits": public_logits,
        "public_trade_flat_decision_probs": public_probs,
        "public_trade_flat_decision_index": public_index,
        "public_trade_flat_decision": "FLAT" if public_index == 1 else "TRADE",
    }


def _validate_reported_direction_ssot(head_out: dict[str, Any]) -> dict[str, Any]:
    ssot = _direction_ssot_from_logits(
        head_out.get("direction_logits"),
        head_out.get("public_trade_flat_decision_logits"),
        context="decision",
    )
    reported_direction_probs = _require_finite_vector(
        head_out.get("direction_probs"),
        name="direction_probs",
        size=3,
        context="decision",
    )
    if not np.allclose(reported_direction_probs, ssot["direction_probs"], rtol=1e-6, atol=1e-7):
        raise RuntimeError("[SMART_ENTRY] decision SSOT direction_probs do not match direction_logits")
    reported_public_probs = _require_finite_vector(
        head_out.get("public_trade_flat_decision_probs"),
        name="public_trade_flat_decision_probs",
        size=2,
        context="decision",
    )
    if not np.allclose(
        reported_public_probs,
        ssot["public_trade_flat_decision_probs"],
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError(
            "[SMART_ENTRY] decision SSOT public_trade_flat_decision_probs do not match "
            "public_trade_flat_decision_logits"
        )
    reported_scalars = _require_finite_vector(
        [head_out.get("p_long"), head_out.get("p_short"), head_out.get("p_flat")],
        name="p_long/p_short/p_flat",
        size=3,
        context="decision",
    )
    if not np.allclose(reported_scalars, ssot["direction_probs"], rtol=1e-6, atol=1e-7):
        raise RuntimeError("[SMART_ENTRY] decision SSOT p_long/p_short/p_flat do not match direction_logits")
    reported_public_scalars = _require_finite_vector(
        [head_out.get("p_trade"), head_out.get("p_flat_hier")],
        name="p_trade/p_flat_hier",
        size=2,
        context="decision",
    )
    if not np.allclose(
        reported_public_scalars,
        ssot["public_trade_flat_decision_probs"],
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError(
            "[SMART_ENTRY] decision SSOT p_trade/p_flat_hier do not match "
            "direction_logits"
        )
    for index_key, expected in (
        ("model_direction_index", ssot["model_direction_index"]),
        (
            "public_trade_flat_decision_index",
            ssot["public_trade_flat_decision_index"],
        ),
    ):
        observed = head_out.get(index_key)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, np.integer))
            or int(observed) != int(expected)
        ):
            raise RuntimeError(
                f"[SMART_ENTRY] decision SSOT {index_key} does not match logits"
            )
    for name_key, expected in (
        ("model_direction", ssot["model_direction"]),
        (
            "public_trade_flat_decision",
            ssot["public_trade_flat_decision"],
        ),
    ):
        if head_out.get(name_key) != expected:
            raise RuntimeError(
                f"[SMART_ENTRY] decision SSOT {name_key} does not match logits"
            )
    reported_edge = _require_finite_vector(
        [head_out.get("edge_score")],
        name="edge_score",
        size=1,
        context="decision",
    )[0]
    expected_edge = max(ssot["direction_probs"][:2]) - ssot["direction_probs"][2]
    if not np.isclose(reported_edge, expected_edge, rtol=1e-6, atol=1e-7):
        raise RuntimeError(
            "[SMART_ENTRY] decision SSOT edge_score does not match direction_logits"
        )
    return ssot


def _sigmoid_float(value: float) -> float:
    value = float(np.clip(value, -80.0, 80.0))
    return float(1.0 / (1.0 + np.exp(-value)))


def _required_feature_value(row: np.ndarray, names: list[str], name: str) -> float:
    if name not in names:
        raise RuntimeError(f"[SMART_ENTRY] required model-native evidence feature missing: {name}")
    idx = int(names.index(name))
    if idx < 0 or idx >= len(row):
        raise RuntimeError(
            f"[SMART_ENTRY] evidence feature index out of bounds: {name} index={idx} width={len(row)}"
        )
    value = float(row[idx])
    if not np.isfinite(value):
        raise RuntimeError(f"[SMART_ENTRY] evidence feature is non-finite: {name}={value!r}")
    return value


def _validate_model_native_diagnostics(
    head_out: dict[str, Any],
    diagnostic_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Validate the complete learned evidence surface before action emission."""

    def vector(key: str, size: int) -> np.ndarray:
        return _require_finite_vector(
            head_out.get(key),
            name=key,
            size=size,
            context="decision diagnostic",
        )

    def scalar(key: str) -> float:
        return float(vector(key, 1)[0])

    vector("raw_direction_logits", 3)
    for fusion_name, fusion_width in DIRECTION_EVIDENCE_FUSION_INPUTS:
        vector(fusion_name, fusion_width)
    scalar("path_quality")
    scalar("bad_path_logit_raw")

    # Core path/tradability regressions are mandatory learned evidence even
    # though none of them may act as a post-model direction veto.
    scalar("path_quality_pred")
    scalar("mfe_first_n_pred")
    bad_path_logit = scalar("bad_path_logit")
    if not np.isclose(
        scalar("bad_path_prob"),
        _sigmoid_float(bad_path_logit),
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError("[SMART_ENTRY] bad_path_prob does not match bad_path_logit")
    tradable_logit = scalar("tradable_logit")
    if not np.isclose(
        scalar("tradable_prob"),
        _sigmoid_float(tradable_logit),
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError("[SMART_ENTRY] tradable_prob does not match tradable_logit")

    side_logits = vector("side_logits", 2)
    side_probs = vector("side_probs", 2)
    expected_side_probs = _softmax_np(side_logits)
    if not np.allclose(side_probs, expected_side_probs, rtol=1e-6, atol=1e-7):
        raise RuntimeError("[SMART_ENTRY] side_probs do not match side_logits")
    conditional = np.asarray(
        [scalar("p_long_given_trade"), scalar("p_short_given_trade")],
        dtype=np.float64,
    )
    if not np.allclose(conditional, side_probs, rtol=1e-6, atol=1e-7):
        raise RuntimeError("[SMART_ENTRY] conditional side probabilities do not match side_logits")
    vector("side_utility", 2)
    side_bad_logits = vector("side_bad_path_logit", 2)
    side_bad_probs = np.asarray(
        [scalar("long_bad_path_prob"), scalar("short_bad_path_prob")],
        dtype=np.float64,
    )
    expected_bad = np.asarray([_sigmoid_float(value) for value in side_bad_logits])
    if not np.allclose(side_bad_probs, expected_bad, rtol=1e-6, atol=1e-7):
        raise RuntimeError("[SMART_ENTRY] side bad-path probabilities do not match logits")
    side_validity_logits = vector("side_validity_logit", 2)
    side_validity_probs = np.asarray(
        [scalar("long_validity_prob"), scalar("short_validity_prob")],
        dtype=np.float64,
    )
    expected_validity = np.asarray(
        [_sigmoid_float(value) for value in side_validity_logits]
    )
    if not np.allclose(side_validity_probs, expected_validity, rtol=1e-6, atol=1e-7):
        raise RuntimeError("[SMART_ENTRY] side validity probabilities do not match logits")
    vector("side_mae", 2)

    mtf_logits = vector("mtf_dir_logits", 3)
    mtf_probs = vector("mtf_dir_probs", 3)
    if not np.allclose(mtf_probs, _softmax_np(mtf_logits), rtol=1e-6, atol=1e-7):
        raise RuntimeError("[SMART_ENTRY] mtf_dir_probs do not match mtf_dir_logits")
    for key in (
        "geometry_channel_edge_pressure",
        "geometry_rising_support_rail_long_pressure",
        "geometry_rising_support_rail_short_trap_pressure",
        "geometry_falling_resistance_rail_short_pressure",
        "geometry_falling_resistance_rail_long_trap_pressure",
        "mtf_trend_evidence",
    ):
        scalar(key)
    rail_logits = vector("trendline_rail_logits", 6)
    rail_probs = vector("trendline_rail_probs", 6)
    expected_rail = np.asarray([_sigmoid_float(value) for value in rail_logits])
    if not np.allclose(rail_probs, expected_rail, rtol=1e-6, atol=1e-7):
        raise RuntimeError("[SMART_ENTRY] trendline_rail_probs do not match trendline_rail_logits")

    clean_edge_logit = scalar("clean_edge_logit")
    survival_logit = scalar("survival_logit")
    if not np.isclose(
        scalar("clean_edge_prob"), _sigmoid_float(clean_edge_logit), rtol=1e-6, atol=1e-7
    ):
        raise RuntimeError("[SMART_ENTRY] clean_edge_prob does not match clean_edge_logit")
    if not np.isclose(
        scalar("survival_prob"), _sigmoid_float(survival_logit), rtol=1e-6, atol=1e-7
    ):
        raise RuntimeError("[SMART_ENTRY] survival_prob does not match survival_logit")
    vector("dip_pred", 18)
    vector("forecast_pred", 4)
    vector("timing_pred", 12)
    vector("tail_risk_pred", 6)
    vector("vol_forecast_pred", 3)

    specialist_names = head_out.get("specialist_names")
    observed_specialist_names = (
        list(specialist_names) if isinstance(specialist_names, (list, tuple)) else []
    )
    if observed_specialist_names != list(MODEL_NATIVE_REQUIRED_SPECIALISTS):
        raise RuntimeError("[SMART_ENTRY] specialist_names contract mismatch")
    specialist_gate = vector("specialist_gate", len(MODEL_NATIVE_REQUIRED_SPECIALISTS))
    if bool((specialist_gate < 0.0).any()) or not np.isclose(
        float(specialist_gate.sum()), 1.0, rtol=1e-6, atol=1e-7
    ):
        raise RuntimeError("[SMART_ENTRY] specialist_gate is not a probability simplex")

    calibration_version = head_out.get("calibration_version")
    if not isinstance(calibration_version, str) or not calibration_version.strip():
        raise RuntimeError("[SMART_ENTRY] direction calibration version is missing")
    if head_out.get("direction_calibration_enabled") is not True:
        raise RuntimeError("[SMART_ENTRY] direction calibration must be enabled")
    direction_temperature = scalar("direction_calibration_temperature")
    if direction_temperature <= 0.0:
        raise RuntimeError("[SMART_ENTRY] direction calibration temperature must be positive")
    direction_bias = vector("direction_calibration_bias", 3)
    expected_direction_logits = (
        vector("raw_direction_logits", 3) / direction_temperature + direction_bias
    )
    if not np.allclose(
        vector("direction_logits", 3),
        expected_direction_logits,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise RuntimeError(
            "[SMART_ENTRY] final direction logits do not match raw/temperature+bias"
        )
    if head_out.get("path_calibration_enabled") is not True:
        raise RuntimeError("[SMART_ENTRY] path calibration must be enabled")
    path_calibration = head_out.get("path_calibration")
    if not isinstance(path_calibration, dict) or path_calibration.get("enabled") is not True:
        raise RuntimeError("[SMART_ENTRY] path calibration contract is missing or disabled")
    for key in (
        "version",
        "path_quality_scale",
        "path_quality_shift",
        "bad_path_temperature",
        "bad_path_bias",
    ):
        if key not in path_calibration:
            raise RuntimeError(f"[SMART_ENTRY] path calibration field missing: {key}")
    path_cal_values = _require_finite_vector(
        [
            path_calibration["path_quality_scale"],
            path_calibration["path_quality_shift"],
            path_calibration["bad_path_temperature"],
            path_calibration["bad_path_bias"],
        ],
        name="path_calibration values",
        size=4,
        context="decision diagnostic",
    )
    if float(path_cal_values[0]) <= 0.0 or float(path_cal_values[2]) <= 0.0:
        raise RuntimeError("[SMART_ENTRY] path calibration scales must be positive")
    if not isinstance(path_calibration["version"], str) or not path_calibration["version"].strip():
        raise RuntimeError("[SMART_ENTRY] path calibration version is missing")

    tf_logit = scalar("tf_agreement_logit")
    if not np.isclose(
        scalar("tf_agreement_pred"), _sigmoid_float(tf_logit), rtol=1e-6, atol=1e-7
    ):
        raise RuntimeError("[SMART_ENTRY] tf_agreement_pred does not match tf_agreement_logit")
    path_log_var = scalar("path_quality_log_var")
    expected_std = float(np.exp(0.5 * path_log_var))
    if not np.isfinite(expected_std) or not np.isclose(
        scalar("path_quality_std"), expected_std, rtol=1e-6, atol=1e-7
    ):
        raise RuntimeError("[SMART_ENTRY] path_quality_std does not match path_quality_log_var")
    size_logit = scalar("position_size_logit")
    if not np.isclose(
        scalar("position_size_pred"), _sigmoid_float(size_logit), rtol=1e-6, atol=1e-7
    ):
        raise RuntimeError("[SMART_ENTRY] position_size_pred does not match position_size_logit")

    return {key: head_out[key] for key in diagnostic_keys}


@dataclass(frozen=True)
class SmartCtxSnapshot:
    """One COMPLETED smart-context build — swapped in as a single atomic reference
    (the loader's 2026-06-01 async-refresh pattern) so a decision that grabbed the
    snapshot can never observe a half-refreshed context. Immutable by convention:
    the background refresh builds a NEW snapshot and replaces the reference."""
    multi_tf: dict
    frame_overrides: pd.DataFrame       # bucket ctx_cat + HTF/REGIME_V4 override cols
    cv3_cutoff: pd.Timestamp
    built_utc: pd.Timestamp
    build_seconds: float


def _smart_gate_git_state() -> tuple[str, bool]:
    repo = Path(__file__).resolve().parents[2]
    commit = "unknown"
    dirty = True
    try:
        commit_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if commit_proc.returncode == 0:
            commit = commit_proc.stdout.strip()
        dirty_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if dirty_proc.returncode == 0:
            dirty = bool(dirty_proc.stdout.strip())
    except Exception:
        dirty = True
    return commit, dirty


def assert_smart_serving_gate() -> dict:
    """ONE-TRUTH launch gate for the smart serving path (launcher + runner):
    (1) the TRAIN==SERVE parity gate artifact must be decision=PASS and must
        have been produced for the CONTRACT-ACTIVE v10_entry bundle;
    (2) the directional live-like pocket audit must be decision=PASS for the
        CONTRACT-ACTIVE v10_entry bundle;
    (3) the contract must be the exact model-native seq513 candidate with a complete
        operating_point.
    Raises RuntimeError on any violation; returns the gate report on success.
    """
    from gx1_guards.artifacts import load_decision_entry
    entry = load_decision_entry("v10_entry")
    launch_state = entry.get("xau_direction_launch_state")
    if not isinstance(launch_state, dict):
        raise RuntimeError(
            "[SMART_GATE] artifact guard did not return the validated XAU direction launch state"
        )
    declared_evidence = launch_state.get("serve_gate_evidence")
    if not isinstance(declared_evidence, dict) or set(declared_evidence) != {
        "model_native_serve_parity",
        "model_native_direction_pocket_audit",
    }:
        raise RuntimeError(
            "[SMART_GATE] XAU direction launch state lacks exact serve_gate_evidence"
        )
    rep, parity_authority = _load_declared_gate_event(
        declared_evidence["model_native_serve_parity"],
        "MODEL_NATIVE_SERVE_PARITY",
        label="TRAIN==SERVE parity",
    )
    direction_audit, direction_authority = _load_declared_gate_event(
        declared_evidence["model_native_direction_pocket_audit"],
        "MODEL_NATIVE_DIRECTION_POCKET_AUDIT",
        label="direction pocket audit",
    )
    problems: list[str] = []
    problems.extend(
        serve_gate_event_contract_failures(
            rep,
            evidence_name="model_native_serve_parity",
        )
    )
    problems.extend(
        serve_gate_event_contract_failures(
            direction_audit,
            evidence_name="model_native_direction_pocket_audit",
        )
    )
    problems.extend(cross_gate_contract_failures(rep, direction_audit))
    for evidence_name, observed in (
        ("model_native_serve_parity", parity_authority),
        ("model_native_direction_pocket_audit", direction_authority),
    ):
        declared = declared_evidence[evidence_name]
        if declared != observed:
            problems.append(
                f"XAU direction launch {evidence_name} binding mismatch: "
                f"declared={declared!r} observed={observed!r}"
            )
    if rep.get("decision") != "PASS":
        problems.append(f"parity decision={rep.get('decision')!r} failures={list(rep.get('failures') or [])[:3]}")
    current_commit, worktree_dirty = _smart_gate_git_state()
    parity_commit = str(rep.get("git_commit") or "").strip()
    if not parity_commit:
        problems.append("parity report missing git_commit")
    elif current_commit != parity_commit:
        problems.append(f"parity git_commit {parity_commit} != current git_commit {current_commit}")
    if worktree_dirty:
        problems.append("smart serving git worktree is dirty; rerun parity on the exact source before launch")
    now_utc = pd.Timestamp.now(tz="UTC")
    created_utc = pd.to_datetime(rep.get("created_utc"), utc=True, errors="coerce")
    if pd.isna(created_utc):
        problems.append(f"parity created_utc invalid/missing: {rep.get('created_utc')!r}")
    elif SMART_PARITY_GATE_MAX_AGE_HOURS > 0:
        age_hours = (now_utc - created_utc).total_seconds() / 3600.0
        if age_hours > SMART_PARITY_GATE_MAX_AGE_HOURS:
            problems.append(
                f"parity report stale: age_hours={age_hours:.2f} "
                f"> cap={SMART_PARITY_GATE_MAX_AGE_HOURS:.2f}"
            )
    cutoff_utc = pd.to_datetime(rep.get("live_prebuilt_cutoff"), utc=True, errors="coerce")
    if pd.isna(cutoff_utc):
        problems.append(f"parity live_prebuilt_cutoff invalid/missing: {rep.get('live_prebuilt_cutoff')!r}")
    elif SMART_PARITY_GATE_MAX_CUTOFF_LAG_HOURS > 0:
        cutoff_lag_hours = (now_utc - cutoff_utc).total_seconds() / 3600.0
        if cutoff_lag_hours > SMART_PARITY_GATE_MAX_CUTOFF_LAG_HOURS:
            problems.append(
                f"parity prebuilt cutoff stale: cutoff_lag_hours={cutoff_lag_hours:.2f} "
                f"> cap={SMART_PARITY_GATE_MAX_CUTOFF_LAG_HOURS:.2f}"
            )
    if str(rep.get("bundle_dir")) != str(entry["path"]):
        problems.append(f"parity bundle {rep.get('bundle_dir')} != contract-ACTIVE {entry['path']}")
    bundle_meta_path = Path(str(entry["path"])) / "bundle_metadata.json"
    bundle_state_contract = {}
    if not bundle_meta_path.is_file():
        problems.append(f"contract-ACTIVE bundle metadata missing: {bundle_meta_path}")
    else:
        try:
            bundle_meta = json.loads(bundle_meta_path.read_text(encoding="utf-8"))
            require_model_direction_decision_contract(
                bundle_meta,
                context="[SMART_GATE] contract-ACTIVE bundle",
            )
            raw_contract = bundle_meta.get("model_native_state_contract")
            bundle_state_contract = raw_contract if isinstance(raw_contract, dict) else {}
        except Exception as exc:
            problems.append(f"contract-ACTIVE bundle metadata unreadable: {bundle_meta_path}: {exc}")
    parity_state_contract = rep.get("model_native_state_contract")
    if not isinstance(parity_state_contract, dict):
        problems.append("parity report missing model_native_state_contract")
    else:
        for label, candidate in (
            ("bundle", bundle_state_contract),
            ("parity", parity_state_contract),
        ):
            try:
                validate_state_contract_metadata_v2(candidate, require_artifact=True)
            except Exception as exc:
                problems.append(f"{label} model_native_state_contract v2 invalid: {exc}")
        for key in (
            "schema_version",
            "feature_history_start_utc",
            "rank_fit_start_utc",
            "rank_fit_end_utc",
            "rank_reference_npz",
            "rank_reference_npz_sha256",
            "rank_reference_sidecar_sha256",
            "rank_reference_schema_version",
            "normalization_fit_scope",
            "rank_transform",
            "feature_history_mode",
            "split_reset_allowed",
            "post_fit_rows_in_rank_reference",
            "runtime_rule_free",
        ):
            parity_value = parity_state_contract.get(key)
            bundle_value = bundle_state_contract.get(key)
            if parity_value is None:
                problems.append(f"parity model_native_state_contract missing {key}")
            if bundle_value is not None and parity_value is not None and parity_value != bundle_value:
                problems.append(
                    f"parity model_native_state_contract.{key} {parity_value} != bundle metadata {bundle_value}"
                )
            if parity_value is not None and bundle_value is None:
                problems.append(f"bundle model_native_state_contract missing {key}")
    parity_dataset = str(rep.get("dataset_dir") or "").strip()
    parity_dataset_low = parity_dataset.lower()
    if not parity_dataset:
        problems.append("parity report missing dataset_dir")
    elif "xau" not in parity_dataset_low:
        problems.append(f"parity dataset_dir must be XAU-only, got {parity_dataset}")
    for stale_marker in ("utilityrepair", "20260710", "smart_candidate_20260630", "julyext"):
        if stale_marker in parity_dataset_low:
            problems.append(
                f"parity dataset_dir references stale XAU repair marker {stale_marker!r}: {parity_dataset}"
            )
    if direction_audit.get("decision") != "PASS":
        problems.append(
            f"direction pocket audit decision={direction_audit.get('decision')!r} "
            f"failures={list(direction_audit.get('failures') or [])[:3]}"
        )
    direction_created_utc = pd.to_datetime(direction_audit.get("created_utc"), utc=True, errors="coerce")
    if pd.isna(direction_created_utc):
        problems.append(f"direction pocket audit created_utc invalid/missing: {direction_audit.get('created_utc')!r}")
    else:
        if SMART_DIRECTION_AUDIT_MAX_AGE_HOURS > 0:
            direction_age_hours = (now_utc - direction_created_utc).total_seconds() / 3600.0
            if direction_age_hours > SMART_DIRECTION_AUDIT_MAX_AGE_HOURS:
                problems.append(
                    f"direction pocket audit stale: age_hours={direction_age_hours:.2f} "
                    f"> cap={SMART_DIRECTION_AUDIT_MAX_AGE_HOURS:.2f}"
                )
        if direction_audit.get("required_selection_score_mode") != MODEL_DIRECTION_SELECTION_MODE:
            problems.append(
                "direction pocket audit required_selection_score_mode must be exactly "
                f"{MODEL_DIRECTION_SELECTION_MODE!r}"
            )
        observed_modes_raw = direction_audit.get("observed_selection_score_modes")
        observed_modes = (
            list(observed_modes_raw)
            if isinstance(observed_modes_raw, list)
            else []
        )
        if not observed_modes or any(mode != MODEL_DIRECTION_SELECTION_MODE for mode in observed_modes):
            problems.append(f"direction pocket audit observed_selection_score_modes invalid: {observed_modes_raw!r}")
        for audit_field in ("predictions_parquet", "dataset_dir", "dataset_parquet"):
            audit_path = str(direction_audit.get(audit_field) or "").strip()
            audit_low = audit_path.lower()
            if not audit_path:
                problems.append(f"direction pocket audit missing {audit_field}")
            elif "xau" not in audit_low:
                problems.append(f"direction pocket audit {audit_field} must be XAU-only, got {audit_path}")
            for stale_marker in ("utilityrepair", "20260710", "smart_candidate_20260630", "julyext"):
                if stale_marker in audit_low:
                    problems.append(
                        f"direction pocket audit {audit_field} references stale XAU repair marker "
                        f"{stale_marker!r}: {audit_path}"
                    )
        required_direction_repair_pockets = {
            "rising_channel_support_touch",
            "support_retest_continuation",
            "rising_channel_support_continuation",
            "countertrend_short_trap",
            "short_high_mae_low_mfe_early_failure",
            "falling_channel_resistance_touch",
            "resistance_retest_continuation",
            "falling_channel_resistance_continuation",
            "countertrend_long_trap",
            "long_high_mae_low_mfe_early_failure",
        }
        audit_pockets = direction_audit.get("pockets")
        if not isinstance(audit_pockets, dict):
            problems.append("direction pocket audit lacks pockets dict")
        else:
            missing_pockets = sorted(required_direction_repair_pockets - set(audit_pockets))
            if missing_pockets:
                problems.append(
                    "direction pocket audit lacks required XAU direction-repair pockets: "
                    + ",".join(missing_pockets)
                )
            max_bad_side_rate = DIRECTION_POCKET_MAX_BAD_SIDE_RATE
            min_selected_rows = DIRECTION_POCKET_MIN_SELECTED_ROWS
            short_bad_pockets = {
                "rising_channel_support_touch",
                "support_retest_continuation",
                "rising_channel_support_continuation",
                "countertrend_short_trap",
                "short_high_mae_low_mfe_early_failure",
            }
            long_bad_pockets = {
                "falling_channel_resistance_touch",
                "resistance_retest_continuation",
                "falling_channel_resistance_continuation",
                "countertrend_long_trap",
                "long_high_mae_low_mfe_early_failure",
            }
            utility_pockets = (
                required_direction_repair_pockets
                - {"short_high_mae_low_mfe_early_failure", "long_high_mae_low_mfe_early_failure"}
            )
            for pocket_name in sorted(required_direction_repair_pockets & set(audit_pockets)):
                row = audit_pockets.get(pocket_name)
                if not isinstance(row, dict):
                    problems.append(f"direction pocket audit {pocket_name} is not a metrics dict")
                    continue
                try:
                    rows = int(row.get("rows"))
                    selected_rows = int(row.get("selected_rows"))
                except Exception:
                    problems.append(f"direction pocket audit {pocket_name} lacks integer rows/selected_rows")
                    continue
                if rows < min_selected_rows:
                    problems.append(
                        f"direction pocket audit {pocket_name} rows={rows} < required {min_selected_rows}"
                    )
                if selected_rows < min_selected_rows:
                    problems.append(
                        f"direction pocket audit {pocket_name} selected_rows={selected_rows} < required {min_selected_rows}"
                    )
                if pocket_name in short_bad_pockets:
                    short_rate = float(row.get("selected_side_short_rate", 1.0))
                    if short_rate > max_bad_side_rate:
                        problems.append(
                            f"direction pocket audit {pocket_name} selected SHORT rate {short_rate:.3f} "
                            f"> required {max_bad_side_rate:.3f}"
                        )
                if pocket_name in long_bad_pockets:
                    long_rate = float(row.get("selected_side_long_rate", 1.0))
                    if long_rate > max_bad_side_rate:
                        problems.append(
                            f"direction pocket audit {pocket_name} selected LONG rate {long_rate:.3f} "
                            f"> required {max_bad_side_rate:.3f}"
                        )
                if pocket_name in utility_pockets:
                    mean_pnl = row.get("selected_mean_proxy_pnl_bps")
                    if (
                        mean_pnl is None
                        or float(mean_pnl)
                        <= DIRECTION_POCKET_MIN_MEAN_PROXY_PNL_BPS_EXCLUSIVE
                    ):
                        problems.append(
                            f"direction pocket audit {pocket_name} selected_mean_proxy_pnl_bps={mean_pnl} "
                            "> required 0"
                        )
        if str(direction_audit.get("bundle_dir")) != str(entry["path"]):
            problems.append(
                f"direction pocket audit bundle {direction_audit.get('bundle_dir')} "
                f"!= contract-ACTIVE {entry['path']}"
            )
    if str(entry.get("contract_mode")) != MODEL_NATIVE_CONTRACT_MODE:
        problems.append(f"contract_mode={entry.get('contract_mode')!r}")
    op = entry.get("operating_point")
    try:
        require_model_direction_operating_point(
            op,
            context="v10_entry",
        )
    except RuntimeError as exc:
        problems.append(str(exc))
    if SMART_CTX_MAX_STALENESS_M5 != 0:
        problems.append(
            "GX1_SMART_CTX_MAX_STALENESS_M5 must be 0 for model-direction XAU repair serving; "
            f"got {SMART_CTX_MAX_STALENESS_M5}"
        )
    if problems:
        raise RuntimeError("[SMART_GATE] LAUNCH BLOCKED: " + " | ".join(problems))
    return rep


@dataclass
class SmartEntryLiveInference:
    bundle_dir: Path
    operating_point: dict[str, Any]
    device: str = "cpu"
    _model: Any = field(default=None)
    _meta: dict = field(default_factory=dict)
    _sizing_authority: dict = field(default_factory=dict, repr=False)
    _builder: ModelNativeStateBuilder | None = field(default=None)
    _state_contract: ModelNativeStateContract | None = field(default=None)
    _per_tf_seq_lens: dict[str, int] = field(default_factory=dict)
    _multi_tf_shift: dict = field(default_factory=dict, repr=False)
    _multi_tf_target_availability_shift: pd.Timedelta = field(
        default_factory=lambda: pd.Timedelta(minutes=5),
        repr=False,
    )
    # LAST COMPLETED context snapshot (one atomic reference — loader async pattern)
    # + the in-flight background refresh thread (serving-wave gap 3). The per-M1
    # EXIT path never touches either — no lock exists to starve it.
    _ctx: SmartCtxSnapshot | None = field(default=None, repr=False)
    _ctx_refresh_thread: threading.Thread | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.operating_point = require_model_direction_operating_point(
            self.operating_point,
            context="[SMART_ENTRY]",
        )

    # ── loading ──────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, bundle_dir: Path | None = None, device: str = "cpu") -> "SmartEntryLiveInference":
        from gx1_guards.artifacts import load_decision_entry
        entry = load_decision_entry("v10_entry")
        contract_bundle = Path(entry["path"])
        if bundle_dir is None:
            bundle_dir = contract_bundle
        else:
            bundle_dir = Path(bundle_dir)
            if bundle_dir.resolve() != contract_bundle.resolve():
                raise RuntimeError(
                    f"[SMART_ENTRY] explicit bundle_dir {bundle_dir} != contract-ACTIVE "
                    f"{contract_bundle} — rule 8: serve resolves ONLY through the contract"
                )
        mode = str(entry["contract_mode"])
        if mode != MODEL_NATIVE_CONTRACT_MODE:
            raise RuntimeError(
                f"[SMART_ENTRY] contract v10_entry.contract_mode={mode!r} — this adapter "
                f"serves {MODEL_NATIVE_CONTRACT_MODE} only"
            )
        op = entry["operating_point"]
        op = require_model_direction_operating_point(
            op,
            context="[SMART_ENTRY] contract v10_entry",
        )

        from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
        bundle = load_entry_v10_ctx_bundle(bundle_dir=bundle_dir, device=device)
        model = bundle.transformer_model
        model.eval()
        meta = dict(bundle.metadata)
        require_model_direction_decision_contract(
            meta,
            context="[SMART_ENTRY] contract-ACTIVE bundle",
        )
        launch_state = entry.get("xau_direction_launch_state")
        if not isinstance(launch_state, dict):
            raise RuntimeError(
                "[SMART_ENTRY] artifact guard did not return validated launch state"
            )
        sizing_authority = require_model_native_sizing_authority_contract(
            launch_state.get("sizing_authority_contract"),
            context="[SMART_ENTRY] external sizing adoption",
            required_mode=MODEL_NATIVE_SIZING_MODE_LEARNED,
        )
        prepare_model_native_sizing_authority(
            sizing_authority,
            context="[SMART_ENTRY] startup sizing adoption",
        )
        if str(meta["direction_logit_mode"]) != MODEL_NATIVE_DIRECTION_LOGIT_MODE:
            raise RuntimeError(
                "[SMART_ENTRY] bundle direction_logit_mode must be model_native; got "
                f"{meta['direction_logit_mode']!r}"
            )
        signal_contract = meta["model_native_signal_contract"]
        require_model_native_signal_contract(
            signal_contract,
            context="SMART_ENTRY_BUNDLE",
        )
        state_contract = ModelNativeStateContract.from_metadata(
            meta["model_native_state_contract"],
            require_xau_direction_repair=True,
        )
        if int(meta["seq_input_dim"]) != SIGNAL_DIM_MODEL_NATIVE:
            raise RuntimeError(
                f"[SMART_ENTRY] bundle seq_input_dim={meta['seq_input_dim']} != {SIGNAL_DIM_MODEL_NATIVE}"
            )
        if int(meta["seq_len"]) != SEQ_LEN_MODEL_NATIVE:
            raise RuntimeError(f"[SMART_ENTRY] bundle seq_len={meta['seq_len']} != {SEQ_LEN_MODEL_NATIVE}")
        direction_calibration = meta["direction_calibration"]
        if not isinstance(direction_calibration, dict) or direction_calibration.get("enabled") is not True:
            raise RuntimeError(
                "[SMART_ENTRY] bundle lacks enabled direction_calibration — refusing an "
                "uncalibrated model-direction load"
            )
        path_calibration = meta["path_calibration"]
        if not isinstance(path_calibration, dict) or path_calibration.get("enabled") is not True:
            raise RuntimeError(
                "[SMART_ENTRY] bundle lacks enabled path_calibration — live/replay path heads "
                "must be calibrated before serving"
            )
        mtf = meta["multi_tf"]
        if not isinstance(mtf, dict):
            raise RuntimeError("[SMART_ENTRY] bundle multi_tf contract must be an object")
        if mtf["enabled"] is not True or mtf["v2_mode"] is not True:
            raise RuntimeError("[SMART_ENTRY] bundle must be multi-TF v2 — refusing")
        mtf_shift_minutes = float(mtf["target_availability_shift_minutes"])
        if abs(mtf_shift_minutes - 5.0) > 1e-9:
            raise RuntimeError(
                "[SMART_ENTRY] bundle multi_tf.target_availability_shift_minutes must be 5.0 "
                f"for closed-bar XAU repair serving, got {mtf_shift_minutes!r}"
            )
        per_tf = {
            "M5": int(mtf["m5_seq_len"]),
            "M15": int(mtf["m15_seq_len"]),
            "H1": int(mtf["h1_seq_len"]),
            "H4": int(mtf["h4_seq_len"]),
            "D1": int(mtf["d1_seq_len"]),
        }
        names = [str(x) for x in meta["ordered_signal_names"]]
        builder = ModelNativeStateBuilder(
            ordered_signal_names=names,
            state_contract=state_contract,
            signal_contract=dict(signal_contract),
        )
        LOG.info(
            "[SMART_ENTRY] loaded contract-ACTIVE %s (mode=%s, selection=%s, history_start=%s)",
            bundle_dir.name, mode, MODEL_DIRECTION_SELECTION_MODE, state_contract.feature_history_start_utc,
        )
        return cls(
            bundle_dir=bundle_dir, operating_point=dict(op), device=device,
            _model=model, _meta=meta, _builder=builder, _state_contract=state_contract, _per_tf_seq_lens=per_tf,
            _sizing_authority=sizing_authority,
            _multi_tf_target_availability_shift=pd.Timedelta(minutes=mtf_shift_minutes),
        )

    # ── smart context (in-memory snapshot, refreshed on cv3 cutoff advance) ──
    # The build (~2 min: float32 MTF over full cv3 + frozen-rank buckets + full-
    # frame HTF/REGIME_V4 overrides) ran SYNCHRONOUSLY in the runner loop pre
    # gap-3 — every cv3 cutoff advance starved the per-M1 exit decisions for
    # ~2 min. Now it follows the loader's async-refresh pattern
    # (v12_state_from_prebuilt 2026-06-01): background thread builds a NEW
    # SmartCtxSnapshot on a LOCAL cv3 reference, then swaps ONE attribute
    # (GIL-atomic); decisions read the last completed snapshot and journal
    # context_age_m5_bars. No lock anywhere — the exit path cannot be starved.

    def _build_ctx_snapshot(self, cv3: pd.DataFrame) -> SmartCtxSnapshot:
        """The FULL context build (unchanged math — same one-truth functions the
        blocking path always used). Runs on local state only; safe in a thread."""
        from gx1.execution.v12_model_native_state_live import (
            compute_bucket_ctx_cat_full_frame,
            compute_htf_ctx_full_frame,
        )
        if self._state_contract is None:
            raise RuntimeError("[SMART_ENTRY] model-native state contract not loaded")
        t0 = time.perf_counter()
        cutoff = cv3.index[-1]
        multi_tf = build_multi_tf_from_cv3(cv3)
        # full-frame overrides: ctx_cat buckets (offline frame-global-rank
        # convention) + the 5 long-lookback HTF ctx cols (fresh full-frame
        # recompute; B28's incremental M1-lane stamping is one M5 bar behind
        # the offline convention — parity gate finding 2026-07-08)
        overrides = pd.concat(
            [
                compute_bucket_ctx_cat_full_frame(cv3, self._state_contract),
                compute_htf_ctx_full_frame(cv3, self._state_contract),
            ],
            axis=1,
        )
        return SmartCtxSnapshot(
            multi_tf=multi_tf, frame_overrides=overrides,
            cv3_cutoff=cutoff, built_utc=pd.Timestamp.utcnow(),
            build_seconds=time.perf_counter() - t0,
        )

    def _install_ctx_snapshot(self, snap: SmartCtxSnapshot) -> None:
        """Single-reference swap (GIL-atomic). The builder mirror exists only for
        direct ModelNativeStateBuilder callers; the live decision path passes the
        snapshot's bundle explicitly so it never races the mirror write."""
        self._ctx = snap
        if self._builder is not None:
            self._builder.multi_tf = snap.multi_tf

    def refresh_multi_tf(self, cv3: pd.DataFrame) -> None:
        """BLOCKING context (re)build when cv3's cutoff advanced — the startup /
        parity-gate / offline-driver path (semantics unchanged from pre-gap-3).
        The live runner path uses maybe_schedule_ctx_refresh + predict_live_bar
        instead and never blocks on this."""
        cutoff = cv3.index[-1]
        ctx = self._ctx
        if ctx is not None and ctx.cv3_cutoff == cutoff:
            return
        from gx1.features.htf_features import MULTI_TF_SHIFT
        LOG.info("[SMART_ENTRY] building smart-context snapshot from cv3 (cutoff=%s, blocking)…", cutoff)
        self._multi_tf_shift = dict(MULTI_TF_SHIFT)
        snap = self._build_ctx_snapshot(cv3)
        self._install_ctx_snapshot(snap)
        LOG.info("[SMART_ENTRY] smart-context snapshot ready (cutoff=%s, %.1fs)",
                 cutoff, snap.build_seconds)

    def maybe_schedule_ctx_refresh(self, cv3: pd.DataFrame) -> bool:
        """NON-BLOCKING: schedule a background context rebuild when cv3's cutoff
        advanced past the snapshot's and no refresh is in flight (the loader's
        refresh_if_changed pattern). Returns True only on the scheduling cycle."""
        ctx = self._ctx
        if ctx is None:
            raise RuntimeError(
                "[SMART_ENTRY] no context snapshot — the initial (blocking) "
                "refresh_multi_tf() at startup is mandatory before live decisions"
            )
        if cv3.index[-1] <= ctx.cv3_cutoff:
            return False
        t = self._ctx_refresh_thread
        if t is not None and t.is_alive():
            return False
        t = threading.Thread(
            target=self._async_ctx_refresh, args=(cv3,), daemon=True,
            name="smart_ctx_async_refresh",
        )
        self._ctx_refresh_thread = t
        t.start()
        return True

    def _async_ctx_refresh(self, cv3: pd.DataFrame) -> None:
        """Background-thread worker: full context build on the cv3 reference
        grabbed at schedule time (the loader swaps — never mutates — its frames,
        so this read is race-free), then one atomic snapshot swap. Fail-SAFE:
        on error the previous snapshot stays live and the staleness cap
        (SMART_CTX_MAX_STALENESS_M5) turns a persistent failure into journaled
        Entry NO_DIRECTION events — exits are never affected."""
        try:
            old = self._ctx
            snap = self._build_ctx_snapshot(cv3)
            self._install_ctx_snapshot(snap)
            LOG.info("[smart-ctx-refresh] snapshot cutoff %s → %s (took %.1fs, decisions never blocked)",
                     old.cv3_cutoff if old is not None else None,
                     snap.cv3_cutoff, snap.build_seconds)
        except Exception as exc:  # noqa: BLE001 — fail-safe: keep prior snapshot
            LOG.error(f"[smart-ctx-refresh] FAILED: {exc} — keeping previous snapshot "
                      f"(staleness cap will emit no direction if this persists)")

    @staticmethod
    def context_age_m5_bars(cv3: pd.DataFrame, end_ts: pd.Timestamp,
                            ctx: SmartCtxSnapshot) -> int:
        """cv3 M5 bars in (ctx.cv3_cutoff, end_ts] — 0 ⇒ the snapshot covers the
        decision bar (may be negative for historical end_ts, e.g. the parity gate)."""
        idx = cv3.index
        return int(idx.searchsorted(end_ts, side="right")
                   - idx.searchsorted(ctx.cv3_cutoff, side="right"))

    def _effective_context(
        self, cv3: pd.DataFrame, ctx: SmartCtxSnapshot, end_ts: pd.Timestamp,
    ) -> tuple[dict, pd.DataFrame, int, bool]:
        """Return only a snapshot that exactly covers ``end_ts``.

        A positive age is not repaired, forward-filled, or partially spliced:
        all M5/M15/H1/H4/D1 state must come from one completed full refresh.
        """
        age = self.context_age_m5_bars(cv3, end_ts, ctx)
        if age <= 0:
            return ctx.multi_tf, ctx.frame_overrides, age, False
        raise SmartContextStaleError(
            age=age,
            cap=SMART_CTX_MAX_STALENESS_M5,
            ctx_cutoff=ctx.cv3_cutoff,
            end_ts=end_ts,
        )

    def _prepare_common_history_frame(
        self, loader, cv3: pd.DataFrame, end_ts: pd.Timestamp,
        overrides: pd.DataFrame, multi_tf: dict,
    ) -> pd.DataFrame:
        """Shared common-history build + prepare (ONE truth for the blocking
        gate path and the live async path)."""
        if self._state_contract is None:
            raise RuntimeError("[SMART_ENTRY] model-native state contract not loaded")
        history_start = self._state_contract.feature_history_start_utc
        cv3_idx = cv3.index
        n_from_history_start = int(
            cv3_idx.searchsorted(end_ts, side="right")
            - cv3_idx.searchsorted(history_start, side="left")
        )
        if n_from_history_start < SEQ_LEN_MODEL_NATIVE:
            raise RuntimeError(
                f"[SMART_ENTRY] common-history frame too short: {n_from_history_start} bars"
            )
        joined = loader.get_window(end_ts, n_bars=n_from_history_start)
        history_pos = int(cv3_idx.searchsorted(history_start, side="left"))
        expected_first = cv3_idx[history_pos] if history_pos < len(cv3_idx) else None
        if joined.empty or expected_first is None or joined.index[0] != expected_first:
            raise RuntimeError(
                f"[SMART_ENTRY] common-history window build failed: rows={len(joined)} "
                f"start={joined.index[0] if len(joined) else None} expected_start={expected_first}"
            )
        return self._builder.prepare_frame(
            joined,
            bucket_ctx_cat=overrides,
            multi_tf=multi_tf,
            context_m5=cv3.loc[:end_ts, ["high", "low", "close"]],
        )

    def build_common_history_frame(
        self, loader, end_ts: pd.Timestamp, ctx: SmartCtxSnapshot | None = None,
    ) -> pd.DataFrame:
        """ONE-TRUTH state frame [feature_history_start_utc .. end_ts]
        from the live prebuilt loader (joined cv3+BASE28), prepared with all
        model-native recomputes. Shared by the parity gate and live pipeline.
        ctx=None (gate/startup path): BLOCKING refresh first — behavior and
        values identical to the pre-gap-3 synchronous implementation."""
        if self._builder is None:
            raise RuntimeError("[SMART_ENTRY] not loaded")
        if ctx is None:
            self.refresh_multi_tf(loader._cv3)
            ctx = self._ctx
        cv3 = loader._cv3
        multi_tf, overrides, _age, _spliced = self._effective_context(cv3, ctx, end_ts)
        return self._prepare_common_history_frame(loader, cv3, end_ts, overrides, multi_tf)

    def _multi_tf_window_tensors(
        self, ts: pd.Timestamp, multi_tf: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """Per-TF windows at-or-before ts with the BUNDLE's per-TF seq lens —
        the exact offline dataset path (EntryV10CtxDataset._get_multi_tf_window:
        get_last_n_at_or_before(feats, ts + 5min, n=per_tf,
        tf_shift=MULTI_TF_SHIFT)).
        `multi_tf=None` uses the current snapshot (gate/offline callers)."""
        if multi_tf is None:
            ctx = self._ctx
            if ctx is None:
                raise RuntimeError("[SMART_ENTRY] multi-TF not built — call refresh_multi_tf() first")
            multi_tf = ctx.multi_tf
        from gx1.features.htf_features import get_last_n_at_or_before
        out: dict[str, torch.Tensor] = {}
        availability_ts = pd.Timestamp(ts) + self._multi_tf_target_availability_shift
        for tf, feats in multi_tf.items():
            n = int(self._per_tf_seq_lens[tf])
            arr = get_last_n_at_or_before(
                feats,
                availability_ts,
                n=n,
                tf_shift=self._multi_tf_shift[tf],
            )
            out[f"seq_{tf.lower()}"] = torch.from_numpy(
                arr.astype(np.float32, copy=False)
            ).unsqueeze(0).to(self.device)
        return out

    # ── forward ───────────────────────────────────────────────────────────────

    def forward_states(
        self, states: dict[str, Any], multi_tf: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Forward pre-built seq513 states (from ModelNativeStateBuilder) through
        the calibrated model. Mirrors evaluate_entry_candidate_selective_edge_v1
        _predict_bundle head-for-head. Returns one dict per state row.
        `multi_tf=None` uses the current snapshot (gate/offline callers); the
        live path passes the SAME bundle the states were built with."""
        if self._model is None:
            raise RuntimeError("[SMART_ENTRY] not loaded")
        results: list[dict[str, Any]] = []
        n = states["seq"].shape[0]
        with torch.no_grad():
            for k in range(n):
                ts = pd.Timestamp(states["times"][k])
                seq_t = torch.from_numpy(states["seq"][k]).unsqueeze(0).to(self.device)
                snap_t = torch.from_numpy(states["snap"][k]).unsqueeze(0).to(self.device)
                ctx_cont_t = torch.from_numpy(states["ctx_cont"][k]).unsqueeze(0).to(self.device)
                ctx_cat_t = torch.from_numpy(states["ctx_cat"][k]).unsqueeze(0).to(self.device)
                mtf_kwargs = self._multi_tf_window_tensors(ts, multi_tf=multi_tf)
                out = self._model(seq_t, snap_t, ctx_cat=ctx_cat_t, ctx_cont=ctx_cont_t, **mtf_kwargs)
                for key, value in out.items():
                    if torch.is_tensor(value) and not bool(torch.isfinite(value).all().item()):
                        raise RuntimeError(f"[SMART_ENTRY] non-finite model output '{key}' at {ts}")
                stale_anchor_outputs = [
                    key for key in ("anchor_logits", "delta_logits", "anchor_gate")
                    if out.get(key) is not None
                ]
                if stale_anchor_outputs:
                    raise RuntimeError(
                        "[SMART_ENTRY] model-native bundle emitted forbidden legacy anchor outputs: "
                        + ",".join(stale_anchor_outputs)
                    )
                fusion_evidence: dict[str, Any] = {}
                for output_name, output_width in DIRECTION_EVIDENCE_FUSION_INPUTS:
                    output_value = _require_finite_vector(
                        out.get(output_name),
                        name=output_name,
                        size=output_width,
                        context=f"model forward at {ts}",
                    )
                    fusion_evidence[output_name] = (
                        float(output_value[0])
                        if output_width == 1
                        else output_value.tolist()
                    )
                raw_direction_logits = _require_finite_vector(
                    out.get("raw_direction_logits"),
                    name="raw_direction_logits",
                    size=3,
                    context=f"model forward at {ts}",
                )
                ssot = _direction_ssot_from_logits(
                    out.get("direction_logits"),
                    out.get("public_trade_flat_decision_logits"),
                    context=f"model forward at {ts}",
                )
                probs = ssot["direction_probs"]
                public_trade_flat_probs = ssot["public_trade_flat_decision_probs"]
                p_long, p_short, p_flat = float(probs[0]), float(probs[1]), float(probs[2])
                edge_score = max(p_long, p_short) - p_flat
                path_quality_raw = _require_finite_vector(
                    out.get("path_quality"), name="path_quality", size=1, context=f"model forward at {ts}"
                )
                bad_path_logit = _require_finite_vector(
                    out.get("bad_path_logit"), name="bad_path_logit", size=1, context=f"model forward at {ts}"
                )
                tradable_logit = _require_finite_vector(
                    out.get("tradable_logit"), name="tradable_logit", size=1, context=f"model forward at {ts}"
                )
                mfe_first_n_raw = _require_finite_vector(
                    out.get("mfe_first_n"), name="mfe_first_n", size=1, context=f"model forward at {ts}"
                )
                clean_edge_logit = _require_finite_vector(
                    out.get("clean_edge_logit"), name="clean_edge_logit", size=1, context=f"model forward at {ts}"
                )
                survival_logit = _require_finite_vector(
                    out.get("survival_logit"), name="survival_logit", size=1, context=f"model forward at {ts}"
                )
                dip_pred = _require_finite_vector(
                    out.get("dip_pred"), name="dip_pred", size=18, context=f"model forward at {ts}"
                )
                forecast_pred = _require_finite_vector(
                    out.get("forecast_pred"), name="forecast_pred", size=4, context=f"model forward at {ts}"
                )
                timing_pred = _require_finite_vector(
                    out.get("timing_pred"), name="timing_pred", size=12, context=f"model forward at {ts}"
                )
                tail_risk_pred = _require_finite_vector(
                    out.get("tail_risk_pred"), name="tail_risk_pred", size=6, context=f"model forward at {ts}"
                )
                vol_forecast_pred = _require_finite_vector(
                    out.get("vol_forecast_pred"), name="vol_forecast_pred", size=3, context=f"model forward at {ts}"
                )
                specialist_names = [
                    str(value)
                    for value in self._meta["specialist_fusion"]["trainable_specialists"]
                ]
                if specialist_names != list(MODEL_NATIVE_REQUIRED_SPECIALISTS):
                    raise RuntimeError(
                        "[SMART_ENTRY] model-native specialist order mismatch: "
                        f"observed={specialist_names} expected={list(MODEL_NATIVE_REQUIRED_SPECIALISTS)}"
                    )
                specialist_gate = _require_finite_vector(
                    out.get("specialist_gate"),
                    name="specialist_gate",
                    size=len(MODEL_NATIVE_REQUIRED_SPECIALISTS),
                    context=f"model forward at {ts}",
                )
                if bool((specialist_gate < 0.0).any()) or not np.isclose(
                    float(specialist_gate.sum()), 1.0, rtol=1e-6, atol=1e-7
                ):
                    raise RuntimeError(
                        f"[SMART_ENTRY] model forward at {ts} specialist_gate is not a probability simplex"
                    )
                mtf_logits = _require_finite_vector(
                    out.get("mtf_dir_logits"),
                    name="mtf_dir_logits",
                    size=3,
                    context=f"model forward at {ts}",
                )
                mtf_probs = _softmax_np(mtf_logits)
                side_logits = _require_finite_vector(
                    out.get("side_logits"), name="side_logits", size=2, context=f"model forward at {ts}"
                )
                side_bad_path_logit = _require_finite_vector(
                    out.get("side_bad_path_logit"),
                    name="side_bad_path_logit",
                    size=2,
                    context=f"model forward at {ts}",
                )
                side_validity_logit = _require_finite_vector(
                    out.get("side_validity_logit"),
                    name="side_validity_logit",
                    size=2,
                    context=f"model forward at {ts}",
                )
                trendline_rail_logits = _require_finite_vector(
                    out.get("trendline_rail_logits"),
                    name="trendline_rail_logits",
                    size=6,
                    context=f"model forward at {ts}",
                )
                tf_agreement_logit = _require_finite_vector(
                    out.get("tf_agreement_logit"),
                    name="tf_agreement_logit",
                    size=1,
                    context=f"model forward at {ts}",
                )
                path_quality_log_var = _require_finite_vector(
                    out.get("path_quality_log_var"),
                    name="path_quality_log_var",
                    size=1,
                    context=f"model forward at {ts}",
                )
                position_size_logit = _require_finite_vector(
                    out.get("position_size_logit"),
                    name="position_size_logit",
                    size=1,
                    context=f"model forward at {ts}",
                )
                tf_agreement_pred = _sigmoid_float(float(tf_agreement_logit[0]))
                position_size_pred = _sigmoid_float(float(position_size_logit[0]))
                clean_edge_prob = _sigmoid_float(float(clean_edge_logit[0]))
                survival_prob = _sigmoid_float(float(survival_logit[0]))
                path_quality_std = float(np.exp(0.5 * float(path_quality_log_var[0])))
                if not np.isfinite(path_quality_std):
                    raise RuntimeError(
                        f"[SMART_ENTRY] model forward at {ts} path_quality_std is non-finite"
                    )
                side_probs = _softmax_np(side_logits)
                trendline_rail_probs = [_sigmoid_float(float(x)) for x in trendline_rail_logits]
                p_trade_hier = float(public_trade_flat_probs[0])
                p_flat_hier = float(public_trade_flat_probs[1])
                p_long_given_trade = float(side_probs[0])
                p_short_given_trade = float(side_probs[1])
                long_bad_path_prob = _sigmoid_float(float(side_bad_path_logit[0]))
                short_bad_path_prob = _sigmoid_float(float(side_bad_path_logit[1]))
                long_validity_prob = _sigmoid_float(float(side_validity_logit[0]))
                short_validity_prob = _sigmoid_float(float(side_validity_logit[1]))
                signal_names = [str(x) for x in self._meta["ordered_signal_names"]]
                snap_row = np.asarray(states["snap"][k], dtype=np.float32).reshape(-1)
                geometry_channel_edge = _required_feature_value(
                    snap_row,
                    signal_names,
                    "chart.geometry_channel_edge_pressure",
                )
                geometry_rising_support_rail_long = _required_feature_value(
                    snap_row,
                    signal_names,
                    "chart.geometry_rising_support_rail_long_pressure",
                )
                geometry_rising_support_rail_short_trap = _required_feature_value(
                    snap_row,
                    signal_names,
                    "chart.geometry_rising_support_rail_short_trap_pressure",
                )
                geometry_falling_resistance_rail_short = _required_feature_value(
                    snap_row,
                    signal_names,
                    "chart.geometry_falling_resistance_rail_short_pressure",
                )
                geometry_falling_resistance_rail_long_trap = _required_feature_value(
                    snap_row,
                    signal_names,
                    "chart.geometry_falling_resistance_rail_long_trap_pressure",
                )
                mtf_trend_evidence = _required_feature_value(
                    snap_row,
                    signal_names,
                    "trend.mtf_confluence_trend_direction_score",
                )
                res = {
                    "time": ts,
                    **fusion_evidence,
                    "raw_direction_logits": raw_direction_logits.tolist(),
                    "direction_logits": ssot["direction_logits"].tolist(),
                    "direction_probs": ssot["direction_probs"].tolist(),
                    "model_direction_index": ssot["model_direction_index"],
                    "model_direction": ssot["model_direction"],
                    "public_trade_flat_decision_logits": ssot[
                        "public_trade_flat_decision_logits"
                    ].tolist(),
                    "public_trade_flat_decision_probs": ssot[
                        "public_trade_flat_decision_probs"
                    ].tolist(),
                    "public_trade_flat_decision_index": ssot[
                        "public_trade_flat_decision_index"
                    ],
                    "public_trade_flat_decision": ssot["public_trade_flat_decision"],
                    "p_long": p_long, "p_short": p_short, "p_flat": p_flat,
                    "edge_score": float(edge_score),
                    "session_id": int(states["ctx_cat"][k][0]),
                    "path_quality_pred": float(path_quality_raw[0]),
                    "path_quality": float(path_quality_raw[0]),
                    "bad_path_logit": float(bad_path_logit[0]),
                    "bad_path_prob": _sigmoid_float(float(bad_path_logit[0])),
                    "tradable_logit": float(tradable_logit[0]),
                    "tradable_prob": _sigmoid_float(float(tradable_logit[0])),
                    "mfe_first_n_pred": float(mfe_first_n_raw[0]),
                    "clean_edge_logit": float(clean_edge_logit[0]),
                    "clean_edge_prob": clean_edge_prob,
                    "survival_logit": float(survival_logit[0]),
                    "survival_prob": survival_prob,
                    "dip_pred": dip_pred.tolist(),
                    "forecast_pred": forecast_pred.tolist(),
                    "timing_pred": timing_pred.tolist(),
                    "tail_risk_pred": tail_risk_pred.tolist(),
                    "vol_forecast_pred": vol_forecast_pred.tolist(),
                    "specialist_names": specialist_names,
                    "specialist_gate": specialist_gate.tolist(),
                    "tf_agreement_logit": float(tf_agreement_logit[0]),
                    "tf_agreement_pred": tf_agreement_pred,
                    "path_quality_log_var": float(path_quality_log_var[0]),
                    "path_quality_std": path_quality_std,
                    "position_size_pred": position_size_pred,
                    "position_size_logit": float(position_size_logit[0]),
                    "p_trade": p_trade_hier,
                    "p_flat_hier": p_flat_hier,
                    "p_long_given_trade": p_long_given_trade,
                    "p_short_given_trade": p_short_given_trade,
                    "side_logits": side_logits.tolist(),
                    "side_probs": side_probs.tolist(),
                    "long_bad_path_prob": long_bad_path_prob,
                    "short_bad_path_prob": short_bad_path_prob,
                    "side_validity_logit": side_validity_logit.tolist(),
                    "long_validity_prob": long_validity_prob,
                    "short_validity_prob": short_validity_prob,
                    "mtf_dir_logits": mtf_logits.tolist(),
                    "mtf_dir_probs": mtf_probs.tolist(),
                    "geometry_channel_edge_pressure": geometry_channel_edge,
                    "geometry_rising_support_rail_long_pressure": geometry_rising_support_rail_long,
                    "geometry_rising_support_rail_short_trap_pressure": geometry_rising_support_rail_short_trap,
                    "geometry_falling_resistance_rail_short_pressure": geometry_falling_resistance_rail_short,
                    "geometry_falling_resistance_rail_long_trap_pressure": geometry_falling_resistance_rail_long_trap,
                    "trendline_rail_logits": trendline_rail_logits.tolist(),
                    "trendline_rail_probs": trendline_rail_probs,
                    "mtf_trend_evidence": mtf_trend_evidence,
                    "calibration_version": self._meta["direction_calibration"]["version"],
                    "direction_calibration_enabled": bool(self._meta["direction_calibration"]["enabled"]),
                    "direction_calibration_temperature": self._meta["direction_calibration"]["temperature"],
                    "direction_calibration_bias": self._meta["direction_calibration"]["bias"],
                    "path_calibration_enabled": bool(self._meta["path_calibration"]["enabled"]),
                    "path_calibration": self._meta["path_calibration"],
                }
                results.append(res)
        return results

    # ── live per-M5 forward (async-context path — serving-wave gap 3) ────────

    def predict_live_bar(self, loader, end_ts: pd.Timestamp) -> dict[str, Any]:
        """LIVE per-M5 decision forward: uses the LAST COMPLETED context snapshot
        — NEVER blocks on the ~2-min context refresh (which now runs in a
        background thread, scheduled here on cv3 cutoff advance). One atomic
        snapshot grab keeps state build + model forward internally consistent.

        Fail-closed: raises SmartContextStaleError whenever the snapshot does
        not cover the decision bar (the pipeline journals model-direction
        unavailability and retries next poll).
        Journals staleness on every result: context_age_m5_bars / context_cutoff_ts /
        context_refresh_in_flight / context_mtf_incremental.
        """
        if self._builder is None or self._model is None:
            raise RuntimeError("[SMART_ENTRY] not loaded")
        cv3 = loader._cv3
        self.maybe_schedule_ctx_refresh(cv3)
        ctx = self._ctx   # ONE atomic grab — never re-read during this decision
        if ctx is None:
            raise RuntimeError("[SMART_ENTRY] no context snapshot — startup refresh missing")
        age = self.context_age_m5_bars(cv3, end_ts, ctx)
        if age > SMART_CTX_MAX_STALENESS_M5:
            raise SmartContextStaleError(
                age=age, cap=SMART_CTX_MAX_STALENESS_M5,
                ctx_cutoff=ctx.cv3_cutoff, end_ts=end_ts,
            )
        multi_tf, overrides, age, spliced = self._effective_context(cv3, ctx, end_ts)
        frame = self._prepare_common_history_frame(loader, cv3, end_ts, overrides, multi_tf)
        states = self._builder.build_states(frame, [end_ts])
        head = self.forward_states(states, multi_tf=multi_tf)[0]
        t = self._ctx_refresh_thread
        head["context_age_m5_bars"] = int(max(age, 0))
        head["context_cutoff_ts"] = str(ctx.cv3_cutoff)
        head["context_refresh_in_flight"] = bool(t is not None and t.is_alive())
        head["context_mtf_incremental"] = bool(spliced)
        return head

    # ── decision (operating point from the contract — ONE truth) ─────────────

    def decide(self, head_out: dict[str, Any], atr_bps: float) -> dict[str, Any]:
        """Emit the runner action from the model's final direction argmax.

        ``direction_logits`` plus ``public_trade_flat_decision_logits`` are
        validated again here so no caller can inject a parallel side, threshold,
        or session decision between model forward and live action.
        """
        retired_fields = sorted(
            key
            for key in head_out
            if any(
                fragment in str(key).lower()
                for fragment in RETIRED_RUNTIME_EVIDENCE_FRAGMENTS
            )
        )
        if retired_fields:
            raise RuntimeError(
                "[SMART_ENTRY] retired live overlay fields are forbidden: "
                + ",".join(retired_fields)
            )
        observed_head_fields = frozenset(head_out)
        missing_head_fields = sorted(
            MODEL_NATIVE_DECISION_HEAD_REQUIRED_FIELDS - observed_head_fields
        )
        unexpected_head_fields = sorted(
            observed_head_fields
            - MODEL_NATIVE_DECISION_HEAD_REQUIRED_FIELDS
            - MODEL_NATIVE_DECISION_HEAD_CONTEXT_FIELDS
        )
        observed_context_fields = (
            observed_head_fields & MODEL_NATIVE_DECISION_HEAD_CONTEXT_FIELDS
        )
        partial_context_fields = bool(observed_context_fields) and (
            observed_context_fields != MODEL_NATIVE_DECISION_HEAD_CONTEXT_FIELDS
        )
        if missing_head_fields or unexpected_head_fields or partial_context_fields:
            raise RuntimeError(
                "[SMART_ENTRY] decision head exact schema mismatch: "
                f"missing={missing_head_fields} unexpected={unexpected_head_fields} "
                f"context_fields={sorted(observed_context_fields)}"
            )

        selection_mode = self.operating_point.get("selection_score")
        if selection_mode != MODEL_DIRECTION_SELECTION_MODE:
            raise RuntimeError(
                "[SMART_ENTRY] operating_point.selection_score must be exactly "
                f"{MODEL_DIRECTION_SELECTION_MODE!r}; got {selection_mode!r}"
            )
        ssot = _validate_reported_direction_ssot(head_out)
        direction_index = int(ssot["model_direction_index"])
        model_direction = str(ssot["model_direction"])
        direction_probs = ssot["direction_probs"]
        selected_side = direction_index if direction_index in (0, 1) else None
        action = MODEL_DIRECTION_ACTIONS[direction_index]
        session_id_raw = _require_finite_vector(
            head_out.get("session_id"),
            name="session_id",
            size=1,
            context="decision",
        )[0]
        session_id = int(session_id_raw)
        if float(session_id_raw) != float(session_id) or session_id not in SESSION_NAMES:
            raise RuntimeError(
                "[SMART_ENTRY] session_id must be an exact model-native category "
                f"in {sorted(SESSION_NAMES)}; got {head_out.get('session_id')!r}"
            )
        session = SESSION_NAMES[session_id]
        edge = float(max(direction_probs[0], direction_probs[1]) - direction_probs[2])
        selection_score = float(direction_probs[direction_index])
        atr_bps_value = float(atr_bps)
        if not np.isfinite(atr_bps_value) or atr_bps_value <= 0.0:
            raise RuntimeError(f"[SMART_ENTRY] atr_bps must be finite and positive; got {atr_bps!r}")

        diagnostics = _validate_model_native_diagnostics(
            head_out,
            MODEL_NATIVE_DECISION_DIAGNOSTIC_KEYS,
        )
        sizing_authority = require_model_native_sizing_authority_contract(
            self._sizing_authority,
            context="[SMART_ENTRY] decision sizing",
            required_mode=MODEL_NATIVE_SIZING_MODE_LEARNED,
        )

        # Exit-bound snapshot. Direction fields come only from the validated SSOT.
        # hold_horizon_bars_pred DELIBERATELY ABSENT (blocked head -> -1 sentinel
        # -> HOLD_HORIZON_EXPIRED inert; live-equivalent to the joint replay).
        snapshot = {
            "decision_ts": str(head_out["time"]),
            "runtime_evidence_schema_version": (
                MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION
            ),
            "model_policy": MODEL_NATIVE_RUNTIME_POLICY,
            "session_id": session_id,
            "session": session,
            "direction_logits": ssot["direction_logits"].tolist(),
            "direction_probs": direction_probs.tolist(),
            "model_direction_index": direction_index,
            "model_direction": model_direction,
            "public_trade_flat_decision_logits": ssot[
                "public_trade_flat_decision_logits"
            ].tolist(),
            "public_trade_flat_decision_probs": ssot[
                "public_trade_flat_decision_probs"
            ].tolist(),
            "public_trade_flat_decision_index": ssot[
                "public_trade_flat_decision_index"
            ],
            "public_trade_flat_decision": ssot["public_trade_flat_decision"],
            "selected_side": selected_side,
            "path_quality": head_out["path_quality_pred"],
            "mfe_first_n": head_out["mfe_first_n_pred"],
            "tradable_prob": head_out["tradable_prob"],
            "bad_path_prob": head_out["bad_path_prob"],
            "p_trade": float(ssot["public_trade_flat_decision_probs"][0]),
            "p_flat_hier": float(ssot["public_trade_flat_decision_probs"][1]),
            "atr_bps": atr_bps_value,
            "sizing_authority_contract": sizing_authority,
            **diagnostics,
        }
        snapshot = require_model_native_runtime_evidence(
            snapshot,
            context="SMART_ENTRY_DECISION",
        )
        out = {
            "action": action,
            "action_id": {"SKIP": 0, "TAKE_LONG_NOW": 1, "TAKE_SHORT_NOW": 2}[action],
            "model_direction_index": direction_index,
            "model_direction": model_direction,
            "direction_logits": ssot["direction_logits"].tolist(),
            "direction_probs": direction_probs.tolist(),
            "public_trade_flat_decision_logits": ssot[
                "public_trade_flat_decision_logits"
            ].tolist(),
            "public_trade_flat_decision_probs": ssot[
                "public_trade_flat_decision_probs"
            ].tolist(),
            "public_trade_flat_decision_index": ssot[
                "public_trade_flat_decision_index"
            ],
            "public_trade_flat_decision": ssot["public_trade_flat_decision"],
            "edge_score": edge,
            "selection_score_mode": selection_mode,
            "selection_score": selection_score,
            "session_id": session_id,
            "session": session,
            "p_long": float(direction_probs[0]),
            "p_short": float(direction_probs[1]),
            "p_flat": float(direction_probs[2]),
            "p_trade": float(ssot["public_trade_flat_decision_probs"][0]),
            "p_flat_hier": float(ssot["public_trade_flat_decision_probs"][1]),
            "selected_side": selected_side,
            "sizing_authority_contract": snapshot[
                "sizing_authority_contract"
            ],
            **diagnostics,
            "v10_path_quality_pred": head_out["path_quality_pred"],
            "v10_mfe_pred_at_entry": head_out["mfe_first_n_pred"],
            "v10_tradable_prob": head_out["tradable_prob"],
            "v10_bad_path_prob": head_out["bad_path_prob"],
            "decision_ts": str(head_out["time"]),
            "_v10_snapshot": snapshot,
            "policy": MODEL_NATIVE_RUNTIME_POLICY,
            "stub": False,
        }
        # async-context staleness journal (serving-wave gap 3) — present only on
        # the live predict_live_bar path; the parity gate forwards heads directly.
        for k in ("context_age_m5_bars", "context_cutoff_ts",
                  "context_refresh_in_flight", "context_mtf_incremental"):
            if k in head_out:
                out[k] = head_out[k]
        return out
