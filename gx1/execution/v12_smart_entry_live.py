#!/usr/bin/env python3
"""LIVE model-native seq513 XAU Entry adapter.

Loads a contract-resolved, launch-admitted model-native v10_entry bundle through
the one-truth offline loader, forwards it per M5 close on the exact
owner-declared signal state
(ModelNativeStateBuilder) + live multi-TF windows, and requires the PINNED
operating point read from PROJECT_STATE_artifacts.json to select
``entry_fitted_q_unique_argmax``. The three raw-bps
``entry_action_q_bps`` values are the only LONG/SHORT/FLAT decision. No live
session, threshold, calibration, utility, rail, or side overlay may change it.

Serving architecture: the only live Entry load path is
load_entry_v10_ctx_bundle (full active-head reconstruction),
which the offline evaluator
(evaluate_entry_candidate_selective_edge_v1._predict_bundle) also uses. This
adapter mirrors that forward exactly, so serve must equal the admitted evidence
path.

Entry SSOT:
    action = unique_argmax(entry_action_q_bps)  # LONG=0, SHORT=1, FLAT=2

An exact Q tie fails closed. Genuine auxiliary outcome heads remain learned
representation diagnostics only. ``position_size_logit`` changes execution
units only through the separately admitted sizing owner. The frozen snapshot
binds raw Q, the learned Entry token representation, exact categorical state,
and decision timing; it carries no probability/calibration alias.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from gx1.contracts.immutable_event_authority_v1 import (
    ImmutableEventAuthorityError,
    require_newest_immutable_event,
)
from gx1.contracts.entry_fitted_q_v1 import (
    require_entry_fitted_q_production_economics_readiness,
)
from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_MTF_CONTEXT_COUNT
from gx1.features.htf_features import MULTI_TF_FEATURE_COUNT_V4
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CAT_INDEX_BY_NAME,
    require_model_native_signal_contract,
)
from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION,
    MODEL_NATIVE_RUNTIME_POLICY,
    RETIRED_RUNTIME_EVIDENCE_FRAGMENTS,
    require_model_native_runtime_head_evidence,
    require_model_native_runtime_evidence,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    DIP_HEAD_DIM,
    EXACT_TRENDLINE_EVENT_OUTPUT_DIM,
    FORECAST_HEAD_DIM,
    TAIL_RISK_HEAD_DIM,
    TIMING_HEAD_DIM,
    VOL_FORECAST_HEAD_DIM,
)
from gx1.contracts.entry_model_native_sizing_authority_v1 import (
    MODEL_NATIVE_SIZING_MODE_LEARNED,
    prepare_model_native_sizing_authority,
    require_model_native_sizing_authority_contract,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_ACTION_BY_INDEX,
    MODEL_DIRECTION_ACTION_ID_BY_INDEX,
    MODEL_DIRECTION_FLAT_INDEX,
    MODEL_DIRECTION_LONG_INDEX,
    MODEL_DIRECTION_NAME_BY_INDEX,
    MODEL_DIRECTION_SELECTION_MODE,
    MODEL_DIRECTION_SHORT_INDEX,
    MODEL_DIRECTION_TRADE_INDICES,
    UNIFIED_EXIT_ACTION_ORDER,
    UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM,
    UNIFIED_EXIT_ENTRY_REPRESENTATION_KEY,
    UNIFIED_EXIT_MAX_PATH_BARS,
    UNIFIED_EXIT_MODEL_REPRESENTATION_KEY,
    UNIFIED_EXIT_SIDE_ORDER,
    canonical_unified_evidence_sha256,
    require_model_direction_decision_contract,
    require_model_direction_operating_point,
    require_unified_exit_output,
    require_unified_exit_path_envelope,
    unified_exit_path_tensor,
)
from gx1.execution.v12_model_native_state_live import (
    SEQ_LEN_MODEL_NATIVE,
    SIGNAL_DIM_MODEL_NATIVE,
    ModelNativeStateContract,
    ModelNativeStateBuilder,
    build_multi_tf_from_cv3,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)
from gx1.contracts.entry_exit_feature_surface_v1 import require_m1_feature_window
from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_MTF_CONTEXT_TIMEFRAMES,
    EXIT_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.contracts.unified_exit_incremental_carry_v1 import (
    UNIFIED_EXIT_INCREMENTAL_CARRY_GENESIS_SHA256,
    build_unified_exit_incremental_carry_envelope,
    decode_unified_exit_incremental_carry_tensors,
    require_unified_exit_incremental_carry_envelope,
)
from gx1.contracts.unified_exit_input_v1 import (
    build_unified_exit_input_envelope,
)
from gx1.contracts.entry_decision_token_v1 import (
    entry_decision_token_tensor,
    require_entry_decision_token_snapshot,
)
from gx1.time.session_detector import SESSION_NAME_BY_ID

LOG = logging.getLogger("v12_smart_entry_live")

SESSION_NAMES = SESSION_NAME_BY_ID
MODEL_DIRECTION_NAMES = MODEL_DIRECTION_NAME_BY_INDEX
MODEL_DIRECTION_ACTIONS = MODEL_DIRECTION_ACTION_BY_INDEX
MODEL_NATIVE_REQUIRED_SPECIALISTS = MODEL_NATIVE_TRAINING_SPECIALISTS
MODEL_NATIVE_FORWARD_PARITY_EVIDENCE_KEYS = tuple(dict.fromkeys((
    "entry_action_q_bps",
    "entry_q_joint_hidden",
    "side_mae_bps",
    "trendline_event_logits",
)))
MODEL_NATIVE_DECISION_DIAGNOSTIC_KEYS = tuple(dict.fromkeys((
    *MODEL_NATIVE_FORWARD_PARITY_EVIDENCE_KEYS,
    "dip_pred",
    "forecast_pred",
    "timing_pred",
    "tail_risk_pred",
    "vol_forecast_pred",
    "specialist_names",
    "specialist_gate",
    "tf_gate",
    "family_tf_cooperation_gate",
    "family_tf_feature_gate",
    "position_size_pred",
    "position_size_logit",
    UNIFIED_EXIT_ENTRY_REPRESENTATION_KEY,
)))
MODEL_NATIVE_DECISION_HEAD_REQUIRED_FIELDS = frozenset(
    {
        "time",
        "entry_action_q_bps",
        "entry_action_q_margin_bps",
        "model_direction_index",
        "model_direction",
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


def _entry_q_ssot(
    entry_action_q_bps_value: Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Validate and select the sole Entry authority: unique raw-bps Q argmax."""

    q_values = _require_finite_vector(
        entry_action_q_bps_value,
        name="entry_action_q_bps",
        size=3,
        context=context,
    )
    winner_count = int(np.count_nonzero(q_values == np.max(q_values)))
    if winner_count != 1:
        raise RuntimeError(
            f"[SMART_ENTRY] {context} Entry Q has no unique top action"
        )
    action_index = int(np.argmax(q_values))
    runner_up = float(np.partition(q_values, -2)[-2])
    return {
        "entry_action_q_bps": q_values,
        "model_direction_index": action_index,
        "model_direction": MODEL_DIRECTION_NAMES[action_index],
        "entry_action_q_margin_bps": float(q_values[action_index]) - runner_up,
    }


def _validate_reported_entry_q_ssot(
    head_out: dict[str, Any],
) -> dict[str, Any]:
    ssot = _entry_q_ssot(
        head_out.get("entry_action_q_bps"),
        context="decision",
    )
    for key in ("model_direction_index", "model_direction"):
        if head_out.get(key) != ssot[key]:
            raise RuntimeError(
                f"[SMART_ENTRY] decision {key} mismatches raw Entry-Q argmax"
            )
    observed_margin = _require_finite_vector(
        [head_out.get("entry_action_q_margin_bps")],
        name="entry_action_q_margin_bps",
        size=1,
        context="decision",
    )[0]
    if not np.isclose(
        observed_margin,
        ssot["entry_action_q_margin_bps"],
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError(
            "[SMART_ENTRY] decision Entry-Q margin mismatches raw Q"
        )
    return ssot


def _sigmoid_float(value: float) -> float:
    value = float(np.clip(value, -80.0, 80.0))
    return float(1.0 / (1.0 + np.exp(-value)))


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

    vector("entry_action_q_bps", 3)
    vector("entry_q_joint_hidden", UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM)
    vector("side_mae_bps", 2)
    vector("trendline_event_logits", 4)
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
    tf_gate = vector("tf_gate", ENTRY_MTF_CONTEXT_COUNT)
    if bool((tf_gate < 0.0).any()) or not np.isclose(
        float(tf_gate.sum()), 1.0, rtol=1e-6, atol=1e-7
    ):
        raise RuntimeError("[SMART_ENTRY] tf_gate is not a probability simplex")
    family_tf_cooperation_gate = vector(
        "family_tf_cooperation_gate",
        ENTRY_MTF_CONTEXT_COUNT * len(MODEL_NATIVE_TRAINING_SPECIALISTS),
    )
    if bool((family_tf_cooperation_gate < 0.0).any()) or not np.isclose(
        float(family_tf_cooperation_gate.sum()), 1.0, rtol=1e-6, atol=1e-7
    ):
        raise RuntimeError(
            "[SMART_ENTRY] family_tf_cooperation_gate is not a probability simplex"
        )
    family_tf_feature_gate = vector(
        "family_tf_feature_gate",
        ENTRY_MTF_CONTEXT_COUNT * MULTI_TF_FEATURE_COUNT_V4,
    )
    if bool(
        (family_tf_feature_gate <= 0.0).any()
        or (family_tf_feature_gate >= 2.0).any()
    ):
        raise RuntimeError(
            "[SMART_ENTRY] family_tf_feature_gate is outside learned (0,2) contract"
        )

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


@dataclass
class SmartEntryLiveInference:
    bundle_dir: Path
    operating_point: dict[str, Any]
    device: str = "cpu"
    _bundle_sha256: str = field(default="", repr=False)
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
    _exit_feature_surface_provider: Any = field(default=None, repr=False)
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
    def load_candidate_for_parity(
        cls,
        *,
        bundle_dir: Path,
        operating_point: Mapping[str, Any],
        device: str,
    ) -> "SmartEntryLiveInference":
        """Load an explicit pre-launch candidate through the live adapter.

        No artifact selection, ALLOW state, sizing authority, fallback, or
        "latest" lookup exists on this proof-only path.  The immutable parity
        event binds the exact bundle and complete rule-free operating point.
        """

        adapter = cls._from_strict_bundle(
            bundle_dir=Path(bundle_dir).expanduser().resolve(),
            operating_point=require_model_direction_operating_point(
                operating_point,
                context="[SMART_ENTRY] pre-launch candidate parity",
            ),
            device=device,
            sizing_authority=None,
            load_context="pre-launch candidate parity",
        )
        lineage = adapter._meta.get("run_lineage")
        if (
            not isinstance(lineage, Mapping)
            or lineage.get("training_profile") != "candidate"
            or lineage.get("requested_subsample_rows") != 0
            or lineage.get("physical_train_rows")
            != lineage.get("effective_train_rows")
        ):
            raise RuntimeError(
                "[SMART_ENTRY] parity requires a full-population "
                "candidate-profile bundle"
            )
        return adapter

    @classmethod
    def load_immutable_exit_recovery(
        cls,
        *,
        bundle_dir: Path,
        expected_bundle_sha256: str,
        operating_point: Mapping[str, Any],
        sizing_authority: Mapping[str, Any],
        device: str,
    ) -> "SmartEntryLiveInference":
        """Load only the exact bundle frozen into an existing TradeState.

        This path has no new-Entry authority and never consults the mutable
        active registry. It exists solely so launch revocation or promotion
        cannot strand an already-open trade or switch its Exit model.
        """

        authority = require_model_native_sizing_authority_contract(
            sizing_authority,
            context="[SMART_ENTRY] immutable Exit recovery",
            required_mode=MODEL_NATIVE_SIZING_MODE_LEARNED,
        )
        prepare_model_native_sizing_authority(
            authority,
            context="[SMART_ENTRY] immutable Exit recovery",
        )
        adapter = cls._from_strict_bundle(
            bundle_dir=Path(bundle_dir).expanduser().resolve(strict=True),
            operating_point=require_model_direction_operating_point(
                operating_point,
                context="[SMART_ENTRY] immutable Exit recovery",
            ),
            device=device,
            sizing_authority=authority,
            load_context="immutable Exit recovery",
        )
        if adapter._bundle_sha256 != expected_bundle_sha256:
            raise RuntimeError(
                "[SMART_ENTRY] immutable Exit recovery bundle sha256 mismatch"
            )
        return adapter

    @classmethod
    def _from_strict_bundle(
        cls,
        *,
        bundle_dir: Path,
        operating_point: Mapping[str, Any],
        device: str,
        sizing_authority: Mapping[str, Any] | None,
        load_context: str,
    ) -> "SmartEntryLiveInference":
        """Construct live and candidate adapters from one strict byte path."""

        if not isinstance(device, str) or not device.strip():
            raise RuntimeError(
                "[SMART_ENTRY] device must be an explicit non-empty string"
            )
        op = require_model_direction_operating_point(
            operating_point,
            context=f"[SMART_ENTRY] {load_context}",
        )
        bundle_dir = Path(bundle_dir).expanduser().resolve()

        from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle

        bundle = load_entry_v10_ctx_bundle(bundle_dir=bundle_dir, device=device)
        model = bundle.transformer_model
        model.eval()
        meta = dict(bundle.metadata)
        require_model_direction_decision_contract(
            meta,
            context=f"[SMART_ENTRY] {load_context} bundle",
        )
        require_entry_fitted_q_production_economics_readiness(
            meta.get("entry_fitted_q_production_economics"),
            context=f"SMART_ENTRY_{load_context}_SERVING",
            require_ready=True,
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
            raise RuntimeError(
                f"[SMART_ENTRY] bundle seq_len={meta['seq_len']} != {SEQ_LEN_MODEL_NATIVE}"
            )
        if "direction_calibration" in meta:
            raise RuntimeError(
                "[SMART_ENTRY] Entry-Q bundles must not carry direction calibration"
            )
        if "path_calibration" in meta:
            raise RuntimeError(
                "[SMART_ENTRY] Entry-Q bundles must not carry retired path calibration"
            )
        mtf = meta["multi_tf"]
        if not isinstance(mtf, dict):
            raise RuntimeError(
                "[SMART_ENTRY] bundle multi_tf contract must be an object"
            )
        if mtf["enabled"] is not True or mtf["v4_mode"] is not True:
            raise RuntimeError(
                "[SMART_ENTRY] bundle must be exact multi-TF V4 with all "
                "eight families — refusing"
            )
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
        from gx1.execution.v12_state_from_prebuilt import (
            _require_volatility_squeeze_artifacts_from_bound_cache,
        )

        builder = ModelNativeStateBuilder(
            ordered_signal_names=names,
            state_contract=state_contract,
            signal_contract=dict(signal_contract),
            volatility_squeeze_artifacts=(
                _require_volatility_squeeze_artifacts_from_bound_cache()
            ),
        )
        LOG.info(
            "[SMART_ENTRY] loaded %s %s (mode=%s, selection=%s, history_start=%s)",
            load_context,
            bundle_dir.name,
            MODEL_NATIVE_CONTRACT_MODE,
            MODEL_DIRECTION_SELECTION_MODE,
            state_contract.feature_history_start_utc,
        )
        return cls(
            bundle_dir=bundle_dir,
            operating_point=dict(op),
            device=device,
            _bundle_sha256=bundle.bundle_sha256,
            _model=model,
            _meta=meta,
            _builder=builder,
            _state_contract=state_contract,
            _per_tf_seq_lens=per_tf,
            _sizing_authority=(
                dict(sizing_authority) if sizing_authority is not None else {}
            ),
            _multi_tf_target_availability_shift=pd.Timedelta(
                minutes=mtf_shift_minutes
            ),
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
            compute_htf_ctx_full_frame,
        )
        if self._state_contract is None:
            raise RuntimeError("[SMART_ENTRY] model-native state contract not loaded")
        t0 = time.perf_counter()
        cutoff = cv3.index[-1]
        if self._meta is None:
            raise RuntimeError("[SMART_ENTRY] bundle metadata unavailable")
        mtf_contract = self._meta["multi_tf"]
        multi_tf = build_multi_tf_from_cv3(
            cv3,
            matrix_contract=str(mtf_contract["matrix_contract"]),
            feature_names=[
                str(name) for name in mtf_contract["feature_names"]
            ],
        )
        # full-frame overrides: ctx_cat buckets (offline frame-global-rank
        # convention) + the long-lookback raw HTF ctx cols (fresh full-frame
        # recompute; B28's incremental M1-lane stamping is one M5 bar behind
        # the offline convention — parity gate finding 2026-07-08)
        overrides = compute_htf_ctx_full_frame(
            cv3,
            self._state_contract,
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
        overrides: pd.DataFrame, multi_tf: dict, prebuilt_snapshot,
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
        joined = loader.get_window(
            end_ts,
            n_bars=n_from_history_start,
            snapshot=prebuilt_snapshot,
        )
        history_pos = int(cv3_idx.searchsorted(history_start, side="left"))
        expected_first = cv3_idx[history_pos] if history_pos < len(cv3_idx) else None
        if joined.empty or expected_first is None or joined.index[0] != expected_first:
            raise RuntimeError(
                f"[SMART_ENTRY] common-history window build failed: rows={len(joined)} "
                f"start={joined.index[0] if len(joined) else None} expected_start={expected_first}"
            )
        return self._builder.prepare_frame(
            joined,
            context_overrides=overrides,
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
        prebuilt_snapshot = loader.acquire_serving_snapshot()
        cv3 = prebuilt_snapshot.cv3
        if ctx is None:
            self.refresh_multi_tf(cv3)
            ctx = self._ctx
        multi_tf, overrides, _age, _spliced = self._effective_context(cv3, ctx, end_ts)
        return self._prepare_common_history_frame(
            loader,
            cv3,
            end_ts,
            overrides,
            multi_tf,
            prebuilt_snapshot,
        )

    def _multi_tf_window_tensors(
        self, ts: pd.Timestamp, multi_tf: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """Per-TF windows at-or-before ts with the BUNDLE's per-TF seq lens —
        the exact offline dataset path (EntryV10CtxDataset._get_multi_tf_window:
        slice_multi_tf_v4_window(feats, ts + 5min, n=per_tf,
        tf_shift=MULTI_TF_SHIFT)).
        `multi_tf=None` uses the current snapshot (gate/offline callers)."""
        if multi_tf is None:
            ctx = self._ctx
            if ctx is None:
                raise RuntimeError("[SMART_ENTRY] multi-TF not built — call refresh_multi_tf() first")
            multi_tf = ctx.multi_tf
        from gx1.features.htf_features import slice_multi_tf_v4_window
        out: dict[str, torch.Tensor] = {}
        availability_ts = pd.Timestamp(ts) + self._multi_tf_target_availability_shift
        missing = [tf for tf in ENTRY_MTF_CONTEXT_TIMEFRAMES if tf not in multi_tf]
        if missing:
            raise RuntimeError(
                f"[SMART_ENTRY] Entry MTF cache is incomplete: missing={missing}"
            )
        for tf in ENTRY_MTF_CONTEXT_TIMEFRAMES:
            feats = multi_tf[tf]
            n = int(self._per_tf_seq_lens[tf])
            arr = slice_multi_tf_v4_window(
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
                entry_action_q_bps = _require_finite_vector(
                    out.get("entry_action_q_bps"),
                    name="entry_action_q_bps",
                    size=3,
                    context=f"model forward at {ts}",
                )
                entry_q_joint_hidden = _require_finite_vector(
                    out.get("entry_q_joint_hidden"),
                    name="entry_q_joint_hidden",
                    size=UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM,
                    context=f"model forward at {ts}",
                )
                ssot = _entry_q_ssot(
                    entry_action_q_bps,
                    context=f"model forward at {ts}",
                )
                side_mae_bps = _require_finite_vector(
                    out.get("side_mae_bps"),
                    name="side_mae_bps",
                    size=2,
                    context=f"model forward at {ts}",
                )
                trendline_event_logits = _require_finite_vector(
                    out.get("trendline_event_logits"),
                    name="trendline_event_logits",
                    size=EXACT_TRENDLINE_EVENT_OUTPUT_DIM,
                    context=f"model forward at {ts}",
                )
                dip_pred = _require_finite_vector(
                    out.get("dip_pred"), name="dip_pred", size=DIP_HEAD_DIM, context=f"model forward at {ts}"
                )
                forecast_pred = _require_finite_vector(
                    out.get("forecast_pred"), name="forecast_pred", size=FORECAST_HEAD_DIM, context=f"model forward at {ts}"
                )
                timing_pred = _require_finite_vector(
                    out.get("timing_pred"), name="timing_pred", size=TIMING_HEAD_DIM, context=f"model forward at {ts}"
                )
                tail_risk_pred = _require_finite_vector(
                    out.get("tail_risk_pred"), name="tail_risk_pred", size=TAIL_RISK_HEAD_DIM, context=f"model forward at {ts}"
                )
                vol_forecast_pred = _require_finite_vector(
                    out.get("vol_forecast_pred"), name="vol_forecast_pred", size=VOL_FORECAST_HEAD_DIM, context=f"model forward at {ts}"
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
                tf_gate = _require_finite_vector(
                    out.get("tf_gate"),
                    name="tf_gate",
                    size=ENTRY_MTF_CONTEXT_COUNT,
                    context=f"model forward at {ts}",
                )
                if bool((tf_gate < 0.0).any()) or not np.isclose(
                    float(tf_gate.sum()), 1.0, rtol=1e-6, atol=1e-7
                ):
                    raise RuntimeError(
                        f"[SMART_ENTRY] model forward at {ts} tf_gate is not a probability simplex"
                    )
                family_tf_cooperation_gate = _require_finite_vector(
                    out.get("family_tf_cooperation_gate"),
                    name="family_tf_cooperation_gate",
                    size=ENTRY_MTF_CONTEXT_COUNT
                    * len(MODEL_NATIVE_TRAINING_SPECIALISTS),
                    context=f"model forward at {ts}",
                )
                if bool((family_tf_cooperation_gate < 0.0).any()) or not np.isclose(
                    float(family_tf_cooperation_gate.sum()),
                    1.0,
                    rtol=1e-6,
                    atol=1e-7,
                ):
                    raise RuntimeError(
                        f"[SMART_ENTRY] model forward at {ts} "
                        "family_tf_cooperation_gate is not a probability simplex"
                    )
                family_tf_feature_gate_tensor = out.get(
                    "family_tf_feature_gate"
                )
                if (
                    not isinstance(family_tf_feature_gate_tensor, torch.Tensor)
                    or tuple(family_tf_feature_gate_tensor.shape)
                    != (
                        1,
                        ENTRY_MTF_CONTEXT_COUNT,
                        len(self._meta["multi_tf"]["feature_names"]),
                    )
                    or not bool(
                        torch.isfinite(family_tf_feature_gate_tensor).all().item()
                    )
                ):
                    raise RuntimeError(
                        f"[SMART_ENTRY] model forward at {ts} "
                        "family_tf_feature_gate shape/finite contract invalid"
                    )
                family_tf_feature_gate = (
                    family_tf_feature_gate_tensor.detach()
                    .cpu()
                    .float()
                    .numpy()
                    .reshape(-1)
                )
                if family_tf_feature_gate.shape != (
                    ENTRY_MTF_CONTEXT_COUNT * MULTI_TF_FEATURE_COUNT_V4,
                ):
                    raise RuntimeError(
                        f"[SMART_ENTRY] model forward at {ts} "
                        "family_tf_feature_gate token contract invalid"
                    )
                position_size_logit = _require_finite_vector(
                    out.get("position_size_logit"),
                    name="position_size_logit",
                    size=1,
                    context=f"model forward at {ts}",
                )
                entry_decision_representation = _require_finite_vector(
                    out.get(UNIFIED_EXIT_MODEL_REPRESENTATION_KEY),
                    name=UNIFIED_EXIT_MODEL_REPRESENTATION_KEY,
                    size=UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM,
                    context=f"model forward at {ts}",
                )
                position_size_pred = _sigmoid_float(float(position_size_logit[0]))
                ctx_cat_values = np.asarray(
                    states["ctx_cat"][k],
                    dtype=np.int64,
                ).reshape(-1)
                if ctx_cat_values.shape != (len(MODEL_NATIVE_CTX_CAT_FIELDS),):
                    raise RuntimeError(
                        "[SMART_ENTRY] exact factual-session ctx_cat state is required"
                    )
                res = {
                    "time": ts,
                    "entry_action_q_bps": ssot["entry_action_q_bps"].tolist(),
                    "entry_action_q_margin_bps": ssot[
                        "entry_action_q_margin_bps"
                    ],
                    "entry_q_joint_hidden": entry_q_joint_hidden.tolist(),
                    "model_direction_index": ssot["model_direction_index"],
                    "model_direction": ssot["model_direction"],
                    "session_id": int(
                        states["ctx_cat"][k][
                            MODEL_NATIVE_CTX_CAT_INDEX_BY_NAME["session_id"]
                        ]
                    ),
                    "side_mae_bps": side_mae_bps.tolist(),
                    "trendline_event_logits": trendline_event_logits.tolist(),
                    "dip_pred": dip_pred.tolist(),
                    "forecast_pred": forecast_pred.tolist(),
                    "timing_pred": timing_pred.tolist(),
                    "tail_risk_pred": tail_risk_pred.tolist(),
                    "vol_forecast_pred": vol_forecast_pred.tolist(),
                    "specialist_names": specialist_names,
                    "specialist_gate": specialist_gate.tolist(),
                    "tf_gate": tf_gate.tolist(),
                    "family_tf_cooperation_gate": (
                        family_tf_cooperation_gate.tolist()
                    ),
                    "family_tf_feature_gate": family_tf_feature_gate.tolist(),
                    "position_size_pred": position_size_pred,
                    "position_size_logit": float(position_size_logit[0]),
                    UNIFIED_EXIT_ENTRY_REPRESENTATION_KEY: (
                        entry_decision_representation.tolist()
                    ),
                }
                results.append(res)
        return results

    def build_exit_feature_surface(
        self,
        *,
        decision_time: Any,
        prebuilt_snapshot: Any,
    ) -> Mapping[str, Any]:
        """Return the admitted causal M1 shared feature window.

        There is intentionally no reconstruction here.  Until the immutable
        M1 feature-base publisher is admitted alongside the Entry pair, Exit
        has no legal live feature surface and this method fails closed.
        """

        provider = self._exit_feature_surface_provider
        if not callable(provider):
            raise RuntimeError(
                "[SMART_EXIT] immutable M1 shared feature-base publisher is not admitted"
            )
        value = provider(
            decision_time=decision_time,
            prebuilt_snapshot=prebuilt_snapshot,
        )
        return require_m1_feature_window(
            dict(value),
            context="SMART_EXIT_PROVIDER",
        )

    def bind_admitted_m1_feature_surface(
        self,
        *,
        parquet_path: Path,
        manifest_path: Path,
        dataset_run_id: str,
        pair_generation_id: str,
        parquet_sha256: str,
        manifest_sha256: str,
        feature_field_order_sha256: str,
    ) -> None:
        """Bind one immutable M1 surface to this frozen model adapter.

        This is an explicit admission step.  The adapter does not discover a
        latest file, follow a symlink, or rebuild features when the binding is
        absent or invalid.
        """

        from gx1.execution.v12_m1_feature_surface_provider import (
            M1SharedFeatureSurfaceProvider,
        )

        provider = M1SharedFeatureSurfaceProvider.from_admitted_artifact(
            parquet_path=parquet_path,
            manifest_path=manifest_path,
            dataset_run_id=dataset_run_id,
            pair_generation_id=pair_generation_id,
            parquet_sha256=parquet_sha256,
            manifest_sha256=manifest_sha256,
            feature_field_order=self._meta["ordered_signal_names"],
            feature_field_order_sha256=feature_field_order_sha256,
        )
        self._exit_feature_surface_provider = provider

    def decide_exit(
        self,
        *,
        decision_identity: str,
        entry_snapshot: Mapping[str, Any],
        entry_decision_token_snapshot: Mapping[str, Any],
        exit_path_envelope: Mapping[str, Any],
        exit_feature_surface: Mapping[str, Any],
        entry_bid: float,
        entry_ask: float,
        side: str,
        prior_incremental_carry_envelope: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Emit same-bundle HOLD/EXIT_NOW from one frozen Entry-decision token."""

        if self._model is None or self._meta is None:
            raise RuntimeError("[SMART_EXIT] unified model is not loaded")
        if (
            not isinstance(self._bundle_sha256, str)
            or len(self._bundle_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self._bundle_sha256
            )
        ):
            raise RuntimeError("[SMART_EXIT] bundle identity is unavailable")
        if "runtime_head_evidence_schema_version" in entry_snapshot:
            snapshot = require_model_native_runtime_head_evidence(
                entry_snapshot,
                context="SMART_EXIT_ENTRY_HEAD_SNAPSHOT",
            )
        else:
            snapshot = require_model_native_runtime_evidence(
                entry_snapshot,
                context="SMART_EXIT_ENTRY_SNAPSHOT",
            )
        envelope = require_unified_exit_path_envelope(
            exit_path_envelope,
            context="SMART_EXIT",
        )
        feature_window = require_m1_feature_window(
            dict(exit_feature_surface),
            context="SMART_EXIT",
        )
        mtf_evidence = self.build_exit_mtf_feature_windows(
            decision_time=feature_window["decision_time"]
        )
        exit_mtf_windows = mtf_evidence["windows"]
        token_snapshot = require_entry_decision_token_snapshot(
            entry_decision_token_snapshot
        )
        normalization = self._meta.get("input_normalization")
        expected_normalization_sha256 = (
            normalization.get("contract_sha256")
            if isinstance(normalization, Mapping)
            else None
        )
        if (
            token_snapshot["model_identity_kind"] != "bundle_sha256"
            or token_snapshot["model_identity_sha256"] != self._bundle_sha256
            or token_snapshot["input_normalization_sha256"]
            != expected_normalization_sha256
            or token_snapshot["contract_mode"] != MODEL_NATIVE_CONTRACT_MODE
        ):
            raise RuntimeError(
                "[SMART_EXIT] frozen Entry-decision token model binding mismatch"
            )
        exit_input_envelope = build_unified_exit_input_envelope(
            decision_time=feature_window["decision_time"],
            decision_identity=decision_identity,
            side=side,
            entry_bid=entry_bid,
            entry_ask=entry_ask,
            bundle_sha256=self._bundle_sha256,
            entry_snapshot=snapshot,
            entry_decision_token_snapshot=token_snapshot,
            exit_path_envelope=envelope,
            m1_feature_window=feature_window,
            mtf_windows=exit_mtf_windows,
            mtf_cache_binding=mtf_evidence["cache_binding"],
            per_tf_seq_lens=mtf_evidence["per_tf_seq_lens"],
        )
        if side not in UNIFIED_EXIT_SIDE_ORDER:
            raise RuntimeError("[SMART_EXIT] side is outside the unified contract")
        side_index = UNIFIED_EXIT_SIDE_ORDER.index(side)
        if (
            snapshot.get("selected_side") != side_index
            or snapshot.get("model_direction")
            != ("LONG" if side_index == 0 else "SHORT")
        ):
            raise RuntimeError(
                "[SMART_EXIT] trade side differs from frozen Entry argmax"
            )
        representation = entry_decision_token_tensor(
            exit_input_envelope["entry_decision_token_snapshot"]
        )
        if (
            representation.shape != (UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM,)
            or not np.isfinite(representation).all()
        ):
            raise RuntimeError(
                "[SMART_EXIT] frozen Entry-decision token is invalid"
            )
        path_values = unified_exit_path_tensor(
            path_rows=envelope["path_rows"],
            bars_in_trade=int(envelope["bars_in_trade"]),
            entry_bid=entry_bid,
            entry_ask=entry_ask,
        )
        bars_in_trade = int(envelope["bars_in_trade"])
        if not 1 <= bars_in_trade <= UNIFIED_EXIT_MAX_PATH_BARS:
            raise RuntimeError("[SMART_EXIT] path state is outside current capacity")
        token_sha256 = canonical_unified_evidence_sha256(token_snapshot)
        previous_carry_sha256 = UNIFIED_EXIT_INCREMENTAL_CARRY_GENESIS_SHA256
        prior_carry = None
        prior_mtf_hashes: Mapping[str, str] = {}
        if prior_incremental_carry_envelope is None:
            if bars_in_trade != 1:
                raise RuntimeError("[SMART_EXIT] missing recurrent carry after state zero")
        else:
            prior = require_unified_exit_incremental_carry_envelope(
                prior_incremental_carry_envelope,
                expected_trade_identity=decision_identity,
                expected_side=side,
                expected_bundle_sha256=self._bundle_sha256,
                expected_input_normalization_sha256=(
                    expected_normalization_sha256
                ),
                expected_entry_token_snapshot_sha256=token_sha256,
                expected_step_count=bars_in_trade - 1,
            )
            if pd.Timestamp(prior["last_closed_m1_bar_ts"]) >= pd.Timestamp(
                feature_window["decision_time"]
            ):
                raise RuntimeError(
                    "[SMART_EXIT] recurrent carry clock is not forward"
                )
            previous_carry_sha256 = prior["carry_envelope_sha256"]
            prior_mtf_hashes = prior["mtf_last_row_sha256"]
            prior_carry = self._model.restore_exit_incremental_carry_tensor_state(
                step_count=int(prior["step_count"]),
                batch_size=1,
                tensors=decode_unified_exit_incremental_carry_tensors(
                    prior, device=self.device
                ),
            )

        current_mtf_hashes: dict[str, str] = {}
        incremental_mtf: dict[str, torch.Tensor] = {}
        for timeframe in EXIT_MTF_CONTEXT_TIMEFRAMES:
            tf_name = timeframe.lower()
            window = np.ascontiguousarray(
                np.asarray(exit_mtf_windows[timeframe], dtype=np.dtype("<f4"))
            )
            row = np.ascontiguousarray(window[-1], dtype=np.dtype("<f4"))
            row_sha256 = hashlib.sha256(row.tobytes(order="C")).hexdigest()
            current_mtf_hashes[tf_name] = row_sha256
            if prior_carry is None:
                new_rows = window
            elif prior_mtf_hashes.get(tf_name) == row_sha256:
                new_rows = window[:0]
            else:
                new_rows = window[-1:]
            incremental_mtf[tf_name] = torch.from_numpy(
                np.ascontiguousarray(new_rows, dtype=np.float32)
            ).unsqueeze(0).to(self.device)

        local_rows = (
            np.asarray(feature_window["signal"], dtype=np.float32)
            if prior_carry is None
            else np.asarray(feature_window["signal"][-1:], dtype=np.float32)
        )
        latest_path_row = torch.from_numpy(path_values[-1]).to(self.device)
        with torch.no_grad():
            model_output, next_carry = self._model.forward_exit_incremental_step(
                entry_decision_representation=torch.from_numpy(
                    representation
                )
                .view(1, -1)
                .to(self.device),
                exit_local_rows_x=torch.from_numpy(local_rows).unsqueeze(0).to(
                    self.device
                ),
                exit_state_ctx_cat=torch.from_numpy(
                    feature_window["ctx_cat"]
                )
                .unsqueeze(0)
                .to(self.device),
                exit_state_ctx_cont=torch.from_numpy(
                    feature_window["ctx_cont"]
                )
                .unsqueeze(0)
                .to(self.device),
                exit_path_row_x=latest_path_row.view(1, 1, -1).expand(
                    -1, 2, -1
                ),
                exit_mtf_new_rows=incremental_mtf,
                carry=prior_carry,
            )
        next_carry_envelope = build_unified_exit_incremental_carry_envelope(
            tensor_state=(
                self._model.export_exit_incremental_carry_tensor_state(
                    next_carry
                )
            ),
            step_count=int(next_carry.step_count),
            last_closed_m1_bar_ts=feature_window["decision_time"],
            trade_identity=decision_identity,
            side=side,
            bundle_sha256=self._bundle_sha256,
            input_normalization_sha256=expected_normalization_sha256,
            entry_token_snapshot_sha256=token_sha256,
            full_path_chain_sha256=envelope["full_path_chain_sha256"],
            input_envelope_sha256=exit_input_envelope[
                "input_envelope_sha256"
            ],
            previous_carry_envelope_sha256=previous_carry_sha256,
            mtf_last_row_sha256=current_mtf_hashes,
        )
        require_unified_exit_incremental_carry_envelope(
            next_carry_envelope,
            expected_input_envelope_sha256=exit_input_envelope[
                "input_envelope_sha256"
            ],
            expected_mtf_last_row_sha256=exit_input_envelope[
                "mtf_last_row_sha256"
            ],
            expected_last_closed_m1_bar_ts=feature_window["decision_time"],
            expected_step_count=bars_in_trade,
            expected_previous_carry_envelope_sha256=previous_carry_sha256,
        )
        model_output = {
            name: value[0, side_index, 0]
            for name, value in model_output.items()
            if name in {
                "exit_action_q_bps",
                "exit_action_valid_mask",
            }
        }
        q_values = _require_finite_vector(
            model_output.get("exit_action_q_bps"),
            name="exit_action_q_bps",
            size=2,
            context="unified Exit forward",
        )
        valid_mask = np.asarray(
            model_output.get("exit_action_valid_mask"), dtype=np.bool_
        ).reshape(-1)
        if (
            valid_mask.shape != (2,)
            or not valid_mask.any()
            or not bool(valid_mask[1])
            or bool(valid_mask[0]) != (len(path_values) < UNIFIED_EXIT_MAX_PATH_BARS)
        ):
            raise RuntimeError("[SMART_EXIT] invalid optimal-stopping action mask")
        valid_q = q_values[valid_mask]
        if len(valid_q) > 1 and float(valid_q[0]) == float(valid_q[1]):
            raise RuntimeError("[SMART_EXIT] tied valid Exit Q values have no decision")
        masked_q = np.where(valid_mask, q_values, -np.inf)
        action_index = int(np.argmax(masked_q))
        output = {
            "exit_action_q_bps": q_values.tolist(),
            "exit_action_valid_mask": valid_mask.tolist(),
            "exit_action_index": action_index,
            "action": UNIFIED_EXIT_ACTION_ORDER[action_index],
            "decision_source": "unified_model",
            "exit_input_envelope": exit_input_envelope,
            "exit_incremental_carry_envelope": next_carry_envelope,
            "bundle_sha256": self._bundle_sha256,
            "entry_snapshot_sha256": canonical_unified_evidence_sha256(
                snapshot
            ),
            "exit_path_envelope_sha256": (
                canonical_unified_evidence_sha256(envelope)
            ),
            "exit_input_envelope_sha256": exit_input_envelope[
                "input_envelope_sha256"
            ],
        }
        output["output_evidence_sha256"] = canonical_unified_evidence_sha256(
            output
        )
        return require_unified_exit_output(
            output,
            context="SMART_EXIT",
            expected_bundle_sha256=self._bundle_sha256,
            entry_snapshot=snapshot,
            exit_path_envelope=envelope,
            exit_input_envelope=exit_input_envelope,
        )

    def build_exit_mtf_feature_windows(
        self,
        *,
        decision_time: Any,
    ) -> dict[str, Any]:
        """Slice the exact closed M5/M15/H1/H4/D1 Exit route."""

        if self._ctx is None or self._meta is None:
            raise RuntimeError("[SMART_EXIT] complete MTF cache is unavailable")
        from gx1.contracts.entry_exit_feature_base_v1 import (
            EXIT_DECISION_BAR_SECONDS,
            EXIT_MTF_CONTEXT_TIMEFRAMES,
        )
        from gx1.features.htf_features import (
            get_model_native_multi_tf_route_windows,
        )

        windows = get_model_native_multi_tf_route_windows(
            self._ctx.multi_tf,
            decision_bar_start=pd.Timestamp(decision_time),
            per_tf_seq_lens=self._per_tf_seq_lens,
            route_timeframes=EXIT_MTF_CONTEXT_TIMEFRAMES,
            base_bar_duration=pd.Timedelta(
                seconds=EXIT_DECISION_BAR_SECONDS
            ),
        )
        mtf_meta = self._meta.get("multi_tf")
        if not isinstance(mtf_meta, Mapping):
            raise RuntimeError("[SMART_EXIT] MTF bundle binding is unavailable")
        return {
            "windows": windows,
            "cache_binding": {
                "cache_identity_sha256": mtf_meta.get(
                    "shared_cache_identity_sha256"
                ),
                "manifest_sha256": mtf_meta.get(
                    "shared_cache_manifest_sha256"
                ),
            },
            "per_tf_seq_lens": dict(self._per_tf_seq_lens),
        }

    # ── live per-M5 forward (async-context path — serving-wave gap 3) ────────

    def predict_live_bar(
        self,
        loader,
        end_ts: pd.Timestamp,
        *,
        prebuilt_snapshot,
    ) -> dict[str, Any]:
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
        cv3 = prebuilt_snapshot.cv3
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
        frame = self._prepare_common_history_frame(
            loader,
            cv3,
            end_ts,
            overrides,
            multi_tf,
            prebuilt_snapshot,
        )
        states = self._builder.build_states(frame, [end_ts])
        head = self.forward_states(states, multi_tf=multi_tf)[0]
        t = self._ctx_refresh_thread
        head["context_age_m5_bars"] = int(max(age, 0))
        head["context_cutoff_ts"] = str(ctx.cv3_cutoff)
        head["context_refresh_in_flight"] = bool(t is not None and t.is_alive())
        head["context_mtf_incremental"] = bool(spliced)
        return head

    # ── decision (operating point from the contract — ONE truth) ─────────────

    def _validated_entry_q_ssot(
        self,
        head_out: Mapping[str, Any],
    ) -> tuple[dict[str, Any], str, str]:
        """Validate the exact model-native head and return its sole direction."""

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
        ssot = _validate_reported_entry_q_ssot(dict(head_out))
        direction_index = int(ssot["model_direction_index"])
        action = MODEL_DIRECTION_ACTIONS[direction_index]
        return ssot, selection_mode, action

    def decide_direction(
        self,
        head_out: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Emit only the shared LONG/SHORT/FLAT decision surface.

        Pre-launch parity calls this same method as live ``decide``.  It does
        not consume sizing authority because sizing cannot select or rewrite
        direction.
        """

        ssot, selection_mode, action = self._validated_entry_q_ssot(head_out)
        direction_index = int(ssot["model_direction_index"])
        return {
            "action": action,
            "model_direction_index": direction_index,
            "model_direction": str(ssot["model_direction"]),
            "entry_action_q_bps": ssot["entry_action_q_bps"].tolist(),
            "entry_action_q_margin_bps": ssot[
                "entry_action_q_margin_bps"
            ],
            "selection_score_mode": selection_mode,
            "selection_score": float(
                ssot["entry_action_q_bps"][direction_index]
            ),
        }

    def decide(self, head_out: dict[str, Any], atr_bps: float) -> dict[str, Any]:
        """Emit the runner action from the model's final direction argmax.

        Raw ``entry_action_q_bps`` is validated again here so no caller can
        inject a parallel side, threshold, or session decision between model
        forward and live action.
        """

        ssot, selection_mode, action = self._validated_entry_q_ssot(head_out)
        direction_index = int(ssot["model_direction_index"])
        model_direction = str(ssot["model_direction"])
        entry_action_q_bps = ssot["entry_action_q_bps"]
        selected_side = (
            direction_index
            if direction_index in MODEL_DIRECTION_TRADE_INDICES
            else None
        )
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
        edge = float(
            max(
                entry_action_q_bps[MODEL_DIRECTION_LONG_INDEX],
                entry_action_q_bps[MODEL_DIRECTION_SHORT_INDEX],
            )
            - entry_action_q_bps[MODEL_DIRECTION_FLAT_INDEX]
        )
        selection_score = float(entry_action_q_bps[direction_index])
        atr_bps_value = float(atr_bps)
        if not np.isfinite(atr_bps_value) or atr_bps_value <= 0.0:
            raise RuntimeError(f"[SMART_ENTRY] atr_bps must be finite and positive; got {atr_bps!r}")

        diagnostics = _validate_model_native_diagnostics(
            head_out,
            MODEL_NATIVE_DECISION_DIAGNOSTIC_KEYS,
        )
        require_model_native_sizing_authority_contract(
            self._sizing_authority,
            context="[SMART_ENTRY] decision sizing",
            required_mode=MODEL_NATIVE_SIZING_MODE_LEARNED,
        )

        # Frozen Entry snapshot. Raw Q and the learned token representation are
        # the only decision evidence consumed by the same-bundle Exit owner.
        snapshot = {
            "decision_ts": str(head_out["time"]),
            "runtime_evidence_schema_version": (
                MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION
            ),
            "model_policy": MODEL_NATIVE_RUNTIME_POLICY,
            "session_id": session_id,
            "session": session,
            "entry_action_q_bps": entry_action_q_bps.tolist(),
            "entry_action_q_margin_bps": ssot["entry_action_q_margin_bps"],
            "model_direction_index": direction_index,
            "model_direction": model_direction,
            "selected_side": selected_side,
            "atr_bps": atr_bps_value,
            **diagnostics,
        }
        snapshot = require_model_native_runtime_evidence(
            snapshot,
            context="SMART_ENTRY_DECISION",
        )
        out = {
            "action": action,
            "action_id": MODEL_DIRECTION_ACTION_ID_BY_INDEX[direction_index],
            "model_direction_index": direction_index,
            "model_direction": model_direction,
            "entry_action_q_bps": entry_action_q_bps.tolist(),
            "entry_action_q_margin_bps": ssot["entry_action_q_margin_bps"],
            "edge_score": edge,
            "selection_score_mode": selection_mode,
            "selection_score": selection_score,
            "session_id": session_id,
            "session": session,
            "selected_side": selected_side,
            **diagnostics,
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
