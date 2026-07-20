"""Immutable offline adaptation lifecycle for model-native Entry.

The lifecycle never changes a live direction.  It controls only whether one
exact incumbent bundle may remain launch-admissible while drift, challenger,
shadow, promotion, and rollback evidence advances through a single immutable
chain.  Online gradients and post-model direction rules are always forbidden.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from gx1.contracts.entry_model_native_adaptation_drift_v1 import (
    MODEL_NATIVE_ADAPTATION_DRIFT_RED,
    MODEL_NATIVE_ADAPTATION_DRIFT_STABLE,
    load_bound_adaptation_drift_evidence,
    require_adaptation_bundle_identity,
)
from gx1.contracts.entry_model_native_adaptation_shadow_v1 import (
    ModelNativeAdaptationShadowError,
    load_bound_adaptation_shadow_evidence,
)
from gx1.contracts.entry_model_native_readiness_v1 import artifact_fingerprints
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_CONTRACT_MODE
from gx1.contracts.entry_model_native_sizing_calibration_v1 import (
    ModelNativeSizingContractError,
    require_immutable_json_binding,
    sha256_file,
)
from gx1.contracts.entry_run_lineage_v1 import EntryRunLineageError, require_entry_run_id


MODEL_NATIVE_ADAPTATION_LIFECYCLE_SCHEMA_VERSION = (
    "entry_model_native_adaptation_lifecycle_v2"
)
MODEL_NATIVE_ADAPTATION_LIFECYCLE_EVENT_PREFIX = (
    "ENTRY_MODEL_NATIVE_ADAPTATION_LIFECYCLE"
)
MODEL_NATIVE_ADAPTATION_LIFECYCLE_CONTRACT = (
    "offline_drift_challenger_replay_shadow_promotion_rollback_v2"
)
MODEL_NATIVE_ADAPTATION_LIFECYCLE_MAX_ACTIVATION_AGE_SECONDS = 86_400

TRANSITION_INITIAL_ADMISSION = "INITIAL_ADMISSION"
TRANSITION_MONITOR_REFRESH = "MONITOR_REFRESH"
TRANSITION_DRIFT_DETECTED = "DRIFT_DETECTED"
TRANSITION_CHALLENGER_EVALUATED = "CHALLENGER_EVALUATED"
TRANSITION_SHADOW_EVALUATED = "SHADOW_EVALUATED"
TRANSITION_PROMOTE_CHALLENGER = "PROMOTE_CHALLENGER"
TRANSITION_ROLLBACK = "ROLLBACK"

PHASE_MONITORING = "MONITORING"
PHASE_DRIFT_BLOCKED = "DRIFT_BLOCKED"
PHASE_CHALLENGER_READY = "CHALLENGER_READY"
PHASE_SHADOW_READY = "SHADOW_READY"

_TRANSITION_PHASE = {
    TRANSITION_INITIAL_ADMISSION: PHASE_MONITORING,
    TRANSITION_MONITOR_REFRESH: PHASE_MONITORING,
    TRANSITION_DRIFT_DETECTED: PHASE_DRIFT_BLOCKED,
    TRANSITION_CHALLENGER_EVALUATED: PHASE_CHALLENGER_READY,
    TRANSITION_SHADOW_EVALUATED: PHASE_SHADOW_READY,
    TRANSITION_PROMOTE_CHALLENGER: PHASE_MONITORING,
    TRANSITION_ROLLBACK: PHASE_MONITORING,
}
_ACTIVATING_TRANSITIONS = frozenset(
    {
        TRANSITION_INITIAL_ADMISSION,
        TRANSITION_MONITOR_REFRESH,
        TRANSITION_PROMOTE_CHALLENGER,
        TRANSITION_ROLLBACK,
    }
)
_EVENT_KEYS = frozenset(
    {
        "schema_version",
        "created_utc",
        "json_path",
        "decision",
        "failures",
        "lifecycle_contract",
        "transition",
        "phase",
        "predecessor_event",
        "incumbent_bundle",
        "candidate_bundle",
        "drift_evidence",
        "replay_readiness_evidence",
        "shadow_evidence",
        "admission_evidence",
        "activation_authority",
        "expires_utc",
        "entry_run_id",
        "adaptation_training_mode",
        "online_weight_updates_allowed",
        "post_model_direction_rules_allowed",
    }
)
_BINDING_KEYS = frozenset({"json_path", "sha256"})
_ADMISSION_KEYS = frozenset(
    {
        "serve_gate_evidence",
        "joint_exit_execution_proof_evidence",
        "sizing_runtime_parity_evidence",
    }
)
_SERVE_EVIDENCE_KEYS = frozenset(
    {"model_native_serve_parity", "model_native_direction_pocket_audit"}
)
_HANDOFF_KEYS = frozenset(
    {
        "schema_version",
        "activation_authority",
        "required_next_contract",
        "required_transitions",
        "online_weight_updates_allowed",
        "post_model_direction_rules_allowed",
    }
)
MODEL_NATIVE_ADAPTATION_TRANSITIONS = (
    TRANSITION_INITIAL_ADMISSION,
    TRANSITION_MONITOR_REFRESH,
    TRANSITION_DRIFT_DETECTED,
    TRANSITION_CHALLENGER_EVALUATED,
    TRANSITION_SHADOW_EVALUATED,
    TRANSITION_PROMOTE_CHALLENGER,
    TRANSITION_ROLLBACK,
)
MODEL_NATIVE_REPLAY_READINESS_REQUIRED_ARTIFACTS = frozenset(
    {
        "candidate_readiness",
        "candidate_bundle_audit",
        "selective_edge_report",
        "selective_edge_metrics",
        "candidate_replay_report",
        "candidate_replay_metrics",
        "candidate_replay_monthly",
        "candidate_replay_trades",
        "pretrain_audit",
        "selective_edge_authoritative_predictions",
    }
)


def adaptation_lifecycle_handoff_metadata() -> dict[str, Any]:
    """Return the one exact non-activating replay-to-lifecycle handoff."""

    return {
        "schema_version": "entry_model_native_adaptation_handoff_v1",
        "activation_authority": False,
        "required_next_contract": MODEL_NATIVE_ADAPTATION_LIFECYCLE_CONTRACT,
        "required_transitions": list(MODEL_NATIVE_ADAPTATION_TRANSITIONS),
        "online_weight_updates_allowed": False,
        "post_model_direction_rules_allowed": False,
    }


class ModelNativeAdaptationLifecycleError(RuntimeError):
    """The adaptation lifecycle is absent, stale, or transition-invalid."""


def _fail(context: str, detail: str) -> None:
    raise ModelNativeAdaptationLifecycleError(f"[{context}_INVALID] {detail}")


def _exact_keys(
    value: Mapping[str, Any] | Any,
    expected: frozenset[str],
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(context, "expected an object")
    observed = dict(value)
    missing = sorted(expected - set(observed))
    unexpected = sorted(set(observed) - expected)
    if missing or unexpected:
        _fail(context, f"exact keys mismatch: missing={missing} unexpected={unexpected}")
    return observed


def _sha(value: Any, *, context: str) -> str:
    parsed = str(value or "").strip().lower()
    if len(parsed) != 64 or any(ch not in "0123456789abcdef" for ch in parsed):
        _fail(context, "not an exact SHA-256")
    return parsed


def _utc(value: Any, *, context: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:
        raise ModelNativeAdaptationLifecycleError(
            f"[{context}_INVALID] invalid UTC timestamp"
        ) from exc
    if parsed.tz is None or parsed.utcoffset() is None:
        _fail(context, "timestamp must be timezone-aware UTC")
    return parsed.tz_convert("UTC")


def _binding(
    value: Mapping[str, Any] | Any,
    *,
    context: str,
) -> dict[str, str]:
    observed = _exact_keys(value, _BINDING_KEYS, context=context)
    raw = Path(str(observed["json_path"] or "")).expanduser()
    if not raw.is_absolute() or raw.is_symlink() or "latest" in raw.name.lower():
        _fail(context, "json_path must be absolute, immutable, and non-symlinked")
    path = raw.resolve()
    expected_sha = _sha(observed["sha256"], context=f"{context}.sha256")
    if not path.is_file():
        _fail(context, f"bound event missing: {path}")
    if sha256_file(path) != expected_sha:
        _fail(context, f"bound event hash mismatch: {path}")
    return {"json_path": str(path), "sha256": expected_sha}


def _admission_evidence(
    value: Mapping[str, Any] | Any,
    *,
    context: str,
) -> dict[str, Any]:
    observed = _exact_keys(value, _ADMISSION_KEYS, context=context)
    serve = _exact_keys(
        observed["serve_gate_evidence"],
        _SERVE_EVIDENCE_KEYS,
        context=f"{context}.serve_gate_evidence",
    )
    canonical_serve = {
        name: _binding(
            serve[name], context=f"{context}.serve_gate_evidence.{name}"
        )
        for name in sorted(_SERVE_EVIDENCE_KEYS)
    }
    return {
        "serve_gate_evidence": canonical_serve,
        "joint_exit_execution_proof_evidence": _binding(
            observed["joint_exit_execution_proof_evidence"],
            context=f"{context}.joint_exit_execution_proof_evidence",
        ),
        "sizing_runtime_parity_evidence": _binding(
            observed["sizing_runtime_parity_evidence"],
            context=f"{context}.sizing_runtime_parity_evidence",
        ),
    }


def _load_replay_readiness(
    value: Mapping[str, Any] | Any,
    *,
    bundle_identity: Mapping[str, str],
    context: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        binding = require_immutable_json_binding(
            value,
            event_prefix="ENTRY_REPLAY_READINESS",
            context=f"{context}.binding",
            verify_file=True,
        )
    except ModelNativeSizingContractError as exc:
        raise ModelNativeAdaptationLifecycleError(str(exc)) from exc
    path = Path(binding["json_path"])
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelNativeAdaptationLifecycleError(
            f"[{context}_INVALID] replay readiness is unreadable"
        ) from exc
    if not isinstance(report, dict):
        _fail(context, "replay readiness root is not an object")
    if (
        report.get("schema_version") != "entry_replay_readiness_model_native_v2"
        or report.get("decision") != "READY_FOR_MODEL_NATIVE_REPLAY_REVIEW"
        or report.get("model_native_replay_evidence_ready") is not True
        or report.get("secondary_direction_authority_allowed") is not False
        or report.get("promotion_shadow_live_allowed") is not False
        or report.get("failures") != []
        or report.get("contract_mode") != MODEL_NATIVE_CONTRACT_MODE
    ):
        _fail(context, "replay readiness is not an exact zero-failure model-native PASS")
    handoff = _exact_keys(
        report.get("adaptation_lifecycle_handoff"),
        _HANDOFF_KEYS,
        context=f"{context}.adaptation_lifecycle_handoff",
    )
    if handoff != adaptation_lifecycle_handoff_metadata():
        _fail(context, "replay readiness adaptation handoff mismatch")
    replay_bundle = require_adaptation_bundle_identity(
        report.get("bundle_identity"),
        context=f"{context}.bundle_identity",
    )
    if replay_bundle != bundle_identity:
        _fail(context, "replay readiness byte identity differs from bundle")
    identity = report.get("evidence_identity")
    if not isinstance(identity, dict):
        _fail(context, "replay readiness lacks evidence_identity")
    expected_bundle = bundle_identity["bundle_dir"]
    for field in (
        "candidate_bundle_dir",
        "selective_edge_bundle_dir",
        "replay_identity_candidate_bundle_dir",
    ):
        if Path(str(identity.get(field) or "")).expanduser().resolve() != Path(expected_bundle):
            _fail(context, f"replay readiness {field} differs from bundle")
    if identity.get("replay_identity_ready") is not True:
        _fail(context, "replay identity is not ready")
    artifacts = report.get("artifacts")
    fingerprints = report.get("artifact_fingerprints")
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != set(MODEL_NATIVE_REPLAY_READINESS_REQUIRED_ARTIFACTS)
        or not isinstance(fingerprints, dict)
    ):
        _fail(context, "replay readiness artifact inventory is absent")
    recomputed = artifact_fingerprints(
        {str(name): str(path) for name, path in artifacts.items()}
    )
    if recomputed != fingerprints:
        _fail(context, "replay readiness artifact fingerprints changed")
    return report, binding


def _historical_lifecycle_binding(
    value: Mapping[str, Any] | Any,
    *,
    current_path: Path,
    context: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    binding = _binding(value, context=context)
    path = Path(binding["json_path"])
    if path.parent != current_path.parent or path.name >= current_path.name:
        _fail(context, "predecessor must be an older event in the same family directory")
    inventory = sorted(
        candidate.resolve()
        for candidate in current_path.parent.glob(
            f"{MODEL_NATIVE_ADAPTATION_LIFECYCLE_EVENT_PREFIX}_*.json"
        )
        if candidate.is_file() and not candidate.is_symlink()
    )
    try:
        position = inventory.index(current_path)
    except ValueError:
        _fail(context, "current lifecycle event is absent from its family inventory")
    if position <= 0 or inventory[position - 1] != path:
        _fail(context, "predecessor is not the immediate prior lifecycle event")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelNativeAdaptationLifecycleError(
            f"[{context}_INVALID] predecessor event is unreadable"
        ) from exc
    predecessor = _exact_keys(raw, _EVENT_KEYS, context=context)
    if predecessor["schema_version"] != MODEL_NATIVE_ADAPTATION_LIFECYCLE_SCHEMA_VERSION:
        _fail(context, "predecessor schema mismatch")
    if Path(str(predecessor["json_path"] or "")).expanduser().resolve() != path:
        _fail(context, "predecessor self-reference mismatch")
    return predecessor, binding


def _bundle_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return dict(left) == dict(right)


def adaptation_phase_for_transition(transition: str) -> str:
    """Return the one code-owned phase for a lifecycle transition."""

    phase = _TRANSITION_PHASE.get(str(transition))
    if phase is None:
        raise ValueError(f"unknown adaptation transition: {transition!r}")
    return phase


def adaptation_transition_activates(transition: str) -> bool:
    """Return whether a transition may carry fresh launch authority."""

    adaptation_phase_for_transition(transition)
    return str(transition) in _ACTIVATING_TRANSITIONS


def _rollback_target_appeared_earlier(
    *,
    current_path: Path,
    incumbent: Mapping[str, Any],
    context: str,
) -> bool:
    for path in sorted(
        candidate.resolve()
        for candidate in current_path.parent.glob(
            f"{MODEL_NATIVE_ADAPTATION_LIFECYCLE_EVENT_PREFIX}_*.json"
        )
        if candidate.is_file()
        and not candidate.is_symlink()
        and candidate.name < current_path.name
    ):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ModelNativeAdaptationLifecycleError(
                f"[{context}_INVALID] earlier lifecycle event is unreadable: {path}"
            ) from exc
        earlier = _exact_keys(raw, _EVENT_KEYS, context=f"{context}.{path.name}")
        if earlier["incumbent_bundle"] == incumbent:
            return True
    return False


def _validate_transition(
    event: Mapping[str, Any],
    predecessor: Mapping[str, Any] | None,
    *,
    drift_decision: str,
    context: str,
) -> None:
    transition = str(event["transition"])
    incumbent = event["incumbent_bundle"]
    candidate = event["candidate_bundle"]
    if transition == TRANSITION_INITIAL_ADMISSION:
        if predecessor is not None or candidate is not None or drift_decision != MODEL_NATIVE_ADAPTATION_DRIFT_STABLE:
            _fail(context, "INITIAL_ADMISSION requires no predecessor/candidate and STABLE drift")
        return
    if predecessor is None:
        _fail(context, "non-initial transition requires an immediate predecessor")
    predecessor_incumbent = predecessor["incumbent_bundle"]
    predecessor_candidate = predecessor["candidate_bundle"]
    if transition == TRANSITION_MONITOR_REFRESH:
        if (
            predecessor["phase"] != PHASE_MONITORING
            or not _bundle_equal(incumbent, predecessor_incumbent)
            or candidate is not None
            or drift_decision != MODEL_NATIVE_ADAPTATION_DRIFT_STABLE
        ):
            _fail(context, "MONITOR_REFRESH transition mismatch")
    elif transition == TRANSITION_DRIFT_DETECTED:
        if (
            predecessor["phase"] != PHASE_MONITORING
            or not _bundle_equal(incumbent, predecessor_incumbent)
            or candidate is not None
            or drift_decision != MODEL_NATIVE_ADAPTATION_DRIFT_RED
        ):
            _fail(context, "DRIFT_DETECTED transition mismatch")
    elif transition == TRANSITION_CHALLENGER_EVALUATED:
        if (
            predecessor["phase"] != PHASE_DRIFT_BLOCKED
            or not _bundle_equal(incumbent, predecessor_incumbent)
            or candidate is None
            or _bundle_equal(candidate, incumbent)
            or drift_decision != MODEL_NATIVE_ADAPTATION_DRIFT_RED
        ):
            _fail(context, "CHALLENGER_EVALUATED transition mismatch")
    elif transition == TRANSITION_SHADOW_EVALUATED:
        if (
            predecessor["phase"] != PHASE_CHALLENGER_READY
            or not _bundle_equal(incumbent, predecessor_incumbent)
            or candidate is None
            or not _bundle_equal(candidate, predecessor_candidate)
            or drift_decision != MODEL_NATIVE_ADAPTATION_DRIFT_STABLE
        ):
            _fail(context, "SHADOW_EVALUATED transition mismatch")
    elif transition == TRANSITION_PROMOTE_CHALLENGER:
        if (
            predecessor["phase"] != PHASE_SHADOW_READY
            or predecessor_candidate is None
            or not _bundle_equal(incumbent, predecessor_candidate)
            or candidate is not None
            or drift_decision != MODEL_NATIVE_ADAPTATION_DRIFT_STABLE
        ):
            _fail(context, "PROMOTE_CHALLENGER transition mismatch")
    elif transition == TRANSITION_ROLLBACK:
        if (
            predecessor["phase"] != PHASE_MONITORING
            or _bundle_equal(incumbent, predecessor_incumbent)
            or candidate is not None
            or drift_decision != MODEL_NATIVE_ADAPTATION_DRIFT_STABLE
        ):
            _fail(context, "ROLLBACK transition mismatch")
    else:
        _fail(context, f"unknown transition {transition!r}")


def load_bound_adaptation_lifecycle(
    binding: Mapping[str, Any] | Any,
    *,
    context: str,
    now_utc: Any | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load newest lifecycle authority and validate its immediate transition."""

    try:
        canonical_binding = require_immutable_json_binding(
            binding,
            event_prefix=MODEL_NATIVE_ADAPTATION_LIFECYCLE_EVENT_PREFIX,
            context=f"{context}.binding",
            verify_file=True,
        )
    except ModelNativeSizingContractError as exc:
        raise ModelNativeAdaptationLifecycleError(str(exc)) from exc
    path = Path(canonical_binding["json_path"])
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelNativeAdaptationLifecycleError(
            f"[{context}_INVALID] lifecycle event is unreadable"
        ) from exc
    event = _exact_keys(raw, _EVENT_KEYS, context=context)
    if event["schema_version"] != MODEL_NATIVE_ADAPTATION_LIFECYCLE_SCHEMA_VERSION:
        _fail(context, "schema_version mismatch")
    if event["lifecycle_contract"] != MODEL_NATIVE_ADAPTATION_LIFECYCLE_CONTRACT:
        _fail(context, "lifecycle contract mismatch")
    if event["decision"] != "PASS" or event["failures"] != []:
        _fail(context, "lifecycle event must be zero-failure PASS")
    if Path(str(event["json_path"] or "")).expanduser().resolve() != path:
        _fail(context, "json_path self-reference mismatch")
    transition = str(event["transition"])
    if event["phase"] != _TRANSITION_PHASE.get(transition):
        _fail(context, "phase does not match transition")
    if (
        event["adaptation_training_mode"] != "offline_challenger_only"
        or event["online_weight_updates_allowed"] is not False
        or event["post_model_direction_rules_allowed"] is not False
    ):
        _fail(context, "online updates or post-model direction rules are forbidden")

    incumbent = require_adaptation_bundle_identity(
        event["incumbent_bundle"],
        context=f"{context}.incumbent_bundle",
    )
    if incumbent != event["incumbent_bundle"]:
        _fail(context, "incumbent identity canonicalization mismatch")
    candidate = None
    if event["candidate_bundle"] is not None:
        candidate = require_adaptation_bundle_identity(
            event["candidate_bundle"],
            context=f"{context}.candidate_bundle",
        )
        if candidate != event["candidate_bundle"]:
            _fail(context, "candidate identity canonicalization mismatch")

    predecessor = None
    if event["predecessor_event"] is not None:
        predecessor, predecessor_binding = _historical_lifecycle_binding(
            event["predecessor_event"],
            current_path=path,
            context=f"{context}.predecessor",
        )
        if predecessor_binding != event["predecessor_event"]:
            _fail(context, "predecessor binding canonicalization mismatch")
        if _utc(predecessor["created_utc"], context=f"{context}.predecessor.created") >= _utc(event["created_utc"], context=f"{context}.created"):
            _fail(context, "predecessor chronology is invalid")
    elif transition != TRANSITION_INITIAL_ADMISSION:
        _fail(context, "non-initial lifecycle event lacks predecessor")

    drift, drift_binding = load_bound_adaptation_drift_evidence(
        event["drift_evidence"],
        context=f"{context}.drift_evidence",
        now_utc=event["created_utc"],
    )
    drift_bundle = drift["bundle_identity"]
    expected_drift_bundle = candidate if transition == TRANSITION_SHADOW_EVALUATED else incumbent
    if expected_drift_bundle is None or drift_bundle != expected_drift_bundle:
        _fail(context, "drift evidence bundle differs from transition bundle")
    if drift_binding != event["drift_evidence"]:
        _fail(context, "drift binding canonicalization mismatch")

    replay_identity = candidate if transition == TRANSITION_CHALLENGER_EVALUATED else incumbent
    if transition == TRANSITION_SHADOW_EVALUATED:
        replay_identity = candidate
    _, replay_binding = _load_replay_readiness(
        event["replay_readiness_evidence"],
        bundle_identity=replay_identity,
        context=f"{context}.replay_readiness",
    )
    if replay_binding != event["replay_readiness_evidence"]:
        _fail(context, "replay readiness binding canonicalization mismatch")

    _validate_transition(
        event,
        predecessor,
        drift_decision=drift["decision"],
        context=f"{context}.transition",
    )

    shadow_required = transition in {
        TRANSITION_SHADOW_EVALUATED,
        TRANSITION_PROMOTE_CHALLENGER,
    }
    if shadow_required:
        if predecessor is None:
            _fail(context, "shadow transition lacks predecessor")
        if transition == TRANSITION_SHADOW_EVALUATED:
            shadow_incumbent = incumbent
            shadow_candidate = candidate
        else:
            shadow_incumbent = require_adaptation_bundle_identity(
                predecessor["incumbent_bundle"],
                context=f"{context}.shadow_predecessor_incumbent",
            )
            shadow_candidate = incumbent
            if event["shadow_evidence"] != predecessor["shadow_evidence"]:
                _fail(context, "promotion does not preserve paired shadow evidence")
        if shadow_candidate is None:
            _fail(context, "shadow transition lacks candidate identity")
        try:
            _, shadow_binding = load_bound_adaptation_shadow_evidence(
                event["shadow_evidence"],
                incumbent_bundle=shadow_incumbent,
                candidate_bundle=shadow_candidate,
                context=f"{context}.shadow_evidence",
                now_utc=event["created_utc"],
            )
        except ModelNativeAdaptationShadowError as exc:
            raise ModelNativeAdaptationLifecycleError(str(exc)) from exc
        if shadow_binding != event["shadow_evidence"]:
            _fail(context, "shadow evidence binding canonicalization mismatch")
    elif event["shadow_evidence"] is not None:
        _fail(context, "non-shadow transition cannot carry shadow evidence")

    activating = transition in _ACTIVATING_TRANSITIONS
    if event["activation_authority"] is not activating:
        _fail(context, "activation_authority differs from transition")
    if activating:
        admission = _admission_evidence(
            event["admission_evidence"], context=f"{context}.admission_evidence"
        )
        if admission != event["admission_evidence"]:
            _fail(context, "admission evidence canonicalization mismatch")
        try:
            run_id = require_entry_run_id(event["entry_run_id"])
        except EntryRunLineageError as exc:
            _fail(context, str(exc))
        if run_id != event["entry_run_id"]:
            _fail(context, "activating transition run lineage is not canonical")
        created = _utc(event["created_utc"], context=f"{context}.created_utc")
        expires = _utc(event["expires_utc"], context=f"{context}.expires_utc")
        if (expires - created).total_seconds() != MODEL_NATIVE_ADAPTATION_LIFECYCLE_MAX_ACTIVATION_AGE_SECONDS:
            _fail(context, "activation expiry must be exact")
        now = _utc(
            pd.Timestamp.now(tz="UTC") if now_utc is None else now_utc,
            context=f"{context}.now_utc",
        )
        if now < created or now > expires:
            _fail(context, "activation authority is not fresh")
    else:
        if (
            event["admission_evidence"] is not None
            or event["expires_utc"] is not None
        ):
            _fail(context, "non-activating transition cannot carry admission/expiry")
        if transition == TRANSITION_DRIFT_DETECTED and event["entry_run_id"] is not None:
            _fail(context, "DRIFT_DETECTED cannot invent a run_id")
        if transition in {TRANSITION_CHALLENGER_EVALUATED, TRANSITION_SHADOW_EVALUATED}:
            try:
                run_id = require_entry_run_id(event["entry_run_id"])
            except EntryRunLineageError as exc:
                _fail(context, str(exc))
            if run_id != event["entry_run_id"]:
                _fail(context, "challenger/shadow run lineage is not canonical")

    if transition == TRANSITION_ROLLBACK and not _rollback_target_appeared_earlier(
        current_path=path,
        incumbent=incumbent,
        context=f"{context}.rollback_history",
    ):
        _fail(context, "ROLLBACK target was never an earlier incumbent")
    return event, canonical_binding


def require_launch_adaptation_authority(
    binding: Mapping[str, Any] | Any,
    *,
    accepted_bundle: Path,
    serve_gate_evidence: Mapping[str, Any],
    joint_exit_execution_proof_evidence: Mapping[str, Any],
    sizing_runtime_parity_evidence: Mapping[str, Any],
    context: str,
    now_utc: Any | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Cross-bind newest fresh lifecycle authority to the launch evidence."""

    event, canonical_binding = load_bound_adaptation_lifecycle(
        binding, context=context, now_utc=now_utc
    )
    if event["activation_authority"] is not True or event["phase"] != PHASE_MONITORING:
        _fail(context, "newest lifecycle phase has no launch activation authority")
    if Path(event["incumbent_bundle"]["bundle_dir"]).resolve() != accepted_bundle.resolve():
        _fail(context, "lifecycle incumbent differs from accepted launch bundle")
    expected_admission = {
        "serve_gate_evidence": dict(serve_gate_evidence),
        "joint_exit_execution_proof_evidence": dict(
            joint_exit_execution_proof_evidence
        ),
        "sizing_runtime_parity_evidence": dict(sizing_runtime_parity_evidence),
    }
    if event["admission_evidence"] != expected_admission:
        _fail(context, "lifecycle admission evidence differs from launch evidence")
    return event, canonical_binding


__all__ = [
    "MODEL_NATIVE_ADAPTATION_LIFECYCLE_CONTRACT",
    "MODEL_NATIVE_ADAPTATION_LIFECYCLE_EVENT_PREFIX",
    "MODEL_NATIVE_ADAPTATION_TRANSITIONS",
    "MODEL_NATIVE_ADAPTATION_LIFECYCLE_SCHEMA_VERSION",
    "MODEL_NATIVE_REPLAY_READINESS_REQUIRED_ARTIFACTS",
    "PHASE_CHALLENGER_READY",
    "PHASE_DRIFT_BLOCKED",
    "PHASE_MONITORING",
    "PHASE_SHADOW_READY",
    "TRANSITION_CHALLENGER_EVALUATED",
    "TRANSITION_DRIFT_DETECTED",
    "TRANSITION_INITIAL_ADMISSION",
    "TRANSITION_MONITOR_REFRESH",
    "TRANSITION_PROMOTE_CHALLENGER",
    "TRANSITION_ROLLBACK",
    "TRANSITION_SHADOW_EVALUATED",
    "ModelNativeAdaptationLifecycleError",
    "adaptation_lifecycle_handoff_metadata",
    "adaptation_phase_for_transition",
    "adaptation_transition_activates",
    "load_bound_adaptation_lifecycle",
    "require_launch_adaptation_authority",
]
