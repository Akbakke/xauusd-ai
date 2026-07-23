"""Publish one exact offline Entry adaptation lifecycle transition.

This producer records governance only.  It never trains, mutates a bundle,
changes live weights, selects direction, submits an order, or edits the launch
state.  A failed transition publishes a newer terminal event that is invalid
for activation, preventing an older monitoring PASS from remaining current.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from gx1.contracts.entry_model_native_adaptation_drift_v1 import (
    ModelNativeAdaptationDriftError,
    adaptation_bundle_identity_from_dir,
)
from gx1.contracts.entry_model_native_adaptation_lifecycle_v1 import (
    MODEL_NATIVE_ADAPTATION_LIFECYCLE_CONTRACT,
    MODEL_NATIVE_ADAPTATION_LIFECYCLE_EVENT_PREFIX,
    MODEL_NATIVE_ADAPTATION_LIFECYCLE_MAX_ACTIVATION_AGE_SECONDS,
    MODEL_NATIVE_ADAPTATION_LIFECYCLE_SCHEMA_VERSION,
    TRANSITION_INITIAL_ADMISSION,
    adaptation_phase_for_transition,
    adaptation_transition_activates,
    load_bound_adaptation_lifecycle,
)
from gx1.contracts.entry_model_native_sizing_calibration_v1 import sha256_file
from gx1.contracts.immutable_event_authority_v1 import (
    next_immutable_event_created_utc,
    write_immutable_json_event,
)


class AdaptationLifecycleFinalizationError(RuntimeError):
    """A lifecycle transition could not establish exact evidence."""


def _binding(path: Path) -> dict[str, str]:
    raw = path.expanduser()
    if not raw.is_absolute() or raw.is_symlink() or "latest" in raw.name.lower():
        raise AdaptationLifecycleFinalizationError(
            f"event path must be absolute, immutable, and non-symlinked: {raw}"
        )
    path = raw.resolve()
    if not path.is_file():
        raise AdaptationLifecycleFinalizationError(f"event is missing: {path}")
    return {"json_path": str(path), "sha256": sha256_file(path)}


def _bundle_identity(bundle_dir: Path) -> dict[str, str]:
    try:
        return adaptation_bundle_identity_from_dir(
            bundle_dir,
            context="ADAPTATION_LIFECYCLE_FINALIZER_BUNDLE",
        )
    except ModelNativeAdaptationDriftError as exc:
        raise AdaptationLifecycleFinalizationError(str(exc)) from exc


def _created_after(
    output_dir: Path,
    *event_paths: Path | None,
) -> datetime:
    floors: list[datetime] = []
    for path in event_paths:
        if path is None:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            created = datetime.fromisoformat(
                str(payload["created_utc"]).replace("Z", "+00:00")
            ).astimezone(timezone.utc)
        except Exception as exc:
            raise AdaptationLifecycleFinalizationError(
                f"bound event has invalid created_utc: {path}"
            ) from exc
        floors.append(created)
    return next_immutable_event_created_utc(
        output_dir,
        MODEL_NATIVE_ADAPTATION_LIFECYCLE_EVENT_PREFIX,
        *floors,
    )


def _publish_terminal_failure(
    *,
    output_dir: Path,
    transition: str,
    error: Exception,
) -> None:
    write_immutable_json_event(
        output_dir,
        MODEL_NATIVE_ADAPTATION_LIFECYCLE_EVENT_PREFIX,
        {
            "schema_version": (
                "entry_model_native_adaptation_lifecycle_terminal_failure_v1"
            ),
            "created_utc": next_immutable_event_created_utc(
                output_dir,
                MODEL_NATIVE_ADAPTATION_LIFECYCLE_EVENT_PREFIX,
            ).isoformat(),
            "decision": "FAIL",
            "failures": [f"{type(error).__name__}: {error}"],
            "lifecycle_contract": MODEL_NATIVE_ADAPTATION_LIFECYCLE_CONTRACT,
            "attempted_transition": str(transition),
        },
    )


def finalize_adaptation_lifecycle_transition(
    *,
    transition: str,
    incumbent_bundle_dir: Path,
    drift_evidence_path: Path,
    replay_readiness_path: Path,
    output_dir: Path,
    predecessor_path: Path | None = None,
    candidate_bundle_dir: Path | None = None,
    shadow_evidence_path: Path | None = None,
    admission_evidence: Mapping[str, Any] | None = None,
    entry_run_id: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Publish and self-validate one newest lifecycle state transition."""

    output_dir = output_dir.expanduser().resolve()
    try:
        phase = adaptation_phase_for_transition(transition)
        activates = adaptation_transition_activates(transition)
        if transition == TRANSITION_INITIAL_ADMISSION and predecessor_path is not None:
            raise AdaptationLifecycleFinalizationError(
                "INITIAL_ADMISSION cannot name a predecessor"
            )
        if transition != TRANSITION_INITIAL_ADMISSION and predecessor_path is None:
            raise AdaptationLifecycleFinalizationError(
                "non-initial transition requires predecessor_path"
            )
        predecessor_binding = (
            None if predecessor_path is None else _binding(predecessor_path)
        )
        drift_binding = _binding(drift_evidence_path)
        replay_binding = _binding(replay_readiness_path)
        shadow_binding = (
            None if shadow_evidence_path is None else _binding(shadow_evidence_path)
        )
        incumbent = _bundle_identity(incumbent_bundle_dir)
        candidate = (
            None
            if candidate_bundle_dir is None
            else _bundle_identity(candidate_bundle_dir)
        )
        created = _created_after(
            output_dir,
            drift_evidence_path.expanduser().resolve(),
            replay_readiness_path.expanduser().resolve(),
            shadow_evidence_path.expanduser().resolve()
            if shadow_evidence_path is not None
            else None,
            predecessor_path.expanduser().resolve()
            if predecessor_path is not None
            else None,
        )
        expires = (
            created
            + timedelta(
                seconds=MODEL_NATIVE_ADAPTATION_LIFECYCLE_MAX_ACTIVATION_AGE_SECONDS
            )
            if activates
            else None
        )
        payload = {
            "schema_version": MODEL_NATIVE_ADAPTATION_LIFECYCLE_SCHEMA_VERSION,
            "created_utc": created.isoformat(),
            "decision": "PASS",
            "failures": [],
            "lifecycle_contract": MODEL_NATIVE_ADAPTATION_LIFECYCLE_CONTRACT,
            "transition": str(transition),
            "phase": phase,
            "predecessor_event": predecessor_binding,
            "incumbent_bundle": incumbent,
            "candidate_bundle": candidate,
            "drift_evidence": drift_binding,
            "replay_readiness_evidence": replay_binding,
            "shadow_evidence": shadow_binding,
            "admission_evidence": (
                dict(admission_evidence) if admission_evidence is not None else None
            ),
            "activation_authority": activates,
            "expires_utc": expires.isoformat() if expires is not None else None,
            "entry_run_id": entry_run_id,
            "adaptation_training_mode": "offline_challenger_only",
            "online_weight_updates_allowed": False,
            "post_model_direction_rules_allowed": False,
        }
        event_path, event = write_immutable_json_event(
            output_dir,
            MODEL_NATIVE_ADAPTATION_LIFECYCLE_EVENT_PREFIX,
            payload,
        )
        load_bound_adaptation_lifecycle(
            _binding(event_path),
            context="ADAPTATION_LIFECYCLE_FINALIZER_SELF_VALIDATION",
            now_utc=created,
        )
        return event_path, event
    except Exception as exc:
        try:
            _publish_terminal_failure(
                output_dir=output_dir,
                transition=transition,
                error=exc,
            )
        except Exception as publication_exc:
            exc.add_note(
                f"terminal lifecycle failure publication also failed: {publication_exc}"
            )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition", required=True)
    parser.add_argument("--incumbent-bundle-dir", type=Path, required=True)
    parser.add_argument("--candidate-bundle-dir", type=Path)
    parser.add_argument("--shadow-evidence", type=Path)
    parser.add_argument("--drift-evidence", type=Path, required=True)
    parser.add_argument("--replay-readiness", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--launch-state-json", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _launch_admission(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    raw = path.expanduser()
    if not raw.is_absolute() or raw.is_symlink() or not raw.is_file():
        raise AdaptationLifecycleFinalizationError(
            "launch state must be an explicit absolute regular file"
        )
    try:
        state = json.loads(raw.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AdaptationLifecycleFinalizationError(
            "launch state is unreadable"
        ) from exc
    if not isinstance(state, dict):
        raise AdaptationLifecycleFinalizationError(
            "launch state root must be an object"
        )
    required = {
        "serve_gate_evidence",
        "joint_exit_execution_proof_evidence",
        "sizing_runtime_parity_evidence",
    }
    if not required.issubset(state):
        raise AdaptationLifecycleFinalizationError(
            "launch state lacks complete adaptation admission evidence"
        )
    return {key: state[key] for key in sorted(required)}


def main() -> int:
    args = _parser().parse_args()
    event_path, event = finalize_adaptation_lifecycle_transition(
        transition=args.transition,
        incumbent_bundle_dir=args.incumbent_bundle_dir,
        candidate_bundle_dir=args.candidate_bundle_dir,
        shadow_evidence_path=args.shadow_evidence,
        drift_evidence_path=args.drift_evidence,
        replay_readiness_path=args.replay_readiness,
        predecessor_path=args.predecessor,
        admission_evidence=_launch_admission(args.launch_state_json),
        entry_run_id=args.run_id,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "json_path": str(event_path),
                "sha256": sha256_file(event_path),
                "transition": event["transition"],
                "phase": event["phase"],
                "activation_authority": event["activation_authority"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
