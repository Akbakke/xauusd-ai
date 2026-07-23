from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gx1.contracts.entry_model_native_adaptation_lifecycle_v1 import (
    PHASE_DRIFT_BLOCKED,
    PHASE_MONITORING,
    PHASE_SHADOW_READY,
    TRANSITION_CHALLENGER_EVALUATED,
    TRANSITION_DRIFT_DETECTED,
    TRANSITION_INITIAL_ADMISSION,
    TRANSITION_MONITOR_REFRESH,
    TRANSITION_PROMOTE_CHALLENGER,
    TRANSITION_ROLLBACK,
    TRANSITION_SHADOW_EVALUATED,
    ModelNativeAdaptationLifecycleError,
    load_bound_adaptation_lifecycle,
    require_launch_adaptation_authority,
)
from gx1.scripts.finalize_entry_model_native_adaptation_lifecycle_v1 import (
    finalize_adaptation_lifecycle_transition,
)
from tests.model_native_adaptation_support import (
    event_binding,
    write_adaptation_bundle,
    write_adaptation_drift,
    write_adaptation_shadow,
    write_initial_adaptation_lifecycle,
    write_replay_readiness,
)


def _admission(root: Path, name: str) -> dict[str, object]:
    bindings: list[dict[str, str]] = []
    for index in range(4):
        path = root / f"{name}_evidence_{index}_20260720T120000123456Z.json"
        path.write_text("{\"decision\":\"PASS\"}\n", encoding="utf-8")
        bindings.append(event_binding(path))
    return {
        "serve_gate_evidence": {
            "model_native_serve_parity": bindings[0],
            "model_native_direction_pocket_audit": bindings[1],
        },
        "joint_exit_execution_proof_evidence": bindings[2],
        "sizing_runtime_parity_evidence": bindings[3],
    }


def test_initial_lifecycle_is_fresh_and_cross_bound_to_launch_evidence(
    tmp_path: Path,
) -> None:
    bundle, identity = write_adaptation_bundle(tmp_path, "incumbent")
    admission = _admission(tmp_path, "incumbent")
    lifecycle = write_initial_adaptation_lifecycle(
        tmp_path,
        bundle=bundle,
        identity=identity,
        admission_evidence=admission,
    )
    event, binding = load_bound_adaptation_lifecycle(
        lifecycle["artifact"],
        context="UNIT_INITIAL_ADAPTATION",
        now_utc=lifecycle["event"]["created_utc"],
    )

    assert binding == lifecycle["artifact"]
    assert event["phase"] == PHASE_MONITORING
    assert event["activation_authority"] is True
    launch_event, launch_binding = require_launch_adaptation_authority(
        lifecycle["artifact"],
        accepted_bundle=bundle,
        serve_gate_evidence=admission["serve_gate_evidence"],
        joint_exit_execution_proof_evidence=admission[
            "joint_exit_execution_proof_evidence"
        ],
        sizing_runtime_parity_evidence=admission[
            "sizing_runtime_parity_evidence"
        ],
        context="UNIT_INITIAL_ADAPTATION_LAUNCH",
        now_utc=event["created_utc"],
    )
    assert launch_binding == binding
    assert launch_event["incumbent_bundle"] == identity

    stale = pd.Timestamp(event["expires_utc"]) + pd.Timedelta(microseconds=1)
    with pytest.raises(
        ModelNativeAdaptationLifecycleError,
        match="activation authority is not fresh",
    ):
        load_bound_adaptation_lifecycle(
            lifecycle["artifact"],
            context="UNIT_EXPIRED_ADAPTATION",
            now_utc=stale,
        )

    refreshed_drift = write_adaptation_drift(
        tmp_path,
        bundle=bundle,
        identity=identity,
        family="monitor_refresh_drift",
    )
    refresh_path, refresh = finalize_adaptation_lifecycle_transition(
        transition=TRANSITION_MONITOR_REFRESH,
        incumbent_bundle_dir=bundle,
        drift_evidence_path=refreshed_drift["event_path"],
        replay_readiness_path=lifecycle["replay"]["event_path"],
        predecessor_path=lifecycle["event_path"],
        output_dir=tmp_path / "adaptation_lifecycle",
        admission_evidence=admission,
        entry_run_id="UNIT_MONITOR_REFRESH",
    )
    refreshed, _ = load_bound_adaptation_lifecycle(
        event_binding(refresh_path),
        context="UNIT_MONITOR_REFRESH",
        now_utc=refresh["created_utc"],
    )
    assert refreshed["transition"] == TRANSITION_MONITOR_REFRESH
    assert refreshed["phase"] == PHASE_MONITORING
    assert refreshed["activation_authority"] is True


def test_complete_drift_challenger_shadow_promotion_and_rollback_chain(
    tmp_path: Path,
) -> None:
    incumbent, incumbent_identity = write_adaptation_bundle(tmp_path, "incumbent")
    challenger, challenger_identity = write_adaptation_bundle(tmp_path, "challenger")
    incumbent_admission = _admission(tmp_path, "incumbent")
    challenger_admission = _admission(tmp_path, "challenger")
    lifecycle_dir = tmp_path / "adaptation_lifecycle"

    initial = write_initial_adaptation_lifecycle(
        tmp_path,
        bundle=incumbent,
        identity=incumbent_identity,
        admission_evidence=incumbent_admission,
    )
    incumbent_replay = initial["replay"]
    red_drift = write_adaptation_drift(
        tmp_path,
        bundle=incumbent,
        identity=incumbent_identity,
        loss_long=True,
        family="incumbent_red_drift",
    )
    drift_path, drift_event = finalize_adaptation_lifecycle_transition(
        transition=TRANSITION_DRIFT_DETECTED,
        incumbent_bundle_dir=incumbent,
        drift_evidence_path=red_drift["event_path"],
        replay_readiness_path=incumbent_replay["event_path"],
        predecessor_path=initial["event_path"],
        output_dir=lifecycle_dir,
    )
    assert drift_event["phase"] == PHASE_DRIFT_BLOCKED
    assert drift_event["activation_authority"] is False

    challenger_replay = write_replay_readiness(
        tmp_path, bundle=challenger, family="challenger_replay"
    )
    challenger_path, challenger_event = finalize_adaptation_lifecycle_transition(
        transition=TRANSITION_CHALLENGER_EVALUATED,
        incumbent_bundle_dir=incumbent,
        candidate_bundle_dir=challenger,
        drift_evidence_path=red_drift["event_path"],
        replay_readiness_path=challenger_replay["event_path"],
        predecessor_path=drift_path,
        entry_run_id="UNIT_OFFLINE_CHALLENGER_V1",
        output_dir=lifecycle_dir,
    )
    assert challenger_event["activation_authority"] is False

    stable_challenger = write_adaptation_drift(
        tmp_path,
        bundle=challenger,
        identity=challenger_identity,
        family="challenger_shadow_drift",
    )
    paired_shadow = write_adaptation_shadow(
        tmp_path,
        incumbent_bundle=incumbent,
        incumbent_identity=incumbent_identity,
        candidate_bundle=challenger,
        candidate_identity=challenger_identity,
    )
    shadow_path, shadow_event = finalize_adaptation_lifecycle_transition(
        transition=TRANSITION_SHADOW_EVALUATED,
        incumbent_bundle_dir=incumbent,
        candidate_bundle_dir=challenger,
        drift_evidence_path=stable_challenger["event_path"],
        replay_readiness_path=challenger_replay["event_path"],
        shadow_evidence_path=paired_shadow["event_path"],
        predecessor_path=challenger_path,
        entry_run_id="UNIT_OFFLINE_CHALLENGER_V1",
        output_dir=lifecycle_dir,
    )
    assert shadow_event["phase"] == PHASE_SHADOW_READY
    shadow_created = pd.Timestamp(shadow_event["created_utc"])
    assert shadow_created > pd.Timestamp(
        paired_shadow["event"]["created_utc"]
    )
    assert shadow_created > pd.Timestamp(
        challenger_replay["event"]["created_utc"]
    )

    promotion_path, promotion = finalize_adaptation_lifecycle_transition(
        transition=TRANSITION_PROMOTE_CHALLENGER,
        incumbent_bundle_dir=challenger,
        drift_evidence_path=stable_challenger["event_path"],
        replay_readiness_path=challenger_replay["event_path"],
        shadow_evidence_path=paired_shadow["event_path"],
        predecessor_path=shadow_path,
        admission_evidence=challenger_admission,
        entry_run_id="UNIT_PROMOTE_CHALLENGER_V1",
        output_dir=lifecycle_dir,
    )
    assert promotion["phase"] == PHASE_MONITORING
    assert promotion["incumbent_bundle"] == challenger_identity
    assert promotion["activation_authority"] is True

    stable_rollback = write_adaptation_drift(
        tmp_path,
        bundle=incumbent,
        identity=incumbent_identity,
        family="rollback_incumbent_drift",
    )
    rollback_path, rollback = finalize_adaptation_lifecycle_transition(
        transition=TRANSITION_ROLLBACK,
        incumbent_bundle_dir=incumbent,
        drift_evidence_path=stable_rollback["event_path"],
        replay_readiness_path=incumbent_replay["event_path"],
        predecessor_path=promotion_path,
        admission_evidence=incumbent_admission,
        entry_run_id="UNIT_ROLLBACK_V1",
        output_dir=lifecycle_dir,
    )
    loaded, _ = load_bound_adaptation_lifecycle(
        event_binding(rollback_path),
        context="UNIT_ADAPTATION_ROLLBACK",
        now_utc=rollback["created_utc"],
    )
    assert loaded["transition"] == TRANSITION_ROLLBACK
    assert loaded["incumbent_bundle"] == incumbent_identity
    assert loaded["activation_authority"] is True


def test_premature_promotion_publishes_newer_terminal_failure(tmp_path: Path) -> None:
    incumbent, incumbent_identity = write_adaptation_bundle(tmp_path, "incumbent")
    challenger, challenger_identity = write_adaptation_bundle(tmp_path, "challenger")
    initial = write_initial_adaptation_lifecycle(
        tmp_path,
        bundle=incumbent,
        identity=incumbent_identity,
        admission_evidence=_admission(tmp_path, "incumbent"),
    )
    stable_challenger = write_adaptation_drift(
        tmp_path,
        bundle=challenger,
        identity=challenger_identity,
        family="premature_challenger_drift",
    )
    challenger_replay = write_replay_readiness(
        tmp_path, bundle=challenger, family="premature_challenger_replay"
    )

    with pytest.raises(ModelNativeAdaptationLifecycleError, match="transition mismatch"):
        finalize_adaptation_lifecycle_transition(
            transition=TRANSITION_PROMOTE_CHALLENGER,
            incumbent_bundle_dir=challenger,
            drift_evidence_path=stable_challenger["event_path"],
            replay_readiness_path=challenger_replay["event_path"],
            predecessor_path=initial["event_path"],
            admission_evidence=_admission(tmp_path, "challenger"),
            entry_run_id="UNIT_PREMATURE_PROMOTION",
            output_dir=tmp_path / "adaptation_lifecycle",
        )

    with pytest.raises(ModelNativeAdaptationLifecycleError, match="not newest"):
        load_bound_adaptation_lifecycle(
            initial["artifact"],
            context="UNIT_PREMATURE_PROMOTION_INVALIDATES_OLDER_PASS",
            now_utc=initial["event"]["created_utc"],
        )


def test_replay_readiness_requires_exact_artifact_inventory(tmp_path: Path) -> None:
    incumbent, incumbent_identity = write_adaptation_bundle(tmp_path, "incumbent")
    drift = write_adaptation_drift(
        tmp_path,
        bundle=incumbent,
        identity=incumbent_identity,
        family="inventory_drift",
    )
    incomplete_replay = write_replay_readiness(
        tmp_path,
        bundle=incumbent,
        family="incomplete_replay_inventory",
        omit_artifact="candidate_replay_trades",
    )

    with pytest.raises(
        ModelNativeAdaptationLifecycleError,
        match="artifact inventory is absent",
    ):
        finalize_adaptation_lifecycle_transition(
            transition=TRANSITION_INITIAL_ADMISSION,
            incumbent_bundle_dir=incumbent,
            drift_evidence_path=drift["event_path"],
            replay_readiness_path=incomplete_replay["event_path"],
            admission_evidence=_admission(tmp_path, "inventory"),
            entry_run_id="UNIT_INCOMPLETE_REPLAY_INVENTORY",
            output_dir=tmp_path / "inventory_lifecycle",
        )
