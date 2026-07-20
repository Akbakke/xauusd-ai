from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from gx1.contracts.entry_model_native_adaptation_lifecycle_v1 import (
    MODEL_NATIVE_REPLAY_READINESS_REQUIRED_ARTIFACTS,
    adaptation_lifecycle_handoff_metadata,
)
from gx1.contracts.entry_model_native_adaptation_drift_v1 import (
    adaptation_bundle_identity_from_dir,
)
from gx1.contracts.entry_model_native_readiness_v1 import artifact_fingerprints
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_CONTRACT_MODE
from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
from gx1.scripts.finalize_entry_model_native_adaptation_drift_v1 import (
    finalize_adaptation_drift_evidence,
)
from gx1.scripts.finalize_entry_model_native_adaptation_lifecycle_v1 import (
    finalize_adaptation_lifecycle_transition,
)
from gx1.scripts.finalize_entry_model_native_adaptation_shadow_v1 import (
    finalize_adaptation_shadow_evidence,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event_binding(path: Path) -> dict[str, str]:
    return {"json_path": str(path.resolve()), "sha256": sha256_file(path)}


def adaptation_bundle_identity(bundle: Path) -> dict[str, str]:
    return adaptation_bundle_identity_from_dir(
        bundle,
        context="UNIT_ADAPTATION_BUNDLE",
    )


def write_adaptation_bundle(root: Path, name: str = "bundle") -> tuple[Path, dict[str, str]]:
    bundle = root / name
    bundle.mkdir(parents=True)
    state = bundle / "model_state_dict.pt"
    state.write_bytes(f"unit-adaptation-model-state:{name}".encode("utf-8"))
    state_sha = sha256_file(state)
    metadata = bundle / "bundle_metadata.json"
    metadata.write_text(
        json.dumps({"state_dict_sha256": state_sha}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle, {
        "bundle_dir": str(bundle.resolve()),
        "bundle_metadata_path": str(metadata.resolve()),
        "bundle_metadata_sha256": sha256_file(metadata),
        "model_state_dict_path": str(state.resolve()),
        "model_state_dict_sha256": state_sha,
    }


def adaptation_rows(
    *,
    start: pd.Timestamp,
    scope: str,
    identity: dict[str, str],
    loss_long: bool = False,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    sessions = ("ASIA", "EU", "OVERLAP", "US")
    regimes = ("LOW", "HIGH")
    for index in range(480):
        direction = index % 3
        probabilities = {
            0: (0.90, 0.05, 0.05),
            1: (0.05, 0.90, 0.05),
            2: (0.05, 0.05, 0.90),
        }[direction]
        pnl = 0.0 if direction == 2 else 5.0 + (index % 5) * 0.1
        if loss_long and direction == 0:
            pnl = -2.0
        entry_bid = 2_500.0
        entry_ask = 2_500.2
        if direction == 0:
            exit_bid = entry_ask * (1.0 + pnl / 10_000.0)
            exit_ask = exit_bid + 0.2
        elif direction == 1:
            exit_ask = entry_bid * (1.0 - pnl / 10_000.0)
            exit_bid = exit_ask - 0.2
        else:
            exit_bid = entry_bid
            exit_ask = entry_ask
        records.append(
            {
                "time": start + pd.Timedelta(minutes=30 * index),
                "evidence_scope": scope,
                "model_direction_index": direction,
                "p_long": probabilities[0],
                "p_short": probabilities[1],
                "p_flat": probabilities[2],
                "entry_bid": entry_bid,
                "entry_ask": entry_ask,
                "exit_bid": exit_bid,
                "exit_ask": exit_ask,
                "realized_pnl_bps": pnl,
                "session": sessions[index % len(sessions)],
                "vol_regime": regimes[index % len(regimes)],
                "bundle_metadata_sha256": identity["bundle_metadata_sha256"],
                "model_state_dict_sha256": identity["model_state_dict_sha256"],
                "outcome_source_id": f"{scope}-{index:04d}",
                "order_submitted": False,
            }
        )
    return pd.DataFrame(records)


def write_adaptation_drift(
    root: Path,
    *,
    bundle: Path,
    identity: dict[str, str],
    loss_long: bool = False,
    family: str = "drift_events",
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC").floor("s")
    reference = adaptation_rows(
        start=now - pd.Timedelta(days=31),
        scope="candidate_test",
        identity=identity,
    )
    observations = adaptation_rows(
        start=now - pd.Timedelta(days=10),
        scope="broker_shadow",
        identity=identity,
        loss_long=loss_long,
    )
    reference_path = root / f"{family}_reference_20260720T120000123456Z.parquet"
    observation_path = root / f"{family}_observations_20260720T120000123456Z.parquet"
    reference.to_parquet(reference_path, index=False)
    observations.to_parquet(observation_path, index=False)
    event_path, event = finalize_adaptation_drift_evidence(
        bundle_dir=bundle,
        reference_rows_path=reference_path,
        observation_rows_path=observation_path,
        output_dir=root / family,
    )
    return {
        "event": event,
        "artifact": event_binding(event_path),
        "event_path": event_path,
    }


def paired_shadow_rows(
    *,
    start: pd.Timestamp,
    incumbent_identity: dict[str, str],
    candidate_identity: dict[str, str],
    superior: bool = True,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    sessions = ("ASIA", "EU", "OVERLAP", "US")
    regimes = ("LOW", "HIGH")
    for index in range(480):
        candidate_direction = index % 3
        if superior:
            incumbent_direction = 2 if candidate_direction in (0, 1) else 0
        else:
            incumbent_direction = candidate_direction
        candidate_probabilities = [0.05, 0.05, 0.05]
        candidate_probabilities[candidate_direction] = 0.90
        incumbent_probabilities = [0.05, 0.05, 0.05]
        incumbent_probabilities[incumbent_direction] = 0.90
        entry_bid = 2_500.0
        entry_ask = 2_500.2
        if candidate_direction == 0:
            exit_bid = entry_ask * (1.0 + 5.0 / 10_000.0)
            exit_ask = exit_bid + 0.2
        elif candidate_direction == 1:
            exit_ask = entry_bid * (1.0 - 5.0 / 10_000.0)
            exit_bid = exit_ask - 0.2
        elif superior:
            exit_bid = entry_ask * (1.0 - 5.0 / 10_000.0)
            exit_ask = exit_bid + 0.2
        else:
            exit_bid = entry_bid
            exit_ask = entry_ask

        def pnl(direction: int) -> float:
            if direction == 0:
                return (exit_bid - entry_ask) / entry_ask * 10_000.0
            if direction == 1:
                return (entry_bid - exit_ask) / entry_bid * 10_000.0
            return 0.0

        records.append(
            {
                "time": start + pd.Timedelta(minutes=30 * index),
                "candidate_direction_index": candidate_direction,
                "candidate_p_long": candidate_probabilities[0],
                "candidate_p_short": candidate_probabilities[1],
                "candidate_p_flat": candidate_probabilities[2],
                "incumbent_direction_index": incumbent_direction,
                "incumbent_p_long": incumbent_probabilities[0],
                "incumbent_p_short": incumbent_probabilities[1],
                "incumbent_p_flat": incumbent_probabilities[2],
                "entry_bid": entry_bid,
                "entry_ask": entry_ask,
                "exit_bid": exit_bid,
                "exit_ask": exit_ask,
                "candidate_realized_pnl_bps": pnl(candidate_direction),
                "incumbent_realized_pnl_bps": pnl(incumbent_direction),
                "session": sessions[index % len(sessions)],
                "vol_regime": regimes[index % len(regimes)],
                "candidate_bundle_metadata_sha256": candidate_identity[
                    "bundle_metadata_sha256"
                ],
                "candidate_model_state_dict_sha256": candidate_identity[
                    "model_state_dict_sha256"
                ],
                "incumbent_bundle_metadata_sha256": incumbent_identity[
                    "bundle_metadata_sha256"
                ],
                "incumbent_model_state_dict_sha256": incumbent_identity[
                    "model_state_dict_sha256"
                ],
                "outcome_source_id": f"paired-shadow-{index:04d}",
                "order_submitted": False,
            }
        )
    return pd.DataFrame(records)


def write_adaptation_shadow(
    root: Path,
    *,
    incumbent_bundle: Path,
    incumbent_identity: dict[str, str],
    candidate_bundle: Path,
    candidate_identity: dict[str, str],
    superior: bool = True,
    family: str = "paired_shadow",
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    rows = paired_shadow_rows(
        start=pd.Timestamp.now(tz="UTC").floor("s") - pd.Timedelta(days=10),
        incumbent_identity=incumbent_identity,
        candidate_identity=candidate_identity,
        superior=superior,
    )
    rows_path = root / f"{family}_rows_20260720T120000123456Z.parquet"
    rows.to_parquet(rows_path, index=False)
    event_path, event = finalize_adaptation_shadow_evidence(
        incumbent_bundle_dir=incumbent_bundle,
        candidate_bundle_dir=candidate_bundle,
        paired_rows_path=rows_path,
        output_dir=root / family,
    )
    return {
        "event": event,
        "artifact": event_binding(event_path),
        "event_path": event_path,
        "rows_path": rows_path,
    }


def write_replay_readiness(
    root: Path,
    *,
    bundle: Path,
    family: str = "replay_readiness",
    omit_artifact: str | None = None,
) -> dict[str, Any]:
    artifacts: dict[str, str] = {}
    for name in sorted(MODEL_NATIVE_REPLAY_READINESS_REQUIRED_ARTIFACTS):
        artifact = root / f"{family}_{name}_20260720T120000123456Z.json"
        artifact.write_text("{\"decision\":\"PASS\"}\n", encoding="utf-8")
        artifacts[name] = str(artifact.resolve())
    if omit_artifact is not None:
        artifacts.pop(omit_artifact)
    event_path, event = write_immutable_json_event(
        root / family,
        "ENTRY_REPLAY_READINESS",
        {
            "schema_version": "entry_replay_readiness_model_native_v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "decision": "READY_FOR_MODEL_NATIVE_REPLAY_REVIEW",
            "model_native_replay_evidence_ready": True,
            "secondary_direction_authority_allowed": False,
            "promotion_shadow_live_allowed": False,
            "adaptation_lifecycle_handoff": adaptation_lifecycle_handoff_metadata(),
            "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
            "failures": [],
            "bundle_identity": adaptation_bundle_identity(bundle),
            "evidence_identity": {
                "candidate_bundle_dir": str(bundle.resolve()),
                "selective_edge_bundle_dir": str(bundle.resolve()),
                "replay_identity_candidate_bundle_dir": str(bundle.resolve()),
                "replay_identity_ready": True,
            },
            "artifacts": artifacts,
            "artifact_fingerprints": artifact_fingerprints(artifacts),
        },
    )
    return {
        "event": event,
        "artifact": event_binding(event_path),
        "event_path": event_path,
    }


def write_initial_adaptation_lifecycle(
    root: Path,
    *,
    bundle: Path,
    identity: dict[str, str],
    admission_evidence: dict[str, Any],
) -> dict[str, Any]:
    drift = write_adaptation_drift(
        root,
        bundle=bundle,
        identity=identity,
        family="initial_drift",
    )
    replay = write_replay_readiness(
        root,
        bundle=bundle,
        family="initial_replay_readiness",
    )
    event_path, event = finalize_adaptation_lifecycle_transition(
        transition="INITIAL_ADMISSION",
        incumbent_bundle_dir=bundle,
        drift_evidence_path=drift["event_path"],
        replay_readiness_path=replay["event_path"],
        output_dir=root / "adaptation_lifecycle",
        admission_evidence=admission_evidence,
        entry_run_id="UNIT_INITIAL_ADAPTATION_ADMISSION",
    )
    return {
        "event": event,
        "artifact": event_binding(event_path),
        "event_path": event_path,
        "drift": drift,
        "replay": replay,
    }
