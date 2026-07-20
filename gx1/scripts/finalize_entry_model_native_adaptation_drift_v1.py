"""Publish immutable row-recomputed Entry adaptation drift evidence.

The producer reads explicit TEST-reference and settled broker-shadow parquets
for one exact bundle.  It does not train, promote, submit orders, or change the
launch selector.  A failed refresh publishes a newer malformed-for-admission
terminal event so an older ``STABLE`` result cannot remain authoritative.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from gx1.contracts.entry_model_native_adaptation_drift_v1 import (
    MODEL_NATIVE_ADAPTATION_DRIFT_CONTRACT,
    MODEL_NATIVE_ADAPTATION_DRIFT_EVENT_PREFIX,
    MODEL_NATIVE_ADAPTATION_DRIFT_SCHEMA_VERSION,
    ModelNativeAdaptationDriftError,
    adaptation_bundle_identity_from_dir,
    load_bound_adaptation_drift_evidence,
    recompute_adaptation_drift_evidence,
)
from gx1.contracts.entry_model_native_sizing_calibration_v1 import sha256_file
from gx1.contracts.immutable_event_authority_v1 import (
    next_immutable_event_created_utc,
    write_immutable_json_event,
)


class AdaptationDriftFinalizationError(RuntimeError):
    """The exact adaptation drift event could not be produced."""


def _binding(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    return {"json_path": str(path), "sha256": sha256_file(path)}


def _canonical_parquet(path: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    absolute = candidate if candidate.is_absolute() else Path.cwd() / candidate
    if (
        candidate.suffix != ".parquet"
        or "latest" in candidate.name.lower()
        or any(component.is_symlink() for component in (absolute, *absolute.parents))
    ):
        raise AdaptationDriftFinalizationError(
            f"{label} must be a canonical immutable parquet"
        )
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise AdaptationDriftFinalizationError(f"{label} is missing: {resolved}")
    return resolved


def _source_binding(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _bundle_identity(bundle_dir: Path) -> dict[str, str]:
    try:
        return adaptation_bundle_identity_from_dir(
            bundle_dir,
            context="ADAPTATION_DRIFT_FINALIZER_BUNDLE",
        )
    except ModelNativeAdaptationDriftError as exc:
        raise AdaptationDriftFinalizationError(str(exc)) from exc


def _publish_terminal_failure(
    *,
    output_dir: Path,
    inputs: dict[str, str],
    error: Exception,
) -> None:
    payload = {
        "schema_version": "entry_model_native_adaptation_drift_terminal_failure_v1",
        "created_utc": next_immutable_event_created_utc(
            output_dir,
            MODEL_NATIVE_ADAPTATION_DRIFT_EVENT_PREFIX,
        ).isoformat(),
        "decision": "DRIFT",
        "failures": [f"{type(error).__name__}: {error}"],
        "drift_contract": MODEL_NATIVE_ADAPTATION_DRIFT_CONTRACT,
        "inputs": inputs,
    }
    write_immutable_json_event(
        output_dir,
        MODEL_NATIVE_ADAPTATION_DRIFT_EVENT_PREFIX,
        payload,
    )


def finalize_adaptation_drift_evidence(
    *,
    bundle_dir: Path,
    reference_rows_path: Path,
    observation_rows_path: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Publish newest STABLE/DRIFT evidence from exact immutable row inputs."""

    output_dir = output_dir.expanduser().resolve()
    inputs = {
        "bundle_dir": str(bundle_dir.expanduser().absolute()),
        "reference_rows_path": str(reference_rows_path.expanduser().absolute()),
        "observation_rows_path": str(observation_rows_path.expanduser().absolute()),
    }
    try:
        identity = _bundle_identity(bundle_dir)
        reference_path = _canonical_parquet(
            reference_rows_path, label="TEST reference rows"
        )
        observation_path = _canonical_parquet(
            observation_rows_path, label="broker-shadow observation rows"
        )
        reference_binding = _source_binding(reference_path)
        observation_binding = _source_binding(observation_path)
        reference = pd.read_parquet(reference_path)
        observations = pd.read_parquet(observation_path)
        created = pd.Timestamp(
            next_immutable_event_created_utc(
                output_dir,
                MODEL_NATIVE_ADAPTATION_DRIFT_EVENT_PREFIX,
            )
        )
        recomputed = recompute_adaptation_drift_evidence(
            reference_rows=reference,
            observation_rows=observations,
            bundle_identity=identity,
            event_created_utc=created,
            context="ADAPTATION_DRIFT_FINALIZER",
        )
        payload = {
            "schema_version": MODEL_NATIVE_ADAPTATION_DRIFT_SCHEMA_VERSION,
            "created_utc": created.isoformat(),
            "drift_contract": MODEL_NATIVE_ADAPTATION_DRIFT_CONTRACT,
            "bundle_identity": identity,
            "reference_rows": reference_binding,
            "observation_rows": observation_binding,
            **recomputed,
        }
        event_path, event = write_immutable_json_event(
            output_dir,
            MODEL_NATIVE_ADAPTATION_DRIFT_EVENT_PREFIX,
            payload,
        )
        load_bound_adaptation_drift_evidence(
            _binding(event_path),
            context="ADAPTATION_DRIFT_FINALIZER_SELF_VALIDATION",
            now_utc=created,
        )
        return event_path, event
    except Exception as exc:
        try:
            _publish_terminal_failure(
                output_dir=output_dir,
                inputs=inputs,
                error=exc,
            )
        except Exception as publication_exc:
            exc.add_note(
                f"terminal drift failure publication also failed: {publication_exc}"
            )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--reference-rows", type=Path, required=True)
    parser.add_argument("--observation-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    event_path, event = finalize_adaptation_drift_evidence(
        bundle_dir=args.bundle_dir,
        reference_rows_path=args.reference_rows,
        observation_rows_path=args.observation_rows,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "json_path": str(event_path),
                "sha256": sha256_file(event_path),
                "decision": event["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if event["decision"] == "STABLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
