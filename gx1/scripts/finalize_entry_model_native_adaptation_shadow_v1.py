"""Publish immutable paired incumbent/challenger zero-order shadow evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from gx1.contracts.entry_model_native_adaptation_drift_v1 import (
    ModelNativeAdaptationDriftError,
    adaptation_bundle_identity_from_dir,
)
from gx1.contracts.entry_model_native_adaptation_shadow_v1 import (
    MODEL_NATIVE_ADAPTATION_SHADOW_CONTRACT,
    MODEL_NATIVE_ADAPTATION_SHADOW_EVENT_PREFIX,
    MODEL_NATIVE_ADAPTATION_SHADOW_SCHEMA_VERSION,
    load_bound_adaptation_shadow_evidence,
    recompute_adaptation_shadow_evidence,
)
from gx1.contracts.entry_model_native_sizing_calibration_v1 import sha256_file
from gx1.contracts.immutable_event_authority_v1 import (
    next_immutable_event_created_utc,
    write_immutable_json_event,
)


class AdaptationShadowFinalizationError(RuntimeError):
    """Exact paired shadow evidence could not be produced."""


def _binding(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    return {"json_path": str(path), "sha256": sha256_file(path)}


def _canonical_parquet(path: Path) -> Path:
    raw = path.expanduser()
    absolute = raw if raw.is_absolute() else Path.cwd() / raw
    if (
        raw.suffix != ".parquet"
        or "latest" in raw.name.lower()
        or any(component.is_symlink() for component in (absolute, *absolute.parents))
    ):
        raise AdaptationShadowFinalizationError(
            "paired shadow rows must be canonical immutable parquet"
        )
    resolved = raw.resolve()
    if not resolved.is_file():
        raise AdaptationShadowFinalizationError(
            f"paired shadow rows are missing: {resolved}"
        )
    return resolved


def _bundle(bundle_dir: Path, *, context: str) -> dict[str, str]:
    try:
        return adaptation_bundle_identity_from_dir(bundle_dir, context=context)
    except ModelNativeAdaptationDriftError as exc:
        raise AdaptationShadowFinalizationError(str(exc)) from exc


def _publish_terminal_failure(
    *,
    output_dir: Path,
    error: Exception,
) -> None:
    write_immutable_json_event(
        output_dir,
        MODEL_NATIVE_ADAPTATION_SHADOW_EVENT_PREFIX,
        {
            "schema_version": "entry_model_native_adaptation_shadow_terminal_failure_v1",
            "created_utc": next_immutable_event_created_utc(
                output_dir,
                MODEL_NATIVE_ADAPTATION_SHADOW_EVENT_PREFIX,
            ).isoformat(),
            "decision": "FAIL",
            "failures": [f"{type(error).__name__}: {error}"],
            "shadow_contract": MODEL_NATIVE_ADAPTATION_SHADOW_CONTRACT,
        },
    )


def finalize_adaptation_shadow_evidence(
    *,
    incumbent_bundle_dir: Path,
    candidate_bundle_dir: Path,
    paired_rows_path: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Publish a paired shadow PASS/FAIL without training or activation."""

    output_dir = output_dir.expanduser().resolve()
    try:
        incumbent = _bundle(
            incumbent_bundle_dir,
            context="ADAPTATION_SHADOW_FINALIZER_INCUMBENT",
        )
        candidate = _bundle(
            candidate_bundle_dir,
            context="ADAPTATION_SHADOW_FINALIZER_CANDIDATE",
        )
        if incumbent == candidate:
            raise AdaptationShadowFinalizationError(
                "incumbent and candidate bundle identities must differ"
            )
        rows_path = _canonical_parquet(paired_rows_path)
        rows_binding = {"path": str(rows_path), "sha256": sha256_file(rows_path)}
        rows = pd.read_parquet(rows_path)
        created = pd.Timestamp(
            next_immutable_event_created_utc(
                output_dir,
                MODEL_NATIVE_ADAPTATION_SHADOW_EVENT_PREFIX,
            )
        )
        recomputed = recompute_adaptation_shadow_evidence(
            paired_rows=rows,
            incumbent_bundle=incumbent,
            candidate_bundle=candidate,
            event_created_utc=created,
            context="ADAPTATION_SHADOW_FINALIZER",
        )
        event_path, event = write_immutable_json_event(
            output_dir,
            MODEL_NATIVE_ADAPTATION_SHADOW_EVENT_PREFIX,
            {
                "schema_version": MODEL_NATIVE_ADAPTATION_SHADOW_SCHEMA_VERSION,
                "created_utc": created.isoformat(),
                "shadow_contract": MODEL_NATIVE_ADAPTATION_SHADOW_CONTRACT,
                "incumbent_bundle": incumbent,
                "candidate_bundle": candidate,
                "paired_rows": rows_binding,
                **recomputed,
            },
        )
        if event["decision"] == "PASS":
            load_bound_adaptation_shadow_evidence(
                _binding(event_path),
                incumbent_bundle=incumbent,
                candidate_bundle=candidate,
                context="ADAPTATION_SHADOW_FINALIZER_SELF_VALIDATION",
                now_utc=created,
            )
        return event_path, event
    except Exception as exc:
        try:
            _publish_terminal_failure(output_dir=output_dir, error=exc)
        except Exception as publication_exc:
            exc.add_note(
                f"terminal shadow failure publication also failed: {publication_exc}"
            )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent-bundle-dir", type=Path, required=True)
    parser.add_argument("--candidate-bundle-dir", type=Path, required=True)
    parser.add_argument("--paired-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    path, event = finalize_adaptation_shadow_evidence(
        incumbent_bundle_dir=args.incumbent_bundle_dir,
        candidate_bundle_dir=args.candidate_bundle_dir,
        paired_rows_path=args.paired_rows,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "json_path": str(path),
                "sha256": sha256_file(path),
                "decision": event["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if event["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
