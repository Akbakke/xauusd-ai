#!/usr/bin/env python3
"""Rebind an immutable train-only field order to the current signal contract.

This small offline producer is for shared Entry/Exit feature-surface
materialization only.  It never re-ranks features and never authorizes
training: the historical selection is accepted only after the current 513
field registry validates it exactly, with the source manifest hash and fit
window retained as provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_SIGNAL_DIM,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SIGNAL_DIM,
    model_native_mandatory_full_stack_metadata,
    model_native_signal_contract_metadata,
    ordered_model_native_signal_fields,
)
from gx1.contracts.gx1_scope_v1 import require_offline_scope
from gx1.features.entry_foundation_structure_v1 import (
    FOUNDATION_STRUCTURE_FEATURE_NAMES,
    FOUNDATION_STRUCTURE_FEATURE_VERSION,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"CURRENT_STRUCTURE_MANIFEST_OUTPUT_NOT_FRESH: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def materialize(*, source_manifest: Path, output: Path, run_id: str) -> dict[str, Any]:
    require_offline_scope("featurebase_build")
    source = source_manifest.expanduser().resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"CURRENT_STRUCTURE_MANIFEST_SOURCE_INVALID: {source}")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("selected_features"), list):
        raise RuntimeError("CURRENT_STRUCTURE_MANIFEST_SOURCE_SELECTION_MISSING")
    selected = [str(value) for value in raw["selected_features"]]
    if len(selected) != MODEL_NATIVE_SELECTED_FEATURE_COUNT:
        raise RuntimeError("CURRENT_STRUCTURE_MANIFEST_SELECTION_DIM_INVALID")
    ordered_model_native_signal_fields(selected)
    contract = model_native_signal_contract_metadata(selected)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    source_hash = _sha256_file(source)
    ranked_remainder = selected[MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT:]
    manifest = {
        "schema_version": "entry_model_native_seq513_current_structure_manifest_v1",
        "created_utc": now,
        "json_path": str(output.expanduser().resolve()),
        "producer": "gx1.scripts.materialize_entry_model_native_current_structure_manifest_v1",
        "producer_version": "v1",
        "entry_run_id": run_id,
        "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
        "decision": "READY_FOR_OFFLINE_SHARED_FEATURE_BASE_MATERIALIZATION",
        "manifest_only": True,
        "selected_features": selected,
        "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "base_signal_feature_count": MODEL_NATIVE_BASE_SIGNAL_DIM,
        "expected_seq_snap_width": MODEL_NATIVE_SIGNAL_DIM,
        "model_native_signal_contract": contract,
        "mandatory_full_stack": model_native_mandatory_full_stack_metadata(),
        "mandatory_selected_feature_count": MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
        "ranked_remainder_feature_count": MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
        "ranked_remainder_features": ranked_remainder,
        "foundation_structure_feature_version": FOUNDATION_STRUCTURE_FEATURE_VERSION,
        "foundation_structure_feature_count": len(FOUNDATION_STRUCTURE_FEATURE_NAMES),
        "foundation_structure_missing_feature_count": 0,
        "foundation_structure_all_required_selected": all(
            name in selected for name in FOUNDATION_STRUCTURE_FEATURE_NAMES
        ),
        "selected_fields_sha256": hashlib.sha256(
            json.dumps(selected, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest(),
        "selection_provenance": {
            "source_manifest": str(source),
            "source_manifest_sha256": source_hash,
            "source_created_utc": raw.get("created_utc"),
            "source_entry_run_id": raw.get("entry_run_id"),
            "fit_scope": raw.get("feature_ranking", {}).get("fit_scope"),
            "fit_start_utc": raw.get("feature_ranking", {}).get("train_start_utc"),
            "fit_end_utc": raw.get("feature_ranking", {}).get("train_end_utc"),
            "selection_rebound_to_current_contract": True,
        },
        "dataset_rebuild_required_before_training": True,
        "training_allowed": False,
        "shadow_live_promotion_allowed": False,
    }
    _write_exclusive(output.expanduser().resolve(), manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
