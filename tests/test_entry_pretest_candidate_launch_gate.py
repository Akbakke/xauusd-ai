from __future__ import annotations

import argparse
import json
from pathlib import Path

from gx1.contracts.entry_pretest_candidate_launch_gate_v1 import (
    AUTHORITY,
    artifact_binding,
    canonical_json_sha256,
    require_pretest_candidate_launch_gate,
)
from gx1.scripts import materialize_entry_pretest_candidate_launch_gate_v1 as materialize


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path.resolve()


def test_pretest_candidate_launch_gate_rehashes_direct_recipe_readiness_and_audit(
    monkeypatch, tmp_path: Path,
) -> None:
    dataset = (tmp_path / "dataset").resolve()
    dataset.mkdir()
    recipe_path = _write_json(
        (tmp_path / "recipe.json").resolve(),
        {
            "run_id": "PRETEST_CANDIDATE_RUN_V1",
            "dataset_run_id": "PRETEST_DATASET_V1",
            "dataset_dir": str(dataset),
            "out_bundle_dir": str((tmp_path / "bundle").resolve()),
        },
    )
    recipe_binding = artifact_binding(recipe_path)
    smoke_path = _write_json(
        (tmp_path / "smoke-audit.json").resolve(),
        {
            "schema_version": "entry_foundation_smoke_bundle_audit_v7",
            "dataset_dir": str(dataset),
        },
    )
    smoke_binding = artifact_binding(smoke_path)
    specialist_path = _write_json((tmp_path / "specialist.json").resolve(), {})
    trainability_path = _write_json(
        (tmp_path / "trainability.json").resolve(),
        {"schema_version": "entry_pretest_trainability_readiness_v1"},
    )
    readiness_bindings = {
        "smoke_bundle_audit": smoke_binding,
        "specialist_audit": artifact_binding(specialist_path),
        "trainability_readiness": artifact_binding(trainability_path),
    }
    readiness_path = _write_json(
        (tmp_path / "readiness.json").resolve(),
        {
            "schema_version": "entry_candidate_readiness_model_native_v1",
            "decision": "READY_FOR_CANDIDATE_TRAINING",
            "failures": [],
            "candidate_training_allowed": True,
            "promotion_shadow_live_allowed": False,
            "activation_authority": False,
            "expected_smoke_dataset_dir": str(dataset),
            "dataset_dir": str(dataset),
            "smoke_bundle_dataset_dir": str(dataset),
            "candidate_recipe": recipe_binding,
            "input_bindings": readiness_bindings,
            "input_bindings_sha256": canonical_json_sha256(readiness_bindings),
        },
    )
    readiness_binding = artifact_binding(readiness_path)
    gate_root = (tmp_path / "gates").resolve()
    # The materializer normally receives a fully validated recipe.  This
    # focused contract test supplies a compact immutable identity so it can
    # exercise both direct-readiness call sites without duplicating the full
    # recipe fixture.
    monkeypatch.setattr(
        materialize,
        "require_pretest_technical_recipe_metadata",
        lambda payload, **_kwargs: payload,
    )
    gate_path, _ = materialize.run(
        argparse.Namespace(
            recipe_json=str(recipe_path),
            recipe_sha256=recipe_binding["sha256"],
            candidate_readiness_json=str(readiness_path),
            candidate_readiness_sha256=readiness_binding["sha256"],
            smoke_bundle_audit_json=str(smoke_path),
            smoke_bundle_audit_sha256=smoke_binding["sha256"],
            out_dir=str(gate_root),
        )
    )

    validated = require_pretest_candidate_launch_gate(
        gate_path,
        artifact_binding(gate_path)["sha256"],
        expected_recipe_path=recipe_path,
        expected_recipe_sha256=recipe_binding["sha256"],
    )

    assert validated["authority"] == AUTHORITY
    assert validated["activation_authority"] is False
    assert validated["candidate_readiness"] == readiness_binding
