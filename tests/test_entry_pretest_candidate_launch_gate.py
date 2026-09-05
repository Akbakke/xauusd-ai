from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from gx1.contracts.entry_pretest_candidate_launch_gate_v1 import (
    AUTHORITY,
    PretestCandidateLaunchGateError,
    artifact_binding,
    canonical_json_sha256,
    require_pretest_candidate_launch_gate,
)
from gx1.scripts import materialize_entry_pretest_candidate_launch_gate_v1 as materialize
from tests.test_entry_candidate_readiness import _fixture, _technical_only_smoke


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path.resolve()


def _gate_inputs(monkeypatch, tmp_path: Path) -> argparse.Namespace:
    report, paths = _fixture(tmp_path)
    _technical_only_smoke(report)
    dataset = paths["dataset"]
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
        report,
    )
    smoke_binding = artifact_binding(smoke_path)
    specialist_path = paths["specialist"]
    trainability_bindings = {
        "candidate_recipe": recipe_binding,
        "smoke_recipe": artifact_binding(_write_json(tmp_path / "smoke-recipe.json", {})),
        "pretrain_audit": {
            key: report["input_audits"]["pretrain"][key] for key in ("path", "sha256")
        },
    }
    trainability_path = _write_json(
        (tmp_path / "trainability.json").resolve(),
        {
            "schema_version": "entry_pretest_trainability_readiness_v1",
            "decision": "READY_FOR_PRETEST_CANDIDATE_TRAINABILITY_REVIEW",
            "failures": [],
            "candidate_training_allowed": False,
            "activation_authority": False,
            "promotion_shadow_live_allowed": False,
            "dataset_dir": str(dataset),
            "dataset_run_id": "PRETEST_DATASET_V1",
            "candidate_run_id": "PRETEST_CANDIDATE_RUN_V1",
            "input_bindings": trainability_bindings,
            "input_bindings_sha256": canonical_json_sha256(trainability_bindings),
        },
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
    return argparse.Namespace(
            recipe_json=str(recipe_path),
            recipe_sha256=recipe_binding["sha256"],
            candidate_readiness_json=str(readiness_path),
            candidate_readiness_sha256=readiness_binding["sha256"],
            smoke_bundle_audit_json=str(smoke_path),
            smoke_bundle_audit_sha256=smoke_binding["sha256"],
            out_dir=str(gate_root),
    )


def _require_gate(args, gate_path):
    return require_pretest_candidate_launch_gate(
        gate_path,
        artifact_binding(gate_path)["sha256"],
        expected_recipe_path=args.recipe_json,
        expected_recipe_sha256=args.recipe_sha256,
    )


def test_pretest_candidate_launch_gate_rehashes_direct_recipe_readiness_and_audit(
    monkeypatch, tmp_path: Path,
) -> None:
    args = _gate_inputs(monkeypatch, tmp_path)
    gate_path, _ = materialize.run(args)
    validated = _require_gate(args, gate_path)

    assert validated["authority"] == AUTHORITY
    assert validated["activation_authority"] is False
    assert validated["candidate_readiness"] == artifact_binding(Path(args.candidate_readiness_json))


@pytest.mark.parametrize("malformed", [{}, {"path": "relative.json", "sha256": "a" * 64}, None])
def test_gate_rejects_missing_or_malformed_trainability(monkeypatch, tmp_path, malformed):
    args = _gate_inputs(monkeypatch, tmp_path)
    path = Path(args.candidate_readiness_json)
    payload = json.loads(path.read_text())
    payload["input_bindings"]["trainability_readiness"] = malformed
    payload["input_bindings_sha256"] = canonical_json_sha256(payload["input_bindings"])
    _write_json(path, payload)
    args.candidate_readiness_sha256 = artifact_binding(path)["sha256"]
    with pytest.raises(PretestCandidateLaunchGateError, match="binding"):
        materialize.run(args)


@pytest.mark.parametrize("artifact", ["model_state_dict", "specialist", "predictions", "pretrain", "smoke_recipe"])
def test_gate_rehashes_transitive_evidence_after_publication(monkeypatch, tmp_path, artifact):
    args = _gate_inputs(monkeypatch, tmp_path)
    gate_path, _ = materialize.run(args)
    report = json.loads(Path(args.smoke_bundle_audit_json).read_text())
    if artifact == "model_state_dict":
        path = Path(report["bundle_artifacts"][artifact]["path"])
    elif artifact == "predictions":
        path = Path(report["prediction_evidence"]["path"])
    elif artifact == "smoke_recipe":
        path = tmp_path / "smoke-recipe.json"
    else:
        path = Path(report["input_audits"][artifact]["path"])
    path.write_bytes(b"changed after publication")
    with pytest.raises(PretestCandidateLaunchGateError, match="hash mismatch"):
        _require_gate(args, gate_path)


def test_gate_rejects_unproven_head_liveness(monkeypatch, tmp_path):
    args = _gate_inputs(monkeypatch, tmp_path)
    path = Path(args.smoke_bundle_audit_json)
    payload = json.loads(path.read_text())
    payload["liveness_contract"]["all_active_head_predictions_live"] = False
    _write_json(path, payload)
    args.smoke_bundle_audit_sha256 = artifact_binding(path)["sha256"]
    with pytest.raises(PretestCandidateLaunchGateError, match="pipeline is not proven"):
        materialize.run(args)


@pytest.mark.parametrize("mutation", ["legacy", "wrong_candidate", "failed"])
def test_gate_rejects_rehashed_but_incompatible_trainability(monkeypatch, tmp_path, mutation):
    args = _gate_inputs(monkeypatch, tmp_path)
    readiness_path = Path(args.candidate_readiness_json)
    readiness = json.loads(readiness_path.read_text())
    path = Path(readiness["input_bindings"]["trainability_readiness"]["path"])
    payload = json.loads(path.read_text())
    if mutation == "legacy":
        payload["schema_version"] = "entry_model_native_seq513_trainability_readiness_v1"
    elif mutation == "failed":
        payload["decision"] = "FAIL"
    else:
        payload["input_bindings"]["candidate_recipe"] = payload["input_bindings"]["smoke_recipe"]
        payload["input_bindings_sha256"] = canonical_json_sha256(payload["input_bindings"])
    _write_json(path, payload)
    readiness["input_bindings"]["trainability_readiness"] = artifact_binding(path)
    readiness["input_bindings_sha256"] = canonical_json_sha256(readiness["input_bindings"])
    _write_json(readiness_path, readiness)
    args.candidate_readiness_sha256 = artifact_binding(readiness_path)["sha256"]
    with pytest.raises(PretestCandidateLaunchGateError, match="trainability"):
        materialize.run(args)
