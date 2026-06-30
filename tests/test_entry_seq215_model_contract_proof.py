import json
from pathlib import Path

import numpy as np
import pytest

from gx1.features.entry_specialist_feature_groups_v1 import (
    CHALLENGER_SEQ215_SPECIALIST_MODEL_CONTRACT,
    CHALLENGER_SEQ215_TRAINING_SPECIALISTS,
    SPECIALIST_FUSION_ACTIVE_HEADS,
    SPECIALIST_FUSION_BLOCKED_HEADS,
    required_training_specialists_for_mode,
    specialist_model_contract_for_mode,
)
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import _load_specialist_fusion_contract
from gx1.scripts.audit_entry_foundation_smoke_bundle_v1 import (
    _gate_stats,
    _pretrain_manifest_contract_report,
    _sha256_file,
    _specialist_gate_failures,
)
from gx1.scripts.materialize_entry_feature_ai_inventory_v1 import (
    _specialist_contract_provenance as _inventory_contract_provenance,
)
from gx1.scripts.materialize_entry_specialist_challenger_extension_manifest_v1 import (
    _specialist_contract_provenance as _challenger_manifest_contract_provenance,
)


SEQ215_SPECIALIST_AUDIT = Path(
    "/home/andre2/GX1_DATA/reports/entry_specialist_feature_group_audit_20260628_v1/"
    "challenger_seq215_20260630_contract8/ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.json"
)


def _contract_payload() -> dict:
    return json.loads(json.dumps(CHALLENGER_SEQ215_SPECIALIST_MODEL_CONTRACT))


def _write_artifacts(tmp_path: Path) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for name in (
        "training_readiness",
        "feature_audit",
        "target_audit",
        "specialist_audit",
        "foundation_guardrails",
        "smoke_dataset_manifest",
        "worktree_hygiene",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"artifact": name}), encoding="utf-8")
        artifacts[name] = path
    return artifacts


def _seq215_pretrain_manifest(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    artifacts = _write_artifacts(tmp_path)
    bundle_dir = tmp_path / "bundle"
    dataset_dir = tmp_path / "smoke_dataset"
    bundle_dir.mkdir()
    dataset_dir.mkdir()
    artifact_sha256 = {key: _sha256_file(path) for key, path in artifacts.items()}
    artifact_fingerprints = {
        key: {
            "path": str(path),
            "sha256": artifact_sha256[key],
            "exists": True,
        }
        for key, path in artifacts.items()
        if key != "training_readiness"
    }
    manifest = {
        "schema_version": "entry_foundation_smoke_train_run_manifest_v1",
        "specialist_contract_mode": "challenger_seq215",
        "out_bundle_dir": str(bundle_dir),
        "promotion_shadow_live_allowed": False,
        "trainer_started_by_manifest_writer": False,
        "inputs": {
            "smoke_dataset_dir": str(dataset_dir),
            "training_readiness_json": str(artifacts["training_readiness"]),
            "feature_audit_json": str(artifacts["feature_audit"]),
            "target_audit_json": str(artifacts["target_audit"]),
            "specialist_audit_json": str(artifacts["specialist_audit"]),
            "foundation_guardrails_json": str(artifacts["foundation_guardrails"]),
            "smoke_dataset_manifest": str(artifacts["smoke_dataset_manifest"]),
            "worktree_hygiene_json": str(artifacts["worktree_hygiene"]),
        },
        "artifact_sha256": artifact_sha256,
        "preflight_contracts": {
            "training_readiness": {
                "foundation_contract_ready_for_smoke": True,
                "artifact_provenance_decision": "PASS",
                "artifact_fingerprints": artifact_fingerprints,
            },
            "feature_foundation": {
                "decision": "PASS",
                "foundation_objective_coverage_all_present": True,
                "foundation_objective_coverage": [{"objective": "hh_hl_lh_ll", "missing_count": 0}],
                "foundation_objective_liveness_all_live": True,
                "foundation_objective_liveness": [
                    {
                        "split": "train",
                        "objective": "hh_hl_lh_ll",
                        "missing_count": 0,
                        "nonfinite_count": 0,
                        "near_constant_count": 0,
                    }
                ],
                "foundation_source_field_liveness_all_live": True,
                "min_required_source_active_rate": 0.0001,
                "min_required_source_active_count": 1,
                "foundation_source_field_liveness": [
                    {
                        "split": "train",
                        "source_field": "chart.foundation_hh_state",
                        "observed": True,
                        "nonfinite_count": 0,
                        "active_count": 12,
                        "active_rate": 0.1,
                        "near_constant": False,
                    }
                ],
                "foundation_source_fields_by_split": {"train": {"source_missing_count": 0}},
            },
            "target_foundation": {
                "decision": "PASS",
                "active_training_heads": list(SPECIALIST_FUSION_ACTIVE_HEADS),
                "blocked_heads": list(SPECIALIST_FUSION_BLOCKED_HEADS),
            },
            "specialist_contract": {
                "contract_mode": "challenger_seq215",
                "audit_contract_mode": "challenger_seq215",
                "architecture_active_heads": list(SPECIALIST_FUSION_ACTIVE_HEADS),
                "architecture_blocked_heads": list(SPECIALIST_FUSION_BLOCKED_HEADS),
                "required_training_specialists": list(CHALLENGER_SEQ215_TRAINING_SPECIALISTS),
                "trainable_specialists": list(CHALLENGER_SEQ215_TRAINING_SPECIALISTS),
                "specialist_model_contract_valid": True,
                "specialist_model_contract_failures": [],
                "specialist_model_contract": _contract_payload(),
                "foundation_objective_routing_all_present_and_expected": True,
                "foundation_objective_routing": [
                    {"objective": "hh_hl_lh_ll", "missing_count": 0, "misrouted_count": 0}
                ],
                "specialist_input_liveness_all_live": True,
                "specialist_input_liveness": [
                    {
                        "split": "train",
                        "specialist": name,
                        "live_feature_count": 3,
                        "min_required_live_feature_count": 1,
                        "nonfinite_count": 0,
                    }
                    for name in CHALLENGER_SEQ215_TRAINING_SPECIALISTS
                ],
            },
            "smoke_dataset": {
                "schema_version": "entry_foundation_seq215_smoke_dataset_v1",
                "audit_provenance_schema_version": "entry_foundation_smoke_dataset_audit_provenance_v1",
                "audit_provenance_all_artifacts_present": True,
                "audit_provenance_all_artifact_hashes_present": True,
                "audit_provenance_artifacts": {
                    key: {
                        "path": str(artifacts[key]),
                        "sha256": artifact_sha256[key],
                        "decision": "PASS",
                    }
                    for key in ("feature_audit", "target_audit", "specialist_audit")
                },
                "split_hashes": {
                    split: {
                        "source_manifest_sha256": "a" * 64,
                        "out_parquet_sha256": "b" * 64,
                        "out_manifest_sha256": "c" * 64,
                    }
                    for split in ("train", "val", "test")
                },
            },
            "worktree_hygiene": {
                "decision": "BLOCKED_BY_DIRTY_GIT",
                "foundation_cleanup_critical_gate_review": {
                    "critical_gate_path_count": 12,
                    "ok_count": 12,
                    "missing_from_repo": [],
                    "dirty_missing_from_stage": [],
                },
            },
        },
    }
    return manifest, bundle_dir, dataset_dir, tmp_path / "pretrain_manifest.json"


@pytest.mark.parametrize(
    "provenance_factory",
    (_inventory_contract_provenance, _challenger_manifest_contract_provenance),
)
def test_report_only_seq215_contract_provenance_separates_active_and_target_modes(provenance_factory) -> None:
    provenance = provenance_factory()
    active = provenance["active_foundation"]
    target = provenance["target_challenger"]

    assert active["contract_mode"] == "foundation_seq146"
    assert active["required_training_specialists"] == list(
        required_training_specialists_for_mode("foundation_seq146")
    )
    assert active["required_training_specialist_count"] == 6
    assert active["specialist_model_contract"] == specialist_model_contract_for_mode("foundation_seq146")

    assert target["contract_mode"] == "challenger_seq215"
    assert target["required_training_specialists"] == list(
        required_training_specialists_for_mode("challenger_seq215")
    )
    assert target["required_training_specialist_count"] == 8
    assert target["specialist_model_contract"] == specialist_model_contract_for_mode("challenger_seq215")
    assert set(target["additional_training_specialists_vs_active_foundation"]) == {
        "chart_geometry_encoder",
        "price_action_candle_encoder",
    }
    assert target["contract_registered"] is True
    assert target["contract_update_required_before_training"] is False
    assert provenance["contract_update_required_before_training"] is False


def test_seq215_trainer_loader_requires_exact_challenger_contract_mode() -> None:
    indices, meta = _load_specialist_fusion_contract(
        SEQ215_SPECIALIST_AUDIT,
        expected_signal_dim=215,
        contract_mode="challenger_seq215",
    )

    assert meta["contract_mode"] == "challenger_seq215"
    assert meta["audit_contract_mode"] == "challenger_seq215"
    assert set(indices) == set(CHALLENGER_SEQ215_TRAINING_SPECIALISTS)
    assert "chart_geometry_encoder" in indices
    assert "price_action_candle_encoder" in indices
    assert set(meta["active_heads"]) == set(SPECIALIST_FUSION_ACTIVE_HEADS)
    assert meta["blocked_heads"] == list(SPECIALIST_FUSION_BLOCKED_HEADS)
    assert meta["specialist_model_contract"] == _contract_payload()

    with pytest.raises(RuntimeError, match="SPECIALIST_MODEL_CONTRACT_INVALID"):
        _load_specialist_fusion_contract(
            SEQ215_SPECIALIST_AUDIT,
            expected_signal_dim=215,
            contract_mode="foundation_seq146",
        )


def test_seq215_gate_liveness_requires_all_eight_experts() -> None:
    names = list(CHALLENGER_SEQ215_TRAINING_SPECIALISTS)
    gate = np.full((5, len(names)), 1.0 / len(names), dtype=np.float32)

    assert (
        _specialist_gate_failures(
            split="val",
            gate_report=_gate_stats(gate, names),
            specialist_names=names,
            required_specialists=tuple(names),
            min_active_specialists=len(names),
            min_gate_entropy=0.05,
        )
        == []
    )

    missing_price_action = names[:-1]
    failures = _specialist_gate_failures(
        split="val",
        gate_report=_gate_stats(gate[:, :-1], missing_price_action),
        specialist_names=missing_price_action,
        required_specialists=tuple(names),
        min_active_specialists=len(names),
        min_gate_entropy=0.05,
    )

    assert any("metadata missing required specialists" in failure for failure in failures)
    assert any("collapsed active_specialist_count" in failure for failure in failures)


def test_seq215_pretrain_manifest_contract_preserves_exact_mode_and_eight_experts(tmp_path: Path) -> None:
    manifest, bundle_dir, dataset_dir, manifest_path = _seq215_pretrain_manifest(tmp_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = _pretrain_manifest_contract_report(
        manifest_path,
        expected_bundle_dir=bundle_dir,
        expected_dataset_dir=dataset_dir,
    )

    assert report["decision"] == "PASS"
    assert report["specialist_contract_mode"] == "challenger_seq215"
    assert report["expected_required_training_specialists"] == sorted(CHALLENGER_SEQ215_TRAINING_SPECIALISTS)
    assert report["specialist_required_training_set_exact"] is True
    assert report["specialist_trainable_set_exact"] is True
    assert report["specialist_model_contract_set_exact"] is True

    manifest["specialist_contract_mode"] = "foundation_seq146"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = _pretrain_manifest_contract_report(
        manifest_path,
        expected_bundle_dir=bundle_dir,
        expected_dataset_dir=dataset_dir,
    )

    assert report["decision"] == "FAIL"
    assert report["specialist_contract_mode"] == "foundation_seq146"
    assert report["specialist_trainable_set_exact"] is False
    assert any("trainable set is not exact" in failure for failure in report["failures"])
