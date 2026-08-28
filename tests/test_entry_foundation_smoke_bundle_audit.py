from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.contracts.entry_fitted_q_v1 import (
    entry_fitted_q_contract,
    entry_fitted_q_production_economics_readiness,
)
from gx1.contracts.entry_model_native_readiness_v1 import (
    MODEL_NATIVE_ACTIVE_HEADS,
    MODEL_NATIVE_BLOCKED_HEADS,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    training_objective_contract_metadata,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
    model_native_signal_contract_metadata,
)
from gx1.scripts import audit_entry_foundation_smoke_bundle_v1 as audit


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_training_objective_proof_requires_exact_meta_lock_identity(
    tmp_path: Path,
) -> None:
    objective = training_objective_contract_metadata()
    _write_json(
        tmp_path / "bundle_metadata.json",
        {"model_native_training_objective": objective},
    )
    _write_json(
        tmp_path / "MASTER_TRANSFORMER_LOCK.json",
        {"model_native_training_objective": objective},
    )
    report = audit._model_native_training_objective_contract_report(
        bundle_dir=tmp_path,
        metadata={"model_native_training_objective": objective},
    )
    assert report["decision"] == "PASS"

    mutated = json.loads(json.dumps(objective))
    mutated["schema_version"] = "stale"
    _write_json(
        tmp_path / "MASTER_TRANSFORMER_LOCK.json",
        {"model_native_training_objective": mutated},
    )
    report = audit._model_native_training_objective_contract_report(
        bundle_dir=tmp_path,
        metadata={"model_native_training_objective": objective},
    )
    assert report["decision"] == "FAIL"


def test_smoke_audit_contract_is_fitted_q_and_economics_blocked() -> None:
    assert MODEL_NATIVE_ACTIVE_HEADS[0] == "entry_action_q"
    assert "trade_side_hierarchy" in MODEL_NATIVE_BLOCKED_HEADS
    q_contract = entry_fitted_q_contract()
    economics = entry_fitted_q_production_economics_readiness()
    assert q_contract["decision"] == (
        "unique_argmax(entry_action_q_bps_over_valid_actions)"
    )
    assert q_contract["production_authority_ready"] is False
    assert economics["gross_research_training_allowed"] is True
    assert economics["gross_q_production_decision_authority_allowed"] is False
    assert economics["bundle_serving_admission_allowed"] is False


def test_smoke_audit_parser_has_no_implicit_artifact_defaults() -> None:
    parser = audit.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_smoke_bundle_audit_is_cpu_only_even_when_auto_is_requested() -> None:
    assert audit._device_arg("auto") == "cpu"
    with pytest.raises(SystemExit, match="SMOKE_BUNDLE_AUDIT_CPU_ONLY"):
        audit._device_arg("cuda")
    with pytest.raises(SystemExit):
        audit.build_parser().parse_args(["--device", "cuda"])


def test_dataset_manifest_contract_keeps_a_successfully_validated_signal_contract(
    tmp_path: Path,
) -> None:
    """A validator returning None must not erase the proved payload."""

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    parquet = dataset_dir / "fixture_val.parquet"
    parquet.touch()
    signal_contract = model_native_signal_contract_metadata(
        (
            *MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
            *MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
        )
    )
    manifest = dataset_dir / "fixture_val.manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
            "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
            "expected_seq_snap_width": MODEL_NATIVE_SIGNAL_DIM,
            "output_data_path": str(parquet),
            "extra": {
                "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
                "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
                "model_native_signal_contract": signal_contract,
            },
        },
    )

    report, observed = audit._dataset_manifest_contract(
        dataset_dir=dataset_dir,
        manifests={"val": manifest},
    )

    assert report["decision"] == "PASS"
    assert observed == signal_contract


def test_attended_only_bundle_cannot_pass_smoke_bundle_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "require_bundle_commit_manifest", lambda _path: None)
    monkeypatch.setattr(
        audit,
        "_read_json",
        lambda path: {"execution_tier": "attended_only"}
        if path.name == "bundle_metadata.json"
        else {"execution_tier": "attended_only"},
    )
    monkeypatch.setattr(audit, "load_entry_v10_ctx_bundle", lambda **_kwargs: None)

    report, _metadata, _direction, _loaded = audit._bundle_contract_report(
        bundle_dir=tmp_path,
        device="cpu",
    )

    assert report["decision"] == "FAIL"
    assert any("attended-only bundles" in failure for failure in report["failures"])
