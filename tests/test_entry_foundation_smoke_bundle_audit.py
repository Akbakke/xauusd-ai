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

