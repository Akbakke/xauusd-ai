from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gx1.scripts import exit_iql_artifact_primitives_v1 as gate


EXIT_IQL_PRIMITIVE_CONSUMERS = (
    "materialize_build_candidate_forward_outcome_dataset_v1.py",
    "materialize_build_canonical_features_v1.py",
    "materialize_build_canonical_features_v2.py",
    "materialize_build_exit_iql_per_bar_dataset_v1.py",
    "materialize_build_exit_iql_per_bar_dataset_v2_m1.py",
    "materialize_build_exit_iql_v2.py",
    "materialize_exit_hold_exit_now_mdp_reward_contract_v1.py",
    "materialize_exit_off_policy_eval_harness_v1.py",
    "materialize_exit_per_bar_state_feature_contract_v1.py",
    "materialize_run_exit_iql_with_v2_state_and_reward_variants_v1.py",
)


def test_exit_iql_artifact_roots_fail_closed() -> None:
    assert gate.validate_explicit_artifact_roots([Path("/tmp/EXIT_IQL_V1_20260716T000000Z_LOCK")])

    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/EXIT_IQL_UNLOCKED")])


def test_exit_iql_forbidden_action_gate_preserves_contract() -> None:
    assert gate.validate_no_forbidden_actions() == {"status_v1": "PASS", "failures_v1": []}

    blocked = gate.validate_no_forbidden_actions(
        r6=True,
        adapter=True,
        iql_production=True,
        iql_training_now=True,
        package=True,
        freeze=True,
        promo=True,
        live=True,
        optuna=True,
        broad_sweep=True,
    )
    assert blocked == {
        "status_v1": "FAIL",
        "failures_v1": [
            "R6_FORBIDDEN",
            "ADAPTER_BUILD_FORBIDDEN",
            "IQL_PRODUCTION_FORBIDDEN",
            "IQL_TRAINING_FORBIDDEN_IN_CONTRACT_GATE",
            "PACKAGE_BUILD_FORBIDDEN",
            "FREEZE_FORBIDDEN",
            "PROMO_FORBIDDEN",
            "LIVE_FORBIDDEN",
            "OPTUNA_FORBIDDEN",
            "BROAD_SWEEP_FORBIDDEN",
        ],
    }


def test_exit_iql_json_serialization_preserves_numpy_and_path_behavior(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    gate._write_json(
        output,
        {
            "path_v1": tmp_path,
            "integer_v1": np.int64(7),
            "float_v1": np.float64(2.5),
            "bool_v1": np.bool_(True),
            "nan_v1": np.float64(np.nan),
        },
    )

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "bool_v1": True,
        "float_v1": 2.5,
        "integer_v1": 7,
        "nan_v1": None,
        "path_v1": str(tmp_path),
    }


def test_exit_iql_consumers_do_not_import_retired_entry_contract() -> None:
    scripts_root = Path(__file__).resolve().parents[1] / "gx1" / "scripts"
    retired = "materialize_build_iql_offline_data_contract_research_only_v1"
    expected = "exit_iql_artifact_primitives_v1 as contract_gate"

    for filename in EXIT_IQL_PRIMITIVE_CONSUMERS:
        source = (scripts_root / filename).read_text(encoding="utf-8")
        assert retired not in source, filename
        assert expected in source, filename
