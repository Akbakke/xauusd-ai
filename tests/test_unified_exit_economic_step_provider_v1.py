from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts import unified_exit_economics_objective_v2 as economics
from gx1.contracts.unified_exit_economic_step_provider_v1 import (
    LazyUnifiedExitEconomicStepProviderV1,
)
from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import canonical_sha256


AUTHORITY_PATH = Path(
    "/home/andre2/src/GX1_EXIT_LIFECYCLE_V2/docs/evidence/"
    "UNIFIED_EXIT_PROSPECTIVE_COST_POLICY_V1_20260910/parameter_authority.json"
)


def _readiness(policy_sha256: str) -> dict:
    hurdle = economics.seal_frozen_capital_hurdle_owner_artifact(
        {
            "schema_version": economics.FROZEN_CAPITAL_HURDLE_SCHEMA_VERSION,
            "decision": "PASS",
            "owner_kind": "project_preregistered_prospective_policy",
            "applicable_splits": ["train", "val"],
            "fitted_splits": [],
            "validation_or_test_used": False,
            "train_split_sha256": "1" * 64,
            "train_fold_sha256": "2" * 64,
            "source_lineage_sha256": "3" * 64,
            "effective_annual_return_hurdle": 0.10,
            "annual_continuous_hurdle_rate": math.log1p(0.10),
            "rate_conversion_formula": "rho=ln(1+effective_annual_return)",
            "rate_unit": "continuous_per_wall_clock_year",
            "seconds_per_year": economics.SECONDS_PER_YEAR,
            "source_method": "project_preregistered_before_val_v1",
            "source_method_artifact_sha256": "4" * 64,
        }
    )
    objective = economics.build_unified_exit_economics_objective_contract(
        capital_hurdle_artifact=hurdle,
        expected_train_split_sha256="1" * 64,
        expected_train_fold_sha256="2" * 64,
        expected_source_lineage_sha256="3" * 64,
        policy_sha256=policy_sha256,
    )
    return {
        "schema_version": "gx1_unified_exit_training_economics_readiness_v3",
        "mode": "economics_objective_v2",
        "capital_hurdle_artifact": hurdle,
        "economics_objective_contract": objective,
        "expected_train_split_sha256": "1" * 64,
        "expected_train_fold_sha256": "2" * 64,
        "expected_source_lineage_sha256": "3" * 64,
        "policy_sha256": policy_sha256,
        "proper_policy_certificate_sha256": None,
        "test_data_used": False,
    }


def _provider(
    *,
    readiness_policy_sha256: str | None = None,
    economic_terminal: bool = False,
):
    authority = json.loads(AUTHORITY_PATH.read_text())
    policy = json.loads(Path(authority["policy"]["path"]).read_text())
    tape_path = Path(policy["executable_bid_ask"]["parquet"]["path"])
    times = pd.DatetimeIndex(
        pd.to_datetime(pd.read_parquet(tape_path, columns=["time"])["time"], utc=True)
    )
    start = int(times.searchsorted(pd.Timestamp("2025-06-02T00:00:00Z")))
    assert times[start + 1] - times[start] == pd.Timedelta(minutes=1)
    compact = pd.DataFrame(
        {
            "entry_row_index": [0],
            "entry_m1_start_row": [start],
            "long_lifecycle_state_count": [3],
            "short_lifecycle_state_count": [3],
            "long_economic_terminal": [economic_terminal],
            "short_economic_terminal": [economic_terminal],
            "m1_source_sha256": [policy["executable_bid_ask"]["parquet"]["sha256"]],
        }
    )
    manifest = {
        "split": "train",
        "test_accessed": False,
        "gap_classification_source_sha256": policy["executable_bid_ask"]["manifest"][
            "sha256"
        ],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    readiness = _readiness(readiness_policy_sha256 or authority["authority_sha256"])
    provider = LazyUnifiedExitEconomicStepProviderV1(
        compact_rows=compact,
        compact_manifest=manifest,
        economics_readiness=readiness,
        cost_parameter_authority_path=AUTHORITY_PATH,
    )
    return provider, readiness


def test_production_provider_consumes_committed_source_rich_authority() -> None:
    provider, readiness = _provider()
    long_exit = provider(0, 0, "exit_now", 0, 1)["steps"][0]
    short_exit = provider(0, 1, "exit_now", 0, 1)["steps"][0]
    assert long_exit["execution_slippage"]["value_bps"] == 4.0
    assert short_exit["execution_slippage"]["value_bps"] == 4.0
    hold = provider(0, 0, "hold", 0, 1)["steps"][0]
    assert hold["financing_or_swap"]["value_bps"] == pytest.approx(
        -540.0 * 60.0 / economics.SECONDS_PER_YEAR
    )
    assert hold["risk_utility_penalty"]["value_bps"] == 0.0
    assert (
        provider.economic_exit_step_manifest["economics_objective_contract_sha256"]
        == readiness["economics_objective_contract"]["contract_sha256"]
    )


def test_provider_rejects_readiness_not_bound_to_cost_authority() -> None:
    with pytest.raises(RuntimeError, match="POLICY_BINDING_INVALID"):
        _provider(readiness_policy_sha256="f" * 64)


def test_vectorized_training_projection_is_byte_exact_to_scalar_composition() -> None:
    provider, readiness = _provider()
    contract = readiness["economics_objective_contract"]
    for side_index in (0, 1):
        projection = provider.materialize_training_projection(
            0, side_index, 0, 3, 2
        )
        scalar_exit = np.asarray(
            [
                economics.compose_economic_step(step, contract=contract)[
                    "undiscounted_risk_adjusted_utility_increment_bps"
                ]
                for step in provider(0, side_index, "exit_now", 0, 3)["steps"]
            ],
            dtype=np.float32,
        )
        scalar_hold = np.asarray(
            [
                economics.compose_economic_step(step, contract=contract)[
                    "undiscounted_risk_adjusted_utility_increment_bps"
                ]
                for step in provider(0, side_index, "hold", 0, 2)["steps"]
            ],
            dtype=np.float32,
        )
        vector_exit = np.asarray(projection["exit_reward_bps"], dtype=np.float32)
        vector_hold = np.asarray(projection["hold_reward_bps"], dtype=np.float32)
        assert vector_exit.tobytes() == scalar_exit.tobytes()
        assert vector_hold.tobytes() == scalar_hold.tobytes()
        assert not projection["exit_reward_bps"].flags.writeable
        assert not projection["hold_reward_bps"].flags.writeable


def test_terminal_event_identity_matches_scalar_and_vectorized_paths() -> None:
    provider, _ = _provider(economic_terminal=True)
    scalar = provider(0, 0, "exit_now", 0, 3)
    projection = provider.materialize_training_projection(0, 0, 0, 3, 2)
    assert [step["event_kind"] for step in scalar["steps"]] == [
        "EXIT_NOW",
        "EXIT_NOW",
        "ECONOMIC_TERMINAL",
    ]
    assert projection["exit_event_kind_index"].tolist() == [0, 0, 2]
