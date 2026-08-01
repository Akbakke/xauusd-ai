from __future__ import annotations

import pytest

from gx1.contracts.gx1_scope_v1 import (
    GX1_DRIFT_ADAPTATION_ALLOWED,
    GX1_ENTRY_TIMEFRAME,
    GX1_EXIT_TIMEFRAME,
    GX1_LIVE_OPERATION_ALLOWED,
    require_offline_scope,
    scope_contract,
)


def test_scope_is_offline_shared_featurebase_only() -> None:
    contract = scope_contract()
    assert contract["scope"] == "OFFLINE_SHARED_FEATUREBASE_ONLY"
    assert contract["entry_timeframe"] == GX1_ENTRY_TIMEFRAME == "M5"
    assert contract["exit_timeframe"] == GX1_EXIT_TIMEFRAME == "M1"
    assert contract["live_operation_allowed"] is GX1_LIVE_OPERATION_ALLOWED is False
    assert contract["drift_adaptation_allowed"] is GX1_DRIFT_ADAPTATION_ALLOWED is False


@pytest.mark.parametrize(
    "operation",
    ("live", "paper", "daemon", "live_tail", "drift", "promotion", "launch"),
)
def test_scope_rejects_operational_and_drift_work(operation: str) -> None:
    with pytest.raises(RuntimeError, match="GX1_OFFLINE_SCOPE_FORBIDDEN"):
        require_offline_scope(operation)


def test_scope_accepts_only_offline_evidence_work() -> None:
    for operation in ("featurebase_build", "offline_train", "offline_oos", "offline_replay", "contract_test"):
        assert require_offline_scope(operation) == operation
