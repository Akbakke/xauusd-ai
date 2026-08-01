"""Single active GX1 scope: offline shared-featurebase work only."""

from __future__ import annotations

GX1_SCOPE_SCHEMA_VERSION = "gx1_offline_shared_featurebase_scope_v1"
GX1_SCOPE_NAME = "OFFLINE_SHARED_FEATUREBASE_ONLY"
GX1_INSTRUMENT = "XAUUSD"
GX1_ENTRY_TIMEFRAME = "M5"
GX1_EXIT_TIMEFRAME = "M1"
GX1_LIVE_OPERATION_ALLOWED = False
GX1_DRIFT_ADAPTATION_ALLOWED = False

ALLOWED_SCOPE_OPERATIONS = frozenset(
    {
        "featurebase_build",
        "offline_train",
        "offline_oos",
        "offline_replay",
        "contract_test",
    }
)


def require_offline_scope(operation: str) -> str:
    """Reject live/operational work at the active contract boundary."""

    value = str(operation or "").strip().lower()
    if value not in ALLOWED_SCOPE_OPERATIONS:
        raise RuntimeError(
            "GX1_OFFLINE_SCOPE_FORBIDDEN: "
            f"operation={operation!r} allowed={sorted(ALLOWED_SCOPE_OPERATIONS)}"
        )
    return value


def scope_contract() -> dict[str, object]:
    return {
        "schema_version": GX1_SCOPE_SCHEMA_VERSION,
        "scope": GX1_SCOPE_NAME,
        "instrument": GX1_INSTRUMENT,
        "entry_timeframe": GX1_ENTRY_TIMEFRAME,
        "exit_timeframe": GX1_EXIT_TIMEFRAME,
        "live_operation_allowed": GX1_LIVE_OPERATION_ALLOWED,
        "drift_adaptation_allowed": GX1_DRIFT_ADAPTATION_ALLOWED,
        "allowed_operations": sorted(ALLOWED_SCOPE_OPERATIONS),
    }
