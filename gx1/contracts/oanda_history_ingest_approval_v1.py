"""Narrow, read-only OANDA history-ingest authorization for the 2026-08-29 audit.

This is deliberately *not* an operational-scope expansion.  It admits one
named decision only, on the canonical M5 candle route, for two immutable
publications:

* a direct tape that ends exactly at the sealed TEST boundary; and
* its successor, retained separately as current-market research material.

The successor cannot be a preflight input: callers of the preflight cache
publisher must independently bind a source ending strictly before TEST.
"""

from __future__ import annotations

from gx1_guards.gates import GateError, require_retrain_vedtak

OANDA_HISTORY_INGEST_APPROVAL_SCHEMA_VERSION = (
    "gx1_oanda_history_ingest_approval_v1"
)
OANDA_HISTORY_INGEST_APPROVAL_ID = "OANDA_M5_PRETEST_CURRENT_20260829"
OANDA_HISTORY_PRETEST_START_UTC = "2019-01-01T00:00:00Z"
OANDA_HISTORY_PRETEST_END_UTC = "2026-07-01T00:00:00Z"
_SUCCESSOR_MODE = "CANONICAL_NATIVE_SUCCESSOR_MODE"


def require_approved_oanda_history_ingest(
    *,
    vedtak_id: str | None,
    timeframe: str | None,
    publication_mode: str | None,
    start_utc: str | None,
    end_utc: str | None,
) -> str:
    """Fail closed unless this exact read-only historical authorization applies."""

    vedtak = require_retrain_vedtak(vedtak_id)
    if vedtak != OANDA_HISTORY_INGEST_APPROVAL_ID:
        raise GateError(
            "GX1_OANDA_HISTORY_INGEST_FORBIDDEN: explicit authorization "
            "does not match the one approved read-only history intake."
        )
    if str(timeframe or "").strip().upper() != "M5":
        raise GateError(
            "GX1_OANDA_HISTORY_INGEST_FORBIDDEN: authorization is M5-only."
        )
    mode = str(publication_mode or "").strip()
    if mode == "bootstrap":
        if (
            str(start_utc or "") != OANDA_HISTORY_PRETEST_START_UTC
            or str(end_utc or "") != OANDA_HISTORY_PRETEST_END_UTC
        ):
            raise GateError(
                "GX1_OANDA_HISTORY_INGEST_FORBIDDEN: bootstrap must be the "
                "exact direct-M5 pre-TEST interval."
            )
    elif mode == _SUCCESSOR_MODE:
        # The native successor implementation CAS-binds the parent and proves
        # a byte-exact overlap before append.  It separately enforces a
        # completed end time; its materialization is retained outside the
        # pre-TEST input root.
        if start_utc is not None:
            raise GateError(
                "GX1_OANDA_HISTORY_INGEST_FORBIDDEN: successor start is "
                "inherited exclusively from its immutable parent."
            )
        if not str(end_utc or "").strip():
            raise GateError(
                "GX1_OANDA_HISTORY_INGEST_FORBIDDEN: successor end is required."
            )
    else:
        raise GateError(
            "GX1_OANDA_HISTORY_INGEST_FORBIDDEN: publication mode is not "
            "approved for the read-only history intake."
        )
    return vedtak
