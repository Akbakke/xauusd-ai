"""Narrow, read-only OANDA history-ingest authorization.

This is deliberately *not* an operational-scope expansion.  Each named
decision admits exactly one canonical candle route (M1 or M5) and one
bootstrap start, for two immutable publications:

* a direct tape that ends exactly at the sealed TEST boundary; and
* its successor, retained separately as current-market research material.

The successor cannot be a preflight input: callers of the preflight cache
publisher must independently bind a source ending strictly before TEST.
"""

from __future__ import annotations

from gx1.contracts.xau_tape_provenance_v1 import (
    CANONICAL_NATIVE_SUCCESSOR_MODE,
)
from gx1_guards.gates import GateError, require_retrain_vedtak

OANDA_HISTORY_INGEST_APPROVAL_SCHEMA_VERSION = (
    "gx1_oanda_history_ingest_approval_v1"
)
OANDA_HISTORY_PRETEST_END_UTC = "2026-07-01T00:00:00Z"
# decision id -> (admitted timeframes, exact bootstrap start). Every bootstrap
# ends at the sealed TEST boundary above.
# 2026-08-29: the pre-TEST audit intake from 2019 (one id per timeframe).
# 2026-09-26: operator decision "hent fra 2005" -- falling gold markets
# (2008, 2011-2015, 2016, 2018) are absent from the 2019 tape. One id admits
# both timeframes because the canonical pair producer and its lineage
# validator require M1 and M5 to carry the same source decision
# (gx1/execution/v12_canonical_incremental.py, v12_state_from_prebuilt.py).
# The first 2005 intake used one id per timeframe and cannot form a pair; its
# tapes remain immutable history.
OANDA_HISTORY_INGEST_APPROVALS = {
    "OANDA_M1_PRETEST_CURRENT_20260829": (frozenset({"M1"}), "2019-01-01T00:00:00Z"),
    "OANDA_M5_PRETEST_CURRENT_20260829": (frozenset({"M5"}), "2019-01-01T00:00:00Z"),
    "OANDA_PAIR_PRETEST_2005_20260927": (frozenset({"M1", "M5"}), "2005-01-01T00:00:00Z"),
}
_SUCCESSOR_MODE = CANONICAL_NATIVE_SUCCESSOR_MODE


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
    normalized_timeframe = str(timeframe or "").strip().upper()
    if normalized_timeframe not in {"M1", "M5"}:
        raise GateError(
            "GX1_OANDA_HISTORY_INGEST_FORBIDDEN: authorization is limited "
            "to canonical M1 or M5 history."
        )
    approval = OANDA_HISTORY_INGEST_APPROVALS.get(vedtak)
    if approval is None or normalized_timeframe not in approval[0]:
        raise GateError(
            "GX1_OANDA_HISTORY_INGEST_FORBIDDEN: explicit authorization "
            "does not match an approved read-only history intake."
        )
    approved_start = approval[1]
    mode = str(publication_mode or "").strip()
    if mode == "bootstrap":
        if (
            str(start_utc or "") != approved_start
            or str(end_utc or "") != OANDA_HISTORY_PRETEST_END_UTC
        ):
            raise GateError(
                "GX1_OANDA_HISTORY_INGEST_FORBIDDEN: bootstrap must be the "
                "exact approved pre-TEST interval."
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
