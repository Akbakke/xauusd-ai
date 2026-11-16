"""Exact-clock TRAIN state lookup for future queries and current-only M5 issues.

The caller owns the query schedule and the hash-bound M1 tape. Current-only
lookup returns the state at the issue clock and never reads a future cell.
Neither path publishes targets or authorizes fit.
"""
from __future__ import annotations

import numpy as np

TIMEFRAMES = ("M5", "M15", "H1", "H4", "D1")
NATIVE_K = (1, 3, 12, 48, 288)
VALID_BITS = (1, 2, 4, 5, 7)
MINUTE_NS = 60_000_000_000
TICK_NS = 5 * MINUTE_NS


def _require(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError("MARKET_STATE_TIME_QUERY_" + code)


class TimeQueryTargetTape:
    """Validate a source tape once, then gather only exact mature query rows."""

    def __init__(self, validity_ns: np.ndarray, source_row: np.ndarray,
                 allowed_bits: np.ndarray, *, path_observed: np.ndarray | None = None,
                 gap_kind: np.ndarray | None = None):
        _require(isinstance(validity_ns, np.ndarray)
                 and validity_ns.dtype == np.int64 and validity_ns.ndim == 1, "TAPE_AXIS")
        n = len(validity_ns)
        _require(isinstance(source_row, np.ndarray)
                 and source_row.dtype == np.int64 and source_row.shape == (n,)
                 and isinstance(allowed_bits, np.ndarray)
                 and allowed_bits.dtype == np.uint8
                 and allowed_bits.shape == (n, len(TIMEFRAMES))
                 and n > 0, "TAPE_SHAPE")
        _require(bool(np.all(validity_ns >= 0))
                 and bool(np.all(validity_ns % MINUTE_NS == 0))
                 and bool(np.all(np.diff(validity_ns) > 0))
                 and bool(np.all(source_row >= 0))
                 and bool(np.all(np.diff(source_row) > 0))
                 and bool(np.isin(allowed_bits, VALID_BITS).all()), "TAPE_DOMAIN")
        # Bytes backing makes caller mutation and write-flag reversal ineffective.
        self.validity_ns = np.frombuffer(np.ascontiguousarray(validity_ns).tobytes(), dtype=np.int64)
        self.source_row = np.frombuffer(np.ascontiguousarray(source_row).tobytes(), dtype=np.int64)
        self.allowed_bits = np.frombuffer(np.ascontiguousarray(allowed_bits).tobytes(),
                                          dtype=np.uint8).reshape(n, len(TIMEFRAMES))
        _require((path_observed is None) == (gap_kind is None), "CURRENT_AUX_PAIR")
        if path_observed is not None:
            _require(isinstance(path_observed, np.ndarray)
                     and path_observed.dtype == np.bool_
                     and path_observed.shape == allowed_bits.shape
                     and bool(np.all(path_observed <= np.isin(allowed_bits, (1, 2, 4))))
                     and isinstance(gap_kind, np.ndarray)
                     and gap_kind.dtype == np.uint8 and gap_kind.shape == (n,)
                     and bool(np.isin(gap_kind, (0, 1, 2)).all()), "CURRENT_AUX_DOMAIN")
            self.path_observed = np.frombuffer(
                np.ascontiguousarray(path_observed).tobytes(), dtype=np.bool_).reshape(n, len(TIMEFRAMES))
            self.gap_kind = np.frombuffer(np.ascontiguousarray(gap_kind).tobytes(), dtype=np.uint8)
        else:
            self.path_observed = None
            self.gap_kind = None

    def gather_current(self, issue_ns: np.ndarray) -> dict[str, np.ndarray | str]:
        """Read only the exact closed M1 state at each M5 issue, never a future row."""
        _require(self.path_observed is not None and self.gap_kind is not None,
                 "CURRENT_AUX_REQUIRED")
        _require(isinstance(issue_ns, np.ndarray) and issue_ns.dtype == np.int64
                 and issue_ns.ndim == 1 and len(issue_ns) > 0
                 and bool(np.all(issue_ns >= 0))
                 and bool(np.all(issue_ns % TICK_NS == 0)), "CURRENT_CLOCK")
        at = np.searchsorted(self.validity_ns, issue_ns)
        present = at < len(self.validity_ns)
        present[present] &= self.validity_ns[at[present]] == issue_ns[present]
        allowed = np.full((len(issue_ns), len(TIMEFRAMES)), 7, dtype=np.uint8)
        source = np.full(len(issue_ns), -1, dtype=np.int64)
        path = np.zeros_like(allowed, dtype=np.bool_)
        gap = np.zeros(len(issue_ns), dtype=np.uint8)
        if bool(present.any()):
            selected = at[present]
            allowed[present] = self.allowed_bits[selected]
            source[present] = self.source_row[selected]
            path[present] = self.path_observed[selected]
            gap[present] = self.gap_kind[selected]
        _require(bool(np.all((source >= 0) == present)), "CURRENT_SOURCE_CLOCK")
        return {"issue_ns": issue_ns.copy(), "allowed": allowed,
                "m1_source_row": source, "path_observed": path,
                "gap_kind": gap, "present": present,
                "activity_known": present[:, None] & (allowed != 7),
                "side_known": present[:, None] & np.isin(allowed, (1, 4)),
                "clock_basis": "historical_closed_M1_event_time_not_online_delivery"}

    def gather(self, issue_ns: np.ndarray, query_k: np.ndarray, *,
               fold_upper_ns: int) -> dict[str, np.ndarray]:
        """Return five allowed sets at v=t+5k, with immature queries censored first."""
        _require(isinstance(issue_ns, np.ndarray)
                 and issue_ns.dtype == np.int64 and issue_ns.ndim == 1
                 and len(issue_ns) > 0
                 and isinstance(query_k, np.ndarray)
                 and query_k.dtype == np.int64
                 and query_k.shape == (len(issue_ns), len(TIMEFRAMES))
                 and type(fold_upper_ns) is int and fold_upper_ns > 0,
                 "QUERY_SHAPE")
        _require(bool(np.all(issue_ns >= 0))
                 and bool(np.all(issue_ns % TICK_NS == 0))
                 and bool(np.all(issue_ns < fold_upper_ns))
                 and bool(np.all(query_k >= 1))
                 and bool(np.all(query_k <= np.asarray(NATIVE_K, dtype=np.int64)[None, :]))
                 and bool(np.all(issue_ns <= np.iinfo(np.int64).max
                                 - 288 * TICK_NS)), "QUERY_DOMAIN")
        validity = issue_ns[:, None] + query_k * TICK_NS
        mature = validity < fold_upper_ns
        allowed = np.full(query_k.shape, 7, dtype=np.uint8)
        source = np.full(query_k.shape, -1, dtype=np.int64)
        # Never read the M1 tape for an immature future query.
        wanted = validity[mature]
        at = np.searchsorted(self.validity_ns, wanted)
        present = at < len(self.validity_ns)
        present[present] &= self.validity_ns[at[present]] == wanted[present]
        mature_positions = np.flatnonzero(mature)
        selected = mature_positions[present]
        allowed.flat[selected] = self.allowed_bits[at[present],
                                                  selected % len(TIMEFRAMES)]
        source.flat[selected] = self.source_row[at[present]]
        eligible = mature & (allowed != 7) & (source >= 0)
        return {"issue_ns": issue_ns.copy(), "query_k": query_k.copy(),
                "validity_ns": validity, "allowed": allowed,
                "m1_source_row": source, "mature": mature,
                "eligible": eligible}
