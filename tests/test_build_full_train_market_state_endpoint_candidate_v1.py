from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from gx1.audit import build_full_train_market_state_endpoint_candidate_v1 as worker


def ns(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 10**9)


def test_m5_boundary_group_discards_val_and_prior_bar() -> None:
    starts = np.asarray([
        ns("2024-04-30T23:55:00Z"),
        ns("2024-05-01T00:00:00Z"),
        ns("2025-12-31T23:50:00Z"),
        ns("2025-12-31T23:55:00Z"),
        ns("2026-01-01T00:00:00Z"),
    ], dtype=np.int64)
    keep, issues, rows = worker.bound_m5_issue_clock(
        starts, ns(worker.TRAIN_START), ns(worker.TRAIN_END), np)
    assert keep.tolist() == [True, True, True, True, False]
    assert issues.tolist() == [
        ns("2024-05-01T00:05:00Z"),
        ns("2025-12-31T23:55:00Z"),
    ]
    assert rows.tolist() == [1, 2]


def test_missing_future_endpoint_stays_unknown() -> None:
    counts = {"flat": 1, "down": 0, "up": 0,
              "partial_complete": 0, "partial_early": 1,
              "middle_unknown": 1}
    present = np.asarray([True, True, True, False])
    masks = np.asarray([worker.FLAT, worker.ACTIVE, worker.ALL, worker.ALL],
                       dtype=np.uint8)
    summary = worker.summarize_partition(masks, present, counts, np)
    assert (summary["exact_singleton"], summary["partial_active"],
            summary["unknown_all"], summary["missing_exact_endpoint"],
            summary["other_unknown"]) == (1, 1, 2, 1, 0)
    masks[-1] = worker.FLAT
    with pytest.raises(RuntimeError, match="MISSING_TO_UNKNOWN"):
        worker.summarize_partition(masks, present, counts, np)


def test_exactly_twenty_utc_months() -> None:
    edges = worker.month_edges()
    assert len(edges) == 21
    assert edges[0] == ns(worker.TRAIN_START)
    assert edges[-1] == ns(worker.TRAIN_END)
    assert np.all(np.diff(edges) > 0)
