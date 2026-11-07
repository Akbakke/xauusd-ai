"""Seven exact local-side demotions; small-receipt preflight and fail paths."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from gx1.audit import verify_market_state_local_side_exceptions_v1 as audit


def synthetic_masks():
    issue = np.asarray(sorted([audit.base.month_edges()[0] + audit.base.MINUTE_NS,
                               *(row[1] for row in audit.EXCEPTIONS)]), np.int64)
    allowed = np.full((len(issue), len(audit.TFS)), 7, np.uint8)
    endpoint = np.full_like(allowed, -1, dtype=np.int64)
    for tf, stamp, source_row, old_side in audit.EXCEPTIONS:
        i = int(np.searchsorted(issue, stamp))
        j = audit.TFS.index(tf)
        allowed[i, j] = 1 if old_side == 0 else 4
        endpoint[i, j] = source_row
    return {
        "issue_ns": issue,
        "m5_source_row": np.arange(len(issue), dtype=np.int64),
        "lead_minutes": np.asarray(audit.base.LEADS, np.int16),
        "m1_source_row": endpoint,
        "allowed_set": allowed,
        "activity_observed": allowed != 7,
        "activity_active": np.isin(allowed, (1, 4, 5)),
        "side_observed": np.isin(allowed, (1, 4)),
        "side_up": allowed == 4,
        "singleton_observed": np.isin(allowed, (1, 2, 4)),
    }


def counts(mask):
    return {tf: dict(zip(("DOWN", "FLAT", "UP", "ACTIVE", "ALL"),
                         (int(np.count_nonzero(mask[:, j] == v)) for v in audit.VALUES)))
            for j, tf in enumerate(audit.TFS)}


def test_exact_seven_demote_only_side_and_preserve_activity():
    before = synthetic_masks()
    after, changes = audit.correct_seven(before)
    assert len(changes) == 7
    assert np.count_nonzero(before["allowed_set"] != after["allowed_set"]) == 7
    assert all(after["allowed_set"][np.searchsorted(after["issue_ns"], x["issue_ns"]),
                                     audit.TFS.index(x["timeframe"])] == 5 for x in changes)
    for name in ("issue_ns", "m5_source_row", "lead_minutes", "m1_source_row",
                 "activity_observed", "activity_active"):
        assert np.array_equal(before[name], after[name])
    assert before["side_observed"].sum() == 7
    assert after["side_observed"].sum() == after["side_up"].sum() == 0
    assert after["singleton_observed"].sum() == 0
    assert after["activity_observed"].sum() == after["activity_active"].sum() == 7


def test_exact_row_or_bit_mismatch_fails_closed():
    before = synthetic_masks()
    broken = {k: v.copy() for k, v in before.items()}
    i = int(np.searchsorted(broken["issue_ns"], audit.EXCEPTIONS[0][1]))
    j = audit.TFS.index("H4")
    broken["m1_source_row"][i, j] += 1
    with pytest.raises(RuntimeError, match="EXACT_ROW_H4"):
        audit.correct_seven(broken)
    broken = {k: v.copy() for k, v in before.items()}
    broken["allowed_set"][i, j] = 5
    with pytest.raises(RuntimeError, match="EXACT_ROW_H4"):
        audit.correct_seven(broken)


def test_twenty_by_five_monthly_counts_only_seven_deltas(monkeypatch):
    before = synthetic_masks()
    after, changes = audit.correct_seven(before)
    selected = {"monthly": audit.monthly_counts(before["issue_ns"], before["allowed_set"]),
                "summary": {"counts_by_tf": counts(before["allowed_set"])}}
    monkeypatch.setattr(audit, "EXPECTED_CORRECTED_COUNTS", {
        tf: counts(after["allowed_set"])[tf] for tf in ("H4", "D1")})
    monkeypatch.setattr(audit, "EXPECTED_SUMMARY", {
        "activity_observed_cells": 7,
        "side_observed_cells": 0,
        "singleton_observed_cells": 0})
    actual = audit.verify_counts(before, after, selected, changes)
    assert len(actual["monthly"]) == 20
    assert set(actual["monthly"]["2025-03"]) == set(audit.TFS)
    assert actual["monthly"]["2025-03"]["D1"]["5"] == 1
    bad = deepcopy(selected)
    bad["monthly"]["2025-03"]["D1"]["4"] += 1
    with pytest.raises(RuntimeError, match="CORRECTED_MONTHLY_COUNTS"):
        audit.verify_counts(before, after, bad, changes)


def synthetic_receipts(monkeypatch):
    months = [audit.month_name(int(x)) for x in audit.base.month_edges()[:-1]]
    side_monthly = {tf: {name: {"local_unknown": sum(
        1 for x in audit.EXCEPTIONS if x[0] == tf and audit.month_name(x[1]) == name),
        "local_opposite": 0} for name in months}
        for tf in ("M15", "H1", "H4", "D1")}
    examples = {tf: [dict(issue_ns=stamp, m1_source_row=source_row,
                           old_side=old_side, cold_side=-1)
                     for found_tf, stamp, source_row, old_side in audit.EXCEPTIONS
                     if found_tf == tf]
                for tf in side_monthly}
    side_result = {
        "schema_version": "gx1_mtf_state_side_locality_result_v1",
        "status": "COMPLETE", "decision": "DIAGNOSTIC_ONLY",
        "plan": audit.INPUTS["side"]["plan"],
        "old_receipts": audit.mtf.INPUTS,
        "m1_rows": audit.base.M1_ROWS,
        "m5_issues": audit.SCOPE["issue_rows"],
        "model_forwards": 0, "optimizer_steps": 0,
        "test_data_used": False, "target_locked": False,
        "mismatch_examples": examples,
        "totals": {tf: {"endpoints": n, "local_unknown": (6 if tf == "H4" else 1 if tf == "D1" else 0),
                         "local_opposite": 0}
                   for tf, n in (("M15", 45411), ("H1", 44635),
                                 ("H4", 46092), ("D1", 36633))},
        "monthly": side_monthly,
    }
    selected_result = {
        "status": "COMPLETE", "decision": "PASS_SELECTED_ACTIVITY_ROW_MASK_SOURCE_ONLY",
        "row_masks": audit.INPUTS["selected"]["row_masks"],
        "activity_semantics_selected": True,
        "five_tf_target_locked": False,
        "monthly": {name: {} for name in months},
    }
    side_terminal = {"status": "COMPLETE", "decision": "DIAGNOSTIC_ONLY",
                     "exit_code": 0, "result": audit.INPUTS["side"]["result"],
                     "test_data_used": False}
    docs = {
        audit.INPUTS["selected"]["result"]["path"]: selected_result,
        audit.INPUTS["side"]["plan"]["path"]: {
            "stage": audit.mtf.STAGE, "inputs": audit.mtf.INPUTS,
            "scope": audit.mtf.SCOPE, "limits": audit.mtf.LIMITS,
            "test_data_used": False},
        audit.INPUTS["side"]["result"]["path"]: side_result,
        audit.INPUTS["side"]["terminal"]["path"]: side_terminal,
    }
    monkeypatch.setattr(audit.mtf, "small_receipts", lambda: ({}, {}, {}, {}))
    monkeypatch.setattr(audit.base, "read_json", lambda ref: docs[ref["path"]])
    return docs


def test_preclaim_small_receipts_success_and_wrong_terminal_failure(monkeypatch):
    docs = synthetic_receipts(monkeypatch)
    monkeypatch.setattr(audit, "load_selected", lambda *_args:
                        pytest.fail("NPZ read before CLAIM"))
    assert len(audit.check_small_receipts()[1]["mismatch_examples"]["H4"]) == 6
    docs[audit.INPUTS["side"]["terminal"]["path"]]["result"] = {
        "path": "/wrong", "sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="RECEIPT_BINDING"):
        audit.check_small_receipts()


def test_preclaim_rejects_missing_or_changed_exception(monkeypatch):
    docs = synthetic_receipts(monkeypatch)
    result = docs[audit.INPUTS["side"]["result"]["path"]]
    result["mismatch_examples"]["H4"][0]["cold_side"] = 0
    with pytest.raises(RuntimeError, match="EXACT_SEVEN"):
        audit.check_small_receipts()
    result["mismatch_examples"]["H4"][0]["cold_side"] = -1
    result["monthly"]["H4"]["2024-08"]["local_unknown"] = 0
    with pytest.raises(RuntimeError, match="MONTH_EXCEPTIONS_H4_2024-08"):
        audit.check_small_receipts()
