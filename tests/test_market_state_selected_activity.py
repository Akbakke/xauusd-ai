"""Focused selected activity mask and preclaim receipt checks."""
from pathlib import Path

import numpy as np
import pytest

from gx1.audit import verify_market_state_selected_activity_v1 as selected


def synthetic_arrays():
    n = selected.SCOPE["m5_issues"]
    expected = np.empty((n, len(selected.TFS)), np.uint8)
    gray = np.zeros_like(expected, dtype=np.bool_)
    for j, tf in enumerate(selected.TFS):
        start = 0
        for value, count in zip(selected.VALUES, selected.EXPECTED_COUNTS[tf]):
            expected[start:start + count, j] = value
            start += count
        assert start == n
        active_rows = np.flatnonzero(expected[:, j] == 5)
        gray[active_rows[:selected.EXPECTED_GRAY[j]], j] = True
    old_set = expected.copy()
    old_set[gray] = 7
    single_set = old_set.copy()
    single_set[gray] = 2
    old = {
        "issue_ns": np.arange(n, dtype=np.int64) * selected.base.MINUTE_NS
                    + selected.base.month_edges()[0],
        "m5_source_row": np.arange(n, dtype=np.int64),
        "lead_minutes": np.asarray(selected.base.LEADS, np.int16),
        "allowed_set": old_set,
        "m1_source_row": np.full_like(expected, -1, dtype=np.int64),
    }
    ratios = np.ones_like(expected, dtype=np.float64)
    ratios[gray] = 0.75
    single = {**old, "allowed_set": single_set,
              "complete_range_atr": ratios,
              "complete_range_atr_valid": gray.copy(),
              "atr_to_quoted_spread": np.ones_like(ratios),
              "atr_to_quoted_spread_valid": np.zeros_like(gray)}
    cohorts = {"below_0_5": np.zeros_like(gray),
               "gray_0_5_to_1_0": gray,
               "at_least_1_0": np.zeros_like(gray)}
    return old, single, cohorts, expected


def test_changes_only_273200_gray_cells_and_separates_activity_side():
    old, single, cohorts, expected = synthetic_arrays()
    out, report = selected.select_masks(old, single, cohorts)
    gray = cohorts["gray_0_5_to_1_0"]
    assert np.array_equal(out["allowed_set"], expected)
    assert report["changed_cells"] == 273200
    assert report["unchanged_cells"] == expected.size - 273200
    assert report["counts_by_tf"]["M5"] == {
        "DOWN": 18921, "FLAT": 8182, "UP": 20749,
        "ACTIVE": 70346, "ALL": 508}
    assert out["activity_observed"][gray].all()
    assert out["activity_active"][gray].all()
    assert not out["side_observed"][gray].any()
    assert not out["singleton_observed"][gray].any()
    assert not out["side_up"][gray].any()
    assert np.array_equal(out["m1_source_row"], old["m1_source_row"])
    assert np.array_equal(out["issue_ns"], old["issue_ns"])


def test_rejects_gray_source_or_side_mismatch():
    old, single, cohorts, _ = synthetic_arrays()
    r = int(np.flatnonzero(cohorts["gray_0_5_to_1_0"][:, 0])[0])
    old["allowed_set"][r, 0] = 4
    with pytest.raises(RuntimeError, match="GRAY_IDENTITY"):
        selected.select_masks(old, single, cohorts)
    old, single, cohorts, _ = synthetic_arrays()
    single["allowed_set"][r, 0] = 4
    with pytest.raises(RuntimeError, match="GRAY_IDENTITY"):
        selected.select_masks(old, single, cohorts)
    old, single, cohorts, _ = synthetic_arrays()
    single["complete_range_atr"][r, 0] = 1.0
    with pytest.raises(RuntimeError, match="GRAY_IDENTITY"):
        selected.select_masks(old, single, cohorts)


def test_monthly_partition_preserves_all_five_timeframes():
    old, single, cohorts, _ = synthetic_arrays()
    out, _ = selected.select_masks(old, single, cohorts)
    monthly = selected.monthly_counts(out["issue_ns"], out["allowed_set"])
    assert len(monthly) == 20
    assert sum(monthly[m]["M5"]["5"] for m in monthly) == 70346
    assert sum(monthly[m]["D1"]["7"] for m in monthly) == 36119
    bad = out["allowed_set"].copy()
    bad[0, 0] = 0
    with pytest.raises(RuntimeError, match="MONTH_AXIS"):
        selected.monthly_counts(out["issue_ns"], bad)


def test_each_month_and_tf_gray_relabel_matches_saved_receipt():
    old, single, cohorts, _ = synthetic_arrays()
    out, _ = selected.select_masks(old, single, cohorts)
    issues = out["issue_ns"]
    edges = np.asarray(selected.base.month_edges(), np.int64)
    assigned = np.searchsorted(edges[1:], issues, side="right")
    import time
    monthly = {}
    for i, edge in enumerate(edges[:-1]):
        name = time.strftime("%Y-%m", time.gmtime(edge // 10**9))
        monthly[name] = {}
        for j, tf in enumerate(selected.TFS):
            changed = (old["allowed_set"][assigned == i, j] == 7) & (
                out["allowed_set"][assigned == i, j] == 5)
            monthly[name][tf] = {"comparison_to_old": {
                "old_gray_to_flat": int(np.count_nonzero(changed))}}
    receipt = {"monthly": monthly}
    report = selected.verify_monthly_relabels(
        issues, old["allowed_set"], out["allowed_set"], receipt)
    assert sum(report[m][tf] for m in report for tf in selected.TFS) == 273200
    monthly["2024-05"]["M5"]["comparison_to_old"]["old_gray_to_flat"] += 1
    with pytest.raises(RuntimeError, match="RELABEL_MONTH_2024-05_M5"):
        selected.verify_monthly_relabels(
            issues, old["allowed_set"], out["allowed_set"], receipt)


def synthetic_receipts(monkeypatch):
    stage = "synthetic_selected_activity_preflight_test"
    monkeypatch.setattr(selected, "STAGE", stage)
    plan_ref = {"path": "/synthetic/selected_plan.json", "sha256": "1" * 64}
    plan = {
        "schema_version": selected.PLAN_SCHEMA,
        "stage": stage, "status": "PREPARED_NOT_ADMITTED",
        "inputs": selected.INPUTS,
        "source_owners": {
            name: {"path": str(selected.REPO / rel), "sha256": "2" * 64}
            for name, rel in selected.SOURCE_PATHS.items()},
        "scope": selected.SCOPE, "limits": selected.LIMITS,
        "output_dir": "/home/andre2/GX1_RUNS/" + stage.upper(),
        "test_data_used": False,
    }
    old_plan = {"inputs": selected.gray.INPUTS,
                "schema_version": selected.gray.PLAN_SCHEMA,
                "stage": selected.gray.STAGE}
    gray_result = {
        "schema_version": "gx1_market_state_gray_physical_result_v1",
        "status": "COMPLETE", "decision": "DIAGNOSTIC_ONLY_TARGET_OPEN",
        "plan": selected.INPUTS["gray_plan"],
        "source_refs": {"candidates": selected.gray.INPUTS},
        "m5_issues": selected.SCOPE["m5_issues"],
        "model_forwards": 0, "optimizer_steps": 0,
        "test_data_used": False, "target_locked": False,
        "total": {tf: {"gray_0_5_to_1_0": {"n": n}}
                  for tf, n in zip(selected.TFS, selected.EXPECTED_GRAY)},
    }
    gray_terminal = {
        "status": "COMPLETE", "decision": "DIAGNOSTIC_ONLY_TARGET_OPEN",
        "exit_code": 0, "result": selected.INPUTS["gray_result"],
        "test_data_used": False,
    }
    docs = {plan_ref["path"]: plan,
            selected.INPUTS["gray_plan"]["path"]: old_plan,
            selected.INPUTS["gray_result"]["path"]: gray_result,
            selected.INPUTS["gray_terminal"]["path"]: gray_terminal}
    monkeypatch.setattr(selected.base, "read_json", lambda ref: docs[ref["path"]])
    monkeypatch.setattr(selected.base, "checked", lambda *_args, **_kwargs:
                        (b"{}", (1, 2, 3, 4, 5, 6)))
    monkeypatch.setattr(selected.gray, "check_plan", lambda *_args:
                        ({}, {"synthetic": True}))
    monkeypatch.setattr(selected.gray, "git_state", lambda:
                        ("c" * 40, "work/gx1-current", ""))
    return plan_ref, docs


def test_preflight_checks_small_receipts_without_claim_or_payload(monkeypatch):
    plan_ref, _ = synthetic_receipts(monkeypatch)
    monkeypatch.setattr(selected, "check_claim_and_policy", lambda *_args:
                        pytest.fail("claim read in preflight"))
    monkeypatch.setattr(selected.gray, "load_masks", lambda *_args:
                        pytest.fail("NPZ read in preflight"))
    report = selected.preflight_read_only(plan_ref)
    assert report["status"] == "READY_NO_CLAIM_NO_PAYLOAD"
    assert report["gray_cells_expected"] == 273200
    assert report["source_owner_refs_checked"] == len(selected.SOURCE_PATHS)
    assert report["claim_created"] is False
    assert report["npz_or_raw_payload_read"] is False


def test_preflight_rejects_wrong_gray_receipt(monkeypatch):
    plan_ref, docs = synthetic_receipts(monkeypatch)
    result = docs[selected.INPUTS["gray_result"]["path"]]
    result["total"]["H4"]["gray_0_5_to_1_0"]["n"] -= 1
    with pytest.raises(RuntimeError, match="GRAY_RECEIPT_COUNTS"):
        selected.preflight_read_only(plan_ref)
    result["total"]["H4"]["gray_0_5_to_1_0"]["n"] += 1
    docs[selected.INPUTS["gray_terminal"]["path"]]["exit_code"] = 1
    with pytest.raises(RuntimeError, match="GRAY_RECEIPT"):
        selected.preflight_read_only(plan_ref)
