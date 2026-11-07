"""Focused selected-mask, locality, and receipt checks without raw TRAIN reads."""
import io
import json
from pathlib import Path

import numpy as np
import pytest

from gx1.audit import verify_mtf_state_side_locality_v1 as audit
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target


def test_cold_replay_matches_m5_owner_on_same_window_and_resets_at_unknown_gap():
    n = 8
    open_ = np.array([100., 100., 100., 100., 100., 100., 100., 100.])
    high = np.array([100.2, 100.8, 101.3, 101.2, 101.1, 100.7, 100.6, 100.6])
    low = np.array([99.8, 99.9, 100.1, 100.2, 100.1, 99.5, 99.6, 99.7])
    close = np.array([100., 100.7, 101.2, 100.9, 100.8, 99.6, 99.9, 100.2])
    theta = np.full(n, 0.7)
    gap = np.zeros(n, bool)
    actual = audit.cold_side(7, 5, open_, high, low, close, theta, gap, target)
    expected = audit.m5.local_side(7, open_, high, low, close, theta, gap, target)
    assert actual == expected
    gap[3] = True
    actual = audit.cold_side(7, 5, open_, high, low, close, theta, gap, target)
    expected = audit.m5.local_side(7, open_, high, low, close, theta, gap, target)
    assert actual == expected and actual[1] is True
    with pytest.raises(RuntimeError, match="LOCAL_ROW"):
        audit.cold_side(2, 5, open_, high, low, close, theta, gap, target)


def test_side_report_partitions_same_unknown_opposite_and_old_entry():
    idx = np.arange(4)
    local = np.array([target.UP, target.UNKNOWN, target.DOWN, target.UP], np.int8)
    expected = np.array([target.UP, target.UP, target.UP, target.DOWN], np.int8)
    age = np.array([-1, 1440, -1, 60], np.int32)
    before = np.array([False, True, False, True])
    flat = np.array([False, True, False, False])
    closure = np.array([False, False, True, False])
    report = audit.summarize(idx, local, expected, age, before, flat, closure)
    assert report == {
        "endpoints": 4, "local_same": 1, "local_unknown": 1,
        "local_opposite": 2, "side_entry_older_than_local_window": 2,
        "side_entry_at_or_before_issue": 2, "flat_since_side_entry": 1,
        "known_closure_touching_window": 1,
    }
    with pytest.raises(RuntimeError, match="SUMMARY_SIDE"):
        audit.summarize(idx, np.array([1, -1, 0, 2], np.int8), expected,
                        age, before, flat, closure)


def synthetic_receipts(monkeypatch):
    full_plan = {
        "schema_version": audit.base.PLAN_SCHEMA,
        "test_data_used": False,
        "train_start_utc": audit.base.TRAIN_START,
        "train_end_utc_exclusive": audit.base.TRAIN_END,
        "lead_minutes": list(audit.base.LEADS),
        "factors": [0.5, 1.0],
        "m1_expected_rows": audit.base.M1_ROWS,
        "m1": {"source": "m1"}, "m5": {"source": "m5"},
        "closure": {"authority": "closure"},
    }
    monthly = {"2024-05": {
        tf: {"down": count // 2, "up": count - count // 2}
        for tf, _, _, count in audit.TF_SPECS}}
    full_result = {
        "decision": "PASS_FULL_TRAIN_CANDIDATE_SOURCE_ONLY",
        "plan": audit.FULL_REFS["plan"],
        "row_masks": audit.FULL_REFS["row_masks"],
        "m1_rows": audit.base.M1_ROWS,
        "m5_issues": audit.SCOPE["issue_rows"],
        "model_calls": 0, "optimizer_steps": 0,
        "test_data_used": False, "target_locked": False,
        "monthly": monthly,
    }
    full_terminal = {"status": "COMPLETE", "exit_code": 0,
                     "result": audit.FULL_REFS["result"],
                     "row_masks": audit.FULL_REFS["row_masks"],
                     "test_data_used": False}
    single_plan = {"schema_version": "gx1_single_threshold_market_state_candidate_plan_v1",
                   "factor": 1.0, "test_data_used": False,
                   "m1": full_plan["m1"], "m5": full_plan["m5"],
                   "closure": full_plan["closure"],
                   "lead_minutes": list(audit.base.LEADS)}
    single_result = {
        "decision": "PASS_SINGLE_THRESHOLD_CANDIDATE_SOURCE_ONLY",
        "plan": audit.m5.PREVIOUS_REFS["plan"],
        "row_masks_and_ratios": audit.m5.PREVIOUS_REFS["row_masks"],
        "m1_rows": audit.base.M1_ROWS,
        "m5_issues": audit.SCOPE["issue_rows"],
        "model_calls": 0, "optimizer_steps": 0,
        "test_data_used": False, "target_locked": False,
        "monthly": monthly,
    }
    single_terminal = {"status": "COMPLETE", "exit_code": 0,
                       "result": audit.m5.PREVIOUS_REFS["result"],
                       "row_masks_and_ratios": audit.m5.PREVIOUS_REFS["row_masks"],
                       "test_data_used": False}
    selected_plan = {
        "stage": "market_state_selected_activity_masks_20261107_001",
        "test_data_used": False,
        "inputs": {
            "full_plan": audit.FULL_REFS["plan"],
            "full_masks": audit.FULL_REFS["row_masks"],
            "single_plan": audit.m5.PREVIOUS_REFS["plan"],
            "single_masks": audit.m5.PREVIOUS_REFS["row_masks"],
        },
    }
    selected_result = {
        "status": "COMPLETE",
        "decision": "PASS_SELECTED_ACTIVITY_ROW_MASK_SOURCE_ONLY",
        "plan": audit.SELECTED_REFS["plan"],
        "row_masks": audit.SELECTED_REFS["row_masks"],
        "activity_semantics_selected": True,
        "five_tf_target_locked": False,
        "model_forwards": 0, "optimizer_steps": 0,
        "test_data_used": False,
    }
    selected_terminal = {
        "status": "COMPLETE",
        "decision": selected_result["decision"],
        "exit_code": 0,
        "result": audit.SELECTED_REFS["result"],
        "row_masks": audit.SELECTED_REFS["row_masks"],
        "test_data_used": False,
    }
    docs = {
        audit.FULL_REFS["plan"]["path"]: full_plan,
        audit.FULL_REFS["result"]["path"]: full_result,
        audit.FULL_REFS["terminal"]["path"]: full_terminal,
        audit.m5.PREVIOUS_REFS["plan"]["path"]: single_plan,
        audit.m5.PREVIOUS_REFS["result"]["path"]: single_result,
        audit.m5.PREVIOUS_REFS["terminal"]["path"]: single_terminal,
        audit.SELECTED_REFS["plan"]["path"]: selected_plan,
        audit.SELECTED_REFS["result"]["path"]: selected_result,
        audit.SELECTED_REFS["terminal"]["path"]: selected_terminal,
    }
    monkeypatch.setattr(audit.base, "read_json", lambda ref: docs[ref["path"]])
    return docs


def test_small_receipts_pass_without_npz_or_raw_and_reject_bad_lineage(monkeypatch):
    docs = synthetic_receipts(monkeypatch)
    assert len(audit.small_receipts()) == 4
    docs[audit.SELECTED_REFS["plan"]["path"]]["inputs"]["full_masks"] = {
        "path": "/wrong", "sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="SELECTED_RECEIPTS"):
        audit.small_receipts()
    docs[audit.SELECTED_REFS["plan"]["path"]]["inputs"]["full_masks"] = audit.FULL_REFS["row_masks"]
    docs[audit.SELECTED_REFS["terminal"]["path"]]["result"] = {
        "path": "/wrong-result", "sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="SELECTED_RECEIPTS"):
        audit.small_receipts()


def test_preflight_read_only_success_and_existing_output_failure(monkeypatch, tmp_path):
    plan = {"output_dir": str(tmp_path / "new_root")}
    monkeypatch.setattr(audit, "inspect_plan", lambda _: (
        plan, {}, {name: (1, 2) for name in audit.SOURCE_PATHS},
        ("a" * 40, "work/gx1-current", "")))
    report = audit.preflight_read_only({"path": "/plan", "sha256": "b" * 64})
    assert report["status"] == "READY_NO_CLAIM"
    assert report["prior_small_receipts_checked"] == 9
    assert report["claim_created"] is False
    assert report["native_parts_or_raw_payload_read"] is False
    (tmp_path / "new_root").mkdir()
    with pytest.raises(RuntimeError, match="PREFLIGHT_OUTPUT_EXISTS"):
        audit.preflight_read_only({"path": "/plan", "sha256": "b" * 64})


def test_claim_only_root_rejects_extra_file_and_wrong_head(tmp_path):
    out = tmp_path / "claim_root"
    out.mkdir()
    plan = {"output_dir": str(out),
            "source_owners": {"worker": {"path": "/worker", "sha256": "a" * 64}}}
    plan_ref = {"path": "/plan", "sha256": "b" * 64}
    claim = {
        "schema_version": audit.CLAIM_SCHEMA, "stage": audit.STAGE,
        "plan": plan_ref,
        "policy": {"path": str(audit.REPO / "NEXT_RUN_POLICY.json"),
                   "sha256": "c" * 64},
        "source_head": "d" * 40,
        "worker": plan["source_owners"]["worker"],
        "source_only_audit": True, "test_data_used": False,
    }
    (out / "CLAIM.json").write_text(json.dumps(claim))
    assert audit.read_claim(plan, plan_ref, ("d" * 40, "work/gx1-current", ""))[0] == claim
    with pytest.raises(RuntimeError, match="CLAIM"):
        audit.read_claim(plan, plan_ref, ("e" * 40, "work/gx1-current", ""))
    (out / "RESULT.json").write_text("{}")
    with pytest.raises(RuntimeError, match="CLAIM_ONLY"):
        audit.read_claim(plan, plan_ref, ("d" * 40, "work/gx1-current", ""))


def test_selected_mask_axes_and_singleton_parity_before_raw_source(monkeypatch):
    issue = np.arange(5, dtype=np.int64) * audit.base.M5_NS
    parent = np.arange(5, dtype=np.int64)
    leads = np.asarray(audit.base.LEADS, np.int16)
    endpoint = np.arange(25, dtype=np.int64).reshape(5, 5)
    full = np.array([[2, 1, 1, 1, 1],
                     [7, 4, 4, 4, 4],
                     [5, 7, 7, 7, 7],
                     [1, 2, 2, 2, 2],
                     [4, 5, 5, 5, 5]], np.uint8)
    selected = full.copy()
    selected[2, 1:] = 5
    single = full.copy()
    single[2, 1:] = 2
    common = {"issue_ns": issue, "m5_source_row": parent,
              "lead_minutes": leads, "m1_source_row": endpoint}
    def pack(**data):
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **data)
        return buffer.getvalue()
    saved = {
        "full": pack(**common, allowed_set=full),
        "single": pack(**common, allowed_set=single,
                       complete_range_atr=np.zeros((5, 5)),
                       complete_range_atr_valid=np.ones((5, 5), bool),
                       atr_to_quoted_spread=np.ones((5, 5)),
                       atr_to_quoted_spread_valid=np.ones((5, 5), bool)),
        "selected": pack(**common, allowed_set=selected,
                         activity_observed=selected != 7,
                         activity_active=np.isin(selected, (1, 4, 5)),
                         side_observed=np.isin(selected, (1, 4)),
                         side_up=selected == 4,
                         singleton_observed=np.isin(selected, (1, 2, 4))),
    }
    monkeypatch.setattr(audit, "SCOPE", {"issue_rows": 5})
    monkeypatch.setattr(audit, "TF_SPECS",
                        tuple((tf, col, window, 2)
                              for tf, col, window, _ in audit.TF_SPECS))
    monkeypatch.setattr(audit, "INPUTS", {
        name: {"row_masks": {"path": name, "sha256": "0" * 64}}
        for name in saved})
    monkeypatch.setattr(audit.base, "checked",
                        lambda ref, **_: (saved[ref["path"]], (1, 2, 3)))
    assert np.array_equal(audit.load_masks()["allowed_set"], selected)
    broken = selected.copy()
    broken[1, 1] = 1
    saved["selected"] = pack(**common, allowed_set=broken,
                             activity_observed=broken != 7,
                             activity_active=np.isin(broken, (1, 4, 5)),
                             side_observed=np.isin(broken, (1, 4)),
                             side_up=broken == 4,
                             singleton_observed=np.isin(broken, (1, 2, 4)))
    with pytest.raises(RuntimeError, match="MASK_SIDES_M15"):
        audit.load_masks()
