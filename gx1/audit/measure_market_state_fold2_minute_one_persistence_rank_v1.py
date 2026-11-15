"""One-use read-only causal current-S sign reference for minute-one S.

Scores the same predeclared Fold 2 fit and later development rows on all five
TF. No learned parameter, sign flip, TEST, model fit or trading decision.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from gx1.audit import measure_market_state_fold2_exact_prior_later_v1 as prior
from gx1.audit import measure_market_state_fold2_h1_sign_rank_v1 as sign_owner
from gx1.audit import measure_market_state_fold2_causal_transition_v1 as transition
from gx1.audit import build_market_state_time_query_tape_v1 as tape_owner
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts.market_state_minute_query_candidate_v1 import MinuteQueryTargetTape
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = prior.REPO
POLICY = prior.POLICY
STAGE = "market_state_fold2_minute_one_persistence_rank_20261115_001"
PLAN = REPO / "configs/research/MARKET_STATE_FOLD2_MINUTE_ONE_PERSISTENCE_RANK_20261115_001.json"
ROOT = Path("/home/andre2/GX1_RUNS/MARKET_STATE_FOLD2_MINUTE_ONE_PERSISTENCE_RANK_20261115_001")
TIMEFRAMES = ("M5", "M15", "H1", "H4", "D1")
CODE = (
    "gx1/audit/measure_market_state_fold2_minute_one_persistence_rank_v1.py",
    "gx1/audit/measure_market_state_fold2_exact_prior_later_v1.py",
    "gx1/audit/measure_market_state_fold2_h1_sign_rank_v1.py",
    "gx1/audit/measure_market_state_fold2_causal_transition_v1.py",
    "gx1/audit/measure_market_state_causal_persistence_reference_v1.py",
    "gx1/audit/build_market_state_time_query_tape_v1.py",
    "gx1/contracts/market_state_minute_query_candidate_v1.py",
    "gx1/contracts/market_state_five_endpoint_target_binding_v1.py",
    "gx1/contracts/market_state_time_query_native_F_objective_v1.py",
    "gx1/contracts/immutable_event_authority_v1.py",
    "gx1/contracts/gx1_capped_execution_v1.py",
    "scripts/gx1_capped_run.sh",
)


def need(ok: object, code: str) -> None:
    if not bool(ok):
        raise RuntimeError("FOLD2_MINUTE_ONE_PERSISTENCE_" + code)


def cell(truth: dict, now: np.ndarray, j: int,
         subset: np.ndarray | None = None) -> dict:
    bits = truth["allowed"][:, j]
    mask = truth["eligible"][:, j] & truth["mature"][:, j] & np.isin(bits, (1, 4))
    if subset is not None:
        need(subset.shape == mask.shape and subset.dtype == np.bool_, "SUBSET")
        mask &= subset
    labels = (bits[mask] == 4).astype(np.uint8)
    same = sign_owner.sign(now, j)[mask]
    h1 = sign_owner.sign(now, 2)[mask]
    return {
        "known_future_side_rows": int(mask.sum()),
        "future_up_rows": int(labels.sum()),
        "future_down_rows": int(len(labels) - labels.sum()),
        "current_certified_direction_rows": int(np.count_nonzero(same)),
        "current_flat_rows": int(np.count_nonzero(now[mask, j] == 2)),
        "current_active_side_unknown_rows": int(np.count_nonzero(now[mask, j] == 5)),
        "current_all_unknown_rows": int(np.count_nonzero(now[mask, j] == 7)),
        "current_missing_rows": int(np.count_nonzero(now[mask, j] == 0)),
        "same_tf_current_sign_auc": sign_owner.rank_auc(labels, same),
        "h1_current_sign_control_auc": sign_owner.rank_auc(labels, h1) if j == 0 else None,
        "constant_rank_reference_auc": 0.5
            if 0 < labels.sum() < len(labels) else None,
    }


def preflight(plan_sha: str) -> tuple:
    need(Path.cwd().resolve() == REPO
         and prior.git("branch", "--show-current") == "work/gx1-current"
         and prior.git("status", "--porcelain") == "", "CLEAN_SOURCE")
    head = prior.git("rev-parse", "HEAD")
    need(not ROOT.exists() and prior.sha(PLAN) == plan_sha, "PLAN_ROOT")
    plan = json.loads(PLAN.read_text())
    need(plan["schema_version"] == "gx1_fold2_minute_one_persistence_rank_plan_v1"
         and plan["stage"] == STAGE and plan["status"] == "ADMITTED_READ_ONLY"
         and plan["source_base_head"] == "3a98cbf159066c45daf0f2f24eb9dfc4d0a2762e"
         and plan["output_root"] == str(ROOT)
         and plan["scope"] == {
             "fold_id": 2, "fit_rows": 47338, "later_rows": 4096,
             "timeframes": list(TIMEFRAMES), "query_minute_all_TF": 1,
             "primary_cell": "M5_future_minute1_known_side",
             "predictor": "same_TF_current_bit4_plus1_bit1_minus1_else_zero",
             "control": "current_H1_sign_same_M5_rows",
             "metric": "tie_aware_rank_auc_no_fit",
             "weekly_primary": True, "model_forwards": 0,
             "optimizer_steps": 0, "test_data_used": False}
         and plan["limits"] == {
             "class": "audit", "memory": "4G", "swap": "512M",
             "wall_seconds": 900, "max_invocations": 1}, "PLAN")
    need(subprocess.run(("git", "merge-base", "--is-ancestor",
                         plan["source_base_head"], head), cwd=REPO).returncode == 0,
         "ANCESTOR")
    need(set(plan["source_owners"]) == set(CODE)
         and all(prior.sha(REPO / path) == digest
                 for path, digest in plan["source_owners"].items()), "SOURCE_HASH")
    need(all(Path(ref["path"]).is_file()
             and prior.sha(Path(ref["path"])) == ref["sha256"]
             for ref in plan["input_refs"].values()), "INPUT_HASH")
    policy = json.loads(POLICY.read_text())
    need(policy["training_enabled"] is False
         and policy["native_training_enabled"] is False
         and policy[STAGE] == {
             "status": "ADMITTED_READ_ONLY", "plan": prior.ref(PLAN),
             "output_root": str(ROOT), "max_invocations": 1,
             "claimed_invocations": 0, "completed_invocations": 0,
             "data_reads_authorized": True, "model_forwards_authorized": 0,
             "optimizer_steps_authorized": 0, "training_authorized": False,
             "test_authorized": False, "broker_access_allowed": False,
             "relaunch_allowed": False}, "POLICY")
    return head, prior.ref(POLICY), plan


def measure(head: str, policy_ref: dict, plan: dict) -> dict:
    bound = state.load_bound_five_endpoint_targets()
    coverage = json.loads(Path(plan["input_refs"]["coverage"]["path"]).read_text())
    fit, later_rows, later_issues = prior.axes(coverage, bound)
    fit_rows, fit_issues = fit["fit_parent_rows"], fit["fit_decision_ns"]
    need(len(fit_rows) == 47338 and len(later_rows) == 4096
         and coverage["score_start_ns"] == 1735689600000000000
         and coverage["score_end_exclusive_ns"] == 1743465600000000000,
         "COHORT_AXIS")
    now_fit, reasons_fit = transition._current(bound, fit_issues)
    now_later, reasons_later = transition._current(bound, later_issues)
    need(np.array_equal(now_fit == 0, reasons_fit != 0)
         and np.array_equal(now_later == 0, reasons_later != 0),
         "CAUSAL_CURRENT")
    path = Path(plan["input_refs"]["candidate_tape"]["path"])
    with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as archive:
        need(set(archive.files) == tape_owner.TAPE_KEYS, "TAPE_MEMBERS")
        tape = MinuteQueryTargetTape(
            np.array(archive["validity_ns"], copy=True),
            np.array(archive["m1_source_row"], copy=True),
            np.array(archive["allowed_set"], copy=True),
            np.array(archive["path_observed"], copy=True),
            np.array(archive["gap_kind"], copy=True))
    source = json.loads(Path(plan["input_refs"]["minute_source_review"]["path"]).read_text())
    cohorts = {}
    weekly = {}
    for name, issues, now, cutoff in (
        ("fit", fit_issues, now_fit, coverage["score_start_ns"]),
        ("later", later_issues, now_later, coverage["score_end_exclusive_ns"])):
        query = np.ones((len(issues), 5), np.int64)
        truth = tape.gather_minutes(issues, query, fold_upper_ns=cutoff)
        observed = {}
        for j, tf in enumerate(TIMEFRAMES):
            allowed = truth["allowed"][:, j]
            observed[tf] = {
                "mature": int(truth["mature"][:, j].sum()),
                "eligible": int(truth["eligible"][:, j].sum()),
                "known_side": int(np.count_nonzero(
                    truth["eligible"][:, j] & np.isin(allowed, (1, 4))))}
        need(observed == source["minute1_all_TF"][name], "REUSED_MINUTE_COUNTS")
        cohorts[name] = {
            tf: cell(truth, now, j) for j, tf in enumerate(TIMEFRAMES)}
        if name == "later":
            weeks = objective._weeks(issues)
            weekly = {str(week): cell(truth, now, 0, weeks == week)
                      for week in np.unique(weeks)}
            need(len(weekly) == 14, "WEEK_COUNT")
    need(prior.sha(path) == plan["input_refs"]["candidate_tape"]["sha256"]
         and prior.git("rev-parse", "HEAD") == head
         and prior.git("status", "--porcelain") == ""
         and prior.ref(POLICY) == policy_ref, "BOOKEND")
    return {
        "schema_version": "gx1_fold2_minute_one_persistence_rank_result_v1",
        "status": "COMPLETE", "stage": STAGE, "source_head": head,
        "plan": prior.ref(PLAN), "policy": policy_ref,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "fit_rows": len(fit_issues), "later_rows": len(later_issues),
        "scores": cohorts, "later_M5_by_utc_week": weekly,
        "model_forwards": 0, "optimizer_steps": 0, "model_fit": False,
        "test_data_used": False, "trade_selection": False,
    }


def run(plan_sha: str) -> None:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    head, policy_ref, plan = preflight(plan_sha)
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap["memory_max"] == cap["memory_high"] == 4 * 1024**3
         and cap["swap"] == 512 * 1024**2 and lock is not None, "CAP")
    ROOT.mkdir(parents=False, exist_ok=False)
    _publish_bytes(ROOT / "CLAIM.json", prior.encode({
        "schema_version": "gx1_fold2_minute_one_persistence_rank_claim_v1",
        "stage": STAGE, "plan": prior.ref(PLAN),
        "policy": policy_ref, "source_head": head, "test_data_used": False}))
    claim = prior.ref(ROOT / "CLAIM.json")
    started = time.monotonic()
    try:
        result = measure(head, policy_ref, plan)
        result["claim"] = claim
        result["elapsed_seconds"] = time.monotonic() - started
        _publish_bytes(ROOT / "RESULT.json", prior.encode(result))
        terminal = {
            "schema_version": "gx1_fold2_minute_one_persistence_rank_terminal_v1",
            "status": "COMPLETE", "exit_code": 0, "stage": STAGE,
            "claim": claim, "result": prior.ref(ROOT / "RESULT.json"),
            "source_head": head, "test_data_used": False}
        _publish_bytes(ROOT / "TERMINAL.json", prior.encode(terminal))
    except BaseException as exc:
        terminal = {
            "schema_version": "gx1_fold2_minute_one_persistence_rank_terminal_v1",
            "status": "FAILED_CONSUMED", "exit_code": 1, "stage": STAGE,
            "claim": claim, "result": None, "source_head": head,
            "reason": type(exc).__name__ + ":" + str(exc),
            "test_data_used": False}
        _publish_bytes(ROOT / "TERMINAL.json", prior.encode(terminal))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-sha", required=True)
    run(parser.parse_args().plan_sha)


if __name__ == "__main__":
    main()
