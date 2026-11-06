"""One-use TRAIN source verifier for the proposed five-TF endpoint state.

This publishes row masks, not a locked target or a model result. The old 1.0
ATR tape was ephemeral, so it is rebuilt from the same bound M1/ATR source.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path("/home/andre2/src/GX1_CURRENT")
STAGE = "market_state_endpoint_candidate_rowwise_20261106_001"
SEMANTIC_SHA = "bcd9a82ff6e3218cf58539ec0c4d88cdbb80b1e7c497d2991d31aab071033bf7"
CLOSURE_SHA = "bbd2811dbcd04cdf5d3ac01e4f79c30b405cd085c97ce12093648af699203603"
LEADS = (5, 15, 60, 240, 1440)
MONTH_EDGES = (1714521600000000000, 1717200000000000000, 1719792000000000000,
               1722470400000000000, 1725148800000000000, 1727740800000000000)
CUTOFF = MONTH_EDGES[-1]
PARENTS = 29849
BUDGET = {"memory_max": 4294967296, "swap": 536870912, "pids": 64,
          "numerical_threads": 1, "max_wall_seconds": 1800, "max_invocations": 1,
          "max_json_bytes": 16777216}
DOWN, FLAT, UP, ACTIVE, ALL = 1, 2, 4, 5, 7  # allowed-class bits


def need(ok, key):
    if not ok:
        raise RuntimeError("ENDPOINT_CANDIDATE_" + key)


def exact_positions(axis, issues, lead, cutoff, np):
    """An observed, mature M1 validity at the exact wall-UTC lead only."""
    wanted = issues + lead * 60_000_000_000
    pos = np.searchsorted(axis, wanted)
    safe = np.minimum(pos, len(axis) - 1)
    present = (wanted < cutoff) & (pos < len(axis)) & (axis[safe] == wanted)
    return np.where(present, pos, -1), present


def classify_rows(a05, a10, early, rows, col, target, np):
    """Return allowed sets; no early post-gap side promotion."""
    blocked = target.ATR_UNAVAILABLE | target.LOCAL_SPAN_INCOMPLETE | target.UNKNOWN_SOURCE_GAP
    theta05, theta10 = a05["threshold"][rows, col], a10["threshold"][rows, col]
    range05, range10 = a05["local_range"][rows, col], a10["local_range"][rows, col]
    complete05 = ((a05["reason"][rows, col] & blocked) == 0) & (theta05 > 0)
    complete10 = ((a10["reason"][rows, col] & blocked) == 0) & (theta10 > 0)
    need(np.array_equal(complete05, complete10)
         and np.array_equal(range05.view(np.uint64), range10.view(np.uint64))
         and np.array_equal(theta10.view(np.uint64), (2 * theta05).view(np.uint64)),
         "FACTOR_GEOMETRY")
    flat = complete05 & (range05 < theta05)
    active = complete10 & (range10 >= theta10)
    middle = complete10 & (range10 >= theta05) & (range10 < theta10)
    state = a10["state"][rows, col]
    observed = a10["observed"][rows, col]
    side_down = active & observed & (state == target.DOWN)
    side_up = active & observed & (state == target.UP)
    partial_complete = active & ~side_down & ~side_up
    partial_early = early[rows]
    need(not np.any(flat & active) and not np.any(flat & partial_early)
         and not np.any(active & partial_early)
         and not np.any(middle & (flat | active | partial_early))
         and not np.any(partial_early & complete10)
         and not np.any(active & observed & (state == target.FLAT)), "MASK_DISJOINT")
    allowed = np.full(len(rows), ALL, np.uint8)
    allowed[flat] = FLAT
    allowed[side_down] = DOWN
    allowed[side_up] = UP
    allowed[partial_complete | partial_early] = ACTIVE
    counts = {"flat": int(flat.sum()), "down": int(side_down.sum()),
              "up": int(side_up.sum()), "partial_complete": int(partial_complete.sum()),
              "partial_early": int(partial_early.sum()), "middle_unknown": int(middle.sum())}
    need(sum(counts[k] for k in ("flat", "down", "up", "partial_complete",
                                  "partial_early", "middle_unknown")) <= len(rows), "PARTITION")
    return allowed, counts


def expected_counts(semantic, closure):
    f05, f10 = semantic["factors"]["0.5"], semantic["factors"]["1.0"]
    return {
        "flat": f05["flat"],
        "down_up": (f10["saved_threeclass_counts"]["DOWN"]
                    + f10["saved_threeclass_counts"]["UP"]
                    - f10["threeclass_without_complete_activity"]),
        "partial_complete": f10["partial_up_or_down"],
        "partial_early": closure["early_active_proof_exact"],
        "middle_unknown": f10["flat"] - f05["flat"],
    }


def verify_partition(counts, expected):
    """Fail closed on every month/TF, not only the five rolled-up totals."""
    need(counts["flat"] == expected["flat"]
         and counts["down"] + counts["up"] == expected["down_up"]
         and all(counts[k] == expected[k] for k in
                 ("partial_complete", "partial_early", "middle_unknown")),
         "SAVED_25_PARTITION_DIVERGENCE")


def publish(root, name, value, old):
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    need(len(raw) <= BUDGET["max_json_bytes"], "JSON_BUDGET")
    _publish_bytes(root / name, raw)
    (root / name).chmod(0o444)
    return old.ref(root / name)


def verify(plan, old, target, closure_worker, np):
    import pandas as pd
    import pyarrow.parquet as pq

    inputs = plan["inputs"]
    sp, cp = old.read(inputs["semantic_plan"]), old.read(inputs["closure_plan"])
    sem, clo = old.read(inputs["semantic_result"]), old.read(inputs["closure_result"])
    need(inputs["semantic_result"]["sha256"] == SEMANTIC_SHA
         and inputs["closure_result"]["sha256"] == CLOSURE_SHA
         and sem["plan"] == inputs["semantic_plan"]
         and clo["plan"] == inputs["closure_plan"]
         and not sem["test_data_used"] and not clo["test_data_used"]
         and sem["model_calls"] == clo["model_calls"] == 0
         and not sem["target_factor_selected"] and not clo["target_selected"], "IMMUTABLE_PRIORS")
    need(tuple(sp["lead_minutes"]) == tuple(cp["lead_minutes"]) == LEADS
         and tuple(sp["month_edges_ns"]) == tuple(cp["month_edges_ns"]) == MONTH_EDGES
         and sp["report_parents"] == cp["report_parents"] == PARENTS
         and sp["report_cutoff_ns"] == cp["report_cutoff_ns"] == CUTOFF
         and sem["report_parents"] == clo["report_parents"] == PARENTS
         and clo["report_cutoff_ns"] == CUTOFF
         and tuple(clo["lead_minutes"]) == LEADS
         and tuple(clo["month_edges_ns"]) == MONTH_EDGES
         and set(sem["monthly"]) == set(clo["monthly"]) == set(map(str, MONTH_EDGES[:-1])),
         "REPORT_SCOPE")
    for key in ("prior_result", "prior_series", "m1", "closure_authority", "exact_schedule"):
        need(sp["inputs"][key] == cp["inherited_inputs"][key], "SOURCE_IDENTITY:" + key)
    for item in (*sp["inputs"].values(), *cp["inherited_inputs"].values()):
        need(old.ref(item["path"]) == item, "SOURCE_HASH")
    prior = old.read(sp["inputs"]["prior_result"])
    need(prior["series"] == sp["inputs"]["prior_series"]
         and prior["target_sources"]["m1"] == sp["inputs"]["m1"]
         and not prior["test_data_used"], "PRIOR_SERIES")
    base = target.load_dynamic_mtf_state_series(
        sp["inputs"]["prior_series"], expected_source_refs=prior["target_sources"],
        expected_cutoff_ns=prior["fold"]["series_cutoff_ns"])
    a05, axis = base.arrays, base.arrays["validity_ns"]
    issues = a05["issue_ns"][:PARENTS]
    need(np.array_equal(a05["parent_rows"][:PARENTS], np.arange(PARENTS))
         and int(np.searchsorted(a05["issue_ns"], CUTOFF)) == PARENTS
         and np.all(np.diff(axis) > 0), "CLOCK")
    proof = prior["clock_slice_proof"]
    source = pq.read_table(sp["inputs"]["m1"]["path"],
        columns=["time", "open", "high", "low", "close"],
        filters=[("time", ">=", pd.Timestamp(proof["requested_start_ns"], tz="UTC").to_pydatetime()),
                 ("time", "<", pd.Timestamp(proof["requested_end_exclusive_ns"], tz="UTC").to_pydatetime())],
        use_threads=False).to_pandas()
    need(len(source) == proof["selected_rows"]
         and np.array_equal(pd.DatetimeIndex(source["time"]).as_unit("ns").asi8
                            + target.MINUTE_NS, axis), "M1_CLOCK")
    atr = old.reuse_atr(base, target)
    target.MINIMUM_MOVE_ATR = 1.0
    try:
        atr["definition_sha256"] = target.dynamic_mtf_state_target_contract()["contract_sha256"]
        alt = target.build_dynamic_mtf_state_series(
            source, tf_atr=atr, source_ref=sp["inputs"]["m1"],
            atr_source_ref=prior["target_sources"]["m5_atr"],
            cutoff_ns=base.metadata["cutoff_ns"],
            source_row_offset=base.metadata["source_row_offset"],
            closure_authority=old.read(sp["inputs"]["closure_authority"]),
            exact_schedule=old.read(sp["inputs"]["exact_schedule"]))
    finally:
        target.MINIMUM_MOVE_ATR = 0.5
    a10 = alt.arrays
    need(np.array_equal(a10["validity_ns"], axis)
         and np.array_equal(a10["local_range"], a05["local_range"]), "TAPE_GEOMETRY")
    times = target._clock(source, target.EXIT_DECISION_BAR_SECONDS)
    known, _ = target._known_closures(
        times, sp["inputs"]["m1"], old.read(sp["inputs"]["closure_authority"]),
        old.read(sp["inputs"]["exact_schedule"]))
    unknown_gap = np.zeros(len(axis), bool)
    for row in np.flatnonzero(np.diff(axis) != target.MINUTE_NS) + 1:
        unknown_gap[row] = int(row) not in known
    early_by_tf = {}
    for col, tf in enumerate(target.TIMEFRAMES):
        early, complete = closure_worker.early_active_proof(
            source["high"].to_numpy(np.float64), source["low"].to_numpy(np.float64),
            a10["threshold"][:, col], unknown_gap, target.LOCAL_WINDOW_M1_BARS[tf], np)
        complete_a10 = ((a10["reason"][:, col] & (target.ATR_UNAVAILABLE |
                         target.LOCAL_SPAN_INCOMPLETE | target.UNKNOWN_SOURCE_GAP)) == 0)
        need(not np.any(early & complete) and not np.any(early & complete_a10),
             "INDEPENDENT_POST_GAP")
        early_by_tf[tf] = early
    allowed = np.full((PARENTS, len(LEADS)), ALL, np.uint8)
    source_rows = np.full(allowed.shape, -1, np.int32)
    monthly = {}
    for start, stop in zip(MONTH_EDGES[:-1], MONTH_EDGES[1:]):
        lo, hi = np.searchsorted(issues, [start, stop])
        need(hi > lo, "MONTH")
        monthly[str(start)] = {}
        for col, tf in enumerate(target.TIMEFRAMES):
            pos, present = exact_positions(axis, issues[lo:hi], LEADS[col], CUTOFF, np)
            rows = pos[present]
            mask, counts = classify_rows(a05, a10, early_by_tf[tf], rows, col, target, np)
            allowed[lo:hi, col][present] = mask
            source_rows[lo:hi, col][present] = rows.astype(np.int32)
            expected = expected_counts(sem["monthly"][str(start)][tf],
                                       clo["monthly"][str(start)][tf])
            verify_partition(counts, expected)
            need(len(rows) == sem["monthly"][str(start)][tf]["exact_native_validity"]
                 == clo["monthly"][str(start)][tf]["exact_native_validity"]
                 and hi - lo == sem["monthly"][str(start)][tf]["issues"]
                 == clo["monthly"][str(start)][tf]["issues"], "NATIVE_ENDPOINT_PARITY")
            monthly[str(start)][tf] = {"issues": int(hi - lo),
                "exact_m1_endpoint": int(present.sum()), **counts,
                "exact_singleton": counts["flat"] + counts["down"] + counts["up"],
                "partial_active": counts["partial_complete"] + counts["partial_early"],
                "unknown_all": int((allowed[lo:hi, col] == ALL).sum())}
    need(len(monthly) == 5 and all(len(v) == 5 for v in monthly.values()), "25_PARTITIONS")
    totals = {tf: {k: sum(monthly[str(m)][tf][k] for m in MONTH_EDGES[:-1])
                   for k in ("exact_singleton", "partial_active", "unknown_all")}
              for tf in target.TIMEFRAMES}
    need(tuple(totals[tf]["exact_singleton"] for tf in target.TIMEFRAMES)
         == (12404, 14841, 14896, 14280, 7610)
         and tuple(totals[tf]["partial_active"] for tf in target.TIMEFRAMES)
         == (1878, 468, 95, 97, 906), "AGGREGATE_PARITY")
    return issues, allowed, source_rows, monthly, totals, sp, cp


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True,
                        help="bound TRAIN-only plan JSON in CURRENT/configs/research")
    parser.add_argument("--plan-sha256", required=True, help="exact SHA-256 of plan bytes")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="precreated run directory containing only integrator CLAIM.json")
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    need(Path.cwd() == REPO and args.plan.is_absolute()
         and args.plan.parent == REPO / "configs/research"
         and args.out_dir.is_absolute() and args.out_dir.parent == Path("/home/andre2/GX1_RUNS")
         and args.out_dir.is_dir() and {p.name for p in args.out_dir.iterdir()} == {"CLAIM.json"}
         and len(args.plan_sha256) == 64, "CLI_SCOPE")
    sys.path.insert(0, str(REPO))
    import numpy as np
    from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
    from gx1.scripts import audit_dynamic_mtf_state_target_sensitivity_v1 as old
    from gx1.scripts import audit_dynamic_mtf_state_closure_observability_v1 as closure_worker
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    plan_ref, worker_ref = old.ref(args.plan), old.ref(__file__)
    need(plan_ref["sha256"] == args.plan_sha256, "PLAN_SHA")
    claim_ref = old.ref(args.out_dir / "CLAIM.json")
    claim = old.read(claim_ref)
    need(set(claim) == {"schema_version", "plan", "worker", "policy", "source_head"}
         and claim["schema_version"] == STAGE + "_claim_v1"
         and claim["plan"] == plan_ref and claim["worker"] == worker_ref
         and claim["policy"]["path"] == str(REPO / "NEXT_RUN_POLICY.json"), "CLAIM")
    plan = old.read(plan_ref)
    need(plan["schema_version"] == STAGE + "_plan_v1" and plan["stage"] == STAGE
         and plan["run_root"] == str(args.out_dir) and plan["worker"] == worker_ref
         and plan["budget"] == BUDGET and plan["factors"] == [0.5, 1.0]
         and plan["report_parents"] == PARENTS and plan["report_cutoff_ns"] == CUTOFF
         and plan["lead_minutes"] == list(LEADS)
         and plan["month_edges_ns"] == list(MONTH_EDGES)
         and set(plan["inputs"]) == {"semantic_plan", "semantic_result",
                                      "closure_plan", "closure_result"}
         and set(plan["source_owners"]) == {"target_contract", "sensitivity_helper",
                                            "closure_helper"}, "PLAN")
    need(plan["source_owners"]["target_contract"] == old.ref(target.__file__)
         and plan["source_owners"]["sensitivity_helper"] == old.ref(old.__file__)
         and plan["source_owners"]["closure_helper"] == old.ref(closure_worker.__file__)
         and plan["inputs"]["semantic_result"]["sha256"] == SEMANTIC_SHA
         and plan["inputs"]["closure_result"]["sha256"] == CLOSURE_SHA,
         "SOURCE_OWNERS")
    policy = old.read(claim["policy"])
    gate = policy[STAGE]
    need(policy["training_enabled"] is False and gate["plan"] == plan_ref
         and gate["execution_authorized"] is True
         and gate["data_reads_authorized"] is True
         and gate["target_builds_authorized"] == 1
         and gate["model_calls_authorized"] == 0
         and gate["test_authorized"] is False
         and gate["claimed_invocations"] == gate["completed_invocations"] == 0
         and gate["max_invocations"] == 1, "ADMISSION")
    binding = {"plan": plan_ref, "policy": claim["policy"],
               "worker": worker_ref, "source_head": claim["source_head"]}
    caps = require_capped_cpu_audit_execution()
    caps["lock"] = require_capped_lock_ancestry()
    need(all(caps[k] == BUDGET[k] for k in
             ("memory_max", "swap", "pids", "numerical_threads")), "CAPS")
    snapshots = {item["path"]: old.regular(item["path"]) for item in
                 (*plan["inputs"].values(), *plan["source_owners"].values(),
                  plan_ref, claim["policy"], worker_ref, claim_ref)}
    old.freeze(binding, plan, snapshots)
    started = time.monotonic()

    def deadline(*_):
        raise TimeoutError("ENDPOINT_DEADLINE")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(BUDGET["max_wall_seconds"])
    try:
        issues, allowed, source_rows, monthly, totals, sp, cp = verify(
            plan, old, target, closure_worker, np)
        need(subprocess.check_output(["git", "rev-parse", "HEAD"],
              cwd=REPO, text=True).strip() == claim["source_head"], "HEAD_CHANGED")
        for item in (*plan["inputs"].values(), *plan["source_owners"].values(),
                     *sp["inputs"].values(), *cp["inherited_inputs"].values(),
                     plan_ref, worker_ref, claim_ref, claim["policy"]):
            need(old.ref(item["path"]) == item, "BOOKEND_SOURCE")
        old.freeze(binding, plan, snapshots)
        buf = io.BytesIO()
        np.savez_compressed(buf, issue_ns=issues, lead_minutes=np.asarray(LEADS, np.int16),
                            allowed_set=allowed, source_row=source_rows)
        from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
        _publish_bytes(args.out_dir / "ROW_MASKS.npz", buf.getvalue())
        (args.out_dir / "ROW_MASKS.npz").chmod(0o444)
        rows_ref = old.ref(args.out_dir / "ROW_MASKS.npz")
        publish(args.out_dir, "RESULT.json", {
            "schema_version": STAGE + "_result_v1", "plan": plan_ref,
            "claim": claim_ref, "source_head": claim["source_head"],
            "inputs": plan["inputs"], "row_masks": rows_ref,
            "allowed_bits": {"DOWN": DOWN, "FLAT": FLAT, "UP": UP,
                             "ACTIVE_SIDE_UNKNOWN": ACTIVE, "ALL_UNKNOWN": ALL},
            "monthly": monthly, "totals": totals,
            "elapsed_seconds": time.monotonic() - started,
            "test_data_used": False, "model_calls": 0, "optimizer_steps": 0,
            "target_locked": False, "source_only": True,
            "limits": ["M1 OHLC cannot prove arbitrary intraminute ordering",
                       "Aggregate parity is not full TRAIN calendar qualification",
                       "No loss/scorer/baseline or forecast edge established"]}, old)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
