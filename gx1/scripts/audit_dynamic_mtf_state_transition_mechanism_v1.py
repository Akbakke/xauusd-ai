"""One-use TRAIN-only explanation of FLAT/activity threshold transitions.

Reads saved 0.5 target arrays. The 1.0 activity boundary is exactly twice
the saved threshold; no target rebuilding, raw quotes, model, or factor choice.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

REPO = Path("/home/andre2/src/GX1_CURRENT")
STAGE = "dynamic_mtf_state_transition_mechanism_20261106_001"
PLAN_PATH = REPO / "configs/research/DYNAMIC_MTF_STATE_TRANSITION_MECHANISM_20261106_001.json"
ROOT = Path("/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_TRANSITION_MECHANISM_20261106_001")
BUDGET = {"memory_max": 4294967296, "swap": 536870912, "pids": 64,
          "numerical_threads": 1, "max_wall_seconds": 1800,
          "max_invocations": 1, "max_json_bytes": 8388608}
MINUTE_NS = 60000000000
TFS = ("M5", "M15", "H1", "H4", "D1")
FACTORS = ("0.5", "1.0")
BLOCKED = 1 | 2 | 4


def need(ok, name):
    if not ok:
        raise RuntimeError("STATE_TRANSITION_MECHANISM_" + name)


def quantiles(values, np):
    values = np.asarray(values, dtype=np.float64)
    need(np.isfinite(values).all(), "NONFINITE_DISTANCE")
    if values.size == 0:
        return {"count": 0, "p10": None, "median": None, "p90": None}
    p10, p50, p90 = np.percentile(values, (10, 50, 90))
    return {"count": int(values.size), "p10": float(p10),
            "median": float(p50), "p90": float(p90)}


def analyze(axis, threshold, local_range, reason, atr_clock, np):
    """Classify each consecutive complete source-minute pair once."""
    n = len(axis)
    need(n >= 2 and all(len(x) == n for x in
         (threshold, local_range, reason, atr_clock)), "SHAPE")
    need(np.all(axis[1:] > axis[:-1])
         and np.isfinite(threshold).all() and np.isfinite(local_range).all(),
         "DOMAIN")
    need(np.all(threshold >= 0) and np.all(local_range >= 0), "NONNEGATIVE")
    valid = ((reason & BLOCKED) == 0) & (threshold > 0)
    flat = local_range < threshold
    pair = valid[:-1] & valid[1:]
    exact_minute = (axis[1:] - axis[:-1]) == MINUTE_NS
    eligible = pair & exact_minute
    flips = flat[:-1] != flat[1:]
    strict_flip = eligible & flips
    gap_flip = pair & ~exact_minute & flips
    current_range_previous_threshold = local_range[1:] < threshold[:-1]
    range_alone_flip = current_range_previous_threshold != flat[:-1]
    threshold_required = strict_flip & ~range_alone_flip
    range_sufficient = strict_flip & range_alone_flip
    update_suppressed = eligible & ~flips & range_alone_flip
    atr_clock_changed = atr_clock[:-1] != atr_clock[1:]
    threshold_changed = threshold[:-1] != threshold[1:]
    need(np.all(~threshold_required | threshold_changed)
         and int(threshold_required.sum() + range_sufficient.sum())
             == int(strict_flip.sum()), "COUNTERFACTUAL_PARTITION")
    actual_margin = np.abs(local_range[1:] / np.where(threshold[1:] > 0,
                                                     threshold[1:], 1.0) - 1.0)
    prior_margin = np.abs(local_range[:-1] / np.where(threshold[:-1] > 0,
                                                    threshold[:-1], 1.0) - 1.0)
    groups = {
        "atr_clock_and_value_unchanged": ~atr_clock_changed & ~threshold_changed,
        "atr_clock_changed_value_unchanged": atr_clock_changed & ~threshold_changed,
        "atr_clock_changed_value_changed": atr_clock_changed & threshold_changed,
        "atr_clock_unchanged_value_changed": ~atr_clock_changed & threshold_changed,
    }
    by_update = {name: int((strict_flip & mask).sum()) for name, mask in groups.items()}
    need(sum(by_update.values()) == int(strict_flip.sum()), "UPDATE_PARTITION")
    return {
        "source_rows": int(n),
        "complete_activity_rows": int(valid.sum()),
        "row_adjacent_pairs": int(n - 1),
        "nonconsecutive_time_pairs": int((~exact_minute).sum()),
        "incomplete_or_unavailable_pairs": int((~pair).sum()),
        "complete_cross_gap_pairs": int((pair & ~exact_minute).sum()),
        "eligible_consecutive_complete_pairs": int(eligible.sum()),
        "strict_consecutive_flips": int(strict_flip.sum()),
        "cross_gap_row_adjacent_flips": int(gap_flip.sum()),
        "row_adjacent_flips_including_gaps": int((pair & flips).sum()),
        "strict_flips_by_atr_update": by_update,
        "threshold_update_required_flips": int(threshold_required.sum()),
        "range_alone_sufficient_flips": int(range_sufficient.sum()),
        "range_flip_suppressed_by_threshold_update": int(update_suppressed.sum()),
        "absolute_current_range_over_threshold_minus_one_at_strict_flips":
            quantiles(actual_margin[strict_flip], np),
        "absolute_previous_range_over_threshold_minus_one_at_strict_flips":
            quantiles(prior_margin[strict_flip], np),
        "month_edges_censor_cross_month_pairs": True,
        "exact_consecutive_clock_required": True,
    }


def self_test(np):
    axis = np.array([0, 1, 2, 4, 5], dtype=np.int64) * MINUTE_NS
    theta = np.array([1.0, 1.0, 2.0, 2.0, 2.0])
    local = np.array([0.9, 1.1, 1.1, 3.0, 1.1])
    reason = np.zeros(5, dtype=np.uint8)
    atr = np.array([10, 10, 20, 20, 20], dtype=np.int64)
    x = analyze(axis, theta, local, reason, atr, np)
    need(x["strict_consecutive_flips"] == 3
         and x["cross_gap_row_adjacent_flips"] == 1
         and x["range_alone_sufficient_flips"] == 2
         and x["threshold_update_required_flips"] == 1
         and x["strict_flips_by_atr_update"]["atr_clock_and_value_unchanged"] == 2
         and x["strict_flips_by_atr_update"]["atr_clock_changed_value_changed"] == 1,
         "SYNTHETIC_SUCCESS")
    bad = False
    try:
        analyze(axis[::-1], theta, local, reason, atr, np)
    except RuntimeError as exc:
        bad = str(exc) == "STATE_TRANSITION_MECHANISM_DOMAIN"
    need(bad, "SYNTHETIC_ERROR")
    suppressed = analyze(np.array([0, MINUTE_NS], dtype=np.int64),
        np.array([2.0, 3.0]), np.array([1.9, 2.1]),
        np.zeros(2, dtype=np.uint8), np.array([10, 20], dtype=np.int64), np)
    need(suppressed["strict_consecutive_flips"] == 0
         and suppressed["range_flip_suppressed_by_threshold_update"] == 1,
         "SYNTHETIC_SUPPRESSED")
    censored = analyze(axis, theta, local,
        np.array([0, 1, 0, 0, 0], dtype=np.uint8), atr, np)
    need(censored["incomplete_or_unavailable_pairs"] == 2
         and censored["strict_consecutive_flips"] == 1,
         "SYNTHETIC_CENSORED")
    for status in ("COMPLETE", "FAILED_CONSUMED"):
        json.dumps({"status": status, "exit_code": 0 if status == "COMPLETE" else 1},
                   allow_nan=False)


def publish(name, value, old):
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    need(len(payload) <= BUDGET["max_json_bytes"], "JSON_BUDGET")
    _publish_bytes(ROOT / name, payload)
    (ROOT / name).chmod(0o444)
    return old.ref(ROOT / name)


def preflight_metadata(plan, old, target):
    prior = old.read(plan["inputs"]["prior_result"])
    semantic = old.read(plan["inputs"]["semantic_result"])
    terminal = old.read(plan["inputs"]["semantic_terminal"])
    manifest = old.read(plan["inputs"]["prior_series"])
    need(prior["series"] == plan["inputs"]["prior_series"]
         and prior["target_contract_sha256"] == plan["target_contract_sha256"]
         and prior["test_data_used"] is False
         and prior["fold"]["series_cutoff_ns"] > plan["report_cutoff_ns"],
         "PRIOR_BINDING")
    need(terminal["status"] == "COMPLETE" and terminal["exit_code"] == 0
         and terminal["result"] == plan["inputs"]["semantic_result"]
         and terminal["test_data_used"] is False, "SEMANTIC_TERMINAL")
    need(semantic["report_parents"] == 29849
         and semantic["report_cutoff_ns"] == plan["report_cutoff_ns"]
         and set(semantic["monthly"]) == set(str(x) for x in plan["month_edges_ns"][:-1])
         and semantic["target_factor_selected"] is False
         and semantic["model_calls"] == semantic["optimizer_steps"] == 0
         and semantic["test_data_used"] is False
         and semantic["later_aggregates_computed"] is False,
         "SEMANTIC_RESULT")
    need(manifest["metadata"]["test_data_used"] is False
         and manifest["metadata"]["definition"]["contract_sha256"] ==
             plan["target_contract_sha256"]
         and all(name in manifest["arrays"] for name in
                 ("validity_ns", "threshold", "local_range", "reason",
                  "atr_available_ns")), "SERIES_MANIFEST")
    for month in plan["month_edges_ns"][:-1]:
        for tf in TFS:
            for factor in FACTORS:
                item = semantic["monthly"][str(month)][tf]["factors"][factor]
                need("binary_runs_unique_source" in item
                     and "eligible_unique_m1_minutes" in item["binary_runs_unique_source"]
                     and "transitions" in item["binary_runs_unique_source"],
                     "SEMANTIC_KEYS")
    need(target.MINIMUM_MOVE_ATR == 0.5
         and target.dynamic_mtf_state_target_contract()["contract_sha256"] ==
             plan["target_contract_sha256"], "TARGET_SOURCE")
    return prior, semantic, manifest


def audit(plan, binding, claim, snapshots, old, target, np, prior, semantic, manifest):
    series = target.load_dynamic_mtf_state_series(
        plan["inputs"]["prior_series"],
        expected_source_refs=prior["target_sources"],
        expected_cutoff_ns=prior["fold"]["series_cutoff_ns"])
    a = series.arrays
    axis = a["validity_ns"]
    need(len(axis) == manifest["metadata"]["rows"]
         and np.all(axis[1:] > axis[:-1])
         and plan["month_edges_ns"][0] >= axis[0]
         and plan["report_cutoff_ns"] <= axis[-1], "SERIES_CLOCK")
    report = {}
    for start, stop in zip(plan["month_edges_ns"][:-1], plan["month_edges_ns"][1:]):
        lo, hi = np.searchsorted(axis, [start, stop])
        need(hi - lo >= 2 and stop <= plan["report_cutoff_ns"], "TRAIN_MONTH")
        per_tf = {}
        for col, tf in enumerate(TFS):
            values = {}
            for factor in FACTORS:
                mult = 1.0 if factor == "0.5" else 2.0
                entry = analyze(axis[lo:hi], a["threshold"][lo:hi, col] * mult,
                    a["local_range"][lo:hi, col], a["reason"][lo:hi, col],
                    a["atr_available_ns"][lo:hi, col], np)
                old_runs = semantic["monthly"][str(start)][tf]["factors"][factor][
                    "binary_runs_unique_source"]
                need(entry["complete_activity_rows"] ==
                     old_runs["eligible_unique_m1_minutes"]
                     and entry["row_adjacent_flips_including_gaps"] ==
                     sum(old_runs["transitions"].values())
                     and entry["row_adjacent_flips_including_gaps"] ==
                     entry["strict_consecutive_flips"] +
                     entry["cross_gap_row_adjacent_flips"],
                     "SEMANTIC_ROW_PARITY")
                values[factor] = entry
            per_tf[tf] = values
        report[str(start)] = per_tf
    old.freeze(binding, plan, snapshots)
    for item in plan["inputs"].values():
        need(old.ref(item["path"]) == item, "INPUT_CHANGED")
    return {
        "schema_version": STAGE + "_result_v1",
        "plan": binding["plan"], "policy": binding["policy"],
        "claim": claim, "source_head": binding["source_head"],
        "month_edges_ns": plan["month_edges_ns"],
        "report_cutoff_ns": plan["report_cutoff_ns"],
        "factors": list(FACTORS), "timeframes": list(TFS),
        "reports": report,
        "series_reused": True, "alternative_target_builds": 0,
        "raw_m1_reads": 0, "model_calls": 0, "optimizer_steps": 0,
        "test_data_used": False, "later_aggregates_computed": False,
        "factor_selected": False,
        "limits": [
            "This partitions binary FLAT/non-FLAT switches; it does not prove UP/DOWN target quality.",
            "The counterfactual holds the previous ATR threshold only for one current observation.",
            "Known closure gaps are excluded from strict consecutive-minute switches.",
            "Threshold update attribution is descriptive, not causal market attribution.",
            "No hysteresis parameter or state candidate is selected.",
        ],
    }


def main():
    need(len(sys.argv) == 1 and Path.cwd() == REPO, "CLI")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    from gx1.scripts import audit_dynamic_mtf_state_target_sensitivity_v1 as old
    from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    self_test(np)
    need(ROOT.is_dir() and not any(p.is_symlink() for p in (ROOT, *ROOT.parents)),
         "RUN_ROOT")
    binding = old.read(old.ref(ROOT / "EXECUTION_BINDING.json"))
    need(set(binding) == {"plan", "policy", "worker", "source_head"}
         and binding["plan"]["path"] == str(PLAN_PATH)
         and binding["policy"]["path"] == str(REPO / "NEXT_RUN_POLICY.json")
         and binding["worker"] == old.ref(__file__), "BINDING")
    plan = old.read(binding["plan"])
    need(plan["stage"] == STAGE and plan["schema_version"] == STAGE + "_plan_v1"
         and plan["run_root"] == str(ROOT) and plan["worker"] == binding["worker"]
         and plan["budget"] == BUDGET and plan["factors"] == [0.5, 1.0]
         and plan["timeframes"] == list(TFS)
         and plan["month_edges_ns"][-1] == plan["report_cutoff_ns"] ==
             1727740800000000000
         and plan["month_edges_ns"][0] == 1714521600000000000
         and len(plan["month_edges_ns"]) == 6
         and all(x < y for x, y in zip(plan["month_edges_ns"],
                                       plan["month_edges_ns"][1:]))
         and set(plan["inputs"]) ==
             {"prior_result", "prior_series", "semantic_result", "semantic_terminal"}
         and all("TEST" not in Path(x["path"]).parts for x in plan["inputs"].values()),
         "PLAN")
    policy = old.read(binding["policy"])
    gate = policy[STAGE]
    need(policy["training_enabled"] is False and gate["plan"] == binding["plan"]
         and gate["execution_authorized"] is True
         and gate["data_reads_authorized"] is True
         and gate["target_builds_authorized"] == 0
         and gate["model_calls_authorized"] == 0
         and gate["claimed_invocations"] == gate["completed_invocations"] == 0
         and gate["max_invocations"] == 1
         and gate["test_authorized"] is False
         and gate["later_aggregate_authorized"] is False, "ADMISSION")
    need(all(not (ROOT / name).exists() and not (ROOT / name).is_symlink()
             for name in ("CLAIM.json", "RESULT.json", "TERMINAL.json")),
         "CONSUMED")
    limits = require_capped_cpu_audit_execution()
    limits["lock"] = require_capped_lock_ancestry()
    need(all(limits[k] == BUDGET[k] for k in
             ("memory_max", "swap", "pids", "numerical_threads")), "CAPS")
    snapshots = {x["path"]: old.regular(x["path"]) for x in
                 (*plan["inputs"].values(), *plan["source_owners"].values(),
                  binding["plan"], binding["worker"], binding["policy"])}
    prior, semantic, manifest = preflight_metadata(plan, old, target)
    for x in manifest["arrays"].values():
        snapshots[x["path"]] = old.regular(x["path"])
    old.freeze(binding, plan, snapshots)
    claim = publish("CLAIM.json", {
        "schema_version": STAGE + "_claim_v1", "plan": binding["plan"],
        "policy": binding["policy"], "worker": binding["worker"],
        "source_head": binding["source_head"], "limits": limits,
        "target_builds": 0, "raw_m1_reads": 0, "model_calls": 0}, old)
    started = time.monotonic()

    def deadline(*_):
        raise TimeoutError("STATE_TRANSITION_MECHANISM_1800_SECOND_DEADLINE")

    signal.signal(signal.SIGALRM, deadline)
    signal.signal(signal.SIGTERM, deadline)
    signal.alarm(BUDGET["max_wall_seconds"])
    try:
        result = audit(plan, binding, claim, snapshots, old, target, np,
                       prior, semantic, manifest)
        result["elapsed_seconds"] = time.monotonic() - started
        result_ref = publish("RESULT.json", result, old)
        publish("TERMINAL.json", {
            "schema_version": STAGE + "_terminal_v1", "status": "COMPLETE",
            "exit_code": 0, "claim": claim, "result": result_ref,
            "source_head": binding["source_head"], "test_data_used": False}, old)
    except BaseException as exc:
        publish("TERMINAL.json", {
            "schema_version": STAGE + "_terminal_v1", "status": "FAILED_CONSUMED",
            "exit_code": 1, "claim": claim, "source_head": binding["source_head"],
            "error": type(exc).__name__ + ":" + str(exc)[:500],
            "test_data_used": False}, old)
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
