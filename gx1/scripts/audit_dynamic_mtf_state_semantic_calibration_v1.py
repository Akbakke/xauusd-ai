"""One-use TRAIN-only semantic target calibration. No model or factor selection."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

REPO = Path("/home/andre2/src/GX1_CURRENT")
STAGE = "dynamic_mtf_state_semantic_calibration_20261106_001"
PLAN_PATH = REPO / "configs/research/DYNAMIC_MTF_STATE_SEMANTIC_CALIBRATION_20261106_001.json"
ROOT = Path("/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_SEMANTIC_CALIBRATION_20261106_001")
BUDGET = {"memory_max": 4294967296, "swap": 536870912, "pids": 64,
          "numerical_threads": 1, "max_wall_seconds": 1800,
          "max_invocations": 1, "max_json_bytes": 16777216}


def need(ok, name):
    if not ok:
        raise RuntimeError("SEMANTIC_TARGET_" + name)


def publish(name, value, old):
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    need(len(raw) <= BUDGET["max_json_bytes"], "JSON_SIZE")
    _publish_bytes(ROOT / name, raw)
    (ROOT / name).chmod(0o444)
    return old.ref(ROOT / name)


def quantiles(values, np):
    values = np.asarray(values, np.float64)
    need(np.isfinite(values).all(), "FINITE")
    if not len(values):
        return {"count": 0, "p10": None, "median": None, "p90": None}
    p = np.percentile(values, [10, 50, 90])
    return {"count": len(values), "p10": float(p[0]), "median": float(p[1]), "p90": float(p[2])}


def activity(tape, rows, col, target, np):
    a = tape.arrays
    theta, span, reason = (a[name][rows, col] for name in ("threshold", "local_range", "reason"))
    blocked = target.ATR_UNAVAILABLE | target.LOCAL_SPAN_INCOMPLETE | target.UNKNOWN_SOURCE_GAP
    valid = ((reason & blocked) == 0) & np.isfinite(theta) & (theta > 0) & np.isfinite(span)
    flat = valid & (span < theta)
    known, state = a["observed"][rows, col], a["state"][rows, col]
    need(np.all(~flat | (known & (state == target.FLAT)))
         and np.all(~(valid & known) | ((state == target.FLAT) == flat)),
         "ACTIVITY_CLASS")
    return valid, flat, valid & ~flat, known, state


def run_lengths(labels, valid, axis, names, np):
    code = np.where(valid, labels, -1)
    first = np.flatnonzero((code >= 0) & np.r_[True, code[1:] != code[:-1]])
    last = np.flatnonzero((code >= 0) & np.r_[code[:-1] != code[1:], True])
    need(len(first) == len(last) and np.all(first <= last), "RUNS")
    left = (first == 0) | (code[np.maximum(first - 1, 0)] < 0)
    right = (last == len(code) - 1) | (code[np.minimum(last + 1, len(code) - 1)] < 0)
    complete = ~left & ~right
    active_minutes = last - first + 1
    wall_minutes = (axis[last] - axis[first]) // 60000000000 + 1
    by_state = {}
    for value, name in enumerate(names):
        pick = code[first] == value
        by_state[name] = {
            "runs": int(pick.sum()), "left_censored": int((pick & left).sum()),
            "right_censored": int((pick & right).sum()),
            "completed": int((pick & complete).sum()),
            "active_minutes_observed_lower_bound": quantiles(active_minutes[pick], np),
            "active_minutes_completed": quantiles(active_minutes[pick & complete], np),
            "wall_minutes_completed": quantiles(wall_minutes[pick & complete], np)}
    adjacent = (code[:-1] >= 0) & (code[1:] >= 0)
    transitions = {a + ">" + b: int(np.count_nonzero(
        adjacent & (code[:-1] == i) & (code[1:] == j)))
        for i, a in enumerate(names) for j, b in enumerate(names) if i != j}
    return {"eligible_unique_m1_minutes": int(valid.sum()), "by_state": by_state,
            "transitions": transitions,
            "censor_rule": "month boundary or unknown label censors; observed class change completes a run"}


def pick_witness(source, row, window, parent, issue, tf, month, category, col, tapes, np):
    need(row >= window - 1, "WITNESS_WINDOW")
    parts = np.array_split(np.arange(row - window + 1, row + 1), min(12, window))
    candles = []
    for part in parts:
        sl = source.iloc[part]
        candles.append({
            "first_m1_start_utc": str(sl["time"].iloc[0]),
            "last_m1_start_utc": str(sl["time"].iloc[-1]),
            "observed_m1_bars": len(part),
            "open": float(sl["open"].iloc[0]), "high": float(sl["high"].max()),
            "low": float(sl["low"].min()), "close": float(sl["close"].iloc[-1])})
    labels = {}
    for factor, tape in tapes.items():
        a = tape.arrays
        labels[factor] = {name: (int(a[name][row, col]) if name in ("state", "reason")
                                  else bool(a[name][row, col]) if name == "observed"
                                  else float(a[name][row, col]))
                          for name in ("state", "reason", "observed", "threshold", "local_range")}
    return {"month_start_ns": int(month), "timeframe": tf, "category": category,
            "parent_row": int(parent), "issue_ns": int(issue),
            "validity_ns": int(tapes["0.5"].arrays["validity_ns"][row]),
            "quoted_spread_usd_at_validity": float(source["ask_close"].iloc[row] - source["bid_close"].iloc[row]),
            "labels": labels, "trajectory": candles,
            "trajectory_limit": "up to 12 consecutive active-minute OHLC summaries; not intraminute order"}


def audit(plan, binding, claim, snapshots, old):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target

    for item in plan["inputs"].values():
        need(old.ref(item["path"]) == item, "INPUT_HASH")
    previous = old.read(plan["inputs"]["prior_result"])
    saved = old.read(plan["inputs"]["sensitivity_result"])
    need(previous["series"] == plan["inputs"]["prior_series"]
         and previous["target_sources"]["m1"] == plan["inputs"]["m1"]
         and previous["target_contract_sha256"] == plan["existing_contract_sha256"]
         and previous["test_data_used"] is False, "PRIOR")
    need(saved["factors"] == [0.5, 1.0] and saved["report_parents"] == plan["report_parents"]
         and saved["report_cutoff_ns"] == plan["report_cutoff_ns"]
         and saved["month_edges_ns"] == plan["month_edges_ns"]
         and saved["lead_minutes"] == plan["lead_minutes"]
         and saved["model_calls"] == saved["optimizer_steps"] == 0
         and saved["test_data_used"] is False
         and saved["alternative_series_published"] is False, "SENSITIVITY")
    for name in ("calendar_source_reference", "closure_authority", "exact_schedule", "published_calendar"):
        need(previous["calendar_binding"][name] == plan["inputs"][name], "CALENDAR")
    manifest = old.read(plan["inputs"]["prior_series"])
    for item in manifest["arrays"].values():
        snapshots[item["path"]] = old.regular(item["path"])
    base = target.load_dynamic_mtf_state_series(
        plan["inputs"]["prior_series"], expected_source_refs=previous["target_sources"],
        expected_cutoff_ns=previous["fold"]["series_cutoff_ns"])
    a = base.arrays
    count, cutoff = plan["report_parents"], plan["report_cutoff_ns"]
    issues, parents, axis = a["issue_ns"][:count], a["parent_rows"][:count], a["validity_ns"]
    need(np.array_equal(parents, np.arange(count, dtype=np.int64))
         and int(np.searchsorted(a["issue_ns"], cutoff)) == count, "PARENTS")
    proof = previous["clock_slice_proof"]
    source = pq.read_table(plan["inputs"]["m1"]["path"],
        columns=["time", "open", "high", "low", "close", "bid_close", "ask_close"],
        filters=[("time", ">=", pd.Timestamp(proof["requested_start_ns"], tz="UTC").to_pydatetime()),
                 ("time", "<", pd.Timestamp(proof["requested_end_exclusive_ns"], tz="UTC").to_pydatetime())],
        use_threads=False).to_pandas()
    need(np.array_equal(pd.DatetimeIndex(source["time"]).as_unit("ns").asi8 + target.MINUTE_NS, axis),
         "SOURCE_CLOCK")
    bid, ask, close = (source[name].to_numpy() for name in ("bid_close", "ask_close", "close"))
    need(np.isfinite(source[["open", "high", "low", "close", "bid_close", "ask_close"]].to_numpy()).all()
         and np.all(ask > bid) and np.all(close > 0), "QUOTES")
    atr = old.reuse_atr(base, target)
    target.MINIMUM_MOVE_ATR = 1.0
    atr["definition_sha256"] = target.dynamic_mtf_state_target_contract()["contract_sha256"]
    alt = target.build_dynamic_mtf_state_series(
        source[["time", "open", "high", "low", "close"]], tf_atr=atr,
        source_ref=plan["inputs"]["m1"], atr_source_ref=previous["target_sources"]["m5_atr"],
        cutoff_ns=base.metadata["cutoff_ns"], source_row_offset=base.metadata["source_row_offset"],
        closure_authority=old.read(plan["inputs"]["closure_authority"]),
        exact_schedule=old.read(plan["inputs"]["exact_schedule"]))
    prefix_rows = int(np.searchsorted(axis, cutoff))
    need(np.array_equal(alt.arrays["validity_ns"][:prefix_rows], axis[:prefix_rows])
         and np.array_equal(alt.arrays["local_range"][:prefix_rows], a["local_range"][:prefix_rows])
         and np.array_equal(alt.arrays["threshold"][:prefix_rows], 2 * a["threshold"][:prefix_rows]),
         "TRAIN_RATIO_PARITY")
    tapes = {"0.5": base, "1.0": alt}
    monthly, witnesses = {}, []
    for start, stop in zip(plan["month_edges_ns"][:-1], plan["month_edges_ns"][1:]):
        lo, hi = np.searchsorted(issues, [start, stop])
        source_lo, source_hi = np.searchsorted(axis, [start, stop])
        need(hi > lo and source_hi > source_lo, "MONTH")
        report = {}
        for col, tf in enumerate(target.TIMEFRAMES):
            lead = plan["lead_minutes"][col]
            wanted = issues[lo:hi] + lead * target.MINUTE_NS
            mature = wanted < cutoff
            pos = np.searchsorted(axis, wanted)
            present = mature & (pos < len(axis)) & (axis[np.minimum(pos, len(axis) - 1)] == wanted)
            rows, selected_parents, selected_issues = pos[present], parents[lo:hi][present], issues[lo:hi][present]
            entry = {"issues": int(hi - lo), "mature": int(mature.sum()),
                     "exact_native_validity": len(rows), "factors": {}}
            for factor, tape in tapes.items():
                valid, flat, nonflat, known, state = activity(tape, rows, col, target, np)
                reference = saved["reports"][factor][str(start)][tf]["leads"][str(lead)]
                counts = {name: int(np.count_nonzero(known & (state == code)))
                          for code, name in enumerate(target.STATE_NAMES)}
                need(reference["issues"] == hi - lo and reference["observed"] == int(known.sum())
                     and reference["state_counts"] == counts, "SAVED_COUNTS")
                theta = tape.arrays["threshold"][rows, col]
                spread = ask[rows] - bid[rows]
                scale = valid & (spread > 0)
                subset = np.arange(source_lo, source_hi)
                src_valid, src_flat, _, src_known, src_state = activity(tape, subset, col, target, np)
                strict_path = src_known & tape.arrays["path_observed"][subset, col]
                entry["factors"][factor] = {
                    "saved_threeclass_counts": counts,
                    "complete_activity": int(valid.sum()), "flat": int(flat.sum()),
                    "nonflat": int(nonflat.sum()),
                    "partial_up_or_down": int((nonflat & ~known).sum()),
                    "threeclass_without_complete_activity": int((known & ~valid).sum()),
                    "scale_at_native_validity": {
                        "mask": "complete local window and exact mature native validity",
                        "threshold_usd": quantiles(theta[scale], np),
                        "threshold_bps": quantiles(10000 * theta[scale] / close[rows][scale], np),
                        "threshold_to_quoted_spread": quantiles(theta[scale] / spread[scale], np),
                        "share_threshold_le_two_spreads": float(np.mean(theta[scale] <= 2 * spread[scale])) if scale.any() else None},
                    "binary_runs_unique_source": run_lengths(src_flat.astype(np.int8), src_valid,
                        axis[source_lo:source_hi], ("NONFLAT", "FLAT"), np),
                    "threeclass_runs_strict_path_unique_source": run_lengths(src_state, strict_path,
                        axis[source_lo:source_hi], target.STATE_NAMES, np)}
            old_valid, old_flat, _, _, _ = activity(base, rows, col, target, np)
            alt_valid, alt_flat, alt_nonflat, alt_known, _ = activity(alt, rows, col, target, np)
            categories = {
                "factor_disagreement": old_valid & alt_valid & (old_flat != alt_flat),
                "partial_direction_at_1_0": alt_nonflat & ~alt_known}
            for category in plan["witness_categories"]:
                hits = np.flatnonzero(categories[category])
                if not len(hits):
                    continue
                selected = min(hits, key=lambda i: hashlib.sha256(
                    (STAGE + "|" + str(start) + "|" + tf + "|" + category + "|" +
                     str(int(selected_parents[i]))).encode()).digest())
                witnesses.append(pick_witness(source, int(rows[selected]), target.LOCAL_WINDOW_M1_BARS[tf],
                    selected_parents[selected], selected_issues[selected], tf, start, category,
                    col, tapes, np))
            report[tf] = entry
        monthly[str(start)] = report
    need(len(witnesses) <= plan["max_witnesses"], "WITNESS_CAP")
    old.freeze(binding, plan, snapshots)
    for item in plan["inputs"].values():
        need(old.ref(item["path"]) == item, "BOOKEND")
    return {"schema_version": STAGE + "_result_v1", "plan": binding["plan"],
            "policy": binding["policy"], "claim": claim, "source_head": binding["source_head"],
            "prior_sensitivity": plan["inputs"]["sensitivity_result"],
            "monthly": monthly, "witnesses": witnesses,
            "witness_selection": "one minimum SHA256 rank per month/TF/category among eligible native-lead parents",
            "report_parents": count, "report_cutoff_ns": cutoff,
            "reused_saved_0_5_tape": True, "ephemeral_1_0_target_builds": 1,
            "alternative_series_published": False, "later_aggregates_computed": False,
            "model_calls": 0, "optimizer_steps": 0, "test_data_used": False,
            "target_factor_selected": False,
            "limits": ["TRAIN-only source semantics, not forecast quality or economic edge",
                       "M1 OHLC cannot resolve intraminute event order",
                       "Quoted spread is historical, not executable cost",
                       "Censored durations are lower bounds until an observed class change",
                       "Twelve OHLC summaries cannot show every intrasegment event",
                       "Overlapping parent targets are not independent trials"]}


def main():
    need(len(sys.argv) == 1 and Path.cwd() == REPO, "CLI")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from gx1.scripts import audit_dynamic_mtf_state_target_sensitivity_v1 as old
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    need(ROOT.is_dir() and not any(p.is_symlink() for p in (ROOT, *ROOT.parents)), "RUN_ROOT")
    binding = old.read(old.ref(ROOT / "EXECUTION_BINDING.json"))
    need(set(binding) == {"plan", "policy", "worker", "source_head"}
         and binding["plan"]["path"] == str(PLAN_PATH)
         and binding["policy"]["path"] == str(REPO / "NEXT_RUN_POLICY.json")
         and binding["worker"] == old.ref(__file__), "BINDING")
    plan = old.read(binding["plan"])
    need(plan["schema_version"] == STAGE + "_plan_v1" and plan["stage"] == STAGE
         and plan["run_root"] == str(ROOT) and plan["worker"] == binding["worker"]
         and plan["budget"] == BUDGET and plan["factors"] == [0.5, 1.0]
         and plan["lead_minutes"] == [5, 15, 60, 240, 1440]
         and plan["report_parents"] == 29849 and plan["report_cutoff_ns"] == 1727740800000000000
         and plan["witness_categories"] == ["factor_disagreement", "partial_direction_at_1_0"]
         and plan["max_witnesses"] == 50
         and set(plan["inputs"]) == {"prior_result", "prior_series", "m1", "m1_manifest",
             "calendar_source_reference", "closure_authority", "exact_schedule",
             "published_calendar", "sensitivity_result"}
         and all("TEST" not in Path(item["path"]).parts for item in plan["inputs"].values()), "PLAN")
    policy = old.read(binding["policy"])
    gate = policy[STAGE]
    need(policy["training_enabled"] is False and gate["plan"] == binding["plan"]
         and gate["execution_authorized"] is True and gate["data_reads_authorized"] is True
         and gate["target_builds_authorized"] == 1 and gate["model_calls_authorized"] == 0
         and gate["claimed_invocations"] == gate["completed_invocations"] == 0
         and gate["max_invocations"] == 1 and gate["test_authorized"] is False
         and gate["later_aggregate_authorized"] is False, "ADMISSION")
    need(all(not (ROOT / name).exists() and not (ROOT / name).is_symlink()
             for name in ("CLAIM.json", "RESULT.json", "TERMINAL.json")), "CONSUMED")
    limits = require_capped_cpu_audit_execution()
    limits["lock"] = require_capped_lock_ancestry()
    need(all(limits[k] == BUDGET[k] for k in
             ("memory_max", "swap", "pids", "numerical_threads")), "CAPS")
    snapshots = {item["path"]: old.regular(item["path"]) for item in
                 (*plan["inputs"].values(), *plan["source_owners"].values(),
                  binding["plan"], binding["worker"], binding["policy"])}
    old.freeze(binding, plan, snapshots)
    claim = publish("CLAIM.json", {
        "schema_version": STAGE + "_claim_v1", "plan": binding["plan"],
        "policy": binding["policy"], "worker": binding["worker"],
        "source_head": binding["source_head"], "limits": limits,
        "target_builds": 1, "model_calls": 0,
        "scope": "May-September 2024 TRAIN semantic calibration only; no factor choice"}, old)
    started, before = time.monotonic(), old.stats()

    def deadline(*_):
        raise TimeoutError("SEMANTIC_TARGET_1800_SECOND_DEADLINE")

    signal.signal(signal.SIGALRM, deadline)
    signal.signal(signal.SIGTERM, deadline)
    signal.alarm(BUDGET["max_wall_seconds"])
    try:
        result = audit(plan, binding, claim, snapshots, old)
        result["elapsed_seconds"] = time.monotonic() - started
        result["io_before"], result["io_after"] = before, old.stats()
        output = publish("RESULT.json", result, old)
        publish("TERMINAL.json", {
            "schema_version": STAGE + "_terminal_v1", "status": "COMPLETE", "exit_code": 0,
            "claim": claim, "result": output, "source_head": binding["source_head"],
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": old.stats()["peak_rss_bytes"], "test_data_used": False}, old)
    except BaseException as exc:
        publish("TERMINAL.json", {
            "schema_version": STAGE + "_terminal_v1", "status": "FAILED_CONSUMED", "exit_code": 1,
            "claim": claim, "source_head": binding["source_head"],
            "error": type(exc).__name__ + ":" + str(exc)[:500],
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": old.stats()["peak_rss_bytes"], "test_data_used": False}, old)
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
