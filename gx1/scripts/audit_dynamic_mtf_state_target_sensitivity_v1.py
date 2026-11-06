"""One claimed, TRAIN-only sensitivity audit of the dynamic five-TF state target.

Rebuild the frozen 0.5 ATR labels exactly, then compare a process-local 1.0 ATR
diagnostic on the same physical source and fixed May-September 2024 issue set.
No alternative series, model, VAL or TEST result is published.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import stat
import subprocess
import sys
import time

REPO = Path("/home/andre2/src/GX1_CURRENT")
STAGE = "dynamic_mtf_state_target_sensitivity_20261106_001"
PLAN_PATH = REPO / "configs/research/DYNAMIC_MTF_STATE_TARGET_SENSITIVITY_20261106_001.json"
ROOT = Path("/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_TARGET_SENSITIVITY_20261106_001")
BUDGET = {
    "memory_max": 4294967296,
    "swap": 536870912,
    "pids": 64,
    "numerical_threads": 1,
    "max_wall_seconds": 1800,
    "max_invocations": 1,
    "max_json_bytes": 67108864,
}
STAT_KEYS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
JSON_BYTES = 0


def require(ok, reason):
    if not ok:
        raise RuntimeError("DYNAMIC_STATE_SENSITIVITY_" + reason)


def regular(path):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts
            and not any(p.is_symlink() for p in (path, *path.parents)), "PATH:" + str(path))
    s = path.lstat()
    require(stat.S_ISREG(s.st_mode), "REGULAR:" + str(path))
    return {k: int(getattr(s, k)) for k in STAT_KEYS}


def ref(path):
    path = Path(path)
    before = regular(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    require(regular(path) == before, "HASH_BOOKEND:" + str(path))
    return {"path": str(path), "sha256": digest.hexdigest()}


def read(binding):
    require(set(binding) == {"path", "sha256"}
            and regular(binding["path"])["st_size"] <= 16 * 1024**2
            and ref(binding["path"]) == binding, "SMALL_JSON_REF")
    return json.loads(Path(binding["path"]).read_text())


def publish(name, value):
    global JSON_BYTES
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    require(JSON_BYTES + len(data) <= BUDGET["max_json_bytes"], "JSON_BUDGET")
    _publish_bytes(ROOT / name, data)
    (ROOT / name).chmod(0o444)
    JSON_BYTES += len(data)
    return ref(ROOT / name)


def freeze(binding, plan, snapshots):
    require(subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
            == "work/gx1-current", "BRANCH")
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
            == binding["source_head"], "HEAD")
    require(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True) == "", "DIRTY")
    require(ref(binding["policy"]["path"]) == binding["policy"]
            and ref(binding["plan"]["path"]) == binding["plan"]
            and ref(binding["worker"]["path"]) == binding["worker"], "BINDING_CHANGED")
    for item in plan["source_owners"].values():
        require(ref(item["path"]) == item, "SOURCE_CHANGED")
    for path, before in snapshots.items():
        require(regular(path) == before, "PAYLOAD_STAT_CHANGED:" + path)


def stats():
    result = {line.split(":")[0]: int(line.split(":")[1])
              for line in Path("/proc/self/io").read_text().splitlines()}
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return result


def reuse_atr(series, target):
    """Recover the exact causal ATR surface from the published 0.5 ATR tape."""
    import numpy as np

    require(target.MINIMUM_MOVE_ATR == 0.5, "DEFAULT_FACTOR")
    a, surfaces = series.arrays, {}
    for col, tf in enumerate(target.TIMEFRAMES):
        available, theta = a["atr_available_ns"][:, col], a["threshold"][:, col]
        good = available > 0
        require(np.all(theta[~good] == 0.0) and np.all(theta[good] > 0.0), "ATR_MASK:" + tf)
        times, first, inverse = np.unique(available[good], return_index=True, return_inverse=True)
        require(len(times) > 0, "ATR_SUPPORT:" + tf)
        values = theta[good][first]
        require(np.array_equal(theta[good].view(np.uint64),
                               values[inverse].view(np.uint64)), "ATR_GROUP:" + tf)
        atr = (2.0 * values).astype(np.float64)
        pos = np.searchsorted(times, a["validity_ns"], side="right") - 1
        safe = np.maximum(pos, 0)
        roundtrip = np.where(pos >= 0, 0.5 * atr[safe], 0.0)
        require(np.array_equal(roundtrip.view(np.uint64), theta.view(np.uint64))
                and np.array_equal(np.where(pos >= 0, times[safe], 0), available),
                "ATR_ASOF_ROUNDTRIP:" + tf)
        surfaces[tf] = {"available_ns": times.astype(np.int64), "atr": atr}
    return {
        "schema_version": target.SCHEMA_VERSION + "_causal_atr",
        "cutoff_ns": series.metadata["cutoff_ns"],
        "definition_sha256": target.dynamic_mtf_state_target_contract()["contract_sha256"],
        "surfaces": surfaces,
    }


def monthly_report(series, source, *, cutoff_ns, parents, issues, month_edges, leads, target):
    """Only pre-October issues and their strictly mature labels enter this report."""
    import numpy as np

    a = series.arrays
    output = {}
    for start, stop in zip(month_edges[:-1], month_edges[1:]):
        lo = int(np.searchsorted(issues, start))
        hi = int(np.searchsorted(issues, stop))
        require(hi > lo and np.all(issues[lo:hi] < cutoff_ns), "MONTH_ISSUES")
        q = target.query_dynamic_mtf_states(
            series, parent_rows=parents[lo:hi], issue_ns=issues[lo:hi],
            cutoff_ns=cutoff_ns, lead_minutes=leads)
        require(np.all(q["validity_ns"][q["observed"].any(axis=1)] < cutoff_ns),
                "FUTURE_LABEL_CUTOFF")
        month = {}
        window = (a["validity_ns"] >= start) & (a["validity_ns"] < stop)
        rows = np.flatnonzero(window)
        require(len(rows) > 0, "MONTH_SOURCE_ROWS")
        for col, tf in enumerate(target.TIMEFRAMES):
            lead_values = {}
            for lead_col, lead in enumerate(leads):
                observed = q["observed"][:, col, lead_col]
                states = q["state"][:, col, lead_col]
                reasons = q["reason"][:, col, lead_col]
                lead_values[str(lead)] = {
                    "issues": hi - lo,
                    "observed": int(observed.sum()),
                    "state_counts": {name: int(np.count_nonzero(observed & (states == state)))
                                     for state, name in enumerate(target.STATE_NAMES)},
                    "unknown": int((~observed).sum()),
                    "intraminute_order_flagged": int(np.count_nonzero(
                        (reasons & target.INTRAMINUTE_ORDER_UNKNOWN) != 0)),
                    "unknown_intraminute_order": int(np.count_nonzero(
                        ~observed & ((reasons & target.INTRAMINUTE_ORDER_UNKNOWN) != 0))),
                    "path_observed": int(q["path_observed"][:, col, lead_col].sum()),
                    "immature": int(np.count_nonzero((reasons & target.IMMATURE) != 0)),
                }
            sub = a["state"][rows, col]
            obs = a["observed"][rows, col]
            adjacent = np.diff(a["validity_ns"][rows]) == target.MINUTE_NS
            pairs = adjacent & obs[:-1] & obs[1:]
            transitions = {
                left + ">" + right: int(np.count_nonzero(
                    pairs & (sub[:-1] == left_id) & (sub[1:] == right_id)))
                for left_id, left in enumerate(target.STATE_NAMES)
                for right_id, right in enumerate(target.STATE_NAMES) if left_id != right_id
            }
            theta = a["threshold"][rows, col]
            wide = (source["high"].to_numpy()[rows] - source["low"].to_numpy()[rows]) >= theta
            valid_theta = theta > 0
            month[tf] = {
                "leads": lead_values,
                "source_state_counts": {name: int(np.count_nonzero(obs & (sub == state)))
                                        for state, name in enumerate(target.STATE_NAMES)},
                "source_rows": len(rows),
                "source_observed": int(obs.sum()),
                "state_changes_between_adjacent_observed_minutes": transitions,
                "wide_M1_bar_count": int(np.count_nonzero(wide & valid_theta)),
                "wide_M1_bar_denominator": int(valid_theta.sum()),
            }
        output[str(start)] = month
    return output


def audit(plan, claim, binding, snapshots):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target

    require(target.MINIMUM_MOVE_ATR == plan["factors"][0] == 0.5
            and plan["factors"][1] == 1.0, "FACTOR_BINDING")
    require(target.dynamic_mtf_state_target_contract()["contract_sha256"]
            == plan["existing_contract_sha256"], "CURRENT_TARGET_CONTRACT")
    for item in plan["inputs"].values():
        require(ref(item["path"]) == item, "FRESH_INPUT_HASH")
    previous = read(plan["inputs"]["prior_result"])
    require(previous["series"] == plan["inputs"]["prior_series"]
            and previous["target_sources"]["m1"] == plan["inputs"]["m1"]
            and previous["target_contract_sha256"] == plan["existing_contract_sha256"]
            and previous["test_data_used"] is False
            and previous["status"] == "COMPLETE_TARGET_AUDIT_PENDING_RULE_REVIEW"
            and previous["counts"]["model_calls"] == 0, "PRIOR_RESULT")
    calendar = previous["calendar_binding"]
    require(calendar["source_reference"] == plan["inputs"]["calendar_source_reference"]
            and calendar["closure_authority"] == plan["inputs"]["closure_authority"]
            and calendar["exact_schedule"] == plan["inputs"]["exact_schedule"]
            and calendar["published_calendar"] == plan["inputs"]["published_calendar"],
            "CALENDAR_BINDING")
    manifest = read(plan["inputs"]["m1_manifest"])
    require(manifest["timestamp_semantics"] == "bar_start_utc"
            and manifest["timeframe"] == "M1" and manifest["instrument"] == "XAU_USD"
            and manifest["quote_complete_m1"] is True
            and manifest["output_parquet"] == plan["inputs"]["m1"]["path"]
            and manifest["output_parquet_sha256"] == plan["inputs"]["m1"]["sha256"],
            "M1_SOURCE_MANIFEST")
    source_reference = read(calendar["source_reference"])
    schedule, authority = read(calendar["exact_schedule"]), read(calendar["closure_authority"])
    require(source_reference["calendar"] == calendar["published_calendar"]
            and source_reference["m1_source"] == plan["inputs"]["m1"]
            and source_reference["m1_manifest"] == plan["inputs"]["m1_manifest"]
            and source_reference["test_data_used"] is False, "CALENDAR_SOURCE")
    series_manifest = read(plan["inputs"]["prior_series"])
    for item in series_manifest["arrays"].values():
        snapshots[item["path"]] = regular(item["path"])
    old = target.load_dynamic_mtf_state_series(
        plan["inputs"]["prior_series"], expected_source_refs=previous["target_sources"],
        expected_cutoff_ns=previous["fold"]["series_cutoff_ns"])
    a = old.arrays
    require(old.metadata["rows"] == previous["clock_slice_proof"]["selected_rows"]
            and old.metadata["parents"] == previous["fold"]["all_materialized_parents"]
            and old.metadata["closure_binding"] is not None, "SAVED_SERIES")
    parents, issues = a["parent_rows"], a["issue_ns"]
    cutoff = plan["report_cutoff_ns"]
    count = plan["report_parents"]
    require(np.array_equal(parents[:count], np.arange(count, dtype=np.int64))
            and int(np.searchsorted(issues, cutoff)) == count
            and np.all(issues[:count] < cutoff), "REPORT_PARENT_AXIS")
    proof = previous["clock_slice_proof"]
    times = a["validity_ns"] - target.MINUTE_NS
    require(np.array_equal(a["source_row"], np.arange(
                proof["row_start"], proof["row_end_exclusive"], dtype=np.int64))
            and hashlib.sha256(times.astype("<i8", copy=False).tobytes()).hexdigest()
                == proof["selected_clock_sha256"], "SOURCE_CLOCK_PROOF")
    source = pq.read_table(
        plan["inputs"]["m1"]["path"],
        columns=["time", "open", "high", "low", "close"],
        filters=[
            ("time", ">=", pd.Timestamp(proof["requested_start_ns"], tz="UTC").to_pydatetime()),
            ("time", "<", pd.Timestamp(proof["requested_end_exclusive_ns"], tz="UTC").to_pydatetime()),
        ],
        use_threads=False,
    ).to_pandas()
    require(np.array_equal(pd.DatetimeIndex(source.time).as_unit("ns").asi8, times),
            "EXACT_M1_PRICE_SLICE")
    atr = reuse_atr(old, target)
    baseline = target.build_dynamic_mtf_state_series(
        source, tf_atr=atr, source_ref=plan["inputs"]["m1"],
        atr_source_ref=previous["target_sources"]["m5_atr"],
        cutoff_ns=old.metadata["cutoff_ns"],
        source_row_offset=old.metadata["source_row_offset"],
        closure_authority=authority, exact_schedule=schedule)
    baseline = target.bind_dynamic_mtf_state_parents(
        baseline, parent_rows=parents, issue_ns=issues)
    for name in target.SERIES_ARRAY_DTYPES:
        require(baseline.arrays[name].tobytes() == a[name].tobytes(),
                "EXACT_BASELINE_PARITY:" + name)
    month_edges = plan["month_edges_ns"]
    leads = plan["lead_minutes"]
    base_report = monthly_report(
        baseline, source, cutoff_ns=cutoff, parents=parents[:count], issues=issues[:count],
        month_edges=month_edges, leads=leads, target=target)
    del baseline
    gc.collect()

    # This process-local diagnostic changes the target contract hash, never the
    # production source/default or the saved 0.5 series. No 1.0 tape is published.
    target.MINIMUM_MOVE_ATR = plan["factors"][1]
    require(target.dynamic_mtf_state_target_contract()["minimum_move_atr"] == 1.0,
            "ALTERNATIVE_CONTRACT_FACTOR")
    atr["definition_sha256"] = target.dynamic_mtf_state_target_contract()["contract_sha256"]
    alternative = target.build_dynamic_mtf_state_series(
        source, tf_atr=atr, source_ref=plan["inputs"]["m1"],
        atr_source_ref=previous["target_sources"]["m5_atr"],
        cutoff_ns=old.metadata["cutoff_ns"],
        source_row_offset=old.metadata["source_row_offset"],
        closure_authority=authority, exact_schedule=schedule)
    alternative = target.bind_dynamic_mtf_state_parents(
        alternative, parent_rows=parents, issue_ns=issues)
    alt_report = monthly_report(
        alternative, source, cutoff_ns=cutoff, parents=parents[:count], issues=issues[:count],
        month_edges=month_edges, leads=leads, target=target)
    require(set(base_report) == set(alt_report), "MONTH_PARITY")
    freeze(binding, plan, snapshots)
    for item in plan["inputs"].values():
        require(ref(item["path"]) == item, "INPUT_CHANGED")
    return {
        "schema_version": STAGE + "_result_v1",
        "plan": binding["plan"], "policy": binding["policy"], "claim": claim,
        "source_head": binding["source_head"],
        "factors": plan["factors"],
        "target_contract_sha256": {
            "baseline": plan["existing_contract_sha256"],
            "diagnostic": target.dynamic_mtf_state_target_contract()["contract_sha256"],
        },
        "baseline_exact_array_parity": list(target.SERIES_ARRAY_DTYPES),
        "report_cutoff_ns": cutoff,
        "report_parents": count,
        "lead_minutes": leads,
        "month_edges_ns": month_edges,
        "reports": {"0.5": base_report, "1.0": alt_report},
        "later_aggregates_computed": False,
        "alternative_series_published": False,
        "model_calls": 0,
        "optimizer_steps": 0,
        "atr_builds": 0,
        "target_builds": 2,
        "test_data_used": False,
        "source_and_payload_bookends_unchanged": True,
    }


def main():
    require(len(sys.argv) == 1 and Path.cwd() == REPO, "CLI")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    require(ROOT.is_dir() and not any(p.is_symlink() for p in (ROOT, *ROOT.parents)),
            "RUN_ROOT")
    binding = read(ref(ROOT / "EXECUTION_BINDING.json"))
    require(set(binding) == {"plan", "policy", "worker", "source_head"}
            and binding["plan"]["path"] == str(PLAN_PATH)
            and binding["policy"]["path"] == str(REPO / "NEXT_RUN_POLICY.json")
            and binding["worker"] == ref(__file__), "EXECUTION_BINDING")
    plan = read(binding["plan"])
    require(plan["schema_version"] == STAGE + "_plan_v1"
            and plan["stage"] == STAGE and plan["run_root"] == str(ROOT)
            and plan["worker"] == binding["worker"] and plan["budget"] == BUDGET
            and plan["factors"] == [0.5, 1.0]
            and plan["lead_minutes"] == [5, 15, 60, 240, 1440]
            and plan["report_parents"] == 29849
            and plan["report_cutoff_ns"] == 1727740800000000000
            and plan["month_edges_ns"] == [1714521600000000000, 1717200000000000000,
                                           1719792000000000000, 1722470400000000000,
                                           1725148800000000000, 1727740800000000000]
            and set(plan["inputs"]) == {"prior_result", "prior_series", "m1", "m1_manifest",
                                        "calendar_source_reference", "closure_authority",
                                        "exact_schedule", "published_calendar"}
            and all("TEST" not in Path(item["path"]).parts
                    for item in plan["inputs"].values()),
            "PLAN")
    policy = read(binding["policy"])
    gate = policy[STAGE]
    require(policy["training_enabled"] is False
            and gate["plan"] == binding["plan"]
            and gate["execution_authorized"] is True
            and gate["data_reads_authorized"] is True
            and gate["target_builds_authorized"] == 2
            and gate["model_calls_authorized"] == 0
            and gate["claimed_invocations"] == gate["completed_invocations"] == 0
            and gate["max_invocations"] == 1
            and gate["test_authorized"] is False
            and gate["later_aggregate_authorized"] is False,
            "EXACT_TARGET_ONLY_ADMISSION")
    require(all(not (ROOT / name).exists() and not (ROOT / name).is_symlink()
                for name in ("CLAIM.json", "RESULT.json", "TERMINAL.json")),
            "CONSUMED")
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    limits = require_capped_cpu_audit_execution()
    limits["lock"] = require_capped_lock_ancestry()
    require(all(limits[k] == BUDGET[k] for k in
                ("memory_max", "swap", "pids", "numerical_threads")), "CAPS")
    snapshots = {item["path"]: regular(item["path"]) for item in (
        *plan["inputs"].values(), *plan["source_owners"].values(),
        binding["plan"], binding["worker"], binding["policy"])}
    freeze(binding, plan, snapshots)
    claim = publish("CLAIM.json", {
        "schema_version": STAGE + "_claim_v1",
        "plan": binding["plan"], "policy": binding["policy"],
        "worker": binding["worker"], "source_head": binding["source_head"],
        "limits": limits, "target_builds": 2, "model_calls": 0,
        "report_scope": "May-September 2024 TRAIN only",
    })
    started = time.monotonic()
    before = stats()

    def deadline(*_args):
        raise TimeoutError("DYNAMIC_STATE_SENSITIVITY_1800_SECOND_DEADLINE")

    signal.signal(signal.SIGALRM, deadline)
    signal.signal(signal.SIGTERM, deadline)
    signal.alarm(BUDGET["max_wall_seconds"])
    try:
        result = audit(plan, claim, binding, snapshots)
        result["elapsed_seconds"] = time.monotonic() - started
        result["io_before"], result["io_after"] = before, stats()
        result_ref = publish("RESULT.json", result)
        terminal = {
            "schema_version": STAGE + "_terminal_v1",
            "status": "COMPLETE", "exit_code": 0, "claim": claim,
            "result": result_ref, "source_head": binding["source_head"],
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": stats()["peak_rss_bytes"],
            "test_data_used": False,
        }
        publish("TERMINAL.json", terminal)
    except BaseException as exc:
        publish("TERMINAL.json", {
            "schema_version": STAGE + "_terminal_v1",
            "status": "FAILED_CONSUMED", "exit_code": 1,
            "claim": claim, "source_head": binding["source_head"],
            "error": type(exc).__name__ + ":" + str(exc)[:500],
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": stats()["peak_rss_bytes"],
            "test_data_used": False,
        })
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
