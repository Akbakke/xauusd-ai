"""Bounded synthetic qualification of the explicit minute price-target axis."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import types

sys.dont_write_bytecode = True
R = Path("/home/andre2/src/GX1_CURRENT")
Q = Path("/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2")
META_SHA = "a648c47880b6b195d1213741a0826d34c7e475bd75193313a4c26f7723798c7a"
TEST_FILE = "tests/test_entry_direction_price_projection_v1.py"


def require(ok, reason):
    if not ok:
        raise RuntimeError("MINUTE_TARGET_QUALIFICATION_" + reason)


def main():
    hp = Q / "inspect_direction_gc_parquet_metadata.py"
    raw = hp.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == META_SHA, "METADATA_OWNER")
    meta = types.ModuleType("minute_target_metadata")
    meta.__file__ = str(hp)
    exec(compile(raw, str(hp), "exec"), meta.__dict__)
    pref = {"path": sys.argv[1], "sha256": sys.argv[2]}
    polref = {"path": str(R / "NEXT_RUN_POLICY.json"), "sha256": sys.argv[3]}
    head = sys.argv[4]
    plan, policy = meta.read_json(pref), meta.read_json(polref)
    stage = policy["direction_minute_target_synthetic"]
    require(plan["schema_version"] == "gx1_direction_minute_target_synthetic_plan_v1"
        and plan["wall_seconds"] == 120 and plan["outer_timeout_seconds"] == 150
        and plan["test_file"] == TEST_FILE and plan["synthetic_only"] is True
        and plan["test_data_used"] is False and plan["payload_paths"] == [], "PLAN")
    require(stage["status"] == "ADMITTED" and stage["plan"] == pref
        and stage["execution_authorized"] is True and stage["saved_payload_reads_authorized"] is False
        and stage["max_invocations"] == 1 and stage["claimed_invocations"] == stage["completed_invocations"] == 0
        and stage["relaunch_allowed"] is False, "ONE_SHOT_AUTHORITY")
    require(all(policy[k] is False for k in ("training_enabled", "native_training_enabled",
        "native_launch_authorized", "full_epoch_training_allowed", "full_val_allowed")), "NATIVE_GATES")
    require(meta.git_head() == head, "HEAD")
    for ref in plan["sources"].values():
        meta.read_ref(ref)
    require(plan["sources"]["worker"]["path"] == str(Path(__file__).resolve()), "WORKER")
    design = meta.read_json(plan["design"])
    admission = meta.read_json(plan["source_admission"])
    require(design["horizons"]["dense_minutes"] == list(range(1, 61))
        and design["horizons"]["fixed_report_anchors"] == [1, 5, 10, 15, 30, 60]
        and str(Path(__file__).relative_to(R)) in admission["allowed_repo_files"], "SOURCE_SCOPE")
    cap = meta.source_module("minute_target_cap", plan["sources"]["cap"])
    proof, lock = cap.require_capped_cpu_audit_execution(), cap.require_capped_lock_ancestry()
    require(lock["scope"] == proof["cgroup_path"][len("/sys/fs/cgroup"):], "LOCK_SCOPE")
    io = meta.source_module("minute_target_io", plan["sources"]["io"])
    meta.PUBLISH = meta.source_module("minute_target_publish", plan["sources"]["publication"])
    out = Path(plan["out_dir"])
    require(out.parent == Q and out.parent.resolve(strict=True) == Q, "OUTPUT")
    out.mkdir()
    claim = meta.publish(out, "INVOCATION_CLAIM.json", {
        "plan": pref, "policy": polref, "source_head": head, "execution": proof, "lock": lock})
    start, previous = time.monotonic(), sys.getprofile()
    counters = {"payload_open_attempts": 0, "synthetic_query_calls": 0}
    records = []
    used = None

    def expired(*args):
        raise RuntimeError("MINUTE_TARGET_QUALIFICATION_WALL_LIMIT")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(plan["wall_seconds"])
    try:
        os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        os.environ.pop("PYTEST_ADDOPTS", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        sys.path.insert(0, str(R))
        import numpy as np
        import pandas  # Runtime imports only; no source data loaded.
        import pytest
        import torch
        from gx1.contracts import entry_causal_m1_outcomes_v1 as quote
        torch.set_num_threads(1)
        require(not torch.cuda.is_initialized(), "CPU_ONLY")
        io.np = np
        before = io.io_state()

        def guard(event, args):
            require(event not in ("subprocess.Popen", "os.system", "socket.connect",
                "socket.bind", "os.exec"), "PROCESS_NETWORK")
            if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(args[0]))
                writing = bool(args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
                forbidden = "/GX1_DATA/" in str(path) or path.suffix.lower() in (
                    ".npz", ".npy", ".parquet", ".pt", ".pth", ".ckpt", ".arrow", ".feather", ".pkl", ".pickle")
                if forbidden:
                    counters["payload_open_attempts"] += 1
                    raise RuntimeError("MINUTE_TARGET_QUALIFICATION_REAL_PAYLOAD_DENIED")
                require(not writing or (path.is_absolute() and out in path.parents), "OUTPUT_ONLY_WRITES")

        sys.addaudithook(guard)

        def watch(frame, event, result):
            # Existing endpoint tests query the same owner on in-memory synthetic
            # fixtures. This exact code is the only exception to the old raw-source
            # observer. File reads and native operations stay forbidden.
            if frame.f_code is quote.query_dynamic_m1_prefix.__code__:
                if event == "call":
                    counters["synthetic_query_calls"] += 1
                    require(counters["synthetic_query_calls"] <= 12, "SYNTHETIC_QUERY_CAP")
                    index = frame.f_locals["index"]
                    require(len(index.source.times) <= 256, "SYNTHETIC_QUERY_SIZE")
                return
            io.observe_work(frame, event, result)

        class Results:
            def pytest_runtest_logreport(self, report):
                records.append({"nodeid": report.nodeid, "phase": report.when,
                    "outcome": report.outcome, "duration_seconds": report.duration})

        sys.setprofile(watch)
        exit_code = int(pytest.main([TEST_FILE, "-q", "-s", "--assert=plain",
            "-p", "no:cacheprovider", "-p", "no:logging", "-o", "addopts=",
            "--confcutdir", str(R / "tests")], plugins=[Results()]))
        sys.setprofile(previous)
        calls = [row for row in records if row["phase"] == "call"]
        require(exit_code == 0 and len(calls) == plan["expected_test_count"]
            and len({row["nodeid"] for row in calls}) == len(calls)
            and len(records) == 3 * len(calls) and all(row["outcome"] == "passed" for row in records),
            "ALL_FIXED_CASES_PASSED")
        require(not any(io.FORBIDDEN.values()) and counters["payload_open_attempts"] == 0
            and not torch.cuda.is_initialized(), "NO_GENUINE_OR_NATIVE_WORK")
        used = io.io_state()["rchar"] - before["rchar"]
        require(used <= plan["max_rchar_bytes"], "IO_CAP")
        for path, (identity, digest) in list(meta.PINNED.items()):
            require(meta.regular(path) == identity, "SOURCE_IDENTITY")
            meta.read_ref({"path": path, "sha256": digest})
        require(meta.git_head() == head, "HEAD_BOOKEND")
        summary = meta.publish(out, "SUMMARY.json", {"status": "PASS_SYNTHETIC_ONLY",
            "plan": pref, "source_head": head, "cases": calls,
            "passed": len(calls), "scope": design["scope"], "horizons": design["horizons"],
            "legacy_parity_scope": "Synthetic original tests and dense-column parity only; no genuine full-TRAIN target comparison.",
            "native_heads_or_loss_changed": False, "genuine_targets_materialized": False,
            "direction_improvement_qualified": False, "economic_edge_qualified": False, "test_data_used": False})
        terminal = {"status": "COMPLETE", "exit_code": 0, "plan": pref, "policy": polref,
            "claim": claim, "summary": summary, "source_head": head, "source_unchanged": True,
            "execution": proof, "lock": lock, "counters": {**counters, **io.FORBIDDEN},
            "elapsed_seconds": time.monotonic() - start, "rchar_bytes": used, "test_data_used": False}
    except Exception as exc:
        sys.setprofile(previous)
        meta.publish(out, "TERMINAL.json", {"status": "FAILED", "exit_code": 1,
            "plan": pref, "policy": polref, "claim": claim, "source_head": head,
            "error": type(exc).__name__ + ": " + str(exc), "test_reports": records,
            "execution": proof, "lock": lock, "counters": {**counters, **io.FORBIDDEN},
            "elapsed_seconds": time.monotonic() - start, "test_data_used": False})
        raise
    finally:
        sys.setprofile(previous)
    result = meta.publish(out, "TERMINAL.json", terminal)
    signal.alarm(0)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
