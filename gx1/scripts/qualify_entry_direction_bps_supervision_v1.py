"""Bounded synthetic qualification of signed-bps supervision; no genuine native work."""
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
TEST_FILES = (
    "tests/test_entry_native_direction_bps_v1.py",
    "tests/test_entry_native_direction_research_v1.py",
    "tests/test_entry_native_direction_research_state_v1.py",
    "tests/test_entry_native_direction_feature_view_v1.py",
    "tests/test_entry_native_direction_prefix_fit_window_v1.py",
)
ROUTE_TEST_FILES = (
    "tests/test_entry_native_direction_bps_qualification_v1.py",
    "tests/test_entry_native_direction_research_campaign_v1.py",
    "tests/test_entry_native_direction_research_window_v1.py",
)
METADATA_IMPORT = 'import sys; import gx1.contracts.entry_native_direction_research_campaign_v1; assert "torch" not in sys.modules'


def require(ok, reason):
    if not ok:
        raise RuntimeError("BPS_SUPERVISION_QUALIFICATION_" + reason)


def main():
    hp = Q / "inspect_direction_gc_parquet_metadata.py"
    raw = hp.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == META_SHA, "METADATA_OWNER")
    meta = types.ModuleType("bps_supervision_metadata")
    meta.__file__ = str(hp)
    exec(compile(raw, str(hp), "exec"), meta.__dict__)
    pref = {"path": sys.argv[1], "sha256": sys.argv[2]}
    polref = {"path": str(R / "NEXT_RUN_POLICY.json"), "sha256": sys.argv[3]}
    head = sys.argv[4]
    plan, policy = meta.read_json(pref), meta.read_json(polref)
    route = plan["schema_version"] == "gx1_direction_bps_genuine_route_synthetic_plan_v1"
    tests = ROUTE_TEST_FILES if route else TEST_FILES
    stage = policy["direction_bps_genuine_route_synthetic" if route else "direction_bps_supervision_synthetic"]
    require(plan["schema_version"] == ("gx1_direction_bps_genuine_route_synthetic_plan_v1" if route
        else "gx1_direction_bps_supervision_synthetic_plan_v1")
        and plan["wall_seconds"] == 180 and plan["outer_timeout_seconds"] == 210
        and plan["test_files"] == list(tests) and plan["synthetic_only"] is True
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
    require(design["horizons"] == [1, 5, 15] and design["bps_weight"] == 1
        and str(Path(__file__).relative_to(R)) in admission["allowed_repo_files"], "SOURCE_SCOPE")
    cap = meta.source_module("bps_supervision_cap", plan["sources"]["cap"])
    proof, lock = cap.require_capped_cpu_audit_execution(), cap.require_capped_lock_ancestry()
    require(lock["scope"] == proof["cgroup_path"][len("/sys/fs/cgroup"):], "LOCK_SCOPE")
    io = meta.source_module("bps_supervision_io", plan["sources"]["io"])
    meta.PUBLISH = meta.source_module("bps_supervision_publish", plan["sources"]["publication"])
    out = Path(plan["out_dir"])
    require(out.parent == Q and out.parent.resolve(strict=True) == Q, "OUTPUT")
    out.mkdir()
    claim = meta.publish(out, "INVOCATION_CLAIM.json", {
        "plan": pref, "policy": polref, "source_head": head, "execution": proof, "lock": lock})
    start, previous = time.monotonic(), sys.getprofile()
    counters = {"payload_open_attempts": 0, "synthetic_payload_accesses": 0,
        "synthetic_module_constructions": 0, "synthetic_module_calls": 0,
        "synthetic_backwards": 0, "synthetic_optimizer_steps": 0,
        "genuine_native_calls": 0, "metadata_child_imports": 0, "synthetic_autograd_grad_calls": 0}
    records = []
    used = None

    def expired(*args):
        raise RuntimeError("BPS_SUPERVISION_QUALIFICATION_WALL_LIMIT")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(plan["wall_seconds"])
    try:
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        os.environ.pop("PYTEST_ADDOPTS", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        sys.path.insert(0, str(R))
        import numpy as np
        import pandas  # Runtime imports only; no source data loaded.
        import pytest
        import torch
        import torch._dynamo
        from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
        from gx1.contracts import entry_causal_m1_outcomes_v1 as quote
        from gx1.contracts import entry_native_direction_research_state_v1
        real_native_codes = {EntryV10CtxHybridTransformer.__init__.__code__,
            EntryV10CtxHybridTransformer.forward.__code__, quote.query_dynamic_m1_prefix.__code__}
        module_init = torch.nn.Module.__init__.__code__
        module_call = torch.nn.Module._call_impl.__code__
        torch.set_num_threads(1)
        require(not torch.cuda.is_initialized(), "CPU_ONLY")
        io.np = np
        before = io.io_state()

        def guard(event, args):
            if event == "subprocess.Popen":
                require(route and counters["metadata_child_imports"] == 0
                    and args[0] == sys.executable and args[1] == [sys.executable, "-c", METADATA_IMPORT]
                    and args[2] is None and args[3] is None, "ONLY_FIXED_METADATA_IMPORT_CHILD")
                counters["metadata_child_imports"] += 1
            require(event not in ("os.system", "socket.connect", "socket.bind", "os.exec"), "PROCESS_NETWORK")
            if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(args[0]))
                writing = bool(args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
                in_output = path.is_absolute() and out in path.parents
                payload = path.suffix.lower() in (
                    ".npz", ".npy", ".parquet", ".pt", ".pth", ".ckpt", ".arrow", ".feather", ".pkl", ".pickle")
                if payload and in_output:
                    counters["synthetic_payload_accesses"] += 1
                if "/GX1_DATA/" in str(path) or (payload and not in_output):
                    counters["payload_open_attempts"] += 1
                    raise RuntimeError("BPS_SUPERVISION_QUALIFICATION_REAL_PAYLOAD_DENIED")
                require(not writing or (path.is_absolute() and out in path.parents), "OUTPUT_ONLY_WRITES")

        sys.addaudithook(guard)

        def watch(frame, event, result):
            if event != "call":
                return
            code = frame.f_code
            if code in real_native_codes:
                counters["genuine_native_calls"] += 1
                raise RuntimeError("BPS_SUPERVISION_REAL_NATIVE_OR_SOURCE_QUERY_DENIED")
            if code is module_init:
                counters["synthetic_module_constructions"] += 1
            elif code is module_call:
                counters["synthetic_module_calls"] += 1
            elif code.co_name == "backward" and "/torch/" in code.co_filename:
                counters["synthetic_backwards"] += 1
            elif code.co_name == "grad" and "/torch/autograd/" in code.co_filename:
                counters["synthetic_autograd_grad_calls"] += 1
            elif code.co_name == "step" and "/torch/optim/adamw.py" in code.co_filename:
                counters["synthetic_optimizer_steps"] += 1
            for key, maximum in plan["synthetic_work_limits"].items():
                require(counters[key] <= maximum, "SYNTHETIC_WORK_CAP:" + key)

        class Results:
            def pytest_runtest_logreport(self, report):
                records.append({"nodeid": report.nodeid, "phase": report.when,
                    "outcome": report.outcome, "duration_seconds": report.duration})

        sys.setprofile(watch)
        exit_code = int(pytest.main([*tests, "-q", "-s", "--assert=plain",
            "-p", "no:cacheprovider", "-p", "no:logging", "-o", "addopts=",
            "--confcutdir", str(R / "tests"), "--basetemp", str(out / "tmp")], plugins=[Results()]))
        sys.setprofile(previous)
        calls = [row for row in records if row["phase"] == "call"]
        require(exit_code == 0 and len(calls) == plan["expected_test_count"]
            and len({row["nodeid"] for row in calls}) == len(calls)
            and len(records) == 3 * len(calls) and all(row["outcome"] == "passed" for row in records),
            "ALL_FIXED_CASES_PASSED")
        require(counters["genuine_native_calls"] == counters["payload_open_attempts"] == 0
            and counters["metadata_child_imports"] == (1 if route else 0)
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
            "legacy_parity_scope": ("Synthetic B16 campaign scope, unchanged B2 guards, probe/update/exact-resume mechanics; genuine qualification still required." if route else
                "Synthetic unchanged direction tests, RNG/shared-parameter identity and report/optimizer branches; genuine qualification still required."),
            "native_heads_or_loss_changed": not route, "genuine_targets_materialized": False,
            "direction_improvement_qualified": False, "economic_edge_qualified": False, "test_data_used": False})
        terminal = {"status": "COMPLETE", "exit_code": 0, "plan": pref, "policy": polref,
            "claim": claim, "summary": summary, "source_head": head, "source_unchanged": True,
            "execution": proof, "lock": lock, "counters": {**counters, **{}},
            "elapsed_seconds": time.monotonic() - start, "rchar_bytes": used, "test_data_used": False}
    except Exception as exc:
        sys.setprofile(previous)
        meta.publish(out, "TERMINAL.json", {"status": "FAILED", "exit_code": 1,
            "plan": pref, "policy": polref, "claim": claim, "source_head": head,
            "error": type(exc).__name__ + ": " + str(exc), "test_reports": records,
            "execution": proof, "lock": lock, "counters": {**counters, **{}},
            "elapsed_seconds": time.monotonic() - start, "test_data_used": False})
        raise
    finally:
        sys.setprofile(previous)
    result = meta.publish(out, "TERMINAL.json", terminal)
    signal.alarm(0)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
