"""One explicitly admitted, input-only M5-entry clock observation.

This does not train or invoke a model, resolve targets, or read native TEST.
M1 and M5 source files retain their canonical schema. All clock evidence is
external to those bundles. A first-after-launch M5 slot is claimed before any
request. A missing source bar or failed attempt ends this one-shot observation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
import math
import os
import sys
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from gx1.contracts.entry_dynamic_mtf_state_targets_v1 import (
    _build_input_tf_atr, dynamic_mtf_state_target_contract,
)
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_audit_execution, require_capped_lock_ancestry,
)
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
from gx1.contracts.oanda_history_ingest_approval_v1 import (
    INPUT_CLOCK_PROBE_DECISION_ID, INPUT_CLOCK_FOLLOWUP_DECISION_ID,
)
from gx1.contracts.xau_input_clock_v1 import (
    RECEIPT_SCHEMA, TIMEFRAMES, assess_input_timing, capture_clock_sample,
    input_binding_sha256,
)
from gx1.contracts.xau_tape_provenance_v1 import (
    CANONICAL_NATIVE_PRODUCER_SOURCE_FILES,
    canonical_json_sha256, native_timeframe_policy, sha256_file,
)
from gx1.scripts import backfill_xauusd_m5_from_oanda as source

REPO = Path(__file__).resolve().parents[2]
GOAL_KEY = "xau_input_physical_clock_goal_20261105"
PLAN_SCHEMA = "gx1_xau_input_clock_plan_v1"
M5_NS = 300_000_000_000
MINUTE_NS = 60_000_000_000
PLAN_RELATIVE = "configs/research/XAU_INPUT_PHYSICAL_CLOCK_20261105_001.json"
CAPTURE_ROOT = Path("/home/andre2/GX1_RUNS/XAU_INPUT_PHYSICAL_CLOCK_20261105_001/CAPTURE")
FOLLOWUP_PLAN_RELATIVE = "configs/research/XAU_INPUT_PHYSICAL_CLOCK_20261105_003.json"
FOLLOWUP_CAPTURE_ROOT = Path("/home/andre2/GX1_RUNS/XAU_INPUT_PHYSICAL_CLOCK_20261105_003/CAPTURE")
# Independent immutable plans and claims; old approval never admits the followup.
PLAN_SCOPES = {
    INPUT_CLOCK_PROBE_DECISION_ID: (PLAN_RELATIVE, CAPTURE_ROOT, None),
    INPUT_CLOCK_FOLLOWUP_DECISION_ID: (FOLLOWUP_PLAN_RELATIVE, FOLLOWUP_CAPTURE_ROOT, "followup_measurement"),
}
LIMITS = {"warmup_calendar_days": 35, "m1_context_minutes": 5,
          "max_get_attempts": 4, "max_response_bytes": 16 * 1024 * 1024,
          "http_connect_timeout_seconds": 10, "http_read_timeout_seconds": 30,
          "wall_seconds": 900, "kill_grace_seconds": 10}


FROZEN_CODE = tuple(sorted(set(CANONICAL_NATIVE_PRODUCER_SOURCE_FILES) | {
    "gx1/scripts/audit_xau_input_physical_clock_v1.py",
    "gx1/contracts/xau_input_clock_v1.py",
    "gx1/contracts/oanda_history_ingest_approval_v1.py",
    "gx1/contracts/entry_dynamic_mtf_state_targets_v1.py",
    "gx1/contracts/entry_exit_feature_base_v1.py",
    "gx1/contracts/entry_gc_mtf_history_v1.py",
    "gx1/contracts/gx1_capped_execution_v1.py",
    "gx1/contracts/immutable_event_authority_v1.py",
    "gx1/features/htf_features.py",
    "gx1/features/technical_indicators_v1.py",
    "gx1/utils/artifact_primitives_v1.py",
    "scripts/gx1_capped_run.sh",
}))


def require(ok, reason):
    if not ok:
        raise RuntimeError("XAU_INPUT_CLOCK_PROBE_" + reason)


def require_timeout_parent():
    """The total cap covers validation/ATR too, including unresponsive descendants."""
    parent = Path("/proc") / str(os.getppid())
    require((parent / "exe").resolve() == Path("/usr/bin/timeout").resolve(), "HARD_TIMEOUT_PARENT")
    args = (parent / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    own = Path("/proc/self/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    require(len(args) >= 5 and args[1:3] ==
            [f"--kill-after={LIMITS['kill_grace_seconds']}s".encode(),
             f"{LIMITS['wall_seconds']}s".encode()]
            and Path(os.fsdecode(args[3])).resolve() == Path(sys.executable).resolve()
            and args[4:] == own[1:], "HARD_TIMEOUT_COMMAND")


def utc_text(ns):
    value = datetime.fromtimestamp(ns // 1_000_000_000, timezone.utc)
    value += timedelta(microseconds=(ns % 1_000_000_000) // 1000)
    return value.isoformat()


def publish_json(path, value):
    _publish_bytes(path, json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode() + b"\n")


def file_ref(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def validate_plan(plan):
    require(plan["schema_version"] == PLAN_SCHEMA and
            plan["decision_id"] in PLAN_SCOPES, "PLAN_ID")
    require(plan["source"] == {"environment": "practice", "instrument": "XAU_USD",
            "endpoint": "/instruments/XAU_USD/candles", "prices": "MBA"}, "SOURCE")
    require(plan["limits"] == LIMITS, "LIMITS")
    require(plan["capture_root"] == str(PLAN_SCOPES[plan["decision_id"]][1]), "OUTPUT_SCOPE")
    require(plan["slot_rule"] == "first_M5_close_strictly_after_claim_clock", "SLOT_RULE")
    require(plan["source_order"] == ["M5", "M1"], "SOURCE_ORDER")
    require(plan["training_enabled"] is False and plan["native_test_access"] is False
            and plan["model_calls"] == 0 and plan["orders"] is False
            and plan["spending_usd"] == 0, "AUTHORITY")
    start, end = (pd.Timestamp(plan[k]) for k in ("issue_window_start_utc", "issue_window_end_utc"))
    require(start.tzinfo is not None and end.tzinfo is not None
            and start.value < end.value and start.value % M5_NS == end.value % M5_NS == 0,
            "ISSUE_WINDOW")
    require(set(plan["source_sha256"]) == set(FROZEN_CODE), "SOURCE_INVENTORY")
    return plan


def verify_source_files(plan):
    for relative, digest in plan["source_sha256"].items():
        candidate = REPO / relative
        require(candidate.is_file() and not candidate.is_symlink() and
                candidate.resolve().is_relative_to(REPO) and sha256_file(candidate) == digest,
                "SOURCE_CHANGED")


def load_admitted_plan(path, expected_sha256):
    """Reject before credential access, network, claim or data writes."""
    require(path in {REPO / scope[0] for scope in PLAN_SCOPES.values()}
            and not path.is_symlink(), "PLAN_PATH")
    require(sha256_file(path) == expected_sha256, "PLAN_SHA")
    plan = validate_plan(json.loads(path.read_text()))
    policy = json.loads((REPO / "NEXT_RUN_POLICY.json").read_text())
    relative, _, gate_key = PLAN_SCOPES[plan["decision_id"]]
    require(path == REPO / relative, "PLAN_PATH_ID")
    goal = policy[GOAL_KEY] if gate_key is None else policy[GOAL_KEY].get(gate_key, {})
    require(all(goal.get(key) is True for key in
                ("execution_authorized", "collection_authorized", "broker_network_authorized")),
            "ACQUISITION_NOT_ADMITTED")
    require(goal.get("completed_invocations") == 0, "PLAN_CONSUMED")
    require(goal.get("measurement_plan") == file_ref(path), "POLICY_PLAN")
    require(goal.get("operator_measurement_approval") == plan["decision_id"],
            "OPERATOR_APPROVAL")
    require(policy["training_enabled"] is False and policy["native_launch_authorized"] is False,
            "MODEL_GATE")
    require(REPO == Path("/home/andre2/src/GX1_CURRENT") and Path.cwd() == REPO, "WORKSPACE")
    require(subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO,
                                   text=True).strip() == "work/gx1-current", "BRANCH")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True),
            "DIRTY_SOURCE")
    verify_source_files(plan)
    return plan


def requests_for_slot(plan, issue_ns):
    """Exact parameters are sealed into CLAIM before the first GET."""
    end = pd.Timestamp(issue_ns, unit="ns", tz="UTC")
    spans = {"M5": (end - pd.Timedelta(days=plan["limits"]["warmup_calendar_days"]), end),
             "M1": (end - pd.Timedelta(minutes=plan["limits"]["m1_context_minutes"]), end)}
    requests = []
    for tf in plan["source_order"]:
        cursor, stop = spans[tf]
        _, native = native_timeframe_policy(tf)
        while cursor < stop:
            next_end = min(stop, cursor + pd.Timedelta(days=native["request_chunk_days"]))
            requests.append({"from": source._request_timestamp(cursor),
                             "to": source._request_timestamp(next_end),
                             "granularity": tf, "price": "MBA"})
            cursor = next_end
    require(len(requests) == plan["limits"]["max_get_attempts"], "REQUEST_BUDGET")
    return spans, requests


class Journal:
    """Small external no-replace journal; raw market responses are never copied here."""
    def __init__(self, root):
        self.root, self.events, self.previous = root, [], None

    def write(self, event, payload, stamp=None):
        stamp = capture_clock_sample() if stamp is None else stamp
        record = {"event": event, "stamp": stamp, "payload": payload,
                  "previous_sha256": self.previous}
        path = self.root / f"event-{len(self.events):04d}.json"
        publish_json(path, record)
        self.previous = sha256_file(path)
        self.events.append(record)
        return stamp

    def observe(self, event, payload):
        stamp = capture_clock_sample()  # precedes response hashing/serialization
        safe = {k: v for k, v in payload.items() if k != "response"}
        if "response" in payload:
            safe["parsed_response_sha256"] = canonical_json_sha256(payload["response"])
        self.write(event, safe, stamp)


class BoundCandleClient:
    """Only the four already claimed candle GETs; no retry, redirect or account endpoint."""
    env = "practice"
    base_url = "https://api-fxpractice.oanda.com/v3"

    def __init__(self, client, expected_requests, journal, limits, deadline):
        require(client.env == self.env and client.base_url == self.base_url, "CLIENT_SOURCE")
        self.session, self.expected, self.journal = client.session, expected_requests, journal
        self.limits, self.deadline, self.attempts = limits, deadline, 0
        self.session.trust_env = False
        require(self.session.get_adapter(self.base_url).max_retries.total == 0, "TRANSPORT_RETRY")

    def _request(self, method, path, *, params):
        require(method == "GET" and path == "/instruments/XAU_USD/candles", "ENDPOINT")
        require(self.attempts < len(self.expected)
                and params == self.expected[self.attempts], "REQUEST_NOT_CLAIMED")
        require(time.monotonic() < self.deadline, "DEADLINE")
        self.journal.write("http_attempt_claimed", {"attempt": self.attempts + 1,
                                                  "endpoint": path, "params": params})
        self.attempts += 1  # consumed even if request outcome is unknown
        with self.session.request("GET", self.base_url + path, params=params,
                                  allow_redirects=False, stream=True, verify=True,
                                  timeout=(self.limits["http_connect_timeout_seconds"],
                                           self.limits["http_read_timeout_seconds"])) as response:
            self.journal.write("http_response_status", {"attempt": self.attempts,
                                                        "status_code": response.status_code})
            chunks, count = [], 0
            for part in response.iter_content(chunk_size=65536):
                require(time.monotonic() < self.deadline, "DEADLINE")
                count += len(part)
                require(count <= self.limits["max_response_bytes"], "RESPONSE_BYTES")
                chunks.append(part)
            stamp = capture_clock_sample()
            raw = b"".join(chunks)
            self.journal.write("response_body_buffered",
                               {"attempt": self.attempts, "bytes": count,
                                "body_sha256": hashlib.sha256(raw).hexdigest()}, stamp)
            if response.status_code != 200 or response.history:
                # Preserve the bounded provider error privately, outside Git and
                # terminal messages. Status alone cannot identify a bad request.
                error_path = self.journal.root / f"http-error-{self.attempts:04d}.body"
                _publish_bytes(error_path, raw)
                self.journal.write("http_error_body_preserved",
                                   {"attempt": self.attempts, "ref": file_ref(error_path)})
                require(False, "HTTP_STATUS_OR_REDIRECT")
        value = json.loads(raw)
        require(isinstance(value, dict), "RESPONSE_OBJECT")
        return value


def published_frame(root, manifest):
    frames = []
    for year, expected in sorted(manifest["year_sha256"].items()):
        path = root / year / "part-000.parquet"
        require(sha256_file(path) == expected, "PUBLISHED_YEAR_SHA")
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def capture_inputs(plan, issue_ns, client, journal, *, repo_root=REPO):
    """One source/ATR snapshot only; no whole-model readiness or TRAIN parity claim."""
    spans, _ = requests_for_slot(plan, issue_ns)
    root, descriptors, frames = journal.root, {}, {}
    for tf in plan["source_order"]:
        start, end = spans[tf]
        output = root / tf
        manifest = source.materialize_native_xau_snapshot(
            client=client, timeframe=tf, vedtak_id=plan["decision_id"],
            start_utc=start, end_utc=end, out_root=output, repo_root=repo_root,
            event_observer=journal.observe)
        descriptors[tf] = file_ref(output / "MANIFEST.json")
        frames[tf] = published_frame(output, manifest)
        for chunk in manifest["source_chunks"]:
            payload = json.loads(gzip.decompress((output / chunk["relative_path"]).read_bytes()))
            witnessed = [e for e in journal.events if e["event"] == "source_response_received"
                         and e["payload"]["timeframe"] == tf
                         and e["payload"]["sequence"] == chunk["sequence"]]
            require(len(witnessed) == 1 and witnessed[0]["payload"]["parsed_response_sha256"] ==
                    canonical_json_sha256(payload["response"]), "RESPONSE_WITNESS_BINDING")
    expected_m1 = pd.date_range(spans["M1"][0], spans["M1"][1], freq="1min", inclusive="left").asi8
    actual_m1 = pd.DatetimeIndex(pd.to_datetime(frames["M1"]["time"], utc=True)).as_unit("ns").asi8
    require(np.array_equal(actual_m1, expected_m1), "M1_CONTEXT_INCOMPLETE")
    close = {}
    for tf in ("M1", "M5"):
        _, native = native_timeframe_policy(tf)
        close[tf] = int(pd.Timestamp(frames[tf]["time"].iloc[-1]).value
                        + native["bar_seconds"] * 1_000_000_000)
        require(close[tf] == issue_ns, "MISSING_DECLARED_END_BAR_" + tf)
    atr = _build_input_tf_atr(frames["M5"], cutoff_ns=issue_ns + MINUTE_NS)
    journal.write("atr_computed", {"definition_sha256": atr["definition_sha256"]})
    arrays, selected = {}, {}
    for tf in TIMEFRAMES:
        item = atr["surfaces"][tf]
        require(len(item["atr"]) > 0 and math.isfinite(float(item["atr"][-1]))
                and float(item["atr"][-1]) > 0, "ATR_SUPPORT_" + tf)
        arrays[tf + "_close_ns"], arrays[tf + "_atr"] = item["available_ns"], item["atr"]
        selected[tf] = int(item["available_ns"][-1])
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    atr_path = root / "ATR_INPUT.npz"
    _publish_bytes(atr_path, buffer.getvalue())
    atr_stamp = journal.write("atr_snapshot_published", {"ref": file_ref(atr_path)})
    snapshot = {"scope": "M1_native_M5_five_TF_ATR_only_not_complete_model_inputs",
                "issue_ns": issue_ns, "source_refs": descriptors,
                "source_bar_close_ns": close, "atr": file_ref(atr_path),
                "atr_contract_sha256": atr["definition_sha256"], "atr_bar_close_ns": selected,
                "warmup_scope": "35-calendar-day explicitly separate probe origin; no original TRAIN/serve parity claim"}
    input_path = root / "INPUT_SNAPSHOT.json"
    publish_json(input_path, snapshot)
    ready = journal.write("input_snapshot_published", {"ref": file_ref(input_path)})
    receipt = {"schema_version": RECEIPT_SCHEMA, "issue_ns": issue_ns}
    for tf in ("M1", "M5"):
        received = [e["stamp"] for e in journal.events if e["event"] == "source_response_received"
                    and e["payload"]["timeframe"] == tf]
        published = [e for e in journal.events if e["event"] == "bundle_published"
                     and e["payload"]["timeframe"] == tf]
        require(len(published) == 1 and published[0]["payload"]["manifest_sha256"] ==
                descriptors[tf]["sha256"], "PUBLICATION_BINDING")
        receipt[tf.lower()] = {"ref": descriptors[tf], "bar_close_ns": close[tf],
                              "received": received[-1], "published": published[0]["stamp"]}
    receipt["atr"] = {"source_m5_ref": descriptors["M5"], "contract_sha256": atr["definition_sha256"],
                      "value_sha256": sha256_file(atr_path), "bar_close_ns": selected, "ready": atr_stamp}
    receipt["input_ready"] = {"stamp": ready, "input_sha256": sha256_file(input_path)}
    receipt["input_ready"]["binding_sha256"] = input_binding_sha256(receipt)
    assessment = assess_input_timing(receipt, expected_m1_ref=descriptors["M1"],
                                    expected_m5_ref=descriptors["M5"],
                                    expected_atr_contract_sha256=dynamic_mtf_state_target_contract()["contract_sha256"])
    publish_json(root / "INPUT_TIMING.json", {"receipt": receipt, "assessment": assessment})
    return assessment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args()
    plan = load_admitted_plan(args.plan, args.expected_plan_sha256)
    require_timeout_parent()
    require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    start = capture_clock_sample()
    issue_ns = (start["utc_ns"] // M5_NS + 1) * M5_NS
    require(pd.Timestamp(plan["issue_window_start_utc"]).value <= issue_ns <
            pd.Timestamp(plan["issue_window_end_utc"]).value, "ISSUE_OUTSIDE_ADMITTED_WINDOW")
    spans, requests = requests_for_slot(plan, issue_ns)
    capture_root = PLAN_SCOPES[plan["decision_id"]][1]
    require(capture_root.parent.is_dir() and capture_root.parent.resolve() == capture_root.parent
            and not capture_root.exists()
            and not capture_root.is_symlink(), "CLAIM_ALREADY_EXISTS")
    capture_root.mkdir()
    source._fsync_directory(capture_root.parent)
    journal = Journal(capture_root)
    deadline = time.monotonic() + plan["limits"]["wall_seconds"]
    journal.write("claim", {"plan": file_ref(args.plan), "source_sha256": plan["source_sha256"],
                           "issue_ns": issue_ns, "requests": requests,
                           "deadline_clock": "local monotonic; external UTC uncertainty remains UNKNOWN"}, start)
    client = None
    try:
        while time.time_ns() < issue_ns:
            require(time.monotonic() < deadline, "DEADLINE")
            time.sleep(max(0, min(30, (issue_ns - time.time_ns()) / 1_000_000_000)))
        client = BoundCandleClient(source._load_oanda_client(), requests, journal, plan["limits"], deadline)
        assessment = capture_inputs(plan, issue_ns, client, journal)
        require(client.attempts == len(requests), "ATTEMPT_COUNT")
        verify_source_files(plan)
        journal.write("terminal", {"status": "MEASURED", "assessment": assessment,
                                  "get_attempts": client.attempts, "training_enabled": False})
        print(json.dumps({"status": "MEASURED", "absolute_timing": assessment["status"],
                          "get_attempts": client.attempts}))
    except BaseException as exc:
        journal.write("terminal", {"status": "FAILED_OR_INTERRUPTED", "error_type": type(exc).__name__,
                                  "error": str(exc)[:300], "get_attempts": 0 if client is None else client.attempts,
                                  "training_enabled": False})
        raise


if __name__ == "__main__":
    main()
