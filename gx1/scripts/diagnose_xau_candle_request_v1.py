"""One separately admitted diagnostic of the consumed candle request.

No model, labels, ATR, source replacement or automatic clock-probe continuation.
The old claim and plan remain immutable. A new claim is consumed even on failure.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from gx1.scripts import audit_xau_input_physical_clock_v1 as probe

REPO = probe.REPO
DECISION_ID = "XAU_HTTP400_DIAGNOSTIC_20261105_002"
PLAN_RELATIVE = "configs/research/" + DECISION_ID + ".json"
RUN_ROOT = Path("/home/andre2/GX1_RUNS/XAU_INPUT_PHYSICAL_CLOCK_20261105_001/HTTP400_DIAGNOSTIC_002")
REQUEST = {"from": "2026-10-01T09:45:00.000000000Z",
           "to": "2026-10-16T09:45:00.000000000Z", "granularity": "M5", "price": "MBA"}
ENDPOINT = "/instruments/XAU_USD/candles"
LIMITS = {key: probe.LIMITS[key] for key in (
    "max_response_bytes", "http_connect_timeout_seconds", "http_read_timeout_seconds",
    "wall_seconds", "kill_grace_seconds")}
LIMITS["max_get_attempts"] = 1
SOURCE_FILES = tuple(sorted(set(probe.FROZEN_CODE) | {
    "gx1/scripts/diagnose_xau_candle_request_v1.py"}))


def require(ok, reason):
    if not ok:
        raise RuntimeError("XAU_REQUEST_DIAGNOSTIC_" + reason)


def validate_plan(plan):
    require(plan["schema_version"] == "gx1_xau_request_diagnostic_plan_v1"
            and plan["decision_id"] == DECISION_ID, "PLAN_ID")
    require(plan["request"] == {"method": "GET", "endpoint": ENDPOINT, "params": REQUEST}, "REQUEST")
    require(plan["source"] == {"environment": "practice", "instrument": "XAU_USD"}, "SOURCE")
    require(plan["limits"] == LIMITS and plan["run_root"] == str(RUN_ROOT), "BOUNDS")
    require(plan["model_calls"] == 0 and plan["training_enabled"] is False
            and plan["test_access"] is False and plan["orders"] is False
            and plan["spending_usd"] == 0, "AUTHORITY")
    require(set(plan["source_sha256"]) == set(SOURCE_FILES), "SOURCE_INVENTORY")
    start, end = (datetime.fromisoformat(plan[k]) for k in ("not_before_utc", "expires_utc"))
    require(start.tzinfo is not None and end.tzinfo is not None and start < end, "WINDOW")
    return plan


def load_admitted_plan(path, expected_sha256):
    require(path == REPO / PLAN_RELATIVE and not path.is_symlink(), "PLAN_PATH")
    require(probe.sha256_file(path) == expected_sha256, "PLAN_SHA")
    plan = validate_plan(json.loads(path.read_text()))
    policy = json.loads((REPO / "NEXT_RUN_POLICY.json").read_text())
    gate = policy[probe.GOAL_KEY]["http400_diagnostic"]
    require(all(gate.get(k) is True for k in (
        "execution_authorized", "collection_authorized", "broker_network_authorized")), "NOT_ADMITTED")
    require(gate.get("plan") == probe.file_ref(path)
            and gate.get("operator_approval") == DECISION_ID, "EXACT_APPROVAL")
    require(policy["training_enabled"] is False and policy["native_launch_authorized"] is False,
            "MODEL_GATE")
    require(REPO == Path("/home/andre2/src/GX1_CURRENT") and Path.cwd() == REPO, "WORKSPACE")
    require(subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO,
                                   text=True).strip() == "work/gx1-current", "BRANCH")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True), "DIRTY_SOURCE")
    now = datetime.now(timezone.utc)
    require(datetime.fromisoformat(plan["not_before_utc"]) <= now <
            datetime.fromisoformat(plan["expires_utc"]), "EXPIRED_OR_EARLY")
    probe.verify_source_files(plan)
    return plan


def observe_request(plan, journal, *, client_factory=probe.source._load_oanda_client):
    """Shared bounded transport owns the only attempt and preserves provider errors."""
    client = None
    terminal = {"decision_id": DECISION_ID, "model_calls": 0, "test_data_used": False,
                "relaunch_authorized": False, "clock_readiness_measured": False}
    try:
        client = probe.BoundCandleClient(client_factory(), [dict(REQUEST)], journal,
                                        plan["limits"], time.monotonic() + LIMITS["wall_seconds"])
        value = client._request("GET", ENDPOINT, params=dict(REQUEST))
        path = journal.root / "PARSED_RESPONSE.json"
        probe.publish_json(path, value)
        terminal.update(status="HTTP200_RESPONSE_ONLY_NO_SOURCE_OR_CLOCK_QUALIFICATION",
                        response=probe.file_ref(path), exit_code=0)
    except Exception as exc:
        # Provider body is preserved by the transport, never echoed into logs.
        terminal.update(status="FAILED_DIAGNOSTIC", error_type=type(exc).__name__, exit_code=1)
    terminal["get_attempts"] = 0 if client is None else client.attempts
    terminal["http_status_codes"] = [e["payload"]["status_code"] for e in journal.events
                                     if e["event"] == "http_response_status"]
    journal.write("diagnostic_result", dict(terminal))
    return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args()
    plan = load_admitted_plan(args.plan, args.expected_plan_sha256)
    probe.require_timeout_parent()
    probe.require_capped_cpu_audit_execution()
    probe.require_capped_lock_ancestry()
    require(RUN_ROOT.parent.is_dir() and RUN_ROOT.parent.resolve() == RUN_ROOT.parent
            and not RUN_ROOT.exists() and not RUN_ROOT.is_symlink(), "CLAIM_EXISTS")
    RUN_ROOT.mkdir()
    probe.source._fsync_directory(RUN_ROOT.parent)
    journal = probe.Journal(RUN_ROOT)
    journal.write("claim", {"plan": probe.file_ref(args.plan), "request": plan["request"],
                            "source_sha256": plan["source_sha256"]})
    terminal = observe_request(plan, journal)
    try:
        probe.verify_source_files(plan)
        terminal["source_unchanged"] = True
    except Exception:
        terminal.update(status="SOURCE_CHANGED_AFTER_DIAGNOSTIC", source_unchanged=False, exit_code=1)
    probe.publish_json(RUN_ROOT / "TERMINAL.json", terminal)
    print(json.dumps({k: terminal[k] for k in ("status", "get_attempts", "http_status_codes", "exit_code")}))
    return terminal["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
