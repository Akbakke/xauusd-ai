"""Synthetic API responses only. No broker, credential, model or TEST access."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from gx1.contracts.oanda_history_ingest_approval_v1 import (
    require_approved_oanda_history_ingest,
)
from gx1_guards.gates import GateError
from gx1.scripts import audit_xau_input_physical_clock_v1 as probe


def plan():
    return {"schema_version": probe.PLAN_SCHEMA,
            "decision_id": probe.INPUT_CLOCK_PROBE_DECISION_ID,
            "source": {"environment": "practice", "instrument": "XAU_USD",
                       "endpoint": "/instruments/XAU_USD/candles", "prices": "MBA"},
            "limits": dict(probe.LIMITS), "capture_root": str(probe.CAPTURE_ROOT),
            "slot_rule": "first_M5_close_strictly_after_claim_clock",
            "source_order": ["M5", "M1"], "training_enabled": False,
            "native_test_access": False, "model_calls": 0, "orders": False,
            "spending_usd": 0,
            "issue_window_start_utc": "2026-11-05T08:00:00Z",
            "issue_window_end_utc": "2026-11-06T17:00:00Z",
            "source_sha256": {name: "1" * 64 for name in probe.FROZEN_CODE}}


def native_response(params):
    tf = params["granularity"]
    step = "5min" if tf == "M5" else "1min"
    times = pd.date_range(params["from"], params["to"], freq=step, inclusive="left")
    candles = []
    for stamp in times:
        candles.append({"complete": True, "volume": 10,
                        "time": stamp.isoformat().replace("+00:00", "Z"),
                        "mid": {"o": "10", "h": "11", "l": "9", "c": "10"},
                        "bid": {"o": "9.9", "h": "10.9", "l": "8.9", "c": "9.9"},
                        "ask": {"o": "10.1", "h": "11.1", "l": "9.1", "c": "10.1"}})
    return {"instrument": "XAU_USD", "granularity": tf, "candles": candles}


class Response:
    history = ()
    def __init__(self, value, status=200):
        self.raw, self.status_code = json.dumps(value).encode(), status
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def iter_content(self, chunk_size):
        for offset in range(0, len(self.raw), chunk_size):
            yield self.raw[offset:offset + chunk_size]


class Session:
    def __init__(self, *, status=200, omit_last_m5=False, omit_middle_m1=False):
        self.calls, self.status, self.omit_last_m5 = [], status, omit_last_m5
        self.omit_middle_m1 = omit_middle_m1
    def get_adapter(self, url):
        return SimpleNamespace(max_retries=SimpleNamespace(total=0))
    def request(self, method, url, **kwargs):
        assert method == "GET" and url == "https://api-fxpractice.oanda.com/v3/instruments/XAU_USD/candles"
        assert kwargs["allow_redirects"] is False and kwargs["verify"] is True
        assert kwargs["stream"] is True
        self.calls.append(kwargs["params"])
        value = native_response(kwargs["params"])
        if self.omit_last_m5 and value["granularity"] == "M5":
            value["candles"] = value["candles"][:-1]
        if self.omit_middle_m1 and value["granularity"] == "M1":
            value["candles"].pop(len(value["candles"]) // 2)
        return Response(value, self.status)


def client(session, requests, journal, *, limits=None):
    base = SimpleNamespace(env="practice", base_url=probe.BoundCandleClient.base_url, session=session)
    return probe.BoundCandleClient(base, requests, journal, limits or probe.LIMITS,
                                  probe.time.monotonic() + 300)


def test_plan_has_one_m5_slot_and_exact_four_native_requests():
    value = probe.validate_plan(plan())
    issue = pd.Timestamp("2026-11-05T10:00:00Z").value
    spans, requests = probe.requests_for_slot(value, issue)
    assert [v["granularity"] for v in requests] == ["M5", "M5", "M5", "M1"]
    assert requests[-1] == {"from": "2026-11-05T09:55:00.000000000Z", "to": "2026-11-05T10:00:00.000000000Z",
                            "granularity": "M1", "price": "MBA"}
    assert spans["M5"][1] - spans["M5"][0] == pd.Timedelta(days=35)
    altered = deepcopy(value)
    altered["limits"]["max_get_attempts"] = 5
    with pytest.raises(RuntimeError, match="LIMITS"):
        probe.validate_plan(altered)


def test_new_probe_id_does_not_open_old_historical_ingest():
    with pytest.raises(GateError, match="FORBIDDEN"):
        require_approved_oanda_history_ingest(
            vedtak_id=probe.INPUT_CLOCK_PROBE_DECISION_ID, timeframe="M5",
            publication_mode="bootstrap", start_utc="2009-06-01T00:00:00Z",
            end_utc="2026-07-01T00:00:00Z")


def test_disabled_policy_fails_before_credentials_claim_or_network(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "REPO", tmp_path)
    path = tmp_path / probe.PLAN_RELATIVE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan()))
    (tmp_path / "NEXT_RUN_POLICY.json").write_text(json.dumps({
        probe.GOAL_KEY: {"collection_authorized": False, "broker_network_authorized": False}}))
    def forbidden(*args, **kwargs):
        pytest.fail("closed admission must not load credentials")
    monkeypatch.setattr(probe.source, "_load_oanda_client", forbidden)
    with pytest.raises(RuntimeError, match="ACQUISITION_NOT_ADMITTED"):
        probe.load_admitted_plan(path, probe.sha256_file(path))
    assert not (tmp_path / "CAPTURE").exists()


@pytest.mark.parametrize("bad", ["endpoint", "params", "repeat", "redirect", "byte_cap", "http_400", "http_400_byte_cap"])
def test_transport_budget_and_fail_closed_route(tmp_path, bad):
    params = {"from": "2024-11-01T11:59:00Z", "to": "2024-11-01T12:00:00Z",
              "granularity": "M1", "price": "MBA"}
    journal = probe.Journal(tmp_path)
    session = Session(status=302 if bad == "redirect" else 400 if bad.startswith("http_400") else 200)
    limits = dict(probe.LIMITS)
    if bad in {"byte_cap", "http_400_byte_cap"}: limits["max_response_bytes"] = 8
    bound = client(session, [params], journal, limits=limits)
    path = "/accounts/forbidden" if bad == "endpoint" else "/instruments/XAU_USD/candles"
    actual = dict(params)
    if bad == "params": actual["price"] = "M"
    if bad == "repeat":
        bound._request("GET", path, params=actual)
    with pytest.raises(RuntimeError):
        bound._request("GET", path, params=actual)
    expected = 0 if bad in {"endpoint", "params"} else 1
    assert len(session.calls) == bound.attempts == expected
    if bad in {"redirect", "http_400", "http_400_byte_cap"}:
        statuses = [e["payload"]["status_code"] for e in journal.events if e["event"] == "http_response_status"]
        assert statuses == [302 if bad == "redirect" else 400]
        saved = [e for e in journal.events if e["event"] == "http_error_body_preserved"]
        error_path = tmp_path / "http-error-0001.body"
        if bad == "http_400_byte_cap":
            assert not saved and not error_path.exists()
        else:
            assert len(saved) == 1 and saved[0]["payload"]["ref"] == probe.file_ref(error_path)
            assert error_path.read_bytes() == json.dumps(native_response(params)).encode()
            assert all("response" not in e["payload"] for e in journal.events)


@pytest.mark.parametrize("missing", [None, "last_m5", "middle_m1"])
def test_synthetic_canonical_pair_to_actual_atr_ready_chain(tmp_path, monkeypatch, missing):
    # Only source cleanliness is simulated: all real canonical validation,
    # source snapshots, publication, numerical ATR and clock binding still run.
    monkeypatch.setattr(probe.source, "_require_clean_repository", lambda *a, **k: "1" * 40)
    value = plan()
    issue = pd.Timestamp("2024-11-01T12:00:00Z").value
    _, requests = probe.requests_for_slot(value, issue)
    journal = probe.Journal(tmp_path)
    session = Session(omit_last_m5=missing == "last_m5", omit_middle_m1=missing == "middle_m1")
    bound = client(session, requests, journal)
    if missing:
        with pytest.raises(RuntimeError, match="MISSING_DECLARED_END_BAR_M5" if missing == "last_m5" else "M1_CONTEXT_INCOMPLETE"):
            probe.capture_inputs(value, issue, bound, journal)
        assert not (tmp_path / "INPUT_TIMING.json").exists()
    else:
        assessment = probe.capture_inputs(value, issue, bound, journal)
        assert assessment["status"] == "UNKNOWN"
        assert any(reason.startswith("UTC_INTERVAL_UNKNOWN") for reason in assessment["reasons"])
        witness = json.loads((tmp_path / "INPUT_TIMING.json").read_text())
        assert witness["receipt"]["m1"]["bar_close_ns"] == issue
        assert witness["receipt"]["m5"]["bar_close_ns"] == issue
        assert set(witness["receipt"]["atr"]["bar_close_ns"]) == set(probe.TIMEFRAMES)
        assert (tmp_path / "M1" / "MANIFEST.json").exists()
        assert (tmp_path / "M5" / "MANIFEST.json").exists()
        assert all("response" not in event["payload"] for event in journal.events)
    assert len(session.calls) == 4
    assert sum(e["event"] == "source_response_received" for e in journal.events) == 4
    assert sum(e["event"] == "bundle_published" for e in journal.events) == 2


def test_total_timeout_is_required_even_after_transport():
    # This pytest process is directly under the capped runner, not timeout.
    with pytest.raises(RuntimeError, match="HARD_TIMEOUT_PARENT"):
        probe.require_timeout_parent()


def test_exact_timeout_parent_is_verified_in_a_real_child_process():
    import subprocess
    import sys
    command = ["/usr/bin/timeout", "--kill-after=10s", "900s", sys.executable, "-B", "-c",
               "from gx1.scripts.audit_xau_input_physical_clock_v1 import require_timeout_parent; require_timeout_parent()"]
    result = subprocess.run(command, cwd=probe.REPO, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
