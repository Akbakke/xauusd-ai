"""Offline only; the consumed clock probe is never relaunched."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from gx1.scripts import diagnose_xau_candle_request_v1 as diagnostic


def plan():
    return {"schema_version": "gx1_xau_request_diagnostic_plan_v1",
            "decision_id": diagnostic.DECISION_ID,
            "request": {"method": "GET", "endpoint": diagnostic.ENDPOINT, "params": dict(diagnostic.REQUEST)},
            "source": {"environment": "practice", "instrument": "XAU_USD"},
            "limits": dict(diagnostic.LIMITS), "run_root": str(diagnostic.RUN_ROOT),
            "model_calls": 0, "training_enabled": False, "test_access": False,
            "orders": False, "spending_usd": 0,
            "not_before_utc": "2026-11-05T10:55:00+00:00", "expires_utc": "2026-11-06T17:00:00+00:00",
            "source_sha256": {name: "1" * 64 for name in diagnostic.SOURCE_FILES}}


def test_fixed_request_and_one_attempt_are_not_changed_by_a_plan():
    p = plan()
    diagnostic.validate_plan(p)
    for key, value in (("granularity", "M1"), ("price", "M"), ("to", "2026-11-01T00:00:00Z")):
        altered = deepcopy(p)
        altered["request"]["params"][key] = value
        with pytest.raises(RuntimeError, match="REQUEST"):
            diagnostic.validate_plan(altered)
    p["limits"]["max_get_attempts"] = 2
    with pytest.raises(RuntimeError, match="BOUNDS"):
        diagnostic.validate_plan(p)


def test_new_diagnostic_needs_its_own_exact_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "REPO", tmp_path)
    path = tmp_path / diagnostic.PLAN_RELATIVE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan()))
    (tmp_path / "NEXT_RUN_POLICY.json").write_text(json.dumps({diagnostic.probe.GOAL_KEY: {
        "operator_measurement_approval": "XAU_INPUT_PHYSICAL_CLOCK_20261105_001",
        "http400_diagnostic": {"execution_authorized": False, "collection_authorized": False,
                               "broker_network_authorized": False}}}))
    with pytest.raises(RuntimeError, match="NOT_ADMITTED"):
        diagnostic.load_admitted_plan(path, diagnostic.probe.sha256_file(path))
    assert not (tmp_path / "HTTP400_DIAGNOSTIC_002").exists()


@pytest.mark.parametrize("status", [200, 400])
def test_one_response_is_preserved_without_automatic_continuation(tmp_path, status):
    value = {"candles": []} if status == 200 else {"errorMessage": "synthetic invalid request"}
    raw = json.dumps(value).encode()
    calls = []
    class Response:
        status_code = status
        history = ()
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def iter_content(self, chunk_size): yield raw
    class Session:
        def get_adapter(self, url): return SimpleNamespace(max_retries=SimpleNamespace(total=0))
        def request(self, method, url, **kwargs):
            assert method == "GET" and url.endswith(diagnostic.ENDPOINT)
            assert kwargs["params"] == diagnostic.REQUEST
            assert kwargs["allow_redirects"] is False and kwargs["verify"] is True
            calls.append(kwargs)
            return Response()
    client = SimpleNamespace(env="practice", base_url=diagnostic.probe.BoundCandleClient.base_url,
                             session=Session())
    journal = diagnostic.probe.Journal(tmp_path)
    result = diagnostic.observe_request(plan(), journal, client_factory=lambda: client)
    assert result["get_attempts"] == len(calls) == 1 and result["http_status_codes"] == [status]
    assert result["clock_readiness_measured"] is False and result["relaunch_authorized"] is False
    if status == 400:
        assert result["exit_code"] == 1
        assert (tmp_path / "http-error-0001.body").read_bytes() == raw
        assert not (tmp_path / "PARSED_RESPONSE.json").exists()
    else:
        assert result["exit_code"] == 0
        assert json.loads((tmp_path / "PARSED_RESPONSE.json").read_text()) == value
    assert all("synthetic invalid request" not in p.read_text() for p in tmp_path.glob("event-*.json"))
