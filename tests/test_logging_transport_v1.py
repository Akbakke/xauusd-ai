from __future__ import annotations

import json
from pathlib import Path

from gx1.execution.logging_transport_v1 import (
    EVENT_TRACE_FILE,
    RINGBUFFER_FILE,
    SUMMARY_FILE,
    ReplayObservabilityTransport,
)


def test_first_only_policy_suppresses_repeated_info(tmp_path: Path) -> None:
    transport = ReplayObservabilityTransport(
        output_dir=tmp_path,
        run_id="RUN_A",
        chunk_id="chunk_0",
        flush_interval_sec=0.0,
        ringbuffer_max=20,
    )

    first = transport.record_event(
        "ENTRY_NO_TRADE_CLOSED_WINDOW_PROOF",
        reason="PATTERN_A",
        ts="2026-04-23T00:00:00+00:00",
        key="A::2026-04-23",
        payload={"window": "21:55-23:00"},
    )
    second = transport.record_event(
        "ENTRY_NO_TRADE_CLOSED_WINDOW_PROOF",
        reason="PATTERN_A",
        ts="2026-04-23T00:01:00+00:00",
        key="A::2026-04-23",
        payload={"window": "21:55-23:00"},
    )

    assert first is True
    assert second is False

    summary_path = transport.flush(reason="pytest", force=True)
    assert summary_path == tmp_path / "observability_transport_v1" / SUMMARY_FILE
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    family = summary["families"]["ENTRY_NO_TRADE_CLOSED_WINDOW_PROOF"]
    assert family["total_count"] == 2
    assert family["emitted_count"] == 1
    assert family["suppressed_count"] == 1


def test_abnormal_event_goes_to_trace_and_ringbuffer(tmp_path: Path) -> None:
    transport = ReplayObservabilityTransport(
        output_dir=tmp_path,
        run_id="RUN_B",
        chunk_id="chunk_0",
        flush_interval_sec=0.0,
        ringbuffer_max=5,
    )

    emitted = transport.record_event(
        "MID_EDGE_10_50_PROBE",
        reason="SURPRISING_BRANCH",
        ts="2026-04-23T00:05:00+00:00",
        trade_id="T1",
        key="T1::SURPRISING_BRANCH::NONE",
        payload={"trade_uid": "U1"},
        abnormal=True,
    )
    assert emitted is False
    transport.flush(reason="pytest", force=True)

    trace_path = tmp_path / "observability_transport_v1" / EVENT_TRACE_FILE
    assert trace_path.exists()
    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["family"] == "MID_EDGE_10_50_PROBE"
    assert event["reason"] == "SURPRISING_BRANCH"

    ringbuffer = json.loads((tmp_path / "observability_transport_v1" / RINGBUFFER_FILE).read_text(encoding="utf-8"))
    assert len(ringbuffer["events"]) == 1
