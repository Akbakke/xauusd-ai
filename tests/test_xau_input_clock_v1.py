"""Synthetic timing arithmetic only; no market, model, collector or clock proof."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

from gx1.contracts import xau_input_clock_v1 as clock

ISSUE = 1_735_689_600_000_000_000
BOOT = "11111111-1111-4111-8111-111111111111"
M1 = {"path": "/synthetic/m1-manifest.json", "sha256": "1" * 64}
M5 = {"path": "/synthetic/m5-manifest.json", "sha256": "2" * 64}
ATR = "3" * 64


def stamp(index, *, utc=None, bounded=True):
    utc = ISSUE - (7 - index) * 1000 if utc is None else utc
    result = {"utc_ns": utc, "monotonic_before_ns": 100 * index,
              "monotonic_after_ns": 100 * index + 2, "boot_id": BOOT,
              "process_id": 42, "process_start_ticks": 17}
    if bounded:
        result["utc_interval"] = {"lower_ns": utc - 2, "upper_ns": utc + 2,
                                 "evidence_ref": {"path": "/synthetic/clock-evidence.json", "sha256": "4" * 64}}
    return result


def receipt():
    result = {
        "schema_version": clock.RECEIPT_SCHEMA, "issue_ns": ISSUE,
        "m1": {"ref": dict(M1), "bar_close_ns": ISSUE - 60_000_000_000,
               "received": stamp(1), "published": stamp(2)},
        "m5": {"ref": dict(M5), "bar_close_ns": ISSUE - 300_000_000_000,
               "received": stamp(3), "published": stamp(4)},
        "atr": {"source_m5_ref": dict(M5), "contract_sha256": ATR, "value_sha256": "5" * 64,
                "bar_close_ns": {tf: ISSUE - 300_000_000_000 for tf in clock.TIMEFRAMES},
                "ready": stamp(5)},
        "input_ready": {"stamp": stamp(6), "input_sha256": "6" * 64},
    }
    reseal(result)
    return result


def reseal(value):
    value["input_ready"]["binding_sha256"] = clock.input_binding_sha256(value)


def assess(value):
    return clock.assess_input_timing(value, expected_m1_ref=M1, expected_m5_ref=M5,
                                     expected_atr_contract_sha256=ATR)


def test_capture_reads_actual_process_identity_without_inventing_uncertainty():
    observed = clock.capture_clock_sample()
    fields = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
    assert observed["process_id"] == os.getpid()
    assert observed["process_start_ticks"] == int(fields[19])
    assert observed["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    assert 0 <= observed["monotonic_before_ns"] <= observed["monotonic_after_ns"]
    assert set(observed) == {"utc_ns", "monotonic_before_ns", "monotonic_after_ns",
                             "boot_id", "process_id", "process_start_ticks"}


def test_synthetic_complete_chain_is_on_time_and_assessment_is_pure(monkeypatch):
    value = receipt()
    before = copy.deepcopy(value)
    def forbidden(*args, **kwargs):
        raise AssertionError("assessment must not open references")
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr("builtins.open", forbidden)
    result = assess(value)
    assert result["status"] == "ON_TIME"
    assert result["reasons"] == []
    assert set(result["component_status"].values()) == {"ON_TIME"}
    assert value == before


def test_atr_can_be_late_although_all_event_bars_closed():
    value = receipt()
    value["atr"]["ready"] = stamp(5, utc=ISSUE + 10)
    value["input_ready"]["stamp"] = stamp(6, utc=ISSUE + 100)
    result = assess(value)
    assert result["status"] == "LATE"
    assert result["component_status"]["atr.ready"] == "LATE"
    assert max(value["atr"]["bar_close_ns"].values()) < ISSUE


@pytest.mark.parametrize("component", ["m1", "m5", "atr", "input_ready"])
def test_missing_utc_uncertainty_is_unknown(component):
    value = receipt()
    sample = (value[component]["published"] if component in ("m1", "m5")
              else value["atr"]["ready"] if component == "atr" else value["input_ready"]["stamp"])
    del sample["utc_interval"]
    result = assess(value)
    assert result["status"] == "UNKNOWN"
    assert any(reason.startswith("UTC_INTERVAL_UNKNOWN:") for reason in result["reasons"])


@pytest.mark.parametrize("field,new", [
    ("boot_id", "22222222-2222-4222-8222-222222222222"),
    ("process_id", 43), ("process_start_ticks", 18),
])
def test_restart_or_changed_process_epoch_is_unknown(field, new):
    value = receipt()
    value["atr"]["ready"][field] = new
    result = assess(value)
    assert result["status"] == "UNKNOWN"
    assert "CLOCK_EPOCH_CHANGED_OR_RESTART_UNQUALIFIED" in result["reasons"]


@pytest.mark.parametrize("fault,match", [
    ("m5_source", "M5_SOURCE_BINDING"), ("atr_source", "ATR_M5_SOURCE_BINDING"),
    ("atr_contract", "ATR_CONTRACT_BINDING"), ("timeframe", "ATR_TIMEFRAMES_FIELDS"),
    ("future_bar", "ATR_D1_BAR_AFTER_ISSUE"), ("input_hash", "INPUT_BINDING_MISMATCH"),
])
def test_wrong_source_contract_tf_or_unbound_input_is_rejected(fault, match):
    value = receipt()
    if fault == "m5_source":
        value["m5"]["ref"]["sha256"] = "7" * 64
    elif fault == "atr_source":
        value["atr"]["source_m5_ref"]["sha256"] = "7" * 64
    elif fault == "atr_contract":
        value["atr"]["contract_sha256"] = "7" * 64
    elif fault == "timeframe":
        value["atr"]["bar_close_ns"]["M30"] = value["atr"]["bar_close_ns"].pop("M15")
    elif fault == "future_bar":
        value["atr"]["bar_close_ns"]["D1"] = ISSUE + 1
    else:
        value["input_ready"]["input_sha256"] = "7" * 64
    if fault != "input_hash":
        reseal(value)
    with pytest.raises(clock.InputClockError, match=match):
        assess(value)


@pytest.mark.parametrize("fault,match", [
    ("ready_before_receive", "MONOTONIC_ORDER"),
    ("utc_backward", "UTC_BACKWARD"),
    ("reversed_bracket", "MONOTONIC_BRACKET"),
    ("contradictory_interval", "UTC_INTERVAL_CONTRADICTION"),
    ("missing_clock_ref", "UTC_INTERVAL_FIELDS"),
])
def test_contradictory_clock_evidence_is_rejected(fault, match):
    value = receipt()
    if fault == "ready_before_receive":
        value["input_ready"]["stamp"] = stamp(0)
    elif fault == "utc_backward":
        value["atr"]["ready"]["utc_ns"] = value["m5"]["published"]["utc_ns"] - 1
    elif fault == "reversed_bracket":
        value["atr"]["ready"]["monotonic_after_ns"] = 499
    elif fault == "contradictory_interval":
        interval = value["atr"]["ready"]["utc_interval"]
        interval["lower_ns"] = ISSUE - 20_000
        interval["upper_ns"] = ISSUE - 10_000
    else:
        del value["atr"]["ready"]["utc_interval"]["evidence_ref"]
    with pytest.raises(clock.InputClockError, match=match):
        assess(value)


def test_m1_m5_have_no_cross_order_requirement_and_deadline_equality_is_explicit():
    value = receipt()
    value["m1"]["received"] = stamp(0)
    value["m1"]["published"] = stamp(5, utc=ISSUE - 1500)
    value["m1"]["published"]["monotonic_before_ns"] = 550
    value["m1"]["published"]["monotonic_after_ns"] = 552
    value["input_ready"]["stamp"]["utc_interval"]["upper_ns"] = ISSUE
    assert assess(value)["status"] == "ON_TIME"
    value["input_ready"]["stamp"]["utc_interval"]["upper_ns"] = ISSUE + 1
    assert assess(value)["status"] == "UNKNOWN"
