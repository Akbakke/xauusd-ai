"""Synthetic source mechanics only; no GC market quality or economic evidence."""
import csv
import gzip
import json
from pathlib import Path

import pytest

from gx1.contracts.gc_order_flow_source_v1 import (
    F_BAD_TS_RECV, F_MAYBE_BAD_BOOK, GC_PRETEST_END_NS, GC_RESEARCH_ID,
    GC_SOURCE_SCHEMA, UNDEF_PRICE, audit_gc_records, require_gc_file_declaration,
)
from gx1.scripts import research_ta_campaign_v1 as ta


def declaration(**changes):
    item = {
        "provider": "databento", "dataset": "GLBX.MDP3", "schema": "mbp-1",
        "raw_symbol": "GCZ5", "instrument_id": 42, "publisher_id": 1,
        "start_ns": 1_735_689_600_000_000_000, "end_ns": 1_735_689_660_000_000_000,
        "path": "/synthetic/gc.csv", "receipt_path": "/synthetic/receipt.json",
        "sha256": "0" * 64, "receipt_sha256": "0" * 64, "bytes": 1,
        "max_records": 100, "pretty_ts": False, "pretty_px": False,
        "compression": "none",
    }
    return {**item, **changes}


def event(**changes):
    return {
        "ts_recv": 1_735_689_600_000_000_002, "ts_event": 1_735_689_600_000_000_001,
        "rtype": 1, "publisher_id": 1, "instrument_id": 42,
        "action": "T", "side": "B", "price": 2_600_100_000_000, "size": 5,
        "flags": 128, "sequence": 10, "bid_px_00": 2_600_000_000_000,
        "ask_px_00": 2_600_100_000_000, "bid_sz_00": 10, "ask_sz_00": 12,
        **changes,
    }


def test_delta_uses_trade_aggressor_and_accounts_for_unknown_volume():
    rows = [event(), event(side="A", size=3), event(side="N", size=2),
            event(action="M", side="B", size=100)]
    report = audit_gc_records(rows, declaration())
    assert report["known_aggressor_delta_contracts"] == 2
    assert report["buy_aggressor_volume_contracts"] == 5
    assert report["sell_aggressor_volume_contracts"] == 3
    assert report["unknown_aggressor_volume_contracts"] == 2
    assert report["known_aggressor_volume_share"] == 0.8
    assert report["admitted_to_model_or_economic_test"] is False
    assert report["continuous_coverage_proven"] is False


def test_tbbo_never_claims_event_book_ofi_capability():
    report = audit_gc_records([event()], declaration(schema="tbbo"))
    assert report["contains_event_book_schema"] is False
    assert report["ofi_computed"] is False
    report = audit_gc_records([event()], declaration())
    assert report["contains_event_book_schema"] is True
    assert report["ofi_computed"] is False


def test_vendor_bad_flags_unknown_books_and_sequence_jumps_are_reported():
    rows = [event(flags=128 | F_BAD_TS_RECV | F_MAYBE_BAD_BOOK),
            event(sequence=30, bid_px_00=UNDEF_PRICE, bid_sz_00=0)]
    report = audit_gc_records(rows, declaration())
    assert report["counters"]["bad_receive_timestamp_records"] == 1
    assert report["counters"]["possible_bad_book_records"] == 1
    assert report["counters"]["unavailable_two_sided_book_records"] == 1
    assert report["counters"]["sequence_jumps"] == 1
    assert report["sequence_jumps_are_gap_proof"] is False


@pytest.mark.parametrize("changes", [
    {"side": "unknown"}, {"action": "F"}, {"size": 0}, {"size": 1.0},
    {"price": UNDEF_PRICE}, {"instrument_id": 43}, {"publisher_id": 2},
    {"ts_recv": 1_735_689_660_000_000_000}, {"flags": 256}, {"rtype": 0},
    {"sequence": 1 << 32}, {"size": 1 << 32}, {"bid_sz_00": 1 << 32},
    {"bid_px_00": 1 << 63},
])
def test_invalid_trade_or_source_identity_is_rejected(changes):
    with pytest.raises(ValueError):
        audit_gc_records([event(**changes)], declaration())


@pytest.mark.parametrize("changes", [
    {"raw_symbol": "GC.c.0"}, {"raw_symbol": "GCZ5-GCG6"}, {"dataset": "OTHER"},
    {"end_ns": GC_PRETEST_END_NS + 1}, {"max_records": 0}, {"pretty_ts": True},
    {"instrument_id": 1 << 32}, {"publisher_id": 1 << 16},
])
def test_declaration_rejects_continuous_symbols_and_sealed_test(changes):
    with pytest.raises(ValueError):
        require_gc_file_declaration(declaration(**changes))


def test_receive_order_and_record_budget_fail_without_partial_success():
    with pytest.raises(ValueError, match="RECEIVE_TIME_REVERSAL"):
        audit_gc_records([event(), event(ts_recv=1_735_689_600_000_000_001)], declaration())
    with pytest.raises(ValueError, match="BUDGET_EXCEEDED"):
        audit_gc_records([event(), event()], declaration(max_records=1))
    with pytest.raises(ValueError, match="NO_RECORDS"):
        audit_gc_records([], declaration())


def test_empty_source_plan_reports_blocked_without_output(tmp_path):
    spec_path = tmp_path / "PLAN.json"
    spec_path.write_text("{}")
    spec = {"research_id": GC_RESEARCH_ID, "source_audit_schema": GC_SOURCE_SCHEMA, "files": []}
    report = ta.audit_gc_source(spec, spec_path)
    assert report["status"] == "BLOCKED_NO_BOUND_GC_FILES"
    assert report["record_audit_performed"] is False
    assert set(tmp_path.iterdir()) == {spec_path}


def bound_spec(tmp_path, monkeypatch, compression):
    root = tmp_path / "source"
    root.mkdir()
    output_root = tmp_path / "runs"
    monkeypatch.setattr(ta, "GC_LOCAL_SOURCE_ROOT", root)
    monkeypatch.setattr(ta, "GC_AUDIT_OUTPUT_ROOT", output_root)
    path = root / ("gc.csv.gz" if compression == "gzip" else "gc.csv")
    opener = gzip.open if compression == "gzip" else Path.open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(event()))
        writer.writeheader()
        writer.writerow(event())
    receipt = root / "RECEIPT.json"
    receipt.write_text(json.dumps({"evidence_class": "synthetic"}))
    item = declaration(path=str(path), receipt_path=str(receipt),
                       sha256=ta.sha(path), receipt_sha256=ta.sha(receipt),
                       bytes=path.stat().st_size, compression=compression)
    spec = {"research_id": GC_RESEARCH_ID, "source_audit_schema": GC_SOURCE_SCHEMA,
            "files": [item], "source_root": str(root), "output_directory": str(output_root / "AUDIT_001")}
    spec_path = tmp_path / "PLAN.json"
    spec_path.write_text(json.dumps(spec))
    return spec, spec_path


@pytest.mark.parametrize("compression", ["none", "gzip"])
def test_local_audit_hashes_sources_and_publishes_without_admission(tmp_path, monkeypatch, compression):
    spec, path = bound_spec(tmp_path, monkeypatch, compression)
    report = ta.audit_gc_source(spec, path)
    assert report["record_audit_performed"] is True
    assert report["files"][0]["known_aggressor_delta_contracts"] == 5
    assert report["files"][0]["source_receipt_semantics_verified"] is False
    terminal = json.loads((Path(spec["output_directory"]) / "TERMINAL.json").read_text())
    assert terminal["result_sha256"] == ta.sha(Path(spec["output_directory"]) / "RESULT.json")
    with pytest.raises(FileExistsError):
        ta.audit_gc_source(spec, path)


def test_source_hash_mismatch_stops_before_publication(tmp_path, monkeypatch):
    spec, path = bound_spec(tmp_path, monkeypatch, "none")
    spec["files"][0]["sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="HASH_MISMATCH"):
        ta.audit_gc_source(spec, path)
    assert not Path(spec["output_directory"]).exists()


def test_duplicate_files_and_invalid_later_path_fail_before_any_source_hash(tmp_path, monkeypatch):
    spec, path = bound_spec(tmp_path, monkeypatch, "none")
    spec["files"].append(dict(spec["files"][0]))
    def forbidden_hash(_):
        raise AssertionError("source bytes must not be opened")
    monkeypatch.setattr(ta, "sha", forbidden_hash)
    with pytest.raises(RuntimeError, match="DUPLICATE_SOURCE"):
        ta.audit_gc_source(spec, path)
    spec["files"][1]["path"] = str(tmp_path / "outside.csv")
    with pytest.raises(ValueError):
        ta.audit_gc_source(spec, path)
    assert not Path(spec["output_directory"]).exists()


def test_external_or_symlink_path_is_rejected_before_content_read(tmp_path):
    allowed = tmp_path / "research"
    allowed.mkdir()
    outside = tmp_path / "sealed"
    outside.mkdir()
    secret = outside / "not_to_be_opened.csv"
    secret.write_text("unread")
    with pytest.raises(ValueError):
        ta._gc_bound_local_path(str(secret), allowed)
    link = allowed / "linked"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="CANONICAL_FILE"):
        ta._gc_bound_local_path(str(link / secret.name), allowed)


def test_shared_json_writer_never_overwrites_existing_evidence(tmp_path):
    path = tmp_path / "RESULT.json"
    ta.write_json(path, {"first": True})
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="OUTPUT_EXISTS"):
        ta.write_json(path, {"second": True})
    assert path.read_bytes() == before
