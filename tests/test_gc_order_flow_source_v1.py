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
        "action": "T", "side": "B", "symbol": "GCZ5", "price": 2_600_100_000_000, "size": 5,
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


def test_passive_fill_is_counted_but_never_added_to_aggressor_volume():
    report = audit_gc_records([event(action="F", side="B", size=7)], declaration())
    assert report["actions"]["F"] == 1
    assert report["counters"].get("trade_records", 0) == 0
    assert report["buy_aggressor_volume_contracts"] == 0
    assert report["sell_aggressor_volume_contracts"] == 0
    assert report["known_aggressor_delta_contracts"] == 0
    daily = report["daily_receive_activity_utc"]["2025-01-01"]
    assert daily["action_F"] == 1
    assert daily.get("trade_records", 0) == 0


def test_utc_daily_activity_keeps_receive_and_event_dates_separate_and_checks_symbol():
    rows = [
        event(ts_event=1_735_689_599_999_999_999),
        event(ts_event=1_735_689_600_000_000_001, side="N", size=2,
              flags=128 | F_BAD_TS_RECV | F_MAYBE_BAD_BOOK),
    ]
    report = audit_gc_records(rows, declaration())
    daily = report["daily_receive_activity_utc"]["2025-01-01"]
    assert daily["records"] == 2
    assert daily["trade_records"] == 2
    assert daily["unknown_aggressor_volume_contracts"] == 2
    assert daily["bad_receive_timestamp_records"] == 1
    assert daily["possible_bad_book_records"] == 1
    assert report["daily_event_record_counts_utc"] == {"2024-12-31": 1, "2025-01-01": 1}
    with pytest.raises(ValueError, match="RAW_SYMBOL_MAPPING_MISMATCH"):
        audit_gc_records([event(symbol="GCM5")], declaration())


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
    {"side": "unknown"}, {"action": "?"}, {"size": 0}, {"size": 1.0},
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


def test_registered_goal_covers_four_ordered_steps_without_claiming_completion():
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "configs/research/GC_ORDER_FLOW_RESEARCH_001.json").read_text())
    policy = json.loads((root / "NEXT_RUN_POLICY.json").read_text())
    packages = spec["work_packages"]
    progress = policy["current_work"]["gc_goal_progress"]
    assert [item["step"] for item in packages] == [1, 2, 3, 4]
    assert [item["depends_on"] for item in packages] == [[], [1], [2], [3]]
    assert all(item["completion_requires"] for item in packages)
    assert [item["step"] for item in progress] == [item["step"] for item in packages]
    assert spec["goal_completion"]["requires_all_four_steps_and_hash_bound_genuine_evidence"] is True
    assert packages[0]["structural_audit_alone_completes_step"] is False
    assert packages[1]["synthetic_tests_alone_complete_step"] is False
    assert packages[3]["ninth_specialist_required_regardless_of_evidence"] is False
    research = policy["gc_order_flow_research_20261006"]
    assert research["required_steps"] == [item["step"] for item in packages]
    if research["source_qualified"] is False:
        assert all(item["completed"] is False for item in progress)
    for item in progress:
        if item["completed"]:
            assert item["genuine_evidence"]
    assert spec["scope"]["source_fetch"] is True
    assert spec["scope"]["spending"] is True
    assert spec["scope"]["research_fit_ready"] is False
    assert spec["scope"]["native_training"] is False
    assert spec["scope"]["test_access"] is False
    validation = spec["historical_validation_design"]
    assert "chronological_folds" not in spec["comparisons"]["fit_recipe_not_yet_bound"]
    design = validation["predeclared_26_month_design"]
    assert design["status"] == "PLANNING_ONLY_NOT_EXECUTABLE"
    assert design["data_window_utc"] == {
        "start_inclusive": "2024-05-01T00:00:00Z",
        "end_exclusive": "2026-07-01T00:00:00Z",
        "calendar_months": 26,
    }
    folds = design["folds"]
    assert [fold["fold_id"] for fold in folds] == [f"dev_0{i}" for i in range(1, 6)]
    assert [fold["validation_end_utc_exclusive"] for fold in folds] == [
        "2025-08-01T00:00:00Z",
        "2025-11-01T00:00:00Z",
        "2026-02-01T00:00:00Z",
        "2026-05-01T00:00:00Z",
        "2026-07-01T00:00:00Z",
    ]
    assert all(fold["train_start_utc_inclusive"] == "2024-05-01T00:00:00Z" for fold in folds)
    assert all(
        fold["train_end_utc_exclusive"] == fold["validation_start_utc_inclusive"]
        for fold in folds
    )
    assert all(
        folds[i]["validation_end_utc_exclusive"] == folds[i + 1]["validation_start_utc_inclusive"]
        for i in range(len(folds) - 1)
    )
    assert design["all_folds_are_pre_TEST_development_VAL"] is True
    assert design["test_access_authorized"] is False
    assert (
        design["boundary_and_purge_rules"]
        ["exclude_final_rows_whose_required_outcome_reaches_or_crosses_sealed_TEST"]
        is True
    )
    budget = spec["budget_guard"]
    assert budget["free_credit_ceiling_usd"] == 125.0
    assert budget["metadata_only_quote_usd"] == 119.65511028468302
    assert budget["nominal_reserve_usd"] == 5.344889715316981
    assert budget["hard_cost_cap_usd"] == 120.0
    assert budget["do_not_fetch_if_requoted_estimate_exceeds_usd"] == 120.0
    assert budget["metadata_only_quote_usd"] <= budget["do_not_fetch_if_requoted_estimate_exceeds_usd"]
    assert budget["quote_is_acquisition_ready"] is True
    assert budget["download_authorized"] is True
    assert budget["spending_authorized"] is True
    assert budget["pay_as_you_go_authorized"] is False
    assert budget["payment_card_charge_authorized"] is False
    assert budget["mbp1_exact_raw_symbol_quote_usd"] == 119.64002859592402
    assert budget["supporting_data_exact_raw_symbol_quote_usd"] == 0.015081688759
    assert budget["supporting_data_definition_quote_usd"] == 0.001169070599
    assert budget["supporting_data_ohlcv_1d_quote_usd"] == 0.01391261816
    assert budget["operator_reported_remaining_credit_usd"] == 125.0
    assert budget["account_free_balance_verified"] is False
    assert budget["quote_basis"]["exact_outright_raw_symbol_list_bound"] is True
    assert len(budget["quote_basis"]["resolved_outright_raw_symbols"]) == 15
    assert budget["quote_basis"]["mapped_period_segments"] == 33
    assert budget["quote_basis"]["supporting_data_exactly_requoted"] is True
    assert budget["quote_basis"]["supporting_data_quote_period_segments"] == 23
    planning = research["planning_only_calendar_design"]
    assert planning["status"] == "PREDECLARED_NOT_EXECUTABLE"
    assert planning["combined_exact_raw_symbol_estimate_usd"] == 119.65511028468302
    assert planning["supporting_data_requoted_exactly"] is True
    assert planning["operator_reported_remaining_credit_usd"] == 125.0
    assert planning["do_not_fetch_if_requoted_estimate_exceeds_usd"] == 120.0
    assert planning["source_fetch_authorized"] is True
    assert planning["spending_authorized"] is True
    assert planning["research_fit_authorized"] is False
    assert planning["test_access_authorized"] is False
    assert validation["all_agreed_feature_families_not_all_time_splits_in_training"] is True
    assert set(validation["evidence_levels"]) == {"snapshot_research", "native_v38", "final_TEST"}
    assert validation["backtest_requirements"]
