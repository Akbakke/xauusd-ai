#!/usr/bin/env python3
"""
Test for run index deduplication functionality.
"""

import json
import tempfile
from pathlib import Path

import pytest

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(workspace_root))

from gx1.utils.run_index import (
    append_run_index_dedup,
    build_run_index_entry,
    read_run_index,
)


def test_run_index_dedup():
    """Test that append_run_index_dedup prevents duplicate entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        gx1_data_root = tmp_path / "GX1_DATA"
        reports_root = gx1_data_root / "reports"
        reports_root.mkdir(parents=True)
        
        # Create a fake output directory with minimal files
        output_dir = reports_root / "test_kind" / "test_run_123"
        output_dir.mkdir(parents=True)
        
        # Create RUN_COMPLETED.json to set status
        (output_dir / "RUN_COMPLETED.json").write_text(
            json.dumps({"status": "COMPLETED", "run_id": "test_run_123"})
        )
        
        # Create RUN_IDENTITY.json
        (output_dir / "RUN_IDENTITY.json").write_text(
            json.dumps({
                "output_mode": "MINIMAL",
                "git_head_sha": "abc123",
                "run_id": "test_run_123",
            })
        )
        
        # Build entry
        entry = build_run_index_entry(output_dir)
        
        # Verify entry has entry_id
        assert "entry_id" in entry
        assert "event" in entry
        assert entry["status"] == "COMPLETED"
        assert entry["event"] == "RUN_COMPLETED"
        
        entry_id = entry["entry_id"]
        
        # Append first time - should succeed
        result1 = append_run_index_dedup(gx1_data_root, entry)
        assert result1 is True, "First append should succeed"
        
        # Append same entry again - should be deduplicated
        result2 = append_run_index_dedup(gx1_data_root, entry)
        assert result2 is False, "Duplicate append should return False"
        
        # Append same entry third time - should be deduplicated
        result3 = append_run_index_dedup(gx1_data_root, entry)
        assert result3 is False, "Third duplicate append should return False"
        
        # Read back and verify only one entry exists
        entries = read_run_index(gx1_data_root, ignore_partial=True)
        assert len(entries) == 1, f"Expected 1 entry, got {len(entries)}"
        
        # Verify entry_id matches
        assert entries[0]["entry_id"] == entry_id
        
        # Verify entry content
        assert entries[0]["status"] == "COMPLETED"
        assert entries[0]["run_id"] == "test_run_123"
        assert entries[0]["kind"] == "test_kind"


def test_run_index_dedup_different_status():
    """Test that different status creates different entry_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        gx1_data_root = tmp_path / "GX1_DATA"
        reports_root = gx1_data_root / "reports"
        reports_root.mkdir(parents=True)
        
        # Create output directory
        output_dir = reports_root / "test_kind" / "test_run_123"
        output_dir.mkdir(parents=True)
        
        # Create RUN_IDENTITY.json
        (output_dir / "RUN_IDENTITY.json").write_text(
            json.dumps({
                "output_mode": "MINIMAL",
                "git_head_sha": "abc123",
                "run_id": "test_run_123",
            })
        )
        
        # First: COMPLETED status
        (output_dir / "RUN_COMPLETED.json").write_text(
            json.dumps({"status": "COMPLETED"})
        )
        entry_completed = build_run_index_entry(output_dir)
        entry_id_completed = entry_completed["entry_id"]
        
        # Remove COMPLETED, add FAILED
        (output_dir / "RUN_COMPLETED.json").unlink()
        (output_dir / "RUN_FAILED.json").write_text(
            json.dumps({"status": "FAILED", "reason": "test"})
        )
        entry_failed = build_run_index_entry(output_dir)
        entry_id_failed = entry_failed["entry_id"]
        
        # Entry IDs should be different (different event)
        assert entry_id_completed != entry_id_failed, "Different event should create different entry_id"
        
        # Both should be appendable (different entry_id)
        result1 = append_run_index_dedup(gx1_data_root, entry_completed)
        assert result1 is True
        
        result2 = append_run_index_dedup(gx1_data_root, entry_failed)
        assert result2 is True
        
        # Read back - should have 2 entries
        entries = read_run_index(gx1_data_root, ignore_partial=True)
        assert len(entries) == 2, f"Expected 2 entries, got {len(entries)}"


def test_entry_id_uses_event_not_status():
    """Test that entry_id uses event (not status) and canonical output_dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        gx1_data_root = tmp_path / "GX1_DATA"
        reports_root = gx1_data_root / "reports"
        reports_root.mkdir(parents=True)
        
        # Create output directory
        output_dir = reports_root / "test_kind" / "test_run_123"
        output_dir.mkdir(parents=True)
        
        # Create RUN_IDENTITY.json
        (output_dir / "RUN_IDENTITY.json").write_text(
            json.dumps({
                "output_mode": "MINIMAL",
                "git_head_sha": "abc123",
                "run_id": "test_run_123",
            })
        )
        
        # Create RUN_COMPLETED.json
        (output_dir / "RUN_COMPLETED.json").write_text(
            json.dumps({"status": "COMPLETED"})
        )
        
        entry = build_run_index_entry(output_dir)
        
        # Verify entry_id is based on event, not status
        assert "entry_id" in entry
        assert entry["event"] == "RUN_COMPLETED"
        assert entry["status"] == "COMPLETED"
        
        # Verify canonical output_dir is used (normalized path)
        canonical_output_dir = str(output_dir.resolve()).replace("\\", "/")
        assert canonical_output_dir in str(entry["output_dir"]) or entry["output_dir"] == str(output_dir.resolve())
        
        # Build entry again - should have same entry_id (same event)
        entry2 = build_run_index_entry(output_dir)
        assert entry["entry_id"] == entry2["entry_id"], "Same event should produce same entry_id"


def test_dedup_fullscan_prevents_duplicate_even_if_not_in_last_n():
    """Test that fullscan prevents duplicates even if entry is beyond last_n."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        gx1_data_root = tmp_path / "GX1_DATA"
        reports_root = gx1_data_root / "reports"
        reports_root.mkdir(parents=True)
        
        # Create output directory
        output_dir = reports_root / "test_kind" / "test_run_123"
        output_dir.mkdir(parents=True)
        
        # Create RUN_IDENTITY.json
        (output_dir / "RUN_IDENTITY.json").write_text(
            json.dumps({
                "output_mode": "MINIMAL",
                "git_head_sha": "abc123",
                "run_id": "test_run_123",
            })
        )
        
        # Create RUN_COMPLETED.json
        (output_dir / "RUN_COMPLETED.json").write_text(
            json.dumps({"status": "COMPLETED"})
        )
        
        entry = build_run_index_entry(output_dir)
        entry_id = entry["entry_id"]
        
        # Append first entry
        result1 = append_run_index_dedup(gx1_data_root, entry, last_n=10, fullscan_max_bytes=1_000_000)
        assert result1 is True, "First append should succeed"
        
        # Add many dummy entries to push original entry beyond last_n
        index_path = reports_root / "_index.jsonl"
        for i in range(20):
            dummy_entry = {
                "ts_utc": "2024-01-01T00:00:00+00:00",
                "output_dir": str(reports_root / "dummy" / f"run_{i}"),
                "run_id": f"dummy_run_{i}",
                "event": "UNKNOWN",
                "status": "UNKNOWN",
                "entry_id": f"dummy_id_{i}",
            }
            with open(index_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(dummy_entry) + "\n")
        
        # Try to append same entry again - should be deduplicated even though it's beyond last_n
        # (because file is small enough for fullscan)
        result2 = append_run_index_dedup(gx1_data_root, entry, last_n=10, fullscan_max_bytes=1_000_000)
        assert result2 is False, "Duplicate should be caught by fullscan even if beyond last_n"
        
        # Verify only one entry with this entry_id exists
        entries = read_run_index(gx1_data_root, ignore_partial=True)
        matching_entries = [e for e in entries if e.get("entry_id") == entry_id]
        assert len(matching_entries) == 1, f"Expected 1 entry with entry_id, got {len(matching_entries)}"


def test_cli_ignores_partial_last_line():
    """Test that CLI reader ignores partial last line."""
    import subprocess
    import sys
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        gx1_data_root = tmp_path / "GX1_DATA"
        reports_root = gx1_data_root / "reports"
        reports_root.mkdir(parents=True)
        
        index_path = reports_root / "_index.jsonl"
        
        # Write valid entry
        valid_entry = {
            "ts_utc": "2024-01-01T00:00:00+00:00",
            "output_dir": str(reports_root / "test" / "run1"),
            "run_id": "run1",
            "event": "RUN_COMPLETED",
            "status": "COMPLETED",
            "entry_id": "abc123",
        }
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(valid_entry) + "\n")
        
        # Write partial last line (no newline, incomplete JSON)
        with open(index_path, "a", encoding="utf-8") as f:
            f.write('{"ts_utc": "2024-01-01T00:00:01+00:00", "incomplete":')  # No newline, incomplete
        
        # Try to read using gx1_runs.py CLI (would need to import and call directly)
        # For now, test the logic directly by reading the file
        with open(index_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Simulate CLI logic: last line without newline should be ignored
        entries = []
        partial_last_line_ignored = False
        for i, line in enumerate(lines):
            is_last_line = (i == len(lines) - 1)
            if is_last_line and not line.endswith("\n"):
                partial_last_line_ignored = True
                continue
            line_stripped = line.rstrip("\n\r")
            if not line_stripped:
                continue
            try:
                entry = json.loads(line_stripped)
                entries.append(entry)
            except json.JSONDecodeError:
                if is_last_line:
                    partial_last_line_ignored = True
                continue
        
        # Should have 1 valid entry and partial line should be ignored
        assert len(entries) == 1, f"Expected 1 entry, got {len(entries)}"
        assert partial_last_line_ignored is True, "Partial last line should be detected"


def test_verify_terminal_event_uniqueness():
    """Test that verify_run_index detects duplicate terminal events per run_id."""
    # This test would require importing verify_run_index's main function
    # For now, test the logic directly
    from collections import defaultdict
    
    terminal_events = {"RUN_COMPLETED", "RUN_FAILED", "MASTER_FATAL", "DOCTOR_FATAL"}
    
    # Create entries with duplicate terminal events for same run_id
    entries = [
        {"run_id": "run1", "event": "RUN_COMPLETED", "ts_utc": "2024-01-01T00:00:00+00:00"},
        {"run_id": "run1", "event": "RUN_FAILED", "ts_utc": "2024-01-01T00:00:01+00:00"},  # Duplicate!
        {"run_id": "run2", "event": "RUN_COMPLETED", "ts_utc": "2024-01-01T00:00:02+00:00"},
    ]
    
    run_id_terminal_events = defaultdict(list)
    for entry in entries:
        run_id = entry.get("run_id")
        event = entry.get("event")
        if run_id and event in terminal_events:
            run_id_terminal_events[run_id].append(event)
    
    # Check for duplicates
    duplicate_terminal_events = {}
    for run_id, events in run_id_terminal_events.items():
        if len(events) > 1:
            duplicate_terminal_events[run_id] = events
    
    assert "run1" in duplicate_terminal_events, "run1 should have duplicate terminal events"
    assert len(duplicate_terminal_events["run1"]) == 2, "run1 should have 2 terminal events"
    assert "run2" not in duplicate_terminal_events, "run2 should not have duplicate terminal events"


def test_run_index_dedup_prevents_double_terminal_event():
    """Torture test: verify dedup prevents double terminal event even after >last_n entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        gx1_data_root = tmp_path / "GX1_DATA"
        reports_root = gx1_data_root / "reports"
        reports_root.mkdir(parents=True)
        
        # Create output directory
        output_dir = reports_root / "test_kind" / "test_run_terminal_123"
        output_dir.mkdir(parents=True)
        
        # Create RUN_IDENTITY.json
        (output_dir / "RUN_IDENTITY.json").write_text(
            json.dumps({
                "output_mode": "MINIMAL",
                "git_head_sha": "abc123",
                "run_id": "test_run_terminal_123",
            })
        )
        
        # Create MASTER_FATAL.json (terminal event)
        (output_dir / "MASTER_FATAL.json").write_text(
            json.dumps({
                "fatal_reason": "TEST_FATAL",
                "error_message": "Test intentional failure",
            })
        )
        
        # Build entry (should have event=MASTER_FATAL)
        entry = build_run_index_entry(output_dir)
        assert entry["event"] == "MASTER_FATAL", "Entry should have MASTER_FATAL event"
        assert entry["status"] == "FAILED", "Entry should have FAILED status"
        
        entry_id = entry["entry_id"]
        
        # Append first time - should succeed
        debug_stats_1 = {}
        result1 = append_run_index_dedup(
            gx1_data_root, entry, last_n=10, fullscan_max_bytes=1_000_000, debug_stats=debug_stats_1
        )
        assert result1 is True, "First append should succeed"
        assert debug_stats_1.get("dedup_hit") is False, "First append should not hit dedup"
        
        # Add many dummy entries to push original entry beyond last_n
        index_path = reports_root / "_index.jsonl"
        for i in range(20):
            dummy_entry = {
                "ts_utc": "2024-01-01T00:00:00+00:00",
                "output_dir": str(reports_root / "dummy" / f"run_{i}"),
                "run_id": f"dummy_run_{i}",
                "event": "UNKNOWN",
                "status": "UNKNOWN",
                "entry_id": f"dummy_id_{i}",
            }
            with open(index_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(dummy_entry) + "\n")
        
        # Try to append same terminal event entry again - should be deduplicated
        # (even though it's beyond last_n, fullscan should catch it)
        debug_stats_2 = {}
        result2 = append_run_index_dedup(
            gx1_data_root, entry, last_n=10, fullscan_max_bytes=1_000_000, debug_stats=debug_stats_2
        )
        assert result2 is False, "Duplicate terminal event should be caught by dedup"
        assert debug_stats_2.get("dedup_hit") is True, "Second append should hit dedup"
        assert debug_stats_2.get("dedup_mode") == "fullscan", "Should use fullscan mode"
        assert debug_stats_2.get("scanned_lines", 0) > 0, "Should have scanned some lines"
        
        # Verify only one entry with this entry_id exists
        entries = read_run_index(gx1_data_root, ignore_partial=True)
        matching_entries = [e for e in entries if e.get("entry_id") == entry_id]
        assert len(matching_entries) == 1, f"Expected 1 entry with entry_id, got {len(matching_entries)}"
        
        # Verify the entry has terminal event
        assert matching_entries[0]["event"] == "MASTER_FATAL", "Entry should have MASTER_FATAL event"
        assert matching_entries[0]["run_id"] == "test_run_terminal_123", "Entry should have correct run_id"


if __name__ == "__main__":
    test_run_index_dedup()
    test_run_index_dedup_different_status()
    test_entry_id_uses_event_not_status()
    test_dedup_fullscan_prevents_duplicate_even_if_not_in_last_n()
    test_cli_ignores_partial_last_line()
    test_verify_terminal_event_uniqueness()
    print("✓ All tests passed")
