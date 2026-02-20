#!/usr/bin/env python3
"""
Guardrail: exits jsonl must exist when exit transformer is active (even with 0 trades).

Tests the helper that creates the placeholder file so POSTRUN gate passes.
Deterministic unit test (no replay or market data).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from gx1.execution.exit_logs import create_exit_jsonl_placeholder


def test_create_exit_jsonl_placeholder_creates_file():
    """create_exit_jsonl_placeholder creates logs/exits/exits_<run_id>.jsonl under run_dir."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        run_id = "TEST_RUN_001"
        path = create_exit_jsonl_placeholder(run_dir, run_id)
        assert path == run_dir / "logs" / "exits" / "exits_TEST_RUN_001.jsonl"
        assert path.exists()
        assert path.read_text(encoding="utf-8") == ""


def test_create_exit_jsonl_placeholder_idempotent():
    """Calling again does not overwrite with empty if file already has content."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        run_id = "TEST_RUN_002"
        path1 = create_exit_jsonl_placeholder(run_dir, run_id)
        path1.write_text('{"event": "one"}\n', encoding="utf-8")
        path2 = create_exit_jsonl_placeholder(run_dir, run_id)
        assert path2 == path1
        assert path2.read_text(encoding="utf-8") == '{"event": "one"}\n'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
