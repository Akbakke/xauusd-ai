#!/usr/bin/env python3
"""Smoke test for import_proof: structure of collect_import_proof() and write/assert helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

from gx1.utils.import_proof import (
    FORBIDDEN_IMPORT_SUBSTRINGS,
    assert_no_forbidden_imports,
    collect_import_proof,
    write_import_proof,
)


def test_collect_import_proof_structure():
    """collect_import_proof() returns dict with required keys; modules_loaded is a list."""
    proof = collect_import_proof()
    assert isinstance(proof, dict)
    assert "python_exe" in proof
    assert "cwd" in proof
    assert "pid" in proof
    assert "sys_path" in proof
    assert "modules_loaded" in proof
    assert "forbidden_hits" in proof
    assert "timestamp_utc" in proof
    assert isinstance(proof["modules_loaded"], list)
    assert isinstance(proof["forbidden_hits"], list)
    assert isinstance(proof["sys_path"], list)


def test_write_import_proof_roundtrip():
    """write_import_proof() writes JSON that can be read back."""
    proof = collect_import_proof()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "IMPORT_PROOF.json"
        write_import_proof(path, proof)
        assert path.exists()
        data = path.read_text(encoding="utf-8")
        assert "modules_loaded" in data
        assert "forbidden_hits" in data


def test_assert_no_forbidden_imports_empty_passes():
    """assert_no_forbidden_imports() does not raise when forbidden_hits is empty."""
    proof = {"forbidden_hits": []}
    assert_no_forbidden_imports(proof, "test_context")


def test_assert_no_forbidden_imports_raises_on_hits():
    """assert_no_forbidden_imports() raises RuntimeError with [TRUTH_IMPORT_GHOST] when forbidden_hits non-empty."""
    proof = {"forbidden_hits": ["gx1.policy.exit_master_v1"]}
    try:
        assert_no_forbidden_imports(proof, "test_context")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "[TRUTH_IMPORT_GHOST]" in str(e)
        assert "exit_master_v1" in str(e)


def test_forbidden_substrings_defined():
    """FORBIDDEN_IMPORT_SUBSTRINGS is non-empty and contains expected entries."""
    assert len(FORBIDDEN_IMPORT_SUBSTRINGS) >= 1
    assert "replay_eval_gated_parallel" in FORBIDDEN_IMPORT_SUBSTRINGS


if __name__ == "__main__":
    test_collect_import_proof_structure()
    test_write_import_proof_roundtrip()
    test_assert_no_forbidden_imports_empty_passes()
    test_forbidden_substrings_defined()
    print("OK: import_proof smoke tests passed")
