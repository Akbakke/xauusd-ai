#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test that empty attribution fallback is bit-for-bit deterministic.

TRUTH requires attribution file to exist; when no decisions, we write
a deterministic empty attribution. This test ensures:
- Two writes produce identical file contents (hash)
- Structure matches contract: contract_id, total_decisions, by_decision, by_reason
"""
import hashlib
import tempfile
from pathlib import Path

import pytest


def test_attribution_empty_fallback_deterministic():
    """Empty attribution file is bit-for-bit stable between writes."""
    from gx1.utils.atomic_json import atomic_write_json_deterministic

    payload = {
        "contract_id": "ATTRIBUTION_V1",
        "total_decisions": 0,
        "by_decision": {},
        "by_reason": {},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = Path(tmpdir) / "attribution_1.json"
        p2 = Path(tmpdir) / "attribution_2.json"

        assert atomic_write_json_deterministic(p1, payload), "Write 1 failed"
        assert atomic_write_json_deterministic(p2, payload), "Write 2 failed"

        c1 = p1.read_bytes()
        c2 = p2.read_bytes()

        h1 = hashlib.sha256(c1).hexdigest()
        h2 = hashlib.sha256(c2).hexdigest()

        assert h1 == h2, f"Hash mismatch: {h1} vs {h2}"


def test_attribution_empty_fallback_structure():
    """Empty attribution has required fields for merge."""
    from gx1.utils.atomic_json import atomic_write_json_deterministic
    import json

    payload = {
        "contract_id": "ATTRIBUTION_V1",
        "total_decisions": 0,
        "by_decision": {},
        "by_reason": {},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "attribution.json"
        assert atomic_write_json_deterministic(p, payload)

        data = json.loads(p.read_text())
        assert data["contract_id"] == "ATTRIBUTION_V1"
        assert data["total_decisions"] == 0
        assert data["by_decision"] == {}
        assert data["by_reason"] == {}
