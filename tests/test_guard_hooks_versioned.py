#!/usr/bin/env python3
"""Drift tripwire for the Claude Code guardrails themselves.

The guard hooks + settings live in ~/.claude/ — OUTSIDE any git repo — so they could be
changed or deleted with no history (the exact "working code can vanish/drift silently" risk,
one level above the trading code). This test pins them: it asserts the LIVE hooks + settings
are byte-identical to the versioned copies committed under the repo's .claude/ when the local
installation exists. The committed reference is always syntax-validated, including on CI.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
LIVE_HOME = Path("/home/andre2/.claude")

# (live path, versioned reference path)
PAIRS = [
    (LIVE_HOME / "hooks" / "guard_bash.py", REPO / ".claude" / "hooks" / "guard_bash.py"),
    (LIVE_HOME / "hooks" / "guard_write.py", REPO / ".claude" / "hooks" / "guard_write.py"),
    (LIVE_HOME / "hooks" / "check_python.py", REPO / ".claude" / "hooks" / "check_python.py"),
    (LIVE_HOME / "settings.json", REPO / ".claude" / "settings.reference.json"),
]


@pytest.mark.parametrize("live,ref", PAIRS, ids=[p[1].name for p in PAIRS])
def test_guard_artifact_matches_versioned_reference(live: Path, ref: Path):
    assert ref.exists(), (
        f"versioned reference missing: {ref}. The guardrail is live but not tracked — "
        f"commit it:  cp {live} {ref}"
    )
    ref_b = ref.read_bytes()
    if ref.suffix == ".py":
        compile(ref_b, str(ref), "exec")
    elif ref.suffix == ".json":
        json.loads(ref_b)
    else:  # pragma: no cover - PAIRS is a fixed local contract
        raise AssertionError(f"unsupported guard reference type: {ref}")
    if not live.exists():
        return
    live_b = live.read_bytes()
    assert live_b == ref_b, (
        f"GUARD DRIFT: live {live} != versioned {ref}.\n"
        f"  If the change is INTENTIONAL, update the ref:  cp {live} {ref}  (then commit).\n"
        f"  If UNINTENTIONAL, restore the live guard:       cp {ref} {live}\n"
        f"  (live={len(live_b)}B vs ref={len(ref_b)}B)"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
