#!/usr/bin/env python3
"""Drift tripwire for the Claude Code guardrails themselves.

The guard hooks + settings live in ~/.claude/ — OUTSIDE any git repo — so they could be
changed or deleted with no history (the exact "working code can vanish/drift silently" risk,
one level above the trading code). This test pins them: it asserts the LIVE hooks + settings
are byte-identical to the versioned copies committed under the repo's .claude/. If they drift,
this fails loudly with a re-sync command. Intentional guard changes = update BOTH (live + ref)
so the change is recorded in git.

Skips gracefully where ~/.claude is absent (CI / other machines).
"""
from __future__ import annotations

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
    if not live.exists():
        pytest.skip(f"live guard artifact absent (other machine/CI): {live}")
    assert ref.exists(), (
        f"versioned reference missing: {ref}. The guardrail is live but not tracked — "
        f"commit it:  cp {live} {ref}"
    )
    live_b = live.read_bytes()
    ref_b = ref.read_bytes()
    assert live_b == ref_b, (
        f"GUARD DRIFT: live {live} != versioned {ref}.\n"
        f"  If the change is INTENTIONAL, update the ref:  cp {live} {ref}  (then commit).\n"
        f"  If UNINTENTIONAL, restore the live guard:       cp {ref} {live}\n"
        f"  (live={len(live_b)}B vs ref={len(ref_b)}B)"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
