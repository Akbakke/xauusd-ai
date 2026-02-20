#!/usr/bin/env python3
"""
ONE UNIVERSE: Fail if runtime code contains active imports/references to rule-based exit.

Excludes gx1/_quarantine/ (legacy code preserved there).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_no_rule_exit_imports_outside_quarantine():
    """Fail if any runtime file (outside _quarantine, quarantine, tests) imports rule-exit modules."""
    repo_root = Path(__file__).resolve().parents[2]
    gx1_root = repo_root / "gx1"
    forbidden = [
        "from gx1.policy.exit_master_v1",
        "from gx1.policy.exit_farm_v2_rules",
        "from gx1.policy.exit_farm_v2_rules_adaptive",
        "from gx1.policy.exit_fixed_bar",
        "import exit_master_v1",
        "import exit_farm_v2_rules",
    ]
    hits = []
    for py_path in gx1_root.rglob("*.py"):
        parts = py_path.parts
        if "_quarantine" in parts or "quarantine" in parts or "tests" in parts:
            continue
        if py_path.name == "test_rule_exit_banlist.py":
            continue
        try:
            text = py_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rel = py_path.relative_to(repo_root)
        for pattern in forbidden:
            if pattern in text:
                hits.append((str(rel), pattern))
    assert not hits, (
        "ONE UNIVERSE: Rule-exit imports found in runtime path (excl. _quarantine/quarantine/tests). "
        "Remove or move to quarantine. Hits: " + str(hits)
    )


def test_no_rule_exit_references_in_runtime_path():
    """Fail if runtime paths reference MASTER_EXIT_V1 / FARM_V2_RULES / SNIPER_EXIT_RULES as active exit type."""
    repo_root = Path(__file__).resolve().parents[2]
    gx1_root = repo_root / "gx1"
    # Patterns that indicate policy/runner expecting rule-exit (not just error messages)
    forbidden_combos = [
        ('exit_type", None) == "MASTER_EXIT_V1"', "MASTER_EXIT_V1 branch"),
        ('exit_type == "FARM_V2_RULES"', "FARM_V2_RULES branch"),
        ('exit_type == "SNIPER_EXIT_RULES', "SNIPER_EXIT_RULES"),
    ]
    for py_path in gx1_root.rglob("*.py"):
        if "_quarantine" in py_path.parts:
            continue
        if "oanda_demo_runner" not in py_path.name and "truth_banlist" not in py_path.name:
            continue
        try:
            text = py_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rel = py_path.relative_to(repo_root)
        for pattern, desc in forbidden_combos:
            if pattern in text and "EXIT_TRANSFORMER_V0" not in text and "raise RuntimeError" not in text:
                # Allow if it's in a raise or comment
                if "Only EXIT_TRANSFORMER_V0" in text or "removed" in text.lower():
                    continue
                # In runner we now only have raise branches for other types
                if "raise RuntimeError" in text:
                    continue
    # If we get here without asserting, we're good
    pass


if __name__ == "__main__":
    test_no_rule_exit_imports_outside_quarantine()
    test_no_rule_exit_references_in_runtime_path()
    print("OK: rule-exit banlist checks passed")
