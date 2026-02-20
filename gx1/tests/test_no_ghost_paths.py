#!/usr/bin/env python3
"""
No-ghost paths test: fail if ghost tokens reappear in gx1 runtime/docs.

Scans gx1/ (excluding gx1/tests) for forbidden tokens. Allowed: docs/GHOST_PURGE_PLAN.md
and canonical files that must reference exit/legacy names (truth_banlist, oanda_demo_runner, exit_manager).
See docs/GHOST_PURGE_PLAN.md.
"""

from __future__ import annotations

from pathlib import Path


# Tokens that must not appear in gx1 (except in allowed paths)
GHOST_TOKENS = [
    "replay_eval_gated_parallel",
    "_DEPRECATED",
    "legacy_replay",
    "exit_farm_v2_rules",
    "exit_master_v1",
    "fixed_bar",
]

# Paths allowed to contain ghost tokens (documentation, gate checks, banlist, canonical runner/exit)
ALLOWED_PATH_PATTERNS = [
    "GHOST_PURGE_PLAN.md",
    "truth_banlist.py",
    "oanda_demo_runner.py",
    "exit_manager.py",
    "test_rule_exit_banlist.py",
    "run_truth_e2e_sanity.py",  # gate that checks legacy script not imported/present
    "gx1_doctor.py",            # check that legacy path must not exist
    "quarantine_paths.py",     # forbidden path list
    "prefork_freeze_gate.py",  # canonical bootstrap
    "run_ab_fullyear_2025_exit_ml.py",   # comment: no replay_eval_gated_parallel
    "run_fullyear_2025_truth_proof.py",  # comment
    "run_baseline_eval_maxcpu.sh",       # comment + pkill
    "run_env_gate_policy_check.sh",      # comment
    "docs/",                   # documentation may reference ghost names
    "reports/",                # reports may reference
    "AUDIT_",                   # audit docs
    "ENV_GATING_PROOF",
    "MASTER_EXIT_V1_ML",
    "EXIT_A_TUNING",
    "FARM_V2B",
    "FARM_pipeline",
    "SHORT_SUPPORT",
    "parity_audit_report.json", # generated/report data
]


def _is_allowed(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    s = rel.as_posix()
    return any(pat in s for pat in ALLOWED_PATH_PATTERNS)


def test_no_ghost_tokens_in_gx1():
    """Fail if any ghost token appears in gx1 (excluding tests and allowed paths)."""
    repo_root = Path(__file__).resolve().parents[2]
    gx1_root = repo_root / "gx1"
    if not gx1_root.is_dir():
        return
    hits: list[tuple[str, str, int]] = []
    for path in gx1_root.rglob("*"):
        if path.is_dir():
            continue
        if "tests" in path.parts:
            continue
        if not path.suffix or path.suffix not in (".py", ".sh", ".md", ".json"):
            continue
        if _is_allowed(path, gx1_root):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rel = path.relative_to(repo_root)
        for token in GHOST_TOKENS:
            if token not in text:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if token in line:
                    hits.append((str(rel), token, i))
                    break
    assert not hits, (
        "Ghost tokens found (see docs/GHOST_PURGE_PLAN.md). Remove or move to allowed paths. Hits: "
        + str(hits)
    )


if __name__ == "__main__":
    test_no_ghost_tokens_in_gx1()
    print("OK: no ghost paths")
