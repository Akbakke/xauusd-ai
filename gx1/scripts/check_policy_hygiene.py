#!/home/andre2/venvs/gx1/bin/python
"""
Repo policy hygiene gate: ensures only the canonical policy and its KEEP dependencies exist,
and that Python code does not hardcode other policy paths.

- Scan gx1/configs/policies/**/*.yaml → fail if any file not in KEEP.
- Scan Python for "gx1/configs/policies/" → fail if path is not canonical or KEEP or TRUTH_CANONICAL_POLICY_RELATIVE.

Deterministic, fast. Exit 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# SSoT: must match truth_banlist and DELETE_PLAN KEEP
KEEP_YAML_RELATIVE = [
    "gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml",
    "gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/MASTER_EXIT_V1_A.yaml",
    "gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/SNIPER_RISK_GUARD_V1.yaml",
]

# Paths allowed in Python source (must match KEEP or be reference to constant)
ALLOWED_POLICY_PATHS_IN_CODE = set(KEEP_YAML_RELATIVE)
POLICY_PATH_PATTERN = re.compile(r"gx1/configs/policies/[^\s\"'\\)]+")


def _find_policy_yamls(root: Path) -> list[Path]:
    policies_dir = root / "gx1" / "configs" / "policies"
    if not policies_dir.is_dir():
        return []
    out: list[Path] = []
    for p in policies_dir.rglob("*.yaml"):
        if p.is_file():
            try:
                rel = p.relative_to(root).as_posix()
                out.append(Path(rel))
            except ValueError:
                continue
    return sorted(out)


def _check_yaml_inventory(root: Path, verbose: bool) -> list[str]:
    """Return list of forbidden YAML paths (not in KEEP)."""
    found = _find_policy_yamls(root)
    keep_set = set(KEEP_YAML_RELATIVE)
    forbidden: list[str] = []
    for rel in found:
        s = rel.as_posix()
        if s not in keep_set:
            forbidden.append(s)
    if verbose and found:
        for s in sorted(set(f.as_posix() for f in found)):
            status = "KEEP" if s in keep_set else "FORBIDDEN"
            print(f"  {s} [{status}]")
    return sorted(forbidden)


def _check_python_hardcoded(root: Path, verbose: bool) -> list[tuple[str, int, str]]:
    """Return list of (file, line_no, line) with forbidden policy paths."""
    violations: list[tuple[str, int, str]] = []
    gx1_root = root / "gx1"
    if not gx1_root.is_dir():
        return violations
    for py_path in gx1_root.rglob("*.py"):
        try:
            rel = py_path.relative_to(root).as_posix()
        except ValueError:
            continue
        if rel == "gx1/scripts/check_policy_hygiene.py":
            continue
        text = py_path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if "gx1/configs/policies/" not in line:
                continue
            if "TRUTH_CANONICAL_POLICY_RELATIVE" in line:
                continue
            matches = POLICY_PATH_PATTERN.findall(line)
            for m in matches:
                if m not in ALLOWED_POLICY_PATHS_IN_CODE:
                    violations.append((rel, i, line.strip()))
                    break
    return violations


def main() -> int:
    ap = argparse.ArgumentParser(description="Policy hygiene: only canonical policy + KEEP deps; no hardcoded other paths.")
    ap.add_argument("--verbose", "-v", action="store_true", help="More output")
    ap.add_argument("--root", type=Path, default=None, help="Repo root (default: parent of gx1/scripts)")
    args = ap.parse_args()
    root = args.root or Path(__file__).resolve().parents[2]
    root = root.resolve()

    failed = False

    # 1) YAML inventory
    forbidden_yamls = _check_yaml_inventory(root, args.verbose)
    if forbidden_yamls:
        failed = True
        print("FORBIDDEN policy YAMLs (not in KEEP):")
        for p in forbidden_yamls:
            print(f"  {p}")
    elif args.verbose:
        print("YAML inventory: all under KEEP")

    # 2) Python hardcoded paths
    violations = _check_python_hardcoded(root, args.verbose)
    if violations:
        failed = True
        print("FORBIDDEN hardcoded policy paths in Python (use canonical or TRUTH_CANONICAL_POLICY_RELATIVE):")
        for path, line_no, line in violations:
            print(f"  {path}:{line_no}: {line[:80]}{'...' if len(line) > 80 else ''}")
    elif args.verbose:
        print("Python: no forbidden policy paths")

    if failed:
        print("\nFAIL: run fixes then re-run this script.")
        return 1
    print("PASS: policy hygiene OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
